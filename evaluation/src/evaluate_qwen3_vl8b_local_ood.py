"""Evaluate the locally deployed Qwen3-VL-8B model on the frozen OOD CoT set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path('/path/to/trajectory_generation')
DATASET = 'sv30_20260920_r100_r3'
SOURCE_RUN = 'eval_gpt6_luna_ood10_cot_20260925'
RUN_ID = os.environ.get('EVAL_RUN_ID', 'eval_qwen3_vl8b_ood10_cot_20260927')
MODEL_DIR = ROOT / 'model' / 'Qwen3-VL-8B-Instruct'
DEVICE = os.environ.get('EVAL_DEVICE', 'cuda:0')
MAX_PIXELS = int(os.environ.get('EVAL_MAX_PIXELS', str(1920 * 960)))
# The local model can continue verbose free-form generation when given a 20k cap;
# 4096 is ample for the required four-step answer and prevents one runaway sample
# from blocking the full evaluation.
MAX_NEW_TOKENS = int(os.environ.get('EVAL_MAX_NEW_TOKENS', '4096'))
CITIES = ('philadelphia', 'buenos_aires', 'berlin', 'osaka', 'melbourne')
CITY_NAMES = ('Philadelphia', 'Buenos Aires', 'Berlin', 'Osaka', 'Melbourne')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def normalize_title(value: str) -> str:
    return value.replace('\u2011', '-')


def score_output(content: str, truth: list[int], titles: list[str]) -> dict[str, Any]:
    answer_matches = re.findall(r'<answer>\s*(\[[^\]]*\])\s*</answer>', content, re.S | re.I)
    prediction = None
    if len(answer_matches) == 1:
        try:
            candidate = json.loads(answer_matches[0])
            if (isinstance(candidate, list) and len(candidate) == 8
                    and all(type(value) is int for value in candidate)
                    and sorted(candidate) == list(range(8))):
                prediction = candidate
        except (TypeError, ValueError):
            pass
    thoughts = re.findall(r'<think>(.*?)</think>', content, re.S | re.I)
    format_ok = bool(re.fullmatch(r'\s*<think>.*?</think>\s*<answer>.*?</answer>\s*', content, re.S | re.I))
    found_titles = re.findall(r'^Step [1-4]: [^\n]+', thoughts[0], re.M) if len(thoughts) == 1 else []
    titles_match = [normalize_title(title) for title in found_titles] == [normalize_title(title) for title in titles]
    format_ok = format_ok and len(thoughts) == 1 and titles_match
    result: dict[str, Any] = {
        'valid_answer': prediction is not None,
        'cot_format_valid': bool(format_ok),
        'prediction': prediction,
        'exact_match': prediction == truth,
    }
    if prediction is not None:
        ranks = {value: index for index, value in enumerate(truth)}
        inversions = sum(
            ranks[prediction[i]] > ranks[prediction[j]]
            for i in range(8) for j in range(i + 1, 8)
        )
        edges = set(zip(truth, truth[1:]))
        result.update(
            kendall_tau=1 - 2 * inversions / 28,
            pairwise_accuracy=1 - inversions / 28,
            directed_adjacency_accuracy=sum(edge in edges for edge in zip(prediction, prediction[1:])) / 7,
            position_accuracy=sum(left == right for left, right in zip(prediction, truth)) / 8,
        )
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful = [row for row in rows if row.get('status') == 'ok']
    valid = [row for row in successful if row.get('valid_answer')]

    def mean(key: str, items: list[dict[str, Any]]) -> float | None:
        values = [float(row[key]) for row in items if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        'total': total,
        'api_success': len(successful),
        'valid_answers': len(valid),
        'exact_matches': sum(bool(row.get('exact_match')) for row in rows),
        'exact_match_accuracy': sum(bool(row.get('exact_match')) for row in rows) / total if total else None,
        'valid_output_rate': len(valid) / total if total else None,
        'cot_format_valid_rate': sum(bool(row.get('cot_format_valid')) for row in successful) / total if total else None,
        'kendall_tau_valid_answers': mean('kendall_tau', valid),
        'pairwise_accuracy_valid_answers': mean('pairwise_accuracy', valid),
        'directed_adjacency_accuracy_valid_answers': mean('directed_adjacency_accuracy', valid),
        'position_accuracy_valid_answers': mean('position_accuracy', valid),
        'mean_input_tokens': mean('input_tokens', successful),
        'mean_output_tokens': mean('output_tokens', successful),
        'mean_generation_seconds': mean('generation_seconds', successful),
    }


def build_document(manifest: dict[str, Any], rows: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    per_city = {city: aggregate([row for row in rows if row.get('city') == city]) for city in CITIES}
    overall = aggregate(rows)
    macro = [value['exact_match_accuracy'] for value in per_city.values() if value['exact_match_accuracy'] is not None]
    overall['city_macro_exact_match_accuracy'] = sum(macro) / len(macro) if macro else None
    return {
        'run_id': RUN_ID,
        'complete': complete,
        'model': 'Qwen3-VL-8B-Instruct',
        'model_path': str(MODEL_DIR),
        'provider': 'Local Transformers offline inference',
        'updated_at': now(),
        'configuration': {
            'dataset': DATASET,
            'source_run': SOURCE_RUN,
            'seed': manifest['seed'],
            'sampling': manifest['sampling'],
            'device': DEVICE,
            'dtype': 'bfloat16',
            'attention': 'sdpa',
            'max_pixels_per_image': MAX_PIXELS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'prompt': manifest['prompt'],
            'counts': manifest['counts'],
            'image_transport': 'local original files; filenames are not included in model messages',
        },
        'metrics': overall,
        'per_city': per_city,
        'token_policy': 'Local Transformers exposes generated output length but no separate hidden reasoning token counter.',
        'sequences': rows,
    }


def write_summary(path: Path, document: dict[str, Any]) -> None:
    metrics = document['metrics']
    def pct(value: float | None) -> str:
        return f'{value:.2%}' if value is not None else '—'

    lines = [
        '# Qwen3-VL-8B-Instruct local OOD CoT evaluation',
        '',
        f"Status: {'complete' if document['complete'] else 'in progress'}. Offline BF16 inference on one GPU.",
        '',
        'The frozen five-city sample, shuffle, prompt, and ground truth are reused from '
        f'`{SOURCE_RUN}`. Images are read from the server dataset; no filenames or original frame IDs are sent in the model messages.',
        '',
        '| City | Population | Sample | Valid answers | Exact matches | Exact accuracy | Kendall tau |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for city, name in zip(CITIES, CITY_NAMES):
        item = document['per_city'][city]
        tau = f"{item['kendall_tau_valid_answers']:.4f}" if item['kendall_tau_valid_answers'] is not None else '—'
        lines.append(
            f"| {name} | {document['configuration']['counts'][city]['population']} | {item['total']} | "
            f"{item['valid_answers']} | {item['exact_matches']} | {pct(item['exact_match_accuracy'])} | {tau} |"
        )
    lines.extend([
        '',
        f"Overall exact-order accuracy: {pct(metrics['exact_match_accuracy'])}; "
        f"city macro average: {pct(metrics['city_macro_exact_match_accuracy'])}.",
        f"Valid-answer rate: {pct(metrics['valid_output_rate'])}; "
        f"CoT-format compliance: {pct(metrics['cot_format_valid_rate'])}.",
        f"Mean input tokens: {metrics['mean_input_tokens']:.2f}; "
        f"mean generated output tokens: {metrics['mean_output_tokens']:.2f}; "
        f"mean generation time: {metrics['mean_generation_seconds']:.2f} seconds.",
        '',
        'Exact accuracy uses all sampled trajectories as the denominator; ordering metrics use valid permutations only.',
        'Local Transformers inference does not expose a separate hidden reasoning-token count. Generated output tokens include the visible response, including any explicit CoT and answer.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, default=ROOT / 'datasets' / DATASET / 'evaluations' / SOURCE_RUN / 'sample_manifest.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'datasets' / DATASET / 'evaluations' / RUN_ID)
    parser.add_argument('--limit', type=int, default=0, help='Process at most this many pending trajectories; 0 means all.')
    parser.add_argument('--start-index', type=int, default=0, help='Inclusive manifest index for a parallel shard.')
    parser.add_argument('--end-index', type=int, default=0, help='Exclusive manifest index for a parallel shard; 0 means the end.')
    parser.add_argument('--resume', action='store_true', help='Resume rows already saved in results.json.')
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f'Model directory does not exist: {MODEL_DIR}')
    manifest = load_json(args.source_manifest)
    if manifest['dataset'] != DATASET or len(manifest['sequences']) != 352:
        raise SystemExit('The source manifest is not the expected 352-trajectory OOD manifest')
    source_prompt = manifest['prompt']

    output_path = args.output_dir / 'results.json'
    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        previous = load_json(output_path)
        rows_by_id = {row['id']: row for row in previous.get('sequences', []) if row.get('status') == 'ok'}
    start_index = max(0, args.start_index)
    end_index = args.end_index if args.end_index > 0 else len(manifest['sequences'])
    if start_index >= end_index or end_index > len(manifest['sequences']):
        raise SystemExit(f'Invalid shard range: {start_index}:{end_index}')
    shard_samples = manifest['sequences'][start_index:end_index]
    pending = [row for row in shard_samples if row['id'] not in rows_by_id]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f'Loaded {len(manifest["sequences"])} frozen trajectories; {len(pending)} pending.', flush=True)
    if not pending:
        document = build_document(manifest, [rows_by_id[row['id']] for row in shard_samples], True)
        save_json(output_path, document)
        write_summary(args.output_dir / 'summary.md', document)
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(DEVICE)
    print(f'Loading {MODEL_DIR} on {DEVICE}...', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={'': DEVICE},
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': MAX_PIXELS}
    print('Model loaded; starting sequential evaluation.', flush=True)

    for offset, sample in enumerate(pending, 1):
        started = time.perf_counter()
        row: dict[str, Any] = {
            'id': sample['id'], 'city': sample['city'], 'custom_id': sample['custom_id'],
            'shuffle_presented_to_original': sample['shuffle_presented_to_original'],
            'ground_truth_presented_indices': sample['ground_truth_presented_indices'],
        }
        try:
            presented = [sample['images'][index] for index in sample['shuffle_presented_to_original']]
            images = []
            for image in presented:
                path = (ROOT / image['path']).resolve()
                if not path.is_file() or not path.is_relative_to(ROOT):
                    raise FileNotFoundError(path)
                with Image.open(path) as source:
                    if source.size != (1920, 960):
                        raise ValueError(f'Unexpected image size for {path}: {source.size}')
                    images.append(source.convert('RGB'))
            content: list[dict[str, Any]] = [{'type': 'text', 'text': source_prompt['user']}]
            for index in range(8):
                content.extend([{'type': 'text', 'text': f'Image {index}:'}, {'type': 'image'}])
            messages = [
                {'role': 'system', 'content': source_prompt['system']},
                {'role': 'user', 'content': content},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors='pt').to(DEVICE)
            input_tokens = int(inputs['input_ids'].shape[-1])
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            new_tokens = generated[:, input_tokens:]
            response = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()
            row.update(
                status='ok', api_success=True, input_tokens=input_tokens,
                output_tokens=int(new_tokens.shape[-1]), generation_seconds=round(generation_seconds, 3),
                response=response,
            )
            row.update(score_output(response, sample['ground_truth_presented_indices'], source_prompt['step_titles']))
            del inputs, generated, new_tokens, images
        except Exception as error:
            row.update(status='error', api_success=False, error=f'{type(error).__name__}: {error}')
        row['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        rows_by_id[sample['id']] = row
        rows = [rows_by_id[row['id']] for row in shard_samples if row['id'] in rows_by_id]
        document = build_document(manifest, rows, len(rows) == len(shard_samples))
        save_json(output_path, document)
        write_summary(args.output_dir / 'summary.md', document)
        print(
            f"[{offset}/{len(pending)}] {sample['city']}/{sample['id']}: "
            f"status={row['status']} exact={row.get('exact_match')} "
            f"output_tokens={row.get('output_tokens', '—')} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    rows = [rows_by_id[row['id']] for row in shard_samples if row['id'] in rows_by_id]
    complete = len(rows) == len(shard_samples)
    document = build_document(manifest, rows, complete)
    save_json(output_path, document)
    write_summary(args.output_dir / 'summary.md', document)
    print(json.dumps(document['metrics'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
