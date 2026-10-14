"""Freeze the full RL split and merge the audited SFT adapter; never alter source data."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streetview_sft_common as common
from PIL import Image

ROOT = common.ROOT
ID = 'streetview_grpo_opsd_qwen3vl8b_lora_v1'
DATA = ROOT / 'training' / ID
RUN = ROOT / 'runs' / ID
MERGED = ROOT / 'models/streetview_sft_checkpoint500_merged_bf16'
SFT = ROOT / 'runs/streetview_sft_qwen3vl8b_lora_v1/training/checkpoint-500'


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def image_check(path):
    with Image.open(path) as image:
        assert image.size == (1024, 512) and image.format == 'JPEG', str(path)
        image.verify()


def prepare_data():
    DATA.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'dataset/streetview'
    cot = source / 'cot/train_qwen38_flash_cot_20260927'
    split_doc = json.loads((source / 'split_manifest.json').read_text())
    frozen = json.loads((cot / 'manifest.json').read_text())
    split = {x['id']: x for x in split_doc['sequences']}
    permutations = {x['id']: x for x in frozen['sequences']}
    mappings = common.read_jsonl(cot / 'image_path_mapping.jsonl')
    assert len(mappings) == len({x['id'] for x in mappings}) == 18149
    setup = (RUN / 'experiment_setup.md').read_text(encoding='utf-8')
    teacher = re.search(r'### 3\.3.*?```tex\n(.*?)\n```', setup, re.S).group(1)
    assert teacher.count('{GT_ORDER}') == 1
    (DATA / 'teacher_prompt.txt').write_text(teacher, encoding='utf-8')
    common.atomic_json(DATA / 'prompt.json', frozen['prompt'])
    processor = common.processor()
    records, all_paths, student_lengths, teacher_lengths = [], set(), [], []
    for item in mappings:
        identity = item['id']
        info = split[identity]
        assert info['split'] == 'train'
        city = info.get('city') or identity.split('/')[0]
        # Source identity formats can differ; the frozen manifest owns city labels.
        city = permutations[identity].get('city', city)
        if city in common.DEV_CITIES:
            continue
        assert city not in common.OOD_CITIES
        perm = item['shuffle_presented_to_original']
        assert perm == permutations[identity]['shuffle_presented_to_original']
        assert sorted(perm) == list(range(8))
        frames = {im['frame']: im['local_path'] for im in item['images']}
        paths = [frames[k] for k in perm]
        assert len(set(paths)) == 8
        for rel in paths:
            path = (source / rel).resolve()
            assert path.is_relative_to(source.resolve()) and path.is_file(), str(path)
            all_paths.add(path)
        row = {'id': identity, 'city': city, 'images': paths, 'shuffle': perm,
               'ground_truth': sorted(range(8), key=perm.__getitem__)}
        assert row['ground_truth'] == item['ground_truth_presented_indices']
        assert row['ground_truth'] == permutations[identity]['ground_truth_presented_indices']
        messages = common.messages(row, frozen['prompt'], False)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, add_vision_id=False)
        student_length = len(common.text_ids(processor, text))
        privilege = teacher.format(GT_ORDER=json.dumps(row['ground_truth']))
        messages[-1]['content'].append({'type': 'text', 'text': '\n\n' + privilege})
        teacher_text = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, add_vision_id=False)
        teacher_length = len(common.text_ids(processor, teacher_text))
        assert student_length <= 5120 and teacher_length <= 5376, (identity, student_length, teacher_length)
        row.update(prompt_length=student_length, teacher_prompt_length=teacher_length)
        student_lengths.append(student_length)
        teacher_lengths.append(teacher_length)
        records.append(row)
        if len(records) % 1000 == 0:
            print('manifest', len(records), flush=True)
    assert len(records) == 16704, (len(records), sorted({r['city'] for r in records}))
    assert len({r['city'] for r in records}) == 23
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(image_check, sorted(all_paths)):
            pass
    records.sort(key=lambda r: r['id'])
    write_jsonl(DATA / 'train.jsonl', records)
    train_ids = {r['id'] for r in records}
    for name, count in [('validation_probe', 256), ('validation_generation', 1445), ('ood_test', 3525)]:
        rows = common.read_jsonl(common.DATA / f'{name}.jsonl')
        assert len(rows) == count and not train_ids.intersection(r['id'] for r in rows)
        write_jsonl(DATA / f'{name}.jsonl', rows)
    from collections import Counter
    report = {'status': 'passed', 'train_count': len(records), 'image_count': len(all_paths),
              'cities': dict(Counter(r['city'] for r in records)),
              'student_prompt_lengths': [min(student_lengths), max(student_lengths)],
              'teacher_prompt_lengths': [min(teacher_lengths), max(teacher_lengths)],
              'teacher_prompt_sha256': hashlib.sha256(teacher.encode()).hexdigest(),
              'source_hashes': {f: common.hash_file(cot / f) for f in ['manifest.json', 'image_path_mapping.jsonl']},
              'derived_hashes': {p.name: common.hash_file(p) for p in DATA.glob('*.jsonl')},
              'time': time.time()}
    common.atomic_json(DATA / 'data_verification.json', report)
    print(json.dumps(report), flush=True)


def merge_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    if (MERGED / 'merge_verification.json').is_file():
        report = json.loads((MERGED / 'merge_verification.json').read_text())
        assert report['sft_adapter_sha256'] == common.hash_file(SFT / 'adapter_model.safetensors')
        print('Verified merged model already exists', flush=True)
        return
    assert not MERGED.exists(), 'Incomplete merged output requires inspection before retry'
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForImageTextToText.from_pretrained(
        str(common.MODEL), torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='flash_attention_2', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(SFT), is_trainable=False).eval()
    p = common.processor()
    prompt = json.loads((DATA / 'prompt.json').read_text())
    row = common.read_jsonl(DATA / 'validation_probe.jsonl')[0]
    batch = common.inputs(p, row, prompt, target=False)
    batch = {k: v.to('cuda') for k, v in batch.items()}
    with torch.inference_mode():
        before = model(**batch, logits_to_keep=1).logits[:, -1].float().log_softmax(-1).cpu()
    model = model.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode():
        after = model(**batch, logits_to_keep=1).logits[:, -1].float().log_softmax(-1).cpu()
    diff = after - before
    assert torch.isfinite(diff).all()
    report = {'base': str(common.MODEL), 'adapter': str(SFT),
              'sft_adapter_sha256': common.hash_file(SFT / 'adapter_model.safetensors'),
              'probe_id': row['id'], 'logprob_mae': diff.abs().mean().item(),
              'logprob_max_abs': diff.abs().max().item(),
              'top1_before': before.argmax(-1).tolist(), 'top1_after': after.argmax(-1).tolist(),
              'note': 'BF16 merge numerical check; full greedy probe equivalence still required',
              'time': time.time()}
    assert report['logprob_mae'] < 0.1, report
    MERGED.mkdir(parents=True)
    model.save_pretrained(MERGED, safe_serialization=True, max_shard_size='4GB')
    p.save_pretrained(MERGED)
    common.atomic_json(MERGED / 'merge_verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        prepare_data()
        merge_model()
        common.atomic_json(RUN / 'provision/prepare_status.json', {'status': 'completed', 'time': time.time()})
    except Exception as exc:
        common.atomic_json(RUN / 'provision/prepare_status.json', {'status': 'failed', 'error': repr(exc), 'time': time.time()})
        raise
