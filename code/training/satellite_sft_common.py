"""Shared data, preprocessing and evaluation contract for Satellite SFT."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN_ID = 'satellite_sft_qwen3vl8b_lora_v1'
MODEL = Path(__import__('os').environ.get('LFM_MODEL', '/path/to/models/Qwen3-VL-8B-Instruct'))
DATA = ROOT / 'training' / RUN_ID
RUN = ROOT / 'runs' / RUN_ID
DEV_CITIES = {'toronto', 'vienna'}
OOD_CITIES = {'philadelphia', 'buenos_aires', 'berlin', 'osaka', 'melbourne'}
SEED = 20261006
TARGET_MODULES = r'.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$'


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def processor():
    p = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    p.image_processor.min_pixels = 65536
    p.image_processor.max_pixels = 524288
    p.image_processor.size = {'shortest_edge': 65536, 'longest_edge': 524288}
    p.tokenizer.padding_side = 'right'
    assert p.image_processor.patch_size == 16
    assert p.image_processor.merge_size == 2
    assert p.tokenizer.convert_tokens_to_ids('<|im_end|>') == 151645
    return p


def messages(record, prompt, include_target):
    content = [{'type': 'text', 'text': prompt['user'] + '\n'}]
    for k in range(8):
        content += [{'type': 'text', 'text': f'Image {k}:\n'}, {'type': 'image'}]
    result = [{'role': 'system', 'content': prompt['system']}, {'role': 'user', 'content': content}]
    if include_target:
        result.append({'role': 'assistant', 'content': record['content']})
    return result


def render(p, record, prompt, target):
    return p.apply_chat_template(messages(record, prompt, target), tokenize=False,
                                 add_generation_prompt=not target, add_vision_id=False)


def text_ids(p, text, grids=None):
    grids = grids or [[1, 44, 44] for _ in range(8)]
    assert text.count('<|image_pad|>') == 8
    for grid in grids:
        n = grid[0] * grid[1] * grid[2] // 4
        text = text.replace('<|image_pad|>', '<|placeholder|>' * n, 1)
    text = text.replace('<|placeholder|>', '<|image_pad|>')
    return p.tokenizer(text, add_special_tokens=False)['input_ids']


def inputs(p, record, prompt, target=True):
    opened = []
    try:
        for rel in record['images']:
            path = (ROOT / 'dataset/satellite' / rel).resolve()
            assert path.is_relative_to((ROOT / 'dataset/satellite').resolve())
            with Image.open(path) as im:
                assert im.size == (768, 768), (path, im.size)
                opened.append(im.convert('RGB'))
        result = p(text=[render(p, record, prompt, target)], images=opened,
                   padding=False, return_tensors='pt')
    finally:
        for im in opened:
            im.close()
    result.pop('token_type_ids', None)
    result.pop('mm_token_type_ids', None)
    grid = result['image_grid_thw']
    assert grid.shape == (8, 3)
    assert grid.tolist() == [[1, 44, 44]] * 8, grid.tolist()
    ids = result['input_ids'][0]
    assert int((ids == 151655).sum()) == 3872
    assert ids.numel() <= 8192
    if target:
        prefix = text_ids(p, render(p, record, prompt, False), grid.tolist())
        assert ids[:len(prefix)].tolist() == prefix, 'Prompt/target boundary mismatch'
        labels = result['input_ids'].clone()
        labels[:, :len(prefix)] = -100
        eos = torch.where(ids == 151645)[0][-1].item()
        labels[:, eos + 1:] = -100
        decoded = p.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
        assert decoded == record['content'] + '<|im_end|>', 'Assistant supervision changed'
        result['labels'] = labels
    return result


class Records(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]


class Collator:
    def __init__(self, p, prompt):
        self.p, self.prompt = p, prompt
    def __call__(self, batch):
        assert len(batch) == 1, 'This run is audited for one complete trajectory per device'
        return inputs(self.p, batch[0], self.prompt, True)


def score(text, gt, step_titles, clipped=False):
    answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.S)
    pred = None
    if len(answers) == 1:
        try:
            candidate = json.loads(answers[0])
            if isinstance(candidate, list) and len(candidate) == 8 and all(type(x) is int for x in candidate) and sorted(candidate) == list(range(8)):
                pred = candidate
        except (ValueError, TypeError):
            pass
    normalized = text.replace('\u2011', '-')
    blocks = re.fullmatch(r'\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*', normalized, flags=re.S)
    fmt = bool(blocks and normalized.count('<think>') == 1 and normalized.count('<answer>') == 1)
    if fmt:
        body = blocks[1]
        positions = [body.find(x.replace('\u2011', '-')) for x in step_titles]
        fmt = all(x >= 0 for x in positions) and positions == sorted(positions) and all(body.count(x.replace('\u2011', '-')) == 1 for x in step_titles)
    out = {'valid_answer': pred is not None, 'cot_format_valid': fmt,
           'exact_match': pred == gt, 'truncated': clipped, 'prediction': pred,
           'pairwise_accuracy': 0., 'kendall_tau': -1.,
           'directed_adjacency_accuracy': 0., 'position_accuracy': 0.}
    if pred is not None:
        pos = {v: k for k, v in enumerate(pred)}
        correct = sum(pos[gt[i]] < pos[gt[j]] for i in range(8) for j in range(i + 1, 8))
        edges = set(zip(pred, pred[1:]))
        out.update(pairwise_accuracy=correct / 28, kendall_tau=2 * correct / 28 - 1,
                   directed_adjacency_accuracy=sum((gt[k], gt[k + 1]) in edges for k in range(7)) / 7,
                   position_accuracy=sum(a == b for a, b in zip(gt, pred)) / 8)
    return out


def summarize(rows):
    keys = ['valid_answer', 'cot_format_valid', 'exact_match', 'truncated',
            'pairwise_accuracy', 'kendall_tau', 'directed_adjacency_accuracy', 'position_accuracy', 'output_tokens']
    def agg(items):
        return {'n': len(items), **{k: sum(x[k] for x in items) / len(items) for k in keys}}
    assert rows
    cities = sorted(set(x['city'] for x in rows))
    per = {c: agg([x for x in rows if x['city'] == c]) for c in cities}
    return {'overall': agg(rows), 'per_city': per,
            'city_macro_average': {k: sum(v[k] for v in per.values()) / len(per) for k in keys},
            'invalid_score_policy': 'invalid permutation: pairwise/adjacency/position=0, kendall_tau=-1; all records remain in denominators'}


@torch.inference_mode()
def generation_eval(model, p, records, prompt, directory, rank=0, world=1):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'predictions.rank{rank}.jsonl'
    existing = read_jsonl(output) if output.exists() else []
    complete = {x['id'] for x in existing}
    selected = records[rank::world]
    allowed = {x['id'] for x in selected}
    assert complete <= allowed
    was_training = model.training
    cache_config = getattr(model.config, 'text_config', model.config)
    old_cache = cache_config.use_cache
    model.eval()
    cache_config.use_cache = True
    device = next(model.parameters()).device
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Match standalone baseline inference, independently of Trainer's TF32 setting.
        torch.backends.cuda.matmul.allow_tf32 = False
        with output.open('a', encoding='utf-8') as f:
            for i, record in enumerate(selected):
                if record['id'] in complete:
                    continue
                started = time.monotonic()
                batch = inputs(p, record, prompt, False).to(device)
                generated = model.generate(**batch, do_sample=False, max_new_tokens=3072,
                                           eos_token_id=151645, pad_token_id=p.tokenizer.pad_token_id,
                                           use_cache=True)
                target = generated[0, batch['input_ids'].shape[1]:]
                text = p.tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                clipped = len(target) >= 3072 and int(target[-1]) != 151645
                row = {'id': record['id'], 'city': record['city'], 'ground_truth': record['ground_truth'],
                       'shuffle': record['shuffle'], 'text': text, 'output_tokens': len(target),
                       'seconds': time.monotonic() - started,
                       **score(text, record['ground_truth'], prompt['step_titles'], clipped)}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                atomic_json(directory / f'progress.rank{rank}.json', {'completed': len(complete) + 1,
                            'total': len(selected), 'last_id': record['id'], 'time': time.time()})
                complete.add(record['id'])
                del batch, generated, target
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        cache_config.use_cache = old_cache
        model.train(was_training)
    if world > 1:
        torch.distributed.barrier()
    result = None
    if rank == 0:
        rows = [x for k in range(world) for x in read_jsonl(directory / f'predictions.rank{k}.jsonl')]
        assert len(rows) == len(records) and {x['id'] for x in rows} == {x['id'] for x in records}
        result = summarize(rows)
        atomic_json(directory / 'summary.json', result)
    if world > 1:
        torch.distributed.barrier()
    return result
