"""Derive frozen Satellite RL inputs and a single-round BF16 frozen SFT base."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
ID = 'satellite_grpo_opsd_qwen3vl8b_lora_v1'
DATA = ROOT / 'training' / ID
RUN = ROOT / 'runs' / ID
SFT_DATA = ROOT / 'training/satellite_sft_qwen3vl8b_lora_v1'
SFT = ROOT / 'runs/satellite_sft_qwen3vl8b_lora_v1/training/checkpoint-700'
BASE = Path(__import__('os').environ.get('LFM_MODEL', '/path/to/models/Qwen3-VL-8B-Instruct'))
MERGED = ROOT / 'models/satellite_sft_checkpoint700_single_round_bf16'
TEACHER_HASH = '6c19e1c06dc8b07953dc0a5bd9d8ff4e43fa94f9a25cc0d5252e309d1b75fd92'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def prepare_data():
    from PIL import Image
    source = ROOT / 'dataset/satellite/cot/satellite_cot_qwen38_flash_all_20261004'
    manifest = json.loads((source / 'manifest.json').read_text())
    split = json.loads((source / 'split_manifest.json').read_text())
    by_id = {r['id']: r for r in split['sequences']}
    assert len(by_id) == 23435
    dev = {'toronto', 'vienna'}
    ood = {'philadelphia', 'buenos_aires', 'berlin', 'osaka', 'melbourne'}
    DATA.mkdir(parents=True, exist_ok=True)
    assert not (DATA / 'train.jsonl').exists(), 'Refuse to replace frozen derived data'
    records = []
    selected = set()
    header_count = 0
    for item in manifest['sequences']:
        ident = item['id']
        assert item['city'] == by_id[ident]['city']
        if item['city'] in dev | ood:
            continue
        assert by_id[ident]['split'] == 'train'
        assert ident not in selected
        selected.add(ident)
        permutation = item['shuffle_presented_to_original']
        assert sorted(permutation) == list(range(8))
        gt = [permutation.index(i) for i in range(8)]
        assert gt == item['ground_truth_presented_indices']
        frames = {}
        for im in item['images']:
            p = im['path'].replace('dataset/satellite/final/', '')
            assert p.startswith('cities/') and '..' not in Path(p).parts
            frame = int(re.fullmatch(r'frame_(\d+)\.jpg', Path(p).name)[1])
            assert frame not in frames
            frames[frame] = p
            with Image.open(ROOT / 'dataset/satellite' / p) as image:
                assert image.size == (768, 768) and image.format == 'JPEG', (ident, p)
            header_count += 1
        assert set(frames) == set(range(8))
        records.append({'id': ident, 'city': item['city'],
                        'images': [frames[i] for i in permutation],
                        'shuffle_presented_to_original': permutation,
                        'ground_truth': gt})
        if len(records) % 1000 == 0:
            print(f'Headers verified: {len(records)} trajectories / {header_count} images', flush=True)
    assert len(records) == 17452 and len({r['city'] for r in records}) == 23
    (DATA / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    hashes = {}
    eval_sets = {}
    for name, expected in [('validation_probe', 256), ('validation_generation', 1684), ('ood_test', 4299)]:
        path = SFT_DATA / (name + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = {r['id'] for r in rows}
        assert len(rows) == len(ids) == expected and not selected & ids
        assert {r['city'] for r in rows} <= (ood if name == 'ood_test' else dev)
        for r in rows:
            assert len(r['images']) == 8 and sorted(r['ground_truth']) == list(range(8))
            assert all((ROOT / 'dataset/satellite' / p).is_file() for p in r['images'])
        shutil.copy2(path, DATA / path.name)
        assert sha(path) == sha(DATA / path.name)
        hashes[name] = sha(path)
        eval_sets[name] = ids
    assert eval_sets['validation_probe'] <= eval_sets['validation_generation']
    assert not eval_sets['validation_generation'] & eval_sets['ood_test']
    shutil.copy2(SFT_DATA / 'prompt.json', DATA / 'prompt.json')
    teacher = ROOT / 'training/streetview_grpo_opsd_qwen3vl8b_lora_v1/teacher_prompt.txt'
    assert sha(teacher) == TEACHER_HASH
    shutil.copy2(teacher, DATA / 'teacher_prompt.txt')
    write(DATA / 'data_verification.json', {
        'status': 'headers_and_splits_verified', 'train_count': len(records),
        'training_image_headers_verified': header_count,
        'cities': dict(sorted(Counter(r['city'] for r in records).items())),
        'manifest_sha256': sha(source / 'manifest.json'),
        'split_manifest_sha256': sha(source / 'split_manifest.json'),
        'train_sha256': sha(DATA / 'train.jsonl'), 'evaluation_sha256': hashes,
        'prompt_sha256': sha(DATA / 'prompt.json'), 'teacher_prompt_sha256': TEACHER_HASH,
        'all_train_trajectories_included': True, 'cot_quality_filter_applied': False,
        'train_validation_ood_disjoint': True, 'tail_batch': 76,
        'cycles': 182, 'optimizer_updates_per_rank': 728,
        'source_images_modified': False,
    })
    print('Frozen Satellite RL data verified: 17452 train; 256 probe; 1684 validation; 4299 OOD.', flush=True)


def merge():
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import torch
    torch.set_num_threads(8)
    assert not MERGED.exists(), 'Refuse to overwrite a merged model'
    config = json.loads((SFT / 'adapter_config.json').read_text())
    assert config['r'] == 64 and config['lora_alpha'] == 128
    assert not config.get('use_rslora') and not config.get('use_dora')
    adapter = load_file(SFT / 'adapter_model.safetensors')
    index = json.loads((BASE / 'model.safetensors.index.json').read_text())['weight_map']
    MERGED.mkdir(parents=True)
    adapted = unchanged = 0
    source_shards = {}
    merged_shards = {}
    for shard in sorted(set(index.values())):
        state = {}
        with safe_open(BASE / shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                original = f.get_tensor(key)
                assert original.dtype == torch.bfloat16
                prefix = 'base_model.model.' + key.removesuffix('.weight')
                a = adapter.get(prefix + '.lora_A.weight')
                b = adapter.get(prefix + '.lora_B.weight')
                if a is None:
                    assert b is None
                    state[key] = original
                    unchanged += 1
                else:
                    assert a.dtype == b.dtype == torch.float32
                    state[key] = (original.float() + (b @ a) * (config['lora_alpha'] / config['r'])).bfloat16()
                    adapted += 1
        save_file(state, MERGED / shard, metadata={'format': 'pt'})
        with safe_open(MERGED / shard, framework='pt', device='cpu') as f:
            assert set(f.keys()) == set(state)
            for key, value in state.items():
                assert torch.equal(value, f.get_tensor(key)), key
        source_shards[shard] = sha(BASE / shard)
        merged_shards[shard] = sha(MERGED / shard)
        print('Single-round merge and read-back verified: ' + shard, flush=True)
        del state
    assert adapted == 252 and len(adapter) == 504
    for name in ('model.safetensors.index.json', 'config.json', 'generation_config.json',
                 'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json',
                 'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json',
                 'chat_template.jinja', 'vocab.json', 'merges.txt', 'chat_template.json'):
        if (BASE / name).is_file():
            shutil.copy2(BASE / name, MERGED / name)
    write(MERGED / 'merge_verification.json', {
        'status': 'weights_verified', 'recipe': 'BF16(FP32(base) + FP32(B @ A) * alpha / rank)',
        'base': str(BASE), 'sft_checkpoint': str(SFT),
        'sft_adapter_sha256': sha(SFT / 'adapter_model.safetensors'),
        'adapted_tensors': adapted, 'unchanged_tensors': unchanged,
        'source_shard_sha256': source_shards, 'merged_shard_sha256': merged_shards,
        'generation_validation': 'pending_step0_probe',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['data', 'merge', 'all'], default='all')
    args = parser.parse_args()
    start = time.monotonic()
    if args.part in ('data', 'all'):
        prepare_data()
    if args.part in ('merge', 'all'):
        merge()
    print(f'Preparation finished in {time.monotonic() - start:.1f}s', flush=True)
