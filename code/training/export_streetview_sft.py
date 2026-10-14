"""Export derived manifests; preserve frozen teacher output and source data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

from PIL import Image
from streetview_sft_common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-pixels', action='store_true')
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'dataset/streetview'
    cot = source / 'cot/train_qwen38_flash_cot_20260927'
    frozen = json.loads((cot / 'manifest.json').read_text())
    prompt = frozen['prompt']
    atomic_json(DATA / 'prompt.json', prompt)
    base_split = json.loads((source / 'split_manifest.json').read_text())
    split = {x['id']: x for x in base_split['sequences']}
    permutations = {x['id']: x for x in frozen['sequences']}
    mapping = {x['id']: x for x in read_jsonl(cot / 'image_path_mapping.jsonl')}
    p = processor()
    assert hash_file(MODEL / 'tokenizer.json') == 'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7'
    train, dev_loss, dev_full, ood = [], [], [], []
    all_ids, lengths = set(), []
    rows = 0
    for d in read_jsonl(cot / 'cot_data.jsonl'):
        rows += 1
        assert d['id'] not in all_ids
        all_ids.add(d['id'])
        if not d['sft_eligible'] and d['city'] not in DEV_CITIES:
            continue
        perm = d['shuffle_presented_to_original']
        gt = sorted(range(8), key=perm.__getitem__)
        assert sorted(perm) == list(range(8))
        assert gt == d['ground_truth_presented_indices']
        assert perm == permutations[d['id']]['shuffle_presented_to_original'] == mapping[d['id']]['shuffle_presented_to_original']
        assert split[d['id']]['split'] == 'train' and d['city'] not in OOD_CITIES
        frames = {im['frame']: im for im in mapping[d['id']]['images']}
        images = [frames[original]['local_path'] for original in perm]
        assert len(images) == len(set(images)) == 8
        for rel in images:
            path = (source / rel).resolve()
            assert path.is_relative_to(source.resolve()) and path.is_file()
            with Image.open(path) as im:
                assert im.size == (1024, 512) and im.format == 'JPEG'
        record = {'id': d['id'], 'custom_id': d['custom_id'], 'city': d['city'], 'images': images,
                  'shuffle': perm, 'ground_truth': gt}
        if d['city'] in DEV_CITIES:
            dev_full.append(record.copy())
        if not d['sft_eligible']:
            continue
        assert all(d[k] for k in ['api_success','exact_match','cot_format_valid','thinking_length_valid'])
        assert d['finish_reason'] == 'stop' and d['output_thinking_tokens'] < 2048
        assert d['prediction'] == gt
        target_score = score(d['content'], gt, prompt['step_titles'])
        assert target_score['cot_format_valid'] and target_score['exact_match']
        record['content'] = d['content']
        record['input_length'] = len(text_ids(p, render(p, record, prompt, True)))
        assert record['input_length'] <= 8192
        lengths.append(record['input_length'])
        (dev_loss if d['city'] in DEV_CITIES else train).append(record)
        if len(lengths) % 500 == 0:
            atomic_json(DATA / 'export_progress.json', {'eligible_checked': len(lengths), 'time': time.time()})
    for d in base_split['sequences']:
        if d['split'] != 'ood_test':
            continue
        route = (source / d['manifest']).parent
        order = list(range(8))
        rng = random.Random(int(hashlib.sha256(f'{SEED}|ood|{d["id"]}'.encode()).hexdigest(), 16))
        rng.shuffle(order)
        images = [str((route / f'frame_{k:02d}.jpg').relative_to(source)) for k in order]
        for rel in images:
            path = (source / rel).resolve()
            assert path.is_relative_to(source.resolve()) and path.is_file()
            with Image.open(path) as im:
                assert im.size == (1024, 512) and im.format == 'JPEG'
        ood.append({'id': d['id'], 'city': d['city'], 'images': images, 'shuffle': order,
                    'ground_truth': sorted(range(8), key=order.__getitem__)})
    assert rows == 18149 and len(train) == 9999 and len(dev_loss) == 932 and len(dev_full) == 1445 and len(ood) == 3525
    dev_probe = []
    for city in sorted(DEV_CITIES):
        candidates = sorted((x for x in dev_full if x['city'] == city), key=lambda x: hashlib.sha256(f'{SEED}{city}{x["id"].split("/")[-1]}'.encode()).hexdigest())
        dev_probe += candidates[:128]
    names = {'train': train, 'validation_loss': dev_loss, 'validation_generation': dev_full,
             'validation_probe': dev_probe, 'ood_test': ood}
    for name, records in names.items():
        with (DATA / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert not {x['id'] for x in train} & {x['id'] for x in dev_full + ood}
    candidates = sorted(train, key=lambda x: x['input_length'], reverse=True)[:8]
    candidates += sorted(train, key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())[:24]
    for record in candidates:
        batch = inputs(p, record, prompt, True)
        assert batch['input_ids'].shape[1] == record['input_length']
    report = {'verified': True, 'counts': {k: len(v) for k,v in names.items()},
              'per_city_train': dict(Counter(x['city'] for x in train)), 'max_sequence': max(lengths),
              'mean_sequence': sum(lengths)/len(lengths), 'source_hashes': {name: hash_file(cot/name) for name in ['manifest.json','cot_data.jsonl','image_path_mapping.jsonl']},
              'derived_hashes': {name: hash_file(DATA/f'{name}.jsonl') for name in names},
              'base_split_sha256':hash_file(source/'split_manifest.json'),
              'prompt_sha256':hash_file(DATA/'prompt.json'),
              'checks': ['all selected local JPEG headers 1024x512', 'all frozen mappings agree', 'no train/dev/OOD overlap',
                         'all selected teacher targets pass frozen quality flags', 'full expanded chat-template token counts <=8192',
                         '32 representative full processor inputs including longest match token counts and assistant masks'],
              'ood_shuffle_seed': SEED, 'ood_shuffle_scheme': 'Python Random from SHA256(seed|ood|id), frozen in ood_test.jsonl',
              'time': time.time()}
    atomic_json(DATA / 'export_verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
