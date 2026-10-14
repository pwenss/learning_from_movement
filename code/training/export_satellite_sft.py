"""Export independent Satellite SFT inputs without changing frozen source data."""
from collections import Counter
import math
import os
from pathlib import Path
import time

from satellite_sft_common import *


def current_relative_path(old):
    prefix = 'dataset/satellite/final/'
    if old.startswith(prefix):
        return old[len(prefix):]
    prefix = 'dataset/satellite/'
    assert old.startswith(prefix), old
    return old[len(prefix):]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Export must run on CPU'
    torch.set_num_threads(4)
    source = ROOT / 'dataset/satellite'
    cot = source / 'cot/satellite_cot_qwen38_flash_all_20261004'
    DATA.mkdir(parents=True, exist_ok=True)
    assert not (DATA / 'export_verification.json').exists(), 'Existing export requires explicit verification, not overwrite'
    source_files = ['manifest.json', 'split_manifest.json', 'cot_data.jsonl',
                    'summary.json', 'verification.json']
    source_hashes = {name: hash_file(cot / name) for name in source_files}
    frozen = json.loads((cot / 'manifest.json').read_text())
    split_doc = json.loads((cot / 'split_manifest.json').read_text())
    original = {x['id']: x for x in frozen['sequences']}
    split = {x['id']: x for x in split_doc['sequences']}
    assert len(original) == len(split) == 23435
    assert set(original) == set(split)
    assert set(split_doc['ood_cities']) == OOD_CITIES
    assert json.loads((cot / 'verification.json').read_text()).get('verified') is True
    prompt = frozen['prompt']
    assert 'satellite' in prompt['system'].lower()
    p = processor()
    assert hash_file(MODEL / 'tokenizer.json') == 'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7'
    train, dev_loss, dev_full, ood = [], [], [], []
    ids, lengths = set(), []
    pixels, eligible_train_pool = 0, 0
    with (cot / 'cot_data.jsonl').open(encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ident = d['id']
            assert ident not in ids
            ids.add(ident)
            m, sp = original[ident], split[ident]
            assert d['city'] == m['city'] == sp['city']
            assert d['split'] == m['split'] == sp['split']
            assert (d['split'] == 'ood_test') == (d['city'] in OOD_CITIES)
            perm = d['shuffle_presented_to_original']
            assert perm == m['shuffle_presented_to_original']
            assert sorted(perm) == list(range(8))
            gt = sorted(range(8), key=perm.__getitem__)
            assert gt == d['ground_truth_presented_indices'] == m['ground_truth_presented_indices']
            frames = {}
            for im in m['images']:
                rel = current_relative_path(im['path'])
                k = int(Path(rel).stem.split('_')[-1])
                assert k not in frames
                frames[k] = rel
            assert set(frames) == set(range(8))
            images = [frames[k] for k in perm]
            assert images == [current_relative_path(x) for x in d['image_paths']]
            assert d['image_labels'] == [f'Image {k}' for k in range(8)]
            assert d['image_order'] == 'shuffled_presentation'
            assert len(set(images)) == 8
            for rel in images:
                path = (source / rel).resolve()
                assert path.is_relative_to(source.resolve()) and path.is_file()
                assert path.parent.name == ident.split('/')[-1]
                with Image.open(path) as im:
                    assert im.size == (768, 768) and im.format == 'JPEG'
                pixels += 1
            record = {'id': ident, 'custom_id': d['custom_id'], 'city': d['city'],
                      'images': images, 'shuffle': perm, 'ground_truth': gt}
            assert d['sft_export_eligible'] == (d['sft_eligible'] and d['split'] == 'train')
            if d['city'] in OOD_CITIES:
                ood.append(record)
            elif d['city'] in DEV_CITIES:
                dev_full.append(record.copy())
            if d['sft_export_eligible']:
                eligible_train_pool += 1
                assert all(d[k] for k in ['api_success', 'exact_match', 'cot_format_valid', 'thinking_length_valid'])
                assert d['finish_reason'] == 'stop' and d['output_thinking_tokens'] < 2048
                assert d['prediction'] == gt
                result = score(d['content'], gt, prompt['step_titles'])
                assert result['cot_format_valid'] and result['exact_match']
                record['content'] = d['content']
                record['input_length'] = len(text_ids(p, render(p, record, prompt, True)))
                assert record['input_length'] <= 8192, (ident, record['input_length'])
                lengths.append(record['input_length'])
                (dev_loss if d['city'] in DEV_CITIES else train).append(record)
            if len(ids) % 1000 == 0:
                atomic_json(DATA / 'export_progress.json', {'checked': len(ids), 'images': pixels, 'time': time.time()})
                print('EXPORT_PROGRESS', len(ids), flush=True)
    assert ids == set(original)
    assert pixels == 187480 and eligible_train_pool == 13164
    assert (len(train), len(dev_loss), len(dev_full), len(ood)) == (12021, 1143, 1684, 4299)
    dev_probe = []
    for city in sorted(DEV_CITIES):
        candidates = sorted((x for x in dev_full if x['city'] == city),
                            key=lambda x: hashlib.sha256(f'{SEED}{city}{x["id"].split("/")[-1]}'.encode()).hexdigest())
        dev_probe.extend(candidates[:128])
    groups = {'train': train, 'validation_loss': dev_loss, 'validation_generation': dev_full,
              'validation_probe': dev_probe, 'ood_test': ood}
    a, b, c = ({x['id'] for x in group} for group in [train, dev_full, ood])
    assert not (a & b or a & c or b & c)
    assert {x['city'] for x in train}.isdisjoint(DEV_CITIES | OOD_CITIES)
    candidates = sorted(train, key=lambda x: x['input_length'], reverse=True)[:8]
    candidates += sorted(train, key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())[:24]
    for record in candidates:
        batch = inputs(p, record, prompt, True)
        assert batch['input_ids'].shape[1] == record['input_length']
    assert source_hashes == {name: hash_file(cot / name) for name in source_files}
    atomic_json(DATA / 'prompt.json', prompt)
    for name, records in groups.items():
        dest = DATA / f'{name}.jsonl'
        tmp = dest.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(tmp, dest)
    report = {'verified': True, 'counts': {k: len(v) for k, v in groups.items()},
              'per_city_train': dict(Counter(x['city'] for x in train)),
              'max_sequence': max(lengths), 'mean_sequence': sum(lengths) / len(lengths),
              'source_hashes': source_hashes,
              'derived_hashes': {name: hash_file(DATA / f'{name}.jsonl') for name in groups},
              'prompt_sha256': hash_file(DATA / 'prompt.json'),
              'source_image_size': [768, 768], 'image_grid_thw': [[1, 44, 44]] * 8,
              'visual_tokens_per_trajectory': 3872, 'image_headers_verified': pixels,
              'max_steps_two_epochs': 2 * math.ceil(math.ceil(len(train) / 2) / 16),
              'ood_shuffle_scheme': 'original frozen Satellite teacher presentation; no reshuffling',
              'dev_split_scheme': 'complete held-out Toronto and Vienna cities; no within-city split',
              'target_field': 'content', 'checks': ['all frozen IDs, split, shuffle, ground truth and image mapping agree',
              'all 187480 current JPEG headers verified 768x768', 'all eligible expanded sequences <=8192',
              '32 actual CPU processor checks including longest, visual grid and assistant-only labels',
              'source files unchanged; train/dev/OOD disjoint; OOD never exported into SFT'],
              'time': time.time()}
    atomic_json(DATA / 'export_verification.json', report)
    (DATA / 'export_progress.json').unlink(missing_ok=True)
    print('EXPORT_VERIFIED', json.dumps({k: report[k] for k in ['counts', 'max_sequence', 'mean_sequence', 'max_steps_two_epochs']}), flush=True)


if __name__ == '__main__':
    main()
