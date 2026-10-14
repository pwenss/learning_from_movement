"""Check every merged tensor against the frozen base and SFT LoRA on CPU."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import shutil
import time

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import streetview_sft_common as common


def main(precise_output=None):
    torch.set_num_threads(8)
    root = common.ROOT
    merged = root / 'models/streetview_sft_checkpoint500_merged_bf16'
    adapter = root / 'runs/streetview_sft_qwen3vl8b_lora_v1/training/checkpoint-500'
    config = json.loads((adapter / 'adapter_config.json').read_text())
    assert config['r'] == 64 and config['lora_alpha'] == 128
    assert not config.get('use_rslora') and not config.get('use_dora')
    weights = load_file(adapter / 'adapter_model.safetensors')
    base_index = json.loads((common.MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    merged_index = json.loads((merged / 'model.safetensors.index.json').read_text())['weight_map']
    assert base_index.keys() == merged_index.keys()
    unchanged = changed = total = unequal = 0
    double_rounding_differences = 0
    precise_state = {} if precise_output else None
    max_difference = 0.0
    started = time.monotonic()
    with ExitStack() as stack:
        handles = {}
        def get(directory, index, key):
            filename = directory / index[key]
            if filename not in handles:
                handles[filename] = stack.enter_context(safe_open(filename, framework='pt', device='cpu'))
            return handles[filename].get_tensor(key)
        for key in sorted(base_index):
            original = get(common.MODEL, base_index, key)
            actual = get(merged, merged_index, key)
            assert original.shape == actual.shape and original.dtype == actual.dtype == torch.bfloat16
            prefix = 'base_model.model.' + key.removesuffix('.weight')
            a = weights.get(prefix + '.lora_A.weight')
            b = weights.get(prefix + '.lora_B.weight')
            if a is None:
                assert b is None and torch.equal(actual, original), key
                unchanged += 1
                if precise_state is not None:
                    precise_state[key] = original
                continue
            assert b is not None and a.dtype == b.dtype == torch.float32
            delta_weight = (b @ a) * (config['lora_alpha'] / config['r'])
            precise = (original.float() + delta_weight).bfloat16()
            # The source PEFT safe_merge casts the delta to base dtype before addition.
            expected = original + delta_weight.bfloat16()
            double_rounding_differences += int((precise != expected).sum())
            if precise_state is not None:
                precise_state[key] = precise
            mismatch = int((expected != actual).sum())
            delta = (expected.float() - actual.float()).abs().max().item()
            # CPU/GPU FP32 reductions may straddle a BF16 rounding boundary.
            assert mismatch / actual.numel() < 1e-4, (key, mismatch, actual.numel())
            assert torch.allclose(actual.float(), expected.float(), atol=2e-6, rtol=.008), (key, delta)
            unequal += mismatch
            total += actual.numel()
            changed += 1
            max_difference = max(max_difference, delta)
    assert changed == 252 and len(weights) == 2 * changed
    report = {'status': 'passed', 'unchanged_tensors_exact': unchanged,
              'adapted_tensors': changed, 'adapted_elements': total,
              'cpu_gpu_rounding_mismatched_elements': unequal,
              'double_vs_single_rounding_different_elements': double_rounding_differences,
              'max_cpu_gpu_merged_weight_difference': max_difference,
              'base': str(common.MODEL), 'adapter_sha256': common.hash_file(adapter / 'adapter_model.safetensors'),
              'seconds': time.monotonic() - started,
              'scope': 'All tensor merge arithmetic and frozen weights; does not prove greedy generation equivalence.'}
    common.atomic_json(merged / 'full_weight_verification.json', report)
    if precise_state is not None:
        target = Path(precise_output)
        assert not target.exists(), 'Refuse to overwrite an existing candidate'
        target.mkdir(parents=True)
        for shard in sorted(set(merged_index.values())):
            save_file({key: precise_state[key] for key in merged_index if merged_index[key] == shard},
                      target / shard, metadata={'format': 'pt'})
        shutil.copy2(merged / 'model.safetensors.index.json', target / 'model.safetensors.index.json')
        for filename in ('config.json', 'generation_config.json', 'preprocessor_config.json',
                         'processor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                         'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
                         'vocab.json', 'merges.txt', 'chat_template.json'):
            if (merged / filename).is_file():
                shutil.copy2(merged / filename, target / filename)
        for shard in sorted(set(merged_index.values())):
            with safe_open(target / shard, framework='pt', device='cpu') as saved:
                for key in saved.keys():
                    assert torch.equal(saved.get_tensor(key), precise_state[key]), key
        common.atomic_json(target / 'merge_verification.json', {
            'status': 'weights_verified', 'recipe': 'BF16(FP32(base) + FP32(B @ A) * alpha / rank)',
            'base': str(common.MODEL), 'sft_adapter_sha256': report['adapter_sha256'],
            'tensor_count': len(precise_state), 'generation_validation': 'pending'})
        print('Saved and read-back verified single-rounding candidate: ' + str(target), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precise-output')
    main(parser.parse_args().precise_output)
