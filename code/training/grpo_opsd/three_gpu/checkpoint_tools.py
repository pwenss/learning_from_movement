"""Read back native FSDP DTensor LoRA/Adam without rounding FP32 adapters."""
import argparse
import json
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

import streetview_sft_common as common


def inspect(checkpoint, expected_update, output=None):
    checkpoint = Path(checkpoint)
    world = json.loads((checkpoint / 'fsdp_config.json').read_text())['world_size']
    assert world in (2, 3), world
    paths = [checkpoint / f'{kind}_world_size_2_rank_{rank}.pt'
             for rank in range(world) for kind in ('model', 'optim', 'extra_state')]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    shards = [torch.load(checkpoint / f'model_world_size_2_rank_{r}.pt',
                         map_location='cpu', weights_only=False) for r in range(world)]
    assert len(shards[0]) == 504 and all(shard.keys() == shards[0].keys() for shard in shards)
    merged = {}
    for name in sorted(shards[0]):
        assert 'lora_' in name and 'language_model.layers.' in name
        values = [s[name] for s in shards]
        if isinstance(values[0], DTensor):
            assert all(isinstance(v, DTensor) for v in values)
            placement = values[0].placements
            assert all(v.placements == placement and v.shape == values[0].shape for v in values)
            assert len(placement) == 1
            local = [v._local_tensor for v in values]
            if placement[0].is_shard():
                dim = placement[0].dim
                value = torch.cat(local, dim=dim).narrow(dim, 0, values[0].shape[dim]).contiguous()
            else:
                assert placement[0].is_replicate() and torch.equal(local[0], local[1])
                value = local[0]
            assert value.shape == values[0].shape
        else:
            assert all(isinstance(v, torch.Tensor) for v in values)
            assert torch.equal(values[0], values[1])
            value = values[0]
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
        merged[name] = value
    assert sum(v.numel() for v in merged.values()) == 174587904
    moment_elements = 0; populated = []
    for rank in range(world):
        optim = torch.load(checkpoint / f'optim_world_size_2_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        count = 0
        for s in optim['state'].values():
            if not s:continue
            step = int(s['step'].item()) if isinstance(s['step'], torch.Tensor) else int(s['step'])
            assert step == expected_update
            first, second = s['exp_avg'], s['exp_avg_sq']
            assert first.shape == second.shape and first.dtype == second.dtype == torch.float32
            assert torch.isfinite(first).all() and torch.isfinite(second).all() and (second >= 0).all()
            moment_elements += first.numel();count += 1
        assert count > 0
        populated.append(count)
        extra = torch.load(checkpoint / f'extra_state_world_size_2_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        assert isinstance(extra['lr_scheduler'], dict) and isinstance(extra['rng'], dict)
    assert moment_elements == 174587904, moment_elements
    hashes = {p.name: common.hash_file(p) for p in paths}
    assert before == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    report = {'status': 'passed', 'world_size': world, 'checkpoint': str(checkpoint),
              'expected_optimizer_update': expected_update, 'tensor_count': 504,
              'dtype': 'float32', 'global_lora_elements': 174587904,
              'global_adam_moment_elements': moment_elements, 'populated_states_per_rank': populated,
              'source_sha256': hashes, 'checkpoint_unchanged_during_audit': True}
    if output:
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger
        from safetensors.torch import load_file
        from run_training import config_for
        output = Path(output)
        expected = {k.replace('.default.weight', '.weight'): v for k, v in merged.items()}
        merger = FSDPModelMerger(ModelMergerConfig(operation='merge', backend='fsdp',
                    local_dir=str(checkpoint), target_dir=str(output),
                    hf_model_config_path=str(checkpoint / 'huggingface')))
        merger.save_lora_adapter(merged)
        actual = load_file(output / 'lora_adapter/adapter_model.safetensors')
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
        config_path = output / 'lora_adapter/adapter_config.json'
        config = json.loads(config_path.read_text());model = config_for('formal').actor_rollout_ref.model
        assert config['r'] == model.lora_rank and config['lora_alpha'] == model.lora_alpha
        config.update(target_modules=model.target_modules, base_model_name_or_path=model.path,
                      lora_dropout=0.0, inference_mode=True)
        common.atomic_json(config_path, config)
        report['exact_tensor_equality'] = True
        common.atomic_json(output / 'lora_adapter/export_verification.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--expected-update', type=int, required=True)
    p.add_argument('--output')
    p.add_argument('--report')
    a = p.parse_args();torch.set_num_threads(4)
    report = inspect(a.checkpoint, a.expected_update, a.output)
    if a.report:common.atomic_json(Path(a.report), report)
    print(json.dumps(report, indent=2))
