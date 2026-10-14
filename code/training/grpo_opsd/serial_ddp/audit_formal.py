"""Full epoch acceptance for serial shared-vLLM and replicated LoRA DDP."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
STEPS = [*range(17, 171, 17), 174]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tensor_digest(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inspect_checkpoint(path, step):
    config = json.loads((path / 'ddp_config.json').read_text())
    assert config['backend'] == 'torch_ddp' and config['world_size'] == 3
    assert config['global_step'] == step and config['optimizer_update'] == step * 4
    assert len(config['replica_adapter_sha256']) == len(config['replica_adam_sha256']) == 3
    assert len(set(config['replica_adapter_sha256'])) == len(set(config['replica_adam_sha256'])) == 1
    assert config['base_dtype'] == 'bfloat16' and config['adapter_dtype'] == 'float32'
    weights = load_file(str(path / 'adapter_model.safetensors'))
    assert len(weights) == 504 and sum(t.numel() for t in weights.values()) == 174587904
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in weights.values())
    assert tensor_digest(weights.items()) == config['replica_adapter_sha256'][0]
    optimizer = torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert sum(s['exp_avg'].numel() for s in optimizer['state'].values()) == 174587904
    for state in optimizer['state'].values():
        assert int(state['step'].item()) == step * 4
        assert all(state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                   for k in ('exp_avg', 'exp_avg_sq'))
        assert (state['exp_avg_sq'] >= 0).all()
    moments = tensor_digest((f'{i}/{key}', state[key])
                            for i, state in enumerate(optimizer['state'].values())
                            for key in ('exp_avg', 'exp_avg_sq'))
    assert moments == config['replica_adam_sha256'][0]
    for rank in range(3):
        extra = torch.load(path / f'rank_{rank}_rng.pt', map_location='cpu', weights_only=False)
        assert set(extra['rng']) == {'torch', 'cuda', 'numpy', 'python'}
        assert extra['lr_scheduler']
    assert (path / 'adapter_config.json').is_file()
    return {'optimizer_update': step * 4, 'three_replica_hashes_match': True,
            'adapter_sha256': config['replica_adapter_sha256'][0],
            'adam_sha256': moments, 'three_rank_rng': True}


def audit(directory):
    directory = Path(directory)
    config = yaml.safe_load((directory / 'resolved_config.yaml').read_text())
    actor = config['actor_rollout_ref']['actor']
    rollout = config['actor_rollout_ref']['rollout']
    assert config['trainer']['n_gpus_per_node'] == 3
    assert config['data']['train_batch_size'] == 96 and rollout['n'] == 8
    assert (actor['ppo_mini_batch_size'], actor['ppo_epochs'], actor['ppo_micro_batch_size_per_gpu']) == (24, 1, 1)
    assert actor['use_kl_loss'] and actor['kl_loss_coef'] == .001 and actor['kl_loss_type'] == 'low_var_kl'
    assert config['distillation']['distillation_loss']['distillation_loss_coef'] == .01
    assert config['algorithm']['rollout_correction']['bypass_mode']
    assert not config['algorithm']['use_kl_in_reward']
    assert config['trainer']['total_epochs'] == 1 and config['trainer']['save_freq'] == 17
    assert config['serial_ddp']['actor_backend'] == 'torch_ddp'
    assert config['serial_ddp']['shared_vllm_base']
    assert rollout['logprobs_mode'] == 'raw_logprobs'
    assert (rollout['temperature'], rollout['top_p'], rollout['top_k']) == (1, 1, -1)
    updates = []
    for rank in range(3):
        models = rows(directory / f'audit/model_rank_{rank}.jsonl')
        assert len(models) == 1 and models[0]['backend'] == 'torch_ddp'
        assert models[0]['replicated_trainable_parameters'] == 174587904
        assert models[0]['base_dtype'] == 'torch.bfloat16' and models[0]['trainable_dtype'] == 'torch.float32'
        records = rows(directory / f'audit/updates_rank_{rank}.jsonl')
        assert [r['optimizer_update'] for r in records] == list(range(1, 697))
        for record in records:
            update = record['optimizer_update']
            assert record['rollout_step'] == (update + 3) // 4
            assert record['minibatch_index'] == (update - 1) % 4
            assert (record['local_responses'], record['global_responses']) == (64, 192)
            assert all(math.isfinite(v) for v in record['metrics'].values())
            assert record['metrics']['grad_norm'] > 0 and record['joint_loss_formula_error'] < 2e-5
        updates.append(records)
    for i in range(696):
        metrics = updates[0][i]['metrics']
        keys = [k for k in metrics if not k.startswith(('perf/', 'mfu'))]
        assert all(abs(updates[r][i]['metrics'][k] - metrics[k]) < 1e-6
                   for r in (1, 2) for k in keys)
    by_step = {}
    for row in rows(directory / 'audit/rollout_groups.jsonl'):
        by_step.setdefault(row['rollout_step'], []).extend(row['records'])
    assert sorted(by_step) == list(range(1, 175))
    counts = Counter()
    for group in by_step.values():
        per_id = Counter(r['id'] for r in group)
        assert len(group) == 768 and len(per_id) == 96 and set(per_id.values()) == {8}
        assert all(r['reward'] in (0, 1) and 1 <= r['response_length'] <= 3072 for r in group)
        counts.update(per_id)
    train = rows(ROOT / 'training/streetview_grpo_opsd_qwen3vl8b_lora_v1/train.jsonl')
    assert len(train) == len({r['id'] for r in train}) == 16704
    assert len({r['city'] for r in train}) == 23
    assert set(counts) == {r['id'] for r in train} and set(counts.values()) == {8}
    stages = rows(directory / 'audit/serial_stages.jsonl')
    assert [s['rollout_step'] for s in stages] == list(range(1, 175))
    assert all(s['stage_order'] == ['student', 'reference', 'opsd'] and
               s['shared_server_pids'] and s['scores_finite'] and s['responses'] == 768 for s in stages)
    assert (directory / 'exitcode').read_text().strip() == '0'
    assert int((directory / 'latest_checkpointed_iteration.txt').read_text()) == 174
    checkpoints = {}
    for step in STEPS:
        assert (directory / f'global_step_{step}/data.pt').is_file()
        checkpoints[step] = inspect_checkpoint(directory / f'global_step_{step}/actor', step)
    events = EventAccumulator(str(directory / 'tensorboard'), size_guidance={'scalars': 0})
    events.Reload()
    timings = {}
    for name in ('gen', 'serial_student', 'serial_reference', 'serial_opsd', 'serial_switch',
                 'ref', 'update_actor', 'update_weights', 'step'):
        points = {e.step: e.value for e in events.Scalars('timing_s/' + name)}
        assert set(points) == set(range(1, 175))
        assert all(math.isfinite(v) and v >= 0 for v in points.values())
        timings[name] = points
    result = {'status': 'passed', 'rollout_cycles': 174, 'optimizer_updates_per_rank': 696,
              'train_cities': 23, 'unique_trajectories': 16704, 'responses': 133632, 'all_three_ranks_audited': True,
              'serial_shared_vllm': True, 'checkpoints': checkpoints, 'timing_seconds': timings,
              'generation_validation_complete': False}
    temp = directory / 'acceptance.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(directory / 'acceptance.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.directory)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checkpoints', 'timing_seconds')}, indent=2))
