"""Per-rank invariants for shared-two-GPU deployment; loss implementation unchanged."""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (None, None)


def write(name, payload):
    directory = Path(os.environ['STREETVIEW_AUDIT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    rank, world = rank_world()
    if rank is not None and name != 'rollout_groups.jsonl':
        name = name.replace('.jsonl', f'_rank_{rank}.jsonl')
    with (directory / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': time.time(), 'pid': os.getpid(), 'rank': rank,
                                 'world_size': world, **payload}) + '\n')


def model_initialized(engine):
    if engine.engine_config.forward_only:
        return
    rank, world = rank_world()
    assert world == 2
    uuid = 'GPU-' + str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).removeprefix('GPU-')
    assert uuid == ['GPU-aded96b3-1317-039c-a3b5-cf8871defa07',
                    'GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa'][rank], (rank, uuid)
    adapted = [n for n, m in engine.module.named_modules() if hasattr(m, 'lora_A')]
    assert len(adapted) == 252 and all('language_model.layers.' in n for n in adapted)
    parameters = {n: p for n, p in engine.module.named_parameters() if p.requires_grad}
    assert parameters and all('lora_A' in n or 'lora_B' in n for n in parameters)
    assert all(p.dtype == torch.float32 for p in parameters.values())
    local_count = sum(p.numel() for p in parameters.values())
    total = torch.tensor(local_count, device=torch.cuda.current_device(), dtype=torch.int64)
    dist.all_reduce(total)
    assert total.item() == 174587904, total.item()
    write('model.jsonl', {'gpu_uuid': uuid, 'lora_modules': adapted,
                         'local_trainable_parameters': local_count,
                         'global_trainable_parameters': total.item(), 'trainable_dtype': 'torch.float32'})


def batch_started(data):
    assert data.shape[0] == 256, data.shape
    assert tu.get(data, 'mini_batch_size') == 128
    assert tu.get(data, 'epochs') == 1


def metric_value(value):
    if isinstance(value, Metric):
        return float(value.aggregate())
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    if isinstance(value, list):
        if value and isinstance(value[0], Metric):
            return float(Metric.aggregate_dp(value))
        return sum(metric_value(v) for v in value) / max(len(value), 1)
    return float(value)


def minibatch_finished(worker, data, output, index):
    assert data.shape[0] == 64 and index in range(4)
    steps = {int(s['step'].item()) if isinstance(s['step'], torch.Tensor) else int(s['step'])
             for s in worker.engine.optimizer.state.values() if 'step' in s}
    assert steps and len(steps) == 1, steps
    update = next(iter(steps))
    assert (update - 1) % 4 == index
    metrics = {} if output is None else {k: metric_value(v) for k, v in tu.get(output, 'metrics').items()}
    assert all(math.isfinite(v) for v in metrics.values()), metrics
    if update == 1:
        mask = data['response_mask'].bool()
        difference = (data['old_log_probs'] - data['ref_log_prob']).abs()[mask].max().item()
        assert difference <= 2e-3, difference
        metrics['audit/initial_student_reference_max_difference'] = difference
    write('updates.jsonl', {'optimizer_update': update, 'rollout_step': (update + 3) // 4,
                           'minibatch_index': index, 'local_responses': 64,
                           'global_responses': 128, 'metrics': metrics})
