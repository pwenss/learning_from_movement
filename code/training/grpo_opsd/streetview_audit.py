"""Fail-fast invariants and per-optimizer-update observations; no loss changes."""
import json
import math
import os
from pathlib import Path
import time

import torch
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric


def write(name, payload):
    directory = Path(os.environ['STREETVIEW_AUDIT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': time.time(), 'pid': os.getpid(), **payload}) + '\n')


def model_initialized(engine):
    if engine.engine_config.forward_only:
        return
    uuid = 'GPU-' + str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).removeprefix('GPU-')
    expected_uuid = os.environ.get('STREETVIEW_EXPECTED_ACTOR_UUID', 'GPU-aded96b3-1317-039c-a3b5-cf8871defa07')
    assert uuid == expected_uuid, ('Actor assigned to unexpected GPU', uuid, expected_uuid)
    adapted = [name for name, module in engine.module.named_modules() if hasattr(module, 'lora_A')]
    assert len(adapted) == 252 and all('language_model.layers.' in name for name in adapted), adapted
    trainable = {name: parameter for name, parameter in engine.module.named_parameters() if parameter.requires_grad}
    assert trainable and all('lora_A' in name or 'lora_B' in name for name in trainable), list(trainable)
    assert all(parameter.dtype == torch.float32 for parameter in trainable.values())
    write('model.jsonl', {'gpu_uuid': uuid, 'lora_modules': adapted,
                         'trainable_parameters': sum(p.numel() for p in trainable.values()),
                         'trainable_dtypes': sorted({str(p.dtype) for p in trainable.values()})})


def batch_started(data):
    assert data.shape[0] == 512, data.shape
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
    assert data.shape[0] == 128 and index in range(4)
    if output is None:
        return
    metrics = {key: metric_value(value) for key, value in tu.get(output, 'metrics').items()}
    assert all(math.isfinite(v) for v in metrics.values()), metrics
    steps = [int(state['step'].item()) if isinstance(state['step'], torch.Tensor) else int(state['step'])
             for state in worker.engine.optimizer.state.values() if 'step' in state]
    assert steps and len(set(steps)) == 1
    update = steps[0]
    assert (update - 1) % 4 == index, (update, index)
    if update == 1:
        mask = data['response_mask'].bool()
        initial_difference = (data['old_log_probs'] - data['ref_log_prob']).abs()[mask].max().item()
        assert initial_difference <= 2e-3, ('Initial RL adapter changed SFT policy', initial_difference)
        metrics['audit/initial_student_reference_max_difference'] = initial_difference
    write('updates.jsonl', {'optimizer_update': update, 'rollout_step': (update + 3) // 4,
                           'minibatch_index': index, 'responses': data.shape[0], 'metrics': metrics})
