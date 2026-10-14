"""Actual DDP replica, update and native joint-loss evidence."""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric


def json_scalar(value):
    """Convert NumPy/Torch scalar metrics without weakening finite-value checks."""
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Unsupported audit JSON value: {type(value).__name__}')


def write(name, payload):
    directory=Path(os.environ['STREETVIEW_AUDIT_DIR']);directory.mkdir(parents=True, exist_ok=True)
    rank=dist.get_rank() if dist.is_initialized() else None
    if rank is not None and name != 'rollout_groups.jsonl':
        name=name.replace('.jsonl', f'_rank_{rank}.jsonl')
    with (directory/name).open('a') as f:
        f.write(json.dumps({'time':time.time(), 'pid':os.getpid(), 'rank':rank,
            'world_size':dist.get_world_size() if rank is not None else None, **payload},
            default=json_scalar)+'\n')


def model_initialized(engine):
    assert isinstance(engine.module, DistributedDataParallel)
    assert dist.get_world_size() == 3
    model=engine.module.module
    adapted=[n for n,m in model.named_modules() if hasattr(m,'lora_A')]
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(adapted)==252 and all('language_model.layers.' in n for n in adapted)
    assert sum(p.numel() for p in trainable.values())==174587904
    assert all('lora_' in n and p.dtype==torch.float32 for n,p in trainable.items())
    assert all(p.dtype==torch.bfloat16 for p in model.parameters() if not p.requires_grad)
    uuid='GPU-'+str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).removeprefix('GPU-')
    assert uuid==['GPU-529b98e5-8518-ef78-4293-5c074fe02056',
             'GPU-aded96b3-1317-039c-a3b5-cf8871defa07',
             'GPU-0c8286b3-cb9b-7243-1d88-752a84460579'][dist.get_rank()]
    write('model.jsonl', {'backend':'torch_ddp','gpu_uuid':uuid,'lora_modules':adapted,
        'replicated_trainable_parameters':174587904,'trainable_dtype':'torch.float32',
        'base_dtype':'torch.bfloat16','gradient_accumulation_no_sync':True})


def batch_started(data):
    assert data.shape[0] in (256,203) and tu.get(data,'mini_batch_size')==192 and tu.get(data,'epochs')==1


def metric_value(value):
    if isinstance(value,Metric):return float(value.aggregate())
    if isinstance(value,torch.Tensor):return float(value.detach().mean().item())
    if isinstance(value,list):
        if value and isinstance(value[0],Metric):return float(Metric.aggregate_dp(value))
        return sum(metric_value(v) for v in value)/max(len(value),1)
    return float(value)


def minibatch_finished(worker,data,output,index):
    assert data.shape[0] in (64,11) and index in range(4)
    if data.shape[0]==11:
        assert index==3, 'Only the final mini-batch may contain eleven dispatch rows'
    steps={int(s['step'].item()) for s in worker.engine.optimizer.state.values() if 'step' in s}
    assert len(steps)==1
    update=next(iter(steps));assert (update-1)%4==index
    metrics={k:metric_value(v) for k,v in tu.get(output,'metrics').items()}
    assert all(math.isfinite(v) for v in metrics.values()) and metrics['grad_norm']>0,metrics
    expected=metrics['actor/pg_loss']+.01*metrics['distillation/loss']+.001*metrics['kl_loss']
    assert abs(expected-metrics['loss'])<2e-5,(expected,metrics)
    local_real = int((data['response_mask'].sum(-1)>0).sum())
    global_real = torch.tensor(local_real, device='cuda');dist.all_reduce(global_real)
    write('updates.jsonl',{'backend':'torch_ddp','optimizer_update':update,'rollout_step':(update+3)//4,
        'minibatch_index':index,'local_responses':local_real,'global_responses':global_real.item(),
        'dispatch_rows_per_rank':data.shape[0],'zero_loss_padding_rows':data.shape[0]-local_real,'metrics':metrics,
        'joint_loss_formula_error':abs(expected-metrics['loss'])})
