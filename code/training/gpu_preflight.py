"""Verify selected UUIDs, BF16, attention backend and two-rank NCCL communication."""
import importlib.metadata
import importlib.util
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
from streetview_sft_common import RUN, atomic_json


def main():
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa,GPU-aded96b3-1317-039c-a3b5-cf8871defa07'
    torch.cuda.set_device(local)
    assert torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported()
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=5))
    x = torch.tensor([rank + 1.], device='cuda')
    torch.distributed.all_reduce(x)
    assert x.item() == 3.
    q = torch.randn(1, 4, 64, 128, dtype=torch.bfloat16, device='cuda')
    y = torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
    assert torch.isfinite(y).all()
    fa = importlib.util.find_spec('flash_attn') is not None
    if fa:
        from flash_attn import flash_attn_func
        z = flash_attn_func(q.transpose(1,2),q.transpose(1,2),q.transpose(1,2),causal=True)
        assert torch.isfinite(z).all()
    versions = {}
    for package in ['torch','torchvision','transformers','peft','accelerate','flash-attn','tokenizers','huggingface-hub','pillow']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    atomic_json(RUN / f'gpu_preflight.rank{rank}.json', {'passed':True,'rank':rank,
                'cuda_device':local,'device_name':torch.cuda.get_device_name(local),
                'cuda_runtime':torch.version.cuda,'nccl':torch.cuda.nccl.version(),
                'attention':'flash_attention_2' if fa else 'sdpa','versions':versions})
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print('GPU_PREFLIGHT_PASSED',rank,flush=True)


if __name__ == '__main__':
    main()
