"""Resume-safe two-device greedy evaluation with strict trajectory metrics."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

from peft import PeftModel
import torch
from transformers import Qwen3VLForConditionalGeneration

from streetview_sft_common import *


def load_model(adapter=None):
    backend = 'flash_attention_2' if __import__('importlib').util.find_spec('flash_attn') else 'sdpa'
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16,
                    local_files_only=True, attn_implementation=backend)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.to(torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--adapter')
    args = ap.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if world > 1:
        # Shared GPU load can make full OOD ranks finish many hours apart.
        # Match the controller's bounded 96-hour evaluation stage.
        torch.distributed.init_process_group('nccl', timeout=timedelta(hours=96))
    p = processor()
    prompt = json.loads((DATA / 'prompt.json').read_text())
    model = load_model(args.adapter)
    result = generation_eval(model, p, read_jsonl(args.manifest), prompt, args.output, rank, world)
    if rank == 0:
        print(json.dumps(result, indent=2), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
