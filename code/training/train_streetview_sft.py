"""Audited BF16 language-only LoRA SFT; smoke mode is isolated from formal training."""
import argparse
import csv
from datetime import timedelta
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from peft import LoraConfig, get_peft_model, TaskType
import torch
from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

from streetview_sft_common import *


class AuditCallback(TrainerCallback):
    def __init__(self, output, smoke, p, prompt, probe, defer_generation=False):
        self.output, self.smoke = Path(output), smoke
        self.p, self.prompt, self.probe = p, prompt, probe
        self.last_update = time.monotonic()
        self.grad_checks = 0
        self.min_nonzero = None
        self.defer_generation = defer_generation

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step > 1:
            gradients = [p.grad for n,p in model.named_parameters() if p.requires_grad]
            assert all(g is not None for g in gradients), 'A trainable adapter has no gradient'
            norms = torch.stack([g.detach().float().norm() for g in gradients])
            assert bool(torch.isfinite(norms).all()), 'Non-finite adapter gradient'
            nonzero = int((norms > 0).sum())
            assert nonzero == len(gradients), 'An adapter parameter has a zero gradient'
            self.min_nonzero = nonzero if self.min_nonzero is None else min(nonzero, self.min_nonzero)
            self.grad_checks += 1

    def on_step_end(self, args, state, control, **kwargs):
        if not self.smoke and control.should_save:
            # DefaultFlow also requests a save at max_steps (626), outside the
            # regular 100-step evaluation schedule.
            control.should_evaluate = True
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self.last_update
        self.last_update = time.monotonic()
        rank = int(os.environ.get('RANK', 0))
        atomic_json(self.output / f'progress.rank{rank}.json', {
            'step': state.global_step, 'max_steps': state.max_steps, 'epoch': state.epoch,
            'last_optimizer_seconds': elapsed, 'time': time.time(),
            'allocated_GiB': torch.cuda.memory_allocated()/2**30,
            'reserved_GiB': torch.cuda.memory_reserved()/2**30,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
            'inactive_split_GiB': torch.cuda.memory_stats().get('inactive_split_bytes.all.current',0)/2**30,
            'gradient_checks': self.grad_checks, 'min_nonzero_adapter_tensors': self.min_nonzero})
        if self.smoke:
            assert torch.cuda.max_memory_reserved()/2**30 <= 55, 'Preflight memory budget exceeded'

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            assert math.isfinite(logs['loss']), 'Non-finite training loss'
        if int(os.environ.get('RANK', 0)) == 0:
            with (self.output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': state.global_step, 'time': time.time(), **(logs or {})})+'\n')

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.smoke and any(x.get('step') == state.global_step and
                                  'eval_loss' in x for x in state.log_history):
            return control
        control.should_save = True
        if not self.smoke:
            # Each saved candidate needs its own loss for selection tie-breaking.
            control.should_evaluate = True
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # A restart may happen after saving weights but before generation or
        # ranking finishes. Complete that saved candidate before continuing.
        if not self.smoke and state.global_step > 0:
            checkpoint = self.output / f'checkpoint-{state.global_step}'
            assert checkpoint.is_dir()
            if self.defer_generation:
                self.last_update = time.monotonic()
                return control
            return self.on_save(args, state, control, model=model, **kwargs)
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint = self.output / f'checkpoint-{state.global_step}'
        if self.smoke:
            return control
        rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
        if world > 1:
            torch.distributed.barrier()
        if self.defer_generation:
            if rank == 0:
                self.p.save_pretrained(checkpoint)
            if world > 1:
                torch.distributed.barrier()
            self.last_update = time.monotonic()
            return control
        pmodel = model.module if hasattr(model, 'module') else model
        result = generation_eval(pmodel, self.p, self.probe, self.prompt,
                                 self.output / 'validation' / checkpoint.name, rank, world)
        if rank == 0:
            self.p.save_pretrained(checkpoint)
            best_file = self.output / 'best_checkpoint.json'
            previous = json.loads(best_file.read_text()) if best_file.exists() else None
            val_loss = next((x['eval_loss'] for x in reversed(state.log_history) if 'eval_loss' in x), None)
            key = [result['overall']['exact_match'], result['overall']['directed_adjacency_accuracy'],
                   -(val_loss if val_loss is not None else float('inf'))]
            if previous is None or key > previous['selection_key']:
                atomic_json(best_file, {'checkpoint': str(checkpoint), 'step': state.global_step,
                            'selection_key': key, 'validation': result, 'validation_loss': val_loss,
                            'protocol': 'fixed 256 validation trajectories; greedy; max_new_tokens=3072'})
            best = Path(json.loads(best_file.read_text())['checkpoint'])
            checkpoints = sorted((x for x in self.output.glob('checkpoint-*') if x.is_dir() and x.name[11:].isdigit()),
                                 key=lambda x: int(x.name[11:]))
            keep = {best, *checkpoints[-2:]}
            for folder in checkpoints:
                if folder not in keep:
                    assert folder.resolve().parent == self.output.resolve()
                    shutil.rmtree(folder)
        if world > 1:
            torch.distributed.barrier()
        self.last_update = time.monotonic()
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume')
    ap.add_argument('--defer-generation', action='store_true')
    ap.add_argument('--max-steps', type=int, default=-1)
    args = ap.parse_args()
    assert not (args.smoke and args.defer_generation)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.multiprocessing.set_start_method('spawn', force=True)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    assert torch.cuda.is_bf16_supported()
    set_seed(SEED)
    assert json.loads((DATA / 'export_verification.json').read_text())['verified']
    assert json.loads((MODEL / 'model_verification.json').read_text())['verified']
    output = RUN / ('preflight' if args.smoke else 'training')
    output.mkdir(parents=True, exist_ok=True)
    p = processor()
    prompt = json.loads((DATA / 'prompt.json').read_text())
    train = read_jsonl(DATA / 'train.jsonl')
    if args.smoke:
        # Every rank sees repeated extreme-length and ordinary records.
        longest = sorted(train, key=lambda x: x['input_length'], reverse=True)[:64]
        ordinary = sorted(train, key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())[:64]
        train = (longest + ordinary) * 8
    val = read_jsonl(DATA / 'validation_loss.jsonl')
    # Pin two complete epochs including each rank's short final accumulation group.
    world = int(os.environ.get('WORLD_SIZE', 1))
    max_steps = args.max_steps if args.max_steps >= 0 else 2 * math.ceil(math.ceil(len(train) / world) / 16)
    backend = 'flash_attention_2' if __import__('importlib').util.find_spec('flash_attn') else 'sdpa'
    config = {
        'run_id': RUN_ID, 'smoke': args.smoke, 'base_model': str(MODEL), 'attention': backend,
        'rank':64, 'alpha':128, 'dropout':.05, 'learning_rate':5e-5,
        'micro_batch':1, 'gradient_accumulation':16, 'world_size':int(os.environ.get('WORLD_SIZE',1)),
        'epochs':2, 'max_steps':max_steps, 'seed':SEED, 'min_pixels':65536, 'max_pixels':524288, 'max_sequence':8192,
        'cuda_allocator':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default'),
        'base_revision':json.loads((MODEL/'model_verification.json').read_text())['revision'],
        'resume_from':args.resume, 'generation_validation_schedule':'after_training' if args.defer_generation else 'during_training',
        'image_size':[1024,512], 'images_per_trajectory':8,
        'assistant_only_loss':True, 'target_field':'content', 'frozen_visual_encoder':True,
        'code_hashes':{n:hash_file(Path(__file__).parent/n) for n in
                       ['train_streetview_sft.py','streetview_sft_common.py']},
        'target_modules': TARGET_MODULES, 'train_count':len(train), 'validation_loss_count':len(val)}
    targs = TrainingArguments(
        output_dir=str(output), bf16=True, tf32=True, learning_rate=5e-5,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, num_train_epochs=2,
        max_steps=max_steps, optim='adamw_torch', adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, weight_decay=.01, lr_scheduler_type='cosine', warmup_ratio=.03,
        max_grad_norm=1., gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False, ddp_timeout=345600,
        dataloader_num_workers=4, dataloader_pin_memory=True, dataloader_prefetch_factor=2,
        eval_strategy='no' if args.smoke else 'steps', eval_steps=100,
        prediction_loss_only=True, save_strategy='steps', save_steps=25 if args.smoke else 100,
        logging_steps=1 if args.smoke else 10, logging_nan_inf_filter=False,
        report_to='none', remove_unused_columns=False, seed=SEED, data_seed=SEED,
        save_safetensors=True, save_total_limit=None, save_only_model=False)
    config['ddp_timeout_seconds'] = targs.ddp_timeout
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16,
                    local_files_only=True, attn_implementation=backend)
    matched = [n for n,m in model.named_modules() if re.fullmatch(TARGET_MODULES,n)]
    assert len(matched) == 252, (len(matched), matched[:8])
    model = get_peft_model(model, LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,
                    target_modules=TARGET_MODULES,bias='none',task_type=TaskType.CAUSAL_LM))
    trainable = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    assert sum(trainable.values()) == 174587904
    assert all('language_model.layers.' in n and 'lora_' in n for n in trainable)
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if int(os.environ.get('RANK',0)) == 0:
        atomic_json(output/'configuration.json', config)
        atomic_json(output/'trainable_parameters.json', {'parameters':trainable,'matched_modules':matched})
    callback = AuditCallback(output,args.smoke,p,prompt,read_jsonl(DATA/'validation_probe.jsonl'),
                             defer_generation=args.defer_generation)
    trainer = Trainer(model=model,args=targs,train_dataset=Records(train),eval_dataset=Records(val),
                      data_collator=Collator(p,prompt),processing_class=p,callbacks=[callback])
    raw = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],
                         check=True,capture_output=True,text=True,timeout=10).stdout
    placement = [row[1].strip() for row in csv.reader(io.StringIO(raw)) if int(row[0].strip()) == os.getpid()]
    expected_uuid = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank]
    assert placement == [expected_uuid], (os.getpid(), placement, expected_uuid)
    atomic_json(output/f'gpu_placement.rank{int(os.environ.get("RANK",0))}.json',
                {'verified':True,'pid':os.getpid(),'gpu_uuid':placement[0],'local_rank':local_rank,
                 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'time':time.time()})
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()
    if args.smoke:
        if int(os.environ.get('RANK',0)) == 0:
            assert callback.grad_checks > 0
            atomic_json(output/'preflight_complete.json', {'passed':True,'step':trainer.state.global_step,
                        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                        'gradient_checks':callback.grad_checks,'resume_from':args.resume,'time':time.time()})
    else:
        if int(os.environ.get('RANK',0)) == 0:
            if args.defer_generation:
                assert all((output/f'checkpoint-{step}'/'adapter_model.safetensors').is_file()
                           for step in [100,200,300,313,400,500,600,626])
            else:
                assert (output/'best_checkpoint.json').exists()
            assert trainer.state.global_step == max_steps and trainer.state.epoch >= 1.999
            atomic_json(output/'training_complete.json', {'completed':True,'step':trainer.state.global_step,
                         'epoch':trainer.state.epoch,'time':time.time()})
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
