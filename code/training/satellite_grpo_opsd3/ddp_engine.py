"""Replicated LoRA DDP engine using pinned verl's VLM forward and native losses.

The frozen SFT base stays BF16, while trainable LoRA and Adam states stay FP32.
Only LoRA gradients are synchronized, once per accumulated optimizer update.
No FSDP wrapper, flattened parameter, sharded gradient or sharded Adam is used.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from types import FunctionType
from inspect import signature

import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from verl.utils.model import convert_weight_keys
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.config.engine import EngineConfig, FSDPEngineConfig


@dataclass
class SatelliteDDPEngineConfig(FSDPEngineConfig):
    """Preserve the native field schema with a separately registered backend."""

    strategy: str = 'satellite_ddp'

    def __post_init__(self):
        EngineConfig.__post_init__(self)
        assert self.strategy == 'satellite_ddp'


def tensor_digest(parameters):
    digest = sha256()
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def zero_padding_loss(*, model_output, data, **kwargs):
    assert int(data['response_mask'].sum()) == 0
    values = model_output['log_probs']
    if values.is_nested:
        values = values.values()
    # Keep the real model graph and DDP hooks for the final synchronous
    # backward, while excluding this row from losses and native range metrics.
    return values.sum() * 0.0, {}


@EngineRegistry.register(model_type='language_model', backend='satellite_ddp', device='cuda')
class SatelliteDDPEngine(FSDPEngineWithLMHead):
    """Reuse tensor preparation/forward, replacing all FSDP storage/lifecycle code."""

    def _wrap_ddp(self, module, *, init_sync):
        return DDP(module, device_ids=[torch.cuda.current_device()], init_sync=init_sync,
                   broadcast_buffers=False, find_unused_parameters=False,
                   gradient_as_bucket_view=True, bucket_cap_mb=25)

    def forward_step(self, micro_batch, loss_function, forward_only):
        if not forward_only and int(micro_batch['response_mask'].sum()) == 0:
            assert micro_batch.shape[0] == 1
            from streetview_audit import write
            write('zero_loss_padding_micro.jsonl', {'dispatch_padding': True,
                'valid_loss_tokens': 0, 'native_real_sample_loss_unchanged': True})
            loss_function = zero_padding_loss
        return super().forward_step(micro_batch, loss_function, forward_only)

    def _init_device_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh
        self.device_mesh = init_device_mesh('cuda', (dist.get_world_size(),), mesh_dim_names=['ddp'])
        self.ulysses_device_mesh = self.ulysses_parallel_group = None
        self.ulysses_sequence_parallel_size = 1
        self.use_ulysses_sp = False

    def initialize(self):
        assert self._is_lora and not self.engine_config.forward_only and not self._qat_enabled
        torch.manual_seed(self.engine_config.seed)
        torch.cuda.manual_seed_all(self.engine_config.seed)
        # All ranks load real checkpoint tensors. FSDP's meta-on-nonzero-rank
        # initialization would leave DDP with unmaterialized parameters.
        original = FSDPEngineWithLMHead._build_module
        namespace = {**original.__globals__, 'get_init_weight_context_manager': lambda **_: nullcontext}
        build = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        module = build(self)
        module.enable_input_require_grads()
        module = get_peft_model(module, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha, target_modules=self.model_config.target_modules,
            bias='none', lora_dropout=0.0))
        for parameter in module.parameters():
            parameter.data = parameter.data.to(torch.float32 if parameter.requires_grad else torch.bfloat16)
        module = module.to(torch.cuda.current_device())
        try:
            self.pass_packed_cu_seqlens = 'cu_seqlens' in signature(module.forward).parameters
        except (TypeError,ValueError):
            self.pass_packed_cu_seqlens = False
        self.module = self._wrap_ddp(module, init_sync=True)
        self._parameter_ids = tuple(id(p) for p in module.parameters())
        self._ddp_rebuild_required = False
        self.optimizer = self._build_optimizer(self.module)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        self._autocast_dtype = torch.bfloat16
        self.scaler = None
        from streetview_audit import model_initialized
        model_initialized(self)
        self.to('cpu', model=self._is_offload_param, optimizer=self._is_offload_optimizer)

    @contextmanager
    def _gradient_sync_context(self, *, is_last_micro_batch):
        assert not self._ddp_rebuild_required, 'DDP hooks must be rebuilt after device conversion'
        with (nullcontext() if is_last_micro_batch else self.module.no_sync()):
            yield

    @contextmanager
    def train_mode(self, **kwargs):
        automatic = not kwargs.get('disable_auto_offload', False)
        if automatic:
            self.to('cuda', model=self._is_offload_param, optimizer=self._is_offload_optimizer)
        self.mode = 'train'
        self.module.train()
        try:
            yield
        finally:
            self.optimizer_zero_grad()
            self.mode = None
            if automatic:
                self.to('cpu', model=self._is_offload_param, optimizer=self._is_offload_optimizer)

    @contextmanager
    def eval_mode(self, **kwargs):
        automatic = not kwargs.get('disable_auto_offload', False)
        if automatic:
            self.to('cuda', model=self._is_offload_param, optimizer=False)
        self.mode = 'eval'; self.module.eval()
        try:
            yield
        finally:
            self.mode = None
            if automatic:
                self.to('cpu', model=self._is_offload_param, optimizer=False)

    def to(self, device, model=True, optimizer=True, grad=True):
        target = torch.device('cuda', torch.cuda.current_device()) if device == 'cuda' else torch.device('cpu')
        if model:
            plain = self.module.module
            source = next(plain.parameters()).device
            if source != target:
                plain.to(target)
                assert tuple(id(p) for p in plain.parameters()) == self._parameter_ids
                # CPU/CUDA conversion preserves Parameter identities but resets
                # their AccumulateGrad nodes, invalidating the reducer's hooks.
                # Rebuild the actual DDP reducer after loading, without another
                # base-weight broadcast. Replicas were initialized or updated
                # synchronously before the previous offload.
                self._ddp_rebuild_required = True
            if target.type == 'cuda' and self._ddp_rebuild_required:
                previous = self.module
                self.module = None
                del previous
                self.module = self._wrap_ddp(plain, init_sync=False)
                self._ddp_rebuild_required = False
        if optimizer and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != 'step':
                        state[key] = value.to(target)
        if target.type == 'cpu':
            torch.cuda.empty_cache()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)
        assert torch.isfinite(norm), f'Nonfinite DDP gradient norm: {norm}'
        norms = [None] * dist.get_world_size()
        dist.all_gather_object(norms, norm.item())
        assert max(norms) - min(norms) <= max(1e-7, abs(norms[0]) * 1e-6), ('DDP gradients disagree', norms)
        self.optimizer.step()
        return norm.item()

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        assert base_sync_done, 'Frozen base must not be retransmitted or merged with RL adapters'
        params = get_peft_model_state_dict(self.module.module)
        params = convert_weight_keys(params, self.module.module)
        # Stage only the small adapter for CUDA IPC; never reload the frozen base.
        staged = {name:value.detach().to(torch.cuda.current_device()).contiguous() for name,value in params.items()}
        return iter(staged.items()), self.module.module.peft_config['default'].to_dict()

    def disable_adapter(self):
        return self.module.module.disable_adapter()

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert hdfs_path is None
        path = Path(local_path);path.mkdir(parents=True, exist_ok=True)
        params = {k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(self.module.module).items()}
        fingerprint = tensor_digest(params.items())
        digests = [None] * dist.get_world_size();dist.all_gather_object(digests, fingerprint)
        assert len(set(digests)) == 1, 'DDP replicas diverged'
        moment_digest=tensor_digest((f'{i}/{key}',state[key])
            for i,state in enumerate(self.optimizer.state.values()) for key in ('exp_avg','exp_avg_sq'))
        moment_digests=[None]*dist.get_world_size();dist.all_gather_object(moment_digests,moment_digest)
        assert len(set(moment_digests))==1,'DDP Adam replicas diverged'
        rng = {'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(),
               'numpy':np.random.get_state(), 'python':random.getstate()}
        torch.save({'rng':rng, 'lr_scheduler':self.lr_scheduler.state_dict()}, path / f'rank_{self.rank}_rng.pt')
        steps = {int(s['step'].item()) for s in self.optimizer.state.values() if 'step' in s}
        assert steps == {global_step * 4}, steps
        if self.rank == 0:
            save_file(params, str(path / 'adapter_model.safetensors'))
            self.module.module.peft_config['default'].save_pretrained(path)
            # One canonical optimizer copy suffices for replicated DDP. All ranks
            # retain their separate RNG states, without duplicating frozen weights.
            torch.save(self.optimizer.state_dict(), path / 'optimizer.pt')
            (path / 'ddp_config.json').write_text(json.dumps({'backend':'torch_ddp', 'world_size':dist.get_world_size(),
                'optimizer_update':next(iter(steps)), 'global_step':global_step,
                'replica_adapter_sha256':digests,'replica_adam_sha256':moment_digests,
                'base_model':self.model_config.path,
                'base_dtype':'bfloat16', 'adapter_dtype':'float32'}, indent=2) + '\n')
        dist.barrier()
        if os.environ.get('STREETVIEW_CHECKPOINT_ROUNDTRIP')=='1':
            # A real save/load round trip at the end of the bounded benchmark.
            self.load_checkpoint(local_path)
            restored=tensor_digest(get_peft_model_state_dict(self.module.module).items())
            restored_moments=tensor_digest((f'{i}/{key}',state[key])
                for i,state in enumerate(self.optimizer.state.values()) for key in ('exp_avg','exp_avg_sq'))
            assert restored==fingerprint and restored_moments==moment_digest
            from streetview_audit import write
            write('checkpoint_roundtrip.jsonl',{'global_step':global_step,'optimizer_update':next(iter(steps)),
                'adapter_exact':True,'adam_exact':True,'rng_restored':True,'scheduler_restored':True})

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        from peft import set_peft_model_state_dict
        path=Path(local_path)
        set_peft_model_state_dict(self.module.module, load_file(str(path / 'adapter_model.safetensors')))
        self.optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=False))
        rng_path=path / f'rank_{self.rank}_rng.pt'
        added_rank=False
        assert rng_path.exists(), 'All three rank RNG files must exist'
        assert json.loads((path/'ddp_config.json').read_text())['world_size']==3
        extra=torch.load(rng_path,map_location='cpu',weights_only=False)
        self.lr_scheduler.load_state_dict(extra['lr_scheduler'])
        rng=extra['rng'];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        from streetview_audit import write
        actual=tensor_digest(get_peft_model_state_dict(self.module.module).items())
        moments=tensor_digest((f'{i}/{key}',state[key]) for i,state in enumerate(self.optimizer.state.values()) for key in ('exp_avg','exp_avg_sq'))
        cfg=json.loads((path/'ddp_config.json').read_text())
        assert actual==cfg['replica_adapter_sha256'][0] and moments==cfg['replica_adam_sha256'][0]
        assert {int(s['step'].item()) for s in self.optimizer.state.values()}=={cfg['optimizer_update']}
        write('checkpoint_loaded.jsonl',{'path':str(path),'optimizer_update':cfg['optimizer_update'],'adapter_sha256':actual,'adam_sha256':moments,'new_rank_rng_seeded':added_rank,'scheduler':self.lr_scheduler.state_dict()})
        self.to('cpu', model=self._is_offload_param, optimizer=self._is_offload_optimizer)
        dist.barrier()
