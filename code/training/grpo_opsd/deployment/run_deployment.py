"""Shared-two-GPU native verl recipe; independent benchmark/formal outputs."""
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
import ray

from run_training import config_for
from verl.trainer.main_ppo_v0 import BaseTaskRunner, TaskRunner
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.config import validate_config


class SharedPoolRunner(BaseTaskRunner):
    def add_teacher_model_resource_pool(self, config):
        self.mapping[Role.TeacherModel] = 'global_pool'

    def init_resource_pool_mgr(self, config):
        assert config.trainer.nnodes == config.distillation.nnodes == 1
        assert config.trainer.n_gpus_per_node == config.distillation.n_gpus_per_node == 2
        assert not config.reward.reward_model.enable
        return ResourcePoolManager(resource_pool_spec={'global_pool': [2]}, mapping=self.mapping)

    # Reuse the pinned implementation, including native datasets/loss/checkpoints.
    run = TaskRunner.__ray_metadata__.modified_class.run


def native_run_with_staged_teacher():
    from inspect import unwrap
    from types import FunctionType
    import verl.trainer.main_ppo_v0 as native
    from staged_teacher import StagedTeacherTrainer
    original = unwrap(native.TaskRunner.__ray_metadata__.modified_class.run)
    # Ray's tracing wrapper closes over the original method. Replacing globals
    # on that wrapper leaves its enclosed native trainer constructor unchanged.
    assert 'RayPPOTrainer' in original.__code__.co_names
    assert not original.__code__.co_freevars
    # Preserve the pinned TaskRunner code and all of its globals except the
    # trainer constructor. No mutation of native module globals is needed.
    namespace = {**original.__globals__, 'RayPPOTrainer': StagedTeacherTrainer}
    execute = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    execute.__kwdefaults__ = original.__kwdefaults__
    return execute


class StagedSharedPoolRunner(SharedPoolRunner):
    def run(self, config):
        return native_run_with_staged_teacher()(self, config)


def deployment_config(variant, mode, cycles):
    c = config_for('formal')
    c.trainer.n_gpus_per_node = c.distillation.n_gpus_per_node = 2
    c.actor_rollout_ref.actor.fsdp_config.reshard_after_forward = variant not in ('dp_zero2', 'dp_zero2_staged')
    tp = 2 if variant in ('tp_full', 'tp_full_nccl', 'tp_full_nccl_local', 'tp_local') else 1
    c.actor_rollout_ref.rollout.tensor_model_parallel_size = tp
    teacher = c.distillation.teacher_models.teacher_model.inference
    teacher.tensor_model_parallel_size = tp
    if tp == 2:
        # Match DP2's total admission limits: one TP2 engine replaces two TP1
        # replicas. Keep prompt/response caps and GPU memory budgets unchanged.
        c.actor_rollout_ref.rollout.max_num_seqs = 32
        teacher.max_num_seqs = 16
    if variant in ('tp_full_nccl_local', 'tp_local'):
        # The entire inference replica is on the same physical two-GPU node.
        # This must agree with the shared trainer pool; leaving the inherited
        # single-GPU value creates two fake nodes and mismatched IPC ranks.
        c.actor_rollout_ref.rollout.n_gpus_per_node = 2
    if variant in ('tp_full_nccl', 'tp_full_nccl_local', 'tp_local'):
        # external_launcher gives each worker one visible GPU. The pinned
        # custom-all-reduce P2P probe incorrectly indexes two visible devices;
        # native NCCL collectives support this layout without that probe.
        for inference in (c.actor_rollout_ref.rollout, teacher):
            OmegaConf.update(inference, 'engine_kwargs.vllm.disable_custom_all_reduce',
                             True, force_add=True)
    # Two colocated BF16 engines; preserve other users' allocation on GPU 1.
    teacher.gpu_memory_utilization = 0.30
    c.actor_rollout_ref.rollout.agent.agent_loop_manager_class = 'deployment_runtime.DeploymentManager'
    c.trainer.experiment_name += '_' + variant + '_' + mode
    original = Path(c.trainer.default_local_dir).parent
    c.trainer.default_local_dir = str(original / 'deployment' / variant / mode)
    c.trainer.resume_mode = 'disable' if mode == 'benchmark' else 'auto'
    if mode == 'benchmark':
        c.trainer.total_training_steps = cycles
        c.trainer.save_freq = cycles
    # Ray appends a session timestamp, PID and socket name. Keep all candidate
    # directories short enough for the Linux AF_UNIX 107-byte path limit.
    c.ray_kwargs.ray_init._temp_dir = '/tmp/sv_rl_' + variant
    OmegaConf.resolve(c)
    validate_config(c, use_reference_policy=True, use_critic=False)
    return c


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['dp_full', 'dp_zero2', 'tp_full', 'tp_full_nccl', 'tp_full_nccl_local', 'tp_local', 'dp_zero2_staged'], required=True)
    p.add_argument('--mode', choices=['benchmark', 'formal'], default='benchmark')
    p.add_argument('--cycles', type=int, default=2)
    p.add_argument('--config-only', action='store_true')
    a = p.parse_args()
    c = deployment_config(a.variant, a.mode, a.cycles)
    out = Path(c.trainer.default_local_dir)
    if a.config_only:
        print('Validated shared pool: 2 GPUs; variant=' + a.variant, flush=True)
        raise SystemExit(0)
    assert os.environ.get('STREETVIEW_DEPLOYMENT_WORLD_SIZE') == '2'
    assert not (out / 'resolved_config.yaml').exists(), 'Inspect previous attempt before reusing output'
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(c, out / 'resolved_config.yaml')
    if not a.config_only:
        os.environ['STREETVIEW_AUDIT_DIR'] = str(out / 'audit')
        os.environ['TENSORBOARD_DIR'] = str(out / 'tensorboard')
        from verl.trainer.main_ppo import run_ppo
        try:
            runner = StagedSharedPoolRunner if a.variant == 'dp_zero2_staged' else SharedPoolRunner
            run_ppo(c, ray.remote(runner))
        finally:
            ray.shutdown()
