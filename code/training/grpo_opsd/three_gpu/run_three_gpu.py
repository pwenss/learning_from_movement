"""Fresh SFT initialization with three colocated FSDP/vLLM ranks."""
import argparse
import os
from pathlib import Path

import ray
from omegaconf import OmegaConf
from run_deployment import SharedPoolRunner, deployment_config
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils.config import validate_config

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
DIRECTORY = ROOT / 'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1/deployment/dp3_full/formal'


class ThreePoolRunner(SharedPoolRunner):
    def init_resource_pool_mgr(self, config):
        assert config.trainer.nnodes == config.distillation.nnodes == 1
        assert config.trainer.n_gpus_per_node == config.distillation.n_gpus_per_node == 3
        assert not config.reward.reward_model.enable
        return ResourcePoolManager(resource_pool_spec={'global_pool': [3]}, mapping=self.mapping)


def configuration():
    c = deployment_config('dp_full', 'formal', 2)
    c.trainer.n_gpus_per_node = c.distillation.n_gpus_per_node = 3
    OmegaConf.update(c, 'ray_kwargs.ray_init.num_gpus', 3, force_add=True)
    c.data.train_batch_size = 96
    c.actor_rollout_ref.actor.ppo_mini_batch_size = 24
    c.actor_rollout_ref.rollout.n_gpus_per_node = 3
    c.actor_rollout_ref.rollout.agent.num_workers = 3
    c.actor_rollout_ref.rollout.agent.agent_loop_manager_class = 'three_gpu_runtime.ThreeGPUManager'
    c.trainer.save_freq = 17
    c.trainer.total_epochs = 1
    c.trainer.total_training_steps = None
    c.trainer.resume_mode = 'disable'
    c.trainer.default_local_dir = str(DIRECTORY)
    c.trainer.experiment_name = 'streetview_grpo_opsd_dp3_full_formal'
    c.ray_kwargs.ray_init._temp_dir = '/tmp/sv_rl_dp3'
    OmegaConf.resolve(c)
    validate_config(c, use_reference_policy=True, use_critic=False)
    assert c.actor_rollout_ref.rollout.n == 8
    assert c.actor_rollout_ref.actor.ppo_epochs == 1
    assert c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert c.actor_rollout_ref.actor.fsdp_config.reshard_after_forward
    assert c.distillation.distillation_loss.distillation_loss_coef == .01
    assert c.actor_rollout_ref.actor.kl_loss_coef == .001
    assert 16704 % c.data.train_batch_size == 0
    assert 16704 // c.data.train_batch_size == 174
    assert c.data.train_batch_size * 8 // 3 == 256
    assert c.actor_rollout_ref.actor.ppo_mini_batch_size * 8 // 3 == 64
    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    config = configuration()
    if args.config_only:
        print('Validated: GPUs 2,3,1; B=96 n=8 mini=24; 174 cycles / 696 updates; fresh SFT')
    else:
        assert os.environ.get('STREETVIEW_DEPLOYMENT_WORLD_SIZE') == '3'
        assert not DIRECTORY.exists(), 'Review existing output before restarting'
        DIRECTORY.mkdir(parents=True)
        OmegaConf.save(config, DIRECTORY / 'resolved_config.yaml')
        os.environ['STREETVIEW_AUDIT_DIR'] = str(DIRECTORY / 'audit')
        os.environ['TENSORBOARD_DIR'] = str(DIRECTORY / 'tensorboard')
        from verl.trainer.main_ppo import run_ppo
        run_ppo(config, ray.remote(ThreePoolRunner))
