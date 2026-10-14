"""Compose, validate and launch the auditable native verl training configuration."""
import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import ray
import verl


def config_for(mode):
    directory = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / 'trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer')
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(directory / 'experiment.yaml'))
    if mode == 'pilot':
        config.trainer.total_training_steps = 5
        config.trainer.experiment_name += '_pilot'
        config.trainer.default_local_dir = str(Path(config.trainer.default_local_dir).parent / 'pilot')
        config.trainer.resume_mode = 'disable'
    OmegaConf.resolve(config)
    a = config.actor_rollout_ref
    assert (config.data.train_batch_size, a.rollout.n, a.actor.ppo_mini_batch_size, a.actor.ppo_epochs) == (64, 8, 16, 1)
    assert a.actor.ppo_micro_batch_size_per_gpu == 1 and not a.actor.use_dynamic_bsz
    assert config.distillation.distillation_loss.distillation_loss_coef == 0.01
    assert a.actor.use_kl_loss and a.actor.kl_loss_coef == 0.001
    assert not config.algorithm.use_kl_in_reward
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'formal'], default='pilot')
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    config = config_for(args.mode)
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    output = Path(config.trainer.default_local_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / 'resolved_config.yaml')
    if not args.config_only:
        os.environ['TENSORBOARD_DIR'] = str(output / 'tensorboard')
        from verl.trainer.main_ppo import run_ppo
        from verl.trainer.main_ppo_v0 import TaskRunner
        try:
            run_ppo(config, TaskRunner)
        finally:
            ray.shutdown()
