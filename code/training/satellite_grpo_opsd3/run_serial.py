"""Fresh Satellite SFT-700 initialization, three GPU DDP and post-save probe."""
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import ray
from serial_runtime import DDPActorRolloutWorker, native_task_run
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.config import validate_config
import satellite_sft_common as common

ROOT = common.ROOT
RUN = ROOT / 'runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023'
DATA = ROOT / 'training/satellite_grpo_opsd_qwen3vl8b_lora_v1'
CODE = Path(__file__).resolve().parent


class SerialRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self, config):
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(DDPActorRolloutWorker)
        self.mapping[Role.ActorRollout] = 'global_pool'
        return DDPActorRolloutWorker, RayWorkerGroup

    def add_teacher_model_resource_pool(self, config):
        self.mapping[Role.TeacherModel] = 'global_pool'

    def init_resource_pool_mgr(self, config):
        return ResourcePoolManager(resource_pool_spec={'global_pool': [3]}, mapping=self.mapping)

    def run(self, config):
        return native_task_run()(self, config)


def configuration(mode):
    c = OmegaConf.load(CODE / 'experiment.yaml')
    out = RUN / 'deployment/ddp_resident' / mode
    c.trainer.default_local_dir = str(out)
    c.trainer.validation_data_dir = str(out / 'probe_evaluations')
    c.trainer.experiment_name = 'satellite_ddp_resident_' + mode
    c.ray_kwargs.ray_init._temp_dir = '/tmp/satellite_rl3_' + mode
    OmegaConf.update(c, 'satellite_tail_preflight', mode == 'tail_preflight', force_add=True)
    if mode == 'tail_preflight':
        # Exercise the real 76-trajectory tail geometry including all four
        # optimizer updates, a real checkpoint round trip and post-save probe.
        c.data.gen_batch_size = 76
        c.trainer.total_training_steps = 1
        c.trainer.save_freq = 1
    else:
        assert mode == 'formal'
        c.data.gen_batch_size = None
        c.trainer.total_training_steps = None
        c.trainer.save_freq = 17
    OmegaConf.resolve(c)
    validate_config(c, use_reference_policy=True, use_critic=False)
    a = c.actor_rollout_ref.actor
    r = c.actor_rollout_ref.rollout
    assert c.ray_kwargs.ray_init.num_gpus == c.trainer.n_gpus_per_node == c.distillation.n_gpus_per_node == 3
    assert c.actor_rollout_ref.rollout.n_gpus_per_node == 3
    assert (c.data.train_batch_size, r.n, a.ppo_mini_batch_size, a.ppo_epochs,
            a.ppo_micro_batch_size_per_gpu) == (96, 8, 24, 1, 1)
    assert a.optim.lr == 1e-6 and a.kl_loss_coef == .001
    assert c.distillation.distillation_loss.distillation_loss_coef == .01
    assert r.temperature == 1 and r.top_p == 1 and r.top_k == -1 and r.logprobs_mode == 'raw_logprobs'
    assert c.trainer.resume_mode == 'disable' and c.trainer.resume_from_path is None
    assert c.actor_rollout_ref.model.path == c.distillation.teacher_models.teacher_model.model_path
    assert 'satellite_sft_checkpoint700' in c.actor_rollout_ref.model.path
    return c


def preflight_inputs(c):
    data = json.loads((DATA / 'data_verification.json').read_text())
    assert data['status'] == 'headers_and_splits_verified' and data['train_count'] == 17452
    assert common.hash_file(DATA / 'train.jsonl') == data['train_sha256']
    merge = json.loads((Path(c.actor_rollout_ref.model.path) / 'merge_verification.json').read_text())
    assert merge['status'] == 'weights_verified' and merge['adapted_tensors'] == 252
    prompt = json.loads((DATA / 'prompt.json').read_text())
    processor = common.processor()
    records = common.read_jsonl(DATA / 'train.jsonl')
    from task_runtime import teacher_tokens
    checks = []
    for record in records[::max(1, len(records)//8)]:
        student = common.inputs(processor, record, prompt, False)
        assert student['image_grid_thw'].tolist() == [[1,44,44]] * 8
        assert student['input_ids'][0].tolist().count(151655) == 3872
        raw = common.messages(record, prompt, False)
        ids = teacher_tokens(processor, raw, record['ground_truth'], (DATA / 'teacher_prompt.txt').read_text())
        assert len(ids) + 3072 + 1 <= 9216
        checks.append({'id':record['id'],'student_tokens':student['input_ids'].shape[1],
                       'teacher_tokens':len(ids),'image_grid':[1,44,44]})
        del student
    common.atomic_json(RUN / 'provision/processor_preflight.json', {'status':'passed','records':checks,
        'visual_tokens':3872,'max_pixels':524288,'prompt_sha256':common.hash_file(DATA / 'prompt.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tail_preflight','formal'], default='tail_preflight')
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    config = configuration(args.mode)
    if args.config_only:
        print('Satellite configuration validated: B96/n8/mini24/epochs1/micro1; GPUs0,2,3; SFT700 fresh.')
    else:
        out = Path(config.trainer.default_local_dir)
        assert not out.exists(), 'Refuse to overwrite an existing run'
        preflight_inputs(config)
        out.mkdir(parents=True)
        OmegaConf.save(config, out / 'resolved_config.yaml')
        os.environ['STREETVIEW_AUDIT_DIR'] = str(out / 'audit')
        os.environ['TENSORBOARD_DIR'] = str(out / 'tensorboard')
        if args.mode == 'tail_preflight':
            os.environ['STREETVIEW_CHECKPOINT_ROUNDTRIP'] = '1'
        from verl.trainer.main_ppo import run_ppo
        try:
            run_ppo(config, ray.remote(SerialRunner))
        finally:
            ray.shutdown()
