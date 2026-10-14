"""Fresh SFT-500 benchmark/formal run of serial shared vLLM + actual DDP."""
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
import ray
from run_three_gpu import configuration as previous_configuration
from serial_runtime import DDPActorRolloutWorker,native_task_run
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import ResourcePoolManager,Role
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.config import validate_config

ROOT=Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN=ROOT/'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1'


class SerialRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self,config):
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(DDPActorRolloutWorker)
        self.mapping[Role.ActorRollout]='global_pool'
        return DDPActorRolloutWorker,RayWorkerGroup

    def add_teacher_model_resource_pool(self,config):
        self.mapping[Role.TeacherModel]='global_pool'

    def init_resource_pool_mgr(self,config):
        return ResourcePoolManager(resource_pool_spec={'global_pool':[4]},mapping=self.mapping)

    def run(self,config):
        return native_task_run()(self,config)


def configuration(variant,mode,cycles=2):
    c=previous_configuration()
    c.trainer.n_gpus_per_node=c.distillation.n_gpus_per_node=4
    c.actor_rollout_ref.rollout.n_gpus_per_node=4
    c.actor_rollout_ref.rollout.agent.num_workers=4
    OmegaConf.update(c,'ray_kwargs.ray_init.num_gpus',4,force_add=True)
    c.actor_rollout_ref.actor.fsdp_config.model_dtype='bfloat16'
    # Native FSDPActorConfig provides the configuration schema only. The custom
    # TrainingWorker selects streetview_ddp and the actual wrapper is torch DDP.
    c.actor_rollout_ref.actor.fsdp_config.param_offload=variant=='ddp_offload'
    c.actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    r=c.actor_rollout_ref.rollout
    r.agent.agent_loop_manager_class='serial_runtime.SerialAgentManager'
    r.max_model_len=9216
    r.gpu_memory_utilization=.45
    r.logprobs_mode='raw_logprobs'
    c.algorithm.rollout_correction.bypass_mode=True
    c.algorithm.rollout_correction.rollout_is=None
    c.algorithm.rollout_correction.rollout_rs=None
    c.algorithm.rollout_correction.loss_type='ppo_clip'
    c.actor_rollout_ref.actor.policy_loss.loss_mode='bypass_mode'
    c.actor_rollout_ref.actor.policy_loss.rollout_correction=OmegaConf.to_container(c.algorithm.rollout_correction)
    c.trainer.default_local_dir=str(RUN/'deployment/serial_ddp4'/variant/mode)
    c.trainer.experiment_name='streetview_serial_'+variant+'_'+mode
    c.ray_kwargs.ray_init._temp_dir='/tmp/sv_serial4_'+('off' if variant=='ddp_offload' else 'res')
    c.trainer.resume_mode='resume_path'
    c.trainer.resume_from_path=str(RUN/'deployment/serial_ddp/ddp_resident/formal/global_step_51')
    c.trainer.del_local_ckpt_after_load=False
    c.trainer.total_training_steps=cycles if mode=='benchmark' else None
    c.trainer.save_freq=cycles if mode=='benchmark' else 17
    OmegaConf.update(c,'serial_ddp',{'actor_backend':'torch_ddp','shared_vllm_base':True,
        'stage_order':['student','reference','opsd'],'variant':variant,
        'canonical_checkpoint_without_base':True},force_add=True)
    OmegaConf.resolve(c)
    validate_config(c,use_reference_policy=True,use_critic=False)
    assert (c.data.train_batch_size,r.n,c.actor_rollout_ref.actor.ppo_mini_batch_size)==(96,8,24)
    assert r.temperature==1 and r.top_p==1 and r.top_k==-1
    return c


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['ddp_offload','ddp_resident'],default='ddp_offload')
    p.add_argument('--mode',choices=['benchmark','formal'],default='benchmark')
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--config-only',action='store_true')
    a=p.parse_args();c=configuration(a.variant,a.mode,a.cycles)
    if a.config_only:
        print('Validated: serial shared vLLM, torch DDP; B96/n8/mini24/micro1; '+a.variant)
    else:
        out=Path(c.trainer.default_local_dir)
        assert not out.exists(),'Review existing outputs before retrying'
        out.mkdir(parents=True)
        OmegaConf.save(c,out/'resolved_config.yaml')
        os.environ['STREETVIEW_AUDIT_DIR']=str(out/'audit')
        os.environ['TENSORBOARD_DIR']=str(out/'tensorboard')
        if a.mode=='benchmark':os.environ['STREETVIEW_CHECKPOINT_ROUNDTRIP']='1'
        from verl.trainer.main_ppo import run_ppo
        try:run_ppo(c,ray.remote(SerialRunner))
        finally:ray.shutdown()
