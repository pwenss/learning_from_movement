"""One vLLM SFT base per GPU: rollout -> reference -> privileged scoring.

All requests in one stage finish before adapter mode changes. The teacher never
sees RL adapters; the student never sees the privileged GT prompt. Temporary
CPU image/request data are released after the final scoring stage.
"""
import asyncio
import copy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType
from inspect import unwrap
from uuid import uuid4

import ray
import torch
from tensordict import TensorDict
import streetview_sft_common as common

from streetview_runtime import DATA, TEACHER_HASH, StreetViewWorker, align_teacher_response, teacher_tokens
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs
from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class SharedPolicyServer(vLLMHttpServer):
    @property
    def lora_as_adapter(self):
        return getattr(self, '_policy_mode', 'student') == 'student' and super().lora_as_adapter

    async def set_policy_mode(self, mode):
        assert mode in ('student','frozen_sft')
        if mode == 'student':
            from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
            assert VLLM_LORA_INT_ID in await self.engine.list_loras(), 'Student adapter is not loaded'
        self._policy_mode = mode
        await self.engine.reset_prefix_cache(reset_connector=True)
        return {'pid':os.getpid(),'replica_rank':self.replica_rank,'mode':mode,
                'shared_base_model':self.model_config.path}

    def _resolve_sleep_level(self):
        # The shared frozen base is immutable, so it must survive all sleeps.
        return 1


class SharedPolicyReplica(vLLMReplica):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.server_class = ray.remote(SharedPolicyServer)


class SharedServerManager(LLMServerManager):
    def __init__(self,*args,**kwargs):
        self.rollout_replica_class = SharedPolicyReplica
        super().__init__(*args,**kwargs)


class DDPTrainingWorker(TrainingWorker):
    def __init__(self,config):
        import ddp_engine  # Register the actual torch DDP implementation.
        engine_values = {field.name: getattr(config.engine_config, field.name)
                         for field in fields(config.engine_config) if field.init}
        engine_values['strategy'] = 'streetview_ddp'
        config = replace(config, engine_config=ddp_engine.StreetViewDDPEngineConfig(**engine_values))
        super().__init__(config)


class DDPActorRolloutWorker(ActorRolloutRefWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.actor_worker_cls = DDPTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # Both runtimes loaded this identical immutable BF16 SFT checkpoint.
        # Synchronize only new RL adapters, never transmit another base copy.
        assert self.config.model.path == self.distillation_config['teacher_models']['teacher_model']['model_path']
        self.base_sync_done = True


class NoSeparateTeacher:
    def __init__(self,config,**kwargs):
        # Native dataclass conversion normalizes the single teacher's key.
        self.routing_keys = tuple(omega_conf_to_dataclass(config.distillation).teacher_models)

    def get_client(self):
        # Retain native distillation configuration/loss wiring, allocate no GPU
        # pool, teacher model, teacher server or extra frozen weights.
        return {key:None for key in self.routing_keys}


class SerialWorker(StreetViewWorker):
    def __init__(self,*args,**kwargs):
        AgentLoopWorker.__init__(self,*args,**kwargs)
        self.template=(DATA/'teacher_prompt.txt').read_text()
        assert hashlib.sha256(self.template.encode()).hexdigest()==TEACHER_HASH
        self.pending=asyncio.Semaphore(2*self.rollout_config.max_num_seqs)
        self.retained_inputs=None

    async def _compute_teacher_logprobs(self,*args,**kwargs):
        # No teacher request is issued while rollout is running.
        return

    def _postprocess(self,inputs,input_non_tensor_batch=None,validate=False):
        assert not validate and self.retained_inputs is None
        self.retained_inputs=inputs
        self.retained_extra=input_non_tensor_batch['extra_info']
        return super()._postprocess(inputs,input_non_tensor_batch,validate)

    async def score_stage(self,stage):
        assert stage in ('reference','opsd') and self.retained_inputs is not None
        async def score(output,extra):
            async with self.pending:
                prompt_ids=output.prompt_ids[0][output.attention_mask[0,:output.prompt_ids.shape[1]].bool()].tolist()
                length=int(output.response_mask.sum())
                response_ids=output.response_ids[0,:length].tolist()
                rendered = self.processor.apply_chat_template(output.extra_fields['raw_prompt'],
                    tokenize=False, add_generation_prompt=True, add_vision_id=False)
                assert common.text_ids(self.processor, rendered) == prompt_ids, 'Student prompt differs from frozen protocol'
                prompt=(prompt_ids if stage=='reference' else teacher_tokens(
                    self.processor,output.extra_fields['raw_prompt'],[int(v) for v in extra['ground_truth']],self.template))
                assert len(prompt) + length + 1 <= 9216
                multi=output.multi_modal_data or {}
                scored=await self.llm_client.generate(request_id=uuid4().hex,
                    prompt_ids=prompt+response_ids,
                    sampling_params={'max_tokens':1,'temperature':1.0,'top_p':1.0,'top_k':-1,
                        'prompt_logprobs':0,'ignore_eos':True,'detokenize':False},
                    image_data=multi.get('images'),video_data=multi.get('videos'),
                    mm_processor_kwargs=output.mm_processor_kwargs)
                ids=torch.tensor(scored.extra_fields['prompt_ids'],dtype=torch.int32)
                probabilities=torch.tensor(scored.extra_fields['prompt_logprobs'],dtype=torch.float32)
                native_ids,native_scores=align_teacher_response(ids,probabilities,len(prompt),prompt_ids,response_ids)
                if stage=='reference':
                    scores=native_scores[len(prompt_ids)-1:len(prompt_ids)-1+length,0]
                    padded=torch.zeros(1,self.rollout_config.response_length)
                    padded[0,:length]=scores
                    return padded
                return _pad_teacher_outputs(native_ids,native_scores,
                    output.prompt_ids.shape[1],output.response_ids.shape[1],
                    len(prompt_ids),length,self.tokenizer.pad_token_id)
        results=await asyncio.gather(*[score(o,e) for o,e in zip(self.retained_inputs,self.retained_extra,strict=True)])
        if stage=='reference':
            return DataProto(batch=TensorDict({'ref_log_prob':torch.cat(results)},batch_size=len(results)))
        teacher_ids=torch.cat([r[0] for r in results]);teacher_logprobs=torch.cat([r[1] for r in results])
        self.retained_inputs=self.retained_extra=None
        return DataProto(batch=TensorDict({'teacher_ids':teacher_ids,'teacher_logprobs':teacher_logprobs},
                                         batch_size=len(results)))


class SerialAgentManager(AgentLoopManager):
    def __init__(self,*args,**kwargs):
        self.agent_loop_workers_class=ray.remote(SerialWorker)
        super().__init__(*args,**kwargs)

    @auto_await
    async def score_stage(self,stage):
        return DataProto.concat(await asyncio.gather(*[w.score_stage.remote(stage) for w in self.agent_loop_workers]))


class SerialStages:
    def __init__(self,manager,servers,directory):
        assert len(servers)==3
        self.manager=manager;self.servers=servers;self.directory=Path(directory)

    def __getattr__(self,name):
        return getattr(self.manager,name)

    @auto_await
    async def generate_sequences(self,prompts):
        from streetview_audit import write
        step=int(prompts.meta_info['global_steps'])
        begin=time.monotonic();identities=await asyncio.gather(*[s.set_policy_mode.remote('student') for s in self.servers])
        output=await self.manager.generate_sequences(prompts)
        student_seconds=time.monotonic()-begin
        assert 'rollout_log_probs' in output.batch
        assert torch.isfinite(output.batch['rollout_log_probs'][output.batch['response_mask'].bool()]).all()
        switch=time.monotonic()
        frozen=await asyncio.gather(*[s.set_policy_mode.remote('frozen_sft') for s in self.servers])
        assert [r['pid'] for r in identities]==[r['pid'] for r in frozen]
        switch_seconds=time.monotonic()-switch
        start=time.monotonic();reference=await self.manager.score_stage('reference')
        reference_seconds=time.monotonic()-start
        start=time.monotonic();teacher=await self.manager.score_stage('opsd')
        teacher_seconds=time.monotonic()-start
        output=output.union(reference).union(teacher)
        # Restore adapter mode before native sleep/weight synchronization.
        await asyncio.gather(*[s.set_policy_mode.remote('student') for s in self.servers])
        timing=output.meta_info.setdefault('timing',{})
        timing.update(serial_student=student_seconds,serial_reference=reference_seconds,
                      serial_opsd=teacher_seconds,serial_switch=switch_seconds)
        mask=output.batch['response_mask'].bool()
        difference=(output.batch['rollout_log_probs']-output.batch['ref_log_prob']).abs()[mask]
        write('serial_stages.jsonl',{'rollout_step':step,'responses':len(output),
            'student_seconds':student_seconds,'reference_seconds':reference_seconds,'opsd_seconds':teacher_seconds,
            'switch_seconds':switch_seconds,'student_servers':identities,'frozen_servers':frozen,
            'shared_server_pids':True,'old_reference_mean_difference':difference.mean().item(),
            'old_reference_max_difference':difference.max().item(),
            'stage_order':['student','reference','opsd'],'scores_finite':True})
        return output


class SerialTrainer(RayPPOTrainer):
    def init_workers(self):
        # Clone the pinned initialization with only the server factories replaced.
        # Keep the native GRPO/OPSD loss, dataset, optimizer loop and checkpoint scheduler.
        original=RayPPOTrainer.init_workers
        namespace={**original.__globals__,'LLMServerManager':SharedServerManager}
        # The native function locally imports its teacher factory. Substitute the
        # import in this private clone, leaving the pinned checkout untouched.
        import inspect,textwrap
        source=textwrap.dedent(inspect.getsource(original))
        source=source.replace('from verl.experimental.teacher_loop import MultiTeacherModelManager',
                              'MultiTeacherModelManager = NoSeparateTeacher')
        namespace['NoSeparateTeacher']=NoSeparateTeacher
        exec(compile(source,'<serial_ddp_init_workers>','exec'),namespace)
        namespace['init_workers'](self)
        self.async_rollout_manager=SerialStages(self.async_rollout_manager,
            self.llm_server_manager.server_handles,self.config.trainer.default_local_dir)

    def _compute_ref_log_prob(self,batch):
        assert 'ref_log_prob' in batch.batch
        return batch.select(batch_keys=['ref_log_prob'])


def native_task_run():
    from verl.trainer.main_ppo_v0 import TaskRunner
    original=unwrap(TaskRunner.__ray_metadata__.modified_class.run)
    assert not original.__code__.co_freevars
    execute=FunctionType(original.__code__,{**original.__globals__,'RayPPOTrainer':SerialTrainer},
                         original.__name__,original.__defaults__)
    execute.__kwdefaults__=original.__kwdefaults__
    return execute
