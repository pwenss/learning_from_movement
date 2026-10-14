"""One vLLM SFT base per GPU: rollout -> reference -> privileged scoring.

All requests in one stage finish before adapter mode changes. The teacher never
sees RL adapters; the student never sees the privileged GT prompt. Temporary
CPU image/request data are released after the final scoring stage.
"""
import asyncio
import __future__
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
import satellite_sft_common as common

from task_runtime import DATA, TEACHER_HASH, SatelliteWorker, align_teacher_response, teacher_tokens
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


def global_real_responses(data, worker):
    worker._metric_local_real_responses = int((data['response_mask'].sum(-1)>0).sum())
    count = torch.tensor(worker._metric_local_real_responses, device='cuda')
    torch.distributed.all_reduce(count, group=worker.engine.get_data_parallel_group())
    assert count.item() > 0
    worker._metric_global_real_responses = count.item()
    return count.item()


def compact_metric(value, mean_weight):
    from verl.utils.metric import Metric, AggregationType
    if isinstance(value, list) and value and isinstance(value[0], Metric):
        merged = value[0].init_list()
        for part in value:
            merged.extend(part)
        value = merged
    if isinstance(value, Metric):
        scalar = value.aggregate()
        if value.aggregation == AggregationType.MEAN:
            scalar *= mean_weight
        return Metric(aggregation=value.aggregation, value=float(scalar))
    return value


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
        engine_values['strategy'] = 'satellite_ddp'
        config = replace(config, engine_config=ddp_engine.SatelliteDDPEngineConfig(**engine_values))
        super().__init__(config)

    def _postprocess_output(self, output, **kwargs):
        if not kwargs['forward_only']:
            # Native Metric.aggregate_dp requires equal value-list lengths.
            # Collapse real micro-batch metrics before its gather, preserving
            # SUM/MIN/MAX and weighting means by real rows, excluding padding.
            weight = (self._metric_local_real_responses * self.engine.get_data_parallel_size()
                      / self._metric_global_real_responses)
            output['metrics'] = {k: compact_metric(v, weight) for k,v in output['metrics'].items()}
        return super()._postprocess_output(output, **kwargs)

    def train_mini_batch(self, data):
        # Preserve native optimization, with a private iterator accepting the
        # Tail uses 203 rows/rank: 64+64+64+11; one global row is zero-loss
        # dispatch padding added only after real rollout and GRPO advantages.
        import inspect, textwrap, math
        from types import SimpleNamespace
        from verl.utils import tensordict_utils as tu
        iterator_source = textwrap.dedent(inspect.getsource(tu.make_iterator))
        line = 'assert tensordict.batch_size[0] % mini_batch_size == 0, f"{tensordict.batch_size[0]} % {mini_batch_size} != 0"'
        assert line in iterator_source
        iterator_source = iterator_source.replace(line, 'assert mini_batch_size > 0')
        iterator_namespace = dict(tu.make_iterator.__globals__)
        exec(compile(iterator_source, '<satellite_tail_iterator>', 'exec', flags=__future__.annotations.compiler_flag), iterator_namespace)
        original = unwrap(TrainingWorker.train_mini_batch)
        source = textwrap.dedent(inspect.getsource(original))
        source = source[source.index('def train_mini_batch'):]
        old = 'data.shape[0] // mini_batch_size_per_gpu * epochs'
        assert old in source
        source = source.replace(old, 'math.ceil(data.shape[0] / mini_batch_size_per_gpu) * epochs')
        needle = 'global_token_num=NonTensorData(global_token_num),'
        assert needle in source
        source = source.replace(needle, needle + '\n                    global_batch_size=global_real_responses(mini_batch_td, self),')
        namespace = {**original.__globals__, 'math': math,
                     'global_real_responses': global_real_responses,
                     'tu': SimpleNamespace(**{**vars(tu), 'make_iterator': iterator_namespace['make_iterator']})}
        exec(compile(source, '<satellite_tail_native_train>', 'exec', flags=__future__.annotations.compiler_flag), namespace)
        return namespace['train_mini_batch'](self, data)


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


class SerialWorker(SatelliteWorker):
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
        assert self.retained_inputs is None
        if validate:
            return super()._postprocess(inputs,input_non_tensor_batch,validate)
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
        if prompts.meta_info.get('validate', False):
            write('probe_stages.jsonl', {'cycle':step,'responses':len(output),
                'seconds':student_seconds,'servers':identities,
                'student_only':True,'gt_teacher_scoring_invoked':False})
            return output
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
    def _balance_batch(self, batch, metrics, *args, **kwargs):
        if len(batch) % 3:
            assert len(batch) == 608
            # Native equal-cardinality token balancing requires divisibility.
            # Keep all real rows; dispatch padding happens after advantages.
            metrics['satellite/tail_balance_skipped'] = 1
            return
        return super()._balance_batch(batch, metrics, *args, **kwargs)

    def _update_actor(self, batch):
        if len(batch) % 3:
            assert len(batch) == 608
            padding = copy.deepcopy(batch.select_idxs([0]))
            padding.batch['response_mask'].zero_()
            padding.batch['advantages'].zero_()
            if 'rollout_is_weights' in padding.batch:
                padding.batch['rollout_is_weights'].zero_()
            batch = DataProto.concat([batch, padding])
            assert len(batch) == 609
            from streetview_audit import write
            write('tail_dispatch_padding.jsonl', {'cycle': self.global_steps,
                'real_responses': 608, 'dispatch_rows': 609,
                'padding_response_mask_sum': int(padding.batch['response_mask'].sum()),
                'added_generation_requests': 0, 'added_trajectory_ids': 0})
        return super()._update_actor(batch)

    def _create_dataloader(self, *args, **kwargs):
        import inspect, textwrap
        original = RayPPOTrainer._create_dataloader
        source = textwrap.dedent(inspect.getsource(original))
        assert source.count('drop_last=True') == 1
        source = source.replace('drop_last=True', 'drop_last=False')
        namespace = dict(original.__globals__)
        exec(compile(source, '<satellite_all_train_loader>', 'exec', flags=__future__.annotations.compiler_flag), namespace)
        result = namespace['_create_dataloader'](self, *args, **kwargs)
        assert len(self.train_dataset) == 17452 and len(self.val_dataset) == 256
        assert len(self.train_dataloader) == (230 if self.config.satellite_tail_preflight else 182)
        return result

    def _validate(self, merged=False):
        import inspect, textwrap
        original = RayPPOTrainer._validate
        source = textwrap.dedent(inspect.getsource(original))
        self._probe_token_ids = []
        needle = 'sample_outputs.extend(output_texts)'
        assert needle in source
        source = source.replace(needle, needle + '''
        self._probe_token_ids.extend([ids[:int(mask.sum())].cpu().tolist()
            for ids, mask in zip(output_ids, test_output_gen_batch.batch['response_mask'], strict=True)])''')
        needle = 'self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)'
        assert needle in source
        source = source.replace(needle, 'self._probe_ids = list(sample_uids)\n    ' + needle)
        namespace = dict(original.__globals__)
        exec(compile(source, '<satellite_probe_recording>', 'exec', flags=__future__.annotations.compiler_flag), namespace)
        return namespace['_validate'](self, merged=merged)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        # Synchronous, read-back-scored recording: finish the complete probe
        # before the next training cycle may start.
        records = {r['id']: r for r in common.read_jsonl(DATA / 'validation_probe.jsonl')}
        assert len(outputs) == len(set(self._probe_ids)) == len(records) == 256
        assert set(self._probe_ids) == set(records)
        prompt = json.loads((DATA / 'prompt.json').read_text())
        rows = []
        for uid, text, gt, score, ids in zip(self._probe_ids, outputs, gts, scores, self._probe_token_ids, strict=True):
            record = records[uid]
            assert json.loads(gt) == record['ground_truth']
            clipped = len(ids) >= 3072 and ids[-1] != 151645
            result = common.score(text, record['ground_truth'], prompt['step_titles'], clipped)
            assert float(result['exact_match']) == float(score)
            rows.append({'id':uid,'city':record['city'],'ground_truth':record['ground_truth'],
                         'text':text,'output_token_ids':ids,'output_tokens':len(ids),**result})
        directory = Path(dump_path) / f'cycle_{self.global_steps:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        predictions = directory / 'predictions.jsonl'
        assert not predictions.exists(), 'Refuse to overwrite an existing probe'
        predictions.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        reread = common.read_jsonl(predictions)
        assert reread == rows
        summary = common.summarize(reread)
        common.atomic_json(directory / 'summary.json', summary)
        checkpoint = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
        report = {'status':'passed','cycle':self.global_steps,'n':256,
                  'protocol':{'engine':'vllm','gpus':[0,2,3],'tp':1,'temperature':0,
                              'n':1,'max_response_tokens':3072,'stop_token_ids':[151645],
                              'privileged_gt_in_student_prompt':False},
                  'prompt_sha256':common.hash_file(DATA / 'prompt.json'),
                  'probe_sha256':common.hash_file(DATA / 'validation_probe.jsonl'),
                  'predictions_sha256':common.hash_file(predictions),
                  'raw_answers_rescored':True,'weight_sync_before_probe':True,
                  'summary':summary,'time':time.time()}
        if self.global_steps:
            cfg = json.loads((checkpoint / 'actor/ddp_config.json').read_text())
            assert cfg['optimizer_update'] == self.global_steps * 4 and cfg['world_size'] == 3
            assert len(set(cfg['replica_adapter_sha256'])) == len(set(cfg['replica_adam_sha256'])) == 1
            assert all((checkpoint / f'actor/rank_{rank}_rng.pt').is_file() for rank in range(3))
            assert (checkpoint / 'data.pt').is_file()
            report['checkpoint_adapter_sha256'] = common.hash_file(checkpoint / 'actor/adapter_model.safetensors')
            report['optimizer_update'] = cfg['optimizer_update']
        common.atomic_json(directory / 'report.json', report)
        common.atomic_json(Path(dump_path) / 'latest.json', report)
        print(f"Satellite probe cycle {self.global_steps}: EM={summary['overall']['exact_match']:.4%}; 256 IDs audited", flush=True)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        # Native fit saves before syncing vLLM. Explicitly sync here to ensure
        # this probe evaluates the just-saved weights, never the old rollout.
        self.checkpoint_manager.update_weights(self.global_steps)
        metrics = self._validate()
        from streetview_audit import write
        write('checkpoint_probe.jsonl', {'cycle':self.global_steps,'metrics':metrics,
                                      'checkpoint_saved_before_probe':True,'status':'passed'})

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
