"""Task adapters for native verl GRPO + PG-OPD; no custom optimization loss."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import ray
import torch
from datasets import Dataset

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset

import streetview_sft_common as common

ROOT = common.ROOT
DATA = ROOT / 'training/streetview_grpo_opsd_qwen3vl8b_lora_v1'
TEACHER_HASH = '6c19e1c06dc8b07953dc0a5bd9d8ff4e43fa94f9a25cc0d5252e309d1b75fd92'


class StreetViewDataset(RLHFDataset):
    """Use every frozen RL record and preserve the SFT image/text ordering."""

    def _read_files_and_tokenize(self):
        prompt = json.loads((DATA / 'prompt.json').read_text())
        content = prompt['user'] + '\n' + ''.join(f'Image {k}:\n<image>' for k in range(8))
        rows = []
        for filename in self.data_files:
            for index, record in enumerate(common.read_jsonl(filename)):
                rows.append({
                    'data_source': 'streetview',
                    'prompt': [{'role': 'system', 'content': prompt['system']},
                               {'role': 'user', 'content': content}],
                    'images': [str(ROOT / 'dataset/streetview' / p) for p in record['images']],
                    'reward_model': {'style': 'rule', 'ground_truth': json.dumps(record['ground_truth'])},
                    'extra_info': {'index': index, 'id': record['id'], 'city': record['city'],
                                   'ground_truth': record['ground_truth']},
                })
        if self.max_samples > 0:
            # Only explicit pilot/probe subsets; never silently remove an overlong record.
            rows = rows[:self.max_samples]
        self.dataframe = Dataset.from_list(rows)
        print(f'StreetViewDataset: {len(rows)} frozen records; filtering disabled', flush=True)


class StreetViewAgent(SingleTurnAgentLoop):
    async def run(self, sampling_params, priority=0, **kwargs):
        sampling_params = dict(sampling_params)
        # Existing SFT validation uses only im_end; ignore other model-config EOS IDs.
        sampling_params.update(ignore_eos=True, stop_token_ids=[151645])
        return await super().run(sampling_params, priority=priority, **kwargs)


def teacher_tokens(processor, raw_prompt, gt, template):
    assert len(gt) == 8 and sorted(gt) == list(range(8))
    messages = copy.deepcopy(raw_prompt)
    messages[-1]['content'].append({'type': 'text', 'text': '\n\n' + template.format(GT_ORDER=json.dumps(gt))})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
    ids = common.text_ids(processor, text)
    assert len(ids) <= 5376, len(ids)
    assert ids.count(151655) == 4096
    return ids


def align_teacher_response(ids, logprobs, teacher_prompt_length, student_prompt_ids, response_ids):
    """verl's vLLM adapter returns scores shifted to the predicting logit position."""
    assert ids.shape == logprobs.shape == (teacher_prompt_length + len(response_ids), 1)
    start = teacher_prompt_length - 1
    response_scores = logprobs[start:start + len(response_ids)]
    response_tokens = ids[start:start + len(response_ids)]
    assert response_tokens[:, 0].tolist() == response_ids, 'Teacher next-token alignment mismatch'
    assert torch.isfinite(response_scores).all(), 'Nonfinite teacher response logprobs'
    # Native trainer pads and later causal-shifts the full sequence. Prompt scores
    # are masked; substitute harmless zeros without exposing privileged tokens.
    native_ids = torch.tensor((student_prompt_ids + response_ids)[1:] + [0], dtype=ids.dtype).unsqueeze(-1)
    native_scores = torch.cat([torch.zeros((len(student_prompt_ids) - 1, 1), dtype=logprobs.dtype),
                               response_scores, torch.zeros((1, 1), dtype=logprobs.dtype)])
    return native_ids, native_scores


class StreetViewWorker(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = (DATA / 'teacher_prompt.txt').read_text(encoding='utf-8')
        assert hashlib.sha256(self.template.encode()).hexdigest() == TEACHER_HASH
        rollout = self.config.actor_rollout_ref.rollout
        workers = rollout.agent.num_workers
        assert rollout.max_num_seqs >= workers and rollout.max_num_seqs % workers == 0
        self.pending = asyncio.Semaphore(rollout.max_num_seqs // workers)

    async def generate_sequences(self, batch):
        self.rollout_step = batch.meta_info.get('global_steps')
        return await super().generate_sequences(batch)

    async def _run_agent_loop(self, *args, **kwargs):
        # Bound decoded image copies and requests across each worker's coroutines.
        async with self.pending:
            return await super()._run_agent_loop(*args, **kwargs)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        if not self.distillation_enabled or validate:
            return
        raw = sample_kwargs['raw_prompt']
        expected = self.processor.apply_chat_template(raw, tokenize=False, add_generation_prompt=True, add_vision_id=False)
        assert common.text_ids(self.processor, expected) == prompt_ids, 'Student prompt changed from frozen SFT protocol'
        gt = [int(value) for value in sample_kwargs['extra_info']['ground_truth']]
        teacher_prompt = teacher_tokens(self.processor, raw, gt, self.template)
        assert len(teacher_prompt) + len(response_ids) + 1 <= 9216
        teacher_ids, teacher_scores = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=teacher_prompt + response_ids,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            mm_processor_output=getattr(output, 'mm_processor_output', None),
            routing_key='streetview',
        )
        teacher_ids, teacher_scores = align_teacher_response(
            teacher_ids, teacher_scores, len(teacher_prompt), prompt_ids, response_ids)
        output.extra_fields['teacher_ids'] = teacher_ids
        output.extra_fields['teacher_logprobs'] = teacher_scores

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        result = super()._postprocess(inputs, input_non_tensor_batch, validate)
        if not validate and os.environ.get('STREETVIEW_AUDIT_DIR'):
            from collections import defaultdict
            from streetview_audit import write
            groups = defaultdict(list)
            rows = []
            for output, extra in zip(inputs, input_non_tensor_batch['extra_info'], strict=True):
                length = int(output.response_mask.sum().item())
                reward = float(output.reward_score)
                groups[extra['id']].append(reward)
                rows.append({'id': extra['id'], 'reward': reward, 'response_length': length})
            # Each worker receives whole 8-response groups with the current 2-worker split.
            assert all(len(values) == 8 for values in groups.values())
            write('rollout_groups.jsonl', {'rollout_step': self.rollout_step, 'responses': len(rows),
                  'groups': len(groups), 'all_correct_groups': sum(all(v == 1 for v in vs) for vs in groups.values()),
                  'all_wrong_groups': sum(all(v == 0 for v in vs) for vs in groups.values()),
                  'records': rows})
        return result


class StreetViewManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(StreetViewWorker)
        super().__init__(*args, **kwargs)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else list(ground_truth)
    prompt = json.loads((DATA / 'prompt.json').read_text())
    result = common.score(solution_str, gt, prompt['step_titles'])
    return {'score': float(result['exact_match']),
            **{k: float(v) for k, v in result.items() if k not in ('prediction', 'truncated')}}
