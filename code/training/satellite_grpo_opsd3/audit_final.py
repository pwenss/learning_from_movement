"""Read-only full-epoch acceptance for Satellite DDP GRPO+OPSD and every probe."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from safetensors.torch import load_file
import torch
import yaml
import satellite_sft_common as common

ROOT = common.ROOT
RUN = ROOT / 'runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023'
DATA = ROOT / 'training/satellite_grpo_opsd_qwen3vl8b_lora_v1'
DIRECTORY = RUN / 'deployment/ddp_resident/formal'
CHECKPOINTS = [17,34,51,68,85,102,119,136,153,170,182]
GPU_UUIDS = __import__('os').environ.get('LFM_GPU_UUIDS', 'GPU-PLACEHOLDER-0,GPU-PLACEHOLDER-1').split(',')


def rows(path):
    return common.read_jsonl(path)


def digest(parameters):
    h = hashlib.sha256()
    for name, value in sorted(parameters):
        assert torch.isfinite(value).all(), name
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def stopped(identity):
    p = Path('/proc') / str(identity['pid'])
    try:
        stat = (p/'stat').read_text().split(') ',1)[1].split()
        return (p.stat().st_uid != identity['uid'] or stat[19] != identity['start_ticks']
                or stat[0] in ['Z','X'])
    except FileNotFoundError:
        return True


def audit_checkpoint(cycle, expected_cursor=None):
    directory = DIRECTORY / f'global_step_{cycle}'
    actor = directory / 'actor'
    config = json.loads((actor/'ddp_config.json').read_text())
    assert config['backend']=='torch_ddp' and config['world_size']==3
    assert config['global_step']==cycle and config['optimizer_update']==4*cycle
    assert len(config['replica_adapter_sha256'])==len(config['replica_adam_sha256'])==3
    assert len(set(config['replica_adapter_sha256']))==len(set(config['replica_adam_sha256']))==1
    adapter = load_file(actor/'adapter_model.safetensors')
    assert len(adapter)==504 and sum(x.numel() for x in adapter.values())==174587904
    assert all(x.dtype==torch.float32 for x in adapter.values())
    assert digest(adapter.items())==config['replica_adapter_sha256'][0]
    adapter_sha = common.hash_file(actor/'adapter_model.safetensors')
    del adapter
    optimizer = torch.load(actor/'optimizer.pt',map_location='cpu',weights_only=False)
    state = optimizer['state']
    assert len(state)==504
    assert {int(s['step'].item()) for s in state.values()}=={cycle*4}
    assert all(s[k].dtype==torch.float32 for s in state.values() for k in ('exp_avg','exp_avg_sq'))
    moments = digest((f'{i}/{key}',s[key]) for i,s in enumerate(state.values()) for key in ('exp_avg','exp_avg_sq'))
    assert moments==config['replica_adam_sha256'][0]
    del optimizer,state
    schedulers = []
    rng_shas = []
    for rank in range(3):
        path = actor/f'rank_{rank}_rng.pt'
        extra = torch.load(path,map_location='cpu',weights_only=False)
        assert set(extra['rng'])=={'torch','cuda','numpy','python'}
        assert all(extra['rng'][k].dtype==torch.uint8 and extra['rng'][k].numel()>0 for k in ['torch','cuda'])
        assert isinstance(extra['rng']['numpy'],tuple) and isinstance(extra['rng']['python'],tuple)
        schedulers.append(extra['lr_scheduler'])
        rng_shas.append(common.hash_file(path))
    assert all(s==schedulers[0] for s in schedulers) and schedulers[0]['last_epoch']==cycle
    loader = torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    snap = loader['_snapshot']
    expected = min(cycle*96,17452) if expected_cursor is None else expected_cursor
    main = snap['_main_snapshot']
    assert loader['_steps_since_snapshot']==0 and snap['_snapshot_step']==cycle
    assert main['_sampler_iter_state']['samples_yielded']==expected
    assert main['_sampler_iter_state']['sampler_iter_state']['yielded']==expected
    return {'cycle':cycle,'optimizer_update':cycle*4,'data_cursor':expected,
            'adapter_sha256':adapter_sha,'adapter_tensor_digest':config['replica_adapter_sha256'][0],
            'adam_tensor_digest':moments,'three_rank_rng_sha256':rng_shas,'scheduler_epoch':cycle}


def audit_probe(cycle, tokenizer, expected, prompt, checkpoint=None):
    directory = DIRECTORY/f'probe_evaluations/cycle_{cycle:03d}'
    predictions = rows(directory/'predictions.jsonl')
    report = json.loads((directory/'report.json').read_text())
    assert report['status']=='passed' and report['cycle']==cycle and report['n']==256
    assert len(predictions)==len({r['id'] for r in predictions})==256
    assert {r['id'] for r in predictions}==set(expected)
    assert report['predictions_sha256']==common.hash_file(directory/'predictions.jsonl')
    assert report['probe_sha256']==common.hash_file(DATA/'validation_probe.jsonl')
    assert report['prompt_sha256']==common.hash_file(DATA/'prompt.json')
    protocol=report['protocol']
    assert protocol['engine']=='vllm' and protocol['gpus']==[0,2,3] and protocol['tp']==1
    assert protocol['temperature']==0 and protocol['n']==1 and protocol['max_response_tokens']==3072
    assert protocol['stop_token_ids']==[151645] and protocol['privileged_gt_in_student_prompt'] is False
    assert report['weight_sync_before_probe'] and report['raw_answers_rescored']
    if checkpoint:
        assert report['checkpoint_adapter_sha256']==checkpoint['adapter_sha256']
        assert report['optimizer_update']==cycle*4
    for row in predictions:
        source=expected[row['id']]
        assert row['ground_truth']==source['ground_truth'] and row['city']==source['city']
        ids=row['output_token_ids']
        assert row['output_tokens']==len(ids) and 1<=len(ids)<=3072
        assert tokenizer.decode(ids,skip_special_tokens=True)==row['text']
        clipped=len(ids)==3072 and ids[-1]!=151645
        score=common.score(row['text'],source['ground_truth'],prompt['step_titles'],clipped)
        assert all(row[k]==v for k,v in score.items()),row['id']
    summary=common.summarize(predictions)
    assert summary==json.loads((directory/'summary.json').read_text())==report['summary']
    return {'cycle':cycle,'optimizer_update':cycle*4,'summary':summary,
            'predictions_sha256':report['predictions_sha256'],'status':'passed'}


def audit():
    exit_record=json.loads((RUN/'provision/formal_exit.json').read_text())
    assert exit_record['exit_code']==0 and stopped(exit_record['launcher'])
    frozen=json.loads((RUN/'provision/frozen_inputs.json').read_text())
    assert all(common.hash_file(Path(path))==value for path,value in frozen.items())
    preflight=json.loads((RUN/'provision/preflight_acceptance.json').read_text())
    assert preflight['status']=='passed' and preflight['checkpoint_roundtrip_all_three_ranks']
    cfg=yaml.safe_load((DIRECTORY/'resolved_config.yaml').read_text())
    actor=cfg['actor_rollout_ref']['actor'];rollout=cfg['actor_rollout_ref']['rollout']
    assert (cfg['data']['train_batch_size'],rollout['n'],actor['ppo_mini_batch_size'],
            actor['ppo_epochs'],actor['ppo_micro_batch_size_per_gpu'])==(96,8,24,1,1)
    assert cfg['trainer']['total_epochs']==1 and cfg['trainer']['save_freq']==17
    assert cfg['trainer']['resume_mode']=='disable' and cfg['trainer']['resume_from_path'] is None
    assert cfg['trainer']['n_gpus_per_node']==cfg['distillation']['n_gpus_per_node']==3
    assert actor['optim']['lr']==1e-6 and actor['use_kl_loss'] and actor['kl_loss_coef']==.001
    assert cfg['distillation']['distillation_loss']['distillation_loss_coef']==.01
    assert not cfg['algorithm']['use_kl_in_reward']
    assert rollout['logprobs_mode']=='raw_logprobs' and (rollout['temperature'],rollout['top_p'],rollout['top_k'])==(1,1,-1)
    assert cfg['actor_rollout_ref']['model']['path']==cfg['distillation']['teacher_models']['teacher_model']['model_path']
    assert 'satellite_sft_checkpoint700' in cfg['actor_rollout_ref']['model']['path']
    all_updates=[]
    for rank in range(3):
        model=rows(DIRECTORY/f'audit/model_rank_{rank}.jsonl')
        assert len(model)==1 and model[0]['backend']=='torch_ddp' and model[0]['world_size']==3
        assert model[0]['gpu_uuid']==GPU_UUIDS[rank]
        updates=rows(DIRECTORY/f'audit/updates_rank_{rank}.jsonl')
        assert [u['optimizer_update'] for u in updates]==list(range(1,729))
        for u in updates:
            step=u['optimizer_update'];local=(10 if rank==2 else 11) if step==728 else 64
            assert u['rollout_step']==(step+3)//4 and u['minibatch_index']==(step-1)%4
            assert u['local_responses']==local and u['global_responses']==(32 if step==728 else 192)
            m=u['metrics'];assert all(math.isfinite(v) for v in m.values()) and m['grad_norm']>0
            assert abs(m['loss']-(m['actor/pg_loss']+.01*m['distillation/loss']+.001*m['kl_loss']))<2e-5
        all_updates.append(updates)
    for i in range(728):
        norms=[rank[i]['metrics']['grad_norm'] for rank in all_updates]
        assert max(norms)-min(norms)<=max(1e-7,abs(norms[0])*1e-6)
    groups=defaultdict(list)
    for g in rows(DIRECTORY/'audit/rollout_groups.jsonl'):
        groups[g['rollout_step']].extend(g['records'])
    assert sorted(groups)==list(range(1,183))
    ids=Counter()
    for cycle,answers in groups.items():
        n=76 if cycle==182 else 96
        counts=Counter(a['id'] for a in answers)
        assert len(counts)==n and len(answers)==n*8 and set(counts.values())=={8}
        assert all(a['reward'] in (0,1) and 1<=a['response_length']<=3072 for a in answers)
        ids.update(counts)
    train=rows(DATA/'train.jsonl')
    assert len(train)==len({r['id'] for r in train})==17452
    assert {r['id'] for r in train}==set(ids) and set(ids.values())=={8}
    assert len({r['city'] for r in train})==23
    stages=rows(DIRECTORY/'audit/serial_stages.jsonl')
    assert [r['rollout_step'] for r in stages]==list(range(1,183))
    assert all(r['responses']==(608 if r['rollout_step']==182 else 768)
               and r['scores_finite'] and r['shared_server_pids']
               and r['stage_order']==['student','reference','opsd'] for r in stages)
    padding=rows(DIRECTORY/'audit/tail_dispatch_padding.jsonl')
    assert len(padding)==1 and padding[0]['cycle']==182 and padding[0]['real_responses']==608
    assert padding[0]['dispatch_rows']==609 and padding[0]['padding_response_mask_sum']==0
    assert padding[0]['added_generation_requests']==padding[0]['added_trajectory_ids']==0
    zero=rows(DIRECTORY/'audit/zero_loss_padding_micro_rank_2.jsonl')
    assert len(zero)==1 and zero[0]['valid_loss_tokens']==0
    assert int((DIRECTORY/'latest_checkpointed_iteration.txt').read_text())==182
    saved={int(p.name.removeprefix('global_step_')) for p in DIRECTORY.glob('global_step_*') if p.is_dir()}
    assert saved==set(CHECKPOINTS)
    checkpoints={str(c):audit_checkpoint(c) for c in CHECKPOINTS}
    expected={r['id']:r for r in rows(DATA/'validation_probe.jsonl')}
    prompt=json.loads((DATA/'prompt.json').read_text());tokenizer=common.processor().tokenizer
    probes={str(c):audit_probe(c,tokenizer,expected,prompt,checkpoints.get(str(c))) for c in [0,*CHECKPOINTS]}
    milestones=rows(DIRECTORY/'audit/checkpoint_probe.jsonl')
    assert [x['cycle'] for x in milestones]==CHECKPOINTS
    assert all(x['status']=='passed' and x['checkpoint_saved_before_probe'] for x in milestones)
    assert all(x['student_only'] and not x['gt_teacher_scoring_invoked'] for x in rows(DIRECTORY/'audit/probe_stages.jsonl'))
    merged=Path(cfg['actor_rollout_ref']['model']['path'])
    merge=json.loads((merged/'merge_verification.json').read_text())
    assert merge['status']=='weights_verified'
    assert common.hash_file(Path(merge['sft_checkpoint'])/'adapter_model.safetensors')==merge['sft_adapter_sha256']
    assert all(common.hash_file(merged/name)==h for name,h in merge['merged_shard_sha256'].items())
    report={'status':'passed','rollout_cycles':182,'optimizer_updates_per_rank':728,
            'unique_training_trajectories':17452,'responses':139616,'training_cities':23,
            'all_three_rank_losses_and_gradients_finite_and_synchronized':True,
            'saved_checkpoints':checkpoints,'probe_evaluations':probes,
            'initialization':'fresh Satellite SFT checkpoint-700',
            'formal_exit_code':0,'frozen_code_and_data_unchanged':True,
            'all_checkpoint_probes_complete':True,'full_validation_and_ood_performed':False,
            'verification_script_sha256':common.hash_file(Path(__file__)),'time':time.time()}
    common.atomic_json(DIRECTORY/'acceptance.json',report)
    return report


if __name__=='__main__':
    torch.set_num_threads(4)
    result=audit()
    print(json.dumps({k:v for k,v in result.items() if k not in ['saved_checkpoints','probe_evaluations']},indent=2))
