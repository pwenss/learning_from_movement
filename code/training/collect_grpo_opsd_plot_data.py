"""Read-only export of accepted formal GRPO+OPSD branches for figures.

Run on training-host; stdout is a compact JSON figure source. Original audits,
responses, checkpoints and logs remain on the server.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUNS = ROOT/'runs'
SV = RUNS/'streetview_grpo_opsd_qwen3vl8b_lora_v1'
SAT = RUNS/'satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023'
S3 = SV/'deployment/serial_ddp/ddp_resident/formal'
S4 = SV/'deployment/serial_ddp4/ddp_resident/formal'
SF = SAT/'deployment/ddp_resident/formal'
SOURCES = {}


def read(path):
    b = Path(path).read_bytes()
    SOURCES[str(Path(path).relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return b.decode('utf-8')


def rows(path):
    return [json.loads(l) for l in read(path).splitlines() if l.strip()]


def cycle_log(path, lo, hi):
    result = {}
    for line in read(path).splitlines():
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        clean = re.sub(r'np\.[a-zA-Z0-9_]+\(([-+0-9.eE]+)\)', r'\1', clean)
        match = re.search(r'\bstep:(\d+) - ', clean)
        if not match or not lo <= int(match[1]) <= hi:
            continue
        pairs = {}
        for item in clean.split(' - '):
            if ':' not in item:
                continue
            k, value = item.rsplit(':', 1)
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                pairs[k] = number
        keys = ['timing_s/step', 'timing_s/update_actor', 'timing_s/serial_student',
                'timing_s/serial_reference', 'timing_s/serial_opsd', 'response_length/mean',
                'actor/pg_loss', 'actor/kl_loss', 'actor/distillation/loss',
                'actor/loss', 'actor/grad_norm', 'critic/score/mean']
        result[int(match[1])] = {k: pairs[k] for k in keys if k in pairs}
    return result


def branch(directory, lo, hi, ranks, max_update):
    audits = [rows(directory/f'audit/updates_rank_{r}.jsonl') for r in range(ranks)]
    selected = [[x for x in rr if lo <= x['rollout_step'] <= hi and x['optimizer_update'] <= max_update] for rr in audits]
    assert all([x['optimizer_update'] for x in rr] == [x['optimizer_update'] for x in selected[0]] for rr in selected)
    updates = []
    for i, x in enumerate(selected[0]):
        norms = [rr[i]['metrics']['grad_norm'] for rr in selected]
        assert max(norms)-min(norms) <= 1e-6*max(1, norms[0]) and norms[0] > 0
        assert all(all(math.isfinite(v) for v in rr[i]['metrics'].values()) for rr in selected)
        m = x['metrics']
        joint = m['actor/pg_loss'] + .01*m['distillation/loss'] + .001*m['kl_loss']
        assert abs(joint-m['loss']) < 2e-6
        updates.append({'update':x['optimizer_update'], 'cycle':x['rollout_step'],
          'loss':m['loss'], 'pg_loss':m['actor/pg_loss'], 'opsd_loss':m['distillation/loss'],
          'reference_kl':sum(rr[i]['metrics']['kl_loss'] for rr in selected)/ranks,
          'entropy_mc':sum(rr[i]['metrics']['rollout_corr/rollout_log_ppl'] for rr in selected)/ranks,
          'grad_norm':m['grad_norm'], 'time':x['time'], 'ranks':ranks})
    groups = defaultdict(list)
    for x in rows(directory/'audit/rollout_groups.jsonl'):
        if lo <= x['rollout_step'] <= hi:
            groups[x['rollout_step']].append(x)
    rollouts = []
    for cycle, rr in sorted(groups.items()):
        answers = [a for r in rr for a in r['records']]
        counts = Counter(a['id'] for a in answers)
        assert len(answers) == 768 and len(counts) == 96 and set(counts.values()) == {8}
        rewards = [a['reward'] for a in answers]
        assert set(rewards) <= {0, 1}
        rollouts.append({'cycle':cycle, 'em':sum(rewards)/768, 'correct_responses':sum(rewards),
          'responses':768, 'trajectories':96, 'response_tokens':sum(a['response_length'] for a in answers)/768,
          'all_correct_groups':sum(r['all_correct_groups'] for r in rr),
          'all_wrong_groups':sum(r['all_wrong_groups'] for r in rr)})
    stages = [{k:x[k] for k in ('rollout_step','student_seconds','reference_seconds','opsd_seconds')}
              for x in rows(directory/'audit/serial_stages.jsonl') if lo <= x['rollout_step'] <= hi]
    return updates, rollouts, stages


def probe_summary(path, cycle):
    s = json.loads(read(path))['overall']
    assert s['n'] == 256
    return {'cycle':cycle, 'update':cycle*4, 'em':s['exact_match'],
      'exact':round(s['exact_match']*256), 'adjacency':s['directed_adjacency_accuracy'],
      'position':s['position_accuracy'], 'pairwise':s['pairwise_accuracy'], 'tau':s['kendall_tau'],
      'truncated':s['truncated'], 'valid':s['valid_answer'], 'cot_valid':s['cot_format_valid'],
      'tokens':s['output_tokens']}


def main():
    u3, r3, st3 = branch(S3, 1, 51, 3, 204)
    u4, r4, st4 = branch(S4, 52, 68, 4, 272)
    us, rs, sts = branch(SF, 1, 70, 3, 279)
    assert [x['update'] for x in u3+u4] == list(range(1,273))
    assert [x['update'] for x in us] == list(range(1,280))
    logs_sv = {**cycle_log(S3.parent/'formal_launcher.log',1,51), **cycle_log(S4.parent/'formal_launcher.log',52,68)}
    logs_sat = cycle_log(SAT/'provision/formal_launcher.log',1,69)
    # StreetView was paused immediately after checkpoint68, before its cycle
    # timing/aggregate console row was printed. All 272 updates are audited.
    assert set(logs_sv) == set(range(1,68)) and set(logs_sat) == set(range(1,70)), {'street_missing':sorted(set(range(1,68))-set(logs_sv)), 'satellite_missing':sorted(set(range(1,70))-set(logs_sat))}
    probes_sv = [probe_summary(S4/'probe_evaluation_step68/merged_sft_step0/summary.json',0)]
    for c in [17,34,51,68]:
        d=S4/f'probe_evaluation_step68/global_step_{c}/validation_probe'
        probes_sv.append(probe_summary(d/'summary.json',c))
    probes_sat = [probe_summary(SF/f'probe_evaluations/cycle_{c:03d}/summary.json',c) for c in [0,17,34,51,68]]
    merged = json.loads(read(RUNS/'merged_sft_ood_20261014/independent_final_verification.json'))
    assert merged['status']=='passed'
    evals = {
      'streetview':{'validation':json.loads(read(S4/'validation_cycle68_gpu2_7/summary.json')),
                   'ood':json.loads(read(S4/'ood_cycle68_gpu2_7_fast/summary.json'))},
      'satellite':{'validation':json.loads(read(SAT/'evaluation/cycle_051_full/validation_generation/summary.json')),
                   'ood':json.loads(read(SAT/'evaluation/cycle_051_full/ood_test/summary.json'))}}
    stop = json.loads(read(SAT/'provision/controlled_stop_for_cycle51_evaluation.json'))
    assert stop['observed_updates']==[279]*3 and not stop['full_epoch_completed']
    entropy_source=ROOT/'runtime/verl/verl/trainer/ppo/rollout_corr_helper.py'
    entropy_code=read(entropy_source)
    assert 'metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()' in entropy_code
    output = {'schema':2,'root':str(ROOT),'selection':{'streetview':68,'satellite':51},
      'metric_definitions':{
        'entropy_mc':'rollout_corr/rollout_log_ppl: response-masked mean negative rollout log probability per sequence, then mean across sequences and equally sized ranks; natural logs; sampled rollout-policy entropy estimate, not full-vocabulary entropy',
        'reference_kl':'kl_loss: raw reference KL estimator before coefficient 0.001, averaged across equally sized ranks',
        'entropy_source':str(entropy_source.relative_to(ROOT))},
      'branch_rules':{'streetview':'serial_ddp cycles1-51 then serial_ddp4 cycles52-68; exclude old updates205-214; no printed cycle68 timing row',
                      'satellite':'formal updates1-279; cycle70 rollout complete, actor only3/4 updates; cycle metrics1-69'},
      'streetview':{'updates':u3+u4,'rollouts':r3+r4,'stages':st3+st4,'cycles':[{'cycle':c,**v} for c,v in sorted(logs_sv.items())],'probes':probes_sv},
      'satellite':{'updates':us,'rollouts':rs,'stages':sts,'cycles':[{'cycle':c,**v} for c,v in sorted(logs_sat.items())],'probes':probes_sat},
      'final_evaluations':evals,'merged_sft_ood':merged['results'],'source_sha256':SOURCES}
    print(json.dumps(output))


if __name__=='__main__':main()
