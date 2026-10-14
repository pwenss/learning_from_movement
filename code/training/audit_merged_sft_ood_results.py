"""Independent CPU audit/recovery of completed merged-SFT OOD inference.

Does not launch inference or change frozen source/model files. Resolves the
Satellite reference protocol at its evaluation root, preserving the initial
coordinator exception in the acceptance record.
"""
import hashlib
import importlib
import json
import time
from pathlib import Path

import evaluate_merged_sft_ood as ev


def main():
    started = time.time()
    initial = json.loads((ev.OUT/'status.json').read_text())
    coordinator = json.loads((ev.ROOT/'runs/merged_sft_ood_20261014_identity.json').read_text())
    assert not ev.alive(coordinator), 'Do not audit over an active coordinator'
    results = {}
    for case, config in ev.CASES.items():
        common = importlib.import_module(case+'_sft_common')
        model, data = ev.paths(case)
        directory = ev.OUT/case
        exits = json.loads((directory/'worker_exits.json').read_text())
        assert len(exits) == 3 and all(x['exit_code'] == 0 and not ev.alive(x['identity']) for x in exits)
        protocol = json.loads((directory/'protocol.json').read_text())
        assert protocol == ev.contract(case)
        assert protocol['adapter'] is None and protocol['rl_adapter_loaded'] is False
        assert protocol['privileged_gt_in_student_prompt'] is False
        expected_rows = common.read_jsonl(data/'ood_test.jsonl')
        expected = {r['id']: r for r in expected_rows}
        rows = common.read_jsonl(directory/'predictions.jsonl')
        assert len(rows) == len({r['id'] for r in rows}) == len(expected) == config['n']
        assert [r['id'] for r in rows] == [r['id'] for r in expected_rows]
        shards = [r for gpu in ev.GPUS for r in common.read_jsonl(directory/f'gpu_{gpu}/predictions.jsonl')]
        assert len(shards) == len(rows) and {r['id']: r for r in shards} == {r['id']: r for r in rows}
        for gpu in ev.GPUS:
            shard_dir = directory/f'gpu_{gpu}'
            shard = common.read_jsonl(shard_dir/'predictions.jsonl')
            progress = json.loads((shard_dir/'progress.json').read_text())
            assert progress['phase'] == 'complete' and progress['completed'] == progress['total'] == len(shard)
            assert common.summarize(shard) == json.loads((shard_dir/'summary.json').read_text())
        processor = common.processor()
        prompt = json.loads((data/'prompt.json').read_text())
        protocol_sha = ev.sha(directory/'protocol.json')
        for row in rows:
            source = expected[row['id']]
            assert all(row[k] == source[k] for k in ('city', 'shuffle', 'ground_truth'))
            assert row['protocol_sha256'] == protocol_sha
            ids = row['output_token_ids']
            assert 1 <= len(ids) == row['output_tokens'] <= 3072
            assert processor.tokenizer.decode(ids, skip_special_tokens=True,
                       clean_up_tokenization_spaces=False) == row['text']
            prompt_ids = common.text_ids(processor, common.render(processor, source, prompt, False))
            assert hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest() == row['prompt_token_sha256']
            score = common.score(row['text'], source['ground_truth'], prompt['step_titles'],
                                len(ids) == 3072 and ids[-1] != 151645)
            assert all(row[k] == value for k, value in score.items())
        summary = common.summarize(rows)
        assert summary == json.loads((directory/'summary.json').read_text())
        frozen = json.loads((directory/'frozen_inputs.json').read_text())
        assert all(ev.sha(path) == digest for path, digest in frozen.items())
        reference = ev.ROOT/config['reference']
        protocol_path = reference/'protocol.json' if case == 'streetview' else reference.parent/'protocol.json'
        ref = json.loads(protocol_path.read_text())
        assert ref['model'] == protocol['model'] and ref['model_config_sha256'] == protocol['model_config_sha256']
        assert ref['prompt_sha256'] == protocol['prompt_sha256']
        split_hash = ref['split_sha256']['ood_test'] if isinstance(ref['split_sha256'], dict) else ref['split_sha256']
        assert split_hash == protocol['split_sha256']
        assert ref['temperature'] == 0 and ref['stop_token_ids'] == [151645]
        assert ref.get('max_response_length', ref.get('max_response_tokens')) == 3072
        assert ref['vllm'] == protocol['vllm'] and ref['torch'] == protocol['torch']
        ref_summary = json.loads((reference/'summary.json').read_text())
        ref_rows = common.read_jsonl(reference/'predictions.jsonl')
        assert len(ref_rows) == len({r['id'] for r in ref_rows}) == config['n']
        assert {r['id'] for r in ref_rows} == set(expected)
        assert all(r['ground_truth'] == expected[r['id']]['ground_truth'] for r in ref_rows)
        assert common.summarize(ref_rows) == ref_summary
        exact = sum(r['exact_match'] for r in rows)
        ref_exact = sum(r['exact_match'] for r in ref_rows)
        previous = json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else None
        if previous:
            assert previous['status'] == 'passed' and previous['summary'] == summary
            assert previous['predictions_sha256'] == ev.sha(directory/'predictions.jsonl')
        report = {'status':'passed', 'policy':'merged_sft_training_start', 'modality':case,
            'n':len(rows), 'unique_ids':len(expected), 'exact_matches':exact, 'summary':summary,
            'raw_answers_rescored':True, 'token_ids_decoded':True, 'frozen_inputs_unchanged':True,
            'protocol_sha256':protocol_sha, 'predictions_sha256':ev.sha(directory/'predictions.jsonl'),
            'reference_policy_protocol':str(protocol_path), 'reference_checkpoint_ood_em':ref_summary['overall']['exact_match'],
            'worker_exits':exits, 'independently_verified':True, 'time':time.time()}
        if previous is None:
            ev.atomic(directory/'report.json', report)
            ev.atomic(directory/'acceptance.json', report)
        results[case] = {**report, 'reference_exact_matches':ref_exact,
            'rl_minus_merged_correct':ref_exact-exact,
            'rl_minus_merged_em_pp':100*(ref_exact-exact)/config['n'],
            'per_city_exact_counts':{city:sum(r['exact_match'] for r in rows if r['city'] == city)
                                    for city in summary['per_city']},
            'generation_seconds':max(json.loads((directory/f'gpu_{gpu}/progress.json').read_text())['seconds'] for gpu in ev.GPUS)}
        print(json.dumps({'case':case,'status':'passed','n':config['n'],'exact_matches':exact,
                          'em':summary['overall']['exact_match']}), flush=True)
    result = {'status':'passed','results':results,'training_resumed':False,
              'all_workers_actual_exit_zero':True,'all_bound_processes_stopped':True,
              'independent_auditor':ev.ident(__import__('os').getpid()),
              'auditor_sha256':ev.sha(Path(__file__)), 'audit_seconds':time.time()-started,
              'original_coordinator_status':initial,
              'recovered_reference_protocol_path_only':initial['phase'] == 'needs_attention',
              'inference_repeated':False,'time':time.time()}
    ev.atomic(ev.OUT/'independent_final_verification.json', result)
    ev.atomic(ev.OUT/'acceptance.json', result)
    ev.atomic(ev.OUT/'status.json', {'phase':'complete_after_independent_audit','time':time.time(),
              'original_coordinator_status':initial,'result':str(ev.OUT/'acceptance.json')})
    ev.atomic(ev.OUT/'audit_recovery_exit.json', {'exit_code':0,'audit_identity':result['independent_auditor'],
                                               'acceptance':str(ev.OUT/'acceptance.json'),'time':time.time()})
    print(json.dumps({'status':'passed','results':{
       k:{key:v for key,v in r.items() if key not in ('worker_exits',)} for k,r in results.items()},
       'audit_seconds':result['audit_seconds']}), flush=True)


if __name__ == '__main__':
    main()
