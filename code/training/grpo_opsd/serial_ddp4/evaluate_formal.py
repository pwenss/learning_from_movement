"""Post-training generation using the existing frozen selection protocol."""
import argparse
import json
from pathlib import Path

import posttrain_evaluation as evaluation
from audit_formal import STEPS, inspect_checkpoint


def main(directory):
    directory = Path(directory)
    assert json.loads((directory / 'acceptance.json').read_text())['status'] == 'passed'
    output = directory / 'evaluation'
    assert not output.exists(), 'Review prior evaluation before retrying'
    output.mkdir()
    evaluation.OUTPUT = output
    evaluation.status('checking_canonical_adapters')
    adapters = {}
    for step in STEPS:
        target = directory / f'global_step_{step}/actor'
        inspect_checkpoint(target, step)
        adapters[step] = target
    evaluation.status('checkpoint_probes', jobs=11)
    evaluation.run_jobs([{'name': f'probe_{step}', 'split': 'validation_probe',
        'output': str(output / f'global_step_{step}/validation_probe'),
        'model_args': ['--adapter', str(adapters[step])]} for step in STEPS], gpus=(0, 1, 2, 3))
    controls = {name: evaluation.verify_results(path, 'validation_probe')
                for name, path in evaluation.BASELINES.items()}
    table = []
    for step in STEPS:
        result = evaluation.verify_results(output / f'global_step_{step}/validation_probe', 'validation_probe')
        for key in ('vllm', 'torch', 'prompt_sha256', 'split_sha256', 'temperature',
                    'max_response_length', 'stop_token_ids'):
            assert result['protocol'][key] == controls['original_sft']['protocol'][key]
            assert controls['merged_sft_step0']['protocol'][key] == controls['original_sft']['protocol'][key]
        table.append({'rollout_step': step, 'optimizer_update': step * 4, **result})
    best = max(table, key=lambda r: (r['exact_matches'], r['directed_edges_correct'], -r['rollout_step']))
    selection = {'selected_using': 'validation_probe_only', 'best_training_candidate': best['rollout_step'],
                 'best_candidate_exact_matches': best['exact_matches'], 'probe_n': 256,
                 'merged_step0_exact_matches': controls['merged_sft_step0']['exact_matches'],
                 'original_sft_exact_matches': controls['original_sft']['exact_matches'],
                 'replace_existing_sft': best['exact_matches'] > controls['original_sft']['exact_matches'],
                 'candidate_adapter': str(adapters[best['rollout_step']]), 'full_ood_used_for_selection': False}
    evaluation.atomic(output / 'probe_table.json', {'controls': controls, 'training_candidates': table})
    evaluation.atomic(output / 'selection.json', selection)
    policies = {'original_sft': ['--original-sft'], 'merged_sft_step0': [],
                'best_training_candidate': ['--adapter', selection['candidate_adapter']]}
    evaluation.status('full_comparisons', jobs=6, selection=selection)
    evaluation.run_jobs([{'name': f'{name}_{split}', 'split': split,
        'output': str(output / 'full_comparison' / name / split), 'model_args': args}
        for name, args in policies.items() for split in ('validation_generation', 'ood_test')], gpus=(1, 2, 3))
    results = {name: {split: evaluation.verify_results(output / 'full_comparison' / name / split, split)
                      for split in ('validation_generation', 'ood_test')} for name in policies}
    evaluation.atomic(output / 'full_comparison.json', results)
    evaluation.status('evaluations_complete_report_pending', selection=selection, generation_validation_complete=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--directory', required=True)
    a = p.parse_args()
    try:main(a.directory)
    except BaseException as e:
        evaluation.status('attention_required', error=type(e).__name__ + ': ' + str(e));raise
