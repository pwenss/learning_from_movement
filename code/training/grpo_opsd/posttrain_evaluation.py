"""Run frozen-protocol checkpoint selection and post-training comparisons.

Uses GPUs 1 and 2 only after the formal training audit passes. Selection uses
validation_probe exclusively; full validation and OOD never affect selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from run_after_pilot import CODE, DATA, ENV, PYTHON, ROOT, RUN, process_identity


OUTPUT = RUN / 'evaluation'
EXPORTS = RUN / 'exports'
STEPS = [*range(25, 251, 25), 261]
BASELINES = {'original_sft': RUN / 'baselines/original_sft_probe',
             'merged_sft_step0': RUN / 'baselines/single_round_sft_probe'}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(phase, **fields):
    value = {'phase': phase, 'time': time.time(), 'pid': os.getpid(), **fields}
    atomic(OUTPUT / 'status.json', value)
    print(json.dumps(value), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs():
    paths = [CODE / x for x in ('evaluate_checkpoint.py', 'export_adapter.py', 'experiment.yaml',
                                'run_training.py', 'posttrain_evaluation.py')]
    paths += [ROOT / 'code/training/streetview_sft_common.py']
    paths += [DATA / x for x in ('validation_probe.jsonl', 'validation_generation.jsonl',
                                'ood_test.jsonl', 'prompt.json')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_results(directory, split):
    import streetview_sft_common as common
    expected = {r['id']: r for r in common.read_jsonl(DATA / f'{split}.jsonl')}
    records = common.read_jsonl(directory / 'predictions.jsonl')
    assert len(records) == len(expected) and {r['id'] for r in records} == set(expected)
    protocol = json.loads((directory / 'protocol.json').read_text())
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert protocol['prompt_sha256'] == sha(DATA / 'prompt.json')
    assert protocol['split_sha256'] == sha(DATA / f'{split}.jsonl')
    assert (protocol['temperature'], protocol['max_response_length'], protocol['stop_token_ids']) == (0, 3072, [151645])
    prompt = json.loads((DATA / 'prompt.json').read_text())
    for row in records:
        assert row['protocol_sha256'] == protocol_hash
        assert row['ground_truth'] == expected[row['id']]['ground_truth']
        rescored = common.score(row['text'], row['ground_truth'], prompt['step_titles'], row['truncated'])
        for key, value in rescored.items():
            assert row[key] == value, (row['id'], key)
    summary = common.summarize(records)
    assert json.loads((directory / 'summary.json').read_text()) == summary, 'Stored summary does not match raw predictions'
    return {'n': len(records), 'exact_matches': sum(r['exact_match'] for r in records),
            'directed_edges_correct': sum(round(r['directed_adjacency_accuracy'] * 7) for r in records),
            'summary': summary, 'protocol': protocol}


def gpu_ready(gpu):
    free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    return int(free.strip()) >= 40 * 1024


def run_jobs(jobs, gpus=(1, 2)):
    assert len(gpus) > 0 and len(set(gpus)) == len(gpus) and all(gpu in (1, 2, 3) for gpu in gpus)
    # Longest expected jobs first; each GPU takes another job as soon as it is free.
    pending = queue.Queue()
    counts = {'validation_probe': 256, 'validation_generation': 1445, 'ood_test': 3525}
    for job in sorted(jobs, key=lambda j: -counts[j['split']]):
        pending.put(job)
    failed = []
    failed_lock = threading.Lock()

    def worker(gpu):
        while True:
            with failed_lock:
                if failed:
                    return
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                deadline = time.monotonic() + 300
                while not gpu_ready(gpu):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'GPU {gpu} does not have 40 GiB free')
                    time.sleep(10)
                directory = Path(job['output'])
                directory.mkdir(parents=True, exist_ok=True)
                command = [PYTHON, str(CODE / 'evaluate_checkpoint.py'), '--split', job['split'],
                           '--output', str(directory), *job['model_args']]
                environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                                   VLLM_WORKER_MULTIPROC_METHOD='spawn')
                print(json.dumps({'job': job['name'], 'gpu': gpu, 'event': 'started'}), flush=True)
                with (directory / 'launcher.log').open('a') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=environment, check=True, timeout=28800)
                verify_results(directory, job['split'])
                print(json.dumps({'job': job['name'], 'gpu': gpu, 'event': 'verified_complete'}), flush=True)
            except Exception as error:
                with failed_lock:
                    failed.append({'job': job['name'], 'gpu': gpu, 'error': str(error)})
                return
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    assert not failed and pending.empty(), {'failed': failed, 'pending_jobs': pending.qsize()}


def main(supervisor_pid=None, plan_only=False):
    if plan_only:
        controls = {name: {k: v for k, v in verify_results(path, 'validation_probe').items()
                           if k in ('n', 'exact_matches', 'directed_edges_correct')}
                    for name, path in BASELINES.items()}
        print(json.dumps({'required_formal_checkpoints': STEPS, 'probe_controls': controls,
                          'probe_jobs': 11, 'full_comparison_jobs': 6,
                          'gpu_jobs_started': False}, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another evaluation supervisor already holds the lock')
    frozen = frozen_inputs()
    if supervisor_pid:
        identity = process_identity(supervisor_pid)
        assert identity and 'run_after_pilot.py' in identity['command'], identity
        atomic(OUTPUT / 'manifest.json', {'supervisor_pid': supervisor_pid, 'identity': identity,
                                        'frozen_inputs': frozen})
        status('waiting_for_formal_training', supervisor_pid=supervisor_pid)
        deadline = time.monotonic() + 23 * 86400
        while process_identity(supervisor_pid) == identity:
            if time.monotonic() >= deadline:
                raise TimeoutError('Formal supervisor did not finish within 23 days')
            time.sleep(30)
        previous = json.loads((RUN / 'provision/formal_supervisor.json').read_text())
        assert previous['supervisor_pid'] == supervisor_pid
        assert previous['phase'] == 'training_complete_evaluation_pending', previous
    assert frozen_inputs() == frozen, 'Evaluation code, configuration or data changed while waiting'
    subprocess.run([PYTHON, str(CODE / 'audit_training_run.py'), '--phase', 'formal',
                    '--require-complete'], check=True, timeout=90)
    status('exporting_checkpoints')
    adapters = {}
    for step in STEPS:
        target = EXPORTS / f'global_step_{step}'
        subprocess.run([PYTHON, str(CODE / 'export_adapter.py'), '--checkpoint',
                        str(RUN / f'training/global_step_{step}/actor'), '--output', str(target)],
                       check=True, timeout=180)
        adapters[step] = target / 'lora_adapter'
    jobs = [{'name': f'probe_{step}', 'split': 'validation_probe',
             'output': str(OUTPUT / f'global_step_{step}/validation_probe'),
             'model_args': ['--adapter', str(adapters[step])]} for step in STEPS]
    status('checkpoint_probes', jobs=len(jobs))
    run_jobs(jobs)
    controls = {name: verify_results(path, 'validation_probe') for name, path in BASELINES.items()}
    table = []
    for step in STEPS:
        result = verify_results(OUTPUT / f'global_step_{step}/validation_probe', 'validation_probe')
        for key in ('vllm', 'torch', 'prompt_sha256', 'split_sha256', 'temperature',
                    'max_response_length', 'stop_token_ids'):
            assert result['protocol'][key] == controls['original_sft']['protocol'][key], key
            assert controls['merged_sft_step0']['protocol'][key] == controls['original_sft']['protocol'][key], key
        table.append({'rollout_step': step, 'optimizer_update': step * 4, **result})
    best = max(table, key=lambda r: (r['exact_matches'], r['directed_edges_correct'], -r['rollout_step']))
    selection = {'selected_using': 'validation_probe_only', 'best_training_candidate': best['rollout_step'],
                 'best_candidate_exact_matches': best['exact_matches'], 'probe_n': 256,
                 'merged_step0_exact_matches': controls['merged_sft_step0']['exact_matches'],
                 'original_sft_exact_matches': controls['original_sft']['exact_matches'],
                 'replace_existing_sft': best['exact_matches'] > controls['original_sft']['exact_matches'],
                 'candidate_adapter': str(adapters[best['rollout_step']]),
                 'full_ood_used_for_selection': False}
    atomic(OUTPUT / 'probe_table.json', {'controls': controls, 'training_candidates': table})
    # Freeze the choice before any full-validation/OOD generation starts.
    atomic(OUTPUT / 'selection.json', selection)
    policies = {'original_sft': ['--original-sft'], 'merged_sft_step0': [],
                'best_training_candidate': ['--adapter', selection['candidate_adapter']]}
    jobs = [{'name': f'{name}_{split}', 'split': split,
             'output': str(OUTPUT / 'full_comparison' / name / split), 'model_args': model_args}
            for name, model_args in policies.items() for split in ('validation_generation', 'ood_test')]
    status('full_comparisons', selection=selection, jobs=len(jobs))
    run_jobs(jobs)
    comparisons = {name: {split: verify_results(OUTPUT / 'full_comparison' / name / split, split)
                          for split in ('validation_generation', 'ood_test')} for name in policies}
    atomic(OUTPUT / 'full_comparison.json', comparisons)
    status('evaluations_complete_report_pending', selection=selection, generation_validation_complete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-supervisor', type=int)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    cuda = ENV / 'lib/python3.12/site-packages/nvidia/cu13'
    os.environ.update(OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUDA_HOME=str(cuda),
                      PATH=f'{ENV / "bin"}:{cuda / "bin"}:' + os.environ['PATH'],
                      LD_LIBRARY_PATH=str(cuda / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
                      PYTHONPATH=f'{ROOT / "code/training"}:{CODE}:{ROOT / "runtime/verl"}')
    # PYTHONPATH also applies to imports in this process, not just child jobs.
    import sys
    sys.path.insert(0, str(ROOT / 'code/training'))
    try:
        main(args.wait_for_supervisor, args.plan_only)
    except Exception as error:
        if not args.plan_only:
            status('attention_required', error=f'{type(error).__name__}: {error}')
        raise
