"""Gate a fresh formal run on a specific successful full pilot.

This supervisor has bounded waits and preserves the exact training recipe.
Post-training generation evaluation remains a separate required stage.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN = ROOT / 'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1'
CODE = ROOT / 'code/training/grpo_opsd'
DATA = ROOT / 'training/streetview_grpo_opsd_qwen3vl8b_lora_v1'
ENV = Path('/path/to/workspace/conda-envs/streetview_verl_grpo_opsd_py312')
PYTHON = str(ENV / 'bin/python')
STATE = RUN / 'provision/formal_supervisor.json'


def write_state(phase, **fields):
    value = {'phase': phase, 'time': time.time(), 'supervisor_pid': os.getpid(), **fields}
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def process_identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        assert directory.stat().st_uid == os.getuid(), 'Pilot belongs to another user'
        stat = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return {'start_ticks': stat[19], 'command': command}
    except FileNotFoundError:
        return None


def snapshot():
    paths = [CODE / n for n in ('experiment.yaml', 'agent.yaml', 'run_training.py',
             'launch.sh', 'streetview_runtime.py', 'streetview_audit.py', 'apply_audit_patch.py',
             'audit_training_run.py', 'export_adapter.py')]
    paths += [ROOT / 'code/training/streetview_sft_common.py']
    paths += [DATA / n for n in ('train.jsonl', 'validation_probe.jsonl', 'prompt.json', 'teacher_prompt.txt')]
    result = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    repo = str(ROOT / 'runtime/verl')
    result['verl_revision'] = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'],
                                                     text=True, timeout=10).strip()
    assert result['verl_revision'] == '704a9f01bda8b9863c02c113547191bf167bc88f'
    patch = subprocess.check_output(['git', '-C', repo, 'diff', '--binary', 'HEAD'], timeout=10)
    result['verl_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    return result


def check_gpu_availability():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.splitlines()}
    return free.get(2, 0) >= 70 * 1024 and free.get(1, 0) >= 40 * 1024


def main(pilot_pid=None):
    lock = (RUN / 'provision/formal_supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another formal supervisor already holds the lock')
    identity = process_identity(pilot_pid) if pilot_pid is not None else None
    if pilot_pid is not None:
        assert identity and str(CODE / 'launch.sh') + ' pilot' in identity['command'], identity
    else:
        controlled = json.loads((RUN / 'pilot/controlled_stop.json').read_text())
        assert controlled['status'] == 'completed_controlled_stop'
        assert process_identity(controlled['driver_identity']['pid']) is None, 'Original pilot driver is still live'
    frozen = snapshot()
    manifest = {'pilot_pid': pilot_pid, 'pilot_identity': identity, 'frozen_inputs': frozen}
    (RUN / 'provision/formal_supervisor_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_state('waiting_for_pilot', **manifest)
    deadline = time.monotonic() + 129600
    while pilot_pid is not None and process_identity(pilot_pid) == identity:
        if time.monotonic() >= deadline:
            raise TimeoutError('Specific pilot did not terminate within 36 hours')
        time.sleep(30)
    assert snapshot() == frozen, 'Training code/config/data changed after supervisor start'
    write_state('auditing_pilot', pilot_pid=pilot_pid)
    subprocess.run([PYTHON, str(CODE / 'audit_training_run.py'), '--phase', 'pilot',
                    '--require-complete', '--output', str(RUN / 'pilot/acceptance.json')],
                   check=True, timeout=90)
    exported = RUN / 'pilot/export_check'
    assert not exported.exists(), 'Unexpected existing temporary export'
    subprocess.run([PYTHON, str(CODE / 'export_adapter.py'), '--checkpoint',
                    str(RUN / 'pilot/global_step_5/actor'), '--output', str(exported)],
                   check=True, timeout=180)
    verification = json.loads((exported / 'lora_adapter/export_verification.json').read_text())
    assert verification['status'] == 'passed' and verification['exact_tensor_equality']
    verification['temporary_export_removed'] = True
    (RUN / 'pilot/adapter_export_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    assert exported.resolve().parent == (RUN / 'pilot').resolve()
    shutil.rmtree(exported)
    # A prior formal checkpoint requires an explicit resume review, not a fresh launch.
    assert not list((RUN / 'training').glob('global_step_*')), 'Formal checkpoints already exist'
    assert not (RUN / 'training/latest_checkpointed_iteration.txt').exists()
    deadline = time.monotonic() + 300
    while not check_gpu_availability():
        if time.monotonic() >= deadline:
            raise RuntimeError('Insufficient free memory on the requested GPUs after pilot shutdown')
        time.sleep(10)
    assert snapshot() == frozen
    with (RUN / 'provision/formal_launcher.log').open('w') as log:
        process = subprocess.Popen(['timeout', '1814400', 'bash', str(CODE / 'launch.sh'), 'formal'],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        write_state('formal_training', pilot_pid=pilot_pid, formal_pid=process.pid,
                    planned_rollout_cycles=261, planned_optimizer_updates=1044)
        returncode = process.wait(timeout=1814460)
    assert returncode == 0, f'Formal training exited with status {returncode}'
    subprocess.run([PYTHON, str(CODE / 'audit_training_run.py'), '--phase', 'formal',
                    '--require-complete', '--output', str(RUN / 'training/acceptance.json')],
                   check=True, timeout=90)
    write_state('training_complete_evaluation_pending', formal_pid=process.pid,
                generation_validation_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pilot-pid', type=int)
    source.add_argument('--pilot-completed', action='store_true')
    args = parser.parse_args()
    cuda = ENV / 'lib/python3.12/site-packages/nvidia/cu13'
    os.environ.update(OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUDA_HOME=str(cuda),
                      PATH=f'{ENV / "bin"}:{cuda / "bin"}:' + os.environ['PATH'],
                      LD_LIBRARY_PATH=str(cuda / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
                      PYTHONPATH=f'{ROOT / "code/training"}:{CODE}:{ROOT / "runtime/verl"}')
    try:
        main(args.pilot_pid)
    except Exception as error:
        write_state('attention_required', error=f'{type(error).__name__}: {error}')
        raise
