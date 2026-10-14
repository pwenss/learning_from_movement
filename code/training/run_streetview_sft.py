"""Persistent, locked experiment orchestration; each stage has logs and a timeout."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
ENV = Path('/path/to/workspace/conda-envs/streetview_sft_qwen3vl')
MODEL = Path(__import__('os').environ.get('LFM_MODEL', '/path/to/models/Qwen3-VL-8B-Instruct'))
ID = 'streetview_sft_qwen3vl8b_lora_v1'
RUN, DATA, CODE = ROOT/'runs'/ID, ROOT/'training'/ID, ROOT/'code/training'


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    os.replace(tmp,path)


def status(phase, **kwargs):
    value = {'phase':phase,'pid':os.getpid(),'time':time.time(),**kwargs}
    atomic(RUN/'experiment_status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def ready(path, key):
    return path.exists() and json.loads(path.read_text()).get(key) is True


def wait_dependencies():
    deadline = time.monotonic()+5400
    while True:
        setup = RUN/'setup_exit.json'
        if setup.exists() and json.loads(setup.read_text())['exit_code'] != 0:
            raise RuntimeError('Environment setup failed; inspect setup.log')
        env_ok = setup.exists() and json.loads(setup.read_text())['exit_code'] == 0
        model_ok = ready(MODEL/'model_verification.json','verified')
        status('waiting_dependencies',environment_ready=env_ok,model_ready=model_ok)
        if env_ok and model_ok:
            return
        assert time.monotonic()<deadline,'Dependency preparation deadline exceeded'
        time.sleep(30)


def stage(name, argv, timeout, done=None):
    if done and done():
        print('STAGE_ALREADY_COMPLETE',name,flush=True)
        return
    status(name,command=argv)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa,GPU-aded96b3-1317-039c-a3b5-cf8871defa07',
               OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTORCH_KERNEL_CACHE_PATH=str(RUN/'kernel_cache'))
    (RUN/'kernel_cache').mkdir(parents=True,exist_ok=True)
    with (RUN/f'{name}.log').open('a') as log:
        proc = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=str(CODE),env=env,start_new_session=True)
        atomic(RUN/f'{name}.pid.json',{'pid':proc.pid,'started':time.time(),'command':argv})
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
            raise
    atomic(RUN/f'{name}.exit.json',{'exit_code':rc,'ended':time.time()})
    if rc:
        raise RuntimeError(f'{name} exited {rc}; inspect {name}.log')
    if done:
        assert done(),f'{name} returned zero without its required verified output'


def evaluate(name, manifest, adapter=None):
    argv=[str(ENV/'bin/torchrun'),'--standalone','--nproc_per_node=2',str(CODE/'evaluate_streetview_sft.py'),
          '--manifest',str(DATA/f'{manifest}.jsonl'),'--output',str(RUN/'evaluations'/name)]
    if adapter: argv += ['--adapter',str(adapter)]
    stage(name,argv,timeout=345600,done=lambda:(RUN/'evaluations'/name/'summary.json').exists())


def validate_saved_checkpoints():
    training = RUN/'training'
    assert ready(training/'training_complete.json','completed')
    completed = json.loads((training/'training_complete.json').read_text())
    assert completed['step'] == 626 and completed['epoch'] >= 1.999
    metrics = [json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines() if line.strip()]
    best = None
    for step in [100,200,300,313,400,500,600,626]:
        checkpoint = training/f'checkpoint-{step}'
        assert (checkpoint/'adapter_model.safetensors').is_file()
        output = training/'validation'/checkpoint.name
        argv = [str(ENV/'bin/torchrun'),'--standalone','--nproc_per_node=2',
                str(CODE/'evaluate_streetview_sft.py'),'--manifest',str(DATA/'validation_probe.jsonl'),
                '--output',str(output),'--adapter',str(checkpoint)]
        stage(f'checkpoint_validation_{step}',argv,345600,done=lambda output=output:(output/'summary.json').exists())
        result = json.loads((output/'summary.json').read_text())
        assert result['overall']['n'] == 256
        losses = [x['eval_loss'] for x in metrics if x['step'] == step and 'eval_loss' in x]
        assert losses and math.isfinite(losses[-1])
        key = [result['overall']['exact_match'],result['overall']['directed_adjacency_accuracy'],-losses[-1]]
        if best is None or key > best['selection_key']:
            best = {'checkpoint':str(checkpoint),'step':step,'selection_key':key,
                    'validation':result,'validation_loss':losses[-1],
                    'protocol':'fixed 256 validation trajectories; greedy; max_new_tokens=3072; generation deferred until training completes'}
    atomic(training/'best_checkpoint.json',best)
    atomic(RUN/'deferred_checkpoint_selection.json',{'verified':True,'completed_steps':[100,200,300,313,400,500,600,626],
            'selected_step':best['step'],'training_completed_before_selection':True,'time':time.time()})


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (RUN/'controller.pid').write_text(str(os.getpid()))
    try:
        wait_dependencies()
        py, tr = str(ENV/'bin/python'),str(ENV/'bin/torchrun')
        ddp=[tr,'--standalone','--nproc_per_node=2']
        stage('gpu_preflight',ddp+[str(CODE/'gpu_preflight.py')],600,
              lambda: all(ready(RUN/f'gpu_preflight.rank{k}.json','passed') for k in range(2)))
        stage('export', [py,'-B',str(CODE/'export_streetview_sft.py')],7200,
              lambda:ready(DATA/'export_verification.json','verified'))
        def smoke_ok():
            p=RUN/'preflight/preflight_complete.json'
            return ready(p,'passed') and json.loads(p.read_text())['step'] >= 25
        stage('preflight',ddp+[str(CODE/'train_streetview_sft.py'),'--smoke','--max-steps','25'],7200,smoke_ok)
        def resume_ok():
            p=RUN/'preflight/preflight_complete.json'
            return smoke_ok() and json.loads(p.read_text()).get('resume_from') and json.loads(p.read_text())['step'] == 26
        stage('preflight_resume',ddp+[str(CODE/'train_streetview_sft.py'),'--smoke','--max-steps','26',
              '--resume',str(RUN/'preflight/checkpoint-25')],1800,resume_ok)
        stage('resume_state_verification',[py,'-B',str(CODE/'verify_preflight_resume.py')],300,
              lambda:ready(RUN/'resume_state_verification.json','verified'))
        evaluate('baseline_validation_probe','validation_probe')
        training=RUN/'training'
        argv=ddp+[str(CODE/'train_streetview_sft.py'),'--defer-generation']
        checkpoints=sorted((x for x in training.glob('checkpoint-*') if (x/'trainer_state.json').exists()),
                           key=lambda x:int(x.name[11:])) if training.exists() else []
        if checkpoints:
            argv += ['--resume',str(checkpoints[-1])]
        stage('formal_training',argv,345600,lambda:ready(training/'training_complete.json','completed'))
        validate_saved_checkpoints()
        best=json.loads((training/'best_checkpoint.json').read_text())
        checkpoint=Path(best['checkpoint'])
        assert checkpoint.resolve().parent==training.resolve()
        evaluate('sft_validation_full','validation_generation',checkpoint)
        # Test starts only after checkpoint/configuration have been frozen using validation.
        atomic(RUN/'frozen_final_checkpoint.json',best)
        evaluate('baseline_ood_full','ood_test')
        evaluate('sft_ood_full','ood_test',checkpoint)
        report={'completed':True,'run_id':ID,'best_checkpoint':best,
                'export':json.loads((DATA/'export_verification.json').read_text()),
                'training':json.loads((training/'training_complete.json').read_text()),
                'evaluations':{name:json.loads((RUN/'evaluations'/name/'summary.json').read_text()) for name in
                               ['baseline_validation_probe','sft_validation_full','baseline_ood_full','sft_ood_full']},
                'time':time.time()}
        atomic(RUN/'experiment_result.json',report)
        lines=['# Street View Qwen3-VL-8B-Instruct SFT result','',f'Run: `{ID}`',
               f'Best checkpoint: `{checkpoint}`','', '| Evaluation | N | Exact match | Adjacency | Format valid |',
               '| --- | ---: | ---: | ---: | ---: |']
        for name,result in report['evaluations'].items():
            a=result['overall'];lines.append(f'| {name} | {a["n"]} | {a["exact_match"]:.4%} | {a["directed_adjacency_accuracy"]:.4%} | {a["cot_format_valid"]:.4%} |')
        (RUN/'experiment_result.md').write_text('\n'.join(lines)+'\n')
        status('complete',result=str(RUN/'experiment_result.json'),best_checkpoint=str(checkpoint))
    except Exception as error:
        status('failed',error=type(error).__name__+': '+str(error))
        traceback.print_exc()
        sys.exit(1)


if __name__=='__main__':
    main()
