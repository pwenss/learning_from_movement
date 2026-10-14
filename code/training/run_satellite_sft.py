"""Independent bounded Satellite experiment, gated on verified Street View completion."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT=Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN=ROOT/'runs/satellite_sft_qwen3vl8b_lora_v1'
DATA=ROOT/'training/satellite_sft_qwen3vl8b_lora_v1'
CODE=ROOT/'code/training'
TRAIN_ENV=Path('/path/to/workspace/conda-envs/streetview_sft_qwen3vl')
INFER_ENV=Path('/path/to/workspace/inference-envs/streetview_vllm_qwen3vl')
UUIDS='GPU-18cf292b-568d-aa6c-f0ac-3b7889f125aa,GPU-aded96b3-1317-039c-a3b5-cf8871defa07'


def atomic(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)


def proc_identity(pid):
    p=Path(f'/proc/{pid}')
    if not p.exists():return None
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':int(fields[19]),
            'pgid':int(fields[2]),'state':fields[0]}


def stage(name,argv,seconds,cpu=False,extra=None):
    atomic(RUN/'pipeline_status.json',{'phase':name,'time':time.time(),'pid':os.getpid()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else UUIDS,
             CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4' if cpu else '8',
             MKL_NUM_THREADS='4' if cpu else '8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
             TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             PYTORCH_KERNEL_CACHE_PATH=str(RUN/'kernel_cache'),
             VLLM_CACHE_ROOT=str(RUN/'vllm/kernel_cache'),
             SATELLITE_VLLM_CONCURRENCY='8',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    env.update(extra or {})
    with (RUN/f'{name}.log').open('a') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,cwd=CODE,env=env,start_new_session=True)
        identity=proc_identity(child.pid)
        assert identity and identity['pgid']==child.pid
        atomic(RUN/f'{name}.pid.json',{**identity,'started':time.time(),'command':argv,'timeout_seconds':seconds})
        try:rc=child.wait(timeout=seconds)
        except BaseException:
            if child.poll() is None:
                current=proc_identity(child.pid)
                assert current and all(current[k]==identity[k] for k in ['uid','start_ticks','pgid'])
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    current=proc_identity(child.pid)
                    assert current and all(current[k]==identity[k] for k in ['uid','start_ticks','pgid'])
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
            raise
    atomic(RUN/f'{name}.exit.json',{'exit_code':rc,'ended':time.time()})
    assert rc==0,f'{name} failed; inspect {name}.log'
    print('STAGE_COMPLETE',name,flush=True)


def verify_handoff():
    street=ROOT/'runs/streetview_sft_qwen3vl8b_lora_v1'
    handoff=json.loads((street/'completion_handoff_verification.json').read_text())
    assert handoff['verified'] and handoff['all_own_generation_processes_exited']
    assert json.loads((street/'final_audit.json').read_text())['verified']
    assert json.loads((street/'vllm/pipeline.exit.json').read_text())['exit_code']==0
    assert json.loads((street/'experiment_status.json').read_text())['phase']=='complete'
    for old in handoff['generation_identities']:
        now=proc_identity(old['pid'])
        assert now is None or any(now[k]!=old[k] for k in ['uid','start_ticks']), 'Street View generation still live'
    exported=json.loads((DATA/'export_verification.json').read_text())
    assert exported['verified']
    independent=json.loads((RUN/'data_independent_verification.json').read_text())
    assert independent['verified'] and independent['derived_hashes']==exported['derived_hashes']
    assert json.loads((RUN/'data_independent_audit.exit.json').read_text())['exit_code']==0
    workflow=json.loads((RUN/'callback_workflow_verification.json').read_text())
    assert workflow['verified'] and workflow['temporary_files_removed']
    assert json.loads((RUN/'callback_workflow.exit.json').read_text())['exit_code']==0
    assert not (RUN/'pipeline.exit.json').exists(), 'Review previous pipeline outcome explicitly'
    assert not list((RUN/'training').glob('checkpoint-*')), 'Fresh-base Satellite run expected'
    return handoff


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        handoff=verify_handoff()
        atomic(RUN/'streetview_handoff.json',handoff)
        for directory in ['kernel_cache','vllm']:(RUN/directory).mkdir(exist_ok=True)
        py=str(TRAIN_ENV/'bin/python');infer=str(INFER_ENV/'bin/python')
        ddp=[str(TRAIN_ENV/'bin/torchrun'),'--standalone','--nproc_per_node=2']
        for env,name in [(TRAIN_ENV,'environment.lock.txt'),(INFER_ENV,'vllm_environment.lock.txt')]:
            frozen=subprocess.run([str(env/'bin/python'),'-m','pip','freeze'],capture_output=True,text=True,check=True,timeout=60)
            (RUN/name).write_text(frozen.stdout)
        snapshot=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        (RUN/'gpu_processes_before.txt').write_text(snapshot.stdout)
        stage('gpu_preflight',ddp+[str(CODE/'gpu_preflight_satellite.py')],600)
        stage('preflight',ddp+[str(CODE/'train_satellite_sft.py'),'--smoke','--max-steps','25'],10800)
        stage('preflight_resume',ddp+[str(CODE/'train_satellite_sft.py'),'--smoke','--max-steps','26',
              '--resume',str(RUN/'preflight/checkpoint-25')],3600)
        stage('resume_state_verification',[py,'-B',str(CODE/'verify_satellite_preflight_resume.py')],600,cpu=True)
        stage('pretraining_audit',[py,'-B',str(CODE/'audit_satellite_sft_result.py'),'--pretraining'],3600,cpu=True)
        stage('formal_training',ddp+[str(CODE/'train_satellite_sft.py'),'--defer-generation'],345600)
        stage('generation_pilot',[infer,'-B','-u',str(CODE/'evaluate_satellite_vllm.py'),'--pilot','--pilot-count','64'],7200)
        stage('formal_generation',[infer,'-B','-u',str(CODE/'evaluate_satellite_vllm.py')],172800)
        stage('retained_recovery_audit',[py,'-B',str(CODE/'verify_satellite_checkpoint_recovery.py')],3600,cpu=True)
        stage('final_audit',[py,'-B',str(CODE/'audit_satellite_sft_result.py')],3600,cpu=True)
        assert json.loads((RUN/'final_audit.json').read_text())['verified']
        atomic(RUN/'pipeline.exit.json',{'exit_code':0,'verified':True,'ended':time.time()})
        atomic(RUN/'pipeline_status.json',{'phase':'complete','time':time.time()})
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        atomic(RUN/'pipeline.exit.json',{'exit_code':1,'error':error,'ended':time.time()})
        atomic(RUN/'pipeline_status.json',{'phase':'failed','error':error,'time':time.time()})
        raise


if __name__=='__main__':main()
