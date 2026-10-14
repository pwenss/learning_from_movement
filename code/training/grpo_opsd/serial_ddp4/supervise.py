"""Four-GPU continuation from verified cycle51; preserve complete historical lineage."""
import os,json,time,subprocess,hashlib,fcntl
from pathlib import Path
ROOT=Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
RUN=ROOT/'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1'
HERE=ROOT/'code/training/grpo_opsd/serial_ddp4'
ENV=Path('/path/to/workspace/conda-envs/streetview_verl_grpo_opsd_py312')
FOLDER=RUN/'deployment/serial_ddp4/ddp_resident'
OLD=RUN/'deployment/serial_ddp/ddp_resident/formal'
STATE=RUN/'provision/serial_ddp4_state.json'
def ident(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 st=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':st[19]}
def state(phase,**kw):
 x={'time':time.time(),'phase':phase,'mode':'formal','variant':'ddp_resident','physical_gpus':[0,1,2,3],'rank_gpu_order':[0,2,3,1],'supervisor':ident(os.getpid()),'directory':str(FOLDER/'formal'),**kw}
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(STATE);print(json.dumps(x),flush=True)
def main():
 FOLDER.mkdir(parents=True,exist_ok=True)
 lock=(FOLDER/'formal.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (FOLDER/'formal').exists()
 assert json.loads((OLD/'controlled_stop_for_four_gpu.json').read_text())['status']=='stopped'
 frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
 cuda=ENV/'lib/python3.12/site-packages/nvidia/cu13'
 os.environ.update(PYTHONPATH=f'{HERE}:{ROOT}/code/training/grpo_opsd/three_gpu:{ROOT}/code/training/grpo_opsd/deployment:{ROOT}/code/training:{ROOT}/code/training/grpo_opsd:{ROOT}/runtime/verl',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_HOME=str(cuda),LD_LIBRARY_PATH=str(cuda/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
 with (FOLDER/'formal_launcher.log').open('x') as log:
  child=subprocess.Popen(['timeout','1814400','bash',str(HERE/'launch.sh'),'--variant','ddp_resident','--mode','formal'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  state('formal_initializing',formal_identity=ident(child.pid),resume_cycle=51,resume_update=204,planned_cycles=174,planned_optimizer_updates_per_existing_rank=696,new_rank_remaining_updates=492)
  deadline=time.monotonic()+120
  while not (FOLDER/'formal').exists():
   if child.poll() is not None:raise RuntimeError('Launch failed before output creation')
   if time.monotonic()>deadline:raise TimeoutError('Output initialization deadline')
   time.sleep(2)
  for step in (17,34,51):(FOLDER/'formal'/f'global_step_{step}').symlink_to(OLD/f'global_step_{step}',target_is_directory=True)
  (FOLDER/'formal'/'migration.json').write_text(json.dumps({'resume_cycle':51,'resume_update':204,'prefix':str(OLD),'new_physical_gpus':[0,1,2,3],'unchanged_global_batch':96,'new_local_micro_batches_per_update':48,'sources':frozen},indent=2))
  with (FOLDER/'resource_monitor.log').open('x') as resource:
   monitor=subprocess.Popen([str(ENV/'bin/python'),str(HERE/'monitor_resources.py'),'--pid',str(child.pid),'--output',str(FOLDER/'formal_resources.json'),'--max-seconds','1814500'],stdout=resource,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  state('formal_training',formal_identity=ident(child.pid),resume_cycle=51,resume_update=204,planned_cycles=174,planned_optimizer_updates_per_existing_rank=696,new_rank_remaining_updates=492)
  code=child.wait(timeout=1814460);monitor.wait(timeout=45)
 (FOLDER/'formal'/'exitcode').write_text(str(code)+'\n')
 if code:state('attention_required',exitcode=code);return
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 state('auditing_training')
 subprocess.run([str(ENV/'bin/python'),str(HERE/'audit_formal.py'),'--directory',str(FOLDER/'formal')],check=True,timeout=1800)
 state('training_complete_generation_pending',completed_cycles=174)
 with (FOLDER/'evaluation_launcher.log').open('x') as log:
  subprocess.run([str(ENV/'bin/python'),str(HERE/'evaluate_formal.py'),'--directory',str(FOLDER/'formal')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=259200)
 state('evaluations_complete_report_pending')
if __name__=='__main__':
 try:main()
 except BaseException as e:state('attention_required',error=type(e).__name__+': '+str(e));raise
