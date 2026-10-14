"""Evaluate both immutable merged SFT starting policies on full OOD.

Three TP1 vLLM replicas, separate modality manifests, no RL adapters or GT.
All inference runs in the pinned RL environment; full results are rescored.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__import__('os').environ.get('LFM_ROOT', '/path/to/visual_trajectory'))
OUT = ROOT/'runs/merged_sft_ood_20261014'
GPUS = [0, 2, 3]
UUIDS = ['GPU-529b98e5-8518-ef78-4293-5c074fe02056',
         'GPU-aded96b3-1317-039c-a3b5-cf8871defa07',
         'GPU-0c8286b3-cb9b-7243-1d88-752a84460579']
CASES = {
 'streetview': {'n':3525, 'size':[1024,512], 'visual_tokens':4096,
  'model':'streetview_sft_checkpoint500_single_round_bf16',
  'run':'streetview_grpo_opsd_qwen3vl8b_lora_v1', 'context':8192, 'batch':32,
  'reference':'runs/streetview_grpo_opsd_qwen3vl8b_lora_v1/deployment/serial_ddp4/ddp_resident/formal/ood_cycle68_gpu2_7_fast'},
 'satellite': {'n':4299, 'size':[768,768], 'visual_tokens':3872,
  'model':'satellite_sft_checkpoint700_single_round_bf16',
  'run':'satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023', 'context':9216, 'batch':96,
  'reference':'runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023/evaluation/cycle_051_full/ood_test'}
}


def atomic(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def ident(pid):
    q=Path('/proc')/str(pid)
    try:
        a=(q/'stat').read_text().split(') ',1)[1].split()
        return {'pid':pid,'uid':q.stat().st_uid,'start_ticks':a[19],'state':a[0]}
    except (FileNotFoundError,ProcessLookupError):return None


def alive(bound):
    a=ident(bound['pid'])
    return a and a['uid']==bound['uid'] and a['start_ticks']==bound['start_ticks'] and a['state'] not in ('Z','X')


def state(phase,**kw):
    obj={'phase':phase,'time':time.time(),'coordinator':ident(os.getpid()),**kw}
    atomic(OUT/'status.json',obj);print(json.dumps(obj),flush=True)


def paths(case):
    c=CASES[case]
    return ROOT/'models'/c['model'], ROOT/('training/'+case+'_grpo_opsd_qwen3vl8b_lora_v1')


def common_for(case):
    return importlib.import_module(case+'_sft_common')


def contract(case):
    model,data=paths(case);c=CASES[case]
    return {'modality':case,'policy':'merged_sft_training_start','cycle':0,'adapter':None,
       'model':str(model),'model_config_sha256':sha(model/'config.json'),
       'prompt_sha256':sha(data/'prompt.json'),'split_sha256':sha(data/'ood_test.jsonl'),
       'engine':'vllm','vllm':importlib.metadata.version('vllm'),
       'torch':importlib.metadata.version('torch'),'gpus':GPUS,'tp':1,'dtype':'bfloat16',
       'temperature':0,'top_p':1,'top_k':-1,'n':1,'max_response_tokens':3072,
       'stop_token_ids':[151645],'ignore_eos':True,'repetition_penalty':1,
       'presence_penalty':0,'frequency_penalty':0,'privileged_gt_in_student_prompt':False,
       'image_size':c['size'],'min_pixels':65536,'max_pixels':524288,
       'max_model_len':c['context'],'max_num_seqs':16,'max_num_batched_tokens':8192,
       'submission_batch':c['batch'],'gpu_memory_utilization':.6,
       'enable_prefix_caching':False,'mm_processor_cache_gb':0,'seed':20261009,
       'rl_adapter_loaded':False,'tf32':False}


def worker(case,rank):
    import torch
    from PIL import Image
    from vllm import LLM,SamplingParams
    common=common_for(case);model,data=paths(case);c=CASES[case];gpu=GPUS[rank]
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert actual==UUIDS[rank]
    torch.backends.cuda.matmul.allow_tf32=False
    p=common.processor();prompt=json.loads((data/'prompt.json').read_text())
    directory=OUT/case;proto=json.loads((directory/'protocol.json').read_text())
    assert contract(case)==proto and proto['adapter'] is None and not proto['rl_adapter_loaded']
    ph=sha(directory/'protocol.json')
    atomic(directory/f'worker_{gpu}_identity.json',{**ident(os.getpid()),'gpu':gpu,'uuid':actual,'phase':'loading_engine'})
    llm=LLM(model=str(model),tokenizer=str(common.MODEL),dtype='bfloat16',tensor_parallel_size=1,
       max_model_len=c['context'],max_num_seqs=16,max_num_batched_tokens=8192,
       gpu_memory_utilization=.6,enable_lora=False,limit_mm_per_prompt={'image':8,'video':0},
       mm_processor_kwargs={'min_pixels':65536,'max_pixels':524288,
         'size':{'shortest_edge':65536,'longest_edge':524288}},
       mm_processor_cache_gb=0,enable_prefix_caching=False,generation_config='vllm',seed=20261009)
    params=SamplingParams(temperature=0,top_p=1,top_k=-1,n=1,max_tokens=3072,
       repetition_penalty=1,presence_penalty=0,frequency_penalty=0,
       stop_token_ids=[151645],ignore_eos=True)
    selected=common.read_jsonl(data/'ood_test.jsonl')[rank::3]
    d=directory/f'gpu_{gpu}';d.mkdir(parents=True)
    atomic(directory/f'worker_{gpu}_identity.json',{**ident(os.getpid()),'gpu':gpu,'uuid':actual,'phase':'engine_ready'})
    done=0;started=time.monotonic()
    atomic(d/'progress.json',{'completed':0,'total':len(selected),'time':time.time(),'phase':'generating'})
    with (d/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for start in range(0,len(selected),c['batch']):
            chunk=selected[start:start+c['batch']];requests=[];expected_ids=[];opened=[]
            try:
                for i,record in enumerate(chunk):
                    assert len(record['images'])==8
                    images=[]
                    for rel in record['images']:
                        path=(ROOT/'dataset'/case/rel).resolve()
                        assert path.is_relative_to((ROOT/'dataset'/case).resolve())
                        with Image.open(path) as im:
                            assert list(im.size)==c['size'];images.append(im.convert('RGB'))
                    opened.extend(images)
                    rendered=common.render(p,record,prompt,False);ids=common.text_ids(p,rendered)
                    assert ids.count(151655)==c['visual_tokens'] and len(ids)+3072<=c['context']
                    if start==0 and i==0:
                        assert common.inputs(p,record,prompt,False)['input_ids'][0].tolist()==ids
                    requests.append({'prompt':rendered,'multi_modal_data':{'image':images}});expected_ids.append(ids)
                outputs=llm.generate(requests,params,use_tqdm=False)
                assert len(outputs)==len(chunk)
                for record,expected,result in zip(chunk,expected_ids,outputs,strict=True):
                    assert result.prompt_token_ids==expected and len(result.outputs)==1
                    seq=result.outputs[0];ids=list(seq.token_ids);assert 1<=len(ids)<=3072
                    text=p.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    row={**{k:record[k] for k in ['id','city','shuffle','ground_truth']},'text':text,
                       'output_token_ids':ids,'output_tokens':len(ids),'protocol_sha256':ph,
                       'prompt_token_sha256':hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                       'finish_reason':seq.finish_reason,'stop_reason':seq.stop_reason,
                       **common.score(text,record['ground_truth'],prompt['step_titles'],len(ids)==3072 and ids[-1]!=151645)}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush();done+=len(chunk)
            finally:
                for im in opened:im.close()
            atomic(d/'progress.json',{'completed':done,'total':len(selected),'phase':'generating',
               'seconds':time.monotonic()-started,'time':time.time()})
            print(f'{case} GPU{gpu}: {done}/{len(selected)}',flush=True)
    assert done==len(selected)
    atomic(d/'summary.json',common.summarize(common.read_jsonl(d/'predictions.jsonl')))
    atomic(d/'progress.json',{'completed':done,'total':len(selected),'phase':'complete',
       'seconds':time.monotonic()-started,'time':time.time()})


def audit_case(case,exits):
    common=common_for(case);model,data=paths(case);d=OUT/case;p=common.processor()
    source=common.read_jsonl(data/'ood_test.jsonl');expected={r['id']:r for r in source}
    rows=[r for g in GPUS for r in common.read_jsonl(d/f'gpu_{g}/predictions.jsonl')]
    assert len(rows)==len({r['id'] for r in rows})==len(expected)==CASES[case]['n']
    assert {r['id'] for r in rows}==set(expected)
    proto=json.loads((d/'protocol.json').read_text());assert proto==contract(case)
    ph=sha(d/'protocol.json');prompt=json.loads((data/'prompt.json').read_text())
    for row in rows:
        s=expected[row['id']]
        assert all(row[k]==s[k] for k in ['city','shuffle','ground_truth'])
        ids=row['output_token_ids'];assert 1<=len(ids)==row['output_tokens']<=3072
        assert p.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==row['text']
        assert row['protocol_sha256']==ph
        prompt_ids=common.text_ids(p,common.render(p,s,prompt,False))
        assert hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()==row['prompt_token_sha256']
        scores=common.score(row['text'],s['ground_truth'],prompt['step_titles'],len(ids)==3072 and ids[-1]!=151645)
        assert all(row[k]==v for k,v in scores.items())
    by_id={r['id']:r for r in rows};ordered=[by_id[s['id']] for s in source]
    path=d/'predictions.jsonl';assert not path.exists()
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered),encoding='utf-8')
    summary=common.summarize(common.read_jsonl(path));atomic(d/'summary.json',summary)
    frozen=json.loads((d/'frozen_inputs.json').read_text())
    assert all(sha(k)==v for k,v in frozen.items())
    reference=ROOT/CASES[case]['reference'];ref=json.loads((reference/'protocol.json').read_text())
    assert ref['model']==proto['model'] and ref['model_config_sha256']==proto['model_config_sha256']
    assert ref['prompt_sha256']==proto['prompt_sha256']
    split_hash=ref['split_sha256'];split_hash=split_hash['ood_test'] if isinstance(split_hash,dict) else split_hash
    assert split_hash==proto['split_sha256'] and ref['temperature']==0 and ref['stop_token_ids']==[151645]
    assert ref['vllm']==proto['vllm'] and ref['torch']==proto['torch']
    assert all(x['exit_code']==0 and not alive(x['identity']) for x in exits)
    report={'status':'passed','policy':'merged_sft_training_start','modality':case,
       'n':len(rows),'unique_ids':len(by_id),'exact_matches':sum(r['exact_match'] for r in rows),
       'summary':summary,'raw_answers_rescored':True,'token_ids_decoded':True,
       'protocol_sha256':ph,'predictions_sha256':sha(path),'frozen_inputs_unchanged':True,
       'reference_policy_protocol':str(reference/'protocol.json'),'worker_exits':exits,'time':time.time(),
       'reference_checkpoint_ood_em':json.loads((reference/'summary.json').read_text())['overall']['exact_match']}
    atomic(d/'report.json',report);atomic(d/'acceptance.json',report)
    return report


def coordinator():
    assert not OUT.exists(),'Refuse duplicate evaluation'
    OUT.mkdir(parents=True)
    state('preparing_merged_sft_ood',cases=list(CASES))
    results={}
    for case,c in CASES.items():
        model,data=paths(case);common=common_for(case);d=OUT/case;d.mkdir()
        state('freezing_inputs',case=case)
        assert json.loads((model/'merge_verification.json').read_text())['status']=='weights_verified'
        assert len(common.read_jsonl(data/'ood_test.jsonl'))==c['n']
        proto=contract(case);atomic(d/'protocol.json',proto)
        files=[Path(__file__).resolve(),Path(common.__file__),data/'prompt.json',data/'ood_test.jsonl']
        files.extend(p for p in model.iterdir() if p.is_file())
        frozen={str(p):sha(p) for p in files};atomic(d/'frozen_inputs.json',frozen)
        merge=json.loads((model/'merge_verification.json').read_text())
        for name,digest in merge.get('merged_shard_sha256',{}).items():assert frozen[str(model/name)]==digest
        processes=[];handles=[];bindings=[]
        try:
            for rank,gpu in enumerate(GPUS):
                free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=8)
                assert int(free.strip())>=55*1024,f'GPU{gpu} not available; no other processes will be stopped'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_WORKER_MULTIPROC_METHOD='spawn')
                h=(d/f'worker_{gpu}.log').open('ab',buffering=0);handles.append(h)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--case',case,'--worker',str(rank)],
                   cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(proc);bindings.append({**ident(proc.pid),'gpu':gpu,'rank':rank})
            state('generating_ood',case=case,workers=bindings,n=c['n'])
            deadline=time.monotonic()+8*3600
            while any(p.poll() is None for p in processes):
                failed=[(b['gpu'],p.returncode) for b,p in zip(bindings,processes) if p.poll() not in (None,0)]
                assert not failed,f'Nonzero evaluation exit: {failed}'
                assert time.monotonic()<deadline,'Eight-hour evaluation bound reached'
                time.sleep(15)
            exits=[{'identity':b,'gpu':b['gpu'],'exit_code':p.returncode} for b,p in zip(bindings,processes)]
            atomic(d/'worker_exits.json',exits);assert all(p.returncode==0 for p in processes)
            state('auditing_ood',case=case,workers=bindings)
            results[case]=audit_case(case,exits)
            print(f"{case}: OOD EM={results[case]['summary']['overall']['exact_match']:.4%}",flush=True)
        except BaseException as e:
            state('needs_attention',case=case,error=repr(e),workers=bindings)
            for b,proc in zip(bindings,processes):
                if proc.poll() is None and alive(b):os.killpg(proc.pid,signal.SIGTERM)
            for proc in processes:
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            raise
        finally:
            for h in handles:h.close()
    atomic(OUT/'acceptance.json',{'status':'passed','results':results,'time':time.time(),'training_resumed':False})
    state('complete',acceptance=str(OUT/'acceptance.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=list(CASES));parser.add_argument('--worker',type=int,choices=range(3))
    args=parser.parse_args()
    if args.worker is None:coordinator()
    else:
        assert args.case;worker(args.case,args.worker)
