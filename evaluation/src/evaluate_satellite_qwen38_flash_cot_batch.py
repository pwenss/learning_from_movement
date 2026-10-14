from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time, uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/path/to/trajectory_generation')
DATA_ROOT = Path(os.environ.get('SATELLITE_DATA_ROOT', '/tmp/satellite_trajectory_v4_1k_work'))
DATASET = 'satellite_trajectory_v4_1k'
RUN_ID = os.environ.get('SATELLITE_EVAL_RUN_ID', 'satellite_eval_qwen38_flash_ood10_cot_20261003')
MODEL = os.environ.get('SATELLITE_EVAL_MODEL', 'qwen3.8-flash')
API = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')
SEED = int(os.environ.get('SATELLITE_EVAL_SEED', '20260925'))
MAX_TOKENS = 20000
THINKING_BUDGET = 20000
OOD = {'philadelphia','buenos_aires','berlin','osaka','melbourne'}
TERMINAL = {'completed','failed','expired','cancelled','canceled'}
OOD_SAMPLE_FRACTION = 0.10
BASE_URL = 'https://vlm-streetview.eu.cc'

SYSTEM_PROMPT = '''You are a visual spatial reasoning assistant specializing in trajectory reconstruction from multiple satellite observations.

Analyze the provided images and produce a detailed, evidence-grounded explanation following the four steps specified in the user prompt. Base your conclusions on visible overhead scene evidence and the stated image-orientation constraint. Account for possible differences in spatial extent and apparent scale across observations. Distinguish clear visual correspondences from uncertain matches, and do not invent visual details.

Refer to each observation by its displayed label, Image 0 through Image 7. These labels identify the order of presentation, not the spatial order along the trajectory.

Follow the required output format exactly. Within <think>, use all four step titles from the user prompt verbatim and in the specified order. Within <answer>, output only a list of eight integer indices, using each index from 0 to 7 exactly once. Do not include any text outside these two blocks.'''
USER_PROMPT = '''You are given 8 satellite observation images sampled along a spatial trajectory. The images have been shuffled and are presented in an arbitrary order, labeled Image 0 through Image 7. Each label immediately precedes its corresponding image. Each image is oriented so that its heading, corresponding to the upward direction of the image, is aligned with the direction of movement along the trajectory. Different observations may have different scales due to changes in observation altitude; infer movement from changes in observation center position rather than from scale changes. Reconstruct the spatial order in which these 8 observations would be encountered when following the trajectory.

Provide a detailed, evidence-grounded explanation using strictly the following four steps, followed by the final order.

Step 1: Global Scene Understanding
Provide a high-level overview of the geographic environment and the spatial characteristics of the route depicted across the images. Identify major land-cover patterns, transportation structures, building distributions, water bodies, or other large-scale geographic features that characterize the observed area. Focus on the overall character of the scene along the route rather than inferring their relative order at this stage.

Step 2: Cross-View Correspondence Identification
Identify visual correspondences across different observations by locating recurring geographic features that appear in multiple images and distinctive local cues that only appear in a subset of observations. Useful correspondences may include roads, intersections, building complexes, blocks, water features, vegetation boundaries, open spaces, or other persistent spatial structures. Focus on correspondences that provide useful evidence for determining the trajectory order.

Step 3: Viewpoint Transformation Analysis
Compare observations linked by shared geographic features, focusing on potentially adjacent pairs. For each pair, analyze how corresponding features change across observations as the image center moves along the trajectory. Examine their relative position within the image, orientation, visibility, overlap, and apparent size. Treat scale changes as possible altitude changes rather than evidence of forward or backward movement, and infer movement primarily from changes in observation center position. Use these transformations together with the constraint that the upward direction of each image is aligned with the direction of movement to infer relative order. Check whether the inferred transformations are spatially consistent and whether the observations can form a continuous movement path.

Step 4: Final Sequence Determination
Combine the inferred pairwise spatial relationships to construct the final trajectory order, ensuring that the sequence represents a spatially continuous path and that every image index appears exactly once.

Output using exactly the following format:

<think>
Step 1: Global Scene Understanding
...

Step 2: Cross-View Correspondence Identification
...

Step 3: Viewpoint Transformation Analysis
...

Step 4: Final Sequence Determination
...
</think>

<answer>
[i0, i1, i2, i3, i4, i5, i6, i7]
</answer>'''
STEP_TITLES = [f'Step {i}: {name}' for i, name in enumerate(['Global Scene Understanding','Cross-View Correspondence Identification','Viewpoint Transformation Analysis','Final Sequence Determination'],1)]
ANSWER_RE = re.compile(r'<answer>\s*\[\s*([0-7](?:\s*,\s*[0-7]){7})\s*\]\s*</answer>', re.I|re.S)
LIST_RE = re.compile(r'\[\s*([0-7](?:\s*,\s*[0-7]){7})\s*\]')

def now(): return datetime.now(timezone.utc).isoformat()
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def stable_shuffle(identifier):
    vals=list(range(8)); b=hashlib.sha256(f'{SEED}:{identifier}'.encode()).digest(); random.Random(int.from_bytes(b[:8],'big')).shuffle(vals); return vals
def custom_id(identifier): return 'sat-' + hashlib.sha256(identifier.encode()).hexdigest()[:28]
def parse_answer(text):
    m=ANSWER_RE.search(text or '')
    if not m:
        ms=list(LIST_RE.finditer(text or '')); m=ms[-1] if ms else None
    if not m: raise ValueError('No valid 8-index answer')
    p=[int(x.strip()) for x in m.group(1).split(',')]
    if sorted(p)!=list(range(8)): raise ValueError(f'Not a permutation: {p}')
    return p
def score(text, truth):
    pred=parse_answer(text)
    think=re.findall(r'<think>(.*?)</think>', text or '', re.I|re.S)
    answer=re.findall(r'<answer>.*?</answer>', text or '', re.I|re.S)
    outside=re.sub(r'<think>.*?</think>|<answer>.*?</answer>', '', text or '', flags=re.I|re.S).strip()
    format_ok=(len(think)==1 and len(answer)==1 and not outside and all(t in think[0] for t in STEP_TITLES))
    return {'prediction':pred,'valid_answer':True,'exact_match':pred==truth,'correct_positions':sum(a==b for a,b in zip(pred,truth)),'accuracy':sum(a==b for a,b in zip(pred,truth))/8,'cot_format_valid':format_ok}

class Runner:
  def __init__(self):
    for rel in ('env/packages','env/r2-packages','code/trajectory/server'): sys.path.insert(0,str(ROOT/rel))
    from common import load_project_environment
    load_project_environment()
    import requests
    self.requests=requests
    key=os.environ.get('DASHSCOPE_API_KEY')
    if not key:
      p=ROOT/'config'/'keys.env'
      if p.exists():
       for line in p.read_text(encoding='utf-8').splitlines():
        n,sep,v=line.removeprefix('export ').partition('=')
        if sep and n.strip()=='DASHSCOPE_API_KEY': key=v.strip().strip('"\''); break
    if not key: raise RuntimeError('DASHSCOPE_API_KEY not set')
    self.headers={'Authorization':'Bearer '+key}
    self.run_dir=ROOT/'runs'/RUN_ID; self.output=ROOT/'datasets'/DATASET/'evaluations'/RUN_ID
    self.manifest_path=self.output/'sample_manifest.json'; self.split_path=self.output/'split_manifest.json'; self.state_path=self.run_dir/'state.json'
    self.r2_prefix=f'{DATASET}/evaluations/{RUN_ID}'; self.base_url=BASE_URL
    self.run_dir.mkdir(parents=True,exist_ok=True); self.output.mkdir(parents=True,exist_ok=True)
  def prepare(self):
    if self.manifest_path.exists() and self.state_path.exists(): return load(self.manifest_path)
    cities={}; train=[]; test=[]; samples=[]
    city_root=DATA_ROOT/'cities'
    for city_dir in sorted(p for p in city_root.iterdir() if p.is_dir()):
      city=city_dir.name; seqroot=city_dir/'trajectory_sequences'; rows=[]
      for traj in sorted(p for p in seqroot.iterdir() if p.is_dir()):
        frames=sorted(traj.glob('frame_*.jpg'))
        if len(frames)!=8: raise RuntimeError(f'{traj} has {len(frames)} frames')
        ident=f'{city}/{traj.name}'; sh=stable_shuffle(ident); inv=[sh.index(i) for i in range(8)]
        row={'id':ident,'city':city,'trajectory_dir':str(traj),'custom_id':custom_id(ident),'shuffle_presented_to_original':sh,'ground_truth_presented_indices':inv,'images':[]}
        for frame in frames:
          name=uuid.uuid4().hex+'.jpg'; row['images'].append({'path':str(frame),'filename':name,'key':f'{self.r2_prefix}/images/{name}','url':f'{self.base_url}/{self.r2_prefix}/images/{name}'})
        rows.append(row)
      n=len(rows); k=round(n*OOD_SAMPLE_FRACTION) if city in OOD else 0
      ranked=sorted(rows,key=lambda r:hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest())
      selected={r['id'] for r in ranked[:k]}
      split='test' if city in OOD else 'train'
      cities[city]={'split':split,'population':n,'sample':len(selected)}
      (test if city in OOD else train).extend(rows)
      samples.extend(r for r in rows if r['id'] in selected)
    split={'run_id':RUN_ID,'dataset':DATASET,'seed':SEED,'split_rule':'Five fixed OOD cities; all other cities train','ood_cities':sorted(OOD),'train_cities':sorted(c for c in cities if c not in OOD),'cities':cities,'counts':{'train_trajectories':len(train),'test_trajectories':len(test),'test_sample_trajectories':len(samples),'test_sample_images':len(samples)*8},'created_at':now()}
    save(self.split_path,split)
    manifest={'run_id':RUN_ID,'dataset':DATASET,'model':MODEL,'provider':'Alibaba Cloud DashScope','seed':SEED,'sampling':'Independent SHA-256 ranking within each OOD city; 10% rounded to nearest integer','thinking_enabled':True,'thinking_budget':THINKING_BUDGET,'max_tokens':MAX_TOKENS,'r2_prefix':self.r2_prefix,'r2_base_url':self.base_url,'prompt':{'system':SYSTEM_PROMPT,'user':USER_PROMPT,'step_titles':STEP_TITLES,'source':'visual_trajectory_reconstruction_spec.md satellite CoT prompt'},'counts':{'cities':len(OOD),'trajectories':len(samples),'images':len(samples)*8},'cities':{c:cities[c] for c in sorted(OOD)},'created_at':now(),'sequences':sorted(samples,key=lambda r:r['id'])}
    save(self.manifest_path,manifest); save(self.state_path,{'run_id':RUN_ID,'status':'prepared','batches':[],'images_verified':False,'created_at':now()}); return manifest
  def persist(self,state): state['updated_at']=now(); save(self.state_path,state); save(self.output/'batch_submission.json',{'run_id':RUN_ID,'batches':state['batches'],'updated_at':now()})
  def upload(self,manifest,state,workers=32):
    if state.get('images_verified'): return
    from r2_images import load_settings
    settings=load_settings(self.run_dir,{'config_file':'config/r2.json'}); settings.update(key_prefix=self.r2_prefix,upload_workers=workers)
    from boto3.session import Session
    from botocore.config import Config
    secret=json.loads((ROOT/'config'/'r2_credentials.json').read_text(encoding='utf-8'))
    client=Session(aws_access_key_id=secret['access_key_id'],aws_secret_access_key=secret['secret_access_key'],region_name='auto').client('s3',endpoint_url=settings['endpoint'],config=Config(signature_version='s3v4',s3={'addressing_style':'path'},connect_timeout=15,read_timeout=90,proxies={'http':'http://127.0.0.1:18236','https':'http://127.0.0.1:18236'},max_pool_connections=workers,retries={'mode':'standard','total_max_attempts':3}))
    items=[im for r in manifest['sequences'] for im in r['images']]
    def one(im): client.put_object(Bucket=settings['bucket'],Key=im['key'],Body=Path(im['path']).read_bytes(),ContentType='image/jpeg',CacheControl='public, max-age=31536000, immutable')
    try:
      with ThreadPoolExecutor(max_workers=workers) as pool:
       for i,f in enumerate(as_completed([pool.submit(one,im) for im in items]),1): f.result();
       state.update(status='uploading',images_uploaded=i,images_total=len(items)); self.persist(state)
       if i%250==0 or i==len(items): print(f'R2 uploaded {i}/{len(items)}',flush=True)
    finally: client.close()
    samples={r['city']:r['images'][0] for r in manifest['sequences']}
    for city,im in samples.items():
      resp=self.requests.get(im['url'],timeout=(10,60)); resp.raise_for_status();
    state.update(status='images_verified',images_verified=True,images_uploaded=len(items),images_total=len(items),public_city_checks=len(samples)); self.persist(state)
  def request(self,row,manifest):
    content=[{'type':'text','text':manifest['prompt']['user']}]
    for shown,original in enumerate(row['shuffle_presented_to_original']): content += [{'type':'text','text':f'Image {shown}:'},{'type':'image_url','image_url':{'url':row['images'][original]['url']}}]
    return {'custom_id':row['custom_id'],'method':'POST','url':'/v1/chat/completions','body':{'model':MODEL,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':content}],'enable_thinking':True,'thinking_budget':THINKING_BUDGET,'temperature':0,'max_tokens':MAX_TOKENS}}
  def submit(self,manifest,state):
    if not state.get('images_verified'): raise RuntimeError('images not uploaded')
    done={b['city'] for b in state['batches'] if b.get('batch_id')}
    for city in sorted(OOD):
      if city in done: continue
      rows=[r for r in manifest['sequences'] if r['city']==city]; payload='\n'.join(json.dumps(self.request(r,manifest),ensure_ascii=False,separators=(',',':')) for r in rows)+'\n'
      resp=self.requests.post(API+'/files',headers=self.headers,data={'purpose':'batch'},files={'file':(f'{RUN_ID}_{city}.jsonl',payload.encode(),'application/jsonl')},timeout=(15,600)); resp.raise_for_status(); fid=resp.json()['id']
      resp=self.requests.post(API+'/batches',headers={**self.headers,'Content-Type':'application/json'},json={'input_file_id':fid,'endpoint':'/v1/chat/completions','completion_window':'24h','metadata':{'evaluation':RUN_ID,'city':city}},timeout=(15,300)); resp.raise_for_status(); bid=resp.json()['id']
      state['batches'].append({'city':city,'input_file_id':fid,'batch_id':bid,'status':resp.json().get('status'),'custom_ids':[r['custom_id'] for r in rows],'trajectory_count':len(rows),'created_at':now()}); self.persist(state); print(f'Submitted {city}: {bid}',flush=True); time.sleep(25)
  def collect(self,b,state):
    res=self.requests.get(API+'/batches/'+b['batch_id'],headers=self.headers,timeout=(15,120)); res.raise_for_status(); info=res.json(); b.update(status=info.get('status'),request_counts=info.get('request_counts'),checked_at=now())
    if info.get('status') in TERMINAL and not b.get('collected'):
      out=self.output/'responses'; out.mkdir(parents=True,exist_ok=True); path=out/(b['batch_id']+'.jsonl'); seen=[]
      with path.open('w',encoding='utf-8') as h:
       for fid in dict.fromkeys([info.get('output_file_id'),info.get('error_file_id')]):
        if not fid: continue
        rr=self.requests.get(API+'/files/'+fid+'/content',headers=self.headers,timeout=(15,300)); rr.raise_for_status()
        for line in rr.iter_lines():
         if line: h.write(line.decode()+'\n'); seen.append(json.loads(line))
      b.update(collected=True,result_path=str(path),returned_count=len(seen),successful_count=sum(1 for x in seen if (x.get('response') or {}).get('status_code')==200)); print(f'Collected {b["city"]}: {b["successful_count"]}/{len(b["custom_ids"])}',flush=True)
    self.persist(state)
  def report(self,manifest,state):
    by={}
    for b in state['batches']:
      if not b.get('collected'): continue
      for line in Path(b['result_path']).read_text(encoding='utf-8').splitlines():
       if not line: continue
       item=json.loads(line); by[item.get('custom_id')]=item
    rows=[]
    for r in manifest['sequences']:
      item=by.get(r['custom_id']); base={'id':r['id'],'city':r['city'],'custom_id':r['custom_id'],'shuffle_presented_to_original':r['shuffle_presented_to_original'],'ground_truth_presented_indices':r['ground_truth_presented_indices'],'image_order':'shuffled_presentation','image_paths':[r['images'][i]['path'] for i in r['shuffle_presented_to_original']],'image_urls':[r['images'][i]['url'] for i in r['shuffle_presented_to_original']]}
      if item and (item.get('response') or {}).get('status_code')==200:
       body=item['response']['body']; msg=body['choices'][0]['message']; text=msg.get('content',''); base.update(status='ok',content=text,model=body.get('model'),usage=body.get('usage',{}),finish_reason=body['choices'][0].get('finish_reason'))
       try: base.update(score(text,r['ground_truth_presented_indices']))
       except Exception as e: base.update(status='invalid_output',error=str(e))
      else: base.update(status='api_failed',error=(item or {}).get('error'))
      rows.append(base)
    ok=[x for x in rows if x.get('status')=='ok']; exact=sum(x.get('exact_match',False) for x in ok); pos=sum(x.get('correct_positions',0) for x in ok)
    summary={'run_id':RUN_ID,'dataset':DATASET,'provider':'Alibaba Cloud DashScope','model_requested':MODEL,'thinking_enabled':True,'test_trajectories':len(rows),'returned':len(ok),'failed':len(rows)-len(ok),'exact_match_count':exact,'exact_match_accuracy':exact/len(rows) if rows else None,'position_accuracy':pos/(8*len(rows)) if rows else None,'batch_statuses':{b['city']:b.get('status') for b in state['batches']},'manifest':str(self.manifest_path),'split_manifest':str(self.split_path),'created_at':now()}
    save(self.output/'results.json',{'summary':summary,'sequences':rows}); save(self.output/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return summary
  def run(self,poll):
    m=self.prepare(); s=load(self.state_path); self.upload(m,s); s=load(self.state_path); self.submit(m,s)
    while True:
      s=load(self.state_path)
      for b in s['batches']:
       if b.get('batch_id') and not b.get('collected'): self.collect(b,s)
      s=load(self.state_path); pending=sum(not b.get('collected') for b in s['batches']); print(json.dumps({'pending_batches':pending,'statuses':{b['city']:b.get('status') for b in s['batches']},'checked_at':now()},ensure_ascii=False),flush=True)
      if pending==0: s['status']='complete'; self.persist(s); return self.report(m,s)
      time.sleep(poll)

def main():
 p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','upload','submit','run','poll','report']); p.add_argument('--poll-seconds',type=int,default=60); p.add_argument('--workers',type=int,default=32); a=p.parse_args(); r=Runner(); m=r.prepare(); s=load(r.state_path)
 if a.action=='prepare': print(json.dumps(m['counts'],ensure_ascii=False))
 elif a.action=='upload': r.upload(m,s,a.workers)
 elif a.action=='submit': r.submit(m,s)
 elif a.action=='poll':
  for b in s['batches']:
   if b.get('batch_id') and not b.get('collected'): r.collect(b,s)
  r.report(m,load(r.state_path))
 elif a.action=='report': r.report(m,s)
 else: r.run(a.poll_seconds)
if __name__=='__main__': main()
