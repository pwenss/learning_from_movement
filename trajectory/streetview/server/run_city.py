"""Run one complete city, then stop for inspection. Server root is fixed."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
import secrets
from pathlib import Path
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from stage_store import checkpoint, enabled as stage_enabled, progress

sys.dont_write_bytecode = True
from common import (CITY_IDS, SERVER_ROOT, bearing, confined, dates_valid, digest,
                    distance, load, load_keys, load_project_environment, runtime_paths, save, turn, validate_route)


def write_jsonl(path, rows):
    path=confined(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=confined(path.with_name(path.name+'.tmp'))
    with temp.open('w',encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def database(run):
    from storage import connect
    db=connect(run)
    db.executescript('''
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,city_id TEXT,status TEXT,attempts INTEGER DEFAULT 0,data TEXT);
        CREATE TABLE IF NOT EXISTS starts(pano_id TEXT PRIMARY KEY,sample_id TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS city_starts(city_id TEXT,pano_id TEXT,sample_id TEXT UNIQUE,PRIMARY KEY(city_id,pano_id));
        CREATE TABLE IF NOT EXISTS panos(pano_id TEXT PRIMARY KEY,sample_id TEXT);
        CREATE TABLE IF NOT EXISTS metadata(id TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE IF NOT EXISTS api_attempts(id TEXT PRIMARY KEY,attempts INTEGER);
        CREATE TABLE IF NOT EXISTS requests(custom_id TEXT PRIMARY KEY,sample_id TEXT);
        CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,city_id TEXT,status TEXT,provider_id TEXT,payload_path TEXT);
        CREATE INDEX IF NOT EXISTS samples_city_status_id ON samples(city_id,status,id);
        CREATE INDEX IF NOT EXISTS batches_city_status ON batches(city_id,status);
    ''')
    return db


def update(db,sample,status):
    with db:
        db.execute('UPDATE samples SET status=?,data=? WHERE id=?',(status,json.dumps(sample,allow_nan=False),sample['sample_id']))


def renderer():
    source=confined(SERVER_ROOT/'code/trajectory/src')
    sys.path.insert(0,str(source))
    from streetview_images import render_view
    return render_view


def make_request(sample,run,config):
    from PIL import Image
    render=renderer()
    node=sample['start']
    city,sep,country=sample['name'].rpartition(',')
    if not sep:
        city,country=sample['name'],sample['name']
    prompt=(f'You are given multiple Street View images captured at the same location in {city.strip()}, {country.strip()}, '
        'each facing a different candidate heading. The images are labeled sequentially as 0, 1, and so on. '
        'Choose the candidate whose heading most likely matches the actual forward direction of the Street View survey vehicle at capture time.\n'
        'Return ONLY JSON: {"selected_index": integer, "confidence": number between 0 and 1, "reason": "brief explanation"}.')
    content=[{'type':'text','text':prompt}]
    hashes=[]
    base_url=os.environ.get('STREETVIEW_IMAGE_BASE_URL','').rstrip('/')
    if not base_url.startswith('https://'):
        raise RuntimeError('Batch requires a verified public HTTPS candidate image endpoint')
    candidate_files=[]
    sample['candidate_files']=candidate_files
    for i,link in enumerate(node['links']):
        image,meta=render(node['pano_id'],node['lat'],node['lon'],link['heading'],horizontal_fov=90,
            vertical_fov=90,output_size=(640,640),zoom=3,source_heading=node['source_pano_heading'],
            source_date=node['date'],source_world_size=node['source_pano_world_size'],source_tile_size=node['source_pano_tile_size'])
        encoded=io.BytesIO()
        image.save(encoded,format='JPEG',quality=75,optimize=True,subsampling=2)
        image.close()
        raw=encoded.getvalue()
        with Image.open(io.BytesIO(raw)) as check:
            check.load()
            if check.size!=(640,640):
                raise ValueError('Invalid candidate image')
        hashes.append(hashlib.sha256(raw).hexdigest())
        filename=secrets.token_hex(32)+'.jpg'
        file=confined(run/'work'/sample['city_id']/'public_images'/filename)
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(raw)
        candidate_files.append(str(file))
        content.extend([{'type':'text','text':str(i)}, {'type':'image_url','image_url':{
            'url':base_url+('/'+sample['city_id'] if config.get('multi_city_image_paths') else '')+'/'+filename}}])
    sample['prompt']=prompt
    sample['candidate_hashes']=hashes
    sample['candidate_files']=candidate_files
    sample['image_source']='robolyst/streetview 0.1.0 unofficial panorama tiles'
    custom='r'+digest([config['dataset_id'],sample['sample_id'],node['pano_id'],prompt,hashes,config])[:32]
    request={'custom_id':custom,'body':{'messages':[{'role':'user','content':content}],
             'reasoning':{'effort':'high'},'include_reasoning':True,'response_format':{'type':'json_object'},
             'max_tokens':20000,'stream':False}}
    if stage_enabled(run):
        sample['request']=request
    else:
        path=run/'work'/sample['city_id']/(sample['sample_id']+'.request.json')
        save(path,request)
        sample['request_path']=str(path)
    sample['request_hash']=digest(request)
    return custom


def lookup_start(maps,db,sample):
    from maps import MapsError
    try:
        request={'location':{'lat':sample['lat'],'lng':sample['lon']}}
        radius=sample.get('start_search_radius_m',100)
        if radius is not None:
            request['radius']=radius
        sample['start_search_radius_m']=radius
        sample['start']=maps.get(request)
        return sample,'pano_bound'
    except MapsError as error:
        sample['error']=str(error)
        return sample,'no_pano' if str(error)=='ZERO_RESULTS' else 'start_metadata_error'


def bind_start(db,sample,status,city_local=False):
    # Only the ordered coordinator decides which sample owns duplicate starts.
    if status=='pano_bound':
        with db:
            if city_local:
                owner=db.execute('SELECT sample_id FROM city_starts WHERE city_id=? AND pano_id=?',
                                 (sample['city_id'],sample['start']['pano_id'])).fetchone()
            else:
                owner=db.execute('SELECT sample_id FROM starts WHERE pano_id=?',(sample['start']['pano_id'],)).fetchone()
            if owner and owner[0]!=sample['sample_id']:
                sample['owner_sample_id']=owner[0]
                status='duplicate_start_pano'
            else:
                db.execute('INSERT OR IGNORE INTO starts VALUES (?,?)',(sample['start']['pano_id'],sample['sample_id']))
                if city_local:
                    db.execute('INSERT OR IGNORE INTO city_starts VALUES (?,?,?)',
                               (sample['city_id'],sample['start']['pano_id'],sample['sample_id']))
            db.execute('UPDATE samples SET status=?,data=? WHERE id=?',(status,json.dumps(sample),sample['sample_id']))
    else:
        update(db,sample,status)


def prescreen_start(maps,db,sample):
    from maps import MapsError,valid_metadata
    node=sample['start']
    if node['pano_id'].startswith('CAoS'):
        return sample,'tile_incompatible_start_pano_id'
    if not node['links']:
        return sample,'no_navigation_links'
    if not valid_metadata(node):
        return sample,'start_metadata_error'
    viable=False
    errors=[]
    for link in node['links']:
        if link['pano_id']==node['pano_id']:
            continue
        try:
            target=maps.get({'pano':link['pano_id']})
            if (valid_metadata(target) and not target['pano_id'].startswith('CAoS')
                    and 1<distance(node,target)<30 and dates_valid([node,target])):
                viable=True
                break
        except MapsError as error:
            errors.append(str(error))
    if not viable:
        sample['prescreen_errors']=errors
        return sample,'start_metadata_error' if errors else 'no_viable_initial_link'
    return sample,'start_eligible'


def prepare_candidate(sample,run,config):
    attempt=int(sample.get('candidate_attempts',0))+1
    sample['candidate_attempts']=attempt
    try:
        custom=make_request(sample,run,config)
    except Exception as error:
        # Tile helper already does 4 attempts on the same tile. No replacement.
        sample['candidate_error_type']=type(error).__name__
        # Keep only exception classes/status, never URLs, headers or credentials.
        chain=[]
        cause=error
        while cause is not None and len(chain)<6:
            chain.append(type(cause).__name__)
            cause=cause.__cause__ or cause.__context__
        sample['candidate_error_chain']=chain
        for name in sample.get('candidate_files',[]):
            path=confined(Path(name))
            if path.parent==confined(run/'work'/sample['city_id']/'public_images'):
                path.unlink(missing_ok=True)
        sample['candidate_files']=[]
        return sample,'candidate_image_error',None
    sample.pop('candidate_error_type',None)
    sample.pop('candidate_error_chain',None)
    sample.pop('candidate_error',None)
    return sample,'candidates_ready',custom


class SearchBudget(RuntimeError):
    pass


def search(sample,maps,db,config,used=None):
    from maps import MapsError,valid_metadata
    start=sample['start']
    selected=sample['decision']['selected_pano_id']
    if used is None:
        used={r[0] for r in db.execute('SELECT pano_id FROM panos')}
    if start['pano_id'] in used or selected in used:
        return None,'global_pano_conflict'
    budget=sample.setdefault('search_budget',{'expansions':0,'seconds':0})
    previous=time.monotonic()
    metadata_errors=[]
    conflicts=False
    def tick(expansion=True):
        nonlocal previous
        now=time.monotonic()
        budget['seconds']+=now-previous
        previous=now
        if expansion:
            budget['expansions']+=1
        update(db,sample,'llm_selected')
        if budget['expansions']>config['max_expansions'] or budget['seconds']>config['search_seconds']:
            raise SearchBudget()
    def dfs(nodes,edges):
        nonlocal conflicts
        tick()
        node=nodes[-1]
        if len(nodes)==8:
            return (nodes,edges) if dates_valid(nodes,final=True) else None
        links=node['links'] if len(nodes)>1 else [l for l in node['links'] if l['pano_id']==selected]
        visited={n['pano_id'] for n in nodes}
        for link in links:
            target_id=link['pano_id']
            if target_id in used:
                conflicts=True
                continue
            if target_id in visited or (edges and turn(edges[-1]['heading'],link['heading'])>90):
                continue
            try:
                target=maps.get({'pano':target_id})
            except MapsError as error:
                metadata_errors.append(str(error))
                continue
            tick(False)
            if not valid_metadata(target) or target['pano_id'].startswith('CAoS'):
                continue
            if not 1<distance(node,target)<30 or not dates_valid(nodes+[target]):
                continue
            result=dfs(nodes+[target],edges+[link])
            if result:
                return result
        return None
    try:
        target=maps.get({'pano':selected})
        tick(False)
        if not valid_metadata(target) or not 1<distance(start,target)<30 or not dates_valid([start,target]):
            return None,'llm_selected_nonviable_link'
        result=dfs([start],[])
    except SearchBudget:
        return None,'search_budget_exceeded'
    except MapsError as error:
        sample['navigation_error']=str(error)
        return None,'navigation_request_error'
    sample['navigation_errors']=metadata_errors
    if not result:
        return None,('navigation_request_error' if metadata_errors else 'global_pano_conflict' if conflicts else 'no_valid_8_frame_path')
    nodes,edges=result
    frames=[]
    for i,node in enumerate(nodes):
        frames.append({**node,'frame':i,'file':f'frame_{i:02d}.jpg',
                       'heading':edges[i]['heading'] if i<7 else edges[6]['heading'],
                       'render_heading':bearing(node,nodes[i+1]) if i<7 else frames[6]['render_heading'],
                       'render_pitch':0,'next_pano_id':nodes[i+1]['pano_id'] if i<7 else None,
                       'next_in_sequence':i<7})
    validate_route(frames,start['pano_id'],selected)
    return frames,None


def image_check(folder,manifest):
    from PIL import Image
    for frame in manifest['frames']:
        file=confined(folder/frame['file'])
        if hashlib.sha256(file.read_bytes()).hexdigest()!=frame['sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(file) as image:
            image.load()
            if image.format!='JPEG' or image.size!=(1920,960):
                raise ValueError('Image decode/dimensions failed')


def render_sample(sample,run,city_dir):
    final=confined(city_dir/'trajectory_sequences_180'/sample['sample_id'])
    frames=sample['frames']
    if final.exists():
        manifest=load(final/'trajectory_180.json')
        if manifest['navigation_hash']!=digest(frames):
            raise RuntimeError('Existing output differs from frozen navigation')
        image_check(final,manifest)
        return
    staging=confined(run/'tmp'/('render_'+sample['sample_id']))
    staging.mkdir(parents=True,exist_ok=True)
    render=renderer()
    rendered=[]
    for frame in frames:
        image,meta=render(frame['pano_id'],frame['lat'],frame['lon'],frame['render_heading'],
            horizontal_fov=180,vertical_fov=90,output_size=(1920,960),zoom=3,
            source_heading=frame['source_pano_heading'],source_date=frame['date'],
            source_world_size=frame['source_pano_world_size'],source_tile_size=frame['source_pano_tile_size'])
        file=confined(staging/frame['file'])
        image.save(file,format='JPEG',quality=92,subsampling=0)
        image.close()
        rendered.append({**frame,**meta,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    manifest={'id':sample['sample_id'],'name':sample['name'],'navigation_hash':digest(frames),
              'image_source':'robolyst/streetview 0.1.0 unofficial panorama tiles','frames':rendered}
    save(staging/'trajectory_180.json',manifest)
    image_check(staging,manifest)
    final.parent.mkdir(parents=True,exist_ok=True)
    staging.rename(final)


def finish_sample(sample_id,maps,db,run,output,config):
    status,raw=db.execute('SELECT status,data FROM samples WHERE id=?',(sample_id,)).fetchone()
    sample=json.loads(raw)
    rank=CITY_IDS.index(sample['city_id'])+1
    city_dir=confined(output/'cities'/f'{rank:02d}_{sample["city_id"]}')
    city_dir.mkdir(parents=True,exist_ok=True)
    if status=='llm_selected':
        frames,error=search(sample,maps,db,config)
        if error:
            update(db,sample,error)
            return
        sample['frames']=frames
        with db:
            for frame in frames:
                db.execute('INSERT INTO panos VALUES (?,?)',(frame['pano_id'],sample_id))
            db.execute('UPDATE samples SET status=?,data=? WHERE id=?',('navigation_accepted',json.dumps(sample),sample_id))
        status='navigation_accepted'
    if status=='navigation_accepted':
        try:
            render_sample(sample,run,city_dir)
            update(db,sample,'complete')
        except Exception as error:
            sample['render_error_type']=type(error).__name__
            update(db,sample,'render_error')


def search_task(maps,db,job):
    sample,config,used=job
    if 'search_proposal' not in sample:
        frames,error=search(sample,maps,db,config,used)
        sample['search_proposal']={'frames':frames,'error':error}
        # A finished speculative search survives a crash before ordered acceptance.
        update(db,sample,'llm_selected')
    return sample


def accept_proposal(db,sample):
    proposal=sample['search_proposal']
    frames,error=proposal['frames'],proposal['error']
    if error:
        sample.pop('search_proposal')
        update(db,sample,error)
        return True
    validate_route(frames,sample['start']['pano_id'],sample['decision']['selected_pano_id'])
    with db:
        # The dataset flock admits one coordinator. Workers never reserve panos.
        used={row[0] for row in db.execute('SELECT pano_id FROM panos')}
        if any(frame['pano_id'] in used for frame in frames):
            return False
        sample.pop('search_proposal')
        sample['frames']=frames
        for frame in frames:
            db.execute('INSERT INTO panos VALUES (?,?)',(frame['pano_id'],sample['sample_id']))
        db.execute('UPDATE samples SET status=?,data=? WHERE id=?',
                   ('navigation_accepted',json.dumps(sample),sample['sample_id']))
    return True


def finish_parallel(ids,db,run,output,config,workers,resources=None):
    from parallel import MapsPool,ordered_window
    resources=resources or {}
    used=frozenset(row[0] for row in db.execute('SELECT pano_id FROM panos'))
    jobs=[]
    accepted=[]
    for sample_id in sorted(ids):
        status,raw=db.execute('SELECT status,data FROM samples WHERE id=?',(sample_id,)).fetchone()
        sample=json.loads(raw)
        if status=='llm_selected':
            jobs.append((sample,config,used))
        elif status=='navigation_accepted':
            accepted.append(sample)
    stage_mode=stage_enabled(run)
    cities={json.loads(db.execute('SELECT data FROM samples WHERE id=?',(sid,)).fetchone()[0])['city_id'] for sid in ids}
    if stage_mode:
        for city in cities:
            save(run/'state'/f'preparation_{city}.json',{'city_id':city,'stage':'searching_paths','updated_at':time.time()})
    pending={}
    with ThreadPoolExecutor(max_workers=workers['render'],thread_name_prefix='render') as render_pool:
        def drain(block=False):
            if not pending:
                return
            done,_=wait(pending,timeout=None if block else 0,return_when=FIRST_COMPLETED)
            for future in done:
                sample=pending.pop(future)
                try:
                    future.result()
                    update(db,sample,'complete')
                except Exception as error:
                    sample['render_error_type']=type(error).__name__
                    update(db,sample,'render_error')

        def enqueue(sample):
            while len(pending)>=workers['render']*2:
                drain(True)
            city_dir=confined(output/'cities'/f'{CITY_IDS.index(sample["city_id"])+1:02d}_{sample["city_id"]}')
            pending[render_pool.submit(render_sample,sample,run,city_dir)]=sample

        if not stage_mode:
            for sample in accepted:
                enqueue(sample)
        with nullcontext(resources['maps']) if 'maps' in resources else MapsPool(run,workers['maps']) as pool:
            for index,(_,sample) in enumerate(ordered_window(pool,search_task,jobs,workers['maps']*2),1):
                while not accept_proposal(db,sample):
                    # Preserve the selected first hop and cumulative search budget.
                    sample.pop('search_proposal')
                    update(db,sample,'llm_selected')
                    current=frozenset(row[0] for row in db.execute('SELECT pano_id FROM panos'))
                    sample=pool.submit(search_task,(sample,config,current)).result()
                if 'frames' in sample:
                    if stage_mode:
                        accepted.append(sample)
                    else:
                        enqueue(sample)
                drain()
                if index%25==0:
                    print(f'Parallel search confirmed {index}/{len(jobs)}; render queue {len(pending)}',flush=True)
        if stage_mode:
            for city in cities:
                checkpoint(db,city,'paths_completed')
                save(run/'state'/f'preparation_{city}.json',{'city_id':city,'stage':'rendering','updated_at':time.time()})
            for sample in accepted:
                enqueue(sample)
        while pending:
            drain(True)
    for city in cities:
        checkpoint(db,city,'rendering_completed')


def lookup_parallel(db,run,city_id,workers,radius=100,resources=None):
    from parallel import MapsPool,ordered_window
    resources=resources or {}
    with nullcontext(resources['maps']) if 'maps' in resources else MapsPool(run,workers['maps']) as pool:
        # Resolve all fixed points before any prescreening or candidate rendering.
        rows=[json.loads(raw) for raw, in db.execute(
            "SELECT data FROM samples WHERE city_id=? AND status='sampled' ORDER BY id",(city_id,))]
        for sample in rows:
            sample['start_search_radius_m']=radius
        for index,(_,result) in enumerate(ordered_window(pool,lookup_start,rows,workers['maps']*2),1):
            with resources.get('binding',nullcontext()):
                bind_start(db,*result,city_local=bool(resources))
            if index%100==0:
                print(f'{city_id}: start lookup {index}/{len(rows)}',flush=True)


def prescreen_parallel(db,run,city_id,workers,resources=None):
    from parallel import MapsPool,ordered_window
    resources=resources or {}
    with nullcontext(resources['maps']) if 'maps' in resources else MapsPool(run,workers['maps']) as pool:
        rows=[json.loads(raw) for raw, in db.execute(
            "SELECT data FROM samples WHERE city_id=? AND status='pano_bound' ORDER BY id",(city_id,))]
        for _,(sample,status) in ordered_window(pool,prescreen_start,rows,workers['maps']*2):
            update(db,sample,status)


def prepare_parallel(db,run,city_id,workers):
    # Compatibility wrapper for the single-round stage test.
    lookup_parallel(db,run,city_id,workers)
    prescreen_parallel(db,run,city_id,workers)


def acquire_starts(db,run,output,city_id,config,workers,resources=None):
    from sampling import SAMPLING_POLICY,extend_city_round,refresh_manifest
    resources=resources or {}
    path=run/'state'/f'acquisition_{city_id}.json'
    if path.exists():
        state=load(path)
    else:
        queried=db.execute("SELECT count(*) FROM samples WHERE city_id=? AND status!='sampled'",(city_id,)).fetchone()[0]
        policy=dict(config.get('sampling_policy',SAMPLING_POLICY))
        if queried:
            # Never reinterpret a previously running legacy city as a new experiment.
            policy={'version':1,'radius_m':None,'points_per_round':1500,'max_rounds':1,
                    'target_start_panos':None,'count_mode':'unique_panos','stop_check':'after_completed_round'}
        state={'city_id':city_id,'policy':policy,'rounds':[],'complete':False}
        save(path,state)
    policy=state['policy']
    first_round=max((r['round'] for r in state['rounds']),default=0)+1
    for round_number in range(first_round,policy['max_rounds']+1):
        if state['complete']:
            break
        progress(run,city_id,'sampling',round=round_number)
        if round_number>1:
            extend_city_round(run,city_id,policy,round_number)
        frozen=load(run/'state'/f'samples_{city_id}.json')
        points=frozen['points'][:round_number*policy['points_per_round']]
        with db:
            for point in points:
                db.execute('INSERT OR IGNORE INTO samples(id,city_id,status,data) VALUES (?,?,?,?)',
                    (point['sample_id'],city_id,'sampled',json.dumps(point)))
        with resources.get('manifest',nullcontext()):
            refresh_manifest(run,output,config)
        checkpoint(db,city_id,f'sampling_round_{round_number}')
        progress(run,city_id,'start_lookup',round=round_number)
        # Completed rounds resume from snapshots; unfinished lookups replay.
        if resources:
            lookup_parallel(db,run,city_id,workers,policy['radius_m'],resources)
        else:
            lookup_parallel(db,run,city_id,workers,policy['radius_m'])
        if resources:
            returned=db.execute('SELECT count(*) FROM city_starts WHERE city_id=?',(city_id,)).fetchone()[0]
        else:
            returned=db.execute('SELECT count(*) FROM starts JOIN samples ON starts.sample_id=samples.id WHERE samples.city_id=?',
                                (city_id,)).fetchone()[0]
        record={'round':round_number,'cumulative_sampled_points':len(points),'unique_start_panos':returned}
        state['rounds']=[r for r in state['rounds'] if r['round']!=round_number]+[record]
        target=policy['target_start_panos']
        reached=target is not None and returned>=target
        state.update(unique_start_panos=returned,sampled_points=len(points),target_reached=reached,
                     complete=reached or round_number==policy['max_rounds'])
        state['stop_reason']='target_reached' if reached else 'max_rounds_reached' if state['complete'] else None
        save(path,state)
        checkpoint(db,city_id,f'start_lookup_round_{round_number}')
        print(f'{city_id}: round {round_number}/{policy["max_rounds"]}, {returned} unique start panos; {state["stop_reason"] or "another round needed"}',flush=True)
    return state


def candidates_parallel(ids,db,run,config,workers,resources=None):
    from parallel import ordered_window
    source_path=run/'state/reuse_source_decisions.json'
    source=load(source_path) if source_path.exists() else {}
    rows=[]
    for sample_id in sorted(ids):
        status,raw=db.execute('SELECT status,data FROM samples WHERE id=?',(sample_id,)).fetchone()
        if status=='start_eligible' or (status=='candidate_image_error' and int(json.loads(raw).get('candidate_attempts',0))<3):
            rows.append(json.loads(raw))
    resources=resources or {}
    with nullcontext(resources['candidates']) if 'candidates' in resources else ThreadPoolExecutor(max_workers=workers['render'],thread_name_prefix='candidate') as pool:
        def prepare(sample):
            return prepare_candidate(sample,run,config)
        for index,(_,result) in enumerate(ordered_window(pool,prepare,rows,workers.get('candidate',workers['render'])*2),1):
            sample,status,custom=result
            if status=='candidates_ready' and reuse_decision(sample,source,config):
                status='llm_selected'
            with db:
                if custom:
                    db.execute('INSERT OR IGNORE INTO requests VALUES (?,?)',(custom,sample['sample_id']))
                db.execute('UPDATE samples SET status=?,data=? WHERE id=?',(status,json.dumps(sample),sample['sample_id']))
            if index%25==0:
                print(f'{sample["city_id"]}: candidate images {index}/{len(rows)}',flush=True)


def reuse_decision(sample,source,config):
    """Reuse only an audited decision for byte-identical images and identical inputs."""
    import copy
    old=source.get('samples',{}).get(sample['sample_id'])
    if not old or 'decision' not in old:
        return False
    if any(source.get('config',{}).get(k)!=config.get(k)
           for k in ('base_model','model','reasoning_effort','max_tokens')):
        return False
    if any(sample.get(k)!=old.get(k) for k in ('start','prompt','candidate_hashes')):
        return False
    if not sample.get('candidate_hashes'):
        return False
    sample['decision']=copy.deepcopy(old['decision'])
    sample['batch_history']=copy.deepcopy(old.get('batch_history',[]))
    sample['decision_reuse']={'source_dataset':source['source_dataset'],
        'source_sample_id':old['sample_id'],'source_request_hash':old['request_hash'],
        'verified':'identical start metadata, prompt, candidate JPEG hashes and model parameters'}
    return True


def cleanup_city(db,run,city_id):
    # Delete only enumerated intermediate artifacts belonging to this city.
    paths=[]
    for raw, in db.execute('SELECT data FROM samples WHERE city_id=?',(city_id,)):
        sample=json.loads(raw)
        if sample.get('request_path'):
            paths.append(Path(sample['request_path']))
        paths.extend(Path(p) for p in sample.get('candidate_files',[]))
        staging=confined(run/'tmp'/('render_'+sample['sample_id']))
        for i in range(8):
            paths.append(staging/f'frame_{i:02d}.jpg')
        paths.append(staging/'trajectory_180.json')
    for path, in db.execute('SELECT payload_path FROM batches WHERE city_id=?',(city_id,)):
        paths.extend([Path(path),Path(path).with_suffix('.result.json')])
    for path in paths:
        path=confined(path)
        if path.is_file():
            path.unlink()
    for raw, in db.execute('SELECT data FROM samples WHERE city_id=?',(city_id,)):
        sample=json.loads(raw)
        staging=confined(run/'tmp'/('render_'+sample['sample_id']))
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()


def summarize(db,run,output,city_id):
    rank=CITY_IDS.index(city_id)+1
    city_dir=confined(output/'cities'/f'{rank:02d}_{city_id}')
    city_dir.mkdir(parents=True,exist_ok=True)
    rows=[(status,json.loads(raw)) for status,raw in db.execute('SELECT status,data FROM samples WHERE city_id=? ORDER BY id',(city_id,))]
    counts=Counter(status for status,_ in rows)
    transient={'sampled','pano_bound','start_eligible','candidates_ready','llm_selected','navigation_accepted'}
    pending=sum(n for status,n in counts.items() if status in transient)
    successful=[s for status,s in rows if status=='complete']
    index=[]
    seen=set()
    for sample in successful:
        validate_route(sample['frames'],sample['start']['pano_id'],sample['decision']['selected_pano_id'])
        manifest=load(city_dir/'trajectory_sequences_180'/sample['sample_id']/'trajectory_180.json')
        image_check(city_dir/'trajectory_sequences_180'/sample['sample_id'],manifest)
        ids={f['pano_id'] for f in sample['frames']}
        if seen & ids:
            raise ValueError('Final pano conflict')
        seen.update(ids)
        index.append({'id':sample['sample_id'],'path':sample['sample_id']+'/trajectory_180.json'})
    summary={'city_id':city_id,'sampled':len(rows),'complete':len(successful),'pending':pending,
             'terminal_failed':len(rows)-len(successful)-pending,'outcomes':dict(counts),
             'image_count':len(successful)*8,'unique_panos':len(seen),
             'generation_rate':len(successful)/len(rows) if rows else None,
             'minimum_sample_distance_m':load(run/'state'/f'samples_{city_id}.json')['minimum_distance_m']}
    usage=Counter()
    counted=set()
    for _,sample in rows:
        if sample.get('decision_reuse'):
            continue
        for attempt in sample.get('batch_history',[]):
            identity=(attempt['batch_id'],attempt['custom_id'])
            if identity in counted:
                continue
            counted.add(identity)
            body=(attempt.get('response',{}).get('response') or {}).get('body') or {}
            for key in ('prompt_tokens','completion_tokens','total_tokens','cost'):
                value=(body.get('usage') or {}).get(key)
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    usage[key]+=value
    summary['llm_usage']=dict(usage)
    summary['reused_llm_decisions']=sum(bool(s.get('decision_reuse')) for _,s in rows)
    acquisition_path=run/'state'/f'acquisition_{city_id}.json'
    acquisition=load(acquisition_path) if acquisition_path.exists() else None
    if acquisition:
        summary['acquisition']=acquisition
    write_jsonl(city_dir/'sample_outcomes.jsonl',[{'sample_id':s['sample_id'],'query':{'lat':s['lat'],'lon':s['lon']},'status':status,
                 'start_pano_id':s.get('start',{}).get('pano_id'),'owner_sample_id':s.get('owner_sample_id')} for status,s in rows])
    write_jsonl(city_dir/'accepted_navigation.jsonl',[{'id':s['sample_id'],'frames':s['frames'],'decision':s['decision']} for s in successful])
    write_jsonl(city_dir/'start_heading_audit.jsonl',[s for _,s in rows if s.get('request_path') or s.get('request')])
    (city_dir/'trajectory_sequences_180').mkdir(exist_ok=True)
    save(city_dir/'trajectory_sequences_180/index.json',index)
    expected=load(run/'state'/f'samples_{city_id}.json')['points']
    complete=(pending==0 and {s['sample_id'] for _,s in rows}=={s['sample_id'] for s in expected}
              and (acquisition['complete'] if acquisition else len(rows)==1500))
    summary['city_complete']=complete
    save(city_dir/'summary.json',summary)
    if complete:
        inspection=confined(city_dir/'inspection.md')
        inspection.write_text(f'# {city_id} — first-city inspection\n\n'
            f'{len(rows)} fixed points; {len(successful)} complete trajectories; {len(successful)*8} JPEG images.\n\n'
            f'Minimum sampled-point separation: {summary["minimum_sample_distance_m"]:.3f} m.\n\n'
            '[Statistics](summary.json) · [Trajectory index](trajectory_sequences_180/index.json)\n\n'
            'State: awaiting_review. No subsequent city is started by this command.\n',encoding='utf8')
    global_index_path=output/'index.json'
    global_index=load(global_index_path) if global_index_path.exists() else []
    if complete and not any(x['city_id']==city_id for x in global_index):
        global_index.append({'city_id':city_id,'path':f'cities/{rank:02d}_{city_id}/inspection.md','complete':len(successful)})
    save(global_index_path,global_index)
    save(output/'summary.json',{'dataset_complete':len(global_index)==30,'state':'awaiting_review' if complete else 'city_in_progress',
          'completed_cities':len(global_index),'total_cities':30,
          'frozen_points':len(load(run/'state/frozen.json')['samples']),'latest_city':summary})
    return complete


def run_city(config,samples,run,output,city_id,workers=None):
    from batch import Batch
    workers=workers or {"maps":4,"render":4,"tiles":8}
    db=database(run)
    rank=CITY_IDS.index(city_id)+1
    if rank>1:
        # Continuation is deliberately explicit, never auto-advance after first city.
        approval=run/'state/continuation.json'
        if not approval.exists() or load(approval).get('approved') is not True:
            raise RuntimeError('First-city inspection required before subsequent cities')
        previous=output/'cities'/f'{rank-1:02d}_{CITY_IDS[rank-2]}'/'summary.json'
        if not previous.exists() or load(previous).get('pending')!=0:
            raise RuntimeError('Previous city has not completed')
    city_samples=[s for s in samples if s['city_id']==city_id]
    with db:
        for sample in city_samples:
            db.execute('INSERT OR IGNORE INTO samples(id,city_id,status,data) VALUES (?,?,?,?)',
                (sample['sample_id'],city_id,'sampled',json.dumps(sample)))
    city_dir=confined(output/'cities'/f'{rank:02d}_{city_id}')
    city_dir.mkdir(parents=True,exist_ok=True)
    try:
        # Concurrency is operational; never change frozen model/sampling settings.
        renderer()
        from streetview_images import configure_downloads
        configure_downloads(workers['tiles'])
        save(run/'state/runtime_concurrency.json',workers)
        print(f'Parallel pipeline: maps={workers["maps"]}, render={workers["render"]}, global tiles={workers["tiles"]}',flush=True)
        acquire_starts(db,run,output,city_id,config,workers)
        prescreen_parallel(db,run,city_id,workers)
        checkpoint(db,city_id,'prescreen_completed')
        ids=[r[0] for r in db.execute('SELECT id FROM samples WHERE city_id=? ORDER BY id',(city_id,))]
        pilot_path=run/'state'/f'pilot_{city_id}.json'
        pilot_ok=pilot_path.exists() and load(pilot_path).get('complete_trajectories',0)>0
        eligible=[r[0] for r in db.execute("SELECT id FROM samples WHERE city_id=? AND status IN ('start_eligible','candidates_ready','llm_selected','navigation_accepted','complete') ORDER BY id",(city_id,))]
        if rank==1 and not pilot_ok and (eligible or pilot_path.exists()):
            pilot_ids=load(pilot_path)['sample_ids'] if pilot_path.exists() else eligible[:10]
            if not pilot_path.exists():
                save(pilot_path,{'sample_ids':pilot_ids,'valid_decisions':0,'complete_trajectories':0})
            candidates_parallel(pilot_ids,db,run,config,workers)
            checkpoint(db,city_id,'pilot_candidates_completed')
            Batch(run,db,config,city_id).run_group(pilot_ids)
            valid=sum('decision' in json.loads(db.execute('SELECT data FROM samples WHERE id=?',(sid,)).fetchone()[0]) for sid in pilot_ids)
            if not valid:
                raise RuntimeError('No valid pilot decisions; do not scale an unverified Batch')
            finish_parallel(pilot_ids,db,run,output,config,workers)
            complete=sum(db.execute('SELECT status FROM samples WHERE id=?',(sid,)).fetchone()[0]=='complete' for sid in pilot_ids)
            save(pilot_path,{'sample_ids':pilot_ids,'valid_decisions':valid,'complete_trajectories':complete})
            if not complete:
                raise RuntimeError('Pilot has no complete images; inspect before scaling')
        candidates_parallel(ids,db,run,config,workers)
        checkpoint(db,city_id,'candidates_completed')
        Batch(run,db,config,city_id).run_group(ids)
        finish_parallel(ids,db,run,output,config,workers)
        if not summarize(db,run,output,city_id):
            raise RuntimeError('City still has pending tasks')
        checkpoint(db,city_id,'complete')
        cleanup_city(db,run,city_id)
        print(f'{city_id}: complete; awaiting_review. Subsequent cities NOT started.',flush=True)
    finally:
        db.close()



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('--dataset',default='sv30_20260920_r100_r3')
    parser.add_argument('--seed',default='sv30-fixed-20260919-v1')
    parser.add_argument('--city',choices=CITY_IDS,default=CITY_IDS[0])
    parser.add_argument('--maps-workers',type=int,default=4)
    parser.add_argument('--render-workers',type=int,default=4)
    parser.add_argument('--tile-workers',type=int,default=8)
    args=parser.parse_args()
    if any(not 1 <= n <= 32 for n in (args.maps_workers,args.render_workers,args.tile_workers)):
        parser.error('Worker counts must be in [1,32]')
    run,output=runtime_paths(args.dataset)
    import tempfile
    tempfile.tempdir=str(run/'tmp')
    os.environ['XDG_CONFIG_HOME']=str(run/'tmp/config')
    os.environ['XDG_DATA_HOME']=str(run/'tmp/data')
    if args.action=='status':
        state=confined(run/'state/generation.sqlite')
        if stage_enabled(run):
            from stage_store import read_status
            result=read_status(run,args.city)
            result['counts_as_of']='last_city_checkpoint'
        elif state.exists():
            db=sqlite3.connect(state.as_uri()+'?mode=ro',uri=True,timeout=5)
            try:
                result={'city_id':args.city,
                    'sample_states':dict(db.execute('SELECT status,count(*) FROM samples WHERE city_id=? GROUP BY status',(args.city,))),
                    'batch_states':dict(db.execute('SELECT status,count(*) FROM batches WHERE city_id=? GROUP BY status',(args.city,)))}
            finally:
                db.close()
        else:
            result={'state':'preparing' if (output/'dataset_config.json').exists() else 'not_started'}
        summary=output/'summary.json'
        if summary.exists():
            result['summary']=load(summary)
        acquisition=run/'state'/f'acquisition_{args.city}.json'
        if acquisition.exists():
            result['acquisition']=load(acquisition)
        full_generation=run/'state/full_generation.json'
        if full_generation.exists():
            result['full_generation']=load(full_generation)
            result['city_preparation']={path.stem.removeprefix('preparation_'):load(path)
                                        for path in sorted((run/'state').glob('preparation_*.json'))}
        print(json.dumps(result,ensure_ascii=False))
        return
    import fcntl
    lock=confined(run/'state/run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another process owns this dataset') from None
    from sampling import prepare
    load_project_environment()
    try:
        if args.action=='run':
            previous_summary=output/'cities'/f'{CITY_IDS.index(args.city)+1:02d}_{args.city}'/'summary.json'
            if previous_summary.exists():
                completed=load(previous_summary)
                if (completed.get('pending')==0 and completed.get('sampled',0)>0
                        and completed.get('city_complete',completed.get('sampled')==1500)):
                    print(f'{args.city}: already complete; existing results and acquisition policy preserved.',flush=True)
                    return
            load_keys()
            if not os.environ.get('STREETVIEW_IMAGE_BASE_URL','').startswith('https://'):
                raise RuntimeError('Luna Batch requires an approved HTTPS candidate-image endpoint; configure before resuming')
        config,samples=prepare(run,output,args.dataset,args.seed)
        if args.action=='run':
            from stage_store import initialize
            initialize(run)
            run_city(config,samples,run,output,args.city,{'maps':args.maps_workers,'render':args.render_workers,'tiles':args.tile_workers})
        blocker=confined(run/'state/blocker.json')
        if blocker.exists():
            blocker.unlink()
    except Exception as error:
        save(run/'state/blocker.json',{'state':'blocked','stage':args.action,'city_id':args.city,
             'error_type':type(error).__name__,'updated_at':time.time()})
        raise


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        # Deliberately omit traceback/provider response bodies from shared logs.
        print(f'BLOCKED: {type(error).__name__}: {error}',file=sys.stderr,flush=True)
        sys.exit(1)
