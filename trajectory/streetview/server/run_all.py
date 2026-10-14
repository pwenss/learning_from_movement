"""Prepare cities concurrently with city Batches and rolling generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
from threading import Lock
import time
from urllib.parse import urlsplit
import socket

sys.dont_write_bytecode=True
from common import CITY_IDS, SERVER_ROOT, confined, load, load_keys, runtime_paths, save
from batch import Batch, BatchRateLimit, BatchCreditLimit, BatchSlotsFull, BatchPollError, inflight_count, effective_request_limit, effective_inflight_limit
from batch_tunnels import BatchTunnels, TunnelUnavailable
from parallel import MapsPool
from storage import connect, error_details
from stage_store import initialize, checkpoint as save_city_stage, prune_candidates, durable_save
from run_city import (database, acquire_starts, prescreen_parallel, candidates_parallel,
                      finish_parallel, summarize, cleanup_city, renderer, lookup_parallel)


def complete_city(output, city):
    path=output/'cities'/f'{CITY_IDS.index(city)+1:02d}_{city}'/'summary.json'
    if not path.exists():
        return False
    row=load(path)
    return row.get('city_complete') is True and row.get('pending')==0


def verify_browser_proxy():
    value=os.environ.get('STREETVIEW_BROWSER_PROXY') or os.environ.get('HTTPS_PROXY')
    if not value:
        raise RuntimeError('Maps browser proxy is not configured; restore the SSH reverse tunnel first')
    parsed=urlsplit(value)
    if not parsed.hostname or not parsed.port:
        raise RuntimeError('Maps browser proxy URL is invalid')
    sock=socket.create_connection((parsed.hostname,parsed.port),timeout=5)
    sock.close()


def process_alive(record):
    result=subprocess.run(['ps','-p',str(record['pid']),'-o','args='],capture_output=True,text=True)
    return result.returncode==0 and result.stdout.strip()==record['command']


def start_process(command, log, env):
    with log.open('ab') as stream:
        process=subprocess.Popen(command,cwd=SERVER_ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return {'pid':process.pid,'command':' '.join(command)}


def ensure_services(run, dataset):
    """Keep the same public URL across coordinator restarts and pending Batches."""
    import requests
    probe_session=requests.Session()
    probe_session.trust_env=False
    record_path=run/'state/multi_city_services.json'
    env=dict(os.environ)
    # The browser proxy is only for Maps JavaScript. Tile and public-image
    # probes use direct connections and must not inherit HTTPS_PROXY.
    env['STREETVIEW_TILE_DIRECT']='1'
    transport_path=run/'state/image_transport.json'
    transport=load(transport_path) if transport_path.exists() else {}
    r2_base=None
    if transport.get('mode')=='r2':
        from r2_images import load_settings
        settings=load_settings(run,transport)
        r2_base=settings['public_base_url']
        legacy=transport.get('legacy_named_transport')
        if not legacy:
            os.environ['STREETVIEW_IMAGE_BASE_URL']=r2_base
            probe_session.close()
            return
        # Existing accepted HTTPS payloads still need their original origin.
        transport=legacy
    state=load(record_path) if record_path.exists() else {}
    if state.get('closed_at'):
        state={}
    server=state.get('server')
    if not server or not process_alive(server):
        command=['python3','-S',str(SERVER_ROOT/'code/trajectory/server/serve_candidates.py'),
                 '--dataset',dataset,'--all-cities']
        state['server']=start_process(command,run/'logs/multi_city_images.log',env)
        save(record_path,state)
    if transport.get('mode')=='named_tunnel':
        named_path=run/'state/named_image_services.json'
        named=load(named_path) if named_path.exists() else {}
        if not named.get('server') or not process_alive(named['server']):
            command=['python3','-S',str(SERVER_ROOT/'code/trajectory/server/serve_candidates.py'),
                     '--dataset',dataset,'--all-cities','--port',str(transport['origin_port'])]
            named['server']=start_process(command,run/'logs/named_images.log',env)
            save(named_path,named)
        supervisor=run/'state/named_tunnel.json'
        current=load(supervisor) if supervisor.exists() else {}
        command=['python3','-S',str(SERVER_ROOT/'code/trajectory/server/keep_named_tunnel.py'),
                 '--dataset',dataset]
        record={'pid':current.get('supervisor_pid',0),'command':' '.join(command)}
        if not process_alive(record):
            named['supervisor']=start_process(command,run/'logs/named_tunnel_supervisor.log',env)
            save(named_path,named)
        # Batch.prepare verifies a real JPEG at the configured hostname before
        # every new POST. Old Quick Tunnels continue serving accepted payloads.
        os.environ['STREETVIEW_IMAGE_BASE_URL']=r2_base or transport['base_url']
        probe_session.close()
        return
    tunnel=state.get('tunnel')
    if not tunnel or not process_alive(tunnel):
        if state.get('url'):
            raise RuntimeError('Candidate tunnel stopped; existing Batch URLs need reconciliation before a new tunnel')
        log=run/'logs/multi_city_tunnel.log'
        # New service state has no accepted URLs; discard only its own old log.
        log.write_text('')
        command=[str(SERVER_ROOT/'env/cloudflared'),'tunnel','--config',str(SERVER_ROOT/'config/cloudflared.yml'),
                 '--no-autoupdate','--protocol','http2','--url','http://127.0.0.1:18237',
                 '--metrics','127.0.0.1:18238']
        state['tunnel']=start_process(command,log,env)
        save(record_path,state)
    if not state.get('url'):
        for _ in range(60):
            text=(run/'logs/multi_city_tunnel.log').read_text(errors='replace')
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',text)
            if match:
                state['url']=match[0];save(record_path,state);break
            if not process_alive(state['tunnel']):
                raise RuntimeError('Candidate tunnel failed to start')
            time.sleep(2)
        else:
            raise RuntimeError('Candidate tunnel URL unavailable')
    # A 404 proves the origin is reachable without exposing a directory listing.
    for _ in range(15):
        try:
            response=probe_session.get(state['url']+'/healthcheck',timeout=20)
            if response.status_code==404 and process_alive(state['server']) and process_alive(state['tunnel']):
                # Check an actual JPEG through the tunnel, then remove the probe.
                import io
                import secrets
                from PIL import Image
                buffer=io.BytesIO();Image.new('RGB',(1,1),(0,0,0)).save(buffer,format='JPEG')
                name=secrets.token_hex(32)+'.jpg'
                probe=confined(run/'work'/CITY_IDS[1]/'public_images'/name)
                probe.parent.mkdir(parents=True,exist_ok=True);probe.write_bytes(buffer.getvalue())
                try:
                    fetched=probe_session.get(state['url']+'/'+CITY_IDS[1]+'/'+name,timeout=30)
                    if fetched.status_code!=200 or fetched.content!=buffer.getvalue():
                        continue
                finally:
                    probe.unlink(missing_ok=True)
                os.environ['STREETVIEW_IMAGE_BASE_URL']=state['url']
                probe_session.close()
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    probe_session.close()
    raise RuntimeError('Public candidate endpoint not reachable')


def close_services(run):
    path=run/'state/multi_city_services.json'
    if not path.exists():
        return
    state=load(path)
    named_path=run/'state/named_image_services.json'
    if named_path.exists():
        named=load(named_path)
        state['named_server']=named.get('server')
        supervisor_path=run/'state/named_tunnel.json'
        if supervisor_path.exists():
            supervisor=load(supervisor_path)
            state['named_supervisor']={'pid':supervisor.get('supervisor_pid',0),
                'command':' '.join(['python3','-S',str(SERVER_ROOT/'code/trajectory/server/keep_named_tunnel.py'),
                                    '--dataset',run.name])}
    for key in ('tunnel','server','named_server','named_supervisor'):
        record=state.get(key)
        if not record:
            continue
        try:
            fd=os.pidfd_open(record['pid'])
        except ProcessLookupError:
            continue
        try:
            if process_alive(record):
                signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:
            os.close(fd)
    state['closed_at']=time.time();save(path,state)


def prepare_city(city, run, output, config, workers, resources):
    # Each coordinator owns its connection; pooled Maps workers own theirs.
    db=connect(run)
    state_path=run/'state'/f'preparation_{city}.json'
    try:
        prune_candidates(db,run,city)
        save(state_path,{'city_id':city,'stage':'acquiring_starts','updated_at':time.time()})
        repair=run/'state'/f'network_recovery_{city}.json'
        if repair.exists() and not load(repair).get('replayed'):
            # Recovery replays frozen coordinates, never replaces or resamples.
            lookup_parallel(db,run,city,workers,100,resources)
            acquisition_path=run/'state'/f'acquisition_{city}.json'
            acquisition=load(acquisition_path)
            for row in acquisition['rounds']:
                row['unique_start_panos']=db.execute(
                    'SELECT count(*) FROM city_starts s JOIN samples p ON p.id=s.sample_id '
                    'WHERE s.city_id=? AND json_extract(p.data,\'$.sample_index\')<?',
                    (city,row['cumulative_sampled_points'])).fetchone()[0]
            if acquisition['rounds']:
                count=acquisition['rounds'][-1]['unique_start_panos']
                reached=count>=acquisition['policy']['target_start_panos']
                acquisition.update(unique_start_panos=count,target_reached=reached,
                    complete=reached or len(acquisition['rounds'])>=3,
                    stop_reason='target_reached' if reached else 'max_rounds_reached' if len(acquisition['rounds'])>=3 else None)
            save(acquisition_path,acquisition)
            record=load(repair);record['replayed']=True;save(repair,record)
            save_city_stage(db,city,'network_recovery_completed')
        acquire_starts(db,run,output,city,config,workers,resources)
        save(state_path,{'city_id':city,'stage':'prescreening','updated_at':time.time()})
        prescreen_parallel(db,run,city,workers,resources)
        save_city_stage(db,city,'prescreen_completed')
        ids=[r[0] for r in db.execute('SELECT id FROM samples WHERE city_id=? ORDER BY id',(city,))]
        save(state_path,{'city_id':city,'stage':'candidate_images','updated_at':time.time()})
        for attempt in range(3):
            candidates_parallel(ids,db,run,{**config,'multi_city_image_paths':True},workers,resources)
            retryable=any(int(json.loads(r[0]).get('candidate_attempts',0))<3 for r in db.execute(
                "SELECT data FROM samples WHERE city_id=? AND status='candidate_image_error'",(city,)))
            if not retryable:
                break
            time.sleep(10*(attempt+1))
        outcomes=dict(db.execute('SELECT status,count(*) FROM samples WHERE city_id=? GROUP BY status',(city,)))
        if outcomes.get('candidate_image_error') and not any(outcomes.get(s) for s in ('candidates_ready','llm_selected','complete')):
            raise RuntimeError('All candidate renders failed; inspect network before accepting this city')
        save_city_stage(db,city,'candidates_completed')
        save(state_path,{'city_id':city,'stage':'prepared','sampled':len(ids),'updated_at':time.time()})
        return ids
    except Exception as error:
        previous=load(state_path) if state_path.exists() else {}
        save(state_path,{'city_id':city,'stage':'blocked','failed_stage':previous.get('stage'),**error_details(error),
                         'updated_at':time.time()})
        raise
    finally:
        db.close()


def initial_submitted(db, city):
    """Every ready start has an accepted intent, or already has an LLM result."""
    unresolved=db.execute("SELECT count(*) FROM batches WHERE city_id=? AND status IN ('submitting','submission_unknown','rejected','provider_validation_error')",(city,)).fetchone()[0]
    if unresolved:
        raise RuntimeError('Unresolved Batch submission for '+city)
    pending_candidate_errors=0
    if db.execute("SELECT count(*) FROM samples WHERE city_id=? AND status IN ('sampled','pano_bound','start_eligible')",(city,)).fetchone()[0]:
        return False
    for raw, in db.execute("SELECT data FROM samples WHERE city_id=? AND status='candidate_image_error'",(city,)):
        if int(json.loads(raw).get('candidate_attempts',0)) < 3:
            pending_candidate_errors += 1
    if pending_candidate_errors:
        return False
    return db.execute("SELECT count(*) FROM samples WHERE city_id=? AND status='candidates_ready' AND attempts=0",(city,)).fetchone()[0]==0


def preparation_complete(db,run,city):
    acquisition=run/'state'/f'acquisition_{city}.json'
    recovery=run/'state'/f'network_recovery_{city}.json'
    if not acquisition.exists() or not load(acquisition).get('complete'):
        return False
    if recovery.exists() and not load(recovery).get('replayed'):
        return False
    rows=db.execute('SELECT status,data FROM samples WHERE city_id=?',(city,)).fetchall()
    return bool(rows) and all(status not in ('sampled','pano_bound','start_eligible') and
        (status!='candidate_image_error' or int(json.loads(raw).get('candidate_attempts',0))>=3)
        for status,raw in rows)


def generate_city(city,ids,run,output,config,workers,resources):
    db=connect(run)
    try:
        finish_parallel(ids,db,run,output,config,workers,resources)
        if not summarize(db,run,output,city):
            raise RuntimeError('City has unfinished tasks: '+city)
        save_city_stage(db,city,'complete')
        cleanup_city(db,run,city)
    finally:
        db.close()


def select_batch_city(db, remaining, prepared, max_inflight=2, retry_at=None):
    """Fill an open global slot with a different city; prioritize due repairs.

    A city with accepted/ambiguous work first waits for its results. Explicit
    413 splits still share the global cap in Batch.submit. Credit/image backoff
    applies only to its own city and never owns the other available slot.
    """
    if max_inflight is not None and inflight_count(db)>=max_inflight:
        return None
    candidates=[];now=time.time();retry_at=retry_at or {}
    for city in remaining:
        if city not in prepared or inflight_count(db,city) or now<retry_at.get(city,0):
            continue
        unresolved=db.execute("SELECT count(*) FROM batches WHERE city_id=? "
                              "AND status IN ('rejected','provider_validation_error')",(city,)).fetchone()[0]
        if unresolved:continue
        count,repairs=db.execute("SELECT count(*),sum(attempts>0) FROM samples WHERE city_id=? "
                                "AND status='candidates_ready' AND attempts<4",(city,)).fetchone()
        if count:candidates.append((not bool(repairs),city))
    # Stable sorting retains dataset order within each priority group.
    return min(candidates,key=lambda item:item[0])[1] if candidates else None


def execute(run, output, config, workers, city_workers=3, retry_seconds=300,
            max_inflight_batches=None, poll_seconds=60, max_batch_requests=None):
    max_inflight_batches=effective_inflight_limit(run,max_inflight_batches)
    state_path=run/'state/full_generation.json'
    prior=load(state_path) if state_path.exists() else {}
    state={'started_at':prior.get('started_at',time.time()),'phase':'rolling_generation',
           'city_workers':city_workers,'workers':workers,'max_inflight_batches':max_inflight_batches,
           'max_batch_requests':max_batch_requests,'batch_city':prior.get('batch_city'),
           'submission_policy':'two_cities_inflight','batch_cities':[],
           'cities':{},'inflight_batches':0}
    def checkpoint():
        state['updated_at']=time.time();save(state_path,state)
    def city_state(city,stage,**details):
        value={'stage':stage,**details}
        if state['cities'].get(city)!=value:
            state['cities'][city]=value;checkpoint()
    db=database(run)
    try:
        # Finish terminal tunnel cleanup even when a city completed immediately
        # before the previous coordinator exited.
        for city in CITY_IDS:
            BatchTunnels(run,city).reconcile(db)
        completed={city for city in CITY_IDS if complete_city(output,city)}
        remaining=[city for city in CITY_IDS if city not in completed]
        for city in CITY_IDS:
            state['cities'][city]={'stage':'complete' if city in completed else 'queued_for_preparation'}
        checkpoint()
        if not remaining:
            state['phase']='complete';checkpoint();close_services(run);return
        order_path=run/'state/generation_order.json'
        order=load(order_path)['cities'] if order_path.exists() else [c for c in CITY_IDS if c in completed]
        if len(order)!=len(set(order)) or any(c not in CITY_IDS for c in order):
            raise RuntimeError('Invalid saved generation order')
        unfinished=[c for c in order if c not in completed]
        if len(unfinished)>1:
            raise RuntimeError('Multiple unfinished cities in saved generation order')
        durable_save(order_path,{'cities':order})
        ensure_services(run,config['dataset_id'])
        renderer()
        from streetview_images import configure_downloads
        configure_downloads(workers['tiles'])
        save(run/'state/runtime_concurrency.json',workers)
        prepared={};errors={};retry_at={};poll_at={}
        for city in remaining:
            if preparation_complete(db,run,city):
                prepared[city]=[r[0] for r in db.execute('SELECT id FROM samples WHERE city_id=? ORDER BY id',(city,))]
                city_state(city,'prepared')
            previous=prior.get('cities',{}).get(city,{})
            retry_at[city]=previous.get('retry_at',0)
        with MapsPool(run,workers['maps']) as maps_pool, ThreadPoolExecutor(max_workers=workers.get('candidate',16),thread_name_prefix='candidate') as candidate_pool:
            resources={'maps':maps_pool,'candidates':candidate_pool,'binding':Lock(),'manifest':Lock()}
            with ThreadPoolExecutor(max_workers=city_workers,thread_name_prefix='prepare') as cities, ThreadPoolExecutor(max_workers=1,thread_name_prefix='generate') as downstream:
                futures={cities.submit(prepare_city,c,run,output,config,workers,resources):c for c in remaining if c not in prepared}
                generation=None;generating=None
                while len(completed)<len(CITY_IDS):
                    for future in list(futures):
                        if not future.done():continue
                        city=futures.pop(future)
                        try:
                            prepared[city]=future.result();city_state(city,'prepared')
                        except Exception as error:
                            errors[city]=type(error).__name__
                            city_state(city,'blocked',**error_details(error))
                            print(f'{city}: preparation blocked ({type(error).__name__}); other cities continue',flush=True)
                    if generation is not None and generation.done():
                        # Fatal downstream errors stop the coordinator: another city must
                        # never consume uncheckpointed pano reservations from this one.
                        generation.result()
                        completed.add(generating)
                        city_state(generating,'complete',completed_at=time.time())
                        print(f'{generating}: complete',flush=True)
                        generation=generating=None
                        state.pop('current_city',None);checkpoint()
                    # Poll even while preparation is active, and include pre-existing IDs.
                    for city in remaining:
                        if city in completed or city==generating:continue
                        if not inflight_count(db,city) or time.time()<poll_at.get(city,0):continue
                        try:
                            Batch(run,db,config,city,max_inflight=max_inflight_batches,max_requests=max_batch_requests).collect()
                            if inflight_count(db,city):city_state(city,'waiting_batch')
                        except BatchPollError:
                            city_state(city,'waiting_batch_poll_retry')
                        except Exception as error:
                            errors[city]=type(error).__name__
                            city_state(city,'blocked',**error_details(error))
                        poll_at[city]=time.time()+poll_seconds
                    batch_city=select_batch_city(db,[c for c in remaining if c not in completed
                                                    and c not in errors and c!=generating],
                                                 prepared,max_inflight_batches,retry_at)
                    if state.get('batch_city')!=batch_city:
                        state['batch_city']=batch_city;checkpoint()
                    ready=[]
                    for city in remaining:
                        if city not in prepared or city in completed or city==generating or city in errors:continue
                        unresolved=db.execute("SELECT count(*) FROM batches WHERE city_id=? AND status IN ('rejected','provider_validation_error')",(city,)).fetchone()[0]
                        if unresolved:
                            errors[city]='UnresolvedBatch';city_state(city,'blocked',error_type='UnresolvedBatch');continue
                        pending=db.execute("SELECT count(*),sum(attempts<4) FROM samples WHERE city_id=? AND status='candidates_ready'",(city,)).fetchone()
                        if pending[0]:
                            if pending[1]:
                                if inflight_count(db,city):city_state(city,'waiting_batch')
                                elif city!=batch_city and time.time()>=retry_at.get(city,0):
                                    city_state(city,'waiting_batch_slots')
                                continue
                            # Exhausted item retries are final outcomes, not new submissions.
                            Batch(run,db,config,city,max_inflight=max_inflight_batches,max_requests=max_batch_requests).submit_pending(prepared[city])
                        if inflight_count(db,city):
                            city_state(city,'waiting_batch');continue
                        ready.append(city)
                        city_state(city,'ready_for_generation')
                    if generation is None and ready:
                        resume=next((c for c in order if c not in completed),None)
                        # A saved unfinished city always precedes newly ready cities.
                        city=resume if resume in ready else (None if resume else ready[0])
                        if city is not None:
                            if city not in order:
                                order.append(city);durable_save(order_path,{'cities':order})
                            state['current_city']=city;city_state(city,'searching_and_rendering')
                            print(f'{city}: rolling path search and rendering begins',flush=True)
                            generating=city
                            generation=downstream.submit(generate_city,city,prepared[city],run,output,config,workers,resources)
                    # Start downstream work before potentially slow HTTP submissions.
                    # HTTP submissions remain serialized, but fill every available
                    # global slot so two different cities can be in flight together.
                    submit_limit = max_inflight_batches if max_inflight_batches is not None else 1
                    submitted = 0
                    while submitted < submit_limit:
                        city = select_batch_city(
                            db,
                            [c for c in remaining if c not in completed and c != generating
                             and c not in errors],
                            prepared, max_inflight_batches, retry_at)
                        if city is None:
                            break
                        try:
                            Batch(run,db,config,city,max_inflight=max_inflight_batches,
                                  max_requests=max_batch_requests).submit_pending(prepared[city])
                            submitted += 1
                            if inflight_count(db,city):city_state(city,'waiting_batch')
                            retry_at[city]=time.time()+poll_seconds
                        except BatchSlotsFull:
                            retry_at[city]=time.time()+poll_seconds
                            city_state(city,'waiting_batch_slots',retry_at=retry_at[city])
                            break
                        except TunnelUnavailable:
                            retry_at[city]=time.time()+retry_seconds
                            city_state(city,'waiting_batch_images',retry_at=retry_at[city])
                            print(f'{city}: Batch image service unavailable before POST; queued for retry',flush=True)
                        except BatchRateLimit as error:
                            retry_at[city]=time.time()+max(retry_seconds,error.retry_after or 0)
                            city_state(city,'waiting_credit' if isinstance(error,BatchCreditLimit) else 'waiting_batch_quota',
                                       retry_at=retry_at[city],limit_source=error.limit_source,reason=error.reason)
                            print(f'{city}: {type(error).__name__}; queued for retry',flush=True)
                        except Exception as error:
                            errors[city]=type(error).__name__;city_state(city,'blocked',**error_details(error))
                    active=inflight_count(db)
                    active_cities=[c for c in remaining if inflight_count(db,c)]
                    if state['inflight_batches']!=active or state['batch_cities']!=active_cities:
                        state['inflight_batches']=active;state['batch_cities']=active_cities;checkpoint()
                    if len(completed)==len(CITY_IDS):break
                    if generation is None and not futures and any(c in errors for c in order if c not in completed):
                        raise RuntimeError('Saved unfinished generation city requires reconciliation')
                    if errors and not futures and generation is None and all(c in completed or c in errors for c in CITY_IDS):
                        raise RuntimeError('City reconciliation required: '+','.join(errors))
                    # No constant state writes: only transitions and city checkpoints.
                    waiting=list(futures)+([generation] if generation is not None else [])
                    if waiting:wait(waiting,timeout=1,return_when=FIRST_COMPLETED)
                    else:time.sleep(1)
        state['phase']='complete';state['completed_at']=time.time();checkpoint()
        summary=load(output/'summary.json');summary['state']='complete';save(output/'summary.json',summary)
        close_services(run)
    except Exception as error:
        state['failed_phase']=state['phase'];state['phase']='blocked';state.update(error_details(error));checkpoint()
        print(f'Full generation blocked: {type(error).__name__}: {error}',flush=True)
        raise
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='sv30_20260920_r100_r3')
    parser.add_argument('--city-workers',type=int,default=3)
    parser.add_argument('--max-batch-requests',type=int,default=0,
                        help='0: whole city, no local request cap; positive: optional cap')
    parser.add_argument('--max-inflight-batches',type=int,default=2,
                        help='Global cap including accepted and ambiguous Batches')
    parser.add_argument('--maps-workers',type=int,default=16)
    parser.add_argument('--candidate-workers',type=int,default=32)
    parser.add_argument('--render-workers',type=int,default=32)
    parser.add_argument('--tile-workers',type=int,default=32)
    args=parser.parse_args()
    if args.max_batch_requests < 0:
        parser.error('Batch request cap must be zero or positive')
    if args.max_inflight_batches < 1:
        parser.error('In-flight Batch cap must be positive')
    if not 1<=args.city_workers<=5 or any(not 1<=v<=32 for v in (args.maps_workers,args.candidate_workers,args.render_workers,args.tile_workers)):
        parser.error('City workers must be 1..5; resource workers must be 1..32')
    run,output=runtime_paths(args.dataset)
    args.max_batch_requests=effective_request_limit(run,args.max_batch_requests)
    args.max_inflight_batches=effective_inflight_limit(run,args.max_inflight_batches)
    import tempfile
    tempfile.tempdir=str(run/'tmp')
    os.environ['XDG_CONFIG_HOME']=str(run/'tmp/config')
    os.environ['XDG_DATA_HOME']=str(run/'tmp/data')
    lock=(run/'state/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize(run)
    load_keys()
    from network import NetworkGuard
    NetworkGuard(run).wait(force=True)
    config=load(output/'dataset_config.json')
    if config['dataset_id']!=args.dataset:
        raise RuntimeError('Dataset identity mismatch')
    save(run/'state/continuation.json',{'approved':True,'scope':'all_remaining_cities',
         'workflow':'parallel_prepare_city_batches_rolling_generation','approved_at':time.time()})
    execute(run,output,config,{'maps':args.maps_workers,'candidate':args.candidate_workers,
                               'render':args.render_workers,'tiles':args.tile_workers},args.city_workers,
            max_batch_requests=args.max_batch_requests,max_inflight_batches=args.max_inflight_batches)


if __name__=='__main__':
    main()
