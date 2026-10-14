"""Durable OpenRouter Batch submission with item-level repair, never voting."""
from __future__ import annotations

import json
import math
import os
import time

import requests

from common import digest, load, save
from stage_store import checkpoint, progress
from copy import deepcopy
from urllib.parse import urlsplit
from batch_tunnels import BatchTunnels

API = 'https://openrouter.ai/api/beta/batches'


def effective_inflight_limit(run, requested):
    """Count accepted, submitting and ambiguous Batches against a global cap."""
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError('Invalid requested in-flight Batch cap')
    path = run/'state/batch_policy.json'
    cap = load(path).get('max_inflight_batches') if path.exists() else None
    if cap is None:
        return requested
    if type(cap) is not int or cap < 1:
        raise ValueError('Invalid persisted in-flight Batch cap')
    return cap if requested is None else min(requested, cap)


def effective_request_limit(run, requested):
    """None/0 means a whole-city batch; a positive persisted cap still applies."""
    if requested == 0 and type(requested) is int:
        requested = None
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError('Invalid requested Batch request cap')
    path = run/'state/batch_policy.json'
    if not path.exists():
        return requested
    cap = load(path)['max_batch_requests']
    if cap is None:
        return requested
    if type(cap) is not int or cap < 1:
        raise ValueError('Invalid persisted Batch request cap')
    return cap if requested is None else min(requested, cap)


class BatchRateLimit(RuntimeError):
    """An explicit HTTP 429 is safe to retry; unknown acceptance is not."""
    def __init__(self,message,retry_after=None,limit_source=None,reason=None):
        super().__init__(message)
        self.retry_after=retry_after
        self.limit_source=limit_source
        self.reason=reason


class BatchCreditLimit(BatchRateLimit):
    """Explicit 402: distinguish balance, key cap and temporary in-flight budget."""


class BatchSlotsFull(RuntimeError):
    """Wait for an existing batch to settle before reserving another slot."""


class BatchPollError(RuntimeError):
    """Read-only polling failed; retain the accepted ID and retry later."""


def validate_image_urls(items):
    """Batch accepts public HTTP(S) images, never inline data URLs."""
    for request in items:
        for message in (request.get('body') or {}).get('messages') or []:
            for part in message.get('content') or []:
                if not isinstance(part,dict) or part.get('type')!='image_url':
                    continue
                image=part.get('image_url') or {}
                url=image.get('url') if isinstance(image,dict) else None
                parsed=urlsplit(url) if isinstance(url,str) else None
                if not parsed or parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError('Batch images require public HTTP(S) URLs')


def inflight_count(db,city=None):
    sql="SELECT count(*) FROM batches WHERE status IN ('submitted','submitting','submission_unknown')"
    return db.execute(sql+(' AND city_id=?' if city is not None else ''),
                      (city,) if city is not None else ()).fetchone()[0]


def parse_decision(body, count):
    content = body['choices'][0]['message']['content']
    result = json.loads(content)
    index, confidence = result['selected_index'], result['confidence']
    if type(index) is not int or not 0 <= index < count:
        raise ValueError('Invalid selected_index')
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid confidence')
    if not isinstance(result['reason'],str):
        raise ValueError('Invalid reason')
    return result


class Batch:
    def __init__(self, run, db, config, city_id, max_inflight=None, max_requests=None):
        self.run, self.db, self.config, self.city_id = run,db,config,city_id
        self.max_inflight=effective_inflight_limit(run,max_inflight)
        self.max_requests=effective_request_limit(run,max_requests)
        self.tunnels=BatchTunnels(run,city_id)
        self.headers = {'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'], 'Content-Type':'application/json'}

    def submit(self, items):
        if not items:
            return
        self.max_requests=effective_request_limit(self.run,self.max_requests)
        self.max_inflight=effective_inflight_limit(self.run,self.max_inflight)
        validate_image_urls(items)
        if self.max_requests is not None and len(items)>self.max_requests:
            for offset in range(0,len(items),self.max_requests):
                self.submit(items[offset:offset+self.max_requests])
            return
        if os.environ.get('STREETVIEW_NETWORK_GUARD')=='1':
            from network import NetworkGuard
            NetworkGuard(self.run).wait(force=True)
        # Store payload and intent before POST. A crash between server acceptance
        # and local ID persistence stays ambiguous; never resubmit blindly.
        identity = digest(items)
        previous=self.db.execute('SELECT status FROM batches WHERE id=?',(identity,)).fetchone()
        if previous and previous[0]=='rejected_size':
            if len(items)==1:
                raise RuntimeError('A single start exceeds Batch payload limits')
            middle=len(items)//2
            self.submit(items[:middle]);self.submit(items[middle:])
            return
        if previous and previous[0] not in ('rejected_rate_limit','rejected_credit_limit'):
            raise RuntimeError('Existing Batch intent must be reconciled before resubmission')
        if self.max_inflight is not None and inflight_count(self.db)>=self.max_inflight:
            raise BatchSlotsFull('Maximum in-flight Batch count reached')
        payload_path = self.run/'work'/self.city_id/(identity+'.json')
        outgoing = self.tunnels.prepare(identity,items)
        payload = {'endpoint':'/v1/chat/completions','model':self.config['model'],'requests':outgoing}
        save(payload_path,payload)
        with self.db:
            # Count and reserve atomically, including recursive 413 splits.
            if self.max_inflight is not None and inflight_count(self.db)>=self.max_inflight:
                raise BatchSlotsFull('Maximum in-flight Batch count reached')
            self.db.execute('INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)',
                (identity,self.city_id,'submitting',None,str(payload_path)))
        checkpoint(self.db,self.city_id,'batch_submitting')
        self.tunnels.protect(identity)
        progress(self.run,self.city_id,'batch_submitting')
        try:
            response = requests.post(API,headers=self.headers,json=payload,timeout=180)
        except requests.RequestException as error:
            with self.db:
                self.db.execute('UPDATE batches SET status=? WHERE id=?',('submission_unknown',identity))
            checkpoint(self.db,self.city_id,'batch_submission_unknown')
            # Retain a safe diagnostic; request/response bodies may contain secrets.
            raise RuntimeError('Batch submission_unknown ('+type(error).__name__+
                               '): reconcile server acceptance before resuming') from None
        if response.status_code == 413:
            with self.db:
                self.db.execute('UPDATE batches SET status=? WHERE id=?',('rejected_size',identity))
            checkpoint(self.db,self.city_id,'batch_rejected_size')
            self.tunnels.release(identity,'rejected_size')
            if len(items) == 1:
                raise RuntimeError('A single start exceeds Batch payload limits')
            middle=len(items)//2
            self.submit(items[:middle])
            self.submit(items[middle:])
            return
        if response.status_code in (429,402):
            status='rejected_rate_limit' if response.status_code==429 else 'rejected_credit_limit'
            with self.db:
                self.db.execute('UPDATE batches SET status=? WHERE id=?',(status,identity))
            checkpoint(self.db,self.city_id,status)
            self.tunnels.release(identity,status)
            error=BatchRateLimit if response.status_code==429 else BatchCreditLimit
            try:
                metadata=(response.json().get('error') or {}).get('metadata') or {}
                source=metadata.get('limit_source')
                reason=metadata.get('reason')
            except (ValueError,AttributeError,TypeError):
                source=reason=None
            if source not in ('openrouter_in_flight_budget','openrouter_key_limit','openrouter_credits'):
                source=None
            if reason not in ('in_flight_budget_exhausted','weight_exceeds_budget'):
                reason=None
            try:
                retry_after=float(getattr(response,'headers',{}).get('Retry-After'))
                if not math.isfinite(retry_after) or retry_after<0:retry_after=None
            except (ValueError,TypeError):retry_after=None
            raise error(f'Batch HTTP {response.status_code}; explicitly rejected, no accepted ID',
                        retry_after=retry_after,limit_source=source,reason=reason)
        if response.status_code not in (200,201,202):
            # Do not echo response text: providers may echo images or credentials.
            with self.db:
                self.db.execute('UPDATE batches SET status=? WHERE id=?',
                    ('rejected' if response.status_code < 500 else 'submission_unknown',identity))
            checkpoint(self.db,self.city_id,'batch_submission_unresolved')
            raise RuntimeError(f'Batch HTTP {response.status_code}; no model or protocol fallback; inspect provider compatibility')
        try:
            batch_id = response.json()['id']
            if not isinstance(batch_id,str) or not batch_id:
                raise ValueError()
        except (KeyError,ValueError):
            raise RuntimeError('Batch acceptance without valid ID: submission_unknown') from None
        with self.db:
            self.db.execute('UPDATE batches SET status=?, provider_id=? WHERE id=?',('submitted',batch_id,identity))
            for item in items:
                sample=self.db.execute('SELECT sample_id FROM requests WHERE custom_id=?',(item['custom_id'],)).fetchone()[0]
                self.db.execute('UPDATE samples SET attempts=attempts+1 WHERE id=?',(sample,))
        checkpoint(self.db,self.city_id,'batch_submitted')
        self.tunnels.protect(identity,batch_id)
        progress(self.run,self.city_id,'waiting_batch')
        print(f'Batch submitted: {batch_id}, {len(items)} starts',flush=True)

    def collect(self):
        self.tunnels.reconcile(self.db)
        rows=self.db.execute("SELECT id,status,provider_id,payload_path FROM batches WHERE city_id=? AND status NOT IN ('collected','rejected_size','rejected_rate_limit','rejected_credit_limit')",(self.city_id,)).fetchall()
        all_done=True
        unresolved=None
        poll_failed=False
        for identity,status,batch_id,path in rows:
            if status in ('submitting','submission_unknown','rejected','provider_validation_error'):
                unresolved=identity
                continue
            progress(self.run,self.city_id,'waiting_batch')
            result=None
            for attempt in range(4):
                try:
                    r=requests.get(API+'/'+batch_id,headers=self.headers,timeout=90)
                    r.raise_for_status()
                    result=r.json()
                    break
                except (requests.RequestException,ValueError):
                    if attempt == 3:
                        poll_failed=True
                        break
                    time.sleep(2**attempt)
            if result is None:
                continue
            if result.get('status') not in ('completed','failed','expired','cancelled'):
                all_done=False
                continue
            save(self.run/'work'/self.city_id/(identity+'.result.json'),result)
            if result.get('status')=='failed' and result.get('error') and not result.get('results'):
                with self.db:
                    self.db.execute('UPDATE batches SET status=? WHERE id=?',('provider_validation_error',identity))
                checkpoint(self.db,self.city_id,'batch_provider_validation_error')
                raise RuntimeError('Batch configuration rejected before item execution; inspect saved envelope, do not retry unchanged inputs')
            items=load(__import__('pathlib').Path(path))['requests']
            expected={x['custom_id'] for x in items}
            received={}
            for item in result.get('results') or []:
                custom=item.get('custom_id')
                if custom not in expected or custom in received:
                    raise RuntimeError('Batch returned unknown or duplicate custom_id')
                received[custom]=item
            # Keep the terminal provider envelope for diagnosis and usage;
            # per-start audit below is the canonical final record.
            with self.db:
                self._apply_results(expected,received,batch_id)
                self.db.execute('UPDATE batches SET status=? WHERE id=?',('collected',identity))
            checkpoint(self.db,self.city_id,'batch_collected')
            self.tunnels.release(identity,'collected')
            progress(self.run,self.city_id,'batch_collected')
        if unresolved:
            raise RuntimeError('Unresolved Batch submission; reconcile before continuing: '+unresolved[:12])
        if poll_failed:
            raise BatchPollError('Cannot collect Batch; stored batch ID will be resumed')
        return all_done

    def _apply_results(self,expected,received,batch_id):
        for custom in expected:
            sample_id = self.db.execute('SELECT sample_id FROM requests WHERE custom_id=?',(custom,)).fetchone()[0]
            sample_row=self.db.execute('SELECT data FROM samples WHERE id=?',(sample_id,)).fetchone()
            sample=json.loads(sample_row[0])
            item=received.get(custom,{'error':'missing_result'})
            response=item.get('response') or {}
            body=response.get('body') or {}
            actual=body.get('model')
            if actual and actual.removesuffix(':batch') != self.config['base_model']:
                raise RuntimeError('Batch actual model differs from frozen model')
            sample.setdefault('batch_history',[]).append({'batch_id':batch_id,'custom_id':custom,'response':item})
            if 'decision' not in sample:
                try:
                    if response.get('status_code',200) != 200 or item.get('error'):
                        raise ValueError('Request failed')
                    parsed=parse_decision(body,len(sample['start']['links']))
                    sample['decision']={**parsed,'selected_pano_id':sample['start']['links'][parsed['selected_index']]['pano_id'],
                        'batch_id':batch_id,'custom_id':custom,'actual_model':actual,
                        'provider':'OpenRouter','requested_model':self.config['model'],'submission_type':'openrouter_batch'}
                    status='llm_selected'
                except (KeyError,TypeError,ValueError,IndexError):
                    status='candidates_ready'
                    sample['last_llm_error']='llm_invalid_output' if body.get('choices') else 'llm_request_error'
            else:
                status='llm_selected'
            self.db.execute('UPDATE samples SET status=?,data=? WHERE id=?',(status,json.dumps(sample),sample_id))

    def run_group(self, ids):
        while True:
            if not self.collect():
                time.sleep(60)
                continue
            if not self.submit_pending(ids):
                return

    def submit_pending(self, ids):
        """Submit ready items without waiting; preserve accepted shards on resume."""
        self.tunnels.reconcile(self.db)
        active=set()
        for status,path in self.db.execute('SELECT status,payload_path FROM batches WHERE city_id=?',(self.city_id,)):
            if status in ('submitting','submission_unknown','rejected','provider_validation_error'):
                raise RuntimeError('Unresolved Batch submission; reconcile before continuing')
            if status=='submitted':
                for request in load(__import__('pathlib').Path(path))['requests']:
                    row=self.db.execute('SELECT sample_id FROM requests WHERE custom_id=?',(request['custom_id'],)).fetchone()
                    if row:
                        active.add(row[0])
        pending=[]
        attempts_by_id={}
        exhausted=False
        for sample_id in ids:
            status,attempts,raw=self.db.execute('SELECT status,attempts,data FROM samples WHERE id=?',(sample_id,)).fetchone()
            if status!='candidates_ready' or sample_id in active:
                continue
            sample=json.loads(raw)
            if attempts>=4:
                exhausted=True
                with self.db:
                    self.db.execute('UPDATE samples SET status=? WHERE id=?',(sample.get('last_llm_error','llm_request_error'),sample_id))
                continue
            request=deepcopy(sample['request']) if sample.get('request') else load(__import__('pathlib').Path(sample['request_path']))
            request.pop('_sample_id',None)
            request['custom_id']=request['custom_id'].split('-a')[0]+'-a'+str(attempts)
            attempts_by_id[request['custom_id']]=sample_id
            pending.append(request)
        if not pending:
            if exhausted:checkpoint(self.db,self.city_id,'batch_requests_settled')
            return False
        with self.db:
            for custom,sample_id in attempts_by_id.items():
                self.db.execute('INSERT OR IGNORE INTO requests VALUES (?,?)',(custom,sample_id))
        # Each outgoing shard gets its own URL; stored sample requests and all
        # previously accepted payloads remain unchanged.
        self.submit(pending)
        return True
