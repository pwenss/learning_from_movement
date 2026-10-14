"""In-memory working state with atomic, whole-city stage checkpoints."""
import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from common import CITY_IDS, confined, load, save
from storage import Connection

TABLES={
    'samples':['id','city_id','status','attempts','data'],
    'starts':['pano_id','sample_id'],
    'city_starts':['city_id','pano_id','sample_id'],
    'panos':['pano_id','sample_id'],
    'requests':['custom_id','sample_id'],
    'batches':['id','city_id','status','provider_id','payload_path'],
}
SCHEMA='''
CREATE TABLE samples(id TEXT PRIMARY KEY,city_id TEXT,status TEXT,attempts INTEGER DEFAULT 0,data TEXT);
CREATE TABLE starts(pano_id TEXT PRIMARY KEY,sample_id TEXT UNIQUE);
CREATE TABLE city_starts(city_id TEXT,pano_id TEXT,sample_id TEXT UNIQUE,PRIMARY KEY(city_id,pano_id));
CREATE TABLE panos(pano_id TEXT PRIMARY KEY,sample_id TEXT);
CREATE TABLE requests(custom_id TEXT PRIMARY KEY,sample_id TEXT);
CREATE TABLE batches(id TEXT PRIMARY KEY,city_id TEXT,status TEXT,provider_id TEXT,payload_path TEXT);
CREATE TABLE metadata(id TEXT PRIMARY KEY,payload TEXT);
CREATE TABLE api_attempts(id TEXT PRIMARY KEY,attempts INTEGER);
CREATE INDEX samples_city_status_id ON samples(city_id,status,id);
'''
_states={}
_load_lock=RLock()


def durable_save(path,value):
    save(path,value)
    # The accepted Batch ID must survive a filesystem restart, including rename.
    if os.name=='posix':
        fd=os.open(confined(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def enabled(run):
    return str(Path(run).resolve()) in _states or (run/'state/city_stages/mode.json').exists()


class StageState:
    def __init__(self,run):
        self.run=confined(run)
        self.directory=confined(run/'state/city_stages')
        self.gate=RLock()
        self.raw=sqlite3.connect(':memory:',isolation_level=None,check_same_thread=False)
        self.raw.executescript(SCHEMA)

    def restore(self, value):
        with self.gate:
            for table,columns in TABLES.items():
                self.raw.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in columns)})',value['tables'][table])

    def checkpoint(self,city,phase):
        if city not in CITY_IDS:
            raise ValueError('Unknown city')
        with self.gate:
            tables={}
            for table in TABLES:
                if table in ('samples','batches','city_starts'):
                    sql=f'SELECT * FROM {table} WHERE city_id=?'
                else:
                    sql=f'SELECT * FROM {table} WHERE sample_id IN (SELECT id FROM samples WHERE city_id=?)'
                tables[table]=self.raw.execute(sql,(city,)).fetchall()
            acquisition=self.run/'state'/f'acquisition_{city}.json'
            recovery=self.run/'state'/f'network_recovery_{city}.json'
            value={'version':1,'city_id':city,'completed_stage':phase,'saved_at':time.time(),
                   'sample_states':dict(Counter(r[2] for r in tables['samples'])),
                   'batch_states':dict(Counter(r[2] for r in tables['batches'])),
                   'acquisition':load(acquisition) if acquisition.exists() else None,
                   'recovery':load(recovery) if recovery.exists() else None,'tables':tables}
            durable_save(self.directory/f'{city}.json',value)
            return value


class MemoryConnection(Connection):
    def __init__(self,state):
        self.raw=state.raw;self.gate=state.gate;self.run=state.run
        self.retry_seconds=0;self.transaction=False;self.stage_state=state

    def close(self):
        # Workers share this memory store until the coordinator process exits.
        pass

    def _record(self,*args):
        # Unexpected failures are reported at city level by the coordinator.
        pass

    def checkpoint(self,city,phase):
        return self.stage_state.checkpoint(city,phase)


def initialize(run):
    key=str(Path(run).resolve())
    with _load_lock:
        if key in _states:
            return _states[key]
        state=StageState(run)
        marker=state.directory/'mode.json'
        try:
            if marker.exists():
                for city in load(marker)['cities']:
                    value=load(state.directory/f'{city}.json');state.restore(value)
                    # Stage data and round metadata must resume from one boundary.
                    for field,prefix in [('acquisition','acquisition_'),('recovery','network_recovery_')]:
                        path=run/'state'/f'{prefix}{city}.json'
                        if value[field] is not None:
                            if not path.exists() or load(path)!=value[field]:
                                save(path,value[field])
                        elif path.exists():
                            path.unlink()
            else:
                legacy=confined(run/'state/generation.sqlite')
                imported=0
                if legacy.exists():
                    # No writes to the legacy DB; metadata/cache tables are not imported.
                    source=sqlite3.connect(legacy.as_uri()+'?mode=ro',uri=True,timeout=10)
                    try:
                        for table,columns in TABLES.items():
                            rows=source.execute(f'SELECT {",".join(columns)} FROM {table}').fetchall()
                            state.raw.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in columns)})',rows)
                            if table=='samples':imported=len(rows)
                    finally:source.close()
                    # Retain legacy candidate requests in the whole-city snapshot.
                    for sid,raw in state.raw.execute('SELECT id,data FROM samples').fetchall():
                        sample=json.loads(raw);path=sample.get('request_path')
                        if path and Path(path).is_file():
                            sample['request']=load(Path(path))
                            state.raw.execute('UPDATE samples SET data=? WHERE id=?',(json.dumps(sample),sid))
                for city in CITY_IDS:state.checkpoint(city,'migration')
                durable_save(marker,{'version':1,'storage':'memory_with_city_stage_checkpoints','cities':CITY_IDS,
                             'migrated_samples':imported,'created_at':time.time()})
            _states[key]=state
            return state
        except BaseException:
            state.raw.close();raise


def connect(run):
    return MemoryConnection(initialize(run))


def checkpoint(db,city,phase):
    if isinstance(db,MemoryConnection):
        db.checkpoint(city,phase)


def read_status(run,city):
    path=run/'state/city_stages'/f'{city}.json'
    value=load(path)
    return {k:value[k] for k in ['city_id','completed_stage','saved_at','sample_states','batch_states']}


def progress(run,city,stage,**details):
    if not enabled(run):
        return
    path=run/'state'/f'preparation_{city}.json'
    value={'city_id':city,'stage':stage,**details}
    old=load(path) if path.exists() else {}
    if any(old.get(k)!=v for k,v in value.items()):
        save(path,{**value,'updated_at':time.time()})


def prune_candidates(db,run,city):
    """Validate directories once; never resolve thousands of NFS leaf paths."""
    if not isinstance(db,MemoryConnection):
        return
    directory=confined(run/'work'/city)
    images=confined(directory/'public_images')
    keep=set();legacy_requests=set()
    for raw, in db.execute('SELECT data FROM samples WHERE city_id=?',(city,)):
        sample=json.loads(raw)
        for name in sample.get('candidate_files',[]):
            path=Path(name)
            if path.parent==images:
                keep.add(path.name)
        if not sample.get('request') and sample.get('request_path'):
            old=Path(sample['request_path'])
            if old.parent==directory:
                legacy_requests.add(old.name)
    # Accepted payloads also protect images absent from an incomplete legacy row.
    for status,payload in db.execute('SELECT status,payload_path FROM batches WHERE city_id=?',(city,)):
        if status in ('collected','rejected_size','rejected_rate_limit','rejected_credit_limit'):
            continue
        for request in load(Path(payload))['requests']:
            for message in request['body']['messages']:
                for part in message.get('content',[]):
                    if isinstance(part,dict) and part.get('type')=='image_url':
                        name=Path(urlsplit(part['image_url']['url']).path).name
                        if re.fullmatch(r'[0-9a-f]{64}\.jpg',name):
                            keep.add(name)
    # Only unlink enumerated leaf entries under the validated directories.
    # unlink removes a symlink itself, without following it to another location.
    if directory.exists():
        for path in directory.iterdir():
            if path.name.endswith('.request.json') and path.name not in legacy_requests:
                path.unlink(missing_ok=True)
    if images.exists():
        for path in images.iterdir():
            if re.fullmatch(r'[0-9a-f]{64}\.jpg',path.name) and path.name not in keep:
                path.unlink(missing_ok=True)
