"""Serialize short SQLite transactions on NFS; never hold cursors across work."""
from collections import deque
import json
import sqlite3
from threading import Lock, RLock
import time
import traceback

from common import confined

_registry_lock=Lock()
_gates={}


def error_details(error):
    result={'error_type':type(error).__name__,
            'traceback':[{'file':f.filename,'line':f.lineno,'function':f.name}
                         for f in traceback.extract_tb(error.__traceback__)]}
    if isinstance(error,sqlite3.Error):
        result.update(message=str(error),sqlite_code=getattr(error,'sqlite_errorcode',None),
                      sqlite_name=getattr(error,'sqlite_errorname',None))
    return result


class Rows:
    """Materialized results release SQLite read locks before the caller iterates."""
    def __init__(self, cursor):
        try:
            self.description=cursor.description
            self.rowcount=cursor.rowcount
            self.lastrowid=cursor.lastrowid
            self.rows=deque(cursor.fetchall()) if cursor.description else deque()
        finally:
            cursor.close()

    def fetchone(self):
        return self.rows.popleft() if self.rows else None

    def fetchall(self):
        result=list(self.rows);self.rows.clear();return result

    def __iter__(self):
        return self

    def __next__(self):
        if not self.rows:
            raise StopIteration
        return self.rows.popleft()


class Connection:
    def __init__(self, run, timeout=2, retry_seconds=120):
        self.path=confined(run/'state/generation.sqlite')
        with _registry_lock:
            self.gate=_gates.setdefault(str(self.path),RLock())
        self.retry_seconds=retry_seconds
        self.transaction=False
        self.run=run
        # Explicit transactions below; a standalone mutation is autocommitted.
        with self.gate:
            self.raw=sqlite3.connect(self.path,timeout=timeout,isolation_level=None)

    def _retry(self, operation, function):
        deadline=time.monotonic()+self.retry_seconds
        attempt=0
        while True:
            try:
                return function()
            except sqlite3.OperationalError as error:
                code=getattr(error,'sqlite_errorcode',0) or 0
                busy=(code & 255) in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED)
                if not busy or time.monotonic()>=deadline:
                    self._record(operation,error,attempt,False)
                    raise
                attempt+=1
                if attempt==1:
                    self._record(operation,error,attempt,True)
                time.sleep(min(.1*2**min(attempt-1,5),2))

    def _record(self, operation, error, attempt, retrying):
        # SQL parameters, credentials and HTTP payloads are deliberately excluded.
        path=confined(self.run/'state/database_errors.jsonl')
        with path.open('a',encoding='utf8') as handle:
            handle.write(json.dumps({'at':time.time(),'operation':operation,'retry':attempt,
                                     'retrying':retrying,**error_details(error)})+'\n')

    def execute(self, sql, parameters=()):
        with self.gate:
            return self._retry(sql.split(None,1)[0],lambda:Rows(self.raw.execute(sql,parameters)))

    def executescript(self, sql):
        with self.gate:
            if self.transaction:
                raise RuntimeError('Schema scripts cannot run inside a transaction')
            return self._retry('schema',lambda:Rows(self.raw.executescript(sql)))

    def __enter__(self):
        self.gate.acquire()
        try:
            if self.transaction:
                raise RuntimeError('Nested SQLite transactions are not supported')
            self._retry('BEGIN IMMEDIATE',lambda:self.raw.execute('BEGIN IMMEDIATE').close())
            self.transaction=True
            return self
        except BaseException:
            self.gate.release()
            raise

    def __exit__(self, kind, error, tb):
        try:
            if kind is None:
                try:
                    # A BUSY commit leaves the transaction active: retry COMMIT,
                    # never replay increments, network calls or the body.
                    self._retry('COMMIT',self.raw.commit)
                except BaseException:
                    self.raw.rollback()
                    raise
            else:
                self.raw.rollback()
        finally:
            self.transaction=False
            self.gate.release()
        return False

    def commit(self):
        with self.gate:
            self._retry('COMMIT',self.raw.commit)

    def rollback(self):
        with self.gate:
            self.raw.rollback()

    def close(self):
        with self.gate:
            self.raw.close()


def connect(run, **kwargs):
    from stage_store import enabled, connect as stage_connect
    if enabled(run):
        return stage_connect(run)
    return Connection(run,**kwargs)
