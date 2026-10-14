"""Concurrency, rollback and external-reader lock regression tests on NFS."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier, Event, Thread
import time
import unittest
from unittest.mock import patch

import common
from storage import connect


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.qa_storage_')
        self.root=Path(self.tmp.name).resolve();(self.root/'state').mkdir()
        self.guard=patch.object(common,'SERVER_ROOT',self.root);self.guard.start()
        self.db=connect(self.root,timeout=.05,retry_seconds=5)
        self.db.executescript('CREATE TABLE counter (id INTEGER PRIMARY KEY,n INTEGER); INSERT INTO counter VALUES (1,0);')

    def tearDown(self):
        self.db.close();self.guard.stop();self.tmp.cleanup()

    def test_seven_writers_preserve_atomic_read_modify_write(self):
        gate=Barrier(7)
        def worker(_):
            db=connect(self.root,timeout=.05,retry_seconds=5)
            gate.wait()
            try:
                for _ in range(4):
                    with db:
                        n=db.execute('SELECT n FROM counter WHERE id=1').fetchone()[0]
                        db.execute('UPDATE counter SET n=? WHERE id=1',(n+1,))
            finally:db.close()
        with ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(worker,range(7)))
        self.assertEqual(self.db.execute('SELECT n FROM counter').fetchone()[0],28)
        self.assertFalse((self.root/'state/database_errors.jsonl').exists())

    def test_rollback_removes_all_writes(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with self.db:
                self.db.execute('UPDATE counter SET n=9 WHERE id=1')
                self.db.execute('INSERT INTO counter VALUES (1,4)')
        self.assertEqual(self.db.execute('SELECT n FROM counter').fetchone()[0],0)

    def test_partial_iteration_does_not_hold_read_lock(self):
        with self.db:
            self.db.execute('INSERT INTO counter VALUES (2,0)')
            self.db.execute('INSERT INTO counter VALUES (3,0)')
        rows=self.db.execute('SELECT * FROM counter');rows.fetchone()
        raw=sqlite3.connect(self.root/'state/generation.sqlite',timeout=.1)
        try:
            with raw:raw.execute('UPDATE counter SET n=7')
        finally:raw.close()
        self.assertEqual(len(rows.fetchall()),2)

    def test_busy_commit_retries_without_replaying_increment(self):
        # Inject only the first COMMIT failure, as SQLite specifies for BUSY:
        # transaction remains active and must be committed, not replayed.
        actual=self.db.raw
        class CommitBusyOnce:
            calls=0
            def __getattr__(self,name):return getattr(actual,name)
            def commit(self):
                self.calls+=1
                if self.calls==1:
                    error=sqlite3.OperationalError('database is locked')
                    error.sqlite_errorcode=sqlite3.SQLITE_BUSY
                    error.sqlite_errorname='SQLITE_BUSY'
                    raise error
                actual.commit()
        self.db.raw=CommitBusyOnce()
        with self.db:self.db.execute('UPDATE counter SET n=n+1')
        self.assertEqual(self.db.execute('SELECT n FROM counter').fetchone()[0],1)
        self.assertEqual(self.db.raw.calls,2)
        self.assertIn('COMMIT',(self.root/'state/database_errors.jsonl').read_text())

    def test_non_busy_error_is_not_retried_and_records_callsite(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.db.execute('SELECT * FROM nonexistent')
        import json
        record=json.loads((self.root/'state/database_errors.jsonl').read_text())
        self.assertEqual(record['sqlite_name'],'SQLITE_ERROR')
        self.assertEqual(record['retry'],0)
        self.assertTrue(record['traceback'])

    def test_external_writer_release_allows_begin_retry(self):
        raw=sqlite3.connect(self.root/'state/generation.sqlite',timeout=.1,check_same_thread=False)
        raw.execute('BEGIN IMMEDIATE');raw.execute('UPDATE counter SET n=10')
        original=self.db._record;released=Event()
        def release():
            released.wait(5)
            raw.commit();raw.close()
        thread=Thread(target=release);thread.start()
        def record(*args):
            original(*args);released.set()
        try:
            with patch.object(self.db,'_record',side_effect=record):
                with self.db:self.db.execute('UPDATE counter SET n=n+1')
        finally:
            released.set();thread.join()
        self.assertEqual(self.db.execute('SELECT n FROM counter').fetchone()[0],11)


if __name__=='__main__':unittest.main()
