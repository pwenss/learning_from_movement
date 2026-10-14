"""Bounded workers; browser and SQLite connections stay on their owning thread."""
from concurrent.futures import Future
from queue import Queue
import sqlite3
from threading import Lock, Thread

from common import confined


class RequestLocks:
    def __init__(self):
        self.guard = Lock()
        self.locks = {}

    def for_request(self, identity):
        with self.guard:
            return self.locks.setdefault(identity, Lock())


class MapsPool:
    def __init__(self, run, workers=4, factory=None):
        if factory is None:
            from maps import Maps
            factory = Maps
        self.run, self.factory = run, factory
        self.locks = RequestLocks()
        if factory.__module__ == 'maps':
            from network import NetworkGuard
            self.locks.network = NetworkGuard(run)
        self.queue = Queue(maxsize=workers * 2)
        self.threads = [Thread(target=self._worker, args=(i,), name=f'maps-{i}')
                        for i in range(workers)]

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return self

    def submit(self, function, *args):
        future = Future()
        self.queue.put((future, function, args))
        return future

    def _worker(self, index):
        from storage import connect
        db = connect(self.run)
        maps = None
        try:
            while True:
                job = self.queue.get()
                if job is None:
                    return
                future, function, args = job
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if maps is None:
                        maps = self.factory(self.run, db, worker_id=index, request_locks=self.locks)
                    future.set_result(function(maps, db, *args))
                except BaseException as error:
                    future.set_exception(error)
        finally:
            if maps is not None:
                maps.close()
            db.close()

    def __exit__(self, *exc):
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()


def ordered_window(pool, function, items, window=8):
    """Bound submitted work while delivering results in fixed input order."""
    from collections import deque
    source = iter(items)
    pending = deque()
    for _ in range(window):
        try:
            item = next(source)
        except StopIteration:
            break
        pending.append((item, pool.submit(function, item)))
    while pending:
        item, future = pending.popleft()
        yield item, future.result()
        try:
            item = next(source)
        except StopIteration:
            continue
        pending.append((item, pool.submit(function, item)))
