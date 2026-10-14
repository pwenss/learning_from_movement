"""Pause work while the configured Maps route is unavailable."""
import os
import time
from threading import Lock

import requests
from common import save


class NetworkGuard:
    def __init__(self, run):
        self.run=run
        self.lock=Lock()
        self.checked=0
        self.status=None

    def wait(self, force=False):
        with self.lock:
            if not force and time.monotonic()-self.checked<15:
                return
            paused=False
            while True:
                try:
                    proxy=os.environ.get('STREETVIEW_BROWSER_PROXY')
                    with requests.Session() as session:
                        session.trust_env=False
                        response=session.get('https://maps.googleapis.com/maps/api/js',
                            proxies={'http':proxy,'https':proxy} if proxy else None,timeout=(5,15))
                        response.raise_for_status()
                    self.checked=time.monotonic()
                    if self.status!='online':
                        save(self.run/'state/network.json',{'status':'online','checked_at':time.time()})
                    self.status='online'
                    if paused:
                        print('Maps network recovered; continuing the same pending queries.',flush=True)
                    return
                except requests.RequestException as error:
                    if self.status!='waiting_network':
                        save(self.run/'state/network.json',{'status':'waiting_network','error_type':type(error).__name__,
                                                          'checked_at':time.time()})
                    self.status='waiting_network'
                    if not paused:
                        print('Maps network unavailable; queries paused until the proxy recovers.',flush=True)
                    paused=True
                    time.sleep(15)
