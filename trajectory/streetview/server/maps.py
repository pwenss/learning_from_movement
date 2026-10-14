"""Maps JavaScript metadata bridge. No location resampling or pano substitution."""
from __future__ import annotations

import json
import math
import os
import time

from playwright.sync_api import sync_playwright

from common import confined, date_parts, digest


class MapsError(RuntimeError):
    pass


class Maps:
    def __init__(self, run, db, worker_id=None, request_locks=None):
        self.db = db
        self.request_locks = request_locks
        self.network = getattr(request_locks,'network',None)
        if self.network:
            self.network.wait()
        key = os.environ['GOOGLE_MAPS_API_KEY']
        html = ('<!doctype html><meta charset="utf-8"><script>window.ready=false;'
                'window.gm_authFailure=()=>window.authFailed=true;'
                'window.init=()=>{window.service=new google.maps.StreetViewService();window.ready=true};'
                '</script><script async src="https://maps.googleapis.com/maps/api/js?key='+key+
                '&v=weekly&callback=init"></script>').encode()
        self.playwright = sync_playwright().start()
        options={}
        if os.environ.get('STREETVIEW_BROWSER_PROXY'):
            from urllib.parse import urlsplit,unquote
            proxy=urlsplit(os.environ['STREETVIEW_BROWSER_PROXY'])
            options['proxy']={'server':f'{proxy.scheme}://{proxy.hostname}:{proxy.port}',
                              'bypass':'127.0.0.1,localhost'}
            if proxy.username:
                options['proxy'].update(username=unquote(proxy.username),password=unquote(proxy.password or ''))
        self.context = self.playwright.chromium.launch_persistent_context(
            str(confined(run/'tmp'/('browser' if worker_id is None else f'browser_{worker_id}'))),headless=True,accept_downloads=False,
            args=['--disable-breakpad','--disable-crash-reporter'],**options)
        self.page = self.context.pages[0]
        # Serve the local bootstrap from memory. Proxy CONNECT must never send
        # server-local loopback URLs to the laptop's loopback interface.
        bootstrap='http://127.0.0.1:8765/'
        self.page.route(bootstrap,lambda route:route.fulfill(status=200,content_type='text/html; charset=utf-8',body=html))
        self.page.goto(bootstrap,wait_until='domcontentloaded')
        try:
            self.page.wait_for_function('window.ready || window.authFailed',timeout=60000)
        except Exception:
            self.close()
            raise MapsError('Maps JavaScript initialization failed; check network and key restrictions') from None
        if self.page.evaluate('Boolean(window.authFailed)'):
            self.close()
            raise MapsError('Maps JavaScript authentication failed')

    def close(self):
        self.context.close()
        self.playwright.stop()

    def get(self, request):
        if self.request_locks is not None:
            # Same request shares its cache and retry budget across all workers.
            with self.request_locks.for_request(digest(request)):
                return self._get(request)
        return self._get(request)

    def _get(self, request):
        cache_id = digest(request)
        found = self.db.execute('SELECT payload FROM metadata WHERE id=?',(cache_id,)).fetchone()
        if found:
            value = json.loads(found[0])
            if 'error' in value:
                raise MapsError(value['error'])
            return value
        previous = self.db.execute('SELECT attempts FROM api_attempts WHERE id=?',(cache_id,)).fetchone()
        attempts = previous[0] if previous else 0
        error = 'metadata_retry_exhausted'
        transient_failures=0
        while attempts < 4:
            if self.network:
                self.network.wait()
            attempts += 1
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO api_attempts VALUES (?,?)',(cache_id,attempts))
            result = self.page.evaluate('''request => new Promise(resolve => {
                let done=false;
                const finish=x=>{if(!done){done=true;clearTimeout(timer);resolve(x)}};
                const timer=setTimeout(()=>finish({error:"TIMEOUT"}),15000);
                window.service.getPanorama(request,(d,status)=>{
                    if(status!=="OK"||!d){finish({error:String(status)});return;}
                    const size=x=>x?{width:Number(x.width),height:Number(x.height)}:null;
                    finish({pano_id:d.location.pano,lat:d.location.latLng.lat(),lon:d.location.latLng.lng(),
                        date:d.imageDate||null,source_pano_heading:Number(d.tiles?.centerHeading),
                        source_pano_world_size:size(d.tiles?.worldSize),source_pano_tile_size:size(d.tiles?.tileSize),
                        links:(d.links||[]).filter(x=>x&&x.pano).map(x=>({pano_id:x.pano,heading:Number(x.heading),description:x.description||null}))});
                });
            })''', request)
            if 'error' not in result:
                if request.get('pano') and request['pano'] != result['pano_id']:
                    raise MapsError('pano_id_changed')
                with self.db:
                    self.db.execute('INSERT INTO metadata VALUES (?,?)',(cache_id,json.dumps(result)))
                return result
            error = result['error']
            if error in ('UNKNOWN_ERROR','TIMEOUT','OVER_QUERY_LIMIT') and self.network:
                # Infrastructure failures must not consume a point or its finite
                # query budget. Retry the identical request after connectivity.
                attempts-=1
                with self.db:
                    self.db.execute('INSERT OR REPLACE INTO api_attempts VALUES (?,?)',(cache_id,attempts))
                transient_failures+=1
                if transient_failures==4:
                    print('Maps transient errors persist; retaining and retrying the same query.',flush=True)
                self.network.wait(force=True)
                time.sleep(min(60,2**min(transient_failures,6)))
                continue
            if error in ('REQUEST_DENIED','INVALID_REQUEST'):
                raise RuntimeError('Systemic Maps failure: '+error)
            if error == 'ZERO_RESULTS':
                with self.db:
                    self.db.execute('INSERT INTO metadata VALUES (?,?)',(cache_id,json.dumps(result)))
                break
            if attempts < 4:
                time.sleep(2**attempts)
        raise MapsError(error)


def valid_metadata(node):
    try:
        date_parts(node.get('date'))
        if not node['pano_id'] or not all(math.isfinite(node[k]) for k in ('lat','lon','source_pano_heading')):
            return False
        for key in ('source_pano_world_size','source_pano_tile_size'):
            if any(not math.isfinite(node[key][d]) or node[key][d] <= 0 for d in ('width','height')):
                return False
        return all(math.isfinite(link['heading']) for link in node['links'])
    except (TypeError, ValueError, KeyError):
        return False
