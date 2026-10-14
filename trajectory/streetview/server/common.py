"""Project-confined paths, atomic records, and geometry for server generation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

SERVER_ROOT = Path('/path/to/trajectory_generation')
CITY_IDS = ['nyc_new_york', 'sf_san_francisco', 'chicago', 'philadelphia', 'toronto', 'vancouver',
            'mexico_city', 'sao_paulo', 'rio_de_janeiro', 'buenos_aires', 'london', 'paris',
            'berlin', 'madrid', 'rome', 'amsterdam', 'zurich', 'vienna', 'stockholm', 'prague',
            'tokyo', 'osaka', 'seoul', 'taipei', 'singapore', 'bangkok', 'kuala_lumpur',
            'sydney', 'melbourne', 'auckland']


def confined(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(SERVER_ROOT.resolve()):
        raise ValueError('Path escapes the authorized project root')
    return resolved


def save(path: Path, value) -> None:
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = confined(path.with_name(path.name + '.tmp'))
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load(path: Path):
    return json.loads(confined(path).read_text(encoding='utf-8'))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def distance(a, b):
    lat1, lat2 = math.radians(a['lat']), math.radians(b['lat'])
    dlat = lat2 - lat1
    dlon = math.radians(b['lon'] - a['lon'])
    v = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon / 2)**2
    return 12742000 * math.asin(math.sqrt(min(1, max(0, v))))


def turn(a, b):
    delta = abs(a-b) % 360
    return min(delta, 360-delta)


def date_parts(value):
    m = re.fullmatch(r'(\d{4})(?:-(\d{2}))?', value or '')
    if not m or int(m[1]) < 1 or (m[2] and not 1 <= int(m[2]) <= 12):
        raise ValueError('Invalid capture date')
    return int(m[1]), int(m[2]) if m[2] else None


def dates_valid(nodes, final=False):
    try:
        parts = [date_parts(n['date']) for n in nodes]
    except (ValueError, KeyError):
        return False
    if len({p[0] for p in parts}) != 1:
        return False
    return not (final and all(p[1] is not None for p in parts)) or len(set(parts)) == 1


def bearing(a, b):
    x, y = math.radians(a['lat']), math.radians(b['lat'])
    d = math.radians(b['lon']-a['lon'])
    return math.degrees(math.atan2(math.sin(d)*math.cos(y),
                       math.cos(x)*math.sin(y)-math.sin(x)*math.cos(y)*math.cos(d))) % 360


def validate_route(frames, start, selected):
    if len(frames) != 8 or len({f['pano_id'] for f in frames}) != 8:
        raise ValueError('Frame count or pano uniqueness failed')
    if frames[0]['pano_id'] != start or frames[1]['pano_id'] != selected:
        raise ValueError('Fixed start/first hop mismatch')
    if not dates_valid(frames, final=True):
        raise ValueError('Date constraint failed')
    for i, frame in enumerate(frames):
        if not all(math.isfinite(frame[k]) for k in ('lat', 'lon', 'heading', 'render_heading')):
            raise ValueError('Non-finite metadata')
        if i and turn(frames[i-1]['heading'], frame['heading']) > 90:
            raise ValueError('Heading constraint failed')
        if i < 7:
            target = frames[i+1]
            if not 1 < distance(frame, target) < 30:
                raise ValueError('Distance constraint failed')
            if frame['next_pano_id'] != target['pano_id'] or not frame['next_in_sequence']:
                raise ValueError('Broken chain')
            if not any(l['pano_id'] == target['pano_id'] and l['heading'] == frame['heading']
                       for l in frame['links']):
                raise ValueError('Missing original navigation link')
            if turn(frame['render_heading'], bearing(frame, target)) > 0.00001:
                raise ValueError('Render bearing mismatch')
    last = frames[-1]
    if last['next_pano_id'] is not None or last['next_in_sequence']:
        raise ValueError('Terminal outbound link')
    if any(last[k] != frames[-2][k] for k in ('heading', 'render_heading')):
        raise ValueError('Terminal heading mismatch')


def load_project_environment():
    path = confined(SERVER_ROOT / 'config/keys.env')
    if path.exists():
        if path.stat().st_mode & 0o077:
            path.chmod(0o600)
        for line in path.read_text().splitlines():
            name, sep, value = line.removeprefix('export ').partition('=')
            if sep and name.strip() in ('GOOGLE_MAPS_API_KEY', 'OPENROUTER_API_KEY',
                                        'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY',
                                        'STREETVIEW_BROWSER_PROXY','STREETVIEW_IMAGE_BASE_URL'):
                os.environ[name.strip()] = value.strip().strip('\"\'')


def load_keys():
    load_project_environment()
    missing = [n for n in ('GOOGLE_MAPS_API_KEY', 'OPENROUTER_API_KEY') if not os.environ.get(n)]
    if missing:
        raise RuntimeError('Missing credentials: ' + ', '.join(missing) + '; configure config/keys.env')


def runtime_paths(dataset):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', dataset):
        raise ValueError('Invalid dataset ID')
    run = confined(SERVER_ROOT / 'runs' / dataset)
    output = confined(SERVER_ROOT / 'datasets' / dataset)
    for path in (run/'state', run/'tmp', run/'logs', run/'cache', run/'work', output):
        path.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(run/'tmp')
    os.environ['XDG_CACHE_HOME'] = str(run/'cache')
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(confined(SERVER_ROOT/'env/browsers'))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return run, output
