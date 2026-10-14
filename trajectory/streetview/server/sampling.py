"""Freeze separated random points before any Street View queries."""
from __future__ import annotations

import math
import random
import time

import requests
from shapely.geometry import Point, shape
from shapely.geometry import mapping
from shapely.ops import unary_union

from common import CITY_IDS, SERVER_ROOT, confined, digest, distance, load, save


SAMPLING_POLICY = {'version':2, 'radius_m':100, 'points_per_round':1500,
                   'max_rounds':3, 'target_start_panos':1000, 'count_mode':'unique_panos',
                   'stop_check':'after_completed_round', 'minimum_distance_m_exclusive':100}


def boundary(city):
    if city['id'] == 'tokyo':
        # Tokyo Metropolis also includes remote islands and extensive sea areas.
        # Use the union of the 23 named special wards for the urban city dataset.
        source = load(SERVER_ROOT/'code/trajectory/data/tokyo_23_wards.geojson')
        features = source['features']
        expected = {'Chiyoda','Chuo','Minato','Shinjuku','Bunkyo','Taito','Sumida',
                    'Koto','Shinagawa','Meguro','Ota','Setagaya','Shibuya','Nakano',
                    'Suginami','Toshima','Kita','Arakawa','Itabashi','Nerima',
                    'Adachi','Katsushika','Edogawa'}
        if (len(features) != 23 or
                {f['properties']['ward'] for f in features} != expected or
                len({f['properties']['osm_id'] for f in features}) != 23):
            raise ValueError('Tokyo requires exactly 23 distinct special wards')
        polygons = [shape(f['geometry']) for f in features]
        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                raise ValueError('Invalid Tokyo ward geometry')
            west,south,east,north = polygon.bounds
            if not (139.5 < west < east < 140 and 35.4 < south < north < 35.9):
                raise ValueError('Tokyo ward outside urban administrative bounds')
        geometry = unary_union(polygons)
        if not geometry.is_valid or geometry.is_empty:
            raise ValueError('Invalid Tokyo special-ward union')
        return {'type':'Feature','geometry':mapping(geometry),'properties':{
            **city, 'display_name':'Tokyo 23 special wards, Japan',
            'source':'OpenStreetMap Nominatim; union of 23 special-ward administrative boundaries',
            'scope':'tokyo_23_special_wards', 'ward_count':23,
            'wards':[f['properties'] for f in features],
            'source_hash':digest(source)}}
    city_name, _, country = city['name'].rpartition(',')
    city_name = city_name.strip() or city['name']
    queries = [{'city': city_name, 'country': country.strip()}, {'q': city['name']},
               {'q': 'City of '+city['name']}, {'q': 'Municipality of '+city['name']}]
    overrides = {'sydney': 'City of Sydney, Australia', 'singapore': 'Republic of Singapore',
                 'melbourne': 'City of Melbourne, Victoria, Australia'}
    if city['id'] in overrides:
        queries.insert(0, {'q': overrides[city['id']]})
    for query in queries:
        for attempt in range(4):
            time.sleep(max(1.1, 2**attempt if attempt else 1.1))
            try:
                response = requests.get('https://nominatim.openstreetmap.org/search',
                    params={'format':'jsonv2', 'limit':10, 'polygon_geojson':1,
                            'polygon_threshold':'0.0005', **query},
                    headers={'User-Agent':'streetview-trajectory-research/1.0', 'Accept-Language':'en'}, timeout=45)
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError('City boundary network unavailable; no points were replaced or sampled outside the boundary') from None
                continue
            if response.status_code == 429 or response.status_code >= 500:
                continue
            response.raise_for_status()
            for row in response.json():
                if row.get('geojson', {}).get('type') in ('Polygon','MultiPolygon') and row.get('category', row.get('class')) == 'boundary' and row.get('type') == 'administrative':
                    geometry = shape(row['geojson'])
                    if not geometry.is_valid or geometry.is_empty:
                        raise ValueError('Invalid administrative boundary: '+city['id'])
                    return {'type':'Feature','geometry':row['geojson'], 'properties':{
                        **city, 'osm_type':row['osm_type'], 'osm_id':row['osm_id'],
                        'display_name':row['display_name'], 'query':query,
                        'source':'OpenStreetMap Nominatim', 'fetched_at':time.time()}}
            break
    raise RuntimeError('Administrative boundary unresolved: '+city['id'])


def ring_area(ring):
    xy = [(math.radians(x), math.sin(math.radians(y))) for x,y in ring]
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(xy, xy[1:]+xy[:1])))/2


def sample_points(feature, seed, count=1500, min_distance=100.0, max_attempts=2000000, previous=()):
    geo = feature['geometry']
    coordinates = [geo['coordinates']] if geo['type']=='Polygon' else geo['coordinates']
    polygons = [shape({'type':'Polygon','coordinates':p}) for p in coordinates]
    weights = [ring_area(p[0])-sum(ring_area(h) for h in p[1:]) for p in coordinates]
    if any(w <= 0 for w in weights):
        raise ValueError('Nonpositive polygon area')
    rng = random.Random(int(digest([seed, feature['properties']['id']]),16))
    points, grid = [], {}
    def cell_for(point):
        lat,lon = math.radians(point['lat']),math.radians(point['lon'])
        xyz = (6371000*math.cos(lat)*math.cos(lon),6371000*math.cos(lat)*math.sin(lon),6371000*math.sin(lat))
        return tuple(math.floor(v/min_distance) for v in xyz)
    for point in previous:
        grid.setdefault(cell_for(point),[]).append(point)
    attempts = 0
    while len(points) < count and attempts < max_attempts:
        polygon = rng.choices(polygons, weights=weights, k=1)[0]
        west,south,east,north = polygon.bounds
        # Keep the selected component until a point inside it is found: weighting
        # must not be multiplied by its bounding-box rejection rate.
        while attempts < max_attempts:
            attempts += 1
            sin_lat = rng.uniform(math.sin(math.radians(south)), math.sin(math.radians(north)))
            point = {'lat':math.degrees(math.asin(sin_lat)), 'lon':rng.uniform(west,east)}
            if polygon.contains(Point(point['lon'], point['lat'])):
                break
        else:
            break
        cell = cell_for(point)
        neighbors = [q for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                     for q in grid.get((cell[0]+dx,cell[1]+dy,cell[2]+dz), [])]
        if any(distance(point,q) <= min_distance for q in neighbors):
            continue
        grid.setdefault(cell,[]).append(point)
        points.append(point)
    if len(points) != count:
        raise RuntimeError(f'Sampling budget exhausted: {len(points)}/{count}; no threshold relaxation')
    # Independent full pairwise check; 1.1M pairs per city, bounded memory.
    minimum = min((distance(a,b) for i,a in enumerate(points) for b in points[i+1:]+list(previous)), default=None)
    full_geometry=shape(geo)
    if (minimum is not None and minimum <= min_distance) or not all(full_geometry.contains(Point(p['lon'],p['lat'])) for p in points):
        raise ValueError('Independent sampling validation failed')
    return points, minimum, attempts


def extend_city_round(run, city_id, policy, round_number):
    """Freeze an additional round atomically; never replace earlier candidates."""
    path=run/'state'/f'samples_{city_id}.json'
    saved=load(path)
    rounds=saved.setdefault('rounds',[{'round':1,'count':len(saved['points'])}])
    if round_number <= len(rounds):
        return saved['points']
    if round_number != len(rounds)+1 or round_number > policy['max_rounds']:
        raise ValueError('Invalid sampling round')
    feature=load(run/'state'/f'boundary_{city_id}.json')
    if digest(feature)!=saved['boundary_hash']:
        raise ValueError('Frozen boundary changed')
    seed=saved['seed']+f':round:{round_number}'
    points,minimum,attempts=sample_points(feature,seed,count=policy['points_per_round'],
        min_distance=policy['minimum_distance_m_exclusive'],previous=saved['points'])
    start=len(saved['points'])
    template=saved['points'][0]
    for i,point in enumerate(points,start):
        saved['points'].append({'sample_id':f'{city_id}__{i:06d}','city_id':city_id,
            'city_rank':template['city_rank'],'name':template['name'],'sample_index':i,
            'sampling_round':round_number,**point})
    saved['minimum_distance_m']=min(saved['minimum_distance_m'],minimum)
    rounds.append({'round':round_number,'count':len(points),'seed':seed,'attempts':attempts})
    save(path,saved)
    return saved['points']


def refresh_manifest(run,output,config):
    samples=[]
    for city_id in CITY_IDS:
        path=run/'state'/f'samples_{city_id}.json'
        if path.exists():
            samples.extend(load(path)['points'])
    save(run/'state/frozen.json',{'config_hash':digest(config),'samples':samples,'samples_hash':digest(samples)})
    from run_city import write_jsonl
    write_jsonl(output/'sampling_manifest.jsonl',samples)
    return samples


def prepare(run, output, dataset, seed):
    catalog = {c['id']:c for c in load(SERVER_ROOT/'code/trajectory/data/trajectory_routes.json')}
    config = {'dataset_id':dataset,'seed':seed,'city_ids':CITY_IDS,'points_per_city':1500,
              'minimum_distance_m_exclusive':100,'sampling_algorithm':'MT19937/SHA256 spherical-area-components + sequential distance rejection v1',
              'base_model':'openai/gpt-5.6-luna','model':'openai/gpt-5.6-luna:batch',
              'reasoning_effort':'high','max_tokens':20000,'max_expansions':2000,'search_seconds':300}
    config_path = output/'dataset_config.json'
    if config_path.exists():
        saved_config=load(config_path)
        legacy={k:v for k,v in saved_config.items() if k!='sampling_policy'}
        if 'first_round_points_per_city' in legacy:
            legacy['points_per_city']=legacy.pop('first_round_points_per_city')
        if legacy != config:
            raise RuntimeError('Frozen configuration mismatch')
        config=saved_config
    else:
        config['first_round_points_per_city']=config.pop('points_per_city')
        config['sampling_policy']=dict(SAMPLING_POLICY)
        save(config_path,config)
    features, samples = [], []
    for rank,city_id in enumerate(CITY_IDS,1):
        city = catalog[city_id]
        cached = confined(run/'state'/f'boundary_{city_id}.json')
        feature = load(cached) if cached.exists() else boundary(city)
        if not cached.exists():
            save(cached,feature)
        points_path = run/'state'/f'samples_{city_id}.json'
        if points_path.exists():
            saved = load(points_path)
            if saved['boundary_hash'] != digest(feature) or saved['seed'] != seed:
                raise RuntimeError('Frozen sampling input mismatch')
        else:
            points, minimum, attempts = sample_points(feature,seed)
            rows = [{'sample_id':f'{city_id}__{i:06d}','city_id':city_id,'city_rank':rank,
                     'name':city['name'],'sample_index':i, **p} for i,p in enumerate(points)]
            saved = {'seed':seed,'boundary_hash':digest(feature),'minimum_distance_m':minimum,
                     'attempts':attempts,'points':rows}
            save(points_path,saved)
        features.append(feature)
        samples.extend(saved['points'])
        print(f'Frozen {rank}/30 {city_id}: {len(saved["points"])} points, min distance {saved["minimum_distance_m"]:.3f}m',flush=True)
    manifest = {'config_hash':digest(config), 'samples':samples, 'samples_hash':digest(samples)}
    save(run/'state/frozen.json',manifest)
    save(output/'city_boundaries.geojson',{'type':'FeatureCollection','features':features})
    from run_city import write_jsonl
    write_jsonl(output/'sampling_manifest.jsonl',samples)
    return config, samples
