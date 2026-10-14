"""Render georeferenced route figures with Google basemaps, 50 x 30 cm."""
import concurrent.futures
import datetime
import io
import json
import math
import pathlib
import time
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from pyproj import Geod

plt.rcParams.update({'pdf.fonttype':42, 'ps.fonttype':42, 'font.family':'Arial'})

FIGURE_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = FIGURE_ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
WORK = FIGURE_ROOT / '.work' / 'routes'
INPUT = FIGURE_ROOT / 'data' / 'routes'
STATUS = WORK / 'status.json'
ROUTE_COLOR = '#098579'
WIDTH_CM, HEIGHT_CM, DPI = 50, 30, 300
MARKER_DIAMETER_PT = 70
MARKER_OUTLINE_PT = 1
LINE_WIDTH_PT = 1


def status(state, **details):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state=state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def world_point(lat, lon, zoom):
    scale = 256 * 2**zoom
    return ((lon + 180) / 360 * scale,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * scale)


def geographic_point(x, y, zoom):
    scale = 256 * 2**zoom
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/scale)))),
            x/scale*360-180)


def add_scale_bar(ax, left, top, width, height, zoom):
    # Measure the horizontal bar at its actual latitude using WGS84 geodesics.
    geod = Geod(ellps='WGS84')
    right_x, bar_y = 0.975, 0.025
    world_right = left + right_x*width
    world_y = top + (1-bar_y)*height
    lat, lon_right = geographic_point(world_right, world_y, zoom)
    _, lon_target = geographic_point(world_right-0.14*width, world_y, zoom)
    target = geod.inv(lon_target, lat, lon_right, lat)[2]
    unit = 10**math.floor(math.log10(target))
    distance = max(v*unit for v in (1,2,5,10) if v*unit <= target)
    low, high = 0.0, float(width)
    for _ in range(50):
        length = (low+high)/2
        _, lon_left = geographic_point(world_right-length, world_y, zoom)
        actual = geod.inv(lon_left, lat, lon_right, lat)[2]
        if actual < distance:
            low = length
        else:
            high = length
    length = (low+high)/2
    _, lon_left = geographic_point(world_right-length, world_y, zoom)
    actual = geod.inv(lon_left, lat, lon_right, lat)[2]
    assert abs(actual-distance) < 1e-5
    left_x = right_x-length/width
    tick = 2/(HEIGHT_CM/2.54*72)
    halo = [effects.withStroke(linewidth=2,foreground='white')]
    for xs, ys in (([left_x,right_x],[bar_y,bar_y]),
                   ([left_x,left_x],[bar_y-tick,bar_y+tick]),
                   ([right_x,right_x],[bar_y-tick,bar_y+tick])):
        ax.plot(xs,ys,transform=ax.transAxes,color='#222222',linewidth=1,
                solid_capstyle='butt',path_effects=halo,zorder=8)
    label = f'{distance:g} m' if distance < 1000 else f'{distance/1000:g} km'
    ax.text((left_x+right_x)/2,bar_y+4/(HEIGHT_CM/2.54*72),label,
            transform=ax.transAxes,ha='center',va='bottom',fontfamily='Arial',
            fontsize=6,color='#222222',path_effects=[effects.withStroke(linewidth=1.5,foreground='white')],zorder=9)
    return dict(distance_m=distance,actual_geodesic_distance_m=actual,
                latitude=lat,longitude_endpoints=[lon_left,lon_right],
                native_length_pixels=length,font='Arial',font_size_pt=6,
                position='bottom right',label=label)


def fetch_tile(mode, x, y, z):
    tile_dir = FIGURE_ROOT / 'data' / 'tiles' / mode / str(z)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f'{x}_{y}.png'
    if tile_path.exists():
        with Image.open(tile_path) as tile:
            tile.load()
            assert tile.size == (256,256)
            return tile.convert('RGB')
    params = dict(lyrs='m' if mode == 'street' else 's', x=x, y=y, z=z, hl='en')
    if mode == 'street':
        params['apistyle'] = 's.t:0|s.e:l|p.v:off'
    url = 'https://mt1.google.com/vt?' + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(2):
        try:
            request = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=8) as response:
                raw = response.read()
            with Image.open(io.BytesIO(raw)) as tile:
                tile.load()
                assert tile.size == (256,256)
                result = tile.convert('RGB')
            tile_path.write_bytes(raw)
            return result
        except Exception as error:
            last_error = error
    raise RuntimeError(f'Unable to retrieve tile {mode}/{z}/{x}/{y}: {last_error}')


def render(mode, expected_id):
    manifest_file = INPUT / f'{mode}_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    route_id = manifest.get('trajectory_id') or manifest.get('id')
    assert route_id == expected_id, (route_id, expected_id)
    frames = manifest['frames']
    assert len(frames) == 8
    coords = [(float(f.get('latitude', f.get('lat'))),float(f.get('longitude', f.get('lon')))) for f in frames]
    points0 = np.asarray([world_point(lat,lon,0) for lat,lon in coords])
    spans = np.ptp(points0,axis=0)
    # The route occupies at most 68% of width and 72% of height, leaving room
    # for numbered circles at the true observation centers.
    view_width0 = max(spans[0]/0.68, spans[1]/0.72*(WIDTH_CM/HEIGHT_CM))
    zoom = min(22 if mode=='street' else 20,
               max(16,math.ceil(math.log2(3400/view_width0))))
    points = np.asarray([world_point(lat,lon,zoom) for lat,lon in coords])
    native_width = max(5,int(math.ceil(view_width0*2**zoom/5))*5)
    native_height = native_width*3//5
    center = (points.min(axis=0)+points.max(axis=0))/2
    left,top = np.floor(center-np.asarray([native_width,native_height])/2).astype(int)
    right,bottom = left+native_width,top+native_height
    x0,x1 = left//256,(right-1)//256
    y0,y1 = top//256,(bottom-1)//256
    tiles = [(x,y) for x in range(x0,x1+1) for y in range(y0,y1+1)]
    status('downloading',route=route_id,zoom=zoom,tiles=len(tiles))
    mosaic = Image.new('RGB',((x1-x0+1)*256,(y1-y0+1)*256))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_tile,mode,x,y,zoom):(x,y) for x,y in tiles}
        for future in concurrent.futures.as_completed(futures):
            x,y = futures[future]
            mosaic.paste(future.result(),((x-x0)*256,(y-y0)*256))
    background = mosaic.crop((left-x0*256,top-y0*256,right-x0*256,bottom-y0*256))
    assert background.size == (native_width,native_height)
    assert np.asarray(background).std() > 10, 'Blank basemap'
    background.save(WORK / f'{mode}_basemap.png')
    xy = points - np.asarray([left,top])
    # The native mosaic and the annotations use the same Web Mercator grid.
    roundtrip = xy+np.asarray([left,top])
    assert np.max(np.abs(roundtrip-points)) < 1e-7
    assert np.all(xy[:,0]>0) and np.all(xy[:,0]<native_width)
    assert np.all(xy[:,1]>0) and np.all(xy[:,1]<native_height)
    fig = plt.figure(figsize=(WIDTH_CM/2.54, HEIGHT_CM/2.54),facecolor='white')
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(background,extent=(0,native_width,native_height,0),origin='upper',interpolation='lanczos')
    ax.set_xlim(0,native_width)
    ax.set_ylim(native_height,0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.plot(xy[:,0],xy[:,1],color=ROUTE_COLOR,linewidth=LINE_WIDTH_PT,
            solid_capstyle='round',solid_joinstyle='round',zorder=3)
    radius = MARKER_DIAMETER_PT/(WIDTH_CM/2.54*72)*native_width/2
    for i,(x,y) in enumerate(xy):
        ax.add_patch(Circle((x,y),radius,facecolor='white',
                           edgecolor=ROUTE_COLOR,linewidth=MARKER_OUTLINE_PT,zorder=5))
        ax.text(x,y,str(i),ha='center',va='center',fontfamily='Arial',
                fontsize=7,fontweight='normal',color=ROUTE_COLOR,zorder=6)
    scale_bar = add_scale_bar(ax,left,top,native_width,native_height,zoom)
    # Basemap attribution is separate from geographic labels.
    credit = 'Google Maps' if mode=='street' else 'Google satellite imagery'
    ax.text(0.012,0.012,credit,transform=ax.transAxes,fontsize=12,fontfamily='Arial',
            color='#444444' if mode=='street' else 'white',ha='left',va='bottom',
            path_effects=[effects.withStroke(linewidth=2,foreground='white' if mode=='street' else '#333333')],zorder=7)
    prefix = ('streetview_' if mode=='street' else 'satellite_')+route_id+'_50x30cm'
    provenance = dict(route_id=route_id,source_trajectory_id=manifest.get('source_trajectory_id',route_id),
        dimensions_cm=[50,30],dpi=DPI,projection='Web Mercator',zoom=zoom,
        map_source='https://mt1.google.com/vt',geographic_labels=False,
        frame_centers=[dict(number=i,latitude=lat,longitude=lon) for i,(lat,lon) in enumerate(coords)],
        colors=dict(route=ROUTE_COLOR,marker_edge=ROUTE_COLOR,marker_fill='white'),
        annotation_style=dict(marker_diameter_pt=MARKER_DIAMETER_PT,marker_outline_pt=MARKER_OUTLINE_PT,
                              line_width_pt=LINE_WIDTH_PT,font='Arial',font_size_pt=7,final_frame_highlight=False),
        scale_bar=scale_bar,
        native_basemap_size=background.size,viewport_pixels=dict(left=int(left),top=int(top),width=native_width,height=native_height))
    fig.savefig(OUT/(prefix+'.png'),dpi=DPI,metadata={'Description':json.dumps(provenance),'Title':route_id})
    fig.savefig(OUT/(prefix+'.pdf'),dpi=DPI,metadata={'Title':route_id,'Subject':json.dumps(provenance),
        'Creator':'Matplotlib; basemap: Google'})
    plt.close(fig)
    status('rendered',route=route_id,dimensions_cm=[50,30],zoom=zoom,
           native_size=list(background.size),png=str(OUT/(prefix+'.png')),pdf=str(OUT/(prefix+'.pdf')))


if __name__=='__main__':
    try:
        render('street','nyc_new_york__001369')
        render('satellite','madrid__satellite_v4__000217')
        status('complete',figures=2,dimensions_cm=[50,30],dpi=DPI)
    except Exception:
        import traceback
        status('failed',error=traceback.format_exc())
        raise
