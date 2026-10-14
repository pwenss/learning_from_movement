import io
import json
import pathlib
import re
import unicodedata
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
from pyproj import Transformer
import shapefile

FIGURE_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = FIGURE_ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
WORK = FIGURE_ROOT / 'data' / 'cities'
SPEC = FIGURE_ROOT.parent / 'evaluation' / 'visual_trajectory_reconstruction_spec.md'
LAND_URL = 'https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip'
PLACES_URL = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip'

# Reference coordinates disambiguate identically named cities; plotted points
# are taken from Natural Earth's populated-places database.
CITY_DEFINITIONS = [
    ('nyc_new_york', 'New York', -74.0060, 40.7128),
    ('sf_san_francisco', 'San Francisco', -122.4194, 37.7749),
    ('chicago', 'Chicago', -87.6298, 41.8781),
    ('philadelphia', 'Philadelphia', -75.1652, 39.9526),
    ('toronto', 'Toronto', -79.3832, 43.6532),
    ('vancouver', 'Vancouver', -123.1207, 49.2827),
    ('mexico_city', 'Mexico City', -99.1332, 19.4326),
    ('sao_paulo', 'Sao Paulo', -46.6333, -23.5505),
    ('rio_de_janeiro', 'Rio de Janeiro', -43.1729, -22.9068),
    ('buenos_aires', 'Buenos Aires', -58.3816, -34.6037),
    ('london', 'London', -0.1276, 51.5072),
    ('paris', 'Paris', 2.3522, 48.8566),
    ('berlin', 'Berlin', 13.4050, 52.5200),
    ('madrid', 'Madrid', -3.7038, 40.4168),
    ('rome', 'Rome', 12.4964, 41.9028),
    ('amsterdam', 'Amsterdam', 4.9041, 52.3676),
    ('zurich', 'Zurich', 8.5417, 47.3769),
    ('vienna', 'Vienna', 16.3738, 48.2082),
    ('stockholm', 'Stockholm', 18.0686, 59.3293),
    ('prague', 'Prague', 14.4378, 50.0755),
    ('tokyo', 'Tokyo', 139.6917, 35.6895),
    ('osaka', 'Osaka', 135.5023, 34.6937),
    ('seoul', 'Seoul', 126.9780, 37.5665),
    ('taipei', 'Taipei', 121.5654, 25.0330),
    ('singapore', 'Singapore', 103.8198, 1.3521),
    ('bangkok', 'Bangkok', 100.5018, 13.7563),
    ('kuala_lumpur', 'Kuala Lumpur', 101.6869, 3.1390),
    ('sydney', 'Sydney', 151.2093, -33.8688),
    ('melbourne', 'Melbourne', 144.9631, -37.8136),
    ('auckland', 'Auckland', 174.7633, -36.8485),
]
TEST_IDS = {'philadelphia', 'buenos_aires', 'berlin', 'osaka', 'melbourne'}


def reader(file):
    with zipfile.ZipFile(file) as archive:
        shp = next(n for n in archive.namelist() if n.endswith('.shp'))
        stem = shp[:-4]
        return shapefile.Reader(shp=io.BytesIO(archive.read(shp)),
            shx=io.BytesIO(archive.read(stem + '.shx')),
            dbf=io.BytesIO(archive.read(stem + '.dbf')), encoding='utf-8')


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value))
                   if not unicodedata.combining(c)).lower().strip()


spec = SPEC.read_text(encoding='utf-8')
city_table = spec.split('#### 1.1.2 City Counts by Continent', 1)[1].split('### 1.2', 1)[0]
split = dict(re.findall(r'\| `([a-z_]+)` \| [\d,]+ \| (Train|OOD test) \|', city_table))
assert set(split) == {c[0] for c in CITY_DEFINITIONS}
assert {c for c, s in split.items() if s == 'OOD test'} == TEST_IDS
places = list(reader(WORK / 'places.zip').iterShapeRecords())
locations = []
for city_id, label, ref_lon, ref_lat in CITY_DEFINITIONS:
    candidates = []
    for row in places:
        record = row.record.as_dict()
        names = [v for k, v in record.items() if 'name' in k.lower()]
        if normalize(label) in {normalize(v) for v in names}:
            lon, lat = row.shape.points[0]
            candidates.append(((lon-ref_lon)**2+(lat-ref_lat)**2, lon, lat))
    assert candidates, f'No populated-place match for {label}'
    distance, lon, lat = min(candidates)
    assert distance < 1.0, (label, distance)
    locations.append(dict(city_id=city_id, label=label, longitude=lon,
                          latitude=lat, split=split[city_id]))
assert len(locations) == 30 and sum(c['split']=='OOD test' for c in locations)==5

PROJ = '+proj=eck3 +lon_0=0 +datum=WGS84 +units=m +no_defs'
transform = Transformer.from_crs('EPSG:4326', PROJ, always_xy=True)

plt.rcParams.update({'font.family':'Arial', 'font.size':12,
                     'svg.fonttype':'none', 'axes.linewidth':0.5})
fig = plt.figure(figsize=(12, 4.8), facecolor='none')
ax = fig.add_axes([0.015, 0.015, 0.97, 0.97])
ax.set_aspect('equal')
ax.set_axis_off()

# Compute map extents without drawing an ocean background or globe outline.
edge_lon = np.r_[np.linspace(-180,180,361), np.full(181,180),
                 np.linspace(180,-180,361), np.full(181,-180)]
edge_lat = np.r_[np.full(361,85), np.linspace(85,-58,181),
                 np.full(361,-58), np.linspace(-58,85,181)]
ex, ey = transform.transform(edge_lon, edge_lat)

for shape in reader(WORK / 'land.zip').iterShapes():
    # Omit Antarctica geometry rather than merely hiding it behind a crop.
    if shape.bbox[3] < -60:
        continue
    vertices, codes = [], []
    breaks = list(shape.parts) + [len(shape.points)]
    for start,end in zip(breaks[:-1],breaks[1:]):
        ring = np.asarray(shape.points[start:end])
        x,y = transform.transform(ring[:,0],ring[:,1])
        projected = np.column_stack([x,y])
        vertices.extend(projected)
        codes.extend([Path.MOVETO]+[Path.LINETO]*(len(projected)-2)+[Path.CLOSEPOLY])
    ax.add_patch(PathPatch(Path(np.asarray(vertices),codes), facecolor='#E8ECEE',
                          edgecolor='#CED5D8',linewidth=0.3,zorder=2))

COLORS = {'Train':'#0072B2','OOD test':'#D55E00'}
artists = []
for group in ['Train','OOD test']:
    rows = [c for c in locations if c['split']==group]
    x,y = transform.transform([c['longitude'] for c in rows],[c['latitude'] for c in rows])
    scatter = ax.scatter(x,y,s=196,facecolors='none',edgecolors=COLORS[group],linewidths=5,zorder=5 if group=='Train' else 6)
    scatter.set_gid('train-city-markers' if group=='Train' else 'test-city-markers')
    artists.append(scatter)
assert sum(len(artist.get_offsets()) for artist in artists)==30
ax.set_xlim(min(ex)*1.012,max(ex)*1.012)
ax.set_ylim(min(ey)*1.014,max(ey)*1.014)
provenance = json.dumps(dict(projection=PROJ,land_source=LAND_URL,
    city_point_source=PLACES_URL,split_source=str(SPEC),cities=locations,
    style=dict(land_only=True,antarctica=False,legend=False,graticule=False,
               background='transparent',marker_area_pt2=196,marker_fill='none',
               marker_edge_width_pt=5,land_color='#E8ECEE',
               point_colors=COLORS,palette_reference='https://jfly.uni-koeln.de/color/')),
    ensure_ascii=False)
fig.savefig(OUT/'trajectory_cities_train_test.svg',transparent=True,metadata={
    'Title':'Trajectory dataset cities: train and test',
    'Description':provenance,'Creator':'Matplotlib; geographic data: Natural Earth'})
fig.savefig(OUT/'trajectory_cities_train_test.png',dpi=600,transparent=True,metadata={
    'Title':'Trajectory dataset cities: train and test','Description':provenance})
plt.close(fig)
print(json.dumps({'cities':30,'train':25,'test':5,'projection':'Eckert III',
                 'png':str(OUT/'trajectory_cities_train_test.png'),
                 'svg':str(OUT/'trajectory_cities_train_test.svg')}),flush=True)
