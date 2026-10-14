"""Generate independent satellite-image trajectories for the 30-city dataset.

Completed routes from this pipeline version are resumable. City indexes are
atomically refreshed after every route. Only the city boundaries are shared
with the Street View production dataset; trajectories have independent IDs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import requests
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

EARTH_RADIUS_M = 6_378_137.0
TILE_SIZE = 256
DEFAULT_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
PIPELINE_VERSION = "satellite_v4_extent_100_400_truncated_normal_dynamic_zoom"
DISTANCE_RATIO_BOUNDS = (0.2, 0.6)
HEADING_TURN_BOUNDS = (-30.0, 30.0)
EXTENT_CHANGE_BOUNDS = (-50.0, 50.0)
EXTENT_BOUNDS = (100.0, 400.0)
CITY_IDS = [
    "nyc_new_york", "sf_san_francisco", "chicago", "philadelphia", "toronto", "vancouver",
    "mexico_city", "sao_paulo", "rio_de_janeiro", "buenos_aires", "london", "paris",
    "berlin", "madrid", "rome", "amsterdam", "zurich", "vienna", "stockholm", "prague",
    "tokyo", "osaka", "seoul", "taipei", "singapore", "bangkok", "kuala_lumpur",
    "sydney", "melbourne", "auckland",
]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def wrap_heading(value: float) -> float:
    return value % 360.0


def turn(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    return min(delta, 360.0 - delta)


def truncated_normal(rng: random.Random, mean: float, sigma: float, low: float, high: float) -> float:
    if sigma <= 0 or low >= high:
        raise ValueError("Invalid truncated normal parameters")
    while True:
        value = rng.gauss(mean, sigma)
        if low <= value <= high:
            return value


def local_to_geo(anchor_lat: float, anchor_lon: float, x: float, y: float) -> tuple[float, float]:
    lat = anchor_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = anchor_lon + math.degrees(x / (EARTH_RADIUS_M * max(1e-9, math.cos(math.radians(anchor_lat)))))
    return lat, lon


def geo_to_tile(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, lat))
    scale = 2**zoom
    px = (lon + 180.0) / 360.0 * scale * TILE_SIZE
    sin_lat = math.sin(math.radians(lat))
    py = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale * TILE_SIZE
    return px, py


def tile_meters_per_pixel(lat: float, zoom: int) -> float:
    return 2 * math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat)) / (TILE_SIZE * 2**zoom)


def choose_zoom(lat: float, extent: float, size: int, max_zoom: int = 22) -> int:
    required = math.log2(2 * math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat)) * size / (TILE_SIZE * extent))
    return min(max_zoom, max(1, math.ceil(required)))


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return [part for item in geometry.geoms for part in polygon_parts(item)]


def local_footprint(frame: dict) -> Polygon:
    half = frame["extent_m"] / 2.0
    heading = math.radians(frame["heading_deg"])
    forward = (math.sin(heading), math.cos(heading))
    right = (math.cos(heading), -math.sin(heading))
    x0, y0 = frame["center_x_m"], frame["center_y_m"]
    return Polygon([(x0 + sf * half * forward[0] + sr * half * right[0],
                     y0 + sf * half * forward[1] + sr * half * right[1])
                    for sf, sr in ((-1, -1), (-1, 1), (1, 1), (1, -1))])


def geographic_footprint(anchor_lat: float, anchor_lon: float, frame: dict) -> Polygon:
    points = [local_to_geo(anchor_lat, anchor_lon, x, y) for x, y in local_footprint(frame).exterior.coords]
    return Polygon([(lon, lat) for lat, lon in points])


def generate_local(rng: random.Random) -> tuple[list[dict], list[tuple[float, float]]]:
    centers = [(0.0, 0.0)]
    frames = []
    extent = rng.uniform(*EXTENT_BOUNDS)
    heading = rng.uniform(0.0, 360.0)
    for index in range(8):
        if index:
            change = truncated_normal(rng, 0.0, 50.0,
                                     max(EXTENT_CHANGE_BOUNDS[0], EXTENT_BOUNDS[0] - extent),
                                     min(EXTENT_CHANGE_BOUNDS[1], EXTENT_BOUNDS[1] - extent))
            extent += change
            if index < 7:
                heading = wrap_heading(heading + truncated_normal(rng, 0.0, 30.0, *HEADING_TURN_BOUNDS))
        ratio = truncated_normal(rng, 0.4, 0.2, *DISTANCE_RATIO_BOUNDS) if index < 7 else None
        distance = ratio * extent if ratio is not None else None
        x, y = centers[-1]
        frames.append({
            "frame_id": index,
            "center_x_m": x,
            "center_y_m": y,
            "heading_deg": heading,
            "extent_m": extent,
            "distance_to_next_m": distance,
            "distance_extent_ratio": ratio,
            "turn_from_previous_deg": None if index == 0 else (heading - frames[-1]["heading_deg"] + 180) % 360 - 180,
            "extent_change_from_previous_m": None if index == 0 else extent - frames[-1]["extent_m"],
        })
        if index < 7:
            centers.append((x + distance * math.sin(math.radians(heading)), y + distance * math.cos(math.radians(heading))))
    line = LineString(centers)
    return frames, centers if line.is_simple else []


def place_trajectory(geometry, rng: random.Random) -> tuple[float, float, list[dict]]:
    polygons = polygon_parts(geometry)
    boundary = unary_union(polygons)
    minx, miny, maxx, maxy = geometry.bounds
    for _ in range(4000):
        anchor_lon = rng.uniform(minx, maxx)
        anchor_lat = rng.uniform(miny, maxy)
        anchor = Point(anchor_lon, anchor_lat)
        if not any(poly.contains(anchor) for poly in polygons):
            continue
        frames, centers = generate_local(rng)
        if not centers:
            continue
        geographic = [local_to_geo(anchor_lat, anchor_lon, x, y) for x, y in centers]
        if not all(any(poly.contains(Point(lon, lat)) for poly in polygons) for lat, lon in geographic):
            continue
        # Keep the complete rotated footprint inside the downloaded boundary.
        if not all(boundary.covers(geographic_footprint(anchor_lat, anchor_lon, frame)) for frame in frames):
            continue
        for frame, (lat, lon) in zip(frames, geographic):
            frame["latitude"] = lat
            frame["longitude"] = lon
        return anchor_lat, anchor_lon, frames
    raise RuntimeError("Unable to place a trajectory inside the supplied city boundary")


class TileCoverageError(RuntimeError):
    pass


class TileStore:
    def __init__(self, root: Path, url_template: str, timeout: float = 30.0, workers: int = 16):
        self.root = root
        self.url_template = url_template
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.thread_state = threading.local()

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get(self, zoom: int, x: int, y: int) -> Image.Image:
        scale = 2**zoom
        x_wrapped = x % scale
        if y < 0 or y >= scale:
            return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (239, 239, 239))
        path = self.root / str(zoom) / str(x_wrapped) / f"{y}.jpg"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = self.url_template.format(x=x_wrapped, y=y, z=zoom)
            last = None
            if not hasattr(self.thread_state, "session"):
                self.thread_state.session = requests.Session()
                self.thread_state.session.headers.update({"User-Agent": "satellite-trajectory-research/2.0"})
            for attempt in range(4):
                try:
                    response = self.thread_state.session.get(url, timeout=self.timeout)
                    if response.status_code == 404:
                        raise TileCoverageError(f"Tile coverage unavailable: {url}")
                    response.raise_for_status()
                    with Image.open(io.BytesIO(response.content)) as image:
                        image.verify()
                    temporary = path.with_suffix(".jpg.tmp")
                    temporary.write_bytes(response.content)
                    temporary.replace(path)
                    break
                except (requests.RequestException, OSError) as error:
                    last = error
                    if attempt < 3:
                        time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"Tile download failed: {url}: {last}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def mosaic(self, lat: float, lon: float, zoom: int, side_px: int) -> tuple[Image.Image, float, float]:
        center_x, center_y = geo_to_tile(lat, lon, zoom)
        min_x, max_x = math.floor((center_x - side_px / 2 - 1) / TILE_SIZE), math.floor((center_x + side_px / 2 + 1) / TILE_SIZE)
        min_y, max_y = math.floor((center_y - side_px / 2 - 1) / TILE_SIZE), math.floor((center_y + side_px / 2 + 1) / TILE_SIZE)
        canvas = Image.new("RGB", ((max_x - min_x + 1) * TILE_SIZE, (max_y - min_y + 1) * TILE_SIZE))
        coordinates = [(x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1)]
        futures = [self.executor.submit(self.get, zoom, x, y) for x, y in coordinates]
        tiles, errors = [], []
        # Drain every request before retrying a lower zoom or a new placement.
        for future in futures:
            try:
                tiles.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise next((error for error in errors if not isinstance(error, TileCoverageError)), errors[0])
        for (x, y), tile in zip(coordinates, tiles):
            canvas.paste(tile, ((x - min_x) * TILE_SIZE, (y - min_y) * TILE_SIZE))
        origin_x = min_x * TILE_SIZE
        origin_y = min_y * TILE_SIZE
        return canvas, center_x - origin_x, center_y - origin_y


def render_frame(store: TileStore, frame: dict, output: Path, size: int, max_zoom: int = 22) -> None:
    requested_zoom = choose_zoom(frame["latitude"], frame["extent_m"], size, max_zoom)
    for zoom in range(requested_zoom, min(18, requested_zoom) - 1, -1):
        mpp = tile_meters_per_pixel(frame["latitude"], zoom)
        crop_side = max(32, math.ceil(frame["extent_m"] * math.sqrt(2.0) / mpp) + 8)
        try:
            mosaic, center_x, center_y = store.mosaic(frame["latitude"], frame["longitude"], zoom, crop_side)
            break
        except TileCoverageError:
            if zoom == min(18, requested_zoom):
                raise
    mpp = tile_meters_per_pixel(frame["latitude"], zoom)
    left = round(center_x - crop_side / 2)
    top = round(center_y - crop_side / 2)
    source = mosaic.crop((left, top, left + crop_side, top + crop_side))
    rotated = source.rotate(frame["heading_deg"], resample=Image.Resampling.BICUBIC, expand=True)
    target_side = max(16, round(frame["extent_m"] / mpp))
    cx, cy = rotated.width / 2, rotated.height / 2
    crop_left, crop_top = round(cx - target_side / 2), round(cy - target_side / 2)
    crop = rotated.crop((crop_left, crop_top, crop_left + target_side, crop_top + target_side))
    crop.resize((size, size), Image.Resampling.LANCZOS).save(output, format="JPEG", quality=92, subsampling=0, optimize=True)
    frame.update({"requested_zoom": requested_zoom, "zoom": zoom, "source_meters_per_pixel": mpp,
                  "output_meters_per_pixel": frame["extent_m"] / size, "source_crop_pixels": crop.width,
                  "upsampled": crop.width < size})


def load_boundaries(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [payload]
    result = {}
    for feature in features:
        properties = feature.get("properties") or {}
        city_id = properties.get("id") or properties.get("city_id")
        if city_id:
            result[str(city_id)] = shape(feature.get("geometry") or feature)
    missing = [city for city in CITY_IDS if city not in result]
    if missing:
        raise RuntimeError(f"City boundaries missing: {', '.join(missing)}")
    return result


def route_complete(route_dir: Path, expected_frames: int = 8) -> bool:
    manifest = route_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        return value.get("pipeline_version") == PIPELINE_VERSION and len(value.get("frames", [])) == expected_frames and all((route_dir / frame["file"]).is_file() for frame in value["frames"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streetview-root", type=Path, required=True, help="Street View dataset root, e.g. /home/.../datasets/sv30_...")
    parser.add_argument("--output-root", type=Path, required=True, help="Satellite dataset output root")
    parser.add_argument("--boundaries", type=Path, default=None, help="GeoJSON boundaries; defaults to <streetview-root>/city_boundaries.geojson")
    parser.add_argument("--count-per-city", type=int, default=10)
    parser.add_argument("--total-count", type=int, default=None, help="Distribute an exact total evenly across selected cities")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-zoom", type=int, default=22, help="Upper limit for dynamic per-frame tile zoom")
    parser.add_argument("--tile-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--tile-cache", type=Path, default=None)
    parser.add_argument("--tile-url", default=DEFAULT_TILE_URL)
    parser.add_argument("--city", action="append", dest="cities", choices=CITY_IDS)
    args = parser.parse_args()
    selected_cities = list(dict.fromkeys(args.cities or CITY_IDS))
    if not 1 <= args.count_per_city <= 1000 or args.image_size < 128 or not 1 <= args.max_zoom <= 22 or not 1 <= args.tile_workers <= 32:
        raise SystemExit("count-per-city, image-size, max-zoom, or tile-workers is out of range")
    if args.total_count is not None and not len(selected_cities) <= args.total_count <= len(selected_cities) * 1000:
        raise SystemExit("total-count must allocate between 1 and 1000 trajectories to each selected city")
    boundaries = load_boundaries(args.boundaries or args.streetview_root / "city_boundaries.geojson")
    with TileStore(args.tile_cache or args.output_root / "tile_cache", args.tile_url, workers=args.tile_workers) as tile_store:
        generate_dataset(args, selected_cities, boundaries, tile_store)


def generate_dataset(args, selected_cities, boundaries, tile_store) -> None:
    catalog = []
    for city_index, city in enumerate(selected_cities):
        city_dir = args.output_root / "cities" / city
        city_dir.mkdir(parents=True, exist_ok=True)
        count = args.count_per_city if args.total_count is None else args.total_count // len(selected_cities) + (city_index < args.total_count % len(selected_cities))
        route_ids = [f"{city}__satellite_v4__{index:06d}" for index in range(count)]
        index_rows = []
        for route_index, route_id in enumerate(route_ids):
            route_dir = city_dir / "trajectory_sequences" / route_id
            if route_complete(route_dir):
                manifest = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
                index_rows.append({"id": route_id, "path": f"trajectory_sequences/{route_id}/manifest.json", "frame_count": len(manifest["frames"])})
                print(f"[{city}] {route_id}: existing", flush=True)
                continue
            rng = random.Random(args.seed + CITY_IDS.index(city) * 100_003 + route_index)
            route_dir.mkdir(parents=True, exist_ok=True)
            # Some city-boundary areas have no Google satellite tile coverage and
            # return HTTP 404. Keep the requested route count by resampling the
            # independent trajectory until all eight frames are renderable.
            for placement_attempt in range(200):
                anchor_lat, anchor_lon, frames = place_trajectory(boundaries[city], rng)
                for frame in frames:
                    frame["overlap_with_next"] = None
                for first, second in zip(frames, frames[1:]):
                    first_shape, second_shape = local_footprint(first), local_footprint(second)
                    first["overlap_with_next"] = first_shape.intersection(second_shape).area / first_shape.area
                try:
                    for frame in frames:
                        render_frame(tile_store, frame, route_dir / f"frame_{frame['frame_id']:02d}.jpg", args.image_size, args.max_zoom)
                except TileCoverageError as error:
                    print(f"[{city}] {route_id}: tile coverage retry {placement_attempt + 1}: {error}", flush=True)
                    continue
                break
            else:
                raise RuntimeError(f"Unable to render {route_id} after 200 tile-coverage attempts")
            payload = {
                "trajectory_id": route_id, "city_id": city, "name": city.replace("_", " "),
                "pipeline_version": PIPELINE_VERSION, "seed": args.seed + CITY_IDS.index(city) * 100_003 + route_index,
                "num_frames": 8, "num_control_points": 8, "anchor": {"latitude": anchor_lat, "longitude": anchor_lon},
                "extent_range_m": list(EXTENT_BOUNDS), "max_extent_change_m": 50, "distance_extent_ratio_range": list(DISTANCE_RATIO_BOUNDS),
                "sampling": {"method": "rejection_truncated_normal", "distance_ratio": {"mean": 0.4, "sigma": 0.2, "bounds": [0.2, 0.6]},
                             "heading_turn_deg": {"mean": 0, "sigma": 30, "bounds": list(HEADING_TURN_BOUNDS)},
                             "extent_change_m": {"mean": 0, "sigma": 50, "bounds": list(EXTENT_CHANGE_BOUNDS), "extent_bounds": list(EXTENT_BOUNDS)}},
                "max_turn_deg": 30, "aspect_ratio": "1:1", "heading_definition": "center_to_next_center",
                "gt_order": list(range(8)), "image_source": "Google satellite tiles (lyrs=s)",
                "tile_url": args.tile_url, "zoom_mode": "dynamic", "max_zoom": args.max_zoom,
                "zoom_levels": sorted({frame["zoom"] for frame in frames}), "image_size": [args.image_size, args.image_size],
                "validation": {
                    "extent_range_valid": all(EXTENT_BOUNDS[0] <= frame["extent_m"] <= EXTENT_BOUNDS[1] for frame in frames),
                    "extent_change_valid": all(abs(frames[i]["extent_m"] - frames[i - 1]["extent_m"]) <= 50 for i in range(1, 8)),
                    "distance_ratio_valid": all(0.2 <= frame["distance_extent_ratio"] <= 0.6 for frame in frames[:7]),
                    "heading_change_valid": all(turn(frames[i]["heading_deg"], frames[i - 1]["heading_deg"]) <= 30 for i in range(1, 8)),
                    "self_intersection": not LineString([(frame["center_x_m"], frame["center_y_m"]) for frame in frames]).is_simple,
                    "boundary_footprints_valid": True,
                },
                "frames": [{**frame, "file": f"frame_{frame['frame_id']:02d}.jpg", "heading": frame["heading_deg"], "lat": frame["latitude"], "lon": frame["longitude"]} for frame in frames],
            }
            atomic_json(route_dir / "manifest.json", payload)
            index_rows.append({"id": route_id, "path": f"trajectory_sequences/{route_id}/manifest.json", "frame_count": 8})
            atomic_json(city_dir / "index.json", index_rows)
            print(f"[{city}] {route_id}: generated", flush=True)
        atomic_json(city_dir / "index.json", index_rows)
        atomic_json(city_dir / "summary.json", {"city_id": city, "complete": len(index_rows), "image_count": len(index_rows) * 8})
        catalog.append({"city_id": city, "path": f"cities/{city}/index.json", "complete": len(index_rows)})
    existing = json.loads((args.output_root / "index.json").read_text(encoding="utf-8")) if (args.output_root / "index.json").exists() else []
    merged = {row["city_id"]: row for row in existing}
    merged.update({row["city_id"]: row for row in catalog})
    atomic_json(args.output_root / "index.json", [merged[city] for city in CITY_IDS if city in merged])
    atomic_json(args.output_root / "summary.json", {"dataset_id": "satellite_trajectory_v4_full", "pipeline_version": PIPELINE_VERSION, "seed": args.seed,
                "cities": len(merged), "total_trajectories": sum(row["complete"] for row in merged.values()),
                "total_images": sum(row["complete"] for row in merged.values()) * 8, "image_source": "Google satellite tiles (lyrs=s)"})


if __name__ == "__main__":
    main()
