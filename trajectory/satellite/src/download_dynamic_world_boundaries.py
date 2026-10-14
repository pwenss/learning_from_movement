"""Download annual Dynamic World label mode composites clipped to city boundaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import ee


DATASET = "GOOGLE/DYNAMICWORLD/V1"
USER_AGENT = "dynamic-world-boundary-download/1.0"
TILE_PIXELS = 4500
MAX_DIRECT_PIXELS = 20_000_000


def coordinate_pairs(value):
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            yield float(value[0]), float(value[1])
        else:
            for item in value:
                yield from coordinate_pairs(item)


def bounds(geometry):
    pairs = list(coordinate_pairs(geometry["coordinates"]))
    return (min(x for x, _ in pairs), min(y for _, y in pairs),
            max(x for x, _ in pairs), max(y for _, y in pairs))


def estimated_pixels(box, scale):
    west, south, east, north = box
    center_lat = (south + north) / 2.0
    meters_per_degree_lon = 111_320.0 * max(0.1, math.cos(math.radians(center_lat)))
    cols = max(1.0, (east - west) * meters_per_degree_lon / scale)
    rows = max(1.0, (north - south) * 111_320.0 / scale)
    return cols * rows, cols, rows


def grid(box, scale):
    _, cols, rows = estimated_pixels(box, scale)
    nx = max(1, math.ceil(cols / TILE_PIXELS))
    ny = max(1, math.ceil(rows / TILE_PIXELS))
    west, south, east, north = box
    for ix in range(nx):
        for iy in range(ny):
            x1 = west + (east - west) * ix / nx
            x2 = west + (east - west) * (ix + 1) / nx
            y1 = south + (north - south) * iy / ny
            y2 = south + (north - south) * (iy + 1) / ny
            yield ix, iy, {"type": "Polygon", "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]}


def download_zip(image, region, destination, scale, max_pixels):
    params = {
        "name": destination.stem,
        "scale": scale,
        "region": region,
        "format": "ZIPPED_GEO_TIFF",
        "maxPixels": max_pixels,
    }
    last_error = None
    for attempt in range(3):
        try:
            url = image.getDownloadURL(params)
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=900) as response:
                payload = response.read()
            if not payload.startswith(b"PK"):
                raise ValueError("Earth Engine response is not a ZIP archive")
            destination.write_bytes(payload)
            with ZipFile(destination) as archive:
                if not archive.namelist():
                    raise ValueError("downloaded ZIP contains no files")
            return {"bytes": len(payload), "attempts": attempt + 1}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(last_error or "download failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = f"{args.year:04d}-01-01"
    end = f"{args.year + 1:04d}-01-01"
    collection = ee.ImageCollection(DATASET).filterDate(start, end).select("label")
    rows = []

    for feature in boundaries["features"]:
        city = feature["properties"]["id"]
        geometry = feature["geometry"]
        city_ee = ee.Geometry(geometry)
        city_collection = collection.filterBounds(city_ee)
        scene_count = int(city_collection.size().getInfo())
        row = {"city_id": city, "year": args.year, "scale_m": args.scale,
               "scene_count": scene_count, "downloaded": False, "parts": []}
        city_dir = args.output_root / city
        city_dir.mkdir(parents=True, exist_ok=True)
        try:
            if scene_count == 0:
                raise RuntimeError("no Dynamic World scenes in requested year")
            image = city_collection.mode().rename("label").clip(city_ee)
            box = bounds(geometry)
            pixels, _, _ = estimated_pixels(box, args.scale)
            if pixels <= MAX_DIRECT_PIXELS:
                destination = city_dir / f"{city}_dynamic_world_{args.year}_mode_{args.scale}m.zip"
                result = download_zip(image, geometry, destination, args.scale, 100_000_000)
                row["parts"].append({"file": str(destination), **result})
                row["mode"] = "direct"
            else:
                row["mode"] = "tiled"
                for ix, iy, tile in grid(box, args.scale):
                    destination = city_dir / f"{city}_dynamic_world_{args.year}_mode_{args.scale}m_x{ix:03d}_y{iy:03d}.zip"
                    tile_image = image.clip(ee.Geometry(tile))
                    result = download_zip(tile_image, tile, destination, args.scale, 30_000_000)
                    row["parts"].append({"file": str(destination), "x": ix, "y": iy, **result})
            row["downloaded"] = True
            row["part_count"] = len(row["parts"])
            row["estimated_bbox_pixels"] = int(pixels)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    manifest = {"dataset": DATASET, "year": args.year, "scale_m": args.scale,
                "selection": "per-pixel annual mode", "boundaries": str(args.boundaries),
                "created_at": datetime.now(timezone.utc).isoformat(), "cities": rows,
                "summary": {"cities": len(rows),
                            "downloaded_cities": sum(row["downloaded"] for row in rows),
                            "failed_cities": sum(not row["downloaded"] for row in rows),
                            "files": sum(len(row["parts"]) for row in rows),
                            "bytes": sum(part.get("bytes", 0) for row in rows for part in row["parts"])}}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
