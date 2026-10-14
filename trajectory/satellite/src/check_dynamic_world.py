"""Check existing satellite frames against Dynamic World and download city composites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import ee

import generate_satellite_trajectories as pipeline

CLASS_NAMES = ["water", "trees", "grass", "flooded_vegetation", "crops",
               "shrub_and_scrub", "built", "bare", "snow_and_ice"]
NON_BUILT = [name for name in CLASS_NAMES if name != "built"]


def ee_geometry(geometry):
    return ee.Geometry(geometry.__geo_interface__)


def frame_polygon(manifest, frame):
    points = [pipeline.local_to_geo(manifest["anchor"]["latitude"], manifest["anchor"]["longitude"], x, y)
              for x, y in pipeline.local_footprint(frame).exterior.coords]
    return {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in points]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--image-year", type=int, required=True,
                        help="Calendar year used to build the per-pixel Dynamic World mode composite")
    parser.add_argument("--download-scale", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true",
                        help="Only screen frames; keep existing city downloads unchanged")
    args = parser.parse_args()
    ee.Initialize(project=args.project)
    year_start = f"{args.image_year:04d}-01-01"
    year_end = f"{args.image_year + 1:04d}-01-01"
    collection = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterDate(year_start, year_end).select("label"))
    boundaries = pipeline.load_boundaries(args.boundaries)
    args.download_root.mkdir(parents=True, exist_ok=True)
    manifests = sorted(args.dataset_root.glob("cities/*/trajectory_sequences/*/manifest.json"))
    frame_rows, city_rows, city_images, city_metadata = [], [], {}, {}
    manifests_by_city = {}
    for path in manifests:
        manifest = json.loads(path.read_text())
        manifests_by_city.setdefault(manifest["city_id"], []).append(manifest)
    for city, city_manifests in manifests_by_city.items():
        city_geometry = ee_geometry(boundaries[city])
        city_collection = collection.filterBounds(city_geometry)
        scene_count = int(city_collection.size().getInfo())
        city_metadata[city] = {"mode_year": args.image_year, "scene_count": scene_count}
        if scene_count == 0:
            continue
        # Use the most frequent Dynamic World label independently at every pixel
        # across the full calendar year. This removes single-scene cloud/mask gaps.
        city_image = city_collection.mode().rename("label")
        city_images[city] = city_image
        features = []
        for manifest in city_manifests:
            for frame in manifest["frames"]:
                features.append(ee.Feature(ee.Geometry(frame_polygon(manifest, frame)), {
                    "trajectory_id": manifest["trajectory_id"], "frame_id": frame["frame_id"]}))
        reduced = city_image.reduceRegions(
            collection=ee.FeatureCollection(features),
            reducer=ee.Reducer.frequencyHistogram(), scale=10,
            tileScale=4).getInfo()["features"]
        for feature in reduced:
            properties = feature.get("properties", {})
            histogram = (properties.get("histogram") or properties.get("label")
                         or properties.get("label_mode") or {})
            manifest_id = properties["trajectory_id"]
            path = next(path for path in manifests if json.loads(path.read_text())["trajectory_id"] == manifest_id)
            manifest = json.loads(path.read_text())
            frame = next(frame for frame in manifest["frames"] if frame["frame_id"] == properties["frame_id"])
            counts = {CLASS_NAMES[int(key)]: int(value) for key, value in histogram.items()}
            total = sum(counts.values())
            shares = {name: (counts.get(name, 0) / total if total else 0.0) for name in CLASS_NAMES}
            max_non_built = max((shares[name] for name in NON_BUILT), default=0.0)
            frame_rows.append({"city_id": city, "trajectory_id": manifest["trajectory_id"],
                               "frame_id": frame["frame_id"], "class_counts": counts,
                               "class_shares": shares, "max_non_built_share": max_non_built,
                               "valid_pixel_count": total, "has_data": total > 0,
                               "passes": bool(total and max_non_built >= 1.0 - 1e-12)})
    for city in pipeline.CITY_IDS:
        geometry = boundaries[city]
        city_file = args.download_root / f"{city}_dynamic_world_label_100m.zip"
        row = {"city_id": city, "file": str(city_file), "downloaded": False,
               **city_metadata.get(city, {"mode_year": args.image_year, "scene_count": 0})}
        if args.skip_download:
            row["download_skipped"] = True
            parts_manifest = city_file.with_suffix(".parts.json")
            row["downloaded"] = city_file.is_file() or parts_manifest.is_file()
            city_rows.append(row)
            continue
        try:
            city_image = city_images.get(city)
            if city_image is None:
                row["error"] = f"No Dynamic World scenes available in {args.image_year} for {city}"
                city_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                continue
            last_error = None
            for attempt in range(3):
                try:
                    # Refresh the signed URL on every retry; Earth Engine download
                    # URLs can expire or be invalidated after a transient failure.
                    url = city_image.clip(ee_geometry(geometry)).getDownloadURL({
                        "name": f"{city}_dynamic_world_label", "scale": args.download_scale,
                        "region": geometry.__geo_interface__, "format": "ZIPPED_GEO_TIFF",
                        "maxPixels": 10_000_000,
                    })
                    request = Request(url, headers={"User-Agent": "dynamic-world-trajectory-check/1.0"})
                    with urlopen(request, timeout=180) as response:
                        payload = response.read()
                    if not payload.startswith(b"PK"):
                        raise ValueError("download did not return a ZIP archive")
                    city_file.write_bytes(payload)
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            row.update({"downloaded": True, "bytes": city_file.stat().st_size})
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        city_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"dataset_id": "satellite_trajectory_v4_full", "dynamic_world_collection": "GOOGLE/DYNAMICWORLD/V1",
              "mode_year": args.image_year, "selection": "per-pixel annual mode",
              "city_metadata": city_metadata,
              "rule": "at least one non-built class share == 1.00",
              "frames": frame_rows, "cities": city_rows,
              "summary": {"frames": len(frame_rows), "passing_frames": sum(row["passes"] for row in frame_rows),
                          "failing_frames": sum(not row["passes"] for row in frame_rows),
                          "downloaded_city_composites": sum(row["downloaded"] for row in city_rows)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
