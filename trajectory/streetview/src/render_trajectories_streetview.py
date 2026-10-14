"""Render trajectory images with robolyst/streetview, without official image APIs."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from streetview_images import DEFAULT_ZOOM, TOOL_NAME, TOOL_VERSION, render_view
from pano_uniqueness import validate_global_pano_uniqueness


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("STREETVIEW_TRAJECTORY_ROOT", ROOT / "results" / "trajectory_sequences"))
OUTPUT_ROOT = Path(os.environ.get("STREETVIEW_OUTPUT_ROOT_180", ROOT / "results" / "trajectory_sequences_180"))
ROUTE_IDS = {value.strip() for value in os.environ.get("STREETVIEW_ROUTE_IDS", "").split(",") if value.strip()}
ZOOM = int(os.environ.get("STREETVIEW_TILE_ZOOM", str(DEFAULT_ZOOM)))
OUTPUT_WIDTH = int(os.environ.get("STREETVIEW_180_WIDTH", "1920"))
OUTPUT_HEIGHT = int(os.environ.get("STREETVIEW_180_HEIGHT", str(OUTPUT_WIDTH // 2)))
WORKERS = max(1, int(os.environ.get("STREETVIEW_RENDER_WORKERS", "4")))
SOURCE_NAVIGATION = os.environ.get("STREETVIEW_NAVIGATION_METADATA", "../../../data/llm_guided_navigation_100.json")
IMAGE_SOURCE = f"{TOOL_NAME} {TOOL_VERSION} unofficial panorama tiles"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def valid_jpeg(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.format == "JPEG" and image.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    except (OSError, ValueError):
        return False


def route_complete(route_id: str) -> dict | None:
    directory = OUTPUT_ROOT / route_id
    manifest_path = directory / "trajectory_180.json"
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    rendering = manifest.get("rendering", {})
    frames = manifest.get("frames", [])
    if rendering.get("image_source") != IMAGE_SOURCE or len(frames) != 8:
        return None
    if all(valid_jpeg(directory / frame["file"]) for frame in frames):
        return manifest
    return None


def render_route(route_directory: Path, route_id: str) -> dict:
    source = read_json(route_directory / "trajectory.json")
    frames = source.get("frames", [])
    if len(frames) != 8:
        raise RuntimeError(f"Expected 8 frames for {route_id}, got {len(frames)}")
    output_directory = OUTPUT_ROOT / route_id
    output_directory.mkdir(parents=True, exist_ok=True)
    rendered_frames = []
    for frame in frames:
        pano_id = frame["pano_id"]
        lat, lon = float(frame["lat"]), float(frame["lon"])
        center_heading = float(frame.get("render_heading", frame.get("heading", 0)))
        output_file = output_directory / frame["file"]
        # Old official-API images are deliberately overwritten unless a route
        # manifest proves that every frame already uses this image source.
        image, metadata = render_view(
            pano_id,
            lat,
            lon,
            center_heading,
            horizontal_fov=180,
            vertical_fov=90,
            output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
            zoom=ZOOM,
            source_heading=float(frame["source_pano_heading"]) if frame.get("source_pano_heading") is not None else None,
            source_date=frame.get("date"),
            source_world_size=frame.get("source_pano_world_size"),
            source_tile_size=frame.get("source_pano_tile_size"),
        )
        image.save(output_file, format="JPEG", quality=92, subsampling=0)
        image.close()
        rendered_frames.append(
            {
                **frame,
                "source_tile_zoom": ZOOM,
                "source_pano_heading": metadata["source_pano_heading"],
                "source_pano_world_size": metadata.get("source_pano_world_size"),
                "source_pano_tile_size": metadata.get("source_pano_tile_size"),
                "source_zoom_image_width": metadata["source_zoom_image_width"],
                "source_zoom_image_height": metadata["source_zoom_image_height"],
                "source_pano_date": metadata.get("date") or frame.get("date"),
                "source_tile_request_count": metadata["tile_request_count"],
                "image_source": metadata["image_source"],
                "projection": "equirectangular crop",
                "horizontal_fov_requested": 180,
                "vertical_fov": 90,
                "output_width": OUTPUT_WIDTH,
                "output_height": OUTPUT_HEIGHT,
                "center_heading": center_heading,
            }
        )
        print(f"{route_id}/{frame['file']}", flush=True)
    manifest = {
        "id": source["id"],
        "name": source["name"],
        "source": "Google Street View panorama tiles downloaded through robolyst/streetview",
        "rendering": {
            "image_source": IMAGE_SOURCE,
            "official_google_image_api_used": False,
            "projection": "equirectangular crop",
            "output_size": f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
            "horizontal_fov_requested": 180,
            "vertical_fov": 90,
            "tile_zoom": ZOOM,
            "center_heading": "trajectory render_heading; terminal frame repeats previous render heading",
            "source_navigation_metadata": SOURCE_NAVIGATION,
        },
        "frames": rendered_frames,
    }
    (output_directory / "trajectory_180.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"id": route_id, "name": source["name"], "frame_count": 8}


def main() -> None:
    if os.environ.get("STREETVIEW_OUTPUT_MODE", "180").lower() != "180":
        raise SystemExit("The robolyst/streetview renderer currently supports STREETVIEW_OUTPUT_MODE=180 only.")
    if ZOOM < 1 or ZOOM > 5:
        raise SystemExit("STREETVIEW_TILE_ZOOM must be an integer from 1 through 5.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    failed: list[dict] = []
    route_directories = sorted(path for path in INPUT_ROOT.iterdir() if path.is_dir())
    pending: list[tuple[Path, str]] = []
    effective_routes = {}
    for manifest_path in OUTPUT_ROOT.glob("*/trajectory_180.json"):
        manifest = read_json(manifest_path)
        effective_routes[manifest["id"]] = manifest
    for route_directory in route_directories:
        route_id = route_directory.name
        if ROUTE_IDS and route_id not in ROUTE_IDS:
            continue
        complete = route_complete(route_id)
        if complete:
            effective_routes[route_id] = complete
            index.append({"id": route_id, "name": complete["name"], "frame_count": 8})
            continue
        pending.append((route_directory, route_id))
        effective_routes[route_id] = read_json(route_directory / "trajectory.json")
    # Check the actual retained/rendered manifests before downloading any tile.
    validate_global_pano_uniqueness(effective_routes.values())
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(render_route, route_directory, route_id): route_id
            for route_directory, route_id in pending
        }
        for future in as_completed(futures):
            route_id = futures[future]
            try:
                index.append(future.result())
            except Exception as error:
                failed.append({"id": route_id, "error": str(error)})
                print(f"Skipping {route_id}: {error}", flush=True)
    index.sort(key=lambda row: row["id"])
    failed.sort(key=lambda row: row["id"])
    (OUTPUT_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "failed_routes.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(f"Generated {len(index)} 180-degree trajectories; skipped {len(failed)}.")


if __name__ == "__main__":
    main()
