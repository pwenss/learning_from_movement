"""Validate navigation output and prepare Street View image manifests.

The navigation JSON is produced by maps_js_navigation.html. The initial link
comes from the OpenRouter GPT-5.6 Luna decision; later links retain the order returned
by StreetViewPanoramaData.links[] and are selected through constrained
backtracking.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from pano_uniqueness import validate_global_pano_uniqueness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("STREETVIEW_TRAJECTORY_ROOT", PROJECT_ROOT / "results" / "trajectory_sequences"))
NAVIGATION_JSON = Path(os.environ.get("STREETVIEW_NAVIGATION_JSON", PROJECT_ROOT / "data" / "llm_guided_navigation_100.json"))
REQUESTED_ROUTE_IDS = {value.strip() for value in os.environ.get("STREETVIEW_ROUTE_IDS", "").split(",") if value.strip()}
IMAGE_SIZE = "640x640"
REQUESTED_FOV = 180
RENDER_PITCH = 0


def bearing_degrees(current: dict[str, Any], target: dict[str, Any]) -> float:
    """Return the initial great-circle bearing from current to target."""
    lat1 = math.radians(float(current["lat"]))
    lat2 = math.radians(float(target["lat"]))
    delta_lon = math.radians(float(target["lon"]) - float(current["lon"]))
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def geodesic_distance_m(current: dict[str, Any], target: dict[str, Any]) -> float:
    radius = 6_371_000.0
    lat1, lat2 = math.radians(float(current["lat"])), math.radians(float(target["lat"]))
    dlat = lat2 - lat1
    dlon = math.radians(float(target["lon"]) - float(current["lon"]))
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def add_render_parameters(frames: list[dict[str, Any]]) -> None:
    """Add the view direction used for each downloaded frame.

    Every non-terminal frame points at the next saved frame by coordinate
    bearing. The terminal frame repeats the previous render direction.
    """
    previous_heading: float | None = None
    for index, frame in enumerate(frames):
        if index + 1 < len(frames):
            render_heading = bearing_degrees(frame, frames[index + 1])
            heading_source = "bearing(current pano position, next sequence pano position)"
        elif previous_heading is not None:
            render_heading = previous_heading
            heading_source = "repeated previous sequence render heading (terminal frame)"
        else:
            render_heading = float(frame["heading"])
            heading_source = "native links[].heading fallback (single-frame sequence)"
        frame["render_heading"] = round(render_heading, 6)
        frame["render_pitch"] = RENDER_PITCH
        frame["render_fov_requested"] = REQUESTED_FOV
        frame["render_heading_source"] = heading_source
        previous_heading = render_heading


def main() -> None:
    if not NAVIGATION_JSON.exists():
        raise SystemExit("Run the Maps JavaScript API navigation extractor first.")

    navigation = json.loads(NAVIGATION_JSON.read_text(encoding="utf-8"))
    if navigation.get("error"):
        raise RuntimeError(navigation["error"])
    # Validate the complete dataset even when only selected routes are emitted.
    routes = navigation.get("routes", [])
    replacing = {route["id"] for route in routes if not REQUESTED_ROUTE_IDS or route["id"] in REQUESTED_ROUTE_IDS}
    validate_global_pano_uniqueness(routes)
    effective_routes = {route["id"]: route for route in routes}
    for manifest_path in OUT.glob("*/trajectory.json"):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["id"] not in replacing:
            effective_routes[existing["id"]] = existing
    validate_global_pano_uniqueness(effective_routes.values())
    OUT.mkdir(exist_ok=True)
    index: list[dict[str, Any]] = []
    for route in navigation.get("routes", []):
        route_id = route["id"]
        if REQUESTED_ROUTE_IDS and route_id not in REQUESTED_ROUTE_IDS:
            continue
        frames = route.get("frames", [])
        if not frames:
            raise RuntimeError(f"No navigation frames returned for {route_id}")
        if len(frames) != 8:
            raise RuntimeError(f"Expected exactly 8 frames for {route_id}, got {len(frames)}")
        dates = [frame.get("date") for frame in frames]
        parsed_dates = []
        for value in dates:
            if not value:
                parsed_dates.append(None)
                continue
            match = value.split("-")
            if not match or len(match[0]) != 4:
                parsed_dates.append(None)
            else:
                parsed_dates.append((int(match[0]), int(match[1]) if len(match) > 1 and len(match[1]) == 2 else None))
        if any(value is None for value in parsed_dates):
            raise RuntimeError(f"Missing or malformed capture date in {route_id}; cannot verify date constraint")
        if all(month is not None for _, month in parsed_dates):
            if len(set(parsed_dates)) != 1:
                raise RuntimeError(f"Frames in {route_id} are not from the same month")
        elif len({year for year, _ in parsed_dates}) != 1:
            raise RuntimeError(f"Frames in {route_id} are not from the same year")
        if len({frame.get("pano_id") for frame in frames}) != len(frames):
            raise RuntimeError(f"Visited pano repeated in {route_id}")
        distances = [geodesic_distance_m(previous, current) for previous, current in zip(frames, frames[1:])]
        if any(distance <= 1 or distance >= 30 for distance in distances):
            raise RuntimeError(f"Adjacent pano distance outside open interval (1m, 30m) in {route_id}: {distances}")
        for frame, distance in zip(frames[1:], distances):
            frame["distance_from_previous_m"] = round(distance, 6)
        for previous, current in zip(frames, frames[1:]):
            turn = float(current.get("heading_turn_from_previous", 0))
            if turn > 90:
                raise RuntimeError(f"Absolute heading turn exceeds 90° in {route_id}/{current['frame']}: {turn}")
        add_render_parameters(frames)
        folder = OUT / route_id
        folder.mkdir(parents=True, exist_ok=True)
        for frame in frames[:-1]:
            if not frame.get("heading_source", "").endswith("links[].heading"):
                raise RuntimeError(f"Frame {route_id}/{frame['frame']} is missing links[].heading provenance")
        if frames[-1].get("heading_source") != "repeated previous saved-frame heading":
            raise RuntimeError(f"Terminal frame heading provenance is invalid in {route_id}")
        for current, next_frame in zip(frames, frames[1:]):
            if current.get("next_pano_id") != next_frame.get("pano_id"):
                raise RuntimeError(f"Navigation chain is discontinuous at {route_id}/{current['frame']}")
            if not current.get("next_in_sequence", True):
                raise RuntimeError(f"Non-terminal frame is marked terminal at {route_id}/{current['frame']}")
        if frames and frames[-1].get("next_in_sequence", False):
            raise RuntimeError(f"Terminal frame is missing next_in_sequence=false for {route_id}")
        if frames[-1].get("next_pano_id") is not None:
            raise RuntimeError(f"Terminal frame must not reference an outbound pano in {route_id}")
        if len(frames) > 1 and float(frames[-1]["heading"]) != float(frames[-2]["heading"]):
            raise RuntimeError(f"Terminal frame heading does not repeat frame 6 in {route_id}")
        if len(frames) > 1 and frames[-1]["render_heading"] != frames[-2]["render_heading"]:
            raise RuntimeError(f"Terminal render heading does not repeat previous frame in {route_id}")
        manifest = {
            "id": route_id,
            "name": route["name"],
            "source": "Maps JavaScript API StreetViewService links[].heading; image rendering deferred to robolyst/streetview",
            "rendering": {
                "size_requested": IMAGE_SIZE,
                "fov_requested": REQUESTED_FOV,
                "fov_effective_limit_observed": 120,
                "pitch": RENDER_PITCH,
                "heading": "render_heading",
                "heading_source": "bearing to next saved frame; terminal frame repeats frame 6 heading",
            },
            "frames": frames,
        }
        (folder / "trajectory.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        index.append({"id": route_id, "name": route["name"], "path": str(folder), "frame_count": len(frames)})
    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Generated {len(index)} trajectories in {OUT.resolve()}")


if __name__ == "__main__":
    main()
