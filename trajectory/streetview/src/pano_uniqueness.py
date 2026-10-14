"""Dataset-wide panorama uniqueness checks, run before writing or rendering."""
from __future__ import annotations


def validate_global_pano_uniqueness(routes) -> None:
    owners = {}
    route_ids = set()
    for route in routes:
        route_id = route["id"]
        if route_id in route_ids:
            raise RuntimeError(f"Duplicate route ID: {route_id}")
        route_ids.add(route_id)
        for index, frame in enumerate(route.get("frames", [])):
            pano = frame.get("pano_id")
            if not pano:
                raise RuntimeError(f"Missing pano ID: {route_id}/frame {index}")
            if pano in owners:
                raise RuntimeError(
                    f"Global pano conflict {pano}: {owners[pano]} and {route_id}/frame {index}"
                )
            owners[pano] = f"{route_id}/frame {index}"
