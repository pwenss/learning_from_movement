"""Summarize adjacent-node distance and maximum heading turn distributions."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = ROOT / "results" / "trajectory_sequences_180"
OUTPUT = ROOT / "results" / "navigation" / "trajectory_metric_statistics.json"


def haversine_m(a: dict, b: dict) -> float:
    radius = 6_371_000.0
    lat1, lat2 = math.radians(float(a["lat"])), math.radians(float(b["lat"]))
    dlat = lat2 - lat1
    dlon = math.radians(float(b["lon"]) - float(a["lon"]))
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def describe(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def bins(values: list[float], edges: list[float]) -> list[dict]:
    result = []
    for low, high in zip(edges, edges[1:]):
        count = sum(low <= value < high for value in values)
        result.append({"lower_inclusive": low, "upper_exclusive": high, "count": count})
    result.append({"lower_inclusive": edges[-1], "upper_inclusive": None, "count": sum(value >= edges[-1] for value in values)})
    return result


def main() -> None:
    index = json.loads((TRAJECTORY_ROOT / "index.json").read_text(encoding="utf-8"))
    rows = []
    for entry in index:
        manifest = json.loads((TRAJECTORY_ROOT / entry["id"] / "trajectory_180.json").read_text(encoding="utf-8"))
        frames = manifest["frames"]
        distances = [haversine_m(a, b) for a, b in zip(frames, frames[1:])]
        turns = [float(frame.get("heading_turn_from_previous", 0.0)) for frame in frames[1:]]
        rows.append({
            "id": manifest["id"],
            "name": manifest["name"],
            "adjacent_distance_m": distances,
            "adjacent_distance_mean_m": statistics.mean(distances),
            "adjacent_distance_max_m": max(distances),
            "max_heading_turn_degrees": max(turns),
        })
    distance_means = [row["adjacent_distance_mean_m"] for row in rows]
    all_distances = [distance for row in rows for distance in row["adjacent_distance_m"]]
    max_turns = [row["max_heading_turn_degrees"] for row in rows]
    output = {
        "trajectory_count": len(rows),
        "adjacent_edge_count": len(all_distances),
        "distance_definition": "Haversine geodesic distance between consecutive saved frame coordinates; 7 segments per trajectory",
        "heading_definition": "maximum absolute circular heading turn across consecutive saved links; 7 turns per trajectory",
        "adjacent_distance_mean_m": describe(distance_means),
        "all_adjacent_distances_m": describe(all_distances),
        "all_adjacent_distance_bins_m": bins(all_distances, [0, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 30, 50]),
        "max_heading_turn_degrees": describe(max_turns),
        "distance_mean_bins_m": bins(distance_means, [0, 10, 20, 30, 40, 50, 75, 100, 150, 250]),
        "max_heading_turn_bins_degrees": bins(max_turns, [0, 10, 20, 30, 45, 60, 75, 90]),
        "per_trajectory": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "trajectory_count": len(rows),
        "distance_mean_m": output["adjacent_distance_mean_m"]["mean"],
        "max_heading_turn_mean_degrees": output["max_heading_turn_degrees"]["mean"],
        "output": str(OUTPUT),
    }))


if __name__ == "__main__":
    main()
