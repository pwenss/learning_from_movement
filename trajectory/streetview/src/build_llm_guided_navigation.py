"""Merge and validate the final LLM-guided navigation results."""
from __future__ import annotations

import json
import os
from pathlib import Path
from pano_uniqueness import validate_global_pano_uniqueness


ROOT = Path(__file__).resolve().parents[1]
ROUTES_FILE = Path(os.environ.get("STREETVIEW_START_LINKS_JSON", ROOT / "data" / "llm_guided_start_links.json"))
DEFAULT_INPUTS = [
    ROOT / "data" / "llm_guided_navigation.json",
    ROOT / "data" / "llm_guided_alternative_navigation.json",
    ROOT / "data" / "llm_guided_alternative_navigation_retry.json",
    ROOT / "data" / "llm_guided_marseille_navigation_extra.json",
]
OUTPUT = Path(os.environ.get("STREETVIEW_FINAL_NAVIGATION_OUTPUT", ROOT / "data" / "llm_guided_navigation_100.json"))
AUDIT_FILE = Path(os.environ.get("STREETVIEW_LLM_AUDIT_OUTPUT", ROOT / "results" / "navigation" / "llm_guided_audit.json"))
SUMMARY_FILE = Path(os.environ.get("STREETVIEW_FINAL_SUMMARY_OUTPUT", ROOT / "results" / "navigation" / "llm_guided_final_summary.json"))


def main() -> None:
    expected = json.loads(ROUTES_FILE.read_text(encoding="utf-8"))["routes"]
    expected_ids = [route["id"] for route in expected]
    expected_provider = "OpenRouter"
    expected_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
    merged: dict[str, dict] = {}
    sources: dict[str, str] = {}

    configured_inputs = [value.strip() for value in os.environ.get("STREETVIEW_NAVIGATION_INPUTS", "").split(",") if value.strip()]
    inputs = [Path(value) for value in configured_inputs] if configured_inputs else [path for path in DEFAULT_INPUTS if path.exists()]
    if not inputs and OUTPUT.exists():
        inputs = [OUTPUT]

    for source in inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for route in payload.get("routes", []):
            route_id = route["id"]
            if route_id in merged:
                raise RuntimeError(f"Duplicate successful route: {route_id}")
            if len(route.get("frames", [])) != 8:
                raise RuntimeError(f"Expected 8 frames for {route_id}")
            override = route.get("initial_override") or {}
            if override.get("provider") != expected_provider or override.get("model") != expected_model:
                raise RuntimeError(
                    f"Initial-link provenance mismatch for {route_id}: "
                    f"provider={override.get('provider')!r}, model={override.get('model')!r}"
                )
            merged[route_id] = route
            sources[route_id] = source.name

    missing = [route_id for route_id in expected_ids if route_id not in merged]
    unexpected = sorted(set(merged) - set(expected_ids))
    if missing or unexpected:
        raise RuntimeError(f"Navigation mismatch: missing={missing}, unexpected={unexpected}")

    output = {
        "target_route_count": len(expected_ids),
        "method": "OpenRouter GPT-5.6 Luna-guided initial navigation link from random in-city start positions",
        "routes": [merged[route_id] for route_id in expected_ids],
        "failed_routes": [],
        "route_sources": sources,
    }
    validate_global_pano_uniqueness(output["routes"])
    original_starts = {route["id"]: route["pano_id"] for route in expected}
    alternative_ids = [
        route_id
        for route_id in expected_ids
        if merged[route_id].get("start_pano") != original_starts[route_id]
    ]
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    if audit.get("provider") != expected_provider or audit.get("model") != expected_model:
        raise RuntimeError(
            "Refusing to build final navigation from a non-OpenRouter/GPT-5.6-Luna audit: "
            f"provider={audit.get('provider')!r}, model={audit.get('model')!r}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    final_root = Path(os.environ.get("STREETVIEW_OUTPUT_ROOT_180", ROOT / "results" / "trajectory_sequences_180_llm_guided_final"))
    if not final_root.exists():
        final_root = ROOT / "results" / "trajectory_sequences_180"
    summary = {
        "provider": audit.get("provider", expected_provider),
        "model": audit["model"],
        "decision_count_per_start": audit.get("decision_count_per_start", 1),
        "prompt_template": audit["prompt_template"].replace("{{", "{").replace("}}", "}"),
        "audited_trajectory_count": len(audit["routes"]),
        "original_start_position_count": len(expected_ids) - len(alternative_ids),
        "alternative_start_position_count": len(alternative_ids),
        "alternative_start_position_route_ids": alternative_ids,
        "final_trajectory_count": len(merged),
        "frames_per_trajectory": 8,
        "final_image_count": len(list(final_root.glob("*/*.jpg"))) if final_root.exists() else None,
        "constraint_failed_count": 0,
        "constraints": {
            "unique_panos": True,
            "global_unique_panos": True,
            "max_absolute_heading_turn_degrees": 90,
            "adjacent_geodesic_distance_m": "(1, 30)",
            "terminal_heading": "same as frame 6",
            "capture_date": "same YYYY-MM when available, otherwise same YYYY",
        },
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(output['routes'])} validated routes to {OUTPUT}")


if __name__ == "__main__":
    main()
