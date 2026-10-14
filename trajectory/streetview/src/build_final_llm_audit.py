"""Keep exactly the OpenRouter GPT-5.6 Luna decision used by each final trajectory."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = Path(os.environ.get("STREETVIEW_FINAL_NAVIGATION_OUTPUT", ROOT / "data" / "llm_guided_navigation_100.json"))
AUDIT = Path(os.environ.get("STREETVIEW_LLM_AUDIT_OUTPUT", ROOT / "results" / "navigation" / "llm_guided_audit.json"))


def main() -> None:
    expected_provider = "OpenRouter"
    expected_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    navigation = json.loads(NAVIGATION.read_text(encoding="utf-8"))
    candidates: dict[tuple[str, str, str], dict] = {}
    for row in audit.get("routes", []):
        if row.get("selected_pano_id"):
            candidates[(row["id"], row["start_pano"], row["selected_pano_id"])] = {
                **row,
                "start_kind": "original",
            }

    configured = [value.strip() for value in os.environ.get("STREETVIEW_ALTERNATIVE_OVERRIDE_INPUTS", "").split(",") if value.strip()]
    for path in map(Path, configured):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for route_id, choices in payload.items():
            for row in choices if isinstance(choices, list) else [choices]:
                if not row or not row.get("next_pano_id"):
                    continue
                normalized = {
                    "id": route_id,
                    "name": next(route["name"] for route in navigation["routes"] if route["id"] == route_id),
                    "prompt": row["prompt"],
                    "start_pano": row["start_pano"],
                    "candidate_links": row["candidate_links"],
                    "candidate_image_sha256": row["candidate_image_sha256"],
                    "selected_index": row["selected_index"],
                    "selected_pano_id": row["next_pano_id"],
                    "selected_heading": row["selected_heading"],
                    "confidence": row["confidence"],
                    "reason": row["reason"],
                    "raw_response": row["raw_response"],
                    "model": row["model"],
                    "provider": row["provider"],
                    "candidate_image_source": row.get("candidate_image_source"),
                    "source_pano_heading": row.get("source_pano_heading"),
                    "source_pano_heading_source": row.get("source_pano_heading_source"),
                    "start_kind": "alternative",
                }
                candidates[(route_id, row["start_pano"], row["next_pano_id"])] = normalized

    final_rows = []
    for route in navigation["routes"]:
        override = route["initial_override"]
        key = (route["id"], override["start_pano"], override["next_pano_id"])
        if key not in candidates:
            raise RuntimeError(f"No matching model decision for final route {route['id']}: {key}")
        row = candidates[key]
        if row.get("provider") != expected_provider or row.get("model") != expected_model:
            raise RuntimeError(f"Wrong model provenance for {route['id']}")
        final_rows.append(row)

    if len(final_rows) != 100 or len({row["id"] for row in final_rows}) != 100:
        raise RuntimeError(f"Expected 100 unique final audit rows, got {len(final_rows)}")
    output = {
        "provider": expected_provider,
        "model": expected_model,
        "reasoning_effort": audit.get("reasoning_effort", "high"),
        "max_tokens": audit.get("max_tokens", 20000),
        "decision_count_per_start": 1,
        "prompt_template": audit["prompt_template"],
        "routes": final_rows,
    }
    AUDIT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote final model audit for {len(final_rows)} trajectories to {AUDIT}")


if __name__ == "__main__":
    main()
