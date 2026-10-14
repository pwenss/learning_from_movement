"""Summarize per-trajectory single-decision LLM outcomes."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "navigation" / "llm_guided_audit.json"
OUTPUT = ROOT / "results" / "navigation" / "llm_guided_decision_statistics.json"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = []
    for route in audit["routes"]:
        rows.append(
            {
                "id": route["id"],
                "name": route["name"],
                "selected_index": int(route["selected_index"]),
                "confidence": float(route["confidence"]),
            }
        )
    confidence = [row["confidence"] for row in rows]
    summary = {
        "provider": audit.get("provider"),
        "model": audit["model"],
        "prompt_template": audit["prompt_template"].replace("{{", "{").replace("}}", "}"),
        "trajectory_count": len(rows),
        "decision_count_per_start": 1,
        "mean_confidence": round(sum(confidence) / len(confidence), 4),
        "min_confidence": min(confidence),
        "max_confidence": max(confidence),
        "per_trajectory": rows,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote decision statistics for {len(rows)} trajectories to {OUTPUT}")


if __name__ == "__main__":
    main()
