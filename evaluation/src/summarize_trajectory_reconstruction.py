"""Build a normalized model-performance summary and a random baseline."""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = MODULE_ROOT / "results" / "streetview"
RANDOM_OUTPUT = RESULTS / "random_trajectory_reconstruction.json"
SUMMARY_OUTPUT = RESULTS / "trajectory_reconstruction_performance_summary.json"
RANDOM_SEED = 2026091501


def kendall_tau(prediction: list[int], truth: list[int]) -> float:
    truth_position = {value: index for index, value in enumerate(truth)}
    concordant = 0
    discordant = 0
    for first in range(len(prediction)):
        for second in range(first + 1, len(prediction)):
            if truth_position[prediction[first]] < truth_position[prediction[second]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / math.comb(len(prediction), 2)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    total = len(rows)
    exact = sum(bool(row["exact_match"]) for row in successful)
    correct_positions = sum(int(row["correct_positions"]) for row in successful)
    return {
        "total_sequences": total,
        "successful_sequences": len(successful),
        "failed_sequences": total - len(successful),
        "valid_output_rate": len(successful) / total,
        "mean_kendall_tau": sum(float(row["kendall_tau"]) for row in successful) / len(successful),
        "exact_match_count": exact,
        "exact_match_accuracy": exact / len(successful),
        "exact_match_accuracy_all_sequences": exact / total,
        "correct_positions": correct_positions,
        "total_positions": 8 * len(successful),
        "position_accuracy": correct_positions / (8 * len(successful)),
        "position_accuracy_all_sequences": correct_positions / (8 * total),
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def model_name(document: dict[str, Any], source: Path) -> str:
    description = document.get("model_description")
    model = description or document.get("model") or document.get("requested_model") or source.stem
    reasoning = document.get("reasoning_effort")
    thinking = document.get("thinking_enabled") or bool(document.get("thinking"))
    if reasoning or thinking:
        suffix = f"thinking {reasoning}" if reasoning else "thinking"
        return f"{model} ({suffix})"
    return str(model)


def normalized_metrics(document: dict[str, Any]) -> dict[str, Any]:
    metrics = document["metrics"]
    total = int(metrics["total_sequences"])
    successful = int(metrics["successful_sequences"])
    exact_count = int(metrics["exact_match_count"])
    correct_positions = int(metrics["correct_positions"])
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in document["sequences"] if row.get("status") == "ok"]
    thinking_tokens = []
    for row in document["sequences"]:
        if row.get("status") != "ok":
            continue
        usage_row = row.get("usage", {})
        details = usage_row.get("completion_tokens_details") or usage_row.get("output_tokens_details") or {}
        if details.get("reasoning_tokens") is not None:
            thinking_tokens.append(int(details["reasoning_tokens"]))
    usage = metrics.get("usage", {})
    thinking_configured = bool(document.get("reasoning_effort")) or bool(document.get("thinking_enabled")) or bool(document.get("thinking"))
    return {
        "total_sequences": total,
        "successful_sequences": successful,
        "failed_sequences": total - successful,
        "valid_output_rate": successful / total,
        "mean_kendall_tau": float(metrics["mean_kendall_tau"]),
        "exact_match_count": exact_count,
        "exact_match_accuracy_all_sequences": exact_count / total,
        "position_accuracy_all_sequences": correct_positions / (8 * total),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "thinking_tokens_reported_sequences": len(thinking_tokens),
        "total_thinking_tokens": sum(thinking_tokens),
        "mean_thinking_tokens_per_successful_sequence": (
            sum(thinking_tokens) / successful
            if thinking_tokens and len(thinking_tokens) == successful
            else (None if thinking_configured else 0.0)
        ),
        "mean_elapsed_seconds_per_successful_sequence": sum(elapsed) / len(elapsed) if elapsed else None,
    }


def main() -> None:
    sources = sorted(path for path in RESULTS.glob("*_trajectory_reconstruction.json") if path != RANDOM_OUTPUT)
    if not sources:
        raise RuntimeError("No model evaluation files found")
    documents = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sources]
    reference_path, reference = documents[0]
    reference_rows = {row["id"]: row for row in reference["sequences"]}
    reference_ids = sorted(reference_rows)
    prompt = " ".join(reference["prompt"].split())
    for path, document in documents:
        rows = {row["id"]: row for row in document["sequences"]}
        if sorted(rows) != reference_ids:
            raise RuntimeError(f"Sequence IDs differ in {path.name}")
        if document.get("shuffle_seed") != reference.get("shuffle_seed"):
            raise RuntimeError(f"Shuffle seed differs in {path.name}")
        if " ".join(document["prompt"].split()) != prompt:
            raise RuntimeError(f"Prompt differs in {path.name}")
        for sequence_id in reference_ids:
            if rows[sequence_id]["ground_truth_presented_indices"] != reference_rows[sequence_id]["ground_truth_presented_indices"]:
                raise RuntimeError(f"Ground truth differs in {path.name}/{sequence_id}")

    random_rows = []
    for sequence_id in reference_ids:
        source_row = reference_rows[sequence_id]
        digest = hashlib.sha256(f"{RANDOM_SEED}:{sequence_id}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        prediction = list(range(8))
        rng.shuffle(prediction)
        truth = list(source_row["ground_truth_presented_indices"])
        correct_positions = sum(predicted == expected for predicted, expected in zip(prediction, truth))
        random_rows.append({
            "id": sequence_id,
            "status": "ok",
            "shuffle_presented_to_original": source_row["shuffle_presented_to_original"],
            "ground_truth_presented_indices": truth,
            "prediction": prediction,
            "kendall_tau": kendall_tau(prediction, truth),
            "correct_positions": correct_positions,
            "position_accuracy": correct_positions / 8,
            "exact_match": prediction == truth,
        })
    random_document = {
        "evaluation": "visual spatial trajectory reconstruction",
        "model": "random permutation baseline",
        "strategy": "one independent uniform random permutation of indices 0 through 7 per sequence",
        "random_seed": RANDOM_SEED,
        "reference_file": reference_path.name,
        "input_directory": reference["input_directory"],
        "shuffle_seed": reference["shuffle_seed"],
        "prompt": None,
        "metric_definitions": reference["metric_definitions"],
        "theoretical_expectation": {
            "mean_kendall_tau": 0.0,
            "exact_match_accuracy": 1 / math.factorial(8),
            "position_accuracy": 1 / 8,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": aggregate(random_rows),
        "sequences": random_rows,
    }
    RANDOM_OUTPUT.write_text(json.dumps(random_document, ensure_ascii=False, indent=2), encoding="utf-8")

    models = []
    for path, document in documents:
        models.append({
            "name": model_name(document, path),
            "model": document.get("model") or document.get("requested_model"),
            "model_description": document.get("model_description"),
            "provider": document.get("provider"),
            "reasoning_effort": document.get("reasoning_effort"),
            "source_file": path.name,
            **normalized_metrics(document),
        })
    models.append({
        "name": "Random permutation",
        "model": "random",
        "model_description": None,
        "provider": None,
        "reasoning_effort": None,
        "source_file": RANDOM_OUTPUT.name,
        **normalized_metrics(random_document),
    })
    models.sort(key=lambda row: row["mean_kendall_tau"], reverse=True)
    for rank, row in enumerate(models, 1):
        row["rank_by_mean_kendall_tau"] = rank
    summary = {
        "evaluation": "visual spatial trajectory reconstruction",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "comparability": {
            "sequence_count": len(reference_ids),
            "identical_sequence_ids": True,
            "identical_prompt": True,
            "identical_shuffle_seed": True,
            "identical_ground_truth_permutations": True,
            "shuffle_seed": reference["shuffle_seed"],
            "input_directory": reference["input_directory"],
        },
        "primary_ranking_metric": "mean_kendall_tau over valid outputs",
        "invalid_output_policy": "Exact-match and position accuracy use all 100 requested sequences; invalid outputs count as incorrect.",
        "random_baseline": {
            "empirical_seed": RANDOM_SEED,
            "theoretical_mean_kendall_tau": 0.0,
            "theoretical_exact_match_accuracy": 1 / math.factorial(8),
            "theoretical_position_accuracy": 1 / 8,
        },
        "models": models,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"random": random_document["metrics"], "models": models}, ensure_ascii=False))


if __name__ == "__main__":
    main()
