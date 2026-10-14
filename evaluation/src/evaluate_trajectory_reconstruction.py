"""Evaluate shuffled Street View trajectory reconstruction with GPT-5.6 Sol."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "streetview" / "results" / "trajectory_sequences_180"
DEFAULT_OUTPUT_FILE = MODULE_ROOT / "results" / "streetview" / "gpt_5_6_sol_trajectory_reconstruction.json"
CONFIG_FILE = Path.home() / ".codex" / "config.toml"

PROMPT = (
    "You are given 8 observation images sampled along a spatial trajectory. "
    "The images have been shuffled and are presented in an arbitrary order.  "
    "The input images are indexed from 0 to 7 according to their presented order. "
    "The camera's central viewing direction in each image is aligned with the direction "
    "of movement along the trajectory.  Reconstruct the spatial order in which these 8 "
    "observations would be encountered when following the trajectory. Use the spatial "
    "relationships across views, including viewpoint changes, scene geometry, and "
    "recurring visual landmarks.  Return only the 8 image indices in the reconstructed "
    "trajectory order. Each index from 0 to 7 must appear exactly once.  Use exactly the "
    "following format:  [0, 1, 2, 3, 4, 5, 6, 7] "
)

RESULT_RE = re.compile(r"\[\s*([0-7](?:\s*,\s*[0-7]){7})\s*\]")
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    reasoning_effort: str | None
    service_tier: str | None


def parse_simple_toml(path: Path) -> dict[str, Any]:
    """Read the small subset of TOML needed from the Codex configuration."""
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if raw_value.startswith('"') and raw_value.endswith('"'):
            value: Any = raw_value[1:-1]
        elif raw_value in {"true", "false"}:
            value = raw_value == "true"
        else:
            value = raw_value
        result[f"{section}.{key}" if section else key] = value
    return result


def load_settings(config_file: Path) -> Settings:
    config = parse_simple_toml(config_file)
    provider = str(config.get("model_provider", "openai"))
    prefix = f"model_providers.{provider}."
    return Settings(
        model=os.environ.get("OPENAI_VISION_MODEL", str(config.get("model", "gpt-5.6-sol"))),
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            str(config.get(prefix + "base_url", "https://api.openai.com/v1")),
        ).rstrip("/"),
        reasoning_effort=(
            os.environ.get("OPENAI_REASONING_EFFORT", str(config.get("model_reasoning_effort", "high"))).strip()
            or None
        ),
        service_tier=(
            os.environ.get("OPENAI_SERVICE_TIER", str(config.get("service_tier", ""))).strip()
            or None
        ),
    )


def stable_shuffle(sequence_id: str, seed: int) -> list[int]:
    # Each sequence receives an independent deterministic shuffle.
    rng = random.Random(f"{seed}:{sequence_id}")
    order = list(range(8))
    rng.shuffle(order)
    return order


def data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def extract_output_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)
            ).strip()
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts).strip()


def parse_prediction(text: str) -> list[int]:
    match = RESULT_RE.fullmatch(text.strip()) or RESULT_RE.search(text)
    if not match:
        raise ValueError(f"response does not contain a valid 8-index list: {text[:300]!r}")
    prediction = [int(value.strip()) for value in match.group(1).split(",")]
    if sorted(prediction) != list(range(8)):
        raise ValueError(f"response is not a permutation of 0..7: {prediction}")
    return prediction


def expected_order(shuffle: list[int]) -> list[int]:
    """Convert original chronological frame order to presented-image indices."""
    presented_index = {original_frame: index for index, original_frame in enumerate(shuffle)}
    return [presented_index[frame] for frame in range(8)]


def kendall_tau(prediction: list[int], expected: list[int]) -> float:
    position = {item: index for index, item in enumerate(expected)}
    concordant = 0
    discordant = 0
    for left in range(7):
        for right in range(left + 1, 8):
            if position[prediction[left]] < position[prediction[right]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / 28


def make_payload(
    settings: Settings,
    frames: list[Path],
    shuffle: list[int],
    detail: str,
    api_style: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    if api_style == "responses":
        content: list[dict[str, Any]] = [{"type": "input_text", "text": PROMPT}]
        for original_frame in shuffle:
            content.append(
                {"type": "input_image", "image_url": data_url(frames[original_frame]), "detail": detail}
            )
        payload: dict[str, Any] = {
            "model": settings.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        if settings.reasoning_effort:
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        if settings.service_tier:
            payload["service_tier"] = settings.service_tier
        return payload

    # Alibaba DashScope's OpenAI-compatible endpoint uses Chat Completions content types.
    content = [{"type": "text", "text": PROMPT}]
    for original_frame in shuffle:
        content.append({"type": "image_url", "image_url": {"url": data_url(frames[original_frame])}})
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "enable_thinking": bool(settings.reasoning_effort),
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort
    return payload


def call_responses_api(
    api_key: str,
    settings: Settings,
    payload: dict[str, Any],
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for attempt in range(max_attempts):
        request = urllib.request.Request(
            f"{settings.base_url}/responses",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRYABLE_STATUS or attempt + 1 == max_attempts:
                raise RuntimeError(f"HTTP {error.code}: {response_body[:1000]}") from error
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"request failed: {error}") from error
        time.sleep(min(60.0, 2.0**attempt) + random.random())
    raise RuntimeError("unreachable")


def call_chat_completions_api(
    api_key: str,
    settings: Settings,
    payload: dict[str, Any],
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for attempt in range(max_attempts):
        request = urllib.request.Request(
            f"{settings.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRYABLE_STATUS or attempt + 1 == max_attempts:
                raise RuntimeError(f"HTTP {error.code}: {response_body[:1000]}") from error
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"request failed: {error}") from error
        time.sleep(min(60.0, 2.0**attempt) + random.random())
    raise RuntimeError("unreachable")


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    total = len(rows)
    exact = sum(bool(row.get("exact_match")) for row in successful)
    correct_positions = sum(int(row.get("correct_positions", 0)) for row in successful)
    metrics: dict[str, Any] = {
        "total_sequences": total,
        "successful_sequences": len(successful),
        "failed_sequences": total - len(successful),
        "valid_output_rate": len(successful) / total if total else None,
        "mean_kendall_tau": (
            sum(float(row["kendall_tau"]) for row in successful) / len(successful) if successful else None
        ),
        "exact_match_count": exact,
        "exact_match_accuracy": exact / len(successful) if successful else None,
        "exact_match_accuracy_all_sequences": exact / total if total else None,
        "correct_positions": correct_positions,
        "total_positions": 8 * len(successful),
        "position_accuracy": correct_positions / (8 * len(successful)) if successful else None,
        "position_accuracy_all_sequences": correct_positions / (8 * total) if total else None,
    }
    metrics["usage"] = {
        "input_tokens": sum(
            int(row.get("usage", {}).get("input_tokens", row.get("usage", {}).get("prompt_tokens", 0)) or 0)
            for row in successful
        ),
        "output_tokens": sum(
            int(row.get("usage", {}).get("output_tokens", row.get("usage", {}).get("completion_tokens", 0)) or 0)
            for row in successful
        ),
        "total_tokens": sum(int(row.get("usage", {}).get("total_tokens", 0) or 0) for row in successful),
    }
    return metrics


def write_results(
    path: Path,
    settings: Settings,
    input_dir: Path,
    seed: int,
    detail: str,
    api_style: str,
    max_output_tokens: int,
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    document = {
        "evaluation": "visual spatial trajectory reconstruction",
        "model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "thinking_enabled": api_style == "chat-completions" and bool(settings.reasoning_effort),
        "service_tier": settings.service_tier,
        "image_detail": detail,
        "api_style": api_style,
        "max_output_tokens": max_output_tokens,
        "base_url": settings.base_url,
        "input_directory": str(input_dir.resolve()),
        "shuffle_seed": seed,
        "prompt": PROMPT,
        "metric_definitions": {
            "kendall_tau": "Kendall tau-a over all 28 unordered pairs; no ties are possible for permutations.",
            "exact_match_accuracy": "Fraction of successful sequences whose full predicted order equals ground truth.",
            "position_accuracy": "Fraction of positions in successful sequences containing the ground-truth image index.",
            "*_all_sequences": "Accuracy over all requested sequences, treating invalid model outputs as incorrect.",
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": aggregate(rows),
        "sequences": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_sequences(input_dir: Path) -> dict[str, list[Path]]:
    sequences: dict[str, list[Path]] = {}
    for directory in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        frames = [directory / f"frame_{index:02d}.jpg" for index in range(8)]
        missing = [path.name for path in frames if not path.is_file()]
        if missing:
            raise ValueError(f"{directory.name}: missing {', '.join(missing)}")
        sequences[directory.name] = frames
    if not sequences:
        raise ValueError(f"no trajectory directories found under {input_dir}")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--detail", choices=("low", "high", "original", "auto"), default="original")
    parser.add_argument("--api-style", choices=("responses", "chat-completions"), default="responses")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sequences (for testing).")
    parser.add_argument("--retry-errors", action="store_true", help="Retry rows previously saved with status=error.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    settings = load_settings(args.config)
    sequences = validate_sequences(args.input_dir)
    if args.limit:
        sequences = dict(list(sequences.items())[: args.limit])

    existing = load_existing(args.output)
    rows_by_id = {
        str(row["id"]): row
        for row in existing.get("sequences", [])
        if row.get("id") in sequences and (row.get("status") == "ok" or not args.retry_errors)
    }
    lock = threading.Lock()

    def evaluate(sequence_id: str, frames: list[Path]) -> dict[str, Any]:
        shuffle = stable_shuffle(sequence_id, args.seed)
        truth = expected_order(shuffle)
        started = time.monotonic()
        try:
            payload = make_payload(
                settings,
                frames,
                shuffle,
                args.detail,
                args.api_style,
                args.max_output_tokens,
            )
            if args.api_style == "responses":
                response = call_responses_api(api_key, settings, payload, args.timeout, args.max_attempts)
            else:
                response = call_chat_completions_api(api_key, settings, payload, args.timeout, args.max_attempts)
            raw_text = extract_output_text(response)
            prediction = parse_prediction(raw_text)
            correct_positions = sum(a == b for a, b in zip(prediction, truth))
            return {
                "id": sequence_id,
                "status": "ok",
                "shuffle_presented_to_original": shuffle,
                "ground_truth_presented_indices": truth,
                "prediction": prediction,
                "raw_output": raw_text,
                "kendall_tau": kendall_tau(prediction, truth),
                "correct_positions": correct_positions,
                "position_accuracy": correct_positions / 8,
                "exact_match": prediction == truth,
                "response_id": response.get("id"),
                "usage": response.get("usage", {}),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as error:
            return {
                "id": sequence_id,
                "status": "error",
                "shuffle_presented_to_original": shuffle,
                "ground_truth_presented_indices": truth,
                "error": str(error),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    pending = [(sequence_id, frames) for sequence_id, frames in sequences.items() if sequence_id not in rows_by_id]
    print(
        f"Model={settings.model} reasoning={settings.reasoning_effort} detail={args.detail}; "
        f"completed={len(rows_by_id)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(evaluate, sequence_id, frames): sequence_id for sequence_id, frames in pending}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                rows_by_id[row["id"]] = row
                write_results(
                    args.output,
                    settings,
                    args.input_dir,
                    args.seed,
                    args.detail,
                    args.api_style,
                    args.max_output_tokens,
                    rows_by_id,
                )
                completed = len(rows_by_id)
            if row["status"] == "ok":
                print(
                    f"[{completed}/{len(sequences)}] {row['id']}: tau={row['kendall_tau']:.4f}, "
                    f"positions={row['correct_positions']}/8, exact={row['exact_match']}",
                    flush=True,
                )
            else:
                stdout_encoding = sys.stdout.encoding or "utf-8"
                safe_error = str(row["error"]).encode(stdout_encoding, errors="backslashreplace").decode(stdout_encoding)
                print(f"[{completed}/{len(sequences)}] {row['id']}: ERROR {safe_error}", flush=True)

    write_results(
        args.output,
        settings,
        args.input_dir,
        args.seed,
        args.detail,
        args.api_style,
        args.max_output_tokens,
        rows_by_id,
    )
    metrics = aggregate([rows_by_id[key] for key in sorted(rows_by_id)])
    print(json.dumps(metrics, indent=2), flush=True)
    if metrics["failed_sequences"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
