"""Evaluate the CoT prompt with Qwen3.8-Max (thinking disabled) via DashScope Batch."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from evaluate_trajectory_reconstruction import expected_order, kendall_tau, stable_shuffle, validate_sequences


MODULE_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = MODULE_ROOT.parent / "streetview" / "results" / "trajectory_sequences_180"
PROMPT_FILE = MODULE_ROOT / "prompt.md"
STATE_FILE = MODULE_ROOT / "results" / "streetview" / "qwen3_8_max_cot_no_thinking_batch_state.json"
OUTPUT_FILE = MODULE_ROOT / "results" / "streetview" / "qwen3_8_max_cot_no_thinking_batch_trajectory_reconstruction.json"
IMAGE_MAP_FILE = Path(tempfile.gettempdir()) / "qwen3_8_max_cot_batch_image_urls.json"
API_ROOT = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
MODEL = "qwen3.8-max"
SEED = 20260915
MAX_TOKENS = 8192
MAX_BATCH_LINE_BYTES = 5_900_000
LITTERBOX_API = "https://litterbox.catbox.moe/resources/internals/api.php"
ANSWER_RE = re.compile(r"<answer>\s*\[\s*([0-7](?:\s*,\s*[0-7]){7})\s*\]\s*</answer>", re.I | re.S)
LIST_RE = re.compile(r"\[\s*([0-7](?:\s*,\s*[0-7]){7})\s*\]")
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)


def cot_prompt() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    match = re.search(r"^## CoT Prompt\s*$\s*```text\s*\n(.*?)\n```", text, re.M | re.S)
    if not match:
        raise ValueError("CoT Prompt not found in prompt.md")
    return match.group(1).strip()


def image_token(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_id(sequence_id: str) -> str:
    return "r-" + hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:24]


def parse_prediction(text: str) -> list[int]:
    match = ANSWER_RE.search(text)
    if match is None:
        matches = list(LIST_RE.finditer(text))
        match = matches[-1] if matches else None
    if match is None:
        raise ValueError("No valid 8-index list")
    prediction = [int(value.strip()) for value in match.group(1).split(",")]
    if sorted(prediction) != list(range(8)):
        raise ValueError(f"Not a permutation: {prediction}")
    return prediction


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def upload_one_image(path: Path, attempts: int = 6) -> tuple[str, str]:
    token = image_token(path)
    opaque_name = f"{token}.jpg"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with path.open("rb") as handle:
                response = requests.post(
                    LITTERBOX_API,
                    data={"reqtype": "fileupload", "time": "24h"},
                    files={"fileToUpload": (opaque_name, handle, "image/jpeg")},
                    timeout=180,
                )
            response.raise_for_status()
            url = response.text.strip()
            if not re.fullmatch(r"https://litter\.catbox\.moe/[A-Za-z0-9._-]+", url):
                raise RuntimeError(f"Unexpected upload response: {url[:200]}")
            return token, url
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Unable to upload {path.name}: {last_error}")


def prepare_images(workers: int, limit: int = 0) -> None:
    sequences = validate_sequences(INPUT_DIR)
    if limit:
        sequences = dict(list(sequences.items())[:limit])
    unique: dict[str, Path] = {}
    for frames in sequences.values():
        for path in frames:
            unique.setdefault(image_token(path), path)
    existing: dict[str, str] = {}
    if IMAGE_MAP_FILE.exists():
        existing = json.loads(IMAGE_MAP_FILE.read_text(encoding="utf-8"))
    pending = [path for token, path in unique.items() if token not in existing]
    print(f"Images: {len(unique)} total, {len(pending)} to upload", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(upload_one_image, path): path for path in pending}
        completed = 0
        for future in as_completed(futures):
            token, url = future.result()
            existing[token] = url
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                write_json(IMAGE_MAP_FILE, existing)
                print(f"Uploaded {completed}/{len(pending)}", flush=True)
    write_json(IMAGE_MAP_FILE, existing)


def build_jsonl_line(
    sequence_id: str,
    frames: list[Path],
    image_urls: dict[str, str] | None,
    prompt: str,
    jpeg_quality: int | None = None,
) -> dict[str, Any]:
    shuffle = stable_shuffle(sequence_id, SEED)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for original_index in shuffle:
        path = frames[original_index]
        if image_urls is None:
            if jpeg_quality is None:
                mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                image_bytes = path.read_bytes()
            else:
                mime_type = "image/jpeg"
                with Image.open(path) as image:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                    image_bytes = buffer.getvalue()
            image_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        else:
            image_url = image_urls[image_token(path)]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )
    return {
        "custom_id": request_id(sequence_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
    }


def encoded_line_size(line: dict[str, Any]) -> int:
    return len(json.dumps(line, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def build_sized_base64_line(sequence_id: str, frames: list[Path], prompt: str) -> tuple[dict[str, Any], int]:
    """Build a Base64 request safely below DashScope's 6 MiB per-line limit."""
    line = build_jsonl_line(sequence_id, frames, None, prompt)
    if encoded_line_size(line) <= MAX_BATCH_LINE_BYTES:
        return line, 100
    for quality in (88, 82, 76, 70, 64, 58, 52, 46, 40):
        line = build_jsonl_line(sequence_id, frames, None, prompt, jpeg_quality=quality)
        if encoded_line_size(line) <= MAX_BATCH_LINE_BYTES:
            return line, quality
    raise RuntimeError(f"Unable to fit {sequence_id} below {MAX_BATCH_LINE_BYTES} bytes")


def upload_batch_file(api_key: str, lines: list[dict[str, Any]]) -> dict[str, Any]:
    payload = "\n".join(json.dumps(line, ensure_ascii=False, separators=(",", ":")) for line in lines) + "\n"
    response = requests.post(
        f"{API_ROOT}/files",
        headers=auth_headers(api_key),
        data={"purpose": "batch"},
        files={"file": ("qwen3_8_max_cot_batch.jsonl", payload.encode("utf-8"), "application/jsonl")},
        timeout=300,
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(f"File upload HTTP {response.status_code}: {response.text[:3000]}")
    return response.json()


def create_batch(api_key: str, input_file_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_ROOT}/batches",
        headers={**auth_headers(api_key), "Content-Type": "application/json"},
        json={
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"evaluation": "trajectory-cot-no-thinking"},
        },
        timeout=180,
    )
    if response.status_code not in {200, 201, 202}:
        raise RuntimeError(f"Batch creation HTTP {response.status_code}: {response.text[:3000]}")
    return response.json()


def submit(api_key: str, limit: int, image_mode: str, batch_size: int) -> None:
    sequences = validate_sequences(INPUT_DIR)
    if limit:
        sequences = dict(list(sequences.items())[:limit])
    image_urls: dict[str, str] | None = None
    if image_mode == "public-url":
        if not IMAGE_MAP_FILE.exists():
            raise RuntimeError(f"Image URL map is missing: run prepare-images first ({IMAGE_MAP_FILE})")
        image_urls = json.loads(IMAGE_MAP_FILE.read_text(encoding="utf-8"))
        required = {image_token(path) for frames in sequences.values() for path in frames}
        missing = required - set(image_urls)
        if missing:
            raise RuntimeError(f"Image URL map is missing {len(missing)} images")
    prompt = cot_prompt()
    lines = [build_jsonl_line(sequence_id, frames, image_urls, prompt) for sequence_id, frames in sequences.items()]
    state = {
        "evaluation": "Qwen3.8-Max CoT Prompt, thinking disabled",
        "provider": "Alibaba Cloud DashScope",
        "model": MODEL,
        "thinking_enabled": False,
        "shuffle_seed": SEED,
        "image_transport": "base64 data URLs" if image_urls is None else "24-hour temporary public URLs with opaque names",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "request_mapping": {request_id(sequence_id): sequence_id for sequence_id in sequences},
        "batches": [],
    }
    for offset in range(0, len(lines), batch_size):
        chunk = lines[offset : offset + batch_size]
        uploaded = upload_batch_file(api_key, chunk)
        batch = create_batch(api_key, uploaded["id"])
        state["batches"].append(
            {
                "id": batch["id"],
                "input_file_id": uploaded["id"],
                "status": batch.get("status"),
                "offset": offset,
                "count": len(chunk),
            }
        )
        write_json(STATE_FILE, state)
        print(f"Submitted {batch['id']}: {offset + 1}-{min(offset + batch_size, len(lines))}", flush=True)


def retry_failed(api_key: str) -> None:
    """Replace only validation-failed batches, compressing lines that exceeded 6 MiB."""
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    sequences = validate_sequences(INPUT_DIR)
    sequence_items = list(sequences.items())
    prompt = cot_prompt()
    original_batches = [item for item in state["batches"] if not item.get("retry_of")]
    offset = 0
    retry_specs: list[tuple[dict[str, Any], int, int]] = []
    for item in original_batches:
        count = int(item["count"])
        item.setdefault("offset", offset)
        if item.get("status") == "failed" and not item.get("superseded"):
            retry_specs.append((item, int(item["offset"]), count))
        offset += count
    if not retry_specs:
        print("No failed batches require retry", flush=True)
        return
    for original, start, count in retry_specs:
        chunk: list[dict[str, Any]] = []
        qualities: dict[str, int] = {}
        for sequence_id, frames in sequence_items[start : start + count]:
            line, quality = build_sized_base64_line(sequence_id, frames, prompt)
            chunk.append(line)
            qualities[sequence_id] = quality
        uploaded = upload_batch_file(api_key, chunk)
        batch = create_batch(api_key, uploaded["id"])
        original["superseded"] = True
        state["batches"].append(
            {
                "id": batch["id"],
                "input_file_id": uploaded["id"],
                "status": batch.get("status"),
                "offset": start,
                "count": count,
                "retry_of": original["id"],
                "jpeg_qualities": qualities,
            }
        )
        write_json(STATE_FILE, state)
        print(f"Retried {original['id']} as {batch['id']}: {start + 1}-{start + count}", flush=True)


def get_batch(api_key: str, batch_id: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(f"{API_ROOT}/batches/{batch_id}", headers=auth_headers(api_key), timeout=120)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:3000]}")
            return response.json()
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt + 1 < 6:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Unable to read batch {batch_id}: {last_error}")


def get_file_content(api_key: str, file_id: str) -> str:
    response = requests.get(f"{API_ROOT}/files/{file_id}/content", headers=auth_headers(api_key), timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"File download HTTP {response.status_code}: {response.text[:3000]}")
    return response.text


def check(api_key: str) -> None:
    """Refresh Batch status once; collect immediately when every active batch is terminal."""
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    terminal = {"completed", "failed", "expired", "cancelled", "canceled"}
    active = [item for item in state["batches"] if not item.get("superseded")]
    for item in active:
        batch = get_batch(api_key, item["id"])
        item["status"] = batch.get("status")
        item["request_counts"] = batch.get("request_counts")
        if batch.get("status") in terminal:
            item["result"] = batch
    write_json(STATE_FILE, state)
    counts: dict[str, int] = {}
    for item in active:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    print(json.dumps({"batch_statuses": counts}, ensure_ascii=False), flush=True)
    if all(item.get("status") in terminal for item in active):
        collect(api_key, 0)


def sum_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    totals: dict[str, float] = {}
    for target, keys in aliases.items():
        totals[target] = sum(
            float(next((row.get("usage", {}).get(key) for key in keys if row.get("usage", {}).get(key) is not None), 0) or 0)
            for row in rows
        )
    return totals


def collect(api_key: str, poll_seconds: int) -> None:
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    terminal = {"completed", "failed", "expired", "cancelled", "canceled"}
    while True:
        pending = 0
        for item in state["batches"]:
            if item.get("superseded"):
                continue
            if item.get("status") in terminal and "result" in item:
                continue
            batch = get_batch(api_key, item["id"])
            item["status"] = batch.get("status")
            item["request_counts"] = batch.get("request_counts")
            if batch.get("status") in terminal:
                item["result"] = batch
            else:
                pending += 1
        write_json(STATE_FILE, state)
        counts: dict[str, int] = {}
        for item in state["batches"]:
            if item.get("superseded"):
                continue
            counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
        print(f"Statuses: {counts}", flush=True)
        if pending == 0:
            break
        time.sleep(poll_seconds)
    incomplete = [
        item
        for item in state["batches"]
        if not item.get("superseded")
        and (item.get("status") != "completed" or not item.get("result", {}).get("output_file_id"))
    ]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} batches did not complete successfully: {[(x['id'], x.get('status')) for x in incomplete]}")
    output_lines: list[dict[str, Any]] = []
    error_lines: list[dict[str, Any]] = []
    result_batch_by_custom_id: dict[str, str] = {}
    for item in state["batches"]:
        if item.get("superseded"):
            continue
        batch = item["result"]
        batch_output = [json.loads(line) for line in get_file_content(api_key, batch["output_file_id"]).splitlines() if line.strip()]
        output_lines.extend(batch_output)
        result_batch_by_custom_id.update({line.get("custom_id"): item["id"] for line in batch_output})
        if batch.get("error_file_id"):
            batch_errors = [json.loads(line) for line in get_file_content(api_key, batch["error_file_id"]).splitlines() if line.strip()]
            error_lines.extend(batch_errors)
            result_batch_by_custom_id.update({line.get("custom_id"): item["id"] for line in batch_errors})
    mapping = state["request_mapping"]
    rows: list[dict[str, Any]] = []
    for result in output_lines:
        sequence_id = mapping.get(result.get("custom_id"), result.get("custom_id"))
        shuffle = stable_shuffle(sequence_id, SEED)
        truth = expected_order(shuffle)
        body = ((result.get("response") or {}).get("body") or {})
        try:
            message = body["choices"][0]["message"]
            raw_output = message.get("content") or ""
            prediction = parse_prediction(raw_output)
            correct = sum(left == right for left, right in zip(prediction, truth))
            think_match = THINK_RE.search(raw_output)
            thinking_text = think_match.group(1).strip() if think_match else None
            rows.append(
                {
                    "id": sequence_id,
                    "status": "ok",
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "output_thinking": thinking_text,
                    "reasoning_content": message.get("reasoning_content"),
                    "ground_truth_presented_indices": truth,
                    "kendall_tau": kendall_tau(prediction, truth),
                    "correct_positions": correct,
                    "position_accuracy": correct / 8,
                    "exact_match": prediction == truth,
                    "usage": body.get("usage", {}),
                    "batch_id": result_batch_by_custom_id.get(result.get("custom_id")),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "id": sequence_id,
                    "status": "error",
                    "error": str(error),
                    "raw_response": body,
                    "ground_truth_presented_indices": truth,
                    "batch_id": result_batch_by_custom_id.get(result.get("custom_id")),
                }
            )
    for result in error_lines:
        sequence_id = mapping.get(result.get("custom_id"), result.get("custom_id"))
        if not any(row["id"] == sequence_id for row in rows):
            rows.append({"id": sequence_id, "status": "request_error", "error": result.get("error"), "batch_id": result_batch_by_custom_id.get(result.get("custom_id"))})
    rows.sort(key=lambda row: row["id"])
    successful = [row for row in rows if row["status"] == "ok"]
    exact = sum(row["exact_match"] for row in successful)
    correct = sum(row["correct_positions"] for row in successful)
    submitted = datetime.fromisoformat(state["submitted_at"])
    completed_at = datetime.now(timezone.utc)
    metrics = {
        "total_sequences": len(mapping),
        "returned_sequences": len(rows),
        "successful_sequences": len(successful),
        "failed_sequences": len(mapping) - len(successful),
        "valid_output_rate": len(successful) / len(mapping),
        "mean_kendall_tau": sum(row["kendall_tau"] for row in successful) / len(successful) if successful else None,
        "exact_match_count": exact,
        "exact_match_accuracy_all_sequences": exact / len(mapping),
        "correct_positions": correct,
        "total_positions": 8 * len(mapping),
        "position_accuracy_all_sequences": correct / (8 * len(mapping)),
        "usage": sum_usage(successful),
        "total_time_seconds_submission_to_collection": (completed_at - submitted).total_seconds(),
    }
    write_json(
        OUTPUT_FILE,
        {
            "evaluation": "visual spatial trajectory reconstruction CoT Prompt",
            "submission_type": "dashscope_openai_batch_base64_image_data_urls",
            "provider": "Alibaba Cloud DashScope",
            "model_requested": MODEL,
            "thinking_enabled": False,
            "shuffle_seed": SEED,
            "prompt": cot_prompt(),
            "metrics": metrics,
            "batches": [
                {
                    **{key: item.get("result", {}).get(key, item.get(key)) for key in ("id", "status", "request_counts", "created_at", "completed_at")},
                    "retry_of": item.get("retry_of"),
                }
                for item in state["batches"]
                if not item.get("superseded")
            ],
            "sequences": rows,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if IMAGE_MAP_FILE.exists():
        IMAGE_MAP_FILE.unlink()
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-images", "submit", "retry-failed", "check", "collect", "run"))
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image-mode", choices=("public-url", "base64"), default="public-url")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.action in {"prepare-images", "run"} and args.image_mode == "public-url":
        prepare_images(args.upload_workers, args.limit)
    if args.action in {"submit", "retry-failed", "check", "collect", "run"}:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit("DASHSCOPE_API_KEY is not set")
        if args.action in {"submit", "run"}:
            submit(api_key, args.limit, args.image_mode, args.batch_size)
        if args.action == "retry-failed":
            retry_failed(api_key)
        if args.action == "check":
            check(api_key)
        if args.action in {"collect", "run"}:
            collect(api_key, args.poll_seconds)


if __name__ == "__main__":
    main()
