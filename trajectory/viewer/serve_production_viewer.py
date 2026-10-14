"""Serve the production trajectory viewer without third-party dependencies."""
from __future__ import annotations

import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MODULE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = MODULE_ROOT.parents[1]
PROJECT_ROOT = CODE_ROOT.parent if CODE_ROOT.name == "code" else CODE_ROOT
DATASET_ID = os.environ.get("STREETVIEW_DATASET", "sv30_20260920_r100_r3")
DATASET_ROOT = Path(os.environ.get("STREETVIEW_DATASET_ROOT", PROJECT_ROOT / "datasets" / DATASET_ID)).resolve()
CONFIG_FILE = MODULE_ROOT / "viewer_config.json"
VIEWER_CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.is_file() else {}
SATELLITE_DATASET_ROOT = Path(os.environ.get("SATELLITE_DATASET_ROOT", VIEWER_CONFIG.get("satellite_dataset_root", PROJECT_ROOT / "datasets" / "satellite_trajectory_v1"))).resolve()
EVALUATION_ROOT = Path(os.environ.get("STREETVIEW_EVALUATION_ROOT", PROJECT_ROOT / "evaluation" / "results")).resolve()
VIEWER_FILE = MODULE_ROOT / "production_viewer.html"
HOST = os.environ.get("STREETVIEW_VIEWER_HOST", "127.0.0.1")
PORT = int(os.environ.get("STREETVIEW_VIEWER_PORT", "8787"))


CITY_LABELS = {
    "nyc_new_york": "New York", "sf_san_francisco": "San Francisco", "chicago": "Chicago",
    "philadelphia": "Philadelphia", "toronto": "Toronto", "vancouver": "Vancouver",
    "mexico_city": "Mexico City", "sao_paulo": "Sao Paulo", "rio_de_janeiro": "Rio de Janeiro",
    "buenos_aires": "Buenos Aires", "london": "London", "paris": "Paris", "berlin": "Berlin",
    "madrid": "Madrid", "rome": "Rome", "amsterdam": "Amsterdam", "zurich": "Zurich",
    "vienna": "Vienna", "stockholm": "Stockholm", "prague": "Prague", "tokyo": "Tokyo",
    "osaka": "Osaka", "seoul": "Seoul", "taipei": "Taipei", "singapore": "Singapore",
    "bangkok": "Bangkok", "kuala_lumpur": "Kuala Lumpur", "sydney": "Sydney",
    "melbourne": "Melbourne", "auckland": "Auckland",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog():
    catalog = []
    for row in read_json(DATASET_ROOT / "index.json"):
        city_directory = Path(row["path"]).parent
        summary_path = DATASET_ROOT / city_directory / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        city_id = row["city_id"]
        complete = int(row.get("complete") or summary.get("complete") or 0)
        catalog.append({
            "id": city_id,
            "label": CITY_LABELS.get(city_id, city_id.replace("_", " ")),
            "complete": complete,
            "imageCount": int(summary.get("image_count") or complete * 8),
            "directory": city_directory.as_posix(),
        })
    return catalog


if not DATASET_ROOT.exists():
    raise SystemExit(f"Dataset directory does not exist: {DATASET_ROOT}")

HTML = VIEWER_FILE.read_text(encoding="utf-8")
CITIES = load_catalog()
CITY_BY_ID = {city["id"]: city for city in CITIES}
ROUTE_INDEX_CACHE = {}
EVAL_RUN_CACHE = {}
FRAME_RE = re.compile(r"^frame_[0-7]{2}\.jpg$")
ROUTE_RE = re.compile(r"^[-a-z0-9_]+$", re.IGNORECASE)
EVAL_RE = re.compile(r"^eval_[a-z0-9_]+$", re.IGNORECASE)


def city_root(city_id: str):
    city = CITY_BY_ID.get(city_id)
    return DATASET_ROOT / city["directory"] / "trajectory_sequences_180" if city else None


def route_index(city_id: str):
    if city_id not in ROUTE_INDEX_CACHE:
        rows = read_json(city_root(city_id) / "index.json")
        ROUTE_INDEX_CACHE[city_id] = [{**row, "city_id": city_id} for row in rows]
    return ROUTE_INDEX_CACHE[city_id]


def route_manifest(city_id: str, route_id: str):
    route = next((row for row in route_index(city_id) if row["id"] == route_id), None)
    if not route or not isinstance(route.get("path"), str) or ".." in Path(route["path"]).parts:
        return None
    root = city_root(city_id).resolve()
    target = (root / route["path"]).resolve()
    if root not in target.parents or not target.is_file():
        return None
    return read_json(target)


def satellite_city_root(city_id: str):
    target = SATELLITE_DATASET_ROOT / "cities" / city_id / "trajectory_sequences"
    return target if target.is_dir() else None


def satellite_route_index(city_id: str):
    root = satellite_city_root(city_id)
    if root is None:
        return []
    path = root.parent / "index.json"
    if not path.is_file():
        return []
    rows = read_json(path)
    return [{**row, "city_id": city_id} for row in rows]


def satellite_manifest(city_id: str, route_id: str):
    root = satellite_city_root(city_id)
    if root is None or not ROUTE_RE.fullmatch(route_id):
        return None
    row = next((item for item in satellite_route_index(city_id) if item.get("id") == route_id), None)
    if not row or not isinstance(row.get("path"), str) or ".." in Path(row["path"]).parts:
        return None
    target = (SATELLITE_DATASET_ROOT / "cities" / city_id / row["path"]).resolve()
    base = (SATELLITE_DATASET_ROOT / "cities" / city_id).resolve()
    if base not in target.parents or not target.is_file():
        return None
    return read_json(target)


def evaluation_runs():
    if not EVALUATION_ROOT.is_dir():
        return []
    runs = []
    for directory in sorted(EVALUATION_ROOT.iterdir()):
        if not directory.is_dir() or not EVAL_RE.fullmatch(directory.name):
            continue
        result_path = directory / "results.json"
        manifest_path = directory / "sample_manifest.json"
        if not result_path.is_file() or not manifest_path.is_file():
            continue
        results = read_json(result_path)
        metrics = results.get("metrics") or {}
        runs.append({
            "id": directory.name,
            "label": results.get("model_requested") or results.get("model") or directory.name,
            "model": results.get("model_requested") or results.get("model") or "unknown",
            "provider": results.get("provider") or "",
            "updated_at": results.get("updated_at") or results.get("created_at") or "",
            "total": int(metrics.get("total") or len(results.get("sequences") or [])),
            "exact_match_accuracy": metrics.get("exact_match_accuracy"),
            "kendall_tau": metrics.get("kendall_tau_valid_answers"),
            "position_accuracy": metrics.get("position_accuracy_valid_answers"),
        })
    return runs


def evaluation_run(run_id: str):
    if not EVAL_RE.fullmatch(run_id):
        return None
    if run_id not in EVAL_RUN_CACHE:
        directory = EVALUATION_ROOT / run_id
        result_path, manifest_path = directory / "results.json", directory / "sample_manifest.json"
        if not result_path.is_file() or not manifest_path.is_file():
            return None
        results = read_json(result_path)
        manifest = read_json(manifest_path)
        manifest_by_id = {row.get("id"): row for row in manifest.get("sequences") or []}
        records = {}
        for row in results.get("sequences") or []:
            item = dict(row)
            item["images"] = (manifest_by_id.get(row.get("id")) or {}).get("images") or []
            records[row.get("id")] = item
        EVAL_RUN_CACHE[run_id] = {"results": results, "records": records}
    return EVAL_RUN_CACHE[run_id]


def evaluation_summary(run_id: str):
    run = evaluation_run(run_id)
    if not run:
        return None
    results = run["results"]
    metrics = results.get("metrics") or {}
    records = run["records"]
    return {
        "id": run_id,
        "label": results.get("model_requested") or results.get("model") or run_id,
        "model": results.get("model_requested") or results.get("model") or "unknown",
        "provider": results.get("provider") or "",
        "updated_at": results.get("updated_at") or results.get("created_at") or "",
        "metrics": metrics,
        "cities": results.get("per_city") or {},
        "routes": [{
            "id": row["id"], "city": row.get("city"),
            "valid_answer": row.get("valid_answer"), "exact_match": row.get("exact_match"),
            "kendall_tau": row.get("kendall_tau"), "position_accuracy": row.get("position_accuracy"),
        } for row in records.values()],
    }


def model_output(row):
    """Return model text from normalized fields or the provider response envelope."""
    content = row.get("content")
    reasoning = row.get("reasoning_content")
    model = row.get("model")
    for attempt in reversed(row.get("attempts") or []):
        result = (attempt.get("result") or {})
        response = (result.get("response") or {})
        body = (response.get("body") or {})
        choice = (body.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        content = content or message.get("content")
        reasoning = reasoning or message.get("reasoning_content")
        model = model or body.get("model")
        if content or reasoning or model:
            break
    return content, reasoning, model


def evaluation_record(run_id: str, route_id: str):
    summary = evaluation_summary(run_id)
    run = evaluation_run(run_id)
    if not summary or not run or not ROUTE_RE.fullmatch(route_id):
        return None
    row = run["records"].get(route_id)
    if not row:
        return None
    manifest = route_manifest(row.get("city"), route_id)
    if not manifest:
        return None
    images = []
    # Manifest images are in original trajectory order, as in the evaluator.
    for index, original_index in enumerate(row["shuffle_presented_to_original"]):
        image = row["images"][original_index]
        file_name = Path(image.get("path") or "").name
        if FRAME_RE.fullmatch(file_name):
            images.append({"presented_index": index, "original_index": int(original_index), "file": file_name})
    response = {key: row.get(key) for key in (
        "id", "city", "custom_id", "shuffle_presented_to_original", "ground_truth_presented_indices",
        "valid_answer", "cot_format_valid", "prediction", "exact_match", "kendall_tau",
        "pairwise_accuracy", "directed_adjacency_accuracy", "position_accuracy", "content",
        "reasoning_content", "finish_reason", "model",
    ) if key in row}
    content, reasoning, model = model_output(row)
    if content is not None:
        response["content"] = content
    if reasoning is not None:
        response["reasoning_content"] = reasoning
    if model is not None:
        response["model"] = model
    response["images"] = images
    response["frames"] = manifest.get("frames") or []
    response["model_label"] = summary["label"]
    return response


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "StreetViewViewer/1.0"

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)

    def send_bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        try:
            if not parts:
                return self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            if parts == ["index.html"]:
                return self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            if parts == ["api", "meta"]:
                summary_path = DATASET_ROOT / "summary.json"
                summary = read_json(summary_path) if summary_path.exists() else {}
                return self.send_json({
                    "dataset_id": DATASET_ID,
                    "dataset_complete": bool(summary.get("dataset_complete")),
                    "total_trajectories": sum(city["complete"] for city in CITIES),
                    "total_images": sum(city["imageCount"] for city in CITIES),
                    "cities": CITIES,
                })
            if parts == ["api", "boundaries"]:
                path = DATASET_ROOT / "city_boundaries.geojson"
                if not path.exists():
                    return self.send_json({"error": "Boundaries unavailable"}, HTTPStatus.NOT_FOUND)
                return self.send_json(read_json(path))
            if parts == ["api", "satellite", "meta"]:
                summary_path = SATELLITE_DATASET_ROOT / "summary.json"
                summary = read_json(summary_path) if summary_path.is_file() else {}
                satellite_cities = []
                for city in CITIES:
                    complete = len(satellite_route_index(city["id"]))
                    if complete:
                        satellite_cities.append({
                            "id": city["id"],
                            "label": city["label"],
                            "complete": complete,
                            "imageCount": complete * 8,
                            "directory": f"cities/{city['id']}",
                        })
                return self.send_json({
                    "available": SATELLITE_DATASET_ROOT.is_dir(),
                    "dataset_id": summary.get("dataset_id", SATELLITE_DATASET_ROOT.name),
                    "total_trajectories": int(summary.get("total_trajectories") or 0),
                    "cities": satellite_cities,
                })
            if len(parts) == 5 and parts[:3] == ["api", "satellite", "cities"] and parts[4] == "routes":
                city_id = parts[3]
                if city_id not in CITY_BY_ID:
                    return self.send_json({"error": "Unknown city"}, HTTPStatus.NOT_FOUND)
                return self.send_json(satellite_route_index(city_id))
            if len(parts) == 6 and parts[:3] == ["api", "satellite", "cities"] and parts[4] == "routes":
                city_id, route_id = parts[3], parts[5]
                manifest = satellite_manifest(city_id, route_id) if city_id in CITY_BY_ID else None
                if manifest is None:
                    return self.send_json({"error": "Unknown satellite trajectory"}, HTTPStatus.NOT_FOUND)
                return self.send_json(manifest)
            if parts == ["api", "evaluations"]:
                return self.send_json({"runs": evaluation_runs()})
            if len(parts) == 4 and parts[:2] == ["api", "evaluations"] and parts[3] == "routes":
                summary = evaluation_summary(parts[2])
                if not summary:
                    return self.send_json({"error": "Unknown evaluation"}, HTTPStatus.NOT_FOUND)
                return self.send_json(summary)
            if len(parts) == 5 and parts[:2] == ["api", "evaluations"] and parts[3] == "routes":
                record = evaluation_record(parts[2], parts[4])
                if not record:
                    return self.send_json({"error": "Unknown evaluated trajectory"}, HTTPStatus.NOT_FOUND)
                return self.send_json(record)
            if len(parts) == 4 and parts[:2] == ["api", "cities"] and parts[3] == "routes":
                city_id = parts[2]
                if city_id not in CITY_BY_ID:
                    return self.send_json({"error": "Unknown city"}, HTTPStatus.NOT_FOUND)
                return self.send_json(route_index(city_id))
            if len(parts) == 5 and parts[:2] == ["api", "cities"] and parts[3] == "routes":
                city_id, route_id = parts[2], parts[4]
                manifest = route_manifest(city_id, route_id) if city_id in CITY_BY_ID else None
                if manifest is None:
                    return self.send_json({"error": "Unknown trajectory"}, HTTPStatus.NOT_FOUND)
                return self.send_json(manifest)
            if len(parts) == 4 and parts[0] == "media" and FRAME_RE.fullmatch(parts[3]) and ROUTE_RE.fullmatch(parts[2]):
                city_id, route_id, file_name = parts[1], parts[2], parts[3]
                if city_id not in CITY_BY_ID or not any(row["id"] == route_id for row in route_index(city_id)):
                    return self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                image_path = (city_root(city_id) / route_id / file_name).resolve()
                if city_root(city_id).resolve() not in image_path.parents or not image_path.is_file():
                    return self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(image_path.name)[0] or "image/jpeg")
                self.send_header("Content-Length", str(image_path.stat().st_size))
                self.send_header("Cache-Control", "public, max-age=86400, immutable")
                self.end_headers()
                with image_path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.wfile.write(chunk)
                return
            if len(parts) == 4 and parts[0] == "satellite" and FRAME_RE.fullmatch(parts[3]) and ROUTE_RE.fullmatch(parts[2]):
                city_id, route_id, file_name = parts[1], parts[2], parts[3]
                manifest = satellite_manifest(city_id, route_id) if city_id in CITY_BY_ID else None
                if manifest is None:
                    return self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                base = (SATELLITE_DATASET_ROOT / "cities" / city_id / "trajectory_sequences" / route_id).resolve()
                image_path = (base / file_name).resolve()
                if base not in image_path.parents or not image_path.is_file():
                    return self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(image_path.stat().st_size))
                self.send_header("Cache-Control", "public, max-age=86400, immutable")
                self.end_headers()
                with image_path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.wfile.write(chunk)
                return
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, KeyError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    server = ThreadingHTTPServer((HOST, PORT), ViewerHandler)
    print(f"Production trajectory viewer: http://{HOST}:{PORT}/", flush=True)
    print(f"Dataset: {DATASET_ID} ({len(CITIES)} cities, {sum(city['complete'] for city in CITIES)} trajectories)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
