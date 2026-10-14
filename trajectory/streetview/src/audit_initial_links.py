"""Use OpenRouter GPT-5.6 Luna to select each route's first link."""
from __future__ import annotations
import base64, hashlib, http.client, io, json, os, random, shutil, ssl, tempfile, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from streetview_images import candidate_image_bytes

ROOT = Path(__file__).resolve().parents[1]
START_FILE = Path(os.environ.get("STREETVIEW_START_LINKS_JSON", ROOT / "data" / "llm_guided_start_links.json"))
OUT_FILE = Path(os.environ.get("STREETVIEW_LLM_AUDIT_OUTPUT", ROOT / "results" / "navigation" / "llm_guided_audit.json"))
OVERRIDE_FILE = Path(os.environ.get("STREETVIEW_INITIAL_LINK_OVERRIDES_OUTPUT", ROOT / "data" / "llm_guided_overrides.json"))
IMAGE_DIR = Path(os.environ.get("STREETVIEW_INITIAL_LINK_IMAGE_DIR", Path(tempfile.gettempdir()) / "streetview_initial_link_images"))
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
PROVIDER = "OpenRouter"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "20000"))
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
PROMPT_TEMPLATE = (
    "You are given multiple Street View images captured at the same location in {city}, {country}, each facing a different candidate heading. "
    "The images are labeled sequentially as 0, 1, and so on. Choose the candidate whose heading most likely matches the actual forward direction "
    "of the Street View survey vehicle at capture time.\n"
    'Return ONLY JSON: {{"selected_index": integer, "confidence": number between 0 and 1, "reason": "brief explanation"}}.'
)

def static_image(
    pano: str,
    heading: float,
    lat: float,
    lon: float,
    source_heading: float | None = None,
    source_date: str | None = None,
    source_world_size: dict | None = None,
    source_tile_size: dict | None = None,
) -> bytes:
    """Compatibility name: render locally; never call Street View Static API."""
    return candidate_image_bytes(
        pano,
        lat,
        lon,
        heading,
        source_heading=source_heading,
        source_date=source_date,
        source_world_size=source_world_size,
        source_tile_size=source_tile_size,
    )


def normalize_candidate_image(image: bytes) -> bytes:
    """Remove source JPEG metadata and produce a stable 640x640 RGB JPEG."""
    with Image.open(io.BytesIO(image)) as source:
        normalized = source.convert("RGB")
        if normalized.size != (640, 640):
            raise ValueError(f"Unexpected candidate image size: {normalized.size}")
        output = io.BytesIO()
        normalized.save(output, format="JPEG", quality=75, optimize=True, subsampling=2)
        return output.getvalue()

def route_location(route: dict) -> tuple[str, str]:
    name = str(route.get("name") or route.get("id") or "unknown")
    parts = [part.strip() for part in name.rsplit(",", 1)]
    return (parts[0], parts[1]) if len(parts) == 2 else (name, name)


def rendered_prompt(route: dict) -> str:
    city, country = route_location(route)
    return PROMPT_TEMPLATE.format(city=city, country=country)


def parse_json_response(text: str) -> dict:
    """Parse a JSON object even if the model wrapped it in a code fence."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")
    return result


def ask_llm(route: dict, candidates: list[dict], images: list[bytes]) -> dict:
    content = [{"type": "text", "text": rendered_prompt(route)}]
    for i, image in enumerate(images):
        content.append({"type": "text", "text": str(i)})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "reasoning": {"effort": "high"},
        "include_reasoning": True,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=900, context=ssl.create_default_context()) as response:
                raw = json.loads(response.read())
            message = raw["choices"][0]["message"]
            text = message.get("content")
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if part.get("type") == "text")
            if not isinstance(text, str):
                raise ValueError(f"OpenRouter returned no text content: {json.dumps(raw)[:2000]}")
            result = parse_json_response(text)
            index = int(result["selected_index"])
            if not 0 <= index < len(candidates): raise ValueError("selected_index out of range")
            confidence = float(result.get("confidence", 0))
            if not 0 <= confidence <= 1: raise ValueError("confidence out of range")
            return {
                "selected_index": index,
                "confidence": confidence,
                "reason": str(result.get("reason", "")),
                "raw_response": text.strip(),
            }
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRYABLE_STATUS or attempt == 3:
                raise RuntimeError(f"OpenRouter HTTP {error.code}: {response_body[:2000]}") from error
        except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.IncompleteRead, json.JSONDecodeError, KeyError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"OpenRouter request failed: {error}") from error
        time.sleep(min(30.0, 2.0 ** attempt) + random.random())
    raise RuntimeError("unreachable")

def main() -> None:
    if not OPENROUTER_KEY: raise SystemExit("Set OPENROUTER_API_KEY before running.")
    starts = {r["id"]: r for r in json.loads(START_FILE.read_text(encoding="utf-8")).get("routes", [])}
    route_ids = list(starts)
    requested = {v.strip() for v in os.environ.get("STREETVIEW_ROUTE_IDS", "").split(",") if v.strip()}
    if requested: route_ids = [r for r in route_ids if r in requested]
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    previous_rows = []
    if OUT_FILE.exists():
        try: previous_rows = json.loads(OUT_FILE.read_text(encoding="utf-8")).get("routes", [])
        except Exception: previous_rows = []
    records = {
        row["id"]: row
        for row in previous_rows
        if row.get("id") in starts
    }
    completed = {
        route_id
        for route_id, row in records.items()
        if row.get("status") != "error"
        and row.get("provider") == PROVIDER
        and row.get("model") == MODEL
        and row.get("start_pano") == starts[route_id].get("pano_id")
        and row.get("candidate_links") == starts[route_id].get("links")
    }

    def process_route(route_id: str) -> dict:
        route = starts.get(route_id, {})
        candidates = route.get("links", [])
        if not candidates: raise RuntimeError("No candidate links")
        images = []
        for i, candidate in enumerate(candidates):
            image_path = IMAGE_DIR / route_id / f"candidate_{i:02d}.jpg"; image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists(): image = image_path.read_bytes()
            else:
                image = static_image(
                    route["pano_id"],
                    float(candidate["heading"]),
                    float(route["lat"]),
                    float(route["lon"]),
                    float(route["source_pano_heading"]) if route.get("source_pano_heading") is not None else None,
                    route.get("date"),
                    route.get("source_pano_world_size"),
                    route.get("source_pano_tile_size"),
                )
                image_path.write_bytes(image)
            image = normalize_candidate_image(image)
            images.append(image)
        decision = ask_llm(route, candidates, images)
        selected = candidates[decision["selected_index"]]
        row = {
            "id": route_id,
            "name": route.get("name"),
            "prompt": rendered_prompt(route),
            "start_pano": route.get("pano_id"),
            "candidate_links": candidates,
            "candidate_image_sha256": [hashlib.sha256(image).hexdigest() for image in images],
            "selected_index": decision["selected_index"],
            "selected_pano_id": selected.get("pano_id"),
            "selected_heading": selected.get("heading"),
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "raw_response": decision["raw_response"],
            "model": MODEL,
            "provider": PROVIDER,
            "candidate_image_source": "robolyst/streetview 0.1.0 unofficial panorama tiles; locally rendered 90x90 degree view",
            "source_pano_heading": route.get("source_pano_heading"),
            "source_pano_heading_source": "Google Maps JavaScript StreetViewPanoramaData.tiles.centerHeading" if route.get("source_pano_heading") is not None else "robolyst/streetview search_panoramas exact pano match",
        }
        return row

    pending = [route_id for route_id in sorted(route_ids) if route_id not in completed]
    errors = []
    workers = max(1, int(os.environ.get("STREETVIEW_LLM_WORKERS", "4")))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True); OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_route, route_id): route_id for route_id in pending}
        for future in as_completed(futures):
            route_id = futures[future]
            try: records[route_id] = future.result()
            except Exception as error:
                records[route_id] = {"id": route_id, "status": "error", "error": str(error)}; errors.append(route_id)
            rows = [records[route_id] for route_id in sorted(records)]
            overrides = {
                row["id"]: {
                    "start_pano": row["start_pano"],
                    "next_pano_id": row["selected_pano_id"],
                    "provider": PROVIDER,
                    "model": MODEL,
                }
                for row in rows if row.get("selected_pano_id")
            }
            OUT_FILE.write_text(json.dumps({"provider": PROVIDER, "model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS, "decision_count_per_start": 1, "prompt_template": PROMPT_TEMPLATE, "routes": rows}, indent=2), encoding="utf-8")
            OVERRIDE_FILE.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
            status = "ERROR" if records[route_id].get("status") == "error" else "selected"
            selected_count = sum(1 for row in records.values() if row.get("selected_pano_id"))
            print(f"[{selected_count}/{len(starts)} selected] {route_id}: {status}", flush=True)
    rows = [records[route_id] for route_id in sorted(records)]
    overrides = {
        row["id"]: {
            "start_pano": row["start_pano"],
            "next_pano_id": row["selected_pano_id"],
            "provider": PROVIDER,
            "model": MODEL,
        }
        for row in rows if row.get("selected_pano_id")
    }
    OUT_FILE.write_text(json.dumps({"provider": PROVIDER, "model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS, "decision_count_per_start": 1, "prompt_template": PROMPT_TEMPLATE, "routes": rows}, indent=2), encoding="utf-8")
    OVERRIDE_FILE.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    if errors: raise RuntimeError(f"Audit failed for {len(errors)} routes: {', '.join(errors)}")
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    print(f"Selected initial links for {len(rows)} routes. Overrides: {OVERRIDE_FILE}")

if __name__ == "__main__": main()
