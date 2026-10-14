"""Ask OpenRouter GPT-5.6 Luna to select links for alternative start panos."""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from audit_initial_links import MODEL, PROVIDER, ask_llm, normalize_candidate_image, rendered_prompt, static_image

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get("STREETVIEW_ALTERNATIVE_STARTS_JSON", ROOT / "data" / "llm_guided_alternative_start_candidates.json"))
OUTPUT = Path(os.environ.get("STREETVIEW_ALTERNATIVE_OVERRIDES_OUTPUT", ROOT / "data" / "llm_guided_alternative_overrides.json"))
TEMP = Path(tempfile.gettempdir()) / "streetview_alternative_start_images"
START_COUNT = int(os.environ.get("STREETVIEW_ALTERNATIVE_START_LIMIT", "1"))
START_OFFSET = int(os.environ.get("STREETVIEW_ALTERNATIVE_START_OFFSET", "0"))

def process(route: dict) -> tuple[str, list[dict]]:
    choices = []
    starts = route.get("starts", [])[START_OFFSET:START_OFFSET + START_COUNT]
    for start_index, start in enumerate(starts, start=START_OFFSET):
        images = []
        for link_index, link in enumerate(start["links"]):
            file = TEMP / route["id"] / f"start_{start_index:02d}_link_{link_index:02d}.jpg"
            file.parent.mkdir(parents=True, exist_ok=True)
            image = file.read_bytes() if file.exists() else static_image(
                start["pano_id"],
                float(link["heading"]),
                float(start["lat"]),
                float(start["lon"]),
                float(start["source_pano_heading"]) if start.get("source_pano_heading") is not None else None,
                start.get("date"),
                start.get("source_pano_world_size"),
                start.get("source_pano_tile_size"),
            )
            if not file.exists(): file.write_bytes(image)
            images.append(normalize_candidate_image(image))
        decision = ask_llm(route, start["links"], images)
        selected = start["links"][decision["selected_index"]]
        choices.append({
            "start_index": start_index,
            "start_pano": start["pano_id"],
            "lat": start["lat"],
            "lon": start["lon"],
            "start_date": start.get("date"),
            "prompt": rendered_prompt(route),
            "candidate_links": start["links"],
            "candidate_image_sha256": [hashlib.sha256(image).hexdigest() for image in images],
            "selected_index": decision["selected_index"],
            "next_pano_id": selected["pano_id"],
            "selected_heading": selected["heading"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "raw_response": decision["raw_response"],
            "model": MODEL,
            "provider": PROVIDER,
            "candidate_image_source": "robolyst/streetview 0.1.0 unofficial panorama tiles; locally rendered 90x90 degree view",
            "source_pano_heading": start.get("source_pano_heading"),
            "source_pano_heading_source": "Google Maps JavaScript StreetViewPanoramaData.tiles.centerHeading" if start.get("source_pano_heading") is not None else "robolyst/streetview search_panoramas exact pano match",
        })
    return route["id"], choices

def main() -> None:
    routes = json.loads(INPUT.read_text(encoding="utf-8"))["routes"]
    requested = {v.strip() for v in os.environ.get("STREETVIEW_ROUTE_IDS", "").split(",") if v.strip()}
    if requested: routes = [route for route in routes if route["id"] in requested]
    if OUTPUT.exists():
        try:
            results = json.loads(OUTPUT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            results = {}
    else:
        results = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("STREETVIEW_LLM_WORKERS", "4"))) as pool:
        futures = {pool.submit(process, route): route["id"] for route in routes}
        for n, future in enumerate(as_completed(futures), 1):
            route_id = futures[future]
            try:
                route_id, choices = future.result()
            except Exception as error:
                print(f"[{n}/{len(routes)}] {route_id}: ERROR {error}", flush=True)
                continue
            existing = {choice.get("start_index"): choice for choice in results.get(route_id, [])}
            existing.update({choice["start_index"]: choice for choice in choices})
            results[route_id] = [existing[index] for index in sorted(existing)]
            OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[{n}/{len(routes)}] {route_id}: {len(choices)} alternative starts", flush=True)
    shutil.rmtree(TEMP, ignore_errors=True)

if __name__ == "__main__": main()
