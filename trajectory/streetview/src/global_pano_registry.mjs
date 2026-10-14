import fs from "node:fs";
import path from "node:path";

// Rebuild reservations from accepted navigation, so retries and separate
// extraction processes share the same dataset scope without a second database.
export function reservedPanos(moduleRoot, replacingRouteIds) {
  const replacing = new Set(replacingRouteIds);
  const configured = (process.env.STREETVIEW_NAVIGATION_INPUTS || "")
    .split(",").map((value) => value.trim()).filter(Boolean);
  const explicit = process.env.STREETVIEW_RESERVED_NAVIGATION_JSON;
  const defaults = [
    "llm_guided_navigation_100.json", "llm_guided_navigation.json",
    "llm_guided_alternative_navigation.json", "llm_guided_alternative_navigation_retry.json",
  ].map((name) => path.join(moduleRoot, "data", name));
  const files = explicit ? [explicit] : configured.length ? configured : defaults.filter((file) => fs.existsSync(file));
  const panos = new Set();
  for (const file of files) {
    const payload = JSON.parse(fs.readFileSync(path.resolve(file), "utf8"));
    if (!Array.isArray(payload.routes)) throw new Error(`Missing routes in ${file}`);
    for (const route of payload.routes) {
      if (replacing.has(route.id)) continue;
      if (route.frames?.length !== 8) throw new Error(`Incomplete accepted route ${route.id} in ${file}`);
      for (const frame of route.frames) {
        if (!frame.pano_id) throw new Error(`Missing pano ID in ${file}/${route.id}`);
        panos.add(frame.pano_id);
      }
    }
  }
  return [...panos];
}
