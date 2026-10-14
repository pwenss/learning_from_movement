import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { addCityBoundaries } from "./city_boundaries.mjs";
const require = createRequire(import.meta.url);
const { chromium } = require("C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const key = process.env.GOOGLE_MAPS_API_KEY;
if (!key) throw new Error("Set GOOGLE_MAPS_API_KEY before running.");
const moduleRoot = fileURLToPath(new URL("..", import.meta.url));
const sourceRoot = path.join(moduleRoot, "src");
const ids = new Set((process.env.STREETVIEW_ROUTE_IDS || "").split(",").map((v) => v.trim()).filter(Boolean));
const maxRandomAttempts = Number.parseInt(process.env.STREETVIEW_RANDOM_START_ATTEMPTS || "200", 10);
const randomSeed = process.env.STREETVIEW_RANDOM_SEED || randomUUID();
if (!Number.isInteger(maxRandomAttempts) || maxRandomAttempts < 1) throw new Error("STREETVIEW_RANDOM_START_ATTEMPTS must be a positive integer.");
const seeds = JSON.parse(fs.readFileSync(path.join(moduleRoot, "data", "trajectory_routes.json"), "utf8"));
const selectedSeeds = ids.size ? seeds.filter((r) => ids.has(r.id)) : seeds;
const routes = await addCityBoundaries(selectedSeeds);
const server = http.createServer((request, response) => {
  if (request.url !== "/maps_js_start_links.html") { response.writeHead(404); response.end(); return; }
  response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  response.end(fs.readFileSync(path.join(sourceRoot, "maps_js_start_links.html"), "utf8")
    .replace("__TRAJECTORY_ROUTES_JSON__", JSON.stringify(routes).replace(/<\//g, "<\\/"))
    .replace("__MAX_RANDOM_ATTEMPTS__", String(maxRandomAttempts))
    .replace("__RANDOM_SEED_JSON__", JSON.stringify(randomSeed)));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" });
try {
  const page = await browser.newPage();
  page.on("console", (message) => console.log(message.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/maps_js_start_links.html#${encodeURIComponent(key)}`, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(() => {
    const text = document.querySelector("#output")?.textContent || "";
    if (!text || text === "loading") return false;
    try { return JSON.parse(text).complete === true; } catch { return false; }
  }, undefined, { timeout: 1800000 });
  const data = JSON.parse(await page.locator("#output").textContent());
  const output = path.resolve(process.env.STREETVIEW_START_LINKS_OUTPUT || path.join(moduleRoot, "data", "llm_guided_start_links.json"));
  fs.mkdirSync(path.dirname(output), { recursive: true }); fs.writeFileSync(output, JSON.stringify(data, null, 2));
  console.log(`Wrote ${data.routes.length} start-link records to ${output}`);
} finally { await browser.close(); server.close(); }
