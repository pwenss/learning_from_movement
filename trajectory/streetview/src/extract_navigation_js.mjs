import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { reservedPanos } from "./global_pano_registry.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const key = process.env.GOOGLE_MAPS_API_KEY;
if (!key) throw new Error("Set GOOGLE_MAPS_API_KEY before running.");

const moduleRoot = fileURLToPath(new URL("..", import.meta.url));
const sourceRoot = path.join(moduleRoot, "src");
const requestedRouteIds = (process.env.STREETVIEW_ROUTE_IDS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const allRouteSeeds = JSON.parse(fs.readFileSync(path.join(moduleRoot, "data", "trajectory_routes.json"), "utf8"));
const routeSeeds = requestedRouteIds.length
  ? allRouteSeeds.filter((route) => requestedRouteIds.includes(route.id))
  : allRouteSeeds;
const globalPanos = reservedPanos(moduleRoot, routeSeeds.map((route) => route.id));
const overridePath = process.env.STREETVIEW_INITIAL_LINK_OVERRIDES;
let initialLinkOverrides = {};
if (overridePath && fs.existsSync(path.resolve(overridePath))) {
  initialLinkOverrides = JSON.parse(fs.readFileSync(path.resolve(overridePath), "utf8"));
}
if (requestedRouteIds.length && routeSeeds.length !== requestedRouteIds.length) {
  const found = new Set(routeSeeds.map((route) => route.id));
  const missing = requestedRouteIds.filter((id) => !found.has(id));
  throw new Error(`Unknown route id(s): ${missing.join(", ")}`);
}
const server = http.createServer((request, response) => {
  if (request.url !== "/maps_js_navigation.html") {
    response.writeHead(404);
    response.end("Not found");
    return;
  }
  response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  const html = fs.readFileSync(path.join(sourceRoot, "maps_js_navigation.html"), "utf8")
    .replace("__TRAJECTORY_ROUTES_JSON__", JSON.stringify(routeSeeds).replace(/<\//g, "<\\/"))
    .replace("</script>", `window.__GLOBAL_USED_PANOS__ = ${JSON.stringify(globalPanos)};window.__INITIAL_LINK_OVERRIDES__ = ${JSON.stringify(initialLinkOverrides).replace(/<\//g, "<\\/")};</script>`);
  response.end(html);
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
const executablePath = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const browser = await chromium.launch({ headless: true, executablePath });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/maps_js_navigation.html#${encodeURIComponent(key)}`, {
    waitUntil: "domcontentloaded",
    timeout: 60_000,
  });
  await page.waitForFunction(() => {
    const text = document.querySelector("#output")?.textContent || "";
    return text.length > 0 && text !== "loading";
  }, undefined, { timeout: 1_800_000 });
  const data = JSON.parse(await page.locator("#output").textContent());
  if (data.error) throw new Error(data.error);
  const outputPath = path.resolve(process.env.STREETVIEW_NAVIGATION_OUTPUT || path.join(moduleRoot, "data", "llm_guided_navigation.json"));
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
  console.log(data.routes.map((route) => `${route.id}: ${route.frames.length} frames`).join("\n"));
} finally {
  await browser.close();
  server.close();
}
