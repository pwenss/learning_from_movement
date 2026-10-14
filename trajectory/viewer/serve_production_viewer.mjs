import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const moduleRoot = path.dirname(fileURLToPath(import.meta.url));
const codeRoot = path.resolve(moduleRoot, "..", "..");
const projectRoot = path.basename(codeRoot) === "code" ? path.dirname(codeRoot) : codeRoot;
const datasetId = process.env.STREETVIEW_DATASET || "sv30_20260920_r100_r3";
const datasetRoot = path.resolve(
  process.env.STREETVIEW_DATASET_ROOT || path.join(projectRoot, "datasets", datasetId),
);
const configFile = path.join(moduleRoot, "viewer_config.json");
const viewerConfig = fs.existsSync(configFile) ? JSON.parse(fs.readFileSync(configFile, "utf8")) : {};
const satelliteDatasetRoot = path.resolve(
  process.env.SATELLITE_DATASET_ROOT || viewerConfig.satellite_dataset_root || path.join(projectRoot, "datasets", "satellite_trajectory_v1"),
);
const evaluationRoot = path.resolve(
  process.env.STREETVIEW_EVALUATION_ROOT || path.join(projectRoot, "evaluation", "results"),
);
const viewerFile = path.join(moduleRoot, "production_viewer.html");
const host = process.env.STREETVIEW_VIEWER_HOST || "127.0.0.1";
const port = Number(process.env.STREETVIEW_VIEWER_PORT || 8787);

const cityLabels = {
  nyc_new_york: "New York",
  sf_san_francisco: "San Francisco",
  chicago: "Chicago",
  philadelphia: "Philadelphia",
  toronto: "Toronto",
  vancouver: "Vancouver",
  mexico_city: "Mexico City",
  sao_paulo: "Sao Paulo",
  rio_de_janeiro: "Rio de Janeiro",
  buenos_aires: "Buenos Aires",
  london: "London",
  paris: "Paris",
  berlin: "Berlin",
  madrid: "Madrid",
  rome: "Rome",
  amsterdam: "Amsterdam",
  zurich: "Zurich",
  vienna: "Vienna",
  stockholm: "Stockholm",
  prague: "Prague",
  tokyo: "Tokyo",
  osaka: "Osaka",
  seoul: "Seoul",
  taipei: "Taipei",
  singapore: "Singapore",
  bangkok: "Bangkok",
  kuala_lumpur: "Kuala Lumpur",
  sydney: "Sydney",
  melbourne: "Melbourne",
  auckland: "Auckland",
};

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function loadCatalog() {
  const rows = readJson(path.join(datasetRoot, "index.json"));
  return rows.map((row) => {
    const cityDirectory = path.dirname(row.path);
    const cityId = row.city_id;
    const summaryPath = path.join(datasetRoot, cityDirectory, "summary.json");
    const summary = fs.existsSync(summaryPath) ? readJson(summaryPath) : {};
    return {
      id: cityId,
      label: cityLabels[cityId] || cityId.replaceAll("_", " "),
      complete: Number(row.complete || summary.complete || 0),
      imageCount: Number(summary.image_count || (row.complete || 0) * 8),
      directory: cityDirectory,
    };
  });
}

const cities = loadCatalog();
const cityById = new Map(cities.map((city) => [city.id, city]));
const routeIndexCache = new Map();
const evaluationCache = new Map();

function cityRoot(cityId) {
  const city = cityById.get(cityId);
  if (!city) return null;
  return path.join(datasetRoot, city.directory, "trajectory_sequences_180");
}

function routeIndex(cityId) {
  if (!routeIndexCache.has(cityId)) {
    const root = cityRoot(cityId);
    const rows = readJson(path.join(root, "index.json"));
    routeIndexCache.set(cityId, rows.map((row) => ({ ...row, city_id: cityId })));
  }
  return routeIndexCache.get(cityId);
}

function routeManifest(cityId, routeId) {
  const route = routeIndex(cityId).find((row) => row.id === routeId);
  if (!route || typeof route.path !== "string" || route.path.includes("..")) return null;
  const root = cityRoot(cityId);
  const expected = path.resolve(root, route.path);
  if (!expected.startsWith(`${path.resolve(root)}${path.sep}`)) return null;
  return readJson(expected);
}

function satelliteCityRoot(cityId) {
  const root = path.join(satelliteDatasetRoot, "cities", cityId, "trajectory_sequences");
  return fs.existsSync(root) ? root : null;
}

function satelliteRouteIndex(cityId) {
  const root = satelliteCityRoot(cityId);
  if (!root) return [];
  const index = path.join(path.dirname(root), "index.json");
  if (!fs.existsSync(index)) return [];
  return readJson(index).map((row) => ({ ...row, city_id: cityId }));
}

function satelliteManifest(cityId, routeId) {
  if (!/^[-a-z0-9_]+$/i.test(routeId)) return null;
  const root = satelliteCityRoot(cityId);
  if (!root) return null;
  const row = satelliteRouteIndex(cityId).find((item) => item.id === routeId);
  if (!row || typeof row.path !== "string" || row.path.includes("..")) return null;
  const cityRootPath = path.resolve(satelliteDatasetRoot, "cities", cityId);
  const target = path.resolve(cityRootPath, row.path);
  if (!target.startsWith(`${cityRootPath}${path.sep}`) || !fs.existsSync(target)) return null;
  return readJson(target);
}

function evaluationRuns() {
  if (!fs.existsSync(evaluationRoot)) return [];
  return fs.readdirSync(evaluationRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory())
    .filter((entry) => /^eval_[a-z0-9_]+$/i.test(entry.name)).map((entry) => {
      const directory = path.join(evaluationRoot, entry.name);
      const resultPath = path.join(directory, "results.json");
      const manifestPath = path.join(directory, "sample_manifest.json");
      if (!fs.existsSync(resultPath) || !fs.existsSync(manifestPath)) return null;
      const results = readJson(resultPath);
      const metrics = results.metrics || {};
      return {
        id: entry.name,
        label: results.model_requested || results.model || entry.name,
        model: results.model_requested || results.model || "unknown",
        provider: results.provider || "",
        updated_at: results.updated_at || results.created_at || "",
        total: Number(metrics.total || (results.sequences || []).length),
        exact_match_accuracy: metrics.exact_match_accuracy,
        kendall_tau: metrics.kendall_tau_valid_answers,
        position_accuracy: metrics.position_accuracy_valid_answers,
      };
    }).filter(Boolean);
}

function evaluationRun(runId) {
  if (!/^eval_[a-z0-9_]+$/i.test(runId)) return null;
  if (evaluationCache.has(runId)) return evaluationCache.get(runId);
  const directory = path.join(evaluationRoot, runId);
  const resultPath = path.join(directory, "results.json");
  const manifestPath = path.join(directory, "sample_manifest.json");
  if (!fs.existsSync(resultPath) || !fs.existsSync(manifestPath)) return null;
  const results = readJson(resultPath);
  const manifest = readJson(manifestPath);
  const manifestById = new Map((manifest.sequences || []).map((row) => [row.id, row]));
  const records = new Map((results.sequences || []).map((row) => [row.id, {
    ...row,
    images: (manifestById.get(row.id) || {}).images || [],
  }]));
  const value = { results, records };
  evaluationCache.set(runId, value);
  return value;
}

function evaluationSummary(runId) {
  const run = evaluationRun(runId);
  if (!run) return null;
  const results = run.results;
  return {
    id: runId,
    label: results.model_requested || results.model || runId,
    model: results.model_requested || results.model || "unknown",
    provider: results.provider || "",
    updated_at: results.updated_at || results.created_at || "",
    metrics: results.metrics || {},
    cities: results.per_city || {},
    routes: [...run.records.values()].map((row) => ({
      id: row.id, city: row.city, valid_answer: row.valid_answer,
      exact_match: row.exact_match, kendall_tau: row.kendall_tau,
      position_accuracy: row.position_accuracy,
    })),
  };
}

function modelOutput(row) {
  let content = row.content;
  let reasoning = row.reasoning_content;
  let model = row.model;
  for (const attempt of [...(row.attempts || [])].reverse()) {
    const body = (((attempt.result || {}).response || {}).body || {});
    const message = (((body.choices || [])[0] || {}).message || {});
    content ||= message.content;
    reasoning ||= message.reasoning_content;
    model ||= body.model;
    if (content || reasoning || model) break;
  }
  return { content, reasoning, model };
}

function evaluationRecord(runId, routeId) {
  const summary = evaluationSummary(runId);
  const run = evaluationRun(runId);
  if (!summary || !run || !/^[-a-z0-9_]+$/i.test(routeId)) return null;
  const row = run.records.get(routeId);
  if (!row) return null;
  const manifest = routeManifest(row.city, routeId);
  if (!manifest) return null;
  // Manifest images are in original trajectory order, as in the evaluator.
  const images = row.shuffle_presented_to_original.map((originalIndex, index) => {
    const image = row.images[originalIndex];
    const file = path.basename(image.path || "");
    if (!/^frame_[0-7]{2}\.jpg$/i.test(file)) return null;
    return { presented_index: index, original_index: Number(originalIndex), file };
  }).filter(Boolean);
  const response = {};
  for (const key of ["id", "city", "custom_id", "shuffle_presented_to_original", "ground_truth_presented_indices", "valid_answer", "cot_format_valid", "prediction", "exact_match", "kendall_tau", "pairwise_accuracy", "directed_adjacency_accuracy", "position_accuracy", "finish_reason"]) {
    if (key in row) response[key] = row[key];
  }
  const output = modelOutput(row);
  if (output.content != null) response.content = output.content;
  if (output.reasoning != null) response.reasoning_content = output.reasoning;
  if (output.model != null) response.model = output.model;
  response.images = images;
  response.frames = manifest.frames || [];
  response.model_label = summary.label;
  return response;
}

function sendJson(response, value, status = 200) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*",
  });
  response.end(body);
}

function sendText(response, body, contentType, status = 200) {
  response.writeHead(status, { "content-type": contentType, "cache-control": "no-store" });
  response.end(body);
}

function serveImage(request, response, cityId, routeId, fileName) {
  if (!cityById.has(cityId) || !/^[-a-z0-9_]+$/i.test(routeId) || !/^frame_[0-7]{2}\.jpg$/i.test(fileName)) {
    return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
  }
  const route = routeIndex(cityId).find((row) => row.id === routeId);
  if (!route) return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
  const imagePath = path.join(cityRoot(cityId), routeId, fileName);
  if (!fs.existsSync(imagePath)) return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
  response.writeHead(200, {
    "content-type": "image/jpeg",
    "cache-control": "public, max-age=86400, immutable",
    "content-length": fs.statSync(imagePath).size,
  });
  fs.createReadStream(imagePath).pipe(response);
}

function start() {
  if (!fs.existsSync(datasetRoot)) throw new Error(`Dataset directory does not exist: ${datasetRoot}`);
  const html = fs.readFileSync(viewerFile, "utf8");
  const summary = fs.existsSync(path.join(datasetRoot, "summary.json"))
    ? readJson(path.join(datasetRoot, "summary.json"))
    : {};
  const boundariesPath = path.join(datasetRoot, "city_boundaries.geojson");

  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url, `http://${host}:${port}`);
      if (url.pathname === "/" || url.pathname === "/index.html") {
        return sendText(response, html, "text/html; charset=utf-8");
      }
      if (url.pathname === "/api/meta") {
        return sendJson(response, {
          dataset_id: datasetId,
          dataset_complete: Boolean(summary.dataset_complete),
          total_trajectories: cities.reduce((sum, city) => sum + city.complete, 0),
          total_images: cities.reduce((sum, city) => sum + city.imageCount, 0),
          cities,
        });
      }
      if (url.pathname === "/api/boundaries" && fs.existsSync(boundariesPath)) {
        return sendJson(response, readJson(boundariesPath));
      }
      if (url.pathname === "/api/satellite/meta") {
        const summaryPath = path.join(satelliteDatasetRoot, "summary.json");
        const summary = fs.existsSync(summaryPath) ? readJson(summaryPath) : {};
        return sendJson(response, {
          available: fs.existsSync(satelliteDatasetRoot),
          dataset_id: summary.dataset_id || path.basename(satelliteDatasetRoot),
          total_trajectories: Number(summary.total_trajectories || 0),
          cities: cities.filter((city) => satelliteRouteIndex(city.id).length),
        });
      }
      const satelliteRoutes = /^\/api\/satellite\/cities\/([^/]+)\/routes$/.exec(url.pathname);
      if (satelliteRoutes) {
        const cityId = decodeURIComponent(satelliteRoutes[1]);
        if (!cityById.has(cityId)) return sendJson(response, { error: "Unknown city" }, 404);
        return sendJson(response, satelliteRouteIndex(cityId));
      }
      const satelliteRoute = /^\/api\/satellite\/cities\/([^/]+)\/routes\/([^/]+)$/.exec(url.pathname);
      if (satelliteRoute) {
        const cityId = decodeURIComponent(satelliteRoute[1]);
        const routeId = decodeURIComponent(satelliteRoute[2]);
        const manifest = cityById.has(cityId) ? satelliteManifest(cityId, routeId) : null;
        if (!manifest) return sendJson(response, { error: "Unknown satellite trajectory" }, 404);
        return sendJson(response, manifest);
      }
      if (url.pathname === "/api/evaluations") {
        return sendJson(response, { runs: evaluationRuns() });
      }
      const evaluationRoutes = /^\/api\/evaluations\/([^/]+)\/routes$/.exec(url.pathname);
      if (evaluationRoutes) {
        const summary = evaluationSummary(decodeURIComponent(evaluationRoutes[1]));
        if (!summary) return sendJson(response, { error: "Unknown evaluation" }, 404);
        return sendJson(response, summary);
      }
      const evaluationRoute = /^\/api\/evaluations\/([^/]+)\/routes\/([^/]+)$/.exec(url.pathname);
      if (evaluationRoute) {
        const record = evaluationRecord(decodeURIComponent(evaluationRoute[1]), decodeURIComponent(evaluationRoute[2]));
        if (!record) return sendJson(response, { error: "Unknown evaluated trajectory" }, 404);
        return sendJson(response, record);
      }
      const cityRoutes = /^\/api\/cities\/([^/]+)\/routes$/.exec(url.pathname);
      if (cityRoutes) {
        const cityId = decodeURIComponent(cityRoutes[1]);
        if (!cityById.has(cityId)) return sendJson(response, { error: "Unknown city" }, 404);
        return sendJson(response, routeIndex(cityId));
      }
      const manifestMatch = /^\/api\/cities\/([^/]+)\/routes\/([^/]+)$/.exec(url.pathname);
      if (manifestMatch) {
        const cityId = decodeURIComponent(manifestMatch[1]);
        const routeId = decodeURIComponent(manifestMatch[2]);
        const manifest = cityById.has(cityId) ? routeManifest(cityId, routeId) : null;
        if (!manifest) return sendJson(response, { error: "Unknown trajectory" }, 404);
        return sendJson(response, manifest);
      }
      const imageMatch = /^\/media\/([^/]+)\/([^/]+)\/(frame_[0-7]{2}\.jpg)$/.exec(url.pathname);
      if (imageMatch) return serveImage(request, response, decodeURIComponent(imageMatch[1]), decodeURIComponent(imageMatch[2]), imageMatch[3]);
      const satelliteImage = /^\/satellite\/([^/]+)\/([^/]+)\/(frame_[0-7]{2}\.jpg)$/.exec(url.pathname);
      if (satelliteImage) {
        const cityId = decodeURIComponent(satelliteImage[1]);
        const routeId = decodeURIComponent(satelliteImage[2]);
        const manifest = cityById.has(cityId) ? satelliteManifest(cityId, routeId) : null;
        if (!manifest) return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
        const imagePath = path.resolve(satelliteDatasetRoot, "cities", cityId, "trajectory_sequences", routeId, satelliteImage[3]);
        const base = path.resolve(satelliteDatasetRoot, "cities", cityId, "trajectory_sequences", routeId);
        if (!imagePath.startsWith(`${base}${path.sep}`) || !fs.existsSync(imagePath)) return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
        response.writeHead(200, { "content-type": "image/jpeg", "cache-control": "public, max-age=86400, immutable", "content-length": fs.statSync(imagePath).size });
        return fs.createReadStream(imagePath).pipe(response);
      }
      return sendText(response, "Not found", "text/plain; charset=utf-8", 404);
    } catch (error) {
      console.error(error);
      return sendJson(response, { error: error.message }, 500);
    }
  });
  server.listen(port, host, () => {
    console.log(`Production trajectory viewer: http://${host}:${port}/`);
    console.log(`Dataset: ${datasetId} (${cities.length} cities, ${cities.reduce((sum, city) => sum + city.complete, 0)} trajectories)`);
  });
}

start();
