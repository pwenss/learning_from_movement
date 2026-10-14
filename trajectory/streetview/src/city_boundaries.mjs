import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

function routeLocation(name) {
  const separator = name.lastIndexOf(",");
  return separator < 0
    ? { city: name.trim(), country: "" }
    : { city: name.slice(0, separator).trim(), country: name.slice(separator + 1).trim() };
}

export async function addCityBoundary(route) {
  const { city, country } = routeLocation(route.name);
  const curl = process.platform === "win32" ? "curl.exe" : "curl";
  const waitForNominatim = () => new Promise((resolve) => setTimeout(resolve, 1100));
  const query = async (parameters) => {
    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("format", "jsonv2");
    url.searchParams.set("limit", "10");
    url.searchParams.set("polygon_geojson", "1");
    url.searchParams.set("polygon_threshold", "0.0005");
    for (const [name, value] of Object.entries(parameters)) {
      if (value) url.searchParams.set(name, value);
    }
    const { stdout } = await execFileAsync(curl, [
      "--fail-with-body",
      "--silent",
      "--show-error",
      "--retry", "3",
      "--max-time", "45",
      "--user-agent", "streetview-trajectory-research/1.0",
      "--header", "Accept-Language: en",
      url.toString(),
    ], { maxBuffer: 50 * 1024 * 1024 });
    return JSON.parse(stdout);
  };

  const selectBoundary = (candidates) => {
    const polygons = candidates.filter((candidate) => ["Polygon", "MultiPolygon"].includes(candidate.geojson?.type));
    return polygons.find((candidate) => candidate.category === "boundary" && candidate.type === "administrative") || null;
  };
  const compositeQueries = {
    "Mumbai, India": ["Mumbai City District, India", "Mumbai Suburban District, India"],
  };
  const administrativeQueries = {
    "Johannesburg, South Africa": "City of Johannesburg Metropolitan Municipality, South Africa",
    "Gothenburg, Sweden": "Göteborgs kommun, Sweden",
    "Sydney, Australia": "City of Sydney, Australia",
    "Brisbane, Australia": "City of Brisbane, Australia",
    "Singapore": "Republic of Singapore",
  };
  const queries = [
    ...(administrativeQueries[route.name] ? [{ q: administrativeQueries[route.name] }] : []),
    { city, country },
    { q: [city, country].filter(Boolean).join(", ") },
    { q: [`Municipality of ${city}`, country].filter(Boolean).join(", ") },
    { q: [`City of ${city}`, country].filter(Boolean).join(", ") },
  ];
  let boundary = null;
  for (let index = 0; index < queries.length && !boundary; index += 1) {
    if (index > 0) await waitForNominatim();
    boundary = selectBoundary(await query(queries[index]));
  }
  if (!boundary && compositeQueries[route.name]) {
    const parts = [];
    for (const text of compositeQueries[route.name]) {
      await waitForNominatim();
      const part = selectBoundary(await query({ q: text }));
      if (!part || part.category !== "boundary") throw new Error(`No administrative boundary part found for ${text}`);
      parts.push(part);
    }
    const boxes = parts.map((part) => part.boundingbox.map(Number));
    boundary = {
      geojson: {
        type: "MultiPolygon",
        coordinates: parts.flatMap((part) => part.geojson.type === "Polygon" ? [part.geojson.coordinates] : part.geojson.coordinates),
      },
      boundingbox: [
        Math.min(...boxes.map((box) => box[0])),
        Math.max(...boxes.map((box) => box[1])),
        Math.min(...boxes.map((box) => box[2])),
        Math.max(...boxes.map((box) => box[3])),
      ],
      osm_type: "composite",
      osm_id: parts.map((part) => `${part.osm_type}/${part.osm_id}`).join("+"),
      display_name: parts.map((part) => part.display_name).join(" + "),
      category: "boundary",
      type: "administrative",
    };
  }
  if (!boundary) throw new Error(`No administrative boundary found for ${route.name}`);
  const [south, north, west, east] = boundary.boundingbox.map(Number);
  return {
    ...route,
    city_boundary: boundary.geojson,
    city_bounds: [south, north, west, east],
    city_boundary_metadata: {
      osm_type: boundary.osm_type,
      osm_id: boundary.osm_id,
      display_name: boundary.display_name,
      category: boundary.category,
      type: boundary.type,
    },
  };
}

export async function addCityBoundaries(routes) {
  const result = [];
  for (const route of routes) {
    result.push(await addCityBoundary(route));
    if (routes.length > 1) await new Promise((resolve) => setTimeout(resolve, 1100));
  }
  return result;
}
