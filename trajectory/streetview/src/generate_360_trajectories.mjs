// Compatibility entry point for the current 180-degree trajectory renderer.
// Image acquisition is delegated to robolyst/streetview; no official Google
// Street View Static API or Map Tiles API endpoint is called by this stage.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const script = fileURLToPath(new URL("./render_trajectories_streetview.py", import.meta.url));
const bundledPython = "C:\\Users\\admin\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const python = process.env.PYTHON_EXECUTABLE || (fs.existsSync(bundledPython) ? bundledPython : "python");
const result = spawnSync(python, [script], {
  cwd: path.dirname(script),
  env: { ...process.env, STREETVIEW_OUTPUT_MODE: process.env.STREETVIEW_OUTPUT_MODE || "180" },
  stdio: "inherit",
});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
