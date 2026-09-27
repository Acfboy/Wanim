#!/usr/bin/env node
// Time (and optionally CPU-profile) a wanim scene's generation phase.
//
// Usage:
//   node scripts/profile_scene.mjs <user-scene.js> <SceneName> [-ql] [--repeat N]
//   node --cpu-prof --cpu-prof-dir=/tmp/opencode scripts/profile_scene.mjs ...
import { register } from "node:module";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const dist = process.env.WANIM_DIST ? resolve(process.env.WANIM_DIST) : join(wanimRoot, "dist");
process.env.WANIM_DIST = dist;
if (!existsSync(join(dist, "index.js"))) {
  console.error("wanim is not built; run `pnpm build` first.");
  process.exit(2);
}
register(pathToFileURL(join(here, "wanim-loader.mjs")).href);

const args = process.argv.slice(2);
const file = resolve(args[0]);
const sceneName = args[1];
const quality = args.find((a) => /^-q(l|m|h|4k)$/.test(a)) ?? "-ql";
const repeatIndex = args.indexOf("--repeat");
const repeat = repeatIndex >= 0 ? Number(args[repeatIndex + 1]) : 1;

const wanim = await import("wanim");
const module = await import(pathToFileURL(file).href);
const SceneClass = module[sceneName];
if (SceneClass === undefined) {
  console.error(`scene '${sceneName}' not exported by ${file}`);
  process.exit(1);
}

wanim.apply_quality(quality);
const times = [];
let frames = 0;
for (let i = 0; i < repeat; i += 1) {
  const out = join("/tmp/opencode", `${sceneName}_${i}.mjson`);
  const started = performance.now();
  wanim.run_scene(SceneClass, out);
  times.push(performance.now() - started);
  try {
    const doc = JSON.parse((await import("node:fs")).readFileSync(out, "utf-8"));
    frames = doc.meta.num_frames;
  } catch { /* ignore */ }
}
const best = Math.min(...times);
console.log(`scene ${sceneName} (${quality}): frames=${frames}`);
console.log(`  generation: ${best.toFixed(0)} ms -> ${(frames / (best / 1000)).toFixed(1)} gen fps  (runs: ${times.map((t) => t.toFixed(0)).join(", ")})`);
