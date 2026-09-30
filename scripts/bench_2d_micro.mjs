#!/usr/bin/env node
// Micro-benchmarks for the 2D hot paths (construct-time primitives).
//
// Usage:
//   node scripts/bench_2d_micro.mjs [--repeat N] [--json out.json]
//
// Prints per-op median ms on the built `dist/` bundle.  These are the
// primitives that dominate scene `construct` time (camera/background, text,
// geometry, ticks, vector fields).  Use it as a before/after regression bench.
import { register } from "node:module";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
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
const num = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const repeat = num("--repeat", 7);
const jsonIndex = args.indexOf("--json");
const jsonPath = jsonIndex >= 0 ? args[jsonIndex + 1] : null;
const baselineIndex = args.indexOf("--baseline");
const baselinePath = baselineIndex >= 0 ? args[baselineIndex + 1] : null;
const baseline = baselinePath ? JSON.parse(readFileSync(baselinePath, "utf-8")) : null;

const wanim = await import("wanim");
const {
  Camera,
  Axes,
  NumberLine,
  ArrowVectorField,
  DecimalNumber,
  Circle,
  Square,
  Line,
  Triangle,
  Arrow,
  GREEN,
} = wanim;
wanim.apply_quality("-ql");

const ops = {
  camera: () => new Camera(),
  number: () => new DecimalNumber(3.14),
  circle: () => new Circle({ radius: 0.5 }),
  square: () => new Square({ side_length: 1 }),
  line: () => new Line([0, 0, 0], [1, 1, 0]),
  triangle: () => new Triangle(),
  arrow: () => new Arrow([0, 0, 0], [1, 1, 0]),
  numberline: () => new NumberLine(),
  axes: () => new Axes(),
  vecfield: () =>
    new ArrowVectorField((p) => [-p[1], p[0], 0], { x_range: [-3, 3, 1], y_range: [-2, 2, 1] }),
};

function bench(fn, iterations) {
  for (let i = 0; i < Math.min(iterations, 10); i += 1) fn();
  const samples = [];
  for (let r = 0; r < repeat; r += 1) {
    const started = performance.now();
    for (let i = 0; i < iterations; i += 1) fn();
    samples.push((performance.now() - started) / iterations);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

const results = {};
for (const [name, fn] of Object.entries(ops)) results[name] = bench(fn, 60);

const fmt = (v) => (v < 0.01 ? v.toExponential(2) : v.toFixed(3));
console.log("2D hot-path micro-benchmarks (median ms/op)");
console.log("op".padEnd(14), "ms".padStart(10), baseline ? "before".padStart(10) + "  ratio" : "");
for (const [name, v] of Object.entries(results)) {
  let extra = "";
  if (baseline && typeof baseline[name] === "number") {
    extra = `  ${fmt(baseline[name]).padStart(9)}  ${(baseline[name] / v).toFixed(2)}x`;
  }
  console.log(name.padEnd(14), fmt(v).padStart(10) + extra);
}

if (jsonPath) {
  writeFileSync(jsonPath, JSON.stringify(results, null, 2) + "\n");
  console.log(`wrote ${jsonPath}`);
}
void GREEN;
