#!/usr/bin/env node
// Microbenchmarks for the graphing hot paths (task 11).
//
// Usage:
//   node scripts/bench_hotpaths.mjs [--repeat N] [--samples N] [--json path]
//
// Measures, on the built `dist/` bundle:
//   - a single axes.coords_to_point(t, sin t) call
//   - number_to_point on a NumberLine
//   - axes.plot(sin) (203 samples)
//   - an end-to-end SinAndCosFunctionPlot-style construct (JS equivalent)
//   - a full run_scene generation for the same scene
//
// Prints a table and (with --json) a machine-readable snapshot so it can be
// rerun as a before/after regression benchmark.
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
const num = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const repeat = num("--repeat", 7);
const jsonIndex = args.indexOf("--json");
const jsonPath = jsonIndex >= 0 ? args[jsonIndex + 1] : null;
const baselineIndex = args.indexOf("--baseline");
const baselinePath = baselineIndex >= 0 ? args[baselineIndex + 1] : null;
let baseline = null;
if (baselinePath) {
  const fs = await import("node:fs");
  baseline = JSON.parse(fs.readFileSync(baselinePath, "utf-8"));
}

const wanim = await import("wanim");
const {
  Axes,
  NumberLine,
  Line,
  Scene,
  apply_quality,
  GREEN,
  BLUE,
  RED,
  YELLOW,
  UP,
  DOWN,
  TAU,
} = wanim;

apply_quality("-ql");

const sin = (x) => Math.sin(x);
const cos = (x) => Math.cos(x);

function makeAxes() {
  return new Axes({
    x_range: [-10, 10.3, 1],
    y_range: [-1.5, 1.5, 1],
    x_length: 10,
    axis_config: { color: GREEN },
    x_axis_config: {
      numbers_to_include: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
      numbers_with_elongated_ticks: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
    },
    tips: false,
  });
}

// Mirror of the Python SinAndCosFunctionPlot construct body (mobject-only,
// no rendering), so the number is directly comparable across both variants.
function constructSinCos() {
  const axes = makeAxes();
  const axes_labels = axes.get_axis_labels();
  const sin_graph = axes.plot(sin, { color: BLUE });
  const cos_graph = axes.plot(cos, { color: RED });
  const sin_label = axes.get_graph_label(sin_graph, "\\sin(x)", { x_val: -10, direction: [UP[0] / 2, UP[1] / 2, UP[2] / 2] });
  const cos_label = axes.get_graph_label(cos_graph, "\\cos(x)", { x_val: 10, direction: [DOWN[0] / 2, DOWN[1] / 2, DOWN[2] / 2] });
  const vert_line = axes.get_vertical_line(axes.i2gp(TAU, cos_graph), { color: YELLOW, line_func: Line });
  return { axes, axes_labels, sin_graph, cos_graph, sin_label, cos_label, vert_line };
}

class BenchSinCos extends Scene {
  construct() {
    const parts = constructSinCos();
    this.add(
      parts.axes,
      parts.axes_labels,
      parts.sin_graph,
      parts.cos_graph,
      parts.sin_label,
      parts.cos_label,
      parts.vert_line,
    );
  }
}

function bench(fn, iterations) {
  // warm up
  for (let i = 0; i < Math.min(iterations, 50); i += 1) fn();
  const samples = [];
  for (let r = 0; r < repeat; r += 1) {
    const started = performance.now();
    for (let i = 0; i < iterations; i += 1) fn();
    samples.push((performance.now() - started) / iterations);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

const axes = makeAxes();
const numberLine = new NumberLine({ x_range: [-4, 4, 1], unit_size: 1 });

const results = {};
results.coords_to_point_us = bench(() => axes.coords_to_point(1.234, Math.sin(1.234)), 20000) * 1000;
results.number_to_point_us = bench(() => numberLine.number_to_point(1.234), 20000) * 1000;
results.plot_ms = bench(() => axes.plot(sin, { color: BLUE }), 20);
results.construct_ms = bench(constructSinCos, 10);

const out = join("/tmp/opencode", `bench_sincos.mjson`);
const t0 = performance.now();
wanim.run_scene(BenchSinCos, out);
results.run_scene_ms = performance.now() - t0;

const fmt = (v, digits = 2) => (v < 0.01 ? v.toExponential(2) : v.toFixed(digits));
const rows = [
  ["coords_to_point(t, sin t)", "coords_to_point_us", "us/call", 2],
  ["NumberLine.number_to_point", "number_to_point_us", "us/call", 2],
  ["axes.plot(sin) 203 samples", "plot_ms", "ms", 3],
  ["SinAndCos construct", "construct_ms", "ms", 3],
  ["run_scene (construct+ser.)", "run_scene_ms", "ms", 3],
];
if (baseline) {
  console.log("hot-path benchmark (median); delta vs baseline");
  console.log(
    `  ${"case".padEnd(30)} ${"before".padStart(11)} ${"after".padStart(11)} ${"speedup".padStart(9)}`,
  );
  for (const [label, key, unit, digits] of rows) {
    const before = baseline[key] ?? 0;
    const after = results[key] ?? 0;
    const speedup = after > 0 ? before / after : 0;
    console.log(
      `  ${label.padEnd(30)} ${(fmt(before, digits) + " " + unit).padStart(11)} ${(fmt(after, digits) + " " + unit).padStart(11)} ${(speedup.toFixed(1) + "x").padStart(9)}`,
    );
  }
} else {
  console.log("hot-path benchmark (median of runs)");
  console.log("  coords_to_point(t, sin t) :", fmt(results.coords_to_point_us), "us/call");
  console.log("  NumberLine.number_to_point:", fmt(results.number_to_point_us), "us/call");
  console.log("  axes.plot(sin) 203 samples :", fmt(results.plot_ms, 3), "ms");
  console.log("  SinAndCos construct        :", fmt(results.construct_ms, 3), "ms");
  console.log("  run_scene (construct+ser.) :", fmt(results.run_scene_ms, 3), "ms");
}

if (jsonPath) {
  const fs = await import("node:fs");
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log("wrote", jsonPath);
}
