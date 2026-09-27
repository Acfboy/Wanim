#!/usr/bin/env node
// Summarize a V8 .cpuprofile by self time (default) or total time.
//
// Usage: node scripts/analyze_cpuprofile.mjs <profile.cpuprofile> [--total] [--top N]
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
const byTotal = args.includes("--total");
const topIndex = args.indexOf("--top");
const top = topIndex >= 0 ? Number(args[topIndex + 1]) : 25;

const profile = JSON.parse(readFileSync(file, "utf-8"));
const byId = new Map();
for (const node of profile.nodes) byId.set(node.id, node);

const selfTime = new Map();
for (let i = 0; i < profile.samples.length; i += 1) {
  const id = profile.samples[i];
  const dt = profile.timeDeltas[i] ?? 0;
  selfTime.set(id, (selfTime.get(id) ?? 0) + dt);
}

const totalTime = new Map();
function computeTotal(id, seen) {
  if (totalTime.has(id)) return totalTime.get(id);
  if (seen.has(id)) return 0;
  seen.add(id);
  const node = byId.get(id);
  let sum = selfTime.get(id) ?? 0;
  for (const child of node?.children ?? []) sum += computeTotal(child, seen);
  totalTime.set(id, sum);
  return sum;
}
for (const id of byId.keys()) computeTotal(id, new Set());

const label = (node) => {
  const frame = node.callFrame;
  const name = frame.functionName || "(anonymous)";
  const url = (frame.url || "").replace(/^file:\/\//, "");
  const short = url.replace(/.*\/wanim\//, "").replace(/.*\/numpy-ts\//, "numpy-ts/");
  return `${name} @ ${short}:${frame.lineNumber + 1}`;
};

const table = byTotal ? totalTime : selfTime;
const rows = [...table.entries()]
  .map(([id, time]) => ({ time, node: byId.get(id) }))
  .filter((row) => row.node)
  .sort((a, b) => b.time - a.time)
  .slice(0, top);

const totalDuration = profile.endTime - profile.startTime;
console.log(`total ${(totalDuration / 1000).toFixed(0)} ms  (${profile.samples.length} samples)`);
console.log(`${byTotal ? "TOTAL" : "SELF"} TIME`);
for (const row of rows) {
  console.log(`${(row.time / 1000).toFixed(1).padStart(8)} ms  ${((row.time / totalDuration) * 100).toFixed(1).padStart(5)}%  ${label(row.node)}`);
}
