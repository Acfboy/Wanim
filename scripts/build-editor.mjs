#!/usr/bin/env node
// Bundle the playground's CodeMirror editor for the demo page.
//
// The editor is authored in demo/editor.js and mirrors the website
// playground's CodeMirror setup (Python language, One Dark theme).  The demo
// page loads the bundle from dist/editor.browser.js like the other browser
// bundles; CodeMirror is a devDependency and is not part of the published
// package.
//
// Env overrides:
//   WANIM_EDITOR_OUT   output file (default dist/editor.browser.js)
//   WANIM_MINIFY       "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP    "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const entry = resolve(root, "demo", "editor.js");

if (!existsSync(entry)) {
  console.error(`wanim: ${entry} not found`);
  process.exit(1);
}

const outfile = resolve(
  process.env.WANIM_EDITOR_OUT ?? resolve(root, "dist", "editor.browser.js"),
);
mkdirSync(dirname(outfile), { recursive: true });

await build({
  entryPoints: [entry],
  outfile,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome110"],
  minify: process.env.WANIM_MINIFY !== "0",
  sourcemap: process.env.WANIM_SOURCEMAP === "1",
  logLevel: "info",
});

console.log(`wanim: wrote ${outfile}`);
