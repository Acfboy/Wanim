#!/usr/bin/env node
// Bundle the `<wanim-scene>` custom element for the browser.
//
// The element is authored in src/element.ts and compiled by `pnpm build` (tsc)
// to dist/element.js.  This script bundles that file into a small, minified ES
// module at dist/wanim-element.browser.js.
//
// The bundle deliberately keeps `py2js.browser.js` and `rapydscript.js` as
// *runtime* URLs relative to the bundle itself (resolved through
// `import.meta.url`), so the whole browser/ directory can be served as-is and
// the Python bridge only downloads on first run.  Importing the module has no
// side effects beyond registering the custom element (and is a no-op under
// Node/SSR, where `customElements` does not exist).
//
// Env overrides:
//   WANIM_ELEMENT_OUT  output file (default dist/wanim-element.browser.js)
//   WANIM_MINIFY       "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP    "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const entry = resolve(root, "dist", "element.js");

if (!existsSync(entry)) {
  console.error("wanim: dist/element.js not found; run `pnpm build` first.");
  process.exit(1);
}

const outfile = resolve(
  process.env.WANIM_ELEMENT_OUT ?? resolve(root, "dist", "wanim-element.browser.js"),
);

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
