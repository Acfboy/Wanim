#!/usr/bin/env node
// Bundle the py2js browser bridge for the demo site.
//
// The page compiles Python to JS with the browser rapydscript compiler
// (`dist/rapydscript.js`, built by `build-rapydscript-browser.mjs`) and then
// runs it against the wanim Canvas2D backend using the runtime shims in
// `py2js/src/browser.ts`.
//
// `wanim` and `numpy-ts` stay external but are referenced by *relative* URLs
// (the bundles all live in `dist/`), so the bridge works even when the page's
// import map is missing or was applied too late.  All three files resolve the
// same URLs, which keeps the numpy-ts instance (and its WASM heap) shared.
//
// Env overrides:
//   PY2JS_BROWSER_OUT  output file (default dist/py2js.browser.js)
//   WANIM_MINIFY       "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP    "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const entry = resolve(wanimRoot, "py2js", "src", "browser.ts");

if (!existsSync(resolve(wanimRoot, "dist", "index.js"))) {
  console.error("wanim is not built; run `pnpm build` first.");
  process.exit(1);
}

const outfile = resolve(process.env.PY2JS_BROWSER_OUT ?? resolve(wanimRoot, "dist", "py2js.browser.js"));

await build({
  entryPoints: [entry],
  outfile,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome110"],
  sourcemap: process.env.WANIM_SOURCEMAP === "1",
  minify: process.env.WANIM_MINIFY !== "0",
  logLevel: "info",
  plugins: [
    {
      name: "shared-relative",
      setup(build) {
        build.onResolve({ filter: /^numpy-ts$/ }, () => ({
          path: "./numpy-ts.browser.js",
          external: true,
        }));
        build.onResolve({ filter: /dist\/index\.js$/ }, () => ({
          path: "./wanim.browser.js",
          external: true,
        }));
      },
    },
  ],
  banner: {
    js: "if (typeof globalThis.process === 'undefined') { globalThis.process = { cwd: () => '/', env: {}, argv: [], platform: 'browser' }; }",
  },
  define: { "process.env.NODE_ENV": '"production"' },
});

console.log(`wanim: wrote ${outfile}`);
