#!/usr/bin/env node
// Bundle the built wanim library for the browser.
//
// The library is authored as plain Node/ESM with a handful of `node:*` imports
// (config files, TeX caches, fonts).  For the browser we bundle `dist/index.js`
// and alias those builtins to tiny shims (`scripts/shims/*.js`); the shimmed
// paths are only reached by TeX/typst/font code that the Canvas2D demo does not
// use.
//
// Output: dist/wanim.browser.js, an ES module usable from an import map entry
// `"wanim"`.  `dist/` is gitignored: the demo page references it, so run
// `pnpm build:demo` before `pnpm demo`.
//
// The browser text pipeline defaults to free CDN fonts (Tinos + Noto Serif
// SC, see `DEFAULT_FONT_URLS`); embedders can point it elsewhere with
// `configure_fonts([...])`.  Only the bundle itself is written here.
//
// Env overrides:
//   WANIM_BROWSER_OUT  output file (default dist/wanim.browser.js)
//   WANIM_MINIFY       "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP    "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const shim = (name) => resolve(here, "shims", name);

const outfile = resolve(process.env.WANIM_BROWSER_OUT ?? resolve(root, "dist", "wanim.browser.js"));
const minify = process.env.WANIM_MINIFY !== "0";
const sourcemap = process.env.WANIM_SOURCEMAP === "1";

// numpy-ts is kept external, but referenced by a *relative* URL rather than a
// bare specifier: the bundle then resolves it against its own location, so it
// works even when the page's import map is missing or was applied too late.
const NUMPY_RELATIVE = "./numpy-ts.browser.js";

await build({
  entryPoints: [resolve(root, "dist", "index.js")],
  outfile,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome110"],
  sourcemap,
  // NOTE: identifiers (including parameter names) are deliberately *not*
  // mangled.  py2js maps Python `**kwargs` onto wanim methods by reading the
  // method's parameter names at runtime (`Function.prototype.toString`), so
  // mangling them (esbuild's `minifyIdentifiers`, part of `minify: true`) would
  // silently break calls like `self.set_camera_orientation(phi=..., theta=...)`.
  // Whitespace/syntax minification is safe and, after gzip, nearly as small.
  minifyWhitespace: minify,
  minifySyntax: minify,
  // wanim's scene discovery is name-based (`prototype.constructor.name ===
  // "Scene"`); bundling turns `class Scene` into an anonymous class expression
  // unless names are preserved.
  keepNames: true,
  logLevel: "info",
  // Keep numpy-ts external so the page loads it from a single URL and can
  // `configureWasm()` *before* wanim's first array operation (a heap sized at
  // import time cannot be resized afterwards).
  plugins: [
    {
      name: "numpy-relative",
      setup(build) {
        build.onResolve({ filter: /^numpy-ts$/ }, () => ({
          path: NUMPY_RELATIVE,
          external: true,
        }));
      },
    },
  ],
  alias: {
    "node:fs/promises": shim("node-fs-promises.js"),
    "node:fs": shim("node-fs.js"),
    "node:path": shim("node-path.js"),
    "node:os": shim("node-os.js"),
    "node:child_process": shim("node-child_process.js"),
  },
  banner: {
    js: "if (typeof globalThis.process === 'undefined') { globalThis.process = { cwd: () => '/', env: {}, argv: [], stdout: { write() {} }, platform: 'browser' }; }",
  },
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});

// Ship numpy-ts next to the bundle so the relative import resolves for the
// demo and other local consumers serving `dist/`.
const numpySource = resolve(root, "node_modules", "numpy-ts", "dist", "numpy-ts.browser.js");
if (existsSync(numpySource)) {
  const numpyOut = resolve(dirname(outfile), "numpy-ts.browser.js");
  mkdirSync(dirname(numpyOut), { recursive: true });
  copyFileSync(numpySource, numpyOut);
  console.log(`wanim: wrote ${numpyOut}`);
} else {
  console.warn(`wanim: numpy-ts browser build not found at ${numpySource}`);
}

console.log(`wanim: wrote ${outfile}`);
