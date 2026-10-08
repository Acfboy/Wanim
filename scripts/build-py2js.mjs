#!/usr/bin/env node
// Bundle the py2js browser bridge for the demo site.
//
// The page compiles Python to JS with the browser rapydscript compiler
// (`dist/rapydscript.js`, built by `build-rapydscript-browser.mjs`) and then
// runs it against the wanim Canvas2D backend using the runtime shims in
// `py2js/src/browser.ts`.
//
// `wanim` and `numpy-ts` stay external.  By default they are referenced by
// *relative* URLs (the bundles all live in `dist/`), so the bridge works even
// when the page's import map is missing or was applied too late, and all three
// files resolve the same URLs, keeping the numpy-ts instance (and its WASM
// heap) shared.  This is the build the demo site and the combined `wanim`
// package use.
//
// With `--bare` (or `PY2JS_EXTERNAL=bare`) the externals are left as *bare*
// specifiers instead: `import ... from "wanim"` / `"numpy-ts"`.  That is the
// variant shipped in the standalone `wanim-py` npm package, where the
// consumer's import map (or bundler) resolves those two names; the default
// output path becomes `dist/py2js.bare.browser.js` so the demo build in
// `dist/py2js.browser.js` is not clobbered.
//
// Env overrides:
//   PY2JS_BROWSER_OUT  output file (default dist/py2js.browser.js, or
//                      dist/py2js.bare.browser.js with --bare)
//   PY2JS_EXTERNAL     "bare" for bare specifiers (same as --bare)
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

const bare =
  process.argv.includes("--bare") || process.env.PY2JS_EXTERNAL === "bare";
const externals = bare
  ? { numpy: "numpy-ts", wanim: "wanim" }
  : { numpy: "./numpy-ts.browser.js", wanim: "./wanim.browser.js" };
const defaultOut = resolve(wanimRoot, "dist", bare ? "py2js.bare.browser.js" : "py2js.browser.js");
const outfile = resolve(process.env.PY2JS_BROWSER_OUT ?? defaultOut);

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
      name: bare ? "shared-bare" : "shared-relative",
      setup(build) {
        build.onResolve({ filter: /^numpy-ts$/ }, () => ({
          path: externals.numpy,
          external: true,
        }));
        build.onResolve({ filter: /dist\/index\.js$/ }, () => ({
          path: externals.wanim,
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
