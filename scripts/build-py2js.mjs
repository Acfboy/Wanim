#!/usr/bin/env node
// Bundle the py2js browser bundles for the demo site and the npm package.
//
// Three entry points (see py2js/src):
//
//   compiler.ts  -> dist/py2js.browser.js      (Python -> JS, no externals)
//   runtime.ts   -> dist/py-runtime.browser.js (rapydscript shims, no externals)
//   browser.ts   -> dist/bridge.browser.js     (compiler + runtime + DOM glue)
//
// The compiler and runtime bundles have no external imports at all: the
// compiler only needs the global `RapydScript` compiler, and the runtime's
// wanim/numpy-ts namespaces are injected through `createRuntimeCore()`.
//
// The bridge imports the two sibling bundles, wanim and numpy-ts.  By default
// those are referenced by *relative* URLs (the bundles all live in `dist/`),
// so the bridge works even when the page's import map is missing or was
// applied too late, and all the files resolve the same URLs, keeping the
// numpy-ts instance (and its WASM heap) shared.  This is the build the demo
// site uses.
//
// With `--bare` (or `PY2JS_EXTERNAL=bare`) a second bridge variant is emitted
// for the published package: `numpy-ts` stays a *bare* specifier for the
// consumer's import map / bundler, while the sibling bundles stay relative
// (the package ships them side by side in `browser/`).  The default output
// becomes `dist-pkg/bridge.browser.js` so the demo build is not clobbered.
//
// Env overrides:
//   PY2JS_COMPILER_OUT   compiler bundle (default dist/py2js.browser.js)
//   PY2JS_RUNTIME_OUT    runtime bundle (default dist/py-runtime.browser.js)
//   PY2JS_BRIDGE_OUT     bridge bundle (default dist/bridge.browser.js, or
//                        dist-pkg/bridge.browser.js with --bare)
//   PY2JS_EXTERNAL       "bare" for the package bridge variant (same as --bare)
//   WANIM_MINIFY         "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP      "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const srcDir = resolve(wanimRoot, "py2js", "src");

if (!existsSync(resolve(wanimRoot, "dist", "index.js"))) {
  console.error("wanim is not built; run `pnpm build` first.");
  process.exit(1);
}

const bare =
  process.argv.includes("--bare") || process.env.PY2JS_EXTERNAL === "bare";

const minify = process.env.WANIM_MINIFY !== "0";
const sourcemap = process.env.WANIM_SOURCEMAP === "1";
const banner = {
  js: "if (typeof globalThis.process === 'undefined') { globalThis.process = { cwd: () => '/', env: {}, argv: [], platform: 'browser' }; }",
};

const shared = {
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome110"],
  sourcemap,
  minify,
  logLevel: "info",
  banner,
  define: { "process.env.NODE_ENV": '"production"' },
};

async function buildSimple(entry, outfile) {
  await build({ ...shared, entryPoints: [resolve(srcDir, entry)], outfile });
  console.log(`wanim: wrote ${outfile}`);
}

async function buildBridge(outfile) {
  const externals = {
    numpy: bare ? "numpy-ts" : "./numpy-ts.browser.js",
    wanim: "./wanim.browser.js",
    compiler: "./py2js.browser.js",
    runtime: "./py-runtime.browser.js",
  };
  await build({
    ...shared,
    entryPoints: [resolve(srcDir, "browser.ts")],
    outfile,
    plugins: [
      {
        name: bare ? "bridge-bare" : "bridge-relative",
        setup(build) {
          build.onResolve({ filter: /^numpy-ts$/ }, () => ({
            path: externals.numpy,
            external: true,
          }));
          build.onResolve({ filter: /dist\/index\.js$/ }, () => ({
            path: externals.wanim,
            external: true,
          }));
          build.onResolve({ filter: /^\.\/compiler\.js$/ }, () => ({
            path: externals.compiler,
            external: true,
          }));
          build.onResolve({ filter: /^\.\/runtime\.js$/ }, () => ({
            path: externals.runtime,
            external: true,
          }));
        },
      },
    ],
  });
  console.log(`wanim: wrote ${outfile}`);
}

if (bare) {
  const outfile = resolve(
    process.env.PY2JS_BRIDGE_OUT ?? resolve(wanimRoot, "dist-pkg", "bridge.browser.js"),
  );
  await buildBridge(outfile);
} else {
  const compilerOut = resolve(
    process.env.PY2JS_COMPILER_OUT ?? resolve(wanimRoot, "dist", "py2js.browser.js"),
  );
  const runtimeOut = resolve(
    process.env.PY2JS_RUNTIME_OUT ?? resolve(wanimRoot, "dist", "py-runtime.browser.js"),
  );
  const bridgeOut = resolve(
    process.env.PY2JS_BRIDGE_OUT ?? resolve(wanimRoot, "dist", "bridge.browser.js"),
  );
  await buildSimple("compiler.ts", compilerOut);
  await buildSimple("runtime.ts", runtimeOut);
  await buildBridge(bridgeOut);
}
