#!/usr/bin/env node
// Bundle the py2js browser runtime for the editor demo.
//
// The demo page compiles Python to JS with the browser rapydscript compiler
// (`demo/rapydscript.js`, built by `build-rapydscript-browser.mjs`) and then
// runs the result against the wanim Canvas2D backend using the runtime shims in
// `py2js/src/browser.ts`.
//
// `wanim` and `numpy-ts` are kept external so the page's import map resolves the
// single shared instances loaded by the wanim browser bundle.
import { build } from "esbuild";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const repoRoot = resolve(wanimRoot, "..");
const entry = resolve(repoRoot, "py2js", "src", "browser.ts");

if (!existsSync(resolve(wanimRoot, "dist", "index.js"))) {
  console.error("wanim is not built; run `pnpm build` first.");
  process.exit(1);
}

await build({
  entryPoints: [entry],
  outfile: resolve(wanimRoot, "demo", "py2js.browser.js"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome110"],
  sourcemap: true,
  logLevel: "info",
  // Keep the shared singletons external; the page import map points them at the
  // same URLs the wanim bundle uses.
  external: ["numpy-ts"],
  plugins: [
    {
      name: "wanim-shared",
      setup(build) {
        build.onResolve({ filter: /wanim\/dist\/index\.js$/ }, () => ({
          path: "wanim",
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

console.log("wanim: wrote demo/py2js.browser.js");
