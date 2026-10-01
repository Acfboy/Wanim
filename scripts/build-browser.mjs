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
// Fonts are shipped in `assets/fonts` and copied next to the bundle
// (`<bundle dir>/fonts/`), because the text pipeline resolves its default font
// chain relative to `import.meta.url`.  The chain is Times New Roman (Latin)
// plus FZShuSong (CJK), with per-glyph fallback.
//
// Env overrides:
//   WANIM_BROWSER_OUT  output file (default dist/wanim.browser.js)
//   WANIM_MINIFY       "0" to disable minification (default: minify)
//   WANIM_SOURCEMAP    "1" to emit a sourcemap (default: none)
import { build } from "esbuild";
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const shim = (name) => resolve(here, "shims", name);

const outfile = resolve(process.env.WANIM_BROWSER_OUT ?? resolve(root, "dist", "wanim.browser.js"));
const minify = process.env.WANIM_MINIFY !== "0";
const sourcemap = process.env.WANIM_SOURCEMAP === "1";

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
  external: ["numpy-ts"],
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

const fontSource = resolve(root, "assets", "fonts");
if (!existsSync(fontSource)) {
  throw new Error(`wanim: shipped fonts are missing at ${fontSource}`);
}
const fontDest = resolve(dirname(outfile), "fonts");
mkdirSync(fontDest, { recursive: true });
for (const name of readdirSync(fontSource)) {
  if (!/\.(ttf|otf)$/i.test(name)) continue;
  cpSync(resolve(fontSource, name), resolve(fontDest, name));
  console.log(`wanim: copied ${name} -> ${resolve(fontDest, name)}`);
}

console.log(`wanim: wrote ${outfile}`);
