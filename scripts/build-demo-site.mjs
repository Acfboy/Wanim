#!/usr/bin/env node
// Produce a small, deployable copy of the Python editor demo in `demo-site/`.
//
//   pnpm build:demo-site
//
// The directory is self-contained except for two third-party libraries that are
// referenced from official CDNs (so they never cost the deployer traffic):
//
//   * numpy-ts  — https://cdn.jsdelivr.net/npm/numpy-ts@1.7.0/...
//   * Prism.js  — https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/...
//
// rapydscript is *modified* for wanim (operator overloading + auto-await for
// `self.play`), so it cannot come from a CDN and is shipped here.
//
// Contents written:
//   demo-site/index.html
//   demo-site/style.css
//   demo-site/demo.js
//   demo-site/wanim.browser.js      (minified, no sourcemap)
//   demo-site/py2js.browser.js    (minified, no sourcemap)
//   demo-site/rapydscript.js        (patched compiler)
//   demo-site/fonts/DejaVuSans.ttf  (for Text/MarkupText)
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const demoDir = join(wanimRoot, "demo");
const outDir = join(wanimRoot, "..", "demo-site");

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
const run = (args, env = {}) =>
  execFileSync(pnpm, args, {
    cwd: wanimRoot,
    stdio: "inherit",
    env: { ...process.env, WANIM_MINIFY: "1", WANIM_SOURCEMAP: "0", ...env },
  });

// Ensure dist/ is fresh, then produce the minified browser bundles.
run(["build"]);
run(["exec", "node", "scripts/build-browser.mjs"], {
  WANIM_BROWSER_OUT: join(outDir, "wanim.browser.js"),
});
run(["exec", "node", "scripts/build-editor.mjs"], {
  WANIMPY_BROWSER_OUT: join(outDir, "py2js.browser.js"),
});
run(["exec", "node", "scripts/build-rapydscript-browser.mjs"], {
  RAPYDSCRIPT_OUT_DIR: outDir,
});

// Copy the page assets and point the import map at the CDN copies of numpy-ts.
for (const file of ["style.css", "demo.js"]) {
  copyFileSync(join(demoDir, file), join(outDir, file));
}
const html = readFileSync(join(demoDir, "index.html"), "utf-8").replace(
  "../node_modules/numpy-ts/dist/numpy-ts.browser.js",
  "https://cdn.jsdelivr.net/npm/numpy-ts@1.7.0/dist/numpy-ts.browser.js",
);
writeFileSync(join(outDir, "index.html"), html);

console.log(`\nwanim: demo site written to ${outDir}`);
console.log("Serve it with any static file server, e.g. `pnpm demo-site`.");
