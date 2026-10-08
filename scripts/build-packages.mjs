#!/usr/bin/env node
// Assemble the publishable npm packages into `packages-out/` (gitignored).
//
//   packages-out/wanim-core/  the browser library: tsc dist + browser bundles
//                             + the runtime config/template resources
//   packages-out/wanim-py/    the py2js bridge: bare-specifier browser build,
//                             patched rapydscript compiler, Node toolbox
//   packages-out/wanim/       the combined zero-config package (relative
//                             imports, everything in one browser/ directory)
//
// The per-package metadata (package.json, README.md, LICENSE) lives in
// `packaging/<name>/`; this script builds the artifacts and copies the two
// together.  Generated output is never committed.
//
// Usage:
//   node scripts/build-packages.mjs                 # rebuild dist/ + assemble
//   node scripts/build-packages.mjs --skip-build    # reuse the current dist/
//   node scripts/build-packages.mjs --pack          # also npm pack tarballs
import { execFileSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const outRoot = resolve(root, "packages-out");
const staticRoot = resolve(root, "packaging");
const dist = resolve(root, "dist");
const skipBuild = process.argv.includes("--skip-build");
const doPack = process.argv.includes("--pack");
const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
const node = process.execPath;

/** Artifacts each package needs from `dist/` (checked before assembling). */
const REQUIRED_DIST = [
  "index.js",
  "index.d.ts",
  "wanim.browser.js",
  "py2js.browser.js",
  "py2js.bare.browser.js",
  "rapydscript.js",
  "numpy-ts.browser.js",
  "wanim-element.browser.js",
  "chunks",
];

function run(command, args, options = {}) {
  execFileSync(command, args, { cwd: root, stdio: "inherit", ...options });
}

function buildDist() {
  console.log("wanim: building dist/ (tsc + browser + py2js + rapydscript + element + py2js bare)");
  run(pnpm, ["build"]);
  run(node, ["scripts/build-browser.mjs"]);
  run(node, ["scripts/build-rapydscript-browser.mjs"]);
  run(node, ["scripts/build-py2js.mjs"]);
  run(node, ["scripts/build-py2js.mjs", "--bare"]);
  run(node, ["scripts/build-element.mjs"]);
}

function checkDist() {
  const missing = REQUIRED_DIST.filter((name) => !existsSync(resolve(dist, name)));
  if (missing.length > 0) {
    console.error(
      `wanim: dist/ is incomplete (missing ${missing.join(", ")}); ` +
        "run `pnpm build:packages` without --skip-build first " +
        "(the bare py2js bundle is not part of `build:demo`).",
    );
    process.exit(1);
  }
}

function copyStatic(name) {
  const target = resolve(outRoot, name);
  cpSync(resolve(staticRoot, name), target, { recursive: true });
  return target;
}

function copyFile(src, dst) {
  mkdirSync(dirname(dst), { recursive: true });
  cpSync(src, dst);
}

/**
 * Copy the tsc output but leave the root-level browser bundles behind: they
 * are packaged under `browser/` instead (with the same names the bundles use
 * for their relative imports).
 */
const BROWSER_ONLY_DIST = new Set([
  "wanim.browser.js",
  "py2js.browser.js",
  "py2js.bare.browser.js",
  "rapydscript.js",
  "numpy-ts.browser.js",
  "wanim-element.browser.js",
  "element.js",
  "element.d.ts",
  "chunks",
]);

function copyDistTree(target) {
  cpSync(dist, target, {
    recursive: true,
    filter: (src) => {
      const rel = relative(dist, src);
      if (rel === "") return true;
      return !BROWSER_ONLY_DIST.has(rel.split(sep)[0]);
    },
  });
}

function copyBrowserBundle(targetDir, { chunks = true } = {}) {
  copyFile(resolve(dist, "wanim.browser.js"), join(targetDir, "wanim.browser.js"));
  copyFile(resolve(dist, "numpy-ts.browser.js"), join(targetDir, "numpy-ts.browser.js"));
  if (chunks) cpSync(resolve(dist, "chunks"), join(targetDir, "chunks"), { recursive: true });
}

function assembleCore() {
  const pkg = copyStatic("wanim-core");
  copyDistTree(resolve(pkg, "dist"));
  copyBrowserBundle(resolve(pkg, "browser"));
  // Runtime resources the tsc output reads by relative path (see
  // src/_config/utils.ts and src/utils/file_ops.ts).
  copyFile(resolve(root, "src/_config/default.cfg"), resolve(pkg, "src/_config/default.cfg"));
  cpSync(resolve(root, "src/templates"), resolve(pkg, "src/templates"), { recursive: true });
  return pkg;
}

function assemblePy() {
  const pkg = copyStatic("wanim-py");
  const browserDir = resolve(pkg, "browser");
  copyFile(resolve(dist, "py2js.bare.browser.js"), join(browserDir, "py2js.browser.js"));
  copyFile(resolve(dist, "rapydscript.js"), join(browserDir, "rapydscript.js"));
  // Node toolbox + browser declarations straight from the TypeScript sources.
  const pyDist = resolve(pkg, "dist");
  run(node, [
    "node_modules/typescript/bin/tsc",
    "py2js/src/node.ts",
    "py2js/src/browser.ts",
    "--outDir", pyDist,
    "--target", "ES2022",
    "--module", "ESNext",
    "--moduleResolution", "Bundler",
    "--lib", "ES2022,DOM",
    "--skipLibCheck",
    "--strict",
    "--declaration",
  ]);
  // The tsc browser.js is superseded by the esbuild bundle (bare externals,
  // minified); keep only its declarations, rewritten to the published names.
  rmSync(join(pyDist, "browser.js"));
  const browserTypes = join(pyDist, "browser.d.ts");
  const rewritten = readFileSync(browserTypes, "utf8").replace(
    '"../../dist/index.js"',
    '"wanim"',
  );
  writeFileSync(browserTypes, rewritten);
  return pkg;
}

function assembleCombined() {
  const pkg = copyStatic("wanim");
  copyBrowserBundle(resolve(pkg, "browser"));
  // The relative-import py2js bundle plus the compiler make this package
  // zero-config: the whole browser/ directory works as served.
  copyFile(resolve(dist, "py2js.browser.js"), resolve(pkg, "browser/py2js.browser.js"));
  copyFile(resolve(dist, "rapydscript.js"), resolve(pkg, "browser/rapydscript.js"));
  // The <wanim-scene> custom element (lazy: py2js/rapydscript load on first run).
  copyFile(
    resolve(dist, "wanim-element.browser.js"),
    resolve(pkg, "browser/wanim-element.browser.js"),
  );
  return pkg;
}

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else yield path;
  }
}

function dirSize(dir) {
  let total = 0;
  for (const entry of walk(dir)) total += statSync(entry).size;
  return total;
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function packAll(packages) {
  const tarballDir = resolve(outRoot, "tarballs");
  mkdirSync(tarballDir, { recursive: true });
  for (const name of packages) {
    const stdout = execFileSync(
      "npm",
      ["pack", "--json", "--pack-destination", tarballDir],
      { cwd: resolve(outRoot, name), encoding: "utf8" },
    );
    const [info] = JSON.parse(stdout);
    console.log(
      `wanim: packed ${info.filename} — ${info.files.length} files, ` +
        `${formatBytes(info.size)} packed / ${formatBytes(info.unpackedSize)} unpacked`,
    );
  }
}

const packages = ["wanim-core", "wanim-py", "wanim"];

if (skipBuild) {
  console.log("wanim: --skip-build, reusing the current dist/");
} else {
  buildDist();
}
checkDist();

rmSync(outRoot, { recursive: true, force: true });
mkdirSync(outRoot, { recursive: true });
assembleCore();
assemblePy();
assembleCombined();

for (const name of packages) {
  const pkg = resolve(outRoot, name);
  console.log(`wanim: ${name} -> ${pkg} (${formatBytes(dirSize(pkg))})`);
}

if (doPack) {
  packAll(packages);
  console.log(`wanim: tarballs in ${resolve(outRoot, "tarballs")}`);
}
