#!/usr/bin/env node
// Assemble the publishable npm package into `packages-out/wanim/` (gitignored).
//
//   packages-out/wanim/   `wanimjs`: the tsc dist (core + Node entries),
//                         the zero-config browser bundles and the runtime
//                         config/template resources, with subpath exports for
//                         the py2js compiler, the py runtime and the bridge.
//
// The package metadata (package.json, README.md, LICENSE) lives in
// `packaging/wanim/`; this script builds the artifacts and copies the two
// together.  Generated output is never committed.
//
// The browser/ directory ships no numpy-ts and no MathJax code: the package
// variant of wanim.browser.js keeps both as bare externals (MathJax stays a
// lazy glue chunk with `import "mathjax-full/..."`), and the bridge keeps
// numpy-ts bare too.  Consumers provide them through an import map or a
// bundler; `mathjax-full`, `numpy-ts` and `opentype.js` are regular
// dependencies of the package.
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
const distPkg = resolve(root, "dist-pkg");
const skipBuild = process.argv.includes("--skip-build");
const doPack = process.argv.includes("--pack");
const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
const node = process.execPath;

/** Artifacts the package needs from `dist/` (checked before assembling). */
const REQUIRED_DIST = [
  "index.js",
  "index.d.ts",
  "wanim.browser.js",
  "bridge.browser.js",
  "py2js.browser.js",
  "py-runtime.browser.js",
  "rapydscript.js",
  "wanim-element.browser.js",
  "chunks",
];

/** Artifacts the package needs from `dist-pkg/` (bare/external variants). */
const REQUIRED_DIST_PKG = ["wanim.browser.js", "bridge.browser.js"];

function run(command, args, options = {}) {
  execFileSync(command, args, { cwd: root, stdio: "inherit", ...options });
}

function buildDist() {
  console.log(
    "wanim: building dist/ (tsc + browser + rapydscript + py2js + element + package variants)",
  );
  run(pnpm, ["build"]);
  // Dev bundles: the demo and other local consumers serving `dist/` rely on
  // the relative numpy-ts URL and the full on-demand MathJax chunks.
  run(node, ["scripts/build-browser.mjs"]);
  run(node, ["scripts/build-rapydscript-browser.mjs"]);
  run(node, ["scripts/build-py2js.mjs"]);
  run(node, ["scripts/build-element.mjs"]);
  // Package variants: no numpy-ts / MathJax code in the output.
  rmSync(distPkg, { recursive: true, force: true });
  run(node, ["scripts/build-browser.mjs"], {
    env: {
      ...process.env,
      WANIM_NUMPY: "bare",
      WANIM_MATHJAX: "external",
      WANIM_BROWSER_OUT: resolve(distPkg, "wanim.browser.js"),
    },
  });
  run(node, ["scripts/build-py2js.mjs", "--bare"]);
}

function checkDist() {
  const missing = [
    ...REQUIRED_DIST.filter((name) => !existsSync(resolve(dist, name))),
    ...REQUIRED_DIST_PKG.filter((name) => !existsSync(resolve(distPkg, name))),
  ];
  if (missing.length > 0) {
    console.error(
      `wanim: build output is incomplete (missing ${missing.join(", ")}); ` +
        "run `pnpm build:packages` without --skip-build first.",
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
  "bridge.browser.js",
  "py2js.browser.js",
  "py2js.bare.browser.js",
  "py-runtime.browser.js",
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

/**
 * Compile the py2js Node entries with tsc and lay the output out under the
 * published names:
 *
 *   compiler_node.js/.d.ts  -> py2js.js/.d.ts
 *   runtime_node.js/.d.ts   -> py-runtime.js/.d.ts
 *   browser.d.ts            -> bridge.d.ts (import of the repo-internal
 *                              `../../dist/index.js` rewritten to the package)
 *   browser.js              -> dropped (the browser bundle supersedes it)
 *
 * The remaining py2js modules (preprocess, compile_error, runtime_core, the
 * python facades, ...) are copied as-is: the renamed entries import them at
 * runtime, and the `.d.ts` files resolve `./compiler.js` / `./runtime.js`
 * through their declarations.
 */
function copyPy2jsDist(target) {
  const tmp = resolve(outRoot, ".py2js-tsc");
  rmSync(tmp, { recursive: true, force: true });
  run(node, [
    "node_modules/typescript/bin/tsc",
    "py2js/src/compiler_node.ts",
    "py2js/src/runtime_node.ts",
    "py2js/src/browser.ts",
    "--outDir", tmp,
    "--target", "ES2022",
    "--module", "ESNext",
    "--moduleResolution", "Bundler",
    "--lib", "ES2022,DOM",
    "--skipLibCheck",
    "--strict",
    "--declaration",
  ]);
  for (const name of readdirSync(tmp)) {
    const from = join(tmp, name);
    if (!statSync(from).isFile()) continue;
    if (name === "browser.js") continue;
    if (name === "browser.d.ts") {
      const rewritten = readFileSync(from, "utf8").replace(
        '"../../dist/index.js"',
        '"wanimjs"',
      );
      writeFileSync(join(target, "bridge.d.ts"), rewritten);
      continue;
    }
    const renamed = {
      "compiler_node.js": "py2js.js",
      "compiler_node.d.ts": "py2js.d.ts",
      "runtime_node.js": "py-runtime.js",
      "runtime_node.d.ts": "py-runtime.d.ts",
    }[name] ?? name;
    copyFile(from, join(target, renamed));
  }
  rmSync(tmp, { recursive: true, force: true });
}

function assemble() {
  const pkg = copyStatic("wanim");
  // Browser bundles.  The package variants keep numpy-ts / MathJax external,
  // so no upstream code is shipped; `chunks/` only holds wanim's small
  // `mathjax-full` glue chunks.
  const browserDir = resolve(pkg, "browser");
  copyFile(resolve(distPkg, "wanim.browser.js"), join(browserDir, "wanim.browser.js"));
  cpSync(resolve(distPkg, "chunks"), join(browserDir, "chunks"), { recursive: true });
  copyFile(resolve(dist, "py2js.browser.js"), join(browserDir, "py2js.browser.js"));
  copyFile(resolve(dist, "py-runtime.browser.js"), join(browserDir, "py-runtime.browser.js"));
  copyFile(resolve(distPkg, "bridge.browser.js"), join(browserDir, "bridge.browser.js"));
  copyFile(resolve(dist, "rapydscript.js"), join(browserDir, "rapydscript.js"));
  copyFile(
    resolve(dist, "wanim-element.browser.js"),
    join(browserDir, "wanim-element.browser.js"),
  );
  // Node entries (tsc) plus the py2js declarations.
  const pkgDist = resolve(pkg, "dist");
  copyDistTree(pkgDist);
  copyPy2jsDist(pkgDist);
  // Runtime resources the tsc output reads by relative path (see
  // src/_config/utils.ts and src/utils/file_ops.ts).
  copyFile(resolve(root, "src/_config/default.cfg"), resolve(pkg, "src/_config/default.cfg"));
  cpSync(resolve(root, "src/templates"), resolve(pkg, "src/templates"), { recursive: true });
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

const packages = ["wanim"];

if (skipBuild) {
  console.log("wanim: --skip-build, reusing the current dist/");
} else {
  buildDist();
}
checkDist();

rmSync(outRoot, { recursive: true, force: true });
mkdirSync(outRoot, { recursive: true });
const pkg = assemble();

console.log(`wanim: ${packages[0]} -> ${pkg} (${formatBytes(dirSize(pkg))})`);

if (doPack) {
  packAll(packages);
  console.log(`wanim: tarballs in ${resolve(outRoot, "tarballs")}`);
}
