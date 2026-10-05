#!/usr/bin/env node
// Fetch, patch and build rapydscript-ng, then export the browser compiler.
//
// wanim does not vendor rapydscript-ng: this script clones it from upstream at
// the pinned commit, applies `py2js/patches/rapydscript-ng.patch`
// (operator overloading + auto-`await` for `self.play`) and builds the compiler
// with its own bootstrap (`node bin/rapydscript self --complete`).  The browser
// build (`web-repl-export`) is copied to the demo site as `rapydscript.js`.
//
// Env overrides:
//   RAPYDSCRIPT_DIR       checkout cache (default: <wanim>/.rapydscript-ng)
//   RAPYDSCRIPT_OUT_DIR   where to write rapydscript.js (default: <wanim>/dist)
//   RAPYDSCRIPT_FORCE     "1" to re-clone from scratch
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, copyFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");

const RAPYDSCRIPT_REPO = "https://github.com/kovidgoyal/rapydscript-ng.git";
// The upstream commit wanim's patch is written against (rapydscript-ng 0.8.6).
const RAPYDSCRIPT_COMMIT = "818f18787c43c33ff00d7162fcd5aacc1186c6df";
const PATCH = resolve(wanimRoot, "py2js", "patches", "rapydscript-ng.patch");

const cacheDir = resolve(process.env.RAPYDSCRIPT_DIR ?? join(wanimRoot, ".rapydscript-ng"));
const outDir = resolve(process.env.RAPYDSCRIPT_OUT_DIR ?? join(wanimRoot, "dist"));

function git(args) {
  execFileSync("git", ["-C", cacheDir, ...args], { stdio: "inherit" });
}

function ensureCheckout() {
  if (process.env.RAPYDSCRIPT_FORCE === "1") {
    rmSync(cacheDir, { recursive: true, force: true });
  }
  if (!existsSync(join(cacheDir, ".git"))) {
    console.log(`wanim: cloning rapydscript-ng -> ${cacheDir}`);
    rmSync(cacheDir, { recursive: true, force: true });
    execFileSync("git", ["clone", RAPYDSCRIPT_REPO, cacheDir], { stdio: "inherit" });
  }
  git(["fetch", "--quiet", "origin", RAPYDSCRIPT_COMMIT]);
  git(["checkout", "--force", "--detach", RAPYDSCRIPT_COMMIT]);
  // Drop the previous patch (if any) so it can be applied cleanly every time.
  git(["clean", "-fdq", "src", "tools"]);
}

function applyPatch() {
  try {
    execFileSync("git", ["-C", cacheDir, "apply", "--check", PATCH], { stdio: "pipe" });
  } catch {
    // Already applied (checked out at the patched state).
    verifyPatched();
    return;
  }
  git(["apply", PATCH]);
  verifyPatched();
}

function verifyPatched() {
  // `git apply --check` failing used to be read as "already applied", so a
  // stale/broken patch silently produced a *pristine* compiler (no operator
  // overloading, no auto-await, no baselib fixes) and every scene regressed
  // at runtime. Fail loudly instead when the patch markers are missing.
  const markers = [
    ["src/output/operators.pyj", "op_overload_map"],
    ["src/baselib-builtins.pyj", "int(True)"],
  ];
  const missing = markers.filter(([file, text]) => {
    try {
      return !readFileSync(join(cacheDir, file), "utf8").includes(text);
    } catch {
      return true;
    }
  });
  if (missing.length > 0) {
    throw new Error(
      `wanim: rapydscript patch did not apply (${missing.map(([f]) => f).join(", ")}); ` +
        `refusing to build a pristine compiler. Fix ${PATCH}.`,
    );
  }
}

function buildCompiler() {
  console.log("wanim: building rapydscript compiler (self --complete)");
  execFileSync(process.execPath, ["bin/rapydscript", "self", "--complete"], {
    cwd: cacheDir,
    stdio: "inherit",
  });
}

function exportBrowserCompiler() {
  const tmp = join(cacheDir, ".wanim-export");
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  console.log("wanim: exporting rapydscript browser compiler (web-repl-export)");
  execFileSync(process.execPath, ["bin/rapydscript", "web-repl-export", tmp], {
    cwd: cacheDir,
    stdio: "inherit",
  });
  mkdirSync(outDir, { recursive: true });
  const out = join(outDir, "rapydscript.js");
  copyFileSync(join(tmp, "rapydscript.js"), out);
  rmSync(tmp, { recursive: true, force: true });
  console.log(`wanim: wrote ${out}`);
}

ensureCheckout();
applyPatch();
buildCompiler();
exportBrowserCompiler();
