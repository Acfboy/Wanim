#!/usr/bin/env node
// Produce a browser-loadable rapydscript compiler (`demo/rapydscript.js`) from
// the (py2js-patched) third-party rapydscript-ng checkout.
//
// rapydscript-ng ships a `web-repl-export` command that embeds the compiler,
// baselib and stdlib into a single classic script exposing `RapydScript`.  The
// py2js patch threads the `operator_overloading` output option through
// `tools/export.js` so the browser `compile()` emits `ρσ_op_*` calls.
//
// Output: demo/rapydscript.js (gitignored build artifact).
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");
const repoRoot = resolve(wanimRoot, "..");
const rapydscriptRoot = join(repoRoot, "third-party", "rapydscript-ng");
const outDir = process.env.RAPYDSCRIPT_OUT_DIR
  ? resolve(process.env.RAPYDSCRIPT_OUT_DIR)
  : join(wanimRoot, "demo");
mkdirSync(outDir, { recursive: true });

if (!existsSync(join(rapydscriptRoot, "dev", "compiler.js"))) {
  console.error(
    `rapydscript-ng dev/compiler.js missing; run \`node bin/rapydscript self --complete\` in ${rapydscriptRoot}`,
  );
  process.exit(1);
}

const tmp = join(outDir, ".rapydscript-export");
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp, { recursive: true });

execFileSync(process.execPath, ["bin/rapydscript", "web-repl-export", tmp], {
  cwd: rapydscriptRoot,
  stdio: "inherit",
});

copyFileSync(join(tmp, "rapydscript.js"), join(outDir, "rapydscript.js"));
rmSync(tmp, { recursive: true, force: true });
console.log(`wanim: wrote ${join(outDir, "rapydscript.js")}`);
