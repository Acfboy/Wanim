// Regression test: tuple values inside Python dict literals must compile to
// arrays, not comma expressions.
//
// `{"value_range": (0, 15)}` used to compile the value to `0, 15` (a JS comma
// expression), so the dict ended up holding only the last element (`15`) and
// any consumer reading `value_range[0]` saw `undefined`.  The fix wraps an
// AST_Seq dict value in an AST_Array in rapydscript's `object_()` parser (see
// py2js/patches/rapydscript-ng.patch).
//
// Run with:  pnpm test:py2js   (needs `pnpm build:rapydscript` first).
// Skips with a notice when the browser compiler bundle is not built, so the
// rest of the node test suite stays runnable without a rapydscript checkout.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");
const rapydscriptPath = resolve(root, "dist", "rapydscript.js");

if (!existsSync(rapydscriptPath)) {
  console.log("skip - dist/rapydscript.js not built (run pnpm build:rapydscript)");
  process.exit(0);
}

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

// Load the patched browser compiler into a plain sandbox; `bare: false` output
// carries rapydscript's baselib, so it runs standalone.
const sandbox = {
  console: { log() {}, warn() {}, error() {} },
  self: null,
  window: null,
  navigator: { userAgent: "node" },
  setTimeout,
  clearTimeout,
};
sandbox.self = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync(rapydscriptPath, "utf8"), sandbox);

async function runPython(source) {
  const js = await sandbox.RapydScript.compile(source, "test.py", { bare: false, beautify: true });
  const lines = [];
  const previous = sandbox.console;
  sandbox.console = { log: (...args) => lines.push(args.join(" ")), warn() {}, error() {} };
  try {
    vm.runInContext(js, sandbox);
  } finally {
    sandbox.console = previous;
  }
  return lines;
}

await check("tuple values in dict literals compile to arrays", async () => {
  const lines = await runPython(
    [
      'd = {"v": (-15, 15)}',
      'e = {"v": (0, 15), "w": (1, 2)}',
      'print(len(d["v"]))',
      'print(d["v"][0])',
      'print(d["v"][1])',
      'print(len(e["v"]))',
      'print(e["v"][0])',
      'print(e["w"][1])',
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["2", "-15", "15", "2", "0", "2"]);
});

await check("nested tuple in a dict value stays a single element", async () => {
  const lines = await runPython(
    [
      'n = {"k": ((1, 2), 3)}',
      'print(len(n["k"]))',
      'print(n["k"][0][1])',
      'print(n["k"][1])',
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["2", "2", "3"]);
});

await check("keyword-argument tuples still compile to arrays", async () => {
  const lines = await runPython(
    [
      "def f(value_range=(0, 1)):",
      "    return value_range[1]",
      "print(f(value_range=(-4, 9)))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["9"]);
});

console.log(`\n${passed} passed`);
