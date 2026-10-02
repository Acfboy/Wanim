// Plain-node test for py2js/src/runtime_core.ts (Python builtins shims).
//
// Run with:  pnpm test:py2js
import assert from "node:assert/strict";

import { createRuntimeCore } from "../dist/runtime_core.js";

class FakeNDArray {}
const core = createRuntimeCore({
  wanim: {},
  np: { round: (x) => x },
  NDArray: FakeNDArray,
});

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

check("round() rounds halves to even (CPython semantics)", () => {
  assert.equal(core.pythonRound(2.5), 2);
  assert.equal(core.pythonRound(3.5), 4);
  assert.equal(core.pythonRound(-2.5), -2);
  assert.equal(core.pythonRound(0.5), 0);
  assert.equal(core.pythonRound(1.5), 2);
});

check("round() accepts ndigits", () => {
  assert.equal(core.pythonRound(3.14159, 2), 3.14);
  assert.equal(core.pythonRound(2.675, 2), 2.67);
  assert.equal(core.pythonRound(1.005, 2), 1.0);
});

check("round() handles negative ndigits", () => {
  assert.equal(core.pythonRound(150, -2), 200);
  assert.equal(core.pythonRound(250, -2), 200);
  assert.equal(core.pythonRound(350, -2), 400);
});

check("round() passes non-finite values through", () => {
  assert.equal(core.pythonRound(Infinity), Infinity);
  assert.ok(Number.isNaN(core.pythonRound(NaN)));
});

check("numeric % still is modulo", () => {
  assert.equal(core.pythonMod(7, 3), 1);
});

check("np.round accepts Python scalars (numpy-ts does not)", () => {
  const np = core.buildNumpy();
  assert.equal(np.round(2.5), 2);
  assert.equal(np.round(2.675, 2), 2.67);
  assert.equal(np.round(-2.675, 2), -2.67);
  assert.equal(np.around(3.14159, 3), 3.142);
  assert.deepEqual(np.round([1.25, 1.35], 1), [1.2, 1.4]);
});

check("% formatting with a tuple of values", () => {
  assert.equal(core.pythonMod("%s / %s / %d", [5, 2.5, 3.9]), "5 / 2.5 / 3");
});

check("% formatting with a single value", () => {
  assert.equal(core.pythonMod("%.2f", 2.5), "2.50");
  assert.equal(core.pythonMod("%5.1f", 3.14), "  3.1");
});

check("% formatting flags and width", () => {
  assert.equal(core.pythonMod("%-5d|", 42), "42   |");
  assert.equal(core.pythonMod("%05d", 42), "00042");
  assert.equal(core.pythonMod("%+d", 42), "+42");
});

check("% formatting bases", () => {
  assert.equal(core.pythonMod("%x", 255), "ff");
  assert.equal(core.pythonMod("%X", 255), "FF");
  assert.equal(core.pythonMod("%#o", 8), "0o10");
});

check("% formatting mapping form", () => {
  assert.equal(core.pythonMod("%(name)s=%(value)d", { name: "x", value: 1 }), "x=1");
});

check("% formatting exponent and percent", () => {
  assert.equal(core.pythonMod("%e", 12345.678), "1.234568e+04");
  assert.equal(core.pythonMod("%s%%", "done"), "done%");
  assert.equal(core.pythonMod("%s", null), "None");
});

check("wanFlatten1 concatenates one level", () => {
  assert.deepEqual(core.wanFlatten1([[1, 2], [3], []]), [1, 2, 3]);
  assert.deepEqual(core.wanFlatten1([]), []);
  assert.deepEqual(core.wanFlatten1([["a"], ["b", "c"]]), ["a", "b", "c"]);
});

check("wanFlatten1 is exposed to compiled scenes as a global", () => {
  const runtime = core.buildRuntime();
  assert.equal(typeof runtime.globals.__wan_flatten1, "function");
  assert.deepEqual(runtime.globals.__wan_flatten1([[1], [2, 3]]), [1, 2, 3]);
});

console.log(`\n${passed} tests passed`);
