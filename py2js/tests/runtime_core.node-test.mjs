// Plain-node test for py2js/src/runtime_core.ts (Python builtins shims).
//
// Run with:  pnpm test:py2js
import assert from "node:assert/strict";

import * as npReal from "numpy-ts";

import { createRuntimeCore } from "../dist/runtime_core.js";

class FakeNDArray {}
const core = createRuntimeCore({
  wanim: {},
  np: { round: (x) => x },
  NDArray: FakeNDArray,
});

const ndCore = createRuntimeCore({
  wanim: {},
  np: npReal,
  NDArray: npReal.NDArray,
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

check("wanSlice builds numpy slice strings", () => {
  assert.equal(ndCore.wanSlice(null, null), ":");
  assert.equal(ndCore.wanSlice(1, 2), "1:2");
  assert.equal(ndCore.wanSlice(null, null, -1), "::-1");
  assert.equal(ndCore.wanSlice(0, -1), "0:-1");
});

check("wanGetItem slices NDArray columns (pts[:, 1])", () => {
  const pts = npReal.asarray([[1, 2, 3], [4, 5, 6]]);
  const col = ndCore.wanGetItem(pts, ndCore.wanSlice(null, null), 1);
  assert.deepEqual(col.toArray(), [2, 5]);
  assert.deepEqual(
    ndCore.wanGetItem(pts, "0:2", "1:3").toArray(),
    [[2, 3], [5, 6]],
  );
  assert.deepEqual(ndCore.wanGetItem(pts, "...", 0).toArray(), [1, 4]);
  assert.equal(ndCore.wanGetItem(pts, 0, 1).toArray(), 2);
});

check("wanGetItem descends nested plain lists numpy-style", () => {
  const grid = [[1, 2, 3], [4, 5, 6]];
  assert.deepEqual(ndCore.wanGetItem(grid, ":", 1), [2, 5]);
  assert.deepEqual(ndCore.wanGetItem(grid, 0, "1:"), [2, 3]);
  assert.equal(ndCore.wanGetItem(grid, 1, -1), 6);
  assert.deepEqual(ndCore.wanGetItem(grid, "::-1", ":"), [[4, 5, 6], [1, 2, 3]]);
});

check("wanGetItem helpers are exposed to compiled scenes as globals", () => {
  const runtime = ndCore.buildRuntime();
  assert.equal(typeof runtime.globals.__wan_getitem, "function");
  assert.equal(typeof runtime.globals.__wan_slice, "function");
  assert.equal(runtime.globals.__wan_slice(null, null), ":");
});

check("NDArray.astype accepts Python types and string aliases", () => {
  ndCore.buildRuntime(); // applies the NDArray prototype patch
  function ρσ_float(v) { return v; }
  function ρσ_int(v) { return v; }
  const fromType = npReal.arange(5).astype(ρσ_float);
  assert.equal(fromType.dtype, "float64");
  assert.deepEqual(fromType.toArray(), [0, 1, 2, 3, 4]);
  assert.equal(npReal.arange(5).astype(ρσ_int).dtype, "int64");
  assert.equal(npReal.arange(5).astype("float").dtype, "float64");
  assert.equal(npReal.arange(5).astype("float64").dtype, "float64");
});

check("wanMatmul multiplies matrices and matrix-vector (rot @ center)", () => {
  const rot = npReal.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
  const center = [3, 0, 0];
  assert.deepEqual(ndCore.wanMatmul(rot, center).toArray(), [0, 3, 0]);
  assert.deepEqual(
    ndCore.wanMatmul(rot, rot).toArray(),
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
  );
});

check("wanMatmul accepts plain nested lists and inner products", () => {
  assert.deepEqual(
    ndCore.wanMatmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]).toArray(),
    [[19, 22], [43, 50]],
  );
  assert.equal(ndCore.wanMatmul([1, 2, 3], [4, 5, 6]), 32);
});

check("wanMatmul prefers __matmul__/__rmatmul__ dunders", () => {
  const left = { __matmul__(other) { return ["left", other]; } };
  assert.deepEqual(ndCore.wanMatmul(left, 7), ["left", 7]);
  const right = { __rmatmul__(other) { return ["right", other]; } };
  assert.deepEqual(ndCore.wanMatmul(7, right), ["right", 7]);
});

check("wanMatmul is exposed to compiled scenes as a global", () => {
  const runtime = ndCore.buildRuntime();
  assert.equal(typeof runtime.globals.__wan_matmul, "function");
});

check("NDArray numeric slice serves plain-JS copies (ρσ_eslice path)", () => {
  ndCore.buildRuntime();
  const arr = npReal.arange(5).astype("float64");
  // `ρσ_eslice(arr, -1)` does `arr.slice().reverse()` then numeric
  // `arr.slice(start, end).filter(...)`: every step must work on NDArrays.
  assert.deepEqual([...arr.slice()].map(Number), [0, 1, 2, 3, 4]);
  assert.deepEqual([...arr.slice().reverse()].map(Number), [4, 3, 2, 1, 0]);
  assert.deepEqual([...arr.slice(1, 3)].map(Number), [1, 2]);
  assert.deepEqual(
    arr.slice().reverse().slice(0, 5).filter((_, i) => i % 1 === 0).map(Number),
    [4, 3, 2, 1, 0],
  );
  // String specs still go through numpy slicing and stay NDArrays.
  assert.deepEqual(arr.slice("1:3").toArray(), [1, 2]);
  assert.deepEqual(arr.slice("::-1").toArray(), [4, 3, 2, 1, 0]);
});

check("wanReversed reverses arrays without mutating, strings, and iterables", () => {
  const arr = [1, 2, 3];
  assert.deepEqual(core.wanReversed(arr), [3, 2, 1]);
  assert.deepEqual(arr, [1, 2, 3]);
  assert.deepEqual(core.wanReversed("abc"), ["c", "b", "a"]);
  // Any Symbol.iterator object (e.g. a VGroup/Mobject) reverses in iteration order.
  const seq = { *[Symbol.iterator]() { yield "a"; yield "b"; yield "c"; } };
  assert.deepEqual(core.wanReversed(seq), ["c", "b", "a"]);
  assert.deepEqual(core.wanReversed(new Map([["k1", 1], ["k2", 2]])), ["k2", "k1"]);
});

check("wanReversed prefers __reversed__ and falls back to the sequence protocol", () => {
  const custom = { __reversed__() { return ["x", "y"]; } };
  assert.deepEqual(core.wanReversed(custom), ["x", "y"]);
  const gen = { __reversed__() { return (function* () { yield 1; yield 2; })(); } };
  assert.deepEqual(core.wanReversed(gen), [1, 2]);
  // No iterator: __len__ + __getitem__ (negative indices must not occur here).
  const seqLike = { __len__() { return 3; }, __getitem__(i) { return ["a", "b", "c"][i]; } };
  assert.deepEqual(core.wanReversed(seqLike), ["c", "b", "a"]);
  assert.throws(() => core.wanReversed(42), TypeError);
  assert.throws(() => core.wanReversed(null), TypeError);
});

check("wanReversed tolerates Mobject-like Proxies that throw on missing attrs", () => {
  // Mirrors `createMobjectProxy`: absent attributes raise AttributeError
  // instead of yielding `undefined`, so dunders must be probed with `in`.
  const target = {
    *[Symbol.iterator]() { yield "a"; yield "b"; },
    __len__() { return 2; },
    __getitem__(i) { return ["a", "b"][i]; },
  };
  const proxy = new Proxy(target, {
    get(obj, prop, receiver) {
      if (prop in obj || typeof prop === "symbol") return Reflect.get(obj, prop, receiver);
      const error = new Error(`VGroup object has no attribute '${String(prop)}'`);
      error.name = "AttributeError";
      throw error;
    },
  });
  assert.deepEqual(core.wanReversed(proxy), ["b", "a"]);
});

check("wanReversed is exposed to compiled scenes as a global", () => {
  const runtime = core.buildRuntime();
  assert.equal(typeof runtime.globals.__wan_reversed, "function");
});

console.log(`\n${passed} tests passed`);
