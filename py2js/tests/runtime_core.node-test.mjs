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

check("numpy facade exposes linalg.LinAlgError for except handlers", () => {
  const np = ndCore.buildNumpy();
  assert.equal(typeof np.linalg.LinAlgError, "function");
  const error = new np.linalg.LinAlgError("singular");
  assert.ok(error instanceof Error);
  assert.equal(error.name, "LinAlgError");
});

check("numpy facade accepts rapydscript range() objects as array-likes", () => {
  const np = ndCore.buildNumpy();
  // Mimics rapydscript's ρσ_range: a plain {start, step, stop} bag with
  // Symbol.iterator, not a JS array. `np.prod(range(1, k + 1))` is valid numpy
  // but used to throw `Cannot read properties of undefined (reading 'length')`.
  const fakeRange = (start, stop, step = 1) => {
    const length = Math.max(Math.ceil((stop - start) / step), 0);
    const ans = { start, step, stop };
    ans[Symbol.iterator] = function () {
      let i = start - step;
      let idx = -1;
      const it = {};
      it.next = () => {
        i += step;
        idx += 1;
        if (idx >= length) return { done: true };
        return { done: false, value: i };
      };
      it[Symbol.iterator] = () => it;
      return it;
    };
    ans.__len__ = () => length;
    return ans;
  };
  assert.equal(np.prod(fakeRange(1, 4)), 6);
  assert.equal(np.sum(fakeRange(1, 4)), 6);
  assert.equal(np.mean(fakeRange(1, 4)), 2);
  assert.equal(np.prod(fakeRange(1, 1)), 1); // empty range: identity
  assert.deepEqual(np.array(fakeRange(1, 4)).tolist(), [1, 2, 3]);
  assert.deepEqual(np.asarray([fakeRange(1, 3), fakeRange(1, 3)]).tolist(), [[1, 2], [1, 2]]);
  // Strings stay rejected (they also have Symbol.iterator but are not arrays).
  assert.throws(() => np.prod("ab"));
});

check("python random: seed(7) matches CPython exactly", () => {
  const runtime = ndCore.buildRuntime();
  const random = runtime.globals.random;
  random.seed(7);
  assert.deepEqual(
    [random.random(), random.random(), random.random()],
    [0.32383276483316237, 0.15084917392450192, 0.6509344730398537],
  );
  assert.equal(random.getrandbits(32), 311111475);
});

check("python random: int/str/float seeds match CPython", () => {
  const runtime = ndCore.buildRuntime();
  const random = runtime.globals.random;
  random.seed(12345);
  assert.deepEqual(
    [random.random(), random.random()],
    [0.41661987254534116, 0.010169169457068361],
  );
  random.seed(-7);
  assert.equal(random.random(), 0.32383276483316237); // abs(-7) == 7
  random.seed("hello");
  assert.deepEqual(
    [random.random(), random.random(), random.random()],
    [0.3537754404730722, 0.6631985810268619, 0.5476053663383964],
  );
  random.seed(7.5);
  assert.deepEqual(
    [random.random(), random.random()],
    [0.007428081502091954, 0.8037855946327452],
  );
});

check("python random: detached from-imports keep working", () => {
  const runtime = ndCore.buildRuntime();
  const seed = runtime.globals.random.seed;
  const rand = runtime.globals.random.random;
  seed(7);
  assert.equal(rand(), 0.32383276483316237);
});

check("python random: distributions match CPython (seed 7)", () => {
  const runtime = ndCore.buildRuntime();
  const random = runtime.globals.random;
  random.seed(7);
  assert.deepEqual(
    [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)],
    [6, 3, 7],
  );
  random.seed(7);
  assert.deepEqual(
    [random.uniform(1, 2), random.uniform(1, 2)],
    [1.3238327648331625, 1.150849173924502],
  );
  random.seed(7);
  assert.deepEqual(
    [random.choice([10, 20, 30, 40, 50]), random.choice([10, 20, 30, 40, 50])],
    [30, 20],
  );
  random.seed(7);
  const list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  random.shuffle(list);
  assert.deepEqual(list, [8, 3, 1, 4, 7, 0, 9, 6, 2, 5]);
  random.seed(7);
  assert.deepEqual(
    random.sample(Array.from({ length: 100 }, (_, i) => i), 5),
    [41, 19, 50, 83, 6],
  );
  random.seed(7);
  assert.deepEqual(
    [random.gauss(0, 1), random.gauss(0, 1)],
    [-0.2558802884476004, 0.511431512516514],
  );
});

check("python math: combinatorics match CPython", () => {
  const runtime = ndCore.buildRuntime();
  const math = runtime.globals.math;
  assert.equal(math.comb(3, 1), 3);
  assert.equal(math.comb(3, 5), 0);
  assert.equal(math.comb(10, 5), 252);
  assert.equal(math.perm(3, 1), 3);
  assert.equal(math.perm(5, 0), 1);
  assert.equal(math.factorial(5), 120);
  assert.equal(math.factorial(0), 1);
  assert.equal(math.gcd(12, 8), 4);
  assert.equal(math.gcd(0, 0), 0);
  assert.equal(math.lcm(4, 6), 12);
  assert.equal(math.lcm(), 1);
  assert.equal(math.lcm(0, 5), 0);
  assert.equal(math.isqrt(17), 4);
  assert.throws(() => math.comb(-1, 0), RangeError);
  assert.throws(() => math.factorial(3.5), TypeError);
});

check("python math: floats and constants match JS Math", () => {
  const runtime = ndCore.buildRuntime();
  const math = runtime.globals.math;
  assert.equal(math.pi, Math.PI);
  assert.equal(math.e, Math.E);
  assert.equal(math.tau, Math.PI * 2);
  assert.equal(math.inf, Infinity);
  assert.ok(Number.isNaN(math.nan));
  assert.equal(math.sqrt(16), 4);
  assert.equal(math.sin(0), 0);
  assert.equal(math.log(Math.E), 1);
  assert.equal(math.log(8, 2), 3);
  assert.equal(math.degrees(Math.PI), 180);
  assert.equal(math.radians(180), Math.PI);
  assert.equal(math.hypot(3, 4), 5);
  assert.equal(math.dist([0, 0], [3, 4]), 5);
  assert.equal(math.isclose(1.0, 1.0 + 1e-10), true);
  assert.equal(math.isfinite(1.5), true);
  assert.equal(math.isinf(Infinity), true);
  assert.equal(math.isnan(NaN), true);
  assert.equal(math.floor(3.7), 3);
  assert.equal(math.ceil(3.2), 4);
  assert.equal(math.trunc(-3.7), -3);
  assert.deepEqual(math.modf(3.7), [0.7000000000000002, 3]);
  assert.deepEqual(math.frexp(8), [0.5, 4]);
  assert.equal(math.remainder(5, 3), -1);
  assert.equal(math.fsum([0.1, 0.2, 0.3]), 0.6);
  assert.equal(math.erf(0), 0);
  assert.equal(math.exp(0), 1);
  assert.equal(math.pow(2, 10), 1024);
  // Detached from-imports keep working (closures, like `random`).
  const comb = runtime.globals.math.comb;
  assert.equal(comb(4, 2), 6);
});

check("python random: Random instances and getstate round-trip", () => {
  const runtime = ndCore.buildRuntime();
  const Random = runtime.globals.random.Random;
  const r = Random(42);
  assert.deepEqual(
    [r.random(), r.random(), r.random()],
    [0.6394267984578837, 0.025010755222666936, 0.27502931836911926],
  );
  const random = runtime.globals.random;
  random.seed(5);
  const state = random.getstate();
  const a = [random.random(), random.random()];
  random.setstate(state);
  assert.deepEqual([random.random(), random.random()], a);
});

check("numpy facade stacks plain Python lists (column_stack et al.)", () => {
  const np = ndCore.buildNumpy();
  const rows = (a) => Array.from(a, (r) => [...r]);
  assert.deepEqual(rows(np.column_stack([[1, 2], [3, 4]])), [[1, 3], [2, 4]]);
  assert.deepEqual(rows(np.vstack([[1, 2], [3, 4]])), [[1, 2], [3, 4]]);
  assert.deepEqual([...np.hstack([[1, 2], [3, 4]])], [1, 2, 3, 4]);
});

check("numpy facade gcd/lcm accept Python scalars and lists", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.gcd(4, 6), 2);
  assert.equal(np.gcd(2, 1), 1);
  assert.equal(np.gcd(0, 0), 0);
  assert.equal(np.gcd(-4, 6), 2);
  assert.equal(np.lcm(4, 6), 12);
  assert.equal(np.lcm(0, 5), 0);
  assert.deepEqual(np.gcd([4, 6, 0], 2).toArray(), [2, 2, 2]);
  assert.deepEqual(np.lcm([4, 6], [6, 8]).toArray(), [12, 24]);
  assert.throws(() => np.gcd(4.5, 2), TypeError);
});

check("numpy facade exposes legacy np.random.RandomState", () => {
  const np = ndCore.buildNumpy();
  assert.equal(typeof np.random.RandomState, "function");
  // Bare-call construction (rapydscript emits `RandomState(1)`, no `new`).
  const rng = np.random.RandomState(1);
  const a = rng.uniform(-Math.PI, Math.PI);
  assert.equal(typeof a, "number");
  assert.ok(a >= -Math.PI && a <= Math.PI);
  // Deterministic per seed: a fresh instance replays the same stream.
  const rng2 = np.random.RandomState(1);
  assert.equal(rng2.uniform(-Math.PI, Math.PI), a);
  // Reseeding replays the stream as well.
  rng2.seed(1);
  assert.equal(rng2.uniform(-Math.PI, Math.PI), a);
  // Common legacy methods exist and return sane shapes.
  assert.equal(typeof rng.rand(), "number");
  assert.equal(typeof rng.randn(), "number");
  assert.equal(typeof rng.random(), "number");
  const i = rng.randint(0, 10);
  assert.equal(typeof i, "number");
  assert.ok(i >= 0 && i < 10);
  const n = rng.normal(0, 1);
  assert.equal(typeof n, "number");
});

check("numpy facade np.random distributions accept kwargs and array params", () => {
  const np = ndCore.buildNumpy();
  // rapydscript maps kwargs via __argnames__; without it the kwargs bag is
  // passed as a single positional and numpy-ts throws
  // "Cannot convert object to primitive value" (error 139).
  assert.deepEqual(np.random.normal.__argnames__, ["loc", "scale", "size"]);
  assert.deepEqual(np.random.uniform.__argnames__, ["low", "high", "size"]);
  assert.deepEqual(np.random.standard_normal.__argnames__, ["size"]);
  const place = (names, pos, kw) => {
    const out = [...pos];
    for (const k of Object.keys(kw)) {
      const i = names.indexOf(k);
      if (i >= 0) out[i] = kw[k];
    }
    return out;
  };
  // The error-139 call shape: NDArray loc/scale passed as kwargs.
  np.random.seed(7);
  const loc = np.array([0, 0, 0]);
  const scale = np.multiply(0.6, np.ones([3]));
  const a = np.random.normal(...place(np.random.normal.__argnames__, [], { loc, scale, size: [5, 3] }));
  assert.deepEqual([...a.shape], [5, 3]);
  // Same distribution as the scalar call: standard_normal(size)*scale + loc.
  np.random.seed(7);
  const b = np.random.normal(0, 0.6, [5, 3]);
  for (const [x, y] of a.tolist().flat().map((v, i) => [v, b.tolist().flat()[i]])) {
    assert.ok(Math.abs(x - y) < 1e-12);
  }
  assert.ok(a.tolist().flat().every(Number.isFinite));
  // Generator methods share the wrappers.
  const rng = np.random.default_rng(7);
  assert.deepEqual(rng.normal.__argnames__, ["loc", "scale", "size"]);
  const g = rng.normal(...place(rng.normal.__argnames__, [], { loc, scale, size: [2, 3] }));
  assert.deepEqual([...g.shape], [2, 3]);
  assert.ok(g.tolist().flat().every(Number.isFinite));
});

console.log(`\n${passed} tests passed`);
