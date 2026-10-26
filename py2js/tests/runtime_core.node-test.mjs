// Plain-node test for py2js/src/runtime_core.ts (Python builtins shims).
//
// Run with:  pnpm test:py2js
import assert from "node:assert/strict";
import vm from "node:vm";

import * as npReal from "numpy-ts";

import { createRuntimeCore, installPythonShims } from "../dist/runtime_core.js";

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

check("pythonSum matches CPython on numbers, lists and generators", () => {
  assert.equal(core.pythonSum([1, 2, 3]), 6);
  assert.equal(core.pythonSum([1, 2, 3], 10), 16);
  assert.equal(core.pythonSum([]), 0);
  assert.equal(core.pythonSum([], 5), 5);
  assert.deepEqual(core.pythonSum([[1, 2], [3]]), [1, 2, 3]);
  function* gen() {
    yield 1;
    yield 2;
    yield 3;
  }
  assert.equal(core.pythonSum(gen()), 6);
  assert.throws(() => core.pythonSum(["a", "b"]), TypeError);
  assert.throws(() => core.pythonSum(123), TypeError);
  assert.throws(() => core.pythonSum(), TypeError);
});

check("pythonSum adds NDArray terms elementwise (error 270)", () => {
  ndCore.buildRuntime(); // NDArray prototype patch like the browser
  const np = ndCore.buildNumpy();
  const t = np.linspace(0, 1, 4);
  function* terms() {
    yield ndCore.pythonMul(2, t);
    yield ndCore.pythonMul(3, t);
  }
  const total = ndCore.pythonSum(terms());
  const got = total.toArray();
  const want = [0, 5 / 3, 10 / 3, 5];
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i += 1) {
    assert.ok(Math.abs(got[i] - want[i]) < 1e-12);
  }
  // The downstream `np.fft.rfft` from the failing scene accepts the result.
  const spectrum = npReal.fft.rfft(total);
  assert.equal(spectrum.shape[0], 3);
});

check("__wan_sum is exposed to compiled scenes as a global", () => {
  const runtime = core.buildRuntime();
  assert.equal(typeof runtime.globals.__wan_sum, "function");
  assert.equal(runtime.globals.__wan_sum([1, 2]), 3);
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

check("wanGetItem supports None/newaxis on NDArrays (error 242)", () => {
  ndCore.buildRuntime(); // applies the NDArray slice patch like the browser
  const xs = npReal.linspace(-2, 1, 5);
  const row = ndCore.wanGetItem(xs, null, ndCore.wanSlice(null, null));
  assert.deepEqual(row.shape, [1, 5]);
  const ys = npReal.linspace(-1.5, 1.5, 4);
  const col2 = ndCore.wanGetItem(ys, ndCore.wanSlice(null, null), null);
  assert.deepEqual(col2.shape, [4, 1]);
  // Broadcasts like the Mandelbrot `xs[None, :] + 1j * ys[:, None]` grid.
  const grid = npReal.asarray([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(ndCore.wanGetItem(grid, null, ":", ":").shape, [1, 2, 3]);
  // Fully-indexed numerics still use numpy semantics after the patch.
  assert.equal(ndCore.wanGetItem(grid, 0, 1).toArray(), 2);
});

check("wanGetItem supports boolean-mask and fancy row indices", () => {
  ndCore.buildRuntime(); // applies the NDArray slice patch like the browser
  const pts = npReal.asarray([[1, 2], [3, 4]]);
  // Comparison results are the bool-dtype masks comparisons produce
  // (`pts[:, 1] > 2.5`); numpy-ts infers float64 for literal boolean arrays.
  const mask = npReal.greater(npReal.asarray([2, 4]), 2.5);
  assert.deepEqual(ndCore.wanGetItem(pts, mask).toArray(), [[3, 4]]);
  assert.deepEqual(ndCore.wanGetItem(pts, mask, 0).toArray(), [3]);
  assert.deepEqual(ndCore.wanGetItem(pts, mask, 1).toArray(), [4]);
  const idx = npReal.asarray([1, 0]);
  assert.deepEqual(ndCore.wanGetItem(pts, idx, 1).toArray(), [4, 2]);
  assert.deepEqual(ndCore.wanGetItem(pts, idx, ":").toArray(), [[3, 4], [1, 2]]);
  // Plain-array masks/indices behave the same.
  assert.deepEqual(ndCore.wanGetItem(pts, [true, false], 1).toArray(), [2]);
  assert.deepEqual(ndCore.wanGetItem(pts, [1, 0], 0).toArray(), [3, 1]);
  // The stereographic-scene shape: boolean row filter, then a column index.
  const eq = npReal.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  const keep = npReal.greater(npReal.asarray([0.2, -0.5, 0.9]), 0);
  assert.deepEqual(ndCore.wanGetItem(eq, keep, 0).toArray(), [1, 7]);
  assert.deepEqual(ndCore.wanGetItem(eq, keep, 2).toArray(), [3, 9]);
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

check("linalg results are full NDArrays with Python methods (.T)", () => {
  const np = ndCore.buildNumpy();
  const a = np.array([[1.0, -1.5], [1.5, 0.5]]);
  // numpy-ts returns NDArrayCore from linalg.*: `.T` used to be `undefined`,
  // so `V = np.linalg.svd(a)[2].T` gave None and `V[1, 0]` threw
  // `'NoneType' object is not subscriptable`.
  const [, , vt] = np.linalg.svd(a);
  const v = vt.T;
  assert.deepEqual(v.shape, [2, 2]);
  assert.equal(typeof v.get([1, 0]), "number");
  assert.deepEqual(np.linalg.inv(a).T.shape, [2, 2]);
  const [w, vec] = np.linalg.eig(np.array([[2.0, 1.0], [1.0, 2.0]]));
  assert.deepEqual(w.T.shape, [2]);
  assert.deepEqual(vec.T.shape, [2, 2]);
  // Wrapping must not copy: full NDArrays keep their identity.
  assert.equal(np.asarray(a), a);
});

check("linalg numerical failures throw np.linalg.LinAlgError", () => {
  const np = ndCore.buildNumpy();
  // numpy-ts throws `Error: inv: singular matrix`; Python catches LinAlgError.
  assert.throws(
    () => np.linalg.inv([[1, 2], [2, 4]]),
    (error) => error instanceof np.linalg.LinAlgError && /singular/.test(error.message),
  );
  assert.throws(
    () => np.linalg.cholesky([[1, 2], [2, 1]]),
    (error) => error instanceof np.linalg.LinAlgError && /positive definite/.test(error.message),
  );
});

check("np dtype constructors cast scalars and array-likes", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.uint8(300), 44);
  assert.equal(np.uint8(-1), 255);
  assert.equal(np.int8(200), -56);
  assert.equal(np.uint16(70000), 4464);
  assert.equal(np.int64(5), 5);
  assert.equal(np.uint64(5), 5);
  assert.equal(np.float64(2), 2);
  assert.equal(np.float32(0.69), 0.6899999976158142);
  assert.equal(np.bool_(2), true);
  assert.equal(np.bool_(0), false);
  assert.equal(np.complex128(2.5).re, 2.5);
  assert.equal(np.complex(0, Math.PI / 4).im, Math.PI / 4);
  const casted = np.uint8([[0, 128], [255, 64]]);
  assert.equal(casted.dtype, "uint8");
  assert.deepEqual(casted.tolist(), [[0, 128], [255, 64]]);
});

check("np dtype constructors work as dtype= arguments and with astype()", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.zeros(3, np.uint8).dtype, "uint8");
  assert.equal(np.ones(3, np.float32).dtype, "float32");
  assert.equal(np.arange(0, 3, 1, np.int16).dtype, "int16");
  assert.equal(np.array([[1, 2]], np.uint16).dtype, "uint16");
  assert.equal(np.arange(5).astype(np.bool_).dtype, "bool");
});

check("np.dtype normalizes numeric specs and rejects structured dtypes", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.dtype("f4"), "float32");
  assert.equal(np.dtype("float"), "float64");
  assert.equal(np.dtype(np.float32), "float32");
  assert.equal(np.dtype(np.uint8), "uint8");
  assert.equal(np.zeros(2, np.dtype("u1")).dtype, "uint8");
  assert.throws(() => np.dtype([["point", "f4"]]), /structured dtypes are not supported/);
  assert.throws(() => np.dtype("object"), /unsupported dtype/);
});

check("np legacy constants and aliases", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.infty, Infinity);
  assert.equal(np.Inf, Infinity);
  assert.equal(np.PINF, Infinity);
  assert.equal(np.NINF, -Infinity);
  assert.ok(Number.isNaN(np.NaN));
  assert.equal(np.product([1, 2, 3, 4]), 24);
  assert.equal(np.prod([1, 2, 3, 4]), 24);
  assert.deepEqual(np.cumproduct([1, 2, 3]).tolist(), [1, 2, 6]);
});

check("scalar shims bridge .item()/.real/.imag onto unboxed primitives", () => {
  installPythonShims({ Number, Boolean, BigInt });
  assert.equal((5).item(), 5);
  assert.equal((1.5).real, 1.5);
  assert.equal((1.5).imag, 0);
  assert.equal(true.item(), true);
  assert.equal(true.real, 1);
  assert.equal(true.imag, 0);
  assert.equal((7n).item(), 7n);
});

check("string replace/split shims delegate RegExp calls to native", () => {
  // Install the shims in an isolated realm so this test does not mutate the
  // test process's own String.prototype.
  const sandbox = { console };
  const context = vm.createContext(sandbox);
  for (const name of ["Object", "Array", "String", "RegExp", "Number", "Boolean", "BigInt"]) {
    sandbox[name] = vm.runInContext(name, context);
  }
  installPythonShims(sandbox);
  const proto = sandbox.String.prototype;
  const regexp = (source, flags) => new sandbox.RegExp(source, flags);

  // rapydscript's `str.format` trims `:g` output with replace(/0+$/g, "").
  // With the Python replace shim intercepting the regex it did nothing, so
  // `f"{1.0:g}"` rendered "1.00000" instead of "1".
  assert.equal(proto.replace.call("1.00000", regexp("0+$", "g"), ""), "1.");
  assert.equal(proto.replace.call("abc", regexp("b"), (m) => m.toUpperCase()), "aBc");

  // Python semantics are unchanged for string patterns.
  assert.equal(proto.replace.call("a-b-c", "-", "+"), "a+b+c");
  assert.equal(proto.replace.call("a-b-c", "-", "+", 1), "a+b-c");
  assert.equal(proto.replace.call("a-b-c", "-", "+", 0), "a-b-c");

  assert.deepEqual(Array.from(proto.split.call("a1b2c", regexp("[0-9]"))), ["a", "b", "c"]);
  assert.deepEqual(Array.from(proto.split.call("a,b", ",")), ["a", "b"]);
  assert.deepEqual(Array.from(proto.split.call("  a  b ", undefined)), ["a", "b"]);
  assert.deepEqual(Array.from(proto.split.call("", undefined)), []);
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

check("np.random.RandomState replays numpy's legacy MT19937 stream", () => {
  const np = ndCore.buildNumpy();
  const nums = (x) => {
    const v = x && typeof x.tolist === "function" ? x.tolist() : x;
    return Array.isArray(v) ? v.map(Number) : Number(v);
  };
  // Reference values from CPython numpy (`np.random.RandomState(seed)`).
  for (const [seed, expected] of [
    [185104, [0.22110513689684463, 0.5562520794428832, 0.7719918076935661]],
    [780151, [0.9013098450794184, 0.8585832902345384, 0.0567201856385835]],
    [33, [0.24851012743772993, 0.44997542105079547, 0.4109408029965408]],
  ]) {
    const rng = np.random.RandomState(seed);
    assert.deepEqual(nums(rng.random(32)).slice(0, 3), expected);
  }
  // Stream continues across calls on the same instance.
  const rng = np.random.RandomState(185104);
  assert.ok(Math.abs(nums(rng.random(32)).reduce((a, b) => a + b, 0) - 17.185795920594238) < 1e-12);
  assert.equal(nums(rng.random()), 0.7282318438918697);
  // Other legacy distributions match numpy too.
  assert.equal(nums(np.random.RandomState(1).uniform(-Math.PI, Math.PI)), -0.521366120872004);
  assert.deepEqual(nums(np.random.RandomState(1).randint(0, 10, 5)), [5, 8, 9, 5, 0]);
  assert.deepEqual(
    nums(np.random.RandomState(7).normal(0, 1, 3)),
    [1.690525703800356, -0.4659373705408328, 0.0328201636785844],
  );
  assert.equal(nums(np.random.RandomState(42).choice(10)), 6);
  assert.deepEqual(nums(np.random.RandomState(42).permutation(5)), [1, 4, 2, 0, 3]);
  // Instances are independent: interleaved draws replay each seed's stream.
  const a = np.random.RandomState(11);
  const b = np.random.RandomState(12);
  const a1 = nums(a.random());
  nums(b.random());
  const ref = np.random.RandomState(11);
  assert.equal(a1, nums(ref.random()));
  assert.equal(nums(a.random()), nums(ref.random()));
  // Reseeding replays the stream.
  b.seed(12);
  assert.equal(nums(b.random()), nums(np.random.RandomState(12).random()));
  // Instance draws leave the module-global stream untouched.
  np.random.seed(999);
  np.random.random();
  const tmp = np.random.RandomState(5);
  nums(tmp.random(3));
  np.random.seed(999);
  np.random.random();
  const expected = np.random.random();
  np.random.seed(999);
  np.random.random();
  const tmp2 = np.random.RandomState(5);
  nums(tmp2.random(3));
  assert.equal(np.random.random(), expected);
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

check("numpy facade append/insert/delete accept axis kwarg and NDArray rows", () => {
  const np = ndCore.buildNumpy();
  // Without __argnames__ rapydscript's ρσ_interpolate_kwargs hands the kwargs
  // bag as a trailing object and numpy-ts throws
  // "Cannot convert object to primitive value" (error 152).
  assert.deepEqual(np.append.__argnames__, ["arr", "values", "axis"]);
  assert.deepEqual(np.insert.__argnames__, ["arr", "obj", "values", "axis"]);
  assert.deepEqual(np.delete.__argnames__, ["arr", "obj", "axis"]);
  const place = (names, pos, kw) => {
    const out = [...pos];
    for (const k of Object.keys(kw)) {
      const i = names.indexOf(k);
      if (i >= 0) out[i] = kw[k];
    }
    return out;
  };
  // The error-152 call shape: `np.append(a, [a[-1]], axis=0)` where the
  // values list nests an NDArray row (numpy-ts dtype inference rejects it
  // unless unwrapped to a plain list first).
  const a = np.array([[0, 4, 0]]);
  const row = a[a.shape[0] - 1];
  const grown = np.append(...place(np.append.__argnames__, [a, [row]], { axis: 0 }));
  assert.deepEqual(grown.tolist(), [[0, 4, 0], [0, 4, 0]]);
  // Positional form keeps working too.
  assert.deepEqual(np.append(a, [row], 0).tolist(), [[0, 4, 0], [0, 4, 0]]);
  // insert/delete share the same axis-kwarg shape (axis-insert itself is
  // not implemented by numpy-ts standalone, so only the mapping is
  // exercised for insert; delete with axis works end to end).
  assert.deepEqual(np.insert([1, 2, 3], 1, 99).tolist(), [1, 99, 2, 3]);
  const del = np.delete(...place(np.delete.__argnames__, [grown, 0], { axis: 0 }));
  assert.deepEqual(del.tolist(), [[0, 4, 0]]);
});

check("plain-object dict literals support update/clear/copy/pop/setdefault (error 167)", () => {
  // The error-167 call shape: `state.update(t1=..., ...)` compiles to
  // `ρσ_interpolate_kwargs.call(state, state.update, [kwbag])`, where the
  // kwargs bag arrives as a single positional (no __argnames__ on update).
  // Without the shim `state.update` is undefined and rapydscript throws
  // "Cannot read properties of undefined (reading '__argnames__')".
  const state = { t1: 1, w1: 0, t2: 2, w2: 0 };
  assert.equal(typeof state.update, "function");
  const kwbag = Object.create(null);
  kwbag.t1 = 3;
  kwbag.w1 = 0.5;
  state.update(kwbag);
  assert.deepEqual([state.t1, state.w1, state.t2, state.w2], [3, 0.5, 2, 0]);
  // Positional mappings and pair lists merge too; no-arg is a no-op.
  state.update({ t2: 4 });
  state.update([["w2", 6]]);
  state.update();
  assert.deepEqual([state.t1, state.w1, state.t2, state.w2], [3, 0.5, 4, 6]);
  // The shim stays non-enumerable: keys/spread/JSON see data only.
  assert.deepEqual(Object.keys(state).sort(), ["t1", "t2", "w1", "w2"]);
  assert.deepEqual({ ...state }.t1, 3);
  assert.ok(!Object.prototype.propertyIsEnumerable.call(state, "update"));
  // Companions.
  assert.deepEqual(state.copy(), { t1: 3, w1: 0.5, t2: 4, w2: 6 });
  assert.equal(state.setdefault("t1", 99), 3);
  assert.equal(state.setdefault("new", 7), 7);
  assert.equal(state.new, 7);
  assert.equal(state.pop("new"), 7);
  assert.ok(!("new" in state));
  assert.equal(state.pop("missing", "fallback"), "fallback");
  assert.throws(() => state.pop("missing"), /KeyError/);
  state.clear();
  assert.deepEqual(Object.keys(state), []);
  // Class instances still throw instead of silently merging.
  class Foo {}
  assert.throws(() => ({}.update.call(new Foo(), { a: 1 })), /non-dict/);
  // Real Maps delegate.
  const m = new Map([["a", 1]]);
  m.update({ b: 2 });
  assert.deepEqual([...m.entries()], [["a", 1], ["b", 2]]);
  // defineProperties with {value:} descriptors keeps working (an inherited
  // `get`/`set`/`value` would poison it -- none of those are installed).
  const target = {};
  Object.defineProperties(target, { x: { value: 1, enumerable: false } });
  assert.equal(target.x, 1);
  assert.equal({}.update, Object.prototype.update);
});

check("wanDictGet serves dict.get without an Object.prototype.get (error 275)", () => {
  // `counts.get(ch, 0)` desugars to `__wan_dict_get(counts, ch, 0)` because
  // `Object.prototype` must never gain a `get` (it poisons
  // `Object.defineProperties` via the descriptor protocol).
  const get = core.wanDictGet;
  const counts = {};
  for (const ch of "abide") counts[ch] = get(counts, ch, 0) + 1;
  assert.deepEqual(counts, { a: 1, b: 1, i: 1, d: 1, e: 1 });
  assert.equal(get(counts, "missing", 0), 0);
  assert.equal(get(counts, "missing"), null);
  assert.equal(get(counts, "a"), 1);
  // Real Maps delegate; other receivers with a native `get` delegate too.
  const m = new Map([["x", 5]]);
  assert.equal(get(m, "x", 0), 5);
  assert.equal(get(m, "y", 7), 7);
  assert.equal(get(m, "y"), null);
  assert.throws(() => get([1, 2], "a", 0), /no attribute 'get'/);
  assert.equal(typeof Object.prototype.get, "undefined");
});

check("np.where accepts scalar conditions (numpy-ts does not)", () => {
  const np = ndCore.buildNumpy();
  // Error 263: nested np.where over a 0-d np.asarray(t) scalar, where the
  // `x < -2` comparison scalarizes to a plain boolean.
  assert.equal(np.where(true, 0.0, 1.0), 0);
  assert.equal(np.where(false, 0.0, 1.0), 1);
  const cond0d = ndCore.pythonLt(np.asarray(-3.0), -2);
  assert.equal(cond0d, true);
  assert.equal(np.where(cond0d, 0.0, 1.0), 0);
  // Array conditions keep working, with scalar or array branches.
  assert.deepEqual(
    np.where(np.asarray([1, 0, 0]), 0.0, 1.0).tolist(),
    [0, 1, 1],
  );
  assert.deepEqual(
    np.where(np.asarray([1, 0]), np.asarray([7, 8]), np.asarray([9, 10])).tolist(),
    [7, 10],
  );
});

check("nested np.where matches the kinked() scene helper", () => {
  const np = ndCore.buildNumpy();
  const kinked = (t) => {
    const x = np.asarray(t);
    return np.where(
      ndCore.pythonLt(x, -2), 0.0,
      np.where(ndCore.pythonLt(x, -1), ndCore.pythonSub(ndCore.pythonNeg(x), 2),
        np.where(ndCore.pythonLt(x, 1), x,
          np.where(ndCore.pythonLt(x, 2), ndCore.pythonAdd(ndCore.pythonNeg(x), 2), 0.0))),
    );
  };
  assert.equal(kinked(-3), 0);
  assert.equal(kinked(-1.5), -0.5);
  assert.equal(kinked(0), 0);
  assert.equal(kinked(1.5), 0.5);
  assert.equal(kinked(3), 0);
  assert.deepEqual(kinked(np.linspace(-3, 3, 5)).tolist(), [0, -0.5, 0, 0.5, 0]);
});

check("bitwise helpers keep numpy boolean masks elementwise (stereographic scene)", () => {
  const np = ndCore.buildNumpy();
  const x = np.arange(6);
  const a = ndCore.pythonGt(x, 2); // [F,F,F,T,T,T]
  const b = ndCore.pythonLt(x, 4); // [T,T,T,T,F,F]
  assert.equal(a.dtype, "bool");
  // `sign_x ^ sign_y ^ sign_z` must chain elementwise, not coerce to 0.
  const chained = ndCore.pythonBitXor(ndCore.pythonBitXor(a, b), a);
  assert.equal(chained.dtype, "bool");
  assert.deepEqual(chained.tolist(), [1, 1, 1, 1, 0, 0]);
  assert.deepEqual(ndCore.pythonBitAnd(a, b).tolist(), [0, 0, 0, 1, 0, 0]);
  assert.deepEqual(ndCore.pythonBitOr(a, b).tolist(), [1, 1, 1, 1, 1, 1]);
  assert.deepEqual(ndCore.pythonInvert(a).tolist(), [1, 1, 1, 0, 0, 0]);
  // `is_light[:, None]` in the stereographic scene then broadcasts against
  // the 3-channel colors in `np.where`.
  const mask2d = ndCore.wanGetItem(chained, ndCore.wanSlice(null, null), null);
  assert.deepEqual(mask2d.shape, [6, 1]);
  const colors = np.where(mask2d, np.asarray([1, 2, 3]), np.asarray([4, 5, 6]));
  assert.deepEqual(colors.shape, [6, 3]);
  assert.deepEqual(colors.tolist()[0], [1, 2, 3]);
});

check("bitwise helpers fall back to Python scalars and integers", () => {
  assert.equal(ndCore.pythonBitXor(true, false), true);
  assert.equal(ndCore.pythonBitAnd(true, false), false);
  assert.equal(ndCore.pythonBitOr(false, false), false);
  assert.equal(ndCore.pythonInvert(true), false);
  assert.equal(ndCore.pythonBitAnd(6, 3), 2);
  assert.equal(ndCore.pythonBitOr(4, 1), 5);
  assert.equal(ndCore.pythonBitXor(6, 3), 5);
  assert.equal(ndCore.pythonInvert(0), -1);
  assert.deepEqual(ndCore.pythonBitAnd([6, 3], [3, 3]), [2, 3]);
});

check("bitwise op helpers are exposed to compiled scenes as globals", () => {
  const runtime = ndCore.buildRuntime();
  assert.equal(typeof runtime.globals["ρσ_op_bitand"], "function");
  assert.equal(typeof runtime.globals["ρσ_op_bitor"], "function");
  assert.equal(typeof runtime.globals["ρσ_op_bitxor"], "function");
  assert.equal(typeof runtime.globals["ρσ_op_invert"], "function");
});

check("np.interp accepts scalar x and plain lists (numpy-ts does not)", () => {
  const np = ndCore.buildNumpy();
  assert.equal(np.interp(0.5, np.asarray([0, 1, 2]), np.asarray([10, 20, 30])), 15);
  assert.equal(np.interp(0.5, [0, 1, 2], [10, 20, 30]), 15);
  assert.deepEqual(
    np.interp([0.5, 1.5], [0, 1, 2], [10, 20, 30]).tolist(),
    [15, 25],
  );
});

check("np.convolve accepts plain lists and mode positionally", () => {
  const np = ndCore.buildNumpy();
  assert.deepEqual(np.convolve([1, 2, 3], [0, 1, 0.5], "same").tolist(), [1, 2.5, 4]);
  assert.deepEqual(
    np.convolve(np.asarray([1, 2, 3]), np.asarray([0, 1, 0.5]), "same").tolist(),
    [1, 2.5, 4],
  );
});

check("np.searchsorted accepts scalar v and side (error 376)", () => {
  const np = ndCore.buildNumpy();
  const edges = np.cumsum([0, 0.1, 0.5, 0.2, 0.05, 0.15]);
  // Scalar v mirrors numpy (scalar in, scalar out); numpy-ts requires NDArray
  // inputs and returns a broken 0-size array for a plain number.
  assert.equal(np.searchsorted(edges, 0.314, "right"), 2);
  assert.equal(np.searchsorted(edges, 0.1, "right"), 2);
  assert.equal(np.searchsorted(edges, 0.1, "left"), 1);
  assert.equal(np.searchsorted([0, 0.1, 0.6, 0.8, 0.85, 1], 0.999, "right"), 5);
  // Array v keeps the NDArray result.
  assert.deepEqual(
    np.searchsorted(edges, [0.314, 0.5], "right").tolist(),
    [2, 2],
  );
  // `side=` arrives as a kwargs bag through rapydscript; it needs argnames.
  assert.deepEqual(np.searchsorted.__argnames__, ["a", "v", "side", "sorter"]);
});

check("NDArray single-dim getitem serves boolean masks and fancy indices (error 333)", () => {
  ndCore.buildRuntime(); // applies the NDArray prototype patch
  const pts = npReal.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32]]);
  // `rng_points[norms < 4]`: an elementwise comparison mask filters rows.
  const mask = ndCore.pythonLt(npReal.asarray([1, 5, 2]), 4);
  assert.deepEqual(pts.__getitem__(mask).tolist(), [[10, 11, 12], [30, 31, 32]]);
  // Plain boolean lists filter the same way; integer lists pick rows.
  assert.deepEqual(pts.__getitem__([true, false, true]).tolist(), [[10, 11, 12], [30, 31, 32]]);
  assert.deepEqual(pts.__getitem__([2, 0]).tolist(), [[30, 31, 32], [10, 11, 12]]);
  assert.deepEqual(pts.__getitem__(npReal.asarray([0, 2])).tolist(), [[10, 11, 12], [30, 31, 32]]);
  // Scalars keep the Proxy behavior (row views, negative indices).
  assert.deepEqual(pts.__getitem__(0).tolist(), [10, 11, 12]);
  assert.deepEqual(pts.__getitem__(-1).tolist(), [30, 31, 32]);
  // 1-D masks filter elements.
  assert.deepEqual(npReal.asarray([5, 6, 7]).__getitem__([true, false, true]).tolist(), [5, 7]);
});

check("NDArray __eq__/__ne__ are elementwise so (a == b).all() works", () => {
  ndCore.buildRuntime(); // applies the NDArray prototype patch
  const a = npReal.asarray([1, 2, 3]);
  const b = npReal.asarray([1, 2, 4]);
  // The compiled `a == b` reaches `ρσ_equals`, which prefers `a.__eq__`.
  const eq = a.__eq__(b);
  assert.ok(eq !== undefined, "__eq__ must be defined on NDArray");
  assert.equal(typeof eq.all, "function");
  assert.equal(eq.all(), false);
  assert.equal(a.__eq__(npReal.asarray([1, 2, 3])).all(), true);
  assert.equal(a.__ne__(b).any(), true);
  // 0-d comparisons collapse to plain booleans so `if arr == x:` keeps working.
  assert.equal(npReal.asarray(1).__eq__(npReal.asarray(1)), true);
});

check("pythonFormat covers the int mini-language (error 342)", () => {
  const f = core.pythonFormat;
  assert.equal(f(5, "04b"), "0101");
  assert.equal(f(2, "04b"), "0010");
  assert.equal(f(0, "04b"), "0000");
  assert.equal(f(15, "04b"), "1111");
  assert.equal(f(-5, "08b"), "-0000101");
  assert.equal(f(255, "#x"), "0xff");
  assert.equal(f(255, "#X"), "0XFF");
  assert.equal(f(8, "#o"), "0o10");
  assert.equal(f(5, "#b"), "0b101");
  assert.equal(f(0, "#x"), "0x0");
  assert.equal(f(5, "#d"), "5");
  assert.equal(f(5, "+d"), "+5");
  assert.equal(f(5, " d"), " 5");
  assert.equal(f(5, ""), "5");
  assert.equal(f(5), "5");
  assert.equal(f(1000000, ",d"), "1,000,000");
  assert.equal(f(1000000, ","), "1,000,000");
  assert.equal(f(1000000, "_d"), "1_000_000");
  assert.equal(f(255, "_b"), "1111_1111");
  assert.equal(f(0x12345678, "_x"), "1234_5678");
  assert.equal(f(65, "c"), "A");
  assert.equal(f(true, "d"), "1");
  assert.equal(f(true, ""), "True");
  assert.equal(f(false, "s"), "False");
  assert.equal(f(5, ">5"), "    5");
  assert.equal(f(5, "<5"), "5    ");
  assert.equal(f(5, "^5"), "  5  ");
  assert.equal(f(-5, "=8d"), "-      5");
  assert.equal(f(-5, "=08d"), "-0000005");
  assert.equal(f(5, "0=8d"), "00000005");
  assert.equal(f(5, "0>8d"), "00000005");
  assert.equal(f(10n, "04b"), "1010");
  // NOTE: JS cannot tell `3.0` from `3`, so integer-valued numbers always
  // take the int path (CPython would reject `format(3.0, "d")`); only
  // genuinely fractional values raise for int-only codes.
  assert.equal(f(3.0, "d"), "3");
  assert.throws(() => f(3.5, "d"), /Unknown format code/);
  assert.throws(() => f(5, ".2d"), /Precision not allowed/);
  assert.throws(() => f(5, "s"), /Unknown format code/);
  assert.throws(() => f(1000000, ",n"), /Cannot specify/);
  assert.throws(() => f(255, ",b"), /Cannot specify/);
});

check("pythonFormat covers floats and strings", () => {
  const f = core.pythonFormat;
  assert.equal(f(3.14159, ".2f"), "3.14");
  assert.equal(f(2.675, ".2f"), "2.67");
  assert.equal(f(0.5, "%"), "50.000000%");
  assert.equal(f(0.5, ".1%"), "50.0%");
  assert.equal(f(12345.678, "e"), "1.234568e+04");
  assert.equal(f(12345.678, ".2E"), "1.23E+04");
  assert.equal(f(12345.678, "g"), "12345.7");
  assert.equal(f(2.0, "g"), "2");
  assert.equal(f(2.5, "#g"), "2.50000");
  assert.equal(f(2.5, ""), "2.5");
  assert.equal(f(-2.5, ""), "-2.5");
  assert.equal(f(100.0, ""), "100"); // JS numbers lose `.0`, like str()
  assert.equal(f(3.14159, ".2"), "3.1");
  assert.equal(f(3.14159, ".4"), "3.142");
  assert.equal(f(2.5, "n"), "2.5");
  assert.equal(f(1000000.5, "n"), "1e+06");
  assert.equal(f(1234.567, ","), "1,234.567");
  assert.equal(f(1234.5, ",.2f"), "1,234.50");
  assert.equal(f(1e6, ",g"), "1e+06");
  assert.equal(f(Infinity, ".2f"), "inf");
  assert.equal(f(-Infinity, "+.0f"), "-inf");
  assert.equal(f(NaN, "f"), "nan");
  assert.equal(f("ab", ""), "ab");
  assert.equal(f("hello", ".2"), "he");
  assert.equal(f("ab", "^5"), " ab  ");
  assert.equal(f("ab", "*>5"), "***ab");
  assert.equal(f("x", "04"), "x000");
  assert.throws(() => f("ab", "=5"), /=/);
  assert.throws(() => f("ab", "d"), /Unknown format code/);
});

check("format is exposed to compiled scenes as a global", () => {
  const runtime = core.buildRuntime();
  assert.equal(typeof runtime.globals.format, "function");
  assert.equal(runtime.globals.format(9, "04b"), "1001");
});

check("native class wrappers do not expose __argnames__ (kwargs pass through)", () => {
  // rapydscript's ρσ_interpolate_kwargs consumes the trailing kwargs bag when
  // the callee exposes __argnames__, mapping named keys onto positional slots
  // and *dropping the rest*.  Constructor wrappers must therefore hide it from
  // rapydscript (the raw class keeps it for `parseConstructorParams`), so the
  // bag reaches the wrapper's own `mapConstructorArgs`, which merges extras
  // into the trailing options slot.  Regression: `PathSlider(path,
  // is_closed=True)` silently lost `is_closed`.
  class PathSlider {
    constructor(path, options = {}) {
      this.path = path;
      this.options = options;
    }
  }
  PathSlider.__argnames__ = ["path", "options"];
  const runtime = createRuntimeCore({ wanim: { PathSlider }, np: {}, NDArray: FakeNDArray });
  const wrapped = runtime.wrapWanimNamespace({ PathSlider }).PathSlider;
  assert.equal(wrapped.__argnames__, undefined);
  assert.deepEqual(PathSlider.__argnames__, ["path", "options"]);

  const bag = Object.create(null);
  bag[Symbol("kwargs-object")] = true;
  bag.value_range = [0, 360];
  bag.is_closed = true;
  const instance = wrapped("ring", bag);
  assert.equal(instance.path, "ring");
  assert.equal(instance.options.is_closed, true);
  assert.deepEqual(instance.options.value_range, [0, 360]);
});

check("subclass constructors do not inherit the parent's __argnames__", () => {
  // A class static is inherited through the constructor prototype chain.
  // `Slider(options)` extends `PathSlider(path, options)`: reading the
  // inherited ["path", "options"] made the bridge map `start`/`end`/
  // `value_range`/... into a nonexistent slot and build the Slider with no
  // options at all (default value_range [0, 1], init_value 0).
  class PathSlider {
    constructor(path, options = {}) {
      this.path = path;
      this.options = options;
    }
  }
  PathSlider.__argnames__ = ["path", "options"];
  class Slider extends PathSlider {
    constructor(options = {}) {
      super(undefined, options);
    }
  }
  const runtime = createRuntimeCore({ wanim: { PathSlider, Slider }, np: {}, NDArray: FakeNDArray });
  const wrapped = runtime.wrapWanimNamespace({ PathSlider, Slider });

  const bag = Object.create(null);
  bag[Symbol("kwargs-object")] = true;
  bag.start = [-2, 0, 0];
  bag.value_range = [0.5, 3];
  bag.init_value = 1;
  const instance = wrapped.Slider(bag);
  assert.deepEqual(instance.options.value_range, [0.5, 3]);
  assert.equal(instance.options.init_value, 1);
  assert.deepEqual(instance.options.start, [-2, 0, 0]);
});

console.log(`\n${passed} tests passed`);
