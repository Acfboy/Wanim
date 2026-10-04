/**
 * Python `math` module for the wanim browser/JS runtime.
 *
 * Manim scenes (including original 3b1b sources) routinely use the Python
 * standard-library `math` module (`import math`, `math.comb(...)`,
 * `math.sqrt(...)`, `from math import pi, sin, ...`), which cannot exist in
 * the browser as-is. This module ports CPython's `math` (scalar double
 * semantics backed by JS `Math` plus exact integer helpers for
 * `factorial`/`comb`/`perm`/`gcd`/`lcm`/`isqrt`) so that `preprocess.ts` can
 * accept `import math` / `from math import ...` and compiled scenes observe
 * the same values as CPython (within double precision).
 *
 * Fidelity notes (verified against CPython 3.12 semantics):
 *   - Integer-valued inputs arrive as JS numbers (rapydscript represents
 *     every int as a double), so huge-int exactness beyond 2**53 is
 *     unrepresentable anyway; `comb`/`perm`/`factorial` use a multiplicative
 *     loop that stays exact within the double range.
 *   - `fsum` uses Neumaier compensated summation like CPython's `math.fsum`.
 *   - `remainder` implements IEEE 754 remainder with round-half-to-even
 *     quotient selection, matching CPython's `math.remainder`.
 *   - `erf`/`erfc` use the Abramowitz-Stegun 7.1.26 approximation
 *     (|error| <= 1.5e-7); `gamma`/`lgamma` use Lanczos (g=7, 9 terms).
 *   - `ulp`/`nextafter` operate on the exact IEEE 754 binary64
 *     representation via DataView.
 *
 * Known deviations from CPython (all inherent to the JS runtime):
 *   - Integers past 2**53 lose precision (rapydscript doubles); big-int
 *     results (e.g. `factorial(200)`) are doubles, not arbitrary precision.
 *   - `erf`/`erfc`/`gamma`/`lgamma` are approximations, not correctly
 *     rounded to the last ulp.
 */

type Any = any;

// ---------------------------------------------------------------------------
// Small Python-semantics helpers (mirrors `python_random.ts`, which cannot be
// imported here without duplicating its kwargs logic).
// ---------------------------------------------------------------------------

/** Coerce a rapydscript int-like (`operator.index`) to a JS number. */
function asIndex(value: Any, name: string): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") {
    if (!Number.isInteger(value)) {
      throw new TypeError(`'float' object cannot be interpreted as an integer (${name})`);
    }
    return value;
  }
  if (typeof value === "bigint") return Number(value);
  if (value !== null && value !== undefined && typeof value.__index__ === "function") {
    return asIndex(value.__index__(), name);
  }
  throw new TypeError(`'${typeof value}' object cannot be interpreted as an integer (${name})`);
}

/** Coerce to a float like CPython's `PyFloat_AsDouble` (bool/int/float ok). */
function asFloat(value: Any, name: string): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  if (value !== null && value !== undefined && typeof value.__float__ === "function") {
    return asFloat(value.__float__(), name);
  }
  throw new TypeError(`must be real number, not '${typeof value}' (${name})`);
}

/**
 * Detect a trailing rapydscript kwargs bag (mirrors `detectKwargs` in
 * `runtime_core.ts`, which cannot be imported here without a module cycle).
 */
function takeKwargsBag(args: Any[]): Record<string, Any> | null {
  if (args.length === 0) return null;
  const last = args[args.length - 1];
  if (last === null || typeof last !== "object") return null;
  if (Object.getPrototypeOf(last) !== null) return null;
  let tagged = false;
  for (const symbol of Object.getOwnPropertySymbols(last)) {
    if (String(symbol).includes("kwargs-object")) {
      tagged = true;
      break;
    }
  }
  if (!tagged) {
    for (const key of Object.keys(last)) {
      if (key.startsWith("kwargs-object-Symbol-")) {
        tagged = true;
        break;
      }
    }
  }
  if (!tagged) return null;
  const plain: Record<string, Any> = {};
  for (const key of Object.keys(last)) plain[key] = last[key];
  return plain;
}

/** Map a kwargs bag onto positional slots (unknown keys raise TypeError). */
function applyKwargs(fn: string, names: string[], positional: Any[], kwargs: Record<string, Any>): Any[] {
  const out = positional.slice();
  for (const [key, value] of Object.entries(kwargs)) {
    if (key.startsWith("kwargs-object")) continue;
    const index = names.indexOf(key);
    if (index < 0) {
      throw new TypeError(`${fn}() got an unexpected keyword argument '${key}'`);
    }
    out[index] = value;
  }
  return out;
}

/** Split `(…args, maybeKwargsBag)` into positionals plus mapped names. */
function splitCall(fn: string, names: string[], args: Any[]): Any[] {
  const bag = takeKwargsBag(args);
  const positional = bag !== null ? args.slice(0, -1) : args.slice();
  return bag !== null ? applyKwargs(fn, names, positional, bag) : positional;
}

function required(value: Any, fn: string, name: string): Any {
  if (value === undefined) {
    throw new TypeError(`${fn}() missing required argument '${name}'`);
  }
  return value;
}

/** Require an integer argument (TypeError for floats, ValueError if < 0). */
function asNonNegInt(value: Any, fn: string, name: string): number {
  const n = asIndex(value, name);
  if (n < 0) throw new RangeError(`${fn}(): ${name} must be a non-negative integer`);
  return n;
}

// ---------------------------------------------------------------------------
// Exact integer helpers.
// ---------------------------------------------------------------------------

function factorialN(n: number): number {
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

/** Binomial coefficient via the multiplicative formula (exact in doubles). */
function combN(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  const small = Math.min(k, n - k);
  let result = 1;
  for (let i = 1; i <= small; i += 1) {
    result = (result * (n - small + i)) / i;
  }
  return Math.round(result);
}

function permN(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i += 1) result *= n - i;
  return result;
}

function gcd2(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

// ---------------------------------------------------------------------------
// Special functions (approximations).
// ---------------------------------------------------------------------------

/** `erf` via Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7). */
function erfImpl(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return 1;
  if (x === -Infinity) return -1;
  if (x === 0) return x; // exact zero (the A&S fit below is off by ~1e-9 at 0)
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

/** Lanczos (g=7, 9 coefficients) log-gamma for x > 0. */
function logGammaPositive(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    // Reflection; only reached for gamma(x) with 0 < x < 0.5 from `gammaImpl`.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGammaPositive(1 - x);
  }
  const z = x - 1;
  let a = c[0]!;
  for (let i = 1; i < 9; i += 1) a += c[i]! / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Neumaier compensated summation (what CPython's `fsum` uses). */
function fsumImpl(values: Iterable<Any>): number {
  let sum = 0;
  let comp = 0;
  for (const raw of values) {
    const x = asFloat(raw, "fsum");
    const t = sum + x;
    if (Math.abs(sum) >= Math.abs(x)) comp += sum - t + x;
    else comp += x - t + sum;
    sum = t;
  }
  return sum + comp;
}

function toFloat64Bits(x: number): bigint {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, x, false);
  const bytes = new Uint8Array(buf);
  let bits = 0n;
  for (const byte of bytes) bits = (bits << 8n) | BigInt(byte);
  return bits;
}

function fromFloat64Bits(bits: bigint): number {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  for (let i = 7; i >= 0; i -= 1) {
    view.setUint8(i, Number(bits & 0xffn));
    bits >>= 8n;
  }
  return view.getFloat64(0, false);
}

/** Next representable binary64 after `x` towards `y` (single step). */
function nextAfterStep(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (x === y) return y;
  if (x === 0) {
    // Smallest subnormal with the sign of `y` (CPython copies the sign of y
    // for zero, including `nextafter(0, -0) == -0` which `x === y` covers).
    return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  }
  if (x === Infinity) return Number.MAX_VALUE;
  if (x === -Infinity) return -Number.MAX_VALUE;
  const bits = toFloat64Bits(x);
  // Increasing magnitudes move away from zero in unsigned order for
  // positives and towards zero for negatives.
  const step = (x > 0) === (y > x) ? 1n : -1n;
  return fromFloat64Bits((bits + step) & ((1n << 64n) - 1n));
}

// ---------------------------------------------------------------------------
// Module definition.
// ---------------------------------------------------------------------------

/** Names exported by `from math import *` (functions + constants). */
export const MATH_PUBLIC = [
  "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt",
  "ceil", "comb", "copysign", "cos", "cosh", "degrees", "dist", "erf",
  "erfc", "exp", "exp2", "expm1", "fabs", "factorial", "floor", "fmod",
  "frexp", "fsum", "gamma", "gcd", "hypot", "isclose", "isfinite",
  "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma", "log", "log10",
  "log1p", "log2", "modf", "nextafter", "perm", "pow", "prod", "radians",
  "remainder", "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "ulp",
  "pi", "e", "tau", "inf", "nan",
];

export interface PythonMathModule extends Record<string, Any> {
  pi: number;
  e: number;
  tau: number;
  inf: number;
  nan: number;
}

function annotate(fn: Any, names: string[]): Any {
  fn.__argnames__ = [...names];
  return fn;
}

function seqToNumbers(seq: Any, fn: string): number[] {
  if (seq === null || seq === undefined || typeof seq.length !== "number") {
    throw new TypeError(`${fn}(): argument must be a sequence`);
  }
  const out: number[] = [];
  for (let i = 0; i < seq.length; i += 1) out.push(asFloat(seq[i], fn));
  return out;
}

/** Build the `math` module object exposed as a runtime global. */
export function buildPythonMath(): PythonMathModule {
  const mod: Record<string, Any> = {};

  mod.acos = annotate((...args: Any[]) => {
    const [x] = splitCall("acos", ["x"], args);
    return Math.acos(asFloat(required(x, "acos", "x"), "acos"));
  }, ["x"]);
  mod.acosh = annotate((...args: Any[]) => {
    const [x] = splitCall("acosh", ["x"], args);
    return Math.acosh(asFloat(required(x, "acosh", "x"), "acosh"));
  }, ["x"]);
  mod.asin = annotate((...args: Any[]) => {
    const [x] = splitCall("asin", ["x"], args);
    return Math.asin(asFloat(required(x, "asin", "x"), "asin"));
  }, ["x"]);
  mod.asinh = annotate((...args: Any[]) => {
    const [x] = splitCall("asinh", ["x"], args);
    return Math.asinh(asFloat(required(x, "asinh", "x"), "asinh"));
  }, ["x"]);
  mod.atan = annotate((...args: Any[]) => {
    const [x] = splitCall("atan", ["x"], args);
    return Math.atan(asFloat(required(x, "atan", "x"), "atan"));
  }, ["x"]);
  mod.atan2 = annotate((...args: Any[]) => {
    const [y, x] = splitCall("atan2", ["y", "x"], args);
    return Math.atan2(
      asFloat(required(y, "atan2", "y"), "atan2"),
      asFloat(required(x, "atan2", "x"), "atan2"),
    );
  }, ["y", "x"]);
  mod.atanh = annotate((...args: Any[]) => {
    const [x] = splitCall("atanh", ["x"], args);
    return Math.atanh(asFloat(required(x, "atanh", "x"), "atanh"));
  }, ["x"]);
  mod.cbrt = annotate((...args: Any[]) => {
    const [x] = splitCall("cbrt", ["x"], args);
    return Math.cbrt(asFloat(required(x, "cbrt", "x"), "cbrt"));
  }, ["x"]);
  mod.ceil = annotate((...args: Any[]) => {
    const [x] = splitCall("ceil", ["x"], args);
    return Math.ceil(asFloat(required(x, "ceil", "x"), "ceil"));
  }, ["x"]);

  mod.comb = annotate((...args: Any[]) => {
    const [nRaw, kRaw] = splitCall("comb", ["n", "k"], args);
    const n = asIndex(required(nRaw, "comb", "n"), "n");
    const k = asIndex(required(kRaw, "comb", "k"), "k");
    if (n < 0 || k < 0) throw new RangeError("comb(): n must be a non-negative integer");
    return combN(n, k);
  }, ["n", "k"]);

  mod.copysign = annotate((...args: Any[]) => {
    const [x, y] = splitCall("copysign", ["x", "y"], args);
    const fx = asFloat(required(x, "copysign", "x"), "copysign");
    const fy = asFloat(required(y, "copysign", "y"), "copysign");
    // Preserve signed zero like CPython (Math.abs loses the sign of y=±0).
    if (fy === 0) return 1 / fy === -Infinity ? -Math.abs(fx) : Math.abs(fx);
    return fy < 0 || Object.is(fy, -0) ? -Math.abs(fx) : Math.abs(fx);
  }, ["x", "y"]);
  mod.cos = annotate((...args: Any[]) => {
    const [x] = splitCall("cos", ["x"], args);
    return Math.cos(asFloat(required(x, "cos", "x"), "cos"));
  }, ["x"]);
  mod.cosh = annotate((...args: Any[]) => {
    const [x] = splitCall("cosh", ["x"], args);
    return Math.cosh(asFloat(required(x, "cosh", "x"), "cosh"));
  }, ["x"]);
  mod.degrees = annotate((...args: Any[]) => {
    const [x] = splitCall("degrees", ["x"], args);
    return (asFloat(required(x, "degrees", "x"), "degrees") * 180) / Math.PI;
  }, ["x"]);

  mod.dist = annotate((...args: Any[]) => {
    const [p, q] = splitCall("dist", ["p", "q"], args);
    const a = seqToNumbers(required(p, "dist", "p"), "dist");
    const b = seqToNumbers(required(q, "dist", "q"), "dist");
    if (a.length !== b.length) throw new Error("dist(): both points must have the same number of dimensions");
    let sum = 0;
    for (let i = 0; i < a.length; i += 1) sum += (a[i]! - b[i]!) ** 2;
    return Math.sqrt(sum);
  }, ["p", "q"]);

  mod.erf = annotate((...args: Any[]) => {
    const [x] = splitCall("erf", ["x"], args);
    return erfImpl(asFloat(required(x, "erf", "x"), "erf"));
  }, ["x"]);
  mod.erfc = annotate((...args: Any[]) => {
    const [x] = splitCall("erfc", ["x"], args);
    return 1 - erfImpl(asFloat(required(x, "erfc", "x"), "erfc"));
  }, ["x"]);
  mod.exp = annotate((...args: Any[]) => {
    const [x] = splitCall("exp", ["x"], args);
    return Math.exp(asFloat(required(x, "exp", "x"), "exp"));
  }, ["x"]);
  mod.exp2 = annotate((...args: Any[]) => {
    const [x] = splitCall("exp2", ["x"], args);
    return 2 ** asFloat(required(x, "exp2", "x"), "exp2");
  }, ["x"]);
  mod.expm1 = annotate((...args: Any[]) => {
    const [x] = splitCall("expm1", ["x"], args);
    return Math.expm1(asFloat(required(x, "expm1", "x"), "expm1"));
  }, ["x"]);
  mod.fabs = annotate((...args: Any[]) => {
    const [x] = splitCall("fabs", ["x"], args);
    return Math.abs(asFloat(required(x, "fabs", "x"), "fabs"));
  }, ["x"]);

  mod.factorial = annotate((...args: Any[]) => {
    const [xRaw] = splitCall("factorial", ["x"], args);
    const n = asIndex(required(xRaw, "factorial", "x"), "x");
    if (n < 0) throw new RangeError("factorial(): x must be a non-negative integer");
    return factorialN(n);
  }, ["x"]);
  mod.floor = annotate((...args: Any[]) => {
    const [x] = splitCall("floor", ["x"], args);
    return Math.floor(asFloat(required(x, "floor", "x"), "floor"));
  }, ["x"]);
  mod.fmod = annotate((...args: Any[]) => {
    const [x, y] = splitCall("fmod", ["x", "y"], args);
    const fx = asFloat(required(x, "fmod", "x"), "fmod");
    const fy = asFloat(required(y, "fmod", "y"), "fmod");
    if (fy === 0) throw new RangeError("fmod(): math domain error");
    // Preserve signed zero of the dividend like CPython.
    if (fx === 0) return fx;
    const q = Math.trunc(fx / fy);
    return fx - q * fy;
  }, ["x", "y"]);
  mod.frexp = annotate((...args: Any[]) => {
    const [x] = splitCall("frexp", ["x"], args);
    const fx = asFloat(required(x, "frexp", "x"), "frexp");
    if (fx === 0) return [fx, 0];
    if (!Number.isFinite(fx)) return [fx, 0];
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat64(0, fx, false);
    const bits = toFloat64Bits(fx);
    const expBits = Number((bits >> 52n) & 0x7ffn);
    let mantissa: number;
    let exp: number;
    if (expBits === 0) {
      // Subnormal: scale up, then compensate the exponent.
      const scaled = frexpScaled(fx);
      mantissa = scaled[0];
      exp = scaled[1];
    } else {
      exp = expBits - 1022;
      const sign = fx < 0 || Object.is(fx, -0) ? -1 : 1;
      const fracBits = bits & ((1n << 52n) - 1n);
      mantissa = sign * (0.5 + Number(fracBits) / 2 ** 53);
    }
    return [mantissa, exp];
  }, ["x"]);

  mod.fsum = annotate((...args: Any[]) => {
    const [iterable] = splitCall("fsum", ["iterable"], args);
    return fsumImpl(required(iterable, "fsum", "iterable") as Iterable<Any>);
  }, ["iterable"]);

  mod.gamma = annotate((...args: Any[]) => {
    const [x] = splitCall("gamma", ["x"], args);
    return gammaImpl(asFloat(required(x, "gamma", "x"), "gamma"));
  }, ["x"]);

  mod.gcd = annotate((...args: Any[]) => {
    const positional = splitCall("gcd", ["a"], args);
    if (positional.length === 0) return 0;
    let g = 0;
    for (const raw of positional) g = gcd2(g, asIndex(raw, "gcd"));
    return Math.abs(g);
  }, ["a"]);

  mod.hypot = annotate((...args: Any[]) => {
    const positional = splitCall("hypot", ["x"], args);
    // `math.hypot` takes variadic coordinates (and no kwargs bag survives
    // rapydscript's own `__argnames__` mapping); sum robustly to avoid
    // overflow like CPython does.
    let max = 0;
    const scaled: number[] = [];
    for (const raw of positional) {
      const v = Math.abs(asFloat(raw, "hypot"));
      scaled.push(v);
      if (v > max) max = v;
    }
    if (max === 0) return 0;
    if (max === Infinity) return Infinity;
    let sum = 0;
    for (const v of scaled) sum += (v / max) ** 2;
    return max * Math.sqrt(sum);
  }, ["x"]);

  mod.isclose = annotate((...args: Any[]) => {
    const [aRaw, bRaw, relTolRaw, absTolRaw] = splitCall(
      "isclose", ["a", "b", "rel_tol", "abs_tol"], args,
    );
    const a = asFloat(required(aRaw, "isclose", "a"), "isclose");
    const b = asFloat(required(bRaw, "isclose", "b"), "isclose");
    const relTol = relTolRaw === undefined ? 1e-9 : asFloat(relTolRaw, "isclose");
    const absTol = absTolRaw === undefined ? 0 : asFloat(absTolRaw, "isclose");
    if (relTol < 0 || absTol < 0) throw new RangeError("isclose(): tolerances must be non-negative");
    if (a === b) return true;
    if (a === Infinity || a === -Infinity || b === Infinity || b === -Infinity) return false;
    return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
  }, ["a", "b", "rel_tol", "abs_tol"]);

  mod.isfinite = annotate((...args: Any[]) => {
    const [x] = splitCall("isfinite", ["x"], args);
    return Number.isFinite(asFloat(required(x, "isfinite", "x"), "isfinite"));
  }, ["x"]);
  mod.isinf = annotate((...args: Any[]) => {
    const [x] = splitCall("isinf", ["x"], args);
    const fx = asFloat(required(x, "isinf", "x"), "isinf");
    return fx === Infinity || fx === -Infinity;
  }, ["x"]);
  mod.isnan = annotate((...args: Any[]) => {
    const [x] = splitCall("isnan", ["x"], args);
    return Number.isNaN(asFloat(required(x, "isnan", "x"), "isnan"));
  }, ["x"]);

  mod.isqrt = annotate((...args: Any[]) => {
    const [nRaw] = splitCall("isqrt", ["n"], args);
    const n = asIndex(required(nRaw, "isqrt", "n"), "n");
    if (n < 0) throw new RangeError("isqrt(): n must be a non-negative integer");
    return Math.floor(Math.sqrt(n));
  }, ["n"]);

  mod.lcm = annotate((...args: Any[]) => {
    const positional = splitCall("lcm", ["a"], args);
    if (positional.length === 0) return 1;
    let result = 1;
    for (const raw of positional) {
      const v = asIndex(raw, "lcm");
      if (v === 0) return 0;
      result = Math.abs((result / gcd2(result, v)) * v);
    }
    return result;
  }, ["a"]);

  mod.ldexp = annotate((...args: Any[]) => {
    const [x, i] = splitCall("ldexp", ["x", "i"], args);
    return (
      asFloat(required(x, "ldexp", "x"), "ldexp") *
      2 ** asIndex(required(i, "ldexp", "i"), "i")
    );
  }, ["x", "i"]);
  mod.lgamma = annotate((...args: Any[]) => {
    const [x] = splitCall("lgamma", ["x"], args);
    return lgammaImpl(asFloat(required(x, "lgamma", "x"), "lgamma"));
  }, ["x"]);

  mod.log = annotate((...args: Any[]) => {
    const [xRaw, baseRaw] = splitCall("log", ["x", "base"], args);
    const x = asFloat(required(xRaw, "log", "x"), "log");
    if (baseRaw === undefined || baseRaw === null) return Math.log(x);
    const base = asFloat(baseRaw, "log");
    return Math.log(x) / Math.log(base);
  }, ["x", "base"]);
  mod.log10 = annotate((...args: Any[]) => {
    const [x] = splitCall("log10", ["x"], args);
    return Math.log10(asFloat(required(x, "log10", "x"), "log10"));
  }, ["x"]);
  mod.log1p = annotate((...args: Any[]) => {
    const [x] = splitCall("log1p", ["x"], args);
    return Math.log1p(asFloat(required(x, "log1p", "x"), "log1p"));
  }, ["x"]);
  mod.log2 = annotate((...args: Any[]) => {
    const [x] = splitCall("log2", ["x"], args);
    return Math.log2(asFloat(required(x, "log2", "x"), "log2"));
  }, ["x"]);

  mod.modf = annotate((...args: Any[]) => {
    const [x] = splitCall("modf", ["x"], args);
    const fx = asFloat(required(x, "modf", "x"), "modf");
    if (!Number.isFinite(fx)) return fx === 0 ? [fx, fx] : [fx > 0 ? 0 : -0, fx];
    const intPart = Math.trunc(fx);
    // Preserve signed zero of the input: `modf(-0.1) == (-0.1, -0.0)`.
    const frac = fx - intPart;
    const fixedFrac = frac === 0 && (fx < 0 || Object.is(fx, -0)) ? -0 : frac;
    const fixedInt = intPart === 0 && (fx < 0 || Object.is(fx, -0)) ? -0 : intPart;
    return [fixedFrac, fixedInt];
  }, ["x"]);

  mod.nextafter = annotate((...args: Any[]) => {
    const [xRaw, yRaw, stepsRaw] = splitCall("nextafter", ["x", "y", "steps"], args);
    const x = asFloat(required(xRaw, "nextafter", "x"), "nextafter");
    const y = asFloat(required(yRaw, "nextafter", "y"), "nextafter");
    const steps = stepsRaw === undefined || stepsRaw === null ? 1 : asIndex(stepsRaw, "steps");
    if (steps < 0) throw new RangeError("nextafter(): steps must be non-negative");
    let out = x;
    for (let i = 0; i < steps; i += 1) {
      const next = nextAfterStep(out, y);
      out = next;
      if (out === y) break;
    }
    return out;
  }, ["x", "y", "steps"]);

  mod.perm = annotate((...args: Any[]) => {
    const [nRaw, kRaw] = splitCall("perm", ["n", "k"], args);
    const n = asIndex(required(nRaw, "perm", "n"), "n");
    const k = kRaw === undefined || kRaw === null ? n : asIndex(kRaw, "perm");
    if (n < 0 || k < 0) throw new RangeError("perm(): n must be a non-negative integer");
    return permN(n, k);
  }, ["n", "k"]);

  mod.pow = annotate((...args: Any[]) => {
    const [x, y] = splitCall("pow", ["x", "y"], args);
    return (
      asFloat(required(x, "pow", "x"), "pow") ** asFloat(required(y, "pow", "y"), "pow")
    );
  }, ["x", "y"]);

  mod.prod = annotate((...args: Any[]) => {
    const [iterableRaw, startRaw] = splitCall("prod", ["iterable", "start"], args);
    const iterable = required(iterableRaw, "prod", "iterable") as Iterable<Any>;
    let acc = startRaw === undefined || startRaw === null ? 1 : (startRaw as number);
    for (const raw of iterable as Iterable<Any>) acc = (acc as number) * (raw as number);
    return acc;
  }, ["iterable", "start"]);

  mod.radians = annotate((...args: Any[]) => {
    const [x] = splitCall("radians", ["x"], args);
    return (asFloat(required(x, "radians", "x"), "radians") * Math.PI) / 180;
  }, ["x"]);

  mod.remainder = annotate((...args: Any[]) => {
    const [x, y] = splitCall("remainder", ["x", "y"], args);
    const fx = asFloat(required(x, "remainder", "x"), "remainder");
    const fy = asFloat(required(y, "remainder", "y"), "remainder");
    if (fy === 0 || Number.isNaN(fx) || Number.isNaN(fy)) return NaN;
    if (fx === Infinity || fx === -Infinity) return NaN;
    if (fy === Infinity || fy === -Infinity) return fx;
    const q = fx / fy;
    // Round half to even, like C `remainder` / CPython `math.remainder`.
    const trunc = Math.trunc(q);
    const frac = Math.abs(q - trunc);
    let n: number;
    if (frac < 0.5) n = trunc;
    else if (frac > 0.5) n = trunc + (q > 0 ? 1 : -1);
    else n = trunc % 2 === 0 ? trunc : trunc + (q > 0 ? 1 : -1);
    const r = fx - n * fy;
    // Signed zero follows the dividend, like CPython.
    if (r === 0) return fx < 0 || Object.is(fx, -0) ? -0 : 0;
    return r;
  }, ["x", "y"]);

  mod.sin = annotate((...args: Any[]) => {
    const [x] = splitCall("sin", ["x"], args);
    return Math.sin(asFloat(required(x, "sin", "x"), "sin"));
  }, ["x"]);
  mod.sinh = annotate((...args: Any[]) => {
    const [x] = splitCall("sinh", ["x"], args);
    return Math.sinh(asFloat(required(x, "sinh", "x"), "sinh"));
  }, ["x"]);
  mod.sqrt = annotate((...args: Any[]) => {
    const [x] = splitCall("sqrt", ["x"], args);
    return Math.sqrt(asFloat(required(x, "sqrt", "x"), "sqrt"));
  }, ["x"]);
  mod.tan = annotate((...args: Any[]) => {
    const [x] = splitCall("tan", ["x"], args);
    return Math.tan(asFloat(required(x, "tan", "x"), "tan"));
  }, ["x"]);
  mod.tanh = annotate((...args: Any[]) => {
    const [x] = splitCall("tanh", ["x"], args);
    return Math.tanh(asFloat(required(x, "tanh", "x"), "tanh"));
  }, ["x"]);
  mod.trunc = annotate((...args: Any[]) => {
    const [x] = splitCall("trunc", ["x"], args);
    return Math.trunc(asFloat(required(x, "trunc", "x"), "trunc"));
  }, ["x"]);
  mod.ulp = annotate((...args: Any[]) => {
    const [x] = splitCall("ulp", ["x"], args);
    return ulpImpl(asFloat(required(x, "ulp", "x"), "ulp"));
  }, ["x"]);

  mod.pi = Math.PI;
  mod.e = Math.E;
  mod.tau = Math.PI * 2;
  mod.inf = Infinity;
  mod.nan = NaN;

  return mod as PythonMathModule;
}

function frexpScaled(x: number): [number, number] {
  // Scale a subnormal up by 2**64 (exact), decompose, then shift back.
  const scaled = x * 2 ** 64;
  const bits = toFloat64Bits(scaled);
  const expBits = Number((bits >> 52n) & 0x7ffn);
  const exp = expBits - 1022 - 64;
  const sign = scaled < 0 ? -1 : 1;
  const fracBits = bits & ((1n << 52n) - 1n);
  return [sign * (0.5 + Number(fracBits) / 2 ** 53), exp];
}

function gammaImpl(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return Infinity;
  if (x === 0) return 1 / x; // ±inf, matching CPython (preserves signed zero)
  if (Number.isInteger(x) && x < 0) return NaN; // poles -> ValueError in CPython
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gammaImpl(1 - x));
  return Math.exp(logGammaPositive(x));
}

function lgammaImpl(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return Infinity;
  if (x === 0) return Infinity;
  if (Number.isInteger(x) && x < 0) return Infinity;
  if (x < 0.5) {
    // log|Gamma(x)| via reflection; CPython returns the log of the absolute
    // value for negative non-integers.
    return Math.log(Math.PI) - Math.log(Math.abs(Math.sin(Math.PI * x))) - logGammaPositive(1 - x);
  }
  return logGammaPositive(x);
}

function ulpImpl(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity || x === -Infinity) return Infinity;
  const ax = Math.abs(x);
  if (ax === 0) return Number.MIN_VALUE;
  if (ax === Number.MAX_VALUE) {
    // Distance to the predecessor (nextafter towards zero).
    return ax - nextAfterStep(ax, 0);
  }
  const up = nextAfterStep(ax, Infinity);
  if (up === Infinity) return up - ax;
  const down = nextAfterStep(ax, 0);
  return Math.min(up - ax, ax - down);
}
