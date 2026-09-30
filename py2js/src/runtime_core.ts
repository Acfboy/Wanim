/**
 * Runtime shim that bridges the Python-ish code produced by rapydscript to the
 * wanim (TypeScript manim port) and numpy-ts libraries.
 *
 * Two things make a straight Python->JS translation of manim code fail:
 *
 *   1. Python classes overload arithmetic (`UP + LEFT`, `2 * UP`, ...), which
 *      JavaScript operators cannot dispatch on.  The patched rapydscript emits
 *      calls to `ρσ_op_add`/`ρσ_op_sub`/... (see the `operator_overloading`
 *      output option) and those helpers are implemented here.
 *   2. wanim classes are ES6 classes, which can neither be called without
 *      `new` nor `.apply`-ed the way rapydscript's class model expects.  We
 *      wrap every exported wanim class in a plain function that forwards to
 *      `new C(...args)` and give the prototype an `__init__` so that a
 *      rapydscript subclass (`class Foo(Scene)`) initializes its base.
 *
 * This module is dependency-injected so the very same logic runs under Node
 * (`runtime.ts`) and in the browser (`runtime_browser.ts`).
 */
import type * as NumpyTS from "numpy-ts";

type Any = any;

export interface RuntimeDeps {
  /** The wanim namespace (classes + helpers). */
  wanim: Record<string, Any>;
  /** A numpy-ts namespace (possibly a Python facade built by `buildNumpy`). */
  np: Record<string, Any>;
  /** The numpy-ts `NDArray` constructor, used for `instanceof` checks. */
  NDArray: new (...args: Any[]) => Any;
}

export interface PythonRuntime {
  globals: Record<string, Any>;
  wanim: Record<string, Any>;
  np: Record<string, Any>;
}

export const NUMERIC_OP_NAMES = new Set(["add", "subtract", "multiply", "divide", "negative", "power"]);

function isArrayLike(value: unknown): value is readonly number[] {
  return Array.isArray(value);
}

/** Return a scalar for 0-d NDArray results, mirroring numpy scalars. */
function scalarize(value: Any): Any {
  if (value !== null && value !== undefined && Array.isArray(value.shape) && value.shape.length === 0) {
    const raw = value.toArray();
    return Array.isArray(raw) ? raw[0] : raw;
  }
  return value;
}

/**
 * rapydscript passes Python keyword arguments as a single object tagged with a
 * `kwargs-object` symbol (created inside the compiled module).  Detect that
 * marker without needing access to the module-private symbol.
 */
function detectKwargs(value: unknown): Record<string, Any> | null {
  if (value === null || typeof value !== "object") return null;
  let tagged = false;
  for (const symbol of Object.getOwnPropertySymbols(value)) {
    if (String(symbol).includes("kwargs-object")) {
      tagged = true;
      break;
    }
  }
  if (!tagged) {
    for (const key of Object.keys(value)) {
      if (key.startsWith("kwargs-object-Symbol-")) {
        tagged = true;
        break;
      }
    }
  }
  if (!tagged) return null;
  const plain: Record<string, Any> = {};
  for (const key of Object.keys(value)) plain[key] = (value as Any)[key];
  return plain;
}

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | null = null;
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote !== null) {
      if (ch === "\\") i += 1;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      quote = ch;
    } else if (ch === "(" || ch === "[" || ch === "{") {
      depth += 1;
    } else if (ch === ")" || ch === "]" || ch === "}") {
      depth -= 1;
    } else if (ch === "," && depth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

function stripDefault(param: string): string {
  let depth = 0;
  let quote: string | null = null;
  for (let i = 0; i < param.length; i += 1) {
    const ch = param[i];
    if (quote !== null) {
      if (ch === "\\") i += 1;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") quote = ch;
    else if (ch === "(" || ch === "[" || ch === "{") depth += 1;
    else if (ch === ")" || ch === "]" || ch === "}") depth -= 1;
    else if (ch === "=" && depth === 0 && param[i + 1] !== "=" && param[i - 1] !== "=" && param[i - 1] !== "!" && param[i - 1] !== "<" && param[i - 1] !== ">") {
      return param.slice(0, i);
    }
  }
  return param;
}

/** Remove `//` and `/* *\/` comments (tsc preserves them in output) while
 * respecting string/template literals, so parameter lists can be parsed. */
function stripJsComments(src: string): string {
  let out = "";
  let quote: string | null = null;
  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];
    if (quote !== null) {
      out += ch;
      if (ch === "\\") {
        out += src[i + 1] ?? "";
        i += 1;
      } else if (ch === quote) {
        quote = null;
      }
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      quote = ch;
      out += ch;
      continue;
    }
    if (ch === "/" && src[i + 1] === "/") {
      while (i < src.length && src[i] !== "\n") i += 1;
      out += "\n";
      continue;
    }
    if (ch === "/" && src[i + 1] === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i += 1;
      i += 1;
      out += " ";
      continue;
    }
    out += ch;
  }
  return out;
}

/** Extract parameter names from a (non-minified, tsc-emitted) function. */
function extractParamList(src: string, open: number): (string | null)[] | null {
  if (open < 0) return null;
  let depth = 0;
  let close = open;
  for (let i = open; i < src.length; i += 1) {
    if (src[i] === "(") depth += 1;
    else if (src[i] === ")") {
      depth -= 1;
      if (depth === 0) {
        close = i;
        break;
      }
    }
  }
  const inner = src.slice(open + 1, close);
  if (inner.trim() === "") return [];
  return splitTopLevel(inner).map((param) => {
    const stripped = stripDefault(param).trim();
    if (stripped.startsWith("...")) return stripped;
    if (stripped.includes("{") || stripped.includes("[")) return null;
    return stripped;
  });
}

function parseParamNames(fn: (...a: Any[]) => Any): (string | null)[] | null {
  const src = stripJsComments(Function.prototype.toString.call(fn));
  return extractParamList(src, src.indexOf("("));
}

const OPTIONS_PARAM = /(options|opts|kwargs|config|parameters)$/i;

/** Map keyword names onto positional parameter slots, extras into the options bag. */
function placeKwargs(params: (string | null)[], positional: Any[], kwargs: Record<string, Any>): Any[] {
  const extras: Record<string, Any> = {};
  const hasRest = params.length > 0 && typeof params[params.length - 1] === "string" && (params[params.length - 1] as string).startsWith("...");
  const optionsIndex = !hasRest && params.length > 0 && typeof params[params.length - 1] === "string" && OPTIONS_PARAM.test(params[params.length - 1] as string)
    ? params.length - 1
    : -1;
  for (const [key, value] of Object.entries(kwargs)) {
    const index = params.indexOf(key);
    if (index >= 0) positional[index] = value;
    else extras[key] = value;
  }
  if (Object.keys(extras).length > 0) {
    if (optionsIndex >= 0) {
      const existing = positional[optionsIndex];
      positional[optionsIndex] = existing && typeof existing === "object" ? { ...existing, ...extras } : extras;
    } else {
      positional.push(extras);
    }
  }
  return positional;
}

const PYTHON_STR_METHODS = [
  "upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit", "join",
  "replace", "startswith", "endswith", "find", "rfind", "index", "rindex",
  "count", "capitalize", "title", "zfill", "ljust", "rjust", "center",
  "isdigit", "isalpha", "isalnum", "isspace", "islower", "isupper",
  "swapcase", "partition", "rpartition", "expandtabs", "splitlines", "format",
];

/**
 * Install Python string-method shims on the vm context's own String.prototype.
 * rapydscript exposes these on its `ρσ_str` function object (not on
 * String.prototype), so `"x".upper()` fails unless we bridge them.
 */
export function installPythonShims(sandbox: Record<string, Any>): void {
  const S = sandbox.String?.prototype as Any;
  if (S === undefined) return;
  for (const name of PYTHON_STR_METHODS) {
    Object.defineProperty(S, name, {
      configurable: true,
      writable: true,
      value: function (this: Any, ...args: Any[]) {
        const fn = sandbox.ρσ_str?.[name];
        if (typeof fn === "function") return fn(String(this), ...args);
        throw new TypeError(`String has no Python method '${name}'`);
      },
    });
  }
  // Python `replace` replaces all occurrences (JS replaces the first).
  Object.defineProperty(S, "replace", {
    configurable: true,
    writable: true,
    value: function (this: Any, oldValue: Any, newValue: Any, count?: number) {
      const text = String(this);
      const search = String(oldValue);
      if (search === "") return text;
      if (count === undefined || count < 0) return text.split(search).join(String(newValue));
      let remaining = count;
      let result = "";
      let index = 0;
      while (remaining > 0) {
        const found = text.indexOf(search, index);
        if (found < 0) break;
        result += text.slice(index, found) + String(newValue);
        index = found + search.length;
        remaining -= 1;
      }
      return result + text.slice(index);
    },
  });
  // Python `split()` with no separator splits on runs of whitespace.
  Object.defineProperty(S, "split", {
    configurable: true,
    writable: true,
    value: function (this: Any, sep?: Any, maxsplit?: number) {
      const text = String(this);
      if (sep === undefined || sep === null) {
        const parts = text.trim().split(/\s+/);
        return text.trim() === "" ? [] : maxsplit === undefined ? parts : parts.slice(0, maxsplit);
      }
      const parts = text.split(String(sep));
      return maxsplit === undefined ? parts : parts.slice(0, maxsplit + 1);
    },
  });
}

/** Build the runtime helpers bound to a specific wanim/numpy pair. */
export function createRuntimeCore(deps: RuntimeDeps) {
  const { wanim, np, NDArray } = deps;

  function isNdArray(value: unknown): value is NumpyTS.NDArray {
    return value instanceof (NDArray as Any);
  }

  function asNd(value: Any): Any {
    if (isNdArray(value)) return value;
    if (Array.isArray(value) || typeof value === "number") return np.asarray(value);
    return value;
  }

  function elementwise(a: readonly number[], b: readonly number[], f: (x: number, y: number) => number): number[] {
    if (a.length !== b.length) return undefined as unknown as number[];
    const out = new Array<number>(a.length);
    for (let i = 0; i < a.length; i += 1) out[i] = f(a[i], b[i]);
    return out;
  }

  function pythonAdd(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__add__ === "function") return a.__add__(b);
    if (b !== null && b !== undefined && typeof b.__radd__ === "function") return b.__radd__(a);
    if (typeof a === "number" && typeof b === "number") return a + b;
    if (typeof a === "string" || typeof b === "string") return String(a) + String(b);
    if (isArrayLike(a) && isArrayLike(b)) {
      const r = elementwise(a, b, (x, y) => x + y);
      if (r !== undefined) return r;
      return scalarize(np.add(asNd(a), asNd(b)));
    }
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v + b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => v + a);
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.add(asNd(a), asNd(b)));
    return a + b;
  }

  function pythonSub(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__sub__ === "function") return a.__sub__(b);
    if (b !== null && b !== undefined && typeof b.__rsub__ === "function") return b.__rsub__(a);
    if (typeof a === "number" && typeof b === "number") return a - b;
    if (isArrayLike(a) && isArrayLike(b)) {
      const r = elementwise(a, b, (x, y) => x - y);
      if (r !== undefined) return r;
      return scalarize(np.subtract(asNd(a), asNd(b)));
    }
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v - b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => a - v);
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.subtract(asNd(a), asNd(b)));
    return a - b;
  }

  function pythonMul(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__mul__ === "function") return a.__mul__(b);
    if (b !== null && b !== undefined && typeof b.__rmul__ === "function") return b.__rmul__(a);
    if (typeof a === "number" && typeof b === "number") return a * b;
    if (typeof a === "string" && typeof b === "number") return a.repeat(b);
    if (typeof a === "number" && typeof b === "string") return b.repeat(a);
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v * b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => v * a);
    if (isArrayLike(a) && isArrayLike(b)) {
      const r = elementwise(a, b, (x, y) => x * y);
      if (r !== undefined) return r;
    }
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.multiply(asNd(a), asNd(b)));
    return a * b;
  }

  function pythonDiv(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__truediv__ === "function") return a.__truediv__(b);
    if (b !== null && b !== undefined && typeof b.__rtruediv__ === "function") return b.__rtruediv__(a);
    if (typeof a === "number" && typeof b === "number") return a / b;
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v / b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => a / v);
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.divide(asNd(a), asNd(b)));
    return a / b;
  }

  function pythonFloorDiv(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__floordiv__ === "function") return a.__floordiv__(b);
    if (typeof a === "number" && typeof b === "number") return Math.floor(a / b);
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => Math.floor(v / b));
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => Math.floor(a / v));
    if (isArrayLike(a) || isArrayLike(b) || isNdArray(a) || isNdArray(b)) {
      return scalarize(np.floor_divide(asNd(a), asNd(b)));
    }
    return Math.floor(a / b);
  }

  function pythonMod(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__mod__ === "function") return a.__mod__(b);
    if (typeof a === "number" && typeof b === "number") {
      const r = a % b;
      return r !== 0 && r < 0 !== b < 0 ? r + b : r;
    }
    if (isArrayLike(a) && typeof b === "number") {
      return a.map((v) => {
        const r = v % b;
        return r !== 0 && r < 0 !== b < 0 ? r + b : r;
      });
    }
    if (isArrayLike(a) || isArrayLike(b) || isNdArray(a) || isNdArray(b)) {
      return scalarize(np.mod(asNd(a), asNd(b)));
    }
    return a % b;
  }

  function pythonPow(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__pow__ === "function") return a.__pow__(b);
    if (typeof a === "number" && typeof b === "number") return a ** b;
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v ** b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => a ** v);
    if (isArrayLike(a) || isArrayLike(b) || isNdArray(a) || isNdArray(b)) {
      return scalarize(np.power(asNd(a), asNd(b)));
    }
    return a ** b;
  }

  function pythonNeg(a: Any): Any {
    if (a !== null && a !== undefined && typeof a.__neg__ === "function") return a.__neg__();
    if (typeof a === "number") return -a;
    if (isArrayLike(a)) return a.map((v) => -v);
    if (isNdArray(a)) return scalarize(np.negative(a));
    return -a;
  }

  /**
   * Python bound-method semantics for attribute *reads*.
   *
   * rapydscript compiles `square.scale` to a plain JS property read, which
   * returns the unbound prototype function (no `__self__`).  Python instead
   * yields a bound method carrying `__self__`/`__func__`, which `ApplyMethod`
   * and the updater helpers rely on.  The patched compiler emits `ρσ_getdot`
   * for non-callee reads (when `operator_overloading` is on); this rebuilds
   * the CPython shape for data-property methods on a Mobject receiver and
   * returns everything else unchanged:
   *   - non-function values;
   *   - functions whose receiver is not a Mobject (`np.linalg.norm`);
   *   - accessor properties, whose getter result (`mobject.animate`, a
   *     callable Proxy) must not be re-bound into a plain wrapper.
   */
  function isMobjectLike(value: unknown): boolean {
    const Mobject = wanim.Mobject as (new (...args: Any[]) => Any) | undefined;
    return typeof Mobject === "function" && value instanceof (Mobject as Any);
  }

  function isAccessorProperty(obj: Any, name: string): boolean {
    let cur: Any = obj;
    while (cur !== null && cur !== undefined) {
      const descriptor = Object.getOwnPropertyDescriptor(cur, name);
      if (descriptor !== undefined) {
        return typeof descriptor.get === "function" || typeof descriptor.set === "function";
      }
      cur = Object.getPrototypeOf(cur);
    }
    return false;
  }

  function getDot(obj: Any, name: string): Any {
    const value = (obj as Any)[name];
    if (typeof value !== "function") return value;
    if (!isMobjectLike(obj)) return value;
    if (isAccessorProperty(obj, name)) return value;
    if ((value as Any).__self__ === obj) return value;
    const bound = function (this: Any, ...args: Any[]): Any {
      return (value as Any).apply(obj, args);
    } as Any;
    bound.__self__ = obj;
    bound.__func__ = value;
    // `always`/`f_always` key off these historical spellings
    // (see `bind_mobject_method` in wanim).
    bound.mobject = obj;
    bound.originalMethod = value;
    return bound;
  }

  function isNativeClass(fn: unknown): fn is new (...args: Any[]) => Any {
    return typeof fn === "function" && /^class[\s{]/.test(Function.prototype.toString.call(fn));
  }

  const constructorParamsCache = new WeakMap<Any, (string | null)[] | null>();

  /** Parameter names of a native class constructor (walking up the chain). */
  function parseConstructorParams(Ctor: new (...a: Any[]) => Any): (string | null)[] | null {
    if (constructorParamsCache.has(Ctor)) return constructorParamsCache.get(Ctor) ?? null;
    let cur: Any = Ctor;
    let result: (string | null)[] | null = null;
    while (typeof cur === "function") {
      const src = stripJsComments(Function.prototype.toString.call(cur));
      const match = /\bconstructor\s*\(/.exec(src);
      if (match) {
        result = extractParamList(src, src.indexOf("(", match.index));
        break;
      }
      cur = Object.getPrototypeOf(cur);
    }
    constructorParamsCache.set(Ctor, result);
    return result;
  }

  /** Apply Python keyword-argument semantics to a native constructor call. */
  function mapConstructorArgs(Ctor: new (...a: Any[]) => Any, args: Any[]): Any[] {
    const kwargs = detectKwargs(args[args.length - 1]);
    if (kwargs === null) return args;
    const params = parseConstructorParams(Ctor);
    const positional = args.slice(0, -1);
    if (params === null || params.length === 0) return [...positional, kwargs];
    // Many wanim constructors accept a leading options/kwargs bag (e.g.
    // `Arc(radiusOrOptions, ...)`, `Dot(pointOrOptions, ...)`).  When the call is
    // kwargs-only and the first parameter is such a bag, hand it the whole
    // kwargs object so wanim's own `isPlainOptionsBag` path runs.
    const firstIsOptionsBag =
      typeof params[0] === "string" && OPTIONS_PARAM.test(params[0] as string);
    if (positional.length === 0 && firstIsOptionsBag) return [kwargs];
    return placeKwargs(params, positional, kwargs);
  }

  function callWithKwargs(fn: (...a: Any[]) => Any, thisArg: Any, args: Any[]): Any {
    const kwargs = detectKwargs(args[args.length - 1]);
    if (kwargs === null) return fn.apply(thisArg, args);
    const params = parseParamNames(fn);
    const positional = args.slice(0, -1);
    if (params === null || params.length === 0) {
      return fn.apply(thisArg, [...positional, kwargs]);
    }
    return fn.apply(thisArg, placeKwargs(params, positional, kwargs));
  }

  /**
   * Wrap a class prototype's methods so a trailing rapydscript kwargs object is
   * expanded into the method's named parameters (or merged into a trailing
   * options bag).  Constructors are untouched.
   */
  function wrapPrototypeMethods(Ctor: new (...args: Any[]) => Any): void {
    for (const name of Object.getOwnPropertyNames(Ctor.prototype)) {
      if (name === "constructor") continue;
      const descriptor = Object.getOwnPropertyDescriptor(Ctor.prototype, name);
      if (descriptor === undefined || typeof descriptor.value !== "function") continue;
      const original = descriptor.value as (...a: Any[]) => Any;
      if ((original as Any).__wanim_kwargs_wrapped__) continue;
      const wrapped = function (this: Any, ...args: Any[]) {
        return callWithKwargs(original, this, args);
      };
      (wrapped as Any).__wanim_kwargs_wrapped__ = true;
      (wrapped as Any).__wanim_original__ = original;
      Object.defineProperty(Ctor.prototype, name, {
        configurable: descriptor.configurable,
        enumerable: descriptor.enumerable,
        writable: descriptor.writable,
        value: wrapped,
      });
    }
  }

  function wrapNativeClass(Ctor: new (...args: Any[]) => Any): Any {
    wrapPrototypeMethods(Ctor);
    const Wrapper = function (...args: Any[]) {
      return new Ctor(...mapConstructorArgs(Ctor, args));
    };
    Wrapper.prototype = Ctor.prototype;
    (Wrapper as Any).__rapydscript_native__ = true;
    (Wrapper as Any).__wrapped_native__ = Ctor;
    // Copy static members (e.g. `_original__init__`, `get_scene_classes_from_module`).
    for (const key of Object.getOwnPropertyNames(Ctor)) {
      if (key === "prototype" || key === "length" || key === "name") continue;
      try {
        (Wrapper as Any)[key] = (Ctor as Any)[key];
      } catch {
        /* ignore read-only statics */
      }
    }
    // Let rapydscript subclasses initialize the native base through `__init__`.
    if (!Object.prototype.hasOwnProperty.call(Ctor.prototype, "__init__")) {
      Object.defineProperty(Ctor.prototype, "__init__", {
        configurable: true,
        writable: true,
        value: function (this: Any, ...args: Any[]) {
          const instance = new Ctor(...args);
          Object.assign(this, instance);
        },
      });
    }
    return Wrapper;
  }

  // Wrapping the (large, static) wanim namespace is idempotent, so cache it:
  // the editor demo rebuilds the runtime on every Run, and re-walking every
  // class prototype was pure overhead.
  const wrappedNamespaceCache = new WeakMap<object, Record<string, Any>>();

  /** Wrap every wanim class so rapydscript can `new` it via a plain call. */
  function wrapWanimNamespace(ns: Record<string, Any>): Record<string, Any> {
    const cached = wrappedNamespaceCache.get(ns);
    if (cached !== undefined) return cached;
    const out: Record<string, Any> = {};
    for (const [key, value] of Object.entries(ns)) {
      out[key] = isNativeClass(value) ? wrapNativeClass(value as Any) : value;
    }
    wrappedNamespaceCache.set(ns, out);
    return out;
  }

  /**
   * Python-ish numpy facade.  numpy-ts mirrors numpy's *function* names but
   * requires NDArray inputs for most ufuncs and omits constants such as `pi`.
   */
  function buildNumpy(): Record<string, Any> {
    const facade: Record<string, Any> = { ...(np as Record<string, Any>) };
    facade.pi = Math.PI;
    facade.e = Math.E;
    facade.inf = Infinity;
    facade.nan = NaN;
    facade.euler_gamma = 0.5772156649015329;
    facade.newaxis = null;

    const scalarUfuncs = [
      "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
      "exp", "log", "log2", "log10", "sqrt", "cbrt", "absolute", "abs", "sign",
      "floor", "ceil", "rint", "trunc", "square", "reciprocal", "deg2rad", "rad2deg",
    ];
    for (const name of scalarUfuncs) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      facade[name] = (x: Any, ...rest: Any[]) => scalarize(fn(asNd(x), ...rest));
    }
    // Plain-number fast path: `np.cos(t)` on a scalar otherwise boxes the value
    // into an NDArray, dispatches through numpy-ts (WASM/alloc), then unwraps —
    // ~30µs per call, which dominates e.g. a `ParametricFunction` lambda called
    // once per sample.  A JS double is bit-identical to numpy's float64 result
    // for these elementwise functions, so answer directly.  `rint` is excluded
    // (numpy rounds half to even; `Math.round` does not).
    const numberUfuncs: Record<string, (x: number) => number> = {
      sin: Math.sin, cos: Math.cos, tan: Math.tan,
      arcsin: Math.asin, arccos: Math.acos, arctan: Math.atan,
      sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
      exp: Math.exp, log: Math.log, log2: Math.log2, log10: Math.log10,
      sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, absolute: Math.abs,
      sign: Math.sign, floor: Math.floor, ceil: Math.ceil, trunc: Math.trunc,
      square: (x) => x * x, reciprocal: (x) => 1 / x,
      deg2rad: (x) => (x * Math.PI) / 180, rad2deg: (x) => (x * 180) / Math.PI,
    };
    for (const name of Object.keys(numberUfuncs)) {
      const fast = numberUfuncs[name];
      const wrapped = facade[name];
      if (typeof wrapped !== "function") continue;
      facade[name] = (x: Any, ...rest: Any[]) =>
        typeof x === "number" && rest.length === 0 ? fast(x) : wrapped(x, ...rest);
    }
    // numpy-ts requires an array shape; Python accepts an int (`np.zeros(3)`).
    for (const name of ["zeros", "ones", "empty"]) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      facade[name] = (shape: Any, ...rest: Any[]) =>
        fn(typeof shape === "number" ? [shape] : shape, ...rest);
    }
    {
      const fn = (np as Any).full;
      if (typeof fn === "function") {
        facade.full = (shape: Any, fill: Any, ...rest: Any[]) =>
          fn(typeof shape === "number" ? [shape] : shape, fill, ...rest);
      }
    }
    return facade;
  }

  /** Build the global environment the compiled Python module runs in. */
  function buildRuntime(): PythonRuntime {
    const globals: Record<string, Any> = {};
    Object.assign(globals, wrapWanimNamespace(wanim));
    const numpyFacade = buildNumpy();
    globals.np = numpyFacade;
    globals.numpy = numpyFacade;
    globals.ρσ_op_add = pythonAdd;
    globals.ρσ_op_sub = pythonSub;
    globals.ρσ_op_mul = pythonMul;
    globals.ρσ_op_div = pythonDiv;
    globals.ρσ_op_floordiv = pythonFloorDiv;
    globals.ρσ_op_mod = pythonMod;
    globals.ρσ_op_pow = pythonPow;
    globals.ρσ_op_neg = pythonNeg;
    globals.ρσ_getdot = getDot;
    return { globals, wanim, np: numpyFacade };
  }

  return {
    isNdArray,
    buildNumpy,
    wrapWanimNamespace,
    buildRuntime,
    pythonAdd,
    pythonSub,
    pythonMul,
    pythonDiv,
    pythonFloorDiv,
    pythonMod,
    pythonPow,
    pythonNeg,
    getDot,
  };
}
