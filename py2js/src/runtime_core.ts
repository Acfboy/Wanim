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
  // rapydscript's kwargs bags are `Object.create(None)` with a private symbol
  // tag.  Every other object a call site can trail (arrays, NDArrays, Mobjects,
  // ManimColors, ...) has a prototype, so reject it here without enumerating
  // its keys: this function is called for every wrapped method invocation, and
  // `Object.keys()` on a 10k-point array allocated 10k strings for nothing.
  if (Object.getPrototypeOf(value) !== null) return null;
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
  // Explicit `__argnames__` (set in wanim source) survives bundling, while
  // `Function.prototype.toString` parameter names do not: esbuild renames
  // colliding locals when hoisting modules into one bundle (e.g.
  // `arrange_in_grid(rows=...)` became `rows2`), silently breaking the
  // Python kwargs -> positional mapping. Prefer the explicit metadata.
  const explicit = (fn as Any).__argnames__;
  if (Array.isArray(explicit)) return [...explicit];
  const src = stripJsComments(Function.prototype.toString.call(fn));
  return extractParamList(src, src.indexOf("("));
}

const OPTIONS_PARAM = /(options|opts|kwargs|config|parameters)$/i;

// Suffix applied by `renameJsReservedWords` (see `preprocess.ts`): Python
// identifiers that collide with JavaScript reserved words (`function`, `new`,
// `default`, ...) are rewritten to `<name>_modified` before compilation, so
// the translated JS stays valid.  wanim's TypeScript side cannot declare a
// parameter literally named `function` either, and follows the Python
// convention of a trailing underscore instead (`function_`, as in
// `ParametricFunction(function_, ...)` / `ApplyFunction(function_, ...)`).
// The kwargs bridge below must therefore accept the renamed spelling:
// `function_modified` -> `function_`.
const MODIFIED_SUFFIX = "_modified";

/**
 * Resolve a (possibly `_modified`-renamed) kwargs key to the actual parameter
 * name.  Exact matches win; a renamed key falls back to its base name and to
 * the base-plus-underscore TS spelling; an unrenamed key falls back to the
 * trailing-underscore spelling (Python `function=` vs TS `function_` when the
 * rename step is ever bypassed).
 */
function resolveKwargKey(key: string, params: (string | null)[]): string {
  if (params.indexOf(key) >= 0) return key;
  if (key.endsWith(MODIFIED_SUFFIX)) {
    const base = key.slice(0, -MODIFIED_SUFFIX.length);
    if (params.indexOf(base) >= 0) return base;
    if (params.indexOf(`${base}_`) >= 0) return `${base}_`;
    // No positional slot: normalize the extras/options bag to the base name
    // so downstream option parsing never sees the `_modified` spelling.
    return base;
  }
  if (params.indexOf(`${key}_`) >= 0) return `${key}_`;
  return key;
}

/**
 * Normalize a kwargs bag for a callee whose parameter names are unknown
 * (pass-through path): strip the `_modified` rename so the receiver sees the
 * Python spelling.  Both `base` and `base_` aliases are provided (when absent
 * from the original bag) because TS receivers may declare either spelling.
 */
function normalizeKwargsBag(kwargs: Record<string, Any>): Record<string, Any> {
  const out: Record<string, Any> = {};
  for (const [key, value] of Object.entries(kwargs)) {
    if (key.endsWith(MODIFIED_SUFFIX)) {
      const base = key.slice(0, -MODIFIED_SUFFIX.length);
      if (!(base in kwargs) && !(base in out)) out[base] = value;
      const underscored = `${base}_`;
      if (!(underscored in kwargs) && !(underscored in out)) out[underscored] = value;
      out[key] = value;
    } else {
      out[key] = value;
    }
  }
  return out;
}

/** Map keyword names onto positional parameter slots, extras into the options bag. */
function placeKwargs(params: (string | null)[], positional: Any[], kwargs: Record<string, Any>): Any[] {
  const extras: Record<string, Any> = {};
  const hasRest = params.length > 0 && typeof params[params.length - 1] === "string" && (params[params.length - 1] as string).startsWith("...");
  const optionsIndex = !hasRest && params.length > 0 && typeof params[params.length - 1] === "string" && OPTIONS_PARAM.test(params[params.length - 1] as string)
    ? params.length - 1
    : -1;
  for (const [key, value] of Object.entries(kwargs)) {
    const resolved = resolveKwargKey(key, params);
    const index = params.indexOf(resolved);
    if (index >= 0) positional[index] = value;
    else extras[resolved] = value;
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

// --- Python `%` string formatting -----------------------------------------
//
// rapydscript lowers `%` to `ρσ_op_mod`, so a string left operand ("%s" % x)
// would otherwise fall through to numeric/numpy modulo.  These helpers
// implement CPython's printf-style formatting for the common conversion
// specifiers, flags, width, precision and the `%(name)s` mapping form.

interface PercentSpec {
  minus: boolean;
  plus: boolean;
  space: boolean;
  hash: boolean;
  zero: boolean;
  width: number | null;
  precision: number | null;
}

function pyStr(value: Any): string {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (Array.isArray(value)) return `[${value.map(pyStr).join(", ")}]`;
  return String(value);
}

function padNumeric(sign: string, digits: string, spec: PercentSpec): string {
  const body = sign + digits;
  if (spec.width === null || body.length >= spec.width) return body;
  const pad = spec.width - body.length;
  if (spec.minus) return sign + digits + " ".repeat(pad);
  if (spec.zero) return sign + "0".repeat(pad) + digits;
  return " ".repeat(pad) + body;
}

function padText(text: string, spec: PercentSpec): string {
  if (spec.width === null || text.length >= spec.width) return text;
  const pad = " ".repeat(spec.width - text.length);
  return spec.minus ? text + pad : pad + text;
}

function normalizeExponent(text: string, upper: boolean): string {
  const match = /^([0-9.]+)[eE]([+-])(\d+)$/.exec(text);
  if (match === null) return text;
  return `${match[1]}${upper ? "E" : "e"}${match[2]}${match[3]!.padStart(2, "0")}`;
}

function stripTrailingZeros(text: string): string {
  return text.indexOf(".") < 0 ? text : text.replace(/\.?0+$/, "");
}

function formatGeneral(abs: number, precision: number, alt: boolean, upper: boolean): string {
  if (abs === 0) return alt && precision > 1 ? `0.${"0".repeat(precision - 1)}` : "0";
  const p = precision === 0 ? 1 : precision;
  const exponent = Math.floor(Math.log10(abs));
  if (exponent < -4 || exponent >= p) {
    let mantissa = abs.toExponential(p - 1);
    if (!alt) {
      const [head, tail] = mantissa.split("e");
      mantissa = `${stripTrailingZeros(head!)}e${tail}`;
    }
    return normalizeExponent(mantissa, upper);
  }
  const text = abs.toFixed(Math.max(0, p - 1 - exponent));
  return alt ? text : stripTrailingZeros(text);
}

function formatPercentValue(value: Any, conversion: string, spec: PercentSpec): string {
  switch (conversion) {
    case "s":
    case "a": {
      let text = pyStr(value);
      if (spec.precision !== null) text = text.slice(0, spec.precision);
      return padText(text, spec);
    }
    case "r": {
      let text = typeof value === "string" ? `'${value}'` : pyStr(value);
      if (spec.precision !== null) text = text.slice(0, spec.precision);
      return padText(text, spec);
    }
    case "c": {
      const text = typeof value === "number" ? String.fromCharCode(value) : pyStr(value).charAt(0);
      return padText(text, spec);
    }
    case "d":
    case "i":
    case "u": {
      const n = Math.trunc(Number(value)) || 0;
      const sign = n < 0 ? "-" : spec.plus ? "+" : spec.space ? " " : "";
      let digits = Math.abs(n).toString(10);
      if (spec.precision !== null && spec.precision > digits.length) {
        digits = "0".repeat(spec.precision - digits.length) + digits;
      }
      return padNumeric(sign, digits, spec);
    }
    case "o":
    case "x":
    case "X": {
      const n = Math.trunc(Number(value)) || 0;
      const sign = n < 0 ? "-" : spec.plus ? "+" : spec.space ? " " : "";
      let digits = Math.abs(n).toString(conversion === "o" ? 8 : 16);
      if (conversion === "X") digits = digits.toUpperCase();
      if (spec.hash) digits = conversion === "o" ? `0o${digits}` : `${conversion === "X" ? "0X" : "0x"}${digits}`;
      if (spec.precision !== null && spec.precision > digits.length) {
        digits = "0".repeat(spec.precision - digits.length) + digits;
      }
      return padNumeric(sign, digits, spec);
    }
    case "f":
    case "F":
    case "e":
    case "E":
    case "g":
    case "G": {
      const num = Number(value);
      const negative = num < 0 || Object.is(num, -0);
      const abs = Math.abs(num);
      const precision = spec.precision ?? 6;
      let digits: string;
      if (conversion === "f" || conversion === "F") digits = abs.toFixed(precision);
      else if (conversion === "e" || conversion === "E") {
        digits = normalizeExponent(abs.toExponential(precision), conversion === "E");
      } else {
        digits = formatGeneral(abs, precision, spec.hash, conversion === "G");
      }
      const sign = negative ? "-" : spec.plus ? "+" : spec.space ? " " : "";
      return padNumeric(sign, digits, spec);
    }
    default:
      throw new Error(`unsupported format character '${conversion}'`);
  }
}

function hasPercentMappingKey(format: string): boolean {
  for (let i = 0; i < format.length; i += 1) {
    if (format[i] !== "%") continue;
    i += 1;
    if (format[i] !== "%" && format[i] === "(") return true;
  }
  return false;
}

export function pythonPercentFormat(format: string, values: Any): string {
  const isSequence = Array.isArray(values);
  const args: Any[] = isSequence ? values : [values];
  const mapping: Record<string, Any> | null =
    !isSequence && values !== null && typeof values === "object" && hasPercentMappingKey(format)
      ? values
      : null;
  let argIndex = 0;
  const nextArg = (): Any => {
    if (argIndex >= args.length) throw new Error("not enough arguments for format string");
    return args[argIndex++];
  };
  let out = "";
  let i = 0;
  while (i < format.length) {
    if (format[i] !== "%") {
      out += format[i];
      i += 1;
      continue;
    }
    i += 1;
    if (i >= format.length) throw new Error("incomplete format");
    if (format[i] === "%") {
      out += "%";
      i += 1;
      continue;
    }
    let key: string | null = null;
    if (format[i] === "(") {
      const close = format.indexOf(")", i);
      if (close < 0) throw new Error("incomplete format key");
      key = format.slice(i + 1, close);
      i = close + 1;
    }
    const spec: PercentSpec = {
      minus: false, plus: false, space: false, hash: false, zero: false, width: null, precision: null,
    };
    for (;;) {
      const flag = format[i];
      if (flag === "-") spec.minus = true;
      else if (flag === "+") spec.plus = true;
      else if (flag === " ") spec.space = true;
      else if (flag === "#") spec.hash = true;
      else if (flag === "0") spec.zero = true;
      else break;
      i += 1;
    }
    if (format[i] === "*") {
      spec.width = Number(nextArg());
      i += 1;
    } else {
      const start = i;
      while (i < format.length && format[i]! >= "0" && format[i]! <= "9") i += 1;
      if (i > start) spec.width = Number(format.slice(start, i));
    }
    if (format[i] === ".") {
      i += 1;
      if (format[i] === "*") {
        spec.precision = Number(nextArg());
        i += 1;
      } else {
        const start = i;
        while (i < format.length && format[i]! >= "0" && format[i]! <= "9") i += 1;
        spec.precision = i > start ? Number(format.slice(start, i)) : 0;
      }
    }
    const conversion = format[i] ?? "";
    i += 1;
    let value: Any;
    if (key !== null && mapping !== null) {
      value = mapping[key];
    } else if (mapping !== null) {
      // A mapping operand must be addressed with `%(name)s` on every specifier.
      throw new Error("not enough arguments for format string");
    } else {
      value = nextArg();
    }
    out += formatPercentValue(value, conversion, spec);
  }
  return out;
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

  /**
   * True when `value` is a numeric coordinate vector (a non-empty JS array of
   * numbers). Plain JS arrays double as both Python lists and manim vectors
   * (`RIGHT = [1, 0, 0]`); only the all-numbers case is vector math — a list
   * of Mobjects (or any non-number) follows Python list semantics.
   */
  function isNumericVector(value: unknown): value is readonly number[] {
    return Array.isArray(value) && value.length > 0 && value.every((v) => typeof v === "number");
  }

  /** Python `list * int` repetition (shallow, like CPython). */
  function repeatList(list: readonly Any[], times: number): Any[] {
    const n = Math.trunc(times);
    if (!Number.isFinite(n) || n <= 0) return [];
    const out: Any[] = new Array(list.length * n);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < list.length; j += 1) out[i * list.length + j] = list[j];
    }
    return out;
  }

  /**
   * Concatenate one level of nesting: `[[a, b], [c]]` -> `[a, b, c]`.
   *
   * Target of the multi-`for` comprehension desugar in `preprocess.ts` (which
   * emits `__wan_flatten1(...)` calls because rapydscript-ng only parses a
   * single `for` clause per comprehension). rapydscript lists are plain JS
   * arrays, so this is a verbatim `concat`; a non-array inner value is kept
   * as-is defensively (the desugar only ever passes lists of lists).
   */
  function wanFlatten1(nested: Any): Any[] {
    const out: Any[] = [];
    if (nested === null || nested === undefined) return out;
    const outer = nested as Any;
    if (typeof outer.length !== "number") {
      out.push(nested);
      return out;
    }
    for (let i = 0; i < outer.length; i += 1) {
      const inner = outer[i];
      if (
        inner !== null &&
        inner !== undefined &&
        typeof inner !== "string" &&
        typeof (inner as Any).length === "number"
      ) {
        for (let j = 0; j < (inner as Any).length; j += 1) out.push((inner as Any)[j]);
      } else {
        out.push(inner);
      }
    }
    return out;
  }

  function pythonAdd(a: Any, b: Any): Any {
    if (a !== null && a !== undefined && typeof a.__add__ === "function") return a.__add__(b);
    if (b !== null && b !== undefined && typeof b.__radd__ === "function") return b.__radd__(a);
    if (typeof a === "number" && typeof b === "number") return a + b;
    if (typeof a === "string" || typeof b === "string") return String(a) + String(b);
    if (isArrayLike(a) && isArrayLike(b)) {
      // Python `list + list` concatenates; only all-number vectors add
      // elementwise (e.g. `center + RIGHT * 2`).
      if (!isNumericVector(a) || !isNumericVector(b)) return [...a, ...b];
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
    if (isArrayLike(a) && typeof b === "number") {
      // Python `list * int` repeats; numeric vectors scale.
      if (!isNumericVector(a)) {
        if (!Number.isInteger(b)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
        return repeatList(a, b);
      }
      return a.map((v) => v * b);
    }
    if (typeof a === "number" && isArrayLike(b)) {
      if (!isNumericVector(b)) {
        if (!Number.isInteger(a)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
        return repeatList(b, a);
      }
      return b.map((v) => v * a);
    }
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
    // `"..." % x` is Python's printf-style string formatting, not modulo.
    if (typeof a === "string") return pythonPercentFormat(a, b);
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
   * Round a JS number the way CPython's `round(x, n)` does.
   *
   * Scaling by `10 ** n` is not enough: `2.675 * 100` is exactly `267.5` in
   * binary floating point, which would round the wrong way.  CPython instead
   * rounds the *exact decimal expansion* of the double, half to even.  We get
   * that expansion from `toFixed(100)` and do the rounding in `BigInt`, so the
   * result matches Python even for the classic `round(2.675, 2) == 2.67`.
   */
  function pythonRoundNumber(value: number, n: number): number {
    if (!Number.isFinite(value)) return value;
    if (value === 0) return value;
    const negative = value < 0;
    const abs = Math.abs(value);
    if (Math.abs(value) >= 1e21) {
      // Fixed notation (and therefore `toFixed`) stops at 1e21; fall back to a
      // scaled `Math.round` for these (rare) magnitudes.
      const factor = 10 ** n;
      return Math.round(value * factor) / factor;
    }
    if (n > 100) return value; // rounding below the exact digits cannot change it
    if (n < -323) return value * 0; // `10 ** n` underflows; everything rounds to zero
    const fixed = abs.toFixed(100);
    const dot = fixed.indexOf(".");
    const digits = fixed.slice(0, dot) + fixed.slice(dot + 1); // exact abs * 10^100
    const x = BigInt(digits);
    const d = 100 - n; // we need round(x / 10^d)
    let q: bigint;
    if (d <= 0) {
      q = x * 10n ** BigInt(-d);
    } else {
      const denom = 10n ** BigInt(d);
      q = x / denom;
      const remainder = x % denom;
      const twice = remainder * 2n;
      if (twice > denom) q += 1n;
      else if (twice === denom && q % 2n !== 0n) q += 1n; // exact half -> even
    }
    const result = Number(q) / 10 ** n;
    return negative ? -result : result;
  }

  /**
   * Python `round()`.  JavaScript's `Math.round` is *not* equivalent: CPython
   * rounds halves to the nearest even digit (`round(2.5) == 2`) and accepts a
   * second `ndigits` argument (`round(3.14159, 2)`).  numpy arrays are handed
   * to `np.round` (also half-to-even).
   */
  function pythonRound(value: Any, ndigits?: Any): Any {
    const n = ndigits === undefined || ndigits === null ? 0 : Math.trunc(Number(ndigits));
    if (Array.isArray(value)) return value.map((v) => pythonRound(v, ndigits));
    if (isNdArray(value)) return scalarize(np.round(value, n));
    if (typeof value !== "number") return Math.round(value as number);
    return pythonRoundNumber(value, n);
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
    const explicitCtor = (Ctor as Any).__argnames__;
    if (Array.isArray(explicitCtor)) {
      constructorParamsCache.set(Ctor, [...explicitCtor]);
      return [...explicitCtor];
    }
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
    if (params === null || params.length === 0) return [...positional, normalizeKwargsBag(kwargs)];
    // Many wanim constructors accept a leading options/kwargs bag (e.g.
    // `Arc(radiusOrOptions, ...)`, `Dot(pointOrOptions, ...)`).  When the call is
    // kwargs-only and the first parameter is such a bag, hand it the whole
    // kwargs object so wanim's own `isPlainOptionsBag` path runs.
    const firstIsOptionsBag =
      typeof params[0] === "string" && OPTIONS_PARAM.test(params[0] as string);
    if (positional.length === 0 && firstIsOptionsBag) return [normalizeKwargsBag(kwargs)];
    return placeKwargs(params, positional, kwargs);
  }

  function callWithKwargs(fn: (...a: Any[]) => Any, thisArg: Any, args: Any[]): Any {
    const kwargs = detectKwargs(args[args.length - 1]);
    if (kwargs === null) return fn.apply(thisArg, args);
    const params = parseParamNames(fn);
    const positional = args.slice(0, -1);
    if (params === null || params.length === 0) {
      return fn.apply(thisArg, [...positional, normalizeKwargsBag(kwargs)]);
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
      const wrapped = function (this: Any): Any {
        // Fast path: the trailing argument is not a rapydscript kwargs bag
        // (plain numbers/arrays/Mobjects/...).  `arguments` avoids the rest
        // array every call site used to allocate; only tagged null-prototype
        // bags take the mapping path.
        const count = arguments.length;
        const last: unknown = count > 0 ? arguments[count - 1] : undefined;
        if (last === null || typeof last !== "object" || Object.getPrototypeOf(last) !== null) {
          return original.apply(this, arguments as unknown as Any[]);
        }
        return callWithKwargs(original, this, Array.prototype.slice.call(arguments) as Any[]);
      };
      (wrapped as Any).__wanim_kwargs_wrapped__ = true;
      (wrapped as Any).__wanim_original__ = original;
      // Propagate explicit `__argnames__` so rapydscript's own
      // `ρσ_interpolate_kwargs` (which reads `f.__argnames__` before calling)
      // maps kwargs correctly even when the bundle renamed the parameters.
      if (Array.isArray((original as Any).__argnames__)) {
        (wrapped as Any).__argnames__ = [...(original as Any).__argnames__];
      }
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
    // Python allows unbound calls `Base.method(self, ...)` (e.g.
    // `LinearTransformationScene.setup(self)`). rapydscript compiles them to
    // `Base.method(self, ...)` verbatim, but native instance methods live on
    // `prototype`, so `Base.method` is undefined. Expose every prototype
    // method as a static taking `self` as its first argument. The lookup runs
    // against `Ctor.prototype` at call time, so overrides resolve to this
    // class's implementation (matching CPython's unbound-method semantics),
    // and kwargs bags flow through to the wrapped prototype method untouched.
    // Names already present on the wrapper (including Function/Object members
    // such as `call`/`toString`) are left alone to avoid breaking the function.
    const seenUnbound = new Set<string>();
    let unboundProto: Any = Ctor.prototype;
    while (unboundProto !== null && unboundProto !== undefined && unboundProto !== Object.prototype) {
      for (const key of Object.getOwnPropertyNames(unboundProto)) {
        if (key === "constructor" || key === "__proto__" || seenUnbound.has(key)) continue;
        seenUnbound.add(key);
        const descriptor = Object.getOwnPropertyDescriptor(unboundProto, key);
        if (descriptor === undefined || typeof descriptor.value !== "function") continue;
        if (key in (Wrapper as Any)) continue;
        const methodName = key;
        try {
          Object.defineProperty(Wrapper, methodName, {
            configurable: true,
            writable: true,
            value: function (...args: Any[]) {
              const self = args[0];
              const rest = args.slice(1);
              const fn = (Ctor.prototype as Any)[methodName];
              if (typeof fn !== "function") {
                throw new TypeError(`${(Ctor as Any).name ?? "Class"}.${methodName} is not a function`);
              }
              return fn.apply(self, rest);
            },
          });
        } catch {
          /* ignore read-only statics */
        }
      }
      unboundProto = Object.getPrototypeOf(unboundProto);
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
      "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sqrt", "cbrt",
      "absolute", "abs", "fabs", "sign", "floor", "ceil", "rint", "trunc",
      "square", "reciprocal", "deg2rad", "rad2deg", "degrees", "radians",
      "sinc",
    ];
    for (const name of scalarUfuncs) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      facade[name] = (x: Any, ...rest: Any[]) => {
        const clean = rest.length > 0 && detectKwargs(rest[rest.length - 1]) !== null
          ? rest.slice(0, -1)
          : rest;
        return scalarize((fn as Any)(asNd(x), ...clean));
      };
    }
    // Two-argument ufuncs need the same treatment: numpy-ts requires NDArray
    // inputs, so `np.arctan2(0, 1)` on plain Python scalars throws
    // "undefined is not iterable".  Coerce both operands like the unary case.
    const binaryUfuncs = [
      "arctan2", "atan2", "hypot", "power", "pow", "float_power",
      "maximum", "minimum", "fmax", "fmin", "remainder", "mod", "fmod",
      "copysign", "nextafter", "ldexp", "logaddexp", "logaddexp2",
      "add", "subtract", "multiply", "divide", "true_divide", "floor_divide",
      "heaviside",
    ];
    for (const name of binaryUfuncs) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      facade[name] = (a: Any, b: Any, ...rest: Any[]) => {
        const clean = rest.length > 0 && detectKwargs(rest[rest.length - 1]) !== null
          ? rest.slice(0, -1)
          : rest;
        return scalarize((fn as Any)(asNd(a), asNd(b), ...clean));
      };
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
      exp: Math.exp, exp2: (x) => 2 ** x, expm1: Math.expm1,
      log: Math.log, log2: Math.log2, log10: Math.log10, log1p: Math.log1p,
      sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, absolute: Math.abs,
      fabs: Math.abs,
      sign: Math.sign, floor: Math.floor, ceil: Math.ceil, trunc: Math.trunc,
      square: (x) => x * x, reciprocal: (x) => 1 / x,
      deg2rad: (x) => (x * Math.PI) / 180, rad2deg: (x) => (x * 180) / Math.PI,
      degrees: (x) => (x * 180) / Math.PI, radians: (x) => (x * Math.PI) / 180,
      sinc: (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)),
    };
    for (const name of Object.keys(numberUfuncs)) {
      const fast = numberUfuncs[name];
      const wrapped = facade[name];
      if (typeof wrapped !== "function") continue;
      facade[name] = (x: Any, ...rest: Any[]) =>
        typeof x === "number" && rest.length === 0 ? fast(x) : wrapped(x, ...rest);
    }
    // Plain-number fast path for two-argument ufuncs (same WASM/alloc reason
    // as above).  `remainder` follows Python/numpy sign-of-divisor semantics,
    // while `fmod` keeps C sign-of-dividend semantics (plain JS `%`).
    const numberBinaryUfuncs: Record<string, (a: number, b: number) => number> = {
      arctan2: Math.atan2, atan2: Math.atan2, hypot: Math.hypot,
      power: (a, b) => a ** b, pow: (a, b) => a ** b, float_power: (a, b) => a ** b,
      maximum: Math.max, minimum: Math.min, fmax: Math.max, fmin: Math.min,
      remainder: (a, b) => {
        const r = a % b;
        return r !== 0 && (r < 0 !== b < 0) ? r + b : r;
      },
      mod: (a, b) => {
        const r = a % b;
        return r !== 0 && (r < 0 !== b < 0) ? r + b : r;
      },
      fmod: (a, b) => a % b,
      copysign: (a, b) => (b < 0 || Object.is(b, -0) ? -Math.abs(a) : Math.abs(a)),
      ldexp: (a, b) => a * 2 ** b,
      logaddexp: (a, b) => Math.log(Math.exp(a) + Math.exp(b)),
      logaddexp2: (a, b) => Math.log2(2 ** a + 2 ** b),
      add: (a, b) => a + b, subtract: (a, b) => a - b,
      multiply: (a, b) => a * b, divide: (a, b) => a / b,
      true_divide: (a, b) => a / b, floor_divide: (a, b) => Math.floor(a / b),
      heaviside: (a, b) => (a < 0 ? 0 : a > 0 ? 1 : b),
    };
    for (const name of Object.keys(numberBinaryUfuncs)) {
      const fast = numberBinaryUfuncs[name];
      const wrapped = facade[name];
      if (typeof wrapped !== "function") continue;
      facade[name] = (a: Any, b: Any, ...rest: Any[]) =>
        typeof a === "number" && typeof b === "number" && rest.length === 0
          ? fast(a, b)
          : wrapped(a, b, ...rest);
    }
    // numpy-ts' `round`/`around` never coerce their input, so they throw on
    // Python scalars (and plain lists): `np.round(x, 6)` on a float fails with
    // "undefined is not iterable".  Route both names through Python `round`,
    // which accepts numbers, nested lists and NDArrays and rounds half to even
    // exactly like numpy.
    const numpyRound = (a: Any, decimals: Any = 0): Any => pythonRound(a, decimals);
    facade.round = numpyRound;
    facade.around = numpyRound;
    // Python `np.array(obj, dtype=float)` passes the *type* `float` (rapydscript's
    // `ρσ_float` function) as `dtype`, and rapydscript delivers kwargs as a
    // trailing null-prototype bag.  numpy-ts only accepts dtype *strings* (e.g.
    // `"float64"`) as a positional argument, so a raw passthrough crashes with
    // `TypeError: Cannot convert object to primitive value` (browser WASM build)
    // or `Unknown dtype: [object Object]`.  Normalize both the kwargs bag and a
    // positional Python-type dtype into a numpy-ts dtype string.
    function normalizeNumpyDtype(dtype: Any): string | undefined {
      if (dtype === null || dtype === undefined) return undefined;
      if (typeof dtype === "string") {
        const s = dtype.trim().toLowerCase();
        const alias: Record<string, string> = {
          float64: "float64", f8: "float64", double: "float64", d: "float64",
          float: "float64",
          float32: "float32", f4: "float32", single: "float32",
          float16: "float16", f2: "float16", half: "float16",
          complex128: "complex128", c16: "complex128", complex: "complex128",
          cdouble: "complex128",
          complex64: "complex64", c8: "complex64", csingle: "complex64",
          int64: "int64", i8: "int64", int: "int64", long: "int64",
          int32: "int32", i4: "int32",
          int16: "int16", i2: "int16", short: "int16",
          int8: "int8", i1: "int8", byte: "int8",
          uint64: "uint64", u8: "uint64",
          uint32: "uint32", u4: "uint32",
          uint16: "uint16", u2: "uint16",
          uint8: "uint8", u1: "uint8", ubyte: "uint8",
          bool: "bool", "?": "bool",
        };
        const mapped = alias[s];
        if (mapped !== undefined) return mapped;
        // Accept an already-valid numpy-ts dtype verbatim (e.g. "bool").
        const supported: Any = (np as Any).SUPPORTED_DTYPES;
        if (Array.isArray(supported) && (supported as string[]).includes(s)) return s;
        if (Array.isArray(supported) && (supported as string[]).includes(dtype.trim())) {
          return dtype.trim();
        }
        // Unsupported (e.g. "object", "str"): let numpy-ts infer instead of throwing.
        return undefined;
      }
      if (typeof dtype === "function") {
        const rawName = String((dtype as Any).__name__ ?? dtype.name ?? "").toLowerCase();
        const has = (sub: string): boolean => rawName.includes(sub);
        // Sized variants first, then the generic Python builtins
        // (`ρσ_float`/`ρσ_int`/`ρσ_bool` from rapydscript's baselib).
        if (has("complex64") || has("csingle") || has("c8")) return "complex64";
        if (has("complex")) return "complex128";
        if (has("float16") || has("half") || has("f2")) return "float16";
        if (has("float32") || has("single") || has("f4")) return "float32";
        if (has("float")) return "float64";
        if (has("int8") || has("i1") || has("byte")) return "int8";
        if (has("int16") || has("i2") || has("short")) return "int16";
        if (has("int32") || has("i4")) return "int32";
        if (has("uint8") || has("u1")) return "uint8";
        if (has("uint16") || has("u2")) return "uint16";
        if (has("uint32") || has("u4")) return "uint32";
        if (has("uint64") || has("u8")) return "uint64";
        if (has("int") || has("long")) return "int64";
        if (has("bool")) return "bool";
        if (has("number")) return "float64";
        if (has("boolean")) return "bool";
        // Unknown callable dtype: infer instead of throwing.
        return undefined;
      }
      return undefined;
    }

    function callNumpyWithDtype(fn: (...a: Any[]) => Any, dtypeIndex: number, args: Any[]): Any {
      const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
      const positional = kwargs !== null ? args.slice(0, -1) : args.slice();
      const kwargsDtype = kwargs !== null ? (kwargs as Record<string, Any>)["dtype"] : undefined;
      // Positional dtype (e.g. `np.array(x, float)`) also needs normalization.
      const rawDtype = positional.length > dtypeIndex ? positional[dtypeIndex] : kwargsDtype;
      const normalized = normalizeNumpyDtype(rawDtype);
      if (normalized !== undefined) {
        positional[dtypeIndex] = normalized;
      } else {
        // No (supportable) dtype: drop the dtype slot and any other kwargs
        // (`copy`, `order`, ...) that numpy-ts does not accept.
        if (positional.length > dtypeIndex) positional.length = dtypeIndex;
      }
      return fn(...positional);
    }

    function wrapNumpyCreation(name: string, dtypeIndex: number, shapeFirst = false): void {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") return;
      facade[name] = (...args: Any[]): Any => {
        const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
        let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
        if (shapeFirst && positional.length > 0 && typeof positional[0] === "number") {
          positional = [[positional[0]], ...positional.slice(1)];
        }
        return callNumpyWithDtype(fn, dtypeIndex, kwargs !== null
          ? [...positional, kwargs]
          : positional);
      };
    }

    // numpy-ts requires an array shape; Python accepts an int (`np.zeros(3)`).
    // All of these also accept `dtype` (positionally or as `dtype=...`), which
    // is normalized above (Python types -> numpy-ts strings).
    for (const name of ["zeros", "ones", "empty"]) {
      wrapNumpyCreation(name, 1, true);
    }
    wrapNumpyCreation("full", 2, true);
    for (const name of [
      "array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
      "zeros_like", "ones_like", "empty_like",
    ]) {
      wrapNumpyCreation(name, 1);
    }
    wrapNumpyCreation("full_like", 2);
    for (const name of ["arange", "linspace", "eye", "identity", "logspace", "geomspace"]) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      // `arange`/`linspace`/`eye` take dtype last (index 3); `identity` takes it
      // at index 1.  Derive from the underlying arity instead of hardcoding.
      const dtypeIndex = name === "identity" ? 1 : 3;
      wrapNumpyCreation(name, dtypeIndex);
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
    // Python builtins rapydscript's baselib does not provide.  `abs`, `min`,
    // `max` and `sum` come from baselib; `round` has no JS equivalent that
    // matches CPython's banker's rounding / `ndigits` argument.
    globals.round = pythonRound;
    // One-level flatten for the multi-`for` comprehension desugar (see
    // `desugarMultiForComprehensions` in `preprocess.ts`). The name is
    // reserved: the desugar refuses to run when user code already uses it.
    globals.__wan_flatten1 = wanFlatten1;
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
    pythonRound,
    wanFlatten1,
    getDot,
  };
}
