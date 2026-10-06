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
import { buildPythonItertools } from "./python_itertools.js";
import { buildPythonMath } from "./python_math.js";
import { buildPythonRandom } from "./python_random.js";

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

/**
 * True when `value` is a non-array Python iterable that numpy should accept
 * as an array-like.  rapydscript's `range()` returns such an object (a plain
 * `{start, step, stop}` bag with `Symbol.iterator`, not a JS array), so
 * `np.prod(range(1, k + 1))` -- valid numpy -- reaches the facade as a
 * non-array and numpy-ts throws
 * `Cannot read properties of undefined (reading 'length')`.
 *
 * Strings (which also have `Symbol.iterator`), Maps (which iterate entries)
 * and NDArrays are excluded; wanim Mobjects are excluded by the callers via
 * `isMobjectLike` (they are iterable containers, not numeric data).
 */
function isGenericIterable(value: unknown): value is Iterable<Any> {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return false;
  if (typeof value !== "object" && typeof value !== "function") return false;
  if (value instanceof Map) return false;
  if (Array.isArray(value)) return false;
  return typeof (value as Any)[Symbol.iterator] === "function";
}

/** Return a scalar for 0-d NDArray results, mirroring numpy scalars. */
function scalarize(value: Any): Any {
  if (value === null || value === undefined) return value;
  // NOTE: Mobject instances are Proxies that throw AttributeError on unknown
  // property reads (e.g. `.shape`), so probe defensively: a missing `.shape`
  // means "not a 0-d array", not an error (error 239: `[*vgroup]` spread
  // threw inside `map(scalarize)` and fell through to `Object.keys`).
  let shape: unknown;
  try {
    shape = (value as Any).shape;
  } catch {
    return value;
  }
  if (!Array.isArray(shape) || (shape as unknown[]).length !== 0) return value;
  try {
    const raw = (value as Any).toArray();
    return Array.isArray(raw) ? raw[0] : raw;
  } catch {
    return value;
  }
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
 * Plain-object view of a Python dict.
 *
 * rapydscript compiles `{...}` dict literals to plain JS objects (`{}`), not
 * to its `ρσ_dict` (Map-backed) type -- the latter requires an explicit
 * `from __python__ import dict_literals`, which wanim never enables.  A plain
 * object therefore has no `.items()`/`.keys()`/..., so the extremely common
 * `for k, v in d.items()` fails with `d.items is not a function` (error 156).
 * These helpers implement the read side of the Python dict API over plain
 * objects, plus the write side (`update`/`clear`/`pop`/`setdefault`/`copy`):
 * a `state = {"t1": ..., ...}; state.update(t1=..., ...)` cell -- the standard
 * manim updater pattern -- otherwise fails with
 * `TypeError: Cannot read properties of undefined (reading '__argnames__')`
 * (error 167), because `state.update` resolves to `undefined` and rapydscript
 * still routes the kwargs call through `ρσ_interpolate_kwargs`.
 * They are installed (non-enumerable) on `Object.prototype` so every
 * plain dict literal picks them up.  Only genuine plain objects (prototype is
 * `Object.prototype`) are served -- class instances (Mobjects, ...), arrays
 * and NDArrays keep throwing as before, and real `Map`s delegate to
 * `entries()`.
 */
function isRealMap(value: Any): value is Map<Any, Any> {
  if (value === null || value === undefined) return false;
  if (typeof value !== "object") return false;
  // `instanceof` fails across iframe realms (scenes run in an iframe whose
  // `Map` differs from this realm's), so fall back to the brand string.
  if (value instanceof Map) return true;
  try {
    return Object.prototype.toString.call(value) === "[object Map]";
  } catch {
    return false;
  }
}

function isPlainDictObject(value: Any): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value !== "object") return false;
  if (Array.isArray(value)) return false;
  if (isRealMap(value)) return false;
  // Realm-agnostic plain-object check: a plain dict literal's prototype is
  // some realm's `Object.prototype` (whose own prototype is `null`); a class
  // instance's prototype chain is longer.  Comparing against this realm's
  // `Object.prototype` directly fails for iframe-realm objects.
  let proto: Any;
  try {
    proto = Object.getPrototypeOf(value);
  } catch {
    return false;
  }
  if (proto === null) return true;
  try {
    return Object.getPrototypeOf(proto) === null;
  } catch {
    return false;
  }
}

function restoreDictKey(key: string): string | number {
  // Plain JS objects coerce every key to a string, so `{0: v}` and
  // `{"0": v}` both arrive as `"0"`. Python keeps `0` (int) and `"0"`
  // (str) distinct; the int-key form is far more common (e.g.
  // `parity = {0: 0, 1: 1, 3: 0}` indexed back into a VGroup via
  // `for pos, v in parity.items()`), so restore the numeric form when the
  // key round-trips exactly (`String(Number(k)) === k`). Non-numeric keys
  // (`"a"`, `""`) and non-canonical forms (`"01"`) stay strings.
  if (key === "") return key;
  const n = Number(key);
  if (!Number.isNaN(n) && String(n) === key) return n;
  return key;
}

function dictItemsOf(obj: Any): Array<[string | number, Any]> {
  if (isRealMap(obj)) return [...(obj as Map<Any, Any>).entries()] as Array<[string | number, Any]>;
  if (!isPlainDictObject(obj)) {
    throw new TypeError("dict.items() called on a non-dict object");
  }
  return Object.keys(obj).map((k) => [restoreDictKey(k), (obj as Any)[k]] as [string | number, Any]);
}

function dictKeysOf(obj: Any): Array<string | number> {
  if (isRealMap(obj)) return [...(obj as Map<Any, Any>).keys()] as Array<string | number>;
  if (!isPlainDictObject(obj)) {
    throw new TypeError("dict.keys() called on a non-dict object");
  }
  return Object.keys(obj).map(restoreDictKey);
}

function dictValuesOf(obj: Any): Any[] {
  if (isRealMap(obj)) return [...(obj as Map<Any, Any>).values()];
  if (!isPlainDictObject(obj)) {
    throw new TypeError("dict.values() called on a non-dict object");
  }
  return Object.keys(obj).map((k) => (obj as Any)[k]);
}

/**
 * Entries of a `dict.update(...)` source: a kwargs bag / plain mapping (own
 * enumerable keys), a real `Map`, a rapydscript Map-backed `ρσ_dict`
 * (entries live in `jsmap`), or a list of `[key, value]` pairs.
 */
function dictUpdateEntriesOf(source: Any): Array<[Any, Any]> {
  if (source === null || source === undefined) {
    throw new TypeError("dict.update() argument must be a mapping or an iterable of pairs");
  }
  if (Array.isArray(source)) {
    return (source as Any[]).map((pair) => [(pair as Any)[0], (pair as Any)[1]] as [Any, Any]);
  }
  if (isRealMap(source)) return [...(source as Map<Any, Any>).entries()] as Array<[Any, Any]>;
  const jsmap = (source as Any).jsmap;
  if (jsmap !== undefined && jsmap !== null && typeof jsmap === "object") {
    if (isRealMap(jsmap)) return [...(jsmap as Map<Any, Any>).entries()] as Array<[Any, Any]>;
    if (typeof jsmap.entries === "function") {
      const iterator = jsmap.entries();
      if (iterator !== null && typeof iterator === "object" && typeof iterator.next === "function") {
        const out: Array<[Any, Any]> = [];
        let result = iterator.next();
        while (!result.done) {
          out.push([result.value[0], result.value[1]]);
          result = iterator.next();
        }
        return out;
      }
    }
  }
  if (typeof source === "object" || typeof source === "function") {
    return Object.keys(source)
      .filter((key) => !key.startsWith("kwargs-object-Symbol-"))
      .map((key) => [restoreDictKey(key), (source as Any)[key]] as [Any, Any]);
  }
  throw new TypeError("dict.update() argument must be a mapping or an iterable of pairs");
}

/**
 * Install Python dict-method shims for plain-object dict literals on the
 * given `Object.prototype` (the vm context's own when available).
 *
 * NOTE: `get`/`set`/`value`/`writable`/`enumerable`/`configurable` must NEVER
 * be defined here: `Object.defineProperties` reads those fields (including
 * inherited ones) off every property descriptor, so an inherited `get` turns
 * each `{value: ...}` descriptor into an invalid mixed descriptor and every
 * subsequent `defineProperties` call throws (this broke the rapydscript
 * compiler itself).  Only names none of which `ToPropertyDescriptor` reads
 * (`items`/`keys`/`values`/`__contains__`/`update`/`clear`/`copy`/`pop`/
 * `setdefault`/`set_default`) are installed.
 */
export function installPythonDictShims(ObjectProto: Any): void {
  if (ObjectProto === undefined || ObjectProto === null) return;
  // Clean up the `get` shim installed by the first (broken) revision of this
  // fix: it poisons `Object.defineProperties` via the descriptor protocol
  // (see above) and must not survive in long-lived daemon processes.
  try {
    if (
      Object.prototype.hasOwnProperty.call(ObjectProto, "get") &&
      String((ObjectProto as Any).get).includes("dictGetOf")
    ) {
      delete (ObjectProto as Any).get;
    }
  } catch {
    /* keep going: the `define` calls below still guard per-name */
  }
  // NOTE: no early return on `__wanim_dict_patched__`: the flag only marks
  // that an older (read-side-only) revision ran.  `define` below is
  // per-name idempotent, so re-running it installs methods added later
  // (e.g. `update`) even in long-lived daemon processes whose prototype was
  // patched before they existed.
  try {
    Object.defineProperty(ObjectProto, "__wanim_dict_patched__", {
      configurable: true,
      writable: true,
      enumerable: false,
      value: true,
    });
  } catch {
    return;
  }
  const define = (name: string, value: Any): void => {
    try {
      if (Object.prototype.hasOwnProperty.call(ObjectProto, name)) return;
      Object.defineProperty(ObjectProto, name, {
        configurable: true,
        writable: true,
        enumerable: false,
        value,
      });
    } catch {
      /* read-only prototype: dict literals keep failing loudly */
    }
  };
  define("items", function (this: Any): Array<[string | number, Any]> {
    return dictItemsOf(this);
  });
  define("keys", function (this: Any): Array<string | number> {
    return dictKeysOf(this);
  });
  define("values", function (this: Any): Any[] {
    return dictValuesOf(this);
  });
  define("__contains__", function (this: Any, key: Any): boolean {
    if (isRealMap(this)) return (this as Map<Any, Any>).has(key);
    if (!isPlainDictObject(this)) return false;
    return Object.prototype.hasOwnProperty.call(this, key);
  });
  // The write side of the dict API.  All names here avoid the
  // `ToPropertyDescriptor` fields listed above.  The functions intentionally
  // carry no `__argnames__`, so rapydscript's `ρσ_interpolate_kwargs` hands a
  // `d.update(k=v, ...)` kwargs bag over as a single positional, which
  // `dictUpdateEntriesOf` then merges like any other mapping.
  define("update", function (this: Any): void {
    const sources: Any[] = Array.prototype.slice.call(arguments);
    const merged: Array<[Any, Any]> = [];
    for (const source of sources) merged.push(...dictUpdateEntriesOf(source));
    if (isRealMap(this)) {
      for (const [key, value] of merged) (this as Map<Any, Any>).set(key, value);
      return;
    }
    if (!isPlainDictObject(this)) {
      throw new TypeError("dict.update() called on a non-dict object");
    }
    for (const [key, value] of merged) (this as Any)[key] = value;
  });
  define("clear", function (this: Any): void {
    if (isRealMap(this)) {
      (this as Map<Any, Any>).clear();
      return;
    }
    if (!isPlainDictObject(this)) {
      throw new TypeError("dict.clear() called on a non-dict object");
    }
    for (const key of Object.keys(this)) delete (this as Any)[key];
  });
  define("copy", function (this: Any): Any {
    if (isRealMap(this)) return new Map(this as Map<Any, Any>);
    if (!isPlainDictObject(this)) {
      throw new TypeError("dict.copy() called on a non-dict object");
    }
    return { ...(this as Any) };
  });
  define("pop", function (this: Any, key: Any, defval?: Any): Any {
    if (isRealMap(this)) {
      const map = this as Map<Any, Any>;
      if (map.has(key)) {
        const value = map.get(key);
        map.delete(key);
        return value;
      }
      if (arguments.length > 1) return defval;
      throw new Error(`KeyError: ${String(key)}`);
    }
    if (!isPlainDictObject(this)) {
      throw new TypeError("dict.pop() called on a non-dict object");
    }
    if (Object.prototype.hasOwnProperty.call(this, key)) {
      const value = (this as Any)[key];
      delete (this as Any)[key];
      return value;
    }
    if (arguments.length > 1) return defval;
    throw new Error(`KeyError: ${String(key)}`);
  });
  const setDefault = function (this: Any, key: Any, defval?: Any): Any {
    if (isRealMap(this)) {
      const map = this as Map<Any, Any>;
      if (!map.has(key)) map.set(key, defval);
      return map.get(key);
    }
    if (!isPlainDictObject(this)) {
      throw new TypeError("dict.setdefault() called on a non-dict object");
    }
    if (!Object.prototype.hasOwnProperty.call(this, key)) (this as Any)[key] = defval;
    return (this as Any)[key];
  };
  define("setdefault", setDefault);
  define("set_default", setDefault);
}

/**
 * Stable Python `list.sort(key=..., reverse=...)` served in place.
 *
 * rapydscript's baselib exposes this as `Array.prototype.pysort`, but the
 * compiler still emits a plain `.sort(...)` call: with a `key=` kwarg the
 * call reaches the *native* `Array.prototype.sort` as a single kwargs-bag
 * object, which throws
 * `TypeError: The comparison function must be either a function or undefined`
 * (e.g. `letters.submobjects.sort(key=lambda m: ...)`).  The decorate /
 * sort / undecorate below mirrors `pysort` semantics (stable; `reverse`
 * only flips comparisons, equal keys keep original order) without the
 * dict-keyed lookup `pysort` uses (which collapses duplicate primitives).
 */
function pythonListSortInPlace(arr: Any[], key: Any, reverse: Any, nativeSort: Any): void {
  const hasKey = typeof key === "function";
  const rev = !!reverse;
  const decorated = arr.map((value: Any, index: number) => ({
    value,
    index,
    k: hasKey ? (key as Any)(value) : value,
  }));
  nativeSort.call(decorated, (a: Any, b: Any): number => {
    let cmp = 0;
    try {
      if (a.k < b.k) cmp = -1;
      else if (a.k > b.k) cmp = 1;
    } catch {
      cmp = 0;
    }
    if (cmp !== 0) return rev ? -cmp : cmp;
    return a.index - b.index;
  });
  for (let i = 0; i < arr.length; i += 1) arr[i] = decorated[i]!.value;
}

/**
 * Wrap `Array.prototype.sort` so Python `list.sort(key=..., reverse=...)`
 * calls work while native comparator calls pass through untouched.
 *
 * The wrapper deliberately carries no `__argnames__`, so rapydscript's
 * `ρσ_interpolate_kwargs` hands a `sort(key=..., reverse=...)` kwargs bag
 * over as a single positional, which is intercepted here.  Native uses
 * (`sort()`, `sort(undefined)`, `sort(comparatorFn)`) keep native behavior;
 * a single 1-arity function is treated as a Python `key` (a native
 * comparator takes 2 args), anything else object-shaped with `key`/
 * `reverse` goes through the Python path.  Extra kwargs raise the same
 * `TypeError` CPython raises.
 */
export function installPythonArraySortShim(ArrayProto: Any): void {
  if (ArrayProto === undefined || ArrayProto === null) return;
  try {
    if ((ArrayProto as Any).__wanim_sort_patched__ === true) return;
  } catch {
    return;
  }
  let nativeSort: Any;
  try {
    nativeSort = ArrayProto.sort;
  } catch {
    return;
  }
  if (typeof nativeSort !== "function") return;
  const wrapper = function (this: Any, ...args: Any[]): Any {
    if (!Array.isArray(this)) return nativeSort.apply(this, args);
    // Native comparator fast path (covers internal uses such as
    // `splices.sort((a, b) => ...)`).
    if (args.length === 1 && typeof args[0] === "function") {
      let arity: number = (args[0] as Any).length;
      try {
        const names = (args[0] as Any).__argnames__;
        if (Array.isArray(names)) arity = names.length;
      } catch {
        /* keep .length */
      }
      if (arity >= 2) return nativeSort.apply(this, args);
      // 1-arity function passed positionally: Python `key`, not a
      // comparator (invalid Python3, but `pysort` allows it positionally).
      pythonListSortInPlace(this, args[0], undefined, nativeSort);
      return undefined;
    }
    if (args.length === 0 || (args.length === 1 && args[0] === undefined)) {
      // `sort()` / `sort(undefined)`: Python defaults.  Native string sort
      // would mis-order numbers lexicographically, so serve Python order.
      // Return `this` (like native) so chaining keeps working; Python code
      // ignores the return value either way.
      pythonListSortInPlace(this, undefined, undefined, nativeSort);
      return this;
    }
    if (args.length === 1 && args[0] === null) {
      pythonListSortInPlace(this, undefined, undefined, nativeSort);
      return this;
    }
    // Split a trailing kwargs bag (`sort(key, reverse=True)` arrives as
    // `[keyFn, kwbag]` since the callee has no `__argnames__`).
    let positional: Any[] = args.slice();
    let kwargs: Record<string, Any> | null = null;
    if (positional.length > 0) {
      const maybe = detectKwargs(positional[positional.length - 1]);
      if (maybe !== null) {
        kwargs = maybe;
        positional = positional.slice(0, -1);
      }
    }
    if (kwargs !== null || positional.length > 0) {
      // Defensive cross-realm fallback: a plain `{key, reverse}` object
      // without the kwargs symbol (symbol identity differs per realm).
      if (kwargs === null && positional.length === 1 && positional[0] !== null && typeof positional[0] === "object" && !Array.isArray(positional[0])) {
        try {
          const keys = Object.keys(positional[0]);
          if (keys.length > 0 && keys.every((k) => k === "key" || k === "reverse")) {
            kwargs = {};
            for (const k of keys) (kwargs as Any)[k] = (positional[0] as Any)[k];
            positional = [];
          }
        } catch {
          /* fall through to native */
        }
      }
      if (kwargs !== null || positional.length > 0) {
        let key: Any = positional.length > 0 ? positional[0] : undefined;
        let reverse: Any = positional.length > 1 ? positional[1] : undefined;
        if (positional.length > 2) {
          throw new TypeError(`sort() takes at most 2 positional arguments (${positional.length} given)`);
        }
        if (kwargs !== null) {
          if (positional.length > 0 && ("key" in (kwargs as Any) || "reverse" in (kwargs as Any))) {
            // Both positional and keyword for the same slot: keep CPython's
            // error shape without enumerating every combination.
            if (positional.length > 0 && "key" in (kwargs as Any) && positional[0] !== undefined) {
              throw new TypeError("sort() got multiple values for argument 'key'");
            }
            if (positional.length > 1 && "reverse" in (kwargs as Any) && positional[1] !== undefined) {
              throw new TypeError("sort() got multiple values for argument 'reverse'");
            }
          }
          if ("key" in (kwargs as Any)) key = (kwargs as Any)["key"];
          if ("reverse" in (kwargs as Any)) reverse = (kwargs as Any)["reverse"];
          const unknown = Object.keys(kwargs as Any).filter((k) => k !== "key" && k !== "reverse");
          if (unknown.length > 0) {
            throw new TypeError(`sort() got an unexpected keyword argument '${unknown[0]}'`);
          }
        }
        if (key === null) key = undefined;
        if (key !== undefined && key !== null && typeof key !== "function") {
          throw new TypeError(`'${typeof key}' object is not callable`);
        }
        pythonListSortInPlace(this, key, reverse, nativeSort);
        return undefined;
      }
    }
    return nativeSort.apply(this, args);
  };
  try {
    Object.defineProperty(ArrayProto, "sort", {
      configurable: true,
      writable: true,
      enumerable: false,
      value: wrapper,
    });
    Object.defineProperty(ArrayProto, "__wanim_sort_patched__", {
      configurable: true,
      writable: true,
      enumerable: false,
      value: true,
    });
  } catch {
    /* read-only prototype: Python sort-with-key keeps failing loudly */
  }
}

/**
 * Install Python string-method shims on the vm context's own String.prototype.
 * rapydscript exposes these on its `ρσ_str` function object (not on
 * String.prototype), so `"x".upper()` fails unless we bridge them.
 */
export function installPythonShims(sandbox: Record<string, Any>): void {
  installPythonDictShims(sandbox.Object?.prototype);
  installPythonArraySortShim((sandbox as Any).Array?.prototype);
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
  // Plain-object dict literals (`{"a": 1}` compiles to `{}`) need Python
  // dict methods in every realm that runs translated code (Node unit tests,
  // the browser main window, ...).  The iframe sandbox that actually runs
  // scenes gets its own `Object.prototype` patched via `installPythonShims`.
  try {
    installPythonDictShims((globalThis as Any).Object?.prototype);
  } catch {
    /* non-configurable host: sandbox patch still covers scenes */
  }
  try {
    installPythonArraySortShim((globalThis as Any).Array?.prototype);
  } catch {
    /* non-configurable host: sandbox patch still covers scenes */
  }

  function isNdArray(value: unknown): value is NumpyTS.NDArray {
    if (value === null || value === undefined) return false;
    try {
      if (value instanceof (NDArray as Any)) return true;
    } catch {
      /* cross-realm prototype chain may throw: fall through to duck-typing */
    }
    // Duck-typing fallback: the browser runs translated scenes in an iframe
    // whose `NDArray` (via a second copy of `numpy-ts.browser.js` or a
    // different realm's `Map`/prototype) fails `instanceof`, even though the
    // value is a genuine NDArray (error 311: `d` had `shape=[10]`,
    // `dtype=float64` and `_storage/_base/_raw` keys but `instanceof` was
    // false, so `1j*4*d` fell through to `complexMul` and threw
    // `'complex' and 'object'`).  Genuine NDArrays expose `shape` (array),
    // `dtype` (string) and `tolist` (function); Mobjects are excluded
    // defensively (their Proxy throws on `.shape`, caught below) and plain
    // arrays never have all three.
    try {
      const v = value as Any;
      if (Array.isArray(v.shape) && typeof v.dtype === "string" && typeof v.tolist === "function") {
        return true;
      }
      if (Array.isArray(v.shape) && v._storage !== undefined && v._raw !== undefined) {
        return true;
      }
    } catch {
      return false;
    }
    return false;
  }

  function asNd(value: Any): Any {
    if (isNdArray(value)) return value;
    if (Array.isArray(value) || typeof value === "number") return np.asarray(value);
    // Python complex scalars (`1j`, `complex(...)`) are plain `{re, im}`
    // objects; numpy-ts ufuncs require NDArray inputs and throw
    // "undefined is not iterable" on them (e.g. `np.exp(TAU*1j*f*t)`).
    // numpy-ts' `asarray` understands `{re, im}`, so coerce explicitly.
    if (isComplexLike(value)) {
      try {
        const [re, im] = complexParts(value, "asarray");
        return np.asarray({ re, im });
      } catch {
        return value;
      }
    }
    // rapydscript `range()` objects (and other generic iterables) are valid
    // numpy array-likes; materialize them so ufuncs/arithmetic accept them.
    // Mobjects are iterable containers, not data: leave them untouched.
    if (isGenericIterable(value) && !isMobjectLike(value)) {
      try {
        return np.asarray(deepUnwrapNd(value));
      } catch {
        return value;
      }
    }
    return value;
  }

  /**
   * Recursively convert NDArrays nested in plain JS arrays to plain lists.
   *
   * numpy-ts' `array` infers dtype by inspecting elements as primitives, so
   * `np.array([nd1, nd2])` (a common manim pattern: stacking point vectors)
   * yields `NaN`s instead of a 2-D array.  Unwrap each NDArray via `tolist()`
   * first; plain values pass through untouched.
   */
  function deepUnwrapNd(value: Any): Any {
    if (isNdArray(value)) {
      try {
        const plain = (value as Any).tolist?.();
        if (plain !== undefined) return plain;
      } catch {
        /* fall through to iterator */
      }
      try {
        if (typeof (value as Any)[Symbol.iterator] === "function") {
          return [...(value as Any)].map(deepUnwrapNd);
        }
      } catch {
        /* return as-is */
      }
      return value;
    }
    if (Array.isArray(value)) {
      let dirty = false;
      const out = new Array(value.length);
      for (let i = 0; i < value.length; i += 1) {
        const unwrapped = deepUnwrapNd(value[i]);
        out[i] = unwrapped;
        if (unwrapped !== value[i]) dirty = true;
      }
      return dirty ? out : value;
    }
    // Materialize rapydscript `range()` objects (and other generic iterables)
    // nested anywhere in the input, so `np.array(range(3))` and
    // `np.prod(range(1, k + 1))` see plain lists.  Mobjects are iterable
    // containers, not data: leave them untouched.
    if (isGenericIterable(value) && !isNdArray(value) && !isMobjectLike(value)) {
      try {
        const arr = [...(value as Iterable<Any>)];
        const out = new Array(arr.length);
        for (let i = 0; i < arr.length; i += 1) out[i] = deepUnwrapNd(arr[i]);
        return out;
      } catch {
        return value;
      }
    }
    return value;
  }

  /**
   * `scalarize` plus complex normalization: numpy-ts returns 0-d complex
   * results as bare `{re, im}` objects (no `.real`/`.imag` aliases, no
   * arithmetic dunders), so `p.real` / `p + q` on a `np.exp(complex)` result
   * would fail.  Map such scalars back to `makeComplex`, which carries both
   * spellings and the dunders.  Real scalars and non-0-d arrays pass through
   * untouched.
   */
  function scalarizeComplex(value: Any): Any {
    const s = scalarize(value);
    if (s !== value && isComplexLike(s)) {
      try {
        const [re, im] = complexParts(s, "ufunc");
        return makeComplex(re, im);
      } catch {
        return s;
      }
    }
    return s;
  }

  /**
   * Normalize a Python-side `dtype` (positional or `dtype=...`) into a
   * numpy-ts dtype string.
   *
   * Python `np.array(obj, dtype=float)` passes the *type* `float`
   * (rapydscript's `ρσ_float` function) as `dtype`, and common aliases such
   * as `"float"`/`"int"` are not valid numpy-ts dtypes either.  Returns
   * `undefined` when there is no (supportable) dtype so the caller lets
   * numpy-ts infer instead of throwing `Unknown dtype`.
   */
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

  /**
   * Teach numpy-ts NDArrays the Python call shapes rapydscript emits.
   *
   * 1. `f(*arr, kw=...)` compiles to `arr.concat([kwbag])`, but NDArray has no
   *    `concat`.  Spread via the (working) iterator so the splat rows plus the
   *    trailing kwargs bag reach `ρσ_interpolate_kwargs` like a plain array.
   * 2. `f(*arr)` without kwargs compiles to `f.apply(thisArg, arr)`, which
   *    reads `length` plus numeric indices.  Indices already work through the
   *    Proxy (`arr[0]` is a row view); expose `length` as `shape[0]`.
   * 3. `arr.mean(axis=0)` compiles to `arr.mean(kwbag)`; baselib only maps the
   *    kwargs bag onto positional slots when `f.__argnames__` exists.  Annotate
   *    the reduction methods (nump-ts signatures: `mean(axis?, keepdims?)`,
   *    `percentile(q, axis?, keepdims?)`, ...) so the mapping happens.
   * 4. `arr.astype(float)` passes rapydscript's `ρσ_float` function as dtype
   *    while numpy-ts only accepts dtype strings; normalize to `"float64"`.
   * 5. `arr[::-1]` compiles to `ρσ_eslice(arr, -1)`, whose baselib assumes
   *    JS-array `.slice()/.reverse()/.filter()`.  Serve numeric/no-arg
   *    `.slice()` from a plain-JS copy so the extended-slice path works.
   */
  function patchNDArrayForPython(): void {
    const proto = (NDArray as Any)?.prototype;
    if (proto === undefined || proto === null) return;
    if ((proto as Any).__wanim_python_patched__ === true) return;
    (proto as Any).__wanim_python_patched__ = true;
    if (typeof proto.concat !== "function") {
      Object.defineProperty(proto, "concat", {
        configurable: true,
        writable: true,
        value: function (this: Any, ...args: Any[]): Any[] {
          let rows: Any[];
          try {
            rows = typeof this[Symbol.iterator] === "function" ? [...this] : [];
          } catch {
            rows = [];
          }
          if (rows.length === 0 && Array.isArray(this.shape) && this.shape.length > 0) {
            try {
              rows = Array.from({ length: this.shape[0] as number }, (_, i) => this[i]);
            } catch {
              rows = [];
            }
          }
          return rows.concat(...args);
        },
      });
    }
    if (Object.getOwnPropertyDescriptor(proto, "length") === undefined) {
      try {
        Object.defineProperty(proto, "length", {
          configurable: true,
          get(this: Any): number {
            return Array.isArray(this?.shape) ? (this.shape[0] as number) : 0;
          },
        });
      } catch {
        /* read-only prototype: splats without kwargs keep failing loudly */
      }
    }
    const axisReductions = [
      "sum", "mean", "prod", "max", "min", "ptp", "var", "std",
      "nansum", "nanmean", "nanprod", "nanmin", "nanmax", "nanvar", "nanstd",
      "all", "any", "median", "nanmedian", "cumsum", "cumprod",
      "nancumsum", "nancumprod", "argmin", "argmax", "nanargmin", "nanargmax",
      "sort", "argsort",
    ];
    for (const name of axisReductions) {
      try {
        const fn = proto[name];
        if (typeof fn === "function" && !Array.isArray(fn.__argnames__)) {
          fn.__argnames__ = ["axis", "keepdims", "dtype", "out"];
        }
      } catch {
        /* ignore missing methods across numpy-ts versions */
      }
    }
    for (const name of ["percentile", "quantile", "nanpercentile", "nanquantile"]) {
      try {
        const fn = proto[name];
        if (typeof fn === "function" && !Array.isArray(fn.__argnames__)) {
          fn.__argnames__ = ["q", "axis", "keepdims"];
        }
      } catch {
        /* ignore */
      }
    }
    for (const name of ["partition", "argpartition"]) {
      try {
        const fn = proto[name];
        if (typeof fn === "function" && !Array.isArray(fn.__argnames__)) {
          fn.__argnames__ = ["kth", "axis"];
        }
      } catch {
        /* ignore */
      }
    }
    try {
      if (typeof proto.astype === "function" && !Array.isArray(proto.astype.__argnames__)) {
        proto.astype.__argnames__ = ["dtype", "copy"];
      }
      if (typeof proto.astype === "function" && (proto.astype as Any).__wanim_dtype_wrapped__ !== true) {
        const originalAstype = proto.astype;
        const wrappedAstype = function (this: Any, ...args: Any[]): Any {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          const positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          const kwargsDtype = kwargs !== null ? (kwargs as Record<string, Any>)["dtype"] : undefined;
          const rawDtype = positional.length > 0 ? positional[0] : kwargsDtype;
          const normalized = normalizeNumpyDtype(rawDtype);
          if (normalized !== undefined) {
            positional[0] = normalized;
          } else if (rawDtype !== undefined && rawDtype !== null) {
            // Unsupported dtype (e.g. "object", which numpy-ts cannot
            // represent): keep the current dtype instead of throwing
            // `Unknown dtype`.
            try {
              positional[0] = this.dtype;
            } catch {
              positional.length = 0;
            }
          }
          // `copy` defaults to true in numpy; numpy-ts requires a boolean.
          if (positional.length > 1 && (positional[1] === undefined || positional[1] === null)) {
            positional[1] = true;
          }
          return (originalAstype as Any).apply(this, positional);
        };
        (wrappedAstype as Any).__argnames__ = ["dtype", "copy"];
        (wrappedAstype as Any).__wanim_dtype_wrapped__ = true;
        proto.astype = wrappedAstype;
      }
    } catch {
      /* ignore */
    }
    try {
      if (typeof proto.clip === "function" && !Array.isArray(proto.clip.__argnames__)) {
        proto.clip.__argnames__ = ["a_min", "a_max"];
      }
    } catch {
      /* ignore */
    }
    // rapydscript's `ρσ_eslice` (extended slices such as `arr[::-1]`) assumes
    // JS-array semantics: `arr.slice().reverse()` then numeric
    // `arr.slice(start, end).filter(...)`.  numpy-ts' `.slice` instead takes
    // `"start:stop[:step]"` strings and returns an NDArray without `.reverse`.
    // Serve numeric/no-arg `.slice()` from a plain-JS copy so that path works;
    // string specs keep delegating to the original numpy slicing.  The result
    // is a plain array, which the runtime's arithmetic (`pythonMul`, ...)
    // already handles elementwise.
    try {
      if (typeof proto.slice === "function" && (proto.slice as Any).__wanim_slice_wrapped__ !== true) {
        const originalSlice = proto.slice;
        // Stash the true numpy-ts slice so multi-dim `__wan_getitem` (which
        // passes several numeric indices, e.g. `a[j, i]`) keeps numpy
        // semantics instead of falling into the plain-JS `ρσ_eslice` path
        // below (which would misread `(j, i)` as `rows.slice(j, i)`).
        try {
          if ((proto as Any).__wanim_original_slice__ === undefined) {
            (proto as Any).__wanim_original_slice__ = originalSlice;
          }
        } catch {
          /* ignore */
        }
        const wrappedSlice = function (this: Any, ...args: Any[]): Any {
          const hasStringSpec = args.some((a) => typeof a === "string");
          if (!hasStringSpec) {
            let plain: Any[];
            try {
              if (typeof this.tolist === "function") {
                plain = this.tolist() as Any[];
              } else if (typeof this[Symbol.iterator] === "function") {
                plain = [...this];
              } else {
                return (originalSlice as Any).apply(this, args);
              }
            } catch {
              return (originalSlice as Any).apply(this, args);
            }
            if (args.length === 0) return plain.slice();
            const start = args[0] === undefined || args[0] === null ? undefined : Math.trunc(Number(args[0]));
            const end = args.length < 2 || args[1] === undefined || args[1] === null
              ? undefined
              : Math.trunc(Number(args[1]));
            return plain.slice(start, end);
          }
          return (originalSlice as Any).apply(this, args);
        };
        (wrappedSlice as Any).__wanim_slice_wrapped__ = true;
        proto.slice = wrappedSlice;
      }
    } catch {
      /* ignore */
    }
    // rapydscript's `len()` (`ρσ_len`) only treats plain JS arrays / typed
    // arrays as array-like; an NDArray falls through to
    // `Object.keys(obj).length` (the Proxy's own keys: `_storage`, `_base`,
    // `_raw` -> 3).  That breaks the extremely common `range(len(pts) - 1)`
    // pattern (e.g. Koch snowflake subdivision): `len` returns 3 for a
    // shape-`[2, 3]` array, so `pts[2]` throws
    // `Index 2 is out of bounds for size 2`.  Provide `__len__`, which
    // `ρσ_len` prefers, mirroring `numpy.ndarray.__len__` (`shape[0]`, and
    // `TypeError` for 0-d arrays).
    try {
      if (typeof proto.__len__ !== "function") {
        Object.defineProperty(proto, "__len__", {
          configurable: true,
          writable: true,
          value: function (this: Any): number {
            const shape = this?.shape;
            if (!Array.isArray(shape) || shape.length === 0) {
              throw new TypeError("len() of unsized object");
            }
            return shape[0] as number;
          },
        });
      }
    } catch {
      /* ignore */
    }
    // `ρσ_eslice` with a negative step calls `arr.slice().reverse()`; plain
    // arrays get that from `Array.prototype`, but NDArrays have no `reverse`.
    // Provide one over a plain-JS copy (non-mutating, unlike `Array.reverse`).
    try {
      if (typeof proto.reverse !== "function") {
        Object.defineProperty(proto, "reverse", {
          configurable: true,
          writable: true,
          value: function (this: Any): Any[] {
            try {
              if (typeof this.tolist === "function") return (this.tolist() as Any[]).reverse();
              if (typeof this[Symbol.iterator] === "function") return [...this].reverse();
            } catch {
              /* fall through */
            }
            return [];
          },
        });
      }
    } catch {
      /* ignore */
    }
    // Elementwise `==`/`!=` for the `ρσ_equals`/`ρσ_not_equals` path.
    // `write_smart_equality` emits `(a === b || typeof a === "object" &&
    // ρσ_equals(a, b))`, and `ρσ_equals` prefers `a.__eq__` when present, so
    // defining `__eq__`/`__ne__` here makes `(ndarray == k).sum()` work
    // (error 204) without touching the `==` codegen.  0-d results become
    // plain booleans so `if arr == x:` keeps working.
    const defineComparisonDunder = (
      name: string,
      numpyOp: string,
    ): void => {
      try {
        if (typeof (proto as Any)[name] === "function") return;
        Object.defineProperty(proto, name, {
          configurable: true,
          writable: true,
          value: function (this: Any, other: Any): Any {
            try {
              const fn = (np as Any)[numpyOp];
              if (typeof fn === "function") {
                const res = (fn as Any)(this, asNd(other));
                if (
                  res !== null &&
                  res !== undefined &&
                  Array.isArray((res as Any).shape) &&
                  (res as Any).shape.length === 0
                ) {
                  return Boolean(scalarize(res));
                }
                return res;
              }
            } catch {
              /* fall through to undefined */
            }
            return undefined;
          },
        });
      } catch {
        /* ignore */
      }
    };
    defineComparisonDunder("__eq__", "equal");
    defineComparisonDunder("__ne__", "not_equal");
  }

  /**
   * Expose CPython's `complex.real`/`complex.imag` (and `.conjugate()`) on
   * numpy-ts' `Complex` values.
   *
   * numpy-ts spells the parts `re`/`im` (with a `conj()` method), so
   * `r.real` on an element of `np.roots(...)` (a `Complex` instance) is
   * `undefined` and `float(r.real)` throws `ValueError` (error 271).  Define
   * `real`/`imag` getters aliasing `re`/`im` plus a `conjugate` alias for
   * `conj`, mirroring the `{re, im, real, imag}` shape `makeComplex` already
   * provides for Python-side complex values.
   */
  function patchComplexForPython(): void {
    try {
      const Ctor = (np as Any)?.Complex;
      const proto = Ctor?.prototype;
      if (proto === undefined || proto === null) return;
      if ((proto as Any).__wanim_python_patched__ === true) return;
      (proto as Any).__wanim_python_patched__ = true;
      if (Object.getOwnPropertyDescriptor(proto, "real") === undefined) {
        Object.defineProperty(proto, "real", {
          configurable: true,
          get(this: Any): number {
            return this?.re;
          },
        });
      }
      if (Object.getOwnPropertyDescriptor(proto, "imag") === undefined) {
        Object.defineProperty(proto, "imag", {
          configurable: true,
          get(this: Any): number {
            return this?.im;
          },
        });
      }
      if (typeof proto.conjugate !== "function") {
        Object.defineProperty(proto, "conjugate", {
          configurable: true,
          writable: true,
          value: function (this: Any): Any {
            try {
              if (typeof this?.conj === "function") return this.conj();
            } catch {
              /* fall through to manual construction */
            }
            const C = (np as Any)?.Complex;
            if (typeof C === "function") {
              try {
                return new C(this?.re ?? 0, -(this?.im ?? 0));
              } catch {
                /* fall through */
              }
            }
            return makeComplex(this?.re ?? 0, -(this?.im ?? 0));
          },
        });
      }
    } catch {
      /* keep bare {re, im} semantics when the Complex class is unavailable */
    }
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

  /**
   * CPython-compatible `reversed()`.
   *
   * rapydscript's baselib `ρσ_reversed` only accepts plain JS arrays and
   * strings, while CPython accepts any object with `__reversed__` or the
   * sequence protocol (`__len__` + `__getitem__`) -- notably wanim
   * `Mobject`/`VGroup` instances, which are iterable class objects (see
   * `desugarReversedBuiltin` in `preprocess.ts`, which rewrites bare
   * `reversed(x)` calls to this helper).  Always returns a plain array in
   * reverse order, which the rest of the pipeline (`zip`, `for`, `list`,
   * ...) already handles; this matches observing the reversed iteration
   * order (the CPython iterator itself is not observable from rapydscript).
   */
  function wanReversed(iterable: Any): Any[] {
    if (typeof iterable === "string") return iterable.split("").reverse();
    if (Array.isArray(iterable)) return [...iterable].reverse();
    if (iterable !== null && iterable !== undefined) {
      const obj = iterable as Any;
      // NOTE: wanim `Mobject`s are Proxies whose `get` trap throws
      // `AttributeError` for absent attributes (see `createMobjectProxy`), so
      // dunders must be probed with `in` (the `has` trap), never read blindly.
      const dunder = (name: string): Any => {
        try {
          if (
            (typeof obj === "object" || typeof obj === "function") &&
            name in obj
          ) {
            return obj[name];
          }
        } catch {
          /* hostile Proxy: treat as absent */
        }
        return undefined;
      };
      const reversedMethod = dunder("__reversed__");
      if (typeof reversedMethod === "function") {
        const rev = reversedMethod.call(obj);
        if (Array.isArray(rev)) return [...rev];
        if (rev !== null && rev !== undefined && typeof rev[Symbol.iterator] === "function") {
          return [...(rev as Iterable<Any>)];
        }
      }
      let iteratorMethod: Any;
      try {
        iteratorMethod = obj[Symbol.iterator];
      } catch {
        iteratorMethod = undefined;
      }
      if (typeof iteratorMethod === "function") {
        const items = obj instanceof Map ? [...obj.keys()] : [...(obj as Iterable<Any>)];
        return items.reverse();
      }
      // Sequence-protocol fallback (`__len__` + `__getitem__`); `length` covers
      // the Mobject Proxy's array-like view (`f(*mobject)` splats, see
      // `createMobjectProxy`) and plain array-likes rapydscript cannot see.
      const lenMethod = dunder("__len__");
      let len: Any;
      try {
        len = typeof lenMethod === "function" ? lenMethod.call(obj) : obj.length;
      } catch {
        len = undefined;
      }
      if (typeof len === "number" && Number.isFinite(len)) {
        const n = Math.max(0, Math.trunc(len));
        const getMethod = dunder("__getitem__");
        const get = typeof getMethod === "function"
          ? (index: number) => getMethod.call(obj, index)
          : (index: number) => obj[index];
        const out: Any[] = new Array(n);
        for (let i = 0; i < n; i += 1) out[i] = get(n - 1 - i);
        return out;
      }
    }
    throw new TypeError("argument to reversed() must be a sequence");
  }

  /**
   * CPython-compatible `tuple()`.
   *
   * rapydscript's baselib provides `list`/`dict`/`set`/`str`/... but has no
   * `tuple` builtin, so a bare `tuple(x)` call (e.g.
   * `"q=(%.2f, ...)" % tuple(q)` where `q` is an NDArray row) fails with
   * `ReferenceError: tuple is not defined`.  rapydscript represents both
   * lists and tuples as plain JS arrays, so `tuple()` is `list()` semantics:
   * return a fresh plain array holding the iterable's items.  0-d NDArray
   * elements are scalarized so `"%f" % tuple(row)` formats numbers.
   */
  function pythonTuple(iterable?: Any): Any[] {
    if (iterable === undefined) return [];
    if (Array.isArray(iterable)) return iterable.slice();
    if (typeof iterable === "string") return iterable.split("");
    if (iterable !== null && iterable !== undefined) {
      const obj = iterable as Any;
      let iteratorMethod: Any;
      try {
        iteratorMethod = obj[Symbol.iterator];
      } catch {
        iteratorMethod = undefined;
      }
      if (typeof iteratorMethod === "function") {
        try {
          const items = obj instanceof Map
            ? [...(obj as Map<Any, Any>).keys()]
            : [...(obj as Iterable<Any>)];
          return items.map(scalarize);
        } catch {
          /* fall through to length fallback */
        }
      }
      try {
        const length = obj.length;
        if (typeof length === "number" && Number.isFinite(length)) {
          const n = Math.max(0, Math.trunc(length));
          const out: Any[] = new Array(n);
          for (let i = 0; i < n; i += 1) out[i] = scalarize(obj[i]);
          return out;
        }
      } catch {
        /* fall through to throw */
      }
      if (typeof obj === "object") {
        try {
          // `tuple(dict)` yields its keys; restore numeric keys the same
          // way `dict.keys()` does (plain-object int keys arrive as strings).
          return Object.keys(obj).map(restoreDictKey);
        } catch {
          /* fall through to throw */
        }
      }
    }
    throw new TypeError(`'${typeof iterable}' object is not iterable`);
  }
  (pythonTuple as Any).__argnames__ = ["iterable"];

  /**
   * Spread helper for the `desugarStarredDisplays` step in `preprocess.ts`.
   *
   * rapydscript-ng parses `*args` in calls but rejects `*` inside list/tuple
   * displays (`[*a, b]` fails with "Unexpected token: operator «*»"), so the
   * desugar rewrites `[*a, b, *c]` into
   * `[].concat(__wan_spread(a), [b], __wan_spread(c))` (tuples share the
   * plain-array representation, so the same form covers `(*a, b)`).  This is
   * `tuple(x)` semantics: any iterable becomes a fresh plain array, while a
   * non-iterable raises `TypeError` like CPython's `[*123]`.  Plain
   * `[].concat` always concatenates (unlike wanim's overloaded `+`, which
   * adds length-3 point vectors elementwise), so the desugared form stays
   * exact for numeric vectors too.
   */
  function wanSpread(iterable: Any): Any[] {
    return pythonTuple(iterable);
  }
  (wanSpread as Any).__argnames__ = ["iterable"];

  /**
   * CPython-compatible `sum(iterable, start=0)`.
   *
   * rapydscript's baselib `sum` accumulates with plain JS `+`, so summing
   * numpy values silently produces garbage (`0 + ndarray` coerces the array
   * to a string) instead of an elementwise sum.  This helper accumulates with
   * `pythonAdd` (numpy/complex aware) instead, so
   * `sum(a * np.sin(...) for ...)` over an NDArray argument returns an
   * NDArray like CPython/numpy does (error 270: the baselib version returned
   * a string and the downstream `np.fft.rfft` threw
   * `undefined is not iterable`).  Plain numbers behave exactly like the
   * baselib version, including list concatenation for list terms.
   * Target of the `desugarSumBuiltin` preprocess step (see `preprocess.ts`),
   * which rewrites bare `sum(...)` calls to this helper.
   */
  function pythonSum(iterable: Any, start?: Any): Any {
    if (iterable === undefined) {
      throw new TypeError("sum() missing 1 required positional argument: 'iterable'");
    }
    let acc = start === undefined ? 0 : start;
    const addOne = (value: Any): void => {
      if (typeof value === "string") {
        throw new TypeError("sum() can't sum strings [use ''.join(seq) instead]");
      }
      acc = pythonAdd(acc, value);
    };
    if (iterable !== null && iterable !== undefined) {
      const obj = iterable as Any;
      let iteratorMethod: Any;
      try {
        iteratorMethod = obj[Symbol.iterator];
      } catch {
        iteratorMethod = undefined;
      }
      if (typeof iteratorMethod === "function") {
        // NOTE: only the materialization itself is guarded here: an
        // element-level `addOne` failure (e.g. the string check above) must
        // propagate to the caller, not fall through to another path.
        let items: Any[] | undefined;
        try {
          items = obj instanceof Map
            ? [...(obj as Map<Any, Any>).keys()]
            : [...(obj as Iterable<Any>)];
        } catch {
          items = undefined;
        }
        if (items !== undefined) {
          for (const value of items) addOne(value);
          return acc;
        }
      }
      // `.next()`-only iterators without `Symbol.iterator`.
      if (typeof obj.next === "function") {
        let r = obj.next();
        while (!r.done) {
          addOne(r.value);
          r = obj.next();
        }
        return acc;
      }
      let count: number | undefined;
      try {
        const length = obj.length;
        count = typeof length === "number" && Number.isFinite(length)
          ? Math.max(0, Math.trunc(length))
          : undefined;
      } catch {
        count = undefined;
      }
      if (count !== undefined) {
        for (let i = 0; i < count; i += 1) addOne(obj[i]);
        return acc;
      }
      if (typeof obj === "object") {
        let keys: string[] | undefined;
        try {
          keys = Object.keys(obj);
        } catch {
          keys = undefined;
        }
        if (keys !== undefined) {
          // `sum(dict)` sums its keys; restore numeric keys the same way
          // `dict.keys()` does (plain-object int keys arrive as strings).
          for (const key of keys) addOne(restoreDictKey(key));
          return acc;
        }
      }
    }
    throw new TypeError(`'${typeof iterable}' object is not iterable`);
  }
  (pythonSum as Any).__argnames__ = ["iterable", "start"];

  /**
   * True when an NDArray operand carries complex dtype (numpy-ts stores it as
   * the string `"complex64"`/`"complex128"`).  `isComplexLike` deliberately
   * excludes NDArrays, so `complexArray * 2` (complex NDArray vs real scalar)
   * would otherwise fall through to the buggy numpy-ts multiply path below.
   */
  function isComplexDtypeNd(value: Any): boolean {
    if (!isNdArray(value)) return false;
    try {
      const dtype = (value as Any).dtype;
      return typeof dtype === "string" && dtype.toLowerCase().includes("complex");
    } catch {
      return false;
    }
  }

  function needsComplexArrayOp(a: Any, b: Any): boolean {
    if (!isNdArray(a) && !isNdArray(b)) return false;
    return isComplexLike(a) || isComplexLike(b) || isComplexDtypeNd(a) || isComplexDtypeNd(b);
  }

  function ndToPlainForComplex(value: Any): Any {
    if (isNdArray(value)) {
      try {
        const plain = (value as Any).tolist?.();
        return plain;
      } catch {
        return value;
      }
    }
    return value;
  }

  function complexScalarOp(op: string, x: Any, y: Any): Any {
    if (op === "add") return complexAdd(x, y);
    if (op === "sub") return complexSub(x, y);
    if (op === "mul") return complexMul(x, y);
    if (op === "div") return complexDiv(x, y);
    return complexPow(x, y);
  }

  function broadcastComplexOp(op: string, a: Any, b: Any): Any {
    const aArr = Array.isArray(a);
    const bArr = Array.isArray(b);
    if (aArr && bArr) {
      if (a.length !== b.length) {
        throw new Error("operands could not be broadcast together");
      }
      return a.map((av: Any, i: number) => broadcastComplexOp(op, av, (b as Any[])[i]));
    }
    if (aArr) return (a as Any[]).map((av: Any) => broadcastComplexOp(op, av, b));
    if (bArr) return (b as Any[]).map((bv: Any) => broadcastComplexOp(op, a, bv));
    return complexScalarOp(op, a, b);
  }

  /**
   * Elementwise complex arithmetic involving at least one NDArray.
   *
   * numpy-ts' `multiply` (and friends) mis-broadcast a 0-d complex scalar
   * against a vector (`np.multiply(asarray({re:0,im:4}), asarray([0.5,1]))`
   * yields `NaN`s -- error 311: `1j * 4.0 * d` where `d` is shape-`[n]`),
   * so the `complexMul` scalar path above cannot serve it either (it throws
   * `unsupported operand ... 'complex' and 'object'`).  Unwrap NDArrays via
   * `tolist()`, apply the scalar complex op with NumPy-style scalar
   * broadcasting in plain JS, and wrap back with `asarray` (which infers
   * `complex128` from `{re, im}` elements).  Returns `undefined` when the
   * shapes do not broadcast, letting the caller fall through.
   */
  function tryComplexArrayOp(op: string, a: Any, b: Any): Any | undefined {
    try {
      const plainA = ndToPlainForComplex(a);
      const plainB = ndToPlainForComplex(b);
      if (!Array.isArray(plainA) && !Array.isArray(plainB)) {
        return complexScalarOp(op, plainA, plainB);
      }
      const nested = broadcastComplexOp(op, plainA, plainB);
      try {
        return (np as Any).asarray(nested);
      } catch {
        return undefined;
      }
    } catch {
      return undefined;
    }
  }

  function pythonAdd(a: Any, b: Any): Any {
    if (needsComplexArrayOp(a, b)) {
      const r = tryComplexArrayOp("add", a, b);
      if (r !== undefined) return r;
    }
    if (isComplexLike(a) || isComplexLike(b)) return complexAdd(a, b);
    if (a !== null && a !== undefined && typeof a.__add__ === "function") return a.__add__(b);
    if (b !== null && b !== undefined && typeof b.__radd__ === "function") return b.__radd__(a);
    if (typeof a === "number" && typeof b === "number") return a + b;
    if (typeof a === "string" || typeof b === "string") return String(a) + String(b);
    if (isArrayLike(a) && isArrayLike(b)) {
      // Python `list + list` concatenates. Plain JS arrays double as manim
      // point vectors (e.g. `center + RIGHT * 2`, where `get_center()`
      // returns a plain `number[]` in wanim but an ndarray in Python), so a
      // narrow elementwise exception exists for same-length length-3 numeric
      // vectors only. Every other shape concatenates: notably `used = used +
      // [n]` (length-1 or differing lengths) must grow the list instead of
      // doing vector math (error 186: `[1] + [3]` became `[4]`, poisoning
      // downstream `(-1) ** 1.5` into NaN and `float(NaN)` into ValueError).
      if (
        isNumericVector(a) &&
        isNumericVector(b) &&
        a.length === b.length &&
        a.length === 3
      ) {
        const r = elementwise(a, b, (x, y) => x + y);
        if (r !== undefined) return r;
        return scalarize(np.add(asNd(a), asNd(b)));
      }
      return [...a, ...b];
    }
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v + b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => v + a);
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.add(asNd(a), asNd(b)));
    return a + b;
  }

  function pythonSub(a: Any, b: Any): Any {
    if (needsComplexArrayOp(a, b)) {
      const r = tryComplexArrayOp("sub", a, b);
      if (r !== undefined) return r;
    }
    if (isComplexLike(a) || isComplexLike(b)) return complexSub(a, b);
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
    if (needsComplexArrayOp(a, b)) {
      const r = tryComplexArrayOp("mul", a, b);
      if (r !== undefined) return r;
    }
    if (isComplexLike(a) || isComplexLike(b)) return complexMul(a, b);
    if (a !== null && a !== undefined && typeof a.__mul__ === "function") return a.__mul__(b);
    if (b !== null && b !== undefined && typeof b.__rmul__ === "function") return b.__rmul__(a);
    if (typeof a === "number" && typeof b === "number") return a * b;
    if (typeof a === "string" && typeof b === "number") return a.repeat(b);
    if (typeof a === "number" && typeof b === "string") return b.repeat(a);
    if (isArrayLike(a) && typeof b === "number") {
      // Python `list * int` repeats; only 3D manim point vectors scale
      // (mirrors pythonAdd's length-3 rule). Plain JS arrays double as both,
      // so `[0] * 11` must repeat to 11 zeros instead of scaling to `[0]`
      // (error 281: heights stayed length-1, heights[k] was undefined, and
      // axes.c2p(k, NaN) poisoned move_to).
      if (isNumericVector(a) && a.length === 3) return a.map((v) => v * b);
      if (!Number.isInteger(b)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
      return repeatList(a, b);
    }
    if (typeof a === "number" && isArrayLike(b)) {
      if (isNumericVector(b) && b.length === 3) return b.map((v) => v * a);
      if (!Number.isInteger(a)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
      return repeatList(b, a);
    }
    if (isArrayLike(a) && isArrayLike(b)) {
      const r = elementwise(a, b, (x, y) => x * y);
      if (r !== undefined) return r;
    }
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.multiply(asNd(a), asNd(b)));
    return a * b;
  }

  function pythonDiv(a: Any, b: Any): Any {
    if (needsComplexArrayOp(a, b)) {
      const r = tryComplexArrayOp("div", a, b);
      if (r !== undefined) return r;
    }
    if (isComplexLike(a) || isComplexLike(b)) return complexDiv(a, b);
    if (a !== null && a !== undefined && typeof a.__truediv__ === "function") return a.__truediv__(b);
    if (b !== null && b !== undefined && typeof b.__rtruediv__ === "function") return b.__rtruediv__(a);
    if (typeof a === "number" && typeof b === "number") return a / b;
    if (isArrayLike(a) && typeof b === "number") return a.map((v) => v / b);
    if (typeof a === "number" && isArrayLike(b)) return b.map((v) => a / v);
    if (isNdArray(a) || isNdArray(b)) return scalarize(np.divide(asNd(a), asNd(b)));
    return a / b;
  }

  function pythonFloorDiv(a: Any, b: Any): Any {
    // CPython: `complex // x` raises (there is no complex floor division).
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("unsupported operand type(s) for //: 'complex'");
    }
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
    // CPython: `complex % x` raises (there is no complex modulo).
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("can't mod complex numbers.");
    }
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
    if (needsComplexArrayOp(a, b)) {
      const r = tryComplexArrayOp("pow", a, b);
      if (r !== undefined) return r;
    }
    if (isComplexLike(a) || isComplexLike(b)) return complexPow(a, b);
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
    if (isComplexLike(a)) return complexNeg(a);
    if (a !== null && a !== undefined && typeof a.__neg__ === "function") return a.__neg__();
    if (typeof a === "number") return -a;
    if (isArrayLike(a)) return a.map((v) => -v);
    if (isNdArray(a)) return scalarize(np.negative(a));
    return -a;
  }

  // --- Python comparisons (numpy elementwise) ------------------------------
  //
  // rapydscript emits `<`/`>`/`<=`/`>=` as plain JS operators, which cannot
  // dispatch on numpy arrays: `(np.random.random((n, m)) < p)` coerces the
  // NDArray to a primitive (`false`) so `.sum()` fails with
  // "is not a function" (error 204).  The patched compiler routes these four
  // operators through `ρσ_op_lt`/`ρσ_op_gt`/`ρσ_op_le`/`ρσ_op_ge` when
  // `operator_overloading` is on (see `py2js/patches/rapydscript-ng.patch`);
  // these helpers implement Python semantics with a numpy fast path:
  //
  //   - complex operands raise like CPython (no ordering on complex);
  //   - `__lt__`/`__gt__`/`__le__`/`__ge__` dunders are honored first;
  //   - when either side is an NDArray, delegate to numpy-ts
  //     (`less`/`greater`/`less_equal`/`greater_equal`) with broadcasting,
  //     returning the boolean NDArray (0-d results become plain booleans);
  //   - otherwise fall back to plain JS `<`/`>`/`<=`/`>=` so numbers, strings
  //     and existing non-numpy code behave exactly as before.
  //
  // `==`/`!=` keep their existing `ρσ_equals` path; NDArray elementwise
  // equality for that path is provided via `__eq__`/`__ne__` on the NDArray
  // prototype in `patchNDArrayForPython` below.
  function comparisonNumpy(op: string, a: Any, b: Any): Any | undefined {
    if (!isNdArray(a) && !isNdArray(b)) return undefined;
    const fn = (np as Any)[op];
    if (typeof fn !== "function") return undefined;
    const res = (fn as Any)(asNd(a), asNd(b));
    if (
      res !== null &&
      res !== undefined &&
      Array.isArray((res as Any).shape) &&
      (res as Any).shape.length === 0
    ) {
      return Boolean(scalarize(res));
    }
    return res;
  }

  function pythonLt(a: Any, b: Any): Any {
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("'<' not supported between instances of 'complex'");
    }
    if (a !== null && a !== undefined && typeof a.__lt__ === "function") return a.__lt__(b);
    if (b !== null && b !== undefined && typeof b.__gt__ === "function") return b.__gt__(a);
    const viaNumpy = comparisonNumpy("less", a, b);
    if (viaNumpy !== undefined) return viaNumpy;
    return a < b;
  }

  function pythonGt(a: Any, b: Any): Any {
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("'>' not supported between instances of 'complex'");
    }
    if (a !== null && a !== undefined && typeof a.__gt__ === "function") return a.__gt__(b);
    if (b !== null && b !== undefined && typeof b.__lt__ === "function") return b.__lt__(a);
    const viaNumpy = comparisonNumpy("greater", a, b);
    if (viaNumpy !== undefined) return viaNumpy;
    return a > b;
  }

  function pythonLe(a: Any, b: Any): Any {
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("'<=' not supported between instances of 'complex'");
    }
    if (a !== null && a !== undefined && typeof a.__le__ === "function") return a.__le__(b);
    if (b !== null && b !== undefined && typeof b.__ge__ === "function") return b.__ge__(a);
    const viaNumpy = comparisonNumpy("less_equal", a, b);
    if (viaNumpy !== undefined) return viaNumpy;
    return a <= b;
  }

  function pythonGe(a: Any, b: Any): Any {
    if (isComplexLike(a) || isComplexLike(b)) {
      throw new TypeError("'>=' not supported between instances of 'complex'");
    }
    if (a !== null && a !== undefined && typeof a.__ge__ === "function") return a.__ge__(b);
    if (b !== null && b !== undefined && typeof b.__le__ === "function") return b.__le__(a);
    const viaNumpy = comparisonNumpy("greater_equal", a, b);
    if (viaNumpy !== undefined) return viaNumpy;
    return a >= b;
  }

  // --- Python complex ------------------------------------------------------
  //
  // rapydscript-ng has no complex-number support: its tokenizer rejects the
  // `j`-suffixed literal ("Invalid syntax: 0j") and the baselib defines no
  // `complex` builtin.  `desugarComplexLiterals` (see `preprocess.ts`)
  // rewrites every decimal `...j` literal into `__wan_complex(0, imag)` before
  // compilation; this section implements the runtime side:
  //
  //   - `pythonComplex(...)` (also exposed as the `complex` and
  //     `__wan_complex` globals), mirroring CPython's
  //     `complex([real[, imag]])` including the string form
  //     (`complex("1+2j")`, `complex("3j")`, `complex("2")`);
  //   - an immutable `{ re, im, real, imag, ...dunders }` representation:
  //     `re`/`im` match wanim's `ComplexPoint` (so `complex_to_R3(z)` keeps
  //     working) while `real`/`imag` match CPython's attribute names (so
  //     `z.real` / `z.imag` keep working);
  //   - explicit complex branches in the numeric operators below, so plain
  //     `{ re, im }` points (e.g. from `ComplexPlane.point_to_number`, which
  //     carry no methods) interoperate with real numbers and
  //     method-carrying complex values alike.
  //
  // Known limitations (documented in ): `==`/`!=` on
  // complex values compare object identity, not value equality.

  /**
   * True when `value` is a complex number: either a value created by
   * `makeComplex`/`pythonComplex` (marked) or any plain `{ re, im }` /
   * `{ real, imag }` pair such as wanim's `ComplexPoint`.  Arrays and
   * NDArrays are never complex, even though an NDArray row view could
   * duck-type `{ re, im }` through its Proxy.
   */
  function isComplexLike(value: unknown): boolean {
    if (value === null || value === undefined || typeof value !== "object") return false;
    if (Array.isArray(value) || isNdArray(value)) return false;
    const obj = value as Record<string, unknown>;
    if (obj.__wan_complex__ === true) return true;
    if (typeof obj.re === "number" && typeof obj.im === "number") return true;
    if (typeof obj.real === "number" && typeof obj.imag === "number") return true;
    return false;
  }

  /** Split a numeric-ish value into `[re, im]`; throws `TypeError` otherwise. */
  function complexParts(value: Any, op: string): [number, number] {
    if (typeof value === "number") return [value, 0];
    if (typeof value === "boolean") return [value ? 1 : 0, 0];
    if (isNdArray(value)) {
      const scalar = scalarize(value);
      if (scalar !== value) return complexParts(scalar, op);
    }
    if (isComplexLike(value)) {
      const obj = value as Any;
      const re = typeof obj.re === "number" ? obj.re : obj.real;
      const im = typeof obj.im === "number" ? obj.im : obj.imag;
      return [re, im];
    }
    throw new TypeError(`unsupported operand type(s) for ${op}: 'complex' and '${typeof value}'`);
  }

  function complexAdd(a: Any, b: Any): Any {
    const [ar, ai] = complexParts(a, "+");
    const [br, bi] = complexParts(b, "+");
    return makeComplex(ar + br, ai + bi);
  }

  function complexSub(a: Any, b: Any): Any {
    const [ar, ai] = complexParts(a, "-");
    const [br, bi] = complexParts(b, "-");
    return makeComplex(ar - br, ai - bi);
  }

  function complexMul(a: Any, b: Any): Any {
    const [ar, ai] = complexParts(a, "*");
    const [br, bi] = complexParts(b, "*");
    return makeComplex(ar * br - ai * bi, ar * bi + ai * br);
  }

  function complexDiv(a: Any, b: Any): Any {
    const [ar, ai] = complexParts(a, "/");
    const [br, bi] = complexParts(b, "/");
    const denom = br * br + bi * bi;
    if (denom === 0) {
      const error = new Error("complex division by zero");
      error.name = "ZeroDivisionError";
      throw error;
    }
    return makeComplex(
      (ar * br + ai * bi) / denom,
      (ai * br - ar * bi) / denom,
    );
  }

  function complexPow(a: Any, b: Any): Any {
    const [ar, ai] = complexParts(a, "**");
    const [br, bi] = complexParts(b, "**");
    if (ar === 0 && ai === 0) {
      if (br === 0 && bi === 0) return makeComplex(1, 0);
      if (br > 0) return makeComplex(0, 0);
      const error = new Error("0.0 to a negative or complex power");
      error.name = "ZeroDivisionError";
      throw error;
    }
    // `a ** b == exp(b * log(a))`, with `log` in polar form.
    const radius = Math.hypot(ar, ai);
    const theta = Math.atan2(ai, ar);
    const logRadius = Math.log(radius);
    const magnitude = Math.exp(br * logRadius - bi * theta);
    const angle = br * theta + bi * logRadius;
    return makeComplex(magnitude * Math.cos(angle), magnitude * Math.sin(angle));
  }

  function complexNeg(a: Any): Any {
    const [re, im] = complexParts(a, "unary -");
    return makeComplex(-re, -im);
  }

  /** Create an immutable complex value (see above for the representation). */
  function makeComplex(re: number, im: number): Any {
    const obj: Record<string, Any> = {
      re,
      im,
      real: re,
      imag: im,
      __wan_complex__: true,
    };
    obj.__add__ = (other: Any) => complexAdd(obj, other);
    obj.__radd__ = (other: Any) => complexAdd(other, obj);
    obj.__sub__ = (other: Any) => complexSub(obj, other);
    obj.__rsub__ = (other: Any) => complexSub(other, obj);
    obj.__mul__ = (other: Any) => complexMul(obj, other);
    obj.__rmul__ = (other: Any) => complexMul(other, obj);
    obj.__truediv__ = (other: Any) => complexDiv(obj, other);
    obj.__rtruediv__ = (other: Any) => complexDiv(other, obj);
    obj.__pow__ = (other: Any) => complexPow(obj, other);
    obj.__rpow__ = (other: Any) => complexPow(other, obj);
    obj.__neg__ = () => makeComplex(-re, -im);
    obj.conjugate = () => makeComplex(re, -im);
    return obj;
  }

  /** Parse one float component of `complex(text)` (allows `inf`/`nan`). */
  function parseComplexComponent(part: string, full: string, imag: boolean): number {
    const fail = (): never => {
      const error = new Error(`complex() arg is a malformed string: ${full}`);
      error.name = "ValueError";
      throw error;
    };
    if (imag && (part === "" || part === "+")) return 1;
    if (imag && part === "-") return -1;
    if (part === "") fail();
    const lower = part.toLowerCase();
    if (lower === "inf" || lower === "+inf" || lower === "infinity" || lower === "+infinity") {
      return Infinity;
    }
    if (lower === "-inf" || lower === "-infinity") return -Infinity;
    if (lower === "nan" || lower === "+nan" || lower === "-nan") return NaN;
    const value = Number(part);
    if (Number.isNaN(value)) fail();
    return value;
  }

  /**
   * Parse the full text of `complex(text)`, mirroring CPython: optional
   * whitespace, an optional real part, and an optional imaginary part
   * suffixed with `j`/`J` (a bare `j` means `1j`).  Single underscores
   * between digits are accepted like CPython.
   */
  function parseComplexText(text: string): [number, number] {
    const fail = (): never => {
      const error = new Error(`complex() arg is a malformed string: ${text}`);
      error.name = "ValueError";
      throw error;
    };
    const compact = String(text).replace(/\s+/g, "").replace(/(?<=\d)_(?=\d)/g, "");
    if (compact === "") fail();
    const last = compact[compact.length - 1]!;
    if (last !== "j" && last !== "J") {
      return [parseComplexComponent(compact, text, false), 0];
    }
    const body = compact.slice(0, -1);
    // Split the body into real/imaginary at the last interior `+`/`-` that is
    // not part of an exponent (mirrors how Python tokenizes `1e-3+2j`).
    let split = -1;
    for (let k = body.length - 1; k > 0; k -= 1) {
      const c = body[k]!;
      if ((c === "+" || c === "-") && body[k - 1]!.toLowerCase() !== "e") {
        split = k;
        break;
      }
    }
    if (split === -1) return [0, parseComplexComponent(body, text, true)];
    return [
      parseComplexComponent(body.slice(0, split), text, false),
      parseComplexComponent(body.slice(split), text, true),
    ];
  }

  /**
   * CPython's `complex([real[, imag]])`.
   *
   * `complex(x, y)` is `x + y*1j`, so a complex first argument contributes
   * its parts (`complex(1+2j, 3) == (1+5j)`); a string first argument forbids
   * the second one.  0-d NDArrays unwrap to their scalar like the ufunc
   * wrappers do.
   */
  function pythonComplex(real?: Any, imag?: Any): Any {
    if (typeof real === "string") {
      if (imag !== undefined) {
        throw new TypeError("complex() can't take second arg if first is a string");
      }
      const [re, im] = parseComplexText(real);
      return makeComplex(re, im);
    }
    if (real === undefined) {
      if (imag === undefined) return makeComplex(0, 0);
      throw new TypeError("complex() first argument must be a string or a number, not 'NoneType'");
    }
    if (real === null) {
      throw new TypeError("complex() first argument must be a string or a number, not 'NoneType'");
    }
    const [ar, ai] = complexParts(real, "complex()");
    if (imag === undefined) return makeComplex(ar, ai);
    if (typeof imag === "string") {
      throw new TypeError("complex() second argument must be a number");
    }
    const [br, bi] = complexParts(imag, "complex()");
    return makeComplex(ar - bi, ai + br);
  }

  /**
   * Factory target of the `desugarComplexLiterals` preprocess step (see
   * `preprocess.ts`): `0j` becomes `__wan_complex(0, 0)`.  Always called with
   * two real numbers; behaves like `complex(real, imag)` otherwise.
   */
  function wanComplex(real?: Any, imag?: Any): Any {
    return pythonComplex(real, imag);
  }

  /**
   * PEP 465 matrix multiplication (`a @ b`).
   *
   * Target of the `desugarMatmul` preprocess step (see `preprocess.ts`),
   * because rapydscript-ng only accepts `@` as a decorator.  Dunder dispatch
   * mirrors CPython (`__matmul__` first, then the reflected `__rmatmul__`);
   * plain arrays and NDArrays go through `np.matmul` like every other
   * numeric operator goes through its numpy-ts ufunc.
   */
  function wanMatmul(a: Any, b: Any): Any {
    const matmul = (a as Any)?.__matmul__;
    if (typeof matmul === "function") return matmul.call(a, b);
    const rmatmul = (b as Any)?.__rmatmul__;
    if (typeof rmatmul === "function") return rmatmul.call(b, a);
    const fn = (np as Any).matmul;
    if (typeof fn !== "function") {
      throw new Error("matmul is not available in this numpy build");
    }
    return scalarize(fn(asNd(a), asNd(b)));
  }

  // --- numpy multi-dim subscripts (`a[:, 1]`) -------------------------------
  //
  // rapydscript-ng only parses a single index/slice per `[...]`, so
  // `desugarNumpySubscripts` (see `preprocess.ts`) rewrites `a[:, 1]` into
  // `__wan_getitem(a, __wan_slice(None, None), 1)`.  These are the runtime
  // targets of that desugar.  Single subscripts never reach this path
  // (rapydscript compiles them directly via `ρσ_getitem`/`ρσ_getslice`).

  /** Render one slice bound the way numpy-ts' string slices expect. */
  function wanSliceBound(value: Any): string {
    if (value === null || value === undefined) return "";
    if (typeof value === "boolean") return value ? "1" : "0";
    if (typeof value === "number") return String(value);
    if (isNdArray(value)) return wanSliceBound(scalarize(value));
    const n = Number(value);
    if (Number.isNaN(n)) {
      throw new TypeError("slice indices must be integers or None");
    }
    return String(Math.trunc(n));
  }

  /** Build one `"start:stop[:step]"` dimension for `__wan_getitem`. */
  function wanSliceDim(start: Any, stop: Any, step?: Any): string {
    const base = `${wanSliceBound(start)}:${wanSliceBound(stop)}`;
    return step === undefined || step === null
      ? base
      : `${base}:${wanSliceBound(step)}`;
  }

  /** Normalize a step-less/stepped slice string against `length`. */
  function parseSliceParts(spec: string): [string, string, string] {
    const parts = spec.split(":");
    return [
      parts.length > 0 ? (parts[0] ?? "") : "",
      parts.length > 1 ? (parts[1] ?? "") : "",
      parts.length > 2 ? (parts[2] ?? "") : "",
    ];
  }

  function parseSliceInt(text: string): number | null {
    if (text === "") return null;
    const n = Number(text);
    return Number.isNaN(n) ? null : Math.trunc(n);
  }

  /**
   * Python `slice.indices(length)` expansion for one dimension: returns the
   * concrete indices a `"start:stop[:step]"` string selects.
   */
  function sliceIndices(length: number, spec: string): number[] {
    const [startText, stopText, stepText] = parseSliceParts(spec);
    const step = stepText === "" ? 1 : parseSliceInt(stepText);
    if (step === null) throw new TypeError("slice indices must be integers or None");
    if (step === 0) throw new Error("slice step cannot be zero");
    const start = parseSliceInt(startText);
    const stop = parseSliceInt(stopText);
    if (step > 0) {
      let lo = start === null ? 0 : start < 0 ? Math.max(0, start + length) : Math.min(length, start);
      let hi = stop === null ? length : stop < 0 ? Math.max(0, stop + length) : Math.min(length, stop);
      const out: number[] = [];
      for (let k = lo; k < hi; k += step) out.push(k);
      return out;
    }
    let lo = start === null ? length - 1 : start < 0 ? Math.max(-1, start + length) : Math.min(length - 1, start);
    let hi = stop === null ? -1 : stop < 0 ? Math.max(-1, stop + length) : Math.min(length, stop);
    const out: number[] = [];
    for (let k = lo; k > hi; k += step) out.push(k);
    return out;
  }

  /**
   * Multi-dim getitem over nested plain-JS lists (numpy-style): the first spec
   * selects along the outer axis and the rest recurses into each selected row,
   * so `[[1, 2], [3, 4]][:, 1]` is the column `[2, 4]`.
   */
  function wanGetItemArray(arr: Any[], specs: Any[]): Any {
    if (specs.length === 0) return arr;
    const [first, ...rest] = specs as [Any, ...Any[]];
    if (typeof first === "number" || typeof first === "boolean") {
      let index =
        typeof first === "boolean" ? (first ? 1 : 0) : Math.trunc(first);
      if (index < 0) index += arr.length;
      const selected = arr[index];
      if (rest.length === 0) return selected;
      if (!Array.isArray(selected)) {
        throw new TypeError("'int' object is not subscriptable");
      }
      return wanGetItemArray(selected, rest);
    }
    if (typeof first === "string") {
      let rows: Any[];
      if (first === ":" || first === "..." || first === "") {
        rows = arr.slice();
      } else if (!first.includes(":")) {
        const index = parseSliceInt(first);
        if (index === null) {
          throw new TypeError("slice indices must be integers or None");
        }
        const selected = arr[index < 0 ? index + arr.length : index];
        if (rest.length === 0) return selected;
        if (!Array.isArray(selected)) {
          throw new TypeError("'int' object is not subscriptable");
        }
        return wanGetItemArray(selected, rest);
      } else {
        rows = sliceIndices(arr.length, first).map((k) => arr[k]);
      }
      if (rest.length === 0) return rows;
      return rows.map((row) => {
        if (!Array.isArray(row)) {
          throw new TypeError("'int' object is not subscriptable");
        }
        return wanGetItemArray(row, rest);
      });
    }
    if (first === null || first === undefined) {
      // `None` (numpy `newaxis`): wrap, then apply the rest inside.
      return rest.length === 0 ? [arr] : wanGetItemArray([arr], rest);
    }
    throw new TypeError("invalid index type");
  }

  /**
   * Multi-dim getitem target of the `desugarNumpySubscripts` preprocess step.
   * Each spec is a number (integer index), a `"start:stop[:step]"` slice
   * string (see `wanSliceDim`), or `"..."` (Ellipsis).
   */
  function wanGetItem(obj: Any, ...specs: Any[]): Any {
    if (obj === null || obj === undefined) {
      throw new TypeError("'NoneType' object is not subscriptable");
    }
    if (isNdArray(obj)) {
      const ndim = (obj.shape as readonly number[]).length;
      const dot = specs.indexOf("...");
      let finalSpecs = specs;
      if (dot !== -1) {
        const fill = Math.max(0, ndim - (specs.length - 1));
        finalSpecs = [
          ...specs.slice(0, dot),
          ...new Array<string>(fill).fill(":"),
          ...specs.slice(dot + 1),
        ];
      }
      // `None` inside `[...]` is numpy's `newaxis`: `xs[None, :]` adds a
      // leading size-1 axis.  numpy-ts' `.slice` rejects the extra index
      // ("Too many indices for array"), so handle it here with
      // `np.expand_dims` (error 242: `xs[None, :]` in `escape_grid`).
      if (finalSpecs.some((s) => s === null || s === undefined)) {
        const baseSpecs = finalSpecs.filter((s) => s !== null && s !== undefined);
        const sliceFn =
          (NDArray as Any)?.prototype?.__wanim_original_slice__ ?? (obj as Any).slice;
        const base =
          baseSpecs.length === 0 ? obj : sliceFn.apply(obj, baseSpecs);
        const keepsDim = (spec: Any): boolean => {
          if (typeof spec !== "string") return false;
          if (spec.includes(":")) return true;
          return spec === "..." || spec === "";
        };
        let result = base;
        let nonesSeen = 0;
        for (let p = 0; p < finalSpecs.length; p += 1) {
          if (finalSpecs[p] !== null && finalSpecs[p] !== undefined) continue;
          let keepsBefore = 0;
          for (let q = 0; q < p; q += 1) {
            const prior = finalSpecs[q];
            if (prior === null || prior === undefined) continue;
            if (keepsDim(prior)) keepsBefore += 1;
          }
          const axis = keepsBefore + nonesSeen;
          result = (np as Any).expand_dims(result, axis);
          nonesSeen += 1;
        }
        return result;
      }
      const sliceFn =
        (NDArray as Any)?.prototype?.__wanim_original_slice__ ?? (obj as Any).slice;
      return sliceFn.apply(obj, finalSpecs);
    }
    if (Array.isArray(obj)) return wanGetItemArray(obj, specs);
    if (typeof obj === "string") {
      throw new TypeError("string indices must be integers");
    }
    const getitem = (obj as Any).__getitem__;
    if (typeof getitem === "function") {
      return getitem.call(obj, specs.length === 1 ? specs[0] : specs);
    }
    if (specs.length === 1) return (obj as Any)[specs[0] as Any];
    throw new TypeError("object is not subscriptable with multiple indices");
  }

  // --- numpy multi-dim setitem (`a[:, 1] = v`) ------------------------------
  //
  // Target of the subscript-store branch of `desugarNumpySubscripts` (see
  // `preprocess.ts`): `a[:, 1] = v` becomes
  // `__wan_setitem(a, __wan_slice(None, None), 1, (v))` because a bare
  // `__wan_getitem(...) = ...` is not valid JavaScript. The last argument is
  // always the assigned value; the preceding arguments are the same specs
  // `__wan_getitem` accepts. Returns the value so chained stores
  // (`a[:, 0] = b[:, 0] = v`) work.

  /** True when `v` broadcasts as a scalar for setitem distribution. */
  function isSetScalar(v: Any): boolean {
    if (v === null || v === undefined) return true;
    if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") {
      return true;
    }
    if (isNdArray(v)) return (v.shape as readonly number[]).length === 0;
    if (Array.isArray(v)) return false;
    return true;
  }

  /** Convert an NDArray value to a plain scalar/nested list for distribution. */
  function plainSetValue(v: Any): Any {
    if (isNdArray(v)) {
      if ((v.shape as readonly number[]).length === 0) return scalarize(v);
      try {
        return (v as Any).tolist();
      } catch {
        try {
          return [...(v as Any)];
        } catch {
          return v;
        }
      }
    }
    return v;
  }

  /** The `k`-th element of a (possibly NDArray) sequence value. */
  function setValueAt(v: Any, k: number): Any {
    if (isNdArray(v)) {
      try {
        const row = (v as Any)[k];
        if (row !== undefined) return row;
      } catch {
        /* fall through to tolist */
      }
      try {
        return ((v as Any).tolist() as Any[])[k];
      } catch {
        return [...(v as Any)][k];
      }
    }
    return (v as Any[])[k];
  }

  function normalizeSetIndex(idx: Any, length: number): number {
    const n = typeof idx === "boolean" ? (idx ? 1 : 0) : Math.trunc(Number(idx));
    if (!Number.isFinite(n)) {
      throw new TypeError("slice indices must be integers or None");
    }
    let k = n < 0 ? n + length : n;
    if (k < 0 || k >= length) {
      throw new Error(`index ${String(idx)} is out of bounds for axis with size ${length}`);
    }
    return k;
  }

  function parseSetIntSpec(spec: string): number | null {
    const n = Number(spec);
    return Number.isNaN(n) ? null : Math.trunc(n);
  }

  function setNdRec(
    arr: Any,
    shape: readonly number[],
    specs: Any[],
    dim: number,
    prefix: number[],
    val: Any,
  ): void {
    if (dim === specs.length) {
      if (!isSetScalar(val)) {
        throw new Error("setting an array element with a sequence");
      }
      const scalar = isNdArray(val) ? scalarize(val) : val;
      (arr as Any).set(prefix, scalar);
      return;
    }
    const spec = specs[dim];
    const length = shape[dim] as number;
    if (typeof spec === "number" || typeof spec === "boolean") {
      setNdRec(arr, shape, specs, dim + 1, [...prefix, normalizeSetIndex(spec, length)], val);
      return;
    }
    if (typeof spec === "string") {
      if (spec === ":" || spec === "..." || spec === "") {
        if (isSetScalar(val)) {
          const scalar = isNdArray(val) ? scalarize(val) : val;
          for (let k = 0; k < length; k += 1) {
            setNdRec(arr, shape, specs, dim + 1, [...prefix, k], scalar);
          }
        } else {
          const seq = plainSetValue(val) as Any[];
          if (!Array.isArray(seq) || seq.length !== length) {
            throw new Error("could not broadcast input array from shape");
          }
          for (let k = 0; k < length; k += 1) {
            setNdRec(arr, shape, specs, dim + 1, [...prefix, k], seq[k]);
          }
        }
        return;
      }
      if (!spec.includes(":")) {
        const parsed = parseSetIntSpec(spec);
        if (parsed === null) throw new TypeError("slice indices must be integers or None");
        setNdRec(arr, shape, specs, dim + 1, [...prefix, normalizeSetIndex(parsed, length)], val);
        return;
      }
      const indices = sliceIndices(length, spec);
      if (isSetScalar(val)) {
        const scalar = isNdArray(val) ? scalarize(val) : val;
        for (const k of indices) {
          setNdRec(arr, shape, specs, dim + 1, [...prefix, k], scalar);
        }
      } else {
        const seq = plainSetValue(val) as Any[];
        if (!Array.isArray(seq) || seq.length !== indices.length) {
          throw new Error("could not broadcast input array from shape");
        }
        for (let t = 0; t < indices.length; t += 1) {
          setNdRec(arr, shape, specs, dim + 1, [...prefix, indices[t] as number], seq[t]);
        }
      }
      return;
    }
    if (spec === null || spec === undefined) {
      throw new TypeError("newaxis is not assignable");
    }
    throw new TypeError("fancy indexing assignment is not supported");
  }

  function wanSetItemNd(arr: Any, specs: Any[], value: Any): Any {
    const shape = arr.shape as readonly number[];
    const ndim = shape.length;
    let finalSpecs = specs;
    const dot = specs.indexOf("...");
    if (dot !== -1) {
      const fill = Math.max(0, ndim - (specs.length - 1));
      finalSpecs = [
        ...specs.slice(0, dot),
        ...new Array<string>(fill).fill(":"),
        ...specs.slice(dot + 1),
      ];
    }
    if (finalSpecs.length < ndim) {
      finalSpecs = [...finalSpecs, ...new Array<string>(ndim - finalSpecs.length).fill(":")];
    }
    if (finalSpecs.length !== ndim) {
      throw new Error(`too many indices for array with ${ndim} dimensions`);
    }
    setNdRec(arr, shape, finalSpecs, 0, [], value);
    return value;
  }

  function setArrayRec(container: Any[], specs: Any[], dim: number, val: Any): void {
    const spec = specs[dim];
    const isLast = dim === specs.length - 1;
    if (typeof spec === "number" || typeof spec === "boolean") {
      const idx = normalizeSetIndex(spec, container.length);
      if (isLast) {
        container[idx] = isNdArray(val) && (val.shape as readonly number[]).length === 0
          ? scalarize(val)
          : isNdArray(val)
            ? plainSetValue(val)
            : val;
        return;
      }
      const sub = container[idx];
      if (!Array.isArray(sub)) throw new TypeError("'int' object is not subscriptable");
      setArrayRec(sub, specs, dim + 1, val);
      return;
    }
    if (typeof spec === "string") {
      let indices: number[];
      if (spec === ":" || spec === "..." || spec === "") {
        indices = container.map((_, k) => k);
      } else if (!spec.includes(":")) {
        const parsed = parseSetIntSpec(spec);
        if (parsed === null) throw new TypeError("slice indices must be integers or None");
        const idx = normalizeSetIndex(parsed, container.length);
        if (isLast) {
          container[idx] = isNdArray(val) && (val.shape as readonly number[]).length === 0
            ? scalarize(val)
            : isNdArray(val)
              ? plainSetValue(val)
              : val;
          return;
        }
        const sub = container[idx];
        if (!Array.isArray(sub)) throw new TypeError("'int' object is not subscriptable");
        setArrayRec(sub, specs, dim + 1, val);
        return;
      } else {
        indices = sliceIndices(container.length, spec);
      }
      if (isLast) {
        if (isSetScalar(val)) {
          const scalar = isNdArray(val) ? scalarize(val) : val;
          for (const k of indices) container[k] = scalar;
        } else {
          const seq = plainSetValue(val) as Any[];
          if (!Array.isArray(seq) || seq.length !== indices.length) {
            throw new Error("could not broadcast input array from shape");
          }
          for (let t = 0; t < indices.length; t += 1) {
            const elt = seq[t];
            container[indices[t] as number] = isNdArray(elt) && (elt.shape as readonly number[]).length === 0
              ? scalarize(elt)
              : isNdArray(elt)
                ? plainSetValue(elt)
                : elt;
          }
        }
        return;
      }
      if (isSetScalar(val)) {
        const scalar = isNdArray(val) ? scalarize(val) : val;
        for (const k of indices) {
          const sub = container[k];
          if (!Array.isArray(sub)) throw new TypeError("'int' object is not subscriptable");
          setArrayRec(sub, specs, dim + 1, scalar);
        }
      } else {
        const seq = plainSetValue(val) as Any[];
        if (!Array.isArray(seq) || seq.length !== indices.length) {
          throw new Error("could not broadcast input array from shape");
        }
        for (let t = 0; t < indices.length; t += 1) {
          const sub = container[indices[t] as number];
          if (!Array.isArray(sub)) throw new TypeError("'int' object is not subscriptable");
          setArrayRec(sub, specs, dim + 1, seq[t]);
        }
      }
      return;
    }
    if (spec === null || spec === undefined) {
      throw new TypeError("newaxis is not assignable");
    }
    throw new TypeError("fancy indexing assignment is not supported");
  }

  function wanSetItemArray(arr: Any[], specs: Any[], value: Any): Any {
    const finalSpecs = specs.map((s) => (s === "..." ? ":" : s));
    setArrayRec(arr, finalSpecs, 0, value);
    return value;
  }

  /**
   * Multi-dim setitem target of the `desugarNumpySubscripts` preprocess step.
   * The last argument is the assigned value; the preceding arguments are the
   * same specs `wanGetItem` accepts. Returns the value for chained stores.
   */
  function wanSetItem(obj: Any, ...args: Any[]): Any {
    if (args.length < 2) {
      throw new TypeError("__wan_setitem requires at least one index and a value");
    }
    const value = args[args.length - 1];
    const specs = args.slice(0, -1);
    if (obj === null || obj === undefined) {
      throw new TypeError("'NoneType' object is not subscriptable");
    }
    if (isNdArray(obj)) return wanSetItemNd(obj, specs, value);
    if (Array.isArray(obj)) return wanSetItemArray(obj, specs, value);
    if (typeof obj === "string") {
      throw new TypeError("'str' object does not support item assignment");
    }
    const setitem = (obj as Any).__setitem__;
    if (typeof setitem === "function") {
      return setitem.call(obj, specs.length === 1 ? specs[0] : specs, value);
    }
    if (specs.length === 1) {
      (obj as Any)[specs[0] as Any] = value;
      return value;
    }
    throw new TypeError("object does not support multi-dim item assignment");
  }

  /**
   * Python `dict.get(key[, default])` target of the `desugarDictGet`
   * preprocess step (see `preprocess.ts`).
   *
   * Plain-object dict literals have no `.get` (and `Object.prototype` must
   * never gain one: it poisons `Object.defineProperties` via the descriptor
   * protocol), so `d.get(k, dflt)` is rewritten to `__wan_dict_get(d, k,
   * dflt)` before compilation.  Real `Map`s delegate to `has`/`get`; plain
   * dict objects use `hasOwnProperty`; any other receiver with its own
   * `.get` (e.g. a rapydscript `**kwargs` bag with `ρσ_dict_methods`, or a
   * wanim config/map object) delegates to it; anything else raises the
   * same `TypeError` the direct call raised before.  A missing key yields
   * the default when one is supplied, else `None` (`null`).
   */
  function wanDictGet(obj: Any, key: Any, defval?: Any): Any {
    const hasDefault = arguments.length > 2;
    if (obj === null || obj === undefined) {
      throw new TypeError("'NoneType' object has no attribute 'get'");
    }
    if (isRealMap(obj)) {
      const map = obj as Map<Any, Any>;
      if (map.has(key)) return map.get(key);
      return hasDefault ? defval : null;
    }
    if (isPlainDictObject(obj)) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) return (obj as Any)[key as Any];
      return hasDefault ? defval : null;
    }
    const maybeGet = (obj as Any).get;
    if (typeof maybeGet === "function") {
      return hasDefault ? maybeGet.call(obj, key, defval) : maybeGet.call(obj, key);
    }
    throw new TypeError(`'${typeof obj}' object has no attribute 'get'`);
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
   * Python `abs()`.
   *
   * rapydscript's baselib `abs` is `Math.abs`, so `abs(1+1j)` (a plain
   * `{re, im}` object) yields `NaN` instead of the magnitude (error 185:
   * `r = abs(c)` with `c = 1.5+0j`).  Complex inputs return `hypot(re, im)`
   * like CPython; NDArrays delegate to `np.absolute` (which handles real and
   * complex dtypes); other scalars keep `Math.abs` semantics.
   */
  function pythonAbs(value: Any): Any {
    if (isComplexLike(value)) {
      const [re, im] = complexParts(value, "abs()");
      return Math.hypot(re, im);
    }
    if (typeof value === "number") return Math.abs(value);
    if (typeof value === "boolean") return value ? 1 : 0;
    if (typeof value === "bigint") return value < 0n ? -value : value;
    if (isNdArray(value)) {
      try {
        return scalarizeComplex((np as Any).absolute(value));
      } catch {
        return scalarize(value);
      }
    }
    if (value !== null && value !== undefined && typeof (value as Any).__abs__ === "function") {
      return (value as Any).__abs__();
    }
    return Math.abs(value as number);
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
    if ((value === undefined || value === null) && (name === "real" || name === "imag")) {
      // Bare `{re, im}` complex values (e.g. a plain object, or a numpy-ts
      // `Complex` from before the prototype patch): alias to `re`/`im` so
      // `z.real`/`z.imag` keep working even without the prototype getters.
      try {
        if (obj !== null && obj !== undefined && typeof obj === "object") {
          const alias = name === "real" ? (obj as Any).re : (obj as Any).im;
          if (typeof alias === "number") return alias;
        }
      } catch {
        /* fall through to the raw (undefined) value */
      }
    }
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
    // Internal machinery classes (the animate/updater builders) shuttle
    // kwargs bags around as *values* and separate args from kwargs
    // themselves; mapping a trailing bag onto their slots would detag it
    // (see `__wanim_skip_kwargs_bridge__` in mobject.ts).
    if ((Ctor as Any).__wanim_skip_kwargs_bridge__) return;
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
        return scalarizeComplex((fn as Any)(asNd(x), ...clean));
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
    const BINARY_COMPLEX_OP: Record<string, string> = {
      add: "add", subtract: "sub", multiply: "mul", divide: "div",
      true_divide: "div", power: "pow", pow: "pow", float_power: "pow",
    };
    for (const name of binaryUfuncs) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      facade[name] = (a: Any, b: Any, ...rest: Any[]) => {
        const clean = rest.length > 0 && detectKwargs(rest[rest.length - 1]) !== null
          ? rest.slice(0, -1)
          : rest;
        // numpy-ts mis-broadcasts 0-d complex against vectors for at least
        // `multiply` (NaNs, error 311); route complex-involving pairs through
        // the manual broadcast used by the `ρσ_op_*` operators when there is
        // no extra `out=`-style arg to forward.
        const complexOp = BINARY_COMPLEX_OP[name];
        if (complexOp !== undefined && clean.length === 0 && needsComplexArrayOp(a, b)) {
          const manual = tryComplexArrayOp(complexOp, a, b);
          if (manual !== undefined) return scalarizeComplex(manual);
        }
        return scalarizeComplex((fn as Any)(asNd(a), asNd(b), ...clean));
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
    // `np.conj`/`np.conjugate` on Python scalars throw
    // "undefined is not iterable": numpy-ts requires NDArray inputs, so
    // `np.conj(1)` and `np.conj(complex(...))` (a plain `{re, im}` object)
    // both fail.  Coerce scalars/lists before delegating and map our complex
    // representation back to `makeComplex` (which carries `.real`/`.imag` and
    // dunders, unlike numpy-ts' bare `{re, im}`).
    for (const name of ["conj", "conjugate"]) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      const wrapped = (x: Any, ...rest: Any[]): Any => {
        const clean = rest.length > 0 && detectKwargs(rest[rest.length - 1]) !== null
          ? rest.slice(0, -1)
          : rest;
        if (clean.length === 0) {
          if (isComplexLike(x)) {
            const [re, im] = complexParts(x, name);
            return makeComplex(re, -im);
          }
          if (typeof x === "number" || typeof x === "boolean" || typeof x === "bigint") {
            return x;
          }
          if (isNdArray(x)) {
            const result = (fn as Any)(x);
            const scalar = scalarize(result);
            if (scalar !== result && isComplexLike(scalar)) {
              const [re, im] = complexParts(scalar, name);
              return makeComplex(re, im);
            }
            return result;
          }
          if (Array.isArray(x)) {
            try {
              return (fn as Any)((np as Any).asarray(deepUnwrapNd(x)));
            } catch {
              /* fall through to the generic path */
            }
          }
        }
        const coerced = asNd(x);
        if (isComplexLike(coerced)) {
          const [re, im] = complexParts(coerced, name);
          return makeComplex(re, -im);
        }
        const result = (fn as Any)(coerced, ...clean);
        const scalar = scalarize(result);
        if (scalar !== result && isComplexLike(scalar)) {
          const [re, im] = complexParts(scalar, name);
          return makeComplex(re, im);
        }
        return scalar;
      };
      (wrapped as Any).__argnames__ = ["x"];
      facade[name] = wrapped;
    }
    // `np.angle`/`np.real`/`np.imag` on Python complex scalars throw
    // "undefined is not iterable" (error 185: `np.angle(c)` where `c` is
    // `1.5+0j`): numpy-ts requires NDArray inputs.  Coerce like the ufunc
    // wrappers above and scalarize 0-d results back to Python numbers.
    // `angle` mirrors numpy's `(z, deg=False)` signature.
    try {
      const origAngle = (np as Any).angle;
      if (typeof origAngle === "function") {
        const wrappedAngle = (z: Any, ...rest: Any[]): Any => {
          const kwargs = rest.length > 0 ? detectKwargs(rest[rest.length - 1]) : null;
          let positional = kwargs !== null ? rest.slice(0, -1) : rest.slice();
          if (kwargs !== null) {
            positional = placeKwargs(["deg"], positional, kwargs as Record<string, Any>);
          }
          const deg = positional.length > 0 ? positional[0] : undefined;
          const coerced = asNd(z);
          const result = deg === undefined || deg === null
            ? (origAngle as Any)(coerced)
            : (origAngle as Any)(coerced, deg);
          return scalarizeComplex(result);
        };
        (wrappedAngle as Any).__argnames__ = ["z", "deg"];
        facade.angle = wrappedAngle;
      }
    } catch {
      /* keep the raw numpy-ts angle on wrap failure */
    }
    for (const name of ["real", "imag"]) {
      try {
        const origFn = (np as Any)[name];
        if (typeof origFn !== "function") continue;
        const wrappedPart = (x: Any): Any => scalarizeComplex((origFn as Any)(asNd(x)));
        (wrappedPart as Any).__argnames__ = ["val"];
        facade[name] = wrappedPart;
      } catch {
        /* keep the raw numpy-ts real/imag on wrap failure */
      }
    }
    // `np.gcd`/`np.lcm` on Python scalars (or plain lists) throw
    // "undefined is not iterable": numpy-ts requires integer NDArray inputs
    // and additionally rejects float dtype, so even `np.gcd(4, 6)` fails.
    // Implement both in plain JS with numpy integer semantics (non-negative
    // results, `gcd(0, 0) == 0`, `lcm` as `abs(a * b) // gcd`), elementwise
    // with scalar broadcasting for list/NDArray inputs.  Scalar inputs yield
    // a plain Python number; array inputs are wrapped back as int32 NDArrays
    // (like the ufunc wrappers above) so downstream numpy code keeps working.
    const toGcdInt = (value: Any, name: string): number | bigint => {
      if (typeof value === "bigint") return value;
      if (typeof value === "boolean") return value ? 1 : 0;
      if (typeof value === "number" && Number.isInteger(value)) return value;
      throw new TypeError(
        `ufunc '${name}' not supported for input '${String(value)}'. GCD/LCM are only defined for integers.`,
      );
    };
    const scalarGcd = (a: number | bigint, b: number | bigint): number | bigint => {
      if (typeof a === "bigint" || typeof b === "bigint") {
        let x = typeof a === "bigint" ? a : BigInt(a as number);
        let y = typeof b === "bigint" ? b : BigInt(b as number);
        if (x < 0n) x = -x;
        if (y < 0n) y = -y;
        while (y !== 0n) {
          const t = x % y;
          x = y;
          y = t;
        }
        return x <= BigInt(Number.MAX_SAFE_INTEGER) && x >= BigInt(Number.MIN_SAFE_INTEGER)
          ? Number(x)
          : x;
      }
      let x = Math.abs(a as number);
      let y = Math.abs(b as number);
      while (y !== 0) {
        const t = x % y;
        x = y;
        y = t;
      }
      return x;
    };
    const scalarLcm = (a: number | bigint, b: number | bigint): number | bigint => {
      const g = scalarGcd(a, b);
      const num = (v: number | bigint): number => (typeof v === "bigint" ? Number(v) : v);
      if (num(g) === 0) return 0;
      if (typeof a === "bigint" || typeof b === "bigint" || typeof g === "bigint") {
        const ba = typeof a === "bigint" ? a : BigInt(a as number);
        const bb = typeof b === "bigint" ? b : BigInt(b as number);
        const bg = typeof g === "bigint" ? g : BigInt(g as number);
        const r = (ba < 0n ? -ba : ba) / bg * (bb < 0n ? -bb : bb);
        return r <= BigInt(Number.MAX_SAFE_INTEGER) && r >= BigInt(Number.MIN_SAFE_INTEGER)
          ? Number(r)
          : r;
      }
      return Math.abs((a as number) * (b as number)) / (g as number);
    };
    const makeIntUfunc = (name: string, scalarOp: (a: number | bigint, b: number | bigint) => number | bigint) => {
      const elementwise = (x: Any, y: Any): Any => {
        if (Array.isArray(x) && Array.isArray(y)) {
          if (x.length !== y.length) {
            throw new Error(`operands could not be broadcast together for ufunc '${name}'`);
          }
          return x.map((xi, i) => elementwise(xi, y[i]));
        }
        if (Array.isArray(x)) return x.map((xi) => elementwise(xi, y));
        if (Array.isArray(y)) return y.map((yi) => elementwise(x, yi));
        return scalarOp(toGcdInt(x, name), toGcdInt(y, name));
      };
      const fn = (a: Any, b: Any, ...rest: Any[]): Any => {
        const kwargs = rest.length > 0 && detectKwargs(rest[rest.length - 1]) !== null
          ? rest[rest.length - 1] as Record<string, Any>
          : null;
        let positional = kwargs !== null ? [a, b, ...rest.slice(0, -1)] : [a, b, ...rest];
        if (kwargs !== null) {
          positional = placeKwargs(["x1", "x2"], positional, kwargs);
        }
        const x1 = positional.length > 0 ? positional[0] : undefined;
        const x2 = positional.length > 1 ? positional[1] : undefined;
        const left = deepUnwrapNd(x1);
        const right = deepUnwrapNd(x2);
        const out = elementwise(left, right);
        if (!Array.isArray(left) && !Array.isArray(right)) return out;
        // Array input: mirror the ufunc wrappers and hand back an NDArray.
        const leaves: Any[] = [];
        const collect = (node: Any): void => {
          if (Array.isArray(node)) {
            for (const elt of node) collect(elt);
          } else {
            leaves.push(node);
          }
        };
        collect(out);
        const fitsInt32 = leaves.every(
          (v) => typeof v === "number" && Number.isInteger(v) && v >= -2147483648 && v <= 2147483647,
        );
        try {
          return (np as Any).asarray(out, fitsInt32 ? "int32" : "float64");
        } catch {
          return out;
        }
      };
      (fn as Any).__argnames__ = ["x1", "x2"];
      return fn;
    };
    try {
      facade.gcd = makeIntUfunc("gcd", scalarGcd);
    } catch {
      /* keep the raw numpy-ts gcd on wrap failure */
    }
    try {
      facade.lcm = makeIntUfunc("lcm", scalarLcm);
    } catch {
      /* keep the raw numpy-ts lcm on wrap failure */
    }
    // numpy-ts' `round`/`around` never coerce their input, so they throw on
    // Python scalars (and plain lists): `np.round(x, 6)` on a float fails with
    // "undefined is not iterable".  Route both names through Python `round`,
    // which accepts numbers, nested lists and NDArrays and rounds half to even
    // exactly like numpy.
    const numpyRound = (a: Any, decimals: Any = 0): Any => pythonRound(a, decimals);
    facade.round = numpyRound;
    facade.around = numpyRound;
    // numpy-ts' `clip` requires NDArray inputs, so `np.clip(t, 0.0, 1.0)` on a
    // Python scalar (or a plain list) throws "undefined is not iterable".  This
    // is hit e.g. by per-sample callbacks passed to `axes.plot` (see
    // `ParametricFunction._points_flat`, which maps the user function over
    // plain JS numbers).  Coerce scalar/list inputs before delegating and
    // scalarize 0-d results back to Python numbers.  A plain-number fast path
    // avoids a WASM round-trip per sample.
    try {
      const origClip = (np as Any).clip;
      if (typeof origClip === "function") {
        const coerceClipArg = (value: Any): Any => {
          if (isNdArray(value)) return value;
          if (typeof value === "number" || typeof value === "boolean") {
            try {
              return (np as Any).asarray(value);
            } catch {
              return value;
            }
          }
          if (Array.isArray(value)) {
            try {
              return (np as Any).asarray(deepUnwrapNd(value));
            } catch {
              return value;
            }
          }
          if (isGenericIterable(value) && !isMobjectLike(value)) {
            try {
              return (np as Any).asarray(deepUnwrapNd(value));
            } catch {
              return value;
            }
          }
          return value;
        };
        const wrappedClip = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(["a", "a_min", "a_max", "out"], positional, kwargs as Record<string, Any>);
          }
          const a = positional.length > 0 ? positional[0] : undefined;
          const aMin = positional.length > 1 ? positional[1] : undefined;
          const aMax = positional.length > 2 ? positional[2] : undefined;
          if (typeof a === "number" && typeof aMin === "number" && typeof aMax === "number") {
            let v = a;
            if (!Number.isNaN(aMin)) v = Math.max(v, aMin);
            if (!Number.isNaN(aMax)) v = Math.min(v, aMax);
            return v;
          }
          const coerced = [a, aMin, aMax].map((v) =>
            v === null || v === undefined ? v : coerceClipArg(v),
          );
          // numpy-ts `clip` takes only (a, a_min, a_max); drop `out`/kwargs.
          return scalarize((origClip as Any)(coerced[0], coerced[1], coerced[2]));
        };
        (wrappedClip as Any).__argnames__ = ["a", "a_min", "a_max"];
        facade.clip = wrappedClip;
      }
    } catch {
      /* keep the raw numpy-ts clip on wrap failure */
    }
    // `np.where` / `np.interp` / `np.convolve` on Python scalars (or plain
    // lists) throw in numpy-ts, which requires NDArray storage:
    // `np.where(True, 0.0, 1.0)` fails with
    // `TypeError: Cannot read properties of undefined (reading 'length')`
    // (error 263: nested `np.where` over a 0-d `np.asarray(t)` scalar, where
    // the `x < -2` comparison scalarizes to a plain boolean), and
    // `np.interp(t, xs, ys)` on a scalar `t` fails with
    // `undefined is not iterable` (same scene, per-sample `axes.plot`
    // callbacks).  `np.convolve` on plain lists likewise misbehaves, and its
    // `mode=` kwarg needs `__argnames__` mapping (without it rapydscript's
    // `ρσ_interpolate_kwargs` hands the kwargs bag as a trailing object and
    // numpy-ts silently returns a wrong-shaped result).  Coerce the array
    // operands (scalars included, via `coerceReductionArg` below -- a
    // function declaration, so hoisted) before delegating and scalarize 0-d
    // results back to Python numbers, mirroring numpy (scalar in, scalar
    // out).  `mode` / `left` / `right` / `period` pass through untouched.
    try {
      const origWhere = (np as Any).where;
      if (typeof origWhere === "function") {
        const wrappedWhere = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(["condition", "x", "y"], positional, kwargs as Record<string, Any>);
          }
          if (positional.length <= 1) {
            const cond = positional.length > 0 ? coerceReductionArg(positional[0]) : undefined;
            return (origWhere as Any)(cond);
          }
          const cond = coerceReductionArg(positional[0]);
          const x = positional[1] === null || positional[1] === undefined
            ? positional[1]
            : coerceReductionArg(positional[1]);
          const y = positional[2] === null || positional[2] === undefined
            ? positional[2]
            : coerceReductionArg(positional[2]);
          return scalarizeComplex((origWhere as Any)(cond, x, y));
        };
        (wrappedWhere as Any).__argnames__ = ["condition", "x", "y"];
        facade.where = wrappedWhere;
      }
    } catch {
      /* keep the raw numpy-ts where on wrap failure */
    }
    try {
      const origInterp = (np as Any).interp;
      if (typeof origInterp === "function") {
        const wrappedInterp = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(
              ["x", "xp", "fp", "left", "right", "period"],
              positional,
              kwargs as Record<string, Any>,
            );
          }
          const x = positional.length > 0 ? coerceReductionArg(positional[0]) : undefined;
          const xp = positional.length > 1 ? coerceReductionArg(positional[1]) : undefined;
          const fp = positional.length > 2 ? coerceReductionArg(positional[2]) : undefined;
          return scalarizeComplex((origInterp as Any)(x, xp, fp, ...positional.slice(3)));
        };
        (wrappedInterp as Any).__argnames__ = ["x", "xp", "fp", "left", "right", "period"];
        facade.interp = wrappedInterp;
      }
    } catch {
      /* keep the raw numpy-ts interp on wrap failure */
    }
    try {
      const origConvolve = (np as Any).convolve;
      if (typeof origConvolve === "function") {
        const wrappedConvolve = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(["a", "v", "mode"], positional, kwargs as Record<string, Any>);
          }
          if (positional.length > 0) positional[0] = coerceReductionArg(positional[0]);
          if (positional.length > 1) positional[1] = coerceReductionArg(positional[1]);
          return (origConvolve as Any)(...positional);
        };
        (wrappedConvolve as Any).__argnames__ = ["a", "v", "mode"];
        facade.convolve = wrappedConvolve;
      }
    } catch {
      /* keep the raw numpy-ts convolve on wrap failure */
    }
    // `normalizeNumpyDtype` (hoisted to `createRuntimeCore` scope above) maps
    // a positional/`dtype=` Python-type dtype into a numpy-ts dtype string.
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
        // `np.array([v1, v2])` where each element is an NDArray row (e.g.
        // stacking point vectors) confuses numpy-ts' dtype inference (NaNs).
        // Unwrap nested NDArrays to plain lists first.  A rapydscript
        // `range()` object is also valid input (`np.array(range(3))`), so
        // materialize generic iterables the same way.  NDArray inputs are
        // passed through untouched: unwrapping a top-level NDArray via
        // `tolist()` destroys the template for the `*_like` family
        // (`np.zeros_like(np.asarray(x))` on a 0-d template became a plain
        // number and threw "undefined is not iterable", error 188).
        // `*_like` additionally requires an NDArray template (numpy-ts
        // rejects plain lists/scalars), so coerce non-NDArray array-likes
        // (Python allows `np.zeros_like([1, 2, 3])`) via `asarray` first.
        if (name.endsWith("_like")) {
          if (positional.length > 0 && !isNdArray(positional[0])) {
            const tmpl = positional[0];
            if (
              Array.isArray(tmpl) ||
              typeof tmpl === "number" ||
              typeof tmpl === "boolean" ||
              typeof tmpl === "bigint" ||
              isGenericIterable(tmpl)
            ) {
              try {
                positional = [(np as Any).asarray(deepUnwrapNd(tmpl)), ...positional.slice(1)];
              } catch {
                /* leave the original input so the underlying fn throws */
              }
            }
          }
        } else if (positional.length > 0 && Array.isArray(positional[0])) {
          positional = [deepUnwrapNd(positional[0]), ...positional.slice(1)];
        } else if (
          positional.length > 0 &&
          !isNdArray(positional[0]) &&
          isGenericIterable(positional[0])
        ) {
          try {
            positional = [deepUnwrapNd(positional[0]), ...positional.slice(1)];
          } catch {
            /* keep the original input on materialization failure */
          }
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
    // numpy-ts `linspace` only accepts scalar start/stop (`(number, number)`)
    // and has no `endpoint`/`retstep`/`axis` (its 4th positional is `dtype`,
    // so `endpoint=False` is silently dropped).  Valid numpy instead accepts
    // array-likes (`np.linspace([0,0,0],[1,1,0],3)` -> shape (3,3)) and the
    // full `(start, stop, num, endpoint, retstep, dtype, axis)` signature.
    // Without this, array start/stop returns `[NaN,NaN,NaN]` and downstream
    // `Dot(point=pts[1])` receives `NaN`, so `Arc` throws
    // "arc_center is not iterable" (error 305).  Implement numpy semantics
    // in plain JS and hand back NDArrays.
    try {
      const origLinspace = (np as Any).linspace;
      if (typeof origLinspace === "function") {
        const linspaceArgnames = ["start", "stop", "num", "endpoint", "retstep", "dtype", "axis"];
        const lerpRec = (a: Any, b: Any, t: number): Any => {
          if (Array.isArray(a) && Array.isArray(b)) {
            if (a.length === b.length) return a.map((av, i) => lerpRec(av, b[i], t));
            if (a.length === 1) return b.map((bv) => lerpRec(a[0], bv, t));
            if (b.length === 1) return a.map((av) => lerpRec(av, b[0], t));
            // Mismatched shapes: fall back to elementwise over the longer
            // side is wrong; let the NDArray path below throw/broadcast.
            // Here just broadcast up to the longer length when one side is
            // scalar-like after unwrapping (defensive; numpy would broadcast).
            throw new Error("operands could not be broadcast together");
          }
          if (Array.isArray(a)) return a.map((av) => lerpRec(av, b, t));
          if (Array.isArray(b)) return b.map((bv) => lerpRec(a, bv, t));
          const an = Number(a);
          const bn = Number(b);
          return an + (bn - an) * t;
        };
        const stepRec = (a: Any, b: Any, div: number): Any => {
          if (Array.isArray(a) && Array.isArray(b)) {
            if (a.length === b.length) return a.map((av, i) => stepRec(av, b[i], div));
            if (a.length === 1) return b.map((bv) => stepRec(a[0], bv, div));
            if (b.length === 1) return a.map((av) => stepRec(av, b[0], div));
            throw new Error("operands could not be broadcast together");
          }
          if (Array.isArray(a)) return a.map((av) => stepRec(av, b, div));
          if (Array.isArray(b)) return b.map((bv) => stepRec(a, bv, div));
          if (div === 0) return NaN;
          return (Number(b) - Number(a)) / div;
        };
        const linspaceFn = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(linspaceArgnames, positional, kwargs as Record<string, Any>);
          }
          let start = positional.length > 0 ? positional[0] : undefined;
          let stop = positional.length > 1 ? positional[1] : undefined;
          let num = positional.length > 2 ? positional[2] : undefined;
          let endpoint = positional.length > 3 ? positional[3] : undefined;
          let retstep = positional.length > 4 ? positional[4] : undefined;
          let dtype = positional.length > 5 ? positional[5] : undefined;
          let axis = positional.length > 6 ? positional[6] : undefined;
          if (start === undefined) start = 0;
          if (stop === undefined) stop = 1;
          if (num === undefined || num === null) num = 50;
          if (endpoint === undefined || endpoint === null) endpoint = true;
          if (retstep === undefined || retstep === null) retstep = false;
          if (axis === undefined || axis === null) axis = 0;
          const n = Math.floor(Number(num));
          const ep = endpoint === false || endpoint === 0 ? false : Boolean(endpoint);
          const wantStep = retstep === true || (typeof retstep === "number" && retstep !== 0)
            ? true
            : Boolean(retstep) && retstep !== false && retstep !== 0;
          const startPlain = deepUnwrapNd(start);
          const stopPlain = deepUnwrapNd(stop);
          const normalized = normalizeNumpyDtype(dtype);
          const makeArray = (nested: Any): Any => {
            try {
              return normalized !== undefined
                ? (np as Any).asarray(nested, normalized)
                : (np as Any).asarray(nested);
            } catch {
              return (np as Any).asarray(nested);
            }
          };
          if (!Number.isFinite(n) || n <= 0) {
            const empty = makeArray([]);
            if (wantStep) return [empty, NaN];
            return empty;
          }
          const div = ep ? n - 1 : n;
          const samples: Any[] = new Array(n);
          if (n === 1) {
            samples[0] = lerpRec(startPlain, stopPlain, 0);
          } else {
            for (let i = 0; i < n; i += 1) {
              const t = ep ? i / div : i / n;
              samples[i] = lerpRec(startPlain, stopPlain, t);
            }
          }
          let result: Any = makeArray(samples);
          const axisNum = Number(axis);
          if (axisNum !== 0 && samples.length > 0 && Array.isArray(samples[0])) {
            try {
              const mv = (np as Any).moveaxis;
              if (typeof mv === "function") {
                result = mv(result, 0, axisNum);
              } else if (typeof result.transpose === "function" && result.shape?.length === 2) {
                result = result.transpose();
              }
            } catch {
              /* keep axis=0 layout on move failure */
            }
          }
          if (wantStep) {
            let step: Any;
            try {
              step = stepRec(startPlain, stopPlain, div === 0 ? 0 : div);
            } catch {
              step = NaN;
            }
            if (Array.isArray(step)) {
              try {
                step = normalized !== undefined
                  ? (np as Any).asarray(step, normalized)
                  : (np as Any).asarray(step);
              } catch {
                /* keep plain list on wrap failure */
              }
            } else {
              step = Number(step);
            }
            return [result, step];
          }
          return result;
        };
        (linspaceFn as Any).__argnames__ = [...linspaceArgnames];
        facade.linspace = linspaceFn;
      }
    } catch {
      /* keep the raw numpy-ts linspace on wrap failure */
    }
    for (const name of ["arange", "eye", "identity", "logspace", "geomspace"]) {
      const fn = (np as Any)[name];
      if (typeof fn !== "function") continue;
      // `arange`/`eye` take dtype last (index 3); `identity` takes it
      // at index 1.  Derive from the underlying arity instead of hardcoding.
      const dtypeIndex = name === "identity" ? 1 : 3;
      wrapNumpyCreation(name, dtypeIndex);
    }
    // numpy-ts reduction functions take `axis` positionally, but Python calls
    // them with kwargs (`np.mean(a, axis=0)`, `arr.mean(axis=0)`).  rapydscript
    // only maps the kwargs bag onto positional slots when `f.__argnames__`
    // exists, so annotate both the module functions and (via the prototype
    // patch above) the NDArray methods.  numpy-ts also requires NDArray
    // inputs, so `np.mean([1, 2, 3])` on a plain Python list throws
    // `Cannot read properties of undefined (reading 'length')`; coerce the
    // leading array argument (plain list or scalar) before delegating and
    // scalarize 0-d results back to Python numbers.
    patchNDArrayForPython();
    patchComplexForPython();
    function coerceReductionArg(value: Any): Any {
      if (isNdArray(value)) return value;
      if (Array.isArray(value)) {
        try {
          return (np as Any).asarray(deepUnwrapNd(value));
        } catch {
          return value;
        }
      }
      if (typeof value === "number" || typeof value === "boolean") {
        try {
          return (np as Any).asarray(value);
        } catch {
          return value;
        }
      }
      // rapydscript `range()` objects are valid numpy array-likes
      // (`np.prod(range(1, k + 1))`); materialize generic iterables the same
      // way plain lists are coerced above.
      if (isGenericIterable(value) && !isMobjectLike(value)) {
        try {
          return (np as Any).asarray(deepUnwrapNd(value));
        } catch {
          return value;
        }
      }
      return value;
    }
    function wrapReductionFn(
      origFn: (...a: Any[]) => Any,
      argnames: string[],
    ): (...a: Any[]) => Any {
      const wrapped = (...args: Any[]): Any => {
        const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
        let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
        if (kwargs !== null) {
          positional = placeKwargs(argnames, positional, kwargs as Record<string, Any>);
        }
        if (positional.length > 0) {
          positional = [coerceReductionArg(positional[0]), ...positional.slice(1)];
        }
        return scalarizeComplex((origFn as Any)(...positional));
      };
      (wrapped as Any).__argnames__ = [...argnames];
      return wrapped;
    }
    const moduleAxisReductions = [
      "sum", "mean", "prod", "max", "min", "ptp", "var", "std",
      "nansum", "nanmean", "nanprod", "nanmin", "nanmax", "nanvar", "nanstd",
      "all", "any", "median", "nanmedian", "cumsum", "cumprod",
      "nancumsum", "nancumprod", "argmin", "argmax", "nanargmin", "nanargmax",
      "sort", "argsort",
    ];
    for (const name of moduleAxisReductions) {
      try {
        const fn = facade[name];
        if (typeof fn === "function" && (fn as Any).__wanim_reduction_wrapped__ !== true) {
          facade[name] = wrapReductionFn(fn, ["a", "axis", "keepdims", "dtype", "out"]);
          (facade[name] as Any).__wanim_reduction_wrapped__ = true;
        }
      } catch {
        /* ignore */
      }
    }
    for (const name of ["percentile", "quantile", "nanpercentile", "nanquantile"]) {
      try {
        const fn = facade[name];
        if (typeof fn === "function" && (fn as Any).__wanim_reduction_wrapped__ !== true) {
          facade[name] = wrapReductionFn(fn, ["a", "q", "axis", "keepdims"]);
          (facade[name] as Any).__wanim_reduction_wrapped__ = true;
        }
      } catch {
        /* ignore */
      }
    }
    // Plain-JS vectors (`RIGHT = [1, 0, 0]`, `get_center()` returns
    // `number[]`) reach numpy as plain arrays, but numpy-ts requires NDArray
    // inputs for `linalg.*`, `dot`/`matmul`/`cross`/etc. and throws
    // `Unknown dtype: undefined` on them (e.g. `np.linalg.norm(point -
    // center)` where both sides are plain `number[]`).  Coerce plain-array
    // arguments to NDArray before delegating.  Only `Array.isArray` values
    // are converted, so numeric options (`ord`, `axis=0`, ...) pass through
    // untouched; axis tuples stay plain tuples.
    function coerceArrayArg(value: Any): Any {
      if (isNdArray(value)) return value;
      if (Array.isArray(value)) {
        try {
          return (np as Any).asarray(deepUnwrapNd(value));
        } catch {
          return value;
        }
      }
      // Same range-like handling as reductions: `np.dot(range(3), ...)` etc.
      if (isGenericIterable(value) && !isMobjectLike(value)) {
        try {
          return (np as Any).asarray(deepUnwrapNd(value));
        } catch {
          return value;
        }
      }
      return value;
    }
    // numpy-ts returns multi-output linalg results as plain objects
    // (`{w, v}` for eig, `{q, r}` for qr, `{u, s, vt}` for svd,
    // `{sign, logabsdet}` for slogdet, `{x, residuals, rank, s}` for
    // lstsq) while numpy returns tuples.  Python code unpacks them
    // (`w, v = np.linalg.eig(a)`) or indexes them (`res[0]`), both of
    // which fail on the object form (error 320).  Normalize to tuples.
    function normalizeLinalgResult(key: string, result: Any): Any {
      if (result === null || result === undefined) return result;
      if (Array.isArray(result) || isNdArray(result)) return result;
      if (typeof result !== "object") return result;
      try {
        if ((key === "eig" || key === "eigh") && "w" in result && "v" in result) {
          return [(result as Any).w, (result as Any).v];
        }
        if (key === "qr" && "q" in result && "r" in result) {
          return [(result as Any).q, (result as Any).r];
        }
        if (key === "svd" && "u" in result && "s" in result) {
          const vt = (result as Any).vt ?? (result as Any).vh ?? (result as Any).Vh;
          if (vt !== undefined) return [(result as Any).u, (result as Any).s, vt];
        }
        if (key === "slogdet" && "sign" in result && ("logabsdet" in result || "logdet" in result)) {
          return [(result as Any).sign, (result as Any).logabsdet ?? (result as Any).logdet];
        }
        if (key === "lstsq" && "x" in result && "residuals" in result) {
          return [(result as Any).x, (result as Any).residuals, (result as Any).rank, (result as Any).s];
        }
      } catch {
        /* fall through and return the raw result */
      }
      return result;
    }
    function wrapArrayFn(
      origFn: (...a: Any[]) => Any,
      arrayCount: number,
      argnames: string[],
      linalgKey?: string,
    ): (...a: Any[]) => Any {
      const wrapped = (...args: Any[]): Any => {
        const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
        let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
        if (kwargs !== null) {
          positional = placeKwargs(argnames, positional, kwargs as Record<string, Any>);
        }
        for (let i = 0; i < arrayCount && i < positional.length; i += 1) {
          positional[i] = coerceArrayArg(positional[i]);
        }
        const result = (origFn as Any)(...positional);
        if (linalgKey !== undefined) return normalizeLinalgResult(linalgKey, result);
        return result;
      };
      (wrapped as Any).__argnames__ = [...argnames];
      return wrapped;
    }
    // `multi_dot([A, B, ...])` takes a single list of matrices: coerce each
    // element instead of the outer list.
    function wrapMultiDot(origFn: (...a: Any[]) => Any, argnames: string[]): (...a: Any[]) => Any {
      const wrapped = (...args: Any[]): Any => {
        const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
        let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
        if (kwargs !== null) {
          positional = placeKwargs(argnames, positional, kwargs as Record<string, Any>);
        }
        if (positional.length > 0 && Array.isArray(positional[0])) {
          positional = [positional[0].map(coerceArrayArg), ...positional.slice(1)];
        }
        return (origFn as Any)(...positional);
      };
      (wrapped as Any).__argnames__ = [...argnames];
      return wrapped;
    }
    const LINALG_WRAPS: Record<string, { count: number; names: string[] }> = {
      norm: { count: 1, names: ["x", "ord", "axis", "keepdims"] },
      vector_norm: { count: 1, names: ["x", "axis", "keepdims", "ord"] },
      matrix_norm: { count: 1, names: ["x", "keepdims", "ord"] },
      det: { count: 1, names: ["a"] },
      slogdet: { count: 1, names: ["a"] },
      inv: { count: 1, names: ["a"] },
      solve: { count: 2, names: ["a", "b"] },
      tensorsolve: { count: 2, names: ["a", "b", "axes"] },
      lstsq: { count: 2, names: ["a", "b", "rcond"] },
      pinv: { count: 1, names: ["a", "rcond", "hermitian"] },
      matrix_rank: { count: 1, names: ["A", "tol", "hermitian"] },
      matrix_power: { count: 1, names: ["A", "n"] },
      cond: { count: 1, names: ["x", "p"] },
      qr: { count: 1, names: ["a", "mode"] },
      cholesky: { count: 1, names: ["x", "upper"] },
      svd: { count: 1, names: ["a", "full_matrices", "compute_uv", "hermitian"] },
      eig: { count: 1, names: ["a"] },
      eigh: { count: 1, names: ["a", "UPLO"] },
      eigvals: { count: 1, names: ["a"] },
      eigvalsh: { count: 1, names: ["a", "UPLO"] },
      svdvals: { count: 1, names: ["x"] },
      tensorinv: { count: 1, names: ["a", "ind"] },
      dot: { count: 2, names: ["a", "b", "out"] },
      matmul: { count: 2, names: ["x1", "x2", "out"] },
      vdot: { count: 2, names: ["a", "b"] },
      inner: { count: 2, names: ["a", "b"] },
      outer: { count: 2, names: ["a", "b", "out"] },
      cross: { count: 2, names: ["a", "b", "axisa", "axisb", "axisc", "axis"] },
      tensordot: { count: 2, names: ["a", "b", "axes"] },
      vecdot: { count: 2, names: ["a", "b", "axis"] },
      kron: { count: 2, names: ["a", "b"] },
      trace: { count: 1, names: ["a", "offset", "axis1", "axis2", "dtype", "out"] },
      diagonal: { count: 1, names: ["a", "offset", "axis1", "axis2"] },
      transpose: { count: 1, names: ["a", "axes"] },
      matrix_transpose: { count: 1, names: ["x"] },
      permute_dims: { count: 1, names: ["x", "axes"] },
    };
    try {
      const origLinalg = (np as Any).linalg;
      if (origLinalg !== null && typeof origLinalg === "object") {
        const wrappedLinalg: Record<string, Any> = { ...(origLinalg as Record<string, Any>) };
        for (const key of Object.keys(origLinalg as Record<string, Any>)) {
          const origFn = (origLinalg as Any)[key];
          if (typeof origFn !== "function") continue;
          try {
            if (key === "multi_dot") {
              wrappedLinalg[key] = wrapMultiDot(origFn, ["arrays"]);
              continue;
            }
            const spec = LINALG_WRAPS[key];
            const count = spec !== undefined
              ? spec.count
              : (origFn.length >= 2 ? 2 : 1);
            const names = spec !== undefined
              ? spec.names
              : ["a", "b", "c", "d", "e", "f", "g", "h"].slice(0, Math.max(origFn.length, 1));
            wrappedLinalg[key] = wrapArrayFn(origFn, count, names, key);
          } catch {
            /* keep original on wrap failure */
          }
        }
        facade.linalg = wrappedLinalg;
        // numpy-ts ships no `LinAlgError` (its `linalg.solve` returns NaN
        // instead of throwing on singular input), but valid numpy/manim code
        // catches `np.linalg.LinAlgError` -- and the `except` desugar in
        // `preprocess.ts` hoists exactly such attribute paths into aliases
        // that are matched with `instanceof`.  Provide the class so the
        // attribute exists and `except np.linalg.LinAlgError:` compiles and
        // runs (matching nothing unless actually thrown, like CPython).
        if (typeof (wrappedLinalg as Any).LinAlgError !== "function") {
          class LinAlgError extends Error {
            constructor(message?: string) {
              super(message);
              this.name = "LinAlgError";
            }
          }
          (wrappedLinalg as Any).LinAlgError = LinAlgError;
        }
      }
    } catch {
      /* leave facade.linalg untouched */
    }
    const TOPLEVEL_ARRAY_WRAPS: Record<string, { count: number; names: string[] }> = {
      dot: { count: 2, names: ["a", "b", "out"] },
      matmul: { count: 2, names: ["x1", "x2", "out"] },
      vdot: { count: 2, names: ["a", "b"] },
      inner: { count: 2, names: ["a", "b"] },
      outer: { count: 2, names: ["a", "b", "out"] },
      kron: { count: 2, names: ["a", "b"] },
      cross: { count: 2, names: ["a", "b", "axisa", "axisb", "axisc", "axis"] },
      tensordot: { count: 2, names: ["a", "b", "axes"] },
      vecdot: { count: 2, names: ["a", "b", "axis"] },
      trace: { count: 1, names: ["a", "offset", "axis1", "axis2", "dtype", "out"] },
      diagonal: { count: 1, names: ["a", "offset", "axis1", "axis2"] },
      transpose: { count: 1, names: ["a", "axes"] },
      matrix_transpose: { count: 1, names: ["x"] },
      permute_dims: { count: 1, names: ["x", "axes"] },
      // `append`/`insert`/`delete` take `axis` (often as a kwarg, e.g.
      // `np.append(a, [row], axis=0)`): without `__argnames__` rapydscript's
      // `ρσ_interpolate_kwargs` hands the kwargs bag as a trailing object and
      // numpy-ts throws `TypeError: Cannot convert object to primitive
      // value` (error 152).  Coercing also unwraps NDArray rows nested in
      // plain lists (`[arr[-1]]`) via `deepUnwrapNd`, which numpy-ts'
      // dtype inference otherwise rejects.
      append: { count: 2, names: ["arr", "values", "axis"] },
      insert: { count: 3, names: ["arr", "obj", "values", "axis"] },
      delete: { count: 2, names: ["arr", "obj", "axis"] },
    };
    for (const [name, spec] of Object.entries(TOPLEVEL_ARRAY_WRAPS)) {
      try {
        const origFn = (np as Any)[name];
        if (typeof origFn !== "function") continue;
        // Do not clobber an existing facade wrapper (e.g. ufunc fast paths).
        if ((facade as Any)[name] !== origFn) continue;
        (facade as Any)[name] = wrapArrayFn(origFn, spec.count, spec.names);
      } catch {
        /* ignore */
      }
    }
    try {
      const origMultiDot = (np as Any).multi_dot;
      if (typeof origMultiDot === "function" && (facade as Any).multi_dot === origMultiDot) {
        (facade as Any).multi_dot = wrapMultiDot(origMultiDot, ["arrays"]);
      }
    } catch {
      /* ignore */
    }
    // --- stacking helpers on plain Python lists --------------------------
    //
    // `column_stack`/`vstack`/`hstack`/`dstack`/`row_stack`/`stack`/
    // `concatenate` take a single sequence of arrays, but numpy-ts requires
    // NDArray storage: a plain-JS inner vector (which is how a Python list or
    // an elementwise `p1[:2] - p0[:2]` arrives, e.g.
    // `np.column_stack([p1[:2] - p0[:2], p2[:2] - p0[:2]])`) throws
    // `TypeError: undefined is not iterable`.  Coerce each element of the
    // sequence like `multi_dot` does above; NDArray elements pass through
    // untouched.
    const SEQUENCE_WRAPS: Record<string, string[]> = {
      column_stack: ["tup"],
      row_stack: ["tup"],
      vstack: ["tup"],
      hstack: ["tup"],
      dstack: ["tup"],
      stack: ["arrays", "axis", "out"],
      concatenate: ["a", "axis", "out", "dtype", "casting"],
    };
    for (const [name, argnames] of Object.entries(SEQUENCE_WRAPS)) {
      try {
        const origFn = (np as Any)[name];
        if (typeof origFn !== "function") continue;
        // Do not clobber an existing facade wrapper.
        if ((facade as Any)[name] !== origFn) continue;
        const wrapped = (...args: Any[]): Any => {
          const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
          let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
          if (kwargs !== null) {
            positional = placeKwargs(argnames, positional, kwargs as Record<string, Any>);
          }
          if (positional.length > 0 && Array.isArray(positional[0])) {
            positional[0] = positional[0].map(coerceArrayArg);
          }
          return (origFn as Any)(...positional);
        };
        (wrapped as Any).__argnames__ = [...argnames];
        (facade as Any)[name] = wrapped;
      } catch {
        /* ignore */
      }
    }
    // --- np.random shuffle/permutation/choice on plain Python lists --------
    //
    // numpy-ts' random ops require NDArray storage: `Generator.shuffle` /
    // `permutation` / `choice` on a plain JS array (which is how a Python list
    // arrives, e.g. a list of ManimColors) throws
    // `TypeError: Cannot read properties of undefined (reading 'copy')` (or
    // returns `undefined` for `choice`).  CPython's `Generator` instead accepts
    // any array-like, including lists of arbitrary objects, shuffling in place
    // (`shuffle`) or via index sampling (`permutation` / `choice`).
    //
    // Route plain-array inputs through index delegation so the same RNG state
    // drives the result deterministically: `permutation(n)` yields the shuffled
    // index array (verified identical to `shuffle` on `arange(n)` for seed 2),
    // and `choice(n, ...)` yields sampled indices.  Mapping those indices back
    // onto the list preserves the seed's determinism without reimplementing
    // the bit generator.  NDArray / integer inputs keep delegating to the
    // original implementation untouched.
    try {
      const origRandom = (np as Any).random;
      if (origRandom !== null && typeof origRandom === "object") {
        const toNumberList = (value: Any): number[] => {
          try {
            if (value !== null && typeof value === "object" && typeof (value as Any).tolist === "function") {
              const nested = (value as Any).tolist() as Any;
              const out: number[] = [];
              const walk = (node: Any): void => {
                if (Array.isArray(node)) {
                  for (const elt of node) walk(elt);
                } else {
                  out.push(Number(node));
                }
              };
              walk(nested);
              return out;
            }
          } catch {
            /* fall through */
          }
          if (Array.isArray(value)) return (value as Any[]).map((elt) => Number(elt));
          return [Number(value)];
        };
        const mapIndicesToElements = (nested: Any, elements: Any[]): Any => {
          if (Array.isArray(nested)) return nested.map((elt) => mapIndicesToElements(elt, elements));
          return elements[Number(nested) as number];
        };
        const applyPermutationInPlace = (list: Any[], permFn: (...a: Any[]) => Any): void => {
          if (list.length <= 1) return;
          const indices = toNumberList(permFn(list.length));
          const copy = list.slice();
          for (let i = 0; i < list.length; i += 1) {
            list[i] = copy[indices[i] as number];
          }
        };
        const permutedCopy = (list: Any[], permFn: (...a: Any[]) => Any): Any => {
          if (list.length === 0) return [];
          const indices = toNumberList(permFn(list.length));
          const out = indices.map((k) => list[k as number]);
          // All-number results come back as an NDArray like numpy would return;
          // object lists (e.g. ManimColors, which NDArray cannot hold) stay a
          // plain list so iteration keeps working.
          if (out.length > 0 && out.every((elt) => typeof elt === "number")) {
            try {
              return (np as Any).asarray(out);
            } catch {
              return out;
            }
          }
          return out;
        };
        const stripKwargs = (args: Any[]): { positional: Any[]; kwargs: Record<string, Any> | null } => {
          if (args.length === 0) return { positional: args, kwargs: null };
          const kwargs = detectKwargs(args[args.length - 1]);
          return kwargs !== null
            ? { positional: args.slice(0, -1), kwargs: kwargs as Record<string, Any> }
            : { positional: args.slice(), kwargs: null };
        };
        const choiceFromPlainList = (
          list: Any[],
          rawArgs: Any[],
          indexChoiceFn: (...a: Any[]) => Any,
        ): Any => {
          const { positional, kwargs } = stripKwargs(rawArgs);
          // positional[0] is the list itself; the rest is (size, replace, p).
          const rest = positional.slice(1);
          let size: Any = rest.length > 0 ? rest[0] : undefined;
          let replace: Any = rest.length > 1 ? rest[1] : undefined;
          let p: Any = rest.length > 2 ? rest[2] : undefined;
          if (kwargs !== null) {
            if (size === undefined && "size" in kwargs) size = kwargs["size"];
            if (replace === undefined && "replace" in kwargs) replace = kwargs["replace"];
            if (p === undefined && "p" in kwargs) p = kwargs["p"];
          }
          if (size === null) size = undefined;
          if (replace === undefined || replace === null) replace = true;
          if (list.length === 0) {
            throw new Error("a must be non-empty");
          }
          const indexArgs: Any[] = [list.length];
          if (size !== undefined) indexArgs.push(size);
          if (rest.length > 1 || (kwargs !== null && "replace" in kwargs)) indexArgs.push(replace);
          if (rest.length > 2 || (kwargs !== null && "p" in kwargs)) {
            while (indexArgs.length < 2) indexArgs.push(true);
            indexArgs.push(p);
          }
          const sampled = indexChoiceFn(...indexArgs);
          if (size === undefined) return list[Number(sampled) as number];
          if (
            sampled !== null &&
            typeof sampled === "object" &&
            typeof (sampled as Any).tolist === "function"
          ) {
            try {
              return mapIndicesToElements((sampled as Any).tolist(), list);
            } catch {
              /* fall through to flat mapping */
            }
          }
          if (Array.isArray(sampled)) return mapIndicesToElements(sampled, list);
          return mapIndicesToElements(toNumberList(sampled), list);
        };
        // Legacy `np.random.normal(loc=..., scale=..., size=...)` (and every
        // other legacy distribution, including `Generator` methods from
        // `default_rng`) is a bare numpy-ts closure with no `__argnames__`,
        // so rapydscript's `ρσ_interpolate_kwargs` hands it the kwargs bag as
        // a single positional argument and numpy-ts throws
        // `TypeError: Cannot convert object to primitive value` (error 139).
        // Map kwargs onto positional slots like the other facade wrappers.
        // Additionally numpy-ts returns NaNs/nulls (or throws) when a
        // distribution parameter is an array, so `normal` with array
        // `loc`/`scale` is routed through `standard_normal` with explicit
        // broadcasting (`standard_normal(size) * scale + loc`), which is
        // exactly what numpy computes, and `uniform` with array `low`/`high`
        // is routed through scalar `uniform(0, 1, size)` as
        // `low + (high - low) * U`.  Other distributions keep delegating
        // (their array-parameter support stays an upstream limitation).
        const RANDOM_DIST_ARGNAMES: Record<string, string[]> = {
          random: ["size"], random_sample: ["size"], ranf: ["size"], sample: ["size"],
          uniform: ["low", "high", "size"],
          normal: ["loc", "scale", "size"],
          standard_normal: ["size"],
          exponential: ["scale", "size"],
          standard_exponential: ["size"],
          gamma: ["shape", "scale", "size"],
          standard_gamma: ["shape", "size"],
          beta: ["a", "b", "size"],
          chisquare: ["df", "size"],
          noncentral_chisquare: ["df", "nonc", "size"],
          f: ["dfnum", "dfden", "size"],
          noncentral_f: ["dfnum", "dfden", "nonc", "size"],
          standard_cauchy: ["size"],
          standard_t: ["df", "size"],
          laplace: ["loc", "scale", "size"],
          logistic: ["loc", "scale", "size"],
          lognormal: ["mean", "sigma", "size"],
          gumbel: ["loc", "scale", "size"],
          pareto: ["a", "size"],
          power: ["a", "size"],
          rayleigh: ["scale", "size"],
          triangular: ["left", "mode", "right", "size"],
          wald: ["mean", "scale", "size"],
          weibull: ["a", "size"],
          poisson: ["lam", "size"],
          binomial: ["n", "p", "size"],
          geometric: ["p", "size"],
          hypergeometric: ["ngood", "nbad", "nsample", "size"],
          logseries: ["p", "size"],
          negative_binomial: ["n", "p", "size"],
          zipf: ["a", "size"],
          multinomial: ["n", "pvals", "size"],
          multivariate_normal: ["mean", "cov", "size"],
          dirichlet: ["alpha", "size"],
          vonmises: ["mu", "kappa", "size"],
          randint: ["low", "high", "size"],
          random_integers: ["low", "high", "size"],
          // `Generator.integers(low, high=None, size=None, ...)` is the new
          // API backing `default_rng` (numpy-ts' `Generator` only ships
          // `integers`, not `randint`).  Without this entry the bare
          // numpy-ts closure has no `__argnames__`, so rapydscript's
          // `ρσ_interpolate_kwargs` hands the kwargs bag (e.g. `{size: n}`)
          // as a trailing positional and numpy-ts throws
          // `TypeError: Cannot convert object to primitive value` inside its
          // `size.reduce(...)` (error 241: `rng.integers(0, 2, size=n)`).
          // `dtype`/`endpoint` are accepted positionally for kwargs mapping
          // but dropped on delegation: numpy-ts' `integers(low, high, size)`
          // takes only three args (extras are ignored by JS arity).
          integers: ["low", "high", "size", "dtype", "endpoint"],
        };
        const coerceDistParam = (value: Any): Any =>
          Array.isArray(value) ? coerceArrayArg(value) : value;
        const normalWithArrayParams = (
          stdNormalFn: (...a: Any[]) => Any,
          origNormalFn: (...a: Any[]) => Any,
          positional: Any[],
        ): Any => {
          let loc = positional.length > 0 ? positional[0] : undefined;
          let scale = positional.length > 1 ? positional[1] : undefined;
          const size = positional.length > 2 ? positional[2] : undefined;
          if (loc === undefined || loc === null) loc = 0;
          if (scale === undefined || scale === null) scale = 1;
          const locA = coerceDistParam(loc);
          const scaleA = coerceDistParam(scale);
          if ((isNdArray(locA) || isNdArray(scaleA)) && size !== undefined && size !== null) {
            const z = (stdNormalFn as Any)(size);
            return (np as Any).add((np as Any).multiply(z, scaleA), locA);
          }
          return size === undefined
            ? (origNormalFn as Any)(locA, scaleA)
            : (origNormalFn as Any)(locA, scaleA, size);
        };
        const uniformWithArrayParams = (
          origUniformFn: (...a: Any[]) => Any,
          positional: Any[],
        ): Any => {
          let low = positional.length > 0 ? positional[0] : undefined;
          let high = positional.length > 1 ? positional[1] : undefined;
          const size = positional.length > 2 ? positional[2] : undefined;
          if (low === undefined || low === null) low = 0;
          if (high === undefined || high === null) high = 1;
          const lowA = coerceDistParam(low);
          const highA = coerceDistParam(high);
          if (!isNdArray(lowA) && !isNdArray(highA)) {
            return size === undefined || size === null
              ? (origUniformFn as Any)(low, high)
              : (origUniformFn as Any)(low, high, size);
          }
          // numpy-ts returns nulls/NaNs when low/high are arrays; emulate
          // numpy via low + (high - low) * U[0,1), letting np ufuncs
          // broadcast (verified: multiply/add broadcast [30,3] x [3]).
          if (size !== undefined && size !== null) {
            const u = (origUniformFn as Any)(0, 1, size);
            return (np as Any).add((np as Any).multiply(u, (np as Any).subtract(highA, lowA)), lowA);
          }
          // No size: numpy broadcasts low/high to a common shape.
          try {
            const l = isNdArray(lowA) ? lowA : (np as Any).asarray(lowA);
            const h = isNdArray(highA) ? highA : (np as Any).asarray(highA);
            const template = (np as Any).add(l, h);
            const shape = (template as Any)?.shape as number[] | undefined;
            if (!shape || shape.length === 0) {
              const u = (origUniformFn as Any)(0, 1);
              return (np as Any).add((np as Any).multiply(u, (np as Any).subtract(highA, lowA)), lowA);
            }
            const u = (origUniformFn as Any)(0, 1, shape);
            return (np as Any).add((np as Any).multiply(u, (np as Any).subtract(highA, lowA)), lowA);
          } catch {
            return (origUniformFn as Any)(lowA, highA);
          }
        };
        const wrapDistFn = (
          origFn: (...a: Any[]) => Any,
          stdNormalFn: Any,
          origNormalFn: Any,
          argnames: string[],
          isNormal: boolean,
          isUniform = false,
        ): ((...a: Any[]) => Any) => {
          const fn = function (...args: Any[]): Any {
            const kwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
            let positional = kwargs !== null ? args.slice(0, -1) : args.slice();
            if (kwargs !== null) {
              positional = placeKwargs(argnames, positional, kwargs as Record<string, Any>);
            }
            // `integers` carries unsupported trailing params (`dtype`,
            // `endpoint`) purely so kwargs mapping accepts them; numpy-ts'
            // `integers(low, high, size)` takes only three args.  Drop them
            // (and any unknown-extras bag `placeKwargs` appended) before the
            // size-sensitive coercion below, so a tuple/list `size` stays a
            // plain array in the trailing slot instead of being coerced to
            // an NDArray (which numpy-ts' `Array.isArray(size)` check
            // rejects with `Cannot convert object to primitive value`).
            if (
              argnames.length === 5 &&
              argnames[3] === "dtype" &&
              argnames[4] === "endpoint"
            ) {
              positional = positional.slice(0, 3);
            }
            if (isNormal) {
              return normalWithArrayParams(stdNormalFn, origNormalFn, positional);
            }
            if (isUniform) {
              return uniformWithArrayParams(origFn, positional);
            }
            // Coerce distribution parameters (everything but the trailing
            // `size`, which numpy-ts wants as a plain array/number) so plain
            // Python lists reach NDArray storage.
            if (positional.length > 1) {
              positional = positional.map((elt, i) =>
                i < positional.length - 1 ? coerceDistParam(elt) : elt,
              );
            }
            return (origFn as Any)(...positional);
          };
          (fn as Any).__argnames__ = [...argnames];
          return fn;
        };
        const wrapGenerator = (gen: Any): Any => {
          if (gen === null || typeof gen !== "object") return gen;
          const origShuffle = gen.shuffle;
          const origPerm = gen.permutation;
          const origChoice = gen.choice;
          if (
            typeof origShuffle !== "function" ||
            typeof origPerm !== "function" ||
            typeof origChoice !== "function"
          ) {
            return gen;
          }
          const wrapped: Any = { ...gen };
          const shuffleFn = function (...args: Any[]): Any {
            const { positional } = stripKwargs(args);
            const target = positional.length > 0 ? positional[0] : undefined;
            if (Array.isArray(target)) {
              applyPermutationInPlace(target, (...a: Any[]) => (origPerm as Any)(...a));
              return undefined;
            }
            return (origShuffle as Any)(...positional);
          };
          (shuffleFn as Any).__argnames__ = ["x"];
          wrapped.shuffle = shuffleFn;
          const permFn = function (...args: Any[]): Any {
            const { positional } = stripKwargs(args);
            const target = positional.length > 0 ? positional[0] : undefined;
            if (Array.isArray(target)) return permutedCopy(target, (...a: Any[]) => (origPerm as Any)(...a));
            return (origPerm as Any)(...positional);
          };
          (permFn as Any).__argnames__ = ["x"];
          wrapped.permutation = permFn;
          const choiceFn = function (...args: Any[]): Any {
            const { positional } = stripKwargs(args);
            const target = positional.length > 0 ? positional[0] : undefined;
            if (Array.isArray(target)) {
              return choiceFromPlainList(target, args, (...a: Any[]) => (origChoice as Any)(...a));
            }
            return (origChoice as Any)(...args);
          };
          (choiceFn as Any).__argnames__ = ["a", "size", "replace", "p"];
          wrapped.choice = choiceFn;
          // `Generator` distribution methods are the same bare numpy-ts
          // closures as the legacy module functions: give them kwargs
          // mapping (plus the array-`loc`/`scale` fix for `normal`) so
          // `rng.normal(loc=..., scale=..., size=...)` works like numpy.
          try {
            for (const [name, argnames] of Object.entries(RANDOM_DIST_ARGNAMES)) {
              try {
                if (name === "shuffle" || name === "permutation" || name === "choice") continue;
                const origFn = (gen as Any)[name];
                if (typeof origFn !== "function") continue;
                (wrapped as Any)[name] = wrapDistFn(
                  origFn,
                  (gen as Any).standard_normal,
                  (gen as Any).normal,
                  argnames,
                  name === "normal",
                  name === "uniform",
                );
              } catch {
                /* keep the raw generator method on wrap failure */
              }
            }
          } catch {
            /* distribution wrapping is best-effort */
          }
          return wrapped;
        };
        const origDefaultRng = origRandom.default_rng;
        if (typeof origDefaultRng === "function") {
          const wrappedDefaultRng = function (...args: Any[]): Any {
            return wrapGenerator((origDefaultRng as Any)(...args));
          };
          (wrappedDefaultRng as Any).__argnames__ = ["seed"];
          const wrappedRandom: Record<string, Any> = { ...(origRandom as Record<string, Any>) };
          wrappedRandom.default_rng = wrappedDefaultRng;
          // Legacy `np.random.shuffle/permutation/choice` share the same plain
          // list limitation; wrap them with the same index-delegation strategy
          // driven by the global RNG so `np.random.seed(n)` determinism is kept.
          try {
            const legacyShuffle = origRandom.shuffle;
            const legacyPerm = origRandom.permutation;
            const legacyChoice = origRandom.choice;
            if (typeof legacyShuffle === "function" && typeof legacyPerm === "function") {
              const shuffleFn = function (...args: Any[]): Any {
                const { positional } = stripKwargs(args);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) {
                  applyPermutationInPlace(target, (...a: Any[]) => (legacyPerm as Any)(...a));
                  return undefined;
                }
                return (legacyShuffle as Any)(...positional);
              };
              (shuffleFn as Any).__argnames__ = ["x"];
              wrappedRandom.shuffle = shuffleFn;
            }
            if (typeof legacyPerm === "function") {
              const permFn = function (...args: Any[]): Any {
                const { positional } = stripKwargs(args);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) return permutedCopy(target, (...a: Any[]) => (legacyPerm as Any)(...a));
                return (legacyPerm as Any)(...positional);
              };
              (permFn as Any).__argnames__ = ["x"];
              wrappedRandom.permutation = permFn;
            }
            if (typeof legacyChoice === "function") {
              const choiceFn = function (...args: Any[]): Any {
                const { positional } = stripKwargs(args);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) {
                  return choiceFromPlainList(target, args, (...a: Any[]) => (legacyChoice as Any)(...a));
                }
                return (legacyChoice as Any)(...args);
              };
              (choiceFn as Any).__argnames__ = ["a", "size", "replace", "p"];
              wrappedRandom.choice = choiceFn;
            }
          } catch {
            /* keep default_rng-only wrapping */
          }
          // Map the legacy module-level distributions onto the same wrappers.
          try {
            for (const [name, argnames] of Object.entries(RANDOM_DIST_ARGNAMES)) {
              try {
                const origFn = (origRandom as Any)[name];
                if (typeof origFn !== "function") continue;
                // Do not clobber the plain-list wrappers above.
                if ((wrappedRandom as Any)[name] !== origFn) continue;
                (wrappedRandom as Any)[name] = wrapDistFn(
                  origFn,
                  (origRandom as Any).standard_normal,
                  (origRandom as Any).normal,
                  argnames,
                  name === "normal",
                  name === "uniform",
                );
              } catch {
                /* keep the raw numpy-ts distribution on wrap failure */
              }
            }
          } catch {
            /* distribution wrapping is best-effort */
          }
          // Legacy `np.random.RandomState(seed)` (e.g.
          // `rng = np.random.RandomState(1)` then `rng.uniform(-np.pi, np.pi)`):
          // numpy-ts only ships the new `Generator` API (`default_rng`) and has
          // no `RandomState` class, so anycourseware using the legacy constructor
          // fails with `RandomState is not a function`.  Provide a `RandomState`
          // factory backed by `default_rng(seed)` with the widely used legacy
          // distribution methods delegating to the inner generator.
          //
          // NOTE: the bit stream intentionally differs from CPython's MT19937
          // legacy stream (numpy-ts has no MT19937); the factory is
          // deterministic per seed, which is what wanim scenes need.
          // The factory is a plain function (not an ES class) because
          // rapydscript compiles `RandomState(1)` to a bare call, and class
          // constructors throw when invoked without `new`.
          try {
            const coerceRsSeed = (s: Any): Any => {
              if (s === undefined || s === null) return undefined;
              if (typeof s === "number" && Number.isInteger(s)) return s;
              if (typeof s === "bigint") {
                const n = Number(s);
                if (Number.isInteger(n)) return n;
                return undefined;
              }
              if (Array.isArray(s)) {
                try {
                  return (np as Any).asarray(deepUnwrapNd(s));
                } catch {
                  return undefined;
                }
              }
              const n = Number(s);
              if (Number.isInteger(n)) return n;
              return undefined;
            };
            const freshRsGen = (s: Any): Any => {
              const seedArg = coerceRsSeed(s);
              try {
                return (origDefaultRng as Any)(...(seedArg === undefined ? [] : [seedArg]));
              } catch {
                return (origDefaultRng as Any)();
              }
            };
            const RandomStateFactory = function (...args: Any[]): Any {
              const ctorKwargs = args.length > 0 ? detectKwargs(args[args.length - 1]) : null;
              let ctorPos = ctorKwargs !== null ? args.slice(0, -1) : args.slice();
              if (ctorKwargs !== null) {
                ctorPos = placeKwargs(["seed"], ctorPos, ctorKwargs as Record<string, Any>);
              }
              let gen: Any = freshRsGen(ctorPos.length > 0 ? ctorPos[0] : undefined);
              const rs: Record<string, Any> = {};
              const delegate = (method: string, argnames: string[]): Any => {
                const fn = function (...callArgs: Any[]): Any {
                  const kw = callArgs.length > 0 ? detectKwargs(callArgs[callArgs.length - 1]) : null;
                  let pos = kw !== null ? callArgs.slice(0, -1) : callArgs.slice();
                  if (kw !== null) pos = placeKwargs(argnames, pos, kw as Record<string, Any>);
                  const target = (gen as Any)[method];
                  if (typeof target === "function") return (target as Any).apply(gen, pos);
                  const modFn = (origRandom as Any)[method];
                  if (typeof modFn === "function") return (modFn as Any)(...pos);
                  throw new Error(`RandomState has no method '${method}'`);
                };
                (fn as Any).__argnames__ = [...argnames];
                return fn;
              };
              const seedFn = function (...callArgs: Any[]): Any {
                const kw = callArgs.length > 0 ? detectKwargs(callArgs[callArgs.length - 1]) : null;
                let pos = kw !== null ? callArgs.slice(0, -1) : callArgs.slice();
                if (kw !== null) pos = placeKwargs(["seed"], pos, kw as Record<string, Any>);
                gen = freshRsGen(pos.length > 0 ? pos[0] : undefined);
                return undefined;
              };
              (seedFn as Any).__argnames__ = ["seed"];
              rs.seed = seedFn;
              rs.uniform = delegate("uniform", ["low", "high", "size"]);
              rs.random_sample = delegate("random", ["size"]);
              rs.random = delegate("random", ["size"]);
              rs.ranf = delegate("random", ["size"]);
              rs.sample = delegate("random", ["size"]);
              rs.normal = (function () {
                // Like `delegate("normal", ...)` above, but array `loc`/`scale`
                // go through `standard_normal` (numpy-ts returns NaNs for array
                // distribution parameters).  Closed over `gen` (not the method)
                // so `rs.seed(...)` reseeding keeps working.
                const fn = function (...callArgs: Any[]): Any {
                  const kw = callArgs.length > 0 ? detectKwargs(callArgs[callArgs.length - 1]) : null;
                  let pos = kw !== null ? callArgs.slice(0, -1) : callArgs.slice();
                  if (kw !== null) pos = placeKwargs(["loc", "scale", "size"], pos, kw as Record<string, Any>);
                  return normalWithArrayParams(
                    (...a: Any[]) => (gen as Any).standard_normal(...a),
                    (...a: Any[]) => (gen as Any).normal(...a),
                    pos,
                  );
                };
                (fn as Any).__argnames__ = ["loc", "scale", "size"];
                return fn;
              })();
              rs.standard_normal = delegate("standard_normal", ["size"]);
              rs.exponential = delegate("exponential", ["scale", "size"]);
              rs.poisson = delegate("poisson", ["lam", "size"]);
              rs.binomial = delegate("binomial", ["n", "p", "size"]);
              rs.random_integers = delegate("random_integers", ["low", "high", "size"]);
              // `rand(d0, d1, ...)` / `randn(d0, d1, ...)` take dims as varargs.
              const randFn = function (...dims: Any[]): Any {
                const kw = dims.length > 0 ? detectKwargs(dims[dims.length - 1]) : null;
                let shape: Any[] = kw !== null ? dims.slice(0, -1) : dims.slice();
                if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0] as Any[];
                if (shape.length === 0) return (gen as Any).random();
                return (gen as Any).random(shape.length === 1 ? shape[0] : shape);
              };
              rs.rand = randFn;
              const randnFn = function (...dims: Any[]): Any {
                const kw = dims.length > 0 ? detectKwargs(dims[dims.length - 1]) : null;
                let shape: Any[] = kw !== null ? dims.slice(0, -1) : dims.slice();
                if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0] as Any[];
                if (shape.length === 0) return (gen as Any).standard_normal();
                return (gen as Any).standard_normal(shape.length === 1 ? shape[0] : shape);
              };
              rs.randn = randnFn;
              // Legacy `randint(low, high=None, size=None)`: one-arg form means
              // `integers(0, low)`.
              const randintFn = function (...callArgs: Any[]): Any {
                const kw = callArgs.length > 0 ? detectKwargs(callArgs[callArgs.length - 1]) : null;
                let pos = kw !== null ? callArgs.slice(0, -1) : callArgs.slice();
                if (kw !== null) pos = placeKwargs(["low", "high", "size"], pos, kw as Record<string, Any>);
                let low = pos.length > 0 ? pos[0] : undefined;
                let high = pos.length > 1 ? pos[1] : undefined;
                const size = pos.length > 2 ? pos[2] : undefined;
                if (high === undefined || high === null) {
                  high = low;
                  low = 0;
                }
                const iargs = size === undefined || size === null ? [low, high] : [low, high, size];
                if (typeof (gen as Any).integers === "function") {
                  return (gen as Any).integers(...iargs);
                }
                return (origRandom as Any).randint(...iargs);
              };
              (randintFn as Any).__argnames__ = ["low", "high", "size"];
              rs.randint = randintFn;
              // New-API alias: `RandomState(...).integers(...)` delegates to
              // the same path (extra `dtype`/`endpoint` kwargs are dropped
              // there, matching the `wrapDistFn` truncation above).
              rs.integers = randintFn;
              // `shuffle` / `permutation` / `choice` need the same plain-list
              // index delegation as `wrapGenerator` (lists of arbitrary objects
              // cannot enter NDArray storage), but bound to the *current*
              // generator so `rs.seed(...)` reseeding keeps working.
              const rsShuffleFn = function (...callArgs: Any[]): Any {
                const { positional } = stripKwargs(callArgs);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) {
                  applyPermutationInPlace(target, (...a: Any[]) => (gen as Any).permutation(...a));
                  return undefined;
                }
                return (gen as Any).shuffle(...positional);
              };
              (rsShuffleFn as Any).__argnames__ = ["x"];
              rs.shuffle = rsShuffleFn;
              const rsPermFn = function (...callArgs: Any[]): Any {
                const { positional } = stripKwargs(callArgs);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) {
                  return permutedCopy(target, (...a: Any[]) => (gen as Any).permutation(...a));
                }
                return (gen as Any).permutation(...positional);
              };
              (rsPermFn as Any).__argnames__ = ["x"];
              rs.permutation = rsPermFn;
              const rsChoiceFn = function (...callArgs: Any[]): Any {
                const { positional } = stripKwargs(callArgs);
                const target = positional.length > 0 ? positional[0] : undefined;
                if (Array.isArray(target)) {
                  return choiceFromPlainList(target, callArgs, (...a: Any[]) => (gen as Any).choice(...a));
                }
                return (gen as Any).choice(...callArgs);
              };
              (rsChoiceFn as Any).__argnames__ = ["a", "size", "replace", "p"];
              rs.choice = rsChoiceFn;
              return rs;
            };
            (RandomStateFactory as Any).__argnames__ = ["seed"];
            wrappedRandom.RandomState = RandomStateFactory;
          } catch {
            /* leave RandomState missing rather than breaking the facade */
          }
          facade.random = wrappedRandom;
          if ((facade as Any).default_rng === origDefaultRng) {
            (facade as Any).default_rng = wrappedDefaultRng;
          }
        }
      }
    } catch {
      /* random wrapping is best-effort; numpy-ts paths stay usable */
    }
    return facade;
  }

  /**
   * CPython's `__import__` builtin for dynamic imports.
   *
   * Valid Python such as `__import__("numpy").cos(x)` (seen in the wild, e.g.
   * a slanted-shadow updater closing over `tracker`) otherwise fails with
   * `ReferenceError: __import__ is not defined`, because rapydscript's
   * baselib provides no such builtin. Only the modules available in the
   * browser/JS runtime can be returned (`numpy`, `math`, `random`,
   * `itertools`, `manim`); anything else raises `ModuleNotFoundError` like
   * CPython. Dotted paths drill into the facade (`numpy.linalg`), honouring
   * the `fromlist` rule (bare `__import__("a.b")` returns top module `a`,
   * `from a.b import c` passes a non-empty `fromlist` and gets `a.b`).
   */
  function hasFromEntries(fromlist: Any): boolean {
    if (fromlist === null || fromlist === undefined) return false;
    if (typeof fromlist === "string") return fromlist.length > 0;
    if (Array.isArray(fromlist)) return fromlist.length > 0;
    if (typeof fromlist === "object" && typeof (fromlist as Any).length === "number") {
      return (fromlist as Any).length > 0;
    }
    return !!fromlist;
  }

  function resolveImportModule(
    modules: Record<string, Any>,
    name: Any,
    fromlist?: Any,
  ): Any {
    const modName = String(name);
    const parts = modName.split(".");
    const top = parts[0] as string;
    const topMod = modules[top];
    if (topMod === undefined) {
      const error = new Error(`No module named '${modName}'`);
      error.name = "ModuleNotFoundError";
      throw error;
    }
    let cur: Any = topMod;
    for (let i = 1; i < parts.length; i += 1) {
      const part = parts[i] as string;
      if (cur !== null && cur !== undefined && part in cur) {
        cur = cur[part];
      } else {
        const error = new Error(`No module named '${modName}'`);
        error.name = "ModuleNotFoundError";
        throw error;
      }
    }
    // CPython: bare `__import__("a.b")` returns top module `a`; with a
    // non-empty `fromlist` (the `from a.b import c` form) it returns `a.b`.
    if (parts.length > 1 && !hasFromEntries(fromlist)) return topMod;
    return cur;
  }

  function makePythonImport(modules: Record<string, Any>): Any {
    const fn = function (name: Any, _g?: Any, _l?: Any, fromlist?: Any, _level?: Any): Any {
      return resolveImportModule(modules, name, fromlist);
    };
    (fn as Any).__argnames__ = ["name", "globals", "locals", "fromlist", "level"];
    return fn;
  }

  /** Build the global environment the compiled Python module runs in. */
  function buildRuntime(): PythonRuntime {
    const globals: Record<string, Any> = {};
    const wrapped = wrapWanimNamespace(wanim);
    Object.assign(globals, wrapped);
    const numpyFacade = buildNumpy();
    globals.np = numpyFacade;
    globals.numpy = numpyFacade;
    // Python's `random` module (see `python_random.ts`): `import random` and
    // `from random import ...` are stripped by `preprocess.ts`, and the
    // module object is provided here as a global instead.
    globals.random = buildPythonRandom();
    // Python's `math` module (see `python_math.ts`): `import math` and
    // `from math import ...` are stripped by `preprocess.ts`, and the
    // module object is provided here as a global instead.
    globals.math = buildPythonMath();
    // Python's `itertools` module (see `python_itertools.ts`):
    // `import itertools` and `from itertools import ...` are stripped by
    // `preprocess.ts`, and the module object is provided here as a global
    // instead.
    globals.itertools = buildPythonItertools();
    globals.ρσ_op_add = pythonAdd;
    globals.ρσ_op_sub = pythonSub;
    globals.ρσ_op_mul = pythonMul;
    globals.ρσ_op_div = pythonDiv;
    globals.ρσ_op_floordiv = pythonFloorDiv;
    globals.ρσ_op_mod = pythonMod;
    globals.ρσ_op_pow = pythonPow;
    globals.ρσ_op_neg = pythonNeg;
    globals.ρσ_op_lt = pythonLt;
    globals.ρσ_op_gt = pythonGt;
    globals.ρσ_op_le = pythonLe;
    globals.ρσ_op_ge = pythonGe;
    globals.ρσ_getdot = getDot;
    // Python builtins rapydscript's baselib does not provide.  `min` and
    // `max` come from baselib; `round` has no JS equivalent that
    // matches CPython's banker's rounding / `ndigits` argument.  `abs` comes
    // from baselib as `Math.abs` but returns NaN on complex inputs. The
    // compiled JS declares its own local `var abs = Math.abs`, which shadows
    // this override, so bare `abs(x)` calls are rewritten to `__wan_abs(x)`
    // by the `desugarAbsBuiltin` step (see `preprocess.ts`); both names point
    // at the same complex-aware `pythonAbs`.  `sum` comes from baselib but
    // accumulates with plain JS `+`, which stringifies NDArray terms instead
    // of adding them elementwise, so bare `sum(...)` calls are likewise
    // rewritten to `__wan_sum(...)` by the `desugarSumBuiltin` step.
    // `complex`
    // (and the `__wan_complex` factory the `desugarComplexLiterals` step
    // emits) likewise has no baselib equivalent.
    globals.round = pythonRound;
    globals.abs = pythonAbs;
    globals.__wan_abs = pythonAbs;
    globals.complex = pythonComplex;
    globals.__wan_complex = wanComplex;
    // `tuple()` has no rapydscript baselib equivalent (only `list` exists),
    // so bare `tuple(x)` calls fail with `ReferenceError: tuple is not
    // defined`.  Tuples and lists share the plain-array representation.
    globals.tuple = pythonTuple;
    // One-level flatten for the multi-`for` comprehension desugar (see
    // `desugarMultiForComprehensions` in `preprocess.ts`). The name is
    // reserved: the desugar refuses to run when user code already uses it.
    globals.__wan_flatten1 = wanFlatten1;
    // Multi-dim subscript helpers for the `desugarNumpySubscripts` step
    // (likewise reserved names; see `preprocess.ts`).
    globals.__wan_getitem = wanGetItem;
    globals.__wan_setitem = wanSetItem;
    globals.__wan_slice = wanSliceDim;
    // Matrix-multiplication helper for the `desugarMatmul` step (likewise a
    // reserved name; see `preprocess.ts`).
    globals.__wan_matmul = wanMatmul;
    // CPython-compatible `reversed()` for the `desugarReversedBuiltin` step
    // (likewise a reserved name; see `preprocess.ts`). rapydscript's own
    // `reversed` only handles plain arrays/strings.
    globals.__wan_reversed = wanReversed;
    // Spread helper for the `desugarStarredDisplays` step (likewise a
    // reserved name; see `preprocess.ts`). rapydscript accepts `*args` in
    // calls but rejects `*` inside list/tuple displays.
    globals.__wan_spread = wanSpread;
    // Numpy/complex-aware `sum()` for the `desugarSumBuiltin` step (likewise
    // a reserved name; see `preprocess.ts`). rapydscript's own `sum`
    // accumulates with plain JS `+` and stringifies NDArray terms.
    globals.__wan_sum = pythonSum;
    // Python `dict.get()` for the `desugarDictGet` step (likewise a reserved
    // name; see `preprocess.ts`). Plain-object dict literals have no `.get`
    // and `Object.prototype` must never gain one.
    globals.__wan_dict_get = wanDictGet;
    // CPython's `__import__` for dynamic imports such as
    // `__import__("numpy").cos(x)`. rapydscript's baselib provides no such
    // builtin, so bare `__import__` calls fail with `ReferenceError`.
    // Only the browser/JS-runtime modules are resolvable here.
    globals.__import__ = makePythonImport({
      numpy: numpyFacade,
      np: numpyFacade,
      math: globals.math,
      random: globals.random,
      itertools: globals.itertools,
      manim: wrapped,
    });
    return { globals, wanim, np: numpyFacade };
  }

  return {
    isNdArray,
    buildNumpy,
    buildPythonRandom,
    buildPythonMath,
    buildPythonItertools,
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
    pythonLt,
    pythonGt,
    pythonLe,
    pythonGe,
    pythonRound,
    pythonAbs,
    pythonComplex,
    wanComplex,
    pythonTuple,
    pythonSum,
    isComplexLike,
    wanFlatten1,
    wanGetItem,
    wanSetItem,
    wanDictGet,
    wanSlice: wanSliceDim,
    wanMatmul,
    wanReversed,
    wanSpread,
    getDot,
    resolveImportModule,
    makePythonImport,
  };
}
