/**
 * Python `itertools` module for the wanim browser/JS runtime.
 *
 * Manim scenes (including original 3b1b sources) routinely use the Python
 * standard-library `itertools` module (`import itertools as it`,
 * `it.combinations(...)`, `it.product(...)`, ...), which cannot exist in the
 * browser as-is.  This module ports CPython's `itertools` (`Lib/itertools.py`
 * semantics, operating on plain JS arrays since rapydscript represents Python
 * lists/tuples as arrays) so that `preprocess.ts` can accept
 * `import itertools` / `from itertools import ...` and compiled scenes
 * observe the same values as CPython.
 *
 * Fidelity notes:
 *   - Finite combinators (`combinations`, `permutations`,
 *     `combinations_with_replacement`, `product`, `chain`, `compress`,
 *     `dropwhile`, `takewhile`, `filterfalse`, `starmap`, `pairwise`,
 *     `batched`, `accumulate`, `groupby`, `zip_longest`, `tee`, bounded
 *     `islice`/`repeat`) return eager plain-JS arrays, exactly like
 *     `list(CPython_itertools.x(...))`.  Eager arrays work with every
 *     rapydscript iteration form (`for x in ...`, `list(...)`, unpacking).
 *   - Infinite iterators (`count`, `cycle`, unbounded `repeat`) cannot be
 *     materialized, so they return lazy generator objects (iterable via
 *     `Symbol.iterator`).  `islice` consumes its input lazily, so the
 *     standard `islice(count(), n)` / `islice(cycle(p), n)` bounding pattern
 *     works; materializing an unbounded iterator directly (e.g.
 *     `list(count())`) hangs, exactly like CPython.
 *   - `groupby` returns an eager array of `[key, groupArray]` pairs (like
 *     `[(k, list(g)) for k, g in CPython_groupby(...)]`); `tee` returns an
 *     array of `n` independent array copies.
 *
 * Known deviations from CPython (all inherent to the JS runtime):
 *   - Integers past 2**53 lose precision (rapydscript doubles).
 *   - Lazy iterators are one-shot JS generators, not re-iterable
 *     `itertools` objects; `list(gen)` relies on rapydscript's `list()`
 *     accepting `Symbol.iterator` iterables.
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

/** Split `(…args, maybeKwargsBag)` into positionals plus the raw kwargs bag. */
function splitRaw(args: Any[]): { positional: Any[]; kwargs: Record<string, Any> } {
  const bag = takeKwargsBag(args);
  if (bag === null) return { positional: args.slice(), kwargs: {} };
  const kwargs: Record<string, Any> = {};
  for (const key of Object.keys(bag)) {
    if (!key.startsWith("kwargs-object")) kwargs[key] = bag[key];
  }
  return { positional: args.slice(0, -1), kwargs };
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
  const { positional, kwargs } = splitRaw(args);
  return applyKwargs(fn, names, positional, kwargs);
}

function required(value: Any, fn: string, name: string): Any {
  if (value === undefined) {
    throw new TypeError(`${fn}() missing required argument '${name}'`);
  }
  return value;
}

/** Require a non-negative integer argument. */
function asNonNegInt(value: Any, fn: string, name: string): number {
  const n = asIndex(value, name);
  if (n < 0) throw new RangeError(`${fn}(): ${name} must be a non-negative integer`);
  return n;
}

/** Python `None` arrives as `null` or `undefined` from rapydscript. */
function isNone(value: Any): boolean {
  return value === null || value === undefined;
}

// ---------------------------------------------------------------------------
// Iterable helpers.
// ---------------------------------------------------------------------------

/**
 * Materialize a finite iterable into a plain array.
 *
 * Plain arrays (rapydscript lists/tuples) are copied; strings split into
 * chars (like CPython iteration); anything with `Symbol.iterator` (NDArrays,
 * wanim Mobjects/VGroups, generators) is spread; otherwise a numeric
 * `length` plus indexing is used (the Mobject Proxy array-like view).
 */
function toArray(iterable: Any, fn: string): Any[] {
  if (Array.isArray(iterable)) return iterable.slice();
  if (typeof iterable === "string") return iterable.split("");
  if (iterable !== null && iterable !== undefined) {
    let iteratorMethod: Any;
    try {
      iteratorMethod = (iterable as Any)[Symbol.iterator];
    } catch {
      iteratorMethod = undefined;
    }
    if (typeof iteratorMethod === "function") {
      try {
        return [...(iterable as Iterable<Any>)];
      } catch {
        /* fall through to length-based access */
      }
    }
    try {
      const len = (iterable as Any).length;
      if (typeof len === "number" && Number.isFinite(len)) {
        const n = Math.max(0, Math.trunc(len));
        const out: Any[] = new Array(n);
        for (let i = 0; i < n; i += 1) out[i] = (iterable as Any)[i];
        return out;
      }
    } catch {
      /* fall through to the TypeError */
    }
  }
  throw new TypeError(`'${typeof iterable}' object is not iterable (${fn})`);
}

/** Lazily yield every element of `iterable` (works for infinite inputs too). */
function* iterEach(iterable: Any, fn: string): Generator<Any> {
  if (Array.isArray(iterable)) {
    for (const elt of iterable) yield elt;
    return;
  }
  if (typeof iterable === "string") {
    for (const ch of iterable.split("")) yield ch;
    return;
  }
  if (iterable !== null && iterable !== undefined) {
    let iteratorMethod: Any;
    try {
      iteratorMethod = (iterable as Any)[Symbol.iterator];
    } catch {
      iteratorMethod = undefined;
    }
    if (typeof iteratorMethod === "function") {
      yield* (iterable as Iterable<Any>);
      return;
    }
    try {
      const len = (iterable as Any).length;
      if (typeof len === "number" && Number.isFinite(len)) {
        const n = Math.max(0, Math.trunc(len));
        for (let i = 0; i < n; i += 1) yield (iterable as Any)[i];
        return;
      }
    } catch {
      /* fall through to the TypeError */
    }
  }
  throw new TypeError(`'${typeof iterable}' object is not iterable (${fn})`);
}

// ---------------------------------------------------------------------------
// Module definition.
// ---------------------------------------------------------------------------

/** Names exported by `from itertools import *`. */
export const ITERTOOLS_PUBLIC = [
  "accumulate", "batched", "chain", "combinations",
  "combinations_with_replacement", "compress", "count", "cycle",
  "dropwhile", "filterfalse", "groupby", "islice", "pairwise",
  "permutations", "product", "repeat", "starmap", "takewhile",
  "tee", "zip_longest",
];

export interface PythonItertoolsModule extends Record<string, Any> {}

function annotate(fn: Any, names: string[]): Any {
  fn.__argnames__ = [...names];
  return fn;
}

function defaultAdd(a: Any, b: Any): Any {
  if (typeof a === "number" && typeof b === "number") return a + b;
  if (typeof a === "string" || typeof b === "string") return String(a) + String(b);
  if (Array.isArray(a) && Array.isArray(b)) return [...a, ...b];
  return (a as number) + (b as number);
}

/** Build the `itertools` module object exposed as a runtime global. */
export function buildPythonItertools(): PythonItertoolsModule {
  const mod: Record<string, Any> = {};

  mod.accumulate = annotate((...args: Any[]) => {
    const [iterableRaw, funcRaw, initialRaw] = splitCall(
      "accumulate", ["iterable", "func", "initial"], args,
    );
    const pool = toArray(required(iterableRaw, "accumulate", "iterable"), "accumulate");
    const func = isNone(funcRaw) ? undefined : funcRaw;
    const hasInitial = !isNone(initialRaw);
    const out: Any[] = [];
    let acc = hasInitial ? initialRaw : undefined;
    let started = hasInitial;
    if (hasInitial) out.push(initialRaw);
    for (const elt of pool) {
      if (!started) {
        acc = elt;
        started = true;
      } else {
        acc = func === undefined
          ? defaultAdd(acc, elt)
          : (func as Any)(acc, elt);
      }
      out.push(acc);
    }
    return out;
  }, ["iterable", "func", "initial"]);

  mod.batched = annotate((...args: Any[]) => {
    const { positional, kwargs } = splitRaw(args);
    let strict = false;
    for (const key of Object.keys(kwargs)) {
      if (key === "strict") strict = !!kwargs[key];
      else throw new TypeError(`batched() got an unexpected keyword argument '${key}'`);
    }
    if (positional.length < 2) throw new TypeError("batched() missing required argument 'n'");
    if (positional.length > 2) throw new TypeError(`batched() takes 2 positional arguments but ${positional.length} were given`);
    const pool = toArray(positional[0], "batched");
    const n = asIndex(positional[1], "n");
    if (n < 1) throw new RangeError("batched(): n must be at least one");
    const out: Any[][] = [];
    for (let i = 0; i < pool.length; i += n) out.push(pool.slice(i, i + n));
    if (strict && pool.length % n !== 0) {
      throw new Error("batched(): iterable is not divisible by n");
    }
    return out;
  }, ["iterable", "n"]);

  const chainFn = annotate((...args: Any[]) => {
    const { positional } = splitRaw(args);
    const out: Any[] = [];
    for (const it of positional) {
      for (const elt of iterEach(it, "chain")) out.push(elt);
    }
    return out;
  }, []);
  (chainFn as Any).from_iterable = annotate((...args: Any[]) => {
    const [iterable] = splitCall("chain.from_iterable", ["iterable"], args);
    const out: Any[] = [];
    for (const inner of iterEach(required(iterable, "chain.from_iterable", "iterable"), "chain.from_iterable")) {
      for (const elt of iterEach(inner, "chain.from_iterable")) out.push(elt);
    }
    return out;
  }, ["iterable"]);
  mod.chain = chainFn;

  mod.combinations = annotate((...args: Any[]) => {
    const [iterableRaw, rRaw] = splitCall("combinations", ["iterable", "r"], args);
    const pool = toArray(required(iterableRaw, "combinations", "iterable"), "combinations");
    const r = asNonNegInt(required(rRaw, "combinations", "r"), "combinations", "r");
    const n = pool.length;
    const out: Any[][] = [];
    if (r > n) return out;
    if (r === 0) return [[]];
    const indices = Array.from({ length: r }, (_, i) => i);
    out.push(indices.map((i) => pool[i]));
    for (;;) {
      let i = r - 1;
      while (i >= 0 && indices[i] === i + n - r) i -= 1;
      if (i < 0) return out;
      indices[i]! += 1;
      for (let j = i + 1; j < r; j += 1) indices[j] = indices[j - 1]! + 1;
      out.push(indices.map((k) => pool[k]));
    }
  }, ["iterable", "r"]);

  mod.combinations_with_replacement = annotate((...args: Any[]) => {
    const [iterableRaw, rRaw] = splitCall(
      "combinations_with_replacement", ["iterable", "r"], args,
    );
    const pool = toArray(
      required(iterableRaw, "combinations_with_replacement", "iterable"),
      "combinations_with_replacement",
    );
    const r = asNonNegInt(
      required(rRaw, "combinations_with_replacement", "r"),
      "combinations_with_replacement", "r",
    );
    const n = pool.length;
    const out: Any[][] = [];
    if (r === 0) return [[]];
    if (n === 0) return out;
    const indices = new Array<number>(r).fill(0);
    out.push(indices.map((i) => pool[i]));
    for (;;) {
      let i = r - 1;
      while (i >= 0 && indices[i] === n - 1) i -= 1;
      if (i < 0) return out;
      const next = indices[i]! + 1;
      for (let j = i; j < r; j += 1) indices[j] = next;
      out.push(indices.map((k) => pool[k]));
    }
  }, ["iterable", "r"]);

  mod.compress = annotate((...args: Any[]) => {
    const [dataRaw, selectorsRaw] = splitCall("compress", ["data", "selectors"], args);
    const data = toArray(required(dataRaw, "compress", "data"), "compress");
    const selectors = toArray(required(selectorsRaw, "compress", "selectors"), "compress");
    const out: Any[] = [];
    for (let i = 0; i < data.length && i < selectors.length; i += 1) {
      if (selectors[i]) out.push(data[i]);
    }
    return out;
  }, ["data", "selectors"]);

  mod.count = annotate((...args: Any[]) => {
    const [startRaw, stepRaw] = splitCall("count", ["start", "step"], args);
    const start = isNone(startRaw) ? 0 : Number(startRaw);
    const step = isNone(stepRaw) ? 1 : Number(stepRaw);
    function* gen(): Generator<number> {
      let value = start;
      for (;;) {
        yield value;
        value += step;
      }
    }
    return gen();
  }, ["start", "step"]);

  mod.cycle = annotate((...args: Any[]) => {
    const [iterableRaw] = splitCall("cycle", ["iterable"], args);
    const pool = toArray(required(iterableRaw, "cycle", "iterable"), "cycle");
    function* gen(): Generator<Any> {
      if (pool.length === 0) return;
      for (;;) for (const elt of pool) yield elt;
    }
    return gen();
  }, ["iterable"]);

  mod.dropwhile = annotate((...args: Any[]) => {
    const [predRaw, iterableRaw] = splitCall("dropwhile", ["predicate", "iterable"], args);
    const pred = required(predRaw, "dropwhile", "predicate") as Any;
    const out: Any[] = [];
    let dropping = true;
    for (const elt of iterEach(required(iterableRaw, "dropwhile", "iterable"), "dropwhile")) {
      if (dropping && pred(elt)) continue;
      dropping = false;
      out.push(elt);
    }
    return out;
  }, ["predicate", "iterable"]);

  mod.filterfalse = annotate((...args: Any[]) => {
    const [predRaw, iterableRaw] = splitCall("filterfalse", ["predicate", "iterable"], args);
    const pred = required(predRaw, "filterfalse", "predicate") as Any;
    const out: Any[] = [];
    for (const elt of iterEach(required(iterableRaw, "filterfalse", "iterable"), "filterfalse")) {
      const keep = isNone(pred) ? !elt : !pred(elt);
      if (keep) out.push(elt);
    }
    return out;
  }, ["predicate", "iterable"]);

  mod.groupby = annotate((...args: Any[]) => {
    const [iterableRaw, keyRaw] = splitCall("groupby", ["iterable", "key"], args);
    const pool = toArray(required(iterableRaw, "groupby", "iterable"), "groupby");
    const keyFn = isNone(keyRaw) ? undefined : (keyRaw as Any);
    const out: [Any, Any[]][] = [];
    let i = 0;
    const keyOf = (elt: Any): Any => (keyFn === undefined ? elt : keyFn(elt));
    while (i < pool.length) {
      const key = keyOf(pool[i]);
      const group: Any[] = [pool[i]];
      i += 1;
      while (i < pool.length && keyOf(pool[i]) === key) {
        group.push(pool[i]);
        i += 1;
      }
      out.push([key, group]);
    }
    return out;
  }, ["iterable", "key"]);

  mod.islice = annotate((...args: Any[]) => {
    const { positional, kwargs } = splitRaw(args);
    if (Object.keys(kwargs).length > 0) {
      throw new TypeError(`islice() got an unexpected keyword argument '${Object.keys(kwargs)[0]}'`);
    }
    let start = 0;
    let stop: Any;
    let step = 1;
    if (positional.length === 2) {
      stop = positional[1];
    } else if (positional.length === 3) {
      start = positional[1] === null || positional[1] === undefined ? 0 : asIndex(positional[1], "start");
      stop = positional[2];
    } else if (positional.length === 4) {
      start = positional[1] === null || positional[1] === undefined ? 0 : asIndex(positional[1], "start");
      stop = positional[2];
      step = positional[3] === null || positional[3] === undefined ? 1 : asIndex(positional[3], "step");
    } else {
      throw new TypeError(`islice() takes 2, 3 or 4 positional arguments but ${positional.length} were given`);
    }
    if (positional.length >= 3 && start < 0) throw new RangeError("islice(): start must be a non-negative integer or None");
    if (step <= 0) throw new RangeError("islice(): step must be a positive integer or None");
    const iterable = required(positional[0], "islice", "iterable");
    const stopN = stop === null || stop === undefined ? Infinity : asIndex(stop, "stop");
    if (stopN !== Infinity && stopN < 0) throw new RangeError("islice(): stop must be a non-negative integer or None");
    const out: Any[] = [];
    let index = 0;
    let emit = 0;
    for (const elt of iterEach(iterable, "islice")) {
      if (index >= stopN) break;
      if (index >= start && (index - start) % step === 0) {
        out.push(elt);
        emit += 1;
        if (start + emit * step >= stopN && step > 0) {
          // Enough emitted; keep draining guard cheap by continuing to break
          // once the index passes stop (handled at loop head).
        }
      }
      index += 1;
    }
    return out;
  }, ["iterable", "stop"]);

  mod.pairwise = annotate((...args: Any[]) => {
    const [iterableRaw] = splitCall("pairwise", ["iterable"], args);
    const pool = toArray(required(iterableRaw, "pairwise", "iterable"), "pairwise");
    const out: Any[][] = [];
    for (let i = 0; i + 1 < pool.length; i += 1) out.push([pool[i], pool[i + 1]]);
    return out;
  }, ["iterable"]);

  mod.permutations = annotate((...args: Any[]) => {
    const [iterableRaw, rRaw] = splitCall("permutations", ["iterable", "r"], args);
    const pool = toArray(required(iterableRaw, "permutations", "iterable"), "permutations");
    const n = pool.length;
    const r = isNone(rRaw) ? n : asNonNegInt(rRaw, "permutations", "r");
    const out: Any[][] = [];
    if (r > n) return out;
    if (r === 0) return [[]];
    // Heap-style index cycles, mirroring CPython's `permutations` order.
    const indices = Array.from({ length: n }, (_, i) => i);
    const cycles = Array.from({ length: r }, (_, i) => n - i);
    out.push(indices.slice(0, r).map((i) => pool[i]));
    for (;;) {
      let done = true;
      for (let i = r - 1; i >= 0; i -= 1) {
        cycles[i]! -= 1;
        if (cycles[i] === 0) {
          const first = indices[i]!;
          for (let j = i; j < n - 1; j += 1) indices[j] = indices[j + 1]!;
          indices[n - 1] = first;
          cycles[i] = n - i;
        } else {
          const j = n - cycles[i]!;
          const tmp = indices[i]!;
          indices[i] = indices[j]!;
          indices[j] = tmp;
          out.push(indices.slice(0, r).map((k) => pool[k]));
          done = false;
          break;
        }
      }
      if (done) return out;
    }
  }, ["iterable", "r"]);

  mod.product = annotate((...args: Any[]) => {
    const { positional, kwargs } = splitRaw(args);
    let repeat = 1;
    for (const key of Object.keys(kwargs)) {
      if (key === "repeat") repeat = asIndex(kwargs[key], "repeat");
      else throw new TypeError(`product() got an unexpected keyword argument '${key}'`);
    }
    if (repeat < 0) throw new RangeError("product(): repeat must be a non-negative integer");
    const pools = positional.map((p) => toArray(p, "product"));
    const repeated: Any[][] = [];
    for (let i = 0; i < repeat; i += 1) repeated.push(...pools);
    if (repeated.length === 0) return [[]];
    let out: Any[][] = [[]];
    for (const pool of repeated) {
      const next: Any[][] = [];
      for (const prefix of out) {
        for (const elt of pool) next.push([...prefix, elt]);
      }
      out = next;
    }
    return out;
  }, []);

  mod.repeat = annotate((...args: Any[]) => {
    const [objectRaw, timesRaw] = splitCall("repeat", ["object", "times"], args);
    const object = required(objectRaw, "repeat", "object");
    if (isNone(timesRaw)) {
      function* gen(): Generator<Any> {
        for (;;) yield object;
      }
      return gen();
    }
    const n = asNonNegInt(timesRaw, "repeat", "times");
    return new Array(n).fill(object);
  }, ["object", "times"]);

  mod.starmap = annotate((...args: Any[]) => {
    const [funcRaw, iterableRaw] = splitCall("starmap", ["function", "iterable"], args);
    const func = required(funcRaw, "starmap", "function") as Any;
    const out: Any[] = [];
    for (const elt of iterEach(required(iterableRaw, "starmap", "iterable"), "starmap")) {
      const callArgs = Array.isArray(elt) ? elt : toArray(elt, "starmap");
      out.push(func(...callArgs));
    }
    return out;
  }, ["function", "iterable"]);

  mod.takewhile = annotate((...args: Any[]) => {
    const [predRaw, iterableRaw] = splitCall("takewhile", ["predicate", "iterable"], args);
    const pred = required(predRaw, "takewhile", "predicate") as Any;
    const out: Any[] = [];
    for (const elt of iterEach(required(iterableRaw, "takewhile", "iterable"), "takewhile")) {
      if (!pred(elt)) break;
      out.push(elt);
    }
    return out;
  }, ["predicate", "iterable"]);

  mod.tee = annotate((...args: Any[]) => {
    const [iterableRaw, nRaw] = splitCall("tee", ["iterable", "n"], args);
    const pool = toArray(required(iterableRaw, "tee", "iterable"), "tee");
    const n = isNone(nRaw) ? 2 : asNonNegInt(nRaw, "tee", "n");
    return Array.from({ length: n }, () => pool.slice());
  }, ["iterable", "n"]);

  mod.zip_longest = annotate((...args: Any[]) => {
    const { positional, kwargs } = splitRaw(args);
    let fillvalue: Any = undefined;
    for (const key of Object.keys(kwargs)) {
      if (key === "fillvalue") fillvalue = kwargs[key];
      else throw new TypeError(`zip_longest() got an unexpected keyword argument '${key}'`);
    }
    const pools = positional.map((p) => toArray(p, "zip_longest"));
    const longest = pools.reduce((m, p) => Math.max(m, p.length), 0);
    const out: Any[][] = [];
    for (let i = 0; i < longest; i += 1) {
      out.push(pools.map((p) => (i < p.length ? p[i] : fillvalue)));
    }
    return out;
  }, []);

  return mod as PythonItertoolsModule;
}
