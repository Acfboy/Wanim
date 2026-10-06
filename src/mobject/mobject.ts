import { add, array, concatenate, multiply, subtract, type NDArray } from "numpy-ts";
import { MethodWithArgs } from "../data_structures.js";
import { DEFAULT_MOBJECT_TO_EDGE_BUFFER, DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, DL, DOWN, IN, LEFT, MED_SMALL_BUFF, ORIGIN, OUT, RIGHT, TAU, UP } from "../constants.js";
import { BLACK, ManimColor, WHITE, color_gradient, interpolate_color, type ParsableManimColor } from "../utils/color/core.js";
import { angle_between_vectors, cross, normalize, rotation_matrix } from "../utils/space_ops.js";
import { MultiAnimationOverrideException } from "../utils/exceptions.js";
import { remove_list_redundancies } from "../utils/iterables.js";
import { straight_path, type PathFuncType } from "../utils/paths.js";
import { addPoints, isNDArray, linspace, nestedRows, norm, numeric, pointAt, pointFrom, pointRows, pointsFrom, pythonTypeName, scalePoint, subtractPoints, toPythonRepr, toPythonString, _cContiguousRows } from "../utils/arrays.js";
import { Animation } from "../animation/animation.js";
import { config } from "../_config/utils.js";

export type MobjectUpdater = ((mobject: Mobject) => void) | ((mobject: Mobject, dt: number) => void);
export type MobjectArray = Mobject[];

/** TS mirror of a Python ``slice`` (as passed to ``Mobject.__getitem__``). */
export interface MobjectSlice {
  start?: number | null;
  stop?: number | null;
  step?: number | null;
}

/**
 * Mirrors ``slice.indices(length)``: normalizes start/stop/step exactly like
 * Python (including the ``-1`` stop sentinel for negative steps).
 */
export function sliceRange(
  length: number,
  slice: MobjectSlice,
): { start: number; stop: number; step: number } {
  let step = slice.step ?? null;
  if (step === null || step === undefined) step = 1;
  if (step === 0) {
    const error = new Error("slice step cannot be zero");
    error.name = "ValueError";
    throw error;
  }
  let lower: number;
  let upper: number;
  if (step > 0) {
    lower = 0;
    upper = length;
  } else {
    lower = -1;
    upper = length - 1;
  }
  let start = slice.start ?? null;
  if (start === null || start === undefined) {
    start = step < 0 ? upper : lower;
  } else {
    if (!Number.isInteger(start)) start = Math.trunc(start);
    if (start < 0) start = Math.max(start + length, lower);
    else start = Math.min(start, upper);
  }
  let stop = slice.stop ?? null;
  if (stop === null || stop === undefined) {
    stop = step < 0 ? lower : upper;
  } else {
    if (!Number.isInteger(stop)) stop = Math.trunc(stop);
    if (stop < 0) stop = Math.max(stop + length, lower);
    else stop = Math.min(stop, upper);
  }
  return { start, stop, step };
}

/** Materializes ``range(start, stop, step)`` values (mirrors Python ``range``). */
export function rangeValues(start: number, stop: number, step: number): number[] {
  const result: number[] = [];
  if (step > 0) {
    for (let i = start; i < stop; i += step) result.push(i);
  } else if (step < 0) {
    for (let i = start; i > stop; i += step) result.push(i);
  }
  return result;
}

/** Applies Python slice semantics to an array (mirrors ``list[slice]``). */
export function applySlice<T>(items: T[], slice: MobjectSlice): T[] {
  const r = sliceRange(items.length, slice);
  return rangeValues(r.start, r.stop, r.step).map((i) => items[i] as T);
}

export interface MobjectOptions {
  color?: ParsableManimColor | null;
  name?: string;
  dim?: number;
  target?: Mobject | null;
  z_index?: number;
  [key: string]: unknown;
}

/** Options bag accepted by ``Mobject.arrange`` (mirrors its Python kwargs). */
export interface ArrangeOptions {
  direction?: readonly number[] | NDArray;
  buff?: number;
  center?: boolean;
  aligned_edge?: readonly number[] | NDArray;
  [key: string]: unknown;
}

export type FunctionLike = (...args: any[]) => any;
const overrideFunctions = new WeakMap<object, FunctionLike>();

export function get_animation_override(method: FunctionLike): FunctionLike | undefined {
  return overrideFunctions.get(method) ?? (method as FunctionLike & { _override_animate?: FunctionLike })._override_animate;
}

function cloneValue(value: unknown, seen = new Map<object, unknown>()): unknown {
  if (value === null || value === undefined || typeof value !== "object") return value;
  if (seen.has(value as object)) return seen.get(value as object);
  if (isNDArray(value)) return value.copy();
  if (value instanceof ManimColor) return new ManimColor(value);
  if (Array.isArray(value)) {
    const result = new Array(value.length);
    seen.set(value, result);
    for (let index = 0; index < value.length; index += 1) result[index] = cloneValue(value[index], seen);
    return result;
  }
  if (value instanceof Set) {
    const result = new Set<unknown>();
    seen.set(value, result);
    for (const item of value) result.add(cloneValue(item, seen));
    return result;
  }
  if (value instanceof Map) {
    const result = new Map<unknown, unknown>();
    seen.set(value, result);
    for (const [key, item] of value) result.set(key, cloneValue(item, seen));
    return result;
  }
  if (value instanceof Mobject) return value.copy_with_seen(seen);
  if (value instanceof Date) return new Date((value as Date).getTime());
  if (ArrayBuffer.isView(value)) return (value as unknown as { slice(): unknown }).slice();
  const proto = Object.getPrototypeOf(value);
  if (proto === Object.prototype || proto === null) {
    const object = value as Record<string, unknown>;
    const result: Record<string, unknown> = {};
    seen.set(value, result);
    const keys = Object.keys(object);
    for (let index = 0; index < keys.length; index += 1) {
      const key = keys[index] as string;
      result[key] = cloneValue(object[key], seen);
    }
    return result;
  }
  const cloneable = value as {
    __deepcopy__?: (memo?: Map<object, unknown>) => unknown;
    copy?: () => unknown;
    clone?: () => unknown;
  };
  if (typeof cloneable.__deepcopy__ === "function") {
    const result = cloneable.__deepcopy__(seen);
    seen.set(value, result);
    return result;
  }
  if (typeof cloneable.copy === "function") {
    try {
      const result = cloneable.copy();
      seen.set(value, result);
      return result;
    } catch {
      // Fall through to prototype-preserving field copy below.
    }
  }
  if (typeof cloneable.clone === "function") {
    try {
      const result = cloneable.clone();
      seen.set(value, result);
      return result;
    } catch {
      // Fall through to prototype-preserving field copy below.
    }
  }
  // Generic class instance (e.g. _ScaleBase subclasses like LinearBase):
  // preserve the prototype so methods survive Mobject.copy()/generate_target().
  // Mirrors Python copy.deepcopy, which keeps the class.
  const object = value as Record<string, unknown>;
  const result: Record<string, unknown> = Object.create(proto);
  seen.set(value, result);
  const keys = Object.keys(object);
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index] as string;
    result[key] = cloneValue(object[key], seen);
  }
  return result;
}

function vector(value: unknown, dim = 3): number[] {
  return pointFrom(value, dim).toArray() as number[];
}

/**
 * Hint appended to every non-finite geometry error. In JS there is no numpy
 * broadcasting, so `3 * UP` evaluates `3 * [0,1,0]` to `NaN` instead of
 * `[0,3,0]`; that silently poisons every downstream point and only explodes
 * much later (renderer errors / truncated videos). See Mobject.shift/scale/
 * move_to/set_points and renderer/json_renderer.ts.
 */
export const NON_FINITE_GEOMETRY_HINT =
  "non-finite point/vector value (NaN/Infinity). Common cause: `scalar * DIRECTION` (e.g. `3 * UP`) — JS does not broadcast; use `[0,3,0]` or `scalePoint(UP, 3)` instead.";

/** Build the actionable non-finite error (optionally prefixed with context). */
export function nonFiniteGeometryError(context?: string): Error {
  return new Error(context ? `${context}. ${NON_FINITE_GEOMETRY_HINT}` : NON_FINITE_GEOMETRY_HINT);
}

/** True if a number/array/NDArray/complex contains any non-finite component. */
export function hasNonFinite(value: unknown): boolean {
  if (typeof value === "number") return !Number.isFinite(value);
  if (typeof value === "bigint") return false;
  if (isNDArray(value)) {
    // Scan the backing typed array directly when the data is C-contiguous;
    // `toArray()` materialises every element (an extra full pass over the
    // points, per vmobject, per frame in the serializer).
    const arr = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
    const expected = arr.shape.reduce((product, dim) => product * dim, 1);
    if (
      arr.data !== undefined &&
      arr.data.length === expected &&
      (arr.flags === undefined || arr.flags.C_CONTIGUOUS !== false)
    ) {
      const data = arr.data;
      for (let index = 0; index < data.length; index += 1) {
        if (!Number.isFinite(Number(data[index]))) return true;
      }
      return false;
    }
    return hasNonFinite((value as NDArray).toArray());
  }
  if (Array.isArray(value)) return value.some(hasNonFinite);
  if (ArrayBuffer.isView(value)) {
    const view = value as unknown as ArrayLike<unknown>;
    for (let index = 0; index < view.length; index += 1) {
      if (!Number.isFinite(Number(view[index]))) return true;
    }
    return false;
  }
  if (value !== null && typeof value === "object" && "re" in value && "im" in value) {
    const complex = value as { re: number; im: number };
    return !Number.isFinite(Number(complex.re)) || !Number.isFinite(Number(complex.im));
  }
  return false;
}

/** Throw the actionable non-finite error when any component is NaN/Infinity. */
export function assertFinite(value: unknown, context?: string): void {
  if (hasNonFinite(value)) throw nonFiniteGeometryError(context);
}

/** Backwards/readability alias used by point-setting entry points. */
export function assertFinitePoints(value: unknown, context?: string): void {
  assertFinite(value, context);
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return value !== null && value !== undefined && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function";
}

/** True for a plain (non-array, non-Mobject, non-NDArray) options/kwargs bag.
 *  Class instances (ManimColor, Mobject, ...) are NOT bags, so a color or
 *  direction passed positionally is never mistaken for kwargs. */
export function isPlainOptionsObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  if (Array.isArray(value) || isNDArray(value)) return false;
  if (value instanceof Mobject || value instanceof Map || value instanceof Set) return false;
  if (ArrayBuffer.isView(value)) return false;
  if (typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function") return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** True for a mobject that carries no points and whose children are all
 *  childless leaves (Text / simple VGroups). `become` can pair their children
 *  one-to-one and skip structure alignment entirely. */
function isFlatLeafGroup(mob: Mobject): boolean {
  if (!mob.has_no_points()) return false;
  const submobjects = mob.submobjects;
  if (submobjects.length === 0) return false;
  for (const sub of submobjects) {
    if (sub.submobjects.length !== 0) return false;
  }
  return true;
}

/** True when `become` can copy `b`'s data into `a` index-wise: both trees have
 * the same shape (child counts and "has own points" at every node), so
 * `align_data` only pads/aligns point representations that the copy below
 * overwrites.  Skipping it avoids the subpath/bezier re-alignment on every
 * frame of a live `MathTex`/graph readout; the copied geometry is the
 * target's, so the visual result is unchanged (point counts follow the
 * target rather than the subdivision — see ). */
function canDirectBecome(a: Mobject, b: Mobject): boolean {
  if (a.has_points() !== b.has_points()) return false;
  const ca = a.submobjects;
  const cb = b.submobjects;
  if (ca.length !== cb.length) return false;
  for (let index = 0; index < ca.length; index += 1) {
    if (!canDirectBecome(ca[index] as Mobject, cb[index] as Mobject)) return false;
  }
  return true;
}

/**
 * Split a positional-argument list into positional values and a merged
 * options bag. A plain object whose keys intersect `optionKeys` is treated as
 * the trailing/leading Python `**kwargs` bag; everything else stays
 * positional. This lets `new Axes({ x_range, y_range })` and
 * `axes.plot(fn, range, { color })` mirror manim's keyword arguments.
 */
export function splitArgsAndOptions(
  args: unknown[],
  optionKeys?: ReadonlySet<string>,
): { positionals: unknown[]; options: Record<string, unknown> } {
  let end = args.length;
  while (end > 0 && args[end - 1] === undefined) end -= 1;
  const positionals: unknown[] = [];
  const options: Record<string, unknown> = {};
  for (const arg of args.slice(0, end)) {
    if (isPlainOptionsObject(arg)) {
      const keys = Object.keys(arg);
      if (optionKeys === undefined || keys.length === 0 || keys.some((key) => optionKeys.has(key))) {
        Object.assign(options, arg);
        continue;
      }
    }
    positionals.push(arg);
  }
  return { positionals, options };
}

function pointArray(rows: number[][]): NDArray {
  return array(rows) as NDArray;
}

/** True when a point backing store is a float64 typed array or a plain JS
 *  number array, i.e. writable with fractional transform results. Integer
 *  dtype arrays must use the numpy path so the transform upcasts them. */
function isFloatPointData(data: unknown): data is Float64Array | number[] {
  return data instanceof Float64Array || Array.isArray(data);
}

/** Select min/max/center from a pre-scanned column. Non-finite samples
 *  (NaN/Infinity from singular evaluations such as log(0)) are skipped so a
 *  single singular sample does not poison layout the way `np.min`/`np.max`
 *  do in Python; direct NaN/Infinity inputs still fail fast in
 *  shift/scale/move_to/set_points and the serializer. */
function extremumValue(min: number, max: number, finiteCount: number, key: number): number {
  if (finiteCount === 0) return 0;
  if (key < 0) return min;
  if (key === 0) return (min + max) / 2;
  return max;
}

export function repeat_array_impl(value: NDArray, count: number, dim = 3): NDArray {
  if (!Number.isInteger(count)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
  if (count <= 0) throw new TypeError("reduce() of empty iterable with no initial value");
  const rows = nestedRows(value).map((row) => Array.isArray(row) ? row.map((item) => numeric(item)) : [numeric(row)]);
  const width = rows[0]?.length ?? dim;
  const result: number[][] = [];
  for (let index = 0; index < count; index += 1) result.push(...rows.map((row) => [...row, ...Array.from({ length: Math.max(0, width - row.length) }, () => 0)]));
  return array(result.length > 0 ? result : []).reshape(result.length, width) as NDArray;
}

export const repeat_array = repeat_array_impl;

export function init_size(num: number | null | undefined, alignments: string | null | undefined, sizes: Iterable<unknown> | null | undefined): number | null {
  if (num !== null && num !== undefined) return num;
  if (alignments !== null && alignments !== undefined) return alignments.length;
  if (sizes !== null && sizes !== undefined) return [...sizes].length;
  return null;
}

export function init_sizes(sizes: readonly (number | null | undefined)[] | null | undefined, num: number, measures: readonly number[], name: string): number[] {
  const actual = sizes === null || sizes === undefined || sizes.length === 0 ? Array.from({ length: num }, () => null) : sizes;
  if (actual.length !== num) {
    const error = new Error(`${name} has a mismatching size.`);
    error.name = "ValueError";
    throw error;
  }
  return actual.map((size, index) => size === null || size === undefined ? measures[index] : size);
}

export function init_alignments(alignments: string | null | undefined, num: number, char_to_direction: Record<string, readonly number[]>, name: string, dir_: readonly number[] | NDArray, cell_alignment: readonly number[] | NDArray = [0, 0, 0]): number[][] {
  const direction = pointFrom(dir_, 3).toArray() as number[];
  const fallback = pointFrom(cell_alignment, 3).toArray() as number[];
  if (alignments === null || alignments === undefined) return Array.from({ length: num }, () => fallback.map((value, index) => value * (direction[index] ?? 0)));
  if (alignments.length !== num) {
    const error = new Error(`${name}_alignments has a mismatching size.`);
    error.name = "ValueError";
    throw error;
  }
    return [...alignments].map((character) => {
    const direction = char_to_direction[character];
    if (!direction) {
      const error = new Error(`'${character}'`);
      error.name = "KeyError";
      throw error;
    }
    return [...direction];
  });
}

// Protocol / introspection probes that test runners, JS engines and
// `JSON.stringify` make on arbitrary values. Reading an unknown Mobject
// attribute raises a Python-like AttributeError; these must instead resolve
// to a safe value so that pretty-printing and serialization do not fall into
// unbounded attribute enumeration (which previously OOM'd vitest's
// pretty-printer).  Hoisted to module scope: a fresh Set per Mobject creation
// dominated `copy()`/constructor profiles.
const PROTOCOL_PROPS = new Set([
  "then",
  "asymmetricMatch",
  "nodeType",
  "tagName",
  "$$typeof",
  "toJSON",
  "@@iterator",
  "@@toStringTag",
  "size",
]);

/**
 * Escape hatch from the Mobject Proxy.  Reading it returns the raw (un-proxied)
 * target object so internal hot paths (notably `copy_with_seen`) can enumerate
 * own properties without paying the `get` trap once per property.  Not a public
 * API and not enumerable (it only resolves through the trap).
 */
const RAW_MOBJECT_TARGET = Symbol("wanim.raw_mobject_target");

/** Raw/un-proxied target for a Mobject, or the value itself if it is not proxied. */
function rawMobjectTarget(value: Mobject): Mobject {
  const raw = (value as unknown as { [RAW_MOBJECT_TARGET]?: Mobject })[RAW_MOBJECT_TARGET];
  return raw ?? value;
}

function createMobjectProxy(target: Mobject): Mobject {
  return new Proxy(target, {
    get: (object, property, receiver) => {
      if (property === RAW_MOBJECT_TARGET) return object;
      if (property in object || typeof property === "symbol") return Reflect.get(object, property, receiver);
      // NOTE (error 373): rapydscript's `ρσ_interpolate_kwargs` probes
      // `f.__argnames__` (and `__handles_kwarg_interpolation__`) on any
      // callee via a plain attribute read. Mobject instances are Proxies
      // that raise AttributeError for unknown attributes, so passing a
      // builder/Mobject through that probe threw
      // `VGroup object has no attribute '__argnames__'` (e.g.
      // `bunch.animate(run_time=1.5)` where `bunch.animate` is a callable
      // builder Proxy). Dunders that are not real members must resolve to
      // `undefined` (like a plain JS function without the metadata) instead
      // of throwing, so the kwargs bridge takes its no-mapping path.
      if (typeof property === "string" && property.startsWith("__")) return undefined;
      if (property === "toJSON") {
        const typed = object as unknown as {
          get_num_points?: () => number;
          get_center?: () => number[];
          submobjects?: unknown[];
        };
        return () => ({
          name: (object as unknown as { name?: unknown }).name,
          type: object.constructor.name,
          num_points: typeof typed.get_num_points === "function" ? typed.get_num_points() : undefined,
          // Include the center so callers that serialize mobjects (e.g. the
          // render hash) still observe position changes.
          center: typeof typed.get_center === "function" ? [...typed.get_center()] : undefined,
          submobjects: Array.isArray(typed.submobjects) ? typed.submobjects.length : undefined,
        });
      }
      if (typeof property === "string" && property.startsWith("get_")) {
        return () => {
          const name = property.slice(4);
          if (!(name in object)) {
            const error = new Error(`${object.constructor.name} object has no attribute '${name}'`);
            error.name = "AttributeError";
            throw error;
          }
          return (object as unknown as Record<string, unknown>)[name];
        };
      }
      if (typeof property === "string" && property.startsWith("set_")) {
        return (value: unknown) => {
          (object as unknown as Record<string, unknown>)[property.slice(4)] = value;
          return receiver;
        };
      }
      if (PROTOCOL_PROPS.has(property) || (typeof property === "string" && property.startsWith("@@"))) return undefined;
      // Array-like view for rapydscript `f(*mobject)` splats, which compile to
      // `f.apply(thisArg, mobject)` and read `length` plus numeric indices
      // natively. Mirrors `__iter__` order (self first when it has points,
      // then submobjects), consistent with `__len__`/`__getitem__`.
      if (property === "length") return object.__len__();
      if (typeof property === "string" && /^(0|[1-9]\d*)$/.test(property)) {
        const index = Number(property);
        const count = object.__len__();
        if (index < count) {
          if (object.has_no_points()) return object.submobjects[index];
          return index === 0 ? receiver : object.submobjects[index - 1];
        }
        return undefined;
      }
      const error = new Error(`${object.constructor.name} object has no attribute '${String(property)}'`);
      error.name = "AttributeError";
      throw error;
    },
    // NOTE: native `Array.prototype.concat` spreads concat-spreadable objects
    // via `HasProperty` (not `Get`): without this trap `'0' in mobject` is
    // false, so a chained `a.concat(b).concat(c)` (rapydscript `f(*a, *b,
    // *c)`) keeps the first splat (custom `Mobject.concat`, iterator-based)
    // but turns every later Mobject splat into holes. Report the same
    // array-like view (`length` + numeric indices mirroring `__iter__`) here.
    has: (object, property) => {
      if (property === "length") return true;
      if (typeof property === "string" && /^(0|[1-9]\d*)$/.test(property)) {
        try {
          return Number(property) < object.__len__();
        } catch {
          return false;
        }
      }
      return Reflect.has(object, property);
    },
    set: (object, property, value, receiver) => Reflect.set(object, property, value, receiver),
  });
}

export type BackgroundRectangleFactory = (
  mobject: Mobject,
  color: ParsableManimColor | null | undefined,
  fill_opacity: number,
  kwargs: Record<string, unknown>,
) => Mobject;

let backgroundRectangleFactory: BackgroundRectangleFactory | null = null;

export function setBackgroundRectangleFactory(factory: BackgroundRectangleFactory | null): void {
  backgroundRectangleFactory = factory;
}

export interface ImageCapableCamera {
  capture_mobject(mobject: Mobject): void;
  get_image(): unknown;
}

let defaultCameraFactory: (() => ImageCapableCamera) | null = null;

export function setDefaultCameraFactory(factory: (() => ImageCapableCamera) | null): void {
  defaultCameraFactory = factory;
}

/**
 * Normalize the `stretch` / `about_point` / `about_edge` / trailing-kwargs
 * shapes for `match_*` / `rescale_to_fit` / `*_to_fit_*`.
 *
 * Mirrors Python `**kwargs` forwarding (`match_width(m, stretch=True)`):
 * without an explicit `stretch` slot the py2js kwargs bridge appends
 * `{stretch: true}` as a trailing object, which lands in `about_point` and
 * poisons `scale` with NaN ("non-finite point/vector value", error 166).
 * Also tolerates the pre-fix positional order `match_width(m, about_point)`
 * where a point arrives in the `stretch` slot.
 */
export function normalizeMatchKwargs(
  kwargs: Record<string, unknown> | undefined | null,
  stretch: unknown,
  about_point?: readonly number[] | NDArray | null | Record<string, unknown>,
  about_edge?: readonly number[] | NDArray | null | Record<string, unknown>,
): { stretch: boolean; about_point?: readonly number[] | NDArray | null; about_edge?: readonly number[] | NDArray | null } {
  let stretchFlag = false;
  let point = about_point as readonly number[] | NDArray | null | undefined;
  let edge = about_edge as readonly number[] | NDArray | null | undefined;
  const isBag = (value: unknown): value is Record<string, unknown> =>
    value !== null && typeof value === "object" && !Array.isArray(value)
    && !(value instanceof Mobject) && typeof (value as { tolist?: unknown }).tolist !== "function";
  const pullBag = (bag: Record<string, unknown>): void => {
    if ("stretch" in bag && stretchFlag === false) stretchFlag = Boolean((bag as Record<string, unknown>)["stretch"]);
    if ("about_point" in bag && (point === undefined || point === null)) point = (bag as Record<string, unknown>)["about_point"] as never;
    if ("about_edge" in bag && (edge === undefined || edge === null)) edge = (bag as Record<string, unknown>)["about_edge"] as never;
  };
  if (typeof stretch === "boolean" || typeof stretch === "number") {
    stretchFlag = Boolean(stretch);
  } else if (stretch !== undefined && stretch !== null) {
    if (isBag(stretch)) {
      pullBag(stretch as Record<string, unknown>);
    } else {
      // Legacy positional point in the `stretch` slot: shift it to about_point.
      point = (point ?? undefined) !== undefined ? point : undefined;
      if (point === undefined) {
        point = stretch as never;
      } else {
        // Both slots filled (should not happen): prefer explicit point slot.
        if (edge === undefined) edge = point as never;
        point = stretch as never;
      }
    }
  }
  if (isBag(point) && ("stretch" in (point as Record<string, unknown>) || "about_point" in (point as Record<string, unknown>) || "about_edge" in (point as Record<string, unknown>))) {
    const bag = point as unknown as Record<string, unknown>;
    point = undefined;
    pullBag(bag);
  }
  if (isBag(edge) && ("stretch" in (edge as Record<string, unknown>) || "about_point" in (edge as Record<string, unknown>) || "about_edge" in (edge as Record<string, unknown>))) {
    const bag = edge as unknown as Record<string, unknown>;
    edge = undefined;
    pullBag(bag);
  }
  if (kwargs !== undefined && kwargs !== null && typeof kwargs === "object") pullBag(kwargs);
  return { stretch: stretchFlag, about_point: (point ?? null) as never, about_edge: (edge ?? null) as never };
}

export class Mobject {
  name: string;
  dim: number;
  target: Mobject | null;
  z_index: number;
  z_index_group: Mobject | null = null;
  point_hash: string | null = null;
  submobjects: Mobject[] = [];
  updaters: MobjectUpdater[] = [];
  /**
   * Interaction callbacks (see `InteractiveScene`). Unlike updaters — which run
   * every frame — reactions are only applied while the scene is in its
   * interactive "blueprint" phase, so they can react to a control's value.
   */
  reactions: Array<(mobject: Mobject) => void> = [];
  updating_suspended = false;
  /** Backing store for the ``color`` accessor.
   *
   * NOTE: Python's ``Mobject.color`` is a plain attribute, while
   * ``VMobject.color`` is a live property. TS class fields are own data
   * properties that would shadow a subclass prototype accessor, so the base
   * class stores the value in ``_color`` and exposes ``color`` as an
   * accessor pair; ``VMobject`` overrides the pair. */
  protected _color!: ManimColor;
  get color(): ManimColor { return this._color; }
  set color(value: ManimColor) { this._color = value; }
  points: NDArray;
  saved_state: Mobject | null = null;
  original_id: string;
  background_rectangle: Mobject | null = null;
  readonly animation_overrides: Map<unknown, FunctionLike> = new Map();
  /** Per-class animation override table (mirrors Python's ``animation_overrides`` class attribute).
   *
   * NOTE: declared without an initializer (like Python's bare annotation):
   * each class gets its OWN table either via ``__init_subclass__`` or
   * lazily on the first ``add_animation_override`` call, so subclass
   * registrations never leak into the base class.
   */
  static declare animation_overrides: Map<unknown, FunctionLike>;
  /** Saved pre-``set_default`` constructor (mirrors Python's ``_original__init__``). */
  static _original__init__: unknown = undefined;

  /**
   * Mirrors Python ``Mobject.__init_subclass__``: resets the per-class
   * animation override table, registers intrinsic (decorated) animation
   * overrides, and records the original constructor.
   *
   * NOTE: Python invokes this hook automatically on subclass creation; TS
   * has no subclass-creation hook, so subclasses (or test harnesses
   * emulating class creation) call it explicitly, e.g.
   * ``MyMobject.__init_subclass__()``.
   */
  static __init_subclass__(..._kwargs: unknown[]): void {
    const cls = this as unknown as Record<string, unknown>;
    cls["animation_overrides"] = new Map();
    (this as unknown as typeof Mobject)._add_intrinsic_animation_overrides();
    cls["_original__init__"] = this;
  }

  constructor(options: MobjectOptions = {}) {
    const normalized = typeof options === "string" || Array.isArray(options) ? { color: options as ParsableManimColor } : options;
    const ctor = this.constructor as unknown as { _wanimDefaultKwargs?: Record<string, unknown> };
    const defaults = ctor._wanimDefaultKwargs ?? {};
    const merged: Record<string, unknown> = { ...defaults, ...normalized };
    this.name = (merged.name as string | undefined) ?? this.constructor.name;
    this.dim = (merged.dim as number | undefined) ?? 3;
    this.target = (merged.target as Mobject | null | undefined) ?? null;
    this.z_index = (merged.z_index as number | undefined) ?? 0;
    // NOTE: assign the backing field directly (not the accessor): VMobject
    // overrides the `color` accessor to rebuild the rgba arrays, whose
    // backing fields are not initialized yet while super() runs.
    this._color = new ManimColor((merged.color as ParsableManimColor | undefined) ?? WHITE);
    this.points = array([]) as NDArray;
    this.original_id = String(Math.random());
    for (const [key, value] of Object.entries(merged)) {
      if (!(key in this)) (this as Record<string, unknown>)[key] = value;
    }
    this.reset_points();
    this.generate_points();
    Mobject.prototype.init_colors.call(this);
    return createMobjectProxy(this);
  }

  static set_default(kwargs: Record<string, unknown> = {}): void {
    const ctor = this as unknown as { _wanimDefaultKwargs?: Record<string, unknown> };
    if (Object.keys(kwargs).length > 0) {
      ctor._wanimDefaultKwargs = { ...(ctor._wanimDefaultKwargs ?? {}), ...kwargs };
    } else {
      delete ctor._wanimDefaultKwargs;
    }
  }

  static setDefault(kwargs: Record<string, unknown> = {}): void {
    (this as unknown as typeof Mobject).set_default(kwargs);
  }

  animation_override_for(animationClass: unknown): FunctionLike | undefined {
    const fromInstance = this.animation_overrides.get(animationClass);
    if (fromInstance !== undefined) return fromInstance;
    const ctor = this.constructor as unknown as { animation_overrides?: Map<unknown, FunctionLike> };
    return ctor.animation_overrides?.get(animationClass);
  }

  animationOverrideFor(animationClass: unknown): FunctionLike | undefined {
    return this.animation_override_for(animationClass);
  }

  static add_animation_override(animationClass: unknown, overrideFunc: FunctionLike): void {
    const ctor = this as unknown as { animation_overrides?: Map<unknown, FunctionLike> };
    if (!ctor.animation_overrides) ctor.animation_overrides = new Map();
    if (ctor.animation_overrides.has(animationClass)) {
      throw new MultiAnimationOverrideException(
        `The animation ${(animationClass as { name?: string }).name} for ` +
        `${this.name} is overridden by more than one method.`,
      );
    }
    ctor.animation_overrides.set(animationClass, overrideFunc);
  }

  static addAnimationOverride(animationClass: unknown, overrideFunc: FunctionLike): void {
    (this as unknown as typeof Mobject).add_animation_override(animationClass, overrideFunc);
  }

  static _add_intrinsic_animation_overrides(): void {
    // Mirrors Python: scan the class (including inherited methods, subclass
    // first) for methods marked with the `override_animation` decorator
    // (i.e. carrying an `_override_animation` attribute) and register each
    // as an animation override. Dunder attributes are ignored.
    // NOTE: Python invokes this automatically per subclass via
    // `__init_subclass__` (which also resets the per-class override dict);
    // TS has no subclass-creation hook, so this must be called manually and
    // it registers into the (possibly inherited) override map.
    const seen = new Set<string>();
    const consider = (holder: Record<string, unknown>): void => {
      for (const name of Object.getOwnPropertyNames(holder)) {
        if (name.startsWith("__") || seen.has(name)) continue;
        seen.add(name);
        // NOTE: read via the descriptor so accessor properties (getters)
        // are not invoked; Python's getattr(cls, name) likewise returns
        // the descriptor object itself without calling it.
        const descriptor = Object.getOwnPropertyDescriptor(holder, name);
        const method = descriptor?.value;
        if (typeof method === "function") {
          const animationClass = (method as { _override_animation?: unknown })._override_animation;
          if (animationClass !== undefined) {
            (this as unknown as typeof Mobject).add_animation_override(animationClass, method as FunctionLike);
          }
        }
      }
    };
    let current: unknown = this;
    while (current !== null && current !== undefined && current !== Function.prototype) {
      // Scan instance methods on the prototype (mirrors dir(cls) entries
      // resolved on the class) as well as static members on the constructor.
      const proto = (current as { prototype?: Record<string, unknown> }).prototype;
      if (proto !== undefined && proto !== null && typeof proto === "object") {
        consider(proto);
        consider(current as Record<string, unknown>);
        if ((proto as unknown) === Mobject.prototype) break;
      }
      current = Object.getPrototypeOf(current);
    }
  }

  get animate(): _AnimationBuilder {
    return createAnimationBuilder(this);
  }

  get always(): _UpdaterBuilder {
    return createUpdaterBuilder(this);
  }

  get_array_attrs(): string[] {
    return ["points"];
  }

  getArrayAttrs(): string[] {
    return this.get_array_attrs();
  }

  get_updaters(): MobjectUpdater[] {
    return this.updaters;
  }

  getUpdaters(): MobjectUpdater[] {
    return this.get_updaters();
  }

  get_time_based_updaters(): MobjectUpdater[] {
    return this.updaters.filter((updater) => updater.length >= 2);
  }

  getTimeBasedUpdaters(): MobjectUpdater[] {
    return this.get_time_based_updaters();
  }

  has_time_based_updater(): boolean {
    return this.updaters.some((updater) => updater.length >= 2);
  }

  hasTimeBasedUpdater(): boolean {
    return this.has_time_based_updater();
  }

  get_family_updaters(): MobjectUpdater[] {
    return this.get_family().flatMap((mobject) => mobject.get_updaters());
  }

  getFamilyUpdaters(): MobjectUpdater[] {
    return this.get_family_updaters();
  }

  get_color(): ManimColor {
    return this.color;
  }

  getColor(): ManimColor {
    return this.get_color();
  }

  has_points(): boolean {
    return this.get_num_points() > 0;
  }

  has_no_points(): boolean {
    return !this.has_points();
  }

  get_num_points(): number {
    return Array.isArray(this.points) ? this.points.length : this.points.shape[0];
  }

  getNumPoints(): number {
    return this.get_num_points();
  }

  reset_points(): this {
    this.points = array([]).reshape(0, this.dim) as NDArray;
    return this;
  }

  generate_points(): this {
    return this;
  }

  init_colors(): this {
    return this;
  }

  _assert_valid_submobjects(submobjects: Iterable<Mobject>): this {
    return this._assert_valid_submobjects_internal(submobjects, Mobject);
  }

  _assert_valid_submobjects_internal(submobjects: Iterable<Mobject>, mob_class: Function): this {
    let index = 0;
    for (const submob of submobjects) {
      if (!(submob instanceof (mob_class as new (...args: any[]) => Mobject))) {
        let errorMessage = `Only values of type ${mob_class.name} can be added as submobjects of ${this.constructor.name}, but the value ${toPythonString(submob)} (at index ${index}) is of type ${pythonTypeName(submob)}.`;
        if ((submob as unknown) instanceof Mobject) errorMessage += " You can try adding this value into a Group instead.";
        throw new TypeError(errorMessage);
      }
      if (submob === this) {
        const error = new Error(`Cannot add ${this.constructor.name} as a submobject of itself (at index ${index}).`);
        error.name = "ValueError";
        throw error;
      }
      index += 1;
    }
    return this;
  }

  add(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    const values: Mobject[] = [];
    for (const item of mobjects) {
      if (item instanceof Mobject) values.push(item);
      else if (isIterable(item)) values.push(...item as Iterable<Mobject>);
      else values.push(item as Mobject);
    }
    this._assert_valid_submobjects(values);
    const unique = remove_list_redundancies(values);
    if (values.length !== unique.length) {
      console.warn(
        "Attempted adding some Mobject as a child more than once, " +
          "this is not possible. Repetitions are ignored.",
      );
    }
    this.submobjects = this.submobjects.filter((mobject) => !unique.includes(mobject));
    this.submobjects.push(...unique);
    return this;
  }

  insert(index: number, mobject: Mobject): this {
    this._assert_valid_submobjects([mobject]);
    this.submobjects.splice(index, 0, mobject);
    return this;
  }

  add_to_back(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    const values: Mobject[] = [];
    for (const item of mobjects) {
      if (item instanceof Mobject) values.push(item);
      else if (isIterable(item)) values.push(...item as Iterable<Mobject>);
      else values.push(item as Mobject);
    }
    this._assert_valid_submobjects(values);
    this.remove(...values);
    const unique = remove_list_redundancies(values);
    this.submobjects = [...unique, ...this.submobjects];
    return this;
  }

  addToBack(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    return this.add_to_back(...mobjects);
  }

  remove(...mobjects: Mobject[]): this {
    for (const mobject of mobjects) {
      const index = this.submobjects.indexOf(mobject);
      if (index >= 0) this.submobjects.splice(index, 1);
    }
    return this;
  }

  get_family(_recurse = true): Mobject[] {
    // Single pre-order traversal with an identity set. The previous
    // implementation recursed per child and ran O(n^2) `remove_list_redundancies`
    // at every level; this is called several times per transform, so the
    // quadratic dedup dominated group transforms.
    const result: Mobject[] = [];
    const seen = new Set<Mobject>();
    const stack: Mobject[] = [this];
    while (stack.length > 0) {
      const node = stack.pop() as Mobject;
      if (seen.has(node)) continue;
      seen.add(node);
      result.push(node);
      const children = node.submobjects;
      for (let index = children.length - 1; index >= 0; index -= 1) {
        stack.push(children[index] as Mobject);
      }
    }
    return result;
  }

  getFamily(_recurse = true): Mobject[] {
    return this.get_family(_recurse);
  }

  family_members_with_points(): Mobject[] {
    return this.get_family().filter((mobject) => mobject.get_num_points() > 0);
  }

  familyMembersWithPoints(): Mobject[] {
    return this.family_members_with_points();
  }

  get_group_class(): typeof Group {
    return Group;
  }

  getGroupClass(): typeof Group {
    return this.get_group_class();
  }

  static get_mobject_type_class(): typeof Mobject {
    return Mobject;
  }

  static getMobjectTypeClass(): typeof Mobject {
    return Mobject.get_mobject_type_class();
  }

  copy(): this {
    return this.copy_with_seen(new Map<object, unknown>());
  }

  copy_with_seen(seen: Map<object, unknown>): this {
    if (seen.has(this)) return seen.get(this) as this;
    const raw = Object.create(Object.getPrototypeOf(this)) as this;
    const result = createMobjectProxy(raw);
    seen.set(this, result);
    // Enumerate the raw target: `this` is the Proxy, and `Object.entries(this)`
    // would invoke the get trap once per property (dozens of times per mobject,
    // for every mobject in a copy family).
    const source = rawMobjectTarget(this) as unknown as Record<string, unknown>;
    const rawTarget = raw as unknown as Record<string, unknown>;
    const keys = Object.keys(source);
    for (let index = 0; index < keys.length; index += 1) {
      const key = keys[index] as string;
      const value = source[key];
      // Primitives (the majority of fields) need no clone: `cloneValue` returns
      // them unchanged but the call itself is not free.
      if (value === null || value === undefined || typeof value !== "object") {
        rawTarget[key] = value;
      } else if (key === "target" && value instanceof Mobject) {
        rawTarget[key] = value.copy_with_seen(seen);
      } else {
        rawTarget[key] = cloneValue(value, seen);
      }
    }
    raw.original_id = String(Math.random());
    return result as this;
  }

  __deepcopy__(clone_from_id: Map<object, unknown> = new Map()): this {
    if (clone_from_id.has(this)) return clone_from_id.get(this) as this;
    const raw = Object.create(Object.getPrototypeOf(this)) as this;
    const result = createMobjectProxy(raw);
    clone_from_id.set(this, result);
    const source = rawMobjectTarget(this) as unknown as Record<string, unknown>;
    const rawTarget = raw as unknown as Record<string, unknown>;
    const keys = Object.keys(source);
    for (let index = 0; index < keys.length; index += 1) {
      const key = keys[index] as string;
      const value = source[key];
      if (value === null || value === undefined || typeof value !== "object") rawTarget[key] = value;
      else rawTarget[key] = cloneValue(value, clone_from_id);
    }
    raw.original_id = String(Math.random());
    return result as this;
  }

  generate_target(useDeepcopy = false): Mobject | null {
    this.target = null;
    this.target = this.copy();
    return this.target;
  }

  get_image(camera?: ImageCapableCamera | null): unknown {
    const cam = camera ?? (defaultCameraFactory !== null ? defaultCameraFactory() : null);
    if (cam === null || cam === undefined) {
      throw new Error("TODO: cairo rendering not ported yet (Mobject.get_image needs a Camera with a raster backend)");
    }
    cam.capture_mobject(this);
    return cam.get_image();
  }

  getImage(camera?: ImageCapableCamera | null): unknown {
    return this.get_image(camera);
  }

  show(camera?: ImageCapableCamera | null): void {
    (this.get_image(camera) as { show(): void }).show();
  }

  save_image(name: string | null = null): void {
    const image = this.get_image() as { save(path: string): void };
    const videoDir = config.get_dir("video_dir") ?? ".";
    const fileName = `${name ?? String(this.name)}.png`;
    image.save(`${videoDir}/${fileName}`);
  }

  saveImage(name: string | null = null): void {
    this.save_image(name);
  }

  get_center(): number[] {
    return this.get_critical_point(Array.from({ length: this.dim }, () => 0));
  }

  get_center_of_mass(): number[] {
    const all = pointRows(this.get_all_points(), this.dim);
    if (all.length === 0) return Array.from({ length: this.dim }, () => NaN);
    return Array.from(
      { length: this.dim },
      (_, index) => all.reduce((sum, point) => sum + (point[index] ?? 0), 0) / all.length,
    );
  }

  getCenterOfMass(): number[] {
    return this.get_center_of_mass();
  }

  getCenter(): number[] {
    return this.get_center();
  }

  get_critical_point(direction: readonly number[] | NDArray): number[] {
    const result = Array.from({ length: this.dim }, () => 0);
    const boundary = this.get_points_defining_boundary();
    const directionValues = Array.isArray(direction) && (direction.length === 0 || typeof (direction as readonly unknown[])[0] === "number")
      ? direction as readonly number[]
      : pointRows(direction, this.dim)[0] ?? [];
    const dim = this.dim;
    // Fast path: one flat pass over a C-contiguous boundary collects the
    // extrema for all three axes at once, instead of three `pointRows`/
    // `_cContiguousRows` passes. Non-finite samples (singularities such as
    // log(0) = -inf) are skipped so they do not poison layout (error 385).
    const flat = _cContiguousRows(boundary);
    if (flat !== null && flat.rowDim >= 3 && flat.rowCount > 0 && dim === 3) {
      const { data, rowDim, rowCount } = flat;
      let min0 = Infinity; let max0 = -Infinity; let finite0 = 0;
      let min1 = Infinity; let max1 = -Infinity; let finite1 = 0;
      let min2 = Infinity; let max2 = -Infinity; let finite2 = 0;
      for (let row = 0; row < rowCount; row += 1) {
        const base = row * rowDim;
        const v0 = Number(data[base]);
        if (Number.isFinite(v0)) { finite0 += 1; if (v0 < min0) min0 = v0; if (v0 > max0) max0 = v0; }
        const v1 = Number(data[base + 1]);
        if (Number.isFinite(v1)) { finite1 += 1; if (v1 < min1) min1 = v1; if (v1 > max1) max1 = v1; }
        const v2 = Number(data[base + 2]);
        if (Number.isFinite(v2)) { finite2 += 1; if (v2 < min2) min2 = v2; if (v2 > max2) max2 = v2; }
      }
      result[0] = extremumValue(min0, max0, finite0, Number(directionValues[0] ?? 0));
      result[1] = extremumValue(min1, max1, finite1, Number(directionValues[1] ?? 0));
      result[2] = extremumValue(min2, max2, finite2, Number(directionValues[2] ?? 0));
      return result;
    }
    for (let dim = 0; dim < this.dim; dim += 1) {
      result[dim] = this.get_extremum_along_dim(boundary, dim, Number(directionValues[dim] ?? 0));
    }
    return result;
  }

  getCriticalPoint(direction: readonly number[] = ORIGIN): number[] {
    return this.get_critical_point(direction);
  }

  get_left(): number[] { return this.get_critical_point([-1, 0, 0]); }
  get_right(): number[] { return this.get_critical_point([1, 0, 0]); }
  get_top(): number[] { return this.get_critical_point([0, 1, 0]); }
  get_bottom(): number[] { return this.get_critical_point([0, -1, 0]); }
  get_center_point(): number[] { return this.get_center(); }

  get_coord(dim: number, direction: readonly number[] | NDArray = ORIGIN): number {
    const directionValues = pointRows(direction, this.dim)[0] ?? [];
    return this.get_extremum_along_dim(null, dim, Number(directionValues[dim] ?? 0));
  }

  get_x(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(0, direction); }
  get_y(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(1, direction); }
  get_z(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(2, direction); }

  get_extremum_along_dim(points: unknown | null = null, dim = 0, key = 0): number {
    const source = points === null || points === undefined
      ? this.get_points_defining_boundary()
      : points;
    let min = Infinity;
    let max = -Infinity;
    let finiteCount = 0;
    // Fast path: read the requested column straight out of a C-contiguous
    // NDArray instead of materialising every row via `pointRows`.
    // Non-finite samples are skipped (see get_critical_point, error 385).
    const flat = _cContiguousRows(source);
    if (flat !== null) {
      const { data, rowDim, rowCount } = flat;
      if (dim < rowDim) {
        for (let row = 0; row < rowCount; row += 1) {
          const value = Number(data[row * rowDim + dim]);
          if (!Number.isFinite(value)) continue;
          finiteCount += 1;
          if (value < min) min = value;
          if (value > max) max = value;
        }
      } else {
        // Column missing: `pointRows` pads short rows with zeros.
        if (rowCount > 0) { min = 0; max = 0; finiteCount = rowCount; }
      }
    } else {
      const rows = pointRows(source, this.dim);
      for (let row = 0; row < rows.length; row += 1) {
        const value = (rows[row] as number[])[dim] ?? 0;
        if (!Number.isFinite(value)) continue;
        finiteCount += 1;
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
    if (finiteCount === 0) return 0;
    if (key < 0) return min;
    if (key === 0) return (min + max) / 2;
    return max;
  }

  get width(): number { return this.length_over_dim(0); }
  set width(value: number) { this.scale_to_fit_width(value); }
  get height(): number { return this.length_over_dim(1); }
  set height(value: number) { this.scale_to_fit_height(value); }
  get depth(): number { return this.length_over_dim(2); }
  set depth(value: number) { this.scale_to_fit_depth(value); }

  // NOTE: Python `get_width()` / `get_height()` / `get_depth()` (mirrors of
  // the `width` / `height` / `depth` properties).  They must exist as real
  // prototype methods, not only via the Mobject Proxy `get_*` fallback:
  // rapydscript subclasses of wanim classes (e.g. `class BitRow(VGroup)`)
  // are plain objects (the native `__init__` does `Object.assign(this,
  // new Ctor(...))`), so they never go through the Proxy `get` trap and
  // `rows[0].get_height()` on such an instance fails with
  // `TypeError: ...get_height is not a function` (error 349).  Same reason
  // `set_width` / `set_height` / `set_depth` below are real methods.
  get_width(): number { return this.width; }
  get_height(): number { return this.height; }
  get_depth(): number { return this.depth; }

  length_over_dim(dim: number): number {
    // Same extrema as `Math.max(...points.map(p => p[dim])) - Math.min(...)`,
    // but scanned straight from the C-contiguous backing stores: the old
    // version materialised every point row and spread them into `Math.max`,
    // which dominated `height`/`width`/`font_size` (and therefore every
    // `DecimalNumber.set_value`).  Non-finite samples are skipped so a
    // singular sample does not poison measurement (error 385).
    let min = Infinity;
    let max = -Infinity;
    let finiteCount = 0;
    const dimIndex = dim;
    for (const mobject of this.family_members_with_points()) {
      const points = (mobject as { points?: unknown }).points;
      const flat = _cContiguousRows(points);
      if (flat !== null) {
        const { data, rowDim, rowCount } = flat;
        if (dimIndex < rowDim) {
          for (let row = 0; row < rowCount; row += 1) {
            const value = Number(data[row * rowDim + dimIndex]);
            if (!Number.isFinite(value)) continue;
            finiteCount += 1;
            if (value < min) min = value;
            if (value > max) max = value;
          }
        } else {
          // `pointRows` pads short rows with zeros.
          if (rowCount > 0) {
            if (min > 0) min = 0;
            if (max < 0) max = 0;
            finiteCount += rowCount;
          }
        }
      } else {
        const rows = pointRows(points as never, this.dim);
        for (const point of rows) {
          const value = point[dimIndex] ?? 0;
          if (!Number.isFinite(value)) continue;
          finiteCount += 1;
          if (value < min) min = value;
          if (value > max) max = value;
        }
      }
    }
    if (finiteCount === 0) return 0;
    return max - min;
  }

  scale_to_fit_width(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 0, false, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  scale_to_fit_height(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 1, false, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  scale_to_fit_depth(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 2, false, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  stretch_to_fit_width(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 0, true, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  stretch_to_fit_height(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 1, true, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  stretch_to_fit_depth(value: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, undefined, about_point, about_edge);
    return this.rescale_to_fit(value, 2, true, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  // NOTE: 2019-era manim `set_width(width, stretch=False, ...)` /
  // `set_height` / `set_depth`. Modern manim removed these in favour of the
  // `width`/`height`/`depth` properties plus `scale_to_fit_*` /
  // `stretch_to_fit_*`. Keep the legacy names so 3b1b dataset scenes keep
  // working (e.g. `mu_box.set_width(1.1, stretch=True)`). Without explicit
  // methods the Mobject Proxy `set_*` fallback would silently ignore the
  // `stretch` flag and always scale.
  set_width(width: number, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.rescale_to_fit(width, 0, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  set_height(height: number, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.rescale_to_fit(height, 1, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }
  set_depth(depth: number, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.rescale_to_fit(depth, 2, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  rescale_to_fit(value: number, dim: number, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    const doStretch = opts.stretch;
    const current = this.length_over_dim(dim);
    if (current === 0) return this;
    if (doStretch) {
      this.stretch(value / current, dim, opts.about_point ?? undefined, opts.about_edge ?? undefined);
    } else {
      this.scale(value / current, opts.about_point ?? undefined, opts.about_edge ?? undefined);
    }
    return this;
  }

  shift_onto_screen(options: { buff?: number } = {}): this {
    const buff = options.buff ?? DEFAULT_MOBJECT_TO_EDGE_BUFFER;
    const space_lengths = [Number(config.get("frame_x_radius") ?? 7.111111111111111), Number(config.get("frame_y_radius") ?? 4)];
    for (const vect of [UP, DOWN, LEFT, RIGHT] as const) {
      const abs_vect = [...vect].map((value) => Math.abs(value));
      let dim = 0;
      for (let index = 1; index < abs_vect.length; index += 1) {
        if ((abs_vect[index] ?? 0) > (abs_vect[dim] ?? 0)) dim = index;
      }
      const max_val = (space_lengths[dim] ?? 0) - buff;
      const edge_center = this.get_edge_center(vect);
      const dot = edge_center.reduce((sum, value, index) => sum + value * ((vect as readonly number[])[index] ?? 0), 0);
      if (dot > max_val) {
        this.to_edge(vect, buff);
      }
    }
    return this;
  }

  is_off_screen(): boolean {
    if ((this.get_left()[0] ?? 0) > Number(config.get("frame_x_radius") ?? 7.111111111111111)) {
      return true;
    }
    if ((this.get_right()[0] ?? 0) < -Number(config.get("frame_x_radius") ?? 7.111111111111111)) {
      return true;
    }
    if ((this.get_bottom()[1] ?? 0) > Number(config.get("frame_y_radius") ?? 4)) {
      return true;
    }
    const rv: boolean = (this.get_top()[1] ?? 0) < -Number(config.get("frame_y_radius") ?? 4);
    return rv;
  }

  space_out_submobjects(factor = 1.5, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    this.scale(factor, about_point ?? undefined, about_edge ?? undefined);
    for (const submob of this.submobjects) {
      submob.scale(1.0 / factor);
    }
    return this;
  }

  put_start_and_end_on(start: readonly number[] | NDArray, end: readonly number[] | NDArray): this {
    const [current_start, current_end] = this.get_start_and_end();
    const current_vector = subtractPoints(current_end, current_start, this.dim);
    if (current_vector.every((value) => value === 0)) {
      console.warn(
        "put_start_and_end_on has been called on a closed loop or zero-length mobject. " +
        `${this.constructor.name} will be shifted to start point instead.`,
      );
      this.shift(subtractPoints(vector(start, this.dim), current_start, this.dim));
      return this;
    }
    const target_vector = subtractPoints(vector(end, this.dim), vector(start, this.dim), this.dim);
    const cross_product = cross(current_vector, target_vector);
    const axis = norm(cross_product) !== 0
      ? normalize(cross_product)
      : [...OUT];
    this.scale(
      norm(target_vector) / norm(current_vector),
      current_start,
    );
    this.rotate(
      angle_between_vectors(current_vector, target_vector),
      axis,
      current_start,
    );
    this.shift(subtractPoints(vector(start, this.dim), current_start, this.dim));
    return this;
  }

  get_zenith(): number[] { return this.get_edge_center(OUT); }
  get_nadir(): number[] { return this.get_edge_center(IN); }

  match_dim_size(mobject: Mobject, dim: number, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.rescale_to_fit(mobject.length_over_dim(dim), dim, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  match_width(mobject: Mobject, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.match_dim_size(mobject, 0, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  match_height(mobject: Mobject, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.match_dim_size(mobject, 1, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  match_depth(mobject: Mobject, stretch: boolean | Record<string, unknown> = false, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null, kwargs?: Record<string, unknown>): this {
    const opts = normalizeMatchKwargs(kwargs, stretch, about_point, about_edge);
    return this.match_dim_size(mobject, 2, opts.stretch, opts.about_point ?? undefined, opts.about_edge ?? undefined);
  }

  match_coord(mobject: Mobject, dim: number, direction: readonly number[] | NDArray = ORIGIN): this {
    return this.set_coord(
      mobject.get_coord(dim, direction),
      dim,
      direction,
    );
  }

  match_x(mobject: Mobject, direction: readonly number[] | NDArray = ORIGIN): this {
    return this.match_coord(mobject, 0, direction);
  }

  match_y(mobject: Mobject, direction: readonly number[] | NDArray = ORIGIN): this {
    return this.match_coord(mobject, 1, direction);
  }

  match_z(mobject: Mobject, direction: readonly number[] | NDArray = ORIGIN): this {
    return this.match_coord(mobject, 2, direction);
  }

  align_to(mobject_or_point: Mobject | readonly number[] | NDArray, direction: readonly number[] | NDArray = ORIGIN): this {
    let point: number[];
    if (mobject_or_point instanceof Mobject) {
      point = mobject_or_point.get_critical_point(direction);
    } else {
      point = vector(mobject_or_point, this.dim);
    }
    const direction_values = vector(direction, this.dim);
    for (let dim = 0; dim < this.dim; dim += 1) {
      if ((direction_values[dim] ?? 0) !== 0) {
        this.set_coord(point[dim] ?? 0, dim, direction);
      }
    }
    return this;
  }

  set_colors_by_radial_gradient(
    center?: readonly number[] | NDArray | null,
    radius = 1,
    inner_color: ParsableManimColor = WHITE,
    outer_color: ParsableManimColor = BLACK,
  ): this {
    this.set_submobject_colors_by_radial_gradient(
      center,
      radius,
      inner_color,
      outer_color,
    );
    return this;
  }

  set_submobject_colors_by_radial_gradient(
    center?: readonly number[] | NDArray | null,
    radius = 1,
    inner_color: ParsableManimColor = WHITE,
    outer_color: ParsableManimColor = BLACK,
  ): this {
    const resolved_center = center === undefined || center === null
      ? this.get_center()
      : vector(center, this.dim);
    for (const mob of this.family_members_with_points()) {
      let t = norm(subtractPoints(mob.get_center(), resolved_center, this.dim)) / radius;
      t = Math.min(t, 1);
      const mob_color = interpolate_color(
        new ManimColor(inner_color), new ManimColor(outer_color), t,
      );
      mob.set_color(mob_color, null, false);
    }
    return this;
  }

  sort(point_to_num_func: ((point: number[]) => number) | null = null, submob_func: ((mobject: Mobject) => number) | null = null): this {
    const point_func = point_to_num_func ?? ((point: number[]): number => point[0] ?? 0);
    let key_func = submob_func;
    if (key_func === null || key_func === undefined) {
      key_func = (m: Mobject): number => point_func(m.get_center());
    }
    const resolved = key_func;
    this.submobjects.sort((a, b) => {
      const key_a = resolved(a);
      const key_b = resolved(b);
      if (key_a < key_b) return -1;
      if (key_a > key_b) return 1;
      return 0;
    });
    return this;
  }

  arrange_submobjects(...args: unknown[]): this {
    const [direction, buff, center] = args as [readonly number[] | undefined, number | undefined, boolean | undefined];
    return this.arrange(direction ?? RIGHT, buff ?? DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, center ?? true);
  }

  sort_submobjects(...args: unknown[]): this {
    const [point_to_num_func, submob_func] = args as [((point: number[]) => number) | null | undefined, ((mobject: Mobject) => number) | null | undefined];
    return this.sort(point_to_num_func ?? null, submob_func ?? null);
  }

  set_z_index_by_z_Point3D(): this {
    const z_coord = this.get_center()[this.get_center().length - 1] ?? 0;
    this.set_z_index(z_coord);
    return this;
  }

  scale(scale_factor: number | readonly number[] | NDArray, about_point?: readonly number[] | NDArray | Record<string, unknown> | null, about_edge?: readonly number[] | NDArray | Record<string, unknown> | null): this {
    // NOTE (py2js kwargs bridge): a trailing kwargs bag such as
    // `{about_point: ORIGIN}` must not land verbatim in the point slot (NaN
    // poison). The wrapped-method bridge usually maps it via `__argnames__`
    // (pinned below), but `animate` paths and direct trailing bags are
    // unwrapped here defensively.
    const isBagLike = (value: unknown): value is Record<string, unknown> => {
      if (value === null || value === undefined || typeof value !== "object") return false;
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return false;
      if (isNDArray(value)) return false;
      if (value instanceof Mobject) return false;
      return true;
    };
    let resolved_about_point = about_point as readonly number[] | NDArray | null | undefined;
    let resolved_about_edge = about_edge as readonly number[] | NDArray | null | undefined;
    for (const slot of [about_point, about_edge] as unknown[]) {
      if (isBagLike(slot)) {
        const bag = slot as Record<string, unknown>;
        if (resolved_about_point === undefined && bag["about_point"] !== undefined) resolved_about_point = bag["about_point"] as never;
        else if (resolved_about_point === undefined && bag["aboutPoint"] !== undefined) resolved_about_point = bag["aboutPoint"] as never;
        if (resolved_about_edge === undefined && bag["about_edge"] !== undefined) resolved_about_edge = bag["about_edge"] as never;
        else if (resolved_about_edge === undefined && bag["aboutEdge"] !== undefined) resolved_about_edge = bag["aboutEdge"] as never;
        // A whole kwargs bag arrived in the `about_point` slot (unmapped
        // bridge shape): it is only options, not a point.
        if (slot === about_point && isBagLike(about_point)) resolved_about_point = (resolved_about_point ?? undefined) as never;
        if (slot === about_edge && isBagLike(about_edge)) resolved_about_edge = (resolved_about_edge ?? undefined) as never;
      }
    }
    if (isBagLike(resolved_about_point)) resolved_about_point = undefined;
    if (isBagLike(resolved_about_edge)) resolved_about_edge = undefined;
    assertFinite(scale_factor);
    if (resolved_about_point !== undefined && resolved_about_point !== null) assertFinite(vector(resolved_about_point, this.dim));
    if (resolved_about_edge !== undefined && resolved_about_edge !== null) assertFinite(vector(resolved_about_edge, this.dim));
    const center = resolved_about_point !== undefined && resolved_about_point !== null
      ? vector(resolved_about_point, this.dim)
      : this.get_critical_point(resolved_about_edge ?? ORIGIN);
    const dim = this.dim;
    // NOTE: Python `scale` does `scale_factor * points` with numpy
    // broadcasting, so a vector factor like `[2.0, 0.7, 1.0]` scales each
    // dimension independently (used by 3b1b's `orbit.scale([2.0, 0.7, 1.0])`).
    // The previous scalar-only fast path computed `number * JSArray` -> NaN,
    // poisoning every point ("non-finite point/vector value", error 336).
    let uniformFactor: number | null = null;
    let vectorFactors: number[] | null = null;
    let factorForNumpy: number | NDArray | null = null;
    if (typeof scale_factor === "number" || typeof scale_factor === "bigint") {
      uniformFactor = Number(scale_factor);
      factorForNumpy = uniformFactor;
    } else {
      let rawLen: number | null = null;
      let singleValue: number | null = null;
      if (Array.isArray(scale_factor)) {
        rawLen = scale_factor.length;
        if (rawLen === 1) singleValue = Number(scale_factor[0]);
      } else if (ArrayBuffer.isView(scale_factor)) {
        const flatArr = Array.from(scale_factor as unknown as ArrayLike<unknown>);
        rawLen = flatArr.length;
        if (rawLen === 1) singleValue = Number(flatArr[0]);
      } else if (isNDArray(scale_factor)) {
        const shape = (scale_factor as NDArray).shape;
        const count = shape.length === 0 ? 1 : shape.reduce((acc, size) => acc * size, 1);
        rawLen = count;
        if (count === 1) {
          const flat = (scale_factor as NDArray).toArray() as unknown;
          singleValue = Number(Array.isArray(flat) ? (flat as unknown[])[0] ?? flat : flat);
        }
      }
      if (singleValue !== null && rawLen === 1) {
        uniformFactor = singleValue;
        factorForNumpy = uniformFactor;
      } else if (rawLen !== null && rawLen === dim) {
        const row = pointRows(scale_factor, dim)[0] ?? Array.from({ length: dim }, () => 1);
        vectorFactors = Array.from({ length: dim }, (_, index) => Number(row[index] ?? 1));
        factorForNumpy = array([...vectorFactors]) as NDArray;
      } else if (rawLen === null) {
        const row = pointRows(scale_factor, dim)[0] ?? Array.from({ length: dim }, () => 1);
        vectorFactors = Array.from({ length: dim }, (_, index) => Number(row[index] ?? 1));
        factorForNumpy = array([...vectorFactors]) as NDArray;
      } else {
        // Length mismatch: let numpy broadcasting raise, mirroring Python's
        // ValueError, instead of silently padding/collapsing dimensions.
        if (Array.isArray(scale_factor)) factorForNumpy = array([...scale_factor]) as NDArray;
        else if (ArrayBuffer.isView(scale_factor)) factorForNumpy = array(Array.from(scale_factor as unknown as ArrayLike<number>)) as NDArray;
        else if (isNDArray(scale_factor)) factorForNumpy = scale_factor as NDArray;
        else factorForNumpy = array(pointRows(scale_factor, dim)[0] ?? []) as NDArray;
      }
    }
    if (vectorFactors !== null) assertFinite(vectorFactors);
    for (const mobject of this.family_members_with_points()) {
      // Fast path: `(p - center) * k + center` in one flat typed-array pass,
      // matching `apply_points_function_about_point`'s numpy op order.
      const flat = _cContiguousRows(mobject.points);
      if (flat !== null && flat.rowDim === dim && flat.rowCount > 0 && isFloatPointData(flat.data)) {
        if (vectorFactors === null && factorForNumpy !== null && typeof factorForNumpy !== "number") {
          // Mismatched vector length: fall through to the numpy path so the
          // broadcasting error surfaces instead of silently mis-scaling.
        } else {
          const { data, rowDim, rowCount } = flat;
          const out = mobject.points.copy() as NDArray;
          const outData = (out as NDArray & { data: Float64Array }).data;
          if (uniformFactor !== null) {
            for (let row = 0; row < rowCount; row += 1) {
              const base = row * rowDim;
              for (let d = 0; d < dim; d += 1) {
                outData[base + d] = ((data[base + d] ?? 0) - (center[d] ?? 0)) * uniformFactor + (center[d] ?? 0);
              }
            }
          } else if (vectorFactors !== null) {
            for (let row = 0; row < rowCount; row += 1) {
              const base = row * rowDim;
              for (let d = 0; d < dim; d += 1) {
                outData[base + d] = ((data[base + d] ?? 0) - (center[d] ?? 0)) * (vectorFactors[d] ?? 1) + (center[d] ?? 0);
              }
            }
          }
          mobject.points = out;
          continue;
        }
      }
      {
        const centerArray = array(center);
        const points = mobject.points as unknown as NDArray;
        const factor = uniformFactor !== null ? uniformFactor : (factorForNumpy ?? 1);
        mobject.points = add(multiply(subtract(points, centerArray) as NDArray, factor as never) as NDArray, centerArray) as NDArray;
      }
    }
    return this;
  }

  shift(...vectors: Array<readonly number[] | NDArray>): this {
    if (vectors.length === 0) {
      throw new TypeError("shift expected at least 1 argument, got 0");
    }
    let total = Array.from({ length: this.dim }, () => 0);
    for (const vectorValue of vectors) {
      const component = vector(vectorValue, this.dim);
      assertFinite(component);
      total = addPoints(total, component, this.dim);
    }
    assertFinite(total);
    // A zero shift is a no-op; skip the family walk (and the point-array
    // copies) that live layouts trigger every frame by re-"moving" a group to
    // the position it already occupies.
    if (total.every((value) => value === 0)) return this;
    const dim = this.dim;
    for (const mobject of this.family_members_with_points()) {
      // Fast path: C-contiguous (n, dim) points get a single flat typed-array
      // pass instead of materialising every row (`pointRows` -> `addPoints` ->
      // `pointArray`), which dominated per-tick transforms.
      const flat = _cContiguousRows(mobject.points);
      if (flat !== null && flat.rowDim === dim && flat.rowCount > 0 && isFloatPointData(flat.data)) {
        const { data, rowDim, rowCount } = flat;
        const out = mobject.points.copy() as NDArray;
        const outData = (out as NDArray & { data: Float64Array }).data;
        for (let row = 0; row < rowCount; row += 1) {
          const base = row * rowDim;
          for (let d = 0; d < dim; d += 1) {
            outData[base + d] = (data[base + d] ?? 0) + (total[d] ?? 0);
          }
        }
        mobject.points = out;
      } else {
        mobject.points = pointArray(pointRows(mobject.points, this.dim).map((point) => addPoints(point, total, this.dim)));
      }
    }
    return this;
  }

  move_to(
    pointOrMobject: Mobject | readonly number[] | NDArray,
    alignedEdge: readonly number[] | NDArray = ORIGIN,
    coorMask: readonly number[] | NDArray = [1, 1, 1],
  ): this {
    let target: number[];
    if (pointOrMobject instanceof Mobject) {
      target = pointOrMobject.get_critical_point(alignedEdge);
    } else {
      target = vector(pointOrMobject, this.dim);
    }
    assertFinite(target);
    const pointToAlign = this.get_critical_point(alignedEdge);
    const mask = vector(coorMask, this.dim);
    assertFinite(mask);
    const delta = subtractPoints(target, pointToAlign, this.dim).map((value, index) => value * (mask[index] ?? 1));
    return this.shift(delta);
  }

  center(): this { return this.shift(scalePoint(this.get_center(), -1)); }
  move_to_origin(): this { return this.move_to(ORIGIN); }
  set_x(value: number, direction: readonly number[] | NDArray = ORIGIN): this { return this.set_coord(value, 0, direction); }
  set_y(value: number, direction: readonly number[] | NDArray = ORIGIN): this { return this.set_coord(value, 1, direction); }
  set_z(value: number, direction: readonly number[] | NDArray = ORIGIN): this { return this.set_coord(value, 2, direction); }

  set_coord(value: number, dim: number, direction: readonly number[] | NDArray = ORIGIN): this {
    const current = this.get_coord(dim, direction);
    const shiftVector = Array.from({ length: this.dim }, () => 0);
    shiftVector[dim] = value - current;
    return this.shift(shiftVector);
  }

  align_on_border(direction: readonly number[] | NDArray, buff: number | Record<string, unknown> = DEFAULT_MOBJECT_TO_EDGE_BUFFER): this {
    // NOTE (py2js kwargs bridge, same as `scale` above): `animate.to_corner(
    // DR, buff=0.4)` compiles the `update` closure without `__argnames__`,
    // so the trailing kwargs bag can arrive verbatim as `buff` (an object,
    // not a number) and `buff * value` poisons the shift with NaN
    // ("non-finite point/vector value", error 253). Unwrap it defensively;
    // the wrapped-method bridge usually maps it via `__argnames__` (pinned
    // below), but `animate` paths bypass that mapping.
    let resolvedBuff: unknown = buff;
    if (
      resolvedBuff !== null &&
      typeof resolvedBuff === "object" &&
      !Array.isArray(resolvedBuff) &&
      !ArrayBuffer.isView(resolvedBuff) &&
      !isNDArray(resolvedBuff) &&
      !(resolvedBuff instanceof Mobject)
    ) {
      const bag = resolvedBuff as Record<string, unknown>;
      if (bag["buff"] !== undefined) resolvedBuff = bag["buff"];
      else resolvedBuff = DEFAULT_MOBJECT_TO_EDGE_BUFFER;
    }
    const buffNumber = Number(resolvedBuff);
    const directionValues = pointFrom(direction, this.dim).toArray() as number[];
    const sign = directionValues.map((value) => Math.sign(value));
    const frameRadii = [Number(config.get("frame_x_radius") ?? 7.111111111111111), Number(config.get("frame_y_radius") ?? 4), 0];
    const targetPoint = sign.map((value, index) => value * (frameRadii[index] ?? 0));
    const pointToAlign = this.get_critical_point(direction);
    const directionScaled = directionValues.map((value) => buffNumber * value);
    let shiftValue = targetPoint.map((value, index) => value - (pointToAlign[index] ?? 0) - (directionScaled[index] ?? 0));
    shiftValue = shiftValue.map((value, index) => value * Math.abs(sign[index] ?? 0));
    this.shift(shiftValue.slice(0, this.dim));
    return this;
  }

  to_corner(corner: readonly number[] | NDArray = DL, buff: number | Record<string, unknown> = DEFAULT_MOBJECT_TO_EDGE_BUFFER): this {
    return this.align_on_border(corner, buff as number);
  }

  to_edge(edge: readonly number[] | NDArray = LEFT, buff: number | Record<string, unknown> = DEFAULT_MOBJECT_TO_EDGE_BUFFER): this {
    return this.align_on_border(edge, buff as number);
  }

  rotate(angle: number, axis: readonly number[] | NDArray = [...OUT], about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    const rot_matrix = pointRows(rotation_matrix(angle, axis), 3);
    const dim = this.dim;
    const center = about_point !== undefined && about_point !== null
      ? vector(about_point, dim)
      : this.get_critical_point(about_edge ?? ORIGIN);
    const cx = center[0] ?? 0;
    const cy = center[1] ?? 0;
    const cz = center[2] ?? 0;
    const r00 = rot_matrix[0]?.[0] ?? 0;
    const r01 = rot_matrix[0]?.[1] ?? 0;
    const r02 = rot_matrix[0]?.[2] ?? 0;
    const r10 = rot_matrix[1]?.[0] ?? 0;
    const r11 = rot_matrix[1]?.[1] ?? 0;
    const r12 = rot_matrix[1]?.[2] ?? 0;
    const r20 = rot_matrix[2]?.[0] ?? 0;
    const r21 = rot_matrix[2]?.[1] ?? 0;
    const r22 = rot_matrix[2]?.[2] ?? 0;
    for (const mobject of this.family_members_with_points()) {
      // Fast path: apply the 3x3 rotation row-wise on the flat backing store,
      // reproducing the old `sum_j (p_j - c_j) * R_ij + c_i` accumulation.
      const flat = _cContiguousRows(mobject.points);
      if (flat !== null && flat.rowDim === dim && flat.rowCount > 0 && dim === 3 && isFloatPointData(flat.data)) {
        const { data, rowDim, rowCount } = flat;
        const out = mobject.points.copy() as NDArray;
        const outData = (out as NDArray & { data: Float64Array }).data;
        for (let row = 0; row < rowCount; row += 1) {
          const base = row * rowDim;
          const x = (data[base] ?? 0) - cx;
          const y = (data[base + 1] ?? 0) - cy;
          const z = (data[base + 2] ?? 0) - cz;
          outData[base] = ((0 + x * r00) + y * r01) + z * r02 + cx;
          outData[base + 1] = ((0 + x * r10) + y * r11) + z * r12 + cy;
          outData[base + 2] = ((0 + x * r20) + y * r21) + z * r22 + cz;
        }
        mobject.points = out;
      } else {
        const centerArray = array(center);
        const points = mobject.points as unknown as NDArray;
        const centered = subtract(points, centerArray) as NDArray;
        const result = pointArray(
          pointRows(centered, dim).map((point) =>
            rot_matrix.map((row) => point.reduce((sum, value, index) => sum + value * (row[index] ?? 0), 0)).slice(0, dim),
          ),
        );
        mobject.points = add(result, centerArray) as NDArray;
      }
    }
    return this;
  }

  rotate_about_origin(angle: number, axis: readonly number[] | NDArray = [...OUT]): this {
    return this.rotate(angle, axis, [...ORIGIN]);
  }

  flip(axis: readonly number[] | NDArray = [...UP], about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    return this.rotate(
      TAU / 2, axis, about_point ?? undefined, about_edge ?? undefined,
    );
  }

  pose_at_angle(about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    this.rotate(TAU / 14, addPoints([...RIGHT], [...UP], 3), about_point ?? undefined, about_edge ?? undefined);
    return this;
  }

  stretch(factor: number, dim: number, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    const center = about_point !== undefined && about_point !== null
      ? vector(about_point, this.dim)
      : this.get_critical_point(about_edge ?? ORIGIN);
    // Fast path: one flat pass per member (`(p[dim] - c) * f + c`), matching
    // `apply_points_function_about_point`'s op order without the per-row
    // numpy round trip.  `stretch_to_fit_*` runs per live bar/rectangle in
    // several benchmarks.
    const centerValue = center[dim] ?? 0;
    for (const mobject of this.family_members_with_points()) {
      const flat = _cContiguousRows(mobject.points);
      if (flat !== null && dim < flat.rowDim) {
        const { data, rowDim, rowCount } = flat;
        const out = mobject.points.copy() as NDArray;
        const outData = (out as NDArray & { data: Float64Array }).data;
        for (let row = 0; row < rowCount; row += 1) {
          const base = row * rowDim;
          outData[base + dim] = (Number(data[base + dim] ?? 0) - centerValue) * factor + centerValue;
        }
        mobject.points = out;
      } else {
        const func = (points: NDArray): NDArray => {
          const rows = pointRows(points, this.dim).map((point) => {
            const scaled = [...point];
            scaled[dim] = (scaled[dim] ?? 0) * factor;
            return scaled;
          });
          return pointArray(rows);
        };
        const points = mobject.points as unknown as NDArray;
        const centered = subtract(points, array(center)) as NDArray;
        const result = func(centered);
        mobject.points = add(isNDArray(result) ? result : (array(result) as NDArray), array(center)) as NDArray;
      }
    }
    return this;
  }

  stretch_about_point(factor: number, dim: number, point: readonly number[] | NDArray): this {
    return this.stretch(factor, dim, point);
  }

  apply_function(func: (point: number[]) => number[] | NDArray, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    let resolved_point: readonly number[] | NDArray | undefined = about_point ?? undefined;
    const resolved_edge: readonly number[] | NDArray | undefined = about_edge ?? undefined;
    if (resolved_point === undefined && resolved_edge === undefined) {
      resolved_point = ORIGIN;
    }
    const multi_mapping_function = (points: NDArray): NDArray => {
      return pointArray(
        pointRows(points, this.dim).map((point) => vector(func(point), this.dim)),
      );
    };
    this.apply_points_function_about_point(
      multi_mapping_function,
      resolved_point,
      resolved_edge,
    );
    return this;
  }

  apply_matrix(matrix: number[][] | NDArray, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    // Default to applying matrix about the origin, not mobjects center
    let resolved_point: readonly number[] | NDArray | undefined = about_point ?? undefined;
    const resolved_edge: readonly number[] | NDArray | undefined = about_edge ?? undefined;
    if (resolved_point === undefined && resolved_edge === undefined) {
      resolved_point = ORIGIN;
    }
    const matrix_rows: number[][] = isNDArray(matrix)
      ? pointRows(matrix, (matrix as NDArray).shape[1] ?? this.dim)
      : (matrix as number[][]).map((row) => [...row].map(Number));
    const full_matrix: number[][] = Array.from({ length: this.dim }, (_, row) =>
      Array.from({ length: this.dim }, (_, col) => (row === col ? 1 : 0)),
    );
    for (let row = 0; row < Math.min(this.dim, matrix_rows.length); row += 1) {
      for (let col = 0; col < Math.min(this.dim, matrix_rows[row]?.length ?? 0); col += 1) {
        (full_matrix[row] as number[])[col] = matrix_rows[row]?.[col] ?? 0;
      }
    }
    const dim = this.dim;
    const center = resolved_point !== undefined && resolved_point !== null
      ? vector(resolved_point, dim)
      : this.get_critical_point(resolved_edge ?? ORIGIN);
    // Fast path: apply the (dim x dim) matrix row-wise on the flat backing
    // store, mirroring `apply_points_function_about_point`'s numpy op order.
    const m00 = full_matrix[0]?.[0] ?? 0; const m01 = full_matrix[0]?.[1] ?? 0; const m02 = full_matrix[0]?.[2] ?? 0;
    const m10 = full_matrix[1]?.[0] ?? 0; const m11 = full_matrix[1]?.[1] ?? 0; const m12 = full_matrix[1]?.[2] ?? 0;
    const m20 = full_matrix[2]?.[0] ?? 0; const m21 = full_matrix[2]?.[1] ?? 0; const m22 = full_matrix[2]?.[2] ?? 0;
    const cx = center[0] ?? 0;
    const cy = center[1] ?? 0;
    const cz = center[2] ?? 0;
    for (const mobject of this.family_members_with_points()) {
      const flat = _cContiguousRows(mobject.points);
      if (flat !== null && flat.rowDim === dim && flat.rowCount > 0 && dim === 3 && isFloatPointData(flat.data)) {
        const { data, rowDim, rowCount } = flat;
        const out = mobject.points.copy() as NDArray;
        const outData = (out as NDArray & { data: Float64Array }).data;
        for (let row = 0; row < rowCount; row += 1) {
          const base = row * rowDim;
          const x = (data[base] ?? 0) - cx;
          const y = (data[base + 1] ?? 0) - cy;
          const z = (data[base + 2] ?? 0) - cz;
          outData[base] = ((0 + x * m00) + y * m01) + z * m02 + cx;
          outData[base + 1] = ((0 + x * m10) + y * m11) + z * m12 + cy;
          outData[base + 2] = ((0 + x * m20) + y * m21) + z * m22 + cz;
        }
        mobject.points = out;
      } else {
        const centerArray = array(center);
        const points = mobject.points as unknown as NDArray;
        const centered = subtract(points, centerArray) as NDArray;
        const result = pointArray(
          pointRows(centered, dim).map((point) =>
            full_matrix.map((_, row) => point.reduce((sum, value, col) => sum + value * ((full_matrix[row] as number[])[col] ?? 0), 0)),
          ),
        );
        mobject.points = add(result, centerArray) as NDArray;
      }
    }
    return this;
  }

  apply_points_function_about_point(func: (points: NDArray) => NDArray, about_point?: readonly number[] | NDArray | null, about_edge?: readonly number[] | NDArray | null): this {
    const center = about_point !== undefined && about_point !== null ? vector(about_point, this.dim) : this.get_critical_point(about_edge ?? ORIGIN);
    // Shift/apply/unshift with numpy broadcasting instead of materialising the
    // rows three times per submobject (`pointRows` -> `pointArray` -> `func` ->
    // `pointRows` -> `pointArray`).  This path runs for every submobject of
    // every transform/scale/rotate, so the allocations were a large cost.
    const centerArray = array(center);
    for (const mobject of this.family_members_with_points()) {
      const points = mobject.points as unknown as NDArray;
      const centered = subtract(points, centerArray) as NDArray;
      const result = func(centered);
      mobject.points = add(isNDArray(result) ? result : (array(result) as NDArray), centerArray) as NDArray;
    }
    return this;
  }

  apply_function_to_position(func: (point: number[]) => number[] | NDArray): this {
    this.move_to(func(this.get_center()));
    return this;
  }

  applyFunctionToPosition(func: (point: number[]) => number[] | NDArray): this {
    return this.apply_function_to_position(func);
  }

  apply_function_to_submobject_positions(func: (point: number[]) => number[] | NDArray): this {
    for (const submob of this.submobjects) submob.apply_function_to_position(func);
    return this;
  }

  applyFunctionToSubmobjectPositions(func: (point: number[]) => number[] | NDArray): this {
    return this.apply_function_to_submobject_positions(func);
  }

  apply_over_attr_arrays(func: (value: NDArray) => NDArray): this {    for (const attr of this.get_array_attrs()) {
      const current = (this as unknown as Record<string, NDArray>)[attr];
      if (current !== undefined) (this as unknown as Record<string, NDArray>)[attr] = func(current);
    }
    return this;
  }

  apply_complex_function(function_: (value: { re: number; im: number }) => ({ re: number; im: number } | { real: number; imag: number } | number | NDArray), about_point?: readonly number[], about_edge?: readonly number[]): this {
    const R3_func = (point: number[]): NDArray => {
      const [x, y, z = 0] = point;
      const xyComplex = function_({ re: x, im: y });
      if (isNDArray(xyComplex)) return xyComplex;
      if (typeof xyComplex === "number") return array([xyComplex, 0, z]) as NDArray;
      const real = "re" in xyComplex ? xyComplex.re : xyComplex.real;
      const imaginary = "re" in xyComplex ? xyComplex.im : xyComplex.imag;
      return array([real, imaginary, z]) as NDArray;
    };
    return this.apply_function(R3_func, about_point, about_edge);
  }

  repeat(count: number): this {
    const repeat_array = (value: NDArray): NDArray => repeat_array_impl(value, count, this.dim);
    for (const mobject of this.family_members_with_points()) mobject.apply_over_attr_arrays(repeat_array);
    return this;
  }

  repeat_submobject(submob: Mobject): Mobject {
    return submob.copy();
  }

  reverse_points(): this {
    for (const mobject of this.family_members_with_points()) {
      mobject.apply_over_attr_arrays((arr) => {
        const width = arr.shape.length > 1 ? (arr.shape[arr.shape.length - 1] ?? mobject.dim) : mobject.dim;
        return pointArray([...pointRows(arr, width)].reverse());
      });
    }
    return this;
  }

  copy_array_from(value: unknown): this {
    assertFinitePoints(value);
    this.points = pointsFrom(value, this.dim);
    return this;
  }

  set_points(points: unknown): this { return this.copy_array_from(points); }

  get_start(): number[] { this.throw_error_if_no_points("get_start"); return pointAt(this.points, 0, this.dim); }
  get_end(): number[] { this.throw_error_if_no_points("get_end"); return pointAt(this.points, this.get_num_points() - 1, this.dim); }
  get_midpoint(): NDArray { return this.point_from_proportion(0.5); }

  get_start_and_end(): [number[], number[]] {
    return [this.get_start(), this.get_end()];
  }

  get_edge_center(direction: readonly number[] | NDArray): number[] {
    return this.get_critical_point(direction);
  }

  get_corner(direction: readonly number[] | NDArray): number[] {
    return this.get_critical_point(direction);
  }

  get_boundary_point(direction: readonly number[] | NDArray): number[] {
    const allPoints = pointRows(this.get_points_defining_boundary(), this.dim);
    const directionValues = pointFrom(direction, this.dim).toArray() as number[];
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let index = 0; index < allPoints.length; index += 1) {
      const point = allPoints[index] ?? [];
      const value = point.reduce((sum, coordinate, component) => sum + coordinate * (directionValues[component] ?? 0), 0);
      if (value > bestValue) {
        bestValue = value;
        bestIndex = index;
      }
    }
    return [...(allPoints[bestIndex] ?? Array.from({ length: this.dim }, () => 0))];
  }

  point_from_proportion(alpha: number): NDArray {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  proportion_from_point(point: readonly number[]): number {
    throw new Error("Please override in a child class.");
  }

  get_pieces(n_pieces: number): Group {
    const template = this.copy();
    template.submobjects = [];
    const alphas = linspace(0, 1, n_pieces + 1);
    const parts = alphas.slice(0, -1).map((a1, index) => {
      const a2 = alphas[index + 1] ?? 1;
      // NOTE: pointwise_become_partial is defined on VMobject/PMobject, not
      // on the Mobject base (same as Python, where it resolves at runtime on
      // the concrete subclass); dispatch dynamically here.
      const part = template.copy() as unknown as Mobject & {
        pointwise_become_partial(mobject: Mobject, a: number, b: number): Mobject;
      };
      return part.pointwise_become_partial(this, a1, a2) as Mobject;
    });
    return new Group(...parts);
  }

  getPieces(n_pieces: number): Group {
    return this.get_pieces(n_pieces);
  }

  interpolate_color(mobject1: Mobject, mobject2: Mobject, alpha: number): this {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  save_state(): this {
    if ("saved_state" in this && this.saved_state !== undefined) {
      // Prevent exponential growth of data.
      this.saved_state = null;
    }
    this.saved_state = this.copy();
    return this;
  }

  restore(): this {
    if (this.saved_state === null) throw new Error("Trying to restore without having saved");
    this.become(this.saved_state);
    return this;
  }

  set_color(color: ParsableManimColor = "PURE_YELLOW", alpha: unknown = null, family = true): this {
    void alpha;
    this.color = new ManimColor(color);
    if (family) for (const submob of this.submobjects) submob.set_color(color, alpha, family);
    return this;
  }
  setColor(color: ParsableManimColor = "PURE_YELLOW", alpha: unknown = null, family = true): this { return this.set_color(color, alpha, family); }

  set_color_by_gradient(...colors: ParsableManimColor[]): this {
    this.set_submobject_colors_by_gradient(...colors);
    return this;
  }

  setColorByGradient(...colors: ParsableManimColor[]): this {
    return this.set_color_by_gradient(...colors);
  }

  set_submobject_colors_by_gradient(...colors: ParsableManimColor[]): this {
    if (colors.length === 0) {
      const error = new Error("Need at least one color");
      error.name = "ValueError";
      throw error;
    } else if (colors.length === 1) {
      return this.set_color(colors[0]);
    }
    const mobs = this.family_members_with_points();
    const newColors = color_gradient(colors, mobs.length);
    mobs.forEach((mob, index) => {
      mob.set_color(newColors[index] as ParsableManimColor, null, false);
    });
    return this;
  }

  setSubmobjectColorsByGradient(...colors: ParsableManimColor[]): this {
    return this.set_submobject_colors_by_gradient(...colors);
  }

  set_opacity(opacity: number, family = true): this {
    this.color = this.color.opacity(opacity);
    if (family) for (const submob of this.submobjects) submob.set_opacity(opacity, family);
    return this;
  }

  setOpacity(opacity: number, family = true): this { return this.set_opacity(opacity, family); }

  set_fill(color?: ParsableManimColor, opacity?: number, family = true): this {
    if (color !== undefined) this.set_color(color, null, family);
    if (opacity !== undefined) this.set_opacity(opacity, family);
    return this;
  }

  set_stroke(color?: ParsableManimColor, width?: number, opacity?: number, family = true): this {
    if (color !== undefined) this.set_color(color, null, family);
    if (opacity !== undefined) this.set_opacity(opacity, family);
    return this;
  }

  match_style(other: Mobject, family = true): this {
    this.set_color(other.get_color(), null, family);
    return this;
  }

  match_color(other: Mobject): this { return this.set_color(other.get_color()); }

  fade(darkness = 0.5, family = true): this {
    if (family) for (const submob of this.submobjects) submob.fade(darkness, family);
    return this;
  }

  to_original_color(): this {
    this.set_color(this.color);
    return this;
  }

  toOriginalColor(): this {
    return this.to_original_color();
  }

  fade_to(color: ParsableManimColor, alpha: number, family = true): this {
    if (this.get_num_points() > 0) this.set_color(interpolate_color(this.get_color(), new ManimColor(color), alpha), null, false);
    if (family) for (const submob of this.submobjects) submob.fade_to(color, alpha);
    return this;
  }

  add_background_rectangle(
    color?: ParsableManimColor | null,
    opacity = 0.75,
    kwargs: Record<string, unknown> = {},
  ): this {
    if (backgroundRectangleFactory === null) {
      throw new Error("TODO: add_background_rectangle needs BackgroundRectangle (manim.mobject.geometry.shape_matchers), not ported yet");
    }
    this.background_rectangle = backgroundRectangleFactory(this, color ?? null, opacity, kwargs);
    this.add_to_back(this.background_rectangle);
    return this;
  }

  addBackgroundRectangle(
    color?: ParsableManimColor | null,
    opacity = 0.75,
    kwargs: Record<string, unknown> = {},
  ): this {
    return this.add_background_rectangle(color, opacity, kwargs);
  }

  add_background_rectangle_to_submobjects(kwargs: Record<string, unknown> = {}): this {
    const { color = null, opacity = 0.75, ...rest } = kwargs as {
      color?: ParsableManimColor | null;
      opacity?: number;
      [key: string]: unknown;
    };
    for (const submobject of this.submobjects) submobject.add_background_rectangle(color, opacity, rest);
    return this;
  }

  addBackgroundRectangleToSubmobjects(kwargs: Record<string, unknown> = {}): this {
    return this.add_background_rectangle_to_submobjects(kwargs);
  }

  add_background_rectangle_to_family_members_with_points(kwargs: Record<string, unknown> = {}): this {
    const { color = null, opacity = 0.75, ...rest } = kwargs as {
      color?: ParsableManimColor | null;
      opacity?: number;
      [key: string]: unknown;
    };
    for (const mob of this.family_members_with_points()) mob.add_background_rectangle(color, opacity, rest);
    return this;
  }

  addBackgroundRectangleToFamilyMembersWithPoints(kwargs: Record<string, unknown> = {}): this {
    return this.add_background_rectangle_to_family_members_with_points(kwargs);
  }

  add_updater(update_function: MobjectUpdater, index: number | null = null, call_updater = false): this {
    if (index === null || index === undefined) this.updaters.push(update_function);
    else this.updaters.splice(index, 0, update_function);
    if (call_updater) {
      if (update_function.length >= 2) (update_function as (mobject: Mobject, dt: number) => void)(this, 0);
      else (update_function as (mobject: Mobject) => void)(this);
    }
    return this;
  }

  addUpdater(update_function: MobjectUpdater, index: number | null = null, call_updater = false): this {
    return this.add_updater(update_function, index, call_updater);
  }

  /**
   * Register an interaction reaction (same call shape as {@link add_updater}).
   * Reactions are only invoked by an `InteractiveScene` while it is in its
   * blueprint phase; see the class comment on `reactions`.
   */
  add_reaction(reaction: (mobject: Mobject) => void): this {
    this.reactions.push(reaction);
    return this;
  }

  addReaction(reaction: (mobject: Mobject) => void): this {
    return this.add_reaction(reaction);
  }

  remove_reaction(reaction: (mobject: Mobject) => void): this {
    this.reactions = this.reactions.filter((candidate) => candidate !== reaction);
    return this;
  }

  removeReaction(reaction: (mobject: Mobject) => void): this {
    return this.remove_reaction(reaction);
  }

  clear_reactions(recursive = true): this {
    this.reactions = [];
    if (recursive) for (const submob of this.submobjects) submob.clear_reactions();
    return this;
  }

  clearReactions(recursive = true): this {
    return this.clear_reactions(recursive);
  }

  remove_updater(updateFunction: MobjectUpdater): this {
    this.updaters = this.updaters.filter((updater) => updater !== updateFunction);
    return this;
  }

  update(dt = 0, recursive = true): this {
    if (this.updating_suspended) return this;
    for (const updater of [...this.updaters]) {
      if (updater.length >= 2) (updater as (mobject: Mobject, dt: number) => void)(this, dt);
      else (updater as (mobject: Mobject) => void)(this);
    }
    if (recursive) for (const submob of this.submobjects) submob.update(dt, recursive);
    return this;
  }

  clear_updaters(recursive = true): this {
    this.updaters = [];
    if (recursive) for (const submob of this.submobjects) submob.clear_updaters();
    return this;
  }

  clearUpdaters(recursive = true): this {
    return this.clear_updaters(recursive);
  }

  match_updaters(other: Mobject): this {
    this.clear_updaters();
    for (const updater of other.get_updaters()) this.add_updater(updater);
    return this;
  }

  matchUpdaters(other: Mobject): this {
    return this.match_updaters(other);
  }

  suspend_updating(recursive = true): this {
    this.updating_suspended = true;
    if (recursive) for (const submob of this.submobjects) submob.suspend_updating(recursive);
    return this;
  }

  resume_updating(recursive = true): this {
    this.updating_suspended = false;
    if (recursive) for (const submob of this.submobjects) submob.resume_updating(recursive);
    this.update(0, recursive);
    return this;
  }
  suspendUpdating(recursive = true): this { return this.suspend_updating(recursive); }
  resumeUpdating(recursive = true): this { return this.resume_updating(recursive); }

  apply_to_family(func: (mobject: Mobject) => void): this {
    for (const mob of this.family_members_with_points()) func(mob);
    return this;
  }

  applyToFamily(func: (mobject: Mobject) => void): this {
    return this.apply_to_family(func);
  }

  get_array_attrs_with_values(): Record<string, NDArray> {
    const result: Record<string, NDArray> = {};
    for (const attr of this.get_array_attrs()) result[attr] = (this as unknown as Record<string, NDArray>)[attr];
    return result;
  }

  set(options: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(options)) (this as unknown as Record<string, unknown>)[key] = value;
    return this;
  }

  __getattr__(attr: string): FunctionLike {
    if (attr.startsWith("get_")) {
      const to_get = attr.slice(4);
      return () => {
        if (!(to_get in this)) {
          const error = new Error(`${this.constructor.name} object has no attribute '${to_get}'`);
          error.name = "AttributeError";
          throw error;
        }
        return (this as unknown as Record<string, unknown>)[to_get];
      };
    }
    if (attr.startsWith("set_")) {
      const to_set = attr.slice(4);
      return (value: unknown) => {
        (this as unknown as Record<string, unknown>)[to_set] = value;
        return this;
      };
    }
    const error = new Error(`${this.constructor.name} object has no attribute '${attr}'`);
    error.name = "AttributeError";
    throw error;
  }

  __getattr_getter(): FunctionLike {
    return this.__getattr__("get_foo");
  }

  __getattr_setter(): FunctionLike {
    return this.__getattr__("set_foo");
  }

  arrange(
    direction: readonly number[] | NDArray | ArrangeOptions = RIGHT,
    buff: number | ArrangeOptions = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    center: boolean | ArrangeOptions = true,
    aligned_edge: readonly number[] | NDArray | Record<string, unknown> = ORIGIN,
    kwargs: Record<string, unknown> = {},
  ): this {
    // NOTE: mirrors Python `arrange(direction=RIGHT, buff=..., center=True,
    // **kwargs)` where the extra kwargs are forwarded to `next_to` (e.g.
    // `aligned_edge`). A leading options bag `{ direction, buff, center,
    // aligned_edge, ... }` is accepted (a raw direction is always an array,
    // so a plain object is unambiguously the kwargs bag).
    let resolvedDirection: readonly number[] | NDArray = direction as readonly number[];
    let resolvedBuff: number = buff as number;
    let resolvedCenter: boolean = center as boolean;
    let resolvedAlignedEdge: readonly number[] | NDArray = aligned_edge as readonly number[];
    let extraNextTo: Record<string, unknown> = {};
    if (isPlainOptionsObject(direction)) {
      const options = direction as ArrangeOptions;
      resolvedDirection = options.direction ?? RIGHT;
      resolvedBuff = options.buff ?? DEFAULT_MOBJECT_TO_MOBJECT_BUFFER;
      resolvedCenter = options.center ?? true;
      resolvedAlignedEdge = options.aligned_edge ?? ORIGIN;
      const { direction: _d, buff: _b, center: _c, aligned_edge: _a, ...rest } = options;
      extraNextTo = { ...rest };
    } else {
      if (isPlainOptionsObject(buff)) {
        // Defensive: a kwargs bag arrived in the `buff` slot (e.g. an older
        // py2js bridge that appended unknown kwargs as a trailing object).
        const options = buff as ArrangeOptions;
        if (options.direction !== undefined) resolvedDirection = options.direction;
        resolvedBuff = options.buff ?? DEFAULT_MOBJECT_TO_MOBJECT_BUFFER;
        if (options.center !== undefined) resolvedCenter = options.center;
        if (options.aligned_edge !== undefined) resolvedAlignedEdge = options.aligned_edge;
        const { direction: _d, buff: _b, center: _c, aligned_edge: _a, ...rest } = options;
        extraNextTo = { ...extraNextTo, ...rest };
      }
      if (isPlainOptionsObject(center)) {
        const options = center as unknown as ArrangeOptions;
        if (options.direction !== undefined) resolvedDirection = options.direction;
        if (options.buff !== undefined) resolvedBuff = options.buff;
        resolvedCenter = (options.center as boolean | undefined) ?? true;
        if (options.aligned_edge !== undefined) resolvedAlignedEdge = options.aligned_edge;
        const { direction: _d, buff: _b, center: _c, aligned_edge: _a, ...rest } = options;
        extraNextTo = { ...extraNextTo, ...rest };
      }
      if (isPlainOptionsObject(aligned_edge)) {
        const options = aligned_edge as unknown as Record<string, unknown>;
        // A vector is never a plain object, so this slot holding one means a
        // kwargs bag (old bridge shape or a direct JS call).
        if ("aligned_edge" in options || "direction" in options || "buff" in options || "center" in options) {
          const bag = options as unknown as ArrangeOptions;
          if (bag.direction !== undefined) resolvedDirection = bag.direction;
          if (bag.buff !== undefined) resolvedBuff = bag.buff;
          if (bag.center !== undefined) resolvedCenter = bag.center;
          if (bag.aligned_edge !== undefined) resolvedAlignedEdge = bag.aligned_edge;
          const { direction: _d, buff: _b, center: _c, aligned_edge: _a, ...rest } = bag;
          extraNextTo = { ...extraNextTo, ...rest };
        } else {
          extraNextTo = { ...extraNextTo, ...options };
        }
        resolvedAlignedEdge = resolvedAlignedEdge ?? ORIGIN;
      }
      if (isPlainOptionsObject(kwargs)) extraNextTo = { ...extraNextTo, ...(kwargs as Record<string, unknown>) };
    }
    if (resolvedAlignedEdge === undefined || resolvedAlignedEdge === null) resolvedAlignedEdge = ORIGIN;
    assertFinite(vector(resolvedDirection, this.dim));
    assertFinite(resolvedBuff);
    assertFinite(vector(resolvedAlignedEdge, this.dim));
    const submobjectToAlign = (extraNextTo.submobject_to_align as Mobject | null) ?? null;
    const indexOfSubmobjectToAlign = (extraNextTo.index_of_submobject_to_align as number | null) ?? null;
    const coorMask = (extraNextTo.coor_mask as readonly number[] | NDArray | undefined) ?? [1, 1, 1];
    for (let index = 0; index < this.submobjects.length - 1; index += 1) {
      this.submobjects[index + 1].next_to(
        this.submobjects[index],
        resolvedDirection,
        resolvedBuff,
        resolvedAlignedEdge,
        submobjectToAlign,
        indexOfSubmobjectToAlign,
        coorMask,
      );
    }
    if (resolvedCenter) this.center();
    return this;
  }

  next_to(
    mobject_or_point: Mobject | readonly number[] | NDArray,
    direction: readonly number[] | NDArray = RIGHT,
    buff: number | Record<string, unknown> = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    aligned_edge: readonly number[] | NDArray = ORIGIN,
    submobject_to_align: Mobject | null = null,
    index_of_submobject_to_align: number | null = null,
    coor_mask: readonly number[] | NDArray = [1, 1, 1],
  ): this {
    // NOTE: the parameter names mirror Python's `next_to` exactly so that the
    // py2js kwargs bridge can map Python keyword arguments such as
    // `aligned_edge=`/`coor_mask=` onto their positional slots.
    // NOTE (py2js kwargs bridge, same as `scale`/`align_on_border` above):
    // `animate.next_to(m, UP, buff=0.4)` compiles the `update` closure without
    // `__argnames__`, so the trailing kwargs bag can arrive verbatim as `buff`
    // (an object, not a number) and `buff * value` poisons the shift with NaN
    // ("non-finite point/vector value", error 262). Unwrap it defensively;
    // the wrapped-method bridge usually maps it via `__argnames__` (pinned
    // below), but `animate` paths bypass that mapping.
    let resolvedDirection = direction as readonly number[] | NDArray;
    let resolvedBuff: unknown = buff;
    let resolvedAlignedEdge = aligned_edge as readonly number[] | NDArray;
    let resolvedSubmobjectToAlign = submobject_to_align as Mobject | null;
    let resolvedIndexOfSubmobjectToAlign = index_of_submobject_to_align as number | null;
    let resolvedCoorMask = coor_mask as readonly number[] | NDArray;
    const isBagLike = (value: unknown): value is Record<string, unknown> => {
      if (value === null || value === undefined || typeof value !== "object") return false;
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return false;
      if (isNDArray(value)) return false;
      if (value instanceof Mobject) return false;
      return true;
    };
    for (const slot of [buff, direction, aligned_edge, coor_mask] as unknown[]) {
      if (!isBagLike(slot)) continue;
      const bag = slot as Record<string, unknown>;
      if (bag["buff"] !== undefined) resolvedBuff = bag["buff"];
      if (bag["direction"] !== undefined) resolvedDirection = bag["direction"] as never;
      if (bag["aligned_edge"] !== undefined) resolvedAlignedEdge = bag["aligned_edge"] as never;
      if (bag["submobject_to_align"] !== undefined) resolvedSubmobjectToAlign = bag["submobject_to_align"] as never;
      if (bag["index_of_submobject_to_align"] !== undefined) resolvedIndexOfSubmobjectToAlign = bag["index_of_submobject_to_align"] as never;
      if (bag["coor_mask"] !== undefined) resolvedCoorMask = bag["coor_mask"] as never;
    }
    if (isBagLike(resolvedBuff) && !Array.isArray(resolvedBuff)) {
      const bag = resolvedBuff as Record<string, unknown>;
      resolvedBuff = bag["buff"] ?? DEFAULT_MOBJECT_TO_MOBJECT_BUFFER;
    }
    const npDirection = pointFrom(resolvedDirection, this.dim).toArray() as number[];
    const npAlignedEdge = pointFrom(resolvedAlignedEdge, this.dim).toArray() as number[];
    let targetPoint: number[];
    if (mobject_or_point instanceof Mobject) {
      const mob = mobject_or_point;
      const targetAligner = resolvedIndexOfSubmobjectToAlign !== null && resolvedIndexOfSubmobjectToAlign !== undefined
        ? (mob.__getitem__(resolvedIndexOfSubmobjectToAlign) as Mobject)
        : mob;
      targetPoint = targetAligner.get_critical_point(
        addPoints(npAlignedEdge, npDirection, this.dim),
      );
    } else {
      targetPoint = vector(mobject_or_point, this.dim);
    }
    let aligner: Mobject;
    if (resolvedSubmobjectToAlign !== null && resolvedSubmobjectToAlign !== undefined) {
      aligner = resolvedSubmobjectToAlign;
    } else if (resolvedIndexOfSubmobjectToAlign !== null && resolvedIndexOfSubmobjectToAlign !== undefined) {
      aligner = this.__getitem__(resolvedIndexOfSubmobjectToAlign) as Mobject;
    } else {
      aligner = this;
    }
    const pointToAlign = aligner.get_critical_point(
      subtractPoints(npAlignedEdge, npDirection, this.dim),
    );
    const mask = vector(resolvedCoorMask, this.dim);
    const delta = addPoints(
      subtractPoints(targetPoint, pointToAlign, this.dim),
      scalePoint(npDirection, Number(resolvedBuff), this.dim),
      this.dim,
    ).map((value, index) => value * (mask[index] ?? 1));
    this.shift(delta);
    return this;
  }

  arrange_in_grid(
    rows: number | null | Record<string, unknown> = null,
    cols: number | null = null,
    buff: number | [number, number] = MED_SMALL_BUFF,
    cell_alignment: readonly number[] = ORIGIN,
    row_alignments: string | null = null,
    col_alignments: string | null = null,
    row_heights: Iterable<number | null | undefined> | null = null,
    col_widths: Iterable<number | null | undefined> | null = null,
    flow_order = "rd",
  ): this {
    // Accept a leading options bag (`arrange_in_grid({rows, cols, buff})`),
    // mirroring Python's keyword API. Animation/Mobject instances are not
    // plain objects, so this never misreads a positional numeric argument.
    if (rows !== null && typeof rows === "object" && !Array.isArray(rows)) {
      const opts = rows as Record<string, unknown>;
      return this.arrange_in_grid(
        (opts.rows as number | null) ?? null,
        (opts.cols as number | null) ?? null,
        (opts.buff as number | [number, number]) ?? MED_SMALL_BUFF,
        (opts.cell_alignment as readonly number[]) ?? ORIGIN,
        (opts.row_alignments as string | null) ?? null,
        (opts.col_alignments as string | null) ?? null,
        (opts.row_heights as Iterable<number | null | undefined> | null) ?? null,
        (opts.col_widths as Iterable<number | null | undefined> | null) ?? null,
        (opts.flow_order as string) ?? "rd",
      );
    }
    const resolvedRows = rows as number | null;
    const mobs = [...this.submobjects];
    const startPos = this.get_center();
    const row_heights_list = row_heights === null ? null : [...row_heights];
    const col_widths_list = col_widths === null ? null : [...col_widths];
    let columnCount = init_size(cols, col_alignments, col_widths_list);
    let rowCount = init_size(resolvedRows, row_alignments, row_heights_list);
    if (rowCount === null && columnCount === null) columnCount = Math.ceil(Math.sqrt(mobs.length));
    if (rowCount === null) rowCount = Math.ceil(mobs.length / (columnCount as number));
    if (columnCount === null) columnCount = Math.ceil(mobs.length / rowCount);
    if (rowCount * columnCount < mobs.length) {
      const error = new Error("Too few rows and columns to fit all submobjetcs.");
      error.name = "ValueError";
      throw error;
    }
    const buffX = Array.isArray(buff) ? buff[0] : buff;
    const buffY = Array.isArray(buff) ? buff[1] : buff;
    const rowDirections = init_alignments(row_alignments, rowCount, { u: [0, 1, 0], c: [0, 0, 0], d: [0, -1, 0] }, "row", [1, 0, 0], vector(cell_alignment, 3));
    const colDirections = init_alignments(col_alignments, columnCount, { l: [-1, 0, 0], c: [0, 0, 0], r: [1, 0, 0] }, "col", [0, 1, 0], vector(cell_alignment, 3));
    const mappers: Record<string, (r: number, c: number) => number> = {
      dr: (r, c) => (rowCount - r - 1) + c * rowCount,
      dl: (r, c) => (rowCount - r - 1) + (columnCount - c - 1) * rowCount,
      ur: (r, c) => r + c * rowCount,
      ul: (r, c) => r + (columnCount - c - 1) * rowCount,
      rd: (r, c) => (rowCount - r - 1) * columnCount + c,
      ld: (r, c) => (rowCount - r - 1) * columnCount + (columnCount - c - 1),
      ru: (r, c) => r * columnCount + c,
      lu: (r, c) => r * columnCount + (columnCount - c - 1),
    };
    const mapper = mappers[flow_order];
    if (!mapper) {
      const error = new Error('flow_order must be one of the following values: "dr", "rd", "ld" "dl", "ru", "ur", "lu", "ul".');
      error.name = "ValueError";
      throw error;
    }
    rowDirections.reverse();
    const rowHeightsList = row_heights_list ? [...row_heights_list].reverse() : [];
    const colWidthsList = col_widths_list ? [...col_widths_list] : [];
    const placeholder = new Mobject();
    while (mobs.length < rowCount * columnCount) mobs.push(placeholder);
    const grid = Array.from({ length: rowCount }, (_, r) => Array.from({ length: columnCount }, (_, c) => mobs[mapper(r, c)]));
    const measuredHeights = grid.map((row) => Math.max(...row.map((mob) => mob.height)));
    const measuredWidths = Array.from({ length: columnCount }, (_, c) => Math.max(...grid.map((row) => row[c].width)));
    const heights = init_sizes(rowHeightsList, rowCount, measuredHeights, "row_heights");
    const widths = init_sizes(colWidthsList, columnCount, measuredWidths, "col_widths");
    let y = 0;
    for (let r = 0; r < rowCount; r += 1) {
      let x = 0;
      for (let c = 0; c < columnCount; c += 1) {
        const mob = grid[r][c];
        if (mob !== placeholder) {
          const alignment = addPoints(rowDirections[r], colDirections[c], 3);
          const start = [x, y, 0];
          const end = [x + widths[c], y + heights[r], 0];
          const cell = new Mobject();
          cell.points = pointsFrom([start, end], 3);
          mob.move_to(cell, alignment);
        }
        x += widths[c] + buffX;
      }
      y += heights[r] + buffY;
    }
    return this.move_to(startPos);
  }

  set_z_index(z_index_value: number, family = true): this {
    if (family) for (const submob of this.submobjects) submob.set_z_index(z_index_value, family);
    this.z_index = z_index_value;
    return this;
  }

  setZIndex(value: number, family = true): this { return this.set_z_index(value, family); }

  get_z_index_reference_point(): number[] { return this.get_center(); }

  throw_error_if_no_points(methodName = "method"): void {
    if (!this.has_no_points()) return;
    const pointful = this.family_members_with_points();
    let message = `Cannot call ${this.constructor.name}.${methodName} because ${this.toString()} has no points.`;
    if (pointful.length > 0) message += ` Its family contains ${pointful.length} mobject${pointful.length === 1 ? "" : "s"} with points.`;
    const error = new Error(message);
    error.name = "ValueError";
    throw error;
  }

  become(
    mobject: Mobject,
    match_height = false,
    match_width = false,
    match_depth = false,
    match_center = false,
    stretch = false,
  ): this {
    if (stretch || match_height || match_width || match_depth || match_center) {
      mobject = mobject.copy();
      if (stretch) {
        mobject.stretch_to_fit_height(this.height);
        mobject.stretch_to_fit_width(this.width);
        mobject.stretch_to_fit_depth(this.depth);
      } else {
        if (match_height) mobject.match_height(this);
        if (match_width) mobject.match_width(this);
        if (match_depth) mobject.match_depth(this);
      }
      if (match_center) mobject.move_to(this.get_center());
    }
    // Fast path for group-like mobjects whose children are all leaves
    // (the common `Text.become(Text)` / `VGroup.become(VGroup)` case, e.g. a
    // live readout rebuilt every frame).  Every child's data is overwritten
    // below, so the faithful path's `align_data` only burns bezier
    // subdivision aligning shapes that are thrown away.  The child list is
    // trimmed/extended to match, then children are paired and copied.
    if (isFlatLeafGroup(this) && isFlatLeafGroup(mobject)) {
      const target = mobject.submobjects;
      if (this.submobjects.length > target.length) {
        this.submobjects = this.submobjects.slice(0, target.length);
      } else if (this.submobjects.length < target.length) {
        const additions = target.slice(this.submobjects.length).map((child) => child.copy());
        this.submobjects = [...this.submobjects, ...additions];
      }
      const selfFamily = this.get_family();
      const otherFamily = mobject.get_family();
      for (let index = 0; index < selfFamily.length; index += 1) {
        const sm1 = selfFamily[index]!;
        const sm2 = otherFamily[index]!;
        if ("align_rgbas" in (sm1 as object)) {
          const alignRgbas = (sm1 as unknown as { align_rgbas: (other: unknown) => unknown }).align_rgbas;
          if (typeof alignRgbas === "function") alignRgbas.call(sm1, sm2);
        }
        sm1.points = pointsFrom(sm2.points, sm1.dim);
        sm1.interpolate_color(sm1, sm2, 1);
      }
      return this;
    }
    // The target's curves need no subdivision for this pair: `align_data`'s
    // results are overwritten below anyway, so skip it.
    if (canDirectBecome(this, mobject)) {
      const selfFamily = this.get_family();
      const otherFamily = mobject.get_family();
      for (let index = 0; index < selfFamily.length; index += 1) {
        const sm1 = selfFamily[index]!;
        const sm2 = otherFamily[index]!;
        if ("align_rgbas" in (sm1 as object)) {
          const alignRgbas = (sm1 as unknown as { align_rgbas: (other: unknown) => unknown }).align_rgbas;
          if (typeof alignRgbas === "function") alignRgbas.call(sm1, sm2);
        }
        sm1.points = pointsFrom(sm2.points, sm1.dim);
        sm1.interpolate_color(sm1, sm2, 1);
      }
      return this;
    }
    this.align_data(mobject, true);
    const selfFamily = this.get_family();
    const otherFamily = mobject.get_family();
    for (let index = 0; index < selfFamily.length; index += 1) {
      const sm1 = selfFamily[index]!;
      const sm2 = otherFamily[index]!;
      sm1.points = pointsFrom(sm2.points, sm1.dim);
      sm1.interpolate_color(sm1, sm2, 1);
    }
    return this;
  }

  replace(other: Mobject, dim_to_match = 0, stretch = false): this {
    if (!other.get_num_points() && !other.submobjects.length) {
      const error = new Error("Attempting to replace mobject with no points");
      error.name = "Warning";
      throw error;
    }
    if (stretch) {
      this.stretch_to_fit_width(other.width);
      this.stretch_to_fit_height(other.height);
    } else {
      this.rescale_to_fit(
        other.length_over_dim(dim_to_match),
        dim_to_match,
        false,
      );
    }
    this.shift(subtractPoints(other.get_center(), this.get_center(), this.dim));
    return this;
  }

  surround(
    mobject: Mobject,
    dim_to_match = 0,
    stretch = false,
    buff: number = MED_SMALL_BUFF,
  ): this {
    this.replace(mobject, dim_to_match, stretch);
    const length = mobject.length_over_dim(dim_to_match);
    this.scale((length + buff) / length);
    return this;
  }

  match_points(other: Mobject, _copy_submobjects = true): this {
    for (let index = 0; index < this.get_family().length; index += 1) {
      const source = other.get_family()[index];
      if (source) this.get_family()[index].points = pointsFrom(source.points, this.dim);
    }
    return this;
  }

  align_data(other: Mobject, skipPointAlignment = false): this {
    this.null_point_align(other);
    this.align_submobjects(other);
    if (!skipPointAlignment) this.align_points(other);
    for (let index = 0; index < this.submobjects.length; index += 1) this.submobjects[index].align_data(other.submobjects[index]);
    return this;
  }

  get_point_mobject(center: number[] | readonly number[] | null = null): Mobject {
    void center;
    const error = new Error(
      `get_point_mobject not implemented for ${(this.constructor as unknown as { name: string }).name}`,
    );
    error.name = "NotImplementedError";
    throw error;
  }

  null_point_align(other: Mobject): this {
    if (this.has_no_points() && other.has_points()) other.push_self_into_submobjects();
    if (other.has_no_points() && this.has_points()) this.push_self_into_submobjects();
    return this;
  }

  align_submobjects(other: Mobject): this {
    // NOTE: mirrors Python exactly (add_n_more_submobjects duplicates
    // existing children instead of padding with empty Mobjects). Padding
    // with empty Mobjects never converges for (empty, pointed) pairs:
    // null_point_align pushes the pointed side down one level while the
    // empty side gains an empty child, recursing forever (observed as a
    // stack overflow aligning DashedLines with different dash counts).
    const n1 = this.submobjects.length;
    const n2 = other.submobjects.length;
    this.add_n_more_submobjects(Math.max(0, n2 - n1));
    other.add_n_more_submobjects(Math.max(0, n1 - n2));
    return this;
  }

  align_points(other: Mobject): this {
    const count1 = this.get_num_points();
    const count2 = other.get_num_points();
    if (count1 < count2) this.align_points_with_larger(other);
    else if (count2 < count1) other.align_points_with_larger(this);
    return this;
  }

  align_points_with_larger(_larger_mobject: Mobject): this {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  push_self_into_submobjects(): this {
    const copy = this.copy();
    copy.submobjects = [];
    this.reset_points();
    this.add(copy);
    return this;
  }

  add_n_more_submobjects(n: number): this | null {
    if (n === 0) return null;

    const curr = this.submobjects.length;
    if (curr === 0) {
      // If empty, simply add n point mobjects
      this.submobjects = Array.from({ length: n }, () => this.get_point_mobject());
      return null;
    }

    const target = curr + n;
    // TODO, factor this out to utils so as to reuse
    // with VMobject.insert_n_curves
    // NOTE: mirrors ``(np.arange(target) * curr) // target`` (integer
    // floored division on non-negative values, matching Math.floor).
    const repeat_indices = Array.from(
      { length: target },
      (_, k) => Math.floor((k * curr) / target),
    );
    const split_factors = Array.from(
      { length: curr },
      (_, i) => repeat_indices.filter((index) => index === i).length,
    );
    const new_submobs: Mobject[] = [];
    this.submobjects.forEach((submob, index) => {
      const sf = split_factors[index] ?? 0;
      new_submobs.push(submob);
      for (let k = 1; k < sf; k += 1) new_submobs.push(submob.copy().fade(1));
    });
    this.submobjects = new_submobs;
    return this;
  }

  *__iter__(): IterableIterator<Mobject> {
    if (this.has_points()) yield this;
    yield* this.submobjects;
  }

  *[Symbol.iterator](): IterableIterator<Mobject> {
    yield* this.__iter__();
  }

  /** Let native `Array.prototype.concat` spread Mobjects (e.g. rapydscript
   * `[...]` splats compiling to `[].concat(mobject)`), mirroring `__iter__`
   * order via the Proxy-provided `length`/indices. */
  get [Symbol.isConcatSpreadable](): boolean {
    return true;
  }

  __len__(): number { return this.submobjects.length + (this.has_points() ? 1 : 0); }

  /** Python `x in mobject` membership (``in``/``not in`` operators).
   *
   * manim's `Mobject` defines no `__contains__`, so CPython falls back to
   * `__iter__` and reports membership on identity-or-equality.  Mobjects use
   * default identity equality, hence an identity scan over `__iter__` order
   * (self first when it has points, then submobjects).
   *
   * This also keeps rapydscript's `ρσ_in(val, arr)` working on proxied
   * mobjects: it probes `typeof arr.__contains__`, and reading any other
   * missing attribute through the Mobject proxy raises `AttributeError`
   * instead of yielding `undefined`. */
  __contains__(value: unknown): boolean {
    for (const item of this.__iter__()) {
      if (item === value) return true;
    }
    return false;
  }

  split(): Mobject[] {
    const result: Mobject[] = this.get_num_points() > 0 ? [this] : [];
    return [...result, ...this.submobjects];
  }

  /**
   * JS-array-compatible `concat` so rapydscript `*` splats work on Mobjects.
   *
   * `f(*a, *b)` compiles to `a.concat(b)` and `f(*a, kw=...)` to
   * `a.concat([kwbag])`. Python `*a` iterates `a`, so spread each argument
   * the same way: plain arrays element-wise, other iterables (Mobjects,
   * VGroups, NDArrays, generators) via their iterator, anything else as a
   * single element. Always returns a plain array suitable for `.apply`.
   */
  concat(...args: unknown[]): unknown[] {
    const out: unknown[] = [...this.__iter__()];
    for (const arg of args) {
      if (Array.isArray(arg)) {
        out.push(...arg);
      } else if (
        arg !== null &&
        arg !== undefined &&
        typeof arg !== "string" &&
        typeof (arg as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function"
      ) {
        for (const item of arg as Iterable<unknown>) out.push(item);
      } else {
        out.push(arg);
      }
    }
    return out;
  }

  nonempty_submobjects(): Mobject[] {
    return this.submobjects.filter(
      (submob) => submob.submobjects.length !== 0 || submob.get_num_points() !== 0,
    );
  }

  reduce_across_dimension(
    reduce_func: (values: Iterable<number>) => number,
    dim: number,
  ): number | null {
    if (!(dim >= 0 && dim <= 2)) {
      throw new Error(`AssertionError: dim ${dim} is not in [0, 2]`);
    }
    if (this.submobjects.length === 0 && this.get_num_points() === 0) {
      // If we have no points and no submobjects, return None.
      return null;
    }

    // If we do not have points (but do have submobjects) use only the
    // points from those.
    let rv: number | null;
    if (this.get_num_points() === 0) {
      rv = null;
    } else {
      // Otherwise, be sure to include our own points.
      rv = reduce_func(
        pointRows(this.points, this.dim).map((point) => point[dim] ?? 0),
      );
    }
    // Recursively ask submobjects (if any) for the biggest/smallest
    // dimension they have and compare it to the return value.
    for (const mobj of this.submobjects) {
      const value = mobj.reduce_across_dimension(reduce_func, dim);
      if (rv === null) {
        rv = value;
      } else if (value !== null) {
        rv = reduce_func([value, rv]);
      }
    }
    return rv;
  }

  get_merged_array(array_attr: string): NDArray {
    // Fast path: collect every family member's C-contiguous array in the same
    // pre-order the old recursive `concatenate` produced and write one flat
    // buffer.  Concatenating once per submobject copied the growing result at
    // every level (O(n^2) for large families such as Text glyph groups) and
    // ran on every `get_critical_point`/boundary scan.
    const views: Array<{ data: ArrayLike<number>; rowDim: number; rowCount: number }> = [];
    let rowDim = -1;
    let rowCount = 0;
    let consistent = true;
    const visit = (mob: Mobject): void => {
      const value = (mob as unknown as Record<string, unknown>)[array_attr];
      if (value !== undefined && value !== null) {
        const flat = _cContiguousRows(value);
        if (flat !== null && (rowDim === -1 || flat.rowDim === rowDim)) {
          if (rowDim === -1) rowDim = flat.rowDim;
          rowCount += flat.rowCount;
          views.push(flat);
        } else {
          consistent = false;
        }
      }
      for (const child of mob.submobjects) visit(child);
    };
    visit(this);
    if (!consistent) {
      let result = (this as unknown as Record<string, NDArray>)[array_attr];
      for (const submob of this.submobjects) {
        result = concatenate([result, submob.get_merged_array(array_attr)], 0) as NDArray;
      }
      return result;
    }
    if (rowDim === -1) return array([]) as NDArray;
    const out = new Array<number>(rowCount * rowDim);
    let offset = 0;
    for (const view of views) {
      const length = view.rowCount * view.rowDim;
      for (let index = 0; index < length; index += 1) out[offset++] = numeric(view.data[index]);
    }
    return rowCount > 0 ? (array(out).reshape(rowCount, rowDim) as NDArray) : (array([]).reshape(0, rowDim) as NDArray);
  }

  get_all_points(): NDArray {
    return this.get_merged_array("points");
  }

  get_points_defining_boundary(): NDArray {
    return this.get_all_points();
  }

  getPointsDefiningBoundary(): NDArray {
    return this.get_points_defining_boundary();
  }

  nonemptySubmobjects(): Mobject[] {
    return this.nonempty_submobjects();
  }

  shuffle(recursive = false): this {
    if (recursive) for (const submob of this.submobjects) submob.shuffle(true);
    // Fisher-Yates, mirroring random.shuffle.
    for (let index = this.submobjects.length - 1; index > 0; index -= 1) {
      const other = Math.floor(Math.random() * (index + 1));
      const temp = this.submobjects[index] as Mobject;
      this.submobjects[index] = this.submobjects[other] as Mobject;
      this.submobjects[other] = temp;
    }
    return this;
  }

  shuffle_submobjects(...args: unknown[]): this {
    const [recursive = false] = args as [boolean?];
    return this.shuffle(recursive);
  }

  shuffleSubmobjects(...args: unknown[]): this {
    return this.shuffle_submobjects(...args);
  }

  invert(recursive = false): this {
    if (recursive) for (const submob of this.submobjects) submob.invert(true);
    this.submobjects.reverse();
    return this;
  }

  interpolate(mobject1: Mobject, mobject2: Mobject, alpha: number, path_func: PathFuncType | null = null): this {
    const func = (path_func ?? straight_path()) as PathFuncType;
    this.points = func(mobject1.points, mobject2.points, alpha);
    this.interpolate_color(mobject1, mobject2, alpha);
    return this;
  }

  __getitem__(value: number | MobjectSlice): Mobject | Group {
    if (typeof value === "object" && value !== null) {
      const GroupClass = this.get_group_class();
      if (this.has_no_points()) {
        return new GroupClass(...applySlice(this.submobjects, value));
      }
      // Mirrors `r = range(*value.indices(len(self)))`.
      const r = sliceRange(this.__len__(), value);
      const indices = rangeValues(r.start, r.stop, r.step);
      if (indices.length === 0) {
        // If slice is empty
        return new GroupClass();
      }
      if (!indices.includes(0)) {
        // If self is not included in the slice, we can gain a small speed boost
        // by indexing directly into the submobjects list.
        // (Python slices `self.submobjects[r.start - 1 : stop : r.step]`
        // with an adjusted stop; since slice.indices() never yields negative
        // values this is exactly `submobjects[i - 1]` for each `i` in `r`.)
        return new GroupClass(
          ...indices.map((i) => this.submobjects[i - 1] as Mobject),
        );
      }
      return new GroupClass(
        ...indices.map((i) => (i === 0 ? this : (this.submobjects[i - 1] as Mobject))),
      );
    }
    // Mirrors `index = range(len(self))[value]` (normalizes negatives,
    // raises IndexError out of range).
    let index = value;
    if (!Number.isInteger(index)) {
      const error = new Error("range object index out of range");
      error.name = "TypeError";
      throw error;
    }
    if (index < 0) index += this.__len__();
    if (index < 0 || index >= this.__len__()) {
      const error = new Error("range object index out of range");
      error.name = "IndexError";
      throw error;
    }
    if (this.has_no_points()) {
      return this.submobjects[index] as Mobject;
    }
    return index === 0 ? this : (this.submobjects[index - 1] as Mobject);
  }

  toString(): string { return this.name; }
  valueOf(): string { return this.toString(); }
}

// Explicit Python parameter names for `arrange`/`arrange_in_grid`, surviving
// bundling. py2js maps Python kwargs onto positional slots via
// `Function.toString` param names, but esbuild renames colliding locals when
// bundling (here `rows` became `rows2`), which mis-routed `rows=...` into
// `row_alignments` and crashed in `init_alignments`
// ("alignments is not iterable"). `arrange` additionally accepts `**kwargs`
// forwarded to `next_to` (e.g. `aligned_edge`); without an explicit
// `aligned_edge` slot the bridge appended it as a trailing options object,
// which landed in `buff` and poisoned every `shift` with NaN (see
// 55_patreon_thanks.py: `arrange(DOWN, aligned_edge=LEFT)`).
// `runtime_core` prefers this metadata over parsed names (see also the
// `__argnames__` propagation in `wrapPrototypeMethods`).
(Mobject.prototype.arrange as unknown as { __argnames__?: string[] }).__argnames__ = [
  "direction",
  "buff",
  "center",
  "aligned_edge",
  "kwargs",
];
(Mobject.prototype.arrange_in_grid as unknown as { __argnames__?: string[] }).__argnames__ = [
  "rows",
  "cols",
  "buff",
  "cell_alignment",
  "row_alignments",
  "col_alignments",
  "row_heights",
  "col_widths",
  "flow_order",
];
// `match_*`/`rescale_to_fit`/`*_to_fit_*` accept Python `**kwargs`
// (`stretch=`, `about_point=`, `about_edge=`); pin the names so the py2js
// kwargs bridge keeps mapping them after bundling (same failure as `arrange`
// above: without a `stretch` slot, `match_width(m, stretch=True)` appended
// `{stretch: true}` into `about_point` and `scale` threw non-finite, error 166).
(Mobject.prototype.match_dim_size as unknown as { __argnames__?: string[] }).__argnames__ = [
  "mobject",
  "dim",
  "stretch",
  "about_point",
  "about_edge",
  "kwargs",
];
(Mobject.prototype.match_width as unknown as { __argnames__?: string[] }).__argnames__ = [
  "mobject",
  "stretch",
  "about_point",
  "about_edge",
  "kwargs",
];
(Mobject.prototype.match_height as unknown as { __argnames__?: string[] }).__argnames__ = [
  "mobject",
  "stretch",
  "about_point",
  "about_edge",
  "kwargs",
];
(Mobject.prototype.match_depth as unknown as { __argnames__?: string[] }).__argnames__ = [
  "mobject",
  "stretch",
  "about_point",
  "about_edge",
  "kwargs",
];
(Mobject.prototype.rescale_to_fit as unknown as { __argnames__?: string[] }).__argnames__ = [
  "value",
  "dim",
  "stretch",
  "about_point",
  "about_edge",
  "kwargs",
];
for (const name of ["scale_to_fit_width", "scale_to_fit_height", "scale_to_fit_depth", "stretch_to_fit_width", "stretch_to_fit_height", "stretch_to_fit_depth"] as const) {
  ((Mobject.prototype[name] as unknown as { __argnames__?: string[] }).__argnames__ = [
    "value",
    "about_point",
    "about_edge",
    "kwargs",
  ]);
}
// `Mobject.scale(scale_factor, *, about_point=None, about_edge=None)` already
// spells its parameters like Python, but pin them explicitly so esbuild
// renames cannot break the py2js kwargs bridge (same failure as `arrange`
// above; `animate.scale(2, about_point=ORIGIN)` would otherwise leave the
// kwargs bag in the point slot and throw non-finite, error 251).
(Mobject.prototype.scale as unknown as { __argnames__?: string[] }).__argnames__ = [
  "scale_factor",
  "about_point",
  "about_edge",
];
// `align_on_border`/`to_corner`/`to_edge(direction, buff=)` take a `buff`
// keyword (e.g. `animate.to_corner(DR, buff=0.4)`); pin the Python spellings
// so the py2js kwargs bridge maps the trailing bag instead of leaving the
// whole `{buff: ...}` object in the numeric `buff` slot (which poisons the
// shift with NaN, error 253). `align_on_border` additionally unwraps the bag
// defensively for `animate` paths that bypass the bridge (same as `scale`).
(Mobject.prototype.align_on_border as unknown as { __argnames__?: string[] }).__argnames__ = [
  "direction",
  "buff",
];
(Mobject.prototype.to_corner as unknown as { __argnames__?: string[] }).__argnames__ = [
  "corner",
  "buff",
];
(Mobject.prototype.to_edge as unknown as { __argnames__?: string[] }).__argnames__ = [
  "edge",
  "buff",
];
// `next_to(mobject_or_point, direction, buff=, aligned_edge=, ...)` takes
// `buff`/`aligned_edge` keywords (e.g. `animate.next_to(m, UP, buff=0.4)`);
// pin the Python spellings so the py2js kwargs bridge maps the trailing bag
// instead of leaving the whole `{buff: ...}` object in the numeric `buff`
// slot (which poisons the shift with NaN, error 262). `next_to` additionally
// unwraps the bag defensively for `animate` paths that bypass the bridge
// (same as `scale`/`align_on_border`).
(Mobject.prototype.next_to as unknown as { __argnames__?: string[] }).__argnames__ = [
  "mobject_or_point",
  "direction",
  "buff",
  "aligned_edge",
  "submobject_to_align",
  "index_of_submobject_to_align",
  "coor_mask",
];

export class Group extends Mobject {
  constructor(...args: Array<Mobject | Iterable<Mobject> | MobjectOptions>) {
    // Mirrors Python `Group.__init__(self, *mobjects, **kwargs)`: a trailing
    // plain object is the Mobject kwargs, everything else are members.
    // (Mobject instances carry a class prototype, so they never match.)
    const last = args[args.length - 1];
    let options: MobjectOptions = {};
    let members = args as Array<Mobject | Iterable<Mobject>>;
    if (
      last !== undefined &&
      !(last instanceof Mobject) &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      typeof (last as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== "function"
    ) {
      options = last as MobjectOptions;
      members = args.slice(0, -1) as Array<Mobject | Iterable<Mobject>>;
    }
    super(options);
    this.add(...members);
  }
}

export class _MethodAnimation extends Animation {
  methods: MethodWithArgs[];
  target_mobject: Mobject;

  constructor(mobject: Mobject, methods: MethodWithArgs[]) {
    super(mobject);
    this.methods = methods;
    this.target_mobject = mobject.target ?? mobject.copy();
  }

  finish(): void {
    for (const item of this.methods) item.method.apply(this.mobject, Object.keys(item.kwargs).length > 0 ? [...item.args, item.kwargs] : item.args);
    super.finish();
  }
}

/** Constructor shape shared by the mobject-local helper and the faithful
 *  manim.animation.transform port (both are `(mobject, methods)` classes
 *  named `_MethodAnimation`). */
export type MethodAnimationClass = new (mobject: Mobject, methods: MethodWithArgs[]) => Animation;

let builderAnimationClass: MethodAnimationClass | null = null;

/** @internal Late binding for `_AnimationBuilder.build()` (see its NOTE).
 *  Called once by animation/transform.js on module evaluation; not part of
 *  the public manim API. */
export function __registerBuilderAnimationClass(cls: MethodAnimationClass): void {
  builderAnimationClass = cls;
}

export class _AnimationBuilder {
  mobject: Mobject;
  overridden_animation: any = null;
  is_chaining = false;
  methods: MethodWithArgs[] = [];
  cannot_pass_args = false;
  anim_args: Record<string, unknown> = {};

  /**
   * Pop a trailing py2js kwargs bag (`ρσ_desugar_kwargs`, a null-prototype
   * object tagged with the `kwargs-object` symbol) off `args`.
   *
   * Mirrors Python, where `animate.foo(2, about_point=X)` stores
   * `args=(2,)` and `kwargs={"about_point": X}` separately: rapydscript
   * compiles the call to `update(2, kwargsBag)` (the `update` closure has no
   * `__argnames__`, so `ρσ_interpolate_kwargs` cannot map it), and without
   * this split the bag would ride as a positional argument into
   * `Mobject.scale`'s `about_point` slot and poison geometry with NaN.
   * Returns the bag (tag preserved so the wrapped-method bridge still
   * recognises it) or null. Only tagged null-prototype bags are split —
   * plain objects stay positional.
   */
  private static popTrailingKwargsBag(args: any[]): Record<string, any> | null {
    if (args.length === 0) return null;
    const last = args[args.length - 1];
    if (last === null || typeof last !== "object") return null;
    if (Object.getPrototypeOf(last) !== null) return null;
    let tagged = false;
    try {
      for (const symbol of Object.getOwnPropertySymbols(last)) {
        if (String(symbol).includes("kwargs-object")) {
          tagged = true;
          break;
        }
      }
    } catch {
      return null;
    }
    if (!tagged) {
      try {
        for (const key of Object.keys(last)) {
          if (key.startsWith("kwargs-object-Symbol-")) {
            tagged = true;
            break;
          }
        }
      } catch {
        return null;
      }
    }
    if (!tagged) return null;
    args.pop();
    return last as Record<string, any>;
  }

  constructor(mobject: Mobject) {
    this.mobject = mobject;
    this.mobject.generate_target();
    let callableProxy: _AnimationBuilder;
    const builder = function (...args: any[]): _AnimationBuilder {
      builder.call((args[0] as Record<string, unknown> | undefined) ?? {});
      return callableProxy;
    } as unknown as _AnimationBuilder;
    Object.setPrototypeOf(builder, _AnimationBuilder.prototype);
    Object.assign(builder, this);
    callableProxy = new Proxy(builder, {
      get: (target, property, receiver) => {
        if (property in target || typeof property === "symbol") return Reflect.get(target, property, receiver);
        // NOTE (error 373): `bunch.animate(run_time=1.5)` compiles to
        // `ρσ_interpolate_kwargs.call(bunch, bunch.animate, [bag])`, which
        // reads `f.__argnames__` directly. Routing that probe through
        // `__getattr__` threw `VGroup object has no attribute
        // '__argnames__'`. Like the Mobject Proxy above, unknown dunders
        // resolve to `undefined` so the kwargs bridge takes its no-mapping
        // path (`f.apply(this, args)`). `then` does the same to keep the
        // builder from looking like a Promise. `apply`/`bind` are restored
        // from `Function.prototype` (lost via `setPrototypeOf` below) so
        // that `f.apply` path can actually invoke the builder.
        if (typeof property === "string") {
          if (property === "apply" || property === "bind") {
            const fn = (Function.prototype as unknown as Record<string, unknown>)[property];
            return typeof fn === "function" ? (fn as Function).bind(target) : undefined;
          }
          if (property.startsWith("__") || property === "then") return undefined;
        }
        // NOTE: mirrors Python, where any unknown attribute access on the
        // builder goes through `__getattr__` (which resolves the method on
        // the target mobject, checks overrides, and marks chaining state).
        const update = (target as _AnimationBuilder).__getattr__(String(property));
        return (...args: any[]) => update(...args);
      },
    });
    (builder as unknown as { _callable_proxy: _AnimationBuilder })._callable_proxy = callableProxy;
    return callableProxy;
  }

  __getattr__(method_name: string): (...args: any[]) => this {
    // NOTE: mirrors Python `method = getattr(self.mobject.target,
    // method_name)` — resolved eagerly at access time (missing attributes
    // raise AttributeError here, not at call time).
    const target = this.mobject.target ?? this.mobject;
    const method = (target as unknown as Record<string, unknown>)[method_name] as FunctionLike | undefined;
    if (typeof method !== "function") {
      const error = new Error(
        `${target.constructor.name} object has no attribute '${method_name}'`,
      );
      error.name = "AttributeError";
      throw error;
    }
    const has_overridden_animation = get_animation_override(method) !== undefined;

    if ((this.is_chaining && has_overridden_animation) || this.overridden_animation) {
      const error = new Error(
        "Method chaining is currently not supported for overridden animations",
      );
      error.name = "NotImplementedError";
      throw error;
    }

    const self = this;
    const bound_method: FunctionLike = method;
    function update_target(...method_args: any[]): _AnimationBuilder {
      const bag = _AnimationBuilder.popTrailingKwargsBag(method_args);
      self.apply_method_update(bound_method, has_overridden_animation, method_args, bag ?? {});
      return ((self as unknown as { _callable_proxy?: _AnimationBuilder })._callable_proxy ?? self) as _AnimationBuilder;
    }

    this.is_chaining = true;
    this.cannot_pass_args = true;

    return update_target as unknown as (...args: any[]) => this;
  }

  call(args: Record<string, unknown> = {}): this {
    if (this.cannot_pass_args) {
      const error = new Error("Animation arguments must be passed before accessing methods and can only be passed once");
      error.name = "ValueError";
      throw error;
    }
    this.anim_args = args;
    this.cannot_pass_args = true;
    return this;
  }

  update_target(method_name: string, method_args: any[] = [], method_kwargs: Record<string, any> = {}): this {
    const target = this.mobject.target ?? this.mobject;
    const method = (target as unknown as Record<string, unknown>)[method_name];
    if (typeof method !== "function") throw new Error(`${this.mobject.constructor.name} has no method '${method_name}'`);
    const override = get_animation_override(method as FunctionLike);
    const argsCopy = [...method_args];
    const trailing = _AnimationBuilder.popTrailingKwargsBag(argsCopy);
    let mergedKwargs = method_kwargs ?? {};
    if (trailing !== null) {
      if (Object.keys(mergedKwargs).length === 0) {
        mergedKwargs = trailing;
      } else {
        for (const [key, value] of Object.entries(mergedKwargs)) (trailing as Record<string, unknown>)[key] = value;
        mergedKwargs = trailing;
      }
    }
    this.apply_method_update(method as FunctionLike, override !== undefined, argsCopy, mergedKwargs);
    return ((this as unknown as { _callable_proxy?: _AnimationBuilder })._callable_proxy ?? this) as this;
  }

  /** Shared body of the ``update_target`` closures (mirrors Python's inner ``update_target``). */
  apply_method_update(
    method: FunctionLike,
    has_overridden_animation: boolean,
    method_args: any[] = [],
    method_kwargs: Record<string, any> = {},
  ): this {
    const target = this.mobject.target ?? this.mobject;
    // Defensive: a trailing kwargs bag may still ride positionally (e.g. a
    // direct `apply_method_update` call that bypassed the `update_target`
    // split above). Merge it into `method_kwargs`, preserving its tag so the
    // wrapped-method kwargs bridge (`wrapPrototypeMethods`) still maps it.
    const argsCopy = [...method_args];
    const trailing = _AnimationBuilder.popTrailingKwargsBag(argsCopy);
    let mergedKwargs = method_kwargs ?? {};
    if (trailing !== null) {
      if (Object.keys(mergedKwargs).length === 0) {
        mergedKwargs = trailing;
      } else {
        for (const [key, value] of Object.entries(mergedKwargs)) (trailing as Record<string, unknown>)[key] = value;
        mergedKwargs = trailing;
      }
    }
    if (has_overridden_animation) {
      const override = get_animation_override(method);
      this.overridden_animation = (override as FunctionLike)(
        this.mobject,
        ...argsCopy,
        { anim_args: this.anim_args, ...mergedKwargs },
      );
    } else {
      this.methods.push(new MethodWithArgs(method, argsCopy, mergedKwargs));
      method.apply(target, Object.keys(mergedKwargs).length > 0 ? [...argsCopy, mergedKwargs] : argsCopy);
    }
    this.is_chaining = true;
    this.cannot_pass_args = true;
    return ((this as unknown as { _callable_proxy?: _AnimationBuilder })._callable_proxy ?? this) as this;
  }

  build(): any {
    // NOTE: mirrors Python's function-local
    // `from ..animation.transform import _MethodAnimation` (kept local there
    // to avoid a hard import cycle). A static TS import of
    // animation/transform.js here would be unresolvable for the same reason:
    // transform.js pulls in types/vectorized_mobject.js, whose
    // `class VMobject extends Mobject` must evaluate AFTER Mobject is
    // defined. So transform.js registers its faithful class on import
    // (see the bottom of animation/transform.ts) and build() late-binds to
    // it — exactly like Python's deferred import. The mobject-local
    // _MethodAnimation below is only a fallback for graphs that never load
    // animation/transform.js.
    const animation = this.overridden_animation ??
      new (builderAnimationClass ?? _MethodAnimation)(this.mobject, this.methods);
    for (const [key, value] of Object.entries(this.anim_args)) (animation as Record<string, unknown>)[key] = value;
    return animation;
  }
}

export function createAnimationBuilder(mobject: Mobject): _AnimationBuilder {
  return new _AnimationBuilder(mobject);
}

// The animate/updater builders shuttle rapydscript kwargs bags around as
// *values*: `update_target` splits the method's kwargs bag off and hands it
// to `apply_method_update` as the `method_kwargs` argument, which re-applies
// it positionally so the wrapped-method kwargs bridge (`wrapPrototypeMethods`
// in py2js) can map it onto the target method's slots. If the builders'
// own prototype methods were wrapped, the bridge would mistake that trailing
// bag for kwargs-to-map (e.g. `opacity=` has no slot on
// `apply_method_update`, so it lands in an extras merge that drops the tag),
// and the detagged bag would ride positionally into `color` and throw in
// `ManimColor` (`circles.animate.set_fill(opacity=0.25)`).
// These classes are internal machinery (never Python kwarg call targets),
// so they opt out of prototype wrapping entirely.
(
  _AnimationBuilder as unknown as { __wanim_skip_kwargs_bridge__?: boolean }
).__wanim_skip_kwargs_bridge__ = true;

export class _UpdaterBuilder {
  _mobject: Mobject;

  constructor(mobject: Mobject) {
    this._mobject = mobject;
    const proxy = new Proxy(this, {
      get: (target, property, receiver) => {
        if (property in target || typeof property === "symbol") return Reflect.get(target, property, receiver);
        // NOTE (error 373): same dunder-probe hazard as the Mobject and
        // animation-builder Proxies (`ρσ_interpolate_kwargs` reads
        // `f.__argnames__`); resolve unknown dunders to `undefined` instead
        // of misrouting them into `add_updater`.
        if (typeof property === "string" && (property.startsWith("__") || property === "then")) return undefined;
        return (...args: any[]) => {
          target.add_updater(String(property), ...args);
          return (target as unknown as { _proxy?: _UpdaterBuilder })._proxy ?? target;
        };
      },
    });
    (this as unknown as { _proxy: _UpdaterBuilder })._proxy = proxy;
    return proxy;
  }

  __getattr__(name: string): (...args: any[]) => this {
    return (...args: any[]) => this.add_updater(name, ...args);
  }

  add_updater(method_name: string, ...method_args: any[]): this {
    const possible_kwargs = method_args.length > 0 && method_args[method_args.length - 1] !== null && typeof method_args[method_args.length - 1] === "object" && !Array.isArray(method_args[method_args.length - 1]) ? method_args[method_args.length - 1] as Record<string, unknown> : null;
    const positional_args = possible_kwargs === null ? method_args : method_args.slice(0, -1);
    this._mobject.add_updater((mob: Mobject) => {
      const method = (mob as unknown as Record<string, unknown>)[method_name];
      if (typeof method === "function") (method as FunctionLike).apply(mob, possible_kwargs === null ? positional_args : [...positional_args, possible_kwargs]);
    }, undefined, true);
    return ((this as unknown as { _proxy?: _UpdaterBuilder })._proxy ?? this) as this;
  }
}

export function createUpdaterBuilder(mobject: Mobject): _UpdaterBuilder {
  return new _UpdaterBuilder(mobject);
}

(
  _UpdaterBuilder as unknown as { __wanim_skip_kwargs_bridge__?: boolean }
).__wanim_skip_kwargs_bridge__ = true;

export function override_animate(method: FunctionLike): (animationMethod: FunctionLike) => FunctionLike {
  function decorator(animationMethod: FunctionLike): FunctionLike {
    overrideFunctions.set(method, animationMethod);
    (method as FunctionLike & { _override_animate?: FunctionLike })._override_animate = animationMethod;
    return animationMethod;
  }
  return decorator;
}
