import { allclose, array, NDArray } from "numpy-ts";
import { bump_mobject_geometry_epoch } from "../../utils/render_epoch.js";
import { Mobject, _rawMobject, assertFinitePoints, hasNonFinite, type MobjectOptions, type MobjectSlice, type MobjectUpdater } from "../mobject.js";
import { ManimColor, WHITE, BLACK, interpolate_color, type ParsableManimColor } from "../../utils/color/core.js";
import { pointRows, pointsFrom, pointFrom, pointAt, numeric, addPoints, subtractPoints, scalePoint, norm, toPythonRepr, pythonTypeName, isNDArray, nestedRows, linspace, _cContiguousRows, _readFlatStore, _flatStoreSlots } from "../../utils/arrays.js";
import { bezier, bezier_remap, get_smooth_cubic_bezier_handle_points_rows, integer_interpolate, interpolate, partial_bezier_points, proportions_along_bezier_curve_for_point } from "../../utils/bezier.js";
import { shoelace_direction, rotate_vector, rotation_matrix } from "../../utils/space_ops.js";
import { resize_array, stretch_array_to_length, tuplify, make_even } from "../../utils/iterables.js";
import { CapStyleType, LEFT, ORIGIN, OUT, PI, RendererType, RIGHT, UL, UP } from "../../constants.js";
import { config } from "../../_config/utils.js";

/** Factory used by {@link VDict} with `show_keys` to build the key labels.
 *  Registered by the text module (tex_mobject.ts) to avoid a circular
 *  import; `null` until that module is loaded, mirroring the same pattern as
 *  `setMathTexLabelFactory` in geometry/arc.ts. */
export type TexFactory = (text: string, color?: ParsableManimColor | null) => VMobject;
let texFactory: TexFactory | null = null;
export function setTexFactory(factory: TexFactory | null): void {
  texFactory = factory;
}

export interface VMobjectOptions extends MobjectOptions {
  fill_color?: ParsableManimColor;
  stroke_color?: ParsableManimColor;
  fill_opacity?: number;
  stroke_opacity?: number;
  stroke_width?: number;
  n_points_per_cubic_curve?: number;
  close_new_points?: boolean;
  background_stroke_color?: ParsableManimColor | null;
  background_stroke_opacity?: number;
  background_stroke_width?: number;
  sheen_factor?: number;
  joint_type?: string | null;
  sheen_direction?: readonly number[];
  pre_function_handle_to_anchor_scale_factor?: number;
  make_smooth_after_applying_functions?: boolean;
  background_image?: string | null;
  shade_in_3d?: boolean;
  tolerance_for_point_equality?: number;
  cap_style?: string;
}

function rows(value: unknown, dim: number): number[][] {
  // Fast path: read a C-contiguous 2-D NDArray straight from its typed
  // backing store. `pointRows` would first materialise the whole array via
  // `toArray()` and then copy each row again, which dominates smoothing of
  // high-resolution graphs.
  if (isNDArray(value)) {
    const array = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
    const shape = array.shape;
    const flat = array.data;
    if (
      flat !== undefined &&
      shape.length === 2 &&
      shape[1] === dim &&
      flat.length === shape[0] * dim &&
      (array.flags === undefined || array.flags.C_CONTIGUOUS !== false)
    ) {
      const count = shape[0] ?? 0;
      const result = new Array<number[]>(count);
      for (let i = 0; i < count; i += 1) {
        const row = new Array<number>(dim);
        const base = i * dim;
        for (let c = 0; c < dim; c += 1) row[c] = flat[base + c] ?? 0;
        result[i] = row;
      }
      return result;
    }
  }
  return pointRows(value, dim);
}

function asPoints(rowsValue: number[][], dim: number): NDArray {
  // Build the backing typed array directly: `array(nested).reshape(...)` is
  // several times slower for the large point buffers used by graphs/curves.
  if (rowsValue.length === 0) return array([]).reshape(0, dim) as NDArray;
  const flat = new Float64Array(rowsValue.length * dim);
  for (let i = 0; i < rowsValue.length; i += 1) {
    const row = rowsValue[i] ?? [];
    const base = i * dim;
    for (let c = 0; c < dim; c += 1) flat[base + c] = row[c] ?? 0;
  }
  return array(flat).reshape(rowsValue.length, dim) as NDArray;
}

function curveValue(points: number[][], t: number, dim: number): number[] {
  const [p0, p1, p2, p3] = points;
  const mt = 1 - t;
  return Array.from({ length: dim }, (_, index) => mt ** 3 * (p0[index] ?? 0) + 3 * mt ** 2 * t * (p1[index] ?? 0) + 3 * mt * t ** 2 * (p2[index] ?? 0) + t ** 3 * (p3[index] ?? 0));
}

function curveLength(points: number[][], samplePoints = 10): number {
  let total = 0;
  let previous = curveValue(points, 0, points[0]?.length ?? 3);
  for (let index = 1; index < samplePoints; index += 1) {
    const current = curveValue(points, index / (samplePoints - 1), points[0]?.length ?? 3);
    total += Math.hypot(...current.map((value, component) => value - (previous[component] ?? 0)));
    previous = current;
  }
  return total;
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return value !== null && value !== undefined && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function";
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** `[r, g, b, alpha]` for a color without going through the numpy-ts
 * `to_rgba_with_alpha` round-trip when the internal `(4,)` array is directly
 * readable. `alpha` overrides the stored alpha, matching
 * `ManimColor.to_rgba_with_alpha`. */
function colorRgbaValues(color: ManimColor, alpha: number): number[] {
  const internal = color.internal_value as NDArray | undefined;
  if (internal !== undefined && internal !== null) {
    const shape = internal.shape;
    const data = (internal as NDArray & { data?: ArrayLike<number> }).data;
    if (shape.length === 1 && shape[0] === 4 && data !== undefined && data.length === 4) {
      return [data[0] ?? 0, data[1] ?? 0, data[2] ?? 0, alpha];
    }
  }
  return color.to_rgba_with_alpha(alpha).toArray() as number[];
}

/** Fast check for a `(1, 4)` C-contiguous rgba array, returning its backing
 * numbers. Used to avoid the `pointRows` round-trip in the uniform style
 * paths (the overwhelmingly common case: one color for the whole mobject). */
function singleRgbaData(value: NDArray | undefined): ArrayLike<number> | null {
  if (value === undefined || value === null) return null;
  const shape = value.shape;
  if (shape.length !== 2 || shape[0] !== 1 || shape[1] !== 4) return null;
  const array = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
  const data = array.data;
  if (data === undefined || data.length !== 4) return null;
  if (array.flags !== undefined && array.flags.C_CONTIGUOUS === false) return null;
  return data;
}

/** Materialise a `(N, 4)` C-contiguous rgba array into plain rows directly
 * from its typed backing store, avoiding `pointRows`' `toArray()` round-trip.
 * Falls back to `pointRows` for any other shape/layout. */
function rgbaRows(value: NDArray | undefined): number[][] {
  if (value === undefined || value === null) return pointRows(value, 4);
  const shape = value.shape;
  if (shape.length === 2 && shape[1] === 4) {
    const array = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
    const data = array.data;
    const count = shape[0] ?? 0;
    if (data !== undefined && data.length === count * 4 && (array.flags === undefined || array.flags.C_CONTIGUOUS !== false)) {
      const result = new Array<number[]>(count);
      for (let i = 0; i < count; i += 1) {
        const base = i * 4;
        result[i] = [data[base] ?? 0, data[base + 1] ?? 0, data[base + 2] ?? 0, data[base + 3] ?? 0];
      }
      return result;
    }
  }
  return pointRows(value, 4);
}

/** Unwrap a style value that may be a list/array of colors (as returned by
 * get_style()) down to a single color, mirroring the single-color behavior
 * of set_fill/set_stroke in this port. */
export function singleColor(value: ParsableManimColor | ManimColor[] | NDArray | null | undefined): ParsableManimColor | null | undefined {
  if (value === null || value === undefined) return value;
  if (Array.isArray(value)) {
    for (const entry of value) {
      if (entry !== null && entry !== undefined) return entry as ParsableManimColor;
    }
    return null;
  }
  if (isNDArray(value)) {
    const flat = (value.toArray() as unknown[]).flat(Infinity) as unknown[];
    return (flat.find((entry) => entry !== null && entry !== undefined) as ParsableManimColor | undefined) ?? null;
  }
  return value;
}

/** Unwrap a style opacity that may be a list/array of opacities down to a
 * single number. */
export function singleOpacity(value: number | number[] | NDArray | null | undefined): number | null | undefined {
  if (value === null || value === undefined) return value;
  if (typeof value === "number") return value;
  if (Array.isArray(value)) {
    const found = value.find((entry) => entry !== null && entry !== undefined);
    return found === undefined ? null : Number(found);
  }
  const flat = (value.toArray() as unknown[]).flat(Infinity) as unknown[];
  const found = flat.find((entry) => entry !== null && entry !== undefined);
  return found === undefined ? null : Number(found);
}

/** Parse a stroke/fill color argument (which may be a per-point gradient
 * list, matching Python's `ManimColor.parse`/`ManimColor`) down to a single
 * color, used for the `background_stroke_color` field. */
function parseSingleColor(value: ParsableManimColor | Iterable<ManimColor>): ManimColor {
  const single = singleColor(value as ParsableManimColor | ManimColor[] | NDArray | null | undefined);
  if (single instanceof ManimColor) return single;
  return new ManimColor((single ?? BLACK) as ParsableManimColor);
}

function asRows(value: unknown): number[][] {
  if (isNDArray(value)) {
    const arr = (value as NDArray).toArray() as unknown;
    if (Array.isArray(arr) && Array.isArray(arr[0])) return arr as number[][];
    return [arr as number[]];
  }
  if (Array.isArray(value)) {
    if (Array.isArray(value[0])) return value as number[][];
    return [value as number[]];
  }
  return [[Number(value)]];
}

/** Rank 0 = scalar, 1 = flat vector, 2 = matrix / row list. */
function rankOf(value: unknown): number {
  if (isNDArray(value)) return ((value as NDArray).shape.length ?? 1) >= 2 ? 2 : 1;
  if (Array.isArray(value)) return Array.isArray(value[0]) ? 2 : 1;
  return 0;
}

function flatArray(value: unknown): number[] {
  if (isNDArray(value)) return ((value as NDArray).toArray() as unknown[]).map(Number);
  if (Array.isArray(value)) return (value as unknown[]).map(Number);
  return [Number(value)];
}

function lerp1d(a: number[], b: number[], alpha: number): number[] {
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const av = a.length === 1 ? a[0] : (a[i] ?? 0);
    const bv = b.length === 1 ? b[0] : (b[i] ?? 0);
    out.push(Number(av) * (1 - alpha) + Number(bv) * alpha);
  }
  return out;
}

/**
 * Attributes whose lerp is covered by the static-colour fast path
 * (`sheen_direction` is handled separately — it genuinely changes every
 * frame of a `Rotate`, so it is written in place instead).
 */
const STATIC_COLOR_ATTRS = [
  "fill_rgbas",
  "stroke_rgbas",
  "background_stroke_rgbas",
  "stroke_width",
  "background_stroke_width",
  "sheen_factor",
] as const;

interface StaticColorMemo {
  refs: unknown[];
  staticOk: boolean;
}

/**
 * Per-face static-colour memo for `interpolate_color`.
 *
 * Floating-point lerp is not the identity (`lerp(a, a, t)` can round to
 * `a ± 1ulp`), so per-element adoption of the lerped values almost never
 * triggers for static colours.  Instead, the first frame verifies that the
 * six value attributes are pairwise identical across starting/target/dest
 * (no writes — the destination already holds the values), memoizes the
 * outcome keyed by the eighteen array/scalar references, and later frames
 * only re-check the references: starting/target/destination stores are
 * stable across an animation, so a memo hit means the colours are known
 * static with zero element comparisons and zero allocations.  Any reference
 * change (a new animation's copies, an `alpha === 1` clone, a real morph
 * allocating fresh arrays) re-verifies, and morphing colours always take
 * the allocating slow path, whose fresh array identity is what invalidates
 * the rgba-identity colour caches.
 */
const staticColorMemo = new WeakMap<object, StaticColorMemo>();

function staticMemoRefs(
  r1: Record<string, unknown>,
  r2: Record<string, unknown>,
  rs: Record<string, unknown>,
): unknown[] {
  const refs: unknown[] = [];
  for (const attr of STATIC_COLOR_ATTRS) {
    refs.push(r1[attr], r2[attr], rs[attr]);
  }
  return refs;
}

/** Full static-colour verification (no writes); see `staticColorMemo`. */
function verifyStaticColors(
  r1: Record<string, unknown>,
  r2: Record<string, unknown>,
  rs: Record<string, unknown>,
): boolean {
  for (const attr of STATIC_COLOR_ATTRS) {
    const first = r1[attr];
    const second = r2[attr];
    const dest = rs[attr];
    if (typeof first === "number" || typeof second === "number" || typeof dest === "number") {
      // Mirrors the slow loop's numeric branch selection: any numeric side
      // forces scalar lerp, which is static only for identical numbers.
      if (typeof first !== "number" || typeof second !== "number" || typeof dest !== "number") {
        return false;
      }
      if (!Object.is(first, second) || !Object.is(second, dest)) return false;
      continue;
    }
    const slot = _flatStoreSlots;
    if (
      !_readFlatStore(first, slot[0]!) ||
      !_readFlatStore(second, slot[1]!) ||
      !_readFlatStore(dest, slot[2]!) ||
      slot[0]!.rows !== slot[1]!.rows ||
      slot[0]!.rows !== slot[2]!.rows ||
      slot[0]!.cols !== slot[1]!.cols ||
      slot[0]!.cols !== slot[2]!.cols
    ) {
      return false;
    }
    const firstData = slot[0]!.data!;
    const secondData = slot[1]!.data!;
    const destData = slot[2]!.data!;
    const length = slot[0]!.rows * slot[0]!.cols;
    for (let index = 0; index < length; index += 1) {
      // `Object.is` keeps `-0`/`0` and NaN exactly like the allocating path.
      if (!Object.is(firstData[index], secondData[index]) || !Object.is(secondData[index], destData[index])) {
        return false;
      }
    }
  }
  return true;
}

/** Flat numeric elements of a `sheen_direction`-style attribute, or null. */
function flatSheenValues(value: unknown): ArrayLike<number> | null {
  if (value instanceof Float64Array) return value;
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      if (typeof value[index] !== "number") return null;
    }
    return value as number[];
  }
  return null;
}

/**
 * In-place `sheen_direction` lerp in `lerp1d` flop order (bitwise identical
 * to the allocating path for flat same-length numerics, which covers every
 * real flow — `set_sheen_direction` and `rotate_sheen_direction` both store
 * plain `number[]`).  The destination keeps its `number[]` type, which
 * downstream code handles faster than the NDArray the allocating path
 * produces.
 */
function writeSheenLerp(destValue: unknown, firstValue: unknown, secondValue: unknown, inv: number, alpha: number): boolean {
  const dest = flatSheenValues(destValue);
  const first = flatSheenValues(firstValue);
  const second = flatSheenValues(secondValue);
  if (dest === null || first === null || second === null) return false;
  if (dest.length !== first.length || dest.length !== second.length) return false;
  // The destination must be owned by this mobject: `copy()` deep-copies plain
  // arrays, but a user-aliased array would corrupt the sharer when written in
  // place, so shared destinations take the allocating slow path instead.
  if (dest === first || dest === second) return false;
  for (let index = 0; index < dest.length; index += 1) {
    (dest as number[] | Float64Array)[index] = Number(first[index]) * inv + Number(second[index]) * alpha;
  }
  return true;
}

/** Element-wise lerp with NumPy-style broadcasting: flat vectors stay 1-D,
 * while a single rgba row broadcasts against a full point-aligned array
 * (Python: `(N,4)` with `(1,4)`). */
function interpolateArray(first: unknown, second: unknown, alpha: number): NDArray {
  // Flat fast path: `Transform` lerps every face's rgba arrays each frame, and
  // the row-materialising path below allocated a nested JS array per face.
  const flatFirst = _cContiguousRows(first);
  const flatSecond = _cContiguousRows(second);
  if (
    flatFirst !== null &&
    flatSecond !== null &&
    flatFirst.rowDim === flatSecond.rowDim &&
    flatFirst.rowCount === flatSecond.rowCount
  ) {
    const length = flatFirst.rowCount * flatFirst.rowDim;
    // Equal sources (the common `Rotate`/`Transform` case: the starting and
    // target copies share colours) reuse the existing array.  Allocating a
    // fresh, identical array here made the WebGL backend rebuild every face
    // each frame just because the rgba reference changed.
    let equal = first === second;
    if (!equal) {
      equal = true;
      for (let index = 0; index < length; index += 1) {
        if (Number(flatFirst.data[index] ?? 0) !== Number(flatSecond.data[index] ?? 0)) {
          equal = false;
          break;
        }
      }
    }
    if (equal) return first as NDArray;
    const inv = 1 - alpha;
    const out = new Float64Array(length);
    for (let index = 0; index < length; index += 1) {
      out[index] = Number(flatFirst.data[index] ?? 0) * inv + Number(flatSecond.data[index] ?? 0) * alpha;
    }
    return array(out).reshape(flatFirst.rowCount, flatFirst.rowDim) as NDArray;
  }
  const firstRank = rankOf(first);
  const secondRank = rankOf(second);
  if (firstRank === 0 && secondRank === 0) {
    return array(Number(first) * (1 - alpha) + Number(second) * alpha) as NDArray;
  }
  const left = asRows(first);
  const right = asRows(second);
  const rowCount = Math.max(left.length, right.length);
  const result: number[][] = [];
  for (let i = 0; i < rowCount; i += 1) {
    const a = left.length === 1 ? left[0] : (left[i] ?? left[0]);
    const b = right.length === 1 ? right[0] : (right[i] ?? right[0]);
    result.push(lerp1d(a as number[], b as number[], alpha));
  }
  // Preserve the original rank: flat inputs interpolate to a flat vector.
  if (firstRank <= 1 && secondRank <= 1) return array(result[0] ?? []) as NDArray;
  return array(result) as NDArray;
}

export class VMobject extends Mobject {
  // NOTE: `fill_color`, `stroke_color` and `color` are live properties in
  // Python (setters rewrite the rgba arrays). They are exposed as prototype
  // accessors here; the base `Mobject.color` is itself an accessor backed by
  // `_color`, so the override is not shadowed by an own data property.
  get fill_color(): ManimColor { return this.get_fill_color(); }
  set fill_color(value: ParsableManimColor | Iterable<ManimColor> | null) { this.set_fill(value); }
  get stroke_color(): ManimColor | null { return this.get_stroke_color(); }
  set stroke_color(value: ParsableManimColor | Iterable<ManimColor> | null) { this.set_stroke(value); }
  override get color(): ManimColor { return this.get_color(); }
  override set color(value: ManimColor) { this.set_color(value); }
  fill_opacity: number;
  stroke_opacity: number;
  stroke_width: number;
  background_stroke_color: ManimColor | null = new ManimColor(BLACK);
  background_stroke_opacity = 1;
  background_stroke_width = 0;
  sheen_factor = 0;
  sheen_direction: number[] = [...UL];
  joint_type = "AUTO";
  cap_style = "AUTO";
  close_new_points = false;
  pre_function_handle_to_anchor_scale_factor = 0.01;
  make_smooth_after_applying_functions = false;
  background_image: string | null = null;
  shade_in_3d = false;
  tolerance_for_point_equality = 1e-6;
  n_points_per_cubic_curve: number;
  _bezier_t_values: number[] = [];
  // Accessors so colour changes bump the geometry epoch (see mobject.points).
  private _fill_rgbas?: NDArray;
  get fill_rgbas(): NDArray { return this._fill_rgbas as NDArray; }
  set fill_rgbas(value: NDArray) { this._fill_rgbas = value; bump_mobject_geometry_epoch(); }
  private _stroke_rgbas?: NDArray;
  get stroke_rgbas(): NDArray { return this._stroke_rgbas as NDArray; }
  set stroke_rgbas(value: NDArray) { this._stroke_rgbas = value; bump_mobject_geometry_epoch(); }
  private _background_stroke_rgbas?: NDArray;
  get background_stroke_rgbas(): NDArray { return this._background_stroke_rgbas as NDArray; }
  set background_stroke_rgbas(value: NDArray) { this._background_stroke_rgbas = value; bump_mobject_geometry_epoch(); }

  constructor(options: VMobjectOptions = {}) {
    super(options);
    this.n_points_per_cubic_curve = options.n_points_per_cubic_curve ?? 4;
    this._bezier_t_values = linspace(0, 1, this.n_points_per_cubic_curve);
    this.fill_opacity = options.fill_opacity ?? 0;
    this.stroke_opacity = options.stroke_opacity ?? 1;
    this.stroke_width = options.stroke_width ?? 4;
    this.close_new_points = options.close_new_points ?? false;
    if (options.sheen_factor !== undefined) this.sheen_factor = options.sheen_factor;
    if (options.joint_type !== undefined && options.joint_type !== null) this.joint_type = options.joint_type;
    if (options.sheen_direction !== undefined) this.sheen_direction = [...options.sheen_direction];
    if (options.pre_function_handle_to_anchor_scale_factor !== undefined) {
      this.pre_function_handle_to_anchor_scale_factor = options.pre_function_handle_to_anchor_scale_factor;
    }
    if (options.make_smooth_after_applying_functions !== undefined) {
      this.make_smooth_after_applying_functions = options.make_smooth_after_applying_functions;
    }
    if (options.background_image !== undefined) this.background_image = options.background_image;
    if (options.shade_in_3d !== undefined) this.shade_in_3d = options.shade_in_3d;
    if (options.tolerance_for_point_equality !== undefined) {
      this.tolerance_for_point_equality = options.tolerance_for_point_equality;
    }
    if (options.cap_style !== undefined) this.cap_style = options.cap_style;
    if (options.background_stroke_color !== undefined) {
      this.background_stroke_color = options.background_stroke_color === null
        ? null
        : new ManimColor(options.background_stroke_color);
    }
    if (options.background_stroke_opacity !== undefined) {
      this.background_stroke_opacity = options.background_stroke_opacity;
    }
    if (options.background_stroke_width !== undefined) {
      this.background_stroke_width = options.background_stroke_width;
    }
    // NOTE: rgba arrays follow Python/NumPy semantics: a single row for a flat
    // color (broadcast over every point) and one row per stop only for a real
    // gradient.  Earlier ports expanded these to one row *per point*, which
    // made every style read/write and per-frame color interpolation O(points)
    // instead of O(1); that dominated the render loop.
    const fillColor = new ManimColor(options.fill_color ?? options.color ?? WHITE);
    const strokeColor = options.fill_color === undefined && options.stroke_color === undefined
      ? fillColor
      : new ManimColor(options.stroke_color ?? options.color ?? WHITE);
    this.fill_rgbas = array([
      colorRgbaValues(fillColor, this.fill_opacity),
    ]) as NDArray;
    this.stroke_rgbas = array([
      colorRgbaValues(strokeColor, this.stroke_opacity),
    ]) as NDArray;
    this.background_stroke_rgbas = array([[0, 0, 0, this.background_stroke_opacity]]) as NDArray;
    this.init_colors();
  }

  _assert_valid_submobjects(submobjects: Iterable<Mobject>): this {
    return this._assert_valid_submobjects_internal(submobjects, VMobject);
  }

  *__iter__(): IterableIterator<VMobject> {
    for (const mobject of super.__iter__()) yield mobject as VMobject;
  }

  get n_points_per_curve(): number {
    return this.n_points_per_cubic_curve;
  }

  static get_mobject_type_class(): typeof VMobject {
    return VMobject;
  }

  static getMobjectTypeClass(): typeof VMobject {
    return VMobject.get_mobject_type_class();
  }

  set_points(points: unknown): this {
    assertFinitePoints(points);
    this.points = (
      isNDArray(points) ? points.copy() : array(points as unknown[])
    ) as NDArray;
    return this;
  }

  setPoints(points: unknown): this {
    return this.set_points(points);
  }

  get_last_point(): NDArray {
    const all = rows(this.points, this.dim);
    return array(all[all.length - 1] ?? []) as NDArray;
  }

  get_sheen_factor(): number {
    return this.sheen_factor;
  }

  get_gradient_start_and_end_points(): [number[], number[]] {
    if (this.shade_in_3d) {
      return [get3dVmobStartCorner(this), get3dVmobEndCorner(this)];
    }
    const direction = this.get_sheen_direction();
    const c = this.get_center();
    const cols = [RIGHT, UP, OUT].map((vect) => {
      const edge = this.get_edge_center(vect);
      return edge.map((value, index) => value - (c[index] ?? 0));
    });
    const offset = [0, 1, 2].map((i) =>
      cols.reduce((sum, col, j) => sum + (col[i] ?? 0) * (direction[j] ?? 0), 0),
    );
    return [
      c.map((value, index) => value - (offset[index] ?? 0)),
      c.map((value, index) => value + (offset[index] ?? 0)),
    ];
  }

  getGradientStartAndEndPoints(): [number[], number[]] {
    return this.get_gradient_start_and_end_points();
  }

  get_sheen_direction(): number[] {
    return [...this.sheen_direction];
  }

  rotate_sheen_direction(angle: number, axis: number[] | readonly number[] | NDArray = [...OUT], family = true): this {
    // Hoist the rotation matrix and apply it with scalar math: the previous
    // per-submobject `rotate_vector` rebuilt a 3x3 matrix (trig + numpy
    // matmul) for every family member on every rotate.
    const rot = pointRows(rotation_matrix(angle, axis), 3);
    const r00 = rot[0]?.[0] ?? 0; const r01 = rot[0]?.[1] ?? 0; const r02 = rot[0]?.[2] ?? 0;
    const r10 = rot[1]?.[0] ?? 0; const r11 = rot[1]?.[1] ?? 0; const r12 = rot[1]?.[2] ?? 0;
    const r20 = rot[2]?.[0] ?? 0; const r21 = rot[2]?.[1] ?? 0; const r22 = rot[2]?.[2] ?? 0;
    const dim = this.dim;
    const rotateOne = (vec: number[] | readonly number[] | NDArray): number[] => {
      if (!Array.isArray(vec) || vec.length > 3) {
        return [...(rows(rotate_vector(vec, angle, axis), dim)[0] ?? [])];
      }
      const x = vec[0] ?? 0;
      const y = vec[1] ?? 0;
      const z = vec[2] ?? 0;
      const out = [
        (r00 * x + r01 * y) + r02 * z,
        (r10 * x + r11 * y) + r12 * z,
        (r20 * x + r21 * y) + r22 * z,
      ];
      return dim < 3 ? out.slice(0, dim) : out;
    };
    if (family) {
      for (const submob of this.get_family()) {
        (submob as VMobject).sheen_direction = rotateOne((submob as VMobject).sheen_direction);
      }
    } else {
      this.sheen_direction = rotateOne(this.sheen_direction);
    }
    return this;
  }

  rotateSheenDirection(angle: number, axis: number[] | readonly number[] | NDArray = [...OUT], family = true): this {
    return this.rotate_sheen_direction(angle, axis, family);
  }

  override scale(
    scale_factor: number | readonly number[] | NDArray,
    // NOTE: Python distinguishes these keyword-only args
    // (scale_stroke vs about_point/about_edge). TS callers pass them
    // positionally, so the second slot accepts either a scale_stroke flag
    // or an about_point (mirroring Mobject.scale's positional convention).
    scale_stroke_or_about_point: boolean | readonly number[] | NDArray | Record<string, unknown> | null = false,
    about_point?: readonly number[] | NDArray | Record<string, unknown> | null,
    about_edge?: readonly number[] | NDArray | Record<string, unknown> | null,
  ): this {
    // NOTE (py2js kwargs bridge): a trailing kwargs bag such as
    // `{about_point: ORIGIN}` (e.g. via `animate.scale(2,
    // about_point=ORIGIN)` when the bridge cannot map it) must not land
    // verbatim in the about-point slot (NaN poison). Unwrap any plain-object
    // bag in any slot; the `__argnames__` pinned at the bottom of this file
    // lets the bridge map the Python spellings in the common case.
    const isBagLike = (value: unknown): value is Record<string, unknown> => {
      if (value === null || value === undefined || typeof value !== "object") return false;
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return false;
      if (isNDArray(value)) return false;
      if (typeof (value as { get_num_points?: unknown }).get_num_points === "function") return false;
      return true;
    };
    const pullBag = (bag: Record<string, unknown>): { scaleStroke?: boolean; point?: unknown; edge?: unknown } => {
      const out: { scaleStroke?: boolean; point?: unknown; edge?: unknown } = {};
      if (typeof bag["scale_stroke"] === "boolean") out.scaleStroke = Boolean(bag["scale_stroke"]);
      else if (typeof bag["scaleStroke"] === "boolean") out.scaleStroke = Boolean(bag["scaleStroke"]);
      if (bag["about_point"] !== undefined) out.point = bag["about_point"];
      else if (bag["aboutPoint"] !== undefined) out.point = bag["aboutPoint"];
      if (bag["about_edge"] !== undefined) out.edge = bag["about_edge"];
      else if (bag["aboutEdge"] !== undefined) out.edge = bag["aboutEdge"];
      return out;
    };
    let scale_stroke = false;
    let resolved_about_point = (isBagLike(about_point) ? undefined : about_point) as readonly number[] | NDArray | null | undefined;
    let resolved_about_edge = (isBagLike(about_edge) ? undefined : about_edge) as readonly number[] | NDArray | null | undefined;
    for (const slot of [scale_stroke_or_about_point, about_point, about_edge]) {
      if (isBagLike(slot)) {
        const pulled = pullBag(slot);
        if (pulled.scaleStroke !== undefined && typeof scale_stroke_or_about_point !== "boolean") scale_stroke = pulled.scaleStroke;
        if (pulled.point !== undefined && resolved_about_point === undefined) resolved_about_point = pulled.point as never;
        if (pulled.edge !== undefined && resolved_about_edge === undefined) resolved_about_edge = pulled.edge as never;
      }
    }
    if (typeof scale_stroke_or_about_point === "boolean") {
      scale_stroke = scale_stroke_or_about_point;
    } else if (!isBagLike(scale_stroke_or_about_point) && scale_stroke_or_about_point !== undefined && scale_stroke_or_about_point !== null) {
      if (resolved_about_point === undefined) resolved_about_point = scale_stroke_or_about_point as never;
      else if (resolved_about_edge === undefined) resolved_about_edge = scale_stroke_or_about_point as never;
    }
    if (scale_stroke) {
      // NOTE: a vector factor (e.g. `scale([2, 0.7, 1])`) has no single
      // scalar for stroke width; use the mean of the absolute per-axis
      // factors as the scalar equivalent (uniform factors are exact).
      let strokeFactor: number;
      if (typeof scale_factor === "number" || typeof scale_factor === "bigint") {
        strokeFactor = Math.abs(Number(scale_factor));
      } else {
        const rows = pointRows(scale_factor, this.dim);
        const vec = rows[0] ?? [1];
        let rawLen: number | null = null;
        if (Array.isArray(scale_factor)) rawLen = scale_factor.length;
        else if (ArrayBuffer.isView(scale_factor)) rawLen = (scale_factor as unknown as ArrayLike<unknown>).length;
        else if (isNDArray(scale_factor)) {
          const shape = (scale_factor as NDArray).shape;
          rawLen = shape.length === 0 ? 1 : shape.reduce((acc, size) => acc * size, 1);
        }
        if (rawLen === 1) {
          strokeFactor = Math.abs(Number(vec[0] ?? 1));
        } else {
          const count = rawLen ?? vec.length;
          const total = vec.slice(0, count).reduce((acc, value) => acc + Math.abs(Number(value ?? 0)), 0);
          strokeFactor = count > 0 ? total / count : 1;
        }
      }
      for (const mob of this.get_family()) {
        if (mob instanceof VMobject) {
          mob.set_stroke(
            undefined,
            strokeFactor * mob.get_stroke_width(),
            undefined,
            false,
            false,
          );
          mob.set_background_stroke({
            width: strokeFactor * mob.get_stroke_width(true),
            family: false,
          });
        }
      }
    }
    super.scale(scale_factor, resolved_about_point ?? undefined, resolved_about_edge ?? undefined);
    return this;
  }

  override rotate(
    angle: number,
    axis: readonly number[] | NDArray = [...OUT],
    about_point?: readonly number[] | NDArray | null,
    about_edge?: readonly number[] | NDArray | null,
  ): this {
    this.rotate_sheen_direction(angle, axis);
    super.rotate(angle, axis, about_point ?? undefined, about_edge ?? undefined);
    return this;
  }

  override apply_function(
    function_: (point: number[]) => number[] | NDArray,
    about_point?: readonly number[] | NDArray | null,
    about_edge?: readonly number[] | NDArray | null,
  ): this {
    // One shared family walk for both handle-scalings and the application
    // itself (three full walks per frame otherwise).  Behaviour matches the
    // three separate calls exactly, including the dropped about_point/edge
    // (preserved) and the flat-path fallback order.
    const factor = this.pre_function_handle_to_anchor_scale_factor;
    const members = this.family_members_with_points();
    this.scale_handle_to_anchor_distances_members(members, factor);
    if (!this.apply_function_to_members_flat(members, function_)) {
      super.apply_function(function_);
    }
    this.scale_handle_to_anchor_distances_members(members, 1.0 / factor);
    if (this.make_smooth_after_applying_functions) {
      this.make_smooth();
    }
    return this;
  }

  set_shade_in_3d(value = true, z_index_as_group = false): this {
    for (const submob of this.get_family()) {
      (submob as VMobject).shade_in_3d = value;
      if (z_index_as_group) {
        submob.z_index_group = this;
      }
    }
    return this;
  }

  setShadeIn3d(value = true, z_index_as_group = false): this {
    return this.set_shade_in_3d(value, z_index_as_group);
  }

  set_sheen_direction(direction: number[] | readonly number[] | NDArray, family = true): this {
    const direction_copy = [...pointRows(direction, this.dim)[0] ?? []];
    if (family) {
      for (const submob of this.get_family()) {
        (submob as VMobject).sheen_direction = [...direction_copy];
      }
    } else {
      this.sheen_direction = [...direction_copy];
    }
    return this;
  }

  set_sheen(
    factor: number,
    direction?: number[] | readonly number[] | NDArray | null,
    family = true,
  ): this {
    // Mirrors Python VMobject.set_sheen: recurse into submobjects, store
    // the factor, optionally set the direction (without further recursion),
    // then re-apply stroke/fill colors so the sheen takes effect.
    if (family) {
      for (const submob of this.submobjects) {
        (submob as VMobject).set_sheen(factor, direction, family);
      }
    }
    this.sheen_factor = factor;
    if (direction !== undefined && direction !== null) {
      // Family already handled above, matching Python's family=False here.
      this.set_sheen_direction(direction, false);
    }
    // Reset color to put sheen_factor into effect. A null stroke color means
    // "keep" (Python passes None through set_stroke the same way).
    if (factor !== 0) {
      this.set_stroke(this.get_stroke_color(), undefined, undefined, false, family);
      this.set_fill(this.get_fill_color(), undefined, family);
    }
    return this;
  }

  setSheen(
    factor: number,
    direction?: number[] | readonly number[] | NDArray | null,
    family = true,
  ): this {
    return this.set_sheen(factor, direction, family);
  }

  getSheenDirection(): number[] {
    return this.get_sheen_direction();
  }

  getSheenFactor(): number {
    return this.get_sheen_factor();
  }

  getLastPoint(): NDArray {
    return this.get_last_point();
  }

  consider_points_equals(p0: unknown, p1: unknown): boolean {
    const a = (
      isNDArray(p0) ? p0 : array(p0 as unknown[])
    ) as NDArray;
    const b = (
      isNDArray(p1) ? p1 : array(p1 as unknown[])
    ) as NDArray;
    return allclose(a, b, 1e-05, this.tolerance_for_point_equality);
  }

  considerPointsEquals(p0: unknown, p1: unknown): boolean {
    return this.consider_points_equals(p0, p1);
  }

  gen_cubic_bezier_tuples_from_points(points: unknown): number[][][] {
    const nppcc = this.n_points_per_cubic_curve;
    const all = rows(points, this.dim);
    const remainder = all.length % nppcc;
    const trimmed = all.slice(0, all.length - remainder);
    const result: number[][][] = [];
    for (let i = 0; i < trimmed.length; i += nppcc) {
      result.push(trimmed.slice(i, i + nppcc));
    }
    return result;
  }

  genCubicBezierTuplesFromPoints(points: unknown): number[][][] {
    return this.gen_cubic_bezier_tuples_from_points(points);
  }

  generate_rgbas_array(
    color: ParsableManimColor | Iterable<ManimColor> | null,
    opacity: number | Iterable<number>,
  ): NDArray {
    // First arg can be either a color, or a tuple/list of colors.
    // Likewise, opacity can either be a float, or a tuple of floats.
    // If self.sheen_factor is not zero, and only one color was passed in,
    // a second slightly light color will automatically be added for the gradient.
    const sheen_factor = this.sheen_factor;
    // Fast path: a single color and a single/scalar opacity (the common,
    // uniform case). Avoids `tuplify`/`make_even`/`pointRows` and the
    // redundant `new ManimColor(...)` copy when a ManimColor is already given.
    const singleColor =
      color === null ||
      color === undefined ||
      typeof color === "string" ||
      color instanceof ManimColor ||
      !isIterable(color);
    const singleOpacity = opacity === null || opacity === undefined || typeof opacity === "number";
    if (singleColor && singleOpacity) {
      const c =
        color === null || color === undefined
          ? BLACK
          : color instanceof ManimColor
            ? color
            : new ManimColor(color as ParsableManimColor);
      const alpha = opacity === null || opacity === undefined ? 0.0 : Number(opacity);
      const raw = colorRgbaValues(c, alpha);
      const row = [raw[0] ?? 0, raw[1] ?? 0, raw[2] ?? 0, raw[3] ?? 0];
      if (sheen_factor !== 0) {
        return array([
          row,
          [
            clamp01((row[0] ?? 0) + sheen_factor),
            clamp01((row[1] ?? 0) + sheen_factor),
            clamp01((row[2] ?? 0) + sheen_factor),
            row[3] ?? 0,
          ],
        ]) as NDArray;
      }
      return array([row]) as NDArray;
    }
    const colors: ManimColor[] = (
      tuplify(color as unknown as ManimColor) as unknown[]
    ).map((c) => (c === null || c === undefined ? BLACK : new ManimColor(c as ParsableManimColor)));
    const opacities: number[] = (
      tuplify(opacity as unknown as number) as unknown[]
    ).map((o) => (o === null || o === undefined ? 0.0 : Number(o)));
    const [evenColors, evenOpacities] = make_even(colors, opacities);
    let rows: number[][] = evenColors.map((c, index) =>
      [...(pointRows(c.to_rgba_with_alpha(evenOpacities[index] ?? 0), 4)[0] ?? [0, 0, 0, 0])],
    );

    if (sheen_factor !== 0 && rows.length === 1) {
      const light_rows: number[][] = rows.map((row) => [...row]);
      for (const row of light_rows) {
        row[0] = Math.min(1, Math.max(0, (row[0] ?? 0) + sheen_factor));
        row[1] = Math.min(1, Math.max(0, (row[1] ?? 0) + sheen_factor));
        row[2] = Math.min(1, Math.max(0, (row[2] ?? 0) + sheen_factor));
      }
      rows = [...rows, ...light_rows];
    }
    return array(rows) as NDArray;
  }

  update_rgbas_array(
    array_name: string,
    color: ParsableManimColor | Iterable<ManimColor> | null = null,
    opacity: number | Iterable<number> | null = null,
  ): this {
    const record = this as unknown as Record<string, unknown>;
    const hasColor = color !== null && color !== undefined;
    const hasOpacity = opacity !== null && opacity !== undefined;

    // Fast path: neither channel changes. Python only stretches the current
    // array when it is shorter than the freshly generated one (which, for
    // `(None, None)`, is a single row), so an existing array is returned
    // unchanged.
    if (!hasColor && !hasOpacity) {
      if (!(array_name in (this as unknown as object))) {
        record[array_name] = this.generate_rgbas_array(color, opacity as unknown as number & Iterable<number>);
      }
      return this;
    }

    let curr_rgbas: NDArray | undefined;
    if (array_name in (this as unknown as object)) {
      curr_rgbas = record[array_name] as NDArray | undefined;
    }

    // Fast path: a uniform (single color and/or scalar opacity) update of a
    // single-row rgba array. This skips `generate_rgbas_array`'s tuple
    // machinery, the stretch round-trip and the `array(pointRows(...))`
    // rebuild, touching at most four numbers instead of the whole array.
    const uniformColor =
      !hasColor || typeof color === "string" || color instanceof ManimColor || !isIterable(color);
    const uniformOpacity = !hasOpacity || typeof opacity === "number";
    if (uniformColor && uniformOpacity && this.sheen_factor === 0) {
      const currData = singleRgbaData(curr_rgbas);
      if (currData !== null) {
        const currR = currData[0] ?? 0;
        const currG = currData[1] ?? 0;
        const currB = currData[2] ?? 0;
        const currA = currData[3] ?? 0;
        let r = currR;
        let g = currG;
        let b = currB;
        let a = currA;
        if (hasColor) {
          const c = color instanceof ManimColor ? color : new ManimColor(color as ParsableManimColor);
          const raw = colorRgbaValues(c, hasOpacity ? Number(opacity) : 0);
          r = raw[0] ?? 0;
          g = raw[1] ?? 0;
          b = raw[2] ?? 0;
          if (hasOpacity) a = raw[3] ?? 0;
        } else if (hasOpacity) {
          a = Number(opacity);
        }
        if (r === currR && g === currG && b === currB && a === currA) return this;
        record[array_name] = array([[r, g, b, a]]) as NDArray;
        return this;
      }
    }

    let rgbas = this.generate_rgbas_array(color, opacity as number & Iterable<number>);
    if (curr_rgbas === undefined) {
      record[array_name] = rgbas;
      return this;
    }
    // Match up current rgbas array with the newly calculated one. 99% of
    // the time they'll be the same.
    const curr_len = curr_rgbas.shape[0] ?? 0;
    const new_len = rgbas.shape[0] ?? 0;
    if (curr_len < new_len) {
      curr_rgbas = stretch_array_to_length(curr_rgbas, new_len);
      record[array_name] = curr_rgbas;
    } else if (new_len < curr_len) {
      rgbas = stretch_array_to_length(rgbas, curr_len);
    }
    const curr_rows = pointRows(curr_rgbas, 4);
    const new_rows = pointRows(rgbas, 4);
    // Only update rgb if color was not None, and only update alpha channel
    // if opacity was passed in.
    if (color !== null && color !== undefined) {
      for (let index = 0; index < curr_rows.length; index += 1) {
        (curr_rows[index] as number[]).splice(0, 3, ...((new_rows[index] ?? [0, 0, 0]).slice(0, 3)));
      }
    }
    if (opacity !== null && opacity !== undefined) {
      for (let index = 0; index < curr_rows.length; index += 1) {
        (curr_rows[index] as number[])[3] = new_rows[index]?.[3] ?? 0;
      }
    }
    record[array_name] = array(curr_rows) as NDArray;
    return this;
  }

  init_colors(propagate_colors = true): this {
    // Fast path: the rgba arrays already encode the stored colors/opacities
    // (nothing to propagate and no sheen to bake in), so re-running the
    // setters would be a pure no-op. This is the case for essentially every
    // freshly constructed VMobject, where `init_colors` otherwise rebuilt the
    // three arrays from scratch.
    if (propagate_colors && this.colors_are_current()) return this;
    // Faithful port of Python VMobject.init_colors: re-apply the current
    // style through the setters (which own the rgba arrays). The fill/stroke
    // colors are read back from the arrays (live properties), so the arrays
    // must have been seeded first (see the constructor).
    this.set_fill(this.fill_color, this.fill_opacity, propagate_colors);
    this.set_stroke(
      this.stroke_color ?? undefined,
      this.stroke_width,
      this.stroke_opacity,
      false,
      propagate_colors,
    );
    this.set_background_stroke({
      color: this.background_stroke_color,
      width: this.background_stroke_width,
      opacity: this.background_stroke_opacity,
      family: propagate_colors,
    });
    this.set_sheen(this.sheen_factor, this.sheen_direction, propagate_colors);
    if (!propagate_colors) {
      for (const submob of this.submobjects) (submob as VMobject).init_colors(false);
    }
    return this;
  }

  /** True when every style array already matches the corresponding
   * color/opacity fields, there is nothing to propagate to submobjects and
   * no sheen to bake in — i.e. `init_colors()` would be a no-op. Only
   * single-row (uniform) arrays qualify; gradients must still be collapsed by
   * the full path, matching Python. */
  private colors_are_current(): boolean {
    if (this.submobjects.length !== 0) return false;
    if (this.sheen_factor !== 0) return false;
    const fill = singleRgbaData(this.fill_rgbas);
    const stroke = singleRgbaData(this.stroke_rgbas);
    const background = singleRgbaData(this.background_stroke_rgbas);
    if (fill === null || stroke === null || background === null) return false;
    if (fill[3] !== this.fill_opacity) return false;
    if (stroke[3] !== this.stroke_opacity) return false;
    if (background[3] !== this.background_stroke_opacity) return false;
    const background_color = this.background_stroke_color;
    if (background_color === null || background_color === undefined) return false;
    const rgb = colorRgbaValues(background_color, 0);
    return background[0] === (rgb[0] ?? 0) && background[1] === (rgb[1] ?? 0) && background[2] === (rgb[2] ?? 0);
  }

  set_fill(
    color?: ParsableManimColor | Iterable<ManimColor> | null,
    opacity?: number,
    family = true,
  ): this {
    // Python: recurse first, then update this object's rgbas; only update the
    // stored fill_opacity when an explicit opacity is given.
    if (family) {
      for (const submob of this.submobjects) (submob as VMobject).set_fill(color, opacity, family);
    }
    this.update_rgbas_array("fill_rgbas", color ?? null, opacity ?? null);
    if (opacity !== undefined && opacity !== null) this.fill_opacity = opacity;
    return this;
  }

  set_stroke(
    color?: ParsableManimColor | Iterable<ManimColor> | null,
    width?: number,
    opacity?: number,
    backgroundOrOptions: boolean | { background?: boolean; family?: boolean } = false,
    familyMaybe = true,
  ): this {
    // Python signature: set_stroke(color, width, opacity, background=False,
    // family=True). The 4th positional argument is unambiguously `background`;
    // pass `{ family }` (or the 5th positional) to control propagation.
    // Defensive: when the py2js kwargs bridge cannot map `background=` (e.g.
    // stale/missing `__argnames__`), the trailing kwargs bag `{background: ...}`
    // lands verbatim in the `color` slot and `new ManimColor(bag)` throws
    // `... not Object`. Unwrap such a bag here so
    // `tower.set_stroke(background=True)` is a (background) no-op like Python.
    if (
      color !== null &&
      color !== undefined &&
      typeof color === "object" &&
      !Array.isArray(color) &&
      !(color instanceof ManimColor) &&
      !isNDArray(color)
    ) {
      const bag = color as unknown as Record<string, unknown>;
      const looksLikeKwargsBag =
        typeof bag === "object" &&
        ("background" in bag || "family" in bag || "color" in bag || "width" in bag || "opacity" in bag);
      if (looksLikeKwargsBag) {
        const bagColor = (bag as { color?: unknown }).color;
        const bagWidth = (bag as { width?: unknown }).width;
        const bagOpacity = (bag as { opacity?: unknown }).opacity;
        const bagBackground = (bag as { background?: unknown }).background;
        const bagFamily = (bag as { family?: unknown }).family;
        if (bagColor !== undefined) {
          color = bagColor as ParsableManimColor | Iterable<ManimColor> | null;
        } else {
          color = undefined;
        }
        if (bagWidth !== undefined && width === undefined) width = bagWidth as number;
        if (bagOpacity !== undefined && opacity === undefined) opacity = bagOpacity as number;
        if (bagBackground !== undefined) {
          backgroundOrOptions = (bagFamily !== undefined
            ? { background: Boolean(bagBackground), family: Boolean(bagFamily) }
            : Boolean(bagBackground)) as boolean | { background?: boolean; family?: boolean };
          if (bagFamily !== undefined) familyMaybe = Boolean(bagFamily);
        } else if (bagFamily !== undefined && typeof backgroundOrOptions !== "object") {
          familyMaybe = Boolean(bagFamily);
        } else if (bagFamily !== undefined && typeof backgroundOrOptions === "object" && backgroundOrOptions !== null) {
          (backgroundOrOptions as { family?: boolean }).family = Boolean(bagFamily);
        }
      }
    }
    let background = false;
    let family = familyMaybe;
    if (typeof backgroundOrOptions === "object" && backgroundOrOptions !== null) {
      background = backgroundOrOptions.background ?? false;
      family = backgroundOrOptions.family ?? true;
    } else {
      background = Boolean(backgroundOrOptions);
    }
    if (family) {
      for (const submob of this.submobjects) {
        (submob as VMobject).set_stroke(color, width, opacity, background, family);
      }
    }
    if (background) {
      this.update_rgbas_array("background_stroke_rgbas", color ?? null, opacity ?? null);
      if (width !== undefined) this.background_stroke_width = width;
      if (opacity !== undefined) this.background_stroke_opacity = opacity;
      if (color !== undefined && color !== null) {
        this.background_stroke_color = parseSingleColor(color);
      }
      return this;
    }
    this.update_rgbas_array("stroke_rgbas", color ?? null, opacity ?? null);
    if (width !== undefined) this.stroke_width = width;
    if (opacity !== undefined) this.stroke_opacity = opacity;
    return this;
  }

  set_cap_style(cap_style: CapStyleType): this {
    this.cap_style = cap_style as unknown as string;
    return this;
  }

  setCapStyle(cap_style: CapStyleType): this {
    return this.set_cap_style(cap_style);
  }

  nonempty_submobjects(): VMobject[] {
    return this.submobjects.filter(
      (submob) => submob.submobjects.length !== 0 || submob.get_num_points() !== 0,
    ) as VMobject[];
  }

  override split(): VMobject[] {
    const result: VMobject[] = this.get_num_points() > 0 ? [this] : [];
    return [...result, ...(this.submobjects as VMobject[])];
  }

  set_background_stroke(options: {
    color?: ParsableManimColor | null;
    width?: number;
    opacity?: number;
    family?: boolean;
  } = {}): this {
    const { color = null, width, opacity, family = true } = options;
    if (family) {
      for (const submob of this.submobjects) {
        (submob as VMobject).set_background_stroke({ color, width, opacity, family });
      }
    }
    this.update_rgbas_array("background_stroke_rgbas", color, opacity ?? null);
    if (width !== undefined) this.background_stroke_width = width;
    if (opacity !== undefined) this.background_stroke_opacity = opacity;
    if (color !== undefined && color !== null) this.background_stroke_color = new ManimColor(color);
    return this;
  }

  setBackgroundStroke(options: {
    color?: ParsableManimColor | null;
    width?: number;
    opacity?: number;
    family?: boolean;
  } = {}): this {
    return this.set_background_stroke(options);
  }

  get_fill_rgbas(): NDArray {
    if (this.fill_rgbas !== undefined && this.fill_rgbas !== null) {
      return this.fill_rgbas;
    }
    return array([[0, 0, 0, 0]]) as NDArray;
  }

  get_stroke_rgbas(background = false): NDArray {
    const rgbas = background ? this.background_stroke_rgbas : this.stroke_rgbas;
    if (rgbas !== undefined && rgbas !== null) {
      return rgbas;
    }
    return array([[0, 0, 0, 0]]) as NDArray;
  }

  get_background_image(): string | null {
    return this.background_image;
  }

  get_fill_color(): ManimColor {
    const single = singleRgbaData(this.fill_rgbas);
    if (single !== null) {
      return ((single[0] ?? 0) !== 0 || (single[1] ?? 0) !== 0 || (single[2] ?? 0) !== 0 || (single[3] ?? 0) !== 0
        ? new ManimColor(array([single[0] ?? 0, single[1] ?? 0, single[2] ?? 0]) as ParsableManimColor)
        : null) as ManimColor;
    }
    return (this.get_fill_colors()[0] ?? null) as ManimColor;
  }

  get_stroke_color(background = false): ManimColor | null {
    const single = singleRgbaData(background ? this.background_stroke_rgbas : this.stroke_rgbas);
    if (single !== null) {
      return (single[0] ?? 0) !== 0 || (single[1] ?? 0) !== 0 || (single[2] ?? 0) !== 0 || (single[3] ?? 0) !== 0
        ? new ManimColor(array([single[0] ?? 0, single[1] ?? 0, single[2] ?? 0]) as ParsableManimColor)
        : null;
    }
    return this.get_stroke_colors(background)[0] ?? null;
  }

  get_fill_opacity(): number {
    const values = rgbaRows(this.fill_rgbas).map((row) => row[3] ?? 0);
    return values.length === 0 ? this.fill_opacity : values.every((value) => value === 0) ? 0 : values[0];
  }

  get_fill_colors(): Array<ManimColor | null> {
    return rgbaRows(this.get_fill_rgbas()).map((rgba) =>
      rgba.some((value) => value !== 0)
        ? new ManimColor(array(rgba.slice(0, 3)) as ParsableManimColor)
        : null,
    );
  }

  get_fill_opacities(): NDArray {
    return array(rgbaRows(this.get_fill_rgbas()).map((rgba) => rgba[3] ?? 0)) as NDArray;
  }

  get_stroke_opacity(background = false): number {
    const values = rgbaRows(background ? this.background_stroke_rgbas : this.stroke_rgbas).map((row) => row[3] ?? 0);
    if (background) return values.length === 0 ? this.background_stroke_opacity : values[0] ?? 0;
    return values.length === 0 ? this.stroke_opacity : values[0] ?? 0;
  }

  fade(darkness = 0.5, family = true): this {
    const factor = 1.0 - darkness;
    this.set_fill(undefined, factor * this.get_fill_opacity(), false);
    this.set_stroke(undefined, undefined, factor * this.get_stroke_opacity(), false, false);
    this.set_background_stroke({
      opacity: factor * this.get_stroke_opacity(true),
      family: false,
    });
    super.fade(darkness, family);
    return this;
  }

  override set_opacity(opacity: number, family = true): this {
    // Python VMobject.set_opacity: set_fill + set_stroke + background
    // set_stroke (all family-aware).
    this.set_fill(undefined, opacity, family);
    this.set_stroke(undefined, undefined, opacity, false, family);
    this.set_stroke(undefined, undefined, opacity, true, family);
    return this;
  }

  get_stroke_colors(background = false): Array<ManimColor | null> {
    return rgbaRows(this.get_stroke_rgbas(background)).map((rgba) =>
      rgba.some((value) => value !== 0)
        ? new ManimColor(array(rgba.slice(0, 3)) as ParsableManimColor)
        : null,
    );
  }

  get_stroke_opacities(background = false): NDArray {
    return array(
      rgbaRows(this.get_stroke_rgbas(background)).map((rgba) => rgba[3] ?? 0),
    ) as NDArray;
  }

  get_stroke_width(background = false): number {
    let width: unknown;
    if (background) width = this.background_stroke_width;
    else {
      width = this.stroke_width;
      if (typeof width === "string") width = Number.parseInt(width, 10);
    }
    return Math.max(0.0, Number(width));
  }

  get_color(): ManimColor {
    // Python: `if np.all(self.get_fill_opacities() == 0)` — stroke color is
    // used only when *every* fill opacity is zero (np.all([]) is True, so an
    // empty fill array also yields the stroke color).
    const single = singleRgbaData(this.fill_rgbas);
    const allZero = single !== null
      ? (single[3] ?? 0) === 0
      : rgbaRows(this.fill_rgbas).every((row) => (row[3] ?? 0) === 0);
    return (allZero ? this.get_stroke_color() : this.get_fill_color()) as ManimColor;
  }

  set_color(color?: ParsableManimColor | Iterable<ManimColor>, family?: boolean): this;
  set_color(color?: ParsableManimColor | Iterable<ManimColor>, alpha?: unknown, family?: boolean): this;
  set_color(color?: ParsableManimColor | Iterable<ManimColor>, familyOrAlpha: unknown = true, familyMaybe?: boolean): this {
    // Python: set_color(self, color, family=True) -> set_fill + set_stroke.
    // The extra `alpha` parameter exists only for compatibility with
    // Mobject.set_color(color, alpha, family), which forwards positionally
    // to submobjects that may be VMobjects: set_color(c, null, family).
    let family = true;
    if (familyMaybe !== undefined) {
      family = familyMaybe;
    } else if (typeof familyOrAlpha === "boolean") {
      family = familyOrAlpha;
    } else {
      family = true;
    }
    this.set_fill(color, undefined, family);
    this.set_stroke(color, undefined, undefined, false, family);
    return this;
  }

  setColor(color?: ParsableManimColor | Iterable<ManimColor>, family?: boolean): this;
  setColor(color?: ParsableManimColor | Iterable<ManimColor>, alpha?: unknown, family?: boolean): this;
  setColor(color: ParsableManimColor | Iterable<ManimColor>, familyOrAlpha: unknown = true, familyMaybe?: boolean): this {
    return this.set_color(color, familyOrAlpha as boolean, familyMaybe as boolean);
  }

  get_style(simple = false): Record<string, unknown> {
    const ret: Record<string, unknown> = {
      stroke_opacity: this.get_stroke_opacity(),
      stroke_width: this.get_stroke_width(),
    };
    if (simple) {
      ret["fill_color"] = this.get_fill_color();
      ret["fill_opacity"] = this.get_fill_opacity();
      ret["stroke_color"] = this.get_stroke_color();
    } else {
      ret["fill_color"] = this.get_fill_colors();
      ret["fill_opacity"] = this.get_fill_opacities();
      ret["stroke_color"] = this.get_stroke_colors();
      ret["background_stroke_color"] = this.get_stroke_colors(true);
      ret["background_stroke_width"] = this.get_stroke_width(true);
      ret["background_stroke_opacity"] = this.get_stroke_opacity(true);
      ret["sheen_factor"] = this.get_sheen_factor();
      ret["sheen_direction"] = this.get_sheen_direction();
      ret["background_image"] = this.get_background_image();
    }
    return ret;
  }

  getStyle(simple = false): Record<string, unknown> { return this.get_style(simple); }

  set_style(options: {
    fill_color?: ParsableManimColor | ManimColor[] | NDArray | null;
    fill_opacity?: number | number[] | NDArray | null;
    stroke_color?: ParsableManimColor | ManimColor[] | NDArray | null;
    stroke_width?: number | null;
    stroke_opacity?: number | number[] | NDArray | null;
    background_stroke_color?: ParsableManimColor | ManimColor[] | NDArray | null;
    background_stroke_width?: number | null;
    background_stroke_opacity?: number | number[] | NDArray | null;
    sheen_factor?: number | null;
    sheen_direction?: number[] | readonly number[] | NDArray | null;
    background_image?: string | null;
    family?: boolean;
  } = {}): this {
    const {
      fill_color = null,
      fill_opacity = null,
      stroke_color = null,
      stroke_width = null,
      stroke_opacity = null,
      background_stroke_color = null,
      background_stroke_width = null,
      background_stroke_opacity = null,
      sheen_factor = null,
      sheen_direction = null,
      background_image = null,
      family = true,
    } = options;
    this.set_fill(
      (fill_color ?? undefined) as ParsableManimColor | Iterable<ManimColor> | undefined,
      singleOpacity(fill_opacity) ?? undefined,
      family,
    );
    this.set_stroke(
      (stroke_color ?? undefined) as ParsableManimColor | Iterable<ManimColor> | undefined,
      stroke_width ?? undefined,
      singleOpacity(stroke_opacity) ?? undefined,
      false,
      family,
    );
    this.set_background_stroke({
      color: singleColor(background_stroke_color),
      width: background_stroke_width ?? undefined,
      opacity: singleOpacity(background_stroke_opacity) ?? undefined,
      family,
    });
    if (sheen_factor) {
      this.set_sheen(
        sheen_factor,
        sheen_direction ?? undefined,
        family,
      );
    }
    if (background_image) {
      this.color_using_background_image(background_image);
    }
    return this;
  }

  setStyle(options: Parameters<VMobject["set_style"]>[0] = {}): this {
    return this.set_style(options);
  }

  /** Uniform fast path for {@link match_style}: when both objects carry a
   *  single rgba row for fill/stroke/background and the source has no sheen or
   *  background-image to bake in, copy the scalar style directly instead of
   *  round-tripping through `get_style()`/`set_style()` (which build several
   *  colour/opacity arrays and then refill them). The assignments reproduce
   *  exactly what `set_style(get_style())` would leave behind, including the
   *  `background_stroke_color` / opacity bookkeeping. Returns false to fall
   *  back to the faithful full path. */
  private _match_uniform_style(source: VMobject): boolean {
    if (source.sheen_factor || source.background_image) return false;
    const sFill = singleRgbaData(source.fill_rgbas);
    const sStroke = singleRgbaData(source.stroke_rgbas);
    const sBack = singleRgbaData(source.background_stroke_rgbas);
    const tFill = singleRgbaData(this.fill_rgbas);
    const tStroke = singleRgbaData(this.stroke_rgbas);
    const tBack = singleRgbaData(this.background_stroke_rgbas);
    if (sFill === null || sStroke === null || sBack === null) return false;
    if (tFill === null || tStroke === null || tBack === null) return false;
    const sf0 = sFill[0] ?? 0; const sf1 = sFill[1] ?? 0; const sf2 = sFill[2] ?? 0; const sf3 = sFill[3] ?? 0;
    const ss0 = sStroke[0] ?? 0; const ss1 = sStroke[1] ?? 0; const ss2 = sStroke[2] ?? 0; const ss3 = sStroke[3] ?? 0;
    const sb0 = sBack[0] ?? 0; const sb1 = sBack[1] ?? 0; const sb2 = sBack[2] ?? 0; const sb3 = sBack[3] ?? 0;
    if (tFill[0] !== sf0 || tFill[1] !== sf1 || tFill[2] !== sf2 || tFill[3] !== sf3) {
      this.fill_rgbas = array([[sf0, sf1, sf2, sf3]]) as NDArray;
    }
    this.fill_opacity = sf3;
    if (tStroke[0] !== ss0 || tStroke[1] !== ss1 || tStroke[2] !== ss2 || tStroke[3] !== ss3) {
      this.stroke_rgbas = array([[ss0, ss1, ss2, ss3]]) as NDArray;
    }
    this.stroke_width = source.get_stroke_width();
    this.stroke_opacity = ss3;
    if (sb0 === 0 && sb1 === 0 && sb2 === 0 && sb3 === 0) {
      // `set_background_stroke(null, ...)` only rewrites the alpha channel and
      // leaves the existing rgb / colour field untouched.
      if ((tBack[3] ?? 0) !== sb3) {
        this.background_stroke_rgbas = array([[tBack[0] ?? 0, tBack[1] ?? 0, tBack[2] ?? 0, sb3]]) as NDArray;
      }
    } else {
      if (tBack[0] !== sb0 || tBack[1] !== sb1 || tBack[2] !== sb2 || tBack[3] !== sb3) {
        this.background_stroke_rgbas = array([[sb0, sb1, sb2, sb3]]) as NDArray;
      }
      const bgColor = this.background_stroke_color;
      const bgRgb = bgColor === null || bgColor === undefined ? null : colorRgbaValues(bgColor, 0);
      if (bgRgb === null || bgRgb[0] !== sb0 || bgRgb[1] !== sb1 || bgRgb[2] !== sb2) {
        this.background_stroke_color = new ManimColor(array([sb0, sb1, sb2]) as ParsableManimColor);
      }
    }
    this.background_stroke_width = source.get_stroke_width(true);
    this.background_stroke_opacity = sb3;
    return true;
  }

  match_style(vmobject: VMobject, family = true): this {
    if (!this._match_uniform_style(vmobject)) {
      this.set_style({ ...(vmobject.get_style() as Parameters<VMobject["set_style"]>[0]), family: false });
    }
    if (family) {
      // Does its best to match up submobject lists, and
      // match styles accordingly
      const submobs1 = this.submobjects;
      const submobs2 = vmobject.submobjects;
      if (submobs1.length === 0) return this;
      const second: Mobject[] = submobs2.length === 0 ? [vmobject] : [...submobs2];
      const [even1, even2] = make_even([...submobs1], second);
      even1.forEach((sm1, index) => {
        (sm1 as VMobject).match_style(even2[index] as VMobject);
      });
    }
    return this;
  }

  matchStyle(vmobject: VMobject, family = true): this {
    return this.match_style(vmobject, family);
  }

  color_using_background_image(background_image: string): this {
    this.background_image = background_image;
    this.set_color(WHITE);
    for (const submob of this.submobjects) {
      (submob as VMobject).color_using_background_image(background_image);
    }
    return this;
  }

  colorUsingBackgroundImage(background_image: string): this {
    return this.color_using_background_image(background_image);
  }

  match_background_image(vmobject: VMobject): this {
    this.color_using_background_image(vmobject.get_background_image() as string);
    return this;
  }

  matchBackgroundImage(vmobject: VMobject): this {
    return this.match_background_image(vmobject);
  }

  get_num_curves(): number {
    return Math.floor(this.get_num_points() / this.n_points_per_cubic_curve);
  }

  has_new_path_started(): boolean {
    return this.get_num_points() % this.n_points_per_cubic_curve === 1;
  }

  set_anchors_and_handles(
    anchors1: number[][] | NDArray,
    handles1: number[][] | NDArray,
    handles2: number[][] | NDArray,
    anchors2: number[][] | NDArray,
  ): this {
    const a1 = rows(anchors1, this.dim);
    const h1 = rows(handles1, this.dim);
    const h2 = rows(handles2, this.dim);
    const a2 = rows(anchors2, this.dim);
    assertFinitePoints([a1, h1, h2, a2]);
    if (!(a1.length === h1.length && h1.length === h2.length && h2.length === a2.length)) {
      const error = new Error("assert len(anchors1) == len(handles1) == len(handles2) == len(anchors2)");
      error.name = "AssertionError";
      throw error;
    }
    const nppcc = this.n_points_per_cubic_curve;
    const total = nppcc * a1.length;
    const points: number[][] = Array.from({ length: total }, () => new Array(this.dim).fill(0));
    for (const [index, arr] of [a1, h1, h2, a2].entries()) {
      for (let k = 0; k < arr.length; k += 1) points[(index + k * nppcc) as number] = [...(arr[k] ?? new Array(this.dim).fill(0))];
    }
    this.points = asPoints(points, this.dim);
    return this;
  }

  setAnchorsAndHandles(
    anchors1: number[][] | NDArray,
    handles1: number[][] | NDArray,
    handles2: number[][] | NDArray,
    anchors2: number[][] | NDArray,
  ): this {
    return this.set_anchors_and_handles(anchors1, handles1, handles2, anchors2);
  }

  append_points(points: unknown): this {
    const current = rows(this.points, this.dim);
    // Fast path: plain rows already exactly `dim` wide need no re-slicing.
    const additions: number[][] =
      Array.isArray(points) &&
      (points.length === 0 ||
        (Array.isArray(points[0]) && (points[0] as number[]).length === this.dim))
        ? (points as number[][])
        : rows(points, this.dim);
    this.points = asPoints([...current, ...additions], this.dim);
    return this;
  }

  add_cubic_bezier_curve(
    anchor1: unknown,
    handle1: unknown,
    handle2: unknown,
    anchor2: unknown,
  ): this {
    // TODO, check the len(self.points) % 4 == 0?
    const pts = [anchor1, handle1, handle2, anchor2].map(
      (p) => pointRows(p, this.dim)[0] ?? new Array(this.dim).fill(0),
    );
    this.append_points(pts);
    return this;
  }

  addCubicBezierCurve(
    anchor1: unknown,
    handle1: unknown,
    handle2: unknown,
    anchor2: unknown,
  ): this {
    return this.add_cubic_bezier_curve(anchor1, handle1, handle2, anchor2);
  }

  add_cubic_bezier_curves(curves: unknown): this {
    // Python: self.append_points(curves.flatten()) — the flattened 1D array
    // is assigned into the 2D points buffer, i.e. re-chunked by dim.
    if (isNDArray(curves)) {
      const flat = (curves.toArray() as unknown[]).flat(Infinity).map(numeric);
      const chunked: number[][] = [];
      for (let i = 0; i < flat.length; i += this.dim) {
        const chunk = flat.slice(i, i + this.dim);
        while (chunk.length < this.dim) chunk.push(0);
        chunked.push(chunk);
      }
      this.append_points(chunked);
    } else {
      this.append_points(curves);
    }
    return this;
  }

  addCubicBezierCurves(curves: unknown): this {
    return this.add_cubic_bezier_curves(curves);
  }

  add_cubic_bezier_curve_to(handle1: unknown, handle2: unknown, anchor: unknown): this {
    this.throw_error_if_no_points("add_cubic_bezier_curve_to");
    const h1 = pointRows(handle1, this.dim)[0] ?? new Array(this.dim).fill(0);
    const h2 = pointRows(handle2, this.dim)[0] ?? new Array(this.dim).fill(0);
    const a = pointRows(anchor, this.dim)[0] ?? new Array(this.dim).fill(0);
    const new_points = [h1, h2, a];
    if (this.has_new_path_started()) {
      this.append_points(new_points);
    } else {
      this.append_points([pointAt(this.points, this.get_num_points() - 1, this.dim), ...new_points]);
    }
    return this;
  }

  addCubicBezierCurveTo(handle1: unknown, handle2: unknown, anchor: unknown): this {
    return this.add_cubic_bezier_curve_to(handle1, handle2, anchor);
  }

  add_quadratic_bezier_curve_to(handle: unknown, anchor: unknown): this {
    // How does one approximate a quadratic with a cubic?
    // refer to the Wikipedia page on Bezier curves
    // https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Degree_elevation, accessed Jan 20, 2021
    // 1. Copy the end points, and then
    // 2. Place the 2 middle control points 2/3 along the line segments
    // from the end points to the quadratic curve's middle control point.
    const h = pointRows(handle, this.dim)[0] ?? new Array(this.dim).fill(0);
    const a = pointRows(anchor, this.dim)[0] ?? new Array(this.dim).fill(0);
    const last = pointAt(this.points, this.get_num_points() - 1, this.dim);
    const lerp = (p: number[], q: number[], t: number): number[] =>
      p.map((value, index) => (1 - t) * value + t * (q[index] ?? 0));
    this.add_cubic_bezier_curve_to(lerp(last, h, 2 / 3), lerp(a, h, 2 / 3), a);
    return this;
  }

  addQuadraticBezierCurveTo(handle: unknown, anchor: unknown): this {
    return this.add_quadratic_bezier_curve_to(handle, anchor);
  }

  add_line_to(point: unknown): this {
    const end = pointRows(point, this.dim)[0] ?? new Array(this.dim).fill(0);
    const start = pointAt(this.points, this.get_num_points() - 1, this.dim);
    const interpolated = this._bezier_t_values
      .slice(1)
      .map((t) => start.map((value, index) => (1 - t) * value + t * (end[index] ?? 0)));
    const [handle1, handle2, anchor] = interpolated;
    this.add_cubic_bezier_curve_to(handle1, handle2, anchor);
    return this;
  }

  addLineTo(point: unknown): this {
    return this.add_line_to(point);
  }

  add_smooth_curve_to(...points: unknown[]): this {
    // TODO remove the value error and just add two parameters with one optional
    let handle2: number[] | null;
    let new_anchor: number[];
    if (points.length === 1) {
      handle2 = null;
      new_anchor = pointRows(points[0], this.dim)[0] ?? new Array(this.dim).fill(0);
    } else if (points.length === 2) {
      handle2 = pointRows(points[0], this.dim)[0] ?? new Array(this.dim).fill(0);
      new_anchor = pointRows(points[1], this.dim)[0] ?? new Array(this.dim).fill(0);
    } else {
      const error = new Error(`Only call add_smooth_curve_to with 1 or 2 points`);
      error.name = "ValueError";
      throw error;
    }

    if (this.has_new_path_started()) {
      this.add_line_to(new_anchor);
    } else {
      this.throw_error_if_no_points("add_smooth_curve_to");
      const all = rows(this.points, this.dim);
      const last_h2 = all[all.length - 2] ?? new Array(this.dim).fill(0);
      const last_a2 = all[all.length - 1] ?? new Array(this.dim).fill(0);
      const last_tangent = last_a2.map((value, index) => value - (last_h2[index] ?? 0));
      const handle1 = last_a2.map((value, index) => value + (last_tangent[index] ?? 0));
      if (handle2 === null) {
        const to_anchor_vect = new_anchor.map((value, index) => value - (last_a2[index] ?? 0));
        const rotated = pointRows(
          rotate_vector(last_tangent, PI, to_anchor_vect),
          this.dim,
        )[0] ?? new Array(this.dim).fill(0);
        handle2 = new_anchor.map((value, index) => value - (rotated[index] ?? 0));
      }
      this.append_points([last_a2, handle1, handle2, new_anchor]);
    }
    return this;
  }

  addSmoothCurveTo(...points: unknown[]): this {
    return this.add_smooth_curve_to(...points);
  }

  add_subpath(points: unknown): this {
    const all = rows(points, this.dim);
    if (all.length % 4 !== 0) {
      const error = new Error("assert len(points) % 4 == 0");
      error.name = "AssertionError";
      throw error;
    }
    this.append_points(all);
    return this;
  }

  addSubpath(points: unknown): this {
    return this.add_subpath(points);
  }

  close_path(): this {
    if (!this.is_closed()) {
      const subpaths = this.get_subpaths();
      const last = subpaths[subpaths.length - 1];
      const first = last !== undefined ? pointRows(last, this.dim)[0] ?? [] : [];
      this.add_line_to(first);
    }
    return this;
  }

  closePath(): this {
    return this.close_path();
  }

  isClosed(): boolean {
    return this.is_closed();
  }

  add_points_as_corners(points: unknown): this {
    this.throw_error_if_no_points("add_points_as_corners");
    const corners = rows(points, this.dim);
    if (corners.length === 0) return this;
    const current = rows(this.points, this.dim);
    const start_corners = [current[current.length - 1] ?? new Array(this.dim).fill(0), ...corners.slice(0, -1)];
    const end_corners = corners;
    const trimmed = this.has_new_path_started() ? current.slice(0, -1) : current;
    const nppcc = this.n_points_per_cubic_curve;
    const new_points: number[][] = [];
    for (let k = 0; k < start_corners.length; k += 1) {
      const start = start_corners[k] ?? new Array(this.dim).fill(0);
      const end = end_corners[k] ?? [...start];
      for (let i = 0; i < nppcc; i += 1) {
        const t = nppcc === 1 ? 0 : i / (nppcc - 1);
        new_points.push(start.map((value, column) => (1 - t) * value + t * (end[column] ?? 0)));
      }
    }
    this.points = asPoints([...trimmed, ...new_points], this.dim);
    return this;
  }

  set_points_as_corners(points: unknown): this {
    // This sets the handles aligned with the anchors, i.e. each bezier
    // curve is the segment between its two anchors.
    const corners = rows(points, this.dim);
    if (corners.length === 0) {
      // NOTE: mirrors Python, where `np.zeros((0, dim))` assignment raises
      // "ValueError: could not broadcast input array from shape (0,) into
      // shape (0,3)". Keep the ValueError (message included) instead of
      // silently clearing points.
      const error = new Error(
        "could not broadcast input array from shape (0,) into shape (0,3)",
      );
      error.name = "ValueError";
      throw error;
    }
    const starts = corners.slice(0, -1);
    const ends = corners.slice(1);
    const at = (t: number): number[][] =>
      pointRows(
        interpolate(array(starts) as NDArray, array(ends) as NDArray, t) as NDArray,
        this.dim,
      );
    const [a1, h1, h2, a2] = this._bezier_t_values.map(at);
    this.set_anchors_and_handles(a1 ?? [], h1 ?? [], h2 ?? [], a2 ?? []);
    return this;
  }

  set_points_smoothly(points: unknown): this {
    this.set_points_as_corners(points);
    this.make_smooth();
    return this;
  }

  clear_points(): this {
    this.points = asPoints([], this.dim);
    return this;
  }

  clearPoints(): this {
    return this.clear_points();
  }

  get_anchors_and_handles(): [number[][], number[][], number[][], number[][]] {
    const nppcc = this.n_points_per_cubic_curve;
    const all = rows(this.points, this.dim);
    const out: [number[][], number[][], number[][], number[][]] = [[], [], [], []];
    for (let i = 0; i < nppcc; i += 1) {
      const part: number[][] = [];
      for (let index = i; index < all.length; index += nppcc) {
        part.push([...(all[index] ?? new Array(this.dim).fill(0))]);
      }
      out[i] = part;
    }
    return out;
  }

  getAnchorsAndHandles(): [number[][], number[][], number[][], number[][]] {
    return this.get_anchors_and_handles();
  }

  change_anchor_mode(mode: "jagged" | "smooth"): this {
    if (mode !== "jagged" && mode !== "smooth") {
      const error = new Error('mode must be either "jagged" or "smooth"');
      error.name = "AssertionError";
      throw error;
    }
    const nppcc = this.n_points_per_cubic_curve;
    for (const submob of this.family_members_with_points()) {
      const vmob = submob as VMobject;
      // Work on plain point rows end-to-end: the previous implementation went
      // rows -> NDArray subpaths -> rows again (and padded/realigned each
      // time), which dominated smoothing of high-resolution graphs.
      const subpaths = this._subpaths_rows(rows(vmob.points, this.dim));
      vmob.clear_points();
      for (const subpathRows of subpaths) {
        const stepped: number[][] = [];
        for (let index = 0; index < subpathRows.length; index += nppcc) {
          stepped.push([...(subpathRows[index] ?? new Array(this.dim).fill(0))]);
        }
        stepped.push([...(subpathRows[subpathRows.length - 1] ?? new Array(this.dim).fill(0))]);
        const anchors = stepped;
        let h1: number[][];
        let h2: number[][];
        if (mode === "smooth") {
          const [h1rows, h2rows] = get_smooth_cubic_bezier_handle_points_rows(anchors);
          h1 = h1rows;
          h2 = h2rows;
        } else {
          const a1 = anchors.slice(0, -1);
          const a2 = anchors.slice(1);
          h1 = pointRows(interpolate(array(a1) as NDArray, array(a2) as NDArray, 1.0 / 3) as NDArray, this.dim);
          h2 = pointRows(interpolate(array(a1) as NDArray, array(a2) as NDArray, 2.0 / 3) as NDArray, this.dim);
        }
        // `subpathRows` is a fresh slice whose rows are not shared with any
        // other live data, so the handle slots can be overwritten in place.
        const newSubpath = subpathRows;
        let h1Index = 0;
        for (let index = 1; index < newSubpath.length; index += nppcc) {
          newSubpath[index] = h1[h1Index] ?? new Array(this.dim).fill(0);
          h1Index += 1;
        }
        let h2Index = 0;
        for (let index = 2; index < newSubpath.length; index += nppcc) {
          newSubpath[index] = h2[h2Index] ?? new Array(this.dim).fill(0);
          h2Index += 1;
        }
        vmob.append_points(newSubpath);
      }
    }
    return this;
  }

  changeAnchorMode(mode: "jagged" | "smooth"): this {
    return this.change_anchor_mode(mode);
  }

  make_smooth(): this {
    return this.change_anchor_mode("smooth");
  }

  makeSmooth(): this {
    return this.make_smooth();
  }

  make_jagged(): this {
    return this.change_anchor_mode("jagged");
  }

  makeJagged(): this {
    return this.make_jagged();
  }

  scale_handle_to_anchor_distances(factor: number): this {
    return this.scale_handle_to_anchor_distances_members(this.family_members_with_points(), factor);
  }

  /**
   * `scale_handle_to_anchor_distances` over a precomputed member list, so
   * `apply_function` can share one family walk across both scalings and the
   * application.  Internal; the public signature is unchanged.
   *
   * @internal
   */
  scale_handle_to_anchor_distances_members(members: Mobject[], factor: number): this {
    for (const submob of members) {
      const vmob = submob as VMobject;
      if (factor !== 1 && this.scale_handle_to_anchor_distances_flat(vmob, factor)) {
        continue;
      }
      if (vmob.get_num_points() < this.n_points_per_cubic_curve) {
        continue;
      }
      const [a1, h1, h2, a2] = vmob.get_anchors_and_handles();
      // Python tolerates overflow through PhaseFlow/apply_function (it only
      // emits RuntimeWarning and keeps inf/nan points; the Camera collapses
      // them to the origin at draw time). Do the same here: when the current
      // anchors/handles are already non-finite, skip the handle rescaling
      // instead of throwing via set_anchors_and_handles. Direct user inputs
      // with NaN/Infinity still fail fast in set_points/set_anchors_and_handles.
      if (hasNonFinite([a1, h1, h2, a2])) {
        continue;
      }
      const a1ToH1 = h1.map((row, index) => subtractPoints(row, a1[index] ?? row, this.dim));
      const a2ToH2 = h2.map((row, index) => subtractPoints(row, a2[index] ?? row, this.dim));
      const newH1 = a1.map((row, index) => addPoints(row, scalePoint(a1ToH1[index] ?? new Array(this.dim).fill(0), factor, this.dim), this.dim));
      const newH2 = a2.map((row, index) => addPoints(row, scalePoint(a2ToH2[index] ?? new Array(this.dim).fill(0), factor, this.dim), this.dim));
      if (hasNonFinite([newH1, newH2])) {
        continue;
      }
      vmob.set_anchors_and_handles(a1, newH1, newH2, a2);
    }
    return this;
  }

  scale_handle_to_anchor_distances_flat(vmob: VMobject, factor: number): boolean {
    // In-place version of the handle scaling below: anchors stay, handles
    // move toward them.  Returns false when the slow path must be used.
    const nppcc = this.n_points_per_cubic_curve;
    if (vmob.get_num_points() < nppcc) return true;
    const flat = _cContiguousRows((vmob as { points?: unknown }).points);
    if (flat === null || !(flat.data instanceof Float64Array)) return false;
    const data = flat.data;
    const { rowDim, rowCount } = flat;
    const dim = vmob.dim;
    if (flat.rowDim < dim) return false;
    // `data` is float64: elements are already numbers, so no `Number()`
    // coercion is needed (it cost a call per element per frame).
    for (let index = 0; index < rowCount * rowDim; index += 1) {
      if (!Number.isFinite(data[index]!)) return false;
    }
    for (let k = 0; k + nppcc <= rowCount; k += nppcc) {
      const a1 = k * rowDim;
      const h1 = (k + 1) * rowDim;
      const h2 = (k + 2) * rowDim;
      const a2 = (k + 3) * rowDim;
      for (let c = 0; c < dim; c += 1) {
        const a1c = data[a1 + c] ?? 0;
        const a2c = data[a2 + c] ?? 0;
        data[h1 + c] = a1c + ((data[h1 + c] ?? 0) - a1c) * factor;
        data[h2 + c] = a2c + ((data[h2 + c] ?? 0) - a2c) * factor;
      }
    }
    vmob.bump_points_version();
    return true;
  }

  scaleHandleToAnchorDistances(factor: number): this {
    return this.scale_handle_to_anchor_distances(factor);
  }

  get_start_anchors(): number[][] {
    const result: number[][] = [];
    for (let index = 0; index < this.get_num_points(); index += this.n_points_per_cubic_curve) {
      const point = pointAt(this.points, index, this.dim);
      result.push(point);
    }
    return result;
  }

  get_end_anchors(): number[][] {
    const result: number[][] = [];
    const nppcc = this.n_points_per_cubic_curve;
    const count = this.get_num_points();
    for (let index = nppcc - 1; index < count; index += nppcc) {
      result.push(pointAt(this.points, index, this.dim));
    }
    return result;
  }

  get_anchors(): number[][] {
    if (this.get_num_points() === 1) return rows(this.points, this.dim);
    const starts = this.get_start_anchors();
    const ends = this.get_end_anchors();
    // NOTE: Python returns list(chain(zip(s, e, strict=True))): interleaved
    // start/end anchors (length 2 * num_curves), not just the starts.
    if (starts.length !== ends.length) {
      const error = new Error(
        `zip() argument 2 is shorter than argument 1 (strict zip lengths: ${starts.length}, ${ends.length})`,
      );
      error.name = "ValueError";
      throw error;
    }
    const result: number[][] = [];
    for (let index = 0; index < starts.length; index += 1) {
      result.push(starts[index] ?? []);
      result.push(ends[index] ?? []);
    }
    return result;
  }

  get_nth_curve_points(index: number): NDArray {
    if (!(index < this.get_num_curves())) {
      const error = new Error("assert n < self.get_num_curves()");
      error.name = "AssertionError";
      throw error;
    }
    const nppcc = this.n_points_per_cubic_curve;
    return array(
      rows(this.points, this.dim).slice(nppcc * index, nppcc * (index + 1)),
    ) as NDArray;
  }

  get_nth_curve_function(index: number): (t: number) => number[] {
    const nppcc = this.n_points_per_cubic_curve;
    const start = index * nppcc;
    // Read just the four control points of this curve.  Materialising every
    // row (`pointRows(this.points)`) here made `get_curve_functions_with_lengths`
    // O(n_curves * n_points) == O(n_points^2) per call, which dominated
    // `point_from_proportion` (MoveAlongPath) and arc-length queries.
    const points: number[][] = [];
    for (let offset = 0; offset < 4; offset += 1) {
      points.push(pointAt(this.points, start + offset, this.dim));
    }
    return (t: number) => curveValue(points, t, this.dim);
  }

  get_nth_curve_length_pieces(n: number, sample_points: number | null = null): NDArray {
    if (!(n < this.get_num_curves())) {
      const error = new Error("assert n < self.get_num_curves()");
      error.name = "AssertionError";
      throw error;
    }
    const count = sample_points ?? 10;
    const curve = this.get_nth_curve_function(n);
    const points = linspace(0, 1, count).map((alpha) => curve(alpha));
    const norms: number[] = [];
    for (let index = 1; index < points.length; index += 1) {
      const current = points[index] ?? [];
      const previous = points[index - 1] ?? [];
      norms.push(Math.hypot(...current.map((value, component) => value - (previous[component] ?? 0))));
    }
    return array(norms) as NDArray;
  }

  getNthCurveLengthPieces(n: number, sample_points: number | null = null): NDArray {
    return this.get_nth_curve_length_pieces(n, sample_points);
  }

  get_nth_curve_length(n: number, sample_points: number | null = null): number {
    const [, length] = this.get_nth_curve_function_with_length(n, sample_points);
    return length;
  }

  getNthCurveLength(n: number, sample_points: number | null = null): number {
    return this.get_nth_curve_length(n, sample_points);
  }

  get_nth_curve_function_with_length(index: number, sample_points: number | null = null): [(t: number) => number[], number] {
    const curve = this.get_nth_curve_function(index);
    // NOTE: mirrors Python `length = np.sum(norms)`: the pieces array is 1D
    // with shape (count - 1,), so nestedRows yields a single flat row.
    const flat = (nestedRows(this.get_nth_curve_length_pieces(index, sample_points))[0] ?? []) as unknown[];
    const length = (flat as number[]).reduce((sum, value) => sum + Number(value), 0);
    return [curve, length];
  }

  get_curve_functions_with_lengths(sample_points: number | null = null): Array<[(t: number) => number[], number]> {
    return Array.from({ length: this.get_num_curves() }, (_, index) => this.get_nth_curve_function_with_length(index, sample_points));
  }

  *get_curve_functions(): Generator<(t: number) => number[], void, void> {
    for (let index = 0; index < this.get_num_curves(); index += 1) {
      yield this.get_nth_curve_function(index);
    }
  }

  *getCurveFunctions(): Generator<(t: number) => number[], void, void> {
    yield* this.get_curve_functions();
  }

  get_arc_length(sample_points_per_curve: number | null = null): number {
    return this.get_curve_functions_with_lengths(sample_points_per_curve).reduce((sum, [, length]) => sum + length, 0);
  }

  get_points_defining_boundary(): NDArray {
    // Probably returns all anchors, but this is weird regarding the name of the method.
    // Fast path: read the start/end anchors straight out of each family
    // member's C-contiguous point store into one flat buffer, instead of
    // building a nested array through `pointAt` per anchor (this is called by
    // every `get_critical_point`, e.g. many times per `set_value`).
    const nppcc = this.n_points_per_cubic_curve;
    const dim = this.dim;
    const flat: number[] = [];
    for (const mobject of this.get_family()) {
      const vmob = mobject as VMobject;
      const store = vmob.points as
        | { data?: ArrayLike<number>; shape?: readonly number[]; flags?: { C_CONTIGUOUS?: boolean } }
        | undefined;
      const data = store?.data;
      const shape = store?.shape;
      const rowDim = shape !== undefined && shape.length === 2 ? (shape[1] ?? dim) : null;
      if (
        data !== undefined &&
        rowDim !== null &&
        (store?.flags === undefined || store.flags.C_CONTIGUOUS !== false)
      ) {
        const rowCount = shape?.[0] ?? 0;
        if (rowCount === 1) {
          for (let column = 0; column < dim; column += 1) flat.push(numeric(data[column] ?? 0));
          continue;
        }
        for (let row = 0; row + nppcc <= rowCount; row += nppcc) {
          const startBase = row * rowDim;
          for (let column = 0; column < dim; column += 1) {
            flat.push(numeric(data[startBase + column] ?? 0));
          }
          const endBase = (row + nppcc - 1) * rowDim;
          for (let column = 0; column < dim; column += 1) {
            flat.push(numeric(data[endBase + column] ?? 0));
          }
        }
        continue;
      }
      for (const anchor of vmob.get_anchors()) {
        for (let column = 0; column < dim; column += 1) flat.push(numeric(anchor[column] ?? 0));
      }
    }
    if (flat.length === 0) return array([]).reshape(0, dim) as NDArray;
    return array(flat).reshape(flat.length / dim, dim) as NDArray;
  }

  getPointsDefiningBoundary(): NDArray {
    return this.get_points_defining_boundary();
  }

  /**
   * Single-pass counterpart of `Mobject.get_critical_point` over this
   * class's anchor boundary.  The base implementation builds the full anchor
   * buffer first and scans it per axis; Text / DecimalNumber groups call
   * `get_critical_point` (via `move_to`) many times per frame, so this scans
   * min/max for all axes in one walk without materialising the boundary.
   */
  override get_critical_point(direction: readonly number[] | NDArray): number[] {
    const dim = this.dim;
    if (
      !(
        Array.isArray(direction)
        && (direction.length === 0 || typeof (direction as readonly unknown[])[0] === "number")
      )
    ) {
      return super.get_critical_point(direction);
    }
    const directionValues = direction as readonly number[];
    const nppcc = this.n_points_per_cubic_curve;
    let min0 = Infinity; let max0 = -Infinity; let finite0 = 0;
    let min1 = Infinity; let max1 = -Infinity; let finite1 = 0;
    let min2 = Infinity; let max2 = -Infinity; let finite2 = 0;
    const consider = (x: number, y: number, z: number): void => {
      // Skip non-finite anchor samples (singularities such as log(0), error 385).
      if (Number.isFinite(x)) { finite0 += 1; if (x < min0) min0 = x; if (x > max0) max0 = x; }
      if (Number.isFinite(y)) { finite1 += 1; if (y < min1) min1 = y; if (y > max1) max1 = y; }
      if (Number.isFinite(z)) { finite2 += 1; if (z < min2) min2 = z; if (z > max2) max2 = z; }
    };
    for (const mobject of this.get_family()) {
      const store = (mobject as { points?: unknown }).points as
        | { data?: ArrayLike<number>; shape?: readonly number[]; flags?: { C_CONTIGUOUS?: boolean } }
        | undefined;
      const data = store?.data;
      const shape = store?.shape;
      const rowDim = shape !== undefined && shape.length === 2 ? (shape[1] ?? dim) : null;
      if (
        data !== undefined
        && rowDim !== null
        && (store?.flags === undefined || store.flags.C_CONTIGUOUS !== false)
      ) {
        const rowCount = shape?.[0] ?? 0;
        if (rowCount === 1) {
          consider(
            numeric(data[0] ?? 0),
            numeric(data[1] ?? 0),
            numeric(data[2] ?? 0),
          );
          continue;
        }
        for (let row = 0; row + nppcc <= rowCount; row += nppcc) {
          const startBase = row * rowDim;
          consider(
            numeric(data[startBase] ?? 0),
            numeric(data[startBase + 1] ?? 0),
            numeric(data[startBase + 2] ?? 0),
          );
          const endBase = (row + nppcc - 1) * rowDim;
          consider(
            numeric(data[endBase] ?? 0),
            numeric(data[endBase + 1] ?? 0),
            numeric(data[endBase + 2] ?? 0),
          );
        }
        continue;
      }
      for (const anchor of (mobject as VMobject).get_anchors()) {
        consider(numeric(anchor[0] ?? 0), numeric(anchor[1] ?? 0), numeric(anchor[2] ?? 0));
      }
    }
    if (finite0 === 0 && finite1 === 0 && finite2 === 0) return Array.from({ length: dim }, () => 0);
    const pick = (
      min: number, max: number, finiteCount: number, key: number,
    ): number => (finiteCount === 0 ? 0 : key < 0 ? min : key > 0 ? max : (min + max) / 2);
    const result = new Array<number>(dim);
    result[0] = pick(min0, max0, finite0, Number(directionValues[0] ?? 0));
    result[1] = pick(min1, max1, finite1, Number(directionValues[1] ?? 0));
    result[2] = pick(min2, max2, finite2, Number(directionValues[2] ?? 0));
    return result;
  }

  point_from_proportion(alpha: number): NDArray {
    if (alpha < 0 || alpha > 1) {
      const error = new Error(`Alpha ${alpha} not between 0 and 1.`);
      error.name = "ValueError";
      throw error;
    }
    this.throw_error_if_no_points("point_from_proportion");
    const pointRowsValue = rows(this.points, this.dim);
    if (alpha === 1) return array(pointRowsValue[pointRowsValue.length - 1] ?? [0, 0, 0]) as NDArray;
    const curves = this.get_curve_functions_with_lengths();
    const targetLength = alpha * curves.reduce((sum, [, length]) => sum + length, 0);
    let currentLength = 0;
    for (const [curve, length] of curves) {
      if (currentLength + length >= targetLength) {
        const residue = length === 0 ? 0 : (targetLength - currentLength) / length;
        return array(curve(residue)) as NDArray;
      }
      currentLength += length;
    }
    throw new Error("Not sure how you reached here, please file a bug report at https://github.com/ManimCommunity/manim/issues/new/choose");
  }

  proportion_from_point(point: readonly number[] | NDArray): number {
    this.throw_error_if_no_points("proportion_from_point");
    // Iterate over each bezier curve that the VMobject is composed of,
    // checking if the point lies on that curve. If it does not lie on that
    // curve, add the whole length of the curve to target_length and move
    // onto the next curve. If the point does lie on the curve, add how far
    // along the curve the point is to target_length. Then, divide
    // target_length by the total arc length of the shape to get the
    // proportion along the VMobject the point is at.
    const numCurves = this.get_num_curves();
    const totalLength = this.get_arc_length();
    let targetLength = 0;
    for (let n = 0; n < numCurves; n += 1) {
      const controlPoints = this.get_nth_curve_points(n);
      const length = this.get_nth_curve_length(n);
      const proportions = ((nestedRows(
        proportions_along_bezier_curve_for_point(
          pointFrom(point, this.dim).toArray() as number[],
          pointRows(controlPoints, this.dim),
        ),
      )[0] ?? []) as unknown[]).map(Number);
      if (proportions.length > 0) {
        const proportionAlongNthCurve = Math.max(...proportions);
        targetLength += length * proportionAlongNthCurve;
        break;
      }
      targetLength += length;
      if (n === numCurves - 1) {
        const error = new Error(`Point ${JSON.stringify(point)} does not lie on this curve.`);
        error.name = "ValueError";
        throw error;
      }
    }
    return targetLength / totalLength;
  }

  proportionFromPoint(point: readonly number[] | NDArray): number {
    return this.proportion_from_point(point);
  }

  reverse_direction(): this {
    this.points = asPoints(rows(this.points, this.dim).reverse(), this.dim);
    return this;
  }

  interpolate_color(mobject1: VMobject, mobject2: VMobject, alpha: number): this {
    // Raw targets: one proxy trap per mobject instead of one per attribute
    // access below (24+ traps per face per frame otherwise; see `_rawMobject`).
    // All seven attributes always exist on VMobjects, so raw access is exact.
    const raw1 = _rawMobject(mobject1) as unknown as Record<string, unknown>;
    const raw2 = _rawMobject(mobject2) as unknown as Record<string, unknown>;
    const rawSelf = _rawMobject(this) as unknown as Record<string, unknown>;
    // Static-colour fast path (see `staticColorMemo`): when the six value
    // attributes are memo-verified static, the destination already holds
    // them, so only `sheen_direction` needs its (bitwise-identical) in-place
    // write.  Anything else — morphing colours, exotic dtypes, `alpha === 1`
    // — falls through to the exact original loop below.
    if (alpha !== 1) {
      let staticOk = false;
      const memo = staticColorMemo.get(rawSelf);
      if (memo !== undefined) {
        const refs = staticMemoRefs(raw1, raw2, rawSelf);
        if (refs.length === memo.refs.length && refs.every((ref, i) => ref === memo.refs[i])) {
          staticOk = memo.staticOk;
        } else {
          staticOk = verifyStaticColors(raw1, raw2, rawSelf);
          memo.refs = refs;
          memo.staticOk = staticOk;
        }
      } else {
        staticOk = verifyStaticColors(raw1, raw2, rawSelf);
        staticColorMemo.set(rawSelf, { refs: staticMemoRefs(raw1, raw2, rawSelf), staticOk });
      }
      if (staticOk) {
        const inv = 1 - alpha;
        if (writeSheenLerp(rawSelf["sheen_direction"], raw1["sheen_direction"], raw2["sheen_direction"], inv, alpha)) {
          return this;
        }
      }
    }
    const attributes = ["fill_rgbas", "stroke_rgbas", "background_stroke_rgbas", "stroke_width", "background_stroke_width", "sheen_direction", "sheen_factor"] as const;
    for (const attribute of attributes) {
      const first = raw1[attribute];
      const second = raw2[attribute];
      const result = typeof first === "number" || typeof second === "number" ? Number(first) * (1 - alpha) + Number(second) * alpha : interpolateArray(first, second, alpha);
      rawSelf[attribute] = alpha === 1 ? cloneAttribute(second) : result;
    }
    return this;
  }

  get_group_class(): typeof VGroup { return VGroup; }

  pointwise_become_partial(other: VMobject, a: number, b: number): this {
    if (!(other instanceof VMobject)) {
      throw new TypeError(
        `Expected a VMobject, got value ${other} of type ${pythonTypeName(other)}.`,
      );
    }
    // Partial curve includes three portions:
    // - A middle section, which matches the curve exactly.
    // - A start, which is some ending portion of an inner cubic.
    // - An end, which is the starting portion of a later inner cubic.
    if (a <= 0 && b >= 1) {
      this.set_points(rows(other.points, this.dim));
      return this;
    }
    const num_curves = other.get_num_curves();
    if (num_curves === 0) return this;
    const [lower_index, lower_residue] = integer_interpolate(0, num_curves, a);
    const [upper_index, upper_residue] = integer_interpolate(0, num_curves, b);
    const nppc = this.n_points_per_curve;
    // Copy other.points when other is this to prevent in-place modification.
    const source = other === this ? rows(other.points, this.dim).map((row) => [...row]) : rows(other.points, this.dim);
    const slice = (curve: number): number[][] => source.slice(nppc * curve, nppc * (curve + 1));
    if (lower_index === upper_index) {
      this.points = asPoints(partial_bezier_points(slice(lower_index), lower_residue, upper_residue), this.dim);
    } else {
      const result: number[][] = [
        ...partial_bezier_points(slice(lower_index), lower_residue, 1),
        ...source.slice(nppc * (lower_index + 1), nppc * upper_index),
        ...partial_bezier_points(slice(upper_index), 0, upper_residue),
      ];
      this.points = asPoints(result, this.dim);
    }
    return this;
  }

  append_vectorized_mobject(vectorized_mobject: VMobject): this {
    if (this.has_new_path_started()) {
      // Remove last point, which is starting a new path.
      const current = rows(this.points, this.dim);
      this.points = asPoints(current.slice(0, -1), this.dim);
    }
    this.append_points(vectorized_mobject.points);
    return this;
  }

  appendVectorizedMobject(vectorized_mobject: VMobject): this {
    return this.append_vectorized_mobject(vectorized_mobject);
  }

  get_subcurve(a: number, b: number): this {
    let vmob: this;
    if (this.is_closed() && a > b) {
      vmob = this.copy();
      vmob.pointwise_become_partial(this, a, 1);
      const vmob2 = this.copy();
      vmob2.pointwise_become_partial(this, 0, b);
      vmob.append_vectorized_mobject(vmob2);
    } else {
      vmob = this.copy();
      vmob.pointwise_become_partial(this, a, b);
    }
    return vmob;
  }

  getSubcurve(a: number, b: number): this {
    return this.get_subcurve(a, b);
  }

  get_direction(): "CW" | "CCW" {
    return shoelace_direction(this.get_start_anchors()) as "CW" | "CCW";
  }

  getDirection(): "CW" | "CCW" {
    return this.get_direction();
  }

  force_direction(target_direction: "CW" | "CCW"): this {
    if (target_direction !== "CW" && target_direction !== "CCW") {
      const error = new Error('Invalid input for force_direction. Use "CW" or "CCW"');
      error.name = "ValueError";
      throw error;
    }
    if (this.get_direction() !== target_direction) {
      // Since we already assured the input is CW or CCW,
      // and the directions don't match, we just reverse
      this.reverse_direction();
    }
    return this;
  }

  forceDirection(target_direction: "CW" | "CCW"): this {
    return this.force_direction(target_direction);
  }

  get_point_mobject(center?: readonly number[]): VectorizedPoint {
    const point = new VectorizedPoint(center ?? this.get_center());
    point.match_style(this);
    return point;
  }

  consider_points_equals_2d(p0: readonly number[], p1: readonly number[]): boolean {
    const rtol = 1.0e-5;
    const atol = this.tolerance_for_point_equality;
    const x0 = p0[0] ?? NaN;
    const y0 = p0[1] ?? NaN;
    const x1 = p1[0] ?? NaN;
    const y1 = p1[1] ?? NaN;
    if (x0 !== x1) {
      const x_diff = Math.abs((x0 as number) - (x1 as number));
      if (!(x_diff <= atol + rtol * Math.abs(x1 as number) && Number.isFinite(x_diff))) return false;
    }
    if (y0 === y1) return true;
    const y_diff = Math.abs((y0 as number) - (y1 as number));
    return Boolean(y_diff <= atol + rtol * Math.abs(y1 as number) && Number.isFinite(y_diff));
  }

  get_subpath_split_indices_from_points(points: unknown, n_dims = 3): NDArray {
    const full = fullRows(points);
    const nppcc = this.n_points_per_cubic_curve;
    const n_pts = full.length;
    if (n_pts < nppcc) return array([]).reshape(0, 2) as NDArray;
    const boundary_indices: number[] = [];
    for (let index = nppcc; index < n_pts; index += nppcc) boundary_indices.push(index);
    if (boundary_indices.length === 0) return array([[0, n_pts]]) as NDArray;
    const rtol = 1.0e-5;
    const atol = this.tolerance_for_point_equality;
    const is_split = boundary_indices.map((boundary) => {
      const end = full[boundary - 1] ?? [];
      const start = full[boundary] ?? [];
      for (let dim = 0; dim < n_dims; dim += 1) {
        const e = end[dim] ?? NaN;
        const s = start[dim] ?? NaN;
        const diff = Math.abs(e - s);
        const threshold = atol + rtol * Math.abs(s);
        let close = diff <= threshold;
        close = close && Number.isFinite(diff);
        close = close || e === s;
        if (!close) return true;
      }
      return false;
    });
    const split_points = [0, ...boundary_indices.filter((_, index) => is_split[index]), n_pts];
    const ranges: number[][] = [];
    for (let index = 0; index < split_points.length - 1; index += 1) {
      ranges.push([split_points[index] ?? 0, split_points[index + 1] ?? 0]);
    }
    return array(ranges) as NDArray;
  }

  resize_points(
    new_length: number,
    resize_func: (points: NDArray, length: number) => NDArray = resize_array,
  ): this {
    if (new_length !== this.get_num_points()) this.points = resize_func(this.points, new_length);
    return this;
  }

  is_closed(): boolean {
    const all = rows(this.points, this.dim);
    return this.consider_points_equals(all[0] ?? [], all[all.length - 1] ?? []);
  }

  start_new_path(point: unknown): this {
    const n_points = this.get_num_points();
    const nppc = this.n_points_per_curve;
    const pointRow = pointRows(point, this.dim)[0] ?? new Array(this.dim).fill(0);
    if (n_points % nppc !== 0) {
      const anchors = this.get_start_anchors();
      const last_anchor = anchors[anchors.length - 1] ?? new Array(this.dim).fill(0);
      const closure: number[][] = Array.from(
        { length: nppc - (n_points % nppc) },
        () => [...last_anchor],
      );
      this.append_points([...closure, [...pointRow]]);
    } else {
      this.append_points([[...pointRow]]);
    }
    return this;
  }

  get_cubic_bezier_tuples_from_points(points: unknown): NDArray {
    return array(this.gen_cubic_bezier_tuples_from_points(points)) as NDArray;
  }

  get_cubic_bezier_tuples(): number[][][] {
    return this.gen_cubic_bezier_tuples_from_points(this.points);
  }

  *_gen_subpaths_from_points(
    points: NDArray,
    filter_func: (n: number) => boolean,
    allRows?: number[][],
  ): Generator<NDArray, void, void> {
    const all = allRows ?? rows(points, this.dim);
    const nppcc = this.n_points_per_cubic_curve;
    const split_points: number[] = [0];
    for (let n = nppcc; n < all.length; n += nppcc) {
      if (filter_func(n)) split_points.push(n);
    }
    split_points.push(all.length);
    for (let index = 0; index < split_points.length - 1; index += 1) {
      const start = split_points[index] ?? 0;
      const end = split_points[index + 1] ?? 0;
      if (end - start >= nppcc) yield array(all.slice(start, end)) as NDArray;
    }
  }

  /** Fast inline equivalent of `consider_points_equals` for equal-length point
   * rows: `np.allclose(p, q, rtol=1e-5, atol=tolerance_for_point_equality)`
   * without the numpy-ts allocation churn. Falls back to the general method for
   * mismatched/empty shapes. */
  private _point_rows_equal(p: readonly number[], q: readonly number[]): boolean {
    if (p.length !== q.length || p.length === 0) {
      return this.consider_points_equals(p, q);
    }
    const rtol = 1e-5;
    const atol = this.tolerance_for_point_equality;
    for (let i = 0; i < p.length; i += 1) {
      const x = p[i] ?? 0;
      const y = q[i] ?? 0;
      if (!(Math.abs(x - y) <= atol + rtol * Math.abs(y))) return false;
    }
    return true;
  }

  /** Split `all` (plain point rows) into subpaths, mirroring
   * `_gen_subpaths_from_points` without the NDArray round-trip. */
  private _subpaths_rows(all: number[][]): number[][][] {
    const nppcc = this.n_points_per_cubic_curve;
    const split_points: number[] = [0];
    for (let n = nppcc; n < all.length; n += nppcc) {
      if (!this._point_rows_equal(all[n - 1] ?? [], all[n] ?? [])) split_points.push(n);
    }
    split_points.push(all.length);
    const subpaths: number[][][] = [];
    for (let index = 0; index < split_points.length - 1; index += 1) {
      const start = split_points[index] ?? 0;
      const end = split_points[index + 1] ?? 0;
      if (end - start >= nppcc) subpaths.push(all.slice(start, end));
    }
    return subpaths;
  }

  get_subpaths_from_points(points: NDArray): NDArray[] {
    const all = rows(points, this.dim);
    return [...this._gen_subpaths_from_points(
      points,
      (n) => !this._point_rows_equal(all[n - 1] ?? [], all[n] ?? []),
      all,
    )];
  }

  *gen_subpaths_from_points_2d(points: NDArray): Generator<NDArray, void, void> {
    const all = rows(points, this.dim);
    yield* this._gen_subpaths_from_points(
      points,
      (n) => !this.consider_points_equals_2d(all[n - 1] ?? [], all[n] ?? []),
    );
  }

  get_subpaths(): NDArray[] {
    return this.get_subpaths_from_points(this.points);
  }

  insert_n_curves(n: number): this {
    let new_path_point: NDArray | null = null;
    if (this.has_new_path_started()) new_path_point = this.get_last_point();
    const new_points = this.insert_n_curves_to_point_list(n, this.points);
    this.set_points(new_points);
    if (new_path_point !== null) this.append_points(new_path_point);
    return this;
  }

  insertNCurves(n: number): this {
    return this.insert_n_curves(n);
  }

  insert_n_curves_to_point_list(n: number, points: number[][] | NDArray): NDArray {
    const all = rows(points, this.dim);
    if (all.length === 1) {
      const nppcc = this.n_points_per_cubic_curve;
      return array(Array.from({ length: nppcc * n }, () => [...(all[0] ?? [])])) as NDArray;
    }
    const bezier_tuples = this.gen_cubic_bezier_tuples_from_points(points);
    const current_number_of_curves = bezier_tuples.length;
    const new_number_of_curves = current_number_of_curves + n;
    const new_bezier_tuples = bezier_remap(bezier_tuples, new_number_of_curves);
    return array(new_bezier_tuples.flat().map((point) => point.slice(0, this.dim))) as NDArray;
  }

  align_rgbas(vmobject: VMobject): this {
    const attrs = ["fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"] as const;
    for (const attr of attrs) {
      const first = this[attr];
      const second = vmobject[attr];
      if (first.shape[0] > second.shape[0]) {
        vmobject[attr] = stretch_array_to_length(second, first.shape[0]);
      } else if (second.shape[0] > first.shape[0]) {
        this[attr] = stretch_array_to_length(first, second.shape[0]);
      }
    }
    return this;
  }

  align_points(vmobject: VMobject): this {
    this.align_rgbas(vmobject);
    // TODO: This shortcut can be a bit over eager. What if they have the same length, but different subpath lengths?
    if (this.get_num_points() === vmobject.get_num_points()) return this;

    for (const mob of [this, vmobject]) {
      // If there are no points, add one to
      // wherever the "center" is
      if (mob.has_no_points()) mob.start_new_path(mob.get_center());
      // If there's only one point, turn it into
      // a null curve
      if (mob.has_new_path_started()) mob.add_line_to(mob.get_last_point());
    }

    // Figure out what the subpaths are
    const subpaths1 = this.get_subpaths().map((subpath) => rows(subpath, this.dim));
    const subpaths2 = vmobject.get_subpaths().map((subpath) => rows(subpath, vmobject.dim));
    const n_subpaths = Math.max(subpaths1.length, subpaths2.length);

    const nppcc = this.n_points_per_cubic_curve;

    const get_nth_subpath = (path_list: number[][][], n: number): number[][] => {
      if (n >= path_list.length) {
        // Create a null path at the very end
        if (path_list.length === 0) {
          return Array.from({ length: nppcc }, () => new Array(this.dim).fill(0));
        }
        const last = path_list[path_list.length - 1];
        const lastPoint = last?.[last.length - 1] ?? new Array(this.dim).fill(0);
        return Array.from({ length: nppcc }, () => [...lastPoint]);
      }
      let path = [...(path_list[n] ?? [])];
      // Check for useless points at the end of the path and remove them
      // https://github.com/ManimCommunity/manim/issues/1959
      while (path.length > nppcc) {
        // If the last nppc points are all equal to the preceding point
        const tail = path.slice(path.length - nppcc);
        const preceding = path[path.length - nppcc - 1] ?? [];
        if (tail.every((point) => this.consider_points_equals(point, preceding))) {
          path = path.slice(0, -nppcc);
        } else break;
      }
      return path;
    };

    const new_path1: number[][] = [];
    const new_path2: number[][] = [];
    for (let n = 0; n < n_subpaths; n += 1) {
      // For each pair of subpaths, add points until they are the same length
      const sp1 = get_nth_subpath(subpaths1, n);
      const sp2 = get_nth_subpath(subpaths2, n);
      const diff1 = Math.max(0, Math.floor((sp2.length - sp1.length) / nppcc));
      const diff2 = Math.max(0, Math.floor((sp1.length - sp2.length) / nppcc));
      new_path1.push(...rows(this.insert_n_curves_to_point_list(diff1, sp1), this.dim));
      new_path2.push(...rows(vmobject.insert_n_curves_to_point_list(diff2, sp2), vmobject.dim));
    }
    this.set_points(new_path1.length > 0 ? new_path1 : array([]).reshape(0, this.dim));
    vmobject.set_points(new_path2.length > 0 ? new_path2 : array([]).reshape(0, vmobject.dim));
    return this;
  }

}

function cloneAttribute(value: unknown): unknown {
  if (isNDArray(value)) return value.copy();
  if (Array.isArray(value)) return value.map(cloneAttribute);
  return value;
}

function fullRows(value: unknown): number[][] {
  return nestedRows(value).map((row) => (Array.isArray(row) ? row : [row]).map(numeric));
}

// Ported from manim.mobject.three_d.three_d_utils (dependency of
// VMobject.get_gradient_start_and_end_points when shade_in_3d is set).
function get3dVmobStartCorner(vmob: VMobject): number[] {
  if (vmob.get_num_points() === 0) return [...ORIGIN];
  return pointAt(vmob.points, 0, vmob.dim);
}

function get3dVmobEndCorner(vmob: VMobject): number[] {
  if (vmob.get_num_points() === 0) return [...ORIGIN];
  const index = Math.floor((vmob.get_num_points() - 1) / 6) * 3;
  return pointAt(vmob.points, index, vmob.dim);
}

function callerFunctionName(): string {
  const stack = new Error().stack ?? "";
  const lines = stack.split("\n").slice(1).map((line) => line.trim());
  for (const line of lines) {
    const match = line.match(/^at (\S+)/);
    if (!match) continue;
    const short = (match[1] ?? "").split(".").pop() ?? "";
    if (short === "callerFunctionName" || short === "_throw_error_if_no_submobjects" || short === "_get_submobjects_with_points" || short === "new" || short === "Error") continue;
    return short;
  }
  return "";
}

export interface VectorizedPointOptions extends VMobjectOptions {
  artificial_width?: number;
  artificial_height?: number;
}

export class VectorizedPoint extends VMobject {
  artificial_width = 0.01;
  artificial_height = 0.01;

  constructor(
    location: unknown = ORIGIN,
    color: ParsableManimColor | VectorizedPointOptions = BLACK,
    fill_opacity = 0,
    stroke_width = 0,
    artificial_width = 0.01,
    artificial_height = 0.01,
    options: VectorizedPointOptions = {},
  ) {
    // NOTE: Python assigns artificial_width/height before super().__init__;
    // TS forbids `this` access before super(), so the field defaults above
    // (identical values) cover any reads during construction, and the
    // parameters are applied immediately after.
    let extra: VectorizedPointOptions = options;
    let resolvedColor: ParsableManimColor = BLACK;
    if (
      color !== null &&
      typeof color === "object" &&
      !(color instanceof ManimColor) &&
      !isNDArray(color) &&
      !Array.isArray(color)
    ) {
      extra = { ...(color as VectorizedPointOptions), ...options };
    } else {
      resolvedColor = color as ParsableManimColor;
    }
    super({
      color: resolvedColor,
      fill_opacity,
      stroke_width,
      ...extra,
    });
    this.artificial_width = extra.artificial_width ?? artificial_width;
    this.artificial_height = extra.artificial_height ?? artificial_height;
    this.set_points(
      array([pointRows(location, this.dim)[0] ?? new Array(this.dim).fill(0)]) as NDArray,
    );
  }

  // NOTE: Python installs these getters on the Cairo or OpenGL base class
  // depending on config.renderer; the TS port always uses the Cairo behavior.
  override get width(): number {
    return this.artificial_width;
  }

  override get height(): number {
    return this.artificial_height;
  }

  get_location(): NDArray {
    return array(rows(this.points, this.dim)[0] ?? []) as NDArray;
  }

  set_location(new_loc: unknown): this {
    this.set_points(array([pointRows(new_loc, this.dim)[0] ?? new Array(this.dim).fill(0)]) as NDArray);
    return this;
  }
}

export class VGroup extends VMobject {
  constructor(...vmobjects: (VMobject | Iterable<VMobject>)[]) {
    super();
    this.add(...vmobjects);
  }

  get_type_error_message(invalid_obj: unknown, invalid_indices: [number, number]): string {
    const renderType = "VMobject";
    const repr = toPythonRepr(invalid_obj);
    return `Only values of type ${renderType} can be added as submobjects of VGroup, but the value ${repr} (at index ${invalid_indices[1]} of parameter ${invalid_indices[0]}) is of type ${pythonTypeName(invalid_obj)}.`;
  }

  add(...vmobjects: (VMobject | Iterable<VMobject> | unknown)[]): this {
    const get_type_error_message = (invalid_obj: unknown, invalid_indices: [number, number]): string => this.get_type_error_message(invalid_obj, invalid_indices);
    const valid: VMobject[] = [];
    for (let index = 0; index < vmobjects.length; index += 1) {
      const value = vmobjects[index];
      if (value instanceof VMobject) valid.push(value);
      else if (value instanceof Mobject) {
        throw new TypeError(`${get_type_error_message(value, [index, 0])} You can try adding this value into a Group instead.`);
      } else if (isIterable(value)) {
        let subIndex = 0;
        for (const subvalue of value) {
          if (!(subvalue instanceof VMobject)) throw new TypeError(get_type_error_message(subvalue, [index, subIndex]));
          valid.push(subvalue);
          subIndex += 1;
        }
      } else {
        throw new TypeError(get_type_error_message(value, [index, 0]));
      }
    }
    super.add(...valid);
    return this;
  }

  __repr__(): string {
    return `${this.constructor.name}(${this.submobjects.map((mob) => String(mob)).join(", ")})`;
  }

  __str__(): string {
    const count = this.submobjects.length;
    return `${this.constructor.name} of ${count} submobject${count === 1 ? "" : "s"}`;
  }

  override toString(): string {
    return this.__str__();
  }

  __add__(vmobject: VMobject): VGroup {
    return new VGroup(...(this.submobjects as VMobject[]), vmobject);
  }

  __iadd__(vmobject: VMobject): this {
    return this.add(vmobject);
  }

  __sub__(vmobject: VMobject): VGroup {
    const copy = new VGroup(...(this.submobjects as VMobject[]));
    copy.remove(vmobject);
    return copy;
  }

  __isub__(vmobject: VMobject): this {
    return this.remove(vmobject);
  }

  __setitem__(key: number, value: VMobject | VMobject[]): void {
    this._assert_valid_submobjects(tuplify(value) as VMobject[]);
    this.submobjects[key] = value as VMobject;
  }

  override __getitem__(key: number | MobjectSlice): VMobject | VGroup {
    return super.__getitem__(key) as VMobject | VGroup;
  }
}

export function vmobjectFrom(value: unknown, dim = 3): VMobject {
  const result = new VMobject({ dim });
  result.points = pointsFrom(value, dim);
  return result;
}

export class VDict extends VMobject {
  show_keys = false;
  submob_dict = new Map<unknown, VMobject>();

  constructor(
    mapping_or_iterable: Map<unknown, VMobject> | Record<string, VMobject> | Iterable<[unknown, VMobject]> = {},
    show_keys = false,
    options: VMobjectOptions = {},
  ) {
    super(options);
    this.show_keys = show_keys;
    this.add(mapping_or_iterable);
  }

  toString(): string {
    return `${this.constructor.name}(${toPythonRepr([...this.submob_dict])})`;
  }

  add(
    mapping_or_iterable: unknown = {},
    ..._rest: unknown[]
  ): this {
    for (const [key, value] of normalizeMapping(mapping_or_iterable as Map<unknown, VMobject> | Record<string, VMobject> | Iterable<[unknown, VMobject]>)) {
      this.add_key_value_pair(key, value);
    }
    return this;
  }

  remove(key: unknown): this {
    if (!this.submob_dict.has(key)) {
      const error = new Error(`The given key '${String(key)}' is not present in the VDict`);
      error.name = "KeyError";
      throw error;
    }
    super.remove(this.submob_dict.get(key) as VMobject);
    this.submob_dict.delete(key);
    return this;
  }

  get(key: unknown): VMobject {
    const submob = this.submob_dict.get(key);
    if (submob === undefined) {
      const error = new Error(toPythonRepr(key));
      error.name = "KeyError";
      throw error;
    }
    return submob;
  }

  set(key: unknown, value: VMobject): void;
  set(options: Record<string, unknown>): this;
  set(first: unknown, second?: VMobject): this | void {
    if (second instanceof VMobject || typeof first !== "object" || first === null || Array.isArray(first)) {
      this.__setitem__(first, second as VMobject);
      return;
    }
    return super.set(first as Record<string, unknown>);
  }

  __setitem__(key: unknown, value: VMobject): void {
    if (this.submob_dict.has(key)) this.remove(key);
    this.add([[key, value]]);
  }

  has(key: unknown): boolean {
    return this.submob_dict.has(key);
  }

  __getitem__(key: unknown): VMobject {
    return this.get(key);
  }

  __contains__(key: unknown): boolean {
    return this.has(key);
  }

  get_all_submobjects(): VMobject[] {
    return [...this.submob_dict.values()];
  }

  add_key_value_pair(key: unknown, value: VMobject): this {
    this._assert_valid_submobjects([value]);
    const mob = value;
    if (this.show_keys) {
      if (texFactory === null) {
        throw new Error(
          "VDict.show_keys requires the text module (manim.mobject.text.tex_mobject) to be loaded",
        );
      }
      // NOTE: mirrors Python `Tex(str(key)).next_to(value, LEFT)` and adding
      // the label to the value's own submobjects.
      const keyText = texFactory(String(key)).next_to(value, [...LEFT]);
      mob.add(keyText);
    }
    this.submob_dict.set(key, mob);
    super.add(value);
    return this;
  }

  __delitem__(key: unknown): void {
    if (!this.submob_dict.has(key)) {
      const error = new Error(toPythonRepr(key));
      error.name = "KeyError";
      throw error;
    }
    this.submob_dict.delete(key);
  }
}

function normalizeMapping(
  mapping: Map<unknown, VMobject> | Record<string, VMobject> | Iterable<[unknown, VMobject]>,
): Array<[unknown, VMobject]> {
  if (mapping instanceof Map) return [...mapping.entries()];
  if (
    mapping !== null &&
    typeof mapping === "object" &&
    Symbol.iterator in (mapping as Record<symbol, unknown>) === false &&
    !Array.isArray(mapping)
  ) {
    return Object.entries(mapping as Record<string, VMobject>);
  }
  return [...(mapping as Iterable<[unknown, VMobject]>)];
}

export class CurvesAsSubmobjects extends VGroup {
  constructor(vmobject?: VMobject, options: VMobjectOptions = {}) {
    super();
    void options;
    if (vmobject !== undefined) {
      const tuples = vmobject.get_cubic_bezier_tuples();
      for (const tup of tuples) {
        const part = new VMobject();
        part.set_points(tup);
        part.match_style(vmobject);
        this.add(part);
      }
    }
  }

  point_from_proportion(alpha: number): NDArray {
    if (alpha < 0 || alpha > 1) {
      const error = new Error(`Alpha ${alpha} not between 0 and 1.`);
      error.name = "ValueError";
      throw error;
    }
    this._throw_error_if_no_submobjects();
    const submobjs_with_pts = this._get_submobjects_with_points();
    if (alpha === 1) {
      const last = submobjs_with_pts[submobjs_with_pts.length - 1];
      const lastRows = rows(last.points, last.dim);
      return array(lastRows[lastRows.length - 1] ?? []) as NDArray;
    }
    const submobjs_arc_lengths = submobjs_with_pts.map((part) => part.get_arc_length());
    const total_length = submobjs_arc_lengths.reduce((sum, length) => sum + length, 0);
    const target_length = alpha * total_length;
    let current_length = 0;
    for (let index = 0; index < submobjs_with_pts.length; index += 1) {
      const part = submobjs_with_pts[index];
      const part_length = submobjs_arc_lengths[index] ?? 0;
      if (current_length + part_length >= target_length) {
        const residue = (target_length - current_length) / part_length;
        return part.point_from_proportion(residue);
      }
      current_length += part_length;
    }
    const error = new Error("Unable to determine a point from the requested proportion.");
    error.name = "ValueError";
    throw error;
  }
  _throw_error_if_no_submobjects(): void {
    if (this.submobjects.length === 0) {
      const caller_name = callerFunctionName();
      throw new Error(
        `Cannot call CurvesAsSubmobjects. ${caller_name} for a CurvesAsSubmobject with no submobjects`,
      );
    }
  }

  _get_submobjects_with_points(): VMobject[] {
    const submobjs_with_pts = this.submobjects.filter(
      (part) => (part as VMobject).get_num_points() > 0,
    ) as VMobject[];
    if (submobjs_with_pts.length === 0) {
      const caller_name = callerFunctionName();
      throw new Error(
        `Cannot call CurvesAsSubmobjects. ${caller_name} for a CurvesAsSubmobject whose submobjects have no points`,
      );
    }
    return submobjs_with_pts;
  }
}

function interpClamped(x: number, xp: number[], fp: number[]): number {
  if (xp.length === 0) return 0;
  if (x <= (xp[0] ?? 0)) return fp[0] ?? 0;
  const last = xp.length - 1;
  if (x >= (xp[last] ?? 0)) return fp[last] ?? 0;
  for (let index = 0; index < last; index += 1) {
    const left = xp[index] ?? 0;
    const right = xp[index + 1] ?? 0;
    if (x >= left && x <= right) {
      const t = right === left ? 0 : (x - left) / (right - left);
      return (fp[index] ?? 0) * (1 - t) + (fp[index + 1] ?? 0) * t;
    }
  }
  return fp[last] ?? 0;
}

export interface DashedVMobjectOptions extends VMobjectOptions {
  num_dashes?: number;
  dashed_ratio?: number;
  dash_offset?: number;
  equal_lengths?: boolean;
}

export class DashedVMobject extends VMobject {
  dashed_ratio: number;
  num_dashes: number;

  constructor(
    vmobject: VMobject,
    num_dashes = 15,
    dashed_ratio = 0.5,
    dash_offset = 0,
    color: ParsableManimColor = WHITE,
    equal_lengths = true,
    options: DashedVMobjectOptions = {},
  ) {
    const {
      num_dashes: numDashesOption,
      dashed_ratio: dashedRatioOption,
      dash_offset: dashOffsetOption,
      equal_lengths: equalLengthsOption,
      ...rest
    } = options;
    super({ color, ...rest });
    this.dashed_ratio = dashedRatioOption ?? dashed_ratio;
    this.num_dashes = numDashesOption ?? num_dashes;
    const resolved_offset = dashOffsetOption ?? dash_offset;
    const resolved_equal_lengths = equalLengthsOption ?? equal_lengths;

    // Work on a copy to avoid mutating the caller's mobject (e.g. removing tips).
    const base_vmobject = vmobject;
    const working = base_vmobject.copy() as VMobject;

    // TipableVMobject instances (Arrow, Vector, etc.) carry tips as submobjects.
    // When dashing such objects, each subcurve would otherwise include its own
    // tip, leading to many overlapping arrowheads. Pop tips from the working
    // copy and re-attach them only once after the dashes are created.
    let tips: VMobject | null = null;
    // NOTE: mirrors Python `hasattr(vmobject, "pop_tips")`. The Mobject proxy
    // throws AttributeError on plain property reads, so use `in` (the `has`
    // trap) instead of reading the property directly.
    if ("pop_tips" in (working as unknown as Record<string, unknown>)) {
      const maybe_pop_tips = (working as unknown as { pop_tips?: () => VGroup }).pop_tips;
      if (typeof maybe_pop_tips === "function") {
        const popped_tips = maybe_pop_tips.call(working);
        if (popped_tips.submobjects.length > 0) {
          tips = popped_tips;
        }
      }
    }

    const r = this.dashed_ratio;
    const n = this.num_dashes;
    if (n > 0) {
      // Assuming total length is 1
      const dash_len = r / n;
      let void_len: number;
      if (working.is_closed()) {
        void_len = (1 - r) / n;
      } else {
        void_len = n === 1 ? 1 - r : (1 - r) / (n - 1);
      }

      const period = dash_len + void_len;
      const phase_shift = (resolved_offset % 1) * period;

      let pattern_len: number;
      if (working.is_closed()) {
        // closed curves have equal amount of dashes and voids
        pattern_len = 1;
      } else {
        // open curves start and end with a dash, so the whole dash pattern with the last void is longer
        pattern_len = 1 + void_len;
      }

      const dash_starts = Array.from({ length: n }, (_, i) => (((i * period + phase_shift) % pattern_len + pattern_len) % pattern_len));
      const dash_ends = Array.from({ length: n }, (_, i) => (((i * period + dash_len + phase_shift) % pattern_len + pattern_len) % pattern_len));

      // closed shapes can handle overflow at the 0-point
      // open shapes need special treatment for it
      if (!working.is_closed()) {
        // due to phase shift being [0...1] range, always the last dash element needs attention for overflow
        // if an entire dash moves out of the shape end:
        if ((dash_ends[dash_ends.length - 1] ?? 0) > 1 && (dash_starts[dash_starts.length - 1] ?? 0) > 1) {
          // remove the last element since it is out-of-bounds
          dash_ends.pop();
          dash_starts.pop();
        } else if ((dash_ends[dash_ends.length - 1] ?? 0) < dash_len) {
          // if it overflowed
          if ((dash_starts[dash_starts.length - 1] ?? 0) < 1) {
            // if the beginning of the piece is still in range
            dash_starts.push(0);
            dash_ends.push(dash_ends[dash_ends.length - 1] ?? 0);
            dash_ends[dash_ends.length - 2] = 1;
          } else {
            dash_starts[dash_starts.length - 1] = 0;
          }
        } else if ((dash_starts[dash_starts.length - 1] ?? 0) > 1 - dash_len) {
          dash_ends[dash_ends.length - 1] = 1;
        }
      }

      if (resolved_equal_lengths) {
        // calculate the entire length by adding up short line-pieces
        let norms: number[] = [0];
        for (let k = 0; k < working.get_num_curves(); k += 1) {
          const pieces = ((nestedRows(working.get_nth_curve_length_pieces(k))[0] ?? []) as unknown[]).map(Number);
          norms = [...norms, ...pieces];
        }
        // add up length-pieces in array form
        const length_vals: number[] = [];
        let running = 0;
        for (const value of norms) {
          running += value;
          length_vals.push(running);
        }
        const ref_points = linspace(0, 1, length_vals.length);
        const curve_length = length_vals[length_vals.length - 1] ?? 0;
        this.add(
          ...dash_starts.map((start, i) => working.get_subcurve(
            interpClamped((start ?? 0) * curve_length, length_vals, ref_points),
            interpClamped((dash_ends[i] ?? 0) * curve_length, length_vals, ref_points),
          )),
        );
      } else {
        this.add(
          ...dash_starts.map((start, i) => working.get_subcurve(start ?? 0, dash_ends[i] ?? 0)),
        );
      }
    }
    // Family is already taken care of by get_subcurve implementation
    if (config.renderer === RendererType.OPENGL) {
      this.match_style(base_vmobject, false);
    } else {
      this.match_style(base_vmobject, false);
    }

    if (tips !== null) {
      this.add(...tips.submobjects);
    }
  }
}

// Explicit Python parameter names for `VMobject.scale`, surviving bundling.
// Python: `scale(scale_factor, scale_stroke=False, *, about_point=None,
// about_edge=None)`. The TS second slot doubles as `about_point` for
// positional callers, but the names must match Python so the py2js kwargs
// bridge maps `about_point=`/`scale_stroke=` (see `wrapPrototypeMethods`).
(VMobject.prototype.scale as unknown as { __argnames__?: string[] }).__argnames__ = [
  "scale_factor",
  "scale_stroke",
  "about_point",
  "about_edge",
];

// Explicit Python parameter names for `VMobject.set_stroke`, surviving
// bundling. Python: `set_stroke(color=None, width=None, opacity=None,
// background=False, family=True)`. The TS 4th slot is named
// `backgroundOrOptions` (it also accepts `{background, family}`), which the
// generic kwargs bridge cannot map `background=` onto -- the bag then lands
// verbatim in `color` and `new ManimColor(bag)` throws `... not Object`
// (e.g. `tower.set_stroke(background=True)`). Pinning the Python spellings
// lets `wrapPrototypeMethods` place `background=`/`family=` correctly.
(VMobject.prototype.set_stroke as unknown as { __argnames__?: string[] }).__argnames__ = [
  "color",
  "width",
  "opacity",
  "background",
  "family",
];
