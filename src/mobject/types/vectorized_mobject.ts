import { allclose, array, NDArray } from "numpy-ts";
import { Mobject, type MobjectOptions, type MobjectUpdater } from "../mobject.js";
import { ManimColor, WHITE, interpolate_color, type ParsableManimColor } from "../../utils/color/core.js";
import { pointRows, pointsFrom, pointFrom, pointAt, numeric, addPoints, subtractPoints, scalePoint, norm, toPythonRepr, pythonTypeName, isNDArray, nestedRows } from "../../utils/arrays.js";
import { ORIGIN } from "../../constants.js";

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
}

function rows(value: unknown, dim: number): number[][] {
  return pointRows(value, dim);
}

function asPoints(rowsValue: number[][], dim: number): NDArray {
  return array(rowsValue.length > 0 ? rowsValue : []).reshape(rowsValue.length, dim) as NDArray;
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

function interpolateArray(first: unknown, second: unknown, alpha: number): NDArray {
  if (typeof first === "number" || typeof second === "number") return array(Number(first) * (1 - alpha) + Number(second) * alpha) as NDArray;
  const left = Array.isArray(first) ? first : isNDArray(first) ? rows(first, 4) : [first];
  const right = Array.isArray(second) ? second : isNDArray(second) ? rows(second, 4) : [second];
  const result = left.map((value, index) => {
    if (Array.isArray(value)) {
      const other = Array.isArray(right[index]) ? right[index] : [Number(right[index])];
      return value.map((item, component) => Number(item) * (1 - alpha) + Number(other[component] ?? 0) * alpha);
    }
    return Number(value) * (1 - alpha) + Number(right[index] ?? 0) * alpha;
  });
  return array(result) as NDArray;
}

export class VMobject extends Mobject {
  fill_color: ManimColor;
  stroke_color: ManimColor;
  fill_opacity: number;
  stroke_opacity: number;
  stroke_width: number;
  background_stroke_color: ManimColor | null = null;
  background_stroke_opacity = 1;
  background_stroke_width = 0;
  sheen_factor = 0;
  sheen_direction: number[] = [0, 0, 0];
  joint_type = "AUTO";
  cap_style = "AUTO";
  close_new_points = false;
  pre_function_handle_to_anchor_scale_factor = 0;
  make_smooth_after_applying_functions = true;
  background_image: string | null = null;
  shade_in_3d = true;
  tolerance_for_point_equality = 1e-6;
  n_points_per_cubic_curve: number;
  fill_rgbas: NDArray;
  stroke_rgbas: NDArray;
  background_stroke_rgbas: NDArray;

  constructor(options: VMobjectOptions = {}) {
    super(options);
    this.n_points_per_cubic_curve = options.n_points_per_cubic_curve ?? 4;
    this.fill_color = new ManimColor(options.fill_color ?? options.color ?? WHITE);
    this.stroke_color = new ManimColor(options.stroke_color ?? options.color ?? WHITE);
    this.fill_opacity = options.fill_opacity ?? 0;
    this.stroke_opacity = options.stroke_opacity ?? 1;
    this.stroke_width = options.stroke_width ?? 4;
    this.close_new_points = options.close_new_points ?? false;
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
    this.fill_rgbas = array([]) as NDArray;
    this.stroke_rgbas = array([]) as NDArray;
    this.background_stroke_rgbas = array([]) as NDArray;
    this.init_colors();
  }

  _assert_valid_submobjects(submobjects: Iterable<Mobject>): this {
    return this._assert_valid_submobjects_internal(submobjects, VMobject);
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

  init_colors(propagate_colors = true): this {
    if (!this.fill_color) this.fill_color = new ManimColor(WHITE);
    if (!this.stroke_color) this.stroke_color = new ManimColor(WHITE);
    // NOTE: mirrors Python generate_rgbas_array, which always yields at least
    // one row even when the mobject has no points yet.
    const pointCount = Math.max(1, this.get_num_points());
    this.fill_rgbas = array(Array.from({ length: pointCount }, () => [...this.fill_color.toRgb().toArray() as number[], this.fill_opacity])) as NDArray;
    this.stroke_rgbas = array(Array.from({ length: pointCount }, () => [...this.stroke_color.toRgb().toArray() as number[], this.stroke_opacity])) as NDArray;
    this.background_stroke_rgbas = array(Array.from({ length: pointCount }, () => [0, 0, 0, this.background_stroke_opacity])) as NDArray;
    if (!propagate_colors) for (const submob of this.submobjects) (submob as VMobject).init_colors(false);
    return this;
  }

  set_fill(color?: ParsableManimColor, opacity?: number, family = true): this {
    if (color !== undefined) this.fill_color = new ManimColor(color);
    if (opacity !== undefined) this.fill_opacity = opacity;
    this.init_colors(family);
    if (family) for (const submob of this.submobjects) submob.set_fill(color, opacity, family);
    return this;
  }

  set_stroke(color?: ParsableManimColor, width?: number, opacity?: number, family = true): this {
    if (color !== undefined) this.stroke_color = new ManimColor(color);
    if (width !== undefined) this.stroke_width = width;
    if (opacity !== undefined) this.stroke_opacity = opacity;
    this.init_colors(family);
    if (family) for (const submob of this.submobjects) submob.set_stroke(color, width, opacity, family);
    return this;
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
    const first = pointRows(this.fill_rgbas, 4)[0];
    return first ? new ManimColor(array(first.slice(0, 3)) as any) : this.fill_color;
  }

  get_stroke_color(): ManimColor {
    const first = pointRows(this.stroke_rgbas, 4)[0];
    return first ? new ManimColor(array(first.slice(0, 3)) as any) : this.stroke_color;
  }

  get_fill_opacity(): number {
    const values = pointRows(this.fill_rgbas, 4).map((row) => row[3] ?? 0);
    return values.length === 0 ? this.fill_opacity : values.every((value) => value === 0) ? 0 : values[0];
  }

  get_stroke_opacity(): number {
    const values = pointRows(this.stroke_rgbas, 4).map((row) => row[3] ?? 0);
    return values.length === 0 ? this.stroke_opacity : values[0];
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
    const opacities = pointRows(this.fill_rgbas, 4).map((row) => row[3] ?? 0);
    return opacities.every((value) => value === 0) ? this.get_stroke_color() : this.get_fill_color();
  }

  set_color(color: ParsableManimColor, family = true): this {
    this.set_fill(color, undefined, family);
    this.set_stroke(color, undefined, undefined, family);
    return this;
  }

  setColor(color: ParsableManimColor, family = true): this { return this.set_color(color, family); }

  get_num_curves(): number {
    return Math.floor(this.get_num_points() / this.n_points_per_cubic_curve);
  }

  has_new_path_started(): boolean {
    return this.get_num_points() % this.n_points_per_cubic_curve === 1;
  }

  append_points(points: unknown): this {
    const current = rows(this.points, this.dim);
    const additions = rows(points, this.dim);
    this.points = asPoints([...current, ...additions], this.dim);
    return this;
  }

  add_line_to(point: unknown): this {
    const end = pointFrom(point, this.dim).toArray() as number[];
    const current = rows(this.points, this.dim);
    if (current.length === 0) {
      const error = new Error("Only call add_line_to with 1 or 2 points");
      error.name = "ValueError";
      throw error;
    }
    const start = current[current.length - 1];
    const handle1 = addPoints(start, scalePoint(subtractPoints(end, start, this.dim), 1 / 3, this.dim), this.dim);
    const handle2 = addPoints(end, scalePoint(subtractPoints(start, end, this.dim), 1 / 3, this.dim), this.dim);
    this.points = asPoints([...current, handle1, handle2, end], this.dim);
    return this;
  }

  set_points_as_corners(points: unknown): this {
    const corners = rows(points, this.dim);
    const result: number[][] = [];
    if (corners.length === 0) {
      this.points = array([]) as NDArray;
      return this;
    }
    result.push(corners[0]);
    for (let index = 1; index < corners.length; index += 1) {
      const start = corners[index - 1];
      const end = corners[index];
      result.push(addPoints(start, scalePoint(subtractPoints(end, start, this.dim), 1 / 3, this.dim), this.dim));
      result.push(addPoints(start, scalePoint(subtractPoints(end, start, this.dim), 2 / 3, this.dim), this.dim));
      result.push(end);
    }
    this.points = asPoints(result, this.dim);
    return this;
  }

  set_points_smoothly(points: unknown): this {
    return this.set_points_as_corners(points);
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

  get_anchors(): number[][] { return this.get_start_anchors(); }

  get_nth_curve_function(index: number): (t: number) => number[] {
    const start = index * this.n_points_per_cubic_curve;
    const points = rows(this.points, this.dim).slice(start, start + 4);
    return (t: number) => curveValue(points, t, this.dim);
  }

  get_nth_curve_function_with_length(index: number): [(t: number) => number[], number] {
    const curve = this.get_nth_curve_function(index);
    const points = rows(this.points, this.dim).slice(index * this.n_points_per_cubic_curve, index * this.n_points_per_cubic_curve + 4);
    return [curve, curveLength(points)];
  }

  get_curve_functions_with_lengths(): Array<[(t: number) => number[], number]> {
    return Array.from({ length: this.get_num_curves() }, (_, index) => this.get_nth_curve_function_with_length(index));
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
    throw new Error("Unable to determine a point from the requested proportion.");
  }

  proportion_from_point(point: readonly number[]): number {
    const curves = this.get_curve_functions_with_lengths();
    const total = curves.reduce((sum, [, length]) => sum + length, 0);
    if (total === 0) return 0;
    let current = 0;
    for (const [curve, length] of curves) {
      const candidate = curve(1);
      if (Math.hypot(...candidate.map((value, index) => value - (point[index] ?? 0))) <= Math.hypot(...curve(0).map((value, index) => value - (point[index] ?? 0)))) return (current + length) / total;
      current += length;
    }
    return 1;
  }

  reverse_direction(): this {
    this.points = asPoints(rows(this.points, this.dim).reverse(), this.dim);
    return this;
  }

  interpolate_color(mobject1: VMobject, mobject2: VMobject, alpha: number): this {
    const attributes = ["fill_rgbas", "stroke_rgbas", "background_stroke_rgbas", "stroke_width", "background_stroke_width", "sheen_direction", "sheen_factor"] as const;
    for (const attribute of attributes) {
      const first = (mobject1 as unknown as Record<string, unknown>)[attribute];
      const second = (mobject2 as unknown as Record<string, unknown>)[attribute];
      const result = typeof first === "number" || typeof second === "number" ? Number(first) * (1 - alpha) + Number(second) * alpha : interpolateArray(first, second, alpha);
      (this as unknown as Record<string, unknown>)[attribute] = alpha === 1 ? cloneAttribute(second) : result;
    }
    return this;
  }

  get_group_class(): typeof VGroup { return VGroup; }

  pointwise_become_partial(other: VMobject, a: number, b: number): this {
    const source = rows(other.points, this.dim);
    const result: number[][] = [];
    for (let index = 0; index < this.get_num_curves(); index += 1) {
      const start = index * this.n_points_per_cubic_curve;
      const curve = source.slice(start, start + 4);
      if (curve.length < 4) continue;
      const steps = 12;
      for (let step = 0; step <= steps; step += 1) {
        const t = a + (b - a) * (step / steps);
        result.push(curveValue(curve, t, this.dim));
      }
    }
    this.points = asPoints(result.length > 0 ? result : source, this.dim);
    return this;
  }

  get_point_mobject(center?: readonly number[]): VectorizedPoint {
    const point = new VectorizedPoint(center ?? this.get_center());
    point.color = this.color;
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

}

function cloneAttribute(value: unknown): unknown {
  if (isNDArray(value)) return value.copy();
  if (Array.isArray(value)) return value.map(cloneAttribute);
  return value;
}

function fullRows(value: unknown): number[][] {
  return nestedRows(value).map((row) => (Array.isArray(row) ? row : [row]).map(numeric));
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

export class VectorizedPoint extends VMobject {
  constructor(point: unknown = ORIGIN, options: VMobjectOptions = {}) {
    super(options);
    this.points = pointsFrom(point, this.dim);
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
}

export function vmobjectFrom(value: unknown, dim = 3): VMobject {
  const result = new VMobject({ dim });
  result.points = pointsFrom(value, dim);
  return result;
}

export class VDict extends VMobject {
  submob_dict = new Map<unknown, VMobject>();

  __delitem__(key: unknown): void {
    if (!this.submob_dict.has(key)) {
      const error = new Error(toPythonRepr(key));
      error.name = "KeyError";
      throw error;
    }
    this.submob_dict.delete(key);
  }
}

export class CurvesAsSubmobjects extends VGroup {
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
