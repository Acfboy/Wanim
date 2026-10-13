import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import {
  DEFAULT_ARROW_TIP_LENGTH,
  DEFAULT_DASH_LENGTH,
  DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
  DEGREES,
  LARGE_BUFF,
  LEFT,
  MED_SMALL_BUFF,
  ORIGIN,
  RendererType,
  RIGHT,
  UP,
} from "../../constants.js";
import { norm, pointRows, subtractPoints } from "../../utils/arrays.js";
import { angle_of_vector, cross, line_intersection, normalize, _normalize_flat } from "../../utils/space_ops.js";
import { WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { Mobject } from "../mobject.js";
import { DashedVMobject, VGroup, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Arc, ArcBetweenPoints, Dot, TipableVMobject, set_line_constructor, type TipableVMobjectOptions } from "./arc.js";
import { ArrowTriangleFilledTip } from "./tips.js";
import type { ArrowTip } from "./tips.js";
import type { TipConstructor } from "./arc.js";
import { set_grid_line_constructor, set_round_corners_constructors, isPlainOptionsBag } from "./polygram.js";

/** Factory used by {@link Vector.coordinate_label} to build a Matrix label.
 *  Registered by matrix.ts to avoid the line.ts -> matrix.ts ->
 *  shape_matchers.ts -> line.ts import cycle. */
export type MatrixFactory = (matrix: unknown[][], options?: Record<string, unknown>) => VMobject;
let matrixFactory: MatrixFactory | null = null;
export function setMatrixFactory(factory: MatrixFactory | null): void {
  matrixFactory = factory;
}

/** Round half to even, mirroring `np.round` (used by coordinate_label). */
function roundHalfEven(value: number): number {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export type LineEndpoint = Mobject | readonly number[] | NDArray;

export interface LineOptions extends TipableVMobjectOptions {
  buff?: number;
  path_arc?: number;
}

export class Line extends TipableVMobject {
  start!: NDArray;
  end!: NDArray;
  buff: number;
  path_arc: number;

  constructor(
    startOrOptions: LineEndpoint | LineOptions = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    buffOrOptions: number | LineOptions = 0,
    pathArcOrOptions: number | LineOptions = 0,
    options: LineOptions = {},
  ) {
    // NOTE: mirrors Python `Line(start, end, buff=0, path_arc=0, **kwargs)`.
    // A leading kwargs bag (`new Line({ start: LEFT, end: RIGHT })`) is also
    // accepted, as are buff/path_arc positionally or inside an options bag.
    let start: LineEndpoint = startOrOptions as LineEndpoint;
    let resolvedEnd: LineEndpoint = end;
    let merged: LineOptions = { ...options };
    if (isPlainOptionsBag(startOrOptions)) {
      const bag = { ...(startOrOptions as LineOptions & { start?: LineEndpoint; end?: LineEndpoint }) };
      if (bag.start !== undefined) start = bag.start;
      if (bag.end !== undefined) resolvedEnd = bag.end;
      delete bag.start;
      delete bag.end;
      merged = { ...bag, ...merged };
    }
    let positionalBuff: number | undefined;
    let positionalPathArc: number | undefined;
    if (typeof buffOrOptions === "number") positionalBuff = buffOrOptions;
    else merged = { ...buffOrOptions, ...merged };
    if (typeof pathArcOrOptions === "number") positionalPathArc = pathArcOrOptions;
    else merged = { ...pathArcOrOptions, ...merged };
    const { buff = positionalBuff ?? 0, path_arc = positionalPathArc ?? 0, ...rest } = merged;
    super({ dim: 3, ...rest });
    // NOTE: Python assigns these fields before super().__init__() so that
    // the generate_points() call inside the base constructor already sees
    // them. TS requires super() first; the base-constructor call is guarded
    // as a no-op below and points are (re)generated here instead.
    this.buff = buff;
    this.path_arc = path_arc;
    this._set_start_and_end_attrs(start, resolvedEnd);
    this.generate_points();
  }

  override generate_points(): this {
    if (!("start" in this) || !("end" in this)) return this;
    if (this.start === undefined || this.end === undefined) return this;
    this.set_points_by_ends(this.start, this.end, this.buff, this.path_arc);
    return this;
  }

  set_points_by_ends(
    start: LineEndpoint,
    end: LineEndpoint,
    buff = 0,
    path_arc = 0,
  ): this {
    this._set_start_and_end_attrs(start, end);
    if (path_arc) {
      // NOTE: mirrors Python exactly: the branch tests the `path_arc`
      // argument, but the arc angle comes from `self.path_arc`.
      const arc = new ArcBetweenPoints(this.start, this.end, this.path_arc);
      this.set_points(arc.points);
    } else {
      // A straight line is a single cubic segment; build its four control
      // points directly instead of round-tripping through
      // `set_points_as_corners` -> numpy `interpolate`.
      const startRow = pointRows(this.start, this.dim)[0] ?? new Array(this.dim).fill(0);
      const endRow = pointRows(this.end, this.dim)[0] ?? new Array(this.dim).fill(0);
      this.points = array(
        this._bezier_t_values.map((t) =>
          startRow.map((value, index) => value + t * ((endRow[index] ?? 0) - value)),
        ),
      ) as NDArray;
    }
    this._account_for_buff(buff);
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }

  _account_for_buff(buff: number): void {
    if (buff <= 0) return;
    const length = this.path_arc === 0 ? this.get_length() : this.get_arc_length();
    if (length < 2 * buff) return;
    const buff_proportion = buff / length;
    this.pointwise_become_partial(this, buff_proportion, 1 - buff_proportion);
  }

  _set_start_and_end_attrs(start: LineEndpoint, end: LineEndpoint): void {
    // If either start or end are Mobjects, this gives their centers
    const rough_start = this._pointify(start);
    const rough_end = this._pointify(end);
    const start_rows = pointRows(rough_start, this.dim)[0] ?? [0, 0, 0];
    const end_rows = pointRows(rough_end, this.dim)[0] ?? [0, 0, 0];
    const vect = _normalize_flat(subtractPoints(end_rows, start_rows, this.dim));
    // Now that we know the direction between them, we can find the
    // appropriate boundary point from start and end, if they're mobjects
    this.start = this._pointify(start, vect);
    this.end = this._pointify(end, vect.map((value) => -value));
  }

  _pointify(
    mob_or_point: LineEndpoint,
    direction?: readonly number[] | NDArray | null,
  ): NDArray {
    if (mob_or_point instanceof Mobject) {
      const mob = mob_or_point;
      if (direction === undefined || direction === null) {
        return array([...mob.get_center()]) as NDArray;
      } else {
        return array([...mob.get_boundary_point(direction)]) as NDArray;
      }
    }
    return array([...pointRows(mob_or_point, this.dim)[0] ?? [0, 0, 0]]) as NDArray;
  }

  set_path_arc(new_value: number): this {
    this.path_arc = new_value;
    this.init_points();
    return this;
  }

  override put_start_and_end_on(start: readonly number[] | NDArray, end: readonly number[] | NDArray): this {
    const curr_start = this._get_start_flat();
    const curr_end = this._get_end_flat();
    if (curr_start.every((value, index) => value === (curr_end[index] ?? 0))) {
      // TODO, any problems with resetting these attrs?
      this.start = array([...pointRows(start, this.dim)[0] ?? [0, 0, 0]]) as NDArray;
      this.end = array([...pointRows(end, this.dim)[0] ?? [0, 0, 0]]) as NDArray;
      this.generate_points();
      return super.put_start_and_end_on(start, end);
    }
    // Straight single-segment line: re-span it directly. Mobject's
    // scale+rotate+shift only maps the existing collinear control points onto
    // the target segment, which is exactly this interpolation, so this is
    // behaviour-equivalent and avoids three numpy transforms (the hot path
    // when an arrow resets its endpoints to the tip base). Only valid when no
    // attached tip makes `get_start`/`get_end` diverge from the point span
    // (e.g. the second tip of a DoubleArrow), otherwise Mobject's scale
    // factor would use the tipped endpoints.
    // NOTE: Arrows (and subclasses) recompute stroke width from length inside
    // `Arrow.scale`, which `Mobject.put_start_and_end_on` reaches via the
    // scale+rotate+shift round trip. Taking the direct re-span fast path here
    // would skip that side effect and diverge from Python (e.g. Arrow.scale(1)
    // during FadeIn's faded-copy creation shrinks 1.75 -> 1.3125 there).
    // Delegate to super for those so the width update happens.
    // NOTE: Mobject instances are Proxies that throw AttributeError on
    // missing members, so probe with `in` instead of reading the property.
    const hasArrowWidthRule =
      "_set_stroke_width_from_length" in (this as unknown as Record<string, unknown>);
    if (
      this.path_arc === 0
      && !this.has_tip()
      && !this.has_start_tip()
      && !hasArrowWidthRule
      && this.points.shape[0] === this.n_points_per_cubic_curve
    ) {
      const startRow = pointRows(start, this.dim)[0] ?? [0, 0, 0];
      const endRow = pointRows(end, this.dim)[0] ?? [0, 0, 0];
      this.points = array(
        this._bezier_t_values.map((t) =>
          startRow.map((value, index) => value + t * ((endRow[index] ?? 0) - value)),
        ),
      ) as NDArray;
      return this;
    }
    return super.put_start_and_end_on(start, end);
  }

  get_vector(): NDArray {
    const start = pointRows(this.get_start(), this.dim)[0] ?? [0, 0, 0];
    const end = pointRows(this.get_end(), this.dim)[0] ?? [0, 0, 0];
    return array(subtractPoints(end, start, this.dim)) as NDArray;
  }

  get_unit_vector(): NDArray {
    return normalize(this.get_vector());
  }

  get_angle(): number {
    return angle_of_vector(this.get_vector()) as number;
  }

  get_projection(point: readonly number[] | NDArray): NDArray {
    const start = pointRows(this.get_start(), this.dim)[0] ?? [0, 0, 0];
    const end = pointRows(this.get_end(), this.dim)[0] ?? [0, 0, 0];
    const unit_vect = _normalize_flat(subtractPoints(end, start, this.dim));
    const point_rows = pointRows(point, this.dim)[0] ?? [0, 0, 0];
    const dot = subtractPoints(point_rows, start, this.dim).reduce(
      (sum, value, index) => sum + value * (unit_vect[index] ?? 0),
      0,
    );
    return array(start.map((value, index) => value + dot * (unit_vect[index] ?? 0))) as NDArray;
  }

  get_slope(): number {
    return Math.tan(this.get_angle());
  }

  set_angle(angle: number, about_point?: readonly number[] | NDArray | null): this {
    if (about_point === undefined || about_point === null) {
      about_point = this.get_start();
    }
    this.rotate(angle - this.get_angle(), undefined, about_point);
    return this;
  }

  set_length(length: number): this {
    const scale_factor: number = length / this.get_length();
    return this.scale(scale_factor);
  }
}

// Registers Line for ArcPolygonFromArcs gap-filling (arc.js must not import
// line.js directly: this module imports Arc from there).
set_line_constructor(Line as unknown as Parameters<typeof set_line_constructor>[0]);

// Registers Line for Rectangle grid lines (polygram.js must not import
// line.js directly: this module reaches polygram.js via tips.js).
set_grid_line_constructor(Line as unknown as Parameters<typeof set_grid_line_constructor>[0]);

// Registers ArcBetweenPoints + Line for Polygram.round_corners (same cycle
// reason as above; the constructors are only invoked inside the method).
set_round_corners_constructors(
  ArcBetweenPoints as unknown as Parameters<typeof set_round_corners_constructors>[0],
  Line as unknown as Parameters<typeof set_round_corners_constructors>[1],
);

export interface TangentLineOptions extends LineOptions {
  d_alpha?: number;
  length?: number;
}

export class TangentLine extends Line {
  length: number;
  d_alpha: number;

  constructor(
    vmob: VMobject,
    alpha: number,
    length = 1,
    d_alpha = 1e-6,
    options: TangentLineOptions = {},
  ) {
    const {
      length: lengthOption,
      d_alpha: dAlphaOption,
      ...rest
    } = options;
    const resolved_length = lengthOption ?? length;
    const resolved_d_alpha = dAlphaOption ?? d_alpha;
    const da = resolved_d_alpha;
    // NOTE: mirrors np.clip(alpha - da, 0, 1) / np.clip(alpha + da, 0, 1).
    const a1 = Math.min(Math.max(alpha - da, 0), 1);
    const a2 = Math.min(Math.max(alpha + da, 0), 1);
    super(
      vmob.point_from_proportion(a1),
      vmob.point_from_proportion(a2),
      rest,
    );
    // NOTE: Python assigns self.length/self.d_alpha before super().__init__()
    // so generate_points() sees final attrs; TS requires super() first.
    // Line.generate_points() already ran on the tangent segment, and the
    // rescale below is the only post-super use, so late assignment is
    // behaviorally identical.
    this.length = resolved_length;
    this.d_alpha = resolved_d_alpha;
    this.scale(this.length / this.get_length());
  }
}

export interface DashedLineOptions extends LineOptions {
  dash_length?: number;
  dashed_ratio?: number;
}

export class DashedLine extends Line {
  dash_length: number;
  dashed_ratio: number;

  constructor(
    startOrOptions: LineEndpoint | DashedLineOptions = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    options: DashedLineOptions = {},
  ) {
    // Python: DashedLine(*args, dash_length=0.05, dashed_ratio=0.5, **kwargs).
    let start: LineEndpoint = startOrOptions as LineEndpoint;
    let resolvedEnd: LineEndpoint = end;
    let bag: DashedLineOptions = { ...options };
    if (isPlainOptionsBag(startOrOptions)) {
      const raw = { ...(startOrOptions as DashedLineOptions & { start?: LineEndpoint; end?: LineEndpoint }) };
      if (raw.start !== undefined) start = raw.start;
      if (raw.end !== undefined) resolvedEnd = raw.end;
      delete raw.start;
      delete raw.end;
      bag = { ...raw, ...bag };
    }
    const { dash_length = DEFAULT_DASH_LENGTH, dashed_ratio = 0.5, ...rest } = bag;
    super(start, resolvedEnd, rest);
    this.dash_length = dash_length;
    this.dashed_ratio = dashed_ratio;
    const dashes = new DashedVMobject(
      this,
      this._calculate_num_dashes(),
      dashed_ratio,
    );
    this.clear_points();
    this.add(...dashes);
  }

  _calculate_num_dashes(): number {
    // Minimum number of dashes has to be 2
    return Math.max(
      2,
      Math.ceil((this.get_length() / this.dash_length) * this.dashed_ratio),
    );
  }

  override get_start(): NDArray {
    return array(this._get_start_flat()) as NDArray;
  }

  override _get_start_flat(): number[] {
    if (this.submobjects.length > 0) {
      return (this.submobjects[0] as Mobject)._get_start_flat();
    }
    return super._get_start_flat();
  }

  override get_end(): NDArray {
    return array(this._get_end_flat()) as NDArray;
  }

  override _get_end_flat(): number[] {
    if (this.submobjects.length > 0) {
      return (this.submobjects[this.submobjects.length - 1] as Mobject)._get_end_flat();
    }
    return super._get_end_flat();
  }

  get_first_handle(): number[] {
    // Type inference of extracting an element from a list, is not
    // supported by numpy, see this numpy issue
    // https://github.com/numpy/numpy/issues/16544
    return pointRows((this.submobjects[0] as VMobject).points, this.dim)[1] ?? [0, 0, 0];
  }

  get_last_handle(): number[] {
    // Type inference of extracting an element from a list, is not
    // supported by numpy, see this numpy issue
    // https://github.com/numpy/numpy/issues/16544
    const rows = pointRows((this.submobjects[this.submobjects.length - 1] as VMobject).points, this.dim);
    return rows[2] ?? [0, 0, 0];
  }
}

export interface ArrowOptions extends LineOptions {
  stroke_width?: number;
  max_tip_length_to_length_ratio?: number;
  max_stroke_width_to_length_ratio?: number;
  tip_shape?: TipConstructor | null;
}

export class Arrow extends Line {
  max_tip_length_to_length_ratio: number;
  max_stroke_width_to_length_ratio: number;
  initial_stroke_width: number;
  /** Set while {@link put_start_and_end_on} repositions the tips, so the
   *  `reset_endpoints_based_on_tip` callback from `add_tip` goes straight to
   *  the body update instead of re-entering the tip handling. */
  private _repositioning_tips = false;

  constructor(
    startOrOptions: LineEndpoint | ArrowOptions = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    buffOrOptions: number | ArrowOptions = MED_SMALL_BUFF,
    options: ArrowOptions = {},
  ) {
    // NOTE: mirrors Python `Arrow(*args, stroke_width=6,
    // buff=MED_SMALL_BUFF, ...)`: args/options may arrive positionally or
    // inside a leading options bag; an explicit option key wins.
    let start: LineEndpoint = startOrOptions as LineEndpoint;
    let resolvedEnd: LineEndpoint = end;
    let merged: ArrowOptions = { ...options };
    if (isPlainOptionsBag(startOrOptions)) {
      const bag = { ...(startOrOptions as ArrowOptions & { start?: LineEndpoint; end?: LineEndpoint }) };
      if (bag.start !== undefined) start = bag.start;
      if (bag.end !== undefined) resolvedEnd = bag.end;
      delete bag.start;
      delete bag.end;
      merged = { ...bag, ...merged };
    }
    let positionalBuff: number | undefined;
    if (typeof buffOrOptions === "number") positionalBuff = buffOrOptions;
    else merged = { ...buffOrOptions, ...merged };
    const {
      stroke_width = 6,
      buff = positionalBuff ?? MED_SMALL_BUFF,
      max_tip_length_to_length_ratio = 0.25,
      max_stroke_width_to_length_ratio = 5,
      tip_shape = null,
      ...rest
    } = merged;
    // NOTE: Python assigns the max_* fields before super().__init__();
    // TS requires super() first — they are assigned here instead, before
    // any method that reads them runs.
    super(start, resolvedEnd, { buff, stroke_width, ...rest });
    this.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio;
    this.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio;
    const resolved_tip_shape = tip_shape ?? ArrowTriangleFilledTip;
    // TODO, should this be affected when Arrow.set_stroke is called?
    this.initial_stroke_width = this.stroke_width;
    this.add_tip(null, resolved_tip_shape);
    this._set_stroke_width_from_length();
  }

  get_default_tip_length(): number {
    const max_ratio = this.max_tip_length_to_length_ratio;
    return Math.min(this.tip_length, max_ratio * this.get_length());
  }

  get_normal_vector(): NDArray {
    const [p0 = [0, 0, 0], p1 = [0, 0, 0], p2 = [0, 0, 0]] = pointRows((this.tip as ArrowTip).get_start_anchors(), this.dim).slice(0, 3);
    return normalize(cross(
      subtractPoints(p2, p1, this.dim),
      subtractPoints(p1, p0, this.dim),
    ));
  }

  reset_normal_vector(): this {
    this.normal_vector = this.get_normal_vector();
    return this;
  }

  override put_start_and_end_on(start: readonly number[] | NDArray, end: readonly number[] | NDArray): this {
    // Fast path: move the straight body directly and reposition the existing
    // tips, instead of Mobject's scale+rotate+shift round trip (which pops and
    // re-creates/re-adds tips and re-applies stroke widths several times per
    // call).  This is the per-frame hot path for live vector fields.
    if (
      this._repositioning_tips
      || this.path_arc !== 0
      || (!this.has_tip() && !this.has_start_tip())
    ) {
      return super.put_start_and_end_on(start, end);
    }
    const tip = this.has_tip() ? this.tip : undefined;
    const startTip = this.has_start_tip() ? this.start_tip : undefined;
    if (tip !== undefined) this.remove(tip);
    if (startTip !== undefined) this.remove(startTip);
    super.put_start_and_end_on(start, end);
    this._set_stroke_width_from_length();
    this._repositioning_tips = true;
    try {
      if (tip !== undefined) this.add_tip(tip, null, null, null, false);
      if (startTip !== undefined) this.add_tip(startTip, null, null, null, true);
    } finally {
      this._repositioning_tips = false;
    }
    return this;
  }

  _set_stroke_width_from_length(): this {
    const max_ratio = this.max_stroke_width_to_length_ratio;
    const width = Math.min(this.initial_stroke_width, max_ratio * this.get_length());
    if (config.renderer === RendererType.OPENGL) {
      // NOTE: the TS VMobject.set_stroke has no OpenGL `recurse` parameter;
      // both branches set a non-propagating stroke width (family=false).
      this.set_stroke(undefined, width, undefined, false, false);
    } else {
      this.set_stroke(undefined, width, undefined, false, false);
    }
    return this;
  }

  override scale(
    factor: number | readonly number[] | NDArray,
    scaleTipsOrAboutPoint: boolean | readonly number[] | NDArray | Record<string, unknown> | null = false,
    aboutPointOrAboutEdge: readonly number[] | NDArray | Record<string, unknown> | null = null,
    aboutEdgeParam: readonly number[] | NDArray | Record<string, unknown> | null = null,
  ): this {
    // NOTE: mirrors Python `Arrow.scale(factor, scale_tips=False,
    // **kwargs)` (`**kwargs` carries `about_point`/`about_edge`): scaling
    // keeps stroke width and tip size fixed unless `scale_tips` is set. The
    // second parameter doubles as `about_point` (Mobject.scale calling
    // convention) when it is not a boolean, so both `scale(2)` /
    // `scale(2, point)` and `scale(2, true, point)` work.
    //
    // NOTE (py2js kwargs bridge): Python callers spell the keywords
    // `scale_tips`/`about_point`/`about_edge`, but the TS positional slots
    // are `scaleTipsOrAboutPoint`/`aboutPointOrAboutEdge`/`aboutEdgeParam`,
    // so the generic `__argnames__` bridge cannot map them and the kwargs
    // bag (`{about_point: ORIGIN}`, e.g. via `animate.scale(2,
    // about_point=ORIGIN)`) would land verbatim in the `about_point` slot
    // and poison `Mobject.scale` with NaN ("non-finite point/vector value").
    // Unwrap any such bag here (tagged null-prototype py2js bags as well as
    // plain objects), and also tolerate the bridge-mapped hole shape
    // `scale(2, undefined, about_point)` produced once `__argnames__` below
    // is set to the Python spellings.
    const isBagLike = (value: unknown): value is Record<string, unknown> => {
      if (value === null || value === undefined || typeof value !== "object") return false;
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return false;
      if (typeof (value as { tolist?: unknown }).tolist === "function") return false;
      if (typeof (value as { toArray?: unknown }).toArray === "function") return false;
      if (value instanceof Mobject) return false;
      if (typeof (value as { get_num_points?: unknown }).get_num_points === "function") return false;
      return true;
    };
    const isPointLike = (value: unknown): boolean => {
      if (value === null || value === undefined) return false;
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return true;
      const obj = value as { toArray?: unknown; tolist?: unknown };
      return typeof obj.toArray === "function" || typeof obj.tolist === "function";
    };
    let resolvedFactor = factor;
    let scaleTips = false;
    let aboutPoint: readonly number[] | NDArray | null | undefined;
    let resolvedAboutEdge: readonly number[] | NDArray | null | undefined;
    const pullBag = (bag: Record<string, unknown>): void => {
      const record = bag as Record<string, unknown>;
      if (typeof record["scale_tips"] === "boolean") scaleTips = Boolean(record["scale_tips"]);
      else if (typeof record["scaleTips"] === "boolean") scaleTips = Boolean(record["scaleTips"]);
      if (aboutPoint === undefined) {
        if (record["about_point"] !== undefined) aboutPoint = record["about_point"] as never;
        else if (record["aboutPoint"] !== undefined) aboutPoint = record["aboutPoint"] as never;
      }
      if (resolvedAboutEdge === undefined) {
        if (record["about_edge"] !== undefined) resolvedAboutEdge = record["about_edge"] as never;
        else if (record["aboutEdge"] !== undefined) resolvedAboutEdge = record["aboutEdge"] as never;
      }
      if ((resolvedFactor === undefined || resolvedFactor === null) && record["factor"] !== undefined) {
        resolvedFactor = record["factor"] as never;
      } else if ((resolvedFactor === undefined || resolvedFactor === null) && record["scale_factor"] !== undefined) {
        resolvedFactor = record["scale_factor"] as never;
      }
    };
    for (const slot of [scaleTipsOrAboutPoint, aboutPointOrAboutEdge, aboutEdgeParam]) {
      if (isBagLike(slot)) pullBag(slot);
    }
    const secondIsBag = isBagLike(scaleTipsOrAboutPoint);
    const thirdIsBag = isBagLike(aboutPointOrAboutEdge);
    const fourthIsBag = isBagLike(aboutEdgeParam);
    if (!secondIsBag) {
      if (typeof scaleTipsOrAboutPoint === "boolean") {
        scaleTips = scaleTipsOrAboutPoint;
      } else if (scaleTipsOrAboutPoint !== undefined && scaleTipsOrAboutPoint !== null && (scaleTipsOrAboutPoint as unknown) !== false) {
        if (isPointLike(scaleTipsOrAboutPoint) && aboutPoint === undefined) {
          aboutPoint = scaleTipsOrAboutPoint as never;
        }
      }
    }
    if (!thirdIsBag && aboutPointOrAboutEdge !== undefined && aboutPointOrAboutEdge !== null) {
      const secondWasPoint = !secondIsBag && isPointLike(scaleTipsOrAboutPoint);
      if (secondWasPoint) {
        if (resolvedAboutEdge === undefined) resolvedAboutEdge = aboutPointOrAboutEdge as never;
      } else if (aboutPoint === undefined) {
        aboutPoint = aboutPointOrAboutEdge as never;
      } else if (resolvedAboutEdge === undefined) {
        resolvedAboutEdge = aboutPointOrAboutEdge as never;
      }
    }
    if (!fourthIsBag && aboutEdgeParam !== undefined && aboutEdgeParam !== null) {
      if (resolvedAboutEdge === undefined) resolvedAboutEdge = aboutEdgeParam as never;
    }
    if (this.get_length() === 0) return this;
    if (scaleTips) {
      super.scale(resolvedFactor, aboutPoint, resolvedAboutEdge);
      this._set_stroke_width_from_length();
      return this;
    }
    const hasTip = this.has_tip();
    const hasStartTip = this.has_start_tip();
    let oldTips: import("../mobject.js").Group | null = null;
    if (hasTip || hasStartTip) {
      oldTips = this.pop_tips();
    }
    super.scale(resolvedFactor, aboutPoint, resolvedAboutEdge);
    this._set_stroke_width_from_length();
    if (hasTip && oldTips !== null) {
      this.add_tip(oldTips.submobjects[0] as ArrowTip, null, null, null, false);
    }
    if (hasStartTip && oldTips !== null) {
      const startTip = (hasTip ? oldTips.submobjects[1] : oldTips.submobjects[0]) as ArrowTip;
      this.add_tip(startTip, null, null, null, true);
    }
    return this;
  }
}

export interface DoubleArrowOptions extends ArrowOptions {
  tip_shape_start?: TipConstructor | null;
  tip_shape_end?: TipConstructor | null;
}

export class DoubleArrow extends Arrow {
  constructor(
    startOrOptions: LineEndpoint | DoubleArrowOptions = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    buffOrOptions: number | DoubleArrowOptions = MED_SMALL_BUFF,
    options: DoubleArrowOptions = {},
  ) {
    // NOTE: mirrors Python `DoubleArrow.__init__(self, *args, **kwargs)`:
    // `tip_shape_end` (when present) is renamed to `tip_shape` for the end
    // tip, and `tip_shape_start` (default ArrowTriangleFilledTip) becomes
    // the start tip added after construction.
    let start: LineEndpoint = startOrOptions as LineEndpoint;
    let resolvedEnd: LineEndpoint = end;
    let merged: DoubleArrowOptions = { ...options };
    if (isPlainOptionsBag(startOrOptions)) {
      const bag = { ...(startOrOptions as DoubleArrowOptions & { start?: LineEndpoint; end?: LineEndpoint }) };
      if (bag.start !== undefined) start = bag.start;
      if (bag.end !== undefined) resolvedEnd = bag.end;
      delete bag.start;
      delete bag.end;
      merged = { ...bag, ...merged };
    }
    let positionalBuff: number | undefined;
    if (typeof buffOrOptions === "number") positionalBuff = buffOrOptions;
    else merged = { ...buffOrOptions, ...merged };
    const {
      tip_shape_start = ArrowTriangleFilledTip,
      tip_shape_end,
      ...rest
    } = merged;
    const forwarded: ArrowOptions = { ...rest };
    if (tip_shape_end !== undefined) forwarded.tip_shape = tip_shape_end;
    if (positionalBuff !== undefined) super(start, resolvedEnd, positionalBuff, forwarded);
    else super(start, resolvedEnd, forwarded);
    this.add_tip(null, tip_shape_start, null, null, true);
  }
}

export interface ElbowOptions extends VMobjectOptions {
  width?: number;
  angle?: number;
}

export class Elbow extends VMobject {
  angle: number;

  constructor(width = 0.2, angle = 0, options: ElbowOptions = {}) {
    const { width: widthOption, angle: angleOption, ...rest } = options;
    super(rest);
    // NOTE: Python assigns self.angle before super().__init__(); TS requires
    // super() first. Elbow defines no generate_points override, so the base
    // constructor's point generation needs no fields from this class.
    this.angle = angleOption ?? angle;
    this.set_points_as_corners([
      [...UP],
      [UP[0]! + RIGHT[0]!, UP[1]! + RIGHT[1]!, UP[2]! + RIGHT[2]!],
      [...RIGHT],
    ]);
    this.rescale_to_fit(widthOption ?? width, 0, false, [...ORIGIN]);
    this.rotate(this.angle, undefined, [...ORIGIN]);
  }
}

export type AngleQuadrant = [number, number] | readonly [number, number];

export interface AngleOptions extends VMobjectOptions {
  radius?: number | null;
  quadrant?: AngleQuadrant;
  other_angle?: boolean;
  dot?: boolean;
  dot_radius?: number | null;
  dot_distance?: number;
  dot_color?: ParsableManimColor;
  elbow?: boolean;
}

export class Angle extends VMobject {
  lines: [Line, Line];
  quadrant: AngleQuadrant;
  dot_distance: number;
  elbow: boolean;
  radius?: number;
  dot_radius?: number;
  // NOTE: mirrors Python, where `angle_value` is only assigned on the
  // non-elbow path (elbow RightAngles have no such attribute). `declare`
  // emits no runtime field so `"angle_value" in angle` is false until
  // assigned, and reads go through the Mobject proxy (AttributeError),
  // exactly like Python's missing attribute.
  declare angle_value: number;

  constructor(line1: Line, line2: Line, options: AngleOptions = {}) {
    const {
      radius = null,
      quadrant = [1, 1] as AngleQuadrant,
      other_angle = false,
      dot = false,
      dot_radius = null,
      dot_distance = 0.55,
      dot_color = WHITE,
      elbow = false,
      ...rest
    } = options;
    super(rest);
    this.lines = [line1, line2];
    this.quadrant = quadrant;
    this.dot_distance = dot_distance;
    this.elbow = elbow;
    let inter: number[];
    try {
      inter = pointRows(line_intersection(
        [line1.get_start(), line1.get_end()],
        [line2.get_start(), line2.get_end()],
      ), this.dim)[0] ?? [0, 0, 0];
    } catch {
      const lines_are_in_xy_plane = [line1, line2].every((line) =>
        [line._get_start_flat(), line._get_end_flat()].every((point) => point[2] === 0),
      );
      const cross_z = ((): number => {
        const v1 = pointRows(line1.get_vector(), 3)[0] ?? [0, 0, 0];
        const v2 = pointRows(line2.get_vector(), 3)[0] ?? [0, 0, 0];
        return (v1[0] ?? 0) * (v2[1] ?? 0) - (v1[1] ?? 0) * (v2[0] ?? 0);
      })();
      if (!(lines_are_in_xy_plane && cross_z === 0)) throw new Error(
        "The lines are parallel, there is no unique intersection point.",
      );
      // When the two lines are parallel or collinear there is no unique
      // intersection point. Rather than raising, Angle becomes an empty
      // Mobject (see issue #1930).
      this.angle_value = 0.0;
      return;
    }

    let resolved_radius: number;
    if (radius === null || radius === undefined) {
      const dist_1 = norm(
        subtractPoints(
          (this.quadrant[0] === 1 ? line1._get_end_flat() : line1._get_start_flat()),
          inter,
          this.dim,
        ),
      );
      const dist_2 = norm(
        subtractPoints(
          (this.quadrant[1] === 1 ? line2._get_end_flat() : line2._get_start_flat()),
          inter,
          this.dim,
        ),
      );
      if (Math.min(dist_1, dist_2) < 0.6) {
        resolved_radius = (2 / 3) * Math.min(dist_1, dist_2);
      } else {
        resolved_radius = 0.4;
      }
    } else {
      this.radius = radius;
      resolved_radius = radius;
    }

    const unit1 = pointRows(line1.get_unit_vector(), 3)[0] ?? [0, 0, 0];
    const unit2 = pointRows(line2.get_unit_vector(), 3)[0] ?? [0, 0, 0];
    const anchor_angle_1 = inter.map(
      (value, index) => value + (this.quadrant[0] ?? 0) * resolved_radius * (unit1[index] ?? 0),
    );
    const anchor_angle_2 = inter.map(
      (value, index) => value + (this.quadrant[1] ?? 0) * resolved_radius * (unit2[index] ?? 0),
    );

    if (elbow) {
      const anchor_middle = inter.map(
        (value, index) =>
          value +
          (this.quadrant[0] ?? 0) * resolved_radius * (unit1[index] ?? 0) +
          (this.quadrant[1] ?? 0) * resolved_radius * (unit2[index] ?? 0),
      );
      const angle_mobject: VMobject = new Elbow(0.2, 0, rest);
      angle_mobject.set_points_as_corners([anchor_angle_1, anchor_middle, anchor_angle_2]);
      this.set_points(angle_mobject.points);
    } else {
      const angle_1 = Number(angle_of_vector(subtractPoints(anchor_angle_1, inter, this.dim)));
      const angle_2 = Number(angle_of_vector(subtractPoints(anchor_angle_2, inter, this.dim)));
      let start_angle: number;
      let angle_fin: number;
      if (!other_angle) {
        start_angle = angle_1;
        if (angle_2 > angle_1) angle_fin = angle_2 - angle_1;
        else angle_fin = 2 * Math.PI - (angle_1 - angle_2);
      } else {
        start_angle = angle_1;
        if (angle_2 < angle_1) angle_fin = -angle_1 + angle_2;
        else angle_fin = -2 * Math.PI + (angle_2 - angle_1);
      }
      this.angle_value = angle_fin;
      const angle_mobject = new Arc(resolved_radius, start_angle, this.angle_value, 9, inter, rest);
      if (dot) {
        let resolved_dot_radius: number;
        if (dot_radius === null || dot_radius === undefined) {
          resolved_dot_radius = resolved_radius / 10;
        } else {
          this.dot_radius = dot_radius;
          resolved_dot_radius = dot_radius;
        }
        const right_dot = new Dot([...ORIGIN], resolved_dot_radius, 0, 1.0, dot_color);
        const center = pointRows(angle_mobject.get_center(), this.dim)[0] ?? [0, 0, 0];
        const diff = subtractPoints(center, inter, this.dim);
        const diff_norm = norm(diff);
        const dot_anchor = inter.map(
          (value, index) => value + ((diff[index] ?? 0) / diff_norm) * resolved_radius * dot_distance,
        );
        right_dot.move_to(dot_anchor);
        this.add(right_dot);
      }
      this.set_points(angle_mobject.points);
    }
  }

  get_lines(): VGroup {
    return new VGroup(...this.lines);
  }

  get_value(degrees = false): number {
    return degrees ? this.angle_value / DEGREES : this.angle_value;
  }

  static from_three_points(
    A: readonly number[] | NDArray,
    B: readonly number[] | NDArray,
    C: readonly number[] | NDArray,
    options: AngleOptions = {},
  ): Angle {
    return new Angle(new Line(B, A), new Line(B, C), options);
  }
}

export interface RightAngleOptions extends AngleOptions {
  length?: number | null;
}

export class RightAngle extends Angle {
  constructor(line1: Line, line2: Line, length: number | null = null, options: RightAngleOptions = {}) {
    const { length: lengthOption, ...rest } = options;
    super(line1, line2, { radius: lengthOption ?? length, elbow: true, ...rest });
  }
}

export interface VectorOptions extends ArrowOptions {
  buff?: number;
}

export class Vector extends Arrow {
  buff: number;

  constructor(
    direction: readonly number[] | NDArray | number[] = [...RIGHT],
    buffOrOptions: number | VectorOptions = 0,
    options: VectorOptions = {},
  ) {
    // NOTE: mirrors Python `Vector(direction=RIGHT, buff=0, **kwargs)` which
    // extends 2-D directions to 3-D (`np.hstack([direction, 0])`) and calls
    // `Arrow(ORIGIN, direction, buff=buff, **kwargs)`.
    let positionalBuff: number | undefined;
    let merged: VectorOptions = { ...options };
    if (typeof buffOrOptions === "number") positionalBuff = buffOrOptions;
    else merged = { ...(buffOrOptions as VectorOptions), ...merged };
    const { buff: buffOption, ...rest } = merged;
    const buff = buffOption ?? positionalBuff ?? 0;
    let dir: readonly number[] | NDArray | number[] = direction;
    if (!Array.isArray(dir) && typeof (dir as NDArray).toArray === "function") {
      const raw = (dir as NDArray).toArray() as unknown;
      dir = (Array.isArray(raw) ? raw : [raw]).map(Number);
    }
    const dirArray = [...(dir as number[])];
    if (dirArray.length === 2) dirArray.push(0);
    super([...ORIGIN], dirArray, { buff, ...rest });
    this.buff = buff;
  }

  coordinate_label(
    integer_labels = true,
    n_dim = 2,
    color: ParsableManimColor | null = null,
    options: Record<string, unknown> = {},
  ): VMobject {
    // NOTE: mirrors Python `Vector.coordinate_label`: build a column Matrix
    // from the (optionally integer-rounded) end point, scale it, then shift it
    // next to the tip on the side the vector points to.
    if (matrixFactory === null) {
      throw new Error(
        "Vector.coordinate_label requires manim.mobject.matrix (Matrix) to be loaded",
      );
    }
    const end = pointRows(this.get_end(), 3)[0] ?? [0, 0, 0];
    let vect = [...end];
    if (integer_labels) {
      // np.round (half-to-even) followed by astype(int) (truncation toward 0).
      vect = vect.map((value) => Math.trunc(roundHalfEven(value)));
    }
    vect = vect.slice(0, n_dim);
    const column = vect.map((value) => [value]);
    const label = matrixFactory(column, { ...options });
    label.scale(LARGE_BUFF - 0.2);

    const shiftDir = [...end];
    if ((shiftDir[0] ?? 0) >= 0) {
      // Pointing right.
      const left = pointRows(label.get_left(), this.dim)[0] ?? [0, 0, 0];
      shiftDir[0] -= (left[0] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (LEFT[0] as number);
      shiftDir[1] -= (left[1] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (LEFT[1] as number);
      shiftDir[2] -= (left[2] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (LEFT[2] as number);
    } else {
      // Pointing left.
      const right = pointRows(label.get_right(), this.dim)[0] ?? [0, 0, 0];
      shiftDir[0] -= (right[0] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (RIGHT[0] as number);
      shiftDir[1] -= (right[1] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (RIGHT[1] as number);
      shiftDir[2] -= (right[2] ?? 0) + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * (RIGHT[2] as number);
    }
    label.shift(shiftDir);
    if (color !== null && color !== undefined) label.set_color(color);
    return label;
  }
}

// Explicit Python parameter names for `Arrow.scale`, surviving bundling.
// Python spells them `scale(factor, scale_tips=False, *, about_point=None,
// about_edge=None)`; the TS positional slots are
// `scaleTipsOrAboutPoint`/`aboutPointOrAboutEdge`/`aboutEdgeParam`, so without
// this the py2js kwargs bridge cannot map `about_point=` (e.g.
// `animate.scale(2, about_point=ORIGIN)`) and the kwargs bag lands verbatim in
// the about-point slot, poisoning `Mobject.scale` with NaN. `runtime_core`
// prefers this metadata over parsed names (see also the `__argnames__`
// propagation in `wrapPrototypeMethods`).
(Arrow.prototype.scale as unknown as { __argnames__?: string[] }).__argnames__ = [
  "factor",
  "scale_tips",
  "about_point",
  "about_edge",
];
