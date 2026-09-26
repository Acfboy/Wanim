import { array, type NDArray } from "numpy-ts";
import { DEFAULT_ARROW_TIP_LENGTH, DEFAULT_DOT_RADIUS, LEFT, ORIGIN, OUT, PI, RIGHT, SMALL_BUFF, TAU, UP } from "../../constants.js";
import { linspace, isNDArray, norm, pointAt, pointRows, subtractPoints } from "../../utils/arrays.js";
import { adjacent_pairs } from "../../utils/iterables.js";
import {
  angle_between_vectors,
  angle_of_vector,
  cartesian_to_spherical,
  line_intersection,
  perpendicular_bisector,
  rotate_vector,
} from "../../utils/space_ops.js";
import { BLACK, BLUE, RED, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import type { Mobject } from "../mobject.js";
import type { ArrowTip } from "./tips.js";
// NOTE: type-only import (erased at compile time): line.js imports Arc from
// this module at runtime, so a runtime import here would cycle.
import type { Line } from "./line.js";

/** Constructor type for arrow tip classes (first two params follow the
 * Python convention: fill_opacity, stroke_width; the rest via options). */
export type TipConstructor = new (
  fill_opacity?: number,
  stroke_width?: number,
  options?: Record<string, unknown>,
) => ArrowTip;

let default_tip_shape: TipConstructor | null = null;

/** Registers the default tip shape used by get_unpositioned_tip when no
 * explicit tip_shape is given. Called by the tips module on load (kept as
 * a registry rather than a static import to avoid a module cycle:
 * tips.js imports Circle from this module). */
export function set_default_tip_shape(ctor: TipConstructor): void {
  default_tip_shape = ctor;
}

/** Minimal structural type for the Line class used for gap-filling in
 * ArcPolygonFromArcs (avoids a module cycle: line.js imports Arc from this
 * module, mirroring the set_default_tip_shape registry above). */
export type LineConstructor = new (
  start: unknown,
  end: unknown,
  options?: Record<string, unknown>,
) => VMobject & {
  get_length(): number;
  insert_n_curves(n: number): VMobject;
};

let line_constructor: LineConstructor | null = null;

/** Registers the Line class used by ArcPolygonFromArcs. Called by the
 * geometry/line module on load. */
export function set_line_constructor(ctor: LineConstructor): void {
  line_constructor = ctor;
}

export interface TipableVMobjectOptions extends VMobjectOptions {
  tip_length?: number;
  normal_vector?: readonly number[] | NDArray;
  tip_style?: Record<string, unknown>;
}

export class TipableVMobject extends VMobject {
  tip_length: number;
  normal_vector: readonly number[] | NDArray;
  tip_style: Record<string, unknown>;
  tip?: ArrowTip;
  start_tip?: ArrowTip;
  _init_positioning_axis?: number[];

  constructor(options: TipableVMobjectOptions = {}) {
    const {
      tip_length = DEFAULT_ARROW_TIP_LENGTH,
      normal_vector = OUT,
      tip_style = {},
      ...rest
    } = options;
    super(rest);
    this.tip_length = tip_length;
    this.normal_vector = normal_vector;
    this.tip_style = tip_style ?? {};
  }

  add_tip(
    tip?: ArrowTip | null,
    tip_shape?: TipConstructor | null,
    tip_length?: number | null,
    tip_width?: number | null,
    at_start = false,
  ): this {
    let resolved = tip ?? null;
    if (resolved === null || resolved === undefined) {
      resolved = this.create_tip(tip_shape ?? null, tip_length ?? null, tip_width ?? null, at_start);
    } else {
      this.position_tip(resolved, at_start);
    }
    this.reset_endpoints_based_on_tip(resolved, at_start);
    this.assign_tip_attr(resolved, at_start);
    this.add(resolved);
    return this;
  }

  create_tip(
    tip_shape?: TipConstructor | null,
    tip_length?: number | null,
    tip_width?: number | null,
    at_start = false,
  ): ArrowTip {
    const tip = this.get_unpositioned_tip(tip_shape ?? null, tip_length ?? null, tip_width ?? null);
    this.position_tip(tip, at_start);
    return tip;
  }

  get_unpositioned_tip(
    tip_shape?: TipConstructor | null,
    tip_length?: number | null,
    tip_width?: number | null,
  ): ArrowTip {
    let shape = tip_shape ?? null;
    if (shape === null || shape === undefined) {
      if (default_tip_shape === null) {
        throw new Error("TODO: default tip shape (ArrowTriangleFilledTip) requires importing the geometry/tips module, not ported yet in this context");
      }
      shape = default_tip_shape;
    }
    const style: Record<string, unknown> = {};
    // NOTE: mirrors Python's special case: ArrowTriangleFilledTip gets a
    // width derived from the default tip length. The class-name check
    // stands in for Python's `tip_shape is ArrowTriangleFilledTip`.
    if (shape.name === "ArrowTriangleFilledTip") {
      style["width"] = tip_width ?? this.get_default_tip_length();
    }
    const resolved_length = tip_length ?? this.get_default_tip_length();
    const color = this.get_color();
    style["fill_color"] = color;
    style["stroke_color"] = color;
    Object.assign(style, this.tip_style);
    const tip = new shape(undefined, undefined, {
      length: resolved_length,
      ...style,
    });
    return tip;
  }

  position_tip(tip: ArrowTip, at_start = false): ArrowTip {
    // Last two control points, defining both the end, and the tangency direction
    let anchor: number[];
    let handle: number[];
    if (at_start) {
      anchor = this.get_start();
      handle = this.get_first_handle();
    } else {
      handle = this.get_last_handle();
      anchor = this.get_end();
    }
    const angles = cartesian_to_spherical(subtractPoints(handle, anchor, this.dim));
    // Rotates the tip along the azimuthal
    tip.rotate((angles[1] ?? 0) - PI - tip.tip_angle);
    if (!("_init_positioning_axis" in this) || this._init_positioning_axis === undefined) {
      // Obtains the perpendicular of the tip
      const axis = [
        Math.sin(angles[1] ?? 0),
        -Math.cos(angles[1] ?? 0),
        0,
      ];
      // Rotates the tip along the vertical wrt the axis
      tip.rotate(-(angles[2] ?? 0) + PI / 2, axis);
      this._init_positioning_axis = axis;
    }
    const tip_point = pointRows(tip.tip_point, this.dim)[0] ?? [0, 0, 0];
    tip.shift(subtractPoints(anchor, tip_point, this.dim));
    return tip;
  }

  reset_endpoints_based_on_tip(tip: ArrowTip, at_start: boolean): this {
    if (this.get_length() === 0) {
      // Zero length, put_start_and_end_on wouldn't work
      return this;
    }
    if (at_start) {
      this.put_start_and_end_on(pointRows(tip.base, this.dim)[0] ?? [0, 0, 0], this.get_end());
    } else {
      this.put_start_and_end_on(this.get_start(), pointRows(tip.base, this.dim)[0] ?? [0, 0, 0]);
    }
    return this;
  }

  assign_tip_attr(tip: ArrowTip, at_start: boolean): this {
    if (at_start) {
      this.start_tip = tip;
    } else {
      this.tip = tip;
    }
    return this;
  }

  has_tip(): boolean {
    return this.tip !== undefined && this.submobjects.includes(this.tip);
  }

  has_start_tip(): boolean {
    return this.start_tip !== undefined && this.submobjects.includes(this.start_tip);
  }

  pop_tips(): import("../mobject.js").Group {
    const [start, end] = this.get_start_and_end();
    const GroupClass = this.get_group_class();
    const result = new GroupClass();
    if (this.has_tip()) {
      result.add(this.tip as ArrowTip);
      this.remove(this.tip as ArrowTip);
    }
    if (this.has_start_tip()) {
      result.add(this.start_tip as ArrowTip);
      this.remove(this.start_tip as ArrowTip);
    }
    if (result.submobjects.length > 0) {
      this.put_start_and_end_on(start, end);
    }
    return result;
  }

  get_tips(): import("../mobject.js").Group {
    const GroupClass = this.get_group_class();
    const result = new GroupClass();
    if (this.tip !== undefined) {
      result.add(this.tip);
    }
    if (this.start_tip !== undefined) {
      result.add(this.start_tip);
    }
    return result;
  }

  get_tip(): VMobject {
    const tips = this.get_tips();
    if (tips.__len__() === 0) {
      throw new Error("tip not found");
    } else {
      const tip: VMobject = tips.submobjects[0] as VMobject;
      return tip;
    }
  }

  get_default_tip_length(): number {
    return this.tip_length;
  }

  get_first_handle(): number[] {
    return pointAt(this.points, 1, this.dim);
  }

  get_last_handle(): number[] {
    const all = pointRows(this.points, this.dim);
    return [...(all[all.length - 2] ?? [0, 0, 0])];
  }

  override get_end(): number[] {
    if (this.has_tip()) {
      return (this.tip as ArrowTip).get_start();
    } else {
      return super.get_end();
    }
  }

  override get_start(): number[] {
    if (this.has_start_tip()) {
      return (this.start_tip as ArrowTip).get_start();
    } else {
      return super.get_start();
    }
  }

  get_length(): number {
    const [start, end] = this.get_start_and_end();
    return norm(subtractPoints(start, end, this.dim));
  }
}

export interface ArcOptions extends TipableVMobjectOptions {}

export class Arc extends TipableVMobject {
  radius: number;
  num_components: number;
  arc_center: NDArray;
  start_angle: number;
  angle: number;
  _failed_to_get_center = false;

  constructor(
    radius: number | null = 1.0,
    start_angle = 0,
    angle = TAU / 4,
    num_components = 9,
    arc_center: readonly number[] | NDArray = [...ORIGIN],
    options: ArcOptions = {},
  ) {
    super(options);
    // NOTE: Python assigns these fields before super().__init__() so that
    // the generate_points() call inside the base constructor already sees
    // them. TS requires super() first; the base-constructor call is guarded
    // as a no-op below and points are (re)generated here instead.
    this.radius = radius ?? 1.0;
    this.num_components = num_components;
    this.arc_center = isNDArray(arc_center) ? arc_center.copy() : array([...arc_center]) as NDArray;
    this.start_angle = start_angle;
    this.angle = angle;
    this._failed_to_get_center = false;
    this.generate_points();
  }

  override generate_points(): this {
    if (this.num_components === undefined || this.arc_center === undefined) return this;
    this._set_pre_positioned_points();
    this.scale(this.radius, [...ORIGIN]);
    this.shift(this.arc_center);
    return this;
  }

  init_points(): this {
    this.set_points(
      Arc._create_quadratic_bezier_points(this.angle, this.start_angle, this.num_components),
    );
    this.scale(this.radius, [...ORIGIN]);
    this.shift(this.arc_center);
    return this;
  }

  static _create_quadratic_bezier_points(angle: number, start_angle = 0, n_components = 8): NDArray {
    const samples: number[][] = linspace(start_angle, start_angle + angle, 2 * n_components + 1).map(
      (a) => [Math.cos(a), Math.sin(a), 0],
    );
    const theta = angle / n_components;
    const scale = 1 / Math.cos(theta / 2);
    for (let index = 1; index < samples.length; index += 2) {
      const row = samples[index] as number[];
      samples[index] = row.map((value) => value * scale);
    }
    const points: number[][] = Array.from({ length: 3 * n_components }, () => [0, 0, 0]);
    for (let index = 0; index < n_components; index += 1) {
      points[index * 3] = [...(samples[index * 2] ?? [0, 0, 0])];
      points[index * 3 + 1] = [...(samples[index * 2 + 1] ?? [0, 0, 0])];
      points[index * 3 + 2] = [...(samples[index * 2 + 2] ?? [0, 0, 0])];
    }
    return array(points) as NDArray;
  }

  _set_pre_positioned_points(): void {
    const anchors: number[][] = linspace(
      this.start_angle,
      this.start_angle + this.angle,
      this.num_components,
    ).map((a) => [
      Math.cos(a) * (RIGHT[0] ?? 0) + Math.sin(a) * (UP[0] ?? 0),
      Math.cos(a) * (RIGHT[1] ?? 0) + Math.sin(a) * (UP[1] ?? 0),
      Math.cos(a) * (RIGHT[2] ?? 0) + Math.sin(a) * (UP[2] ?? 0),
    ]);
    // Figure out which control points will give the appropriate tangent
    // lines to the circle.
    const d_theta = this.angle / (this.num_components - 1.0);
    // Rotate all 90 degrees, via (x, y) -> (-y, x).
    const tangent_vectors = anchors.map((anchor) => [-(anchor[1] ?? 0), anchor[0] ?? 0, 0]);
    // Use tangent vectors to deduce anchors.
    const factor = (4 / 3) * Math.tan(d_theta / 4);
    const handles1 = anchors.slice(0, -1).map((anchor, index) => {
      const tangent = tangent_vectors[index] ?? [0, 0, 0];
      return anchor.map((value, component) => value + factor * (tangent[component] ?? 0));
    });
    const handles2 = anchors.slice(1).map((anchor, index) => {
      const tangent = tangent_vectors[index + 1] ?? [0, 0, 0];
      return anchor.map((value, component) => value - factor * (tangent[component] ?? 0));
    });
    this.set_anchors_and_handles(anchors.slice(0, -1), handles1, handles2, anchors.slice(1));
  }

  get_arc_center(warning = true): number[] {
    const [a1 = [0, 0, 0], h1 = [0, 0, 0], h2 = [0, 0, 0], a2 = [0, 0, 0]] = pointRows(this.points, this.dim).slice(0, 4);
    if (a1.every((value, index) => value === (a2[index] ?? 0))) {
      // For a1 and a2 to lie at the same point arc radius must be zero.
      // Thus arc_center will also lie at that point.
      return [...a1];
    }
    // Tangent vectors.
    const t1 = subtractPoints(h1, a1, this.dim);
    const t2 = subtractPoints(h2, a2, this.dim);
    // Normals.
    const n1 = (pointRows(rotate_vector(t1, TAU / 4), this.dim)[0] ?? [0, 0, 0]);
    const n2 = (pointRows(rotate_vector(t2, TAU / 4), this.dim)[0] ?? [0, 0, 0]);
    try {
      return line_intersection(
        [a1, a1.map((value, index) => value + (n1[index] ?? 0))],
        [a2, a2.map((value, index) => value + (n2[index] ?? 0))],
      );
    } catch {
      if (warning) console.warn("Can't find Arc center, using ORIGIN instead");
      this._failed_to_get_center = true;
      return [...ORIGIN];
    }
  }

  move_arc_center_to(point: readonly number[] | NDArray): this {
    const target = Array.isArray(point) ? [...point].map(Number) : (pointRows(point, this.dim)[0] ?? [0, 0, 0]);
    const center = this.get_arc_center();
    this.shift(target.map((value, index) => value - (center[index] ?? 0)));
    return this;
  }

  stop_angle(): number {
    const all = pointRows(this.points, this.dim);
    const last = all[all.length - 1] ?? [0, 0, 0];
    const raw = Number(angle_of_vector(subtractPoints(last, this.get_arc_center(), this.dim)));
    // NOTE: Python `%` is floored modulo; JS `%` is truncated. Use the
    // floored form so negative angles wrap the same way.
    return (((raw % TAU) + TAU) % TAU);
  }
}

export interface CircleOptions extends ArcOptions {
  radius?: number | null;
  color?: ParsableManimColor;
  arc_center?: readonly number[] | NDArray;
  num_components?: number;
}

export class Circle extends Arc {
  constructor(radius: number | null = null, options: CircleOptions = {}) {
    const {
      radius: radiusOption,
      arc_center = [...ORIGIN],
      num_components = 9,
      color = RED,
      ...rest
    } = options;
    super(radius ?? radiusOption ?? null, 0, TAU, num_components, arc_center, { color, ...rest });
  }

  surround(
    mobject: Mobject,
    dim_to_match = 0,
    stretch = false,
    buffer_factor = 1.2,
  ): this {
    // Ignores dim_to_match and stretch; result will always be a circle.
    this.replace(mobject, dim_to_match, stretch);
    this.width = Math.sqrt(mobject.width ** 2 + mobject.height ** 2);
    return this.scale(buffer_factor);
  }

  point_at_angle(angle: number): NDArray {
    let proportion = angle / TAU;
    proportion -= Math.floor(proportion);
    return this.point_from_proportion(proportion);
  }

  static from_three_points(
    p1: readonly number[] | NDArray,
    p2: readonly number[] | NDArray,
    p3: readonly number[] | NDArray,
    options: CircleOptions = {},
  ): Circle {
    const a = pointRows(p1, 3)[0] ?? [0, 0, 0];
    const b = pointRows(p2, 3)[0] ?? [0, 0, 0];
    const c = pointRows(p3, 3)[0] ?? [0, 0, 0];
    const center = line_intersection(
      perpendicular_bisector([a, b]),
      perpendicular_bisector([b, c]),
    );
    const radius = norm(subtractPoints(a, center, 3));
    return new Circle(radius, options).shift(center);
  }
}

export interface DotOptions extends CircleOptions {
  point?: readonly number[] | NDArray;
  fill_opacity?: number;
  stroke_width?: number;
}

export class Dot extends Circle {
  constructor(
    point: readonly number[] | NDArray = [...ORIGIN],
    radius = DEFAULT_DOT_RADIUS,
    stroke_width = 0,
    fill_opacity = 1.0,
    color: ParsableManimColor = WHITE,
    options: DotOptions = {},
  ) {
    const {
      point: pointOption,
      radius: radiusOption,
      stroke_width: strokeWidthOption,
      fill_opacity: fillOpacityOption,
      color: colorOption,
      ...rest
    } = options;
    super(radiusOption ?? radius, {
      arc_center: pointOption ?? point,
      stroke_width: strokeWidthOption ?? stroke_width,
      fill_opacity: fillOpacityOption ?? fill_opacity,
      color: colorOption ?? color,
      ...rest,
    });
  }
}

export interface AnnotationDotOptions extends DotOptions {
  stroke_color?: ParsableManimColor;
  fill_color?: ParsableManimColor;
}

export class AnnotationDot extends Dot {
  constructor(
    radius = DEFAULT_DOT_RADIUS * 1.3,
    stroke_width = 5,
    stroke_color: ParsableManimColor = WHITE,
    fill_color: ParsableManimColor = BLUE,
    options: AnnotationDotOptions = {},
  ) {
    const {
      radius: radiusOption,
      stroke_width: strokeWidthOption,
      stroke_color: strokeColorOption,
      fill_color: fillColorOption,
      ...rest
    } = options;
    super(
      [...ORIGIN],
      radiusOption ?? radius,
      strokeWidthOption ?? stroke_width,
      1.0,
      WHITE,
      {
        stroke_color: strokeColorOption ?? stroke_color,
        fill_color: fillColorOption ?? fill_color,
        ...rest,
      },
    );
  }
}

export interface LabeledDotOptions extends DotOptions {
  buff?: number;
}

export class LabeledDot extends Dot {
  constructor(
    label: string | Mobject,
    radius: number | null = null,
    buff = SMALL_BUFF,
    options: LabeledDotOptions = {},
  ) {
    const { buff: buffOption, ...rest } = options;
    const resolvedBuff = buffOption ?? buff;
    // NOTE: Python renders a `str` label as `MathTex(label, color=BLACK)`;
    // text/tex rendering is out of scope for the entire port, so only
    // pre-rendered mobject labels are supported here.
    let renderedLabel: Mobject;
    if (typeof label === "string") {
      void BLACK;
      throw new Error("TODO: text/tex not ported yet (LabeledDot string label needs MathTex)");
    } else {
      renderedLabel = label;
    }
    let resolvedRadius = radius;
    if (resolvedRadius === null || resolvedRadius === undefined) {
      resolvedRadius = resolvedBuff + norm([renderedLabel.width, renderedLabel.height]) / 2;
    }
    super([...ORIGIN], resolvedRadius, 0, 1.0, WHITE, rest);
    renderedLabel.move_to(this.get_center());
    this.add(renderedLabel);
  }
}

export interface AnnularSectorOptions extends ArcOptions {
  inner_radius?: number;
  outer_radius?: number;
  angle?: number;
  start_angle?: number;
  fill_opacity?: number;
  stroke_width?: number;
  color?: ParsableManimColor;
  arc_center?: readonly number[] | NDArray;
  num_components?: number;
}

export class AnnularSector extends Arc {
  inner_radius: number;
  outer_radius: number;

  constructor(
    inner_radius = 1,
    outer_radius = 2,
    angle = TAU / 4,
    start_angle = 0,
    fill_opacity = 1,
    stroke_width = 0,
    color: ParsableManimColor = WHITE,
    options: AnnularSectorOptions = {},
  ) {
    const {
      inner_radius: innerOption,
      outer_radius: outerOption,
      angle: angleOption,
      start_angle: startAngleOption,
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      color: colorOption,
      arc_center = [...ORIGIN],
      num_components = 9,
      ...rest
    } = options;
    // NOTE: Python assigns inner/outer_radius before super().__init__() so
    // that generate_points() sees them; TS requires super() first, so the
    // base-constructor call is guarded as a no-op and fields are assigned
    // plus points regenerated here.
    super(null, startAngleOption ?? start_angle, angleOption ?? angle, num_components, arc_center, {
      fill_opacity: fillOpacityOption ?? fill_opacity,
      stroke_width: strokeWidthOption ?? stroke_width,
      color: colorOption ?? color,
      ...rest,
    });
    this.inner_radius = innerOption ?? inner_radius;
    this.outer_radius = outerOption ?? outer_radius;
    this.generate_points();
  }

  override generate_points(): this {
    // NOTE: the base Mobject/Arc constructors invoke generate_points()
    // before this class assigns inner_radius/outer_radius (and the
    // Mobject instance proxy throws on unassigned fields, so a plain
    // `=== undefined` read is unsafe here). Treat that early call as a
    // no-op; the constructor regenerates points afterwards.
    if (!("inner_radius" in this) || !("outer_radius" in this)) return this;
    if (this.inner_radius === undefined || this.outer_radius === undefined) return this;
    const inner_arc = new Arc(this.inner_radius, this.start_angle, this.angle, this.num_components, this.arc_center);
    const outer_arc = new Arc(this.outer_radius, this.start_angle, this.angle, this.num_components, this.arc_center);
    outer_arc.reverse_points();
    this.append_points(inner_arc.points);
    this.add_line_to(pointRows(outer_arc.points, this.dim)[0] ?? [0, 0, 0]);
    this.append_points(outer_arc.points);
    this.add_line_to(pointRows(inner_arc.points, this.dim)[0] ?? [0, 0, 0]);
    return this;
  }

  override init_points(): this {
    this.generate_points();
    return this;
  }
}

export interface SectorOptions extends AnnularSectorOptions {
  radius?: number;
}

export class Sector extends AnnularSector {
  constructor(radius = 1, options: SectorOptions = {}) {
    const {
      radius: radiusOption,
      angle = TAU / 4,
      start_angle = 0,
      fill_opacity = 1,
      stroke_width = 0,
      color = WHITE,
      ...rest
    } = options;
    super(0, radiusOption ?? radius, angle, start_angle, fill_opacity, stroke_width, color, rest);
  }
}

export interface AnnulusOptions extends CircleOptions {
  inner_radius?: number;
  outer_radius?: number;
  fill_opacity?: number;
  stroke_width?: number;
  mark_paths_closed?: boolean;
}
export class Annulus extends Circle {
  inner_radius: number;
  outer_radius: number;
  mark_paths_closed: boolean;

  constructor(
    inner_radius = 1,
    outer_radius = 2,
    fill_opacity = 1,
    stroke_width = 0,
    color: ParsableManimColor = WHITE,
    mark_paths_closed = false,
    options: AnnulusOptions = {},
  ) {
    const {
      inner_radius: innerOption,
      outer_radius: outerOption,
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      color: colorOption,
      mark_paths_closed: markPathsClosedOption,
      ...rest
    } = options;
    super(null, {
      fill_opacity: fillOpacityOption ?? fill_opacity,
      stroke_width: strokeWidthOption ?? stroke_width,
      color: colorOption ?? color,
      ...rest,
    });
    // is this even used?
    this.mark_paths_closed = markPathsClosedOption ?? mark_paths_closed;
    this.inner_radius = innerOption ?? inner_radius;
    this.outer_radius = outerOption ?? outer_radius;
    this.generate_points();
  }

  override generate_points(): this {
    // NOTE: same early-construction guard as AnnularSector (see above).
    if (!("inner_radius" in this) || !("outer_radius" in this)) return this;
    if (this.inner_radius === undefined || this.outer_radius === undefined) return this;
    this.radius = this.outer_radius;
    const outer_circle = new Circle(this.outer_radius);
    const inner_circle = new Circle(this.inner_radius);
    inner_circle.reverse_points();
    this.append_points(outer_circle.points);
    this.append_points(inner_circle.points);
    this.shift(this.arc_center);
    return this;
  }

  override init_points(): this {
    this.generate_points();
    return this;
  }
}

export interface EllipseOptions extends CircleOptions {
  width?: number;
  height?: number;
}

export class Ellipse extends Circle {
  constructor(width = 2, height = 1, options: EllipseOptions = {}) {
    const { width: widthOption, height: heightOption, ...rest } = options;
    super(null, rest);
    this.stretch_to_fit_width(widthOption ?? width);
    this.stretch_to_fit_height(heightOption ?? height);
  }
}

export interface ArcBetweenPointsOptions extends ArcOptions {
  num_components?: number;
  arc_center?: readonly number[] | NDArray;
}

export class ArcBetweenPoints extends Arc {
  constructor(
    start: readonly number[] | NDArray,
    end: readonly number[] | NDArray,
    angle = TAU / 4,
    radius: number | null = null,
    options: ArcBetweenPointsOptions = {},
  ) {
    // NOTE: Python assigns self.radius / derives angle before
    // super().__init__(); TS requires super() first, so the radius/angle
    // resolution happens up front and is passed through to Arc instead.
    const { num_components = 9, arc_center = [...ORIGIN], ...rest } = options;
    let resolved_angle = angle;
    let resolved_radius = radius;
    if (resolved_radius !== null && resolved_radius !== undefined) {
      let working = resolved_radius;
      let sign: number;
      if (working < 0) {
        sign = -2;
        working *= -1;
      } else {
        sign = 2;
      }
      const start_row = pointRows(start, 3)[0] ?? [0, 0, 0];
      const end_row = pointRows(end, 3)[0] ?? [0, 0, 0];
      const halfdist = norm(subtractPoints(start_row, end_row, 3)) / 2;
      if (working < halfdist) {
        const error = new Error(
          `ArcBetweenPoints called with a radius that is
                                smaller than half the distance between the points.`,
        );
        error.name = "ValueError";
        throw error;
      }
      const arc_height = working - Math.sqrt(working ** 2 - halfdist ** 2);
      resolved_angle = Math.acos((working - arc_height) / working) * sign;
      resolved_radius = working;
    }
    super(resolved_radius, 0, resolved_angle, num_components, arc_center, rest);
    if (resolved_angle === 0) {
      this.set_points_as_corners([[...LEFT], [...RIGHT]]);
    }
    this.put_start_and_end_on(start, end);
    if (radius === null || radius === undefined) {
      const center = this.get_arc_center(false);
      if (!this._failed_to_get_center) {
        const start_row = pointRows(start, this.dim)[0] ?? [0, 0, 0];
        this.radius = norm(subtractPoints(start_row, center, this.dim));
      } else {
        this.radius = Infinity;
      }
    }
  }
}

export interface TangentialArcOptions extends ArcBetweenPointsOptions {
  angle?: number;
}

export class TangentialArc extends ArcBetweenPoints {
  line1: Line;
  line2: Line;

  constructor(
    line1: Line,
    line2: Line,
    radius: number,
    corner: readonly [number, number] | readonly number[] = [1, 1],
    options: TangentialArcOptions = {},
  ) {
    const { angle = TAU / 4, ...rest } = options;
    const intersection_point = line_intersection(
      [line1.get_start(), line1.get_end()],
      [line2.get_start(), line2.get_end()],
    );
    const [s1 = 1, s2 = 1] = [...corner];
    // Get unit vector for specified directions
    const line1_unit = line1.get_unit_vector();
    const line2_unit = line2.get_unit_vector();
    const unit_vector1 = line1_unit.map((value) => (s1 ?? 1) * value);
    const unit_vector2 = line2_unit.map((value) => (s2 ?? 1) * value);
    const corner_angle = angle_between_vectors(unit_vector1, unit_vector2);
    const tangent_point_distance = radius / Math.tan(corner_angle / 2);
    // tangent points
    const tangent_point1 = intersection_point.map(
      (value, index) => value + tangent_point_distance * (unit_vector1[index] ?? 0),
    );
    const tangent_point2 = intersection_point.map(
      (value, index) => value + tangent_point_distance * (unit_vector2[index] ?? 0),
    );
    const cross_product =
      (unit_vector1[0] ?? 0) * (unit_vector2[1] ?? 0) -
      (unit_vector1[1] ?? 0) * (unit_vector2[0] ?? 0);
    // Determine start and end points based on orientation
    let start_point: number[];
    let end_point: number[];
    if (cross_product < 0) {
      // Counterclockwise orientation - standard order
      start_point = tangent_point1;
      end_point = tangent_point2;
    } else {
      // Clockwise orientation - reverse the points
      start_point = tangent_point2;
      end_point = tangent_point1;
    }
    super(start_point, end_point, angle, radius, rest);
    this.line1 = line1;
    this.line2 = line2;
  }
}

export interface CurvedArrowOptions extends ArcBetweenPointsOptions {
  angle?: number;
  radius?: number | null;
  tip_shape?: TipConstructor | null;
}

export class CurvedArrow extends ArcBetweenPoints {
  constructor(
    start_point: readonly number[] | NDArray,
    end_point: readonly number[] | NDArray,
    options: CurvedArrowOptions = {},
  ) {
    // NOTE: Python pops tip_shape with default ArrowTriangleFilledTip; the
    // tips module registers that class as the default tip shape (see
    // set_default_tip_shape), so a null shape here resolves to it exactly
    // where Python would use it explicitly.
    const { tip_shape = null, angle = TAU / 4, radius = null, ...rest } = options;
    super(start_point, end_point, angle, radius, rest);
    this.add_tip(null, tip_shape);
  }
}

export interface CubicBezierOptions extends VMobjectOptions {}

export class CubicBezier extends VMobject {
  constructor(
    start_anchor: readonly number[] | NDArray,
    start_handle: readonly number[] | NDArray,
    end_handle: readonly number[] | NDArray,
    end_anchor: readonly number[] | NDArray,
    options: CubicBezierOptions = {},
  ) {
    super(options);
    this.add_cubic_bezier_curve(start_anchor, start_handle, end_handle, end_anchor);
  }
}

export interface ArcPolygonOptions extends VMobjectOptions {
  angle?: number;
  radius?: number | null;
  arc_config?: Array<Record<string, unknown>> | Record<string, unknown> | null;
}

export class ArcPolygon extends VMobject {
  arcs: ArcBetweenPoints[];

  constructor(...args: unknown[]) {
    let options: ArcPolygonOptions = {};
    let vertices: unknown[] = args;
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as ArcPolygonOptions;
      vertices = args.slice(0, -1);
    }
    const { angle = PI / 4, radius = null, arc_config = null, ...rest } = options;
    const n = vertices.length;
    const point_pairs: unknown[][] = vertices.map((_, k) => [
      vertices[k],
      vertices[(k + 1) % n],
    ]);
    let all_arc_configs: Array<Record<string, unknown>>;
    if (!arc_config) {
      if (radius) {
        all_arc_configs = point_pairs.map(() => ({ radius }) as Record<string, unknown>);
      } else {
        all_arc_configs = point_pairs.map(() => ({ angle }) as Record<string, unknown>);
      }
    } else if (!Array.isArray(arc_config)) {
      const shared = arc_config as Record<string, unknown>;
      all_arc_configs = point_pairs.map(() => shared);
    } else {
      if (arc_config.length !== n) {
        throw new Error("AssertionError");
      }
      all_arc_configs = arc_config;
    }
    const arcs = point_pairs.map((pair, index) => {
      const conf = all_arc_configs[index] ?? {};
      const { angle: conf_angle, radius: conf_radius, ...conf_rest } = conf;
      return new ArcBetweenPoints(
        pair[0] as readonly number[] | NDArray,
        pair[1] as readonly number[] | NDArray,
        (conf_angle as number | undefined) ?? TAU / 4,
        (conf_radius as number | null | undefined) ?? null,
        conf_rest as ArcBetweenPointsOptions,
      );
    });
    super(rest);
    // Adding the arcs like this makes ArcPolygon double as a VGroup.
    // Also makes changes to the ArcPolygon, such as scaling, affect
    // the arcs, so that their new values are usable.
    this.add(...arcs);
    for (const arc of arcs) {
      this.append_points(arc.points);
    }
    // This enables the use of ArcPolygon.arcs as a convenience
    // because ArcPolygon[0] returns itself, not the first Arc.
    this.arcs = arcs;
  }
}

export class ArcPolygonFromArcs extends VMobject {
  arcs: Arc[];

  constructor(...args: unknown[]) {
    let options: VMobjectOptions = {};
    let arc_list: unknown[] = args;
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as VMobjectOptions;
      arc_list = args.slice(0, -1);
    }
    const arcs = arc_list as Arc[];
    if (!arcs.every((m) => m instanceof Arc)) {
      const error = new Error(
        "All ArcPolygon submobjects must be of type Arc/ArcBetweenPoints",
      );
      error.name = "ValueError";
      throw error;
    }
    super(options);
    // Adding the arcs like this makes ArcPolygonFromArcs double as a VGroup.
    // Also makes changes to the ArcPolygonFromArcs, such as scaling, affect
    // the arcs, so that their new values are usable.
    this.add(...arcs);
    // This enables the use of ArcPolygonFromArcs.arcs as a convenience
    // because ArcPolygonFromArcs[0] returns itself, not the first Arc.
    this.arcs = [...arcs];
    if (line_constructor === null) {
      throw new Error("TODO: ArcPolygonFromArcs needs Line (import the geometry/line module), not loaded yet");
    }
    for (const pair of adjacent_pairs(arcs)) {
      const arc1 = pair[0] as Arc;
      const arc2 = pair[1] as Arc;
      this.append_points(arc1.points);
      const line = new line_constructor(arc1.get_end(), arc2.get_start());
      const len_ratio = line.get_length() / arc1.get_arc_length();
      if (Number.isNaN(len_ratio) || !Number.isFinite(len_ratio)) {
        continue;
      }
      // NOTE: Python int() truncates toward zero; the ratio is
      // non-negative here, matching Math.trunc.
      line.insert_n_curves(Math.trunc(arc1.get_num_curves() * len_ratio));
      this.append_points(line.points);
    }
  }
}
