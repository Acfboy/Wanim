import { array, type NDArray } from "numpy-ts";
import { DEFAULT_ARROW_TIP_LENGTH, DEFAULT_DOT_RADIUS, ORIGIN, OUT, PI, RIGHT, TAU, UP } from "../../constants.js";
import { linspace, isNDArray, norm, pointAt, pointRows, subtractPoints } from "../../utils/arrays.js";
import {
  angle_of_vector,
  cartesian_to_spherical,
  line_intersection,
  perpendicular_bisector,
  rotate_vector,
} from "../../utils/space_ops.js";
import { BLUE, RED, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import type { Mobject } from "../mobject.js";
import type { ArrowTip } from "./tips.js";

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
