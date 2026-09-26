import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import {
  DEFAULT_ARROW_TIP_LENGTH,
  DEFAULT_DASH_LENGTH,
  DEGREES,
  LEFT,
  MED_SMALL_BUFF,
  ORIGIN,
  RendererType,
  RIGHT,
  UP,
} from "../../constants.js";
import { norm, pointRows, subtractPoints } from "../../utils/arrays.js";
import { angle_of_vector, cross, line_intersection, normalize } from "../../utils/space_ops.js";
import { WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { Mobject } from "../mobject.js";
import { DashedVMobject, VGroup, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Arc, Dot, TipableVMobject, set_line_constructor, type TipableVMobjectOptions } from "./arc.js";
import { ArrowTriangleFilledTip } from "./tips.js";
import type { ArrowTip } from "./tips.js";
import type { TipConstructor } from "./arc.js";

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
    start: LineEndpoint = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    buffOrOptions: number | LineOptions = 0,
    pathArcOrOptions: number | LineOptions = 0,
    options: LineOptions = {},
  ) {
    // NOTE: mirrors Python `Line(start, end, buff=0, path_arc=0, **kwargs)`:
    // buff/path_arc may arrive positionally (as the transpiler will emit for
    // manim-style user code) or inside an options bag. Explicit option keys
    // win over positionals.
    let positionalBuff: number | undefined;
    let positionalPathArc: number | undefined;
    let merged: LineOptions = { ...options };
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
    this._set_start_and_end_attrs(start, end);
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
      throw new Error("TODO: Line with path_arc needs ArcBetweenPoints, not ported yet");
    } else {
      this.set_points_as_corners([
        pointRows(this.start, this.dim)[0] ?? new Array(this.dim).fill(0),
        pointRows(this.end, this.dim)[0] ?? new Array(this.dim).fill(0),
      ]);
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
    const vect = normalize(subtractPoints(end_rows, start_rows, this.dim));
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
    const [curr_start, curr_end] = this.get_start_and_end();
    if (curr_start.every((value, index) => value === (curr_end[index] ?? 0))) {
      // TODO, any problems with resetting these attrs?
      this.start = array([...pointRows(start, this.dim)[0] ?? [0, 0, 0]]) as NDArray;
      this.end = array([...pointRows(end, this.dim)[0] ?? [0, 0, 0]]) as NDArray;
      this.generate_points();
    }
    return super.put_start_and_end_on(start, end);
  }

  get_vector(): number[] {
    return subtractPoints(this.get_end(), this.get_start(), this.dim);
  }

  get_unit_vector(): number[] {
    return normalize(this.get_vector());
  }

  get_angle(): number {
    return angle_of_vector(this.get_vector()) as number;
  }

  get_projection(point: readonly number[] | NDArray): number[] {
    const start = this.get_start();
    const end = this.get_end();
    const unit_vect = normalize(subtractPoints(end, start, this.dim));
    const point_rows = pointRows(point, this.dim)[0] ?? [0, 0, 0];
    const dot = subtractPoints(point_rows, start, this.dim).reduce(
      (sum, value, index) => sum + value * (unit_vect[index] ?? 0),
      0,
    );
    return start.map((value, index) => value + dot * (unit_vect[index] ?? 0));
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

export interface DashedLineOptions extends LineOptions {
  dash_length?: number;
  dashed_ratio?: number;
}

export class DashedLine extends Line {
  dash_length: number;
  dashed_ratio: number;

  constructor(
    start: LineEndpoint = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    options: DashedLineOptions = {},
  ) {
    const { dash_length = DEFAULT_DASH_LENGTH, dashed_ratio = 0.5, ...rest } = options;
    super(start, end, rest);
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

  override get_start(): number[] {
    if (this.submobjects.length > 0) {
      return (this.submobjects[0] as Mobject).get_start();
    } else {
      return super.get_start();
    }
  }

  override get_end(): number[] {
    if (this.submobjects.length > 0) {
      return (this.submobjects[this.submobjects.length - 1] as Mobject).get_end();
    } else {
      return super.get_end();
    }
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

  constructor(
    start: LineEndpoint = [...LEFT],
    end: LineEndpoint = [...RIGHT],
    buffOrOptions: number | ArrowOptions = MED_SMALL_BUFF,
    options: ArrowOptions = {},
  ) {
    // NOTE: mirrors Python `Arrow(*args, stroke_width=6,
    // buff=MED_SMALL_BUFF, ...)`: buff may arrive positionally or inside an
    // options bag; an explicit option key wins over the positional.
    let positionalBuff: number | undefined;
    let merged: ArrowOptions = { ...options };
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
    super(start, end, { buff, stroke_width, ...rest });
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

  get_normal_vector(): number[] {
    const [p0 = [0, 0, 0], p1 = [0, 0, 0], p2 = [0, 0, 0]] = (this.tip as ArrowTip).get_start_anchors().slice(0, 3);
    return normalize(cross(
      subtractPoints(p2, p1, this.dim),
      subtractPoints(p1, p0, this.dim),
    ));
  }

  _set_stroke_width_from_length(): this {
    const max_ratio = this.max_stroke_width_to_length_ratio;
    const width = Math.min(this.initial_stroke_width, max_ratio * this.get_length());
    if (config.renderer === RendererType.OPENGL) {
      // NOTE: the TS VMobject.set_stroke has no OpenGL `recurse` parameter;
      // both branches set a non-propagating stroke width.
      this.set_stroke(undefined, width, undefined, false);
    } else {
      this.set_stroke(undefined, width, undefined, false);
    }
    return this;
  }

  override scale(
    factor: number,
    scaleTipsOrAboutPoint: boolean | readonly number[] | NDArray | null = false,
    aboutPointOrAboutEdge: readonly number[] | NDArray | null = null,
    aboutEdgeParam: readonly number[] | NDArray | null = null,
  ): this {
    // NOTE: mirrors Python `Arrow.scale(factor, scale_tips=False,
    // **kwargs)`: scaling keeps stroke width and tip size fixed unless
    // `scale_tips` is set. The second parameter doubles as `about_point`
    // (Mobject.scale calling convention) when it is not a boolean, so both
    // `scale(2)` / `scale(2, point)` and `scale(0, true, point)` work.
    let scaleTips = false;
    let aboutPoint: readonly number[] | NDArray | null | undefined;
    let resolvedAboutEdge: readonly number[] | NDArray | null | undefined;
    if (typeof scaleTipsOrAboutPoint === "boolean") {
      scaleTips = scaleTipsOrAboutPoint;
      aboutPoint = aboutPointOrAboutEdge ?? undefined;
      resolvedAboutEdge = aboutEdgeParam ?? undefined;
    } else {
      aboutPoint = scaleTipsOrAboutPoint ?? undefined;
      resolvedAboutEdge = aboutPointOrAboutEdge ?? undefined;
    }
    if (this.get_length() === 0) return this;
    if (scaleTips) {
      super.scale(factor, aboutPoint, resolvedAboutEdge);
      this._set_stroke_width_from_length();
      return this;
    }
    const hasTip = this.has_tip();
    const hasStartTip = this.has_start_tip();
    let oldTips: import("../mobject.js").Group | null = null;
    if (hasTip || hasStartTip) {
      oldTips = this.pop_tips();
    }
    super.scale(factor, aboutPoint, resolvedAboutEdge);
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
      inter = line_intersection(
        [line1.get_start(), line1.get_end()],
        [line2.get_start(), line2.get_end()],
      );
    } catch {
      const lines_are_in_xy_plane = [line1, line2].every((line) =>
        [line.get_start(), line.get_end()].every((point) => point[2] === 0),
      );
      const cross_z = ((): number => {
        const v1 = line1.get_vector();
        const v2 = line2.get_vector();
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
          (this.quadrant[0] === 1 ? line1.get_end() : line1.get_start()),
          inter,
          this.dim,
        ),
      );
      const dist_2 = norm(
        subtractPoints(
          (this.quadrant[1] === 1 ? line2.get_end() : line2.get_start()),
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

    const unit1 = line1.get_unit_vector();
    const unit2 = line2.get_unit_vector();
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
        const center = angle_mobject.get_center();
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
