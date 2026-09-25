import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import {
  DEFAULT_ARROW_TIP_LENGTH,
  DEFAULT_DASH_LENGTH,
  LEFT,
  MED_SMALL_BUFF,
  RendererType,
  RIGHT,
} from "../../constants.js";
import { pointRows, subtractPoints } from "../../utils/arrays.js";
import { angle_of_vector, normalize } from "../../utils/space_ops.js";
import { Mobject } from "../mobject.js";
import { DashedVMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { TipableVMobject, type TipableVMobjectOptions } from "./arc.js";
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
    options: LineOptions = {},
  ) {
    const { buff = 0, path_arc = 0, ...rest } = options;
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
    options: ArrowOptions = {},
  ) {
    const {
      stroke_width = 6,
      buff = MED_SMALL_BUFF,
      max_tip_length_to_length_ratio = 0.25,
      max_stroke_width_to_length_ratio = 5,
      tip_shape = null,
      ...rest
    } = options;
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
}
