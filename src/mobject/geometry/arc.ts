import { array, type NDArray } from "numpy-ts";
import { DEFAULT_ARROW_TIP_LENGTH, ORIGIN, OUT, RIGHT, TAU, UP } from "../../constants.js";
import { linspace, isNDArray, pointRows, subtractPoints } from "../../utils/arrays.js";
import {
  angle_of_vector,
  line_intersection,
  rotate_vector,
} from "../../utils/space_ops.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";

export interface TipableVMobjectOptions extends VMobjectOptions {
  tip_length?: number;
  normal_vector?: readonly number[] | NDArray;
  tip_style?: Record<string, unknown>;
}

export class TipableVMobject extends VMobject {
  tip_length: number;
  normal_vector: readonly number[] | NDArray;
  tip_style: Record<string, unknown>;

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

  get_default_tip_length(): number {
    return this.tip_length;
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
