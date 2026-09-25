import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { config } from "../../_config/utils.js";
import { isNDArray, pointFrom, pointRows } from "../../utils/arrays.js";

export class CoordinateSystem {
  dimension: number;
  x_range: number[];
  y_range: number[];
  x_length: number | null;
  y_length: number | null;
  num_sampled_graph_points_per_tick: number;

  constructor(x_range: number[] | null = null, y_range: number[] | null = null, x_length: number | null = null, y_length: number | null = null, dimension = 2) {
    this.dimension = dimension;
    this.x_range = x_range === null ? [Math.round(-config.frame_x_radius), Math.round(config.frame_x_radius), 1] : (x_range.length === 2 ? [...x_range, 1] : [...x_range]);
    this.y_range = y_range === null ? [Math.round(-config.frame_y_radius), Math.round(config.frame_y_radius), 1] : (y_range.length === 2 ? [...y_range, 1] : [...y_range]);
    this.x_length = x_length;
    this.y_length = y_length;
    this.num_sampled_graph_points_per_tick = 10;
  }

  coords_to_point(...coords: (number | number[] | NDArray)[]): NDArray {
    void coords;
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  c2p(...coords: (number | number[] | NDArray)[]): NDArray { return this.coords_to_point(...coords); }
  point_to_coords(_point: readonly number[]): number[] {
    const error = new Error("CoordinateSystem.point_to_coords must be implemented by a subclass.");
    error.name = "NotImplementedError";
    throw error;
  }
  static _origin_shift(_axis_range: readonly number[]): number {
    const error = new Error("CoordinateSystem._origin_shift must be implemented by a subclass.");
    error.name = "NotImplementedError";
    throw error;
  }
  get_axes(): Mobject[] { throw new Error("CoordinateSystem.get_axes must be implemented by a subclass."); }
  get_axis(index: number): Mobject { return this.get_axes()[index]; }
  get_origin(): NDArray { return this.coords_to_point(0, 0); }
}

function axisPosition(value: number, range: number[], length: number): number {
  const min = Number(range[0]);
  const max = Number(range[1]);
  const alpha = max === min ? 0 : (value - min) / (max - min);
  const centered = -length / 2 + alpha * length;
  if (min > 0) return centered + length / 2;
  if (max < 0) return centered - length / 2;
  return centered;
}

function coordinateRows(value: number | number[] | NDArray): number[][] {
  if (isNDArray(value)) return pointRows(value, 3);
  if (Array.isArray(value)) {
    const rows = value as unknown[];
    return rows.length > 0 && Array.isArray(rows[0]) ? (rows as number[][]).map((row) => row.map(Number)) : [(value as number[]).map(Number)];
  }
  return [[Number(value)]];
}

function axisValues(value: number | number[] | NDArray): number[] {
  if (isNDArray(value)) {
    const rows = pointRows(value, 3);
    return rows.length === 1 ? rows[0].slice(0, 1) : rows.map((row) => row[0] ?? 0);
  }
  if (Array.isArray(value)) {
    const rows = value as unknown[];
    return rows.length > 0 && Array.isArray(rows[0]) ? (rows as number[][]).map((row) => Number(row[0])) : (value as number[]).map(Number);
  }
  return [Number(value)];
}

export class NumberPlane extends CoordinateSystem {
  coords_to_point(...coords: (number | number[] | NDArray)[]): NDArray {
    if (coords.length === 0) return array([0, 0, 0]) as NDArray;
    const xLength = this.x_length ?? Math.round(config.frame_width) - 2;
    const yLength = this.y_length ?? Math.round(config.frame_height) - 2;
    const mapPoint = (x: number, y: number): number[] => [
      axisPosition(Number(x), this.x_range, xLength),
      axisPosition(Number(y), this.y_range, yLength),
      0,
    ];
    if (coords.length === 1) {
      const rows = coordinateRows(coords[0]);
      const points = rows.map((row) => mapPoint(row[0] ?? 0, row[1] ?? 0));
      if (points.length === 1) return array(points[0]) as NDArray;
      return array(points) as NDArray;
    }
    const xValues = axisValues(coords[0]);
    const yValues = axisValues(coords[1]);
    const count = Math.max(xValues.length, yValues.length);
    const points = Array.from({ length: count }, (_, index) => mapPoint(xValues[index] ?? xValues[xValues.length - 1] ?? 0, yValues[index] ?? yValues[yValues.length - 1] ?? 0));
    if (count === 1) return array(points[0]) as NDArray;
    // Ground truth (Python NumberPlane.coords_to_point docstring): passing
    // separate per-axis arrays is the *transposed* input format, so the output
    // is the transpose of the point list, e.g.
    //   coords_to_point([0,1,1],[1,1,0]) == coords_to_point([[0,1],[1,1],[1,0]]).T
    const transposed = points[0].map((_, col) => points.map((row) => row[col] ?? 0));
    return array(transposed) as NDArray;
  }
}

export class Axes extends NumberPlane {
  get_axes(): Mobject[] { return []; }

  static override _origin_shift(axis_range: readonly number[]): number {
    if ((axis_range[0] ?? 0) > 0) return axis_range[0] ?? 0;
    if ((axis_range[1] ?? 0) < 0) return axis_range[1] ?? 0;
    return 0;
  }
}

export interface AxisShadingSubmobject {
  get_gradient_start_and_end_points: () => [unknown, unknown];
  get_unit_normal: (point: unknown) => number[];
  set_sheen: (factor: number, ...rest: unknown[]) => unknown;
  family_members_with_points(): AxisShadingSubmobject[];
}

export interface ShadedAxis {
  get_edge_center: (direction: readonly number[]) => unknown;
  family_members_with_points(): AxisShadingSubmobject[];
}

export class ThreeDAxes extends Axes {
  light_source: number[];
  shaded_axes: ShadedAxis[];

  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
    light_source: readonly number[] = [-7, -9, 10],
    shaded_axes: ShadedAxis[] = [],
  ) {
    super(x_range, y_range, x_length, y_length);
    this.light_source = [...light_source];
    this.shaded_axes = shaded_axes;
  }

  _set_axis_shading(): void {
    const make_func = (axis: ShadedAxis): (() => [unknown, unknown]) => {
      const vect = this.light_source;
      const negated = vect.map((value) => -value);
      return (): [unknown, unknown] => [
        axis.get_edge_center(negated),
        axis.get_edge_center(vect),
      ];
    };

    for (const axis of this.shaded_axes) {
      for (const submob of axis.family_members_with_points()) {
        submob.get_gradient_start_and_end_points = make_func(axis);
        submob.get_unit_normal = (_point: unknown): number[] => [1, 1, 1];
        submob.set_sheen(0.2);
      }
    }
  }
}

