import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { config } from "../../_config/utils.js";
import { isNDArray, pointFrom, pointRows } from "../../utils/arrays.js";
import { update_dict_recursively } from "../../utils/config_ops.js";
import { SMALL_BUFF, DL } from "../../constants.js";
import { BLUE_D } from "../../utils/color/index.js";
import type { ParsableManimColor } from "../../utils/color/core.js";

/** Minimal structural surface a family member must provide for
 * prepare_for_nonlinear_transform (mirrors VMobject.get_num_curves /
 * insert_n_curves). */
interface NonlinearTransformable {
  get_num_curves(): number;
  insert_n_curves(n: number): void;
}

function zipPairs<T, U>(first: readonly T[], second: readonly U[]): Array<[T, U]> {
  const length = Math.min(first.length, second.length);
  return Array.from({ length }, (_, index) => [first[index] as T, second[index] as U]);
}

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
  p2c(point: readonly number[]): number[] { return this.point_to_coords(point); }
  polar_to_point(radius: number, azimuth: number): NDArray {
    return this.coords_to_point(radius * Math.cos(azimuth), radius * Math.sin(azimuth));
  }
  point_to_polar(point: readonly number[]): [number, number] {
    const [x = 0, y = 0] = this.point_to_coords(point);
    return [Math.sqrt(x ** 2 + y ** 2), Math.atan2(y, x)];
  }
  pr2pt(radius: number, azimuth: number): NDArray {
    return this.polar_to_point(radius, azimuth);
  }
  pt2pr(point: readonly number[]): [number, number] {
    return this.point_to_polar(point);
  }
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

  __matmul__(coord: number[] | readonly number[] | NDArray | Mobject): NDArray {
    let target: unknown = coord;
    if (target instanceof Mobject) target = target.get_center();
    const items: number[] = isNDArray(target)
      ? [...((target as NDArray).toArray() as number[]).map(Number)]
      : Array.isArray(target)
        ? (target as unknown[]).map(Number)
        : [Number(target)];
    return this.coords_to_point(...items);
  }

  __rmatmul__(point: number[] | readonly number[] | NDArray): number[] {
    return this.point_to_coords(point as readonly number[]);
  }
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

function axisValue(value: number, range: number[], length: number): number {
  // Algebraic inverse of axisPosition above: undo the edge shift applied
  // when the whole range is strictly positive/negative, then invert the
  // linear map from [min, max] onto [-length/2, length/2].
  const min = Number(range[0]);
  const max = Number(range[1]);
  let centered = Number(value);
  if (min > 0) centered -= length / 2;
  else if (max < 0) centered += length / 2;
  const alpha = length === 0 ? 0 : (centered + length / 2) / length;
  return min + alpha * (max - min);
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
  point_to_coords(point: readonly number[]): number[] {
    // Mirrors Axes.point_to_coords for the linear mapping implemented by
    // coords_to_point above (Python resolves this through NumberLine axes;
    // the observable result is the algebraic inverse of the linear map).
    const values = [...point].map(Number);
    const xLength = this.x_length ?? Math.round(config.frame_width) - 2;
    const yLength = this.y_length ?? Math.round(config.frame_height) - 2;
    return [
      axisValue(values[0] ?? 0, this.x_range, xLength),
      axisValue(values[1] ?? 0, this.y_range, yLength),
    ];
  }

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

  prepare_for_nonlinear_transform(num_inserted_curves = 50): this {
    // NOTE: Python NumberPlane is a VGroup (Mobject), so
    // family_members_with_points() always exists there. The TS
    // CoordinateSystem hierarchy is not Mobject-based yet, so the insertion
    // pass only runs when the host provides family members.
    const host = this as unknown as {
      family_members_with_points?: () => NonlinearTransformable[];
    };
    for (const mob of host.family_members_with_points?.() ?? []) {
      const num_curves = mob.get_num_curves();
      if (num_inserted_curves > num_curves) mob.insert_n_curves(num_inserted_curves - num_curves);
    }
    return this;
  }

  prepareForNonlinearTransform(num_inserted_curves = 50): this {
    return this.prepare_for_nonlinear_transform(num_inserted_curves);
  }
}

export class Axes extends NumberPlane {
  get_axes(): Mobject[] { return []; }

  static _update_default_configs(
    default_configs: Array<Record<string, unknown>>,
    passed_configs: Array<Record<string, unknown>>,
  ): void {
    for (const [default_config, passed_config] of zipPairs(default_configs, passed_configs)) {
      if (passed_config !== null && passed_config !== undefined) {
        update_dict_recursively(default_config, passed_config);
      }
    }
  }

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

  *[Symbol.iterator](): Iterator<ShadedAxis> {
    // NOTE: mirrors Python `for axis in self` (ThreeDAxes is a VGroup of
    // axes); the TS port tracks the axes in `shaded_axes` until the Axes
    // hierarchy becomes Mobject-based.
    yield* this.shaded_axes;
  }

  _set_axis_shading(): void {
    const make_func = (axis: ShadedAxis): (() => [unknown, unknown]) => {
      const vect = this.light_source;
      return (): [unknown, unknown] => [
        axis.get_edge_center(vect.map((value) => -value)),
        axis.get_edge_center(vect),
      ];
    };

    for (const axis of this) {
      for (const submob of axis.family_members_with_points()) {
        submob.get_gradient_start_and_end_points = make_func(axis);
        submob.get_unit_normal = (_point: unknown): number[] => [1, 1, 1];
        submob.set_sheen(0.2);
      }
    }
  }
}

export type AzimuthUnits = "PI radians" | "TAU radians" | "degrees" | "gradians" | null;
export type AzimuthDirection = "CW" | "CCW";

export interface PolarPlaneOptions {
  radius_config?: Record<string, unknown> | null;
  background_line_style?: Record<string, unknown> | null;
  faded_line_style?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export class PolarPlane extends Axes {
  azimuth_units: AzimuthUnits;
  azimuth_direction: AzimuthDirection;
  radius_config: Record<string, unknown>;
  background_line_style: Record<string, unknown>;
  azimuth_step: number;
  faded_line_style: Record<string, unknown> | null;
  faded_line_ratio: number;
  make_smooth_after_applying_functions = true;
  azimuth_offset: number;
  azimuth_label_buff: number;
  azimuth_label_font_size: number;
  azimuth_compact_fraction: boolean;
  background_lines: unknown = null;
  faded_lines: unknown = null;

  constructor(
    radius_max: number | null = null,
    size: number | null = null,
    radius_step = 1,
    azimuth_step: number | null = null,
    azimuth_units: AzimuthUnits = "PI radians",
    azimuth_compact_fraction = true,
    azimuth_offset = 0,
    azimuth_direction: AzimuthDirection = "CCW",
    azimuth_label_buff = SMALL_BUFF,
    azimuth_label_font_size = 24,
    radius_config: Record<string, unknown> | null = null,
    background_line_style: Record<string, unknown> | null = null,
    faded_line_style: Record<string, unknown> | null = null,
    faded_line_ratio = 1,
    make_smooth_after_applying_functions = true,
    options: PolarPlaneOptions = {},
  ) {
    const resolved_radius_max = radius_max ?? config.frame_y_radius;
    super(
      [-resolved_radius_max, resolved_radius_max, radius_step],
      [-resolved_radius_max, resolved_radius_max, radius_step],
      size,
      size,
    );
    void options;
    if (
      azimuth_units !== "PI radians" &&
      azimuth_units !== "TAU radians" &&
      azimuth_units !== "degrees" &&
      azimuth_units !== "gradians" &&
      azimuth_units !== null
    ) {
      const error = new Error(
        "Invalid azimuth units. Expected one of: PI radians, TAU radians, degrees, gradians or None.",
      );
      error.name = "ValueError";
      throw error;
    }
    this.azimuth_units = azimuth_units;
    if (azimuth_direction !== "CW" && azimuth_direction !== "CCW") {
      const error = new Error("Invalid azimuth units. Expected one of: CW, CCW.");
      error.name = "ValueError";
      throw error;
    }
    this.azimuth_direction = azimuth_direction;
    this.radius_config = {
      stroke_width: 2,
      include_ticks: false,
      include_tip: false,
      line_to_number_buff: SMALL_BUFF,
      label_direction: [...DL],
      font_size: 24,
    };
    this.background_line_style = {
      stroke_color: BLUE_D as unknown as ParsableManimColor,
      stroke_width: 2,
      stroke_opacity: 1,
    };
    this.azimuth_step = azimuth_step ?? (
      {
        "PI radians": 20,
        "TAU radians": 20,
        degrees: 36,
        gradians: 40,
      } as Record<string, number>
    )[String(azimuth_units)] ?? 1;
    Axes._update_default_configs(
      [this.radius_config, this.background_line_style],
      [radius_config ?? {}, background_line_style ?? {}],
    );
    this.faded_line_style = faded_line_style;
    this.faded_line_ratio = faded_line_ratio;
    this.make_smooth_after_applying_functions = make_smooth_after_applying_functions;
    this.azimuth_offset = azimuth_offset;
    this.azimuth_label_buff = azimuth_label_buff;
    this.azimuth_label_font_size = azimuth_label_font_size;
    this.azimuth_compact_fraction = azimuth_compact_fraction;
    // NOTE: Python calls self._init_background_lines() here (Circle/Line
    // background construction over the Mobject family). The TS
    // CoordinateSystem hierarchy is not Mobject-based yet, so background
    // line construction is deferred until that integration lands;
    // background_lines/faded_lines stay null.
  }

  override prepare_for_nonlinear_transform(num_inserted_curves = 50): this {
    const host = this as unknown as {
      family_members_with_points?: () => NonlinearTransformable[];
    };
    for (const mob of host.family_members_with_points?.() ?? []) {
      const num_curves = mob.get_num_curves();
      if (num_inserted_curves > num_curves) mob.insert_n_curves(num_inserted_curves - num_curves);
    }
    return this;
  }
}

