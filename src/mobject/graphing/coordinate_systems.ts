import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { config } from "../../_config/utils.js";
import { arange, isNDArray, linspace, numeric, pointFrom, pointRows, type ComplexPoint } from "../../utils/arrays.js";
import { update_dict_recursively } from "../../utils/config_ops.js";
import { SMALL_BUFF, DL, RIGHT, TAU, RendererType } from "../../constants.js";
import { BLUE_D } from "../../utils/color/index.js";
import { BLUE, GREEN } from "../../utils/color/core.js";
import { interpolate_color, ManimColor, type ParsableManimColor } from "../../utils/color/core.js";
import { binary_search } from "../../utils/simple_functions.js";
import { angle_of_vector } from "../../utils/space_ops.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { Polygon } from "../geometry/polygram.js";
import { ParametricFunction, type ParametricFunctionOptions } from "./functions.js";
import { LinearBase, type _ScaleBase } from "./scale.js";

/** Minimal structural surface a family member must provide for
 * prepare_for_nonlinear_transform (mirrors VMobject.get_num_curves /
 * insert_n_curves). */
interface NonlinearTransformable {
  get_num_curves(): number;
  insert_n_curves(n: number): void;
}

/** Structural surface an axis mobject must provide for
 * CoordinateSystem.add_coordinates (mirrors NumberLine.add_labels /
 * add_numbers, NumberLine.scaling, NumberLine.get_tick_range and the
 * labels/numbers attributes those calls populate). */
export interface CoordinateAxisLike extends Mobject {
  scaling: _ScaleBase;
  get_tick_range(): NDArray;
  add_labels(
    dict_values: Map<number, unknown> | Record<string, unknown>,
    direction?: readonly number[] | NDArray | null,
    buff?: number | null,
    font_size?: number | null,
    label_constructor?: unknown,
  ): unknown;
  add_numbers(
    x_values?: Iterable<number> | null,
    excluding?: Iterable<number> | null,
    font_size?: number | null,
    label_constructor?: unknown,
    number_config?: Record<string, unknown>,
  ): unknown;
  labels: VGroup | null;
  numbers: VGroup | null;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  if (Array.isArray(value)) return false;
  if (value instanceof Mobject) return false;
  if (value instanceof Map) return false;
  if (isNDArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function zipPairs<T, U>(first: readonly T[], second: readonly U[]): Array<[T, U]> {
  const length = Math.min(first.length, second.length);
  return Array.from({ length }, (_, index) => [first[index] as T, second[index] as U]);
}

/** Trapezoid-rule area under y(x), mirroring np.trapezoid(y, x). */
function trapezoidAreas(y_values: number[], x_values: number[]): number {
  let total = 0;
  for (let index = 1; index < y_values.length; index += 1) {
    total += (((y_values[index - 1] ?? 0) + (y_values[index] ?? 0)) / 2) *
      ((x_values[index] ?? 0) - (x_values[index - 1] ?? 0));
  }
  return total;
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
  get_axes(): VGroup {
    const error = new Error("CoordinateSystem.get_axes must be implemented by a subclass.");
    error.name = "NotImplementedError";
    throw error;
  }
  get_axis(index: number): Mobject { return this.get_axes().__getitem__(index) as Mobject; }
  get_origin(): NDArray { return this.coords_to_point(0, 0); }

  get_axis_labels(): VGroup {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  coordinate_labels!: VGroup;

  add_coordinates(...args: unknown[]): this {
    // NOTE: mirrors Python `add_coordinates(*axes_numbers, **kwargs)`. A
    // trailing plain object is treated as the kwargs bag only when it carries
    // at least one known keyword (direction/buff/font_size/label_constructor/
    // x_values/excluding); a labels dict (numeric keys) stays an axes_number.
    const KNOWN_KWARGS = new Set([
      "direction",
      "buff",
      "font_size",
      "label_constructor",
      "x_values",
      "excluding",
    ]);
    let kwargs: Record<string, unknown> = {};
    let axes_numbers: unknown[] = args;
    const last = args[args.length - 1];
    if (
      isPlainRecord(last) &&
      Object.keys(last).some((key) => KNOWN_KWARGS.has(key))
    ) {
      kwargs = last;
      axes_numbers = args.slice(0, -1);
    }
    this.coordinate_labels = new VGroup();
    // if nothing is passed to axes_numbers, produce axes with default labelling
    if (axes_numbers.length === 0) {
      axes_numbers = Array.from({ length: this.dimension }, () => null);
    }

    const axes = [...this.get_axes().submobjects] as CoordinateAxisLike[];
    // NOTE: mirrors zip(self.axes, axes_numbers, strict=False) — stops at
    // the shorter of the two.
    const count = Math.min(axes.length, axes_numbers.length);
    for (let index = 0; index < count; index += 1) {
      const axis = axes[index] as CoordinateAxisLike;
      const values = axes_numbers[index];
      let labels: VGroup;
      if (values instanceof Map || isPlainRecord(values)) {
        // NOTE: Python add_labels() takes no **kwargs, so unexpected keywords
        // raise TypeError there; mirrored here.
        for (const key of Object.keys(kwargs)) {
          if (!["direction", "buff", "font_size", "label_constructor"].includes(key)) {
            throw new TypeError(`add_labels() got an unexpected keyword argument '${key}'`);
          }
        }
        axis.add_labels(
          values as Map<number, unknown> | Record<string, unknown>,
          (kwargs["direction"] as readonly number[] | NDArray | null | undefined) ?? null,
          (kwargs["buff"] as number | null | undefined) ?? null,
          (kwargs["font_size"] as number | null | undefined) ?? null,
          kwargs["label_constructor"] ?? null,
        );
        labels = axis.labels as VGroup;
      } else if ((values === null || values === undefined) && axis.scaling.custom_labels) {
        const tick_range = (axis.get_tick_range().toArray() as unknown[]).map(Number);
        const custom_labels = [
          ...(axis.scaling.get_custom_labels(tick_range) as Iterable<unknown>),
        ];
        // NOTE: mirrors dict(zip(tick_range, custom_labels, strict=True)).
        if (custom_labels.length !== tick_range.length) {
          const error = new Error(
            `zip() argument 2 is ${custom_labels.length < tick_range.length ? "shorter" : "longer"} than argument 1`,
          );
          error.name = "ValueError";
          throw error;
        }
        axis.add_labels(
          new Map<number, unknown>(
            tick_range.map((tick, tickIndex) => [tick, custom_labels[tickIndex]]),
          ),
          (kwargs["direction"] as readonly number[] | NDArray | null | undefined) ?? null,
          (kwargs["buff"] as number | null | undefined) ?? null,
          (kwargs["font_size"] as number | null | undefined) ?? null,
          kwargs["label_constructor"] ?? null,
        );
        labels = axis.labels as VGroup;
      } else {
        // NOTE: mirrors `axis.add_numbers(values, **kwargs)` — values binds
        // positionally to x_values, the whole kwargs bag forwards as **kwargs.
        axis.add_numbers(
          values as Iterable<number> | null | undefined,
          null,
          null,
          null,
          kwargs,
        );
        labels = axis.numbers as VGroup;
      }
      this.coordinate_labels.add(labels);
    }

    return this;
  }

  plot(
    function_: (x: number) => number,
    x_range: number[] | null = null,
    use_vectorized = false,
    colorscale: Array<ParsableManimColor | [ParsableManimColor, number]> | null = null,
    colorscale_axis = 1,
    options: ParametricFunctionOptions = {},
  ): ParametricFunction {
    const t_range: [number, number, number] = [
      this.x_range[0] ?? 0,
      this.x_range[1] ?? 0,
      this.x_range[2] ?? 1,
    ];
    if (x_range !== null) {
      for (let index = 0; index < x_range.length; index += 1) {
        t_range[index] = x_range[index] ?? t_range[index] ?? 0;
      }
    }
    if (x_range === null || x_range.length < 3) {
      // if t_range has a defined step size, increase the number of sample points per tick
      t_range[2] = (t_range[2] ?? 1) / this.num_sampled_graph_points_per_tick;
    }
    // For axes, the third coordinate of x_range indicates
    // tick frequency.  But for functions, it indicates a
    // sample frequency
    // NOTE: Python reads self.x_axis.scaling; the TS CoordinateSystem
    // hierarchy is not Mobject-based yet and has no x_axis, so the default
    // linear scaling is used until the Axes integration lands.
    const scaling: _ScaleBase = (this as unknown as { x_axis?: { scaling?: _ScaleBase } }).x_axis?.scaling ?? new LinearBase();
    const graph = new ParametricFunction(
      (t: number) => this.coords_to_point(t, function_(t)),
      { t_range, scaling, use_vectorized, ...options },
    );
    graph.underlying_function = function_;

    if (colorscale) {
      let new_colors: ParsableManimColor[];
      let pivots: number[];
      if (Array.isArray(colorscale[0])) {
        const pairs = colorscale as Array<[ParsableManimColor, number]>;
        new_colors = pairs.map(([color]) => color);
        pivots = pairs.map(([, pivot]) => pivot);
      } else {
        new_colors = colorscale as ParsableManimColor[];
        const ranges = [this.x_range, this.y_range];
        const pivot_min = ranges[colorscale_axis]?.[0] ?? 0;
        const pivot_max = ranges[colorscale_axis]?.[1] ?? 1;
        const pivot_frequency = (pivot_max - pivot_min) / (new_colors.length - 1);
        pivots = arange(pivot_min, pivot_max + pivot_frequency, pivot_frequency);
      }
      // NOTE: mirrors Python, which indexes x_range directly here (raising
      // TypeError when x_range is None and colorscale is given).
      const xr = x_range as number[];
      const resolution = xr.length === 2 ? 0.01 : (xr[2] ?? 0.01);
      const sample_points = arange(xr[0] ?? 0, (xr[1] ?? 0) + resolution, resolution);
      const color_list: ParsableManimColor[] = [];
      for (const samp_x of sample_points) {
        const axis_value = ([samp_x, function_(samp_x)] as number[])[colorscale_axis] ?? 0;
        if (axis_value <= (pivots[0] ?? 0)) {
          color_list.push(new_colors[0] as ParsableManimColor);
        } else if (axis_value >= (pivots[pivots.length - 1] ?? 0)) {
          color_list.push(new_colors[new_colors.length - 1] as ParsableManimColor);
        } else {
          for (let i = 0; i < pivots.length; i += 1) {
            if ((pivots[i] ?? 0) > axis_value) {
              const denom = (pivots[i] ?? 0) - (pivots[i - 1] ?? 0);
              let color_index = denom === 0 ? 0 : (axis_value - (pivots[i - 1] ?? 0)) / denom;
              color_index = Math.min(color_index, 1);
              const mob_color = interpolate_color(
                new ManimColor(new_colors[i - 1]),
                new ManimColor(new_colors[i]),
                color_index,
              );
              color_list.push(mob_color);
              break;
            }
          }
        }
      }
      if (config.renderer === RendererType.OPENGL) {
        graph.set_color(color_list as unknown as ParsableManimColor);
      } else {
        graph.set_stroke(color_list as unknown as ParsableManimColor);
        graph.set_sheen_direction([...RIGHT]);
      }
    }

    return graph;
  }

  plot_parametric_curve(
    function_: (t: number) => number[] | readonly number[] | NDArray,
    use_vectorized = false,
    options: ParametricFunctionOptions = {},
  ): ParametricFunction {
    const dim = this.dimension;
    const graph = new ParametricFunction(
      (t: number) => {
        const value = function_(t);
        const raw: number[] = isNDArray(value)
          ? ((value as NDArray).toArray() as unknown[]).map(numeric)
          : [...(value as readonly number[])].map(numeric);
        return this.coords_to_point(...raw.slice(0, dim));
      },
      { use_vectorized, ...options },
    );
    graph.underlying_function = function_;
    return graph;
  }

  plot_polar_graph(
    r_func: (theta: number) => number,
    theta_range: number[] | null = null,
    options: ParametricFunctionOptions = {},
  ): ParametricFunction {
    const resolved = (theta_range === null ? [0, TAU] : [...theta_range]) as [number, number, number];
    const graph = new ParametricFunction(
      (th: number) => this.pr2pt(r_func(th), th),
      { t_range: resolved, ...options },
    );
    graph.underlying_function = r_func;
    return graph;
  }

  input_to_graph_point(x: number, graph: ParametricFunction | VMobject): NDArray {
    const underlying = ("underlying_function" in (graph as object))
      ? (graph as unknown as Record<string, unknown>)["underlying_function"]
      : undefined;
    if (underlying !== undefined && underlying !== null) {
      return (graph as ParametricFunction).function(x);
    }
    const alpha = binary_search(
      (a: number) => {
        const proportion_point = pointRows((graph as VMobject).point_from_proportion(a), 3)[0] ?? [0, 0, 0];
        return this.point_to_coords(proportion_point)[0] ?? 0;
      },
      x,
      0,
      1,
    );
    if (alpha !== null) {
      return (graph as VMobject).point_from_proportion(alpha);
    }
    const startX = this.p2c((graph as VMobject).get_start())[0];
    const endX = this.p2c((graph as VMobject).get_end())[0];
    const error = new Error(
      `x=${x} not located in the range of the graph ([${startX}, ${endX}])`,
    );
    error.name = "ValueError";
    throw error;
  }

  input_to_graph_coords(x: number, graph: ParametricFunction): [number, number] {
    const underlying = (graph as unknown as { underlying_function: (t: number) => number }).underlying_function;
    return [x, underlying(x)];
  }

  i2gc(x: number, graph: ParametricFunction): [number, number] {
    return this.input_to_graph_coords(x, graph);
  }

  i2gp(x: number, graph: ParametricFunction): NDArray {
    return this.input_to_graph_point(x, graph);
  }

  angle_of_tangent(x: number, graph: ParametricFunction, dx = 1e-8): number {
    const p0 = this.input_to_graph_coords(x, graph);
    const p1 = this.input_to_graph_coords(x + dx, graph);
    return Number(
      angle_of_vector([(p1[0] ?? 0) - (p0[0] ?? 0), (p1[1] ?? 0) - (p0[1] ?? 0)]),
    );
  }

  get_area(
    graph: ParametricFunction,
    x_range: readonly [number, number] | [number, number] | null = null,
    color: ParsableManimColor | ParsableManimColor[] = [BLUE, GREEN],
    opacity = 0.3,
    bounded_graph: ParametricFunction | null = null,
    kwargs: Record<string, unknown> = {},
  ): Polygon {
    let a: number;
    let b: number;
    if (x_range === null) {
      a = graph.t_min;
      b = graph.t_max;
    } else {
      [a, b] = [x_range[0] ?? 0, x_range[1] ?? 0];
    }
    if (bounded_graph !== null && bounded_graph !== undefined) {
      if (bounded_graph.t_min > b) {
        const error = new Error(`Ranges not matching: ${bounded_graph.t_min} < ${b}`);
        error.name = "ValueError";
        throw error;
      }
      if (bounded_graph.t_max < a) {
        const error = new Error(`Ranges not matching: ${bounded_graph.t_max} > ${a}`);
        error.name = "ValueError";
        throw error;
      }
      a = Math.max(a, bounded_graph.t_min);
      b = Math.min(b, bounded_graph.t_max);
    }

    // NOTE: graph points are converted to plain rows first: Polygon groups a
    // single array-of-points argument via Array.isArray checks, so raw
    // NDArray vertices would be misread as one point.
    const toRow = (point: unknown): number[] => pointRows(point, 3)[0] ?? [0, 0, 0];
    const inRange = (point: number[]): boolean => {
      const x = this.point_to_coords(point)[0] ?? 0;
      return a <= x && x <= b;
    };
    let points: number[][];
    if (bounded_graph === null || bounded_graph === undefined) {
      points = [
        toRow(this.c2p(a)),
        toRow(graph.function(a)),
        ...pointRows(graph.points, 3).filter(inRange),
        toRow(graph.function(b)),
        toRow(this.c2p(b)),
      ];
    } else {
      const collect = (g: ParametricFunction): number[][] => [
        toRow(g.function(a)),
        ...pointRows(g.points, 3).filter(inRange),
        toRow(g.function(b)),
      ];
      points = [...collect(graph), ...collect(bounded_graph).reverse()];
    }
    // NOTE: `color` may be a list (gradient, mirroring Python's default
    // `(BLUE, GREEN)`). Python's set_fill/set_stroke forward such lists to
    // update_rgbas_array (a point-wise gradient); the TS set_fill/set_color
    // single-parse the color into ManimColor and reject lists, so the
    // gradient path is invoked directly here with identical ordering
    // (set_opacity first, then the color updates).
    const area = new Polygon(points, kwargs).set_opacity(opacity);
    if (Array.isArray(color)) {
      const gradient = color.map((c) => new ManimColor(c));
      area.update_rgbas_array("fill_rgbas", gradient, null);
      area.update_rgbas_array("stroke_rgbas", gradient, null);
      return area;
    }
    return area.set_color(color);
  }

  slope_of_tangent(x: number, graph: ParametricFunction, dx = 1e-8): number {
    // NOTE: mirrors Python `slope_of_tangent(self, x, graph, **kwargs)`
    // (kwargs forwards to angle_of_tangent, whose only keyword is dx).
    const val: number = Math.tan(this.angle_of_tangent(x, graph, dx));
    return val;
  }

  plot_antiderivative_graph(
    graph: ParametricFunction,
    y_intercept = 0,
    samples = 50,
    use_vectorized = false,
    options: ParametricFunctionOptions = {},
  ): ParametricFunction {
    // Mirrors the nested `antideriv` closure in Python: integrates the
    // reference graph's underlying function from 0 to x with the trapezoid
    // rule (np.trapezoid), adding y_intercept. With use_vectorized the input
    // is an array (np.linspace(..., axis=1) layout: one row per x value) and
    // the output is the matching array of areas.
    const underlying = (graph as unknown as { underlying_function: (t: number) => number }).underlying_function;
    const antideriv = (x: number | NDArray): number | NDArray => {
      if (use_vectorized && isNDArray(x)) {
        const x_rows: number[] = [];
        const collect = (value: unknown): void => {
          if (Array.isArray(value)) {
            for (const item of value) collect(item);
          } else {
            x_rows.push(numeric(value));
          }
        };
        collect((x as NDArray).toArray());
        const integrated = x_rows.map((x_value) => {
          const x_vals = linspace(0, x_value, samples);
          const y_vals = x_vals.map((v) => underlying(v));
          return trapezoidAreas(y_vals, x_vals) + y_intercept;
        });
        return array(integrated) as NDArray;
      }
      const x_value = numeric(x as number);
      const x_vals = linspace(0, x_value, samples);
      const y_vals = x_vals.map((v) => underlying(v));
      return trapezoidAreas(y_vals, x_vals) + y_intercept;
    };
    return this.plot(antideriv as (x: number) => number, null, use_vectorized, null, 1, options);
  }

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
    // Like Python (`point.ndim == 2 -> result.T`), a list of points maps
    // row-wise to a list of coordinate pairs.
    if (
      Array.isArray(point) &&
      point.length > 0 &&
      Array.isArray((point as unknown as unknown[])[0])
    ) {
      const single = (p: readonly number[]): number[] => this.point_to_coords(p);
      return (point as unknown as number[][]).map((p) => single(p)) as unknown as number[];
    }
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
  // NOTE: Python Axes.__init__ builds self.axes as a VGroup of NumberLine
  // axes (plus x_axis/y_axis/z_axis shortcuts). That construction is part of
  // a later batch; the field exists here so get_axes() has its faithful
  // target to return.
  axes: VGroup = new VGroup();

  override point_to_coords(point: readonly number[]): number[] {
    // Mirrors Python Axes.point_to_coords: one number per axis via
    // `axis.point_to_number(point)` (transposed for a list of points).
    // NOTE: the TS Axes hierarchy is not Mobject-based yet, so `axes` is
    // empty until the NumberLine integration lands; fall back to the linear
    // NumberPlane mapping in that case.
    const axes = this.get_axes().submobjects;
    const hasAxes =
      axes.length > 0 &&
      axes.every(
        (axis) =>
          typeof (axis as unknown as { point_to_number?: unknown })
            .point_to_number === "function",
      );
    if (!hasAxes) return super.point_to_coords(point);
    type NumberLineLike = {
      point_to_number(point: unknown): number | NDArray;
    };
    if (
      Array.isArray(point) &&
      point.length > 0 &&
      Array.isArray((point as unknown as unknown[])[0])
    ) {
      const points = point as unknown as number[][];
      const perAxis: number[][] = (axes as Mobject[]).map((axis) => {
        const out = (axis as unknown as NumberLineLike).point_to_number(points);
        if (isNDArray(out)) {
          return ((out as NDArray).toArray() as unknown[]).map(numeric);
        }
        return (out as unknown as unknown[]).map(numeric);
      });
      return points.map((_, index) =>
        perAxis.map((column) => column[index] ?? 0),
      ) as unknown as number[];
    }
    return (axes as Mobject[]).map((axis) =>
      numeric(
        (axis as unknown as NumberLineLike).point_to_number([
          ...point,
        ]) as number,
      ),
    );
  }

  override get_axes(): VGroup {
    return this.axes;
  }

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
  num_axis_pieces: number;

  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
    light_source: readonly number[] = [-7, -9, 10],
    shaded_axes: ShadedAxis[] = [],
    num_axis_pieces = 20,
  ) {
    super(x_range, y_range, x_length, y_length);
    this.light_source = [...light_source];
    this.shaded_axes = shaded_axes;
    this.num_axis_pieces = num_axis_pieces;
    if (config.renderer === RendererType.CAIRO) {
      this._add_3d_pieces();
      this._set_axis_shading();
    }
  }

  *[Symbol.iterator](): Iterator<ShadedAxis> {
    // NOTE: mirrors Python `for axis in self` (ThreeDAxes is a VGroup of
    // axes); the TS port tracks the axes in `shaded_axes` until the Axes
    // hierarchy becomes Mobject-based.
    yield* this.shaded_axes;
  }

  _add_3d_pieces(): void {
    // NOTE: mirrors ``for axis in self.axes``; the TS Axes hierarchy is not
    // Mobject-based yet, so `axes` is empty until the NumberLine integration
    // lands and this loop is a no-op in the meantime (same as _set_axis_shading).
    for (const axis of this.axes.submobjects) {
      const mob = axis as Mobject;
      // NOTE: get_pieces returns a Group; unpack its members explicitly
      // (spreading the Group itself would go through Mobject's iterator,
      // which also yields the container when it has points).
      const pieces = new VGroup(...mob.get_pieces(this.num_axis_pieces).submobjects as VMobject[]);
      (mob as unknown as { pieces: VGroup }).pieces = pieces;
      mob.add(pieces);
      (mob as unknown as VMobject).set_stroke(undefined, 0, undefined, false);
      (mob as unknown as VMobject).set_shade_in_3d(true);
    }
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

  override get_axes(): VGroup {
    return this.axes;
  }
}

/** TS stand-in for Python's `float | complex`: a real number, a
 * `{ re, im }` pair (mirroring `complex(x, y)`), or a `[re, im]` tuple. */
export type ComplexNumberLike = number | ComplexPoint | readonly [number, number];

function toComplexPoint(value: ComplexNumberLike): ComplexPoint {
  if (typeof value === "number") return { re: value, im: 0 };
  if (Array.isArray(value)) {
    const pair = value as readonly number[];
    return { re: pair[0] ?? 0, im: pair[1] ?? 0 };
  }
  const c = value as ComplexPoint;
  return { re: c.re ?? 0, im: c.im ?? 0 };
}

export class ComplexPlane extends NumberPlane {
  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
  ) {
    // NOTE: mirrors Python `ComplexPlane.__init__(self, **kwargs)` (which
    // only forwards to NumberPlane); the TS CoordinateSystem hierarchy
    // takes explicit range/length params instead of a kwargs bag.
    super(x_range, y_range, x_length, y_length);
  }

  number_to_point(number: ComplexNumberLike): NDArray {
    // NOTE: mirrors `number = complex(number)` (accepts floats and complex
    // alike) followed by `coords_to_point(number.real, number.imag)`.
    const c = toComplexPoint(number);
    return this.coords_to_point(c.re, c.im);
  }

  n2p(number: ComplexNumberLike): NDArray {
    return this.number_to_point(number);
  }

  point_to_number(point: readonly number[]): ComplexPoint {
    const [x = 0, y = 0] = this.point_to_coords(point);
    return { re: x, im: y };
  }

  p2n(point: readonly number[]): ComplexPoint {
    return this.point_to_number(point);
  }
}

