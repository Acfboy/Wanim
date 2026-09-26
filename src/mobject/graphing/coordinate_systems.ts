import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { config } from "../../_config/utils.js";
import { arange, isNDArray, linspace, numeric, pointFrom, pointRows, type ComplexPoint } from "../../utils/arrays.js";
import { update_dict_recursively, merge_dicts_recursively } from "../../utils/config_ops.js";
import { MED_SMALL_BUFF, SMALL_BUFF, DEGREES, DL, DOWN, DR, LEFT, ORIGIN, OUT, PI, RIGHT, UP, UR, TAU, RendererType } from "../../constants.js";
import { BLUE_D } from "../../utils/color/index.js";
import { BLACK, BLUE, GREEN, WHITE } from "../../utils/color/core.js";
import { color_gradient, interpolate_color, invert_color, ManimColor, type ParsableManimColor } from "../../utils/color/core.js";
import { binary_search } from "../../utils/simple_functions.js";
import { angle_of_vector } from "../../utils/space_ops.js";
import { VDict, VectorizedPoint, VGroup, VMobject } from "../types/vectorized_mobject.js";
import { Polygon, Rectangle, RegularPolygon } from "../geometry/polygram.js";
import { Dot } from "../geometry/arc.js";
import { Arrow, DashedLine, Line } from "../geometry/line.js";
import { ImplicitFunction, ParametricFunction, type ParametricFunctionOptions } from "./functions.js";
import { Surface } from "../three_d/three_dimensions.js";
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
  get_unit_size(): number;
  get_unit_vector(): number[];
  get_projection(point: unknown): number[];
  number_to_point(number: unknown): NDArray;
  point_to_number(point: unknown): number | NDArray;
  _create_label_tex(label: unknown, label_constructor?: unknown, kwargs?: Record<string, unknown>): VMobject;
  get_number_mobject(
    x: number,
    direction?: readonly number[] | NDArray | null,
    buff?: number | null,
    font_size?: number | null,
    label_constructor?: unknown,
    number_config?: Record<string, unknown>,
  ): VMobject;
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

/** Dispatches `self._origin_shift(axis_range)` the way Python's MRO does.
// In Python, NumberPlane extends Axes so the abstract CoordinateSystem stub
// is never hit; the TS hierarchy has NumberPlane extending CoordinateSystem
// directly (Axes extends NumberPlane), so a NumberPlane instance's
// constructor may still hold the throwing base stub. Fall back to the Axes
// implementation in that case (no subclass overrides it). */
function originShiftOf(host: CoordinateSystem, axis_range: readonly number[]): number {
  const ctor = host.constructor as unknown as {
    _origin_shift?: (axis_range: readonly number[]) => number;
  };
  if (typeof ctor._origin_shift === "function") {
    try {
      return ctor._origin_shift(axis_range);
    } catch {
      // fall through to the Axes implementation below
    }
  }
  return Axes._origin_shift(axis_range);
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

  get_x_axis(): CoordinateAxisLike { return this.get_axis(0) as unknown as CoordinateAxisLike; }

  get_y_axis(): CoordinateAxisLike { return this.get_axis(1) as unknown as CoordinateAxisLike; }

  get_z_axis(): CoordinateAxisLike { return this.get_axis(2) as unknown as CoordinateAxisLike; }

  get_x_unit_size(): number { return this.get_x_axis().get_unit_size(); }

  get_y_unit_size(): number { return this.get_y_axis().get_unit_size(); }

  get_x_axis_label(
    label: number | string | VMobject,
    edge: readonly number[] | NDArray = [...UR],
    direction: readonly number[] | NDArray = [...UR],
    buff = SMALL_BUFF,
    kwargs: Record<string, unknown> = {},
  ): Mobject {
    void kwargs;
    return this._get_axis_label(label, this.get_x_axis(), edge, direction, buff);
  }

  get_y_axis_label(
    label: number | string | VMobject,
    edge: readonly number[] | NDArray = [...UR],
    // NOTE: mirrors Python default `direction=UP * 0.5 + RIGHT` = [1, 0.5, 0].
    direction: readonly number[] | NDArray = [1, 0.5, 0],
    buff = SMALL_BUFF,
    kwargs: Record<string, unknown> = {},
  ): Mobject {
    void kwargs;
    return this._get_axis_label(label, this.get_y_axis(), edge, direction, buff);
  }

  _get_axis_label(
    label: number | string | VMobject,
    axis: Mobject,
    edge: readonly number[] | NDArray,
    direction: readonly number[] | NDArray,
    buff = SMALL_BUFF,
  ): Mobject {
    // NOTE: mirrors Python verbatim, including its use of self.x_axis (not
    // the `axis` argument) to construct the label.
    const label_mobject: Mobject = (this as unknown as { x_axis: CoordinateAxisLike }).x_axis._create_label_tex(label);
    label_mobject.next_to(axis.get_edge_center(edge), direction, buff);
    label_mobject.shift_onto_screen({ buff: MED_SMALL_BUFF });
    return label_mobject;
  }

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

  get_line_from_axis_to_point(
    index: number,
    point: readonly number[] | NDArray,
    line_func: new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject = DashedLine as unknown as new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject,
    line_config: Record<string, unknown> | null = null,
    color: ParsableManimColor | null = null,
    stroke_width = 2,
  ): VMobject {
    const resolved_config: Record<string, unknown> = line_config !== null && line_config !== undefined ? { ...line_config } : {};
    const resolved_color = color ?? new VMobject().color;
    resolved_config["color"] = ManimColor.parse(resolved_color as ParsableManimColor);
    resolved_config["stroke_width"] = stroke_width;
    const axis = this.get_axis(index) as unknown as CoordinateAxisLike;
    const line = new line_func(axis.get_projection(point), point, resolved_config);
    return line;
  }

  get_vertical_line(point: readonly number[] | NDArray, kwargs: Record<string, unknown> = {}): VMobject {
    const { line_func, line_config, color, stroke_width, ...rest } = kwargs as {
      line_func?: new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject;
      line_config?: Record<string, unknown> | null;
      color?: ParsableManimColor | null;
      stroke_width?: number;
    };
    if (Object.keys(rest).length > 0) {
      throw new TypeError(`get_line_from_axis_to_point() got unexpected keyword arguments '${Object.keys(rest).join(", ")}'`);
    }
    return this.get_line_from_axis_to_point(
      0,
      point,
      line_func ?? (DashedLine as unknown as new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject),
      line_config ?? null,
      color ?? null,
      stroke_width ?? 2,
    );
  }

  get_horizontal_line(point: readonly number[] | NDArray, kwargs: Record<string, unknown> = {}): VMobject {
    const { line_func, line_config, color, stroke_width, ...rest } = kwargs as {
      line_func?: new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject;
      line_config?: Record<string, unknown> | null;
      color?: ParsableManimColor | null;
      stroke_width?: number;
    };
    if (Object.keys(rest).length > 0) {
      throw new TypeError(`get_line_from_axis_to_point() got unexpected keyword arguments '${Object.keys(rest).join(", ")}'`);
    }
    return this.get_line_from_axis_to_point(
      1,
      point,
      line_func ?? (DashedLine as unknown as new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject),
      line_config ?? null,
      color ?? null,
      stroke_width ?? 2,
    );
  }

  get_lines_to_point(point: readonly number[] | NDArray, kwargs: Record<string, unknown> = {}): VGroup {
    return new VGroup(
      this.get_horizontal_line(point, kwargs) as VMobject,
      this.get_vertical_line(point, kwargs) as VMobject,
    );
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

  plot_implicit_curve(
    func: (x: number, y: number) => number,
    min_depth = 5,
    max_quads = 1500,
    options: Record<string, unknown> = {},
  ): ImplicitFunction {
    const x_scale = this.get_x_axis().scaling;
    const y_scale = this.get_y_axis().scaling;
    const applyScale = (scale: _ScaleBase, value: number): number => {
      const out = scale.function(value);
      return typeof out === "number" ? out : numeric(pointRows(out, 1)[0]?.[0] ?? value);
    };
    const graph = new ImplicitFunction(
      (x: number, y: number) => func(applyScale(x_scale, x), applyScale(y_scale, y)),
      {
        x_range: [...this.x_range].slice(0, 2),
        y_range: [...this.y_range].slice(0, 2),
        min_depth,
        max_quads,
        ...options,
      },
    );
    graph
      .stretch(this.get_x_unit_size(), 0, [...ORIGIN])
      .stretch(this.get_y_unit_size(), 1, [...ORIGIN])
      .shift(this.get_origin());
    return graph;
  }

  plot_surface(
    function_: (u: number, v: number) => number,
    u_range: [number, number] | null = null,
    v_range: [number, number] | null = null,
    colorscale: Array<ParsableManimColor | [ParsableManimColor, number]> | null = null,
    colorscale_axis = 2,
    options: Record<string, unknown> = {},
  ): Surface {
    if (config.renderer === RendererType.OPENGL) {
      // NOTE: Python builds an OpenGLSurface here; the TS port has no
      // OpenGLSurface yet, so the OpenGL branch is a marked stub.
      throw new Error("TODO: plot_surface OpenGL renderer needs OpenGLSurface, not ported yet");
    }
    // NOTE: Python forwards **kwargs to Surface; the TS Surface takes the
    // same knobs positionally (with identical defaults), so known keys are
    // forwarded positionally and unknown keys are ignored (Surface itself
    // ignores its trailing options bag).
    const o = options as Record<string, unknown>;
    const surface = new Surface(
      (u: number, v: number) => this.c2p(u, v, function_(u, v)),
      u_range ?? [0, 1],
      v_range ?? [0, 1],
      (o["resolution"] as number | [number, number] | undefined) ?? 32,
      (o["surface_piece_config"] as Record<string, unknown> | undefined) ?? {},
      (o["fill_color"] as ParsableManimColor | undefined) ?? BLUE_D,
      (o["fill_opacity"] as number | undefined) ?? 1.0,
      (o["checkerboard_colors"] as Iterable<ParsableManimColor> | false | undefined) ?? undefined,
      (o["stroke_color"] as ParsableManimColor | undefined) ?? undefined,
      (o["stroke_width"] as number | undefined) ?? undefined,
      (o["should_make_jagged"] as boolean | undefined) ?? undefined,
      (o["pre_function_handle_to_anchor_scale_factor"] as number | undefined) ?? undefined,
    );
    if (colorscale) {
      // NOTE: Surface.set_fill_by_value lives in a later layer (L20); call
      // through a structural cast so this batch compiles. At runtime it
      // resolves once that batch lands.
      (surface as unknown as {
        set_fill_by_value(opts: Record<string, unknown>): void;
      }).set_fill_by_value({
        axes: this,
        colorscale,
        axis: colorscale_axis,
      });
    }
    return surface;
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

  get_graph_label(
    graph: ParametricFunction,
    label: number | string | VMobject = "f(x)",
    x_val: number | null = null,
    direction: readonly number[] | NDArray = [...RIGHT],
    buff = MED_SMALL_BUFF,
    color: ParsableManimColor | null = null,
    dot = false,
    dot_config: Record<string, unknown> | null = null,
  ): Mobject {
    const resolved_dot_config: Record<string, unknown> = dot_config ?? {};
    const resolved_color = color ?? graph.get_color();
    // NOTE: mirrors Python `self.x_axis._create_label_tex(label)`.
    const label_object: Mobject = (this as unknown as { x_axis: CoordinateAxisLike }).x_axis._create_label_tex(label);
    label_object.set_color(resolved_color as ParsableManimColor);
    let point: NDArray;
    if (x_val === null || x_val === undefined) {
      // Search from right to left
      let found: NDArray | null = null;
      for (const x of linspace(this.x_range[1] ?? 0, this.x_range[0] ?? 0, 100)) {
        const candidate = this.input_to_graph_point(x, graph);
        if ((pointRows(candidate, 3)[0]?.[1] ?? 0) < Number(config.frame_y_radius)) {
          found = candidate;
          break;
        }
      }
      // NOTE: mirrors Python, where `point` keeps the last loop value
      // (the left-most sample, x_range[0]) when no sample breaks.
      point = found ?? this.input_to_graph_point(this.x_range[0] ?? 0, graph);
    } else {
      point = this.input_to_graph_point(x_val, graph);
    }
    label_object.next_to(point, direction, buff);
    label_object.shift_onto_screen();
    if (dot) {
      // NOTE: mirrors Python `Dot(point=point, **dot_config)`: known Dot
      // parameters bind positionally, the rest forwards as options.
      const dc = resolved_dot_config;
      const dot_mob = new Dot(
        point,
        (dc["radius"] as number | undefined) ?? undefined,
        (dc["stroke_width"] as number | undefined) ?? undefined,
        (dc["fill_opacity"] as number | undefined) ?? undefined,
        (dc["color"] as ParsableManimColor | undefined) ?? undefined,
        Object.fromEntries(Object.entries(dc).filter(([k]) => !["radius", "stroke_width", "fill_opacity", "color", "point"].includes(k))),
      );
      label_object.add(dot_mob);
      (label_object as unknown as Record<string, unknown>)["dot"] = dot_mob;
    }
    return label_object;
  }

  angle_of_tangent(x: number, graph: ParametricFunction, dx = 1e-8): number {
    const p0 = this.input_to_graph_coords(x, graph);
    const p1 = this.input_to_graph_coords(x + dx, graph);
    return Number(
      angle_of_vector([(p1[0] ?? 0) - (p0[0] ?? 0), (p1[1] ?? 0) - (p0[1] ?? 0)]),
    );
  }

  get_riemann_rectangles(
    graph: ParametricFunction,
    x_range: number[] | null = null,
    dx = 0.1,
    input_sample_type: string = "left",
    stroke_width = 1,
    stroke_color: ParsableManimColor = BLACK,
    fill_opacity = 1,
    color: Iterable<ParsableManimColor> | ParsableManimColor = [BLUE, GREEN],
    show_signed_area = true,
    bounded_graph: ParametricFunction | null = null,
    blend = false,
    width_scale_factor = 1.001,
  ): VGroup {
    // setting up x_range, overwrite user's third input
    let resolved_range: number[];
    if (x_range === null || x_range === undefined) {
      if (bounded_graph === null || bounded_graph === undefined) {
        resolved_range = [graph.t_min, graph.t_max, dx];
      } else {
        const x_min = Math.max(graph.t_min, bounded_graph.t_min);
        const x_max = Math.min(graph.t_max, bounded_graph.t_max);
        resolved_range = [x_min, x_max, dx];
      }
    } else {
      resolved_range = [...x_range.slice(0, 2), dx];
    }

    const rectangles = new VGroup();
    const x_range_array = arange(resolved_range[0] ?? 0, resolved_range[1] ?? 0, resolved_range[2] ?? dx);

    // NOTE: mirrors Python `isinstance(color, (list, tuple))`.
    let resolved_colors: ManimColor[];
    if (Array.isArray(color)) {
      resolved_colors = (color as ParsableManimColor[]).map((c) => new ManimColor(c));
    } else {
      resolved_colors = [new ManimColor(color as ParsableManimColor)];
    }
    const colors = color_gradient(resolved_colors, x_range_array.length);

    for (let index = 0; index < x_range_array.length; index += 1) {
      const x = x_range_array[index] ?? 0;
      const rect_color = colors[index] as ManimColor;
      let sample_input: number;
      if (input_sample_type === "left") {
        sample_input = x;
      } else if (input_sample_type === "right") {
        sample_input = x + dx;
      } else if (input_sample_type === "center") {
        sample_input = x + 0.5 * dx;
      } else {
        const error = new Error("Invalid input sample type");
        error.name = "ValueError";
        throw error;
      }
      const graph_point = this.input_to_graph_point(sample_input, graph);

      let y_point: number;
      if (bounded_graph === null || bounded_graph === undefined) {
        y_point = originShiftOf(this, this.y_range);
      } else {
        y_point = (bounded_graph as unknown as { underlying_function: (t: number) => number }).underlying_function(x);
      }

      const points = new VGroup(
        ...[
          this.coords_to_point(x, y_point),
          this.coords_to_point(x + width_scale_factor * dx, y_point),
          graph_point,
        ].map((p) => new VectorizedPoint(p)),
      );

      const rect = new Rectangle().replace(points, 0, true);
      rectangles.add(rect);

      // checks if the rectangle is under the x-axis
      let fill: ManimColor = rect_color;
      if ((this.p2c(pointRows(graph_point, 3)[0] ?? [0, 0, 0])[1] ?? 0) < y_point && show_signed_area) {
        fill = invert_color(rect_color);
      }

      // blends rectangles smoothly
      let resolved_stroke: ParsableManimColor = stroke_color;
      if (blend) {
        resolved_stroke = fill;
      }

      rect.set_style({
        fill_color: fill,
        fill_opacity,
        stroke_color: resolved_stroke,
        stroke_width,
      });
    }

    return rectangles;
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

  plot_derivative_graph(
    graph: ParametricFunction,
    color: ParsableManimColor = GREEN,
    options: ParametricFunctionOptions = {},
  ): ParametricFunction {
    const deriv = (x: number): number => this.slope_of_tangent(x, graph);
    return this.plot(deriv, null, false, null, 1, { color, ...options });
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

  get_secant_slope_group(
    x: number,
    graph: ParametricFunction,
    dx: number | null = null,
    dx_line_color: ParsableManimColor = "PURE_YELLOW",
    dy_line_color: ParsableManimColor | null = null,
    dx_label: number | string | null = null,
    dy_label: number | string | null = null,
    include_secant_line = true,
    secant_line_color: ParsableManimColor = GREEN,
    secant_line_length = 10,
  ): VGroup {
    const group = new VGroup() as VGroup & Record<string, unknown>;

    const resolved_dx = dx || ((this.x_range[1] ?? 0) - (this.x_range[0] ?? 0)) / 10;
    const resolved_dy_color = dy_line_color || graph.get_color();

    const p1 = this.input_to_graph_point(x, graph);
    const p2 = this.input_to_graph_point(x + resolved_dx, graph);
    const p1row = pointRows(p1, 3)[0] ?? [0, 0, 0];
    const p2row = pointRows(p2, 3)[0] ?? [0, 0, 0];
    const interim_point = [p2row[0] ?? 0, p1row[1] ?? 0, 0];

    const dx_line = new Line(p1row, interim_point, { color: dx_line_color });
    const df_line = new Line(interim_point, p2row, { color: resolved_dy_color });
    group.dx_line = dx_line;
    group.df_line = df_line;
    group.add(dx_line, df_line);

    const labels = new VGroup();
    const x_axis = (this as unknown as { x_axis: CoordinateAxisLike }).x_axis;
    if (dx_label !== null && dx_label !== undefined) {
      const dx_label_mob = x_axis._create_label_tex(dx_label);
      group.dx_label = dx_label_mob;
      labels.add(dx_label_mob);
      group.add(dx_label_mob);
    }
    if (dy_label !== null && dy_label !== undefined) {
      const df_label_mob = x_axis._create_label_tex(dy_label);
      group.df_label = df_label_mob;
      labels.add(df_label_mob);
      group.add(df_label_mob);
    }

    if (labels.submobjects.length > 0) {
      const max_width = 0.8 * dx_line.width;
      const max_height = 0.8 * df_line.height;
      if (labels.width > max_width) {
        labels.width = max_width;
      }
      if (labels.height > max_height) {
        labels.height = max_height;
      }
    }

    if (dx_label !== null && dx_label !== undefined) {
      const dx_label_mob = group.dx_label as VMobject;
      // NOTE: mirrors Python `np.sign(dx) * DOWN`.
      dx_label_mob.next_to(dx_line, [0, -Math.sign(resolved_dx), 0], dx_label_mob.height / 2);
      dx_label_mob.set_color(dx_line.get_color());
    }

    if (dy_label !== null && dy_label !== undefined) {
      const df_label_mob = group.df_label as VMobject;
      df_label_mob.next_to(df_line, [Math.sign(resolved_dx), 0, 0], df_label_mob.height / 2);
      df_label_mob.set_color(df_line.get_color());
    }

    if (include_secant_line) {
      const secant_line = new Line(p1row, p2row, { color: secant_line_color });
      secant_line.scale(secant_line_length / secant_line.get_length());
      group.secant_line = secant_line;
      group.add(secant_line);
    }
    return group as VGroup;
  }

  get_vertical_lines_to_graph(
    graph: ParametricFunction,
    x_range: number[] | null = null,
    num_lines = 20,
    kwargs: Record<string, unknown> = {},
  ): VGroup {
    const resolved = x_range !== null && x_range !== undefined ? [...x_range] : [...this.x_range];
    return new VGroup(
      ...linspace(resolved[0] ?? 0, resolved[1] ?? 0, num_lines).map((x) =>
        this.get_vertical_line(this.i2gp(x, graph), { ...kwargs }) as VMobject,
      ),
    );
  }

  get_T_label(
    x_val: number,
    graph: ParametricFunction,
    label: number | string | Mobject | null = null,
    label_color: ParsableManimColor | null = null,
    triangle_size = MED_SMALL_BUFF,
    triangle_color: ParsableManimColor | null = WHITE,
    line_func: new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject = Line as unknown as new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject,
    line_color: ParsableManimColor = "PURE_YELLOW",
  ): VGroup {
    const T_label_group = new VGroup();
    const triangle = new RegularPolygon(3, { start_angle: PI / 2, stroke_width: 0 }).set_fill(triangle_color as ParsableManimColor, 1);
    triangle.height = triangle_size;
    triangle.move_to(this.coords_to_point(x_val, 0), [...UP]);
    if (label !== null && label !== undefined) {
      const x_axis = (this as unknown as { x_axis: CoordinateAxisLike }).x_axis;
      const t_label = x_axis._create_label_tex(label) as unknown as Mobject & {
        set_color(c: ParsableManimColor): Mobject;
        next_to(p: unknown, d: unknown): Mobject;
      };
      if (label_color !== null && label_color !== undefined) {
        // NOTE: Python passes color into _create_label_tex; the TS
        // _create_label_tex takes no color, so apply it afterwards.
        t_label.set_color(label_color);
      }
      t_label.next_to(triangle, [...DOWN]);
      T_label_group.add(t_label as unknown as VMobject);
    }

    const v_line = this.get_vertical_line(this.i2gp(x_val, graph), {
      color: line_color,
      line_func,
    }) as VMobject;

    T_label_group.add(triangle as unknown as VMobject, v_line);

    return T_label_group;
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

function unitPosition(value: number, range: number[]): number {
  // Mirrors a unit-sized NumberLine axis centered on the range midpoint
  // (Python NumberPlane/ComplexPlane with x_length=None): pos(x) is x
  // shifted so the range midpoint lands on the origin.
  return Number(value) - (Number(range[0]) + Number(range[1])) / 2;
}

function unitValue(point: number, range: number[]): number {
  // Algebraic inverse of unitPosition above.
  return Number(point) + (Number(range[0]) + Number(range[1])) / 2;
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
  axis_config: Record<string, unknown> = {};
  y_axis_config: Record<string, unknown> = {};
  background_line_style: Record<string, unknown> = {};
  faded_line_style: Record<string, unknown> | null = null;
  faded_line_ratio = 1;
  make_smooth_after_applying_functions = true;
  background_lines: VGroup | null = null;
  faded_lines: VGroup | null = null;
  x_lines: VGroup = new VGroup();
  y_lines: VGroup = new VGroup();

  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
    background_line_style: Record<string, unknown> | null = null,
    faded_line_style: Record<string, unknown> | null = null,
    faded_line_ratio = 1,
    make_smooth_after_applying_functions = true,
    options: {
      axis_config?: Record<string, unknown> | null;
      y_axis_config?: Record<string, unknown> | null;
      [key: string]: unknown;
    } = {},
  ) {
    // NOTE: mirrors Python NumberPlane.__init__ defaults, which use the raw
    // frame radii (unlike Axes/CoordinateSystem, which round them to
    // [-7, 7, 1] / [-4, 4, 1]).
    super(
      x_range ?? [-config.frame_x_radius, config.frame_x_radius, 1],
      y_range ?? [-config.frame_y_radius, config.frame_y_radius, 1],
      x_length,
      y_length,
    );
    this.axis_config = {
      stroke_width: 2,
      include_ticks: false,
      include_tip: false,
      line_to_number_buff: SMALL_BUFF,
      label_direction: [...DR],
      font_size: 24,
    };
    this.y_axis_config = { label_direction: [...DR] };
    this.background_line_style = {
      stroke_color: BLUE_D,
      stroke_width: 2,
      stroke_opacity: 1,
    };
    const { axis_config, y_axis_config, ..._restKw } = options;
    void _restKw;
    Axes._update_default_configs(
      [this.axis_config, this.y_axis_config, this.background_line_style],
      [axis_config ?? null, y_axis_config ?? null, background_line_style],
    );
    this.faded_line_style = faded_line_style;
    this.faded_line_ratio = faded_line_ratio;
    this.make_smooth_after_applying_functions = make_smooth_after_applying_functions;
    // NOTE: Python forwards axis_config/y_axis_config into Axes.__init__
    // here. The TS Axes constructor already ran via super() (TS requires
    // super first), so merge them into this instance's Axes-level configs
    // the same way Axes.__init__ would.
    this._init_background_lines();
  }

  _init_background_lines(): void {
    if (this.faded_line_style === null || this.faded_line_style === undefined) {
      const style: Record<string, unknown> = { ...this.background_line_style };
      for (const key of Object.keys(style)) {
        if (typeof style[key] === "number") {
          style[key] = (style[key] as number) * 0.5;
        }
      }
      this.faded_line_style = style;
    }
    const [background_lines, faded_lines] = this._get_lines();
    (background_lines as VGroup).set_style?.(this.background_line_style as never);
    (faded_lines as VGroup).set_style?.(this.faded_line_style as never);
    this.background_lines = background_lines;
    this.faded_lines = faded_lines;
  }

  _get_lines(): [VGroup, VGroup] {
    // NOTE: mirrors Python NumberPlane._get_lines verbatim. The TS Axes
    // hierarchy is not Mobject-based yet, so get_x_axis()/get_y_axis()
    // resolve against an empty axes group until the NumberLine integration
    // lands; in that case the returned groups are empty.
    let x_axis: CoordinateAxisLike | null = null;
    let y_axis: CoordinateAxisLike | null = null;
    try {
      x_axis = this.get_x_axis() ?? null;
      y_axis = this.get_y_axis() ?? null;
    } catch {
      x_axis = null;
      y_axis = null;
    }
    const x_range = (x_axis as unknown as { x_range?: number[] } | null)?.x_range;
    const y_range = (y_axis as unknown as { x_range?: number[] } | null)?.x_range;
    if (!x_axis || !y_axis || !x_range || !y_range) {
      this.x_lines = new VGroup();
      this.y_lines = new VGroup();
      return [new VGroup(), new VGroup()];
    }
    const [x_lines1, x_lines2] = this._get_lines_parallel_to_axis(
      x_axis,
      y_axis,
      (y_axis as unknown as { x_range: number[] }).x_range[2] ?? 1,
      this.faded_line_ratio,
    );
    const [y_lines1, y_lines2] = this._get_lines_parallel_to_axis(
      y_axis,
      x_axis,
      (x_axis as unknown as { x_range: number[] }).x_range[2] ?? 1,
      this.faded_line_ratio,
    );
    this.x_lines = x_lines1;
    this.y_lines = y_lines1;
    return [new VGroup(...x_lines1.submobjects as VMobject[], ...y_lines1.submobjects as VMobject[]), new VGroup(...x_lines2.submobjects as VMobject[], ...y_lines2.submobjects as VMobject[])];
  }

  _get_lines_parallel_to_axis(
    axis_parallel_to: CoordinateAxisLike,
    axis_perpendicular_to: CoordinateAxisLike,
    freq: number,
    ratio_faded_lines: number,
  ): [VGroup, VGroup] {
    let ratio = ratio_faded_lines;
    if (ratio === 0) ratio = 1;
    const step = (1 / ratio) * freq;
    const start = (axis_parallel_to as unknown as { get_start: () => number[] }).get_start?.();
    const end = (axis_parallel_to as unknown as { get_end: () => number[] }).get_end?.();
    const lines1 = new VGroup();
    const lines2 = new VGroup();
    if (start === undefined || end === undefined) return [lines1, lines2];
    const line = new Line(start, end);
    const unit_vector_axis_perp_to = axis_perpendicular_to.get_unit_vector();
    const perp_range = (axis_perpendicular_to as unknown as { x_range: number[] }).x_range;
    let x_min = perp_range?.[0] ?? 0;
    let x_max = perp_range?.[1] ?? 0;
    const x_min_scaled = (axis_perpendicular_to as unknown as { x_min?: number }).x_min;
    if (x_min_scaled !== undefined && x_min_scaled > 0 && x_min < 0) {
      x_min = 0;
      x_max = Math.abs(perp_range?.[0] ?? 0) + Math.abs(perp_range?.[1] ?? 0);
    }
    const ranges: number[][] = [
      [0],
      arange(step, Math.min(x_max - x_min, x_max), step),
      arange(-step, Math.max(x_min - x_max, x_min), -step),
    ];
    for (const inputs of ranges) {
      inputs.forEach((x, k) => {
        const new_line = line.copy() as Line;
        new_line.shift(unit_vector_axis_perp_to.map((v) => v * x));
        if ((k + 1) % ratio === 0) lines1.add(new_line);
        else lines2.add(new_line);
      });
    }
    return [lines1, lines2];
  }

  get_vector(coords: number[], kwargs: Record<string, unknown> = {}): Arrow {
    (kwargs as Record<string, unknown>)["buff"] = 0;
    return new Arrow(this.coords_to_point(0, 0), this.coords_to_point(...coords), kwargs as never);
  }

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
    // NOTE: a null length mirrors Python's unit-sized NumberLine axes
    // (NumberPlane/ComplexPlane with x_length=None map coords about the
    // range midpoint at unit size); explicit lengths use the linear map.
    const xValue =
      this.x_length === null || this.x_length === undefined
        ? unitValue(values[0] ?? 0, this.x_range)
        : axisValue(values[0] ?? 0, this.x_range, this.x_length);
    const yValue =
      this.y_length === null || this.y_length === undefined
        ? unitValue(values[1] ?? 0, this.y_range)
        : axisValue(values[1] ?? 0, this.y_range, this.y_length);
    return [xValue, yValue];
  }

  coords_to_point(...coords: (number | number[] | NDArray)[]): NDArray {
    if (coords.length === 0) return array([0, 0, 0]) as NDArray;
    const xLength = this.x_length;
    const yLength = this.y_length;
    // NOTE: null lengths mirror Python's unit-sized axes (see
    // point_to_coords); explicit lengths fall back to the frame-derived
    // sizes only when the fields are null here for legacy callers — Axes
    // always stores resolved lengths, so this branch only affects direct
    // NumberPlane/ComplexPlane use with an explicit length on one axis.
    const resolvedXLength = xLength ?? Math.round(config.frame_width) - 2;
    const resolvedYLength = yLength ?? Math.round(config.frame_height) - 2;
    const mapPoint = (x: number, y: number): number[] => [
      xLength === null || xLength === undefined
        ? unitPosition(Number(x), this.x_range)
        : axisPosition(Number(x), this.x_range, resolvedXLength),
      yLength === null || yLength === undefined
        ? unitPosition(Number(y), this.y_range)
        : axisPosition(Number(y), this.y_range, resolvedYLength),
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
  // axes (plus x_axis/y_axis shortcuts) and shifts the group so the
  // lines-center maps to the origin. The TS CoordinateSystem hierarchy is
  // not Mobject-based yet, so no NumberLine axes are constructed here;
  // the config-merging logic (the observable part of __init__) is ported
  // faithfully and `axes` stays an empty VGroup until the NumberLine
  // integration lands (same limitation as get_axes/point_to_coords note).
  axes: VGroup = new VGroup();
  axis_config: Record<string, unknown> = {};
  x_axis_config: Record<string, unknown> = {};
  y_axis_config: Record<string, unknown> = {};

  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
    axis_config: Record<string, unknown> | null = null,
    x_axis_config: Record<string, unknown> | null = null,
    y_axis_config: Record<string, unknown> | null = null,
    tips = true,
  ) {
    // NOTE: mirrors Python Axes.__init__, whose omitted ranges round the
    // frame radii ([-7, 7, 1] / [-4, 4, 1]); NumberPlane/ComplexPlane keep
    // the raw radii, so the rounding is applied here before delegating.
    super(
      x_range ?? [Math.round(-config.frame_x_radius), Math.round(config.frame_x_radius), 1],
      y_range ?? [Math.round(-config.frame_y_radius), Math.round(config.frame_y_radius), 1],
      x_length ?? Math.round(config.frame_width) - 2,
      y_length ?? Math.round(config.frame_height) - 2,
    );

    this.axis_config = {
      include_tip: tips,
      numbers_to_exclude: [0],
    };
    this.x_axis_config = {};
    this.y_axis_config = {
      rotation: 90 * DEGREES,
      label_direction: [...LEFT],
    };

    Axes._update_default_configs(
      [this.axis_config, this.x_axis_config, this.y_axis_config],
      [axis_config, x_axis_config, y_axis_config],
    );

    this.x_axis_config = merge_dicts_recursively(
      this.axis_config,
      this.x_axis_config,
    );
    this.y_axis_config = merge_dicts_recursively(
      this.axis_config,
      this.y_axis_config,
    );

    // excluding the origin tick removes a tick at the 0-point of the axis
    // This is desired for LinearBase because the 0 point is always the x-axis
    // For non-LinearBase, the "0-point" does not have this quality, so it must be included.
    const xScaling = this.x_axis_config["scaling"] as _ScaleBase | undefined | null;
    this.x_axis_config["exclude_origin_tick"] =
      xScaling === null || xScaling === undefined || xScaling instanceof LinearBase;
    const yScaling = this.y_axis_config["scaling"] as _ScaleBase | undefined | null;
    this.y_axis_config["exclude_origin_tick"] =
      yScaling === null || yScaling === undefined || yScaling instanceof LinearBase;
  }

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
    passed_configs: Array<Record<string, unknown> | null | undefined>,
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
  z_range: number[];
  z_length: number | null;
  z_axis_config: Record<string, unknown> = {};
  z_normal: number[];
  dimension = 3;
  z_axis: unknown = null;

  constructor(
    x_range: number[] | null = null,
    y_range: number[] | null = null,
    x_length: number | null = null,
    y_length: number | null = null,
    light_source: readonly number[] = [-7, -9, 10],
    shaded_axes: ShadedAxis[] = [],
    num_axis_pieces = 20,
    z_range: number[] | null = [-4, 4, 1],
    z_length: number | null = null,
    z_axis_config: Record<string, unknown> | null = null,
    z_normal: readonly number[] = [...DOWN],
    depth: unknown = null,
    gloss = 0.5,
    options: {
      axis_config?: Record<string, unknown> | null;
      x_axis_config?: Record<string, unknown> | null;
      y_axis_config?: Record<string, unknown> | null;
      [key: string]: unknown;
    } = {},
  ) {
    // NOTE: positional order keeps (x_range, y_range, x_length, y_length,
    // light_source, shaded_axes, num_axis_pieces) first for backward
    // compatibility with existing TS callers; the Python __init__ order is
    // (x_range, y_range, z_range, x_length, y_length, z_length, ...), so the
    // z_* params ride as trailing optionals here. Python defaults are
    // x_range=(-6,6,1), y_range=(-5,5,1), x_length/y_length=frame_height+2.5.
    const resolvedXRange = x_range ?? [-6, 6, 1];
    const resolvedYRange = y_range ?? [-5, 5, 1];
    const resolvedXLength = x_length ?? config.frame_height + 2.5;
    const resolvedYLength = y_length ?? config.frame_height + 2.5;
    const resolvedZLength = z_length ?? config.frame_height - 1.5;
    super(resolvedXRange, resolvedYRange, resolvedXLength, resolvedYLength, options.axis_config ?? null, options.x_axis_config ?? null, options.y_axis_config ?? null);
    void depth;
    void gloss;
    this.z_range = z_range === null ? [-4, 4, 1] : [...z_range];
    this.z_length = resolvedZLength;
    this.z_axis_config = {};
    Axes._update_default_configs([this.z_axis_config], [z_axis_config]);
    this.z_axis_config = merge_dicts_recursively(this.axis_config, this.z_axis_config);
    this.z_normal = [...z_normal];
    this.light_source = [...light_source];
    this.shaded_axes = shaded_axes;
    this.num_axis_pieces = num_axis_pieces;
    this.dimension = 3;
    const zScaling = this.z_axis_config["scaling"] as _ScaleBase | undefined | null;
    this.z_axis_config["exclude_origin_tick"] =
      zScaling === null || zScaling === undefined || zScaling instanceof LinearBase;
    // NOTE: Python builds the z-axis NumberLine here (_create_axis), rotates
    // it about its origin (-PI/2 about UP, then angle_of_vector(z_normal)),
    // and shifts it onto the x-axis origin. The TS Axes hierarchy is not
    // Mobject-based yet (no NumberLine axes are constructed), so there is no
    // axis object to rotate/shift; z_axis stays null until that integration
    // lands.
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

  get_y_axis_label(
    label: number | string | VMobject,
    edge: readonly number[] | NDArray = [...UR],
    direction: readonly number[] | NDArray = [...UR],
    buff = SMALL_BUFF,
    rotation: number | Record<string, unknown> = PI / 2,
    rotation_axis: readonly number[] | NDArray = [...OUT],
    kwargs: Record<string, unknown> = {},
  ): Mobject {
    // NOTE: `rotation` accepts a kwargs record in the 5th position so this
    // override stays assignable to the base
    // get_y_axis_label(label, edge, direction, buff, kwargs) signature.
    let resolvedRotation: number = PI / 2;
    let resolvedKwargs: Record<string, unknown> = kwargs;
    if (typeof rotation === "number") resolvedRotation = rotation;
    else resolvedKwargs = { ...rotation, ...kwargs };
    void resolvedKwargs;
    const positioned_label = this._get_axis_label(label, this.get_y_axis(), edge, direction, buff);
    positioned_label.rotate(resolvedRotation, rotation_axis);
    return positioned_label;
  }

  get_z_axis_label(
    label: number | string | VMobject,
    edge: readonly number[] | NDArray = [...OUT],
    direction: readonly number[] | NDArray = [...RIGHT],
    buff = SMALL_BUFF,
    rotation: number = PI / 2,
    rotation_axis: readonly number[] | NDArray = [...RIGHT],
    kwargs: Record<string, unknown> = {},
  ): Mobject {
    void kwargs;
    const positioned_label = this._get_axis_label(label, this.get_z_axis(), edge, direction, buff);
    positioned_label.rotate(rotation, rotation_axis);
    return positioned_label;
  }

  get_axis_labels(
    x_label: number | string | VMobject = "x",
    y_label: number | string | VMobject = "y",
    z_label: number | string | VMobject = "z",
  ): VGroup {
    const labels = new VGroup(
      this.get_x_axis_label(x_label) as unknown as VMobject,
      this.get_y_axis_label(y_label) as unknown as VMobject,
      this.get_z_axis_label(z_label) as unknown as VMobject,
    );
    (this as unknown as { axis_labels: VGroup }).axis_labels = labels;
    return labels;
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
  // NOTE: always False after construction (see the VGroup-overwrite note in
  // the constructor); the initializer matches that net Python behavior.
  make_smooth_after_applying_functions = false;
  azimuth_offset: number;
  azimuth_label_buff: number;
  azimuth_label_font_size: number;
  azimuth_compact_fraction: boolean;
  override background_lines: VGroup | null = null;
  override faded_lines: VGroup | null = null;

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
    // NOTE: mirrors Python, which passes x_length=size / y_length=size
    // (None by default) straight through Axes.__init__ to
    // CoordinateSystem.__init__ (stored as-is). The TS Axes constructor
    // substitutes frame-derived defaults for null, so restore the
    // pass-through values here; null lengths select the unit-sized
    // midpoint-centered mapping in coords_to_point/point_to_coords.
    this.x_length = size;
    this.y_length = size;
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
    // NOTE: mirrors Python `super().__init__(..., axis_config=self.radius_config,
    // **kwargs)`: the radius config becomes the Axes-level axis config, so it
    // is merged into axis_config/x_axis_config/y_axis_config here (after
    // super(), which TS requires first). Known Axes options keys forwarded
    // from the trailing options bag win over it, mirroring **kwargs.
    const {
      radius_config: _droppedRadiusConfig,
      background_line_style: _droppedBackgroundLineStyle,
      faded_line_style: _droppedFadedLineStyle,
      faded_line_ratio: _droppedFadedLineRatio,
      axis_config: axisConfigOption,
      x_axis_config: xAxisConfigOption,
      y_axis_config: yAxisConfigOption,
      tips: tipsOption,
      ..._ignoredOptions
    } = options;
    void _droppedRadiusConfig;
    void _droppedBackgroundLineStyle;
    void _droppedFadedLineStyle;
    void _droppedFadedLineRatio;
    void _ignoredOptions;
    Axes._update_default_configs(
      [this.axis_config, this.x_axis_config, this.y_axis_config],
      [
        this.radius_config,
        (axisConfigOption as Record<string, unknown> | undefined) ?? null,
        (axisConfigOption as Record<string, unknown> | undefined) ?? null,
      ],
    );
    if (xAxisConfigOption !== undefined && xAxisConfigOption !== null) {
      update_dict_recursively(this.x_axis_config, xAxisConfigOption as Record<string, unknown>);
    }
    if (yAxisConfigOption !== undefined && yAxisConfigOption !== null) {
      update_dict_recursively(this.y_axis_config, yAxisConfigOption as Record<string, unknown>);
    }
    this.x_axis_config = merge_dicts_recursively(this.axis_config, this.x_axis_config);
    this.y_axis_config = merge_dicts_recursively(this.axis_config, this.y_axis_config);
    void tipsOption;
    // NOTE: mirrors Python _init_background_lines' first half: when no faded
    // style is given, it defaults to a faded copy of background_line_style
    // with every numeric value halved (stroke_width 2 -> 1.0, opacity 1 ->
    // 0.5; colors kept). Only the line *construction* half is deferred (see
    // below); the style derivation is pure dict math so it is ported.
    // (typeof boolean is not "number" here, unlike Python's
    // isinstance(bool, numbers.Number); a boolean style value would halve to
    // 0.5 in Python but is kept as-is here.)
    if (faded_line_style === null || faded_line_style === undefined) {
      const derived: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(this.background_line_style)) {
        derived[key] = typeof value === "number" ? value * 0.5 : value;
      }
      this.faded_line_style = derived;
    } else {
      this.faded_line_style = faded_line_style;
    }
    this.faded_line_ratio = faded_line_ratio;
    // NOTE: mirrors Python, where PolarPlane sets this field to True and then
    // Axes.__init__ runs VGroup.__init__(**kwargs), which unconditionally
    // overwrites it with the VMobject default False (the named parameter never
    // reaches VGroup since it binds in PolarPlane.__init__). Net Python
    // behavior: always False, even when True is passed explicitly.
    void make_smooth_after_applying_functions;
    this.make_smooth_after_applying_functions = false;
    this.azimuth_offset = azimuth_offset;
    this.azimuth_label_buff = azimuth_label_buff;
    this.azimuth_label_font_size = azimuth_label_font_size;
    this.azimuth_compact_fraction = azimuth_compact_fraction;
    // NOTE: Python calls self._init_background_lines() here (Circle/Line
    // background construction over the Mobject family). The TS
    // CoordinateSystem hierarchy is not Mobject-based yet, so background
    // line construction is deferred until that integration lands;
    // background_lines/faded_lines stay null (the NumberPlane super()
    // constructor above populated them with empty groups, so reset here).
    this.background_lines = null;
    this.faded_lines = null;
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

  _get_default_coordinate_values(): Array<number | ComplexPoint> {
    const x_numbers = (this.get_x_axis().get_tick_range().toArray() as unknown[]).map(Number);
    const y_numbers = (this.get_y_axis().get_tick_range().toArray() as unknown[]).map(Number);
    const y_complex: ComplexPoint[] = y_numbers
      .filter((y) => y !== 0)
      .map((y) => ({ re: 0, im: y }));
    return [...x_numbers, ...y_complex];
  }

  get_coordinate_labels(
    ...args: Array<ComplexNumberLike | Record<string, unknown>>
  ): VGroup {
    // NOTE: mirrors Python get_coordinate_labels(*numbers, **kwargs): a
    // trailing plain object is the kwargs bag. Each label is built with
    // NumberLine.get_number_mobject (a DecimalNumber, i.e. text/tex, out of
    // scope), so calls throw there with a TODO until text lands.
    let kwargs: Record<string, unknown> = {};
    let numbers: ComplexNumberLike[] = args as ComplexNumberLike[];
    const last = args[args.length - 1];
    if (last !== null && typeof last === "object" && !Array.isArray(last) && !("re" in (last as Record<string, unknown>) && "im" in (last as Record<string, unknown>))) {
      kwargs = last as Record<string, unknown>;
      numbers = (args.slice(0, -1) as ComplexNumberLike[]);
    }
    if (numbers.length === 0) {
      numbers = this._get_default_coordinate_values();
    }

    this.coordinate_labels = new VGroup();
    for (const number of numbers) {
      const z = toComplexPoint(number);
      let axis: CoordinateAxisLike;
      let value: number;
      if (Math.abs(z.im) > Math.abs(z.re)) {
        axis = this.get_y_axis() as unknown as CoordinateAxisLike;
        value = z.im;
        kwargs = { ...kwargs, unit: "i" };
      } else {
        axis = this.get_x_axis() as unknown as CoordinateAxisLike;
        value = z.re;
      }
      const number_mob = axis.get_number_mobject(value, null, null, null, null, kwargs);
      this.coordinate_labels.add(number_mob as unknown as VMobject);
    }
    return this.coordinate_labels;
  }

  add_coordinates(
    ...args: Array<ComplexNumberLike | Record<string, unknown>>
  ): this {
    // NOTE: mirrors Python `self.add(self.get_coordinate_labels(*numbers,
    // **kwargs))`; the TS CoordinateSystem hierarchy is not Mobject-based
    // yet (see the Axes note), so there is no `add` — storing the labels on
    // coordinate_labels (as CoordinateSystem.add_coordinates does) is the
    // observable part.
    this.get_coordinate_labels(...args);
    return this;
  }
}

