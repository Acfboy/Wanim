import { array, type NDArray } from "numpy-ts";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { config } from "../../_config/utils.js";
import type { ParsableManimColor } from "../../utils/color/core.js";
import { arange, isNDArray, numeric, pointRows } from "../../utils/arrays.js";
import { LinearBase, type _ScaleBase } from "./scale.js";

export type ParametricFunctionT = (t: number) => number[] | readonly number[] | NDArray;

export interface ParametricFunctionOptions extends VMobjectOptions {
  t_range?: [number, number] | [number, number, number];
  scaling?: _ScaleBase;
  dt?: number;
  discontinuities?: number[] | null;
  use_smoothing?: boolean;
  use_vectorized?: boolean;
}

export class ParametricFunction extends VMobject {
  function: (t: number) => NDArray;
  scaling: _ScaleBase;
  dt: number;
  discontinuities: number[] | null;
  use_smoothing: boolean;
  use_vectorized: boolean;
  t_min: number;
  t_max: number;
  t_step: number;
  // NOTE: set dynamically by CoordinateSystem.plot / plot_parametric_curve /
  // plot_polar_graph (mirrors Python's `graph.underlying_function = ...`);
  // read by input_to_graph_point/coords and plot_antiderivative_graph.
  underlying_function?: (t: number) => unknown;

  constructor(function_: ParametricFunctionT, options: ParametricFunctionOptions = {}) {
    const {
      t_range = [0, 1] as [number, number],
      scaling = new LinearBase(),
      dt = 1e-8,
      discontinuities = null,
      use_smoothing = true,
      use_vectorized = false,
      ...rest
    } = options;
    super(rest);

    function internal_parametric_function(t: number): NDArray {
      return array(function_(t)) as NDArray;
    }

    this.function = internal_parametric_function;
    const full_range: [number, number, number] = t_range.length === 2
      ? [t_range[0] ?? 0, t_range[1] ?? 1, 0.01]
      : [t_range[0] ?? 0, t_range[1] ?? 1, t_range[2] ?? 0.01];
    this.scaling = scaling;
    this.dt = dt;
    this.discontinuities = discontinuities === null ? null : [...discontinuities];
    this.use_smoothing = use_smoothing;
    this.use_vectorized = use_vectorized;
    [this.t_min, this.t_max, this.t_step] = full_range;
    // NOTE: Python sets these fields before super().__init__() so that the
    // generate_points() call inside the base constructor already sees them.
    // TS requires super() first; the base-constructor call is guarded as a
    // no-op below and points are (re)generated here instead.
    this.generate_points();
  }

  get_function(): (t: number) => NDArray {
    return this.function;
  }

  get_point_from_function(t: number): NDArray {
    return this.function(t);
  }

  override generate_points(): this {
    // Guard: the base Mobject constructor invokes generate_points() before
    // this class assigns t_min/t_max/t_step (TS requires super() first,
    // whereas Python sets them before super().__init__()). Treat that early
    // call as a no-op; the constructor regenerates afterwards.
    if (this.t_min === undefined || this.t_max === undefined || this.t_step === undefined) return this;
    let boundary_times: number[];
    if (this.discontinuities !== null) {
      const discs = this.discontinuities.filter((t) => this.t_min <= t && t <= this.t_max);
      boundary_times = [
        this.t_min,
        this.t_max,
        ...discs.map((t) => t - this.dt),
        ...discs.map((t) => t + this.dt),
      ].sort((a, b) => a - b);
    } else {
      boundary_times = [this.t_min, this.t_max];
    }
    for (let i = 0; i + 1 < boundary_times.length; i += 2) {
      const t1 = boundary_times[i] ?? 0;
      const t2 = boundary_times[i + 1] ?? 0;
      // Mirrors `np.array([*scaling.function(np.arange(t1, t2, t_step)),
      // scaling.function(t2)])`. NOTE: the scaled samples form a flat 1-D
      // list (not one row per sample), so they are flattened element-wise
      // rather than read via pointRows.
      const scaledOut = this.scaling.function(arange(t1, t2, this.t_step));
      const scaledRaw: unknown = isNDArray(scaledOut) ? scaledOut.toArray() : scaledOut;
      const scaled: number[] = [];
      const collect = (v: unknown): void => {
        if (Array.isArray(v)) for (const item of v) collect(item);
        else scaled.push(numeric(v));
      };
      collect(scaledRaw);
      const endValue = this.scaling.function(t2);
      const t_values = [
        ...scaled,
        typeof endValue === "number" ? endValue : (pointRows(endValue, 1)[0]?.[0] ?? 0),
      ];
      let point_list: number[][];
      if (this.use_vectorized) {
        // Python calls self.function(t_range) once with the whole array;
        // accept a vectorized result, else fall back to per-sample calls.
        const out = (this.function as unknown as (t: unknown) => unknown)(array(t_values));
        if (typeof out !== "object" || out === null || !("shape" in (out as Record<string, unknown>))) {
          point_list = t_values.map(
            (t) => pointRows(this.get_point_from_function(t), this.dim)[0] ?? new Array(this.dim).fill(0),
          );
        } else {
          const asArray = (out as NDArray).toArray() as unknown;
          point_list = Array.isArray(asArray) && Array.isArray(asArray[0])
            ? (asArray as unknown[][]).map((row) => (row as unknown[]).map(Number))
            : t_values.map(
              (t) => pointRows(this.get_point_from_function(t), this.dim)[0] ?? new Array(this.dim).fill(0),
            );
        }
      } else {
        // NOTE: Python calls self.function(t) here; for FunctionGraph that
        // is the wrapped parametric form, so the polymorphic
        // get_point_from_function (which yields the parametric form for
        // FunctionGraph) is used instead.
        point_list = t_values.map(
          (t) => pointRows(this.get_point_from_function(t), this.dim)[0] ?? new Array(this.dim).fill(0),
        );
      }
      this.start_new_path(point_list[0]);
      this.add_points_as_corners(point_list.slice(1));
    }
    if (this.use_smoothing) {
      // NOTE: VMobject.make_smooth is translated in a later batch; smooth here once available.
      // (The Mobject proxy throws AttributeError on missing members, so `in` must guard the access.)
      if ("make_smooth" in this) {
        (this as unknown as { make_smooth: () => void }).make_smooth();
      }
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }
}

export type FunctionGraphFunction = (x: number) => number | number[] | readonly number[] | NDArray;

export interface FunctionGraphOptions extends VMobjectOptions {
  discontinuities?: number[] | null;
  use_smoothing?: boolean;
  use_vectorized?: boolean;
  scaling?: _ScaleBase;
  dt?: number;
}

export class FunctionGraph extends ParametricFunction {
  x_range: [number, number] | [number, number, number];
  parametric_function: (t: number) => NDArray;

  constructor(
    function_: FunctionGraphFunction,
    x_range: [number, number] | [number, number, number] | null = null,
    color: ParsableManimColor = "PURE_YELLOW",
    options: FunctionGraphOptions = {},
  ) {
    const resolved_range: [number, number] | [number, number, number] = x_range ?? [
      -config.frame_x_radius,
      config.frame_x_radius,
    ];
    // NOTE: mirrors Python `lambda t: np.array([t, function(t), 0])`.
    // Python assigns this wrapped form to ParametricFunction (which stores
    // it as self.function) and ALSO keeps it as self.parametric_function;
    // self.function is therefore the wrapped form (NOT the raw user
    // function) after construction. The TS port does the same: super()
    // stores the wrapped form as self.function; only parametric_function
    // is (re)assigned below.
    const parametric_function = (t: number): NDArray =>
      array([t, function_(t) as number, 0]) as NDArray;
    // NOTE: Python stores the raw user function as self.function and passes
    // the wrapped parametric form to ParametricFunction. The TS
    // ParametricFunction constructor takes an options object (not positional
    // t_range/color), so forward them as options.
    const {
      discontinuities = null,
      use_smoothing = true,
      use_vectorized = false,
      scaling = new LinearBase(),
      dt = 1e-8,
      ...rest
    } = options;
    super(parametric_function, {
      t_range: resolved_range,
      scaling,
      dt,
      discontinuities,
      use_smoothing,
      use_vectorized,
      color,
      ...rest,
    });
    this.x_range = resolved_range;
    this.parametric_function = parametric_function;
    // NOTE: super() already stored the wrapped parametric form as
    // self.function (exactly like Python, where ParametricFunction.__init__
    // overwrites the raw assignment); get_function() therefore returns the
    // wrapped form and get_point_from_function the parametric point.
    // Regenerate now that x_range/parametric_function are assigned (see the
    // super-first note on ParametricFunction above).
    this.generate_points();
  }

  override get_function(): (t: number) => NDArray {
    return this.function;
  }

  override get_point_from_function(x: number): NDArray {
    return this.parametric_function(x);
  }

  override generate_points(): this {
    // Guard: the ParametricFunction constructor invokes generate_points()
    // before this class assigns parametric_function/x_range (TS requires
    // super() first, whereas Python sets them before super().__init__()).
    // Treat that early call as a no-op; the constructor regenerates below.
    // NOTE: `in` is used instead of a plain undefined comparison because
    // Mobject instances are Proxies that throw AttributeError on undeclared
    // attribute reads during construction.
    if (!("parametric_function" in this) || this.parametric_function === undefined) return this;
    if (!("x_range" in this) || this.x_range === undefined) return this;
    return super.generate_points();
  }
}

export type ImplicitFunctionT = (x: number, y: number) => number;

export interface ImplicitFunctionOptions extends VMobjectOptions {
  x_range?: number[] | null;
  y_range?: number[] | null;
  min_depth?: number;
  max_quads?: number;
  use_smoothing?: boolean;
}

export type IsolineComputer = (fn: (p: [number, number]) => number, pmin: [number, number], pmax: [number, number], min_depth: number, max_quads: number) => number[][][];

let isolineComputer: IsolineComputer | null = null;

export function setIsolineComputer(computer: IsolineComputer | null): void {
  isolineComputer = computer;
}

export class ImplicitFunction extends VMobject {
  function: ImplicitFunctionT;
  min_depth: number;
  max_quads: number;
  use_smoothing: boolean;
  x_range: number[];
  y_range: number[];
  private __constructed = false;

  constructor(func: ImplicitFunctionT, options: ImplicitFunctionOptions = {}) {
    const {
      x_range = null,
      y_range = null,
      min_depth = 5,
      max_quads = 1500,
      use_smoothing = true,
      ...rest
    } = options;
    super(rest);
    this.function = func;
    this.min_depth = min_depth;
    this.max_quads = max_quads;
    this.use_smoothing = use_smoothing;
    this.x_range = x_range !== null ? [...x_range] : [
      -config.frame_width / 2,
      config.frame_width / 2,
    ];
    this.y_range = y_range !== null ? [...y_range] : [
      -config.frame_height / 2,
      config.frame_height / 2,
    ];
    this.__constructed = true;
    // The base Mobject constructor already called generate_points(), but at
    // that time the ranges were not yet assigned (TS requires super() first,
    // whereas Python sets them before super().__init__()). Regenerate now
    // that the fields are available so points are populated at construction.
    // (No-op while no isoline computer is registered; see generate_points.)
    if (isolineComputer !== null) this.generate_points();
  }

  generate_points(): this {
    // Guard: the base Mobject constructor invokes generate_points() before
    // this class assigns x_range/y_range (TS requires super() first, whereas
    // Python sets them before super().__init__()). Treat that early call as
    // a no-op; the constructor regenerates afterwards.
    if (this.x_range === undefined || this.y_range === undefined) return this;
    if (isolineComputer === null) {
      if (!this.__constructed) return this;
      throw new Error("TODO: ImplicitFunction.generate_points needs the `isosurfaces` plot_isoline routine, not ported yet");
    }
    const pmin: [number, number] = [this.x_range[0] ?? 0, this.y_range[0] ?? 0];
    const pmax: [number, number] = [this.x_range[1] ?? 0, this.y_range[1] ?? 0];
    const curves = isolineComputer(
      (u) => this.function(u[0] ?? 0, u[1] ?? 0),
      pmin,
      pmax,
      this.min_depth,
      this.max_quads,
    )
      .filter((curve) => curve.length > 0)
      .map((curve) => curve.map((point) => [point[0] ?? 0, point[1] ?? 0, 0]));
    for (const curve of curves) {
      if (curve.length === 0) continue;
      this.start_new_path(curve[0]);
      this.add_points_as_corners(curve.slice(1));
    }
    if (this.use_smoothing) {
      // NOTE: VMobject.make_smooth is translated in a later batch; smooth here once available.
      // (The Mobject proxy throws AttributeError on missing members, so `in` must guard the access.)
      if ("make_smooth" in this) {
        (this as unknown as { make_smooth: () => void }).make_smooth();
      }
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }
}
