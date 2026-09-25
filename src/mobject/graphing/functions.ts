import { array, type NDArray } from "numpy-ts";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
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
  }

  get_function(): (t: number) => NDArray {
    return this.function;
  }

  get_point_from_function(t: number): NDArray {
    return this.function(t);
  }
}
