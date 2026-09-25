import { array, type NDArray } from "numpy-ts";
import { Animation, classDefaultsOf, type AnimationOptions, type RateFunction } from "./animation.js";
import { Mobject } from "../mobject/mobject.js";
import { linear } from "../utils/rate_functions.js";
import { pointsFrom, pointFrom, pointRows, isNDArray, nestedRows, addPoints, scalePoint } from "../utils/arrays.js";

export type HomotopyFunction = (x: number, y: number, z: number, t: number) => number[] | NDArray | { re: number; im: number } | { real: number; imag: number } | number;

export class Homotopy extends Animation {
  homotopy: HomotopyFunction;
  apply_function_kwargs: Record<string, unknown>;

  constructor(homotopy: HomotopyFunction, mobject: Mobject, options: AnimationOptions & { apply_function_kwargs?: Record<string, unknown> } = {}) {
    super(mobject, { run_time: 3, ...options });
    this.homotopy = homotopy;
    this.apply_function_kwargs = options.apply_function_kwargs ?? {};
  }

  function_at_time_t(t: number): (point: number[] | NDArray | { re: number; im: number } | { real: number; imag: number }) => NDArray {
    const mapping_function = (point: number[] | NDArray | { re: number; im: number } | { real: number; imag: number }): NDArray => {
      let values: number[];
      if (isNDArray(point)) values = (nestedRows(point)[0] as unknown[] | undefined)?.map(Number) ?? [];
      else if (Array.isArray(point)) values = point.map(Number);
      else if ("re" in point) values = [point.re, point.im, 0];
      else values = [point.real, point.imag, 0];
      if (values.length !== 3) {
        const error = new Error(`not enough values to unpack (expected 3, got ${values.length})`);
        error.name = "ValueError";
        throw error;
      }
      const [x, y, z] = values;
      const result = this.homotopy(x, y, z, t);
      if (isNDArray(result)) return result.copy();
      if (typeof result === "number") return array([result, 0, z]) as NDArray;
      if (Array.isArray(result)) return array(result) as NDArray;
      return array("re" in result ? [result.re, result.im, z] : [result.real, result.imag, z]) as NDArray;
    };
    return mapping_function;
  }

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): this {
    submobject.points = startingSubmobject.points;
    submobject.apply_function(this.function_at_time_t(alpha), this.apply_function_kwargs.about_point as number[] | undefined, this.apply_function_kwargs.about_edge as number[] | undefined);
    return this;
  }
}

export class SmoothedVectorizedHomotopy extends Homotopy {
  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): this {
    super.interpolate_submobject(submobject, startingSubmobject, alpha);
    const vmobject = submobject as Mobject & { make_smooth?: () => void };
    vmobject.make_smooth?.();
    return this;
  }
}

export class ComplexHomotopy extends Homotopy {
  constructor(complex_homotopy: (value: { re: number; im: number }, t: number) => { re: number; im: number } | { real: number; imag: number } | number | number[] | NDArray, mobject: Mobject, options: AnimationOptions = {}) {
    const homotopy: HomotopyFunction = (x, y, _z, t) => {
      const result = complex_homotopy({ re: x, im: y }, t);
      if (typeof result === "number") return [result, 0, _z];
      if (isNDArray(result)) return pointRows(result, 3)[0] ?? [_z, _z, _z];
      if (Array.isArray(result)) return result;
      return "re" in result ? [result.re, result.im, _z] : [result.real, result.imag, _z];
    };
    super(homotopy, mobject, options);
  }
}

export class MoveAlongPath extends Animation {
  path: Mobject & { point_from_proportion: (alpha: number) => NDArray };

  constructor(mobject: Mobject, path: Mobject & { point_from_proportion: (alpha: number) => NDArray }, options: AnimationOptions = {}) {
    super(mobject, { suspend_mobject_updating: false, ...options });
    this.path = path;
  }

  interpolate_mobject(alpha: number): void {
    this.mobject.move_to(this.path.point_from_proportion(this.rate_func(alpha)));
  }
}

export interface PhaseFlowOptions extends AnimationOptions {
  virtual_time?: number;
}

export class PhaseFlow extends Animation {
  virtual_time: number;
  function: (point: number[]) => number[];
  last_alpha?: number;

  constructor(
    function_: (point: number[]) => number[],
    mobject: Mobject,
    options: PhaseFlowOptions = {},
  ) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      virtual_time = (clsDefaults.virtual_time as number | undefined) ?? 1,
      suspend_mobject_updating = (clsDefaults.suspend_mobject_updating as boolean | undefined) ?? false,
      rate_func = (clsDefaults.rate_func as RateFunction | undefined) ?? linear,
      ...rest
    } = options;
    super(mobject, { ...rest, suspend_mobject_updating, rate_func });
    this.virtual_time = virtual_time;
    this.function = function_;
  }

  override interpolate_mobject(alpha: number): void {
    if (this.last_alpha !== undefined) {
      const dt = this.virtual_time * (this.rate_func(alpha) - this.rate_func(this.last_alpha));
      const fn = this.function;
      this.mobject.apply_function((point) => addPoints(point, scalePoint(fn([...point]), dt, this.mobject.dim), this.mobject.dim));
    }
    this.last_alpha = alpha;
  }
}
