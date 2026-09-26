import type { NDArray } from "numpy-ts";
import { Animation, classDefaultsOf, type AnimationOptions, type RateFunction } from "./animation.js";
import { Transform, type TransformOptions } from "./transform.js";
import type { Mobject } from "../mobject/mobject.js";
import { OUT, PI, TAU } from "../constants.js";
import { linear } from "../utils/rate_functions.js";

export interface RotatingOptions extends AnimationOptions {
  angle?: number;
  axis?: readonly number[] | NDArray;
  about_point?: readonly number[] | NDArray | null;
  about_edge?: readonly number[] | NDArray | null;
}

export class Rotating extends Animation {
  angle: number;
  axis: readonly number[] | NDArray;
  about_point: readonly number[] | NDArray | null;
  about_edge: readonly number[] | NDArray | null;

  constructor(
    mobject: Mobject,
    angle?: number,
    axis?: readonly number[] | NDArray,
    about_point?: readonly number[] | NDArray | null,
    about_edge?: readonly number[] | NDArray | null,
    run_time?: number,
    rate_func?: RateFunction,
    options: RotatingOptions = {},
  ) {
    const clsDefaults = classDefaultsOf(new.target ?? Rotating);
    const resolvedAngle = angle ?? (options.angle as number | undefined) ?? (clsDefaults.angle as number | undefined) ?? TAU;
    const resolvedAxis = axis ?? (options.axis as readonly number[] | NDArray | undefined) ?? (clsDefaults.axis as readonly number[] | NDArray | undefined) ?? [...OUT];
    const resolvedAboutPoint = about_point ?? (options.about_point as readonly number[] | NDArray | null | undefined) ?? (clsDefaults.about_point as readonly number[] | NDArray | null | undefined) ?? null;
    const resolvedAboutEdge = about_edge ?? (options.about_edge as readonly number[] | NDArray | null | undefined) ?? (clsDefaults.about_edge as readonly number[] | NDArray | null | undefined) ?? null;
    const resolvedRunTime = run_time ?? (options.run_time as number | undefined) ?? (clsDefaults.run_time as number | undefined) ?? 5;
    const resolvedRateFunc = rate_func ?? (options.rate_func as RateFunction | undefined) ?? (clsDefaults.rate_func as RateFunction | undefined) ?? linear;
    const { angle: _a, axis: _b, about_point: _c, about_edge: _d, run_time: _e, rate_func: _f, ...rest } = options;
    void _a; void _b; void _c; void _d; void _e; void _f;
    super(mobject, { ...rest, run_time: resolvedRunTime, rate_func: resolvedRateFunc });
    this.angle = resolvedAngle;
    this.axis = resolvedAxis;
    this.about_point = resolvedAboutPoint;
    this.about_edge = resolvedAboutEdge;
  }

  override interpolate_mobject(alpha: number): void {
    this.mobject.become(this.starting_mobject);
    this.mobject.rotate(
      this.rate_func(alpha) * this.angle,
      this.axis,
      this.about_point ?? undefined,
      this.about_edge ?? undefined,
    );
  }
}

export interface RotateOptions extends TransformOptions {
  angle?: number;
  axis?: readonly number[] | NDArray;
  about_point?: readonly number[] | NDArray | null;
  about_edge?: readonly number[] | NDArray | null;
}

export class Rotate extends Transform {
  angle: number;
  axis: readonly number[] | NDArray;
  about_point: readonly number[] | NDArray | null;
  about_edge: readonly number[] | NDArray | null;

  constructor(
    mobject: Mobject,
    angle?: number,
    axis?: readonly number[] | NDArray,
    about_point?: readonly number[] | NDArray | null,
    about_edge?: readonly number[] | NDArray | null,
    options: RotateOptions = {},
  ) {
    const { angle: _a, axis: _b, about_point: _c, about_edge: _d, path_arc, path_arc_axis, ...rest } = options;
    void _a; void _b; void _c; void _d;
    const resolvedAngle = angle ?? (options.angle as number | undefined) ?? PI;
    const resolvedAxis = axis ?? (options.axis as readonly number[] | NDArray | undefined) ?? [...OUT];
    // NOTE: TS cannot touch `this` before `super()`; the about_point
    // default (mobject.get_center()) is resolved here into a local first.
    // Mirrors Python: explicit kwargs / positional value wins, otherwise
    // the mobject center.
    const resolvedAboutPoint = about_point ?? (options.about_point as readonly number[] | NDArray | null | undefined) ?? mobject.get_center();
    const resolvedAboutEdge = about_edge ?? (options.about_edge as readonly number[] | NDArray | null | undefined) ?? null;
    super(mobject, {
      ...rest,
      path_arc: (path_arc as number | undefined) ?? resolvedAngle,
      path_arc_axis: (path_arc_axis as readonly number[] | NDArray | undefined) ?? resolvedAxis,
      path_arc_centers: resolvedAboutPoint,
    });
    this.angle = resolvedAngle;
    this.axis = resolvedAxis;
    this.about_edge = resolvedAboutEdge;
    this.about_point = resolvedAboutPoint;
  }

  override create_target(): Mobject {
    const target = this.mobject.copy();
    target.rotate(
      this.angle,
      this.axis,
      this.about_point ?? undefined,
      this.about_edge ?? undefined,
    );
    return target;
  }
}
