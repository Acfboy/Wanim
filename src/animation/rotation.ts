import type { NDArray } from "numpy-ts";
import { Animation, classDefaultsOf, type AnimationOptions, type RateFunction } from "./animation.js";
import type { Mobject } from "../mobject/mobject.js";
import { OUT, TAU } from "../constants.js";
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
