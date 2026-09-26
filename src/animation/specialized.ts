import { LaggedStart, type AnimationGroupOptions } from "./composition.js";
import { Restore } from "./transform.js";
import type { Mobject } from "../mobject/mobject.js";
import { ORIGIN } from "../constants.js";

export interface BroadcastOptions extends AnimationGroupOptions {
  focal_point?: readonly number[];
  n_mobs?: number;
  initial_opacity?: number;
  final_opacity?: number;
  initial_width?: number;
  remover?: boolean;
  lag_ratio?: number;
  run_time?: number;
}

export class Broadcast extends LaggedStart {
  focal_point: readonly number[];
  n_mobs: number;
  initial_opacity: number;
  final_opacity: number;
  initial_width: number;

  constructor(
    mobject: Mobject,
    focal_point: readonly number[] = [...ORIGIN],
    n_mobs = 5,
    initial_opacity = 1,
    final_opacity = 0,
    initial_width = 0.0,
    remover = true,
    lag_ratio = 0.2,
    run_time = 3,
    options: BroadcastOptions = {},
  ) {
    // NOTE: mirrors Python, which assigns the scalar fields before
    // `super().__init__()`; TS requires super() first, so the animation list
    // is built from locals here and the fields are assigned afterwards.
    const {
      focal_point: focalPointOption,
      n_mobs: nMobsOption,
      initial_opacity: initialOpacityOption,
      final_opacity: finalOpacityOption,
      initial_width: initialWidthOption,
      remover: removerOption,
      lag_ratio: lagRatioOption,
      run_time: runTimeOption,
      ...rest
    } = options;
    const resolvedFocalPoint = focalPointOption ?? focal_point;
    const resolvedNMobs = nMobsOption ?? n_mobs;
    const resolvedInitialOpacity = initialOpacityOption ?? initial_opacity;
    const resolvedFinalOpacity = finalOpacityOption ?? final_opacity;
    const resolvedInitialWidth = initialWidthOption ?? initial_width;
    const resolvedRemover = removerOption ?? remover;
    const resolvedLagRatio = lagRatioOption ?? lag_ratio;
    const resolvedRunTime = runTimeOption ?? run_time;

    // NOTE: mirrors Python `fill_o = bool(mobject.fill_opacity)`.
    const fillO = Boolean((mobject as Mobject & { fill_opacity?: number }).fill_opacity);
    const animations: Restore[] = [];
    for (let index = 0; index < resolvedNMobs; index += 1) {
      const mob = mobject.copy();
      if (fillO) {
        mob.set_opacity(resolvedFinalOpacity);
      } else {
        mob.set_stroke(undefined, undefined, resolvedFinalOpacity);
      }
      mob.move_to(resolvedFocalPoint);
      mob.save_state();
      // NOTE: mirrors Python `mob.set(width=self.initial_width)` (the generic
      // attribute setter hitting the `width` property) — the TS `width`
      // setter scales to fit the same way.
      (mob as Mobject & { width: number }).width = resolvedInitialWidth;
      if (fillO) {
        mob.set_opacity(resolvedInitialOpacity);
      } else {
        mob.set_stroke(undefined, undefined, resolvedInitialOpacity);
      }
      animations.push(new Restore(mob, { remover: resolvedRemover }));
    }

    super(...animations, { ...rest, run_time: resolvedRunTime, lag_ratio: resolvedLagRatio });
    this.focal_point = resolvedFocalPoint;
    this.n_mobs = resolvedNMobs;
    this.initial_opacity = resolvedInitialOpacity;
    this.final_opacity = resolvedFinalOpacity;
    this.initial_width = resolvedInitialWidth;
  }
}
