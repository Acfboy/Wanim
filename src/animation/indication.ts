import { smooth, type AnimationOptions, type RateFunction } from "./animation.js";
import { ShowPartial, type ShowPartialOptions } from "./creation.js";
import { Homotopy } from "./movement.js";
import { type Mobject } from "../mobject/mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { interpolate, inverse_interpolate } from "../utils/bezier.js";
import { normalize } from "../utils/space_ops.js";
import { UP } from "../constants.js";

export class ShowPassingFlash extends ShowPartial {
  time_width: number;

  constructor(mobject: VMobject, time_width = 0.1, options: ShowPartialOptions = {}) {
    super(mobject, { remover: true, introducer: true, ...options });
    this.time_width = time_width;
  }

  _get_bounds(alpha: number): [number, number] {
    const tw = this.time_width;
    let upper = interpolate(0, 1 + tw, alpha) as number;
    let lower = upper - tw;
    upper = Math.min(upper, 1);
    lower = Math.max(lower, 0);
    return [lower, upper];
  }

  clean_up_from_scene(scene: { remove: (mobject: Mobject) => void }): void {
    super.clean_up_from_scene(scene);
    for (const [submob, start] of this.get_all_families_zipped()) {
      (submob as VMobject).pointwise_become_partial(start as VMobject, 0, 1);
    }
  }
}

export interface ApplyWaveOptions extends AnimationOptions {
  direction?: readonly number[];
  amplitude?: number;
  wave_func?: RateFunction;
  time_width?: number;
  ripples?: number;
}

export class ApplyWave extends Homotopy {
  constructor(
    mobject: Mobject,
    direction: readonly number[] = UP,
    amplitude = 0.2,
    wave_func: RateFunction = smooth,
    time_width = 1,
    ripples = 1,
    run_time = 2,
    options: AnimationOptions = {},
  ) {
    const x_min = mobject.get_left()[0] ?? 0;
    const x_max = mobject.get_right()[0] ?? 0;
    const vect = normalize([...direction]).map((value) => value * amplitude);

    const wave = (t: number): number => {
      // Mirrored looks better in the way the wave is used.
      t = 1 - t;

      // Clamp input
      if (t >= 1 || t <= 0) return 0;

      const phases = ripples * 2;
      let phase = Math.trunc(t * phases);
      if (phase === 0) {
        // First rising ripple
        return wave_func(t * phases);
      } else if (phase === phases - 1) {
        // last ripple. Rising or falling depending on the number of ripples
        // The (ripples % 2)-term is used to make this distinction.
        t -= phase / phases; // Time relative to the phase
        return (1 - wave_func(t * phases)) * (2 * (ripples % 2) - 1);
      } else {
        // Longer phases:
        phase = Math.trunc((phase - 1) / 2);
        t -= (2 * phase + 1) / phases;

        // Similar to last ripple:
        return (1 - 2 * wave_func(t * ripples)) * (1 - 2 * (phase % 2));
      }
    };

    const homotopy = (x: number, y: number, z: number, t: number): [number, number, number] => {
      const upper = interpolate(0, 1 + time_width, t) as number;
      const lower = upper - time_width;
      const relative_x = inverse_interpolate(x_min, x_max, x) as number;
      const wave_phase = inverse_interpolate(lower, upper, relative_x) as number;
      const nudge = wave(wave_phase);
      return [x + nudge * (vect[0] ?? 0), y + nudge * (vect[1] ?? 0), z + nudge * (vect[2] ?? 0)];
    };
    super(homotopy, mobject, { run_time, ...options });
  }
}

export interface ApplyWaveKwargs extends ApplyWaveOptions {
  run_time?: number;
}

export function createApplyWave(
  mobject: Mobject,
  options: ApplyWaveKwargs = {},
): ApplyWave {
  const { direction = UP, amplitude = 0.2, wave_func = smooth, time_width = 1, ripples = 1, run_time = 2, ...rest } = options;
  return new ApplyWave(mobject, direction, amplitude, wave_func, time_width, ripples, run_time, rest);
}
