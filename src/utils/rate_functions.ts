/** Rate functions (speed curves) for animations. Faithful port of manim/utils/rate_functions.py (layer 0 subset). */
import { sigmoid } from "./simple_functions.js";

export type RateFunction = (t: number, ...args: unknown[]) => number;

/** Decorator: clamp output to 0 for t<0 and 1 for t>1. */
export function unit_interval(fn: RateFunction): RateFunction {
  const wrapper = (t: number, ...args: unknown[]): number => {
    if (0 <= t && t <= 1) {
      return (fn as (...a: unknown[]) => number)(t, ...args);
    } else if (t < 0) {
      return 0;
    } else {
      return 1;
    }
  };
  return wrapper as RateFunction;
}

/** Decorator: return 0 outside [0,1]. */
export function zero(fn: RateFunction): RateFunction {
  const wrapper = (t: number, ...args: unknown[]): number => {
    if (0 <= t && t <= 1) {
      return (fn as (...a: unknown[]) => number)(t, ...args);
    } else {
      return 0;
    }
  };
  return wrapper as RateFunction;
}

export function squish_rate_func(func: RateFunction, a = 0.4, b = 0.6): RateFunction {
  function result(t: number, ...args: unknown[]): number {
    if (a === b) {
      return a;
    }
    let new_t: number;
    if (t < a) {
      new_t = 0.0;
    } else if (t > b) {
      new_t = 1.0;
    } else {
      new_t = (t - a) / (b - a);
    }
    return (func as (...a: unknown[]) => number)(new_t, ...args);
  }
  return result as RateFunction;
}

export const lingering: RateFunction = unit_interval((t: number): number => {
  function identity(tt: number): number {
    return tt;
  }
  // TODO: Isn't this just 0.8 * t?
  return (squish_rate_func(identity as RateFunction, 0, 0.8) as RateFunction)(t);
}) as RateFunction;

export const smooth: RateFunction = unit_interval(((t: number, inflection: number = 10.0): number => {
  const error = sigmoid(-inflection / 2);
  return Math.min(
    Math.max((sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0),
    1,
  );
}) as RateFunction);

export const linear: RateFunction = unit_interval(((t: number): number => {
  return t;
}) as RateFunction);

export function not_quite_there(
  func: RateFunction = smooth,
  proportion = 0.7,
): RateFunction {
  function result(t: number, ...args: unknown[]): number {
    return proportion * (func as (...a: unknown[]) => number)(t, ...args);
  }
  return result as RateFunction;
}

export const smoothstep: RateFunction = unit_interval(((t: number): number => {
  return 3 * t ** 2 - 2 * t ** 3;
}) as RateFunction);

export const smootherstep: RateFunction = unit_interval(((t: number): number => {
  return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3;
}) as RateFunction);

export const smoothererstep: RateFunction = unit_interval(((t: number): number => {
  return 35 * t ** 4 - 84 * t ** 5 + 70 * t ** 6 - 20 * t ** 7;
}) as RateFunction);

export const slow_into: RateFunction = unit_interval(((t: number): number => {
  const val: number = Math.sqrt(1 - (1 - t) * (1 - t));
  return val;
}) as RateFunction);

export const running_start: RateFunction = unit_interval(((t: number, pull_factor: number = -0.5): number => {
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;
  const t6 = t5 * t;
  const mt = 1 - t;
  const mt2 = mt * mt;
  const mt3 = mt2 * mt;
  const mt4 = mt3 * mt;

  // This is equivalent to creating a Bézier with [0, 0, pull_factor, pull_factor, 1, 1, 1]
  // and evaluating it at t.
  return (
    15 * t2 * mt4 * pull_factor +
    20 * t3 * mt3 * pull_factor +
    15 * t4 * mt2 +
    6 * t5 * mt +
    t6
  );
}) as RateFunction);

export const exponential_decay: RateFunction = unit_interval(((t: number, half_life: number = 0.1): number => {
  // The half-life should be rather small to minimize
  // the cut-off error at the end
  const val: number = 1 - Math.exp(-t / half_life);
  return val;
}) as RateFunction);

export const ease_in_sine: RateFunction = unit_interval(((t: number): number => {
  const val: number = 1 - Math.cos((t * Math.PI) / 2);
  return val;
}) as RateFunction);

export const ease_out_sine: RateFunction = unit_interval(((t: number): number => {
  const val: number = Math.sin((t * Math.PI) / 2);
  return val;
}) as RateFunction);

export const ease_in_out_sine: RateFunction = unit_interval(((t: number): number => {
  const val: number = -(Math.cos(Math.PI * t) - 1) / 2;
  return val;
}) as RateFunction);

export const ease_in_quad: RateFunction = unit_interval(((t: number): number => {
  return t * t;
}) as RateFunction);

export const ease_out_quad: RateFunction = unit_interval(((t: number): number => {
  return 1 - (1 - t) * (1 - t);
}) as RateFunction);

export const ease_in_out_quad: RateFunction = unit_interval(((t: number): number => {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}) as RateFunction);

export const ease_in_cubic: RateFunction = unit_interval(((t: number): number => {
  return t * t * t;
}) as RateFunction);

export const ease_out_cubic: RateFunction = unit_interval(((t: number): number => {
  return 1 - Math.pow(1 - t, 3);
}) as RateFunction);

export const ease_in_out_cubic: RateFunction = unit_interval(((t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}) as RateFunction);

export const ease_in_quart: RateFunction = unit_interval(((t: number): number => {
  return t * t * t * t;
}) as RateFunction);

export const ease_out_quart: RateFunction = unit_interval(((t: number): number => {
  return 1 - Math.pow(1 - t, 4);
}) as RateFunction);

export const ease_in_out_quart: RateFunction = unit_interval(((t: number): number => {
  return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
}) as RateFunction);

export const ease_in_quint: RateFunction = unit_interval(((t: number): number => {
  return t * t * t * t * t;
}) as RateFunction);

export const ease_out_quint: RateFunction = unit_interval(((t: number): number => {
  return 1 - Math.pow(1 - t, 5);
}) as RateFunction);

export const ease_in_out_quint: RateFunction = unit_interval(((t: number): number => {
  return t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2;
}) as RateFunction);

export const ease_in_expo: RateFunction = unit_interval(((t: number): number => {
  return t === 0 ? 0 : Math.pow(2, 10 * t - 10);
}) as RateFunction);

export const ease_out_expo: RateFunction = unit_interval(((t: number): number => {
  return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
}) as RateFunction);

export const ease_in_out_expo: RateFunction = unit_interval(((t: number): number => {
  if (t === 0) {
    return 0;
  } else if (t === 1) {
    return 1;
  } else if (t < 0.5) {
    return Math.pow(2, 20 * t - 10) / 2;
  } else {
    return (2 - Math.pow(2, -20 * t + 10)) / 2;
  }
}) as RateFunction);

export const ease_in_circ: RateFunction = unit_interval(((t: number): number => {
  return 1 - Math.sqrt(1 - Math.pow(t, 2));
}) as RateFunction);

export const ease_out_circ: RateFunction = unit_interval(((t: number): number => {
  return Math.sqrt(1 - Math.pow(t - 1, 2));
}) as RateFunction);

export const ease_in_out_circ: RateFunction = unit_interval(((t: number): number => {
  return t < 0.5
    ? (1 - Math.sqrt(1 - Math.pow(2 * t, 2))) / 2
    : (Math.sqrt(1 - Math.pow(-2 * t + 2, 2)) + 1) / 2;
}) as RateFunction);

export const ease_in_back: RateFunction = unit_interval(((t: number): number => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return c3 * t * t * t - c1 * t * t;
}) as RateFunction);

export const ease_out_back: RateFunction = unit_interval(((t: number): number => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}) as RateFunction);

export const ease_in_out_back: RateFunction = unit_interval(((t: number): number => {
  const c1 = 1.70158;
  const c2 = c1 * 1.525;
  return t < 0.5
    ? (Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
    : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
}) as RateFunction);

export const ease_in_elastic: RateFunction = unit_interval(((t: number): number => {
  const c4 = (2 * Math.PI) / 3;
  if (t === 0) {
    return 0;
  } else if (t === 1) {
    return 1;
  } else {
    const val: number = -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
    return val;
  }
}) as RateFunction);

export const ease_out_elastic: RateFunction = unit_interval(((t: number): number => {
  const c4 = (2 * Math.PI) / 3;
  if (t === 0) {
    return 0;
  } else if (t === 1) {
    return 1;
  } else {
    const val: number = Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    return val;
  }
}) as RateFunction);

export const ease_in_out_elastic: RateFunction = unit_interval(((t: number): number => {
  const c5 = (2 * Math.PI) / 4.5;
  if (t === 0) {
    return 0;
  } else if (t === 1) {
    return 1;
  } else if (t < 0.5) {
    const val: number = -(Math.pow(2, 20 * t - 10) * Math.sin((20 * t - 11.125) * c5)) / 2;
    return val;
  } else {
    const val: number = (Math.pow(2, -20 * t + 10) * Math.sin((20 * t - 11.125) * c5)) / 2 + 1;
    return val;
  }
}) as RateFunction);

function ease_out_bounce_raw(t: number): number {
  const n1 = 7.5625;
  const d1 = 2.75;

  if (t < 1 / d1) {
    return n1 * t * t;
  } else if (t < 2 / d1) {
    return n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75;
  } else if (t < 2.5 / d1) {
    return n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375;
  } else {
    return n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375;
  }
}

export const ease_out_bounce: RateFunction = unit_interval((ease_out_bounce_raw as unknown as RateFunction));
