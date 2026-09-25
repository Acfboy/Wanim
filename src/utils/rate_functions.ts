/** Rate functions (speed curves) for animations. Faithful port of manim/utils/rate_functions.py (layer 0 subset). */

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
