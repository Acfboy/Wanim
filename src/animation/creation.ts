import { Animation, type AnimationOptions } from "./animation.js";
import { Mobject } from "../mobject/mobject.js";
import { VMobject, VGroup } from "../mobject/types/vectorized_mobject.js";
import { pointRows } from "../utils/arrays.js";

export interface ShowPartialOptions extends AnimationOptions {
  mobject?: VMobject;
}

export class ShowPartial extends Animation {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    if (!mobject || typeof (mobject as unknown as { pointwise_become_partial?: unknown }).pointwise_become_partial !== "function") {
      const constructorName = new.target.name;
      throw new TypeError(`${constructorName} only works for VMobjects.`);
    }
    super(mobject, options);
  }

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): void {
    const mobject = submobject as VMobject;
    const [a, b] = this._get_bounds(alpha);
    mobject.pointwise_become_partial(startingSubmobject as VMobject, a, b);
  }

  _get_bounds(_alpha: number): [number, number] {
    const error = new Error("Please use Create or ShowPassingFlash");
    error.name = "NotImplementedError";
    throw error;
  }
}

export class Create extends ShowPartial {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    super(mobject, { lag_ratio: 1, introducer: true, ...options });
  }

  _get_bounds(alpha: number): [number, number] {
    return [0, alpha];
  }
}

export class Uncreate extends Create {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    super(mobject, { reverse_rate_function: true, introducer: false, remover: true, ...options });
  }
}

export class ShowIncreasingSubsets extends Animation {
  all_submobs: Mobject[];
  int_func: (value: number) => number;

  constructor(group: Mobject, options: AnimationOptions & { int_func?: (value: number) => number } = {}) {
    const allSubmobs = [...group.submobjects];
    for (const mobject of allSubmobs) mobject.set_opacity(0);
    super(group, options);
    this.all_submobs = allSubmobs;
    this.int_func = options.int_func ?? Math.floor;
  }

  interpolate_mobject(alpha: number): void {
    const nSubmobs = this.all_submobs.length;
    const value = this.reverse_rate_function ? 1 - this.rate_func(alpha) : this.rate_func(alpha);
    const index = Math.trunc(Number(this.int_func(value * nSubmobs)));
    this.update_submobject_list(index);
  }

  update_submobject_list(index: number): void {
    if (!Number.isInteger(index)) {
      throw new TypeError("slice indices must be integers or None or have an __index__ method");
    }
    for (const mobject of this.all_submobs.slice(0, index)) mobject.set_opacity(1);
    for (const mobject of this.all_submobs.slice(index)) mobject.set_opacity(0);
  }
}

export class ShowSubmobjectsOneByOne extends ShowIncreasingSubsets {
  update_submobject_list(index: number): void {
    const current = this.all_submobs.slice(0, index);
    for (const mobject of current.slice(0, -1)) mobject.set_opacity(0);
    if (current.length > 0) current[current.length - 1].set_opacity(1);
  }
}
