import { zeros } from "numpy-ts";
import { Mobject, type MobjectOptions } from "./mobject.js";

export interface ValueTrackerOptions extends MobjectOptions {
  value?: number;
}

export class ValueTracker extends Mobject {
  constructor(value = 0, options: MobjectOptions = {}) {
    const { value: _ignored, ...rest } = options as ValueTrackerOptions;
    void _ignored;
    super(rest);
    this.set({ points: zeros([1, 3]) });
    this.set_value(value);
  }

  get_value(): number {
    return Number(this.points.get([0, 0]));
  }

  set_value(value: number): this {
    this.points.set([0, 0], value);
    return this;
  }

  increment_value(d_value: number): this {
    this.set_value(this.get_value() + d_value);
    return this;
  }
}
