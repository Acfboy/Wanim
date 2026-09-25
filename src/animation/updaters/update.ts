import { Animation, type AnimationOptions } from "../animation.js";
import type { Mobject } from "../../mobject/mobject.js";

export interface UpdateFromFuncOptions extends AnimationOptions {
  suspend_mobject_updating?: boolean;
}

export class UpdateFromFunc extends Animation {
  update_function: (mobject: Mobject, ...args: unknown[]) => void;

  constructor(
    mobject: Mobject,
    update_function: (mobject: Mobject, ...args: unknown[]) => void,
    options: UpdateFromFuncOptions = {},
  ) {
    const { suspend_mobject_updating = false, ...rest } = options;
    super(mobject, { suspend_mobject_updating, ...rest });
    this.update_function = update_function;
  }

  interpolate_mobject(_alpha: number): void {
    this.update_function(this.mobject);
  }
}

export class UpdateFromAlphaFunc extends UpdateFromFunc {
  interpolate_mobject(alpha: number): void {
    this.update_function(this.mobject, this.rate_func(alpha));
  }
}
