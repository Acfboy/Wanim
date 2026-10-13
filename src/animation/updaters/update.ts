import { Animation, type AnimationOptions } from "../animation.js";
import { addPoints, pointRows, subtractPoints } from "../../utils/arrays.js";
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

export class MaintainPositionRelativeTo extends Animation {
  tracked_mobject: Mobject;
  diff: number[];

  constructor(mobject: Mobject, tracked_mobject: Mobject, options: AnimationOptions = {}) {
    super(mobject, options);
    this.tracked_mobject = tracked_mobject;
    this.diff = subtractPoints(
      pointRows(mobject.get_center(), mobject.dim)[0] ?? [0, 0, 0],
      pointRows(tracked_mobject.get_center(), tracked_mobject.dim)[0] ?? [0, 0, 0],
      mobject.dim,
    );
  }

  interpolate_mobject(_alpha: number): void {
    const target = pointRows(this.tracked_mobject.get_center(), this.mobject.dim)[0] ?? [0, 0, 0];
    const location = pointRows(this.mobject.get_center(), this.mobject.dim)[0] ?? [0, 0, 0];
    this.mobject.shift(
      addPoints(subtractPoints(target, location, this.mobject.dim), this.diff, this.mobject.dim),
    );
  }
}
