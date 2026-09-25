import { Animation, classDefaultsOf, type AnimationOptions } from "./animation.js";
import type { Mobject } from "../mobject/mobject.js";
import { interpolate } from "../utils/bezier.js";

/** Structural stand-in for `DecimalNumber` (a text/TeX mobject, out of scope for this port). */
export interface DecimalNumberLike extends Mobject {
  number: number;
  set_value(value: number): unknown;
}

export interface ChangingDecimalOptions extends AnimationOptions {
  suspend_mobject_updating?: boolean;
}

export class ChangingDecimal extends Animation {
  number_update_func: (alpha: number) => number;

  constructor(
    decimal_mob: Mobject,
    number_update_func: (alpha: number) => number,
    options: ChangingDecimalOptions = {},
  ) {
    const clsDefaults = classDefaultsOf(new.target ?? ChangingDecimal);
    const {
      suspend_mobject_updating = (clsDefaults.suspend_mobject_updating as boolean | undefined) ?? false,
      ...rest
    } = options;
    ChangingDecimal.checkValidityOfInput(decimal_mob);
    super(decimal_mob, { ...rest, suspend_mobject_updating });
    this.number_update_func = number_update_func;
  }

  check_validity_of_input(decimal_mob: Mobject): void {
    ChangingDecimal.checkValidityOfInput(decimal_mob);
  }

  private static checkValidityOfInput(decimal_mob: Mobject): void {
    // NOTE: Python checks `isinstance(decimal_mob, DecimalNumber)`, but
    // `DecimalNumber` is a text/TeX mobject (out of scope for this port), so
    // the check is structural: the mobject must actually define `set_value`
    // (tested with `in` because the Mobject proxy fabricates `set_*`
    // accessors on attribute read).
    if (!("set_value" in (decimal_mob as unknown as object))) {
      throw new TypeError("ChangingDecimal can only take in a DecimalNumber");
    }
  }

  override interpolate_mobject(alpha: number): void {
    (this.mobject as unknown as DecimalNumberLike).set_value(
      this.number_update_func(this.rate_func(alpha)),
    );
  }
}

export class ChangeDecimalToValue extends ChangingDecimal {
  constructor(decimal_mob: Mobject, target_number: number, options: ChangingDecimalOptions = {}) {
    const start_number = (decimal_mob as unknown as DecimalNumberLike).number;
    super(decimal_mob, (a: number) => interpolate(start_number, target_number, a) as number, options);
  }
}
