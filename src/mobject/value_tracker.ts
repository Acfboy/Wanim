import { zeros } from "numpy-ts";
import { Mobject, type MobjectOptions } from "./mobject.js";
import type { ComplexPoint } from "../utils/arrays.js";

export type ComplexValue = ComplexPoint | readonly [number, number] | number;

function toComplex(value: ComplexValue): { re: number; im: number } {
  if (typeof value === "number") return { re: value, im: 0 };
  if (Array.isArray(value)) {
    const pair = value as readonly number[];
    return { re: Number(pair[0] ?? 0), im: Number(pair[1] ?? 0) };
  }
  const point = value as ComplexPoint;
  return { re: Number(point.re ?? 0), im: Number(point.im ?? 0) };
}

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

  __bool__(): boolean {
    return Boolean(this.get_value());
  }

  __floordiv__(d_value: number): ValueTracker {
    if (d_value === 0) {
      const error = new Error("float floor division by zero");
      error.name = "ZeroDivisionError";
      throw error;
    }
    return new ValueTracker(Math.floor(this.get_value() / d_value));
  }

  __ifloordiv__(d_value: number): this {
    if (d_value === 0) {
      const error = new Error("float floor division by zero");
      error.name = "ZeroDivisionError";
      throw error;
    }
    this.set_value(Math.floor(this.get_value() / d_value));
    return this;
  }

  __mod__(d_value: number): ValueTracker {
    if (d_value === 0) {
      const error = new Error("float modulo");
      error.name = "ZeroDivisionError";
      throw error;
    }
    const value = this.get_value();
    return new ValueTracker(value - Math.floor(value / d_value) * d_value);
  }

  __imod__(d_value: number): this {
    if (d_value === 0) {
      const error = new Error("float modulo");
      error.name = "ZeroDivisionError";
      throw error;
    }
    const value = this.get_value();
    this.set_value(value - Math.floor(value / d_value) * d_value);
    return this;
  }

  __mul__(d_value: number): ValueTracker {
    return new ValueTracker(this.get_value() * d_value);
  }

  __imul__(d_value: number): this {
    this.set_value(this.get_value() * d_value);
    return this;
  }

  __pow__(d_value: number): ValueTracker {
    return new ValueTracker(this.get_value() ** d_value);
  }

  __ipow__(d_value: number): this {
    this.set_value(this.get_value() ** d_value);
    return this;
  }

  __truediv__(d_value: number): ValueTracker {
    if (d_value === 0) {
      const error = new Error("float division by zero");
      error.name = "ZeroDivisionError";
      throw error;
    }
    return new ValueTracker(this.get_value() / d_value);
  }

  __itruediv__(d_value: number): this {
    if (d_value === 0) {
      const error = new Error("float division by zero");
      error.name = "ZeroDivisionError";
      throw error;
    }
    this.set_value(this.get_value() / d_value);
    return this;
  }
}

export class ComplexValueTracker extends ValueTracker {
  constructor(value: ComplexValue = 0, options: MobjectOptions = {}) {
    super(typeof value === "number" ? value : 0, options);
    this.set_value(value);
  }

  // NOTE: mirrors `get_value() -> complex  # type: ignore [override]` in
  // Python, which intentionally narrows nothing but changes the return kind.
  // @ts-expect-error: complex return is incompatible with ValueTracker's number by design.
  override get_value(): ComplexPoint {
    return {
      re: Number(this.points.get([0, 0])),
      im: Number(this.points.get([0, 1])),
    };
  }

  override set_value(value: ComplexValue): this {
    const z = toComplex(value);
    this.points.set([0, 0], z.re);
    this.points.set([0, 1], z.im);
    return this;
  }
}
