import { divide, log, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "../../utils/arrays.js";

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

export class _ScaleBase {
  custom_labels: boolean;

  constructor(custom_labels = false) {
    this.custom_labels = custom_labels;
  }

  function(_value: number | NDArray): number | NDArray {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  inverse_function(_value: number | NDArray): number | NDArray {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  get_custom_labels(
    _val_range: Iterable<number>,
    ..._kw_args: unknown[]
  ): unknown[] {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }
}

export class LinearBase extends _ScaleBase {
  scale_factor: number;

  constructor(scale_factor = 1) {
    super();
    this.scale_factor = scale_factor;
  }

  override function(value: number): number {
    return this.scale_factor * value;
  }

  override inverse_function(value: number): number {
    return value / this.scale_factor;
  }
}

export class LogBase extends _ScaleBase {
  base: number;
  custom_labels: boolean;

  constructor(base = 10, custom_labels = true) {
    super();
    this.base = base;
    this.custom_labels = custom_labels;
  }

  override function(value: number): number {
    return this.base ** value;
  }

  override inverse_function(value: number | NDArray): number | NDArray {
    if (isNDArray(value)) {
      const flat = flattenNumbers((value as NDArray).toArray());
      const condition = !flat.some((item) => item !== 0);
      if (condition) {
        throw valueError(
          "log(0) is undefined. Make sure the value is in the domain of the function",
        );
      }
      return divide(log(value as never) as never, Math.log(this.base) as never) as unknown as NDArray;
    }
    const scalar = numeric(value);
    if (scalar <= 0) {
      throw valueError(
        "log(0) is undefined. Make sure the value is in the domain of the function",
      );
    }
    return Math.log(scalar) / Math.log(this.base);
  }
}

function flattenNumbers(value: unknown): number[] {
  if (Array.isArray(value)) return (value as unknown[]).flatMap(flattenNumbers);
  return [numeric(value)];
}
