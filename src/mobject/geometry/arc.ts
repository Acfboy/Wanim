import { array, type NDArray } from "numpy-ts";
import { DEFAULT_ARROW_TIP_LENGTH, OUT } from "../../constants.js";
import { linspace } from "../../utils/arrays.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";

export interface TipableVMobjectOptions extends VMobjectOptions {
  tip_length?: number;
  normal_vector?: readonly number[] | NDArray;
  tip_style?: Record<string, unknown>;
}

export class TipableVMobject extends VMobject {
  tip_length: number;
  normal_vector: readonly number[] | NDArray;
  tip_style: Record<string, unknown>;

  constructor(options: TipableVMobjectOptions = {}) {
    const {
      tip_length = DEFAULT_ARROW_TIP_LENGTH,
      normal_vector = OUT,
      tip_style = {},
      ...rest
    } = options;
    super(rest);
    this.tip_length = tip_length;
    this.normal_vector = normal_vector;
    this.tip_style = tip_style ?? {};
  }

  get_default_tip_length(): number {
    return this.tip_length;
  }
}

export class Arc {
  static _create_quadratic_bezier_points(angle: number, start_angle = 0, n_components = 8): NDArray {
    const samples: number[][] = linspace(start_angle, start_angle + angle, 2 * n_components + 1).map(
      (a) => [Math.cos(a), Math.sin(a), 0],
    );
    const theta = angle / n_components;
    const scale = 1 / Math.cos(theta / 2);
    for (let index = 1; index < samples.length; index += 2) {
      const row = samples[index] as number[];
      samples[index] = row.map((value) => value * scale);
    }
    const points: number[][] = Array.from({ length: 3 * n_components }, () => [0, 0, 0]);
    for (let index = 0; index < n_components; index += 1) {
      points[index * 3] = [...(samples[index * 2] ?? [0, 0, 0])];
      points[index * 3 + 1] = [...(samples[index * 2 + 1] ?? [0, 0, 0])];
      points[index * 3 + 2] = [...(samples[index * 2 + 2] ?? [0, 0, 0])];
    }
    return array(points) as NDArray;
  }
}
