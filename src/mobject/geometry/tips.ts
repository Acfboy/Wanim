import { array, type NDArray } from "numpy-ts";
import { pointRows } from "../../utils/arrays.js";
import { angle_of_vector } from "../../utils/space_ops.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";

export class ArrowTip extends VMobject {
  constructor(options: VMobjectOptions = {}, ..._rest: unknown[]) {
    super(options);
    const error = new Error("Has to be implemented in inheriting subclasses.");
    error.name = "NotImplementedError";
    throw error;
  }

  get base(): NDArray {
    return this.point_from_proportion(0.5);
  }

  get tip_point(): NDArray {
    const rows = pointRows(this.points, this.dim);
    return array(rows[0] ?? []) as NDArray;
  }

  get vector(): number[] {
    const tip = pointRows(this.tip_point, 3)[0] ?? [0, 0, 0];
    const base = pointRows(this.base, 3)[0] ?? [0, 0, 0];
    return tip.map((value, index) => value - (base[index] ?? 0));
  }

  get tip_angle(): number {
    return angle_of_vector(this.vector) as number;
  }

  get length(): number {
    const v = this.vector;
    return Math.sqrt(v.reduce((sum, value) => sum + value * value, 0));
  }
}
