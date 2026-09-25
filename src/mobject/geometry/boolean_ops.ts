import { array, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "../../utils/arrays.js";
import { VMobject } from "../types/vectorized_mobject.js";

export class _BooleanOps extends VMobject {
  _convert_2d_to_3d_array(points: Iterable<number[] | readonly number[] | NDArray>, z_dim = 0): NDArray {
    const list_of_points: number[][] = [...points].map((point) =>
      isNDArray(point)
        ? ((point as NDArray).toArray() as unknown[]).map(numeric)
        : [...(point as readonly number[])].map(numeric),
    );
    for (let index = 0; index < list_of_points.length; index += 1) {
      const point = list_of_points[index] as number[];
      if (point.length === 2) list_of_points[index] = [...point, z_dim];
    }
    return array(list_of_points) as NDArray;
  }
}
