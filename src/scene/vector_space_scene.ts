import { array, type NDArray } from "numpy-ts";
import { Scene } from "./scene.js";
import { matrixRows, pointFrom, transposeMatrix } from "../utils/arrays.js";
import { update_dict_recursively } from "../utils/config_ops.js";

export { CoordinateSystem } from "../mobject/graphing/coordinate_systems.js";

export class LinearTransformationScene extends Scene {
  static update_default_configs(
    default_configs: Iterable<Record<string, unknown>>,
    passed_configs: Iterable<Record<string, unknown> | null | undefined>,
  ): void {
    const defaults = [...default_configs];
    const passed = [...passed_configs];
    const length = Math.min(defaults.length, passed.length);
    for (let index = 0; index < length; index += 1) {
      const passed_config = passed[index];
      if (passed_config !== null && passed_config !== undefined) {
        update_dict_recursively(defaults[index] as Record<string, unknown>, passed_config);
      }
    }
  }

  get_matrix_transformation(matrix: unknown): (point: number[] | NDArray) => NDArray {
    return this.get_transposed_matrix_transformation(transposeMatrix(matrixRows(matrix)));
  }

  get_transposed_matrix_transformation(transposed_matrix: unknown): (point: number[] | NDArray) => NDArray {
    const values = matrixRows(transposed_matrix);
    let matrix = values;
    if (values.length === 2 && values.every((row) => row.length === 2)) {
      matrix = [
        [values[0][0], values[0][1], 0],
        [values[1][0], values[1][1], 0],
        [0, 0, 1],
      ];
    } else if (values.length !== 3 || values.some((row) => row.length !== 3)) {
      const error = new Error("Matrix has bad dimensions");
      error.name = "ValueError";
      throw error;
    }
    const finalMatrix = matrix.map((row) => [...row]);
    return (point: number[] | NDArray) => {
      const valuesPoint = pointFrom(point, 3).toArray() as number[];
      return array(Array.from({ length: finalMatrix[0]?.length ?? 0 }, (_, column) => finalMatrix.reduce((sum, row, rowIndex) => sum + (row[column] ?? 0) * (valuesPoint[rowIndex] ?? 0), 0))) as NDArray;
    };
  }
}
