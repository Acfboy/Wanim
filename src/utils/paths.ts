import type { NDArray } from "numpy-ts";
import { interpolate } from "./bezier.js";

export type PathFuncType = (
  start_points: NDArray,
  end_points: NDArray,
  alpha: number,
) => NDArray;

export const STRAIGHT_PATH_THRESHOLD = 0.01;

export function straight_path(): typeof interpolate {
  return interpolate;
}
