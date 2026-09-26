import { array, type NDArray } from "numpy-ts";
import { ORIGIN, UP } from "../../constants.js";
import { pointRows } from "../../utils/arrays.js";
import { get_unit_normal } from "../../utils/space_ops.js";
import type { VMobject } from "../types/vectorized_mobject.js";

function normOf(values: number[] | NDArray): number {
  const rows = pointRows(values, 3);
  const row = rows[0] ?? [0, 0, 0];
  return Math.hypot(...row);
}

export function get_3d_vmob_gradient_start_and_end_points(
  vmob: VMobject,
): [NDArray, NDArray] {
  return [get_3d_vmob_start_corner(vmob), get_3d_vmob_end_corner(vmob)];
}

export function get_3d_vmob_start_corner_index(_vmob: VMobject): 0 {
  return 0;
}

export function get_3d_vmob_end_corner_index(vmob: VMobject): number {
  // NOTE: Python uses floor division `((len(vmob.points) - 1) // 6) * 3`;
  // the operand is non-negative here, so Math.floor matches exactly.
  return Math.floor((vmob.get_num_points() - 1) / 6) * 3;
}

export function get_3d_vmob_start_corner(vmob: VMobject): NDArray {
  if (vmob.get_num_points() === 0) return array([...ORIGIN]) as NDArray;
  return array([...(pointRows(vmob.points, vmob.dim)[get_3d_vmob_start_corner_index(vmob)] ?? [...ORIGIN])]) as NDArray;
}

export function get_3d_vmob_end_corner(vmob: VMobject): NDArray {
  if (vmob.get_num_points() === 0) return array([...ORIGIN]) as NDArray;
  return array([...(pointRows(vmob.points, vmob.dim)[get_3d_vmob_end_corner_index(vmob)] ?? [...ORIGIN])]) as NDArray;
}

export function get_3d_vmob_unit_normal(vmob: VMobject, point_index: number): NDArray {
  const n_points = vmob.get_num_points();
  if (vmob.get_anchors().length <= 2) return array([...UP]) as NDArray;
  const i = point_index;
  const im3 = i > 2 ? i - 3 : n_points - 4;
  const ip3 = i < n_points - 3 ? i + 3 : 3;
  const rows = pointRows(vmob.points, vmob.dim);
  const center = rows[i] ?? [...ORIGIN];
  const pPlus = rows[ip3] ?? [...ORIGIN];
  const pMinus = rows[im3] ?? [...ORIGIN];
  const unit_normal = get_unit_normal(
    pPlus.map((value, index) => value - (center[index] ?? 0)),
    pMinus.map((value, index) => value - (center[index] ?? 0)),
  );
  if (normOf(unit_normal) === 0) return array([...UP]) as NDArray;
  return array([...unit_normal]) as NDArray;
}

export function get_3d_vmob_start_corner_unit_normal(vmob: VMobject): NDArray {
  return get_3d_vmob_unit_normal(vmob, get_3d_vmob_start_corner_index(vmob));
}

export function get_3d_vmob_end_corner_unit_normal(vmob: VMobject): NDArray {
  return get_3d_vmob_unit_normal(vmob, get_3d_vmob_end_corner_index(vmob));
}
