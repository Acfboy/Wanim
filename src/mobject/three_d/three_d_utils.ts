import { array, type NDArray } from "numpy-ts";
import { ORIGIN, UP } from "../../constants.js";
import { pointRows } from "../../utils/arrays.js";
import { _get_unit_normal_flat } from "../../utils/space_ops.js";
import type { VMobject } from "../types/vectorized_mobject.js";

function normOf(values: number[] | NDArray): number {
  const rows = pointRows(values, 3);
  const row = rows[0] ?? [0, 0, 0];
  return Math.hypot(...row);
}

/**
 * True when `vmob` resolves a real `get_unit_normal` (own property or
 * prototype method).
 *
 * VMobject instances are Proxies that fabricate throwing getters for unknown
 * `get_*` names, so `typeof vmob.get_unit_normal === "function"` alone cannot
 * tell a real override from the proxy's placeholder.  The proxy's `has` trap
 * forwards to `Reflect.has`, so `in` is true exactly when a real member exists
 * (the fabricated placeholders are not properties); `_set_axis_shading`'s
 * instance assignment and subclass methods are both detected.  The check is a
 * cheap probe, safe on both proxied and plain VMobjects.
 */
export function has_unit_normal_override(vmob: VMobject): boolean {
  return "get_unit_normal" in vmob
    && typeof (vmob as unknown as { get_unit_normal?: unknown }).get_unit_normal === "function";
}

/**
 * Normalized normal returned by a mobject's own `get_unit_normal` override, or
 * `null` when there is no override (or it yields a zero / non-finite vector,
 * in which case the caller falls back to the geometric normal).
 */
export function vmob_unit_normal_override(vmob: VMobject, point: number[]): number[] | null {
  if (!has_unit_normal_override(vmob)) return null;
  const fn = (vmob as unknown as { get_unit_normal: (p: number[]) => unknown }).get_unit_normal;
  if (typeof fn !== "function") return null;
  const row = pointRows(fn.call(vmob, point), 3)[0] ?? [0, 0, 0];
  const norm = Math.hypot(row[0] ?? 0, row[1] ?? 0, row[2] ?? 0);
  if (!(norm > 0) || !Number.isFinite(norm)) return null;
  return [row[0]! / norm, row[1]! / norm, row[2]! / norm];
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
  const i = point_index;
  const rows = pointRows(vmob.points, vmob.dim);
  const center = rows[i] ?? [...ORIGIN];
  // A per-mobject override (e.g. ThreeDAxes._set_axis_shading) wins over the
  // geometric normal, matching the instance-level `get_unit_normal` patch.
  const override = vmob_unit_normal_override(vmob, center);
  if (override !== null) return array(override) as NDArray;
  if (vmob.get_anchors().length <= 2) return array([...UP]) as NDArray;
  const im3 = i > 2 ? i - 3 : n_points - 4;
  const ip3 = i < n_points - 3 ? i + 3 : 3;
  const pPlus = rows[ip3] ?? [...ORIGIN];
  const pMinus = rows[im3] ?? [...ORIGIN];
  const unit_normal = _get_unit_normal_flat(
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
