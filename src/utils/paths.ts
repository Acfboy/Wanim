import { array, type NDArray } from "numpy-ts";
import { _cContiguousRows } from "./arrays.js";
import { OUT, PI } from "../constants.js";
import { interpolate } from "./bezier.js";
import { _cross_flat, normalize, rotation_matrix } from "./space_ops.js";

export type PathFuncType = (
  start_points: NDArray,
  end_points: NDArray,
  alpha: number,
) => NDArray;

export const STRAIGHT_PATH_THRESHOLD = 0.01;

export function straight_path(): typeof interpolate {
  return interpolate;
}

type VectorLike = NDArray | number[] | number[][] | readonly number[];

function asRows(value: NDArray | VectorLike): number[][] {
  if (Array.isArray(value)) {
    if (value.length > 0 && Array.isArray(value[0])) {
      return (value as unknown[][]).map((row) =>
        (Array.isArray(row) ? row : [row]).map(Number),
      );
    }
    return [(value as unknown[]).map(Number)];
  }
  const arr = value as NDArray;
  const shape = arr.shape ?? [];
  const raw = arr.toArray() as unknown;
  if (shape.length === 1) return [[...(raw as unknown[]).map(Number)]];
  return (raw as unknown[][]).map((row) =>
    (Array.isArray(row) ? row : [row]).map(Number),
  );
}

function broadcast(rows: number[][], count: number): number[][] {
  if (rows.length === count) return rows;
  if (rows.length === 1) {
    const single = rows[0] ?? [];
    return Array.from({ length: count }, () => [...single]);
  }
  return Array.from(
    { length: count },
    (_, index) => [...(rows[index % rows.length] ?? [])],
  );
}

function sub(a: number[], b: number[]): number[] {
  const dim = Math.max(a.length, b.length);
  return Array.from({ length: dim }, (_, k) => (a[k] ?? 0) - (b[k] ?? 0));
}

function add(a: number[], b: number[]): number[] {
  const dim = Math.max(a.length, b.length);
  return Array.from({ length: dim }, (_, k) => (a[k] ?? 0) + (b[k] ?? 0));
}

function applyRotation(rot: NDArray, rows: number[][]): number[][] {
  const mat = (rot.toArray() as unknown[][]).map((row) =>
    (Array.isArray(row) ? row : [row]).map(Number),
  );
  // Row-vector convention: out = row @ R.T == R @ row (as column).
  return rows.map((row) =>
    mat.map((matRow) =>
      matRow.reduce((sum, value, k) => sum + value * (row[k] ?? 0), 0),
    ),
  );
}

export function path_along_circles(
  arc_angle: number,
  circles_centers: NDArray | VectorLike,
  axis: number[] | readonly number[] | NDArray = OUT,
): PathFuncType {
  const unit_axis = normalize(axis, OUT);
  const centerSeed = asRows(circles_centers as NDArray);

  function path(
    start_points: NDArray,
    end_points: NDArray,
    alpha: number,
  ): NDArray {
    // Flat fast path: `Rotate` uses this path function for every face of the
    // animated mobject on every frame.  The row-materialising implementation
    // below allocated several nested JS arrays and re-decomposed the rotation
    // matrices per face.
    const flatStart = _cContiguousRows(start_points);
    const flatEnd = _cContiguousRows(end_points);
    if (
      flatStart !== null &&
      flatEnd !== null &&
      flatStart.rowDim === flatEnd.rowDim &&
      flatStart.rowCount === flatEnd.rowCount &&
      flatStart.rowDim >= 3
    ) {
      const count = flatStart.rowCount;
      const dim = flatStart.rowDim;
      const centers = broadcast(centerSeed, count);
      const negRot = (rotation_matrix(-arc_angle, unit_axis) as NDArray).toArray() as unknown as number[][];
      const rot = (rotation_matrix(alpha * arc_angle, unit_axis) as NDArray).toArray() as unknown as number[][];
      const out = new Float64Array(count * dim);
      for (let index = 0; index < count; index += 1) {
        const center = centers[index] ?? [];
        const base = index * dim;
        // detransformed = center + negRot * (end - center)
        const ex = Number(flatEnd.data[base] ?? 0) - (center[0] ?? 0);
        const ey = Number(flatEnd.data[base + 1] ?? 0) - (center[1] ?? 0);
        const ez = Number(flatEnd.data[base + 2] ?? 0) - (center[2] ?? 0);
        const d0 = (negRot[0]?.[0] ?? 0) * ex + (negRot[0]?.[1] ?? 0) * ey + (negRot[0]?.[2] ?? 0) * ez + (center[0] ?? 0);
        const d1 = (negRot[1]?.[0] ?? 0) * ex + (negRot[1]?.[1] ?? 0) * ey + (negRot[1]?.[2] ?? 0) * ez + (center[1] ?? 0);
        const d2 = (negRot[2]?.[0] ?? 0) * ex + (negRot[2]?.[1] ?? 0) * ey + (negRot[2]?.[2] ?? 0) * ez + (center[2] ?? 0);
        // blended = lerp(start, detransformed, alpha)
        const sx = Number(flatStart.data[base] ?? 0);
        const sy = Number(flatStart.data[base + 1] ?? 0);
        const sz = Number(flatStart.data[base + 2] ?? 0);
        const bx = sx * (1 - alpha) + d0 * alpha - (center[0] ?? 0);
        const by = sy * (1 - alpha) + d1 * alpha - (center[1] ?? 0);
        const bz = sz * (1 - alpha) + d2 * alpha - (center[2] ?? 0);
        // result = center + rot * blended
        out[base] = (rot[0]?.[0] ?? 0) * bx + (rot[0]?.[1] ?? 0) * by + (rot[0]?.[2] ?? 0) * bz + (center[0] ?? 0);
        out[base + 1] = (rot[1]?.[0] ?? 0) * bx + (rot[1]?.[1] ?? 0) * by + (rot[1]?.[2] ?? 0) * bz + (center[1] ?? 0);
        out[base + 2] = (rot[2]?.[0] ?? 0) * bx + (rot[2]?.[1] ?? 0) * by + (rot[2]?.[2] ?? 0) * bz + (center[2] ?? 0);
        // Components beyond xyz are zeroed by the rotation in the row path
        // (`applyRotation` returns a 3-vector, and `center` has no 4th dim).
        for (let k = 3; k < dim; k += 1) out[base + k] = 0;
      }
      return array(out).reshape(count, dim) as NDArray;
    }
    const startRows = asRows(start_points);
    const endRows = asRows(end_points);
    const count = Math.max(startRows.length, endRows.length);
    const starts = broadcast(startRows, count);
    const ends = broadcast(endRows, count);
    const centers = broadcast(centerSeed, count);
    const negRot = rotation_matrix(-arc_angle, unit_axis);
    const detransformed = ends.map((row, index) => {
      const center = centers[index] ?? [];
      return add(center, applyRotation(negRot, [sub(row, center)])[0] ?? []);
    });
    const blended = asRows(
      interpolate(array(starts) as NDArray, array(detransformed) as NDArray, alpha) as NDArray,
    );
    const blendedRows = broadcast(blended, count);
    const rot = rotation_matrix(alpha * arc_angle, unit_axis);
    const result = blendedRows.map((row, index) => {
      const center = centers[index] ?? [];
      return add(center, applyRotation(rot, [sub(row, center)])[0] ?? []);
    });
    return array(result) as NDArray;
  }

  return path;
}

export function path_along_arc(
  arc_angle: number,
  axis: number[] | readonly number[] | NDArray = OUT,
): PathFuncType {
  if (Math.abs(arc_angle) < STRAIGHT_PATH_THRESHOLD) {
    return straight_path() as unknown as PathFuncType;
  }
  const unit_axis = normalize(axis, OUT);

  function path(
    start_points: NDArray,
    end_points: NDArray,
    alpha: number,
  ): NDArray {
    const startRows = asRows(start_points);
    const endRows = asRows(end_points);
    const count = Math.max(startRows.length, endRows.length);
    const starts = broadcast(startRows, count);
    const ends = broadcast(endRows, count);
    const vects = ends.map((row, index) => sub(row, starts[index] ?? []));
    let centers = starts.map((row, index) =>
      (row).map((value, k) => value + 0.5 * (vects[index]?.[k] ?? 0)),
    );
    if (arc_angle !== Math.PI) {
      const crossRows = vects.map((vect) =>
        _cross_flat(unit_axis, vect.map((value) => value / 2.0)),
      );
      const tanHalf = Math.tan(arc_angle / 2);
      centers = centers.map((center, index) =>
        (center).map((value, k) => value + ((crossRows[index]?.[k] ?? 0) / tanHalf)),
      );
    }
    const rot = rotation_matrix(alpha * arc_angle, unit_axis);
    const shifted = starts.map((row, index) => sub(row, centers[index] ?? []));
    const rotated = applyRotation(rot, shifted);
    return array(
      rotated.map((row, index) => add(centers[index] ?? [], row)),
    ) as NDArray;
  }

  return path;
}

export function clockwise_path(): PathFuncType {
  return path_along_arc(-PI);
}

export function clockwisePath(): PathFuncType {
  return clockwise_path();
}

export function counterclockwise_path(): PathFuncType {
  return path_along_arc(PI);
}

export function counterclockwisePath(): PathFuncType {
  return counterclockwise_path();
}

export function spiral_path(
  angle: number,
  axis: number[] | readonly number[] | NDArray = OUT,
): PathFuncType {
  if (Math.abs(angle) < STRAIGHT_PATH_THRESHOLD) {
    return straight_path() as unknown as PathFuncType;
  }
  const unit_axis = normalize(axis, OUT);

  function path(
    start_points: NDArray,
    end_points: NDArray,
    alpha: number,
  ): NDArray {
    const startRows = asRows(start_points);
    const endRows = asRows(end_points);
    const count = Math.max(startRows.length, endRows.length);
    const starts = broadcast(startRows, count);
    const ends = broadcast(endRows, count);
    const rot = rotation_matrix((alpha - 1) * angle, unit_axis);
    const diffs = starts.map((row, index) => sub(ends[index] ?? [], row));
    const rotated = applyRotation(rot, diffs);
    return array(
      starts.map((row, index) =>
        (row).map((value, k) => value + alpha * (rotated[index]?.[k] ?? 0)),
      ),
    ) as NDArray;
  }

  return path;
}
