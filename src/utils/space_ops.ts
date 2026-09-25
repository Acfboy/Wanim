import { array, average, type NDArray } from "numpy-ts";
import { isNDArray, numeric, type ComplexPoint } from "./arrays.js";
import { DOWN, OUT, RIGHT, TAU, UP } from "../constants.js";

type VectorInput = NDArray | number[] | number[][] | readonly number[];

function toRows(value: VectorInput): { rows: number[][]; single: boolean } {
  if (isNDArray(value)) {
    const shape = (value as NDArray).shape;
    const raw = (value as NDArray).toArray() as unknown;
    if (shape.length === 2) {
      return {
        rows: (raw as unknown[][]).map((row) => (Array.isArray(row) ? row : [row]).map(numeric)),
        single: false,
      };
    }
    const flat = (Array.isArray(raw) ? raw : [raw]) as unknown[];
    return { rows: [flat.map(numeric)], single: true };
  }
  if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0])) {
    return {
      rows: (value as unknown[][]).map((row) => (Array.isArray(row) ? row : [row]).map(numeric)),
      single: false,
    };
  }
  return { rows: [[... (value as unknown[])].map(numeric)], single: true };
}

export function norm_squared(v: number[] | readonly number[] | NDArray): number {
  const flat = toRows(v as VectorInput).rows[0] ?? [];
  return flat.reduce((sum, value) => sum + value * value, 0);
}

export function cross(
  v1: number[] | readonly number[] | NDArray,
  v2: number[] | readonly number[] | NDArray,
): number[] {
  const a = toRows(v1 as VectorInput).rows[0] ?? [0, 0, 0];
  const b = toRows(v2 as VectorInput).rows[0] ?? [0, 0, 0];
  return [
    (a[1] ?? 0) * (b[2] ?? 0) - (a[2] ?? 0) * (b[1] ?? 0),
    (a[2] ?? 0) * (b[0] ?? 0) - (a[0] ?? 0) * (b[2] ?? 0),
    (a[0] ?? 0) * (b[1] ?? 0) - (a[1] ?? 0) * (b[0] ?? 0),
  ];
}

export function normalize(
  vect: number[] | readonly number[] | NDArray,
  fall_back?: number[] | readonly number[] | NDArray | null,
): number[] {
  const flat = toRows(vect as VectorInput).rows[0] ?? [];
  const norm = Math.sqrt(flat.reduce((sum, value) => sum + value * value, 0));
  if (norm > 0) return flat.map((value) => value / norm);
  if (fall_back !== undefined && fall_back !== null) {
    return [...(toRows(fall_back as VectorInput).rows[0] ?? [])];
  }
  return new Array(flat.length).fill(0);
}

export function cartesian_to_spherical(
  vec: number[] | readonly number[] | NDArray,
): number[] {
  const flat = toRows(vec as VectorInput).rows[0] ?? [0, 0, 0];
  const norm = Math.sqrt(
    flat.reduce((sum, value) => sum + value * value, 0),
  );
  if (norm === 0) return [0, 0, 0];
  const r = norm;
  const phi = Math.acos((flat[2] ?? 0) / r);
  const theta = Math.atan2(flat[1] ?? 0, flat[0] ?? 0);
  return [r, theta, phi];
}

export function cross2d(a: VectorInput, b: VectorInput): number | number[] {
  const left = toRows(a);
  const right = toRows(b);
  if (left.single && right.single) {
    const u = left.rows[0] ?? [0, 0];
    const v = right.rows[0] ?? [0, 0];
    return (u[0] ?? 0) * (v[1] ?? 0) - (v[0] ?? 0) * (u[1] ?? 0);
  }
  const count = Math.max(left.rows.length, right.rows.length);
  const result: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const u = left.rows[left.single ? 0 : index] ?? [0, 0];
    const v = right.rows[right.single ? 0 : index] ?? [0, 0];
    result.push((u[0] ?? 0) * (v[1] ?? 0) - (u[1] ?? 0) * (v[0] ?? 0));
  }
  return result;
}

export function normalize_along_axis(arr: NDArray, axis: number): NDArray {
  // Faithful port of manim's implementation, which broadcasts via
  // `np.repeat(norms, shape[axis]).reshape(shape)`:
  //   norms = sqrt((array * array).sum(axis)); norms[norms == 0] = 1
  //   buffed = repeat(norms, shape[axis]).reshape(shape); array /= buffed
  // NOTE: for non-last axes this misaligns norms against elements (each
  // norm is repeated shape[axis] times contiguously in C-order). The port
  // replicates that behaviour exactly so results match Python bit-for-bit.
  const shape = [...arr.shape];
  const rank = shape.length;
  const ax = axis < 0 ? axis + rank : axis;
  const axisSize = shape[ax] ?? 0;
  const total = shape.reduce((a, b) => a * b, 1);
  const strides = stridesFor(shape);
  const reduced = shape.filter((_, i) => i !== ax);
  const reducedStrides = stridesFor(reduced);
  const indexOf = (flat: number): number[] =>
    shape.map((size, i) => Math.floor(flat / (strides[i] ?? 1)) % (size ?? 1));
  const reducedIndexOf = (full: number[]): number => {
    const coords = full.filter((_, i) => i !== ax);
    return coords.reduce(
      (acc, coord, i) => acc + coord * (reducedStrides[i] ?? 1),
      0,
    );
  };
  const reducedCount = reduced.reduce((a, b) => a * b, 1);
  const sums = new Array(reducedCount).fill(0);
  const values = new Array(total).fill(0);
  for (let flat = 0; flat < total; flat += 1) {
    const full = indexOf(flat);
    const value = Number(arr.get(full));
    values[flat] = value;
    sums[reducedIndexOf(full)] += value * value;
  }
  const norms = sums.map((sum) => {
    const norm = Math.sqrt(sum);
    return norm === 0 ? 1 : norm;
  });
  for (let flat = 0; flat < total; flat += 1) {
    const full = indexOf(flat);
    arr.set(full, values[flat]! / (norms[Math.floor(flat / axisSize)] ?? 1));
  }
  return arr;
}

function stridesFor(shape: number[]): number[] {
  const strides = new Array(shape.length).fill(0);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    strides[i] = stride;
    stride *= shape[i] ?? 1;
  }
  return strides;
}

export type Quaternion = readonly number[] | number[] | NDArray;

function quatRows(value: Quaternion): number[] {
  if (isNDArray(value)) {
    const raw = (value as NDArray).toArray() as unknown;
    return (Array.isArray(raw) ? raw : [raw]).map(numeric);
  }
  return [...(value as readonly number[])].map(numeric);
}

export function quaternion_mult(...quats: Quaternion[]): number[] | number[][] {
  if (quats.length === 0) {
    return [1, 0, 0, 0];
  }
  let result = quatRows(quats[0]!);
  for (const next of quats.slice(1)) {
    const [w1 = 0, x1 = 0, y1 = 0, z1 = 0] = result;
    const [w2 = 0, x2 = 0, y2 = 0, z2 = 0] = quatRows(next);
    result = [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
      w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ];
  }
  return result;
}

export function quaternion_conjugate(quaternion: Quaternion): number[] {
  const result = quatRows(quaternion);
  for (let index = 1; index < result.length; index += 1) {
    result[index] = -(result[index] ?? 0);
  }
  return result;
}

export function R3_to_complex(point: readonly number[] | number[] | NDArray): ComplexPoint {
  const rows = quatRows(point as number[]);
  return { re: rows[0] ?? 0, im: rows[1] ?? 0 };
}

export function complex_to_R3(complex_num: ComplexPoint | readonly [number, number] | number): number[] {
  if (typeof complex_num === "number") {
    return [complex_num, 0, 0];
  }
  if (Array.isArray(complex_num)) {
    return [complex_num[0] ?? 0, complex_num[1] ?? 0, 0];
  }
  return [(complex_num as ComplexPoint).re ?? 0, (complex_num as ComplexPoint).im ?? 0, 0];
}

export function center_of_mass(points: number[][] | NDArray): number[] {
  if (isNDArray(points)) {
    const result = average(points as never, 0, undefined) as unknown;
    if (isNDArray(result)) {
      return ((result as NDArray).toArray() as unknown[]).map(numeric);
    }
    return [numeric(result)];
  }
  const rows = (points as number[][]).map((row) => [...row].map(numeric));
  if (rows.length === 0) {
    return [];
  }
  const dim = Math.max(...rows.map((row) => row.length));
  return Array.from({ length: dim }, (_, index) => rows.reduce((sum, row) => sum + (row[index] ?? 0), 0) / rows.length);
}

export function angle_of_vector(vector: number[] | readonly number[] | NDArray | number[][]): number | number[] {
  if (isNDArray(vector)) {
    const shape = (vector as NDArray).shape;
    if (shape.length > 1) {
      if ((shape[0] ?? 0) < 2) {
        const error = new Error("Vector must have the correct dimensions. (2, n)");
        error.name = "ValueError";
        throw error;
      }
      const raw = (vector as NDArray).toArray() as unknown[][];
      const xs = (Array.isArray(raw[0]) ? raw[0] : [raw[0]]).map(numeric);
      const ys = (Array.isArray(raw[1]) ? raw[1] : [raw[1]]).map(numeric);
      return xs.map((x, index) => Math.atan2(ys[index] ?? 0, x));
    }
    const flat = ((vector as NDArray).toArray() as unknown[]).map(numeric);
    return Math.atan2(flat[1] ?? 0, flat[0] ?? 0);
  }
  if (Array.isArray(vector) && vector.length > 0 && Array.isArray(vector[0])) {
    // Python falls through to complex(*vector[:2]) here, which raises
    // TypeError for nested sequences. Mirror that failure instead of guessing.
    const error = new TypeError(
      "angle_of_vector() requires a flat sequence of numbers, not a nested sequence",
    );
    throw error;
  }
  const flat = [...(vector as readonly number[])].map(numeric);
  return Math.atan2(flat[1] ?? 0, flat[0] ?? 0);
}

export function shoelace(x_y: number[][] | NDArray): number {
  let xs: number[];
  let ys: number[];
  if (isNDArray(x_y)) {
    const raw = (x_y as NDArray).toArray() as unknown[][];
    xs = raw.map((row) => numeric((row as unknown[])[0]));
    ys = raw.map((row) => numeric((row as unknown[])[1]));
  } else {
    xs = (x_y as number[][]).map((row) => numeric(row[0]));
    ys = (x_y as number[][]).map((row) => numeric(row[1]));
  }
  // np.trapezoid(y, x): sum over segments (x[i+1]-x[i]) * (y[i+1]+y[i]) / 2
  let total = 0;
  for (let index = 0; index + 1 < xs.length; index += 1) {
    total += ((xs[index + 1] ?? 0) - (xs[index] ?? 0)) * ((ys[index + 1] ?? 0) + (ys[index] ?? 0)) / 2;
  }
  return total;
}

export function rotation_matrix(
  angle: number,
  axis: VectorInput,
  homogeneous = false,
): NDArray {
  const flat = toRows(axis).rows[0] ?? [0, 0, 0];
  const norm = Math.hypot(flat[0] ?? 0, flat[1] ?? 0, flat[2] ?? 0);
  let rotation: number[][];
  if (norm === 0) {
    rotation = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  } else {
    const kx = (flat[0] ?? 0) / norm;
    const ky = (flat[1] ?? 0) / norm;
    const kz = (flat[2] ?? 0) / norm;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const oneMinusC = 1 - c;
    rotation = [
      [
        c + kx * kx * oneMinusC,
        kx * ky * oneMinusC - kz * s,
        kx * kz * oneMinusC + ky * s,
      ],
      [
        ky * kx * oneMinusC + kz * s,
        c + ky * ky * oneMinusC,
        ky * kz * oneMinusC - kx * s,
      ],
      [
        kz * kx * oneMinusC - ky * s,
        kz * ky * oneMinusC + kx * s,
        c + kz * kz * oneMinusC,
      ],
    ];
  }
  if (!homogeneous) {
    return array(rotation) as NDArray;
  }
  const homogeneousMatrix = [
    [rotation[0]?.[0] ?? 0, rotation[0]?.[1] ?? 0, rotation[0]?.[2] ?? 0, 0],
    [rotation[1]?.[0] ?? 0, rotation[1]?.[1] ?? 0, rotation[1]?.[2] ?? 0, 0],
    [rotation[2]?.[0] ?? 0, rotation[2]?.[1] ?? 0, rotation[2]?.[2] ?? 0, 0],
    [0, 0, 0, 1],
  ];
  return array(homogeneousMatrix) as NDArray;
}

export function rotation_about_z(angle: number): NDArray {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return array([
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ]) as NDArray;
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

export function quaternion_from_angle_axis(
  angle: number,
  axis: Quaternion | VectorInput,
  axis_normalized = false,
): number[] {
  const normalized = axis_normalized
    ? quatRows(axis as Quaternion)
    : normalize(axis as number[] | readonly number[] | NDArray);
  const half = angle / 2;
  const s = Math.sin(half);
  return [Math.cos(half), ...normalized.map((value) => s * value)];
}

export function angle_axis_from_quaternion(
  quaternion: Quaternion,
): [number, number[]] {
  const flat = quatRows(quaternion);
  const axis = normalize(flat.slice(1), [1, 0, 0]);
  let angle = 2 * Math.acos(flat[0] ?? 0);
  if (angle > TAU / 2) {
    angle = TAU - angle;
  }
  return [angle, axis];
}

export function rotation_matrix_transpose_from_quaternion(
  quat: Quaternion,
): number[][] {
  const quat_inv = quaternion_conjugate(quat);
  return [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(
    (basis) => (quaternion_mult(quat, [0, ...basis], quat_inv) as number[]).slice(1),
  );
}

export function z_to_vector(vector: VectorInput): NDArray {
  const axis_z = normalize(vector as number[] | readonly number[] | NDArray);
  let axis_y = normalize(cross(axis_z, [...RIGHT]));
  let axis_x = cross(axis_y, axis_z);
  if (Math.hypot(...axis_y) === 0) {
    // The vector passed just so happened to be in the x direction.
    axis_x = normalize(cross([...UP], axis_z));
    axis_y = cross(axis_x, axis_z).map((value) => -value);
  }
  return array([
    [axis_x[0] ?? 0, axis_y[0] ?? 0, axis_z[0] ?? 0],
    [axis_x[1] ?? 0, axis_y[1] ?? 0, axis_z[1] ?? 0],
    [axis_x[2] ?? 0, axis_y[2] ?? 0, axis_z[2] ?? 0],
  ]) as NDArray;
}

export function angle_between_vectors(
  v1: VectorInput,
  v2: VectorInput,
): number {
  const a = normalize(v1 as number[] | readonly number[] | NDArray);
  const b = normalize(v2 as number[] | readonly number[] | NDArray);
  const diff = Math.hypot(...a.map((value, index) => value - (b[index] ?? 0)));
  const sum = Math.hypot(...a.map((value, index) => value + (b[index] ?? 0)));
  return 2 * Math.atan2(diff, sum);
}

export function complex_func_to_R3_func(
  complex_func: (value: ComplexPoint) => ComplexPoint | number,
): (point: readonly number[] | number[] | NDArray) => number[] {
  return (p) => complex_to_R3(complex_func(R3_to_complex(p)));
}

export function midpoint(
  point1: VectorInput,
  point2: VectorInput,
): number[] {
  const rows = [
    toRows(point1).rows[0] ?? [],
    toRows(point2).rows[0] ?? [],
  ];
  return center_of_mass(rows);
}

export function line_intersection(
  line1: VectorInput,
  line2: VectorInput,
): number[] {
  const first = toRows(line1).rows;
  const second = toRows(line2).rows;
  if ([...first, ...second].some((point) => (point[2] ?? 0) !== 0)) {
    throw valueError("Coords must be in the xy-plane.");
  }
  const padded = [first, second].map((line) =>
    line.map((point) => [point[0] ?? 0, point[1] ?? 0, 1]),
  );
  const [a1 = [0, 0, 1], b1 = [0, 0, 1]] = padded[0] ?? [];
  const [a2 = [0, 0, 1], b2 = [0, 0, 1]] = padded[1] ?? [];
  const l1 = cross(a1, b1);
  const l2 = cross(a2, b2);
  const [x = 0, y = 0, z = 0] = cross(l1, l2);
  if (z === 0) {
    throw valueError(
      "The lines are parallel, there is no unique intersection point.",
    );
  }
  return [x / z, y / z, 0];
}

export function find_intersection(
  p0s: number[][] | NDArray,
  v0s: number[][] | NDArray,
  p1s: number[][] | NDArray,
  v1s: number[][] | NDArray,
  threshold = 1e-5,
): number[][] {
  const p0rows = toRows(p0s).rows;
  const v0rows = toRows(v0s).rows;
  const p1rows = toRows(p1s).rows;
  const v1rows = toRows(v1s).rows;
  const counts = [p0rows.length, v0rows.length, p1rows.length, v1rows.length];
  if (!counts.every((count) => count === counts[0])) {
    throw valueError(
      `zip() argument 2 is longer than argument 1 (strict zip lengths: ${counts.join(", ")})`,
    );
  }
  return p0rows.map((p0, index) => {
    const v0 = v0rows[index] ?? [];
    const p1 = p1rows[index] ?? [];
    const v1 = v1rows[index] ?? [];
    const normal = cross(v1, cross(v0, v1));
    const denom = Math.max(
      v0.reduce((sum, value, i) => sum + value * (normal[i] ?? 0), 0),
      threshold,
    );
    const along = p1
      .map((value, i) => value - (p0[i] ?? 0))
      .reduce((sum, value, i) => sum + value * (normal[i] ?? 0), 0) / denom;
    return p0.map((value, i) => value + along * (v0[i] ?? 0));
  });
}

export function shoelace_direction(x_y: number[][] | NDArray): string {
  return shoelace(x_y) > 0 ? "CW" : "CCW";
}

export function get_unit_normal(
  v1: number[] | readonly number[] | NDArray,
  v2: number[] | readonly number[] | NDArray,
  tol = 1e-6,
): number[] {
  const np_v1 = [...(toRows(v1 as VectorInput).rows[0] ?? [])];
  const np_v2 = [...(toRows(v2 as VectorInput).rows[0] ?? [])];

  // NOTE: Python divides by the greatest absolute component (not the norm).
  const div1 = Math.max(...np_v1.map((value) => Math.abs(value)));
  const div2 = Math.max(...np_v2.map((value) => Math.abs(value)));
  let u: number[];
  if (div1 === 0.0) {
    if (div2 === 0.0) return [...(DOWN as unknown as number[])];
    u = np_v2.map((value) => value / div2);
  } else if (div2 === 0.0) {
    u = np_v1.map((value) => value / div1);
  } else {
    // Normal scenario: v1 and v2 are both non-null.
    const u1 = np_v1.map((value) => value / div1);
    const u2 = np_v2.map((value) => value / div2);
    const cp = cross(u1, u2);
    const cp_norm = Math.sqrt(norm_squared(cp));
    if (cp_norm > tol) return cp.map((value) => value / cp_norm);
    // Otherwise, v1 and v2 were aligned.
    u = u1;
  }

  // If you are here, you have an "unique", non-zero, unit-ish vector u.
  // If it's also too aligned to the Z axis, just return DOWN.
  if (Math.abs(u[0] ?? 0) < tol && Math.abs(u[1] ?? 0) < tol) {
    return [...(DOWN as unknown as number[])];
  }
  // Otherwise rotate u in the plane it shares with the Z axis, 90° TOWARDS
  // the Z axis: (u x [0, 0, 1]) x u == [-xz, -yz, x²+y²].
  const cp = [
    -(u[0] ?? 0) * (u[2] ?? 0),
    -(u[1] ?? 0) * (u[2] ?? 0),
    (u[0] ?? 0) * (u[0] ?? 0) + (u[1] ?? 0) * (u[1] ?? 0),
  ];
  const cp_norm = Math.sqrt(norm_squared(cp));
  // Because the norm(u) == 0 case was filtered in the beginning, there is
  // no need to check if the norm of cp is 0.
  return cp.map((value) => value / cp_norm);
}
