import { average, type NDArray } from "numpy-ts";
import { isNDArray, numeric, type ComplexPoint } from "./arrays.js";

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
