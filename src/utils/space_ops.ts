import { type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "./arrays.js";

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
