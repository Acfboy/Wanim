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
