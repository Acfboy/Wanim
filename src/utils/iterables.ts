import { array, type NDArray } from "numpy-ts";

function sameValue(first: unknown, second: unknown): boolean {
  if (typeof first === "number" && typeof second === "number") return !Number.isNaN(first) && !Number.isNaN(second) && first === second;
  if (typeof first === "boolean" && typeof second === "number") return Number(first) === second;
  if (typeof first === "number" && typeof second === "boolean") return first === Number(second);
  return Object.is(first, second) || first === second;
}

function ensureHashable(value: unknown): void {
  if (Array.isArray(value) || value instanceof Map || value instanceof Set || (value !== null && typeof value === "object" && (value as { constructor?: { name?: string } }).constructor?.name === "Object")) {
    throw new TypeError(`unhashable type: '${Array.isArray(value) ? "list" : value instanceof Map ? "dict" : value instanceof Set ? "set" : "dict"}'`);
  }
}

export function remove_list_redundancies<T>(lst: readonly T[] | string | Iterable<T>): T[] {
  if (lst instanceof Set) throw new TypeError("'set' object is not reversible");
  const values = typeof lst === "string" ? [...lst] as T[] : Array.isArray(lst) ? [...lst] : Array.from(lst);
  const reversedResult: T[] = [];
  const used: T[] = [];
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    ensureHashable(value);
    if (!used.some((item) => sameValue(item, value))) {
      reversedResult.push(value);
      used.push(value);
    }
  }
  reversedResult.reverse();
  return reversedResult;
}

export function adjacent_n_tuples<T>(objects: readonly T[] | Iterable<T>, n: number): T[][] {
  const items = Array.isArray(objects) ? [...objects] : Array.from(objects);
  if (n === 0) return [];
  const result: T[][] = [];
  for (let index = 0; index < items.length; index += 1) {
    const group: T[] = [];
    for (let k = 0; k < n; k += 1) group.push(items[(index + k) % items.length]);
    result.push(group);
  }
  return result;
}

export function adjacent_pairs<T>(objects: readonly T[] | Iterable<T>): T[][] {
  return adjacent_n_tuples(objects, 2);
}

export function resize_with_interpolation(nparray: unknown, length: number): NDArray {
  const { rows, flat } = elementRows(nparray);
  if (rows.length === length) {
    if (isNDArrayLike(nparray)) return nparray as NDArray;
    return array(flat ? rows.map((row) => row[0]) : rows) as NDArray;
  }
  const count = rows.length;
  const cont_indices = Array.from({ length }, (_, i) => (length === 1 ? 0 : (i * (count - 1)) / (length - 1)));
  const out = cont_indices.map((ci) => {
    const lh = Math.trunc(ci);
    const rh = Math.ceil(ci);
    const a = ci % 1;
    const width = Math.max(rows[lh]?.length ?? 0, rows[rh]?.length ?? 0);
    return Array.from({ length: width }, (_, c) => (1 - a) * (rows[lh]?.[c] ?? 0) + a * (rows[rh]?.[c] ?? 0));
  });
  return array(flat ? out.map((row) => row[0]) : out) as NDArray;
}

function isNDArrayLike(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    "shape" in value &&
    "copy" in value
  );
}

function elementRows(value: unknown): { rows: number[][]; flat: boolean } {
  if (isNDArrayLike(value)) {
    const arr = value as { shape: number[]; toArray: () => unknown };
    const raw = arr.toArray();
    if (arr.shape.length <= 1) {
      const items = (Array.isArray(raw) ? raw : [raw]) as unknown[];
      return { rows: items.map((item) => [Number(item)]), flat: true };
    }
    return { rows: (raw as unknown[][]).map((row) => (Array.isArray(row) ? row : [row]).map(Number)), flat: false };
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return { rows: [], flat: false };
    if (typeof value[0] === "number" || typeof value[0] === "bigint") {
      return { rows: (value as unknown[]).map((item) => [Number(item)]), flat: true };
    }
    return { rows: (value as unknown[][]).map((row) => (Array.isArray(row) ? row : [row]).map(Number)), flat: false };
  }
  return { rows: [], flat: false };
}
