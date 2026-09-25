import { array, zeros, type NDArray } from "numpy-ts";

export function make_even<T, U>(iterable_1: Iterable<T>, iterable_2: Iterable<U>): [T[], U[]] {
  const list_1 = Array.from(iterable_1);
  const list_2 = Array.from(iterable_2);
  const len_list_1 = list_1.length;
  const len_list_2 = list_2.length;
  const length = Math.max(len_list_1, len_list_2);
  if (length === 0) {
    return [[], []];
  }
  return [
    Array.from({ length }, (_, n) => list_1[Math.floor((n * len_list_1) / length)] as T),
    Array.from({ length }, (_, n) => list_2[Math.floor((n * len_list_2) / length)] as U),
  ];
}

export function tuplify(obj: string): [string];
export function tuplify<T>(obj: Iterable<T>): T[];
export function tuplify<T>(obj: T): [T];
export function tuplify<T>(obj: string | Iterable<T> | T): [string] | T[] | [T] {
  if (typeof obj === "string") return [obj];
  if (isIterableObject(obj)) return Array.from(obj as Iterable<T>);
  return [obj as T];
}

export function uniq_chain<T>(...args: Iterable<T>[]): Generator<T, void, void> {
  function* generator(): Generator<T, void, void> {
    const unique_items = new Set<T>();
    for (const iterable of args) {
      for (const x of iterable) {
        if (unique_items.has(x)) continue;
        unique_items.add(x);
        yield x;
      }
    }
  }
  return generator();
}

function isIterableObject(value: unknown): value is Iterable<unknown> {
  return (
    value !== null &&
    value !== undefined &&
    typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function"
  );
}

export function listify<T>(obj: string): string[];
export function listify<T>(obj: Iterable<T>): T[];
export function listify<T>(obj: T): T[];
export function listify<T>(obj: string | Iterable<T> | T): string[] | T[] {
  if (typeof obj === "string") return [obj];
  if (isIterableObject(obj)) return Array.from(obj as Iterable<T>);
  return [obj as T];
}

export function batch_by_property<T, U>(
  items: Iterable<T>,
  property_func: (item: T) => U,
): Array<[T[], U | null]> {
  const batch_prop_pairs: Array<[T[], U | null]> = [];
  let curr_batch: T[] = [];
  let curr_prop: U | null = null;
  for (const item of items) {
    const prop = property_func(item);
    if (prop !== (curr_prop as unknown as U)) {
      if (curr_batch.length > 0) batch_prop_pairs.push([curr_batch, curr_prop]);
      curr_prop = prop as U;
      curr_batch = [item];
    } else {
      curr_batch.push(item);
    }
  }
  if (curr_batch.length > 0) batch_prop_pairs.push([curr_batch, curr_prop]);
  return batch_prop_pairs;
}

export function list_difference_update<T>(l1: Iterable<T>, l2: Iterable<T>): T[] {
  const l2set = new Set(l2);
  return [...l1].filter((e) => !l2set.has(e));
}

export function resize_array(nparray: NDArray, length: number): NDArray {
  const { rows, flat } = elementRows(nparray);
  if (rows.length === length) {
    if (isNDArrayLike(nparray)) return nparray as NDArray;
    return array(flat ? rows.map((row) => row[0]) : rows) as NDArray;
  }
  // np.resize tiles the input in C-order; with the trailing shape preserved
  // this is exactly row cycling.
  if (rows.length === 0) {
    const rest = (nparray as NDArray).shape?.slice(1) ?? [];
    return zeros([length, ...rest]) as NDArray;
  }
  const out: number[][] = Array.from(
    { length },
    (_, i) => [...(rows[i % rows.length] ?? [])],
  );
  return array(flat ? out.map((row) => row[0]) : out) as NDArray;
}

export function stretch_array_to_length(nparray: NDArray, length: number): NDArray {
  const raw = nparray.toArray() as unknown;
  const rowsArray: unknown[] = Array.isArray(raw) ? raw : [];
  const curr_len = rowsArray.length;
  if (curr_len > length) {
    const error = new Error("Trying to stretch array to a length shorter than its own");
    error.name = "Warning";
    throw error;
  }
  const picked = Array.from(
    { length },
    (_, i) => rowsArray[Math.trunc((i / length) * curr_len)] as unknown,
  );
  return array(picked as unknown[]) as NDArray;
}

export function resize_preserving_order(nparray: NDArray, length: number): NDArray {
  const raw = nparray.toArray() as unknown;
  const rowsArray: unknown[] = Array.isArray(raw) ? raw : [];
  if (rowsArray.length === 0) {
    const rest = nparray.shape.slice(1);
    const total = rest.reduce((acc, dim) => acc * dim, length);
    return array(new Array(total).fill(0)).reshape(length, ...rest) as NDArray;
  }
  if (rowsArray.length === length) return nparray;
  const indices = Array.from(
    { length },
    (_, i) => Math.floor((i * rowsArray.length) / length),
  );
  const picked = indices.map((index) => rowsArray[index]);
  return array(picked as unknown[]) as NDArray;
}

export function hash_obj(obj: unknown): number {
  if (obj !== null && typeof obj === "object") {
    if (obj instanceof Map) {
      const pairs = [...obj.entries()].map(
        ([k, v]) => [hash_obj(k), hash_obj(v)] as [number, number],
      );
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      return hashTuple(pairs.map(([k, v]) => k * 31 + v));
    }
    if (obj instanceof Set) {
      const hashes = [...obj].map(hash_obj).sort((a, b) => a - b);
      return hashTuple(hashes);
    }
    if (Array.isArray(obj)) return hashTuple(obj.map(hash_obj));
    if (isNDArrayLike(obj)) {
      return hashTuple(
        JSON.stringify((obj as NDArray).toArray())
          .split("")
          .map((ch) => ch.charCodeAt(0)),
      );
    }
    if ((obj as object).constructor?.name === "Object") {
      const entries = Object.entries(obj as Record<string, unknown>).map(
        ([k, v]) => [hash_obj(k), hash_obj(v)] as [number, number],
      );
      entries.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      return hashTuple(entries.map(([k, v]) => k * 31 + v));
    }
  }
  return hashPrimitive(obj);
}

function hashTuple(items: number[]): number {
  let hash = 0x345678;
  for (const item of items) {
    hash = Math.imul(hash ^ (item | 0), 1000003);
    hash ^= hash >>> 16;
  }
  hash = Math.imul(hash ^ items.length, 0x45d9f3b);
  hash ^= hash >>> 16;
  return hash | 0;
}

function hashPrimitive(value: unknown): number {
  if (value === null || value === undefined) {
    const key = value === null ? "None" : "NoneType";
    return stringHash(key);
  }
  switch (typeof value) {
    case "number":
      if (Number.isNaN(value)) return 0;
      if (Object.is(value, -0)) return stringHash("-0.0");
      return stringHash(`float:${value}`);
    case "bigint":
      return stringHash(`int:${value}`);
    case "boolean":
      return value ? 1 : 0;
    case "string":
      return stringHash(value);
    case "function":
      return stringHash((value as Function).toString());
    default:
      return stringHash(String(value));
  }
}

function stringHash(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = Math.imul(31, hash) + value.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

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
