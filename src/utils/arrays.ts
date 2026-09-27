import { array, NDArray } from "numpy-ts";

export type Numeric = number | bigint;
export type Point3D = NDArray<"float64">;
export type Point3DLike = number[] | readonly number[] | number[][] | readonly (readonly number[])[] | NDArray | ComplexPoint;
export type Vector3DLike = Point3DLike;
export type MatrixLike = number[][] | readonly (readonly number[])[] | NDArray;
export interface ComplexPoint { re: number; im: number }
export type PointRows = number[][];

export function isNDArray(value: unknown): value is NDArray {
  return value instanceof NDArray || (
    typeof value === "object" &&
    value !== null &&
    "shape" in value &&
    "toArray" in value &&
    typeof (value as { toArray: unknown }).toArray === "function"
  );
}

export function numeric(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "object" && value !== null && "re" in value && "im" in value) {
    const c = value as ComplexPoint;
    return c.re;
  }
  return Number(value);
}

function numericOrZero(value: unknown): number {
  if (value === undefined || value === null) return 0;
  return numeric(value);
}

export function pythonTypeName(value: unknown): string {
  if (value === null || value === undefined) return "NoneType";
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
  if (typeof value === "bigint") return "int";
  if (typeof value === "string") return "str";
  if (Array.isArray(value)) return "list";
  if (value instanceof Set) return "set";
  if (value instanceof Map) return "dict";
  if (isNDArray(value)) return "numpy.ndarray";
  if (typeof value === "function") return "function";
  return value.constructor?.name ?? "object";
}

export function cloneNumericArray(value: NDArray | number[] | readonly number[]): NDArray {
  if (isNDArray(value)) return value.copy();
  return array(Array.from(value));
}

export function asArray(value: unknown, emptyDim = 3): NDArray {
  if (isNDArray(value)) return value;
  if (value === undefined || value === null) return array([]);
  if (Array.isArray(value) || ArrayBuffer.isView(value)) return array(value as unknown);
  return array([value]);
}

export function nestedRows(value: unknown): unknown[][] {
  if (isNDArray(value)) {
    const result = value.toArray();
    if (Array.isArray(result) && result.length === 0) return [];
    if (Array.isArray(result) && Array.isArray(result[0])) return result as unknown[][];
    if (Array.isArray(result)) return [result];
    return [];
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    const first = value[0];
    if (Array.isArray(first) || isNDArray(first)) {
      // Normalize nested NDArray rows to plain arrays so a mixed list such as
      // `[start_array, ndarray_point]` (common when a method returns an
      // NDArray) is not mistaken for a row of scalars (which would yield NaN).
      return value.map((item) => (isNDArray(item) ? (item.toArray() as unknown[]) : item)) as unknown[][];
    }
    return [value as unknown[]];
  }
  if (ArrayBuffer.isView(value)) return [Array.from(value as unknown as ArrayLike<number>)];
  return [[value]];
}

export function pointRows(value: unknown, dim = 3): number[][] {
  const rows = nestedRows(value);
  return rows.map((row) => {
    const record = row as unknown;
    const values = Array.isArray(record)
      ? (record as unknown[])
      : isNDArray(record)
        ? ((record as NDArray).toArray() as unknown[])
        : [record];
    const point = values.slice(0, dim).map(numeric);
    while (point.length < dim) point.push(0);
    return point;
  });
}

export function pointFrom(value: unknown, dim = 3): Point3D {
  const rows = pointRows(value, dim);
  if (rows.length === 0) return array([0, 0, 0].slice(0, dim)) as Point3D;
  return array(rows[0]) as Point3D;
}

export function pointsFrom(value: unknown, dim = 3): Point3D {
  const rows = pointRows(value, dim);
  return array(rows.length > 0 ? rows : []).reshape(rows.length, dim) as Point3D;
}

export function pointAt(value: NDArray | number[] | readonly number[], index: number, dim = 3): number[] {
  if (isNDArray(value)) {
    // Fast path: read straight from the backing typed array.  `pointRows`
    // materialises *every* row, so using it here would make a single-anchor
    // access O(n_points) and, e.g., `get_start_anchors` O(n_points^2).
    // Only valid for C-contiguous data (`.data` ignores strides).
    const array = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
    const shape = array.shape;
    const flat = array.data;
    const contiguous = array.flags === undefined || array.flags.C_CONTIGUOUS !== false;
    if (flat !== undefined && contiguous && (shape.length === 1 || shape.length === 2)) {
      const rowDim = shape.length === 2 ? (shape[1] ?? 1) : 1;
      const base = index * rowDim;
      return Array.from({ length: dim }, (_, c) => numeric(flat[base + c] ?? 0));
    }
    const rows = pointRows(value, dim);
    return rows[index] ?? [];
  }
  const source = Array.from(value);
  if (source.length >= dim && typeof source[0] === "number") {
    return source.slice(index * dim, (index + 1) * dim).map(numeric);
  }
  const row = source[index];
  return Array.isArray(row) ? row.map(numeric) : [];
}

export function numberRows(value: unknown): number[][] {
  return nestedRows(value).map((row) => (Array.isArray(row) ? row.map(numeric) : [numeric(row)]));
}

export function addPoints(a: readonly number[], b: readonly number[], dim = 3): number[] {
  return Array.from({ length: dim }, (_, i) => numericOrZero(a[i]) + numericOrZero(b[i]));
}

export function subtractPoints(a: readonly number[], b: readonly number[], dim = 3): number[] {
  return Array.from({ length: dim }, (_, i) => numericOrZero(a[i]) - numericOrZero(b[i]));
}

export function scalePoint(a: readonly number[], scalar: number, dim = 3): number[] {
  return Array.from({ length: dim }, (_, i) => numericOrZero(a[i]) * scalar);
}

export function norm(a: readonly number[]): number {
  return Math.hypot(...Array.from(a, numeric));
}

export function normalize(a: readonly number[]): number[] {
  const length = norm(a);
  return length === 0 ? Array.from(a, numeric) : scalePoint(a, 1 / length, a.length);
}

export function dot(a: readonly number[], b: readonly number[]): number {
  let result = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) result += numeric(a[i]) * numeric(b[i]);
  return result;
}

export function matrixRows(value: unknown, rows?: number, cols?: number): number[][] {
  const result = numberRows(value);
  if (rows !== undefined && cols !== undefined && (result.length !== rows || result.some((row) => row.length !== cols))) {
    throw new Error("Matrix has bad dimensions");
  }
  return result;
}

export function identityMatrix(size: number): number[][] {
  return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)));
}

export function transposeMatrix(matrix: readonly (readonly number[])[]): number[][] {
  return matrix[0]?.map((_, j) => matrix.map((row) => numeric(row[j] ?? 0))) ?? [];
}

export function transformPoint(point: unknown, matrix: unknown, dim = 3): Point3D {
  const p = pointFrom(point, dim);
  const m = matrixRows(matrix);
  return array(m.map((row) => row.reduce((sum, value, j) => sum + value * numeric(p[j] ?? 0), 0))) as Point3D;
}

export function lerp(a: number, b: number, alpha: number): number {
  return a * (1 - alpha) + b * alpha;
}

export function arange(start: number, stop: number, step = 1): number[] {
  // NOTE: mirrors np.arange (half-open interval with values computed as
  // start + i * step rather than accumulated: numpy sizes the range up
  // front, so naive accumulation drifts -- e.g. arange(0, 1, 0.1) must
  // yield 10 values ([0, ..., 0.9]), not 11.
  if (step === 0) throw new Error("step cannot be zero");
  const result: number[] = [];
  if (step > 0) {
    for (let i = 0; start + i * step < stop; i += 1) result.push(start + i * step);
  } else {
    for (let i = 0; start + i * step > stop; i += 1) result.push(start + i * step);
  }
  return result;
}

export function linspace(start: number, stop: number, count: number): number[] {
  if (count <= 0) return [];
  if (count === 1) return [start];
  return Array.from({ length: count }, (_, i) => lerp(start, stop, i / (count - 1)));
}

export function inverseInterpolate(a: number, b: number, value: number): number {
  return (value - a) / (b - a);
}

export function toPythonString(value: unknown): string {
  if (value === undefined) return "None";
  if (value === null) return "None";
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "nan";
    if (value === Infinity) return "inf";
    if (value === -Infinity) return "-inf";
    if (Object.is(value, -0)) return "-0.0";
    return String(value);
  }
  if (typeof value === "bigint") return `${value}`;
  if (typeof value === "function") return value.toString();
  if (isNDArray(value)) return toPythonString(value.toArray());
  if (Array.isArray(value)) return `[${value.map(toPythonRepr).join(", ")}]`;
  if (value instanceof Set) return `{${[...value].map(toPythonRepr).join(", ")}}`;
  if (value instanceof Map) return `{${[...value].map(([key, item]) => `${toPythonRepr(key)}: ${toPythonRepr(item)}`).join(", ")}}`;
  if (typeof value === "object") {
    const candidate = value as { toString?: () => string };
    if (typeof candidate.toString === "function" && candidate.toString !== Object.prototype.toString) return candidate.toString();
    const entries = Object.entries(value as Record<string, unknown>).map(([key, item]) => `${toPythonRepr(key)}: ${toPythonRepr(item)}`);
    return `{${entries.join(", ")}}`;
  }
  return String(value);
}

export function toPythonRepr(value: unknown): string {
  if (typeof value === "string") return `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;
  if (typeof value === "number" && Number.isInteger(value) && !Object.is(value, -0)) return String(value);
  if (typeof value === "bigint") return String(value);
  return toPythonString(value);
}
