import { array, type NDArray } from "numpy-ts";
import { isNDArray, pointRows } from "./arrays.js";

export type BezierPoint = number[] | NDArray;
export type BezierAlpha = number | NDArray;

function rowsOf(value: number[] | number[][] | NDArray, dim = 3): number[][] {
  return pointRows(value, dim);
}

function inputNdim(value: number[] | number[][] | NDArray): number {
  if (isNDArray(value)) return (value as NDArray).shape.length;
  if (Array.isArray(value)) return Array.isArray(value[0]) ? 2 : 1;
  return 1;
}

export function interpolate(
  start: number | number[] | NDArray,
  end: number | number[] | NDArray,
  alpha: number | NDArray,
): number | NDArray {
  if (typeof start === "number" && typeof end === "number") {
    if (typeof alpha === "number") return (1 - alpha) * start + alpha * end;
    const alphas = rowsOf(alpha as NDArray, 1).map((row) => row[0] ?? 0);
    return array(
      alphas.map((a) => (1 - a) * (start as number) + a * (end as number)),
    ) as NDArray;
  }
  const startRows = rowsOf(start as number[] | NDArray);
  const endRows = rowsOf(end as number[] | NDArray);
  const dim = Math.max(
    startRows[0]?.length ?? 3,
    endRows[0]?.length ?? 3,
  );
  if (typeof alpha === "number") {
    const a = alpha;
    const result = startRows.map((row, i) => {
      const other = endRows[i] ?? endRows[0] ?? [];
      return Array.from(
        { length: dim },
        (_, c) => (1 - a) * (row[c] ?? 0) + a * (other[c] ?? 0),
      );
    });
    return array(result.length === 1 ? result[0] : result) as NDArray;
  }
  const alphas = rowsOf(alpha, 1).map((row) => row[0] ?? 0);
  const count = Math.max(startRows.length, endRows.length, alphas.length);
  const result = Array.from({ length: count }, (_, i) => {
    const row = startRows[startRows.length === 1 ? 0 : i] ?? [];
    const other = endRows[endRows.length === 1 ? 0 : i] ?? [];
    const a = alphas[alphas.length === 1 ? 0 : i] ?? 0;
    return Array.from(
      { length: dim },
      (_, c) => (1 - a) * (row[c] ?? 0) + a * (other[c] ?? 0),
    );
  });
  return array(result) as NDArray;
}

export const SUBDIVISION_MATRICES: Array<Map<number, number[][]>> = [
  new Map(),
  new Map(),
  new Map(),
  new Map(),
];

export function _get_subdivision_matrix(
  n_points: number,
  n_divisions: number,
): NDArray {
  if (![1, 2, 3, 4].includes(n_points)) {
    const error = new Error(
      "This function does not support subdividing Bézier " +
        "curves with 0 or more than 4 control points.",
    );
    error.name = "NotImplementedError";
    throw error;
  }

  const cached = SUBDIVISION_MATRICES[n_points - 1]?.get(n_divisions);
  if (cached !== undefined) return array(cached) as NDArray;

  const subdivision_matrix: number[][] = Array.from(
    { length: n_points * n_divisions },
    () => new Array(n_points).fill(0),
  );

  if (n_points === 4) {
    for (let i = 0; i < n_divisions; i += 1) {
      const i2 = i * i;
      const i3 = i2 * i;
      const ip1 = i + 1;
      const ip12 = ip1 * ip1;
      const ip13 = ip12 * ip1;
      const nmi = n_divisions - i;
      const nmi2 = nmi * nmi;
      const nmi3 = nmi2 * nmi;
      const nmim1 = nmi - 1;
      const nmim12 = nmim1 * nmim1;
      const nmim13 = nmim12 * nmim1;
      subdivision_matrix[4 * i] = [nmi3, 3 * nmi2 * i, 3 * nmi * i2, i3];
      subdivision_matrix[4 * i + 1] = [
        nmi2 * nmim1,
        2 * nmi * nmim1 * i + nmi2 * ip1,
        nmim1 * i2 + 2 * nmi * i * ip1,
        i2 * ip1,
      ];
      subdivision_matrix[4 * i + 2] = [
        nmi * nmim12,
        nmim12 * i + 2 * nmi * nmim1 * ip1,
        2 * nmim1 * i * ip1 + nmi * ip12,
        i * ip12,
      ];
      subdivision_matrix[4 * i + 3] = [
        nmim13,
        3 * nmim12 * ip1,
        3 * nmim1 * ip12,
        ip13,
      ];
    }
    const divisor = n_divisions * n_divisions * n_divisions;
    for (const row of subdivision_matrix)
      for (let c = 0; c < row.length; c += 1) row[c] /= divisor;
  } else if (n_points === 3) {
    for (let i = 0; i < n_divisions; i += 1) {
      const ip1 = i + 1;
      const nmi = n_divisions - i;
      const nmim1 = nmi - 1;
      subdivision_matrix[3 * i] = [nmi * nmi, 2 * i * nmi, i * i];
      subdivision_matrix[3 * i + 1] = [
        nmi * nmim1,
        i * nmim1 + ip1 * nmi,
        i * ip1,
      ];
      subdivision_matrix[3 * i + 2] = [
        nmim1 * nmim1,
        2 * ip1 * nmim1,
        ip1 * ip1,
      ];
    }
    const divisor = n_divisions * n_divisions;
    for (const row of subdivision_matrix)
      for (let c = 0; c < row.length; c += 1) row[c] /= divisor;
  } else if (n_points === 2) {
    const aux_range = Array.from({ length: n_divisions + 1 }, (_, i) => i);
    for (let r = 0; r < subdivision_matrix.length; r += 1) {
      subdivision_matrix[r][1] =
        r % 2 === 0 ? aux_range[r / 2] : aux_range[(r + 1) / 2];
    }
    const reversed = [...subdivision_matrix].reverse();
    for (let r = 0; r < subdivision_matrix.length; r += 1) {
      subdivision_matrix[r][0] = reversed[r][1];
    }
    for (const row of subdivision_matrix)
      for (let c = 0; c < row.length; c += 1) row[c] /= n_divisions;
  } else {
    for (const row of subdivision_matrix)
      for (let c = 0; c < row.length; c += 1) row[c] = 1;
  }

  SUBDIVISION_MATRICES[n_points - 1]?.set(n_divisions, subdivision_matrix);
  return array(subdivision_matrix) as NDArray;
}

export function get_quadratic_approximation_of_cubic(
  a0: number[] | number[][] | NDArray,
  h0: number[] | number[][] | NDArray,
  h1: number[] | number[][] | NDArray,
  a1: number[] | number[][] | NDArray,
): NDArray {
  const a0c = rowsOf(a0);
  const h0c = rowsOf(h0);
  const h1c = rowsOf(h1);
  const a1c = rowsOf(a1);

  // Python validates inputs by dimensionality: either every argument is a
  // single Point3D (ndim == 1) or every argument is a Point3D_Array
  // (ndim == 2); mixed inputs raise ValueError. num_curves comes from a0.
  const ndims = [a0, h0, h1, a1].map(inputNdim);
  const allSingle = ndims.every((d) => d === 1);
  const allBatched = ndims.every((d) => d === 2);
  if (!allSingle && !allBatched) {
    const error = new Error("All arguments must be Point3D or Point3D_Array.");
    error.name = "ValueError";
    throw error;
  }
  const num_curves = allSingle ? 1 : a0c.length;
  const counts = [a0c.length, h0c.length, h1c.length, a1c.length];
  const dim = a0c[0]?.length ?? h0c[0]?.length ?? h1c[0]?.length ?? a1c[0]?.length ?? 3;
  const valid =
    num_curves >= 1 &&
    counts.every((count) => count === 1 || count === num_curves) &&
    [a0c, h0c, h1c, a1c].every((rows) =>
      rows.every((row) => row.length === dim),
    );
  if (!valid) {
    const error = new Error("All arguments must be Point3D or Point3D_Array.");
    error.name = "ValueError";
    throw error;
  }

  const at = (rows: number[][], i: number): number[] =>
    rows[rows.length === 1 ? 0 : i] ?? new Array(dim).fill(0);

  const result: number[][] = Array.from({ length: 6 * num_curves }, () =>
    new Array(dim).fill(0),
  );
  for (let curve = 0; curve < num_curves; curve += 1) {
    const m0 = at(h0c, curve).map(
      (v, c) => 0.25 * (3 * v + (at(a0c, curve)[c] ?? 0)),
    );
    const m1 = at(h1c, curve).map(
      (v, c) => 0.25 * (3 * v + (at(a1c, curve)[c] ?? 0)),
    );
    const k = m0.map((v, c) => 0.5 * (v + (m1[c] ?? 0)));
    result[6 * curve] = [...at(a0c, curve)];
    result[6 * curve + 1] = [...m0];
    result[6 * curve + 2] = [...k];
    result[6 * curve + 3] = [...k];
    result[6 * curve + 4] = [...m1];
    result[6 * curve + 5] = [...at(a1c, curve)];
  }
  return array(result) as NDArray;
}
