import { array, roots as polyRoots, type NDArray } from "numpy-ts";
import { isNDArray, nestedRows, numeric, pointRows } from "./arrays.js";

function rawRows(value: number[] | number[][] | NDArray): number[][] {
  return nestedRows(value).map((row) =>
    (Array.isArray(row) ? row : [row]).map(numeric),
  );
}

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

export type BezierFn = (t: number | NDArray) => number[] | number[][] | NDArray;

function asControlCurves(points: number[][] | number[][][] | NDArray): {
  curves: number[][][];
  batched: boolean;
} {
  if (isNDArray(points)) {
    const shape = (points as NDArray).shape;
    const nested = (points as NDArray).toArray() as unknown;
    if (shape.length === 3) {
      // Batched layout (d+1, M, 3): transpose to M curves of (d+1) points.
      const stacked = nested as number[][][];
      const degree = stacked.length - 1;
      const count = (stacked[0] as number[][]).length;
      return {
        curves: Array.from({ length: count }, (_, m) =>
          Array.from({ length: degree + 1 }, (_, i) => [...((stacked[i] as number[][])[m] ?? [])].map(numeric)),
        ),
        batched: true,
      };
    }
    return { curves: [rawRows(points as number[] | number[][] | NDArray)], batched: false };
  }
  if (
    Array.isArray(points) &&
    points.length > 0 &&
    Array.isArray(points[0]) &&
    (points[0] as unknown[]).length > 0 &&
    Array.isArray((points[0] as unknown[])[0])
  ) {
    const stacked = points as number[][][];
    const degree = stacked.length - 1;
    const count = (stacked[0] as number[][]).length;
    return {
      curves: Array.from({ length: count }, (_, m) =>
        Array.from({ length: degree + 1 }, (_, i) => [...((stacked[i] as number[][])[m] ?? [])].map(numeric)),
      ),
      batched: true,
    };
  }
  return { curves: [rawRows(points as number[][])], batched: false };
}

function evalBezierCurve(curve: number[][], t: number): number[] {
  const degree = curve.length - 1;
  const dim = curve[0]?.length ?? 3;
  if (degree === 0) return [...(curve[0] ?? [])];
  if (degree === 1) {
    const [p0 = [], p1 = []] = curve;
    return Array.from({ length: dim }, (_, c) => (p0[c] ?? 0) + t * ((p1[c] ?? 0) - (p0[c] ?? 0)));
  }
  if (degree === 2) {
    const [p0 = [], p1 = [], p2 = []] = curve;
    const t2 = t * t;
    const mt = 1 - t;
    const mt2 = mt * mt;
    return Array.from(
      { length: dim },
      (_, c) => mt2 * (p0[c] ?? 0) + 2 * t * mt * (p1[c] ?? 0) + t2 * (p2[c] ?? 0),
    );
  }
  if (degree === 3) {
    const [p0 = [], p1 = [], p2 = [], p3 = []] = curve;
    const t2 = t * t;
    const t3 = t2 * t;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    return Array.from(
      { length: dim },
      (_, c) =>
        mt3 * (p0[c] ?? 0) +
        3 * t * mt2 * (p1[c] ?? 0) +
        3 * t2 * mt * (p2[c] ?? 0) +
        t3 * (p3[c] ?? 0),
    );
  }
  // De Casteljau for higher degrees (mirrors nth_grade_bezier).
  let level = curve.map((point) => [...point]);
  for (let i = 0; i < degree; i += 1) {
    level = level
      .slice(0, degree - i)
      .map((point, k) =>
        point.map((v, c) => v + t * ((level[k + 1]?.[c] ?? 0) - v)),
      );
  }
  return [...(level[0] ?? [])];
}

function colValues(t: NDArray): number[] {
  return rowsOf(t, 1).map((row) => row[0] ?? 0);
}

export function bezier(points: number[][] | number[][][] | NDArray): BezierFn {
  const { curves, batched } = asControlCurves(points);
  if (!batched) {
    const single = curves[0]!;
    return (t: number | NDArray): number[] | NDArray => {
      if (typeof t === "number") return evalBezierCurve(single, t);
      const values = colValues(t as NDArray);
      return array(values.map((value) => evalBezierCurve(single, value))) as NDArray;
    };
  }
  return (t: number | NDArray): number[][] | NDArray => {
    if (typeof t === "number") {
      return curves.map((curve) => evalBezierCurve(curve, t));
    }
    const values = colValues(t as NDArray);
    return array(
      curves.map((curve, m) => evalBezierCurve(curve, values[m] ?? values[0] ?? 0)),
    ) as NDArray;
  };
}

function alphaValues(alpha: NDArray): number[] {
  // NOTE: a 1-D alpha array holds one alpha per element (numpy broadcasting),
  // but rowsOf would wrap it as a single row; unwrap it elementwise instead.
  if (isNDArray(alpha) && (alpha as NDArray).shape.length === 1) {
    return ((alpha as NDArray).toArray() as unknown[]).map(numeric);
  }
  return rowsOf(alpha as NDArray, 1).map((row) => row[0] ?? 0);
}

export function interpolate(
  start: number | number[] | NDArray,
  end: number | number[] | NDArray,
  alpha: number | NDArray,
): number | NDArray {
  if (typeof start === "number" && typeof end === "number") {
    if (typeof alpha === "number") return (1 - alpha) * start + alpha * end;
    const alphas = alphaValues(alpha as NDArray);
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
  const alphas = alphaValues(alpha);
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

export function inverse_interpolate(
  start: number | number[] | NDArray,
  end: number | number[] | NDArray,
  value: number | number[] | NDArray,
): number | NDArray {
  if (typeof start === "number" && typeof end === "number" && typeof value === "number") {
    return (value - start) / (end - start);
  }
  if (typeof start === "number" && typeof end === "number") {
    // NOTE: Python broadcasts scalar (start, end) over an array value:
    // (value - start) / (end - start), preserving the value's shape
    // (e.g. shape (2,) stays (2,), not (2, 3)). Mirror that here instead of
    // routing scalars through the point-row path below.
    const lo = start;
    const hi = end;
    const mapScalar = (item: unknown): unknown =>
      Array.isArray(item)
        ? item.map(mapScalar)
        : (numeric(item) - lo) / (hi - lo);
    const raw: unknown = isNDArray(value)
      ? ((value as NDArray).toArray() as unknown)
      : value;
    const mapped = mapScalar(raw);
    if (typeof mapped === "number") return mapped;
    return array(mapped as number | number[]) as NDArray;
  }
  const startRows = rowsOf(start as number[] | NDArray);
  const endRows = rowsOf(end as number[] | NDArray);
  const valueRows = rowsOf(value as number[] | NDArray);
  const dim = Math.max(
    startRows[0]?.length ?? 1,
    endRows[0]?.length ?? 1,
    valueRows[0]?.length ?? 1,
  );
  const count = Math.max(startRows.length, endRows.length, valueRows.length);
  const result = Array.from({ length: count }, (_, i) => {
    const s = startRows[startRows.length === 1 ? 0 : i] ?? [];
    const e = endRows[endRows.length === 1 ? 0 : i] ?? [];
    const v = valueRows[valueRows.length === 1 ? 0 : i] ?? [];
    return Array.from(
      { length: dim },
      (_, c) => ((v[c] ?? 0) - (s[c] ?? 0)) / ((e[c] ?? 0) - (s[c] ?? 0)),
    );
  });
  if (result.length === 1 && dim === 1) return result[0]?.[0] ?? NaN;
  return array(result.length === 1 ? result[0] : result) as NDArray;
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

export function integer_interpolate(
  start: number,
  end: number,
  alpha: number,
): [number, number] {
  if (alpha >= 1) return [Math.trunc(end - 1), 1.0];
  if (alpha <= 0) return [Math.trunc(start), 0];
  const value = Math.trunc(interpolate(start, end, alpha) as number);
  // NOTE: Python's % is floor-modulo (result takes the divisor's sign, so it
  // always lands in [0, 1) here); JS % is truncated and can go negative.
  const residue = ((end - start) * alpha) % 1;
  return [value, residue < 0 ? residue + 1 : residue];
}

export function match_interpolate(
  new_start: number,
  new_end: number,
  old_start: number,
  old_end: number,
  old_value: number | number[] | NDArray,
): number | NDArray {
  const old_alpha = inverse_interpolate(old_start, old_end, old_value);
  return interpolate(new_start, new_end, old_alpha);
}

export let CP_CLOSED_MEMO: number[] = [1 / 3];
export let UP_CLOSED_MEMO: number[] = [1 / 3];

export let CP_OPEN_MEMO: number[] = [0.5];

export function get_smooth_cubic_bezier_handle_points(
  anchors: number[][] | NDArray,
): [NDArray, NDArray] {
  const rows = rawRows(anchors);
  const n_anchors = rows.length;
  const dim = rows[0]?.length ?? 0;

  if (n_anchors === 1) {
    return [array([]).reshape(0, dim) as NDArray, array([]).reshape(0, dim) as NDArray];
  }

  if (n_anchors === 2) {
    const val = interpolate(
      rows[0] ?? [],
      rows[1] ?? [],
      array([[1 / 3], [2 / 3]]),
    ) as NDArray;
    const valRows = pointRows(val, dim);
    return [array(valRows[0] ?? []) as NDArray, array(valRows[1] ?? []) as NDArray];
  }

  if (is_closed(anchors)) {
    return get_smooth_closed_cubic_bezier_handle_points(anchors);
  }
  return get_smooth_open_cubic_bezier_handle_points(anchors);
}

export function get_smooth_closed_cubic_bezier_handle_points(
  anchors: number[][] | NDArray,
): [NDArray, NDArray] {
  const A = rawRows(anchors);
  const N = A.length - 1;
  const dim = A[0]?.length ?? 0;

  const len_memo = CP_CLOSED_MEMO.length;
  let cp: number[];
  let up: number[];
  if (len_memo < N - 1) {
    cp = new Array(N - 1).fill(0);
    up = new Array(N - 1).fill(0);
    for (let i = 0; i < len_memo; i += 1) {
      cp[i] = CP_CLOSED_MEMO[i] ?? 0;
      up[i] = UP_CLOSED_MEMO[i] ?? 0;
    }
    for (let i = len_memo; i < N - 1; i += 1) {
      cp[i] = 1 / (4 - (cp[i - 1] ?? 0));
      up[i] = -(cp[i] ?? 0) * (up[i - 1] ?? 0);
    }
    CP_CLOSED_MEMO = cp;
    UP_CLOSED_MEMO = up;
  } else {
    cp = CP_CLOSED_MEMO.slice(0, N - 1);
    up = UP_CLOSED_MEMO.slice(0, N - 1);
  }

  const cpAt = (index: number): number => cp[index] ?? 0;
  const upAt = (index: number): number => up[index] ?? 0;

  // The last element of u' is different.
  const cp_last_division = 1 / (3 - cpAt(N - 2));
  const up_last = cp_last_division * (1 - upAt(N - 2));

  // Backward Substitution 1: calculate q.
  const q: number[][] = Array.from({ length: N }, () => new Array(dim).fill(0));
  q[N - 1] = new Array(dim).fill(up_last);
  for (let i = N - 2; i >= 0; i -= 1) {
    const next = q[i + 1] ?? [];
    q[i] = Array.from(
      { length: dim },
      (_, c) => (upAt(i) - cpAt(i) * (next[c] ?? 0)),
    );
  }

  // Forward Substitution 2: calculate Dp (D prime).
  const aux: number[][] = Array.from({ length: N }, (_, i) =>
    (A[i] ?? []).map((v, c) => 4 * v + 2 * ((A[i + 1] ?? [])[c] ?? 0)),
  );
  const Dp: number[][] = Array.from({ length: N }, () => new Array(dim).fill(0));
  Dp[0] = (aux[0] ?? []).map((v) => v / 3);
  for (let i = 1; i < N - 1; i += 1) {
    Dp[i] = (aux[i] ?? []).map((v, c) => cpAt(i) * (v - ((Dp[i - 1] ?? [])[c] ?? 0)));
  }
  Dp[N - 1] = (aux[N - 1] ?? []).map(
    (v, c) => cp_last_division * (v - ((Dp[N - 2] ?? [])[c] ?? 0)),
  );

  // Backward Substitution: Y aliases Dp (exactly as in Python).
  const Y = Dp;
  for (let i = N - 2; i >= 0; i -= 1) {
    Y[i] = (Dp[i] ?? []).map((v, c) => v - cpAt(i) * ((Y[i + 1] ?? [])[c] ?? 0));
  }

  // H1 = Y - 1/(1 + q0 + q{N-1}) * q * (Y0 + Y{N-1}).
  // NOTE: v is an N-vector with v_0 = v_{N-1} = 1, so v^t q = q_0 + q_{N-1}
  // is a dim-vector and the division below is elementwise per component:
  // H1[i][c] = Y[i][c] - q[i][c] * (Y[0][c] + Y[N-1][c]) / (1 + q[0][c] + q[N-1][c]).
  // NOTE: the subtraction term is associated exactly like NumPy evaluates
  // `Y - 1/(1 + q[0] + q[N-1]) * q * (Y[0] + Y[N-1])` (left-associative per
  // component); any other grouping differs by 1 ulp in cancellation regimes.
  const q0 = q[0] ?? [];
  const qLast = q[N - 1] ?? [];
  const y0 = Y[0] ?? [];
  const yLast = Y[N - 1] ?? [];
  const H1: number[][] = Y.map((row, i) =>
    (row ?? []).map((v, c) => {
      const s = 1 / (1 + (q0[c] ?? 0) + (qLast[c] ?? 0));
      return v - s * (((q[i] ?? [])[c] ?? 0) * ((y0[c] ?? 0) + (yLast[c] ?? 0)));
    }),
  );

  // H2[0:N-1] = 2*A[1:N] - H1[1:N]; H2[N-1] = 2*A[N] - H1[0].
  const H2: number[][] = Array.from({ length: N }, () => new Array(dim).fill(0));
  for (let i = 0; i < N - 1; i += 1) {
    H2[i] = (A[i + 1] ?? []).map((v, c) => 2 * v - ((H1[i + 1] ?? [])[c] ?? 0));
  }
  H2[N - 1] = (A[N] ?? []).map((v, c) => 2 * v - ((H1[0] ?? [])[c] ?? 0));

  return [array(H1) as NDArray, array(H2) as NDArray];
}

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  const kk = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < kk; i += 1) result = (result * (n - i)) / (i + 1);
  return Math.round(result);
}

function bankersRound(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  const scaled = value * factor;
  const floored = Math.floor(scaled);
  const diff = scaled - floored;
  let rounded: number;
  if (diff < 0.5) rounded = floored;
  else if (diff > 0.5) rounded = floored + 1;
  else rounded = floored % 2 === 0 ? floored : floored + 1;
  return rounded / factor;
}

export function proportions_along_bezier_curve_for_point(
  point: number[] | NDArray,
  control_points: number[][] | NDArray,
  round_to = 1e-6,
): NDArray {
  // NOTE: Python requires np.shape(point) == np.shape(c_p) for every
  // control point; a flat 3-point vs 3-vectors is the normal case.
  const pointCoords = rawRows(point)[0] ?? [];
  const cpRows = rawRows(control_points);
  for (const c_p of cpRows) {
    if (c_p.length !== pointCoords.length) {
      const error = new Error(
        `Point ${JSON.stringify(point)} and Control Points ${JSON.stringify(control_points)} have different shapes.`,
      );
      error.name = "ValueError";
      throw error;
    }
  }
  const n = cpRows.length - 1;
  const dim = pointCoords.length;

  const decimals = Math.trunc(Math.log10(1 / round_to));
  const perDimRoots: number[][] = [];
  for (let d = 0; d < dim; d += 1) {
    const coord = pointCoords[d] ?? 0;
    const control_coords = cpRows.map((row) => row[d] ?? 0);
    const terms: number[] = [];
    for (let term_power = n; term_power >= 0; term_power -= 1) {
      const outercoeff = choose(n, term_power);
      let inner = 0;
      let sign = 1;
      for (let subterm_num = term_power; subterm_num >= 0; subterm_num -= 1) {
        const innercoeff = choose(term_power, subterm_num) * sign;
        let subterm = innercoeff * (control_coords[subterm_num] ?? 0);
        if (term_power === 0) subterm -= coord;
        inner += subterm;
        sign *= -1;
      }
      terms.push(outercoeff * inner);
    }
    if (terms.every((term) => term === 0)) continue;
    // NOTE: numpy's Polynomial(terms[::-1]).roots() solves the same zeros as
    // numpy.roots(terms) (highest-degree-first); feed terms directly.
    const found = polyRoots(terms) as NDArray;
    const raw = found.toArray() as unknown[];
    if (raw.length === 0) {
      perDimRoots.push([]);
      continue;
    }
    const rounded = raw.map((item) => {
      if (typeof item === "number") return { re: bankersRound(item, decimals), im: 0 };
      const c = item as { re: number; im: number };
      return { re: bankersRound(c.re ?? 0, decimals), im: bankersRound(c.im ?? 0, decimals) };
    });
    perDimRoots.push(rounded.filter((c) => c.im === 0).map((c) => c.re));
  }

  if (perDimRoots.length === 0) {
    const error = new TypeError("reduce() of empty iterable with no initial value");
    throw error;
  }
  // NOTE: Python reduce(np.intersect1d, roots): sorted unique intersection.
  let common = [...new Set(perDimRoots[0] ?? [])].sort((a, b) => a - b);
  for (const list of perDimRoots.slice(1)) {
    const other = new Set(list);
    common = [...new Set(common.filter((v) => other.has(v)))].sort((a, b) => a - b);
  }
  return array(common.filter((r) => 0 <= r && r <= 1)) as NDArray;
}

export function get_smooth_open_cubic_bezier_handle_points(
  anchors: number[][] | NDArray,
): [NDArray, NDArray] {
  // NOTE: unlike rowsOf/pointRows, the anchor width is taken as-is from the
  // input (Python uses A.shape[1]); rows are NOT padded to 3 components.
  const A = rawRows(anchors);
  const N = A.length - 1;
  const dim = A[0]?.length ?? 0;

  const len_memo = CP_OPEN_MEMO.length;
  let cp: number[];
  if (len_memo < N - 1) {
    cp = new Array(N - 1).fill(0);
    for (let i = 0; i < len_memo; i += 1) cp[i] = CP_OPEN_MEMO[i] ?? 0;
    for (let i = len_memo; i < N - 1; i += 1) {
      cp[i] = 1 / (4 - (cp[i - 1] ?? 0));
    }
    CP_OPEN_MEMO = cp;
  } else {
    cp = CP_OPEN_MEMO.slice(0, N - 1);
  }
  // Mirror Python's negative indexing (cp[-1] etc.); on an empty memo this
  // raises IndexError exactly like the Python version does for N <= 1.
  const cpAt = (index: number): number => {
    const resolved = index < 0 ? cp.length + index : index;
    if (resolved < 0 || resolved >= cp.length) {
      const error = new Error("list index out of range");
      error.name = "IndexError";
      throw error;
    }
    return cp[resolved] ?? 0;
  };

  const Dp: number[][] = Array.from({ length: N }, () =>
    new Array(dim).fill(0),
  );
  const add = (a: number[], b: number[]): number[] =>
    a.map((v, c) => v + (b[c] ?? 0));
  const scale = (a: number[], s: number): number[] => a.map((v) => v * s);
  const sub = (a: number[], b: number[]): number[] =>
    a.map((v, c) => v - (b[c] ?? 0));
  Dp[0] = add(scale(A[0] ?? [], 0.5), A[1] ?? []);
  for (let i = 1; i < N - 1; i += 1) {
    const aux = add(scale(A[i] ?? [], 4), scale(A[i + 1] ?? [], 2));
    Dp[i] = scale(sub(aux, Dp[i - 1] ?? []), cpAt(i));
  }
  Dp[N - 1] = scale(
    sub(
      add(scale(A[N - 1] ?? [], 8), A[N] ?? []),
      scale(Dp[N - 2] ?? [], 2),
    ),
    1 / (7 - 2 * cpAt(N - 2)),
  );

  // Backward substitution. H1 aliases Dp, exactly as in Python.
  const H1 = Dp;
  for (let i = N - 2; i >= 0; i -= 1) {
    H1[i] = sub(Dp[i] ?? [], scale(H1[i + 1] ?? [], cpAt(i)));
  }

  const H2: number[][] = Array.from({ length: N }, () =>
    new Array(dim).fill(0),
  );
  for (let i = 0; i < N - 1; i += 1) {
    H2[i] = sub(scale(A[i + 1] ?? [], 2), H1[i + 1] ?? []);
  }
  H2[N - 1] = scale(add(A[N] ?? [], H1[N - 1] ?? []), 0.5);

  return [array(H1) as NDArray, array(H2) as NDArray];
}

export function is_closed(points: number[][] | NDArray): boolean {
  const all = rawRows(points);
  const start = all[0] ?? [];
  const end = all[all.length - 1] ?? [];
  const rtol = 1e-5;
  const atol = 1e-8;
  const tolerance = start.map((s) => atol + rtol * s);
  if (Math.abs((end[0] ?? 0) - (start[0] ?? 0)) > (tolerance[0] ?? 0)) {
    return false;
  }
  if (Math.abs((end[1] ?? 0) - (start[1] ?? 0)) > (tolerance[1] ?? 0)) {
    return false;
  }
  return Math.abs((end[2] ?? 0) - (start[2] ?? 0)) <= (tolerance[2] ?? 0);
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

export function subdivide_bezier(
  points: number[][] | NDArray,
  n_divisions: number,
): number[][] {
  const rows = rawRows(points as number[] | number[][] | NDArray);
  if (n_divisions === 1) return rows.map((row) => [...row]);
  const count = rows.length;
  const dim = rows[0]?.length ?? 3;
  if (count <= 4) {
    const matrix = pointRows(_get_subdivision_matrix(count, n_divisions), count);
    return matrix.map((matrixRow) =>
      Array.from({ length: dim }, (_, column) =>
        matrixRow.reduce((sum, value, index) => sum + value * (rows[index]?.[column] ?? 0), 0),
      ),
    );
  }
  // Fallback for higher-degree curves: successive splitting.
  const beziers: number[][][] = Array.from({ length: n_divisions }, () => rows.map((row) => [...row]));
  const last = beziers[n_divisions - 1];
  if (last) rows.forEach((row, index) => { last[index] = [...row]; });
  for (let curve_num = n_divisions - 1; curve_num > 0; curve_num -= 1) {
    const curr = beziers[curve_num];
    const prev = beziers[curve_num - 1];
    if (!curr || !prev) continue;
    prev[0] = [...(curr[0] ?? [])];
    const a = (n_divisions - curve_num) / (n_divisions - curve_num + 1);
    for (let i = 1; i < count; i += 1) {
      for (let k = 0; k < count - i; k += 1) {
        const left = curr[k] ?? [];
        const right = curr[k + 1] ?? [];
        curr[k] = left.map((value, column) => value + a * ((right[column] ?? 0) - value));
      }
      prev[i] = [...(curr[0] ?? [])];
    }
  }
  return beziers.flat();
}

export function partial_bezier_points(
  points: number[][] | NDArray,
  a: number,
  b: number,
): number[][] {
  const source = rawRows(points as number[] | number[][] | NDArray);
  const dim = source[0]?.length ?? 3;
  // Border cases (mirror Python: fill with the end/start point).
  if (a === 1) return source.map(() => [...(source[source.length - 1] ?? new Array(dim).fill(0))]);
  if (b === 0) return source.map(() => [...(source[0] ?? new Array(dim).fill(0))]);
  const degree = source.length - 1;

  const applyMatrix = (matrix: number[][]): number[][] =>
    matrix.map((matrixRow) =>
      Array.from({ length: dim }, (_, column) =>
        matrixRow.reduce((sum, value, index) => sum + value * (source[index]?.[column] ?? 0), 0),
      ),
    );

  if (degree === 3) {
    const ma = 1 - a;
    const mb = 1 - b;
    const a2 = a * a;
    const b2 = b * b;
    const ma2 = ma * ma;
    const mb2 = mb * mb;
    const a3 = a2 * a;
    const b3 = b2 * b;
    const ma3 = ma2 * ma;
    const mb3 = mb2 * mb;
    return applyMatrix([
      [ma3, 3 * ma2 * a, 3 * ma * a2, a3],
      [ma2 * mb, 2 * ma * a * mb + ma2 * b, a2 * mb + 2 * ma * a * b, a2 * b],
      [ma * mb2, a * mb2 + 2 * ma * mb * b, 2 * a * mb * b + ma * b2, a * b2],
      [mb3, 3 * mb2 * b, 3 * mb * b2, b3],
    ]);
  }

  if (degree === 2) {
    const ma = 1 - a;
    const mb = 1 - b;
    return applyMatrix([
      [ma * ma, 2 * a * ma, a * a],
      [ma * mb, a * mb + ma * b, a * b],
      [mb * mb, 2 * b * mb, b * b],
    ]);
  }

  if (degree === 1) {
    const start = source[0] ?? new Array(dim).fill(0);
    const end = source[1] ?? [...start];
    const direction = end.map((value, column) => value - (start[column] ?? 0));
    return [
      start.map((value, column) => value + a * (direction[column] ?? 0)),
      start.map((value, column) => value + b * (direction[column] ?? 0)),
    ];
  }

  if (degree === 0) return source.map((row) => [...row]);

  // Fallback for higher-degree curves: de Casteljau splitting at a then at u.
  const arr = source.map((row) => [...row]);
  const count = arr.length;
  if (a !== 0) {
    for (let i = 1; i < count; i += 1) {
      for (let k = 0; k < count - i; k += 1) {
        const left = arr[k] ?? [];
        const right = arr[k + 1] ?? [];
        arr[k] = left.map((value, column) => value + a * ((right[column] ?? 0) - value));
      }
    }
  }
  if (b !== 1) {
    const mu = (1 - b) / (1 - a);
    for (let i = 1; i < count; i += 1) {
      // NOTE: NumPy evaluates the whole RHS from pre-update values, so
      // snapshot the row first (ascending order would otherwise reuse the
      // already-updated predecessor).
      const snapshot = arr.map((row) => [...row]);
      for (let k = i; k < count; k += 1) {
        const current = snapshot[k] ?? [];
        const prev = snapshot[k - 1] ?? [];
        arr[k] = current.map((value, column) => value + mu * ((prev[column] ?? 0) - value));
      }
    }
  }
  return arr;
}

export function bezier_remap(
  bezier_tuples: number[][][] | NDArray,
  new_number_of_curves: number,
): number[][][] {
  const tuples: number[][][] = Array.isArray(bezier_tuples)
    ? (bezier_tuples as number[][][]).map((curve) => curve.map((point) => [...point]))
    : (() => {
      const source = bezier_tuples as NDArray;
      const shape = source.shape;
      const nested = source.toArray() as unknown;
      if (shape.length === 3) {
        return (nested as number[][][]).map((curve) => curve.map((point) => [...point]));
      }
      const flat = rawRows(source as unknown as number[] | number[][] | NDArray);
      const nppc = 4;
      const result: number[][][] = [];
      for (let i = 0; i + nppc <= flat.length; i += nppc) result.push(flat.slice(i, i + nppc));
      return result;
    })();
  const current_number_of_curves = tuples.length;
  const nppc = tuples[0]?.length ?? 4;
  const dim = tuples[0]?.[0]?.length ?? 3;
  const repeat_indices: number[] = Array.from(
    { length: new_number_of_curves },
    (_, i) => Math.floor((i * current_number_of_curves) / new_number_of_curves),
  );
  const split_factors: number[] = new Array(current_number_of_curves).fill(0);
  for (const index of repeat_indices) split_factors[index] = (split_factors[index] ?? 0) + 1;
  const result: number[][][] = [];
  tuples.forEach((curve, curveIndex) => {
    const sf = split_factors[curveIndex] ?? 0;
    const subdivided = subdivide_bezier(curve, sf);
    for (let i = 0; i < sf; i += 1) {
      result.push(subdivided.slice(i * nppc, i * nppc + nppc).map((point) => point.slice(0, dim)));
    }
  });
  return result;
}

export function point_lies_on_bezier(
  point: number[] | NDArray,
  control_points: number[][] | NDArray,
  round_to = 1e-6,
): boolean {
  const roots = proportions_along_bezier_curve_for_point(point, control_points, round_to);
  return (roots.toArray() as unknown[]).length > 0;
}
