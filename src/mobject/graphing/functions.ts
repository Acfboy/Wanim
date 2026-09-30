import { array, type NDArray } from "numpy-ts";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { config } from "../../_config/utils.js";
import type { ParsableManimColor } from "../../utils/color/core.js";
import { arange, isNDArray, numeric, pointAt, pointRows } from "../../utils/arrays.js";
import { LinearBase, type _ScaleBase } from "./scale.js";

export type ParametricFunctionT = (t: number) => number[] | readonly number[] | NDArray;

export interface ParametricFunctionOptions extends VMobjectOptions {
  t_range?: [number, number] | [number, number, number];
  scaling?: _ScaleBase;
  dt?: number;
  discontinuities?: number[] | null;
  use_smoothing?: boolean;
  use_vectorized?: boolean;
  /** Internal: flat point-row evaluator set by CoordinateSystem.plot so that
   * generate_points can avoid the per-sample NDArray round-trip. Not part of
   * the public API. */
  _point_flat?: (t: number) => number[];
  /** Internal: batch variant of `_point_flat` (one call per sampled t-list). */
  _points_flat?: (ts: readonly number[]) => number[][];
}

export class ParametricFunction extends VMobject {
  function: (t: number) => NDArray;
  scaling: _ScaleBase;
  dt: number;
  discontinuities: number[] | null;
  use_smoothing: boolean;
  use_vectorized: boolean;
  t_min: number;
  t_max: number;
  t_step: number;
  // NOTE: set dynamically by CoordinateSystem.plot / plot_parametric_curve /
  // plot_polar_graph (mirrors Python's `graph.underlying_function = ...`);
  // read by input_to_graph_point/coords and plot_antiderivative_graph.
  underlying_function?: (t: number) => unknown;
  // Internal flat point evaluator (see ParametricFunctionOptions._point_flat).
  _point_flat?: (t: number) => number[];
  _points_flat?: (ts: readonly number[]) => number[][];

  constructor(function_: ParametricFunctionT, options: ParametricFunctionOptions = {}) {
    const {
      t_range = [0, 1] as [number, number],
      scaling = new LinearBase(),
      dt = 1e-8,
      discontinuities = null,
      use_smoothing = true,
      use_vectorized = false,
      _point_flat = undefined,
      _points_flat = undefined,
      ...rest
    } = options;
    super(rest);

    function internal_parametric_function(t: number): NDArray {
      // Mirrors Python `np.asarray(function(t))`: an NDArray result is passed
      // through without a copy, everything else is wrapped.
      const value = function_(t);
      return isNDArray(value) ? (value as NDArray) : (array(value) as NDArray);
    }

    this.function = internal_parametric_function;
    this._point_flat = _point_flat;
    this._points_flat = _points_flat;
    const full_range: [number, number, number] = t_range.length === 2
      ? [t_range[0] ?? 0, t_range[1] ?? 1, 0.01]
      : [t_range[0] ?? 0, t_range[1] ?? 1, t_range[2] ?? 0.01];
    this.scaling = scaling;
    this.dt = dt;
    this.discontinuities = discontinuities === null ? null : [...discontinuities];
    this.use_smoothing = use_smoothing;
    this.use_vectorized = use_vectorized;
    [this.t_min, this.t_max, this.t_step] = full_range;
    // NOTE: Python sets these fields before super().__init__() so that the
    // generate_points() call inside the base constructor already sees them.
    // TS requires super() first; the base-constructor call is guarded as a
    // no-op below and points are (re)generated here instead.
    this.generate_points();
  }

  get_function(): (t: number) => NDArray {
    return this.function;
  }

  get_point_from_function(t: number): NDArray {
    return this.function(t);
  }

  override generate_points(): this {
    // Guard: the base Mobject constructor invokes generate_points() before
    // this class assigns t_min/t_max/t_step (TS requires super() first,
    // whereas Python sets them before super().__init__()). Treat that early
    // call as a no-op; the constructor regenerates afterwards.
    if (this.t_min === undefined || this.t_max === undefined || this.t_step === undefined) return this;
    let boundary_times: number[];
    if (this.discontinuities !== null) {
      const discs = this.discontinuities.filter((t) => this.t_min <= t && t <= this.t_max);
      boundary_times = [
        this.t_min,
        this.t_max,
        ...discs.map((t) => t - this.dt),
        ...discs.map((t) => t + this.dt),
      ].sort((a, b) => a - b);
    } else {
      boundary_times = [this.t_min, this.t_max];
    }
    for (let i = 0; i + 1 < boundary_times.length; i += 2) {
      const t1 = boundary_times[i] ?? 0;
      const t2 = boundary_times[i + 1] ?? 0;
      // Mirrors `np.array([*scaling.function(np.arange(t1, t2, t_step)),
      // scaling.function(t2)])`. NOTE: the scaled samples form a flat 1-D
      // list (not one row per sample), so they are flattened element-wise
      // rather than read via pointRows.
      let t_values: number[];
      if (this.scaling instanceof LinearBase) {
        // LinearBase.function is a plain multiply; skip the numpy-ts array
        // round-trip for the per-sample t values (hot for graph plots).
        const factor = this.scaling.scale_factor;
        const base = arange(t1, t2, this.t_step);
        t_values = base.map((value) => factor * value);
        t_values.push(factor * t2);
      } else {
        const scaledOut = this.scaling.function(arange(t1, t2, this.t_step));
        const scaledRaw: unknown = isNDArray(scaledOut) ? scaledOut.toArray() : scaledOut;
        const scaled: number[] = [];
        const collect = (v: unknown): void => {
          if (Array.isArray(v)) for (const item of v) collect(item);
          else scaled.push(numeric(v));
        };
        collect(scaledRaw);
        const endValue = this.scaling.function(t2);
        t_values = [
          ...scaled,
          typeof endValue === "number" ? endValue : (pointRows(endValue, 1)[0]?.[0] ?? 0),
        ];
      }
      let point_list: number[][];
      // Hot path: extract the first point row without materialising every row
      // of the returned NDArray (`pointRows` calls `toArray()` on all data).
      const getPoint = (t: number): number[] => {
        if (this._point_flat !== undefined) return this._point_flat(t);
        const value = this.get_point_from_function(t);
        if (isNDArray(value)) return pointAt(value as NDArray, 0, this.dim);
        return pointRows(value, this.dim)[0] ?? new Array(this.dim).fill(0);
      };
      if (this.use_vectorized) {
        // Python calls self.function(t_range) once with the whole array;
        // accept a vectorized result, else fall back to per-sample calls.
        const out = (this.function as unknown as (t: unknown) => unknown)(array(t_values));
        if (typeof out !== "object" || out === null || !("shape" in (out as Record<string, unknown>))) {
          point_list = t_values.map(getPoint);
        } else {
          const asArray = (out as NDArray).toArray() as unknown;
          point_list = Array.isArray(asArray) && Array.isArray(asArray[0])
            ? (asArray as unknown[][]).map((row) => (row as unknown[]).map(Number))
            : t_values.map(getPoint);
        }
      } else if (this._points_flat !== undefined) {
        // Internal batch path: evaluate all samples with the axis constants
        // read once (no NDArray per sample).
        point_list = this._points_flat(t_values);
      } else {
        // NOTE: Python calls self.function(t) here; for FunctionGraph that
        // is the wrapped parametric form, so the polymorphic
        // get_point_from_function (which yields the parametric form for
        // FunctionGraph) is used instead.
        point_list = t_values.map(getPoint);
      }
      this.start_new_path(point_list[0]);
      this.add_points_as_corners(point_list.slice(1));
    }
    if (this.use_smoothing) {
      // NOTE: VMobject.make_smooth is translated in a later batch; smooth here once available.
      // (The Mobject proxy throws AttributeError on missing members, so `in` must guard the access.)
      if ("make_smooth" in this) {
        (this as unknown as { make_smooth: () => void }).make_smooth();
      }
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }
}

export type FunctionGraphFunction = (x: number) => number | number[] | readonly number[] | NDArray;

export interface FunctionGraphOptions extends VMobjectOptions {
  discontinuities?: number[] | null;
  use_smoothing?: boolean;
  use_vectorized?: boolean;
  scaling?: _ScaleBase;
  dt?: number;
}

export class FunctionGraph extends ParametricFunction {
  x_range: [number, number] | [number, number, number];
  parametric_function: (t: number) => NDArray;

  constructor(
    function_: FunctionGraphFunction,
    x_range: [number, number] | [number, number, number] | null = null,
    color: ParsableManimColor = "PURE_YELLOW",
    options: FunctionGraphOptions = {},
  ) {
    const resolved_range: [number, number] | [number, number, number] = x_range ?? [
      -config.frame_x_radius,
      config.frame_x_radius,
    ];
    // NOTE: mirrors Python `lambda t: np.array([t, function(t), 0])`.
    // Python assigns this wrapped form to ParametricFunction (which stores
    // it as self.function) and ALSO keeps it as self.parametric_function;
    // self.function is therefore the wrapped form (NOT the raw user
    // function) after construction. The TS port does the same: super()
    // stores the wrapped form as self.function; only parametric_function
    // is (re)assigned below.
    const parametric_function = (t: number): NDArray =>
      array([t, function_(t) as number, 0]) as NDArray;
    // NOTE: Python stores the raw user function as self.function and passes
    // the wrapped parametric form to ParametricFunction. The TS
    // ParametricFunction constructor takes an options object (not positional
    // t_range/color), so forward them as options.
    const {
      discontinuities = null,
      use_smoothing = true,
      use_vectorized = false,
      scaling = new LinearBase(),
      dt = 1e-8,
      ...rest
    } = options;
    super(parametric_function, {
      t_range: resolved_range,
      scaling,
      dt,
      discontinuities,
      use_smoothing,
      use_vectorized,
      color,
      ...rest,
    });
    this.x_range = resolved_range;
    this.parametric_function = parametric_function;
    // Fast path: FunctionGraph is always a 2D graph `[t, f(t), 0]`, so feed
    // generate_points a flat scalar sampler instead of materialising a fresh
    // 3-element NDArray (and unwrapping it) for every sample. Numerically
    // identical to `np.array([t, function(t), 0])` for scalar functions.
    const flatPoint = (t: number): number[] => [t, numeric(function_(t)), 0];
    this._point_flat = flatPoint;
    this._points_flat = (ts: readonly number[]): number[][] => ts.map(flatPoint);
    // NOTE: super() already stored the wrapped parametric form as
    // self.function (exactly like Python, where ParametricFunction.__init__
    // overwrites the raw assignment); get_function() therefore returns the
    // wrapped form and get_point_from_function the parametric point.
    // Regenerate now that x_range/parametric_function are assigned (see the
    // super-first note on ParametricFunction above).
    this.generate_points();
  }

  override get_function(): (t: number) => NDArray {
    return this.function;
  }

  override get_point_from_function(x: number): NDArray {
    return this.parametric_function(x);
  }

  override generate_points(): this {
    // Guard: the ParametricFunction constructor invokes generate_points()
    // before this class assigns parametric_function/x_range (TS requires
    // super() first, whereas Python sets them before super().__init__()).
    // Treat that early call as a no-op; the constructor regenerates below.
    // NOTE: `in` is used instead of a plain undefined comparison because
    // Mobject instances are Proxies that throw AttributeError on undeclared
    // attribute reads during construction.
    if (!("parametric_function" in this) || this.parametric_function === undefined) return this;
    if (!("x_range" in this) || this.x_range === undefined) return this;
    return super.generate_points();
  }
}

export type ImplicitFunctionT = (x: number, y: number) => number;

export interface ImplicitFunctionOptions extends VMobjectOptions {
  x_range?: number[] | null;
  y_range?: number[] | null;
  min_depth?: number;
  max_quads?: number;
  use_smoothing?: boolean;
}

export type IsolineComputer = (fn: (p: [number, number]) => number, pmin: [number, number], pmax: [number, number], min_depth: number, max_quads: number) => number[][][];

// ---------------------------------------------------------------------------
// Isoline extraction
//
// Faithful TypeScript port of the `isosurfaces` package's 2D `plot_isoline`
// (https://github.com/jared-hughes/isosurfaces), which Python manim imports in
// `ImplicitFunction.generate_points`. Reproducing it exactly (adaptive
// quadtree + simplicial triangulation + curve tracing) keeps the emitted curve
// structurally identical to the reference; the pip package must not be added
// as a dependency.
// ---------------------------------------------------------------------------

interface ValuedPoint {
  pos: number[];
  val: number;
}

/** Internal 2-argument isoline function. The public `computeIsoline` accepts
 * the Python-style `fn([x, y])`; the implementation calls `fn(x, y)` so the hot
 * recursion does not allocate a fresh `[x, y]` array on every evaluation. */
type IsoFn = (x: number, y: number) => number;

interface IsoTriangle {
  vertices: ValuedPoint[];
  next: IsoTriangle | null;
  next_bisect_point: ValuedPoint | null;
  prev: IsoTriangle | null;
  visited: boolean;
}

interface IsoCell {
  vertices: ValuedPoint[];
  depth: number;
  children: IsoCell[];
  parent: IsoCell | null;
  child_direction: number;
}

function valuedPoint(pos: number[], val: number): ValuedPoint {
  // NOTE: `pos` is always a freshly-built 2-element array (or a read-only
  // position shared with an existing point), so store it without copying.
  return { pos, val };
}

function pointValue(pos: number[], fn: IsoFn): ValuedPoint {
  return { pos, val: fn(pos[0]!, pos[1]!) };
}

function midpointPoint(p1: ValuedPoint, p2: ValuedPoint, fn: IsoFn): ValuedPoint {
  const x = (p1.pos[0]! + p2.pos[0]!) / 2;
  const y = (p1.pos[1]! + p2.pos[1]!) / 2;
  return valuedPoint([x, y], fn(x, y));
}

function intersectZero(p1: ValuedPoint, p2: ValuedPoint, fn: IsoFn): ValuedPoint {
  const denom = p1.val - p2.val;
  const k1 = -p2.val / denom;
  const k2 = p1.val / denom;
  const x = k1 * p1.pos[0]! + k2 * p2.pos[0]!;
  const y = k1 * p1.pos[1]! + k2 * p2.pos[1]!;
  return valuedPoint([x, y], fn(x, y));
}

function binarySearchZero(
  p1: ValuedPoint,
  p2: ValuedPoint,
  fn: IsoFn,
  tol: number[],
): [ValuedPoint, boolean] {
  if (Math.abs(p2.pos[0]! - p1.pos[0]!) < tol[0]! && Math.abs(p2.pos[1]! - p1.pos[1]!) < tol[1]!) {
    const pt = intersectZero(p1, p2, fn);
    const isZero =
      pt.val === 0 ||
      (Math.sign(pt.val - p1.val) === Math.sign(p2.val - pt.val) && pt.val < 1e200);
    return [pt, isZero];
  }
  const mid = midpointPoint(p1, p2, fn);
  if (mid.val === 0) return [mid, true];
  if ((mid.val > 0) === (p1.val > 0)) return binarySearchZero(mid, p2, fn, tol);
  return binarySearchZero(p1, mid, fn, tol);
}

function verticesFromExtremes(
  pmin: number[],
  pmax: number[],
  fn: IsoFn,
): ValuedPoint[] {
  const w = [pmax[0]! - pmin[0]!, pmax[1]! - pmin[1]!];
  const result: ValuedPoint[] = [];
  for (let i = 0; i < 4; i += 1) {
    const pos = [
      pmin[0]! + ((i >> 0) & 1) * w[0]!,
      pmin[1]! + ((i >> 1) & 1) * w[1]!,
    ];
    result.push(pointValue(pos, fn));
  }
  return result;
}

function computeChildren(cell: IsoCell, fn: IsoFn): void {
  const first = cell.vertices[0]!;
  const last = cell.vertices[3]!;
  for (let i = 0; i < cell.vertices.length; i += 1) {
    const vertex = cell.vertices[i]!;
    const pmin = [(first.pos[0]! + vertex.pos[0]!) / 2, (first.pos[1]! + vertex.pos[1]!) / 2];
    const pmax = [(last.pos[0]! + vertex.pos[0]!) / 2, (last.pos[1]! + vertex.pos[1]!) / 2];
    cell.children.push({
      vertices: verticesFromExtremes(pmin, pmax, fn),
      depth: cell.depth + 1,
      children: [],
      parent: cell,
      child_direction: i,
    });
  }
}

function shouldDescendDeepCell(cell: IsoCell, tol: number[]): boolean {
  const w = [
    cell.vertices[3]!.pos[0]! - cell.vertices[0]!.pos[0]!,
    cell.vertices[3]!.pos[1]! - cell.vertices[0]!.pos[1]!,
  ];
  if (w[0]! < 10 * tol[0]! && w[1]! < 10 * tol[1]!) return false;
  if (cell.vertices.every((v) => Number.isNaN(v.val))) return false;
  if (cell.vertices.some((v) => Number.isNaN(v.val))) return true;
  const sign0 = Math.sign(cell.vertices[0]!.val);
  return cell.vertices.slice(1).some((v) => Math.sign(v.val) !== sign0);
}

function buildTree(
  fn: IsoFn,
  pmin: number[],
  pmax: number[],
  minDepth: number,
  maxCells: number,
  tol: number[],
): IsoCell {
  const branchingFactor = 4;
  const cap = Math.max(branchingFactor ** minDepth, maxCells);
  const root: IsoCell = {
    vertices: verticesFromExtremes(pmin, pmax, fn),
    depth: 0,
    children: [],
    parent: null,
    child_direction: 0,
  };
  const queue: IsoCell[] = [root];
  let head = 0;
  let leafCount = 1;
  while (head < queue.length && leafCount < cap) {
    // Index-based dequeue: `Array.shift()` is O(n) and this loop is quadratic
    // with it (Python uses a `deque.popleft()`). FIFO order is unchanged.
    const cell = queue[head++]!;
    if (cell.depth < minDepth || shouldDescendDeepCell(cell, tol)) {
      computeChildren(cell, fn);
      queue.push(...cell.children);
      leafCount += branchingFactor - 1;
    }
  }
  return root;
}

function fourTriangles(
  a: ValuedPoint,
  b: ValuedPoint,
  c: ValuedPoint,
  d: ValuedPoint,
  center: ValuedPoint,
): IsoTriangle[] {
  const make = (v0: ValuedPoint, v1: ValuedPoint): IsoTriangle => ({
    vertices: [v0, v1, center],
    next: null,
    next_bisect_point: null,
    prev: null,
    visited: false,
  });
  return [make(a, b), make(b, c), make(c, d), make(d, a)];
}

class Triangulator {
  triangles: IsoTriangle[] = [];
  hangingNext = new Map<string, IsoTriangle>();
  private root: IsoCell;
  private fn: IsoFn;
  private tol: number[];

  constructor(root: IsoCell, fn: IsoFn, tol: number[]) {
    this.root = root;
    this.fn = fn;
    this.tol = tol;
  }

  triangulate(): IsoTriangle[] {
    this.triangulateInside(this.root);
    return this.triangles;
  }

  private triangulateInside(quad: IsoCell): void {
    if (quad.children.length === 0) return;
    for (const child of quad.children) this.triangulateInside(child);
    this.crossingRow(quad.children[0]!, quad.children[1]!);
    this.crossingRow(quad.children[2]!, quad.children[3]!);
    this.crossingCol(quad.children[0]!, quad.children[2]!);
    this.crossingCol(quad.children[1]!, quad.children[3]!);
  }

  private crossingRow(a: IsoCell, b: IsoCell): void {
    if (a.children.length > 0 && b.children.length > 0) {
      this.crossingRow(a.children[1]!, b.children[0]!);
      this.crossingRow(a.children[3]!, b.children[2]!);
    } else if (a.children.length > 0) {
      this.crossingRow(a.children[1]!, b);
      this.crossingRow(a.children[3]!, b);
    } else if (b.children.length > 0) {
      this.crossingRow(a, b.children[0]!);
      this.crossingRow(a, b.children[2]!);
    } else {
      const faceDualA = this.faceDual(a);
      const faceDualB = this.faceDual(b);
      let triangles: IsoTriangle[];
      if (a.depth < b.depth) {
        const edgeDual = this.edgeDual(b.vertices[2]!, b.vertices[0]!);
        triangles = fourTriangles(b.vertices[2]!, faceDualB, b.vertices[0]!, faceDualA, edgeDual);
      } else {
        const edgeDual = this.edgeDual(a.vertices[3]!, a.vertices[1]!);
        triangles = fourTriangles(a.vertices[3]!, faceDualB, a.vertices[1]!, faceDualA, edgeDual);
      }
      this.addFourTriangles(triangles);
    }
  }

  private crossingCol(a: IsoCell, b: IsoCell): void {
    if (a.children.length > 0 && b.children.length > 0) {
      this.crossingCol(a.children[2]!, b.children[0]!);
      this.crossingCol(a.children[3]!, b.children[1]!);
    } else if (a.children.length > 0) {
      this.crossingCol(a.children[2]!, b);
      this.crossingCol(a.children[3]!, b);
    } else if (b.children.length > 0) {
      this.crossingCol(a, b.children[0]!);
      this.crossingCol(a, b.children[1]!);
    } else {
      const faceDualA = this.faceDual(a);
      const faceDualB = this.faceDual(b);
      let triangles: IsoTriangle[];
      if (a.depth < b.depth) {
        const edgeDual = this.edgeDual(b.vertices[0]!, b.vertices[1]!);
        triangles = fourTriangles(b.vertices[0]!, faceDualB, b.vertices[1]!, faceDualA, edgeDual);
      } else {
        const edgeDual = this.edgeDual(a.vertices[2]!, a.vertices[3]!);
        triangles = fourTriangles(a.vertices[2]!, faceDualB, a.vertices[3]!, faceDualA, edgeDual);
      }
      this.addFourTriangles(triangles);
    }
  }

  private addFourTriangles(triangles: IsoTriangle[]): void {
    for (let i = 0; i < 4; i += 1) {
      this.nextSandwich(triangles[i]!, triangles[(i + 1) % 4]!, triangles[(i + 2) % 4]!);
    }
    this.triangles.push(...triangles);
  }

  private setNext(tri1: IsoTriangle, tri2: IsoTriangle, vpos: ValuedPoint, vneg: ValuedPoint): void {
    if (!(vpos.val > 0 && vneg.val <= 0)) return;
    const [intersection, isZero] = binarySearchZero(vpos, vneg, this.fn, this.tol);
    if (!isZero) return;
    tri1.next_bisect_point = intersection;
    tri1.next = tri2;
    tri2.prev = tri1;
  }

  private nextSandwich(a: IsoTriangle, b: IsoTriangle, c: IsoTriangle): void {
    const center = b.vertices[2]!;
    const x = b.vertices[0]!;
    const y = b.vertices[1]!;

    if (center.val > 0 && y.val <= 0) this.setNext(b, c, center, y);
    if (x.val > 0 && center.val <= 0) this.setNext(b, a, x, center);

    const id = `${x.pos[0]! + y.pos[0]!},${x.pos[1]! + y.pos[1]!}`;
    if (y.val > 0 && x.val <= 0) {
      const hanging = this.hangingNext.get(id);
      if (hanging !== undefined) {
        this.setNext(b, hanging, y, x);
        this.hangingNext.delete(id);
      } else {
        this.hangingNext.set(id, b);
      }
    } else if (y.val <= 0 && x.val > 0) {
      const hanging = this.hangingNext.get(id);
      if (hanging !== undefined) {
        this.setNext(hanging, b, x, y);
        this.hangingNext.delete(id);
      } else {
        this.hangingNext.set(id, b);
      }
    }
  }

  private edgeDual(p1: ValuedPoint, p2: ValuedPoint): ValuedPoint {
    if ((p1.val > 0) !== (p2.val > 0)) return midpointPoint(p1, p2, this.fn);
    const dt = 0.01;
    const df1 = this.fn(
      p1.pos[0]! * (1 - dt) + p2.pos[0]! * dt,
      p1.pos[1]! * (1 - dt) + p2.pos[1]! * dt,
    );
    const df2 = this.fn(
      p1.pos[0]! * dt + p2.pos[0]! * (1 - dt),
      p1.pos[1]! * dt + p2.pos[1]! * (1 - dt),
    );
    if ((df1 > 0) === (df2 > 0)) return midpointPoint(p1, p2, this.fn);
    return intersectZero(valuedPoint(p1.pos, df1), valuedPoint(p2.pos, df2), this.fn);
  }

  private faceDual(quad: IsoCell): ValuedPoint {
    return midpointPoint(quad.vertices[0]!, quad.vertices[3]!, this.fn);
  }
}

function traceCurves(triangles: IsoTriangle[]): number[][][] {
  const curves: number[][][] = [];
  for (const triangle of triangles) {
    if (triangle.visited || triangle.next === null) continue;
    const active: ValuedPoint[] = [];
    let cursor: IsoTriangle | null = triangle;
    const start = triangle;
    let closedLoop = false;
    while (cursor.prev !== null) {
      cursor = cursor.prev;
      if (cursor === start) {
        closedLoop = true;
        break;
      }
    }
    while (cursor !== null && !cursor.visited) {
      if (cursor.next_bisect_point !== null) active.push(cursor.next_bisect_point);
      cursor.visited = true;
      cursor = cursor.next;
    }
    if (closedLoop && active.length > 0) active.push(active[0]!);
    curves.push(active.map((v) => [v.pos[0]!, v.pos[1]!]));
  }
  return curves;
}

/** Shared isoline core taking a 2-argument function (see {@link IsoFn}). */
function computeIsolineInternal(
  fn: IsoFn,
  pmin: [number, number],
  pmax: [number, number],
  minDepth: number,
  maxQuads: number,
): number[][][] {
  const tol = [(pmax[0] - pmin[0]) / 1000, (pmax[1] - pmin[1]) / 1000];
  const tree = buildTree(fn, pmin, pmax, minDepth, maxQuads, tol);
  const triangles = new Triangulator(tree, fn, tol).triangulate();
  return traceCurves(triangles);
}

export function computeIsoline(
  fn: (p: [number, number]) => number,
  pmin: [number, number],
  pmax: [number, number],
  minDepth = 5,
  maxQuads = 10000,
): number[][][] {
  return computeIsolineInternal((x, y) => fn([x, y]), pmin, pmax, minDepth, maxQuads);
}

let isolineComputer: IsolineComputer | null = computeIsoline;

export function setIsolineComputer(computer: IsolineComputer | null): void {
  isolineComputer = computer;
}

export class ImplicitFunction extends VMobject {
  function: ImplicitFunctionT;
  min_depth: number;
  max_quads: number;
  use_smoothing: boolean;
  x_range: number[];
  y_range: number[];

  constructor(func: ImplicitFunctionT, options: ImplicitFunctionOptions = {}) {
    const {
      x_range = null,
      y_range = null,
      min_depth = 5,
      max_quads = 1500,
      use_smoothing = true,
      ...rest
    } = options;
    super(rest);
    this.function = func;
    this.min_depth = min_depth;
    this.max_quads = max_quads;
    this.use_smoothing = use_smoothing;
    this.x_range = x_range !== null ? [...x_range] : [
      -config.frame_width / 2,
      config.frame_width / 2,
    ];
    this.y_range = y_range !== null ? [...y_range] : [
      -config.frame_height / 2,
      config.frame_height / 2,
    ];
    // The base Mobject constructor already called generate_points(), but at
    // that time the ranges were not yet assigned (TS requires super() first,
    // whereas Python sets them before super().__init__()). Regenerate now
    // that the fields are available so points are populated at construction.
    // A programmatic `setIsolineComputer(null)` keeps the old lazy behavior
    // (empty until generate_points is called explicitly).
    if (isolineComputer !== null) this.generate_points();
  }

  generate_points(): this {
    // Guard: the base Mobject constructor invokes generate_points() before
    // this class assigns x_range/y_range (TS requires super() first, whereas
    // Python sets them before super().__init__()). Treat that early call as
    // a no-op; the constructor regenerates afterwards.
    if (this.x_range === undefined || this.y_range === undefined) return this;
    if (isolineComputer === null) {
      throw new Error("TODO: ImplicitFunction.generate_points needs the `isosurfaces` plot_isoline routine, not ported yet");
    }
    const pmin: [number, number] = [this.x_range[0] ?? 0, this.y_range[0] ?? 0];
    const pmax: [number, number] = [this.x_range[1] ?? 0, this.y_range[1] ?? 0];
    // The default computer is the ported isoline routine itself; call it with
    // the user function's native `(x, y)` signature so its hot recursion does
    // not allocate an argument array per evaluation. An injected computer
    // (tests) still receives the Python-style single-array wrapper.
    const curves = (
      isolineComputer === computeIsoline
        ? computeIsolineInternal(
            (x, y) => this.function(x, y),
            pmin,
            pmax,
            this.min_depth,
            this.max_quads,
          )
        : isolineComputer(
            (u) => this.function(u[0] ?? 0, u[1] ?? 0),
            pmin,
            pmax,
            this.min_depth,
            this.max_quads,
          )
    )
      .filter((curve) => curve.length > 0)
      .map((curve) => curve.map((point) => [point[0] ?? 0, point[1] ?? 0, 0]));
    for (const curve of curves) {
      if (curve.length === 0) continue;
      this.start_new_path(curve[0]);
      this.add_points_as_corners(curve.slice(1));
    }
    if (this.use_smoothing) {
      // NOTE: VMobject.make_smooth is translated in a later batch; smooth here once available.
      // (The Mobject proxy throws AttributeError on missing members, so `in` must guard the access.)
      if ("make_smooth" in this) {
        (this as unknown as { make_smooth: () => void }).make_smooth();
      }
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }
}
