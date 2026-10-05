import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import { RendererType } from "../../constants.js";
import { isNDArray, numeric, pointRows } from "../../utils/arrays.js";
import { VMobject } from "../types/vectorized_mobject.js";

/** Verbs emitted by {@link SkiaPath}, mirroring `pathops.PathVerb`. */
export enum PathVerb {
  MOVE = "move",
  LINE = "line",
  QUAD = "quad",
  CUBIC = "cubic",
  CLOSE = "close",
}

export interface SkiaPathCommand {
  verb: PathVerb;
  /** Control/end points as [x, y] pairs (empty for CLOSE). */
  points: number[][];
}

/**
 * Minimal local stand-in for `pathops.Path` (Skia).
 *
 * The Python implementation uses the `pathops` (Skia) library for boolean
 * path operations; there is no equivalent npm dependency available here (and
 * third-party packages other than numpy-ts are out of scope), so this shim
 * records the same move/quad/cubic/close command sequence the original
 * builds. It additionally implements the `[verb, points]` iteration protocol
 * used by `_convert_skia_path_to_vmobject`, and `getPen()` returning itself
 * (mirroring `SkiaPath.getPen()` used by the boolean combinators).
 */
export class SkiaPath {
  commands: SkiaPathCommand[] = [];

  moveTo(x: number, y: number): this {
    this.commands.push({ verb: PathVerb.MOVE, points: [[x, y]] });
    return this;
  }

  lineTo(x: number, y: number): this {
    this.commands.push({ verb: PathVerb.LINE, points: [[x, y]] });
    return this;
  }

  quadTo(cpx: number, cpy: number, x: number, y: number): this {
    this.commands.push({ verb: PathVerb.QUAD, points: [[cpx, cpy], [x, y]] });
    return this;
  }

  cubicTo(
    cp1x: number, cp1y: number,
    cp2x: number, cp2y: number,
    x: number, y: number,
  ): this {
    this.commands.push({
      verb: PathVerb.CUBIC,
      points: [[cp1x, cp1y], [cp2x, cp2y], [x, y]],
    });
    return this;
  }

  close(): this {
    this.commands.push({ verb: PathVerb.CLOSE, points: [] });
    return this;
  }

  getPen(): this {
    return this;
  }

  *[Symbol.iterator](): Generator<[PathVerb, number[][]], void, void> {
    for (const command of this.commands) {
      yield [command.verb, command.points];
    }
  }
}

export class _BooleanOps extends VMobject {
  _convert_2d_to_3d_array(points: Iterable<number[] | readonly number[] | NDArray>, z_dim = 0): NDArray {
    const list_of_points: number[][] = [...points].map((point) =>
      isNDArray(point)
        ? ((point as NDArray).toArray() as unknown[]).map(numeric)
        : [...(point as readonly number[])].map(numeric),
    );
    for (let index = 0; index < list_of_points.length; index += 1) {
      const point = list_of_points[index] as number[];
      if (point.length === 2) list_of_points[index] = [...point, z_dim];
    }
    return array(list_of_points) as NDArray;
  }

  _convert_vmobject_to_skia_path(vmobject: VMobject): SkiaPath {
    const path = new SkiaPath();

    // Mirrors `np.all(np.isfinite(vmobject.points))`.
    const flat = (vmobject.points.toArray() as unknown[]).flat(Infinity) as unknown[];
    const all_finite = flat.every((value) => Number.isFinite(Number(value)));
    let points: NDArray;
    if (all_finite) {
      points = vmobject.points;
    } else {
      points = array([[0, 0, 0]]) as NDArray; // point invalid?
    }

    if (pointRows(points, vmobject.dim).length === 0) {
      // what? No points so return empty path
      return path;
    }

    // In OpenGL it's quadratic bezier curves while on Cairo it's cubic...
    if (config.renderer === RendererType.OPENGL) {
      const subpaths = vmobject.get_subpaths_from_points(points);
      for (const subpath of subpaths) {
        // NOTE: Python calls `get_bezier_tuples_from_points` (quadratic
        // triples, OpenGL renderer only). wanim's VMobject stores cubic
        // tuples; chunk by n_points_per_curve so quadratic-mode mobjects
        // (nppc == 3) yield exact quads, mirroring `for _p0, p1, p2 in quads`.
        const sub_rows = pointRows(subpath, vmobject.dim);
        const nppc = vmobject.n_points_per_curve;
        const start = sub_rows[0] ?? [0, 0, 0];
        path.moveTo(start[0] ?? 0, start[1] ?? 0);
        for (let index = 0; index + nppc <= sub_rows.length; index += nppc) {
          const tuple = sub_rows.slice(index, index + nppc);
          const p1 = tuple[1] ?? tuple[0] ?? [0, 0, 0];
          const p2 = tuple[nppc - 1] ?? tuple[0] ?? [0, 0, 0];
          path.quadTo(p1[0] ?? 0, p1[1] ?? 0, p2[0] ?? 0, p2[1] ?? 0);
        }
        if (vmobject.consider_points_equals(sub_rows[0], sub_rows[sub_rows.length - 1])) {
          path.close();
        }
      }
    } else if (config.renderer === RendererType.CAIRO) {
      const subpaths = vmobject.gen_subpaths_from_points_2d(points);
      for (const subpath of subpaths) {
        const quads = vmobject.gen_cubic_bezier_tuples_from_points(subpath);
        const sub_rows = pointRows(subpath, vmobject.dim);
        const start = sub_rows[0] ?? [0, 0, 0];
        path.moveTo(start[0] ?? 0, start[1] ?? 0);
        for (const quad of quads) {
          const p1 = quad[1] ?? [0, 0, 0];
          const p2 = quad[2] ?? [0, 0, 0];
          const p3 = quad[3] ?? [0, 0, 0];
          path.cubicTo(
            p1[0] ?? 0, p1[1] ?? 0,
            p2[0] ?? 0, p2[1] ?? 0,
            p3[0] ?? 0, p3[1] ?? 0,
          );
        }
        if (
          vmobject.consider_points_equals_2d(
            sub_rows[0] ?? [0, 0, 0],
            sub_rows[sub_rows.length - 1] ?? [0, 0, 0],
          )
        ) {
          path.close();
        }
      }
    }

    return path;
  }

  _convert_skia_path_to_vmobject(path: SkiaPath): VMobject {
    const vmobject = this as unknown as VMobject;
    let current_path_start: number[] = [0, 0, 0];
    for (const [path_verb, points] of path) {
      if (path_verb === PathVerb.MOVE) {
        const parts = pointRows(this._convert_2d_to_3d_array(points), this.dim);
        for (const part of parts) {
          current_path_start = [...part];
          vmobject.start_new_path(part);
        }
      } else if (path_verb === PathVerb.CUBIC) {
        const [n1 = [0, 0, 0], n2 = [0, 0, 0], n3 = [0, 0, 0]] = pointRows(
          this._convert_2d_to_3d_array(points),
          this.dim,
        );
        vmobject.add_cubic_bezier_curve_to(n1, n2, n3);
      } else if (path_verb === PathVerb.LINE) {
        const parts = pointRows(this._convert_2d_to_3d_array(points), this.dim);
        vmobject.add_line_to(parts[0] ?? [0, 0, 0]);
      } else if (path_verb === PathVerb.CLOSE) {
        vmobject.add_line_to(current_path_start);
      } else if (path_verb === PathVerb.QUAD) {
        const [n1 = [0, 0, 0], n2 = [0, 0, 0]] = pointRows(
          this._convert_2d_to_3d_array(points),
          this.dim,
        );
        vmobject.add_quadratic_bezier_curve_to(n1, n2);
      } else {
        throw new Error(`Unsupported: ${path_verb}`);
      }
    }
    return vmobject;
  }
}

/**
 * Local polygon boolean engine backing the `pathops` combinators below.
 *
 * The Python implementation delegates to the Skia `pathops` library, which
 * has no equivalent among the allowed dependencies (only numpy-ts may be
 * used). Instead this file implements exact boolean ops on flattened
 * polygons:
 *
 * 1. Each `SkiaPath` subpath is flattened to a polygon ring (quadratic and
 *    cubic segments are subdivided; straight segments are kept as-is).
 * 2. Every subject edge is intersected against every clip edge and both
 *    rings are split at the crossing points.
 * 3. Each split edge is classified as inside/outside the other operand
 *    (even-odd rule; points exactly on the boundary count as inside).
 * 4. Edges are selected per operator and stitched back into rings:
 *    - union: subject-outside + clip-outside (forward),
 *    - intersection: subject-inside + clip-inside (forward),
 *    - difference (subject minus clip): subject-outside (forward) +
 *      clip-inside (reversed),
 *    - xor: computed as union minus intersection.
 *
 * Input rings are normalized to counter-clockwise orientation first, which
 * guarantees that at every transversal crossing exactly one arrival edge
 * and one departure edge meet, so greedy stitching reproduces the correct
 * boundary. Result rings are re-oriented by nesting depth (solids CCW,
 * holes CW) so the canvas `nonzero` fill renders holes correctly.
 *
 * Known limitations (degenerate inputs): tangent touches, overlapping
 * collinear edges, self-intersecting subpaths within a single operand, and
 * open (unclosed) subpaths, which are dropped. These either fall back to a
 * containment heuristic or keep the input outline.
 */

type XYRing = number[][];
type BooleanOp = "union" | "intersection" | "difference";

const BOOL_EPS = 1e-9;
const CUBIC_FLATTEN_SEGMENTS = 12;
const QUAD_FLATTEN_SEGMENTS = 10;

function quadPointAt(
  p0: number[], p1: number[], p2: number[], t: number,
): [number, number] {
  const mt = 1 - t;
  return [
    (mt * mt * (p0[0] ?? 0) + 2 * mt * t * (p1[0] ?? 0) + t * t * (p2[0] ?? 0)),
    (mt * mt * (p0[1] ?? 0) + 2 * mt * t * (p1[1] ?? 0) + t * t * (p2[1] ?? 0)),
  ];
}

function cubicPointAt(
  p0: number[], p1: number[], p2: number[], p3: number[], t: number,
): [number, number] {
  const mt = 1 - t;
  const a = mt * mt * mt;
  const b = 3 * mt * mt * t;
  const c = 3 * mt * t * t;
  const d = t * t * t;
  return [
    (a * (p0[0] ?? 0) + b * (p1[0] ?? 0) + c * (p2[0] ?? 0) + d * (p3[0] ?? 0)),
    (a * (p0[1] ?? 0) + b * (p1[1] ?? 0) + c * (p2[1] ?? 0) + d * (p3[1] ?? 0)),
  ];
}

function signedRingArea(ring: XYRing): number {
  let total = 0;
  for (let index = 0; index < ring.length; index += 1) {
    const current = ring[index] ?? [0, 0];
    const next = ring[(index + 1) % ring.length] ?? [0, 0];
    total += (current[0] ?? 0) * (next[1] ?? 0) - (next[0] ?? 0) * (current[1] ?? 0);
  }
  return total / 2;
}

function pointsNear(ax: number, ay: number, bx: number, by: number): boolean {
  return Math.abs(ax - bx) <= BOOL_EPS && Math.abs(ay - by) <= BOOL_EPS;
}

/** Remove consecutive duplicates (and a duplicated closing point). */
function dedupeRing(points: number[][]): XYRing {
  const cleaned: XYRing = [];
  for (const point of points) {
    const x = point[0] ?? 0;
    const y = point[1] ?? 0;
    const last = cleaned[cleaned.length - 1];
    if (last === undefined || !pointsNear(last[0] ?? 0, last[1] ?? 0, x, y)) {
      cleaned.push([x, y]);
    }
  }
  while (
    cleaned.length > 1 &&
    pointsNear(
      cleaned[0]?.[0] ?? 0, cleaned[0]?.[1] ?? 0,
      cleaned[cleaned.length - 1]?.[0] ?? 0, cleaned[cleaned.length - 1]?.[1] ?? 0,
    )
  ) {
    cleaned.pop();
  }
  return cleaned;
}

/** Flatten one `SkiaPath` into closed polygon rings (open subpaths dropped). */
function flattenSkiaPathToRings(path: SkiaPath): XYRing[] {
  const rings: XYRing[] = [];
  let current: number[][] | null = null;
  const flush = (closed: boolean): void => {
    if (current !== null) {
      const first = current[0];
      const last = current[current.length - 1];
      const coincidentClosed =
        first !== undefined && last !== undefined &&
        pointsNear(first[0] ?? 0, first[1] ?? 0, last[0] ?? 0, last[1] ?? 0);
      if (closed || coincidentClosed) {
        const ring = dedupeRing(current);
        if (ring.length >= 3 && Math.abs(signedRingArea(ring)) > 1e-12) {
          rings.push(ring);
        }
      }
    }
    current = null;
  };
  for (const [verb, points] of path) {
    if (verb === PathVerb.MOVE) {
      if (current !== null) flush(false);
      const start = points[0] ?? [0, 0];
      current = [[start[0] ?? 0, start[1] ?? 0]];
    } else if (current === null) {
      continue;
    } else if (verb === PathVerb.LINE) {
      const end = points[0] ?? [0, 0];
      current.push([end[0] ?? 0, end[1] ?? 0]);
    } else if (verb === PathVerb.QUAD) {
      const p0 = current[current.length - 1] ?? [0, 0];
      const p1 = points[0] ?? [0, 0];
      const p2 = points[1] ?? [0, 0];
      for (let index = 1; index <= QUAD_FLATTEN_SEGMENTS; index += 1) {
        current.push(quadPointAt(p0, p1, p2, index / QUAD_FLATTEN_SEGMENTS) as number[]);
      }
    } else if (verb === PathVerb.CUBIC) {
      const p0 = current[current.length - 1] ?? [0, 0];
      const p1 = points[0] ?? [0, 0];
      const p2 = points[1] ?? [0, 0];
      const p3 = points[2] ?? [0, 0];
      for (let index = 1; index <= CUBIC_FLATTEN_SEGMENTS; index += 1) {
        current.push(cubicPointAt(p0, p1, p2, p3, index / CUBIC_FLATTEN_SEGMENTS) as number[]);
      }
    } else if (verb === PathVerb.CLOSE) {
      flush(true);
    }
  }
  if (current !== null) flush(false);
  return rings;
}

function pointOnSegment(
  px: number, py: number, ax: number, ay: number, bx: number, by: number,
): boolean {
  const cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
  if (Math.abs(cross) > BOOL_EPS) return false;
  const dot = (px - ax) * (px - bx) + (py - ay) * (py - by);
  return dot <= BOOL_EPS;
}

/** Even-odd point-in-ring test; boundary points count as inside. */
function pointInRing(px: number, py: number, ring: XYRing): boolean {
  return pointRingStatus(px, py, ring) !== "out";
}

type PointStatus = "in" | "on" | "out";

function pointRingStatus(px: number, py: number, ring: XYRing): PointStatus {
  for (let index = 0; index < ring.length; index += 1) {
    const current = ring[index] ?? [0, 0];
    const next = ring[(index + 1) % ring.length] ?? [0, 0];
    if (pointOnSegment(px, py, current[0] ?? 0, current[1] ?? 0, next[0] ?? 0, next[1] ?? 0)) {
      return "on";
    }
  }
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const current = ring[index] ?? [0, 0];
    const prev = ring[previous] ?? [0, 0];
    const xi = current[0] ?? 0;
    const yi = current[1] ?? 0;
    const xj = prev[0] ?? 0;
    const yj = prev[1] ?? 0;
    if ((yi > py) !== (yj > py)) {
      const intersectX = xj + ((py - yj) / (yi - yj)) * (xi - xj);
      if (px < intersectX) inside = !inside;
    }
  }
  return inside ? "in" : "out";
}

/** Even-odd containment across all rings of one operand. */
function pointInRings(px: number, py: number, rings: XYRing[]): boolean {
  let status: PointStatus = "out";
  for (const ring of rings) {
    const ringStatus = pointRingStatus(px, py, ring);
    if (ringStatus === "on") return true;
    if (ringStatus === "in") status = status === "in" ? "out" : "in";
  }
  return status === "in";
}

/** Three-state classification: strictly inside, on the boundary, or outside. */
function classifyPointInRings(px: number, py: number, rings: XYRing[]): PointStatus {
  let status: PointStatus = "out";
  for (const ring of rings) {
    const ringStatus = pointRingStatus(px, py, ring);
    if (ringStatus === "on") return "on";
    if (ringStatus === "in") status = status === "in" ? "out" : "in";
  }
  return status;
}

interface SegCrossing {
  x: number;
  y: number;
  t: number;
  u: number;
}

/** Proper segment-segment crossing; parallel/collinear pairs give null. */
function segmentCrossing(
  ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number,
): SegCrossing | null {
  const rX = bx - ax;
  const rY = by - ay;
  const sX = dx - cx;
  const sY = dy - cy;
  const denom = rX * sY - rY * sX;
  if (Math.abs(denom) < 1e-14) return null;
  const qX = cx - ax;
  const qY = cy - ay;
  const t = (qX * sY - qY * sX) / denom;
  const u = (qX * rY - qY * rX) / denom;
  if (t < -BOOL_EPS || t > 1 + BOOL_EPS || u < -BOOL_EPS || u > 1 + BOOL_EPS) {
    return null;
  }
  const tc = Math.min(1, Math.max(0, t));
  const uc = Math.min(1, Math.max(0, u));
  // Endpoint-only touches do not split: containment handles them.
  if (tc <= BOOL_EPS || tc >= 1 - BOOL_EPS || uc <= BOOL_EPS || uc >= 1 - BOOL_EPS) {
    return null;
  }
  return { x: ax + tc * rX, y: ay + tc * rY, t: tc, u: uc };
}

interface DirectedEdge {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  /** Which operand this edge came from (0 = subject, 1 = clip). */
  src: 0 | 1;
}

interface CutPoint {
  t: number;
  x: number;
  y: number;
}

function splitOperandEdges(
  rings: XYRing[],
  cuts: Map<number, Map<number, CutPoint[]>>,
  other: XYRing[],
  src: 0 | 1,
): { edge: DirectedEdge; status: PointStatus }[] {
  const result: { edge: DirectedEdge; status: PointStatus }[] = [];
  for (let ringIndex = 0; ringIndex < rings.length; ringIndex += 1) {
    const ring = rings[ringIndex] ?? [];
    const ringCuts = cuts.get(ringIndex);
    for (let edgeIndex = 0; edgeIndex < ring.length; edgeIndex += 1) {
      const start = ring[edgeIndex] ?? [0, 0];
      const end = ring[(edgeIndex + 1) % ring.length] ?? [0, 0];
      const edgeCuts = [...(ringCuts?.get(edgeIndex) ?? [])]
        .sort((left, right) => left.t - right.t);
      const spine: number[][] = [[start[0] ?? 0, start[1] ?? 0]];
      for (const cut of edgeCuts) {
        const last = spine[spine.length - 1] ?? [0, 0];
        if (!pointsNear(last[0] ?? 0, last[1] ?? 0, cut.x, cut.y)) {
          spine.push([cut.x, cut.y]);
        }
      }
      const last = spine[spine.length - 1] ?? [0, 0];
      if (!pointsNear(last[0] ?? 0, last[1] ?? 0, end[0] ?? 0, end[1] ?? 0)) {
        spine.push([end[0] ?? 0, end[1] ?? 0]);
      }
      for (let index = 0; index + 1 < spine.length; index += 1) {
        const from = spine[index] ?? [0, 0];
        const to = spine[index + 1] ?? [0, 0];
        if (pointsNear(from[0] ?? 0, from[1] ?? 0, to[0] ?? 0, to[1] ?? 0)) continue;
        const midX = ((from[0] ?? 0) + (to[0] ?? 0)) / 2;
        const midY = ((from[1] ?? 0) + (to[1] ?? 0)) / 2;
        result.push({
          edge: { ax: from[0] ?? 0, ay: from[1] ?? 0, bx: to[0] ?? 0, by: to[1] ?? 0, src },
          status: classifyPointInRings(midX, midY, other),
        });
      }
    }
  }
  return result;
}

function pointKey(x: number, y: number): string {
  return `${Math.round(x / BOOL_EPS)}:${Math.round(y / BOOL_EPS)}`;
}

/** Stitch directed edges into closed rings by matching end points to starts. */
function stitchDirectedEdges(edges: DirectedEdge[]): XYRing[] {
  const starts = new Map<string, number[]>();
  edges.forEach((edge, index) => {
    const key = pointKey(edge.ax, edge.ay);
    const bucket = starts.get(key);
    if (bucket === undefined) starts.set(key, [index]);
    else bucket.push(index);
  });
  const used = new Array<boolean>(edges.length).fill(false);
  const rings: XYRing[] = [];
  for (let seed = 0; seed < edges.length; seed += 1) {
    if (used[seed]) continue;
    used[seed] = true;
    const first = edges[seed] as DirectedEdge;
    let currentSrc: 0 | 1 = first.src;
    const ring: XYRing = [[first.ax, first.ay], [first.bx, first.by]];
    let steps = edges.length;
    while (steps-- > 0) {
      const tip = ring[ring.length - 1] ?? [0, 0];
      if (pointsNear(tip[0] ?? 0, tip[1] ?? 0, ring[0]?.[0] ?? 0, ring[0]?.[1] ?? 0)) break;
      const candidates = starts.get(pointKey(tip[0] ?? 0, tip[1] ?? 0)) ?? [];
      // Prefer continuing on the same operand: at a genuine transversal
      // crossing each operand contributes exactly one selected edge, so no
      // same-source continuation exists and the boundary must switch
      // operands; at a mere touch (e.g. a hole vertex lying exactly on an
      // outer ring) each ring passes through, and staying on the same
      // operand keeps both rings intact.
      const sameSource = candidates.find((index) => !used[index] && edges[index]?.src === currentSrc);
      const next = sameSource ?? candidates.find((index) => !used[index]);
      if (next === undefined) break;
      used[next] = true;
      const edge = edges[next] as DirectedEdge;
      currentSrc = edge.src;
      ring.push([edge.bx, edge.by]);
    }
    const cleaned = dedupeRing(ring);
    if (cleaned.length >= 3 && Math.abs(signedRingArea(cleaned)) > 1e-12) {
      rings.push(cleaned);
    }
  }
  return rings;
}

/**
 * Orient result rings by nesting depth so `nonzero` fills render holes:
 * even-depth (solid) rings go CCW, odd-depth (hole) rings go CW.
 */
function orientRingsByDepth(rings: XYRing[]): XYRing[] {
  return rings.map((ring) => {
    let depth = 0;
    const probe = ring[0] ?? [0, 0];
    for (const other of rings) {
      if (other === ring || other.length === 0) continue;
      if (pointInRing(probe[0] ?? 0, probe[1] ?? 0, other)) depth += 1;
    }
    const area = signedRingArea(ring);
    if (depth % 2 === 0 && area < 0) return [...ring].reverse();
    if (depth % 2 !== 0 && area > 0) return [...ring].reverse();
    return ring;
  });
}

/** Boolean op over two ring lists. Inputs are normalized to CCW first. */
function booleanRings(subject: XYRing[], clip: XYRing[], op: BooleanOp): XYRing[] {
  const subj: XYRing[] = subject.map((ring) =>
    signedRingArea(ring) < 0 ? [...ring].reverse() : [...ring],
  );
  const clp: XYRing[] = clip.map((ring) =>
    signedRingArea(ring) < 0 ? [...ring].reverse() : [...ring],
  );
  const subjCuts = new Map<number, Map<number, CutPoint[]>>();
  const clipCuts = new Map<number, Map<number, CutPoint[]>>();
  const recordCut = (
    store: Map<number, Map<number, CutPoint[]>>,
    ringIndex: number, edgeIndex: number, t: number, x: number, y: number,
  ): void => {
    let byRing = store.get(ringIndex);
    if (byRing === undefined) {
      byRing = new Map<number, CutPoint[]>();
      store.set(ringIndex, byRing);
    }
    const list = byRing.get(edgeIndex);
    const cut = { t, x, y };
    if (list === undefined) byRing.set(edgeIndex, [cut]);
    else list.push(cut);
  };
  subj.forEach((ring, ringIndex) => {
    ring.forEach((point, edgeIndex) => {
      const next = ring[(edgeIndex + 1) % ring.length] ?? [0, 0];
      clp.forEach((clipRing, clipRingIndex) => {
        clipRing.forEach((clipPoint, clipEdgeIndex) => {
          const clipNext = clipRing[(clipEdgeIndex + 1) % clipRing.length] ?? [0, 0];
          const crossing = segmentCrossing(
            point[0] ?? 0, point[1] ?? 0, next[0] ?? 0, next[1] ?? 0,
            clipPoint[0] ?? 0, clipPoint[1] ?? 0, clipNext[0] ?? 0, clipNext[1] ?? 0,
          );
          if (crossing !== null) {
            recordCut(subjCuts, ringIndex, edgeIndex, crossing.t, crossing.x, crossing.y);
            recordCut(clipCuts, clipRingIndex, clipEdgeIndex, crossing.u, crossing.x, crossing.y);
          }
        });
      });
    });
  });
  // T-junction pass: a vertex of one operand lying strictly inside an
  // edge of the other (e.g. axis-aligned squares touching edge to edge)
  // never produces a proper crossing above, so split the host edge there.
  const recordTouch = (
    store: Map<number, Map<number, CutPoint[]>>,
    ringIndex: number, edgeIndex: number, t: number, x: number, y: number,
  ): void => {
    if (t <= BOOL_EPS || t >= 1 - BOOL_EPS) return;
    recordCut(store, ringIndex, edgeIndex, t, x, y);
  };
  const touchHostEdge = (
    px: number, py: number,
    hostRings: XYRing[],
    store: Map<number, Map<number, CutPoint[]>>,
  ): void => {
    hostRings.forEach((hostRing, hostRingIndex) => {
      hostRing.forEach((hostPoint, hostEdgeIndex) => {
        const hostNext = hostRing[(hostEdgeIndex + 1) % hostRing.length] ?? [0, 0];
        const ax = hostPoint[0] ?? 0;
        const ay = hostPoint[1] ?? 0;
        const bx = hostNext[0] ?? 0;
        const by = hostNext[1] ?? 0;
        if (pointsNear(px, py, ax, ay) || pointsNear(px, py, bx, by)) return;
        if (!pointOnSegment(px, py, ax, ay, bx, by)) return;
        const lengthSquared = (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
        if (lengthSquared <= 0) return;
        const t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / lengthSquared;
        recordTouch(store, hostRingIndex, hostEdgeIndex, t, px, py);
      });
    });
  };
  subj.forEach((ring) => {
    ring.forEach((point) => {
      touchHostEdge(point[0] ?? 0, point[1] ?? 0, clp, clipCuts);
    });
  });
  clp.forEach((ring) => {
    ring.forEach((point) => {
      touchHostEdge(point[0] ?? 0, point[1] ?? 0, subj, subjCuts);
    });
  });
  const subjEdges = splitOperandEdges(subj, subjCuts, clp, 0);
  const clipEdges = splitOperandEdges(clp, clipCuts, subj, 1);
  const selected: DirectedEdge[] = [];
  const keepSubject = (status: PointStatus): boolean =>
    op === "union" ? status !== "in" : op === "intersection" ? status !== "out" : status === "out";
  const keepClip = (status: PointStatus): boolean =>
    op === "union" ? status !== "in" : op === "intersection" ? status !== "out" : status === "in";
  for (const { edge, status } of subjEdges) {
    if (keepSubject(status)) selected.push(edge);
  }
  for (const { edge, status } of clipEdges) {
    if (!keepClip(status)) continue;
    if (op === "difference") {
      selected.push({ ax: edge.bx, ay: edge.by, bx: edge.ax, by: edge.ay, src: edge.src });
    } else {
      selected.push(edge);
    }
  }
  // Collinear overlaps (e.g. identical inputs) select the shared boundary
  // from both operands. Same-direction copies collapse to one; an edge
  // selected in opposite directions is interior to the union (or a
  // zero-area intersection), so both copies are dropped.
  const keyOf = (edge: DirectedEdge): string =>
    `${pointKey(edge.ax, edge.ay)}>${pointKey(edge.bx, edge.by)}`;
  const reverseKeyOf = (edge: DirectedEdge): string =>
    `${pointKey(edge.bx, edge.by)}>${pointKey(edge.ax, edge.ay)}`;
  const keyCounts = new Map<string, number>();
  for (const edge of selected) {
    keyCounts.set(keyOf(edge), (keyCounts.get(keyOf(edge)) ?? 0) + 1);
  }
  const deduped: DirectedEdge[] = [];
  const seen = new Set<string>();
  for (const edge of selected) {
    if ((keyCounts.get(reverseKeyOf(edge)) ?? 0) > 0) continue;
    const key = keyOf(edge);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(edge);
  }
  if (deduped.length === 0) return [];
  return orientRingsByDepth(stitchDirectedEdges(deduped));
}

function resolvePen(outpen: { getPen(): SkiaPath } | SkiaPath): SkiaPath {
  const candidate = outpen as SkiaPath;
  if (candidate !== null && typeof candidate === "object" && typeof candidate.moveTo === "function") {
    return candidate;
  }
  return (outpen as { getPen(): SkiaPath }).getPen();
}

function emitRingsToPen(rings: XYRing[], pen: SkiaPath): void {
  for (const ring of rings) {
    if (ring.length === 0) continue;
    const start = ring[0] ?? [0, 0];
    pen.moveTo(start[0] ?? 0, start[1] ?? 0);
    for (let index = 1; index < ring.length; index += 1) {
      const point = ring[index] ?? [0, 0];
      pen.lineTo(point[0] ?? 0, point[1] ?? 0);
    }
    pen.close();
  }
}

function pathops_union(paths: SkiaPath[], outpen: { getPen(): SkiaPath } | SkiaPath): void {
  const pen = resolvePen(outpen);
  if (paths.length === 0) return;
  let acc = flattenSkiaPathToRings(paths[0] as SkiaPath);
  for (let index = 1; index < paths.length; index += 1) {
    acc = booleanRings(acc, flattenSkiaPathToRings(paths[index] as SkiaPath), "union");
  }
  emitRingsToPen(acc, pen);
}

function pathops_difference(
  paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath } | SkiaPath,
): void {
  const pen = resolvePen(outpen);
  let acc: XYRing[] = [];
  for (const path of paths) acc.push(...flattenSkiaPathToRings(path));
  let clipRings: XYRing[] = [];
  for (const path of clip) clipRings.push(...flattenSkiaPathToRings(path));
  emitRingsToPen(booleanRings(acc, clipRings, "difference"), pen);
}

function pathops_intersection(
  paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath } | SkiaPath,
): void {
  const pen = resolvePen(outpen);
  let acc: XYRing[] = [];
  for (const path of paths) acc.push(...flattenSkiaPathToRings(path));
  let clipRings: XYRing[] = [];
  for (const path of clip) clipRings.push(...flattenSkiaPathToRings(path));
  emitRingsToPen(booleanRings(acc, clipRings, "intersection"), pen);
}

function pathops_xor(
  paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath } | SkiaPath,
): void {
  const pen = resolvePen(outpen);
  let subject: XYRing[] = [];
  for (const path of paths) subject.push(...flattenSkiaPathToRings(path));
  let clipRings: XYRing[] = [];
  for (const path of clip) clipRings.push(...flattenSkiaPathToRings(path));
  const union = booleanRings(subject, clipRings, "union");
  const inter = booleanRings(subject, clipRings, "intersection");
  emitRingsToPen(booleanRings(union, inter, "difference"), pen);
}

export class Union extends _BooleanOps {
  constructor(...args: unknown[]) {
    let vmobjects: VMobject[];
    let options: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as Record<string, unknown>;
      vmobjects = args.slice(0, -1) as VMobject[];
    } else {
      vmobjects = args as VMobject[];
    }
    if (vmobjects.length < 2) {
      const error = new Error("At least 2 mobjects needed for Union.");
      error.name = "ValueError";
      throw error;
    }
    super(options);
    const paths = vmobjects.map((vmobject) => this._convert_vmobject_to_skia_path(vmobject));
    const outpen = new SkiaPath();
    pathops_union(paths, outpen.getPen());
    this._convert_skia_path_to_vmobject(outpen);
  }
}

export class Difference extends _BooleanOps {
  constructor(subject: VMobject, clip: VMobject, options: Record<string, unknown> = {}) {
    super(options);
    const outpen = new SkiaPath();
    pathops_difference(
      [this._convert_vmobject_to_skia_path(subject)],
      [this._convert_vmobject_to_skia_path(clip)],
      outpen.getPen(),
    );
    this._convert_skia_path_to_vmobject(outpen);
  }
}

export class Intersection extends _BooleanOps {
  constructor(...args: unknown[]) {
    let vmobjects: VMobject[];
    let options: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as Record<string, unknown>;
      vmobjects = args.slice(0, -1) as VMobject[];
    } else {
      vmobjects = args as VMobject[];
    }
    if (vmobjects.length < 2) {
      const error = new Error("At least 2 mobjects needed for Intersection.");
      error.name = "ValueError";
      throw error;
    }
    super(options);
    const outpen = new SkiaPath();
    pathops_intersection(
      [this._convert_vmobject_to_skia_path(vmobjects[0] as VMobject)],
      [this._convert_vmobject_to_skia_path(vmobjects[1] as VMobject)],
      outpen.getPen(),
    );
    let current = outpen;
    for (let index = 2; index < vmobjects.length; index += 1) {
      const next = new SkiaPath();
      pathops_intersection(
        [current],
        [this._convert_vmobject_to_skia_path(vmobjects[index] as VMobject)],
        next.getPen(),
      );
      current = next;
    }
    this._convert_skia_path_to_vmobject(current);
  }
}

export class Exclusion extends _BooleanOps {
  constructor(subject: VMobject, clip: VMobject, options: Record<string, unknown> = {}) {
    super(options);
    const outpen = new SkiaPath();
    pathops_xor(
      [this._convert_vmobject_to_skia_path(subject)],
      [this._convert_vmobject_to_skia_path(clip)],
      outpen.getPen(),
    );
    this._convert_skia_path_to_vmobject(outpen);
  }
}
