/**
 * WebGL2 backend for `ThreeDCamera` frames.
 *
 * The Canvas2D backend projects every point, resolves every face's shading and
 * builds one path per face on the CPU, then asks the rasterizer to fill and
 * stroke each face separately — that caps a 24x48 sphere at a few FPS.  This
 * backend keeps the cairo semantics but moves the work to the GPU:
 *
 *  - Geometry is uploaded in world space once per shape change.  The vertex
 *    shader applies exactly `ThreeDCamera.project_points` (rotation matrix,
 *    linear/exponential perspective, zoom) and then the same world->device
 *    transform as `setCanvasTransform`.
 *  - `modified_rgbas` shading is evaluated in the vertex shader from the
 *    per-face corner positions/normals, so moving the light source or toggling
 *    `should_apply_shading` is a uniform change, not a geometry rebuild.
 *  - Fills are triangulated on the CPU (ear clipping) and stroked with
 *    screen-space miter joins computed in the vertex shader, so stroke width
 *    matches `cairo_line_width_multiple` behaviour at any depth.
 *  - Colours come from the VMobject's raw rgba rows and reproduce
 *    `set_cairo_context_color`: a flat colour for uniform rows, otherwise a
 *    two-stop linear gradient between the VMobject's gradient endpoints,
 *    evaluated per fragment exactly like `cairo.LinearGradient`.
 *  - Painter's order matches cairo: the sorted display list is replayed in
 *    order (per mobject: background stroke, fill, foreground stroke), with the
 *    triangles of every mobject appended to one element buffer and drawn in a
 *    single `drawElements` call (WebGL preserves primitive order).
 *  - Animated mobjects whose new points are an affine image of the uploaded
 *    ones keep their geometry: the affine transform is fitted on the CPU and
 *    applied by the shader, so `Rotate`/`shift`/`scale` animations do not
 *    re-upload a single vertex.
 *
 * Mobjects this backend cannot express (multiple fill subpaths, >2 colour
 * stops, fixed-in-frame/fixed-orientation members, background images) return
 * `null` from {@link WebGLThreeDRenderer.prepare} and are drawn by the
 * Canvas2D fallback instead.
 */
import { array } from "numpy-ts";
import { _cContiguousRows } from "../utils/arrays.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { ThreeDCamera, _cornerNormalsFlat, type MobjectBaseColors, type BaseColorInfo } from "../camera/three_d_camera.js";
import { computeSubpathRangesFlat } from "./canvas.js";
import { mobject_geometry_epoch } from "../utils/render_epoch.js";

/** Interleaved vertex layout (floats): position(3) prev(3) next(3) grad0(3)
 *  grad1(3) side(1) halfWidth(1) color0(4) color1(4) normal0(3) normal1(3)
 *  modelIndex(1). */
const FLOATS_PER_VERTEX = 32;
const POSITION_OFFSET = 0;
const PREV_OFFSET = 3;
const NEXT_OFFSET = 6;
const GRAD0_OFFSET = 9;
const GRAD1_OFFSET = 12;
const SIDE_OFFSET = 15;
const HALF_WIDTH_OFFSET = 16;
const COLOR0_OFFSET = 17;
const COLOR1_OFFSET = 21;
const NORMAL0_OFFSET = 25;
const NORMAL1_OFFSET = 28;
const MODEL_INDEX_OFFSET = 31;

/** Number of distinct affine models a frame can carry in the uniform array. */
const MODEL_SLOTS = 64;

const ZERO_RGBA = [0, 0, 0, 0];
const ZERO_VEC = [0, 0, 0];

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_prev;
layout(location = 2) in vec3 a_next;
layout(location = 3) in vec3 a_grad0;
layout(location = 4) in vec3 a_grad1;
layout(location = 5) in float a_side;
layout(location = 8) in float a_half_width;
layout(location = 6) in vec4 a_color0;
layout(location = 7) in vec4 a_color1;
layout(location = 9) in vec3 a_normal0;
layout(location = 10) in vec3 a_normal1;
layout(location = 11) in float a_model_index;

uniform mat3 u_rot;          // rotation matrix rows as columns
uniform vec3 u_center;
uniform float u_focal;
uniform float u_zoom;
uniform bool u_exponential;
uniform vec4 u_screen;       // widthMult, heightMult, addX, addY
uniform vec2 u_viewport;     // pixel width / height
uniform vec3 u_light;
uniform bool u_shading;
uniform vec4 u_models[${MODEL_SLOTS * 3}]; // per-model rows: [a b c t]

out float v_t;
out vec4 v_color0;
out vec4 v_color1;

vec3 apply_model(vec3 p) {
  int base = int(a_model_index) * 3;
  vec4 r0 = u_models[base];
  vec4 r1 = u_models[base + 1];
  vec4 r2 = u_models[base + 2];
  return vec3(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}

vec3 apply_model_dir(vec3 v) {
  int base = int(a_model_index) * 3;
  vec4 r0 = u_models[base];
  vec4 r1 = u_models[base + 1];
  vec4 r2 = u_models[base + 2];
  return vec3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

float shade(vec3 point, vec3 normal) {
  vec3 diff = u_light - point;
  float d = length(diff);
  vec3 to_sun = d > 0.0 ? diff / d : vec3(0.0);
  float c = dot(normalize(normal), to_sun);
  float light = 0.5 * c * c * c;
  if (light < 0.0) light *= 0.5;
  return light;
}

vec2 project(vec3 world) {
  vec3 p = world - u_center;
  vec3 r = vec3(dot(u_rot[0], p), dot(u_rot[1], p), dot(u_rot[2], p));
  float factor;
  if (u_exponential) {
    factor = r.z < 0.0 ? u_focal / (u_focal - r.z) : exp(r.z / u_focal);
  } else {
    factor = u_focal / (u_focal - r.z);
    if (u_focal - r.z < 0.0) factor = 1e6;
  }
  vec2 proj = r.xy * factor * u_zoom;
  return vec2(proj.x * u_screen.x + u_screen.z, proj.y * (-u_screen.y) + u_screen.w);
}

void main() {
  vec3 world = apply_model(a_position);
  vec2 cur = project(world);
  vec2 pos = cur;
  if (a_side != 0.0) {
    vec2 prev = project(apply_model(a_prev));
    vec2 next = project(apply_model(a_next));
    vec2 d1 = cur - prev;
    vec2 d2 = next - cur;
    float l1 = length(d1);
    float l2 = length(d2);
    vec2 n1 = l1 > 0.0 ? vec2(-d1.y, d1.x) / l1 : vec2(0.0);
    vec2 n2 = l2 > 0.0 ? vec2(-d2.y, d2.x) / l2 : vec2(0.0);
    vec2 m = n1 + n2;
    float lm = length(m);
    vec2 dir;
    if (lm < 1e-6) {
      dir = l1 > 0.0 ? n1 : n2;
    } else {
      m /= lm;
      float denom = dot(m, n1);
      // Cairo's default miter limit is 10; clamp there instead of beveling.
      dir = denom > 1e-4 ? m * min(1.0 / denom, 10.0) : n1;
    }
    pos = cur + dir * (a_half_width * a_side);
  }
  vec2 g0 = project(apply_model(a_grad0));
  vec2 g1 = project(apply_model(a_grad1));
  vec2 seg = g1 - g0;
  float len2 = dot(seg, seg);
  v_t = len2 > 0.0 ? dot(pos - g0, seg) / len2 : 0.0;
  vec4 c0 = a_color0;
  vec4 c1 = a_color1;
  if (u_shading) {
    c0.rgb += shade(apply_model(a_grad0), apply_model_dir(a_normal0));
    c1.rgb += shade(apply_model(a_grad1), apply_model_dir(a_normal1));
  }
  v_color0 = c0;
  v_color1 = c1;
  gl_Position = vec4(
    pos.x / u_viewport.x * 2.0 - 1.0,
    1.0 - pos.y / u_viewport.y * 2.0,
    0.0,
    1.0
  );
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float v_t;
in vec4 v_color0;
in vec4 v_color1;
out vec4 outColor;
void main() {
  outColor = mix(v_color0, v_color1, clamp(v_t, 0.0, 1.0));
}
`;

interface GeometryRecord {
  points: unknown;
  pointsVersion: number;
  fillSource: unknown;
  strokeSource: unknown;
  bgStrokeSource: unknown;
  baseColors: MobjectBaseColors;
  strokeWidthRaw: unknown;
  bgStrokeWidthRaw: unknown;
  jointType: string;
  capStyle: string;
  lineMultiple: number;
  zoomBucket: number;
  /**
   * Vertex floats.  After `alloc_vertices` this is a *view* into the shared
   * `vertexHeap` at `vertexOffset` (not a copy), so position updates rewrite
   * the heap in place with zero copies and `upload_pending` ships exactly
   * those bytes to the GPU.  The view is rebound whenever the heap moves
   * (`ensure_vertex_capacity` growth, `compact_vertices`); a dead record's
   * view may alias a live record's heap range and must not be read.
   */
  vertexData: Float32Array;
  /** Triangle indices, already offset by `vertexOffset` (in vertices). */
  indexData: Uint32Array;
  vertexCount: number;
  /** Start of this record's vertices inside the shared vertex heap (floats). */
  vertexOffset: number;
  /** Copy of the points the vertex data was built from (affine fitting). */
  basePoints: Float32Array;
  /** Base->current affine model, or null when the geometry is up to date. */
  model: Float32Array | null;
  modelSlot: number;
  /** Set when the model-slot budget overflowed; forces a geometry rebuild. */
  forceRebuild: boolean;
  /**
   * Per-vertex source point rows (pos, prev, next, grad0, grad1) for
   * topology-stable position updates, or null when the face was subdivided
   * (curved) and must be rebuilt on any change.
   */
  vertexRows: Int32Array | null;
  /** Corner rows used to recompute the face normals on a position update. */
  cornerStartRow: number;
  cornerEndRow: number;
  /**
   * Subpath ranges the record was built from (topology validation, checked
   * allocation-free by `ranges_match_cached`).
   */
  rangeList: Array<readonly [number, number]>;
  rowDim: number;
}

export interface WebGLThreeDRendererOptions {
  antialias?: boolean;
}

function isColorTransparent(info: BaseColorInfo): boolean {
  if (info.gradient && info.colors.length >= 2) {
    return (info.colors[0]?.[3] ?? 0) === 0 && (info.colors[1]?.[3] ?? 0) === 0;
  }
  return (info.colors[0]?.[3] ?? 0) === 0;
}

function flattenCubic(
  out: number[][],
  outRows: number[],
  p0: number[],
  p1: number[],
  p2: number[],
  p3: number[],
  p3Row: number,
  tolerance: number,
): void {
  // Straight segments (the overwhelmingly common case for surfaces) collapse
  // to their endpoints; `set_points_as_corners` produces exactly collinear
  // handles.
  const dx = (p3[0] ?? 0) - (p0[0] ?? 0);
  const dy = (p3[1] ?? 0) - (p0[1] ?? 0);
  const dz = (p3[2] ?? 0) - (p0[2] ?? 0);
  const chord2 = dx * dx + dy * dy + dz * dz;
  const cross = (p: number[]): number => {
    const ax = (p[0] ?? 0) - (p0[0] ?? 0);
    const ay = (p[1] ?? 0) - (p0[1] ?? 0);
    const az = (p[2] ?? 0) - (p0[2] ?? 0);
    const cx = dy * az - dz * ay;
    const cy = dz * ax - dx * az;
    const cz = dx * ay - dy * ax;
    return Math.sqrt(cx * cx + cy * cy + cz * cz);
  };
  const d1 = cross(p1);
  const d2 = cross(p2);
  if (d1 * d1 <= tolerance * tolerance * Math.max(chord2, 1e-30) && d2 * d2 <= tolerance * tolerance * Math.max(chord2, 1e-30)) {
    out.push(p3);
    outRows.push(p3Row);
    return;
  }
  // de Casteljau split at 0.5.  Subdivided midpoints have no source row (-1);
  // only the segment end keeps its row.
  const m01 = mid(p0, p1);
  const m12 = mid(p1, p2);
  const m23 = mid(p2, p3);
  const m012 = mid(m01, m12);
  const m123 = mid(m12, m23);
  const m = mid(m012, m123);
  flattenCubic(out, outRows, p0, m01, m012, m, -1, tolerance);
  flattenCubic(out, outRows, m, m123, m23, p3, p3Row, tolerance);
}

function mid(a: number[], b: number[]): number[] {
  return [
    ((a[0] ?? 0) + (b[0] ?? 0)) / 2,
    ((a[1] ?? 0) + (b[1] ?? 0)) / 2,
    ((a[2] ?? 0) + (b[2] ?? 0)) / 2,
  ];
}

function pointsEqual(a: number[], b: number[], atol: number): boolean {
  const rtol = 1e-5;
  for (let dim = 0; dim < 3; dim += 1) {
    const e = a[dim] ?? NaN;
    const s = b[dim] ?? NaN;
    const diff = Math.abs(e - s);
    if (!((diff <= atol + rtol * Math.abs(s) && Number.isFinite(diff)) || e === s)) return false;
  }
  return true;
}

/**
 * Ear-clipping triangulation of a simple (possibly non-planar-ish) 3D polygon.
 *
 * The polygon is projected onto its dominant plane (chosen from the Newell
 * normal) before clipping: a cube face parallel to the world XY plane has zero
 * area there, so a world-XY projection would produce no triangles and drop the
 * face entirely.
 */
function triangulatePolygon(poly: number[][]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  let nx = 0;
  let ny = 0;
  let nz = 0;
  for (let i = 0; i < n; i += 1) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    nx += ((a[1] ?? 0) - (b[1] ?? 0)) * ((a[2] ?? 0) + (b[2] ?? 0));
    ny += ((a[2] ?? 0) - (b[2] ?? 0)) * ((a[0] ?? 0) + (b[0] ?? 0));
    nz += ((a[0] ?? 0) - (b[0] ?? 0)) * ((a[1] ?? 0) + (b[1] ?? 0));
  }
  const ax = Math.abs(nx);
  const ay = Math.abs(ny);
  const az = Math.abs(nz);
  // Which 3D axes map to the 2D (u, v) plane: drop the dominant normal axis.
  let uAxis: number;
  let vAxis: number;
  if (ax >= ay && ax >= az) { uAxis = 1; vAxis = 2; }
  else if (ay >= az) { uAxis = 2; vAxis = 0; }
  else { uAxis = 0; vAxis = 1; }
  const flat: number[][] = poly.map((p) => [p[uAxis] ?? 0, p[vAxis] ?? 0]);
  return triangulatePolygon2D(flat);
}

/** Ear clipping in 2D (any orientation). */
function triangulatePolygon2D(poly: number[][]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  const indices: number[] = [];
  let area = 0;
  for (let i = 0; i < n; i += 1) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    area += (a[0] ?? 0) * (b[1] ?? 0) - (b[0] ?? 0) * (a[1] ?? 0);
  }
  if (area === 0) return [];
  const ccw = area > 0;
  const remaining: number[] = [];
  for (let i = 0; i < n; i += 1) remaining.push(ccw ? i : n - 1 - i);
  let guard = n * n + 16;
  while (remaining.length > 3 && guard > 0) {
    guard -= 1;
    let clipped = false;
    for (let k = 0; k < remaining.length; k += 1) {
      const i0 = remaining[(k + remaining.length - 1) % remaining.length]!;
      const i1 = remaining[k]!;
      const i2 = remaining[(k + 1) % remaining.length]!;
      const a = poly[i0]!;
      const b = poly[i1]!;
      const c = poly[i2]!;
      const cross =
        ((b[0] ?? 0) - (a[0] ?? 0)) * ((c[1] ?? 0) - (a[1] ?? 0)) -
        ((b[1] ?? 0) - (a[1] ?? 0)) * ((c[0] ?? 0) - (a[0] ?? 0));
      if (cross <= 0) continue;
      let contains = false;
      for (const other of remaining) {
        if (other === i0 || other === i1 || other === i2) continue;
        const p = poly[other]!;
        const c1 =
          ((b[0] ?? 0) - (a[0] ?? 0)) * ((p[1] ?? 0) - (a[1] ?? 0)) -
          ((b[1] ?? 0) - (a[1] ?? 0)) * ((p[0] ?? 0) - (a[0] ?? 0));
        const c2 =
          ((c[0] ?? 0) - (b[0] ?? 0)) * ((p[1] ?? 0) - (b[1] ?? 0)) -
          ((c[1] ?? 0) - (b[1] ?? 0)) * ((p[0] ?? 0) - (b[0] ?? 0));
        const c3 =
          ((a[0] ?? 0) - (c[0] ?? 0)) * ((p[1] ?? 0) - (c[1] ?? 0)) -
          ((a[1] ?? 0) - (c[1] ?? 0)) * ((p[0] ?? 0) - (c[0] ?? 0));
        if (c1 >= 0 && c2 >= 0 && c3 >= 0) {
          contains = true;
          break;
        }
      }
      if (contains) continue;
      indices.push(i0, i1, i2);
      remaining.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  if (remaining.length === 3) indices.push(remaining[0]!, remaining[1]!, remaining[2]!);
  return indices;
}

/**
 * Whether the current points still produce `cached` subpath ranges, without
 * allocating: every cached range start (except 0) must be a split boundary
 * and every other multiple-of-`nppcc` joint must be continuous, which is
 * exactly what `computeSubpathRangesFlat` (2 dims) computes.  The cached
 * ranges must also tile `[0, rowCount)` contiguously.
 */
function ranges_match_cached(
  data: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
  nppcc: number,
  tolerance: number,
  cached: ReadonlyArray<readonly [number, number]>,
): boolean {
  if (cached.length === 0) {
    return computeSubpathRangesFlat(data, rowDim, rowCount, nppcc, tolerance, 2).length === 0;
  }
  if (cached[0]![0] !== 0) return false;
  for (let i = 0; i < cached.length; i += 1) {
    const start = cached[i]![0];
    const end = cached[i]![1];
    if (i > 0 && cached[i - 1]![1] !== start) return false;
    if (i === cached.length - 1 && end !== rowCount) return false;
  }
  const rtol = 1.0e-5;
  const atol = tolerance;
  const isSplit = (boundary: number): boolean => {
    const endBase = (boundary - 1) * rowDim;
    const startBase = boundary * rowDim;
    for (let dim = 0; dim < 2; dim += 1) {
      const e = data[endBase + dim] ?? NaN;
      const s = data[startBase + dim] ?? NaN;
      const diff = Math.abs((e as number) - (s as number));
      const close = (diff <= atol + rtol * Math.abs(s as number) && Number.isFinite(diff)) || e === s;
      if (!close) return true;
    }
    return false;
  };
  let rangeIndex = 0;
  let nextSplit = cached[0]![1] === rowCount ? -1 : cached[0]![1];
  for (let boundary = nppcc; boundary < rowCount; boundary += nppcc) {
    const split = isSplit(boundary);
    if (boundary === nextSplit) {
      if (!split) return false;
      rangeIndex += 1;
      nextSplit = rangeIndex < cached.length
        ? (cached[rangeIndex]![1] === rowCount ? -1 : cached[rangeIndex]![1])
        : -1;
    } else if (split) {
      return false;
    }
  }
  return true;
}

/**
 * Scratch space for `fitAffine`/`invert4`: the fit runs per face per frame,
 * and these temporaries were the bulk of its allocations.  Module state is
 * safe: fitting is fully synchronous with no reentrancy.
 */
const fitScratchM = new Float64Array(16);
const fitScratchRhs = new Float64Array(12);
const fitScratchRow = new Float64Array(4);
const fitScratchAugmented = new Float64Array(32);
const fitScratchInv = new Float64Array(16);

/**
 * Gauss-Jordan inversion of a 4x4 matrix in row-major order.  The result
 * lives in shared scratch space: consume it immediately, do not retain it.
 */
function invert4(m: Float64Array): Float64Array | null {
  const a = fitScratchAugmented;
  // The whole augmented row must be rebuilt: the identity half keeps
  // elimination residue from the previous fit, and stale values there would
  // corrupt the inverse (the old per-call allocation was zero-initialised).
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) a[r * 8 + c] = m[r * 4 + c]!;
    for (let c = 4; c < 8; c += 1) a[r * 8 + c] = 0;
    a[r * 8 + 4 + r] = 1;
  }
  for (let col = 0; col < 4; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < 4; r += 1) {
      if (Math.abs(a[r * 8 + col]!) > Math.abs(a[pivot * 8 + col]!)) pivot = r;
    }
    if (Math.abs(a[pivot * 8 + col]!) < 1e-12) return null;
    if (pivot !== col) {
      for (let c = 0; c < 8; c += 1) {
        const t = a[col * 8 + c]!;
        a[col * 8 + c] = a[pivot * 8 + c]!;
        a[pivot * 8 + c] = t;
      }
    }
    const d = a[col * 8 + col]!;
    for (let c = 0; c < 8; c += 1) a[col * 8 + c] = a[col * 8 + c]! / d;
    for (let r = 0; r < 4; r += 1) {
      if (r === col) continue;
      const f = a[r * 8 + col]!;
      if (f === 0) continue;
      for (let c = 0; c < 8; c += 1) a[r * 8 + c] = a[r * 8 + c]! - f * a[col * 8 + c]!;
    }
  }
  const inv = fitScratchInv;
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) inv[r * 4 + c] = a[r * 8 + 4 + c]!;
  }
  return inv;
}

/** Maximum residual (relative to the point scale) allowed for an affine fit. */
function model_residual_ok(maxErr: number, scale: number): boolean {
  return maxErr <= 1e-6 * (1 + scale);
}

/**
 * Fit the 3x4 affine map `current ≈ A · [base; 1]` by least squares.  Returns
 * the 12 coefficients in row-major order (`A[row][col]`, translation in the
 * last column), or null when the fit is singular or not affine within
 * tolerance (in which case the caller rebuilds the geometry).
 */
function fitAffine(
  base: Float32Array,
  current: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
): Float32Array | null {
  const m = fitScratchM;
  const rhs = fitScratchRhs;
  const row = fitScratchRow;
  m.fill(0);
  rhs.fill(0);
  // `current` is float64 in every real flow (point stores are float64); the
  // direct read avoids a `Number()` call per element per face per frame.
  const currentF64 = current instanceof Float64Array ? current : null;
  for (let i = 0; i < rowCount; i += 1) {
    const x = base[i * 3]!;
    const y = base[i * 3 + 1]!;
    const z = base[i * 3 + 2]!;
    const c = i * rowDim;
    row[0] = x;
    row[1] = y;
    row[2] = z;
    row[3] = 1;
    const c0 = currentF64 !== null ? currentF64[c]! : Number(current[c]!);
    const c1 = currentF64 !== null ? currentF64[c + 1]! : Number(current[c + 1]!);
    const c2 = currentF64 !== null ? currentF64[c + 2]! : Number(current[c + 2]!);
    for (let a = 0; a < 4; a += 1) {
      const ra = row[a]!;
      for (let b = 0; b < 4; b += 1) m[a * 4 + b] += ra * row[b]!;
      rhs[a * 3] += ra * c0;
      rhs[a * 3 + 1] += ra * c1;
      rhs[a * 3 + 2] += ra * c2;
    }
  }
  const inv = invert4(m);
  if (inv === null) return null;
  const out = new Float32Array(12);
  for (let a = 0; a < 4; a += 1) {
    for (let b = 0; b < 3; b += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += inv[a * 4 + k]! * rhs[k * 3 + b]!;
      out[b * 4 + a] = sum;
    }
  }
  // Verifying with a precomputed scale allows an exact early exit:
  // `model_matches` cannot exit early (its threshold needs the full-data
  // scale), but here the threshold is known up front, so the predicate is
  // identical with strictly less work on failure.
  if (!verifyFit(out, base, current, rowDim, rowCount)) return null;
  return out;
}

/**
 * Exact counterpart of the `model_matches` check used to verify a fresh fit:
 * the scale pass fixes the `model_residual_ok` threshold up front, so
 * failing rows exit immediately.  Returns exactly what `model_matches`
 * would (`maxErr <= tol` iff no row exceeds `tol`).
 */
function verifyFit(
  model: Float32Array,
  base: Float32Array,
  current: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
): boolean {
  const currentF64 = current instanceof Float64Array ? current : null;
  let scale = 0;
  const total = rowCount * rowDim;
  if (currentF64 !== null) {
    for (let i = 0; i < total; i += 1) {
      const abs = Math.abs(currentF64[i]!);
      if (abs > scale) scale = abs;
    }
  } else {
    for (let i = 0; i < total; i += 1) {
      const abs = Math.abs(Number(current[i]!));
      if (abs > scale) scale = abs;
    }
  }
  const tol = 1e-6 * (1 + scale);
  for (let i = 0; i < rowCount; i += 1) {
    const x = base[i * 3]!;
    const y = base[i * 3 + 1]!;
    const z = base[i * 3 + 2]!;
    const c = i * rowDim;
    for (let b = 0; b < 3; b += 1) {
      const pred =
        model[b * 4]! * x + model[b * 4 + 1]! * y + model[b * 4 + 2]! * z + model[b * 4 + 3]!;
      const actual = currentF64 !== null ? currentF64[c + b]! : Number(current[c + b]!);
      if (Math.abs(pred - actual) > tol) return false;
    }
  }
  return true;
}

/** Whether `model` maps every base point onto the current points. */
function model_matches(
  model: Float32Array,
  base: Float32Array,
  current: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
): boolean {
  let maxErr = 0;
  let scale = 0;
  const currentF64 = current instanceof Float64Array ? current : null;
  for (let i = 0; i < rowCount; i += 1) {
    const x = base[i * 3]!;
    const y = base[i * 3 + 1]!;
    const z = base[i * 3 + 2]!;
    const c = i * rowDim;
    for (let b = 0; b < 3; b += 1) {
      const pred =
        model[b * 4]! * x + model[b * 4 + 1]! * y + model[b * 4 + 2]! * z + model[b * 4 + 3]!;
      const actual = currentF64 !== null ? currentF64[c + b]! : Number(current[c + b]!);
      const err = Math.abs(pred - actual);
      if (err > maxErr) maxErr = err;
      const abs = Math.abs(actual);
      if (abs > scale) scale = abs;
    }
  }
  return model_residual_ok(maxErr, scale);
}

/**
 * Cheap pre-check before a full `model_matches` verification: evaluate the
 * candidate model on up to four spread rows and reject obvious non-matches
 * without scanning every row.  A rejection only skips that candidate's full
 * scan — the least-squares fit below still runs whenever nothing verifies,
 * so incremental affine motion keeps fitting its new model exactly as
 * before.  Exact-affine candidates (residual ~1e-15) always pass through to
 * the full verification; waves fail on the first sample rows.
 */
function model_sample_rejects(
  model: Float32Array,
  base: Float32Array,
  current: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
): boolean {
  // No closures or index arrays here (a previous version allocated both per
  // call — twice per face per frame — and it showed up in profiles).
  const f64 = current instanceof Float64Array ? current : null;
  for (let s = 0; s < 4; s += 1) {
    let i: number;
    if (rowCount < 4) {
      if (s >= rowCount) break;
      i = s;
    } else {
      i = s === 0 ? 0 : s === 1 ? (rowCount / 3) | 0 : s === 2 ? ((2 * rowCount) / 3) | 0 : rowCount - 1;
    }
    const x = base[i * 3]!;
    const y = base[i * 3 + 1]!;
    const z = base[i * 3 + 2]!;
    const c = i * rowDim;
    const a0 = f64 !== null ? f64[c]! : Number(current[c]!);
    const a1 = f64 !== null ? f64[c + 1]! : Number(current[c + 1]!);
    const a2 = f64 !== null ? f64[c + 2]! : Number(current[c + 2]!);
    // NaN actuals are ignored for the scale (they can never exceed the
    // tolerance anyway), exactly like the loop this replaces.
    let localScale = 0;
    if (a0 === a0) {
      const m = Math.abs(a0);
      if (m > localScale) localScale = m;
    }
    if (a1 === a1) {
      const m = Math.abs(a1);
      if (m > localScale) localScale = m;
    }
    if (a2 === a2) {
      const m = Math.abs(a2);
      if (m > localScale) localScale = m;
    }
    const tol = 1e-6 * (1 + localScale);
    for (let b = 0; b < 3; b += 1) {
      const actual = b === 0 ? a0 : b === 1 ? a1 : a2;
      const pred =
        model[b * 4]! * x + model[b * 4 + 1]! * y + model[b * 4 + 2]! * z + model[b * 4 + 3]!;
      if (Math.abs(pred - actual) > tol) return true;
    }
  }
  return false;
}

export class WebGLThreeDRenderer {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;
  private ebo: WebGLBuffer | null = null;
  private modelVbo: WebGLBuffer | null = null;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private records = new Map<VMobject, GeometryRecord>();
  /** CPU staging copy of the shared vertex heap (offsets survive growth). */
  private vertexHeap = new Float32Array(4096 * FLOATS_PER_VERTEX);
  private vertexHeapUsed = 0;
  /** GPU vertex buffer capacity, in floats. */
  private vertexCapacity = 0;
  /** Freed ranges inside the heap, `{ offset, size }` in floats. */
  private freeRanges: Array<{ offset: number; size: number }> = [];
  /** Per-vertex model index, parallel to the vertex heap (one float/vertex). */
  private modelIndexHeap = new Float32Array(4096);
  private indexScratch = new Uint32Array(8192);
  private indexCount = 0;
  /** GPU element buffer capacity, in indices. */
  private indexCapacity = 0;
  /** Dirty ranges uploaded once per frame. */
  private pendingVertexMin = Infinity;
  private pendingVertexMax = -Infinity;
  private pendingModelMin = Infinity;
  private pendingModelMax = -Infinity;
  private width = 0;
  private height = 0;
  /**
   * When the global geometry epoch and the camera's record-relevant state are
   * unchanged from the previous frame, every cached record is still valid and
   * `prepare` can skip per-face validation entirely (camera-only movement).
   */
  private recordsValid = false;
  private frameState = {
    epoch: -1,
    lineMultiple: NaN,
    zoomBucket: NaN,
    fif: -1,
    fio: -1,
    frameHeight: NaN,
    pixelHeight: NaN,
  };
  private frameRecords: GeometryRecord[] = [];
  private frameModelSlots = new Map<Float32Array, number>();
  private modelArray = new Float32Array(MODEL_SLOTS * 12);
  /** Last fitted affine transform, reused by the other faces of its group. */
  private lastFittedModel: Float32Array | null = null;

  constructor(options: WebGLThreeDRendererOptions = {}) {
    try {
      const canvas = typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(1, 1)
        : typeof document !== "undefined"
          ? document.createElement("canvas")
          : null;
      if (canvas === null) return;
      const gl = (canvas as HTMLCanvasElement).getContext("webgl2", {
        alpha: true,
        antialias: options.antialias ?? true,
        // Keep the default (premultiplied) framebuffer: blending with
        // SRC_ALPHA leaves premultiplied values in the buffer, which is what
        // `drawImage` expects when compositing the offscreen canvas.
        premultipliedAlpha: true,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: false,
      }) as WebGL2RenderingContext | null;
      if (gl === null) return;
      this.canvas = canvas;
      this.gl = gl;
      this.init(gl);
    } catch {
      this.gl = null;
    }
  }

  available(): boolean {
    return this.gl !== null && this.program !== null && this.vbo !== null && this.ebo !== null && this.vao !== null && this.modelVbo !== null;
  }

  /**
   * Release the WebGL context.  Each scene run creates a renderer, and the
   * browser only keeps a handful of live WebGL contexts, so embedders that
   * re-run scenes must dispose the previous one.
   */
  dispose(): void {
    const gl = this.gl;
    if (gl !== null) {
      const lose = gl.getExtension("WEBGL_lose_context");
      lose?.loseContext();
    }
    this.gl = null;
    this.program = null;
    this.vao = null;
    this.vbo = null;
    this.ebo = null;
    this.modelVbo = null;
    this.records.clear();
    this.frameRecords.length = 0;
  }

  private init(gl: WebGL2RenderingContext): void {
    const compile = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (shader === null) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("wanim webgl shader", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };
    const vs = compile(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    if (vs === null || fs === null) return;
    const program = gl.createProgram();
    if (program === null) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("wanim webgl link", gl.getProgramInfoLog(program));
      return;
    }
    this.program = program;
    gl.useProgram(program);
    for (const name of ["u_rot", "u_center", "u_focal", "u_zoom", "u_exponential", "u_screen", "u_viewport", "u_light", "u_shading", "u_models"]) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    const ebo = gl.createBuffer();
    const modelVbo = gl.createBuffer();
    if (vao === null || vbo === null || ebo === null || modelVbo === null) return;
    this.vao = vao;
    this.vbo = vbo;
    this.ebo = ebo;
    this.modelVbo = modelVbo;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const stride = FLOATS_PER_VERTEX * 4;
    const attribs: Array<[number, number, number]> = [
      // location, size, offsetFloats
      [0, 3, POSITION_OFFSET],
      [1, 3, PREV_OFFSET],
      [2, 3, NEXT_OFFSET],
      [3, 3, GRAD0_OFFSET],
      [4, 3, GRAD1_OFFSET],
      [5, 1, SIDE_OFFSET],
      [8, 1, HALF_WIDTH_OFFSET],
      [6, 4, COLOR0_OFFSET],
      [7, 4, COLOR1_OFFSET],
      [9, 3, NORMAL0_OFFSET],
      [10, 3, NORMAL1_OFFSET],
    ];
    for (const [location, size, offsetFloats] of attribs) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offsetFloats * 4);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, modelVbo);
    gl.enableVertexAttribArray(11);
    gl.vertexAttribPointer(11, 1, gl.FLOAT, false, 4, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  /** Resize/clear the offscreen surface and reset the per-frame batches. */
  begin(camera: ThreeDCamera): void {
    const gl = this.gl;
    if (gl === null || this.canvas === null) return;
    const width = camera.pixel_width;
    const height = camera.pixel_height;
    if (this.width !== width || this.height !== height) {
      this.width = width;
      this.height = height;
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.indexCount = 0;
    this.frameRecords.length = 0;
    this.frameModelSlots.clear();
    // NOTE: dirty vertex/model ranges are *not* reset here — `prepare` builds
    // (and dirties) records before `begin`, and only `upload_pending` (called
    // from `flush`) may clear them.
    const state = this.camera_state(camera);
    this.recordsValid =
      state.epoch === this.frameState.epoch &&
      state.lineMultiple === this.frameState.lineMultiple &&
      state.zoomBucket === this.frameState.zoomBucket &&
      state.fif === this.frameState.fif &&
      state.fio === this.frameState.fio &&
      state.frameHeight === this.frameState.frameHeight &&
      state.pixelHeight === this.frameState.pixelHeight;
    this.frameState = state;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private camera_state(camera: ThreeDCamera): WebGLThreeDRenderer["frameState"] {
    const zoom = camera.get_zoom();
    return {
      epoch: mobject_geometry_epoch(),
      lineMultiple: camera.cairo_line_width_multiple,
      zoomBucket: Math.round(Math.log2(Math.max(1, zoom)) * 2),
      fif: camera.fixed_in_frame_mobjects.size,
      fio: camera.fixed_orientation_mobjects.size,
      frameHeight: camera.frame_height,
      pixelHeight: camera.pixel_height,
    };
  }

  /**
   * Build (or validate from cache) the geometry record of a mobject.  Returns
   * `null` when this backend cannot draw it; the caller then falls back to
   * Canvas2D for the frame.
   */
  prepare(mobject: VMobject, camera: ThreeDCamera): GeometryRecord | null {
    if (this.recordsValid) {
      const existing = this.records.get(mobject);
      if (existing !== undefined) return existing;
    }
    return this.ensure_record(mobject, camera);
  }

  /** Append a prepared record to the current frame batch. */
  push(record: GeometryRecord): void {
    if (record.indexData.length === 0) return;
    const needed = this.indexCount + record.indexData.length;
    this.ensure_index_capacity(needed);
    this.indexScratch.set(record.indexData, this.indexCount);
    this.indexCount = needed;
    this.frameRecords.push(record);
  }

  /** Draw the accumulated batch (painter order preserved). */
  flush(camera: ThreeDCamera): void {
    const gl = this.gl;
    if (gl === null || this.program === null || this.vao === null || this.vbo === null || this.ebo === null) return;
    if (this.indexCount === 0) return;
    this.assign_model_slots(gl);
    const pixelWidth = camera.pixel_width;
    const pixelHeight = camera.pixel_height;
    const frameWidth = camera.frame_width;
    const frameHeight = camera.frame_height;
    const widthMult = pixelWidth / frameWidth;
    const heightMult = pixelHeight / frameHeight;
    const frameCenter = camera.frame_center;
    const [r0, r1, r2] = camera.get_rotation_rows();
    const lightRows = _cContiguousRows(camera.light_source.points);
    const light: [number, number, number] = lightRows !== null && lightRows.rowCount > 0
      ? [Number(lightRows.data[0] ?? 0), Number(lightRows.data[1] ?? 0), Number(lightRows.data[2] ?? 0)]
      : this.light_source;
    this.light_source = light;
    gl.useProgram(this.program);
    gl.uniformMatrix3fv(this.uniforms["u_rot"] ?? null, false, new Float32Array([
      r0[0] ?? 0, r0[1] ?? 0, r0[2] ?? 0,
      r1[0] ?? 0, r1[1] ?? 0, r1[2] ?? 0,
      r2[0] ?? 0, r2[1] ?? 0, r2[2] ?? 0,
    ]));
    gl.uniform3f(
      this.uniforms["u_center"] ?? null,
      frameCenter[0] ?? 0,
      frameCenter[1] ?? 0,
      frameCenter[2] ?? 0,
    );
    gl.uniform1f(this.uniforms["u_focal"] ?? null, camera.get_focal_distance());
    gl.uniform1f(this.uniforms["u_zoom"] ?? null, camera.get_zoom());
    gl.uniform1i(this.uniforms["u_exponential"] ?? null, camera.exponential_projection ? 1 : 0);
    gl.uniform4f(
      this.uniforms["u_screen"] ?? null,
      widthMult,
      heightMult,
      pixelWidth / 2 - (frameCenter[0] ?? 0) * widthMult,
      pixelHeight / 2 + (frameCenter[1] ?? 0) * heightMult,
    );
    gl.uniform2f(this.uniforms["u_viewport"] ?? null, pixelWidth, pixelHeight);
    gl.uniform3f(this.uniforms["u_light"] ?? null, light[0], light[1], light[2]);
    gl.uniform1i(this.uniforms["u_shading"] ?? null, camera.should_apply_shading ? 1 : 0);
    gl.uniform4fv(this.uniforms["u_models"] ?? null, this.modelArray);
    this.upload_pending();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, this.indexScratch.subarray(0, this.indexCount));
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  private light_source: [number, number, number] = [0, 0, 0];

  /**
   * Assign this frame's model slots and write the per-vertex model indices.
   * Slots are per-frame and shared by records whose fitted transform object is
   * identical (all faces of one animated group).
   */
  private assign_model_slots(gl: WebGL2RenderingContext): void {
    this.modelArray.fill(0);
    this.modelArray[0] = 1;
    this.modelArray[5] = 1;
    this.modelArray[10] = 1;
    this.frameModelSlots.clear();
    let slotCount = 1;
    for (const record of this.frameRecords) {
      const model = record.model;
      if (model === null) {
        if (record.modelSlot !== 0) this.write_model_slot(record, 0);
        continue;
      }
      let slot = this.frameModelSlots.get(model);
      if (slot === undefined) {
        if (slotCount >= MODEL_SLOTS) {
          // More simultaneously animated groups than the uniform array can
          // hold: force a geometry rebuild for this record next frame.
          record.model = null;
          record.forceRebuild = true;
          if (record.modelSlot !== 0) this.write_model_slot(record, 0);
          continue;
        }
        slot = slotCount;
        slotCount += 1;
        this.frameModelSlots.set(model, slot);
        const base = slot * 12;
        for (let index = 0; index < 12; index += 1) this.modelArray[base + index] = model[index]!;
      }
      if (record.modelSlot !== slot) this.write_model_slot(record, slot);
    }
  }

  private write_model_slot(record: GeometryRecord, slot: number): void {
    record.modelSlot = slot;
    const base = record.vertexOffset / FLOATS_PER_VERTEX;
    this.modelIndexHeap.fill(slot, base, base + record.vertexCount);
    if (base < this.pendingModelMin) this.pendingModelMin = base;
    if (base + record.vertexCount > this.pendingModelMax) this.pendingModelMax = base + record.vertexCount;
  }

  private upload_pending(): void {
    const gl = this.gl;
    if (gl === null || this.vbo === null) return;
    if (this.pendingVertexMin < this.pendingVertexMax) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(
        gl.ARRAY_BUFFER,
        this.pendingVertexMin * 4,
        this.vertexHeap.subarray(this.pendingVertexMin, this.pendingVertexMax),
      );
      this.pendingVertexMin = Infinity;
      this.pendingVertexMax = -Infinity;
    }
    if (this.modelVbo !== null && this.pendingModelMin < this.pendingModelMax) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.modelVbo);
      gl.bufferSubData(
        gl.ARRAY_BUFFER,
        this.pendingModelMin * 4,
        this.modelIndexHeap.subarray(this.pendingModelMin, this.pendingModelMax),
      );
      this.pendingModelMin = Infinity;
      this.pendingModelMax = -Infinity;
    }
  }

  /** Upload a record's vertices into the shared heap (once per shape change). */
  private alloc_vertices(record: GeometryRecord): void {
    const count = record.vertexData.length;
    if (count === 0) {
      record.vertexOffset = 0;
      return;
    }
    let offset = -1;
    for (let index = 0; index < this.freeRanges.length; index += 1) {
      const range = this.freeRanges[index]!;
      if (range.size >= count) {
        offset = range.offset;
        if (range.size === count) this.freeRanges.splice(index, 1);
        else {
          range.offset += count;
          range.size -= count;
        }
        break;
      }
    }
    if (offset < 0) {
      if (this.freeRanges.length > 0 && this.vertexHeapUsed + count > this.vertexHeap.length) {
        this.compact_vertices();
      }
      offset = this.vertexHeapUsed;
      this.vertexHeapUsed += count;
      this.ensure_vertex_capacity(this.vertexHeapUsed);
    }
    this.vertexHeap.set(record.vertexData, offset);
    record.vertexOffset = offset;
    // Re-point at the heap: later position updates rewrite the vertices
    // through this view, landing directly in the upload heap with no copy.
    // (`ensure_vertex_capacity` may grow the heap afterwards — it rebinds
    // every live record, and this record is not yet in the map here.)
    record.vertexData = this.vertexHeap.subarray(offset, offset + count);
    // Global indices: `indexData` is local to the record right after build.
    // Indices address vertices, so the float offset must be divided by the
    // vertex stride.
    const vertexBase = offset / FLOATS_PER_VERTEX;
    const indexData = record.indexData;
    if (indexData.length > 0) {
      for (let index = 0; index < indexData.length; index += 1) indexData[index] = indexData[index]! + vertexBase;
    }
    // Default model index 0 (identity); `assign_model_slots` rewrites it when
    // an affine model is active.
    this.modelIndexHeap.fill(0, vertexBase, vertexBase + record.vertexCount);
    record.modelSlot = 0;
    if (offset < this.pendingVertexMin) this.pendingVertexMin = offset;
    if (offset + count > this.pendingVertexMax) this.pendingVertexMax = offset + count;
    if (vertexBase < this.pendingModelMin) this.pendingModelMin = vertexBase;
    if (vertexBase + record.vertexCount > this.pendingModelMax) this.pendingModelMax = vertexBase + record.vertexCount;
  }

  private free_vertices(record: GeometryRecord): void {
    const count = record.vertexCount * FLOATS_PER_VERTEX;
    if (count > 0) this.freeRanges.push({ offset: record.vertexOffset, size: count });
  }

  /** Repack all live records, removing fragmentation, and re-upload. */
  private compact_vertices(): void {
    const gl = this.gl;
    const live = [...this.records.values()]
      .filter((record) => record.vertexCount > 0)
      .sort((a, b) => a.vertexOffset - b.vertexOffset);
    const heap = new Float32Array(Math.max(this.vertexHeap.length, 4096 * FLOATS_PER_VERTEX));
    const modelHeap = new Float32Array(heap.length / FLOATS_PER_VERTEX);
    let offset = 0;
    for (const record of live) {
      const count = record.vertexCount * FLOATS_PER_VERTEX;
      const vertexBase = offset / FLOATS_PER_VERTEX;
      const deltaVertices = vertexBase - record.vertexOffset / FLOATS_PER_VERTEX;
      heap.set(this.vertexHeap.subarray(record.vertexOffset, record.vertexOffset + count), offset);
      modelHeap.set(
        this.modelIndexHeap.subarray(record.vertexOffset / FLOATS_PER_VERTEX, record.vertexOffset / FLOATS_PER_VERTEX + record.vertexCount),
        vertexBase,
      );
      record.vertexOffset = offset;
      // The heap moved: rebind the record's view (see `vertexData`).
      record.vertexData = heap.subarray(offset, offset + count);
      if (deltaVertices !== 0) {
        for (let index = 0; index < record.indexData.length; index += 1) {
          record.indexData[index] = record.indexData[index]! + deltaVertices;
        }
      }
      offset += count;
    }
    this.vertexHeap = heap;
    this.modelIndexHeap = modelHeap;
    this.vertexHeapUsed = offset;
    this.freeRanges = [];
    if (gl !== null && this.vbo !== null) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, heap, 0, offset);
    }
    if (gl !== null && this.modelVbo !== null) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.modelVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, modelHeap, 0, offset / FLOATS_PER_VERTEX);
    }
    this.pendingVertexMin = Infinity;
    this.pendingVertexMax = -Infinity;
    this.pendingModelMin = Infinity;
    this.pendingModelMax = -Infinity;
  }

  private ensure_vertex_capacity(floats: number): void {
    const gl = this.gl;
    if (floats > this.vertexHeap.length) {
      let size = this.vertexHeap.length;
      while (size < floats) size *= 2;
      const heap = new Float32Array(size);
      heap.set(this.vertexHeap);
      this.vertexHeap = heap;
      const modelHeap = new Float32Array(size / FLOATS_PER_VERTEX);
      modelHeap.set(this.modelIndexHeap);
      this.modelIndexHeap = modelHeap;
      // The heap moved: rebind every live record's view (see `vertexData`).
      for (const record of this.records.values()) {
        const count = record.vertexCount * FLOATS_PER_VERTEX;
        if (count > 0) {
          record.vertexData = heap.subarray(record.vertexOffset, record.vertexOffset + count);
        }
      }
    }
    if (gl === null || this.vbo === null || this.modelVbo === null) return;
    if (floats <= this.vertexCapacity) return;
    let capacity = Math.max(this.vertexCapacity * 2, floats, 4096 * FLOATS_PER_VERTEX);
    if (capacity < this.vertexHeap.length) capacity = this.vertexHeap.length;
    this.vertexCapacity = capacity;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertexHeap, 0, this.vertexHeapUsed);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.modelVbo);
    const modelCapacity = capacity / FLOATS_PER_VERTEX;
    gl.bufferData(gl.ARRAY_BUFFER, modelCapacity * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.modelIndexHeap, 0, this.vertexHeapUsed / FLOATS_PER_VERTEX);
    this.pendingVertexMin = Infinity;
    this.pendingVertexMax = -Infinity;
    this.pendingModelMin = Infinity;
    this.pendingModelMax = -Infinity;
  }

  private ensure_index_capacity(indices: number): void {
    const gl = this.gl;
    if (indices > this.indexScratch.length) {
      let size = this.indexScratch.length;
      while (size < indices) size *= 2;
      const scratch = new Uint32Array(size);
      scratch.set(this.indexScratch);
      this.indexScratch = scratch;
    }
    if (gl === null || this.ebo === null) return;
    if (indices <= this.indexCapacity) return;
    this.indexCapacity = Math.max(this.indexCapacity * 2, indices, 8192);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indexCapacity * 4, gl.DYNAMIC_DRAW);
  }

  private ensure_record(vmobject: VMobject, camera: ThreeDCamera): GeometryRecord | null {
    const vm = vmobject;
    if (vm.shade_in_3d !== true) return null;
    if (camera.fixed_in_frame_mobjects.has(vm) || camera.fixed_orientation_mobjects.has(vm)) return null;
    if (vm.background_image !== null && vm.background_image !== undefined) return null;
    const points = vm.points;
    if (points === undefined || points === null || (points.shape?.[0] ?? 0) === 0) return null;
    const fillSource = vm.fill_rgbas ?? ZERO_RGBA_SOURCE;
    const strokeSource = vm.stroke_rgbas ?? ZERO_RGBA_SOURCE;
    const bgStrokeSource = vm.background_stroke_rgbas ?? ZERO_RGBA_SOURCE;
    const zoom = camera.get_zoom();
    const zoomBucket = Math.round(Math.log2(Math.max(1, zoom)) * 2);
    const lineMultiple = camera.cairo_line_width_multiple;
    const existing = this.records.get(vm);
    if (
      existing !== undefined &&
      existing.fillSource === fillSource &&
      existing.strokeSource === strokeSource &&
      existing.bgStrokeSource === bgStrokeSource &&
      existing.strokeWidthRaw === vm.stroke_width &&
      existing.bgStrokeWidthRaw === vm.background_stroke_width &&
      existing.jointType === vm.joint_type &&
      existing.capStyle === vm.cap_style &&
      existing.lineMultiple === lineMultiple &&
      existing.zoomBucket === zoomBucket
    ) {
      if (existing.pointsVersion === vm.points_version && !existing.forceRebuild) return existing;
      // Geometry moved: an affine transform of the uploaded points needs no
      // rebuild at all — the shader applies it.
      const pts = _cContiguousRows(points);
      if (pts === null || pts.rowDim < 3 || pts.rowCount !== existing.basePoints.length / 3) {
        // Layout itself changed; fall through to a full rebuild below.
      } else {
        // Sample-gated verification: a candidate that fails a few sample
        // rows cannot verify, so its full scan is skipped.  (A sample reject
        // only skips verification scans — the fit below still runs whenever
        // nothing verifies, so incremental affine motion such as per-frame
        // `rotate()` keeps fitting its new model exactly as before.  In the
        // hairline case where samples reject but the full scan would have
        // passed, the record takes the fit/update path instead of reusing the
        // old model; both describe the same geometry within the fit
        // tolerance, i.e. far below a pixel.)
        const base = existing.basePoints;
        const data = pts.data;
        const rowDim = pts.rowDim;
        const rowCount = pts.rowCount;
        const prevModel = existing.model;
        let model: Float32Array | null = null;
        if (
          prevModel !== null &&
          !model_sample_rejects(prevModel, base, data, rowDim, rowCount) &&
          model_matches(prevModel, base, data, rowDim, rowCount)
        ) {
          model = prevModel;
        }
        if (model === null) {
          const sharedModel = this.lastFittedModel;
          if (
            sharedModel !== null &&
            !model_sample_rejects(sharedModel, base, data, rowDim, rowCount) &&
            model_matches(sharedModel, base, data, rowDim, rowCount)
          ) {
            model = sharedModel;
          }
        }
        if (model === null) {
          model = fitAffine(base, data, rowDim, rowCount);
          if (model !== null) this.lastFittedModel = model;
        }
        if (model !== null && !existing.forceRebuild) {
          existing.model = model;
          existing.points = points;
          existing.pointsVersion = vm.points_version;
          return existing;
        }
      }
      // Topology-stable position update: same point layout, so the
      // triangulation, stroke plan and colors are reused; only the uploaded
      // positions and normals move.  This is the `Surface` wave / ValueTracker
      // case: every frame would otherwise pay a full rebuild.
      if (
        existing.vertexRows !== null &&
        !existing.forceRebuild &&
        pts !== null &&
        pts.rowDim === existing.rowDim &&
        pts.rowCount === existing.basePoints.length / 3 &&
        ranges_match_cached(pts.data, pts.rowDim, pts.rowCount, vm.n_points_per_cubic_curve, vm.tolerance_for_point_equality, existing.rangeList)
      ) {
        this.update_record_positions(existing, vm, pts);
        return existing;
      }
    }
    const baseColors = camera.get_mobject_base_colors(vm);
    if (baseColors === null) return null;
    const record = this.build_record(vm, camera, baseColors, fillSource, strokeSource, bgStrokeSource, zoom, zoomBucket, lineMultiple);
    if (record === null) return null;
    if (existing !== undefined) this.free_vertices(existing);
    this.alloc_vertices(record);
    this.records.set(vm, record);
    return record;
  }

  /**
   * Rewrite a record's positions and normals in place from the current points.
   * Only valid when the point layout (row count, subpath ranges) is unchanged
   * and no subdivision happened at build time (see `vertexRows`).  Colors,
   * triangulation and stroke plan are reused; shading follows automatically
   * because the shader evaluates it from the updated normals.
   */
  private update_record_positions(
    record: GeometryRecord,
    vm: VMobject,
    pts: { data: ArrayLike<number>; rowDim: number; rowCount: number },
  ): void {
    // Normals only: the full base-colour resolution (rgba row reads per
    // source) is wasted here — the record reuses its baked colours and only
    // the positions/normals move.  Same normals `get_mobject_base_colors`
    // would report (shared across all three colour sources).
    const normals = _cornerNormalsFlat(vm.points);
    const n0 = normals?.normal0 ?? ZERO_VEC;
    const n1 = normals?.normal1 ?? ZERO_VEC;
    const { data, rowDim } = pts;
    // `data` is float64 in every real flow (point stores are float64); direct
    // reads avoid a `Number()` call per component per vertex per frame.
    const f64 = data instanceof Float64Array ? data : null;
    const read = (index: number): number =>
      f64 !== null ? f64[index]! : Number(data[index] ?? 0);
    const vertexData = record.vertexData;
    const vertexRows = record.vertexRows!;
    for (let v = 0; v < record.vertexCount; v += 1) {
      const base = v * FLOATS_PER_VERTEX;
      const rbase = v * 5;
      const rPos = vertexRows[rbase]! * rowDim;
      vertexData[base] = read(rPos);
      vertexData[base + 1] = read(rPos + 1);
      vertexData[base + 2] = read(rPos + 2);
      const rPrev = vertexRows[rbase + 1]! * rowDim;
      vertexData[base + 3] = read(rPrev);
      vertexData[base + 4] = read(rPrev + 1);
      vertexData[base + 5] = read(rPrev + 2);
      const rNext = vertexRows[rbase + 2]! * rowDim;
      vertexData[base + 6] = read(rNext);
      vertexData[base + 7] = read(rNext + 1);
      vertexData[base + 8] = read(rNext + 2);
      const rG0 = vertexRows[rbase + 3]! * rowDim;
      vertexData[base + 9] = read(rG0);
      vertexData[base + 10] = read(rG0 + 1);
      vertexData[base + 11] = read(rG0 + 2);
      const rG1 = vertexRows[rbase + 4]! * rowDim;
      vertexData[base + 12] = read(rG1);
      vertexData[base + 13] = read(rG1 + 1);
      vertexData[base + 14] = read(rG1 + 2);
      vertexData[base + 25] = n0[0] ?? 0;
      vertexData[base + 26] = n0[1] ?? 0;
      vertexData[base + 27] = n0[2] ?? 0;
      vertexData[base + 28] = n1[0] ?? 0;
      vertexData[base + 29] = n1[1] ?? 0;
      vertexData[base + 30] = n1[2] ?? 0;
    }
    record.points = vm.points;
    record.pointsVersion = vm.points_version;
    record.model = null;
    record.forceRebuild = false;
    const offset = record.vertexOffset;
    const count = record.vertexCount * FLOATS_PER_VERTEX;
    // No heap copy here: `record.vertexData` is a view into the heap (see
    // `alloc_vertices`), so the rewrite above already landed in the upload
    // bytes; only the dirty range remains to be marked.
    if (offset < this.pendingVertexMin) this.pendingVertexMin = offset;
    if (offset + count > this.pendingVertexMax) this.pendingVertexMax = offset + count;
  }

  private build_record(
    vm: VMobject,
    camera: ThreeDCamera,
    baseColors: MobjectBaseColors,
    fillSource: unknown,
    strokeSource: unknown,
    bgStrokeSource: unknown,
    zoom: number,
    zoomBucket: number,
    lineMultiple: number,
  ): GeometryRecord | null {
    const pts = _cContiguousRows(vm.points);
    if (pts === null || pts.rowDim < 3 || pts.rowCount < 4) return null;
    const fillInfo = baseColors.fill;
    const strokeInfo = baseColors.stroke;
    const bgStrokeInfo = baseColors.bgStroke;
    const strokeWidth = vm.get_stroke_width(false);
    const bgStrokeWidth = vm.get_stroke_width(true);
    const drawFill = fillInfo !== null && !isColorTransparent(fillInfo);
    const drawStroke = strokeWidth > 0 && strokeInfo !== null && !isColorTransparent(strokeInfo);
    const drawBgStroke = bgStrokeWidth > 0 && bgStrokeInfo !== null && !isColorTransparent(bgStrokeInfo);
    const basePoints = new Float32Array(pts.rowCount * 3);
    for (let index = 0; index < pts.rowCount; index += 1) {
      const base = index * pts.rowDim;
      basePoints[index * 3] = Number(pts.data[base] ?? 0);
      basePoints[index * 3 + 1] = Number(pts.data[base + 1] ?? 0);
      basePoints[index * 3 + 2] = Number(pts.data[base + 2] ?? 0);
    }
    const base = {
      points: vm.points,
      pointsVersion: vm.points_version,
      fillSource,
      strokeSource,
      bgStrokeSource,
      baseColors,
      strokeWidthRaw: vm.stroke_width,
      bgStrokeWidthRaw: vm.background_stroke_width,
      jointType: vm.joint_type,
      capStyle: vm.cap_style,
      lineMultiple,
      zoomBucket,
      basePoints,
      model: null,
      modelSlot: 0,
      forceRebuild: false,
    };
    if (!drawFill && !drawStroke && !drawBgStroke) {
      // Nothing visible: an empty record still lets the mobject be "handled".
      return {
        ...base,
        vertexData: new Float32Array(0),
        indexData: new Uint32Array(0),
        vertexCount: 0,
        vertexOffset: 0,
        vertexRows: null,
        cornerStartRow: 0,
        cornerEndRow: 0,
        rangeList: [],
        rowDim: 3,
      };
    }
    // Flattening tolerance: a quarter device pixel at the current zoom.
    const tolerance = Math.max(camera.frame_height / camera.pixel_height, 1e-9) * 0.25 / Math.max(1, zoom);
    const ranges = computeSubpathRangesFlat(
      pts.data,
      pts.rowDim,
      pts.rowCount,
      vm.n_points_per_cubic_curve,
      vm.tolerance_for_point_equality,
      2,
    );
    if (ranges.length === 0) return null;
    const subpaths: Array<{ poly: number[][]; rows: number[] }> = [];
    for (const [start, end] of ranges) {
      const flat = this.flatten_subpath(pts.data, pts.rowDim, start!, end!, tolerance);
      if (flat.poly.length >= 2) subpaths.push(flat);
    }
    if (subpaths.length === 0) return null;
    if (drawFill && subpaths.length > 1) return null;

    const fillIndices: number[] = [];
    let fillPolygon: number[][] | null = null;
    let fillPolyRows: number[] | null = null;
    if (drawFill) {
      const { poly, rows } = subpaths[0]!;
      const closed = pointsEqual(poly[0]!, poly[poly.length - 1]!, vm.tolerance_for_point_equality);
      const body = closed ? poly.slice(0, -1) : poly;
      const bodyRows = closed ? rows.slice(0, -1) : rows;
      if (body.length >= 3) {
        const tris = triangulatePolygon(body);
        if (tris.length > 0) {
          fillPolygon = body;
          fillPolyRows = bodyRows;
          fillIndices.push(...tris);
        }
      }
    }

    // Device-space stroke half-widths (see `apply_stroke`: width *
    // cairo_line_width_multiple, then the world->device x scale).
    const deviceScale = camera.pixel_width / camera.frame_width;
    const halfWidth = (strokeWidth * lineMultiple * deviceScale) / 2;
    const bgHalfWidth = (bgStrokeWidth * lineMultiple * deviceScale) / 2;

    // Preallocate the exact vertex/index storage: animations rebuild every
    // face each frame, and growing JS number arrays then copying dominated the
    // rebuild.
    let strokeSegments = 0;
    for (const { poly } of subpaths) {
      const n = poly.length;
      if (n < 2) continue;
      const closed = pointsEqual(poly[0]!, poly[n - 1]!, 1e-6);
      const count = closed ? n - 1 : n;
      strokeSegments += closed ? count : count - 1;
    }
    const strokeVerts = strokeSegments * 4;
    const fillVerts = fillPolygon !== null ? fillPolygon.length : 0;
    const totalVerts =
      (drawBgStroke ? strokeVerts : 0) +
      (drawFill && fillPolygon !== null ? fillVerts : 0) +
      (drawStroke ? strokeVerts : 0);
    const totalIndices =
      (drawBgStroke ? strokeSegments * 6 : 0) +
      (fillPolygon !== null ? fillIndices.length : 0) +
      (drawStroke ? strokeSegments * 6 : 0);
    const vertexData = new Float32Array(totalVerts * FLOATS_PER_VERTEX);
    const indexData = new Uint32Array(totalIndices);
    let vertexCursor = 0;
    let indexCursor = 0;

    // Per-vertex source rows (pos, prev, next, grad0, grad1) for
    // topology-stable position updates.  Any -1 (a subdivided midpoint) marks
    // the whole record as rebuild-only.
    const vertexRows = new Int32Array(totalVerts * 5);
    let subdivided = false;
    const addVertex = (
      p: number[],
      prev: number[],
      next: number[],
      side: number,
      vertexHalfWidth: number,
      info: BaseColorInfo | null,
      pRow = -1,
      prevRow = -1,
      nextRow = -1,
      g0Row = -1,
      g1Row = -1,
    ): number => {
      const index = vertexCursor;
      const offset = index * FLOATS_PER_VERTEX;
      const rowBase = index * 5;
      vertexRows[rowBase] = pRow;
      vertexRows[rowBase + 1] = prevRow;
      vertexRows[rowBase + 2] = nextRow;
      vertexRows[rowBase + 3] = g0Row;
      vertexRows[rowBase + 4] = g1Row;
      if (pRow < 0 || prevRow < 0 || nextRow < 0 || g0Row < 0 || g1Row < 0) subdivided = true;
      const c0 = info?.colors[0] ?? ZERO_RGBA;
      const c1 = info?.colors[1] ?? c0;
      const n0 = info?.normal0 ?? ZERO_VEC;
      const n1 = info?.normal1 ?? ZERO_VEC;
      vertexData[offset] = p[0] ?? 0;
      vertexData[offset + 1] = p[1] ?? 0;
      vertexData[offset + 2] = p[2] ?? 0;
      vertexData[offset + 3] = prev[0] ?? 0;
      vertexData[offset + 4] = prev[1] ?? 0;
      vertexData[offset + 5] = prev[2] ?? 0;
      vertexData[offset + 6] = next[0] ?? 0;
      vertexData[offset + 7] = next[1] ?? 0;
      vertexData[offset + 8] = next[2] ?? 0;
      vertexData[offset + 9] = info?.start[0] ?? 0;
      vertexData[offset + 10] = info?.start[1] ?? 0;
      vertexData[offset + 11] = info?.start[2] ?? 0;
      vertexData[offset + 12] = info?.end[0] ?? 0;
      vertexData[offset + 13] = info?.end[1] ?? 0;
      vertexData[offset + 14] = info?.end[2] ?? 0;
      vertexData[offset + 15] = side;
      vertexData[offset + 16] = vertexHalfWidth;
      vertexData[offset + 17] = c0[0] ?? 0;
      vertexData[offset + 18] = c0[1] ?? 0;
      vertexData[offset + 19] = c0[2] ?? 0;
      vertexData[offset + 20] = c0[3] ?? 0;
      vertexData[offset + 21] = c1[0] ?? 0;
      vertexData[offset + 22] = c1[1] ?? 0;
      vertexData[offset + 23] = c1[2] ?? 0;
      vertexData[offset + 24] = c1[3] ?? 0;
      vertexData[offset + 25] = n0[0] ?? 0;
      vertexData[offset + 26] = n0[1] ?? 0;
      vertexData[offset + 27] = n0[2] ?? 0;
      vertexData[offset + 28] = n1[0] ?? 0;
      vertexData[offset + 29] = n1[1] ?? 0;
      vertexData[offset + 30] = n1[2] ?? 0;
      vertexData[offset + 31] = 0;
      vertexCursor += 1;
      return index;
    };

    // Cairo order per mobject: background stroke, fill, foreground stroke.
    const startRow = 0;
    const endRow = Math.floor((pts.rowCount - 1) / 6) * 3;
    if (drawBgStroke) {
      indexCursor = this.append_stroke_geometry(subpaths, addVertex, indexData, indexCursor, bgStrokeInfo, bgHalfWidth, startRow, endRow);
    }
    if (drawFill && fillPolygon !== null && fillPolyRows !== null && fillIndices.length > 0) {
      const vertexBase = vertexCursor;
      for (let k = 0; k < fillPolygon.length; k += 1) {
        const row = fillPolyRows[k]!;
        addVertex(fillPolygon[k]!, fillPolygon[k]!, fillPolygon[k]!, 0, 0, fillInfo, row, row, row, startRow, endRow);
      }
      for (const tri of fillIndices) indexData[indexCursor++] = vertexBase + tri;
    }
    if (drawStroke) {
      indexCursor = this.append_stroke_geometry(subpaths, addVertex, indexData, indexCursor, strokeInfo, halfWidth, startRow, endRow);
    }

    return {
      ...base,
      vertexData,
      indexData,
      vertexCount: vertexCursor,
      vertexOffset: 0,
      vertexRows: subdivided ? null : vertexRows,
      cornerStartRow: startRow,
      cornerEndRow: endRow,
      rangeList: ranges.map((r) => [r[0] ?? 0, r[1] ?? 0] as const),
      rowDim: pts.rowDim,
    };
  }

  private append_stroke_geometry(
    subpaths: Array<{ poly: number[][]; rows: number[] }>,
    addVertex: (
      p: number[],
      prev: number[],
      next: number[],
      side: number,
      halfWidth: number,
      info: BaseColorInfo | null,
      pRow?: number,
      prevRow?: number,
      nextRow?: number,
      g0Row?: number,
      g1Row?: number,
    ) => number,
    indexData: Uint32Array,
    indexCursor: number,
    info: BaseColorInfo | null,
    halfWidth: number,
    startRow: number,
    endRow: number,
  ): number {
    let cursor = indexCursor;
    for (const { poly, rows } of subpaths) {
      const n = poly.length;
      if (n < 2) continue;
      const closed = pointsEqual(poly[0]!, poly[n - 1]!, 1e-6);
      const count = closed ? n - 1 : n;
      const pointAt = (index: number): number[] => {
        if (index < 0) return closed ? poly[count - 1]! : poly[0]!;
        if (index >= count) return closed ? poly[index - count]! : poly[count - 1]!;
        return poly[index]!;
      };
      const rowAt = (index: number): number => {
        if (index < 0) return closed ? rows[count - 1]! : rows[0]!;
        if (index >= count) return closed ? rows[index - count]! : rows[count - 1]!;
        return rows[index]!;
      };
      // A closed subpath has `count` segments (the last one wraps back to the
      // first point); an open one has `count - 1`.
      const segments = closed ? count : count - 1;
      for (let i = 0; i < segments; i += 1) {
        const p0 = pointAt(i);
        const p1 = pointAt(i + 1);
        const prev0 = pointAt(i - 1);
        const next0 = pointAt(i + 1);
        const prev1 = pointAt(i);
        const next1 = pointAt(i + 2);
        const r0 = rowAt(i);
        const r1 = rowAt(i + 1);
        const rp0 = rowAt(i - 1);
        const rn0 = rowAt(i + 1);
        const rp1 = rowAt(i);
        const rn1 = rowAt(i + 2);
        const a = addVertex(p0, prev0, next0, 1, halfWidth, info, r0, rp0, rn0, startRow, endRow);
        const b = addVertex(p0, prev0, next0, -1, halfWidth, info, r0, rp0, rn0, startRow, endRow);
        const c = addVertex(p1, prev1, next1, 1, halfWidth, info, r1, rp1, rn1, startRow, endRow);
        const d = addVertex(p1, prev1, next1, -1, halfWidth, info, r1, rp1, rn1, startRow, endRow);
        indexData[cursor++] = a;
        indexData[cursor++] = b;
        indexData[cursor++] = c;
        indexData[cursor++] = b;
        indexData[cursor++] = d;
        indexData[cursor++] = c;
      }
    }
    return cursor;
  }

  private flatten_subpath(
    data: ArrayLike<number>,
    rowDim: number,
    start: number,
    end: number,
    tolerance: number,
  ): { poly: number[][]; rows: number[] } {
    const pointAt = (index: number): number[] => {
      const base = index * rowDim;
      return [Number(data[base] ?? 0), Number(data[base + 1] ?? 0), Number(data[base + 2] ?? 0)];
    };
    const poly: number[][] = [];
    const rows: number[] = [];
    if (end - start < 4) {
      for (let i = start; i < end; i += 1) {
        poly.push(pointAt(i));
        rows.push(i);
      }
      return { poly, rows };
    }
    poly.push(pointAt(start));
    rows.push(start);
    for (let i = start; i + 3 < end; i += 4) {
      flattenCubic(
        poly,
        rows,
        pointAt(i),
        pointAt(i + 1),
        pointAt(i + 2),
        pointAt(i + 3),
        i + 3,
        tolerance,
      );
    }
    return { poly, rows };
  }
}

/** Shared fallback rgba source for VMobjects without an explicit array. */
const ZERO_RGBA_SOURCE = array([[0, 0, 0, 0]]);
