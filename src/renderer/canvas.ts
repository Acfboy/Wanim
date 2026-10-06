/**
 * Canvas2D rendering backend.
 *
 * Faithful port of the rasterization half of the original manim renderer
 * (`the original manim renderer`): the serialized
 * per-frame display list is turned back into VMobjects and drawn with the
 * browser Canvas2D API instead of cairo.
 *
 * The cairo camera draws in manim world coordinates through a user->device
 * matrix (see `Camera.get_cairo_context`):
 *
 *   x' =  (pw / fw) * x + (pw / 2) - fc_x * (pw / fw)
 *   y' = -(ph / fh) * y + (ph / 2) + fc_y * (ph / fh)
 *
 * Canvas2D has the same (y-down, top-left origin) device space, so the matrix
 * maps 1:1 onto `ctx.setTransform`.  `ctx.bezierCurveTo` consumes cubic beziers
 * directly, so `VMobject.points` (groups of 4 control points) can be replayed
 * without flattening.
 *
 * The Python renderer builds a cairo `LinearGradient` whenever a colour array
 * holds more than one row.  wanim (like manim) expands one row per point, so
 * that branch is always taken in Python.  We keep the same rule but add a fast
 * path: if every row is identical, a flat colour is used (visually identical,
 * much cheaper).  Gradients are only built for genuinely multi-colour arrays.
 */
import { array, type NDArray } from "numpy-ts";
import { Camera } from "../camera/camera.js";
import { ThreeDCamera } from "../camera/three_d_camera.js";
import { CapStyleType, LineJointType } from "../constants.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { _cContiguousRows, isNDArray, pointRows } from "../utils/arrays.js";

/** One serialized VMobject record (see `renderer/json_renderer.ts`). */
export interface MJSONMobject {
  p: number[][];
  f?: number[][] | null;
  s?: number[][] | null;
  bs?: number[][] | null;
  sw: number;
  bsw: number;
  jt: number;
  cs: number;
  z: number;
  bi?: string | null;
  /** 3D scenes only: `shade_in_3d` membership. */
  s3d?: boolean;
  /** 3D scenes only: fixed-in-frame membership. */
  fif?: boolean;
  /** 3D scenes only: evaluated fixed-orientation center. */
  fo?: number[] | null;
  [key: string]: unknown;
}

/** Serialized camera geometry for one frame. */
export interface MJSONCamera {
  fc: number[];
  fw: number;
  fh: number;
  /** 3D scenes only: marks the frame as rendered through a `ThreeDCamera`. */
  "3d"?: boolean;
  /** 3D scenes only: focal distance, euler angles, zoom, light source, shading. */
  fd?: number;
  phi?: number;
  theta?: number;
  gamma?: number;
  zoom?: number;
  ls?: number[];
  sh?: number;
  sa?: boolean;
  ep?: boolean;
  [key: string]: unknown;
}

/** One frame record: camera + display list + number of identical repeats. */
export interface MJSONFrame {
  cam: MJSONCamera;
  m: MJSONMobject[];
  repeat?: number;
}

export interface MJSONMeta {
  width: number;
  height: number;
  frame_rate: number;
  background_color: string;
  background_opacity: number;
  num_frames: number;
  num_plays: number;
  [key: string]: unknown;
}

export interface MJSONDocument {
  meta: MJSONMeta;
  frames: MJSONFrame[];
}

/**
 * The subset of `Camera` needed to draw.  A real `Camera` satisfies it, as
 * does the lightweight `PreviewCamera` used by the serialized-file player.
 */
export interface CanvasCamera {
  pixel_width: number;
  pixel_height: number;
  frame_width: number;
  frame_height: number;
  frame_center: readonly number[];
  cairo_line_width_multiple: number;
  transform_points_pre_display(mobject: VMobject, points: NDArray): NDArray;
  /** Optional flat projection fast path (ThreeDCamera); returns projected
   *  (x, y) pairs in camera space or `null` to fall back. */
  transform_points_pre_display_flat?(
    mobject: VMobject,
    points: unknown,
  ): { data: Float64Array; rowDim: 2; rowCount: number } | null;
  /** Optional flat single-point projection (gradient endpoints). */
  project_point_flat?(point: readonly number[]): [number, number];
  /** Optional resolved colour stops, avoiding the rgba NDArray round trip. */
  get_draw_color_info?(
    vmobject: VMobject,
    source: NDArray,
  ): { colors: number[][]; gradient: boolean; start: number[]; end: number[] } | null;
  /** Camera-applied colours (ThreeDCamera shades `shade_in_3d` faces). */
  get_fill_rgbas?(vmobject: VMobject): NDArray;
  get_stroke_rgbas?(vmobject: VMobject, background?: boolean): NDArray;
}

const CANVAS_LINE_JOIN: Record<string, CanvasLineJoin> = {
  ROUND: "round",
  BEVEL: "bevel",
  MITER: "miter",
};

const CANVAS_LINE_CAP: Record<string, CanvasLineCap> = {
  ROUND: "round",
  BUTT: "butt",
  SQUARE: "square",
};

/** Rebuild a VMobject holding exactly the serialized bezier data. */
export function deserialize_vmobject(record: MJSONMobject): VMobject {
  const vmobject = new VMobject();
  const points = record.p ?? [];
  vmobject.set_points(points.length > 0 ? array(points as unknown[]) : array([]));
  for (const [key, attr] of [
    ["f", "fill_rgbas"],
    ["s", "stroke_rgbas"],
    ["bs", "background_stroke_rgbas"],
  ] as const) {
    const value = record[key];
    if (value !== null && value !== undefined) {
      (vmobject as unknown as Record<string, unknown>)[attr] = array(value as unknown[]);
    }
  }
  vmobject.stroke_width = Number(record.sw);
  vmobject.background_stroke_width = Number(record.bsw);
  vmobject.joint_type = jointTypeName(Number(record.jt));
  vmobject.cap_style = capStyleName(Number(record.cs));
  vmobject.z_index = Number(record.z);
  if (record.bi !== null && record.bi !== undefined) {
    vmobject.background_image = record.bi;
  }
  return vmobject;
}

function jointTypeName(value: number): string {
  switch (value) {
    case LineJointType.ROUND: return "ROUND";
    case LineJointType.BEVEL: return "BEVEL";
    case LineJointType.MITER: return "MITER";
    default: return "AUTO";
  }
}

function capStyleName(value: number): string {
  switch (value) {
    case CapStyleType.ROUND: return "ROUND";
    case CapStyleType.BUTT: return "BUTT";
    case CapStyleType.SQUARE: return "SQUARE";
    default: return "AUTO";
  }
}

/** Last style written to each context, so the fill/stroke hot paths do not
 * re-assign (and re-parse) an unchanged CSS colour for every mobject. */
interface CanvasStyleState {
  fill: string | CanvasGradient | null;
  stroke: string | CanvasGradient | null;
}
const CANVAS_STYLE_STATE = new WeakMap<CanvasRenderingContext2D, CanvasStyleState>();

function styleState(ctx: CanvasRenderingContext2D): CanvasStyleState {
  let state = CANVAS_STYLE_STATE.get(ctx);
  if (state === undefined) {
    state = { fill: null, stroke: null };
    CANVAS_STYLE_STATE.set(ctx, state);
  }
  return state;
}

/** Fill the canvas with a colour (no pixel array involved). */
export function clearCanvas(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  backgroundCss: string,
): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = backgroundCss;
  const state = styleState(ctx);
  state.fill = backgroundCss;
  // Reusing one canvas for a new run (`canvas.width = ...` resets the context
  // to its default black styles) or an embedder assigning styles directly
  // invalidates the cached stroke: drop it so the next stroke is written even
  // when its colour matches the previous run's last one.  Without this, a
  // second run of a scene drew blue strokes with the context's default black
  // and the shapes vanished on the black background.
  state.stroke = null;
  ctx.fillRect(0, 0, camera.pixel_width, camera.pixel_height);
}

/** Set the world->device transform mirroring cairo's camera matrix. */
export function setCanvasTransform(ctx: CanvasRenderingContext2D, camera: CanvasCamera): void {
  const pw = camera.pixel_width;
  const ph = camera.pixel_height;
  const fw = camera.frame_width;
  const fh = camera.frame_height;
  const fc = pointRows(camera.frame_center, 3)[0] ?? [0, 0, 0];
  const widthMult = pw / fw;
  const heightMult = ph / fh;
  ctx.setTransform(
    widthMult,
    0,
    0,
    -heightMult,
    pw / 2 - (fc[0] ?? 0) * widthMult,
    ph / 2 + (fc[1] ?? 0) * heightMult,
  );
}

/**
 * Compute `[start, end)` index ranges of the cubic subpaths in `points`,
 * mirroring `VMobject.get_subpath_split_indices_from_points(points, n_dims=2)`.
 * Implemented locally to avoid NDArray round-trips in the hot draw loop.
 */
export function computeSubpathRanges(
  points: number[][],
  nppcc: number,
  tolerance: number,
  nDims = 2,
): number[][] {
  const nPts = points.length;
  if (nPts < nppcc) return [];
  const boundaries: number[] = [];
  for (let index = nppcc; index < nPts; index += nppcc) boundaries.push(index);
  if (boundaries.length === 0) return [[0, nPts]];
  const rtol = 1.0e-5;
  const atol = tolerance;
  const splitPoints: number[] = [0];
  for (const boundary of boundaries) {
    const end = points[boundary - 1] ?? [];
    const start = points[boundary] ?? [];
    let split = false;
    for (let dim = 0; dim < nDims; dim += 1) {
      const e = end[dim] ?? NaN;
      const s = start[dim] ?? NaN;
      const diff = Math.abs(e - s);
      const close = (diff <= atol + rtol * Math.abs(s) && Number.isFinite(diff)) || e === s;
      if (!close) {
        split = true;
        break;
      }
    }
    if (split) splitPoints.push(boundary);
  }
  splitPoints.push(nPts);
  const ranges: number[][] = [];
  for (let index = 0; index < splitPoints.length - 1; index += 1) {
    ranges.push([splitPoints[index] ?? 0, splitPoints[index + 1] ?? 0]);
  }
  return ranges;
}

/** Read a points NDArray as rows, avoiding `pointRows`' nested-array pass when
 *  the data is a plain C-contiguous matrix (the common case). */
function pointsToRows(value: NDArray): number[][] {
  const arr = value as NDArray & { data?: ArrayLike<number>; flags?: { C_CONTIGUOUS?: boolean } };
  const shape = arr.shape;
  const data = arr.data;
  const contiguous = arr.flags === undefined || arr.flags.C_CONTIGUOUS !== false;
  if (data !== undefined && contiguous && shape.length === 2) {
    const count = shape[0] ?? 0;
    const dim = shape[1] ?? 0;
    const rows = new Array<number[]>(count);
    for (let index = 0; index < count; index += 1) {
      const base = index * dim;
      const row = new Array<number>(dim);
      for (let c = 0; c < dim; c += 1) row[c] = Number(data[base + c] ?? 0);
      rows[index] = row;
    }
    return rows;
  }
  return pointRows(value, 3);
}

/** Flat-range counterpart of {@link computeSubpathRanges}: walks the backing
 * store directly so the draw path never materialises point rows. */
function computeSubpathRangesFlat(
  data: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
  nppcc: number,
  tolerance: number,
  nDims = 2,
): number[][] {
  if (rowCount < nppcc) return [];
  const boundaries: number[] = [];
  for (let index = nppcc; index < rowCount; index += nppcc) boundaries.push(index);
  if (boundaries.length === 0) return [[0, rowCount]];
  const rtol = 1.0e-5;
  const atol = tolerance;
  const splitPoints: number[] = [0];
  for (const boundary of boundaries) {
    const endBase = (boundary - 1) * rowDim;
    const startBase = boundary * rowDim;
    let split = false;
    for (let dim = 0; dim < nDims; dim += 1) {
      const e = data[endBase + dim] ?? NaN;
      const s = data[startBase + dim] ?? NaN;
      const diff = Math.abs((e as number) - (s as number));
      const close = (diff <= atol + rtol * Math.abs(s as number) && Number.isFinite(diff)) || e === s;
      if (!close) {
        split = true;
        break;
      }
    }
    if (split) splitPoints.push(boundary);
  }
  splitPoints.push(rowCount);
  const ranges: number[][] = [];
  for (let index = 0; index < splitPoints.length - 1; index += 1) {
    ranges.push([splitPoints[index] ?? 0, splitPoints[index + 1] ?? 0]);
  }
  return ranges;
}

/** Set `ctx`'s current path from a VMobject's points, mirroring `set_cairo_context_path`. */
export function setCanvasPath(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject): void {
  const projectedFlat = camera.transform_points_pre_display_flat?.(vmobject, vmobject.points) ?? null;
  const nppcc = vmobject.n_points_per_cubic_curve;
  const transformed = projectedFlat === null
    ? camera.transform_points_pre_display(vmobject, vmobject.points)
    : null;
  const flat = projectedFlat ?? _cContiguousRows(transformed);
  if (flat !== null && flat.rowDim >= 2) {
    const { data, rowDim, rowCount } = flat;
    if (rowCount === 0) return;
    const ranges = computeSubpathRangesFlat(data, rowDim, rowCount, nppcc, vmobject.tolerance_for_point_equality, 2);
    if (ranges.length === 0) return;
    ctx.beginPath();
    for (const [start, end] of ranges) {
      if (end - start < nppcc) continue;
      ctx.moveTo(data[start * rowDim] ?? 0, data[start * rowDim + 1] ?? 0);
      for (let i = start; i <= end - nppcc; i += nppcc) {
        const h1 = (i + 1) * rowDim;
        const h2 = (i + 2) * rowDim;
        const anchor = (i + 3) * rowDim;
        ctx.bezierCurveTo(
          data[h1] ?? 0, data[h1 + 1] ?? 0,
          data[h2] ?? 0, data[h2 + 1] ?? 0,
          data[anchor] ?? 0, data[anchor + 1] ?? 0,
        );
      }
      if (
        vmobject.consider_points_equals_2d(
          [data[start * rowDim] ?? 0, data[start * rowDim + 1] ?? 0],
          [data[(end - 1) * rowDim] ?? 0, data[(end - 1) * rowDim + 1] ?? 0],
        )
      ) {
        ctx.closePath();
      }
    }
    return;
  }
  const points = pointsToRows(transformed as NDArray);
  if (points.length === 0) return;
  const ranges = computeSubpathRanges(points, nppcc, vmobject.tolerance_for_point_equality, 2);
  if (ranges.length === 0) return;
  ctx.beginPath();
  for (const [startIdx, endIdx] of ranges) {
    if ((endIdx as number) - (startIdx as number) < nppcc) continue;
    const start = startIdx as number;
    const end = endIdx as number;
    ctx.moveTo(points[start]?.[0] ?? 0, points[start]?.[1] ?? 0);
    for (let i = start; i <= end - nppcc; i += nppcc) {
      const h1 = points[i + 1] ?? [0, 0];
      const h2 = points[i + 2] ?? [0, 0];
      const anchor = points[i + 3] ?? [0, 0];
      ctx.bezierCurveTo(
        h1[0] ?? 0, h1[1] ?? 0,
        h2[0] ?? 0, h2[1] ?? 0,
        anchor[0] ?? 0, anchor[1] ?? 0,
      );
    }
    if (vmobject.consider_points_equals_2d(points[start] ?? [], points[end - 1] ?? [])) {
      ctx.closePath();
    }
  }
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

/** Convert a normalized `[r, g, b, a]` row to a CSS colour string. */
export function rgbaToCss(rgba: readonly number[]): string {
  const r = Math.round(clamp01(rgba[0] ?? 0) * 255);
  const g = Math.round(clamp01(rgba[1] ?? 0) * 255);
  const b = Math.round(clamp01(rgba[2] ?? 0) * 255);
  const a = clamp01(rgba[3] ?? 0);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function allRowsEqual(rgbas: number[][]): boolean {
  if (rgbas.length <= 1) return true;
  const first = rgbas[0] ?? [];
  for (let index = 1; index < rgbas.length; index += 1) {
    const row = rgbas[index] ?? [];
    if (row[0] !== first[0] || row[1] !== first[1] || row[2] !== first[2] || row[3] !== first[3]) {
      return false;
    }
  }
  return true;
}

/** Resolved `rgba()` strings, keyed by packed colour channel bytes.  Fill and
 * stroke colours repeat heavily across mobjects and frames; building the same
 * CSS string thousands of times per second was measurable.  Only 0/255 alpha
 * values are cached (fractional alpha is rare and would need a float key). */
const RGBA_CSS_CACHE = new Map<number, string>();
const RGBA_CSS_CACHE_LIMIT = 512;

function rgbaValuesToCss(r: number, g: number, b: number, a: number): string {
  const ri = Math.round(clamp01(r) * 255);
  const gi = Math.round(clamp01(g) * 255);
  const bi = Math.round(clamp01(b) * 255);
  if (a === 0 || a === 1) {
    const key = ((ri << 24) | (gi << 16) | (bi << 8) | (a === 0 ? 0 : 1)) >>> 0;
    let css = RGBA_CSS_CACHE.get(key);
    if (css === undefined) {
      css = `rgba(${ri}, ${gi}, ${bi}, ${a})`;
      if (RGBA_CSS_CACHE.size >= RGBA_CSS_CACHE_LIMIT) RGBA_CSS_CACHE.clear();
      RGBA_CSS_CACHE.set(key, css);
    }
    return css;
  }
  return `rgba(${ri}, ${gi}, ${bi}, ${clamp01(a)})`;
}

/** Uniform-colour fast path over a flat rgba store; `null` when the rows
 * differ (a real gradient). */
function uniformFlatColor(
  data: ArrayLike<number>,
  rowDim: number,
  rowCount: number,
): string | null {
  if (rowCount === 0 || rowDim !== 4) return null;
  const r = Number(data[0] ?? 0);
  const g = Number(data[1] ?? 0);
  const b = Number(data[2] ?? 0);
  const a = Number(data[3] ?? 0);
  const length = rowCount * rowDim;
  for (let index = rowDim; index < length; index += rowDim) {
    if (
      Number(data[index]) !== r ||
      Number(data[index + 1]) !== g ||
      Number(data[index + 2]) !== b ||
      Number(data[index + 3]) !== a
    ) {
      return null;
    }
  }
  return rgbaValuesToCss(r, g, b, a);
}

function resolveColorStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  source: NDArray,
  vmobject: VMobject,
): string | CanvasGradient {
  // Fast path: a C-contiguous rgba store is scanned in place.  `pointRows`
  // materialised one JS array per point just to compare four channels (this
  // ran for every fill and every stroke of every mobject, every frame).
  const flat = _cContiguousRows(source);
  if (flat !== null) {
    const uniform = uniformFlatColor(flat.data, flat.rowDim, flat.rowCount);
    if (uniform !== null) return uniform;
    return gradientColorStyle(ctx, camera, pointRows(source, 4), vmobject);
  }
  return colorStyle(ctx, camera, pointRows(source, 4), vmobject);
}

function gradientColorStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  rgbas: number[][],
  vmobject: VMobject,
): string | CanvasGradient {
  if (rgbas.length === 0) return "rgba(0, 0, 0, 0)";
  const [startPoint, endPoint] = vmobject.get_gradient_start_and_end_points();
  const transformed = pointRows(
    camera.transform_points_pre_display(vmobject, array([startPoint, endPoint]) as NDArray),
    3,
  );
  const p0 = transformed[0] ?? [0, 0];
  const p1 = transformed[1] ?? [0, 0];
  const gradient = ctx.createLinearGradient(p0[0] ?? 0, p0[1] ?? 0, p1[0] ?? 0, p1[1] ?? 0);
  const last = rgbas.length - 1;
  for (let index = 0; index < rgbas.length; index += 1) {
    const offset = last === 0 ? 0 : index / last;
    gradient.addColorStop(offset, rgbaToCss(rgbas[index] ?? []));
  }
  return gradient;
}

/**
 * Resolve a colour array to a Canvas style, mirroring `set_cairo_context_color`.
 * A single (or uniform) colour yields a flat `rgba()` string; otherwise a
 * `CanvasGradient` is built from the VMobject's gradient endpoints.
 */
export function colorStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  rgbas: number[][] | NDArray,
  vmobject: VMobject,
): string | CanvasGradient {
  const flat = isNDArray(rgbas) ? _cContiguousRows(rgbas) : null;
  if (flat !== null) {
    const uniform = uniformFlatColor(flat.data, flat.rowDim, flat.rowCount);
    if (uniform !== null) return uniform;
  }
  const rows = isNDArray(rgbas) ? pointRows(rgbas, 4) : rgbas;
  if (rows.length === 0) return "rgba(0, 0, 0, 0)";
  if (allRowsEqual(rows)) return rgbaToCss(rows[0]!);
  return gradientColorStyle(ctx, camera, rows, vmobject);
}

/** True when every row of an rgba store has alpha 0 (nothing to fill). */
function flatAlphaIsZero(source: NDArray): boolean {
  const flat = _cContiguousRows(source);
  if (flat === null || flat.rowDim !== 4 || flat.rowCount === 0) return false;
  const { data } = flat;
  const length = flat.rowCount * 4;
  for (let index = 3; index < length; index += 4) {
    if (Number(data[index]) !== 0) return false;
  }
  return true;
}

/** Fill the current path, mirroring `apply_fill`. */
/** Build a Canvas style from a camera-resolved colour description. */
function colorInfoStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  info: { colors: number[][]; gradient: boolean; start: number[]; end: number[] },
): string | CanvasGradient {
  const first = info.colors[0];
  if (first === undefined) return "rgba(0, 0, 0, 0)";
  if (!info.gradient || info.colors.length < 2) return rgbaToCss(first);
  const p0 = camera.project_point_flat !== undefined
    ? camera.project_point_flat(info.start)
    : [info.start[0] ?? 0, info.start[1] ?? 0];
  const p1 = camera.project_point_flat !== undefined
    ? camera.project_point_flat(info.end)
    : [info.end[0] ?? 0, info.end[1] ?? 0];
  const gradient = ctx.createLinearGradient(p0[0] ?? 0, p0[1] ?? 0, p1[0] ?? 0, p1[1] ?? 0);
  gradient.addColorStop(0, rgbaToCss(info.colors[0]!));
  gradient.addColorStop(1, rgbaToCss(info.colors[1]!));
  return gradient;
}

/** Resolve fill/stroke colours: the camera's flat colour info when available,
 *  otherwise the rgba-array path. */
function resolveStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  vmobject: VMobject,
  source: NDArray,
): string | CanvasGradient {
  const info = camera.get_draw_color_info?.(vmobject, source) ?? null;
  if (info !== null) return colorInfoStyle(ctx, camera, info);
  return resolveColorStyle(ctx, camera, source, vmobject);
}

export function applyCanvasFill(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject): void {
  const source = camera.get_fill_rgbas !== undefined
    ? camera.get_fill_rgbas(vmobject)
    : vmobject.get_fill_rgbas();
  // A fully transparent fill is a no-op for the rasterizer; skipping it avoids
  // a style parse + path fill per stroke-only mobject (lines, arrows, ...).
  if (flatAlphaIsZero(source)) return;
  const style = resolveStyle(ctx, camera, vmobject, source);
  const state = styleState(ctx);
  if (state.fill !== style) {
    ctx.fillStyle = style;
    state.fill = style;
  }
  ctx.fill();
}

/** Stroke the current path, mirroring `apply_stroke`. */
export function applyCanvasStroke(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  vmobject: VMobject,
  background = false,
): void {
  const width = vmobject.get_stroke_width(background);
  if (width === 0) return;
  const source = camera.get_stroke_rgbas !== undefined
    ? camera.get_stroke_rgbas(vmobject, background)
    : vmobject.get_stroke_rgbas(background);
  const style = resolveStyle(ctx, camera, vmobject, source);
  const state = styleState(ctx);
  if (state.stroke !== style) {
    ctx.strokeStyle = style;
    state.stroke = style;
  }
  ctx.lineWidth = width * camera.cairo_line_width_multiple;
  const joint = CANVAS_LINE_JOIN[vmobject.joint_type];
  if (joint !== undefined) ctx.lineJoin = joint;
  const cap = CANVAS_LINE_CAP[vmobject.cap_style];
  if (cap !== undefined) ctx.lineCap = cap;
  ctx.stroke();
}

/** Draw one VMobject: background stroke, fill, then foreground stroke. */
export function drawVmobject(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject): void {
  if (vmobject.get_num_points() === 0) return;
  // Fully invisible mobject (no strokes, transparent uniform fill): skip the
  // whole path build.  Only for the plain 2D camera; a 3D camera computes the
  // fill colours from shading.
  if (
    camera.get_fill_rgbas === undefined
    && vmobject.get_stroke_width(true) === 0
    && vmobject.get_stroke_width(false) === 0
    && flatAlphaIsZero(vmobject.get_fill_rgbas())
  ) {
    return;
  }
  setCanvasPath(ctx, camera, vmobject);
  applyCanvasStroke(ctx, camera, vmobject, true);
  applyCanvasFill(ctx, camera, vmobject);
  applyCanvasStroke(ctx, camera, vmobject, false);
}

/** `meta.background_color` + opacity -> a CSS colour for the canvas backdrop. */
export function backgroundCss(meta: Pick<MJSONMeta, "background_color" | "background_opacity">): string {
  const hex = meta.background_color ?? "#000000";
  const match = /^#?([0-9a-fA-F]{6})$/.exec(hex.replace(/^#/, ""));
  if (match === null) return `rgba(0, 0, 0, ${clamp01(meta.background_opacity ?? 1)})`;
  const value = match[1]!;
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${clamp01(meta.background_opacity ?? 1)})`;
}

/** Whether a frame was serialized through a `ThreeDCamera`. */
export function isThreeDFrame(frame: MJSONFrame): boolean {
  return frame.cam["3d"] === true;
}

/**
 * Restore the camera geometry stored in a frame.  For 3D frames this also
 * restores the euler angles / projection state and regenerates the rotation
 * matrix, mirroring `draw_frame` in the original manim renderer.
 */
export function configureFrameCamera(camera: CanvasCamera, frame: MJSONFrame): void {
  const cam = frame.cam;
  camera.frame_center = cam.fc;
  camera.frame_width = cam.fw;
  camera.frame_height = cam.fh;
  if (!isThreeDFrame(frame) || !(camera instanceof ThreeDCamera)) return;
  camera.set_phi(cam.phi ?? 0);
  camera.set_theta(cam.theta ?? 0);
  camera.set_gamma(cam.gamma ?? 0);
  camera.set_focal_distance(cam.fd ?? 20);
  camera.set_zoom(cam.zoom ?? 1);
  if (Array.isArray(cam.ls)) camera.light_source.move_to(cam.ls as readonly number[]);
  camera.shading_factor = cam.sh ?? 0.2;
  camera.should_apply_shading = cam.sa ?? true;
  camera.exponential_projection = cam.ep ?? false;
  camera.reset_rotation_matrix();
}

/**
 * Restore the per-mobject 3D membership flags (`s3d`/`fif`/`fo`) onto a freshly
 * deserialized copy, mirroring the Python renderer's `draw_frame`.
 */
export function applyMobject3DFlags(
  camera: ThreeDCamera,
  vmobject: VMobject,
  record: MJSONMobject,
): void {
  if (record.s3d) (vmobject as { shade_in_3d?: boolean }).shade_in_3d = true;
  if (record.fif) camera.fixed_in_frame_mobjects.add(vmobject);
  if (record.fo !== null && record.fo !== undefined) {
    const center = record.fo;
    camera.fixed_orientation_mobjects.set(vmobject, () => center);
  }
}

/**
 * Draw one serialized frame onto `ctx`.  Restores the camera geometry stored in
 * the frame, clears the canvas and replays the display list.
 */
export function drawFrame(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  frame: MJSONFrame,
  background = "#000000",
): void {
  configureFrameCamera(camera, frame);
  clearCanvas(ctx, camera, background);
  setCanvasTransform(ctx, camera);
  const threeD = isThreeDFrame(frame) && camera instanceof ThreeDCamera;
  if (threeD) {
    camera.fixed_in_frame_mobjects.clear();
    camera.fixed_orientation_mobjects.clear();
  }
  for (const record of frame.m) {
    const vmobject = deserialize_vmobject(record);
    if (threeD) applyMobject3DFlags(camera, vmobject, record);
    drawVmobject(ctx, camera, vmobject);
  }
}

/**
 * Lightweight camera for the serialized-file player.  `Camera`'s constructor
 * builds a full pixel array (up to hundreds of MB for 4K) that the Canvas
 * backend never reads, so this override makes `init_background` a no-op.
 */
export class PreviewCamera extends Camera {
  override init_background(): void {
    this.background = array([]) as NDArray;
  }
}

/** 3D counterpart of {@link PreviewCamera}: a `ThreeDCamera` that never
 *  allocates the (unused) RGBA pixel array. */
export class PreviewThreeDCamera extends ThreeDCamera {
  override init_background(): void {
    this.background = array([]) as NDArray;
  }
}
