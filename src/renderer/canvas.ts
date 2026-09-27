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
import { CapStyleType, LineJointType } from "../constants.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { pointRows } from "../utils/arrays.js";

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
  [key: string]: unknown;
}

/** Serialized camera geometry for one frame. */
export interface MJSONCamera {
  fc: number[];
  fw: number;
  fh: number;
  "3d"?: boolean;
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

/** Fill the canvas with a colour (no pixel array involved). */
export function clearCanvas(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  backgroundCss: string,
): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = backgroundCss;
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

/** Set `ctx`'s current path from a VMobject's points, mirroring `set_cairo_context_path`. */
export function setCanvasPath(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject): void {
  const transformed = camera.transform_points_pre_display(vmobject, vmobject.points);
  const points = pointRows(transformed, 3);
  if (points.length === 0) return;
  const nppcc = vmobject.n_points_per_cubic_curve;
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

/**
 * Resolve a colour array to a Canvas style, mirroring `set_cairo_context_color`.
 * A single (or uniform) colour yields a flat `rgba()` string; otherwise a
 * `CanvasGradient` is built from the VMobject's gradient endpoints.
 */
export function colorStyle(
  ctx: CanvasRenderingContext2D,
  camera: CanvasCamera,
  rgbas: number[][],
  vmobject: VMobject,
): string | CanvasGradient {
  if (rgbas.length === 0) return "rgba(0, 0, 0, 0)";
  if (allRowsEqual(rgbas)) return rgbaToCss(rgbas[0]!);
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

/** Fill the current path, mirroring `apply_fill`. */
export function applyCanvasFill(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject): void {
  const rgbas = pointRows(vmobject.get_fill_rgbas(), 4);
  ctx.fillStyle = colorStyle(ctx, camera, rgbas, vmobject) as string | CanvasGradient;
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
  const rgbas = pointRows(vmobject.get_stroke_rgbas(background), 4);
  ctx.strokeStyle = colorStyle(ctx, camera, rgbas, vmobject) as string | CanvasGradient;
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
  const cam = frame.cam;
  camera.frame_center = cam.fc;
  camera.frame_width = cam.fw;
  camera.frame_height = cam.fh;
  clearCanvas(ctx, camera, background);
  setCanvasTransform(ctx, camera);
  for (const record of frame.m) {
    drawVmobject(ctx, camera, deserialize_vmobject(record));
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
