import { array, identity, matmul, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { BLACK, get_shaded_rgb } from "../utils/color/core.js";
import { DEGREES, ORIGIN, RIGHT } from "../constants.js";
import { _cContiguousRows, isNDArray, pointRows } from "../utils/arrays.js";
import { extract_mobject_family_members } from "../utils/family.js";
import { mobject_geometry_epoch } from "../utils/render_epoch.js";
import { rotation_about_z, rotation_matrix } from "../utils/space_ops.js";
import { Mobject } from "../mobject/mobject.js";
import { Point } from "../mobject/types/point_cloud_mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { ValueTracker } from "../mobject/value_tracker.js";
import {
  get_3d_vmob_end_corner,
  get_3d_vmob_end_corner_unit_normal,
  get_3d_vmob_start_corner,
  get_3d_vmob_start_corner_unit_normal,
} from "../mobject/three_d/three_d_utils.js";
import { Camera, type CameraOptions, type CaptureOptions } from "./camera.js";

/** Flat (row-major, C-contiguous) point rows as returned by
 *  {@link ThreeDCamera.project_points_flat} — two columns (x, y), in the same
 *  camera-space the Canvas2D transform consumes. */
export interface FlatProjectedPoints {
  data: Float64Array;
  rowDim: 2;
  rowCount: number;
}

/** The three colour infos of one VMobject plus the inputs they depend on
 *  (the input fields let consumers validate cached geometry cheaply). */
export interface MobjectColorInfos {
  points: unknown;
  pointsVersion: number;
  fillSource: unknown;
  strokeSource: unknown;
  bgStrokeSource: unknown;
  light: unknown;
  shading: boolean;
  fill: DrawColorInfo | null;
  stroke: DrawColorInfo | null;
  bgStroke: DrawColorInfo | null;
}

/** Unshaded colour stops plus the corner normals the shader needs to apply
 *  `modified_rgbas` shading on the GPU. */
export interface BaseColorInfo {
  colors: number[][];
  gradient: boolean;
  start: number[];
  end: number[];
  normal0: number[];
  normal1: number[];
}

/** Per-source base colour descriptions for the WebGL backend. */
export interface MobjectBaseColors {
  fill: BaseColorInfo | null;
  stroke: BaseColorInfo | null;
  bgStroke: BaseColorInfo | null;
}

/**
 * The colour/gradient description a VMobject's rgba array resolves to, with
 *  the exact semantics of `set_cairo_context_color`: one colour is flat, more
 *  than one becomes a linear gradient between the object's gradient endpoints. */
export interface DrawColorInfo {
  /** One or two rgba rows (the GPU shader supports at most two stops). */
  colors: number[][];
  /** True when a linear gradient between {@link start} and {@link end} applies. */
  gradient: boolean;
  /** World-space gradient endpoints (only meaningful when `gradient`). */
  start: number[];
  end: number[];
}


export interface ThreeDCameraOptions extends CameraOptions {
  focal_distance?: number;
  shading_factor?: number;
  default_distance?: number;
  light_source_start_point?: readonly number[];
  should_apply_shading?: boolean;
  exponential_projection?: boolean;
  phi?: number;
  theta?: number;
  gamma?: number;
  zoom?: number;
}

/** Center function used by fixed-orientation mobjects. */
export type CenterFunc = () => readonly number[] | NDArray;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** Fallback rgba source for VMobjects without an explicit array. */
const TRANSPARENT_RGBA = array([[0, 0, 0, 0]]) as NDArray;

/**
 * Corner unit normals of a shaded 3D VMobject's point store, or null when the
 * store is unusable.  Same corners/normals `get_mobject_base_colors` reports
 * (shared by all three colour sources); the WebGL position-update path needs
 * only these, without resolving any rgba rows.
 *
 * @internal
 */
export function _cornerNormalsFlat(points: unknown): { normal0: number[]; normal1: number[] } | null {
  const pts = _cContiguousRows(points);
  if (pts === null || pts.rowDim < 3 || pts.rowCount < 4) return null;
  const n = pts.rowCount;
  const endIndex = Math.floor((n - 1) / 6) * 3;
  return {
    normal0: vmobUnitNormalFlat(pts.data, pts.rowDim, n, 0),
    normal1: vmobUnitNormalFlat(pts.data, pts.rowDim, n, endIndex),
  };
}

/** Read row `index` of a C-contiguous rgba store into a fresh `[r, g, b, a]`. */
function readRgbaRow(data: ArrayLike<number>, index: number): number[] {
  const base = index * 4;
  return [
    Number(data[base] ?? 0),
    Number(data[base + 1] ?? 0),
    Number(data[base + 2] ?? 0),
    Number(data[base + 3] ?? 0),
  ];
}

/**
 * Midpoint of the bounding box over ALL point rows, read straight from the
 * backing store (the `Mobject.get_center()` equivalent the depth sort needs;
 * `get_center()` allocated a boundary array and a result array per face).
 * Returns null when there are no rows (the caller falls back to
 * `get_z_index_reference_point()`).
 */
function zReferenceDotFlat(mob: Mobject, rx: number, ry: number, rz: number): number {
  const center = bboxCenterFlat((mob as { points?: unknown }).points);
  if (center === null) {
    const fallback = mob.get_z_index_reference_point();
    return (fallback[0] ?? 0) * rx + (fallback[1] ?? 0) * ry + (fallback[2] ?? 0) * rz;
  }
  return center[0]! * rx + center[1]! * ry + center[2]! * rz;
}

/**
 * Bounding-box midpoint backing {@link zReferenceDotFlat}, exposed as a
 * module function so the depth-sort cache can reuse it without recomputing
 * the dot product (the camera moves every drag frame while the centers do
 * not).
 */
function bboxCenterFlat(points: unknown): [number, number, number] | null {
  const pts = _cContiguousRows(points);
  if (pts === null || pts.rowDim < 3 || pts.rowCount === 0) return null;
  const { data, rowDim, rowCount } = pts;
  const flat = data instanceof Float64Array ? data : null;
  let min0 = Infinity; let max0 = -Infinity;
  let min1 = Infinity; let max1 = -Infinity;
  let min2 = Infinity; let max2 = -Infinity;
  let nan0 = false; let nan1 = false; let nan2 = false;
  for (let row = 0; row < rowCount; row += 1) {
    const base = row * rowDim;
    // Float64 elements are already numbers; the per-element `Number()` call
    // showed up in profiles, so it is kept only for exotic stores.
    const v0 = flat !== null ? flat[base]! : Number(data[base]);
    if (v0 !== v0) nan0 = true; else { if (v0 < min0) min0 = v0; if (v0 > max0) max0 = v0; }
    const v1 = flat !== null ? flat[base + 1]! : Number(data[base + 1]);
    if (v1 !== v1) nan1 = true; else { if (v1 < min1) min1 = v1; if (v1 > max1) max1 = v1; }
    const v2 = flat !== null ? flat[base + 2]! : Number(data[base + 2]);
    if (v2 !== v2) nan2 = true; else { if (v2 < min2) min2 = v2; if (v2 > max2) max2 = v2; }
  }
  return [
    nan0 ? NaN : (min0 + max0) / 2,
    nan1 ? NaN : (min1 + max1) / 2,
    nan2 ? NaN : (min2 + max2) / 2,
  ];
}

/**
 * Flat port of `get_3d_vmob_unit_normal` (and, in turn, `get_unit_normal`).
 * Reading the three points straight from the backing store avoids materialising
 * every row through numpy for each face and each frame.
 */
function vmobUnitNormalFlat(
  data: ArrayLike<number>,
  rowDim: number,
  nPoints: number,
  pointIndex: number,
): number[] {
  // `len(vmob.get_anchors()) <= 2` ⇔ at most one cubic curve ⇔ n <= 4.
  if (nPoints <= 4) return [0, 1, 0];
  const i = pointIndex;
  const im3 = i > 2 ? i - 3 : nPoints - 4;
  const ip3 = i < nPoints - 3 ? i + 3 : 3;
  const baseI = i * rowDim;
  const baseP = ip3 * rowDim;
  const baseM = im3 * rowDim;
  const v1 = [
    Number(data[baseP] ?? 0) - Number(data[baseI] ?? 0),
    Number(data[baseP + 1] ?? 0) - Number(data[baseI + 1] ?? 0),
    Number(data[baseP + 2] ?? 0) - Number(data[baseI + 2] ?? 0),
  ];
  const v2 = [
    Number(data[baseM] ?? 0) - Number(data[baseI] ?? 0),
    Number(data[baseM + 1] ?? 0) - Number(data[baseI + 1] ?? 0),
    Number(data[baseM + 2] ?? 0) - Number(data[baseI + 2] ?? 0),
  ];
  return unitNormalFlat(v1, v2);
}

/** Flat port of `manim.utils.space_ops.get_unit_normal`. */
function unitNormalFlat(v1: number[], v2: number[]): number[] {
  const tol = 1e-6;
  const div1 = Math.max(Math.abs(v1[0] ?? 0), Math.abs(v1[1] ?? 0), Math.abs(v1[2] ?? 0));
  const div2 = Math.max(Math.abs(v2[0] ?? 0), Math.abs(v2[1] ?? 0), Math.abs(v2[2] ?? 0));
  let u: number[];
  if (div1 === 0) {
    if (div2 === 0) return [0, -1, 0];
    u = [(v2[0] ?? 0) / div2, (v2[1] ?? 0) / div2, (v2[2] ?? 0) / div2];
  } else if (div2 === 0) {
    u = [(v1[0] ?? 0) / div1, (v1[1] ?? 0) / div1, (v1[2] ?? 0) / div1];
  } else {
    const u1 = [(v1[0] ?? 0) / div1, (v1[1] ?? 0) / div1, (v1[2] ?? 0) / div1];
    const u2 = [(v2[0] ?? 0) / div2, (v2[1] ?? 0) / div2, (v2[2] ?? 0) / div2];
    const cp = [
      u1[1]! * u2[2]! - u1[2]! * u2[1]!,
      u1[2]! * u2[0]! - u1[0]! * u2[2]!,
      u1[0]! * u2[1]! - u1[1]! * u2[0]!,
    ];
    const cpNorm = Math.sqrt(cp[0]! * cp[0]! + cp[1]! * cp[1]! + cp[2]! * cp[2]!);
    if (cpNorm > tol) return [cp[0]! / cpNorm, cp[1]! / cpNorm, cp[2]! / cpNorm];
    u = u1;
  }
  if (Math.abs(u[0] ?? 0) < tol && Math.abs(u[1] ?? 0) < tol) return [0, -1, 0];
  const cp = [
    -(u[0] ?? 0) * (u[2] ?? 0),
    -(u[1] ?? 0) * (u[2] ?? 0),
    (u[0] ?? 0) * (u[0] ?? 0) + (u[1] ?? 0) * (u[1] ?? 0),
  ];
  const cpNorm = Math.sqrt(cp[0]! * cp[0]! + cp[1]! * cp[1]! + cp[2]! * cp[2]!);
  return [cp[0]! / cpNorm, cp[1]! / cpNorm, cp[2]! / cpNorm];
}

/** Flat port of `get_shaded_rgb`: returns the rgba row with the light added. */
function shadeFlat(
  rgba: number[],
  point: number[],
  unitNormal: number[],
  lightSource: number[],
): number[] {
  const dx = (lightSource[0] ?? 0) - (point[0] ?? 0);
  const dy = (lightSource[1] ?? 0) - (point[1] ?? 0);
  const dz = (lightSource[2] ?? 0) - (point[2] ?? 0);
  const norm = Math.hypot(dx, dy, dz);
  const toSun = norm > 0 ? [dx / norm, dy / norm, dz / norm] : [0, 0, 0];
  const dot =
    (unitNormal[0] ?? 0) * (toSun[0] ?? 0) +
    (unitNormal[1] ?? 0) * (toSun[1] ?? 0) +
    (unitNormal[2] ?? 0) * (toSun[2] ?? 0);
  let light = 0.5 * dot ** 3;
  if (light < 0) light *= 0.5;
  return [
    (rgba[0] ?? 0) + light,
    (rgba[1] ?? 0) + light,
    (rgba[2] ?? 0) + light,
    rgba[3] ?? 0,
  ];
}

/**
 * Faithful port of `manim.camera.three_d_camera.ThreeDCamera`.
 *
 * The camera keeps a rotation matrix derived from the euler-like angles
 * `phi`/`theta`/`gamma`, projects points for display, sorts 3D mobjects by
 * depth and tracks fixed-in-frame / fixed-orientation membership.  The
 * external `the original manim renderer` performs the actual rasterization from the
 * serialized camera state.
 */
export class ThreeDCamera extends Camera {
  focal_distance: number;
  phi: number;
  theta: number;
  gamma: number;
  zoom: number;
  shading_factor: number;
  default_distance: number;
  light_source_start_point: readonly number[];
  light_source: Point;
  should_apply_shading: boolean;
  exponential_projection: boolean;

  phi_tracker: ValueTracker;
  theta_tracker: ValueTracker;
  focal_distance_tracker: ValueTracker;
  gamma_tracker: ValueTracker;
  zoom_tracker: ValueTracker;

  fixed_orientation_mobjects: Map<Mobject, CenterFunc> = new Map();
  fixed_in_frame_mobjects: Set<Mobject> = new Set();
  rotation_matrix: NDArray = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as NDArray;
  /** Cache key (phi/theta/gamma) and flat rows backing {@link rotation_matrix}. */
  private _rotation_key = "";
  /** Unsorted display-list cache keyed by (mobjects, structure epoch). */
  private _displayedCache: {
    epoch: number;
    source: Iterable<Mobject>;
    includeSubmobjects: boolean;
    excluded: Iterable<Mobject> | null | undefined;
    result: Mobject[];
  } | null = null;
  /** Flat rows of {@link rotation_matrix}, refreshed with it. */
  private _rotation_rows: [number[], number[], number[]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  /** Shading/gradient colour info keyed by (vmobject, rgba source). */
  private _colorInfoCache = new WeakMap<VMobject, Map<unknown, {
    points: unknown;
    pointsVersion: number;
    light: unknown;
    shaded: boolean;
    info: DrawColorInfo;
    result?: NDArray;
  }>>();
  /** All three colour infos of one mobject, keyed by the inputs they depend on. */
  private _mobjectColorCache = new WeakMap<VMobject, MobjectColorInfos>();
  /** Depth-sort bounding-box centers, keyed by mobject and validated by
   *  `points_version` (see `get_mobjects_to_display`). */
  private _sortCenterCache = new WeakMap<Mobject, { version: number; center: [number, number, number] }>();

  // NOTE: `frame_center` is backed by a Point mobject (mirrors the Python
  // @property). The base Camera constructor assigns `frame_center` while
  // `_frame_center_point` does not exist yet (TS fields initialize after
  // super()), so the write is stashed and replayed below.
  private declare _frame_center_point?: Point;
  private declare _pending_frame_center?: readonly number[];

  constructor(
    focal_distanceOrOptions: number | ThreeDCameraOptions = 20.0,
    shading_factor = 0.2,
    default_distance = 5.0,
    // Python: 9 * DOWN + 7 * LEFT + 10 * OUT.
    light_source_start_point: readonly number[] = [-7, -9, 10],
    should_apply_shading = true,
    exponential_projection = false,
    phi = 0,
    theta = -90 * DEGREES,
    gamma = 0,
    zoom = 1,
    options: ThreeDCameraOptions = {},
  ) {
    // Accept both the Python positional signature and a leading options bag
    // (the TS constructors route configuration through an options object).
    let bag: ThreeDCameraOptions;
    let positionalFocal = focal_distanceOrOptions as number;
    if (isPlainObject(focal_distanceOrOptions)) {
      bag = { ...focal_distanceOrOptions, ...options };
      positionalFocal = (bag.focal_distance as number | undefined) ?? 20.0;
    } else {
      bag = { ...options };
    }
    const resolvedFocal = bag.focal_distance ?? positionalFocal;
    const resolvedShading = bag.shading_factor ?? shading_factor;
    const resolvedDefaultDistance = bag.default_distance ?? default_distance;
    const resolvedLightSource = bag.light_source_start_point ?? light_source_start_point;
    const resolvedShouldApply = bag.should_apply_shading ?? should_apply_shading;
    const resolvedExponential = bag.exponential_projection ?? exponential_projection;
    const resolvedPhi = bag.phi ?? phi;
    const resolvedTheta = bag.theta ?? theta;
    const resolvedGamma = bag.gamma ?? gamma;
    const resolvedZoom = bag.zoom ?? zoom;

    // `frame_center` is consumed by the overridden accessor below; the rest of
    // the bag is forwarded to Camera.
    const { frame_center: _frameCenter, ...cameraOptions } = bag;
    void _frameCenter;
    super(cameraOptions);

    // Create the frame-center Point, replaying the write the base constructor
    // made through the overridden setter.
    const pendingCenter = this._pending_frame_center ?? ORIGIN;
    this._frame_center_point = new Point(pendingCenter, BLACK, { stroke_width: 0 });
    this._pending_frame_center = undefined;

    this.focal_distance = resolvedFocal;
    this.phi = resolvedPhi;
    this.theta = resolvedTheta;
    this.gamma = resolvedGamma;
    this.zoom = resolvedZoom;
    this.shading_factor = resolvedShading;
    this.default_distance = resolvedDefaultDistance;
    this.light_source_start_point = resolvedLightSource;
    this.light_source = new Point(resolvedLightSource);
    this.should_apply_shading = resolvedShouldApply;
    this.exponential_projection = resolvedExponential;
    this.max_allowable_norm = 3 * Number(config.frame_width);

    this.phi_tracker = new ValueTracker(this.phi);
    this.theta_tracker = new ValueTracker(this.theta);
    this.focal_distance_tracker = new ValueTracker(this.focal_distance);
    this.gamma_tracker = new ValueTracker(this.gamma);
    this.zoom_tracker = new ValueTracker(this.zoom);

    this.reset_rotation_matrix();
  }

  override get frame_center(): readonly number[] {
    if (this._frame_center_point === undefined) {
      return this._pending_frame_center ?? ORIGIN;
    }
    return pointRows(this._frame_center_point.points, 3)[0] ?? [0, 0, 0];
  }

  override set frame_center(value: readonly number[]) {
    if (this._frame_center_point === undefined) {
      this._pending_frame_center = value;
      return;
    }
    this._frame_center_point.move_to(value as readonly number[]);
  }

  override capture_mobjects(mobjects: Iterable<Mobject>, options: CaptureOptions = {}): void {
    this.reset_rotation_matrix();
    super.capture_mobjects(mobjects, options);
  }

  get_value_trackers(): ValueTracker[] {
    return [
      this.phi_tracker,
      this.theta_tracker,
      this.focal_distance_tracker,
      this.gamma_tracker,
      this.zoom_tracker,
    ];
  }

  modified_rgbas(vmobject: VMobject, rgbas: NDArray): NDArray {
    if (!this.should_apply_shading) return rgbas;
    const vm = vmobject as VMobject;
    if (vm.shade_in_3d && vm.get_num_points() > 0) {
      const info = this.get_shaded_color_info(vm, rgbas);
      if (info !== null) {
        // Cache the NDArray view too: fill, background stroke and foreground
        // stroke each ask for the same two shaded rows every frame.
        const entry = this.lookup_color_entry(vm, rgbas);
        if (entry?.result !== undefined) return entry.result;
        const result = array(info.colors) as NDArray;
        if (entry !== undefined) entry.result = result;
        return result;
      }
      // Slow fallback (non-contiguous source array, ...): the original
      // row-materialising implementation.
      return this.modified_rgbas_slow(vm, rgbas);
    }
    return rgbas;
  }

  /**
   * Flat-array implementation of `modified_rgbas` for shaded 3D mobjects.
   *
   * Shading only depends on the VMobject's two corner points, their unit
   * normals and the light source; every frame recomputed the same values for
   * static geometry (and went through numpy row materialisation to do it).
   * This returns the same two rgba rows as `modified_rgbas` plus the gradient
   * endpoints, so both the Canvas2D and the WebGL backend can consume it.
   */
  get_shaded_color_info(vmobject: VMobject, source: NDArray): DrawColorInfo | null {
    const vm = vmobject as VMobject;
    const cached = this.lookup_color_entry(vm, source);
    if (cached !== undefined && cached.shaded) return cached.info;
    const points = vm.points;
    const lightPoints = this.light_source.points;
    const info = this.compute_shaded_color_info(vm, source, points, lightPoints);
    if (info !== null) {
      this.store_color_entry(vm, source, { points, pointsVersion: vm.points_version, light: lightPoints, shaded: true, info });
    }
    return info;
  }

  private color_entries(vm: VMobject): Map<unknown, {
    points: unknown;
    pointsVersion: number;
    light: unknown;
    shaded: boolean;
    info: DrawColorInfo;
    result?: NDArray;
  }> {
    let entries = this._colorInfoCache.get(vm);
    if (entries === undefined) {
      entries = new Map();
      this._colorInfoCache.set(vm, entries);
    }
    return entries;
  }

  /** Valid (non-stale) cache entry for `(vmobject, rgba source)`, if any. */
  private lookup_color_entry(vm: VMobject, source: NDArray): {
    points: unknown;
    pointsVersion: number;
    light: unknown;
    shaded: boolean;
    info: DrawColorInfo;
    result?: NDArray;
  } | undefined {
    const entry = this._colorInfoCache.get(vm)?.get(source);
    if (entry === undefined) return undefined;
    // Version (not reference): point stores are mutated in place by the
    // rotate/apply/interpolate fast paths while keeping their identity.
    if (entry.pointsVersion !== vm.points_version || entry.points !== vm.points) return undefined;
    if (entry.shaded && entry.light !== this.light_source.points) return undefined;
    return entry;
  }

  private store_color_entry(vm: VMobject, source: NDArray, entry: {
    points: unknown;
    pointsVersion: number;
    light: unknown;
    shaded: boolean;
    info: DrawColorInfo;
    result?: NDArray;
  }): void {
    this.color_entries(vm).set(source, entry);
  }

  private compute_shaded_color_info(
    vm: VMobject,
    source: NDArray,
    points: NDArray,
    lightPoints: NDArray,
  ): DrawColorInfo | null {
    const pts = _cContiguousRows(points);
    const src = _cContiguousRows(source);
    if (pts === null || pts.rowDim < 3 || pts.rowCount < 4 || src === null || src.rowDim < 4) {
      return null;
    }
    const light = _cContiguousRows(lightPoints);
    const lightPoint: number[] = light !== null && light.rowCount > 0
      ? [Number(light.data[0]), Number(light.data[1]), Number(light.data[2])]
      : [0, 0, 0];
    const n = pts.rowCount;
    const rowDim = pts.rowDim;
    const startCorner = [Number(pts.data[0]), Number(pts.data[1]), Number(pts.data[2])];
    const endIndex = Math.floor((n - 1) / 6) * 3;
    const endCorner = [
      Number(pts.data[endIndex * rowDim]),
      Number(pts.data[endIndex * rowDim + 1]),
      Number(pts.data[endIndex * rowDim + 2]),
    ];
    const startNormal = vmobUnitNormalFlat(pts.data, rowDim, n, 0);
    const endNormal = vmobUnitNormalFlat(pts.data, rowDim, n, endIndex);
    // `rgbas.repeat(2, axis=0)` for <2 rows, else the first two rows.
    const row0 = src.rowCount >= 1 ? readRgbaRow(src.data, 0) : [0, 0, 0, 0];
    const row1 = src.rowCount >= 2 ? readRgbaRow(src.data, 1) : row0.slice();
    const c0 = shadeFlat(row0, startCorner, startNormal, lightPoint);
    const c1 = shadeFlat(row1, endCorner, endNormal, lightPoint);
    return {
      colors: [c0, c1],
      gradient: true,
      start: startCorner,
      end: endCorner,
    };
  }

  /**
   * Resolve a VMobject's rgba array to the colour stops the rasterizer needs,
   * mirroring `set_cairo_context_color`: shading for `shade_in_3d` objects,
   * flat colour for uniform rows, a two-stop linear gradient otherwise.
   * Returns `null` for shapes the WebGL backend cannot express (more than two
   * stops, non-contiguous data), which then fall back to Canvas2D.
   */
  /**
   * All three colour infos (fill / stroke / background stroke) of a mobject in
   * one cache lookup.  The WebGL record validation runs this for every face on
   * every frame, so it reads the rgba sources straight from their fields and
   * only recomputes what actually changed.
   */
  get_mobject_color_infos(vmobject: VMobject): MobjectColorInfos {
    const vm = vmobject as VMobject;
    const points = vm.points;
    const fillSource = vm.fill_rgbas ?? TRANSPARENT_RGBA;
    const strokeSource = vm.stroke_rgbas ?? TRANSPARENT_RGBA;
    const bgStrokeSource = vm.background_stroke_rgbas ?? TRANSPARENT_RGBA;
    const light = this.light_source.points;
    const shading = this.should_apply_shading;
    const cached = this._mobjectColorCache.get(vm);
    const pointsVersion = vm.points_version;
    if (
      cached !== undefined &&
      cached.points === points &&
      cached.pointsVersion === pointsVersion &&
      cached.fillSource === fillSource &&
      cached.strokeSource === strokeSource &&
      cached.bgStrokeSource === bgStrokeSource &&
      cached.light === light &&
      cached.shading === shading
    ) {
      return cached;
    }
    const entry = {
      points,
      pointsVersion,
      fillSource,
      strokeSource,
      bgStrokeSource,
      light,
      shading,
      fill: this.get_draw_color_info(vm, fillSource),
      stroke: this.get_draw_color_info(vm, strokeSource),
      bgStroke: this.get_draw_color_info(vm, bgStrokeSource),
    };
    this._mobjectColorCache.set(vm, entry);
    return entry;
  }

  /**
   * Base (unshaded) colours and corner normals for the WebGL backend.  The
   * shader applies `modified_rgbas` shading per corner, so moving the light
   * source or toggling `should_apply_shading` never invalidates cached
   * geometry.
   */
  get_mobject_base_colors(vmobject: VMobject): MobjectBaseColors | null {
    const vm = vmobject as VMobject;
    if (vm.shade_in_3d !== true) return null;
    const pts = _cContiguousRows(vm.points);
    if (pts === null || pts.rowDim < 3 || pts.rowCount < 4) return null;
    const n = pts.rowCount;
    const rowDim = pts.rowDim;
    const startCorner = [Number(pts.data[0]), Number(pts.data[1]), Number(pts.data[2])];
    const endIndex = Math.floor((n - 1) / 6) * 3;
    const endCorner = [
      Number(pts.data[endIndex * rowDim]),
      Number(pts.data[endIndex * rowDim + 1]),
      Number(pts.data[endIndex * rowDim + 2]),
    ];
    const normal0 = vmobUnitNormalFlat(pts.data, rowDim, n, 0);
    const normal1 = vmobUnitNormalFlat(pts.data, rowDim, n, endIndex);
    const resolve = (source: NDArray): BaseColorInfo | null => {
      const src = _cContiguousRows(source);
      if (src === null || src.rowDim < 4 || src.rowCount === 0) return null;
      const row0 = readRgbaRow(src.data, 0);
      if (src.rowCount === 1) {
        return { colors: [row0], gradient: false, start: startCorner, end: endCorner, normal0, normal1 };
      }
      if (src.rowCount > 2) return null;
      const row1 = readRgbaRow(src.data, 1);
      const equal =
        row0[0] === row1[0] && row0[1] === row1[1] && row0[2] === row1[2] && row0[3] === row1[3];
      return {
        colors: equal ? [row0] : [row0, row1],
        gradient: !equal,
        start: startCorner,
        end: endCorner,
        normal0,
        normal1,
      };
    };
    return {
      fill: resolve(vm.fill_rgbas ?? TRANSPARENT_RGBA),
      stroke: resolve(vm.stroke_rgbas ?? TRANSPARENT_RGBA),
      bgStroke: resolve(vm.background_stroke_rgbas ?? TRANSPARENT_RGBA),
    };
  }

  get_draw_color_info(vmobject: VMobject, source: NDArray): DrawColorInfo | null {
    const vm = vmobject as VMobject;
    if (this.should_apply_shading && vm.shade_in_3d && vm.get_num_points() > 0) {
      return this.get_shaded_color_info(vm, source);
    }
    const cached = this.lookup_color_entry(vm, source);
    if (cached !== undefined && !cached.shaded) return cached.info;
    const src = _cContiguousRows(source);
    if (src === null || src.rowDim < 4 || src.rowCount === 0) return null;
    const row0 = readRgbaRow(src.data, 0);
    let info: DrawColorInfo;
    if (src.rowCount === 1) {
      info = { colors: [row0], gradient: false, start: [0, 0, 0], end: [0, 0, 0] };
    } else if (src.rowCount > 2) {
      return null;
    } else {
      const row1 = readRgbaRow(src.data, 1);
      if (row0[0] === row1[0] && row0[1] === row1[1] && row0[2] === row1[2] && row0[3] === row1[3]) {
        info = { colors: [row0], gradient: false, start: [0, 0, 0], end: [0, 0, 0] };
      } else {
        const [startPoint, endPoint] = vm.get_gradient_start_and_end_points();
        info = { colors: [row0, row1], gradient: true, start: startPoint, end: endPoint };
      }
    }
    this.store_color_entry(vm, source, { points: vm.points, pointsVersion: vm.points_version, light: null, shaded: false, info });
    return info;
  }

  private modified_rgbas_slow(vm: VMobject, rgbas: NDArray): NDArray {
    const light_source_point = pointRows(this.light_source.points, 3)[0] ?? [0, 0, 0];
    const sourceRows = pointRows(rgbas, 4);
    let shaded_rgbas: number[][];
    if (sourceRows.length < 2) {
      // np.repeat(rgbas, 2, axis=0): each row duplicated (0 rows -> 0 rows).
      shaded_rgbas = sourceRows.flatMap((row) => [[...row], [...row]]);
      if (shaded_rgbas.length < 2) {
        shaded_rgbas = [
          [...(sourceRows[0] ?? [0, 0, 0, 0])],
          [...(sourceRows[0] ?? [0, 0, 0, 0])],
        ];
      }
    } else {
      shaded_rgbas = sourceRows.slice(0, 2).map((row) => [...row]);
    }
    const startShaded = get_shaded_rgb(
      shaded_rgbas[0]!.slice(0, 3),
      pointRows(get_3d_vmob_start_corner(vm), 3)[0] ?? [0, 0, 0],
      pointRows(get_3d_vmob_start_corner_unit_normal(vm), 3)[0] ?? [0, 0, 0],
      light_source_point,
    );
    const endShaded = get_shaded_rgb(
      shaded_rgbas[1]!.slice(0, 3),
      pointRows(get_3d_vmob_end_corner(vm), 3)[0] ?? [0, 0, 0],
      pointRows(get_3d_vmob_end_corner_unit_normal(vm), 3)[0] ?? [0, 0, 0],
      light_source_point,
    );
    for (let channel = 0; channel < 3; channel += 1) {
      shaded_rgbas[0]![channel] = startShaded[channel] ?? 0;
      shaded_rgbas[1]![channel] = endShaded[channel] ?? 0;
    }
    return array(shaded_rgbas) as NDArray;
  }

  override get_stroke_rgbas(vmobject: VMobject, background = false): NDArray {
    return this.modified_rgbas(vmobject, vmobject.get_stroke_rgbas(background));
  }

  override get_fill_rgbas(vmobject: VMobject): NDArray {
    return this.modified_rgbas(vmobject, vmobject.get_fill_rgbas());
  }

  override get_mobjects_to_display(
    mobjects: Iterable<Mobject>,
    include_submobjects = true,
    excluded_mobjects?: Iterable<Mobject> | null,
  ): Mobject[] {
    // The family extraction (walking every submobject of every mobject) only
    // changes when the scene structure changes; the geometry epoch is bumped
    // by every structural mutation, so the unsorted list can be cached while
    // the depth sort still runs per frame (camera movement reorders it).
    const epoch = mobject_geometry_epoch();
    const cache = this._displayedCache;
    let displayed: Mobject[];
    if (
      cache !== null &&
      cache.epoch === epoch &&
      cache.source === mobjects &&
      cache.includeSubmobjects === include_submobjects &&
      cache.excluded === excluded_mobjects
    ) {
      displayed = cache.result;
    } else {
      displayed = super.get_mobjects_to_display(
        mobjects,
        include_submobjects,
        excluded_mobjects,
      );
      this._displayedCache = {
        epoch,
        source: mobjects,
        includeSubmobjects: include_submobjects,
        excluded: excluded_mobjects,
        result: displayed,
      };
    }
    const row = this._rotation_rows[2] ?? [0, 0, 1];
    const rx = row[0] ?? 0;
    const ry = row[1] ?? 0;
    const rz = row[2] ?? 0;
    const count = displayed.length;
    // Depth keys in a typed array and an index sort: the previous
    // decorate-sort-undecorate allocated one `{mob, key, index}` object per
    // face (thousands per frame on a surface).
    const keys = new Float64Array(count);
    for (let index = 0; index < count; index += 1) {
      const mob = displayed[index]!;
      // Python `hasattr(mob, "shade_in_3d")`: the Mobject proxy raises
      // AttributeError for missing attributes, so probe with `in` (handled
      // by the proxy's `has` trap, which is cheaper than the throwing `get`).
      const shade = mob instanceof VMobject
        ? (mob as VMobject).shade_in_3d === true
        : "shade_in_3d" in mob
          ? Boolean((mob as { shade_in_3d?: boolean }).shade_in_3d)
          : false;
      if (!shade) {
        keys[index] = Infinity;
        continue;
      }
      // Bounding-box centers are cached by points version: camera-only
      // movement (the interactive drag case) re-sorts identical geometry, so
      // the full row scan runs only when the points actually changed.
      const version = (mob as Mobject).points_version;
      const cached = this._sortCenterCache.get(mob);
      let center: [number, number, number] | null;
      if (cached !== undefined && cached.version === version) {
        center = cached.center;
      } else {
        center = bboxCenterFlat((mob as { points?: unknown }).points);
        if (center !== null) this._sortCenterCache.set(mob, { version, center });
      }
      keys[index] = center === null
        ? zReferenceDotFlat(mob, rx, ry, rz)
        : center[0]! * rx + center[1]! * ry + center[2]! * rz;
    }
    const order = new Array<number>(count);
    for (let index = 0; index < count; index += 1) order[index] = index;
    // Subtraction comparator: identical ordering to the branchy version
    // (NaN and ±Infinity fall through `||` to the index tiebreak, matching
    // the old `<`/`>`-both-false behaviour), but measurably faster.
    order.sort((a, b) => keys[a]! - keys[b]! || a - b);
    const result = new Array<Mobject>(count);
    for (let index = 0; index < count; index += 1) result[index] = displayed[order[index]!]!;
    return result;
  }

  get_phi(): number {
    return this.phi_tracker.get_value();
  }

  get_theta(): number {
    return this.theta_tracker.get_value();
  }

  get_focal_distance(): number {
    return this.focal_distance_tracker.get_value();
  }

  get_gamma(): number {
    return this.gamma_tracker.get_value();
  }

  get_zoom(): number {
    return this.zoom_tracker.get_value();
  }

  set_phi(value: number): void {
    this.phi_tracker.set_value(value);
  }

  set_theta(value: number): void {
    this.theta_tracker.set_value(value);
  }

  set_focal_distance(value: number): void {
    this.focal_distance_tracker.set_value(value);
  }

  set_gamma(value: number): void {
    this.gamma_tracker.set_value(value);
  }

  set_zoom(value: number): void {
    this.zoom_tracker.set_value(value);
  }

  reset_rotation_matrix(): void {
    // The rotation matrix only depends on (phi, theta, gamma); regenerating it
    // through numpy's `matmul` on every frame (and for every frame capture) was
    // pure overhead whenever the camera angles were unchanged.
    const phi = this.get_phi();
    const theta = this.get_theta();
    const gamma = this.get_gamma();
    const key = `${phi}\u0000${theta}\u0000${gamma}`;
    if (key === this._rotation_key) return;
    this._rotation_key = key;
    this.rotation_matrix = this.generate_rotation_matrix();
    const rows = pointRows(this.rotation_matrix, 3);
    this._rotation_rows = [
      rows[0] ?? [1, 0, 0],
      rows[1] ?? [0, 1, 0],
      rows[2] ?? [0, 0, 1],
    ];
  }

  get_rotation_matrix(): NDArray {
    return this.rotation_matrix;
  }

  /** The rotation matrix rows as plain arrays (cached alongside the matrix). */
  get_rotation_rows(): [number[], number[], number[]] {
    return this._rotation_rows;
  }

  generate_rotation_matrix(): NDArray {
    const phi = this.get_phi();
    const theta = this.get_theta();
    const gamma = this.get_gamma();
    const matrices: NDArray[] = [
      rotation_about_z(-theta - 90 * DEGREES),
      rotation_matrix(-phi, RIGHT),
      rotation_about_z(gamma),
    ];
    let result: NDArray = identity(3) as NDArray;
    for (const matrix of matrices) {
      result = matmul(matrix, result) as NDArray;
    }
    return result;
  }

  project_points(points: NDArray): NDArray {
    const frame_center = this.frame_center;
    const focal_distance = this.get_focal_distance();
    const zoom = this.get_zoom();
    const rotRows = this._rotation_rows;

    const rows = pointRows(points, 3).map((point) =>
      point.map((value, index) => value - (frame_center[index] ?? 0)),
    );
    // points @ rot_matrix.T: component j = dot(point, row j of rot_matrix).
    const rotated = rows.map((point) => [0, 1, 2].map((j) => {
      const row = rotRows[j] ?? [0, 0, 0];
      return (
        (point[0] ?? 0) * (row[0] ?? 0) +
        (point[1] ?? 0) * (row[1] ?? 0) +
        (point[2] ?? 0) * (row[2] ?? 0)
      );
    }));
    const zs = rotated.map((row) => row[2] ?? 0);
    for (const i of [0, 1]) {
      for (let index = 0; index < rotated.length; index += 1) {
        const z = zs[index] ?? 0;
        let factor: number;
        if (this.exponential_projection) {
          if (z < 0) factor = focal_distance / (focal_distance - z);
          else factor = Math.exp(z / focal_distance);
        } else {
          factor = focal_distance / (focal_distance - z);
          if (focal_distance - z < 0) factor = 10 ** 6;
        }
        rotated[index]![i] = (rotated[index]?.[i] ?? 0) * factor * zoom;
      }
    }
    return array(rotated.length > 0 ? rotated : []) as NDArray;
  }

  /**
   * Flat-array projection used by the Canvas2D draw path: reads a C-contiguous
   * point matrix straight from its backing store and writes projected (x, y)
   * pairs into a `Float64Array`, skipping the per-point row materialisation and
   * the final NDArray allocation of {@link project_points}.
   *
   * Returns `null` when the input is not a plain contiguous matrix or holds a
   * non-finite value (the caller then falls back to the NDArray path, which
   * replicates Python's `[[0, 0, 0]]` fallback).
   */
  project_points_flat(points: unknown): FlatProjectedPoints | null {
    const flat = _cContiguousRows(points);
    if (flat === null || flat.rowDim < 3) return null;
    const { data, rowDim, rowCount } = flat;
    if (rowCount === 0) return { data: new Float64Array(0), rowDim: 2, rowCount: 0 };
    const frame_center = this.frame_center;
    const focal = this.get_focal_distance();
    const zoom = this.get_zoom();
    const [r0, r1, r2] = this._rotation_rows;
    const a0 = r0[0] ?? 0; const a1 = r0[1] ?? 0; const a2 = r0[2] ?? 0;
    const b0 = r1[0] ?? 0; const b1 = r1[1] ?? 0; const b2 = r1[2] ?? 0;
    const c0 = r2[0] ?? 0; const c1 = r2[1] ?? 0; const c2 = r2[2] ?? 0;
    const fx = frame_center[0] ?? 0;
    const fy = frame_center[1] ?? 0;
    const fz = frame_center[2] ?? 0;
    const exponential = this.exponential_projection;
    const out = new Float64Array(rowCount * 2);
    for (let index = 0; index < rowCount; index += 1) {
      const base = index * rowDim;
      const x = Number(data[base]) - fx;
      const y = Number(data[base + 1]) - fy;
      const z = Number(data[base + 2]) - fz;
      if (!Number.isFinite(x + y + z)) return null;
      const rz = x * c0 + y * c1 + z * c2;
      let factor: number;
      if (exponential) {
        factor = rz < 0 ? focal / (focal - rz) : Math.exp(rz / focal);
      } else {
        factor = focal / (focal - rz);
        if (focal - rz < 0) factor = 10 ** 6;
      }
      const scale = factor * zoom;
      out[index * 2] = (x * a0 + y * a1 + z * a2) * scale;
      out[index * 2 + 1] = (x * b0 + y * b1 + z * b2) * scale;
    }
    return { data: out, rowDim: 2, rowCount };
  }

  project_point(point: NDArray | readonly number[]): NDArray {
    const rows = isNDArray(point) ? point : array([...point]);
    const reshaped = pointRows(rows, 3);
    const single = array(reshaped.length > 0 ? [reshaped[0]!] : [[0, 0, 0]]) as NDArray;
    const projected = this.project_points(single);
    return array([pointRows(projected, 3)[0] ?? [0, 0, 0]]) as NDArray;
  }

  /** Project a single world point without any NDArray allocation. */
  project_point_flat(point: readonly number[]): [number, number] {
    const frame_center = this.frame_center;
    const x = (point[0] ?? 0) - (frame_center[0] ?? 0);
    const y = (point[1] ?? 0) - (frame_center[1] ?? 0);
    const z = (point[2] ?? 0) - (frame_center[2] ?? 0);
    const [r0, r1, r2] = this._rotation_rows;
    const rz = x * (r2[0] ?? 0) + y * (r2[1] ?? 0) + z * (r2[2] ?? 0);
    const focal = this.get_focal_distance();
    let factor: number;
    if (this.exponential_projection) {
      factor = rz < 0 ? focal / (focal - rz) : Math.exp(rz / focal);
    } else {
      factor = focal / (focal - rz);
      if (focal - rz < 0) factor = 10 ** 6;
    }
    const scale = factor * this.get_zoom();
    return [
      (x * (r0[0] ?? 0) + y * (r0[1] ?? 0) + z * (r0[2] ?? 0)) * scale,
      (x * (r1[0] ?? 0) + y * (r1[1] ?? 0) + z * (r1[2] ?? 0)) * scale,
    ];
  }

  override transform_points_pre_display(mobject: Mobject, points: NDArray): NDArray {
    const base = super.transform_points_pre_display(mobject, points);
    const fixed_orientation = this.fixed_orientation_mobjects.has(mobject);
    const fixed_in_frame = this.fixed_in_frame_mobjects.has(mobject);

    if (fixed_in_frame) return base;
    if (fixed_orientation) {
      const centerFunc = this.fixed_orientation_mobjects.get(mobject)!;
      const center = pointRows(centerFunc() as never, 3)[0] ?? [0, 0, 0];
      const new_center = pointRows(this.project_point(array(center)), 3)[0] ?? [0, 0, 0];
      const rows = pointRows(base, 3).map((row) =>
        row.map((value, index) => value + ((new_center[index] ?? 0) - (center[index] ?? 0))),
      );
      return array(rows.length > 0 ? rows : []) as NDArray;
    }
    return this.project_points(base);
  }

  /**
   * Flat counterpart of {@link transform_points_pre_display} used by the
   * Canvas2D draw path.  Returns `null` for fixed-in-frame / fixed-orientation
   * members (whose transform needs the full NDArray machinery), letting the
   * caller fall back to the regular path.
   */
  transform_points_pre_display_flat(mobject: Mobject, points: unknown): FlatProjectedPoints | null {
    if (this.fixed_in_frame_mobjects.has(mobject) || this.fixed_orientation_mobjects.has(mobject)) {
      return null;
    }
    return this.project_points_flat(points);
  }

  add_fixed_orientation_mobjects(
    ...args: Array<Mobject | Record<string, unknown>>
  ): void {
    let kwargs: Record<string, unknown> = {};
    let mobjects = args as Mobject[];
    const last = args[args.length - 1];
    if (
      last !== undefined &&
      !(last instanceof Mobject) &&
      typeof last === "object" &&
      !Array.isArray(last)
    ) {
      kwargs = last as Record<string, unknown>;
      mobjects = args.slice(0, -1) as Mobject[];
    }
    const use_static_center_func = Boolean(kwargs.use_static_center_func ?? false);
    const center_func = (kwargs.center_func as CenterFunc | undefined) ?? null;

    const get_static_center_func = (mobject: Mobject): CenterFunc => {
      const point = mobject.get_center();
      return () => point;
    };

    for (const mobject of mobjects) {
      let func: CenterFunc;
      if (center_func) func = center_func;
      else if (use_static_center_func) func = get_static_center_func(mobject);
      else func = () => mobject.get_center();
      for (const submob of mobject.get_family()) {
        this.fixed_orientation_mobjects.set(submob, func);
      }
    }
  }

  add_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    for (const mobject of extract_mobject_family_members(mobjects)) {
      this.fixed_in_frame_mobjects.add(mobject);
    }
  }

  remove_fixed_orientation_mobjects(...mobjects: Mobject[]): void {
    for (const mobject of extract_mobject_family_members(mobjects)) {
      this.fixed_orientation_mobjects.delete(mobject);
    }
  }

  remove_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    for (const mobject of extract_mobject_family_members(mobjects)) {
      this.fixed_in_frame_mobjects.delete(mobject);
    }
  }
}
