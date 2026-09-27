import { array, identity, matmul, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { BLACK, get_shaded_rgb } from "../utils/color/core.js";
import { DEGREES, ORIGIN, RIGHT } from "../constants.js";
import { isNDArray, pointRows } from "../utils/arrays.js";
import { extract_mobject_family_members } from "../utils/family.js";
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
    return rgbas;
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
    const displayed = super.get_mobjects_to_display(
      mobjects,
      include_submobjects,
      excluded_mobjects,
    );
    const rotRows = pointRows(this.get_rotation_matrix(), 3);
    const zKey = (mob: Mobject): number => {
      // Python `hasattr(mob, "shade_in_3d")`: the Mobject proxy raises
      // AttributeError for missing attributes, so probe with `in`.
      const shade = "shade_in_3d" in mob
        ? Boolean((mob as { shade_in_3d?: boolean }).shade_in_3d)
        : false;
      if (!shade) return Infinity;
      // Assign a number to a three dimensional mobject based on how close it
      // is to the camera: np.dot(point, rot_matrix.T)[2] == dot(point, row 2).
      const center = mob.get_z_index_reference_point();
      const row = rotRows[2] ?? [0, 0, 1];
      return (
        (center[0] ?? 0) * (row[0] ?? 0) +
        (center[1] ?? 0) * (row[1] ?? 0) +
        (center[2] ?? 0) * (row[2] ?? 0)
      );
    };
    // Compute each depth key exactly once per mobject (Python's
    // `sorted(key=z_key)` is decorate-sort-undecorate).  Calling `zKey` from
    // inside the comparator instead would recompute the centre O(n log n)
    // times, which dominated 3D frame generation.
    const keyed = displayed.map((mob, index) => ({ mob, key: zKey(mob), index }));
    keyed.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : a.index - b.index));
    return keyed.map((entry) => entry.mob);
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
    this.rotation_matrix = this.generate_rotation_matrix();
  }

  get_rotation_matrix(): NDArray {
    return this.rotation_matrix;
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
    const rotRows = pointRows(this.get_rotation_matrix(), 3);

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

  project_point(point: NDArray | readonly number[]): NDArray {
    const rows = isNDArray(point) ? point : array([...point]);
    const reshaped = pointRows(rows, 3);
    const single = array(reshaped.length > 0 ? [reshaped[0]!] : [[0, 0, 0]]) as NDArray;
    const projected = this.project_points(single);
    return array([pointRows(projected, 3)[0] ?? [0, 0, 0]]) as NDArray;
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
