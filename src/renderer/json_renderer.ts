/**
 * Serialization of the per-frame VMobject display list into JSON records.
 *
 * Faithful port of `the original manim interpreter/json_renderer.py`: a Scene is run with
 * the normal manim machinery (compile_animation_data / begin_animations /
 * play_internal), but every frame the *complete* display list produced by
 * `Camera.get_mobjects_to_display` is serialized (bezier points + style)
 * instead of being rasterized.  The resulting `.mjson` document is consumed
 * by the original manim renderer to rebuild an mp4.
 *
 * A frame record is `{cam: {fc, fw, fh}, m: [mobject record, ...], repeat}`.
 * A mobject record is `{p, f, s, bs, sw, bsw, jt, cs, z, bi}`.
 *
 * NOTE: ThreeDCamera is not ported to wanim yet, so the `s3d/fif/fo` and 3D
 * camera keys are only emitted when the camera exposes the required methods
 * (see ).
 */
import { writeFileSync } from "node:fs";
import { config } from "../_config/utils.js";
import { Camera } from "../camera/camera.js";
import { CapStyleType, LineJointType } from "../constants.js";
import { NON_FINITE_GEOMETRY_HINT, hasNonFinite, type Mobject } from "../mobject/mobject.js";
import { AbstractImageMobject } from "../mobject/types/image_mobject.js";
import { PMobject } from "../mobject/types/point_cloud_mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { isNDArray } from "../utils/arrays.js";
import { EndSceneEarlyException } from "../utils/exceptions.js";
import type { Scene } from "../scene/scene.js";

const _ROUND = 6;

export interface JSONCairoRendererOptions {
  camera_class?: new (...args: any[]) => Camera;
  skip_animations?: boolean;
}

/**
 * Marker for numbers that are integers in the Python original (`z_index`,
 * `jt`, `cs`, `repeat`, meta dimensions, ...).  The custom serializer below
 * needs this distinction because Python `json.dumps` writes `854` for an int
 * but `854.0` for a float, and the Python renderer passes `repeat` straight
 * to `range()` (which rejects floats).
 */
export class PyInt {
  readonly value: number;

  constructor(value: number) {
    this.value = Math.trunc(Number(value));
  }
}

/** Unwrap a `PyInt` marker (or pass a plain number through). */
function intValue(value: unknown): number {
  return value instanceof PyInt ? value.value : Math.trunc(Number(value));
}

function plainValue(value: unknown): unknown {
  return isNDArray(value) ? value.toArray() : value;
}

/** Mirrors `np.round(np.asarray(arr, dtype=float), nd).tolist()`. */
function roundList(value: unknown, nd = _ROUND): unknown {
  const source = plainValue(value);
  if (!Array.isArray(source)) return roundNumber(Number(source), nd);
  return source.map((entry) => roundList(entry, nd));
}

/** np.round rounds ties to even (unlike JS Math.round). */
function roundNumber(value: number, nd: number): number {
  if (!Number.isFinite(value)) return value;
  const factor = 10 ** nd;
  const scaled = value * factor;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let rounded: number;
  if (diff > 0.5) rounded = floor + 1;
  else if (diff < 0.5) rounded = floor;
  else rounded = floor % 2 === 0 ? floor : floor + 1;
  return rounded / factor;
}

/** Python `repr(float)`: integral floats keep a trailing `.0`. */
function pythonFloatRepr(value: number): string {
  if (!Number.isFinite(value)) {
    if (Number.isNaN(value)) return "NaN";
    return value > 0 ? "Infinity" : "-Infinity";
  }
  if (Object.is(value, -0)) return "-0.0";
  if (Number.isInteger(value)) {
    if (Math.abs(value) < 1e16) return `${value}.0`;
    return value.toExponential().replace("e", "e+").replace("e+-", "e-");
  }
  return String(value);
}

/**
 * Minimal `json.dumps(..., separators=(",", ":"), ensure_ascii=False)`
 * equivalent with Python float repr.  Object key order is insertion order.
 */
export function jsonDumpsPython(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (value instanceof PyInt) return String(value.value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return pythonFloatRepr(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => jsonDumpsPython(entry)).join(",")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .map(([key, entry]) => `${JSON.stringify(key)}:${jsonDumpsPython(entry)}`);
    return `{${entries.join(",")}}`;
  }
  throw new TypeError(`Cannot serialize value of type ${typeof value}`);
}

function enumInt(value: unknown, table: Record<string, number>, label: string): number {
  if (typeof value === "number") return Math.trunc(value);
  const text = String(value);
  if (!Number.isNaN(Number(text)) && text.trim() !== "") return Math.trunc(Number(text));
  const resolved = table[text.toUpperCase()];
  if (typeof resolved === "number") return resolved;
  const error = new Error(`invalid ${label}: ${text}`);
  error.name = "ValueError";
  throw error;
}

function jointTypeValue(value: unknown): number {
  return enumInt(value, LineJointType as unknown as Record<string, number>, "LineJointType");
}

function capStyleValue(value: unknown): number {
  return enumInt(value, CapStyleType as unknown as Record<string, number>, "CapStyleType");
}

/** Whether `camera` behaves like a (not-yet-ported) `ThreeDCamera`. */
function isThreeDCamera(camera: unknown): camera is {
  reset_rotation_matrix(): void;
  get_focal_distance(): number;
  get_phi(): number;
  get_theta(): number;
  get_gamma(): number;
  get_zoom(): number;
  light_source: { points: unknown };
  shading_factor: number;
  should_apply_shading: boolean;
  exponential_projection: boolean;
  fixed_in_frame_mobjects: Set<unknown>;
  fixed_orientation_mobjects: Map<unknown, () => unknown>;
  frame_center: unknown;
} {
  return (
    camera !== null &&
    camera !== undefined &&
    typeof (camera as { reset_rotation_matrix?: unknown }).reset_rotation_matrix === "function" &&
    typeof (camera as { get_focal_distance?: unknown }).get_focal_distance === "function"
  );
}

function memberOf(collection: unknown, item: unknown): boolean {
  if (collection instanceof Set) return collection.has(item);
  if (Array.isArray(collection)) return collection.includes(item);
  return false;
}

/** Throw a descriptive non-finite error rather than writing a poisoned
 *  `.mjson` (a NaN in the point/rgba stream explodes only much later, in the
 *  Python renderer, as a cryptic failure or a truncated video). */
function assertFiniteSerialized(value: unknown, field: string, className: string): void {
  if (hasNonFinite(value)) {
    throw new Error(`non-finite ${field} in ${className}. ${NON_FINITE_GEOMETRY_HINT}`);
  }
}

/** Serialize a single VMobject's bezier data and style. */
export function serialize_vmobject(vmobject: VMobject, camera?: unknown): Record<string, unknown> {
  const className = vmobject.constructor.name;
  assertFiniteSerialized(vmobject.points, "points", className);
  const record: Record<string, unknown> = { p: roundList(vmobject.points) };
  const styleFields: Array<[unknown, string]> = [
    [vmobject.fill_rgbas, "f"],
    [vmobject.stroke_rgbas, "s"],
    [vmobject.background_stroke_rgbas, "bs"],
  ];
  for (const [value, short] of styleFields) {
    if (value !== null && value !== undefined) {
      assertFiniteSerialized(value, "rgba", className);
    }
    record[short] = value !== null && value !== undefined ? roundList(value) : null;
  }
  record.sw = Number(vmobject.get_stroke_width());
  record.bsw = Number(vmobject.get_stroke_width(true));
  record.jt = new PyInt(jointTypeValue(vmobject.joint_type));
  record.cs = new PyInt(capStyleValue(vmobject.cap_style));
  record.z = new PyInt(vmobject.z_index ?? 0);
  const backgroundImage = (vmobject as { background_image?: unknown }).background_image ?? null;
  if (backgroundImage !== null && typeof backgroundImage !== "string") {
    const error = new Error(
      "VMobjects colored using a raster background image cannot be serialized yet",
    );
    error.name = "NotImplementedError";
    throw error;
  }
  record.bi = backgroundImage;
  if (isThreeDCamera(camera)) {
    // TODO: unreachable until ThreeDCamera lands (see ).
    record.s3d = Boolean((vmobject as { shade_in_3d?: boolean }).shade_in_3d);
    record.fif = memberOf(camera.fixed_in_frame_mobjects, vmobject);
    const centerFunc = camera.fixed_orientation_mobjects.get(vmobject) ?? null;
    record.fo = centerFunc !== null ? roundList(centerFunc()) : null;
  }
  return record;
}

/** Serialize one full frame: camera geometry plus the display list. */
export function serialize_frame(camera: Camera, sceneMobjects: Iterable<Mobject>): Record<string, unknown> {
  if (isThreeDCamera(camera)) {
    // The z-ordering depends on the camera rotation matrix, which
    // ThreeDCamera.capture_mobjects would regenerate first.
    camera.reset_rotation_matrix();
  }
  const mobjects = camera.get_mobjects_to_display(sceneMobjects, true);
  const unsupported = new Set<string>();
  for (const mobject of mobjects) {
    if (mobject instanceof PMobject || mobject instanceof AbstractImageMobject) {
      unsupported.add(mobject.constructor.name);
    }
  }
  if (unsupported.size > 0) {
    const error = new Error(
      "PMobjects and image mobjects cannot be serialized, found: "
      + [...unsupported].sort().join(", "),
    );
    error.name = "NotImplementedError";
    throw error;
  }
  const vmobjects = mobjects.filter((mobject): mobject is VMobject => mobject instanceof VMobject);
  assertFiniteSerialized(camera.frame_center, "camera frame_center", camera.constructor.name);
  assertFiniteSerialized(camera.frame_width, "camera frame_width", camera.constructor.name);
  assertFiniteSerialized(camera.frame_height, "camera frame_height", camera.constructor.name);
  const cam: Record<string, unknown> = {
    fc: roundList(camera.frame_center),
    fw: Number(camera.frame_width),
    fh: Number(camera.frame_height),
  };
  if (isThreeDCamera(camera)) {
    cam["3d"] = true;
    cam.fd = Number(camera.get_focal_distance());
    cam.phi = Number(camera.get_phi());
    cam.theta = Number(camera.get_theta());
    cam.gamma = Number(camera.get_gamma());
    cam.zoom = Number(camera.get_zoom());
    cam.ls = roundList((camera.light_source.points as unknown[])[0]);
    cam.sh = Number(camera.shading_factor);
    cam.sa = Boolean(camera.should_apply_shading);
    cam.ep = Boolean(camera.exponential_projection);
  }
  return {
    cam,
    m: vmobjects.map((mobject) => serialize_vmobject(mobject, camera)),
  };
}

interface Section {
  skip_animations: boolean;
}

/**
 * Stand-in for `SceneFileWriter`: nothing is written to disk here, but it
 * exposes everything `Scene.render` / `next_section` / `add_subcaption` /
 * `add_sound` / `abort_encode_jobs` reach for.
 */
export class _DummyFileWriter {
  sections: Section[] = [{ skip_animations: false }];
  subcaptions: unknown[] = [];

  constructor(..._args: unknown[]) {}

  next_section(..._args: unknown[]): void {
    this.sections.push({ skip_animations: false });
  }

  is_already_cached(_hashInvocation: string | null): boolean {
    return false;
  }

  add_partial_movie_file(_hashAnimation: string | null): void {}

  begin_animation(_allowWrite = false, ..._args: unknown[]): void {}

  end_animation(_allowWrite = false, ..._args: unknown[]): void {}

  write_frame(..._args: unknown[]): void {}

  finish(): void {}

  save_image(_image: unknown): void {}

  abort_encode_jobs(_options: { reraise_encoder_failures?: boolean } = {}): void {}

  add_sound(..._args: unknown[]): void {}

  get image_file_path(): string {
    return "";
  }

  get movie_file_path(): string {
    return "";
  }

  get gif_file_path(): string {
    return "";
  }
}

/**
 * Runs the scene exactly like `CairoRenderer` does, but every frame the full
 * display list is serialized instead of being rasterized.
 */
export class JSONCairoRenderer {
  camera: Camera;
  num_plays = 0;
  time = 0;
  static_image: unknown = null;
  skip_animations: boolean;
  file_writer: _DummyFileWriter;
  frames: Array<Record<string, unknown>> = [];
  _pending_frame: Record<string, unknown> | null = null;
  output_path: string;
  private _original_skipping_status: boolean;
  private _lastFrameKey: string | null = null;

  constructor(outputPath: string, options: JSONCairoRendererOptions = {}) {
    const CameraClass = options.camera_class ?? Camera;
    this.camera = new CameraClass();
    this.skip_animations = options.skip_animations ?? false;
    this._original_skipping_status = this.skip_animations;
    this.file_writer = new _DummyFileWriter();
    this.output_path = outputPath;
  }

  init_scene(_scene: Scene): void {
    this.file_writer = new _DummyFileWriter(this, _scene.constructor.name);
  }

  play(scene: Scene, ...args: Array<unknown>): void {
    // Reset skip_animations to the original state.
    this.skip_animations = this._original_skipping_status;
    this.update_skipping_status();

    // `Scene.play` forwards any play-level options (run_time, lag_ratio,
    // ...) as a trailing plain object; `compile_animation_data` reads it,
    // so the args are passed through intact (mirrors Python's
    // `renderer.play(scene, *args, **kwargs)`).
    scene.compile_animation_data(...(args as Array<never>));

    if (this.skip_animations) {
      this.time += scene.duration;
    }
    this.file_writer.add_partial_movie_file(null);

    this.file_writer.begin_animation(!this.skip_animations);
    scene.begin_animations();

    // Save a static image, to avoid rendering non moving objects.
    this.save_static_frame_data(scene, scene.static_mobjects);

    if (scene.is_current_animation_frozen_frame()) {
      this.update_frame(scene, scene.moving_mobjects);
      this.freeze_current_frame(scene.duration);
    } else {
      scene.play_internal();
    }
    this.file_writer.end_animation(!this.skip_animations);

    this.num_plays += 1;
  }

  save_static_frame_data(_scene: Scene, _staticMobjects: Iterable<Mobject>): null {
    // Disable the static-image optimization: every frame must carry the
    // complete display list so the renderer can rebuild frames alone.
    this.static_image = null;
    return null;
  }

  update_frame(
    scene: Scene,
    _mobjects: Iterable<Mobject> | null = null,
    _includeSubmobjects = true,
    ignoreSkipping = true,
    ..._kwargs: unknown[]
  ): void {
    if (this.skip_animations && !ignoreSkipping) return;
    // Always serialize the *complete* scene state, never only the moving
    // mobjects: the static-image optimization is disabled.
    this._pending_frame = serialize_frame(this.camera, scene.mobjects);
  }

  render(scene: Scene, _time: number, _movingMobjects: Iterable<Mobject> | null = null): void {
    this.update_frame(scene, _movingMobjects);
    this.add_frame(null);
  }

  /**
   * Append the pending serialized display list, deduplicating identical
   * consecutive frames into a `repeat` count.
   */
  add_frame(_frame: unknown, numFrames = 1): void {
    const dt = 1 / this.camera.frame_rate;
    if (this.skip_animations) return;
    this.time += numFrames * dt;
    const record = this._pending_frame;
    if (record === null) {
      throw new Error("JSONCairoRenderer.add_frame called with no pending frame");
    }
    const key = jsonDumpsPython(record.cam) + "|" + jsonDumpsPython(record.m);
    if (this.frames.length > 0 && this._lastFrameKey === key) {
      const last = this.frames[this.frames.length - 1] as Record<string, unknown>;
      last.repeat = new PyInt(intValue(last.repeat) + numFrames);
      this._pending_frame = null;
      return;
    }
    record.repeat = new PyInt(numFrames);
    this.frames.push(record);
    this._lastFrameKey = key;
    this._pending_frame = null;
  }

  freeze_current_frame(duration: number): void {
    const dt = 1 / this.camera.frame_rate;
    this.add_frame(null, Math.trunc(duration / dt));
  }

  update_skipping_status(): void {
    const sections = this.file_writer.sections;
    if (sections.length > 0 && sections[sections.length - 1]!.skip_animations) {
      this.skip_animations = true;
    }
    if (config.save_last_frame) {
      this.skip_animations = true;
    }
    if (config.from_animation_number > 0 && this.num_plays < config.from_animation_number) {
      this.skip_animations = true;
    }
    if (config.upto_animation_number >= 0 && this.num_plays > config.upto_animation_number) {
      this.skip_animations = true;
      throw new EndSceneEarlyException();
    }
  }

  scene_finished(scene: Scene): void {
    if (this.num_plays) {
      this.file_writer.finish();
    } else if (config.write_to_movie) {
      config.save_last_frame = true;
      config.write_to_movie = false;
    } else {
      this.static_image = null;
      this.update_frame(scene);
    }

    if (config.save_last_frame) {
      this.static_image = null;
      this.update_frame(scene);
      // Commit the final frame so a `save_last_frame` render (and a scene with
      // no animations) still produces a serialized frame for the renderer.
      if (this._pending_frame !== null) {
        this.add_frame(null);
      }
    }
  }

  clear_screen(): void {
    this.frames = [];
    this._pending_frame = null;
    this.static_image = null;
    this._lastFrameKey = null;
  }

  write_output(): string {
    const document = {
      meta: {
        width: new PyInt(this.camera.pixel_width),
        height: new PyInt(this.camera.pixel_height),
        frame_rate: new PyInt(this.camera.frame_rate),
        background_color: this.camera.background_color.to_hex(),
        background_opacity: this.camera.background_opacity,
        num_frames: new PyInt(
          this.frames.reduce((sum, frame) => sum + intValue((frame as { repeat?: unknown }).repeat), 0),
        ),
        num_plays: new PyInt(this.num_plays),
      },
      frames: this.frames,
    };
    writeFileSync(this.output_path, jsonDumpsPython(document), { encoding: "utf-8" });
    return this.output_path;
  }
}
