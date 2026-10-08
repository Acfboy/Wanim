/**
 * `InteractiveThreeDScene`: an `InteractiveScene` whose camera is the manim
 * `ThreeDCamera` and whose background pointer/wheel gestures orbit and zoom it.
 *
 * The two parents can not be combined by inheritance (`InteractiveScene`
 * carries the storyboard/control machinery; `ThreeDScene` carries the camera
 * semantics), so the camera methods are *borrowed* from `ThreeDScene`: every
 * implementation there only touches the shared `Scene` surface (`camera`,
 * `renderer`, `add`/`remove`/`play`), which `InteractiveScene` provides too.
 * The thin wrappers below keep the exact Python signatures (and pin
 * `__argnames__`, like the rest of the public API) while avoiding a copy of
 * the camera logic.
 *
 * Interaction contract (matches `InteractiveScene`):
 *  - controls are hit-tested first; only a press that lands on the background
 *    starts an orbit, so sliders/switches keep working under a 3D camera;
 *  - gestures are only live while the blueprint phase is idle (animations
 *    never compete with user input);
 *  - headless hosts (`NullInteractiveHost`, Node) never deliver pointer/wheel
 *    events, so a run there is a single blueprint/construct pass as usual.
 *
 * Python usage:
 *
 * ```python
 * class Demo(InteractiveThreeDScene):
 *     def blueprint(self):
 *         axes = ThreeDAxes()
 *         sphere = Sphere(radius=1.5, color=BLUE)
 *         self.add(axes, sphere)
 *         self.orbit_sensitivity = 0.006   # optional tuning
 * ```
 */
import { DEGREES } from "../constants.js";
import type { Animation } from "../animation/animation.js";
import { ThreeDCamera } from "../camera/three_d_camera.js";
import { Mobject } from "../mobject/mobject.js";
import { InteractiveScene, type InteractiveSceneOptions } from "./interactive_scene.js";
import { ThreeDScene } from "./three_d_scene.js";

export type InteractiveThreeDSceneOptions = InteractiveSceneOptions;

/** Orbit keeps phi strictly inside `(0, PI)` so the camera never flips at a pole. */
export const ORBIT_MIN_PHI = 1e-3;
export const ORBIT_MAX_PHI = Math.PI - 1e-3;
/** Zoom range the background wheel may reach (avoids degenerate projection). */
export const ORBIT_MIN_ZOOM = 0.05;
export const ORBIT_MAX_ZOOM = 20;

/** Wheel `deltaMode` -> pixel scale: lines and pages are converted to pixels. */
const WHEEL_DELTA_PIXELS = [1, 16, 100];

function clamp(value: number, low: number, high: number): number {
  return value < low ? low : value > high ? high : value;
}

export class InteractiveThreeDScene extends InteractiveScene {
  ambient_camera_rotation: unknown = null;
  default_angled_camera_orientation_kwargs: Record<string, number> = {
    phi: 70 * DEGREES,
    theta: -135 * DEGREES,
  };

  /** Whether background drags orbit the camera (default `True`). */
  orbit_controls = true;
  /** Radians of phi/theta rotation per CSS pixel dragged. */
  orbit_sensitivity = 0.005;
  /** Zoom exponent per wheel pixel: `zoom *= exp(-delta * zoom_sensitivity)`. */
  zoom_sensitivity = 0.0015;

  private _orbiting = false;
  private _orbit_x = 0;
  private _orbit_y = 0;

  constructor(options: InteractiveThreeDSceneOptions = {}) {
    super({
      ...options,
      camera_class:
        options.camera_class ?? (ThreeDCamera as unknown as new (...args: any[]) => any),
    });
  }

  /** Allow background drags to orbit the camera (default). */
  enable_orbit_controls(): this {
    this.orbit_controls = true;
    return this;
  }

  /** Turn camera orbit/zoom off; controls keep working. */
  disable_orbit_controls(): this {
    this.orbit_controls = false;
    this._orbiting = false;
    return this;
  }

  // --- camera methods borrowed from ThreeDScene -----------------------------
  // Each implementation reads only the shared Scene surface, so calling it
  // with this instance as `this` is exactly the inherited behaviour.

  set_camera_orientation(
    phi: number | null = null,
    theta: number | null = null,
    gamma: number | null = null,
    zoom: number | null = null,
    focal_distance: number | null = null,
    frame_center: Mobject | number[] | readonly number[] | null = null,
    kwargs: Record<string, unknown> = {},
  ): void {
    ThreeDScene.prototype.set_camera_orientation.call(
      this as unknown as ThreeDScene,
      phi,
      theta,
      gamma,
      zoom,
      focal_distance,
      frame_center,
      kwargs,
    );
  }

  begin_ambient_camera_rotation(rate = 0.02, about = "theta"): void {
    ThreeDScene.prototype.begin_ambient_camera_rotation.call(
      this as unknown as ThreeDScene,
      rate,
      about,
    );
  }

  stop_ambient_camera_rotation(about = "theta"): void {
    ThreeDScene.prototype.stop_ambient_camera_rotation.call(this as unknown as ThreeDScene, about);
  }

  begin_3dillusion_camera_rotation(
    rate = 1,
    origin_phi: number | null = null,
    origin_theta: number | null = null,
  ): void {
    ThreeDScene.prototype.begin_3dillusion_camera_rotation.call(
      this as unknown as ThreeDScene,
      rate,
      origin_phi,
      origin_theta,
    );
  }

  stop_3dillusion_camera_rotation(): void {
    ThreeDScene.prototype.stop_3dillusion_camera_rotation.call(this as unknown as ThreeDScene);
  }

  move_camera(
    phi: number | null = null,
    theta: number | null = null,
    gamma: number | null = null,
    zoom: number | null = null,
    focal_distance: number | null = null,
    frame_center: Mobject | number[] | readonly number[] | null = null,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
  ): void {
    ThreeDScene.prototype.move_camera.call(
      this as unknown as ThreeDScene,
      phi,
      theta,
      gamma,
      zoom,
      focal_distance,
      frame_center,
      added_anims,
      kwargs,
    );
  }

  override get_moving_mobjects(...animations: Animation[]): Mobject[] {
    return ThreeDScene.prototype.get_moving_mobjects.call(
      this as unknown as ThreeDScene,
      ...animations,
    );
  }

  add_fixed_orientation_mobjects(...args: Array<Mobject | Record<string, unknown>>): void {
    ThreeDScene.prototype.add_fixed_orientation_mobjects.apply(
      this as unknown as ThreeDScene,
      args,
    );
  }

  add_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    ThreeDScene.prototype.add_fixed_in_frame_mobjects.apply(this as unknown as ThreeDScene, mobjects);
  }

  remove_fixed_orientation_mobjects(...mobjects: Mobject[]): void {
    ThreeDScene.prototype.remove_fixed_orientation_mobjects.apply(
      this as unknown as ThreeDScene,
      mobjects,
    );
  }

  remove_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    ThreeDScene.prototype.remove_fixed_in_frame_mobjects.apply(
      this as unknown as ThreeDScene,
      mobjects,
    );
  }

  set_to_default_angled_camera_orientation(kwargs: Record<string, number> = {}): void {
    ThreeDScene.prototype.set_to_default_angled_camera_orientation.call(
      this as unknown as ThreeDScene,
      kwargs,
    );
  }

  // --- background gestures --------------------------------------------------

  /** The scene camera when it really is a `ThreeDCamera`, else `null`. */
  private three_d_camera(): ThreeDCamera | null {
    const camera = this.camera as Partial<ThreeDCamera> | null | undefined;
    if (
      camera === null ||
      camera === undefined ||
      typeof camera.get_theta !== "function" ||
      typeof camera.set_theta !== "function" ||
      typeof camera.get_phi !== "function" ||
      typeof camera.set_phi !== "function"
    ) {
      return null;
    }
    return camera as ThreeDCamera;
  }

  protected override on_background_pointer_down(_point: number[], event?: PointerEvent): void {
    if (!this.orbit_controls || event === undefined) return;
    if (this.three_d_camera() === null) return;
    this._orbiting = true;
    this._orbit_x = event.clientX;
    this._orbit_y = event.clientY;
  }

  protected override on_background_pointer_move(_point: number[], event?: PointerEvent): void {
    if (!this._orbiting || event === undefined) return;
    const camera = this.three_d_camera();
    if (camera === null) {
      this._orbiting = false;
      return;
    }
    const dx = event.clientX - this._orbit_x;
    const dy = event.clientY - this._orbit_y;
    this._orbit_x = event.clientX;
    this._orbit_y = event.clientY;
    // Same convention as manim's OpenGL orbit controls: horizontal motion
    // turns theta, vertical motion tips phi, and the scene follows the
    // pointer.  Each event applies its own incremental delta, so the
    // sensitivity stays exact even when frames are dropped.
    if (dx !== 0) camera.set_theta(camera.get_theta() - dx * this.orbit_sensitivity);
    if (dy !== 0) {
      camera.set_phi(clamp(camera.get_phi() - dy * this.orbit_sensitivity, ORBIT_MIN_PHI, ORBIT_MAX_PHI));
    }
  }

  protected override on_background_pointer_up(_event?: PointerEvent): void {
    this._orbiting = false;
  }

  protected override on_background_wheel(event: WheelEvent): void {
    if (!this.orbit_controls) return;
    const camera = this.three_d_camera();
    if (camera === null) return;
    const scale = WHEEL_DELTA_PIXELS[event.deltaMode ?? 0] ?? 1;
    const delta = event.deltaY * scale;
    if (delta === 0) return;
    // Exponential so zooming is symmetric: scrolling up magnifies, down
    // shrinks, and equal and opposite scrolls cancel exactly.
    const factor = Math.exp(-delta * this.zoom_sensitivity);
    camera.set_zoom(clamp(camera.get_zoom() * factor, ORBIT_MIN_ZOOM, ORBIT_MAX_ZOOM));
  }

  protected override on_interaction_reset(): void {
    this._orbiting = false;
  }
}

// Pin the Python-facing parameter names: esbuild may rename colliding
// parameters when it hoists modules into the browser bundle, and py2js maps
// Python keyword arguments by reading these names at runtime (see
// `parseParamNames` in py2js/src/runtime_core.ts).
(InteractiveThreeDScene.prototype.set_camera_orientation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["phi", "theta", "gamma", "zoom", "focal_distance", "frame_center", "kwargs"];
(InteractiveThreeDScene.prototype.begin_ambient_camera_rotation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["rate", "about"];
(InteractiveThreeDScene.prototype.stop_ambient_camera_rotation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["about"];
(InteractiveThreeDScene.prototype.begin_3dillusion_camera_rotation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["rate", "origin_phi", "origin_theta"];
(InteractiveThreeDScene.prototype.move_camera as unknown as {
  __argnames__?: string[];
}).__argnames__ = [
  "phi",
  "theta",
  "gamma",
  "zoom",
  "focal_distance",
  "frame_center",
  "added_anims",
  "kwargs",
];
(InteractiveThreeDScene.prototype.set_to_default_angled_camera_orientation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["kwargs"];
