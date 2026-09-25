import { platform } from "node:os";
import { array, matmul, transpose, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { OUT, SHIFT_VALUE } from "../constants.js";
import { MethodWithArgs } from "../data_structures.js";
import { Mobject } from "../mobject/mobject.js";
import { pointRows } from "../utils/arrays.js";
import { cross, normalize, rotation_matrix } from "../utils/space_ops.js";
import { Animation, prepare_animation } from "../animation/animation.js";
import { AnimationGroup } from "../animation/composition.js";
import { DefaultSectionType } from "./section.js";

export type SceneInteractAction = SceneInteractRerun | SceneInteractContinue | MethodWithArgs | string;

/** Mirrors manim.utils.opengl.translation_matrix (4x4 homogeneous). */
function translation_matrix_4x4(x = 0, y = 0, z = 0): NDArray {
  return array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1],
  ]) as NDArray;
}

/** Mirrors manim.utils.opengl.z_rotation_matrix (4x4 homogeneous). */
function z_rotation_matrix_4x4(z = 0): NDArray {
  const c = Math.cos(z);
  const s = Math.sin(z);
  return array([
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]) as NDArray;
}

function vec3(value: number[] | readonly number[] | NDArray): number[] {
  return [...(pointRows(value, 3)[0] ?? [0, 0, 0])];
}

function mat44(value: NDArray): number[][] {
  const rows = pointRows(value, 4);
  while (rows.length < 4) rows.push([0, 0, 0, 0]);
  return rows.slice(0, 4);
}

function mat44Column(matrix: NDArray, column: number, length = 3): number[] {
  return mat44(matrix).slice(0, length).map((row) => row[column] ?? 0);
}

export class SceneInteractContinue {
  sender: string;

  constructor(sender: string) {
    this.sender = sender;
  }
}

export class SceneInteractRerun {
  sender: string;
  kwargs: Record<string, unknown>;

  constructor(sender: string, kwargs: Record<string, unknown> = {}) {
    this.sender = sender;
    this.kwargs = kwargs;
  }
}

export class RerunSceneHandler {
  queue: Array<SceneInteractAction> | { put: (action: SceneInteractAction) => void };

  constructor(
    queue: Array<SceneInteractAction> | { put: (action: SceneInteractAction) => void },
  ) {
    this.queue = queue;
  }

  on_modified(_event: unknown): void {
    const action = new SceneInteractRerun("file");
    if (Array.isArray(this.queue)) this.queue.push(action);
    else this.queue.put(action);
  }
}

export function load_module_into_namespace(
  module: Record<string, unknown>,
  namespace: Record<string, unknown>,
): void {
  for (const name in module) namespace[name] = (module as Record<string, unknown>)[name];
}

function formatG(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Number.isInteger(value) && Math.abs(value) < 1e21) return String(value);
  let text = value.toPrecision(6);
  if (text.includes("e") || text.includes("E")) {
    text = text.replace(/(\.\d*?)0+e/, "$1e").replace(/\.e/, "e");
  } else if (text.includes(".")) {
    text = text.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  }
  return text;
}

export class TimeProgression implements Iterable<number> {
  times: number[];
  description: string;
  total: number | null;
  leave: boolean;
  ascii: boolean | null;
  disabled: boolean;

  constructor(
    times: number[],
    description: string,
    total: number | null = null,
    leave = true,
    ascii: boolean | null = null,
    disabled = false,
  ) {
    this.times = times;
    this.description = description;
    this.total = total;
    this.leave = leave;
    this.ascii = ascii;
    this.disabled = disabled;
  }

  [Symbol.iterator](): IterableIterator<number> {
    return this.times[Symbol.iterator]();
  }
}

export class Scene {
  mobjects: Mobject[] = [];
  foreground_mobjects: Mobject[] = [];
  meshes: Mobject[] = [];
  animations: Animation[] = [];
  camera: any = null;
  renderer: any = null;
  key_to_function_map: Map<string, () => void> = new Map();
  mouse_press_callbacks: Array<() => void> = [];
  interactive_mode = false;
  quit_interaction = false;
  camera_target: number[] = [0, 0, 0];
  // NOTE: Python uses OpenGLPoint (an OpenGLMobject) here; the OpenGL
  // renderer is not ported yet, so plain Mobjects track the positions.
  mouse_point: Mobject = new Mobject();
  mouse_drag_point: Mobject = new Mobject();
  always_update_mobjects: boolean;
  skip_animation_preview = false;
  queue: Array<SceneInteractAction> = [];

  constructor(renderer: any = null, camera_class: new (...args: any[]) => any = class {}, always_update_mobjects = false) {
    this.renderer = renderer;
    this.camera = new camera_class();
    this.always_update_mobjects = always_update_mobjects;
  }

  add(...mobjects: Mobject[]): this {
    for (const mobject of mobjects) if (!this.mobjects.includes(mobject)) this.mobjects.push(mobject);
    return this;
  }

  remove(...mobjects: Mobject[]): this {
    const meshes_to_remove = new Set(mobjects);
    function lambda_function(mesh: Mobject): boolean {
      return !meshes_to_remove.has(mesh);
    }
    this.meshes = this.meshes.filter(lambda_function);
    for (const mobject of mobjects) {
      const index = this.mobjects.indexOf(mobject);
      if (index >= 0) this.mobjects.splice(index, 1);
      const foreground_index = this.foreground_mobjects.indexOf(mobject);
      if (foreground_index >= 0) this.foreground_mobjects.splice(foreground_index, 1);
    }
    return this;
  }

  clear(): this {
    this.mobjects = [];
    this.foreground_mobjects = [];
    return this;
  }

  next_section(
    name = "unnamed",
    section_type: string = DefaultSectionType.NORMAL,
    skip_animations = false,
  ): void {
    this.renderer.file_writer.next_section(name, section_type, skip_animations);
  }

  get_mobject_family_members(): Mobject[] {
    return this.mobjects.flatMap((mobject) => mobject.get_family());
  }

  get_moving_mobjects(...animations: Animation[]): Mobject[] {
    // NOTE (dependency pull-in): ported to support
    // ThreeDScene.get_moving_mobjects, which calls super().
    function collect_animation_mobjects(nested: Iterable<Animation>): Mobject[] {
      const result: Mobject[] = [];
      for (const anim of nested) {
        // Keep composition wrappers as conservative boundaries for the
        // static frame cache, in addition to their nested animations.
        result.push(...anim.mobject.get_family());
        if (anim instanceof AnimationGroup) {
          result.push(...collect_animation_mobjects(anim.animations));
        }
      }
      return result;
    }
    const animation_mobjects = new Set(collect_animation_mobjects(animations));
    const mobjects = this.get_mobject_family_members();
    const consider_moving = (mob: Mobject): boolean => {
      return (
        animation_mobjects.has(mob) ||
        mob.updaters.length > 0 ||
        this.foreground_mobjects.includes(mob) ||
        mob.submobjects.some((sub) => consider_moving(sub))
      );
    };
    const index = mobjects.findIndex((mob) => consider_moving(mob));
    return index === -1 ? [] : mobjects.slice(index);
  }

  play(...animations: (Animation | Mobject)[]): this {
    const prepared = animations.map((animation) => prepare_animation(animation));
    this.animations.push(...prepared);
    for (const animation of prepared) {
      animation._setup_scene(this);
      animation.begin();
      animation.finish();
      animation.clean_up_from_scene(this);
    }
    return this;
  }

  wait(_duration = 1): this { return this; }

  updaters: Array<(dt: number) => void> = [];

  setup(): void {}

  tear_down(): void {}

  construct(): void {}

  toString(): string {
    return this.constructor.name;
  }

  update_self(dt: number): void {
    for (const func of this.updaters) func(dt);
  }

  add_updater(func: (dt: number) => void): void {
    this.updaters.push(func);
  }

  remove_updater(func: (dt: number) => void): void {
    this.updaters = this.updaters.filter((candidate) => candidate !== func);
  }

  get_time_progression(
    run_time: number,
    description: string,
    n_iterations: number | null = null,
    override_skip_animations = false,
  ): TimeProgression {
    let times: number[];
    if ((this.renderer?.skip_animations ?? false) && !override_skip_animations) {
      times = [run_time];
    } else {
      const step = 1 / (config.frame_rate as number);
      times = [];
      // np.arange(0, run_time, step): values are i * step, not accumulated.
      for (let i = 0; i * step < run_time; i += 1) times.push(i * step);
    }
    return new TimeProgression(
      times,
      description,
      n_iterations,
      (config.get("progress_bar") as string) === "leave",
      platform() === "win32" ? true : null,
      (config.get("progress_bar") as string) === "none",
    );
  }

  static validate_run_time(
    run_time: number,
    method: (...args: unknown[]) => unknown,
    parameter_name = "run_time",
  ): number {
    const method_name = `${this.name}.${method.name}()`;
    if (run_time <= 0) {
      const error = new Error(
        `${method_name} has a ${parameter_name} of ${formatG(run_time)} <= 0 seconds which Manim cannot render. ` +
          `The ${parameter_name} must be a positive number.`,
      );
      error.name = "ValueError";
      throw error;
    }
    const fps = config.frame_rate as number;
    const seconds_per_frame = 1 / fps;
    if (run_time < seconds_per_frame) {
      console.warn(
        `The original ${parameter_name} of ${method_name}, ${formatG(run_time)} seconds, is too short for the current frame ` +
          `rate of ${formatG(fps)} FPS. Rendering with the shortest possible ${parameter_name} of ${formatG(seconds_per_frame)} seconds instead.`,
      );
      run_time = seconds_per_frame;
    }
    return run_time;
  }

  on_mouse_motion(point: number[] | readonly number[] | NDArray, d_point: number[] | readonly number[] | NDArray): void {
    const target = vec3(point);
    const delta = vec3(d_point);
    this.mouse_point.move_to(target);
    const pressedKeys = this.renderer?.pressed_keys as Set<number> | number[] | undefined;
    const hasShift = pressedKeys instanceof Set
      ? pressedKeys.has(SHIFT_VALUE)
      : Array.isArray(pressedKeys) && pressedKeys.includes(SHIFT_VALUE);
    if (hasShift) {
      const shift = [-delta[0], -delta[1], -delta[2]];
      shift[0] = (shift[0] ?? 0) * (this.camera.get_width() / 2);
      shift[1] = (shift[1] ?? 0) * (this.camera.get_height() / 2);
      const transform = this.camera.inverse_rotation_matrix as NDArray;
      const rotated = matmul(transpose(transform), array(shift)) as NDArray;
      this.camera.shift(vec3(rotated));
    }
  }

  on_mouse_scroll(point: number[] | readonly number[] | NDArray, offset: number[] | readonly number[] | NDArray): void {
    // NOTE: Python asserts isinstance(self.camera, OpenGLCamera); the
    // OpenGL renderer/camera is not ported yet and `camera` is `any`, so
    // the check is omitted and behavior binds to the real camera once it
    // lands.
    if (!config.use_projection_stroke_shaders) {
      const off = vec3(offset);
      const factor = 1 + Math.atan(-2.1 * (off[1] ?? 0));
      this.camera.scale(factor, this.camera_target);
    }
    this.mouse_scroll_orbit_controls(point, offset);
  }

  on_mouse_drag(
    point: number[] | readonly number[] | NDArray,
    d_point: number[] | readonly number[] | NDArray,
    buttons: number,
    modifiers: number,
  ): void {
    // NOTE: same OpenGLCamera note as on_mouse_scroll.
    const target = vec3(point);
    const delta = vec3(d_point);
    this.mouse_drag_point.move_to(target);
    if (buttons === 1) {
      this.camera.increment_theta(-(delta[0] ?? 0));
      this.camera.increment_phi(delta[1] ?? 0);
    } else if (buttons === 4) {
      const camera_x_axis = mat44Column(this.camera.model_matrix as NDArray, 0);
      const horizontal_shift_vector = camera_x_axis.map((value) => -(delta[0] ?? 0) * value);
      const outCross = cross([...OUT], camera_x_axis);
      const vertical_shift_vector = outCross.map((value) => -(delta[1] ?? 0) * value);
      const total_shift_vector = horizontal_shift_vector.map(
        (value, index) => value + (vertical_shift_vector[index] ?? 0),
      );
      this.camera.shift(total_shift_vector.map((value) => 1.1 * value));
    }
    this.mouse_drag_orbit_controls(point, d_point, buttons, modifiers);
  }

  on_key_press(symbol: number, modifiers: number): void {
    void modifiers;
    let char: string;
    try {
      char = String.fromCodePoint(symbol);
    } catch {
      console.warn("The value of the pressed key is too large.");
      return;
    }
    if (char === "r") {
      this.camera.to_default_state();
      this.camera_target = [0, 0, 0];
    } else if (char === "q") {
      this.quit_interaction = true;
    } else if (this.key_to_function_map.has(char)) {
      this.key_to_function_map.get(char)?.();
    }
  }

  on_key_release(symbol: number, modifiers: number): void {
    void symbol;
    void modifiers;
  }

  mouse_scroll_orbit_controls(point: number[] | readonly number[] | NDArray, offset: number[] | readonly number[] | NDArray): void {
    void point;
    const off = vec3(offset);
    const target = vec3(this.camera_target);
    const position = vec3(this.camera.get_position());
    const sign = Math.sign(off[1] ?? 0);
    const shiftVector = target.map((value, index) => 0.01 * (value - (position[index] ?? 0)) * sign);
    this.camera.model_matrix = matmul(
      translation_matrix_4x4(shiftVector[0] ?? 0, shiftVector[1] ?? 0, shiftVector[2] ?? 0),
      this.camera.model_matrix as NDArray,
    );
  }

  mouse_drag_orbit_controls(
    point: number[] | readonly number[] | NDArray,
    d_point: number[] | readonly number[] | NDArray,
    buttons: number,
    modifiers: number,
  ): void {
    void point;
    void modifiers;
    const delta = vec3(d_point);
    // Left click drag.
    if (buttons === 1) {
      // Translate to target the origin and rotate around the z axis.
      const negatedTarget = vec3(this.camera_target).map((value) => -value);
      this.camera.model_matrix = matmul(
        z_rotation_matrix_4x4(-(delta[0] ?? 0)),
        matmul(
          translation_matrix_4x4(negatedTarget[0] ?? 0, negatedTarget[1] ?? 0, negatedTarget[2] ?? 0),
          this.camera.model_matrix as NDArray,
        ) as NDArray,
      );
      // Rotation off of the z axis.
      const camera_position = vec3(this.camera.get_position());
      const camera_y_axis = mat44Column(this.camera.model_matrix as NDArray, 1);
      const axis_of_rotation = normalize(cross(camera_y_axis, camera_position));
      let rotation = rotation_matrix(delta[1] ?? 0, axis_of_rotation, true);
      const maximum_polar_angle = this.camera.maximum_polar_angle as number;
      const minimum_polar_angle = this.camera.minimum_polar_angle as number;
      const potential_camera_model_matrix = matmul(rotation, this.camera.model_matrix as NDArray) as NDArray;
      const potential_camera_location = mat44Column(potential_camera_model_matrix, 3);
      const potential_camera_y_axis = mat44Column(potential_camera_model_matrix, 1);
      const axisSign = (potential_camera_y_axis[2] ?? 0) !== 0
        ? Math.sign(potential_camera_y_axis[2] ?? 0)
        : 1;
      const potential_polar_angle = axisSign * Math.acos(
        (potential_camera_location[2] ?? 0) / Math.hypot(...potential_camera_location),
      );
      if (minimum_polar_angle <= potential_polar_angle && potential_polar_angle <= maximum_polar_angle) {
        this.camera.model_matrix = potential_camera_model_matrix;
      } else {
        const currentSign = (camera_y_axis[2] ?? 0) !== 0 ? Math.sign(camera_y_axis[2] ?? 0) : 1;
        const current_polar_angle = currentSign * Math.acos(
          (camera_position[2] ?? 0) / Math.hypot(...camera_position),
        );
        let polar_angle_delta: number;
        if (potential_polar_angle > maximum_polar_angle) {
          polar_angle_delta = maximum_polar_angle - current_polar_angle;
        } else {
          polar_angle_delta = minimum_polar_angle - current_polar_angle;
        }
        rotation = rotation_matrix(polar_angle_delta, axis_of_rotation, true);
        this.camera.model_matrix = matmul(rotation, this.camera.model_matrix as NDArray);
      }
      // Translate to target the original target.
      const target = vec3(this.camera_target);
      this.camera.model_matrix = matmul(
        translation_matrix_4x4(target[0] ?? 0, target[1] ?? 0, target[2] ?? 0),
        this.camera.model_matrix as NDArray,
      );
    } else if (buttons === 4) {
      // Right click drag.
      const camera_x_axis = mat44Column(this.camera.model_matrix as NDArray, 0);
      const horizontal_shift_vector = camera_x_axis.map((value) => -(delta[0] ?? 0) * value);
      const outCross = cross([...OUT], camera_x_axis);
      const vertical_shift_vector = outCross.map((value) => -(delta[1] ?? 0) * value);
      const total_shift_vector = horizontal_shift_vector.map(
        (value, index) => value + (vertical_shift_vector[index] ?? 0),
      );
      this.camera.model_matrix = matmul(
        translation_matrix_4x4(total_shift_vector[0] ?? 0, total_shift_vector[1] ?? 0, total_shift_vector[2] ?? 0),
        this.camera.model_matrix as NDArray,
      );
      this.camera_target = vec3(this.camera_target).map(
        (value, index) => value + (total_shift_vector[index] ?? 0),
      );
    }
  }

  check_interactive_embed_is_valid(): boolean {
    if (config.get("force_window")) return true;
    if (this.skip_animation_preview) {
      console.warn("Disabling interactive embed as 'skip_animation_preview' is enabled");
      return false;
    } else if (config.get("write_to_movie")) {
      console.warn("Disabling interactive embed as 'write_to_movie' is enabled");
      return false;
    } else if (config.get("format")) {
      console.warn(
        "Disabling interactive embed as '--format' is set as " + config.get("format"),
      );
      return false;
    } else if (!this.renderer?.window) {
      console.warn("Disabling interactive embed as no window was created");
      return false;
    } else if (config.get("dry_run")) {
      console.warn("Disabling interactive embed as dry_run is enabled");
      return false;
    }
    return true;
  }

  /**
   * Mirrors the nested `get_embedded_method(method_name)` factory inside
   * `interactive_embed`: returns a closure pushing
   * `MethodWithArgs(method, args, kwargs)` onto this scene's queue.
   * Extracted as a method so the closure body is directly testable
   * (it would otherwise be hidden inside `interactive_embed`, which
   * requires an IPython/OpenGL session).
   */
  get_embedded_method(methodName: string): (...args: unknown[]) => void {
    const method = (this as unknown as Record<string, (...args: unknown[]) => void>)[methodName];
    const embedded_method = (...args: unknown[]): void => {
      // NOTE: mirrors Python's MethodWithArgs(method, args, kwargs); the
      // TS interactive shell has no **kwargs channel, so kwargs is {}.
      this.queue.push(new MethodWithArgs(method, [...args], {}));
    };
    return embedded_method;
  }

  /**
   * Mirrors the nested `embedded_rerun(*args, **kwargs)` inside
   * `interactive_embed.ipython`: queues `SceneInteractRerun("keyboard")`
   * and exits the embedded shell.
   */
  create_embedded_rerun(shell: { exiter: () => void }): (...args: unknown[]) => void {
    const embedded_rerun = (..._args: unknown[]): void => {
      this.queue.push(new SceneInteractRerun("keyboard"));
      shell.exiter();
    };
    return embedded_rerun;
  }

  /**
   * Mirrors the nested `rerun_callback(sender, data)` inside
   * `_configure_pygui`: queues `SceneInteractRerun("gui")`.
   */
  create_rerun_callback(): (sender: unknown, data: unknown) => void {
    const rerun_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractRerun("gui"));
    };
    return rerun_callback;
  }

  /**
   * Mirrors the nested `continue_callback(sender, data)` inside
   * `_configure_pygui`: queues `SceneInteractContinue("gui")`.
   */
  create_continue_callback(): (sender: unknown, data: unknown) => void {
    const continue_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractContinue("gui"));
    };
    return continue_callback;
  }

  /**
   * Mirrors the nested `scene_selection_callback(sender, data)` inside
   * `_configure_pygui`: sets `config.scene_names` to the selected scene
   * and queues `SceneInteractRerun("gui")`. Python reads the selection
   * via `dpg.get_value(sender)`; the TS port accepts either a plain
   * string or a `{ get_value() }` handle since DearPyGUI is unavailable.
   */
  create_scene_selection_callback(): (sender: { get_value?: () => string } | string, data: unknown) => void {
    const scene_selection_callback = (sender: { get_value?: () => string } | string, _data: unknown): void => {
      const selected = typeof sender === "string"
        ? sender
        : sender.get_value?.() ?? "";
      config.scene_names = [selected];
      this.queue.push(new SceneInteractRerun("gui"));
    };
    return scene_selection_callback;
  }

  interactive_embed(): void {
    if (!this.check_interactive_embed_is_valid()) return;
    this.interactive_mode = true;

    const ipython = (
      shell: { exiter: () => void; call: (options: { local_ns: Record<string, unknown> }) => void },
      namespace: Record<string, unknown>,
    ): void => {
      const load_module_into_namespace = (
        module: Record<string, unknown>,
        target: Record<string, unknown>,
      ): void => {
        for (const name in module) target[name] = module[name];
      };

      // Mirrors `import manim` / `import manim.opengl` namespace injection.
      // There is no IPython kernel in the TypeScript runtime, so the
      // modules resolve to empty namespaces here.
      load_module_into_namespace({}, namespace);
      load_module_into_namespace({}, namespace);

      const embedded_rerun = this.create_embedded_rerun(shell);
      namespace["rerun"] = embedded_rerun;

      shell.call({ local_ns: namespace });
      this.queue.push(new SceneInteractContinue("keyboard"));
    };
    void ipython;

    const get_embedded_method = (methodName: string): ((...args: unknown[]) => void) => {
      return this.get_embedded_method(methodName);
    };
    void get_embedded_method;
    console.warn("interactive_embed requires an IPython terminal session, which is not available in the TypeScript runtime.");
  }

  _configure_pygui(update = true): void {
    // NOTE: DearPyGUI has no TypeScript equivalent; only the queue/config
    // behavior of the GUI callbacks is ported. The window construction itself
    // is out of scope for the interpreter target.
    const rerun_callback = this.create_rerun_callback();

    const continue_callback = this.create_continue_callback();

    const scene_selection_callback = this.create_scene_selection_callback();
    void rerun_callback;
    void continue_callback;
    void scene_selection_callback;
    void update;
    console.warn("_configure_pygui requires DearPyGUI, which is not available in the TypeScript runtime.");
  }
}
