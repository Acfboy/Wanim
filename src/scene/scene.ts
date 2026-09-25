import { platform } from "node:os";
import { config } from "../_config/utils.js";
import { Mobject } from "../mobject/mobject.js";
import { Animation, prepare_animation } from "../animation/animation.js";

export type SceneInteractAction = SceneInteractRerun | string;

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
  interactive_mode = false;
  always_update_mobjects: boolean;

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

  get_mobject_family_members(): Mobject[] {
    return this.mobjects.flatMap((mobject) => mobject.get_family());
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

  on_key_press(symbol: number, _modifiers: number): void {
    const char = String.fromCodePoint(symbol);
    if (char === "q") this.interactive_mode = false;
    else this.key_to_function_map.get(char)?.();
  }

  on_key_release(symbol: number, modifiers: number): void {
    void symbol;
    void modifiers;
  }
}
