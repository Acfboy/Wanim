import { platform } from "node:os";
import { config } from "../_config/utils.js";
import { MethodWithArgs } from "../data_structures.js";
import { Mobject } from "../mobject/mobject.js";
import { Animation, prepare_animation } from "../animation/animation.js";
import { DefaultSectionType } from "./section.js";

export type SceneInteractAction = SceneInteractRerun | SceneInteractContinue | MethodWithArgs | string;

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
  interactive_mode = false;
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

      const embedded_rerun = (..._args: unknown[]): void => {
        this.queue.push(new SceneInteractRerun("keyboard"));
        shell.exiter();
      };
      namespace["rerun"] = embedded_rerun;

      shell.call({ local_ns: namespace });
      this.queue.push(new SceneInteractContinue("keyboard"));
    };
    void ipython;

    const get_embedded_method = (methodName: string): ((...args: unknown[]) => void) => {
      const method = (this as unknown as Record<string, (...args: unknown[]) => void>)[methodName];
      const embedded_method = (...args: unknown[]): void => {
        // NOTE: mirrors Python's MethodWithArgs(method, args, kwargs); the
        // TS interactive shell has no **kwargs channel, so kwargs is {}.
        this.queue.push(new MethodWithArgs(method, [...args], {}));
      };
      return embedded_method;
    };
    void get_embedded_method;
    console.warn("interactive_embed requires an IPython terminal session, which is not available in the TypeScript runtime.");
  }

  _configure_pygui(update = true): void {
    // NOTE: DearPyGUI has no TypeScript equivalent; only the queue/config
    // behavior of the GUI callbacks is ported. The window construction itself
    // is out of scope for the interpreter target.
    const rerun_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractRerun("gui"));
    };

    const continue_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractContinue("gui"));
    };

    const scene_selection_callback = (sender: { get_value?: () => string } | string, _data: unknown): void => {
      const selected = typeof sender === "string"
        ? sender
        : sender.get_value?.() ?? "";
      config.scene_names = [selected];
      this.queue.push(new SceneInteractRerun("gui"));
    };
    void rerun_callback;
    void continue_callback;
    void scene_selection_callback;
    void update;
    console.warn("_configure_pygui requires DearPyGUI, which is not available in the TypeScript runtime.");
  }
}
