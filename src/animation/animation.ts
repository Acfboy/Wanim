import { Mobject, Group, get_animation_override, type FunctionLike } from "../mobject/mobject.js";
import type { _AnimationBuilder } from "../mobject/mobject.js";
import { DEFAULT_ANIMATION_LAG_RATIO, DEFAULT_ANIMATION_RUN_TIME } from "../constants.js";
import { linear, smooth } from "../utils/rate_functions.js";

export { linear, smooth };

export type RateFunction = (alpha: number) => number;
export type AnimationInput = Mobject | null | undefined;
export interface AnimationOptions {
  mobject?: AnimationInput;
  lag_ratio?: number;
  run_time?: number;
  rate_func?: RateFunction;
  reverse_rate_function?: boolean;
  name?: string | null;
  remover?: boolean;
  suspend_mobject_updating?: boolean;
  introducer?: boolean;
  _on_finish?: (scene: any) => void;
  use_override?: boolean;
  [key: string]: unknown;
}

const animationOverrides = new WeakMap<FunctionLike, AnimationClass>();

export function getAnimationOverride(method: FunctionLike): AnimationClass | undefined {
  return animationOverrides.get(method) ?? (get_animation_override(method) as unknown as AnimationClass | undefined) ?? (method as FunctionLike & { _override_animation?: AnimationClass })._override_animation;
}

export type AnimationClass = new (...args: any[]) => Animation;

/** Per-class default kwargs installed by `set_default` (mirrors Python `partialmethod` wrapping). */
export function classDefaultsOf(target: unknown): Record<string, unknown> {
  return ((target as { _wanimDefaultKwargs?: Record<string, unknown> } | null | undefined)?._wanimDefaultKwargs) ?? {};
}

export function animationOverrideForMobject(
  mobject: Mobject,
  cls: unknown,
): ((...args: any[]) => Animation) | undefined {
  const mob = mobject as unknown as {
    animation_override_for?: (cls: unknown) => ((...args: any[]) => Animation) | undefined;
  };
  if (typeof mob.animation_override_for === "function") {
    return mob.animation_override_for(cls) as ((...args: any[]) => Animation) | undefined;
  }
  return undefined;
}

export class Animation {
  static DEFAULT_ANIMATION_RUN_TIME = DEFAULT_ANIMATION_RUN_TIME;
  static DEFAULT_ANIMATION_LAG_RATIO = DEFAULT_ANIMATION_LAG_RATIO;
  /** Mirrors Python `Animation._original__init__`: the unwrapped constructor. */
  static _original__init__: unknown = Animation;
  mobject!: Mobject;
  starting_mobject!: Mobject;
  lag_ratio!: number;
  rate_func!: RateFunction;
  reverse_rate_function!: boolean;
  name!: string | null;
  remover!: boolean;
  introducer!: boolean;
  suspend_mobject_updating!: boolean;
  on_finish!: (scene: any) => void;
  protected _run_time!: number;

  /**
   * Mirrors Python `Animation.__new__`: when `mobject` defines an animation
   * override for the class being constructed (and `use_override` is true),
   * the override animation is returned instead of a fresh instance.
   * Returns the override animation, or null when normal construction applies.
   */
  static __new__(
    mobject: AnimationInput = null,
    use_override = true,
    ...args: unknown[]
  ): Animation | null {
    void args;
    if (mobject instanceof Mobject && use_override) {
      const cls = this as unknown;
      const func = animationOverrideForMobject(mobject, cls);
      if (func !== undefined && func !== null) {
        return (func as (...callArgs: unknown[]) => Animation)(mobject);
      }
    }
    return null;
  }

  /** Mirrors Python `Animation.__init_subclass__`: records the unwrapped constructor. */
  static __init_subclass__(..._kwargs: unknown[]): void {
    void _kwargs;
    const cls = this as unknown as { _original__init__?: unknown };
    // NOTE: TS has no automatic subclass-creation hook; this records the
    // current constructor so `set_default()` with no arguments can restore
    // original defaults (mirroring `cls.__init__ = cls._original__init__`).
    if (cls._original__init__ === undefined || cls._original__init__ === Animation._original__init__) {
      cls._original__init__ = this;
    }
  }

  /**
   * Mirrors Python `Animation.set_default`: stores default keyword arguments
   * for this class (merged in the constructor). Called with no arguments,
   * the original defaults are restored.
   */
  static set_default(kwargs: Record<string, unknown> = {}): void {
    const ctor = this as unknown as {
      _wanimDefaultKwargs?: Record<string, unknown>;
      _original__init__?: unknown;
    };
    if (ctor._original__init__ === undefined) ctor._original__init__ = this;
    if (Object.keys(kwargs).length > 0) {
      ctor._wanimDefaultKwargs = { ...(ctor._wanimDefaultKwargs ?? {}), ...kwargs };
    } else {
      delete ctor._wanimDefaultKwargs;
    }
  }

  static setDefault(kwargs: Record<string, unknown> = {}): void {
    (this as unknown as typeof Animation).set_default(kwargs);
  }

  constructor(mobjectOrOptions: AnimationInput | AnimationOptions = null, options: AnimationOptions = {}) {
    const first = mobjectOrOptions as AnimationOptions;
    const isOptions = mobjectOrOptions !== null && typeof mobjectOrOptions === "object" && !(mobjectOrOptions instanceof Mobject) && ("mobject" in first || "run_time" in first || "lag_ratio" in first || "rate_func" in first || "remover" in first || "introducer" in first);
    let actualOptions = isOptions ? { ...(first as AnimationOptions), ...options } : options;
    const mobject = isOptions ? actualOptions.mobject ?? null : (mobjectOrOptions as AnimationInput);
    // Mirrors `Animation.__new__` override interception (Python `__new__`
    // runs before `__init__` on every construction).
    const useOverride = (actualOptions.use_override as boolean | undefined) ?? true;
    if (mobject instanceof Mobject && useOverride) {
      const cls = (new.target as unknown) ?? this.constructor;
      const func = animationOverrideForMobject(mobject, cls);
      if (func !== undefined && func !== null) {
        const { use_override: _ignored, ...restOptions } = actualOptions;
        return (func as (...callArgs: unknown[]) => Animation)(mobject, restOptions) as unknown as this;
      }
    }
    // Mirrors `set_default` (Python `partialmethod` wrapping): per-class
    // default kwargs apply underneath explicitly passed options.
    const clsDefaults = ((this.constructor as unknown as { _wanimDefaultKwargs?: Record<string, unknown> })._wanimDefaultKwargs) ?? {};
    actualOptions = { ...clsDefaults, ...actualOptions };
    this._typecheck_input(mobject);
    this.mobject = mobject ?? new Mobject();
    this.starting_mobject = new Mobject();
    this._run_time = DEFAULT_ANIMATION_RUN_TIME;
    this.run_time = actualOptions.run_time ?? DEFAULT_ANIMATION_RUN_TIME;
    this.rate_func = actualOptions.rate_func ?? smooth;
    this.reverse_rate_function = actualOptions.reverse_rate_function ?? false;
    this.name = actualOptions.name ?? null;
    this.remover = actualOptions.remover ?? false;
    this.introducer = actualOptions.introducer ?? false;
    this.suspend_mobject_updating = actualOptions.suspend_mobject_updating ?? true;
    this.lag_ratio = actualOptions.lag_ratio ?? DEFAULT_ANIMATION_LAG_RATIO;
    this.on_finish = actualOptions._on_finish ?? (() => undefined);
    for (const [key, value] of Object.entries(actualOptions)) {
      if (!["mobject", "run_time", "lag_ratio", "rate_func", "reverse_rate_function", "name", "remover", "introducer", "suspend_mobject_updating", "_on_finish", "use_override"].includes(key)) (this as unknown as Record<string, unknown>)[key] = value;
    }
  }

  get run_time(): number {
    return this._run_time;
  }

  set run_time(value: number) {
    if (value < 0) {
      const error = new Error(`The run_time of ${this.constructor.name} cannot be negative. The given value was ${value}.`);
      error.name = "ValueError";
      throw error;
    }
    this._run_time = value;
  }

  _typecheck_input(mobject: AnimationInput): void {
    if (mobject !== null && mobject !== undefined && !(mobject instanceof Mobject)) throw new TypeError("Animation only works on Mobjects");
  }

  create_starting_mobject(): Mobject {
    return this.mobject.copy();
  }

  begin(): void {
    this.starting_mobject = this.create_starting_mobject();
    if (this.suspend_mobject_updating) this.mobject.suspend_updating();
    this.interpolate(0);
  }

  finish(): void {
    this.interpolate(1);
    if (this.suspend_mobject_updating) this.mobject.resume_updating();
  }

  interpolate(alpha: number): void {
    this.interpolate_mobject(alpha);
  }

  interpolate_mobject(alpha: number): void {
    const families = this.get_all_families_zipped();
    for (let index = 0; index < families.length; index += 1) {
      const pair = families[index];
      this.interpolate_submobject(pair[0], pair[1], this.get_sub_alpha(alpha, index, families.length));
    }
  }

  interpolate_submobject(_submobject: Mobject, _startingSubmobject: Mobject, _alpha: number): void {}

  get_sub_alpha(alpha: number, index: number, numSubmobjects: number): number {
    const fullLength = (numSubmobjects - 1) * this.lag_ratio + 1;
    const value = alpha * fullLength;
    const lower = index * this.lag_ratio;
    return this.reverse_rate_function ? this.rate_func(1 - (value - lower)) : this.rate_func(value - lower);
  }

  get_all_mobjects(): Mobject[] {
    return [this.mobject, this.starting_mobject];
  }

  get_all_families_zipped(): Mobject[][] {
    // NOTE: mirrors Python `zip(*(mob.family_members_with_points() for mob
    // in self.get_all_mobjects()), strict=False)`: the zip runs over ALL
    // mobjects (subclasses such as FadeTransform extend that list), not just
    // the (mobject, starting_mobject) pair. Shorter families are padded with
    // the first family's member, matching the `?? mobject` convention used
    // by the subclass zips in this codebase.
    const families = this.get_all_mobjects().map((mob) => mob.family_members_with_points());
    const first = families[0] ?? [];
    return first.map((mobject, index) => families.map((family) => family[index] ?? mobject));
  }

  get_all_mobjects_to_update(): Mobject[] {
    return this.get_all_mobjects().filter((mobject) => mobject !== this.mobject);
  }

  update_mobjects(dt: number): void {
    for (const mobject of this.get_all_mobjects_to_update()) mobject.update(dt);
  }

  clean_up_from_scene(scene: any): void {
    this.on_finish(scene);
    if (this.is_remover()) scene?.remove(this.mobject);
  }

  _setup_scene(scene: any): void {
    if (scene === null || scene === undefined) return;
    if (this.is_introducer() && !scene.get_mobject_family_members?.().includes(this.mobject)) scene.add(this.mobject);
  }

  set_run_time(run_time: number): this {
    this.run_time = run_time;
    return this;
  }

  get_run_time(): number {
    return this.run_time;
  }

  set_rate_func(rateFunc: RateFunction): this {
    this.rate_func = rateFunc;
    return this;
  }

  get_rate_func(): RateFunction {
    return this.rate_func;
  }

  set_name(name: string): this {
    this.name = name;
    return this;
  }

  is_remover(): boolean {
    return this.remover;
  }

  is_introducer(): boolean {
    return this.introducer;
  }

  copy(): this {
    const result = Object.create(Object.getPrototypeOf(this)) as this;
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Mobject) (result as unknown as Record<string, unknown>)[key] = value.copy();
      else (result as unknown as Record<string, unknown>)[key] = value;
    }
    return result;
  }

  toString(): string {
    return this.name ?? `${this.constructor.name}(${this.mobject.toString()})`;
  }

  __str__(): string {
    return this.toString();
  }

  __repr__(): string {
    return this.toString();
  }
}

export interface WaitOptions extends AnimationOptions {
  run_time?: number;
  stop_condition?: (() => boolean) | null;
  frozen_frame?: boolean | null;
}

export class Wait extends Animation {
  duration: number;
  stop_condition: (() => boolean) | null;
  is_static_wait: boolean | null;

  constructor(run_time_or_options: number | WaitOptions = 1, options: WaitOptions = {}) {
    const first = run_time_or_options as WaitOptions;
    const isOptions = run_time_or_options !== null && typeof run_time_or_options === "object";
    const merged: WaitOptions = isOptions ? { ...first, ...options } : options;
    const run_time = isOptions ? (merged.run_time ?? 1) : (run_time_or_options as number);
    const stop_condition = merged.stop_condition ?? null;
    const frozen_frame = merged.frozen_frame ?? null;
    if (stop_condition && frozen_frame) {
      const error = new Error("A static Wait animation cannot have a stop condition.");
      error.name = "ValueError";
      throw error;
    }
    // NOTE: Python consumes run_time/stop_condition/frozen_frame/rate_func as
    // named params (only the remainder flows into Animation as **kwargs), so
    // they are stripped here to avoid stray attributes via the base-class
    // unknown-key copy loop.
    const { run_time: _rt, stop_condition: _sc, frozen_frame: _ff, ...rest } = merged;
    super(null, {
      ...rest,
      run_time,
      rate_func: merged.rate_func ?? linear,
    });
    // NOTE: mirrors Python `self.duration = run_time` (kept alongside the
    // Animation run_time machinery) and the OpenGL workaround below: the
    // Cairo Mobject has no shader wrappers, so an empty list is attached
    // dynamically the same way Python assigns the attribute.
    this.duration = run_time;
    this.stop_condition = stop_condition;
    this.is_static_wait = frozen_frame;
    (this.mobject as unknown as Record<string, unknown>)["shader_wrapper_list"] = [];
  }

  begin(): void {}
  finish(): void {}
  clean_up_from_scene(_scene: any): void {}
  update_mobjects(_dt: number): void {}
  interpolate(_alpha: number): void {}
}

export class Add extends Animation {
  constructor(...args: Array<Mobject | AnimationOptions>) {
    let options: AnimationOptions = {};
    const last = args[args.length - 1];
    if (last !== undefined && !(last instanceof Mobject) && typeof last === "object" && !Array.isArray(last)) {
      options = last as AnimationOptions;
      args = args.slice(0, -1);
    }
    const mobjects = args as Mobject[];
    // NOTE: mirrors Python `mobjects[0] if len(mobjects) == 1 else
    // Group(*mobjects)`: zero mobjects yields an empty Group.
    const mobject = mobjects.length === 1 ? mobjects[0] : new Group(...mobjects);
    super(mobject, { run_time: 0, introducer: true, ...options });
  }

  begin(): void {}
  finish(): void {}
  clean_up_from_scene(_scene: any): void {}
  update_mobjects(_dt: number): void {}
  interpolate(_alpha: number): void {}
}

export function prepare_animation(
  animation: Animation | _AnimationBuilder | Mobject | (() => Animation),
): Animation {
  // Mirrors Python: animation factories (_AnimationBuilder, incl. the OpenGL
  // variant which also exposes build()) are built first, an Animation passes
  // through unchanged, anything else raises TypeError.
  // NOTE: the Mobject/function fallbacks below are TS conveniences with no
  // Python counterpart (Python raises TypeError for both); they keep
  // historical Scene.play call sites working without a renderer.
  const candidate = animation as unknown as { build?: unknown };
  if (
    candidate !== null &&
    (typeof candidate === "object" || typeof candidate === "function") &&
    typeof candidate.build === "function"
  ) {
    return (candidate.build as () => Animation).call(animation);
  }
  if (animation instanceof Animation) return animation;
  if (typeof animation === "function") return (animation as () => Animation)();
  if (animation instanceof Mobject) return new Animation(animation);
  throw new TypeError(`Object ${String(animation)} cannot be converted to an animation`);
}

export function override_animation(animationClass: AnimationClass): (func: FunctionLike) => FunctionLike {
  function decorator(func: FunctionLike): FunctionLike {
    animationOverrides.set(func, animationClass);
    (func as FunctionLike & { _override_animation: AnimationClass })._override_animation = animationClass;
    return func;
  }
  return decorator;
}

export function animation_class_of(func: FunctionLike): AnimationClass | undefined {
  return (func as FunctionLike & { _override_animation?: AnimationClass })._override_animation;
}
