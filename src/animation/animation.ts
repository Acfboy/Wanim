import { Mobject, Group, get_animation_override, type FunctionLike } from "../mobject/mobject.js";
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

export class Animation {
  static DEFAULT_ANIMATION_RUN_TIME = DEFAULT_ANIMATION_RUN_TIME;
  static DEFAULT_ANIMATION_LAG_RATIO = DEFAULT_ANIMATION_LAG_RATIO;
  mobject: Mobject;
  starting_mobject: Mobject;
  lag_ratio: number;
  rate_func: RateFunction;
  reverse_rate_function: boolean;
  name: string | null;
  remover: boolean;
  introducer: boolean;
  suspend_mobject_updating: boolean;
  on_finish: (scene: any) => void;
  protected _run_time: number;

  constructor(mobjectOrOptions: AnimationInput | AnimationOptions = null, options: AnimationOptions = {}) {
    const first = mobjectOrOptions as AnimationOptions;
    const isOptions = mobjectOrOptions !== null && typeof mobjectOrOptions === "object" && !(mobjectOrOptions instanceof Mobject) && ("mobject" in first || "run_time" in first || "lag_ratio" in first || "rate_func" in first || "remover" in first || "introducer" in first);
    const actualOptions = isOptions ? { ...(first as AnimationOptions), ...options } : options;
    const mobject = isOptions ? actualOptions.mobject ?? null : (mobjectOrOptions as AnimationInput);
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

  get_all_families_zipped(): [Mobject, Mobject][] {
    const first = this.mobject.family_members_with_points();
    const second = this.starting_mobject.family_members_with_points();
    return first.map((mobject, index) => [mobject, second[index] ?? mobject]);
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
}

export class Wait extends Animation {
  constructor(...mobjects: Mobject[]) {
    const mobject = mobjects.length === 1 ? mobjects[0] : new Group(...mobjects);
    super(mobject, { run_time: 0, introducer: true });
  }

  begin(): void {}
  finish(): void {}
  clean_up_from_scene(_scene: any): void {}
  update_mobjects(_dt: number): void {}
  interpolate(_alpha: number): void {}
}

export class Add extends Animation {
  constructor(...mobjects: Mobject[]) {
    // NOTE: mirrors Python `mobjects[0] if len(mobjects) == 1 else
    // Group(*mobjects)`: zero mobjects yields an empty Group (no raise).
    const mobject = mobjects.length === 1 ? mobjects[0] : new Group(...mobjects);
    super(mobject, { run_time: 0, introducer: true });
  }

  begin(): void {}
  finish(): void {}
  clean_up_from_scene(_scene: any): void {}
  update_mobjects(_dt: number): void {}
  interpolate(_alpha: number): void {}
}

export function prepare_animation(animation: Animation | Mobject | (() => Animation)): Animation {
  if (animation instanceof Animation) return animation;
  if (typeof animation === "function") return animation();
  if (animation instanceof Mobject) return new Animation(animation);
  throw new TypeError("Expected an Animation, Mobject, or animation factory.");
}

export type AnimationClass = new (...args: any[]) => Animation;

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
