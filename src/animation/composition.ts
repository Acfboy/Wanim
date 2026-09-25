import { Animation, type AnimationOptions, type RateFunction, linear, prepare_animation } from "./animation.js";
import { Mobject, Group } from "../mobject/mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import { remove_list_redundancies } from "../utils/iterables.js";

export interface AnimationTiming {
  anim: Animation;
  start: number;
  end: number;
}

export type AnimationTimingTable = AnimationTiming[] & { start: number[]; end: number[]; anim: Animation[] };

export interface AnimationGroupOptions extends AnimationOptions {
  group?: Group | VGroup | Mobject;
  lag_ratio?: number;
}

function flattenAnimations(values: unknown[]): Animation[] {
  const result: Animation[] = [];
  for (const value of values) {
    if (value instanceof Animation) result.push(value);
    else if (Array.isArray(value)) result.push(...flattenAnimations(value));
    else if (value && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function") result.push(...flattenAnimations([...(value as Iterable<unknown>)]));
    else if (typeof value === "function") result.push(prepare_animation(value as () => Animation));
    else throw new TypeError("AnimationGroup only accepts Animation objects");
  }
  return result;
}

function timingTable(values: AnimationTiming[]): AnimationTimingTable {
  const table = values as AnimationTimingTable;
  return new Proxy(table, {
    get(target, property, receiver) {
      if (property === "start") return target.map((item) => item.start);
      if (property === "end") return target.map((item) => item.end);
      if (property === "anim") return target.map((item) => item.anim);
      return Reflect.get(target, property, receiver);
    },
  });
}

export class AnimationGroup extends Animation {
  animations: Animation[];
  group: Mobject;
  anims_with_timings: AnimationTimingTable = timingTable([]);
  anims_begun: boolean[] = [];
  anims_finished: boolean[] = [];
  max_end_time = 0;
  anim_group_time = 0;

  constructor(...args: any[]) {
    let options: AnimationGroupOptions = {};
    const last = args[args.length - 1];
    if (last && typeof last === "object" && !(last instanceof Animation) && !Array.isArray(last) && !(last instanceof Mobject) && typeof (last as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== "function") options = last as AnimationGroupOptions;
    const animationArgs = options === last ? args.slice(0, -1) : args;
    const prepared = flattenAnimations(animationArgs).map((animation) => prepare_animation(animation));
    const groupOption = options.group;
    const group = groupOption ?? new Group(...remove_list_redundancies(prepared.filter((animation) => !animation.is_introducer()).map((animation) => animation.mobject)));
    super(group, { ...options, rate_func: options.rate_func ?? linear, lag_ratio: options.lag_ratio ?? 0 });
    this.animations = prepared;
    this.group = group;
    this.run_time = this.init_run_time(options.run_time);
  }

  get_all_mobjects(): Mobject[] {
    return this.group.submobjects;
  }

  begin(): void {
    if (this.animations.length === 0) {
      const error = new Error(`Trying to play ${this.toString()} without animations, this is not supported. Please add at least one subanimation.`);
      error.name = "ValueError";
      throw error;
    }
    this.anim_group_time = 0;
    if (this.suspend_mobject_updating) this.group.suspend_updating();
    for (const animation of this.animations) animation.begin();
  }

  _setup_scene(scene: any): void {
    for (const animation of this.animations) animation._setup_scene(scene);
  }

  finish(): void {
    for (const animation of this.animations) animation.finish();
    this.anims_begun = this.anims_begun.map(() => true);
    this.anims_finished = this.anims_finished.map(() => true);
    if (this.suspend_mobject_updating) this.group.resume_updating();
  }

  clean_up_from_scene(scene: any): void {
    this.on_finish(scene);
    for (const animation of this.animations) {
      if (this.remover) animation.remover = this.remover;
      animation.clean_up_from_scene(scene);
    }
  }

  update_mobjects(dt: number): void {
    for (let index = 0; index < this.animations.length; index += 1) {
      if (this.anims_begun[index] && !this.anims_finished[index]) this.animations[index].update_mobjects(dt);
    }
  }

  init_run_time(runTime?: number): number {
    this.build_animations_with_timings();
    this.max_end_time = this.anims_with_timings.length === 0 ? 0 : Math.max(...this.anims_with_timings.map((item) => item.end));
    return runTime === undefined ? this.max_end_time : runTime;
  }

  build_animations_with_timings(): void {
    const runTimes = this.animations.map((animation) => animation.run_time);
    const numAnimations = runTimes.length;
    this.anims_with_timings = timingTable(Array.from({ length: numAnimations }, (_, index) => ({ anim: this.animations[index], start: 0, end: 0 })));
    this.anims_begun = Array.from({ length: numAnimations }, () => false);
    this.anims_finished = Array.from({ length: numAnimations }, () => false);
    if (numAnimations === 0) return;
    const lags = runTimes.slice(0, -1).map((runTime) => runTime * this.lag_ratio);
    this.anims_with_timings.forEach((timing, index) => {
      timing.start = index === 0 ? 0 : lags.slice(0, index).reduce((sum, value) => sum + value, 0);
      timing.end = timing.start + runTimes[index];
    });
  }

  interpolate(alpha: number): void {
    // Faithful port of Python AnimationGroup.interpolate: sub-alphas are
    // clamped into [0, 1] (forward: >1 -> 1; backward: <0 -> 0), zero
    // run_time anims get sub-alpha 1 forward / 0 backward, and only
    // ongoing animations are updated.
    const time = this.rate_func(alpha) * this.max_end_time;
    const backwards = time < this.anim_group_time;
    const newBegun: boolean[] = [];
    const newFinished: boolean[] = [];
    for (let index = 0; index < this.anims_with_timings.length; index += 1) {
      const timing = this.anims_with_timings[index];
      newBegun.push(time >= timing.start);
      newFinished.push(time > timing.end);
    }
    for (let index = 0; index < this.anims_with_timings.length; index += 1) {
      const timing = this.anims_with_timings[index];
      if ((this.anims_begun[index] || newBegun[index]) && (!this.anims_finished[index] || !newFinished[index])) {
        let runTime = timing.end - timing.start;
        const zeroRunTime = runTime === 0;
        if (zeroRunTime) runTime = 1;
        let subAlpha = (time - timing.start) / runTime;
        if (backwards) {
          if (subAlpha < 0 || zeroRunTime) subAlpha = 0;
        } else {
          if (subAlpha > 1 || zeroRunTime) subAlpha = 1;
        }
        timing.anim.interpolate(subAlpha);
      }
    }
    this.anim_group_time = time;
    this.anims_begun = newBegun;
    this.anims_finished = newFinished;
  }
}

export class Succession extends AnimationGroup {
  active_index = 0;
  active_animation: Animation | null = null;
  scene: unknown = null;

  override _setup_scene(scene: any): void {
    if (scene === null || scene === undefined) return;
    if (this.is_introducer()) {
      for (const animation of this.animations) {
        if (!animation.is_introducer() && animation.mobject !== null && animation.mobject !== undefined) {
          scene.add(animation.mobject);
        }
      }
    }
    this.scene = scene;
  }

  begin(): void {
    if (this.animations.length === 0) {
      const error = new Error("A Succession must contain at least one animation.");
      error.name = "ValueError";
      throw error;
    }
    this.active_index = 0;
    this.update_active_animation();
  }

  update_active_animation(): void {
    this.active_animation = this.animations[this.active_index] ?? null;
    if (!this.active_animation) return;
    this.active_animation.begin();
    const start = this.anims_with_timings[this.active_index]?.start ?? 0;
    const end = this.anims_with_timings[this.active_index]?.end ?? 0;
    this.anim_group_time = start;
    this.max_end_time = end;
  }

  next_animation(): void {
    this.active_animation?.finish();
    this.active_index += 1;
    if (this.active_index >= this.animations.length) {
      this.active_animation = null;
      this.anim_group_time = 0;
      this.max_end_time = 0;
    } else this.update_active_animation();
  }

  interpolate(alpha: number): void {
    if (!this.active_animation) return;
    const timing = this.anims_with_timings[this.active_index];
    if (!timing) return;
    const runTime = timing.end - timing.start;
    this.active_animation.interpolate(runTime === 0 ? 1 : (this.rate_func(alpha) * (timing.end - timing.start)) / runTime);
  }
}
