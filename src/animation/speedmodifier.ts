import { Animation, Wait, prepare_animation, type AnimationOptions, type RateFunction } from "./animation.js";
import { AnimationGroup } from "./composition.js";
import { type Mobject, type _AnimationBuilder } from "../mobject/mobject.js";

export interface ChangeSpeedOptions extends AnimationOptions {
  speedinfo?: Record<number, number>;
  affects_speed_updaters?: boolean;
}

export class ChangeSpeed extends Animation {
  anim: Animation;
  speedinfo: Record<number, number>;
  conditions: Array<(t: number) => boolean> = [];
  functions: Array<(t: number) => number> = [];
  speed_modifier: (x: number, init_speed: number, final_speed: number) => number;
  f_inv_1: (init_speed: number, final_speed: number) => number;
  affects_speed_updaters: boolean;
  t = 0;
  static dt = 0;
  static is_changing_dt = false;

  constructor(
    anim: Animation | _AnimationBuilder,
    speedinfo: Record<number, number> | ChangeSpeedOptions = {},
    rate_func?: RateFunction,
    affects_speed_updaters = true,
    options: AnimationOptions = {},
  ) {
    const optionKeys = Object.keys(speedinfo as Record<string, unknown>);
    const looksLikeSpeedinfo = optionKeys.length > 0
      && optionKeys.every((key) => Number.isFinite(Number(key)))
      && !("speedinfo" in (speedinfo as Record<string, unknown>));
    const resolved: ChangeSpeedOptions = looksLikeSpeedinfo
      ? { speedinfo: speedinfo as Record<number, number>, rate_func, affects_speed_updaters, ...options }
      : { ...options, ...(speedinfo as ChangeSpeedOptions) };
    if (rate_func !== undefined && looksLikeSpeedinfo) resolved.rate_func = rate_func;

    // NOTE: `this` is not yet initialized; mirror Python by preparing the
    // inner animation first via static helpers.
    let inner: Animation;
    if ((anim as Animation).constructor !== undefined && anim instanceof AnimationGroup) {
      const group = anim as AnimationGroup;
      const ctor = (anim as object).constructor as new (...args: unknown[]) => AnimationGroup;
      inner = new ctor(
        ...group.animations.map((item) => ChangeSpeed.setup(item)),
        {
          group: group.group,
          run_time: group.run_time,
          rate_func: group.rate_func,
          lag_ratio: (group as unknown as { lag_ratio?: number }).lag_ratio,
        },
      );
    } else {
      inner = ChangeSpeed.setup(anim as Animation | _AnimationBuilder);
    }

    const affects = resolved.affects_speed_updaters ?? affects_speed_updaters;
    if (affects) {
      if (ChangeSpeed.is_changing_dt) {
        throw new Error("Only one animation at a time can play that changes speed (dt) for ChangeSpeed updaters");
      }
      ChangeSpeed.is_changing_dt = true;
    }

    const outerRateFunc = resolved.rate_func ?? inner.rate_func;

    const speed_modifier = (x: number, init_speed: number, final_speed: number): number =>
      ((final_speed ** 2 - init_speed ** 2) * x ** 2) / 4 + init_speed * x;
    const f_inv_1 = (init_speed: number, final_speed: number): number => 2 / (init_speed + final_speed);

    const sourceSpeedinfo: Record<number, number> = { ...(resolved.speedinfo ?? {}) };
    if (!(0 in sourceSpeedinfo)) sourceSpeedinfo[0] = 1;
    if (!(1 in sourceSpeedinfo)) {
      const sorted = Object.entries(sourceSpeedinfo)
        .map(([node, speed]) => [Number(node), speed] as [number, number])
        .sort((a, b) => a[0] - b[0]);
      sourceSpeedinfo[1] = sorted[sorted.length - 1]?.[1] ?? 1;
    }
    // NOTE: plain-object key order is unreliable for numeric-like keys
    // ("1" enumerates before "0.3"), so keep an explicitly sorted list.
    const sortedEntries = Object.entries(sourceSpeedinfo)
      .map(([node, speed]) => [Number(node), speed] as [number, number])
      .sort((a, b) => a[0] - b[0]);
    const sortedSpeedinfo = Object.fromEntries(sortedEntries) as Record<number, number>;

    const scaledTotalTime = ChangeSpeed.computeScaledTotalTime(sortedSpeedinfo, f_inv_1);

    super(inner.mobject, {
      ...resolved,
      rate_func: outerRateFunc,
      run_time: scaledTotalTime * inner.run_time,
    });
    this.anim = inner;
    this.speed_modifier = speed_modifier;
    this.f_inv_1 = f_inv_1;
    this.speedinfo = sortedSpeedinfo;
    this.affects_speed_updaters = affects;
    if (affects) this.t = 0;

    const conditions: Array<(t: number) => boolean> = [];
    const functions: Array<(t: number) => number> = [];
    let prevnode = 0;
    let init_speed = sortedEntries[0]?.[1] ?? 1;
    let curr_time = 0;
    for (const [node, final_speed] of sortedEntries.slice(1)) {
      const dur = node - prevnode;
      const capturedTime = curr_time;
      const capturedInit = init_speed;
      const capturedFinal = final_speed;
      const capturedDur = dur;
      const capturedPrev = prevnode;
      const condition = (t: number): boolean => {
        const lower_bound = capturedTime / scaledTotalTime;
        const upper_bound = (capturedTime + f_inv_1(capturedInit, capturedFinal) * capturedDur) / scaledTotalTime;
        return lower_bound <= t && t <= upper_bound;
      };
      conditions.push(condition);
      const fn = (t: number): number =>
        speed_modifier((scaledTotalTime * t - capturedTime) / capturedDur, capturedInit, capturedFinal) * capturedDur
        + capturedPrev;
      functions.push(fn);
      curr_time += f_inv_1(init_speed, final_speed) * dur;
      prevnode = node;
      init_speed = final_speed;
    }
    this.conditions = conditions;
    this.functions = functions;

    const func = (t: number): number => {
      if (t === 1) ChangeSpeed.is_changing_dt = false;
      const mapped = this.rate_func(t);
      let new_t = 0;
      for (let index = 0; index < this.conditions.length; index += 1) {
        if (this.conditions[index](mapped)) {
          new_t = this.functions[index](mapped);
          break;
        }
      }
      if (this.affects_speed_updaters) {
        ChangeSpeed.dt = (new_t - this.t) * this.anim.run_time;
        this.t = new_t;
      }
      return new_t;
    };

    this.anim.set_rate_func(func);
  }

  static computeScaledTotalTime(
    speedinfo: Record<number, number>,
    inverse: (first: number, last: number) => number = (first, last) => 2 / (first + last),
  ): number {
    const entries = Object.entries(speedinfo)
      .map(([node, speed]) => [Number(node), speed] as const)
      .sort((a, b) => a[0] - b[0]);
    let previousNode = 0;
    let initialSpeed = entries[0]?.[1] ?? 1;
    let totalTime = 0;
    for (let index = 1; index < entries.length; index += 1) {
      const entry = entries[index];
      if (!entry) continue;
      const [node, finalSpeed] = entry;
      totalTime += (node - previousNode) * inverse(initialSpeed, finalSpeed);
      previousNode = node;
      initialSpeed = finalSpeed;
    }
    return totalTime;
  }

  static setup(anim: Animation | _AnimationBuilder): Animation {
    // NOTE: mirrors Python `type(anim) is Wait` (exact-type check, not isinstance).
    if ((anim as object).constructor === Wait) {
      const wait = anim as Wait;
      wait.interpolate = (_alpha: number): void => {
        wait.rate_func(_alpha);
      };
    }
    if (typeof (anim as _AnimationBuilder).build === "function" && !(anim instanceof Animation)) {
      return ((anim as unknown as { build: () => Animation }).build());
    }
    return prepare_animation(anim as Animation | Mobject);
  }

  setup(anim: Animation | _AnimationBuilder): Animation {
    return ChangeSpeed.setup(anim);
  }

  get_scaled_total_time(): number {
    return ChangeSpeed.computeScaledTotalTime(this.speedinfo, this.f_inv_1);
  }

  static add_updater(mobject: any, updateFunction: (...args: any[]) => any, index?: number, callUpdater = false): any {
    if (updateFunction.length >= 2) {
      mobject.add_updater((mob: any, dt: number) => updateFunction(mob, ChangeSpeed.is_changing_dt ? ChangeSpeed.dt : dt), index, callUpdater);
    } else mobject.add_updater(updateFunction, index, callUpdater);
    return mobject;
  }

  interpolate(alpha: number): void {
    this.anim.interpolate(alpha);
  }

  update_mobjects(dt: number): void {
    this.anim.update_mobjects(dt);
  }

  finish(): void {
    ChangeSpeed.is_changing_dt = false;
    this.anim.finish();
  }

  begin(): void {
    this.anim.begin();
  }

  clean_up_from_scene(scene: unknown): void {
    this.anim.clean_up_from_scene(scene);
  }

  _setup_scene(scene: unknown): void {
    this.anim._setup_scene(scene);
  }
}
