import { Animation, type AnimationOptions } from "./animation.js";

export interface ChangeSpeedOptions extends AnimationOptions {
  speedinfo?: Record<number, number>;
  affects_speed_updaters?: boolean;
}

function computeScaledTotalTime(speedinfo: Record<number, number>): number {
  const inverse = (first: number, last: number): number => 2 / (first + last);
  const entries = Object.entries(speedinfo).map(([node, speed]) => [Number(node), speed] as const).sort((a, b) => a[0] - b[0]);
  let previousNode = 0;
  let initialSpeed = entries[0]?.[1] ?? 1;
  let totalTime = 0;
  for (let index = 1; index < entries.length; index += 1) {
    const [node, finalSpeed] = entries[index];
    totalTime += (node - previousNode) * inverse(initialSpeed, finalSpeed);
    previousNode = node;
    initialSpeed = finalSpeed;
  }
  return totalTime;
}

export class ChangeSpeed extends Animation {
  anim: Animation;
  speedinfo: Record<number, number>;
  affects_speed_updaters: boolean;
  static dt = 0;
  static is_changing_dt = false;
  protected f_inv_1: (initial: number, final: number) => number;

  constructor(anim: Animation, optionsOrSpeedinfo: ChangeSpeedOptions | Record<number, number> = {}, rate_func?: (alpha: number) => number, affects_speed_updaters = true) {
    const optionKeys = Object.keys(optionsOrSpeedinfo as Record<string, unknown>);
    const isSpeedinfo = optionKeys.length > 0 && optionKeys.every((key) => Number.isFinite(Number(key))) && !("speedinfo" in optionsOrSpeedinfo);
    const options: ChangeSpeedOptions = isSpeedinfo ? { speedinfo: optionsOrSpeedinfo as Record<number, number>, rate_func, affects_speed_updaters } : optionsOrSpeedinfo as ChangeSpeedOptions;
    const sourceSpeedinfo = { ...(options.speedinfo ?? {}) };
    if (!(0 in sourceSpeedinfo)) sourceSpeedinfo[0] = 1;
    if (!(1 in sourceSpeedinfo)) sourceSpeedinfo[1] = Object.entries(sourceSpeedinfo).sort((a, b) => Number(a[0]) - Number(b[0])).at(-1)?.[1] ?? 1;
    const speedinfo = Object.fromEntries(Object.entries(sourceSpeedinfo).sort((a, b) => Number(a[0]) - Number(b[0]))) as Record<number, number>;
    const scaledTotalTime = computeScaledTotalTime(speedinfo);
    super(anim.mobject, { ...options, run_time: (options.run_time ?? anim.run_time) * scaledTotalTime });
    this.anim = anim;
    this.speedinfo = speedinfo;
    this.affects_speed_updaters = options.affects_speed_updaters ?? true;
    this.f_inv_1 = (first: number, last: number) => 2 / (first + last);
  }

  get_scaled_total_time(): number {
    let previousNode = 0;
    let initialSpeed = this.speedinfo[0];
    let totalTime = 0;
    const entries = Object.entries(this.speedinfo).map(([node, speed]) => [Number(node), speed] as const).sort((a, b) => a[0] - b[0]);
    for (let index = 1; index < entries.length; index += 1) {
      const [node, finalSpeed] = entries[index];
      const duration = node - previousNode;
      totalTime += duration * this.f_inv_1(initialSpeed, finalSpeed);
      previousNode = node;
      initialSpeed = finalSpeed;
    }
    return totalTime;
  }

  static add_updater(mobject: any, updateFunction: (...args: any[]) => any, index?: number, callUpdater = false): any {
    if (updateFunction.length >= 2) {
      mobject.add_updater((mob: any, dt: number) => updateFunction(mob, ChangeSpeed.is_changing_dt ? ChangeSpeed.dt : dt), index, callUpdater);
    } else mobject.add_updater(updateFunction, index, callUpdater);
    return mobject;
  }
}
