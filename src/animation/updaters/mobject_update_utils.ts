import { Mobject } from "../../mobject/mobject.js";
import { normalize } from "../../utils/space_ops.js";
import { clip } from "../../utils/simple_functions.js";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { RIGHT } from "../../constants.js";
import { type NDArray } from "numpy-ts";

export type BoundMethod = ((...args: any[]) => any) & {
  originalMethod?: (...args: any[]) => any;
  mobject?: Mobject;
  __self__?: Mobject;
  __func__?: (...args: any[]) => any;
};

export function bind_mobject_method(mobject: Mobject, name: string): BoundMethod {
  const method = (mobject as unknown as Record<string, unknown>)[name];
  if (typeof method !== "function") throw new Error(`Mobject has no method '${name}'`);
  // NOTE: mirrors a CPython bound method: the facade carries `__self__` /
  // `__func__` so it passes `ApplyMethod.check_validity_of_input` (and the
  // `Scene.play` bound-method rejection) exactly like Python's
  // `mobject.method`. `mobject`/`originalMethod` are kept for `always` /
  // `f_always` (the port's historical spelling).
  const bound = (method as (...args: any[]) => any).bind(mobject) as BoundMethod;
  bound.__self__ = mobject;
  bound.__func__ = method as (...args: any[]) => any;
  bound.mobject = mobject;
  bound.originalMethod = method as (...args: any[]) => any;
  return bound;
}

export function assert_is_mobject_method(method: BoundMethod): Mobject {
  return assertMobjectMethod(method);
}

function assertMobjectMethod(method: BoundMethod): Mobject {
  const mobject = (method as BoundMethod & { mobject?: Mobject }).mobject;
  if (!(mobject instanceof Mobject)) throw new Error("Expected a mobject method");
  return mobject;
}

export function always(method: BoundMethod, ...args: any[]): Mobject {
  const mobject = assertMobjectMethod(method);
  const function_ = method.originalMethod ?? method;
  mobject.add_updater((mob: Mobject) => function_.call(mob, ...args));
  return mobject;
}

export function f_always(method: BoundMethod, ...arg_generators: (() => any)[]): Mobject {
  const mobject = assertMobjectMethod(method);
  const function_ = method.originalMethod ?? method;
  const updater = (mob: Mobject): void => {
    const args = arg_generators.map((generator) => generator());
    function_.call(mob, ...args);
  };
  mobject.add_updater(updater);
  return mobject;
}

export function always_redraw(func: () => Mobject): Mobject {
  const mob = func();
  mob.add_updater((mob: Mobject) => mob.become(func()));
  return mob;
}

/** Structural surface of Animation needed by turn_animation_into_updater
 * (mirrors the attributes/methods the Python updater touches). Kept
 * structural to avoid a runtime import cycle with animation.ts. */
export interface TurnableAnimation {
  mobject: Mobject;
  total_time: number;
  suspend_mobject_updating: boolean;
  begin(): void;
  get_run_time(): number;
  interpolate(alpha: number): void;
  update_mobjects(dt: number): void;
  finish(): void;
}

export function turn_animation_into_updater(
  animation: TurnableAnimation,
  cycle = false,
  delay = 0,
  kwargs: Record<string, unknown> = {},
): Mobject {
  // NOTE: mirrors Python, whose **kwargs are accepted but never used.
  void kwargs;
  const mobject = animation.mobject;
  animation.suspend_mobject_updating = false;
  animation.begin();
  animation.total_time = -delay;

  function update(m: Mobject, dt: number): void {
    if (animation.total_time >= 0) {
      const run_time = animation.get_run_time();

      // handle zero/negative runtime safely
      if (run_time <= 0) {
        // instantly snap to final state once, then remove updater
        animation.interpolate(1);
        animation.update_mobjects(dt);
        animation.finish();
        m.remove_updater(update);
        return;
      }

      const time_ratio = animation.total_time / run_time;
      let alpha: number;
      if (cycle) {
        alpha = time_ratio % 1;
      } else {
        alpha = clip(time_ratio, 0, 1);
        if (alpha >= 1) {
          animation.finish();
          m.remove_updater(update);
          return;
        }
      }
      animation.interpolate(alpha);
      animation.update_mobjects(dt);
    }
    animation.total_time += dt;
  }

  mobject.add_updater(update);
  return mobject;
}

export function cycle_animation(
  animation: TurnableAnimation,
  kwargs: Record<string, unknown> = {},
): Mobject {
  return turn_animation_into_updater(animation, true, 0, kwargs);
}

export function always_shift<M extends Mobject>(
  mobject: M,
  direction: readonly number[] | NDArray = RIGHT,
  rate = 0.1,
): M {
  const directionArray = isNDArray(direction)
    ? (pointRows(direction, 3)[0] ?? [0, 0, 0])
    : [...direction];
  const unit = normalize(directionArray);
  mobject.add_updater((m: Mobject, dt: number) => {
    m.shift(unit.map((value) => dt * rate * value));
  });
  return mobject;
}
