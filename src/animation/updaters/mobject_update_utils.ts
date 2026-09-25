import { Mobject } from "../../mobject/mobject.js";
import { normalize } from "../../utils/space_ops.js";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { RIGHT } from "../../constants.js";
import { type NDArray } from "numpy-ts";

export type BoundMethod = ((...args: any[]) => any) & { originalMethod?: (...args: any[]) => any; mobject?: Mobject };

export function bind_mobject_method(mobject: Mobject, name: string): BoundMethod {
  const method = (mobject as unknown as Record<string, unknown>)[name];
  if (typeof method !== "function") throw new Error(`Mobject has no method '${name}'`);
  return Object.assign((method as (...args: any[]) => any).bind(mobject), { mobject, originalMethod: method as (...args: any[]) => any });
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

export function turn_animation_into_updater(animation: any, mobject: Mobject): Mobject {
  mobject.add_updater((_mob: Mobject) => animation.interpolate(animation.rate_func(0)));
  return mobject;
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
