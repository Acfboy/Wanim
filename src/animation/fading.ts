import type { NDArray } from "numpy-ts";
import { Transform } from "./transform.js";
import type { AnimationOptions } from "./animation.js";
import { ORIGIN } from "../constants.js";
import { Group, Mobject } from "../mobject/mobject.js";
import { pointRows, scalePoint, subtractPoints } from "../utils/arrays.js";

export interface FadeOptions extends AnimationOptions {
  shift?: number[] | readonly number[] | NDArray | null;
  target_position?: number[] | readonly number[] | NDArray | Mobject | null;
  scale?: number;
}

function isOptionsBag(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function splitMobjectsAndOptions(
  args: Array<Mobject | FadeOptions>,
): [Mobject[], FadeOptions] {
  const last = args[args.length - 1];
  if (last !== undefined && isOptionsBag(last)) {
    return [args.slice(0, -1) as Mobject[], last as FadeOptions];
  }
  return [args as Mobject[], {}];
}

export class _Fade extends Transform {
  point_target: boolean;
  shift_vector: number[];
  scale_factor: number;

  constructor(...args: Array<Mobject | FadeOptions>) {
    const [mobjects, options] = splitMobjectsAndOptions(args);
    if (mobjects.length === 0) {
      const error = new Error("At least one mobject must be passed.");
      error.name = "ValueError";
      throw error;
    }
    const mobject = mobjects.length === 1 ? mobjects[0]! : new Group(...mobjects);
    const { shift, target_position, scale = 1, ...rest } = options;
    super(mobject, { ...rest });
    let point_target = false;
    let shiftVector: number[];
    if (shift === null || shift === undefined) {
      if (target_position !== null && target_position !== undefined) {
        const target = target_position instanceof Mobject
          ? target_position.get_center()
          : target_position;
        shiftVector = subtractPoints(
          pointRows(target, mobject.dim)[0] ?? [0, 0, 0],
          pointRows(mobject.get_center(), mobject.dim)[0] ?? [0, 0, 0],
          mobject.dim,
        );
        point_target = true;
      } else {
        shiftVector = [...ORIGIN];
      }
    } else {
      shiftVector = [...(pointRows(shift, mobject.dim)[0] ?? new Array(mobject.dim).fill(0))];
    }
    this.point_target = point_target;
    this.shift_vector = shiftVector;
    this.scale_factor = scale;
  }

  _create_faded_mobject(fadeIn: boolean): Mobject {
    const faded_mobject = this.mobject.copy();
    faded_mobject.fade(1);
    const direction_modifier = fadeIn && !this.point_target ? -1 : 1;
    faded_mobject.shift(scalePoint(this.shift_vector, direction_modifier, faded_mobject.dim));
    faded_mobject.scale(this.scale_factor);
    return faded_mobject;
  }
}

export class FadeIn extends _Fade {
  constructor(...args: Array<Mobject | FadeOptions>) {
    const [mobjects, options] = splitMobjectsAndOptions(args);
    super(...mobjects, { ...options, introducer: true });
  }

  override create_target(): Mobject {
    return this.mobject;
  }

  override create_starting_mobject(): Mobject {
    return this._create_faded_mobject(true);
  }
}

export class FadeOut extends _Fade {
  constructor(...args: Array<Mobject | FadeOptions>) {
    const [mobjects, options] = splitMobjectsAndOptions(args);
    super(...mobjects, { ...options, remover: true });
  }

  override create_target(): Mobject {
    return this._create_faded_mobject(false);
  }

  override clean_up_from_scene(scene: any): void {
    super.clean_up_from_scene(scene);
    this.interpolate(0);
  }
}
