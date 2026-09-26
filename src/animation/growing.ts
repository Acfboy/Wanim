import type { NDArray } from "numpy-ts";
import { Transform, type TransformOptions } from "./transform.js";
import type { Mobject } from "../mobject/mobject.js";
import type { ParsableManimColor } from "../utils/color/core.js";

export interface GrowFromPointOptions extends TransformOptions {
  point_color?: ParsableManimColor | null;
}

function looksLikeOptions(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  return ["run_time", "rate_func", "lag_ratio", "remover", "introducer", "suspend_mobject_updating", "name", "path_func", "path_arc"].some(
    (key) => key in (value as Record<string, unknown>),
  );
}

export class GrowFromPoint extends Transform {
  point: Mobject | readonly number[] | NDArray;
  point_color: ParsableManimColor | null;

  constructor(
    mobject: Mobject,
    point: Mobject | readonly number[] | NDArray,
    pointColorOrOptions: ParsableManimColor | null | GrowFromPointOptions = null,
    options: GrowFromPointOptions = {},
  ) {
    let resolvedColor: ParsableManimColor | null = null;
    let rest: TransformOptions = { ...options };
    if (looksLikeOptions(pointColorOrOptions)) {
      const opts = pointColorOrOptions as GrowFromPointOptions;
      const { point_color: _pc, ...r } = opts;
      resolvedColor = (_pc as ParsableManimColor | null | undefined) ?? null;
      rest = { ...(r as TransformOptions), ...rest };
    } else {
      resolvedColor = (pointColorOrOptions as ParsableManimColor | null) ?? null;
      const { point_color: pc, ...r } = rest as GrowFromPointOptions;
      if (resolvedColor === null && pc !== undefined) {
        resolvedColor = (pc as ParsableManimColor | null) ?? null;
        rest = r;
      } else if (pc !== undefined) {
        rest = r;
      }
    }
    super(mobject, { introducer: true, ...rest });
    this.point = point;
    this.point_color = resolvedColor;
  }

  override create_target(): Mobject {
    return this.mobject;
  }

  override create_starting_mobject(): Mobject {
    const start = super.create_starting_mobject();
    start.scale(0);
    start.move_to(this.point);
    if (this.point_color) {
      start.set_color(this.point_color);
    }
    return start;
  }
}
