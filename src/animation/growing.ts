import type { NDArray } from "numpy-ts";
import { Transform, type TransformOptions } from "./transform.js";
import { PI } from "../constants.js";
import { spiral_path } from "../utils/paths.js";
import type { Mobject } from "../mobject/mobject.js";
import type { ParsableManimColor } from "../utils/color/core.js";

export interface GrowFromPointOptions extends TransformOptions {
  point_color?: ParsableManimColor | null;
}

function looksLikeOptions(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  return ["run_time", "rate_func", "lag_ratio", "remover", "introducer", "suspend_mobject_updating", "name", "path_func", "path_arc", "point_color"].some(
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

export interface GrowFromCenterOptions extends TransformOptions {
  point_color?: ParsableManimColor | null;
}

export class GrowFromCenter extends GrowFromPoint {
  constructor(
    mobject: Mobject,
    pointColorOrOptions: ParsableManimColor | null | GrowFromCenterOptions = null,
    options: GrowFromCenterOptions = {},
  ) {
    // NOTE: mirrors Python, which computes `point = mobject.get_center()`
    // before delegating; TS requires super() first, so the point is computed
    // here and threaded through.
    super(
      mobject,
      mobject.get_center(),
      pointColorOrOptions as ParsableManimColor | null | GrowFromPointOptions,
      options as GrowFromPointOptions,
    );
  }
}

export interface GrowFromEdgeOptions extends TransformOptions {
  point_color?: ParsableManimColor | null;
}

export class GrowFromEdge extends GrowFromPoint {
  constructor(
    mobject: Mobject,
    edge: readonly number[] | NDArray,
    pointColorOrOptions: ParsableManimColor | null | GrowFromEdgeOptions = null,
    options: GrowFromEdgeOptions = {},
  ) {
    // NOTE: mirrors Python, which computes `point =
    // mobject.get_critical_point(edge)` before delegating; see GrowFromCenter.
    super(
      mobject,
      mobject.get_critical_point(edge),
      pointColorOrOptions as ParsableManimColor | null | GrowFromPointOptions,
      options as GrowFromPointOptions,
    );
  }
}

export interface GrowArrowOptions extends TransformOptions {
  point_color?: ParsableManimColor | null;
}

export class GrowArrow extends GrowFromPoint {
  constructor(
    arrow: Mobject & { get_start(): number[] },
    pointColorOrOptions: ParsableManimColor | null | GrowArrowOptions = null,
    options: GrowArrowOptions = {},
  ) {
    // NOTE: mirrors Python, which computes `point = arrow.get_start()`
    // before delegating; see GrowFromCenter.
    super(
      arrow,
      arrow.get_start(),
      pointColorOrOptions as ParsableManimColor | null | GrowFromPointOptions,
      options as GrowArrowOptions,
    );
  }

  override create_starting_mobject(): Mobject {
    const startArrow = this.mobject.copy();
    // NOTE: mirrors Python `start_arrow.scale(0, scale_tips=True,
    // about_point=self.point)`; `self.point` is a point array here (computed
    // from `get_start` in the constructor), falling back to the mobject's
    // center if a Mobject was somehow stored.
    const aboutPoint = this.point instanceof Object && "get_center" in (this.point as object)
      ? (this.point as Mobject).get_center()
      : (this.point as readonly number[] | NDArray);
    (startArrow as unknown as {
      scale: (factor: number, scale_tips: boolean, about_point?: readonly number[] | NDArray) => unknown;
    }).scale(0, true, aboutPoint);
    if (this.point_color) {
      startArrow.set_color(this.point_color);
    }
    return startArrow;
  }
}

export interface SpinInFromNothingOptions extends GrowFromCenterOptions {
  angle?: number;
}

export class SpinInFromNothing extends GrowFromCenter {
  angle: number;

  constructor(
    mobject: Mobject,
    angle: number | SpinInFromNothingOptions = PI / 2,
    pointColorOrOptions: ParsableManimColor | null | SpinInFromNothingOptions = null,
    options: SpinInFromNothingOptions = {},
  ) {
    // NOTE: mirrors Python `(mobject, angle=PI/2, point_color=None,
    // **kwargs)`; `angle`/`point_color` may also arrive inside options bags.
    let resolvedAngle = PI / 2;
    let rest: SpinInFromNothingOptions = { ...options };
    if (typeof angle === "number") {
      resolvedAngle = angle;
    } else if (angle !== null && typeof angle === "object") {
      const { angle: a, ...r } = angle;
      if (a !== undefined) resolvedAngle = a;
      rest = { ...r, ...rest };
    }
    let resolvedColor: ParsableManimColor | null = null;
    if (
      pointColorOrOptions !== null &&
      typeof pointColorOrOptions === "object" &&
      !Array.isArray(pointColorOrOptions) &&
      ("angle" in (pointColorOrOptions as Record<string, unknown>) ||
        "point_color" in (pointColorOrOptions as Record<string, unknown>) ||
        looksLikeOptions(pointColorOrOptions))
    ) {
      const { angle: a, point_color: pc, ...r } = pointColorOrOptions as SpinInFromNothingOptions;
      if (a !== undefined) resolvedAngle = a;
      if (pc !== undefined) resolvedColor = (pc as ParsableManimColor | null) ?? null;
      rest = { ...r, ...rest };
    } else {
      resolvedColor = (pointColorOrOptions as ParsableManimColor | null) ?? null;
    }
    if (rest.angle !== undefined) {
      resolvedAngle = rest.angle;
      const { angle: _a, ...r } = rest;
      void _a;
      rest = r;
    }
    if (rest.point_color !== undefined && resolvedColor === null) {
      resolvedColor = (rest.point_color as ParsableManimColor | null) ?? null;
      const { point_color: _pc, ...r } = rest;
      void _pc;
      rest = r;
    }
    super(
      mobject,
      resolvedColor,
      { path_func: spiral_path(resolvedAngle), ...rest } as GrowFromCenterOptions,
    );
    this.angle = resolvedAngle;
  }
}
