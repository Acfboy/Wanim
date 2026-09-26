import { array, type NDArray } from "numpy-ts";
import { pointRows } from "../../utils/arrays.js";
import { angle_of_vector } from "../../utils/space_ops.js";
import { DEFAULT_ARROW_TIP_LENGTH, PI } from "../../constants.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Circle } from "./arc.js";
import { Square } from "./polygram.js";
import { Triangle } from "./polygram.js";
import { isPlainOptionsBag } from "./polygram.js";
import type { ParsableManimColor } from "../../utils/color/core.js";

export class ArrowTip extends VMobject {
  constructor(options: VMobjectOptions = {}, ..._rest: unknown[]) {
    super(options);
    const error = new Error("Has to be implemented in inheriting subclasses.");
    error.name = "NotImplementedError";
    throw error;
  }

  get base(): NDArray {
    return tipBase(this);
  }

  get tip_point(): NDArray {
    return tipPoint(this);
  }

  get vector(): number[] {
    return tipVector(this);
  }

  get tip_angle(): number {
    return tipAngle(this);
  }

  get length(): number {
    return tipLength(this);
  }
}

/** Shared tip-geometry helpers (mirroring the ArrowTip properties) so that
 * tip classes using single-inheritance (e.g. extending Triangle/Circle
 * directly, standing in for Python's multiple inheritance) share behavior. */
export function tipBase(tip: VMobject): NDArray {
  return tip.point_from_proportion(0.5);
}

export function tipPoint(tip: VMobject): NDArray {
  const rows = pointRows(tip.points, tip.dim);
  return array(rows[0] ?? []) as NDArray;
}

export function tipVector(tip: VMobject): number[] {
  const tip_point = pointRows(tipPoint(tip), 3)[0] ?? [0, 0, 0];
  const base = pointRows(tipBase(tip), 3)[0] ?? [0, 0, 0];
  return tip_point.map((value, index) => value - (base[index] ?? 0));
}

export function tipAngle(tip: VMobject): number {
  return angle_of_vector(tipVector(tip)) as number;
}

export function tipLength(tip: VMobject): number {
  const v = tipVector(tip);
  return Math.sqrt(v.reduce((sum, value) => sum + value * value, 0));
}

export interface ArrowTriangleTipOptions extends VMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  length?: number;
  width?: number;
  start_angle?: number;
}

export class ArrowTriangleTip extends Triangle {
  constructor(
    fillOpacityOrOptions: number | ArrowTriangleTipOptions = 0,
    stroke_width = 3,
    length: number = DEFAULT_ARROW_TIP_LENGTH,
    width: number = DEFAULT_ARROW_TIP_LENGTH,
    start_angle: number = PI,
    options: ArrowTriangleTipOptions = {},
  ) {
    // Python: ArrowTriangleTip(fill_opacity=0, stroke_width=3, length=0.35,
    // width=0.35, start_angle=PI, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(fillOpacityOrOptions);
    const bag = isBag
      ? { ...(fillOpacityOrOptions as ArrowTriangleTipOptions), ...options }
      : options;
    const positionalFill = isBag ? undefined : (fillOpacityOrOptions as number);
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      length: lengthOption,
      width: widthOption,
      start_angle: startAngleOption,
      color: colorOption,
      ...rest
    } = bag;
    const resolved_length = lengthOption ?? length;
    const resolved_width = widthOption ?? width;
    super({
      color: colorOption ?? undefined,
      fill_opacity: fillOpacityOption ?? positionalFill ?? 0,
      stroke_width: strokeWidthOption ?? stroke_width,
      start_angle: startAngleOption ?? start_angle,
      ...rest,
    });
    this.width = resolved_width;
    this.stretch_to_fit_width(resolved_length);
    this.stretch_to_fit_height(resolved_width);
  }

  get base(): NDArray {
    return tipBase(this);
  }

  get tip_point(): NDArray {
    return tipPoint(this);
  }

  get vector(): number[] {
    return tipVector(this);
  }

  get tip_angle(): number {
    return tipAngle(this);
  }

  get length(): number {
    return tipLength(this);
  }
}

export class ArrowTriangleFilledTip extends ArrowTriangleTip {
  constructor(fill_opacity = 1, stroke_width = 0, options: ArrowTriangleTipOptions = {}) {
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      ...rest
    } = options;
    super(fillOpacityOption ?? fill_opacity, strokeWidthOption ?? stroke_width, DEFAULT_ARROW_TIP_LENGTH, DEFAULT_ARROW_TIP_LENGTH, PI, rest);
  }
}

export interface ArrowCircleTipOptions extends VMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  length?: number;
  start_angle?: number;
}

export class ArrowCircleTip extends Circle {
  start_angle: number;

  constructor(
    fillOpacityOrOptions: number | ArrowCircleTipOptions = 0,
    stroke_width = 3,
    length: number = DEFAULT_ARROW_TIP_LENGTH,
    start_angle: number = PI,
    options: ArrowCircleTipOptions = {},
  ) {
    // Python: ArrowCircleTip(fill_opacity=0, stroke_width=3, length=0.35,
    // start_angle=PI, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(fillOpacityOrOptions);
    const bag = isBag
      ? { ...(fillOpacityOrOptions as ArrowCircleTipOptions), ...options }
      : options;
    const positionalFill = isBag ? undefined : (fillOpacityOrOptions as number);
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      length: lengthOption,
      start_angle: _droppedStartAngle,
      color: colorOption,
      ...rest
    } = bag;
    void _droppedStartAngle;
    const resolved_length = lengthOption ?? length;
    super(null, {
      color: colorOption ?? undefined,
      fill_opacity: fillOpacityOption ?? positionalFill ?? 0,
      stroke_width: strokeWidthOption ?? stroke_width,
      ...rest,
    });
    // NOTE: mirrors Python, which assigns self.start_angle BEFORE
    // Circle.__init__() runs — but Circle.__init__() forwards
    // start_angle=0 to Arc.__init__(), overwriting it. The net Python
    // behavior is start_angle == 0; since TS must call super() first,
    // assigning here would diverge (PI instead of 0), so the quirk is
    // reproduced by keeping Arc's default. The parameter is still
    // accepted for signature compatibility.
    void start_angle;
    this.start_angle = 0;
    this.width = resolved_length;
    this.stretch_to_fit_height(resolved_length);
  }

  get base(): NDArray {
    return tipBase(this);
  }

  get tip_point(): NDArray {
    return tipPoint(this);
  }

  get vector(): number[] {
    return tipVector(this);
  }

  get tip_angle(): number {
    return tipAngle(this);
  }

  get length(): number {
    return tipLength(this);
  }
}

export class ArrowCircleFilledTip extends ArrowCircleTip {
  constructor(fill_opacity = 1, stroke_width = 0, options: ArrowCircleTipOptions = {}) {
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      ...rest
    } = options;
    super(fillOpacityOption ?? fill_opacity, strokeWidthOption ?? stroke_width, DEFAULT_ARROW_TIP_LENGTH, PI, rest);
  }
}

export interface ArrowSquareTipOptions extends VMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  length?: number;
  start_angle?: number;
}

export class ArrowSquareTip extends Square {
  start_angle: number;

  constructor(
    fillOpacityOrOptions: number | ArrowSquareTipOptions = 0,
    stroke_width = 3,
    length: number = DEFAULT_ARROW_TIP_LENGTH,
    start_angle: number = PI,
    options: ArrowSquareTipOptions = {},
  ) {
    // Python: ArrowSquareTip(fill_opacity=0, stroke_width=3, length=0.35,
    // start_angle=PI, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(fillOpacityOrOptions);
    const bag = isBag
      ? { ...(fillOpacityOrOptions as ArrowSquareTipOptions), ...options }
      : options;
    const positionalFill = isBag ? undefined : (fillOpacityOrOptions as number);
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      length: lengthOption,
      start_angle: startAngleOption,
      color: colorOption,
      ...rest
    } = bag;
    const resolved_length = lengthOption ?? length;
    super(resolved_length, {
      color: colorOption ?? undefined,
      fill_opacity: fillOpacityOption ?? positionalFill ?? 0,
      stroke_width: strokeWidthOption ?? stroke_width,
      ...rest,
    });
    // NOTE: mirrors Python, which assigns self.start_angle before
    // Square.__init__() runs. The value is not read during construction,
    // so assigning after super() is behaviorally identical.
    this.start_angle = startAngleOption ?? start_angle;
    this.width = resolved_length;
    this.stretch_to_fit_height(resolved_length);
  }

  get base(): NDArray {
    return tipBase(this);
  }

  get tip_point(): NDArray {
    return tipPoint(this);
  }

  get vector(): number[] {
    return tipVector(this);
  }

  get tip_angle(): number {
    return tipAngle(this);
  }

  get length(): number {
    return tipLength(this);
  }
}

export class ArrowSquareFilledTip extends ArrowSquareTip {
  constructor(fill_opacity = 1, stroke_width = 0, options: ArrowSquareTipOptions = {}) {
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      ...rest
    } = options;
    super(fillOpacityOption ?? fill_opacity, strokeWidthOption ?? stroke_width, DEFAULT_ARROW_TIP_LENGTH, PI, rest);
  }
}

export interface StealthTipOptions extends VMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  length?: number;
  start_angle?: number;
}

export class StealthTip extends VMobject {
  start_angle: number;

  constructor(
    fillOpacityOrOptions: number | StealthTipOptions = 1,
    stroke_width = 3,
    length: number = DEFAULT_ARROW_TIP_LENGTH / 2,
    start_angle: number = PI,
    options: StealthTipOptions = {},
  ) {
    // Python: StealthTip(fill_opacity=1, stroke_width=3, length=0.175,
    // start_angle=PI, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(fillOpacityOrOptions);
    const bag = isBag
      ? { ...(fillOpacityOrOptions as StealthTipOptions), ...options }
      : options;
    const positionalFill = isBag ? undefined : (fillOpacityOrOptions as number);
    const {
      fill_opacity: fillOpacityOption,
      stroke_width: strokeWidthOption,
      length: lengthOption,
      start_angle: startAngleOption,
      color: colorOption,
      ...rest
    } = bag;
    super({
      fill_opacity: fillOpacityOption ?? positionalFill ?? 1,
      stroke_width: strokeWidthOption ?? stroke_width,
      color: colorOption ?? undefined,
      ...rest,
    });
    // NOTE: mirrors Python, which assigns self.start_angle before
    // VMobject.__init__() runs. The value is not read during construction,
    // so assigning after super() is behaviorally identical.
    this.start_angle = startAngleOption ?? start_angle;
    this.set_points_as_corners([
      [2, 0, 0],
      [-1.2, 1.6, 0],
      [0, 0, 0],
      [-1.2, -1.6, 0],
      [2, 0, 0],
    ]);
    this.scale((lengthOption ?? length) / this.length);
  }

  get base(): NDArray {
    return tipBase(this);
  }

  get tip_point(): NDArray {
    return tipPoint(this);
  }

  get vector(): number[] {
    return tipVector(this);
  }

  get tip_angle(): number {
    return tipAngle(this);
  }

  get length(): number {
    // NOTE: unlike other tips, the stealth length is the height of the
    // encompassing triangle (otherwise the tip scales too large).
    const v = tipVector(this);
    return Math.sqrt(v.reduce((sum, value) => sum + value * value, 0)) * 1.6;
  }
}

export type ArrowTipOptions = VMobjectOptions & {
  color?: ParsableManimColor;
};

// Register the default tip shape for TipableVMobject.get_unpositioned_tip
// (mirrors the lazy `from manim.mobject.geometry.tips import
// ArrowTriangleFilledTip` inside the Python method, while avoiding a
// static module cycle between arc.js and tips.js).
import { set_default_tip_shape } from "./arc.js";
set_default_tip_shape(ArrowTriangleFilledTip);
