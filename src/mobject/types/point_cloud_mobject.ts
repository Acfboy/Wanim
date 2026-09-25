import { array, type NDArray } from "numpy-ts";
import {
  DEFAULT_POINT_DENSITY_1D,
  DEFAULT_POINT_DENSITY_2D,
  DEFAULT_STROKE_WIDTH,
} from "../../constants.js";
import { interpolate } from "../../utils/bezier.js";
import { ManimColor, color_to_rgba, type ParsableManimColor } from "../../utils/color/core.js";
import { arange, isNDArray, pointRows } from "../../utils/arrays.js";
import { Mobject, type MobjectOptions } from "../mobject.js";

export interface PMobjectOptions extends MobjectOptions {
  stroke_width?: number;
}

function toRows(value: unknown, width: number): number[][] {
  if (isNDArray(value)) return pointRows(value, width);
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (Array.isArray(value[0])) return (value as unknown[][]).map((row) => (row as unknown[]).map(Number));
    return [[...(value as unknown[]).map(Number)]];
  }
  return pointRows(value, width);
}

function appendRows(current: NDArray, extra: number[][], width: number): NDArray {
  const base = pointRows(current, width);
  return array([...base, ...extra]) as NDArray;
}

export function rgba_to_color(rgba: number[] | readonly number[] | NDArray): ManimColor {
  const values = Array.isArray(rgba)
    ? [...rgba].map(Number)
    : pointRows(rgba, 4)[0] ?? [0, 0, 0, 1];
  return ManimColor.from_rgba(values);
}

export class PMobject extends Mobject {
  stroke_width: number;
  rgbas: NDArray;

  constructor(options: PMobjectOptions = {}) {
    const { stroke_width = DEFAULT_STROKE_WIDTH, ...rest } = options;
    super(rest);
    this.stroke_width = stroke_width;
    this.rgbas = array([]).reshape(0, 4) as NDArray;
    this.reset_points();
  }

  reset_points(): this {
    this.rgbas = array([]).reshape(0, 4) as NDArray;
    this.points = array([]).reshape(0, 3) as NDArray;
    return this;
  }

  get_array_attrs(): string[] {
    return [...super.get_array_attrs(), "rgbas"];
  }

  add_points(
    points: number[][] | NDArray,
    rgbas?: number[][] | NDArray | null,
    color?: ParsableManimColor | null,
    alpha = 1.0,
  ): this {
    const pointRowsValue = toRows(points, this.dim);
    const numNewPoints = pointRowsValue.length;
    this.points = appendRows(this.points, pointRowsValue, this.dim);
    let rgbaRows: number[][];
    if (rgbas === undefined || rgbas === null) {
      const base = color !== undefined && color !== null ? new ManimColor(color) : this.color;
      const rgba = pointRows(color_to_rgba(base, alpha), 4)[0] ?? [0, 0, 0, 1];
      rgbaRows = Array.from({ length: numNewPoints }, () => [...rgba]);
    } else {
      rgbaRows = toRows(rgbas, 4);
      if (rgbaRows.length !== pointRowsValue.length) {
        const error = new Error("points and rgbas must have same length");
        error.name = "ValueError";
        throw error;
      }
    }
    this.rgbas = appendRows(this.rgbas, rgbaRows, 4);
    return this;
  }

  fade_to(color: ParsableManimColor, alpha: number, family = true): this {
    // NOTE: mirrors Python `interpolate(self.rgbas, color_to_rgba(color),
    // alpha)` elementwise, preserving the (N,4) shape. The shared
    // bezier.interpolate() truncates rows to width 3 and collapses single
    // rows to 1-D, so the lerp is done explicitly here.
    const startRows = pointRows(this.rgbas, 4);
    const endRow = pointRows(color_to_rgba(color), 4)[0] ?? [0, 0, 0, 1];
    const out = startRows.map((row) =>
      row.map((value, index) => (1 - alpha) * value + alpha * (endRow[index] ?? 0)),
    );
    this.rgbas = array(out.length > 0 ? out : [[]]).reshape(out.length, 4) as NDArray;
    for (const mob of this.submobjects) (mob as PMobject).fade_to(color, alpha, family);
    return this;
  }

  get_all_rgbas(): NDArray {
    return this.get_merged_array("rgbas");
  }

  ingest_submobjects(): this {
    const attrs = this.get_array_attrs();
    for (const attr of attrs) {
      (this as unknown as Record<string, NDArray>)[attr] = this.get_merged_array(attr);
    }
    this.submobjects = [];
    return this;
  }

  get_color(): ManimColor {
    return rgba_to_color(pointRows(this.rgbas, 4)[0] ?? [0, 0, 0, 1]);
  }

  point_from_proportion(alpha: number): NDArray {
    const index = alpha * (this.get_num_points() - 1);
    const rows = this.points.toArray() as unknown[][];
    return array(rows[Math.floor(index)] ?? []) as NDArray;
  }
}

export interface Mobject1DOptions extends PMobjectOptions {
  density?: number;
}

export class Mobject1D extends PMobject {
  density: number;
  epsilon: number;

  constructor(options: Mobject1DOptions = {}) {
    const { density = DEFAULT_POINT_DENSITY_1D, ...rest } = options;
    super(rest);
    this.density = density;
    this.epsilon = 1.0 / this.density;
  }

  add_line(
    start: number[] | NDArray,
    end: number[] | NDArray,
    color?: ParsableManimColor | null,
  ): this {
    const startRow = toRows(start, 3)[0] ?? [0, 0, 0];
    const endRow = toRows(end, 3)[0] ?? [0, 0, 0];
    const diff = endRow.map((value, index) => value - (startRow[index] ?? 0));
    const length = Math.hypot(...diff);
    let points: number[][];
    if (length === 0) {
      points = [[...startRow]];
    } else {
      const epsilon = this.epsilon / length;
      points = arange(0, 1, epsilon).map((t) => {
        const mixed = interpolate(startRow, endRow, t);
        if (isNDArray(mixed)) return pointRows(mixed, 3)[0] ?? [0, 0, 0];
        if (Array.isArray(mixed)) return [...mixed].map(Number);
        return [Number(mixed), Number(mixed), Number(mixed)];
      });
    }
    this.add_points(points, null, color ?? null);
    return this;
  }
}

export interface Mobject2DOptions extends PMobjectOptions {
  density?: number;
}

export class Mobject2D extends PMobject {
  density: number;
  epsilon: number;

  constructor(options: Mobject2DOptions = {}) {
    const { density = DEFAULT_POINT_DENSITY_2D, ...rest } = options;
    super(rest);
    this.density = density;
    this.epsilon = 1.0 / this.density;
  }
}
