import { array, type NDArray } from "numpy-ts";
import {
  DEFAULT_POINT_DENSITY_1D,
  DEFAULT_POINT_DENSITY_2D,
  DEFAULT_STROKE_WIDTH,
  ORIGIN,
  RIGHT,
  UP,
} from "../../constants.js";
import { interpolate } from "../../utils/bezier.js";
import { BLACK, ManimColor, color_to_rgba, rgba_to_color, type ParsableManimColor } from "../../utils/color/core.js";
import { arange, isNDArray, linspace, pointRows } from "../../utils/arrays.js";
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

  set_color(color: ParsableManimColor = "PURE_YELLOW", family = true): this {
    const rgba = pointRows(color_to_rgba(color), 4)[0] ?? [0, 0, 0, 1];
    const mobs = family ? this.family_members_with_points() : [this];
    for (const mob of mobs) {
      const pm = mob as PMobject;
      const count = pm.rgbas.shape[0] ?? 0;
      pm.rgbas = array(count > 0 ? Array.from({ length: count }, () => [...rgba]) : []).reshape(count, 4) as NDArray;
    }
    this.color = ManimColor.parse(color) as ManimColor;
    return this;
  }

  setColor(color: ParsableManimColor = "PURE_YELLOW", family = true): this {
    return this.set_color(color, family);
  }

  get_stroke_width(): number {
    return this.stroke_width;
  }

  getStrokeWidth(): number {
    return this.get_stroke_width();
  }

  set_stroke_width(width: number, family = true): this {
    const mobs = family ? this.family_members_with_points() : [this];
    for (const mob of mobs) (mob as PMobject).stroke_width = width;
    return this;
  }

  setStrokeWidth(width: number, family = true): this {
    return this.set_stroke_width(width, family);
  }

  filter_out(condition: (point: number[]) => boolean): this {
    for (const mob of this.family_members_with_points()) {
      const pm = mob as PMobject;
      const pointRowsValue = pointRows(pm.points, pm.dim);
      const rgbaRows = pointRows(pm.rgbas, 4);
      const kept_points = pointRowsValue.filter((point) => !condition(point));
      const kept_rgbas = rgbaRows.filter((_, index) => !condition(pointRowsValue[index] ?? []));
      pm.points = array(kept_points.length > 0 ? kept_points : []).reshape(kept_points.length, pm.dim) as NDArray;
      pm.rgbas = array(kept_rgbas.length > 0 ? kept_rgbas : []).reshape(kept_rgbas.length, 4) as NDArray;
    }
    return this;
  }

  filterOut(condition: (point: number[]) => boolean): this {
    return this.filter_out(condition);
  }

  thin_out(factor = 5): this {
    // NOTE: mirrors Python, which sizes the index range from
    // self.get_num_points() (not each family member's count).
    const num_points = this.get_num_points();
    const indices: number[] = [];
    for (let index = 0; index < num_points; index += factor) indices.push(index);
    for (const mob of this.family_members_with_points()) {
      mob.apply_over_attr_arrays((arr) => {
        const width = arr.shape.length > 1 ? (arr.shape[arr.shape.length - 1] ?? mob.dim) : mob.dim;
        const all = pointRows(arr, width);
        const selected = indices.map((index) => {
          if (index >= all.length) {
            const error = new Error(`index ${index} is out of bounds for axis 0 with size ${all.length}`);
            error.name = "IndexError";
            throw error;
          }
          return all[index] ?? [];
        });
        return array(selected.length > 0 ? selected : []).reshape(selected.length, width) as NDArray;
      });
    }
    return this;
  }

  thinOut(factor = 5): this {
    return this.thin_out(factor);
  }

  sort_points(predicate: (point: number[]) => number = (point) => point[0] ?? 0): this {
    for (const mob of this.family_members_with_points()) {
      const pointRowsValue = pointRows((mob as PMobject).points, mob.dim);
      const keys = pointRowsValue.map((point) => predicate(point));
      const indices = keys.map((_, index) => index).sort((a, b) => (keys[a] ?? 0) - (keys[b] ?? 0));
      mob.apply_over_attr_arrays((arr) => {
        const width = arr.shape.length > 1 ? (arr.shape[arr.shape.length - 1] ?? mob.dim) : mob.dim;
        const all = pointRows(arr, width);
        return array(indices.length > 0 ? indices.map((index) => all[index] ?? []) : []).reshape(indices.length, width) as NDArray;
      });
    }
    return this;
  }

  sortPoints(predicate: (point: number[]) => number = (point) => point[0] ?? 0): this {
    return this.sort_points(predicate);
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

export interface PointCloudDotOptions extends Mobject1DOptions {
  center?: number[] | readonly number[] | NDArray;
  radius?: number;
  color?: ParsableManimColor;
}

export class PointCloudDot extends Mobject1D {
  radius: number;

  constructor(
    center: number[] | readonly number[] | NDArray = [...ORIGIN],
    radius = 2.0,
    stroke_width = 2,
    density = DEFAULT_POINT_DENSITY_1D,
    color: ParsableManimColor = "PURE_YELLOW",
    options: PointCloudDotOptions = {},
  ) {
    const { center: _droppedCenter, radius: _droppedRadius, density: _droppedDensity, color: _droppedColor, stroke_width: _droppedStroke, ...rest } = options;
    void _droppedCenter;
    void _droppedRadius;
    void _droppedDensity;
    void _droppedColor;
    void _droppedStroke;
    super({ stroke_width, density, color, ...rest });
    this.radius = radius;
    this.epsilon = 1.0 / density;
    // NOTE: the base Mobject constructor already called generate_points(),
    // but at that time radius/epsilon were not yet assigned (TS requires
    // super() first, whereas Python sets them before super().__init__()).
    // Regenerate now that the fields are available.
    this.init_points();
    this.shift(center);
  }

  init_points(): this {
    this.reset_points();
    this.generate_points();
    return this;
  }

  generate_points(): this {
    const rings: number[][] = [];
    for (const r of arange(this.epsilon, this.radius, this.epsilon)) {
      const count = Math.trunc((2 * Math.PI * (r + this.epsilon)) / this.epsilon);
      for (const theta of linspace(0, 2 * Math.PI, count)) {
        rings.push([
          r * (Math.cos(theta) * (RIGHT[0] ?? 0) + Math.sin(theta) * (UP[0] ?? 0)),
          r * (Math.cos(theta) * (RIGHT[1] ?? 0) + Math.sin(theta) * (UP[1] ?? 0)),
          r * (Math.cos(theta) * (RIGHT[2] ?? 0) + Math.sin(theta) * (UP[2] ?? 0)),
        ]);
      }
    }
    this.add_points(rings);
    return this;
  }
}

export interface PointOptions extends PMobjectOptions {
  location?: number[] | readonly number[] | NDArray;
}

export class Point extends PMobject {
  location: number[] | readonly number[] | NDArray;

  constructor(
    location: number[] | readonly number[] | NDArray = [...ORIGIN],
    color: ParsableManimColor = BLACK,
    options: PointOptions = {},
  ) {
    // NOTE: Python sets self.location BEFORE super().__init__() so that the
    // generate_points() call inside the base constructor already sees it. TS
    // cannot assign fields before super(), so the location is forwarded
    // through options (the Mobject constructor merges unknown keys onto the
    // instance) and assigned again below to survive field initialization.
    const { location: _dropped, ...rest } = options;
    void _dropped;
    super({ color, ...rest, location } as PMobjectOptions);
    this.location = location;
    this.init_points();
  }

  init_points(): this {
    this.reset_points();
    this.generate_points();
    this.set_points(toRows(this.location, this.dim));
    return this;
  }

  generate_points(): this {
    // NOTE: runs once inside the base constructor before the location
    // assignment above; fall back to ORIGIN there (reset by init_points()).
    const loc = this.location ?? ORIGIN;
    this.add_points(toRows(loc, this.dim));
    return this;
  }
}
