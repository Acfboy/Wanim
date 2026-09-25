import { VGroup, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { isNDArray, norm, pointRows, subtractPoints } from "../../utils/arrays.js";
import { regular_vertices } from "../../utils/space_ops.js";
import { TAU, DL, DR, SMALL_BUFF, UL, UR } from "../../constants.js";
import { BLUE, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import type { NDArray } from "numpy-ts";

export interface PolygramOptions extends VMobjectOptions {
  color?: ParsableManimColor;
}

export class Polygram extends VMobject {
  constructor(...args: unknown[]) {
    let vertex_groups: unknown[] = args;
    let options: PolygramOptions = {};
    const last = args[args.length - 1];
    if (last !== undefined && typeof last === "object" && !Array.isArray(last) && !isNDArray(last)) {
      options = last as PolygramOptions;
      vertex_groups = args.slice(0, -1);
    }
    super({ color: BLUE, ...options });
    for (const vertices of vertex_groups) {
      const rows = pointRows(vertices, this.dim);
      if (rows.length === 0) continue;
      const [first_vertex, ...rest] = rows;
      this.start_new_path(first_vertex);
      this.add_points_as_corners([...rest, first_vertex ?? [0, 0, 0]]);
    }
  }

  get_vertices(): number[][] {
    return this.get_start_anchors();
  }
  get_vertex_groups(): number[][][] {
    const vertex_groups: number[][][] = [];
    const starts = this.get_start_anchors();
    const ends = this.get_end_anchors();

    // TODO: If any of the original vertex groups contained the starting
    // vertex N times, then .get_vertex_groups() splits it into N groups.
    let group: number[][] = [];
    for (let index = 0; index < starts.length; index += 1) {
      group.push(starts[index] ?? []);
      if (this.consider_points_equals(ends[index], group[0])) {
        vertex_groups.push(group.map((point) => [...point]));
        group = [];
      }
    }
    return vertex_groups;
  }

  round_corners(
    radius: number | number[] = 0.5,
    evenly_distribute_anchors = false,
    components_per_rounded_corner = 2,
  ): this {
    void evenly_distribute_anchors;
    void components_per_rounded_corner;
    if (radius === 0) return this;
    // NOTE: the general path needs ArcBetweenPoints and Line
    // (manim.mobject.geometry.arc / line), which live in other translation
    // batches; only the radius == 0 fast path is ported here.
    throw new Error("TODO: Polygram.round_corners with nonzero radius needs ArcBetweenPoints/Line, not ported yet");
  }

  roundCorners(
    radius: number | number[] = 0.5,
    evenly_distribute_anchors = false,
    components_per_rounded_corner = 2,
  ): this {
    return this.round_corners(radius, evenly_distribute_anchors, components_per_rounded_corner);
  }
}

export type PolygonVertex = readonly number[] | NDArray | number[];

export class Polygon extends Polygram {
  constructor(...args: unknown[]) {
    // Python passes the whole vertices tuple as a single vertex group:
    // super().__init__(vertices, **kwargs). A trailing plain object is
    // treated as kwargs; everything else is wrapped into one group,
    // unless a single array/NDArray of points was given (already a group).
    let options: PolygramOptions = {};
    let vertices: unknown[] = args;
    const last = args[args.length - 1];
    if (args.length > 1 && isOptionsObject(last)) {
      options = last as PolygramOptions;
      vertices = args.slice(0, -1);
    }
    if (vertices.length === 1 && (Array.isArray(vertices[0]) || isNDArray(vertices[0]))) {
      const single = vertices[0] as unknown[];
      const looksLikeGroup = Array.isArray(single)
        ? single.length === 0 || Array.isArray(single[0]) || isNDArray(single[0] as unknown)
        : true;
      if (looksLikeGroup) {
        super(single, options);
        return;
      }
    }
    super(vertices, options);
  }
}

function isOptionsObject(value: unknown): boolean {
  return value !== null && typeof value === "object" && !Array.isArray(value) && !isNDArray(value);
}

export interface RegularPolygramOptions extends PolygramOptions {
  density?: number;
  radius?: number;
  start_angle?: number | null;
}

export class RegularPolygram extends Polygram {
  start_angle: number | null;

  constructor(num_vertices: number, options: RegularPolygramOptions = {}) {
    const { density = 2, radius = 1, start_angle = null, ...rest } = options;
    // Regular polygrams can be expressed by the number of their vertices
    // and their density. This relation can be expressed as its Schläfli
    // symbol: {num_vertices/density}.
    let count = num_vertices;
    let dense = density;
    const num_gons = gcd(count, dense);
    count = Math.floor(count / num_gons);
    dense = Math.floor(dense / num_gons);

    // Utility function for generating the individual polygon vertices.
    const gen_polygon_vertices = (
      angle: number | null,
    ): [number[][], number | null] => {
      const [reg_vertices, resolved] = regular_vertices(count, radius, angle);
      const reg_rows = pointRows(reg_vertices, 3);
      const vertices: number[][] = [];
      let i = 0;
      for (;;) {
        vertices.push([...(reg_rows[i] ?? [0, 0, 0])]);
        i += dense;
        i %= count;
        if (i === 0) break;
      }
      return [vertices, resolved];
    };

    // NOTE: Python assigns self.start_angle from the first call before
    // super().__init__(); TS requires super() first, so vertex groups are
    // computed up front and the field is assigned after construction.
    const [first_group, resolved_angle] = gen_polygon_vertices(start_angle);
    const vertex_groups: number[][][] = [first_group];
    for (let i = 1; i < num_gons; i += 1) {
      const angle = (resolved_angle ?? 0) + (i / num_gons) * TAU / count;
      const [group] = gen_polygon_vertices(angle);
      vertex_groups.push(group);
    }
    super(...vertex_groups, rest);
    this.start_angle = resolved_angle;
  }
}

function gcd(a: number, b: number): number {
  let x = Math.trunc(Math.abs(a));
  let y = Math.trunc(Math.abs(b));
  while (y !== 0) {
    const temp = x % y;
    x = y;
    y = temp;
  }
  return x;
}

export interface RegularPolygonOptions extends RegularPolygramOptions {}

export class RegularPolygon extends RegularPolygram {
  constructor(n = 6, options: RegularPolygonOptions = {}) {
    super(n, { density: 1, ...options });
  }
}

export interface TriangleOptions extends RegularPolygonOptions {}

export class Triangle extends RegularPolygon {
  constructor(options: TriangleOptions = {}) {
    super(3, options);
  }
}

export interface RectangleOptions extends PolygramOptions {
  color?: ParsableManimColor;
  height?: number;
  width?: number;
  grid_xstep?: number | null;
  grid_ystep?: number | null;
  mark_paths_closed?: boolean;
  close_new_points?: boolean;
}

export class Rectangle extends Polygon {
  grid_lines: VGroup;

  constructor(options: RectangleOptions = {}) {
    const {
      color = WHITE,
      height = 2.0,
      width = 4.0,
      grid_xstep = null,
      grid_ystep = null,
      ...rest
    } = options;
    super(UR, UL, DL, DR, { color, ...rest });
    this.stretch_to_fit_width(width);
    this.stretch_to_fit_height(height);

    this.grid_lines = new VGroup();

    if (grid_xstep || grid_ystep) {
      // NOTE: grid lines need Line (manim.mobject.geometry.line), which
      // lives in another translation batch.
      throw new Error("TODO: Rectangle grid lines need Line, not ported yet");
    }
  }
}

export interface RoundedRectangleOptions extends RectangleOptions {
  corner_radius?: number | number[];
}

export class RoundedRectangle extends Rectangle {
  corner_radius: number | number[];

  constructor(options: RoundedRectangleOptions = {}) {
    const { corner_radius = 0.5, ...rest } = options;
    super(rest);
    this.corner_radius = corner_radius;
    this.round_corners(this.corner_radius);
  }
}

export interface SquareOptions extends RectangleOptions {
  side_length?: number;
}

export class Square extends Rectangle {
  constructor(side_length = 2.0, options: SquareOptions = {}) {
    const { side_length: sideLengthOption, ...rest } = options;
    const resolved = sideLengthOption ?? side_length;
    super({ height: resolved, width: resolved, ...rest });
  }

  get side_length(): number {
    const vertices = this.get_vertices();
    return norm(subtractPoints(vertices[0] ?? [0, 0, 0], vertices[1] ?? [0, 0, 0], 3));
  }

  set side_length(value: number) {
    this.scale(value / this.side_length);
  }
}
