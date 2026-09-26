import { VGroup, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { addPoints, isNDArray, norm, pointRows, scalePoint, subtractPoints } from "../../utils/arrays.js";
import { adjacent_n_tuples, adjacent_pairs } from "../../utils/iterables.js";
import { angle_between_vectors, normalize, regular_vertices } from "../../utils/space_ops.js";
import { QuickHull, type Facet, type SubFacet } from "../../utils/qhull.js";
import { DOWN, RIGHT, TAU, DL, DR, SMALL_BUFF, UL, UR } from "../../constants.js";
import { BLUE, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { array, type NDArray } from "numpy-ts";

// Lazy Line access for Rectangle grid lines (mirrors Python's lazy
// `from manim.mobject.geometry.line import Line` inside
// Rectangle.__init__; a top-level import would cycle via
// line -> tips -> polygram and break module evaluation order, depending on
// entry point). line.ts registers its Line class via
// set_grid_line_constructor; grid lines require line.ts to be loaded.
export type GridLineConstructor = new (...args: any[]) => VMobject;

let gridLineConstructor: GridLineConstructor | null = null;

export function set_grid_line_constructor(ctor: GridLineConstructor): void {
  gridLineConstructor = ctor;
}

// Runtime constructors for Polygram.round_corners (mirrors the lazy
// `from manim.mobject.geometry.arc import ArcBetweenPoints` /
// `from manim.mobject.geometry.line import Line` inside the Python method;
// top-level imports would cycle via line -> tips -> polygram). Registered by
// the geometry/line module on load (it already imports ArcBetweenPoints).
export interface RoundCornersArc {
  points: NDArray;
  get_end(): number[];
  get_start(): number[];
  get_arc_length(): number;
}

export type RoundCornersArcConstructor = new (
  start: unknown,
  end: unknown,
  angle?: number,
  radius?: number | null,
  options?: Record<string, unknown>,
) => RoundCornersArc;

export interface RoundCornersLine {
  points: NDArray;
  get_start(): number[];
  get_end(): number[];
  get_length(): number;
  insert_n_curves(n: number): unknown;
}

export type RoundCornersLineConstructor = new (
  start: unknown,
  end: unknown,
  ...args: unknown[]
) => RoundCornersLine;

let roundCornersArcConstructor: RoundCornersArcConstructor | null = null;
let roundCornersLineConstructor: RoundCornersLineConstructor | null = null;

export function set_round_corners_constructors(
  arcCtor: RoundCornersArcConstructor,
  lineCtor: RoundCornersLineConstructor,
): void {
  roundCornersArcConstructor = arcCtor;
  roundCornersLineConstructor = lineCtor;
}

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
    if (radius === 0) {
      return this;
    }
    if (roundCornersArcConstructor === null || roundCornersLineConstructor === null) {
      throw new Error(
        "TODO: Polygram.round_corners with nonzero radius needs the geometry/line module to be imported (register ArcBetweenPoints/Line via set_round_corners_constructors)",
      );
    }
    const arcCtor = roundCornersArcConstructor;
    const lineCtor = roundCornersLineConstructor;

    const new_points: number[][] = [];

    for (const vertex_group of this.get_vertex_groups()) {
      const arcs: RoundCornersArc[] = [];

      // Repeat the radius list as necessary in order to provide a radius
      // for each vertex.
      let radius_list: number[];
      if (typeof radius === "number") {
        radius_list = new Array(vertex_group.length).fill(radius);
      } else {
        const repeats = Math.ceil(vertex_group.length / radius.length);
        radius_list = Array.from(
          { length: repeats * radius.length },
          (_, index) => radius[index % radius.length] ?? 0,
        );
      }

      const triples = adjacent_n_tuples(vertex_group, 3);
      for (let index = 0; index < triples.length; index += 1) {
        const current_radius = radius_list[index] ?? 0;
        const [v1 = [0, 0, 0], v2 = [0, 0, 0], v3 = [0, 0, 0]] = triples[index] ?? [];
        const vect1 = subtractPoints(v2, v1, this.dim);
        const vect2 = subtractPoints(v3, v2, this.dim);
        const unit_vect1 = normalize(vect1);
        const unit_vect2 = normalize(vect2);

        let corner_angle = angle_between_vectors(vect1, vect2);
        // Negative radius gives concave curves
        corner_angle *= Math.sign(current_radius);

        // Distance between vertex and start of the arc
        let cut_off_length = current_radius * Math.tan(corner_angle / 2);

        // Clamp the cut-off length so that a (near) 180 degree turn
        // between consecutive edges - which makes ``tan(angle / 2)``
        // diverge to infinity - cannot push the arc endpoints to
        // infinity.  This also bounds the rounded corner to at most half
        // of the shorter adjacent edge, matching the expected behavior
        // for degenerate collinear polygons (see issue #3052).
        const max_cut_off =
          Math.min(norm(vect1), norm(vect2)) / 2;
        cut_off_length = Math.min(
          Math.max(cut_off_length, -max_cut_off),
          max_cut_off,
        );

        // Determines counterclockwise vs. clockwise
        const cross_z =
          (vect1[0] ?? 0) * (vect2[1] ?? 0) - (vect1[1] ?? 0) * (vect2[0] ?? 0);
        const turn_sign = Math.sign(cross_z);

        const arc = new arcCtor(
          v2.map((value, dim) => value - (unit_vect1[dim] ?? 0) * cut_off_length),
          v2.map((value, dim) => value + (unit_vect2[dim] ?? 0) * cut_off_length),
          turn_sign * corner_angle,
          null,
          { num_components: components_per_rounded_corner },
        );
        arcs.push(arc);
      }

      let average_arc_length = 1.0;
      if (evenly_distribute_anchors) {
        // Determine the average length of each curve
        const nonzero_length_arcs = arcs.filter(
          (arc) => pointRows(arc.points, this.dim).length > 4,
        );
        if (nonzero_length_arcs.length > 0) {
          const total_arc_length = nonzero_length_arcs.reduce(
            (sum, arc) => sum + arc.get_arc_length(),
            0,
          );
          const num_curves =
            nonzero_length_arcs.reduce(
              (sum, arc) => sum + pointRows(arc.points, this.dim).length,
              0,
            ) / 4;
          average_arc_length = total_arc_length / num_curves;
        }
      }

      // To ensure that we loop through starting with last
      const ordered = [arcs[arcs.length - 1] as RoundCornersArc, ...arcs.slice(0, -1)];

      for (const pair of adjacent_pairs(ordered)) {
        const arc1 = pair[0] as RoundCornersArc;
        const arc2 = pair[1] as RoundCornersArc;
        new_points.push(...pointRows(arc1.points, this.dim));

        const line = new lineCtor(arc1.get_end(), arc2.get_start());

        // Make sure anchors are evenly distributed, if necessary
        if (evenly_distribute_anchors) {
          line.insert_n_curves(Math.ceil(line.get_length() / average_arc_length));
        }

        new_points.push(...pointRows(line.points, this.dim));
      }
    }

    if (new_points.length > 0) {
      this.set_points(array(new_points) as NDArray);
    }

    return this;
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

export interface StarOptions extends PolygramOptions {
  outer_radius?: number;
  inner_radius?: number | null;
  density?: number;
  start_angle?: number | null;
}

export class Star extends Polygon {
  start_angle: number | null;

  constructor(n = 5, options: StarOptions = {}) {
    const {
      outer_radius = 1,
      inner_radius = null,
      density = 2,
      start_angle = TAU / 4,
      ...rest
    } = options;
    const inner_angle = TAU / (2 * n);
    let resolved_inner_radius = inner_radius;
    if (resolved_inner_radius === null || resolved_inner_radius === undefined) {
      // See https://math.stackexchange.com/a/2136292 for an overview of
      // how to calculate the inner radius of a perfect star.
      if (density <= 0 || density >= n / 2) {
        const error = new Error(
          `Incompatible density ${density} for number of points ${n}`,
        );
        error.name = "ValueError";
        throw error;
      }
      const outer_angle = (TAU * density) / n;
      const inverse_x =
        1 -
        Math.tan(inner_angle) *
          ((Math.cos(outer_angle) - 1) / Math.sin(outer_angle));
      resolved_inner_radius =
        outer_radius / (Math.cos(inner_angle) * inverse_x);
    }
    const [outer_vertices, resolved_start_angle] = regular_vertices(
      n,
      outer_radius,
      start_angle,
    );
    const [inner_vertices] = regular_vertices(
      n,
      resolved_inner_radius,
      (resolved_start_angle ?? 0) + inner_angle,
    );
    const outer_rows = pointRows(outer_vertices, 3);
    const inner_rows = pointRows(inner_vertices, 3);
    const vertices: number[][] = [];
    for (let index = 0; index < n; index += 1) {
      vertices.push([...(outer_rows[index] ?? [0, 0, 0])]);
      vertices.push([...(inner_rows[index] ?? [0, 0, 0])]);
    }
    super(...vertices, rest);
    this.start_angle = resolved_start_angle;
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

    let v = this.get_vertices();
    this.grid_lines = new VGroup();

    if (grid_xstep || grid_ystep) {
      // NOTE: mirrors Python's lazy `from manim.mobject.geometry.line
      // import Line` (see module hook above).
      if (gridLineConstructor === null) {
        throw new Error(
          "TODO: Rectangle grid lines need line.ts to be imported (register Line via set_grid_line_constructor)",
        );
      }
      v = this.get_vertices();
    }
    // NOTE: guarded non-null by the throw above whenever grid lines are built.
    const lineCtor = gridLineConstructor as GridLineConstructor;

    if (grid_xstep) {
      const step = Math.abs(grid_xstep);
      const count = Math.trunc(width / step);
      const grid = new VGroup();
      for (let i = 1; i < count; i += 1) {
        const start = addPoints(v[1] ?? [0, 0, 0], scalePoint(RIGHT, i * step, 3), 3);
        const end = addPoints(start, scalePoint(DOWN, height, 3), 3);
        grid.add(new lineCtor(start, end, { color }));
      }
      this.grid_lines.add(grid);
    }

    if (grid_ystep) {
      const step = Math.abs(grid_ystep);
      const count = Math.trunc(height / step);
      const grid = new VGroup();
      for (let i = 1; i < count; i += 1) {
        const start = addPoints(v[1] ?? [0, 0, 0], scalePoint(DOWN, i * step, 3), 3);
        const end = addPoints(start, scalePoint(RIGHT, width, 3), 3);
        grid.add(new lineCtor(start, end, { color }));
      }
      this.grid_lines.add(grid);
    }

    if (this.grid_lines.submobjects.length > 0) {
      this.add(this.grid_lines);
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

export interface CutoutOptions extends VMobjectOptions {}

export class Cutout extends VMobject {
  constructor(main_shape: VMobject, ...args: unknown[]) {
    let options: CutoutOptions = {};
    let mobjects: unknown[] = args;
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !(last instanceof VMobject) &&
      !Array.isArray(last) &&
      !isNDArray(last)
    ) {
      options = last as CutoutOptions;
      mobjects = args.slice(0, -1);
    }
    super(options);
    this.append_points(main_shape.points);
    const sub_direction: "CCW" | "CW" =
      main_shape.get_direction() === "CW" ? "CCW" : "CW";
    for (const mobject of mobjects) {
      this.append_points((mobject as VMobject).force_direction(sub_direction).points);
    }
  }
}

export interface ConvexHullOptions extends PolygramOptions {
  tolerance?: number;
}

export class ConvexHull extends Polygram {
  constructor(...args: unknown[]) {
    // NOTE: locals are computed before super() (which must come first among
    // `this` uses, but plain local computation beforehand is allowed).
    let points: unknown[] = args;
    let tolerance = 1e-5;
    let options: ConvexHullOptions = {};
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last)
    ) {
      const { tolerance: toleranceOption, ...rest } = last as ConvexHullOptions;
      tolerance = toleranceOption ?? tolerance;
      options = rest;
      points = args.slice(0, -1);
    }
    // Build Convex Hull (mirrors np.array(points)[:, :2]).
    const matrix = points.map((point) => {
      const row = pointRows(point, 3)[0] ?? [0, 0, 0];
      return [row[0] ?? 0, row[1] ?? 0];
    });
    const hull = new QuickHull(tolerance);
    hull.build(matrix);
    // Extract vertices: walk the hull boundary facet by facet. The TS
    // QuickHull stores facets in an array keyed by string (instead of a
    // Python object set), so set difference/pop become key-filtered pop.
    const live = hull.facets.filter((facet) => !hull.removed.has(facet.key()));
    let facet = live.pop() as Facet;
    const subfacets: SubFacet[] = [...facet.subfacets];
    while (subfacets.length <= live.length) {
      const sf = subfacets[subfacets.length - 1] as SubFacet;
      const neighborMap = hull.neighbors.get(sf.key()) ?? new Map<string, Facet>();
      facet = [...neighborMap.values()].find(
        (candidate) => candidate.key() !== facet.key(),
      ) as Facet;
      const next = facet.subfacets.find(
        (candidate) => candidate.key() !== sf.key(),
      ) as SubFacet;
      subfacets.push(next);
    }
    // Setup vertices as Point3D.
    const vertices = subfacets.flatMap((subfacet) =>
      subfacet.coordinates.map((row) => [row[0] ?? 0, row[1] ?? 0, 0]),
    );
    super(vertices, options);
  }
}
