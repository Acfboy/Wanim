// End-to-end geometry ground-truth tests (2D shapes).
//
// Expected values in this file were captured by running the equivalent
// construction with the reference Manim (the split original manim used by the
// `the original manim interpreter` differential harness) and rounding to 6 decimals, e.g.
//   Circle(radius=2.0, arc_center=[1.0, -0.5, 0.0]).get_arc_center()
// 
// these tests pin the method-level values that the rendered-comparison
// harness is insensitive to (scalars such as length/angle/radius, vertex
// order, arc centers, ...).
import { describe, expect, it } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
import {
  Arc,
  ArcBetweenPoints,
  AnnularSector,
  Annulus,
  ArcPolygon,
  Arrow,
  ArrowCircleTip,
  Brace,
  Circle,
  Cross,
  CurvedArrow,
  DashedLine,
  Dot,
  DoubleArrow,
  Elbow,
  Ellipse,
  Line,
  Polygon,
  Rectangle,
  RegularPolygon,
  RegularPolygram,
  RoundedRectangle,
  Sector,
  Square,
  Star,
  SurroundingRectangle,
  TangentLine,
  Triangle,
  Vector,
  Angle,
  PI,
  DOWN,
} from "../src/index.js";

function norm(value: number, digits = 4): number {
  const factor = 10 ** digits;
  const rounded = Math.round(value * factor) / factor;
  return rounded === 0 ? 0 : rounded;
}

function normArr(values: readonly number[] | Float64Array | number[], digits = 4): number[] {
  return [...values].map((value) => norm(value, digits));
}

function expectArr(actual: readonly number[] | Float64Array, expected: readonly number[], digits = 4): void {
  expect(normArr(actual, digits)).toEqual(expected.map((value) => norm(value, digits)));
}

function pointsOf(mob: { points: { toArray(): unknown }; dim: number }): number[][] {
  return pointRows(mob.points as never, mob.dim) as number[][];
}

function vertices(mob: { get_vertices(): number[][] }): number[][] {
  return mob.get_vertices().map((row) => [...row]);
}

describe("e2e geometry ground truth (2D)", () => {
  it("Line endpoints, vector, angle, slope, projection", () => {
    const line = new Line([-2, -1, 0], [2, 1, 0]);
    expectArr(line.get_start(), [-2, -1, 0]);
    expectArr(line.get_end(), [2, 1, 0]);
    expectArr(line.get_vector(), [4, 2, 0]);
    expectArr(line.get_unit_vector(), [0.894427, 0.447214, 0]);
    expect(norm(line.get_angle())).toBe(norm(0.463648));
    expect(norm(line.get_slope())).toBe(0.5);
    expect(norm(line.get_length())).toBe(norm(4.472136));
    expectArr(line.get_projection([1, 1, 0]), [1.2, 0.6, 0]);

    line.set_length(1.0);
    expect(norm(line.get_length())).toBe(1.0);

    const l2 = new Line([-1, 0, 0], [1, 0, 0]);
    l2.set_angle(PI / 3);
    expect(norm(l2.get_angle())).toBe(norm(1.047198));
  });

  it("Line buff shortens both ends", () => {
    const line = new Line([-2, 0, 0], [2, 0, 0], 0.5);
    expectArr(line.get_start(), [-1.5, 0, 0]);
    expectArr(line.get_end(), [1.5, 0, 0]);
    expect(norm(line.get_length())).toBe(3.0);
  });

  it("Circle arc center, start/end, point_at_angle, num points", () => {
    const circle = new Circle(2.0, { arc_center: [1.0, -0.5, 0.0] });
    expect(norm(circle.radius)).toBe(2.0);
    expectArr(circle.get_arc_center(), [1, -0.5, 0]);
    expectArr(circle.get_start(), [3, -0.5, 0]);
    expectArr(circle.get_end(), [3, -0.5, 0]);
    expectArr(circle.point_at_angle(PI / 3), [1.998741, 1.232783, 0]);
    expect(pointsOf(circle).length).toBe(32);
  });

  it("Arc center, endpoints, stop angle and arc length", () => {
    const arc = new Arc(1.5, 0.3, 1.7);
    expect(norm(arc.radius)).toBe(1.5);
    expectArr(arc.get_arc_center(), [0, 0, 0]);
    expectArr(arc.get_start(), [1.433005, 0.44328, 0]);
    expectArr(arc.get_end(), [-0.62422, 1.363946, 0]);
    expect(norm(arc.stop_angle())).toBe(2.0);
    expect(norm(arc.get_arc_length())).toBe(norm(2.549941));
  });

  it("ArcBetweenPoints derived radius and endpoints", () => {
    const abp = new ArcBetweenPoints([-2, 0, 0], [2, 0, 0], PI / 3);
    expect(norm(abp.radius)).toBe(4.0);
    expectArr(abp.get_arc_center(), [0, 3.464102, 0]);
    expectArr(abp.get_start(), [-2, 0, 0]);
    expectArr(abp.get_end(), [2, 0, 0]);
  });

  it("Circle.from_three_points", () => {
    const c = Circle.from_three_points([0, 0, 0], [2, 0, 0], [0, 2, 0]);
    expectArr(c.get_arc_center(), [1, 1, 0]);
    expect(norm(c.radius)).toBe(norm(1.414214));
  });

  it("Dot center and radius", () => {
    const dot = new Dot([1, 1, 0], 0.3);
    expectArr(dot.get_center(), [1, 1, 0]);
    expect(norm(dot.radius)).toBe(0.3);
  });

  it("Square and Rectangle vertices", () => {
    const square = new Square(3.0);
    expect(norm(square.side_length)).toBe(3.0);
    expect(vertices(square)).toEqual([
      [1.5, 1.5, 0],
      [-1.5, 1.5, 0],
      [-1.5, -1.5, 0],
      [1.5, -1.5, 0],
    ]);

    const rect = new Rectangle({ width: 3.0, height: 1.5 });
    expect(norm(rect.width)).toBe(3.0);
    expect(norm(rect.height)).toBe(1.5);
    expect(vertices(rect)).toEqual([
      [1.5, 0.75, 0],
      [-1.5, 0.75, 0],
      [-1.5, -0.75, 0],
      [1.5, -0.75, 0],
    ]);
  });

  it("RoundedRectangle", () => {
    const rr = new RoundedRectangle(0.5, { width: 4.0, height: 2.0 });
    expect(norm(rr.width)).toBe(4.0);
    expect(norm(rr.height)).toBe(2.0);
    expect(pointsOf(rr).length).toBe(32);
  });

  it("Triangle and RegularPolygon vertices", () => {
    const tri = new Triangle();
    expect(vertices(tri).map((row) => normArr(row))).toEqual([
      [0, 1, 0],
      [-0.866025, -0.5, 0],
      [0.866025, -0.5, 0],
    ].map((row) => normArr(row)));

    const rp5 = new RegularPolygon(5);
    expect(vertices(rp5).map((row) => normArr(row))).toEqual([
      [0, 1, 0],
      [-0.951057, 0.309017, 0],
      [-0.587785, -0.809017, 0],
      [0.587785, -0.809017, 0],
      [0.951057, 0.309017, 0],
    ].map((row) => normArr(row)));

    const rp7 = new RegularPolygon(7, { radius: 2.0, start_angle: PI / 2 });
    expect(vertices(rp7).map((row) => normArr(row))).toEqual([
      [0, 2, 0],
      [-1.563663, 1.24698, 0],
      [-1.949856, -0.445042, 0],
      [-0.867767, -1.801938, 0],
      [0.867767, -1.801938, 0],
      [1.949856, -0.445042, 0],
      [1.563663, 1.24698, 0],
    ].map((row) => normArr(row)));
  });

  it("RegularPolygram with density produces one vertex group", () => {
    const rpg = new RegularPolygram(7, { density: 3 });
    const groups = rpg.get_vertex_groups().map((group) => group.map((row) => normArr(row)));
    expect(groups).toEqual([
      [
        [0, 1, 0],
        [-0.433884, -0.900969, 0],
        [0.781831, 0.62349, 0],
        [-0.974928, -0.222521, 0],
        [0.974928, -0.222521, 0],
        [-0.781831, 0.62349, 0],
        [0.433884, -0.900969, 0],
      ].map((row) => normArr(row)),
    ]);
  });

  it("Star vertices (default and explicit inner radius)", () => {
    const star = new Star(5);
    expect(vertices(star).map((row) => normArr(row))).toEqual([
      [0, 1, 0],
      [-0.224514, 0.309017, 0],
      [-0.951057, 0.309017, 0],
      [-0.363271, -0.118034, 0],
      [-0.587785, -0.809017, 0],
      [0, -0.381966, 0],
      [0.587785, -0.809017, 0],
      [0.363271, -0.118034, 0],
      [0.951057, 0.309017, 0],
      [0.224514, 0.309017, 0],
    ].map((row) => normArr(row)));

    const star6 = new Star(6, { inner_radius: 0.4 });
    expect(vertices(star6).map((row) => normArr(row))).toEqual([
      [0, 1, 0],
      [-0.2, 0.34641, 0],
      [-0.866025, 0.5, 0],
      [-0.4, 0, 0],
      [-0.866025, -0.5, 0],
      [-0.2, -0.34641, 0],
      [0, -1, 0],
      [0.2, -0.34641, 0],
      [0.866025, -0.5, 0],
      [0.4, 0, 0],
      [0.866025, 0.5, 0],
      [0.2, 0.34641, 0],
    ].map((row) => normArr(row)));
  });

  it("Ellipse size", () => {
    const e = new Ellipse(3.0, 1.0);
    expect(norm(e.width)).toBe(3.0);
    expect(norm(e.height)).toBe(1.0);
  });

  it("Annulus / AnnularSector / Sector structure", () => {
    const ann = new Annulus(1.0, 2.0);
    expect(norm(ann.inner_radius)).toBe(1.0);
    expect(norm(ann.outer_radius)).toBe(2.0);
    expect(pointsOf(ann).length).toBe(64);

    const asec = new AnnularSector(1.0, 2.0, PI / 2);
    expect(norm(asec.inner_radius)).toBe(1.0);
    expect(norm(asec.outer_radius)).toBe(2.0);
    expect(pointsOf(asec).length).toBe(72);

    const sec = new Sector(1.5, { angle: PI / 3 });
    expect(norm(sec.inner_radius)).toBe(0.0);
    expect(norm(sec.outer_radius)).toBe(1.5);
  });

  it("Angle value in radians/degrees and lines", () => {
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [1, 1, 0]);
    const angle = new Angle(l1, l2, { radius: 0.7 });
    expect(norm(angle.get_value())).toBe(norm(0.785398));
    expect(norm(angle.get_value(true))).toBe(45.0);
    expect(angle.get_lines().submobjects.length).toBe(2);
  });

  it("Arrow default tip length scales with length", () => {
    const arrow = new Arrow([-2, -1, 0], [2, 1, 0]);
    expect(norm(arrow.get_length())).toBe(norm(3.972136));
    expect(norm(arrow.get_default_tip_length())).toBe(0.35);
    expect(norm(arrow.tip!.length)).toBe(0.35);
    expectArr(arrow.get_normal_vector(), [0, 0, -1]);
    expect(norm(arrow.stroke_width)).toBe(6.0);

    const long = new Arrow([-4, 0, 0], [4, 0, 0]);
    expect(norm(long.get_default_tip_length())).toBe(0.35);
    expect(norm(long.tip!.length)).toBe(0.35);

    // Regression: get_unpositioned_tip must honour a non-default tip_length
    // (previously the style bag was bound to the positional `length` slot for
    // custom tip shapes, yielding NaN geometry / ignored length).
    const custom = new Arrow([-4, 0, 0], [4, 0, 0], { tip_length: 1.0 });
    expect(norm(custom.get_default_tip_length())).toBe(1.0);
    expect(norm(custom.tip!.length)).toBe(1.0);

    const short = new Arrow([-0.3, 0, 0], [0.3, 0, 0], { tip_length: 1.0 });
    expect(norm(short.get_default_tip_length())).toBe(0.025);
    expect(norm(short.tip!.length)).toBe(0.025);
  });

  it("Arrow custom tip shapes get finite geometry", () => {
    for (const tip_shape of [ArrowCircleTip]) {
      const arrow = new Arrow([-2, -1, 0], [2, 1, 0], { tip_shape });
      const rows = pointsOf(arrow);
      for (const row of rows) {
        for (const value of row) expect(Number.isFinite(value)).toBe(true);
      }
      expect(norm(arrow.tip!.length)).toBe(0.35);
    }
  });

  it("DashedLine dash count and endpoints", () => {
    const dl = new DashedLine([-2, 0, 0], [2, 0, 0]);
    expect(dl._calculate_num_dashes()).toBe(40);
    expectArr(dl.get_start(), [-2, 0, 0]);
    expectArr(dl.get_end(), [2, 0, 0]);
  });

  it("DoubleArrow endpoints", () => {
    const da = new DoubleArrow([-2, 0, 0], [2, 0, 0]);
    expectArr(da.get_start(), [-1.75, 0, 0]);
    expectArr(da.get_end(), [1.75, 0, 0]);
  });

  it("Vector", () => {
    const v = new Vector([2.0, 1.0, 0.0]);
    expectArr(v.get_end(), [2, 1, 0]);
  });

  it("Polygon vertices", () => {
    const poly = new Polygon([-1, -1, 0], [1, -1, 0], [0, 1, 0]);
    expect(vertices(poly)).toEqual([
      [-1, -1, 0],
      [1, -1, 0],
      [0, 1, 0],
    ]);
  });

  it("Polygram.round_corners with a repeated radius list", () => {
    const poly = new Polygon([-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]);
    poly.round_corners([0.1, 0.3]);
    const rows = pointsOf(poly);
    expect(rows.length).toBe(32);
    expectArr(rows[0]!, [-1, -0.7, 0]);
    expectArr(rows[1]!, [-1, -0.86569, 0]);
  });

  it("SurroundingRectangle buff and size", () => {
    const sr = new SurroundingRectangle(new Circle(0.5));
    expect(norm(sr.width)).toBe(1.2);
    expect(norm(sr.height)).toBe(1.2);
    expect(norm(sr.buff as number)).toBe(0.1);
  });

  it("Cross size", () => {
    const cross = new Cross(new Circle());
    expect(norm(cross.width)).toBe(2.0);
  });

  it("Elbow width/angle", () => {
    const elbow = new Elbow(0.5, PI / 6);
    expect(norm(elbow.angle)).toBe(norm(0.523599));
    expect(norm(elbow.width)).toBe(norm(0.683013));
  });

  it("Brace width and direction", () => {
    const brace = new Brace(new Square(), { direction: DOWN });
    expect(norm(brace.width)).toBe(2.0);
    expectArr(brace.get_direction(), [0, -1, 0]);
  });

  it("TangentLine length and endpoints", () => {
    const tl = new TangentLine(new Circle(2.0), 0.25, 2.0);
    expect(norm(tl.get_length())).toBe(2.0);
    expectArr(tl.get_start(), [1, 2, 0]);
    expectArr(tl.get_end(), [-1, 2, 0]);
  });

  it("ArcPolygon point count", () => {
    const ap = new ArcPolygon([-1, -1, 0], [1, -1, 0], [0, 1, 0]);
    expect(pointsOf(ap).length).toBe(96);
  });

  it("CurvedArrow does not produce NaN points", () => {
    const ca = new CurvedArrow([-2, 0, 0], [2, 0, 0], { radius: -5 });
    for (const row of pointsOf(ca)) {
      for (const value of row) expect(Number.isFinite(value)).toBe(true);
    }
  });
});
