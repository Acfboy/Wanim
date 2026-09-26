// P1-D: geometry constructor options-bag normalization.
//
// Every Python geometry constructor accepts its parameters as keywords. The
// TS port spells kwargs as an options object. These tests assert that a
// leading options bag produces exactly the same geometry as the positional
// call (previously `new Square({ side_length: 2 })` bound the bag to the
// numeric parameter and produced NaN points).
import { describe, expect, it } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
import { PI } from "../src/constants.js";
import type { NDArray } from "numpy-ts";
import {
  Square,
  Rectangle,
  Triangle,
  RegularPolygon,
  RegularPolygram,
  Star,
  RoundedRectangle,
} from "../src/mobject/geometry/polygram.js";
import {
  Circle,
  Dot,
  Annulus,
  Sector,
  Arc,
} from "../src/mobject/geometry/arc.js";
import { Line, Arrow, DashedLine, DoubleArrow } from "../src/mobject/geometry/line.js";
import {
  ArrowTriangleTip,
  ArrowTriangleFilledTip,
  ArrowCircleTip,
  ArrowCircleFilledTip,
  ArrowSquareTip,
  ArrowSquareFilledTip,
  StealthTip,
} from "../src/mobject/geometry/tips.js";
import {
  Sphere,
  Cube,
  Prism,
  Cone,
  Cylinder,
  Torus,
  Dot3D,
  Line3D,
  Arrow3D,
} from "../src/mobject/three_d/three_dimensions.js";
import type { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function rows(points: NDArray): number[][] {
  return pointRows(points, (points.shape[1] as number | undefined) ?? 3);
}

function expectNoNaN(values: number[][]): void {
  for (const row of values) {
    for (const value of row) expect(Number.isNaN(value)).toBe(false);
  }
}

function expectSamePoints(a: VMobject, b: VMobject, tolerance = 1e-9): void {
  expect(a.get_num_points()).toBe(b.get_num_points());
  const first = rows(a.points);
  const second = rows(b.points);
  expectNoNaN(first);
  expectNoNaN(second);
  expect(first.length).toBe(second.length);
  for (let i = 0; i < first.length; i += 1) {
    for (let d = 0; d < (first[i]?.length ?? 0); d += 1) {
      expect(Math.abs((first[i]?.[d] ?? 0) - (second[i]?.[d] ?? 0))).toBeLessThan(tolerance);
    }
  }
}

describe("geometry options-bag matches positional construction", () => {
  it("Square", () => {
    expectSamePoints(new Square(2), new Square({ side_length: 2 }));
    expectSamePoints(new Square(2, { color: "red" }), new Square({ side_length: 2, color: "red" }));
  });

  it("Rectangle options bag", () => {
    expectSamePoints(new Rectangle(), new Rectangle({ height: 2, width: 4 }));
    const rect = new Rectangle({ height: 3, width: 2 });
    expect(rect.width).toBeCloseTo(2, 9);
    expect(rect.height).toBeCloseTo(3, 9);
    expect(rect.get_num_points()).toBeGreaterThan(0);
  });

  it("Triangle options bag", () => {
    expectSamePoints(new Triangle(), new Triangle({}));
  });

  it("RegularPolygon", () => {
    expectSamePoints(new RegularPolygon(5), new RegularPolygon({ n: 5 }));
    expectSamePoints(new RegularPolygon(5, { radius: 2 }), new RegularPolygon({ n: 5, radius: 2 }));
  });

  it("Star", () => {
    expectSamePoints(new Star(5), new Star({ n: 5 }));
    expectSamePoints(new Star(5, { outer_radius: 2 }), new Star({ n: 5, outer_radius: 2 }));
  });

  it("RoundedRectangle", () => {
    expectSamePoints(new RoundedRectangle(0.5), new RoundedRectangle({ corner_radius: 0.5 }));
  });

  it("Circle", () => {
    expectSamePoints(new Circle(2), new Circle({ radius: 2 }));
    expectSamePoints(new Circle(2, { color: "green" }), new Circle({ radius: 2, color: "green" }));
  });

  it("Dot", () => {
    expectSamePoints(new Dot([1, 2, 0], 0.1), new Dot({ point: [1, 2, 0], radius: 0.1 }));
  });

  it("Annulus", () => {
    expectSamePoints(new Annulus(1, 2), new Annulus({ inner_radius: 1, outer_radius: 2 }));
  });

  it("Sector", () => {
    expectSamePoints(new Sector(2), new Sector({ radius: 2 }));
  });

  it("Arc", () => {
    expectSamePoints(
      new Arc(2, 0.3, 1.2, 5),
      new Arc({ radius: 2, start_angle: 0.3, angle: 1.2, num_components: 5 }),
    );
  });

  it("Line", () => {
    expectSamePoints(new Line([0, 0, 0], [1, 1, 0], 0.1), new Line({ start: [0, 0, 0], end: [1, 1, 0], buff: 0.1 }));
  });

  it("Arrow", () => {
    expectSamePoints(
      new Arrow([0, 0, 0], [1, 1, 0], 0.1),
      new Arrow({ start: [0, 0, 0], end: [1, 1, 0], buff: 0.1 }),
    );
  });

  it("DashedLine", () => {
    expectSamePoints(
      new DashedLine([0, 0, 0], [1, 1, 0]),
      new DashedLine({ start: [0, 0, 0], end: [1, 1, 0] }),
    );
  });

  it("DoubleArrow", () => {
    expectSamePoints(
      new DoubleArrow([0, 0, 0], [1, 1, 0]),
      new DoubleArrow({ start: [0, 0, 0], end: [1, 1, 0] }),
    );
  });

  it("ArrowTriangleTip", () => {
    expectSamePoints(
      new ArrowTriangleTip(0.5, 2, 1, 0.5, PI / 2),
      new ArrowTriangleTip({ fill_opacity: 0.5, stroke_width: 2, length: 1, width: 0.5, start_angle: PI / 2 }),
    );
  });

  it("ArrowTriangleFilledTip", () => {
    expectSamePoints(new ArrowTriangleFilledTip(), new ArrowTriangleFilledTip({}));
    expectSamePoints(new ArrowTriangleFilledTip(0.5, 2), new ArrowTriangleFilledTip({ fill_opacity: 0.5, stroke_width: 2 }));
  });

  it("ArrowCircleTip / Filled", () => {
    expectSamePoints(new ArrowCircleTip(0.5, 2, 1, PI), new ArrowCircleTip({ fill_opacity: 0.5, stroke_width: 2, length: 1 }));
    expectSamePoints(new ArrowCircleFilledTip(), new ArrowCircleFilledTip({}));
  });

  it("ArrowSquareTip / Filled", () => {
    expectSamePoints(new ArrowSquareTip(0.5, 2, 1, PI), new ArrowSquareTip({ fill_opacity: 0.5, stroke_width: 2, length: 1 }));
    expectSamePoints(new ArrowSquareFilledTip(), new ArrowSquareFilledTip({}));
  });

  it("StealthTip", () => {
    expectSamePoints(new StealthTip(1, 3, 0.5, PI), new StealthTip({ fill_opacity: 1, stroke_width: 3, length: 0.5 }));
    expectSamePoints(new StealthTip(), new StealthTip({}));
  });

  it("Sphere", () => {
    expectSamePoints(new Sphere([1, 0, 0], 2), new Sphere({ center: [1, 0, 0], radius: 2 }));
  });

  it("Cube", () => {
    expectSamePoints(new Cube(3), new Cube({ side_length: 3 }));
  });

  it("Prism", () => {
    expectSamePoints(new Prism([2, 3, 4]), new Prism({ dimensions: [2, 3, 4] }));
  });

  it("Cone", () => {
    expectSamePoints(new Cone(2, 3), new Cone({ base_radius: 2, height: 3 }));
  }, 30000);

  it("Cylinder", () => {
    expectSamePoints(new Cylinder(2, 3), new Cylinder({ radius: 2, height: 3 }));
  });

  it("Torus", () => {
    expectSamePoints(new Torus(2, 0.5), new Torus({ major_radius: 2, minor_radius: 0.5 }));
  });

  it("Dot3D", () => {
    expectSamePoints(new Dot3D([1, 0, 0], 0.1), new Dot3D({ point: [1, 0, 0], radius: 0.1 }));
  });

  it("Line3D", () => {
    const positional = new Line3D([0, 0, 0], [1, 1, 0], 0.1);
    const bag = new Line3D({ start: [0, 0, 0], end: [1, 1, 0], thickness: 0.1 });
    expect(bag.thickness).toBe(positional.thickness);
    expect(bag.vect).toEqual(positional.vect);
    expect(bag.start.toArray()).toEqual(positional.start.toArray());
  });

  it("Arrow3D", () => {
    const positional = new Arrow3D([0, 0, 0], [1, 1, 0], 0.1, 0.3, 0.08);
    const bag = new Arrow3D({ start: [0, 0, 0], end: [1, 1, 0], thickness: 0.1, height: 0.3, base_radius: 0.08 });
    expect(bag.thickness).toBe(positional.thickness);
    expect(bag.vect).toEqual(positional.vect);
  }, 30000);

  it("RegularPolygram leading bag", () => {
    expectSamePoints(new RegularPolygram(4, { density: 1 }), new RegularPolygram({ num_vertices: 4, density: 1 }));
  });

  it("unsupported/missing numeric arg does not produce NaN", () => {
    expect(() => new RegularPolygram({ density: 2 })).toThrow(/num_vertices/);
    expectNoNaN(rows(new Square({ side_length: 3 }).points));
  });
});
