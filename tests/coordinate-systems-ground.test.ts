import { describe, expect, it } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Scene } from "../src/scene/scene.js";
import { VectorScene, LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import {
  Axes,
  ComplexPlane,
  CoordinateSystem,
  NumberPlane,
  PolarPlane,
  ThreeDAxes,
} from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";

// Ground truths captured with:
//   original manim
//   from manim import *; a=Axes(); print(len(a.axes), a.c2p(0,0), a.get_x_unit_size())
// (manim 0.21.0). Each assertion cites the exact observed value.

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function rows(value: unknown): number[][] {
  const raw = (value as { toArray?: () => unknown }).toArray
    ? (value as { toArray: () => unknown }).toArray()
    : value;
  const arr = raw as unknown[];
  return Array.isArray(arr[0])
    ? (arr as number[][]).map((row) => row.map(Number))
    : [(arr as number[]).map(Number)];
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function expectRowClose(actual: unknown, expected: number[], tol = 1e-9): void {
  const r = rows(actual)[0] ?? [];
  expect(r.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(r[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

describe("hierarchy matches Python (Axes(VGroup, CoordinateSystem))", () => {
  it("Axes/NumberPlane/ComplexPlane/PolarPlane/ThreeDAxes are Mobject-based", () => {
    const axes = new Axes();
    expect(axes).toBeInstanceOf(CoordinateSystem);
    expect(axes).toBeInstanceOf(VGroup);
    expect(axes).toBeInstanceOf(Mobject);

    const plane = new NumberPlane();
    expect(plane).toBeInstanceOf(Axes);
    expect(plane).toBeInstanceOf(VGroup);

    const polar = new PolarPlane();
    expect(polar).toBeInstanceOf(Axes);

    const complex = new ComplexPlane();
    expect(complex).toBeInstanceOf(NumberPlane);
    expect(complex).toBeInstanceOf(Axes);

    const three = new ThreeDAxes();
    expect(three).toBeInstanceOf(Axes);
  });
});

describe("Axes ground truth (original manim)", () => {
  it("builds x/y NumberLine axes (original manim: len(a.axes)==2)", () => {
    const a = new Axes();
    expect(a.get_axes()).toBeInstanceOf(VGroup);
    expect(a.get_axes().submobjects.length).toBe(2);
    expect(a.get_x_axis()).toBeInstanceOf(NumberLine);
    expect(a.get_y_axis()).toBeInstanceOf(NumberLine);
    expect(a.x_axis).toBe(a.get_x_axis());
    expect(a.y_axis).toBe(a.get_y_axis());
  });

  it("defaults (original manim: x_range [-7,7,1], y_range [-4,4,1], lengths 12/6, xu .857142, yu .75)", () => {
    const a = new Axes();
    expect(a.x_range).toEqual([-7, 7, 1]);
    expect(a.y_range).toEqual([-4, 4, 1]);
    expectClose(a.x_length ?? NaN, 12);
    expectClose(a.y_length ?? NaN, 6);
    expectClose(a.get_x_unit_size(), 0.8571428571428571);
    expectClose(a.get_y_unit_size(), 0.75);
  });

  it("coords_to_point matches the original manim (0,0)/(1,0)/(0,1)", () => {
    const a = new Axes();
    expectRowClose(a.c2p(0, 0), [0, 0, 0]);
    expectRowClose(a.c2p(1, 0), [0.8571428571428571, 0, 0]);
    expectRowClose(a.c2p(0, 1), [0, 0.75, 0]);
    expectRowClose(a.get_origin(), [0, 0, 0]);
  });

  it("asymmetric ranges (original manim: c2p(0,0)=[-2.5,-2.5,0], c2p(10,5)=[2.5,2.5,0], xu .5)", () => {
    const b = new Axes([0, 10, 1], [0, 5, 1], 5, 5);
    expectRowClose(b.c2p(0, 0), [-2.5, -2.5, 0]);
    expectRowClose(b.c2p(10, 5), [2.5, 2.5, 0]);
    expectClose(b.get_x_unit_size(), 0.5);
  });

  it("coords_to_point input shapes match the original manim", () => {
    const a = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    // original manim: a.c2p([1,2,0]) == [1,2,0] (flat list == c2p(x,y,z))
    expectRowClose(a.c2p([1, 2, 0]), [1, 2, 0]);
    // original manim: a.c2p([0,1,1],[1,1,0]) == [[0,1,1],[1,1,0],[0,0,0]]
    expect(rows(a.c2p([0, 1, 1], [1, 1, 0])).map((r) => r.map((v) => Math.round(v * 1e6) / 1e6))).toEqual([
      [0, 1, 1],
      [1, 1, 0],
      [0, 0, 0],
    ]);
    // original manim: a.c2p([[0,1],[1,1],[1,0]]) == [[0,1,0],[1,1,0],[1,0,0]]
    expect(rows(a.c2p([[0, 1], [1, 1], [1, 0]])).map((r) => r.map((v) => Math.round(v * 1e6) / 1e6))).toEqual([
      [0, 1, 0],
      [1, 1, 0],
      [1, 0, 0],
    ]);
    // original manim: a.c2p([1],[2]) == [[1],[2],[0]] (2D per-axis arrays transpose)
    const arrScalars = rows(a.c2p([1], [2])).map((r) => r.map((v) => Math.round(v * 1e6) / 1e6));
    expect(arrScalars).toEqual([[1], [2], [0]]);
  });

  it("point_to_coords matches the original manim", () => {
    const a = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    expect((a.point_to_coords([1, 1, 0]) as unknown as { toArray(): number[] }).toArray().map(Number)).toEqual([1, 1]);
  });
});

describe("NumberPlane ground truth (original manim)", () => {
  it("default background grid is 22/0 with 7 x-lines and 15 y-lines", () => {
    const p = new NumberPlane();
    expect(p.background_lines?.submobjects.length).toBe(22);
    expect(p.faded_lines?.submobjects.length).toBe(0);
    expect(p.x_lines.submobjects.length).toBe(7);
    expect(p.y_lines.submobjects.length).toBe(15);
    expect(p.x_range[0]).toBeCloseTo(-7.111111111111111, 9);
    expect(p.y_range).toEqual([-4, 4, 1]);
    expect(p.x_length).toBeNull();
    expect(p.y_length).toBeNull();
    expectClose(p.get_x_unit_size(), 1);
    expectRowClose(p.c2p(1, 1), [1, 1, 0]);
  });

  it("explicit grid (original manim: 8/0, x_lines 3, y_lines 5, family 10)", () => {
    const p = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    expect(p.background_lines?.submobjects.length).toBe(8);
    expect(p.faded_lines?.submobjects.length).toBe(0);
    expect(p.x_lines.submobjects.length).toBe(3);
    expect(p.y_lines.submobjects.length).toBe(5);
    expect(p.family_members_with_points().length).toBe(10);
    expectRowClose(p.c2p(1, 1), [1, 1, 0]);
  });

  it("faded_line_ratio=2 (original manim: 20 background + 24 faded)", () => {
    const p = new NumberPlane(null, null, null, null, null, null, 2);
    expect(p.background_lines?.submobjects.length).toBe(20);
    expect(p.faded_lines?.submobjects.length).toBe(24);
  });
});

describe("PolarPlane ground truth (original manim)", () => {
  it("default plane (original manim: bg 25, faded 0, origin [0,0,0], c2p(1,1)=[1,1,0])", () => {
    const pp = new PolarPlane();
    expect(pp.get_axes().submobjects.length).toBe(2);
    expect(pp.background_lines?.submobjects.length).toBe(25);
    expect(pp.faded_lines?.submobjects.length).toBe(0);
    expectRowClose(pp.get_origin(), [0, 0, 0]);
    expectRowClose(pp.c2p(1, 1), [1, 1, 0]);
  });
});

describe("ComplexPlane ground truth (original manim)", () => {
  it("default plane (original manim: bg 22, n2p(2+1j)=[2,1,0], 22 default coords)", () => {
    const cp = new ComplexPlane();
    expect(cp.background_lines?.submobjects.length).toBe(22);
    expectRowClose(cp.n2p({ re: 2, im: 1 }), [2, 1, 0]);
    const back = cp.p2n([2, 1, 0]);
    expectClose(back.re, 2);
    expectClose(back.im, 1);
    expect(cp._get_default_coordinate_values().length).toBe(22);
  });
});

describe("ThreeDAxes ground truth (original manim)", () => {
  it("builds the z-axis and maps coordinates (zu 0.8125, xu 0.875)", () => {
    const t = new ThreeDAxes();
    expect(t.get_axes().submobjects.length).toBe(3);
    expect(t.get_z_axis()).toBeInstanceOf(NumberLine);
    expectClose(t.get_z_unit_size(), 0.8125);
    expectClose(t.get_x_unit_size(), 0.875);
    expectClose(t.get_y_unit_size(), 1.05);
    expectRowClose(t.c2p(0, 0, 1), [0, 0, 0.8125]);
    expectRowClose(t.c2p(1, 0, 0), [0.875, 0, 0]);
    expectRowClose(t.c2p(0, 1, 0), [0, 1.05, 0]);
    expect(t.x_range).toEqual([-6, 6, 1]);
    expect(t.z_range).toEqual([-4, 4, 1]);
  });

  it("plot_surface faces carry the faithful nonzero z (original manim corner ~0.067708)", () => {
    const t = new ThreeDAxes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const surf = t.plot_surface((u, v) => u + v, [0, 1], [0, 1], null, 2, {
      resolution: 4,
    });
    expect(surf.submobjects.length).toBe(16);
    const face0 = surf.list_of_faces[0]!;
    const corners = rows(face0.points);
    expect(corners.length).toBeGreaterThanOrEqual(4);
    expectClose((corners[1]?.[2] ?? NaN), 0.067708, 1e-6);
    expect(Math.abs(corners[1]?.[2] ?? 0)).toBeGreaterThan(0);
  });
});

describe("downstream Mobject integration (planes are real VMobjects)", () => {
  it("Scene.add(axes) + family traversal + restructure work", () => {
    const scene = new Scene();
    const axes = new Axes();
    scene.add(axes as unknown as Mobject);
    expect(scene.mobjects.includes(axes as unknown as Mobject)).toBe(true);
    const family = scene.get_mobject_family_members();
    // family includes the axes and its x/y NumberLines.
    expect(family.length).toBeGreaterThanOrEqual(3);
    expect(family.includes(axes.get_x_axis() as unknown as Mobject)).toBe(true);
  });

  it("VectorScene.add_plane/add_axes return Mobject-backed planes", () => {
    const vs = new VectorScene();
    const plane = vs.add_plane();
    const axes = vs.add_axes();
    expect(plane).toBeInstanceOf(NumberPlane);
    expect(axes).toBeInstanceOf(Axes);
    expect(vs.mobjects.includes(plane as unknown as Mobject)).toBe(true);
    expect(vs.mobjects.includes(axes as unknown as Mobject)).toBe(true);
  });

  it("LinearTransformationScene unit square is 1x1 at [0.5,0.5,0]", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const sq = scene.get_unit_square();
    expectClose(sq.width, 1);
    expectClose(sq.height, 1);
    expectRowClose(sq.get_center(), [0.5, 0.5, 0]);
    expect(sq).toBeInstanceOf(VMobject);
  });
});
