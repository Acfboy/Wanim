import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Line, DashedLine, DoubleArrow, Arrow } from "../src/mobject/geometry/line.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { StealthTip, ArrowCircleTip } from "../src/mobject/geometry/tips.js";
import { Union, Difference, Intersection, Exclusion } from "../src/mobject/geometry/boolean_ops.js";
import { pointRows } from "../src/utils/arrays.js";
import { computeSubpathRanges } from "../src/renderer/canvas.js";
import { MobjectTable } from "../src/mobject/table.js";
import { SampleSpace } from "../src/mobject/graphing/probability.js";
import { Axes, ComplexPlane } from "../src/mobject/graphing/coordinate_systems.js";
import {
  Dot3D,
  Prism,
  Cone,
  Cylinder,
  Line3D,
  Torus,
} from "../src/mobject/three_d/three_dimensions.js";
import { Tetrahedron, Polyhedron } from "../src/mobject/three_d/polyhedra.js";
import { LEFT, RIGHT } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= Math.max(1e-9, Math.abs(b) * tol);
}

function expectVecClose(actual: number[] | NDArray, expected: number[] | NDArray, tol = 1e-6): void {
  const arr: number[] = Array.isArray(actual)
    ? [...actual]
    : [...(actual.toArray() as unknown as number[])].map(Number);
  const exp: number[] = Array.isArray(expected)
    ? [...expected]
    : [...(expected.toArray() as unknown as number[])].map(Number);
  expect(arr.length).toBe(exp.length);
  for (let i = 0; i < exp.length; i += 1) {
    expect(close(arr[i] ?? NaN, exp[i] ?? NaN, tol)).toBe(true);
  }
}

// Ground truths captured from the original manim implementation (manim 0.21.0):
// Square(2.0): width=2 height=2 side_length=2,
//   verts=[[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]
// Line([-1,0,0],[2,0,0]): len 3; proj of [0,1,0] -> [0,0,0]
// set_points_by_ends([1,2,0],[3,4,0]) -> start [1,2,0] end [3,4,0]
// DashedLine default: start [-1,0,0] end [1,0,0]
// Square() gradient start/end: [[1,-1,0],[-1,1,0]]
// StealthTip: length 0.175, 16 points; ArrowCircleTip: width=height=0.35
// Axes slopes for x^2: at -2 -> -4.0, at 0 -> ~1e-8, at 1 -> 2.0
// ComplexPlane p2n roundtrip of 1+2j -> {re:1, im:2}; SampleSpace w=h=3
// complete_p_list([0.3,0.3])=[0.3,0.3,0.4]; complete_p_list(0.5)=[0.5,0.5]
// Dot3D([1,2,3]) center [1,2,3]; Prism dims [3,2,1]
// Cone(2,3): base width 4.0, new_height 3
// Cylinder(1,2).add_bases(): submobjects grow by 2; base radii 1
// Line3D([0,0,0],[2,2,2]): len sqrt(12), dir [1,1,1]/sqrt(3)
// Torus R=3 r=1; Tetrahedron faces=4; create_faces single tri -> 1 face, 12 pts
// MobjectTable 2x2: elements=4, rows=2, cols=2
// Boolean ops are computed by the local polygon engine (no skia `pathops`
// dependency); only the <2-input ValueError arity checks mirror Python
// exactly, the rest is validated by area below and in boolean-ops.test.ts.

/** Absolute area summed over a VMobject's subpaths (exact for polylines). */
function booleanArea(mob: VMobject): number {
  const rows = pointRows(mob.points, mob.dim).map(
    (row) => [Number(row[0] ?? 0), Number(row[1] ?? 0)],
  );
  const ranges = computeSubpathRanges(rows, mob.n_points_per_cubic_curve, 1e-6);
  let total = 0;
  for (const [start, end] of ranges) {
    const s = start as number;
    const e = end as number;
    if (e - s < 3) continue;
    for (let i = s; i < e; i += 1) {
      const a = rows[i] ?? [0, 0];
      const b = rows[(i + 1 - s) % (e - s) + s] ?? [0, 0];
      total += (a[0] ?? 0) * (b[1] ?? 0) - (b[0] ?? 0) * (a[1] ?? 0);
    }
  }
  return Math.abs(total) / 2;
}

describe("Square", () => {
  it("side_length=2 gives width/height/side_length 2", () => {
    const s = new Square(2.0);
    expect(close(s.width, 2.0)).toBe(true);
    expect(close(s.height, 2.0)).toBe(true);
    expect(close(s.side_length, 2.0)).toBe(true);
  });
  it("vertices match python order", () => {
    const s = new Square(2.0);
    const verts = pointRows(s.get_vertices(), 3);
    const expected = [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]];
    expect(verts.length).toBe(4);
    for (let i = 0; i < 4; i += 1) expectVecClose(verts[i] ?? [], expected[i] ?? []);
  });
  it("side_length=1 gives width 1; setter rescales", () => {
    const s = new Square(1.0);
    expect(close(s.width, 1.0)).toBe(true);
    s.side_length = 2.0;
    expect(close(s.width, 2.0, 1e-6)).toBe(true);
    expect(close(s.side_length, 2.0, 1e-6)).toBe(true);
  });
});

describe("Line.set_points_by_ends / get_projection", () => {
  it("basic line start/end/length", () => {
    const l = new Line([-1, 0, 0], [2, 0, 0]);
    expectVecClose(l.get_start(), [-1, 0, 0]);
    expectVecClose(l.get_end(), [2, 0, 0]);
    expect(close(l.get_length(), 3.0)).toBe(true);
  });
  it("get_projection drops perpendicular component", () => {
    const l = new Line([-1, 0, 0], [2, 0, 0]);
    expectVecClose(l.get_projection([0, 1, 0]), [0, 0, 0]);
    expectVecClose(l.get_projection([0, 0, 5]), [0, 0, 0]);
    // diagonal line projection
    const d = new Line([0, 0, 0], [2, 0, 0]);
    expectVecClose(d.get_projection([1, 1, 0]), [1, 0, 0]);
  });
  it("set_points_by_ends re-targets the line", () => {
    const l = new Line([0, 0, 0], [0, 0, 0]);
    l.set_points_by_ends([1, 2, 0], [3, 4, 0]);
    expectVecClose(l.get_start(), [1, 2, 0]);
    expectVecClose(l.get_end(), [3, 4, 0]);
  });
});

describe("DashedLine.get_start/get_end", () => {
  it("returns first/last dash endpoints matching full line", () => {
    const dl = new DashedLine([-1, 0, 0], [1, 0, 0]);
    expectVecClose(dl.get_start(), [-1, 0, 0], 1e-4);
    expectVecClose(dl.get_end(), [1, 0, 0], 1e-4);
  });
  it("empty dashed line raises ValueError like Python", () => {
    // Ground truth: Python raises
    // `ValueError: Cannot call DashedLine.get_start because DashedLine has no
    // points` once submobjects are emptied (own points were cleared in the
    // constructor), and the TS port mirrors that.
    const dl = new DashedLine([-1, 0, 0], [1, 0, 0]);
    dl.submobjects = [];
    expect(() => dl.get_start()).toThrow(/no points/);
    expect(() => dl.get_end()).toThrow(/no points/);
  });
});

describe("DoubleArrow", () => {
  it("has tips on both ends", () => {
    const da = new DoubleArrow([-1, 0, 0], [1, 0, 0]);
    // Python: 2 submobjects (the two tips)
    expect(da.submobjects.length).toBe(2);
    expect(da instanceof Arrow).toBe(true);
  });
  it("start tip sits left of end tip (end tip added first)", () => {
    const da = new DoubleArrow([-1, 0, 0], [1, 0, 0]);
    expect(da.submobjects.length).toBe(2);
    // Arrow.__init__ adds the end tip first, DoubleArrow appends the start
    // tip second, so submobjects[0] is the right/end tip.
    const c0 = (da.submobjects[0] as VMobject).get_center();
    const c1 = (da.submobjects[1] as VMobject).get_center();
    expect(c0[0] ?? 0).toBeGreaterThan(c1[0] ?? 0);
  });
});

describe("VMobject.get_gradient_start_and_end_points", () => {
  it("matches python for default Square", () => {
    const v = new Square();
    const [a, b] = v.get_gradient_start_and_end_points();
    expectVecClose(a, [1, -1, 0]);
    expectVecClose(b, [-1, 1, 0]);
  });
  it("is symmetric about the center", () => {
    const v = new Square();
    const [a, b] = v.get_gradient_start_and_end_points();
    const c = v.get_center();
    for (let i = 0; i < 3; i += 1) {
      expect(close(((a[i] ?? 0) + (b[i] ?? 0)) / 2, c[i] ?? 0)).toBe(true);
    }
  });
});

describe("StealthTip / ArrowCircleTip", () => {
  it("StealthTip length ~0.175 with 16 points", () => {
    const st = new StealthTip();
    expect(close(st.get_length(), 0.175, 1e-6)).toBe(true);
    expect(st.get_num_points()).toBe(16);
  });
  it("ArrowCircleTip width/height 0.35", () => {
    const ac = new ArrowCircleTip();
    expect(close(ac.width, 0.35)).toBe(true);
    expect(close(ac.height, 0.35)).toBe(true);
  });
});

describe("MobjectTable", () => {
  it("2x2 table of Lines: elements/rows/columns", () => {
    const t = new MobjectTable([
      [new Line(), new Line()],
      [new Line(), new Line()],
    ]);
    expect(t.elements.submobjects.length).toBe(4);
    expect(t.get_rows().submobjects.length).toBe(2);
    expect(t.get_columns().submobjects.length).toBe(2);
    expect(t.mob_table.length).toBe(2);
    expect(t.mob_table[0]?.length).toBe(2);
  });
  it("uses identity element_to_mobject (entries preserved)", () => {
    const a = new Line([-1, 0, 0], [1, 0, 0]);
    const t = new MobjectTable([[a]]);
    expect(t.mob_table[0]?.[0]).toBe(a);
  });
});

describe("SampleSpace", () => {
  it("defaults width/height 3, label scale 1", () => {
    const ss = new SampleSpace();
    expect(close(ss.width, 3.0)).toBe(true);
    expect(close(ss.height, 3.0)).toBe(true);
    expect(ss.default_label_scale_val).toBe(1);
  });
  it("complete_p_list appends remainder", () => {
    const ss = new SampleSpace();
    expect(ss.complete_p_list([0.3, 0.3])).toEqual([0.3, 0.3, 0.4]);
    expect(ss.complete_p_list(0.5)).toEqual([0.5, 0.5]);
  });
  it("vertical division of [0.5,0.5] gives 2 parts", () => {
    const ss = new SampleSpace();
    const div = ss.get_vertical_division([0.5, 0.5]);
    expect(div.submobjects.length).toBe(2);
  });
  it("add_label stores label; add_title builds a Tex", () => {
    const ss = new SampleSpace();
    ss.add_label("hello");
    expect(ss.label).toBe("hello");
    ss.add_title();
    expect(ss.title).toBeDefined();
    expect(
      (ss.title as unknown as { tex_string: string }).tex_string,
    ).toContain("Sample space");
  });
});

describe("CoordinateSystem.get_area / slope_of_tangent", () => {
  it("slope_of_tangent of x^2 matches python", () => {
    const ax = new Axes();
    const curve = ax.plot((x: number) => x ** 2);
    expect(close(ax.slope_of_tangent(-2, curve), -4.0, 1e-4)).toBe(true);
    // Python ground truth at x=0 is 1e-08 (finite-difference artifact of the
    // default dx=1e-8 step), not exactly 0.
    expect(close(ax.slope_of_tangent(0, curve), 1e-8, 1e-2)).toBe(true);
    expect(close(ax.slope_of_tangent(1, curve), 2.0, 1e-4)).toBe(true);
  });
  it("get_area returns a polygon with opacity 0.3 and nonzero points", () => {
    const ax = new Axes();
    const curve = ax.plot((x: number) => x ** 2);
    const area = ax.get_area(curve, [0, 1]);
    expect(area.get_num_points()).toBeGreaterThan(0);
    expect(close(area.fill_opacity, 0.3)).toBe(true);
  });
  it("get_area with bounded_graph encloses between curves", () => {
    const ax = new Axes();
    const curve = ax.plot((x: number) => x ** 2);
    const bg = ax.plot((x: number) => x);
    const area = ax.get_area(curve, [0, 1], undefined, 0.3, bg);
    expect(area.get_num_points()).toBeGreaterThan(0);
  });
  it("get_area raises ValueError on non-overlapping bounded range", () => {
    const ax = new Axes();
    const curve = ax.plot((x: number) => x ** 2);
    const bg = ax.plot((x: number) => x, [5, 6] as unknown as number[]);
    expect(() => ax.get_area(curve, [0, 1], undefined, 0.3, bg)).toThrow(/Ranges not matching/);
  });
});

describe("ComplexPlane.point_to_number", () => {
  it("origin maps to 0 and n2p roundtrips", () => {
    const cp = new ComplexPlane();
    const z0 = cp.point_to_number([0, 0, 0]);
    expect(close(z0.re, 0)).toBe(true);
    expect(close(z0.im, 0)).toBe(true);
    const p = cp.n2p({ re: 1, im: 2 });
    const back = cp.point_to_number([...(p.toArray() as unknown as number[])]);
    expect(close(back.re, 1, 1e-6)).toBe(true);
    expect(close(back.im, 2, 1e-6)).toBe(true);
  });
});

describe("Dot3D / Prism / Cone / Cylinder / Line3D / Torus", () => {
  it("Dot3D centers at the given point", () => {
    const d = new Dot3D([1, 2, 3], 0.5);
    expectVecClose(d.get_center(), [1, 2, 3], 1e-4);
  });
  it("Prism stores dimensions", () => {
    const p = new Prism([3, 2, 1]);
    expect([...p.dimensions]).toEqual([3, 2, 1]);
    expect(p.submobjects.length).toBeGreaterThan(0);
  });
  it("Cone base width 4 for base_radius 2; new_height 3", () => {
    const c = new Cone(2, 3);
    expect(close(c.base_circle.width, 4.0, 1e-6)).toBe(true);
    expect(close(c.new_height, 3)).toBe(true);
    expect(close(c.theta, Math.PI - Math.atan(2 / 3))).toBe(true);
  });
  it("Cylinder.add_bases adds two circle caps of matching radius", () => {
    const cyl = new Cylinder(1, 2, [...[0, 0, 1]], [0, Math.PI * 2], false);
    const before = cyl.submobjects.length;
    cyl.add_bases();
    expect(cyl.submobjects.length).toBe(before + 2);
    expect(close(cyl.base_top.radius, 1)).toBe(true);
    expect(close(cyl.base_bottom.radius, 1)).toBe(true);
  });
  it("Line3D endpoints/length/direction match python", () => {
    const l = new Line3D([0, 0, 0], [2, 2, 2]);
    expectVecClose([...(l.get_start().toArray() as unknown as number[])], [0, 0, 0]);
    expectVecClose([...(l.get_end().toArray() as unknown as number[])], [2, 2, 2]);
    expect(close(l.length, Math.sqrt(12))).toBe(true);
    const inv = 1 / Math.sqrt(3);
    expectVecClose(l.direction, [inv, inv, inv]);
    expect(l.resolution).toEqual([2, 24]);
  });
  it("Torus stores R and r", () => {
    const t = new Torus(3, 1);
    expect(t.R).toBe(3);
    expect(t.r).toBe(1);
  });
});

describe("Polyhedron.create_faces", () => {
  it("tetrahedron has 4 faces", () => {
    const t = new Tetrahedron();
    expect(t.faces.submobjects.length).toBe(4);
  });
  it("single triangle face builds a 12-point polygon", () => {
    const t = new Tetrahedron();
    const g = (t as Polyhedron).create_faces([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]);
    expect(g.submobjects.length).toBe(1);
    expect((g.submobjects[0] as VMobject).get_num_points()).toBe(12);
  });
});

describe("boolean ops", () => {
  it("Union/Intersection reject fewer than 2 mobjects", () => {
    const s = new Square();
    expect(() => new Union(s)).toThrow(/At least 2 mobjects needed for Union/);
    expect(() => new Intersection(s)).toThrow(/At least 2 mobjects needed for Intersection/);
  });
  it("boolean combinators compute real geometry via the local engine", () => {
    // The port implements boolean ops on flattened polygons (no skia
    // `pathops` dependency). Two side-2 squares shifted by +-0.5 overlap in
    // a 1x2 lens: union 6, difference 2, intersection 2, exclusion 4.
    const s1 = new Square().shift([-0.5, 0, 0]);
    const s2 = new Square().shift([0.5, 0, 0]);
    expect(booleanArea(new Union(s1, s2))).toBeCloseTo(6, 0);
    expect(booleanArea(new Difference(s1, s2))).toBeCloseTo(2, 0);
    expect(booleanArea(new Intersection(s1, s2))).toBeCloseTo(2, 0);
    expect(booleanArea(new Exclusion(s1, s2))).toBeCloseTo(4, 0);
  });
});

// --- integration: batch symbols interacting with earlier-ported code ---

describe("integration", () => {
  it("DoubleArrow chain: projection onto arrow line + gradient endpoints", () => {
    const da = new DoubleArrow([-2, 0, 0], [2, 0, 0]);
    expectVecClose(da.get_projection([0, 3, 0]), [0, 0, 0]);
    const [g0, g1] = (da as unknown as VMobject).get_gradient_start_and_end_points();
    const c = da.get_center();
    for (let i = 0; i < 3; i += 1) {
      expect(close(((g0[i] ?? 0) + (g1[i] ?? 0)) / 2, c[i] ?? 0, 1e-6)).toBe(true);
    }
  });
  it("Axes plot -> slope -> area chain on sin curve", () => {
    const ax = new Axes();
    const curve = ax.plot((x: number) => 2 * Math.sin(x));
    const slope = ax.slope_of_tangent(0, curve);
    // d/dx 2 sin x at 0 = 2
    expect(close(slope, 2.0, 1e-4)).toBe(true);
    const area = ax.get_area(curve, [0, Math.PI]);
    expect(area.get_num_points()).toBeGreaterThan(0);
  });
  it("MobjectTable of Squares arranges 2x2 with real geometry", () => {
    const t = new MobjectTable([
      [new Square(1), new Square(1)],
      [new Square(1), new Square(1)],
    ]);
    expect(t.get_rows().submobjects.length).toBe(2);
    for (const row of t.get_rows().submobjects as unknown as VGroup[]) {
      expect((row as VGroup).submobjects.length).toBe(2);
    }
  });
  it("Line3D direction feeds Line.get_projection-style math consistently", () => {
    const l3 = new Line3D([0, 0, 0], [1, 0, 0]);
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expectVecClose(l3.direction, l.get_unit_vector());
  });
  it("SampleSpace vertical division widths sum to full width", () => {
    const ss = new SampleSpace();
    const div = ss.get_vertical_division([0.5, 0.5]);
    const parts = div.submobjects as VMobject[];
    const total = parts.reduce((s, p) => s + p.width, 0);
    expect(close(total, ss.width, 1e-6)).toBe(true);
  });
});
