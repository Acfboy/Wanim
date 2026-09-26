// Ground-truth tests for batch layers 18..18 (~1361 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): DecimalTable/DecimalMatrix/MobjectMatrix defaults,
// BraceText/BraceLabel string labels and BarChart axis numbers need the
// LaTeX pipeline (this container has no `latex` binary; Python raises
// FileNotFoundError there). The TS port keeps clearly-marked TODO stubs for
// those paths; tests assert the stub behavior plus the tex-free layout paths
// (documented inline as deviations D1..D5).
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import {
  Axes,
  ComplexPlane,
  CoordinateSystem,
  NumberPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine, UnitInterval } from "../src/mobject/graphing/number_line.js";
import { BarChart, SampleSpace } from "../src/mobject/graphing/probability.js";
import { ParametricFunction } from "../src/mobject/graphing/functions.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Line } from "../src/mobject/geometry/line.js";
import {
  ArrowCircleFilledTip,
  ArrowSquareTip,
  ArrowTriangleTip,
} from "../src/mobject/geometry/tips.js";
import {
  Annulus,
  AnnotationDot,
  CurvedDoubleArrow,
} from "../src/mobject/geometry/arc.js";
import {
  RoundedRectangle,
  Square,
  Triangle,
} from "../src/mobject/geometry/polygram.js";
import { SurroundingRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { Brace, BraceLabel, BraceText } from "../src/mobject/svg/brace.js";
import {
  Arrow3D,
  Cylinder,
  Dot3D,
} from "../src/mobject/three_d/three_dimensions.js";
import { Scene } from "../src/scene/scene.js";
import { DecimalTable } from "../src/mobject/table.js";
import {
  DecimalMatrix,
  Matrix,
  MobjectMatrix,
  matrix_to_tex_string,
} from "../src/mobject/matrix.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(actual[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function makeAxes(): Axes {
  return new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
}

function centerOf(m: Mobject): number[] {
  return m.get_center();
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Square.__init__
// Python: Square(2.0) verts [[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]],
//   width 2.0, side_length 2.0.
// ---------------------------------------------------------------------------
describe("Square.__init__", () => {
  it("builds a side-2 square at the origin", () => {
    const s = new Square(2.0);
    expectRowsClose(rowsOf(s.get_vertices(), 3), [
      [1.0, 1.0, 0.0],
      [-1.0, 1.0, 0.0],
      [-1.0, -1.0, 0.0],
      [1.0, -1.0, 0.0],
    ]);
    expectClose(s.width, 2.0);
    expectClose(s.side_length, 2.0);
    expectRowsClose([centerOf(s)], [[0, 0, 0]]);
  });

  it("honors a custom side length", () => {
    // Python: Square(3.0).width == 3.0, verts at +/-1.5.
    const s = new Square(3.0);
    expectClose(s.width, 3.0);
    expectRowsClose(rowsOf(s.get_vertices(), 3), [
      [1.5, 1.5, 0],
      [-1.5, 1.5, 0],
      [-1.5, -1.5, 0],
      [1.5, -1.5, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Triangle.__init__
// Python: Triangle() verts [[~0,1,0],[-0.8660254,-0.5,0],[0.8660254,-0.5,0]].
// ---------------------------------------------------------------------------
describe("Triangle.__init__", () => {
  it("builds the unit-circumradius triangle", () => {
    const t = new Triangle();
    expectRowsClose(rowsOf(t.get_vertices(), 3), [
      [2.220446049250313e-16, 1.0, 0.0],
      [-0.8660254037844389, -0.4999999999999996, 0.0],
      [0.8660254037844383, -0.5000000000000006, 0.0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.RoundedRectangle.__init__
// Python: RoundedRectangle(corner_radius=0.5, width=4, height=2) ->
//   width 4.0, height 2.0, corner_radius 0.5, 32 points.
// ---------------------------------------------------------------------------
describe("RoundedRectangle.__init__", () => {
  it("keeps width/height and rounds the corners", () => {
    const rr = new RoundedRectangle({ corner_radius: 0.5, width: 4, height: 2 });
    expectClose(rr.width, 4.0);
    expectClose(rr.height, 2.0);
    expect(rr.corner_radius).toBe(0.5);
    expect(rr.get_num_points()).toBe(32);
    expectRowsClose([centerOf(rr)], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.probability.SampleSpace.__init__
// Python: SampleSpace() -> width 3.0, height 3.0, center origin.
// ---------------------------------------------------------------------------
describe("SampleSpace.__init__", () => {
  it("builds a 3x3 square at the origin", () => {
    const ss = new SampleSpace();
    expectClose(ss.width, 3.0);
    expectClose(ss.height, 3.0);
    expectRowsClose([centerOf(ss)], [[0, 0, 0]]);
  });

  it("honors custom height/width", () => {
    // Python: SampleSpace(height=2, width=4) -> 4 x 2.
    const ss = new SampleSpace({ height: 2, width: 4 });
    expectClose(ss.width, 4.0);
    expectClose(ss.height, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.probability.BarChart._create_bar
// Python (BarChart-equivalent Axes: x_range [0,3,1], y_range [0,3,0.75],
// x_length 3, y_length 4; grafted via __new__ to skip the latex number
// labels, then _create_bar):
//   bar0 val=1 w=0.6 h=1.333.. center=[-1,-1.333..,0]
//   bar1 val=2 w=0.6 h=2.666.. center=[~0,-0.666..,0]
//   bar2 val=3 w=0.6 h=4.0 center=[1,0,0]
//   neg val=-1.5 (bar 0): h=2.0 center=[-1,-3,0]; zero val: h=0.
// ---------------------------------------------------------------------------
describe("BarChart._create_bar", () => {
  function chart(): BarChart {
    return new BarChart([1, 2, 3]);
  }

  it("creates positioned bars matching the axis mapping", () => {
    const bc = chart();
    const b0 = bc._create_bar(0, 1);
    expectClose(b0.width, 0.6);
    expectClose(b0.height, 4 / 3);
    expectRowsClose([centerOf(b0)], [[-1, -4 / 3, 0]]);
    const b1 = bc._create_bar(1, 2);
    expectClose(b1.width, 0.6);
    expectClose(b1.height, 8 / 3);
    expectRowsClose([centerOf(b1)], [[0, -2 / 3, 0]]);
    const b2 = bc._create_bar(2, 3);
    expectClose(b2.width, 0.6);
    expectClose(b2.height, 4.0);
    expectRowsClose([centerOf(b2)], [[1, 0, 0]]);
  });

  it("handles negative and zero values", () => {
    // Python: _create_bar(0, -1.5) -> h=2.0 center [-1,-3,0];
    //   _create_bar(1, 0.0) -> h=0.0 center [~0,-2,0].
    const bc = chart();
    const neg = bc._create_bar(0, -1.5);
    expectClose(neg.height, 2.0);
    expectRowsClose([centerOf(neg)], [[-1, -3, 0]]);
    const zero = bc._create_bar(1, 0.0);
    expectClose(zero.height, 0.0);
    expectRowsClose([centerOf(zero)], [[0, -2, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.__init__
// Python: CoordinateSystem() -> dimension 2, x_range [-7,7,1],
//   y_range [-4,4,1], x_length None, y_length None.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.__init__", () => {
  it("stores frame-derived default ranges", () => {
    const cs = new CoordinateSystem();
    expect(cs.dimension).toBe(2);
    expect(cs.x_range).toEqual([-7, 7, 1]);
    expect(cs.y_range).toEqual([-4, 4, 1]);
    expect(cs.x_length).toBeNull();
    expect(cs.y_length).toBeNull();
  });

  it("honors explicit ranges and lengths", () => {
    // Python: CoordinateSystem([-3,3,1], [-2,2,1], 6, 4).
    const cs = new CoordinateSystem([-3, 3, 1], [-2, 2, 1], 6, 4);
    expect(cs.x_range).toEqual([-3, 3, 1]);
    expect(cs.y_range).toEqual([-2, 2, 1]);
    expect(cs.x_length).toBe(6);
    expect(cs.y_length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.Axes
// Python: Axes() -> x_range [-7,7,1], y_range [-4,4,1], x_length 12,
//   y_length 6; Axes([-3,3,1],[-2,2,1],6,4).c2p(1,1) == [1,1,0].
// ---------------------------------------------------------------------------
describe("Axes.__init__", () => {
  it("uses frame-derived defaults", () => {
    const ax = new Axes();
    expect(ax.x_range).toEqual([-7, 7, 1]);
    expect(ax.y_range).toEqual([-4, 4, 1]);
    expect(ax.x_length).toBe(12);
    expect(ax.y_length).toBe(6);
  });

  it("maps coordinates linearly with explicit ranges", () => {
    const ax = makeAxes();
    expect(ax.x_length).toBe(6);
    expect(ax.y_length).toBe(4);
    const p = ax.c2p(1, 1).toArray() as number[];
    expectRowsClose([[...p]], [[1.0000000000000002, 1.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_riemann_rectangles
// Python (ax as above, g = plot(x^2), x_range=[0,2], dx=0.5):
//   4 rects, width*height areas [0.0, 0.125125, 0.5005, 1.126125].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_riemann_rectangles", () => {
  it("builds left-sample rectangles with ground-truth areas", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const rects = ax.get_riemann_rectangles(g, [0, 2], 0.5);
    expect(rects.submobjects.length).toBe(4);
    const areas = (rects.submobjects as VMobject[]).map((r) => r.width * r.height);
    const want = [0.0, 0.125125, 0.5005, 1.126125];
    expect(areas.length).toBe(want.length);
    for (let i = 0; i < want.length; i += 1) expectClose(areas[i] ?? NaN, want[i] ?? NaN, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_derivative_graph.deriv
// Python: deriv(x) == ax.slope_of_tangent(x, g) with g = plot(x^2):
//   x=0 -> 1e-08; x=0.5 -> 1.0000000111022302; x=1 -> 1.9999999999999996;
//   x=2 -> 4.000000000000002.
// (In TS the closure is inline in plot_derivative_graph; the ported
// underlying_function of the returned graph IS deriv, so it is probed
// directly together with slope_of_tangent.)
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_derivative_graph.deriv", () => {
  it("matches slope_of_tangent samples for x^2", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    expectClose(ax.slope_of_tangent(0.0, g), 1e-8, 1e-6);
    expectClose(ax.slope_of_tangent(0.5, g), 1.0000000111022302);
    expectClose(ax.slope_of_tangent(1.0, g), 1.9999999999999996);
    expectClose(ax.slope_of_tangent(2.0, g), 4.000000000000002);
  });

  it("plots the derivative graph through (1, 2)", () => {
    // Python: dg = plot_derivative_graph(g) has 240 points; the curve passes
    // through c2p(1, 2) == [1, 2, 0] (derivative of x^2 at x=1 is 2).
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const dg = ax.plot_derivative_graph(g);
    expect(dg).toBeInstanceOf(ParametricFunction);
    expect(dg.get_num_points()).toBe(240);
    const deriv = (dg as unknown as { underlying_function: (x: number) => number })
      .underlying_function;
    expectClose(deriv(1.0), 2.0);
    const target = [1.0, 2.0, 0.0];
    const rows = rowsOf(dg.points, dg.dim);
    let best = Infinity;
    for (const row of rows) {
      const d = Math.hypot(
        (row[0] ?? 0) - target[0]!,
        (row[1] ?? 0) - target[1]!,
        (row[2] ?? 0) - target[2]!,
      );
      if (d < best) best = d;
    }
    expect(best < 0.05).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ComplexPlane.p2n
// Python: ComplexPlane() (kwargs-only ctor, unrounded frame ranges
// (-7.111.., 7.111.., 1) x (-4, 4, 1), unit-sized axes):
//   p2n([0,0,0]) == 0j; n2p(1+1j) == [1,1,0]; p2n(n2p(1+2j)) == (1+2j);
//   ComplexPlane(x_range=[-3,3,1], y_range=[-2,2,1], x_length=6, y_length=4):
//   n2p(1+1j) == [1,1,0].
// ---------------------------------------------------------------------------
describe("ComplexPlane.p2n", () => {
  it("inverts n2p on the default plane", () => {
    const cp = new ComplexPlane();
    expect(cp.x_range[0] ?? NaN).toBeCloseTo(-7.111111111111111, 9);
    const origin = cp.p2n([0, 0, 0]);
    expectClose(origin.re, 0);
    expectClose(origin.im, 0);
    const p = cp.n2p({ re: 1, im: 1 }).toArray() as number[];
    expectRowsClose([[ ...p]], [[1.0000000000000002, 1.0, 0.0]]);
    const back = cp.p2n(cp.n2p({ re: 1, im: 2 }).toArray() as number[]);
    expectClose(back.re, 1);
    expectClose(back.im, 2);
  });

  it("maps unit-sized on an explicit plane", () => {
    // Python: explicit ranges/lengths behave like Axes.
    const cp = new ComplexPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const p = cp.n2p({ re: 1, im: 1 }).toArray() as number[];
    expectRowsClose([[ ...p]], [[1.0000000000000002, 1.0, 0.0]]);
    const back = cp.p2n([...p]);
    expectClose(back.re, 1);
    expectClose(back.im, 1);
  });
});

// ---------------------------------------------------------------------------
// NumberPlane null-length mapping (supporting ComplexPlane.p2n defaults)
// Python: NumberPlane() -> x_range (-7.111.., 7.111.., 1) (UNROUNDED frame
// radii, unlike Axes), lengths None, unit-sized axes about the midpoint:
//   c2p(0,0)==[0,0,0], c2p(1,1)==[1,1,0];
// NumberPlane(x_range=[0,10,1], y_range=[0,5,1]) (lengths None):
//   c2p(0,0)==[-5,-2.5,0], c2p(5,2.5)==[0,0,0], c2p(10,5)==[5,2.5,0].
// ---------------------------------------------------------------------------
describe("NumberPlane null-length unit mapping", () => {
  it("uses raw frame radii and unit size by default", () => {
    const np = new NumberPlane();
    expectClose(np.x_range[0] ?? NaN, -7.111111111111111);
    expectClose(np.x_range[1] ?? NaN, 7.111111111111111);
    expect(np.x_length).toBeNull();
    expectRowsClose([[...(np.c2p(0, 0).toArray() as number[])]], [[0, 0, 0]]);
  });

  it("centers asymmetric ranges on their midpoint at unit size", () => {
    const np = new NumberPlane([0, 10, 1], [0, 5, 1]);
    expectRowsClose([[...(np.c2p(0, 0).toArray() as number[])]], [[-5.000000000000001, -2.5, 0.0]]);
    expectRowsClose([[...(np.c2p(5, 2.5).toArray() as number[])]], [[0, 0, 0]]);
    expectRowsClose([[...(np.c2p(10, 5).toArray() as number[])]], [[5.0, 2.5, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.n2p
// Python: NumberLine(x_range=[-3,3,1], length=6): n2p(0)==[0,0,0],
//   n2p(3)==[3,0,0], n2p(-3)==[-3,0,0], n2p(-1.5)==[-1.5,0,0].
// ---------------------------------------------------------------------------
describe("NumberLine.n2p", () => {
  it("maps numbers linearly onto the line", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    for (const [input, want] of [
      [0, [0, 0, 0]],
      [3, [3, 0, 0]],
      [-3, [-3, 0, 0]],
      [-1.5, [-1.5, 0, 0]],
    ] as Array<[number, number[]]>) {
      const got = [...(nl.n2p(input).toArray() as number[])];
      expectRowsClose([got], [want]);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.UnitInterval
// Python: UnitInterval(unit_size=10): x_range [0,1,0.1], unit_size 10,
//   length 10.0; n2p(0)==[-5,0,0], n2p(1)==[5,0,0], n2p(0.5)==[0,0,0].
// ---------------------------------------------------------------------------
describe("UnitInterval", () => {
  it("spans [0,1] with the unit size as its length", () => {
    const ui = new UnitInterval(10);
    expect(ui.unit_size).toBe(10);
    expectClose(ui.x_range[0] ?? NaN, 0);
    expectClose(ui.x_range[1] ?? NaN, 1);
    expectRowsClose([[...(ui.n2p(0).toArray() as number[])]], [[-5, 0, 0]]);
    expectRowsClose([[...(ui.n2p(1).toArray() as number[])]], [[5, 0, 0]]);
    expectRowsClose([[...(ui.n2p(0.5).toArray() as number[])]], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Dot3D.__init__
// Python: Dot3D([1,2,3], radius=0.08) -> center [1,2,3], width 0.16.
// ---------------------------------------------------------------------------
describe("Dot3D.__init__", () => {
  it("places a small sphere at the given point", () => {
    const d = new Dot3D([1, 2, 3], 0.08);
    expectRowsClose([centerOf(d as unknown as Mobject)], [[1, 2, 3]]);
    expectClose((d as unknown as Mobject).width, 0.16000000000000003);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cylinder.__init__
// Python: Cylinder(radius=1, height=2) -> height 2.0, width 2.0, depth 2.0;
//   Cylinder(radius=2, height=3) (axis along Z) -> width 4.0, depth 3.0.
// ---------------------------------------------------------------------------
describe("Cylinder.__init__", () => {
  it("builds a z-axis cylinder with ground-truth extents", () => {
    const c = new Cylinder(1, 2);
    const m = c as unknown as Mobject;
    expectClose(m.height, 2.0);
    expectClose(m.width, 2.0);
    const c2 = new Cylinder(2, 3) as unknown as Mobject;
    expectClose(c2.width, 4.0);
    expectClose(c2.depth, 3.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Arrow3D
// Python: Arrow3D([0,0,0],[0,0,3]): get_start()==[0,0,0],
//   get_end()==[0,0,3].
// ---------------------------------------------------------------------------
describe("Arrow3D", () => {
  it("keeps the original endpoints", () => {
    const a = new Arrow3D([0, 0, 0], [0, 0, 3]);
    expectRowsClose([[...(a.get_start().toArray() as number[])]], [[0, 0, 0]]);
    expectRowsClose([[...(a.get_end().toArray() as number[])]], [[0, 0, 3]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.generate_points
// Python: Line([1,2,3],[4,6,3]) -> start [1,2,3], end [4,6,3], 4 points;
//   put_start_and_end_on([2,3,0],[5,7,0]) moves both ends.
// ---------------------------------------------------------------------------
describe("Line.generate_points", () => {
  it("interpolates start/end with 4 points", () => {
    const l = new Line([1, 2, 3], [4, 6, 3]);
    expectRowsClose([l.get_start()], [[1, 2, 3]]);
    expectRowsClose([l.get_end()], [[4, 6, 3]]);
    expect(l.get_num_points()).toBe(4);
  });

  it("re-points via put_start_and_end_on", () => {
    // Python: put_start_and_end_on([2,3,0],[5,7,0]).
    const l = new Line([0, 0, 0], [1, 0, 0]);
    l.put_start_and_end_on([2, 3, 0], [5, 7, 0]);
    expectRowsClose([l.get_start()], [[2, 3, 0]]);
    expectRowsClose([l.get_end()], [[5, 7, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.get_angle
// Python: Line([0,0,0],[3,4,0]).get_angle() == atan2(4,3) ==
//   0.9272952180016122; Line to [-1,0,0] -> pi.
// ---------------------------------------------------------------------------
describe("Line.get_angle", () => {
  it("returns atan2(dy, dx) of the line vector", () => {
    expectClose(new Line([0, 0, 0], [3, 4, 0]).get_angle(), 0.9272952180016122);
    expectClose(new Line([0, 0, 0], [-1, 0, 0]).get_angle(), Math.PI);
    expectClose(new Line([0, 0, 0], [0, 1, 0]).get_angle(), Math.PI / 2);
  });
});

// ---------------------------------------------------------------------------
// Arrow tips: ArrowTriangleTip / ArrowCircleFilledTip / ArrowSquareTip
// Python: widths/heights all 0.35; point counts 12 (triangle), 32
// (filled circle), 16 (square).
// ---------------------------------------------------------------------------
describe("arrow tips", () => {
  it("builds the triangle tip at the default size", () => {
    const t = new ArrowTriangleTip();
    expectClose(t.width, 0.35);
    expectClose(t.height, 0.35);
    expect(t.get_num_points()).toBe(12);
  });

  it("builds the filled-circle tip at the default size", () => {
    const t = new ArrowCircleFilledTip();
    expectClose(t.width, 0.35);
    expectClose(t.height, 0.35);
    expect(t.get_num_points()).toBe(32);
  });

  it("builds the square tip at the default size", () => {
    const t = new ArrowSquareTip();
    expectClose(t.width, 0.35);
    expectClose(t.height, 0.35);
    expect(t.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.CurvedDoubleArrow
// Python: CurvedDoubleArrow([-2,0,0],[2,0,0]): 2 submobjects (tips),
//   angle pi/2, start [-2,0,0], end [2,~0,0].
// ---------------------------------------------------------------------------
describe("CurvedDoubleArrow", () => {
  it("carries a tip at each end of the arc", () => {
    const ca = new CurvedDoubleArrow([-2, 0, 0], [2, 0, 0]);
    expect(ca.submobjects.length).toBe(2);
    expectClose(ca.angle, Math.PI / 2);
    expectRowsClose([ca.get_start()], [[-2, 0, 0]]);
    expectRowsClose([ca.get_end()], [[2, 4.440892098500626e-16, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.AnnotationDot
// Python: AnnotationDot() -> height == width == 0.208.
// ---------------------------------------------------------------------------
describe("AnnotationDot", () => {
  it("uses the scaled-up dot radius", () => {
    const d = new AnnotationDot();
    expectClose(d.height, 0.20800000000000002);
    expectClose(d.width, 0.20800000000000002);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Annulus.init_points
// Python: Annulus(inner_radius=1, outer_radius=2) -> 64 points, width 4.0,
//   inner_radius 1, outer_radius 2, center origin.
// ---------------------------------------------------------------------------
describe("Annulus.init_points", () => {
  it("builds the double-circle band", () => {
    const an = new Annulus(1, 2);
    expect(an.get_num_points()).toBe(64);
    expectClose(an.width, 4.0);
    expect(an.inner_radius).toBe(1);
    expect(an.outer_radius).toBe(2);
    expectRowsClose([centerOf(an)], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.BraceLabel.shift_brace
// Python (latex-free path via __new__ + manual label, since even a VMobject
// label goes through MathTex in Python and needs `latex`):
//   brace width 2.0, brace center [0,-1.3364925,0],
//   label center [0,-2.022985,0];
//   shift_brace(scaled square at [3,0,0]) returns self, brace center
//   [3,-2.3364925,0], width 4.0, label follows to [3,-3.022985,0];
//   shift_brace([sq, sq]) (list form) works.
// DEVIATION (D1): TS BraceLabel accepts a VMobject label directly (mirroring
// NumberLine._create_label_tex leniency); Python always calls the label
// constructor. String labels stay tex stubs in TS (asserted below).
// ---------------------------------------------------------------------------
describe("BraceLabel.shift_brace", () => {
  function labeled(): BraceLabel {
    return new BraceLabel(new Square(), new Square().scale(0.3) as unknown as VMobject);
  }

  it("rebuilds the brace on a new target and drags the label along", () => {
    const bl = labeled();
    expectClose(bl.brace.width, 2.0);
    expectRowsClose([centerOf(bl.brace)], [[0, -1.3364925, 0]]);
    const ret = bl.shift_brace(new Square().scale(2).shift([3, 0, 0]));
    expect(ret).toBe(bl);
    expectClose(bl.brace.width, 4.0);
    expectRowsClose([centerOf(bl.brace)], [[3, -2.3364925, 0]]);
    expectRowsClose([centerOf(bl.label)], [[3, -3.022985, 0]]);
  });

  it("accepts a list of mobjects as the new target", () => {
    const bl = labeled();
    bl.shift_brace([new Square(), new Square()]);
    expectClose(bl.brace.width, 2.0);
  });

  it("marks string labels as tex-backed stubs", () => {
    expect(() => new BraceLabel(new Square(), "x")).toThrowError(/text\/tex not ported/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.BraceText
// Python default label_constructor is Text (needs the text pipeline).
// DEVIATION (D2): TS throws TODO without a caller-supplied constructor and
// accepts one returning a VMobject.
// ---------------------------------------------------------------------------
describe("BraceText", () => {
  it("throws for the default Text label pipeline", () => {
    expect(() => new BraceText(new Square(), "hello")).toThrowError(/text\/tex not ported/);
  });

  it("builds with a caller-supplied VMobject constructor", () => {
    const label = new Square().scale(0.2);
    const bt = new BraceText(new Square(), "hello", () => label);
    expect(bt.label).toBe(label);
    expect(bt.submobjects.length).toBe(2);
    expect(bt.brace).toBeInstanceOf(Brace);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.SurroundingRectangle
// Python: SurroundingRectangle(Square(), buff=0.1) -> 2.2 x 2.2 at origin;
//   buff=(0.2,0.3) -> 2.4 x 2.6.
// ---------------------------------------------------------------------------
describe("SurroundingRectangle", () => {
  it("pads the target uniformly with a scalar buff", () => {
    const sr = new SurroundingRectangle(new Square(), { buff: 0.1 });
    expectClose(sr.width, 2.2);
    expectClose(sr.height, 2.2);
    expectRowsClose([centerOf(sr)], [[0, 0, 0]]);
  });

  it("pads axes independently with a tuple buff", () => {
    // Python: buff=(0.2,0.3) -> width 2.4, height 2.6.
    const sr = new SurroundingRectangle(new Square(), { buff: [0.2, 0.3] });
    expectClose(sr.width, 2.4);
    expectClose(sr.height, 2.6);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add
// Python: Scene().add(m) returns the scene, mobjects == [m];
//   add(a, b, a) keeps all three entries (len 3).
// ---------------------------------------------------------------------------
describe("Scene.add", () => {
  it("adds mobjects and returns the scene", () => {
    const s = new Scene();
    const m = new Square();
    const ret = s.add(m);
    expect(ret).toBe(s);
    expect(s.mobjects.length).toBe(1);
    expect(s.mobjects[0]).toBe(m);
  });

  it("keeps repeated entries like Python", () => {
    const s = new Scene();
    const a = new Square();
    const b = new Square();
    s.add(a, b, a);
    expect(s.mobjects.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.align_data
// Python: Mobject().align_data(Mobject()) returns self; Square vs
// Line([0,0,0],[1,0,0]) (16 vs 4 points) aligns both to 16 points;
// skip_point_alignment=True leaves points untouched but still returns self.
// ---------------------------------------------------------------------------
describe("Mobject.align_data", () => {
  it("returns self on empty mobjects", () => {
    const m1 = new Mobject();
    const m2 = new Mobject();
    expect(m1.align_data(m2)).toBe(m1);
  });

  it("aligns point counts across a Square and a Line", () => {
    const s = new Square();
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expect(s.get_num_points()).toBe(16);
    expect(l.get_num_points()).toBe(4);
    expect(s.align_data(l)).toBe(s);
    expect(s.get_num_points()).toBe(16);
    expect(l.get_num_points()).toBe(16);
  });

  it("skips point alignment on request", () => {
    const s = new Square();
    const scaled = new Square().scale(2);
    expect(s.align_data(scaled, true)).toBe(s);
    expect(s.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.DecimalTable
// Python default element_to_mobject is DecimalNumber (LaTeX-backed here).
// DEVIATION (D3): TS throws TODO for the default factory; the inherited
// Table layout is exercised with an identity factory over Squares:
//   2x2 -> row_dim 2, col_dim 2, center origin,
//   entries (-1.65,1.4),(1.65,1.4),(-1.65,-1.4),(1.65,-1.4).
// ---------------------------------------------------------------------------
describe("DecimalTable", () => {
  const idFactory = (m: unknown) => m as VMobject;

  it("marks the DecimalNumber default as a tex stub", () => {
    expect(() => new DecimalTable([["1.5"]], {})).toThrowError(/text\/tex not ported/);
  });

  it("inherits the Table grid layout with a custom factory", () => {
    const t = new DecimalTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
    expect(t.row_dim).toBe(2);
    expect(t.col_dim).toBe(2);
    expectRowsClose([centerOf(t)], [[0, 0, 0]]);
    const entries = t.get_entries_without_labels() as VGroup;
    expect(entries.submobjects.length).toBe(4);
    const want = [
      [-1.65, 1.4, 0],
      [1.65, 1.4, 0],
      [-1.65, -1.4, 0],
      [1.65, -1.4, 0],
    ];
    for (let i = 0; i < 4; i += 1) {
      expectRowsClose([centerOf(entries.submobjects[i] as Mobject)], [want[i]!]);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.DecimalMatrix / MobjectMatrix
// Python element defaults (DecimalNumber / identity) and _add_brackets both
// need LaTeX here.
// DEVIATION (D4): TS Matrix._add_brackets throws TODO (tex not ported), so
// default construction of either subclass throws; layout is exercised via a
// bracket-free subclass plus the pure matrix_to_tex_string helper.
// Python (bracket-free _organize_mob_matrix over 2x2 Squares, v_buff=0.8,
// h_buff=1.3, corner DR): (-1,1),(0.3,1),(-1,0.2),(0.3,0.2) pre-centering;
// full Matrix.__init__ centers -> (-0.65,0.4),(0.65,0.4),(-0.65,-0.4),
// (0.65,-0.4).
// matrix_to_tex_string([[1,2],[3,4]]) ==
//   "\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]";
// matrix_to_tex_string([1,2,3]) uses a single {c} column.
// ---------------------------------------------------------------------------
describe("DecimalMatrix / MobjectMatrix", () => {
  class BracketFreeMatrix extends MobjectMatrix {
    override _add_brackets(): this {
      this.brackets = new VGroup();
      return this;
    }
  }

  it("marks the tex-backed paths as stubs", () => {
    expect(() => new DecimalMatrix([[1, 2], [3, 4]], {})).toThrowError(
      /text\/tex not ported/,
    );
    expect(() => new MobjectMatrix([[new Square()]], {})).toThrowError(
      /text\/tex not ported/,
    );
    expect(() => new Matrix([[new Square()]], {})).toThrowError(/text\/tex not ported/);
  });

  it("organizes the mob matrix before centering", () => {
    const m = new BracketFreeMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      {},
    );
    // NOTE: the constructor centers afterwards; undo it to compare with the
    // raw _organize_mob_matrix probe.
    const raw = [
      [-1.0, 1.0, 0.0],
      [0.30000000000000004, 1.0, 0.0],
      [-1.0, 0.19999999999999996, 0.0],
      [0.30000000000000004, 0.19999999999999996, 0.0],
    ];
    const flat = m.mob_matrix.flat();
    expect(flat.length).toBe(4);
    for (let i = 0; i < 4; i += 1) {
      const c = centerOf(flat[i] as Mobject);
      // centered actual + back-shifted mean == raw probe: the constructor
      // centers the whole matrix, whose entry-mean is (-0.35, 0.6, 0).
      expectRowsClose([[c[0]! - 0.35, c[1]! + 0.6, c[2]!]], [raw[i]!]);
    }
  });

  it("centers the organized matrix", () => {
    const m = new BracketFreeMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      {},
    );
    const flat = m.mob_matrix.flat();
    const want = [
      [-0.65, 0.4, 0],
      [0.65, 0.4, 0],
      [-0.65, -0.4, 0],
      [0.65, -0.4, 0],
    ];
    for (let i = 0; i < 4; i += 1) {
      expectRowsClose([centerOf(flat[i] as Mobject)], [want[i]!], 1e-6);
    }
    expectRowsClose([centerOf(m as unknown as Mobject)], [[0, 0, 0]]);
  });

  it("renders tex strings without LaTeX", () => {
    expect(matrix_to_tex_string([[1, 2], [3, 4]])).toBe(
      "\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]",
    );
    expect(matrix_to_tex_string([1, 2, 3])).toBe(
      "\\left[ \\begin{array}{c}1 \\\\ 2 \\\\ 3\\end{array} \\right]",
    );
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols against earlier-ported machinery
// ---------------------------------------------------------------------------
describe("integration: layers-18 batch with earlier ports", () => {
  it("bar height equals the axis y-mapping distance", () => {
    // Python: |c2p(0,v)[1] - c2p(0,0)[1]| for the BarChart-equivalent axes.
    const ax = new Axes([0, 3, 1], [0, 3, 0.75], 3, 4);
    const yOf = (v: number): number => (ax.c2p(0, v).toArray() as number[])[1] ?? NaN;
    const base = yOf(0);
    const bc = new BarChart([1, 2, 3]);
    for (const v of [1, 2, 3]) {
      const bar = bc._create_bar(0, v);
      expectClose(bar.height, Math.abs(yOf(v) - base));
    }
  });

  it("aligns a SurroundingRectangle chain and adds it to a Scene", () => {
    // Python: SurroundingRectangle(Square()) centers origin; Scene.add keeps
    // identity; align_data returns self with matched point counts.
    const sq = new Square();
    const sr = new SurroundingRectangle(sq, { buff: 0.1 });
    const s = new Scene();
    expect(s.add(sr)).toBe(s);
    expect(s.mobjects[0]).toBe(sr as unknown as Mobject);
    const other = new SurroundingRectangle(new Square().scale(2), { buff: 0.1 });
    expect(sr.align_data(other as unknown as Mobject)).toBe(sr);
    expect(sr.get_num_points()).toBe((other as unknown as Mobject).get_num_points());
  });

  it("resolves Axes <-> NumberLine mappings consistently", () => {
    // Python: Axes([-3,3,1],[-2,2,1],6,4).c2p(2,0)[0] == 2 and
    // NumberLine([-3,3,1], length=6).n2p(2)[0] == 2.
    const ax = makeAxes();
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const fromAxes = (ax.c2p(2, 0).toArray() as number[])[0] ?? NaN;
    const fromLine = (nl.n2p(2).toArray() as number[])[0] ?? NaN;
    expectClose(fromAxes, 2.0);
    expectClose(fromLine, 2.0);
    expectClose(fromAxes, fromLine);
  });

  it("animates a shifted line via the animate-builder path", () => {
    // Python: l.animate.shift([1,0,0]); begin(); finish() moves start/end.
    const l = new Line([0, 0, 0], [1, 0, 0]);
    const builder = l.animate as unknown as {
      shift(v: number[]): unknown;
      build(): { begin(): void; finish(): void };
    };
    builder.shift([1, 0, 0]);
    const anim = builder.build();
    anim.begin();
    anim.finish();
    expectRowsClose([l.get_start()], [[1, 0, 0]]);
    expectRowsClose([l.get_end()], [[2, 0, 0]]);
  });
});
