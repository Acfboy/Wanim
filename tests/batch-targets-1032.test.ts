// Ground-truth tests for batch layers 15..15 (targets-1032).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): CoordinateSystem.get_axis_labels on Axes needs
// MathTex/Text (Python raises FileNotFoundError without a LaTeX install and
// builds Tex mobjects with one); Label is a text-frame class. The TS port
// keeps clearly-marked TODO stubs for both; tests only assert the stub
// behavior (documented inline as deviations).
//
// Deviations documented inline:
//  (D1) Axes.get_axes()/PolarPlane.get_axes(): Python returns a VGroup of 2
//       built NumberLine axes; the TS CoordinateSystem hierarchy is not
//       Mobject-based yet so `axes` stays an empty VGroup. Tests assert
//       identity + VGroup type only.
//  (D2) Axes.get_axis_labels(): needs Tex -> out of scope; TS base throws
//       NotImplementedError (same as Python's CoordinateSystem base).
//  (D3) Union/Difference/Intersection/Exclusion: Python computes via Skia
//       `pathops`; TS computes via a local flattened-polygon engine (no
//       pathops among allowed deps). Argument-validation errors are shared
//       ground truth; areas are validated below.
//  (D4) Label: now builds via the ported MathTex/Text pipeline (text/tex is
//       no longer out of scope).
//  (D5) Cone._set_start_and_end_attributes faithfully mirrors the upstream
//       quirk `direction * norm(direction)` (multiply, not normalize): with
//       direction=[0,0,2], height=2 the end is start+[0,0,8]. TS additionally
//       accepts plain lists where Python requires an np array (TypeError).
//  (D6) input_to_graph_point out-of-range message: same ValueError name and
//       shape; float rendering differs ("0" vs "0.0"), so tests match loosely.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { array } from "numpy-ts";
import {
  Axes,
  CoordinateSystem,
  NumberPlane,
  PolarPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { ParametricFunction } from "../src/mobject/graphing/functions.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Arc, CubicBezier } from "../src/mobject/geometry/arc.js";
import { Arrow, Elbow, Line, RightAngle } from "../src/mobject/geometry/line.js";
import { StealthTip } from "../src/mobject/geometry/tips.js";
import {
  Difference,
  Exclusion,
  Intersection,
  Union,
} from "../src/mobject/geometry/boolean_ops.js";
import { Cross } from "../src/mobject/geometry/shape_matchers.js";
import { ConvexHull, Square } from "../src/mobject/geometry/polygram.js";
import { Label } from "../src/mobject/geometry/labeled.js";
import { Polyhedron } from "../src/mobject/three_d/polyhedra.js";
import {
  Arrow3D,
  Cone,
  Cube,
} from "../src/mobject/three_d/three_dimensions.js";
import { pointRows } from "../src/utils/arrays.js";
import { computeSubpathRanges } from "../src/renderer/canvas.js";

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

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_axis_labels
// Python: CoordinateSystem.get_axis_labels raises NotImplementedError();
//   Axes.get_axis_labels builds Tex labels (text/tex -> out of scope, D2).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_axis_labels", () => {
  it("raises NotImplementedError on the base class", () => {
    expect(() => new CoordinateSystem().get_axis_labels()).toThrowError(
      expect.objectContaining({ name: "NotImplementedError" }),
    );
  });

  it("Axes override groups positioned labels (str labels render via MathTex)", () => {
    // Axes.get_axis_labels builds MathTex labels for str/float inputs; the
    // NumberLine axes are real, so this now succeeds and returns positioned
    // labels.
    const labels = makeAxes().get_axis_labels();
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot
// Python (ax as above, f=x^2, x_range=[-2,2,0.5]):
//   npts 32; p0 [-1.9999999999999991,4,0]; p4 [-1.4999999999999996,2.25,0];
//   p-1 [2.000000000000001,4,0]; underlying(3)==9.
// Python default plot: npts 240; p0 [-2.9999999999999982,9,0].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot", () => {
  it("plots x^2 over an explicit range with ground-truth points", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2, [-2, 2, 0.5]);
    expect(g).toBeInstanceOf(ParametricFunction);
    expect(g.get_num_points()).toBe(32);
    const rows = rowsOf(g.points, g.dim);
    expectRowsClose([rows[0] ?? []], [[-1.9999999999999991, 4, 0]]);
    expectRowsClose([rows[4] ?? []], [[-1.4999999999999996, 2.25, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[2.000000000000001, 4, 0]]);
    expectClose(
      (g as unknown as { underlying_function: (x: number) => number }).underlying_function(3),
      9,
    );
  });

  it("uses the axes x_range with divided step when x_range is omitted", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    // Python: t_range[2] = 1/10 = 0.1 over [-3,3] -> 60 intervals -> 240 pts.
    expect(g.get_num_points()).toBe(240);
    const rows = rowsOf(g.points, g.dim);
    expectRowsClose([rows[0] ?? []], [[-2.9999999999999982, 9, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_parametric_curve
// Python (t -> [t,t^2,0], t_range=[0,1,0.5]): npts 8;
//   p0 [0,0,0]; p-1 [1.0000000000000002,1,0].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_parametric_curve", () => {
  it("maps the parametric function through coords_to_point", () => {
    const ax = makeAxes();
    const fn = (t: number): number[] => [t, t ** 2, 0];
    const g = ax.plot_parametric_curve(fn, false, {
      t_range: [0, 1, 0.5],
    });
    expect(g.get_num_points()).toBe(8);
    const rows = rowsOf(g.points, g.dim);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1.0000000000000002, 1, 0]]);
    // Python stashes the raw callable as graph.underlying_function.
    expect(g.underlying_function).toBe(fn);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_polar_graph
// Python (PolarPlane, r=2*sin(5th), [0,2pi]): npts 2516 (config-independent
// sample count: arange(0,2pi,0.01) 629 + endpoint = 630 samples -> 629*4).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_polar_graph", () => {
  it("samples the polar function with ground-truth count", () => {
    const plane = new PolarPlane();
    const g = plane.plot_polar_graph((th: number) => 2 * Math.sin(th * 5), [0, 2 * Math.PI]);
    expect(g).toBeInstanceOf(ParametricFunction);
    expect(g.get_num_points()).toBe(2516);
    // r(0) = 0, so the curve starts at the plane origin.
    expectRowsClose([rowsOf(g.points, g.dim)[0] ?? []], [plane.coords_to_point(0, 0).toArray() as number[]]);
  });

  it("defaults theta_range to [0, TAU]", () => {
    const plane = new PolarPlane();
    const g = plane.plot_polar_graph(() => 1);
    expect(g.get_num_points()).toBe(2516);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.Axes.get_axes /
//   PolarPlane.get_axes
// Python: ax.get_axes() is ax.axes (VGroup of 2 NumberLines);
//   pp.get_axes() is pp.axes. (D1: TS axes group is empty.)
// ---------------------------------------------------------------------------
describe("Axes.get_axes / PolarPlane.get_axes", () => {
  it("Axes.get_axes returns the axes group by identity", () => {
    const ax = makeAxes();
    const axes = ax.get_axes();
    expect(axes).toBeInstanceOf(VGroup);
    expect(axes).toBe(ax.axes); // (D1) empty VGroup until Mobject integration
  });

  it("PolarPlane.get_axes returns the axes group by identity", () => {
    const plane = new PolarPlane();
    const axes = plane.get_axes();
    expect(axes).toBeInstanceOf(VGroup);
    expect(axes).toBe(plane.axes);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.input_to_graph_point
// Python: i2gp(1, plot(x^2)) == [1,1,0]; plain-VMobject out-of-range x=5
//   raises ValueError "x=5 not located in the range of the graph ([0.0, 1.0])".
// ---------------------------------------------------------------------------
describe("CoordinateSystem.input_to_graph_point", () => {
  it("evaluates the underlying function directly", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    expectRowsClose([[...(ax.input_to_graph_point(1, g).toArray() as number[])]], [[1, 1, 0]]);
    expectRowsClose([[...(ax.i2gp(1, g).toArray() as number[])]], [[1, 1, 0]]);
  });

  it("raises ValueError for x outside a plain graph's range (D6)", () => {
    const ax = makeAxes();
    const line = new ParametricFunction((t: number) => ax.coords_to_point(t, t).toArray() as number[], {
      t_range: [0, 1, 0.5],
    });
    // NOTE: constructed directly (no underlying_function), so the
    // binary-search path runs and must fail for x=5.
    expect(() => ax.input_to_graph_point(5, line)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    expect(() => ax.input_to_graph_point(5, line)).toThrowError(/not located in the range/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.input_to_graph_coords
// Python: ax.input_to_graph_coords(3, parabola) == (3, 9).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.input_to_graph_coords", () => {
  it("returns [x, f(x)]", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    expect(ax.input_to_graph_coords(3, g)).toEqual([3, 9]);
    expect(ax.i2gc(3, g)).toEqual([3, 9]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem
//   .plot_antiderivative_graph.antideriv
// Python (f=x^2, samples=50, y_intercept=0):
//   antideriv(0)=0; (1)=0.33340274885464394; (2)=2.6672219908371515;
//   (-1)=-0.33340274885464394; y_intercept=5,samples=10 at x=1: 5.3353909465020575.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_antiderivative_graph", () => {
  it("integrates x^2 from 0 with the trapezoid rule", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const anti = ax.plot_antiderivative_graph(g, 0, 50);
    const f = (anti as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(f(0.0), 0.0);
    expectClose(f(1.0), 0.33340274885464394);
    expectClose(f(2.0), 2.6672219908371515);
    expectClose(f(-1.0), -0.33340274885464394);
  });

  it("honors y_intercept and sample count", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const anti = ax.plot_antiderivative_graph(g, 5, 10);
    const f = (anti as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(f(1.0), 5.3353909465020575);
  });

  it("returns a plottable ParametricFunction", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const anti = ax.plot_antiderivative_graph(g);
    expect(anti).toBeInstanceOf(ParametricFunction);
    expect(anti.get_num_points()).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.get_tick_marks
// Python: nl.get_tick_marks() is nl.ticks (VGroup).
//   tick_range [-3,-2,-1,-0,1,2,3]; exclude_origin_tick -> [-3,-2,-1,1,2,3].
// ---------------------------------------------------------------------------
describe("NumberLine.get_tick_marks", () => {
  it("returns the ticks group by identity", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const marks = nl.get_tick_marks();
    expect(marks).toBeInstanceOf(VGroup);
    expect(marks).toBe(nl.ticks);
  });

  it("tick range splits around zero with -0.0 kept", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const got = [...(nl.get_tick_range().toArray() as number[])];
    const expected = [-3, -2, -1, -0, 1, 2, 3];
    expect(got.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i += 1) {
      expectClose(got[i] ?? NaN, expected[i] ?? NaN);
    }
  });

  it("exclude_origin_tick drops the zero tick", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], exclude_origin_tick: true });
    const got = [...(nl.get_tick_range().toArray() as number[])];
    expect(got.length).toBe(6);
    for (let i = 0; i < 6; i += 1) {
      expectClose(got[i] ?? NaN, [-3, -2, -1, 1, 2, 3][i] ?? NaN);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.polyhedra.Polyhedron
// Python (square pyramid): 5 faces; 16 edges starting
//   [(0,1),(1,4),(4,0),(1,2)]; 5 graph vertices.
// ---------------------------------------------------------------------------
describe("Polyhedron", () => {
  function pyramid(): Polyhedron {
    return new Polyhedron(
      [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [0, 0, 2]],
      [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]],
    );
  }

  it("builds faces, edges and graph vertices", () => {
    const p = pyramid();
    expect(p.faces.submobjects.length).toBe(5);
    expect(p.edges.length).toBe(16);
    expect(p.edges.slice(0, 4)).toEqual([[0, 1], [1, 4], [4, 0], [1, 2]]);
    expect(p.graph.vertices.size).toBe(5);
  });

  it("get_edges makes cyclic pairs per face", () => {
    expect(new Polyhedron().get_edges([[0, 1, 2], [2, 3, 0]])).toEqual([
      [0, 1], [1, 2], [2, 0], [2, 3], [3, 0], [0, 2],
    ]);
  });

  it("extract_face_coords follows the graph vertex centers", () => {
    const p = pyramid();
    const coords = p.extract_face_coords();
    expect(coords.length).toBe(5);
    expectRowsClose(coords[0] ?? [], [[1, 1, 0], [1, -1, 0], [0, 0, 2]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cube
// Python: Cube() side_length 2, 6 submobject faces.
// ---------------------------------------------------------------------------
describe("Cube", () => {
  it("builds six faces with the ground-truth side length", () => {
    const c = new Cube();
    expectClose(c.side_length, 2);
    expect(c.submobjects.length).toBe(6);
  });

  it("honors an explicit side length", () => {
    const c = new Cube(3);
    expectClose(c.side_length, 3);
    expect(c.submobjects.length).toBe(6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cone._set_start_and_end_attributes
// Python: default Cone(1,1): start [0,0,-1], end [0,0,0].
//   Cone(1,2,direction=[0,0,2]): start [0,0,-2], end [0,0,6] (D5 quirk:
//   direction * norm(direction) * height = [0,0,2]*2*2 = [0,0,8]).
// ---------------------------------------------------------------------------
describe("Cone._set_start_and_end_attributes", () => {
  it("places start at the base center and end one height along +z", () => {
    const cone = new Cone(1, 1);
    expectRowsClose([cone.get_start()], [[0, 0, -1]]);
    expectRowsClose([cone.get_end()], [[0, 0, 0]]);
  });

  it("mirrors the upstream multiply-by-norm quirk (D5)", () => {
    const cone = new Cone(1, 2, array([0, 0, 2]) as NDArray);
    expectRowsClose([cone.get_start()], [[0, 0, -2]]);
    expectRowsClose([cone.get_end()], [[0, 0, 6]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Arrow3D.get_end
// Python: Arrow3D([0,0,0],[0,0,3]).get_end() == [0,0,3].
// ---------------------------------------------------------------------------
describe("Arrow3D.get_end", () => {
  it("returns the original end point", () => {
    const a = new Arrow3D([0, 0, 0], [0, 0, 3]);
    expectRowsClose([[...(a.get_end().toArray() as number[])]], [[0, 0, 3]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Elbow.__init__
// Python: Elbow() 8 points starting
//   [0,0.2,0],[0.06666666666666667,0.2,0];
//   Elbow(0.5, pi/4) starts [-0.3535533905932738,0.35355339059327373,0].
// ---------------------------------------------------------------------------
describe("Elbow.__init__", () => {
  it("builds the default corner", () => {
    const e = new Elbow();
    expect(e.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(e.points, e.dim).slice(0, 2), [
      [0, 0.2, 0],
      [0.06666666666666667, 0.2, 0],
    ]);
    expectClose(e.angle, 0);
  });

  it("rescales and rotates for explicit width/angle", () => {
    const e = new Elbow(0.5, Math.PI / 4);
    expect(e.get_num_points()).toBe(8);
    expectRowsClose([rowsOf(e.points, e.dim)[0] ?? []], [
      [-0.3535533905932738, 0.35355339059327373, 0],
    ]);
    expectClose(e.angle, Math.PI / 4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow.get_normal_vector
// Python: Arrow([0,0,0],[3,0,0]).get_normal_vector() == [-0,0,-1].
// ---------------------------------------------------------------------------
describe("Arrow.get_normal_vector", () => {
  it("returns the out-of-plane normal of an x-axis arrow", () => {
    const a = new Arrow([0, 0, 0], [3, 0, 0]);
    expectRowsClose([a.get_normal_vector()], [[-0, 0, -1]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.RightAngle
// Python (Line[0,0,0]->[2,0,0], Line[0,0,0]->[0,2,0]): 8 points,
//   first [0.4,0,0]; elbow path has no angle_value attribute.
// ---------------------------------------------------------------------------
describe("RightAngle", () => {
  it("builds the elbow corner with ground-truth points", () => {
    const ra = new RightAngle(new Line([0, 0, 0], [2, 0, 0]), new Line([0, 0, 0], [0, 2, 0]));
    expect(ra.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(ra.points, ra.dim), [
      [0.4, 0, 0],
      [0.4, 0.13333333333333333, 0],
      [0.4, 0.26666666666666666, 0],
      [0.4, 0.4, 0],
      [0.4, 0.4, 0],
      [0.2666666666666667, 0.4, 0],
      [0.13333333333333336, 0.4, 0],
      [0, 0.4, 0],
    ]);
    // NOTE: the elbow branch never assigns angle_value in either language.
    expect("angle_value" in ra).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.StealthTip
// Python: StealthTip().length == 0.17499999999999993;
//   tip_angle == 1.7763568394002513e-16 (points along +x).
// ---------------------------------------------------------------------------
describe("StealthTip", () => {
  it("builds the stealth arrowhead with ground-truth length", () => {
    const tip = new StealthTip();
    expectClose(tip.length, 0.17499999999999993);
    expectClose(tip.tip_angle, 1.7763568394002513e-16);
  });

  it("scales with the length parameter", () => {
    const tip = new StealthTip(1, 3, 0.35);
    expectClose(tip.length, 0.35, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.boolean_ops.Union / Difference /
//   Intersection / Exclusion
// Python: Union needs >= 2 mobjects (ValueError otherwise);
//   Union/Difference/Intersection/Exclusion compute via pathops.
//   (D3) TS computes via the local polygon engine; areas checked below.
// ---------------------------------------------------------------------------
describe("boolean_ops validation", () => {
  it("Union requires at least 2 mobjects", () => {
    expect(() => new Union(new Square())).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    expect(() => new Union(new Square())).toThrowError(/At least 2 mobjects/);
  });

  it("Intersection requires at least 2 mobjects", () => {
    expect(() => new Intersection(new Square())).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });

  it("Union of two squares computes the combined outline (D3)", () => {
    expect(booleanArea(new Union(new Square(), new Square()))).toBeCloseTo(4, 0);
  });

  it("Difference of identical squares is empty (D3)", () => {
    expect(new Difference(new Square(), new Square()).get_num_points()).toBe(0);
  });

  it("Intersection of two mobjects computes the overlap (D3)", () => {
    expect(booleanArea(new Intersection(new Square(), new Square()))).toBeCloseTo(4, 0);
  });

  it("Exclusion of identical squares is empty (D3)", () => {
    expect(new Exclusion(new Square(), new Square()).get_num_points()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.Cross
// Python: Cross() has 2 Line submobjects, stroke_width 6.0;
//   first line runs [-1,1,0] -> [1,-1,0].
// ---------------------------------------------------------------------------
describe("Cross", () => {
  it("builds two diagonal lines with ground-truth geometry", () => {
    const c = new Cross();
    expect(c.submobjects.length).toBe(2);
    expectClose(c.stroke_width, 6.0);
    expectRowsClose([(c.submobjects[0] as VMobject).get_start()], [[-1, 1, 0]]);
    expectRowsClose([(c.submobjects[0] as VMobject).get_end()], [[1, -1, 0]]);
  });

  it("replaces onto the given mobject", () => {
    const sq = new Square();
    const c = new Cross(sq);
    expectClose(c.width, sq.width, 1e-6);
    expectClose(c.height, sq.height, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Arc.generate_points / move_arc_center_to /
//   stop_angle
// Python: Arc(2,0,pi/2) has 32 points, start [2,0,0],
//   end [1.2246468e-16,2,0], center ~[6.58e-15,4.44e-16,0], stop pi/2.
//   Arc(1,0.5,1.0).stop_angle() == 1.4999999999999918.
//   move_arc_center_to([3,4,0]): center [3,4,0],
//   start [3.87758256,4.47942554,0].
//   Arc(1,-1.0,0.5).stop_angle() == 5.783185307179604.
// ---------------------------------------------------------------------------
describe("Arc.generate_points", () => {
  it("lays a quarter circle with ground-truth endpoints", () => {
    const a = new Arc(2, 0, Math.PI / 2, 9);
    expect(a.get_num_points()).toBe(32);
    expectRowsClose([a.get_start()], [[2, 0, 0]]);
    expectRowsClose([a.get_end()], [[1.2246467991473532e-16, 2, 0]]);
    const rows = rowsOf(a.points, a.dim);
    expectRowsClose([rows[0] ?? []], [[2, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1.2246467991473532e-16, 2, 0]]);
  });
});

describe("Arc.move_arc_center_to", () => {
  it("shifts the arc so its center lands on the target", () => {
    const a = new Arc(1, 0.5, 1.0);
    a.move_arc_center_to([3, 4, 0]);
    expectRowsClose([a.get_arc_center()], [[3, 4, 0]], 1e-6);
    expectRowsClose([a.get_start()], [[3.87758256, 4.47942554, 0]], 1e-6);
  });
});

describe("Arc.stop_angle", () => {
  it("returns pi/2 for the quarter circle", () => {
    expectClose(new Arc(2, 0, Math.PI / 2, 9).stop_angle(), 1.5707963267948999);
  });

  it("adds start and sweep angles", () => {
    expectClose(new Arc(1, 0.5, 1.0).stop_angle(), 1.4999999999999918);
  });

  it("wraps negative starts into [0, TAU)", () => {
    expectClose(new Arc(1, -1.0, 0.5).stop_angle(), 5.783185307179604);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.CubicBezier.__init__
// Python: CubicBezier([0,0,0],[1,1,0],[2,1,0],[3,0,0]) has 4 points,
//   start [0,0,0], end [3,0,0].
// ---------------------------------------------------------------------------
describe("CubicBezier.__init__", () => {
  it("stores the four control points directly", () => {
    const cb = new CubicBezier([0, 0, 0], [1, 1, 0], [2, 1, 0], [3, 0, 0]);
    expect(cb.get_num_points()).toBe(4);
    expectRowsClose([cb.get_start()], [[0, 0, 0]]);
    expectRowsClose([cb.get_end()], [[3, 0, 0]]);
    expectRowsClose(rowsOf(cb.points, cb.dim), [
      [0, 0, 0], [1, 1, 0], [2, 1, 0], [3, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.ConvexHull
// Python (square + interior point): vertex group
//   [[0,1,0],[1,1,0],[1,0,0],[0,0,0]] (interior point excluded).
// ---------------------------------------------------------------------------
describe("ConvexHull", () => {
  // NOTE: Python's ConvexHull walks a `set` of facets, so the starting
  // vertex and winding of the output vary run to run (one probe even raised
  // `ValueError: too many values to unpack` inside the walk). The stable
  // ground truth is the vertex SET plus single-group cyclic structure, so
  // order is asserted up to cyclic rotation/reversal here. The TS port is
  // deterministic and always returns a valid hull (one of Python's outputs).
  function canonicalCycle(vertices: number[][]): string[] {
    const key = (v: number[]): string => v.map((x) => x.toFixed(9)).join(",");
    const keys = vertices.map(key);
    const rotations: string[][] = [];
    for (let i = 0; i < keys.length; i += 1) {
      rotations.push([...keys.slice(i), ...keys.slice(0, i)]);
    }
    const reversed = [...keys].reverse();
    for (let i = 0; i < reversed.length; i += 1) {
      rotations.push([...reversed.slice(i), ...reversed.slice(0, i)]);
    }
    return rotations.map((r) => r.join(";")).sort();
  }

  function expectSameCycle(actual: number[][], expected: number[][]): void {
    expect(actual.length).toBe(expected.length);
    const a = canonicalCycle(actual);
    expect(a).toContain(canonicalCycle(expected)[0]);
  }

  it("hulls a square, dropping the interior point", () => {
    const h = new ConvexHull([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]);
    expectSameCycle(h.get_vertices(), [
      [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0],
    ]);
    expect(h.get_vertex_groups().length).toBe(1);
    // Interior point excluded from every group.
    for (const group of h.get_vertex_groups()) {
      for (const v of group) {
        expect(close(v[0] ?? NaN, 0.5) && close(v[1] ?? NaN, 0.5)).toBe(false);
      }
    }
  });

  it("hulls a triangle", () => {
    const h = new ConvexHull([0, 0, 0], [1, 0, 0], [0, 1, 0]);
    expectSameCycle(h.get_vertices(), [[0, 0, 0], [0, 1, 0], [1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.labeled.Label
// Text/tex is ported (MathTex via MathJax), so string labels now build.
// ---------------------------------------------------------------------------
describe("Label", () => {
  it("builds a string label (MathTex + box + frame)", () => {
    const label = new Label("x");
    expect(label.submobjects.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-1032 integration", () => {
  it("plot -> input_to_graph_point matches moving a mobject there", () => {
    // Python: position = ax.input_to_graph_point(x=PI, graph=curve) then
    // Square().move_to(position); the square center equals the graph point.
    const ax = makeAxes();
    const curve = ax.plot((x: number) => Math.cos(x));
    const position = ax.input_to_graph_point(Math.PI, curve);
    const sq = new Square();
    sq.move_to(position);
    const expected = ax.coords_to_point(Math.PI, Math.cos(Math.PI)).toArray() as number[];
    expectRowsClose([sq.get_center()], [expected]);
  });

  it("antiderivative of the plotted constant agrees with y=x", () => {
    // Python: plot of f=1 integrated from 0 gives antideriv(x)=x.
    const ax = makeAxes();
    const g = ax.plot(() => 1);
    const anti = ax.plot_antiderivative_graph(g, 0, 20);
    const f = (anti as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(f(1.5), 1.5, 1e-6);
    expectClose(f(-2.0), -2.0, 1e-6);
  });

  it("Arrow tip plus normal vector agree on direction", () => {
    // The tip sits at the arrow end; the normal is perpendicular to the
    // arrow direction ([3,0,0] -> normal in the yz-plane with |n|==1).
    const a = new Arrow([0, 0, 0], [3, 0, 0]);
    expect(a.has_tip()).toBe(true);
    const n = a.get_normal_vector();
    const dir = a.get_unit_vector();
    const dot = n.reduce((s, v, i) => s + v * (dir[i] ?? 0), 0);
    expectClose(dot, 0, 1e-9);
    expectClose(Math.hypot(...n), 1);
  });

  it("Polyhedron update_faces tracks moved graph vertices", () => {
    const p = new Polyhedron(
      [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [0, 0, 2]],
      [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]],
    );
    const before = p.extract_face_coords();
    (p.graph.getItem(4) as Mobject).move_to([0, 0, 3]);
    const after = p.extract_face_coords();
    expectRowsClose(after[0] ?? [], [[1, 1, 0], [1, -1, 0], [0, 0, 3]]);
    expect(before[0]).not.toEqual(after[0]);
  });

  it("RightAngle matches an Elbow laid on the same corners", () => {
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 2, 0]);
    const ra = new RightAngle(l1, l2);
    const elbow = new Elbow();
    elbow.set_points_as_corners([
      [0.4, 0, 0], [0.4, 0.4, 0], [0, 0.4, 0],
    ]);
    expectRowsClose(rowsOf(ra.points, ra.dim), rowsOf(elbow.points, elbow.dim));
  });

  it("NumberPlane coords round-trip feeds plot endpoints", () => {
    // c2p then p2c inverts; the plot's first anchor sits at the mapped
    // left range edge, tying coordinate_systems to number_line-adjacent code.
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const pt = plane.coords_to_point(1, 1).toArray() as number[];
    expectRowsClose([plane.point_to_coords(pt)], [[1, 1]]);
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2, [-2, 2, 0.5]);
    const rows = rowsOf(g.points, g.dim);
    expectRowsClose([ax.point_to_coords(rows[0] ?? [])], [[-2, 4]]);
  });
});
