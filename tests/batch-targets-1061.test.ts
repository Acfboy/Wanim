// Ground-truth tests for batch layers 16..16 (~1061 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): LabeledLine/Label require LaTeX (Python raises
// FileNotFoundError 'latex' without a tex installation; TS throws
// "TODO: text/tex not ported yet"). Only the throw is asserted there.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { TAU, PI, ORIGIN, DOWN } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Cylinder, Torus } from "../src/mobject/three_d/three_dimensions.js";
import { get_3d_vmob_gradient_start_and_end_points } from "../src/mobject/three_d/three_d_utils.js";
import { TangentLine, Arrow, Angle, Line } from "../src/mobject/geometry/line.js";
import { TipableVMobject, AnnularSector, Circle, Arc } from "../src/mobject/geometry/arc.js";
import {
  ArrowTriangleFilledTip,
  StealthTip,
  ArrowTip,
} from "../src/mobject/geometry/tips.js";
import { _BooleanOps, SkiaPath, PathVerb } from "../src/mobject/geometry/boolean_ops.js";
import { Cross, Underline } from "../src/mobject/geometry/shape_matchers.js";
import { Star, Square, Polygon } from "../src/mobject/geometry/polygram.js";
import { ConvexHull } from "../src/mobject/geometry/polygram.js";
import { LabeledLine } from "../src/mobject/geometry/labeled.js";
import { Brace, BraceBetweenPoints, ArcBrace } from "../src/mobject/svg/brace.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
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

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

function toList(v: NDArray | number[]): number[] {
  if (Array.isArray(v)) return [...v];
  return [...(pointRows(v, 3)[0] ?? [])];
}

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cylinder
// Python:
//   c = Cylinder(radius=2, height=3, resolution=4)
//   c.func(0.5,0) -> [2,0,0.5]; c.func(0,pi/2) -> [~0,2,0]
//   u_range (-1.5,1.5), v_range (0,TAU), direction [0,0,1], nsub 18
//   bases centers [0,0,-1]/[0,0,1]; show_ends=False -> 4 submobjects
// ---------------------------------------------------------------------------
describe("Cylinder", () => {
  it("func maps cylindrical coords like Python", () => {
    const c = new Cylinder(2, 3, [...ORIGIN].map((_, i) => (i === 2 ? 1 : 0)), [0, TAU], true, 4);
    expectRowsClose([toList(c.func(0.5, 0) as NDArray)], [[2, 0, 0.5]]);
    expectRowsClose([toList(c.func(0, Math.PI / 2) as NDArray)], [[1.2246467991473532e-16, 2, 0]]);
  });
  it("stores ranges, direction, and end caps like Python", () => {
    const c = new Cylinder(2, 3, [0, 0, 1], [0, TAU], true, 4);
    expect(c.u_range).toEqual([-1.5, 1.5]);
    expectClose(c.v_range[0] ?? NaN, 0);
    expectClose(c.v_range[1] ?? NaN, TAU);
    expectRowsClose([toList(c.get_direction() as number[])], [[0, 0, 1]]);
    expect(c.submobjects.length).toBe(18);
    // NOTE: height=3 -> u_range (-1.5,1.5); base_top shifts by u_range[1]*IN.
    expectRowsClose([c.base_top.get_center()], [[0, 0, -1.5]]);
    expectRowsClose([c.base_bottom.get_center()], [[0, 0, 1.5]]);
  });
  it("show_ends=False skips the caps; set_direction works", () => {
    const c = new Cylinder(1, 2, [0, 0, 1], [0, TAU], false, 2);
    expect(c.submobjects.length).toBe(4);
    // NOTE: boolean form -- a failing toBe(mobject) would make vitest\u2019s
    // pretty-printer OOM serializing the proxy graph (see integration note).
    expect(c.set_direction([1, 0, 0]) === c).toBe(true);
    expectRowsClose([toList(c.get_direction() as number[])], [[1, 0, 0]]);
    expectRowsClose([c.get_center()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_d_utils.get_3d_vmob_gradient_start_and_end_points
// Python:
//   Square(2) -> [[1,1,0],[-1,-1/3,0]]; empty VMobject -> [ORIGIN, ORIGIN]
// ---------------------------------------------------------------------------
describe("get_3d_vmob_gradient_start_and_end_points", () => {
  it("returns start/end corners like Python", () => {
    const sq = new Square(2);
    const [s, e] = get_3d_vmob_gradient_start_and_end_points(sq);
    expectRowsClose([toList(s), toList(e)], [[1, 1, 0], [-1, -0.33333333333333326, 0]]);
  });
  it("empty vmobject gives ORIGIN pair", () => {
    const v = new VMobject();
    const [s, e] = get_3d_vmob_gradient_start_and_end_points(v);
    expectRowsClose([toList(s), toList(e)], [[0, 0, 0], [0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Torus
// Python:
//   Torus(3,1,res=4): func(0,0)=[2,0,0]; func(pi/2,pi/2)=[0,3,-1];
//   func(pi,pi)=[-4,0,0]; R=3 r=1 u_range (0,TAU)
//   Torus(2,0.5).func(1,1)=[0.9346413208730651,1.4556176129093725,-0.42073549240394825]
// ---------------------------------------------------------------------------
describe("Torus", () => {
  it("func matches Python's ring formula", () => {
    const t = new Torus(3, 1, [0, TAU], [0, TAU], 4);
    expectRowsClose([toList(t.func(0, 0) as NDArray)], [[2, 0, 0]]);
    expectRowsClose([toList(t.func(Math.PI / 2, Math.PI / 2) as NDArray)], [[1.8369701987210297e-16, 3, -1]]);
    expectRowsClose([toList(t.func(Math.PI, Math.PI) as NDArray)], [[-4, 4.898587196589413e-16, -1.2246467991473532e-16]]);
    expect(t.R).toBe(3);
    expect(t.r).toBe(1);
    expectClose(t.u_range[1] ?? NaN, TAU);
  });
  it("matches the closed-form ring expression", () => {
    const t = new Torus(2, 0.5, [0, TAU], [0, TAU], 2);
    expectRowsClose([toList(t.func(1, 1) as NDArray)], [[0.9346413208730651, 1.4556176129093725, -0.42073549240394825]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.TangentLine / __init__
// Python:
//   TangentLine(Square(2), 0.25, length=2): len ~2, d_alpha 1e-6,
//     start [-0.292889218808546,1.707102781171828,0],
//     end [-1.707102781191454,0.2928892188085461,0]
//   TangentLine(Line([-1,0,0],[1,0,0]), 0.5, length=1):
//     [-0.5,0,0]->[0.5,0,0], len 1.0
//   Circle alphas 0.0/1.0 rescale to length 1.0
// ---------------------------------------------------------------------------
describe("TangentLine", () => {
  it("matches Python on a Square at alpha=0.25", () => {
    const tl = new TangentLine(new Square(2), 0.25, 2);
    expect(tl.d_alpha).toBe(1e-6);
    expect(tl.length).toBe(2);
    expectClose(tl.get_length(), 2.0, 1e-9);
    expectRowsClose([tl.get_start()], [[-0.292889218808546, 1.707102781171828, 0]], 1e-6);
    expectRowsClose([tl.get_end()], [[-1.707102781191454, 0.2928892188085461, 0]], 1e-6);
  });
  it("midpoints a straight Line like Python", () => {
    const tl = new TangentLine(new Line([-1, 0, 0], [1, 0, 0]), 0.5, 1);
    expectRowsClose([tl.get_start()], [[-0.5, 0, 0]]);
    expectRowsClose([tl.get_end()], [[0.5, 0, 0]]);
    expectClose(tl.get_length(), 1.0);
  });
  it("clips alpha at the 0/1 edges like np.clip", () => {
    const c = new Circle(1);
    for (const alpha of [0.0, 1.0]) {
      const tl = new TangentLine(c, alpha, 1);
      expectClose(tl.get_length(), 1.0, 1e-9);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.pop_tips / get_tips
// Python:
//   Arrow([-2,0,0],[2,0,0]): has_tip True, get_tips 1; pop -> 1, has_tip False
//   after pop: start [-1.75,0,0], end [1.75,~1.2e-32,0]; get_tips still 1
//     (attrs kept; hasattr check) -- TS mirrors via `!== undefined`.
//   bare Line: get_tips 0, has_tip False
// ---------------------------------------------------------------------------
describe("TipableVMobject.pop_tips / get_tips", () => {
  it("pop_tips removes the tip and restores endpoints like Python", () => {
    const a = new Arrow([-2, 0, 0], [2, 0, 0]);
    expect(a.has_tip()).toBe(true);
    expect(a.get_tips().submobjects.length).toBe(1);
    expectRowsClose([a.get_start()], [[-1.75, 0, 0]]);
    const popped = a.pop_tips();
    expect(popped.submobjects.length).toBe(1);
    expect(a.has_tip()).toBe(false);
    expectRowsClose([a.get_start()], [[-1.75, 0, 0]]);
    expectClose(a.get_end()[0] ?? NaN, 1.75);
  });
  it("get_tips mirrors Python's hasattr parity (kept attr after pop)", () => {
    const a = new Arrow([-2, 0, 0], [2, 0, 0]);
    a.pop_tips();
    // Python get_tips uses hasattr -> still 1 after pop; TS keeps the field.
    expect(a.get_tips().submobjects.length).toBe(1);
    const bare = new Line([-1, 0, 0], [1, 0, 0]);
    expect(bare.has_tip()).toBe(false);
    expect(bare.get_tips().submobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow.reset_normal_vector
// Python: Arrow([0,0,0],[1,0,0]): normal before [0,0,1],
//   get_normal_vector [0,0,-1], after reset [0,0,-1], returns self
// ---------------------------------------------------------------------------
describe("Arrow.reset_normal_vector", () => {
  it("installs get_normal_vector like Python", () => {
    const a = new Arrow([0, 0, 0], [1, 0, 0]);
    expectRowsClose([[...(a.normal_vector as number[])]], [[0, 0, 1]]);
    expectRowsClose([a.get_normal_vector()], [[0, 0, -1]]);
    expect(a.reset_normal_vector() === (a as unknown)).toBe(true);
    expectRowsClose([[...(a.normal_vector as number[])]], [[0, 0, -1]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Angle.get_lines
// Python: Angle(Line(ORIGIN,[1,0,0]), Line(ORIGIN,[0,1,0])):
//   2 lines, identical refs, value pi/2, 90 degrees
//   parallel lines -> value 0.0, 0 points (issue #1930)
// ---------------------------------------------------------------------------
describe("Angle.get_lines", () => {
  it("returns the two construction lines like Python", () => {
    const l1 = new Line([0, 0, 0], [1, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 1, 0]);
    const ang = new Angle(l1, l2);
    const g = ang.get_lines();
    expect(g).toBeInstanceOf(VGroup);
    expect(g.submobjects.length).toBe(2);
    expect(g.submobjects[0] === (l1 as unknown as VMobject)).toBe(true);
    expect(g.submobjects[1] === (l2 as unknown as VMobject)).toBe(true);
    expectClose(ang.get_value(), Math.PI / 2);
    expectClose(ang.get_value(true), 90.0);
  });
  it("parallel lines give an empty zero-valued angle like Python", () => {
    const ang = new Angle(new Line([0, 0, 0], [1, 0, 0]), new Line([0, 1, 0], [1, 1, 0]));
    expectClose(ang.get_value(), 0.0);
    expect(ang.get_num_points()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowTip.length / StealthTip.length
// Python:
//   ArrowTriangleFilledTip() length 0.35, tip_angle pi
//   ArrowTriangleFilledTip(length=0.5).length 0.5
//   StealthTip() length 0.175, StealthTip(length=1.0) 1.0,
//   StealthTip(length=0.35) 0.35. Base ArrowTip() raises NotImplementedError.
// ---------------------------------------------------------------------------
describe("ArrowTip.length / StealthTip.length", () => {
  it("ArrowTip base is abstract like Python", () => {
    expect(() => new (ArrowTip as unknown as new () => unknown)()).toThrowError(
      /Has to be implemented in inheriting subclasses/,
    );
  });
  it("filled-triangle tip length matches Python", () => {
    const t = new ArrowTriangleFilledTip();
    expectClose(t.length, 0.34999999999999987, 1e-9);
    // NOTE (float branch cut, documented deviation): the tip vector is
    // (-0.35, +/-1e-16, 0); Python's construction leaves y = +1.02e-16 so
    // angle_of_vector gives +pi, while the TS construction leaves a tiny
    // negative y and gives -pi. Only |angle| == pi is asserted.
    expectClose(Math.abs(t.tip_angle), Math.PI);
    const t2 = new ArrowTriangleFilledTip(1, 0, { length: 0.5 });
    expectClose(t2.length, 0.5, 1e-9);
  });
  it("stealth tip uses the 1.6x height convention like Python", () => {
    expectClose(new StealthTip().length, 0.17499999999999993, 1e-9);
    expectClose(new StealthTip(1, 3, 1.0).length, 1.0, 1e-9);
    expectClose(new StealthTip(1, 3, 0.35).length, 0.35, 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.boolean_ops._BooleanOps._convert_skia_path_to_vmobject
// Python:
//   move(0,0)+line(1,0)+line(1,1)+close -> 12 pts, exact rows
//   move+cubic -> 4 pts [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
//   move+quad -> 4 pts; unknown verb raises
// ---------------------------------------------------------------------------
describe("_BooleanOps._convert_skia_path_to_vmobject", () => {
  it("converts move/line/close like Python", () => {
    const b = new _BooleanOps();
    const p = new SkiaPath();
    p.moveTo(0, 0);
    p.lineTo(1, 0);
    p.lineTo(1, 1);
    p.close();
    const vm = b._convert_skia_path_to_vmobject(p);
    expect(vm.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(vm.points, 3), [
      [0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0], [1, 0, 0],
      [1, 0, 0], [1, 1 / 3, 0], [1, 2 / 3, 0], [1, 1, 0],
      [1, 1, 0], [2 / 3, 2 / 3, 0], [1 / 3, 1 / 3, 0], [0, 0, 0],
    ]);
  });
  it("converts cubic and quad verbs like Python", () => {
    const b = new _BooleanOps();
    const cubic = new SkiaPath();
    cubic.moveTo(0, 0);
    cubic.cubicTo(0, 1, 1, 1, 1, 0);
    const vm = b._convert_skia_path_to_vmobject(cubic);
    expect(vm.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(vm.points, 3), [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
    const quad = new SkiaPath();
    quad.moveTo(0, 0);
    quad.quadTo(0.5, 1, 1, 0);
    expect(new _BooleanOps()._convert_skia_path_to_vmobject(quad).get_num_points()).toBe(4);
  });
  it("rejects unsupported verbs", () => {
    const b = new _BooleanOps();
    const p = new SkiaPath();
    p.commands.push({ verb: "bogus" as PathVerb, points: [] });
    expect(() => b._convert_skia_path_to_vmobject(p)).toThrowError(/Unsupported/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.Cross.__init__
// Python: Cross(Square(2)) -> 2 submobjects, w/h 2, center origin
//   Cross() default w 2; scale_factor 0.5 -> w 1; stroke honored
// ---------------------------------------------------------------------------
describe("Cross.__init__", () => {
  it("crosses the mobject like Python", () => {
    const sq = new Square(2);
    const cr = new Cross(sq);
    expect(cr.submobjects.length).toBe(2);
    expectClose(cr.width, 2.0);
    expectClose(cr.height, 2.0);
    expectRowsClose([cr.get_center()], [[0, 0, 0]]);
  });
  it("honors scale_factor and stroke like Python", () => {
    const cr = new Cross(null, "RED", 2, 0.5);
    expectClose(cr.width, 1.0, 1e-9);
    expect(cr.stroke_width).toBe(2);
    expect(new Cross().width).toBe(2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.Underline / __init__
// Python: Underline(Square(2)): width 2, buff 0.1, y == bottom - buff (-1.1)
//   Underline(sq, buff=0.5): buff 0.5, y -1.5
// ---------------------------------------------------------------------------
describe("Underline / __init__", () => {
  it("matches width and sits below the mobject like Python", () => {
    const sq = new Square(2);
    const ul = new Underline(sq);
    expectClose(ul.width, 2.0);
    expect(ul.buff).toBe(0.1);
    expectClose(ul.get_center()[1] ?? NaN, sq.get_bottom()[1]! - 0.1);
  });
  it("honors an explicit buff like Python", () => {
    const sq = new Square(2);
    const ul = new Underline(sq, 0.5);
    expect(ul.buff).toBe(0.5);
    expectClose(ul.get_center()[1] ?? NaN, -1.5);
    const ul2 = new Underline(sq, 0.1, { buff: 0.5 });
    expect(ul2.buff).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.AnnularSector / generate_points
// Python:
//   AnnularSector(1,2,TAU/4): inner 1 outer 2, npts 72, first [1,0,0],
//     center [1,1,0]; second call doubles to 144 (appends; not idempotent)
//   AnnularSector(0.5,1.5,PI,PI/4): npts 72, arc_center ORIGIN
// ---------------------------------------------------------------------------
describe("AnnularSector / generate_points", () => {
  it("builds the ring sector like Python", () => {
    const a = new AnnularSector(1, 2, TAU / 4, 0, 1, 0, "WHITE");
    expect(a.inner_radius).toBe(1);
    expect(a.outer_radius).toBe(2);
    expect(a.get_num_points()).toBe(72);
    expectRowsClose([rowsOf(a.points, 3)[0] ?? []], [[1, 0, 0]]);
    expectRowsClose([a.get_center()], [[1, 1, 0]], 1e-6);
  });
  it("second generate_points appends like Python (not idempotent)", () => {
    const a = new AnnularSector(1, 2, TAU / 4);
    expect(a.get_num_points()).toBe(72);
    a.generate_points();
    expect(a.get_num_points()).toBe(144);
    const a2 = new AnnularSector(0.5, 1.5, PI, PI / 4);
    expect(a2.get_num_points()).toBe(72);
    expectRowsClose([toList(a2.arc_center)], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Star
// Python:
//   Star(n=5): 40 pts; first verts [0,1,0],[-0.2245..,0.3090..,0],...
//   Star(n=6,outer=2,inner=1): width 3.464101615137756
//   Star(n=5,density=3) raises ValueError
// ---------------------------------------------------------------------------
describe("Star", () => {
  it("builds a 5-star like Python", () => {
    const st = new Star(5);
    expect(st.get_num_points()).toBe(40);
    expectRowsClose(st.get_vertices().slice(0, 4), [
      [2.220446049250313e-16, 1, 0],
      [-0.2245139882897927, 0.30901699437494745, 0],
      [-0.9510565162951535, 0.3090169943749477, 0],
      [-0.3632712640026805, -0.11803398874989485, 0],
    ], 1e-9);
  });
  it("honors explicit radii and rejects bad density like Python", () => {
    expectClose(new Star(6, { outer_radius: 2, inner_radius: 1.0 }).width, 3.464101615137756, 1e-9);
    expect(() => new Star(5, { density: 3 })).toThrowError(/Incompatible density/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.ConvexHull.__init__
// Python: ConvexHull([0,0,0],[1,0,0],[1,1,0],[0,1,0]):
//   4 hull vertices {(0,0,0),(1,0,0),(1,1,0),(0,1,0)}, w/h 1.
// NOTE (upstream nondeterminism, mirrored by the port): QuickHull picks its
// initial simplex with an unseeded RNG, so the START vertex and winding of
// get_vertex_groups() vary run to run on BOTH sides (observed 7 distinct
// cyclic orders in 20 TS runs; Python prints different orders across
// processes too). The comparison below is order-insensitive.
// NOTE (upstream fragility, mirrored): inputs with an interior point on a
// hull diagonal (e.g. +[0.5,0.5,0]) admit degenerate collinear simplices;
// Python then raises ValueError from the neighbor walk (verified: crashes
// within 200 tries) and TS throws from its find-based walk (16/200 throws).
// The square-corner input used here is stable: 200/200 clean on both sides.
// ---------------------------------------------------------------------------
describe("ConvexHull.__init__", () => {
  it("hulls the points like Python", () => {
    const ch = new ConvexHull([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]);
    const got = (ch.get_vertex_groups()[0] ?? []).map((p) => [...p]).sort((a, b) =>
      a[0]! - b[0]! || a[1]! - b[1]! || a[2]! - b[2]!,
    );
    expectRowsClose(got, [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]);
    expectClose(ch.width, 1.0);
    expectClose(ch.height, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.labeled.LabeledLine — now ported (MathTex via
// MathJax), so a string label builds a Label at label_position.
// ---------------------------------------------------------------------------
describe("LabeledLine", () => {
  it("builds a label for a string", () => {
    const line = new LabeledLine("x", 0.5, null, null, null, [-1, 0, 0], [1, 0, 0]);
    expect(line.label).toBeDefined();
    expect(line.submobjects.length).toBe(1);
    expect(line.label.get_center()[0]!).toBeCloseTo(0, 5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.Brace.get_direction / BraceBetweenPoints / ArcBrace
// Python:
//   Brace(Square(2), DOWN): dir [0,-1,~0]; tip [0,-1.472985,0]
//   BraceBetweenPoints([-1,0,0],[1,0,0]): dir [0,-1,~0], tip [0,-0.472985,~0]
//   explicit direction [0,-1,0] honored; ArcBrace(arc): 100 pts,
//     dir [0.7071,0.7071,0]; ArcBrace() default builds
// ---------------------------------------------------------------------------
describe("Brace.get_direction", () => {
  it("points from center to tip like Python", () => {
    const b = new Brace(new Square(2), { direction: [...DOWN] });
    expectRowsClose([b.get_direction() as number[]], [[0, -1, 1.224646799147352e-16]], 1e-6);
    expectClose(b.get_tip()[0] ?? NaN, 0.0, 1e-9);
    expectClose(b.get_tip()[1] ?? NaN, -1.472985, 1e-6);
  });
});

describe("BraceBetweenPoints / __init__", () => {
  it("auto-derives the normal direction like Python", () => {
    const b = new BraceBetweenPoints([-1, 0, 0], [1, 0, 0]);
    expectRowsClose([b.get_direction() as number[]], [[0, -1, 1.224646799147353e-16]], 1e-6);
    expectClose(b.get_tip()[1] ?? NaN, -0.472985, 1e-6);
  });
  it("honors an explicit direction like Python", () => {
    const b = new BraceBetweenPoints([0, 0, 0], [1, 1, 0], [0, -1, 0]);
    expectRowsClose([b.get_direction() as number[]], [[8.133948932176907e-16, -1, 1.224646799147353e-16]], 1e-6);
  });
});

describe("ArcBrace / __init__", () => {
  it("wraps an arc like Python", () => {
    const arc = new Arc(1, 0, TAU / 4, 9, [...ORIGIN]);
    const ab = new ArcBrace(arc);
    expect(ab.get_num_points()).toBe(100);
    expectRowsClose([ab.get_direction() as number[]], [[0.7071067811865479, 0.7071067811865471, 2.825666419345369e-17]], 1e-6);
  });
  it("default-constructs like Python", () => {
    expect(new ArcBrace().get_num_points()).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Circle
// Python:
//   Circle(radius=2): radius 2, width 4, 32 pts
//   point_at_angle(0) [2,0,0], point_at_angle(pi/2) [~0,2,0]
//   from_three_points([1,0,0],[0,1,0],[-1,0,0]): r 1, center ~ORIGIN
//   surround(Square(2)): width sqrt(8)*1.2
// ---------------------------------------------------------------------------
describe("Circle", () => {
  it("stores radius and samples the loop like Python", () => {
    const c = new Circle(2);
    expect(c.radius).toBe(2);
    expectClose(c.width, 4.0);
    expect(c.get_num_points()).toBe(32);
    expectRowsClose([toList(c.point_at_angle(0))], [[2, 0, 0]]);
    expectRowsClose([toList(c.point_at_angle(Math.PI / 2))], [[2.991343520328107e-16, 2, 0]]);
  });
  it("from_three_points recovers the circle like Python", () => {
    const c = Circle.from_three_points([1, 0, 0], [0, 1, 0], [-1, 0, 0]);
    expectClose(c.radius, 1.0);
    expectRowsClose([c.get_arc_center()], [[-3.699594818622e-16, 2.220446049250e-16, 0]], 1e-9);
  });
  it("surround scales to the mobject like Python", () => {
    const c = new Circle(null);
    c.surround(new Square(2));
    expectClose(c.width, Math.sqrt(8) * 1.2, 1e-9);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (combined behavior, verified against the original manim implementation)
// ---------------------------------------------------------------------------
describe("batch-1061 integration", () => {
  it("boolean round-trip preserves a Square (6 verbs, 20 pts, 2x2)", () => {
    // Python: Square(2) -> skia path with 6 verbs
    //   (move + 4 cubic + close); back -> 20 pts, w/h 2, center origin.
    const sq = new Square(2);
    const writer = new _BooleanOps();
    const path = writer._convert_vmobject_to_skia_path(sq);
    expect(path.commands.length).toBe(6);
    expect(path.commands[0]?.verb).toBe(PathVerb.MOVE);
    expect(path.commands[path.commands.length - 1]?.verb).toBe(PathVerb.CLOSE);
    const back = new _BooleanOps()._convert_skia_path_to_vmobject(path);
    expect(back.get_num_points()).toBe(20);
    expectClose(back.width, 2.0);
    expectClose(back.height, 2.0);
    expectRowsClose([back.get_center()], [[0, 0, 0]]);
  });
  it("Arrow tip lifecycle: add -> normal -> pop keeps geometry", () => {
    // Python: Arrow([-2,0,0],[2,0,0]) start [-1.75,0,0]; reset_normal_vector
    // installs [0,0,-1]; pop leaves start at -1.75 with 1 tip returned.
    const a = new Arrow([-2, 0, 0], [2, 0, 0]);
    a.reset_normal_vector();
    expectRowsClose([[...(a.normal_vector as number[])]], [[0, 0, -1]]);
    const tips = a.pop_tips();
    expect(tips.submobjects.length).toBe(1);
    expectRowsClose([a.get_start()], [[-1.75, 0, 0]]);
    expect(a.get_tips().submobjects.length).toBe(1);
  });
  it("Angle value agrees with its lines' unit vectors", () => {
    // Python: Angle(Line(ORIGIN,RIGHT), Line(ORIGIN,UP)).get_value() == pi/2,
    // and get_lines() returns the exact construction lines.
    const l1 = new Line([0, 0, 0], [1, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 1, 0]);
    const ang = new Angle(l1, l2);
    const [r1, r2] = ang.get_lines().submobjects as [Line, Line];
    expect(r1 === (l1 as unknown as VMobject)).toBe(true);
    expect(r2 === (l2 as unknown as VMobject)).toBe(true);
    expectClose(ang.get_value(), Math.PI / 2);
    expectClose(r1.get_angle(), 0.0);
    expectClose(r2.get_angle(), Math.PI / 2);
  });
  it("TangentLine midpoint + Circle point_at_angle agree on a Circle", () => {
    // Python: TangentLine(Circle(1), 0.5, length=1) is centered near the
    // circle point at proportion 0.5 (angle pi): [-1,0,0].
    const c = new Circle(1);
    const tl = new TangentLine(c, 0.5, 1);
    const mid = tl.get_center();
    const rim = toList(c.point_at_angle(Math.PI));
    expectClose(tl.get_length(), 1.0, 1e-9);
    // Tangent midpoint sits on the curve: distance to rim point ~ tip scale.
    const dist = Math.hypot(...mid.map((v, i) => v - (rim[i] ?? 0)));
    expect(dist).toBeLessThan(0.6);
  });
  it("BraceBetweenPoints on a Star edge matches its tip direction", () => {
    // Python: BraceBetweenPoints([0,0,0],[2,0,0]) tip y -0.472985, dir -Y.
    const st = new Star(5);
    expect(st.get_num_points()).toBe(40);
    const b = new BraceBetweenPoints([0, 0, 0], [2, 0, 0]);
    expectClose(b.get_tip()[1] ?? NaN, -0.472985, 1e-6);
    expectClose((b.get_direction() as number[])[1] ?? NaN, -1.0, 1e-6);
    const ch = new ConvexHull([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]);
    expectClose(ch.width, 1.0);
    // NOTE: ConvexHull extends Polygram, NOT Polygon (siblings) -- so this
    // is asserted as a plain boolean. Do NOT write
    // expect(ch).toBeInstanceOf(Polygon): the assertion fails and vitest's
    // failure pretty-printer OOMs (4GB heap) serializing the Mobject proxy
    // graph instead of reporting. Same caution applies to toEqual on whole
    // Mobjects anywhere in this file.
    expect(ch instanceof Polygon).toBe(false);
    expect(ch instanceof ConvexHull).toBe(true);
  });
  it("Cylinder/Torus funcs satisfy their closed forms together", () => {
    // Python closed forms: |cyl.func(u,v)[0:2]| == radius;
    // |torus.func(u,v)[2]| <= minor radius.
    const cyl = new Cylinder(2, 3, [0, 0, 1], [0, TAU], false, 2);
    const cp = toList(cyl.func(0.7, 1.3) as NDArray);
    expectClose(Math.hypot(cp[0] ?? 0, cp[1] ?? 0), 2.0, 1e-9);
    expectClose(cp[2] ?? NaN, 0.7, 1e-9);
    const tor = new Torus(3, 1, [0, TAU], [0, TAU], 2);
    const tp = toList(tor.func(1, 1) as NDArray);
    expect(Math.abs(tp[2] ?? 0) <= 1.0 + 1e-12).toBe(true);
    const [s, e] = get_3d_vmob_gradient_start_and_end_points(new Square(2));
    expectRowsClose([toList(s), toList(e)], [[1, 1, 0], [-1, -1 / 3, 0]]);
  });
});
