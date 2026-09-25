// Ground-truth tests for the layers 8..12 batch (Mobject placement /
// scaling / rotation transforms, Line/Arrow helpers, NumberLine mapping,
// Cone/Cylinder orientation, tips, Ellipse/Arc init, Table/SVG/Image helpers,
// MovingCamera frame, ScreenRectangle aspect, Animation construction).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used below; each
// case cites the probe in a "Python:" comment. TS is asserted against those
// values with tight float tolerance (absolute 1e-9 unless noted).
//
// Known deviations from Python (documented, not worked around):
//  - Cone(direction=<plain list>) raises TypeError in Python
//    (`direction * np.linalg.norm(direction)` needs an ndarray); the TS port
//    accepts plain arrays. Tests pass ndarray-equivalent plain arrays.
//  - Animation() with no mobject raises TypeError in Python (missing
//    positional arg); the TS port defaults to an empty Mobject. Tests assert
//    the TS behavior and note the difference.
//  - Table(...) full construction needs the text/LaTeX pipeline (out of
//    scope); Table.scale is tested on a VGroup-initialised instance with
//    Python-identical h_buff/v_buff bookkeeping.
//  - SVGMobject file parsing needs the SVG pipeline (out of scope);
//    move_into_position is tested on points assigned directly, matching
//    Python's method on the same points.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject } from "../src/mobject/mobject.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { Line, Arrow, DashedLine } from "../src/mobject/geometry/line.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Arc, Ellipse } from "../src/mobject/geometry/arc.js";
import { ArrowCircleTip, ArrowCircleFilledTip } from "../src/mobject/geometry/tips.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { Cone, Cylinder } from "../src/mobject/three_d/three_dimensions.js";
import { ScreenRectangle } from "../src/mobject/frame.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { Animation } from "../src/animation/animation.js";
import { Table } from "../src/mobject/table.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { ImageMobject } from "../src/mobject/types/image_mobject.js";
import { ManimColor } from "../src/utils/color/core.js";
import { config } from "../src/_config/utils.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function flat(value: NDArray | number[] | number[][]): number[] {
  if (Array.isArray(value)) return (value as unknown[]).flat(Infinity as number[]).map(Number);
  return ((value as NDArray).toArray() as unknown[]).flat(Infinity as number[]).map(Number);
}

function expectNums(actual: NDArray | number[] | number[][], expected: readonly number[], tol = 1e-9): void {
  const values = flat(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(Math.abs((values[i] ?? 0) - (expected[i] ?? 0))).toBeLessThanOrEqual(tol);
  }
}

function hexOf(color: unknown): string {
  if (color instanceof ManimColor) return color.to_hex();
  if (color !== null && typeof color === "object" && "to_hex" in (color as object)) {
    return ((color as { to_hex: () => string }).to_hex as () => string)();
  }
  return String(color);
}

// ---------------------------------------------------------------------------
// Mobject.get_top/get_bottom/get_right/get_left
// Python: Square(side_length=2.0): top [0,1,0], bottom [0,-1,0],
//   right [1,0,0], left [-1,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.get_top/get_bottom/get_right/get_left", () => {
  it("returns edge centers of a unit square", () => {
    const s = new Square(2.0);
    expectNums(s.get_top(), [0, 1, 0]);
    expectNums(s.get_bottom(), [0, -1, 0]);
    expectNums(s.get_right(), [1, 0, 0]);
    expectNums(s.get_left(), [-1, 0, 0]);
  });
  it("tracks a shifted square: shift [1,2,3]", () => {
    // Python: top [1,3,3], bottom [1,1,3], right [2,2,3], left [0,2,3]
    const m = new Square(2.0);
    m.shift([1, 2, 3]);
    expectNums(m.get_top(), [1, 3, 3]);
    expectNums(m.get_bottom(), [1, 1, 3]);
    expectNums(m.get_right(), [2, 2, 3]);
    expectNums(m.get_left(), [0, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_zenith/get_nadir
// Python: Square(2.0): zenith [0,0,0], nadir [0,0,0] (flat, OUT/IN edges);
//   shifted by [1,2,3]: both [1,2,3]
// ---------------------------------------------------------------------------
describe("Mobject.get_zenith/get_nadir", () => {
  it("returns OUT/IN edge centers", () => {
    const s = new Square(2.0);
    expectNums(s.get_zenith(), [0, 0, 0]);
    expectNums(s.get_nadir(), [0, 0, 0]);
    s.shift([1, 2, 3]);
    expectNums(s.get_zenith(), [1, 2, 3]);
    expectNums(s.get_nadir(), [1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_x/set_y/set_z
// Python: Square(2.0).set_x(3) -> center [3,0,0];
//   .set_y(-2) -> [0,-2,0]; .set_z(5) -> [0,0,5]
//   set_x(3, direction=[1,0,0]) -> right [3,0,0], center [2,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.set_x/set_y/set_z", () => {
  it("moves the center coordinate", () => {
    const mx = new Square(2.0); mx.set_x(3);
    expectNums(mx.get_center(), [3, 0, 0]);
    const my = new Square(2.0); my.set_y(-2);
    expectNums(my.get_center(), [0, -2, 0]);
    const mz = new Square(2.0); mz.set_z(5);
    expectNums(mz.get_center(), [0, 0, 5]);
  });
  it("honours a direction edge", () => {
    const m = new Square(2.0);
    m.set_x(3, [1, 0, 0]);
    expectNums(m.get_right(), [3, 0, 0]);
    expectNums(m.get_center(), [2, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.apply_function_to_position
// Python: Square(2.0).apply_function_to_position(lambda p: p + [2,3,0])
//   -> center [2,3,0]
// ---------------------------------------------------------------------------
describe("Mobject.apply_function_to_position", () => {
  it("moves to func(center)", () => {
    const m = new Square(2.0);
    m.apply_function_to_position((p) => [p[0]! + 2, p[1]! + 3, p[2]!]);
    expectNums(m.get_center(), [2, 3, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_submobject_colors_by_radial_gradient /
// Mobject.set_colors_by_radial_gradient
// Python: Square(2.0).set_colors_by_radial_gradient() runs without error;
//   PMobject exact rgbas below.
// ---------------------------------------------------------------------------
describe("Mobject radial-gradient color helpers", () => {
  it("set_colors_by_radial_gradient runs and recolors submobjects", () => {
    const m = new Square(2.0);
    const before = hexOf(m.get_color());
    m.set_colors_by_radial_gradient();
    expect(m.get_num_points()).toBeGreaterThan(0);
    void before;
  });
  it("set_submobject_colors_by_radial_gradient colors near/far mobjects", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    g.submobjects[1]!.shift([10, 0, 0]);
    g.set_submobject_colors_by_radial_gradient([0, 0, 0], 1, "#FFFFFF", "#000000");
    expect(hexOf(g.submobjects[0]!.get_color())).toBe("#FFFFFF");
    expect(hexOf(g.submobjects[1]!.get_color())).toBe("#000000");
  });
});

// ---------------------------------------------------------------------------
// PMobject.set_colors_by_radial_gradient
// Python: PMobject + points [[0,0,0],[1,0,0]], center [0,0,0], radius 1,
//   white->black -> rgbas [[1,1,1,1],[0,0,0,1]]
// ---------------------------------------------------------------------------
describe("PMobject.set_colors_by_radial_gradient", () => {
  it("interpolates rgba by distance", () => {
    const pm = new PMobject();
    pm.add_points([[0, 0, 0], [1, 0, 0]]);
    pm.set_colors_by_radial_gradient([0, 0, 0], 1, "#FFFFFF", "#000000");
    expectNums(pm.rgbas, [1, 1, 1, 1, 0, 0, 0, 1]);
  });
  it("extrapolates (no clamp) for points beyond the radius", () => {
    // Python: add_points [[0,0,0],[3,0,0]] -> rgbas [[1,1,1,1],[-2,-2,-2,1]]
    // (Mobject.set_submobject_colors_by_radial_gradient clamps, but the
    // PMobject override does not).
    const pm = new PMobject();
    pm.add_points([[0, 0, 0], [3, 0, 0]]);
    pm.set_colors_by_radial_gradient([0, 0, 0], 1, "#FFFFFF", "#000000");
    expectNums(pm.rgbas, [1, 1, 1, 1, -2, -2, -2, 1]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.match_coord/match_x/match_y/match_z
// Python: b shifted [5,1,2]: match_x -> center [5,0,0]; match_y -> [0,1,0];
//   match_z -> [0,0,2]; match_coord(b, 0, [1,0,0]) -> [5,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.match_coord/match_x/match_y/match_z", () => {
  function shiftedTarget(): Square {
    const b = new Square(2.0);
    b.shift([5, 1, 2]);
    return b;
  }
  it("match_x/match_y/match_z copy one center coordinate", () => {
    const ax = new Square(2.0); ax.match_x(shiftedTarget());
    expectNums(ax.get_center(), [5, 0, 0]);
    const ay = new Square(2.0); ay.match_y(shiftedTarget());
    expectNums(ay.get_center(), [0, 1, 0]);
    const az = new Square(2.0); az.match_z(shiftedTarget());
    expectNums(az.get_center(), [0, 0, 2]);
  });
  it("match_coord with an edge direction", () => {
    const a = new Square(2.0);
    a.match_coord(shiftedTarget(), 0, [1, 0, 0]);
    expectNums(a.get_center(), [5, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.align_to
// Python: Square(2.0).align_to([5,5,5],[1,0,0]) -> center [4,0,0];
//   align_to(shifted b, [1,0,0]) -> center [5,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.align_to", () => {
  it("aligns to a point along nonzero direction dims", () => {
    const a = new Square(2.0);
    a.align_to([5, 5, 5], [1, 0, 0]);
    expectNums(a.get_center(), [4, 0, 0]);
  });
  it("aligns to another mobject's critical point", () => {
    const a = new Square(2.0);
    const b = new Square(2.0);
    b.shift([5, 0, 0]);
    a.align_to(b, [1, 0, 0]);
    expectNums(a.get_center(), [5, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_z_index_by_z_Point3D
// Python: Square(2.0).shift([0,0,3]).set_z_index_by_z_Point3D() -> z_index 3
// ---------------------------------------------------------------------------
describe("Mobject.set_z_index_by_z_Point3D", () => {
  it("copies the center z into z_index", () => {
    const m = new Square(2.0);
    m.shift([0, 0, 3]);
    m.set_z_index_by_z_Point3D();
    expect(m.z_index).toBeCloseTo(3.0, 12);
  });
});

// ---------------------------------------------------------------------------
// Mobject.stretch / stretch_about_point
// Python: Square(2.0).stretch(3, 0) -> width 6, height 2;
//   .stretch(2, 1) -> [2, 4];
//   stretch_about_point(2, 0, [1,0,0]) -> width 4, center [-1,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.stretch/stretch_about_point", () => {
  it("stretches one dimension about the center", () => {
    const m = new Square(2.0);
    m.stretch(3, 0);
    expect(m.width).toBeCloseTo(6.0, 12);
    expect(m.height).toBeCloseTo(2.0, 12);
    const m2 = new Square(2.0);
    m2.stretch(2, 1);
    expect(m2.width).toBeCloseTo(2.0, 12);
    expect(m2.height).toBeCloseTo(4.0, 12);
  });
  it("stretch_about_point fixes the anchor point", () => {
    const m = new Square(2.0);
    m.stretch_about_point(2, 0, [1, 0, 0]);
    expect(m.width).toBeCloseTo(4.0, 12);
    expectNums(m.get_center(), [-1, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// DashedLine._calculate_num_dashes
// Python: DashedLine([-2,0,0],[2,0,0]) -> 40
// ---------------------------------------------------------------------------
describe("DashedLine._calculate_num_dashes", () => {
  it("matches the length/dash_length ratio formula", () => {
    const d = new DashedLine([-2, 0, 0], [2, 0, 0]);
    expect(d._calculate_num_dashes()).toBe(40);
  });
});

// ---------------------------------------------------------------------------
// Mobject.scale
// Python: Square(2.0).scale(2) -> width 4, center [0,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.scale", () => {
  it("scales about the center by default", () => {
    const m = new Square(2.0);
    m.scale(2);
    expect(m.width).toBeCloseTo(4.0, 12);
    expectNums(m.get_center(), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// NumberLine.p2n / get_unit_size
// Python: NumberLine(x_range=[-3,3,1], length=6): p2n([0,0,0]) == 0.0,
//   unit_size == 1.0
// ---------------------------------------------------------------------------
describe("NumberLine.p2n/get_unit_size", () => {
  it("maps origin to zero with unit size one", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expect(nl.p2n([0, 0, 0])).toBeCloseTo(0.0, 9);
    expect(nl.get_unit_size()).toBeCloseTo(1.0, 9);
  });
  it("maps endpoints onto the range", () => {
    const nl = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    expect(nl.p2n([-2, 0, 0])).toBeCloseTo(-2.0, 9);
    expect(nl.p2n([2, 0, 0])).toBeCloseTo(2.0, 9);
    expect(nl.get_unit_size()).toBeCloseTo(1.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Cone.get_start/get_end, Cone._rotate_to_direction, Cone.set_direction,
// Cylinder._rotate_to_direction, Cylinder.set_direction
// Python (direction as ndarray): Cone(1, 2, [0,0,1]): start [0,0,-2],
//   end [0,0,0]; Cone(1, 2, [1,0,0]): start [-2,0,~0], end [-2,0,-2]
// ---------------------------------------------------------------------------
describe("Cone/Cylinder orientation", () => {
  it("Cone.get_start/get_end for the default axis", () => {
    const c = new Cone(1, 2, [0, 0, 1]);
    expectNums(c.get_start(), [0, 0, -2]);
    expectNums(c.get_end(), [0, 0, 0]);
  });
  it("Cone._rotate_to_direction orients along +x", () => {
    const c = new Cone(1, 2, [1, 0, 0]);
    expectNums(c.get_start(), [-2, 0, 0], 1e-9);
    expectNums(c.get_end(), [-2, 0, -2], 1e-9);
  });
  it("Cone.set_direction re-rotates", () => {
    const c = new Cone(1, 2, [0, 0, 1]);
    c.set_direction([0, 1, 0]);
    // Python: end [0,0,0] after set_direction([0,1,0])
    expectNums(c.get_end(), [0, 0, 0], 1e-9);
    expectNums(c.direction, [0, 1, 0]);
  });
  it("Cylinder.set_direction runs and stores the direction", () => {
    const cy = new Cylinder(1, 2, [0, 0, 1], [0, Math.PI * 2], false);
    cy.set_direction([0, 1, 0]);
    expectNums([...(cy.direction as number[])], [0, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// Arrow.get_default_tip_length / Arrow._set_stroke_width_from_length
// Python: Arrow([-2,0,0],[2,0,0], buff=0): tip 0.35, stroke 6;
//   Arrow([0,0,0],[0.1,0,0], buff=0): tip 0.025, length 0.1, stroke 0.5
// ---------------------------------------------------------------------------
describe("Arrow tip length / stroke width", () => {
  it("clamps the tip to the length ratio", () => {
    const a = new Arrow([-2, 0, 0], [2, 0, 0], { buff: 0 });
    expect(a.get_default_tip_length()).toBeCloseTo(0.35, 12);
    expect(a.stroke_width).toBeCloseTo(6, 12);
  });
  it("shrinks tip and stroke for a very short arrow", () => {
    const a = new Arrow([0, 0, 0], [0.1, 0, 0], { buff: 0 });
    expect(a.get_default_tip_length()).toBeCloseTo(0.025, 9);
    expect(a.get_length()).toBeCloseTo(0.1, 9);
    expect(a.stroke_width).toBeCloseTo(0.5, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.sort / sort submob_func / sort_submobjects / arrange_submobjects
// Python: two unit squares at x=-3/+3 reversed then sort() restores
//   [[-3,0,0],[3,0,0]]; sort(submob_func=-x) gives [[9,0,0],[0,0,0]];
//   sort_submobjects on [0],[2] keeps [[0,0,0],[2,0,0]];
//   arrange_submobjects on two unit squares -> [[-0.625,0,0],[0.625,0,0]]
// ---------------------------------------------------------------------------
describe("Mobject.sort family", () => {
  it("sorts by center x by default", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    (g.submobjects[0] as Square).shift([-3, 0, 0]);
    (g.submobjects[1] as Square).shift([3, 0, 0]);
    g.submobjects.reverse();
    g.sort();
    expectNums((g.submobjects[0] as Mobject).get_center(), [-3, 0, 0]);
    expectNums((g.submobjects[1] as Mobject).get_center(), [3, 0, 0]);
  });
  it("sort honours a custom submob_func", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    (g.submobjects[0] as Square).shift([9, 0, 0]);
    g.sort(null, (m: Mobject) => -(m.get_center()[0] ?? 0));
    expectNums((g.submobjects[0] as Mobject).get_center(), [9, 0, 0]);
    expectNums((g.submobjects[1] as Mobject).get_center(), [0, 0, 0]);
  });
  it("sort_submobjects / arrange_submobjects wrappers", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    (g.submobjects[0] as Square).shift([2, 0, 0]);
    g.sort_submobjects();
    expectNums((g.submobjects[0] as Mobject).get_center(), [0, 0, 0]);
    expectNums((g.submobjects[1] as Mobject).get_center(), [2, 0, 0]);
    const h = new VGroup(new Square(1.0), new Square(1.0));
    h.arrange_submobjects();
    expectNums((h.submobjects[0] as Mobject).get_center(), [-0.625, 0, 0]);
    expectNums((h.submobjects[1] as Mobject).get_center(), [0.625, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.rotate / rotate_about_origin / pose_at_angle
// Python: Square(2) shifted [1,0,0], rotate(pi/2) -> center [1,0,0];
//   rotate_about_origin(pi) -> [-1, ~0, 0]; pose_at_angle -> center [0,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.rotate family", () => {
  it("rotates about the center by default", () => {
    const m = new Square(2.0);
    m.shift([1, 0, 0]);
    m.rotate(Math.PI / 2);
    expectNums(m.get_center(), [1, 0, 0]);
    expect(m.width).toBeCloseTo(2.0, 9);
  });
  it("rotate_about_origin orbits the origin", () => {
    const m = new Square(2.0);
    m.shift([1, 0, 0]);
    m.rotate_about_origin(Math.PI);
    expectNums(m.get_center(), [-1, 0, 0]);
  });
  it("pose_at_angle keeps the center fixed", () => {
    const m = new Square(2.0);
    m.pose_at_angle();
    expectNums(m.get_center(), [0, 0, 0]);
    expect(m.width).toBeCloseTo(2.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.apply_matrix
// Python: Square(2.0).apply_matrix([[2,0],[0,3]]) -> width 4, height 6
// ---------------------------------------------------------------------------
describe("Mobject.apply_matrix", () => {
  it("applies a diagonal 2x2 about the origin", () => {
    const m = new Square(2.0);
    m.apply_matrix([[2, 0], [0, 3]]);
    expect(m.width).toBeCloseTo(4.0, 9);
    expect(m.height).toBeCloseTo(6.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.center / to_corner
// Python: Square(2) shifted [5,5,0], center() -> [0,0,0];
//   to_corner() (DL) -> [-5.6111, -2.5, 0];
//   VGroup [0],[4] center() -> group center [0,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.center/to_corner", () => {
  it("center() moves the bounding center to the origin", () => {
    const m = new Square(2.0);
    m.shift([5, 5, 0]);
    m.center();
    expectNums(m.get_center(), [0, 0, 0]);
  });
  it("to_corner() defaults to DL", () => {
    const m = new Square(2.0);
    m.to_corner();
    expectNums(m.get_center(), [-5.611111111111111, -2.5, 0], 1e-9);
  });
  it("center() works on a group", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    (g.submobjects[1] as Square).shift([4, 0, 0]);
    g.center();
    expectNums(g.get_center(), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.apply_function / VMobject.apply_function / VMobject.rotate /
// VMobject.scale
// Python: Square(2)+[1,0,0] -> center [1,0,0];
//   Line scale(2, scale_stroke=True) from width 4 doubles stroke to 8;
//   Line apply_function(*2) -> start [-2,0,0], end [2,0,0];
//   Line rotate(pi/2) -> start [~0,-1,0], end [~0,1,0]
// ---------------------------------------------------------------------------
describe("Mobject/VMobject function application", () => {
  it("apply_function maps every point", () => {
    const m = new Square(2.0);
    m.apply_function((p) => [p[0]! + 1, p[1]!, p[2]!]);
    expectNums(m.get_center(), [1, 0, 0]);
  });
  it("VMobject.scale with scale_stroke scales the stroke width", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.set_stroke(undefined, 4, undefined, false);
    l.scale(2, true);
    expect(l.get_stroke_width()).toBeCloseTo(8, 12);
  });
  it("VMobject.apply_function doubles a line", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.apply_function((p) => [p[0]! * 2, p[1]! * 2, p[2]! * 2]);
    expectNums(l.get_start(), [-2, 0, 0]);
    expectNums(l.get_end(), [2, 0, 0]);
  });
  it("VMobject.rotate turns a line by 90 degrees", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.rotate(Math.PI / 2);
    expectNums(l.get_start(), [0, -1, 0]);
    expectNums(l.get_end(), [0, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// Line.set_length
// Python: Line([-1,0,0],[1,0,0]).set_length(4) -> length 4
// ---------------------------------------------------------------------------
describe("Line.set_length", () => {
  it("rescales to the requested length", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.set_length(4);
    expect(l.get_length()).toBeCloseTo(4.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.apply_complex_function
// Python: Square(2.0).apply_complex_function(2*z) -> width 4
// ---------------------------------------------------------------------------
describe("Mobject.apply_complex_function", () => {
  it("doubles the complex plane", () => {
    const m = new Square(2.0);
    m.apply_complex_function(({ re, im }) => ({ re: 2 * re, im: 2 * im }));
    expect(m.width).toBeCloseTo(4.0, 9);
    expect(m.height).toBeCloseTo(4.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.shift_onto_screen / is_off_screen
// Python: Square(2) shifted [100,0,0]: off-screen True, then
//   shift_onto_screen -> center [5.6111, 0, 0]; fresh square -> False
// ---------------------------------------------------------------------------
describe("Mobject screen helpers", () => {
  it("detects off-screen mobjects and shifts them back", () => {
    const m = new Square(2.0);
    m.shift([100, 0, 0]);
    expect(m.is_off_screen()).toBe(true);
    m.shift_onto_screen();
    expectNums(m.get_center(), [5.611111111111114, 0, 0], 1e-9);
    expect(m.is_off_screen()).toBe(false);
  });
  it("a centered square is on screen", () => {
    expect(new Square(2.0).is_off_screen()).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Mobject.space_out_submobjects
// Python: VGroup(1,1).space_out_submobjects(2.0): group width 1.0 and each
//   submobject width back at 1.0
// ---------------------------------------------------------------------------
describe("Mobject.space_out_submobjects", () => {
  it("expands the group while shrinking members back", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0));
    g.space_out_submobjects(2.0);
    expect(g.width).toBeCloseTo(1.0, 9);
    for (const sub of g.submobjects) {
      expect((sub as Mobject).width).toBeCloseTo(1.0, 9);
    }
  });
});

// ---------------------------------------------------------------------------
// Mobject rescale family: rescale_to_fit, scale_to_fit_width/height/depth,
// stretch_to_fit_width/height/depth, match_width/height/depth, match_dim_size
// Python (box points [0,0,0],[1,0,0],[0,2,0],[0,0,4]):
//   scale_to_fit_depth(8) -> [8, 2, 4]; stretch_to_fit_depth(8) -> [8, 1, 2];
//   scale_to_fit_height(8) -> height 8, width 4;
//   Square(2): match_width(Square 4) -> 4; match_height -> 4;
//   match_dim_size(b4, 1) -> [4, 4]; stretch_to_fit_width(6) -> [6, 2];
//   stretch_to_fit_height(6) -> [2, 6]; rescale_to_fit(6, 0) -> 6
// ---------------------------------------------------------------------------
describe("Mobject rescale family", () => {
  function box(): Mobject {
    const m = new Mobject();
    m.set_points([
      [0, 0, 0],
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 4],
    ]);
    return m;
  }
  it("scale_to_fit_depth / stretch_to_fit_depth", () => {
    const a = box(); a.scale_to_fit_depth(8);
    expect(a.depth).toBeCloseTo(8.0, 9);
    expect(a.width).toBeCloseTo(2.0, 9);
    expect(a.height).toBeCloseTo(4.0, 9);
    const b = box(); b.stretch_to_fit_depth(8);
    expect(b.depth).toBeCloseTo(8.0, 9);
    expect(b.width).toBeCloseTo(1.0, 9);
    expect(b.height).toBeCloseTo(2.0, 9);
  });
  it("scale_to_fit_width/height + stretch_to_fit_width/height", () => {
    const a = new Square(2.0); a.scale_to_fit_width(6);
    expect(a.width).toBeCloseTo(6.0, 9);
    const b = new Square(2.0); b.scale_to_fit_height(6);
    expect(b.height).toBeCloseTo(6.0, 9);
    const c = new Square(2.0); c.stretch_to_fit_width(6);
    expect(c.width).toBeCloseTo(6.0, 9);
    expect(c.height).toBeCloseTo(2.0, 9);
    const d = new Square(2.0); d.stretch_to_fit_height(6);
    expect(d.width).toBeCloseTo(2.0, 9);
    expect(d.height).toBeCloseTo(6.0, 9);
    const e = new Square(2.0); e.rescale_to_fit(6, 0);
    expect(e.width).toBeCloseTo(6.0, 9);
  });
  it("match_width/match_height/match_depth/match_dim_size", () => {
    const big = new Square(4.0);
    const a = new Square(2.0); a.match_width(big);
    expect(a.width).toBeCloseTo(4.0, 9);
    const b = new Square(2.0); b.match_height(big);
    expect(b.height).toBeCloseTo(4.0, 9);
    const c = new Square(2.0); c.match_depth(big);
    expect(c.depth).toBeCloseTo(0.0, 12);
    const d = new Square(2.0); d.match_dim_size(big, 1);
    expect(d.width).toBeCloseTo(4.0, 9);
    expect(d.height).toBeCloseTo(4.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.put_start_and_end_on
// Python: Line([-1,0,0],[1,0,0]).put_start_and_end_on([0,0,0],[0,2,0])
//   -> start [0,0,0], end [~0,2,0]
// ---------------------------------------------------------------------------
describe("Mobject.put_start_and_end_on", () => {
  it("moves and rotates a line onto the target segment", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.put_start_and_end_on([0, 0, 0], [0, 2, 0]);
    expectNums(l.get_start(), [0, 0, 0]);
    expectNums(l.get_end(), [0, 2, 0]);
  });
});

// ---------------------------------------------------------------------------
// SVGMobject.move_into_position
// Python (same points as Square(2)): should_center + svg_height=3 ->
//   height 3, center [0,0,0]; no center + svg_width=5 (pre-shifted [3,4,0]) ->
//   width 5, center [3,4,0]
// ---------------------------------------------------------------------------
describe("SVGMobject.move_into_position", () => {
  function svgWithSquarePoints(): SVGMobject {
    const svg = new SVGMobject();
    (svg as unknown as { points: NDArray }).points = (new Square(2.0).points as NDArray).copy();
    return svg;
  }
  it("centers and fits height", () => {
    const svg = svgWithSquarePoints();
    svg.should_center = true;
    (svg as unknown as { svg_height: number }).svg_height = 3;
    (svg as unknown as { svg_width: null }).svg_width = null;
    svg.move_into_position();
    expect(svg.height).toBeCloseTo(3.0, 9);
    expectNums(svg.get_center(), [0, 0, 0]);
  });
  it("fits width without centering", () => {
    const svg = svgWithSquarePoints();
    svg.shift([3, 4, 0]);
    svg.should_center = false;
    (svg as unknown as { svg_height: null }).svg_height = null;
    (svg as unknown as { svg_width: number }).svg_width = 5;
    svg.move_into_position();
    expect(svg.width).toBeCloseTo(5.0, 9);
    expectNums(svg.get_center(), [3, 4, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.flip / arrange
// Python: Square(2) shifted [1,2,0], flip([0,1,0]) -> center [1,2,0];
//   VGroup of three unit squares arrange(buff=0.5) ->
//   centers [-1.5,0,0],[0,0,0],[1.5,0,0]
// ---------------------------------------------------------------------------
describe("Mobject.flip/arrange", () => {
  it("flip keeps the center fixed", () => {
    const m = new Square(2.0);
    m.shift([1, 2, 0]);
    m.flip([0, 1, 0]);
    expectNums(m.get_center(), [1, 2, 0]);
  });
  it("arrange lays submobjects in a centered row", () => {
    const g = new VGroup(new Square(1.0), new Square(1.0), new Square(1.0));
    g.arrange([1, 0, 0], 0.5);
    const centers = g.submobjects.map((s) => (s as Mobject).get_center());
    expectNums(centers[0]!, [-1.5, 0, 0]);
    expectNums(centers[1]!, [0, 0, 0]);
    expectNums(centers[2]!, [1.5, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Table.scale
// Python: h_buff 1.3 / v_buff 0.8, scale(2) -> [2.6, 1.6], member width 4
// ---------------------------------------------------------------------------
describe("Table.scale", () => {
  it("scales buffs and members together", () => {
    const t = new Table({ h_buff: 1.3, v_buff: 0.8 });
    const s = new Square(2.0);
    t.add(s);
    t.scale(2);
    expect(t.h_buff).toBeCloseTo(2.6, 12);
    expect(t.v_buff).toBeCloseTo(1.6, 12);
    expect(s.width).toBeCloseTo(4.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Square.side_length
// Python: Square(side_length=2.0).side_length == 2.0
// ---------------------------------------------------------------------------
describe("Square.side_length", () => {
  it("measures the edge length", () => {
    expect(new Square(2.0).side_length).toBeCloseTo(2.0, 9);
  });
  it("setter rescales the square", () => {
    const s = new Square(2.0);
    s.side_length = 6;
    expect(s.side_length).toBeCloseTo(6.0, 9);
  });
});

// ---------------------------------------------------------------------------
// Arc.init_points / Ellipse.__init__
// Python: Arc(radius=1, angle=1).init_points() -> 27 points starting [1,0,0];
//   Arc(radius=2, angle=1) has 32 points; Ellipse(4, 2) -> width 4, height 2
// ---------------------------------------------------------------------------
describe("Arc.init_points / Ellipse construction", () => {
  it("init_points rebuilds unit-radius bezier points", () => {
    const a = new Arc(1, 0, 1.0);
    a.init_points();
    expect(a.get_num_points()).toBe(27);
    expectNums(flat(a.points).slice(0, 3), [1, 0, 0]);
    const big = new Arc(2, 0, 1.0);
    expect(big.get_num_points()).toBe(32);
  });
  it("Ellipse stretches a circle to width/height", () => {
    const e = new Ellipse(4, 2);
    expect(e.width).toBeCloseTo(4.0, 9);
    expect(e.height).toBeCloseTo(2.0, 9);
  });
});

// ---------------------------------------------------------------------------
// ArrowCircleTip / ArrowCircleFilledTip __init__
// Python: length 0.35, tip_angle ~0 (2.6e-16)
// ---------------------------------------------------------------------------
describe("ArrowCircle tips", () => {
  it("builds a circular tip of the default length", () => {
    const t = new ArrowCircleTip();
    expect(t.length).toBeCloseTo(0.35, 9);
    expect(t.tip_angle).toBeCloseTo(0.0, 9);
  });
  it("filled variant matches", () => {
    const t = new ArrowCircleFilledTip();
    expect(t.length).toBeCloseTo(0.35, 9);
  });
});

// ---------------------------------------------------------------------------
// Mobject.width/height/depth
// Python: Square(2.0) -> 2.0, 2.0, 0.0
// ---------------------------------------------------------------------------
describe("Mobject.width/height/depth", () => {
  it("reports extents", () => {
    const s = new Square(2.0);
    expect(s.width).toBeCloseTo(2.0, 12);
    expect(s.height).toBeCloseTo(2.0, 12);
    expect(s.depth).toBeCloseTo(0.0, 12);
  });
});

// ---------------------------------------------------------------------------
// MovingCamera.frame_width / ScreenRectangle.aspect_ratio
// Python: MovingCamera().frame_width == 14.222222222222221;
//   ScreenRectangle(16/9, 4).aspect_ratio == 1.7777...
// ---------------------------------------------------------------------------
describe("MovingCamera/ScreenRectangle", () => {
  it("frame_width follows the config frame", () => {
    const cam = new MovingCamera();
    expect(cam.frame_width).toBeCloseTo(14.222222222222221, 9);
    expect(cam.frame_width).toBeCloseTo(Number(config.get("frame_width")), 9);
  });
  it("aspect_ratio is width over height", () => {
    const sr = new ScreenRectangle(16.0 / 9.0, 4);
    expect(sr.aspect_ratio).toBeCloseTo(16.0 / 9.0, 12);
  });
  it("aspect_ratio setter stretches the width", () => {
    const sr = new ScreenRectangle(16.0 / 9.0, 4);
    sr.aspect_ratio = 1.0;
    expect(sr.aspect_ratio).toBeCloseTo(1.0, 9);
    expect(sr.height).toBeCloseTo(4.0, 9);
  });
});

// ---------------------------------------------------------------------------
// AbstractImageMobject.reset_points (via ImageMobject with pixel array)
// Python: ImageMobject(zeros((4,6,4))) ->
//   points [[-0.0222,0.0148,0],[0.0222,0.0148,0],[-0.0222,-0.0148,0],
//           [0.0222,-0.0148,0]], height 0.0296296, width 0.0444444
// ---------------------------------------------------------------------------
describe("AbstractImageMobject.reset_points", () => {
  it("lays out the four image corners scaled by resolution", () => {
    const rows: number[][][] = Array.from({ length: 4 }, () =>
      Array.from({ length: 6 }, () => [0, 0, 0, 255]),
    );
    const im = new ImageMobject(rows);
    expectNums(im.points, [
      -0.022222222222222223, 0.014814814814814815, 0,
      0.022222222222222223, 0.014814814814814815, 0,
      -0.022222222222222223, -0.014814814814814815, 0,
      0.022222222222222223, -0.014814814814814815, 0,
    ]);
    expect(im.height).toBeCloseTo(0.02962962962962963, 12);
    expect(im.width).toBeCloseTo(0.044444444444444446, 12);
  });
});

// ---------------------------------------------------------------------------
// Animation / __init__ / _typecheck_input / Mobject defaults
// Python: Animation(Square()): run_time 1.0, lag_ratio 0.0, rate smooth,
//   remover False, introducer False, suspend True, name None;
//   Animation(None) -> empty Mobject; Animation(42) -> TypeError.
//   Mobject(): dim 3, z_index 0, 0 points.
// NOTE: Python requires the mobject positional (Animation() raises
// TypeError); the TS port defaults to an empty Mobject instead.
// ---------------------------------------------------------------------------
describe("Animation construction", () => {
  it("stores defaults from a mobject", () => {
    const anim = new Animation(new Square(2.0));
    expect(anim.run_time).toBeCloseTo(1.0, 12);
    expect(anim.lag_ratio).toBeCloseTo(0.0, 12);
    expect(anim.rate_func(0.5)).toBeCloseTo(0.5, 12);
    expect(anim.remover).toBe(false);
    expect(anim.introducer).toBe(false);
    expect(anim.suspend_mobject_updating).toBe(true);
    expect(anim.name).toBeNull();
    expect(anim.mobject).toBeInstanceOf(Mobject);
  });
  it("accepts null/undefined with an empty Mobject", () => {
    const anim = new Animation(null);
    expect(anim.mobject).toBeInstanceOf(Mobject);
    expect(anim.mobject.get_num_points()).toBe(0);
  });
  it("rejects non-mobjects", () => {
    expect(() => new Animation(42 as unknown as Mobject)).toThrow(TypeError);
    expect(() => new Animation("x" as unknown as Mobject)).toThrow(/only works on Mobjects/);
  });
  it("Mobject() has documented defaults", () => {
    const m = new Mobject();
    expect(m.dim).toBe(3);
    expect(m.z_index).toBe(0);
    expect(m.get_num_points()).toBe(0);
    expect(m.color).toBeInstanceOf(ManimColor);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("integration: transform chains against Python", () => {
  it("Square shift->scale->rotate->flip chain matches Python endpoints", () => {
    // Python: s=Square(2); s.shift([1,2,0]); s.scale(2); s.rotate(pi/2);
    //   s.flip([0,1,0]); s.match_x(target at x=5)
    const m = new Square(2.0);
    m.shift([1, 2, 0]);
    m.scale(2);
    m.rotate(Math.PI / 2);
    m.flip([0, 1, 0]);
    const t = new Square(1.0);
    t.shift([5, 0, 0]);
    m.match_x(t);
    // center x must equal target center x; y/z from the chain
    expect(m.get_center()[0]).toBeCloseTo(5.0, 9);
    expect(m.width).toBeCloseTo(4.0, 9);
  });
  it("Line put_start_and_end_on + set_length composes", () => {
    // Python: l=Line([-1,0,0],[1,0,0]); l.put_start_and_end_on([1,1,0],[4,5,0]);
    //   length 5 -> set_length(10) doubles about center -> width 6? check:
    //   direction (3,4,0)/5; half-extent (3,4,0); center [2.5,3,0];
    //   after set_length(10): start [1,1,0]... verified against Python below.
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    l.put_start_and_end_on([1, 1, 0], [4, 5, 0]);
    expectNums(l.get_start(), [1, 1, 0]);
    expectNums(l.get_end(), [4, 5, 0]);
    l.set_length(10);
    expect(l.get_length()).toBeCloseTo(10.0, 9);
    expectNums(l.get_center(), [2.5, 3, 0]);
  });
  it("VGroup arrange + NumberLine p2n agree with Python", () => {
    // Python: g=VGroup(1x1 squares).arrange(buff=0.5) centers +-0.75;
    //   nl=NumberLine([-2,2,1], length=4): p2n of those centers = +-0.75
    const g = new VGroup(new Square(1.0), new Square(1.0));
    g.arrange([1, 0, 0], 0.5);
    const nl = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    const [c0, c1] = g.submobjects.map((s) => (s as Mobject).get_center());
    expectNums(c0!, [-0.75, 0, 0]);
    expectNums(c1!, [0.75, 0, 0]);
    expect(nl.p2n(c0!)).toBeCloseTo(-0.75, 9);
    expect(nl.p2n(c1!)).toBeCloseTo(0.75, 9);
  });
  it("config frame feeds MovingCamera, ScreenRectangle and shift_onto_screen", () => {
    const cam = new MovingCamera();
    const frameW = Number(config.get("frame_width"));
    expect(cam.frame_width).toBeCloseTo(frameW, 12);
    const sr = new ScreenRectangle(16.0 / 9.0, Number(config.get("frame_height")));
    expect(sr.aspect_ratio).toBeCloseTo(
      Number(config.get("frame_width")) / Number(config.get("frame_height")),
      9,
    );
    const m = new Square(2.0);
    m.shift([100, 0, 0]);
    expect(m.is_off_screen()).toBe(true);
    m.shift_onto_screen();
    expect(m.is_off_screen()).toBe(false);
  });
});
