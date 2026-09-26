// Ground-truth tests for manim.mobject.types.vectorized_mobject.VMobject.
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): VMobject.color_using_background_image /
// match_background_image / get_background_image only store an opaque image
// handle (a string in the TS port); no tex rendering is exercised.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { VMobject, VGroup, VDict } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { CapStyleType } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-6): void {
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

// ---------------------------------------------------------------------------
// VMobject.__init__ defaults
// Python: fill_opacity 0.0, stroke_opacity 1.0, stroke_width 4,
//   background BLACK opacity 1.0 width 0, sheen 0.0 dir UL [-1,1,0],
//   tol 1e-6, nppcc 4; fill [[1,1,1,0]] stroke [[1,1,1,1]] bg [[0,0,0,1]];
//   fill color #FFFFFF op 0.0; stroke #FFFFFF op 1.0 w 4.0; get_color #FFFFFF
// ---------------------------------------------------------------------------
describe("VMobject defaults", () => {
  it("exposes documented constructor defaults and initial rgbas", () => {
    const v = new VMobject();
    expectClose(v.fill_opacity, 0.0);
    expectClose(v.stroke_opacity, 1.0);
    expectClose(v.stroke_width, 4);
    expect(String(v.background_stroke_color)).toBe("#000000");
    expectClose(v.background_stroke_opacity, 1.0);
    expectClose(v.background_stroke_width, 0);
    expectClose(v.sheen_factor, 0.0);
    expectRowsClose([[...v.sheen_direction]], [[-1.0, 1.0, 0.0]]);
    expectClose(v.tolerance_for_point_equality, 1e-6);
    expect(v.n_points_per_cubic_curve).toBe(4);
    expectRowsClose(rowsOf(v.fill_rgbas, 4), [[1, 1, 1, 0]]);
    expectRowsClose(rowsOf(v.stroke_rgbas, 4), [[1, 1, 1, 1]]);
    expectRowsClose(rowsOf(v.background_stroke_rgbas, 4), [[0, 0, 0, 1]]);
    expect(String(v.get_fill_color())).toBe("#FFFFFF");
    expectClose(v.get_fill_opacity(), 0.0);
    expect(String(v.get_stroke_color())).toBe("#FFFFFF");
    expectClose(v.get_stroke_opacity(), 1.0);
    expectClose(v.get_stroke_width(), 4.0);
    expect(String(v.get_color())).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// generate_rgbas_array / update_rgbas_array
// Python: gen(RED, 0.5) -> [[0.9882352941176471, 0.3843137254901961,
//   0.3333333333333333, 0.5]]; with sheen -0.5 -> 2 rows, light row
//   [0.4882352941176471, 0.0, 0.0, 0.5];
//   gen([RED,BLUE],[0.5,0.8]) -> the two rows above (second
//   [0.34509803921568627, 0.7686274509803922, 0.8666666666666667, 0.8]);
//   update_rgbas_array("fill_rgbas", None, 0.2) keeps rgb, sets alpha 0.2.
// ---------------------------------------------------------------------------
describe("VMobject rgbas generation", () => {
  it("generates a single rgba row", () => {
    const v = new VMobject();
    expectRowsClose(rowsOf(v.generate_rgbas_array("red" as never, 0.5), 4), [
      [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5],
    ], 1e-6);
  });

  it("appends a sheen-lightened row when sheen_factor != 0", () => {
    const v = new VMobject();
    v.set_sheen(-0.5, [-1, 1, 0]);
    const out = rowsOf(v.generate_rgbas_array("red" as never, 0.5), 4);
    expect(out.length).toBe(2);
    expectRowsClose(out, [
      [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5],
      [0.4882352941176471, 0.0, 0.0, 0.5],
    ], 1e-6);
  });

  it("pairs lists of colors and opacities", () => {
    const v = new VMobject();
    const out = rowsOf(v.generate_rgbas_array(["red", "blue"] as never, [0.5, 0.8] as never), 4);
    expectRowsClose(out, [
      [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5],
      [0.34509803921568627, 0.7686274509803922, 0.8666666666666667, 0.8],
    ], 1e-6);
  });

  it("update_rgbas_array with color=None only touches opacity", () => {
    const u = new VMobject();
    u.set_fill("red", 0.8);
    u.update_rgbas_array("fill_rgbas", null, 0.2);
    expectRowsClose(rowsOf(u.fill_rgbas, 4), [
      [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.2],
    ], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// set_fill / set_stroke / set_cap_style / set_background_stroke
// Python: set_fill(RED, 0.5) -> color #FC6255 op 0.5;
//   set_stroke(BLUE, 7, 0.3) -> #58C4DD 7.0 0.3;
//   background set_stroke(GREEN, 9, 0.8) -> #83C167 9.0 0.8;
//   set_cap_style(ROUND) stores CapStyleType.ROUND.
// ---------------------------------------------------------------------------
describe("VMobject set_fill / set_stroke", () => {
  it("sets fill color and opacity", () => {
    const v = new VMobject();
    v.set_fill("red", 0.5);
    expect(String(v.get_fill_color())).toBe("#FC6255");
    expectClose(v.get_fill_opacity(), 0.5);
  });

  it("sets stroke color/width/opacity", () => {
    const v = new VMobject();
    v.set_stroke("blue", 7, 0.3);
    expect(String(v.get_stroke_color())).toBe("#58C4DD");
    expectClose(v.get_stroke_width(), 7.0);
    expectClose(v.get_stroke_opacity(), 0.3);
  });

  it("sets background stroke via background flag", () => {
    // NOTE deviation: Python's 4th positional arg is `background`, but the TS
    // port uses the 4th slot for `family` when only 4 args are given (see
    // set_stroke docstring); the 5-arg form recovers the Python meaning, so
    // the ground-truth background values are asserted through it.
    const v = new VMobject();
    v.set_stroke("green", 9, 0.8, true, true);
    expect(String(v.get_stroke_color(true))).toBe("#83C167");
    expectClose(v.get_stroke_width(true), 9.0);
    expectClose(v.get_stroke_opacity(true), 0.8);
  });

  it("stores the cap style", () => {
    const v = new VMobject();
    v.set_cap_style(CapStyleType.ROUND);
    expect(v.cap_style).toBe(CapStyleType.ROUND);
  });

  it("set_background_stroke options object mirrors set_stroke background", () => {
    const v = new VMobject();
    v.set_background_stroke({ color: "green", width: 9, opacity: 0.8 });
    expect(String(v.get_stroke_color(true))).toBe("#83C167");
    expectClose(v.get_stroke_width(true), 9.0);
    expectClose(v.get_stroke_opacity(true), 0.8);
  });
});

// ---------------------------------------------------------------------------
// set_style / get_style / match_style / set_color / set_opacity / fade
// Python: get_style keys [background_image, background_stroke_color,
//   background_stroke_opacity, background_stroke_width, fill_color,
//   fill_opacity, sheen_direction, sheen_factor, stroke_color,
//   stroke_opacity, stroke_width], stroke_width 5, sheen 0.0;
//   match_style copies fill #83C167 op 0.2 width 9.0;
//   set_color(RED) -> fill+stroke #FC6255;
//   set_opacity(0.5) -> fill/stroke/bg opacities 0.5; fade(0.5) halves to 0.5.
// ---------------------------------------------------------------------------
describe("VMobject style api", () => {
  it("get_style exposes the full style dict", () => {
    const st = new VMobject();
    st.set_fill("red", 0.7);
    st.set_stroke("blue", 5, 0.9);
    const gs = st.get_style() as Record<string, unknown>;
    expect(Object.keys(gs).sort()).toEqual([
      "background_image", "background_stroke_color", "background_stroke_opacity",
      "background_stroke_width", "fill_color", "fill_opacity", "sheen_direction",
      "sheen_factor", "stroke_color", "stroke_opacity", "stroke_width",
    ]);
    expectClose(gs["stroke_width"] as number, 5);
    expectClose(gs["sheen_factor"] as number, 0.0);
  });

  it("get_style(simple=True) returns scalar colors", () => {
    // Python: {'stroke_opacity': 1.0, 'stroke_width': 4,
    //   'fill_color': '#FC6255', 'fill_opacity': 0.7, 'stroke_color': '#FFFFFF'}
    const ss = new VMobject();
    ss.set_fill("red", 0.7);
    const s = ss.get_style(true) as Record<string, unknown>;
    expect(String(s["fill_color"])).toBe("#FC6255");
    expectClose(s["fill_opacity"] as number, 0.7);
    expect(String(s["stroke_color"])).toBe("#FFFFFF");
  });

  it("match_style copies the source style", () => {
    const mA = new VMobject();
    const mB = new VMobject();
    mB.set_fill("green", 0.2);
    mB.set_stroke("yellow", 9, 0.4);
    mA.match_style(mB);
    expect(String(mA.get_fill_color())).toBe("#83C167");
    expectClose(mA.get_fill_opacity(), 0.2);
    expectClose(mA.get_stroke_width(), 9.0);
  });

  it("set_color paints fill and stroke", () => {
    const sc = new VMobject();
    sc.set_color("red");
    expect(String(sc.get_fill_color())).toBe("#FC6255");
    expect(String(sc.get_stroke_color())).toBe("#FC6255");
  });

  it("set_opacity floors fill/stroke/background opacities", () => {
    const so = new VMobject();
    so.set_fill("red", 1.0);
    so.set_stroke("blue", 4, 1.0);
    so.set_opacity(0.5);
    expectClose(so.get_fill_opacity(), 0.5);
    expectClose(so.get_stroke_opacity(), 0.5);
    expectClose(so.get_stroke_opacity(true), 0.5);
  });

  it("fade halves opacities", () => {
    const fd = new VMobject();
    fd.set_fill("red", 1.0);
    fd.set_stroke("blue", 4, 1.0);
    fd.fade(0.5);
    expectClose(fd.get_fill_opacity(), 0.5);
    expectClose(fd.get_stroke_opacity(), 0.5);
  });

  it("scale with scale_stroke scales the stroke width", () => {
    // Python: Square(2) stroke 10, scale(0.5, scale_stroke=True) -> 5.0
    const sw = new Square(2.0);
    sw.set_stroke(undefined, 10);
    sw.scale(0.5, true);
    expectClose(sw.get_stroke_width(), 5.0);
  });
});

// ---------------------------------------------------------------------------
// fill/stroke getters incl. zero-rgba -> None
// Python: zeroed fill_rgbas -> get_fill_colors() [None], opacities [0.0]
// ---------------------------------------------------------------------------
describe("VMobject color getters", () => {
  it("returns None fill color for all-zero rgbas", () => {
    const z = new VMobject();
    z.set_points([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2]]);
    (z.fill_rgbas as NDArray) = (z.fill_rgbas as unknown as { copy: () => NDArray }).copy();
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    z.fill_rgbas = array([[0, 0, 0, 0]]) as NDArray;
    expect(z.get_fill_colors()).toEqual([null]);
    expectRowsClose(rowsOf(z.get_fill_opacities() as NDArray, 1).map((r) => [r[0] ?? 0]), [[0.0]]);
  });

  it("get_color prefers fill unless all fill opacities are zero", () => {
    const v = new VMobject();
    v.set_fill("red", 0.0);
    v.set_stroke("blue", 4, 1.0);
    expect(String(v.get_color())).toBe(String(v.get_stroke_color()));
    v.set_fill("red", 0.5);
    expect(String(v.get_color())).toBe(String(v.get_fill_color()));
  });
});

// ---------------------------------------------------------------------------
// sheen api + gradient points
// Python: set_sheen(-0.3, [1,0,0]) -> factor -0.3 dir [1,0,0];
//   rotate_sheen_direction(PI, OUT) -> [-1.0, 1.2e-16, 0.0];
//   Square(2) gradient -> [[1,-1,0],[-1,1,0]];
//   shade_in_3d Square gradient -> [[1,1,0],[-1,-0.333...,0]].
// ---------------------------------------------------------------------------
describe("VMobject sheen", () => {
  it("sets factor and direction", () => {
    const sh = new VMobject();
    sh.set_sheen(-0.3, [1, 0, 0]);
    expectClose(sh.get_sheen_factor(), -0.3);
    expectRowsClose([sh.get_sheen_direction()], [[1, 0, 0]]);
  });

  it("rotates the sheen direction", () => {
    const sh = new VMobject();
    sh.set_sheen(-0.3, [1, 0, 0]);
    sh.rotate_sheen_direction(Math.PI, [0, 0, 1]);
    expectRowsClose([sh.get_sheen_direction()], [[-1.0, 1.2246467991473532e-16, 0.0]]);
  });

  it("computes 2D gradient start/end from the sheen direction", () => {
    const g = new Square(2.0);
    const [s, e] = g.get_gradient_start_and_end_points();
    expectRowsClose([s, e], [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]);
  });

  it("computes 3D gradient corners when shade_in_3d", () => {
    // Python: Square shade_in_3d gradient -> [[1,1,0],[-1,-0.3333,0]]
    const sh3 = new Square();
    sh3.set_shade_in_3d(true);
    const [s, e] = sh3.get_gradient_start_and_end_points();
    expectRowsClose([s, e], [[1.0, 1.0, 0.0], [-1.0, -0.33333333333333326, 0.0]], 1e-6);
  });

  it("set_shade_in_3d flags the family", () => {
    const sv = new VMobject();
    sv.set_shade_in_3d(true);
    expect(sv.shade_in_3d).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// point storage: set_points / resize_points / append_points / clear_points
// Python: 4 pts -> 1 curve; anchors/handles split as
//   [[[0,0,0]],[[1,0,0]],[[1,1,0]],[[2,2,2]]]; start [[0,0,0]] end [[2,2,2]];
//   anchors [[0,0,0],[2,2,2]]; nth(0) the 4 rows; arc length 3.740746952058444;
//   pieces [0.30117..., 9 values]; resize 4->8 keeps prefix; append adds row.
// ---------------------------------------------------------------------------
describe("VMobject point storage", () => {
  function fourPoint(): VMobject {
    const vp = new VMobject();
    vp.set_points([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2]]);
    return vp;
  }

  it("splits anchors and handles", () => {
    const vp = fourPoint();
    expect(vp.get_num_curves()).toBe(1);
    expect(vp.get_num_points()).toBe(4);
    const [a1, h1, h2, a2] = vp.get_anchors_and_handles();
    expectRowsClose(a1, [[0, 0, 0]]);
    expectRowsClose(h1, [[1, 0, 0]]);
    expectRowsClose(h2, [[1, 1, 0]]);
    expectRowsClose(a2, [[2, 2, 2]]);
    expectRowsClose(vp.get_start_anchors(), [[0, 0, 0]]);
    expectRowsClose(vp.get_end_anchors(), [[2, 2, 2]]);
    expectRowsClose(vp.get_anchors(), [[0, 0, 0], [2, 2, 2]]);
    expectRowsClose(rowsOf(vp.get_nth_curve_points(0), 3), [
      [0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2],
    ]);
  });

  it("measures arc length via sampled pieces", () => {
    const vp = fourPoint();
    expectClose(vp.get_arc_length(), 3.740746952058444, 1e-6);
    // NOTE: get_nth_curve_length_pieces returns a 1D array of shape (9,);
    // flatten it before comparing element-wise.
    const flat = (vp.get_nth_curve_length_pieces(0).toArray() as unknown[]).map(Number);
    expect(flat.length).toBe(9);
    const expectedPieces = [0.3011715961075224, 0.2626017470542191, 0.2610494078543821,
      0.2908740254662105, 0.34400929290545473, 0.4159433339421653,
      0.5058883740124869, 0.6148649447117623, 0.7443442300042402];
    for (let i = 0; i < expectedPieces.length; i += 1) {
      expectClose(flat[i] ?? NaN, expectedPieces[i] ?? NaN, 1e-6);
    }
    expectClose(vp.get_nth_curve_length(0), 3.740746952058444, 1e-6);
  });

  it("set_anchors_and_handles interleaves the four arrays", () => {
    // Python: 2 anchors -> 8 points interleaved a1/h1/h2/a2
    const v = new VMobject();
    v.set_anchors_and_handles(
      [[0, 0, 0], [1, 0, 0]], [[0.3, 0, 0], [1.3, 0, 0]],
      [[0.6, 0, 0], [1.6, 0, 0]], [[1, 0, 0], [2, 0, 0]],
    );
    expectRowsClose(rowsOf(v.points, 3), [
      [0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1, 0, 0],
      [1, 0, 0], [1.3, 0, 0], [1.6, 0, 0], [2, 0, 0],
    ]);
  });

  it("rejects mismatched anchor/handle lengths", () => {
    expect(() =>
      new VMobject().set_anchors_and_handles([[0, 0, 0]], [[0, 0, 0], [1, 1, 1]], [[0, 0, 0]], [[0, 0, 0]]),
    ).toThrowError();
  });

  it("resize_points stretches the buffer; same length is a no-op", () => {
    const rz = new VMobject();
    rz.set_points([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    const before = rowsOf(rz.points, 3);
    rz.resize_points(4);
    expectRowsClose(rowsOf(rz.points, 3), before);
    rz.resize_points(8);
    expect(rz.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(rz.points, 3).slice(0, 2), [[0, 0, 0], [1, 0, 0]]);
  });

  it("append_points concatenates rows", () => {
    const az = new VMobject();
    az.set_points([[0, 0, 0]]);
    az.append_points([[1, 1, 1]]);
    expectRowsClose(rowsOf(az.points, 3), [[0, 0, 0], [1, 1, 1]]);
  });

  it("single-point mobjects report one anchor and a last point", () => {
    // Python: 1 pt -> 0 curves, has_new_path True, last [1,2,3]
    const op = new VMobject();
    op.set_points([[1, 2, 3]]);
    expect(op.get_num_curves()).toBe(0);
    expect(op.has_new_path_started()).toBe(true);
    expectRowsClose(rowsOf(op.get_last_point() as NDArray, 3), [[1.0, 2.0, 3.0]]);
    expectRowsClose(op.get_anchors(), [[7 - 6, 8 - 6, 9 - 6]].map(() => [1, 2, 3]));
  });
});

// ---------------------------------------------------------------------------
// path building: start_new_path / add_cubic_bezier_curve[_to] / add_line_to
// Python: start_new_path([5,5,0]) -> 1 pt, has_new_path True, last [5,5,0];
//   add_cubic_bezier_curve_to appends 3 handles after the stored last point;
//   add_line_to([1,0,0]) from origin -> [[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0]];
//   open-path start_new_path closes with repeated last anchor:
//   [[0,0,0],[1,0,0]] + start [5,5,0] ->
//   [[0,0,0],[1,0,0],[0,0,0],[0,0,0],[5,5,0]].
// ---------------------------------------------------------------------------
describe("VMobject path building", () => {
  it("starts a new path with a single point", () => {
    const vn = new VMobject();
    vn.start_new_path([5, 5, 0]);
    expect(vn.has_new_path_started()).toBe(true);
    expectRowsClose(rowsOf(vn.points, 3), [[5, 5, 0]]);
    expectRowsClose(rowsOf(vn.get_last_point() as NDArray, 3), [[5, 5, 0]]);
  });

  it("closes an open path when starting a new one", () => {
    const v4 = new VMobject();
    v4.set_points([[0, 0, 0], [1, 0, 0]]);
    v4.start_new_path([5, 5, 0]);
    expectRowsClose(rowsOf(v4.points, 3), [
      [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [5, 5, 0],
    ]);
  });

  it("appends a full cubic curve", () => {
    const v2 = new VMobject();
    v2.add_cubic_bezier_curve([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]);
    expectRowsClose(rowsOf(v2.points, 3), [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
  });

  it("adds a cubic segment continuing from the last point", () => {
    const vn = new VMobject();
    vn.start_new_path([5, 5, 0]);
    vn.add_cubic_bezier_curve_to([6, 5, 0], [7, 5, 0], [8, 5, 0]);
    expect(vn.has_new_path_started()).toBe(false);
    expectRowsClose(rowsOf(vn.points, 3), [[5, 5, 0], [6, 5, 0], [7, 5, 0], [8, 5, 0]]);
  });

  it("adds a straight line as a cubic with third-point handles", () => {
    const vl = new VMobject();
    vl.start_new_path([0, 0, 0]);
    vl.add_line_to([1, 0, 0]);
    expect(vl.get_num_curves()).toBe(1);
    expectRowsClose(rowsOf(vl.points, 3), [
      [0, 0, 0], [0.3333333333333333, 0, 0], [0.6666666666666666, 0, 0], [1, 0, 0],
    ]);
  });

  it("elevates a quadratic curve to cubic", () => {
    // Python: quad([1,1,0],[2,0,0]) from origin ->
    //   [[0,0,0],[2/3,2/3,0],[4/3,2/3,0],[2,0,0]]
    const vq = new VMobject();
    vq.start_new_path([0, 0, 0]);
    vq.add_quadratic_bezier_curve_to([1, 1, 0], [2, 0, 0]);
    expectRowsClose(rowsOf(vq.points, 3), [
      [0, 0, 0], [0.6666666666666666, 0.6666666666666666, 0],
      [1.3333333333333333, 0.6666666666666666, 0], [2, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// add_smooth_curve_to (1-pt, 2-pt, new-path, arity error)
// Python: Line+([2,0,0]) appends [[1,0,0],[4/3,0,0],[5/3,0,0],[2,0,0]];
//   Line+([1.5,0.5,0],[2,0,0]) appends [[1,0,0],[4/3,0,0],[1.5,0.5,0],[2,0,0]];
//   new-path start + ([1,1,0]) -> line thirds; 0 args -> ValueError.
// ---------------------------------------------------------------------------
describe("VMobject add_smooth_curve_to", () => {
  it("infers the handle for a single anchor", () => {
    const s1 = new Line([0, 0, 0], [1, 0, 0]);
    s1.add_smooth_curve_to([2, 0, 0]);
    expectRowsClose(rowsOf(s1.points, 3).slice(-4), [
      [1, 0, 0], [1.3333333333333335, 0, 0], [1.6666666666666665, 0, 0], [2, 0, 0],
    ]);
  });

  it("uses the given handle for two points", () => {
    const s2 = new Line([0, 0, 0], [1, 0, 0]);
    s2.add_smooth_curve_to([1.5, 0.5, 0], [2, 0, 0]);
    expectRowsClose(rowsOf(s2.points, 3).slice(-4), [
      [1, 0, 0], [1.3333333333333335, 0, 0], [1.5, 0.5, 0], [2, 0, 0],
    ]);
  });

  it("falls back to a line on a fresh path", () => {
    const s3 = new VMobject();
    s3.start_new_path([0, 0, 0]);
    s3.add_smooth_curve_to([1, 1, 0]);
    expectRowsClose(rowsOf(s3.points, 3), [
      [0, 0, 0], [0.3333333333333333, 0.3333333333333333, 0],
      [0.6666666666666666, 0.6666666666666666, 0], [1, 1, 0],
    ]);
  });

  it("rejects arities other than 1 or 2", () => {
    expect(() => new VMobject().add_smooth_curve_to()).toThrowError();
    expect(() => new VMobject().add_smooth_curve_to([0, 0, 0], [1, 1, 0], [2, 2, 0])).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// corners: set_points_as_corners / add_points_as_corners / set_points_smoothly
// Python: corners [[0,0,0],[1,0,0],[1,1,0]] -> 8 pts (thirds per segment);
//   add_points_as_corners from origin with [[1,0,0],[1,1,0]] -> same 8 pts;
//   empty corners array is a no-op (Line keeps 4 pts);
//   set_points_smoothly first rows [[0,0,0],[1/3,-1/12,0]].
// ---------------------------------------------------------------------------
describe("VMobject corners", () => {
  const expected = [
    [0, 0, 0], [0.3333333333333333, 0, 0], [0.6666666666666666, 0, 0], [1, 0, 0],
    [1, 0, 0], [1, 0.3333333333333333, 0], [1, 0.6666666666666666, 0], [1, 1, 0],
  ];

  it("sets corners as third-point beziers", () => {
    const vc = new VMobject();
    vc.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expectRowsClose(rowsOf(vc.points, 3), expected);
  });

  it("appends corners after the current end point", () => {
    const ap = new VMobject();
    ap.start_new_path([0, 0, 0]);
    ap.add_points_as_corners([[1, 0, 0], [1, 1, 0]]);
    expectRowsClose(rowsOf(ap.points, 3), expected);
  });

  it("ignores an empty corners array", () => {
    const e = new Line([0, 0, 0], [1, 0, 0]);
    e.add_points_as_corners([]);
    expect(e.get_num_points()).toBe(4);
  });

  it("set_points_smoothly smooths the corner path", () => {
    // Python: smoothly([[0,0,0],[1,0,0],[2,1,0]])[:2] ->
    //   [[0,0,0],[1/3,-1/12,0]]
    const sm = new VMobject();
    sm.set_points_smoothly([[0, 0, 0], [1, 0, 0], [2, 1, 0]]);
    expectRowsClose(rowsOf(sm.points, 3).slice(0, 2), [
      [0, 0, 0], [0.33333333333333337, -0.08333333333333333, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// change_anchor_mode / make_smooth / make_jagged
// Python: smooth 3-corner path ->
//   [[0,0,0],[1/3,-1/12,0],[2/3,-1/6,0],[1,0,0],
//    [1,0,0],[4/3,1/6,0],[5/3,7/12,0],[2,1,0]];
//   jagged restores thirds; bad mode -> AssertionError.
// ---------------------------------------------------------------------------
describe("VMobject anchor modes", () => {
  it("smooths then jags handles", () => {
    const vs = new VMobject();
    vs.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 1, 0]]);
    vs.make_smooth();
    expectRowsClose(rowsOf(vs.points, 3), [
      [0, 0, 0], [0.33333333333333337, -0.08333333333333333, 0],
      [0.6666666666666667, -0.16666666666666666, 0], [1, 0, 0],
      [1, 0, 0], [1.3333333333333333, 0.16666666666666666, 0],
      [1.6666666666666665, 0.5833333333333334, 0], [2, 1, 0],
    ]);
    vs.make_jagged();
    expectRowsClose(rowsOf(vs.points, 3), [
      [0, 0, 0], [0.3333333333333333, 0, 0], [0.6666666666666666, 0, 0], [1, 0, 0],
      [1, 0, 0], [1.3333333333333335, 0.3333333333333333, 0],
      [1.6666666666666665, 0.6666666666666666, 0], [2, 1, 0],
    ]);
  });

  it("rejects unknown modes", () => {
    expect(() => new VMobject().change_anchor_mode("bad" as never)).toThrowError();
  });

  it("smooths submobject families in place", () => {
    const f = new VMobject();
    const ch = new VMobject();
    ch.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    f.add(ch);
    f.make_smooth();
    expectRowsClose(rowsOf(ch.points, 3).slice(0, 2), [
      [0, 0, 0], [0.41666666666666674, -0.08333333333333333, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// subpaths: get_subpaths / get_subpaths_from_points / split indices
// Python: continuous 2-curve corners -> 1 subpath; jump at [1,0,0]->[5,5,0]
//   -> 2 subpaths; get_subpath_split_indices 8-pt jump -> [[0,4],[4,8]].
// ---------------------------------------------------------------------------
describe("VMobject subpaths", () => {
  it("keeps continuous curves in one subpath", () => {
    const vc = new VMobject();
    vc.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(vc.get_subpaths().length).toBe(1);
  });

  it("splits subpaths where anchors jump", () => {
    const sj = new VMobject();
    sj.set_points([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [5, 5, 0], [5, 5, 0], [5, 5, 0], [6, 5, 0]]);
    const subs = sj.get_subpaths();
    expect(subs.length).toBe(2);
    expectRowsClose(rowsOf(subs[0]!, 3).slice(0, 1), [[0, 0, 0]]);
    expectRowsClose(rowsOf(subs[1]!, 3).slice(0, 1), [[5, 5, 0]]);
  });

  it("reports split index ranges", () => {
    const v = new VMobject();
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    const pts = array([
      [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0],
      [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 2, 0],
    ]) as NDArray;
    expectRowsClose(rowsOf(v.get_subpath_split_indices_from_points(pts), 2), [[0, 4], [4, 8]]);
  });

  it("trims remainders when extracting bezier tuples", () => {
    // Python: 3 points (remainder 3) -> []
    const g = new VMobject();
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    const out = g.get_cubic_bezier_tuples_from_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) as NDArray);
    expect(rowsOf(out as NDArray, 3).length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// curve queries: nth function / lengths / arc length / get_anchors edge
// Python: Line([0,0,0],[3,4,0]) pieces are 9 x 5/9, length 5.0;
//   1 curve function; nth(0)(0.5) on unit segment [0.5,0,0];
//   Line tuples [[[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0]]].
// ---------------------------------------------------------------------------
describe("VMobject curve queries", () => {
  it("samples the nth curve function", () => {
    const sp = new VMobject();
    sp.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(sp.get_nth_curve_function(0)(0.5)).toBeDefined();
    expectRowsClose([sp.get_nth_curve_function(0)(0.5)], [[0.5, 0, 0]]);
  });

  it("approximates the 3-4-5 line length", () => {
    const nc = new Line([0, 0, 0], [3, 4, 0]);
    expectClose(nc.get_nth_curve_length(0), 5.0);
    expectClose(nc.get_nth_curve_function_with_length(0)[1], 5.0);
    expect(nc.get_nth_curve_length_pieces(0).shape[0]).toBe(9);
  });

  it("lists curve functions with lengths", () => {
    const nc = new Line([0, 0, 0], [3, 4, 0]);
    expect([...nc.get_curve_functions()].length).toBe(1);
    expect(nc.get_curve_functions_with_lengths().length).toBe(1);
    expectClose(nc.get_arc_length(), 5.0);
  });

  it("returns cubic tuples for a line", () => {
    const t = new Line([0, 0, 0], [1, 0, 0]).get_cubic_bezier_tuples();
    const flat = t as unknown as number[][][];
    expectRowsClose(flat[0]!, [[0, 0, 0], [0.3333333333333333, 0, 0], [0.6666666666666666, 0, 0], [1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// point_from_proportion / proportion_from_point (+ errors)
// Python: Line([0,0,0],[2,0,0]) pfp .5 -> [1,0,0], 0 -> start, 1 -> end,
//   arc length 2.0, proportion([1,0,0]) 0.5; alpha 2 -> ValueError;
//   off-curve point -> ValueError.
// ---------------------------------------------------------------------------
describe("VMobject proportion api", () => {
  it("maps proportions to points on a line", () => {
    const line = new Line([0, 0, 0], [2, 0, 0]);
    expectRowsClose(rowsOf(line.point_from_proportion(0.5), 3), [[1, 0, 0]]);
    expectRowsClose(rowsOf(line.point_from_proportion(0), 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(line.point_from_proportion(1), 3), [[2, 0, 0]]);
    expectClose(line.get_arc_length(), 2.0);
  });

  it("inverts a point on the curve", () => {
    const line = new Line([0, 0, 0], [2, 0, 0]);
    expectClose(line.proportion_from_point([1, 0, 0]), 0.5);
  });

  it("rejects out-of-range alpha and off-curve points", () => {
    expect(() => new Line([0, 0, 0], [1, 0, 0]).point_from_proportion(2)).toThrowError();
    expect(() => new Line([0, 0, 0], [1, 0, 0]).proportion_from_point([5, 5, 5])).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// is_closed / close_path / get_direction / reverse / force
// Python: open 3-corner path not closed; close -> closed, 12 pts;
//   closed 4-corner loop is_closed True; Circle/Square CCW;
//   Square reverse -> CW; force same keeps CCW; force "XX" -> ValueError.
// ---------------------------------------------------------------------------
describe("VMobject closedness and direction", () => {
  it("closes an open corner path", () => {
    const cp = new VMobject();
    cp.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(cp.is_closed()).toBe(false);
    cp.close_path();
    expect(cp.is_closed()).toBe(true);
    expect(cp.get_num_points()).toBe(12);
  });

  it("detects a loop closed by corners", () => {
    const vc2 = new VMobject();
    vc2.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]);
    expect(vc2.is_closed()).toBe(true);
  });

  it("reports CCW for Circle and Square", () => {
    expect(new Circle().get_direction()).toBe("CCW");
    expect(new Square().get_direction()).toBe("CCW");
  });

  it("reverses and forces direction", () => {
    const sq2 = new Square();
    const d0 = sq2.get_direction();
    sq2.reverse_direction();
    expect(d0).toBe("CCW");
    expect(sq2.get_direction()).toBe("CW");
    const fdc = new Circle();
    expect(fdc.force_direction(fdc.get_direction()).get_direction()).toBe("CCW");
    expect(() => fdc.force_direction("XX" as never)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// consider_points_equals[_2d]
// Python: ([0,0,0],[1e-7,0,0]) True, ([0,0,0],[1,0,0]) False;
//   2d ([0,0],[0,0]) True, ([0,0],[1,0]) False.
// ---------------------------------------------------------------------------
describe("VMobject point equality", () => {
  it("compares 3D points within tolerance", () => {
    const vp = new VMobject();
    expect(vp.consider_points_equals([0, 0, 0], [1e-7, 0, 0])).toBe(true);
    expect(vp.consider_points_equals([0, 0, 0], [1, 0, 0])).toBe(false);
  });

  it("compares 2D points", () => {
    const vp = new VMobject();
    expect(vp.consider_points_equals_2d([0, 0], [0, 0])).toBe(true);
    expect(vp.consider_points_equals_2d([0, 0], [1, 0])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// insert_n_curves / align_points / align_rgbas / append_vectorized_mobject
// Python: Line 1 curve +1 -> 2 curves 8 pts; Line/Circle align -> 32/32;
//   single-point insert(2) -> 8 repeated rows; append drops a pending
//   new-path point: start([0,0,0]) + 1-curve corners -> 4 pts;
//   align_rgbas stretches the shorter fill array to length 2.
// ---------------------------------------------------------------------------
describe("VMobject alignment", () => {
  it("inserts curves by subdivision", () => {
    const ins = new Line([0, 0, 0], [1, 0, 0]);
    expect(ins.get_num_curves()).toBe(1);
    ins.insert_n_curves(1);
    expect(ins.get_num_curves()).toBe(2);
    expect(ins.get_num_points()).toBe(8);
  });

  it("repeats a single point for insert_n_curves_to_point_list", () => {
    // Python: insert(2, [[9,9,9]]) -> 8 repeated rows
    const g = new VMobject();
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    const out = g.insert_n_curves_to_point_list(2, array([[9, 9, 9]]) as NDArray);
    expectRowsClose(rowsOf(out, 3), Array.from({ length: 8 }, () => [9, 9, 9]));
  });

  it("aligns a line with a circle", () => {
    const a1 = new Line([0, 0, 0], [1, 0, 0]);
    const a2 = new Circle();
    a1.align_points(a2);
    expect(a1.get_num_points()).toBe(32);
    expect(a2.get_num_points()).toBe(32);
  });

  it("appends another vmobject, dropping a pending new-path point", () => {
    const h1 = new VMobject();
    h1.start_new_path([0, 0, 0]);
    const h2 = new VMobject();
    h2.set_points_as_corners([[1, 1, 0], [2, 2, 0]]);
    h1.append_vectorized_mobject(h2);
    expectRowsClose(rowsOf(h1.points, 3), [
      [1, 1, 0], [1.3333333333333335, 1.3333333333333335, 0],
      [1.6666666666666665, 1.6666666666666665, 0], [2, 2, 0],
    ]);
  });

  it("align_rgbas stretches the shorter array", () => {
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    const a1 = new VMobject();
    const a2 = new VMobject();
    a1.fill_rgbas = array([[1, 0, 0, 1], [1, 0, 0, 1]]) as NDArray;
    a2.fill_rgbas = array([[0, 1, 0, 0.5]]) as NDArray;
    a1.align_rgbas(a2);
    expect(a1.fill_rgbas.shape[0]).toBe(2);
    expect(a2.fill_rgbas.shape[0]).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// pointwise_become_partial / get_subcurve (incl. self + closed wraparound)
// Python: Circle partial(0.25,0.75) -> 20 pts, first [6.12e-17,1,0];
//   self-partial same; closed get_subcurve(0.75,0.25) -> 20 pts;
//   non-VMobject -> TypeError; add_subpath non-multiple-of-4 -> AssertionError.
// ---------------------------------------------------------------------------
describe("VMobject partial curves", () => {
  it("extracts the middle of a circle", () => {
    const src = new Circle(1.0);
    const dst = new Circle(1.0);
    dst.pointwise_become_partial(src, 0.25, 0.75);
    expect(dst.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(dst.points, 3).slice(0, 1), [[6.123233995736766e-17, 1.0, 0.0]]);
  });

  it("supports in-place self partials", () => {
    const sc = new Circle(1.0);
    sc.pointwise_become_partial(sc, 0.25, 0.75);
    expect(sc.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(sc.points, 3).slice(0, 1), [[6.123233995736766e-17, 1.0, 0.0]]);
  });

  it("wraps closed subcurves with a > b", () => {
    const cc = new Circle(1.0);
    expect(cc.get_subcurve(0.75, 0.25).get_num_points()).toBe(20);
    const sub = new Circle(1.0).get_subcurve(0.25, 0.75);
    expect(sub.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(sub.points, 3).slice(0, 1), [[6.123233995736766e-17, 1.0, 0.0]]);
  });

  it("rejects non-VMobjects and bad subpaths", () => {
    expect(() => new Circle(1.0).pointwise_become_partial(new Mobject() as unknown as VMobject, 0, 1)).toThrow(TypeError);
    const { array } = require("numpy-ts") as typeof import("numpy-ts");
    expect(() => new VMobject().add_subpath(array([[0, 0, 0], [1, 0, 0]]) as NDArray)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// interpolate_color / get_point_mobject / scale_handle / apply / rotate
// Python: RED->GREEN etc alpha .5 -> fill
//   [0.7509803921568627,0.5705882352941176,0.3686274509803922,0.5],
//   stroke_width 6.0; get_point_mobject([1,2,3]) keeps #FC6255 fill;
//   scale_handle halves [1/3,0,0]->[1/6,0,0];
//   apply(+[0,1,0]) shifts; rotate PI/2 about origin maps [1,0,0]->[0,1,0].
// ---------------------------------------------------------------------------
describe("VMobject color interpolation and transforms", () => {
  it("blends fill rgbas and stroke width", () => {
    const m1 = new VMobject();
    m1.set_fill("red", 1.0);
    m1.set_stroke("blue", 4, 1.0);
    const m2 = new VMobject();
    m2.set_fill("green", 0.0);
    m2.set_stroke("yellow", 8, 0.0);
    const t = new VMobject();
    t.interpolate_color(m1, m2, 0.5);
    expectRowsClose(rowsOf(t.fill_rgbas, 4), [
      [0.7509803921568627, 0.5705882352941176, 0.3686274509803922, 0.5],
    ], 1e-6);
    expectClose(t.stroke_width, 6.0);
  });

  it("builds a styled point mobject", () => {
    const pm = new Square();
    pm.set_fill("red", 0.9);
    const q = pm.get_point_mobject([1, 2, 3]);
    expectRowsClose(rowsOf(q.points, 3), [[1, 2, 3]]);
    expect(String(q.get_fill_color())).toBe("#FC6255");
  });

  it("scales handle-to-anchor distances", () => {
    const shd = new Line([0, 0, 0], [1, 0, 0]);
    shd.scale_handle_to_anchor_distances(0.5);
    expectRowsClose(rowsOf(shd.points, 3).slice(1, 2), [[0.16666666666666666, 0, 0]]);
  });

  it("applies a point function", () => {
    const af = new Line([0, 0, 0], [1, 0, 0]);
    af.apply_function((p) => [p[0]! + 0, p[1]! + 1, p[2]! + 0]);
    expectRowsClose(rowsOf(af.points, 3).slice(0, 1), [[0, 1, 0]]);
    expectRowsClose(rowsOf(af.points, 3).slice(-1), [[1, 1, 0]]);
  });

  it("rotates about a point", () => {
    const ro = new Line([1, 0, 0], [2, 0, 0]);
    ro.rotate(Math.PI / 2, [0, 0, 1], [0, 0, 0]);
    expectRowsClose(rowsOf(ro.points, 3).slice(0, 1), [[2.220446049250313e-16, 1.0, 0.0]], 1e-6);
    expectRowsClose(rowsOf(ro.points, 3).slice(-1), [[4.440892098500626e-16, 2.0, 0.0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// split / nonempty_submobjects / boundary / background image handle
// Python: parent with 1 pointful + 1 empty child -> nonempty 1;
//   empty VMobject split -> []; Square boundary[:2] [[1,1,0],[-1,1,0]];
//   color_using_background_image stores "img.png", fill #FFFFFF,
//   match_background_image copies the handle.
// ---------------------------------------------------------------------------
describe("VMobject structure helpers", () => {
  it("lists nonempty submobjects and splits", () => {
    const parent = new VMobject();
    const ch1 = new VMobject();
    ch1.set_points([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2]]);
    const ch2 = new VMobject();
    parent.add(ch1, ch2);
    expect(parent.nonempty_submobjects().length).toBe(1);
    expect(new VMobject().split().length).toBe(0);
    expect(ch1.split()[0]).toBe(ch1);
  });

  it("collects boundary anchors across the family", () => {
    expectRowsClose(rowsOf(new Square().get_points_defining_boundary(), 3).slice(0, 2), [
      [1, 1, 0], [-1, 1, 0],
    ]);
  });

  it("stores (opaque) background image handles without tex", () => {
    const bv = new VMobject();
    bv.color_using_background_image("img.png");
    expect(bv.get_background_image()).toBe("img.png");
    expect(String(bv.get_fill_color())).toBe("#FFFFFF");
    const bv2 = new VMobject();
    bv2.match_background_image(bv);
    expect(bv2.get_background_image()).toBe("img.png");
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: VMobject chain with geometry + config-fed rendering inputs
// ---------------------------------------------------------------------------
describe("VMobject integration", () => {
  it("builds a styled square, dashes it via subcurves, and fades it", () => {
    // Python: Square side 2 has 4 corners closed (12 pts after close path
    // semantics); here mirror the probe: corners + close -> 12 pts, CCW,
    // then fade halves opacities (matches the fade unit probe).
    const sq = new Square(2.0);
    expect(sq.get_direction()).toBe("CCW");
    sq.set_fill("red", 1.0);
    sq.set_stroke("blue", 4, 1.0);
    const mid = sq.point_from_proportion(0.5);
    expect(rowsOf(mid as NDArray, 3).length).toBe(1);
    const sub = sq.get_subcurve(0.0, 0.5);
    expect(sub.get_num_points()).toBeGreaterThan(0);
    sq.fade(0.5);
    expectClose(sq.get_fill_opacity(), 0.5);
  });

  it("aligns a Line and Square then interpolates their colors", () => {
    // Python: align_points equalizes point counts (Line 4 -> Square count);
    // interpolate_color then blends widths (4 & 8 at .5 -> 6.0).
    const l = new Line([0, 0, 0], [1, 0, 0]);
    const s = new Square();
    l.set_stroke(undefined, 4);
    s.set_stroke(undefined, 8);
    l.align_points(s);
    expect(l.get_num_points()).toBe(s.get_num_points());
    const t = new Line([0, 0, 0], [1, 0, 0]);
    t.align_points(s);
    t.interpolate_color(l, s, 0.5);
    expectClose(t.stroke_width, 6.0);
  });

  it("drives proportion round-trip through arc length", () => {
    // Python: Line([0,0,0],[2,0,0]) arc 2.0, pfp(.25)=[0.5,0,0],
    // proportion back .25.
    const line = new Line([0, 0, 0], [2, 0, 0]);
    expectClose(line.get_arc_length(), 2.0);
    const p = rowsOf(line.point_from_proportion(0.25), 3)[0]!;
    expectRowsClose([p], [[0.5, 0, 0]]);
    expectClose(line.proportion_from_point(p), 0.25);
  });
});

// ---------------------------------------------------------------------------
// VGroup dunders / VDict dunders (same module, verified against Python 0.21.0)
// Python: repr(VGroup(Square(), Square())) == "VGroup(Square, Square)";
//   str == "VGroup of 2 submobjects"; g + Square() -> new VGroup of 3;
//   g - a -> 1; g += Square() -> 3 in place; g -= a -> 1, a removed;
//   g[0] = new -> identity; g[0] is new; 'sq' in VDict True, 'xx' False.
// ---------------------------------------------------------------------------
describe("VGroup and VDict dunders", () => {
  it("renders repr and str", () => {
    const g = new VGroup(new Square(), new Square());
    expect(g.__repr__()).toBe("VGroup(Square, Square)");
    expect(g.__str__()).toBe("VGroup of 2 submobjects");
    expect(new VGroup(new Square()).__str__()).toBe("VGroup of 1 submobject");
  });

  it("adds and subtracts submobjects", () => {
    const a = new Square();
    const b = new Square();
    const g = new VGroup(a, b);
    const added = g.__add__(new Square());
    expect(added).toBeInstanceOf(VGroup);
    expect(added.submobjects.length).toBe(3);
    expect(g.submobjects.length).toBe(2);
    const sub = g.__sub__(a);
    expect(sub.submobjects.length).toBe(1);
    expect(sub.submobjects[0]).toBe(b);
  });

  it("iadd/isub mutate in place", () => {
    const a = new Square();
    const g2 = new VGroup(a, new Square());
    g2.__iadd__(new Square());
    expect(g2.submobjects.length).toBe(3);
    g2.__isub__(a);
    expect(g2.submobjects.length).toBe(2);
    expect(g2.submobjects.includes(a)).toBe(false);
  });

  it("supports index get/set", () => {
    const g = new VGroup(new VMobject(), new VMobject());
    const fresh = new VMobject();
    g.__setitem__(0, fresh);
    expect(g.submobjects[0]).toBe(fresh);
    expect(g.__getitem__(0)).toBe(fresh);
  });

  it("VDict __getitem__/__contains__ mirror get/has", () => {
    const d = new VDict({ sq: new Square() } as Record<string, VMobject>);
    expect(d.__contains__("sq")).toBe(true);
    expect(d.__contains__("xx")).toBe(false);
    expect(d.__getitem__("sq")).toBe(d.get("sq"));
  });
});
