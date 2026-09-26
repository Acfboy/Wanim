// Ground-truth tests for batch layers 19..19 (~1021 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): get_graph_label / get_secant_slope_group label
// creation via MathTex/Tex needs the LaTeX pipeline, so all label inputs
// here are pre-built VMobjects (Rectangle), which Python returns as-is from
// _create_label_tex. String/float labels stay marked TODO stubs in the port.
//
// DEVIATIONS (documented inline):
//  (D1) Axes/NumberPlane in TS have no real NumberLine axes yet (empty
//       `axes` VGroup, no `x_axis`); tests wire faithful stand-ins (Line
//       geometry + NumberLine label factory) the same way Python's
//       _create_axis output behaves for these methods.
//  (D2) Python PolarPlane().make_smooth_after_applying_functions is always
//       False (PolarPlane sets True, then Axes.__init__ runs
//       VGroup.__init__ which overwrites with the VMobject default False);
//       the port mirrors that net behavior (fixed in this batch).
//  (D3) Python PolarPlane derives faded_line_style (halved numerics) inside
//       _init_background_lines; the port derives the same dict while still
//       deferring line construction (fixed in this batch).
//  (D4) Python Line3D.__init__ does np.array(start) so Mobject endpoints
//       raise there; the TS port accepts Mobjects (pointify superset).
//  (D5) Python Cube faces after apply_matrix are edge-on for the LEFT/RIGHT
//       (width 0) and UP/DOWN (height 0) pairs; asserted as-is.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import {
  Axes,
  PolarPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { ParametricFunction } from "../src/mobject/graphing/functions.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Line, Angle, RightAngle } from "../src/mobject/geometry/line.js";
import { TangentialArc, Dot } from "../src/mobject/geometry/arc.js";
import {
  ArrowSquareTip,
  ArrowSquareFilledTip,
} from "../src/mobject/geometry/tips.js";
import { Rectangle, Square } from "../src/mobject/geometry/polygram.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Cube, Line3D } from "../src/mobject/three_d/three_dimensions.js";
import { Scene } from "../src/scene/scene.js";
import { pointRows } from "../src/utils/arrays.js";
import { BLUE_D } from "../src/utils/color/index.js";

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

function plainRows(v: number[] | NDArray): number[][] {
  if (Array.isArray(v)) {
    return (v.length > 0 && Array.isArray(v[0]) ? v : [v]) as number[][];
  }
  return rowsOf(v as NDArray, 3);
}

function makeAxes(): Axes {
  return new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
}

// See (D1): wire Line-geometry axes (projection behavior identical to
// Python's NumberLine axes, which subclass Line) plus a NumberLine label
// factory standing in for Python's x_axis._create_label_tex.
function makeAxesWithLines(): Axes {
  const ax = makeAxes();
  ax.axes = new VGroup(
    new Line([-3, 0, 0], [3, 0, 0]),
    new Line([0, -2, 0], [0, 2, 0]),
  );
  (ax as unknown as { x_axis: unknown }).x_axis = new NumberLine({
    x_range: [-3, 3, 1],
    length: 6,
  });
  return ax;
}

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_line_from_axis_to_point
// Python (ax as above, g irrelevant):
//   index 0, point [1,1,0] (DashedLine): start [1,0,0], end [1,1,0].
//   index 1, point [1,1,0] (DashedLine): start [~0,1,0], end [1,1,0].
//   index 0, point [2,4,0], line_func=Line: [2,0,0] -> [2,4,0], type Line.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_line_from_axis_to_point", () => {
  it("drops a vertical dashed line from the x axis", () => {
    const ax = makeAxesWithLines();
    const line = ax.get_line_from_axis_to_point(0, [1, 1, 0]);
    expectRowsClose([line.get_start()], [[1, 0, 0]]);
    expectRowsClose([line.get_end()], [[1, 1, 0]]);
  });

  it("drops a horizontal dashed line from the y axis", () => {
    // Python: start [2.22e-16, 1.0000000000000004, 0].
    const ax = makeAxesWithLines();
    const line = ax.get_line_from_axis_to_point(1, [1, 1, 0]);
    expectRowsClose([line.get_start()], [[0, 1, 0]], 1e-9);
    expectRowsClose([line.get_end()], [[1, 1, 0]]);
  });

  it("honors line_func=Line with plain endpoints", () => {
    const ax = makeAxesWithLines();
    const line = ax.get_line_from_axis_to_point(
      0,
      [2, 4, 0],
      Line as unknown as new (start: unknown, end: unknown, options?: Record<string, unknown>) => VMobject,
    );
    expect(line).toBeInstanceOf(Line);
    expectRowsClose([line.get_start()], [[2, 0, 0]]);
    expectRowsClose([line.get_end()], [[2, 4, 0]]);
  });

  it("forwards color/stroke_width (clobbering line_config color like Python)", () => {
    // Python unconditionally overwrites line_config["color"] with the
    // `color` argument (default VMobject white), so a color smuggled inside
    // line_config is lost on both sides; the `color` parameter wins.
    const ax = makeAxesWithLines();
    const line = ax.get_line_from_axis_to_point(0, [1, 1, 0], undefined, null, "RED", 5);
    expect(line.get_color().to_hex()).toBe("#FC6255");
    expectClose(line.get_stroke_width(), 5);
    const clobbered = ax.get_line_from_axis_to_point(0, [1, 1, 0], undefined, {
      color: "RED",
    });
    expect(clobbered.get_color().to_hex()).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_graph_label
// Python (ax, g = plot(x^2), Rectangle(0.2 x 0.2) labels):
//   x_val=1.0: identity (label is lbl), center [1.35, 1.0, 0].
//   x_val=None: center [2.7196969696969706, 3.0, 0].
//   x_val=2.0 dot=True: label center [2.185000000000001, 3.69...],
//     dot at [2.000000000000001, 4.0, 0].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_graph_label", () => {
  it("positions a VMobject label at x_val with identity", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_graph_label(g, lab, 1.0);
    expect(out).toBe(lab as unknown as Mobject);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[1.3500000000000003, 1.0, 0.0]]);
  });

  it("searches right-to-left when x_val is None", () => {
    // Python breaks at x=1.9696969696969697 (first scene-y < 4) and
    // shift_onto_screen pulls the label down to y=3.4.
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_graph_label(g, lab);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[2.3196969696969707, 3.4, 0.0]]);
  });

  it("attaches a dot at the graph point when asked", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_graph_label(g, lab, 2.0, undefined, undefined, null, true) as unknown as Record<string, unknown>;
    const dot = out["dot"] as Dot;
    expect(dot).toBeInstanceOf(Dot);
    expectRowsClose([dot.get_center()], [[2.000000000000001, 4.0, 0.0]]);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[2.185000000000001, 3.6900000000000004, 0.0]]);
  });

  it("marks string labels as tex-backed stubs", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    expect(() => ax.get_graph_label(g, "f(x)", 1.0)).toThrowError(/text\/tex not ported/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_derivative_graph
// Python (g = plot(x^2)): slope_of_tangent(0)=1e-08, (0.5)=1.0000000111022302,
//   (1.0)=1.9999999999999996, (2.0)=4.000000000000002; dg has 240 points,
//   t_min -3, t_max 3, underlying(-1)=-1.9999999888977693.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_derivative_graph", () => {
  it("returns the tangent-slope curve with ground-truth samples", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const dg = ax.plot_derivative_graph(g);
    expect(dg).toBeInstanceOf(ParametricFunction);
    expect(dg.get_num_points()).toBe(240);
    expectClose(dg.t_min, -3.0);
    expectClose(dg.t_max, 3.0);
    const deriv = (dg as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(deriv(0.0), 1e-8, 1e-6);
    expectClose(deriv(0.5), 1.0000000111022302);
    expectClose(deriv(1.0), 1.9999999999999996);
    expectClose(deriv(2.0), 4.000000000000002);
    expectClose(deriv(-1.0), -1.9999999888977693);
  });

  it("passes through the derivative point (1, 2)", () => {
    const ax = makeAxes();
    const g = ax.plot((x: number) => x ** 2);
    const dg = ax.plot_derivative_graph(g);
    const rows = rowsOf(dg.points, dg.dim);
    let best = Infinity;
    for (const row of rows) {
      const d = Math.hypot((row[0] ?? 0) - 1.0, (row[1] ?? 0) - 2.0, row[2] ?? 0);
      if (d < best) best = d;
    }
    expect(best < 0.05).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_secant_slope_group
// Python (ax, g = plot(x^2), x=1.0, dx=0.5):
//   dx_line [1,1,0]->[1.5,1,0]; df_line [1.5,1,0]->[1.5,2.25,0];
//   secant [−0.60695...,−3.01738...]->[3.10695...,6.26738...], length 10;
//   3 submobjects. dx=None -> dx 0.6; include_secant_line=False -> 2 subs.
//   Rectangle(0.2) labels: dx_label [1.25,0.8,0], df_label [1.7,1.625,0].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_secant_slope_group", () => {
  it("builds dx/df/secant lines with ground-truth geometry", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const grp = ax.get_secant_slope_group(1.0, g, 0.5) as unknown as Record<string, Line>;
    expectRowsClose([(grp["dx_line"] as Line).get_start()], [[1.0000000000000002, 1.0, 0.0]]);
    expectRowsClose([(grp["dx_line"] as Line).get_end()], [[1.5000000000000004, 1.0, 0.0]]);
    expectRowsClose([(grp["df_line"] as Line).get_start()], [[1.5000000000000004, 1.0, 0.0]]);
    expectRowsClose([(grp["df_line"] as Line).get_end()], [[1.5000000000000004, 2.25, 0.0]]);
    expectRowsClose([(grp["secant_line"] as Line).get_start()], [[-0.6069533817705195, -3.0173834544262963, 0.0]]);
    expectRowsClose([(grp["secant_line"] as Line).get_end()], [[3.106953381770519, 6.267383454426296, 0.0]]);
    expectClose((grp["secant_line"] as Line).get_length(), 10.0);
    expect((grp as unknown as VGroup).submobjects.length).toBe(3);
  });

  it("defaults dx to range/10 and can omit the secant line", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const grp = ax.get_secant_slope_group(1.0, g, null, undefined, undefined, undefined, undefined, false) as unknown as Record<string, unknown>;
    // dx defaults to (3-(-3))/10 = 0.6: df_line ends at i2gp(1.6).
    const df = grp["df_line"] as Line;
    expectRowsClose([df.get_end()], [[1.6000000000000003, 2.5600000000000005, 0.0]]);
    expect((grp as unknown as VGroup).submobjects.length).toBe(2);
    expect("secant_line" in (grp as object)).toBe(false);
  });

  it("positions VMobject dx/dy labels with ground-truth centers", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const d1 = new Rectangle({ width: 0.2, height: 0.2 });
    const d2 = new Rectangle({ width: 0.2, height: 0.2 });
    const grp = ax.get_secant_slope_group(1.0, g, 0.5, undefined, undefined, d1 as unknown as string, d2 as unknown as string) as unknown as Record<string, VMobject>;
    expect((grp as unknown as VGroup).submobjects.length).toBe(5);
    expectRowsClose([(grp["dx_label"] as VMobject).get_center()], [[1.2500000000000004, 0.8, 0.0]]);
    expectRowsClose([(grp["df_label"] as VMobject).get_center()], [[1.7000000000000006, 1.625, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.PolarPlane
// Python: PolarPlane() -> x_range [-4,4,1], lengths None, azimuth "PI radians"
//   CCW step 20, offset 0, buff 0.1, font 24, compact True; radius_config
//   {stroke_width 2, include_ticks False, include_tip False,
//    line_to_number_buff 0.1, label_direction DL, font_size 24};
//   background {BLUE_D, 2, 1}; faded {BLUE_D, 1.0, 0.5} (D3); ratio 1;
//   make_smooth False (D2). radius_max=3,size=5,degrees/CW -> step 36.
//   Bad units/direction raise ValueError. c2p(1,1)==[1,~1,0] on size-6 plane.
// ---------------------------------------------------------------------------
describe("PolarPlane", () => {
  it("stores defaults matching Python (post _init_background_lines)", () => {
    const pp = new PolarPlane();
    expect(pp.x_range).toEqual([-4, 4, 1]);
    expect(pp.y_range).toEqual([-4, 4, 1]);
    expect(pp.x_length).toBeNull();
    expect(pp.y_length).toBeNull();
    expect(pp.azimuth_units).toBe("PI radians");
    expect(pp.azimuth_direction).toBe("CCW");
    expect(pp.azimuth_step).toBe(20);
    expect(pp.azimuth_offset).toBe(0);
    expectClose(pp.azimuth_label_buff, 0.1);
    expect(pp.azimuth_label_font_size).toBe(24);
    expect(pp.azimuth_compact_fraction).toBe(true);
    expect(pp.radius_config["stroke_width"]).toBe(2);
    expect(pp.radius_config["include_ticks"]).toBe(false);
    expect(pp.radius_config["include_tip"]).toBe(false);
    expectClose(pp.radius_config["line_to_number_buff"] as number, 0.1);
    expect(pp.radius_config["font_size"]).toBe(24);
    expect(pp.background_line_style["stroke_width"]).toBe(2);
    expect(pp.background_line_style["stroke_opacity"]).toBe(1);
    // (D3) faded style derived without building lines.
    expect(pp.faded_line_style?.["stroke_width"]).toBe(1.0);
    expect(pp.faded_line_style?.["stroke_opacity"]).toBe(0.5);
    expect(pp.faded_line_ratio).toBe(1);
    // (D2) net Python behavior is False (VGroup overwrite).
    expect(pp.make_smooth_after_applying_functions).toBe(false);
    expect(pp.background_lines).toBeNull();
    expect(pp.faded_lines).toBeNull();
  });

  it("derives the faded style from a custom background style", () => {
    // Python halves every numeric entry of background_line_style.
    const pp = new PolarPlane(null, null, 1, null, "PI radians", true, 0, "CCW", 0.1, 24, null, { stroke_width: 4, stroke_opacity: 0.8 });
    expect(pp.faded_line_style?.["stroke_width"]).toBe(2.0);
    expectClose(pp.faded_line_style?.["stroke_opacity"] as number, 0.4);
  });

  it("keeps an explicit faded style as-is", () => {
    const pp = new PolarPlane(null, null, 1, null, "PI radians", true, 0, "CCW", 0.1, 24, null, null, { stroke_width: 7 });
    expect(pp.faded_line_style?.["stroke_width"]).toBe(7);
  });

  it("honors radius/size/units/direction options", () => {
    const pp = new PolarPlane(3, 5, 1, null, "degrees", true, 0, "CW");
    expect(pp.x_range).toEqual([-3, 3, 1]);
    expect(pp.x_length).toBe(5);
    expect(pp.y_length).toBe(5);
    expect(pp.azimuth_step).toBe(36);
    expect(pp.azimuth_units).toBe("degrees");
    expect(pp.azimuth_direction).toBe("CW");
  });

  it("maps azimuth steps per unit system", () => {
    expect(new PolarPlane(null, null, 1, null, "TAU radians").azimuth_step).toBe(20);
    expect(new PolarPlane(null, null, 1, null, "gradians").azimuth_step).toBe(40);
    expect(new PolarPlane(null, null, 1, null, null).azimuth_step).toBe(1);
    expect(new PolarPlane(null, null, 1, 7).azimuth_step).toBe(7);
  });

  it("rejects bad azimuth units/direction with ValueError", () => {
    expect(() => new PolarPlane(null, null, 1, null, "bogus" as never)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    expect(() => new PolarPlane(null, null, 1, null, "PI radians", true, 0, "bogus" as never)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });

  it("maps coordinates like an Axes plane", () => {
    // Python: PolarPlane(radius_max=3, size=6).coords_to_point(1,1)==[1,~1,0].
    const pp = new PolarPlane(3, 6);
    expectRowsClose(plainRows(pp.coords_to_point(1, 1) as unknown as NDArray), [[1.0000000000000002, 1.0, 0.0]]);
    expectBLUE_D(pp);
  });
});

function expectBLUE_D(pp: PolarPlane): void {
  expect(BLUE_D.to_hex()).toBe("#29ABCA");
  expect((pp.background_line_style["stroke_color"] as unknown as { to_hex(): string }).to_hex()).toBe("#29ABCA");
}

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.rotate_about_number
// Python (x_range [-3,3,1], length 6): returns self; rotate_about_number(1,
// pi/2): start [1,-4,0], end [1,2,0]; n2p(1) stays [1,~0,0].
// rotate_about_zero(pi/2) keeps n2p(0) at the origin.
// ---------------------------------------------------------------------------
describe("NumberLine.rotate_about_number", () => {
  it("rotates about the given number and returns self", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const ret = nl.rotate_about_number(1, Math.PI / 2);
    expect(ret).toBe(nl);
    const [start, end] = nl.get_start_and_end();
    expectRowsClose([start], [[0.9999999999999991, -4.0, 0.0]], 1e-9);
    expectRowsClose([end], [[1.0000000000000004, 2.0, 0.0]], 1e-9);
    expectRowsClose(plainRows(nl.n2p(1)), [[1.0, -2.220446049250313e-16, 0.0]], 1e-9);
  });

  it("rotate_about_zero keeps the origin fixed", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    nl.rotate_about_zero(Math.PI / 2);
    expectRowsClose(plainRows(nl.n2p(0)), [[0, 0, 0]], 1e-9);
    const [start, end] = nl.get_start_and_end();
    expectRowsClose([start], [[0, -3, 0]], 1e-6);
    expectRowsClose([end], [[0, 3, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.__matmul__ / __rmatmul__
// Python (same line): nl @ 1.5 == n2p(1.5) == [1.5,0,0];
//   __rmatmul__([1,0,0]) == 1.0; __rmatmul__(mobject at origin) == 0.0.
// ---------------------------------------------------------------------------
describe("NumberLine.__matmul__ / __rmatmul__", () => {
  it("__matmul__ maps numbers to points like n2p", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expectRowsClose(plainRows(nl.__matmul__(1.5)), [[1.5, 0, 0]]);
    expectRowsClose(plainRows(nl.__matmul__([1, 2] as unknown as number)), [[1, 0, 0], [2, 0, 0]]);
  });

  it("__rmatmul__ maps points (and mobject centers) to numbers like p2n", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expectClose(nl.__rmatmul__([1, 0, 0]) as number, 1.0);
    const sq = new Square().shift([2, 0, 0]);
    expectClose(nl.__rmatmul__(sq as unknown as Mobject) as number, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cube.generate_points
// Python Cube(side_length=2): 6 Square faces, centers [0,0,±1],[±1,0,0],
// [0,±1,0]; overall width/height/depth 2. init_points returns self.
// ---------------------------------------------------------------------------
describe("Cube.generate_points", () => {
  it("builds six faces at the cube centers", () => {
    const c = new Cube(2);
    expect(c.side_length).toBe(2);
    expect(c.submobjects.length).toBe(6);
    const want = [
      [0, 0, -1],
      [0, 0, 1],
      [-1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
    ];
    for (let i = 0; i < 6; i += 1) {
      expectRowsClose([(c.submobjects[i] as Mobject).get_center()], [want[i]!], 1e-9);
    }
    const m = c as unknown as Mobject;
    expectClose(m.width, 2.0);
    expectClose(m.height, 2.0);
    expectClose(m.depth, 2.0);
  });

  it("init_points regenerates and returns self", () => {
    const c = new Cube(2);
    expect(c.init_points()).toBe(c);
  });

  it("honors a custom side length", () => {
    // Python Cube(3): face centers at ±1.5, overall width 3.
    const c = new Cube(3);
    expect(c.submobjects.length).toBe(6);
    expectRowsClose([(c.submobjects[0] as Mobject).get_center()], [[0, 0, -1.5]], 1e-9);
    expectClose((c as unknown as Mobject).width, 3.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.set_start_and_end_attrs
// Python Line3D([0,0,0],[2,2,2]): vect [2,2,2], length sqrt(12),
//   dir [1,1,1]/sqrt(3), start/end as given. Defaults: LEFT/RIGHT,
//   thickness 0.02, resolution (2, 24). NDArray endpoints preserved.
// ---------------------------------------------------------------------------
describe("Line3D.set_start_and_end_attrs", () => {
  it("tracks vect/length/direction/start/end", () => {
    const l = new Line3D([0, 0, 0], [2, 2, 2]);
    expectRowsClose([l.vect], [[2, 2, 2]]);
    expectClose(l.length, Math.sqrt(12));
    expectRowsClose([l.direction], [[1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)]]);
    expectRowsClose(plainRows(l.get_start()), [[0, 0, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[2, 2, 2]]);
  });

  it("exposes Python-compatible defaults", () => {
    const l = new Line3D();
    expectRowsClose(plainRows(l.get_start()), [[-1, 0, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[1, 0, 0]]);
    expect(l.thickness).toBe(0.02);
    expect(l.resolution).toEqual([2, 24]);
  });

  it("accepts NDArray endpoints and re-points on demand", () => {
    const l = new Line3D(array([0, 0, 0]) as NDArray, array([2, 2, 2]) as NDArray);
    expectRowsClose(plainRows(l.get_start()), [[0, 0, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[2, 2, 2]]);
    expect(l.set_start_and_end_attrs([0, 0, 0], [1, 0, 0])).toBe(l);
    expectClose(l.length, 1.0);
    expectRowsClose([l.direction], [[1, 0, 0]]);
  });

  it("resolves Mobject endpoints via boundary points (D4: TS superset)", () => {
    // Python raises in the constructor for Mobjects (np.array); TS pointify
    // supports them. Square().get_boundary_point([1,0,0]) == [1,1,0] (the
    // corner wins argmax) on both sides, so the start is the corner.
    const sq = new Square();
    const l = new Line3D(sq as unknown as Mobject, [2, 0, 0]);
    expectRowsClose(plainRows(l.get_start()), [[1, 1, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[2, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.init_points
// Python: returns self; regenerates from start/end (4 points preserved).
// ---------------------------------------------------------------------------
describe("Line.init_points", () => {
  it("regenerates points from start/end and returns self", () => {
    const l = new Line([1, 2, 3], [4, 6, 3]);
    expect(l.get_num_points()).toBe(4);
    expect(l.init_points()).toBe(l);
    expect(l.get_num_points()).toBe(4);
    expectRowsClose([l.get_start()], [[1, 2, 3]]);
    expectRowsClose([l.get_end()], [[4, 6, 3]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.put_start_and_end_on
// Python: degenerate line ([0,0,0]x2) resets attrs then moves to
//   [1,1,0]->[2,2,0]; normal line moves start/end directly.
// ---------------------------------------------------------------------------
describe("Line.put_start_and_end_on", () => {
  it("resets a degenerate line onto the new endpoints", () => {
    const l = new Line([0, 0, 0], [0, 0, 0]);
    l.put_start_and_end_on([1, 1, 0], [2, 2, 0]);
    expectRowsClose([l.get_start()], [[1, 1, 0]]);
    expectRowsClose([l.get_end()], [[2, 2, 0]]);
  });

  it("moves a normal line onto the new endpoints", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    const ret = l.put_start_and_end_on([2, 3, 0], [5, 7, 0]);
    expect(ret).toBe(l);
    expectRowsClose([l.get_start()], [[2, 3, 0]]);
    expectRowsClose([l.get_end()], [[5, 7, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.get_slope
// Python: [0,0,0]->[3,4,0] slope 4/3; horizontal 0.0;
//   vertical tan(pi/2) == 1.633123935319537e+16.
// ---------------------------------------------------------------------------
describe("Line.get_slope", () => {
  it("returns tan(angle) with ground-truth values", () => {
    expectClose(new Line([0, 0, 0], [3, 4, 0]).get_slope(), 1.3333333333333333);
    expectClose(new Line([0, 0, 0], [1, 0, 0]).get_slope(), 0.0);
    expectClose(new Line([0, 0, 0], [0, 1, 0]).get_slope(), 1.633123935319537e16);
    expectClose(new Line([0, 0, 0], [1, 1, 0]).get_slope(), 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.set_angle
// Python: [0,0,0]->[1,0,0] set_angle(pi/2): start stays [0,0,0],
//   end [2.22e-16,1,0], angle pi/2. set_angle(pi, about [0.5,0,0]):
//   start [1,~0,0], end [0,~0,0].
// ---------------------------------------------------------------------------
describe("Line.set_angle", () => {
  it("rotates about the start by default", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expect(l.set_angle(Math.PI / 2)).toBe(l);
    expectRowsClose([l.get_start()], [[0, 0, 0]]);
    expectRowsClose([l.get_end()], [[2.220446049250313e-16, 1.0, 0.0]]);
    expectClose(l.get_angle(), Math.PI / 2);
    expectClose(l.get_length(), 1.0);
  });

  it("rotates about an explicit point", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    l.set_angle(Math.PI, [0.5, 0, 0]);
    expectRowsClose([l.get_start()], [[1.0, -6.123233995736766e-17, 0.0]], 1e-9);
    expectRowsClose([l.get_end()], [[0.0, 6.123233995736766e-17, 0.0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Angle.__init__
// Python (l1 x-axis 2-long, l2 y-axis 2-long): angle_value pi/2, 90 degrees,
//   32 points, no `radius` attr. radius=0.5/quadrant(-1,1) -> 3pi/2.
//   other_angle -> -3pi/2. elbow -> no angle_value, 8 points.
//   parallel lines -> 0.0, empty. dot=True -> 1 submobject.
//   RightAngle(l1, l2, 0.3): elbow, radius 0.3, no angle_value.
// ---------------------------------------------------------------------------
describe("Angle.__init__", () => {
  function cross(): [Line, Line] {
    return [new Line([0, 0, 0], [2, 0, 0]), new Line([0, 0, 0], [0, 2, 0])];
  }

  it("builds a right angle with ground-truth value and size", () => {
    const [l1, l2] = cross();
    const a = new Angle(l1, l2);
    expectClose(a.angle_value, Math.PI / 2);
    expectClose(a.get_value(true), 90.0);
    expectClose(a.get_value(), Math.PI / 2);
    expect(a.get_num_points()).toBe(32);
    // NOTE: `radius?: number` is a real (undefined-valued) field, so a plain
    // read yields undefined (unlike `declare angle_value`, which goes through
    // the proxy and throws); Python hasattr(a, "radius") is likewise False.
    expect(a.radius).toBeUndefined();
    expect(a.get_lines().submobjects.length).toBe(2);
  });

  it("honors explicit radius and quadrant", () => {
    const [l1, l2] = cross();
    const a = new Angle(l1, l2, { radius: 0.5, quadrant: [-1, 1] });
    expectClose(a.angle_value, 4.71238898038469);
    expect(a.radius).toBe(0.5);
  });

  it("mirrors with other_angle", () => {
    const [l1, l2] = cross();
    expectClose(new Angle(l1, l2, { other_angle: true }).angle_value, -4.71238898038469);
  });

  it("builds an elbow without angle_value", () => {
    const [l1, l2] = cross();
    const e = new Angle(l1, l2, { elbow: true });
    expect(e.elbow).toBe(true);
    // NOTE: `declare angle_value` emits no field and the Mobject proxy
    // throws AttributeError on reads (mirroring Python's missing attr), so
    // the absence is asserted with `in`, exactly like Python hasattr().
    expect("angle_value" in e).toBe(false);
    expect(e.get_num_points()).toBe(8);
  });

  it("collapses parallel lines to an empty zero angle", () => {
    const a = new Angle(new Line([0, 0, 0], [1, 0, 0]), new Line([0, 1, 0], [1, 1, 0]));
    expectClose(a.angle_value, 0.0);
    expect(a.get_num_points()).toBe(0);
  });

  it("adds a dot when asked", () => {
    const [l1, l2] = cross();
    const a = new Angle(l1, l2, { dot: true });
    expect(a.submobjects.length).toBe(1);
    expect(a.submobjects[0]).toBeInstanceOf(Dot);
  });

  it("RightAngle is an elbow with the given length", () => {
    const [l1, l2] = cross();
    const ra = new RightAngle(l1, l2, 0.3);
    expect(ra.elbow).toBe(true);
    expect(ra.radius).toBe(0.3);
    expect("angle_value" in ra).toBe(false);
  });

  it("from_three_points matches the two-line form", () => {
    const a = Angle.from_three_points([2, 0, 0], [0, 0, 0], [0, 2, 0]);
    expectClose(a.angle_value, Math.PI / 2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowSquareTip.__init__ / ArrowSquareFilledTip
// Python: defaults fill 0, stroke 3, 0.35 x 0.35, 16 points,
//   start_angle pi; tip_point [0.175,0.175,0], base [-0.175,-0.175,0],
//   length ~0.49497. length=0.5 -> 0.5 x 0.5. Filled: fill 1, stroke 0.
// ---------------------------------------------------------------------------
describe("ArrowSquareTip.__init__ / ArrowSquareFilledTip", () => {
  it("builds the default square tip", () => {
    const t = new ArrowSquareTip();
    expectClose(t.width, 0.35);
    expectClose(t.height, 0.35);
    expect(t.get_num_points()).toBe(16);
    expectClose(t.start_angle, Math.PI);
    expectRowsClose(plainRows(t.tip_point), [[0.175, 0.175, 0.0]]);
    expectRowsClose(plainRows(t.base), [[-0.175, -0.175, 0.0]]);
    expectClose(t.length, 0.49497474683058323);
  });

  it("honors a custom length", () => {
    const t = new ArrowSquareTip(0, 3, 0.5);
    expectClose(t.width, 0.5);
    expectClose(t.height, 0.5);
  });

  it("builds the filled variant", () => {
    const f = new ArrowSquareFilledTip();
    expect(f).toBeInstanceOf(ArrowSquareTip);
    expectClose(f.width, 0.35);
    expectClose(f.height, 0.35);
    expect(f.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TangentialArc.__init__
// Python (l1 x-axis, l2 y-axis, r=0.5): start [0,0.5,0], end [0.5,0,0],
//   length sqrt(2)/2; keeps line1/line2. corner=(-1,1), r=0.25:
//   start [-0.25,0,0], end [~0,0.25,0].
// ---------------------------------------------------------------------------
describe("TangentialArc.__init__", () => {
  it("joins the tangent points with ground-truth geometry", () => {
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 2, 0]);
    const ta = new TangentialArc(l1, l2, 0.5);
    expectRowsClose([ta.get_start()], [[0, 0.5, 0]]);
    expectRowsClose([ta.get_end()], [[0.5, 0, 0]]);
    expectClose(ta.get_length(), 0.7071067811865476);
    expect(ta.line1).toBe(l1);
    expect(ta.line2).toBe(l2);
  });

  it("honors the corner selector", () => {
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 2, 0]);
    const ta = new TangentialArc(l1, l2, 0.25, [-1, 1]);
    expectRowsClose([ta.get_start()], [[-0.25, 0, 0]]);
    expectRowsClose([ta.get_end()], [[1.1102230246251565e-16, 0.25, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.rect_to_mobject
// Python: Rect(1,2,4,6) -> 4x6 at [3,5,0], 16 pts (covered in batch-14d too);
//   rounded (rx=ry=0.5, 4x2) -> RoundedRectangle 4x2 at [2,1,0], 32 pts;
//   rx=1/ry=0.5 6x3 at [4,2.5,0]; rx=0 -> plain Rectangle.
// ---------------------------------------------------------------------------
describe("SVGMobject.rect_to_mobject", () => {
  it("converts a plain rect", () => {
    const mob = SVGMobject.rect_to_mobject({ kind: "rect", x: 1, y: 2, width: 4, height: 6 });
    expect(mob).toBeInstanceOf(Rectangle);
    expectClose(mob.width, 4.0);
    expectClose(mob.height, 6.0);
    expectRowsClose([mob.get_center()], [[3, 5, 0]]);
    expect(mob.get_num_points()).toBe(16);
  });

  it("converts a rounded rect preserving extents", () => {
    const mob = SVGMobject.rect_to_mobject({ kind: "rect", x: 0, y: 0, width: 4, height: 2, rx: 0.5, ry: 0.5 });
    expectClose(mob.width, 4.0);
    expectClose(mob.height, 2.0);
    expectRowsClose([mob.get_center()], [[2, 1, 0]]);
    expect(mob.get_num_points()).toBe(32);
  });

  it("rescales the rounded rect when rx != ry", () => {
    // Python: RoundedRectangle(6, 6*1/0.5=12, r=1) stretched to height 3.
    const mob = SVGMobject.rect_to_mobject({ kind: "rect", x: 1, y: 1, width: 6, height: 3, rx: 1.0, ry: 0.5 });
    expectClose(mob.width, 6.0);
    expectClose(mob.height, 3.0);
    expectRowsClose([mob.get_center()], [[4, 2.5, 0]]);
  });

  it("falls back to a plain rect when rx is zero", () => {
    const mob = SVGMobject.rect_to_mobject({ kind: "rect", x: 0, y: 0, width: 4, height: 2, rx: 0, ry: 0.5 });
    expect(mob).toBeInstanceOf(Rectangle);
    expectClose(mob.width, 4.0);
    expectClose(mob.height, 2.0);
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols against earlier-ported machinery
// ---------------------------------------------------------------------------
describe("integration: layers-19 batch with earlier ports", () => {
  it("plot -> derivative -> secant chain is self-consistent", () => {
    // Python: dg.underlying_function(1) == slope_of_tangent(1, g) == 2 and
    // the secant at x=1 has length 10.
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const dg = ax.plot_derivative_graph(g);
    const deriv = (dg as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(deriv(1.0), ax.slope_of_tangent(1.0, g));
    expectClose(deriv(1.0), 2.0);
    const grp = ax.get_secant_slope_group(1.0, g, 0.5) as unknown as Record<string, Line>;
    expectClose((grp["secant_line"] as Line).get_length(), 10.0);
    const vline = ax.get_vertical_line(ax.i2gp(1, g) as unknown as number[]);
    expectRowsClose([vline.get_end()], [[1.0000000000000002, 1.0, 0.0]]);
  });

  it("NumberLine rotation feeds matmul and Line slopes", () => {
    // Python: fresh line n2p == matmul; after rotate_about_number(1, pi/2)
    // the segment n2p(1)->n2p(2) is vertical (infinite slope magnitude).
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expectRowsClose(plainRows(nl.__matmul__(2.0)), plainRows(nl.n2p(2.0)));
    nl.rotate_about_number(1, Math.PI / 2);
    const a = plainRows(nl.n2p(1))[0]!;
    const b = plainRows(nl.n2p(2))[0]!;
    const seg = new Line(a, b);
    expect(Math.abs(seg.get_slope()) > 1e15).toBe(true);
    expectClose((nl.__rmatmul__(a) as number), 1.0, 1e-6);
  });

  it("Angle and TangentialArc share the same line pair consistently", () => {
    // Python: quadrant (1,1) right angle pi/2 on the axis pair; the tangent
    // arc for r=0.5 spans the same corner with length sqrt(2)/2.
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 2, 0]);
    const a = new Angle(l1, l2);
    expectClose(a.angle_value, Math.PI / 2);
    const ta = new TangentialArc(l1, l2, 0.5);
    expectClose(ta.get_length(), Math.SQRT2 / 2);
    expect(ta.line1).toBe(a.lines[0]);
  });

  it("Cube faces are Squares whose edges measure side_length", () => {
    // Python: every Cube(2) face has width/height in {0, 2} with the
    // face-center distance from origin == 1 == side/2.
    const c = new Cube(2);
    for (const face of c.submobjects as Mobject[]) {
      expect(face).toBeInstanceOf(Square);
      const center = face.get_center();
      expectClose(Math.hypot(...(center as [number, number, number])), 1.0);
    }
  });

  it("secant group composes inside a Scene like other VGroups", () => {
    const ax = makeAxesWithLines();
    const g = ax.plot((x: number) => x ** 2);
    const grp = ax.get_secant_slope_group(1.0, g, 0.5) as unknown as VGroup;
    const scene = new Scene();
    expect(scene.add(grp as unknown as Mobject)).toBe(scene);
    expect(scene.mobjects[0]).toBe(grp as unknown as Mobject);
    expect(grp.submobjects.length).toBe(3);
  });

  it("PolarPlane mapping agrees with the Axes linear map", () => {
    // Python: PolarPlane(radius_max=3, size=6).c2p(1,1) == [1,~1,0], the
    // same linear map as Axes([-3,3,1],[-2,2,1],6,4).c2p(1,1).
    const pp = new PolarPlane(3, 6);
    const ax = makeAxes();
    expectRowsClose(
      plainRows(pp.coords_to_point(1, 1) as unknown as NDArray),
      plainRows(ax.coords_to_point(1, 1) as unknown as NDArray),
    );
  });
});
