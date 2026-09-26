// Ground-truth tests for batch layers 4..4 (~1058 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here; each block cites the
// probe in a "Python:" comment. TS is asserted against those values (1e-9
// relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): nothing in this batch touches text/tex.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import {
  ManimColor,
  RandomColorGenerator,
  rgb_to_color,
  rgb_to_hex,
} from "../src/utils/color/core.js";
import { Transform } from "../src/animation/transform.js";
import { ThreeDScene, SpecialThreeDScene } from "../src/scene/three_d_scene.js";
import { Scene } from "../src/scene/scene.js";
import {
  ThreeDAxes,
  type ShadedAxis,
  type AxisShadingSubmobject,
} from "../src/mobject/graphing/coordinate_systems.js";
import { Surface } from "../src/mobject/three_d/three_dimensions.js";
import {
  LinearTransformationScene,
  X_COLOR,
  Y_COLOR,
} from "../src/scene/vector_space_scene.js";
import { config } from "../src/_config/utils.js";
import { DEGREES } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";

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

// ---------------------------------------------------------------------------
// manim.utils.color.core.ManimColor.__invert__
// Python:
//   for c in ['#FF0000','#123456','#000000','#FFFFFF']: (~c).to_hex()
//   -> ['#00FFFF','#EDCBA9','#FFFFFF','#000000']
//   ManimColor('#FF0000').invert(with_alpha=True).to_hex(True) -> '#00FFFF00'
//   ManimColor('#FF0000').invert().to_hex(True) -> '#00FFFFFF' (alpha kept)
// ---------------------------------------------------------------------------
describe("ManimColor.__invert__", () => {
  it("linearly inverts rgb like Python", () => {
    const cases: Array<[string, string]> = [
      ["#FF0000", "#00FFFF"],
      ["#123456", "#EDCBA9"],
      ["#000000", "#FFFFFF"],
      ["#FFFFFF", "#000000"],
    ];
    for (const [input, expected] of cases) {
      expect(new ManimColor(input).__invert__().to_hex()).toBe(expected);
    }
  });
  it("invert() keeps alpha; invert(with_alpha=True) inverts it", () => {
    // Python: invert() -> '#00FFFFFF'; invert(with_alpha=True) -> '#00FFFF00'.
    expect(new ManimColor("#FF0000").invert().to_hex(true)).toBe("#00FFFFFF");
    expect(new ManimColor("#FF0000").invert(true).to_hex(true)).toBe("#00FFFF00");
    expect(new ManimColor("#FF0000").__invert__().to_hex(true)).toBe("#00FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.{rgb_to_color, rgb_to_hex}
// Python:
//   rgb_to_color([1.0,0.0,0.0]).to_hex() -> '#FF0000' (all-float list: float path)
//   rgb_to_color([1.0,0,0]).to_hex() -> '#010000' (mixed int/float: int path!)
//   rgb_to_color(np.array([255,0,0])).to_hex() -> '#FF0000' (int path)
//   rgb_to_hex([0.0,1.0,0.0]) -> '#00FF00'
//   rgb_to_hex(np.array([0,0,255])) -> '#0000FF'
//   rgb_to_hex([0.5,0.5,0.5]) -> '#7F7F7F' (truncation)
// ---------------------------------------------------------------------------
describe("rgb_to_color / rgb_to_hex", () => {
  it("rgb_to_color handles float and int arrays like Python", () => {
    // Python: rgb_to_color([1.0,0.0,0.0]).to_hex() -> '#FF0000' (float path);
    //   rgb_to_color(np.array([255,0,0])).to_hex() -> '#FF0000' (int path);
    //   rgb_to_color([1.0,0,0]).to_hex() -> '#010000' (mixed list: int path).
    // NOTE (language boundary, also documented on the ManimColor
    // constructor): a plain JS list cannot express "integral float"
    // ([1.0,0,0] is [1,0,0] in JS and takes the int path), so the float
    // probes use float64 NDArrays -- the TS-idiomatic float input whose
    // dtype selects the float path exactly like numpy float scalars do.
    expect(rgb_to_color(array([1.0, 0.0, 0.0]) as NDArray).to_hex()).toBe("#FF0000");
    expect(rgb_to_color(array([255, 0, 0], "int64") as NDArray).to_hex()).toBe("#FF0000");
    expect(rgb_to_color(array([0.0, 0.0, 1.0]) as NDArray).to_hex()).toBe("#0000FF");
    // Python: mixed int/float plain lists take the int path.
    expect(rgb_to_color([1.0, 0, 0]).to_hex()).toBe("#010000");
  });
  it("rgb_to_hex matches Python including truncation", () => {
    // Python: rgb_to_hex([0.0,1.0,0.0]) -> '#00FF00';
    //   rgb_to_hex(np.array([0,0,255])) -> '#0000FF';
    //   rgb_to_hex([0.5,0.5,0.5]) -> '#7F7F7F' (truncation).
    expect(rgb_to_hex(array([0.0, 1.0, 0.0]) as NDArray)).toBe("#00FF00");
    expect(rgb_to_hex(array([0, 0, 255], "uint8") as NDArray)).toBe("#0000FF");
    // Non-integral floats take the float path even as plain lists.
    expect(rgb_to_hex([0.5, 0.5, 0.5])).toBe("#7F7F7F");
    expect(rgb_to_hex(array([0.5, 0.5, 0.5]) as NDArray)).toBe("#7F7F7F");
  });
  it("documents the integral-plain-list boundary (int path)", () => {
    // Python rgb_to_hex([0.0,1.0,0.0]) is '#00FF00', but the JS literal
    // [0,1,0] carries no float type, so the port routes it through the int
    // (1/255) path: [0/255, 1/255, 0/255] -> '#000100'. NDArray float64
    // input (above) is the way to spell Python-float input in TS.
    expect(rgb_to_hex([0, 1, 0])).toBe("#000100");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.RandomColorGenerator._random_color
// Python (seeded): RandomColorGenerator(seed=42).next() x3
//   -> ['#8B4513', '#BBBBBB', '#BBBBBB']; repeat run identical.
// _random_color() uses an unseeded singleton: only palette membership and
// ManimColor type are asserted (nondeterministic by design).
// ---------------------------------------------------------------------------
describe("RandomColorGenerator._random_color", () => {
  it("seeded generator reproduces Python's sequence", () => {
    const g = new RandomColorGenerator(42);
    expect([g.next().to_hex(), g.next().to_hex(), g.next().to_hex()]).toEqual([
      "#8B4513",
      "#BBBBBB",
      "#BBBBBB",
    ]);
    const g2 = new RandomColorGenerator(42);
    expect([g2.next().to_hex(), g2.next().to_hex(), g2.next().to_hex()]).toEqual([
      "#8B4513",
      "#BBBBBB",
      "#BBBBBB",
    ]);
  });
  it("_random_color returns a palette ManimColor", () => {
    const c = RandomColorGenerator._random_color();
    expect(c).toBeInstanceOf(ManimColor);
    const palette = new Set(
      new RandomColorGenerator().colors.map((m) => m.to_hex()),
    );
    expect(palette.has(c.to_hex())).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.set_color
// Python: VMobject with corners, set_color('#FF0000')
//   fill_rgbas [[1,0,0,0]] (untouched default fill opacity 0),
//   stroke_rgbas [[1,0,0,1]]; get_fill_color/get_stroke_color -> '#FF0000'.
//   family=False recolors only the parent (sub keeps '#FFFFFF').
// ---------------------------------------------------------------------------
describe("VMobject.set_color", () => {
  it("sets fill+stroke to the color, preserving opacities", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(v.set_color("#FF0000")).toBe(v);
    // NOTE: Python keeps a single rgba row here; the TS styling pipeline
    // stretches rgbas to one row per point (documented deviation, cf.
    // batch-layer4 header). Every row must carry the observable color.
    const fillRows = rowsOf(v.fill_rgbas, 4);
    expect(fillRows.length).toBe(v.get_num_points());
    for (const row of fillRows) expectRowsClose([row], [[1, 0, 0, 0]]);
    const strokeRows = rowsOf(v.stroke_rgbas, 4);
    expect(strokeRows.length).toBe(v.get_num_points());
    for (const row of strokeRows) expectRowsClose([row], [[1, 0, 0, 1]]);
    expect(v.get_fill_color()?.to_hex()).toBe("#FF0000");
    expect(v.get_stroke_color()?.to_hex()).toBe("#FF0000");
  });
  it("family=False leaves submobjects untouched", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const sub = new VMobject();
    sub.set_points_as_corners([[0, 0, 0], [1, 1, 1]]);
    v.add(sub);
    v.set_color("#00FF00", false);
    expect(v.get_fill_color()?.to_hex()).toBe("#00FF00");
    expect(sub.get_fill_color()?.to_hex()).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_style
// Python (fill '#FF0000'/0.5, stroke '#0000FF'/w2/0.25):
//   simple -> stroke_opacity 0.25, stroke_width 2, fill '#FF0000'/0.5,
//     stroke '#0000FF'.
//   full keys: background_image, background_stroke_color,
//     background_stroke_opacity (1.0), background_stroke_width (0.0),
//     fill_color (['#FF0000']), fill_opacity ([0.5]), sheen_direction
//     ([-1,1,0]), sheen_factor (0.0), stroke_color (['#0000FF']),
//     stroke_opacity (0.25), stroke_width (2); background_image None;
//     background_stroke_color (['#000000']).
// ---------------------------------------------------------------------------
describe("VMobject.get_style", () => {
  function styled(): VMobject {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.set_fill("#FF0000", 0.5);
    v.set_stroke("#0000FF", 2, 0.25);
    return v;
  }
  it("simple style matches Python", () => {
    const s = styled().get_style(true);
    expectClose(s["stroke_opacity"] as number, 0.25);
    expect(s["stroke_width"]).toBe(2);
    expect((s["fill_color"] as ManimColor).to_hex()).toBe("#FF0000");
    expectClose(s["fill_opacity"] as number, 0.5);
    expect((s["stroke_color"] as ManimColor).to_hex()).toBe("#0000FF");
  });
  it("full style matches Python keys and values", () => {
    const s = styled().get_style();
    expect(Object.keys(s).sort()).toEqual([
      "background_image",
      "background_stroke_color",
      "background_stroke_opacity",
      "background_stroke_width",
      "fill_color",
      "fill_opacity",
      "sheen_direction",
      "sheen_factor",
      "stroke_color",
      "stroke_opacity",
      "stroke_width",
    ]);
    expectClose(s["stroke_opacity"] as number, 0.25);
    expect(s["stroke_width"]).toBe(2);
    // NOTE: Python returns one entry per color array (single row); the TS
    // port stores one row per point, so the arrays repeat the observable
    // color -- every entry must match Python's value.
    const fillHexes = (s["fill_color"] as ManimColor[]).map((c) => c?.to_hex());
    expect(fillHexes.length).toBeGreaterThan(0);
    for (const hex of fillHexes) expect(hex).toBe("#FF0000");
    const fillOpacities = ((s["fill_opacity"] as NDArray).toArray() as unknown[]).map(Number);
    expect(fillOpacities.length).toBe(fillHexes.length);
    for (const op of fillOpacities) expectClose(op, 0.5);
    const strokeHexes = (s["stroke_color"] as ManimColor[]).map((c) => c?.to_hex());
    for (const hex of strokeHexes) expect(hex).toBe("#0000FF");
    const bgHexes = (s["background_stroke_color"] as ManimColor[]).map((c) => c?.to_hex());
    expect(bgHexes.length).toBeGreaterThan(0);
    for (const hex of bgHexes) expect(hex).toBe("#000000");
    expect(s["background_stroke_width"]).toBe(0);
    expectClose(s["background_stroke_opacity"] as number, 1.0);
    expect(s["sheen_factor"]).toBe(0);
    expectRowsClose([[...(s["sheen_direction"] as number[])]], [[-1, 1, 0]]);
    expect(s["background_image"]).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_nth_curve_function_with_length
// Python: corners [[0,0,0],[1,0,0],[1,1,0]] -> 2 curves; curve 0 length 1.0,
//   fn(0)=[0,0,0], fn(0.5)=[0.5,0,0], fn(1)=[1,0,0]; curve 1 length 1.0;
//   sample_points=5 still 1.0.
// ---------------------------------------------------------------------------
describe("VMobject.get_nth_curve_function_with_length", () => {
  it("returns the curve function and its length like Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(v.get_num_curves()).toBe(2);
    const [fn0, len0] = v.get_nth_curve_function_with_length(0);
    expectClose(len0, 1.0);
    expectRowsClose([fn0(0)], [[0, 0, 0]]);
    expectRowsClose([fn0(0.5)], [[0.5, 0, 0]]);
    expectRowsClose([fn0(1)], [[1, 0, 0]]);
    const [, len1] = v.get_nth_curve_function_with_length(1);
    expectClose(len1, 1.0);
    const [, lenS] = v.get_nth_curve_function_with_length(0, 5);
    expectClose(lenS, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.change_anchor_mode
// Python: corners [[0,0,0],[1,0,0],[1,1,0]]; smooth ->
//   [[0,0,0],[0.4166..,-0.0833..,0],[0.8333..,-0.1666..,0],[1,0,0],
//    [1,0,0],[1.1666..,0.1666..,0],[1.0833..,0.5833..,0],[1,1,0]];
//   then jagged ->
//   [[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0],
//    [1,0,0],[1,1/3,0],[1,2/3,0],[1,1,0]].
//   bad mode -> AssertionError 'mode must be either "jagged" or "smooth"'.
// ---------------------------------------------------------------------------
describe("VMobject.change_anchor_mode", () => {
  it("smooth then jagged match Python exactly", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(v.change_anchor_mode("smooth")).toBe(v);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.416666666667, -0.083333333333, 0.0],
      [0.833333333333, -0.166666666667, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.166666666667, 0.166666666667, 0.0],
      [1.083333333333, 0.583333333333, 0.0],
      [1.0, 1.0, 0.0],
    ], 1e-9);
    v.change_anchor_mode("jagged");
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.333333333333, 0.0],
      [1.0, 0.666666666667, 0.0],
      [1.0, 1.0, 0.0],
    ], 1e-9);
  });
  it("rejects unknown modes like Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(() => v.change_anchor_mode("bad" as "smooth")).toThrowError(
      /mode must be either "jagged" or "smooth"/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.scale_handle_to_anchor_distances
// Python: corners [[0,0,0],[1,0,0],[2,0,0]]; factor 0.5 ->
//   handles at 1/6, 5/6, 7/6, 11/6; factor 2.0 ->
//   handles at 2/3, 1/3, 5/3, 4/3 (overshoot past the anchor).
// ---------------------------------------------------------------------------
describe("VMobject.scale_handle_to_anchor_distances", () => {
  it("factor 0.5 matches Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    expect(v.scale_handle_to_anchor_distances(0.5)).toBe(v);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.166666666667, 0.0, 0.0],
      [0.833333333333, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.166666666667, 0.0, 0.0],
      [1.833333333333, 0.0, 0.0],
      [2.0, 0.0, 0.0],
    ], 1e-9);
  });
  it("factor 2.0 matches Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    v.scale_handle_to_anchor_distances(2.0);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.666666666667, 0.0, 0.0],
      [1.333333333333, 0.0, 0.0],
      [2.0, 0.0, 0.0],
    ], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform.path_arc / path_func
// Python: default path_arc 0; path_func([0,0,0],[1,1,0],0.5) -> [0.5,0.5,0];
//   path_arc=pi/2, path_func([0,0,0],[1,0,0],0.5) -> [0.5,-0.207106781187,0];
//   explicit path_func is kept as-is; assigning None keeps the old one;
//   path_arc_centers gives a circles path; default axis OUT.
// ---------------------------------------------------------------------------
describe("Transform.path_arc / path_func", () => {
  function at(
    fn: (s: NDArray, e: NDArray, a: number) => unknown,
    s: number[],
    e: number[],
    a: number,
  ): number[] {
    return [...(pointRows(fn(array(s) as NDArray, array(e) as NDArray, a) as NDArray, 3)[0] ?? [])];
  }
  it("defaults to arc 0 with a linear midpoint", () => {
    const tr = new Transform(new Mobject(), new Mobject());
    expect(tr.path_arc).toBe(0);
    expectRowsClose([at(tr.path_func, [0, 0, 0], [1, 1, 0], 0.5)], [[0.5, 0.5, 0]]);
    expect([...(pointRows(tr.path_arc_axis as NDArray, 3)[0] ?? tr.path_arc_axis as number[])]).toEqual([0, 0, 1]);
  });
  it("setting path_arc rebuilds the arc path like Python", () => {
    const tr = new Transform(new Mobject(), new Mobject());
    tr.path_arc = Math.PI / 2;
    expectClose(tr.path_arc, Math.PI / 2);
    expectRowsClose([at(tr.path_func, [0, 0, 0], [1, 0, 0], 0.5)], [[0.5, -0.207106781187, 0]], 1e-9);
  });
  it("explicit path_func wins; None assignment keeps the old one", () => {
    const custom = (s: NDArray, e: NDArray, a: number): NDArray => {
      const ss = pointRows(s, 3)[0] ?? [0, 0, 0];
      const ee = pointRows(e, 3)[0] ?? [0, 0, 0];
      return array([ss.map((v, i) => (1 - a) * v + a * (ee[i] ?? 0))]) as NDArray;
    };
    const tr = new Transform(new Mobject(), new Mobject(), { path_func: custom });
    expect(tr.path_func).toBe(custom);
    tr.path_func = null;
    expect(tr.path_func).toBe(custom);
  });
  it("path_arc_centers builds a circles path", () => {
    const tr = new Transform(new Mobject(), new Mobject(), {
      path_arc: Math.PI,
      path_arc_centers: [[0, 0, 0]],
    });
    expect(tr.path_func).not.toBeNull();
    expect(typeof tr.path_func).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.begin_3dillusion_camera_rotation
//   .update_theta / .update_phi
// Python (pure math of the closures, rate=1):
//   theta origin 2.0, dt=0.5 -> 2.0+0.2*sin(0.5)=2.0958851077208407,
//     tracker 0.5; second dt=0.5 -> 2.0+0.2*sin(1.0)=2.1682941969615794.
//   phi origin 1.0, dt=0.5 -> 1.0+0.1*cos(0.5)-0.1=0.9877582561890373;
//     second -> 1.0+0.1*cos(1.0)-0.1=0.9540302305868139.
// ---------------------------------------------------------------------------
describe("ThreeDScene.begin_3dillusion_camera_rotation updaters", () => {
  function illusionScene(theta0: number, phi0: number): ThreeDScene {
    const scene = new ThreeDScene();
    const thetaTracker = new ValueTracker(theta0);
    const phiTracker = new ValueTracker(phi0);
    scene.renderer = { camera: { theta_tracker: thetaTracker, phi_tracker: phiTracker } };
    // Avoid Scene.add bookkeeping; only the updater math is under test.
    scene.add = (() => scene) as unknown as typeof scene.add;
    scene.begin_3dillusion_camera_rotation(1, phi0, theta0);
    return scene;
  }
  it("update_theta oscillates around the origin like Python", () => {
    const scene = illusionScene(2.0, 1.0);
    const thetaTracker = scene.renderer.camera.theta_tracker as ValueTracker;
    expect(thetaTracker.updaters.length).toBe(1);
    const update = thetaTracker.updaters[0] as (m: unknown, dt: number) => void;
    update(thetaTracker, 0.5);
    expectClose(thetaTracker.get_value(), 2.0958851077208407);
    update(thetaTracker, 0.5);
    expectClose(thetaTracker.get_value(), 2.1682941969615794);
  });
  it("update_phi oscillates around the origin like Python", () => {
    const scene = illusionScene(2.0, 1.0);
    const phiTracker = scene.renderer.camera.phi_tracker as ValueTracker;
    expect(phiTracker.updaters.length).toBe(1);
    const update = phiTracker.updaters[0] as (m: unknown, dt: number) => void;
    update(phiTracker, 0.5);
    expectClose(phiTracker.get_value(), 0.9877582561890373);
    update(phiTracker, 0.5);
    expectClose(phiTracker.get_value(), 0.9540302305868139);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.SpecialThreeDScene
// Python __init__ defaults:
//   cut_axes_at_radius True,
//   camera_config {'should_apply_shading': True, 'exponential_projection': True},
//   three_d_axes_config {'num_axis_pieces': 1, 'axis_config': {'unit_size': 2,
//     'tick_frequency': 1, 'numbers_with_elongated_ticks': [0,1,2],
//     'stroke_width': 2}},
//   sphere_config {'radius': 2, 'resolution': (24, 48)},
//   default_angled_camera_position {'phi': 70*DEGREES, 'theta': -110*DEGREES},
//   low_quality_config {'camera_config': {'should_apply_shading': False},
//     'three_d_axes_config': {'num_axis_pieces': 1},
//     'sphere_config': {'resolution': (12, 24)}}.
// ---------------------------------------------------------------------------
describe("SpecialThreeDScene", () => {
  it("stores Python's default configs", () => {
    const scene = new SpecialThreeDScene();
    expect(scene.cut_axes_at_radius).toBe(true);
    expect(scene.camera_config).toEqual({ should_apply_shading: true, exponential_projection: true });
    expect(scene.three_d_axes_config).toEqual({
      num_axis_pieces: 1,
      axis_config: { unit_size: 2, tick_frequency: 1, numbers_with_elongated_ticks: [0, 1, 2], stroke_width: 2 },
    });
    expect(scene.sphere_config).toEqual({ radius: 2, resolution: [24, 48] });
    expectClose(scene.default_angled_camera_position["phi"] ?? NaN, 70 * DEGREES);
    expectClose(scene.default_angled_camera_position["theta"] ?? NaN, -110 * DEGREES);
    expect(scene.low_quality_config).toEqual({
      camera_config: { should_apply_shading: false },
      three_d_axes_config: { num_axis_pieces: 1 },
      sphere_config: { resolution: [12, 24] },
    });
  });
  it("get_sphere / get_default_camera_position match Python", () => {
    const scene = new SpecialThreeDScene();
    expect(scene.get_sphere().radius).toBe(2);
    expect(scene.get_sphere({ radius: 3 }).radius).toBe(3);
    expect(scene.get_default_camera_position()).toBe(scene.default_angled_camera_position);
    expect(scene.get_axes()).toBeInstanceOf(ThreeDAxes);
  }, 30000);
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ThreeDAxes._set_axis_shading
// Python: for every axis, every pointful submobject gets
//   get_gradient_start_and_end_points = (edge(-v), edge(v)) with
//   v = light_source, get_unit_normal = ones(3), sheen 0.2.
// original manim probe: 79/79 family members shaded; gradient
//   [[5.25,0.175,0],[-5.25,-0.175,0]]; normal [1,1,1].
// ---------------------------------------------------------------------------
describe("ThreeDAxes._set_axis_shading", () => {
  it("installs gradient endpoints, unit normal, and sheen 0.2 (original manim)", () => {
    const axes = new ThreeDAxes();
    axes._set_axis_shading();
    const xAxis = axes.get_x_axis() as unknown as VMobject;
    const fams = xAxis.family_members_with_points();
    expect(fams.length).toBeGreaterThan(0);
    for (const m of fams) {
      expect(m.get_sheen_factor()).toBe(0.2);
      expect(m.get_unit_normal([0, 0, 0])).toEqual([1, 1, 1]);
    }
    expectRowsClose(
      fams[0]!.get_gradient_start_and_end_points() as number[][],
      [[5.25, 0.175, 0], [-5.25, -0.175, 0]],
    );
  });
  it("custom light_source feeds the gradient endpoints (original manim)", () => {
    const axes = new ThreeDAxes(null, null, null, null, [1, 2, 3]);
    axes._set_axis_shading();
    const xAxis = axes.get_x_axis() as unknown as VMobject;
    const fam = xAxis.family_members_with_points()[0]!;
    expectRowsClose(
      fam.get_gradient_start_and_end_points() as number[][],
      [[-5.25, -0.175, 0], [5.25, 0.175, 0]],
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Surface.set_fill_by_checkerboard
// Python: Surface(func, u/v [0,1], resolution=2) -> 4 faces
//   [(u,v)]: [(0,0),(0,1),(1,0),(1,1)];
//   set_fill_by_checkerboard('#FF0000','#00FF00')
//   -> ['#FF0000','#00FF00','#00FF00','#FF0000'];
//   with opacity=0.3 every face opacity is 0.3.
// ---------------------------------------------------------------------------
describe("Surface.set_fill_by_checkerboard", () => {
  function flatSurface(): Surface {
    return new Surface(
      (u: number, v: number) => [u, v, 0],
      [0, 1],
      [0, 1],
      2,
      {},
      "#FFFFFF",
      1.0,
      false,
    );
  }
  it("alternates colors by (u+v) parity like Python", () => {
    const s = flatSurface();
    expect(s.list_of_faces.length).toBe(4);
    expect(s.set_fill_by_checkerboard("#FF0000", "#00FF00")).toBe(s);
    expect(s.list_of_faces.map((f) => [f.u_index, f.v_index])).toEqual([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    expect(s.list_of_faces.map((f) => f.get_fill_color()?.to_hex())).toEqual([
      "#FF0000",
      "#00FF00",
      "#00FF00",
      "#FF0000",
    ]);
  });
  it("opacity kwarg applies to every face", () => {
    const s = flatSurface();
    s.set_fill_by_checkerboard("#FF0000", "#00FF00", { opacity: 0.3 });
    for (const f of s.list_of_faces) {
      expectClose(f.get_fill_opacity(), 0.3);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.on_mouse_scroll / on_mouse_drag
// Python: on_mouse_scroll factor = 1+arctan(-2.1*offset[1]), then
//   camera.scale(factor, about_point=camera_target).
//   offset[1]=2 -> -0.3370531459259951; offset[1]=-1.5 -> 2.263398853672075.
// Python on_mouse_drag: buttons==1 -> increment_theta(-dx),
//   increment_phi(dy); buttons==4 with identity model matrix and
//   d=[0.3,0.4,0] -> shift 1.1*[-0.3,-0.4,0] = [-0.33,-0.44,-0.0].
// (Orbit-control tails are overridden out; they belong to other symbols.)
// ---------------------------------------------------------------------------
class MouseTestScene extends Scene {
  scrollOrbits: Array<{ point: unknown; offset: unknown }> = [];
  dragOrbits: Array<{ point: unknown; dPoint: unknown; buttons: number; modifiers: number }> = [];
  override mouse_scroll_orbit_controls(point: never, offset: never): void {
    this.scrollOrbits.push({ point, offset });
  }
  override mouse_drag_orbit_controls(point: never, dPoint: never, buttons: number, modifiers: number): void {
    this.dragOrbits.push({ point, dPoint, buttons, modifiers });
  }
}

function stubCamera() {
  return {
    scaleCalls: [] as Array<{ factor: number; about: unknown }>,
    thetaCalls: [] as number[],
    phiCalls: [] as number[],
    shiftCalls: [] as number[][],
    model_matrix: array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) as NDArray,
    scale(factor: number, about: unknown): void {
      this.scaleCalls.push({ factor, about });
    },
    increment_theta(v: number): void {
      this.thetaCalls.push(v);
    },
    increment_phi(v: number): void {
      this.phiCalls.push(v);
    },
    shift(v: number[]): void {
      this.shiftCalls.push([...v]);
    },
  };
}

describe("Scene.on_mouse_scroll", () => {
  it("scales by 1+arctan(-2.1*offset_y) about the camera target", () => {
    const scene = new MouseTestScene();
    const camera = stubCamera();
    scene.camera = camera as never;
    scene.camera_target = [1, 2, 3];
    scene.on_mouse_scroll([0, 0, 0], [0, 2, 0]);
    expect(camera.scaleCalls.length).toBe(1);
    // Python: 1+arctan(-2.1*2.0) = -0.3370531459259951.
    expectClose(camera.scaleCalls[0]?.factor ?? NaN, -0.3370531459259951);
    expect(camera.scaleCalls[0]?.about).toEqual([1, 2, 3]);
    expect(scene.scrollOrbits.length).toBe(1);
  });
  it("handles negative scroll offsets like Python", () => {
    const scene = new MouseTestScene();
    const camera = stubCamera();
    scene.camera = camera as never;
    scene.on_mouse_scroll([0, 0, 0], [0, -1.5, 0]);
    // Python: 1+arctan(-2.1*-1.5) = 2.263398853672075.
    expectClose(camera.scaleCalls[0]?.factor ?? NaN, 2.263398853672075);
  });
});

describe("Scene.on_mouse_drag", () => {
  it("left-drag rotates theta/phi and moves the drag point", () => {
    const scene = new MouseTestScene();
    const camera = stubCamera();
    scene.camera = camera as never;
    scene.mouse_drag_point.set_points(array([[0, 0, 0]]) as NDArray);
    scene.on_mouse_drag([1, 2, 3], [0.3, 0.4, 0], 1, 0);
    expect(camera.thetaCalls).toEqual([-0.3]);
    expect(camera.phiCalls).toEqual([0.4]);
    expectRowsClose(rowsOf(scene.mouse_drag_point.points, 3), [[1, 2, 3]]);
    expect(scene.dragOrbits.length).toBe(1);
  });
  it("right-drag shifts by 1.1*total like Python", () => {
    const scene = new MouseTestScene();
    const camera = stubCamera();
    scene.camera = camera as never;
    scene.mouse_drag_point.set_points(array([[0, 0, 0]]) as NDArray);
    scene.on_mouse_drag([0, 0, 0], [0.3, 0.4, 0], 4, 0);
    expect(camera.shiftCalls.length).toBe(1);
    // Python: 1.1*([-0.3,-0.4,-0.0]) = [-0.33,-0.44,-0.0].
    expectRowsClose(camera.shiftCalls, [[-0.33, -0.44000000000000006, -0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.LinearTransformationScene
// Python: defaults include_background/foreground True, show_coordinates
//   False, show_basis_vectors True, stroke width 6, leave_ghost False,
//   i_hat '#83C167' (GREEN_C), j_hat '#FC6255' (RED_C);
//   background_plane_kwargs keys {color, axis_config, background_line_style};
//   foreground_plane_kwargs keys {x_range, y_range, faded_line_ratio=1},
//   x/y ranges [-frame_width, frame_width, 1.0]; ghost_vectors empty VGroup.
//   update_default_configs deep-merges passed dicts, skips None.
//   background_plane_kwargs={'color':'#FF0000'} overrides the default color.
// ---------------------------------------------------------------------------
describe("LinearTransformationScene", () => {
  it("stores Python's constructor defaults", () => {
    const scene = new LinearTransformationScene();
    expect(scene.include_background_plane).toBe(true);
    expect(scene.include_foreground_plane).toBe(true);
    expect(scene.show_coordinates).toBe(false);
    expect(scene.show_basis_vectors).toBe(true);
    expect(scene.basis_vector_stroke_width).toBe(6);
    expect(scene.leave_ghost_vectors).toBe(false);
    expect(scene.i_hat_color.to_hex()).toBe(X_COLOR.to_hex());
    expect(scene.j_hat_color.to_hex()).toBe(Y_COLOR.to_hex());
    expect(scene.i_hat_color.to_hex()).toBe("#83C167");
    expect(scene.j_hat_color.to_hex()).toBe("#FC6255");
    expect(Object.keys(scene.background_plane_kwargs).sort()).toEqual([
      "axis_config",
      "background_line_style",
      "color",
    ]);
    expect(Object.keys(scene.foreground_plane_kwargs).sort()).toEqual([
      "faded_line_ratio",
      "x_range",
      "y_range",
    ]);
    expect(scene.foreground_plane_kwargs["faded_line_ratio"]).toBe(1);
    expect(scene.ghost_vectors).toBeInstanceOf(VGroup);
    expect(scene.ghost_vectors.submobjects.length).toBe(0);
  });
  it("foreground ranges follow config.frame_width like Python", () => {
    const scene = new LinearTransformationScene();
    const w = config.frame_width as number;
    expect(scene.foreground_plane_kwargs["x_range"]).toEqual([-w, w, 1.0]);
    expect(scene.foreground_plane_kwargs["y_range"]).toEqual([-w, w, 1.0]);
  });
  it("update_default_configs merges and skips nulls like Python", () => {
    const d1: Record<string, unknown> = { a: 1, n: { x: 1, y: 2 } };
    const d2: Record<string, unknown> = { b: 2 };
    LinearTransformationScene.update_default_configs([d1, d2], [{ n: { x: 9 } }, null]);
    expect(d1).toEqual({ a: 1, n: { x: 9, y: 2 } });
    expect(d2).toEqual({ b: 2 });
  });
  it("passed plane kwargs override the defaults", () => {
    const scene = new LinearTransformationScene(
      true,
      true,
      { color: "#FF0000" },
      null,
      true,
    );
    expect(scene.show_coordinates).toBe(true);
    const color = scene.background_plane_kwargs["color"];
    const hex = color instanceof ManimColor ? color.to_hex() : (color as string);
    expect(hex).toBe("#FF0000");
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols combined with earlier-ported code, verified
// against the original manim implementation the same way.
// ---------------------------------------------------------------------------
describe("batch integration", () => {
  it("set_color -> get_style round-trips the color", () => {
    // Python: VMobject corners + set_color('#00FF00') -> simple style
    // fill/stroke '#00FF00', stroke_opacity 1.0 (fill opacity stays 0).
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.set_color("#00FF00");
    const s = v.get_style(true);
    expect((s["fill_color"] as ManimColor).to_hex()).toBe("#00FF00");
    expect((s["stroke_color"] as ManimColor).to_hex()).toBe("#00FF00");
    expectClose(s["stroke_opacity"] as number, 1.0);
    expectClose(s["fill_opacity"] as number, 0.0);
  });
  it("arc length agrees with summed curve-function lengths", () => {
    // Python: two straight unit curves -> get_arc_length() == 2.0,
    // equal to len0+len1 from get_nth_curve_function_with_length.
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    const [, len0] = v.get_nth_curve_function_with_length(0);
    const [, len1] = v.get_nth_curve_function_with_length(1);
    expectClose(v.get_arc_length(), 2.0);
    expectClose(len0 + len1, v.get_arc_length());
  });
  it("Transform path_arc drives point interpolation along the arc", () => {
    // Python: Transform(m, target, path_arc=pi/2); path_func(start, end,
    // 0.5) for start=[0,0,0], end=[1,0,0] -> [0.5,-0.207106781187,0].
    const m = new VMobject();
    m.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const target = new VMobject();
    target.set_points_as_corners([[1, 0, 0], [2, 0, 0]]);
    const tr = new Transform(m, target, { path_arc: Math.PI / 2 });
    const mid = tr.path_func(array([0, 0, 0]) as NDArray, array([1, 0, 0]) as NDArray, 0.5);
    expectRowsClose(rowsOf(mid as NDArray, 3), [[0.5, -0.207106781187, 0]], 1e-9);
  });
  it("checkerboard face color surfaces through get_style", () => {
    // A checkerboard face reports its assigned color via get_style too.
    const s = new Surface(
      (u: number, v: number) => [u, v, 0],
      [0, 1],
      [0, 1],
      2,
      {},
      "#FFFFFF",
      1.0,
      false,
    );
    s.set_fill_by_checkerboard("#FF0000", "#00FF00");
    const face = s.list_of_faces[0] as VMobject;
    expect(face.get_fill_color()?.to_hex()).toBe("#FF0000");
    expect(((face.get_style(true)["fill_color"] as ManimColor).to_hex())).toBe("#FF0000");
    expect(((s.list_of_faces[3] as VMobject).get_style(true)["fill_color"] as ManimColor).to_hex()).toBe("#FF0000");
  });
  it("change_anchor_mode + scale_handles round-trip on a smooth curve", () => {
    // Python: smooth the corner shape, scale handles by 1.0 (identity),
    // points unchanged; scale by 0.0 collapses handles onto anchors.
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.change_anchor_mode("smooth");
    const before = rowsOf(v.points, 3).map((r) => [...r]);
    v.scale_handle_to_anchor_distances(1.0);
    expectRowsClose(rowsOf(v.points, 3), before);
    v.scale_handle_to_anchor_distances(0.0);
    const [a1, h1, h2, a2] = v.get_anchors_and_handles();
    expectRowsClose(h1, a1);
    expectRowsClose(h2, a2);
  });
});
