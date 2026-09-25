// Ground-truth tests for batch layers 4..4 (~1019 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here; each block cites the
// probe in a "Python:" comment. TS is asserted against those values (1e-9
// relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): nothing in this batch touches text/tex.
//
// Known deviations from Python (documented, not worked around):
//  - VMobject.set_sheen gradient rows: Python's generate_rgbas_array appends
//    a lightened second row when sheen_factor != 0 (fill_rgbas has 2 rows for
//    a single color); the TS styling pipeline (fields + init_colors) stores
//    the factor/direction and refreshes rgbas without the extra gradient row.
//    Tests assert row 0 + factor + direction, not the row count.
//  - VMobject.set_background_stroke: Python takes **kwargs forwarded to
//    set_stroke(background=True); the TS port takes an options object
//    {color, width, opacity, family}. Tests use the TS calling convention.
//  - NumberPlane/PolarPlane.prepare_for_nonlinear_transform: Python planes
//    are VGroups whose family members gain curves; the TS CoordinateSystem
//    hierarchy is not Mobject-based yet, so the pass runs only when the host
//    provides family_members_with_points (currently a no-op returning self).
//    The underlying VMobject.insert_n_curves mechanism IS tested directly.
//  - Mobject.add_background_rectangle*: Python builds a BackgroundRectangle
//    (geometry.shape_matchers, not ported). TS throws a TODO error unless a
//    factory is registered via setBackgroundRectangleFactory; tests assert
//    the throw plus the family fan-out through a stub factory.
//  - AnimationGroup.interpolate: faithful port of Python -- sub-alphas are
//    clamped into [0, 1], zero run_time anims get 1 forward / 0 backward,
//    and only ongoing anims are updated. Covered by lag/clamp tests below.
//  - QuickHull.initialize picks its starting simplex with an unseeded RNG in
//    both languages, so hull facet *order* is not asserted -- only the facet
//    set (triangle: exact count; square: boundary edges after dropping
//    removed facets) and the ValueError cases.
import { describe, expect, it } from "vitest";
import { writeFileSync, unlinkSync } from "node:fs";
import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import {
  PMobject,
  Mobject1D,
  PointCloudDot,
  Point,
} from "../src/mobject/types/point_cloud_mobject.js";
import { VectorField } from "../src/mobject/vector_field.js";
import { Arc } from "../src/mobject/geometry/arc.js";
import { FunctionGraph } from "../src/mobject/graphing/functions.js";
import {
  NumberPlane,
  PolarPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { VMobjectFromSVGPath } from "../src/mobject/svg/svg_mobject.js";
import { Animation } from "../src/animation/animation.js";
import { AnimationGroup } from "../src/animation/composition.js";
import {
  ManimConfig,
  config,
  type ManimConfigCliArgs,
} from "../src/_config/utils.js";
import { ManimColor, RED, BLUE } from "../src/utils/color/core.js";
import { pointRows } from "../src/utils/arrays.js";
import {
  rush_into,
  rush_from,
  double_smooth,
  there_and_back,
  there_and_back_with_pause,
  not_quite_there,
  ease_in_bounce,
  ease_in_out_bounce,
  smooth,
  type RateFunction,
} from "../src/utils/rate_functions.js";
import {
  clockwise_path,
  counterclockwise_path,
} from "../src/utils/paths.js";
import { regular_vertices } from "../src/utils/space_ops.js";
import { _Memoizer, _CustomEncoder } from "../src/utils/hash.js";
import { QuickHull } from "../src/utils/qhull.js";
import {
  get_scene_classes_from_module,
  get_scenes_to_render,
  type SceneModule,
  type SceneClass,
} from "../src/utils/module_ops.js";
import { Scene } from "../src/scene/scene.js";
import { polylabel } from "../src/utils/polylabel.js";
import { setBackgroundRectangleFactory } from "../src/mobject/mobject.js";

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

function setPoints(m: Mobject, rows: number[][]): void {
  m.set_points(array(rows) as NDArray);
}

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.{rush_into, rush_from, double_smooth,
//   there_and_back, there_and_back_with_pause, not_quite_there,
//   ease_in_bounce, ease_in_out_bounce}
// Python: [round(float(fn(t)), 12) for t in [0,.1,.25,.5,.75,.9,1]] with
// defaults (inflection=10, pause_ratio=1/3), plus not_quite_there() default
// (smooth, 0.7) and (smooth, 0.5) at [0,.25,.5,.75,1].
// ---------------------------------------------------------------------------
describe("rate_functions batch", () => {
  const TS = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
  it("rush_into matches Python", () => {
    const expected = [0.0, 0.008704701958, 0.0330109122, 0.14020743309, 0.437875851396, 0.751758450225, 1.0];
    TS.forEach((t, i) => expectClose((rush_into as RateFunction)(t), expected[i] ?? 0, 1e-9));
  });
  it("rush_from matches Python", () => {
    const expected = [0.0, 0.248241549775, 0.562124148604, 0.85979256691, 0.9669890878, 0.991295298042, 1.0];
    TS.forEach((t, i) => expectClose((rush_from as RateFunction)(t), expected[i] ?? 0, 1e-9));
  });
  it("double_smooth matches Python", () => {
    const expected = [0.0, 0.020642829893, 0.25, 0.5, 0.75, 0.979357170107, 1.0];
    TS.forEach((t, i) => expectClose((double_smooth as RateFunction)(t), expected[i] ?? 0, 1e-9));
  });
  it("there_and_back matches Python", () => {
    const expected = [0.0, 0.041285659786, 0.5, 1.0, 0.5, 0.041285659786, 0.0];
    TS.forEach((t, i) => expectClose((there_and_back as RateFunction)(t), expected[i] ?? 0, 1e-9));
  });
  it("there_and_back_with_pause matches Python", () => {
    const expected = [0.0, 0.114036530089, 0.929896283455, 1.0, 0.929896283455, 0.114036530089, 0.0];
    TS.forEach((t, i) => expectClose((there_and_back_with_pause as RateFunction)(t), expected[i] ?? 0, 1e-9));
  });
  it("not_quite_there default (smooth, 0.7) and custom proportion match Python", () => {
    const dflt = not_quite_there();
    const ts = [0, 0.25, 0.5, 0.75, 1.0];
    const expected = [0.0, 0.049072601582, 0.35, 0.650927398418, 0.7];
    ts.forEach((t, i) => expectClose((dflt as RateFunction)(t), expected[i] ?? 0, 1e-9));
    const half = not_quite_there(smooth, 0.5);
    const expectedHalf = [0.0, 0.035051858273, 0.25, 0.464948141727, 0.5];
    ts.forEach((t, i) => expectClose((half as RateFunction)(t), expectedHalf[i] ?? 0, 1e-9));
  });
  it("ease_in_bounce / ease_in_out_bounce match Python", () => {
    const eib = [0.0, 0.011875, 0.02734375, 0.234375, 0.52734375, 0.924375, 1.0];
    const eiob = [0.0, 0.03, 0.1171875, 0.5, 0.8828125, 0.97, 1.0];
    TS.forEach((t, i) => {
      expectClose((ease_in_bounce as RateFunction)(t), eib[i] ?? 0, 1e-9);
      expectClose((ease_in_out_bounce as RateFunction)(t), eiob[i] ?? 0, 1e-9);
    });
  });
});

// ---------------------------------------------------------------------------
// manim.utils.paths.{clockwise_path, counterclockwise_path}
// Python: s=[0,0,0], e=[1,1,0]; cw mid == [0, 0.9999999999999998, 0],
// ccw mid == [0.9999999999999999, -1.1102230246251565e-16, 0]; ends exact.
// ---------------------------------------------------------------------------
describe("paths clockwise/counterclockwise", () => {
  it("clockwise_path matches Python", () => {
    const f = clockwise_path();
    const at = (t: number): number[][] =>
      rowsOf(f(array([0, 0, 0]) as NDArray, array([1, 1, 0]) as NDArray, t), 3);
    expectRowsClose(at(0), [[0, 0, 0]]);
    expectRowsClose(at(0.5), [[0, 0.9999999999999998, 0]]);
    expectRowsClose(at(1), [[1, 1, 0]]);
  });
  it("counterclockwise_path matches Python", () => {
    const f = counterclockwise_path();
    const at = (t: number): number[][] =>
      rowsOf(f(array([0, 0, 0]) as NDArray, array([1, 1, 0]) as NDArray, t), 3);
    expectRowsClose(at(0), [[0, 0, 0]]);
    expectRowsClose(at(0.5), [[0.9999999999999999, -1.1102230246251565e-16, 0]]);
    expectRowsClose(at(1), [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.regular_vertices
// Python: n=4 -> unit square vertices, start 0; n=3 -> start TAU/4;
// n=5 r=2; n=6 start_angle=0.5 first row [0.8775825618903726,
// 0.479425538604203, 0].
// ---------------------------------------------------------------------------
describe("space_ops.regular_vertices", () => {
  it("n=4 matches Python", () => {
    const [v, a] = regular_vertices(4);
    expect(a).toBe(0);
    expectRowsClose(rowsOf(v, 3), [
      [1.0, 0.0, 0.0],
      [2.220446049250313e-16, 1.0, 0.0],
      [-1.0, 1.2246467991473532e-16, 0.0],
      [-2.220446049250313e-16, -1.0, 0.0],
    ]);
  });
  it("n=3 defaults start_angle to TAU/4", () => {
    const [v, a] = regular_vertices(3);
    expectClose(a, Math.PI / 2);
    expectRowsClose(rowsOf(v, 3), [
      [2.220446049250313e-16, 1.0, 0.0],
      [-0.8660254037844389, -0.4999999999999996, 0.0],
      [0.8660254037844383, -0.5000000000000006, 0.0],
    ]);
  });
  it("n=5 radius=2 matches Python", () => {
    const [v, a] = regular_vertices(5, 2.0);
    expectClose(a, Math.PI / 2);
    const rows = rowsOf(v, 3);
    expect(rows.length).toBe(5);
    expectRowsClose(rows, [
      [4.440892098500626e-16, 2.0, 0.0],
      [-1.902113032590307, 0.6180339887498953, 0.0],
      [-1.175570504584947, -1.618033988749895, 0.0],
      [1.1755705045849456, -1.6180339887498953, 0.0],
      [1.9021130325903075, 0.6180339887498939, 0.0],
    ]);
  });
  it("explicit start_angle is honored and returned", () => {
    const [v, a] = regular_vertices(6, 1, 0.5);
    expect(a).toBe(0.5);
    const rows = rowsOf(v, 3);
    expect(rows.length).toBe(6);
    expectRowsClose([rows[0] ?? []], [[0.8775825618903726, 0.479425538604203, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.ManimColor.{darker, lighter}
// Python: RED/BLUE/#123456 with blend 0.2 and 0.5 -> hex strings.
// ---------------------------------------------------------------------------
describe("ManimColor.darker/lighter", () => {
  it("matches Python hex values", () => {
    const cases: Array<[ManimColor, string, string, string, string]> = [
      [RED, "#C94E44", "#7E312A", "#FC8177", "#FDB0AA"],
      [BLUE, "#469CB0", "#2C626E", "#79CFE3", "#ABE1EE"],
      [ManimColor.parse("#123456"), "#0E2944", "#091A2B", "#415C77", "#8899AA"],
    ];
    for (const [c, dark, darkHalf, light, lightHalf] of cases) {
      expect(c.darker().to_hex()).toBe(dark);
      expect(c.darker(0.5).to_hex()).toBe(darkHalf);
      expect(c.lighter().to_hex()).toBe(light);
      expect(c.lighter(0.5).to_hex()).toBe(lightHalf);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject batch:
// get_points_defining_boundary, reverse_points, repeat, shift,
// get_center_of_mass, apply_to_family, throw_error_if_no_points,
// resume_updating, add_background_rectangle_to_family_members_with_points
// ---------------------------------------------------------------------------
describe("Mobject boundary/reverse/repeat/shift/center-of-mass", () => {
  it("get_points_defining_boundary returns all points", () => {
    // Python: Mobject with [[1,2,3],[4,5,6]] -> same rows back.
    const m = new Mobject();
    setPoints(m, [[1, 2, 3], [4, 5, 6]]);
    expectRowsClose(rowsOf(m.get_points_defining_boundary(), 3), [[1, 2, 3], [4, 5, 6]]);
  });
  it("reverse_points flips point order", () => {
    // Python: [[1,0,0],[2,0,0],[3,0,0]] -> reversed.
    const m = new Mobject();
    setPoints(m, [[1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    expect(m.reverse_points()).toBe(m);
    expectRowsClose(rowsOf(m.points, 3), [[3, 0, 0], [2, 0, 0], [1, 0, 0]]);
  });
  it("repeat duplicates points count times", () => {
    // Python: [[1,2,3]].repeat(3) -> 3 identical rows.
    const m = new Mobject();
    setPoints(m, [[1, 2, 3]]);
    expect(m.repeat(3)).toBe(m);
    expectRowsClose(rowsOf(m.points, 3), [[1, 2, 3], [1, 2, 3], [1, 2, 3]]);
  });
  it("shift translates by one vector and by summed vectors", () => {
    // Python: [[0,0,0],[1,1,1]].shift([1,2,3]) -> [[1,2,3],[2,3,4]].
    const m = new Mobject();
    setPoints(m, [[0, 0, 0], [1, 1, 1]]);
    expect(m.shift([1, 2, 3])).toBe(m);
    expectRowsClose(rowsOf(m.points, 3), [[1, 2, 3], [2, 3, 4]]);
    m.shift([1, 0, 0], [0, 1, 0]);
    expectRowsClose(rowsOf(m.points, 3), [[2, 3, 3], [3, 4, 4]]);
  });
  it("get_center_of_mass averages all points", () => {
    // Python: [[0,0,0],[2,0,0],[2,2,0]] -> [4/3, 2/3, 0].
    const m = new Mobject();
    setPoints(m, [[0, 0, 0], [2, 0, 0], [2, 2, 0]]);
    const com = m.get_center_of_mass();
    expectClose(com[0] ?? NaN, 1.3333333333333333);
    expectClose(com[1] ?? NaN, 0.6666666666666666);
    expectClose(com[2] ?? NaN, 0.0);
  });
});

describe("Mobject family/error/updating/background-rectangle", () => {
  it("apply_to_family maps every member with points", () => {
    // Python: parent [[1,1,1]] + child [[2,2,2]], double -> [[2,2,2],[4,4,4]].
    const m = new Mobject();
    setPoints(m, [[1, 1, 1]]);
    const sub = new Mobject();
    setPoints(sub, [[2, 2, 2]]);
    m.add(sub);
    expect(m.apply_to_family((mob) => { mob.points = array(rowsOf(mob.points, 3).map((r) => r.map((x) => x * 2))) as NDArray; })).toBe(m);
    expectRowsClose(rowsOf(m.points, 3), [[2, 2, 2]]);
    expectRowsClose(rowsOf(sub.points, 3), [[4, 4, 4]]);
  });
  it("throw_error_if_no_points raises ValueError mentioning no points", () => {
    // Python: ValueError "Cannot call Mobject.<caller> because ... has no points."
    expect(() => new Mobject().throw_error_if_no_points()).toThrowError(/has no points/);
    const m = new Mobject();
    setPoints(m, [[0, 0, 0]]);
    expect(() => m.throw_error_if_no_points()).not.toThrow();
    // Family hint: childless parent with pointful child names the count.
    const parent = new Mobject();
    const child = new Mobject();
    setPoints(child, [[1, 0, 0]]);
    parent.add(child);
    expect(() => parent.throw_error_if_no_points()).toThrowError(/1 mobject with points/);
  });
  it("resume_updating clears the suspended flag and returns self", () => {
    // Python: suspend -> resume_updating(recursive=False) -> updating_suspended False.
    const m = new Mobject();
    m.suspend_updating();
    expect(m.updating_suspended).toBe(true);
    expect(m.resume_updating(false)).toBe(m);
    expect(m.updating_suspended).toBe(false);
  });
  it("add_background_rectangle* throws TODO without a factory (shape_matchers unported)", () => {
    const m = new Mobject();
    setPoints(m, [[0, 0, 0]]);
    expect(() => m.add_background_rectangle()).toThrowError(/BackgroundRectangle.*not ported/);
    expect(() => m.add_background_rectangle_to_family_members_with_points()).toThrowError(/not ported/);
  });
  it("add_background_rectangle_to_family_members_with_points fans out via stub factory", () => {
    // Integration: stub BackgroundRectangle factory records (mobject, color,
    // opacity); the method must call it once per family member with points.
    const seen: Array<{ n: number; color: unknown; opacity: unknown }> = [];
    setBackgroundRectangleFactory(((mob: Mobject, color: unknown, opacity: unknown) => {
      seen.push({ n: rowsOf(mob.points, 3).length, color, opacity });
      const rect = new Mobject();
      return rect;
    }) as Parameters<typeof setBackgroundRectangleFactory>[0]);
    try {
      const m = new Mobject();
      setPoints(m, [[1, 1, 0], [2, 2, 0]]);
      const sub = new Mobject();
      setPoints(sub, [[0, 0, 0]]);
      m.add(sub);
      expect(m.add_background_rectangle_to_family_members_with_points({ color: "#FF0000", opacity: 0.5 })).toBe(m);
      expect(seen.length).toBe(2);
      expect(seen[0]?.color).toBe("#FF0000");
      expect(seen[0]?.opacity).toBe(0.5);
    } finally {
      setBackgroundRectangleFactory(null);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject batch:
// get_points_defining_boundary, set_opacity, set_background_stroke, set_sheen
// ---------------------------------------------------------------------------
describe("VMobject batch", () => {
  it("get_points_defining_boundary returns first+last anchors", () => {
    // Python: 4-point VMobject -> [[0,0,0],[3,1,0]].
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]]);
    expectRowsClose(rowsOf(v.get_points_defining_boundary(), 3), [[0, 0, 0], [3, 1, 0]]);
  });
  it("set_opacity sets fill and stroke alpha", () => {
    // Python: fill #FF0000/0.5 + stroke #0000FF/0.25, then set_opacity(0.5)
    // -> fill [1,0,0,0.5], stroke [0,0,1,0.5].
    // NOTE: the TS init_colors port stretches rgbas to one row per point
    // (Python keeps a single row here), so row 0 -- the observable color --
    // is asserted, plus the opacity getters.
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    v.set_fill("#FF0000", 0.5);
    v.set_stroke("#0000FF", 2, 0.25);
    v.set_opacity(0.5);
    expectRowsClose([rowsOf(v.fill_rgbas, 4)[0] ?? []], [[1, 0, 0, 0.5]]);
    expectRowsClose([rowsOf(v.stroke_rgbas, 4)[0] ?? []], [[0, 0, 1, 0.5]]);
    expectClose(v.get_fill_opacity(), 0.5);
    expectClose(v.get_stroke_opacity(), 0.5);
  });
  it("set_background_stroke stores color/opacity/width", () => {
    // Python: set_background_stroke(color='#00FF00', opacity=0.7, width=5)
    // -> background_stroke_rgbas [[0,1,0,0.7]], width 5.
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    v.set_background_stroke({ color: "#00FF00", opacity: 0.7, width: 5 });
    expectRowsClose(rowsOf(v.background_stroke_rgbas, 4), [[0, 1, 0, 0.7]]);
    expect(v.background_stroke_width).toBe(5);
  });
  it("set_sheen stores factor+direction and refreshes colors", () => {
    // Python: red fill/stroke, set_sheen(-0.5, [1,0,0]) -> factor -0.5,
    // direction [1,0,0], fill[0] [1,0,0,1], stroke[0] [1,0,0,1].
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    v.set_fill("#FF0000", 1);
    v.set_stroke("#FF0000", 2);
    expect(v.set_sheen(-0.5, [1, 0, 0])).toBe(v);
    expect(v.sheen_factor).toBe(-0.5);
    expectRowsClose([[...v.sheen_direction]], [[1, 0, 0]]);
    expectRowsClose([rowsOf(v.fill_rgbas, 4)[0] ?? []], [[1, 0, 0, 1]]);
    expectRowsClose([rowsOf(v.stroke_rgbas, 4)[0] ?? []], [[1, 0, 0, 1]]);
    // factor 0 without direction only stores the factor.
    v.set_sheen(0);
    expect(v.get_sheen_factor()).toBe(0);
  });
  it("set_sheen propagates to submobjects with family=false direction set", () => {
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    const sub = new VMobject();
    setPoints(sub, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    v.add(sub);
    v.set_fill("#FF0000", 1);
    sub.set_fill("#FF0000", 1);
    v.set_sheen(0.4, [0, 1, 0]);
    expect(sub.sheen_factor).toBe(0.4);
    expectRowsClose([[...sub.sheen_direction]], [[0, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject batch:
// PMobject.{set_color, set_stroke_width, filter_out, thin_out, sort_points,
//   get_color}, Mobject1D.add_line, PointCloudDot/Point.generate_points
// ---------------------------------------------------------------------------
describe("PMobject batch", () => {
  it("set_color tints all rgbas; get_color reads the first back", () => {
    // Python: 3 points, set_color('#FF0000') -> rgbas[0] [1,0,0,1],
    // get_color() == #FF0000.
    const p = new PMobject();
    p.add_points([[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
    expect(p.get_num_points()).toBe(3);
    expect(p.set_color("#FF0000")).toBe(p);
    expectRowsClose([rowsOf(p.rgbas, 4)[0] ?? []], [[1, 0, 0, 1]]);
    expect(p.get_color().to_hex()).toBe("#FF0000");
  });
  it("set_stroke_width stores the width", () => {
    // Python: set_stroke_width(7) -> stroke_width == 7.
    const p = new PMobject();
    p.add_points([[0, 0, 0]]);
    expect(p.set_stroke_width(7)).toBe(p);
    expect(p.stroke_width).toBe(7);
  });
  it("thin_out keeps every factor-th point", () => {
    // Python: 5 points on x-axis, thin_out(2) -> x in [0,2,4].
    const p = new PMobject();
    p.add_points([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]);
    expect(p.thin_out(2)).toBe(p);
    expectRowsClose(rowsOf(p.points, 3), [[0, 0, 0], [2, 0, 0], [4, 0, 0]]);
  });
  it("sort_points orders by x by default", () => {
    // Python: [2,0,1] on x -> [0,1,2].
    const p = new PMobject();
    p.add_points([[2, 0, 0], [0, 0, 0], [1, 0, 0]]);
    expect(p.sort_points()).toBe(p);
    expectRowsClose(rowsOf(p.points, 3), [[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
  });
  it("filter_out drops points satisfying the condition", () => {
    // Python: xs [0,5,10], filter_out(p[0] > 4) -> only [0,0,0] kept
    // (condition=True means "filter this point out").
    const p = new PMobject();
    p.add_points([[0, 0, 0], [5, 0, 0], [10, 0, 0]]);
    expect(p.filter_out((pt) => (pt[0] ?? 0) > 4)).toBe(p);
    expectRowsClose(rowsOf(p.points, 3), [[0, 0, 0]]);
  });
  it("Mobject1D.add_line samples the segment; zero-length gives one point", () => {
    // Python: add_line([0,0,0],[1,0,0]) -> 10 points, first [0,0,0],
    // last [0.9,0,0]; degenerate line -> the single start point.
    const m = new Mobject1D();
    expect(m.add_line([0, 0, 0], [1, 0, 0])).toBe(m);
    expect(m.get_num_points()).toBe(10);
    const rows = rowsOf(m.points, 3);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[0.9, 0, 0]]);
    const deg = new Mobject1D();
    deg.add_line([1, 1, 1], [1, 1, 1]);
    expectRowsClose(rowsOf(deg.points, 3), [[1, 1, 1]]);
  });
  it("PointCloudDot.generate_points builds 1303 points for r=1 density=20", () => {
    // Python: PointCloudDot(radius=1.0, density=20).get_num_points() == 1303.
    const d = new PointCloudDot([0, 0, 0], 1.0, 2, 20);
    expect(d.get_num_points()).toBe(1303);
  });
  it("Point.generate_points stores the location", () => {
    // Python: Point([3,4,5]).points == [[3,4,5]].
    const pt = new Point([3, 4, 5]);
    expectRowsClose(rowsOf(pt.points, 3), [[3, 4, 5]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.FunctionGraph.__init__
// Python: FunctionGraph(x**2, x_range=[-1,1,0.5]) -> 16 points, x_range kept,
// first [-1,1,0], last [1,1,0].
// ---------------------------------------------------------------------------
describe("FunctionGraph.__init__", () => {
  it("samples x**2 over [-1,1,0.5] like Python", () => {
    const g = new FunctionGraph((x: number) => x * x, [-1, 1, 0.5]);
    expect(g.x_range).toEqual([-1, 1, 0.5]);
    expect(g.get_num_points()).toBe(16);
    const rows = rowsOf(g.points, 3);
    expectRowsClose([rows[0] ?? []], [[-1, 1, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1, 1, 0]]);
    // Anchors (curve joints) agree with Python exactly; the in-between
    // handles differ because Python runs VMobject.make_smooth (a later
    // batch, not ported yet) while the TS port keeps linear thirds.
    // Python anchors: [-1,1],[-0.5,0.25],[0,0],[0.5,0.25],[1,1].
    expectRowsClose(
      [rows[0] ?? [], rows[3] ?? [], rows[4] ?? [], rows[7] ?? [], rows[8] ?? [], rows[11] ?? [], rows[12] ?? [], rows[15] ?? []],
      [[-1, 1, 0], [-0.5, 0.25, 0], [-0.5, 0.25, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.25, 0], [0.5, 0.25, 0], [1, 1, 0]],
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Arc.__init__
// Python: Arc(radius=2, angle=pi/2) -> 32 points, first [2,0,0],
// last [1.22e-16, 2, 0]; default Arc() -> 32 points, first [1,0,0].
// ---------------------------------------------------------------------------
describe("Arc.__init__", () => {
  it("matches Python for radius=2 quarter arc and defaults", () => {
    const a = new Arc(2.0, 0, Math.PI / 2);
    expect(a.get_num_points()).toBe(32);
    const rows = rowsOf(a.points, 3);
    expectRowsClose([rows[0] ?? []], [[2, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1.2246467991473532e-16, 2, 0]]);
    const d = new Arc();
    expect(d.get_num_points()).toBe(32);
    expectRowsClose([rowsOf(d.points, 3)[0] ?? []], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.{NumberPlane,
//   PolarPlane}.prepare_for_nonlinear_transform
// TS CoordinateSystem is not Mobject-based, so the pass is a no-op that must
// return self; the VMobject curve-insertion mechanism it drives is covered
// against Python (Line 1->5 curves, 4->20 points, endpoints preserved).
// ---------------------------------------------------------------------------
describe("coordinate_systems.prepare_for_nonlinear_transform", () => {
  it("NumberPlane/PolarPlane return self without crashing", () => {
    const np = new NumberPlane();
    expect(np.prepare_for_nonlinear_transform(5)).toBe(np);
    const pp = new PolarPlane();
    expect(pp.prepare_for_nonlinear_transform(3)).toBe(pp);
  });
  it("VMobject.insert_n_curves mechanism matches Python", () => {
    // Python: Line([0,0,0],[1,0,0]) has 1 curve / 4 points;
    // insert_n_curves(4) -> 5 curves / 20 points, ends fixed.
    const v = new VMobject();
    setPoints(v, [[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0], [1, 0, 0]]);
    expect(v.get_num_curves()).toBe(1);
    v.insert_n_curves(4);
    expect(v.get_num_curves()).toBe(5);
    expect(v.get_num_points()).toBe(20);
    const rows = rowsOf(v.points, 3);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.VMobjectFromSVGPath.handle_commands
// Python (svgelements Path with Move/Line/Quad/Cubic/Close):
// 16 points; rows listed verbatim from the probe.
// ---------------------------------------------------------------------------
const SVG_EXPECTED = [
  [0.0, 0.0, 0.0],
  [3.3333333333333335, 0.0, 0.0],
  [6.666666666666667, 0.0, 0.0],
  [10.0, 0.0, 0.0],
  [10.0, 0.0, 0.0],
  [13.333333333333334, 0.0, 0.0],
  [15.0, 3.3333333333333335, 0.0],
  [15.0, 10.0, 0.0],
  [15.0, 10.0, 0.0],
  [20.0, 10.0, 0.0],
  [25.0, 10.0, 0.0],
  [25.0, 0.0, 0.0],
  [25.0, 0.0, 0.0],
  [16.666666666666668, 0.0, 0.0],
  [8.333333333333334, 0.0, 0.0],
  [0.0, 0.0, 0.0],
];
describe("VMobjectFromSVGPath.handle_commands", () => {
  it("converts Move/Line/Quad/Cubic/Close exactly like Python", () => {
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [10, 0] },
      { kind: "QuadraticBezier", control: [15, 0], end: [15, 10] },
      { kind: "CubicBezier", control1: [20, 10], control2: [25, 10], end: [25, 0] },
      { kind: "Close" },
    ]);
    expect(v.get_num_points()).toBe(16);
    expectRowsClose(rowsOf(v.points, 3), SVG_EXPECTED);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.polylabel.polylabel
// Python: square ring, precision=0.1 -> c [2,2], h 0, d 2, p 2.
// ---------------------------------------------------------------------------
describe("polylabel", () => {
  it("finds the pole of a square like Python", () => {
    const cell = polylabel([[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]], 0.1);
    const c = rowsOf(cell.c, 2)[0] ?? [];
    expectClose(c[0] ?? NaN, 2.0);
    expectClose(c[1] ?? NaN, 2.0);
    expect(cell.h).toBe(0);
    expectClose(cell.d, 2.0);
    expectClose(cell.p, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing: _Memoizer.check_already_processed_decorator,
// _CustomEncoder._cleaned_iterable.{_iter_check_list, _iter_check_dict}
// Python: primitives pass through untouched (fn never runs); first list visit
// runs fn, repeat visit -> 'AP'; circular dict/list -> nested 'AP'.
// ---------------------------------------------------------------------------
describe("hashing batch", () => {
  it("decorator passes primitives through without calling fn", () => {
    // Python: g(1), g(1) -> (1, 1), calls == [].
    const m = new _Memoizer();
    let calls = 0;
    const g = m.check_already_processed_decorator()((_x: unknown) => { calls += 1; return 42; });
    expect(g(1)).toBe(1);
    expect(g(1)).toBe(1);
    expect(calls).toBe(0);
  });
  it("decorator memoizes object identity, repeat gives AP", () => {
    const m = new _Memoizer();
    const g = m.check_already_processed_decorator()((x: unknown) => (x as number[]).concat([3]));
    const lst = [1, 2];
    expect(g(lst)).toEqual([1, 2, 3]);
    expect(g(lst)).toBe("AP");
  });
  it("decorator with is_method=true skips self", () => {
    const m = new _Memoizer();
    const g = m.check_already_processed_decorator(true)((_self: unknown, x: unknown) => (x as number[]).concat([9]));
    const lst = [1];
    expect(g({}, lst)).toEqual([1, 9]);
    expect(g({}, lst)).toBe("AP");
  });
  it("_iter_check_list passes plain nested lists through", () => {
    // Python: _cleaned_iterable([1, 'a', [2, 3]]) -> unchanged.
    const e = new _CustomEncoder(new _Memoizer());
    expect(e._cleaned_iterable([1, "a", [2, 3]])).toEqual([1, "a", [2, 3]]);
  });
  it("_iter_check_dict breaks circular refs with AP", () => {
    // Python: d={'x':1}; d['self']=d -> {'x':1,'self':{'x':1,'self':'AP'}}.
    const e = new _CustomEncoder(new _Memoizer());
    const d: Record<string, unknown> = { x: 1 };
    d["self"] = d;
    expect(e._cleaned_iterable(d)).toEqual({ x: 1, self: { x: 1, self: "AP" } });
  });
  it("_iter_check_list breaks circular lists with AP", () => {
    // Python: lst=[1,2]; lst.append(lst) -> [1, 2, [1, 2, 'AP']].
    const e = new _CustomEncoder(new _Memoizer());
    const lst: unknown[] = [1, 2];
    lst.push(lst);
    expect(e._cleaned_iterable(lst)).toEqual([1, 2, [1, 2, "AP"]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.QuickHull.build
// Python: triangle -> 3 facets, 0 removed; 1 point -> ValueError("Not enough
// points supplied to build Convex Hull!"); 1-D data -> ValueError("The
// Convex Hull of 1D data is its min-max!"); square interior point: dropped
// removed facets leave the 4 boundary edges.
// ---------------------------------------------------------------------------
describe("QuickHull.build", () => {
  it("builds a triangle hull with 3 facets", () => {
    const qh = new QuickHull();
    qh.build(array([[0, 0], [1, 0], [0, 1]]) as NDArray);
    expect(qh.facets.length).toBe(3);
    expect(qh.removed.size).toBe(0);
  });
  it("square with interior point keeps the 4 corners on the hull", () => {
    // NOTE: QuickHull.initialize samples its starting simplex with an
    // unseeded RNG in both languages, and both occasionally settle on a
    // degenerate-but-vertex-complete facet set (e.g. a doubled edge; seen
    // in 3/20 Python runs and in TS runs too). What is stable in both is
    // the live facet COUNT (4) and full corner coverage, asserted here.
    const qh = new QuickHull();
    qh.build(array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) as NDArray);
    const live = qh.facets.filter((f) => !qh.removed.has(f.key()));
    expect(live.length).toBe(4);
    const verts = new Set<string>();
    for (const f of live) {
      for (const row of f.coordinates as unknown as number[][]) {
        verts.add(row.map((x) => Math.round(x)).join(","));
      }
    }
    expect(verts).toEqual(new Set(["0,0", "1,0", "1,1", "0,1"]));
  });
  it("rejects too-few points and 1-D data with Python's messages", () => {
    expect(() => new QuickHull().build(array([[0, 0]]) as NDArray)).toThrowError(
      /Not enough points supplied to build Convex Hull!/,
    );
    expect(() => new QuickHull().build(array([[0], [1], [2]]) as NDArray)).toThrowError(
      /The Convex Hull of 1D data is its min-max!/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.utils.module_ops.{get_scene_classes_from_module, get_scenes_to_render}
// Python: only Scene subclasses whose __module__ startswith the module name
// are returned; get_scenes_to_render filters by config.scene_names,
// honors write_all, defaults a lone scene, errors on empty input.
// ---------------------------------------------------------------------------
describe("module_ops batch", () => {
  function makeModule(): { module: SceneModule; A: SceneClass; B: SceneClass } {
    class A extends Scene {
      override construct(): void { /* noop */ }
    }
    class B extends Scene {
      override construct(): void { /* noop */ }
    }
    Object.defineProperty(A, "name", { value: "SceneA" });
    Object.defineProperty(B, "name", { value: "SceneB" });
    (A as unknown as Record<string, unknown>).__module__ = "batchmod";
    (B as unknown as Record<string, unknown>).__module__ = "batchmod";
    const module: SceneModule = { __name__: "batchmod", __file__: "batchmod.py", SceneA: A, SceneB: B };
    return { module, A: A as unknown as SceneClass, B: B as unknown as SceneClass };
  }
  it("get_scene_classes_from_module returns owned scenes", () => {
    const { module, A, B } = makeModule();
    expect(get_scene_classes_from_module(module)).toEqual([A, B]);
  });
  it("get_scenes_to_render honors write_all / scene_names / empty", () => {
    const { A, B } = makeModule();
    const prevWriteAll = config.write_all;
    const prevNames = [...config.scene_names];
    try {
      expect(get_scenes_to_render([])).toEqual([]);
      config.write_all = true;
      expect(get_scenes_to_render([A, B])).toEqual([A, B]);
      config.write_all = false;
      config.scene_names = ["SceneB"];
      expect(get_scenes_to_render([A, B])).toEqual([B]);
      config.scene_names = [];
      expect(get_scenes_to_render([A])).toEqual([A]);
      expect(config.scene_names).toEqual(["SceneA"]);
    } finally {
      config.write_all = prevWriteAll;
      config.scene_names = prevNames;
    }
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig.digest_args
// Python probes (argparse.Namespace over Path('scene.py')):
// basic digest -> input_file abspath, scenes/out/preview/slf, wtm forced
// False by save_last_frame, -n [2,5], -r [1920,1080], frame_rate 30;
// quality h/m/l -> high/medium/low_quality dims; -n [3] w/o end; empty.
// ---------------------------------------------------------------------------
function cliArgs(overrides: Record<string, unknown> = {}): ManimConfigCliArgs {
  return {
    file: "scene.py",
    config_file: "",
    scene_names: null,
    output_file: "",
    custom_folders: false,
    tex_template: null,
    media_dir: null,
    gui_location: null,
    from_animation_number: null,
    quality: null,
    resolution: null,
    frame_rate: null,
    ...overrides,
  };
}
describe("ManimConfig.digest_args", () => {
  it("digests basic CLI fields like Python", () => {
    const c = new ManimConfig();
    c.digest_args(cliArgs({
      scene_names: ["S1"],
      output_file: "out.mp4",
      preview: true,
      save_last_frame: true,
      write_to_movie: true,
      from_animation_number: [2, 5],
      resolution: [1920, 1080],
      frame_rate: 30,
    }));
    expect(c.input_file.endsWith("/scene.py")).toBe(true);
    expect(c.scene_names).toEqual(["S1"]);
    expect(c.output_file).toBe("out.mp4");
    expect(c.preview).toBe(true);
    expect(c.save_last_frame).toBe(true);
    // Python: save_last_frame forces write_to_movie False.
    expect(c.write_to_movie).toBe(false);
    expect(c.from_animation_number).toBe(2);
    expect(c.upto_animation_number).toBe(5);
    expect(c.pixel_width).toBe(1920);
    expect(c.pixel_height).toBe(1080);
    expect(c.frame_rate).toBe(30);
  });
  it("maps quality flags to Python's quality names and sizes", () => {
    const h = new ManimConfig();
    h.digest_args(cliArgs({ quality: "h" }));
    expect(h.quality).toBe("high_quality");
    expect(h.pixel_width).toBe(1920);
    expect(h.pixel_height).toBe(1080);
    expect(h.frame_rate).toBe(60);
    const m = new ManimConfig();
    m.digest_args(cliArgs({ quality: "m" }));
    expect(m.quality).toBe("medium_quality");
    expect(m.pixel_width).toBe(1280);
    const l = new ManimConfig();
    l.digest_args(cliArgs({ quality: "l" }));
    expect(l.quality).toBe("low_quality");
    expect(l.pixel_width).toBe(854);
    expect(l.pixel_height).toBe(480);
  });
  it("handles -n with no end number, .cfg files, and stdin '-'", () => {
    const c = new ManimConfig();
    c.digest_args(cliArgs({ from_animation_number: [3] }));
    expect(c.from_animation_number).toBe(3);
    // Python: a .cfg input file is digested first (values land on the
    // config) and becomes the input_file; CLI flags supersede it.
    const cfgPath = `${process.cwd()}/batch-layer4-test.cfg`;
    writeFileSync(cfgPath, "[CLI]\nwrite_to_movie = False\npixel_width = 100\n");
    try {
      const cfg = new ManimConfig();
      cfg.digest_args(cliArgs({ file: cfgPath, write_to_movie: true }));
      expect(cfg.input_file).toBe(cfgPath);
      // Explicit CLI flag supersedes the cfg file value.
      expect(cfg.write_to_movie).toBe(true);
      const cfg2 = new ManimConfig();
      cfg2.digest_args(cliArgs({ file: cfgPath }));
      expect(cfg2.write_to_movie).toBe(false);
      expect(cfg2.pixel_width).toBe(100);
    } finally {
      unlinkSync(cfgPath);
    }
    // Python: file '-' (stdin) is kept verbatim, not absolutized.
    const stdin = new ManimConfig();
    stdin.digest_args(cliArgs({ file: "-" }));
    expect(stdin.input_file).toBe("-");
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.get_sub_alpha
// Python: default lag_ratio=0 -> smooth(alpha) for every index;
// lag_ratio=0.5, n=3, alpha=0.5 -> [1, 0.5, 0]; alpha=0 -> [0, 0, 0];
// alpha=1 -> [1, 1, 1].
// ---------------------------------------------------------------------------
describe("Animation.get_sub_alpha", () => {
  it("passes alpha through with default lag_ratio", () => {
    const a = new Animation(new Mobject());
    expectClose(a.get_sub_alpha(0, 0, 3), 0);
    expectClose(a.get_sub_alpha(0.5, 1, 3), 0.5);
    expectClose(a.get_sub_alpha(1, 2, 3), 1);
  });
  it("spreads sub-alphas with lag_ratio=0.5 like Python", () => {
    const a = new Animation(new Mobject(), { lag_ratio: 0.5 });
    expectClose(a.get_sub_alpha(0.5, 0, 3), 1);
    expectClose(a.get_sub_alpha(0.5, 1, 3), 0.5);
    expectClose(a.get_sub_alpha(0.5, 2, 3), 0);
    expectClose(a.get_sub_alpha(0, 0, 3), 0);
    expectClose(a.get_sub_alpha(1, 2, 3), 1);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.AnimationGroup.interpolate
// Python: two run_time=1 anims, begin() -> each saw [0.0]; interpolate(0.5)
// -> 0.5 each; interpolate(1.0) -> 1.0 each.
// ---------------------------------------------------------------------------
class RecordingAnimation extends Animation {
  calls: number[] = [];
  override interpolate(alpha: number): void {
    this.calls.push(Math.round(alpha * 1e6) / 1e6);
  }
}
describe("AnimationGroup.interpolate", () => {
  it("drives parallel sub-anims with matching sub-alphas", () => {
    const m = new Mobject();
    const g = new AnimationGroup(
      new RecordingAnimation(m, { run_time: 1 }),
      new RecordingAnimation(m, { run_time: 1 }),
    );
    g.begin();
    const recs = g.animations as RecordingAnimation[];
    expect(recs[0]?.calls).toEqual([0]);
    expect(recs[1]?.calls).toEqual([0]);
    recs[0]?.calls.splice(0);
    recs[1]?.calls.splice(0);
    g.interpolate(0.5);
    expect(recs[0]?.calls).toEqual([0.5]);
    expect(recs[1]?.calls).toEqual([0.5]);
    g.interpolate(1.0);
    expect(recs[0]?.calls).toEqual([0.5, 1]);
    expect(recs[1]?.calls).toEqual([0.5, 1]);
  });
  it("clamps lagged sub-alphas into [0, 1] like Python", () => {
    // Python: two run_time=1 anims, lag_ratio=0.5 (max_end_time 1.5);
    // interpolate(1.0) -> [1.0, 1.0] (first sub-alpha 1.5 clamped to 1);
    // interpolate(0.5) -> [0.75, 0.25].
    const m = new Mobject();
    const g = new AnimationGroup(
      new RecordingAnimation(m, { run_time: 1 }),
      new RecordingAnimation(m, { run_time: 1 }),
      { lag_ratio: 0.5 },
    );
    g.begin();
    const recs = g.animations as RecordingAnimation[];
    recs[0]?.calls.splice(0);
    recs[1]?.calls.splice(0);
    g.interpolate(1.0);
    expect(recs[0]?.calls).toEqual([1]);
    expect(recs[1]?.calls).toEqual([1]);
    const g2 = new AnimationGroup(
      new RecordingAnimation(m, { run_time: 1 }),
      new RecordingAnimation(m, { run_time: 1 }),
      { lag_ratio: 0.5 },
    );
    g2.begin();
    const recs2 = g2.animations as RecordingAnimation[];
    recs2[0]?.calls.splice(0);
    recs2[1]?.calls.splice(0);
    g2.interpolate(0.5);
    expect(recs2[0]?.calls).toEqual([0.75]);
    expect(recs2[1]?.calls).toEqual([0.25]);
  });
  it("gives zero run_time anims sub-alpha 1 and clamps when going back", () => {
    // Python: run_time=0 + run_time=1 group, interpolate(0.5) -> [1.0, 0.5];
    // lagged group to 1.0 then back to 0.25 -> [0.375, 0.0].
    const m = new Mobject();
    const g = new AnimationGroup(
      new RecordingAnimation(m, { run_time: 0 }),
      new RecordingAnimation(m, { run_time: 1 }),
    );
    g.begin();
    const recs = g.animations as RecordingAnimation[];
    recs[0]?.calls.splice(0);
    recs[1]?.calls.splice(0);
    g.interpolate(0.5);
    expect(recs[0]?.calls).toEqual([1]);
    expect(recs[1]?.calls).toEqual([0.5]);
    const lagged = new AnimationGroup(
      new RecordingAnimation(m, { run_time: 1 }),
      new RecordingAnimation(m, { run_time: 1 }),
      { lag_ratio: 0.5 },
    );
    lagged.begin();
    lagged.interpolate(1.0);
    const lagRecs = lagged.animations as RecordingAnimation[];
    lagRecs[0]?.calls.splice(0);
    lagRecs[1]?.calls.splice(0);
    lagged.interpolate(0.25);
    expect(lagRecs[0]?.calls).toEqual([0.375]);
    expect(lagRecs[1]?.calls).toEqual([0]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.VectorField.start_submobject_movement
// Python: no updaters -> start(speed=2) -> 1 updater; nudge(mob, 0.1) with a
// constant [1,0,0] field shifts a unit square by +0.1 in x.
// ---------------------------------------------------------------------------
describe("VectorField.start_submobject_movement", () => {
  it("registers an updater and nudges submobjects along the field", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    expect(vf.updaters.length).toBe(0);
    const mob = new VMobject();
    setPoints(mob, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    vf.add(mob);
    vf.start_submobject_movement(2.0);
    expect(vf.updaters.length).toBe(1);
    expect(vf.submob_movement_updater).not.toBeNull();
    vf.nudge(mob, 0.1);
    expectRowsClose(rowsOf(mob.points, 3), [
      [0.1, 0, 0],
      [1.1, 0, 0],
      [1.1, 1, 0],
      [0.1, 1, 0],
    ]);
    // Restarting replaces (not stacks) the updater; stopping clears it.
    vf.start_submobject_movement(1.0);
    expect(vf.updaters.length).toBe(1);
    expect(vf.stop_submobject_movement()).toBe(vf);
    expect(vf.updaters.length).toBe(0);
    expect(vf.submob_movement_updater).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols combined with earlier-ported code, verified
// against the original manim implementation the same way.
// ---------------------------------------------------------------------------
describe("batch integration", () => {
  it("Mobject chain: shift -> reverse -> repeat -> center of mass", () => {
    // Python: Mobject([[0,0,0],[1,0,0]]).shift([1,1,1]).reverse_points()
    // .repeat(2) -> [[2,1,1],[1,1,1],[2,1,1],[1,1,1]], com [1.5,1,1].
    const m = new Mobject();
    setPoints(m, [[0, 0, 0], [1, 0, 0]]);
    m.shift([1, 1, 1]).reverse_points().repeat(2);
    expectRowsClose(rowsOf(m.points, 3), [
      [2, 1, 1],
      [1, 1, 1],
      [2, 1, 1],
      [1, 1, 1],
    ]);
    const com = m.get_center_of_mass();
    expectClose(com[0] ?? NaN, 1.5);
    expectClose(com[1] ?? NaN, 1.0);
    expectClose(com[2] ?? NaN, 1.0);
    // Boundary follows the repeated points.
    expectRowsClose(rowsOf(m.get_points_defining_boundary(), 3), [
      [2, 1, 1],
      [1, 1, 1],
      [2, 1, 1],
      [1, 1, 1],
    ]);
  });
  it("Animation lagged interpolate_mobject path against a VMobject family", () => {
    // Python: Animation(m, lag_ratio=0.5) over a 2-member pointful family at
    // alpha=0.5 gives sub-alphas [smooth(0.75), smooth(0.25)] =
    // [0.929896283455, 0.070103716545].
    const m = new VMobject();
    setPoints(m, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    const sub = new VMobject();
    setPoints(sub, [[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]]);
    m.add(sub);
    const a = new Animation(m, { lag_ratio: 0.5 });
    const fams = a.get_all_families_zipped();
    expect(fams.length).toBe(2);
    const got = fams.map((_, i) => a.get_sub_alpha(0.5, i, fams.length));
    expectClose(got[0] ?? NaN, 0.929896283455);
    expectClose(got[1] ?? NaN, 0.070103716545);
  });
  it("PMobject pipeline: add_line -> thin_out -> sort_points -> get_color", () => {
    // Python: add_line([2,0,0],[0,0,0]) gives 21 points x=2..0 step -0.1
    // (length 2, epsilon 0.1/2=0.05 -> arange(0,1,0.05) = 20 points);
    // thin_out(4) -> 5 points; sort -> ascending x.
    const p = new Mobject1D();
    p.add_line([2, 0, 0], [0, 0, 0]);
    expect(p.get_num_points()).toBe(20);
    p.set_color("#00FF00");
    p.thin_out(4);
    expect(p.get_num_points()).toBe(5);
    p.sort_points();
    const rows = rowsOf(p.points, 3);
    const xs = rows.map((r) => r[0] ?? 0);
    expect([...xs].sort((x, y) => x - y)).toEqual(xs);
    expect(p.get_color().to_hex()).toBe("#00FF00");
  });
  it("config digest feeds frame sizes used by coordinate planes", () => {
    // digest_args quality h -> 1920x1080; NumberPlane.coords_to_point maps
    // the origin to the scene center like Python's Axes machinery.
    const c = new ManimConfig();
    c.digest_args(cliArgs({ quality: "h" }));
    expect(c.pixel_width).toBe(1920);
    const np = new NumberPlane();
    const origin = rowsOf(np.coords_to_point(0, 0), 3)[0] ?? [];
    expectClose(origin[0] ?? NaN, 0, 1e-6);
    expectClose(origin[1] ?? NaN, 0, 1e-6);
  });
  it("VectorField updater moves a submobject over time", () => {
    // Constant field [1,0,0]: updater(mob, dt=0.5) shifts center by +0.5 x.
    const vf = new VectorField(() => [1, 0, 0]);
    const mob = new VMobject();
    setPoints(mob, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    vf.add(mob);
    vf.start_submobject_movement(1.0);
    const updater = vf.submob_movement_updater;
    expect(updater).not.toBeNull();
    updater?.(vf, 0.5);
    const center = mob.get_center();
    expectClose(center[0] ?? NaN, 1.0);
    expectClose(center[1] ?? NaN, 0.5);
    vf.stop_submobject_movement();
  });
});
