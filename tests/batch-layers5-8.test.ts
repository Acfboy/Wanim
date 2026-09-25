// Ground-truth tests for the layers 5..8 batch (~1020 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used below; each
// case cites the probe in a "Python:" comment. TS is asserted against those
// values with tight float tolerance (absolute 1e-9 unless noted).
//
// Where no Line/Square classes exist yet in the TS port, a VMobject through
// explicit corners (or Rectangle) stands in for the Python shape; the
// boundary extrema coincide, so the ground truths transfer directly.
//
// Known deviations from Python (documented, not worked around):
//  - hashing JSON separators: Python json.dumps uses (', ', ': ') while the
//    TS port uses JSON.stringify (compact). encode() is compared via parsed
//    JSON; _hash_ndarray (compact separators in both) is asserted exactly.
//  - NumberLine geometry construction is not ported in TS (the constructor
//    builds no line/ticks: "minimal port"); point_to_number is tested with
//    manually supplied endpoints matching Python's computed start/end.
//  - BackgroundRectangle.stroke_color is None in Python (width-0 stroke
//    stores no color) but a ManimColor in TS; fill/stroke_width asserted.
//  - VMobject.set_style takes a single options object in TS (JS has no
//    **kwargs); semantics match Python's keyword arguments.
//  - match_style fill_opacity: Python stores the get_style() array ([0.3]);
//    TS normalises to a scalar; tests assert the numeric value.
//  - scene_classes_from_file: Python executes the module; TS get_module only
//    validates/locates the file (no code execution in the web runtime), so a
//    real scene file yields [] in TS. Validation errors are asserted exactly.
//  - random_color/random_bright_color use an unseeded singleton (Math.random
//    in TS); only the seeded RandomColorGenerator(42) sequence (which matches
//    Python exactly) plus structural properties are asserted.
//  - text/tex rendering is OUT OF SCOPE for the entire port; no tests here.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import {
  PMobject,
  Mobject1D,
  PointCloudDot,
  Point,
} from "../src/mobject/types/point_cloud_mobject.js";
import {
  Arc,
  Circle,
  Dot,
  AnnotationDot,
  AnnularSector,
  Sector,
  Annulus,
  TipableVMobject,
} from "../src/mobject/geometry/arc.js";
import { Rectangle, RegularPolygram } from "../src/mobject/geometry/polygram.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { ArrowTip } from "../src/mobject/geometry/tips.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { VMobjectFromSVGPath } from "../src/mobject/svg/svg_mobject.js";
import { wiggle } from "../src/utils/rate_functions.js";
import {
  _CustomEncoder,
  _Memoizer,
  _hash_ndarray,
} from "../src/utils/hash.js";
import {
  get_module,
  scene_classes_from_file,
} from "../src/utils/module_ops.js";
import {
  ManimColor,
  color_gradient,
  average_color,
  random_color,
  random_bright_color,
  color_to_rgb,
  RandomColorGenerator,
} from "../src/utils/color/core.js";
import { config } from "../src/_config/utils.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function flat(value: NDArray | number[] | number[][]): number[] {
  if (Array.isArray(value)) return (value as unknown[]).flat(Infinity as number[]).map(Number);
  return ((value as NDArray).toArray() as unknown[]).flat(Infinity as number[]).map(Number);
}

function mat(value: NDArray): number[][] {
  return (value.toArray() as unknown[][]).map((row) =>
    (Array.isArray(row) ? row : [row]).map(Number),
  );
}

function expectNums(actual: NDArray | number[] | number[][], expected: readonly number[], tol = 1e-9): void {
  const values = flat(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(Math.abs((values[i] ?? 0) - (expected[i] ?? 0))).toBeLessThanOrEqual(tol);
  }
}

function num(value: unknown): number {
  if (typeof value === "number") return value;
  return flat(value as NDArray)[0] ?? Number.NaN;
}

function hexOf(color: unknown): string {
  if (color instanceof ManimColor) return color.to_hex();
  if (Array.isArray(color)) return hexOf(color[0]);
  if (color !== null && typeof color === "object" && "to_hex" in color) {
    return ((color as { to_hex: () => string }).to_hex as () => string)();
  }
  return String(color);
}

/** A 2x2 square centred on the origin, mirroring Python Square(side_length=2). */
function square2(): VMobject {
  const v = new VMobject();
  v.set_points_as_corners([
    [-1, -1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
  ]);
  return v;
}

// ---------------------------------------------------------------------------
// Mobject.get_extremum_along_dim
// Python: Square(2) dim0 key1 -> 1.0; key-1 -> -1.0; key0 -> 0.0.
// ---------------------------------------------------------------------------
describe("Mobject.get_extremum_along_dim", () => {
  it("returns min/mid/max along a dimension", () => {
    const v = square2();
    expect(v.get_extremum_along_dim(null, 0, 1)).toBeCloseTo(1.0, 12);
    expect(v.get_extremum_along_dim(null, 0, -1)).toBeCloseTo(-1.0, 12);
    expect(v.get_extremum_along_dim(null, 0, 0)).toBeCloseTo(0.0, 12);
    expect(v.get_extremum_along_dim(null, 1, 1)).toBeCloseTo(1.0, 12);
  });
  it("honours an explicit points array", () => {
    // Python: get_extremum_along_dim(points=[[0,5,0],[2,7,0]], dim=1, key=1) -> 7.
    const v = new VMobject();
    expect(v.get_extremum_along_dim([[0, 5, 0], [2, 7, 0]], 1, 1)).toBeCloseTo(7.0, 12);
    expect(v.get_extremum_along_dim([[0, 5, 0], [2, 7, 0]], 1, -1)).toBeCloseTo(5.0, 12);
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_boundary_point
// Python: Square(2).get_boundary_point(UP) -> [1, 1, 0] (argmax of dot).
// ---------------------------------------------------------------------------
describe("Mobject.get_boundary_point", () => {
  it("returns the argmax-dot boundary point", () => {
    expectNums(square2().get_boundary_point([0, 1, 0]), [1, 1, 0]);
    expectNums(square2().get_boundary_point([-1, 0, 0]), [-1, -1, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_start / get_end / get_start_and_end
// Python: Line([-1,0,0],[2,1,0]) start [-1,0,0], end [2,1,0].
// ---------------------------------------------------------------------------
describe("Mobject.get_start/get_end/get_start_and_end", () => {
  it("returns stroke endpoints", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [-1, 0, 0],
      [2, 1, 0],
    ]);
    expectNums(v.get_start(), [-1, 0, 0]);
    expectNums(v.get_end(), [2, 1, 0]);
    const [s, e] = v.get_start_and_end();
    expectNums(s, [-1, 0, 0]);
    expectNums(e, [2, 1, 0]);
  });
  it("throws on an empty mobject", () => {
    expect(() => new VMobject().get_start()).toThrow();
    expect(() => new VMobject().get_end()).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_coord / get_x / get_y / get_z
// Python: Square(2): get_x/get_y/get_z -> 0.0; get_coord(0, [1,0,0]) -> 1.0.
// ---------------------------------------------------------------------------
describe("Mobject.get_coord/get_x/get_y/get_z", () => {
  it("generalises the axis getters", () => {
    const v = square2();
    expect(v.get_x()).toBeCloseTo(0.0, 12);
    expect(v.get_y()).toBeCloseTo(0.0, 12);
    expect(v.get_z()).toBeCloseTo(0.0, 12);
    expect(v.get_coord(0, [1, 0, 0])).toBeCloseTo(1.0, 12);
    expect(v.get_x([1, 0, 0])).toBeCloseTo(1.0, 12);
    expect(v.get_y([0, -1, 0])).toBeCloseTo(-1.0, 12);
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_critical_point / get_center / get_corner / get_edge_center
// Python: Square(2): crit(UP) [0,1,0]; center [0,0,0]; corner(UR) [1,1,0];
// corner(DL) [-1,-1,0]; edge(UP) [0,1,0].
// ---------------------------------------------------------------------------
describe("Mobject critical points", () => {
  it("get_critical_point resolves each box point", () => {
    const v = square2();
    expectNums(v.get_critical_point([0, 1, 0]), [0, 1, 0]);
    expectNums(v.get_center(), [0, 0, 0]);
    expectNums(v.get_corner([1, 1, 0]), [1, 1, 0]);
    expectNums(v.get_corner([-1, -1, 0]), [-1, -1, 0]);
    expectNums(v.get_edge_center([0, 1, 0]), [0, 1, 0]);
  });
  it("returns zeros for an empty mobject", () => {
    // Python: get_critical_point on no points -> zeros(dim).
    expectNums(new VMobject().get_critical_point([0, 1, 0]), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_coord
// Python: Square(2).set_coord(5, 0) -> center [5, 0, 0].
// ---------------------------------------------------------------------------
describe("Mobject.set_coord", () => {
  it("shifts one axis to the target value", () => {
    const v = square2();
    v.set_coord(5, 0);
    expectNums(v.get_center(), [5, 0, 0]);
    expect(v.get_extremum_along_dim(null, 0, 1)).toBeCloseTo(6.0, 12);
  });
});

// ---------------------------------------------------------------------------
// Mobject.apply_points_function_about_point
// Python: Square(2) scaled x2 about origin -> center [0,0,0], width 4.0.
// ---------------------------------------------------------------------------
describe("Mobject.apply_points_function_about_point", () => {
  it("applies func about the critical point", () => {
    const v = square2();
    v.apply_points_function_about_point((pts: NDArray) => {
      const rows = mat(pts);
      return array(rows.map((r) => r.map((x) => x * 2))) as NDArray;
    });
    expectNums(v.get_center(), [0, 0, 0]);
    expect(v.width).toBeCloseTo(4.0, 12);
  });
  it("defaults about_edge to ORIGIN", () => {
    // Python: about_point=None, about_edge=None -> critical point of ORIGIN.
    const v = square2();
    v.apply_points_function_about_point((pts: NDArray) => pts, null, null);
    expectNums(v.get_center(), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.move_to
// Python: Dot().move_to([3,2,0]) -> center [3,2,0].
// ---------------------------------------------------------------------------
describe("Mobject.move_to", () => {
  it("moves the center to a point", () => {
    const d = new Dot([0, 0, 0]);
    d.move_to([3, 2, 0]);
    expectNums(d.get_center(), [3, 2, 0]);
  });
  it("moves onto another mobject's critical point", () => {
    // Python: VGroup chain r1.move_to([3,2,0]) -> [3,2,0]; move_to(mob) aligns.
    const a = square2();
    const b = square2();
    b.move_to([10, 0, 0]);
    a.move_to(b);
    expectNums(a.get_center(), [10, 0, 0]);
  });
  it("honours aligned_edge", () => {
    // Python: b at [10,0,0]; a.move_to(b, [1,0,0]) -> a center [10,0,0]
    // (a's right edge lands on b's right edge: both at x=11).
    const a = square2();
    const b = square2();
    b.move_to([10, 0, 0]);
    a.move_to(b, [1, 0, 0]);
    expectNums(a.get_center(), [10, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.next_to
// Python: Square(1) b.next_to(a, RIGHT, buff=0.5) -> b center [1.5,0,0].
// Scaled here to the +/-1 square: b center [2.5, 0, 0].
// ---------------------------------------------------------------------------
describe("Mobject.next_to", () => {
  it("places the mobject beside another with buff", () => {
    const a = square2();
    const b = square2();
    b.next_to(a, [1, 0, 0], 0.5);
    expectNums(b.get_center(), [2.5, 0, 0]);
    expectNums(a.get_center(), [0, 0, 0]);
  });
  it("accepts a raw point target", () => {
    const b = square2();
    b.next_to([5, 0, 0], [1, 0, 0], 0);
    // point_to_align = b left edge [-1,0,0]; shift = 5-(-1) = 6 -> center 6.
    expectNums(b.get_center(), [6, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.align_on_border / to_edge
// Python: Square(2).to_edge(LEFT, buff=0) -> [-6.111111111111111,0,0];
// to_edge(UP) (default buff 0.5) -> [0,2.5,0]; frame radii (7.111.., 4.0).
// ---------------------------------------------------------------------------
describe("Mobject.align_on_border/to_edge", () => {
  it("aligns to the frame border", () => {
    expect(config.frame_x_radius).toBeCloseTo(7.111111111111111, 12);
    expect(config.frame_y_radius).toBeCloseTo(4.0, 12);
    const left = new Rectangle({ width: 2, height: 2 });
    left.to_edge([-1, 0, 0], 0);
    expectNums(left.get_center(), [-6.111111111111111, 0, 0]);
    const up = new Rectangle({ width: 2, height: 2 });
    up.to_edge([0, 1, 0]);
    expectNums(up.get_center(), [0, 2.5, 0]);
  });
  it("align_on_border matches to_edge", () => {
    const a = new Rectangle({ width: 2, height: 2 });
    a.align_on_border([-1, 0, 0], 0);
    expectNums(a.get_center(), [-6.111111111111111, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_color_by_gradient / set_submobject_colors_by_gradient
// Python: VGroup(3xSquare).set_submobject_colors_by_gradient(RED,BLUE) ->
// ['#FF0000','#7F007F','#0000FF']; 0 colors raises; 1 color -> set_color.
// ---------------------------------------------------------------------------
describe("Mobject gradient colors", () => {
  it("spreads a gradient across family members with points", () => {
    const g = new VGroup(
      new Rectangle({ width: 1, height: 1 }),
      new Rectangle({ width: 1, height: 1 }),
      new Rectangle({ width: 1, height: 1 }),
    );
    g.set_submobject_colors_by_gradient("#FF0000", "#0000FF");
    expect(g.submobjects.map((m) => hexOf((m as VMobject).fill_color))).toEqual([
      "#FF0000",
      "#7F007F",
      "#0000FF",
    ]);
  });
  it("set_color_by_gradient delegates to the submobject gradient", () => {
    const g = new VGroup(
      new Rectangle({ width: 1, height: 1 }),
      new Rectangle({ width: 1, height: 1 }),
    );
    g.set_color_by_gradient("#FF0000", "#0000FF");
    expect(g.submobjects.map((m) => hexOf((m as VMobject).fill_color))).toEqual([
      "#FF0000",
      "#0000FF",
    ]);
  });
  it("raises with no colors and sets a single color directly", () => {
    expect(() => new VGroup().set_submobject_colors_by_gradient()).toThrow();
    const g = new VGroup(new Rectangle({ width: 1, height: 1 }));
    g.set_submobject_colors_by_gradient("#FF0000");
    expect(hexOf((g.submobjects[0] as VMobject).fill_color)).toBe("#FF0000");
  });
});

// ---------------------------------------------------------------------------
// VMobject.fade
// Python: Square fill 0.8 / stroke 0.6, fade(0.5) -> 0.4 / 0.3.
// ---------------------------------------------------------------------------
describe("VMobject.fade", () => {
  it("multiplies fill and stroke opacity by (1-darkness)", () => {
    const v = new VMobject();
    v.set_fill(undefined, 0.8);
    v.set_stroke(undefined, undefined, 0.6);
    v.fade(0.5);
    expect(v.get_fill_opacity()).toBeCloseTo(0.4, 12);
    expect(v.get_stroke_opacity()).toBeCloseTo(0.3, 12);
  });
  it("defaults to darkness 0.5", () => {
    const v = new VMobject();
    v.set_fill(undefined, 1.0);
    v.fade();
    expect(v.get_fill_opacity()).toBeCloseTo(0.5, 12);
  });
});

// ---------------------------------------------------------------------------
// VMobject arc-length family
// Python: VMobject corners [[0,0,0],[3,4,0]]: arc 5.0, nth(0) 5.0,
// 1 curve function of length 5.0; pfp(0.5) [1.5,2,0]; pfp(1) end;
// proportion_from_point(mid) 0.5; errors on alpha>1 / off-curve point.
// ---------------------------------------------------------------------------
describe("VMobject arc length / proportion", () => {
  function segment(): VMobject {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [3, 4, 0],
    ]);
    return v;
  }
  it("get_arc_length sums curve lengths", () => {
    expect(segment().get_arc_length()).toBeCloseTo(5.0, 9);
  });
  it("get_nth_curve_length returns the single curve length", () => {
    expect(segment().get_nth_curve_length(0)).toBeCloseTo(5.0, 9);
  });
  it("get_curve_functions_with_lengths yields one (fn, 5.0)", () => {
    const fns = [...segment().get_curve_functions_with_lengths()];
    expect(fns).toHaveLength(1);
    expect(fns[0]?.[1]).toBeCloseTo(5.0, 9);
    expectNums((fns[0]?.[0] as (t: number) => NDArray)(0.5), [1.5, 2.0, 0]);
  });
  it("point_from_proportion walks the path", () => {
    expectNums(segment().point_from_proportion(0.5), [1.5, 2.0, 0]);
    expectNums(segment().point_from_proportion(1), [3, 4, 0]);
    expectNums(segment().point_from_proportion(0.25), [0.75, 1.0, 0]);
  });
  it("point_from_proportion rejects alpha outside [0,1]", () => {
    expect(() => segment().point_from_proportion(1.5)).toThrowError(
      "Alpha 1.5 not between 0 and 1.",
    );
  });
  it("proportion_from_point inverts point_from_proportion", () => {
    expect(segment().proportion_from_point([1.5, 2, 0])).toBeCloseTo(0.5, 6);
  });
  it("proportion_from_point rejects off-curve points", () => {
    expect(() => segment().proportion_from_point([9, 9, 0])).toThrowError(
      "does not lie on this curve.",
    );
  });
});

// ---------------------------------------------------------------------------
// VMobject.make_smooth / make_jagged
// Python: Line corners smooth -> handles [1/3,0,0],[2/3,0,0]; jagged keeps
// the straight-line look (same anchors); both round-trip on a Square.
// ---------------------------------------------------------------------------
describe("VMobject.make_smooth/make_jagged", () => {
  it("smooths a straight segment into cubic handles", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    v.make_smooth();
    expectNums(mat(v.points)[1] ?? [], [1 / 3, 0, 0]);
    expectNums(mat(v.points)[2] ?? [], [2 / 3, 0, 0]);
  });
  it("jagged collapses handles back onto the line", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    v.make_smooth();
    v.make_jagged();
    for (const row of mat(v.points)) {
      expect(row[1]).toBeCloseTo(0.0, 12);
      expect(row[2]).toBeCloseTo(0.0, 12);
    }
  });
  it("round-trips a closed square", () => {
    // Python: Square().make_smooth()/make_jagged() keep 4 curves.
    const before = square2().get_num_curves();
    const v = square2();
    v.make_smooth();
    expect(v.get_num_curves()).toBe(before);
    v.make_jagged();
    expect(v.get_num_curves()).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// VMobject.init_colors / set_style / match_style
// Python: VMobject() defaults fill_op 0, stroke_op 1, stroke_w 4,
// fill WHITE; set_style(RED,0.3,sw 7); match copies fill #FF0000 op ~0.3 w 7.
// ---------------------------------------------------------------------------
describe("VMobject colors and style", () => {
  it("has Python's construction defaults", () => {
    const v = new VMobject();
    expect(v.get_fill_opacity()).toBeCloseTo(0.0, 12);
    expect(v.get_stroke_opacity()).toBeCloseTo(1.0, 12);
    expect(v.stroke_width).toBe(4);
    expect(hexOf(v.fill_color)).toBe("#FFFFFF");
  });
  it("init_colors reapplies stored colors and returns self", () => {
    const v = new VMobject();
    v.set_fill("#FF0000", 0.3);
    expect(v.init_colors()).toBe(v);
    expect(hexOf(v.fill_color)).toBe("#FF0000");
    expect(v.get_fill_opacity()).toBeCloseTo(0.3, 12);
  });
  it("set_style applies fill/stroke options", () => {
    const v = new Rectangle({ width: 1, height: 1 });
    v.set_style({ fill_color: "#FF0000", fill_opacity: 0.3, stroke_width: 7 });
    expect(hexOf(v.fill_color)).toBe("#FF0000");
    expect(v.get_fill_opacity()).toBeCloseTo(0.3, 12);
    expect(v.stroke_width).toBe(7);
  });
  it("match_style copies another vmobject's style", () => {
    const s = new Rectangle({ width: 1, height: 1 });
    s.set_style({ fill_color: "#FF0000", fill_opacity: 0.3, stroke_width: 7 });
    const t = new Rectangle({ width: 1, height: 1 });
    t.match_style(s);
    expect(hexOf(t.fill_color)).toBe("#FF0000");
    expect(num(t.fill_opacity)).toBeCloseTo(0.3, 9);
    expect(t.stroke_width).toBe(7);
  });
  it("get_style exposes the documented keys", () => {
    // Python keys: background_image, background_stroke_{color,width,opacity},
    // fill_color, fill_opacity, sheen_{direction,factor}, stroke_{color,opacity,width}.
    const keys = Object.keys(new VMobject().get_style()).sort();
    expect(keys).toEqual(
      [
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
      ].sort(),
    );
  });
});

// ---------------------------------------------------------------------------
// VMobject.color_using_background_image / match_background_image
// Python: stores the image object, sets fill WHITE, recurses to submobjects;
// match_background_image copies the source's image.
// ---------------------------------------------------------------------------
describe("VMobject background image", () => {
  it("color_using_background_image stores the image and whitens", () => {
    const v = new Rectangle({ width: 1, height: 1 });
    v.set_fill("#FF0000", 1);
    v.color_using_background_image("fake-image.png");
    expect(v.background_image).toBe("fake-image.png");
    expect(hexOf(v.fill_color)).toBe("#FFFFFF");
  });
  it("match_background_image copies the source image", () => {
    const src = new Rectangle({ width: 1, height: 1 });
    src.color_using_background_image("fake-image.png");
    const dst = new Rectangle({ width: 1, height: 1 });
    dst.match_background_image(src);
    expect(dst.background_image).toBe("fake-image.png");
    expect(hexOf(dst.fill_color)).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// PointCloudDot / Point / PMobject / Mobject1D
// Python: PointCloudDot() 1303 pts, pts[0] [0.1,0,0];
//   PointCloudDot([1,2,0],0.5) 86 pts, center [1,2,0];
//   Point([4,5,0]).points [[4,5,0]];
//   PMobject 3 pts gradient -> rows [1,0,0,1],[0.5,0,0.5,1],[0,0,1,1].
// ---------------------------------------------------------------------------
describe("point cloud mobjects", () => {
  it("PointCloudDot defaults match Python's disc sampling", () => {
    const d = new PointCloudDot();
    expect(mat(d.points)).toHaveLength(1303);
    expectNums(mat(d.points)[0] ?? [], [0.1, 0, 0]);
    expect(d.init_points()).toBe(d);
    expect(mat(d.points)).toHaveLength(1303);
  });
  it("PointCloudDot centres the disc on the given center", () => {
    const d = new PointCloudDot([1, 2, 0], 0.5);
    expect(mat(d.points)).toHaveLength(86);
    expectNums(d.get_center(), [1, 2, 0]);
    expect(d.radius).toBeCloseTo(0.5, 12);
  });
  it("Point.init_points collapses to the location", () => {
    const p = new Point([4, 5, 0]);
    expect(mat(p.points)).toEqual([[4, 5, 0]]);
    expect(p.init_points()).toBe(p);
    expect(mat(p.points)).toEqual([[4, 5, 0]]);
  });
  it("PMobject.set_color_by_gradient writes per-point rgbas", () => {
    const pm = new PMobject();
    pm.add_points([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
    ]);
    pm.set_color_by_gradient("#FF0000", "#0000FF");
    expect(mat(pm.rgbas)).toEqual([
      [1, 0, 0, 1],
      [0.5, 0, 0.5, 1],
      [0, 0, 1, 1],
    ]);
  });
  it("Mobject1D exposes point_from_proportion", () => {
    // Python: Mobject1D is a PMobject; smoke-test the shared helper exists.
    const d = new PointCloudDot([0, 0, 0], 0.5);
    expect(d).toBeInstanceOf(Mobject1D);
    expect(typeof (d as unknown as { point_from_proportion: unknown }).point_from_proportion).toBe(
      "function",
    );
  });
});

// ---------------------------------------------------------------------------
// Arc family: Circle / Dot / AnnotationDot / Sector / Annulus / AnnularSector
// Python: Circle(2) width 4, 32 pts; point_at_angle(pi/2) ~[0,2,0];
//   from_three_points([-1,0,0],[0,1,0],[1,0,0]) center [0,0,0] r 1;
//   Dot(0.5) width 1 fill 1; AnnotationDot width 0.208 stroke 5;
//   Annulus() (1,2,False,op 1,sw 0); Sector() angle pi/2 inner 0 outer 1;
//   Arc(1,0,pi/2).get_length() sqrt(2) = 1.414213562373095.
// ---------------------------------------------------------------------------
describe("arc family constructors", () => {
  it("Circle stores radius and builds 32 bezier points", () => {
    const c = new Circle(2.0);
    expect(c.radius).toBeCloseTo(2.0, 12);
    expect([...c.points.shape]).toEqual([32, 3]);
    expect(c.width).toBeCloseTo(4.0, 9);
  });
  it("Circle.point_at_angle lands on the circumference", () => {
    // Python: Circle(2).point_at_angle(pi/2) -> [2.99e-16, 2.0, 0.0].
    expectNums(new Circle(2.0).point_at_angle(Math.PI / 2), [0, 2, 0]);
    expectNums(new Circle(2.0).point_at_angle(0), [2, 0, 0]);
  });
  it("Circle.from_three_points recovers center and radius", () => {
    const c = Circle.from_three_points([-1, 0, 0], [0, 1, 0], [1, 0, 0]);
    expectNums(c.get_center(), [0, 0, 0]);
    expect(c.radius).toBeCloseTo(1.0, 9);
  });
  it("Dot defaults mirror Python (r=0.08, sw=0, fill 1)", () => {
    const d = new Dot([1, 0, 0]);
    expectNums(d.get_center(), [1, 0, 0]);
    expect(d.width).toBeCloseTo(0.16, 9);
    expect(d.stroke_width).toBe(0);
    expect(d.get_fill_opacity()).toBeCloseTo(1.0, 12);
    const half = new Dot([0, 0, 0], 0.5);
    expect(half.width).toBeCloseTo(1.0, 9);
  });
  it("AnnotationDot has the enlarged stroked style", () => {
    // Python: width 0.208, stroke 5, fill #58C4DD (BLUE), stroke WHITE.
    const a = new AnnotationDot();
    expect(a.width).toBeCloseTo(0.208, 9);
    expect(a.stroke_width).toBe(5);
    expect(hexOf(a.fill_color)).toBe("#58C4DD");
    expect(hexOf(a.stroke_color)).toBe("#FFFFFF");
  });
  it("Annulus defaults mirror Python", () => {
    const an = new Annulus();
    expect(an.inner_radius).toBe(1);
    expect(an.outer_radius).toBe(2);
    expect(an.mark_paths_closed).toBe(false);
    expect(an.get_fill_opacity()).toBeCloseTo(1.0, 12);
    expect(an.stroke_width).toBe(0);
    expect(an.width).toBeCloseTo(4.0, 9);
  });
  it("Sector defaults mirror Python (angle pi/2, inner 0, outer 1)", () => {
    const s = new Sector();
    expect(s.angle).toBeCloseTo(Math.PI / 2, 12);
    expect(s.inner_radius).toBe(0);
    expect(s.outer_radius).toBe(1);
    expect(s.get_arc_length()).toBeGreaterThan(0);
  });
  it("AnnularSector builds a non-degenerate ring segment", () => {
    const a = new AnnularSector(1, 2, Math.PI / 2);
    expect(a.inner_radius).toBe(1);
    expect(a.outer_radius).toBe(2);
    expect(a.get_arc_length()).toBeGreaterThan(0);
  });
  it("TipableVMobject.get_length is the chord length", () => {
    const arc = new Arc(1, 0, Math.PI / 2);
    expect(arc.get_length()).toBeCloseTo(1.414213562373095, 9);
    expect(arc).toBeInstanceOf(TipableVMobject);
  });
});

// ---------------------------------------------------------------------------
// RegularPolygram gen_polygon_vertices (inner closure, tested via constructor)
// Python: RegularPolygram(5, radius=2): 5 curves, center y 0.19098300562505233;
//   RegularPolygram(6, density=2): 6 curves (2 triangles, 0 submobjects).
// ---------------------------------------------------------------------------
describe("RegularPolygram vertex generation", () => {
  it("walks every density-th vertex of the pentagram", () => {
    const p = new RegularPolygram(5, { radius: 2 });
    expect(p.get_num_curves()).toBe(5);
    expectNums(p.get_center(), [0, 0.19098300562505233, 0]);
  });
  it("splits {6/2} into two triangle groups", () => {
    const p = new RegularPolygram(6, { density: 2, radius: 1 });
    expect(p.get_num_curves()).toBe(6);
    expect(p.submobjects).toHaveLength(0);
  });
  it("density 1 gives the plain hexagon", () => {
    const p = new RegularPolygram(6, { density: 1, radius: 1 });
    expect(p.get_num_curves()).toBe(6);
    expectNums(p.get_center(), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// VMobjectFromSVGPath.generate_points / init_points
// (Segment-level geometry covered in batch-layer0; here: the batch symbols.)
// ---------------------------------------------------------------------------
describe("VMobjectFromSVGPath points", () => {
  it("generate_points populates cubic points and returns self", () => {
    // Mirrors batch-layer0's ground truth: M 0 0 L 3 4 -> 4 cubic points.
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [3, 4] },
    ]);
    expect(v.generate_points()).toBe(v);
    expect([...v.points.shape]).toEqual([4, 3]);
    expectNums(v.points, [0, 0, 0, 1, 1.3333333333333333, 0, 2, 2.6666666666666665, 0, 3, 4, 0]);
  });
  it("init_points regenerates and returns self", () => {
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [3, 4] },
    ]);
    expect(v.init_points()).toBe(v);
    expect([...v.points.shape]).toEqual([4, 3]);
  });
  it("stores path flags on construction", () => {
    const v = new VMobjectFromSVGPath([{ kind: "Move", end: [0, 0] }], {
      long_lines: true,
    });
    expect(v.long_lines).toBe(true);
    expect(v.should_subdivide_sharp_curves).toBe(false);
    expect(v.should_remove_null_curves).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing._CustomEncoder
// Python: _cleaned_iterable passes lists/dicts through (memoised);
//   circular [1,2,self] encodes to [1,2,"AP"]; tuple -> list;
//   non-iterable raises TypeError('iterable' is neither ...);
//   default(uint8(7)) -> 7; encode({'x':[1,2]}) -> '{"x": [1, 2]}'.
// ---------------------------------------------------------------------------
describe("_CustomEncoder", () => {
  it("_cleaned_iterable passes plain lists and dicts through", () => {
    const enc = new _CustomEncoder(new _Memoizer());
    expect(enc._cleaned_iterable([1, [2, 3], { a: 4 }])).toEqual([1, [2, 3], { a: 4 }]);
  });
  it("_cleaned_iterable replaces circular refs with AP", () => {
    // Python: e.encode(lst) where lst contains lst -> '[1, 2, "AP"]'.
    const enc = new _CustomEncoder(new _Memoizer());
    const lst: unknown[] = [1, 2];
    lst.push(lst);
    expect(JSON.parse(enc.encode(lst))).toEqual([1, 2, "AP"]);
  });
  it("_cleaned_iterable rejects non-iterables", () => {
    const enc = new _CustomEncoder(new _Memoizer());
    expect(() => enc._cleaned_iterable(42 as never)).toThrowError(
      "'iterable' is neither an iterable nor a dictionary.",
    );
  });
  it("default passes small ints through like np.uint8", () => {
    // Python: default(np.uint8(7)) -> 7.
    expect(new _CustomEncoder(new _Memoizer()).default(7)).toBe(7);
  });
  it("default serialises functions to {code, nonlocals}", () => {
    // Python returns {'code': <source>, 'nonlocals': {...}}; TS uses
    // Function.toString since Python source is unavailable at runtime.
    function sample(): number {
      return 1;
    }
    const out = new _CustomEncoder(new _Memoizer()).default(sample) as Record<string, unknown>;
    expect(Object.keys(out).sort()).toEqual(["code", "nonlocals"]);
    expect(out["nonlocals"]).toEqual({});
    expect(String(out["code"])).toContain("return 1");
  });
  it("default hashes float arrays to the NDARRAY form", () => {
    // Python: default(array([1.0,2.0])) ->
    // 'NDARRAY:"<f8":(2,):dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae'.
    expect(new _CustomEncoder(new _Memoizer()).default(array([1.0, 2.0]))).toBe(
      'NDARRAY:"<f8":(2,):dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae',
    );
  });
  it("encode serialises a dict (compact separators in TS)", () => {
    // Python emits '{"x": [1, 2]}' (', '/': ' separators); TS uses
    // JSON.stringify, so compare parsed.
    expect(JSON.parse(new _CustomEncoder(new _Memoizer()).encode({ x: [1, 2] }))).toEqual({
      x: [1, 2],
    });
  });
});

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.wiggle
// Python: wiggle(0)=0, wiggle(0.25)=0.5, wiggle(0.5)=1.22e-16, wiggle(1)=-0.0;
//   outside [0,1] -> 0 (zero decorator); wiggle(0.25, wiggles=3)=0.3535533905932738.
// ---------------------------------------------------------------------------
describe("wiggle", () => {
  it("matches Python at sample points", () => {
    expect(wiggle(0.0)).toBeCloseTo(0.0, 12);
    expect(wiggle(0.25)).toBeCloseTo(0.5, 9);
    expect(Math.abs(wiggle(0.5))).toBeLessThanOrEqual(1e-9);
    expect(wiggle(1.0)).toBeCloseTo(0.0, 12);
  });
  it("is zero outside [0,1]", () => {
    expect(wiggle(-0.5)).toBe(0);
    expect(wiggle(1.5)).toBe(0);
  });
  it("honours the wiggles parameter", () => {
    expect(wiggle(0.25, 3)).toBeCloseTo(0.3535533905932738, 9);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.module_ops.scene_classes_from_file
// Python executes the file and returns Scene subclasses; TS get_module only
// locates/validates (no code execution in the web runtime).
// ---------------------------------------------------------------------------
describe("scene_classes_from_file", () => {
  function sceneFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "wanim-modops-"));
    const file = join(dir, "my_scene.py");
    writeFileSync(file, "from manim import Scene\nclass A(Scene):\n    def construct(self): pass\n");
    return file;
  }
  it("get_module validates suffix and existence", () => {
    const mod = get_module(sceneFile());
    expect(mod.__file__.endsWith(".py")).toBe(true);
    expect(() => get_module("/nonexistent/scene.py")).toThrowError(/not found/);
    const dir = mkdtempSync(join(tmpdir(), "wanim-modops-"));
    const bad = join(dir, "scene.txt");
    writeFileSync(bad, "x = 1");
    expect(() => get_module(bad)).toThrowError(/not a valid Manim python script/);
  });
  it("full_list returns the (unexecuted) module's scene classes", () => {
    // Deviation: Python returns ['A'] after executing the file; TS cannot
    // execute Python, so the module record holds no classes -> [].
    expect(scene_classes_from_file(sceneFile(), false, true)).toEqual([]);
  });
  it("require_single_scene asserts on the empty render list", () => {
    expect(() => scene_classes_from_file(sceneFile(), true, false)).toThrowError(/single scene/);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color: color_gradient / average_color / random_color
// Python: grad(RED,BLUE,3) ['#FF0000','#7F007F','#0000FF'];
//   grad(R,G,B,5) ['#FF0000','#7F7F00','#00FF00','#007F7F','#0000FF'];
//   grad len 0 -> []; single color repeats; empty raises ValueError;
//   average(RED,BLUE) #7F007F; RandomColorGenerator(42) #8B4513,#BBBBBB,#BBBBBB.
// ---------------------------------------------------------------------------
describe("color utils", () => {
  it("color_gradient interpolates between reference colors", () => {
    expect(color_gradient(["#ff0000", "#0000ff"], 3).map((c) => c.to_hex())).toEqual([
      "#FF0000",
      "#7F007F",
      "#0000FF",
    ]);
    expect(
      color_gradient(["#ff0000", "#00ff00", "#0000ff"], 5).map((c) => c.to_hex()),
    ).toEqual(["#FF0000", "#7F7F00", "#00FF00", "#007F7F", "#0000FF"]);
  });
  it("color_gradient handles edge lengths", () => {
    expect(color_gradient(["#ff0000"], 0)).toEqual([]);
    expect(color_gradient(["#ff0000"], 4).map((c) => c.to_hex())).toEqual([
      "#FF0000",
      "#FF0000",
      "#FF0000",
      "#FF0000",
    ]);
    expect(() => color_gradient([], 3)).toThrowError(
      "Expected 1 or more reference colors. Got 0 colors.",
    );
  });
  it("average_color means the rgb channels", () => {
    expect(average_color("#ff0000", "#0000ff").to_hex()).toBe("#7F007F");
  });
  it("seeded RandomColorGenerator matches Python's sequence", () => {
    // Python: RandomColorGenerator(42).next() x3 -> #8B4513, #BBBBBB, #BBBBBB.
    const g = new RandomColorGenerator(42);
    expect(g.next().to_hex()).toBe("#8B4513");
    expect(g.next().to_hex()).toBe("#BBBBBB");
    expect(g.next().to_hex()).toBe("#BBBBBB");
    expect(new RandomColorGenerator(42).next().to_hex()).toBe("#8B4513");
  });
  it("random_color returns a palette ManimColor", () => {
    expect(random_color()).toBeInstanceOf(ManimColor);
  });
  it("random_bright_color averages a color with white", () => {
    // new_rgb = 0.5 * (rgb + 1) component-wise, so every channel >= 0.5.
    const bright = random_bright_color();
    expect(bright).toBeInstanceOf(ManimColor);
    for (const channel of flat(color_to_rgb(bright))) {
      expect(channel).toBeGreaterThanOrEqual(0.5 - 1e-12);
      expect(channel).toBeLessThanOrEqual(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// NumberLine.point_to_number
// Python: NumberLine(x_range=[-2,2,1], length=4): start [-2,0,0], end [2,0,0];
//   p2n([0,0,0]) 0.0; p2n([2,0,0]) 2.0; p2n([[0.5,0,0],[1,0,0]]) [0.5,1].
// TS builds no NumberLine geometry yet, so the Python-computed endpoints are
// supplied as points and only the projection math is asserted.
// ---------------------------------------------------------------------------
describe("NumberLine.point_to_number", () => {
  function line(): NumberLine {
    const nl = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    nl.set_points([
      [-2, 0, 0],
      [2, 0, 0],
    ]);
    return nl;
  }
  it("projects scene points onto the number range", () => {
    const nl = line();
    expect(nl.point_to_number([0, 0, 0])).toBeCloseTo(0.0, 9);
    expect(nl.point_to_number([2, 0, 0])).toBeCloseTo(2.0, 9);
    expect(nl.point_to_number([-2, 0, 0])).toBeCloseTo(-2.0, 9);
  });
  it("maps batches of points element-wise", () => {
    const out = line().point_to_number([
      [0.5, 0, 0],
      [1, 0, 0],
    ]) as NDArray;
    expectNums(out, [0.5, 1.0]);
  });
});

// ---------------------------------------------------------------------------
// BackgroundRectangle.set_style
// Python: fill BLACK op 0.75 sw 0; set_style(0.7, stroke_width=99 ignored)
// -> op 0.7, sw 0, stroke None (TS keeps a BLACK ManimColor; documented).
// ---------------------------------------------------------------------------
describe("BackgroundRectangle.set_style", () => {
  it("only fill_opacity is changeable", () => {
    const bg = new BackgroundRectangle(new Rectangle({ width: 2, height: 2 }));
    expect(hexOf(bg.fill_color)).toBe("#000000");
    expect(bg.get_fill_opacity()).toBeCloseTo(0.75, 12);
    expect(bg.stroke_width).toBe(0);
    bg.set_style(0.7, { stroke_width: 99 } as never);
    expect(bg.get_fill_opacity()).toBeCloseTo(0.7, 12);
    expect(bg.stroke_width).toBe(0);
    expect(hexOf(bg.fill_color)).toBe("#000000");
  });
});

// ---------------------------------------------------------------------------
// ArrowTip.base (property)
// Python: Arrow(tip).base == tip.point_from_proportion(0.5); the base class
// constructor raises NotImplementedError. TS has no concrete tip subclass
// yet, so the getter is exercised with a VMobject receiver.
// ---------------------------------------------------------------------------
describe("ArrowTip", () => {
  it("constructor raises NotImplementedError like Python", () => {
    expect(() => new ArrowTip()).toThrowError(/implemented in inheriting subclasses/);
  });
  it("base is the halfway point of the tip outline", () => {
    // Python: Arrow([0,0,0],[2,0,0],buff=0).tip.base -> [1.65, 0, 0].
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    const get = Object.getOwnPropertyDescriptor(ArrowTip.prototype, "base")?.get;
    expect(get).toBeTypeOf("function");
    expectNums((get as (this: VMobject) => NDArray).call(v), [1, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("circle arc-length / proportion round-trip", () => {
    // Python: Circle(2).get_arc_length() 12.562410549489542 (sampled);
    // pfp(0.25) [~0, 2, 0]; proportion back 0.25.
    const c = new Circle(2.0);
    expect(c.get_arc_length()).toBeCloseTo(12.562410549489542, 6);
    const p = c.point_from_proportion(0.25);
    expectNums(p, [0, 2, 0]);
    expect(c.proportion_from_point(flat(p))).toBeCloseTo(0.25, 6);
  });
  it("gradient + next_to + move_to chain on a VGroup", () => {
    // Python (Square(2) x2): gradient -> [#FF0000,#0000FF];
    // r2.next_to(r1, RIGHT, 0.5) -> r2 [2.5,0,0]; r1.move_to([3,2,0]).
    const r1 = new Rectangle({ width: 2, height: 2 });
    const r2 = new Rectangle({ width: 2, height: 2 });
    const g = new VGroup(r1, r2);
    g.set_color_by_gradient("#FF0000", "#0000FF");
    r2.next_to(r1, [1, 0, 0], 0.5);
    expectNums(r2.get_center(), [2.5, 0, 0]);
    expect(hexOf(r1.fill_color)).toBe("#FF0000");
    expect(hexOf(r2.fill_color)).toBe("#0000FF");
    r1.move_to([3, 2, 0]);
    expectNums(r1.get_center(), [3, 2, 0]);
  });
  it("fade then match_style across rectangles", () => {
    // Python: fade(0.5) halves 0.8/0.6 -> 0.4/0.3; match copies style.
    const a = new Rectangle({ width: 1, height: 1 });
    a.set_fill(undefined, 0.8);
    a.set_stroke(undefined, undefined, 0.6);
    a.fade(0.5);
    const b = new Rectangle({ width: 1, height: 1 });
    b.match_style(a);
    expect(b.get_fill_opacity()).toBeCloseTo(0.4, 9);
    expect(b.get_stroke_opacity()).toBeCloseTo(0.3, 9);
  });
  it("config frame radii feed to_edge consistently", () => {
    // Python: Square(2).to_edge(LEFT, buff=0) x = -frame_x_radius + 1.
    const r = new Rectangle({ width: 2, height: 2 });
    r.to_edge([-1, 0, 0], 0);
    expect(r.get_center()[0]).toBeCloseTo(-config.frame_x_radius + 1, 9);
  });
  it("Group point queries compose over VMobject corners", () => {
    // get_boundary_point / corners through a Group of two corner squares.
    const g = new Group(square2(), square2());
    g.submobjects[1]?.shift([5, 0, 0]);
    expectNums(g.submobjects[1]?.get_corner([1, 1, 0]) ?? [], [6, 1, 0]);
  });
  it("Mobject1D/Point motion composes with move_to", () => {
    const d = new PointCloudDot([0, 0, 0], 0.5);
    d.move_to([1, 1, 0]);
    expectNums(d.get_center(), [1, 1, 0]);
    expect(mat(d.points)).toHaveLength(86);
  });
});
