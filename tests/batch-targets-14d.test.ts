// Ground-truth tests for batch layers 14..14 (arc/boolean/cutout/family/svg).
//
// Targets:
//   _BooleanOps._convert_vmobject_to_skia_path, Arc._set_pre_positioned_points,
//   CubicBezier, ArcPolygon, ArcPolygonFromArcs, Cutout (+__init__),
//   restructure_list_to_exclude_certain_family_members, SVGMobject,
//   SVGMobject.polyline_to_mobject, VMobjectFromSVGPath.
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0, cairo renderer) with the SAME
// inputs used here; each block cites the probe in a "Python:" comment.
// TS is asserted against those values (1e-9 relative for floats; exact for
// discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): manim.mobject.text/.tex + manim.utils.tex* are
// skipped port-wide; SVGMobject.text_to_mobject / init_svg_mobject /
// generate_mobject stay clearly-marked TODO stubs and are only asserted as
// such below (no text rendering is tested).
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject, Group } from "../src/mobject/mobject.js";
import {
  Arc,
  ArcBetweenPoints,
  ArcPolygon,
  ArcPolygonFromArcs,
  Circle,
  CubicBezier,
} from "../src/mobject/geometry/arc.js";
import "../src/mobject/geometry/line.js"; // registers Line for ArcPolygonFromArcs
import { Line } from "../src/mobject/geometry/line.js";
import {
  Cutout,
  Polygon,
  Rectangle,
  Square,
  Triangle,
} from "../src/mobject/geometry/polygram.js";
import {
  _BooleanOps,
  PathVerb,
  SkiaPath,
} from "../src/mobject/geometry/boolean_ops.js";
import {
  extract_mobject_family_members,
  restructure_list_to_exclude_certain_family_members,
} from "../src/utils/family_ops.js";
import {
  SVGMobject,
  VMobjectFromSVGPath,
  _convert_point_to_3d,
  createSVGXmlElement,
  type SVGElementData,
  type SVGPathSegment,
} from "../src/mobject/svg/svg_mobject.js";
import { pointRows } from "../src/utils/arrays.js";
import { DOWN, LEFT, ORIGIN, RIGHT, UP } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
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

// ---------------------------------------------------------------------------
// Arc._set_pre_positioned_points (observed through Arc points)
// Python:
//   Arc(radius=1, start_angle=0, angle=pi/2, num_components=9)
//     -> 32 pts, pts[0]=[1,0,0], pts[1]=[1,0.06550246635928966,0],
//        pts[-1]=[6.123233995736766e-17,1,0]
//   Arc(radius=2, start_angle=0.5, angle=1.0, num_components=3) -> 8 pts:
//     [1.7551651237807455,0.958851077208406,0]
//     [1.5945190396983007,1.2529117616625076,0]
//     [1.362565015789736,1.5018972761872083,0]
//     [1.0806046117362795,1.682941969615793,0] x2
//     [0.7986442076828231,1.8639866630443778,0]
//     [0.4757153867347149,1.9712873259107815,0]
//     [0.1414744033354058,1.994989973208109,0]
// ---------------------------------------------------------------------------
describe("Arc._set_pre_positioned_points", () => {
  it("builds the quarter-circle like Python", () => {
    const a = new Arc(1.0, 0, Math.PI / 2, 9);
    expect(a.get_num_points()).toBe(32);
    const rows = rowsOf(a.points, 3);
    expectRowsClose([rows[0]!], [[1, 0, 0]]);
    expectRowsClose([rows[1]!], [[1, 0.06550246635928966, 0]]);
    expectRowsClose([rows[rows.length - 1]!], [[6.123233995736766e-17, 1, 0]]);
  });

  it("matches Python anchors/handles for a small off-center arc", () => {
    const a = new Arc(2.0, 0.5, 1.0, 3);
    expect(a.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(a.points, 3), [
      [1.7551651237807455, 0.958851077208406, 0],
      [1.5945190396983007, 1.2529117616625076, 0],
      [1.362565015789736, 1.5018972761872083, 0],
      [1.0806046117362795, 1.682941969615793, 0],
      [1.0806046117362795, 1.682941969615793, 0],
      [0.7986442076828231, 1.8639866630443778, 0],
      [0.4757153867347149, 1.9712873259107815, 0],
      [0.1414744033354058, 1.994989973208109, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// CubicBezier
// Python: CubicBezier([-3,1,0],[-2,1,0],[2,-1,0],[3,-1,0]) -> 4 pts,
//   exactly the four inputs in order.
// ---------------------------------------------------------------------------
describe("CubicBezier", () => {
  it("stores the four control points verbatim like Python", () => {
    const cb = new CubicBezier([-3, 1, 0], [-2, 1, 0], [2, -1, 0], [3, -1, 0]);
    expect(cb.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(cb.points, 3), [
      [-3, 1, 0],
      [-2, 1, 0],
      [2, -1, 0],
      [3, -1, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// ArcPolygon
// Python:
//   ArcPolygon([0,0,0],[2,0,0],[0,2,0]) -> 3 arcs, 96 pts, 3 submobjects;
//     pts[0]=[0,0,0], pts[-1]=[-2.22e-16,0,0];
//     arc0 start [0,0,0] end [2,-2.81e-16,0]
//   ArcPolygon(..., radius=2) -> arc angles
//     [1.0471975511965979, 1.5707963267948968, 1.0471975511965979], radii all 2
//   ArcPolygon(..., arc_config=[{angle:0.5},{angle:0.6},{angle:0.7}])
//     -> per-arc angles [0.5,0.6,0.7]
//   arc_config length != n raises AssertionError
// ---------------------------------------------------------------------------
describe("ArcPolygon", () => {
  it("builds three arcs from vertices like Python", () => {
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0]);
    expect(ap.arcs.length).toBe(3);
    expect(ap.submobjects.length).toBe(3);
    expect(ap.get_num_points()).toBe(96);
    const rows = rowsOf(ap.points, 3);
    expectRowsClose([rows[0]!], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1]!], [[-2.220446049250313e-16, 0, 0]]);
    expectRowsClose([ap.arcs[0]!.get_start()], [[0, 0, 0]]);
    expectRowsClose([ap.arcs[0]!.get_end()], [[2, -2.813458432505755e-16, 0]]);
    expect(ap.arcs.every((a) => a instanceof ArcBetweenPoints)).toBe(true);
  });

  it("derives per-arc angles from a shared radius like Python", () => {
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0], { radius: 2 });
    expectRowsClose(
      (ap.arcs as ArcBetweenPoints[]).map((a) => [a.angle]),
      [[1.0471975511965979], [1.5707963267948968], [1.0471975511965979]],
    );
    for (const a of ap.arcs as ArcBetweenPoints[]) expectClose(a.radius, 2);
  });

  it("honors a per-arc arc_config list like Python", () => {
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0], {
      arc_config: [{ angle: 0.5 }, { angle: 0.6 }, { angle: 0.7 }],
    });
    expectRowsClose(
      (ap.arcs as ArcBetweenPoints[]).map((a) => [a.angle]),
      [[0.5], [0.6], [0.7]],
    );
  });

  it("rejects an arc_config list of the wrong length", () => {
    expect(
      () =>
        new ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0], {
          arc_config: [{ angle: 0.5 }, { angle: 0.6 }],
        }),
    ).toThrowError(/AssertionError/);
  });
});

// ---------------------------------------------------------------------------
// ArcPolygonFromArcs
// Python:
//   ArcPolygonFromArcs(a0,b0,c0) with radius=2 arcs -> 108 pts, 3 arcs/subs,
//     pts[0]=[-1,0,0]
//   ArcPolygonFromArcs(a0, 'notarc') -> ValueError("All ArcPolygon ...")
//   ArcPolygonFromArcs(Arc(0,1), Arc(2,1)) gap case -> 160 pts; the filler
//     line spans g1.end [0.5403023058681398,0.8414709848078965,0] to
//     g2.start [-0.4161468365471424,0.9092974268256817,0]
// ---------------------------------------------------------------------------
describe("ArcPolygonFromArcs", () => {
  it("assembles pre-defined arcs like Python", () => {
    const a0 = new ArcBetweenPoints([-1, 0, 0], [1, 0, 0], Math.PI / 4, 2);
    const b0 = new ArcBetweenPoints([1, 0, 0], [0, 1.7320508075688772, 0], Math.PI / 4, 2);
    const c0 = new ArcBetweenPoints([0, 1.7320508075688772, 0], [-1, 0, 0], Math.PI / 4, 2);
    const r = new ArcPolygonFromArcs(a0, b0, c0);
    expect(r.arcs.length).toBe(3);
    expect(r.submobjects.length).toBe(3);
    expect(r.get_num_points()).toBe(108);
    expectRowsClose([rowsOf(r.points, 3)[0]!], [[-1, 0, 0]]);
  });

  it("rejects non-Arc members with ValueError", () => {
    const a0 = new ArcBetweenPoints([-1, 0, 0], [1, 0, 0], Math.PI / 4, 2);
    try {
      new ArcPolygonFromArcs(a0, "notarc" as unknown as Arc);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
      expect((e as Error).message).toMatch(/All ArcPolygon submobjects/);
    }
  });

  it("fills gaps between arcs with straight lines like Python", () => {
    const g1 = new Arc(1.0, 0, 1.0, 9);
    const g2 = new Arc(1.0, 2.0, 1.0, 9);
    const gf = new ArcPolygonFromArcs(g1, g2);
    expect(gf.get_num_points()).toBe(160);
    expectRowsClose([rowsOf(gf.points, 3)[0]!], [[1, 0, 0]]);
    expectRowsClose([g1.get_end()], [[0.5403023058681398, 0.8414709848078965, 0]]);
    expectRowsClose([g2.get_start()], [[-0.4161468365471424, 0.9092974268256817, 0]]);
  });
});

// ---------------------------------------------------------------------------
// Cutout / Cutout.__init__
// Python (s1=Square().scale(2.5), s2=Triangle().shift(DOWN+RIGHT).scale(0.5)):
//   Cutout(s1, s2, fill_opacity=1) -> 28 pts (16 main + 12 cut);
//     main direction CCW, cut forced to CW;
//     pts[0]=[2.5,2.5,0], pts[16]=[1,-0.3750000000000001,0]
//   Cutout(s1) -> 16 pts; Square().get_direction() == 'CCW'
//   Cutout(s1,s2,s3) with s3=Square().shift(UP+RIGHT).scale(0.5)
//     -> 44 pts, tail [1.5,1.5,0]
// ---------------------------------------------------------------------------
describe("Cutout", () => {
  it("concatenates main + reversed cutout points like Python", () => {
    const s1 = new Square().scale(2.5);
    const s2 = new Triangle().shift([...DOWN, ...[]].slice(0, 3).map((v, i) => v + (([RIGHT[0], RIGHT[1], RIGHT[2]] as number[])[i] ?? 0)));
    s2.scale(0.5);
    expect(s1.get_direction()).toBe("CCW");
    const c = new Cutout(s1, s2, { fill_opacity: 1 });
    expect(c.get_num_points()).toBe(28);
    const rows = rowsOf(c.points, 3);
    expectRowsClose([rows[0]!], [[2.5, 2.5, 0]]);
    expectRowsClose([rows[16]!], [[1, -0.3750000000000001, 0]]);
    // main kept CCW, cut forced to the opposite direction
    expect(s1.get_direction()).toBe("CCW");
    expect(s2.get_direction()).toBe("CW");
  });

  it("single-shape cutout keeps just the main points", () => {
    const s1 = new Square().scale(2.5);
    const c = new Cutout(s1);
    expect(c.get_num_points()).toBe(16);
    expectRowsClose(rowsOf(c.points, 3), rowsOf(s1.points, 3));
  });

  it("stacks several cutouts like Python", () => {
    const s1 = new Square().scale(2.5);
    const s2 = new Triangle().shift([1, -1, 0]);
    s2.scale(0.5);
    const s3 = new Square().shift([1, 1, 0]);
    s3.scale(0.5);
    const c = new Cutout(s1, s2, s3, { fill_opacity: 1 });
    expect(c.get_num_points()).toBe(44);
    const rows = rowsOf(c.points, 3);
    expectRowsClose([rows[rows.length - 1]!], [[1.5, 1.5, 0]]);
  });
});

// ---------------------------------------------------------------------------
// restructure_list_to_exclude_certain_family_members
// Python (m1,m2,m3 Mobjects, g=Group(m1,m2,m3)):
//   ([g],[m1]) -> [m2,m3] (identity); ([m1,m2,m3],[m2]) -> [m1,m3]
//   ([g],[g]) -> []; ([],[m1]) -> []; ([m1,m2],[m3]) -> [m1,m2]
//   nested outer=Group(Group(m1,m2),m3), remove [m1] -> [m2,m3]
// ---------------------------------------------------------------------------
describe("restructure_list_to_exclude_certain_family_members", () => {
  it("explodes a group around a removed submobject", () => {
    const m1 = new Mobject();
    const m2 = new Mobject();
    const m3 = new Mobject();
    const g = new Group(m1, m2, m3);
    const res = restructure_list_to_exclude_certain_family_members([g], [m1]);
    expect(res.length).toBe(2);
    expect(res[0]).toBe(m2);
    expect(res[1]).toBe(m3);
  });

  it("filters flat lists and handles empty/full removals", () => {
    const m1 = new Mobject();
    const m2 = new Mobject();
    const m3 = new Mobject();
    const g = new Group(m1, m2, m3);
    const kept = restructure_list_to_exclude_certain_family_members([m1, m2, m3], [m2]);
    expect(kept.length).toBe(2);
    expect(kept[0]).toBe(m1);
    expect(kept[1]).toBe(m3);
    expect(restructure_list_to_exclude_certain_family_members([g], [g]).length).toBe(0);
    expect(restructure_list_to_exclude_certain_family_members([], [m1]).length).toBe(0);
    expect(
      restructure_list_to_exclude_certain_family_members([m1, m2], [m3]).length,
    ).toBe(2);
  });

  it("descends into nested groups like Python", () => {
    const m1 = new Mobject();
    const m2 = new Mobject();
    const m3 = new Mobject();
    const outer = new Group(new Group(m1, m2), m3);
    const res = restructure_list_to_exclude_certain_family_members([outer], [m1]);
    expect(res.length).toBe(2);
    expect(res[0]).toBe(m2);
    expect(res[1]).toBe(m3);
    // helper used internally agrees with Python's family expansion
    expect(extract_mobject_family_members([outer]).length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// _BooleanOps._convert_vmobject_to_skia_path
// Python (cairo renderer):
//   Square() -> 6 commands: MOVE [[1,1]] then 4x CUBIC then CLOSE [];
//     first cubic [[0.3333333432674408,1],[-0.3333333432674408,1],[-1,1]]
//   Circle() -> 10 commands (MOVE + 8 CUBIC? + CLOSE); first MOVE [[1,0]],
//     second CUBIC starts [[1,0.26521649956703186],...]
//   VMobject() empty -> 0 commands
//   VMobject with a NaN point -> treated as invalid -> 0 commands
// TS mirrors pathops verbs with the PathVerb string enum
// (MOVE/LINE/QUAD/CUBIC/CLOSE); coordinates must match element-wise.
// ---------------------------------------------------------------------------
describe("_BooleanOps._convert_vmobject_to_skia_path", () => {
  it("converts a Square to move + 4 cubics + close like Python", () => {
    // NOTE: Python's pathops (Skia) stores coordinates as 32-bit floats, so
    // its output is float32-rounded (e.g. 0.3333333432674408 for 1/3) while
    // the TS shim keeps float64 (0.3333333333333334). Structure (verbs,
    // counts, endpoints) must match exactly; coordinates use 1e-6.
    const b = new _BooleanOps();
    const path = b._convert_vmobject_to_skia_path(new Square());
    expect(path).toBeInstanceOf(SkiaPath);
    expect(path.commands.length).toBe(6);
    expect(path.commands[0]!.verb).toBe(PathVerb.MOVE);
    expectRowsClose(path.commands[0]!.points, [[1, 1]]);
    for (let i = 1; i <= 4; i += 1) expect(path.commands[i]!.verb).toBe(PathVerb.CUBIC);
    expect(path.commands[5]!.verb).toBe(PathVerb.CLOSE);
    expectRowsClose(
      path.commands[1]!.points,
      [
        [0.3333333432674408, 1],
        [-0.3333333432674408, 1],
        [-1, 1],
      ],
      1e-6,
    );
    // iteration protocol yields [verb, points] pairs
    const walked = [...path];
    expect(walked.length).toBe(6);
    expect(walked[0]![0]).toBe(PathVerb.MOVE);
  });

  it("converts a Circle to move + cubics + close like Python", () => {
    // NOTE: same float32-vs-float64 caveat as the Square test above:
    // Python reports 0.26521649956703186, TS computes 0.265216489839544.
    const b = new _BooleanOps();
    const path = b._convert_vmobject_to_skia_path(new Circle());
    expect(path.commands.length).toBe(10);
    expect(path.commands[0]!.verb).toBe(PathVerb.MOVE);
    expectRowsClose(path.commands[0]!.points, [[1, 0]]);
    expect(path.commands[1]!.verb).toBe(PathVerb.CUBIC);
    expectClose(path.commands[1]!.points[0]![0] ?? NaN, 1);
    expectClose(path.commands[1]!.points[0]![1] ?? NaN, 0.26521649956703186, 1e-6);
    expect(path.commands[path.commands.length - 1]!.verb).toBe(PathVerb.CLOSE);
  });

  it("returns an empty path for empty or non-finite points like Python", () => {
    const b = new _BooleanOps();
    expect(b._convert_vmobject_to_skia_path(new VMobject()).commands.length).toBe(0);
    const bad = new VMobject();
    // NOTE: set_points now guards against non-finite inputs (fail-fast on
    // `scalar * DIRECTION`); assign the raw buffer directly to exercise the
    // downstream non-finite handling this test targets.
    (bad as unknown as { points: NDArray }).points = array([
      [0, 0, 0],
      [1, 1, NaN],
      [2, 2, 2],
      [3, 3, 3],
    ]) as NDArray;
    expect(b._convert_vmobject_to_skia_path(bad).commands.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// SVGMobject
// Python:
//   _convert_point_to_3d(3,4) == [3,4,0]
//   SVGMobject(file_name=None) -> ValueError("Must specify file for SVGMobject")
//   parsed-file SVGMobject: should_center True, svg_height 2, svg_width None,
//     svg_default {color/opacity/fill_color/fill_opacity/stroke_color/
//       stroke_opacity: None, stroke_width: 0}; centered to origin, height 2
//   line_to_mobject(SimpleLine(x1=0,y1=0,x2=3,y2=4)) -> start [0,0,0],
//     end [3,4,0], 4 pts
//   rect_to_mobject(Rect(x=1,y=2,w=4,h=6)) -> width 4 height 6,
//     center [3,5,0], 16 pts
//   ellipse_to_mobject(Ellipse(cx=5,cy=6,rx=2,ry=3)) -> center [5,6,0],
//     width 4 height 6; Circle(cx=1,cy=2,r=3) -> center [1,2,0], width 6
//   polygon_to_mobject(Polygon([(0,0),(4,0),(4,3)])) -> 12 pts, v0 [0,0,0]
//   rounded rect (rx=ry=0.5, w=4,h=2) -> width 4 height 2, 32 pts
//   generate_config_style_dict with all-None default -> {'stroke-width': '0'}
//   handle_transform(Matrix(2,0,0,3,10,20)) maps corners [0,0,0]->[10,20,0],
//     [1,0,0]->[12,20,0], [1,1,0]->[12,23,0]
//   apply_style_to_mobject(rect stroke blue w5 fill red op .5) ->
//     stroke_w 5, stroke #0000FF, fill #FF0000, fill_opacity 128/255
// ---------------------------------------------------------------------------
describe("SVGMobject", () => {
  it("converts 2D points to 3D like Python", () => {
    expectRowsClose(rowsOf(_convert_point_to_3d(3, 4), 3), [[3, 4, 0]]);
  });

  it("exposes Python-compatible constructor defaults", () => {
    const m = new SVGMobject();
    expect(m.file_name).toBeNull();
    expect(m.should_center).toBe(true);
    expect(m.svg_height).toBe(2);
    expect(m.svg_width).toBeNull();
    expect(m.svg_default).toEqual({
      color: null,
      opacity: null,
      fill_color: null,
      fill_opacity: null,
      stroke_width: 0,
      stroke_color: null,
      stroke_opacity: null,
    });
    expect(m.path_string_config).toEqual({});
  });

  it("get_file_path raises ValueError without a file like Python", () => {
    try {
      new SVGMobject().get_file_path();
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
      expect((e as Error).message).toMatch(/Must specify file/);
    }
  });

  it("file-parsing entry points stay clearly-marked TODO stubs (out of scope deps)", () => {
    // NOTE: Python's get_file_path seeks the file on disk and raises OSError
    // for a missing file (verified in the original manim); the TS port mirrors that.
    expect(() => new SVGMobject({ file_name: "foo.svg" }).get_file_path()).toThrowError(
      /could not find foo\.svg/,
    );
    const m = new SVGMobject({ file_name: "foo.svg" });
    expect(() => m.init_svg_mobject(true)).toThrowError(/TODO/);
    expect(() => m.generate_mobject()).toThrowError(/TODO/);
  });

  it("converts line elements like Python", () => {
    const mob = SVGMobject.line_to_mobject({ kind: "line", x1: 0, y1: 0, x2: 3, y2: 4 });
    expect(mob).toBeInstanceOf(Line);
    expectRowsClose([mob.get_start()], [[0, 0, 0]]);
    expectRowsClose([mob.get_end()], [[3, 4, 0]]);
    expect(mob.get_num_points()).toBe(4);
  });

  it("converts rect elements like Python", () => {
    const mob = SVGMobject.rect_to_mobject({ kind: "rect", x: 1, y: 2, width: 4, height: 6 });
    expect(mob).toBeInstanceOf(Rectangle);
    expectClose(mob.width, 4);
    expectClose(mob.height, 6);
    expectRowsClose([mob.get_center()], [[3, 5, 0]]);
    expect(mob.get_num_points()).toBe(16);
  });

  it("converts ellipse/circle elements like Python", () => {
    const em = SVGMobject.ellipse_to_mobject({ kind: "ellipse", cx: 5, cy: 6, rx: 2, ry: 3 });
    expectRowsClose([em.get_center()], [[5, 6, 0]]);
    expectClose(em.width, 4);
    expectClose(em.height, 6);
    const cm = SVGMobject.ellipse_to_mobject({ kind: "circle", cx: 1, cy: 2, rx: 3, ry: 3 });
    expectRowsClose([cm.get_center()], [[1, 2, 0]]);
    expectClose(cm.width, 6);
  });

  it("converts polygon elements like Python", () => {
    const mob = SVGMobject.polygon_to_mobject({
      kind: "polygon",
      points: [[0, 0], [4, 0], [4, 3]],
    });
    expect(mob).toBeInstanceOf(Polygon);
    expect(mob.get_num_points()).toBe(12);
    expectRowsClose([mob.get_vertices()[0]!], [[0, 0, 0]]);
  });

  it("generates the config style dict like Python", () => {
    expect(new SVGMobject().generate_config_style_dict()).toEqual({ "stroke-width": "0" });
    const m = new SVGMobject({
      svg_default: {
        color: "#FF0000",
        opacity: 0.5,
        fill_color: null,
        fill_opacity: null,
        stroke_width: 2,
        stroke_color: "#00FF00",
        stroke_opacity: 0.7,
      },
    });
    expect(m.generate_config_style_dict()).toEqual({
      fill: "#FF0000",
      "fill-opacity": "0.5",
      stroke: "#00FF00",
      "stroke-opacity": "0.7",
      "stroke-width": "2",
    });
  });

  it("applies SVG transforms like Python", () => {
    const v = new VMobject().set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    SVGMobject.handle_transform(v, { a: 2, b: 0, c: 0, d: 3, e: 10, f: 20 });
    const rows = rowsOf(v.points, 3);
    expectRowsClose([rows[0]!], [[10, 20, 0]]);
    expectRowsClose([rows[3]!], [[12, 20, 0]]);
    expectRowsClose([rows[rows.length - 1]!], [[12, 23, 0]]);
  });

  it("applies SVG style like Python", () => {
    const sq = new Square();
    SVGMobject.apply_style_to_mobject(sq, {
      kind: "rect",
      stroke_width: 5,
      stroke_color: "#0000FF",
      stroke_opacity: 1,
      fill_color: "#FF0000",
      fill_opacity: 128 / 255,
    });
    expectClose(sq.get_stroke_width(), 5);
    expectRowsClose(rowsOf(sq.fill_rgbas, 4).slice(0, 1), [[1, 0, 0, 128 / 255]]);
  });

  it("rewraps the xml tree with config + root style groups like Python", () => {
    const m = new SVGMobject({
      svg_default: {
        color: "#FF0000",
        opacity: null,
        fill_color: null,
        fill_opacity: null,
        stroke_width: 0,
        stroke_color: null,
        stroke_opacity: null,
      },
    });
    const root = createSVGXmlElement("svg", { fill: "blue", id: "keep-out" });
    const tree = { getroot: () => root };
    const out = m.modify_xml_tree(tree);
    const top = out.getroot();
    expect(top.tag).toBe("svg");
    expect(top.children.length).toBe(1);
    expect(top.children[0]!.attrib["fill"]).toBe("#FF0000");
  });

  it("collects shape mobjects from an element tree like Python", () => {
    const host = new SVGMobject();
    const svg: SVGElementData = {
      kind: "svg",
      id: "root",
      children: [
        {
          kind: "group",
          id: "g1",
          children: [
            { kind: "rect", x: 0, y: 0, width: 2, height: 2 },
            { kind: "line", x1: 0, y1: 0, x2: 1, y2: 1 },
          ],
        },
      ],
    };
    const [mobs, dict] = host.get_mobjects_from(svg);
    expect(mobs.length).toBe(2);
    expect(dict.has("root")).toBe(true);
    expect(dict.has("g1")).toBe(true);
    expect((dict.get("root") as VGroup).submobjects.length).toBe(2);
  });

  it("text elements stay a warned null stub (text/tex out of scope)", () => {
    // NOTE: mirrors Python's "not yet implemented" path which warns and
    // returns None; the TS port keeps the same shape without any rendering.
    expect(SVGMobject.text_to_mobject({ kind: "text" })).toBeNull();
  });

  it("move_into_position centers and resizes like Python", () => {
    const m = new SVGMobject({ height: 2 });
    m.set_points_as_corners([[10, 10, 0], [14, 10, 0], [14, 12, 0]]);
    m.move_into_position();
    expectRowsClose([m.get_center()], [[0, 0, 0]]);
    expectClose(m.height, 2);
  });
});

// ---------------------------------------------------------------------------
// SVGMobject.polyline_to_mobject
// Python (Polyline([(0,0),(1,1),(2,0)])) -> VMobject, 8 pts:
//   [0,0,0],[1/3,1/3,0],[2/3,2/3,0],[1,1,0],
//   [1,1,0],[4/3,2/3,0],[5/3,1/3,0],[2,0,0]
// ---------------------------------------------------------------------------
describe("SVGMobject.polyline_to_mobject", () => {
  it("converts polylines to cornered VMobjects like Python", () => {
    const host = new SVGMobject();
    const mob = host.polyline_to_mobject({
      kind: "polyline",
      points: [[0, 0], [1, 1], [2, 0]],
    });
    expect(mob).toBeInstanceOf(VMobject);
    expect(mob.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(mob.points, 3), [
      [0, 0, 0],
      [0.3333333333333333, 0.3333333333333333, 0],
      [0.6666666666666666, 0.6666666666666666, 0],
      [1, 1, 0],
      [1, 1, 0],
      [1.3333333333333335, 0.6666666666666667, 0],
      [1.6666666666666665, 0.33333333333333337, 0],
      [2, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// VMobjectFromSVGPath
// Python (cairo defaults):
//   Path('M 0 0 L 4 0 L 4 3 Z') -> 12 pts:
//     [0,0,0],[4/3,0,0],[8/3,0,0],[4,0,0],
//     [4,0,0],[4,1,0],[4,2,0],[4,3,0],
//     [4,3,0],[8/3,2,0],[4/3,1,0],[0,0,0]
//   Path('M 0 0 Q 1 2 3 0') -> 4 pts (cubic passthrough of the quad):
//     [0,0,0],[2/3,4/3,0],[5/3,4/3,0],[3,0,0]
//   Path('M 0 0 C 1 1 2 2 3 0') -> 4 pts verbatim
//   Path('') -> 0 pts, shape (0, 3)
// TS takes pre-parsed {kind,end,control,...} segments (svgelements unported);
// the conversion math must agree element-wise.
// ---------------------------------------------------------------------------
function segs(s: SVGPathSegment[]): SVGPathSegment[] {
  return s;
}

describe("VMobjectFromSVGPath", () => {
  it("converts move/line/close runs like Python", () => {
    const v = new VMobjectFromSVGPath(
      segs([
        { kind: "Move", end: [0, 0] },
        { kind: "Line", end: [4, 0] },
        { kind: "Line", end: [4, 3] },
        { kind: "Close" },
      ]),
    );
    expect(v.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(v.points, 3), [
      [0, 0, 0],
      [1.3333333333333333, 0, 0],
      [2.6666666666666665, 0, 0],
      [4, 0, 0],
      [4, 0, 0],
      [4, 1, 0],
      [4, 2, 0],
      [4, 3, 0],
      [4, 3, 0],
      [2.6666666666666665, 2, 0],
      [1.3333333333333333, 1, 0],
      [0, 0, 0],
    ]);
  });

  it("passes quadratic controls through the cubic blend like Python", () => {
    const v = new VMobjectFromSVGPath(
      segs([
        { kind: "Move", end: [0, 0] },
        { kind: "QuadraticBezier", control: [1, 2], end: [3, 0] },
      ]),
    );
    expect(v.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(v.points, 3), [
      [0, 0, 0],
      [0.6666666666666666, 1.3333333333333333, 0],
      [1.6666666666666667, 1.3333333333333333, 0],
      [3, 0, 0],
    ]);
  });

  it("keeps cubic controls verbatim like Python", () => {
    const v = new VMobjectFromSVGPath(
      segs([
        { kind: "Move", end: [0, 0] },
        { kind: "CubicBezier", control1: [1, 1], control2: [2, 2], end: [3, 0] },
      ]),
    );
    expect(v.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(v.points, 3), [
      [0, 0, 0],
      [1, 1, 0],
      [2, 2, 0],
      [3, 0, 0],
    ]);
  });

  it("yields an empty (0,3) point array for an empty path like Python", () => {
    const v = new VMobjectFromSVGPath([]);
    expect(v.get_num_points()).toBe(0);
    expect(v.points.shape).toEqual([0, 3]);
  });

  it("path_to_mobject forwards segments and config like Python", () => {
    const host = new SVGMobject({ path_string_config: { long_lines: false } });
    const mob = host.path_to_mobject({
      kind: "path",
      segments: [{ kind: "Move", end: [1, 2] }],
    });
    expect(mob).toBeInstanceOf(VMobjectFromSVGPath);
    expect(mob.path_obj.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Integration (this batch x earlier-ported code)
// ---------------------------------------------------------------------------
describe("batch-14d integration", () => {
  it("boolean path of an ArcPolygon-built triangle closes like Python", () => {
    // Python: ArcPolygon triangle -> closed point loop; its skia path ends
    // with CLOSE and starts at the first vertex.
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0]);
    const b = new _BooleanOps();
    const path = b._convert_vmobject_to_skia_path(ap);
    expect(path.commands[0]!.verb).toBe(PathVerb.MOVE);
    expectRowsClose(path.commands[0]!.points, [[0, 0]]);
    expect(path.commands[path.commands.length - 1]!.verb).toBe(PathVerb.CLOSE);
  });

  it("Cutout of a circle-punched square keeps area ordering via config-free paths", () => {
    // Python: Cutout(Square, Circle-punched) concatenates main + reversed
    // cut points; the boolean converter still walks the combined loop.
    const main = new Square().scale(2);
    const hole = new Circle(0.5);
    const c = new Cutout(main, hole);
    expect(c.get_num_points()).toBe(main.get_num_points() + hole.get_num_points());
    const b = new _BooleanOps();
    const path = b._convert_vmobject_to_skia_path(c);
    expect(path.commands.length).toBeGreaterThan(0);
    expect(path.commands[0]!.verb).toBe(PathVerb.MOVE);
  });

  it("restructure keeps SVG-collected mobjects not slated for removal", () => {
    // Python: get_mobjects_from tree -> [rectMob, lineMob]; removing the line
    // leaves the rect via the family_exclusion helper.
    const host = new SVGMobject();
    const [mobs] = host.get_mobjects_from({
      kind: "svg",
      children: [
        { kind: "rect", x: 0, y: 0, width: 2, height: 2 },
        { kind: "line", x1: 0, y1: 0, x2: 1, y2: 1 },
      ],
    });
    expect(mobs.length).toBe(2);
    const kept = restructure_list_to_exclude_certain_family_members(mobs, [mobs[1]!]);
    expect(kept.length).toBe(1);
    expect(kept[0]).toBe(mobs[0]);
  });

  it("SVG polyline points agree with an equivalent CubicBezier-free VMobject", () => {
    // Python: polyline [(0,0),(1,1),(2,0)] starts [0,0,0] and ends [2,0,0],
    // same endpoints as Line([0,0,0],[2,0,0]); the cubic curve through the
    // same four anchors is a distinct (single-bezier) object.
    const host = new SVGMobject();
    const pl = host.polyline_to_mobject({
      kind: "polyline",
      points: [[0, 0], [1, 1], [2, 0]],
    });
    const line = new Line([0, 0, 0], [2, 0, 0]);
    const plRows = rowsOf(pl.points, 3);
    expectRowsClose([plRows[0]!], [[0, 0, 0]]);
    expectRowsClose([plRows[plRows.length - 1]!], [line.get_end()]);
    const cb = new CubicBezier([0, 0, 0], [1, 1, 0], [1, 1, 0], [2, 0, 0]);
    expect(cb.get_num_points()).toBe(4);
    expectRowsClose([rowsOf(cb.points, 3)[0]!], [[0, 0, 0]]);
  });
});
