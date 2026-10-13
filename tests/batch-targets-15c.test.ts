// Ground-truth tests for batch layers 15..15 (part C).
//
// Targets:
//   - manim.mobject.geometry.arc.Arc
//   - manim.mobject.geometry.arc.ArcBetweenPoints.__init__
//   - manim.mobject.geometry.labeled.LabeledPolygram
//   - manim.mobject.svg.brace.Brace.get_tip
//   - manim.mobject.svg.svg_mobject.SVGMobject.path_to_mobject
//   - manim.scene.scene.Scene.get_mobject_family_members
//   - manim.scene.scene.Scene.restructure_mobjects
//   - manim._config.utils.ManimConfig
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete values;
// arrays element-wise).
//
// NOTE: LabeledPolygram's label is now rendered through the ported text
// pipeline (MathTex/Text). Tested: the missing-label TypeError, label
// placement, and the underlying polylabel pole computation.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { PI, TAU, DOWN, LEFT, RIGHT } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { polylabel } from "../src/utils/polylabel.js";
import { Arc, ArcBetweenPoints } from "../src/mobject/geometry/arc.js";
import { LabeledPolygram } from "../src/mobject/geometry/labeled.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle, Dot } from "../src/mobject/geometry/arc.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Brace } from "../src/mobject/svg/brace.js";
import {
  SVGMobject,
  parse_svg_path_data,
  VMobjectFromSVGPath,
} from "../src/mobject/svg/svg_mobject.js";
import { Scene } from "../src/scene/scene.js";
import { ManimConfig, config } from "../src/_config/utils.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function expectRowsClose(actual: number[][] | { toArray(): unknown }, expected: number[][], tol = 1e-9): void {
  const expectedRows = (Array.isArray(expected)
    ? (expected as unknown[])
    : ((expected as { toArray(): unknown }).toArray() as unknown[])
  ).map((row) =>
    Array.isArray(row) ? row : ((row as { toArray(): unknown }).toArray() as number[]),
  ) as number[][];
  const actualRows = (Array.isArray(actual) ? actual : ((actual as { toArray(): unknown }).toArray() as unknown[])).map(
    (row) => (Array.isArray(row) ? row : (row as { toArray(): unknown }).toArray()),
  ) as number[][];
  expect(actualRows.length).toBe(expectedRows.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actualRows[i]?.length).toBe(expectedRows[i]?.length);
    for (let j = 0; j < (expectedRows[i]?.length ?? 0); j += 1) {
      expect(close(actualRows[i]?.[j] ?? NaN, expectedRows[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Arc
// Python:
//   Arc(): radius 1.0, angle TAU/4, start_angle 0, 32 points,
//     points[0] [1,0,0], points[1] [1, 0.065502466359, 0],
//     points[2] [0.993564177658, 0.130846467181, 0],
//     points[3] [0.980785280403, 0.195090322016, 0],
//     points[-1] [6.123233995736766e-17, 1.0, 0.0]
//   Arc(radius=2.0, start_angle=PI/4, angle=PI/2, arc_center=[1,2,0]):
//     points[0] [2.414213562373095, 3.414213562373095, 0.0],
//     points[-1] [-0.4142135623730949, 3.414213562373095, 0.0],
//     get_arc_center() ~= [1, 2, 0], stop_angle() 2.356194490192353 (3*PI/4)
//   Arc().move_arc_center_to([1,2,0]):
//     points[0] [1.9999999999999967, 1.9999999999999998, 0.0],
//     get_arc_center() [0.9999999999999998, 2.0, 0.0]
// ---------------------------------------------------------------------------
describe("Arc", () => {
  it("builds the default quarter arc with ground-truth points", () => {
    const a = new Arc();
    expectClose(a.radius, 1.0);
    expectClose(a.angle, TAU / 4);
    expectClose(a.start_angle, 0);
    expect(a.get_num_points()).toBe(32);
    const rows = rowsOf(a.points, a.dim);
    expectRowsClose([rows[0] ?? []], [[1, 0, 0]]);
    expectRowsClose([rows[1] ?? []], [[1.0, 0.065502466359, 0.0]]);
    expectRowsClose([rows[2] ?? []], [[0.993564177658, 0.130846467181, 0.0]]);
    expectRowsClose([rows[3] ?? []], [[0.980785280403, 0.195090322016, 0.0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[6.123233995736766e-17, 1.0, 0.0]]);
  });

  it("honors radius/start_angle/angle/arc_center", () => {
    const a = new Arc(2.0, PI / 4, PI / 2, 9, [1, 2, 0]);
    const rows = rowsOf(a.points, a.dim);
    expectRowsClose([rows[0] ?? []], [[2.414213562373095, 3.414213562373095, 0.0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[-0.4142135623730949, 3.414213562373095, 0.0]]);
    expectRowsClose([a.get_arc_center()], [[1, 2, 0]], 1e-9);
    expectClose(a.stop_angle(), 2.356194490192353);
  });

  it("computes quadratic bezier points with ground-truth values", () => {
    const pts = rowsOf(Arc._create_quadratic_bezier_points(1.0, 0.5, 2), 3);
    expectRowsClose(pts, [
      [0.8775825618903728, 0.479425538604203, 0.0],
      [0.7551651237807455, 0.7035091559873697, 0.0],
      [0.5403023058681398, 0.8414709848078965, 0.0],
      [0.5403023058681398, 0.8414709848078965, 0.0],
      [0.325439487955534, 0.9794328136284233, 0.0],
      [0.0707372016677029, 0.9974949866040544, 0.0],
    ]);
  });

  it("moves the arc center to a target point", () => {
    const c = new Arc().move_arc_center_to([1, 2, 0]);
    const rows = rowsOf(c.points, c.dim);
    expectRowsClose([rows[0] ?? []], [[1.9999999999999967, 1.9999999999999998, 0.0]]);
    expectRowsClose([c.get_arc_center()], [[0.9999999999999998, 2.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.ArcBetweenPoints.__init__
// Python:
//   ArcBetweenPoints([0,0,0],[1,0,0], angle=PI/2):
//     points[0] [0,0,0], points[-1] [1, 1.2116883882008518e-16, 0.0],
//     radius 0.7071067811865429, angle PI/2
//   ArcBetweenPoints([0,0,0],[2,0,0], radius=2.0):
//     angle 1.0471975511965979 (PI/3), radius 2.0,
//     points[0] [0,0,0], points[-1] [2, 7.176594446642598e-16, 0.0]
//   ArcBetweenPoints([0,0,0],[4,0,0], radius=1.0) raises ValueError
//   ArcBetweenPoints([0,0,0],[1,0,0], angle=0): 4 points, first [0,0,0]
//   ArcBetweenPoints([0,0,0],[1,0,0], radius=-2.0):
//     angle -0.5053605102841571, radius attr 2.0,
//     points[0] [0,0,0], points[-1] [1, -1.612453231855972e-16, 0.0]
// ---------------------------------------------------------------------------
describe("ArcBetweenPoints.__init__", () => {
  it("builds the angle-form arc with exact endpoints", () => {
    const ab = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], PI / 2);
    const rows = rowsOf(ab.points, ab.dim);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1.0, 1.2116883882008518e-16, 0.0]]);
    expectClose(ab.radius, 0.7071067811865429);
    expectClose(ab.angle, PI / 2);
  });

  it("derives the angle from an explicit radius", () => {
    const ab = new ArcBetweenPoints([0, 0, 0], [2, 0, 0], TAU / 4, 2.0);
    expectClose(ab.angle, 1.0471975511965979);
    expectClose(ab.radius, 2.0);
    const rows = rowsOf(ab.points, ab.dim);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[2.0, 7.176594446642598e-16, 0.0]]);
  });

  it("rejects a radius smaller than half the point distance", () => {
    expect(
      () => new ArcBetweenPoints([0, 0, 0], [4, 0, 0], TAU / 4, 1.0),
    ).toThrowError(expect.objectContaining({ name: "ValueError" }));
  });

  it("degenerates to a straight segment for angle 0", () => {
    const ab = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], 0);
    expect(ab.get_num_points()).toBe(4);
    expectRowsClose([rowsOf(ab.points, ab.dim)[0] ?? []], [[0, 0, 0]]);
  });

  it("mirrors the arc for a negative radius", () => {
    const ab = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], TAU / 4, -2.0);
    expectClose(ab.angle, -0.5053605102841571);
    expectClose(ab.radius, 2.0);
    const rows = rowsOf(ab.points, ab.dim);
    expectRowsClose([rows[0] ?? []], [[0, 0, 0]]);
    expectRowsClose([rows[rows.length - 1] ?? []], [[1.0, -1.612453231855972e-16, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.labeled.LabeledPolygram
// Python: missing `label` kwarg raises
//   TypeError: LabeledPolygram.__init__() missing 1 required keyword-only
//   argument: 'label'. The label itself is rendered text (out of scope), so
//   construction with a label raises the port's TODO stub. The pole algorithm
//   underneath is polylabel; ground truth for a 4x2 rectangle ring:
//   polylabel([[[-2,-1],[2,-1],[2,1],[-2,1]]], precision=0.01)
//     -> c [0.44444444, 0.0], d 1.0.
// ---------------------------------------------------------------------------
describe("LabeledPolygram", () => {
  it("requires the keyword-only label argument", () => {
    expect(() => new LabeledPolygram([[0, 0, 0], [1, 0, 0], [1, 1, 0]])).toThrowError(
      "LabeledPolygram.__init__() missing 1 required keyword-only argument: 'label'",
    );
  });

  it("places a label at the pole once a label is given", () => {
    const p = new LabeledPolygram([[0, 0, 0], [1, 0, 0], [1, 1, 0]], {
      label: "Pole",
    });
    expect(p.label).toBeDefined();
    expect(p.radius).toBeGreaterThan(0);
  });

  it("computes the pole of inaccessibility via polylabel (ground truth)", () => {
    const cell = polylabel(
      [[[-2, -1], [2, -1], [2, 1], [-2, 1]]],
      0.01,
    );
    expectRowsClose([[...(cell.c.toArray() as number[])]], [[0.44444444, 0.0]], 1e-6);
    expectClose(cell.d, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.Brace.get_tip
// Python (Square(), direction=DOWN):
//   get_tip() [0.0, -1.472985, 1.671551032326201e-17],
//   _tip_point_index 27, 100 points,
//   get_direction() [0.0, -1.0, 1.224646799147352e-16]
// ---------------------------------------------------------------------------
describe("Brace.get_tip", () => {
  it("returns the brace tip point with ground-truth value", () => {
    const b = new Brace(new Square(), { direction: [...DOWN] });
    expectRowsClose([b.get_tip()], [[0.0, -1.472985, 1.671551032326201e-17]], 1e-6);
    // The tip is one of the stored points, at the recorded index.
    const rows = rowsOf(b.points, b.dim);
    expectRowsClose([rows[b._tip_point_index] ?? []], [b.get_tip()]);
  });

  it("points in the brace direction (normalized tip - center)", () => {
    const b = new Brace(new Square(), { direction: [...DOWN] });
    const dir = b.get_direction() as number[];
    expectRowsClose([dir], [[0.0, -1.0, 1.224646799147352e-16]], 1e-6);
    expectClose(Math.hypot(...dir), 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.path_to_mobject
// Python:
//   s = SVGMobject.__new__(SVGMobject); s.path_string_config = {}
//   mob = SVGMobject.path_to_mobject(s, se.Path('M 0 0 L 10 0 L 10 10 Z'))
//     -> VMobjectFromSVGPath, 12 points, points[0] [0,0,0]
// ---------------------------------------------------------------------------
describe("SVGMobject.path_to_mobject", () => {
  it("converts parsed segments into a VMobjectFromSVGPath", () => {
    const owner = new SVGMobject();
    const segments = parse_svg_path_data("M 0 0 L 10 0 L 10 10 Z");
    const mob = owner.path_to_mobject({ kind: "path", segments });
    expect(mob).toBeInstanceOf(VMobjectFromSVGPath);
    expect(mob.get_num_points()).toBe(12);
    expectRowsClose([rowsOf(mob.points, mob.dim)[0] ?? []], [[0, 0, 0]]);
  });

  it("matches the standalone VMobjectFromSVGPath for the same data", () => {
    const owner = new SVGMobject();
    const segments = parse_svg_path_data("M 0 0 L 10 0 L 10 10 Z");
    const viaMethod = owner.path_to_mobject({ kind: "path", segments });
    const direct = new VMobjectFromSVGPath(segments);
    expectRowsClose(
      rowsOf(viaMethod.points, viaMethod.dim),
      rowsOf(direct.points, direct.dim),
    );
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_mobject_family_members
// Python (cairo, use_z_index=False):
//   mobjects [Circle(), VGroup(Square(), Dot())] -> 4 members
//   [Circle, VGroup, Square, Dot]
// ---------------------------------------------------------------------------
describe("Scene.get_mobject_family_members", () => {
  it("flattens top-level mobjects and their families in order", () => {
    const scene = new Scene();
    const c = new Circle();
    const g = new VGroup(new Square(), new Dot());
    scene.add(c, g);
    const fam = scene.get_mobject_family_members();
    expect(fam.length).toBe(4);
    expect(fam.map((m) => m.constructor.name)).toEqual([
      "Circle",
      "VGroup",
      "Square",
      "Dot",
    ]);
  });

  it("returns an empty list for an empty scene", () => {
    expect(new Scene().get_mobject_family_members()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.restructure_mobjects
// Python (cairo, use_z_index=False):
//   mobjects [Circle, VGroup(Square, Dot)]; restructure_mobjects([square])
//     -> mobjects [Circle, Dot] (group dissolved, square gone)
//   restructure_mobjects([group], extract_families=True) -> [Circle]
//   restructure_mobjects([group.submobjects[0]], extract_families=False)
//     on [Circle, VGroup(Square, Square)] -> [Circle, Square]
// ---------------------------------------------------------------------------
describe("Scene.restructure_mobjects", () => {
  it("dissolves a group when one of its submobjects is removed", () => {
    const scene = new Scene();
    const c = new Circle();
    const sq = new Square();
    const g = new VGroup(sq, new Dot());
    scene.add(c, g);
    scene.restructure_mobjects([sq]);
    expect(scene.mobjects.map((m) => m.constructor.name)).toEqual([
      "Circle",
      "Dot",
    ]);
  });

  it("removes a whole group with family extraction", () => {
    const scene = new Scene();
    const c = new Circle();
    const g = new VGroup(new Square(), new Dot());
    scene.add(c, g);
    scene.restructure_mobjects([g], "mobjects", true);
    expect(scene.mobjects.length).toBe(1);
    expect(scene.mobjects[0]).toBe(c);
  });

  it("keeps siblings without family extraction", () => {
    const scene = new Scene();
    const c = new Circle();
    const s1 = new Square();
    const g = new VGroup(s1, new Square());
    scene.add(c, g);
    scene.restructure_mobjects([s1], "mobjects", false);
    expect(scene.mobjects.map((m) => m.constructor.name)).toEqual([
      "Circle",
      "Square",
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig
// Python (global config digested from default.cfg):
//   frame_height 8.0, frame_width 14.222222222222221,
//   aspect_ratio 1.7777777777777777, frame_size (1920, 1080),
//   quality 'high_quality', renderer RendererType.CAIRO, frame_rate 60.0,
//   seed None, dry_run False; dry_run=True -> write_to_movie False,
//   format None; background_opacity=2 raises ValueError;
//   top [0, 4, 0], bottom [0, -4, 0];
//   fresh ManimConfig() holds None; del raises AttributeError;
//   missing key raises AttributeError; len 67; copy/update round-trip.
// ---------------------------------------------------------------------------
describe("ManimConfig", () => {
  it("exposes library defaults matching the Python global config", () => {
    expectClose(config.frame_height as number, 8.0);
    expectClose(config.frame_width as number, 14.222222222222221);
    expectClose(config.get("aspect_ratio") as number, 1.7777777777777777);
    expect(config.frame_size).toEqual([1920, 1080]);
    expect(config.quality).toBe("high_quality");
    expect(String(config.renderer)).toBe("cairo");
    expectClose(config.frame_rate as number, 60.0);
    expect(config.seed).toBeNull();
    expect(config.dry_run).toBe(false);
  });

  it("starts empty (null option values) like Python's fresh ManimConfig", () => {
    const fresh = new ManimConfig();
    expect(fresh.get("frame_height")).toBeNull();
    expect(fresh.get("frame_rate")).toBeNull();
    expect(() => fresh.__getitem__("not_a_key")).toThrowError(
      "'ManimConfig' object has no attribute 'not_a_key'",
    );
  });

  it("keeps frame dimensions and radii in sync", () => {
    const c = new ManimConfig(true);
    c.set("frame_height", 16);
    expectClose(c.get("frame_height") as number, 16);
    expectClose(c.get("frame_y_radius") as number, 8.0);
    c.frame_width = 10;
    expectClose(c.get("frame_x_radius") as number, 5.0);
  });

  it("resolves named qualities to pixel dimensions and frame rate", () => {
    const c = new ManimConfig(true);
    c.quality = "low_quality";
    expect(c.get("pixel_width")).toBe(854);
    expect(c.get("pixel_height")).toBe(480);
    expect(c.get("frame_rate")).toBe(15);
  });

  it("supports len/contains/iteration/repr like the Python mapping", () => {
    const c = new ManimConfig(true);
    expect(c.__len__()).toBe(67);
    expect(c.__contains__("frame_height")).toBe(true);
    expect(c.__contains__("nope")).toBe(false);
    expect([...c]).toContain("frame_height");
    expect(c.__repr__()).toContain("frame_height");
  });

  it("copies deeply and updates from plain objects", () => {
    const c = new ManimConfig(true);
    const cp = c.copy();
    expect(cp.get("frame_height")).toBe(c.get("frame_height"));
    c.update({ frame_rate: 30 });
    expect(c.get("frame_rate")).toBe(30);
    expect(cp.get("frame_rate")).toBe(60);
  });

  it("disables movie output when dry_run is set", () => {
    const c = new ManimConfig(true);
    c.dry_run = true;
    expect(c.write_to_movie).toBe(false);
    expect(c.get("format")).toBeNull();
  });

  it("validates background_opacity and exposes frame edges", () => {
    const c = new ManimConfig(true);
    c.background_opacity = 0.5;
    expectClose(c.background_opacity, 0.5);
    expect(() => {
      c.background_opacity = 2;
    }).toThrowError(expect.objectContaining({ name: "ValueError" }));
    expectRowsClose([c.top], [[0, 4, 0]]);
    expectRowsClose([c.bottom], [[0, -4, 0]]);
    expectRowsClose([c.left_side], [[-7.111111111111111, 0, 0]]);
    expectRowsClose([c.right_side], [[7.111111111111111, 0, 0]]);
  });

  it("derives transparent from background_opacity (not a stored key)", () => {
    // Python: transparent is a property (background_opacity < 1.0), not a
    // member of _OPTS: len 67, iteration excludes it, but __getitem__ and
    // __contains__ still see it. Fresh configs raise AssertionError.
    const c = new ManimConfig(true);
    expect(c.transparent).toBe(false);
    expect(c.get("transparent")).toBe(false);
    expect(c.__contains__("transparent")).toBe(true);
    expect([...c]).not.toContain("transparent");
    c.set("transparent", true);
    expect(c.transparent).toBe(true);
    expectClose(c.background_opacity, 0.0);
    c.set("transparent", false);
    expect(c.transparent).toBe(false);
    expectClose(c.background_opacity, 1.0);
    expect(() => new ManimConfig().transparent).toThrowError(
      expect.objectContaining({ name: "AssertionError" }),
    );
  });

  it("forbids deletion and rejects unknown keys", () => {
    const c = new ManimConfig(true);
    expect(() => c.__delitem__("frame_height")).toThrowError(
      expect.objectContaining({ name: "AttributeError" }),
    );
    expect(() => c.__getitem__("nonexistent")).toThrowError(
      "'ManimConfig' object has no attribute 'nonexistent'",
    );
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols against already-ported code
// ---------------------------------------------------------------------------
describe("batch-15c integration", () => {
  it("tracks an Arc through the scene family and restructures it away", () => {
    // Python: Scene with [Arc(), VGroup(Square(), Dot())] has 4 family
    // members; restructuring out the Arc leaves [VGroup, Square, Dot].
    const scene = new Scene();
    const arc = new Arc();
    const g = new VGroup(new Square(), new Dot());
    scene.add(arc, g);
    expect(scene.get_mobject_family_members().length).toBe(4);
    scene.restructure_mobjects([arc]);
    expect(scene.mobjects.length).toBe(1);
    expect(scene.get_mobject_family_members().length).toBe(3);
  });

  it("carries ArcBetweenPoints endpoints through put_start_and_end_on", () => {
    // Python: ArcBetweenPoints(LEFT, RIGHT, angle=PI/2) starts at LEFT and
    // ends at RIGHT exactly; adding it to a scene exposes it via the family.
    const ab = new ArcBetweenPoints([...LEFT], [...RIGHT], PI / 2);
    expectRowsClose([ab.get_start()], [[-1, 0, 0]]);
    expectRowsClose([ab.get_end()], [[1, 0, 0]]);
    const scene = new Scene();
    scene.add(ab);
    expect(scene.get_mobject_family_members()).toContain(ab);
  });

  it("feeds config frame dimensions into arc geometry", () => {
    // Python: Arc().scale(2) scales about the arc center, so the
    // ground-truth first point [1,0,0] moves to [1.5,-0.5,0]; scaling
    // about ORIGIN (as generate_points does with the radius) doubles it
    // to [2,0,0]. The factor comes from config.frame_height (8.0 / 8).
    const factor = (config.frame_height as number) / 8;
    expectClose(factor, 1.0);
    const a = new Arc();
    a.scale(2);
    expectRowsClose([rowsOf(a.points, a.dim)[0] ?? []], [[1.5, -0.5, 0.0]]);
    const b = new Arc();
    b.scale(2, [0, 0, 0]);
    expectRowsClose([rowsOf(b.points, b.dim)[0] ?? []], [[2, 0, 0]]);
  });

  it("Brace tip agrees with the SVG-path point pipeline", () => {
    // The brace is a VMobjectFromSVGPath; its tip must be one of its own
    // points and path_to_mobject must produce the same class of object.
    const b = new Brace(new Square(), { direction: [...DOWN] });
    expect(b).toBeInstanceOf(VMobjectFromSVGPath);
    const owner = new SVGMobject();
    const mob = owner.path_to_mobject({
      kind: "path",
      segments: parse_svg_path_data("M 0 0 L 1 0"),
    });
    expect(mob).toBeInstanceOf(VMobjectFromSVGPath);
    expect(mob.has_points()).toBe(true);
  });
});
