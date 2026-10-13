// Ground-truth tests for batch layers 13..14 (~1055 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here; each block cites the
// probe in a "Python:" comment. TS is asserted against those values (1e-9
// relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): nothing in this batch touches text/tex.
import { describe, expect, it, vi } from "vitest";
import type { NDArray } from "numpy-ts";
import { array } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Wait, Animation } from "../src/animation/animation.js";
import { linear } from "../src/utils/rate_functions.js";
import { ShowPartial, Create, DrawBorderThenFill } from "../src/animation/creation.js";
import { FadeTransform } from "../src/animation/transform.js";
import {
  TransformMatchingAbstractBase,
  TransformMatchingShapes,
} from "../src/animation/transform_matching_parts.js";
import { Transform } from "../src/animation/transform.js";
import { Camera } from "../src/camera/camera.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { Scene } from "../src/scene/scene.js";
import { ThreeDScene } from "../src/scene/three_d_scene.js";
import { extract_mobject_family_members } from "../src/utils/family.js";
import { pointRows } from "../src/utils/arrays.js";

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
// manim.animation.animation.Wait (+ __init__)
// Python:
//   Wait() -> run_time 1, duration 1, stop None, static None, rate is linear
//   Wait(2.5) -> run_time 2.5 duration 2.5
//   Wait(run_time=3.0) -> 3.0
//   Wait(1, stop_condition=..., frozen_frame=True) -> ValueError
//   Wait(1, frozen_frame=False).is_static_wait is False
//   begin/interpolate/finish are no-ops
// ---------------------------------------------------------------------------
describe("Wait", () => {
  it("stores defaults (run_time 1, linear rate, static None)", () => {
    const w = new Wait();
    expectClose(w.run_time, 1);
    expectClose(w.duration, 1);
    expect(w.stop_condition).toBeNull();
    expect(w.is_static_wait).toBeNull();
    expect(w.rate_func).toBe(linear);
  });

  it("accepts a positional run_time", () => {
    const w = new Wait(2.5);
    expectClose(w.run_time, 2.5);
    expectClose(w.duration, 2.5);
  });

  it("accepts run_time as an options bag", () => {
    const w = new Wait({ run_time: 3.0 });
    expectClose(w.run_time, 3.0);
    expectClose(w.duration, 3.0);
  });

  it("rejects stop_condition + frozen_frame", () => {
    expect(() => new Wait(1, { stop_condition: () => true, frozen_frame: true })).toThrowError(
      /static Wait/,
    );
  });

  it("keeps frozen_frame=False as non-static", () => {
    const w = new Wait(1, { frozen_frame: false });
    expect(w.is_static_wait).toBe(false);
  });

  it("begin/interpolate/finish/update_mobjects/clean_up are no-ops", () => {
    const w = new Wait(2.0);
    expect(() => {
      w.begin();
      w.interpolate(0.5);
      w.finish();
      w.update_mobjects(0.016);
      w.clean_up_from_scene(new Scene());
    }).not.toThrow();
    expectClose(w.run_time, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_top_level_mobjects.is_top_level
// Python:
//   outer.add(inner); sc.add(outer, inner) -> top == [outer] (len 1)
//   sc2.add(x, y) -> len 2
// ---------------------------------------------------------------------------
describe("Scene.get_top_level_mobjects (is_top_level)", () => {
  it("filters out mobjects that are family of another entry", () => {
    const sc = new Scene();
    const outer = new Mobject();
    const inner = new Mobject();
    outer.add(inner);
    sc.add(outer, inner);
    const top = sc.get_top_level_mobjects();
    expect(top.length).toBe(1);
    expect(top[0]).toBe(outer);
  });

  it("keeps independent mobjects", () => {
    const sc = new Scene();
    const x = new Mobject();
    const y = new Mobject();
    sc.add(x, y);
    expect(sc.get_top_level_mobjects().length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.replace.replace_in_list
// Python:
//   sc.add(p); sc.replace(p, q) -> mobjects[0] is q
//   nested: parent.add(child); replace(child, new) -> parent.submobjects[0] is new
//   unknown old -> ValueError; self-replace warns and keeps m
// ---------------------------------------------------------------------------
describe("Scene.replace (replace_in_list)", () => {
  it("replaces a top-level mobject", () => {
    const sc = new Scene();
    const p = new Mobject();
    const q = new Mobject();
    sc.add(p);
    sc.replace(p, q);
    expect(sc.mobjects[0]).toBe(q);
  });

  it("replaces a nested submobject via BFS + recursion", () => {
    const sc = new Scene();
    const parent = new Mobject();
    const child = new Mobject();
    parent.add(child);
    sc.add(parent);
    const replacement = new Mobject();
    sc.replace(child, replacement);
    expect(parent.submobjects[0]).toBe(replacement);
  });

  it("raises when the old mobject is absent", () => {
    const sc = new Scene();
    sc.add(new Mobject());
    expect(() => sc.replace(new Mobject(), new Mobject())).toThrowError();
  });

  it("self-replace warns and keeps the mobject", () => {
    const sc = new Scene();
    const m = new Mobject();
    sc.add(m);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      sc.replace(m, m);
      expect(warn).toHaveBeenCalled();
      expect(sc.mobjects[0]).toBe(m);
    } finally {
      warn.mockRestore();
    }
  });

  it("raises when either side is None", () => {
    const sc = new Scene();
    expect(() => sc.replace(null as unknown as Mobject, new Mobject())).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// manim.utils.family.extract_mobject_family_members
// Python:
//   a.add(b); b.add(c); extract([a]) -> 3 members, first is a
//   VGroup(Square) only_those_with_points -> 1
//   z_index 5 vs 1 sorted -> [1, 5]; extract([a, a]) dedups -> 3
// ---------------------------------------------------------------------------
describe("extract_mobject_family_members", () => {
  it("extracts the recursive family", () => {
    const a = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    a.add(b);
    b.add(c);
    const fam = extract_mobject_family_members([a]);
    expect(fam.length).toBe(3);
    expect(fam[0]).toBe(a);
  });

  it("filters to members with points", () => {
    const s = new Square();
    const g = new VGroup(s);
    expect(extract_mobject_family_members([g], false, true).length).toBe(1);
  });

  it("sorts by z_index when requested", () => {
    const m1 = new Mobject();
    m1.z_index = 5;
    const m2 = new Mobject();
    m2.z_index = 1;
    const res = extract_mobject_family_members([m1, m2], true);
    expect(res.map((m) => m.z_index)).toEqual([1, 5]);
  });

  it("removes redundancies", () => {
    const a = new Mobject();
    a.add(new Mobject());
    const b = new Mobject();
    // Python: extract([a, a]) dedups the repeated top-level entry but keeps
    // the 2-member family (a + child).
    expect(extract_mobject_family_members([a, a]).length).toBe(2);
    expect(extract_mobject_family_members([a, b]).length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.ShowPartial
// Python:
//   ShowPartial(Mobject()) -> TypeError("... only works for VMobjects.")
//   Create(Square())._get_bounds(0.3) -> (0, 0.3)
//   ShowPartial(Square())._get_bounds -> NotImplementedError
//   Create interp alpha=0.5 on Square(16 pts) -> 12 pts, first [1,1,0]
// ---------------------------------------------------------------------------
describe("ShowPartial", () => {
  it("rejects non-VMobjects", () => {
    expect(() => new ShowPartial(new Mobject() as unknown as VMobject)).toThrowError(
      /only works for VMobjects/,
    );
  });

  it("Create bounds span [0, alpha]", () => {
    const [a, b] = new Create(new Square())._get_bounds(0.3);
    expectClose(a, 0);
    expectClose(b, 0.3);
  });

  it("base _get_bounds raises NotImplementedError", () => {
    expect(() => new ShowPartial(new Square())._get_bounds(0.5)).toThrowError(
      /Please use Create/,
    );
  });

  it("Create.interpolate lands on the same partial point counts as Python", () => {
    // Python: Square 16 pts; alpha 0.0 -> 4 pts, 0.5 -> 12 pts, first pt [1,1,0].
    const sq = new Square();
    expect(sq.get_num_points()).toBe(16);
    const anim = new Create(sq);
    anim.begin();
    anim.interpolate(0.0);
    expect(sq.get_num_points()).toBe(4);
    anim.interpolate(0.5);
    expect(sq.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(sq.points, 3).slice(0, 1), [[1.0, 1.0, 0.0]]);
    anim.interpolate(1.0);
    expect(sq.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.camera.Camera.transform_points_pre_display
// Python:
//   finite pts pass through; inf/nan -> zeros((1, 3))
// ---------------------------------------------------------------------------
describe("Camera.transform_points_pre_display", () => {
  it("passes finite points through unchanged", () => {
    const cam = new Camera();
    const pts = array([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]) as NDArray;
    const out = cam.transform_points_pre_display(new Square(), pts);
    expectRowsClose(rowsOf(out, 3), [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
  });

  it("collapses non-finite points to a single origin", () => {
    const cam = new Camera();
    const bad = array([[Infinity, 0, 0]]) as NDArray;
    expectRowsClose(rowsOf(cam.transform_points_pre_display(new Square(), bad), 3), [
      [0, 0, 0],
    ]);
    const nan = array([[NaN, 0, 0]]) as NDArray;
    expectRowsClose(rowsOf(cam.transform_points_pre_display(new Square(), nan), 3), [
      [0, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.DrawBorderThenFill
// Python (Square fill_opacity=1):
//   run_time 2, stroke_width 2, stroke_color None, introducer True
//   outline fill opacity 0.0, outline stroke width 2
//   non-VMobject -> TypeError; get_all_mobjects len 3
//   interpolate 0.25 / 0.75 both keep points
// ---------------------------------------------------------------------------
describe("DrawBorderThenFill", () => {
  it("stores defaults and builds a fill-less outline", () => {
    const sq = new Square();
    sq.set_fill(undefined, 1);
    const d = new DrawBorderThenFill(sq);
    expectClose(d.run_time, 2);
    expectClose(d.stroke_width, 2);
    expect(d.stroke_color).toBeNull();
    expect(d.introducer).toBe(true);
    expectClose(d.outline.get_fill_opacity(), 0.0);
    expectClose((d.outline as VMobject).get_stroke_width(), 2);
  });

  it("prefers an explicit stroke color", () => {
    const sq = new Square();
    const d = new DrawBorderThenFill(sq, { stroke_color: "#FF0000" });
    expect(String(d.get_stroke_color(sq))).toMatch(/FF0000/i);
  });

  it("rejects non-vectorized mobjects", () => {
    expect(() => new DrawBorderThenFill(new Mobject() as unknown as VMobject)).toThrowError(
      TypeError,
    );
  });

  it("tracks mobject + starting mobject + outline", () => {
    const d = new DrawBorderThenFill(new Square());
    expect(d.get_all_mobjects().length).toBe(3);
  });

  it("interpolates through both the draw and fill phases", () => {
    const d = new DrawBorderThenFill(new Square());
    d.begin();
    d.interpolate(0.25);
    expect(d.mobject.get_num_points()).toBeGreaterThan(0);
    d.interpolate(0.75);
    expect(d.mobject.get_num_points()).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.moving_camera.MovingCamera
// Python: frame is ScreenRectangle, height 8.0, width 14.222222222222221,
//   center [0,0,0], 1 movement indicator (the frame), cairo cache disabled.
// ---------------------------------------------------------------------------
describe("MovingCamera", () => {
  it("builds a ScreenRectangle frame with config geometry", () => {
    const mc = new MovingCamera();
    expect(mc.frame).toBeInstanceOf(Mobject);
    expect(mc.frame.constructor.name).toBe("ScreenRectangle");
    expectClose(mc.frame_height, 8.0);
    expectClose(mc.frame_width, 14.222222222222221);
    expectRowsClose([[...mc.frame_center]], [[0.0, 0.0, 0.0]]);
  });

  it("reports the frame as the movement indicator", () => {
    const mc = new MovingCamera();
    const moving = mc.get_mobjects_indicating_movement();
    expect(moving.length).toBe(1);
    expect(moving[0]).toBe(mc.frame);
  });

  it("never caches the cairo context", () => {
    const mc = new MovingCamera();
    expect(mc.get_cached_cairo_context(mc.pixel_array)).toBeNull();
    mc.cache_cairo_context(mc.pixel_array, {});
    expect(mc.get_cached_cairo_context(mc.pixel_array)).toBeNull();
  });

  it("frame_width setter resizes the frame", () => {
    const mc = new MovingCamera();
    mc.frame_width = 10;
    expectClose(mc.frame_width, 10.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VGroup
// Python:
//   empty -> 0 submobjects, str "VGroup of 0 submobjects"
//   add(triangle, square) -> 2, repr "VGroup(Square, Circle)"
//   duplicate add ignored (len stays); + builds new, - removes
// ---------------------------------------------------------------------------
describe("VGroup", () => {
  it("starts empty with the group str", () => {
    const vg = new VGroup();
    expect(vg.submobjects.length).toBe(0);
    expect(vg.toString()).toBe("VGroup of 0 submobjects");
  });

  it("adds VMobjects and renders the python repr", () => {
    const vg = new VGroup();
    vg.add(new Square(), new Circle());
    expect(vg.submobjects.length).toBe(2);
    expect(vg.__repr__()).toBe("VGroup(Square, Circle)");
  });

  it("ignores duplicate adds", () => {
    const t = new Square();
    const vg = new VGroup(t);
    vg.add(t);
    expect(vg.submobjects.length).toBe(1);
  });

  it("supports the add/sub operator forms", () => {
    const t = new Square();
    const u = new Circle();
    const vg = new VGroup(t);
    const plus = vg.__add__(u);
    expect(plus.submobjects.length).toBe(2);
    expect(vg.submobjects.length).toBe(1);
    vg.add(u);
    const minus = vg.__sub__(u);
    expect(minus.submobjects.length).toBe(1);
    expect(vg.submobjects.length).toBe(2);
  });

  it("rejects non-VMobjects", () => {
    expect(() => new VGroup().add(new Mobject() as unknown as VMobject)).toThrowError(
      TypeError,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts.TransformMatchingAbstractBase
// Python:
//   base get_mobject_parts/get_mobject_key -> NotImplementedError
//   TransformMatchingShapes(Square(), Circle()) -> 3 anims, to_remove 2,
//     to_add is target; identical squares still give [Transform, FadeOut, FadeIn]
// ---------------------------------------------------------------------------
describe("TransformMatchingAbstractBase", () => {
  it("leaves part/key extraction abstract", () => {
    expect(() => TransformMatchingAbstractBase.get_mobject_parts(new Square())).toThrowError(
      /To be implemented/,
    );
    expect(() => TransformMatchingAbstractBase.get_mobject_key(new Square())).toThrowError(
      /To be implemented/,
    );
  });

  it("builds the animation list for mismatched shapes", () => {
    const target = new Circle();
    const t = new TransformMatchingShapes(new Square(), target);
    expect(t.animations.length).toBe(3);
    expect(t.to_remove.length).toBe(2);
    expect(t.to_add).toBe(target);
    expect(t.animations[0]).toBeInstanceOf(Transform);
  });

  it("matches identical shapes into the transform branch", () => {
    const t = new TransformMatchingShapes(new Square(), new Square());
    expect(t.animations.length).toBe(3);
    expect(t.animations.map((a) => a.constructor.name)).toEqual([
      "Transform",
      "FadeOut",
      "FadeIn",
    ]);
  });

  it("cleans up back to the target mobject", () => {
    const s1 = new Square();
    const s2 = new Square();
    const t = new TransformMatchingShapes(s1, s2);
    const sc = new Scene();
    sc.add(s1);
    for (const anim of t.animations) anim.begin();
    t.clean_up_from_scene(sc);
    expect(sc.mobjects.includes(s2)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeTransform
// Python:
//   defaults stretch True dim 1; mobject is Group of 2
//   begin creates ending_mobject; families zipped rows 2 cols 3
//   ghost_to(empty target) and ghost_to(points) both -> opacity 0
//   cleanup removes group and adds the completion target
// ---------------------------------------------------------------------------
describe("FadeTransform", () => {
  it("wraps source + target copy in a Group with defaults", () => {
    const ft = new FadeTransform(new Square(), new Circle());
    expect(ft.stretch).toBe(true);
    expect(ft.dim_to_match).toBe(1);
    expect(ft.mobject).toBeInstanceOf(Group);
    expect(ft.mobject.submobjects.length).toBe(2);
  });

  it("begin snapshots the ending mobject and zips 2x3 families", () => {
    const ft = new FadeTransform(new Square(), new Circle());
    ft.begin();
    expect(ft.ending_mobject).toBeInstanceOf(Group);
    expect(ft.get_all_mobjects().length).toBe(3);
    const zipped = ft.get_all_families_zipped();
    expect(zipped.length).toBe(2);
    expect(zipped[0]?.length).toBe(3);
  });

  it("ghost_to fades the source regardless of target points", () => {
    const ft = new FadeTransform(new Square(), new Circle());
    const e1 = new Square();
    ft.ghost_to(e1, new Mobject());
    expectClose(e1.get_fill_opacity(), 0.0);
    expectClose(e1.get_stroke_opacity(), 0.0);
    const g1 = new Square();
    ft.ghost_to(g1, new Circle());
    expectClose(g1.get_fill_opacity(), 0.0);
    expectClose(g1.get_stroke_opacity(), 0.0);
  });

  it("interpolates and cleans up to the completion target", () => {
    const a = new Square();
    const b = new Circle();
    const ft = new FadeTransform(a, b);
    const sc = new Scene();
    sc.add(a);
    ft.begin();
    ft.interpolate(0.5);
    ft.interpolate(1.0);
    ft.clean_up_from_scene(sc);
    expect(ft.to_add_on_completion).toBe(b);
    expect(sc.mobjects.includes(b)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.set_camera_orientation
// Python (mock renderer camera):
//   phi/theta/gamma/zoom/focal_distance route to set_*; no args -> no calls;
//   frame_center [1,2,3] -> _frame_center.move_to([1,2,3])
// ---------------------------------------------------------------------------
describe("ThreeDScene.set_camera_orientation", () => {
  function mockScene(): { scene: ThreeDScene; camera: Record<string, ReturnType<typeof vi.fn>> & { _frame_center: { move_to: ReturnType<typeof vi.fn> } } } {
    const scene = new ThreeDScene();
    const camera = {
      set_phi: vi.fn(),
      set_theta: vi.fn(),
      set_gamma: vi.fn(),
      set_zoom: vi.fn(),
      set_focal_distance: vi.fn(),
      _frame_center: { move_to: vi.fn() },
    };
    scene.renderer = { camera };
    return { scene, camera };
  }

  it("routes each orientation kwarg to the renderer camera", () => {
    const { scene, camera } = mockScene();
    scene.set_camera_orientation(0.5, 1.0, 0.1, 2.0, 5.0);
    expect(camera.set_phi).toHaveBeenCalledWith(0.5);
    expect(camera.set_theta).toHaveBeenCalledWith(1.0);
    expect(camera.set_gamma).toHaveBeenCalledWith(0.1);
    expect(camera.set_zoom).toHaveBeenCalledWith(2.0);
    expect(camera.set_focal_distance).toHaveBeenCalledWith(5.0);
  });

  it("is a no-op when nothing is passed", () => {
    const { scene, camera } = mockScene();
    scene.set_camera_orientation();
    expect(camera.set_phi).not.toHaveBeenCalled();
    expect(camera.set_theta).not.toHaveBeenCalled();
    expect(camera.set_gamma).not.toHaveBeenCalled();
    expect(camera.set_zoom).not.toHaveBeenCalled();
    expect(camera.set_focal_distance).not.toHaveBeenCalled();
    expect(camera._frame_center.move_to).not.toHaveBeenCalled();
  });

  it("moves the frame center", () => {
    const { scene, camera } = mockScene();
    scene.set_camera_orientation(null, null, null, null, null, [1, 2, 3]);
    expect(camera._frame_center.move_to).toHaveBeenCalledWith([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (batch symbols x already-ported code)
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Scene.replace composes with top-level filtering", () => {
    // Python: outer.add(inner); sc.add(outer) -> top [outer];
    // replace(inner, fresh) recurses into the child and swaps it.
    // (If inner were ALSO added top-level, replace_in_list would hit the
    // top-level entry first via BFS — so only outer is added here.)
    const sc = new Scene();
    const outer = new Group(new Square());
    const inner = outer.submobjects[0] as Mobject;
    sc.add(outer);
    expect(sc.get_top_level_mobjects()).toEqual([outer]);
    const fresh = new Circle();
    sc.replace(inner, fresh);
    expect(outer.submobjects[0]).toBe(fresh);
    expect(sc.get_top_level_mobjects()).toEqual([outer]);
  });

  it("extract_mobject_family_members feeds Camera display selection", () => {
    // Python: Group(a, b) extracts 3 members by default but only 2 with
    // points; Camera.get_mobjects_to_display uses the with-points variant.
    const a = new Square();
    const b = new Circle();
    const g = new Group(a, b);
    expect(extract_mobject_family_members([g]).length).toBe(3);
    expect(extract_mobject_family_members([g], false, true).length).toBe(2);
    const cam = new Camera();
    expect(cam.get_mobjects_to_display([g]).length).toBe(2);
  });

  it("VGroup + Create + Wait chain matches Python point counts", () => {
    // Python: Create(Square()) at alpha 0.5 -> 12 points; Wait is a no-op
    // around it; VGroup holds both participants.
    const sq = new Square();
    const other = new Circle();
    const vg = new VGroup(sq, other);
    expect(vg.submobjects.length).toBe(2);
    const create = new Create(sq);
    create.begin();
    create.interpolate(0.5);
    expect(sq.get_num_points()).toBe(12);
    const w = new Wait(1.0);
    w.begin();
    w.interpolate(0.5);
    w.finish();
    expect(sq.get_num_points()).toBe(12);
  });

  it("FadeTransform end-to-end against a Scene", () => {
    // Python: begin -> interpolate(0/1) keeps center [0,0,0]; cleanup leaves
    // the completion target in the scene.
    const a = new Square();
    const b = new Circle();
    const ft = new FadeTransform(a, b);
    const sc = new Scene();
    sc.add(a);
    ft.begin();
    ft.interpolate(0.0);
    expectRowsClose([[...(ft.mobject.get_center() as unknown as number[])]], [[0, 0, 0]]);
    ft.interpolate(1.0);
    expectRowsClose([[...(ft.mobject.get_center() as unknown as number[])]], [[0, 0, 0]]);
    ft.clean_up_from_scene(sc);
    expect(sc.mobjects.includes(b)).toBe(true);
  });

  it("MovingCamera frame agrees with Camera pre-display + in-frame checks", () => {
    // Python: default MovingCamera frame center [0,0,0]; finite points pass
    // transform_points_pre_display; unit square at origin is in frame.
    const mc = new MovingCamera();
    expectRowsClose([[...(mc.frame_center as unknown as number[])]], [[0, 0, 0]]);
    const pts = array([[1.0, 2.0, 3.0]]) as NDArray;
    expectRowsClose(rowsOf(mc.transform_points_pre_display(new Square(), pts), 3), [
      [1.0, 2.0, 3.0],
    ]);
    expect(mc.is_in_frame(new Square(1.0))).toBe(true);
    const far = new Square(1.0);
    far.move_to([100, 100, 0]);
    expect(mc.is_in_frame(far)).toBe(false);
  });

  it("DrawBorderThenFill outline feeds a TransformMatchingShapes cleanup", () => {
    // Python: DrawBorderThenFill outline has fill opacity 0; a matching-shapes
    // animation over two squares cleans up back to its target.
    const sq = new Square();
    sq.set_fill(undefined, 1);
    const d = new DrawBorderThenFill(sq);
    expectClose(d.outline.get_fill_opacity(), 0.0);
    const s1 = new Square();
    const s2 = new Square();
    const t = new TransformMatchingShapes(s1, s2);
    expect(t.animations[0]).toBeInstanceOf(Animation);
    const sc = new Scene();
    sc.add(s1);
    for (const anim of t.animations) anim.begin();
    t.clean_up_from_scene(sc);
    expect(sc.mobjects.includes(s2)).toBe(true);
  });
});
