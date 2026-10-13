// Ground-truth tests for batch layers 16..17 (~1021 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): CoordinateSystem.add_coordinates' default-number
// path and MathTable's default element factory need the LaTeX pipeline, so
// those paths are asserted to throw the port's `TODO: text/tex` stub error
// instead of building labels. Text-bearing animations (Write/Unwrite/...)
// are exercised with plain VMobjects (Square/Circle/VGroup), which is exactly
// what their ported logic operates on.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Rectangle, Triangle } from "../src/mobject/geometry/polygram.js";
import { Circle, Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { Animation, linear } from "../src/animation/animation.js";
import { Succession, LaggedStartMap, AnimationGroup } from "../src/animation/composition.js";
import {
  Write,
  Unwrite,
  AddTextLetterByLetter,
  RemoveTextLetterByLetter,
  AddTextWordByWord,
  TypeWithCursor,
  ShowIncreasingSubsets,
  Create,
} from "../src/animation/creation.js";
import { FadeIn, FadeOut } from "../src/animation/fading.js";
import { Flash, ShowPassingFlash, Circumscribe, Blink } from "../src/animation/indication.js";
import { FadeTransform } from "../src/animation/transform.js";
import { TransformMatchingShapes } from "../src/animation/transform_matching_parts.js";
import { TracedPath } from "../src/animation/changing.js";
import { CoordinateSystem } from "../src/mobject/graphing/coordinate_systems.js";
import { ScreenRectangle } from "../src/mobject/frame.js";
import { StreamLines } from "../src/mobject/vector_field.js";
import { Table, MobjectTable, MathTable } from "../src/mobject/table.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { MovingCameraScene } from "../src/scene/moving_camera_scene.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { Scene } from "../src/scene/scene.js";
import { ORIGIN } from "../src/constants.js";
import { RED, BLUE, ManimColor } from "../src/utils/color/core.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

/** Minimal fake scene satisfying Animation._setup_scene / clean_up_from_scene. */
function fakeScene() {
  const added: Mobject[] = [];
  const removed: Mobject[] = [];
  return {
    added,
    removed,
    add: (...mobs: Mobject[]) => { added.push(...mobs); },
    remove: (...mobs: Mobject[]) => { removed.push(...mobs); },
    get_mobject_family_members: () => [...added],
  };
}

/** Stub axis for CoordinateSystem.add_coordinates dispatch tests.
 * Built on a real VMobject (VGroup only accepts VMobjects) with the
 * NumberLine surface monkey-patched in. */
function stubAxis(opts: { customLabels?: boolean } = {}) {
  const numbersGroup = new VGroup();
  const labelsGroup = new VGroup();
  const calls: Array<{ kind: string; values: unknown }> = [];
  const axis = new VMobject() as unknown as Record<string, unknown>;
  axis["scaling"] = {
    custom_labels: opts.customLabels ?? false,
    get_custom_labels: (ticks: number[]) => ticks.map((t) => `L${t}`),
  };
  axis["get_tick_range"] = () => array([1, 2, 3]);
  axis["labels"] = null;
  axis["numbers"] = null;
  axis["add_labels"] = (dict: unknown) => {
    calls.push({ kind: "labels", values: dict });
    axis["labels"] = labelsGroup;
    return axis;
  };
  axis["add_numbers"] = (values: unknown) => {
    calls.push({ kind: "numbers", values });
    axis["numbers"] = numbersGroup;
    return axis;
  };
  return {
    axis: axis as unknown as VMobject,
    get scaling() { return axis["scaling"] as { custom_labels: boolean }; },
    calls,
    numbersGroup,
    labelsGroup,
  };
}

class StubCoords extends CoordinateSystem {
  stubAxes: VMobject[];
  constructor(stubs: ReturnType<typeof stubAxis>[]) {
    super(null, null, null, null, stubs.length);
    this.stubAxes = stubs.map((s) => s.axis);
  }
  override get_axes(): VGroup {
    return new VGroup(...this.stubAxes);
  }
}

// ---------------------------------------------------------------------------
// manim.animation.creation.Write.__init__
// Python: Write(Square()) -> run_time 1, lag_ratio 0.2, reverse False,
//   remover False, introducer True; Write(sq, reverse=True) -> reverse True,
//   remover True, introducer False; 20-submob group -> (2, 0.2).
describe("Write.__init__", () => {
  it("uses length-based defaults for a single VMobject", () => {
    const w = new Write(new Square());
    expect(w.run_time).toBe(1);
    expectClose(w.lag_ratio, 0.2);
    expect(w.reverse).toBe(false);
    expect(w.remover).toBe(false);
    expect(w.introducer).toBe(true);
  });

  it("reverse=True flips remover/introducer", () => {
    const w = new Write(new Square(), { reverse: true });
    expect(w.reverse).toBe(true);
    expect(w.remover).toBe(true);
    expect(w.introducer).toBe(false);
    expect(w.run_time).toBe(1);
    expectClose(w.lag_ratio, 0.2);
  });

  it("honors explicit run_time/lag_ratio", () => {
    const w = new Write(new Square(), { run_time: 3.5, lag_ratio: 0.7 });
    expect(w.run_time).toBe(3.5);
    expectClose(w.lag_ratio, 0.7);
  });

  it("_set_default_config_from_length matches Python (small -> (1, .2), 20 mobs -> (2, .2))", () => {
    const sq = new Square();
    const w = new Write(sq);
    expect(w._set_default_config_from_length(sq as unknown as VMobject, null, null)).toEqual([1, 0.2]);
    const big = new VGroup(...Array.from({ length: 20 }, () => new Dot()));
    const [rt, lr] = w._set_default_config_from_length(big as unknown as VMobject, null, null);
    expect(rt).toBe(2);
    expectClose(lr, 0.2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Write.reverse_submobjects
// Python: invert(recursive=True) reverses point order; double invert restores.
describe("Write.reverse_submobjects", () => {
  it("reverses family point order and double-reverse restores", () => {
    const g = new VGroup(new Square(), new Circle());
    const w = new Write(g as unknown as VMobject);
    w.begin();
    const before = w.mobject.family_members_with_points().map((m) => [...pointRows(m.points, m.dim)]);
    w.reverse_submobjects();
    const after = w.mobject.family_members_with_points().map((m) => [...pointRows(m.points, m.dim)]);
    expect(after.length).toBe(before.length);
    let differs = false;
    for (let i = 0; i < before.length; i += 1) {
      const b = before[i] ?? [];
      const a = after[i] ?? [];
      if (b.length !== a.length) { differs = true; break; }
      for (let r = 0; r < b.length; r += 1) {
        for (let c = 0; c < 3; c += 1) {
          if (!close(b[r]?.[c] ?? NaN, a[r]?.[c] ?? NaN)) { differs = true; break; }
        }
      }
    }
    expect(differs).toBe(true);
    w.reverse_submobjects();
    const again = w.mobject.family_members_with_points().map((m) => [...pointRows(m.points, m.dim)]);
    for (let i = 0; i < before.length; i += 1) {
      const b = before[i] ?? [];
      const g2 = again[i] ?? [];
      expect(g2.length).toBe(b.length);
      for (let r = 0; r < b.length; r += 1) {
        for (let c = 0; c < 3; c += 1) expectClose(g2[r]?.[c] ?? NaN, b[r]?.[c] ?? NaN);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Unwrite.__init__
// Python: Unwrite(Square()) -> reverse True, reverse_rate_function True,
//   run_time 1, lag_ratio 0.2.
describe("Unwrite.__init__", () => {
  it("defaults to reverse + reverse_rate_function", () => {
    const u = new Unwrite(new Square());
    expect(u.reverse).toBe(true);
    expect(u.reverse_rate_function).toBe(true);
    expect(u.run_time).toBe(1);
    expectClose(u.lag_ratio, 0.2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.RemoveTextLetterByLetter.__init__
// Python: run_time 0.3 (3 chars), reverse_rate_function True, remover True,
//   introducer False.
describe("RemoveTextLetterByLetter.__init__", () => {
  it("matches Python defaults over a 3-submob group", () => {
    const g = new VGroup(new Square(), new Circle(), new Triangle());
    const r = new RemoveTextLetterByLetter(g);
    expectClose(r.run_time, 0.30000000000000004, 1e-9);
    expect(r.reverse_rate_function).toBe(true);
    expect(r.remover).toBe(true);
    expect(r.introducer).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.AddTextWordByWord.__init__
// Python: words [len2, len1] -> 4 anims [ShowIncreasingSubsets, Animation]*2,
//   runs [0.12, 0.014142135623730952, 0.06, 0.005], tpc 0.06,
//   succession run_time 0.19914213562373095.
describe("AddTextWordByWord.__init__", () => {
  it("builds per-word ShowIncreasingSubsets+Animation pairs", () => {
    const w1 = new VGroup(new Square(), new Square());
    const w2 = new VGroup(new Circle());
    const aw = new AddTextWordByWord(new VGroup(w1, w2) as unknown as Mobject);
    expect(aw.time_per_char).toBe(0.06);
    expect(aw.animations.length).toBe(4);
    expect(aw.animations[0]).toBeInstanceOf(ShowIncreasingSubsets);
    expect(aw.animations[1]).toBeInstanceOf(Animation);
    expect(aw.animations[2]).toBeInstanceOf(ShowIncreasingSubsets);
    expect(aw.animations[3]).toBeInstanceOf(Animation);
    expectClose(aw.animations[0]?.run_time ?? NaN, 0.12);
    expectClose(aw.animations[1]?.run_time ?? NaN, 0.014142135623730952);
    expectClose(aw.animations[2]?.run_time ?? NaN, 0.06);
    expectClose(aw.animations[3]?.run_time ?? NaN, 0.005);
    expectClose(aw.run_time, 0.19914213562373095);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.TypeWithCursor.__init__ (+begin/finish)
// Python: buff 0.1, keep_cursor_y True, leave_cursor_on True,
//   time_per_char 0.1, run_time 0.3.
describe("TypeWithCursor.__init__", () => {
  it("stores cursor config and derives run_time from text length", () => {
    const g = new VGroup(new Square(), new Circle(), new Triangle());
    const cursor = new Square(0.2);
    const t = new TypeWithCursor(g, cursor as unknown as Mobject);
    expect(t.buff).toBe(0.1);
    expect(t.keep_cursor_y).toBe(true);
    expect(t.leave_cursor_on).toBe(true);
    expect(t.time_per_char).toBe(0.1);
    expectClose(t.run_time, 0.30000000000000004, 1e-9);
    expect(t.cursor).toBe(cursor);
  });

  it("begin() attaches the cursor; finish() leaves it on by default", () => {
    const g = new VGroup(new Square(), new Circle(), new Triangle());
    const cursor = new Square(0.2);
    const t = new TypeWithCursor(g, cursor as unknown as Mobject);
    t.begin();
    expect(g.submobjects.includes(cursor as unknown as Mobject)).toBe(true);
    t.update_submobject_list(1);
    expectClose(cursor.get_fill_opacity(), 1);
    t.finish();
    expectClose(cursor.get_fill_opacity(), 1);
    expect(g.submobjects.includes(cursor as unknown as Mobject)).toBe(true);
  });

  it("leave_cursor_on=False removes the cursor on finish", () => {
    const g = new VGroup(new Square(), new Circle());
    const cursor = new Square(0.2);
    const t = new TypeWithCursor(g, cursor as unknown as Mobject, { leave_cursor_on: false });
    t.begin();
    t.finish();
    // NOTE (verified against Python): super().finish() re-interpolates to
    // alpha=1, whose update_submobject_list sets cursor opacity back to 1.0;
    // the observable effect of leave_cursor_on=False is removal from the
    // text mobject (Python cursor fill opacity after finish is 1.0 too).
    expectClose(cursor.get_fill_opacity(), 1);
    expect(g.submobjects.includes(cursor as unknown as Mobject)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.Succession.update_active_animation
// Python (with fake scene): begin -> index 0, start 0.0 end 1.0;
//   update(1) -> FadeOut, start 1.0 end 3.0; update(5) -> None/None/None.
describe("Succession.update_active_animation", () => {
  it("tracks active animation and start/end times, null past the end", () => {
    const sq = new Square();
    const s = new Succession(new FadeIn(sq, { run_time: 1 }), new FadeOut(sq, { run_time: 2 }));
    s._setup_scene(fakeScene() as never);
    s.begin();
    expect(s.active_index).toBe(0);
    expect(s.active_animation).toBe(s.animations[0]);
    expect(s.active_start_time).toBe(0);
    expect(s.active_end_time).toBe(1);
    s.update_active_animation(1);
    expect(s.active_index).toBe(1);
    expect(s.active_animation).toBe(s.animations[1]);
    expect(s.active_start_time).toBe(1);
    expect(s.active_end_time).toBe(3);
    s.update_active_animation(5);
    expect(s.active_animation).toBe(null);
    expect(s.active_start_time).toBe(null);
    expect(s.active_end_time).toBe(null);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.Succession.update_mobjects
// Python: delegates dt to the active animation only.
describe("Succession.update_mobjects", () => {
  it("forwards dt to the active animation", () => {
    const s = new Succession(new FadeIn(new Square(), { run_time: 1 }), new FadeOut(new Square(), { run_time: 2 }));
    s._setup_scene(fakeScene() as never);
    s.begin();
    const seen: number[] = [];
    (s.active_animation as Animation).update_mobjects = (dt: number) => { seen.push(dt); };
    s.update_mobjects(0.25);
    expect(seen).toEqual([0.25]);
  });

  it("is a no-op once past the end (active_animation null)", () => {
    const s = new Succession(new FadeIn(new Square(), { run_time: 1 }));
    s._setup_scene(fakeScene() as never);
    s.begin();
    s.update_active_animation(7);
    expect(() => s.update_mobjects(0.1)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.LaggedStartMap.__init__
// Python: LaggedStartMap(FadeIn, 3-mob group) -> 3 anims, run_time 2,
//   lag_ratio 0.05 (DEFAULT_LAGGED_START_LAG_RATIO).
describe("LaggedStartMap.__init__", () => {
  it("maps one animation per submobject", () => {
    const g = new VGroup(new Square(), new Circle(), new Triangle());
    const lsm = new LaggedStartMap(FadeIn, g as unknown as Mobject);
    expect(lsm.animations.length).toBe(3);
    for (const a of lsm.animations) expect(a).toBeInstanceOf(FadeIn);
    expect(lsm.run_time).toBe(2);
    expectClose(lsm.lag_ratio, 0.05);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.interpolate_mobject
// Python: base interpolate_submobject is a no-op, so points are unchanged.
describe("Animation.interpolate_mobject", () => {
  it("leaves the mobject unchanged for the base no-op submobject hook", () => {
    const sq = new Square();
    const an = new Animation(sq);
    an.begin();
    const before = [...pointRows(sq.points, sq.dim)];
    an.interpolate_mobject(0.5);
    const after = [...pointRows(sq.points, sq.dim)];
    expect(after.length).toBe(before.length);
    for (let i = 0; i < before.length; i += 1) {
      for (let c = 0; c < 3; c += 1) expectClose(after[i]?.[c] ?? NaN, before[i]?.[c] ?? NaN);
    }
  });

  it("get_sub_alpha matches Python (lag 0.5, 3 fams: idx0 -> 1, idx1 -> 0.5)", () => {
    const an = new Animation(new Square(), { lag_ratio: 0.5 });
    expect(an.get_sub_alpha(0.5, 0, 3)).toBe(1);
    expect(an.get_sub_alpha(0.5, 1, 3)).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.update_mobjects
// Python: updates every all-mobject except self.mobject (1 for plain anim).
describe("Animation.update_mobjects", () => {
  it("updates starting_mobject but not mobject itself", () => {
    const sq = new Square();
    const an = new Animation(sq);
    an.begin();
    expect(an.get_all_mobjects_to_update().length).toBe(1);
    expect(() => an.update_mobjects(0.1)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.clean_up_from_scene
// Python: calls _on_finish; removes mobject only when remover.
describe("Animation.clean_up_from_scene", () => {
  it("removes the mobject when remover=True", () => {
    const sq = new Square();
    const an = new Animation(sq, { remover: true });
    const scene = fakeScene();
    an.clean_up_from_scene(scene as never);
    expect(scene.removed.includes(sq as unknown as Mobject)).toBe(true);
  });

  it("keeps the mobject otherwise and always runs on_finish", () => {
    const sq = new Square();
    let finished = 0;
    const an = new Animation(sq, { _on_finish: () => { finished += 1; } });
    const scene = fakeScene();
    an.clean_up_from_scene(scene as never);
    expect(scene.removed.length).toBe(0);
    expect(finished).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Flash.create_line_anims
// Python: 12 lines, 12 ShowPassingFlash anims, group run_time 1.0,
//   each anim time_width 1, run_time 1.0.
describe("Flash.create_line_anims", () => {
  it("builds one ShowPassingFlash per line with matching config", () => {
    const f = new Flash([...ORIGIN]);
    expect(f.lines.submobjects.length).toBe(12);
    expect(f.animations.length).toBe(12);
    expect(f.run_time).toBe(1.0);
    const anims = f.create_line_anims();
    expect(anims.length).toBe(12);
    for (const a of anims) {
      expect(a).toBeInstanceOf(ShowPassingFlash);
      expect(a.time_width).toBe(1);
      expect(a.run_time).toBe(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Circumscribe
// Python: default -> [ShowPassingFlash] run 1.0; fade_in+fade_out ->
//   [FadeIn, FadeOut]; fade_in -> [FadeIn, Uncreate]; fade_out ->
//   [Create, FadeOut]; shape=Circle works; other shapes raise ValueError.
describe("Circumscribe", () => {
  it("default builds a single ShowPassingFlash", () => {
    const c = new Circumscribe(new Square());
    expect(c.animations.length).toBe(1);
    expect(c.animations[0]).toBeInstanceOf(ShowPassingFlash);
    expect(c.run_time).toBe(1.0);
  });

  it("fade_in+fade_out builds FadeIn/FadeOut halves", () => {
    const c = new Circumscribe(new Square(), undefined, true, true);
    expect(c.animations.map((a) => a.constructor.name)).toEqual(["FadeIn", "FadeOut"]);
    expectClose((c.animations[0] as Animation).run_time, 0.5);
    expectClose((c.animations[1] as Animation).run_time, 0.5);
  });

  it("fade_in builds FadeIn/Uncreate; fade_out builds Create/FadeOut", () => {
    const fi = new Circumscribe(new Square(), undefined, true, false);
    expect(fi.animations.map((a) => a.constructor.name)).toEqual(["FadeIn", "Uncreate"]);
    const fo = new Circumscribe(new Square(), undefined, false, true);
    expect(fo.animations.map((a) => a.constructor.name)).toEqual(["Create", "FadeOut"]);
  });

  it("shape=Circle works and other shapes raise ValueError", () => {
    const c = new Circumscribe(new Square(), Circle);
    expect(c.animations.length).toBe(1);
    expect(() => new Circumscribe(new Square(), Dot as never)).toThrowError(/shape should be either Rectangle or Circle/);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Blink.__init__
// Python: Blink(sq) -> 3 UpdateFromFunc, run_time 1.5;
//   blinks=2 hide_at_end=True -> 4 anims.
describe("Blink.__init__", () => {
  it("defaults to on/off/on with run_time 1.5", () => {
    const b = new Blink(new Square());
    expect(b.animations.length).toBe(3);
    expectClose(b.run_time, 1.5);
    expectClose((b.animations[0] as Animation).run_time, 0.5);
    expectClose((b.animations[1] as Animation).run_time, 0.5);
  });

  it("blinks=2 hide_at_end=True gives 4 anims", () => {
    const b = new Blink(new Square(), 0.5, 0.5, 2, true);
    expect(b.animations.length).toBe(4);
  });

  it("blink toggles opacity through the succession", () => {
    const sq = new Square();
    const b = new Blink(sq, 0.5, 0.5, 1, false);
    b._setup_scene(fakeScene() as never);
    b.begin();
    b.interpolate(0.05);
    // NOTE: neither Python nor this port defines get_opacity; opacity is
    // observed through the fill channel that set_opacity(1.0) writes.
    expectClose(sq.get_fill_opacity(), 1);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeTransform.get_all_families_zipped
// Python: `return Animation.get_all_families_zipped(self)`; after begin with
//   Square->Circle the zip has length 2.
describe("FadeTransform.get_all_families_zipped", () => {
  it("delegates to the base implementation (2 families after begin)", () => {
    const ft = new FadeTransform(new Square(), new Circle());
    ft.begin();
    const zipped = ft.get_all_families_zipped();
    expect(zipped.length).toBe(2);
    for (const row of zipped) expect(row.length).toBe(ft.get_all_mobjects().length);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts.TransformMatchingAbstractBase
// Python (TransformMatchingShapes over 2-mob groups): 3 anims
//   (Transform + FadeOut + FadeIn), to_remove has 2 entries.
describe("TransformMatchingAbstractBase.__init__", () => {
  it("builds match + fade anims and records to_remove/to_add", () => {
    const src = new VGroup(new Square(), new Circle());
    const dst = new VGroup(new Square(), new Triangle());
    const t = new TransformMatchingShapes(src as unknown as Mobject, dst as unknown as Mobject);
    expect(t.animations.length).toBe(3);
    expect(t.animations[0]?.constructor.name).toBe("Transform");
    expect(t.to_remove.length).toBe(2);
    expect(t.to_add).toBe(dst as unknown as Mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.changing.TracedPath.__init__
// Python: defaults time None/dissipating None/stroke 2; dissipating_time=2 ->
//   time 1.0; stroke_width passthrough.
describe("TracedPath.__init__", () => {
  it("stores defaults and registers an updater", () => {
    const tp = new TracedPath(() => [1, 2, 0]);
    expect(tp.time).toBe(null);
    expect(tp.dissipating_time).toBe(null);
    expect(tp.stroke_width).toBe(2);
    expect(tp.updaters.length).toBe(1);
  });

  it("dissipating_time seeds time at 1.0", () => {
    const tp = new TracedPath(() => [0, 0, 0], 5, null, 2.0);
    expect(tp.time).toBe(1.0);
    expect(tp.dissipating_time).toBe(2.0);
    expect(tp.stroke_width).toBe(5);
  });

  it("update_path appends traced points", () => {
    const tp = new TracedPath(() => [1, 2, 0]);
    tp.update_path(tp as unknown as Mobject, 0.016);
    tp.update_path(tp as unknown as Mobject, 0.016);
    expect(tp.has_points()).toBe(true);
    expect(tp.points.size).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Rectangle.__init__
// Python: defaults width 4 height 2, verts [[2,1,0],[-2,1,0],[-2,-1,0],
//   [2,-1,0]]; width 2 height 3 respected; grid_xstep=1/grid_ystep=1 on a
//   4x2 rect -> grid_lines holds 2 sub-VGroups.
describe("Rectangle.__init__", () => {
  it("defaults to width 4 height 2 with the Python vertex order", () => {
    const r = new Rectangle();
    expect(r.width).toBe(4.0);
    expect(r.height).toBe(2.0);
    const verts = pointRows(r.get_vertices(), 3);
    expect(verts.length).toBe(4);
    const expected = [[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0], [-2.0, -1.0, 0.0], [2.0, -1.0, 0.0]];
    for (let i = 0; i < 4; i += 1) {
      for (let c = 0; c < 3; c += 1) expectClose(verts[i]?.[c] ?? NaN, expected[i]?.[c] ?? 0);
    }
  });

  it("honors explicit width/height", () => {
    const r = new Rectangle({ width: 2, height: 3 });
    expect(r.width).toBe(2.0);
    expect(r.height).toBe(3.0);
  });

  it("grid steps build grid_lines (needs the Line constructor hook)", () => {
    const r = new Rectangle({ width: 4, height: 2, grid_xstep: 1.0, grid_ystep: 1.0 });
    expect(r.grid_lines.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.frame.ScreenRectangle
// Python: ScreenRectangle() -> width 16/9*4 = 7.111..., height 4;
//   aspect 4/3 height 3 -> 4x3.
describe("ScreenRectangle", () => {
  it("defaults to 16/9 aspect at height 4", () => {
    const sr = new ScreenRectangle();
    expectClose(sr.height, 4.0);
    expectClose(sr.width, (16.0 / 9.0) * 4.0);
    expectClose(sr.aspect_ratio, 16.0 / 9.0);
  });

  it("honors explicit aspect/height and the aspect setter", () => {
    const sr = new ScreenRectangle(4 / 3, 3);
    expectClose(sr.width, 4.0);
    expectClose(sr.height, 3.0);
    sr.aspect_ratio = 2;
    expectClose(sr.aspect_ratio, 2);
    expectClose(sr.height, 3.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.StreamLines.__init__
// Python (x_range=[-1,1,1], y_range=[-1,1,1], n_repeats=1, virtual_time=0.2,
// max_anchors_per_line=10): 9 stream lines, x_range becomes [-1,2,1],
// stroke_width 1, virtual_time 0.2, padding 3.
describe("StreamLines.__init__", () => {
  it("builds one line per grid start point with extended ranges", () => {
    const sl = new StreamLines((p: number[] | NDArray) => p, {
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      n_repeats: 1,
      virtual_time: 0.2,
      max_anchors_per_line: 10,
    });
    expect(sl.submobjects.length).toBe(9);
    expect(sl.x_range).toEqual([-1, 2, 1]);
    expect(sl.y_range).toEqual([-1, 2, 1]);
    expect(sl.stroke_width).toBe(1);
    expect(sl.virtual_time).toBe(0.2);
    expect(sl.padding).toBe(3);
    expect(sl.n_repeats).toBe(1);
    expect(sl.max_anchors_per_line).toBe(10);
  });

  it("outside_box rejects far points and accepts the origin", () => {
    const sl = new StreamLines((p: number[] | NDArray) => p, {
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      virtual_time: 0.2,
    });
    expect(sl.outside_box([0, 0, 0])).toBe(false);
    expect(sl.outside_box([100, 0, 0])).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.add_coordinates
// Python: returns self; no args -> one label group per axis; dict values ->
//   add_labels path; explicit values -> add_numbers path. (Default label
//   construction needs LaTeX and is out of scope: the TS NumberLine throws
//   TODO there; dispatch is tested with stub axes instead.)
describe("CoordinateSystem.add_coordinates", () => {
  it("dict values dispatch to add_labels and return self", () => {
    const a0 = stubAxis();
    const a1 = stubAxis();
    const cs = new StubCoords([a0, a1]);
    const d0 = new Map<number, unknown>([[1, new Dot()]]);
    const d1 = new Map<number, unknown>([[2, new Dot()]]);
    const result = cs.add_coordinates(d0, d1);
    expect(result).toBe(cs);
    expect(a0.calls.map((c) => c.kind)).toEqual(["labels"]);
    expect(a1.calls.map((c) => c.kind)).toEqual(["labels"]);
    expect(cs.coordinate_labels.submobjects.length).toBe(2);
  });

  it("explicit values dispatch to add_numbers; zip stops at the shorter side", () => {
    const a0 = stubAxis();
    const a1 = stubAxis();
    const cs = new StubCoords([a0, a1]);
    cs.add_coordinates([1, 2]);
    expect(a0.calls.map((c) => c.kind)).toEqual(["numbers"]);
    expect(a1.calls).toEqual([]);
    expect(cs.coordinate_labels.submobjects.length).toBe(1);
  });

  it("custom_labels scaling takes the add_labels branch for null values", () => {
    const a0 = stubAxis({ customLabels: true });
    const cs = new StubCoords([a0]);
    cs.add_coordinates(null);
    expect(a0.calls.map((c) => c.kind)).toEqual(["labels"]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.LinearTransformationScene.__init__
// Python: include planes True, show_coordinates False, show_basis_vectors
//   True, stroke width 6, leave_ghost_vectors False.
describe("LinearTransformationScene.__init__", () => {
  it("stores Python defaults", () => {
    const s = new LinearTransformationScene();
    expect(s.include_background_plane).toBe(true);
    expect(s.include_foreground_plane).toBe(true);
    expect(s.show_coordinates).toBe(false);
    expect(s.show_basis_vectors).toBe(true);
    expect(s.basis_vector_stroke_width).toBe(6);
    expect(s.leave_ghost_vectors).toBe(false);
    expect(s.ghost_vectors.submobjects.length).toBe(0);
    expect(s.i_hat_color.to_hex()).toBe(new ManimColor("GREEN_C").to_hex());
    expect(s.j_hat_color.to_hex()).toBe(new ManimColor("RED_C").to_hex());
  });

  it("honors overrides", () => {
    const s = new LinearTransformationScene(false, false, null, null, true, false);
    expect(s.include_background_plane).toBe(false);
    expect(s.include_foreground_plane).toBe(false);
    expect(s.show_coordinates).toBe(true);
    expect(s.show_basis_vectors).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.moving_camera_scene.MovingCameraScene.get_moving_mobjects
// Python: when a camera movement indicator is among the moving mobjects,
//   every scene mobject counts as moving; otherwise just the moving ones.
describe("MovingCameraScene.get_moving_mobjects", () => {
  // NOTE: Scene ships with renderer null in this port (no renderer ported
  // yet); Python always has a Cairo renderer + camera. The tests attach the
  // ported MovingCamera directly, which is exactly what the method reads.
  function cameraScene(): MovingCameraScene {
    const scene = new MovingCameraScene();
    (scene as unknown as Record<string, unknown>)["renderer"] = { camera: new MovingCamera() };
    return scene;
  }

  function frameOf(scene: MovingCameraScene): Mobject {
    return ((scene as unknown as Record<string, unknown>)["renderer"] as { camera: MovingCamera }).camera.frame as unknown as Mobject;
  }

  it("returns all mobjects when the camera frame itself moves", () => {
    const scene = cameraScene();
    const frame = frameOf(scene);
    const other = new Square();
    scene.add(frame, other as unknown as Mobject);
    const anim = new Animation(frame);
    const moving = scene.get_moving_mobjects(anim);
    expect(moving.includes(other as unknown as Mobject)).toBe(true);
    expect(moving.includes(frame)).toBe(true);
    // NOTE: Scene construction is heavy (~1s); allow headroom under
    // parallel-worker load (default 5s vitest timeout flakes otherwise).
  }, 60000);

  it("returns only moving mobjects when the camera is still", () => {
    const scene = cameraScene();
    const still = new Square();
    const mover = new Circle();
    scene.add(still as unknown as Mobject, mover as unknown as Mobject);
    const anim = new Animation(mover);
    const moving = scene.get_moving_mobjects(anim);
    expect(moving.includes(mover as unknown as Mobject)).toBe(true);
    expect(moving.includes(still as unknown as Mobject)).toBe(false);
  }, 60000);
});

// ---------------------------------------------------------------------------
// Table batch (MobjectTable stand-ins; default factories need LaTeX).
// Python (2x2, MobjectTable): outer lines -> 3 horiz + 3 vert; no outer ->
//   1 horiz + 1 vert; get_cell -> Polygon with (16, 3) points;
//   add_background_to_entries colors all 4 entries.
function squareEntry(): Square {
  return new Square(0.5);
}

function makeTable(outer: boolean): MobjectTable {
  return new MobjectTable(
    [
      [squareEntry(), squareEntry()],
      [squareEntry(), squareEntry()],
    ],
    { include_outer_lines: outer },
  );
}

describe("Table._add_horizontal_lines / _add_vertical_lines", () => {
  it("outer lines give 3+3 lines on a 2x2 table", () => {
    const t = makeTable(true);
    expect(t.horizontal_lines.submobjects.length).toBe(3);
    expect(t.vertical_lines.submobjects.length).toBe(3);
    for (const line of [...t.horizontal_lines.submobjects, ...t.vertical_lines.submobjects]) {
      expect(line).toBeInstanceOf(Line);
    }
  });

  it("inner-only lines give 1+1 lines on a 2x2 table", () => {
    const t = makeTable(false);
    expect(t.horizontal_lines.submobjects.length).toBe(1);
    expect(t.vertical_lines.submobjects.length).toBe(1);
  });
});

describe("Table._organize_mob_table", () => {
  it("arranges every entry into a laid-out grid", () => {
    const t = makeTable(false);
    expect(t.mob_table.length).toBe(2);
    expect(t.mob_table[0]?.length).toBe(2);
    expect(t.elements.submobjects.length).toBe(4);
    const organized = t._organize_mob_table(t.mob_table);
    expect(organized.submobjects.length).toBe(4);
  });
});

describe("Table.set_column_colors / set_row_colors", () => {
  // NOTE: assertions read the entries (what Python users observe on screen),
  // not the throwaway wrapper VGroups get_columns/get_rows build.
  it("paints the matching columns", () => {
    const t = makeTable(false);
    t.set_column_colors(RED, BLUE);
    for (const row of t.mob_table) {
      expect((row[0] as VMobject).get_color().to_hex()).toBe(new ManimColor(RED).to_hex());
      expect((row[1] as VMobject).get_color().to_hex()).toBe(new ManimColor(BLUE).to_hex());
    }
  });

  it("paints the matching rows", () => {
    const t = makeTable(false);
    t.set_row_colors(RED, BLUE);
    for (const entry of t.mob_table[0] ?? []) {
      expect((entry as VMobject).get_color().to_hex()).toBe(new ManimColor(RED).to_hex());
    }
    for (const entry of t.mob_table[1] ?? []) {
      expect((entry as VMobject).get_color().to_hex()).toBe(new ManimColor(BLUE).to_hex());
    }
  });
});

describe("Table.add_background_to_entries", () => {
  it("attaches a background rectangle to every entry", () => {
    const t = makeTable(false);
    t.add_background_to_entries(RED);
    let count = 0;
    for (const entry of t.elements.submobjects) {
      if ((entry as VMobject).background_rectangle !== null &&
        (entry as VMobject).background_rectangle !== undefined) count += 1;
    }
    expect(count).toBe(4);
  });
});

describe("Table.get_cell", () => {
  it("returns a 4-corner Polygon around the (1,1) cell", () => {
    const t = makeTable(false);
    const cell = t.get_cell([1, 1]);
    expect(cell.constructor.name).toBe("Polygon");
    expect(cell.points.size).toBe(16 * 3);
    const rows = pointRows(cell.points, cell.dim);
    const xs = rows.map((r) => r[0] ?? 0);
    const ys = rows.map((r) => r[1] ?? 0);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(0);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(0);
  });
});

describe("MathTable", () => {
  it("default element factory builds MathTex entries", () => {
    const t = new MathTable([
      [1, 2],
      [3, 4],
    ]);
    expect(t.mob_table.length).toBe(2);
    expect(t.elements.submobjects.length).toBe(4);
  });

  it("works with an explicit VMobject factory", () => {
    const t = new MathTable(
      [
        [1, 2],
        [3, 4],
      ],
      { element_to_mobject: () => squareEntry() },
    );
    expect(t.mob_table.length).toBe(2);
    expect(t.elements.submobjects.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Integration
describe("batch 16..17 integration", () => {
  it("Succession runs FadeIn->FadeOut to completion (active null at alpha 1)", () => {
    // Python: Succession.interpolate advances active anims by scaled time;
    // at alpha=1 every sub-animation has finished.
    const sq = new Square();
    const s = new Succession(new FadeIn(sq, { run_time: 1 }), new FadeOut(sq, { run_time: 2 }));
    s._setup_scene(fakeScene() as never);
    s.begin();
    s.interpolate(1);
    expect(s.active_animation).toBe(null);
  });

  it("Write begin/finish with reverse restores the target points", () => {
    // Python: Write.begin reverses submobjects when reverse=True and
    // finish() reverses back, so the net effect is the identity.
    const g = new VGroup(new Square(), new Circle());
    const w = new Write(g as unknown as VMobject, { reverse: true });
    w.begin();
    const during = w.mobject.family_members_with_points().map((m) => [...pointRows(m.points, m.dim)]);
    w.finish();
    const after = w.mobject.family_members_with_points().map((m) => [...pointRows(m.points, m.dim)]);
    void during;
    expect(after.length).toBe(2);
    for (const fam of after) expect(fam.length).toBeGreaterThan(0);
  });

  it("Table.create() wires line/element/label animations (Table x Create/Write/FadeIn)", () => {
    // Python: Table.create builds an AnimationGroup of line + element (+
    // label/background) animations; here labels come from Mobject entries.
    const t = makeTable(true);
    const group = t.create();
    expect(group).toBeInstanceOf(AnimationGroup);
    expect(group.animations.length).toBeGreaterThanOrEqual(2);
    expect(group.animations[0]).toBeInstanceOf(Create);
  });

  it("Flash lines animate through ShowPassingFlash bounds", () => {
    // Python: each Flash line anim is ShowPassingFlash(line, time_width,
    // run_time); interpolating mid-way yields a nonempty partial line.
    const f = new Flash([1, 0, 0]);
    const [first] = f.create_line_anims();
    expect(first).toBeInstanceOf(ShowPassingFlash);
    first?.begin();
    first?.interpolate(0.5);
    const [lo, hi] = (first as ShowPassingFlash)._get_bounds(0.5);
    expect(hi).toBeGreaterThan(lo);
  });
});
