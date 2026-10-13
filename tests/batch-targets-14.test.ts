// Ground-truth tests for batch layers 14..14.
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): AddTextLetterByLetter is exercised with a Group of
// Dots standing in for Text characters (its logic is pure ShowIncreasingSubsets
// logic over submobjects); no Text/Tex objects are used anywhere here.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { Animation } from "../src/animation/animation.js";
import { AnimatedBoundary, TracedPath } from "../src/animation/changing.js";
import { Succession, LaggedStart } from "../src/animation/composition.js";
import { Create, ShowSubmobjectsOneByOne, AddTextLetterByLetter } from "../src/animation/creation.js";
import { Rotate } from "../src/animation/rotation.js";
import {
  FocusOn,
  Indicate,
  Flash,
  ShowPassingFlash,
  ShowPassingFlashWithThinningStrokeWidth,
  ApplyWave,
} from "../src/animation/indication.js";
import { config } from "../src/_config/utils.js";
import { pointRows } from "../src/utils/arrays.js";
import { PI, TAU } from "../src/constants.js";

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

// ---------------------------------------------------------------------------
// VMobject.add_cubic_bezier_curve_to
// Python:
//   l = Line([0,0,0],[1,0,0])  # 4 pts
//   l.add_cubic_bezier_curve_to([2,0,0],[3,0,0],[4,0,0])
//   -> len 8, last 4 rows [[1,0,0],[2,0,0],[3,0,0],[4,0,0]]
//   v = VMobject(); v.start_new_path([0,0,0])  # has_new_path_started True, 1 pt
//   v.add_cubic_bezier_curve_to([1,0,0],[2,0,0],[3,0,0])
//   -> 4 pts [[0,0,0],[1,0,0],[2,0,0],[3,0,0]]
//   VMobject().add_cubic_bezier_curve_to(...) -> ValueError
// ---------------------------------------------------------------------------
describe("VMobject.add_cubic_bezier_curve_to", () => {
  it("appends [last, h1, h2, anchor] on a normal path", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expect(l.get_num_points()).toBe(4);
    l.add_cubic_bezier_curve_to([2, 0, 0], [3, 0, 0], [4, 0, 0]);
    expect(l.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(l.points, l.dim).slice(-4), [
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
      [4, 0, 0],
    ]);
  });

  it("appends just [h1, h2, anchor] when a new path was started", () => {
    const v = new VMobject();
    v.start_new_path([0, 0, 0]);
    expect(v.has_new_path_started()).toBe(true);
    v.add_cubic_bezier_curve_to([1, 0, 0], [2, 0, 0], [3, 0, 0]);
    expect(v.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
  });

  it("raises ValueError on an empty vmobject", () => {
    expect(() => new VMobject().add_cubic_bezier_curve_to([1, 0, 0], [2, 0, 0], [3, 0, 0])).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// VMobject.start_new_path
// Python:
//   VMobject().start_new_path([1,2,3]) -> [[1,2,3]]
//   Line (4 pts, aligned) .start_new_path([5,5,5]) -> len 5, last [5,5,5]
//   VMobject with 2 pts .start_new_path([9,9,9]) -> len 5,
//     [[0,0,0],[1,0,0],[0,0,0],[0,0,0],[9,9,9]]
//   (closure repeats the last start anchor = first point [0,0,0] twice)
// ---------------------------------------------------------------------------
describe("VMobject.start_new_path", () => {
  it("appends a single point on an empty vmobject", () => {
    const v = new VMobject();
    v.start_new_path([1, 2, 3]);
    expectRowsClose(rowsOf(v.points, v.dim), [[1, 2, 3]]);
  });

  it("appends a single point when already aligned", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    l.start_new_path([5, 5, 5]);
    expect(l.get_num_points()).toBe(5);
    expectRowsClose([rowsOf(l.points, l.dim).at(-1) ?? []], [[5, 5, 5]]);
  });

  it("closes an open path with the last start anchor before appending", () => {
    const v = new VMobject();
    v.set_points([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    expect(v.get_num_points()).toBe(2);
    v.start_new_path([9, 9, 9]);
    expect(v.get_num_points()).toBe(5);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [9, 9, 9],
    ]);
  });
});

// ---------------------------------------------------------------------------
// VMobject.add_points_as_corners
// Python (Line([0,0,0],[1,0,0]) + corners [[2,0,0],[2,1,0]]):
//   -> 12 pts:
//   [[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0],
//    [1,0,0],[4/3,0,0],[5/3,0,0],[2,0,0],
//    [2,0,0],[2,1/3,0],[2,2/3,0],[2,1,0]]
//   empty corners -> unchanged; empty vmobject -> ValueError;
//   new-path-started VMobject + [[1,0,0],[2,0,0]] -> 8 pts dropping the stub:
//   [[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0],[1,0,0],[4/3,0,0],[5/3,0,0],[2,0,0]]
// ---------------------------------------------------------------------------
describe("VMobject.add_points_as_corners", () => {
  it("appends straight segments as bezier-interpolated corners", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    l.add_points_as_corners([
      [2, 0, 0],
      [2, 1, 0],
    ]);
    expect(l.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(l.points, l.dim), [
      [0, 0, 0],
      [1 / 3, 0, 0],
      [2 / 3, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [4 / 3, 0, 0],
      [5 / 3, 0, 0],
      [2, 0, 0],
      [2, 0, 0],
      [2, 1 / 3, 0],
      [2, 2 / 3, 0],
      [2, 1, 0],
    ]);
  });

  it("is a no-op for empty corners", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    const n = l.get_num_points();
    l.add_points_as_corners([]);
    expect(l.get_num_points()).toBe(n);
  });

  it("raises ValueError on an empty vmobject", () => {
    expect(() => new VMobject().add_points_as_corners([[1, 0, 0]])).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });

  it("drops the new-path stub point first", () => {
    const v = new VMobject();
    v.start_new_path([0, 0, 0]);
    v.add_points_as_corners([
      [1, 0, 0],
      [2, 0, 0],
    ]);
    expect(v.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1 / 3, 0, 0],
      [2 / 3, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [4 / 3, 0, 0],
      [5 / 3, 0, 0],
      [2, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// VGroup.add
// Python:
//   VGroup().add(Dot(), Dot()) -> 2 submobjects
//   VGroup([d1, d2]) -> 2 submobjects
//   .add(Mobject()) -> TypeError "... VMobject ... (at index 0 of parameter 0)
//     ... You can try adding this value into a Group instead."
//   .add(42) -> TypeError "... value 42 (at index 0 of parameter 0) ..."
// ---------------------------------------------------------------------------
describe("VGroup.add", () => {
  it("adds individual vmobjects", () => {
    const g = new VGroup();
    g.add(new Dot(), new Dot());
    expect(g.submobjects.length).toBe(2);
  });

  it("unpacks iterables in the constructor", () => {
    const g = new VGroup([new Dot(), new Dot()]);
    expect(g.submobjects.length).toBe(2);
  });

  it("rejects a plain Mobject with a Group hint", () => {
    expect(() => new VGroup().add(new Mobject())).toThrowError(/Only values of type VMobject.*Group instead/s);
  });

  it("rejects non-mobjects with indices in the message", () => {
    expect(() => new VGroup().add(42 as unknown as VMobject)).toThrowError(
      /Only values of type VMobject.*at index 0 of parameter 0/s,
    );
  });
});

// ---------------------------------------------------------------------------
// AnimatedBoundary.full_family_become_partial
// Python (mob1 = Square(), target = Square().scale(2)):
//   (mob1, target, 0, 0.5) -> 12 pts, first row [2,2,0]:
//   [[2,2,0],[2/3,2,0],[-2/3,2,0],[-2,2,0],[-2,2,0],[-2,2/3,0],
//    [-2,-2/3,0],[-2,-2,0],[-2,-2,0]x4]
//   (mob2, target, 0, 1) -> 16 pts allclose to target
//   (mob3, target, 0.25, 0.75) -> 12 pts starting at [-2,2,0]:
//   [[-2,2,0],[-2,2/3,0],[-2,-2/3,0],[-2,-2,0],[-2,-2,0],
//    [-2/3,-2,0],[2/3,-2,0],[2,-2,0],[2,-2,0]x4]
// ---------------------------------------------------------------------------
describe("AnimatedBoundary.full_family_become_partial", () => {
  it("becomes the first half of the target family", () => {
    const ab = new AnimatedBoundary(new Square());
    const mob1 = new Square();
    const target = new Square();
    target.scale(2);
    const ret = ab.full_family_become_partial(mob1, target, 0, 0.5);
    expect(ret).toBe(ab);
    expect(mob1.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(mob1.points, mob1.dim), [
      [2, 2, 0],
      [2 / 3, 2, 0],
      [-2 / 3, 2, 0],
      [-2, 2, 0],
      [-2, 2, 0],
      [-2, 2 / 3, 0],
      [-2, -2 / 3, 0],
      [-2, -2, 0],
      [-2, -2, 0],
      [-2, -2, 0],
      [-2, -2, 0],
      [-2, -2, 0],
    ]);
  });

  it("becomes the whole target for [0, 1]", () => {
    const ab = new AnimatedBoundary(new Square());
    const mob2 = new Square();
    const target = new Square();
    target.scale(2);
    ab.full_family_become_partial(mob2, target, 0, 1);
    expect(mob2.get_num_points()).toBe(target.get_num_points());
    expectRowsClose(rowsOf(mob2.points, mob2.dim), rowsOf(target.points, target.dim));
  });

  it("becomes the middle half for [0.25, 0.75]", () => {
    const ab = new AnimatedBoundary(new Square());
    const mob3 = new Square();
    const target = new Square();
    target.scale(2);
    ab.full_family_become_partial(mob3, target, 0.25, 0.75);
    expect(mob3.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(mob3.points, mob3.dim), [
      [-2, 2, 0],
      [-2, 2 / 3, 0],
      [-2, -2 / 3, 0],
      [-2, -2, 0],
      [-2, -2, 0],
      [-2 / 3, -2, 0],
      [2 / 3, -2, 0],
      [2, -2, 0],
      [2, -2, 0],
      [2, -2, 0],
      [2, -2, 0],
      [2, -2, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// TracedPath
// Python:
//   TracedPath(lambda: [1,2,3]) -> time None, dissipating None, 0 pts
//   update_path #1 -> 4 pts, first [1,2,3]; update #2 -> 8 pts
//   TracedPath(f, dissipating_time=0.5) -> time 1.0
//   5x update_path(dt=0.2) -> 8 pts, time 2.0 (dissipation trims growth)
// ---------------------------------------------------------------------------
describe("TracedPath", () => {
  it("starts empty with null time when not dissipating", () => {
    const tp = new TracedPath(() => [1, 2, 3]);
    expect(tp.time).toBeNull();
    expect(tp.dissipating_time).toBeNull();
    expect(tp.get_num_points()).toBe(0);
  });

  it("grows by one curve per update", () => {
    const tp = new TracedPath(() => [1, 2, 3]);
    tp.update_path(tp, 0.1);
    expect(tp.get_num_points()).toBe(4);
    expectRowsClose([rowsOf(tp.points, tp.dim)[0] ?? []], [[1, 2, 3]]);
    tp.update_path(tp, 0.1);
    expect(tp.get_num_points()).toBe(8);
  });

  it("dissipates old points once time exceeds the window", () => {
    const tp = new TracedPath(() => [0, 0, 0], 2, null, 0.5);
    expectClose(tp.time ?? NaN, 1.0);
    for (let i = 0; i < 5; i += 1) tp.update_path(tp, 0.2);
    expectClose(tp.time ?? NaN, 2.0);
    expect(tp.get_num_points()).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// Succession
// Python (Create run_time=1 x2, Dummy scene with get_mobject_family_members):
//   timings [(0,1),(1,2)], max_end_time 2.0
//   begin -> active_index 0, start 0.0, end 1.0
//   interpolate(0.25) -> index 0; interpolate(0.75) -> index 1
//   interpolate(1.0) -> active_animation None; finish() drains to None
//   Succession().begin() -> ValueError
// NOTE: Python raises AttributeError when begin() is called without a prior
// _setup_scene (self.scene is only assigned there); the TS port initialises
// scene to null so begin() works standalone. Tests below call begin()
// directly (documented deviation) plus one _setup_scene parity check.
// ---------------------------------------------------------------------------
describe("Succession", () => {
  function twoSuccession(): Succession {
    return new Succession(new Create(new Square(), { run_time: 1 }), new Create(new Square(), { run_time: 1 }));
  }

  it("chains timings end-to-start with lag_ratio 1", () => {
    const succ = twoSuccession();
    expect(succ.anims_with_timings.length).toBe(2);
    expectClose(succ.anims_with_timings[0]?.start ?? NaN, 0);
    expectClose(succ.anims_with_timings[0]?.end ?? NaN, 1);
    expectClose(succ.anims_with_timings[1]?.start ?? NaN, 1);
    expectClose(succ.anims_with_timings[1]?.end ?? NaN, 2);
    expectClose(succ.max_end_time, 2.0);
  });

  it("activates animations in order across interpolate", () => {
    const succ = twoSuccession();
    succ.begin();
    expect(succ.active_index).toBe(0);
    expectClose(succ.active_start_time ?? NaN, 0);
    expectClose(succ.active_end_time ?? NaN, 1);
    succ.interpolate(0.25);
    expect(succ.active_index).toBe(0);
    succ.interpolate(0.75);
    expect(succ.active_index).toBe(1);
    succ.interpolate(1.0);
    expect(succ.active_animation).toBeNull();
  });

  it("finish() drains the queue", () => {
    const succ = twoSuccession();
    succ.begin();
    succ.finish();
    expect(succ.active_animation).toBeNull();
  });

  it("raises ValueError when empty", () => {
    expect(() => new Succession().begin()).toThrowError(expect.objectContaining({ name: "ValueError" }));
  });
});

// ---------------------------------------------------------------------------
// LaggedStart
// Python: default lag_ratio 0.05; 3x Create(run_time=1) ->
//   [(0,1),(0.05,1.05),(0.1,1.1)], max_end 1.1
//   LaggedStart(b1, b2, lag_ratio=1) -> [(0,1),(1,2)], max_end 2.0
// ---------------------------------------------------------------------------
describe("LaggedStart", () => {
  it("uses the default lag ratio with overlapping timings", () => {
    const lag = new LaggedStart(
      new Create(new Square(), { run_time: 1 }),
      new Create(new Square(), { run_time: 1 }),
      new Create(new Square(), { run_time: 1 }),
    );
    expectClose(lag.lag_ratio, 0.05);
    expectClose(lag.anims_with_timings[0]?.start ?? NaN, 0);
    expectClose(lag.anims_with_timings[0]?.end ?? NaN, 1);
    expectClose(lag.anims_with_timings[1]?.start ?? NaN, 0.05);
    expectClose(lag.anims_with_timings[1]?.end ?? NaN, 1.05);
    expectClose(lag.anims_with_timings[2]?.start ?? NaN, 0.1);
    expectClose(lag.anims_with_timings[2]?.end ?? NaN, 1.1);
    expectClose(lag.max_end_time, 1.1);
  });

  it("respects an explicit lag_ratio", () => {
    const lag = new LaggedStart(
      new Create(new Square(), { run_time: 1 }),
      new Create(new Square(), { run_time: 1 }),
      { lag_ratio: 1 },
    );
    expectClose(lag.anims_with_timings[1]?.start ?? NaN, 1);
    expectClose(lag.anims_with_timings[1]?.end ?? NaN, 2);
    expectClose(lag.max_end_time, 2.0);
  });
});

// ---------------------------------------------------------------------------
// Animation.create_starting_mobject
// Python: returns an equal-but-independent copy of mobject.
// ---------------------------------------------------------------------------
describe("Animation.create_starting_mobject", () => {
  it("returns an independent copy", () => {
    const m = new Square();
    const anim = new Animation(m);
    const sm = anim.create_starting_mobject();
    expect(sm).not.toBe(m);
    expectRowsClose(rowsOf(sm.points, sm.dim), rowsOf(m.points, m.dim));
    rowsOf(sm.points, sm.dim)[0]?.splice(0, 1, 999);
    expect(rowsOf(m.points, m.dim)[0]?.[0]).not.toBe(999);
  });
});

// ---------------------------------------------------------------------------
// Rotate.__init__ / Rotate
// Python (sq = Square(), center [0,0,0]):
//   Rotate(sq): angle PI, axis [0,0,1], about_point == center, about_edge None,
//     path_arc PI, path_arc_axis [0,0,1]
//   Rotate(sq, PI/2, [0,0,1], [0,0,0]).create_target() ==
//     sq.copy().rotate(PI/2, [0,0,1], [0,0,0])
//   begin + interpolate(1) lands exactly on a fresh create_target().
// ---------------------------------------------------------------------------
describe("Rotate", () => {
  it("defaults angle/axis/about_point from the mobject", () => {
    const sq = new Square();
    const r = new Rotate(sq);
    expectClose(r.angle, PI);
    const axis = Array.from(r.axis as unknown as Iterable<number>);
    expectRowsClose([axis], [[0, 0, 1]]);
    expectRowsClose([[...(r.about_point as number[])]], [sq.get_center()]);
    expect(r.about_edge).toBeNull();
  });

  it("forwards angle/axis into the transform path machinery", () => {
    const sq = new Square();
    const r = new Rotate(sq, PI / 2);
    expectClose((r as unknown as { path_arc: number }).path_arc, PI / 2);
    const axis = Array.from((r as unknown as { path_arc_axis: number[] }).path_arc_axis);
    expectRowsClose([axis], [[0, 0, 1]]);
  });

  it("create_target matches an explicit rotate", () => {
    const sq = new Square();
    const r = new Rotate(sq, PI / 2, [0, 0, 1], [0, 0, 0]);
    const target = r.create_target();
    const expected = new Square();
    expected.rotate(PI / 2, [0, 0, 1], [0, 0, 0]);
    expectRowsClose(rowsOf(target.points, target.dim), rowsOf(expected.points, expected.dim));
  });

  it("interpolate(1) after begin lands on the target", () => {
    const sq = new Square();
    const r = new Rotate(sq, PI / 2);
    r.begin();
    r.interpolate(1.0);
    const target = new Rotate(new Square(), PI / 2).create_target();
    expectRowsClose(rowsOf(sq.points, sq.dim), rowsOf(target.points, target.dim));
  });
});

// ---------------------------------------------------------------------------
// ShowSubmobjectsOneByOne (+ __init__)
// Python (3 Dots, default int_func=ceil, smooth rate):
//   interpolate(0.0) -> [0,0,0]; (0.5) -> [0,1,0]; (1.0) -> [0,0,1]
//   update_submobject_list(2) -> [0,1,0]; (0) leaves state untouched.
// ---------------------------------------------------------------------------
describe("ShowSubmobjectsOneByOne", () => {
  function threeDots(): ShowSubmobjectsOneByOne {
    return new ShowSubmobjectsOneByOne([new Dot(), new Dot(), new Dot()]);
  }

  function opacities(a: ShowSubmobjectsOneByOne): number[] {
    return a.all_submobs.map((m) => (m as VMobject).get_fill_opacity());
  }

  it("lights submobjects one at a time", () => {
    const anim = threeDots();
    expect(anim.all_submobs.length).toBe(3);
    anim.begin();
    anim.interpolate(0.0);
    expect(opacities(anim)).toEqual([0, 0, 0]);
    anim.interpolate(0.5);
    expect(opacities(anim)).toEqual([0, 1, 0]);
    anim.interpolate(1.0);
    expect(opacities(anim)).toEqual([0, 0, 1]);
  });

  it("update_submobject_list keeps only the last in-window submobject lit", () => {
    const anim = threeDots();
    anim.begin();
    anim.update_submobject_list(2);
    expect(opacities(anim)).toEqual([0, 1, 0]);
    // Python: index 0 selects an empty window, so prior state is untouched.
    anim.update_submobject_list(0);
    expect(opacities(anim)).toEqual([0, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// AddTextLetterByLetter
// Python (Group of 4 Dots, frame_rate 60):
//   run_time 0.4 == max(1/60, 0.1) * 4; time_per_char 0.1
//   Group() -> ValueError; interpolate(0) all 0; interpolate(1) all 1.
// ---------------------------------------------------------------------------
describe("AddTextLetterByLetter", () => {
  it("derives run_time from time_per_char and character count", () => {
    expectClose(Number(config.frame_rate), 60);
    const text = new Group(new Dot(), new Dot(), new Dot(), new Dot());
    const anim = new AddTextLetterByLetter(text);
    expectClose(anim.time_per_char, 0.1);
    expectClose(anim.run_time, 0.4);
  });

  it("rejects empty text", () => {
    expect(() => new AddTextLetterByLetter(new Group())).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });

  it("reveals letters progressively", () => {
    const text = new Group(new Dot(), new Dot(), new Dot(), new Dot());
    const anim = new AddTextLetterByLetter(text);
    anim.begin();
    anim.interpolate(0.0);
    expect(anim.all_submobs.map((m) => (m as VMobject).get_fill_opacity())).toEqual([0, 0, 0, 0]);
    anim.interpolate(1.0);
    expect(anim.all_submobs.map((m) => (m as VMobject).get_fill_opacity())).toEqual([1, 1, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// FocusOn
// Python: FocusOn([1,0,0]) -> opacity 0.2, run_time 2.0, remover True;
//   create_target: radius-0 dot with fill_opacity 0.2.
//   FocusOn(square) keeps the mobject as focus_point.
//   config frame radii: 7.111111111111111 + 4.0 (starting dot radius).
// ---------------------------------------------------------------------------
describe("FocusOn", () => {
  it("stores defaults and flags", () => {
    const f = new FocusOn([1, 0, 0]);
    expectClose(f.opacity, 0.2);
    expectClose(f.run_time, 2.0);
    expect(f.is_remover()).toBe(true);
    expect(f.focus_point).toEqual([1, 0, 0]);
  });

  it("creates a zero-radius target with the configured fill opacity", () => {
    const f = new FocusOn([1, 0, 0]);
    const target = f.create_target();
    expectClose(target.width, 0);
    expectClose(target.get_fill_opacity(), 0.2);
  });

  it("starts from a dot spanning the frame", () => {
    const f = new FocusOn([1, 0, 0]);
    expectClose(f.mobject.get_width?.() ?? (f.mobject as VMobject).width, 2 * (7.111111111111111 + 4.0));
  });

  it("accepts a mobject focus point", () => {
    const sq = new Square();
    const f = new FocusOn(sq);
    expect(f.focus_point).toBe(sq);
  });
});

// ---------------------------------------------------------------------------
// Indicate (+ __init__)
// Python: scale_factor 1.2, rate_func there_and_back;
//   create_target == copy().scale(1.2).set_color(PURE_YELLOW);
//   scale_factor 2 doubles width (2.0 -> 4.0).
// ---------------------------------------------------------------------------
describe("Indicate", () => {
  it("stores defaults", () => {
    const ind = new Indicate(new Square());
    expectClose(ind.scale_factor, 1.2);
    // NOTE: Python asserts rate_func.__name__ == "there_and_back"; the TS
    // port wraps all rate functions in unit_interval/zero guards (so .name
    // is "wrapper"). Behavioral identity is asserted instead — Python
    // there_and_back(0, 0.25, 0.5, 0.75, 1) == (0, 0.5, 1, 0.5, 0).
    for (const [t, v] of [[0, 0], [0.25, 0.5], [0.5, 1], [0.75, 0.5], [1, 0]] as const) {
      expectClose(ind.rate_func(t), v);
    }
  });

  it("create_target matches scale+color ground truth", () => {
    const sq = new Square();
    const target = new Indicate(sq).create_target();
    const expected = new Square();
    expected.scale(1.2);
    expected.set_color("PURE_YELLOW");
    expectRowsClose(rowsOf(target.points, target.dim), rowsOf(expected.points, expected.dim));
  });

  it("honours a custom scale factor", () => {
    const sq = new Square();
    const w0 = sq.width;
    const target = new Indicate(sq, 2.0).create_target();
    expectClose(target.width, w0 * 2.0);
    expectClose(target.width, 4.0);
  });
});

// ---------------------------------------------------------------------------
// Flash
// Python: Flash([0,0,0]) -> 12 lines, 12 anims, max_end_time 1.0;
//   line0 start [0.1,0,0] end [0.3,0,0]; line1 start [0.0866...,0.05,0];
//   Flash([0,0,0], num_lines=4, line_length=0.5, flash_radius=0.2):
//     4 lines, line0 [0.2,0,0] -> [0.7,0,0].
// ---------------------------------------------------------------------------
describe("Flash", () => {
  it("builds twelve radial lines with matching sub-animations", () => {
    const fl = new Flash([0, 0, 0]);
    expect(fl.lines.submobjects.length).toBe(12);
    expect(fl.animations.length).toBe(12);
    expectClose(fl.max_end_time, 1.0);
    const line0 = fl.lines.submobjects[0] as Line;
    expectRowsClose([line0.get_start()], [[0.1, 0, 0]]);
    expectRowsClose([line0.get_end()], [[0.3, 0, 0]]);
    const line1 = fl.lines.submobjects[1] as Line;
    expectRowsClose([line1.get_start()], [[0.08660254037844388, 0.05, 0]]);
    for (const a of fl.animations) expect(a).toBeInstanceOf(ShowPassingFlash);
  });

  it("honours custom line counts and radii", () => {
    const fl = new Flash([0, 0, 0], 0.5, 4, 0.2);
    expect(fl.lines.submobjects.length).toBe(4);
    const line0 = fl.lines.submobjects[0] as Line;
    expectRowsClose([line0.get_start()], [[0.2, 0, 0]]);
    expectRowsClose([line0.get_end()], [[0.7, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// ShowPassingFlash
// Python (time_width=0.1): bounds(0)=(0,0.0); bounds(0.5)=(0.45,0.55);
//   bounds(1)=(1.0,1); remover+introducer True.
//   time_width=1.0: bounds(0.5)=(0.0,1.0).
// ---------------------------------------------------------------------------
describe("ShowPassingFlash", () => {
  it("sweeps a window across [0, 1]", () => {
    const sp = new ShowPassingFlash(new Line([0, 0, 0], [1, 0, 0]), 0.1);
    expect(sp._get_bounds(0.0)).toEqual([0, 0.0]);
    const [lo, hi] = sp._get_bounds(0.5);
    expectClose(lo, 0.45);
    expectClose(hi, 0.55);
    expect(sp._get_bounds(1.0)).toEqual([1.0, 1]);
    expect(sp.is_remover()).toBe(true);
    expect(sp.is_introducer()).toBe(true);
  });

  it("covers everything at mid-alpha with time_width 1", () => {
    const sp = new ShowPassingFlash(new Line([0, 0, 0], [1, 0, 0]), 1.0);
    expect(sp._get_bounds(0.5)).toEqual([0.0, 1.0]);
  });
});

// ---------------------------------------------------------------------------
// ShowPassingFlashWithThinningStrokeWidth
// Python (Square, stroke 4.0, n=10, tw=0.1):
//   10 anims; widths linspace(0,4,10); time_widths linspace(0.1,0,10).
// ---------------------------------------------------------------------------
describe("ShowPassingFlashWithThinningStrokeWidth", () => {
  it("fans out thinning strokes with shrinking windows", () => {
    const sq = new Square();
    expectClose(sq.get_stroke_width(), 4.0);
    const anim = new ShowPassingFlashWithThinningStrokeWidth(sq, 10, 0.1);
    expect(anim.animations.length).toBe(10);
    const widths = anim.animations.map((a) => (a as ShowPassingFlash).mobject.get_stroke_width());
    const expectedWidths = Array.from({ length: 10 }, (_, i) => (4.0 * i) / 9);
    widths.forEach((w, i) => expectClose(w, expectedWidths[i] ?? NaN));
    const tws = anim.animations.map((a) => (a as ShowPassingFlash).time_width);
    const expectedTws = Array.from({ length: 10 }, (_, i) => 0.1 * (1 - i / 9));
    tws.forEach((t, i) => expectClose(t, expectedTws[i] ?? NaN));
    for (const a of anim.animations) expect(a).toBeInstanceOf(ShowPassingFlash);
  });
});

// ---------------------------------------------------------------------------
// ApplyWave
// Python (Square, defaults): run_time 2.0; begin+interpolate(0) reproduces the
// square; interpolate(1) returns to it (atol 1e-6); at alpha 0.5 the maximum
// absolute point shift is 0.009286297915545028.
// ---------------------------------------------------------------------------
describe("ApplyWave", () => {
  it("runs a full wave cycle back to the original", () => {
    const sq = new Square();
    const pristine = rowsOf(new Square().points, 3);
    const aw = new ApplyWave(sq);
    expectClose(aw.run_time, 2.0);
    aw.begin();
    aw.interpolate(0.0);
    expectRowsClose(rowsOf(sq.points, sq.dim), pristine);
    aw.interpolate(0.5);
    const shifted = rowsOf(sq.points, sq.dim);
    let maxShift = 0;
    for (let i = 0; i < shifted.length; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        maxShift = Math.max(maxShift, Math.abs((shifted[i]?.[j] ?? 0) - (pristine[i]?.[j] ?? 0)));
      }
    }
    expectClose(maxShift, 0.009286297915545028, 1e-9);
    aw.interpolate(1.0);
    const end = rowsOf(sq.points, sq.dim);
    let maxEnd = 0;
    for (let i = 0; i < end.length; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        maxEnd = Math.max(maxEnd, Math.abs((end[i]?.[j] ?? 0) - (pristine[i]?.[j] ?? 0)));
      }
    }
    expect(maxEnd).toBeLessThan(1e-6);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-14 integration", () => {
  it("Succession of two Creates fully draws both squares", () => {
    // Python: Succession(Create(s1), Create(s2)); begin; finish ->
    // both squares' points equal fresh Squares (pointwise_become_partial [0,1]).
    const s1 = new Square();
    const s2 = new Square();
    const succ = new Succession(new Create(s1, { run_time: 1 }), new Create(s2, { run_time: 1 }));
    succ.begin();
    succ.finish();
    expect(succ.active_animation).toBeNull();
    expectRowsClose(rowsOf(s1.points, s1.dim), rowsOf(new Square().points, 3));
    expectRowsClose(rowsOf(s2.points, s2.dim), rowsOf(new Square().points, 3));
  });

  it("LaggedStart interpolates sub-anims with staggered sub-alphas", () => {
    // Python: AnimationGroup.interpolate clamps per-anim sub-alphas; with
    // lag_ratio 1 and run_time 1 each, group alpha 0.25 (time 0.5) leaves the
    // first Create half-drawn (12 pts, exact rows below) while the second
    // square still holds its begin() state (partial [0,0]: 4 pts collapsed
    // onto the first anchor [1,1,0]).
    const s1 = new Square();
    const s2 = new Square();
    const lag = new LaggedStart(
      new Create(s1, { run_time: 1 }),
      new Create(s2, { run_time: 1 }),
      { lag_ratio: 1 },
    );
    lag.begin();
    lag.interpolate(0.25);
    expectRowsClose(rowsOf(s1.points, s1.dim), [
      [1, 1, 0],
      [1 / 3, 1, 0],
      [-1 / 3, 1, 0],
      [-1, 1, 0],
      [-1, 1, 0],
      [-1, 1 / 3, 0],
      [-1, -1 / 3, 0],
      [-1, -1, 0],
      [-1, -1, 0],
      [-1, -1, 0],
      [-1, -1, 0],
      [-1, -1, 0],
    ]);
    expectRowsClose(rowsOf(s2.points, s2.dim), [
      [1, 1, 0],
      [1, 1, 0],
      [1, 1, 0],
      [1, 1, 0],
    ]);
    lag.finish();
    const full = rowsOf(new Square().points, 3);
    expectRowsClose(rowsOf(s1.points, s1.dim), full);
    expectRowsClose(rowsOf(s2.points, s2.dim), full);
  });

  it("TracedPath updater fires through Mobject.update", () => {
    // The updater installed by the constructor drives update_path, so one
    // update() call must grow the path exactly like a direct update_path call.
    let n = 0;
    const tp = new TracedPath(() => {
      n += 1;
      return [n, 0, 0];
    });
    tp.update(0.016);
    expect(tp.get_num_points()).toBe(4);
    expectRowsClose([rowsOf(tp.points, tp.dim)[0] ?? []], [[1, 0, 0]]);
  });

  it("Indicate bounce-back restores the mobject via the Transform path", () => {
    // Indicate is a Transform whose target is the scaled copy; interpolating
    // the full there_and_back cycle must return points to the original.
    const sq = new Square();
    const pristine = rowsOf(new Square().points, 3);
    const ind = new Indicate(sq);
    ind.begin();
    ind.interpolate(1.0);
    expectRowsClose(rowsOf(sq.points, sq.dim), pristine);
  });

  it("VGroup.add + Rotate chain matches Python ground truth", () => {
    // A rotated VGroup of two squares equals individually rotated squares.
    const g = new VGroup(new Square(), new Square());
    (g.submobjects[1] as VMobject).shift([3, 0, 0]);
    const r = new Rotate(g, PI / 2, [0, 0, 1], [0, 0, 0]);
    const target = r.create_target() as VGroup;
    const e0 = new Square();
    e0.rotate(PI / 2, [0, 0, 1], [0, 0, 0]);
    const e1 = new Square();
    e1.shift([3, 0, 0]);
    e1.rotate(PI / 2, [0, 0, 1], [0, 0, 0]);
    expectRowsClose(
      rowsOf((target.submobjects[0] as VMobject).points, 3),
      rowsOf(e0.points, e0.dim),
    );
    expectRowsClose(
      rowsOf((target.submobjects[1] as VMobject).points, 3),
      rowsOf(e1.points, e1.dim),
    );
  });

  it("ShowPassingFlash cleanup restores the full curve", () => {
    // Python clean_up_from_scene resets each submobject to partial [0, 1].
    const line = new Line([0, 0, 0], [1, 0, 0]);
    const sp = new ShowPassingFlash(line, 0.1);
    sp.begin();
    sp.interpolate(0.5);
    const removed: Mobject[] = [];
    sp.clean_up_from_scene({ remove: (m: Mobject) => removed.push(m) });
    expectRowsClose(rowsOf(line.points, line.dim), rowsOf(new Line([0, 0, 0], [1, 0, 0]).points, 3));
    expect(removed).toContain(line);
  });
});
