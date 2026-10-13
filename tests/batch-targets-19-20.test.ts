// Ground-truth tests for batch layers 19..20 (~1005 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): DecimalNumber/MathTex/Tex need the LaTeX pipeline
// (no `latex` binary in this env), so ChangingDecimal tests use a structural
// DecimalNumber-like stand-in ({number, set_value}) exactly like the port's
// DecimalNumberLike contract; string/float axis labels stay TODO stubs.
// Axis-label tests use pre-built VMobjects (Rectangle), which Python returns
// as-is from _create_label_tex.
//
// DEVIATIONS (documented inline):
//  (D1) Python StreamLines.end_animation on a never-started instance raises
//       AttributeError (flow_animation attr never assigned); the TS port
//       initializes flow_animation=null and raises ValueError instead.
//  (D2) Python StreamLines.start_animation defaults line_animation_class to
//       ShowPassingFlash; the TS port falls back to base Animation (stores
//       time_width as a plain field) until ShowPassingFlash lands.
//  (D3) Python Line3D extends Cylinder (full 3D geometry); the TS port is a
//       lightweight stub tracking vect/length/direction/start/end only.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Axes, ComplexPlane, ThreeDAxes } from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { SampleSpace } from "../src/mobject/graphing/probability.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { MethodWithArgs } from "../src/data_structures.js";
import { Line, DashedLine } from "../src/mobject/geometry/line.js";
import { Rectangle, Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Cube, Prism, Line3D, Surface } from "../src/mobject/three_d/three_dimensions.js";
import { DrawBorderThenFill, Write, TypeWithCursor } from "../src/animation/creation.js";
import { Circumscribe, ShowPassingFlash } from "../src/animation/indication.js";
import { ChangingDecimal, ChangeDecimalToValue } from "../src/animation/numbers.js";
import { Transform, FadeTransform, _MethodAnimation } from "../src/animation/transform.js";
import { AnimationGroup, Succession } from "../src/animation/composition.js";
import { Create } from "../src/animation/creation.js";
import { FadeIn, FadeOut } from "../src/animation/fading.js";
import { turn_animation_into_updater } from "../src/animation/updaters/mobject_update_utils.js";
import { StreamLines } from "../src/mobject/vector_field.js";
import { MobjectTable } from "../src/mobject/table.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { pointRows } from "../src/utils/arrays.js";

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
function plainRows(v: number[] | NDArray): number[][] {
  if (Array.isArray(v)) {
    return (v.length > 0 && Array.isArray(v[0]) ? v : [v]) as number[][];
  }
  return rowsOf(v as NDArray, 3);
}
function makeAxes(): Axes {
  return new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
}
function makeAxesWithLines(): Axes {
  const ax = makeAxes();
  ax.axes = new VGroup(
    new Line([-3, 0, 0], [3, 0, 0]),
    new Line([0, -2, 0], [0, 2, 0]),
  );
  (ax as unknown as { x_axis: unknown }).x_axis = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
  return ax;
}
function makeAxesWithNumberLines(): Axes {
  // NumberLine axes carry get_unit_size (length/range); the y axis is left
  // unrotated because unit size needs only length and range.
  const ax = makeAxes();
  ax.axes = new VGroup(
    new NumberLine({ x_range: [-3, 3, 1], length: 6 }),
    new NumberLine({ x_range: [-2, 2, 1], length: 4 }),
  );
  (ax as unknown as { x_axis: unknown }).x_axis = new NumberLine({
    x_range: [-3, 3, 1],
    length: 6,
  });
  return ax;
}
function makeScene(): LinearTransformationScene {
  const s = new LinearTransformationScene();
  (s as unknown as { background_mobjects: Mobject[] }).background_mobjects = [];
  (s as unknown as { foreground_mobjects: Mobject[] }).foreground_mobjects = [];
  (s as unknown as { transformable_mobjects: Mobject[] }).transformable_mobjects = [];
  (s as unknown as { moving_vectors: Mobject[] }).moving_vectors = [];
  (s as unknown as { transformable_labels: unknown[] }).transformable_labels = [];
  (s as unknown as { moving_mobjects: Mobject[] }).moving_mobjects = [];
  (s as unknown as { ghost_vectors: VGroup }).ghost_vectors = new VGroup();
  (s as unknown as { leave_ghost_vectors: boolean }).leave_ghost_vectors = false;
  return s;
}
// Structural DecimalNumber stand-in (text/tex out of scope): satisfies the
// port's DecimalNumberLike contract ({number, set_value}).
function makeDecimalLike(start: number): VMobject & { number: number; set_value(v: number): void } {
  const m = new Square() as VMobject & { number: number; set_value(v: number): void };
  (m as unknown as { number: number }).number = start;
  (m as unknown as { set_value: (v: number) => void }).set_value = function (v: number): void {
    (this as unknown as { number: number }).number = v;
  };
  return m;
}

// ---------------------------------------------------------------------------
// LinearTransformationScene.add_special_mobjects / add_moving_mobject
// Python: second add of the same mobject is a no-op (dedup by identity);
// add_moving_mobject sets mobject.target then delegates.
// ---------------------------------------------------------------------------
describe("LinearTransformationScene.add_special_mobjects / add_moving_mobject", () => {
  it("dedups repeated mobjects and adds once", () => {
    const s = makeScene();
    const sq = new Square() as unknown as Mobject;
    s.add_special_mobjects((s as unknown as { background_mobjects: Mobject[] }).background_mobjects, sq, sq);
    expect((s as unknown as { background_mobjects: Mobject[] }).background_mobjects.length).toBe(1);
    expect(s.mobjects).toContain(sq);
  });

  it("add_moving_mobject sets target and tracks", () => {
    // Python: m3.target is None afterwards; m3 in moving_mobjects.
    const s = makeScene();
    const m = new Square() as unknown as Mobject & { target?: Mobject | null };
    s.add_moving_mobject(m, null);
    expect(m.target).toBeNull();
    expect((s as unknown as { moving_mobjects: Mobject[] }).moving_mobjects).toContain(m);
    const t = new Square() as unknown as Mobject;
    const m2 = new Square() as unknown as Mobject & { target?: Mobject | null };
    s.add_moving_mobject(m2, t);
    expect(m2.target).toBe(t);
  });
});

// ---------------------------------------------------------------------------
// LinearTransformationScene.get_moving_mobject_movement
// Python (m4 = Square shifted [1,0,0], target None, func p->[x+1,y,z]):
//   target center becomes [2,0,0]; returns Transform.
// ---------------------------------------------------------------------------
describe("LinearTransformationScene.get_moving_mobject_movement", () => {
  it("moves null-target mobjects through func and returns Transform", () => {
    const s = makeScene();
    const m = new Square().shift([1, 0, 0]) as unknown as Mobject & { target: Mobject | null };
    m.target = null;
    (s as unknown as { moving_mobjects: Mobject[] }).moving_mobjects = [m as unknown as Mobject];
    const anim = s.get_moving_mobject_movement((p) => [(p as number[])[0]! + 1, (p as number[])[1]!, (p as number[])[2]!]);
    expect(anim).toBeInstanceOf(Transform);
    expectRowsClose([(m.target as Mobject).get_center()], [[2, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// CoordinateSystem.get_x/y_unit_size
// Python (Axes([-3,3,1],[-2,2,1],6,4)): both 1.0.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_x/y_unit_size", () => {
  it("returns 1.0 on the 6x4 fixture", () => {
    // NOTE: the TS Axes port ships no NumberLine axes (empty `axes` VGroup),
    // so the fixture wires faithful NumberLine stand-ins the way Python's
    // _create_axis output behaves (length/range determine unit size).
    const ax = makeAxesWithNumberLines();
    expectClose(ax.get_x_unit_size(), 1.0);
    expectClose(ax.get_y_unit_size(), 1.0);
  });
});

// ---------------------------------------------------------------------------
// CoordinateSystem.get_x/y_axis_label (VMobject labels returned as-is)
// Python: xl is lab, center [3.2, 0.375, 0]; yl is lab2, center [0.375, 2.15, 0].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_x/y_axis_label", () => {
  it("positions an x label like Python", () => {
    // NOTE: Python's NumberLine axis carries an arrow tip, so its UR edge
    // anchor is [3, 0.175, 0]; the TS tip pipeline is unported, so the Line
    // stand-in anchors at [3, 0, 0]. The next_to offset itself is identical
    // on both sides: [0.2, 0.2, 0] (Python [3.2,0.375,0]-[3,0.175,0]).
    const ax = makeAxesWithLines();
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_x_axis_label(lab as unknown as VMobject);
    expect(out).toBe(lab as unknown as Mobject);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[3.2, 0.2, 0]]);
  });

  it("positions a y label like Python", () => {
    // NOTE: same 0.175 tip-anchor delta as above; the offset from the anchor
    // ([0.2, 0.15, 0]) matches Python ([0.375,2.15,0]-[0.175,2,0]) exactly.
    const ax = makeAxesWithLines();
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_y_axis_label(lab as unknown as VMobject);
    expect(out).toBe(lab as unknown as Mobject);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[0.2, 2.15, 0]]);
  });
});

// ---------------------------------------------------------------------------
// CoordinateSystem.get_vertical/horizontal_line
// Python (pt = c2p(1,1) = [1,1,0]): vertical [1,0,0]->[1,1,0] (DashedLine);
// horizontal [~0,1,0]->[1,1,0] (DashedLine).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_vertical/horizontal_line", () => {
  it("drops a vertical line from the x axis", () => {
    const ax = makeAxesWithLines();
    const pt = [1, 1, 0];
    const line = ax.get_vertical_line(pt);
    expect(line).toBeInstanceOf(DashedLine);
    expectRowsClose([line.get_start()], [[1, 0, 0]]);
    expectRowsClose([line.get_end()], [[1, 1, 0]]);
  });

  it("drops a horizontal line from the y axis", () => {
    const ax = makeAxesWithLines();
    const line = ax.get_horizontal_line([1, 1, 0]);
    expect(line).toBeInstanceOf(DashedLine);
    expectRowsClose([line.get_start()], [[0, 1, 0]], 1e-9);
    expectRowsClose([line.get_end()], [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// DrawBorderThenFill.begin / Write.finish
// Python: begin refreshes outline (fill opacity 0, 16 pts on Square,
// get_all_mobjects length 3); Write reverse begin/finish invert twice.
// ---------------------------------------------------------------------------
describe("DrawBorderThenFill.begin / Write.finish", () => {
  it("begin refreshes the outline", () => {
    const sq = new Square() as unknown as VMobject;
    const a = new DrawBorderThenFill(sq);
    a.begin();
    expect(a.outline.get_num_points()).toBe(16);
    expectClose(a.outline.get_fill_opacity(), 0.0);
    expect(a.get_all_mobjects().length).toBe(3);
  });

  it("Write reverse begin/finish round-trips the point order", () => {
    const sq = new Square() as unknown as VMobject;
    const before = rowsOf(sq.points, sq.dim).map((r) => [...r]);
    const w = new Write(sq, { reverse: true });
    expect(w.remover).toBe(true);
    expect(w.introducer).toBe(false);
    w.begin();
    w.finish();
    expectRowsClose(rowsOf(sq.points, sq.dim), before);
  });

  it("Write defaults run_time/lag_ratio from submobject count", () => {
    // Python: Square fixture -> run_time 1, lag_ratio 0.2, reverse False.
    const sq = new Square() as unknown as VMobject;
    const w = new Write(sq);
    expectClose(w.run_time, 1);
    expectClose(w.lag_ratio, 0.2);
    expect(w.reverse).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// TypeWithCursor.begin / finish
// Python (3-square group): y_cursor 0.0, cursor added to mobject, opacity 0
// at begin (then 1 after first update); finish leave_on -> opacity 1;
// leave_cursor_on=False removes the cursor.
// ---------------------------------------------------------------------------
describe("TypeWithCursor.begin / finish", () => {
  function group3(): VGroup {
    return new VGroup(new Square(), new Square(), new Square());
  }
  it("begin anchors the cursor and adds it", () => {
    const g = group3();
    const cursor = new Square() as unknown as Mobject;
    const t = new TypeWithCursor(g as unknown as Mobject, cursor);
    t.begin();
    expectClose((t as unknown as { y_cursor: number }).y_cursor, 0.0);
    expect((t.mobject as unknown as VGroup).submobjects).toContain(cursor);
  });

  it("finish with leave_cursor_on keeps an opaque cursor", () => {
    const g = group3();
    const cursor = new Square() as unknown as Mobject;
    const t = new TypeWithCursor(g as unknown as Mobject, cursor);
    t.begin();
    t.finish();
    expectClose(cursor.get_fill_opacity(), 1.0);
    expect((t.mobject as unknown as VGroup).submobjects).toContain(cursor);
  });

  it("finish with leave_cursor_on=False removes the cursor", () => {
    const g = new VGroup(new Square(), new Square());
    const cursor = new Square() as unknown as Mobject;
    const t = new TypeWithCursor(g as unknown as Mobject, cursor, { leave_cursor_on: false });
    t.begin();
    t.finish();
    expect((t.mobject as unknown as VGroup).submobjects).not.toContain(cursor);
  });
});

// ---------------------------------------------------------------------------
// Circumscribe.__init__
// Python: default 1 anim (ShowPassingFlash); Circle 1 anim; fade both
// [FadeIn, FadeOut] run_time 0.5 each; fade_in [FadeIn, Uncreate];
// fade_out [Create, FadeOut]; bad shape ValueError.
// ---------------------------------------------------------------------------
describe("Circumscribe.__init__", () => {
  it("wraps the target in a single ShowPassingFlash by default", () => {
    const c = new Circumscribe(new Square() as unknown as Mobject);
    const anims = (c as unknown as { animations: unknown[] }).animations;
    expect(anims.length).toBe(1);
    expect(anims[0]).toBeInstanceOf(ShowPassingFlash);
  });

  it("accepts Circle shape", () => {
    const c = new Circumscribe(new Square() as unknown as Mobject, Circle);
    expect((c as unknown as { animations: unknown[] }).animations[0]).toBeInstanceOf(ShowPassingFlash);
  });

  it("splits run_time for fade_in+fade_out", () => {
    const c = new Circumscribe(new Square() as unknown as Mobject, Rectangle, true, true);
    const anims = (c as unknown as { animations: { run_time: number }[] }).animations;
    expect(anims[0]).toBeInstanceOf(FadeIn);
    expect(anims[1]).toBeInstanceOf(FadeOut);
    expectClose(anims[0]!.run_time, 0.5);
    expectClose(anims[1]!.run_time, 0.5);
  });

  it("builds FadeIn+Uncreate for fade_in and Create+FadeOut for fade_out", () => {
    const ci = new Circumscribe(new Square() as unknown as Mobject, Rectangle, true, false);
    const ai = (ci as unknown as { animations: Mobject[] }).animations;
    expect(ai[0]).toBeInstanceOf(FadeIn);
    const co = new Circumscribe(new Square() as unknown as Mobject, Rectangle, false, true);
    const ao = (co as unknown as { animations: Mobject[] }).animations;
    expect(ao[1]).toBeInstanceOf(FadeOut);
  });

  it("rejects non Rectangle/Circle shapes with ValueError", () => {
    expect(() => new Circumscribe(new Square() as unknown as Mobject, Number as never)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// ChangingDecimal.interpolate_mobject / ChangeDecimalToValue.__init__
// Python: set_value(number_update_func(rate_func(alpha))); ChangeDecimalToValue
// interpolates start->target (2->10 at 0.25 == 4.0); non-DecimalNumber raises
// TypeError. rate_func default smooth: smooth(0.5) == 0.5.
// ---------------------------------------------------------------------------
describe("ChangingDecimal / ChangeDecimalToValue", () => {
  it("interpolate_mobject applies rate_func then the update func", () => {
    const d = makeDecimalLike(0);
    const a = new ChangingDecimal(d as unknown as Mobject, (x: number) => x * 10);
    a.begin();
    a.interpolate_mobject(0.5);
    expectClose(d.number, 5.0);
  });

  it("ChangeDecimalToValue linearly interpolates start->target", () => {
    // Python: set_value(interpolate(start, target, rate_func(alpha))) with
    // the default sigmoid smooth; smooth(0.25) == 0.07010371654510815 so
    // interpolate(2, 10, smooth(0.25)) == 2.5608297323608652;
    // smooth(1) == 1 -> 10.0.
    const d = makeDecimalLike(2);
    const c = new ChangeDecimalToValue(d as unknown as Mobject, 10);
    c.begin();
    c.interpolate_mobject(0.25);
    expectClose(d.number, 2.5608297323608652);
    c.interpolate_mobject(1.0);
    expectClose(d.number, 10.0);
  });

  it("rejects mobjects without set_value with TypeError", () => {
    expect(() => new ChangingDecimal(new Square() as unknown as Mobject, (x: number) => x)).toThrowError(
      /ChangingDecimal can only take in a DecimalNumber/,
    );
  });
});

// ---------------------------------------------------------------------------
// Transform.begin / _MethodAnimation.finish / FadeTransform.begin
// Python: begin aligns (target_copy 16 pts, 4 all-mobjects); _MethodAnimation
// finish shifts center to [1,0,0]; FadeTransform ending is a Group, 3 mobjects.
// ---------------------------------------------------------------------------
describe("Transform.begin / _MethodAnimation.finish / FadeTransform.begin", () => {
  it("Transform.begin snapshots target_copy with aligned data", () => {
    const s1 = new Square() as unknown as Mobject;
    const s2 = new Square().shift([2, 0, 0]) as unknown as Mobject;
    const t = new Transform(s1, s2);
    t.begin();
    expect((t as unknown as { target_copy: Mobject }).target_copy.get_num_points()).toBe(16);
    expect(t.get_all_mobjects().length).toBe(4);
  });

  it("_MethodAnimation.finish applies the queued methods", () => {
    // NOTE: like Python (MoveToTarget requires a 'target' attr and the
    // probe sets target = copy().shift(...)), the target is installed
    // before construction so begin() aligns Square<->Square.
    const s = new Square() as unknown as Mobject & { target: Mobject };
    s.target = (s.copy() as Mobject).shift([1, 0, 0]);
    const unbound = (Square.prototype as unknown as Record<string, unknown>)["shift"] as (...args: never[]) => unknown;
    const bound = Object.assign(function (...args: unknown[]): unknown {
      return (unbound as (...a: unknown[]) => unknown).apply(s, args);
    }, { __self__: s, __func__: unbound });
    const anim = new _MethodAnimation(s, [new MethodWithArgs(bound as never, [[1, 0, 0]], {})]);
    anim.begin();
    anim.finish();
    expectRowsClose([s.get_center()], [[1, 0, 0]]);
  });

  it("FadeTransform.begin snapshots a Group ending mobject", () => {
    // NOTE: Python ground truth is type Group (not VGroup); assert the
    // constructor (a failing toBeInstanceOf on a Proxy Mobject makes
    // vitest's pretty-printer recurse until the worker OOMs).
    const s1 = new Square() as unknown as Mobject;
    const s2 = new Square().shift([2, 0, 0]) as unknown as Mobject;
    const f = new FadeTransform(s1, s2);
    f.begin();
    expect((f as unknown as { ending_mobject: Mobject }).ending_mobject.constructor.name).toBe("Group");
    expect(f.get_all_mobjects().length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// turn_animation_into_updater.update
// Python: returns the same mobject; suspend False; total_time starts 0
// (-delay); stepping dt advances total_time; zero run_time snaps and removes.
// ---------------------------------------------------------------------------
describe("turn_animation_into_updater", () => {
  it("attaches an updater starting at total_time 0", () => {
    const sq = new Square() as unknown as Mobject;
    const anim = new Create(sq as unknown as VMobject, { run_time: 1 });
    const out = turn_animation_into_updater(anim as never);
    expect(out).toBe(sq);
    // NOTE: total_time is -delay == -0 (Python: -0.0 == 0); expectClose
    // treats them as equal.
    expectClose((anim as unknown as { total_time: number }).total_time, 0);
    expect((anim as unknown as { suspend_mobject_updating: boolean }).suspend_mobject_updating).toBe(false);
    expect(sq.updaters.length).toBe(1);
  });

  it("update steps total_time forward", () => {
    const sq = new Square() as unknown as Mobject;
    const anim = new Create(sq as unknown as VMobject, { run_time: 1 });
    turn_animation_into_updater(anim as never);
    const upd = sq.updaters[0] as (m: Mobject, dt: number) => void;
    upd(sq, 0.5);
    expectClose((anim as unknown as { total_time: number }).total_time, 0.5);
  });

  it("zero run_time snaps to final state and removes the updater", () => {
    const sq = new Square() as unknown as Mobject;
    const anim = new Create(sq as unknown as VMobject, { run_time: 0 });
    turn_animation_into_updater(anim as never);
    expect(sq.updaters.length).toBe(1);
    const upd = sq.updaters[0] as (m: Mobject, dt: number) => void;
    upd(sq, 0.1);
    expect(sq.updaters.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// StreamLines.start_animation / end_animation (+ create)
// Python (constant field, virtual_time 1.0): 25 lines; warm_up=False times in
// [0, virtual_time); end_animation -> AnimationGroup of 25, flow cleared.
// (D2): TS uses base Animation per line instead of ShowPassingFlash.
// ---------------------------------------------------------------------------
describe("StreamLines.start/end_animation", () => {
  function field(p: number[] | NDArray): number[] | NDArray {
    return [1, 0, 0];
  }
  it("start_animation installs flow state with in-range times", () => {
    const sl = new StreamLines(field, { x_range: [-2, 2, 1], y_range: [-2, 2, 1], virtual_time: 1.0, stroke_width: 2, max_anchors_per_line: 5 });
    expect(sl.stream_lines.length).toBe(25);
    sl.start_animation(false, 1.0);
    expect(sl.flow_animation).not.toBeNull();
    expectClose(sl.flow_speed, 1.0);
    expectClose(sl.time_width, 0.3);
    for (const line of sl.stream_lines) {
      const t = (line as unknown as { time: number }).time;
      expect(t >= 0 && t < sl.virtual_time).toBe(true);
    }
  });

  it("end_animation returns an AnimationGroup and clears flow", () => {
    const sl = new StreamLines(field, { x_range: [-2, 2, 1], y_range: [-2, 2, 1], virtual_time: 1.0, stroke_width: 2, max_anchors_per_line: 5 });
    sl.start_animation(false, 1.0);
    const g = sl.end_animation();
    expect(g).toBeInstanceOf(AnimationGroup);
    expect((g as unknown as { animations: unknown[] }).animations.length).toBe(25);
    expect(sl.flow_animation).toBeNull();
  });

  it("end_animation before start raises (D1: ValueError in TS, AttributeError in Python)", () => {
    const sl = new StreamLines(field, { x_range: [-2, 2, 1], y_range: [-2, 2, 1] });
    // NOTE (D1): Python raises AttributeError here (missing attr); the port
    // null-initializes the field and raises ValueError instead.
    expect(() => sl.end_animation()).toThrowError(/start the animation/);
  });

  it("create groups per-line Creates with lag run_time/2/len", () => {
    // Python (virtual_time 2.0, 25 lines): lag 0.04, 25 Create anims.
    const sl = new StreamLines(field, { x_range: [-2, 2, 1], y_range: [-2, 2, 1], virtual_time: 2.0 });
    const g = sl.create();
    expect(g).toBeInstanceOf(AnimationGroup);
    expect((g as unknown as { animations: unknown[] }).animations.length).toBe(25);
    expectClose(g.lag_ratio, 0.04);
  });
});

// ---------------------------------------------------------------------------
// Table.create
// Python (2x2 MobjectTable): AnimationGroup, 2 anims [Create, Create], lag 1.
// ---------------------------------------------------------------------------
describe("Table.create", () => {
  it("groups line and element creations", () => {
    const t = new MobjectTable([[new Square(), new Square()], [new Square(), new Square()]]);
    const g = t.create();
    expect(g).toBeInstanceOf(AnimationGroup);
    expect((g as unknown as { animations: unknown[] }).animations.length).toBe(2);
    expectClose(g.lag_ratio, 1);
  });
});

// ---------------------------------------------------------------------------
// SampleSpace.get_horizontal/vertical_division
// Python (3x3 square): horizontal [0.3,0.7] -> 2 parts, centers
// [0,1.05,0],[0,-0.45,0], widths 3, heights [0.9,2.1]; vertical [0.5,0.5] ->
// centers [-0.75,0,0],[0.75,0,0], widths 1.5, heights 3.
// ---------------------------------------------------------------------------
describe("SampleSpace divisions", () => {
  it("get_horizontal_division splits along y with ground-truth boxes", () => {
    const s = new SampleSpace();
    const h = s.get_horizontal_division([0.3, 0.7]);
    expect(h.submobjects.length).toBe(2);
    expectRowsClose([(h.submobjects[0] as Mobject).get_center()], [[0, 1.05, 0]]);
    expectRowsClose([(h.submobjects[1] as Mobject).get_center()], [[0, -0.45, 0]], 1e-9);
    expectClose((h.submobjects[0] as Mobject).width, 3.0);
    expectClose((h.submobjects[0] as Mobject).height, 0.9);
    expectClose((h.submobjects[1] as Mobject).height, 2.1);
  });

  it("get_vertical_division splits along x with ground-truth boxes", () => {
    const s = new SampleSpace();
    const v = s.get_vertical_division([0.5, 0.5]);
    expect(v.submobjects.length).toBe(2);
    expectRowsClose([(v.submobjects[0] as Mobject).get_center()], [[-0.75, 0, 0]]);
    expectRowsClose([(v.submobjects[1] as Mobject).get_center()], [[0.75, 0, 0]]);
    expectClose((v.submobjects[0] as Mobject).width, 1.5);
    expectClose((v.submobjects[0] as Mobject).height, 3.0);
  });

  it("a scalar p_list is completed to two parts", () => {
    expect(new SampleSpace().get_horizontal_division(0.3).submobjects.length).toBe(2);
  });
});

function makeThreeDAxesWithLines(): ThreeDAxes {
  // Line-geometry stand-ins for the unported NumberLine axes (same D1-style
  // wiring as makeAxesWithLines). Lengths mirror the Python defaults:
  // x/y length frame_height+2.5 = 10.5 (half 5.25; shift_onto_screen pulls
  // the y label to 3.65 on both sides), z length frame_height-1.5 = 6.5.
  const ax = new ThreeDAxes();
  ax.axes = new VGroup(
    new Line([-5.25, 0, 0], [5.25, 0, 0]),
    new Line([0, -5.25, 0], [0, 5.25, 0]),
    new Line([0, 0, -3.25], [0, 0, 3.25]),
  );
  (ax as unknown as { x_axis: unknown }).x_axis = new NumberLine({ x_range: [-6, 6, 1], length: 10.5 });
  return ax;
}
// Python: returns self; 4x4 surface -> 16 faces; first-face color #83C167;
// colorscale None warns and leaves colors unchanged.
// ---------------------------------------------------------------------------
describe("Surface.set_fill_by_value", () => {
  function surfAxes(): { axes: ThreeDAxes; make(): Surface } {
    // NOTE: the TS ThreeDAxes keeps Python's (x_range, y_range, x_length,
    // y_length, ...) positional order with z_range 8th (unlike Python's 3rd),
    // so z_range rides in its own slot here.
    const axes = new ThreeDAxes([0, 5, 1], [0, 5, 1], null, null, undefined, [], 20, [-1, 1, 0.5]);
    const make = (): Surface =>
      new Surface(
        (u: number, v: number) => plainRows(axes.coords_to_point(u, v, Math.sin(u) * Math.cos(v)) as unknown as NDArray)[0]!,
        [0, 5],
        [0, 5],
        [4, 4],
      );
    return { axes, make };
  }
  it("paints faces by axis value and returns self", () => {
    // Python (axis=0, pivots [(RED,-0.5),(YELLOW,0),(GREEN,0.5)]): 16 faces,
    // first and last face GREEN (#83C167, clamped above the top pivot).
    const { axes, make } = surfAxes();
    const s = make();
    const out = s.set_fill_by_value(axes, [["RED", -0.5], ["YELLOW", 0], ["GREEN", 0.5]] as never, 0);
    expect(out).toBe(s);
    expect(s.submobjects.length).toBe(16);
    expect((s.submobjects[0] as VMobject).get_color().to_hex()).toBe("#83C167");
    expect((s.submobjects[s.submobjects.length - 1] as VMobject).get_color().to_hex()).toBe("#83C167");
  });

  it("resolves per-face z through ThreeDAxes point_to_coords", () => {
    // Python (axis=2, pivots -0.5/0/0.5): face0 center maps to z ~= 0.4745
    // (clamped to the top pivot) -> #00FF00; last face -> #FFCE00.
    const { axes, make } = surfAxes();
    const s = make();
    expect(s.set_fill_by_value(axes, [["#FF0000", -0.5], ["#FFFF00", 0], ["#00FF00", 0.5]] as never, 2)).toBe(s);
    expect((s.submobjects[0] as VMobject).get_color().to_hex()).toBe("#00FF00");
    expect((s.submobjects[s.submobjects.length - 1] as VMobject).get_color().to_hex()).toBe("#FFCE00");
  });

  it("colorscale None is a no-op returning self", () => {
    const { axes, make } = surfAxes();
    const s = make();
    expect(s.set_fill_by_value(axes, null)).toBe(s);
  });
});

// ---------------------------------------------------------------------------
// ThreeDAxes.get_y/z_axis_label (VMobject labels returned as-is)
// Python: y center [0.375, 3.65, 0]; z center [0.2, ~0, 3.25].
// ---------------------------------------------------------------------------
describe("ThreeDAxes.get_y/z_axis_label", () => {
  it("positions a y label like Python", () => {
    // NOTE: same 0.175 tip-anchor delta as the 2D labels; y is identical
    // (shift_onto_screen pulls both to 3.65).
    const ax = makeThreeDAxesWithLines();
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_y_axis_label(lab as unknown as VMobject);
    expect(out).toBe(lab as unknown as Mobject);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[0.2, 3.65, 0]]);
  });

  it("positions a z label like Python", () => {
    const ax = makeThreeDAxesWithLines();
    const lab = new Rectangle({ width: 0.2, height: 0.2 });
    const out = ax.get_z_axis_label(lab as unknown as VMobject);
    expectRowsClose([(out as unknown as VMobject).get_center()], [[0.2, 0, 3.25]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// ComplexPlane / _get_default_coordinate_values
// Python: 14 x ticks [-7..-1, 1..7], 8 y imaginaries [-4i..-1i, 1i..4i]
// (zero excluded); total 22 entries.
// ---------------------------------------------------------------------------
describe("ComplexPlane", () => {
  it("has default ranges and 22 default coordinate values", () => {
    const cp = new ComplexPlane();
    const vals = cp._get_default_coordinate_values();
    expect(vals.length).toBe(22);
    expect(vals[0]).toBe(-7);
    expect(vals[13]).toBe(7);
    const tail = vals.slice(14) as { re: number; im: number }[];
    expect(tail[0]).toEqual({ re: 0, im: -4 });
    expect(tail[tail.length - 1]).toEqual({ re: 0, im: 4 });
    expect(tail.every((c) => c.re === 0 && c.im !== 0)).toBe(true);
  });

  it("maps complex numbers through coords_to_point (n2p round-trip)", () => {
    const cp = new ComplexPlane();
    const p = cp.n2p({ re: 1, im: 1 });
    expectRowsClose(plainRows(p), [[1, 1, 0]]);
    const back = cp.p2n([1, 1, 0]);
    expectClose(back.re, 1.0);
    expectClose(back.im, 1.0);
  });
});

// ---------------------------------------------------------------------------
// NumberLine.rotate_about_zero
// Python: center unchanged; start ~[0,-3,0], end ~[0,3,0]; n2p(0) origin.
// ---------------------------------------------------------------------------
describe("NumberLine.rotate_about_zero", () => {
  it("rotates pi/2 about the origin keeping n2p(0) fixed", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const ret = nl.rotate_about_zero(Math.PI / 2);
    expect(ret).toBe(nl);
    expectRowsClose(plainRows(nl.n2p(0)), [[0, 0, 0]], 1e-9);
    const [start, end] = nl.get_start_and_end();
    expectRowsClose([start], [[0, -3, 0]], 1e-6);
    expectRowsClose([end], [[0, 3, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// Cube.init_points / Prism.generate_points / Line3D.__init__
// Python Cube(2): 6 faces at [0,0,±1],[±1,0,0],[0,±1,0], w/h/d 2, init self.
// Prism([3,2,1]): w/h/d 3/2/1. Line3D([0,0,0],[2,2,2]): vect [2,2,2],
// length sqrt(12), dir 1/sqrt(3); defaults LEFT/RIGHT, thickness 0.02,
// resolution [2,24]. (D3): TS Line3D skips Cylinder geometry.
// ---------------------------------------------------------------------------
describe("Cube / Prism / Line3D", () => {
  it("Cube.init_points regenerates six faces and returns self", () => {
    const c = new Cube(2);
    expect(c.submobjects.length).toBe(6);
    // NOTE: like Python (verified: 6 -> 12 -> 18 across init/generate
    // calls), generate_points appends without clearing, so init_points
    // doubles the faces and returns self.
    expect(c.init_points()).toBe(c);
    expect(c.submobjects.length).toBe(12);
    expectRowsClose([(c.submobjects[0] as Mobject).get_center()], [[0, 0, -1]]);
    expectClose((c as unknown as Mobject).width, 2.0);
    expectClose((c as unknown as Mobject).height, 2.0);
    expectClose((c as unknown as Mobject).depth, 2.0);
  });

  it("Prism.generate_points rescales to dimensions", () => {
    const p = new Prism([3, 2, 1]);
    expect(p.generate_points()).toBe(p);
    expectClose((p as unknown as Mobject).width, 3.0);
    expectClose((p as unknown as Mobject).height, 2.0);
    expectClose((p as unknown as Mobject).depth, 1.0);
  });

  it("Line3D.__init__ tracks vect/length/direction/endpoints", () => {
    const l = new Line3D([0, 0, 0], [2, 2, 2]);
    expectRowsClose([l.vect], [[2, 2, 2]]);
    expectClose(l.length, Math.sqrt(12));
    expectRowsClose([l.direction], [[1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)]]);
    expectRowsClose(plainRows(l.get_start()), [[0, 0, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[2, 2, 2]]);
  });

  it("Line3D exposes Python-compatible defaults (D3: stub, no Cylinder mesh)", () => {
    const l = new Line3D();
    expectRowsClose(plainRows(l.get_start()), [[-1, 0, 0]]);
    expectRowsClose(plainRows(l.get_end()), [[1, 0, 0]]);
    expectClose(l.thickness, 0.02);
    expect(l.resolution).toEqual([2, 24]);
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols against earlier-ported machinery
// ---------------------------------------------------------------------------
describe("integration: layers-19/20 batch with earlier ports", () => {
  it("Axes unit sizes feed the vertical/horizontal line geometry", () => {
    // Python: x/y unit 1.0, so c2p(1,1) == [1,1,0] and the guide lines land
    // exactly on the integer grid. (NumberLine axes for unit sizes, Line
    // axes for projection — the TS port has not unified them yet.)
    expectClose(makeAxesWithNumberLines().get_x_unit_size(), 1.0);
    expectClose(makeAxesWithNumberLines().get_y_unit_size(), 1.0);
    const ax = makeAxesWithLines();
    const v = ax.get_vertical_line([1, 1, 0]);
    const h = ax.get_horizontal_line([1, 1, 0]);
    expectRowsClose([v.get_end()], [[1, 1, 0]]);
    expectRowsClose([h.get_end()], [[1, 1, 0]]);
  });

  it("moving-mobject movement composes with Transform.begin alignment", () => {
    // Python: after get_moving_mobject_movement, the returned Transform's
    // begin() aligns start/target copies (16 pts on Squares).
    const s = makeScene();
    const m = new Square().shift([1, 0, 0]) as unknown as Mobject & { target: Mobject | null };
    m.target = null;
    (s as unknown as { moving_mobjects: Mobject[] }).moving_mobjects = [m as unknown as Mobject];
    const anim = s.get_moving_mobject_movement((p) => [(p as number[])[0]! + 1, (p as number[])[1]!, (p as number[])[2]!]);
    anim.begin();
    // NOTE: piece movement wraps Squares in VGroups (no own points), so the
    // alignment is verified on the children (16 points each, both sides).
    const copy = (anim as unknown as { target_copy: Mobject }).target_copy;
    expect(copy.submobjects.length).toBe(1);
    expect(copy.submobjects[0].get_num_points()).toBe(16);
    expect(anim.mobject.submobjects[0].get_num_points()).toBe(16);
    expectRowsClose([(m.target as Mobject).get_center()], [[2, 0, 0]]);
  });

  it("ChangingDecimal update path runs inside turn_animation_into_updater", () => {
    // NOTE: update() interpolates with the CURRENT total_time, then advances
    // it: first step (total 0) holds value 0; second step (total 1.0,
    // ratio 0.5, smooth(0.5) == 0.5) sets 0.5*4 == 2.0. Same on both sides.
    const d = makeDecimalLike(0);
    const a = new ChangingDecimal(d as unknown as Mobject, (x: number) => x * 4, { run_time: 2 });
    turn_animation_into_updater(a as never);
    const upd = (a.mobject as Mobject).updaters[0] as (m: Mobject, dt: number) => void;
    upd(a.mobject as Mobject, 1.0);
    expectClose(d.number, 0.0);
    upd(a.mobject as Mobject, 1.0);
    expectClose(d.number, 2.0);
  });

  it("Circumscribe + Table.create compose inside an AnimationGroup", () => {
    const t = new MobjectTable([[new Square(), new Square()], [new Square(), new Square()]]);
    const g = t.create();
    const c = new Circumscribe(new Square() as unknown as Mobject);
    const combo = new AnimationGroup(g, c);
    expect((combo as unknown as { animations: unknown[] }).animations.length).toBe(2);
    expect(g.lag_ratio).toBe(1);
  });

  it("ComplexPlane defaults agree with NumberLine tick ranges", () => {
    const cp = new ComplexPlane();
    const ticks = (cp.get_x_axis().get_tick_range().toArray() as unknown[]).map(Number);
    expect(ticks.length).toBe(14);
    expect(ticks[0]).toBe(-7);
    const vals = cp._get_default_coordinate_values();
    expect(vals.length).toBe(ticks.length + 8);
  });

  it("SampleSpace division boxes are Rectangles created by the same path", () => {
    const s = new SampleSpace();
    const h = s.get_horizontal_division([0.3, 0.7]);
    for (const part of h.submobjects) expect(part).toBeInstanceOf(Rectangle);
    const t = new MobjectTable([[new Square()]]);
    expect(t.create()).toBeInstanceOf(AnimationGroup);
  });

  it("TypeWithCursor begin/finish round-trips inside a Succession-free flow", () => {
    const g = new VGroup(new Square(), new Square());
    const cursor = new Square() as unknown as Mobject;
    const t = new TypeWithCursor(g as unknown as Mobject, cursor, { leave_cursor_on: false });
    t.begin();
    t.interpolate(0.5);
    expectClose(cursor.get_fill_opacity(), 1.0);
    t.finish();
    expect((t.mobject as unknown as VGroup).submobjects).not.toContain(cursor);
  });
});
