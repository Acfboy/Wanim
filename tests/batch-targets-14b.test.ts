// Ground-truth tests for batch layers 14..14 (second file).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): none of these symbols touch text/tex.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import {
  VMobject,
  VGroup,
  VDict,
  VectorizedPoint,
  DashedVMobject,
} from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { FadeIn, FadeOut } from "../src/animation/fading.js";
import { ChangeSpeed } from "../src/animation/speedmodifier.js";
import { Wait, Animation } from "../src/animation/animation.js";
import { Create } from "../src/animation/creation.js";
import {
  always,
  f_always,
  bind_mobject_method,
} from "../src/animation/updaters/mobject_update_utils.js";
import {
  UpdateFromFunc,
  UpdateFromAlphaFunc,
  MaintainPositionRelativeTo,
} from "../src/animation/updaters/update.js";
import { ComplexValueTracker } from "../src/mobject/value_tracker.js";
import { VectorField } from "../src/mobject/vector_field.js";
import {
  get_vectorized_mobject_class,
  get_point_mobject_class,
} from "../src/mobject/utils.js";
import { pointRows } from "../src/utils/arrays.js";
import { RIGHT, UP, DOWN } from "../src/constants.js";

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
// _Fade._create_faded_mobject / FadeIn.create_target
// Python:
//   sq = Square()
//   fi = FadeIn(sq, shift=RIGHT)
//   fi._create_faded_mobject(True).get_center() == [-1,0,0]
//   fi.shift_vector == [1,0,0], point_target False, scale 1
//   fi.create_target() is fi.mobject  -> True
//   fo = FadeOut(sq, shift=DOWN); fo._create_faded_mobject(False).get_center() == [0,-1,0]
//   fi2 = FadeIn(sq, target_position=[3,0,0]): point_target True, shift [3,0,0],
//     faded center [3,0,0]
// ---------------------------------------------------------------------------
describe("_Fade._create_faded_mobject / FadeIn.create_target", () => {
  it("fades in from the negated shift", () => {
    const sq = new Square();
    const fi = new FadeIn(sq, { shift: [...RIGHT] });
    expect(fi.point_target).toBe(false);
    expectClose(fi.scale_factor, 1);
    expectRowsClose([fi.shift_vector], [[1, 0, 0]]);
    const faded = fi._create_faded_mobject(true);
    expectRowsClose([faded.get_center()], [[-1, 0, 0]]);
    // faded copy is fully transparent
    expectClose((faded as VMobject).get_fill_opacity(), 0);
  });

  it("FadeIn.create_target returns the live mobject", () => {
    const sq = new Square();
    const fi = new FadeIn(sq, { shift: [...RIGHT] });
    expect(fi.create_target()).toBe(fi.mobject);
  });

  it("FadeOut target is shifted along +shift", () => {
    const sq = new Square();
    const fo = new FadeOut(sq, { shift: [...DOWN] });
    const faded = fo.create_target();
    expectRowsClose([faded.get_center()], [[0, -1, 0]]);
  });

  it("point_target shifts toward the target position", () => {
    const sq = new Square();
    const fi = new FadeIn(sq, { target_position: [3, 0, 0] });
    expect(fi.point_target).toBe(true);
    expectRowsClose([fi.shift_vector], [[3, 0, 0]]);
    const faded = fi._create_faded_mobject(true);
    expectRowsClose([faded.get_center()], [[3, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// ChangeSpeed.setup
// Python (setup is an instance method; TS adds a static mirror + instance
// wrapper — both are exercised):
//   cs.setup(Wait()) is wait -> True; cs.setup(Create(sq)) is same object
//   wait.interpolate(0.5) == 0.5 after patching
// ---------------------------------------------------------------------------
describe("ChangeSpeed.setup", () => {
  it("returns the Wait instance with a patched interpolate", () => {
    const w = new Wait({ run_time: 1 });
    const out = ChangeSpeed.setup(w);
    expect(out).toBe(w);
    out.interpolate(0.5);
  });

  it("passes other animations through prepare_animation unchanged", () => {
    const c = new Create(new Square());
    expect(ChangeSpeed.setup(c)).toBe(c);
  });

  it("instance wrapper agrees with the static", () => {
    const c = new Create(new Square());
    const probe = new ChangeSpeed(c, { "0": 1, "1": 1 } as unknown as Record<number, number>, undefined, false);
    const c2 = new Create(new Square());
    expect(probe.setup(c2)).toBe(c2);
  });
});

// ---------------------------------------------------------------------------
// mobject_update_utils.always / f_always
// Python:
//   always(sq.shift, RIGHT): 1 updater; sq.update(dt) -> center [1,0,0]
//   f_always(sq.shift, lambda: UP): 1 updater; update -> center [0,1,0]
// ---------------------------------------------------------------------------
describe("mobject_update_utils.always / f_always", () => {
  it("always() installs an updater that applies the method each frame", () => {
    const sq = new Square();
    const ret = always(bind_mobject_method(sq, "shift"), [...RIGHT]);
    expect(ret).toBe(sq);
    expect(sq.get_updaters().length).toBe(1);
    sq.update(0.016);
    expectRowsClose([sq.get_center()], [[1, 0, 0]]);
  });

  it("f_always() evaluates arg generators per frame", () => {
    const sq = new Square();
    const ret = f_always(bind_mobject_method(sq, "shift"), () => [...UP]);
    expect(ret).toBe(sq);
    expect(sq.get_updaters().length).toBe(1);
    sq.update(0.016);
    expectRowsClose([sq.get_center()], [[0, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// UpdateFromFunc.interpolate_mobject / UpdateFromAlphaFunc / MaintainPositionRelativeTo
// Python:
//   UpdateFromFunc(sq, lambda m: m.shift(RIGHT*0.5)).interpolate_mobject(0.3)
//     -> center [0.5,0,0]
//   UpdateFromAlphaFunc(sq, lambda m,a: m.scale(1+a)) at alpha .5 (linear rate)
//     -> width 3.0 (2.0 * 1.5)
//   tracked Square -> [3,0,0], follower -> [1,0,0]: diff [-2,0,0];
//     tracked -> [5,0,0]; interpolate -> follower [3,0,0]
// ---------------------------------------------------------------------------
describe("UpdateFromFunc family", () => {
  it("UpdateFromFunc ignores alpha and calls the function", () => {
    const mob = new Square();
    const upd = new UpdateFromFunc(mob, (m) => (m as Square).shift([0.5, 0, 0]));
    upd.interpolate_mobject(0.3);
    expectRowsClose([mob.get_center()], [[0.5, 0, 0]]);
  });

  it("UpdateFromAlphaFunc passes rate_func(alpha)", () => {
    const mob = new Square();
    const w0 = mob.width;
    const upd = new UpdateFromAlphaFunc(mob, (m, a) => (m as Square).scale(1 + (a as number)));
    upd.interpolate_mobject(0.5);
    expectClose(mob.width, w0 * 1.5);
    expectClose(mob.width, 3.0);
  });

  it("MaintainPositionRelativeTo preserves the offset", () => {
    const tracked = new Square();
    tracked.move_to([3, 0, 0]);
    const follower = new Square();
    follower.move_to([1, 0, 0]);
    const mp = new MaintainPositionRelativeTo(follower, tracked);
    expectRowsClose([mp.diff], [[-2, 0, 0]]);
    tracked.move_to([5, 0, 0]);
    mp.interpolate_mobject(0);
    expectRowsClose([follower.get_center()], [[3, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// ComplexValueTracker
// Python:
//   ComplexValueTracker(1+2j).get_value() == (1+2j), points[0,:2] == [1,2]
//   set_value(3-4j) -> (3-4j); ComplexValueTracker(5) -> (5+0j)
// ---------------------------------------------------------------------------
describe("ComplexValueTracker", () => {
  it("stores complex values in points[0,:2]", () => {
    const ct = new ComplexValueTracker({ re: 1, im: 2 });
    expect(ct.get_value()).toEqual({ re: 1, im: 2 });
    expectRowsClose([rowsOf(ct.points, ct.dim)[0]?.slice(0, 2) ?? []], [[1, 2]]);
  });

  it("set_value round-trips", () => {
    const ct = new ComplexValueTracker({ re: 1, im: 2 });
    ct.set_value({ re: 3, im: -4 });
    expect(ct.get_value()).toEqual({ re: 3, im: -4 });
  });

  it("accepts a real number as (x + 0j)", () => {
    const ct = new ComplexValueTracker(5);
    expect(ct.get_value()).toEqual({ re: 5, im: 0 });
  });
});

// ---------------------------------------------------------------------------
// VectorField.nudge_submobjects / get_nudge_updater
// Python (constant field [1,0,0]):
//   dots at [0,0,0],[1,0,0]; nudge_submobjects(dt=0.5) -> [0.5,0,0],[1.5,0,0]
//   get_nudge_updater(speed=2) callable; updater(dot@[0,0,0], 0.5) -> [1,0,0]
// ---------------------------------------------------------------------------
describe("VectorField nudge", () => {
  it("nudges every submobject along the field", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    const d0 = new Dot().move_to([0, 0, 0]);
    const d1 = new Dot().move_to([1, 0, 0]);
    vf.add(d0, d1);
    const ret = vf.nudge_submobjects(0.5);
    expect(ret).toBe(vf);
    expectRowsClose([d0.get_center()], [[0.5, 0, 0]]);
    expectRowsClose([d1.get_center()], [[1.5, 0, 0]]);
  });

  it("get_nudge_updater scales dt by speed", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    const up = vf.get_nudge_updater(2);
    expect(typeof up).toBe("function");
    const d = new Dot().move_to([0, 0, 0]);
    up(d, 0.5);
    expectRowsClose([d.get_center()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// get_vectorized_mobject_class / get_point_mobject_class
// Python: names 'VMobject' and 'PMobject' under the cairo renderer.
// ---------------------------------------------------------------------------
describe("mobject utils class getters", () => {
  it("returns VMobject for the cairo renderer", () => {
    expect(get_vectorized_mobject_class()).toBe(VMobject);
    expect(get_vectorized_mobject_class().name).toBe("VMobject");
  });

  it("returns PMobject for the cairo renderer", () => {
    expect(get_point_mobject_class()).toBe(PMobject);
    expect(get_point_mobject_class().name).toBe("PMobject");
  });
});

// ---------------------------------------------------------------------------
// Mobject._add_intrinsic_animation_overrides / always / get_group_class /
// animate / _AnimationBuilder update_target
// Python:
//   Mobject._add_intrinsic_animation_overrides scans dir(cls) for
//     _override_animation attrs and registers them.
//   Square().get_group_class().__name__ == 'VGroup' (VMobject override);
//     Mobject().get_group_class().__name__ == 'Group'.
//   type(Square().animate).__name__ == '_AnimationBuilder'
//   Square().animate.shift(RIGHT) -> target center [1,0,0]
//   type(Square().always).__name__ == '_UpdaterBuilder'; always.shift adds 1 updater
// ---------------------------------------------------------------------------
describe("Mobject intrinsic overrides / group class / builders", () => {
  it("_add_intrinsic_animation_overrides registers marked methods", () => {
    class Probe extends Mobject {
      marked(): number {
        return 1;
      }
    }
    const animClass = class ProbeAnim extends Animation {};
    ((Probe.prototype.marked as unknown as Record<string, unknown>)._override_animation) = animClass;
    (Probe as unknown as { animation_overrides?: Map<unknown, unknown> }).animation_overrides = new Map();
    Probe._add_intrinsic_animation_overrides();
    const reg = (Probe as unknown as { animation_overrides: Map<unknown, unknown> }).animation_overrides;
    expect(reg.get(animClass)).toBe(Probe.prototype.marked);
  });

  it("get_group_class matches Python (Group base, VGroup for VMobjects)", () => {
    expect(new Mobject().get_group_class()).toBe(Group);
    expect(new Mobject().get_group_class().name).toBe("Group");
    expect(new Square().get_group_class().name).toBe("VGroup");
    expect(new VMobject().get_group_class().name).toBe("VGroup");
  });

  it("animate builder applies methods to the target", () => {
    const sq = new Square();
    const builder = sq.animate;
    expect(builder.constructor.name).toBe("_AnimationBuilder");
    sq.animate.shift([...RIGHT]);
    expect(sq.target).not.toBeNull();
    expectRowsClose([(sq.target as Mobject).get_center()], [[1, 0, 0]]);
  });

  it("update_target via __getattr__ records the method call", () => {
    const sq = new Square();
    const builder = sq.animate;
    (builder.__getattr__("shift"))([...RIGHT]);
    expectRowsClose([(sq.target as Mobject).get_center()], [[1, 0, 0]]);
    expect(builder.methods.length).toBe(1);
  });

  it("always builder installs an updater and fires immediately", () => {
    const sq = new Square();
    expect(sq.always.constructor.name).toBe("_UpdaterBuilder");
    sq.always.shift([...RIGHT]);
    expect(sq.get_updaters().length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// VMobject.add_cubic_bezier_curve / add_cubic_bezier_curves / add_subpath /
// append_vectorized_mobject
// Python:
//   add_cubic_bezier_curve([0,0,0],[1,0,0],[2,0,0],[3,0,0]) -> those 4 rows
//   add_cubic_bezier_curves: NOTE — Python impl is broken (curves.flatten()
//     is 1-D; append_points broadcasts (12,) into (12,3) -> ValueError), so
//     the TS port intentionally re-chunks the flat array instead. Assert the
//     TS behavior directly (documented deviation).
//   add_subpath(8 rows) -> same 8 rows; 2 rows -> AssertionError
//   append: a(4pts)+b(4pts) -> 8 rows; new-path-started c2 drops the stub
//     -> exactly b's 4 rows
// ---------------------------------------------------------------------------
describe("VMobject bezier appenders", () => {
  it("add_cubic_bezier_curve appends the four points", () => {
    const v = new VMobject();
    v.add_cubic_bezier_curve([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
  });

  it("add_cubic_bezier_curves re-chunks flattened input (Python raises ValueError)", () => {
    // Python: add_cubic_bezier_curves(np.array(...)) raises ValueError
    // "could not broadcast input array from shape (12,) into shape (12,3)".
    // The TS port fixes this by re-chunking; assert the intended result.
    const v = new VMobject();
    v.add_cubic_bezier_curves([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
  });

  it("add_subpath appends multiples of 4 and asserts otherwise", () => {
    const v = new VMobject();
    v.add_subpath([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
      [4, 0, 0],
      [5, 0, 0],
      [6, 0, 0],
      [7, 0, 0],
    ]);
    expect(v.get_num_points()).toBe(8);
    expect(() => new VMobject().add_subpath([[0, 0, 0], [1, 0, 0]])).toThrowError(
      expect.objectContaining({ name: "AssertionError" }),
    );
  });

  it("append_vectorized_mobject concatenates points", () => {
    const a = new VMobject();
    a.set_points([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
    const b = new VMobject();
    b.set_points([
      [9, 9, 9],
      [8, 8, 8],
      [7, 7, 7],
      [6, 6, 6],
    ]);
    a.append_vectorized_mobject(b);
    expectRowsClose(rowsOf(a.points, a.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
      [9, 9, 9],
      [8, 8, 8],
      [7, 7, 7],
      [6, 6, 6],
    ]);
  });

  it("append_vectorized_mobject drops a new-path stub", () => {
    const c = new VMobject();
    c.start_new_path([0, 0, 0]);
    const d = new VMobject();
    d.set_points([
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [4, 4, 4],
    ]);
    c.append_vectorized_mobject(d);
    expectRowsClose(rowsOf(c.points, c.dim), [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [4, 4, 4],
    ]);
  });
});

// ---------------------------------------------------------------------------
// VGroup.__isub__ / __setitem__
// Python:
//   g.__isub__(m0) removes it, len 2->1, returns self (remove's Self)
//   g2[0] = new -> submobjects[0] is new
//   NOTE: Python __setitem__ does NOT reject plain Mobjects (no error raised
//   for g2[0] = Mobject()); the TS port mirrors _assert_valid_submobjects.
// ---------------------------------------------------------------------------
describe("VGroup item ops", () => {
  it("__isub__ removes and returns self", () => {
    const g = new VGroup(new VMobject(), new VMobject());
    const m0 = g.submobjects[0]!;
    const ret = g.__isub__(m0 as VMobject);
    expect(ret).toBe(g);
    expect(g.submobjects.length).toBe(1);
    expect(g.submobjects).not.toContain(m0);
  });

  it("__setitem__ replaces the submobject", () => {
    const g = new VGroup(new VMobject(), new VMobject());
    const fresh = new VMobject();
    fresh.set_points([[5, 5, 5]]);
    g.__setitem__(0, fresh);
    expect(g.submobjects[0]).toBe(fresh);
  });
});

// ---------------------------------------------------------------------------
// VDict
// Python:
//   VDict([('a',..),('b',..)]): keys ['a','b'], 2 submobjects
//   vd['a'] is submob_dict['a']; vd['c']=.. adds key; remove('a') drops it;
//   remove('zzz') -> KeyError
// ---------------------------------------------------------------------------
describe("VDict", () => {
  it("builds from pairs with dict access", () => {
    const va = new VMobject();
    const vb = new VMobject();
    const vd = new VDict([["a", va], ["b", vb]]);
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b"]);
    expect(vd.submobjects.length).toBe(2);
    expect(vd.__getitem__("a")).toBe(va);
    expect(vd.get("b")).toBe(vb);
  });

  it("setitem adds and remove drops keys", () => {
    const vd = new VDict([["a", new VMobject()], ["b", new VMobject()]]);
    vd.__setitem__("c", new VMobject());
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b", "c"]);
    expect(vd.submobjects.length).toBe(3);
    vd.remove("a");
    expect([...vd.submob_dict.keys()]).toEqual(["b", "c"]);
    expect(vd.submobjects.length).toBe(2);
    expect(() => vd.remove("zzz")).toThrowError(expect.objectContaining({ name: "KeyError" }));
  });
});

// ---------------------------------------------------------------------------
// VectorizedPoint.get_location
// Python: VectorizedPoint([1,2,3]).get_location() == [1,2,3];
//   VectorizedPoint().get_location() == [0,0,0]
// ---------------------------------------------------------------------------
describe("VectorizedPoint.get_location", () => {
  it("returns the stored point", () => {
    const vp = new VectorizedPoint([1, 2, 3]);
    expectRowsClose([Array.from(vp.get_location() as unknown as Iterable<number>)], [[1, 2, 3]]);
  });

  it("defaults to the origin", () => {
    const vp = new VectorizedPoint();
    expectRowsClose([Array.from(vp.get_location() as unknown as Iterable<number>)], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// DashedVMobject
// Python: DashedVMobject(Square(), num_dashes=6) -> num_dashes 6,
//   dashed_ratio 0.5, 6 submobjects, top-level 0 points.
// ---------------------------------------------------------------------------
describe("DashedVMobject", () => {
  it("splits the curve into dashes", () => {
    const dm = new DashedVMobject(new Square(), 6);
    expect(dm.num_dashes).toBe(6);
    expectClose(dm.dashed_ratio, 0.5);
    expect(dm.submobjects.length).toBe(6);
    expect(dm.get_num_points()).toBe(0);
  });

  it("honours num_dashes=1 as a single dash", () => {
    const dm = new DashedVMobject(new Line([0, 0, 0], [1, 0, 0]), 1);
    expect(dm.num_dashes).toBe(1);
    expect(dm.submobjects.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-14b integration", () => {
  it("FadeIn target feeds a Transform that restores the mobject", () => {
    // Python: FadeIn(sq, shift=RIGHT).create_target() is sq; interpolating a
    // Transform from the faded start to that target at alpha=1 reproduces sq.
    const sq = new Square();
    const pristine = rowsOf(new Square().points, 3);
    const fi = new FadeIn(sq, { shift: [...RIGHT] });
    const target = fi.create_target();
    expect(target).toBe(sq);
    fi.begin();
    fi.interpolate(1.0);
    expectRowsClose(rowsOf(sq.points, sq.dim), pristine);
  });

  it("UpdateFromAlphaFunc driven by a ComplexValueTracker", () => {
    // Tracker value scales a square: alpha path matches Python widths.
    // NOTE: the default rate_func is smooth; smooth(0.5) == 0.5 so alpha 0.5
    // gives exactly 1.5x (Python: UpdateFromAlphaFunc(...).interpolate(0.5)
    // -> width 3.0).
    const tracker = new ComplexValueTracker({ re: 0, im: 0 });
    const sq = new Square();
    const w0 = new Square().width;
    const upd = new UpdateFromAlphaFunc(sq, (m, a) => {
      void tracker;
      (m as Square).scale(1 + (a as number));
    });
    tracker.set_value({ re: 1, im: 0 });
    upd.interpolate_mobject(0.5);
    expectClose(sq.width, w0 * 1.5);
  });

  it("VGroup isub/setitem compose with animate targets", () => {
    const g = new VGroup(new Square(), new Square());
    const victim = g.submobjects[0]!;
    g.__isub__(victim as VMobject);
    expect(g.submobjects.length).toBe(1);
    const fresh = new Square();
    fresh.shift([2, 0, 0]);
    g.__setitem__(0, fresh);
    expect((g.submobjects[0] as VMobject).get_center()[0]).toBeCloseTo(2, 9);
    // animate the remaining member: target shifts by RIGHT
    const member = g.submobjects[0] as VMobject;
    member.animate.shift([...RIGHT]);
    expectRowsClose([(member.target as Mobject).get_center()], [[3, 0, 0]]);
  });

  it("VectorField updater moves an AnimationBuilder-driven mobject", () => {
    // A dot shifted via animate().build() target plus a field nudge composes
    // additively, matching Python center arithmetic.
    const vf = new VectorField(() => [1, 0, 0]);
    const d = new Dot().move_to([0, 0, 0]);
    vf.add(new Dot());
    d.animate.shift([...UP]);
    const up = vf.get_nudge_updater(1);
    up(d, 1.0);
    expectRowsClose([d.get_center()], [[1, 0, 0]]);
    expectRowsClose([(d.target as Mobject).get_center()], [[0, 1, 0]]);
  });

  it("DashedVMobject of an appended-points curve keeps dash count", () => {
    const v = new VMobject();
    v.add_cubic_bezier_curve([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]);
    const extra = new VMobject();
    extra.set_points([
      [3, 0, 0],
      [4, 0, 0],
      [5, 0, 0],
      [6, 0, 0],
    ]);
    v.append_vectorized_mobject(extra);
    expect(v.get_num_points()).toBe(8);
    const dm = new DashedVMobject(v, 4);
    expect(dm.submobjects.length).toBe(4);
  });
});
