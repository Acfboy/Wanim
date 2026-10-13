// Ground-truth tests for batch layers 12..13.
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): ChangingDecimal normally takes a DecimalNumber
// (manim.mobject.text.numbers, a TeX mobject) which cannot even be constructed
// here (no `latex` binary in this environment), so it is tested with a
// structural stub exposing set_value/number, mirroring the port's structural
// check. Nothing else in this batch touches text/tex.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Rectangle } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import {
  Animation,
  Add,
  override_animation,
} from "../src/animation/animation.js";
import {
  Create,
  ShowIncreasingSubsets,
  SpiralIn,
  DrawBorderThenFill,
} from "../src/animation/creation.js";
import { AnimationGroup, LaggedStartMap } from "../src/animation/composition.js";
import { Transform, MoveToTarget } from "../src/animation/transform.js";
import { Rotating } from "../src/animation/rotation.js";
import { Homotopy, PhaseFlow, MoveAlongPath } from "../src/animation/movement.js";
import { Wiggle } from "../src/animation/indication.js";
import { UpdateFromFunc } from "../src/animation/updaters/update.js";
import { ChangingDecimal } from "../src/animation/numbers.js";
import {
  TransformMatchingAbstractBase,
  TransformMatchingShapes,
} from "../src/animation/transform_matching_parts.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { ZoomedScene } from "../src/scene/zoomed_scene.js";
import { config } from "../src/_config/utils.js";
import { TAU } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { linear } from "../src/utils/rate_functions.js";

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
// manim.mobject.mobject.Mobject.replace
// Python:
//   s = Square(side_length=2.0); r = Rectangle(width=4.0, height=1.0); r.move_to([3,1,0])
//   s.replace(r) -> width 4.0 height 4.0 center [3,1,0]  (uniform rescale to width)
//   s2.replace(r, stretch=True) -> width 4.0 height 1.0 center [3,1,0]
//   Mobject().replace(Mobject()) -> Warning("Attempting to replace mobject with no points")
// ---------------------------------------------------------------------------
describe("Mobject.replace", () => {
  it("rescales uniformly to the target width and moves to its center", () => {
    const s = new Square(2.0);
    const r = new Rectangle({ width: 4.0, height: 1.0 });
    r.move_to([3, 1, 0]);
    s.replace(r);
    expectClose(s.width, 4.0);
    expectClose(s.height, 4.0);
    expectRowsClose([s.get_center()], [[3, 1, 0]]);
  });

  it("stretches to both dims when stretch=True", () => {
    const s = new Square(2.0);
    const r = new Rectangle({ width: 4.0, height: 1.0 });
    r.move_to([3, 1, 0]);
    s.replace(r, 0, true);
    expectClose(s.width, 4.0);
    expectClose(s.height, 1.0);
    expectRowsClose([s.get_center()], [[3, 1, 0]]);
  });

  it("raises Warning for a target with no points", () => {
    expect(() => new Mobject().replace(new Mobject())).toThrowError(
      "Attempting to replace mobject with no points",
    );
    try {
      new Mobject().replace(new Mobject());
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("Warning");
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.append_points
// Python:
//   v.points = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
//   v.append_points([[2,2,0],[3,3,0]]) -> 6 points, concatenated exactly
// ---------------------------------------------------------------------------
describe("VMobject.append_points", () => {
  it("concatenates new points and returns self", () => {
    const v = new VMobject();
    v.points = pointRows(
      [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
      ],
      3,
    ) as NDArray;
    const ret = v.append_points([
      [2, 2, 0],
      [3, 3, 0],
    ]);
    expect(ret).toBe(v);
    expect(v.get_num_points()).toBe(6);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
      [2, 2, 0],
      [3, 3, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line._pointify
// Python:
//   Line._pointify(ln, [1,2,3]) -> [1,2,3]
//   Line._pointify(ln, Square()) -> [0,0,0]
//   Line._pointify(ln, Square(), [1,0,0]) -> [1,1,0]  (manim boundary quirk)
// ---------------------------------------------------------------------------
describe("Line._pointify", () => {
  it("passes points through and resolves mobjects", () => {
    const ln = new Line([0, 0, 0], [1, 0, 0]);
    expectRowsClose(rowsOf(ln._pointify([1, 2, 3] as unknown as never), 3), [[1, 2, 3]]);
    expectRowsClose(rowsOf(ln._pointify(new Square(2.0)), 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(ln._pointify(new Square(2.0), [1, 0, 0]), 3), [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.rotation.Rotating / __init__
// Python:
//   Rotating(s): angle TAU, axis [0,0,1], about_point None, about_edge None,
//     run_time 5, rate_func linear
//   Rotating(s, angle=1.5, about_point=[1,0,0], run_time=2); begin();
//     interpolate_mobject(0.5) -> center [0.2683111311261793,-0.6816387600233341,0],
//     width 2.82665525779431
// ---------------------------------------------------------------------------
describe("Rotating", () => {
  it("stores constructor defaults like Python", () => {
    const r = new Rotating(new Square());
    expectClose(r.angle, TAU);
    expectRowsClose([[...(r.axis as number[])]], [[0, 0, 1]]);
    expect(r.about_point).toBeNull();
    expect(r.about_edge).toBeNull();
    expect(r.run_time).toBe(5);
    expect(r.rate_func).toBe(linear);
  });

  it("rotates about the given point on interpolate_mobject", () => {
    const s = new Square(2.0);
    const r = new Rotating(s, 1.5, undefined, [1, 0, 0], undefined, 2);
    expect(r.run_time).toBe(2);
    r.begin();
    r.interpolate_mobject(0.5);
    expectRowsClose(
      [s.get_center()],
      [[0.2683111311261793, -0.6816387600233341, 0]],
      1e-7,
    );
    expectClose(s.width, 2.82665525779431, 1e-7);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.__new__ / override_animation
// Python:
//   class MySq(Square):
//       @override_animation(FadeIn)
//       def _fade_override(self, **kw): return Create(self, **kw)
//   type(FadeIn(MySq())).__name__ == 'Create'
//   type(FadeIn(MySq(), use_override=False)).__name__ == 'FadeIn'
//   type(Create(MySq())).__name__ == 'Create'
// ---------------------------------------------------------------------------
describe("Animation.__new__ / override_animation", () => {
  class DummyFade extends Animation {}

  class MySq extends Square {}

  function makeOverridden(): Square {
    // NOTE: like Python (`MySq.add_animation_override` is invoked on the
    // class), registration is class-level via the static method.
    const func = (mob: Mobject) => new Create(mob as VMobject);
    const decorated = override_animation(
      DummyFade as unknown as new (...a: never[]) => Animation,
    )(func as never) as unknown as (mob: Mobject) => Animation;
    MySq.add_animation_override(DummyFade, decorated as never);
    return new MySq();
  }

  it("returns the override animation when one is registered", () => {
    const m = makeOverridden();
    const viaNew = Animation.__new__.call(DummyFade, m);
    expect(viaNew).toBeInstanceOf(Create);
    const fresh = new DummyFade(m, { use_override: false });
    expect(fresh).toBeInstanceOf(DummyFade);
  });

  it("override_animation marks the function", () => {
    const cls = class extends Animation {};
    const fn = (..._a: unknown[]): unknown => null;
    const out = override_animation(cls as never)(fn as never) as unknown as {
      _override_animation?: unknown;
    };
    expect(out).toBe(fn);
    expect(out._override_animation).toBe(cls);
  });

  it("returns null when no override applies", () => {
    expect(Animation.__new__(new Square())).toBeNull();
    expect(Animation.__new__(null)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.__init_subclass__ / set_default
// Python:
//   Rotating(m).run_time == 5; Rotating.set_default(run_time=2) -> 2;
//   Rotating.set_default() -> 5; hasattr(Rotating, '_original__init__') True
// ---------------------------------------------------------------------------
describe("Animation.__init_subclass__ / set_default", () => {
  it("installs and restores default kwargs", () => {
    class Probe extends Animation {}
    Probe.__init_subclass__();
    expect((Probe as unknown as { _original__init__?: unknown })._original__init__).toBeDefined();
    const m = new Square();
    expect(new Probe(m).run_time).toBe(1);
    Probe.set_default({ run_time: 2 });
    expect(new Probe(m).run_time).toBe(2);
    Probe.set_default();
    expect(new Probe(m).run_time).toBe(1);
  });

  it("set_default on Rotating mirrors Python (5 -> 2 -> 5)", () => {
    const m = new Square();
    expect(new Rotating(m).run_time).toBe(5);
    Rotating.set_default({ run_time: 2 });
    try {
      expect(new Rotating(m).run_time).toBe(2);
    } finally {
      Rotating.set_default();
    }
    expect(new Rotating(m).run_time).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Add / interpolate_submobject
// Python:
//   Add(a1, a2): Group of 2, run_time 0.0, introducer True
//   Add(a1).mobject is a1; Add(): empty Group
//   Animation(a1).interpolate_submobject(a1, a1, 0.5) is None
// ---------------------------------------------------------------------------
describe("Add / Animation.interpolate_submobject", () => {
  it("wraps mobjects like Python", () => {
    const a1 = new Square();
    const a2 = new Square();
    const ad = new Add(a1, a2);
    expect(ad.mobject).toBeInstanceOf(Group);
    expect(ad.mobject.submobjects.length).toBe(2);
    expect(ad.run_time).toBe(0);
    expect(ad.is_introducer()).toBe(true);
    const single = new Add(a1);
    expect(single.mobject).toBe(a1);
    const empty = new Add();
    expect(empty.mobject).toBeInstanceOf(Group);
    expect(empty.mobject.submobjects.length).toBe(0);
  });

  it("base interpolate_submobject is a no-op returning undefined", () => {
    const a1 = new Square();
    expect(new Animation(a1).interpolate_submobject(a1, a1, 0.5)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.AnimationGroup.get_all_mobjects / clean_up_from_scene
// Python (with non-introducer Transforms so the group is non-empty):
//   g = AnimationGroup(t1, t2); len(get_all_mobjects()) == 2, members identical
//   to list(group); remover propagates and scene.remove called per animation.
// ---------------------------------------------------------------------------
describe("AnimationGroup.get_all_mobjects / clean_up_from_scene", () => {
  it("returns the group members", () => {
    const s1 = new Square();
    const s2 = new Square();
    const t1 = new Transform(s1, s2.copy());
    const t2 = new Transform(s2, s1.copy());
    const g = new AnimationGroup(t1, t2);
    const mobs = g.get_all_mobjects();
    expect(mobs.length).toBe(2);
    expect(mobs[0]).toBe([...g.group][0]);
    expect(mobs[1]).toBe([...g.group][1]);
  });

  it("propagates remover and cleans up each animation", () => {
    const s1 = new Square();
    const s2 = new Square();
    const t1 = new Transform(s1, s2.copy());
    const t2 = new Transform(s2, s1.copy());
    const g = new AnimationGroup(t1, t2, { remover: true });
    const removed: unknown[] = [];
    const scene = { remove: (m: unknown) => removed.push(m) };
    g.begin();
    g.finish();
    g.clean_up_from_scene(scene);
    expect(t1.remover).toBe(true);
    expect(t2.remover).toBe(true);
    expect(removed.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.LaggedStartMap.__init__ (identity default)
// Python:
//   LaggedStartMap(Create, VGroup(3 squares)) -> 3 Create anims, run_time 2,
//   lag_ratio 0.05
// ---------------------------------------------------------------------------
describe("LaggedStartMap identity", () => {
  it("maps the identity creator over submobjects", () => {
    const vg = new VGroup(new Square(), new Square(), new Square());
    const lm = new LaggedStartMap(Create as never, vg);
    expect(lm.animations.length).toBe(3);
    for (const a of lm.animations) expect(a).toBeInstanceOf(Create);
    expect(lm.run_time).toBe(2);
    expectClose(lm.lag_ratio, 0.05);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform.create_target / interpolate_submobject
// manim.animation.transform.MoveToTarget.check_validity_of_input
// Python:
//   t.create_target() is t.target_mobject
//   Transform(s1, shifted).begin(); interpolate(0.5) -> center [0.5,0,0]
//   MoveToTarget(m with target) ok; MoveToTarget(mobject-without-target-attr)
//     -> ValueError("MoveToTarget called on mobjectwithout attribute 'target'")
// ---------------------------------------------------------------------------
describe("Transform.create_target / interpolate_submobject / MoveToTarget", () => {
  it("create_target returns the target mobject", () => {
    const s1 = new Square();
    const target = s1.copy().shift([1, 0, 0]);
    const t = new Transform(s1, target);
    expect(t.create_target()).toBe(t.target_mobject);
  });

  it("interpolates halfway to the target", () => {
    const s1 = new Square(2.0);
    const t = new Transform(s1, s1.copy().shift([1, 0, 0]));
    t.begin();
    t.interpolate(0.5);
    expectRowsClose([t.mobject.get_center()], [[0.5, 0, 0]], 1e-7);
  });

  it("check_validity_of_input rejects mobjects without target", () => {
    const m = new Square();
    (m as unknown as { target: Mobject }).target = m.copy().shift([2, 0, 0]);
    expect(() => new MoveToTarget(m)).not.toThrow();
    const bare = new MoveToTarget(new Square());
    expect(bare).toBeInstanceOf(MoveToTarget);
    const noTarget = { notATarget: true } as unknown as Mobject;
    expect(() => (bare as unknown as { check_validity_of_input: (x: unknown) => void }).check_validity_of_input(noTarget)).toThrowError(
      "MoveToTarget called on mobjectwithout attribute 'target'",
    );
    try {
      (bare as unknown as { check_validity_of_input: (x: unknown) => void }).check_validity_of_input(noTarget);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.ShowPartial.interpolate_submobject (via Create)
// Python:
//   Create(Square(2)); begin(); interpolate_submobject(sub, start, 0.5)
//   -> sub 12 points vs starting 16
// ---------------------------------------------------------------------------
describe("ShowPartial.interpolate_submobject", () => {
  it("reveals a partial of the starting submobject", () => {
    const s = new Square(2.0);
    const c = new Create(s);
    c.begin();
    const sub = c.mobject.family_members_with_points()[0]!;
    const start = c.starting_mobject.family_members_with_points()[0]!;
    expect(start.get_num_points()).toBe(16);
    c.interpolate_submobject(sub, start, 0.5);
    expect(sub.get_num_points()).toBe(12);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.DrawBorderThenFill.interpolate_submobject
// Python:
//   d = DrawBorderThenFill(Square(2)); begin()
//   outline is Square; families zipped len 1 widths [3]; all_mobjects len 3
//   interpolate_submobject(f[0], f[1], f[2], 0.25) -> fill 0, 12 points
//   ... (0.75) -> fill 0, 16 points   (Square default fill opacity is 0)
// ---------------------------------------------------------------------------
describe("DrawBorderThenFill.interpolate_submobject", () => {
  it("threads outline through families and interpolates in two phases", () => {
    const d = new DrawBorderThenFill(new Square(2.0));
    d.begin();
    expect(d.outline).toBeInstanceOf(Square);
    const fams = d.get_all_families_zipped();
    expect(fams.length).toBe(1);
    expect(fams[0]!.length).toBe(3);
    expect(d.get_all_mobjects().length).toBe(3);
    const mobs = fams[0]!;
    d.interpolate_submobject(mobs[0]!, mobs[1]!, mobs[2]!, 0.25);
    expectClose(mobs[0]!.get_fill_opacity(), 0.0);
    expect(mobs[0]!.get_num_points()).toBe(12);

    const d2 = new DrawBorderThenFill(new Square(2.0));
    d2.begin();
    const mobs2 = d2.get_all_families_zipped()[0]!;
    d2.interpolate_submobject(mobs2[0]!, mobs2[1]!, mobs2[2]!, 0.75);
    expectClose(mobs2[0]!.get_fill_opacity(), 0.0);
    expect(mobs2[0]!.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.SpiralIn / __init__ / interpolate_mobject
// Python (Group of squares at x=0 and x=1):
//   scale_factor 8, fade_in_fraction 0.3, shape_center [0.5,0,0]
//   shape0 final [0,0,0], initial [-4,0,0]
//   interpolate_mobject(0.5) -> center0 [3.0, ~0, 0], fill 0.0
// ---------------------------------------------------------------------------
describe("SpiralIn", () => {
  function twoSquares(): Group {
    const g = new Group(new Square(), new Square());
    g.submobjects[1]!.shift([1, 0, 0]);
    return g;
  }

  it("stores spiral state like Python", () => {
    const g = twoSquares();
    const sp = new SpiralIn(g);
    expect(sp.scale_factor).toBe(8);
    expectClose(sp.fade_in_fraction, 0.3);
    expectRowsClose([sp.shape_center], [[0.5, 0, 0]]);
    const s0 = g.submobjects[0]! as unknown as {
      final_position: number[];
      initial_position: number[];
    };
    expectRowsClose([s0.final_position], [[0, 0, 0]]);
    expectRowsClose([s0.initial_position], [[-4, 0, 0]]);
  });

  it("interpolates along the spiral", () => {
    const g = twoSquares();
    const sp = new SpiralIn(g);
    sp.begin();
    sp.interpolate_mobject(0.5);
    expectRowsClose([g.submobjects[0]!.get_center()], [[3.0, 0, 0]], 1e-7);
    expectClose(g.submobjects[0]!.get_fill_opacity(), 0.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.ShowIncreasingSubsets / __init__ / interpolate_mobject
// Python (4 squares):
//   len(all_submobs) 4, suspend False, reverse False
//   interpolate_mobject(0.5) fill [1,1,0,0]; (1.0) all 1; (0.0) all 0
// ---------------------------------------------------------------------------
describe("ShowIncreasingSubsets", () => {
  function fourSquares(): Group {
    const g = new Group(new Square(), new Square(), new Square(), new Square());
    g.submobjects.forEach((m, i) => m.shift([i, 0, 0]));
    return g;
  }

  it("reveals subsets by index", () => {
    const g = fourSquares();
    const sis = new ShowIncreasingSubsets(g);
    expect(sis.all_submobs.length).toBe(4);
    expect(sis.suspend_mobject_updating).toBe(false);
    expect(sis.reverse_rate_function).toBe(false);
    sis.begin();
    sis.interpolate_mobject(0.5);
    expect(sis.all_submobs.map((m) => m.get_fill_opacity())).toEqual([1, 1, 0, 0]);
    sis.interpolate_mobject(1.0);
    expect(sis.all_submobs.map((m) => m.get_fill_opacity())).toEqual([1, 1, 1, 1]);
    sis.interpolate_mobject(0.0);
    expect(sis.all_submobs.map((m) => m.get_fill_opacity())).toEqual([0, 0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.movement.Homotopy / __init__ / interpolate_submobject
// Python:
//   Homotopy(f, s): run_time 3, apply_function_kwargs {}
//   interpolate_submobject(sub, start, 0.5) with f=(x+t, y+2t, z):
//     points[0] [1.5,2,0], points[4] [-0.5,2,0]
// ---------------------------------------------------------------------------
describe("Homotopy", () => {
  it("stores run_time and kwargs, and applies the homotopy at time t", () => {
    const s = new Square(2.0);
    const h = new Homotopy((x, y, z, t) => [x + t, y + 2 * t, z], s);
    expect(h.run_time).toBe(3);
    expect(h.apply_function_kwargs).toEqual({});
    h.begin();
    const sub = h.mobject.family_members_with_points()[0]!;
    const start = h.starting_mobject.family_members_with_points()[0]!;
    const ret = h.interpolate_submobject(sub, start, 0.5);
    expect(ret).toBe(h);
    expectRowsClose(rowsOf(sub.points, sub.dim).slice(0, 1), [[1.5, 2, 0]], 1e-7);
    expectRowsClose(rowsOf(sub.points, sub.dim).slice(4, 5), [[-0.5, 2, 0]], 1e-7);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.movement.PhaseFlow / __init__ / interpolate_mobject
// Python:
//   PhaseFlow(f, s2, virtual_time=2.0): virtual_time 2.0, suspend False,
//     rate_func linear
//   interpolate 0.0/0.5/1.0 -> centers [0,0,0], [1,0,0], [2,0,0]
// ---------------------------------------------------------------------------
describe("PhaseFlow", () => {
  it("advects the mobject along the vector field", () => {
    const s2 = new Square(2.0);
    const pf = new PhaseFlow((p) => [1, 0, 0].map((v, i) => v + 0 * (p[i] ?? 0)), s2, {
      virtual_time: 2.0,
    });
    expect(pf.virtual_time).toBe(2.0);
    expect(pf.suspend_mobject_updating).toBe(false);
    expect(pf.rate_func).toBe(linear);
    pf.begin();
    pf.interpolate_mobject(0.0);
    expectRowsClose([s2.get_center()], [[0, 0, 0]], 1e-7);
    pf.interpolate_mobject(0.5);
    expectRowsClose([s2.get_center()], [[1, 0, 0]], 1e-7);
    pf.interpolate_mobject(1.0);
    expectRowsClose([s2.get_center()], [[2, 0, 0]], 1e-7);
  });

  it("uses defaults virtual_time=1, suspend=False, linear", () => {
    const pf = new PhaseFlow((p) => [...p], new Square());
    expect(pf.virtual_time).toBe(1);
    expect(pf.suspend_mobject_updating).toBe(false);
    expect(pf.rate_func).toBe(linear);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.movement.MoveAlongPath
// Python:
//   MoveAlongPath(dot, Line([0,0,0],[4,0,0])): suspend False
//   interpolate_mobject(0.5) -> center [2,0,0]
// ---------------------------------------------------------------------------
describe("MoveAlongPath", () => {
  it("moves the mobject along the path proportion", () => {
    const dot = new Square(0.2);
    const path = new Line([0, 0, 0], [4, 0, 0]);
    const mp = new MoveAlongPath(dot, path as never);
    expect(mp.suspend_mobject_updating).toBe(false);
    mp.begin();
    mp.interpolate_mobject(0.5);
    expectRowsClose([dot.get_center()], [[2, 0, 0]], 1e-7);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Wiggle / __init__ / interpolate_submobject
// Python:
//   Wiggle(s): scale 1.1, angle 0.01*TAU, n 6, about None/None, run_time 2
//   interpolate_submobject(sub, start, 0.25):
//     points[0] [1.082463185416053,1.0165005913519836,0]
//     points[5] [-1.0384881227066733,0.38280859316004084,0]
//   Wiggle(copy, scale_value=1.5, n_wiggles=3, rotation_angle=0.5) at 0.5:
//     points[0] [2.155930738364773,0.5596540461516601,0]
// ---------------------------------------------------------------------------
describe("Wiggle", () => {
  it("stores defaults like Python", () => {
    const w = new Wiggle(new Square());
    expectClose(w.scale_value, 1.1);
    expectClose(w.rotation_angle, 0.01 * TAU);
    expect(w.n_wiggles).toBe(6);
    expect(w.scale_about_point).toBeNull();
    expect(w.rotate_about_point).toBeNull();
    expect(w.run_time).toBe(2);
  });

  it("wiggles submobject points like Python", () => {
    const s = new Square(2.0);
    const w = new Wiggle(s);
    w.begin();
    const sub = w.mobject.family_members_with_points()[0]!;
    const start = w.starting_mobject.family_members_with_points()[0]!;
    w.interpolate_submobject(sub, start, 0.25);
    const rows = rowsOf(sub.points, sub.dim);
    // Python `points[:, :] = ...` copies values into distinct storage.
    expect(sub.points).not.toBe(start.points);
    expectRowsClose(rows.slice(0, 1), [[1.082463185416053, 1.0165005913519836, 0]], 1e-7);
    expectRowsClose(rows.slice(5, 6), [[-1.0384881227066733, 0.38280859316004084, 0]], 1e-7);

    const s2 = new Square(2.0);
    const w2 = new Wiggle(s2, { scale_value: 1.5, n_wiggles: 3, rotation_angle: 0.5 });
    w2.begin();
    const sub2 = w2.mobject.family_members_with_points()[0]!;
    const start2 = w2.starting_mobject.family_members_with_points()[0]!;
    w2.interpolate_submobject(sub2, start2, 0.5);
    // NOTE: re-probed on a fresh Square (the first probe accidentally reused
    // a Square already mutated by the earlier Wiggle): step-by-step
    // scale(1.5)-then-rotate(-0.5) of [1,1,0] gives [2.03551..., 0.59723...].
    expectRowsClose(rowsOf(sub2.points, sub2.dim).slice(0, 1), [[2.035512150741863, 0.5972355349292544, 0]], 1e-7);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.update.UpdateFromFunc / __init__
// Python:
//   UpdateFromFunc(sq, f): suspend False
//   begin(); interpolate_mobject(0.3) with f=shift([1,0,0])
//     -> center [2,0,0], 2 calls (begin interpolates at 0 first)
// ---------------------------------------------------------------------------
describe("UpdateFromFunc", () => {
  it("calls the update function on interpolate", () => {
    const sq = new Square();
    let calls = 0;
    const u = new UpdateFromFunc(sq, (m) => {
      calls += 1;
      m.shift([1, 0, 0]);
    });
    expect(u.suspend_mobject_updating).toBe(false);
    u.begin();
    u.interpolate_mobject(0.3);
    expect(calls).toBe(2);
    expectRowsClose([sq.get_center()], [[2, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.numbers.ChangingDecimal
// NOTE: the real DecimalNumber (manim.mobject.text.numbers, TeX-based) cannot
// be constructed here — `latex` is not installed (FileNotFoundError). The port
// checks structurally for `set_value`, so a stub is used to verify the same
// logic Python runs: validity check, suspend default False, and
// set_value(number_update_func(rate_func(alpha))).
// ---------------------------------------------------------------------------
describe("ChangingDecimal", () => {
  // NOTE: Python's DecimalNumber IS a Mobject, and Animation.__init__ (both
  // languages) rejects non-Mobjects; the stub is therefore a real Square
  // carrying the DecimalNumber surface (number/set_value).
  type DecimalStub = Square & { number: number; set_value: (v: number) => void };
  function decimalStub(): DecimalStub {
    const stub = new Square() as DecimalStub;
    (stub as unknown as Record<string, unknown>).number = 0;
    (stub as unknown as Record<string, unknown>).set_value = (v: number) => {
      (stub as unknown as Record<string, unknown>).number = v;
    };
    return stub;
  }

  it("drives set_value from the update func", () => {
    const stub = decimalStub();
    const cd = new ChangingDecimal(stub, (a) => a * 10);
    expect(cd.suspend_mobject_updating).toBe(false);
    cd.begin();
    cd.interpolate_mobject(0.5);
    // smooth(0.5) == 0.5, so value is 5.0
    expectClose(stub.number, 5.0);
  });

  it("rejects mobjects without set_value", () => {
    expect(() => new ChangingDecimal(new Square() as never, (a) => a)).toThrowError(
      "ChangingDecimal can only take in a DecimalNumber",
    );
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts
// Python:
//   base get_mobject_parts/key -> NotImplementedError('To be implemented in subclass.')
//   TransformMatchingShapes.get_mobject_parts(s): 1 part == family_members_with_points
//   key: square==square, square==scaled+shifted square, square!=circle;
//     mobject restored afterwards (center/height unchanged)
// ---------------------------------------------------------------------------
describe("TransformMatchingAbstractBase / TransformMatchingShapes", () => {
  it("base statics raise NotImplementedError", () => {
    const s = new Square();
    expect(() => TransformMatchingAbstractBase.get_mobject_parts(s)).toThrowError(
      "To be implemented in subclass.",
    );
    try {
      TransformMatchingAbstractBase.get_mobject_parts(s);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("NotImplementedError");
    }
    expect(() => TransformMatchingAbstractBase.get_mobject_key(s)).toThrowError(
      "To be implemented in subclass.",
    );
  });

  it("shapes parts are the pointful family members", () => {
    const s = new Square();
    const parts = TransformMatchingShapes.get_mobject_parts(s);
    expect(parts.length).toBe(s.family_members_with_points().length);
    expect(parts[0]).toBe(s.family_members_with_points()[0]);
  });

  it("shape keys match under rigid/scale transforms, differ across shapes", () => {
    const a = new Square();
    const b = new Square();
    const c = new Square();
    c.scale(2).shift([5, 3, 0]);
    const d = new Circle();
    const ka = TransformMatchingShapes.get_mobject_key(a);
    const kb = TransformMatchingShapes.get_mobject_key(b);
    const kc = TransformMatchingShapes.get_mobject_key(c);
    const kd = TransformMatchingShapes.get_mobject_key(d);
    expect(ka).toBe(kb);
    expect(ka).toBe(kc);
    expect(ka).not.toBe(kd);
    // key computation restores the mobject: c unchanged afterwards
    expectRowsClose([c.get_center()], [[5, 3, 0]], 1e-7);
    expectClose(c.height, 4.0);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.moving_camera.MovingCamera.frame_height / frame_center
// Python:
//   mc = MovingCamera(): frame_height 8.0, frame_width 14.222222222222221,
//     frame_center [0,0,0]
//   mc.frame_height = 4.0 -> height 4.0, width unchanged 14.222222222222221
//   mc.frame_center = [1,2,0] -> [1,2,0]
// ---------------------------------------------------------------------------
describe("MovingCamera.frame_height / frame_center", () => {
  it("reads the frame and writes through to it", () => {
    const mc = new MovingCamera();
    expectClose(mc.frame_height, 8.0);
    expectClose(mc.frame_width, 14.222222222222221);
    expectRowsClose([[...mc.frame_center]], [[0, 0, 0]]);
    mc.frame_height = 4.0;
    expectClose(mc.frame_height, 4.0);
    expectClose(mc.frame_width, 14.222222222222221);
    mc.frame_center = [1, 2, 0];
    expectRowsClose([[...mc.frame_center]], [[1, 2, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.zoomed_scene.ZoomedScene.get_zoom_factor
// Python: zoomed_camera.frame.height / zoomed_display.height. A full
// ZoomedScene cannot be constructed headless here (needs a Cairo renderer),
// so the ratio logic is tested with stubs, mirroring the Python formula.
// ---------------------------------------------------------------------------
describe("ZoomedScene.get_zoom_factor", () => {
  it("returns frame height over display height", () => {
    const zs = Object.create(ZoomedScene.prototype) as ZoomedScene;
    (zs as unknown as Record<string, unknown>).zoomed_camera = {
      frame: { height: 8.0 },
    };
    (zs as unknown as Record<string, unknown>).zoomed_display = { height: 2.0 };
    expectClose(zs.get_zoom_factor(), 4.0);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch layers 12..13 integration", () => {
  it("replace feeds a Transform chain matching Python", () => {
    // Python:
    //   s = Square(2); r = Rectangle(4x1).move_to([3,1,0]); s.replace(r)
    //   t = Transform(s, s.copy().shift([1,0,0])); t.begin(); t.interpolate(0.5)
    //   -> center [3.5,1,0], width 4.0
    const s = new Square(2.0);
    const r = new Rectangle({ width: 4.0, height: 1.0 });
    r.move_to([3, 1, 0]);
    s.replace(r);
    const t = new Transform(s, s.copy().shift([1, 0, 0]));
    t.begin();
    t.interpolate(0.5);
    expectRowsClose([t.mobject.get_center()], [[3.5, 1, 0]], 1e-7);
    expectClose(t.mobject.width, 4.0, 1e-7);
  });

  it("Create update path against a VMobject matches Python point counts", () => {
    // Python: begin() interpolates at 0 -> 4 points; interpolate(0.5) -> 12;
    // interpolate(1.0)/finish() restores the full 16-point shape.
    const s = new Square(2.0);
    const c = new Create(s);
    c.begin();
    expect(c.mobject.get_num_points()).toBe(4);
    c.interpolate(0.5);
    expect(c.mobject.get_num_points()).toBe(12);
    c.interpolate(1.0);
    expect(c.mobject.get_num_points()).toBe(16);
    c.finish();
    expectRowsClose([c.mobject.get_center()], [[0, 0, 0]]);
  });

  it("AnimationGroup of Transforms interpolates members jointly", () => {
    // Python: AnimationGroup(Transform(s1, s1+[2,0,0]), Transform(s2, s2+[0,2,0]))
    // interpolate(0.5) moves each member halfway (linear default rate).
    const s1 = new Square(2.0);
    const s2 = new Square(2.0);
    const g = new AnimationGroup(
      new Transform(s1, s1.copy().shift([2, 0, 0])),
      new Transform(s2, s2.copy().shift([0, 2, 0])),
    );
    g.begin();
    g.interpolate(0.5);
    expectRowsClose([s1.get_center()], [[1, 0, 0]], 1e-7);
    expectRowsClose([s2.get_center()], [[0, 1, 0]], 1e-7);
  });

  it("config frame_height feeds MovingCamera defaults", () => {
    expectClose(new MovingCamera().frame_height, Number(config.frame_height));
  });
});
