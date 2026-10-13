// Ground-truth tests for batch layers 13..13 (targets-1122).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here; each block cites the
// probe in a "Python:" comment. TS is asserted against those values (1e-9
// relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): nothing in this batch touches text/tex.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { Animation, prepare_animation } from "../src/animation/animation.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { Create } from "../src/animation/creation.js";
import { Transform, ApplyMethod } from "../src/animation/transform.js";
import { Scene } from "../src/scene/scene.js";
import { get_hash_from_play_call } from "../src/utils/hash.js";
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
// manim.mobject.types.vectorized_mobject.VMobject.set_anchors_and_handles
// Python:
//   v.set_anchors_and_handles([[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]],
//     [[0,2,0],[1,2,0]], [[0,3,0],[1,3,0]])
//   -> points [[0,0,0],[0,1,0],[0,2,0],[0,3,0],[1,0,0],[1,1,0],[1,2,0],[1,3,0]]
//   (points[index::nppcc] = array), get_num_curves() == 2
//   mismatched lengths -> AssertionError
// ---------------------------------------------------------------------------
describe("VMobject.set_anchors_and_handles", () => {
  it("interleaves anchors/handles into points", () => {
    const v = new VMobject();
    v.set_anchors_and_handles(
      [[0, 0, 0], [1, 0, 0]],
      [[0, 1, 0], [1, 1, 0]],
      [[0, 2, 0], [1, 2, 0]],
      [[0, 3, 0], [1, 3, 0]],
    );
    expectRowsClose(rowsOf(v.points, 3), [
      [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
      [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0],
    ]);
    expect(v.get_num_curves()).toBe(2);
  });

  it("round-trips through get_anchors_and_handles", () => {
    const v = new VMobject();
    const a1 = [[0, 0, 0], [1, 0, 0]];
    const h1 = [[0, 1, 0], [1, 1, 0]];
    const h2 = [[0, 2, 0], [1, 2, 0]];
    const a2 = [[0, 3, 0], [1, 3, 0]];
    v.set_anchors_and_handles(a1, h1, h2, a2);
    const [ra1, rh1, rh2, ra2] = v.get_anchors_and_handles();
    expectRowsClose(ra1, a1);
    expectRowsClose(rh1, h1);
    expectRowsClose(rh2, h2);
    expectRowsClose(ra2, a2);
  });

  it("rejects mismatched lengths", () => {
    const v = new VMobject();
    expect(() =>
      v.set_anchors_and_handles([[0, 0, 0]], [[0, 1, 0]], [[0, 2, 0]], [[0, 3, 0], [1, 3, 0]]),
    ).toThrowError(/len\(anchors1\)/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Group (__init__)
// Python:
//   Group(a, b).submobjects == [a, b], get_family() has 3 members
//   Group() -> 0 submobjects; Group(a, a) -> 1 submobject (repetitions ignored)
// ---------------------------------------------------------------------------
describe("Group.__init__", () => {
  it("adds constructor mobjects as submobjects", () => {
    const a = new Square();
    const b = new Square();
    const g = new Group(a, b);
    expect(g.submobjects).toEqual([a, b]);
    expect(g.get_family().length).toBe(3);
  });

  it("builds an empty group with no args", () => {
    expect(new Group().submobjects).toEqual([]);
  });

  it("ignores repeated mobjects", () => {
    const a = new Square();
    const g = new Group(a, a);
    expect(g.submobjects.length).toBe(1);
    expect(g.submobjects[0]).toBe(a);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing.get_hash_from_play_call
// Python:
//   get_hash_from_play_call(scene, camera, [Create(a)], [a])
//     -> "819936691_780824218_557880437" (deterministic across calls,
//        matches ^\d+_\d+_\d+$)
// NOTE deviation: the TS port cannot reproduce the exact Python digests —
// Python hashes its own pickle/JSON-flavoured serialisation (with spaces,
// single-quoted reprs, numpy dtypes), while TS hashes its own JSON output.
// What must agree: the "crc32_crc32_crc32" shape, determinism, and
// sensitivity to the inputs.
// ---------------------------------------------------------------------------
describe("get_hash_from_play_call", () => {
  function makeScene() {
    const scene = new Scene();
    const a = new Square();
    const anim = new Create(a);
    return { scene, a, anim };
  }

  it("returns a crc32_crc32_crc32 string", () => {
    const { scene, a, anim } = makeScene();
    const hash = get_hash_from_play_call(scene, { kind: "camera" }, [anim], [a]);
    expect(hash).toMatch(/^\d+_\d+_\d+$/);
  });

  it("is deterministic for the same inputs", () => {
    const first = makeScene();
    const second = makeScene();
    const h1 = get_hash_from_play_call(first.scene, { kind: "camera" }, [first.anim], [first.a]);
    const h1b = get_hash_from_play_call(first.scene, { kind: "camera" }, [first.anim], [first.a]);
    expect(h1b).toBe(h1);
    // Freshly built equivalents serialise value-identically, so they agree too.
    const h2 = get_hash_from_play_call(second.scene, { kind: "camera" }, [second.anim], [second.a]);
    expect(h2).toBe(h1);
  });

  it("changes when the mobject list changes", () => {
    const { scene, a, anim } = makeScene();
    const cam = { kind: "camera" };
    const h1 = get_hash_from_play_call(scene, cam, [anim], [a]);
    const b = new Square();
    b.move_to([5, 0, 0]);
    const h2 = get_hash_from_play_call(scene, cam, [anim], [b]);
    expect(h2).not.toBe(h1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker (remaining API)
// Python:
//   ValueTracker(2.5).points.shape == (1, 3), points == [[2.5, 0, 0]]
//   set_value returns self; increment_value adds in place and returns self
// (Arithmetic dunders + interpolate are covered in batch-layer13.test.ts;
//  init/get/bool/floordiv/mod/mul/pow/truediv in batch-targets-20260925.)
// ---------------------------------------------------------------------------
describe("ValueTracker set/increment", () => {
  it("stores the value at points[0,0] with shape (1, 3)", () => {
    const v = new ValueTracker(2.5);
    expect(v.points.shape).toEqual([1, 3]);
    expectRowsClose(rowsOf(v.points, 3), [[2.5, 0, 0]]);
  });

  it("set_value replaces the value and returns self", () => {
    const v = new ValueTracker(2.5);
    expect(v.set_value(7.25)).toBe(v);
    expectClose(v.get_value(), 7.25);
  });

  it("increment_value adds in place and returns self", () => {
    const v = new ValueTracker(7.25);
    expect(v.increment_value(1.5)).toBe(v);
    expectClose(v.get_value(), 8.75);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PMobject (remaining API)
// Python:
//   PMobject().points.shape == (0,3), rgbas.shape == (0,4)
//   add_points([[0,0,0],[1,1,1]]) -> points kept, rgbas default to white [1,1,1,1]
//   rgbas length mismatch -> ValueError("points and rgbas must have same length")
//   get_array_attrs() == ["points", "rgbas"]
// (Other PMobject methods covered in batch-layer13.test.ts.)
// ---------------------------------------------------------------------------
describe("PMobject add_points", () => {
  it("starts empty and defaults new-point colors to white", () => {
    const p = new PMobject();
    expect(p.points.shape).toEqual([0, 3]);
    expect(p.rgbas.shape).toEqual([0, 4]);
    p.add_points([[0, 0, 0], [1, 1, 1]]);
    expectRowsClose(rowsOf(p.points, 3), [[0, 0, 0], [1, 1, 1]]);
    expectRowsClose(rowsOf(p.rgbas, 4), [[1, 1, 1, 1], [1, 1, 1, 1]]);
  });

  it("rejects rgbas of a different length", () => {
    const p = new PMobject();
    expect(() =>
      p.add_points([[0, 0, 0]], [[0, 0, 0, 1], [1, 1, 1, 1]]),
    ).toThrowError(/points and rgbas must have same length/);
  });

  it("exposes points and rgbas as array attrs", () => {
    expect(new PMobject().get_array_attrs()).toEqual(["points", "rgbas"]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_restructured_mobject_list
//   .add_safe_mobjects_from_list (nested closure)
// Python:
//   g = Group(a, b): restructure([g], [a]) -> [b]
//   nested: outer = Group(inner = Group(a, b)): restructure([outer], [a]) -> [b]
//   empty to_remove -> identity; everything removed -> []
// ---------------------------------------------------------------------------
describe("Scene.get_restructured_mobject_list", () => {
  it("excludes a removed submobject, keeping its sibling", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    const g = new Group(a, b);
    expect(scene.get_restructured_mobject_list([g], [a])).toEqual([b]);
  });

  it("descends through nested groups", () => {
    // Python: outer=Group(inner); restructure([outer],[a]) -> [Square(b)]
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    const inner = new Group(a, b);
    const outer = new Group(inner);
    const out = scene.get_restructured_mobject_list([outer], [a]);
    expect(out).toEqual([b]);
  });

  it("keeps everything when nothing is removed, drops all when all removed", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    const g = new Group(a, b);
    expect(scene.get_restructured_mobject_list([g], [])).toEqual([g]);
    expect(scene.get_restructured_mobject_list([g], [g])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.AnimationGroup
// Python (Create default run_time == 1):
//   AnimationGroup(a1, a2): run_time == 1.0, max_end_time == 1.0,
//     starts [0, 0], ends [1, 1]; group has 0 submobjects (Creates are introducers)
//   AnimationGroup(a1, a2, lag_ratio=0.5): starts [0, 0.5], ends [1, 1.5],
//     run_time == 1.5
//   AnimationGroup().begin() -> ValueError("Trying to play ... without animations")
// ---------------------------------------------------------------------------
describe("AnimationGroup", () => {
  it("runs introducer animations together with run_time 1", () => {
    const s1 = new Square();
    const s2 = new Square();
    const g = new AnimationGroup(new Create(s1), new Create(s2));
    expectClose(g.run_time, 1.0);
    expectClose(g.max_end_time, 1.0);
    expect(g.anims_with_timings.map((t) => t.start)).toEqual([0, 0]);
    expect(g.anims_with_timings.map((t) => t.end)).toEqual([1, 1]);
    // Creates are introducers, so the auto-built group is empty (Python parity).
    expect(g.group.submobjects.length).toBe(0);
  });

  it("staggers timings with lag_ratio", () => {
    const s1 = new Square();
    const s2 = new Square();
    const g = new AnimationGroup(new Create(s1), new Create(s2), { lag_ratio: 0.5 });
    expect(g.anims_with_timings.map((t) => t.start)).toEqual([0, 0.5]);
    expect(g.anims_with_timings.map((t) => t.end)).toEqual([1, 1.5]);
    expectClose(g.run_time, 1.5);
    expectClose(g.max_end_time, 1.5);
  });

  it("keeps non-introducer mobjects in the auto-built group", () => {
    const s1 = new Square();
    const s2 = new Square();
    const t = new Transform(s1, s2);
    const g = new AnimationGroup(t);
    expect(g.group.submobjects).toEqual([s1]);
  });

  it("refuses to begin with no animations", () => {
    expect(() => new AnimationGroup().begin()).toThrowError(/without animations/);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_moving_mobjects.consider_moving (nested closure)
// Python (scene holds a, b):
//   get_moving_mobjects(Create(a)) -> 2 mobjects
//   c with an updater added -> get_moving_mobjects(Create(a)) -> 3
//   b in foreground_mobjects -> get_moving_mobjects(Create(a)) -> 2
// ---------------------------------------------------------------------------
describe("Scene.get_moving_mobjects", () => {
  it("returns the family slice from the first moving mobject", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    scene.add(a, b);
    expect(scene.get_moving_mobjects(new Create(a)).length).toBe(2);
  });

  it("treats mobjects with updaters as moving", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    const c = new Square();
    scene.add(a, b, c);
    c.add_updater(() => undefined);
    expect(scene.get_moving_mobjects(new Create(a)).length).toBe(3);
  });

  it("treats foreground mobjects as moving", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    scene.add(a, b);
    scene.foreground_mobjects.push(b);
    expect(scene.get_moving_mobjects(new Create(a)).length).toBe(2);
  });

  it("returns [] when nothing moves", () => {
    const scene = new Scene();
    scene.add(new Square());
    expect(scene.get_moving_mobjects()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.prepare_animation
// Python:
//   prepare_animation(FadeIn-like Animation) is the same object
//   prepare_animation(s.animate.scale(2)) -> _MethodAnimation
//   prepare_animation(42) -> TypeError("Object 42 cannot be converted ...")
// NOTE deviation: the TS port additionally accepts bare Mobjects and
// zero-arg factories (documented in animation.ts); those paths have no
// Python counterpart and are only smoke-tested here.
// ---------------------------------------------------------------------------
describe("prepare_animation", () => {
  it("passes animations through unchanged", () => {
    const anim = new Create(new Square());
    expect(prepare_animation(anim)).toBe(anim);
  });

  it("builds _AnimationBuilder factories", () => {
    const s = new Square();
    const built = prepare_animation(s.animate.scale(2) as unknown as Animation);
    expect(built).toBeInstanceOf(Animation);
    expect(built).not.toBe(s.animate);
  });

  it("rejects non-animations", () => {
    expect(() => prepare_animation(42 as unknown as Animation)).toThrow(TypeError);
    expect(() => prepare_animation(42 as unknown as Animation)).toThrowError(
      /cannot be converted to an animation/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.__init__
// Python: Scene() -> mobjects [], foreground [], animations None,
//   random_seed None (config.seed unset), skip_animations False,
//   always_update_mobjects False, renderer CairoRenderer, time 0.0.
// NOTE deviations: the TS port has no cairo renderer yet, so `renderer` is
// null (camera_class defaults to an empty class), `animations` starts as []
// instead of None, and the seed only mirrors config.seed without reseeding
// the global RNG.
// ---------------------------------------------------------------------------
describe("Scene.__init__", () => {
  it("starts with empty mobject lists and stored flags", () => {
    const scene = new Scene();
    expect(scene.mobjects).toEqual([]);
    expect(scene.foreground_mobjects).toEqual([]);
    expect(scene.animations).toEqual([]);
    expect(scene.moving_mobjects).toEqual([]);
    expect(scene.static_mobjects).toEqual([]);
    expect(scene.skip_animations).toBe(false);
    expect(scene.always_update_mobjects).toBe(false);
    expect(scene.interactive_mode).toBe(false);
    expect(scene.subcaptions).toEqual([]);
  });

  it("honours the options bag", () => {
    const scene = new Scene({ skip_animations: true, always_update_mobjects: true, random_seed: 42 });
    expect(scene.skip_animations).toBe(true);
    expect(scene.always_update_mobjects).toBe(true);
    expect(scene.random_seed).toBe(42);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.time
// Python: `time` returns self.renderer.time (0.0 on a fresh scene).
// ---------------------------------------------------------------------------
describe("Scene.time", () => {
  it("is 0 on a fresh scene without a renderer", () => {
    expect(new Scene().time).toBe(0);
  });

  it("reflects the renderer's clock when present", () => {
    const scene = new Scene();
    scene.renderer = { time: 2.5 };
    expectClose(scene.time, 2.5);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add_subcaption
// Python appends srt.Subtitle(index, content, start=time+offset,
// end=time+offset+duration) to renderer.file_writer.subcaptions.
// NOTE deviation: with no renderer/file_writer the TS port stores entries
// in the local `subcaptions` list instead of raising (same fields).
// ---------------------------------------------------------------------------
describe("Scene.add_subcaption", () => {
  it("stores subtitles locally when there is no file writer", () => {
    const scene = new Scene();
    scene.add_subcaption("Hello square!", 1);
    expect(scene.subcaptions.length).toBe(1);
    const [sub] = scene.subcaptions;
    expect(sub?.index).toBe(0);
    expect(sub?.content).toBe("Hello square!");
    expectClose(sub?.start ?? NaN, 0);
    expectClose(sub?.end ?? NaN, 1);
  });

  it("applies the offset to start and end", () => {
    const scene = new Scene();
    scene.add_subcaption("shifted", 2, 0.5);
    const [sub] = scene.subcaptions;
    expectClose(sub?.start ?? NaN, 0.5);
    expectClose(sub?.end ?? NaN, 2.5);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_run_time
// Python: max(animation.run_time); validated via validate_run_time with
// self.play / "total run_time" (0 -> ValueError "... total run_time of 0 ...").
// ---------------------------------------------------------------------------
describe("Scene.get_run_time", () => {
  it("returns the maximum animation run_time", () => {
    const scene = new Scene();
    const slow = new Create(new Square(), { run_time: 3 });
    const fast = new Create(new Square(), { run_time: 2 });
    expectClose(scene.get_run_time([slow, fast]), 3);
  });

  it("rejects a non-positive total run_time", () => {
    const scene = new Scene();
    expect(() => scene.get_run_time([])).toThrowError();
    expect(() =>
      scene.get_run_time([new Create(new Square(), { run_time: 0 })]),
    ).toThrowError(/total run_time/);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.play
// Python: delegates to renderer.play, then appends a subcaption as
// add_subcaption(subcaption, duration ?? run_time, -run_time + offset).
// NOTE deviation: the TS port has no cairo renderer, so animations run
// inline (begin/finish) here; the subcaption math is identical.
// ---------------------------------------------------------------------------
describe("Scene.play", () => {
  it("runs animations to completion inline", () => {
    const scene = new Scene();
    const s = new Square();
    const target = new Square();
    target.move_to([4, 0, 0]);
    scene.play(new Transform(s, target));
    expectRowsClose([[...s.get_center()]], [[4.0, 0.0, 0.0]]);
    expect(scene.animations.length).toBe(1);
  });

  it("appends the subcaption with renderer-time math", () => {
    // No renderer -> time stays 0, run_time 0, duration defaults to 0,
    // offset -0 + 0.25.
    const scene = new Scene();
    scene.play(new Create(new Square()), { subcaption: "hi", subcaption_offset: 0.25 });
    expect(scene.subcaptions.length).toBe(1);
    expect(scene.subcaptions[0]?.content).toBe("hi");
    expectClose(scene.subcaptions[0]?.start ?? NaN, 0.25);
    expectClose(scene.subcaptions[0]?.end ?? NaN, 0.25);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform.__init__
// Python signature: (mobject, target_mobject=None, path_func=None, path_arc=0,
// path_arc_axis=OUT, path_arc_centers=None, replace_mobject_with_target_in_scene=False)
//   target None -> empty Mobject(); explicit target kept; path_arc stored.
// NOTE deviation: the TS port uses `mobject.copy()` (not an empty Mobject)
// as the default target when none is given.
// ---------------------------------------------------------------------------
describe("Transform.__init__", () => {
  it("stores path options and the explicit target", () => {
    const s = new Square();
    const t = new Square();
    const tr = new Transform(s, t);
    expect(tr.target_mobject).toBe(t);
    expectClose(tr.path_arc, 0);
    expect(tr.replace_mobject_with_target_in_scene).toBe(false);
    expect(typeof tr.path_func).toBe("function");
  });

  it("defaults a missing target to a copy of the mobject (deviation)", () => {
    // Python would store an empty Mobject() here; the TS port copies.
    const s = new Square();
    const tr = new Transform(s);
    expect(tr.target_mobject).not.toBe(s);
    expect(tr.target_mobject.get_num_points()).toBe(s.get_num_points());
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform (begin/interpolate path)
// Python: Transform(s1, s2 at [4,0,0]): begin + interpolate(0.5) -> center
// [2,0,0]; finish -> [4,0,0].
// ---------------------------------------------------------------------------
describe("Transform interpolate", () => {
  it("moves the mobject halfway at alpha 0.5 and completes on finish", () => {
    const s1 = new Square();
    const s2 = new Square();
    s2.move_to([4, 0, 0]);
    const tr = new Transform(s1, s2);
    tr.begin();
    tr.interpolate(0.5);
    expectRowsClose([[...s1.get_center()]], [[2.0, 0.0, 0.0]]);
    tr.finish();
    expectRowsClose([[...s1.get_center()]], [[4.0, 0.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyMethod.check_validity_of_input
// Python:
//   bound method (inspect.ismethod) with Mobject __self__ -> passes
//   invoked result / plain function -> ValueError("Whoops, looks like ...")
//   bound method of a non-Mobject -> AssertionError
// TS callers attach __self__/__func__ the way CPython bound methods carry them.
// ---------------------------------------------------------------------------
describe("ApplyMethod.check_validity_of_input", () => {
  function bindMobjectMethod(m: Mobject, name: string) {
    const func = (m as unknown as Record<string, (...args: unknown[]) => unknown>)[name];
    if (typeof func !== "function") throw new Error(`no method ${name}`);
    return Object.assign(func.bind(m), { __self__: m, __func__: func });
  }

  it("accepts a bound mobject method", () => {
    const s = new Square();
    const bound = bindMobjectMethod(s, "shift");
    const probe = new ApplyMethod(bound, [1, 0, 0]);
    expect(() => probe.check_validity_of_input(bound)).not.toThrow();
  });

  it("rejects an already-invoked result / plain function", () => {
    const s = new Square();
    const probe = new ApplyMethod(bindMobjectMethod(s, "shift"), [1, 0, 0]);
    const plain = Object.assign(() => undefined, {});
    expect(() => probe.check_validity_of_input(plain as never)).toThrowError(/accidentally invoked/);
  });

  it("rejects bound methods of non-mobjects", () => {
    const s = new Square();
    const probe = new ApplyMethod(bindMobjectMethod(s, "shift"), [1, 0, 0]);
    const fake = Object.assign(function (this: unknown) {}, { __self__: 42 });
    expect(() => probe.check_validity_of_input(fake as never)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (batch symbols combined with earlier-ported code)
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("AnimationGroup of Transforms completes inside Scene.play", () => {
    // Python: two Transforms to [4,0,0]/[0,4,0] finish at their targets;
    // the group reports max_end_time 1.0 and both squares land on target.
    const scene = new Scene();
    const s1 = new Square();
    const s2 = new Square();
    const t1 = new Square();
    t1.move_to([4, 0, 0]);
    const t2 = new Square();
    t2.move_to([0, 4, 0]);
    const group = new AnimationGroup(new Transform(s1, t1), new Transform(s2, t2));
    expectClose(group.max_end_time, 1.0);
    scene.play(group);
    expectRowsClose([[...s1.get_center()]], [[4.0, 0.0, 0.0]]);
    expectRowsClose([[...s2.get_center()]], [[0.0, 4.0, 0.0]]);
  });

  it("ValueTracker value positions a square that a Transform then moves", () => {
    // Python: tracker.set_value(3.0); square moves to [3,0,0]; Transform to
    // [6,0,0] finishes at [6,0,0].
    const tracker = new ValueTracker(0);
    tracker.set_value(3.0);
    const s = new Square();
    s.move_to([tracker.get_value(), 0, 0]);
    expectRowsClose([[...s.get_center()]], [[3.0, 0.0, 0.0]]);
    const target = new Square();
    target.move_to([6, 0, 0]);
    const tr = new Transform(s, target);
    tr.begin();
    tr.finish();
    expectRowsClose([[...s.get_center()]], [[6.0, 0.0, 0.0]]);
  });

  it("moving mobjects feed the restructured list", () => {
    // Python: scene [a, b]; moving = both under Create(a); excluding a
    // leaves [b].
    const scene = new Scene();
    const a = new Square();
    const b = new Square();
    scene.add(a, b);
    const moving = scene.get_moving_mobjects(new Create(a));
    expect(moving.length).toBe(2);
    expect(scene.get_restructured_mobject_list(moving, [a])).toEqual([b]);
  });

  it("set_anchors_and_handles output drives a VMobject point count", () => {
    // A 2-anchor setup yields 8 points / 2 curves; a PMobject cloud of the
    // same anchors has 2 points and matching array attrs.
    const v = new VMobject();
    v.set_anchors_and_handles(
      [[0, 0, 0], [1, 0, 0]],
      [[0, 1, 0], [1, 1, 0]],
      [[0, 2, 0], [1, 2, 0]],
      [[0, 3, 0], [1, 3, 0]],
    );
    expect(v.get_num_points()).toBe(8);
    const cloud = new PMobject();
    cloud.add_points([[0, 0, 0], [1, 0, 0]]);
    expect(cloud.get_num_points()).toBe(2);
    expect(cloud.get_array_attrs()).toEqual(["points", "rgbas"]);
  });
});
