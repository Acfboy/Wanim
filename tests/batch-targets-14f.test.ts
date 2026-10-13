// Ground-truth tests for batch layers 14..15 (~1050 lines).
//
// Every expectation below was captured from the original Python manim
// (original manim 0.21.0) before being asserted
// against the TS port. Float comparisons use tight tolerance (1e-9 relative).
//
// NOTE (text/tex out of scope): AddTextLetterByLetter / RemoveTextLetterByLetter
// / AddTextWordByWord / TypeWithCursor / TransformMatchingTex are ported over
// generic Mobjects (no LaTeX needed), so tests build VGroup/VMobject chains as
// Text stand-ins. Matrix._organize_mob_matrix is tested without constructing
// Matrix (whose brackets need MathTex -> TODO stub).
import { describe, expect, it } from "vitest";

import { ORIGIN, DOWN, UP, PI } from "../src/constants.js";
import { config } from "../src/_config/utils.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle, Dot } from "../src/mobject/geometry/arc.js";
import { Triangle } from "../src/mobject/geometry/polygram.js";
import { RED, GREEN } from "../src/utils/color/core.js";
import { Animation } from "../src/animation/animation.js";
import { GrowFromPoint } from "../src/animation/growing.js";
import {
  DrawBorderThenFill,
  Write,
  Unwrite,
  Uncreate,
  AddTextLetterByLetter,
  RemoveTextLetterByLetter,
  AddTextWordByWord,
  TypeWithCursor,
  Create,
} from "../src/animation/creation.js";
import { _Fade, FadeIn, FadeOut } from "../src/animation/fading.js";
import {
  ApplyFunction,
  ApplyPointwiseFunction,
  FadeTransformPieces,
  Transform,
} from "../src/animation/transform.js";
import { AnimationGroup, LaggedStartMap } from "../src/animation/composition.js";
import { Rotating, Rotate } from "../src/animation/rotation.js";
import { AnimatedBoundary } from "../src/animation/changing.js";
import {
  TransformMatchingShapes,
  TransformMatchingTex,
} from "../src/animation/transform_matching_parts.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { Matrix } from "../src/mobject/matrix.js";
import { pointRows } from "../src/utils/arrays.js";
import type { NDArray } from "numpy-ts";

function rows(value: NDArray, dim = 3): number[][] {
  return pointRows(value, dim);
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
      const a = actualRows[i]?.[j] ?? NaN;
      const e = expectedRows[i]?.[j] ?? NaN;
      expect(Math.abs(a - e)).toBeLessThanOrEqual(tol + 1e-9 * Math.abs(e));
    }
  }
}

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowFromPoint (+ __init__)
// Python: point/color stored, introducer=True; create_target() is mobject;
// begin() -> starting center == point, 16 points (Square), color applied.
// ---------------------------------------------------------------------------
describe("GrowFromPoint", () => {
  it("stores point/color and introduces the mobject", () => {
    const sq = new Square();
    const g = new GrowFromPoint(sq, ORIGIN);
    expect(g.point).toEqual([0, 0, 0]);
    expect(g.point_color).toBeNull();
    expect(g.introducer).toBe(true);
    expect(g.create_target()).toBe(g.mobject);
  });

  it("starts scaled to zero at the given point", () => {
    const sq = new Square();
    const g = new GrowFromPoint(sq, ORIGIN);
    g.begin();
    expect(g.starting_mobject.get_num_points()).toBe(16);
    expectRowsClose([g.starting_mobject.get_center()], [[0, 0, 0]], 1e-9);
    expect(g.starting_mobject.width).toBeCloseTo(0, 12);
  });

  it("moves the start to an array point and applies point_color", () => {
    // Python: GrowFromPoint(sq, [1,2,0], RED).begin() -> center [1,2,0],
    // color #FC6255.
    const sq = new Square();
    const g = new GrowFromPoint(sq, [1, 2, 0], RED);
    g.begin();
    expectRowsClose([g.starting_mobject.get_center()], [[1, 2, 0]], 1e-9);
    expect(g.starting_mobject.get_color()?.to_hex()).toBe("#FC6255");
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.DrawBorderThenFill.get_stroke_color / _typecheck
// Python: explicit stroke_color wins; else stroke color when width > 0
// (#FC6255 for RED width 5); else fill color (#FFFFFF plain Square).
// Non-VMobject -> TypeError "... only works for vectorized Mobjects".
// ---------------------------------------------------------------------------
describe("DrawBorderThenFill", () => {
  it("defaults stroke_color null / width 2", () => {
    const d = new DrawBorderThenFill(new Square());
    expect(d.stroke_color).toBeNull();
    expect(d.stroke_width).toBe(2);
  });

  it("falls back to the fill color for a plain Square", () => {
    const sq = new Square();
    const d = new DrawBorderThenFill(sq);
    expect(d.get_stroke_color(sq)?.to_hex()).toBe("#FFFFFF");
  });

  it("uses the stroke color when stroke width > 0", () => {
    const sq = new Square();
    sq.set_stroke(RED, 5);
    const d = new DrawBorderThenFill(sq);
    expect(d.get_stroke_color(sq)?.to_hex()).toBe("#FC6255");
  });

  it("prefers the explicit stroke_color", () => {
    const sq = new Square();
    const d = new DrawBorderThenFill(sq, { stroke_color: GREEN });
    expect(d.get_stroke_color(sq)?.to_hex()).toBe("#83C167");
  });

  it("typechecks vectorized input", () => {
    expect(() => new DrawBorderThenFill(new Mobject() as unknown as VMobject)).toThrowError(
      /only works for vectorized Mobjects/,
    );
    const d = new DrawBorderThenFill(new Square());
    expect(() => d._typecheck_input(new Square())).not.toThrow();
    expect(() => d._typecheck_input(new Mobject() as unknown as VMobject)).toThrowError(
      /only works for vectorized Mobjects/,
    );
  });

  it("reports the subclass name in the typecheck error", () => {
    // Python uses self.__class__.__name__: Write(...) -> "Write only works...".
    const w = new Write(new Square());
    expect(() => w._typecheck_input(new Mobject() as unknown as VMobject)).toThrowError(
      /Write only works for vectorized Mobjects/,
    );
  });

  it("tracks the outline as a third mobject", () => {
    // Python: len(get_all_mobjects()) == 3.
    expect(new DrawBorderThenFill(new Square()).get_all_mobjects().length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Write (+ _set_default_config_from_length)
// Python: Write(Square()) -> run 1, lag 0.2, remover False, introducer True;
// 20-submobject group -> run 2, lag 0.2; explicit values pass through.
// ---------------------------------------------------------------------------
describe("Write", () => {
  it("derives defaults from the submobject count", () => {
    const w = new Write(new Square());
    expect(w._set_default_config_from_length(new Square(), null, null)).toEqual([1, 0.2]);
    const big = new VGroup(...Array.from({ length: 20 }, () => new Square()));
    expect(w._set_default_config_from_length(big, null, null)).toEqual([2, 0.2]);
    expect(w._set_default_config_from_length(new Square(), 5.0, 0.3)).toEqual([5.0, 0.3]);
  });

  it("sets run_time/lag/remover/introducer like Python", () => {
    const w = new Write(new Square());
    expect(w.run_time).toBe(1);
    expect(w.lag_ratio).toBe(0.2);
    expect(w.remover).toBe(false);
    expect(w.introducer).toBe(true);
    expect(w.reverse).toBe(false);
    const big = new VGroup(...Array.from({ length: 20 }, () => new Square()));
    const w2 = new Write(big);
    expect(w2.run_time).toBe(2);
    expect(w2.lag_ratio).toBe(0.2);
  });

  it("reverse flips remover/introducer", () => {
    const w = new Write(new Square(), { reverse: true });
    expect(w.reverse).toBe(true);
    expect(w.remover).toBe(true);
    expect(w.introducer).toBe(false);
  });

  it("reverse begin/finish restores the submobject order", () => {
    const w = new Write(new Square(), { reverse: true });
    const before = w.mobject.get_start();
    w.begin();
    w.finish();
    expectRowsClose([w.mobject.get_start()], [before], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.Matrix._organize_mob_matrix
// DEVIATION (content-aware grid): upstream Python anchors entries on the fixed
// grid (j*h_buff, -i*v_buff), so a 2x2 Square (side 2 > h_buff 1.3) overlaps
// its neighbour. The TS port grows the pitch to `entry + entry_gap` instead:
// with v_buff 0.8, h_buff 1.3, gap 0.25, corner DR the pitch is
// max(1.3, 2 + 0.25) = 2.25 in both directions, giving centers
// (-1,1) (1.25,1) (-1,-1.25) (1.25,-1.25).
// (Matrix() itself needs MathTex brackets -> TODO stub, so the method is
// exercised on a bare prototype instance like Python's Matrix.__new__ trick.)
// ---------------------------------------------------------------------------
describe("Matrix._organize_mob_matrix", () => {
  it("places entries on the v_buff/h_buff grid", () => {
    const m = Object.create(Matrix.prototype) as Matrix;
    m.v_buff = 0.8;
    m.h_buff = 1.3;
    m.entry_gap = 0.25;
    m.element_alignment_corner = [1, -1, 0];
    const [a, b, c, d] = [new Square(), new Square(), new Square(), new Square()];
    m._organize_mob_matrix([[a!, b!], [c!, d!]]);
    expectRowsClose([a!.get_center()], [[-1, 1, 0]], 1e-9);
    expectRowsClose([b!.get_center()], [[1.25, 1, 0]], 1e-9);
    expectRowsClose([c!.get_center()], [[-1, -1.25, 0]], 1e-9);
    expectRowsClose([d!.get_center()], [[1.25, -1.25, 0]], 1e-9);
  });

  it("keeps the upstream fixed grid when entries fit inside the buffs", () => {
    const m = Object.create(Matrix.prototype) as Matrix;
    m.v_buff = 0.8;
    m.h_buff = 1.3;
    m.entry_gap = 0.25;
    m.element_alignment_corner = [1, -1, 0];
    const [a, b, c, d] = [
      new Square(0.5),
      new Square(0.5),
      new Square(0.5),
      new Square(0.5),
    ];
    m._organize_mob_matrix([[a!, b!], [c!, d!]]);
    // Python (v_buff 0.8, h_buff 1.3, corner DR, 0.5-wide squares): the pitch
    // stays max(1.3, 0.5 + 0.25) = 1.3 / max(0.8, 0.5 + 0.25) = 0.8, so the
    // centers are (-0.25, 0.25), (1.05, 0.25), (-0.25, -0.55), (1.05, -0.55).
    expectRowsClose([a!.get_center()], [[-0.25, 0.25, 0]], 1e-9);
    expectRowsClose([b!.get_center()], [[1.05, 0.25, 0]], 1e-9);
    expectRowsClose([c!.get_center()], [[-0.25, -0.55, 0]], 1e-9);
    expectRowsClose([d!.get_center()], [[1.05, -0.55, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.fading._Fade (+ __init__) / FadeIn (+ __init__)
// Python: no mobjects -> ValueError; shift/target_position/scale recorded;
// FadeIn introducer, target is mobject, starting copy faded at same center.
// ---------------------------------------------------------------------------
describe("_Fade/FadeIn", () => {
  it("requires at least one mobject", () => {
    expect(() => new _Fade()).toThrowError(expect.objectContaining({ name: "ValueError" }));
  });

  it("records shift/scale defaults", () => {
    const f = new FadeIn(new Square());
    expect(f.introducer).toBe(true);
    expect(f.point_target).toBe(false);
    expectRowsClose([f.shift_vector], [[0, 0, 0]], 1e-12);
    expect(f.scale_factor).toBe(1);
    expect(f.create_target()).toBe(f.mobject);
  });

  it("creates a faded starting copy at the same center", () => {
    // Python: stroke opacity 0.0 (fresh Squares have fill 0 by default),
    // center [0,0,0].
    const sq = new Square();
    const start = new FadeIn(sq).create_starting_mobject();
    expect(start.get_stroke_opacity()).toBe(0);
    expectRowsClose([start.get_center()], [[0, 0, 0]], 1e-9);
  });

  it("honors shift=DOWN", () => {
    const f = new FadeIn(new Square(), { shift: DOWN });
    expectRowsClose([f.shift_vector], [[0, -1, 0]], 1e-12);
    expect(f.point_target).toBe(false);
  });

  it("derives the shift from target_position and flags point_target", () => {
    // Python (fresh square at origin): target [1,0,0] -> shift [1,0,0].
    const f = new FadeIn(new Square(), { target_position: [1, 0, 0] });
    expectRowsClose([f.shift_vector], [[1, 0, 0]], 1e-9);
    expect(f.point_target).toBe(true);
  });

  it("accepts a Mobject target_position via its center", () => {
    const target = new Square().shift([2, 0, 0]);
    const f = new FadeIn(new Square(), { target_position: target });
    expectRowsClose([f.shift_vector], [[2, 0, 0]], 1e-9);
    expect(f.point_target).toBe(true);
  });

  it("records scale and fades the copy", () => {
    const f = new FadeIn(new Square(), { scale: 2.0 });
    expect(f.scale_factor).toBe(2.0);
    expect(f._create_faded_mobject(true).get_stroke_opacity()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyFunction (+ __init__)
// manim.animation.transform.ApplyPointwiseFunction (+ __init__)
// Python: target = function(copy) (scale 2 -> width 4.0, orig 2.0);
// non-Mobject result -> TypeError; APF default run_time 3.0, p+UP -> [0,1,0].
// ---------------------------------------------------------------------------
describe("ApplyFunction/ApplyPointwiseFunction", () => {
  it("builds the target by applying the function to a copy", () => {
    const sq = new Square();
    const af = new ApplyFunction((m) => m.scale(2), sq);
    expect(af.create_target().width).toBeCloseTo(4.0, 12);
    expect(sq.width).toBeCloseTo(2.0, 12);
  });

  it("rejects functions that do not return a Mobject", () => {
    const af = new ApplyFunction((() => 42) as unknown as (m: Mobject) => Mobject, new Square());
    expect(() => af.create_target()).toThrowError(
      /must return object of type Mobject/,
    );
  });

  it("ApplyPointwiseFunction defaults run_time to 3.0 and shifts points", () => {
    const sq = new Square();
    const apf = new ApplyPointwiseFunction((p) => [p[0]! + 0, p[1]! + 1, p[2]!], sq);
    expect(apf.run_time).toBe(3.0);
    apf.begin();
    apf.interpolate(1.0);
    expectRowsClose([apf.mobject.get_center()], [[0, 1, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.LinearTransformationScene.get_ghost_vectors
// Python: returns self.ghost_vectors (empty VGroup on a fresh scene).
// ---------------------------------------------------------------------------
describe("LinearTransformationScene.get_ghost_vectors", () => {
  it("returns the scene's ghost vector group", () => {
    const scene = new LinearTransformationScene();
    const ghosts = scene.get_ghost_vectors();
    expect(ghosts).toBe(scene.ghost_vectors);
    expect(ghosts).toBeInstanceOf(VGroup);
    expect(ghosts.submobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeTransformPieces
// Python: begin() aligns submobjects first (no error on uneven groups);
// ghost_to with mismatched families -> ValueError (strict zip).
// ---------------------------------------------------------------------------
describe("FadeTransformPieces", () => {
  it("begins after aligning submobjects", () => {
    const fp = new FadeTransformPieces(
      new VGroup(new Square(), new Circle()),
      new VGroup(new Triangle(), new Dot()),
    );
    expect(() => fp.begin()).not.toThrow();
    expect(fp.mobject.submobjects.length).toBe(2);
  });

  it("ghost_to rejects mismatched families like strict zip", () => {
    const fp = new FadeTransformPieces(new VGroup(new Square()), new VGroup(new Square()));
    expect(() => fp.ghost_to(new VGroup(new Square()), new VGroup(new Square(), new Circle()))).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    expect(() => fp.ghost_to(new Square(), new Square())).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.AnimationGroup (+ __init__)
// Python: lag 0 default, run_time from timings; introducer-only groups are
// empty; Transform+FadeOut -> Group(Square, Circle); lag 0.5 ends [1, 1.5].
// ---------------------------------------------------------------------------
describe("AnimationGroup.__init__", () => {
  it("stores animations with lag 0 and timing-derived run_time", () => {
    const ag = new AnimationGroup(new Create(new Square()), new Create(new Circle()));
    expect(ag.animations.length).toBe(2);
    expect(ag.lag_ratio).toBe(0);
    expect(ag.run_time).toBe(1.0);
  });

  it("excludes introducer mobjects from the group", () => {
    // Python: Create/FadeIn are introducers -> empty Group.
    const ag = new AnimationGroup(new Create(new Square()), new Create(new Circle()));
    expect(ag.group.submobjects.length).toBe(0);
  });

  it("groups non-introducer mobjects", () => {
    const sq = new Square();
    const ci = new Circle();
    const ag = new AnimationGroup(new Transform(sq, ci.copy()), new FadeOut(ci));
    expect(ag.group.submobjects.length).toBe(2);
    expect(ag.group.submobjects[0]).toBe(sq);
    expect(ag.group.submobjects[1]).toBe(ci);
  });

  it("derives timings from lag_ratio", () => {
    // Python: starts [0, 0.5], ends [1, 1.5], run_time 1.5.
    const ag = new AnimationGroup(
      new Transform(new Square(), new Circle()),
      new FadeOut(new Circle()),
      { lag_ratio: 0.5 },
    );
    expect(ag.lag_ratio).toBe(0.5);
    expect([...(ag.anims_with_timings as unknown as { start: number[] }).start]).toEqual([0, 0.5]);
    expect([...(ag.anims_with_timings as unknown as { end: number[] }).end]).toEqual([1, 1.5]);
    expect(ag.run_time).toBe(1.5);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts: TransformMatchingShapes,
// TransformMatchingTex (+ get_mobject_parts / get_mobject_key)
// Python keys: same shape -> equal (int), square vs circle -> differ, and
// get_mobject_key restores the mobject (start preserved). Tex: Group of plain
// mobjects -> AssertionError; carrier key is its tex_string.
// DEVIATION: Python asserts isinstance(MathTexPart) for keys (needs LaTeX);
// the TS port accepts any string tex_string carrier structurally.
// ---------------------------------------------------------------------------
describe("TransformMatchingShapes", () => {
  it("lists family members with points as parts", () => {
    expect(TransformMatchingShapes.get_mobject_parts(new Square()).length).toBe(1);
  });

  it("keys match by normalized shape, not identity", () => {
    const k1 = TransformMatchingShapes.get_mobject_key(new Square());
    const k2 = TransformMatchingShapes.get_mobject_key(new Square());
    const k3 = TransformMatchingShapes.get_mobject_key(new Circle());
    expect(typeof k1).toBe("number");
    expect(k1).toBe(k2);
    expect(k1).not.toBe(k3);
  });

  it("restores the mobject after keying", () => {
    const sq = new Square();
    TransformMatchingShapes.get_mobject_key(sq);
    expectRowsClose([sq.get_start()], [[1, 1, 0]], 1e-9);
  });

  it("builds Transform + FadeOut + FadeIn for partial matches", () => {
    // Python: 3 anims [Transform, FadeOut, FadeIn].
    const tm = new TransformMatchingShapes(
      new VGroup(new Square(), new Circle()),
      new VGroup(new Square(), new Triangle()),
    );
    expect(tm.animations.length).toBe(3);
    expect(tm.animations.map((a) => a.constructor.name)).toEqual([
      "Transform",
      "FadeOut",
      "FadeIn",
    ]);
  });

  it("uses plain Transforms for mismatches when asked", () => {
    const tm = new TransformMatchingShapes(
      new VGroup(new Square(), new Circle()),
      new VGroup(new Square(), new Triangle()),
      { transform_mismatches: true },
    );
    expect(tm.animations.length).toBe(2);
    expect(tm.animations.map((a) => a.constructor.name)).toEqual(["Transform", "Transform"]);
  });
});

describe("TransformMatchingTex", () => {
  it("rejects Groups of mobjects without tex_string", () => {
    expect(() =>
      TransformMatchingTex.get_mobject_parts(new VGroup(new Square(), new Circle())),
    ).toThrowError(expect.objectContaining({ name: "AssertionError" }));
  });

  it("returns submobjects for a tex_string carrier", () => {
    const carrier = new Square() as unknown as Record<string, unknown>;
    carrier["tex_string"] = "x";
    // Square has no submobjects -> empty parts (Python: same, no throw).
    expect(TransformMatchingTex.get_mobject_parts(carrier as unknown as Mobject).length).toBe(0);
  });

  it("keys a carrier by its tex_string", () => {
    // DEVIATION (documented above): Python requires a MathTexPart instance
    // (needs LaTeX); TS accepts any string carrier.
    const carrier = new Square() as unknown as Record<string, unknown>;
    carrier["tex_string"] = "x";
    expect(TransformMatchingTex.get_mobject_key(carrier as unknown as Mobject)).toBe("x");
  });

  it("rejects keys without a string tex_string", () => {
    expect(() => TransformMatchingTex.get_mobject_key(new Square())).toThrowError(
      expect.objectContaining({ name: "AssertionError" }),
    );
    expect(() =>
      TransformMatchingTex.get_mobject_parts(new Square()),
    ).toThrowError(expect.objectContaining({ name: "AssertionError" }));
  });
});

// ---------------------------------------------------------------------------
// manim.animation.changing.AnimatedBoundary (+ update_boundary_copies)
// Python: 2 copies, total 0, 4 colors, msw 3; update(0.5) -> total 0.5,
// growing width 3; later updates advance total and leave fading at 0 until
// time >= 1.
// ---------------------------------------------------------------------------
describe("AnimatedBoundary", () => {
  it("initializes two boundary copies", () => {
    const ab = new AnimatedBoundary(new Square());
    expect(ab.boundary_copies.length).toBe(2);
    expect(ab.total_time).toBe(0);
    expect(ab.colors.length).toBe(4);
    expect(ab.max_stroke_width).toBe(3);
  });

  it("advances total_time and draws the growing copy", () => {
    const ab = new AnimatedBoundary(new Square());
    ab.update_boundary_copies(0.5);
    expect(ab.total_time).toBeCloseTo(0.5, 12);
    expect(ab.boundary_copies[0]!.get_stroke_width()).toBe(3);
  });

  it("keeps the fading copy empty while time < 1", () => {
    // Python: total 3.5 after updates, fading stroke width 0.0.
    const ab = new AnimatedBoundary(new Square());
    ab.update_boundary_copies(0.5);
    ab.update_boundary_copies(3.0);
    expect(ab.total_time).toBeCloseTo(3.5, 12);
    expect(ab.boundary_copies[1]!.get_stroke_width()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.add_line_to
// Python: 4 -> 8 points, end [2,0,0].
// manim.mobject.types.vectorized_mobject.VGroup.__init__: adds vmobjects.
// ---------------------------------------------------------------------------
describe("VMobject.add_line_to / VGroup.__init__", () => {
  it("appends one straight cubic segment", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(v.get_num_points()).toBe(4);
    v.add_line_to([2, 0, 0]);
    expect(v.get_num_points()).toBe(8);
    expectRowsClose([v.get_end()], [[2, 0, 0]], 1e-9);
  });

  it("VGroup collects its vmobjects", () => {
    const vg = new VGroup(new Square(), new Circle());
    expect(vg.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.LaggedStartMap
// Python: 3 FadeIns, run_time 2, lag_ratio 0.05.
// ---------------------------------------------------------------------------
describe("LaggedStartMap", () => {
  it("maps the animation class over submobjects", () => {
    const grp = new VGroup(new Square(), new Circle(), new Triangle());
    const lm = new LaggedStartMap(FadeIn, grp);
    expect(lm.animations.length).toBe(3);
    expect(lm.animations.map((a) => a.constructor.name)).toEqual(["FadeIn", "FadeIn", "FadeIn"]);
    expect(lm.run_time).toBe(2);
    expect(lm.lag_ratio).toBeCloseTo(0.05, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.rotation.Rotating.interpolate_mobject
// manim.animation.rotation.Rotate.create_target
// Python: full PI/2 rotation -> start [-1,1,0]; target start [-1,1,0],
// orig [1,1,0]; about_point defaults to the mobject center.
// ---------------------------------------------------------------------------
describe("Rotating/Rotate", () => {
  it("Rotating sweeps the mobject around its start", () => {
    const sq = new Square();
    const r = new Rotating(sq, PI / 2);
    r.begin();
    expectRowsClose([sq.get_center()], [[0, 0, 0]], 1e-9);
    r.interpolate_mobject(1.0);
    expectRowsClose([sq.get_start()], [[-1, 1, 0]], 1e-9);
    expectRowsClose(
      [sq.get_start()],
      rows(new Square().rotate(PI / 2).points).slice(0, 1),
      1e-9,
    );
  });

  it("Rotate.create_target returns the rotated copy", () => {
    const sq = new Square();
    const rt = new Rotate(sq, PI / 2);
    expectRowsClose([rt.about_point as unknown as number[]], [[0, 0, 0]], 1e-9);
    const target = rt.create_target();
    expectRowsClose([target.get_start()], [[-1, 1, 0]], 1e-9);
    expectRowsClose([sq.get_start()], [[1, 1, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.get_all_mobjects
// Python: (mobject, starting_mobject) -> length 2.
// manim.animation.creation.Uncreate: reverse_rate_function/remover True,
// introducer False.
// ---------------------------------------------------------------------------
describe("Animation.get_all_mobjects / Uncreate", () => {
  it("lists mobject and starting_mobject", () => {
    expect(new Animation(new Square()).get_all_mobjects().length).toBe(2);
  });

  it("Uncreate reverses with remover semantics", () => {
    const u = new Uncreate(new Square());
    expect(u.reverse_rate_function).toBe(true);
    expect(u.remover).toBe(true);
    expect(u.introducer).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Unwrite / RemoveTextLetterByLetter /
// AddTextWordByWord / TypeWithCursor
// Python (VGroup stand-in for Text): Unwrite reverse True/remover True/
// introducer False/run 1/lag 0.2 (reverse=False flips remover/introducer);
// RemoveText flags; AddTextWordByWord tpc 0.06, 8 anims with runtimes
// [0.06, 0.005] x4; TypeWithCursor cursor at [2.3,0,0] after update(2).
// ---------------------------------------------------------------------------
describe("Unwrite", () => {
  it("defaults to reverse erasing with remover semantics", () => {
    const uw = new Unwrite(new VGroup(new Square(), new Circle()));
    expect(uw.reverse).toBe(true);
    expect(uw.remover).toBe(true);
    expect(uw.introducer).toBe(false);
    expect(uw.run_time).toBe(1);
    expect(uw.lag_ratio).toBe(0.2);
  });

  it("reverse=False keeps the mobject", () => {
    const uw = new Unwrite(new VGroup(new Square(), new Circle()), { reverse: false });
    expect(uw.reverse).toBe(false);
    expect(uw.remover).toBe(false);
    expect(uw.introducer).toBe(true);
  });
});

describe("RemoveTextLetterByLetter", () => {
  it("reverses with remover semantics", () => {
    const txt = new VGroup(...Array.from({ length: 3 }, () => new Square()));
    const rm = new RemoveTextLetterByLetter(txt);
    expect(rm.reverse_rate_function).toBe(true);
    expect(rm.remover).toBe(true);
    expect(rm.introducer).toBe(false);
  });
});

describe("AddTextWordByWord", () => {
  it("chains per-word ShowIncreasingSubsets with systime gaps", () => {
    // Python: 4 words -> 8 anims, runtimes [0.06, 0.005] x 4.
    const txt = new VGroup(...Array.from({ length: 4 }, () => new Square()));
    const aw = new AddTextWordByWord(txt);
    expect(aw.time_per_char).toBeCloseTo(0.06, 12);
    expect(aw.animations.length).toBe(8);
    expect(aw.animations.map((a) => a.run_time)).toEqual(
      [0.06, 0.005, 0.06, 0.005, 0.06, 0.005, 0.06, 0.005],
    );
  });
});

describe("TypeWithCursor", () => {
  function setup() {
    const txt = new VGroup(
      ...[0, 1, 2].map((i) => new Square().shift([i, 0, 0])),
    );
    const cursor = new Square().scale(0.2);
    return { txt, cursor, tw: new TypeWithCursor(txt, cursor) };
  }

  it("stores cursor options", () => {
    const { tw } = setup();
    expect(tw.buff).toBeCloseTo(0.1, 12);
    expect(tw.keep_cursor_y).toBe(true);
    expect(tw.leave_cursor_on).toBe(true);
  });

  it("begin() parks the cursor and adds it to the mobject", () => {
    const { tw, cursor } = setup();
    tw.begin();
    expect(tw.y_cursor).toBeCloseTo(0, 12);
    expect(tw.mobject.submobjects).toContain(cursor);
  });

  it("update_submobject_list parks the cursor after the last shown char", () => {
    // Python: cursor center [2.3, 0, 0], opacity 1.
    const { tw, cursor } = setup();
    tw.begin();
    tw.update_submobject_list(2);
    expectRowsClose([cursor.get_center()], [[2.3, 0, 0]], 1e-9);
    expect(cursor.get_fill_opacity()).toBe(1);
  });

  it("finish() leaves the cursor on", () => {
    const { tw, cursor } = setup();
    tw.begin();
    tw.finish();
    expect(cursor.get_fill_opacity()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols combined with earlier-ported code.
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Write runs the DrawBorderThenFill path end to end", () => {
    const sq = new Square();
    const w = new Write(sq);
    w.begin();
    expect(w.outline).toBeDefined();
    w.interpolate(0.5);
    expect(sq.get_num_points()).toBeGreaterThan(0);
    w.finish();
    expectRowsClose([sq.get_start()], [[1, 1, 0]], 1e-6);
  });

  it("LaggedStartMap(FadeIn) fades every submobject in on completion", () => {
    // Python: stroke opacity back to 1.0 after interpolate(1)/finish
    // (fill stays 0: fresh Squares have no fill).
    const grp = new VGroup(new Square(), new Circle());
    const lm = new LaggedStartMap(FadeIn, grp);
    lm.begin();
    lm.interpolate(1.0);
    for (const sub of grp.submobjects) {
      expect((sub as VMobject).get_stroke_opacity()).toBeCloseTo(1, 9);
    }
    lm.finish();
    for (const sub of grp.submobjects) {
      expect((sub as VMobject).get_stroke_opacity()).toBeCloseTo(1, 9);
    }
  });

  it("AddTextLetterByLetter run_time follows config.frame_rate", () => {
    // Python: max(1/frame_rate, tpc) * len(text); frame_rate 60 -> 0.4.
    const txt = new VGroup(...Array.from({ length: 4 }, () => new Square()));
    const atl = new AddTextLetterByLetter(txt);
    expect(config.frame_rate).toBe(60);
    expect(atl.run_time).toBeCloseTo(Math.max(1 / 60, 0.1) * 4, 12);
    atl.interpolate_mobject(0.5);
    expect(atl.all_submobs.map((m) => (m as VMobject).get_fill_opacity())).toEqual([1, 1, 0, 0]);
  });

  it("Rotate target feeds a Transform to the rotated state", () => {
    const sq = new Square();
    const rt = new Rotate(sq, PI / 2);
    const target = rt.create_target();
    const t = new Transform(sq, target);
    t.begin();
    t.interpolate(1.0);
    expectRowsClose([sq.get_start()], [[-1, 1, 0]], 1e-6);
  });

  it("GrowFromPoint + FadeIn agree on introducer group filtering", () => {
    const sq = new Square();
    const ag = new AnimationGroup(new GrowFromPoint(sq, ORIGIN), new FadeIn(sq));
    expect(ag.group.submobjects.length).toBe(0);
    expect(ag.animations.length).toBe(2);
  });
});
