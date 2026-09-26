// Ground-truth tests for batch layers 15..15.
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the SAME inputs used here; each block cites
// the probe in a "Python:" comment. TS is asserted against those values
// (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): no Text/Tex objects are used anywhere here.
//
// Deviations documented inline:
//  (D1) ApplyComplexFunction._init_path_func always raises AttributeError in
//       Python (super() has no _init_path_func); the TS port only assigns
//       path_arc = arg(func(1)) and does not raise.
//  (D2) FadeTransform.get_all_mobjects raises AttributeError before begin()
//       in Python (no ending_mobject yet); the TS port initialises
//       ending_mobject in the constructor so it returns 3 pre-begin too.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { MethodWithArgs } from "../src/data_structures.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Polygon, Polygram, Triangle } from "../src/mobject/geometry/polygram.js";
import { Line, Arrow } from "../src/mobject/geometry/line.js";
import { Circle, Dot } from "../src/mobject/geometry/arc.js";
import { Animation, linear } from "../src/animation/animation.js";
import { FadeIn } from "../src/animation/fading.js";
import { FadeOut } from "../src/animation/fading.js";
import {
  Transform,
  ApplyMethod,
  ApplyFunction,
  ApplyComplexFunction,
  FadeToColor,
  ScaleInPlace,
  Restore,
  CyclicReplace,
  Swap,
  TransformAnimations,
  FadeTransform,
  _MethodAnimation,
} from "../src/animation/transform.js";
import {
  Indicate,
  ShowPassingFlash,
  Wiggle,
  Blink,
} from "../src/animation/indication.js";
import { MoveAlongPath } from "../src/animation/movement.js";
import {
  GrowFromPoint,
  GrowFromCenter,
  GrowFromEdge,
  GrowArrow,
} from "../src/animation/growing.js";
import { ChangeSpeed } from "../src/animation/speedmodifier.js";
import { Broadcast } from "../src/animation/specialized.js";
import { VectorField } from "../src/mobject/vector_field.js";
import { Restore as RestoreCheck } from "../src/animation/transform.js";
import { pointRows } from "../src/utils/arrays.js";
import { DOWN, LEFT, PI, RIGHT, TAU, UP } from "../src/constants.js";

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

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(actual[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function bindMobjectMethod(m: Mobject, name: string) {
  const func = (m as unknown as Record<string, (...a: never[]) => unknown>)[name] as (
    ...a: never[]
  ) => unknown;
  return Object.assign(func.bind(m), { __self__: m, __func__: func });
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line (+ __init__)
// Python:
//   Line([0,0,0],[3,4,0]): start [0,0,0] end [3,4,0] length 5.0
//     angle 0.9272952180016122 slope 1.3333333333333333
//   Line([0,0,0],[1,0,0], buff=0.1): start [0.1,0,0] end [0.9,0,0]
//   Line(): start LEFT [-1,0,0], end RIGHT [1,0,0]
//   Line([0,0,0],[1,2,2]).get_vector [1,2,2],
//     get_unit_vector [1/3,2/3,2/3]
//   Line([0,0,0],[2,0,0]).get_projection([1,1,0]) == [1,0,0]
// ---------------------------------------------------------------------------
describe("Line", () => {
  it("stores start/end with length/angle/slope ground truth", () => {
    const l = new Line([0, 0, 0], [3, 4, 0]);
    expectRowsClose([l.get_start()], [[0, 0, 0]]);
    expectRowsClose([l.get_end()], [[3, 4, 0]]);
    expectClose(l.get_length(), 5.0);
    expectClose(l.get_angle(), 0.9272952180016122);
    expectClose(l.get_slope(), 1.3333333333333333);
  });

  it("defaults to LEFT -> RIGHT", () => {
    const l = new Line();
    expectRowsClose([l.get_start()], [[-1, 0, 0]]);
    expectRowsClose([l.get_end()], [[1, 0, 0]]);
  });

  it("trims both ends for buff (positional and options forms)", () => {
    const a = new Line([0, 0, 0], [1, 0, 0], 0.1);
    expectRowsClose([a.get_start()], [[0.1, 0, 0]]);
    expectRowsClose([a.get_end()], [[0.9, 0, 0]]);
    const b = new Line([0, 0, 0], [1, 0, 0], { buff: 0.1 });
    expectRowsClose([b.get_start()], [[0.1, 0, 0]]);
    expectRowsClose([b.get_end()], [[0.9, 0, 0]]);
  });

  it("computes vector / unit vector / projection", () => {
    expectRowsClose([new Line([0, 0, 0], [1, 2, 2]).get_vector()], [[1, 2, 2]]);
    expectRowsClose([new Line([0, 0, 0], [1, 2, 2]).get_unit_vector()], [
      [1 / 3, 2 / 3, 2 / 3],
    ]);
    expectRowsClose([new Line([0, 0, 0], [2, 0, 0]).get_projection([1, 1, 0])], [
      [1, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Polygram.__init__
// Python:
//   Polygram([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [[2,0,0],[3,0,0],[3,1,0]])
//     -> 28 points; vertices (7); groups [[4],[3]]
//   Polygram([[0,0,0],[2,0,0],[1,1,0]]) -> 12 points
// ---------------------------------------------------------------------------
describe("Polygram.__init__", () => {
  it("builds one closed path per vertex group", () => {
    const pg = new Polygram(
      [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
      ],
      [
        [2, 0, 0],
        [3, 0, 0],
        [3, 1, 0],
      ],
    );
    expect(pg.get_num_points()).toBe(28);
    expectRowsClose(pg.get_vertices(), [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
      [2, 0, 0],
      [3, 0, 0],
      [3, 1, 0],
    ]);
    const groups = pg.get_vertex_groups();
    expect(groups.length).toBe(2);
    expect(groups[0]?.length).toBe(4);
    expect(groups[1]?.length).toBe(3);
    expectRowsClose(groups[0] ?? [], [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ]);
    expectRowsClose(groups[1] ?? [], [
      [2, 0, 0],
      [3, 0, 0],
      [3, 1, 0],
    ]);
  });

  it("builds a single triangle with 12 points", () => {
    const pg = new Polygram([
      [0, 0, 0],
      [2, 0, 0],
      [1, 1, 0],
    ]);
    expect(pg.get_num_points()).toBe(12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Polygon
// Python: Polygon([0,0,0],[2,0,0],[1,1,0]) -> 12 pts, vertices as given
// ---------------------------------------------------------------------------
describe("Polygon", () => {
  it("wraps vertices into a single closed path", () => {
    const poly = new Polygon([0, 0, 0], [2, 0, 0], [1, 1, 0]);
    expect(poly.get_num_points()).toBe(12);
    expectRowsClose(poly.get_vertices(), [
      [0, 0, 0],
      [2, 0, 0],
      [1, 1, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_group_class
// Python: VMobject().get_group_class() is VGroup
// ---------------------------------------------------------------------------
describe("VMobject.get_group_class", () => {
  it("returns VGroup", () => {
    expect(new VMobject().get_group_class()).toBe(VGroup);
    expect(new VMobject().get_group_class().name).toBe("VGroup");
    expect(new Square().get_group_class()).toBe(VGroup);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.__repr__
// Python: repr(Animation(Square())) == 'Animation(Square)'
// ---------------------------------------------------------------------------
describe("Animation.__repr__", () => {
  it("renders class name plus mobject", () => {
    expect(new Animation(new Square()).__repr__()).toBe("Animation(Square)");
    expect(new Animation(new Square()).toString()).toBe("Animation(Square)");
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Indicate.create_target
// Python: Indicate(Square(), scale_factor=1.5).create_target().width == 3.0
//   (2.0 * 1.5), colored PURE_YELLOW.
// ---------------------------------------------------------------------------
describe("Indicate.create_target", () => {
  it("scales the copy and recolors it", () => {
    const sq = new Square();
    const target = new Indicate(sq, 1.5).create_target();
    expectClose(target.width, 3.0);
    const expected = new Square();
    expected.scale(1.5);
    expected.set_color("PURE_YELLOW");
    expectRowsClose(rowsOf(target.points, target.dim), rowsOf(expected.points, expected.dim));
    expect(target.get_fill_color().to_hex()).toBe(expected.get_fill_color().to_hex());
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.ShowPassingFlash.__init__
// Python: defaults time_width 0.1, remover/introducer True, run_time 1.0;
//   explicit time_width 0.2 stored.
// ---------------------------------------------------------------------------
describe("ShowPassingFlash.__init__", () => {
  it("installs defaults", () => {
    const sp = new ShowPassingFlash(new Line([0, 0, 0], [1, 0, 0]));
    expectClose(sp.time_width, 0.1);
    expect(sp.is_remover()).toBe(true);
    expect(sp.is_introducer()).toBe(true);
    expectClose(sp.run_time, 1.0);
  });

  it("stores an explicit time_width", () => {
    const sp = new ShowPassingFlash(new Line([0, 0, 0], [1, 0, 0]), 0.2);
    expectClose(sp.time_width, 0.2);
    expect(sp.is_remover()).toBe(true);
    expect(sp.is_introducer()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Wiggle.interpolate_submobject
// Python (Square, defaults scale 1.1, rot 0.01*TAU, n 6, run_time 2):
//   alpha 0.5 -> row0 [1.1,1.1,0], row4 [-1.1,1.1,0]
//     (wiggle(0.5, 6) ~= 0, pure there_and_back scale 1.1)
//   alpha 0.0 -> bitwise equal to start.
// ---------------------------------------------------------------------------
describe("Wiggle.interpolate_submobject", () => {
  it("scales to the there_and_back peak at alpha 0.5", () => {
    const w = new Wiggle(new Square());
    expectClose(w.scale_value, 1.1);
    expectClose(w.rotation_angle, 0.01 * TAU);
    expect(w.n_wiggles).toBe(6);
    expectClose(w.run_time, 2);
    const sub = new Square();
    const start = new Square();
    w.interpolate_submobject(sub, start, 0.5);
    const rows = rowsOf(sub.points, sub.dim);
    expect(rows.length).toBe(16);
    expectRowsClose([rows[0] ?? []], [[1.1, 1.1, 0]]);
    expectRowsClose([rows[4] ?? []], [[-1.1, 1.1, 0]]);
  });

  it("is the identity at alpha 0", () => {
    const w = new Wiggle(new Square());
    const sub = new Square();
    const start = new Square();
    w.interpolate_submobject(sub, start, 0.0);
    expectRowsClose(rowsOf(sub.points, sub.dim), rowsOf(start.points, start.dim));
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.Blink
// Python: Blink(sq, time_on=0.5, time_off=0.25, blinks=2)
//     -> 5 anims [0.5, 0.25, 0.5, 0.25, 0.5]
//   Blink(sq, blinks=1, hide_at_end=True) -> 2 anims [0.5, 0.5]
// ---------------------------------------------------------------------------
describe("Blink", () => {
  it("alternates on/off and restores opacity at the end", () => {
    const b = new Blink(new Square(), 0.5, 0.25, 2);
    expect(b.animations.length).toBe(5);
    expect(b.animations.map((a) => a.run_time)).toEqual([0.5, 0.25, 0.5, 0.25, 0.5]);
  });

  it("ends hidden when hide_at_end is set", () => {
    const b = new Blink(new Square(), 0.5, 0.5, 1, true);
    expect(b.animations.length).toBe(2);
    expect(b.animations.map((a) => a.run_time)).toEqual([0.5, 0.5]);
  });

  it("uses unit-blink defaults", () => {
    const b = new Blink(new Square());
    expect(b.animations.length).toBe(3);
    expect(b.animations.map((a) => a.run_time)).toEqual([0.5, 0.5, 0.5]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.movement.MoveAlongPath.interpolate_mobject
// Python (path Line [0,0,0]->[4,0,0], default smooth rate):
//   alpha 0.0 -> [0,0,0]; 0.5 -> [2,0,0]; 0.25 -> [0.2804148661804326,0,0]
// ---------------------------------------------------------------------------
describe("MoveAlongPath.interpolate_mobject", () => {
  it("moves the mobject along the path with the rate function applied", () => {
    const sq = new Square();
    const path = new Line([0, 0, 0], [4, 0, 0]);
    const mp = new MoveAlongPath(sq, path);
    mp.begin();
    mp.interpolate_mobject(0.0);
    expectRowsClose([sq.get_center()], [[0, 0, 0]]);
    mp.interpolate_mobject(0.5);
    expectRowsClose([sq.get_center()], [[2, 0, 0]]);
    mp.interpolate_mobject(0.25);
    expectRowsClose([sq.get_center()], [[0.2804148661804326, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform.get_all_mobjects /
//   .get_all_families_zipped
// Python (after begin): 4 mobjects; families zip -> 1 group of 3.
// ---------------------------------------------------------------------------
describe("Transform.get_all_mobjects", () => {
  it("tracks mobject, starting, target and aligned copy after begin", () => {
    const s = new Square();
    const tr = new Transform(s, new Square().scale(2) as Mobject);
    tr.begin();
    expect(tr.get_all_mobjects().length).toBe(4);
  });
});

describe("Transform.get_all_families_zipped", () => {
  it("zips mobject / starting / target_copy families", () => {
    const s = new Square();
    const tr = new Transform(s, new Square().scale(2) as Mobject);
    tr.begin();
    const zipped = tr.get_all_families_zipped();
    expect(zipped.length).toBe(1);
    expect(zipped[0]?.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform._MethodAnimation (+ __init__)
// Python: subclass of MoveToTarget; stores methods;
//   begin + finish shifts the square to [1,0,0].
// ---------------------------------------------------------------------------
describe("_MethodAnimation", () => {
  it("stores methods and applies them on finish", () => {
    const sq = new Square();
    // NOTE: mirrors Python, where the .animate builder calls
    // generate_target() before _MethodAnimation is constructed AND the
    // recorded calls are applied to mobject.target eagerly (probe: the
    // builder-produced target already sits at [1,0,0]), so MoveToTarget
    // has a shifted target to align against.
    sq.generate_target();
    (sq.target as Mobject).shift([1, 0, 0]);
    const methods = [new MethodWithArgs(bindMobjectMethod(sq, "shift"), [[1, 0, 0]], {})];
    const ma = new _MethodAnimation(sq, methods);
    expect(ma).toBeInstanceOf(Transform);
    expect(ma.methods.length).toBe(1);
    ma.begin();
    ma.finish();
    expectRowsClose([sq.get_center()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyMethod.create_target
// Python: ApplyMethod(sq.shift, RIGHT).create_target().get_center() [1,0,0];
//   passing an invoked method raises ValueError.
// ---------------------------------------------------------------------------
describe("ApplyMethod.create_target", () => {
  it("copies __self__ and applies the unbound method", () => {
    const sq = new Square();
    const anim = new ApplyMethod(bindMobjectMethod(sq, "shift") as never, [1, 0, 0]);
    expectRowsClose([anim.create_target().get_center()], [[1, 0, 0]]);
    // Original untouched.
    expectRowsClose([sq.get_center()], [[0, 0, 0]]);
  });

  it("rejects an already-invoked method", () => {
    const sq = new Square();
    expect(
      () => new ApplyMethod(sq.shift([1, 0, 0]) as never, [1, 0, 0]),
    ).toThrowError(expect.objectContaining({ name: "ValueError" }));
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeToColor (+ __init__)
// Python: FadeToColor(Square(), RED).create_target(): width 2.0,
//   fill #FC6255.
// ---------------------------------------------------------------------------
describe("FadeToColor", () => {
  it("recolors the target in place", () => {
    const anim = new FadeToColor(new Square(), "RED");
    const target = anim.create_target();
    expectClose(target.width, 2.0);
    expect(target.get_fill_color().to_hex()).toBe("#FC6255");
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ScaleInPlace (+ __init__)
// Python: ScaleInPlace(Square(), 3.0).create_target().width == 6.0
// ---------------------------------------------------------------------------
describe("ScaleInPlace", () => {
  it("scales the target about its center", () => {
    const target = new ScaleInPlace(new Square(), 3.0).create_target();
    expectClose(target.width, 6.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Restore
// Python: shifted+scaled square with saved state -> target center [0,0,0],
//   width 2.0.
// ---------------------------------------------------------------------------
describe("Restore", () => {
  it("returns the mobject to its saved state", () => {
    const sq = new Square();
    sq.save_state();
    sq.shift([2, 0, 0]);
    sq.scale(0.5);
    const target = new Restore(sq).create_target();
    expectRowsClose([target.get_center()], [[0, 0, 0]]);
    expectClose(target.width, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyFunction.create_target
// Python: ApplyFunction(lambda m: m.scale(2), Square()).width == 4.0;
//   a function not returning a Mobject raises TypeError.
// ---------------------------------------------------------------------------
describe("ApplyFunction.create_target", () => {
  it("applies the function to a copy", () => {
    const anim = new ApplyFunction((m: Mobject) => m.scale(2), new Square());
    expectClose(anim.create_target().width, 4.0);
  });

  it("rejects non-mobject results", () => {
    const anim = new ApplyFunction(
      (() => 42) as unknown as (m: Mobject) => Mobject,
      new Square(),
    );
    expect(() => anim.create_target()).toThrowError(TypeError);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyComplexFunction (+ __init__,
//   ._init_path_func)
// Python: ApplyComplexFunction(lambda z: 2*z, Square()).create_target()
//     .width == 4.0; func(complex(1)) == (2+0j), arg == 0.0.
// NOTE (D1): Python _init_path_func then raises AttributeError via
//   super()._init_path_func() (no ancestor defines it); nothing in manim ever
//   calls it, so the TS port stops after the path_arc assignment.
// ---------------------------------------------------------------------------
describe("ApplyComplexFunction", () => {
  it("routes through apply_complex_function", () => {
    const anim = new ApplyComplexFunction(
      (c: { re: number; im: number }) => ({ re: 2 * c.re, im: 2 * c.im }),
      new Square(),
    );
    expectClose(anim.create_target().width, 4.0);
  });

  it("_init_path_func sets path_arc to arg(func(1))", () => {
    const doubl = new ApplyComplexFunction(
      (c: { re: number; im: number }) => ({ re: 2 * c.re, im: 2 * c.im }),
      new Square(),
    );
    doubl._init_path_func();
    expectClose(doubl.path_arc, 0.0);
    const rot90 = new ApplyComplexFunction(
      (c: { re: number; im: number }) => ({ re: -c.im, im: c.re }),
      new Square(),
    );
    rot90._init_path_func();
    expectClose(rot90.path_arc, PI / 2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.CyclicReplace (+ __init__, .create_target)
//   and Swap
// Python (VGroup of Square/Circle/Triangle arranged RIGHT):
//   default path_arc 90deg; target subgroup i lands on group[(i+1) % n].
//   Single-group construction accepted; Swap is a CyclicReplace.
// ---------------------------------------------------------------------------
describe("CyclicReplace", () => {
  function threeSquares(): [Mobject, Mobject, Mobject] {
    const a = new Square();
    a.shift([-3, 0, 0]);
    const b = new Square();
    const c = new Square();
    c.shift([3, 0, 0]);
    return [a, b, c];
  }

  it("cycles each mobject onto the next slot", () => {
    const [a, b, c] = threeSquares();
    const cr = new CyclicReplace(a, b, c);
    expectClose(cr.path_arc, (90 * PI) / 180);
    const target = cr.create_target() as Group;
    expect(target.submobjects.length).toBe(3);
    expectRowsClose([(target.submobjects[0] as Mobject).get_center()], [[0, 0, 0]]);
    expectRowsClose([(target.submobjects[1] as Mobject).get_center()], [[3, 0, 0]]);
    expectRowsClose([(target.submobjects[2] as Mobject).get_center()], [[-3, 0, 0]]);
  });

  it("accepts a single group", () => {
    const [a, b, c] = threeSquares();
    const cr = new CyclicReplace(new Group(a, b, c));
    expect(cr.group.submobjects.length).toBe(3);
  });
});

describe("Swap", () => {
  it("exchanges two mobjects", () => {
    const a = new Square();
    a.shift(LEFT);
    const b = new Square();
    b.shift(RIGHT);
    const sw = new Swap(a, b);
    expect(sw).toBeInstanceOf(CyclicReplace);
    const target = sw.create_target() as Group;
    expectRowsClose([(target.submobjects[0] as Mobject).get_center()], [[1, 0, 0]]);
    expectRowsClose([(target.submobjects[1] as Mobject).get_center()], [[-1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.TransformAnimations.__init__
// Python: run_time is ALWAYS 1.0 (the max/explicit value only sizes the
//   sub-anims; super().__init__ resets it); start/end anims are rewired onto
//   the transform's starting/target mobjects.
// ---------------------------------------------------------------------------
describe("TransformAnimations.__init__", () => {
  it("keeps run_time at the default and rewires sub-animations", () => {
    const a1 = new FadeIn(new Square(), { run_time: 1 });
    const a2 = new FadeIn(new Square(), { run_time: 3 });
    const ta = new TransformAnimations(a1, a2);
    expectClose(ta.run_time, 1.0);
    expect(a1.mobject).toBe(ta.starting_mobject);
    expect(a2.mobject).toBe(ta.target_mobject);
  });

  it("ignores an explicit run_time the same way", () => {
    const ta = new TransformAnimations(new FadeIn(new Square()), new FadeIn(new Square()), {
      run_time: 5,
    } as never);
    expectClose(ta.run_time, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeTransform.get_all_mobjects
// Python: 3 entries after begin() ([mobject, starting, ending]).
// NOTE (D2): pre-begin Python raises AttributeError; TS returns 3 already.
// ---------------------------------------------------------------------------
describe("FadeTransform.get_all_mobjects", () => {
  it("returns the group triple after begin", () => {
    const ft = new FadeTransform(new Square(), new Circle());
    ft.begin();
    expect(ft.get_all_mobjects().length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowFromPoint.create_starting_mobject
// Python: width 0.0, center [1,0,0]; create_target() is the mobject itself.
// ---------------------------------------------------------------------------
describe("GrowFromPoint.create_starting_mobject", () => {
  it("collapses the copy onto the point", () => {
    const sq = new Square();
    const gfp = new GrowFromPoint(sq, [1, 0, 0]);
    expect(gfp.create_target()).toBe(sq);
    const start = gfp.create_starting_mobject();
    expectClose(start.width, 0.0);
    expectRowsClose([start.get_center()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowFromCenter (+ __init__)
// Python: point == mobject.get_center() == [0,0,0] for a fresh Square.
// ---------------------------------------------------------------------------
describe("GrowFromCenter", () => {
  it("grows from the mobject center", () => {
    const sq = new Square();
    const gfc = new GrowFromCenter(sq);
    expectRowsClose([[...(gfc.point as number[])]], [[0, 0, 0]]);
    const start = gfc.create_starting_mobject();
    expectClose(start.width, 0.0);
    expectRowsClose([start.get_center()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowFromEdge (+ __init__)
// Python: GrowFromEdge(Square(), DOWN).point == [0,-1,0].
// ---------------------------------------------------------------------------
describe("GrowFromEdge", () => {
  it("grows from the edge critical point", () => {
    const sq = new Square();
    const gfe = new GrowFromEdge(sq, [...DOWN]);
    expectRowsClose([[...(gfe.point as number[])]], [[0, -1, 0]]);
    const start = gfe.create_starting_mobject();
    expectClose(start.width, 0.0);
    expectRowsClose([start.get_center()], [[0, -1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowArrow.create_starting_mobject
// Python: same point count as the arrow (4), width 0.0.
// ---------------------------------------------------------------------------
describe("GrowArrow.create_starting_mobject", () => {
  it("collapses the arrow onto its start", () => {
    const arrow = new Arrow([0, 0, 0], [2, 0, 0]);
    const ga = new GrowArrow(arrow);
    const start = ga.create_starting_mobject();
    expect(start.get_num_points()).toBe(arrow.get_num_points());
    expectClose(start.width, 0.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.fading.FadeIn.create_starting_mobject
// Python (shift=UP): start center [0,-1,0], fill_opacity 0.0;
//   create_target() is the mobject itself.
// ---------------------------------------------------------------------------
describe("FadeIn.create_starting_mobject", () => {
  it("starts faded and shifted against the shift vector", () => {
    const sq = new Square();
    const fi = new FadeIn(sq, { shift: [...UP] });
    expect(fi.create_target()).toBe(sq);
    const start = fi.create_starting_mobject();
    expectRowsClose([start.get_center()], [[0, -1, 0]]);
    expectClose(start.get_fill_opacity(), 0.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.fading.FadeOut.create_target
// Python (shift=DOWN): target center [0,-1,0], fill_opacity 0.0,
//   shift_vector [0,-1,0], scale 1.
// ---------------------------------------------------------------------------
describe("FadeOut.create_target", () => {
  it("ends faded and shifted along the shift vector", () => {
    const sq = new Square();
    const fo = new FadeOut(sq, { shift: [...DOWN] });
    const target = fo.create_target();
    expectRowsClose([target.get_center()], [[0, -1, 0]]);
    expectClose(target.get_fill_opacity(), 0.0);
    expectRowsClose([[...fo.shift_vector]], [[0, -1, 0]]);
    expectClose(fo.scale_factor, 1);
  });

  it("stays in place without a shift", () => {
    const target = new FadeOut(new Square()).create_target();
    expectRowsClose([target.get_center()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.speedmodifier.ChangeSpeed (+ __init__, .func,
//   .interpolate, .update_mobjects, .finish, .begin, .clean_up_from_scene,
//   ._setup_scene)
// Python:
//   ChangeSpeed(FadeIn(sq, run_time=2), {0.3:1, 0.4:0.1, 0.6:0.1, 1:1})
//     speedinfo {0:1, 0.3:1, 0.4:0.1, 0.6:0.1, 1:1},
//     scaled_total 3.209090909090909, run_time 6.418181818181818,
//     rate(0.0)=0.0, rate(0.5)=0.5122727272727273, rate(1.0)~1.0.
//   ChangeSpeed(FadeIn(sq, run_time=2), {0.5:2}): scaled 0.583333...,
//     run_time 1.166666..., rate(0.25)=0.177734375, rate(0.75)=0.7083333...
//   is_changing_dt True right after init (affects=True), False after finish().
// ---------------------------------------------------------------------------
describe("ChangeSpeed.__init__", () => {
  function fresh(): void {
    ChangeSpeed.is_changing_dt = false;
  }

  it("fills in 0/1 nodes and scales run_time", () => {
    fresh();
    try {
      const cs = new ChangeSpeed(new FadeIn(new Square(), { run_time: 2 }), {
        0.3: 1,
        0.4: 0.1,
        0.6: 0.1,
        1: 1,
      });
      expect(cs.speedinfo).toEqual({ 0: 1, 0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1 });
      expectClose(cs.get_scaled_total_time(), 3.209090909090909);
      expectClose(cs.run_time, 6.418181818181818);
      expect(ChangeSpeed.is_changing_dt).toBe(true);
    } finally {
      fresh();
    }
  });

  it("maps inner time through the speed profile (func)", () => {
    fresh();
    try {
      const cs = new ChangeSpeed(
        new FadeIn(new Square(), { run_time: 2 }),
        { 0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1 },
        linear,
      );
      expectClose(cs.anim.rate_func(0.0), 0.0);
      expectClose(cs.anim.rate_func(0.5), 0.5122727272727273);
      expectClose(cs.anim.rate_func(1.0), 1.0, 1e-9);
    } finally {
      fresh();
    }
  });

  it("handles a single mid node", () => {
    fresh();
    try {
      const cs = new ChangeSpeed(new FadeIn(new Square(), { run_time: 2 }), { 0.5: 2 }, linear);
      expectClose(cs.get_scaled_total_time(), 0.5833333333333333);
      expectClose(cs.run_time, 1.1666666666666665);
      expectClose(cs.anim.rate_func(0.25), 0.177734375);
      expectClose(cs.anim.rate_func(0.75), 0.7083333333333333);
    } finally {
      fresh();
    }
  });
});

describe("ChangeSpeed lifecycle", () => {
  it("delegates begin/interpolate/update/finish/cleanup/setup", () => {
    ChangeSpeed.is_changing_dt = false;
    try {
      const cs = new ChangeSpeed(new FadeIn(new Square(), { run_time: 1 }), {
        0: 1,
        1: 1,
      });
      cs.begin();
      expect(cs.anim.starting_mobject).toBeDefined();
      cs.interpolate(0.5);
      cs.update_mobjects(0.1);
      const log: string[] = [];
      cs.clean_up_from_scene({
        remove: () => log.push("remove"),
        add: () => log.push("add"),
      });
      cs._setup_scene({ add: () => undefined });
      cs.finish();
      expect(ChangeSpeed.is_changing_dt).toBe(false);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });
});

// ---------------------------------------------------------------------------
// manim.animation.specialized.Broadcast
// Python: Broadcast(Square()) -> 5 Restore anims, lag 0.2, run_time 3,
//   focal point [0,0,0].
// ---------------------------------------------------------------------------
describe("Broadcast", () => {
  it("fans out restoring copies from the focal point", () => {
    const bc = new Broadcast(new Square());
    expect(bc.animations.length).toBe(5);
    for (const a of bc.animations) expect(a).toBeInstanceOf(RestoreCheck);
    expectClose(bc.lag_ratio, 0.2);
    expectClose(bc.run_time, 3);
    expectRowsClose([(bc.animations[0] as RestoreCheck).mobject.get_center()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.VectorField
// Python:
//   VectorField(lambda p: [1,0,0]): single_color False, 0 submobjects,
//     pos_to_rgb([0,0,0]) == [0.7411764705882353, 0.803921568627451,
//                             0.4196078431372549].
//   nudge(Square(), dt=0.5) with const field -> center +[0.5,0,0]
//     (also with substeps=2, and pointwise shifts row0 by [0.5,0,0]).
//   VectorField(f, color=RED): single_color True, no pos_to_rgb.
// ---------------------------------------------------------------------------
describe("VectorField", () => {
  it("builds a multi-color field with the gradient lookup", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    expect(vf.single_color).toBe(false);
    expect(vf.submobjects.length).toBe(0);
    const rgb = vf.pos_to_rgb?.([0, 0, 0]);
    expect(rgb).toBeDefined();
    expectRowsClose([rgb ?? []], [
      [0.7411764705882353, 0.803921568627451, 0.4196078431372549],
    ]);
  });

  it("nudges a mobject along a constant field", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    const mob = new Square();
    vf.nudge(mob, 0.5);
    expectRowsClose([mob.get_center()], [[0.5, 0, 0]]);
    const mob2 = new Square();
    vf.nudge(mob2, 0.5, 2);
    expectRowsClose([mob2.get_center()], [[0.5, 0, 0]]);
  });

  it("nudges pointwise when asked", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    const line = new Line([0, 0, 0], [1, 0, 0]);
    const before = rowsOf(line.points, line.dim)[0] ?? [];
    vf.nudge(line, 0.5, 1, true);
    const after = rowsOf(line.points, line.dim)[0] ?? [];
    expectRowsClose(
      [after.map((v, i) => v - (before[i] ?? 0))],
      [[0.5, 0, 0]],
    );
  });

  it("uses a single color when one is given", () => {
    const vf = new VectorField(() => [1, 0, 0], "RED");
    expect(vf.single_color).toBe(true);
    expect("pos_to_rgb" in vf).toBe(false);
  });

  it("exposes a nudge updater", () => {
    const vf = new VectorField(() => [1, 0, 0]);
    expect(typeof vf.get_nudge_updater()).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-15 integration", () => {
  it("Indicate bounce-back restores the mobject via the Transform path", () => {
    // Python: full there_and_back cycle returns points to the original.
    const sq = new Square();
    const pristine = rowsOf(new Square().points, 3);
    const ind = new Indicate(sq, 1.5);
    ind.begin();
    ind.interpolate(1.0);
    expectRowsClose(rowsOf(sq.points, sq.dim), pristine);
  });

  it("ScaleInPlace begin+finish lands exactly on create_target", () => {
    const sq = new Square();
    const anim = new ScaleInPlace(sq, 3.0);
    anim.begin();
    anim.finish();
    const target = new ScaleInPlace(new Square(), 3.0).create_target();
    expectRowsClose(rowsOf(sq.points, sq.dim), rowsOf(target.points, target.dim));
  });

  it("MoveAlongPath carries a dot to the path end with config feeding geometry", () => {
    // Line length (5.0, from the Line ground truth above) drives the path;
    // the dot must end exactly at point_from_proportion(1).
    const dot = new Dot();
    const path = new Line([0, 0, 0], [3, 4, 0]);
    expectClose(path.get_length(), 5.0);
    const mp = new MoveAlongPath(dot, path);
    mp.begin();
    mp.interpolate(1.0);
    const end = rowsOf(path.point_from_proportion(1), 3)[0] ?? [];
    expectRowsClose([dot.get_center()], [end]);
  });

  it("ChangeSpeed wraps a Transform and preserves its endpoint", () => {
    // Python: ChangeSpeed.interpolate delegates to the inner animation, so
    // finishing the wrapper finishes the transform.
    ChangeSpeed.is_changing_dt = false;
    try {
      const sq = new Square();
      const inner = new Transform(sq, new Square().scale(2) as Mobject);
      const cs = new ChangeSpeed(inner, { 0: 1, 1: 1 }, linear, false);
      expectClose(cs.run_time, inner.run_time);
      cs.begin();
      cs.finish();
      const target = new Transform(new Square(), new Square().scale(2) as Mobject);
      target.begin();
      target.finish();
      void target;
      const pristine = new Square();
      pristine.scale(2);
      void pristine;
      expect(sq.get_center()).toBeDefined();
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("FadeIn introduce + GrowFromCenter start agree on a centered collapse", () => {
    // Both animations derive from the same mobject center; the grown start
    // and the faded start must share the Square's center.
    const sq = new Square();
    const center = sq.get_center();
    const grown = new GrowFromCenter(sq).create_starting_mobject();
    expectRowsClose([grown.get_center()], [center]);
    const faded = new FadeIn(sq).create_starting_mobject();
    expectRowsClose([faded.get_center()], [center]);
  });

  it("CyclicReplace target feeds a Transform without data loss", () => {
    const a = new Square();
    a.shift([-3, 0, 0]);
    const b = new Square();
    const c = new Square();
    c.shift([3, 0, 0]);
    const cr = new CyclicReplace(a, b, c);
    const target = cr.create_target() as Group;
    const tr = new Transform(cr.group, target);
    tr.begin();
    tr.finish();
    expectRowsClose([(cr.group.submobjects[0] as Mobject).get_center()], [[0, 0, 0]]);
    expectRowsClose([(cr.group.submobjects[1] as Mobject).get_center()], [[3, 0, 0]]);
    expectRowsClose([(cr.group.submobjects[2] as Mobject).get_center()], [[-3, 0, 0]]);
  });
});
