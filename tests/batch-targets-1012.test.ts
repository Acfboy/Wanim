// Ground-truth tests for batch layers 14..14 (~1012 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): ChangeDecimalToValue is exercised with a structural
// DecimalNumber stand-in (a Mobject with `number` + `set_value`), because the
// real DecimalNumber requires a latex install (FileNotFoundError: 'latex') and
// DecimalNumber is a text/TeX mobject anyway. No Text/Tex objects used here.
//
// Known upstream quirks documented inline:
//  - SmoothedVectorizedHomotopy.interpolate_submobject ALWAYS raises NameError
//    in Python 0.21.0 (`VMobject` is only imported under TYPE_CHECKING); the TS
//    port implements the intended behavior (apply homotopy + make_smooth).
//  - TransformAnimations.__init__ always ends with run_time == 1.0 in Python
//    (run_time is popped from kwargs, then super().__init__ resets it to the
//    default); the TS port mirrors this.
//  - TransformAnimations.begin()/interpolate() crash in Python unless the
//    sub-anims were begun in exactly the right order (rewired plain-Mobject
//    targets); only constructor-level behavior is asserted for parity.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Triangle, Polygram } from "../src/mobject/geometry/polygram.js";
import { Circle, Dot, TipableVMobject } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import {
  Animation,
  Add,
} from "../src/animation/animation.js";
import { ApplyWave, Wiggle } from "../src/animation/indication.js";
import {
  Homotopy,
  SmoothedVectorizedHomotopy,
  ComplexHomotopy,
  PhaseFlow,
  MoveAlongPath,
} from "../src/animation/movement.js";
import { ChangeDecimalToValue } from "../src/animation/numbers.js";
import {
  Transform,
  ReplacementTransform,
  TransformFromCopy,
  ClockwiseTransform,
  CounterclockwiseTransform,
  MoveToTarget,
  CyclicReplace,
  TransformAnimations,
  FadeTransform,
} from "../src/animation/transform.js";
import { GrowFromPoint } from "../src/animation/growing.js";
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

function maxDiff(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < (a[i]?.length ?? 0); j += 1) {
      m = Math.max(m, Math.abs((a[i]?.[j] ?? 0) - (b[i]?.[j] ?? 0)));
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// ApplyWave.__init__
// Python:
//   ApplyWave(Square()).run_time == 2
//   homotopy(0,0,0,0.5) == [0, 0.2, 0]   (default UP, amplitude 0.2)
//   homotopy(1,2,3,0.0) == [1, 2, 3]     (t=0 -> no nudge)
//   ApplyWave(sq, RIGHT, 0.5, time_width=0.5, ripples=2):
//     homotopy(0.5,0,0,0.5) == [0.5, 0, 0]
// ---------------------------------------------------------------------------
describe("ApplyWave.__init__", () => {
  it("stores the default run_time and homotopy", () => {
    const aw = new ApplyWave(new Square());
    expectClose(aw.run_time, 2);
    expect(aw).toBeInstanceOf(Homotopy);
    const h = (aw as unknown as { homotopy: (x: number, y: number, z: number, t: number) => number[] }).homotopy;
    expectRowsClose([h(0, 0, 0, 0.5)], [[0, 0.2, 0]]);
    expectRowsClose([h(1, 2, 3, 0.0)], [[1, 2, 3]]);
  });

  it("honours custom direction/amplitude/ripples", () => {
    const aw = new ApplyWave(new Square(), [1, 0, 0], 0.5, undefined, 0.5, 2);
    const h = (aw as unknown as { homotopy: (x: number, y: number, z: number, t: number) => number[] }).homotopy;
    expectRowsClose([h(0.5, 0, 0, 0.5)], [[0.5, 0, 0]]);
  });

  it("honours an explicit run_time", () => {
    expectClose(new ApplyWave(new Square(), undefined, undefined, undefined, undefined, undefined, 5).run_time, 5);
  });
});

// ---------------------------------------------------------------------------
// Wiggle.get_scale_about_point / get_rotate_about_point
// Python:
//   Wiggle(sq): scale 1.1, angle 0.01*TAU, n 6, run_time 2;
//     get_scale_about_point() == get_rotate_about_point() == [0,0,0]
//   Wiggle(sq, scale_about_point=[1,2,3], rotate_about_point=[4,5,6])
//     -> [1,2,3] / [4,5,6]
//   interpolate(0) unchanged; interpolate(0.25) first row
//     [1.000787926393, 1.003256288279, 0]; interpolate(1) back to start.
// ---------------------------------------------------------------------------
describe("Wiggle", () => {
  it("stores defaults and falls back to the mobject center", () => {
    const w = new Wiggle(new Square());
    expectClose(w.scale_value, 1.1);
    expectClose(w.rotation_angle, 0.01 * TAU);
    expect(w.n_wiggles).toBe(6);
    expectClose(w.run_time, 2);
    expectRowsClose([w.get_scale_about_point()], [[0, 0, 0]]);
    expectRowsClose([w.get_rotate_about_point()], [[0, 0, 0]]);
  });

  it("returns explicit about-points", () => {
    const w = new Wiggle(new Square(), { scale_about_point: [1, 2, 3], rotate_about_point: [4, 5, 6] });
    expectRowsClose([w.get_scale_about_point()], [[1, 2, 3]]);
    expectRowsClose([w.get_rotate_about_point()], [[4, 5, 6]]);
  });

  it("interpolate_submobject matches ground truth and cycles back", () => {
    const sq = new Square();
    const w = new Wiggle(sq);
    w.begin();
    w.interpolate(0.0);
    expect(maxDiff(rowsOf(sq.points, sq.dim), rowsOf(w.starting_mobject.points, 3))).toBe(0);
    w.interpolate(0.25);
    expectRowsClose([rowsOf(sq.points, sq.dim)[0] ?? []], [[1.000787926393, 1.003256288279, 0]]);
    w.interpolate(1.0);
    expect(maxDiff(rowsOf(sq.points, sq.dim), rowsOf(w.starting_mobject.points, 3))).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// SmoothedVectorizedHomotopy (+ interpolate_submobject)
// Python 0.21.0: interpolate_submobject ALWAYS raises
//   NameError: name 'VMobject' is not defined
// (VMobject is only imported under TYPE_CHECKING in movement.py). The TS port
// implements the evident intent: apply the homotopy, then make_smooth().
// TS behavior asserted here: homotopy (x+t) at alpha 0.5 shifts the square's
// center to [0.5, 0, 0].
// ---------------------------------------------------------------------------
describe("SmoothedVectorizedHomotopy", () => {
  it("is a Homotopy subclass", () => {
    expect(new SmoothedVectorizedHomotopy((x, y, z) => [x, y, z], new Square())).toBeInstanceOf(Homotopy);
  });

  it("applies the homotopy then smooths", () => {
    const sub = new Square();
    const start = new Square();
    const svh = new SmoothedVectorizedHomotopy((x: number, y: number, z: number) => [x + 0.5, y, z], new Square());
    svh.interpolate_submobject(sub, start, 0.5);
    expectRowsClose([sub.get_center()], [[0.5, 0, 0]]);
    expect(sub.get_num_points()).toBe(start.get_num_points());
  });

  it("rejects non-VMobject submobjects like the upstream assert", () => {
    // Python: `assert isinstance(submobject, VMobject)` in
    // SmoothedVectorizedHomotopy.interpolate_submobject.
    const svh = new SmoothedVectorizedHomotopy((x: number, y: number, z: number) => [x, y, z], new Square());
    expect(() => svh.interpolate_submobject(new Mobject(), new Square(), 0.5)).toThrowError(
      expect.objectContaining({ name: "AssertionError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// ComplexHomotopy (+ __init__)
// Python:
//   ComplexHomotopy(lambda z,t: z*2, Square()):
//     homotopy(1,2,0,0) == [2,4,0]; run_time == 3 (Homotopy default)
//   begin + interpolate(0.5) first row == [2,2,0].
//   Homotopy(lambda:..., Square()): run_time 3, apply_function_kwargs {}.
// ---------------------------------------------------------------------------
describe("ComplexHomotopy", () => {
  it("maps complex pairs through the user function", () => {
    const ch = new ComplexHomotopy((z: { re: number; im: number }) => ({ re: z.re * 2, im: z.im * 2 }), new Square());
    expect(ch).toBeInstanceOf(Homotopy);
    expectClose(ch.run_time, 3);
    const h = (ch as unknown as { homotopy: (x: number, y: number, z: number, t: number) => number[] }).homotopy;
    expectRowsClose([h(1, 2, 0, 0.0)], [[2, 4, 0]]);
  });

  it("interpolates by doubling points", () => {
    const sq = new Square();
    const ch = new ComplexHomotopy((z: { re: number; im: number }) => ({ re: z.re * 2, im: z.im * 2 }), sq);
    ch.begin();
    ch.interpolate(0.5);
    expectRowsClose([rowsOf(sq.points, sq.dim)[0] ?? []], [[2, 2, 0]]);
  });

  it("Homotopy stores run_time 3 and empty apply kwargs", () => {
    const h = new Homotopy((x: number, y: number, z: number) => [x, y, z], new Square());
    expectClose(h.run_time, 3);
    expect(h.apply_function_kwargs).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// PhaseFlow.interpolate_mobject
// Python (fn=[1,0,0] const, virtual_time=2.0):
//   virtual_time 2.0, suspend False, rate linear;
//   begin; interpolate(0) center [0,0,0]; (0.5) -> [1,0,0]; (1.0) -> [2,0,0].
//   Defaults: virtual_time 1, suspend False.
// ---------------------------------------------------------------------------
describe("PhaseFlow.interpolate_mobject", () => {
  it("advects the mobject along the vector field", () => {
    const sq = new Square();
    const pf = new PhaseFlow(() => [1, 0, 0], sq, { virtual_time: 2.0 });
    expectClose(pf.virtual_time, 2.0);
    expect(pf.suspend_mobject_updating).toBe(false);
    pf.begin();
    pf.interpolate(0.0);
    expectRowsClose([sq.get_center()], [[0, 0, 0]]);
    pf.interpolate(0.5);
    expectRowsClose([sq.get_center()], [[1, 0, 0]]);
    pf.interpolate(1.0);
    expectRowsClose([sq.get_center()], [[2, 0, 0]]);
  });

  it("defaults virtual_time to 1 with updating suspended False", () => {
    const pf = new PhaseFlow(() => [0, 0, 0], new Square());
    expectClose(pf.virtual_time, 1);
    expect(pf.suspend_mobject_updating).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// MoveAlongPath.__init__
// Python: suspend False, path stored by identity;
//   interpolate_mobject(0/0.5/1) centers [0,0,0]/[2,0,0]/[4,0,0]
//   on a Line([0,0,0],[4,0,0]) path.
// ---------------------------------------------------------------------------
describe("MoveAlongPath.__init__", () => {
  it("stores the path and moves along it", () => {
    const sq = new Square();
    const path = new Line([0, 0, 0], [4, 0, 0]);
    const mp = new MoveAlongPath(sq, path as unknown as MoveAlongPath["path"]);
    expect(mp.suspend_mobject_updating).toBe(false);
    expect(mp.path).toBe(path);
    mp.begin();
    mp.interpolate_mobject(0.0);
    expectRowsClose([sq.get_center()], [[0, 0, 0]]);
    mp.interpolate_mobject(0.5);
    expectRowsClose([sq.get_center()], [[2, 0, 0]]);
    mp.interpolate_mobject(1.0);
    expectRowsClose([sq.get_center()], [[4, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// ChangeDecimalToValue
// Python ground truth is interpolate(start, target, a): 0->10 gives
//   0 / 5.0 / 10 at a = 0 / 0.5 / 1.
// (Real DecimalNumber needs latex, unavailable here, and is a text/TeX
// mobject out of port scope; the TS port checks structurally for set_value,
// so a Mobject stand-in with number/set_value is used.)
// ---------------------------------------------------------------------------
describe("ChangeDecimalToValue", () => {
  function fakeDecimal(start: number): Mobject & { number: number; set_value(v: number): void } {
    const m = new Square() as unknown as Mobject & { number: number; set_value(v: number): void };
    m.number = start;
    m.set_value = function (v: number): void {
      (this as unknown as { number: number }).number = v;
    };
    return m;
  }

  it("interpolates linearly from the start number", () => {
    const d = fakeDecimal(0);
    const cdv = new ChangeDecimalToValue(d, 10);
    expectClose(cdv.number_update_func(0), 0);
    expectClose(cdv.number_update_func(0.5), 5.0);
    expectClose(cdv.number_update_func(1), 10);
    cdv.begin();
    cdv.interpolate_mobject(0.5);
    expectClose(d.number, 5.0);
    cdv.interpolate_mobject(1.0);
    expectClose(d.number, 10);
  });

  it("starts from a nonzero value", () => {
    const d = fakeDecimal(5);
    const cdv = new ChangeDecimalToValue(d, -5);
    // Python: interpolate(5,-5,0.25) == 2.5
    expectClose(cdv.number_update_func(0.25), 2.5);
  });
});

// ---------------------------------------------------------------------------
// ReplacementTransform (+ __init__) / TransformFromCopy (+ __init__)
// Python:
//   ReplacementTransform(s1, s2): replace flag True, path_arc 0;
//     begin + interpolate(1) first row [2,2,0] (== scaled square).
//   TransformFromCopy(a, b): mobject is b, target is a;
//     interpolate(0) -> [1,1,0]; interpolate(1) -> [2,2,0] (reversed).
// ---------------------------------------------------------------------------
describe("ReplacementTransform", () => {
  it("sets the replace flag and morphs to the target", () => {
    const s1 = new Square();
    const s2 = new Square();
    s2.scale(2);
    const rt = new ReplacementTransform(s1, s2);
    expect(rt.replace_mobject_with_target_in_scene).toBe(true);
    expectClose(rt.path_arc, 0);
    rt.begin();
    rt.interpolate(1.0);
    expectRowsClose([rowsOf(s1.points, s1.dim)[0] ?? []], [[2, 2, 0]]);
  });
});

describe("TransformFromCopy", () => {
  it("swaps mobject/target and interpolates in reverse", () => {
    const a = new Square();
    const b = new Square();
    b.scale(2);
    const tfc = new TransformFromCopy(a, b);
    expect(tfc.mobject).toBe(b);
    expect(tfc.target_mobject).toBe(a);
    tfc.begin();
    tfc.interpolate(0.0);
    expectRowsClose([rowsOf(tfc.mobject.points, tfc.mobject.dim)[0] ?? []], [[1, 1, 0]]);
    tfc.interpolate(1.0);
    expectRowsClose([rowsOf(tfc.mobject.points, tfc.mobject.dim)[0] ?? []], [[2, 2, 0]]);
  });
});

// ---------------------------------------------------------------------------
// ClockwiseTransform / CounterclockwiseTransform (+ __init__)
// Python: path_arc -pi / +pi;
//   Square -> shifted Square([2,0,0]) at alpha 0.5:
//     CW center [1,1,0]; plain Transform [1,0,0]; CCW [1,-1,0].
// ---------------------------------------------------------------------------
describe("ClockwiseTransform / CounterclockwiseTransform", () => {
  function shifted(): { from: Square; to: Square } {
    const from = new Square();
    const to = new Square();
    to.shift([2, 0, 0]);
    return { from, to };
  }

  it("ClockwiseTransform defaults to -pi and arcs above", () => {
    const { from, to } = shifted();
    const cw = new ClockwiseTransform(from, to);
    expectClose(cw.path_arc, -PI);
    cw.begin();
    cw.interpolate(0.5);
    expectRowsClose([from.get_center()], [[1, 1, 0]]);
  });

  it("CounterclockwiseTransform defaults to +pi and arcs below", () => {
    const { from, to } = shifted();
    const ccw = new CounterclockwiseTransform(from, to);
    expectClose(ccw.path_arc, PI);
    ccw.begin();
    ccw.interpolate(0.5);
    expectRowsClose([from.get_center()], [[1, -1, 0]]);
  });

  it("plain Transform goes straight through the middle", () => {
    const { from, to } = shifted();
    const t = new Transform(from, to);
    t.begin();
    t.interpolate(0.5);
    expectRowsClose([from.get_center()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// Line._set_start_and_end_attrs
// Python:
//   Line([0,0,0],[1,0,0]): start [0,0,0], end [1,0,0].
//   Square boundary quirk: get_boundary_point([1,0,0]) == [1,1,0]
//     (max-dot search over stored corner points, first tie wins), so
//     Line(sq,[3,0,0]).start == [1,1,0], .end == [3,0,0];
//     Line([0,0,0],sq).end == [1,1,0], .start == [0,0,0].
// ---------------------------------------------------------------------------
describe("Line._set_start_and_end_attrs", () => {
  it("stores plain point endpoints", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expectRowsClose(rowsOf(l.start, 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(l.end, 3), [[1, 0, 0]]);
  });

  it("uses boundary points for mobject endpoints", () => {
    const sq = new Square();
    expectRowsClose([sq.get_boundary_point([1, 0, 0])], [[1, 1, 0]]);
    const l2 = new Line(sq, [3, 0, 0]);
    expectRowsClose(rowsOf(l2.start, 3), [[1, 1, 0]]);
    expectRowsClose(rowsOf(l2.end, 3), [[3, 0, 0]]);
    const l3 = new Line([0, 0, 0], new Square());
    expectRowsClose(rowsOf(l3.start, 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(l3.end, 3), [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// MoveToTarget (+ __init__)
// Python:
//   MoveToTarget(c) with generated+shifted/scaled target:
//     target_mobject is c.target; begin+interpolate(1) reaches it exactly.
//   MoveToTarget(Square()) does NOT raise (Mobject.target exists, None);
//     Transform then substitutes a fresh Mobject target.
//   MoveToTarget(non-mobject without target) raises ValueError with
//     "MoveToTarget called on mobjectwithout attribute 'target'".
// ---------------------------------------------------------------------------
describe("MoveToTarget", () => {
  it("targets the generated target and lands on it", () => {
    const c = new Circle();
    c.generate_target();
    (c.target as Mobject).shift([2, 0, 0]);
    (c.target as Mobject).scale(0.5);
    const mt = new MoveToTarget(c);
    expect(mt.target_mobject).toBe(c.target);
    mt.begin();
    mt.interpolate(1.0);
    const expected = new Circle();
    expected.shift([2, 0, 0]);
    expected.scale(0.5);
    expect(maxDiff(rowsOf(c.points, c.dim), rowsOf(expected.points, expected.dim))).toBe(0);
  });

  it("accepts a fresh mobject whose target is null", () => {
    const sq = new Square();
    expect(sq.target).toBeNull();
    const mt = new MoveToTarget(sq);
    expect(mt.target_mobject).toBeInstanceOf(Mobject);
    expect((mt.target_mobject as Mobject).get_num_points()).toBe(0);
  });

  it("raises ValueError without a target attribute", () => {
    expect(() => new MoveToTarget({} as unknown as Mobject)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// TipableVMobject
// Python: TipableVMobject(tip_length=0.5): tip_length 0.5, normal OUT,
//   tip_style {}; default tip_length 0.35; has_tip/has_start_tip False.
// ---------------------------------------------------------------------------
describe("TipableVMobject", () => {
  it("stores tip configuration defaults", () => {
    const t = new TipableVMobject();
    expectClose(t.tip_length, 0.35);
    expectRowsClose([[...(t.normal_vector as number[])]], [[0, 0, 1]]);
    expect(t.tip_style).toEqual({});
    expect(t.has_tip()).toBe(false);
    expect(t.has_start_tip()).toBe(false);
    expectClose(t.get_default_tip_length(), 0.35);
  });

  it("honours an explicit tip length", () => {
    const t = new TipableVMobject({ tip_length: 0.5 });
    expectClose(t.tip_length, 0.5);
    expectClose(t.get_default_tip_length(), 0.5);
  });
});

// ---------------------------------------------------------------------------
// CyclicReplace
// Python (a=Square, b=Square+[3,0,0]):
//   group len 2, path_arc pi/2; create_target centers [3,0,0],[0,0,0];
//   single-VGroup-arg form keeps identity;
//   begin + interpolate(1): a -> [3,0,0], b -> [0,0,0].
// ---------------------------------------------------------------------------
describe("CyclicReplace", () => {
  it("cycles targets and swaps positions on interpolate(1)", () => {
    const a = new Square();
    const b = new Square();
    b.shift([3, 0, 0]);
    const cr = new CyclicReplace(a, b);
    expect(cr.group.submobjects.length).toBe(2);
    expectClose(cr.path_arc, PI / 2);
    const target = cr.create_target() as unknown as Group;
    expectRowsClose([(target.submobjects[0] as Mobject).get_center()], [[3, 0, 0]]);
    expectRowsClose([(target.submobjects[1] as Mobject).get_center()], [[0, 0, 0]]);
    cr.begin();
    cr.interpolate(1.0);
    expectRowsClose([a.get_center()], [[3, 0, 0]]);
    expectRowsClose([b.get_center()], [[0, 0, 0]]);
  });

  it("keeps a single Group/VGroup argument by identity", () => {
    const vg = new VGroup(new Square(), new Square());
    const cr = new CyclicReplace(vg as unknown as Mobject);
    expect(cr.group).toBe(vg);
  });
});

// ---------------------------------------------------------------------------
// TransformAnimations
// Python (Transform run_time=1, Transform run_time=2):
//   sub-anims resized to max == 2; final run_time == 1.0 (super().__init__
//   resets it: run_time was popped from kwargs); mobject/target rewired so
//   start_anim.mobject is starting_mobject and end_anim.mobject is
//   target_mobject.
// ---------------------------------------------------------------------------
describe("TransformAnimations", () => {
  it("resizes sub-anims but keeps the default run_time", () => {
    const sa = new Transform(new Square(), ((): Square => { const s = new Square(); s.scale(2); return s; })(), { run_time: 1 });
    const ea = new Transform(new Square(), ((): Square => { const s = new Square(); s.shift([1, 0, 0]); return s; })(), { run_time: 2 });
    const ta = new TransformAnimations(sa, ea);
    expectClose(sa.run_time, 2);
    expectClose(ea.run_time, 2);
    // NOTE: upstream quirk — run_time is always the default here.
    expectClose(ta.run_time, 1.0);
    expect(ta.start_anim.mobject).toBe(ta.starting_mobject);
    expect(ta.end_anim.mobject).toBe(ta.target_mobject);
    // NOTE: Python `TransformAnimations.mobject is start_anim.mobject` is
    // False after construction: start_anim.mobject is rewired to
    // starting_mobject while self.mobject keeps the original square.
    expect(ta.mobject).not.toBe(sa.mobject);
    expect(ta.target_mobject).toBe(ea.mobject);
  });

  it("ignores an explicit run_time the same way (still 1.0)", () => {
    const sa = new Transform(new Square(), new Square(), { run_time: 1 });
    const ea = new Transform(new Square(), new Square(), { run_time: 2 });
    const ta = new TransformAnimations(sa, ea, undefined, { run_time: 5 } as unknown as undefined);
    expectClose(ta.run_time, 1.0);
  });
});

// ---------------------------------------------------------------------------
// FadeTransform.__init__
// Python (s1=Square, s2=scaled Square):
//   stretch True, dim_to_match 1, to_add is s2, mobject Group of 2;
//   custom stretch=False dim 0; begin -> starting/ending Groups,
//   ending sub0 fill opacity 0.
// ---------------------------------------------------------------------------
describe("FadeTransform.__init__", () => {
  it("wraps both mobjects in a group", () => {
    const s1 = new Square();
    const s2 = new Square();
    s2.scale(2);
    const ft = new FadeTransform(s1, s2);
    expect(ft.stretch).toBe(true);
    expect(ft.dim_to_match).toBe(1);
    expect(ft.to_add_on_completion).toBe(s2);
    expect(ft.mobject).toBeInstanceOf(Group);
    expect((ft.mobject as Group).submobjects.length).toBe(2);
  });

  it("honours stretch/dim options and ghosts opacity on begin", () => {
    const s1 = new Square();
    const s2 = new Square();
    s2.scale(2);
    const ft = new FadeTransform(s1, s2, { stretch: false, dim_to_match: 0 });
    expect(ft.stretch).toBe(false);
    expect(ft.dim_to_match).toBe(0);
    ft.begin();
    expect(ft.starting_mobject).toBeInstanceOf(Group);
    expect(ft.ending_mobject).toBeInstanceOf(Group);
    expectClose(((ft.ending_mobject as Group).submobjects[0] as VMobject).get_fill_opacity(), 0.0);
  });
});

// ---------------------------------------------------------------------------
// GrowFromPoint.create_target
// Python: point stored, point_color None, introducer True;
//   create_target() is the mobject itself;
//   starting mobject: width 0, center ORIGIN.
// ---------------------------------------------------------------------------
describe("GrowFromPoint.create_target", () => {
  it("targets the mobject itself with a point-sized start", () => {
    const sq = new Square();
    const gp = new GrowFromPoint(sq, [0, 0, 0]);
    expect(gp.create_target()).toBe(gp.mobject);
    expect(gp.is_introducer()).toBe(true);
    expect(gp.point_color).toBeNull();
    gp.begin();
    gp.interpolate(1.0);
    expect(maxDiff(rowsOf(sq.points, sq.dim), rowsOf(new Square().points, 3))).toBe(0);
    expectClose(gp.starting_mobject.width, 0.0);
    expectRowsClose([gp.starting_mobject.get_center()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// Animation.__str__
// Python: str(Animation(Square())) == "Animation(Square)";
//   with name="hello" -> "hello".
// ---------------------------------------------------------------------------
describe("Animation.__str__", () => {
  it("names the class and mobject by default", () => {
    expect(new Animation(new Square()).__str__()).toBe("Animation(Square)");
    expect(String(new Animation(new Square()))).toBe("Animation(Square)");
  });

  it("prefers an explicit name", () => {
    expect(new Animation(new Square(), { name: "hello" }).__str__()).toBe("hello");
  });
});

// ---------------------------------------------------------------------------
// Add.__init__
// Python: Add(d1, d2) -> Group of 2, run_time 0, introducer;
//   Add(d1).mobject is d1; Add() -> empty Group.
// ---------------------------------------------------------------------------
describe("Add.__init__", () => {
  it("groups several mobjects with zero run_time", () => {
    const d1 = new Dot();
    const d2 = new Dot();
    const ad = new Add(d1, d2);
    expect(ad.mobject).toBeInstanceOf(Group);
    expect(ad.mobject.submobjects.length).toBe(2);
    expectClose(ad.run_time, 0.0);
    expect(ad.is_introducer()).toBe(true);
  });

  it("passes a single mobject through by identity", () => {
    const d1 = new Dot();
    expect(new Add(d1).mobject).toBe(d1);
  });

  it("makes an empty Group with no args", () => {
    const ad = new Add();
    expect(ad.mobject).toBeInstanceOf(Group);
    expect(ad.mobject.submobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Polygram
// Python:
//   Polygram([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]): 16 pts,
//     vertices == the 4 corners, one vertex group equal to them.
//   Two vertex groups -> 24 pts, 2 groups.
// ---------------------------------------------------------------------------
describe("Polygram", () => {
  it("builds a closed path from one vertex group", () => {
    const pg = new Polygram([
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ]);
    expect(pg.get_num_points()).toBe(16);
    expectRowsClose(pg.get_vertices(), [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ]);
    const groups = pg.get_vertex_groups();
    expect(groups.length).toBe(1);
    expectRowsClose(groups[0] ?? [], [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ]);
  });

  it("supports several vertex groups", () => {
    const pg = new Polygram(
      [
        [0, 0, 0],
        [2, 0, 0],
        [2, 2, 0],
      ],
      [
        [5, 5, 0],
        [6, 5, 0],
        [6, 6, 0],
      ],
    );
    expect(pg.get_vertex_groups().length).toBe(2);
    expect(pg.get_num_points()).toBe(24);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-1012 integration", () => {
  it("ApplyWave then Wiggle returns a square to itself", () => {
    // Python: ApplyWave begin/interpolate(1), then Wiggle begin/interpolate(1)
    // -> max abs diff 4.385380947269368e-15 vs the pristine points.
    const sq = new Square();
    const pristine = rowsOf(new Square().points, 3);
    const aw = new ApplyWave(sq);
    aw.begin();
    aw.interpolate(1.0);
    const w = new Wiggle(sq);
    w.begin();
    w.interpolate(1.0);
    expect(maxDiff(rowsOf(sq.points, sq.dim), pristine)).toBeLessThan(1e-12);
  });

  it("MoveToTarget drives a whole VGroup to its generated target", () => {
    // Python: VGroup(Circle, Square) with target subs shifted/scaled ->
    // both diffs exactly 0.0 after begin + interpolate(1).
    const g = new VGroup(new Circle(), new Square());
    g.generate_target();
    ((g.target as unknown as VGroup).submobjects[0] as Mobject).shift([2, 0, 0]);
    ((g.target as unknown as VGroup).submobjects[1] as Mobject).scale(0.5);
    const mt = new MoveToTarget(g);
    mt.begin();
    mt.interpolate(1.0);
    const e0 = new Circle();
    e0.shift([2, 0, 0]);
    const e1 = new Square();
    e1.scale(0.5);
    expect(maxDiff(rowsOf((g.submobjects[0] as VMobject).points, 3), rowsOf(e0.points, e0.dim))).toBe(0);
    expect(maxDiff(rowsOf((g.submobjects[1] as VMobject).points, 3), rowsOf(e1.points, e1.dim))).toBe(0);
  });

  it("ClockwiseTransform arcs off the straight Transform path on a Triangle", () => {
    // Python: Triangle -> scaled+shifted Triangle at alpha 0.5:
    //   CW center [1.375, 1.033493649054, 0]; straight [1.0, 0.25, 0].
    const mkTo = (): Triangle => {
      const t = new Triangle();
      t.scale(2);
      t.shift([2, 0, 0]);
      return t;
    };
    const cwFrom = new Triangle();
    const cw = new ClockwiseTransform(cwFrom, mkTo());
    cw.begin();
    cw.interpolate(0.5);
    expectRowsClose([cwFrom.get_center()], [[1.375, 1.033493649054, 0]]);
    const tFrom = new Triangle();
    const t = new Transform(tFrom, mkTo());
    t.begin();
    t.interpolate(0.5);
    expectRowsClose([tFrom.get_center()], [[1.0, 0.25, 0]]);
  });

  it("MoveAlongPath centers track the path proportion", () => {
    // Python: Square along Line([0,0,0],[4,0,0]) visits [0,0,0]/[2,0,0]/[4,0,0].
    // Cross-checks Line endpoint storage feeding the movement animation.
    const sq = new Square();
    const path = new Line([0, 0, 0], [4, 0, 0]);
    expectRowsClose(rowsOf(path.start, 3), [[0, 0, 0]]);
    const mp = new MoveAlongPath(sq, path as unknown as MoveAlongPath["path"]);
    mp.begin();
    for (const [alpha, x] of [[0, 0], [0.5, 2], [1, 4]] as const) {
      mp.interpolate_mobject(alpha);
      expectRowsClose([sq.get_center()], [[x, 0, 0]]);
    }
  });

  it("FadeTransform beginning ghosts the ending side to zero opacity", () => {
    // Python: begin -> ending is a Group whose sub0 fill opacity is 0.0.
    // Cross-checks FadeTransform init wiring with the Group/opacity stack.
    const s1 = new Square();
    const s2 = new Square();
    s2.scale(2);
    const ft = new FadeTransform(s1, s2);
    ft.begin();
    expect(ft.starting_mobject).toBeInstanceOf(Group);
    expectClose(((ft.ending_mobject as Group).submobjects[0] as VMobject).get_fill_opacity(), 0.0);
    expect(ft.to_add_on_completion).toBe(s2);
  });
});
