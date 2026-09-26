// Ground-truth tests for batch layers 13..13.
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here; each block cites the
// probe in a "Python:" comment. TS is asserted against those values (1e-9
// relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): nothing in this batch touches text/tex.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Mobject, Group, _AnimationBuilder, _UpdaterBuilder } from "../src/mobject/mobject.js";
import { get_mobject_class } from "../src/mobject/utils.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject, Point } from "../src/mobject/types/point_cloud_mobject.js";
import { AbstractImageMobject, ImageMobject } from "../src/mobject/types/image_mobject.js";
import { Square, Rectangle } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { VectorField } from "../src/mobject/vector_field.js";
import { GenericGraph, Graph, DiGraph } from "../src/mobject/graph.js";
import { Axes } from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { ChangeSpeed } from "../src/animation/speedmodifier.js";
import { Create } from "../src/animation/creation.js";
import { Animation } from "../src/animation/animation.js";
import { MaintainPositionRelativeTo } from "../src/animation/updaters/update.js";
import {
  assert_is_mobject_method,
  bind_mobject_method,
} from "../src/animation/updaters/mobject_update_utils.js";
import { Camera } from "../src/camera/camera.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import {
  extract_mobject_family_members,
  restructure_list_to_exclude_certain_family_members,
} from "../src/utils/family.js";
import { print_family } from "../src/utils/debug.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
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

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker.__add__
// Python:
//   ValueTracker(2.5) + 1.5 -> get_value() == 4.0
//   ValueTracker(2.5) + Mobject() -> ValueError("Cannot increment ValueTracker ...")
// ---------------------------------------------------------------------------
describe("ValueTracker.__add__", () => {
  it("adds a scalar into a new tracker", () => {
    const v = new ValueTracker(2.5);
    const out = v.__add__(1.5);
    expect(out).toBeInstanceOf(ValueTracker);
    expectClose(out.get_value(), 4.0);
    expectClose(v.get_value(), 2.5);
  });

  it("rejects Mobject operands", () => {
    const v = new ValueTracker(2.5);
    expect(() => v.__add__(new Mobject())).toThrowError(/Cannot increment ValueTracker/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker.__iadd__
// Python: w = ValueTracker(1.0); w += 2.0 -> 3.0 (same object)
// ---------------------------------------------------------------------------
describe("ValueTracker.__iadd__", () => {
  it("increments in place and returns self", () => {
    const w = new ValueTracker(1.0);
    const out = w.__iadd__(2.0);
    expect(out).toBe(w);
    expectClose(w.get_value(), 3.0);
  });

  it("rejects Mobject operands", () => {
    const w = new ValueTracker(1.0);
    expect(() => w.__iadd__(new Mobject())).toThrowError(/Cannot increment ValueTracker/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker.__sub__ / __isub__
// Python:
//   ValueTracker(2.5) - 0.5 -> 2.0
//   w = ValueTracker(5.0); w -= 1.5 -> 3.5 (same object)
//   Mobject operand -> ValueError("Cannot decrement ValueTracker ...")
// ---------------------------------------------------------------------------
describe("ValueTracker.__sub__ / __isub__", () => {
  it("subtracts into a new tracker", () => {
    const v = new ValueTracker(2.5);
    const out = v.__sub__(0.5);
    expect(out).toBeInstanceOf(ValueTracker);
    expectClose(out.get_value(), 2.0);
    expectClose(v.get_value(), 2.5);
  });

  it("decrements in place and returns self", () => {
    const w = new ValueTracker(5.0);
    const out = w.__isub__(1.5);
    expect(out).toBe(w);
    expectClose(w.get_value(), 3.5);
  });

  it("rejects Mobject operands", () => {
    const v = new ValueTracker(2.5);
    expect(() => v.__sub__(new Mobject())).toThrowError(/Cannot decrement ValueTracker/);
    expect(() => v.__isub__(new Mobject())).toThrowError(/Cannot decrement ValueTracker/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker.interpolate
// Python:
//   a=ValueTracker(0), b=ValueTracker(10), c.interpolate(a,b,0.3) -> 3.0
//   alpha=0 -> 0.0, alpha=1 -> 10.0
// ---------------------------------------------------------------------------
describe("ValueTracker.interpolate", () => {
  it("blends between two trackers", () => {
    const a = new ValueTracker(0.0);
    const b = new ValueTracker(10.0);
    const c = new ValueTracker(0.0);
    c.interpolate(a, b, 0.3);
    expectClose(c.get_value(), 3.0);
    c.interpolate(a, b, 0.0);
    expectClose(c.get_value(), 0.0);
    c.interpolate(a, b, 1.0);
    expectClose(c.get_value(), 10.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.VectorField.nudge
// Python (func = constant [1,0,0]):
//   Circle at origin nudged dt=2 -> center [2,0,0]
//   pointwise dt=1 substeps=2 -> first point delta [1,0,0]
//   Point at origin nudged dt=0.5 -> center [0.5,0,0]
// ---------------------------------------------------------------------------
describe("VectorField.nudge", () => {
  it("shifts a mobject along a constant field", () => {
    const vf = new VectorField((_p) => [1.0, 0.0, 0.0]);
    const c = new Circle(1.0);
    vf.nudge(c, 2.0, 1, false);
    expectRowsClose(rowsOf(c.get_center() as unknown as NDArray, 3), [[2.0, 0.0, 0.0]]);
  });

  it("moves points pointwise when requested", () => {
    const vf = new VectorField((_p) => [1.0, 0.0, 0.0]);
    const c = new Circle(1.0);
    const before = rowsOf(c.points, 3).map((r) => [...r]);
    vf.nudge(c, 1.0, 2, true);
    const after = rowsOf(c.points, 3);
    const delta = (after[0] ?? []).map((v, i) => v - (before[0]?.[i] ?? 0));
    expectRowsClose([delta], [[1.0, 0.0, 0.0]]);
  });

  it("nudges a point cloud dot", () => {
    const vf = new VectorField((_p) => [1.0, 0.0, 0.0]);
    const pt = new Point();
    vf.nudge(pt, 0.5);
    expectRowsClose([[...pt.get_center()]], [[0.5, 0.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.surround
// Python: Square(2).surround(Rectangle(4,1)) -> width 4.25 height 4.25
// (MED_SMALL_BUFF == 0.25)
// ---------------------------------------------------------------------------
describe("Mobject.surround", () => {
  it("replaces then scales by (length+buff)/length", () => {
    const s = new Square(2.0);
    const r = new Rectangle({ width: 4.0, height: 1.0 });
    s.surround(r);
    expectClose(s.width, 4.25);
    expectClose(s.height, 4.25);
    expectRowsClose([[...s.get_center()]], [[0.0, 0.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Circle.surround
// Python:
//   Circle().surround(Square(2)) -> width 3.394112549695428 (= sqrt(8)*1.2)
//   Circle().surround(Square(2), buffer_factor=2.0) -> width 5.656854249492381
// ---------------------------------------------------------------------------
describe("Circle.surround", () => {
  it("fits the diagonal then scales by buffer_factor", () => {
    const c = new Circle();
    c.surround(new Square(2.0));
    expectClose(c.width, 3.394112549695428);
    const c2 = new Circle();
    c2.surround(new Square(2.0), 0, false, 2.0);
    expectClose(c2.width, 5.656854249492381);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.clear_points
// Python: Circle radius 1 has 32 points; clear_points -> 0 points, shape (0,3)
// ---------------------------------------------------------------------------
describe("VMobject.clear_points", () => {
  it("empties the points array", () => {
    const v = new Circle(1.0);
    expect(v.get_num_points()).toBe(32);
    v.clear_points();
    expect(v.get_num_points()).toBe(0);
    expect(v.points.shape).toEqual([0, 3]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.pointwise_become_partial
// Python (Circle r=1, 32 points):
//   partial(0.25, 0.75) -> 20 points, first point ~= [6.12e-17, 1.0, 0.0]
//   partial(0, 1) -> points equal to source
//   non-VMobject -> TypeError
// ---------------------------------------------------------------------------
describe("VMobject.pointwise_become_partial", () => {
  it("extracts the middle portion of the spline", () => {
    const src = new Circle(1.0);
    const dst = new Circle(1.0);
    dst.pointwise_become_partial(src, 0.25, 0.75);
    expect(dst.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(dst.points, 3).slice(0, 1), [[6.123233995736766e-17, 1.0, 0.0]]);
  });

  it("copies all points for the full range", () => {
    const src = new Circle(1.0);
    const dst = new Circle(1.0);
    dst.pointwise_become_partial(src, 0, 1);
    expectRowsClose(rowsOf(dst.points, 3), rowsOf(src.points, 3));
  });

  it("rejects non-VMobjects", () => {
    const dst = new Circle(1.0);
    expect(() => dst.pointwise_become_partial(new Mobject() as unknown as VMobject, 0, 1)).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PMobject.align_points_with_larger
// Python:
//   1-point PMobject aligned with 3-point PMobject -> 3 points, all [5,5,5]
//   non-PMobject -> AssertionError
// ---------------------------------------------------------------------------
describe("PMobject.align_points_with_larger", () => {
  it("stretches arrays to the larger mobject's point count", () => {
    const big = new PMobject();
    big.add_points([[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
    const small = new PMobject();
    small.add_points([[5, 5, 5]]);
    small.align_points_with_larger(big);
    expect(small.get_num_points()).toBe(3);
    expectRowsClose(rowsOf(small.points, 3), [[5, 5, 5], [5, 5, 5], [5, 5, 5]]);
  });

  it("rejects non-PMobjects", () => {
    const small = new PMobject();
    small.add_points([[5, 5, 5]]);
    expect(() => small.align_points_with_larger(new Mobject())).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PMobject.interpolate_color
// Python: two 2-point PMobjects, alpha=0.5 -> rgbas all ones, stroke width 4.0
// ---------------------------------------------------------------------------
describe("PMobject.interpolate_color", () => {
  it("blends rgbas and stroke width", () => {
    const m1 = new PMobject();
    m1.add_points([[0, 0, 0], [1, 0, 0]]);
    const m2 = new PMobject();
    m2.add_points([[0, 0, 0], [0, 2, 0]]);
    const t = new PMobject();
    t.add_points([[0, 0, 0], [0, 0, 0]]);
    t.interpolate_color(m1, m2, 0.5);
    expectRowsClose(rowsOf(t.rgbas, 4), [[1, 1, 1, 1], [1, 1, 1, 1]]);
    expectClose(t.get_stroke_width(), 4.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PMobject.pointwise_become_partial
// Python: 4 points, a=0.25 b=0.75 -> middle 2 points [[1,1,1],[2,2,2]]
// ---------------------------------------------------------------------------
describe("PMobject.pointwise_become_partial", () => {
  it("slices every array attribute", () => {
    const p = new PMobject();
    p.add_points([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    const q = new PMobject();
    q.pointwise_become_partial(p, 0.25, 0.75);
    expect(q.get_num_points()).toBe(2);
    expectRowsClose(rowsOf(q.points, 3), [[1, 1, 1], [2, 2, 2]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.image_mobject.ImageMobject.interpolate_color
// Python: 2x2 arrays, R channel 10 vs 110, alpha=0.5 -> R == 60 at [0,0]
//   fill_opacity/stroke_opacity stay 1.0; mismatched shapes -> AssertionError;
//   non-ImageMobjects -> AssertionError
// ---------------------------------------------------------------------------
describe("ImageMobject.interpolate_color", () => {
  function makeMobject(r: number, h = 2, w = 2): ImageMobject {
    const arr: number[][][] = Array.from({ length: h }, () =>
      Array.from({ length: w }, () => [r, 0, 0, 255]),
    );
    return new ImageMobject(arr, false);
  }

  it("blends pixel arrays elementwise", () => {
    const m1 = makeMobject(10);
    const m2 = makeMobject(110);
    const t = makeMobject(10);
    t.interpolate_color(m1, m2, 0.5);
    const px = (t.pixel_array.toArray() as number[][][])[0]?.[0];
    expect(px?.[0]).toBe(60);
    expectClose(t.fill_opacity, 1.0);
    expectClose(t.stroke_opacity, 1.0);
  });

  it("rejects incompatible inputs", () => {
    const m1 = makeMobject(10);
    const other = new Mobject();
    expect(() => m1.interpolate_color(m1, other as unknown as ImageMobject, 0.5)).toThrowError();
    const m3 = makeMobject(10, 3, 3);
    expect(() => m1.interpolate_color(m1, m3, 0.5)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.image_mobject.AbstractImageMobject
// Python: get_pixel_array / set_color raise NotImplementedError; valid
// resampling values (0) accepted, 99 -> ValueError.
// NOTE deviation: Python cannot even construct AbstractImageMobject directly
// (reset_points calls get_pixel_array which raises); the TS port guards that
// early call so the object can be built and the abstract methods still throw.
// ---------------------------------------------------------------------------
describe("AbstractImageMobject", () => {
  it("raises for unimplemented pixel accessors", () => {
    const ai = new AbstractImageMobject({ scale_to_resolution: 1080 });
    expect(() => ai.get_pixel_array()).toThrowError();
    expect(() => ai.set_color()).toThrowError();
  });

  it("validates the resampling algorithm", () => {
    const ai = new AbstractImageMobject({ scale_to_resolution: 1080 });
    expect(ai.set_resampling_algorithm(0)).toBe(ai);
    expect(() => ai.set_resampling_algorithm(99)).toThrowError(/resampling_algorithm/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.BackgroundRectangle.pointwise_become_partial
// Python: fresh BackgroundRectangle has fill_opacity == original == 0.75;
//   pointwise_become_partial(sq, 0.3, 0.5) -> fill_opacity 0.375
// ---------------------------------------------------------------------------
describe("BackgroundRectangle.pointwise_become_partial", () => {
  it("scales fill opacity by b", () => {
    const sq = new Square();
    const bg = new BackgroundRectangle(sq);
    expectClose(bg.fill_opacity, 0.75);
    expectClose(bg.original_fill_opacity, 0.75);
    bg.pointwise_become_partial(sq, 0.3, 0.5);
    expectClose(bg.fill_opacity, 0.375);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.__matmul__
// Python (Axes x_range=[-3,3,1] y_range=[-2,2,1] x_length=6 y_length=4):
//   ax @ [1,2,0] -> [1.0000000000000004, 2.0, 0.0]
//   ax @ Mobject(center [1,2,0]) -> same
// manim.mobject.graphing.number_line.NumberLine.__rmatmul__
// Python (x_range=[0,10,1], length=10):
//   __rmatmul__([0,0,0]) -> 5.0 ; __rmatmul__(n2p(5)) -> 5.0 ;
//   __matmul__(5) -> [0,0,0]
// ---------------------------------------------------------------------------
describe("CoordinateSystem.__matmul__ / NumberLine.__rmatmul__", () => {
  it("maps coords through Axes", () => {
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const out = rowsOf(ax.__matmul__([1.0, 2.0, 0.0]), 3);
    expectRowsClose(out, [[1.0000000000000004, 2.0, 0.0]]);
    const m = new Mobject();
    m.set_points([[1.0, 2.0, 0.0]]);
    expectRowsClose(rowsOf(ax.__matmul__(m), 3), [[1.0000000000000004, 2.0, 0.0]]);
  });

  it("maps points back through NumberLine", () => {
    const nl = new NumberLine({ x_range: [0, 10, 1], length: 10 });
    expectClose(nl.__rmatmul__([0.0, 0.0, 0.0]) as number, 5.0);
    const p5 = rowsOf(nl.__matmul__(5), 3)[0] ?? [];
    expectRowsClose([p5], [[0.0, 0.0, 0.0]]);
    expectClose(nl.__rmatmul__(p5) as number, 5.0);
    const m = new Mobject();
    m.set_points([p5]);
    expectClose(nl.__rmatmul__(m) as number, 5.0);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.camera.Camera.is_in_frame
// Python (100x100 px, frame 8x8):
//   unit square at origin -> True; at [100,100,0] -> False;
//   at [3.9,0,0] -> True; at [4.6,0,0] -> False
// ---------------------------------------------------------------------------
describe("Camera.is_in_frame", () => {
  it("classifies centered/offset mobjects", () => {
    const cam = new Camera({ pixel_height: 100, pixel_width: 100, frame_height: 8, frame_width: 8 });
    const center = new Square(1.0);
    expect(cam.is_in_frame(center)).toBe(true);
    const far = new Square(1.0);
    far.move_to([100, 100, 0]);
    expect(cam.is_in_frame(far)).toBe(false);
    const edgeIn = new Square(1.0);
    edgeIn.move_to([3.9, 0, 0]);
    expect(cam.is_in_frame(edgeIn)).toBe(true);
    const edgeOut = new Square(1.0);
    edgeOut.move_to([4.6, 0, 0]);
    expect(cam.is_in_frame(edgeOut)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.utils.get_mobject_class
// Python: returns Mobject when renderer is cairo.
// ---------------------------------------------------------------------------
describe("get_mobject_class", () => {
  it("returns the Mobject class", () => {
    expect(get_mobject_class()).toBe(Mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.family_ops.extract_mobject_family_members
// Python:
//   Group(a, b) -> 3 members; only_those_with_points=True -> 2
// manim.utils.family_ops.restructure_list_to_exclude_certain_family_members
//   .add_safe_mobjects_from_list: restructure([g],[a]) -> [b]
// ---------------------------------------------------------------------------
describe("family ops", () => {
  it("extracts families with and without the points filter", () => {
    const a = new Square();
    const b = new Square();
    const g = new Group(a, b);
    expect(extract_mobject_family_members([g]).length).toBe(3);
    expect(extract_mobject_family_members([g], false, true).length).toBe(2);
  });

  it("restructures a list to exclude removed family members", () => {
    const a = new Square();
    const b = new Square();
    const g = new Group(a, b);
    const out = restructure_list_to_exclude_certain_family_members([g], [a]);
    expect(out.length).toBe(1);
    expect(out[0]).toBe(b);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.debug.print_family
// Python: Group(Square(), Square()) prints 3 lines (" Group <id>" ...)
// ---------------------------------------------------------------------------
describe("print_family", () => {
  it("logs one line per family member", () => {
    const g = new Group(new Square(), new Square());
    const lines: string[] = [];
    const orig = console.log;
    console.log = (...args: unknown[]) => {
      lines.push(args.map(String).join(" "));
    };
    try {
      print_family(g);
    } finally {
      console.log = orig;
    }
    expect(lines.length).toBe(3);
    expect(lines[0]).toMatch(/Group/);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.mobject_update_utils.assert_is_mobject_method
// Python: bound method -> None; lambda / non-method -> AssertionError.
// TS port: bound method created via bind_mobject_method returns its mobject;
// anything without a backing Mobject throws.
// ---------------------------------------------------------------------------
describe("assert_is_mobject_method", () => {
  it("returns the mobject for a bound mobject method", () => {
    const s = new Square();
    const bound = bind_mobject_method(s, "shift");
    expect(assert_is_mobject_method(bound)).toBe(s);
  });

  it("throws for plain functions and non-methods", () => {
    const plain = Object.assign(() => {}, {});
    expect(() => assert_is_mobject_method(plain as never)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.update.MaintainPositionRelativeTo (+ __init__)
// Python:
//   a=Square(1) at origin, b=Square(1) at [3,0,0] -> diff [-3,0,0]
//   b.move_to([5,0,0]); interpolate_mobject(0.5) -> a center [2,0,0]
// ---------------------------------------------------------------------------
describe("MaintainPositionRelativeTo", () => {
  it("tracks the mobject while preserving the offset", () => {
    const a = new Square(1.0);
    const b = new Square(1.0);
    b.move_to([3, 0, 0]);
    const anim = new MaintainPositionRelativeTo(a, b);
    expectRowsClose([anim.diff], [[-3.0, 0.0, 0.0]]);
    b.move_to([5, 0, 0]);
    anim.interpolate_mobject(0.5);
    expectRowsClose([[...a.get_center()]], [[2.0, 0.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.speedmodifier.ChangeSpeed (+ add_updater)
// Python (Create(Square()), speedinfo {0:1, 0.5:4, 1:1}):
//   outer run_time 0.4, inner 1.0, get_scaled_total_time() 0.4
//   inner rate_func mapping: 0->0, 0.25->0.033938923906167505,
//     0.5->0.5, 0.75->0.9660610760938325, 1->1
//   add_updater: updater with (mob, dt) param attached once; with
//     is_changing_dt set it receives ChangeSpeed.dt (0 initially);
//     updater without dt param is attached as-is and called.
// ---------------------------------------------------------------------------
describe("ChangeSpeed", () => {
  it("scales run time and maps the rate function", () => {
    const s = new Square();
    const inner = new Create(s);
    const cs = new ChangeSpeed(inner, { 0: 1, 0.5: 4, 1: 1 });
    try {
      expectClose(cs.run_time, 0.4);
      expectClose(inner.run_time, 1.0);
      expectClose(cs.get_scaled_total_time(), 0.4);
      expect(cs.speedinfo[0]).toBe(1);
      const mapped = [0.0, 0.25, 0.5, 0.75].map((t) => cs.anim.rate_func(t));
      const expected = [0.0, 0.033938923906167505, 0.5, 0.9660610760938325];
      for (let i = 0; i < expected.length; i += 1) {
        expectClose(mapped[i] ?? NaN, expected[i] ?? NaN, 1e-9);
      }
      expectClose(cs.anim.rate_func(1.0), 1.0);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("add_updater routes dt through ChangeSpeed.dt while changing", () => {
    ChangeSpeed.is_changing_dt = true;
    ChangeSpeed.dt = 0;
    try {
      const m = new Square();
      const calls: number[] = [];
      const upd = (mob: Mobject, dt: number): void => {
        void mob;
        calls.push(dt);
      };
      ChangeSpeed.add_updater(m, upd);
      expect(m.get_updaters().length).toBe(1);
      m.update(0.016);
      expect(calls).toEqual([0]);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("add_updater attaches dt-less updaters as-is", () => {
    ChangeSpeed.is_changing_dt = false;
    const m = new Square();
    let seen = 0;
    const upd = (_mob: Mobject): void => {
      seen += 1;
    };
    ChangeSpeed.add_updater(m, upd);
    m.update(0.016);
    expect(seen).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graph GenericGraph._populate_edge_dict / _add_created_vertex
// manim.mobject.graph.Graph._populate_edge_dict
// manim.mobject.graph.DiGraph._populate_edge_dict
// Python:
//   GenericGraph._populate_edge_dict -> NotImplementedError
//   Graph([1,2,3],[(1,2),(2,3)]) edges {(1,2),(2,3)}, 3 vertices
//   _populate_edge_dict([(1,2)], Line) creates a Line edge
//   DiGraph([(1,2)]) edges [(1,2)], edge is a Line (with tip)
//   _add_created_vertex duplicate -> ValueError; new vertex stored + moved
// ---------------------------------------------------------------------------
describe("graph edge dicts and vertex creation", () => {
  it("GenericGraph._populate_edge_dict raises", () => {
    const g = new Graph([1, 2], [[1, 2]] as unknown as [number, number][]);
    expect(() => GenericGraph.prototype._populate_edge_dict.call(g, [], Line)).toThrowError();
  });

  it("Graph._populate_edge_dict builds Line edges", () => {
    const g = new Graph([1, 2, 3], [[1, 2], [2, 3]] as unknown as [number, number][]);
    g._populate_edge_dict([[1, 2], [2, 3]] as unknown as [number, number][], Line);
    expect(g.edges.size).toBe(2);
    expect(g.vertices.size).toBe(3);
    for (const edge of g.edges.values()) {
      expect(edge).toBeInstanceOf(Line);
    }
  });

  it("DiGraph._populate_edge_dict builds directed Line edges with tips", () => {
    const d = new DiGraph([1, 2], [[1, 2]] as unknown as [number, number][]);
    d._populate_edge_dict([[1, 2]] as unknown as [number, number][], Line);
    expect(d.edges.size).toBe(1);
    const edge = [...d.edges.values()][0] as Line;
    expect(edge).toBeInstanceOf(Line);
    expect(edge.get_num_points()).toBeGreaterThan(0);
  });

  it("_add_created_vertex stores, moves and guards duplicates", () => {
    const g = new Graph([1, 2], [[1, 2]] as unknown as [number, number][]);
    const existing = g.vertices.get(1) as Mobject;
    expect(() => g._add_created_vertex(1, [0, 0, 0], {}, existing)).toThrowError(/already used/);
    // Python: graph vertices are Dots (they carry points), so move_to lands
    // the new vertex at the requested position. Give the new vertex points
    // the same way (a Square copy) to mirror that ground truth.
    const created = g._add_created_vertex(99, [1, 2, 3], {}, new Square(1.0));
    expect(g.vertices.get(99)).toBe(created);
    expectRowsClose([[...created.get_center()]], [[1.0, 2.0, 3.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.moving_camera.MovingCamera.capture_mobjects
// Python: frame is a ScreenRectangle, frame_height 8.0,
//   frame_width 14.222222222222221, capture works (pixel array (1080,1920,4)).
// TS port has no cairo backend, so VMobject capture would throw there;
// exercise the batch symbol with a plain Mobject (no-op display branch) and
// check frame geometry + movement indicator instead.
// ---------------------------------------------------------------------------
describe("MovingCamera.capture_mobjects", () => {
  it("captures without cairo and exposes the frame", () => {
    const mc = new MovingCamera();
    expect(mc.frame).toBeInstanceOf(Mobject);
    expectClose(mc.frame_height, 8.0);
    expectClose(mc.frame_width, 14.222222222222221);
    expect(() => mc.capture_mobjects([new Mobject()])).not.toThrow();
    expect(mc.get_mobjects_indicating_movement()).toEqual([mc.frame]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.add_animation_override / animation_override_for
// Python:
//   subclass override registered -> animation_override_for returns it;
//   unknown class -> None; base class -> None;
//   duplicate registration -> MultiAnimationOverrideException
// ---------------------------------------------------------------------------
describe("Mobject animation overrides", () => {
  it("registers, looks up and rejects duplicates", () => {
    class OverrideMob extends Mobject {}
    class SomeAnim extends Animation {}
    const fn = (mob: Mobject): Animation => new Animation(mob);
    OverrideMob.add_animation_override(SomeAnim, fn);
    const inst = new OverrideMob();
    expect(inst.animation_override_for(SomeAnim)).toBe(fn);
    expect(inst.animation_override_for(Animation)).toBeUndefined();
    expect(new Mobject().animation_override_for(SomeAnim)).toBeUndefined();
    expect(() => OverrideMob.add_animation_override(SomeAnim, fn)).toThrowError();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject._UpdaterBuilder.__init__
// manim.mobject.mobject._AnimationBuilder.__init__
// Python: _UpdaterBuilder stores _mobject; _AnimationBuilder stores mobject,
//   generates target, methods == [], is_chaining False.
// ---------------------------------------------------------------------------
describe("builders", () => {
  it("_UpdaterBuilder keeps the mobject", () => {
    const m = new Square();
    const ub = new _UpdaterBuilder(m);
    expect((ub as unknown as { _mobject: Mobject })._mobject).toBe(m);
    expect(m.always).toBeInstanceOf(_UpdaterBuilder);
  });

  it("_AnimationBuilder generates a target and starts empty", () => {
    const m = new Square();
    const ab = m.animate;
    expect(ab).toBeInstanceOf(_AnimationBuilder);
    expect(ab.mobject).toBe(m);
    expect(m.target).not.toBeNull();
    expect(ab.methods).toEqual([]);
    expect(ab.is_chaining).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("ValueTracker drives a MaintainPositionRelativeTo chain", () => {
    // Python: tracker 0 -> 10 interpolates to 3.0 at alpha 0.3; a Square
    // tracking b preserves its offset (mirrors the unit probes above).
    const tracker = new ValueTracker(0.0);
    tracker.interpolate(new ValueTracker(0.0), new ValueTracker(10.0), 0.3);
    expectClose(tracker.get_value(), 3.0);
    const a = new Square(1.0);
    const b = new Square(1.0);
    b.move_to([tracker.get_value(), 0, 0]);
    const anim = new MaintainPositionRelativeTo(a, b);
    expectRowsClose([anim.diff], [[-3.0, 0.0, 0.0]]);
    anim.interpolate_mobject(0.0);
    expectRowsClose([[...a.get_center()]], [[0.0, 0.0, 0.0]]);
  });

  it("Camera.is_in_frame agrees with MovingCamera frame after capture", () => {
    // Python: unit square at origin is in frame for the default camera.
    // (TS has no cairo backend, so capture a Point: PMobjects take the
    // pixel-display path instead of the cairo VMobject path.)
    const mc = new MovingCamera();
    const dot = new Point([0, 0, 0]);
    mc.capture_mobjects([dot]);
    const sq = new Square(1.0);
    expect(mc.is_in_frame(sq)).toBe(true);
    const far = new Square(1.0);
    far.move_to([100, 100, 0]);
    expect(mc.is_in_frame(far)).toBe(false);
  });

  it("ChangeSpeed updater nudges a VMobject along the scaled dt", () => {
    // Python: with is_changing_dt set and ChangeSpeed.dt = 0.5, a wrapped
    // (mob, dt) updater receives 0.5 instead of the passed dt.
    ChangeSpeed.is_changing_dt = true;
    ChangeSpeed.dt = 0.5;
    try {
      const vf = new VectorField((_p) => [1.0, 0.0, 0.0]);
      const c = new Circle(1.0);
      ChangeSpeed.add_updater(c, (mob: Mobject, dt: number) => vf.nudge(mob, dt));
      c.update(0.016);
      expectRowsClose([[...c.get_center()]], [[0.5, 0.0, 0.0]]);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("family extraction feeds restructure for graph-adjacent groups", () => {
    // Python: extract(Group(a,b)) has 3 members; removing a leaves [b].
    const a = new Square();
    const b = new Circle();
    const g = new Group(a, b);
    const members = extract_mobject_family_members([g]);
    expect(members.length).toBe(3);
    const rest = restructure_list_to_exclude_certain_family_members([g], [a]);
    expect(rest).toEqual([b]);
    const vg = new VGroup();
    expect(vg).toBeInstanceOf(Mobject);
  });
});
