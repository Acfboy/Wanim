// Ground-truth tests for batch layers 15..15 (second file: batch-15b).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim) with the SAME inputs used here; each block cites
// the probe in a "Python:" comment. TS is asserted against those values
// (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): no Text/Tex objects are used anywhere here; Matrix
// brackets and BarChart axes numbers need LaTeX so those paths are bypassed
// (documented inline as deviations).
//
// Deviations documented inline:
//  (D1) TS subclasses do NOT auto-run __init_subclass__ (Python calls the hook
//       on class creation; TS has no such hook, it must be invoked explicitly).
//  (D2) _AnimationBuilder override path never fires in EITHER language:
//       the decorator records `_override_animation` but __getattr__ checks
//       `hasattr(method, "_override_animate")` (upstream naming mismatch);
//       the TS port mirrors this faithfully.
//  (D3) TS Matrix._add_brackets throws TODO (needs MathTex); tests use a
//       no-bracket subclass to exercise get_entries.
//  (D4) FunctionGraph.function is the WRAPPED parametric form in both
//       languages (Python's ParametricFunction.__init__ overwrites the raw
//       assignment, so get_function()(2) returns [2,4,0]); the TS port
//       mirrors this exactly.
//  (D5) TS CurvesAsSubmobjects allows empty construction; Python requires the
//       vmobject positional arg (TypeError without it).
//  (D6) TS ImplicitFunction.generate_points throws TODO without a registered
//       isoline computer (needs the `isosurfaces` plot_isoline routine).
//  (D7) TS Mobject.__getitem__ raises TypeError("range object index out of
//       range") for non-integer indices; Python raises TypeError with a
//       different message ("range indices must be integers..."). Name matches.
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { array } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import {
  VMobject,
  VGroup,
  CurvesAsSubmobjects,
  VectorizedPoint,
} from "../src/mobject/types/vectorized_mobject.js";
import {
  PMobject,
  Point,
  PointCloudDot,
} from "../src/mobject/types/point_cloud_mobject.js";
import { Square, Triangle } from "../src/mobject/geometry/polygram.js";
import { Line } from "../src/mobject/geometry/line.js";
import { Table } from "../src/mobject/table.js";
import { Matrix } from "../src/mobject/matrix.js";
import { ManimBanner } from "../src/mobject/logo.js";
import { Brace } from "../src/mobject/svg/brace.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import {
  ParametricFunction,
  FunctionGraph,
  ImplicitFunction,
  setIsolineComputer,
} from "../src/mobject/graphing/functions.js";
import { ImageMobjectFromCamera } from "../src/mobject/types/image_mobject.js";
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

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(actual[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.__init_subclass__
// Python:
//   class Foo(Mobject): pass -> Foo.animation_overrides == {},
//     Foo._original__init__ is Foo.__init__
//   class Bar(Foo): pass -> separate dict, Bar._original__init__ is Bar.__init__
// ---------------------------------------------------------------------------
describe("Mobject.__init_subclass__", () => {
  it("resets the per-class override table and records the constructor", () => {
    class Foo extends Mobject {}
    Foo.__init_subclass__();
    const rec = Foo as unknown as Record<string, unknown>;
    expect(rec["animation_overrides"]).toBeInstanceOf(Map);
    expect((rec["animation_overrides"] as Map<unknown, unknown>).size).toBe(0);
    expect(rec["_original__init__"]).toBe(Foo);
  });

  it("gives subclasses independent tables (D1: explicit call)", () => {
    class Foo2 extends Mobject {}
    Foo2.__init_subclass__();
    class Bar2 extends Foo2 {}
    Bar2.__init_subclass__();
    const f = Foo2 as unknown as Record<string, unknown>;
    const b = Bar2 as unknown as Record<string, unknown>;
    expect(b["animation_overrides"]).not.toBe(f["animation_overrides"]);
    expect(b["_original__init__"]).toBe(Bar2);
    // Mutating the child table does not leak into the parent.
    (b["animation_overrides"] as Map<string, string>).set("k", "v");
    expect((f["animation_overrides"] as Map<string, string>).has("k")).toBe(false);
  });

  it("registers intrinsic overrides (ManimBanner.create for Create)", () => {
    ManimBanner.__init_subclass__();
    const rec = ManimBanner as unknown as {
      animation_overrides: Map<unknown, unknown>;
    };
    expect(rec.animation_overrides.size).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.__getitem__
// Python (Group of 3 squares, no points):
//   len 3; g[0] is subm0; g[-1] is subm-1; g[0:2] Group of 2; g[::2] 2 subs
// Python (Square with 2 square submobjects, has points):
//   len 3; s[0] is s; s[1] is sub0; s[0:2] Group(self, sub0); s[1:3] 2 subs
//   g[10] IndexError; g[1.5] TypeError; g[5:5] empty Group
// ---------------------------------------------------------------------------
describe("Mobject.__getitem__", () => {
  function three(): Group {
    return new Group(new Square(), new Square(), new Square());
  }

  it("indexes submobjects for point-less containers", () => {
    const g = three();
    expect(g.__len__()).toBe(3);
    expect(g.__getitem__(0)).toBe(g.submobjects[0]);
    expect(g.__getitem__(-1)).toBe(g.submobjects[2]);
    const sl = g.__getitem__({ start: 0, stop: 2, step: null }) as Group;
    expect(sl).toBeInstanceOf(Group);
    expect(sl.submobjects.length).toBe(2);
    const stepped = g.__getitem__({ start: null, stop: null, step: 2 }) as Group;
    expect(stepped.submobjects.length).toBe(2);
  });

  it("indexes self at 0 for mobjects with points", () => {
    const s = new Square();
    s.add(new Square(), new Square());
    expect(s.__len__()).toBe(3);
    expect(s.__getitem__(0)).toBe(s);
    expect(s.__getitem__(1)).toBe(s.submobjects[0]);
    expect(s.__getitem__(-1)).toBe(s.submobjects[1]);
    const sl = s.__getitem__({ start: 0, stop: 2, step: null }) as Group;
    expect(sl.submobjects.length).toBe(2);
    expect(sl.submobjects[0]).toBe(s);
    expect(sl.submobjects[1]).toBe(s.submobjects[0]);
    const tail = s.__getitem__({ start: 1, stop: 3, step: null }) as Group;
    expect(tail.submobjects.length).toBe(2);
    expect(tail.submobjects[0]).toBe(s.submobjects[0]);
  });

  it("raises IndexError out of range, TypeError for non-integer (D7)", () => {
    const g = three();
    expect(() => g.__getitem__(10)).toThrowError(
      expect.objectContaining({ name: "IndexError" }),
    );
    expect(() => g.__getitem__(1.5 as unknown as number)).toThrowError(
      expect.objectContaining({ name: "TypeError" }),
    );
    const empty = g.__getitem__({ start: 5, stop: 5, step: null }) as Group;
    expect(empty.submobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject._AnimationBuilder.__getattr__
// Python:
//   b = Square().animate; b.shift callable; is_chaining/cannot_pass_args True
//   after f([1,0,0]): target center [1,0,0]
//   b.nonexistent_method_xyz -> AttributeError
//   (D2) override path never fires (checks _override_animate, absent)
// ---------------------------------------------------------------------------
describe("_AnimationBuilder.__getattr__", () => {
  it("resolves a method into an updater that applies to the target", () => {
    const sq = new Square();
    const b = sq.animate;
    const f = b.__getattr__("shift");
    expect(typeof f).toBe("function");
    expect(b.is_chaining).toBe(true);
    expect(b.cannot_pass_args).toBe(true);
    (f as (...a: unknown[]) => unknown)([1, 0, 0]);
    expectRowsClose([(sq.target as Mobject).get_center()], [[1, 0, 0]]);
    // Original untouched.
    expectRowsClose([sq.get_center()], [[0, 0, 0]]);
    expect(b.methods.length).toBe(1);
  });

  it("is reachable through property access on the builder proxy", () => {
    const sq = new Square();
    const b = sq.animate as unknown as Record<string, unknown>;
    const shift = b["shift"] as (...a: unknown[]) => unknown;
    expect(typeof shift).toBe("function");
    shift([0, 2, 0]);
    expectRowsClose([(sq.target as Mobject).get_center()], [[0, 2, 0]]);
  });

  it("raises AttributeError for missing methods", () => {
    const sq = new Square();
    expect(() => sq.animate.__getattr__("nonexistent_method_xyz")).toThrowError(
      expect.objectContaining({ name: "AttributeError" }),
    );
  });

  it("treats overrides as plain methods (D2 upstream quirk)", () => {
    const mb = new ManimBanner();
    const b = mb.animate;
    // `create` IS registered as an intrinsic override for Create, but the
    // builder checks `_override_animate` (absent), so it records a plain
    // method call instead of building an overridden animation.
    const f = b.__getattr__("create");
    expect(typeof f).toBe("function");
    (f as (...a: unknown[]) => unknown)();
    expect(b.overridden_animation).toBeNull();
    expect(b.methods.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table._add_labels
// Python (bypassing the Tex __init__ via __new__):
//   row(2)+col(2)+topleft over 2x2 -> 3 rows x 3 cols
//   row(1)+col(2), no topleft -> 2 rows, first row len 3, [0][0] dummy VMobject
//   col-only over 1x2 -> 2 rows; no labels -> identity (same object)
// ---------------------------------------------------------------------------
describe("Table._add_labels", () => {
  function bareTable(): Table {
    // NOTE: mirrors Python `Table.__new__(Table)` (skips the Tex __init__).
    const t = Object.create(Table.prototype) as Table;
    t.row_labels = null;
    t.col_labels = null;
    t.top_left_entry = null;
    return t;
  }

  it("prepends row labels and a top-left-headed column row", () => {
    const t = bareTable();
    t.row_labels = [new Square(), new Square()];
    t.col_labels = [new Square(), new Square()];
    t.top_left_entry = new Square();
    const mobTable = [
      [new Square(), new Square()],
      [new Square(), new Square()],
    ];
    const out = t._add_labels(mobTable);
    expect(out.length).toBe(3);
    expect(out[0]?.length).toBe(3);
    expect(out[1]?.length).toBe(3);
    expect(out[2]?.length).toBe(3);
    expect(out[1]?.[0]).toBe(t.row_labels[0]);
    expect(out[2]?.[0]).toBe(t.row_labels[1]);
    expect(out[0]?.[0]).toBe(t.top_left_entry);
  });

  it("inserts a dummy placeholder without a top-left entry", () => {
    const t = bareTable();
    t.row_labels = [new Square()];
    t.col_labels = [new Square(), new Square()];
    t.top_left_entry = null;
    const out = t._add_labels([[new Square(), new Square()]]);
    expect(out.length).toBe(2);
    expect(out[0]?.length).toBe(3);
    expect(out[0]?.[0]).toBeInstanceOf(VMobject);
    expect(out[1]?.[0]).toBe(t.row_labels[0]);
  });

  it("handles column-only labels and the no-label identity", () => {
    const t = bareTable();
    t.col_labels = [new Square(), new Square()];
    const out = t._add_labels([[new Square(), new Square()]]);
    expect(out.length).toBe(2);
    const bare = bareTable();
    const m: VMobject[][] = [[new Square()]];
    expect(bare._add_labels(m)).toBe(m);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.get_horizontal_lines / get_vertical_lines /
//   get_entries_without_labels
// Python: both getters return the stored VGroup (identity); pos (1,1) -> e0,
//   (2,2) -> e3 with col_dim 2; no pos -> the whole VGroup.
// ---------------------------------------------------------------------------
describe("Table line/entry getters", () => {
  function tableWithEntries(): Table {
    const t = new Table();
    t.col_dim = 2;
    const subs = [new Square(), new Square(), new Square(), new Square()];
    subs.forEach((s, i) => {
      s.name = `e${i}`;
    });
    t.elements_without_labels = new VGroup(...subs);
    t.horizontal_lines = new VGroup(new Square());
    t.vertical_lines = new VGroup(new Square(), new Square());
    return t;
  }

  it("returns the stored line groups by identity", () => {
    const t = tableWithEntries();
    expect(t.get_horizontal_lines()).toBe(t.horizontal_lines);
    expect(t.get_vertical_lines()).toBe(t.vertical_lines);
    expect(t.get_vertical_lines().submobjects.length).toBe(2);
  });

  it("indexes entries 1-based without labels", () => {
    const t = tableWithEntries();
    expect((t.get_entries_without_labels([1, 1]) as VMobject).name).toBe("e0");
    expect((t.get_entries_without_labels([2, 2]) as VMobject).name).toBe("e3");
    expect((t.get_entries_without_labels([1, 2]) as VMobject).name).toBe("e1");
    const all = t.get_entries_without_labels() as VGroup;
    expect(all).toBe(t.elements_without_labels);
    expect(all.submobjects.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.Matrix.get_entries
// Python (MobjectMatrix with _add_brackets patched out, LaTeX missing):
//   get_entries() is elements, 4 entries for a 2x2.
// ---------------------------------------------------------------------------
class NoBracketMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

describe("Matrix.get_entries", () => {
  it("returns the elements VGroup by identity (D3: brackets need MathTex)", () => {
    const m = new NoBracketMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: ((item: unknown) => item) as never },
    );
    expect(m.get_entries()).toBe(m.elements);
    expect(m.get_entries().submobjects.length).toBe(4);
    expect(m.get_mob_matrix().length).toBe(2);
    expect(m.get_mob_matrix()[0]?.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.logo.ManimBanner (+ scale)
// Python: scale_factor 1.0, 2 submobjects, width 5.699088878784438,
//   height 3.6942714000000003, center [0,0,0]
//   scale(2): factor 2.0, width doubles; scale(0.5): back to 1.0/width;
//   anim width scales with the banner.
// ---------------------------------------------------------------------------
describe("ManimBanner", () => {
  it("builds the logo at the origin with ground-truth size", () => {
    const b = new ManimBanner();
    expectClose(b.scale_factor, 1.0);
    expect(b.submobjects.length).toBe(2);
    expectClose(b.width, 5.699088878784438);
    expectClose(b.height, 3.6942714000000003);
    expectRowsClose([b.get_center()], [[0, 0, 0]]);
  });

  it("scales the factor, geometry and hidden anim together", () => {
    const b = new ManimBanner();
    const w0 = b.width;
    const animW0 = b.anim.width;
    b.scale(2);
    expectClose(b.scale_factor, 2.0);
    expectClose(b.width, w0 * 2);
    expectClose(b.anim.width, animW0 * 2);
    b.scale(0.5);
    expectClose(b.scale_factor, 1.0);
    expectClose(b.width, w0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.add_quadratic_bezier_curve_to
// Python: start [0,0,0], handle [1,1,0], anchor [2,0,0] ->
//   [[0,0,0],[2/3,2/3,0],[4/3,2/3,0],[2,0,0]]
// ---------------------------------------------------------------------------
describe("VMobject.add_quadratic_bezier_curve_to", () => {
  it("elevates the quadratic to a cubic with 2/3 handles", () => {
    const v = new VMobject();
    v.start_new_path([0, 0, 0]);
    v.add_quadratic_bezier_curve_to([1, 1, 0], [2, 0, 0]);
    expect(v.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0, 0, 0],
      [2 / 3, 2 / 3, 0],
      [4 / 3, 2 / 3, 0],
      [2, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.set_points_smoothly
// Python: [[0,0,0],[1,2,0],[2,0,0],[3,1,0]] -> 12 points, exact rows below.
// ---------------------------------------------------------------------------
describe("VMobject.set_points_smoothly", () => {
  it("corners then smooths to the ground-truth handles", () => {
    const v = new VMobject();
    v.set_points_smoothly([
      [0, 0, 0],
      [1, 2, 0],
      [2, 0, 0],
      [3, 1, 0],
    ]);
    expect(v.get_num_points()).toBe(12);
    expectRowsClose(rowsOf(v.points, v.dim), [
      [0.0, 0.0, 0.0],
      [0.33333333333333326, 1.088888888888889, 0.0],
      [0.6666666666666665, 2.177777777777778, 0.0],
      [1.0, 2.0, 0.0],
      [1.0, 2.0, 0.0],
      [1.3333333333333335, 1.8222222222222222, 0.0],
      [1.6666666666666665, 0.37777777777777777, 0.0],
      [2.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [2.3333333333333335, -0.37777777777777777, 0.0],
      [2.666666666666667, 0.3111111111111111, 0.0],
      [3.0, 1.0, 0.0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_point_mobject
// Python (Square): default center [0,0,0], VectorizedPoint; explicit [5,6,0].
// ---------------------------------------------------------------------------
describe("VMobject.get_point_mobject", () => {
  it("returns a style-matched VectorizedPoint at the center", () => {
    const s = new Square();
    const pm = s.get_point_mobject();
    expect(pm).toBeInstanceOf(VectorizedPoint);
    expectRowsClose(rowsOf(pm.points, pm.dim), [[0, 0, 0]]);
  });

  it("honors an explicit center", () => {
    const s = new Square();
    const pm = s.get_point_mobject([5, 6, 0]);
    expectRowsClose(rowsOf(pm.points, pm.dim), [[5, 6, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VGroup.__iadd__
// Python: returns self, appends the vmobject.
// ---------------------------------------------------------------------------
describe("VGroup.__iadd__", () => {
  it("adds in place and returns the same group", () => {
    const g = new VGroup();
    const sq = new Square();
    const ret = g.__iadd__(sq);
    expect(ret).toBe(g);
    expect(g.submobjects.length).toBe(1);
    expect(g.submobjects[0]).toBe(sq);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.CurvesAsSubmobjects
// Python (Line [0,0,0]->[4,0,0]): 1 submobject of 4 pts;
//   pfp(0) [0,0,0], pfp(0.5) [2,0,0], pfp(1) [4,0,0];
//   alpha 2 -> ValueError. (D5) empty Python construction raises TypeError.
// ---------------------------------------------------------------------------
describe("CurvesAsSubmobjects", () => {
  it("splits a line into per-curve submobjects with arc-length lookup", () => {
    const line = new Line([0, 0, 0], [4, 0, 0]);
    const c = new CurvesAsSubmobjects(line);
    expect(c.submobjects.length).toBe(1);
    expect((c.submobjects[0] as VMobject).get_num_points()).toBe(4);
    expectRowsClose([[...(c.point_from_proportion(0).toArray() as number[])]], [
      [0, 0, 0],
    ]);
    expectRowsClose([[...(c.point_from_proportion(0.5).toArray() as number[])]], [
      [2, 0, 0],
    ]);
    expectRowsClose([[...(c.point_from_proportion(1).toArray() as number[])]], [
      [4, 0, 0],
    ]);
  });

  it("rejects out-of-range alpha and empty lookups", () => {
    const c = new CurvesAsSubmobjects(new Line([0, 0, 0], [4, 0, 0]));
    expect(() => c.point_from_proportion(2)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    // (D5) TS permits empty construction where Python requires vmobject.
    const empty = new CurvesAsSubmobjects();
    expect(() => empty.point_from_proportion(0.5)).toThrowError(/no submobjects/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PMobject.get_point_mobject
// Python: PMobject with [[1,2,3],[4,5,6]] -> Point at [2.5,3.5,4.5];
//   explicit [7,8,9].
// ---------------------------------------------------------------------------
describe("PMobject.get_point_mobject", () => {
  it("returns a Point at the cloud center by default", () => {
    const p = new PMobject();
    p.add_points(
      array([
        [1, 2, 3],
        [4, 5, 6],
      ]) as NDArray,
    );
    const pm = p.get_point_mobject();
    expect(pm).toBeInstanceOf(Point);
    expectRowsClose(rowsOf(pm.points, pm.dim), [[2.5, 3.5, 4.5]]);
    const explicit = p.get_point_mobject([7, 8, 9]);
    expectRowsClose(rowsOf(explicit.points, explicit.dim), [[7, 8, 9]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.PointCloudDot
// Python: default radius 2.0, epsilon 0.1, 1303 points;
//   center [1,0,0] radius 1 density 10 -> 334 points, center ~[1,0,0].
// ---------------------------------------------------------------------------
describe("PointCloudDot", () => {
  it("builds the default disc with ground-truth density", () => {
    const d = new PointCloudDot();
    expectClose(d.radius, 2.0);
    expectClose(d.epsilon, 0.1);
    expect(d.get_num_points()).toBe(1303);
  });

  it("shifts the disc to the given center", () => {
    const d = new PointCloudDot([1, 0, 0], 1.0, 2, 10);
    expect(d.get_num_points()).toBe(334);
    expectRowsClose([d.get_center()], [[1.0005966595518263, 0, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.image_mobject.ImageMobjectFromCamera.get_pixel_array
// Python: returns camera.pixel_array (identity); later camera mutations are
//   visible through a second call.
// ---------------------------------------------------------------------------
describe("ImageMobjectFromCamera.get_pixel_array", () => {
  it("returns the live camera array", () => {
    const pixels = array([
      [
        [10, 20, 30, 255],
        [0, 0, 0, 255],
        [1, 2, 3, 255],
      ],
      [
        [4, 5, 6, 255],
        [7, 8, 9, 255],
        [11, 12, 13, 255],
      ],
    ]) as NDArray;
    const camera = { pixel_array: pixels } as never;
    const im = new ImageMobjectFromCamera(camera);
    expect(im.get_pixel_array()).toBe(pixels);
    expect((im.get_pixel_array().toArray() as number[][][])[0]?.[0]).toEqual([
      10, 20, 30, 255,
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.Brace (+ __init__)
// Python (Brace(Square())): 100 points, tip [0,-1.472985,~0],
//   direction [0,-1,~0], buff 0.2.
// Python (direction [1,0,0], buff 0.5): tip [1.772985,~0,~0].
// ---------------------------------------------------------------------------
describe("Brace", () => {
  it("wraps a square with ground-truth tip and direction", () => {
    const b = new Brace(new Square());
    expect(b.get_num_points()).toBe(100);
    expectRowsClose([b.get_tip()], [[0, -1.472985, 0]], 1e-6);
    const dir = b.get_direction() as number[];
    expectRowsClose([dir], [[0, -1, 0]], 1e-6);
    expectClose(b.buff, 0.2);
  });

  it("rotates the frame for an explicit direction", () => {
    const b = new Brace(new Square(), { direction: [1, 0, 0], buff: 0.5 });
    expect(b.get_num_points()).toBe(100);
    expectRowsClose([b.get_tip()], [[1.772985, 0, 0]], 1e-6);
    expectClose(b.buff, 0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.probability.BarChart._update_colors
// Python (bars = 3 squares, default palette): _update_colors matches a direct
//   set_color_by_gradient: [#003F5C, #BC5090, #FFA600].
// NOTE: the full BarChart __init__ needs LaTeX axes numbers, so both sides
//   build the bars manually here; the TS constructor itself is latex-free.
// ---------------------------------------------------------------------------
describe("BarChart._update_colors", () => {
  it("paints bars from the palette gradient", () => {
    const bc = new BarChart([1, 2, 3]);
    bc.bars = new VGroup(new Square(), new Square(), new Square());
    bc._update_colors();
    const hexes = bc.bars.submobjects.map((m) =>
      (m as VMobject).get_fill_color().to_hex(),
    );
    expect(hexes).toEqual(["#003F5C", "#BC5090", "#FFA600"]);
    const ref = new VGroup(new Square(), new Square(), new Square());
    ref.set_color_by_gradient(...bc.bar_colors);
    const refHexes = ref.submobjects.map((m) =>
      (m as VMobject).get_fill_color().to_hex(),
    );
    expect(hexes).toEqual(refHexes);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.ParametricFunction.generate_points
// Python (t -> [t, t^2, 0], t_range [0,1,0.5], smoothing on): 8 points,
//   exact rows below. With discontinuities=[0.5], default step: 400 points.
// ---------------------------------------------------------------------------
describe("ParametricFunction.generate_points", () => {
  it("samples and smooths to the ground-truth points", () => {
    const pf = new ParametricFunction((t: number) => [t, t ** 2, 0], {
      t_range: [0, 1, 0.5],
    });
    expect(pf.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(pf.points, pf.dim), [
      [0.0, 0.0, 0.0],
      [0.16666666666666669, 0.041666666666666685, 0.0],
      [0.33333333333333337, 0.08333333333333337, 0.0],
      [0.5, 0.25, 0.0],
      [0.5, 0.25, 0.0],
      [0.6666666666666666, 0.41666666666666663, 0.0],
      [0.8333333333333333, 0.7083333333333333, 0.0],
      [1.0, 1.0, 0.0],
    ]);
  });

  it("splits the range at discontinuities", () => {
    const pf = new ParametricFunction((t: number) => [t, 0, 0], {
      t_range: [0, 1],
      discontinuities: [0.5],
    });
    expect(pf.get_num_points()).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.FunctionGraph
// Python (x^2, x_range [-1,1,0.5]): 16 points, first [-1,1,0],
//   last [1,1,0]; get_point_from_function(2) == [2,4,0].
//   (D4) function/get_function() is the wrapped parametric form, so
//   get_function()(2) == [2,4,0] in both languages.
// ---------------------------------------------------------------------------
describe("FunctionGraph", () => {
  it("plots x^2 with ground-truth endpoints", () => {
    const fg = new FunctionGraph((x: number) => x ** 2, [-1, 1, 0.5]);
    expect(fg.get_num_points()).toBe(16);
    expectRowsClose([rowsOf(fg.points, fg.dim)[0] ?? []], [[-1, 1, 0]]);
    const all = rowsOf(fg.points, fg.dim);
    expectRowsClose([all[all.length - 1] ?? []], [[1, 1, 0]]);
    expectRowsClose(
      [rowsOf(fg.get_point_from_function(2), 3)[0] ?? []],
      [[2, 4, 0]],
    );
    expectRowsClose(
      [rowsOf(fg.get_function()(2), 3)[0] ?? []],
      [[2, 4, 0]],
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.ImplicitFunction.generate_points
// (D6) without a computer both... actually only TS throws TODO; Python
//   computes a 784-point circle. TS wiring is verified with a stub computer.
// ---------------------------------------------------------------------------
describe("ImplicitFunction.generate_points", () => {
  it("defers without a registered isoline computer (D6)", () => {
    // NOTE: Python always runs plot_isoline in __init__; the TS port defers
    // (construction yields an empty mobject) and only an explicit
    // generate_points() call raises the TODO.
    setIsolineComputer(null);
    const im = new ImplicitFunction(
      (x: number, y: number) => x ** 2 + y ** 2 - 1,
      {
        x_range: [-2, 2],
        y_range: [-2, 2],
        min_depth: 3,
        max_quads: 200,
        use_smoothing: false,
      },
    );
    expect(im.get_num_points()).toBe(0);
    expect(() => im.generate_points()).toThrowError(/plot_isoline/);
  });

  it("pads stub isolines with z=0 and lays them as corners", () => {
    setIsolineComputer(() => [
      [
        [0, 0],
        [1, 0],
      ],
      [],
    ]);
    try {
      const im = new ImplicitFunction(
        (x: number, y: number) => x + y,
        {
          x_range: [-2, 2],
          y_range: [-2, 2],
          min_depth: 3,
          max_quads: 200,
          use_smoothing: false,
        },
      );
      // One non-empty curve of 2 corners: start_new_path lays 1 point, then
      // add_points_as_corners replaces the open path with one 4-point line.
      expect(im.get_num_points()).toBe(4);
      expectRowsClose([rowsOf(im.points, im.dim)[0] ?? []], [[0, 0, 0]]);
      const all = rowsOf(im.points, im.dim);
      expectRowsClose([all[all.length - 1] ?? []], [[1, 0, 0]]);
    } finally {
      setIsolineComputer(null);
    }
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("batch-15b integration", () => {
  it("animate-builder shift matches a direct shift (builder + VMobject)", () => {
    // Python: builder applies method to mobject.target; build() then finishes
    // onto the mobject. Both routes must agree.
    const direct = new Square();
    direct.shift([1, 2, 0]);
    const via = new Square();
    const b = via.animate;
    (b.__getattr__("shift") as (...a: unknown[]) => unknown)([1, 2, 0]);
    const anim = b.build();
    anim.begin();
    anim.finish();
    expectRowsClose(
      rowsOf(via.points, via.dim),
      rowsOf(direct.points, direct.dim),
    );
  });

  it("Table labels plus label-free lookup stay consistent", () => {
    // _add_labels grows the grid while elements_without_labels keeps the
    // raw 1-based indexing (col_dim 2): (2,1) -> e2.
    const t = new Table();
    t.col_dim = 2;
    t.row_labels = [new Square()];
    t.col_labels = [new Square(), new Square()];
    t.top_left_entry = new Square();
    const grid = t._add_labels([[new Square(), new Square()]]);
    expect(grid.length).toBe(2);
    expect(grid[0]?.length).toBe(3);
    t.elements_without_labels = new VGroup(new Square(), new Square());
    expect(
      t.get_entries_without_labels(),
    ).toBe(t.elements_without_labels);
  });

  it("ParametricFunction -> CurvesAsSubmobjects -> point lookup agrees", () => {
    // Python: CurvesAsSubmobjects(Line).point_from_proportion(0.5) == [2,0,0]
    // equals Line.point_from_proportion(0.5); the same chain works for a
    // ParametricFunction line.
    const pf = new ParametricFunction((t: number) => [4 * t, 0, 0], {
      t_range: [0, 1, 0.25],
      use_smoothing: false,
    });
    const curves = new CurvesAsSubmobjects(pf);
    expect(curves.submobjects.length).toBeGreaterThan(0);
    const mid = [...(curves.point_from_proportion(0.5).toArray() as number[])];
    expectRowsClose([mid], [[2, 0, 0]], 1e-6);
    const dot = pf.get_point_mobject(mid);
    expectRowsClose(rowsOf(dot.points, dot.dim), [[2, 0, 0]], 1e-6);
  });

  it("VGroup iadd plus getitem round-trips the added member", () => {
    const g = new VGroup();
    const a = new Square();
    const b = new Triangle();
    g.__iadd__(a);
    g.__iadd__(b);
    expect(g.__getitem__(0)).toBe(a);
    expect(g.__getitem__(1)).toBe(b);
    expect(g.__getitem__(-1)).toBe(b);
  });
});
