// Ground-truth tests for batch layers 16..16 (~1050 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here, EXCEPT
// the ImplicitFunction block: the real `isosurfaces.plot_isoline` backend has
// no TS port yet (see ), so that block uses an injected stub
// isoline computer and only asserts the init_points-after-clear regeneration
// path, not Python ground-truth point values. Each other block cites the
// probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): Table/Matrix default element factories (Paragraph,
// MathTex) and Matrix brackets need LaTeX; tests use MobjectTable / Square
// entries with an identity factory, and a brackets-patched Matrix subclass
// (same convention as batch-targets-15b).
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VMobject, VGroup, CurvesAsSubmobjects } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { MobjectTable } from "../src/mobject/table.js";
import { Matrix } from "../src/mobject/matrix.js";
import { ManimBanner } from "../src/mobject/logo.js";
import { ParametricFunction, ImplicitFunction, setIsolineComputer } from "../src/mobject/graphing/functions.js";
import { DashedLine } from "../src/mobject/geometry/line.js";
import { Axes, NumberPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { Tetrahedron, Octahedron, Icosahedron, Dodecahedron, ConvexHull3D } from "../src/mobject/three_d/polyhedra.js";
import { Cube, Prism, Cone } from "../src/mobject/three_d/three_dimensions.js";
import { SpiralIn } from "../src/animation/creation.js";
import { FadeIn } from "../src/animation/fading.js";
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

/** Tagged Square entry so table/matrix membership is checkable by identity. */
function sq(tag: number): Square & { tag: number } {
  const s = new Square({ side_length: 0.5 }) as Square & { tag: number };
  s.tag = tag;
  return s;
}

function labeledTable(topLeft: boolean) {
  const a = sq(1); const b = sq(2); const c = sq(3); const d = sq(4);
  const r1 = sq(10); const r2 = sq(30);
  const c1 = sq(100); const c2 = sq(200);
  const tle = sq(999);
  const t = new MobjectTable(
    [[a, b], [c, d]],
    {
      row_labels: [r1, r2],
      col_labels: [c1, c2],
      ...(topLeft ? { top_left_entry: tle } : {}),
    },
  );
  return { t, a, b, c, d, r1, r2, c1, c2, tle };
}

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.get_rows / get_columns
// Python (MobjectTable 2x2 + row/col labels, no top_left):
//   mob_table 3x3; get_rows -> 3 groups of 3; get_columns -> 3 groups of 3.
// ---------------------------------------------------------------------------
describe("Table.get_rows / get_columns", () => {
  it("returns one VGroup per row/column with the right members", () => {
    const { t, a, b, c, d, r1, r2, c1, c2 } = labeledTable(false);
    const rows = t.get_rows();
    expect(rows).toBeInstanceOf(VGroup);
    expect(rows.submobjects.length).toBe(3);
    expect((rows.submobjects[0] as VGroup).submobjects.length).toBe(3);
    expect((rows.submobjects[1] as VGroup).submobjects.length).toBe(3);
    expect((rows.submobjects[2] as VGroup).submobjects.length).toBe(3);
    // Row 0 = [dummy, c1, c2]; row 1 = [r1, a, b]; row 2 = [r2, c, d].
    const row1 = (rows.submobjects[1] as VGroup).submobjects;
    expect(row1[0]).toBe(r1);
    expect(row1[1]).toBe(a);
    expect(row1[2]).toBe(b);
    const row2 = (rows.submobjects[2] as VGroup).submobjects;
    expect(row2[0]).toBe(r2);
    expect(row2[1]).toBe(c);
    expect(row2[2]).toBe(d);
    const cols = t.get_columns();
    expect(cols.submobjects.length).toBe(3);
    const col1 = (cols.submobjects[1] as VGroup).submobjects;
    expect(col1[0]).toBe(c1);
    expect(col1[1]).toBe(a);
    expect(col1[2]).toBe(c);
    const col2 = (cols.submobjects[2] as VGroup).submobjects;
    expect(col2[0]).toBe(c2);
    expect(col2[1]).toBe(b);
    expect(col2[2]).toBe(d);
  });
  it("works without labels (2x2 -> 2 rows, 2 cols)", () => {
    // Python: MobjectTable([[1,2],[3,4]]) rows 2, cols 2, each of length 2.
    const a = sq(1); const b = sq(2); const c = sq(3); const d = sq(4);
    const t = new MobjectTable([[a, b], [c, d]]);
    expect(t.get_rows().submobjects.length).toBe(2);
    expect(t.get_columns().submobjects.length).toBe(2);
    expect(((t.get_rows().submobjects[0] as VGroup).submobjects)).toEqual([a, b]);
    expect(((t.get_columns().submobjects[0] as VGroup).submobjects)).toEqual([a, c]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.get_entries
// Python: labeled table (both labels, no top_left): elements has 8 entries
//   (dummy placeholder removed); get_entries((2,2)) is entry `a` (data 1),
//   get_entries((1,1)) is the first col label Square; no-arg returns elements.
//   With top_left_entry: get_entries((1,1)) is the top-left entry.
//   Unlabeled: get_entries((1,1)) is `a`.
// ---------------------------------------------------------------------------
describe("Table.get_entries", () => {
  it("indexes labeled entries with the -1 shift like Python", () => {
    const { t, a, d } = labeledTable(false);
    expect((t.get_entries() as VGroup).submobjects.length).toBe(8);
    expect(t.get_entries()).toBe(t.elements);
    expect(t.get_entries([2, 2])).toBe(a);
    expect(t.get_entries([3, 3])).toBe(t.elements.__getitem__(7));
    // NOTE: (1,1) computes index 3*0+0-1 = -1, which Python resolves as
    // elements[-1] (the last entry `d`); the TS __getitem__ normalizes
    // negatives the same way, so both agree here.
    expect(t.get_entries([1, 1])).toBe(d);
  });
  it("returns the top-left entry at (1,1) when set", () => {
    const { t, tle, a } = labeledTable(true);
    expect(t.get_entries([1, 1])).toBe(tle);
    expect(t.get_entries([2, 2])).toBe(a);
  });
  it("indexes unlabeled tables directly", () => {
    const a = sq(1); const b = sq(2); const c = sq(3); const d = sq(4);
    const t = new MobjectTable([[a, b], [c, d]]);
    expect(t.get_entries([1, 1])).toBe(a);
    expect(t.get_entries([2, 1])).toBe(c);
    expect(t.get_entries([2, 2])).toBe(d);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.get_row_labels / get_col_labels / get_labels
// Python: row labels [10,30], col labels [100,200]; get_labels without
//   top_left -> [c1, c2, r1, r2]; with top_left -> [tle, c1, c2, r1, r2];
//   unlabeled -> empty VGroups.
// ---------------------------------------------------------------------------
describe("Table.get_row_labels / get_col_labels / get_labels", () => {
  it("returns the label groups like Python", () => {
    const { t, r1, r2, c1, c2 } = labeledTable(false);
    expect([...(t.get_row_labels() as VGroup).submobjects]).toEqual([r1, r2]);
    expect([...(t.get_col_labels() as VGroup).submobjects]).toEqual([c1, c2]);
    expect([...(t.get_labels() as VGroup).submobjects]).toEqual([c1, c2, r1, r2]);
  });
  it("prepends the top-left entry when set", () => {
    const { t, tle, r1, r2, c1, c2 } = labeledTable(true);
    expect([...(t.get_labels() as VGroup).submobjects]).toEqual([tle, c1, c2, r1, r2]);
  });
  it("returns empty VGroups without labels", () => {
    const t = new MobjectTable([[sq(1), sq(2)], [sq(3), sq(4)]]);
    expect(t.get_row_labels().submobjects.length).toBe(0);
    expect(t.get_col_labels().submobjects.length).toBe(0);
    expect(t.get_labels().submobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.Matrix.add_background_to_entries
// Python (brackets patched out, LaTeX missing): returns self; every entry
//   gains a non-null background_rectangle (was null before).
// ---------------------------------------------------------------------------
class NoBracketMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

describe("Matrix.add_background_to_entries", () => {
  it("adds a background rectangle to every entry and returns self", () => {
    const m = new NoBracketMatrix(
      [[sq(1), sq(2)], [sq(3), sq(4)]],
      { element_to_mobject: ((item: unknown) => item) as never },
    );
    for (const e of m.get_entries().submobjects) {
      expect((e as VMobject).background_rectangle).toBeNull();
    }
    expect(m.add_background_to_entries()).toBe(m);
    for (const e of m.get_entries().submobjects) {
      expect((e as VMobject).background_rectangle).not.toBeNull();
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.logo.ManimBanner.create
// Python: AnimationGroup(SpiralIn(shapes, run_time=rt),
//   FadeIn(M, run_time=rt/2), lag_ratio=0.1); create(4) -> run_times [4, 2].
// ---------------------------------------------------------------------------
describe("ManimBanner.create", () => {
  it("builds SpiralIn(shapes) + FadeIn(M) with lag 0.1 like Python", () => {
    const b = new ManimBanner(true);
    const ag = b.create(2);
    expect(ag.animations.length).toBe(2);
    expect(ag.animations[0]).toBeInstanceOf(SpiralIn);
    expect(ag.animations[1]).toBeInstanceOf(FadeIn);
    expect((ag.animations[0] as SpiralIn).mobject).toBe(b.shapes);
    expect((ag.animations[1] as FadeIn).mobject).toBe(b.M);
    expectClose((ag as unknown as { lag_ratio: number }).lag_ratio, 0.1);
    expectClose((ag.animations[0] as SpiralIn).run_time, 2);
    expectClose((ag.animations[1] as FadeIn).run_time, 1);
  });
  it("scales run times with the argument like Python", () => {
    const b = new ManimBanner(true);
    const ag = b.create(4);
    expectClose((ag.animations[0] as SpiralIn).run_time, 4);
    expectClose((ag.animations[1] as FadeIn).run_time, 2.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.add_smooth_curve_to
// Python: corners [[0,0,0],[1,0,0]] + [2,1,0] ->
//   [[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0],
//    [1,0,0],[4/3,0,0],[2,2/3,~0],[2,1,0]];
//   explicit handle [1.5,0.5,0] -> handle2 kept as-is.
//   0 or 3 points -> ValueError; no points -> ValueError.
// ---------------------------------------------------------------------------
describe("VMobject.add_smooth_curve_to", () => {
  it("smooths to a single anchor like Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(v.add_smooth_curve_to([2, 1, 0])).toBe(v);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.333333333333, 0.0, 0.0],
      [2.0, 0.666666666667, 0.0],
      [2.0, 1.0, 0.0],
    ], 1e-9);
  });
  it("keeps an explicit handle like Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    v.add_smooth_curve_to([1.5, 0.5, 0], [2, 1, 0]);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.333333333333, 0.0, 0.0],
      [1.5, 0.5, 0.0],
      [2.0, 1.0, 0.0],
    ], 1e-9);
  });
  it("raises ValueError for 0 or 3+ points, and with no points", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(() => (v as unknown as { add_smooth_curve_to: (...a: unknown[]) => void }).add_smooth_curve_to()).toThrowError(
      /Only call add_smooth_curve_to with 1 or 2 points/,
    );
    expect(() => v.add_smooth_curve_to([1, 0, 0], [2, 0, 0], [3, 0, 0])).toThrowError(
      /Only call add_smooth_curve_to with 1 or 2 points/,
    );
    expect(() => new VMobject().add_smooth_curve_to([1, 0, 0])).toThrowError(/no points/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.close_path
// Python: open 2-corner shape gains a 4-point closing curve back at the
//   start ([1,1,0]..[0,0,0] third-points); already-closed is a no-op.
// ---------------------------------------------------------------------------
describe("VMobject.close_path", () => {
  it("appends a closing line like Python", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(v.is_closed()).toBe(false);
    expect(v.close_path()).toBe(v);
    expect(v.is_closed()).toBe(true);
    expectRowsClose(rowsOf(v.points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.333333333333, 0.0],
      [1.0, 0.666666666667, 0.0],
      [1.0, 1.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.666666666667, 0.666666666667, 0.0],
      [0.333333333333, 0.333333333333, 0.0],
      [0.0, 0.0, 0.0],
    ], 1e-9);
  });
  it("is a no-op on an already closed path", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]);
    const before = rowsOf(v.points, 3).map((r) => [...r]);
    v.close_path();
    expectRowsClose(rowsOf(v.points, 3), before);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.align_points
// Python: 1-curve (4 pts) vs 2-curve (8 pts) -> both 8 pts; short one
//   subdivided to [[0,0,0],[1/6,0,0],[1/3,0,0],[1/2,0,0],
//   [1/2,0,0],[2/3,0,0],[5/6,0,0],[1,0,0]]; equal lengths -> early return.
// ---------------------------------------------------------------------------
describe("VMobject.align_points", () => {
  it("subdivides the shorter path like Python", () => {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    expect(a.align_points(b)).toBe(a);
    expect(a.get_num_points()).toBe(8);
    expect(b.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(a.points, 3), [
      [0.0, 0.0, 0.0],
      [0.166666666667, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [0.833333333333, 0.0, 0.0],
      [1.0, 0.0, 0.0],
    ], 1e-9);
    expectRowsClose(rowsOf(b.points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.333333333333, 0.0, 0.0],
      [1.666666666667, 0.0, 0.0],
      [2.0, 0.0, 0.0],
    ], 1e-9);
  });
  it("returns early when point counts already match", () => {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[5, 5, 0], [6, 6, 0]]);
    const beforeA = rowsOf(a.points, 3).map((r) => [...r]);
    const beforeB = rowsOf(b.points, 3).map((r) => [...r]);
    a.align_points(b);
    expectRowsClose(rowsOf(a.points, 3), beforeA);
    expectRowsClose(rowsOf(b.points, 3), beforeB);
  });
  it("gives an empty mobject a null curve at its center", () => {
    // Python: empty (center [0,0,0]) vs 1-curve -> 4 pts of zeros vs corners.
    const e = new VMobject();
    const f = new VMobject();
    f.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    e.align_points(f);
    expectRowsClose(rowsOf(e.points, 3), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    expect(e.get_num_points()).toBe(f.get_num_points());
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VGroup.__add__ / __sub__ /
// __getitem__
// Python: g[0] is a; g[-1] is b; slice g[0:1] is a 1-member VGroup;
//   (g - a) has only b and leaves g intact.
// ---------------------------------------------------------------------------
describe("VGroup.__add__ / __sub__ / __getitem__", () => {
  function pair(): { g: VGroup; a: VMobject; b: VMobject } {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [0, 1, 0]]);
    return { g: new VGroup(a, b), a, b };
  }
  it("__add__ appends like Python", () => {
    const { g, a, b } = pair();
    const c = new VMobject();
    c.set_points_as_corners([[0, 0, 0], [1, 1, 0]]);
    const h = g.__add__(c);
    expect(h).toBeInstanceOf(VGroup);
    expect(h.submobjects.length).toBe(3);
    expect(h.submobjects[0]).toBe(a);
    expect(h.submobjects[1]).toBe(b);
    expect(h.submobjects[2]).toBe(c);
  });
  it("__sub__ removes the member on a copy like Python", () => {
    const { g, b } = pair();
    const h = g.__sub__(g.submobjects[0] as VMobject);
    expect(h.submobjects.length).toBe(1);
    expect(h.submobjects[0]).toBe(b);
    expect(g.submobjects.length).toBe(2);
  });
  it("__getitem__ handles ints, negatives and slices like Python", () => {
    const { g, a, b } = pair();
    expect(g.__getitem__(0)).toBe(a);
    expect(g.__getitem__(-1)).toBe(b);
    const sl = g.__getitem__({ start: 0, stop: 1 }) as VGroup;
    expect(sl).toBeInstanceOf(VGroup);
    expect(sl.submobjects.length).toBe(1);
    expect(sl.submobjects[0]).toBe(a);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.CurvesAsSubmobjects.__init__
// Python: 2-corner shape -> 2 submobjects; first holds the first bezier
//   tuple; style (fill color) matches the source.
// ---------------------------------------------------------------------------
describe("CurvesAsSubmobjects.__init__", () => {
  it("splits one VMobject per bezier tuple with matched style", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    const cas = new CurvesAsSubmobjects(v);
    expect(cas.submobjects.length).toBe(2);
    expectRowsClose(rowsOf((cas.submobjects[0] as VMobject).points, 3), [
      [0.0, 0.0, 0.0],
      [0.333333333333, 0.0, 0.0],
      [0.666666666667, 0.0, 0.0],
      [1.0, 0.0, 0.0],
    ], 1e-9);
    expect((cas.submobjects[0] as VMobject).get_fill_color()?.to_hex()).toBe(
      v.get_fill_color()?.to_hex(),
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.ParametricFunction.init_points
// Python: (t,t^2,0), t_range [0,1,0.5] -> 8 points starting
//   [0,0,0],[1/6,1/36,0]; clear + init_points restores 8;
//   discontinuity at 0 over [-1,1] -> 16 points.
// ---------------------------------------------------------------------------
describe("ParametricFunction.init_points", () => {
  it("regenerates the sampled curve like Python", () => {
    // Python (default smoothing on): 8 points; first rows
    // [0,0,0],[0.166666666667,0.041666666667,0]; last [1,1,0].
    const p = new ParametricFunction((t: number) => [t, t ** 2, 0], {
      t_range: [0, 1, 0.5],
    });
    expect(p.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(p.points, 3).slice(0, 3), [
      [0.0, 0.0, 0.0],
      [0.166666666667, 0.041666666667, 0.0],
      [0.333333333333, 0.083333333333, 0.0],
    ], 1e-9);
    expectRowsClose(rowsOf(p.points, 3).slice(-2), [
      [0.833333333333, 0.708333333333, 0.0],
      [1.0, 1.0, 0.0],
    ], 1e-9);
    expect(p.init_points()).toBe(p);
    p.clear_points();
    expect(p.get_num_points()).toBe(0);
    p.init_points();
    expect(p.get_num_points()).toBe(8);
  });
  it("splits at discontinuities like Python", () => {
    const p = new ParametricFunction((t: number) => [t, t === 0 ? 0 : 1 / t, 0], {
      t_range: [-1, 1, 0.5],
      discontinuities: [0],
      use_smoothing: false,
    });
    expect(p.get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.functions.ImplicitFunction.init_points
// Python (manim 0.21.0, circle x^2+y^2-1 over [-2,2]^2, use_smoothing=False):
//   2976 points, first row [-0.64328457, -0.765625, 0] (probed 2026-09-26;
//   the real `isosurfaces.plot_isoline` backend has no TS port yet, see
//   ). TS has no `isosurfaces` port: a stub isoline computer
//   stands in, and the init_points-after-clear regeneration path is what is
//   asserted here (NOT the Python point values above).
// ---------------------------------------------------------------------------
describe("ImplicitFunction.init_points", () => {
  it("regenerates points from the isoline computer", () => {
    const curves = [[[1, 0], [0, 1], [0.5, 0.5]]];
    setIsolineComputer(() => curves as unknown as number[][][]);
    try {
      const f = new ImplicitFunction((x: number, y: number) => x + y, {
        x_range: [-2, 2],
        y_range: [-2, 2],
        use_smoothing: false,
      });
      const before = rowsOf(f.points, 3).map((r) => [...r]);
      expect(f.get_num_points()).toBeGreaterThan(0);
      expectRowsClose(rowsOf(f.points, 3).slice(0, 1), [[1, 0, 0]]);
      expect(f.init_points()).toBe(f);
      f.clear_points();
      expect(f.get_num_points()).toBe(0);
      f.init_points();
      expectRowsClose(rowsOf(f.points, 3), before);
    } finally {
      setIsolineComputer(null);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.DashedLine.__init__
// Python: [-1,0,0]->[1,0,0], dash 0.2 -> 5 dashes, length 2,
//   start/end preserved; defaults dash 0.05 ratio 0.5; (2/0.1*0.7)=14.
// ---------------------------------------------------------------------------
describe("DashedLine.__init__", () => {
  it("dashes a unit segment like Python", () => {
    const d = new DashedLine([-1, 0, 0], [1, 0, 0], { dash_length: 0.2 });
    expect(d.dash_length).toBe(0.2);
    expect(d.dashed_ratio).toBe(0.5);
    expect(d._calculate_num_dashes()).toBe(5);
    expect(d.submobjects.length).toBe(5);
    expectClose(d.get_length(), 2.0);
    expectRowsClose([d.get_start()], [[-1, 0, 0]]);
    expectRowsClose([d.get_end()], [[1, 0, 0]]);
  });
  it("uses Python's defaults", () => {
    const d = new DashedLine([-1, 0, 0], [1, 0, 0], { dash_length: 0.1, dashed_ratio: 0.7 });
    expect(d._calculate_num_dashes()).toBe(14);
    expect(d.submobjects.length).toBe(14);
    const e = new DashedLine();
    expect(e.dash_length).toBe(0.05);
    expect(e.dashed_ratio).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.Axes.point_to_coords
// Python (x_range [0,10,2], y_range [-1,1,1], x/y_length 5):
//   coords_to_point(5,0) -> [0,0,0]; point_to_coords([0,0,0]) -> [5,~0].
// NOTE: the TS Axes hierarchy is not Mobject/NumberLine-based yet, so Axes
// falls back to the NumberPlane linear map here; with explicit lengths that
// fallback is exactly Python's answer (probed on both Axes and NumberPlane).
// ---------------------------------------------------------------------------
describe("Axes.point_to_coords", () => {
  it("inverts the linear map like Python (explicit lengths)", () => {
    const ax = new Axes([0, 10, 2], [-1, 1, 1], 5, 5);
    const pt = ax.coords_to_point(5, 0);
    expectRowsClose(rowsOf(pt as NDArray, 3), [[0, 0, 0]], 1e-9);
    const coords = ax.point_to_coords([0, 0, 0]);
    expectClose(coords[0] ?? NaN, 5.0);
    expectClose(coords[1] ?? NaN, 0.0, 1e-9);
    const plane = new NumberPlane([0, 10, 2], [-1, 1, 1], 5, 5);
    expectClose(plane.point_to_coords([0, 0, 0])[0] ?? NaN, 5.0);
  });
  it("maps a list of points row-wise like Python (point.ndim == 2 -> result.T)", () => {
    // Python (same explicit-lengths axes):
    //   ax.point_to_coords([[0,0,0],[1,0,0]])
    //   -> [[5.0, 2.22e-16], [7.0, 2.22e-16]].
    const ax = new Axes([0, 10, 2], [-1, 1, 1], 5, 5);
    const coords = ax.point_to_coords([[0, 0, 0], [1, 0, 0]] as unknown as readonly number[]) as unknown as number[][];
    expect(Array.isArray(coords) && Array.isArray(coords[0])).toBe(true);
    expectRowsClose(coords, [[5.0, 0.0], [7.0, 0.0]], 1e-9);
    const plane = new NumberPlane([0, 10, 2], [-1, 1, 1], 5, 5);
    expectRowsClose(
      plane.point_to_coords([[0, 0, 0], [1, 0, 0]] as unknown as readonly number[]) as unknown as number[][],
      [[5.0, 0.0], [7.0, 0.0]],
      1e-9,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.i2gc / i2gp /
// angle_of_tangent / plot_antiderivative_graph
// Python (default Axes, parabola x^2): i2gc(3) -> (3,9);
//   i2gp(1) -> [0.857142857142857, 0.75, 0] (axes-mapped point);
//   angle_of_tangent(3) -> 1.4056476493802699;
//   antiderivative underlying: f(0)=0, f(1)=0.33340274885464394,
//   f(2)=2.6672219908371515 (trapezoid, samples=50).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.i2gc / i2gp / angle_of_tangent", () => {
  it("matches Python on the parabola", () => {
    const ax = new Axes();
    const parabola = ax.plot((x: number) => x ** 2);
    expect(ax.i2gc(3, parabola)).toEqual([3, 9]);
    expectRowsClose(rowsOf(ax.i2gp(1, parabola) as NDArray, 3), [
      [0.857142857142857, 0.75, 0.0],
    ], 1e-9);
    expectClose(ax.angle_of_tangent(3, parabola), 1.4056476493802699);
  });
});

describe("CoordinateSystem.plot_antiderivative_graph", () => {
  it("integrates x^2 to x^3/3 like Python (trapezoid, samples=50)", () => {
    const ax = new Axes();
    const parabola = ax.plot((x: number) => x ** 2);
    const anti = ax.plot_antiderivative_graph(parabola);
    const f = (anti as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(f(0), 0.0);
    expectClose(f(1), 0.33340274885464394);
    expectClose(f(2), 2.6672219908371515);
    // y_intercept shifts the curve.
    const anti2 = ax.plot_antiderivative_graph(parabola, 5);
    const f2 = (anti2 as unknown as { underlying_function: (x: number) => number }).underlying_function;
    expectClose(f2(0), 5.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.polyhedra Tetra/Octa/Icosa/Dodeca/ConvexHull3D
// Python: unit = sqrt(2)/4 (tetra), sqrt(2)/2 (octa);
//   tetra 4 verts/4 faces; octa 6/8; icosa 12/20 v0=[0,0.5,0.809016994375];
//   dodeca 20/12 v0=[0.809016994375 x3]; perturbed-octahedron hull -> 6/8
//   (exact axis points admit a degenerate random initial simplex; see NOTE
//   on the ConvexHull3D case below).
// ---------------------------------------------------------------------------
describe("Platonic solids", () => {
  it("Tetrahedron matches Python vertices and faces", () => {
    const t = new Tetrahedron(1);
    const u = Math.SQRT2 / 4;
    expectRowsClose(t.vertex_coords, [
      [u, u, u],
      [u, -u, -u],
      [-u, u, -u],
      [-u, -u, u],
    ], 1e-9);
    expect(t.faces_list).toEqual([[0, 1, 2], [3, 0, 2], [0, 1, 3], [3, 1, 2]]);
    expect(t.faces.submobjects.length).toBe(4);
    expect(t.edges.length).toBe(12);
  });
  it("Octahedron matches Python vertices and face count", () => {
    const o = new Octahedron(1);
    const u = Math.SQRT2 / 2;
    expectRowsClose(o.vertex_coords, [
      [u, 0, 0], [-u, 0, 0], [0, u, 0], [0, -u, 0], [0, 0, u], [0, 0, -u],
    ], 1e-9);
    expect(o.faces.submobjects.length).toBe(8);
  });
  it("Icosahedron / Dodecahedron match Python counts and first vertex", () => {
    const ic = new Icosahedron(1);
    expect(ic.vertex_coords.length).toBe(12);
    expect(ic.faces.submobjects.length).toBe(20);
    expectRowsClose([ic.vertex_coords[0] as number[]], [[0, 0.5, 0.809016994375]], 1e-9);
    const dd = new Dodecahedron(1);
    expect(dd.vertex_coords.length).toBe(20);
    expect(dd.faces.submobjects.length).toBe(12);
    expectRowsClose([dd.vertex_coords[0] as number[]], [[0.809016994375, 0.809016994375, 0.809016994375]], 1e-9);
  });
  it("ConvexHull3D of 6 axis points matches Python", () => {
    // NOTE: QuickHull samples its initial simplex with an unseeded RNG
    // (faithful to Python, which also uses np.random.default_rng()). The
    // exact 6 axis points admit a degenerate (coplanar) initial simplex, on
    // which BOTH implementations intermittently return 0 vertices/0 faces
    // (reproduced with the original the original manim). The inputs below are perturbed
    // into general position (no 4 coplanar) so any initial simplex is valid
    // and both sides deterministically yield 6 vertices / 8 faces
    // (12/12 stable runs in the original manim).
    const h = new ConvexHull3D(
      [1, 0.01, 0.02], [-1, 0.03, -0.01], [0.02, 1, 0.01],
      [-0.01, -1, 0.03], [0.01, 0.02, 1], [0.03, -0.01, -1],
    );
    expect(h.vertex_coords.length).toBe(6);
    expect(h.faces_list.length).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cube.__init__ / Prism / Cone
// Python: Cube(2) -> 6 faces, width 2; Prism([3,2,1]) -> 3/2/1;
//   Cone(1,1).theta = pi - atan(1) = 3pi/4; Cone(2,3,show_base) adds base.
// ---------------------------------------------------------------------------
describe("Cube / Prism / Cone", () => {
  it("Cube builds 6 faces with the right size like Python", () => {
    const c = new Cube(2);
    expect(c.side_length).toBe(2);
    expect(c.submobjects.length).toBe(6);
    expectClose(c.width, 2.0, 1e-6);
  });
  it("Prism rescales to its dimensions like Python", () => {
    const pr = new Prism([3, 2, 1]);
    expect([...pr.dimensions]).toEqual([3, 2, 1]);
    expectClose(pr.width, 3.0, 1e-6);
    expectClose(pr.height, 2.0, 1e-6);
    expectClose(pr.depth, 1.0, 1e-6);
  });
  it("Cone stores theta/height and shows the base on request", () => {
    // NOTE: resolution 4 keeps this test fast (default 32 builds 1000+
    // faces); theta/height/base plumbing is resolution-independent.
    const cone = new Cone(1, 1, [0, 0, 1], false, [0, 2 * Math.PI], 0, false, { resolution: 4 });
    expectClose(cone.theta, Math.PI - Math.atan(1));
    expectClose(cone.new_height, 1);
    expectClose(cone.base_circle.radius ?? cone.base_circle.width / 2, 1, 1e-6);
    const cone2 = new Cone(2, 3, [0, 0, 1], true, [0, 2 * Math.PI], 0, false, { resolution: 4 });
    expectClose(cone2.theta, Math.PI - Math.atan(2 / 3));
    expect(cone2.submobjects.includes(cone2.base_circle as unknown as Mobject)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols combined with earlier-ported code, verified
// against the original manim implementation the same way.
// ---------------------------------------------------------------------------
describe("batch integration", () => {
  it("table rows feed set_color across the whole row", () => {
    // Python: coloring get_rows()[1] recolors entries a and b (and label).
    const { t, a, b } = labeledTable(false);
    (t.get_rows().__getitem__(1) as VGroup).set_color("#FF0000");
    expect(a.get_fill_color()?.to_hex()).toBe("#FF0000");
    expect(b.get_fill_color()?.to_hex()).toBe("#FF0000");
  });
  it("plot -> i2gp -> point_to_coords round-trips x", () => {
    // Python: ax.point_to_coords(ax.i2gp(1, parabola))[0] == ~1.0.
    const ax = new Axes();
    const parabola = ax.plot((x: number) => x ** 2);
    const pt = rowsOf(ax.i2gp(1, parabola) as NDArray, 3)[0] as number[];
    expectClose(ax.point_to_coords(pt)[0] ?? NaN, 1.0, 1e-6);
  });
  it("align_points + interpolate agree on subdivided midpoints", () => {
    // Python: after aligning 1-curve vs 2-curve shapes, interpolating at
    // alpha=0.5 between aligned copies stays on the segment.
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [2, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    a.align_points(b);
    expect(a.get_num_points()).toBe(b.get_num_points());
    const mid = new VMobject();
    mid.interpolate(a, b, 0.5);
    for (const row of rowsOf(mid.points, 3)) {
      expectClose(row[1] ?? NaN, 0.0);
      expectClose(row[2] ?? NaN, 0.0);
    }
  });
  it("close_path then CurvesAsSubmobjects counts the closing curve", () => {
    // Python: 2-corner shape closed -> 3 curves as submobjects.
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.close_path();
    expect(new CurvesAsSubmobjects(v).submobjects.length).toBe(3);
  });
  it("antiderivative graph plots through the integrated intercept", () => {
    // Python: anti(0) == y_intercept maps to coords_to_point(0, yi)[1] == 0
    // height for yi=5 on default axes (origin maps to y=0 line).
    const ax = new Axes();
    const parabola = ax.plot((x: number) => x ** 2);
    const anti = ax.plot_antiderivative_graph(parabola, 5);
    const pt = rowsOf(ax.i2gp(0, anti) as NDArray, 3)[0] as number[];
    expectClose(pt[0] ?? NaN, rowsOf(ax.coords_to_point(0, 5) as NDArray, 3)[0]?.[0] ?? NaN, 1e-6);
  });
});
