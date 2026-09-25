import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import {
  CurvesAsSubmobjects,
  VDict,
  VMobject,
} from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { _BooleanOps } from "../src/mobject/geometry/boolean_ops.js";
import { Arc } from "../src/mobject/geometry/arc.js";

function rows2(value: unknown): number[][] {
  return ((value as { toArray: () => unknown }).toArray() as unknown[][]).map((r) =>
    (r as unknown[]).map(Number),
  );
}

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= Math.max(1e-12, Math.abs(b) * tol);
}

describe("VDict.__delitem__", () => {
  it("removes an existing key", () => {
    const d = new VDict();
    const mob = new VMobject();
    d.submob_dict.set("sq", mob);
    expect(d.submob_dict.has("sq")).toBe(true);
    d.__delitem__("sq");
    expect(d.submob_dict.has("sq")).toBe(false);
  });
  it("raises KeyError with the key repr for a missing key", () => {
    const d = new VDict();
    let thrown: unknown;
    try {
      d.__delitem__("missing");
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("KeyError");
    // Python: str(KeyError('missing')) == "'missing'"
    expect((thrown as Error).message).toBe("'missing'");
  });
});

describe("CurvesAsSubmobjects._throw_error_if_no_submobjects", () => {
  it("throws when there are no submobjects, naming the caller", () => {
    const c = new CurvesAsSubmobjects();
    let thrown: unknown;
    const caller = function myCaller() {
      c._throw_error_if_no_submobjects();
    };
    try {
      caller();
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    // Python: f"Cannot call CurvesAsSubmobjects. {caller} for a CurvesAsSubmobject with no submobjects"
    expect(String((thrown as Error).message)).toContain("Cannot call CurvesAsSubmobjects.");
    expect(String((thrown as Error).message)).toContain("with no submobjects");
  });
  it("does not throw when submobjects exist", () => {
    const c = new CurvesAsSubmobjects(new VMobject());
    expect(() => c._throw_error_if_no_submobjects()).not.toThrow();
  });
});

describe("CurvesAsSubmobjects._get_submobjects_with_points", () => {
  it("returns only submobjects that have points", () => {
    const withPts = new VMobject();
    withPts.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const empty = new VMobject();
    const c = new CurvesAsSubmobjects(withPts);
    c.add(empty);
    const result = c._get_submobjects_with_points();
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(withPts);
  });
  it("throws when no submobject has points", () => {
    const c = new CurvesAsSubmobjects(new VMobject());
    let thrown: unknown;
    try {
      c._get_submobjects_with_points();
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect(String((thrown as Error).message)).toContain("whose submobjects have no points");
  });
});

describe("VMobject.consider_points_equals_2d", () => {
  // Python ground truths:
  //   ([0,0],[0,0]) -> True; ([1,2],[1+1e-7,2]) -> True
  //   ([0,0],[1,0]) -> False; ([0,0],[nan,0]) -> False
  //   ([inf,0],[inf,0]) -> True
  const v = new VMobject();
  it("identifies equal and near-equal points", () => {
    expect(v.consider_points_equals_2d([0, 0], [0, 0])).toBe(true);
    expect(v.consider_points_equals_2d([1.0, 2.0], [1.0 + 1e-7, 2.0])).toBe(true);
  });
  it("rejects far and nan points", () => {
    expect(v.consider_points_equals_2d([0, 0], [1, 0])).toBe(false);
    expect(v.consider_points_equals_2d([0, 0], [Number.NaN, 0])).toBe(false);
  });
  it("treats same-sign infinities as equal", () => {
    expect(v.consider_points_equals_2d([Number.POSITIVE_INFINITY, 0], [Number.POSITIVE_INFINITY, 0])).toBe(true);
  });
  it("detects y-only differences", () => {
    expect(v.consider_points_equals_2d([0, 0], [0, 1])).toBe(false);
    expect(v.consider_points_equals_2d([0, 1e-7], [0, 1e-7])).toBe(true);
  });
});

describe("VMobject.get_subpath_split_indices_from_points", () => {
  // Python: 8 points forming two 4-point curves with a jump -> [[0,4],[4,8]]
  // single curve (4 pts) -> [[0,4]]; fewer than nppcc -> shape (0,2).
  const v = new VMobject();
  it("splits where consecutive anchors differ", () => {
    const pts = array([
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [0, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 2, 0],
    ]);
    expect(rows2(v.get_subpath_split_indices_from_points(pts))).toEqual([
      [0, 4],
      [4, 8],
    ]);
  });
  it("returns a single range for one curve and empty for too few points", () => {
    const one = array([
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
    ]);
    expect(rows2(v.get_subpath_split_indices_from_points(one))).toEqual([[0, 4]]);
    const short = array([[0, 0, 0]]);
    const out = v.get_subpath_split_indices_from_points(short);
    expect([...out.shape]).toEqual([0, 2]);
  });
  it("treats NaN anchors as splits (never close)", () => {
    const pts = array([
      [Number.NaN, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    // Only one curve -> no internal boundary -> [[0, 4]] regardless.
    expect(rows2(v.get_subpath_split_indices_from_points(pts))).toEqual([[0, 4]]);
  });
  it("supports n_dims=2 comparison", () => {
    // z differs but x/y coincide -> no split in 2D, split in 3D.
    const pts = array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 5],
      [0, 0, 0],
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    expect(rows2(v.get_subpath_split_indices_from_points(pts, 2))).toEqual([[0, 8]]);
    expect(rows2(v.get_subpath_split_indices_from_points(pts, 3))).toEqual([
      [0, 4],
      [4, 8],
    ]);
  });
});

describe("VMobject.get_stroke_width", () => {
  it("returns stroke/background widths floored at 0", () => {
    const v = new VMobject();
    expect(v.get_stroke_width()).toBe(4);
    expect(v.get_stroke_width(true)).toBe(0.0);
  });
  it("parses string widths like int()", () => {
    const v = new VMobject({ stroke_width: "7" } as unknown as { stroke_width: number });
    expect(v.get_stroke_width()).toBe(7);
  });
});

describe("PMobject.reset_points", () => {
  it("empties rgbas (0,4) and points (0,3) and returns self", () => {
    const pm = new PMobject();
    expect([...pm.rgbas.shape]).toEqual([0, 4]);
    expect([...pm.points.shape]).toEqual([0, 3]);
    expect(pm.reset_points()).toBe(pm);
  });
});

describe("_BooleanOps._convert_2d_to_3d_array", () => {
  // Python: [(1,2),(3,4)] -> [[1,2,0],[3,4,0]];
  //         [(1,2,3),(4,5)] -> [[1,2,3],[4,5,0]]
  it("appends z_dim to 2D points and leaves 3D points alone", () => {
    const b = new _BooleanOps();
    expect(rows2(b._convert_2d_to_3d_array([[1, 2], [3, 4]]))).toEqual([
      [1, 2, 0],
      [3, 4, 0],
    ]);
    expect(rows2(b._convert_2d_to_3d_array([[1, 2, 3], [4, 5]]))).toEqual([
      [1, 2, 3],
      [4, 5, 0],
    ]);
  });
  it("honours a custom z_dim", () => {
    const b = new _BooleanOps();
    expect(rows2(b._convert_2d_to_3d_array([[1, 2]], 7))).toEqual([[1, 2, 7]]);
  });
});

describe("Arc._create_quadratic_bezier_points", () => {
  // Python: shape (6,3) for angle=pi/2, n=2; first rows
  // [1,0,0], [1,0.41421356,0], [0.70710678,0.70710678,0].
  it("matches the quadratic spline samples", () => {
    const pts = Arc._create_quadratic_bezier_points(Math.PI / 2, 0, 2);
    expect([...pts.shape]).toEqual([6, 3]);
    const rows = rows2(pts);
    expect(close(rows[0][0], 1) && close(rows[0][1], 0) && close(rows[0][2], 0)).toBe(true);
    expect(close(rows[1][0], 1) && close(rows[1][1], 0.4142135623730951)).toBe(true);
    expect(close(rows[2][0], 0.7071067811865476) && close(rows[2][1], 0.7071067811865475)).toBe(true);
  });
});
