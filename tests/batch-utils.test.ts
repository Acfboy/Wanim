import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import { binary_search } from "../src/utils/simple_functions.js";
import { cross2d } from "../src/utils/space_ops.js";
import { adjacent_n_tuples, resize_with_interpolation } from "../src/utils/iterables.js";
import { _Uniforms } from "../src/utils/config_ops.js";
import { seek_full_path_from_defaults } from "../src/utils/file_ops.js";
import { change_to_rgba_array } from "../src/utils/images.js";
import { QuickHull } from "../src/utils/qhull.js";

function nums(value: unknown): number[] {
  const raw = (value as { toArray: () => unknown }).toArray();
  const flat = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)]);
  return flat(raw);
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = nums(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]) * relTol);
    expect(Math.abs(values[i] - expected[i])).toBeLessThanOrEqual(tol);
  }
}

describe("binary_search", () => {
  // Python: binary_search(x^2+3x+1, 11, 0, 5) == 2.0000457763671875;
  //         binary_search(..., 71, 0, 5) is None.
  it("finds the root within tolerance", () => {
    const f = (x: number) => x ** 2 + 3 * x + 1;
    const solution = binary_search(f, 11, 0, 5);
    expect(solution).not.toBeNull();
    expect(Math.abs((solution as number) - 2.0000457763671875)).toBeLessThanOrEqual(1e-12);
    expect(Math.abs((solution as number) - 2)).toBeLessThan(1e-4);
  });
  it("returns null when the target is out of range", () => {
    expect(binary_search((x: number) => x ** 2 + 3 * x + 1, 71, 0, 5)).toBeNull();
  });
  it("handles decreasing functions (bound swap)", () => {
    expect(binary_search((x: number) => -x, 0, -5, 5)).toBe(0.0);
  });
});

describe("cross2d", () => {
  // Python: cross2d([1,2],[3,4]) == -2;
  //   cross2d([[1,2,0],[1,0,0]], [[3,4,0],[0,1,0]]) == [-2, 1].
  it("computes the scalar determinant", () => {
    expect(cross2d(array([1, 2]), array([3, 4]))).toBe(-2);
  });
  it("computes batched determinants", () => {
    expect(
      cross2d(
        array([
          [1, 2, 0],
          [1, 0, 0],
        ]),
        array([
          [3, 4, 0],
          [0, 1, 0],
        ]),
      ),
    ).toEqual([-2, 1]);
  });
});

describe("adjacent_n_tuples", () => {
  // Python: adjacent_n_tuples([1,2,3,4],2) == [(1,2),(2,3),(3,4),(4,1)], etc.
  it("wraps cyclically for n=2", () => {
    expect(adjacent_n_tuples([1, 2, 3, 4], 2)).toEqual([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 1],
    ]);
  });
  it("wraps cyclically for n=3", () => {
    expect(adjacent_n_tuples([1, 2, 3, 4], 3)).toEqual([
      [1, 2, 3],
      [2, 3, 4],
      [3, 4, 1],
      [4, 1, 2],
    ]);
  });
  it("returns [] for n=0", () => {
    expect(adjacent_n_tuples([1, 2, 3], 0)).toEqual([]);
  });
});

describe("resize_with_interpolation", () => {
  // Python:
  //   resize([[1,2],[3,4]], 2) -> same values
  //   resize([[1,2],[3,4]], 4) -> [[1,2],[1.667,2.667],[2.333,3.333],[3,4]]
  //   resize([[1,2],[3,4]], 1) -> [[1,2]]
  //   resize([1,2,3], 5) -> [1,1.5,2,2.5,3]
  it("keeps values for equal length", () => {
    expectClose(resize_with_interpolation(array([[1, 2], [3, 4]]), 2), [1, 2, 3, 4]);
  });
  it("interpolates when extending", () => {
    expectClose(resize_with_interpolation(array([[1, 2], [3, 4]]), 4), [
      1.0, 2.0, 1.6666666666666667, 2.666666666666667, 2.3333333333333335, 3.333333333333333, 3.0, 4.0,
    ]);
  });
  it("truncates to a single row", () => {
    expectClose(resize_with_interpolation(array([[1, 2], [3, 4]]), 1), [1.0, 2.0]);
  });
  it("interpolates 1-D arrays", () => {
    expectClose(resize_with_interpolation(array([1, 2, 3]), 5), [1.0, 1.5, 2.0, 2.5, 3.0]);
  });
});

describe("_Uniforms", () => {
  it("reads and writes obj.uniforms[name] after __set_name__", () => {
    const u = new _Uniforms<number>();
    u.__set_name__(null, "myattr");
    const obj = { uniforms: { myattr: 3.5 } };
    expect(u.__get__(obj)).toBe(3.5);
    u.__set__(obj, 7.0);
    expect(obj.uniforms["myattr"]).toBe(7.0);
  });
});

describe("seek_full_path_from_defaults", () => {
  it("finds files by absolute path and via default_dir + extension", () => {
    const dir = mkdtempSync(join(tmpdir(), "wanim-seek-"));
    writeFileSync(join(dir, "foo.txt"), "hi");
    expect(seek_full_path_from_defaults(join(dir, "foo.txt"), dir, [".txt"])).toBe(join(dir, "foo.txt"));
    expect(seek_full_path_from_defaults("foo", dir, [".txt"])).toBe(join(dir, "foo.txt"));
  });
  it("raises OSError naming the file when nothing matches", () => {
    const dir = mkdtempSync(join(tmpdir(), "wanim-seek-"));
    let thrown: unknown;
    try {
      seek_full_path_from_defaults("nonexistent_xyz", dir, [".txt"]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("OSError");
    expect((thrown as Error).message).toContain("nonexistent_xyz");
  });
});

describe("change_to_rgba_array", () => {
  // Python: grayscale (2,2) uint8 [[0,128],[255,64]] -> (2,2,4) with rgb
  // tripled and alpha 255; (2,2,3) zeros -> alpha 255 everywhere.
  it("expands grayscale to RGBA with full alpha", () => {
    const out = change_to_rgba_array(
      array([
        [0, 128],
        [255, 64],
      ]),
    );
    expect([...out.shape]).toEqual([2, 2, 4]);
    expect(nums(out)).toEqual([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 64, 64, 64, 255]);
  });
  it("appends full alpha to RGB", () => {
    const out = change_to_rgba_array(
      array([
        [
          [0, 0, 0],
          [1, 2, 3],
        ],
        [
          [4, 5, 6],
          [7, 8, 9],
        ],
      ]),
    );
    expect([...out.shape]).toEqual([2, 2, 4]);
    expect(nums(out)).toEqual([0, 0, 0, 255, 1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255]);
  });
});

describe("QuickHull", () => {
  it("stores the tolerance (default 1e-5)", () => {
    expect(new QuickHull().tolerance).toBe(1e-5);
    expect(new QuickHull(0.1).tolerance).toBe(0.1);
  });
  it("raises ValueError when too few points are supplied", () => {
    let thrown: unknown;
    try {
      new QuickHull().build(array([[0, 0]]));
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe("Not enough points supplied to build Convex Hull!");
  });
  it("raises ValueError for 1-D data", () => {
    let thrown: unknown;
    try {
      new QuickHull().build(array([[0], [1], [2]]));
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe("The Convex Hull of 1D data is its min-max!");
  });
  it("builds a hull whose live facets cover the square corners", () => {
    // NOTE: both implementations pick the initial simplex from an unseeded
    // RNG, so exact facet lists are nondeterministic; assert RNG-independent
    // structural invariants instead (corners must end up as hull vertices,
    // all vertices must be input points).
    const inputs = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0.5, 0.5],
    ];
    const q = new QuickHull();
    q.build(array(inputs));
    expect(q.internal).not.toBeNull();
    expect(q.facets.length).toBeGreaterThan(0);
    const live = q.facets.filter((f) => !q.removed.has(f.key()));
    expect(live.length).toBeGreaterThan(0);
    const verts = new Set(live.flatMap((f) => f.coordinates.map((c) => JSON.stringify(c))));
    for (const corner of inputs.slice(0, 4)) {
      expect(verts.has(JSON.stringify(corner))).toBe(true);
    }
    for (const key of verts) {
      expect(inputs.map((p) => JSON.stringify(p))).toContain(key);
    }
  });
});
