import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Difference, Exclusion, Intersection, Union } from "../src/mobject/geometry/boolean_ops.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import type { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { computeSubpathRanges } from "../src/renderer/canvas.js";
import { pointRows } from "../src/utils/arrays.js";

/** Signed area summed over subpaths (dense shoelace; exact for polylines). */
function signedArea2D(vmobject: VMobject): number {
  const rows = pointRows(vmobject.points as NDArray, vmobject.dim).map(
    (row) => [Number(row[0] ?? 0), Number(row[1] ?? 0)],
  );
  const ranges = computeSubpathRanges(rows, vmobject.n_points_per_cubic_curve, 1e-6);
  let total = 0;
  for (const [start, end] of ranges) {
    const n = (end as number) - (start as number);
    if (n < 3) continue;
    for (let i = start as number; i < (end as number); i += 1) {
      const a = rows[i] ?? [0, 0];
      const b = rows[(i + 1 - (start as number)) % n + (start as number)] ?? [0, 0];
      total += (a[0] ?? 0) * (b[1] ?? 0) - (b[0] ?? 0) * (a[1] ?? 0);
    }
  }
  return total / 2;
}

const ringArea2D = (vmobject: VMobject): number => Math.abs(signedArea2D(vmobject));

describe("boolean ops", () => {
  it("intersects two overlapping squares with the lens area", () => {
    const a = new Square({ side_length: 2 });
    const b = new Square({ side_length: 2 });
    b.shift([1, 0, 0]);
    const inter = new Intersection(a, b);
    expect(ringArea2D(inter)).toBeCloseTo(2, 1);
  });

  it("unions two overlapping squares", () => {
    const a = new Square({ side_length: 2 });
    const b = new Square({ side_length: 2 });
    b.shift([1, 0, 0]);
    const union = new Union(a, b);
    expect(ringArea2D(union)).toBeCloseTo(6, 0);
  });

  it("differences two overlapping squares", () => {
    const a = new Square({ side_length: 2 });
    const b = new Square({ side_length: 2 });
    b.shift([1, 0, 0]);
    const diff = new Difference(a, b);
    expect(ringArea2D(diff)).toBeCloseTo(2, 1);
  });

  it("xors two overlapping squares", () => {
    const a = new Square({ side_length: 2 });
    const b = new Square({ side_length: 2 });
    b.shift([1, 0, 0]);
    const xor = new Exclusion(a, b);
    expect(ringArea2D(xor)).toBeCloseTo(4, 0);
  });

  it("handles disjoint and nested squares", () => {
    const a = new Square({ side_length: 2 });
    const far = new Square({ side_length: 2 });
    far.shift([10, 0, 0]);
    expect(ringArea2D(new Intersection(a, far))).toBeCloseTo(0, 6);
    expect(ringArea2D(new Union(a, far))).toBeCloseTo(8, 0);
    const big = new Square({ side_length: 4 });
    // nested difference keeps outer minus inner = 16 - 4; the signed area
    // also proves the hole winds opposite to the outer ring (nonzero fill).
    expect(ringArea2D(new Difference(big, a))).toBeCloseTo(12, 0);
    expect(signedArea2D(new Difference(big, a))).toBeCloseTo(12, 0);
  });

  it("collapses identical inputs and touches to sane results", () => {
    const a = new Square({ side_length: 2 });
    const same = new Square({ side_length: 2 });
    expect(ringArea2D(new Union(a, same))).toBeCloseTo(4, 0);
    expect(ringArea2D(new Intersection(a, same))).toBeCloseTo(4, 0);
    expect(new Difference(a, same).points.shape[0]).toBe(0);
    // Adjacent squares sharing an edge: union keeps both, minus nothing.
    const right = new Square({ side_length: 2 });
    right.shift([2, 0, 0]);
    expect(ringArea2D(new Union(a, right))).toBeCloseTo(8, 0);
    expect(ringArea2D(new Intersection(a, right))).toBeCloseTo(0, 6);
  });
  it("intersects and unions two overlapping circles", () => {
    const c1 = new Circle({ radius: 2 });
    const c2 = new Circle({ radius: 2 });
    c2.shift([2.4, 0, 0]);
    const inter = new Intersection(c1, c2);
    const interArea = ringArea2D(inter);
    // Analytic lens: 2 r^2 acos(d/2r) - d/2 sqrt(4r^2-d^2), d=2.4, r=2
    const d = 2.4;
    const r = 2;
    const expected = 2 * r * r * Math.acos(d / (2 * r)) - (d / 2) * Math.sqrt(4 * r * r - d * d);
    expect(interArea).toBeCloseTo(expected, 0);
    const union = new Union(c1, c2);
    expect(ringArea2D(union)).toBeCloseTo(2 * Math.PI * r * r - expected, 0);
  });
});
