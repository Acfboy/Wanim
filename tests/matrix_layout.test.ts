import { describe, expect, it } from "vitest";

import { DecimalMatrix, type Matrix } from "../src/mobject/matrix.js";
import { InputDecimalMatrix } from "../src/interactive/controls.js";
import type { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function cell(m: Matrix, i: number, j: number): VMobject {
  return m.get_mob_matrix()[i]![j]!;
}

function xInterval(c: VMobject): [number, number] {
  return [c.get_left()[0] ?? 0, c.get_right()[0] ?? 0];
}

function expectSameMatrix(a: Matrix, b: Matrix): void {
  expect(b.width).toBeCloseTo(a.width, 9);
  expect(b.height).toBeCloseTo(a.height, 9);
  expect(b.get_center()[0]).toBeCloseTo(a.get_center()[0]!, 9);
  expect(b.get_center()[1]).toBeCloseTo(a.get_center()[1]!, 9);
  for (let i = 0; i < a.get_mob_matrix().length; i += 1) {
    for (let j = 0; j < a.get_mob_matrix()[i]!.length; j += 1) {
      const ca = cell(a, i, j).get_center();
      const cb = cell(b, i, j).get_center();
      expect(cb[0]).toBeCloseTo(ca[0]!, 9);
      expect(cb[1]).toBeCloseTo(ca[1]!, 9);
    }
  }
}

describe("Matrix content-aware grid", () => {
  it("keeps the h_buff/v_buff pitch when the entries fit inside the buffs", () => {
    const m = new DecimalMatrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    // DR alignment keeps every cell's right/bottom edge on its anchor, so the
    // pitch is read off the edges (centers differ when glyph widths differ).
    for (let i = 0; i < 2; i += 1) {
      expect(cell(m, i, 1).get_right()[0]! - cell(m, i, 0).get_right()[0]!).toBeCloseTo(1.3, 9);
      expect(cell(m, i, 2).get_right()[0]! - cell(m, i, 1).get_right()[0]!).toBeCloseTo(1.3, 9);
    }
    for (let j = 0; j < 3; j += 1) {
      expect(cell(m, 0, j).get_bottom()[1]! - cell(m, 1, j).get_bottom()[1]!).toBeCloseTo(0.8, 9);
    }
  });

  it("expands only the wide column and keeps at least entry_gap between cells", () => {
    const m = new DecimalMatrix([
      [1.0, 2.0, 1.0],
      [2.0, 233333.0, 3.0],
      [3.0, 4.0, 2.0],
    ]);
    expect(cell(m, 1, 1).width).toBeGreaterThan(1.3);
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j + 1 < 3; j += 1) {
        const left = xInterval(cell(m, i, j));
        const right = xInterval(cell(m, i, j + 1));
        expect(right[0] - left[1]).toBeGreaterThanOrEqual(0.25 - 1e-9);
      }
    }
    // The narrow columns keep the fixed pitch relative to the wide one's
    // right edge (DR alignment keeps every cell's right edge on its anchor).
    expect(cell(m, 1, 1).get_right()[0]! - cell(m, 1, 0).get_right()[0]!).toBeCloseTo(
      Math.max(1.3, cell(m, 1, 1).width + 0.25),
      9,
    );
  });

  it("gives DecimalMatrix and InputDecimalMatrix the same geometry", () => {
    // `InputDecimalNumber` forces `group_with_commas=false`, so pin the same
    // formatting on both sides before comparing geometry.
    const config = { num_decimal_places: 1, group_with_commas: false };
    const data = [
      [1, 2, 1],
      [2, 233333, 3],
      [3, 4, 2],
    ];
    expectSameMatrix(
      new DecimalMatrix(data, { element_to_mobject_config: config }),
      new InputDecimalMatrix(data, { element_to_mobject_config: config }),
    );
    expectSameMatrix(
      new DecimalMatrix([
        [1, 2],
        [3, 4],
      ]),
      new InputDecimalMatrix([
        [1, 2],
        [3, 4],
      ]),
    );
  });

  it("keeps an edited InputDecimalMatrix centered and overlap-free after relayout", () => {
    const m = new InputDecimalMatrix([
      [1, 2],
      [3, 4],
    ]);
    const centerBefore = m.get_center();
    cell(m, 0, 0).set_value(233333);
    m.relayout();
    expect(m.get_center()[0]).toBeCloseTo(centerBefore[0]!, 9);
    expect(m.get_center()[1]).toBeCloseTo(centerBefore[1]!, 9);
    for (let i = 0; i < 2; i += 1) {
      const left = xInterval(cell(m, i, 0));
      const right = xInterval(cell(m, i, 1));
      expect(right[0] - left[1]).toBeGreaterThanOrEqual(0.25 - 1e-9);
    }
    // A second relayout with unchanged content is a no-op.
    const centers = m.get_mob_matrix().map((row) => row.map((c) => [...c.get_center()]));
    m.relayout();
    m.get_mob_matrix().forEach((row, i) => {
      row.forEach((c, j) => {
        expect(c.get_center()[0]).toBeCloseTo(centers[i]![j]![0]!, 9);
        expect(c.get_center()[1]).toBeCloseTo(centers[i]![j]![1]!, 9);
      });
    });
  });
});
