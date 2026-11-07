import { describe, expect, it } from "vitest";

import { InputDecimalMatrix, InputDecimalTable } from "../src/index.js";

function cell(matrix: InputDecimalMatrix, i: number, j: number): any {
  return matrix.get_entries().submobjects[i * 2 + j];
}

describe("InputDecimalMatrix options", () => {
  it("accepts num_decimal_places at the top level", () => {
    const matrix = new InputDecimalMatrix(
      [
        [1.234, 2],
        [3, 4],
      ],
      { num_decimal_places: 2 },
    );
    expect(cell(matrix, 0, 0).num_decimal_places).toBe(2);
    expect(cell(matrix, 0, 0)._get_num_string(1.234)).toBe("1.23");
  });

  it("accepts font_size at the top level", () => {
    const matrix = new InputDecimalMatrix([[1, 2]], { font_size: 100 } as any);
    expect(cell(matrix, 0, 0).font_size).toBeCloseTo(100, 6);
  });

  it("lets an explicit element config win over the shorthand", () => {
    const matrix = new InputDecimalMatrix([[1.234]], {
      num_decimal_places: 3,
      element_to_mobject_config: { num_decimal_places: 1 },
    });
    expect(cell(matrix, 0, 0).num_decimal_places).toBe(1);
  });

  it("exposes get_value(i, j)", () => {
    const matrix = new InputDecimalMatrix([
      [1.5, 2],
      [3, 4],
    ]);
    expect(matrix.get_value(0, 0)).toBeCloseTo(1.5, 9);
    expect(matrix.get_value(1, 1)).toBeCloseTo(4, 9);
    expect(() => matrix.get_value(9, 9)).toThrowError(/out of range/);
  });

  it("keeps a scaled matrix's height after editing and relayout", () => {
    const matrix = new InputDecimalMatrix([
      [1, 2],
      [3, 4],
    ]);
    matrix.scale(2);
    const height = matrix.height;
    const width = matrix.width;
    cell(matrix, 0, 0).set_value(233333);
    matrix.relayout();
    expect(matrix.height).toBeCloseTo(height, 6);
    expect(matrix.width).toBeGreaterThan(width);
    expect(matrix.get_center()[0]).toBeCloseTo(0, 6);
    expect(matrix.get_center()[1]).toBeCloseTo(0, 6);
  });
});

describe("InputDecimalTable options", () => {
  it("accepts num_decimal_places at the top level", () => {
    const table = new InputDecimalTable([[1.234, 2]], { num_decimal_places: 2 });
    const first = table.get_entries_without_labels([1, 1]) as any;
    expect(first.num_decimal_places).toBe(2);
  });

});
