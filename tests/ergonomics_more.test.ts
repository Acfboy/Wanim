import { describe, expect, it } from "vitest";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Triangle } from "../src/mobject/geometry/polygram.js";
import { Matrix } from "../src/mobject/matrix.js";
import { Text } from "../src/mobject/text/text_mobject.js";

describe("arrange_in_grid options bag", () => {
  it("matches the positional form", () => {
    const positional = new VGroup(new Square(), new Circle(), new Triangle());
    const bag = new VGroup(new Square(), new Circle(), new Triangle());
    positional.arrange_in_grid(1, 3, 0.3);
    bag.arrange_in_grid({ rows: 1, cols: 3, buff: 0.3 });
    const a = positional.submobjects.map((m) => m.get_center());
    const b = bag.submobjects.map((m) => m.get_center());
    for (let i = 0; i < a.length; i += 1) {
      for (let k = 0; k < 3; k += 1) expect(b[i][k]).toBeCloseTo(a[i][k], 9);
    }
  });
});

describe("Matrix element_to_mobject accepts a class", () => {
  it("builds entries with the Text class (numbers stringified)", () => {
    const matrix = new Matrix([[1, 2], [3, 4]], { element_to_mobject: Text });
    const entries = matrix.get_entries();
    expect(entries.submobjects.length).toBe(4);
  });

  it("still accepts a factory", () => {
    const matrix = new Matrix([[1, 2], [3, 4]], {
      element_to_mobject: (item) => new Text(String(item)),
    });
    expect(matrix.get_entries().submobjects.length).toBe(4);
  });
});
