import { describe, expect, it } from "vitest";

import { LEFT, RIGHT } from "../src/constants.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Mobject } from "../src/mobject/mobject.js";

describe("move_to keyword bridge", () => {
  it("pins the Python parameter names", () => {
    expect((Mobject.prototype.move_to as unknown as { __argnames__?: string[] }).__argnames__).toEqual([
      "pointOrMobject",
      "aligned_edge",
      "coor_mask",
    ]);
  });

  it("honours aligned_edge=LEFT through a kwargs bag", () => {
    const square = new Square();
    square.move_to([2, 1, 0], { aligned_edge: LEFT });
    expect(square.get_left()[0]).toBeCloseTo(2, 9);
    expect(square.get_center()[0]).toBeCloseTo(3, 9);
  });

  it("honours aligned_edge=RIGHT through a kwargs bag", () => {
    const square = new Square();
    square.move_to([2, 1, 0], { aligned_edge: RIGHT });
    expect(square.get_right()[0]).toBeCloseTo(2, 9);
    expect(square.get_center()[0]).toBeCloseTo(1, 9);
  });

  it("still treats a raw direction array as the alignment edge", () => {
    const square = new Square();
    square.move_to([2, 1, 0], LEFT);
    expect(square.get_left()[0]).toBeCloseTo(2, 9);
  });

  it("honours coor_mask through a kwargs bag", () => {
    const square = new Square();
    square.move_to([2, 1, 0], { coor_mask: [1, 0, 1] });
    expect(square.get_center()[0]).toBeCloseTo(2, 9);
    expect(square.get_center()[1]).toBeCloseTo(0, 9);
  });
});
