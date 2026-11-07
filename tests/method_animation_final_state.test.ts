import { describe, expect, it } from "vitest";

import {
  DashedLine,
  MathTex,
  Square,
  Text,
  VGroup,
} from "../src/index.js";
import { pointRows } from "../src/utils/arrays.js";

function dashInfo(line: DashedLine): { count: number; minX: number; maxX: number } {
  const centers = line.submobjects.map((sub) => pointRows(sub.get_center(), 3)[0]?.[0] ?? 0);
  return {
    count: line.submobjects.length,
    minX: Math.min(...centers),
    maxX: Math.max(...centers),
  };
}

describe("method animation final state", () => {
  it("keeps every target dash after animate.become", () => {
    const line = new DashedLine([-5, 0, 0], [5, 0, 0]);
    const target = new DashedLine([0, 0, 0], [2, 1, 0]);
    const animation: any = line.animate.become(target).build();
    animation.begin();
    animation.interpolate(1);
    animation.finish();
    const final = dashInfo(line);
    const expected = dashInfo(target);
    expect(final.count).toBe(expected.count);
    expect(final.minX).toBeCloseTo(expected.minX, 6);
    expect(final.maxX).toBeCloseTo(expected.maxX, 6);
  });

  it("applies shift exactly once", () => {
    const square = new Square();
    const animation: any = square.animate.shift([2, 0, 0]).build();
    animation.begin();
    animation.interpolate(0.5);
    animation.finish();
    expect(square.get_center()[0]).toBeCloseTo(2, 9);
  });

  it("keeps a Text replacement's new glyphs", () => {
    const text = new Text("ab");
    const target = new Text("abcdef");
    const animation: any = text.animate.become(target).build();
    animation.begin();
    animation.interpolate(1);
    animation.finish();
    expect(text.width).toBeCloseTo(target.width, 6);
    expect(text.submobjects.length).toBe(target.submobjects.length);
  });

  it("keeps a VGroup replacement's new children", () => {
    const group = new VGroup(new Square(), new Square());
    const target = new VGroup(new Square(), new Square(), new Square(), new Square());
    const animation: any = group.animate.become(target).build();
    animation.begin();
    animation.interpolate(1);
    animation.finish();
    expect(group.submobjects.length).toBe(4);
    expect(group.width).toBeCloseTo(target.width, 9);
  });

  it("does not leave a MathTex target behind", () => {
    const tex = new MathTex("a");
    const target = new MathTex("a+b+c");
    const animation: any = tex.animate.become(target).build();
    animation.begin();
    animation.interpolate(1);
    animation.finish();
    expect(tex.width).toBeCloseTo(target.width, 6);
  });
});
