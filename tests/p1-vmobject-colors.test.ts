// P1-B / P1-C: live color properties, gradient color arrays, and the
// Python positional set_stroke signature.
//
// Ground truth captured from manim 0.21.0 (the original manim):
//   v.fill_color = RED            -> get_fill_color #FC6255
//   v.color = BLUE                -> fill #58C4DD, stroke #58C4DD
//   v.set_fill([RED, BLUE])       -> two rows [[0.988…, 0.384…, 0.333…, 0],
//                                            [0.345…, 0.768…, 0.866…, 0]]
//   v.set_stroke(GREEN, 9, 0.8, True) -> background #83C167, width 9, op 0.8
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { pointRows } from "../src/utils/arrays.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

const RED = "#FC6255";
const BLUE = "#58C4DD";
const GREEN = "#83C167";

function styled(): VMobject {
  const v = new VMobject();
  v.set_points([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
  ] as unknown as NDArray);
  return v;
}

function rowColors(array: NDArray): string[] {
  return pointRows(array, 4).map((row) => {
    const [r = 0, g = 0, b = 0] = row;
    return `#${[r, g, b].map((c) => Math.round(c * 255).toString(16).padStart(2, "0")).join("").toUpperCase()}`;
  });
}

describe("VMobject live color properties", () => {
  it("fill_color assignment rewrites the rgba array", () => {
    const v = styled();
    v.fill_color = RED as never;
    expect(String(v.get_fill_color())).toBe(RED);
    for (const color of rowColors(v.fill_rgbas)) expect(color).toBe(RED);
  });

  it("color assignment sets both fill and stroke", () => {
    const v = styled();
    v.color = BLUE as never;
    expect(String(v.get_fill_color())).toBe(BLUE);
    expect(String(v.get_stroke_color())).toBe(BLUE);
  });

  it("stroke_color assignment rewrites the stroke rgba array", () => {
    const v = styled();
    v.stroke_color = GREEN as never;
    expect(String(v.get_stroke_color())).toBe(GREEN);
    for (const color of rowColors(v.stroke_rgbas)) expect(color).toBe(GREEN);
  });
});

describe("VMobject gradient color arrays", () => {
  it("set_fill([RED, BLUE]) builds a two-stop gradient (one row per stop)", () => {
    const v = styled();
    v.set_fill([RED, BLUE] as never);
    const colors = rowColors(v.fill_rgbas);
    // Python ground truth: two rows, one per color stop (broadcast over points
    // at render time), not one row per point.
    expect(colors.length).toBe(2);
    expect(colors[0]).toBe(RED);
    expect(colors[colors.length - 1]).toBe(BLUE);
  });

  it("set_stroke accepts a color array", () => {
    const v = styled();
    v.set_stroke([RED, BLUE] as never, 4, 1);
    const colors = rowColors(v.stroke_rgbas);
    expect(colors[0]).toBe(RED);
    expect(colors[colors.length - 1]).toBe(BLUE);
  });

  it("set_style preserves a fill gradient instead of collapsing it", () => {
    const v = styled();
    v.set_style({ fill_color: [RED, BLUE] as never });
    const colors = rowColors(v.fill_rgbas);
    expect(new Set(colors)).toEqual(new Set([RED, BLUE]));
  });
});

describe("set_stroke Python positional signature", () => {
  it("4th positional arg is background, not family", () => {
    const v = styled();
    v.set_stroke(GREEN, 9, 0.8, true);
    expect(String(v.get_stroke_color(true))).toBe(GREEN);
    expect(v.get_stroke_width(true)).toBe(9);
    expect(v.get_stroke_opacity(true)).toBeCloseTo(0.8, 12);
    // Foreground stroke untouched.
    expect(String(v.get_stroke_color())).toBe("#FFFFFF");
  });

  it("family can be controlled through the options object", () => {
    const parent = styled();
    const child = styled();
    parent.add(child);
    parent.set_stroke(GREEN, 9, 0.8, { family: false });
    expect(String(parent.get_stroke_color())).toBe(GREEN);
    expect(String(child.get_stroke_color())).toBe("#FFFFFF");
    parent.set_stroke(BLUE, 5, undefined, { family: true });
    expect(String(child.get_stroke_color())).toBe(BLUE);
  });
});
