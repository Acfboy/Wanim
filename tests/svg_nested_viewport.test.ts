import { describe, expect, it } from "vitest";
import {
  svg_viewport_transform,
  parse_svg_viewbox,
} from "../src/mobject/svg/svg_mobject.js";
import { Matrix } from "../src/mobject/matrix.js";
import { MathTex } from "../src/mobject/text/tex_mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

// MathJax renders stretched delimiters (tall matrix brackets, big parens, ...)
// as top/bottom glyphs plus an extender placed inside a nested `<svg>` element
// carrying its own viewBox/x/y. Those nested viewports must be honoured, or the
// extender lands in the wrong place and the bracket renders with gaps.

describe("svg_viewport_transform", () => {
  it("parses a viewBox", () => {
    expect(parse_svg_viewbox("0 450.5 667 1802")).toEqual({
      x: 0,
      y: 450.5,
      width: 667,
      height: 1802,
    });
    expect(parse_svg_viewbox("bogus")).toBeNull();
    expect(parse_svg_viewbox(undefined)).toBeNull();
  });

  it("maps a MathJax extender nested <svg> into its viewport", () => {
    // Mirrors MathJax's `\left[ ... \right.` output: the extender <svg>
    // has viewBox "0 450.5 667 1802" and y="-651".
    const matrix = svg_viewport_transform({
      viewbox: "0 450.5 667 1802",
      width: "667",
      height: "1802",
      x: "0",
      y: "-651",
    });
    expect(matrix.a).toBeCloseTo(1, 10);
    expect(matrix.d).toBeCloseTo(1, 10);
    expect(matrix.b).toBeCloseTo(0, 10);
    expect(matrix.c).toBeCloseTo(0, 10);
    expect(matrix.e).toBeCloseTo(0, 10);
    expect(matrix.f).toBeCloseTo(-1101.5, 10);
  });

  it("centres content when the viewport aspect differs (xMidYMid meet)", () => {
    // viewBox 100x100 into a 200x100 viewport: scale 1, centred horizontally.
    const matrix = svg_viewport_transform({
      viewbox: "0 0 100 100",
      width: "200",
      height: "100",
    });
    expect(matrix.a).toBeCloseTo(1, 10);
    expect(matrix.d).toBeCloseTo(1, 10);
    expect(matrix.e).toBeCloseTo(50, 10);
    expect(matrix.f).toBeCloseTo(0, 10);
  });

  it("handles preserveAspectRatio=none and is identity without a viewBox", () => {
    const none = svg_viewport_transform({
      viewbox: "0 0 100 50",
      width: "200",
      height: "100",
      preserveaspectratio: "none",
    });
    expect(none.a).toBeCloseTo(2, 10);
    expect(none.d).toBeCloseTo(2, 10);
    expect(none.e).toBeCloseTo(0, 10);
    expect(none.f).toBeCloseTo(0, 10);

    const identity = svg_viewport_transform({ width: "10", height: "10" });
    expect(identity).toEqual({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  });
});

function leafVerticalIntervals(mob: VMobject): Array<[number, number]> {
  const intervals: Array<[number, number]> = [];
  const visit = (m: VMobject): void => {
    if (m.submobjects.length === 0) {
      if (m.points !== undefined && m.points.shape[0] > 0) {
        intervals.push([m.get_bottom()[1]!, m.get_top()[1]!]);
      }
      return;
    }
    for (const child of m.submobjects) visit(child as VMobject);
  };
  visit(mob);
  return intervals;
}

describe("Matrix bracket continuity", () => {
  it("stretched brackets form an unbroken vertical chain", () => {
    const m = new Matrix([
      ["0.0", "0.4", "0.6"],
      ["0.6", "0.1", "0.2"],
      ["0.4", "0.5", "0.2"],
    ]);
    const brackets = m.get_brackets().submobjects as MathTex[];
    expect(brackets.length).toBe(2);
    for (const bracket of brackets) {
      const intervals = leafVerticalIntervals(bracket).sort((a, b) => a[0] - b[0]);
      expect(intervals.length).toBeGreaterThanOrEqual(3);
      // Each piece must overlap (or at least touch) the previous one.
      for (let i = 1; i < intervals.length; i += 1) {
        expect(intervals[i]![0]).toBeLessThanOrEqual(intervals[i - 1]![1] + 1e-6);
      }
    }
  });
});
