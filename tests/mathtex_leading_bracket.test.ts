import { describe, expect, it } from "vitest";

import { MathTex } from "../src/mobject/text/tex_mobject.js";

describe("MathTex leading bracket", () => {
  it("keeps a single argument that starts with '['", () => {
    const bracket = new MathTex("[A,B]=");
    const guarded = new MathTex("{}[A,B]=");
    expect(bracket.width).toBeGreaterThan(1);
    expect(bracket.width).toBeCloseTo(guarded.width, 9);
    expect(bracket.height).toBeCloseTo(guarded.height, 9);
  });

  it("keeps a bare bracketed list", () => {
    const bracket = new MathTex("[A,B]");
    expect(bracket.width).toBeGreaterThan(0.5);
    expect(bracket.height).toBeGreaterThan(0);
  });

  it("keeps a leading bracket after whitespace", () => {
    const bracket = new MathTex("  [0,1]");
    expect(bracket.width).toBeGreaterThan(0);
  });

  it("does not change a bracket in the middle of an expression", () => {
    const withPrefix = new MathTex("x[A,B]=");
    expect(withPrefix.width).toBeGreaterThan(new MathTex("x=").width);
  });
});
