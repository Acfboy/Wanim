import { describe, expect, it } from "vitest";
import { ManimColor, interpolate_color } from "../src/utils/color/core.js";

// Ground truths captured from original manim:
//   interpolate_color('#FF0000','#0000FF',a) for a in [0, 0.25, 0.5, 1]
//     -> ['#FF0000', '#BF003F', '#7F007F', '#0000FF']
//   interpolate_color('#123456','#654321',0.3) -> '#2A3846'
describe("manim.utils.color.core.interpolate_color", () => {
  it("matches Python hex outputs at representative alphas", () => {
    const red = new ManimColor("#FF0000");
    const blue = new ManimColor("#0000FF");
    expect(interpolate_color(red, blue, 0).toHex()).toBe("#FF0000");
    expect(interpolate_color(red, blue, 0.25).toHex()).toBe("#BF003F");
    expect(interpolate_color(red, blue, 0.5).toHex()).toBe("#7F007F");
    expect(interpolate_color(red, blue, 1).toHex()).toBe("#0000FF");
  });

  it("matches Python for arbitrary colors at alpha 0.3", () => {
    expect(
      interpolate_color(new ManimColor("#123456"), new ManimColor("#654321"), 0.3).toHex(),
    ).toBe("#2A3846");
  });

  it("does not mutate its inputs", () => {
    const first = new ManimColor("#FF0000");
    const second = new ManimColor("#0000FF");
    interpolate_color(first, second, 0.5);
    expect(first.toHex()).toBe("#FF0000");
    expect(second.toHex()).toBe("#0000FF");
  });
});
