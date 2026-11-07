import { describe, expect, it } from "vitest";

import { Slider } from "../src/index.js";
import { Text } from "../src/mobject/text/text_mobject.js";
import { pointRows } from "../src/utils/arrays.js";

function knobX(slider: Slider): number {
  return pointRows(slider.knob.get_center(), 3)[0]?.[0] ?? NaN;
}

describe("Slider live value", () => {
  it("tracks the knob during animate.set_value", () => {
    const slider = new Slider({
      start: [0, 0, 0],
      end: [10, 0, 0],
      value_range: [0, 10],
      init_value: 0,
    });
    const animation: any = slider.animate.set_value(10).build();
    animation.begin();
    for (const alpha of [0.25, 0.5, 0.75]) {
      animation.interpolate(alpha);
      expect(slider.get_value()).toBeCloseTo(knobX(slider), 6);
    }
    animation.finish();
    expect(slider.get_value()).toBeCloseTo(10, 9);
  });

  it("keeps the exact stored value outside animations", () => {
    const slider = new Slider({
      start: [0, 0, 0],
      end: [10, 0, 0],
      value_range: [0, 10],
      init_value: 3,
    });
    expect(slider.get_value()).toBe(3);
    slider.set_value(7);
    expect(slider.get_value()).toBe(7);
  });

  it("ignores an attached mobject's bbox when reading the knob", () => {
    const slider = new Slider({
      start: [0, 0, 0],
      end: [10, 0, 0],
      value_range: [0, 10],
      init_value: 5,
    });
    slider.attach(new Text("long label"));
    expect(slider.get_value()).toBeCloseTo(5, 6);
  });
});
