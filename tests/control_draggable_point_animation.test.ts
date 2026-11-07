import { describe, expect, it } from "vitest";

import { DraggablePoint } from "../src/index.js";
import { Text } from "../src/mobject/text/text_mobject.js";
import { pointRows } from "../src/utils/arrays.js";

function ownCenter(point: DraggablePoint): number[] {
  const rows = pointRows(point.points, 3);
  const xs = rows.map((row) => row[0] ?? 0);
  const ys = rows.map((row) => row[1] ?? 0);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, 0];
}

describe("DraggablePoint live centre", () => {
  it("tracks the dot geometry during animate.shift", () => {
    const point = new DraggablePoint([0, 0, 0]);
    const animation: any = point.animate.shift([2, 0, 0]).build();
    animation.begin();
    for (const alpha of [0.25, 0.5, 0.75]) {
      animation.interpolate(alpha);
      const live = ownCenter(point);
      expect(point.get_center()[0]).toBeCloseTo(live[0] ?? 0, 6);
      expect(point.get_center()[1]).toBeCloseTo(live[1] ?? 0, 6);
    }
    animation.finish();
    expect(point.get_center()[0]).toBeCloseTo(2, 9);
  });

  it("still reports the anchor exactly when a label is attached", () => {
    const point = new DraggablePoint([0, 0, 0]);
    point.attach(new Text("a long label"));
    point.set_value([1.25, -0.5, 0]);
    expect(point.get_center()[0]).toBeCloseTo(1.25, 9);
    expect(point.get_center()[1]).toBeCloseTo(-0.5, 9);
  });
});
