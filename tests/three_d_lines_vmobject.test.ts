import { describe, expect, it } from "vitest";

import { Arrow3D, Line3D, ThreeDScene, VMobject, VGroup } from "../src/index.js";
import { pointRows } from "../src/utils/arrays.js";

function centerOf(mobject: { get_center: () => unknown }): number[] {
  return pointRows(mobject.get_center(), 3)[0] ?? [0, 0, 0];
}

describe("Line3D/Arrow3D are VMobjects", () => {
  it("can be added to groups and the scene", () => {
    const line = new Line3D([0, 0, 0], [2, 2, 2]);
    const arrow = new Arrow3D([0, 0, 0], [0, 0, 2]);
    expect(line).toBeInstanceOf(VMobject);
    expect(arrow).toBeInstanceOf(VMobject);
    expect(() => new VGroup(line, arrow)).not.toThrow();

    const scene = new ThreeDScene();
    expect(() => scene.add(line, arrow)).not.toThrow();
  });

  it("copies without losing the line fields", () => {
    const line = new Line3D([0, 0, 0], [2, 2, 2], 0.02, "#ff0000");
    const copy = line.copy() as Line3D;
    expect(copy).toBeInstanceOf(Line3D);
    expect(pointRows(copy.get_start(), 3)[0]).toEqual([0, 0, 0]);
    expect(pointRows(copy.get_end(), 3)[0]).toEqual([2, 2, 2]);
    expect(copy.get_all_points().size).toBeGreaterThan(0);
    expect(copy.color.to_hex()).toBe("#FF0000");
  });

  it("builds geometry at the correct midpoint", () => {
    const line = new Line3D([0, 0, 0], [2, 2, 2]);
    expect(centerOf(line)[0]).toBeCloseTo(1, 9);
    expect(centerOf(line)[1]).toBeCloseTo(1, 9);
    expect(centerOf(line)[2]).toBeCloseTo(1, 9);
    expect(line.height).toBeGreaterThan(0);
  });

  it("rebuilds geometry when endpoints change", () => {
    const line = new Line3D([0, 0, 0], [2, 2, 2]);
    line.set_start_and_end_attrs([0, 0, 0], [4, 0, 0]);
    expect(line.get_start().toArray()).toEqual([0, 0, 0]);
    expect(line.get_end().toArray()).toEqual([4, 0, 0]);
    expect(centerOf(line)[0]).toBeCloseTo(2, 9);
    expect(line.get_all_points().size).toBeGreaterThan(0);
  });

  it("attaches the cone and end point to Arrow3D", () => {
    const arrow = new Arrow3D([0, 0, 0], [0, 0, 2]);
    expect(arrow.submobjects).toContain(arrow.cone);
    expect(arrow.submobjects).toContain(arrow.end_point);
    expect(pointRows(arrow.get_end(), 3)[0]).toEqual([0, 0, 2]);
    expect(arrow.cone.get_all_points().size).toBeGreaterThan(0);
    expect(arrow.get_all_points().size).toBeGreaterThan(0);
  });
});
