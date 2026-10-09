import { describe, expect, test } from "vitest";
import { ThreeDCamera } from "../src/camera/three_d_camera.js";
import { DEGREES } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { DecimalNumber } from "../src/mobject/text/numbers.js";
import { Square } from "../src/mobject/geometry/polygram.js";

function makeCamera(): ThreeDCamera {
  const camera = new ThreeDCamera({
    pixel_width: 1280,
    pixel_height: 720,
    frame_width: 14.222,
    frame_height: 8,
    phi: 60 * DEGREES,
    theta: -45 * DEGREES,
  });
  camera.reset_rotation_matrix();
  return camera;
}

/** First family member that actually carries geometry. */
function leafOf(mobject: { get_family: () => Array<{ get_num_points?: () => number }> }): {
  mob: never;
  points: never;
} {
  for (const member of mobject.get_family()) {
    if ((member.get_num_points?.() ?? 0) > 0) {
      const points = (member as unknown as { points: never }).points;
      return { mob: member as never, points };
    }
  }
  throw new Error("no leaf with points");
}

describe("fixed-in-frame survives family rebuilds", () => {
  test("glyphs rebuilt by set_value keep their fixed-in-frame status", () => {
    const camera = makeCamera();
    const number = new DecimalNumber(1.0, { num_decimal_places: 2 });
    camera.add_fixed_in_frame_mobjects(number);

    const before = leafOf(number);
    const beforeOut = pointRows(
      camera.transform_points_pre_display(before.mob, before.points),
      3,
    );
    expect(beforeOut.length).toBeGreaterThan(0);
    expect(beforeOut).toEqual(pointRows(before.points, 3));

    // Rebuilding the glyphs (a new string with different characters) replaces
    // the submobjects; the per-frame hook drops the descendant cache.
    number.set_value(0.94);
    camera.reset_rotation_matrix();
    const after = leafOf(number);
    expect(after.mob).not.toBe(before.mob);
    const afterOut = pointRows(
      camera.transform_points_pre_display(after.mob, after.points),
      3,
    );
    expect(afterOut).toEqual(pointRows(after.points, 3));
  });

  test("without registration the glyphs are projected", () => {
    const camera = makeCamera();
    const number = new DecimalNumber(1.0, { num_decimal_places: 2 });
    const glyph = leafOf(number);
    const projected = pointRows(
      camera.transform_points_pre_display(glyph.mob, glyph.points),
      3,
    );
    expect(projected.length).toBeGreaterThan(0);
    expect(projected).not.toEqual(pointRows(glyph.points, 3));
  });

  test("removing a fixed-in-frame root stops the inherited status", () => {
    const camera = makeCamera();
    const number = new DecimalNumber(1.0, { num_decimal_places: 2 });
    camera.add_fixed_in_frame_mobjects(number);
    number.set_value(0.5);
    camera.reset_rotation_matrix();
    expect(camera.is_fixed_in_frame(leafOf(number).mob)).toBe(true);

    camera.remove_fixed_in_frame_mobjects(number);
    camera.reset_rotation_matrix();
    expect(camera.is_fixed_in_frame(leafOf(number).mob)).toBe(false);
  });

  test("non-numeric fixed-in-frame groups still work", () => {
    const camera = makeCamera();
    const square = new Square();
    camera.add_fixed_in_frame_mobjects(square);
    expect(camera.is_fixed_in_frame(square)).toBe(true);
    camera.remove_fixed_in_frame_mobjects(square);
    expect(camera.is_fixed_in_frame(square)).toBe(false);
  });
});
