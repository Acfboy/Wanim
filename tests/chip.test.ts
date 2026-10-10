import { describe, expect, test } from "vitest";
import { Camera } from "../src/camera/camera.js";
import { Chip } from "../src/interactive/controls.js";
import type { InteractiveContext } from "../src/interactive/controls.js";
import type { InteractiveHost } from "../src/interactive/host.js";

function contextFor(): InteractiveContext {
  return { host: { live: false } as unknown as InteractiveHost, camera: new Camera() };
}

describe("Chip", () => {
  test("toggles on press and reports its value", () => {
    const chip = new Chip("i", { value: false, color: "#00ff00" });
    expect(chip.get_value()).toBe(false);
    chip.on_press([0, 0, 0], contextFor());
    expect(chip.get_value()).toBe(true);
    chip.toggle();
    expect(chip.get_value()).toBe(false);
  });

  test("hit testing uses the chip box", () => {
    const chip = new Chip("ik circle", { value: true });
    chip.move_to([2, 1, 0]);
    expect(chip.is_hit([2, 1, 0])).toBe(true);
    expect(chip.is_hit([2.3, 1, 0])).toBe(true);
    expect(chip.is_hit([5, 5, 0])).toBe(false);
  });

  test("python kwargs bridge metadata is pinned", () => {
    expect((Chip as unknown as { __argnames__?: string[] }).__argnames__).toEqual([
      "label",
      "options",
    ]);
  });
});
