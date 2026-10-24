import { describe, expect, test } from "vitest";
import { DraggablePoint, isInteractiveControl } from "../src/interactive/controls.js";
import type { Arrow } from "../src/mobject/geometry/line.js";
import { InteractiveLinearTransformationScene } from "../src/scene/interactive_linear_transformation_scene.js";
import { BLUE } from "../src/utils/color/index.js";

function rows(value: unknown): number[][] {
  const raw = (value as { toArray?: () => unknown }).toArray?.() ?? value;
  if (!Array.isArray(raw)) return [];
  return Array.isArray(raw[0]) ? (raw as number[][]) : [raw as number[]];
}

describe("InteractiveLinearTransformationScene", () => {
  test("sets up the planes and basis vectors without adding any built-in controls", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    expect(scene.background_plane).toBeDefined();
    expect(scene.plane).toBeDefined();
    expect(scene.i_hat).toBeDefined();
    expect(scene.j_hat).toBeDefined();
    expect(scene.get_matrix()).toEqual([
      [1, 0],
      [0, 1],
    ]);

    // Interaction is application logic: the scene must not install drag
    // handles, editors or any other interactive control on its own.
    const controls = scene.mobjects.flatMap((mobject) =>
      mobject.get_family().filter(isInteractiveControl),
    );
    expect(controls).toHaveLength(0);
    expect("basis_handles" in scene).toBe(false);
    expect("matrix_control" in scene).toBe(false);
  });

  test("set_matrix applies the map to transformable mobjects and basis vectors", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.add_unit_square(false);
    scene.set_matrix([
      [2, 0],
      [0, 3],
    ]);
    const center = rows(scene.square.get_center())[0]!;
    expect(center[0]).toBeCloseTo(1, 10);
    expect(center[1]).toBeCloseTo(1.5, 10);

    // The basis vectors point at the matrix columns (the arrow body stops just
    // short of the tip, so compare directions through the body vector).
    const iVector = rows(scene.i_hat.get_vector())[0]!;
    expect(iVector[0]).toBeGreaterThan(1.5);
    expect(iVector[1]).toBeCloseTo(0, 10);
    const jVector = rows(scene.j_hat.get_vector())[0]!;
    expect(jVector[0]).toBeCloseTo(0, 10);
    expect(jVector[1]).toBeGreaterThan(2.4);
  });

  test("a user-wired DraggablePoint can drive a matrix column", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.add_unit_square(false);

    const handle = new DraggablePoint([1, 0, 0]);
    const update = (): void => {
      const point = rows(handle.get_center())[0]!;
      const matrix = scene.get_matrix();
      matrix[0]![0] = point[0]!;
      matrix[1]![0] = point[1]!;
      scene.set_matrix(matrix);
    };
    handle.on_input(update);
    scene.add(handle);

    handle.on_press([1, 0, 0]);
    handle.on_drag([3, 2, 0]);
    handle.on_release();
    expect(scene.get_matrix()[0]![0]).toBeCloseTo(3, 10);
    expect(scene.get_matrix()[1]![0]).toBeCloseTo(2, 10);
    // M = [[3, 0], [2, 1]]: the unit square's centre (0.5, 0.5) -> (1.5, 1.5).
    const center = rows(scene.square.get_center())[0]!;
    expect(center[0]).toBeCloseTo(1.5, 10);
    expect(center[1]).toBeCloseTo(1.5, 10);
  });

  test("add_vector follows the current matrix", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    const arrow = scene.add_vector([2, 1, 0], BLUE, false) as Arrow;
    let vector = rows(arrow.get_vector())[0]!;
    expect(vector[0]).toBeGreaterThan(1.5);
    expect(vector[1]).toBeGreaterThan(0.6);

    scene.set_matrix([
      [0, -1],
      [1, 0],
    ]);
    vector = rows(arrow.get_vector())[0]!;
    expect(vector[0]).toBeLessThan(0);
    expect(vector[1]).toBeGreaterThan(1.5);
  });

  test("a singular matrix collapses a basis arrow instead of warning", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.set_matrix([
      [1, 0],
      [0, 0],
    ]);
    expect(scene.j_hat.get_stroke_opacity()).toBe(0);
    scene.set_matrix([
      [1, 0],
      [0, 1],
    ]);
    expect(scene.j_hat.get_stroke_opacity()).toBe(1);
  });

  test("apply_inverse installs the inverse matrix", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.set_matrix([
      [2, 1],
      [0, 3],
    ]);
    scene.apply_inverse([
      [2, 1],
      [0, 3],
    ]);
    expect(scene.get_matrix()[0]![0]).toBeCloseTo(0.5, 10);
    expect(scene.get_matrix()[0]![1]).toBeCloseTo(-1 / 6, 10);
    expect(scene.get_matrix()[1]![0]).toBeCloseTo(0, 10);
    expect(scene.get_matrix()[1]![1]).toBeCloseTo(1 / 3, 10);
  });

  test("presentation_reset returns to the initial matrix and rebuilds geometry", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.set_matrix([
      [2, 0],
      [0, 2],
    ]);
    scene.presentation_reset();
    expect(scene.get_matrix()).toEqual([
      [1, 0],
      [0, 1],
    ]);
    scene.setup();
    expect(scene.plane).toBeDefined();
    expect(scene.i_hat).toBeDefined();
  });
});
