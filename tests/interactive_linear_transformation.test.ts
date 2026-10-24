import { describe, expect, test } from "vitest";
import { DraggablePoint } from "../src/interactive/controls.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { InteractiveLinearTransformationScene } from "../src/scene/interactive_linear_transformation_scene.js";
import { YELLOW } from "../src/utils/color/index.js";

function rows(value: unknown): number[][] {
  const raw = (value as { toArray?: () => unknown }).toArray?.() ?? value;
  if (!Array.isArray(raw)) return [];
  return Array.isArray(raw[0]) ? (raw as number[][]) : [raw as number[]];
}

describe("DraggablePoint", () => {
  test("starts at the requested point and reports it as an array", () => {
    const point = new DraggablePoint([1, 2, 0], { color: YELLOW, radius: 0.1 });
    expect(rows(point.get_center())[0]!.slice(0, 2)).toEqual([1, 2]);
    expect(rows(point.get_center())[0]!.slice(0, 2)).toEqual([1, 2]);

    point.set_value([-2, 3, 0]);
    expect(rows(point.get_center())[0]!.slice(0, 2)).toEqual([-2, 3]);
  });

  test("drag moves the point by the pointer delta and notifies listeners", () => {
    const point = new DraggablePoint([0, 0, 0]);
    const seen: Array<[unknown, unknown]> = [];
    point.on_input((_control, previous, next) => seen.push([previous, next]));

    // Pressing off-centre must not jump: the dot keeps its offset.
    point.on_press([0.05, 0.02, 0]);
    point.on_drag([1.05, 0.02, 0]);
    expect(rows(point.get_center())[0]!.slice(0, 2)).toEqual([1, 0]);
    expect(seen).toHaveLength(1);
    expect(rows(seen[0]![1])[0]!.slice(0, 2)).toEqual([1, 0]);
  });

  test("hit testing follows the current centre", () => {
    const point = new DraggablePoint([2, 1, 0], { radius: 0.1, hit_radius: 0.3 });
    expect(point.is_hit([2.1, 1.1, 0])).toBe(true);
    expect(point.is_hit([2.5, 1.5, 0])).toBe(false);
    point.set_value([-1, -1, 0]);
    expect(point.is_hit([2.1, 1.1, 0])).toBe(false);
    expect(point.is_hit([-1.1, -0.9, 0])).toBe(true);
  });

  test("attach makes a mobject follow the point", () => {
    const point = new DraggablePoint([1, 1, 0]);
    const label = new Dot([1, 1.5, 0]);
    point.attach(label);
    expect(point.attached).toEqual([label]);

    point.set_value([3, 4, 0]);
    expect(rows(label.get_center())[0]!.slice(0, 2)).toEqual([3, 4.5]);

    point.unattach(label);
    point.set_value([0, 0, 0]);
    expect(rows(label.get_center())[0]!.slice(0, 2)).toEqual([3, 4.5]);
  });

  test("snap rounds dragged coordinates", () => {
    const point = new DraggablePoint([0, 0, 0], { snap: [1, 0.5] });
    point.set_value([1.2, 0.4, 0]);
    expect(rows(point.get_center())[0]!.slice(0, 2)).toEqual([1, 0.5]);
  });
});

describe("InteractiveLinearTransformationScene", () => {
  test("sets up the planes and basis vectors like LinearTransformationScene", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    expect(scene.background_plane).toBeDefined();
    expect(scene.plane).toBeDefined();
    expect(scene.i_hat).toBeDefined();
    expect(scene.j_hat).toBeDefined();
    expect(scene.basis_handles).toHaveLength(2);
    expect(scene.get_matrix()).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  test("set_matrix applies the map to transformable mobjects and basis tips", () => {
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
    expect(rows(scene.basis_handles[0]!.get_center())[0]!.slice(0, 2)).toEqual([2, 0]);
    expect(rows(scene.basis_handles[1]!.get_center())[0]!.slice(0, 2)).toEqual([0, 3]);
  });

  test("dragging a basis handle updates its matrix column and the plane", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    const handle = scene.basis_handles[0]!;
    handle.on_press([1, 0, 0]);
    handle.on_drag([3, 2, 0]);
    handle.on_release();
    expect(scene.get_matrix()[0]![0]).toBeCloseTo(3, 10);
    expect(scene.get_matrix()[1]![0]).toBeCloseTo(2, 10);
    expect(rows(handle.get_center())[0]!.slice(0, 2)).toEqual([3, 2]);
  });

  test("draggable point image and vector image follow the matrix and the drag", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.set_matrix([
      [2, 0],
      [0, 2],
    ]);
    const control = scene.add_draggable_point([1, 1, 0]);
    const record = scene.draggable_points[0]!;
    const firstImage = rows(record.image!.get_center())[0]!;
    expect(firstImage[0]).toBeCloseTo(2, 10);
    expect(firstImage[1]).toBeCloseTo(2, 10);

    control.on_press([1, 1, 0]);
    control.on_drag([2, 1, 0]);
    control.on_release();
    const secondImage = rows(record.image!.get_center())[0]!;
    expect(secondImage[0]).toBeCloseTo(4, 10);
    expect(secondImage[1]).toBeCloseTo(2, 10);

    const vectorControl = scene.add_draggable_vector([1, 0, 0]);
    const vectorRecord = scene.draggable_vectors[0]!;
    const initialVector = rows(vectorRecord.image!.get_vector())[0]!;
    expect(initialVector[0]).toBeGreaterThan(0);
    expect(initialVector[1]).toBeCloseTo(0, 8);

    vectorControl.on_press([1, 0, 0]);
    vectorControl.on_drag([0, 1, 0]);
    vectorControl.on_release();
    const imageVector = rows(vectorRecord.image!.get_vector())[0]!;
    expect(imageVector[0]).toBeCloseTo(0, 8);
    expect(imageVector[1]).toBeGreaterThan(0);
  });

  test("matrix control cells drive the live matrix and stay in sync", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    const control = scene.add_matrix_control();
    const cells = control.mob_matrix;
    cells[0]![0]!.set_value(2);
    // The cell is an interactive input whose listener is private to the scene;
    // drive it the way the live host does, through the callback the scene set.
    (scene as unknown as { _on_matrix_control_input: () => void })._on_matrix_control_input();
    expect(scene.get_matrix()[0]![0]).toBe(2);

    // Dragging a basis tip writes the value back into the editor.
    const handle = scene.basis_handles[0]!;
    handle.on_press([2, 0, 0]);
    handle.on_drag([0, 3, 0]);
    handle.on_release();
    expect(cells[0]![0]!.get_value()).toBeCloseTo(0, 8);
    expect(cells[1]![0]!.get_value()).toBeCloseTo(3, 8);
  });

  test("a singular matrix collapses a basis arrow instead of warning", () => {
    const scene = new InteractiveLinearTransformationScene();
    scene.setup();
    scene.set_matrix([
      [1, 0],
      [0, 0],
    ]);
    expect(scene.j_hat.get_stroke_opacity()).toBe(0);
    expect(scene.basis_handles[1]!.get_center()[1]).toBeCloseTo(0, 10);
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
});
