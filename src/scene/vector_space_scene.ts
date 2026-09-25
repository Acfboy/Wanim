import { array, type NDArray } from "numpy-ts";
import { Scene } from "./scene.js";
import { matrixRows, pointFrom, pointRows, transposeMatrix } from "../utils/arrays.js";
import { update_dict_recursively } from "../utils/config_ops.js";
import { Camera } from "../camera/camera.js";
import { Create } from "../animation/creation.js";
import type { Mobject } from "../mobject/mobject.js";
import { Axes, CoordinateSystem, NumberPlane } from "../mobject/graphing/coordinate_systems.js";
import { ManimColor, WHITE, type ParsableManimColor } from "../utils/color/core.js";

export { CoordinateSystem };

export const X_COLOR = new ManimColor("GREEN_C");
export const Y_COLOR = new ManimColor("RED_C");
export const Z_COLOR = new ManimColor("BLUE_D");

// TODO: Much of this scene type seems dependent on the coordinate system chosen.
// That is, being centered at the origin with grid units corresponding to the
// arbitrary space units.  Change it!
//
// Also, methods I would have thought of as getters, like coords_to_vector, are
// actually doing a lot of animating.
export interface VectorSceneOptions {
  renderer?: unknown;
  camera_class?: new (...args: any[]) => any;
  always_update_mobjects?: boolean;
  [key: string]: unknown;
}

export class VectorScene extends Scene {
  basis_vector_stroke_width: number;
  // NOTE: Python VectorScene never assigns self.plane itself (only
  // LinearTransformationScene.setup does); get_vector() still reads it
  // behind a type-ignore. The field is declared here for the same access.
  plane!: NumberPlane;

  constructor(basis_vector_stroke_width = 6.0, options: VectorSceneOptions = {}) {
    super(
      (options.renderer as null) ?? null,
      options.camera_class,
      options.always_update_mobjects ?? false,
    );
    this.basis_vector_stroke_width = basis_vector_stroke_width;
  }

  add_plane(animate = false, kwargs: Record<string, unknown> = {}): NumberPlane {
    // NOTE: the TS NumberPlane port only supports the geometric range/length
    // options; grid-styling kwargs (axis_config, background_line_style,
    // faded_line_ratio, ...) are accepted but ignored until NumberPlane
    // itself is fully ported.
    const plane = new NumberPlane(
      (kwargs.x_range as number[] | null) ?? null,
      (kwargs.y_range as number[] | null) ?? null,
      (kwargs.x_length as number | null) ?? null,
      (kwargs.y_length as number | null) ?? null,
    );
    if (animate) {
      this.play(new Create(plane as unknown as import("../mobject/types/vectorized_mobject.js").VMobject, { lag_ratio: 0.5 }));
    }
    this.add(plane as unknown as Mobject);
    return plane;
  }

  add_axes(
    animate = false,
    color: ParsableManimColor | Iterable<ParsableManimColor> = WHITE,
  ): Axes {
    void color;
    // NOTE: the TS Axes port takes no color/axis_config options yet; they
    // are accepted for API compatibility and ignored.
    const axes = new Axes();
    if (animate) {
      this.play(new Create(axes as unknown as import("../mobject/types/vectorized_mobject.js").VMobject));
    }
    this.add(axes as unknown as Mobject);
    return axes;
  }

  lock_in_faded_grid(dimness = 0.7, axes_dimness = 0.5): void {
    // NOTE: NumberPlane.get_axes/fade and the axes-group set_color/fade
    // helpers are not ported yet (later batches); this mirrors the Python
    // call sequence through `any` so it binds once they land.
    const plane: any = this.add_plane();
    const axes: any = plane.get_axes();
    plane.fade(dimness);
    axes.set_color(WHITE);
    axes.fade(axes_dimness);
    this.add(axes);

    // TODO
    // error: Missing positional argument "scene" in call to "update_frame" of "CairoRenderer"  [call-arg]
    this.renderer.update_frame();
    // NOTE: Python rebuilds the camera from the current frame
    // (Camera(self.renderer.get_frame())); the TS Camera takes an options
    // object instead, so a fresh default camera is installed.
    this.renderer.camera = new Camera();
    this.clear();
  }

  get_vector(numerical_vector: number[] | readonly number[] | NDArray, kwargs: Record<string, unknown> = {}): any {
    void numerical_vector;
    void kwargs;
    throw new Error("TODO: Arrow (manim.mobject.geometry.line) not ported yet");
  }

  add_vector(
    vector: any,
    color: ParsableManimColor | Iterable<ParsableManimColor> = new ManimColor("PURE_YELLOW"),
    animate = true,
    kwargs: Record<string, unknown> = {},
  ): any {
    void vector;
    void color;
    void animate;
    void kwargs;
    throw new Error("TODO: Vector/GrowArrow (manim.mobject.geometry.line, manim.animation.growing) not ported yet");
  }

  write_vector_coordinates(vector: any, kwargs: Record<string, unknown> = {}): any {
    void vector;
    void kwargs;
    throw new Error("TODO: text/tex not ported yet");
  }

  get_basis_vectors(
    i_hat_color: ParsableManimColor | Iterable<ParsableManimColor> = X_COLOR,
    j_hat_color: ParsableManimColor | Iterable<ParsableManimColor> = Y_COLOR,
  ): any {
    void i_hat_color;
    void j_hat_color;
    throw new Error("TODO: Vector (manim.mobject.geometry.line) not ported yet");
  }

  get_basis_vector_labels(kwargs: Record<string, unknown> = {}): any {
    void kwargs;
    throw new Error("TODO: text/tex not ported yet");
  }

  get_vector_label(
    vector: any,
    label: any,
    at_tip = false,
    direction = "left",
    rotate = false,
    color: ParsableManimColor | null = null,
    label_scale_factor: number | null = null,
  ): any {
    void vector;
    void label;
    void at_tip;
    void direction;
    void rotate;
    void color;
    void label_scale_factor;
    throw new Error("TODO: text/tex not ported yet");
  }

  label_vector(vector: any, label: any, animate = true, kwargs: Record<string, unknown> = {}): any {
    void vector;
    void label;
    void animate;
    void kwargs;
    throw new Error("TODO: text/tex not ported yet");
  }

  position_x_coordinate(x_coord: Mobject, x_line: Mobject, vector: number[] | readonly number[] | NDArray): Mobject {
    const coords = pointRows(vector, 3)[0] ?? [0, 0, 0];
    const direction = [0, -Math.sign(coords[1] ?? 0), 0];
    x_coord.next_to(x_line, direction);
    x_coord.set_color(X_COLOR);
    return x_coord;
  }

  position_y_coordinate(y_coord: Mobject, y_line: Mobject, vector: number[] | readonly number[] | NDArray): Mobject {
    const coords = pointRows(vector, 3)[0] ?? [0, 0, 0];
    const direction = [Math.sign(coords[0] ?? 0), 0, 0];
    y_coord.next_to(y_line, direction);
    y_coord.set_color(Y_COLOR);
    return y_coord;
  }

  coords_to_vector(
    vector: number[] | readonly number[] | NDArray,
    coords_start: number[] | readonly number[] | NDArray = [2, 2, 0],
    clean_up = true,
  ): void {
    void vector;
    void coords_start;
    void clean_up;
    throw new Error("TODO: text/tex not ported yet");
  }

  vector_to_coords(
    vector: number[] | readonly number[] | NDArray,
    integer_labels = true,
    clean_up = true,
  ): any {
    void vector;
    void integer_labels;
    void clean_up;
    // NOTE: in addition to Matrix/MathTex labels (text, out of scope), this
    // needs Arrow/Vector/Line which have not been ported yet.
    throw new Error("TODO: text/tex not ported yet");
  }

  show_ghost_movement(vector: any): void {
    void vector;
    throw new Error("TODO: Dot (manim.mobject.geometry.arc) not ported yet");
  }
}

export class LinearTransformationScene extends Scene {
  static update_default_configs(
    default_configs: Iterable<Record<string, unknown>>,
    passed_configs: Iterable<Record<string, unknown> | null | undefined>,
  ): void {
    const defaults = [...default_configs];
    const passed = [...passed_configs];
    const length = Math.min(defaults.length, passed.length);
    for (let index = 0; index < length; index += 1) {
      const passed_config = passed[index];
      if (passed_config !== null && passed_config !== undefined) {
        update_dict_recursively(defaults[index] as Record<string, unknown>, passed_config);
      }
    }
  }

  get_matrix_transformation(matrix: unknown): (point: number[] | NDArray) => NDArray {
    return this.get_transposed_matrix_transformation(transposeMatrix(matrixRows(matrix)));
  }

  get_transposed_matrix_transformation(transposed_matrix: unknown): (point: number[] | NDArray) => NDArray {
    const values = matrixRows(transposed_matrix);
    let matrix = values;
    if (values.length === 2 && values.every((row) => row.length === 2)) {
      matrix = [
        [values[0][0], values[0][1], 0],
        [values[1][0], values[1][1], 0],
        [0, 0, 1],
      ];
    } else if (values.length !== 3 || values.some((row) => row.length !== 3)) {
      const error = new Error("Matrix has bad dimensions");
      error.name = "ValueError";
      throw error;
    }
    const finalMatrix = matrix.map((row) => [...row]);
    return (point: number[] | NDArray) => {
      const valuesPoint = pointFrom(point, 3).toArray() as number[];
      return array(Array.from({ length: finalMatrix[0]?.length ?? 0 }, (_, column) => finalMatrix.reduce((sum, row, rowIndex) => sum + (row[column] ?? 0) * (valuesPoint[rowIndex] ?? 0), 0))) as NDArray;
    };
  }
}
