import { array, type NDArray } from "numpy-ts";
import { Arrow } from "../mobject/geometry/line.js";
import { Dot } from "../mobject/geometry/arc.js";
import { BLACK } from "../utils/color/core.js";
import { rush_from, rush_into } from "../utils/rate_functions.js";
import { Scene } from "./scene.js";
import { matrixRows, pointFrom, pointRows, transposeMatrix } from "../utils/arrays.js";
import { update_dict_recursively } from "../utils/config_ops.js";
import { config } from "../_config/utils.js";
import { RIGHT, UP } from "../constants.js";
import { Camera } from "../camera/camera.js";
import { Create } from "../animation/creation.js";
import { Animation } from "../animation/animation.js";
import { ApplyPointwiseFunction, Transform } from "../animation/transform.js";
import type { Mobject } from "../mobject/mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import { Axes, CoordinateSystem, NumberPlane } from "../mobject/graphing/coordinate_systems.js";
import { GREY, ManimColor, WHITE, type ParsableManimColor } from "../utils/color/core.js";
import { angle_of_vector } from "../utils/space_ops.js";

export { CoordinateSystem };

export type MappingFunction = (point: number[] | NDArray) => number[] | NDArray;

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
    let vector_cleaned: number[];
    if (vector instanceof Arrow) {
      const end = pointRows(vector.get_end(), 3)[0] ?? [0, 0, 0];
      const start = pointRows(vector.get_start(), 3)[0] ?? [0, 0, 0];
      vector_cleaned = end.map((value, index) => value - (start[index] ?? 0));
    } else {
      const raw: number[] = Array.isArray(vector)
        ? [...vector]
        : pointRows(vector as NDArray, 1).map((row) => row[0] ?? 0);
      vector_cleaned = raw.length === 2 ? [...raw, 0] : raw;
    }

    const x_max = Math.trunc(Number(config.frame_x_radius) + Math.abs(vector_cleaned[0] ?? 0));
    const y_max = Math.trunc(Number(config.frame_y_radius) + Math.abs(vector_cleaned[1] ?? 0));
    // TODO:
    // I think that this should be a VGroup instead of a VMobject.
    const dots = new VMobject();
    for (let x = -x_max; x < x_max; x += 1) {
      for (let y = -y_max; y < y_max; y += 1) {
        dots.add(new Dot([x, y, 0]));
      }
    }
    dots.set_fill(BLACK, 0);
    const halfway = (vector_cleaned ?? []).map((value) => (value ?? 0) / 2);
    const dots_halfway = (dots.copy() as VMobject).shift(halfway).set_fill(WHITE, 1);
    const dots_end = (dots.copy() as VMobject).shift(vector_cleaned);

    this.play(new Transform(dots, dots_halfway as unknown as VMobject, { rate_func: rush_into }));
    this.play(new Transform(dots, dots_end as unknown as VMobject, { rate_func: rush_from }));
    this.remove(dots);
  }
}

export interface LinearTransformationSceneOptions {
  renderer?: unknown;
  camera_class?: new (...args: any[]) => any;
  always_update_mobjects?: boolean;
  [key: string]: unknown;
}

function numberPlaneFromKwargs(kwargs: Record<string, unknown>): NumberPlane {
  // NOTE: the TS NumberPlane port only supports the geometric range/length
  // options; grid-styling kwargs (axis_config, background_line_style,
  // faded_line_ratio, ...) are accepted for API compatibility and ignored.
  return new NumberPlane(
    (kwargs.x_range as number[] | null) ?? null,
    (kwargs.y_range as number[] | null) ?? null,
    (kwargs.x_length as number | null) ?? null,
    (kwargs.y_length as number | null) ?? null,
  );
}

function inverseSquareMatrix(values: number[][]): number[][] {
  // Local Gauss-Jordan inverse for the small square matrices used by
  // apply_inverse / apply_inverse_transpose (numpy.linalg.inv equivalent).
  const n = values.length;
  if (n === 0 || values.some((row) => row.length !== n)) {
    const error = new Error("Matrix has bad dimensions");
    error.name = "ValueError";
    throw error;
  }
  const augmented = values.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(augmented[row]?.[column] ?? 0) > Math.abs(augmented[pivot]?.[column] ?? 0)) pivot = row;
    }
    const pivotValue = augmented[pivot]?.[column] ?? 0;
    if (pivotValue === 0) throw new Error("Singular matrix");
    const pivotRow = augmented[pivot] as number[];
    augmented[pivot] = augmented[column] as number[];
    augmented[column] = pivotRow;
    const divisor = augmented[column]?.[column] ?? 1;
    for (let j = 0; j < 2 * n; j += 1) (augmented[column] as number[])[j] = ((augmented[column] as number[])[j] ?? 0) / divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = (augmented[row] as number[])[column] ?? 0;
      for (let j = 0; j < 2 * n; j += 1) {
        (augmented[row] as number[])[j] = ((augmented[row] as number[])[j] ?? 0) - factor * ((augmented[column] as number[])[j] ?? 0);
      }
    }
  }
  return augmented.map((row) => row.slice(n));
}

export class LinearTransformationScene extends VectorScene {
  include_background_plane: boolean;
  include_foreground_plane: boolean;
  show_coordinates: boolean;
  show_basis_vectors: boolean;
  override basis_vector_stroke_width: number;
  i_hat_color: ManimColor;
  j_hat_color: ManimColor;
  leave_ghost_vectors: boolean;
  background_plane_kwargs: Record<string, unknown>;
  foreground_plane_kwargs: Record<string, unknown>;
  ghost_vectors: VGroup;
  background_plane!: NumberPlane;
  background_mobjects!: Mobject[];
  transformable_mobjects!: Mobject[];
  moving_vectors!: Mobject[];
  transformable_labels!: any[];
  has_already_setup?: boolean;
  basis_vectors!: VGroup;
  i_hat!: Mobject;
  j_hat!: Mobject;
  square!: Mobject;
  title!: any;

  constructor(
    include_background_plane = true,
    include_foreground_plane = true,
    background_plane_kwargs: Record<string, unknown> | null = null,
    foreground_plane_kwargs: Record<string, unknown> | null = null,
    show_coordinates = false,
    show_basis_vectors = true,
    basis_vector_stroke_width = 6,
    i_hat_color: ParsableManimColor = X_COLOR,
    j_hat_color: ParsableManimColor = Y_COLOR,
    leave_ghost_vectors = false,
    options: LinearTransformationSceneOptions = {},
  ) {
    super(basis_vector_stroke_width, options as unknown as VectorSceneOptions);
    // NOTE: Python calls super().__init__(**kwargs) without forwarding
    // basis_vector_stroke_width, then assigns every field directly.
    this.include_background_plane = include_background_plane;
    this.include_foreground_plane = include_foreground_plane;
    this.show_coordinates = show_coordinates;
    this.show_basis_vectors = show_basis_vectors;
    this.basis_vector_stroke_width = basis_vector_stroke_width;
    this.i_hat_color = new ManimColor(i_hat_color);
    this.j_hat_color = new ManimColor(j_hat_color);
    this.leave_ghost_vectors = leave_ghost_vectors;
    this.background_plane_kwargs = {
      color: GREY,
      axis_config: { color: GREY },
      background_line_style: { stroke_color: GREY, stroke_width: 1 },
    };
    this.ghost_vectors = new VGroup();
    const frameWidth = config.frame_width as number;
    this.foreground_plane_kwargs = {
      x_range: [-frameWidth, frameWidth, 1.0],
      y_range: [-frameWidth, frameWidth, 1.0],
      faded_line_ratio: 1,
    };
    LinearTransformationScene.update_default_configs(
      [this.foreground_plane_kwargs, this.background_plane_kwargs],
      [foreground_plane_kwargs, background_plane_kwargs],
    );
  }

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

  setup(): void {
    // The has_already_setup attr is to not break all the old Scenes
    if ((this as { has_already_setup?: boolean }).has_already_setup) return;
    this.has_already_setup = true;
    this.background_mobjects = [];
    this.foreground_mobjects = [];
    this.transformable_mobjects = [];
    this.moving_vectors = [];
    this.transformable_labels = [];
    this.moving_mobjects = [];

    this.background_plane = numberPlaneFromKwargs(this.background_plane_kwargs);
    if (this.show_coordinates) {
      (this.background_plane as unknown as { add_coordinates: () => void }).add_coordinates();
    }
    if (this.include_background_plane) {
      this.add_background_mobject(this.background_plane as unknown as Mobject);
    }
    if (this.include_foreground_plane) {
      this.plane = numberPlaneFromKwargs(this.foreground_plane_kwargs);
      this.add_transformable_mobject(this.plane as unknown as Mobject);
    }
    if (this.show_basis_vectors) {
      this.basis_vectors = this.get_basis_vectors(
        this.i_hat_color,
        this.j_hat_color,
      ) as VGroup;
      this.moving_vectors.push(...(this.basis_vectors as unknown as Mobject[]));
      [this.i_hat, this.j_hat] = this.basis_vectors.submobjects as [Mobject, Mobject];
      this.add(this.basis_vectors as unknown as Mobject);
    }
  }

  add_special_mobjects(mob_list: Mobject[], ...mobs_to_add: Mobject[]): void {
    for (const mobject of mobs_to_add) {
      if (!mob_list.includes(mobject)) {
        mob_list.push(mobject);
        this.add(mobject);
      }
    }
  }

  add_background_mobject(...mobjects: Mobject[]): void {
    this.add_special_mobjects(this.background_mobjects, ...mobjects);
  }

  // NOTE: Python flags this override as conflicting with
  // Scene.add_foreground_mobject(s) (`# type: ignore[override]`); the TS
  // override keeps the same ignore via @ts-expect-error below.
  // @ts-expect-error -- mirrors Python `# type: ignore[override]`: variadic void intentionally narrows Scene.add_foreground_mobject
  add_foreground_mobject(...mobjects: Mobject[]): void {
    this.add_special_mobjects(this.foreground_mobjects, ...mobjects);
  }

  add_transformable_mobject(...mobjects: Mobject[]): void {
    this.add_special_mobjects(this.transformable_mobjects, ...mobjects);
  }

  add_moving_mobject(mobject: Mobject & { target?: Mobject | null }, target_mobject: Mobject | null = null): void {
    mobject.target = target_mobject;
    this.add_special_mobjects(this.moving_mobjects, mobject);
  }

  get_ghost_vectors(): VGroup {
    return this.ghost_vectors;
  }

  get_unit_square(
    color: ParsableManimColor | Iterable<ParsableManimColor> = new ManimColor("PURE_YELLOW"),
    opacity = 0.3,
    stroke_width = 3,
  ): never {
    void color;
    void opacity;
    void stroke_width;
    // NOTE: needs Rectangle (manim.mobject.geometry.polygram, not ported
    // yet) plus NumberPlane unit-size/coords helpers.
    throw new Error("TODO: Rectangle (manim.mobject.geometry.polygram) not ported yet");
  }

  add_unit_square(animate = false, kwargs: Record<string, unknown> = {}): this {
    void animate;
    void kwargs;
    // NOTE: builds on get_unit_square (Rectangle, not ported yet) and plays
    // DrawBorderThenFill (manim.animation.creation, not ported yet).
    throw new Error("TODO: Rectangle (manim.mobject.geometry.polygram) not ported yet");
  }

  override add_vector(
    vector: any,
    color: ParsableManimColor | Iterable<ParsableManimColor> = new ManimColor("PURE_YELLOW"),
    animate = false,
    kwargs: Record<string, unknown> = {},
  ): any {
    const result = super.add_vector(vector, color, animate, kwargs);
    this.moving_vectors.push(result);
    return result;
  }

  override write_vector_coordinates(vector: any, kwargs: Record<string, unknown> = {}): any {
    // NOTE: the label itself is text/tex (out of scope); the base method
    // builds and writes it, then the coords join the foreground list.
    const coords = super.write_vector_coordinates(vector, kwargs);
    this.add_foreground_mobject(coords);
    return coords;
  }

  add_transformable_label(
    vector: any,
    label: any,
    transformation_name: any = "L",
    new_label: any = null,
    kwargs: Record<string, unknown> = {},
  ): any {
    // NOTE: labels are MathTex/text (out of scope).
    const { animate: label_animate = true, ...label_kwargs } = kwargs;
    const label_mob = this.label_vector(vector, label, label_animate as boolean, label_kwargs) as any;
    if (new_label) {
      label_mob.target_text = new_label;
    } else {
      label_mob.target_text = `${transformation_name}(${label_mob.get_tex_string()})`;
    }
    label_mob.vector = vector;
    label_mob.kwargs = { ...kwargs };
    if ("animate" in (label_mob.kwargs as Record<string, unknown>)) {
      delete (label_mob.kwargs as Record<string, unknown>)["animate"];
    }
    this.transformable_labels.push(label_mob);
    return label_mob;
  }

  add_title(title: any, scale_factor = 1.5, animate = false): this {
    // NOTE: non-Mobject titles are Tex/text (out of scope).
    if (typeof title === "string") {
      void scale_factor;
      throw new Error("TODO: text/tex not ported yet");
    }
    const titleMob = title as Mobject;
    (titleMob as unknown as { to_edge: (edge: unknown) => void }).to_edge(UP);
    (titleMob as unknown as { add_background_rectangle: () => void }).add_background_rectangle();
    if (animate) {
      // NOTE: Python plays Write(title); Write (text animation) is not
      // ported yet, so a plain Animation stands in until it lands.
      this.play(new Animation(titleMob));
    }
    this.add_foreground_mobject(titleMob);
    this.title = titleMob;
    return this;
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

  get_piece_movement(pieces: Iterable<Mobject>): Transform {
    const v_pieces = [...pieces].filter((piece) => piece instanceof VMobject) as VMobject[];
    const start = new VGroup(...v_pieces);
    const target = new VGroup(...(v_pieces.map((mob) => (mob as unknown as { target: VMobject }).target) as VMobject[]));
    // don't add empty VGroups
    if (this.leave_ghost_vectors && start.submobjects.length > 0) {
      // start.copy() gives a VGroup of Vectors
      this.ghost_vectors.add(start.copy().fade(0.7) as unknown as VMobject);
      this.add(this.ghost_vectors.submobjects[this.ghost_vectors.submobjects.length - 1] as Mobject);
    }
    return new Transform(start, target, { lag_ratio: 0 });
  }

  get_moving_mobject_movement(func: MappingFunction): Transform {
    for (const m of this.moving_mobjects as Array<Mobject & { target: Mobject | null }>) {
      if (m.target === null || m.target === undefined) {
        m.target = m.copy();
      }
      const temp: number[] = m.get_center();
      const target_point = func(temp);
      (m.target as Mobject).move_to(target_point);
    }
    return this.get_piece_movement(this.moving_mobjects);
  }

  get_vector_movement(func: MappingFunction): Transform {
    for (const v of this.moving_vectors as Array<any>) {
      // NOTE: Vector (manim.mobject.geometry.line) is not ported yet; this
      // goes through `any` so it binds once Vector lands.
      const end = func(v.get_end());
      v.target = new ((v as { constructor: { Vector?: new (...args: any[]) => any } }).constructor.Vector as new (...args: any[]) => any)(end, { color: v.get_color() });
      const norm = Math.hypot(...((v.target.get_end() as number[]) ?? []));
      if (norm < 0.1) {
        v.target.get_tip().scale(norm);
      }
    }
    return this.get_piece_movement(this.moving_vectors);
  }

  get_transformable_label_movement(): Transform {
    for (const label of this.transformable_labels as Array<any>) {
      // NOTE: get_vector_label builds MathTex/text (out of scope); this
      // mirrors the Python call shape through `any` so it binds once text
      // lands. With no transformable labels (the common case until then)
      // the loop is skipped.
      const target_text: unknown = label.target_text;
      label.target = (this as unknown as {
        get_vector_label: (vector: unknown, label: unknown, kwargs?: unknown) => unknown;
      }).get_vector_label(label.vector.target, target_text, label.kwargs);
    }
    return this.get_piece_movement(this.transformable_labels);
  }

  apply_matrix(matrix: unknown, kwargs: Record<string, unknown> = {}): void {
    this.apply_transposed_matrix(transposeMatrix(matrixRows(matrix)), kwargs);
  }

  apply_inverse(matrix: unknown, kwargs: Record<string, unknown> = {}): void {
    this.apply_matrix(inverseSquareMatrix(matrixRows(matrix)), kwargs);
  }

  apply_transposed_matrix(transposed_matrix: unknown, kwargs: Record<string, unknown> = {}): void {
    const func = this.get_transposed_matrix_transformation(transposed_matrix);
    if (!("path_arc" in kwargs)) {
      const right = pointRows(func(pointFrom(RIGHT, 3)), 3)[0] ?? [1, 0, 0];
      const up = pointRows(func(pointFrom(UP, 3)), 3)[0] ?? [0, 1, 0];
      const net_rotation = (
        (angle_of_vector(right) as number) + ((angle_of_vector(up) as number) - Math.PI / 2)
      ) / 2;
      kwargs["path_arc"] = net_rotation;
    }
    this.apply_function(func, [], kwargs);
  }

  apply_inverse_transpose(t_matrix: unknown, kwargs: Record<string, unknown> = {}): void {
    const t_inv = transposeMatrix(inverseSquareMatrix(transposeMatrix(matrixRows(t_matrix))));
    this.apply_transposed_matrix(t_inv, kwargs);
  }

  apply_nonlinear_transformation(func: MappingFunction, kwargs: Record<string, unknown> = {}): void {
    (this.plane as unknown as { prepare_for_nonlinear_transform: () => void }).prepare_for_nonlinear_transform();
    this.apply_function(func, [], kwargs);
  }

  apply_function(
    func: MappingFunction,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
  ): void {
    if (!("run_time" in kwargs)) {
      kwargs["run_time"] = 3;
    }
    // NOTE: mirrors Python `ApplyPointwiseFunction(function, t_mob)`: the
    // pointwise function is passed directly; ApplyMethod applies it via
    // `target.apply_function(function)` in `create_target`.
    const anims: Animation[] = [
      ...(this.transformable_mobjects.map((t_mob) => new ApplyPointwiseFunction(func, t_mob)) as Animation[]),
      this.get_vector_movement(func),
      this.get_transformable_label_movement(),
      this.get_moving_mobject_movement(func),
      ...(this.foreground_mobjects.map((f_mob) => new Animation(f_mob)) as Animation[]),
      ...added_anims,
    ];
    // NOTE: mirrors Scene.compile_animations, which setattr's every play
    // kwarg (run_time, path_arc, ...) onto each animation. The TS Scene.play
    // has no kwargs channel, so the loop is inlined here before playing.
    for (const anim of anims) {
      for (const [key, value] of Object.entries(kwargs)) {
        (anim as unknown as Record<string, unknown>)[key] = value;
      }
    }
    this.play(...anims);
  }
}
