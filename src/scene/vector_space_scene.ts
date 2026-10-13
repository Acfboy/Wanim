import { array, type NDArray } from "numpy-ts";
import { Arrow, Line, Vector, type ArrowOptions, type VectorOptions } from "../mobject/geometry/line.js";
import { Rectangle } from "../mobject/geometry/polygram.js";
import { DrawBorderThenFill } from "../animation/creation.js";
import { GrowArrow } from "../animation/growing.js";
import { Dot } from "../mobject/geometry/arc.js";
import { BLACK } from "../utils/color/core.js";
import { rush_from, rush_into } from "../utils/rate_functions.js";
import { Scene } from "./scene.js";
import { is_thenable } from "../utils/scheduler.js";
import { isNDArray, matrixRows, pointFrom, pointRows, transposeMatrix } from "../utils/arrays.js";
import { update_dict_recursively } from "../utils/config_ops.js";
import { config } from "../_config/utils.js";
import { DOWN, DL, LARGE_BUFF, ORIGIN, OUT, RIGHT, SMALL_BUFF, UP } from "../constants.js";
import { Camera } from "../camera/camera.js";
import { Create, Write } from "../animation/creation.js";
import { FadeOut } from "../animation/fading.js";
import { Animation } from "../animation/animation.js";
import { ApplyFunction, ApplyPointwiseFunction, Transform } from "../animation/transform.js";
import { Group, Mobject, isPlainOptionsObject } from "../mobject/mobject.js";
import { Matrix } from "../mobject/matrix.js";
import { MathTex, Tex } from "../mobject/text/tex_mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import { Axes, CoordinateSystem, NumberPlane } from "../mobject/graphing/coordinate_systems.js";
import { GREY, ManimColor, WHITE, type ParsableManimColor } from "../utils/color/core.js";
import { angle_of_vector, _normalize_flat } from "../utils/space_ops.js";

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
  random_seed?: number | null;
  skip_animations?: boolean;
  [key: string]: unknown;
}

/**
 * The TS scene runner constructs scenes positionally (`new SceneClass({
 * renderer })`), whereas Python forwards `renderer=...` through `**kwargs`.
 * VectorScene's first parameter is a number, so detect an options bag passed
 * in its place and route it to Scene, mirroring Python's keyword forwarding.
 */
function isSceneOptionsBag(value: unknown): value is VectorSceneOptions {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return false;
  return [
    "renderer",
    "camera_class",
    "always_update_mobjects",
    "random_seed",
    "skip_animations",
  ].some((key) => key in (value as Record<string, unknown>));
}

/** Flatten a vector-like input to a plain number[] (mirrors `np.asarray`). */
function toNumberVector(vector: number[] | readonly number[] | NDArray): number[] {
  if (Array.isArray(vector)) return (vector as readonly number[]).map(Number);
  if (isNDArray(vector)) return ((vector as NDArray).toArray() as unknown[]).map(Number);
  return [];
}

export class VectorScene extends Scene {
  basis_vector_stroke_width: number;
  // NOTE: Python VectorScene never assigns self.plane itself (only
  // LinearTransformationScene.setup does); get_vector() still reads it
  // behind a type-ignore. The field is declared here for the same access.
  plane!: NumberPlane;

  constructor(
    basis_vector_stroke_width: number | VectorSceneOptions = 6.0,
    options: VectorSceneOptions = {},
  ) {
    let strokeWidth = basis_vector_stroke_width as number;
    let opts = options;
    if (isSceneOptionsBag(basis_vector_stroke_width)) {
      opts = basis_vector_stroke_width;
      strokeWidth = 6.0;
    }
    super(
      (opts.renderer as null) ?? null,
      opts.camera_class,
      opts.always_update_mobjects ?? false,
      (opts.random_seed as number | null | undefined) ?? undefined,
      (opts.skip_animations as boolean | undefined) ?? false,
    );
    this.basis_vector_stroke_width = strokeWidth;
  }

  add_plane(animate: boolean | Record<string, unknown> = false, kwargs: Record<string, unknown> = {}): NumberPlane {
    // Forward every NumberPlane keyword (including axis_config /
    // background_line_style / faded_line_ratio), mirroring Python's
    // `NumberPlane(**kwargs)`. A plain bag in the `animate` slot is the
    // Python kwargs bag (`add_plane({ background_line_style })`).
    let resolvedAnimate = animate as boolean;
    const merged: Record<string, unknown> = { ...kwargs };
    if (isPlainOptionsObject(animate)) {
      Object.assign(merged, animate);
      resolvedAnimate = false;
    }
    const plane = numberPlaneFromKwargs(merged);
    if (resolvedAnimate) {
      this.play(new Create(plane as unknown as import("../mobject/types/vectorized_mobject.js").VMobject, { lag_ratio: 0.5 }));
    }
    this.add(plane as unknown as Mobject);
    return plane;
  }

  add_axes(
    animate: boolean | Record<string, unknown> = false,
    color: ParsableManimColor | Iterable<ParsableManimColor> | Record<string, unknown> = WHITE,
    kwargs: Record<string, unknown> = {},
  ): Axes {
    // Mirrors Python `add_axes(animate=False, color=WHITE, **kwargs)`: build
    // Axes(**kwargs), set its color, optionally play Create, then add. A plain
    // bag in the animate/color slot carries the Python kwargs.
    let resolvedAnimate = animate as boolean;
    let colorOption: ParsableManimColor | Iterable<ParsableManimColor> | undefined = color as ParsableManimColor;
    const merged: Record<string, unknown> = { ...kwargs };
    if (isPlainOptionsObject(animate)) {
      Object.assign(merged, animate);
      resolvedAnimate = (merged["animate"] as boolean | undefined) ?? false;
    }
    if (isPlainOptionsObject(color)) {
      Object.assign(merged, color);
      colorOption = undefined;
    }
    const resolvedColor = (merged["color"] as ParsableManimColor | Iterable<ParsableManimColor> | undefined) ?? colorOption ?? WHITE;
    delete merged["color"];
    delete merged["animate"];
    const axes = new Axes(merged);
    axes.set_color(resolvedColor as ParsableManimColor);
    if (resolvedAnimate) {
      this.play(new Create(axes as unknown as import("../mobject/types/vectorized_mobject.js").VMobject));
    }
    this.add(axes as unknown as Mobject);
    return axes;
  }

  lock_in_faded_grid(dimness = 0.7, axes_dimness = 0.5): void {
    const plane = this.add_plane();
    const axes = plane.get_axes();
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

  get_vector(numerical_vector: number[] | readonly number[] | NDArray, kwargs: Record<string, unknown> = {}): Arrow {
    // NOTE: mirrors Python `Arrow(self.plane.coords_to_point(0, 0),
    // self.plane.coords_to_point(*numerical_vector[:2]), buff=0, **kwargs)`
    // (including the type-ignore: VectorScene never assigns self.plane
    // itself; only LinearTransformationScene.setup does).
    const plane = this.plane as unknown as {
      coords_to_point(...coords: Array<number | number[] | NDArray>): NDArray;
    };
    const coords = isNDArray(numerical_vector)
      ? [...(numerical_vector.toArray() as number[])].map(Number)
      : [...(numerical_vector as readonly number[])].map(Number);
    return new Arrow(
      plane.coords_to_point(0, 0),
      plane.coords_to_point(...coords.slice(0, 2)),
      0,
      { ...(kwargs as ArrowOptions) },
    );
  }

  add_vector(
    vector: Arrow | number[] | readonly number[] | NDArray,
    color: ParsableManimColor | Iterable<ParsableManimColor> | Record<string, unknown> = new ManimColor("PURE_YELLOW"),
    animate: boolean | Record<string, unknown> = true,
    kwargs: Record<string, unknown> = {},
  ): Arrow | Promise<Arrow> {
    // NOTE: mirrors Python `VectorScene.add_vector`: a pre-made Arrow is
    // used as-is, anything else is converted with np.asarray semantics. A
    // plain bag in the color/animate slot carries the Python kwargs
    // (`add_vector(v, { color: Y })`).
    let resolvedAnimate = animate as boolean;
    let colorOption: ParsableManimColor | Iterable<ParsableManimColor> | undefined = color as ParsableManimColor;
    const merged: Record<string, unknown> = { ...kwargs };
    if (isPlainOptionsObject(color)) {
      Object.assign(merged, color);
      colorOption = undefined;
    }
    if (isPlainOptionsObject(animate)) {
      Object.assign(merged, animate);
      resolvedAnimate = (merged["animate"] as boolean | undefined) ?? true;
    }
    const resolvedColor = (merged["color"] as ParsableManimColor | Iterable<ParsableManimColor> | undefined) ?? colorOption ?? new ManimColor("PURE_YELLOW");
    delete merged["color"];
    delete merged["animate"];
    let arrow: Arrow;
    if (vector instanceof Arrow) {
      arrow = vector;
    } else {
      arrow = new Vector(vector as number[] | NDArray, {
        color: resolvedColor as VectorOptions["color"],
        ...(merged as VectorOptions),
      });
    }
    // NOTE: `play` blocks in Python but returns a promise under the
    // real-time renderer; chain the add/return behind it so `await
    // self.add_vector(...)` (emitted by py2js) serializes like Python.
    // With no async scheduler `play` runs inline and this stays synchronous.
    if (!resolvedAnimate) {
      this.add(arrow);
      return arrow;
    }
    const played = this.play(new GrowArrow(arrow));
    const done = (): Arrow => {
      this.add(arrow);
      return arrow;
    };
    if (is_thenable(played)) return (played as Promise<unknown>).then(done);
    done();
    return arrow;
  }

  write_vector_coordinates(vector: Vector, kwargs: Record<string, unknown> = {}): Matrix | Promise<Matrix> {
    // NOTE: mirrors Python `coords = vector.coordinate_label(**kwargs)` then
    // `self.play(Write(coords))`. The coordinate_label keyword args are split
    // out of the bag so Matrix receives only its own options. The returned
    // union keeps sync callers synchronous while letting py2js `await` the
    // real-time play (see add_vector above).
    const { integer_labels, n_dim, color, ...matrixOptions } = kwargs;
    const coords = vector.coordinate_label(
      integer_labels as boolean | undefined,
      n_dim as number | undefined,
      color as ParsableManimColor | null | undefined,
      matrixOptions,
    ) as Matrix;
    const played = this.play(new Write(coords));
    if (is_thenable(played)) return (played as Promise<unknown>).then(() => coords);
    return coords;
  }

  get_basis_vectors(
    i_hat_color: ParsableManimColor | Iterable<ParsableManimColor> = X_COLOR,
    j_hat_color: ParsableManimColor | Iterable<ParsableManimColor> = Y_COLOR,
  ): VGroup {
    return new VGroup(
      ...(
        [
          [[1, 0], i_hat_color],
          [[0, 1], j_hat_color],
        ] as Array<[number[], ParsableManimColor | Iterable<ParsableManimColor>]>
      ).map(
        ([vect, color]) =>
          new Vector(array(vect) as NDArray, {
            color: color as VectorOptions["color"],
            stroke_width: this.basis_vector_stroke_width,
          }) as unknown as VMobject,
      ),
    );
  }

  get_basis_vector_labels(kwargs: Record<string, unknown> = {}): VGroup {
    // NOTE: mirrors Python `get_basis_vector_labels`: rebuild the basis
    // vectors and label them with \hat{\imath} / \hat{\jmath}.
    const basis = this.get_basis_vectors();
    const i_hat = basis.submobjects[0] as unknown as Vector;
    const j_hat = basis.submobjects[1] as unknown as Vector;
    const at_tip = (kwargs["at_tip"] as boolean | undefined) ?? false;
    const direction = (kwargs["direction"] as string | undefined) ?? "left";
    const rotate = (kwargs["rotate"] as boolean | undefined) ?? false;
    return new VGroup(
      this.get_vector_label(i_hat, "\\hat{\\imath}", at_tip, direction, rotate, X_COLOR, 1) as VMobject,
      this.get_vector_label(j_hat, "\\hat{\\jmath}", at_tip, direction, rotate, Y_COLOR, 1) as VMobject,
    );
  }

  get_vector_label(
    vector: Vector,
    label: string | Mobject,
    at_tip = false,
    direction = "left",
    rotate = false,
    color: ParsableManimColor | null = null,
    label_scale_factor: number = LARGE_BUFF - 0.2,
  ): VMobject {
    // NOTE: mirrors Python `get_vector_label` verbatim (string labels are
    // built with MathTex, everything else is used as-is).
    let label_mob: VMobject;
    if (typeof label === "string") {
      let texString = label;
      if (label.length === 1) texString = `\\vec{\\textbf{${label}}}`;
      const mathTexLabel = new MathTex(texString);
      mathTexLabel.set_color(color ?? (vector.get_color() as ParsableManimColor));
      label_mob = mathTexLabel;
    } else {
      label_mob = label as VMobject;
    }
    label_mob.scale(label_scale_factor);
    label_mob.add_background_rectangle();

    if (at_tip) {
      const vect = _normalize_flat(vector.get_vector());
      label_mob.next_to(vector.get_end(), vect, SMALL_BUFF);
    } else {
      const angle = vector.get_angle();
      if (!rotate) label_mob.rotate(-angle, [...OUT], ORIGIN);
      if (direction === "left") {
        const bottom = pointRows(label_mob.get_bottom(), 3)[0] ?? [0, 0, 0];
        label_mob.shift(bottom.map((value, i) => -value + 0.1 * (UP[i] as number)));
      } else {
        const top = pointRows(label_mob.get_top(), 3)[0] ?? [0, 0, 0];
        label_mob.shift(top.map((value, i) => -value + 0.1 * (DOWN[i] as number)));
      }
      label_mob.rotate(angle, [...OUT], ORIGIN);
      const end = pointRows(vector.get_end(), 3)[0] ?? [0, 0, 0];
      const start = pointRows(vector.get_start(), 3)[0] ?? [0, 0, 0];
      label_mob.shift(end.map((value, i) => (value - (start[i] ?? 0)) / 2));
    }
    return label_mob;
  }

  label_vector(
    vector: Vector,
    label: string | Mobject,
    animate = true,
    kwargs: Record<string, unknown> = {},
  ): VMobject | Promise<VMobject> {
    // NOTE: mirrors Python `label_vector`, which forwards every label kwarg
    // to get_vector_label and optionally plays a Write.
    const at_tip = (kwargs["at_tip"] as boolean | undefined) ?? false;
    const direction = (kwargs["direction"] as string | undefined) ?? "left";
    const rotate = (kwargs["rotate"] as boolean | undefined) ?? false;
    const color = (kwargs["color"] as ParsableManimColor | null | undefined) ?? null;
    const label_scale_factor = (kwargs["label_scale_factor"] as number | undefined) ?? (LARGE_BUFF - 0.2);
    const mathtex_label = this.get_vector_label(
      vector,
      label,
      at_tip,
      direction,
      rotate,
      color,
      label_scale_factor,
    );
    // NOTE: union return keeps sync callers synchronous while the
    // real-time `play` promise can be awaited (see add_vector above).
    if (!animate) {
      this.add(mathtex_label);
      return mathtex_label;
    }
    const played = this.play(new Write(mathtex_label, { run_time: 1 }));
    const done = (): VMobject => {
      this.add(mathtex_label);
      return mathtex_label;
    };
    if (is_thenable(played)) return (played as Promise<unknown>).then(done);
    done();
    return mathtex_label;
  }

  position_x_coordinate(x_coord: VMobject, x_line: Mobject, vector: number[] | readonly number[] | NDArray): VMobject {
    const coords = pointRows(vector, 3)[0] ?? [0, 0, 0];
    const direction = [0, -Math.sign(coords[1] ?? 0), 0];
    x_coord.next_to(x_line, direction);
    x_coord.set_color(X_COLOR);
    return x_coord;
  }

  position_y_coordinate(y_coord: VMobject, y_line: Mobject, vector: number[] | readonly number[] | NDArray): VMobject {
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
    // NOTE: mirrors Python `coords_to_vector`: write the vector's column
    // matrix, peel the components out to the axes and finally draw the
    // vector. Python's `Matrix(vector)` is handed a 1-D vector and would
    // raise (`'int' object is not iterable`); the column reshape below is the
    // shape `vector_to_coords`/`coordinate_label` use and is what the method
    // intends.
    const vec = toNumberVector(vector);
    const starting_mobjects = [...this.mobjects];
    const array = new Matrix(vec.map((value) => [value]));
    array.shift(pointRows(coords_start, 3)[0] ?? [0, 0, 0]);
    const arrow = new Vector(vec);
    const xLine = new Line(ORIGIN, [(vec[0] ?? 0) * (RIGHT[0] as number), 0, 0]);
    const yLine = new Line(xLine.get_end(), arrow.get_end());
    xLine.set_color(X_COLOR);
    yLine.set_color(Y_COLOR);
    const mob_matrix = array.get_mob_matrix();
    const x_coord = (mob_matrix[0] as VMobject[])[0] as VMobject;
    const y_coord = (mob_matrix[1] as VMobject[])[0] as VMobject;

    this.play(new Write(array, { run_time: 1 }));
    this.wait();
    this.play(new ApplyFunction((x) => this.position_x_coordinate(x as VMobject, xLine, vec), x_coord));
    this.play(new Create(xLine));
    this.play(
      new ApplyFunction((y) => this.position_y_coordinate(y as VMobject, yLine, vec), y_coord),
      new FadeOut(array.get_brackets()),
    );
    this.play(new Create(yLine));
    this.play(new Create(arrow));
    this.wait();
    if (clean_up) {
      this.clear();
      this.add(...starting_mobjects);
    }
  }

  vector_to_coords(
    vector: Vector | number[] | readonly number[] | NDArray,
    integer_labels = true,
    clean_up = true,
  ): [Matrix, Line, Line] {
    // NOTE: mirrors Python `vector_to_coords` (including the played
    // component-line/vector sequence).
    const starting_mobjects = [...this.mobjects];
    let arrow: Vector;
    let vec: number[];
    let show_creation = false;
    if (vector instanceof Arrow) {
      arrow = vector as Vector;
      vec = (pointRows(arrow.get_end(), 3)[0] ?? [0, 0, 0]).slice(0, 2);
    } else {
      vec = toNumberVector(vector as number[] | readonly number[] | NDArray);
      arrow = new Vector(vec);
      show_creation = true;
    }
    const array = arrow.coordinate_label(integer_labels) as Matrix;
    const xLine = new Line(ORIGIN, [(vec[0] ?? 0) * (RIGHT[0] as number), 0, 0]);
    const yLine = new Line(xLine.get_end(), arrow.get_end());
    xLine.set_color(X_COLOR);
    yLine.set_color(Y_COLOR);
    const temp = array.get_entries();
    const x_coord = temp.submobjects[0] as VMobject;
    const y_coord = temp.submobjects[1] as VMobject;
    const x_coord_start = this.position_x_coordinate(x_coord.copy(), xLine, vec);
    const y_coord_start = this.position_y_coordinate(y_coord.copy(), yLine, vec);
    const brackets = array.get_brackets();

    if (show_creation) {
      this.play(new Create(arrow));
    }
    this.play(new Create(xLine), new Write(x_coord_start, { run_time: 1 }));
    this.play(new Create(yLine), new Write(y_coord_start, { run_time: 1 }));
    this.wait();
    this.play(
      new Transform(x_coord_start, x_coord, { lag_ratio: 0 }),
      new Transform(y_coord_start, y_coord, { lag_ratio: 0 }),
      new Write(brackets, { run_time: 1 }),
    );
    this.wait();

    this.remove(x_coord_start, y_coord_start, brackets);
    this.add(array);
    if (clean_up) {
      this.clear();
      this.add(...starting_mobjects);
    }
    return [array, xLine, yLine];
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
  // Mirror Python's `NumberPlane(**kwargs)`: split out the positional
  // parameters and forward the remaining kwargs (axis_config, y_axis_config,
  // color, ...) through the trailing options bag.
  const {
    x_range = null,
    y_range = null,
    x_length = null,
    y_length = null,
    background_line_style = null,
    faded_line_style = null,
    faded_line_ratio = 1,
    make_smooth_after_applying_functions = true,
    axis_config = null,
    y_axis_config = null,
    ...rest
  } = kwargs;
  const plane = new NumberPlane(
    x_range as number[] | null,
    y_range as number[] | null,
    x_length as number | null,
    y_length as number | null,
    sanitizeStyleColors(background_line_style) as Record<string, unknown> | null,
    sanitizeStyleColors(faded_line_style) as Record<string, unknown> | null,
    faded_line_ratio as number,
    make_smooth_after_applying_functions as boolean,
    {
      axis_config: sanitizeStyleColors(axis_config),
      y_axis_config: sanitizeStyleColors(y_axis_config),
      ...rest,
    } as Record<string, unknown>,
  );
  // Python's NumberLine receives `color` through axis_config and paints the
  // axes with it; the TS NumberLine drops that key, so apply it here (the
  // background/faded lines keep their own background_line_style colours,
  // matching the Python ordering where the lines are styled afterwards).
  const axisColor = (axis_config as Record<string, unknown> | null)?.color;
  const yAxisColor = (y_axis_config as Record<string, unknown> | null)?.color;
  if (axisColor !== undefined && axisColor !== null) {
    (plane.get_axes() as unknown as Mobject).set_color(axisColor as never);
  }
  if (yAxisColor !== undefined && yAxisColor !== null && yAxisColor !== axisColor) {
    (plane.get_y_axis() as unknown as Mobject).set_color(yAxisColor as never);
  }
  return plane;
}

/**
 * `merge_dicts_recursively` (utils/config_ops) only treats plain dictionaries
 * as mergeable in Python; the TS port currently recurses into any object, so a
 * `ManimColor` colliding with the default `stroke_color` recurses forever.
 * Colour values in style configs are therefore flattened to hex strings here
 * (they are re-parsed by `set_style`), which avoids the collision while
 * producing identical final colours.
 */
function sanitizeStyleColors(style: unknown): unknown {
  if (style === null || style === undefined || typeof style !== "object") return style;
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(style as Record<string, unknown>)) {
    result[key] = value instanceof ManimColor ? value.to_hex() : value;
  }
  return result;
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
    include_background_plane: boolean | LinearTransformationSceneOptions = true,
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
    // The TS scene runner constructs scenes positionally with a leading
    // options bag (`new SceneClass({ renderer })`); Python forwards
    // `renderer=...` as a keyword. Route that bag to Scene (see VectorScene).
    if (isSceneOptionsBag(include_background_plane)) {
      options = include_background_plane;
      include_background_plane = true;
    }
    const backgroundFlag = include_background_plane as boolean;
    super(basis_vector_stroke_width, options as unknown as VectorSceneOptions);
    // NOTE: Python calls super().__init__(**kwargs) without forwarding
    // basis_vector_stroke_width, then assigns every field directly.
    this.include_background_plane = backgroundFlag;
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
  ): Rectangle {
    const single = color as ParsableManimColor;
    const square = new Rectangle({
      color: single,
      width: this.plane.get_x_unit_size(),
      height: this.plane.get_y_unit_size(),
      stroke_color: single,
      stroke_width,
      fill_color: single,
      fill_opacity: opacity,
    });
    square.move_to(this.plane.coords_to_point(0, 0), [...DL]);
    this.square = square;
    return square;
  }

  add_unit_square(animate = false, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    const { color, opacity, stroke_width, ...rest } = kwargs as {
      color?: ParsableManimColor | Iterable<ParsableManimColor>;
      opacity?: number;
      stroke_width?: number;
    };
    void rest;
    const square = this.get_unit_square(
      color ?? new ManimColor("PURE_YELLOW"),
      opacity ?? 0.3,
      stroke_width ?? 3,
    );
    // NOTE: union return keeps sync callers synchronous while the real-time
    // `play` promise can be awaited (see VectorScene.add_vector).
    const done = (): this => {
      this.add_transformable_mobject(square);
      this.bring_to_front(...this.moving_vectors);
      this.square = square;
      return this;
    };
    if (animate) {
      const played = this.play(
        new DrawBorderThenFill(square),
        new Animation(new Group(...this.moving_vectors)),
      );
      if (is_thenable(played)) return (played as Promise<unknown>).then(done);
      done();
      return this;
    }
    done();
    return this;
  }

  override add_vector(
    vector: any,
    color: ParsableManimColor | Iterable<ParsableManimColor> | Record<string, unknown> = new ManimColor("PURE_YELLOW"),
    animate: boolean | Record<string, unknown> = false,
    kwargs: Record<string, unknown> = {},
  ): any {
    // NOTE: super may return a promise under the real-time renderer; register
    // the resolved arrow so `moving_vectors` never holds a promise.
    const result = super.add_vector(vector, color, animate, kwargs) as Arrow | Promise<Arrow>;
    if (is_thenable(result)) {
      return (result as Promise<Arrow>).then((arrow) => {
        this.moving_vectors.push(arrow as unknown as Mobject);
        return arrow;
      });
    }
    this.moving_vectors.push(result as unknown as Mobject);
    return result;
  }

  override write_vector_coordinates(vector: Vector, kwargs: Record<string, unknown> = {}): Matrix | Promise<Matrix> {
    // NOTE: mirrors Python: the base method builds and writes the Matrix
    // label, then the coords join the foreground list.
    const result = super.write_vector_coordinates(vector, kwargs);
    if (is_thenable(result)) {
      return (result as Promise<Matrix>).then((coords) => {
        this.add_foreground_mobject(coords);
        return coords;
      });
    }
    this.add_foreground_mobject(result);
    return result;
  }

  add_transformable_label(
    vector: Vector,
    label: string | Mobject,
    transformation_name: string | Mobject = "L",
    new_label: string | Mobject | null = null,
    kwargs: Record<string, unknown> = {},
  ): any {
    const { animate: label_animate = true, ...label_kwargs } = kwargs;
    // NOTE: label_vector may return a promise under the real-time renderer;
    // finish the bookkeeping once it resolves (see VectorScene.add_vector).
    const finish = (label_mob: any): any => {
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
    };
    const label_mob = this.label_vector(vector, label, label_animate as boolean, label_kwargs);
    if (is_thenable(label_mob)) return (label_mob as Promise<VMobject>).then(finish);
    return finish(label_mob as any);
  }

  add_title(title: Mobject | string, scale_factor = 1.5, animate = false): this | Promise<this> {
    // NOTE: mirrors Python `add_title`: non-Mobject titles are built with
    // Tex and scaled; then it is moved to the top, backgrounded and (if
    // requested) written in. Union return keeps sync callers synchronous
    // while the real-time `play` promise can be awaited (see add_vector).
    let titleMob: VMobject;
    if (title instanceof Mobject) {
      titleMob = title as VMobject;
    } else {
      titleMob = new Tex(String(title)).scale(scale_factor);
    }
    titleMob.to_edge(UP);
    titleMob.add_background_rectangle();
    const done = (): this => {
      this.add_foreground_mobject(titleMob);
      this.title = titleMob;
      return this;
    };
    if (animate) {
      const played = this.play(new Write(titleMob));
      if (is_thenable(played)) return (played as Promise<unknown>).then(done);
      done();
      return this;
    }
    done();
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
      const temp: number[] = pointRows(m.get_center(), 3)[0] ?? [0, 0, 0];
      const target_point = func(temp);
      (m.target as Mobject).move_to(target_point);
    }
    return this.get_piece_movement(this.moving_mobjects);
  }

  get_vector_movement(func: MappingFunction): Transform {
    for (const v of this.moving_vectors as Array<Arrow & { target: Arrow }>) {
      v.target = new Vector(func(v.get_end()), {
        color: v.get_color() as VectorOptions["color"],
      });
      // NOTE: mirrors Python `float(np.linalg.norm(v.target.get_end()))`
      // — the norm of the end *point* (not the arrow length).
      const end = v.target.get_end();
      const norm = Math.hypot(...(Array.isArray(end) ? end : [...(end as unknown as number[])]));
      if (norm < 0.1) {
        v.target.get_tip().scale(norm);
      }
    }
    return this.get_piece_movement(this.moving_vectors);
  }

  get_transformable_label_movement(): Transform {
    for (const label of this.transformable_labels as Array<any>) {
      // NOTE: mirrors Python `self.get_vector_label(label.vector.target,
      // target_text, **label.kwargs)`: the label kwargs are bound to the
      // fixed get_vector_label positions.
      const target_text: unknown = label.target_text;
      const kw = (label.kwargs ?? {}) as Record<string, unknown>;
      label.target = this.get_vector_label(
        label.vector.target as Vector,
        target_text as string | Mobject,
        (kw["at_tip"] as boolean | undefined) ?? false,
        (kw["direction"] as string | undefined) ?? "left",
        (kw["rotate"] as boolean | undefined) ?? false,
        (kw["color"] as ParsableManimColor | null | undefined) ?? null,
        (kw["label_scale_factor"] as number | undefined) ?? (LARGE_BUFF - 0.2),
      );
    }
    return this.get_piece_movement(this.transformable_labels);
  }

  // NOTE: `play` blocks in Python but returns a promise under the real-time
  // renderer. These helpers return its result so py2js (`await
  // self.apply_matrix(...)`) serializes them like Python; with no async
  // scheduler `play` runs inline and everything stays synchronous.
  apply_matrix(matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    return this.apply_transposed_matrix(transposeMatrix(matrixRows(matrix)), kwargs);
  }

  apply_inverse(matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    return this.apply_matrix(inverseSquareMatrix(matrixRows(matrix)), kwargs);
  }

  apply_transposed_matrix(transposed_matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    const func = this.get_transposed_matrix_transformation(transposed_matrix);
    if (!("path_arc" in kwargs)) {
      const right = pointRows(func(pointFrom(RIGHT, 3)), 3)[0] ?? [1, 0, 0];
      const up = pointRows(func(pointFrom(UP, 3)), 3)[0] ?? [0, 1, 0];
      const net_rotation = (
        (angle_of_vector(right) as number) + ((angle_of_vector(up) as number) - Math.PI / 2)
      ) / 2;
      kwargs["path_arc"] = net_rotation;
    }
    return this.apply_function(func, [], kwargs);
  }

  apply_inverse_transpose(t_matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    const t_inv = transposeMatrix(inverseSquareMatrix(transposeMatrix(matrixRows(t_matrix))));
    return this.apply_transposed_matrix(t_inv, kwargs);
  }

  apply_nonlinear_transformation(func: MappingFunction, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    (this.plane as unknown as { prepare_for_nonlinear_transform: () => void }).prepare_for_nonlinear_transform();
    return this.apply_function(func, [], kwargs);
  }

  apply_function(
    func: MappingFunction,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
  ): this | Promise<this> {
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
    return this.play(...anims);
  }
}
