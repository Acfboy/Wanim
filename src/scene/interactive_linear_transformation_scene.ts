/**
 * `InteractiveLinearTransformationScene`: a live, control-agnostic counterpart
 * of manim's `LinearTransformationScene`.
 *
 * The geometry is the one readers know from the classic scene (a static grey
 * background grid, a transformable foreground grid, the `i_hat` / `j_hat` basis
 * vectors), but the current linear map is a piece of state instead of a one-shot
 * `self.play(...)`:
 *
 *  - `set_matrix(A)` / `apply_matrix(A)` install `A` **immediately** and redraw
 *    the foreground plane, the basis vectors and every tracked vector;
 *  - the matrix can be driven from any wanim control by calling `set_matrix`
 *    from a blueprint reaction or an `on_input` listener;
 *  - `add_vector(v)` draws `M @ v` and keeps it in sync;
 *  - `add_transformable_mobject(...)` / `add_moving_mobject(...)` register any
 *    other source-space mobject that the matrix should transform.
 *
 * Deliberately **no interaction is built in**: the class does not add drag
 * handles, matrix editors or other controls. That is application logic, so the
 * scene author wires their own `DraggablePoint` / `Slider` /
 * `InputDecimalMatrix` to `set_matrix` (see the docs for the pattern). This keeps
 * the scene usable for very different stories: dragging a basis tip, typing a
 * matrix, animating between maps, syncing to an external model, etc.
 *
 * Everything is a normal mobject and can be styled/positioned like any other
 * wanim object. The class intentionally borrows (`prototype.call`) the setup and
 * helpers of `LinearTransformationScene`/`VectorScene` rather than inheriting
 * from them: `InteractiveScene` owns the storyboard/control machinery, and a
 * single-inheritance TS hierarchy cannot carry both bases.
 *
 * NOTE: in `blueprint` every transformation is immediate (no animation), so
 * it is safe to call them from reactions and `on_input` listeners. In
 * `construct` the `apply_*` family animates exactly like
 * `LinearTransformationScene` and keeps the matrix state in sync, so a
 * scripted `self.apply_matrix(...)` plays as a transition.
 */
import { array, type NDArray } from "numpy-ts";
import { GREY, ManimColor, type ParsableManimColor } from "../utils/color/index.js";
import { ORIGIN, PI, RIGHT, UP } from "../constants.js";
import { config } from "../_config/utils.js";
import { isNDArray, matrixRows, pointFrom, pointRows, transposeMatrix } from "../utils/arrays.js";
import { angle_of_vector } from "../utils/space_ops.js";
import { is_thenable } from "../utils/scheduler.js";
import { Arrow, type Vector } from "../mobject/geometry/line.js";
import { Rectangle } from "../mobject/geometry/polygram.js";
import { Mobject } from "../mobject/mobject.js";
import { VGroup } from "../mobject/types/vectorized_mobject.js";
import type { NumberPlane } from "../mobject/graphing/coordinate_systems.js";
import type { Animation } from "../animation/animation.js";
import type { Transform } from "../animation/transform.js";
import { InteractiveScene, type InteractiveSceneOptions } from "./interactive_scene.js";
import {
  LinearTransformationScene,
  VectorScene,
  X_COLOR,
  Y_COLOR,
  type MappingFunction,
} from "./vector_space_scene.js";

export interface InteractiveLinearTransformationSceneOptions extends InteractiveSceneOptions {
  /** Initial 2x2 matrix (defaults to the identity). */
  matrix?: unknown;
  [key: string]: unknown;
}

/** A tracked `add_vector` arrow, with its untransformed endpoints. */
interface PlainVectorRecord {
  arrow: Arrow;
  start: [number, number, number];
  end: [number, number, number];
  opacity: number;
}

/** Per-mobject point snapshot used to re-apply the matrix from scratch. */
interface TransformableSnapshot {
  members: Mobject[];
  points: number[][][];
}

function isOptionsBag(value: unknown): value is InteractiveLinearTransformationSceneOptions {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return false;
  return [
    "renderer",
    "camera_class",
    "always_update_mobjects",
    "random_seed",
    "skip_animations",
    "matrix",
  ].some((key) => key in (value as Record<string, unknown>));
}

function identity2x2(): number[][] {
  return [
    [1, 0],
    [0, 1],
  ];
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

/**
 * Normalize `matrix` to a 2x2 list of lists. A flat four-element list, a 2x2
 * numpy array and a 3x3 homogeneous matrix with a `[0, 0, 1]` bottom row are
 * accepted; anything else raises a `ValueError`.
 */
function normalize2x2(matrix: unknown): number[][] {
  if (matrix === undefined || matrix === null) return identity2x2();
  let rows: number[][];
  if (isNDArray(matrix)) {
    rows = matrixRows((matrix as NDArray).toArray());
  } else if (Array.isArray(matrix) && (matrix as unknown[]).every((value) => typeof value === "number")) {
    const flat = (matrix as number[]).map(Number);
    if (flat.length !== 4) throw valueError("Matrix must be 2x2 (or a flat list of four numbers)");
    return [
      [flat[0]!, flat[1]!],
      [flat[2]!, flat[3]!],
    ];
  } else {
    rows = matrixRows(matrix);
  }
  if (rows.length === 2 && rows.every((row) => row.length === 2)) {
    return rows.map((row) => row.map((value) => Number(value)));
  }
  if (
    rows.length === 3 &&
    rows.every((row) => row.length === 3) &&
    rows[2]![0] === 0 &&
    rows[2]![1] === 0 &&
    rows[2]![2] === 1
  ) {
    return [
      [Number(rows[0]![0]!), Number(rows[0]![1]!)],
      [Number(rows[1]![0]!), Number(rows[1]![1]!)],
    ];
  }
  throw valueError("Matrix must be 2x2 (or a 3x3 homogeneous matrix with bottom row [0, 0, 1])");
}

/** Inverse of a 2x2 matrix; raises `ValueError` when it is singular. */
function inverse2x2(matrix: number[][]): number[][] {
  const [[a = 0, b = 0], [c = 0, d = 0]] = matrix as [[number, number], [number, number]];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) throw valueError("Matrix is singular and cannot be inverted");
  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
}

export class InteractiveLinearTransformationScene extends InteractiveScene {
  include_background_plane: boolean;
  include_foreground_plane: boolean;
  show_coordinates: boolean;
  show_basis_vectors: boolean;
  basis_vector_stroke_width: number;
  i_hat_color: ManimColor;
  j_hat_color: ManimColor;
  leave_ghost_vectors: boolean;
  background_plane_kwargs: Record<string, unknown>;
  foreground_plane_kwargs: Record<string, unknown>;
  ghost_vectors: VGroup;
  background_plane!: NumberPlane;
  plane!: NumberPlane;
  background_mobjects: Mobject[] = [];
  /** Foreground mobjects transformed by the current matrix. */
  transformable_mobjects: Mobject[] = [];
  moving_vectors: Mobject[] = [];
  transformable_labels: any[] = [];
  basis_vectors!: VGroup;
  i_hat!: Arrow;
  j_hat!: Arrow;
  square!: Mobject;
  title!: Mobject;
  has_already_setup?: boolean;

  /** Current linear map, as a 2x2 list of rows. */
  matrix: number[][] = identity2x2();

  private _transformable_snapshots = new Map<Mobject, TransformableSnapshot>();
  private _plain_vectors: PlainVectorRecord[] = [];
  /** The matrix a restart returns to (the constructor's initial value). */
  private _initial_matrix: number[][] = identity2x2();

  constructor(
    include_background_plane: boolean | InteractiveLinearTransformationSceneOptions = true,
    include_foreground_plane = true,
    background_plane_kwargs: Record<string, unknown> | null = null,
    foreground_plane_kwargs: Record<string, unknown> | null = null,
    show_coordinates = false,
    show_basis_vectors = true,
    basis_vector_stroke_width = 6,
    i_hat_color: ParsableManimColor = X_COLOR,
    j_hat_color: ParsableManimColor = Y_COLOR,
    leave_ghost_vectors = false,
    options: InteractiveLinearTransformationSceneOptions = {},
  ) {
    // The scene runner constructs every scene as `new SceneClass({ renderer })`;
    // Python forwards `renderer=...` as a keyword. Route a leading options bag
    // to the base class (same convention as VectorScene/LinearTransformationScene).
    let opts = options;
    let backgroundFlag = include_background_plane as boolean;
    if (isOptionsBag(include_background_plane)) {
      opts = { ...include_background_plane, ...options };
      backgroundFlag = true;
    }
    super(opts);

    this.include_background_plane = backgroundFlag;
    this.include_foreground_plane = include_foreground_plane;
    this.show_coordinates = show_coordinates;
    this.show_basis_vectors = show_basis_vectors;
    this.basis_vector_stroke_width = basis_vector_stroke_width;
    this.i_hat_color = new ManimColor(i_hat_color);
    this.j_hat_color = new ManimColor(j_hat_color);
    this.leave_ghost_vectors = leave_ghost_vectors;
    if (opts.matrix !== undefined) this.matrix = normalize2x2(opts.matrix);
    this._initial_matrix = this.matrix.map((row) => [...row]);

    this.background_plane_kwargs = {
      color: GREY,
      axis_config: { color: GREY },
      background_line_style: { stroke_color: GREY, stroke_width: 1 },
    };
    const frameWidth = config.frame_width as number;
    this.foreground_plane_kwargs = {
      x_range: [-frameWidth, frameWidth, 1.0],
      y_range: [-frameWidth, frameWidth, 1.0],
      faded_line_ratio: 1,
    };
    this.ghost_vectors = new VGroup();
    LinearTransformationScene.update_default_configs(
      [this.foreground_plane_kwargs, this.background_plane_kwargs],
      [foreground_plane_kwargs, background_plane_kwargs],
    );
  }

  // --- live matrix state -----------------------------------------------------

  /** The current matrix (a deep copy). */
  get_matrix(): number[][] {
    return this.matrix.map((row) => [...row]);
  }

  /**
   * Replace the current matrix and redraw immediately (no animation). Safe to
   * call from a blueprint reaction or an `on_input` listener.
   *
   * ```python
   * def on_i(control, previous, next_value):
   *     point = control.get_center()
   *     matrix = self.get_matrix()
   *     matrix[0][0] = point[0]
   *     matrix[1][0] = point[1]
   *     self.set_matrix(matrix)
   *
   * handle = DraggablePoint([1, 0, 0], color=GREEN)
   * handle.on_input(on_i)
   * self.add(handle)
   * ```
   */
  set_matrix(matrix: unknown): this {
    this.matrix = normalize2x2(matrix);
    this.refresh_transform();
    return this;
  }

  /**
   * Transform by `matrix`. In `blueprint` this is immediate (compose into the
   * matrix state and redraw, no animation), so reactions and `on_input`
   * listeners stay live. In `construct` it animates exactly like
   * `LinearTransformationScene.apply_matrix` and composes the matrix state
   * once the animation finishes.
   */
  apply_matrix(matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    const applied = normalize2x2(matrix);
    if (this.phase_kind !== "construct") {
      this.matrix = matMul2x2(applied, this.matrix);
      this.refresh_transform();
      return this;
    }
    // NOTE: the base entry points (`apply_matrix`, `apply_transposed_matrix`,
    // ...) call back into `this.apply_function`, which would dispatch to this
    // class's override and play the animation twice while composing the matrix
    // state three times. The construct path therefore builds the pointwise
    // `func` itself and invokes the base `apply_function` directly; that body
    // only calls the (safe, animation-building) movement helpers below.
    return this._play_construct(applied, [], kwargs, () => {
      this.matrix = matMul2x2(applied, this.matrix);
    });
  }

  /**
   * Transform by the inverse of `matrix`. Immediate in `blueprint`, animated
   * in `construct` (see {@link apply_matrix}).
   */
  apply_inverse(matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    const applied = inverse2x2(normalize2x2(matrix));
    if (this.phase_kind !== "construct") {
      return this.set_matrix(matMul2x2(applied, this.matrix));
    }
    return this._play_construct(applied, [], kwargs, () => {
      this.matrix = matMul2x2(applied, this.matrix);
    });
  }

  /**
   * Transform by a transposed matrix. Immediate in `blueprint`, animated in
   * `construct` (see {@link apply_matrix}).
   */
  apply_transposed_matrix(
    transposed_matrix: unknown,
    kwargs: Record<string, unknown> = {},
  ): this | Promise<this> {
    // The base applies the transpose of its argument as the ordinary map.
    const applied = normalize2x2(transposeMatrix(matrixRows(transposed_matrix)));
    if (this.phase_kind !== "construct") {
      this.matrix = matMul2x2(applied, this.matrix);
      this.refresh_transform();
      return this;
    }
    return this._play_construct(applied, [], kwargs, () => {
      this.matrix = matMul2x2(applied, this.matrix);
    });
  }

  /**
   * Transform by an inverse transpose. Immediate in `blueprint`, animated in
   * `construct` (see {@link apply_matrix}).
   */
  apply_inverse_transpose(t_matrix: unknown, kwargs: Record<string, unknown> = {}): this | Promise<this> {
    // Base applies transpose(inverse(transpose(rows))); as an ordinary map
    // that is inverse(transpose(rows)).
    const applied = inverse2x2(normalize2x2(transposeMatrix(matrixRows(t_matrix))));
    if (this.phase_kind !== "construct") {
      this.matrix = matMul2x2(applied, this.matrix);
      this.refresh_transform();
      return this;
    }
    return this._play_construct(applied, [], kwargs, () => {
      this.matrix = matMul2x2(applied, this.matrix);
    });
  }

  /**
   * Apply a nonlinear `func`. Animated in `construct` like the base scene;
   * immediate in `blueprint`. Either way the geometry can no longer be
   * described by a 2x2 matrix afterwards, so the snapshots are rebased onto
   * the current picture and the matrix resets to identity (mirroring manim,
   * which tracks no matrix at all).
   */
  apply_nonlinear_transformation(
    func: MappingFunction,
    kwargs: Record<string, unknown> = {},
  ): this | Promise<this> {
    if (this.phase_kind !== "construct") {
      (this.plane as unknown as { prepare_for_nonlinear_transform: () => void }).prepare_for_nonlinear_transform();
      this._apply_func_to_current(func);
      this._rebase();
      return this;
    }
    (this.plane as unknown as { prepare_for_nonlinear_transform: () => void }).prepare_for_nonlinear_transform();
    return this._play_construct_func(func, [], kwargs, () => this._rebase());
  }

  /**
   * Apply an arbitrary pointwise `func`. Animated in `construct` like the base
   * scene; immediate in `blueprint`. See
   * {@link apply_nonlinear_transformation} for what happens to the matrix
   * state afterwards.
   */
  apply_function(
    func: MappingFunction,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
  ): this | Promise<this> {
    if (this.phase_kind !== "construct") {
      this._apply_func_to_current(func);
      this._rebase();
      return this;
    }
    return this._play_construct_func(func, added_anims, kwargs, () => this._rebase());
  }

  /**
   * Play `applied` as an animated transition in the `construct` phase,
   * through the base implementation (bypassing the virtual dispatch that
   * would otherwise re-enter this class's overrides), then sync the matrix
   * state. `applied` is the ordinary-form 2x2 map being animated.
   */
  private _play_construct(
    applied: number[][],
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
    onDone: () => void,
  ): this | Promise<this> {
    return this._play_construct_func(
      this.get_matrix_transformation(applied),
      added_anims,
      kwargs,
      onDone,
    );
  }

  /** `_play_construct` for an already-built pointwise `func`. */
  private _play_construct_func(
    func: MappingFunction,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
    onDone: () => void,
  ): this | Promise<this> {
    if (!("path_arc" in kwargs)) {
      // Mirror the base default: rotate along the average basis rotation.
      const right = pointRows(func(pointFrom(RIGHT, 3)), 3)[0] ?? [1, 0, 0];
      const up = pointRows(func(pointFrom(UP, 3)), 3)[0] ?? [0, 1, 0];
      const net_rotation = (
        (angle_of_vector(right) as number) + ((angle_of_vector(up) as number) - Math.PI / 2)
      ) / 2;
      kwargs["path_arc"] = net_rotation;
    }
    const played = LinearTransformationScene.prototype.apply_function.call(
      this as unknown as LinearTransformationScene,
      func,
      added_anims,
      kwargs,
    );
    return this._after_play(played, onDone);
  }

  /**
   * Run `update` after `played` finishes. `play` blocks in Python but returns
   * a promise under the real-time renderer; with no async scheduler it runs
   * inline and this stays synchronous.
   */
  private _after_play(played: unknown, update: () => void): this | Promise<this> {
    if (is_thenable(played)) return (played as Promise<unknown>).then(() => {
      update();
      return this;
    });
    update();
    return this;
  }

  /**
   * Re-apply the current matrix to the foreground plane, basis vectors and
   * every tracked vector. Called automatically by `set_matrix`; call it
   * yourself after mutating `self.matrix` in place.
   */
  refresh_transform(): this {
    this._refresh_transformables();
    this._place_basis_vectors();
    this._refresh_plain_vectors();
    return this;
  }

  /** A source-space point mapped through the current matrix (z is preserved). */
  _matrix_point(point: readonly number[]): [number, number, number] {
    const m = this.matrix;
    const [x = 0, y = 0, z = 0] = point;
    return [
      (m[0]?.[0] ?? 0) * x + (m[0]?.[1] ?? 0) * y,
      (m[1]?.[0] ?? 0) * x + (m[1]?.[1] ?? 0) * y,
      z,
    ];
  }

  // --- setup ----------------------------------------------------------------

  override setup(): void {
    if (this.has_already_setup) return;
    // `LinearTransformationScene.setup` builds the planes, arrays and basis
    // vectors; it is safe on this instance because it only touches the shared
    // Scene surface and the fields initialised in the constructor.
    LinearTransformationScene.prototype.setup.call(this as unknown as LinearTransformationScene);
    // NOTE: do not clear `_transformable_snapshots` here: the setup above
    // added the foreground plane through our `add_transformable_mobject`
    // override, which already snapshotted its source-space points. Clearing
    // would drop that snapshot and the grid would never transform again
    // (only vectors, which are re-pointed directly, would move).
    // Re-apply a non-identity initial matrix now.
    this.refresh_transform();
  }

  override presentation_reset(): void {
    super.presentation_reset();
    // The presentation may call setup() again after a rewind; the base guard
    // would otherwise skip rebuilding the planes that were just cleared.
    this.has_already_setup = false;
    this._transformable_snapshots.clear();
    this._plain_vectors = [];
    this.matrix = this._initial_matrix.map((row) => [...row]);
  }

  // --- scene registration (mirrors LinearTransformationScene) ----------------

  add_background_mobject(...mobjects: Mobject[]): void {
    this._add_special(this.background_mobjects, mobjects);
  }

  // @ts-expect-error -- mirrors Python `# type: ignore[override]`: variadic void intentionally narrows Scene.add_foreground_mobject
  add_foreground_mobject(...mobjects: Mobject[]): void {
    this._add_special(this.foreground_mobjects, mobjects);
  }

  /** Add a mobject that the current matrix should transform live. */
  add_transformable_mobject(...mobjects: Mobject[]): void {
    for (const mobject of mobjects) {
      if (!this.transformable_mobjects.includes(mobject)) {
        this.transformable_mobjects.push(mobject);
        this.add(mobject);
      }
      this._snapshot_transformable(mobject);
      this._apply_snapshot(mobject);
    }
  }

  add_moving_mobject(
    mobject: Mobject & { target?: Mobject | null },
    target_mobject: Mobject | null = null,
  ): void {
    mobject.target = target_mobject;
    this._add_special(this.moving_mobjects, [mobject]);
    this._snapshot_transformable(mobject);
    this._apply_snapshot(mobject);
  }

  get_ghost_vectors(): VGroup {
    return this.ghost_vectors;
  }

  private _add_special(list: Mobject[], mobjects: Mobject[]): void {
    for (const mobject of mobjects) {
      if (!list.includes(mobject)) {
        list.push(mobject);
        this.add(mobject);
      }
    }
  }

  // --- borrowed helpers ------------------------------------------------------

  get_vector(numerical_vector: number[] | NDArray, kwargs: Record<string, unknown> = {}): Arrow {
    return VectorScene.prototype.get_vector.call(
      this as unknown as VectorScene,
      numerical_vector,
      kwargs,
    );
  }

  get_basis_vectors(
    i_hat_color: ParsableManimColor = this.i_hat_color,
    j_hat_color: ParsableManimColor = this.j_hat_color,
  ): VGroup {
    return VectorScene.prototype.get_basis_vectors.call(
      this as unknown as VectorScene,
      i_hat_color,
      j_hat_color,
    );
  }

  get_unit_square(
    color: ParsableManimColor | Iterable<ParsableManimColor> = new ManimColor("PURE_YELLOW"),
    opacity = 0.3,
    stroke_width = 3,
  ): Rectangle {
    return LinearTransformationScene.prototype.get_unit_square.call(
      this as unknown as LinearTransformationScene,
      color,
      opacity,
      stroke_width,
    );
  }

  add_unit_square(
    animate = false,
    kwargs: Record<string, unknown> = {},
  ): this | Promise<this> {
    const result = LinearTransformationScene.prototype.add_unit_square.call(
      this as unknown as LinearTransformationScene,
      animate,
      kwargs,
    );
    if (is_thenable(result)) return (result as Promise<unknown>).then(() => this);
    return this;
  }

  add_title(title: Mobject | string, scale_factor = 1.5, animate = false): this | Promise<this> {
    const result = LinearTransformationScene.prototype.add_title.call(
      this as unknown as LinearTransformationScene,
      title,
      scale_factor,
      animate,
    );
    if (is_thenable(result)) return (result as Promise<unknown>).then(() => this);
    return this;
  }

  /**
   * Add a vector that follows the current matrix live. `v` is interpreted in
   * source space: the arrow is drawn from the origin to `matrix @ v`.
   */
  add_vector(
    vector: Arrow | number[] | readonly number[] | NDArray,
    color: ParsableManimColor | Iterable<ParsableManimColor> | Record<string, unknown> = new ManimColor("PURE_YELLOW"),
    animate: boolean | Record<string, unknown> = true,
    kwargs: Record<string, unknown> = {},
  ): Arrow | Promise<Arrow> {
    const result = LinearTransformationScene.prototype.add_vector.call(
      this as unknown as LinearTransformationScene,
      vector as never,
      color as never,
      animate as never,
      kwargs,
    );
    const track = (arrow: Arrow): Arrow => {
      this._track_plain_vector(arrow);
      this._refresh_plain_vectors();
      return arrow;
    };
    if (is_thenable(result)) return (result as Promise<Arrow>).then(track);
    return track(result as Arrow);
  }

  private _track_plain_vector(arrow: Arrow): void {
    if (this._plain_vectors.some((record) => record.arrow === arrow)) return;
    const start = pointRows(arrow.get_start(), 3)[0] ?? [0, 0, 0];
    const end = pointRows(arrow.get_end(), 3)[0] ?? [0, 0, 0];
    this._plain_vectors.push({
      arrow,
      start: [start[0] ?? 0, start[1] ?? 0, start[2] ?? 0],
      end: [end[0] ?? 0, end[1] ?? 0, end[2] ?? 0],
      opacity: Math.max(arrow.get_stroke_opacity(), arrow.get_fill_opacity()),
    });
  }

  // --- delegated movement/transformation helpers ------------------------------
  // The animated `construct` path runs the base `apply_*` implementations on
  // this instance; they call back into these helpers, so they are delegated
  // to the shared `Scene` surface both classes provide.

  get_matrix_transformation(matrix: unknown): (point: number[] | NDArray) => NDArray {
    return LinearTransformationScene.prototype.get_matrix_transformation.call(
      this as unknown as LinearTransformationScene,
      matrix,
    );
  }

  get_transposed_matrix_transformation(
    transposed_matrix: unknown,
  ): (point: number[] | NDArray) => NDArray {
    return LinearTransformationScene.prototype.get_transposed_matrix_transformation.call(
      this as unknown as LinearTransformationScene,
      transposed_matrix,
    );
  }

  get_piece_movement(pieces: Iterable<Mobject>): Transform {
    return LinearTransformationScene.prototype.get_piece_movement.call(
      this as unknown as LinearTransformationScene,
      pieces,
    );
  }

  get_moving_mobject_movement(func: MappingFunction): Transform {
    return LinearTransformationScene.prototype.get_moving_mobject_movement.call(
      this as unknown as LinearTransformationScene,
      func,
    );
  }

  get_vector_movement(func: MappingFunction): Transform {
    return LinearTransformationScene.prototype.get_vector_movement.call(
      this as unknown as LinearTransformationScene,
      func,
    );
  }

  get_transformable_label_movement(): Transform {
    return LinearTransformationScene.prototype.get_transformable_label_movement.call(
      this as unknown as LinearTransformationScene,
    );
  }

  get_vector_label(
    vector: Vector,
    label: string | Mobject,
    at_tip = false,
    direction = "left",
    rotate = false,
    color: ParsableManimColor | null = null,
    label_scale_factor?: number,
  ): Mobject {
    return VectorScene.prototype.get_vector_label.call(
      this as unknown as VectorScene,
      vector,
      label,
      at_tip,
      direction,
      rotate,
      color,
      label_scale_factor,
    );
  }

  // --- transform bookkeeping -------------------------------------------------

  private _place_basis_vectors(): void {
    if (!this.show_basis_vectors || this.i_hat === undefined) return;
    const iEnd = [this.matrix[0]?.[0] ?? 1, this.matrix[1]?.[0] ?? 0, 0];
    const jEnd = [this.matrix[0]?.[1] ?? 0, this.matrix[1]?.[1] ?? 1, 0];
    this._place_arrow(this.i_hat, ORIGIN, iEnd, 1);
    this._place_arrow(this.j_hat, ORIGIN, jEnd, 1);
  }

  private _snapshot_transformable(mobject: Mobject): void {
    if (this._transformable_snapshots.has(mobject)) return;
    const members: Mobject[] = [];
    const points: number[][][] = [];
    for (const member of mobject.get_family()) {
      const rows = pointRows(member.points, member.dim);
      if (rows.length === 0) continue;
      members.push(member);
      points.push(rows.map((row) => [...row]));
    }
    this._transformable_snapshots.set(mobject, { members, points });
  }

  private _apply_snapshot(mobject: Mobject): void {
    const snapshot = this._transformable_snapshots.get(mobject);
    if (snapshot === undefined) return;
    const m = this.matrix;
    const a = m[0]?.[0] ?? 1;
    const b = m[0]?.[1] ?? 0;
    const c = m[1]?.[0] ?? 1;
    const d = m[1]?.[1] ?? 1;
    for (let index = 0; index < snapshot.members.length; index += 1) {
      const member = snapshot.members[index]!;
      const base = snapshot.points[index]!;
      const transformed = new Array<number[]>(base.length);
      for (let row = 0; row < base.length; row += 1) {
        const [x = 0, y = 0, z = 0] = base[row]!;
        transformed[row] = [a * x + b * y, c * x + d * y, z];
      }
      member.points = array(transformed) as NDArray;
    }
  }

  private _refresh_transformables(): void {
    for (const mobject of this._transformable_snapshots.keys()) this._apply_snapshot(mobject);
  }

  /**
   * Re-point an arrow, hiding it when the target segment has zero length
   * (singular matrices collapse a basis vector to the origin). `Arrow` warns
   * and refuses to re-span a zero-length line, so hiding is the correct live
   * representation of a collapsed vector.
   */
  private _place_arrow(
    arrow: Arrow,
    start: readonly number[],
    end: readonly number[],
    opacity: number,
  ): void {
    const length = Math.hypot(
      (end[0] ?? 0) - (start[0] ?? 0),
      (end[1] ?? 0) - (start[1] ?? 0),
      (end[2] ?? 0) - (start[2] ?? 0),
    );
    if (length < 1e-9) {
      arrow.set_opacity(0);
      return;
    }
    arrow.set_opacity(opacity);
    arrow.put_start_and_end_on(start, end);
  }

  private _refresh_plain_vectors(): void {
    for (const record of this._plain_vectors) {
      this._place_arrow(
        record.arrow,
        this._matrix_point(record.start),
        this._matrix_point(record.end),
        record.opacity,
      );
    }
  }

  /**
   * Apply `func` to the *current* geometry of every tracked object (used for
   * arbitrary pointwise maps that cannot be folded into the 2x2 matrix).
   */
  private _apply_func_to_current(func: MappingFunction): void {
    for (const mobject of this._transformable_snapshots.keys()) {
      (mobject as unknown as { apply_function: (f: MappingFunction) => void }).apply_function(func);
    }
    for (const v of this.moving_vectors as Array<Arrow>) {
      const rawEnd = pointRows(v.get_end(), 3)[0] ?? [0, 0, 0];
      const rawStart = pointRows(v.get_start(), 3)[0] ?? [0, 0, 0];
      const newEnd = pointRows(func(rawEnd), 3)[0] ?? [0, 0, 0];
      const newStart = pointRows(func(rawStart), 3)[0] ?? [0, 0, 0];
      v.put_start_and_end_on(newStart, newEnd);
    }
    for (const m of this.moving_mobjects as Array<Mobject>) {
      const center = pointRows(m.get_center(), 3)[0] ?? [0, 0, 0];
      m.move_to(pointRows(func(center), 3)[0] ?? [0, 0, 0]);
    }
    for (const label of this.transformable_labels as Array<any>) {
      try {
        const kw = (label.kwargs ?? {}) as Record<string, unknown>;
        const target = this.get_vector_label(
          label.vector as Vector,
          (label.target_text ?? "") as string | Mobject,
          (kw["at_tip"] as boolean | undefined) ?? false,
          (kw["direction"] as string | undefined) ?? "left",
          (kw["rotate"] as boolean | undefined) ?? false,
          (kw["color"] as ParsableManimColor | null | undefined) ?? null,
          (kw["label_scale_factor"] as number | undefined) ?? undefined,
        );
        (label as unknown as { become: (m: unknown) => void }).become(target);
      } catch {
        /* a label that cannot be rebuilt keeps its old position */
      }
    }
    for (const record of this._plain_vectors) {
      const rawEnd = pointRows(record.arrow.get_end(), 3)[0] ?? [0, 0, 0];
      const rawStart = pointRows(record.arrow.get_start(), 3)[0] ?? [0, 0, 0];
      const newEnd = pointRows(func(rawEnd), 3)[0] ?? [0, 0, 0];
      const newStart = pointRows(func(rawStart), 3)[0] ?? [0, 0, 0];
      this._place_arrow(record.arrow, newStart, newEnd, record.opacity);
      record.start = [newStart[0] ?? 0, newStart[1] ?? 0, newStart[2] ?? 0];
      record.end = [newEnd[0] ?? 0, newEnd[1] ?? 0, newEnd[2] ?? 0];
    }
  }

  /**
   * Rebase the snapshots onto the current picture and reset the matrix to
   * identity. Used after an arbitrary (non-linear) map, which cannot be
   * represented as a 2x2 matrix.
   */
  private _rebase(): void {
    this._transformable_snapshots.clear();
    for (const mobject of this.transformable_mobjects) this._snapshot_transformable(mobject);
    for (const mobject of this.moving_mobjects) this._snapshot_transformable(mobject);
    this.matrix = identity2x2();
  }
}

/**
 * Product of two 2x2 row-list matrices (`left @ right`), for composing an
 * incremental map onto the absolute matrix state.
 */
function matMul2x2(left: number[][], right: number[][]): number[][] {
  const a = left[0]?.[0] ?? 1;
  const b = left[0]?.[1] ?? 0;
  const c = left[1]?.[0] ?? 0;
  const d = left[1]?.[1] ?? 1;
  const e = right[0]?.[0] ?? 1;
  const f = right[0]?.[1] ?? 0;
  const g = right[1]?.[0] ?? 0;
  const h = right[1]?.[1] ?? 1;
  return [
    [a * e + b * g, a * f + b * h],
    [c * e + d * g, c * f + d * h],
  ];
}

// Pin the Python-facing parameter names: esbuild may rename colliding
// parameters when bundling, and py2js maps keyword arguments by reading these
// names at runtime (see `parseParamNames` in py2js/src/runtime_core.ts).
(InteractiveLinearTransformationScene.prototype.set_matrix as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix"];
(InteractiveLinearTransformationScene.prototype.apply_matrix as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_inverse as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix", "kwargs"];
(InteractiveLinearTransformationScene.prototype.get_vector as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["numerical_vector", "kwargs"];
(InteractiveLinearTransformationScene.prototype.get_basis_vectors as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["i_hat_color", "j_hat_color"];
(InteractiveLinearTransformationScene.prototype.get_unit_square as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["color", "opacity", "stroke_width"];
(InteractiveLinearTransformationScene.prototype.add_unit_square as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["animate", "kwargs"];
(InteractiveLinearTransformationScene.prototype.add_title as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["title", "scale_factor", "animate"];
(InteractiveLinearTransformationScene.prototype.add_vector as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["vector", "color", "animate", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_transposed_matrix as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["transposed_matrix", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_inverse_transpose as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["t_matrix", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_nonlinear_transformation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["func", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_function as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["func", "added_anims", "kwargs"];
(InteractiveLinearTransformationScene.prototype.get_matrix_transformation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix"];
(InteractiveLinearTransformationScene.prototype.get_transposed_matrix_transformation as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["transposed_matrix"];
(InteractiveLinearTransformationScene.prototype.get_vector_label as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["vector", "label", "at_tip", "direction", "rotate", "color", "label_scale_factor"];
(InteractiveLinearTransformationScene.prototype.get_piece_movement as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["pieces"];
(InteractiveLinearTransformationScene.prototype.get_moving_mobject_movement as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["func"];
(InteractiveLinearTransformationScene.prototype.get_vector_movement as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["func"];
