/**
 * `InteractiveLinearTransformationScene`: the live counterpart of manim's
 * `LinearTransformationScene`.
 *
 * The geometry is the one readers know from the classic scene (a static grey
 * background grid, a transformable foreground grid, the `i_hat` / `j_hat` basis
 * vectors), but the current linear map is a piece of state instead of a
 * one-shot `self.play(...)`:
 *
 *  - `set_matrix(A)` / `apply_matrix(A)` install `A` **immediately** and redraw
 *    the foreground plane, the basis vectors and every tracked vector, so it
 *    can be called from a `blueprint` reaction or an `on_input` listener;
 *  - the tips of `i_hat` / `j_hat` are `DraggablePoint` handles: dragging one
 *    sets the corresponding column of the matrix and deforms the grid live;
 *  - `add_matrix_control(...)` places an editable `InputDecimalMatrix` whose
 *    cells drive the same state;
 *  - `add_draggable_vector(v)` and `add_draggable_point(p)` add reader-movable
 *    objects in the *source* space, with their images under the current matrix
 *    drawn beside them.
 *
 * Everything is a normal mobject and can be styled/positioned like any other
 * wanim object. The class intentionally borrows (`prototype.call`) the setup and
 * helpers of `LinearTransformationScene`/`VectorScene` rather than inheriting
 * from them: `InteractiveScene` owns the storyboard/control machinery, and a
 * single-inheritance TS hierarchy cannot carry both bases.
 *
 * NOTE: transformations are immediate. Unlike `LinearTransformationScene`,
 * `apply_matrix` does not animate here; use `self.play(...)` with ordinary
 * animations in `construct` when a transition should be animated.
 */
import { array, type NDArray } from "numpy-ts";
import { BLUE, GOLD, GREY, TEAL, YELLOW, ManimColor, type ParsableManimColor } from "../utils/color/index.js";
import { ORIGIN, UL } from "../constants.js";
import { config } from "../_config/utils.js";
import { isNDArray, matrixRows, pointRows } from "../utils/arrays.js";
import { is_thenable } from "../utils/scheduler.js";
import { Arrow, type ArrowOptions, type VectorOptions } from "../mobject/geometry/line.js";
import { Dot } from "../mobject/geometry/arc.js";
import { Rectangle } from "../mobject/geometry/polygram.js";
import { Mobject } from "../mobject/mobject.js";
import { VGroup } from "../mobject/types/vectorized_mobject.js";
import type { NumberPlane } from "../mobject/graphing/coordinate_systems.js";
import {
  DraggablePoint,
  InputDecimalMatrix,
  isInteractiveControl,
  type InputDecimalMatrixOptions,
} from "../interactive/controls.js";
import { InteractiveScene, type InteractiveSceneOptions } from "./interactive_scene.js";
import { LinearTransformationScene, VectorScene, X_COLOR, Y_COLOR } from "./vector_space_scene.js";

export interface InteractiveLinearTransformationSceneOptions extends InteractiveSceneOptions {
  /** Whether the i/j basis tips get drag handles. Defaults to `true`. */
  draggable_basis?: boolean;
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

/** An `add_draggable_vector` control plus the arrows it drives. */
interface DraggableVectorRecord {
  control: DraggablePoint;
  source: Arrow | null;
  image: Arrow | null;
  source_opacity: number;
  image_opacity: number;
}

/** An `add_draggable_point` control plus its image dot. */
interface DraggablePointRecord {
  control: DraggablePoint;
  image: Dot | null;
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
    "draggable_basis",
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

interface DraggableVectorOptions extends VectorOptions {
  /** Colour of the draggable handle / source arrow. */
  color?: ParsableManimColor;
  /** Colour of the image arrow (defaults to gold). */
  image_color?: ParsableManimColor;
  /** Draw the source-space arrow (default `True`). */
  show_source?: boolean;
  /** Draw the transformed image arrow (default `True`). */
  show_image?: boolean;
  /** Handle radius (world units). */
  radius?: number;
  /** Handle pick radius (world units). */
  hit_radius?: number;
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
  /** Whether the i/j tips expose drag handles (default `True`). */
  draggable_basis = true;
  /** The matrix editor added by {@link add_matrix_control}, if any. */
  matrix_control: InputDecimalMatrix | null = null;
  /** Handles added for the basis tips (empty when they are disabled). */
  basis_handles: DraggablePoint[] = [];
  /** Controls created by {@link add_draggable_point}. */
  draggable_points: DraggablePointRecord[] = [];
  /** Controls created by {@link add_draggable_vector}. */
  draggable_vectors: DraggableVectorRecord[] = [];

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
    this.draggable_basis = opts.draggable_basis !== false;
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
   */
  set_matrix(matrix: unknown): this {
    this.matrix = normalize2x2(matrix);
    this.refresh_transform();
    return this;
  }

  /**
   * Alias of {@link set_matrix} for readers coming from
   * `LinearTransformationScene`. Unlike the animated base method this installs
   * the matrix immediately; see the class comment.
   */
  apply_matrix(matrix: unknown, _kwargs: Record<string, unknown> = {}): this {
    return this.set_matrix(matrix);
  }

  /** Install the inverse of `matrix` immediately. */
  apply_inverse(matrix: unknown, _kwargs: Record<string, unknown> = {}): this {
    return this.set_matrix(inverse2x2(normalize2x2(matrix)));
  }

  /**
   * Re-apply the current matrix to the foreground plane, basis vectors and
   * every tracked vector/image. Called automatically by the interaction paths;
   * call it yourself after mutating `self.matrix` in place.
   */
  refresh_transform(): this {
    this._refresh_transformables();
    this._place_basis_vectors();
    this._place_basis_handles();
    this._refresh_plain_vectors();
    this._refresh_draggable_records();
    this._sync_matrix_control();
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
    this._transformable_snapshots.clear();
    // The setup above added the foreground plane through our override, which
    // snapshotted it; re-apply a non-identity initial matrix now.
    this.refresh_transform();
    this._create_basis_handles();
    this._place_basis_handles();
  }

  override presentation_reset(): void {
    super.presentation_reset();
    // The presentation may call setup() again after a rewind; the base guard
    // would otherwise skip rebuilding the planes that were just cleared.
    this.has_already_setup = false;
    this._transformable_snapshots.clear();
    this._plain_vectors = [];
    this.draggable_points = [];
    this.draggable_vectors = [];
    this.basis_handles = [];
    this.matrix_control = null;
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

  // --- interactive helpers ---------------------------------------------------

  /**
   * Add an editable matrix at a screen corner and wire it to the live
   * transformation. The cells keep in sync when the reader drags a basis tip.
   */
  add_matrix_control(
    matrix: unknown = null,
    position: readonly number[] = UL,
    buff = 0.6,
    num_decimal_places = 1,
    options: InputDecimalMatrixOptions = {},
  ): InputDecimalMatrix {
    const rows = normalize2x2(matrix === null ? this.matrix : matrix);
    this.matrix = rows;
    // One editor is enough: replace any previous one so stale cells cannot
    // keep driving the matrix.
    if (this.matrix_control !== null) this.remove(this.matrix_control);
    const control = new InputDecimalMatrix(rows, {
      element_to_mobject_config: { num_decimal_places },
      ...options,
    });
    control.to_corner(position, buff);
    for (const row of control.mob_matrix) {
      for (const cell of row) {
        if (!isInteractiveControl(cell)) continue;
        cell.on_input?.(() => this._on_matrix_control_input());
      }
    }
    this.matrix_control = control;
    this.add(control);
    this.refresh_transform();
    return control;
  }

  /** Read the matrix editor's cells into the current matrix and redraw. */
  private _on_matrix_control_input(): void {
    const control = this.matrix_control;
    if (control === null) return;
    const rows = control.mob_matrix.map((row) =>
      row.map((cell) => {
        const value = (cell as unknown as { get_value?: () => unknown }).get_value?.();
        return typeof value === "number" ? value : Number(value ?? 0);
      }),
    );
    this.matrix = normalize2x2(rows);
    // Do not write the cells back while the reader is typing in one of them.
    this._refresh_transformables();
    this._place_basis_vectors();
    this._place_basis_handles();
    this._refresh_plain_vectors();
    this._refresh_draggable_records();
  }

  /**
   * Add a draggable point (in source space) together with a dot showing its
   * image `matrix @ p` in the transformed plane. Returns the draggable control.
   */
  add_draggable_point(
    point: readonly number[] | NDArray = [0, 0, 0],
    color: ParsableManimColor = YELLOW,
    image_color: ParsableManimColor = TEAL,
    show_image = true,
    radius = 0.1,
    hit_radius = 0.32,
  ): DraggablePoint {
    const row = pointRows(point, 3)[0] ?? [0, 0, 0];
    const control = new DraggablePoint(row, { color, radius, hit_radius });
    const image = show_image
      ? new Dot({ point: this._matrix_point(row), radius, color: image_color, fill_opacity: 0.8 })
      : null;
    const record: DraggablePointRecord = { control, image };
    this.draggable_points.push(record);
    control.on_input(() => this._refresh_draggable_records());
    if (image !== null) this.add(image);
    this.add(control);
    this._refresh_draggable_records();
    return control;
  }

  /**
   * Add a reader-movable vector in source space: the handle sits at `vector`,
   * the source arrow is drawn faintly and the transformed image arrow follows
   * the current matrix. Returns the draggable control (use `get_center()`).
   */
  add_draggable_vector(
    vector: readonly number[] | NDArray | Arrow,
    options: DraggableVectorOptions = {},
  ): DraggablePoint {
    const {
      color = BLUE,
      image_color = GOLD,
      show_source = true,
      show_image = true,
      radius = 0.12,
      hit_radius = 0.34,
      ...arrowOptions
    } = options;
    const sourcePoint = vector instanceof Arrow
      ? (pointRows(vector.get_end(), 3)[0] ?? [0, 0, 0])
      : (pointRows(vector, 3)[0] ?? [0, 0, 0]);
    const control = new DraggablePoint(sourcePoint, { color, radius, hit_radius });
    const source = show_source
      ? new Arrow({
          start: [...ORIGIN],
          end: sourcePoint,
          buff: 0,
          color,
          stroke_width: 4,
          // The source arrow sits under the image; keep it faint by default so
          // the transformed arrow is the one that stands out.
          stroke_opacity: 0.55,
          ...arrowOptions,
        } as ArrowOptions & { start: number[]; end: number[] })
      : null;
    const image = show_image
      ? new Arrow({
          start: [...ORIGIN],
          end: this._matrix_point(sourcePoint),
          buff: 0,
          color: image_color,
          stroke_width: 4,
          ...arrowOptions,
        } as ArrowOptions & { start: number[]; end: number[] })
      : null;
    const record: DraggableVectorRecord = {
      control,
      source,
      image,
      source_opacity: source === null ? 1 : Math.max(source.get_stroke_opacity(), source.get_fill_opacity()),
      image_opacity: image === null ? 1 : Math.max(image.get_stroke_opacity(), image.get_fill_opacity()),
    };
    this.draggable_vectors.push(record);
    control.on_input(() => this._refresh_draggable_records());
    if (source !== null) this.add(source);
    if (image !== null) this.add(image);
    this.add(control);
    this._refresh_draggable_records();
    return control;
  }

  // --- basis drag handles ----------------------------------------------------

  /** Create the i/j drag handles (idempotent). */
  enable_basis_drag(): this {
    this.draggable_basis = true;
    this._create_basis_handles();
    this._place_basis_handles();
    return this;
  }

  /** Remove the i/j drag handles; matrix changes keep working from code/controls. */
  disable_basis_drag(): this {
    this.draggable_basis = false;
    for (const handle of this.basis_handles) this.remove(handle);
    this.basis_handles = [];
    return this;
  }

  private _create_basis_handles(): void {
    if (!this.draggable_basis || !this.show_basis_vectors || this.basis_handles.length > 0) return;
    if (this.i_hat === undefined || this.j_hat === undefined) return;
    const make = (column: 0 | 1, color: ParsableManimColor): DraggablePoint => {
      const end = this._matrix_point(column === 0 ? [1, 0, 0] : [0, 1, 0]);
      const handle = new DraggablePoint(end, { color, radius: 0.12, hit_radius: 0.34 });
      handle.on_input(() => this._on_basis_handle_input(column, handle));
      return handle;
    };
    this.basis_handles = [make(0, this.i_hat_color), make(1, this.j_hat_color)];
    this.add(this.basis_handles[0]!, this.basis_handles[1]!);
  }

  private _on_basis_handle_input(column: 0 | 1, handle: DraggablePoint): void {
    const point = pointRows(handle.get_center(), 3)[0] ?? [0, 0, 0];
    this.matrix[0]![column] = point[0] ?? 0;
    this.matrix[1]![column] = point[1] ?? 0;
    this.refresh_transform();
  }

  private _place_basis_handles(): void {
    if (this.basis_handles.length < 2) return;
    this.basis_handles[0]!.set_value([this.matrix[0]?.[0] ?? 1, this.matrix[1]?.[0] ?? 0, 0]);
    this.basis_handles[1]!.set_value([this.matrix[0]?.[1] ?? 0, this.matrix[1]?.[1] ?? 1, 0]);
  }

  private _place_basis_vectors(): void {
    if (!this.show_basis_vectors || this.i_hat === undefined) return;
    const iEnd = [this.matrix[0]?.[0] ?? 1, this.matrix[1]?.[0] ?? 0, 0];
    const jEnd = [this.matrix[0]?.[1] ?? 0, this.matrix[1]?.[1] ?? 1, 0];
    this._place_arrow(this.i_hat, ORIGIN, iEnd, 1);
    this._place_arrow(this.j_hat, ORIGIN, jEnd, 1);
  }

  // --- transform bookkeeping -------------------------------------------------

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

  private _refresh_draggable_records(): void {
    for (const record of this.draggable_points) {
      if (record.image === null) continue;
      record.image.move_to(this._matrix_point(pointRows(record.control.get_center(), 3)[0] ?? [0, 0, 0]));
    }
    for (const record of this.draggable_vectors) {
      const sourcePoint = pointRows(record.control.get_center(), 3)[0] ?? [0, 0, 0];
      const imagePoint = this._matrix_point(sourcePoint);
      if (record.source !== null) {
        this._place_arrow(record.source, ORIGIN, sourcePoint, record.source_opacity);
      }
      if (record.image !== null) {
        this._place_arrow(record.image, ORIGIN, imagePoint, record.image_opacity);
      }
    }
  }

  /** Write the current matrix back into the editor cells (when they differ). */
  private _sync_matrix_control(): void {
    const control = this.matrix_control;
    if (control === null) return;
    let changed = false;
    for (let row = 0; row < control.mob_matrix.length; row += 1) {
      const cells = control.mob_matrix[row];
      if (cells === undefined) continue;
      for (let column = 0; column < cells.length; column += 1) {
        const cell = cells[column]!;
        const target = this.matrix[row]?.[column] ?? 0;
        const reader = cell as unknown as { get_value?: () => unknown; set_value?: (value: number) => unknown };
        const value = reader.get_value?.();
        if (typeof value === "number" && Math.abs(value - target) < 1e-9) continue;
        reader.set_value?.(target);
        changed = true;
      }
    }
    if (changed) control.relayout();
  }
}

// Pin the Python-facing parameter names: esbuild may rename colliding
// parameters when bundling, and py2js maps keyword arguments by reading these
// names at runtime (see `parseParamNames` in py2js/src/runtime_core.ts).
(InteractiveLinearTransformationScene.prototype.add_matrix_control as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix", "position", "buff", "num_decimal_places", "options"];
(InteractiveLinearTransformationScene.prototype.add_draggable_point as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["point", "color", "image_color", "show_image", "radius", "hit_radius"];
(InteractiveLinearTransformationScene.prototype.add_draggable_vector as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["vector", "options"];
(InteractiveLinearTransformationScene.prototype.set_matrix as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix"];
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
(InteractiveLinearTransformationScene.prototype.apply_matrix as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix", "kwargs"];
(InteractiveLinearTransformationScene.prototype.apply_inverse as unknown as {
  __argnames__?: string[];
}).__argnames__ = ["matrix", "kwargs"];
