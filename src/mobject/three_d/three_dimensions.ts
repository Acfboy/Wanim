import { array, type NDArray } from "numpy-ts";
import { isNDArray, linspace, norm, pointRows } from "../../utils/arrays.js";
import { IN, LEFT, ORIGIN, OUT, PI, RIGHT, TAU, UP, DOWN, X_AXIS, Y_AXIS, Z_AXIS } from "../../constants.js";
import { DEFAULT_DOT_RADIUS, LineJointType } from "../../constants.js";
import { config } from "../../_config/utils.js";
import { RendererType } from "../../constants.js";
import { normalize, z_to_vector } from "../../utils/space_ops.js";
import { BLUE, BLUE_D, BLUE_E, LIGHT_GREY, ManimColor, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { Mobject } from "../mobject.js";
import { VGroup, VectorizedPoint, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Circle } from "../geometry/arc.js";
import { Square } from "../geometry/polygram.js";

export class ThreeDVMobject extends VMobject {
  u_index!: number;
  v_index!: number;
  u1!: number;
  u2!: number;
  v1!: number;
  v2!: number;

  constructor(shade_in_3d: boolean | VMobjectOptions = true, options: VMobjectOptions = {}) {
    // Python: __init__(self, shade_in_3d=True, **kwargs). Accept both the
    // positional boolean and the TS options-object form.
    let resolved: VMobjectOptions;
    let shade: boolean;
    if (typeof shade_in_3d === "object" && shade_in_3d !== null) {
      resolved = shade_in_3d as VMobjectOptions;
      shade = (resolved.shade_in_3d as boolean | undefined) ?? true;
    } else {
      shade = shade_in_3d as boolean;
      resolved = options;
    }
    super({ ...resolved, shade_in_3d: shade });
  }
}

export type SurfacePoint = NDArray;

function point3(values: [number, number, number]): SurfacePoint {
  return array(values) as SurfacePoint;
}

export type SurfaceFunc = (u: number, v: number) => SurfacePoint | number[];

export class Surface extends VGroup {
  u_range: [number, number];
  v_range: [number, number];
  resolution: number | [number, number];
  surface_piece_config: Record<string, unknown>;
  checkerboard_colors: ManimColor[] | false;
  should_make_jagged: boolean;
  list_of_faces: ThreeDVMobject[] = [];
  protected _func: SurfaceFunc;

  constructor(
    func: SurfaceFunc,
    u_range: [number, number] = [0, 1],
    v_range: [number, number] = [0, 1],
    resolution: number | [number, number] = 32,
    surface_piece_config: Record<string, unknown> = {},
    fill_color: ParsableManimColor = BLUE_D,
    fill_opacity = 1.0,
    checkerboard_colors: Iterable<ParsableManimColor> | false = [BLUE_D, BLUE_E],
    stroke_color: ParsableManimColor = LIGHT_GREY,
    stroke_width = 0.5,
    should_make_jagged = false,
    pre_function_handle_to_anchor_scale_factor = 0.00001,
    options: Record<string, unknown> = {},
  ) {
    super();
    void options;
    this.u_range = u_range;
    this.v_range = v_range;
    this.surface_piece_config = surface_piece_config;
    this.resolution = resolution;
    if (checkerboard_colors === false) {
      this.checkerboard_colors = false;
    } else {
      this.checkerboard_colors = [...checkerboard_colors].map((c) => new ManimColor(c));
    }
    this.should_make_jagged = should_make_jagged;
    this.pre_function_handle_to_anchor_scale_factor = pre_function_handle_to_anchor_scale_factor;
    this.list_of_faces = [];
    this._func = func;
    this._setup_in_uv_space(fill_color, fill_opacity, stroke_color, stroke_width);
    const captured = this._func;
    this.apply_function((p: number[]): number[] => {
      const out = captured(p[0] ?? 0, p[1] ?? 0);
      return pointRows(out, 3)[0] ?? [0, 0, 0];
    });
    if (this.should_make_jagged) {
      this.make_jagged();
    }
  }

  func(u: number, v: number): SurfacePoint | number[] {
    return this._func(u, v);
  }

  _setup_in_uv_space(
    fill_color: ParsableManimColor = BLUE_D,
    fill_opacity = 1.0,
    stroke_color: ParsableManimColor = LIGHT_GREY,
    stroke_width = 0.5,
  ): void {
    const [u_values, v_values] = this._get_u_values_and_v_values();
    const faces = new VGroup();
    this.list_of_faces = [];
    for (let i = 0; i < u_values.length - 1; i += 1) {
      for (let j = 0; j < v_values.length - 1; j += 1) {
        const u1 = u_values[i] ?? 0;
        const u2 = u_values[i + 1] ?? 0;
        const v1 = v_values[j] ?? 0;
        const v2 = v_values[j + 1] ?? 0;
        const face = new ThreeDVMobject();
        face.set_points_as_corners([
          [u1, v1, 0],
          [u2, v1, 0],
          [u2, v2, 0],
          [u1, v2, 0],
          [u1, v1, 0],
        ]);
        face.u_index = i;
        face.v_index = j;
        face.u1 = u1;
        face.u2 = u2;
        face.v1 = v1;
        face.v2 = v2;
        faces.add(face);
        this.list_of_faces.push(face);
      }
    }
    faces.set_fill(fill_color, fill_opacity);
    (faces as unknown as { set_stroke: (color?: unknown, width?: number, opacity?: number) => void }).set_stroke(
      stroke_color,
      stroke_width,
      undefined,
    );
    // NOTE: set_stroke(color, width) keeps the current stroke opacity.
    this.add(...faces.submobjects);
    if (this.checkerboard_colors) {
      this.set_fill_by_checkerboard(...this.checkerboard_colors);
    }
  }

  set_fill_by_checkerboard(
    ...args: Array<ParsableManimColor | { opacity?: number | null }>
  ): this {
    let opacity: number | null | undefined = null;
    let colors = [...args] as ParsableManimColor[];
    const last = args[args.length - 1];
    if (
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !(last instanceof ManimColor) &&
      !isNDArray(last) &&
      "opacity" in (last as Record<string, unknown>)
    ) {
      opacity = (last as { opacity?: number | null }).opacity ?? null;
      colors = (args.slice(0, -1) as ParsableManimColor[]);
    }
    const n_colors = colors.length;
    for (const face of this.list_of_faces) {
      const c_index = (face.u_index + face.v_index) % n_colors;
      face.set_fill(colors[c_index], opacity ?? undefined);
    }
    return this;
  }

  setFillByCheckerboard(
    ...args: Array<ParsableManimColor | { opacity?: number | null }>
  ): this {
    return this.set_fill_by_checkerboard(...args);
  }

  _get_u_values_and_v_values(): [number[], number[]] {
    let u_res: number;
    let v_res: number;
    if (typeof this.resolution === "number") {
      u_res = this.resolution;
      v_res = this.resolution;
    } else {
      [u_res, v_res] = this.resolution;
    }
    const u_values = linspace(this.u_range[0], this.u_range[1], u_res + 1);
    const v_values = linspace(this.v_range[0], this.v_range[1], v_res + 1);
    return [u_values, v_values];
  }
}

function spherePoint(radius: number, u: number, v: number): SurfacePoint {
  return array([
    radius * Math.cos(u) * Math.sin(v),
    radius * Math.sin(u) * Math.sin(v),
    -radius * Math.cos(v),
  ]) as SurfacePoint;
}

export class Sphere extends Surface {
  radius: number;

  constructor(
    center: readonly number[] | NDArray = [0, 0, 0],
    radius = 1,
    // NOTE: Python defaults the resolution from config.renderer
    // ((101, 51) for OpenGL, (24, 12) for Cairo); renderers are out of scope
    // for the interpreter target, so the Cairo default is used.
    resolution: number | [number, number] = [24, 12],
    u_range: [number, number] = [0, Math.PI * 2],
    v_range: [number, number] = [0, Math.PI],
  ) {
    // NOTE: the closure captures `radius` instead of `this.func` because
    // `this` cannot be referenced in a super() call.
    super((u: number, v: number): SurfacePoint => spherePoint(radius, u, v), u_range, v_range, resolution);
    this.radius = radius;
    this.shift(pointRows(center, 3)[0] ?? [0, 0, 0]);
  }

  override func(u: number, v: number): SurfacePoint {
    return spherePoint(this.radius, u, v);
  }
}

export class Dot3D extends Sphere {
  constructor(
    point: readonly number[] | NDArray = [...ORIGIN],
    radius = DEFAULT_DOT_RADIUS,
    color: ParsableManimColor = WHITE,
    resolution: number | [number, number] = [8, 8],
    options: Record<string, unknown> = {},
  ) {
    void options;
    super(point, radius, resolution);
    this.set_color(color);
  }
}

export interface CubeOptions {
  fill_opacity?: number;
  fill_color?: ParsableManimColor;
  stroke_width?: number;
  [key: string]: unknown;
}

export class Cube extends VGroup {
  side_length: number;

  constructor(
    side_length = 2,
    fill_opacity = 0.75,
    fill_color: ParsableManimColor = BLUE,
    stroke_width = 0,
    options: CubeOptions = {},
  ) {
    super();
    void options;
    this.side_length = side_length;
    this.set_fill(fill_color, fill_opacity);
    this.set_stroke(undefined, stroke_width);
    this.generate_points();
  }

  generate_points(): this {
    // NOTE: the base VMobject constructor invokes generate_points() before
    // this class assigns side_length (TS requires super() first, whereas
    // Python assigns before super().__init__()). Treat that early call as a
    // no-op; the constructor regenerates afterwards.
    if (!("side_length" in this) || this.side_length === undefined) return this;
    for (const vect of [IN, OUT, LEFT, RIGHT, UP, DOWN]) {
      const face = new Square(this.side_length, {
        shade_in_3d: true,
        // NOTE: VMobject.joint_type is modeled as a string in this port
        // (default "AUTO"); pass the enum member's name for BEVEL.
        joint_type: LineJointType[LineJointType.BEVEL] as string,
      });
      face.flip();
      face.shift(([...OUT] as number[]).map((value) => value * this.side_length / 2.0));
      face.apply_matrix(z_to_vector(vect));
      this.add(face);
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }
}

export interface PrismOptions extends CubeOptions {
  side_length?: number;
}

export class Prism extends Cube {
  dimensions: number[];

  constructor(
    dimensions: readonly number[] | NDArray = [3, 2, 1],
    options: PrismOptions = {},
  ) {
    const { side_length = 2, ...rest } = options;
    super(
      side_length,
      rest.fill_opacity ?? 0.75,
      rest.fill_color ?? BLUE,
      rest.stroke_width ?? 0,
      rest,
    );
    this.dimensions = [...(pointRows(dimensions, 3)[0] ?? [3, 2, 1])];
    // NOTE: Cube's constructor already built the (unscaled) faces through
    // generate_points() — which early-returned on the rescale step while
    // `dimensions` was still unassigned (TS requires super() first, whereas
    // Python assigns dimensions before super().__init__()). Rescale the
    // freshly built faces here instead of regenerating (which would add a
    // second set of faces).
    for (let dim = 0; dim < this.dimensions.length; dim += 1) {
      this.rescale_to_fit(this.dimensions[dim] ?? 0, dim, true);
    }
  }

  override generate_points(): this {
    super.generate_points();
    // Guard: the base Mobject constructor invokes generate_points() before
    // this class assigns dimensions (TS requires super() first, whereas
    // Python sets them before super().__init__()). Treat that early call as
    // rescale-deferred; the constructor rescales afterwards.
    if (!("dimensions" in this) || this.dimensions === undefined) return this;
    for (let dim = 0; dim < this.dimensions.length; dim += 1) {
      this.rescale_to_fit(this.dimensions[dim] ?? 0, dim, true);
    }
    return this;
  }
}

function conePoint(u: number, v: number, theta: number): SurfacePoint {
  const r = u;
  const phi = v;
  return point3([
    r * Math.sin(theta) * Math.cos(phi),
    r * Math.sin(theta) * Math.sin(phi),
    r * Math.cos(theta),
  ]);
}

export interface ConeOptions {
  resolution?: number | [number, number];
  fill_color?: ParsableManimColor;
  fill_opacity?: number;
  stroke_color?: ParsableManimColor;
  stroke_width?: number;
  should_make_jagged?: boolean;
  surface_piece_config?: Record<string, unknown>;
  [key: string]: unknown;
}

export class Cone extends Surface {
  direction: number[];
  theta: number;
  new_height: number;
  base_circle!: Circle;
  start_point!: VectorizedPoint;
  end_point!: VectorizedPoint;
  protected _current_theta = 0;
  protected _current_phi = 0;

  constructor(
    base_radius = 1,
    height = 1,
    direction: readonly number[] | NDArray = [...Z_AXIS],
    show_base = false,
    v_range: [number, number] = [0, TAU],
    u_min = 0,
    checkerboard_colors: Iterable<ParsableManimColor> | false = false,
    options: ConeOptions = {},
  ) {
    // NOTE: Python assigns direction/theta before super().__init__() so the
    // bound self.func sees them. TS requires super() first; the closure
    // below captures the locals instead (identical values).
    const theta = PI - Math.atan(base_radius / height);
    const u_range: [number, number] = [u_min, Math.sqrt(base_radius ** 2 + height ** 2)];
    super(
      (u: number, v: number) => conePoint(u, v, theta),
      u_range,
      v_range,
      options.resolution ?? 32,
      options.surface_piece_config ?? {},
      options.fill_color ?? BLUE_D,
      options.fill_opacity ?? 1.0,
      checkerboard_colors,
      options.stroke_color ?? LIGHT_GREY,
      options.stroke_width ?? 0.5,
      options.should_make_jagged ?? false,
    );
    this.direction = [...(pointRows(direction, 3)[0] ?? [0, 0, 1])];
    this.theta = theta;
    // used for rotations
    this.new_height = height;
    this._current_theta = 0;
    this._current_phi = 0;
    this.base_circle = new Circle(base_radius, {
      color: this.fill_color,
      fill_opacity: this.fill_opacity,
      stroke_width: 0,
    });
    this.base_circle.shift([0, 0, -height]);
    this._set_start_and_end_attributes(direction);
    if (show_base) {
      this.add(this.base_circle);
    }
    this._rotate_to_direction();
  }

  override func(u: number, v: number): SurfacePoint | number[] {
    return conePoint(u, v, this.theta);
  }

  get_start(): number[] {
    return this.start_point.get_center();
  }

  get_end(): number[] {
    return this.end_point.get_center();
  }

  _rotate_to_direction(): void {
    const [x = 0, y = 0, z = 0] = pointRows(this.direction, 3)[0] ?? [0, 0, 0];

    const r = Math.sqrt(x ** 2 + y ** 2 + z ** 2);
    const theta = r > 0 ? Math.acos(z / r) : 0;

    let phi: number;
    if (x === 0) {
      if (y === 0) {
        // along the z axis
        phi = 0;
      } else {
        phi = Math.atan(Infinity);
        if (y < 0) {
          phi += PI;
        }
      }
    } else {
      phi = Math.atan(y / x);
    }
    if (x < 0) {
      phi += PI;
    }

    // Undo old rotation (in reverse order)
    this.rotate(-this._current_phi, [...Z_AXIS], ORIGIN);
    this.rotate(-this._current_theta, [...Y_AXIS], [0, 0, 0]);

    // Do new rotation
    this.rotate(theta, [...Y_AXIS], [0, 0, 0]);
    this.rotate(phi, [...Z_AXIS], [0, 0, 0]);

    // Store values
    this._current_theta = theta;
    this._current_phi = phi;
  }

  set_direction(direction: readonly number[] | NDArray): this {
    this.direction = [...(pointRows(direction, 3)[0] ?? [0, 0, 1])];
    this._rotate_to_direction();
    return this;
  }

  _set_start_and_end_attributes(direction: readonly number[] | NDArray): void {
    const direction_row = pointRows(direction, 3)[0] ?? [0, 0, 1];
    const scale = norm(direction_row);
    const normalized_direction = direction_row.map((value) => value * scale);

    const start = this.base_circle.get_center();
    const end = start.map((value, index) => value + (normalized_direction[index] ?? 0) * this.new_height);
    this.start_point = new VectorizedPoint(start);
    this.end_point = new VectorizedPoint(end);
    this.add(this.start_point, this.end_point);
  }
}

function cylinderPoint(u: number, v: number, radius: number): SurfacePoint {
  const height = u;
  const phi = v;
  const r = radius;
  return point3([r * Math.cos(phi), r * Math.sin(phi), height]);
}

export interface CylinderOptions extends ConeOptions {}

export class Cylinder extends Surface {
  direction!: readonly number[] | NDArray;
  radius: number;
  protected _height: number;
  base_top!: Circle;
  base_bottom!: Circle;
  protected _current_theta = 0;
  protected _current_phi = 0;

  constructor(
    radius = 1,
    height = 2,
    direction: readonly number[] | NDArray = [...Z_AXIS],
    v_range: [number, number] = [0, TAU],
    show_ends = true,
    resolution: number | [number, number] = [24, 24],
    options: CylinderOptions = {},
  ) {
    // NOTE: same super()-first constraint as Cone (see above); the closure
    // captures the radius/height params.
    super(
      (u: number, v: number) => cylinderPoint(u, v, radius),
      [-height / 2, height / 2],
      v_range,
      resolution,
      options.surface_piece_config ?? {},
      options.fill_color ?? BLUE_D,
      options.fill_opacity ?? 1.0,
      false,
      options.stroke_color ?? LIGHT_GREY,
      options.stroke_width ?? 0.5,
      options.should_make_jagged ?? false,
    );
    this._height = height;
    this.radius = radius;
    if (show_ends) {
      this.add_bases();
    }
    this._current_phi = 0;
    this._current_theta = 0;
    this.set_direction(direction);
  }

  override func(u: number, v: number): SurfacePoint | number[] {
    return cylinderPoint(u, v, this.radius);
  }

  add_bases(): this {
    let color: ParsableManimColor;
    let opacity: number;
    if (config.renderer === RendererType.OPENGL) {
      color = this.color;
      opacity = this.fill_opacity;
    } else {
      color = this.fill_color;
      opacity = this.fill_opacity;
    }

    this.base_top = new Circle(this.radius, {
      color,
      fill_opacity: opacity,
      shade_in_3d: true,
      stroke_width: 0,
    });
    this.base_top.shift([0, 0, -(this.u_range[1] ?? 0)]);
    this.base_bottom = new Circle(this.radius, {
      color,
      fill_opacity: opacity,
      shade_in_3d: true,
      stroke_width: 0,
    });
    this.base_bottom.shift([0, 0, -(this.u_range[0] ?? 0)]);
    this.add(this.base_top, this.base_bottom);
    return this;
  }

  _rotate_to_direction(): void {
    const [x = 0, y = 0, z = 0] = pointRows(this.direction, 3)[0] ?? [0, 0, 0];

    const r = Math.sqrt(x ** 2 + y ** 2 + z ** 2);
    const theta = r > 0 ? Math.acos(z / r) : 0;

    let phi: number;
    if (x === 0) {
      if (y === 0) {
        // along the z axis
        phi = 0;
      } else {
        // along the x axis
        phi = Math.atan(Infinity);
        if (y < 0) {
          phi += PI;
        }
      }
    } else {
      phi = Math.atan(y / x);
    }
    if (x < 0) {
      phi += PI;
    }

    // undo old rotation (in reverse direction)
    this.rotate(-this._current_phi, [...Z_AXIS], [0, 0, 0]);
    this.rotate(-this._current_theta, [...Y_AXIS], [0, 0, 0]);

    // do new rotation
    this.rotate(theta, [...Y_AXIS], [0, 0, 0]);
    this.rotate(phi, [...Z_AXIS], [0, 0, 0]);

    // store new values
    this._current_theta = theta;
    this._current_phi = phi;
  }

  set_direction(direction: readonly number[] | NDArray): this {
    // if get_norm(direction) is get_norm(self.direction):
    //     pass
    this.direction = direction;
    this._rotate_to_direction();
    return this;
  }
}

export class Torus {
  R: number;
  r: number;

  constructor(major_radius = 3, minor_radius = 1) {
    this.R = major_radius;
    this.r = minor_radius;
  }

  func(u: number, v: number): SurfacePoint {
    const px = Math.cos(u);
    const py = Math.sin(u);
    const ring = this.R - this.r * Math.cos(v);
    return point3([ring * px, ring * py, -this.r * Math.sin(v)]);
  }
}

export type Line3DEndpoint = Mobject | number[] | readonly number[] | NDArray;

function flatPoint(value: NDArray): number[] {
  const raw = value.toArray() as unknown;
  const items = (Array.isArray(raw) ? raw : [raw]) as unknown[];
  const out: number[] = [];
  const walk = (item: unknown): void => {
    if (Array.isArray(item)) {
      for (const sub of item) walk(sub);
      return;
    }
    out.push(Number(item));
  };
  for (const item of items) walk(item);
  return out;
}

export class Line3D {
  thickness: number;
  color: ParsableManimColor | null;
  resolution: [number, number];
  vect!: number[];
  length!: number;
  direction!: number[];
  start!: NDArray;
  end!: NDArray;

  constructor(
    start: Line3DEndpoint = [...LEFT],
    end: Line3DEndpoint = [...RIGHT],
    thickness = 0.02,
    color: ParsableManimColor | null = null,
    resolution: number | [number, number] = 24,
  ) {
    this.thickness = thickness;
    // NOTE: color is accepted (same position as Python's __init__) and stored,
    // but applying it to geometry awaits the Cylinder port, which needs the
    // cairo renderer (out of scope for the interpreter target).
    this.color = color;
    this.resolution = typeof resolution === "number" ? [2, resolution] : [...resolution] as [number, number];
    this.set_start_and_end_attrs(start, end);
  }

  set_start_and_end_attrs(start: Line3DEndpoint, end: Line3DEndpoint): this {
    const rough_start = flatPoint(this.pointify(start));
    const rough_end = flatPoint(this.pointify(end));
    this.vect = rough_end.map((value, index) => value - (rough_start[index] ?? 0));
    this.length = Math.hypot(...this.vect);
    this.direction = normalize(this.vect);
    // NOTE: Python also rebuilds Cylinder geometry here (super().__init__
    // with height/radius/direction) and shifts to the midpoint; the TS port
    // is a lightweight stub that only tracks the endpoint attributes.
    this.start = this.pointify(start, this.direction);
    this.end = this.pointify(end, this.direction.map((value) => -value));
    return this;
  }

  get_start(): NDArray {
    return this.start;
  }

  get_end(): NDArray {
    return this.end;
  }

  pointify(mob_or_point: Mobject | number[] | readonly number[] | NDArray, direction?: number[] | readonly number[]): NDArray {
    if (mob_or_point instanceof Mobject) {
      const mob = mob_or_point;
      if (direction === undefined || direction === null) return array([...mob.get_center()]) as NDArray;
      const boundary = (mob as unknown as { get_boundary_point: (d: unknown) => number[] }).get_boundary_point(direction);
      return array([...boundary]) as NDArray;
    }
    // Python accepts any array-like (plain sequences and np.ndarray alike)
    // and returns np.array(...) preserving shape; mirror that with a copy.
    if (isNDArray(mob_or_point)) return (mob_or_point as NDArray).copy();
    return array([...(mob_or_point as readonly number[])].map(Number)) as NDArray;
  }
}

export interface Arrow3DOptions {
  thickness?: number;
  height?: number;
  base_radius?: number;
  color?: ParsableManimColor;
  resolution?: number | [number, number];
  [key: string]: unknown;
}

export class Arrow3D extends Line3D {
  cone!: Cone;
  end_point!: VectorizedPoint;

  constructor(
    start: Line3DEndpoint = [...LEFT],
    end: Line3DEndpoint = [...RIGHT],
    thickness = 0.02,
    height = 0.3,
    base_radius = 0.08,
    color: ParsableManimColor = WHITE,
    resolution: number | [number, number] = 24,
    options: Arrow3DOptions = {},
  ) {
    super(start, end, thickness, color, resolution);
    void options;
    this.length = norm(this.vect);
    const end_rows = pointRows(this.end, 3)[0] ?? [0, 0, 0];
    const shortened = end_rows.map((value, index) => value - height * (this.direction[index] ?? 0));
    this.set_start_and_end_attrs(this.start, shortened);
    this.cone = new Cone(base_radius, height, [...this.direction]);
    const np_end = pointRows(end, 3)[0] ?? [0, 0, 0];
    this.cone.shift(np_end);
    this.end_point = new VectorizedPoint(np_end);
    // NOTE: Python adds end_point/cone as submobjects and applies set_color
    // to the whole group; the TS Line3D base is a lightweight stub (not a
    // Mobject), so they are stored as fields and only the color attribute is
    // updated here.
    this.color = color;
  }

  override get_end(): NDArray {
    return array([...this.end_point.get_center()]) as NDArray;
  }
}
