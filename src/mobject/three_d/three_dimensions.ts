import { array, type NDArray } from "numpy-ts";
import { arange, isNDArray, linspace, norm, pointRows } from "../../utils/arrays.js";
import { IN, LEFT, ORIGIN, OUT, PI, RIGHT, TAU, UP, DOWN, X_AXIS, Y_AXIS, Z_AXIS } from "../../constants.js";
import { DEFAULT_DOT_RADIUS, LineJointType } from "../../constants.js";
import { config } from "../../_config/utils.js";
import { RendererType } from "../../constants.js";
import { normalize, z_to_vector, cross, perpendicular_bisector } from "../../utils/space_ops.js";
import { BLUE, BLUE_D, BLUE_E, LIGHT_GREY, ManimColor, WHITE, interpolate_color, type ParsableManimColor } from "../../utils/color/core.js";
import type { ThreeDAxes } from "../graphing/coordinate_systems.js";
import { Mobject } from "../mobject.js";
import { VGroup, VectorizedPoint, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Circle } from "../geometry/arc.js";
import { Square } from "../geometry/polygram.js";
import { isPlainOptionsBag } from "../geometry/polygram.js";

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
    // NOTE: Python forwards fill_color/fill_opacity/stroke_color/stroke_width
    // to VGroup.__init__ (which stores them as fields via VMobject.__init__);
    // TS requires super() first and VGroup takes no style options, so the
    // same field assignment (+init_colors, mirroring VMobject.__init__'s
    // tail) runs here instead.
    this.fill_color = new ManimColor(fill_color);
    this.fill_opacity = fill_opacity;
    this.stroke_color = new ManimColor(stroke_color);
    this.stroke_width = stroke_width;
    this.init_colors();
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

  set_fill_by_value(
    axes: ThreeDAxes,
    colorscale: Iterable<ParsableManimColor | [ParsableManimColor, number]> | null = null,
    axis = 2,
    kwargs: Record<string, unknown> = {},
  ): this {
    // NOTE: mirrors Python's legacy `colors` keyword (accepted only when
    // colorscale is None, and only when it is the sole kwarg).
    let resolvedColorscale = colorscale;
    const rest = { ...kwargs };
    if ("colors" in rest && resolvedColorscale === null) {
      resolvedColorscale = rest["colors"] as Iterable<ParsableManimColor | [ParsableManimColor, number]>;
      delete rest["colors"];
      if (Object.keys(rest).length > 0) {
        const error = new Error(
          `Unsupported keyword argument(s): ${Object.keys(rest).join(", ")}`,
        );
        error.name = "ValueError";
        throw error;
      }
    }
    if (resolvedColorscale === null || resolvedColorscale === undefined) {
      console.warn(
        "The value passed to the colorscale keyword argument was None, "
        + "the surface fill color has not been changed",
      );
      return this;
    }
    const colorscale_list = [...resolvedColorscale];

    const ranges = [axes.x_range, axes.y_range, axes.z_range];
    const first = colorscale_list[0] as ParsableManimColor | [ParsableManimColor, number];
    let new_colors: ManimColor[];
    let pivots: number[];
    if (Array.isArray(first) && first.length === 2 && typeof first[1] === "number") {
      new_colors = (colorscale_list as Array<[ParsableManimColor, number]>).map(([c]) => new ManimColor(c));
      pivots = (colorscale_list as Array<[ParsableManimColor, number]>).map(([, p]) => p);
    } else {
      new_colors = (colorscale_list as ParsableManimColor[]).map((c) => new ManimColor(c));
      const current_range = ranges[axis];
      if (current_range === null || current_range === undefined) {
        throw new TypeError("axis range is None");
      }
      const pivot_min = current_range[0] ?? 0;
      const pivot_max = current_range[1] ?? 0;
      const pivot_frequency = (pivot_max - pivot_min) / (new_colors.length - 1);
      pivots = arange(pivot_min, pivot_max + pivot_frequency, pivot_frequency);
    }

    for (const mob of this.family_members_with_points()) {
      const midpoint = mob.get_midpoint();
      const axis_value = axes.point_to_coords(
        (pointRows(midpoint, 3)[0] ?? [0, 0, 0]) as readonly number[],
      )[axis] ?? 0;
      if (axis_value <= (pivots[0] ?? 0)) {
        mob.set_color(new_colors[0] as ParsableManimColor);
      } else if (axis_value >= (pivots[pivots.length - 1] ?? 0)) {
        mob.set_color(new_colors[new_colors.length - 1] as ParsableManimColor);
      } else {
        for (let i = 0; i < pivots.length; i += 1) {
          if ((pivots[i] ?? 0) > axis_value) {
            const prev = pivots[i - 1] ?? 0;
            const curr = pivots[i] ?? 0;
            let color_index = (axis_value - prev) / (curr - prev);
            color_index = Math.min(color_index, 1);
            const mob_color = interpolate_color(
              new_colors[i - 1] as ManimColor,
              new_colors[i] as ManimColor,
              color_index,
            );
            // NOTE: mirrors Python's renderer branch (OpenGL:
            // set_color(mob_color, recurse=False); Cairo: set_color(mob_color,
            // family=False)); both are non-recursive.
            if (config.renderer === RendererType.OPENGL) {
              (mob as VMobject).set_color(mob_color, false);
            } else if (config.renderer === RendererType.CAIRO) {
              (mob as VMobject).set_color(mob_color, false);
            }
            break;
          }
        }
      }
    }

    return this;
  }
}

function spherePoint(radius: number, u: number, v: number): SurfacePoint {
  return array([
    radius * Math.cos(u) * Math.sin(v),
    radius * Math.sin(u) * Math.sin(v),
    -radius * Math.cos(v),
  ]) as SurfacePoint;
}

export interface SphereOptions {
  fill_color?: ParsableManimColor;
  fill_opacity?: number;
  checkerboard_colors?: Iterable<ParsableManimColor> | false;
  stroke_color?: ParsableManimColor;
  stroke_width?: number;
  should_make_jagged?: boolean;
  surface_piece_config?: Record<string, unknown>;
  pre_function_handle_to_anchor_scale_factor?: number;
  [key: string]: unknown;
}

export class Sphere extends Surface {
  radius: number;

  constructor(
    centerOrOptions: readonly number[] | NDArray | (SphereOptions & {
      center?: readonly number[] | NDArray;
      radius?: number;
      resolution?: number | [number, number];
      u_range?: [number, number];
      v_range?: [number, number];
    }) = [0, 0, 0],
    radius = 1,
    // NOTE: Python defaults the resolution from config.renderer
    // ((101, 51) for OpenGL, (24, 12) for Cairo); renderers are out of scope
    // for the interpreter target, so the Cairo default is used.
    resolution: number | [number, number] = [24, 12],
    u_range: [number, number] = [0, Math.PI * 2],
    v_range: [number, number] = [0, Math.PI],
    options: SphereOptions = {},
  ) {
    // Python: Sphere(center=ORIGIN, radius=1, resolution=None,
    // u_range=(0,TAU), v_range=(0,PI), **kwargs). Accept a leading kwargs bag.
    let bag: SphereOptions & Record<string, unknown>;
    let positionalCenter: readonly number[] | NDArray | undefined;
    if (isPlainOptionsBag(centerOrOptions)) {
      bag = { ...(centerOrOptions as Record<string, unknown>), ...(options as Record<string, unknown>) };
    } else {
      bag = { ...(options as Record<string, unknown>) };
      positionalCenter = centerOrOptions as readonly number[] | NDArray;
    }
    const resolvedCenter = (bag["center"] as readonly number[] | NDArray | undefined) ?? positionalCenter ?? [0, 0, 0];
    const resolvedRadius = (bag["radius"] as number | undefined) ?? radius;
    const resolvedResolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
    const resolvedURange = (bag["u_range"] as [number, number] | undefined) ?? u_range;
    const resolvedVRange = (bag["v_range"] as [number, number] | undefined) ?? v_range;
    for (const key of ["center", "radius", "resolution", "u_range", "v_range"]) delete bag[key];
    // NOTE: the closure captures `radius` instead of `this.func` because
    // `this` cannot be referenced in a super() call.
    super(
      (u: number, v: number): SurfacePoint => spherePoint(resolvedRadius, u, v),
      resolvedURange,
      resolvedVRange,
      resolvedResolution,
      (bag.surface_piece_config as Record<string, unknown> | undefined) ?? {},
      (bag.fill_color as ParsableManimColor | undefined) ?? BLUE_D,
      (bag.fill_opacity as number | undefined) ?? 1.0,
      (bag.checkerboard_colors as Iterable<ParsableManimColor> | false | undefined) ?? [BLUE_D, BLUE_E],
      (bag.stroke_color as ParsableManimColor | undefined) ?? LIGHT_GREY,
      (bag.stroke_width as number | undefined) ?? 0.5,
      (bag.should_make_jagged as boolean | undefined) ?? false,
      (bag.pre_function_handle_to_anchor_scale_factor as number | undefined) ?? 0.00001,
      bag as SphereOptions,
    );
    this.radius = resolvedRadius;
    this.shift(pointRows(resolvedCenter, 3)[0] ?? [0, 0, 0]);
  }

  override func(u: number, v: number): SurfacePoint {
    return spherePoint(this.radius, u, v);
  }
}

export class Dot3D extends Sphere {
  constructor(
    pointOrOptions: readonly number[] | NDArray | (SphereOptions & {
      point?: readonly number[] | NDArray;
      radius?: number;
      color?: ParsableManimColor;
      resolution?: number | [number, number];
    }) = [...ORIGIN],
    radius = DEFAULT_DOT_RADIUS,
    color: ParsableManimColor = WHITE,
    resolution: number | [number, number] = [8, 8],
    options: Record<string, unknown> = {},
  ) {
    // Python: Dot3D(point=ORIGIN, radius=0.08, color=WHITE,
    // resolution=(8,8), **kwargs).
    const isBag = isPlainOptionsBag(pointOrOptions);
    const bag: Record<string, unknown> = isBag
      ? { ...(pointOrOptions as Record<string, unknown>), ...options }
      : { ...options };
    const positionalPoint = isBag ? undefined : (pointOrOptions as readonly number[] | NDArray);
    const resolvedPoint = (bag["point"] as readonly number[] | NDArray | undefined) ?? positionalPoint ?? [...ORIGIN];
    const resolvedRadius = (bag["radius"] as number | undefined) ?? radius;
    const resolvedColor = (bag["color"] as ParsableManimColor | undefined) ?? color;
    const resolvedResolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
    for (const key of ["point", "radius", "color", "resolution"]) delete bag[key];
    super(resolvedPoint, resolvedRadius, resolvedResolution, [0, Math.PI * 2], [0, Math.PI], bag as SphereOptions);
    this.set_color(resolvedColor);
  }
}

export interface CubeOptions {
  side_length?: number;
  fill_opacity?: number;
  fill_color?: ParsableManimColor;
  stroke_width?: number;
  [key: string]: unknown;
}

export class Cube extends VGroup {
  side_length: number;

  constructor(
    sideLengthOrOptions: number | CubeOptions = 2,
    fill_opacity = 0.75,
    fill_color: ParsableManimColor = BLUE,
    stroke_width = 0,
    options: CubeOptions = {},
  ) {
    // Python: Cube(side_length=2, fill_opacity=0.75, fill_color=BLUE_D,
    // stroke_width=0, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(sideLengthOrOptions);
    const bag: CubeOptions = isBag
      ? { ...(sideLengthOrOptions as CubeOptions), ...options }
      : options;
    const positionalSide = isBag ? undefined : (sideLengthOrOptions as number);
    const {
      side_length: sideLengthOption,
      fill_opacity: fillOpacityOption,
      fill_color: fillColorOption,
      stroke_width: strokeWidthOption,
      ...rest
    } = bag;
    void rest;
    super();
    this.side_length = sideLengthOption ?? positionalSide ?? 2;
    this.set_fill(fillColorOption ?? fill_color, fillOpacityOption ?? fill_opacity);
    this.set_stroke(undefined, strokeWidthOption ?? stroke_width);
    this.generate_points();
    // NOTE: Python's Mobject.__init__ calls init_colors() *after*
    // generate_points(), which (with family=True) propagates the Cube's
    // fill/stroke style onto the freshly added Square faces. In particular
    // stroke_width=0 must reach the faces, otherwise they keep the VMobject
    // default 4. TS runs generate_points() after super(), so mirror the
    // propagation explicitly.
    this.init_colors();
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
    dimensionsOrOptions: readonly number[] | NDArray | (PrismOptions & { dimensions?: readonly number[] | NDArray }) = [3, 2, 1],
    options: PrismOptions = {},
  ) {
    // Python: Prism(dimensions=[3,2,1], **kwargs). Accept a leading kwargs bag.
    let bag: PrismOptions & { dimensions?: readonly number[] | NDArray };
    let positionalDimensions: readonly number[] | NDArray | undefined;
    if (isPlainOptionsBag(dimensionsOrOptions)) {
      bag = { ...(dimensionsOrOptions as PrismOptions & { dimensions?: readonly number[] | NDArray }), ...options };
    } else {
      bag = options;
      positionalDimensions = dimensionsOrOptions as readonly number[] | NDArray;
    }
    const { dimensions: dimensionsOption, side_length = 2, ...rest } = bag;
    super(
      side_length,
      rest.fill_opacity ?? 0.75,
      rest.fill_color ?? BLUE,
      rest.stroke_width ?? 0,
      rest,
    );
    this.dimensions = [...(pointRows(dimensionsOption ?? positionalDimensions ?? [3, 2, 1], 3)[0] ?? [3, 2, 1])];
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
    baseRadiusOrOptions: number | (ConeOptions & {
      base_radius?: number;
      height?: number;
      direction?: readonly number[] | NDArray;
      show_base?: boolean;
      v_range?: [number, number];
      u_min?: number;
      checkerboard_colors?: Iterable<ParsableManimColor> | false;
    }) = 1,
    height = 1,
    direction: readonly number[] | NDArray = [...Z_AXIS],
    show_base = false,
    v_range: [number, number] = [0, TAU],
    u_min = 0,
    checkerboard_colors: Iterable<ParsableManimColor> | false = false,
    options: ConeOptions = {},
  ) {
    // Python: Cone(base_radius=1, height=1, direction=OUT, show_base=False,
    // v_range=(0,TAU), u_min=0, checkerboard_colors=False, **kwargs).
    const isBag = isPlainOptionsBag(baseRadiusOrOptions);
    const bag: Record<string, unknown> = isBag
      ? { ...(baseRadiusOrOptions as Record<string, unknown>), ...(options as Record<string, unknown>) }
      : { ...(options as Record<string, unknown>) };
    const positionalBase = isBag ? undefined : (baseRadiusOrOptions as number);
    const base_radius = (bag["base_radius"] as number | undefined) ?? positionalBase ?? 1;
    height = (bag["height"] as number | undefined) ?? height;
    direction = (bag["direction"] as readonly number[] | NDArray | undefined) ?? direction;
    show_base = (bag["show_base"] as boolean | undefined) ?? show_base;
    v_range = (bag["v_range"] as [number, number] | undefined) ?? v_range;
    u_min = (bag["u_min"] as number | undefined) ?? u_min;
    checkerboard_colors = (bag["checkerboard_colors"] as Iterable<ParsableManimColor> | false | undefined) ?? checkerboard_colors;
    for (const key of ["base_radius", "height", "direction", "show_base", "v_range", "u_min", "checkerboard_colors"]) delete bag[key];
    // NOTE: Python assigns direction/theta before super().__init__() so the
    // bound self.func sees them. TS requires super() first; the closure
    // below captures the locals instead (identical values).
    const theta = PI - Math.atan(base_radius / height);
    const u_range: [number, number] = [u_min, Math.sqrt(base_radius ** 2 + height ** 2)];
    super(
      (u: number, v: number) => conePoint(u, v, theta),
      u_range,
      v_range,
      (bag["resolution"] as number | [number, number] | undefined) ?? 32,
      (bag["surface_piece_config"] as Record<string, unknown> | undefined) ?? {},
      (bag["fill_color"] as ParsableManimColor | undefined) ?? BLUE_D,
      (bag["fill_opacity"] as number | undefined) ?? 1.0,
      checkerboard_colors,
      (bag["stroke_color"] as ParsableManimColor | undefined) ?? LIGHT_GREY,
      (bag["stroke_width"] as number | undefined) ?? 0.5,
      (bag["should_make_jagged"] as boolean | undefined) ?? false,
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
    radiusOrOptions: number | (CylinderOptions & {
      radius?: number;
      height?: number;
      direction?: readonly number[] | NDArray;
      v_range?: [number, number];
      show_ends?: boolean;
      resolution?: number | [number, number];
    }) = 1,
    height = 2,
    direction: readonly number[] | NDArray = [...Z_AXIS],
    v_range: [number, number] = [0, TAU],
    show_ends = true,
    resolution: number | [number, number] = [24, 24],
    options: CylinderOptions = {},
  ) {
    // Python: Cylinder(radius=1, height=2, direction=OUT, v_range=(0,TAU),
    // show_ends=True, resolution=(24,24), **kwargs).
    const isBag = isPlainOptionsBag(radiusOrOptions);
    const bag: Record<string, unknown> = isBag
      ? { ...(radiusOrOptions as Record<string, unknown>), ...(options as Record<string, unknown>) }
      : { ...(options as Record<string, unknown>) };
    const positionalRadius = isBag ? undefined : (radiusOrOptions as number);
    const radius = (bag["radius"] as number | undefined) ?? positionalRadius ?? 1;
    height = (bag["height"] as number | undefined) ?? height;
    direction = (bag["direction"] as readonly number[] | NDArray | undefined) ?? direction;
    v_range = (bag["v_range"] as [number, number] | undefined) ?? v_range;
    show_ends = (bag["show_ends"] as boolean | undefined) ?? show_ends;
    resolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
    for (const key of ["radius", "height", "direction", "v_range", "show_ends", "resolution"]) delete bag[key];
    // NOTE: same super()-first constraint as Cone (see above); the closure
    // captures the radius/height params.
    super(
      (u: number, v: number) => cylinderPoint(u, v, radius),
      [-height / 2, height / 2],
      v_range,
      resolution,
      (bag["surface_piece_config"] as Record<string, unknown> | undefined) ?? {},
      (bag["fill_color"] as ParsableManimColor | undefined) ?? BLUE_D,
      (bag["fill_opacity"] as number | undefined) ?? 1.0,
      false,
      (bag["stroke_color"] as ParsableManimColor | undefined) ?? LIGHT_GREY,
      (bag["stroke_width"] as number | undefined) ?? 0.5,
      (bag["should_make_jagged"] as boolean | undefined) ?? false,
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

  // NOTE: Python's Cylinder shadows VMobject.get_direction (which returns
  // the winding direction "CW"/"CCW") with an incompatible vector return
  // type; the `any` return type carries that shadowing into TS (same as
  // Brace.get_direction).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  get_direction(): any {
    return this.direction;
  }
}

function torusPoint(u: number, v: number, R: number, r: number): SurfacePoint {
  const px = Math.cos(u);
  const py = Math.sin(u);
  const ring = R - r * Math.cos(v);
  // NOTE: mirrors `(R - r*cos(v)) * P - r*sin(v) * OUT` with
  // P = [cos u, sin u, 0] and OUT = [0, 0, 1].
  return point3([ring * px, ring * py, -r * Math.sin(v)]);
}

export interface TorusOptions {
  resolution?: number | [number, number];
  surface_piece_config?: Record<string, unknown>;
  fill_color?: ParsableManimColor;
  fill_opacity?: number;
  checkerboard_colors?: Iterable<ParsableManimColor> | false;
  stroke_color?: ParsableManimColor;
  stroke_width?: number;
  should_make_jagged?: boolean;
  [key: string]: unknown;
}

export class Torus extends Surface {
  R: number;
  r: number;

  constructor(
    majorRadiusOrOptions: number | (TorusOptions & {
      major_radius?: number;
      minor_radius?: number;
      u_range?: [number, number];
      v_range?: [number, number];
      resolution?: number | [number, number];
    }) = 3,
    minor_radius = 1,
    u_range: [number, number] = [0, TAU],
    v_range: [number, number] = [0, TAU],
    // NOTE: Python defaults the resolution from config.renderer
    // ((101, 101) for OpenGL, (24, 24) for Cairo); the Cairo default is
    // used here (same convention as Sphere/Cylinder in this port).
    resolution: number | [number, number] = [24, 24],
    options: TorusOptions = {},
  ) {
    // Python: Torus(major_radius=3, minor_radius=1, u_range=(0,TAU),
    // v_range=(0,TAU), resolution=None, **kwargs).
    const isBag = isPlainOptionsBag(majorRadiusOrOptions);
    const bag: Record<string, unknown> = isBag
      ? { ...(majorRadiusOrOptions as Record<string, unknown>), ...(options as Record<string, unknown>) }
      : { ...(options as Record<string, unknown>) };
    const positionalMajor = isBag ? undefined : (majorRadiusOrOptions as number);
    const major_radius = (bag["major_radius"] as number | undefined) ?? positionalMajor ?? 3;
    minor_radius = (bag["minor_radius"] as number | undefined) ?? minor_radius;
    u_range = (bag["u_range"] as [number, number] | undefined) ?? u_range;
    v_range = (bag["v_range"] as [number, number] | undefined) ?? v_range;
    resolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
    for (const key of ["major_radius", "minor_radius", "u_range", "v_range", "resolution"]) delete bag[key];
    // NOTE: same super()-first constraint as Cone/Cylinder (see above);
    // the closure captures the radii params instead of `this`.
    super(
      (u: number, v: number) => torusPoint(u, v, major_radius, minor_radius),
      u_range,
      v_range,
      resolution,
      (bag["surface_piece_config"] as Record<string, unknown> | undefined) ?? {},
      (bag["fill_color"] as ParsableManimColor | undefined) ?? BLUE_D,
      (bag["fill_opacity"] as number | undefined) ?? 1.0,
      (bag["checkerboard_colors"] as Iterable<ParsableManimColor> | false | undefined) ?? [BLUE_D, BLUE_E],
      (bag["stroke_color"] as ParsableManimColor | undefined) ?? LIGHT_GREY,
      (bag["stroke_width"] as number | undefined) ?? 0.5,
      (bag["should_make_jagged"] as boolean | undefined) ?? false,
    );
    this.R = major_radius;
    this.r = minor_radius;
  }

  override func(u: number, v: number): SurfacePoint {
    return torusPoint(u, v, this.R, this.r);
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
    startOrOptions: Line3DEndpoint | {
      start?: Line3DEndpoint;
      end?: Line3DEndpoint;
      thickness?: number;
      color?: ParsableManimColor | null;
      resolution?: number | [number, number];
    } = [...LEFT],
    end: Line3DEndpoint = [...RIGHT],
    thickness = 0.02,
    color: ParsableManimColor | null = null,
    resolution: number | [number, number] = 24,
  ) {
    // Python: Line3D(start=LEFT, end=RIGHT, thickness=0.02, color=None,
    // resolution=24, **kwargs). Accept a leading kwargs bag.
    const isBag = isPlainOptionsBag(startOrOptions);
    const bag = isBag
      ? { ...(startOrOptions as Record<string, unknown>) }
      : {} as Record<string, unknown>;
    const start = isBag
      ? (bag["start"] as Line3DEndpoint | undefined) ?? [...LEFT]
      : (startOrOptions as Line3DEndpoint);
    end = isBag ? (bag["end"] as Line3DEndpoint | undefined) ?? end : end;
    thickness = (bag["thickness"] as number | undefined) ?? thickness;
    color = (bag["color"] as ParsableManimColor | null | undefined) ?? color;
    resolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
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

  static parallel_to(
    line: Line3D,
    point: number[] | readonly number[] | NDArray = [...ORIGIN],
    length = 5,
    options: { thickness?: number; color?: ParsableManimColor | null; resolution?: number | [number, number] } = {},
  ): Line3D {
    const np_point = pointRows(isNDArray(point) ? (point as NDArray).copy() : [...(point as readonly number[])], 3)[0] ?? [0, 0, 0];
    const vect = normalize(line.vect);
    return new Line3D(
      np_point.map((value, index) => value + (vect[index] ?? 0) * length / 2),
      np_point.map((value, index) => value - (vect[index] ?? 0) * length / 2),
      options.thickness ?? 0.02,
      options.color ?? null,
      options.resolution ?? 24,
    );
  }

  static perpendicular_to(
    line: Line3D,
    point: number[] | readonly number[] | NDArray = [...ORIGIN],
    length = 5,
    options: { thickness?: number; color?: ParsableManimColor | null; resolution?: number | [number, number] } = {},
  ): Line3D {
    const np_point = pointRows(isNDArray(point) ? (point as NDArray).copy() : [...(point as readonly number[])], 3)[0] ?? [0, 0, 0];
    const startRows = pointRows(line.start, 3)[0] ?? [0, 0, 0];
    const diff = np_point.map((value, index) => value - (startRows[index] ?? 0));
    const normVec = cross(line.vect, diff);
    if (normVec.every((value) => value === 0)) {
      const error = new Error("Could not find the perpendicular.");
      error.name = "ValueError";
      throw error;
    }

    const [bisStart, bisEnd] = perpendicular_bisector([line.start, line.end], normVec);
    const bisVect = normalize(
      (bisEnd ?? []).map((value, index) => value - ((bisStart ?? [])[index] ?? 0)),
    );
    return new Line3D(
      np_point.map((value, index) => value + (bisVect[index] ?? 0) * length / 2),
      np_point.map((value, index) => value - (bisVect[index] ?? 0) * length / 2),
      options.thickness ?? 0.02,
      options.color ?? null,
      options.resolution ?? 24,
    );
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
    startOrOptions: Line3DEndpoint | Arrow3DOptions = [...LEFT],
    end: Line3DEndpoint = [...RIGHT],
    thickness = 0.02,
    height = 0.3,
    base_radius = 0.08,
    color: ParsableManimColor = WHITE,
    resolution: number | [number, number] = 24,
    options: Arrow3DOptions = {},
  ) {
    // Python: Arrow3D(start=LEFT, end=RIGHT, thickness=0.02, height=0.3,
    // base_radius=0.08, color=WHITE, resolution=24, **kwargs).
    const isBag = isPlainOptionsBag(startOrOptions);
    const bag: Record<string, unknown> = isBag
      ? { ...(startOrOptions as Record<string, unknown>), ...options }
      : { ...options };
    const positionalStart = isBag ? undefined : (startOrOptions as Line3DEndpoint);
    const start = positionalStart ?? (bag["start"] as Line3DEndpoint | undefined) ?? [...LEFT];
    end = (bag["end"] as Line3DEndpoint | undefined) ?? end;
    thickness = (bag["thickness"] as number | undefined) ?? thickness;
    height = (bag["height"] as number | undefined) ?? height;
    base_radius = (bag["base_radius"] as number | undefined) ?? base_radius;
    color = (bag["color"] as ParsableManimColor | undefined) ?? color;
    resolution = (bag["resolution"] as number | [number, number] | undefined) ?? resolution;
    super(start, end, thickness, color, resolution);
    void bag;
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
