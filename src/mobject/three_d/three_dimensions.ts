import { array, type NDArray } from "numpy-ts";
import { isNDArray, linspace, pointRows } from "../../utils/arrays.js";
import { LEFT, RIGHT } from "../../constants.js";
import { normalize } from "../../utils/space_ops.js";
import type { ParsableManimColor } from "../../utils/color/core.js";
import { Mobject } from "../mobject.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";

export type SurfacePoint = NDArray;

function point3(values: [number, number, number]): SurfacePoint {
  return array(values) as SurfacePoint;
}

export type SurfaceFunc = (u: number, v: number) => SurfacePoint | number[];

export class Surface extends VGroup {
  u_range: [number, number];
  v_range: [number, number];
  resolution: number | [number, number];
  protected _func: SurfaceFunc;

  constructor(
    func: SurfaceFunc,
    u_range: [number, number] = [0, 1],
    v_range: [number, number] = [0, 1],
    resolution: number | [number, number] = 32,
  ) {
    super();
    this.u_range = u_range;
    this.v_range = v_range;
    this.resolution = resolution;
    this._func = func;
    this._setup_in_uv_space();
    const captured = this._func;
    this.apply_function((p: number[]): number[] => {
      const out = captured(p[0] ?? 0, p[1] ?? 0);
      return pointRows(out, 3)[0] ?? [0, 0, 0];
    });
    // NOTE: checkerboard fill colors, stroke styling, ThreeDVMobject shading
    // and make_jagged depend on not-yet-translated pieces; they set visual
    // style only and do not affect the surface point geometry built above.
  }

  func(u: number, v: number): SurfacePoint | number[] {
    return this._func(u, v);
  }

  _setup_in_uv_space(): void {
    const [u_values, v_values] = this._get_u_values_and_v_values();
    for (let i = 0; i < u_values.length - 1; i += 1) {
      for (let j = 0; j < v_values.length - 1; j += 1) {
        const u1 = u_values[i] ?? 0;
        const u2 = u_values[i + 1] ?? 0;
        const v1 = v_values[j] ?? 0;
        const v2 = v_values[j + 1] ?? 0;
        const face = new VMobject();
        face.set_points_as_corners([
          [u1, v1, 0],
          [u2, v1, 0],
          [u2, v2, 0],
          [u1, v2, 0],
          [u1, v1, 0],
        ]);
        this.add(face);
      }
    }
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

export class Cone {
  theta: number;

  constructor(theta = 0.5) {
    this.theta = theta;
  }

  func(u: number, v: number): SurfacePoint {
    const r = u;
    const phi = v;
    return point3([
      r * Math.sin(this.theta) * Math.cos(phi),
      r * Math.sin(this.theta) * Math.sin(phi),
      r * Math.cos(this.theta),
    ]);
  }
}

export class Cylinder {
  radius: number;

  constructor(radius = 1) {
    this.radius = radius;
  }

  func(u: number, v: number): SurfacePoint {
    const height = u;
    const phi = v;
    const r = this.radius;
    return point3([r * Math.cos(phi), r * Math.sin(phi), height]);
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
