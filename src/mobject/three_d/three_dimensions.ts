import { array, type NDArray } from "numpy-ts";
import { linspace, pointRows } from "../../utils/arrays.js";
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

export class Line3D {
  thickness: number;

  constructor(thickness = 0.02) {
    this.thickness = thickness;
  }

  pointify(mob_or_point: Mobject | number[] | readonly number[] | NDArray, direction?: number[] | readonly number[]): NDArray {
    if (mob_or_point instanceof Mobject) {
      const mob = mob_or_point;
      if (direction === undefined || direction === null) return array([...mob.get_center()]) as NDArray;
      const boundary = (mob as unknown as { get_boundary_point: (d: unknown) => number[] }).get_boundary_point(direction);
      return array([...boundary]) as NDArray;
    }
    return array([...(mob_or_point as readonly number[])].map(Number)) as NDArray;
  }
}
