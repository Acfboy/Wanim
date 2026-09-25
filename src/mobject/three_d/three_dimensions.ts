import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";

export type SurfacePoint = NDArray;

function point3(values: [number, number, number]): SurfacePoint {
  return array(values) as SurfacePoint;
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
