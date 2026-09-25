import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import { ParametricFunction } from "../src/mobject/graphing/functions.js";
import { Axes, CoordinateSystem } from "../src/mobject/graphing/coordinate_systems.js";
import { LogBase } from "../src/mobject/graphing/scale.js";
import { Cone, Cylinder, Line3D, Torus } from "../src/mobject/three_d/three_dimensions.js";
import { Polyhedron } from "../src/mobject/three_d/polyhedra.js";
import { CapStyleType, LineJointType } from "../src/constants.js";
import { ImageMobject, AbstractImageMobject } from "../src/mobject/types/image_mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function nums(value: unknown): number[] {
  if (Array.isArray(value)) {
    const flat = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)]);
    return flat(value);
  }
  const raw = (value as { toArray: () => unknown }).toArray();
  const flat = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)]);
  return flat(raw);
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = nums(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]) * relTol);
    expect(Math.abs(values[i] - expected[i])).toBeLessThanOrEqual(tol);
  }
}

describe("ParametricFunction internal_parametric_function", () => {
  // Python: pf = ParametricFunction(lambda t: [t, t**2, 0]);
  //   pf.function(0.5) == [0.5, 0.25, 0]; t_min/max/step == 0/1/0.01.
  it("wraps the user function output in an array", () => {
    const pf = new ParametricFunction((t: number) => [t, t ** 2, 0]);
    expectClose(pf.function(0.5), [0.5, 0.25, 0]);
    expectClose(pf.get_point_from_function(0.25), [0.25, 0.0625, 0]);
  });
  it("defaults a 2-element t_range to step 0.01", () => {
    const pf = new ParametricFunction((t: number) => [t, 0, 0]);
    expect([pf.t_min, pf.t_max, pf.t_step]).toEqual([0, 1, 0.01]);
  });
  it("keeps an explicit 3-element t_range", () => {
    const pf = new ParametricFunction((t: number) => [t, 0, 0], { t_range: [1, 3, 0.5] });
    expect([pf.t_min, pf.t_max, pf.t_step]).toEqual([1, 3, 0.5]);
  });
});

describe("CoordinateSystem.point_to_coords", () => {
  it("raises NotImplementedError on the base class", () => {
    const cs = new CoordinateSystem();
    let thrown: unknown;
    try {
      cs.point_to_coords([0, 0, 0]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("NotImplementedError");
  });
});

describe("CoordinateSystem._origin_shift", () => {
  // Python: Axes._origin_shift([1,5,1]) == 1; ([-5,-1,1]) == -1; ([-5,5,1]) == 0.
  it("matches Axes ground truths", () => {
    expect(Axes._origin_shift([1, 5, 1])).toBe(1);
    expect(Axes._origin_shift([-5, -1, 1])).toBe(-1);
    expect(Axes._origin_shift([-5, 5, 1])).toBe(0);
  });
  it("base implementation throws (Python returns Ellipsis there; port uses an explicit error)", () => {
    // NOTE: deviation — CPython's abstract stub body is `...` so it returns
    // Ellipsis instead of raising; the TS port throws to mark it abstract.
    // Only the Axes override carries numeric ground truth.
    let thrown: unknown;
    try {
      CoordinateSystem._origin_shift([0, 1]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("NotImplementedError");
  });
});

describe("LogBase.inverse_function", () => {
  // Python LogBase(10): scalar 100 -> 2.0; array [1,10,100] -> [0,1,2];
  // scalar 0 and all-zero arrays raise ValueError.
  it("inverts scalars", () => {
    expect(new LogBase(10).inverse_function(100)).toBe(2.0);
  });
  it("inverts arrays elementwise", () => {
    expectClose(new LogBase(10).inverse_function(array([1, 10, 100])), [0.0, 1.0, 2.0]);
  });
  it("raises ValueError for non-positive scalars", () => {
    let thrown: unknown;
    try {
      new LogBase(10).inverse_function(0);
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe(
      "log(0) is undefined. Make sure the value is in the domain of the function",
    );
  });
  it("raises ValueError for an all-zero array", () => {
    let thrown: unknown;
    try {
      new LogBase(10).inverse_function(array([0, 0]));
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("ValueError");
  });
});

describe("Cone.func / Cylinder.func / Torus.func", () => {
  // Python Cone(base_radius=0.5, height=1): theta = pi - atan(0.5);
  //   func(1.0, 0.0) == [0.4472135954999581, 0.0, -0.8944271909999159].
  // NOTE: the TS Cone stores theta directly (constructor differs from Python's
  // base_radius/height form, which is out of this batch); func's formula is identical.
  it("Cone.func matches Python given the same theta", () => {
    const theta = Math.PI - Math.atan(0.5 / 1);
    expectClose(new Cone(theta).func(1.0, 0.0), [0.4472135954999581, 0.0, -0.8944271909999159]);
  });
  it("Cylinder.func matches Python", () => {
    // Python Cylinder(radius=2).func(1.5, pi/2) == [~0, 2.0, 1.5].
    expectClose(new Cylinder(2.0).func(1.5, Math.PI / 2), [1.2246467991473532e-16, 2.0, 1.5]);
  });
  it("Torus.func matches Python", () => {
    // Python Torus(3,1).func(0,0) == [2,0,0]; func(pi/2,pi) == [~0,4,~0].
    expectClose(new Torus(3, 1).func(0.0, 0.0), [2.0, 0.0, 0.0]);
    expectClose(new Torus(3, 1).func(Math.PI / 2, Math.PI), [2.4492935982947064e-16, 4.0, -1.2246467991473532e-16]);
  });
});

describe("Line3D.pointify", () => {
  it("passes plain points through as arrays", () => {
    expectClose(new Line3D().pointify([1, 2, 3]), [1, 2, 3]);
  });
  it("returns a mobject's center when no direction is given", () => {
    const mob = new VMobject();
    mob.set_points_as_corners([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    expectClose(new Line3D().pointify(mob), nums(mob.get_center()));
  });
});

describe("Polyhedron.get_edges", () => {
  // Python: get_edges([[0,1,2],[2,3,0]]) ==
  //   [(0,1),(1,2),(2,0),(2,3),(3,0),(0,2)].
  it("creates cyclic pairwise tuples per face", () => {
    expect(new Polyhedron().get_edges([[0, 1, 2], [2, 3, 0]])).toEqual([
      [0, 1],
      [1, 2],
      [2, 0],
      [2, 3],
      [3, 0],
      [0, 2],
    ]);
  });
});

describe("CapStyleType / LineJointType", () => {
  it("has the manim integer values", () => {
    expect(CapStyleType.AUTO).toBe(0);
    expect(CapStyleType.ROUND).toBe(1);
    expect(CapStyleType.BUTT).toBe(2);
    expect(CapStyleType.SQUARE).toBe(3);
    expect(LineJointType.AUTO).toBe(0);
    expect(LineJointType.ROUND).toBe(1);
    expect(LineJointType.BEVEL).toBe(2);
    expect(LineJointType.MITER).toBe(3);
  });
});

describe("ImageMobject.set_opacity", () => {
  // Python: pixel [[[10,20,30,200],[40,50,60,100]]], orig_alpha [[200,100]];
  //   set_opacity(0.5) -> alpha [100, 50], stroke_opacity 0.5;
  //   set_opacity(1.0) restores [200, 100].
  function make() {
    return new ImageMobject({
      pixel_array: array([
        [
          [10, 20, 30, 200],
          [40, 50, 60, 100],
        ],
      ]),
    });
  }
  it("scales the stored original alpha channel", () => {
    const im = make();
    expect(im.orig_alpha_pixel_array).toEqual([[200, 100]]);
    im.set_opacity(0.5);
    expect(nums(im.pixel_array)).toEqual([10, 20, 30, 100, 40, 50, 60, 50]);
    expect(im.stroke_opacity).toBe(0.5);
  });
  it("restores the original alpha at 1.0 (uses orig, not cumulative)", () => {
    const im = make();
    im.set_opacity(0.5);
    im.set_opacity(1.0);
    expect(nums(im.pixel_array)).toEqual([10, 20, 30, 200, 40, 50, 60, 100]);
  });
});

describe("AbstractImageMobject.get_pixel_array", () => {
  it("raises NotImplementedError", () => {
    const m = new AbstractImageMobject({ scale_to_resolution: false });
    let thrown: unknown;
    try {
      m.get_pixel_array();
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("NotImplementedError");
  });
});
