// Regression tests for the ndarray API alignment: manim public methods that
// return `np.ndarray` must return a numpy-ts `NDArray` here, not a plain JS
// array.  Plain arrays silently change `==`, `+`, `*`, `.sum()` and `.all()`
// semantics in translated Python code (e.g.
// `(path.get_start() == path.get_end()).all()`), so the container type is part
// of the public contract.
import { describe, expect, it } from "vitest";
import { array, equal, multiply, NDArray } from "numpy-ts";
import { Mobject } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Line, Arrow } from "../src/mobject/geometry/line.js";
import { Arc } from "../src/mobject/geometry/arc.js";
import { Square, Polygon } from "../src/mobject/geometry/polygram.js";
import { Brace } from "../src/mobject/svg/brace.js";
import { Cone } from "../src/mobject/three_d/three_dimensions.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { Axes } from "../src/mobject/graphing/coordinate_systems.js";
import { Wiggle } from "../src/animation/indication.js";
import {
  angle_of_vector,
  cartesian_to_spherical,
  complex_to_R3,
  cross,
  find_intersection,
  line_intersection,
  normalize,
} from "../src/utils/space_ops.js";
import { bezier_remap, partial_bezier_points } from "../src/utils/bezier.js";

/** Assert the value is an NDArray (not a plain array) and return it typed. */
function expectNDArray(value: unknown): NDArray {
  expect(Array.isArray(value)).toBe(false);
  expect(value).toBeInstanceOf(NDArray);
  expect(typeof (value as NDArray).toArray).toBe("function");
  return value as NDArray;
}

function squareVMobject(): VMobject {
  const v = new VMobject();
  v.set_points_as_corners([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
  ]);
  return v;
}

describe("Mobject core getters return NDArray", () => {
  it("critical-point family", () => {
    const mob = new Mobject().set_points([
      [0, 0, 0],
      [2, 4, 6],
    ]);
    expectNDArray(mob.get_center());
    expectNDArray(mob.get_center_of_mass());
    expectNDArray(mob.get_critical_point([1, 0, 0]));
    expectNDArray(mob.get_left());
    expectNDArray(mob.get_right());
    expectNDArray(mob.get_top());
    expectNDArray(mob.get_bottom());
    expectNDArray(mob.get_center_point());
    expectNDArray(mob.get_zenith());
    expectNDArray(mob.get_nadir());
    expectNDArray(mob.get_edge_center([1, 0, 0]));
    expectNDArray(mob.get_corner([1, 1, 0]));
    expectNDArray(mob.get_boundary_point([1, 1, 0]));
    expectNDArray(mob.get_z_index_reference_point());
  });

  it("start/end and start_and_end", () => {
    const mob = new Mobject().set_points([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expectNDArray(mob.get_start()).toArray();
    expect(mob.get_start().toArray()).toEqual([1, 2, 3]);
    expect(mob.get_end().toArray()).toEqual([4, 5, 6]);
    const [start, end] = mob.get_start_and_end();
    expectNDArray(start);
    expectNDArray(end);
  });

  it("numpy semantics work on the returned points", () => {
    const mob = new Mobject().set_points([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const v = mob.get_start();
    expect((multiply(v, 2) as NDArray).toArray()).toEqual([2, 4, 6]);
    // The exact shape of the original crash: elementwise `==` then `.all()`.
    expect((equal(mob.get_start(), mob.get_end()) as NDArray).all()).toBe(false);
    expect((equal(mob.get_start(), mob.get_start()) as NDArray).all()).toBe(true);
  });
});

describe("VMobject curve queries return NDArray", () => {
  it("anchors, handles and bezier tuples", () => {
    const v = squareVMobject();
    const startAnchors = expectNDArray(v.get_start_anchors());
    expect(startAnchors.shape.length).toBe(2);
    const endAnchors = expectNDArray(v.get_end_anchors());
    expect(endAnchors.shape.length).toBe(2);

    const anchors = v.get_anchors();
    expect(Array.isArray(anchors)).toBe(true);
    for (const anchor of anchors) expectNDArray(anchor);

    const [a1, h1, h2, a2] = v.get_anchors_and_handles();
    for (const part of [a1, h1, h2, a2]) {
      expectNDArray(part);
      expect((part as NDArray).shape.length).toBe(2);
    }

    const tuples = expectNDArray(v.get_cubic_bezier_tuples());
    expect(tuples.shape.length).toBe(3);
    expect(tuples.shape[0]).toBe(v.get_num_curves());
  });

  it("gradient and sheen vectors", () => {
    const v = squareVMobject();
    v.set_sheen(0.2, [1, 0, 0]);
    expectNDArray(v.get_sheen_direction());
    const [start, end] = v.get_gradient_start_and_end_points();
    expectNDArray(start);
    expectNDArray(end);
  });
});

describe("geometry getters return NDArray", () => {
  it("Line / Arrow", () => {
    const line = new Line([0, 0, 0], [3, 4, 0]);
    expect(expectNDArray(line.get_vector()).toArray()).toEqual([3, 4, 0]);
    expect(expectNDArray(line.get_unit_vector()).toArray()).toEqual([0.6, 0.8, 0]);
    expect(expectNDArray(line.get_projection([0, 5, 0])).toArray()).toEqual([2.4, 3.2, 0]);
    expectNDArray(new Arrow([0, 0, 0], [1, 1, 0]).get_normal_vector());
  });

  it("Arc.get_arc_center", () => {
    const arc = new Arc(2, 0, Math.PI / 2);
    expectNDArray(arc.get_arc_center());
  });

  it("Polygram vertices / vertex groups", () => {
    const sq = new Square();
    expect(expectNDArray(sq.get_vertices()).shape.length).toBe(2);
    const groups = sq.get_vertex_groups();
    expect(Array.isArray(groups)).toBe(true);
    for (const group of groups) expectNDArray(group);

    const poly = new Polygon([0, 0, 0], [1, 0, 0], [1, 1, 0]);
    expect(expectNDArray(poly.get_vertices()).shape[0]).toBe(3);
  });

  it("Brace.get_tip", () => {
    expectNDArray(new Brace(new Square()).get_tip());
  });

  it("Cone.get_start / get_end", () => {
    const cone = new Cone();
    expectNDArray(cone.get_start());
    expectNDArray(cone.get_end());
  });
});

describe("graphing getters return NDArray", () => {
  it("NumberLine", () => {
    const nl = new NumberLine();
    expectNDArray(nl.get_start());
    expectNDArray(nl.get_end());
    expectNDArray(nl.get_unit_vector());
    expectNDArray(nl.get_projection([0, 1, 0]));
    const [start, end] = nl.get_start_and_end();
    expectNDArray(start);
    expectNDArray(end);
  });

  it("CoordinateSystem.p2c / point_to_coords / Axes.point_to_coords", () => {
    const axes = new Axes();
    expectNDArray(axes.p2c([1, 1, 0]));
    expectNDArray(axes.point_to_coords([1, 1, 0]));
    const rows = expectNDArray(
      axes.point_to_coords([[0, 0, 0], [1, 0, 0]] as unknown as readonly number[]),
    );
    expect(rows.shape.length).toBe(2);
  });
});

describe("utils and animation return NDArray", () => {
  it("space_ops", () => {
    expect(expectNDArray(cross([1, 0, 0], [0, 1, 0])).toArray()).toEqual([0, 0, 1]);
    expect(expectNDArray(normalize([3, 4, 0])).toArray()).toEqual([0.6, 0.8, 0]);
    expectNDArray(cartesian_to_spherical([1, 1, 1]));
    expect(expectNDArray(complex_to_R3({ re: 3, im: 4 })).toArray()).toEqual([3, 4, 0]);
    expectNDArray(
      line_intersection([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]]),
    );
    expectNDArray(angle_of_vector(array([[1, 0], [0, 1]])));
  });

  it("find_intersection is a superset of manimce (flat point / batched list)", () => {
    // manimgl-style single line: one NDArray point.
    expectNDArray(find_intersection([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]));
    // manimce batched form: list of NDArray points.
    const batch = find_intersection(
      [[0, 0, 0]],
      [[1, 0, 0]],
      [[0, 1, 0]],
      [[0, -1, 0]],
    );
    expect(Array.isArray(batch)).toBe(true);
    for (const point of batch as NDArray[]) expectNDArray(point);
  });

  it("bezier helpers", () => {
    const curve = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    expect(expectNDArray(partial_bezier_points(curve, 0, 1)).shape.length).toBe(2);
    const remapped = expectNDArray(bezier_remap([curve], 3));
    expect(remapped.shape.length).toBe(3);
    expect(remapped.shape[0]).toBe(3);
  });

  it("Wiggle about-points", () => {
    const w = new Wiggle(new Square());
    expectNDArray(w.get_scale_about_point());
    expectNDArray(w.get_rotate_about_point());
  });
});
