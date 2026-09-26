import { describe, expect, it, vi } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import {
  Axes,
  ComplexPlane,
  NumberPlane,
  PolarPlane,
  ThreeDAxes,
} from "../src/mobject/graphing/coordinate_systems.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import {
  ConvexHull3D,
  Dodecahedron,
  Icosahedron,
  Octahedron,
  Tetrahedron,
} from "../src/mobject/three_d/polyhedra.js";
import {
  LinearTransformationScene,
  VectorScene,
} from "../src/scene/vector_space_scene.js";
import { SpecialThreeDScene } from "../src/scene/three_d_scene.js";

// Ground truths captured with original manim
// (manim 0.21.0). Representative citations:
// - Axes(): x_range [-7,7,1], y_range [-4,4,1], x_length 12, y_length 6;
//   c2p(0,0)=[0,0,0], c2p(1,2)=[0.85714286,1.5,0],
//   c2p([0,1,1],[1,1,0])=[[~0,.857,.857],[.75,.75,0],[0,0,0]] (transposed).
// - Axes([0,5,1],[-2,2,0.5]).coords_to_point(2,0) == [-1.2,0,0].
// - NumberPlane(): x_range (-7.1111,7.1111,1), y_range (-4,4,1),
//   x_length None; bg style {BLUE_D w2 op1}, faded {BLUE_D w1.0 op0.5};
//   c2p(1,1)=[1,1,0]; bg/faded submobject counts 22/0.
// - PolarPlane(): x_range [-4,4,1], size None, azimuth_step 20,
//   make_smooth False; _get_lines counts [25,0];
//   PolarPlane(3,4).x_range == [-3,3,1], x_length 4.
// - ComplexPlane().n2p(1+2j)==[1,2,0], n2p(3)==[3,0,0].
// - ThreeDAxes(): x(-6,6,1) y(-5,5,1) z(-4,4,1), lens 10.5/10.5/6.5,
//   c2p(1,2,3)==[0.875,2.1,2.4375].
// - Tetra/Octa/Icosa/Dodeca(): nverts 4/6/12/20, nfaces 4/8/20/12,
//   nedges 12/24/60/60; tetra v0 [.353553]*3, octa v0 [.707107,0,0],
//   icosa v0 [0,.5,.809017], dodeca v0 [.809017]*3; tetra(2) v0 [.707107]*3.
// - ConvexHull3D(tetra pts): 4 verts, 4 faces.
// - BarChart([1,2,3]): x_range [0,3,1], y_range [0,3,0.75],
//   x_length 3, y_length 4, 3 bars, heights ratio 1:2:3.
// - plot_line_graph([0,1,2],[0,1,4]): keys ['line_graph','vertex_dots'],
//   3 dots, line start c2p(0,0), end c2p(2,4)==[1.714,3,0].
// - LinearTransformationScene.setup(): bg plane + plane + 2 basis vectors;
//   get_unit_square(): w/h 1.0, center [0.5,0.5,0].
// - VectorScene.add_axes()->Axes, add_plane()->NumberPlane.

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function toNums(value: number[] | NDArray): number[] {
  if (Array.isArray(value)) return [...value].map(Number);
  return [...(value.toArray() as unknown as number[])].map(Number);
}

function vecClose(actual: number[] | NDArray, expected: number[], tol = 1e-6): void {
  const arr = toNums(actual);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

// ---------------------------------------------------------------------------
// Axes.__init__
// ---------------------------------------------------------------------------

describe("Axes.__init__ (python ground truth)", () => {
  it("defaults round frame radii and resolve lengths", () => {
    const ax = new Axes();
    expect(ax.x_range).toEqual([-7, 7, 1]);
    expect(ax.y_range).toEqual([-4, 4, 1]);
    expect(ax.x_length).toBe(12);
    expect(ax.y_length).toBe(6);
  });

  it("merges axis configs with y rotation + origin-tick exclusion", () => {
    const ax = new Axes();
    expect(ax.axis_config["include_tip"]).toBe(true);
    expect(ax.axis_config["numbers_to_exclude"]).toEqual([0]);
    expect(ax.x_axis_config["exclude_origin_tick"]).toBe(true);
    expect(ax.y_axis_config["exclude_origin_tick"]).toBe(true);
  });

  it("honors custom ranges (python [0,5,1]/[-2,2,0.5])", () => {
    const ax = new Axes([0, 5, 1], [-2, 2, 0.5]);
    expect(ax.x_range).toEqual([0, 5, 1]);
    expect(ax.y_range).toEqual([-2, 2, 0.5]);
    expect(ax.x_length).toBe(12);
  });
});

// ---------------------------------------------------------------------------
// Axes.coords_to_point
// ---------------------------------------------------------------------------

describe("Axes.coords_to_point (python ground truth)", () => {
  it("maps origin and (1,2) through the linear map", () => {
    const ax = new Axes();
    vecClose(ax.coords_to_point(0, 0), [0, 0, 0]);
    vecClose(ax.coords_to_point(1, 2), [0.85714286, 1.5, 0]);
  });

  it("transposes separate per-axis array inputs (python .T)", () => {
    const ax = new Axes();
    const got = toNums(ax.coords_to_point([0, 1, 1], [1, 1, 0]));
    // shape (3,3): rows are x-row, y-row, z-row of the point list.
    expect(got.length).toBe(3);
    const flat = (ax.coords_to_point([0, 1, 1], [1, 1, 0]).toArray() as number[][]).map(
      (row) => row.map(Number),
    );
    expect(close(flat[0]?.[1] ?? NaN, 0.85714286, 1e-6)).toBe(true);
    expect(close(flat[1]?.[0] ?? NaN, 0.75, 1e-6)).toBe(true);
  });

  it("maps (2,0) to [-1.2,0,0] on custom ranges", () => {
    const ax = new Axes([0, 5, 1], [-2, 2, 0.5]);
    vecClose(ax.coords_to_point(2, 0), [-1.2, 0, 0]);
  });

  it("round-trips through point_to_coords", () => {
    const ax = new Axes();
    const back = ax.point_to_coords(toNums(ax.coords_to_point(1, 2)));
    expect(close(back[0] ?? NaN, 1, 1e-9)).toBe(true);
    expect(close(back[1] ?? NaN, 2, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Axes.plot_line_graph
// ---------------------------------------------------------------------------

describe("Axes.plot_line_graph (python ground truth)", () => {
  it("builds line_graph + 3 vertex dots at c2p positions", () => {
    const ax = new Axes();
    const lg = ax.plot_line_graph([0, 1, 2], [0, 1, 4]);
    expect([...lg.submob_dict.keys()]).toEqual(["line_graph", "vertex_dots"]);
    const graph = lg.__getitem__("line_graph");
    vecClose(graph.get_start(), toNums(ax.coords_to_point(0, 0)));
    vecClose(graph.get_end(), [1.71428571, 3, 0], 1e-6);
    expect(lg.__getitem__("vertex_dots").submobjects.length).toBe(3);
  });

  it("omits vertex_dots when add_vertex_dots=false", () => {
    const ax = new Axes();
    const lg = ax.plot_line_graph([0, 1], [0, 1], null, "PURE_YELLOW", false);
    expect([...lg.submob_dict.keys()]).toEqual(["line_graph"]);
  });

  it("raises ValueError on length mismatch (python strict zip)", () => {
    const ax = new Axes();
    try {
      ax.plot_line_graph([0, 1], [0, 1, 2]);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });
});

// ---------------------------------------------------------------------------
// NumberPlane.__init__ / _init_background_lines / _get_lines / get_vector
// ---------------------------------------------------------------------------

describe("NumberPlane.__init__ (python ground truth)", () => {
  it("keeps raw frame radii with null lengths", () => {
    const pl = new NumberPlane();
    expect(close(pl.x_range[0] ?? NaN, -7.111111111111111, 1e-9)).toBe(true);
    expect(close(pl.x_range[1] ?? NaN, 7.111111111111111, 1e-9)).toBe(true);
    expect(pl.y_range).toEqual([-4, 4, 1]);
    expect(pl.x_length).toBeNull();
    expect(pl.y_length).toBeNull();
  });

  it("derives faded style by halving numerics (w2->1.0, op1->0.5)", () => {
    const pl = new NumberPlane();
    expect(pl.background_line_style["stroke_width"]).toBe(2);
    expect(pl.faded_line_style?.["stroke_width"]).toBe(1.0);
    expect(pl.faded_line_style?.["stroke_opacity"]).toBe(0.5);
    expect(pl.faded_line_ratio).toBe(1);
  });

  it("maps coords at unit size about the midpoint (python c2p(1,1)=[1,1,0])", () => {
    const pl = new NumberPlane();
    vecClose(pl.coords_to_point(1, 1), [1, 1, 0]);
    vecClose(pl.coords_to_point(0, 0), [0, 0, 0]);
  });
});

describe("NumberPlane._init_background_lines / _get_lines", () => {
  it("_init_background_lines populates bg + faded groups", () => {
    const pl = new NumberPlane();
    pl._init_background_lines();
    expect(pl.background_lines).toBeInstanceOf(VGroup);
    expect(pl.faded_lines).toBeInstanceOf(VGroup);
  });

  it("_get_lines returns [background, faded] VGroups (python: 22 + 0)", () => {
    const pl = new NumberPlane();
    const [bg, faded] = pl._get_lines();
    expect(bg).toBeInstanceOf(VGroup);
    expect(faded).toBeInstanceOf(VGroup);
    expect(bg.submobjects.length).toBe(22);
    expect(faded.submobjects.length).toBe(0);
  });
});

describe("NumberPlane.get_vector (python ground truth)", () => {
  it("spans origin to coords_to_point with buff 0", () => {
    const pl = new NumberPlane();
    const v = pl.get_vector([1, 2]);
    vecClose(v.get_start(), [0, 0, 0]);
    vecClose(v.get_end(), [1, 2, 0]);
  });
});

// ---------------------------------------------------------------------------
// PolarPlane.__init__ / _init_background_lines / get_vector
// ---------------------------------------------------------------------------

describe("PolarPlane.__init__ (python ground truth)", () => {
  it("defaults to radius_max=frame_y_radius with unit mapping", () => {
    const pp = new PolarPlane();
    expect(pp.x_range).toEqual([-4, 4, 1]);
    expect(pp.y_range).toEqual([-4, 4, 1]);
    expect(pp.x_length).toBeNull();
    expect(pp.azimuth_units).toBe("PI radians");
    expect(pp.azimuth_direction).toBe("CCW");
    expect(pp.azimuth_step).toBe(20);
    // NOTE: mirrors python, where VGroup.__init__ overwrites the True passed
    // to PolarPlane.__init__ with the VMobject default False.
    expect(pp.make_smooth_after_applying_functions).toBe(false);
  });

  it("honors radius_max/size (python PolarPlane(3,4): [-3,3,1], len 4)", () => {
    const pp = new PolarPlane(3, 4);
    expect(pp.x_range).toEqual([-3, 3, 1]);
    expect(pp.y_range).toEqual([-3, 3, 1]);
    expect(pp.x_length).toBe(4);
    expect(pp.y_length).toBe(4);
  });

  it("rejects bad azimuth units/direction like python", () => {
    expect(() => new PolarPlane(null, null, 1, null, "radians" as never)).toThrow(/Invalid azimuth units/);
    expect(() => new PolarPlane(null, null, 1, null, "PI radians", true, 0, "sideways" as never)).toThrow(
      /Invalid azimuth/,
    );
  });
});

describe("PolarPlane._init_background_lines / get_vector", () => {
  it("_init_background_lines derives faded style and stores groups", () => {
    const pp = new PolarPlane();
    pp._init_background_lines();
    expect(pp.background_lines).toBeInstanceOf(VGroup);
    expect(pp.faded_lines).toBeInstanceOf(VGroup);
    expect(pp.faded_line_style?.["stroke_width"]).toBe(1.0);
    expect(pp.faded_line_style?.["stroke_opacity"]).toBe(0.5);
    const [bg, faded] = pp._get_lines();
    expect(bg.submobjects.length).toBe(25);
    expect(faded.submobjects.length).toBe(0);
  });

  it("get_vector spans origin to coords_to_point", () => {
    const pp = new PolarPlane();
    const v = pp.get_vector([1, 2]);
    vecClose(v.get_start(), [0, 0, 0]);
    vecClose(v.get_end(), [1, 2, 0]);
  });
});

// ---------------------------------------------------------------------------
// ComplexPlane.__init__ / number_to_point / n2p
// ---------------------------------------------------------------------------

describe("ComplexPlane.__init__ / number_to_point / n2p", () => {
  it("inherits NumberPlane ranges (python raw frame radii)", () => {
    const cp = new ComplexPlane();
    expect(close(cp.x_range[0] ?? NaN, -7.111111111111111, 1e-9)).toBe(true);
    expect(cp.y_range).toEqual([-4, 4, 1]);
  });

  it("number_to_point maps re/im through c2p (python n2p(1+2j)=[1,2,0])", () => {
    const cp = new ComplexPlane();
    vecClose(cp.number_to_point({ re: 1, im: 2 }), [1, 2, 0]);
    vecClose(cp.number_to_point(3), [3, 0, 0]);
  });

  it("n2p aliases number_to_point incl tuple and string forms", () => {
    const cp = new ComplexPlane();
    vecClose(cp.n2p({ re: 1, im: 2 }), [1, 2, 0]);
    vecClose(cp.n2p([1, 2]), [1, 2, 0]);
    vecClose(cp.n2p("1+2j"), [1, 2, 0]);
    vecClose(cp.n2p(3), [3, 0, 0]);
  });

  it("round-trips through point_to_number", () => {
    const cp = new ComplexPlane();
    const back = cp.point_to_number(toNums(cp.n2p({ re: 1, im: 2 })));
    expect(close(back.re, 1, 1e-9)).toBe(true);
    expect(close(back.im, 2, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// ThreeDAxes.__init__ (+ z mapping fix)
// ---------------------------------------------------------------------------

describe("ThreeDAxes.__init__ (python ground truth)", () => {
  it("defaults ranges (-6,6)/(-5,5)/(-4,4) and lengths 10.5/10.5/6.5", () => {
    const td = new ThreeDAxes();
    expect(td.x_range).toEqual([-6, 6, 1]);
    expect(td.y_range).toEqual([-5, 5, 1]);
    expect(td.z_range).toEqual([-4, 4, 1]);
    expect(td.x_length).toBe(10.5);
    expect(td.y_length).toBe(10.5);
    expect(td.z_length).toBe(6.5);
    expect(td.dimension).toBe(3);
  });

  it("coords_to_point maps z (python c2p(1,2,3)=[0.875,2.1,2.4375])", () => {
    const td = new ThreeDAxes();
    vecClose(td.coords_to_point(1, 2, 3), [0.875, 2.1, 2.4375]);
  });

  it("point_to_coords inverts all three axes", () => {
    const td = new ThreeDAxes();
    const back = td.point_to_coords([0.875, 2.1, 2.4375]);
    expect(back.length).toBe(3);
    expect(close(back[0] ?? NaN, 1, 1e-9)).toBe(true);
    expect(close(back[1] ?? NaN, 2, 1e-9)).toBe(true);
    expect(close(back[2] ?? NaN, 3, 1e-9)).toBe(true);
  });

  it("get_z_unit_size is z_length/span (python 6.5/8)", () => {
    expect(close(new ThreeDAxes().get_z_unit_size(), 0.8125, 1e-12)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Polyhedra __init__
// ---------------------------------------------------------------------------

describe("Platonic solids __init__ (python ground truth)", () => {
  it("Tetrahedron: 4 verts/faces, 12 edges, unit=(e*sqrt2)/4", () => {
    const t = new Tetrahedron(1);
    expect(t.vertex_coords.length).toBe(4);
    expect(t.faces_list.length).toBe(4);
    expect(t.edges.length).toBe(12);
    expect(t.faces_list[0]).toEqual([0, 1, 2]);
    t.vertex_coords[0]?.forEach((v) => expect(close(v, 0.353553, 1e-5)).toBe(true));
  });

  it("Tetrahedron scales with edge_length (python e=2 -> .707107)", () => {
    const t = new Tetrahedron(2);
    t.vertex_coords[0]?.forEach((v) => expect(close(v, 0.707107, 1e-5)).toBe(true));
  });

  it("Octahedron: 6 verts, 8 faces, 24 edges", () => {
    const o = new Octahedron(1);
    expect(o.vertex_coords.length).toBe(6);
    expect(o.faces_list.length).toBe(8);
    expect(o.edges.length).toBe(24);
    expect(o.faces_list[0]).toEqual([2, 4, 1]);
    vecClose(o.vertex_coords[0] ?? [], [0.707107, 0, 0], 1e-5);
  });

  it("Icosahedron: 12 verts, 20 faces, 60 edges", () => {
    const m = new Icosahedron(1);
    expect(m.vertex_coords.length).toBe(12);
    expect(m.faces_list.length).toBe(20);
    expect(m.edges.length).toBe(60);
    expect(m.faces_list[0]).toEqual([1, 8, 0]);
    vecClose(m.vertex_coords[0] ?? [], [0, 0.5, 0.809017], 1e-5);
  });

  it("Dodecahedron: 20 verts, 12 faces, 60 edges", () => {
    const d = new Dodecahedron(1);
    expect(d.vertex_coords.length).toBe(20);
    expect(d.faces_list.length).toBe(12);
    expect(d.edges.length).toBe(60);
    expect(d.faces_list[0]).toEqual([18, 16, 0, 12, 1]);
    (d.vertex_coords[0] ?? []).forEach((v) => expect(close(v, 0.809017, 1e-5)).toBe(true));
  });
});

describe("ConvexHull3D.__init__ (python ground truth)", () => {
  it("tetra points give 4 verts and 4 faces", () => {
    const h = new ConvexHull3D([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]);
    expect(h.vertex_coords.length).toBe(4);
    expect(h.faces_list.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// BarChart.__init__
// ---------------------------------------------------------------------------

describe("BarChart.__init__ (python ground truth)", () => {
  // The y-axis numbers now build real DecimalNumbers (MathJax-rendered).
  // Stub the number mobject where the test only cares about the ported
  // config/geometry arithmetic (keeps these cases fast).
  function withStubbedNumbers<T>(fn: () => T): T {
    const spy = vi
      .spyOn(NumberLine.prototype, "get_number_mobject")
      .mockImplementation(() => new Square() as unknown as VMobject);
    try {
      return fn();
    } finally {
      spy.mockRestore();
    }
  }

  it("derives x_range/y_range/lengths (python [0,3,1]/[0,3,0.75], 3/4)", () => {
    withStubbedNumbers(() => {
      const bc = new BarChart([1, 2, 3]);
      expect(bc.values).toEqual([1, 2, 3]);
      expect(bc.x_range).toEqual([0, 3, 1]);
      expect(bc.y_range).toEqual([0, 3, 0.75]);
      expect(bc.x_length).toBe(3);
      expect(bc.y_length).toBe(4);
      expect(bc.bars.submobjects.length).toBe(3);
    });
  });

  it("bar heights scale with values (python ratio 1:2:3)", () => {
    withStubbedNumbers(() => {
      const bc = new BarChart([1, 2, 3]);
      const bars = bc.bars.submobjects as Array<{ height: number }>;
      expect(close(bars[1]?.height ?? NaN, 2 * (bars[0]?.height ?? NaN), 1e-9)).toBe(true);
      expect(close(bars[2]?.height ?? NaN, 3 * (bars[0]?.height ?? NaN), 1e-9)).toBe(true);
    });
  });

  it("construction builds the real y-axis numbers (python: 4 labels)", () => {
    const bc = new BarChart([1, 2, 3]);
    expect(bc.y_axis.numbers).toBeInstanceOf(VGroup);
    expect(bc.y_axis.numbers!.submobjects.length).toBe(4);
  });

  it("bar_names need Tex labels (text/tex out of scope)", () => {
    expect(() => new BarChart([1, 2], { bar_names: ["a", "b"] })).toThrow(
      /text\/tex not ported/,
    );
  });

  it("explicit two-element y_range gains a computed step", () => {
    withStubbedNumbers(() => {
      const bc = new BarChart([1, 2, 3], { y_range: [0, 5] });
      expect(bc.y_range[0]).toBe(0);
      expect(bc.y_range[1]).toBe(5);
      expect(close(bc.y_range[2] ?? NaN, 0.75, 1e-9)).toBe(true);
    });
  });
});

// ---------------------------------------------------------------------------
// VectorScene / LinearTransformationScene / SpecialThreeDScene
// ---------------------------------------------------------------------------

describe("VectorScene.add_axes / add_plane (python ground truth)", () => {
  it("add_axes returns default Axes", () => {
    const vs = new VectorScene();
    const axes = vs.add_axes();
    expect(axes).toBeInstanceOf(Axes);
    expect(axes.x_range).toEqual([-7, 7, 1]);
  });

  it("add_plane returns a NumberPlane", () => {
    const vs = new VectorScene();
    const plane = vs.add_plane();
    expect(plane).toBeInstanceOf(NumberPlane);
  });

  it("lock_in_faded_grid fades the plane/axes and reseats the camera", () => {
    const vs = new VectorScene();
    const calls: number[] = [];
    (vs as unknown as { renderer: { update_frame: () => void; camera: unknown } }).renderer = {
      update_frame: () => calls.push(1),
      camera: null,
    };
    expect(() => vs.lock_in_faded_grid()).not.toThrow();
    expect(calls.length).toBe(1);
  });
});

describe("LinearTransformationScene.setup / get_unit_square / add_unit_square", () => {
  it("setup builds bg plane, fg plane and 2 basis vectors", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    expect(scene.background_plane).toBeInstanceOf(NumberPlane);
    expect(scene.plane).toBeInstanceOf(NumberPlane);
    expect(scene.basis_vectors.submobjects.length).toBe(2);
    // second setup call is a no-op (python has_already_setup guard).
    scene.setup();
    expect(scene.basis_vectors.submobjects.length).toBe(2);
  });

  it("get_unit_square is 1x1 centered at [0.5,0.5,0] (python)", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const sq = scene.get_unit_square();
    expect(close(sq.width, 1, 1e-9)).toBe(true);
    expect(close(sq.height, 1, 1e-9)).toBe(true);
    vecClose(sq.get_center(), [0.5, 0.5, 0]);
  });

  it("add_unit_square registers the square as transformable", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const ret = scene.add_unit_square();
    expect(ret).toBe(scene);
    expect(scene.transformable_mobjects.includes(scene.square as never)).toBe(true);
  });
});

describe("SpecialThreeDScene.get_axes (python ground truth)", () => {
  it("returns ThreeDAxes and keeps cut/config state", () => {
    const scene = new SpecialThreeDScene();
    expect(scene.cut_axes_at_radius).toBe(true);
    const axes = scene.get_axes();
    expect(axes).toBeInstanceOf(ThreeDAxes);
    expect(axes.dimension).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// integration with earlier-ported code
// ---------------------------------------------------------------------------

describe("integration (this batch x earlier batches)", () => {
  it("plot_line_graph vertices sit exactly on c2p (axes x VMobject)", () => {
    const ax = new Axes();
    const lg = ax.plot_line_graph([0, 1, 2], [0, 1, 4]);
    const dots = lg.__getitem__("vertex_dots").submobjects as VMobject[];
    const expected = [toNums(ax.coords_to_point(0, 0)), toNums(ax.coords_to_point(1, 1)), toNums(ax.coords_to_point(2, 4))];
    dots.forEach((dot, i) => vecClose(dot.get_center(), expected[i] ?? []));
  });

  it("BarChart bar height equals the axes y-span (probability x coords)", () => {
    const spy = vi
      .spyOn(NumberLine.prototype, "get_number_mobject")
      .mockImplementation(() => new Square() as unknown as VMobject);
    try {
      const bc = new BarChart([1, 2, 3]);
      const yOf = (v: number): number => toNums(bc.coords_to_point(0, v))[1] ?? 0;
      const bars = bc.bars.submobjects as Array<{ height: number }>;
      expect(close(bars[0]?.height ?? NaN, Math.abs(yOf(1) - yOf(0)), 1e-9)).toBe(true);
    } finally {
      spy.mockRestore();
    }
  });

  it("ComplexPlane n2p/p2n round-trips through NumberPlane mapping", () => {
    const cp = new ComplexPlane();
    for (const z of [{ re: 2.5, im: -1.5 }, { re: 0, im: 3 }]) {
      const back = cp.point_to_number(toNums(cp.n2p(z)));
      expect(close(back.re, z.re, 1e-9)).toBe(true);
      expect(close(back.im, z.im, 1e-9)).toBe(true);
    }
  });

  it("VectorScene.get_vector lands on the plane mapping (scene x coords)", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const arrow = scene.get_vector([2, -1]);
    vecClose(arrow.get_end(), toNums(scene.plane.coords_to_point(2, -1)));
  });

  it("ThreeDAxes z column feeds plot_surface-style c2p (coords x surface)", () => {
    const td = new ThreeDAxes();
    // c2p(u, v, f(u,v)) keeps the surface height: z out == axisPosition(f).
    const out = toNums(td.coords_to_point(1, 2, 3));
    expect(close(out[2] ?? NaN, 2.4375, 1e-9)).toBe(true);
    expect(td.get_z_unit_size()).toBeGreaterThan(0);
  });

  it("setup + add_unit_square + apply_matrix end-to-end with mocked play", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.add_unit_square();
    const spy = vi.spyOn(scene, "play").mockImplementation(() => undefined as never);
    scene.transformable_mobjects = [scene.square as never];
    scene.apply_matrix([[1, 0], [0, 1]]);
    expect(spy).toHaveBeenCalledOnce();
    spy.mockRestore();
  });
});
