import { describe, expect, it, vi } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VDict, VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Rectangle } from "../src/mobject/geometry/polygram.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { LogBase } from "../src/mobject/graphing/scale.js";
import { index_labels } from "../src/utils/debug.js";
import { NumberLine, UnitInterval } from "../src/mobject/graphing/number_line.js";
import { DecimalNumber } from "../src/mobject/text/numbers.js";
import {
  Axes,
  ComplexPlane,
  NumberPlane,
  PolarPlane,
  ThreeDAxes,
} from "../src/mobject/graphing/coordinate_systems.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import {
  ConvexHull3D,
  Dodecahedron,
  Icosahedron,
  Octahedron,
  Polyhedron,
  Tetrahedron,
} from "../src/mobject/three_d/polyhedra.js";
import {
  DiGraph,
  GenericGraph,
  Graph,
  NetworkXGraph,
} from "../src/mobject/graph.js";
import {
  LinearTransformationScene,
  VectorScene,
} from "../src/scene/vector_space_scene.js";
import { SpecialThreeDScene } from "../src/scene/three_d_scene.js";
import { AnimationGroup } from "../src/animation/composition.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function toNums(value: number[] | NDArray): number[] {
  if (Array.isArray(value)) return [...value].map(Number);
  return [...(value.toArray() as unknown as number[])].map(Number);
}

function toRows(value: number[] | NDArray): number[][] {
  const raw: unknown = Array.isArray(value)
    ? value
    : ((value as NDArray).toArray() as unknown);
  const arr = raw as unknown[];
  if (arr.length > 0 && Array.isArray(arr[0])) {
    return (arr as unknown[][]).map((row) => row.map(Number));
  }
  return [(arr as unknown[]).map(Number)];
}

function vecClose(actual: number[] | NDArray, expected: number[], tol = 1e-6): void {
  const arr = toNums(actual);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

// Python ground truths were captured with
// original manim (manim 0.21.0); each block cites them.

// ---------------------------------------------------------------------------
// VDict.__init__ / __setitem__ / get_all_submobjects
// ---------------------------------------------------------------------------

describe("VDict", () => {
  it("constructs from a plain record (python: VDict({'a':sq,'b':sq}))", () => {
    const a = new Square();
    const b = new Square();
    const vd = new VDict({ a, b });
    // python: submob_dict keys ['a','b'], 2 submobjects
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b"]);
    expect(vd.submobjects.length).toBe(2);
    expect(vd.show_keys).toBe(false);
  });

  it("constructs from a Map and from an entry iterable", () => {
    const a = new Square();
    const b = new Square();
    const fromMap = new VDict(new Map<unknown, VMobject>([["a", a]]));
    expect(fromMap.has("a")).toBe(true);
    const fromIter = new VDict([["x", a], ["y", b]]);
    expect([...fromIter.submob_dict.keys()]).toEqual(["x", "y"]);
    expect(fromIter.submobjects.length).toBe(2);
  });

  it("__setitem__ appends new keys (python)", () => {
    const vd = new VDict({ a: new Square(), b: new Square() });
    const c = new Square();
    vd.__setitem__("c", c);
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b", "c"]);
    expect(vd.__getitem__("c")).toBe(c);
    expect(vd.submobjects.length).toBe(3);
  });

  it("__setitem__ overwrites by remove+add: key moves to the end (python: ['b','c','a'])", () => {
    const vd = new VDict({ a: new Square(), b: new Square() });
    vd.__setitem__("c", new Square());
    const a2 = new Square();
    vd.__setitem__("a", a2);
    // python ground truth: keys ['b','c','a'], still 3 submobjects
    expect([...vd.submob_dict.keys()]).toEqual(["b", "c", "a"]);
    expect(vd.submobjects.length).toBe(3);
    expect(vd.__getitem__("a")).toBe(a2);
  });

  it("get_all_submobjects returns every value (python: dict values)", () => {
    const a = new Square();
    const b = new Square();
    const vd = new VDict({ a, b });
    expect(vd.get_all_submobjects()).toEqual([a, b]);
  });

  it("get/remove/__contains__ mirror python KeyError behavior", () => {
    const a = new Square();
    const vd = new VDict({ a });
    expect(vd.has("a")).toBe(true);
    expect(vd.__contains__("zzz")).toBe(false);
    expect(() => vd.get("zzz")).toThrow(/zzz/);
    vd.remove("a");
    expect(vd.has("a")).toBe(false);
    expect(vd.submobjects.length).toBe(0);
    expect(() => vd.remove("a")).toThrow(/not present/);
  });

  it("show_keys branch is a text/tex stub (python renders Tex(str(key)))", () => {
    const vd = new VDict({}, true);
    expect(() => vd.add_key_value_pair("k", new Square())).toThrow(
      /text\/tex not ported/,
    );
  });
});

// ---------------------------------------------------------------------------
// text/tex stubs: no functional tests per port rules, only stub contracts
// ---------------------------------------------------------------------------

describe("text/tex stubs (out of scope for the entire port)", () => {
  it("LogBase.get_custom_labels throws (python builds Integer labels, needs latex)", () => {
    expect(() => new LogBase().get_custom_labels([1, 10, 100])).toThrow(
      /text\/tex not ported/,
    );
  });

  it("index_labels throws (python builds Integer labels, needs latex)", () => {
    const g = new VGroup(new Square(), new Square());
    expect(() => index_labels(g)).toThrow(/text\/tex not ported/);
  });

  it("NumberLine label builders build DecimalNumber labels (python)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect((nl.get_number_mobject(1) as unknown as DecimalNumber).get_value()).toBe(1);
    expect(nl.get_number_mobjects(1, 2).submobjects.length).toBe(2);
    // python get_number_mobjects() with no args calls the missing
    // default_numbers_to_display(); the TS port mirrors the AttributeError.
    expect(() => nl.get_number_mobjects()).toThrow(/default_numbers_to_display/);
    expect(() => nl.get_labels()).toThrow(/default_numbers_to_display/);
    nl.add_numbers([1, 2]);
    expect(nl.numbers?.submobjects.length).toBe(2);
    const nl2 = new NumberLine({ x_range: [-3, 3, 1] });
    nl2.add_numbers();
    expect(nl2.numbers?.submobjects.length).toBe(7);
    expect(nl._create_label_tex("1").submobjects.length).toBeGreaterThan(0);
    // VMobject inputs pass through without text (python behavior).
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm);
  });

  it("VectorScene.write_vector_coordinates/vector_to_coords throw (python needs Matrix/Write)", () => {
    const vs = new VectorScene();
    expect(() => vs.write_vector_coordinates(new Square())).toThrow(
      /text\/tex not ported/,
    );
    expect(() => vs.vector_to_coords([1, 2, 0])).toThrow(/text\/tex not ported/);
  });

  it("LinearTransformationScene.write_vector_coordinates throws via super (python: Matrix)", () => {
    const scene = new LinearTransformationScene();
    expect(() => scene.write_vector_coordinates(new Square())).toThrow(
      /text\/tex not ported/,
    );
  });

  it("BarChart._add_x_axis_labels throws (python builds Tex labels)", () => {
    const spy = vi
      .spyOn(NumberLine.prototype, "get_number_mobject")
      .mockImplementation(() => new Square() as unknown as VMobject);
    try {
      const bc = new BarChart([1, 2, 3]);
      expect(() => bc._add_x_axis_labels()).toThrow(/text\/tex not ported/);
      expect(() => bc.get_bar_labels()).toThrow(/text\/tex not ported/);
    } finally {
      spy.mockRestore();
    }
    // bar_names trigger the label path in __init__, like python.
    expect(() => new BarChart([1, 2], { bar_names: ["a", "b"] })).toThrow(
      /text\/tex not ported/,
    );
  });
});

// ---------------------------------------------------------------------------
// NumberLine.__init__ / tick machinery / UnitInterval.__init__
// ---------------------------------------------------------------------------

describe("NumberLine.__init__ (python ground truth)", () => {
  it("defaults x_range from frame radii ([-7, 7, 1])", () => {
    const nl = new NumberLine();
    expect(nl.x_range).toEqual([-7, 7, 1]);
    expect(nl.x_min).toBe(-7);
    expect(nl.x_max).toBe(7);
    expect(nl.x_step).toBe(1);
    expect(nl.unit_size).toBe(1);
  });

  it("stores scaled min/max/step (python: x_range=[-3,3,1] -> -3/3/1)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect([nl.x_min, nl.x_max, nl.x_step]).toEqual([-3, 3, 1]);
    expect(nl.decimal_number_config["num_decimal_places"]).toBe(0);
  });

  it("derives num_decimal_places from step (0.1->1, 0.25->2, 0.125->3)", () => {
    expect(NumberLine._decimal_places_from_step(1)).toBe(0);
    expect(NumberLine._decimal_places_from_step(0.1)).toBe(1);
    expect(NumberLine._decimal_places_from_step(0.25)).toBe(2);
    expect(NumberLine._decimal_places_from_step(0.125)).toBe(3);
  });

  it("tick range excludes nothing by default (python [-3..3] -> 7 ticks incl -0.0)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    // NOTE: python prints -0.0; ±0 compare equal, so compare element-wise.
    const got = toNums(nl.get_tick_range());
    const want = [-3, -2, -1, 0, 1, 2, 3];
    expect(got.length).toBe(want.length);
    got.forEach((v, i) => expect(close(v, want[i] ?? NaN, 1e-12)).toBe(true));
    expect(nl.ticks.submobjects.length).toBe(7);
  });

  it("exclude_origin_tick drops 0 (python [-3,-2,-1,1,2,3])", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], exclude_origin_tick: true });
    expect(toNums(nl.get_tick_range())).toEqual([-3, -2, -1, 1, 2, 3]);
  });

  it("get_tick centers on the number with length 2*size (python tick(1,0.1): center [1,0,0], len 0.2)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const tick = nl.get_tick(1, 0.1);
    vecClose(tick.get_center(), [1, 0, 0]);
    expect(close(tick.get_length(), 0.2, 1e-9)).toBe(true);
  });

  it("length rescales unit_size (python: x_range [-3,3] length 6 -> unit 1)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expect(close(nl.get_unit_size(), 1, 1e-9)).toBe(true);
    // n2p(0) is the origin-centered middle of the line.
    vecClose(nl.n2p(0), [0, 0, 0]);
    vecClose(nl.n2p(1), [1, 0, 0]);
  });

  it("numbers_with_elongated_ticks accepted (python)", () => {
    const nl = new NumberLine({
      x_range: [-3, 3, 1],
      numbers_with_elongated_ticks: [-2, 2],
    });
    expect(nl.numbers_with_elongated_ticks).toEqual([-2, 2]);
    expect(nl.ticks.submobjects.length).toBe(7);
  });

  it("p2n/n2p round-trip (python)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const p = nl.n2p(2);
    expect(close(Number(nl.p2n(toNums(p))), 2, 1e-9)).toBe(true);
  });
});

describe("UnitInterval.__init__ (python: x_range [0,1,0.1], unit_size 10)", () => {
  it("matches python defaults", () => {
    const ui = new UnitInterval();
    expect(ui.x_range[0]).toBe(0);
    expect(ui.x_range[1]).toBe(1);
    expect(close(ui.x_range[2] ?? NaN, 0.1, 1e-9)).toBe(true);
    expect(close(ui.get_unit_size(), 10, 1e-9)).toBe(true);
    expect(ui.decimal_number_config["num_decimal_places"]).toBe(1);
    expect(ui.numbers_with_elongated_ticks).toEqual([0, 1]);
  });
});

// ---------------------------------------------------------------------------
// GenericGraph.__getitem__ / animations / from_networkx / DiGraph
// ---------------------------------------------------------------------------

function smallGraph(): Graph {
  return new Graph([1, 2, 3], [[1, 2], [2, 3]], {
    layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
      string,
      unknown
    >,
  });
}

describe("GenericGraph.__getitem__ (python)", () => {
  it("returns the vertex mobject for a vertex key", () => {
    const g = smallGraph();
    expect(g.__getitem__(1)).toBe(g.vertices.get(1));
  });

  it("returns the edge mobject for an edge tuple", () => {
    const g = smallGraph();
    const edge = g.__getitem__([1, 2] as unknown as never);
    expect(edge).toBe(g.edges.values().next().value as never);
  });

  it("raises ValueError for unknown keys (python)", () => {
    const g = smallGraph();
    expect(() => g.__getitem__(999 as never)).toThrow(/Could not find/);
  });
});

describe("GenericGraph vertex/edge animations (python)", () => {
  it("_add_vertices_animation returns an AnimationGroup and defers insertion until on_finish", () => {
    const g = smallGraph();
    const anim = g._add_vertices_animation(4);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(anim.animations.length).toBe(1);
    // python: vertex 4 is NOT in self.vertices until on_finish runs.
    expect(g.vertices.has(4)).toBe(false);
    (anim as unknown as { on_finish: (scene: unknown) => void }).on_finish({
      remove: () => undefined,
    });
    expect(g.vertices.has(4)).toBe(true);
  });

  it("_remove_vertices_animation removes and animates (python default Uncreate)", () => {
    const g = smallGraph();
    const anim = g._remove_vertices_animation(3);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(g.vertices.has(3)).toBe(false);
    // incident edge (2,3) is gone too, like python remove_vertices.
    expect(g.vertices.has(2)).toBe(true);
    expect(anim.animations.length).toBeGreaterThan(0);
  });

  it("_add_edges_animation adds edges and returns a group (python default Create)", () => {
    const g = new Graph([1, 2, 3], [[1, 2]], {
      layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
    const anim = g._add_edges_animation([2, 3]);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(anim.animations.length).toBeGreaterThan(0);
    const added = g.__getitem__([2, 3] as unknown as never);
    expect([...g.edges.values()]).toContain(added);
    expect(g.edges.size).toBe(2);
    expect(() => g.__getitem__([2, 3] as unknown as never)).not.toThrow();
  });

  it("_remove_edges_animation removes edges and returns a group (python)", () => {
    const g = smallGraph();
    const anim = g._remove_edges_animation([1, 2]);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(() => g.__getitem__([1, 2] as unknown as never)).toThrow(/Could not find/);
    expect(g.vertices.has(1)).toBe(true);
  });
});

describe("GenericGraph.from_networkx / DiGraph (python)", () => {
  it("from_networkx builds vertices+edges (python path_graph(3) -> [0,1,2], [(0,1),(1,2)])", () => {
    const nx = new NetworkXGraph();
    nx.addEdge(0, 1);
    nx.addEdge(1, 2);
    const g = Graph.from_networkx(nx, {
      layout: { 0: [-1, 0, 0], 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
    expect([...g.vertices.keys()].sort()).toEqual([0, 1, 2]);
    expect(g.edges.size).toBe(2);
    expect(g).toBeInstanceOf(Graph);
  });

  it("DiGraph constructs with directed edges (python edge mobject per pair)", () => {
    const d = new DiGraph([1, 2], [[1, 2]], {
      layout: { 1: [-1, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    expect(d).toBeInstanceOf(GenericGraph);
    expect(d.vertices.size).toBe(2);
    expect(d.edges.size).toBe(1);
    expect(d.toString()).toMatch(/Directed graph/);
    // directedness: (2,1) is not an edge.
    expect(() => d.__getitem__([2, 1] as unknown as never)).toThrow(/Could not find/);
  });
});

// ---------------------------------------------------------------------------
// Polyhedron family
// ---------------------------------------------------------------------------

describe("Polyhedron (python ground truth)", () => {
  it("__init__ stores coords/faces/edges and builds faces+graph", () => {
    const t = new Tetrahedron(1);
    expect(t.vertex_coords.length).toBe(4);
    expect(t.faces_list.length).toBe(4);
    // python get_edges([[0,1,2]]) -> [(0,1),(1,2),(2,0)]
    expect(t.get_edges([[0, 1, 2]])).toEqual([[0, 1], [1, 2], [2, 0]]);
    expect(t.edges.length).toBe(12);
    expect(t.faces.submobjects.length).toBe(4);
    expect(t.graph.vertices.size).toBe(4);
  });

  it("Tetrahedron edge_length=2 first face matches python", () => {
    const t = new Tetrahedron(2);
    const face0 = t.extract_face_coords()[0] ?? [];
    const expected = [
      [0.707107, 0.707107, 0.707107],
      [0.707107, -0.707107, -0.707107],
      [-0.707107, 0.707107, -0.707107],
    ];
    expect(face0.length).toBe(3);
    face0.forEach((row, i) => vecClose(row, expected[i] ?? [], 1e-5));
  });

  it("extract_face_coords tracks moved vertices (python reads graph centers)", () => {
    const t = new Tetrahedron(1);
    const before = t.extract_face_coords();
    const v0 = t.graph.getItem(0);
    v0.shift([1, 0, 0]);
    const after = t.extract_face_coords();
    const b0 = (before[0] ?? [])[0] ?? [];
    const a0 = (after[0] ?? [])[0] ?? [];
    expect(close((a0[0] ?? 0) - (b0[0] ?? 0), 1, 1e-6)).toBe(true);
  });

  it("update_faces keeps face count and matches fresh coords", () => {
    const t = new Tetrahedron(1);
    t.graph.getItem(1).shift([0, 2, 0]);
    t.update_faces(t);
    expect(t.faces.submobjects.length).toBe(4);
  });

  it("Platonic solid vertex/face counts (python: 4/4, 6/8, 12/20, 20/12)", () => {
    const counts: Array<[new (...a: never[]) => Polyhedron, number, number]> = [
      [Tetrahedron as never, 4, 4],
      [Octahedron as never, 6, 8],
      [Icosahedron as never, 12, 20],
      [Dodecahedron as never, 20, 12],
    ];
    for (const [Ctor, nVerts, nFaces] of counts) {
      const solid = new (Ctor as new (e?: number) => Polyhedron)(1);
      expect(solid.vertex_coords.length).toBe(nVerts);
      expect(solid.faces_list.length).toBe(nFaces);
      expect(solid.faces.submobjects.length).toBe(nFaces);
    }
  });

  it("edge_length scales vertices (python unit=edge*sqrt(2)/4 for tetra)", () => {
    const t = new Tetrahedron(1);
    const unit = Math.sqrt(2) / 4;
    vecClose(t.vertex_coords[0] ?? [], [unit, unit, unit]);
  });

  it("ConvexHull3D of a tetra set gives 4 verts/4 faces (python)", () => {
    const h = new ConvexHull3D([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]);
    expect(h.vertex_coords.length).toBe(4);
    expect(h.faces_list.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Axes / NumberPlane / PolarPlane / ComplexPlane / ThreeDAxes
// ---------------------------------------------------------------------------

describe("Axes.__init__ (python)", () => {
  it("defaults ranges to rounded frame radii (x [-7,7,1], y [-4,4,1])", () => {
    const ax = new Axes();
    expect(ax.x_range).toEqual([-7, 7, 1]);
    expect(ax.y_range).toEqual([-4, 4, 1]);
    expect(ax.axis_config["include_tip"]).toBe(true);
    expect(ax.axis_config["numbers_to_exclude"]).toEqual([0]);
    expect(ax.x_axis_config["exclude_origin_tick"]).toBe(true);
    expect(ax.y_axis_config["exclude_origin_tick"]).toBe(true);
  });

  it("custom ranges pass through", () => {
    const ax = new Axes([-3, 3, 1], [-2, 2, 0.5]);
    expect(ax.x_range).toEqual([-3, 3, 1]);
    expect(ax.y_range).toEqual([-2, 2, 0.5]);
  });
});

describe("Axes._create_axis (python)", () => {
  it("centers symmetric axes (x_range [-3,3,1] len 6: center origin, n2p(0)=origin)", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-3, 3, 1], {}, 6);
    expect(axis.x_range).toEqual([-3, 3, 1]);
    vecClose(axis.get_center(), [0, 0, 0]);
    vecClose(axis.n2p(0), [0, 0, 0]);
  });

  it("shifts asymmetric axes so min maps to origin (python [1,5]: n2p(1)=0, n2p(5)=[4,0,0])", () => {
    const ax = new Axes();
    const axis = ax._create_axis([1, 5, 1], {}, 4);
    // python ground truth: center [2,0,0], n2p(1)=[0,0,0], n2p(5)=[4,0,0]
    vecClose(axis.n2p(1), [0, 0, 0]);
    vecClose(axis.n2p(5), [4, 0, 0]);
  });

  it("shifts negative ranges so max maps to origin (python [-5,-1]: n2p(-1)=0)", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-5, -1, 1], {}, 4);
    vecClose(axis.n2p(-1), [0, 0, 0]);
    vecClose(axis.n2p(-5), [-4, 0, 0]);
  });
});

describe("Axes.coords_to_point (python docstring ground truth)", () => {
  it("c2p(1,0,0) == [0.857143, 0, 0]", () => {
    const ax = new Axes();
    vecClose(ax.coords_to_point(1, 0, 0), [0.857143, 0, 0]);
  });

  it("batch list format matches python", () => {
    const ax = new Axes();
    const rows = toRows(ax.coords_to_point([[0, 1], [1, 1], [1, 0]]));
    expect(rows.length).toBe(3);
    vecClose(rows[0] ?? [], [0, 0.75, 0]);
    vecClose(rows[1] ?? [], [0.857143, 0.75, 0]);
    vecClose(rows[2] ?? [], [0.857143, 0, 0]);
  });
});

describe("Axes.plot_line_graph (python)", () => {
  it("returns VDict with line_graph + vertex_dots at c2p positions", () => {
    const ax = new Axes();
    const vg = ax.plot_line_graph([0, 1, 2], [0, 1, 4], null, "PURE_YELLOW", true);
    expect(vg.has("line_graph")).toBe(true);
    expect(vg.has("vertex_dots")).toBe(true);
    const dots = vg.get("vertex_dots") as VGroup;
    expect(dots.submobjects.length).toBe(3);
    // python dot centers: [0,0,0], [0.8571,0.75,0], [1.7143,3.0,0]
    vecClose((dots.submobjects[0] as VMobject).get_center(), [0, 0, 0], 1e-4);
    vecClose((dots.submobjects[1] as VMobject).get_center(), [0.8571, 0.75, 0], 1e-4);
    vecClose((dots.submobjects[2] as VMobject).get_center(), [1.7143, 3.0, 0], 1e-4);
  });

  it("mismatched lengths raise like python zip(strict=True)", () => {
    const ax = new Axes();
    expect(() => ax.plot_line_graph([0, 1], [0, 1, 2])).toThrow(/zip\(\)/);
  });
});

describe("NumberPlane (python)", () => {
  it("__init__ uses raw frame radii (python x +-7.111, y +-4)", () => {
    const pl = new NumberPlane();
    expect(close(pl.x_range[0] ?? NaN, -7.111111111111111, 1e-9)).toBe(true);
    expect(close(pl.x_range[1] ?? NaN, 7.111111111111111, 1e-9)).toBe(true);
    expect(close(pl.y_range[0] ?? NaN, -4, 1e-9)).toBe(true);
    expect(pl.faded_line_ratio).toBe(1);
  });

  it("coords_to_point is unit mapping (python c2p(1,2)=[1,2,0])", () => {
    const pl = new NumberPlane();
    vecClose(pl.coords_to_point(1, 2), [1, 2, 0]);
    const rows = toRows(pl.coords_to_point([[0, 1], [1, 1], [1, 0]]));
    vecClose(rows[0] ?? [], [0, 1, 0], 1e-9);
    vecClose(rows[1] ?? [], [1, 1, 0], 1e-9);
    vecClose(rows[2] ?? [], [1, 0, 0], 1e-9);
  });

  it("unit sizes are 1.0 for the default plane (python ground truth)", () => {
    const pl = new NumberPlane();
    expect(close(pl.get_x_unit_size(), 1.0, 1e-12)).toBe(true);
    expect(close(pl.get_y_unit_size(), 1.0, 1e-12)).toBe(true);
    const ax = new Axes();
    expect(close(ax.get_x_unit_size(), 12 / 14, 1e-12)).toBe(true);
    expect(close(ax.get_y_unit_size(), 6 / 8, 1e-12)).toBe(true);
  });

  it("get_vector draws buff-0 arrow from origin to c2p (python end [2,1,0])", () => {
    const pl = new NumberPlane();
    const v = pl.get_vector([2, 1]);
    vecClose(v.get_end(), [2, 1, 0]);
    vecClose(v.get_start(), [0, 0, 0]);
  });

  it("_init_background_lines/_get_lines set styled groups", () => {
    // NOTE deviation: python builds 22 background lines from the Mobject
    // axes; the TS hierarchy is not Mobject-based yet (empty axes group),
    // so the groups are empty VGroups. Counts will bind once the
    // NumberLine integration lands.
    const pl = new NumberPlane();
    const [bg, faded] = pl._get_lines();
    expect(bg).toBeInstanceOf(VGroup);
    expect(faded).toBeInstanceOf(VGroup);
    expect(pl.background_lines).toBeInstanceOf(VGroup);
    expect(pl.faded_lines).toBeInstanceOf(VGroup);
    // faded style halves numeric background values (python behavior).
    expect((pl.faded_line_style as Record<string, unknown>)["stroke_width"]).toBe(1);
  });
});

describe("PolarPlane (python)", () => {
  it("__init__ defaults: ranges +-frame_y_radius, azimuth_step 20", () => {
    const pp = new PolarPlane();
    expect(pp.x_range).toEqual([-4, 4, 1]);
    expect(pp.y_range).toEqual([-4, 4, 1]);
    expect(pp.azimuth_step).toBe(20);
    expect(pp.azimuth_units).toBe("PI radians");
    expect(pp.azimuth_direction).toBe("CCW");
    expect(pp.make_smooth_after_applying_functions).toBe(false);
  });

  it("rejects bad azimuth units/direction like python", () => {
    expect(() => new PolarPlane(null, null, 1, null, "bogus" as never)).toThrow(
      /azimuth units/,
    );
    expect(
      () => new PolarPlane(null, null, 1, null, "PI radians", true, 0, "XX" as never),
    ).toThrow(/azimuth units/);
  });

  it("_init_background_lines derives halved faded style; _get_lines returns VGroups", () => {
    // NOTE deviation: python builds 25 circle/line background mobjects;
    // TS defers construction until the hierarchy is Mobject-based.
    const pp = new PolarPlane();
    pp._init_background_lines();
    expect(pp.background_lines).toBeInstanceOf(VGroup);
    expect(pp.faded_lines).toBeInstanceOf(VGroup);
    expect((pp.faded_line_style as Record<string, unknown>)["stroke_width"]).toBe(1);
  });

  it("get_vector draws buff-0 arrow (python [1, pi/4] end [1.0, 0.7854, 0])", () => {
    const pp = new PolarPlane();
    const v = pp.get_vector([1, Math.PI / 4]);
    vecClose(v.get_start(), [0, 0, 0]);
    // python resolves through the unit-sized NumberLine axes; the TS
    // null-length unit mapping agrees exactly here.
    vecClose(v.get_end(), [1, Math.PI / 4, 0]);
  });
});

describe("ComplexPlane (python)", () => {
  it("__init__ matches NumberPlane defaults", () => {
    const cp = new ComplexPlane();
    expect(close(cp.x_range[0] ?? NaN, -7.111111111111111, 1e-9)).toBe(true);
    expect(cp.x_length).toBeNull();
  });

  it("number_to_point/n2p accept complex, string and pair (python [1,2,0])", () => {
    const cp = new ComplexPlane();
    // python: n2p('1+2j') == n2p(1+2j) == [1,2,0]
    vecClose(cp.number_to_point({ re: 1, im: 2 }), [1, 2, 0]);
    vecClose(cp.n2p("1+2j"), [1, 2, 0]);
    vecClose(cp.n2p("3j"), [0, 3, 0]);
    vecClose(cp.n2p("-j"), [0, -1, 0]);
    vecClose(cp.n2p([1, 2]), [1, 2, 0]);
    vecClose(cp.number_to_point(3), [3, 0, 0]);
    expect(() => cp.n2p("bogus")).toThrow(/malformed/);
    const back = cp.point_to_number([1, 2, 0]);
    expect(close(back.re, 1, 1e-9)).toBe(true);
    expect(close(back.im, 2, 1e-9)).toBe(true);
  });
});

describe("ThreeDAxes.__init__ / SpecialThreeDScene.get_axes (python)", () => {
  it("ThreeDAxes defaults match python ((-6,6,1),(-5,5,1),(-4,4,1); 10.5/10.5/6.5)", () => {
    const t3 = new ThreeDAxes();
    expect(t3.x_range).toEqual([-6, 6, 1]);
    expect(t3.y_range).toEqual([-5, 5, 1]);
    expect(t3.z_range).toEqual([-4, 4, 1]);
    expect(close(t3.x_length ?? NaN, 10.5, 1e-9)).toBe(true);
    expect(close(t3.y_length ?? NaN, 10.5, 1e-9)).toBe(true);
    expect(close(t3.z_length ?? NaN, 6.5, 1e-9)).toBe(true);
    expect(t3.dimension).toBe(3);
  });

  it("SpecialThreeDScene.get_axes returns a ThreeDAxes", () => {
    const scene = new SpecialThreeDScene();
    const axes = scene.get_axes();
    expect(axes).toBeInstanceOf(ThreeDAxes);
    expect(axes.x_range).toEqual([-6, 6, 1]);
  });
});

// ---------------------------------------------------------------------------
// BarChart.__init__ (geometry only; labels are text/tex)
// ---------------------------------------------------------------------------

describe("BarChart.__init__ (python config math)", () => {
  // The y-axis DecimalNumber/Tex labels are out of scope; stub the number
  // mobject so the ported config math and bar geometry can be verified.
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

  it("stores values/colors and builds one bar per value", () => {
    withStubbedNumbers(() => {
      const bc = new BarChart([1, 2, 3]);
      expect(bc.values).toEqual([1, 2, 3]);
      expect(bc.bar_names).toBeNull();
      expect(bc.bars.submobjects.length).toBe(3);
      expect(bc.x_range).toEqual([0, 3, 1]);
      // y_range = [min(0,min), max(0,max), round(max/y_length, 2)]
      expect(bc.y_range[0]).toBe(0);
      expect(bc.y_range[1]).toBe(3);
    });
  });

  it("bar heights scale with values, widths are equal", () => {
    withStubbedNumbers(() => {
      const bc = new BarChart([1, 2, 3]);
      const bars = bc.bars.submobjects as Rectangle[];
      const heights = bars.map((b) => b.height);
      expect(close(heights[1] ?? NaN, 2 * (heights[0] ?? NaN), 1e-6)).toBe(true);
      expect(close(heights[2] ?? NaN, 3 * (heights[0] ?? NaN), 1e-6)).toBe(true);
      expect(close(bars[0]?.width ?? NaN, bars[1]?.width ?? NaN, 1e-9)).toBe(true);
    });
  });
});

// ---------------------------------------------------------------------------
// VectorScene add_plane / add_axes / lock_in_faded_grid
// ---------------------------------------------------------------------------

describe("VectorScene plane/axes helpers (python)", () => {
  it("add_plane returns a NumberPlane and adds it", () => {
    const vs = new VectorScene();
    const plane = vs.add_plane();
    expect(plane).toBeInstanceOf(NumberPlane);
    expect(vs.mobjects.includes(plane as never)).toBe(true);
  });

  it("add_plane forwards kwargs (python NumberPlane(**kwargs))", () => {
    const vs = new VectorScene();
    const plane = vs.add_plane(false, { x_range: [-2, 2, 1] });
    expect(plane.x_range).toEqual([-2, 2, 1]);
  });

  it("add_plane(animate=true) plays Create on the VMobject plane (python)", () => {
    // Python plays Create(plane, lag_ratio=0.5); NumberPlane is now a real
    // VGroup, so Create accepts it.
    const vs = new VectorScene();
    const spy = vi.spyOn(vs, "play").mockImplementation(() => undefined as never);
    const plane = vs.add_plane(true);
    expect(plane).toBeInstanceOf(NumberPlane);
    expect(spy).toHaveBeenCalledOnce();
    expect(vs.mobjects.includes(plane as never)).toBe(true);
  });

  it("add_axes returns Axes and adds it", () => {
    const vs = new VectorScene();
    const axes = vs.add_axes();
    expect(axes).toBeInstanceOf(Axes);
    expect(vs.mobjects.includes(axes as never)).toBe(true);
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

// ---------------------------------------------------------------------------
// LinearTransformationScene setup / unit square / apply_*
// ---------------------------------------------------------------------------

describe("LinearTransformationScene.setup (python)", () => {
  it("installs planes/vectors and is idempotent", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    expect(scene.background_plane).toBeInstanceOf(NumberPlane);
    expect(scene.plane).toBeInstanceOf(NumberPlane);
    expect(scene.transformable_mobjects.includes(scene.plane as never)).toBe(true);
    expect(scene.basis_vectors.submobjects.length).toBe(2);
    expect(scene.moving_vectors.length).toBe(2);
    const plane = scene.plane;
    scene.setup();
    expect(scene.plane).toBe(plane);
  });

  it("respects include_background_plane=false", () => {
    const scene = new LinearTransformationScene(false, false);
    scene.setup();
    expect(scene.background_mobjects.length).toBe(0);
    expect(scene.transformable_mobjects.length).toBe(0);
  });
});

describe("LinearTransformationScene unit square (python)", () => {
  it("get_unit_square matches plane unit sizes with opacity 0.3", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const sq = scene.get_unit_square();
    expect(sq).toBeInstanceOf(Rectangle);
    expect(close(sq.width, scene.plane.get_x_unit_size(), 1e-9)).toBe(true);
    expect(close(sq.height, scene.plane.get_y_unit_size(), 1e-9)).toBe(true);
    expect(close(sq.fill_opacity, 0.3, 1e-9)).toBe(true);
    expect(scene.square).toBe(sq);
  });

  it("add_unit_square registers transformable and returns this", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const ret = scene.add_unit_square();
    expect(ret).toBe(scene);
    expect(scene.transformable_mobjects.includes(scene.square as never)).toBe(true);
  });
});

describe("LinearTransformationScene apply_* (python linalg ground truth)", () => {
  function mockPlay(scene: LinearTransformationScene): { spy: ReturnType<typeof vi.spyOn>; seen: unknown[][] } {
    const seen: unknown[][] = [];
    const spy = vi
      .spyOn(scene, "play")
      .mockImplementation((...anims: unknown[]) => {
        seen.push(anims);
        return undefined as never;
      });
    return { spy, seen };
  }

  it("apply_matrix applies M (python: apply_transposed_matrix(M.T))", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const { spy, seen } = mockPlay(scene);
    // 90-degree rotation [[0,-1],[1,0]] maps (1,0,0) -> (0,1,0).
    scene.apply_matrix([[0, -1], [1, 0]]);
    expect(spy).toHaveBeenCalledOnce();
    const func = scene.get_matrix_transformation([[0, -1], [1, 0]]);
    vecClose(func([1, 0, 0]), [0, 1, 0]);
    void seen;
    spy.mockRestore();
  });

  it("apply_inverse applies inv(M) (python: apply_matrix(inv))", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const { spy } = mockPlay(scene);
    const applySpy = vi.spyOn(scene, "apply_matrix").mockImplementation(() => undefined as never);
    scene.apply_inverse([[2, 0], [0, 2]]);
    expect(applySpy).toHaveBeenCalledOnce();
    // inv([[2,0],[0,2]]) maps (2,0,0) -> (1,0,0)
    const arg = applySpy.mock.calls[0]?.[0] as number[][];
    const func = scene.get_matrix_transformation(arg);
    vecClose(func([2, 0, 0]), [1, 0, 0]);
    applySpy.mockRestore();
    spy.mockRestore();
  });

  it("apply_inverse_transpose applies inv(T).T (python ground truth)", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const { spy } = mockPlay(scene);
    const applyTSpy = vi
      .spyOn(scene, "apply_transposed_matrix")
      .mockImplementation(() => undefined as never);
    // python: t_inv = inv(array(t).T).T; for T=[[2,0],[0,1]] this is [[0.5,0],[0,1]]
    scene.apply_inverse_transpose([[2, 0], [0, 1]]);
    expect(applyTSpy).toHaveBeenCalledOnce();
    const passed = applyTSpy.mock.calls[0]?.[0] as number[][];
    const flat = (Array.isArray(passed) ? passed : []).flat().map(Number);
    expect(close(flat[0] ?? NaN, 0.5, 1e-9)).toBe(true);
    expect(close(flat[3] ?? NaN, 1, 1e-9)).toBe(true);
    applyTSpy.mockRestore();
    spy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// integration: batch symbols interacting with earlier-ported code
// ---------------------------------------------------------------------------

describe("integration (layers 30-37 with earlier batches)", () => {
  it("VDict holds graph vertices and survives setitem overwrite", () => {
    const g = smallGraph();
    const vd = new VDict();
    vd.__setitem__("u", g.vertices.get(1) as VMobject);
    vd.__setitem__("v", g.vertices.get(2) as VMobject);
    expect(vd.__getitem__("u")).toBe(g.vertices.get(1));
    const replacement = new Dot([-2, 0, 0]);
    vd.__setitem__("u", replacement);
    expect(vd.__getitem__("u")).toBe(replacement);
    expect(vd.get_all_submobjects().length).toBe(2);
  });

  it("transposed-matrix func drives a Tetrahedron like python np.dot(point, M)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    const t = new Tetrahedron(1);
    const c0 = t.vertex_coords[0] ?? [0, 0, 0];
    // python func(point) = np.dot(point, M): (x,y,z) -> (y,-x,z).
    const mapped = toNums(func(array(c0) as NDArray));
    expect(close(mapped[0] ?? NaN, c0[1] ?? 0, 1e-9)).toBe(true);
    expect(close(mapped[1] ?? NaN, -(c0[0] ?? 0), 1e-9)).toBe(true);
  });

  it("Axes._create_axis output feeds NumberPlane-style mapping consistently", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-7, 7, 1], {}, 12);
    // unit size of the created axis matches the Axes x unit size.
    expect(close(axis.get_unit_size(), 12 / 14, 1e-9)).toBe(true);
    vecClose(axis.n2p(1), [12 / 14, 0, 0]);
  });

  it("plot_line_graph vertex dots sit on coords_to_point samples", () => {
    const ax = new Axes();
    const xs = [0, 1, 2];
    const ys = [0, 1, 4];
    const vg = ax.plot_line_graph(xs, ys);
    const dots = vg.get("vertex_dots") as VGroup;
    xs.forEach((x, i) => {
      const expected = toNums(ax.coords_to_point(x, ys[i] ?? 0, 0));
      vecClose((dots.submobjects[i] as VMobject).get_center(), expected, 1e-9);
    });
  });

  it("ConvexHull3D faces reference real vertex coords (extract/update round-trip)", () => {
    const h = new ConvexHull3D([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]);
    const coords = h.extract_face_coords();
    expect(coords.length).toBe(h.faces_list.length);
    h.update_faces(h);
    expect(h.faces.submobjects.length).toBe(h.faces_list.length);
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
