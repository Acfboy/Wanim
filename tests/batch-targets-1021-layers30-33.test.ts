import { describe, expect, it, vi } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { VDict, VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Square, Rectangle } from "../src/mobject/geometry/polygram.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { LogBase } from "../src/mobject/graphing/scale.js";
import {
  DecimalNumber,
  Integer,
  Variable,
  string_to_mob_map,
} from "../src/mobject/text/numbers.js";
import { index_labels } from "../src/utils/debug.js";
import { NumberLine, UnitInterval } from "../src/mobject/graphing/number_line.js";
import {
  Axes,
  NumberPlane,
  PolarPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import { Tetrahedron, Polyhedron } from "../src/mobject/three_d/polyhedra.js";
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
import { AnimationGroup } from "../src/animation/composition.js";

// Ground truths captured with original manim
// (manim 0.21.0). Representative citations:
// - VDict({'a':sq,'b':sq}) -> keys ['a','b'], 2 submobjects; setitem 'c'
//   appends; overwrite 'a' -> order ['b','c','a'], len 3.
// - Graph([1,2,3],[(1,2),(2,3)],layout={1:[-2,0,0],2:[0,0,0],3:[2,0,0]}):
//   g[1] is Dot, g[(1,2)] is Line, g[999] ValueError "Could not find 999".
// - DiGraph([1,2],[(1,2)]) -> "Directed graph on 2 vertices and 1 edges".
// - Axes()._create_axis([-3,3,1],{},6): center [0,0,0], n2p(0)=[0,0,0],
//   n2p(1)=[1,0,0], unit 1.0; ([1,5,1],4): center [2,0,0], n2p(1)=[0,0,0],
//   n2p(5)=[4,0,0]; ([-5,-1,1],4): n2p(-1)=[0,0,0], n2p(-5)=[-4,0,0].
// - Tetrahedron(2) face0 [[.707107,.707107,.707107],[.707107,-.707107,-.707107],
//   [-.707107,.707107,-.707107]], 12 edges.
// - NumberLine([-3,3,1]): tick_range [-3,-2,-1,-0,1,2,3], unit 1.0.
// - UnitInterval(): x_range [0,1,0.1], unit 10.0, dec_places 1.
// - get_transposed_matrix_transformation([[0,-1],[1,0]]): [1,0,0]->[0,-1,0],
//   [0,1,0]->[1,0,0]. inv([[2,0],[0,2]])=[[.5,0],[0,.5]].
// - DecimalNumber complex: _get_num_string(1.5-2.5j) == '1.50-2.50i';
//   complex(1.5,-2.5)+2.0 == (3.5-2.5j).

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

const dummyFactory = (text: string) => {
  void text;
  return new VMobject();
};

function smallGraph(): Graph {
  return new Graph([1, 2, 3], [[1, 2], [2, 3]], {
    layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
      string,
      unknown
    >,
  });
}

// ---------------------------------------------------------------------------
// VDict.__init__ / __setitem__ / get_all_submobjects
// ---------------------------------------------------------------------------

describe("VDict batch (python ground truth)", () => {
  it("__init__ from plain record keeps keys and submobjects", () => {
    const vd = new VDict({ a: new Square(), b: new Square() });
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b"]);
    expect(vd.submobjects.length).toBe(2);
    expect(vd.show_keys).toBe(false);
  });

  it("__init__ from Map and entry iterable", () => {
    const a = new Square();
    const fromMap = new VDict(new Map<unknown, VMobject>([["x", a]]));
    expect(fromMap.get("x")).toBe(a);
    const b = new Square();
    const fromIter = new VDict([["x", a], ["y", b]]);
    expect([...fromIter.submob_dict.keys()]).toEqual(["x", "y"]);
    expect(fromIter.get_all_submobjects()).toEqual([a, b]);
  });

  it("__setitem__ appends new keys", () => {
    const vd = new VDict({ a: new Square(), b: new Square() });
    const c = new Square();
    vd.__setitem__("c", c);
    expect([...vd.submob_dict.keys()]).toEqual(["a", "b", "c"]);
    expect(vd.__getitem__("c")).toBe(c);
    expect(vd.submobjects.length).toBe(3);
  });

  it("__setitem__ overwrite moves key to end (python ['b','c','a'])", () => {
    const vd = new VDict({ a: new Square(), b: new Square() });
    vd.__setitem__("c", new Square());
    const a2 = new Square();
    vd.__setitem__("a", a2);
    expect([...vd.submob_dict.keys()]).toEqual(["b", "c", "a"]);
    expect(vd.submobjects.length).toBe(3);
    expect(vd.__getitem__("a")).toBe(a2);
  });

  it("get_all_submobjects returns dict values in insertion order", () => {
    const a = new Square();
    const b = new Square();
    const vd = new VDict({ a, b });
    expect(vd.get_all_submobjects()).toEqual([a, b]);
    expect(vd.get_all_submobjects().length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// LogBase.get_custom_labels / index_labels (text/tex stubs)
// ---------------------------------------------------------------------------

describe("LogBase.get_custom_labels + index_labels stubs", () => {
  it("LogBase.get_custom_labels throws text/tex TODO (python builds Integer 10^exp labels)", () => {
    // python: [Integer(base, unit='^{exp}')] per value; needs latex.
    expect(() => new LogBase().get_custom_labels([1, 10, 100])).toThrow(
      /text\/tex not ported/,
    );
    expect(() => new LogBase(2).get_custom_labels([2, 4], 1)).toThrow(
      /text\/tex not ported/,
    );
  });

  it("index_labels throws text/tex TODO (python builds one Integer per submobject)", () => {
    const g = new VGroup(new Square(), new Square());
    expect(() => index_labels(g)).toThrow(/text\/tex not ported/);
    // defaults mirror python signature (label_height 0.15 etc.) — still stub.
    expect(() => index_labels(g, 0.2, 3)).toThrow(/text\/tex not ported/);
  });
});

// ---------------------------------------------------------------------------
// DecimalNumber.increment_value / Variable.__init__
// ---------------------------------------------------------------------------

describe("DecimalNumber.increment_value (python: set_value(get_value()+delta))", () => {
  it("increments real values with default delta 1", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(5, { mob_class: dummyFactory });
    expect(d.get_value()).toBe(5);
    d.increment_value();
    expect(d.get_value()).toBe(6);
  });

  it("increments by explicit and negative deltas", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(5, { mob_class: dummyFactory });
    d.increment_value(2.5);
    expect(close(d.get_value() as number, 7.5, 1e-12)).toBe(true);
    d.increment_value(-7.5);
    expect(close(d.get_value() as number, 0, 1e-12)).toBe(true);
  });

  it("adds delta to the real part of complex values (python complex+float)", () => {
    // python: complex(1.5,-2.5)+2.0 == (3.5-2.5j)
    string_to_mob_map.clear();
    const d = new DecimalNumber({ re: 1.5, im: -2.5 }, { mob_class: dummyFactory });
    d.increment_value(2.0);
    const got = d.get_value() as { re: number; im: number };
    expect(close(got.re, 3.5, 1e-12)).toBe(true);
    expect(close(got.im, -2.5, 1e-12)).toBe(true);
    expect(d._get_num_string(d.get_value() as never)).toBe("3.50-2.50i");
  });

  it("returns self for chaining", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(0, { mob_class: dummyFactory });
    expect(d.increment_value(1)).toBe(d);
  });
});

describe("Variable.__init__ (python needs MathTex for label + '=')", () => {
  it("string labels throw text/tex TODO (python MathTex(label))", () => {
    expect(() => new Variable(0.5, "x")).toThrow(/text\/tex not ported/);
  });

  it("VMobject labels still throw (python adds MathTex('=') next_to label)", () => {
    // NOTE deviation: python accepts a VMobject label without latex (only the
    // '=' needs MathTex); the TS port builds the '=' unconditionally first,
    // so every construction currently throws. Binds once text lands.
    expect(() => new Variable(0.5, new VMobject())).toThrow(
      /text\/tex not ported/,
    );
  });

  it("Integer var_type path also requires text (same '=' stub)", () => {
    expect(() => new Variable(1, new VMobject(), Integer)).toThrow(
      /text\/tex not ported/,
    );
  });
});

// ---------------------------------------------------------------------------
// NumberLine.__init__ / add_numbers / get_number_mobjects / get_labels / UnitInterval
// ---------------------------------------------------------------------------

describe("NumberLine.__init__ (python ground truth)", () => {
  it("defaults x_range from rounded frame radii [-7,7,1]", () => {
    const nl = new NumberLine();
    expect(nl.x_range).toEqual([-7, 7, 1]);
    expect([nl.x_min, nl.x_max, nl.x_step]).toEqual([-7, 7, 1]);
  });

  it("two-element x_range gains step 1 (python [*x_range, 1])", () => {
    const nl = new NumberLine({ x_range: [-3, 3] });
    expect(nl.x_range).toEqual([-3, 3, 1]);
  });

  it("decimal places derive from step (python _decimal_places_from_step)", () => {
    expect(NumberLine._decimal_places_from_step(1)).toBe(0);
    expect(NumberLine._decimal_places_from_step(0.1)).toBe(1);
    const nl = new NumberLine({ x_range: [-3, 3, 0.5] });
    expect(nl.decimal_number_config["num_decimal_places"]).toBe(1);
  });

  it("tick range matches python (7 ticks incl -0.0)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const got = toNums(nl.get_tick_range());
    expect(got.length).toBe(7);
    [-3, -2, -1, 0, 1, 2, 3].forEach((w, i) =>
      expect(close(got[i] ?? NaN, w, 1e-12)).toBe(true),
    );
    expect(nl.ticks.submobjects.length).toBe(7);
  });

  it("length rescales unit_size (python [-3,3] length 6 -> unit 1)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expect(close(nl.get_unit_size(), 1, 1e-9)).toBe(true);
    vecClose(nl.n2p(0), [0, 0, 0]);
    vecClose(nl.n2p(1), [1, 0, 0]);
  });
});

describe("NumberLine label builders (text/tex stubs)", () => {
  it("get_number_mobject throws (python builds DecimalNumber)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(() => nl.get_number_mobject(1)).toThrow(/text\/tex not ported/);
  });

  it("get_number_mobjects forwards to get_number_mobject; empty call mirrors AttributeError", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(() => nl.get_number_mobjects(1, 2)).toThrow(/text\/tex not ported/);
    expect(() => nl.get_number_mobjects()).toThrow(/default_numbers_to_display/);
    try {
      nl.get_number_mobjects();
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("AttributeError");
    }
  });

  it("get_labels delegates to get_number_mobjects (same AttributeError)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(() => nl.get_labels()).toThrow(/default_numbers_to_display/);
  });

  it("add_numbers throws once labels are needed (python DecimalNumber path)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(() => nl.add_numbers([1, 2])).toThrow(/text\/tex not ported/);
    expect(() => nl.add_numbers()).toThrow(/text\/tex not ported/);
  });

  it("_create_label_tex passes VMobjects through (python behavior)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm);
    expect(() => nl._create_label_tex("1")).toThrow(/text\/tex not ported/);
  });
});

describe("UnitInterval.__init__ (python: [0,1,0.1], unit 10)", () => {
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
// VectorScene / LinearTransformationScene batch symbols
// ---------------------------------------------------------------------------

describe("VectorScene.write_vector_coordinates / vector_to_coords stubs", () => {
  it("write_vector_coordinates throws (python Matrix + Write)", () => {
    const vs = new VectorScene();
    expect(() => vs.write_vector_coordinates(new Square())).toThrow(
      /text\/tex not ported/,
    );
  });

  it("vector_to_coords throws (python Matrix/Integer + played lines)", () => {
    const vs = new VectorScene();
    expect(() => vs.vector_to_coords([1, 2, 0])).toThrow(/text\/tex not ported/);
    expect(() => vs.vector_to_coords([1, 2, 0], false)).toThrow(
      /text\/tex not ported/,
    );
  });
});

describe("LinearTransformationScene batch symbols", () => {
  it("write_vector_coordinates delegates to super then foregrounds (currently stub)", () => {
    const scene = new LinearTransformationScene();
    expect(() => scene.write_vector_coordinates(new Square())).toThrow(
      /text\/tex not ported/,
    );
  });

  it("apply_matrix maps (1,0,0)->(0,1,0) for 90-degree rotation", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const spy = vi.spyOn(scene, "play").mockImplementation(() => undefined as never);
    scene.apply_matrix([[0, -1], [1, 0]]);
    expect(spy).toHaveBeenCalledOnce();
    vecClose(scene.get_matrix_transformation([[0, -1], [1, 0]])([1, 0, 0]), [0, 1, 0]);
    spy.mockRestore();
  });

  it("apply_inverse applies inv(M): (2,0,0)->(1,0,0) for 2I", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const playSpy = vi.spyOn(scene, "play").mockImplementation(() => undefined as never);
    const applySpy = vi.spyOn(scene, "apply_matrix").mockImplementation(() => undefined as never);
    scene.apply_inverse([[2, 0], [0, 2]]);
    expect(applySpy).toHaveBeenCalledOnce();
    const arg = applySpy.mock.calls[0]?.[0] as number[][];
    vecClose(scene.get_matrix_transformation(arg)([2, 0, 0]), [1, 0, 0]);
    applySpy.mockRestore();
    playSpy.mockRestore();
  });

  it("apply_inverse_transpose passes inv(T).T ([[2,0],[0,1]] -> [[.5,0],[0,1]])", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [];
    const playSpy = vi.spyOn(scene, "play").mockImplementation(() => undefined as never);
    const tSpy = vi
      .spyOn(scene, "apply_transposed_matrix")
      .mockImplementation(() => undefined as never);
    scene.apply_inverse_transpose([[2, 0], [0, 1]]);
    expect(tSpy).toHaveBeenCalledOnce();
    const flat = ((tSpy.mock.calls[0]?.[0] as number[][]) ?? []).flat().map(Number);
    expect(close(flat[0] ?? NaN, 0.5, 1e-9)).toBe(true);
    expect(close(flat[3] ?? NaN, 1, 1e-9)).toBe(true);
    tSpy.mockRestore();
    playSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// GenericGraph batch symbols
// ---------------------------------------------------------------------------

describe("GenericGraph.__getitem__ (python)", () => {
  it("resolves vertices and edges, ValueError otherwise", () => {
    const g = smallGraph();
    expect(g.__getitem__(1)).toBe(g.vertices.get(1));
    expect(g.__getitem__([1, 2] as never)).toBe(
      [...g.edges.values()][0] as never,
    );
    expect(() => g.__getitem__(999 as never)).toThrow(/Could not find 999/);
    expect(() => g.__getitem__([1, 3] as never)).toThrow(/Could not find/);
  });
});

describe("GenericGraph animation overrides (python @override_animate)", () => {
  it("_add_vertices_animation defers insertion until on_finish", () => {
    const g = smallGraph();
    const anim = g._add_vertices_animation(4);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(anim.animations.length).toBe(1);
    expect(g.vertices.has(4)).toBe(false);
    const fakeScene = { removed: [] as unknown[], remove(m: unknown) { this.removed.push(m); } };
    (anim as unknown as { on_finish: (s: unknown) => void }).on_finish(fakeScene);
    expect(g.vertices.has(4)).toBe(true);
    expect(fakeScene.removed.length).toBe(1);
  });

  it("_add_vertices_animation.on_finish inserts all created vertices", () => {
    const g = smallGraph();
    const anim = g._add_vertices_animation(7, 8);
    expect(anim.animations.length).toBe(2);
    const fakeScene = { remove(_m: unknown) { void _m; } };
    (anim as unknown as { on_finish: (s: unknown) => void }).on_finish(fakeScene);
    expect(g.vertices.has(7)).toBe(true);
    expect(g.vertices.has(8)).toBe(true);
  });

  it("_remove_vertices_animation removes vertices + incident edges", () => {
    const g = smallGraph();
    const anim = g._remove_vertices_animation(3);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(g.vertices.has(3)).toBe(false);
    expect(g.vertices.has(2)).toBe(true);
    expect(anim.animations.length).toBeGreaterThan(0);
  });

  it("_add_edges_animation adds the edge eagerly and animates it", () => {
    const g = new Graph([1, 2, 3], [[1, 2]], {
      layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
    const anim = g._add_edges_animation([2, 3]);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(g.edges.size).toBe(2);
    expect(() => g.__getitem__([2, 3] as never)).not.toThrow();
    expect(anim.animations.length).toBeGreaterThan(0);
  });

  it("_remove_edges_animation removes eagerly, keeps vertices", () => {
    const g = smallGraph();
    const anim = g._remove_edges_animation([1, 2]);
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect(() => g.__getitem__([1, 2] as never)).toThrow(/Could not find/);
    expect(g.vertices.has(1)).toBe(true);
  });
});

describe("GenericGraph.from_networkx / DiGraph", () => {
  it("from_networkx builds the same vertices and edges", () => {
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

  it("DiGraph is directed with its own repr", () => {
    const d = new DiGraph([1, 2], [[1, 2]], {
      layout: { 1: [-1, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    expect(d).toBeInstanceOf(GenericGraph);
    expect(d.vertices.size).toBe(2);
    expect(d.edges.size).toBe(1);
    expect(d.toString()).toBe("Directed graph on 2 vertices and 1 edges");
    expect(() => d.__getitem__([2, 1] as never)).toThrow(/Could not find/);
  });
});

// ---------------------------------------------------------------------------
// Polyhedron.__init__ / extract_face_coords / update_faces
// ---------------------------------------------------------------------------

describe("Polyhedron batch (python ground truth)", () => {
  it("__init__ stores coords/faces/edges and builds faces+graph", () => {
    const t = new Tetrahedron(1);
    expect(t.vertex_coords.length).toBe(4);
    expect(t.faces_list.length).toBe(4);
    expect(t.get_edges([[0, 1, 2]])).toEqual([[0, 1], [1, 2], [2, 0]]);
    expect(t.edges.length).toBe(12);
    expect(t.faces.submobjects.length).toBe(4);
    expect(t.graph.vertices.size).toBe(4);
    expect(t.faces_config["fill_opacity"]).toBe(0.5);
  });

  it("extract_face_coords matches python for edge_length=2", () => {
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

  it("extract_face_coords tracks moved graph vertices", () => {
    const t = new Tetrahedron(1);
    const before = (t.extract_face_coords()[0] ?? [])[0] ?? [];
    t.graph.getItem(0).shift([1, 0, 0]);
    const after = (t.extract_face_coords()[0] ?? [])[0] ?? [];
    expect(close((after[0] ?? 0) - (before[0] ?? 0), 1, 1e-6)).toBe(true);
  });

  it("update_faces keeps face count after a vertex move", () => {
    const t = new Tetrahedron(1) as Tetrahedron & { update_faces(m: never): Polyhedron };
    t.graph.getItem(1).shift([0, 2, 0]);
    t.update_faces(t as never);
    expect(t.faces.submobjects.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Axes._create_axis / BarChart._add_x_axis_labels / PolarPlane._get_lines
// ---------------------------------------------------------------------------

describe("Axes._create_axis (python ground truth)", () => {
  it("centers symmetric axes", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-3, 3, 1], {}, 6);
    expect(axis.x_range).toEqual([-3, 3, 1]);
    vecClose(axis.get_center(), [0, 0, 0]);
    vecClose(axis.n2p(0), [0, 0, 0]);
    vecClose(axis.n2p(1), [1, 0, 0]);
  });

  it("shifts positive ranges so min maps to origin (python [1,5]->[0,4])", () => {
    const ax = new Axes();
    const axis = ax._create_axis([1, 5, 1], {}, 4);
    vecClose(axis.n2p(1), [0, 0, 0]);
    vecClose(axis.n2p(5), [4, 0, 0]);
  });

  it("shifts negative ranges so max maps to origin (python [-5,-1])", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-5, -1, 1], {}, 4);
    vecClose(axis.n2p(-1), [0, 0, 0]);
    vecClose(axis.n2p(-5), [-4, 0, 0]);
  });
});

describe("BarChart._add_x_axis_labels stub + geometry", () => {
  it("throws text/tex TODO (python builds Tex labels)", () => {
    const bc = new BarChart([1, 2, 3]);
    expect(() => bc._add_x_axis_labels()).toThrow(/text\/tex not ported/);
    expect(() => bc.get_bar_labels()).toThrow(/text\/tex not ported/);
    expect(() => new BarChart([1, 2], { bar_names: ["a", "b"] })).toThrow(
      /text\/tex not ported/,
    );
  });

  it("bar geometry stays usable without labels", () => {
    const bc = new BarChart([1, 2, 3]);
    expect(bc.bars.submobjects.length).toBe(3);
    const bars = bc.bars.submobjects as Rectangle[];
    expect(close((bars[1]?.height ?? NaN), 2 * (bars[0]?.height ?? NaN), 1e-6)).toBe(true);
  });
});

describe("PolarPlane._get_lines (deferred construction)", () => {
  it("returns empty VGroups until the Mobject-based axes integration lands", () => {
    // NOTE deviation: python builds 25 circle/line background mobjects; the TS
    // hierarchy is not Mobject-based yet (empty axes group), so both groups
    // are empty VGroups. Counts bind once NumberLine integration lands.
    const pp = new PolarPlane();
    const [bg, faded] = pp._get_lines();
    expect(bg).toBeInstanceOf(VGroup);
    expect(faded).toBeInstanceOf(VGroup);
    expect(bg.submobjects.length).toBe(0);
    expect(faded.submobjects.length).toBe(0);
  });

  it("NumberPlane._get_lines shares the same deferred contract", () => {
    const pl = new NumberPlane();
    const [bg, faded] = pl._get_lines();
    expect(bg).toBeInstanceOf(VGroup);
    expect(faded).toBeInstanceOf(VGroup);
  });
});

// ---------------------------------------------------------------------------
// integration with earlier-ported code
// ---------------------------------------------------------------------------

describe("integration (this batch x earlier batches)", () => {
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

  it("transposed-matrix func drives Tetrahedron verts like python np.dot(point, M)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    const t = new Tetrahedron(1);
    const c0 = t.vertex_coords[0] ?? [0, 0, 0];
    // python: np.dot([x,y,z], [[0,-1,0],[1,0,0],[0,0,1]]) == [y,-x,z].
    const mapped = toNums(func(array(c0) as NDArray));
    expect(close(mapped[0] ?? NaN, c0[1] ?? 0, 1e-9)).toBe(true);
    expect(close(mapped[1] ?? NaN, -(c0[0] ?? 0), 1e-9)).toBe(true);
  });

  it("Axes._create_axis output feeds NumberLine n2p/p2n round-trip", () => {
    const ax = new Axes();
    const axis = ax._create_axis([-7, 7, 1], {}, 12);
    expect(close(axis.get_unit_size(), 12 / 14, 1e-9)).toBe(true);
    vecClose(axis.n2p(1), [12 / 14, 0, 0]);
    expect(close(Number(axis.p2n(toNums(axis.n2p(2)))), 2, 1e-9)).toBe(true);
  });

  it("DecimalNumber increment feeds a graph vertex move (numbers x graph)", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(2, { mob_class: dummyFactory });
    d.increment_value(3);
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.vertices.get(1)!.move_to([d.get_value() as number, 0, 0]);
    vecClose(g.vertices.get(1)!.get_center(), [5, 0, 0]);
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

  it("UnitInterval tick positions sit on the baseline line (number_line x VMobject)", () => {
    const ui = new UnitInterval();
    const p0 = toNums(ui.n2p(0));
    const p1 = toNums(ui.n2p(1));
    // distance between ends equals x_length span (unit_size * 1 range).
    const dist = Math.hypot(p1[0]! - p0[0]!, p1[1]! - p0[1]!, p1[2]! - p0[2]!);
    expect(close(dist, 10, 1e-6)).toBe(true);
  });
});
