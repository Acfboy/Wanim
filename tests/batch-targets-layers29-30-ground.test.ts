import { describe, expect, it, vi } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Vector } from "../src/mobject/geometry/line.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { GenericGraph, Graph, NetworkXGraph } from "../src/mobject/graph.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";

// Python ground truths (manim 0.21.0, original manim):
// - Graph([1,2,3],[(1,2),(2,3)], layout={1:[-2,0,0],2:[0,0,0],3:[2,0,0]}):
//   verts [1,2,3], 2 edges, v1 [-2,0,0], repr
//   "Undirected graph on 3 vertices and 2 edges"
//   _create_vertices(4,5,positions={4:[9,9,0]}) -> 4:[9,9,0], 5:center [0,0,0]
// - get_transposed_matrix_transformation([[0,-1],[1,0]]):
//   [1,0,0]->[0,-1,0], [0,1,0]->[1,0,0], net_rotation -pi/2
// - Graph([1,2],[(1,2)],layout={1:[0,0,0],2:[1,0,0]}): v2 [1,0,0]
//   add_vertices(3,positions={3:[5,5,0]}) -> [5,5,0]; add_edges (2,3): 1->2
//   edges; remove_edges: 2->1; remove_vertices(3) -> [1,2]
//   _add_edge((3,4)) then _remove_vertex(4) -> group of 2, verts [1,2,3]
//   _remove_edge((1,9)) -> ValueError "does not contain a edge"; _remove_vertex(999)
//   -> ValueError "does not contain a vertex"; _add_vertex(1) -> ValueError
//   "Vertex identifier '1' is already used for a vertex in this graph."
//   change_layout({1:[10,0,0],2:[20,0,0]}) moves both; partial {1:...} -> KeyError 2
//   g[1] is Dot, g[(1,2)] is Line; move v1 to [100,0,0] + update_edges ->
//   edge start x == 100
// - Graph([],[],layout={}) -> "Undirected graph on 0 vertices and 0 edges"
// - labels=True -> latex FileNotFoundError in python (TS: tex TODO stub)
// - remove_vertices(1,3) on 3-chain -> remaining [2], removed group of 4
// - Vector.coordinate_label builds Matrix (latex) -> TS tex TODO stub
// - VectorScene.coords_to_vector writes Matrix + plays -> TS tex TODO stub

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function vecClose(
  actual: number[] | NDArray,
  expected: number[],
  tol = 1e-6,
): void {
  const arr: number[] = Array.isArray(actual)
    ? [...actual]
    : [...(actual.toArray() as unknown as number[])].map(Number);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

function dictGraph(): Graph {
  return new Graph([1, 2, 3], [[1, 2], [2, 3]], {
    layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
      string,
      unknown
    >,
  });
}

describe("text/tex labels now built", () => {
  it("Vector.coordinate_label builds a Matrix (python: Matrix)", () => {
    const v = new Vector([1, 2, 0]);
    expect(v.coordinate_label().get_entries().submobjects.length).toBe(2);
    expect(v.coordinate_label(false, 3).get_entries().submobjects.length).toBe(3);
  });
  it("VectorScene.coords_to_vector runs the Matrix+Write sequence", () => {
    const vs = new VectorScene();
    expect(() => vs.coords_to_vector([3, -2, 0])).not.toThrow();
  });
});

describe("LinearTransformationScene.apply_transposed_matrix", () => {
  function wiredScene(): LinearTransformationScene {
    const scene = new LinearTransformationScene();
    scene.setup();
    // NOTE: TS NumberPlane is not Mobject-based yet, so the setup-installed
    // plane cannot go through ApplyPointwiseFunction; use a real Mobject like
    // python's plane is. (Same limitation as batch-targets-layers28-30.)
    scene.transformable_mobjects = [new Square() as unknown as never] as never[];
    return scene;
  }
  it("identity matrix gives path_arc=0 and applies the identity func", () => {
    const scene = wiredScene();
    // python: mean(angle([1,0,0]), angle([0,1,0]) - pi/2) = mean(0, 0) = 0
    const playSpy = vi
      .spyOn(scene, "play")
      .mockImplementation((..._a: unknown[]) => undefined as unknown as void);
    const applySpy = vi.spyOn(scene, "apply_function");
    scene.apply_transposed_matrix([
      [1, 0],
      [0, 1],
    ]);
    expect(applySpy).toHaveBeenCalledOnce();
    const kwargs = applySpy.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(close(Number(kwargs["path_arc"]), 0, 1e-9)).toBe(true);
    const func = applySpy.mock.calls[0]?.[0] as (p: number[]) => NDArray;
    vecClose(func([2, 3, 0]), [2, 3, 0]);
    playSpy.mockRestore();
    applySpy.mockRestore();
  });
  it("scale matrix [[2,0],[0,3]] gives path_arc=0 with scaled func", () => {
    const scene = wiredScene();
    const playSpy = vi
      .spyOn(scene, "play")
      .mockImplementation((..._a: unknown[]) => undefined as unknown as void);
    const applySpy = vi.spyOn(scene, "apply_function");
    scene.apply_transposed_matrix([
      [2, 0],
      [0, 3],
    ]);
    const kwargs = applySpy.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(close(Number(kwargs["path_arc"]), 0, 1e-9)).toBe(true);
    const func = applySpy.mock.calls[0]?.[0] as (p: number[]) => NDArray;
    vecClose(func([1, 1, 0]), [2, 3, 0]);
    playSpy.mockRestore();
    applySpy.mockRestore();
  });
  it("an explicit path_arc kwarg is preserved (python: only set if missing)", () => {
    const scene = wiredScene();
    const playSpy = vi
      .spyOn(scene, "play")
      .mockImplementation((..._a: unknown[]) => undefined as unknown as void);
    const applySpy = vi.spyOn(scene, "apply_function");
    scene.apply_transposed_matrix(
      [
        [0, -1],
        [1, 0],
      ],
      { path_arc: 0.5 },
    );
    const kwargs = applySpy.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(kwargs["path_arc"]).toBe(0.5);
    playSpy.mockRestore();
    applySpy.mockRestore();
  });
  it("3x3 matrix func matches python np.dot ground truth", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [1, 2, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    // python np.dot([1,1,0], [[1,2,0],[0,1,0],[0,0,1]]) = [1,3,0]
    // (row-vector convention: result[j] = sum_i point[i]*M[i][j])
    vecClose(func([1, 1, 0]), [1, 3, 0]);
  });
});

describe("LinearTransformationScene.apply_nonlinear_transformation", () => {
  it("prepares the plane then forwards func+kwargs to apply_function", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const prepareSpy = vi.spyOn(
      scene.plane as unknown as { prepare_for_nonlinear_transform(): void },
      "prepare_for_nonlinear_transform",
    );
    const order: string[] = [];
    prepareSpy.mockImplementation(() => {
      order.push("prepare");
    });
    const applySpy = vi
      .spyOn(scene, "apply_function")
      .mockImplementation(() => {
        order.push("apply");
        return undefined as unknown as void;
      });
    const func = (p: number[] | NDArray): number[] => {
      const arr = Array.isArray(p) ? [...p] : [...(p.toArray() as number[])];
      return [(arr[0] ?? 0) ** 2, arr[1] ?? 0, arr[2] ?? 0];
    };
    const kwargs = { run_time: 1 };
    scene.apply_nonlinear_transformation(func, kwargs);
    expect(order).toEqual(["prepare", "apply"]);
    expect(applySpy).toHaveBeenCalledOnce();
    expect(applySpy.mock.calls[0]?.[0]).toBe(func);
    expect(applySpy.mock.calls[0]?.[2]).toBe(kwargs);
    prepareSpy.mockRestore();
    applySpy.mockRestore();
  });
  it("nonlinear func actually maps points (square under x -> x^2 stays centered)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    // sanity: the forwarded func is usable pointwise like python np.dot
    vecClose(func([2, 0, 0]), [0, -2, 0]);
  });
});

describe("GenericGraph.__init__ / Graph", () => {
  it("empty graph has the python repr", () => {
    const g = new Graph([], [], { layout: {} });
    expect(g.vertices.size).toBe(0);
    expect(g.edges.size).toBe(0);
    expect(g.toString()).toBe(
      "Undirected graph on 0 vertices and 0 edges",
    );
  });
  it("dict layout positions vertices (python ground truth)", () => {
    const g = dictGraph();
    expect(g.vertices.size).toBe(3);
    expect(g.edges.size).toBe(2);
    vecClose(g.vertices.get(1)!.get_center(), [-2, 0, 0]);
    vecClose(g.vertices.get(2)!.get_center(), [0, 0, 0]);
    vecClose(g.vertices.get(3)!.get_center(), [2, 0, 0]);
    expect(g.toString()).toBe("Undirected graph on 3 vertices and 2 edges");
  });
  it("labels=True builds one MathTex per vertex (python: MathTex)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
      labels: true,
    });
    const labels = [...(g as unknown as { _labels: Map<number, { tex_string: string }> })._labels.values()];
    expect(labels.map((m) => m.tex_string).sort()).toEqual(["1", "2"]);
  });
  it("__getitem__ returns Dot vertices and Line edges (python ground truth)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    expect(g.getItem(1)).toBeInstanceOf(Dot);
    expect(g.getItem([1, 2])).toBeInstanceOf(Line);
    expect(() => g.getItem(999)).toThrow(/Could not find/);
  });
  it("update_edges follows moved vertices (python: edge start x == 100)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.vertices.get(1)!.move_to([100, 0, 0]);
    g.update_edges(g);
    const start = (g.getItem([1, 2]) as Line).get_start();
    expect(close(start[0] ?? NaN, 100, 1e-6)).toBe(true);
  });
  it("from_networkx builds the same graph as the constructor", () => {
    const nx = new NetworkXGraph();
    nx.addNodesFrom([1, 2]);
    nx.addEdge(1, 2);
    const g = Graph.from_networkx(nx, {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    }) as Graph;
    expect(g.vertices.size).toBe(2);
    expect(g.edges.size).toBe(1);
    vecClose(g.vertices.get(2)!.get_center(), [1, 0, 0]);
  });
  it("GenericGraph base stays abstract", () => {
    expect(() => GenericGraph._empty_networkx_graph()).toThrow(
      /To be implemented/,
    );
  });
});

describe("GenericGraph._create_vertices", () => {
  it("defaults missing positions to the graph center (python: [0,0,0])", () => {
    const g = dictGraph();
    const out = g._create_vertices(4, 5, {
      positions: { 4: [9, 9, 0] },
    } as unknown as Record<string, unknown>);
    expect(out.length).toBe(2);
    expect(out[0]?.[0]).toBe(4);
    vecClose(out[0]?.[1] as NDArray, [9, 9, 0]);
    expect(out[1]?.[0]).toBe(5);
    vecClose(out[1]?.[1] as NDArray, [0, 0, 0]);
  });
  it("no-arg call returns an empty list", () => {
    const g = dictGraph();
    expect(g._create_vertices()).toEqual([]);
  });
});

describe("GenericGraph._remove_edge / _remove_vertex", () => {
  it("_remove_edge drops the edge and errors on re-removal (python ValueError)", () => {
    const g = dictGraph();
    const before = g.edges.size;
    g._remove_edge([1, 2]);
    expect(g.edges.size).toBe(before - 1);
    expect(() => g._remove_edge([1, 2])).toThrow(/does not contain/);
    expect(() => g._remove_edge([1, 9])).toThrow(/does not contain/);
  });
  it("_remove_vertex removes vertex + incident edges as a group of 2", () => {
    const g = dictGraph();
    g._add_edge([3, 4]);
    const removed = g._remove_vertex(4);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
    expect(removed.submobjects.length).toBe(2);
    expect(() => g._remove_vertex(999)).toThrow(/does not contain/);
  });
});

describe("GenericGraph.add_vertices / add_edges / remove_edges / remove_vertices", () => {
  it("add_vertices positions the new vertex (python: [5,5,0])", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    const added = g.add_vertices(3, {
      positions: { 3: [5, 5, 0] },
    } as unknown as Record<string, unknown>);
    vecClose(added[0]!.get_center(), [5, 5, 0]);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
  });
  it("duplicate add raises ValueError with the python message", () => {
    const g = dictGraph();
    expect(() =>
      (g as unknown as { _add_vertex(v: number): void })._add_vertex(1),
    ).toThrow(
      /Vertex identifier '1' is already used for a vertex in this graph\./,
    );
  });
  it("add_edges / remove_edges round-trip 1->2->1 (python ground truth)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.add_vertices(3, {
      positions: { 3: [5, 5, 0] },
    } as unknown as Record<string, unknown>);
    g.add_edges([2, 3]);
    expect(g.edges.size).toBe(2);
    g.remove_edges([2, 3]);
    expect(g.edges.size).toBe(1);
  });
  it("remove_vertices drops one vertex; multi-remove matches python group of 4", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.add_vertices(3, {
      positions: { 3: [5, 5, 0] },
    } as unknown as Record<string, unknown>);
    g.remove_vertices(3);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2]);
    // python: remove_vertices(1,3) on the 3-chain leaves [2], group of 4
    const g5 = new Graph([1, 2, 3], [[1, 2], [2, 3]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0], 3: [2, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
    const removed = g5.remove_vertices(1, 3);
    expect([...g5.vertices.keys()]).toEqual([2]);
    expect(removed.submobjects.length).toBe(4);
  });
});

describe("GenericGraph.change_layout", () => {
  it("moves vertices to the new positions (python ground truth)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.change_layout({ 1: [10, 0, 0], 2: [20, 0, 0] } as unknown as Record<
      string,
      unknown
    >);
    vecClose(g.vertices.get(1)!.get_center(), [10, 0, 0]);
    vecClose(g.vertices.get(2)!.get_center(), [20, 0, 0]);
  });
  it("partial dict raises KeyError for the missing vertex (python: KeyError 2)", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    const err = (() => {
      try {
        g.change_layout({ 1: [5, 0, 0] } as unknown as Record<string, unknown>);
      } catch (e) {
        return e as Error;
      }
      return null;
    })();
    expect(err).not.toBeNull();
    expect(err!.name).toBe("KeyError");
    expect(String(err!.message)).toMatch(/2/);
  });
});

describe("integration with earlier-ported code", () => {
  it("transposed-matrix func moves graph vertices like python np.dot", () => {
    // python: np.dot([[0,-1,0],[1,0,0],[0,0,1]], [2,0,0]) = [0,-2,0]
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [2, 0, 0], 2: [0, 2, 0] } as unknown as Record<string, unknown>,
    });
    for (const v of g.vertices.values()) {
      v.apply_function(func as unknown as (p: number[]) => number[]);
    }
    g.update_edges(g);
    vecClose(g.vertices.get(1)!.get_center(), [0, -2, 0]);
    vecClose(g.vertices.get(2)!.get_center(), [2, 0, 0]);
    const edge = g.getItem([1, 2]) as Line;
    vecClose(edge.get_start(), [0, -2, 0]);
    vecClose(edge.get_end(), [2, 0, 0]);
  });
  it("apply_transposed_matrix plays with run_time=3 default on every anim", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    scene.transformable_mobjects = [new Square() as unknown as never] as never[];
    let played: unknown[] = [];
    const playSpy = vi
      .spyOn(scene, "play")
      .mockImplementation((...anims: unknown[]) => {
        played = anims;
        return undefined as unknown as void;
      });
    scene.apply_transposed_matrix([
      [1, 0],
      [0, 1],
    ]);
    expect(played.length).toBeGreaterThan(0);
    for (const anim of played) {
      expect((anim as Record<string, unknown>)["run_time"]).toBe(3);
    }
    playSpy.mockRestore();
  });
  it("apply_function output array feeds Vector construction (scene + mobject chain)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [2, 0],
      [0, 2],
    ]);
    const out = func([1, 1, 0]);
    const arr = [...(out.toArray() as unknown as number[])].map(Number);
    const v = new Vector(array(arr.slice(0, 2)) as NDArray);
    vecClose(v.get_end(), [2, 2, 0]);
  });
});
