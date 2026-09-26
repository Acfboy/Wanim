// P1-E: pure-TS graph automatic layouts.
//
// networkx's spring/random layouts are not bit-deterministic, so only
// structural invariants are asserted here (distinct finite vertices, fitted
// scale, nonzero edge lengths). Deterministic layouts are seeded through the
// layout config.
import { describe, expect, it } from "vitest";
import {
  Graph,
  NetworkXGraph,
  _determine_graph_layout,
  _layouts,
} from "../src/mobject/graph.js";
import type { LayoutName } from "../src/mobject/graph.js";
import type { Mobject } from "../src/mobject/mobject.js";

function positions(layout: Map<unknown, number[]> | Map<unknown, { toArray: () => number[] }>): number[][] {
  return [...layout.values()].map((value) => {
    const row = typeof (value as { toArray?: unknown }).toArray === "function"
      ? (value as { toArray: () => number[] }).toArray()
      : (value as number[]);
    return [...row];
  });
}

function graphOf(): NetworkXGraph {
  const g = new NetworkXGraph();
  g.addEdgesFrom([[1, 2], [2, 3], [3, 4], [4, 1]]);
  return g;
}

function expectFiniteDistinct(points: number[][]): void {
  expect(points.length).toBeGreaterThan(0);
  for (const point of points) {
    for (const value of point) expect(Number.isFinite(value)).toBe(true);
  }
  const seen = new Set(points.map((point) => point.map((v) => v.toFixed(6)).join(",")));
  expect(seen.size).toBe(points.length);
}

describe("graph automatic layouts", () => {
  const names: LayoutName[] = ["circular", "random", "shell", "spiral", "spectral", "spring"];

  for (const name of names) {
    it(`${name} produces n finite distinct vertices`, () => {
      const result = _layouts[name](graphOf(), 2, { seed: 1 });
      const points = positions(result as Map<unknown, number[]>);
      expectFiniteDistinct(points);
      expect(points.length).toBe(4);
      // Fitted to layout_scale 2 (max |coordinate| <= scale for most layouts).
      const maxAbs = Math.max(...points.flat().map((v) => Math.abs(v)));
      expect(maxAbs).toBeLessThanOrEqual(2.0001);
    });
  }

  it("circular places vertices on radius 2 around the origin", () => {
    const result = _layouts.circular(graphOf(), 2, {});
    for (const [, value] of result) {
      const point = value.toArray() as number[];
      const radius = Math.hypot(point[0] ?? 0, point[1] ?? 0);
      expect(Math.abs(radius - 2)).toBeLessThan(1e-6);
    }
  });

  it("random/spring are reproducible with the same seed", () => {
    for (const name of ["random", "spring"] as const) {
      const a = positions(_layouts[name](graphOf(), 2, { seed: 42 }) as Map<unknown, number[]>);
      const b = positions(_layouts[name](graphOf(), 2, { seed: 42 }) as Map<unknown, number[]>);
      expect(a).toEqual(b);
    }
  });

  it("default layout='spring' works on new Graph and yields nonzero edges", () => {
    const g = new Graph([1, 2, 3, 4]);
    g.change_layout("spring");
    const centers = [...g.vertices.values()].map((mob: Mobject) => mob.get_center());
    const distinct = new Set(centers.map((c) => c.map((v) => v.toFixed(6)).join(",")));
    expect(distinct.size).toBe(4);
    for (const edge of g.edges.values()) {
      const length = (edge as unknown as { get_length?: () => number }).get_length?.() ?? 0;
      expect(length).toBeGreaterThan(0);
    }
  });

  it("_determine_graph_layout still passes dict/function through", () => {
    const g = graphOf();
    const placement = new Map<unknown, unknown>([[1, [0, 0, 0]]]);
    expect(_determine_graph_layout(g, placement as never)).toBe(placement);
    const custom = _determine_graph_layout(g, ((graph: NetworkXGraph) => {
      const m = new Map<unknown, number[]>();
      for (const node of graph.nodes.keys()) m.set(node, [1, 0, 0]);
      return m;
    }) as never);
    expect(custom.size).toBe(4);
  });

  it("unknown layouts still raise Python's ValueError", () => {
    expect(() => _determine_graph_layout(graphOf(), "nonexistent_xyz" as never)).toThrow(/neither a recognized layout/);
  });
});
