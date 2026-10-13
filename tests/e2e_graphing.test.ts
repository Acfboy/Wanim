// End-to-end 2D graphing differential tests: wanim vs. the split manim
// (original manim) ground truth. All expected values were captured with:
//
//   original manim
//   from manim import *; ...
//
// See tests/e2e_graphing.test.ts header notes per assertion. The original manim has no
// `latex` binary, so text-bearing paths (BarChart y-axis numbers / bar_names,
// ComplexPlane DecimalNumber labels) are covered structurally instead of by
// point-exact comparison.
import { describe, expect, it } from "vitest";
import { Scene } from "../src/scene/scene.js";
import { Create } from "../src/animation/creation.js";
import {
  Axes,
  ComplexPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { pointRows } from "../src/utils/arrays.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import { Graph, DiGraph } from "../src/mobject/graph.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { BLUE, GREEN } from "../src/utils/color/core.js";
import type { Mobject } from "../src/mobject/mobject.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function rows(value: unknown): number[][] {
  const raw = (value as { toArray?: () => unknown }).toArray
    ? (value as { toArray: () => unknown }).toArray()
    : value;
  const arr = raw as unknown[];
  return Array.isArray(arr[0])
    ? (arr as number[][]).map((row) => row.map(Number))
    : [(arr as number[]).map(Number)];
}

function centerOf(mob: Mobject): number[] {
  return pointRows(mob.get_center(), 3)[0] ?? [0, 0, 0];
}

describe("NumberLine include_tip (original manim: line shortened to tip.base)", () => {
  // Python:
  //   nl = NumberLine(x_range=[-4,4,1], length=8, include_tip=True)
  //   nl.points == [[-4,0,0],[-1.45,0,0],[1.1,0,0],[3.65,0,0]]
  //   nl.get_start() == [-4,0,0], nl.get_end() == [4,0,0]
  //   nl.tip.get_stroke_width() == 2.0  (line stroke inherited by the tip)
  it("shortens the baseline to tip.base but keeps logical endpoints", () => {
    const nl = new NumberLine({ x_range: [-4, 4, 1], length: 8, include_tip: true, include_ticks: true });
    const pts = rows(nl.points).map((row) => row.map((v) => Math.round(v * 1e6) / 1e6 + 0));
    expect(pts).toEqual([
      [-4, 0, 0],
      [-1.45, 0, 0],
      [1.1, 0, 0],
      [3.65, 0, 0],
    ]);
    const start = nl.get_start();
    const end = nl.get_end();
    expect(close(start[0] ?? NaN, -4)).toBe(true);
    expect(close(end[0] ?? NaN, 4)).toBe(true);
    expect(close(nl.get_unit_size(), 1)).toBe(true);
    // Tick mapping is unaffected by the shortening (original manim n2p(3) == [3,0,0]).
    expect(close(rows(nl.n2p(3))[0]?.[0] ?? NaN, 3, 1e-6)).toBe(true);
  });

  it("gives the tip the line stroke color/width (original manim: width 2, white)", () => {
    const nl = new NumberLine({ x_range: [-4, 4, 1], length: 8, include_tip: true });
    const tip = (nl as unknown as { tip: VMobject }).tip;
    expect(tip).toBeInstanceOf(VMobject);
    expect(close(tip.get_stroke_width(), 2)).toBe(true);
    expect(tip.get_stroke_color()?.to_hex()).toBe("#FFFFFF");
    expect(nl.has_tip()).toBe(true);
  });

  it("Axes default tip shortens the x-axis (original manim: endpoints -6..6, base 5.65)", () => {
    const a = new Axes();
    const pts = rows(a.get_x_axis().points);
    expect(pts.length).toBe(4);
    expect(close(pts[0]?.[0] ?? NaN, -6, 1e-6)).toBe(true);
    expect(close(pts[3]?.[0] ?? NaN, 5.65, 1e-6)).toBe(true);
    expect(close(a.get_x_axis().get_end()[0] ?? NaN, 6, 1e-6)).toBe(true);
    expect(close(a.get_x_axis().get_start()[0] ?? NaN, -6, 1e-6)).toBe(true);
  });

  it("tips=False leaves the full-length baseline (original manim: -6..6)", () => {
    const a = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4, { include_tip: false });
    const pts = rows(a.get_x_axis().points);
    expect(close(pts[3]?.[0] ?? NaN, 3, 1e-6)).toBe(true);
    expect(close(a.get_x_axis().get_end()[0] ?? NaN, 3, 1e-6)).toBe(true);
  });
});

describe("Axes calculus helpers (original manim values)", () => {
  const axes = new Axes([-4, 4, 1], [-2, 4, 1], 8, 5);
  const graph = axes.plot((x) => 0.1 * x * x, [-4, 4, 1]);

  it("slope_of_tangent / angle_of_tangent match the original manim finite difference", () => {
    // original manim: slope_of_tangent(2, g) == 0.400000002220446,
    //       slope_of_tangent(0, g) == 1e-09,
    //       angle_of_tangent(2, g) == 0.3805063790265425
    expect(close(axes.slope_of_tangent(2, graph), 0.400000002220446, 1e-9)).toBe(true);
    expect(Math.abs(axes.slope_of_tangent(0, graph))).toBeLessThan(1e-7);
    expect(close(axes.angle_of_tangent(2, graph), 0.3805063790265425, 1e-9)).toBe(true);
  });

  it("input_to_graph_point matches the original manim", () => {
    // original manim: i2gp(2) == [2, -0.5, 0]
    const p = rows(axes.input_to_graph_point(2, graph))[0] ?? [];
    expect(close(p[0] ?? NaN, 2, 1e-9)).toBe(true);
    expect(close(p[1] ?? NaN, -0.5, 1e-9)).toBe(true);
  });

  it("get_graph_label (VMobject label) matches the original manim placement", () => {
    // original manim: get_graph_label(g, label=Dot(), x_val=2, direction=RIGHT, buff=.25)
    //       -> center [2.33, -0.5, 0]
    const lab = axes.get_graph_label(graph, new Dot(), 2, [1, 0, 0], 0.25);
    const c = centerOf(lab);
    expect(close(c[0] ?? NaN, 2.33, 1e-4)).toBe(true);
    expect(close(c[1] ?? NaN, -0.5, 1e-4)).toBe(true);
    // original manim: default x_val search -> [4.33, 0.5, 0]
    const lab2 = axes.get_graph_label(graph, new Dot());
    const c2 = centerOf(lab2);
    expect(close(c2[0] ?? NaN, 4.33, 1e-3)).toBe(true);
    expect(close(c2[1] ?? NaN, 0.5, 1e-3)).toBe(true);
  });

  it("get_area builds a closed Polygon matching the original manim", () => {
    // original manim: area = get_area(g, [0,2], [BLUE,GREEN], .5)
    //       len(area.points) == 56, center [1, -0.666667, 0]
    const area = axes.get_area(graph, [0, 2], [BLUE, GREEN], 0.5);
    const pts = rows(area.points);
    expect(pts.length).toBe(56);
    const c = centerOf(area);
    expect(close(c[0] ?? NaN, 1.0, 1e-5)).toBe(true);
    expect(close(c[1] ?? NaN, -0.666667, 1e-5)).toBe(true);
  });

  it("get_vertical_lines_to_graph returns the requested count (original manim: 8)", () => {
    const lines = axes.get_vertical_lines_to_graph(graph, [-2, 2], 8);
    expect(lines).toBeInstanceOf(VGroup);
    expect(lines.submobjects.length).toBe(8);
  });

  it("get_T_label (VMobject label) has label+triangle+line (original manim: 3)", () => {
    const tl = axes.get_T_label(2, graph, new Dot());
    expect(tl).toBeInstanceOf(VGroup);
    expect(tl.submobjects.length).toBe(3);
    // original manim: first submobject (the label) center == [2, -1.413333, 0]
    const c = centerOf(tl.submobjects[0] as Mobject);
    expect(close(c[0] ?? NaN, 2, 1e-4)).toBe(true);
    expect(close(c[1] ?? NaN, -1.413333, 1e-4)).toBe(true);
  });

  it("get_axis_labels accepts VMobject labels", () => {
    const a = new Axes([-4, 4, 1], [-2, 4, 1], 8, 5);
    a.get_axis_labels(new Square({ side_length: 0.3 }), new Dot());
    const labelGroup = a.get_axes();
    expect(labelGroup).toBeInstanceOf(VGroup);
  });
});

describe("Scene integration (axes are Mobjects)", () => {
  it("Scene.add(axes) and Create(axes) work", () => {
    const scene = new Scene();
    const axes = new Axes();
    scene.add(axes as unknown as Mobject);
    expect(scene.mobjects.includes(axes as unknown as Mobject)).toBe(true);
    const scene2 = new Scene();
    const axes2 = new Axes();
    scene2.play(new Create(axes2 as unknown as VMobject));
    expect(scene2.mobjects.includes(axes2 as unknown as Mobject)).toBe(true);
  });
});

describe("Graph layouts (original manim networkx)", () => {
  it("circular positions match the original manim exactly", () => {
    // original manim: {1:[2,0,0], 2:[0,2,0], 3:[-2,0,0], 4:[0,-2,0]}
    const g = new Graph([1, 2, 3, 4], [[1, 2], [2, 3], [3, 4], [4, 1]], { layout: "circular" });
    const expected: Array<[number, number[]]> = [
      [1, [2, 0, 0]],
      [2, [0, 2, 0]],
      [3, [-2, 0, 0]],
      [4, [0, -2, 0]],
    ];
    for (const [v, want] of expected) {
      const c = centerOf(g.vertices.get(v) as Mobject);
      expect(close(c[0] ?? NaN, want[0] ?? NaN, 1e-6)).toBe(true);
      expect(close(c[1] ?? NaN, want[1] ?? NaN, 1e-6)).toBe(true);
    }
  });

  it("tree positions match the original manim exactly (root/scaled siblings)", () => {
    // original manim: Graph([1,2,3,4], [(1,2),(1,3),(1,4)], layout='tree', root_vertex=1)
    //       {1:[0,1,0], 2:[2,-1,0], 3:[0,-1,0], 4:[-2,-1,0]}
    const g = new Graph([1, 2, 3, 4], [[1, 2], [1, 3], [1, 4]], {
      layout: "tree",
      root_vertex: 1,
    });
    const expected: Array<[number, number[]]> = [
      [1, [0, 1, 0]],
      [2, [2, -1, 0]],
      [3, [0, -1, 0]],
      [4, [-2, -1, 0]],
    ];
    for (const [v, want] of expected) {
      const c = centerOf(g.vertices.get(v) as Mobject);
      expect(close(c[0] ?? NaN, want[0] ?? NaN, 1e-6)).toBe(true);
      expect(close(c[1] ?? NaN, want[1] ?? NaN, 1e-6)).toBe(true);
    }
  });

  it("change_layout('tree') moves the vertices like the constructor", () => {
    const g = new Graph([1, 2, 3, 4], [[1, 2], [1, 3], [1, 4]], { layout: "circular" });
    g.change_layout("tree", 2, null, null, 1);
    const c4 = centerOf(g.vertices.get(4) as Mobject);
    expect(close(c4[0] ?? NaN, -2, 1e-6)).toBe(true);
    expect(close(c4[1] ?? NaN, -1, 1e-6)).toBe(true);
  });

  it("spring yields n distinct finite vertices with nonzero edges (structural)", () => {
    // Python's spring layout is not bit-reproducible in wanim (approximate
    // Fruchterman-Reingold); only invariants are asserted (cf. p1-graph-layouts).
    const g = new Graph([1, 2, 3, 4], [[1, 2], [2, 3], [3, 4], [4, 1]], {
      layout: "spring",
      layout_config: { seed: 7 },
    });
    const centers = [...g.vertices.values()].map((m) => centerOf(m));
    for (const c of centers) for (const v of c) expect(Number.isFinite(v)).toBe(true);
    const distinct = new Set(centers.map((c) => c.map((v) => v.toFixed(6)).join(",")));
    expect(distinct.size).toBe(4);
    for (const edge of g.edges.values()) {
      const length = (edge as unknown as { get_length?: () => number }).get_length?.() ?? 0;
      expect(length).toBeGreaterThan(0);
    }
  });

  it("DiGraph tree matches the original manim (same positions as Graph)", () => {
    const g = new DiGraph([1, 2, 3, 4], [[1, 2], [1, 3], [1, 4]], {
      layout: "tree",
      root_vertex: 1,
    });
    const c1 = centerOf(g.vertices.get(1) as Mobject);
    expect(close(c1[0] ?? NaN, 0, 1e-6)).toBe(true);
    expect(close(c1[1] ?? NaN, 1, 1e-6)).toBe(true);
  });
});

describe("BarChart (original manim geometry, text labels disabled)", () => {
  // original manim (with NumberLine.add_numbers disabled to avoid the missing latex):
  //   BarChart([1,2,3,4]).bars -> heights [1,2,3,4], width .6,
  //   centers [-1.5,-1.5],[-0.5,-1],[0.5,-.5],[1.5,0]
  it("adds the bars to the chart and matches the original manim geometry", () => {
    const bc = new BarChart([1, 2, 3, 4]);
    expect(bc.bars.submobjects.length).toBe(4);
    // The bars are real submobjects now (Python: add_to_back(self.bars)).
    expect(bc.submobjects.includes(bc.bars)).toBe(true);
    const expectedCenters = [
      [-1.5, -1.5],
      [-0.5, -1.0],
      [0.5, -0.5],
      [1.5, 0.0],
    ];
    bc.bars.submobjects.forEach((bar, i) => {
      const rect = bar as unknown as { width: number; height: number };
      expect(close(rect.height, i + 1, 1e-6)).toBe(true);
      expect(close(rect.width, 0.6, 1e-6)).toBe(true);
      const c = centerOf(bar as Mobject);
      expect(close(c[0] ?? NaN, expectedCenters[i]?.[0] ?? NaN, 1e-6)).toBe(true);
      expect(close(c[1] ?? NaN, expectedCenters[i]?.[1] ?? NaN, 1e-6)).toBe(true);
    });
  });

  it("bar_names path builds one Tex label per bar", () => {
    const bc = new BarChart([1, 2, 3, 4], { bar_names: ["a", "b", "c", "d"] });
    const labels = bc.x_axis.labels;
    expect(labels).not.toBeNull();
    expect(labels!.submobjects.length).toBe(4);
    expect(
      labels!.submobjects.map(
        (m) => (m as unknown as { tex_string: string }).tex_string,
      ),
    ).toEqual(["\\text{a}", "\\text{b}", "\\text{c}", "\\text{d}"]);
  });
});

describe("ComplexPlane.get_coordinate_labels (anchors from original manim)", () => {
  it("returns one label per requested number anchored on the right axis", () => {
    // original manim anchors (unit=1 plane): n2p(1)=[1,0], n2p(2)=[2,0],
    // n2p(1j)=[0,1], n2p(-2j)=[0,-2]. Labels are DecimalNumbers; their
    // centers sit next_to those anchors (label_direction DR/buff), so assert
    // the relevant coordinate is within 0.5 and the number is correct.
    const cp = new ComplexPlane([-4, 4, 1], [-4, 4, 1], 8, 8);
    const labels = cp.get_coordinate_labels(1, 2, { re: 0, im: 1 }, { re: 0, im: -2 });
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(4);
    const numbers = labels.submobjects.map((l) => Number((l as unknown as { number: number }).number));
    expect(numbers).toEqual([1, 2, 1, -2]);
    const centers = labels.submobjects.map((l) => centerOf(l as Mobject));
    expect(Math.abs((centers[0]?.[0] ?? NaN) - 1)).toBeLessThan(0.5);
    expect(Math.abs((centers[1]?.[0] ?? NaN) - 2)).toBeLessThan(0.5);
    expect(Math.abs((centers[2]?.[1] ?? NaN) - 1)).toBeLessThan(0.5);
    expect(Math.abs((centers[3]?.[1] ?? NaN) - -2)).toBeLessThan(0.5);
  });
});
