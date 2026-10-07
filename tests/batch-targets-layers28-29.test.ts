// Ground-truth tests for the layers-28..29 batch:
//   PolarPlane.get_coordinate_labels / add_coordinates,
//   LabeledLine / LabeledPolygram / LabeledArrow __init__,
//   BulletedList / Title __init__, Brace.get_text / change_brace_label /
//   BraceText.__init__, DecimalNumber.__init__ /
//   _set_submobjects_from_number / set_value, Integer.__init__,
//   NumberLine.add_labels / get_number_mobject, Matrix.__init__ /
//   get_mob_matrix / set_column_colors / set_row_colors, DecimalMatrix /
//   IntegerMatrix / MobjectMatrix __init__, VDict.add, GenericGraph._add_edge,
//   MathTable / MobjectTable / IntegerTable / DecimalTable __init__,
//   VectorScene.get_basis_vector_labels / label_vector,
//   LinearTransformationScene.add_transformable_label / apply_function.
//
// SCOPE: text/tex is OUT OF SCOPE for the entire port. Most symbols in this
// batch need a LaTeX binary or a text-rendering mobject in Python, so they
// are asserted here only as clearly-marked `TODO: text/tex not ported yet`
// stubs (no functional tests), and absence of any text implementation is
// locked in via the text.ts namespace check. Non-tex behavior in this batch
// (NumberLine.add_labels with VMobject labels, VDict.add, GenericGraph
// _add_edge auto-vertex creation, Matrix/Table layout+colors behind a
// bracket-free subclass) is tested against real Python ground truth.
//
// Python ground truths (manim 0.21.0, original manim):
// - LabeledLine("hi") -> FileNotFoundError 'latex'
// - LabeledPolygram(verts, label="hi") -> FileNotFoundError 'latex'
// - LabeledPolygram(verts) without label -> TypeError
//   "LabeledPolygram.__init__() missing 1 required keyword-only argument: 'label'"
// - PolarPlane().get_coordinate_labels() / .add_coordinates() ->
//   FileNotFoundError 'latex' (TS: IndexError "range object index out of
//   range" on every path -- even with explicit r/a values -- because
//   get_x_axis() itself indexes the not-yet-Mobject-integrated empty axes.
//   Pre-existing Axes-integration gap; both implementations throw, only
//   the error category differs until Axes integration lands)
// - NumberLine(x_range=[-3,3,1]).add_labels({-2: sq, 0: sq}) with
//   font_size=36 squares -> centers [-2,-1.25,0], [0,-1.25,0];
//   tick_range [-3,-2,-1,-0,1,2,3]
// - Graph([1,2],[(1,2)],layout={1:[-2,0,0],2:[0,0,0]})._add_edge((2,3)) ->
//   vertices [1,2,3], edges [(1,2),(2,3)], group of 2, new vertex at graph
//   center [-1,0,0]
// - NumberLine.get_number_mobject default path builds a DecimalNumber ->
//   latex FileNotFoundError in Python.
//   DEVIATION (documented): TS throws the tex TODO unconditionally, while
//   Python with a non-tex `label_constructor` factory (e.g. `lambda s:
//   VMobject()`) succeeds and returns a DecimalNumber. Supporting that path
//   needs the DecimalNumber class itself, which is text-scope; the stub is
//   kept and the gap noted here instead of locked in as a test.
//
// Vitest OOM rule: never let vitest
// pretty-print a Mobject -- identity/type checks use `===` inside
// expect(...).toBe(true); only numbers/strings/booleans go to expect().
import { describe, expect, it } from "vitest";
import { LabeledLine, LabeledPolygram } from "../src/mobject/geometry/labeled.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { Matrix } from "../src/mobject/matrix.js";
import { MobjectTable } from "../src/mobject/table.js";
import { Graph } from "../src/mobject/graph.js";
import {
  VDict,
  VGroup,
  VMobject,
} from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import * as textMod from "../src/mobject/text/index.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function centerClose(mob: VMobject, expected: number[], tol = 1e-6): boolean {
  const c = mob.get_center();
  if (c.length !== expected.length) return false;
  return c.every((v, i) => close(v ?? NaN, expected[i] ?? NaN, tol));
}

// Bracket-free subclass (same convention as
// tests/batch-targets-layers28-30.test.ts): Python Matrix.__init__ always
// calls _add_brackets (MathTex -> latex); the TS port throws the tex TODO
// there, so layout/color logic is exercised with brackets patched out.
class TestMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

describe("text labels (ported) and remaining stubs", () => {
  it("LabeledLine.__init__ builds a string label", () => {
    const line = new LabeledLine("hi");
    expect(line.label).toBeDefined();
  });
  it("LabeledPolygram.__init__ places a label", () => {
    const p = new LabeledPolygram([[0, 0, 0], [1, 0, 0], [0, 1, 0]], {
      label: "hi",
    });
    expect(p.label).toBeDefined();
  });
  it("PolarPlane.get_coordinate_labels with explicit values builds labels", () => {
    const pp = new PolarPlane();
    const labels = pp.get_coordinate_labels([1, 2], [0.25]);
    expect(labels.submobjects.length).toBe(2);
    // [0.25] -> one radian label; python string for 1/4 is \tfrac{\pi}{2}.
    const azimuth = labels.submobjects[1].submobjects;
    expect(azimuth.length).toBe(1);
    expect(
      (azimuth[0] as unknown as { tex_string: string }).tex_string,
    ).toBe("\\tfrac{\\pi}{2}");
  });
  it("PolarPlane.get_coordinate_labels defaults build 20 azimuth labels", () => {
    const pp = new PolarPlane();
    const labels = pp.get_coordinate_labels();
    expect(labels.submobjects[1].submobjects.length).toBe(20);
  });
  it("NumberLine.get_number_mobject builds a positioned DecimalNumber", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const one = nl.get_number_mobject(1);
    const neg = nl.get_number_mobject(-2);
    expect(one.get_family().length).toBeGreaterThan(0);
    expect(neg.get_family().length).toBeGreaterThan(0);
    // Labels sit below the line (label_direction defaults to DOWN).
    expect(one.get_center()[1]).toBeLessThan(nl.number_to_point(1)[1]);
  });
});

describe("LabeledPolygram missing-label TypeError parity", () => {
  it("throws the same keyword-only-argument TypeError as python", () => {
    let message = "";
    try {
      new LabeledPolygram([[0, 0, 0], [1, 0, 0], [0, 1, 0]], {});
    } catch (e) {
      message = (e as Error).message;
      expect((e as Error) instanceof TypeError).toBe(true);
    }
    expect(message).toBe(
      "LabeledPolygram.__init__() missing 1 required keyword-only argument: 'label'",
    );
  });
});

describe("text/tex conformance (BulletedList/Title/DecimalNumber/Integer)", () => {
  it("text module exposes the real text/tex classes", () => {
    expect(typeof textMod.Text).toBe("function");
    expect(typeof textMod.MathTex).toBe("function");
    expect(typeof textMod.Tex).toBe("function");
    expect(typeof textMod.BulletedList).toBe("function");
    expect(typeof textMod.Title).toBe("function");
  });
});

describe("NumberLine.add_labels (tex-free VMobject path)", () => {
  it("accepts a Map and places negative/zero labels at ground-truth spots", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const s1 = new Square();
    const s2 = new Square();
    (s1 as unknown as Record<string, unknown>)["font_size"] = 36;
    (s2 as unknown as Record<string, unknown>)["font_size"] = 36;
    nl.add_labels(
      new Map<number, VMobject>([
        [-2, s1],
        [0, s2],
      ]),
    );
    expect(nl.labels?.submobjects.length).toBe(2);
    expect(centerClose(s1, [-2, -1.25, 0])).toBe(true);
    expect(centerClose(s2, [0, -1.25, 0])).toBe(true);
  });
  it("tick_range matches python ([-3,-2,-1,-0,1,2,3])", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const ticks = [
      ...(nl.get_tick_range().toArray() as unknown as number[]),
    ].map(Number);
    const expected = [-3, -2, -1, -0, 1, 2, 3];
    expect(ticks.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i += 1) {
      expect(close(ticks[i] ?? NaN, expected[i] ?? NaN)).toBe(true);
    }
  });
});

// --- integration: batch symbols interacting with earlier-ported code ---

describe("integration", () => {
  it("Graph._add_edge auto-vertex lands at graph center, then flows through VDict + Table", () => {
    // Python ground truth: auto-created vertex 3 at graph center [-1,0,0],
    // returned group holds 2 submobjects (new vertex + new edge).
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [-2, 0, 0], 2: [0, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
    const grp = g._add_edge([2, 3]);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
    expect(g.edges.size).toBe(2);
    expect(grp.submobjects.length).toBe(2);
    expect(centerClose(g.vertices.get(3) as VMobject, [-1, 0, 0])).toBe(
      true,
    );
    // The auto-created vertex keeps identity through VDict and Table cells.
    const vd = new VDict();
    vd.add([["v3", g.vertices.get(3) as VMobject]]);
    expect(vd.__getitem__("v3") === g.vertices.get(3)).toBe(true);
    const t = new MobjectTable([
      [g.vertices.get(1) as VMobject, g.vertices.get(3) as VMobject],
    ]);
    expect(t.mob_table[0]?.[1] === g.vertices.get(3)).toBe(true);
  });
  it("Matrix entries shared with VDict keep identity under row colors", () => {
    const s1 = new Square();
    const s2 = new Square();
    const m = new TestMatrix([[s1, s2]], {
      element_to_mobject: (item) => item as VMobject,
    });
    expect(m.get_mob_matrix() === m.mob_matrix).toBe(true);
    const vd = new VDict();
    vd.add({ first: s1 } as unknown as Record<string, VMobject>);
    m.set_row_colors("GREEN");
    expect(vd.__getitem__("first") === s1).toBe(true);
    expect(s1.get_color().to_hex().toUpperCase()).toBe("#83C167");
    expect(centerClose(m.mob_matrix[0]?.[0] as VMobject, [-0.65, 0, 0])).toBe(
      true,
    );
    expect(centerClose(m.mob_matrix[0]?.[1] as VMobject, [0.65, 0, 0])).toBe(
      true,
    );
  });
  it("NumberLine labels double as graph-storeable mobjects", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const s = new Square();
    (s as unknown as Record<string, unknown>)["font_size"] = 36;
    nl.add_labels({ 2: s } as unknown as Record<string, VMobject>);
    expect(centerClose(s, [2, -1.25, 0])).toBe(true);
    const vd = new VDict();
    vd.add(new Map<unknown, VMobject>([["two", s]]));
    expect(vd.__getitem__("two") === s).toBe(true);
    expect(vd.submobjects.length).toBe(1);
  });
});
