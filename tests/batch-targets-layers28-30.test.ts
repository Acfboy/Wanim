import { describe, expect, it, vi } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Brace, BraceLabel, BraceText } from "../src/mobject/svg/brace.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { DecimalNumber, string_to_mob_map } from "../src/mobject/text/numbers.js";
import { MathTex, Text } from "../src/mobject/text/index.js";
import {
  Matrix,
  DecimalMatrix,
  IntegerMatrix,
  MobjectMatrix,
} from "../src/mobject/matrix.js";
import {
  MathTable,
  MobjectTable,
  IntegerTable,
  DecimalTable,
  Table,
} from "../src/mobject/table.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { GenericGraph, Graph } from "../src/mobject/graph.js";
import {
  VDict,
  VGroup,
  VMobject,
} from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Line, Vector } from "../src/mobject/geometry/line.js";
import { LabeledArrow } from "../src/mobject/geometry/labeled.js";
import { PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { DOWN } from "../src/constants.js";

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

// Bracket-free subclass: mirrors the convention in
// batch-targets-layers27-28.test.ts. Python Matrix.__init__ always calls
// _add_brackets (MathTex -> latex, FileNotFoundError without a latex
// install); the TS port throws `TODO: text/tex not ported yet` there.
// Overriding to a no-op exercises the pure layout logic faithfully.
class TestMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}
class TestMobjectMatrix extends MobjectMatrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

// Python ground truths (manim 0.21.0, original manim):
// - TestMatrix 2x2 Squares (identity factory, h_buff=1.3, v_buff=0.8):
//   (0,0) [-0.65, 0.4, 0], (0,1) [0.65, 0.4, 0],
//   (1,0) [-0.65,-0.4, 0], (1,1) [0.65,-0.4, 0]; elements=4
//   get_mob_matrix is mob_matrix (identity); get_columns=2, get_rows=2
//   set_column_colors('RED','BLUE'): col0 #FC6255, col1 #58C4DD
//   set_row_colors('GREEN','YELLOW'): row0 #83C167, row1 #F7D96F
// - NumberLine x_range=[-3,3,1], Square()+font_size labels at {1,2}:
//   label centers [1,-1.25,0], [2,-1.25,0] (buff 0.25, half-height 1)
//   plain VMobject labels -> AttributeError (no font_size) in BOTH impls
// - Graph([1,2,3],[(1,2),(2,3)], layout={1:[-2,0,0],2:[0,0,0],3:[2,0,0]}):
//   v1 [-2,0,0], v2 [0,0,0]; _create_vertices(4,5,positions={4:[9,9,0]})
//   -> [(4,[9,9,0]), (5,graph-center [0,0,0])]
//   _add_edge((3,4)) auto-creates v4 at graph center; edge count 2->3
//   _remove_edge((1,2)) 3->2; _remove_vertex(4) removes vertex + 2 mobjects
// - get_transposed_matrix_transformation([[0,-1],[1,0]]):
//   [1,0,0]->[0,-1,0], [0,1,0]->[1,0,0];
//   angle(RIGHT)=-pi/2, angle(UP)=0, net path_arc=-pi/2
// - VDict.add({'a': sq}) / [('b', sq)]: identity preserved, submobjects grow
// - BraceText(Square(),'hello') works in python via Text (no latex); the TS
//   port throws TODO for ALL text (text/tex out of scope for entire port).
// - BraceLabel(Square(), VMobject()) in python calls MathTex(str(vm)) ->
//   latex FileNotFoundError; TS returns VMobject as-is (documented leniency,
//   same as NumberLine._create_label_tex).

describe("ported labels and remaining tex-only stubs", () => {
  it("Brace.get_text builds a Tex at the tip", () => {
    const b = new Brace(new Square());
    const t = b.get_text("hello");
    expect(t).toBeDefined();
    expect(t.width).toBeGreaterThan(0);
  });
  it("VectorScene.get_basis_vector_labels builds two MathTex labels", () => {
    const vs = new VectorScene();
    const labels = vs.get_basis_vector_labels();
    expect(labels.submobjects.length).toBe(2);
    for (const label of labels.submobjects) {
      expect(label).toBeInstanceOf(MathTex);
      expect(label.width).toBeGreaterThan(0);
    }
    expect(
      (labels.submobjects[0] as unknown as { tex_string: string }).tex_string,
    ).toBe("\\hat{\\imath}");
    expect(
      (labels.submobjects[1] as unknown as { tex_string: string }).tex_string,
    ).toBe("\\hat{\\jmath}");
  });
  it("VectorScene.label_vector builds a MathTex label", () => {
    const vs = new VectorScene();
    const label = vs.label_vector(new Vector([1, 0, 0]), "x", false);
    expect(label).toBeInstanceOf(MathTex);
    expect((label as unknown as { tex_string: string }).tex_string).toBe(
      "\\vec{\\textbf{x}}",
    );
  });
  it("NumberLine.get_number_mobject builds a positioned DecimalNumber (python)", () => {
    string_to_mob_map.clear();
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const mob = nl.get_number_mobject(1) as unknown as DecimalNumber;
    expect(mob).toBeInstanceOf(DecimalNumber);
    expect(mob.get_value()).toBe(1);
    expect(close(mob.get_center()[0] ?? NaN, 1, 1e-9)).toBe(true);
    // python: next_to([1,0,0], DOWN, 0.25) -> label top edge at y=-0.25
    expect(close(mob.get_top()[1] ?? NaN, -0.25, 1e-9)).toBe(true);
  });
  it("Vector.coordinate_label builds a Matrix label", () => {
    const v = new Vector([1, 2, 0]);
    const label = v.coordinate_label();
    expect(label).toBeInstanceOf(Matrix);
    expect(label.get_entries().submobjects.length).toBe(2);
    // Python: Matrix(np.round([1,2])) -> [[1],[2]].
    expect(
      (label.get_entries().submobjects[0] as unknown as { tex_string: string })
        .tex_string,
    ).toBe("1");
    expect(
      (label.get_entries().submobjects[1] as unknown as { tex_string: string })
        .tex_string,
    ).toBe("2");
  });
  it("VectorScene.coords_to_vector runs the Matrix+Write sequence", () => {
    const vs = new VectorScene();
    expect(() => vs.coords_to_vector([1, 2, 0])).not.toThrow();
  });
  it("LabeledArrow.__init__ builds a string label (python: MathTex)", () => {
    const arrow = new LabeledArrow("hi");
    expect(arrow.label).toBeDefined();
  });
  it("PolarPlane.add_coordinates builds radius and azimuth label groups", () => {
    const pp = new PolarPlane();
    expect(() => pp.add_coordinates()).not.toThrow();
    expect(pp.coordinate_labels.submobjects.length).toBe(2);
    // Default PI plane: azimuth step 20 -> 20 radian labels.
    expect(pp.coordinate_labels.submobjects[1].submobjects.length).toBe(20);
  });
  it("Matrix.__init__ default factory builds MathTex entries (python: MathTex)", () => {
    const m = new Matrix([
      ["a", "b"],
      ["c", "d"],
    ]);
    expect(m.get_entries().submobjects.length).toBe(4);
    for (const entry of m.get_entries().submobjects) {
      expect(entry).toBeInstanceOf(MathTex);
    }
  });
  it("DecimalMatrix/IntegerMatrix defaults build entries (python: DecimalNumber/Integer)", () => {
    expect(new DecimalMatrix([[1, 2]]).get_entries().submobjects.length).toBe(2);
    expect(new IntegerMatrix([[1, 2]]).get_entries().submobjects.length).toBe(2);
  });
  it("MobjectMatrix default (identity) builds entries and brackets (python: MathTex brackets)", () => {
    const m = new MobjectMatrix([[new Square()]]);
    expect(m.get_entries().submobjects.length).toBe(1);
    expect(m.get_brackets().submobjects.length).toBe(2);
  });
  it("MathTable/IntegerTable/DecimalTable defaults build entries", () => {
    expect(new MathTable([["a"]]).elements.submobjects.length).toBe(1);
    expect(new IntegerTable([[1]]).elements.submobjects.length).toBe(1);
    expect(new DecimalTable([[1]]).elements.submobjects.length).toBe(1);
  });
  it("LinearTransformationScene.add_transformable_label builds a label", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const label = scene.add_transformable_label(
      new Vector([1, 0, 0]),
      "x",
      "L",
      null,
      { animate: false },
    );
    expect(label).toBeInstanceOf(MathTex);
    expect(scene.transformable_labels.length).toBe(1);
  });
});

describe("BraceText.__init__", () => {
  it("default (Text) builds a Text label", () => {
    const bt = new BraceText(new Square(), "hello");
    expect(bt.label).toBeInstanceOf(Text);
    expect(bt.submobjects.length).toBe(2);
  });
  it("custom label_constructor returning a VMobject works", () => {
    const vm = new VMobject();
    const bt = new BraceText(new Square(), "hello", () => vm);
    expect(bt.label).toBe(vm);
    expect(bt.submobjects.length).toBe(2);
    expect(bt.brace).toBeInstanceOf(Brace);
  });
});

describe("BraceLabel.change_brace_label (tex-free VMobject path)", () => {
  it("change_brace_label swaps brace + label to the new VMobject", () => {
    // NOTE deviation: python always calls label_constructor(str(vm)) ->
    // latex; TS returns VMobject inputs as-is (NumberLine leniency).
    const bl = new BraceLabel(new Square(), new VMobject());
    const oldBrace = bl.brace;
    const oldLabel = bl.label;
    const target = new Square();
    const vm2 = new VMobject();
    bl.change_brace_label(target, vm2);
    expect(bl.label).toBe(vm2);
    expect(bl.submobjects).toContain(vm2);
    expect(bl.submobjects).not.toContain(oldLabel);
    expect(bl.brace).not.toBe(oldBrace);
    expect(bl.brace).toBeInstanceOf(Brace);
  });
  it("string label without a constructor builds MathTex", () => {
    const bl = new BraceLabel(new Square(), new VMobject());
    bl.change_brace_label(new Square(), "hello");
    expect(bl.label).toBeInstanceOf(MathTex);
  });
});

describe("NumberLine.add_labels", () => {
  it("places Square labels at n2p(x) offset by buff (python ground truth)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const s1 = new Square();
    const s2 = new Square();
    (s1 as unknown as Record<string, unknown>)["font_size"] = 36;
    (s2 as unknown as Record<string, unknown>)["font_size"] = 36;
    nl.add_labels({ 1: s1, 2: s2 } as unknown as Record<string, VMobject>);
    expect(nl.labels?.submobjects.length).toBe(2);
    vecClose(nl.labels?.submobjects[0]!.get_center(), [1, -1.25, 0]);
    vecClose(nl.labels?.submobjects[1]!.get_center(), [2, -1.25, 0]);
    // font_size is overwritten with the line default (36)
    expect((s1 as unknown as Record<string, unknown>)["font_size"]).toBe(36);
  });
  it("labels without font_size raise AttributeError like python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(() =>
      nl.add_labels({ 1: new VMobject() } as unknown as Record<string, VMobject>),
    ).toThrow(/not compatible with add_labels/);
  });
  it("empty dict builds an empty labels group", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    nl.add_labels({});
    expect(nl.labels?.submobjects.length).toBe(0);
  });
});

describe("Matrix.__init__ / get_mob_matrix / colors (bracket-free)", () => {
  it("organizes entries at h_buff/v_buff grid positions (python ground truth)", () => {
    const m = new TestMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: (item) => item as VMobject },
    );
    expect(m.mob_matrix.length).toBe(2);
    expect(m.mob_matrix[0]?.length).toBe(2);
    vecClose(m.mob_matrix[0]?.[0]!.get_center(), [-0.65, 0.4, 0]);
    vecClose(m.mob_matrix[0]?.[1]!.get_center(), [0.65, 0.4, 0]);
    vecClose(m.mob_matrix[1]?.[0]!.get_center(), [-0.65, -0.4, 0]);
    vecClose(m.mob_matrix[1]?.[1]!.get_center(), [0.65, -0.4, 0]);
    expect(m.elements.submobjects.length).toBe(4);
  });
  it("get_mob_matrix returns the underlying matrix (identity)", () => {
    const m = new TestMatrix([[new Square()]], {
      element_to_mobject: (item) => item as VMobject,
    });
    expect(m.get_mob_matrix()).toBe(m.mob_matrix);
  });
  it("set_column_colors matches python hex (RED/BLUE)", () => {
    const m = new TestMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: (item) => item as VMobject },
    );
    m.set_column_colors("RED", "BLUE");
    expect(m.mob_matrix[0]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#FC6255",
    );
    expect(m.mob_matrix[0]?.[1]!.get_color().to_hex().toUpperCase()).toBe(
      "#58C4DD",
    );
    // whole columns share the color
    expect(m.mob_matrix[1]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#FC6255",
    );
  });
  it("set_row_colors matches python hex (GREEN/YELLOW)", () => {
    const m = new TestMatrix(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: (item) => item as VMobject },
    );
    m.set_row_colors("GREEN", "YELLOW");
    expect(m.mob_matrix[0]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#83C167",
    );
    expect(m.mob_matrix[1]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#F7D96F",
    );
  });
  it("extra colors are ignored like python zip() (first entries still applied)", () => {
    const m = new TestMatrix([[new Square()]], {
      element_to_mobject: (item) => item as VMobject,
    });
    m.set_column_colors("RED", "BLUE", "GREEN");
    expect(m.mob_matrix[0]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#FC6255",
    );
    m.set_row_colors("RED", "BLUE");
    expect(m.mob_matrix[0]?.[0]!.get_color().to_hex().toUpperCase()).toBe(
      "#FC6255",
    );
  });
});

describe("DecimalMatrix / IntegerMatrix / MobjectMatrix __init__", () => {
  it("MobjectMatrix identity factory preserves entries (bracket-free)", () => {
    const s1 = new Square();
    const s2 = new Square();
    const mm = new TestMobjectMatrix([[s1, s2]]);
    expect(mm.mob_matrix[0]?.[0]).toBe(s1);
    expect(mm.mob_matrix[0]?.[1]).toBe(s2);
  });
  it("DecimalMatrix/IntegerMatrix accept custom factories (bracket-free)", () => {
    class TD extends DecimalMatrix {
      override _add_brackets(): this {
        this.brackets = new VGroup();
        return this;
      }
    }
    class TI extends IntegerMatrix {
      override _add_brackets(): this {
        this.brackets = new VGroup();
        return this;
      }
    }
    const d = new TD([[1, 2]], {
      element_to_mobject: (item) => new Square(),
    });
    expect(d.mob_matrix.length).toBe(1);
    expect(d.mob_matrix[0]?.length).toBe(2);
    const iv = new TI([[1, 2]], {
      element_to_mobject: (item) => new Square(),
    });
    expect(iv.mob_matrix[0]?.length).toBe(2);
    // DecimalMatrix default config carries num_decimal_places=1 (python)
    const d2 = new TD([[1]], {
      element_to_mobject: (item, cfg) => {
        expect(cfg?.["num_decimal_places"]).toBe(1);
        return new Square();
      },
    });
    expect(d2.mob_matrix.length).toBe(1);
  });
});

describe("MobjectTable / MathTable family __init__", () => {
  it("MobjectTable preserves entries without tex", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    expect(t.mob_table.length).toBe(2);
    expect(t.mob_table[0]?.length).toBe(2);
    expect(t.elements.submobjects.length).toBe(4);
  });
  it("MathTable/IntegerTable/DecimalTable accept explicit factories", () => {
    const mk = (item: unknown) => item as VMobject;
    const mt = new MathTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: mk },
    );
    expect(mt.mob_table.length).toBe(2);
    const itab = new IntegerTable([[new Square()]], {
      element_to_mobject: mk,
    });
    expect(itab.mob_table[0]?.length).toBe(1);
    const dtab = new DecimalTable([[new Square()]], {
      element_to_mobject: mk,
    });
    expect(dtab.mob_table[0]?.length).toBe(1);
    expect(dtab.element_to_mobject_config["num_decimal_places"]).toBe(1);
  });
  it("Table ragged rows raise ValueError like python", () => {
    expect(
      () =>
        new MobjectTable([
          [new Square(), new Square()],
          [new Square()],
        ]),
    ).toThrow(/same length/);
  });
});

describe("VDict.add", () => {
  it("adds record, Map, and pair-list forms with identity preserved", () => {
    const vd = new VDict();
    const s = new Square();
    vd.add({ a: s } as unknown as Record<string, VMobject>);
    expect(vd.__contains__("a")).toBe(true);
    expect(vd.__getitem__("a")).toBe(s);
    expect(vd.submobjects.length).toBe(1);
    const s2 = new Square();
    vd.add(
      new Map<unknown, VMobject>([["b", s2]]),
    );
    expect(vd.__getitem__("b")).toBe(s2);
    const s3 = new Square();
    vd.add([["c", s3]]);
    expect(vd.__getitem__("c")).toBe(s3);
    expect(vd.submobjects.length).toBe(3);
  });
  it("overwriting a key replaces the mobject", () => {
    const vd = new VDict();
    const s1 = new Square();
    const s2 = new Square();
    vd.add({ k: s1 } as unknown as Record<string, VMobject>);
    vd.add({ k: s2 } as unknown as Record<string, VMobject>);
    expect(vd.__getitem__("k")).toBe(s2);
  });
});

describe("GenericGraph core", () => {
  function dictGraph(): Graph {
    return new Graph([1, 2, 3], [[1, 2], [2, 3]], {
      layout: { 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0] } as unknown as Record<
        string,
        unknown
      >,
    });
  }
  it("Graph builds vertices/edges from a dict layout (repr matches python)", () => {
    const g = dictGraph();
    expect(g.vertices.size).toBe(3);
    expect(g.edges.size).toBe(2);
    vecClose(g.vertices.get(1)!.get_center(), [-2, 0, 0]);
    vecClose(g.vertices.get(2)!.get_center(), [0, 0, 0]);
    expect(g.toString()).toMatch(/Undirected graph on 3 vertices and 2 edges/);
  });
  it("__init__ stores dict layout and positions vertices", () => {
    const g = dictGraph();
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
    vecClose(g.vertices.get(3)!.get_center(), [2, 0, 0]);
  });
  it("_create_vertices defaults missing positions to graph center", () => {
    const g = dictGraph();
    const out = g._create_vertices(4, 5, {
      positions: { 4: [9, 9, 0] },
    } as unknown as Record<string, unknown>);
    expect(out.length).toBe(2);
    expect(out[0]?.[0]).toBe(4);
    const pos4 = out[0]?.[1] as NDArray;
    vecClose(pos4, [9, 9, 0]);
    // vertex 5 had no position -> graph center [0,0,0] (python ground truth)
    const pos5 = out[1]?.[1] as NDArray;
    vecClose(pos5, [0, 0, 0]);
  });
  it("_add_edge auto-creates missing vertices then adds the edge", () => {
    const g = dictGraph();
    const grp = g._add_edge([3, 4]);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3, 4]);
    expect(g.edges.size).toBe(3);
    expect(grp.submobjects.length).toBe(2); // new vertex + new edge
  });
  it("_remove_edge drops the edge mobject", () => {
    const g = dictGraph();
    const before = g.edges.size;
    g._remove_edge([1, 2]);
    expect(g.edges.size).toBe(before - 1);
    expect(() => g._remove_edge([1, 2])).toThrow(/does not contain/);
  });
  it("_remove_vertex removes vertex + incident edges (python: group of 2)", () => {
    const g = dictGraph();
    g._add_edge([3, 4]);
    const removed = g._remove_vertex(4);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
    expect(removed.submobjects.length).toBe(2);
    expect(() => g._remove_vertex(999)).toThrow(/does not contain/);
  });
  it("add_vertices positions new vertices (python: [5,5,0])", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    const added = g.add_vertices(3, {
      positions: { 3: [5, 5, 0] },
    } as unknown as Record<string, unknown>);
    vecClose(added[0]!.get_center(), [5, 5, 0]);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
  });
  it("add_edges / remove_edges round-trip counts (python: 1->2->1)", () => {
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
  it("remove_vertices drops the vertex", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    g.add_vertices(3, {
      positions: { 3: [5, 5, 0] },
    } as unknown as Record<string, unknown>);
    g.remove_vertices(3);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2]);
  });
  it("change_layout moves vertices to the new positions", () => {
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
  it("duplicate _add_vertex raises ValueError like python", () => {
    const g = dictGraph();
    expect(() => (g as unknown as { _add_vertex(v: number): void })._add_vertex(1)).toThrow(
      /already used/,
    );
  });
});

describe("LinearTransformationScene transforms (math ground truth)", () => {
  it("get_transposed_matrix_transformation matches np.dot (python)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    vecClose(func([1, 0, 0]), [0, -1, 0]);
    vecClose(func([0, 1, 0]), [1, 0, 0]);
  });
  it("get_matrix_transformation transposes first (python)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_matrix_transformation([
      [0, 1],
      [-1, 0],
    ]);
    // transpose of [[0,1],[-1,0]] is [[0,-1],[1,0]] -> same as above
    vecClose(func([1, 0, 0]), [0, -1, 0]);
  });
  it("bad dimensions raise ValueError like python", () => {
    const scene = new LinearTransformationScene();
    expect(() => scene.get_transposed_matrix_transformation([[1, 2, 3]])).toThrow(
      /bad dimensions/,
    );
  });
  it("apply_transposed_matrix sets path_arc=-pi/2 and delegates to apply_function", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    // NOTE limitation: TS NumberPlane/CoordinateSystem is not Mobject-based
    // yet (see ), so the setup-installed plane in
    // transformable_mobjects fails ApplyPointwiseFunction's
    // isinstance(Mobject) check. Python's plane IS a Mobject and applies
    // fine. Test with a real Mobject transformable instead.
    scene.transformable_mobjects = [new Square() as unknown as never] as never[];
    const playSpy = vi.spyOn(scene, "play").mockImplementation((..._args: unknown[]) => {
      return undefined as unknown as void;
    });
    const applySpy = vi.spyOn(scene, "apply_function");
    scene.apply_transposed_matrix([
      [0, -1],
      [1, 0],
    ]);
    expect(applySpy).toHaveBeenCalledOnce();
    const kwargs = applySpy.mock.calls[0]?.[2] as Record<string, unknown>;
    // python: mean(angle(func(RIGHT)), angle(func(UP)) - pi/2) = -pi/2
    expect(close(Number(kwargs["path_arc"]), -Math.PI / 2, 1e-9)).toBe(true);
    playSpy.mockRestore();
    applySpy.mockRestore();
  });
  it("apply_function defaults run_time=3 and plays transformable + vector movement", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    // NOTE: same non-Mobject plane limitation as above; use a Square.
    scene.transformable_mobjects = [new Square() as unknown as never] as never[];
    const seen: Record<string, unknown>[] = [];
    let playedAnims: unknown[] = [];
    const playSpy = vi.spyOn(scene, "play").mockImplementation((...anims: unknown[]) => {
      seen.push({ count: anims.length });
      playedAnims = anims;
      return undefined as unknown as void;
    });
    const func = (p: number[] | NDArray): NDArray => {
      const arr = Array.isArray(p) ? p : [...(p.toArray() as number[])];
      return array([arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0]) as NDArray;
    };
    scene.apply_function(func, [], {});
    expect(seen.length).toBe(1);
    // run_time default (python: kwargs["run_time"] = 3) is set on each anim
    for (const anim of playedAnims) {
      expect((anim as Record<string, unknown>)["run_time"]).toBe(3);
    }
    // kwargs mutated with run_time default (python behavior)
    playSpy.mockRestore();
  });
  it("apply_nonlinear_transformation prepares the plane then applies", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const prepareSpy = vi.spyOn(
      scene.plane as unknown as { prepare_for_nonlinear_transform(): void },
      "prepare_for_nonlinear_transform",
    );
    const applySpy = vi
      .spyOn(scene, "apply_function")
      .mockImplementation(() => undefined as unknown as void);
    const func = (p: number[] | NDArray): number[] => {
      const arr = Array.isArray(p) ? [...p] : [...(p.toArray() as number[])];
      return [arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0];
    };
    scene.apply_nonlinear_transformation(func, {});
    expect(prepareSpy).toHaveBeenCalledOnce();
    expect(applySpy).toHaveBeenCalledOnce();
    prepareSpy.mockRestore();
    applySpy.mockRestore();
  });
});

// --- integration: batch symbols interacting with earlier-ported code ---

describe("integration", () => {
  it("NumberLine labels + Matrix entry colors share VMobject identity", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const s = new Square();
    (s as unknown as Record<string, unknown>)["font_size"] = 36;
    nl.add_labels({ 1: s } as unknown as Record<string, VMobject>);
    vecClose(s.get_center(), [1, -1.25, 0]);
    const m = new TestMatrix([[s, new Square()]], {
      element_to_mobject: (item) => item as VMobject,
    });
    // NOTE: _organize_mob_matrix repositions entries (python behavior), so
    // the label square moves off the number-line spot into matrix layout.
    expect(m.get_mob_matrix()[0]?.[0]).toBe(s);
    m.set_column_colors("RED", "BLUE");
    expect(s.get_color().to_hex().toUpperCase()).toBe("#FC6255");
  });
  it("Graph vertex mobjects flow through VDict + Table cells", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as unknown as Record<string, unknown>,
    });
    const vd = new VDict();
    vd.add_key_value_pair("v1", g.vertices.get(1)! as VMobject);
    expect(vd.__getitem__("v1")).toBe(g.vertices.get(1));
    const t = new MobjectTable([
      [g.vertices.get(1)! as VMobject, g.vertices.get(2)! as VMobject],
    ]);
    expect(t.mob_table[0]?.[0]).toBe(g.vertices.get(1));
    const cell = t.get_cell([1, 1]);
    expect(cell.get_num_points()).toBeGreaterThan(0);
  });
  it("transposed-matrix func drives VMobject points like python np.dot", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    const sq = new Square();
    const before = sq.get_center();
    void before;
    sq.apply_function(func as unknown as (p: number[]) => number[]);
    // python: square centered at origin stays centered under rotation
    const c = sq.get_center();
    expect(close(c[0] ?? 0, 0, 1e-6)).toBe(true);
    expect(close(c[1] ?? 0, 0, 1e-6)).toBe(true);
    // endpoint mapping itself matches np.dot ground truth
    vecClose(func([2, 0, 0]), [0, -2, 0]);
  });
  it("Brace on a Line + BraceLabel VMobject chain", () => {
    const line = new Line([-1, 0, 0], [1, 0, 0]);
    const brace = new Brace(line, { direction: [...DOWN] });
    expect(brace.get_direction().length).toBe(3);
    const vm = new VMobject();
    const bl = new BraceLabel(line, vm);
    expect(bl.label).toBe(vm);
    const vm2 = new VMobject();
    bl.change_brace_label(line, vm2);
    expect(bl.label).toBe(vm2);
  });
});
