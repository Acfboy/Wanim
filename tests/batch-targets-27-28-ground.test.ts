import { describe, expect, it } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
import type { NDArray } from "numpy-ts";
import { MobjectTable } from "../src/mobject/table.js";
import {
  Matrix,
  MobjectMatrix,
  matrix_to_mobject,
  matrix_to_tex_string,
  get_det_text,
} from "../src/mobject/matrix.js";
import { SampleSpace, BarChart } from "../src/mobject/graphing/probability.js";
import { PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { Label } from "../src/mobject/geometry/labeled.js";
import { Brace, BraceLabel } from "../src/mobject/svg/brace.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { Graph } from "../src/mobject/graph.js";
import {
  VDict,
  VGroup,
  VMobject,
} from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { Line, Vector } from "../src/mobject/geometry/line.js";
import { LEFT, DOWN } from "../src/constants.js";
import { Text, MathTex, Tex } from "../src/mobject/text/index.js";

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

// Bracket-free subclass: Matrix.__init__ always calls _add_brackets which
// needs MathTex (latex) in python and throws TODO in TS. Overriding to a
// no-op exercises the pure layout logic faithfully.
class TestMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

// Python ground truths (manim 0.21.0, original manim):
// - MobjectTable([[Sq,Sq],[Sq,Sq]]) (default Square side 2):
//   mob_table 2x2, elements=4; get_cell((1,1)) verts
//   [[-3.3,2.8,0],[~0,2.8,0],[~0,~0,0],[-3.3,~0,0]]
//   get_highlighted_cell((2,2)) fill #FFFF00
// - matrix_to_tex_string([[1,2],[3,4]]) =
//   '\left[ \begin{array}{cc}1 & 2 \\ 3 & 4\end{array} \right]'
// - TestMatrix 2x2 Squares: centers (0,0) [-0.65,0.4,0], (0,1) [0.65,0.4,0],
//   (1,0) [-0.65,-0.4,0], (1,1) [0.65,-0.4,0]; elements=4
// - VDict: add_key_value_pair('a', sq) -> contains, identity, 1 submobject
// - Graph([1,2],[(1,2)])._add_vertex(3,[1,2,0]) center [1,2,0]; dup -> ValueError
// - NumberLine._create_label_tex(vm) is vm (passthrough)
// - SampleSpace: divide_horizontally([0.5,0.5]) -> 2 horizontal_parts;
//   divide_vertically([0.5,0.5]) -> 2 vertical_parts
// - PolarPlane.get_radian_label builds the label string then returns
//   MathTex(string, font_size, **kwargs) (latex FileNotFoundError here)
// - Table ragged rows -> ValueError("Not all rows in table have the same length.")

describe("Table.__init__ / _table_to_mob_table", () => {
  it("MobjectTable 2x2 preserves entries and layout", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    expect(t.mob_table.length).toBe(2); // python: 2
    expect(t.mob_table[0]?.length).toBe(2); // python: 2
    expect(t.elements.submobjects.length).toBe(4); // python: 4
    expect(t.row_dim).toBe(2);
    expect(t.col_dim).toBe(2);
    expect(t.get_rows().submobjects.length).toBe(2);
    expect(t.get_columns().submobjects.length).toBe(2);
  });
  it("_table_to_mob_table maps each item through element_to_mobject + config", () => {
    const seen: unknown[] = [];
    const t = new MobjectTable([[new Square()]], {
      element_to_mobject: (item, cfg) => {
        seen.push([item, cfg]);
        return item as VMobject;
      },
      element_to_mobject_config: { marker: 7 },
    });
    expect(seen.length).toBe(1);
    expect((seen[0] as unknown[])[1]).toEqual({ marker: 7 });
    const sq = new Square();
    const out = t._table_to_mob_table([[sq]]);
    expect(out.length).toBe(1);
    expect(out[0]?.length).toBe(1);
    expect(out[0]?.[0]).toBe(sq);
  });
  it("ragged rows raise ValueError like python", () => {
    expect(
      () =>
        new MobjectTable([
          [new Square(), new Square()],
          [new Square()],
        ]),
    ).toThrow(/same length/);
  });
  it("get_cell verts + highlighted fill match python", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    const verts = pointRows(t.get_cell([1, 1]).get_vertices(), 3);
    expect(verts.length).toBe(4);
    const expected = [
      [-3.3, 2.8, 0],
      [0, 2.8, 0],
      [0, 0, 0],
      [-3.3, 0, 0],
    ];
    for (let i = 0; i < expected.length; i += 1) {
      vecClose(verts[i]!, expected[i]!);
    }
    const hl = t.get_highlighted_cell([2, 2]);
    expect(hl).toBeInstanceOf(BackgroundRectangle);
    expect(hl.get_fill_color().to_hex().toUpperCase()).toBe("#FFFF00"); // python
  });
});

describe("matrix pure layout + ported tex", () => {
  it("matrix_to_tex_string matches python exactly", () => {
    expect(matrix_to_tex_string([[1, 2], [3, 4]])).toBe(
      "\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]",
    );
  });
  it("matrix_to_mobject builds a MathTex (python: MathTex)", () => {
    const mob = matrix_to_mobject([[1, 2], [3, 4]]);
    expect(mob).toBeInstanceOf(MathTex);
    expect(mob.height).toBeGreaterThan(0);
  });
  it("Matrix._add_brackets builds two MathTex brackets", () => {
    const m = new Matrix([[1, 2], [3, 4]]);
    const ret = m._add_brackets("(", ")", {});
    expect(ret).toBe(m);
    expect(m.get_brackets().submobjects.length).toBe(2);
    for (const bracket of m.get_brackets().submobjects) {
      expect(bracket).toBeInstanceOf(MathTex);
    }
  });
  it("get_det_text builds the determinant VGroup", () => {
    const m = new Matrix([[2, 0], [-1, 1]]);
    expect(get_det_text(m).submobjects.length).toBe(3);
    expect(get_det_text(m, 3).submobjects.length).toBe(5);
  });
  it("_matrix_to_mob_matrix maps via factory; rows/cols match python", () => {
    const sq = () => new Square();
    const m = new TestMatrix(
      [
        [sq(), sq()],
        [sq(), sq()],
      ],
      { element_to_mobject: (item) => item as VMobject },
    );
    expect(m.mob_matrix.length).toBe(2);
    expect(m.mob_matrix[0]?.length).toBe(2);
    // DEVIATION (content-aware grid): upstream puts 2x2 Squares (side 2) on
    // the fixed 1.3/0.8 pitch (overlapping them); the port grows the pitch to
    // max(1.3, 2 + 0.25) = max(0.8, 2 + 0.25) = 2.25 -> centered centers
    // (-1.125, 1.125), (1.125, 1.125), (-1.125, -1.125), (1.125, -1.125).
    vecClose(m.mob_matrix[0]?.[0]!.get_center(), [-1.125, 1.125, 0]);
    vecClose(m.mob_matrix[0]?.[1]!.get_center(), [1.125, 1.125, 0]);
    vecClose(m.mob_matrix[1]?.[0]!.get_center(), [-1.125, -1.125, 0]);
    vecClose(m.mob_matrix[1]?.[1]!.get_center(), [1.125, -1.125, 0]);
    expect(m.get_rows().submobjects.length).toBe(2); // python: 2
    expect(m.get_columns().submobjects.length).toBe(2); // python: 2
    expect(m.get_entries().submobjects.length).toBe(4);
    const one = sq();
    const mapped = m._matrix_to_mob_matrix([[one]]);
    expect(mapped[0]?.[0]).toBe(one);
  });
  it("Matrix default factory builds MathTex entries", () => {
    const m = new Matrix([
      ["a", "b"],
      ["c", "d"],
    ]);
    expect(m.get_entries().submobjects.length).toBe(4);
    for (const entry of m.get_entries().submobjects) {
      expect(entry).toBeInstanceOf(MathTex);
    }
  });
  it("MobjectMatrix default (identity) builds entries and brackets", () => {
    const m = new MobjectMatrix([[new Square()]]);
    expect(m.get_entries().submobjects.length).toBe(1);
    expect(m.get_brackets().submobjects.length).toBe(2);
  });
});

describe("VDict.add_key_value_pair", () => {
  it("stores and retrieves the exact mobject (python ground truth)", () => {
    const vd = new VDict();
    const sq = new Square();
    vd.add_key_value_pair("a", sq);
    expect(vd.__contains__("a")).toBe(true); // python: True
    expect(vd.__getitem__("a")).toBe(sq); // python: identity
    expect(vd.submobjects.length).toBe(1); // python: 1
  });
  it("non-VMobject value raises TypeError like python _assert_valid_submobjects", () => {
    const vd = new VDict();
    expect(() =>
      vd.add_key_value_pair("k", {} as unknown as VMobject),
    ).toThrow(TypeError);
  });
  it("show_keys=true adds a Tex key label (python: Tex)", () => {
    const vd = new VDict({}, true);
    const sq = new Square();
    vd.add_key_value_pair("k", sq);
    expect(sq.submobjects[0]).toBeInstanceOf(Tex);
  });
});

describe("NumberLine._create_label_tex", () => {
  it("VMobject passes through as-is (python ground truth)", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm); // python: True
  });
  it("string builds a MathTex label via the default constructor", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const rendered = nl._create_label_tex("hello");
    expect(rendered).toBeInstanceOf(VMobject);
    expect(rendered.submobjects.length).toBeGreaterThan(0);
  });
});

describe("GenericGraph._add_vertex", () => {
  it("adds a vertex at the given position (python ground truth)", () => {
    const g = new Graph([1, 2], [[1, 2]]);
    const added = g._add_vertex(3, [1, 2, 0]);
    vecClose(added.get_center(), [1, 2, 0]); // python: [1.0, 2.0, 0.0]
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]); // python
  });
  it("duplicate vertex raises ValueError like python", () => {
    const g = new Graph([1, 2], [[1, 2]]);
    expect(() => g._add_vertex(1)).toThrow(/already used/);
  });
  it("label=True builds a MathTex vertex label (python: MathTex)", () => {
    const g = new Graph([1, 2], [[1, 2]]);
    const mob = g._add_vertex(9, [0, 0, 0], true);
    const label = (mob as unknown as { submobjects: Array<{ tex_string: string }> }).submobjects[0];
    expect(label.tex_string).toBe("9");
  });
});

describe("SampleSpace braces/labels + add_title", () => {
  it("divide_horizontally/vertically produce 2 parts (python ground truth)", () => {
    const ss = new SampleSpace();
    ss.divide_horizontally([0.5, 0.5]);
    expect(ss.horizontal_parts!.submobjects.length).toBe(2);
    const ss2 = new SampleSpace();
    ss2.divide_vertically([0.5, 0.5]);
    expect(ss2.vertical_parts!.submobjects.length).toBe(2);
  });
  it("get_side_braces_and_labels with VMobject labels builds 2+2 groups", () => {
    // NOTE deviation: python 0.21.0 forwards min_num_quads into
    // Brace(**kwargs) which raises TypeError (unexpected kwarg); the TS
    // Brace port accepts and ignores it, so the VMobject path succeeds.
    const ss = new SampleSpace();
    ss.divide_horizontally([0.5, 0.5]);
    const res = ss.get_side_braces_and_labels([new VMobject(), new VMobject()]);
    expect(res.submobjects.length).toBe(2);
    expect((res.submobjects[0] as VGroup).submobjects.length).toBe(2);
    expect((res.submobjects[1] as VGroup).submobjects.length).toBe(2);
  });
  it("string labels build MathTex labels (python: MathTex)", () => {
    const ss = new SampleSpace();
    ss.divide_horizontally([0.5, 0.5]);
    const res = ss.get_side_braces_and_labels(["a", "b"]);
    const labels = res.submobjects[1] as VGroup;
    expect(labels.submobjects.length).toBe(2);
    for (const label of labels.submobjects) expect(label).toBeInstanceOf(MathTex);
  });
  it("missing parts raise AssertionError like python assert", () => {
    expect(() => new SampleSpace().get_side_braces_and_labels([new VMobject()])).toThrow(
      /AssertionError/,
    );
    expect(() => new SampleSpace().get_top_braces_and_labels([new VMobject()])).toThrow(
      /AssertionError/,
    );
    expect(() => new SampleSpace().get_bottom_braces_and_labels([new VMobject()])).toThrow(
      /AssertionError/,
    );
  });
  it("top/bottom braces with VMobject labels after vertical split", () => {
    const ss = new SampleSpace();
    ss.divide_vertically([0.5, 0.5]);
    expect(
      ss.get_top_braces_and_labels([new VMobject(), new VMobject()]).submobjects.length,
    ).toBe(2);
    const ss2 = new SampleSpace();
    ss2.divide_vertically([0.5, 0.5]);
    expect(
      ss2.get_bottom_braces_and_labels([new VMobject(), new VMobject()]).submobjects
        .length,
    ).toBe(2);
  });
  it("get_subdivision_braces_and_labels attaches braces/labels/label_kwargs", () => {
    const ss = new SampleSpace();
    ss.divide_horizontally([0.5, 0.5]);
    const parts = ss.horizontal_parts!;
    const res = ss.get_subdivision_braces_and_labels(
      parts,
      [new VMobject(), new VMobject()],
      [...LEFT],
    );
    expect(res.submobjects.length).toBe(2);
    expect((parts as unknown as { braces: VGroup }).braces.submobjects.length).toBe(2);
    expect((parts as unknown as { labels: VGroup }).labels.submobjects.length).toBe(2);
    expect(
      (parts as unknown as { label_kwargs: { buff: number } }).label_kwargs.buff,
    ).toBe(0.1); // SMALL_BUFF
  });
  it("add_title builds a Tex title (python: Tex)", () => {
    const ss = new SampleSpace();
    ss.add_title();
    expect(ss.title).toBeInstanceOf(Tex);
  });
});

describe("text/tex symbols now rendered through the ported pipeline", () => {
  it("text module exposes the real Text/MathTex/Tex classes", () => {
    expect(typeof Text).toBe("function");
    expect(typeof MathTex).toBe("function");
    expect(typeof Tex).toBe("function");
    expect(new Text("hi").get_family().length).toBeGreaterThan(0);
  });
  it("Label.__init__ builds via MathTex (python: MathTex)", () => {
    const label = new Label("hi");
    expect(label.submobjects.length).toBe(3);
  });
  it("Brace.get_tex builds a MathTex at the tip (python: MathTex)", () => {
    const tex = new Brace(new Square()).get_tex("x");
    expect(tex).toBeInstanceOf(MathTex);
  });
  it("BraceLabel.__init__ with VMobject works; string builds MathTex", () => {
    // NOTE deviation: python stringifies even VMobject text and calls
    // MathTex (needs latex); the TS port returns VMobject labels as-is
    // (same leniency as NumberLine._create_label_tex).
    const vm = new VMobject();
    const bl = new BraceLabel(new Square(), vm);
    expect(bl.label).toBe(vm);
    expect(bl.submobjects.length).toBe(2);
    const bl2 = new BraceLabel(new Square(), "hello");
    expect(bl2.label).toBeInstanceOf(MathTex);
  });
  it("BraceLabel.change_label swaps to the new VMobject", () => {
    const bl = new BraceLabel(new Square(), new VMobject());
    const vm2 = new VMobject();
    bl.change_label(vm2);
    expect(bl.label).toBe(vm2);
    expect(bl.submobjects).toContain(vm2);
  });
  it("VectorScene.get_vector_label builds a MathTex (python: MathTex)", () => {
    // python would build MathTex for the string label.
    const label = new VectorScene().get_vector_label(new Vector([1, 0, 0]), "x");
    expect(label).toBeInstanceOf(MathTex);
  });
  it("BarChart.get_bar_labels builds Tex labels (python: Tex/MathTex)", () => {
    const labels = new BarChart([1, 2, 3]).get_bar_labels();
    expect(labels.submobjects.length).toBe(3);
    for (const label of labels.submobjects) expect(label).toBeInstanceOf(Tex);
  });
  it("PolarPlane.get_radian_label builds MathTex with the computed string", () => {
    // python computes number=0.25 PI radians -> "\\tfrac{\\pi}{2}".
    const label = new PolarPlane().get_radian_label(0.25);
    expect(label).toBeInstanceOf(MathTex);
    expect((label as unknown as { tex_string: string }).tex_string).toBe("\\tfrac{\\pi}{2}");
  });
});

describe("LinearTransformationScene", () => {
  it("add_title builds a Tex for a string; keeps a Mobject as-is", () => {
    const scene = new LinearTransformationScene();
    scene.add_title("hello");
    expect(scene.title).toBeInstanceOf(Tex);
    const title = new Square();
    scene.add_title(title as unknown as Parameters<LinearTransformationScene["add_title"]>[0]);
    expect(scene.title).toBe(title);
  });
  it("get_transformable_label_movement with no labels returns a Transform", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const anim = scene.get_transformable_label_movement();
    expect(anim.constructor.name).toBe("Transform"); // python: get_piece_movement([])
  });
  it("get_transposed_matrix_transformation matches np.dot (python)", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    vecClose(func([1, 0, 0]), [0, -1, 0]);
    vecClose(func([0, 1, 0]), [1, 0, 0]);
  });
});

// --- integration: batch symbols interacting with earlier-ported code ---

describe("integration", () => {
  it("MobjectTable cells + highlighted entry share VMobject geometry", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    const cell = t.get_cell([2, 2]);
    expect(cell.get_num_points()).toBeGreaterThan(0);
    const hl = t.get_highlighted_cell([2, 2]);
    expect(hl.get_fill_color().to_hex().toUpperCase()).toBe("#FFFF00");
    const entry = t.get_entries([1, 1]) as VMobject;
    expect(entry).toBe(t.mob_table[0]?.[0]);
  });
  it("SampleSpace division widths sum to full width + braces attach", () => {
    const ss = new SampleSpace();
    const div = ss.get_vertical_division([0.5, 0.5]);
    const total = (div.submobjects as VMobject[]).reduce((s, p) => s + p.width, 0);
    expect(close(total, ss.width, 1e-6)).toBe(true);
    ss.divide_horizontally([0.5, 0.5]);
    const res = ss.get_side_braces_and_labels([new VMobject(), new VMobject()]);
    expect(res.submobjects.length).toBe(2);
    ss.add_braces_and_labels();
    expect(ss.submobjects.length).toBeGreaterThan(0);
  });
  it("VDict + Graph vertex mobjects share VMobject identity", () => {
    const vd = new VDict();
    const g = new Graph([1, 2], [[1, 2]]);
    const dot = g.vertices.get(1)!;
    vd.add_key_value_pair("v1", dot as unknown as VMobject);
    expect(vd.__getitem__("v1")).toBe(dot);
  });
  it("NumberLine passthrough label feeds Brace-on-Line chain", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm);
    const b = new Brace(new Line([-1, 0, 0], [1, 0, 0]), { direction: [...DOWN] });
    expect(b.get_direction().length).toBe(3);
  });
  it("transposed-matrix func drives Vector endpoints like python np.dot", () => {
    const scene = new LinearTransformationScene();
    const func = scene.get_transposed_matrix_transformation([
      [0, -1],
      [1, 0],
    ]);
    vecClose(func([2, 0, 0]), [0, -2, 0]);
    const v = new Vector([1, 0, 0]);
    expect(close(v.get_length(), 1, 1e-6)).toBe(true);
  });
});
