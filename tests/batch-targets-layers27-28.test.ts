import { describe, expect, it, vi } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
import { array, type NDArray } from "numpy-ts";
import { ArrowVectorField } from "../src/mobject/vector_field.js";
import { Table, MobjectTable } from "../src/mobject/table.js";
import {
  Matrix,
  MobjectMatrix,
  matrix_to_mobject,
  matrix_to_tex_string,
  get_det_text,
} from "../src/mobject/matrix.js";
import { SampleSpace, BarChart } from "../src/mobject/graphing/probability.js";
import { PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { Label, LabeledLine, LabeledPolygram } from "../src/mobject/geometry/labeled.js";
import { Brace, BraceLabel } from "../src/mobject/svg/brace.js";
import { MathTex, Tex, Text } from "../src/mobject/text/index.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { Graph } from "../src/mobject/graph.js";
import { VDict, VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { Line, Vector } from "../src/mobject/geometry/line.js";
import { LEFT, DOWN } from "../src/constants.js";
import { config } from "../src/_config/utils.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function vecClose(actual: number[] | NDArray, expected: number[], tol = 1e-6): void {
  const arr: number[] = Array.isArray(actual)
    ? [...actual]
    : [...(actual.toArray() as unknown as number[])].map(Number);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

// Python ground truths (manim 0.21.0, the original manim implementation):
// AVF f(p)=[p1,-p0,0], x_range=[-1,1,1], y_range=[-1,1,1]:
//   count=9, x_range=[-1,2,1], y_range=[-1,2,1], z_range=[0,0.5,0.5],
//   opacity=1.0, length_func(0)=0.225 (0.45*sigmoid(0), sigmoid(0)=0.5),
//   get_vector([1.0,0,0]): start [1,0,0], end [1,-0.328976,0],
//   len 0.3289763603835022 (=0.45*sigmoid(1), sigmoid(1)=0.7310585786300049)
// AVF defaults: x_range=[-8,8.5,0.5], y_range=[-4,4.5,0.5], count=561 (33*17)
// MobjectTable 2x2 Squares: mob_table 2x2, elements=4, rows=2, cols=2
// matrix_to_tex_string([[1,2],[3,4]])=
//   '\left[ \begin{array}{cc}1 & 2 \\ 3 & 4\end{array} \right]'
// VDict: add_key_value_pair('a', sq) -> submob_dict has 'a', get is sq
// NumberLine._create_label_tex(vm) is vm (passthrough)
// Graph([1,2],[(1,2)])._add_vertex(3,[1,2,0]) center [1,2,0]
// Brace(Square()): tip [0,-1.473,0] (y≈-1.473), direction norm 1.0
// Table ragged rows -> ValueError; Graph dup vertex -> ValueError

describe("ArrowVectorField.__init__", () => {
  const field = (p: number[] | NDArray): number[] => {
    const a = Array.isArray(p) ? p : [...(p.toArray() as unknown as number[])].map(Number);
    return [a[1] ?? 0, -(a[0] ?? 0), 0];
  };
  it("explicit 3x3 ranges give 9 vectors with extended ranges", () => {
    const avf = new ArrowVectorField(field, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    expect(avf.submobjects.length).toBe(9); // python: 9
    expect([...avf.x_range]).toEqual([-1, 2, 1]);
    expect([...avf.y_range]).toEqual([-1, 2, 1]);
    expect([...avf.z_range]).toEqual([0, 0.5, 0.5]);
    expect(close(avf.opacity, 1.0)).toBe(true);
    expect(close(avf.length_func(0), 0.225, 1e-9)).toBe(true);
  });
  it("get_vector length matches 0.45*sigmoid(1)", () => {
    const avf = new ArrowVectorField(field, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    const v = avf.get_vector([1.0, 0, 0]);
    vecClose(v.get_start(), [1, 0, 0], 1e-6);
    vecClose(v.get_end(), [1, -0.3289763603835022, 0], 1e-6);
    const len = Math.hypot(
      v.get_end()[0]! - v.get_start()[0]!,
      v.get_end()[1]! - v.get_start()[1]!,
      v.get_end()[2]! - v.get_start()[2]!,
    );
    expect(close(len, 0.3289763603835022, 1e-9)).toBe(true);
  });
  it(
    "default ranges match python frame-derived ranges with 561 vectors",
    { timeout: 60000 },
    () => {
    // NOTE: defaults derive from config.frame_width/frame_height (python
    // ground truth captured at frame_width=14.222..., frame_height=8 gives
    // x_range=[-8,8.5,0.5], y_range=[-4,4.5,0.5], 33*17=561 vectors).
    // Expected values are derived from the live config so the test is
    // robust to other test files mutating config in the same worker.
    const fw = config.frame_width as number;
    const fh = config.frame_height as number;
    const ex = [Math.floor(-fw / 2), Math.ceil(fw / 2) + 0.5, 0.5];
    const ey = [Math.floor(-fh / 2), Math.ceil(fh / 2) + 0.5, 0.5];
    const avf = new ArrowVectorField(field);
    expect([...avf.x_range]).toEqual(ex);
    expect([...avf.y_range]).toEqual(ey);
    const nx = Math.ceil((ex[1]! - ex[0]!) / ex[2]!);
    const ny = Math.ceil((ey[1]! - ey[0]!) / ey[2]!);
    expect(avf.submobjects.length).toBe(nx * ny); // 561 at default config
    if (close(fw, 14.222222222222221, 1e-9) && close(fh, 8, 1e-9)) {
      expect([...avf.x_range]).toEqual([-8, 8.5, 0.5]);
      expect([...avf.y_range]).toEqual([-4, 4.5, 0.5]);
      expect(avf.submobjects.length).toBe(561);
    }
  });
});

describe("Table._table_to_mob_table / Table.__init__", () => {
  it("MobjectTable 2x2 preserves entries and layout", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    expect(t.mob_table.length).toBe(2);
    expect(t.mob_table[0]?.length).toBe(2);
    expect(t.elements.submobjects.length).toBe(4);
    expect(t.get_rows().submobjects.length).toBe(2);
    expect(t.get_columns().submobjects.length).toBe(2);
  });
  it("_table_to_mob_table maps via element_to_mobject", () => {
    const t = new MobjectTable([[new Square()]]);
    const out = t._table_to_mob_table([[new Square()]]);
    expect(out.length).toBe(1);
    expect(out[0]?.length).toBe(1);
    expect(out[0]?.[0] instanceof VMobject).toBe(true);
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
});

describe("matrix (ported tex + pure layout)", () => {
  it("matrix_to_mobject builds a MathTex", () => {
    const mob = matrix_to_mobject([[1, 2], [3, 4]]);
    expect(mob).toBeInstanceOf(MathTex);
    expect(mob.height).toBeGreaterThan(0);
  });
  it("matrix_to_tex_string is pure and matches python", () => {
    expect(matrix_to_tex_string([[1, 2], [3, 4]])).toBe(
      "\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]",
    );
  });
  it("Matrix._add_brackets builds two MathTex brackets", () => {
    const m = new Matrix([[1, 2], [3, 4]]);
    expect(m._add_brackets("[", "]", {})).toBe(m);
    expect(m.get_brackets().submobjects.length).toBe(2);
  });
  it("get_det_text builds the determinant VGroup", () => {
    const m = new Matrix([[2, 0], [-1, 1]]);
    expect(get_det_text(m).submobjects.length).toBe(3);
  });
  it("_matrix_to_mob_matrix / get_rows / get_columns via bracket-free subclass", () => {
    // NOTE: Matrix.__init__ always calls _add_brackets (MathTex, out of
    // scope), so both python (needs latex) and the TS port (TODO throw)
    // cannot construct a Matrix directly here. A subclass overriding
    // _add_brackets to a no-op exercises the pure layout logic faithfully.
    class TestMatrix extends Matrix {
      override _add_brackets(): this {
        this.brackets = new VGroup();
        return this;
      }
    }
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
    expect(m.get_rows().submobjects.length).toBe(2);
    expect(m.get_columns().submobjects.length).toBe(2);
    expect(m.get_entries().submobjects.length).toBe(4);
    // _matrix_to_mob_matrix maps each item through the factory
    const mapped = m._matrix_to_mob_matrix([[sq()]]);
    expect(mapped.length).toBe(1);
    expect(mapped[0]?.[0] instanceof Square).toBe(true);
  });
});

describe("VDict.add_key_value_pair", () => {
  it("stores and retrieves the exact mobject", () => {
    const vd = new VDict();
    const sq = new Square();
    vd.add_key_value_pair("a", sq);
    expect(vd.__contains__("a")).toBe(true);
    expect(vd.__getitem__("a")).toBe(sq);
    expect(vd.submobjects.length).toBe(1);
  });
  it("show_keys=true adds a Tex key label to the value (python: Tex)", () => {
    const vd = new VDict({}, true);
    const sq = new Square();
    vd.add_key_value_pair("k", sq);
    expect(sq.submobjects.length).toBe(1);
    expect(sq.submobjects[0]).toBeInstanceOf(Tex);
  });
});

describe("NumberLine._create_label_tex", () => {
  it("VMobject passes through as-is", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm);
  });
  it("string builds a MathTex label via the default constructor", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const rendered = nl._create_label_tex("hello");
    expect(rendered).toBeInstanceOf(VMobject);
    expect(rendered.submobjects.length).toBeGreaterThan(0);
  });
});

describe("GenericGraph._add_vertex", () => {
  it("adds a vertex at the given position", () => {
    const g = new Graph([1, 2], [[1, 2]]);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2]);
    const added = g._add_vertex(3, [1, 2, 0]);
    vecClose(added.get_center(), [1, 2, 0], 1e-6);
    expect([...g.vertices.keys()].sort()).toEqual([1, 2, 3]);
  });
  it("duplicate vertex raises ValueError like python", () => {
    const g = new Graph([1, 2], [[1, 2]]);
    expect(() => g._add_vertex(1)).toThrow(/already used/);
  });
});

describe("ported text labels", () => {
  it("Label builds a string label", () => {
    expect(new Label("hi").submobjects.length).toBe(3);
  });
  it("LabeledLine builds a string label", () => {
    expect(new LabeledLine("hi").label).toBeDefined();
  });
  it("LabeledPolygram requires label; with label places it", () => {
    // python: TypeError "missing 1 required keyword-only argument: 'label'"
    expect(() => new LabeledPolygram([0, 0, 0], [1, 0, 0])).toThrow(TypeError);
    const p = new LabeledPolygram([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], {
      label: "hi",
    } as unknown as Record<string, unknown>);
    expect(p.label).toBeDefined();
  });
  it("Brace.get_tex builds a MathTex at the tip", () => {
    const b = new Brace(new Square());
    expect(b.get_tex("x")).toBeInstanceOf(MathTex);
  });
  it("VectorScene.get_vector_label builds a MathTex", () => {
    const vs = new VectorScene();
    expect(vs.get_vector_label(new Vector([1, 0, 0]), "x")).toBeInstanceOf(MathTex);
  });
  it("SampleSpace.add_title builds a Tex", () => {
    const ss = new SampleSpace();
    ss.add_title();
    expect(ss.title).toBeInstanceOf(Tex);
  });
  it("BarChart.get_bar_labels builds Tex labels", () => {
    const bc = new BarChart([1, 2, 3]);
    const labels = bc.get_bar_labels();
    expect(labels.submobjects.length).toBe(3);
    for (const label of labels.submobjects) expect(label).toBeInstanceOf(Tex);
  });
  it("PolarPlane.get_radian_label builds a MathTex", () => {
    const pp = new PolarPlane();
    expect(pp.get_radian_label(0.25)).toBeInstanceOf(MathTex);
  });
  it("PolarPlane.get_coordinate_labels builds 20 azimuth labels", () => {
    const pp = new PolarPlane();
    const labels = pp.get_coordinate_labels();
    expect(labels.submobjects[1].submobjects.length).toBe(20);
  });
});

describe("Brace / BraceLabel with VMobject labels (tex-free path)", () => {
  it("Brace tip/direction match python", () => {
    const b = new Brace(new Square());
    const tip = b.get_tip();
    // python ground truth: tip [0, -1.473, 0] (rounded to 3dp)
    expect(close(tip[0] ?? 0, 0, 1e-6)).toBe(true);
    expect(close(tip[2] ?? 0, 0, 1e-6)).toBe(true);
    expect(Math.abs((tip[1] ?? 0) - -1.473) < 0.05).toBe(true);
    const dir = b.get_direction() as number[];
    const n = Math.hypot(...dir);
    expect(close(n, 1.0, 1e-9)).toBe(true);
  });
  it("BraceLabel.__init__ with VMobject label works; string builds MathTex", () => {
    // NOTE deviation: python stringifies even VMobject text and calls
    // MathTex (needs latex); the TS port returns VMobject labels as-is
    // (same leniency as NumberLine._create_label_tex).
    const sq = new Square();
    const vm = new VMobject();
    const bl = new BraceLabel(sq, vm);
    expect(bl.submobjects.length).toBe(2);
    expect(bl.label).toBe(vm);
    expect(new BraceLabel(new Square(), "hello").label).toBeInstanceOf(MathTex);
  });
  it("BraceLabel.change_label swaps to the new VMobject", () => {
    const bl = new BraceLabel(new Square(), new VMobject());
    const vm2 = new VMobject();
    bl.change_label(vm2);
    expect(bl.label).toBe(vm2);
    expect(bl.submobjects).toContain(vm2);
  });
});

describe("SampleSpace braces/labels (tex-free VMobject path)", () => {
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
    const top = ss.get_top_braces_and_labels([new VMobject(), new VMobject()]);
    expect(top.submobjects.length).toBe(2);
    const ss2 = new SampleSpace();
    ss2.divide_vertically([0.5, 0.5]);
    const bottom = ss2.get_bottom_braces_and_labels([new VMobject(), new VMobject()]);
    expect(bottom.submobjects.length).toBe(2);
  });
  it("get_subdivision_braces_and_labels attaches braces/labels to parts", () => {
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
    expect(anim.constructor.name).toBe("Transform");
  });
});

// --- integration: batch symbols interacting with earlier-ported code ---

describe("integration", () => {
  it("ArrowVectorField vectors chain with Vector start/end math", () => {
    const avf = new ArrowVectorField(
      (p: number[] | NDArray) => {
        const a = Array.isArray(p) ? p : [...(p.toArray() as unknown as number[])].map(Number);
        return [a[1] ?? 0, -(a[0] ?? 0), 0];
      },
      { x_range: [-1, 1, 1], y_range: [-1, 1, 1] },
    );
    expect(avf.submobjects.length).toBe(9);
    const v = new Vector([3, 4, 0]);
    expect(close(v.get_length(), 5, 1e-6)).toBe(true);
    const fv = avf.get_vector([0.0, 1.0, 0]);
    expect(fv.get_length()).toBeGreaterThan(0);
  });
  it("MobjectTable of Squares + get_cell polygon geometry", () => {
    const t = new MobjectTable([
      [new Square(1), new Square(1)],
      [new Square(1), new Square(1)],
    ]);
    const cell = t.get_cell([1, 1]);
    expect(cell.get_num_points()).toBeGreaterThan(0);
    // python ground truth (MobjectTable 2x2 Square(1), h_buff=1.3, v_buff=0.8):
    // cell (1,1) corners UL [-2.3,1.8,0], UR [0,1.8,0], DR [0,0,0], DL [-2.3,0,0]
    const verts = pointRows(cell.get_vertices(), 3);
    expect(verts.length).toBe(4);
    const expected = [
      [-2.3, 1.8, 0],
      [0, 1.8, 0],
      [0, 0, 0],
      [-2.3, 0, 0],
    ];
    for (let i = 0; i < expected.length; i += 1) {
      vecClose(verts[i]!, expected[i]!, 1e-6);
    }
    // python: get_highlighted_cell((2,2)) is a BackgroundRectangle with
    // PURE_YELLOW fill (#FFFF00) and real points.
    const hl = t.get_highlighted_cell([2, 2]);
    expect(hl).toBeInstanceOf(BackgroundRectangle);
    expect(hl.get_num_points()).toBeGreaterThan(0);
    expect(hl.get_fill_color().to_hex().toUpperCase()).toBe("#FFFF00");
  });
  it("SampleSpace division widths sum to full width + braces attach", () => {
    const ss = new SampleSpace();
    const div = ss.get_vertical_division([0.5, 0.5]);
    const parts = div.submobjects as VMobject[];
    const total = parts.reduce((s, p) => s + p.width, 0);
    expect(close(total, ss.width, 1e-6)).toBe(true);
    ss.divide_horizontally([0.5, 0.5]);
    const res = ss.get_side_braces_and_labels([new VMobject(), new VMobject()]);
    expect(res.submobjects.length).toBe(2);
  });
  it("VDict + Graph vertex mobjects share VMobject identity", () => {
    const vd = new VDict();
    const g = new Graph([1, 2], [[1, 2]]);
    const dot = g.vertices.get(1)!;
    vd.add_key_value_pair("v1", dot as VMobject);
    expect(vd.__getitem__("v1")).toBe(dot);
  });
  it("NumberLine passthrough label feeds add_labels-free chain; Brace on Line", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const vm = new VMobject();
    expect(nl._create_label_tex(vm)).toBe(vm);
    const b = new Brace(new Line([-1, 0, 0], [1, 0, 0]), { direction: [...DOWN] });
    expect(b.get_direction().length).toBe(3);
  });
});
