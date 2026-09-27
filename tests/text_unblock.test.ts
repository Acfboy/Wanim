import { describe, expect, it } from "vitest";
import {
  MathTex,
  Tex,
  Text,
} from "../src/mobject/text/index.js";
import { DecimalNumber, Integer } from "../src/mobject/text/numbers.js";
import {
  DecimalMatrix,
  IntegerMatrix,
  Matrix,
  get_det_text,
  matrix_to_mobject,
  matrix_to_tex_string,
} from "../src/mobject/matrix.js";
import { Brace, BraceLabel, BraceText } from "../src/mobject/svg/brace.js";
import { Label, LabeledArrow, LabeledLine, LabeledPolygram } from "../src/mobject/geometry/labeled.js";
import { LabeledDot } from "../src/mobject/geometry/arc.js";
import { BackgroundRectangle, SurroundingRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { Rectangle, Square } from "../src/mobject/geometry/polygram.js";
import { Line } from "../src/mobject/geometry/line.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { DOWN, LEFT, MED_SMALL_BUFF, RIGHT, SMALL_BUFF, UP } from "../src/constants.js";

// Ground truth produced with the manim 0.21 interpreter in
// `original manim` (no LaTeX available, so MathTex-dependent
// mobjects were stubbed; the layout logic itself is LaTeX-independent).

function centers(mob: VMobject): number[][] {
  return (mob.submobjects as VMobject[]).map((child) => child.get_center() as number[]);
}

function relativeCenters(mob: VMobject): number[][] {
  const all = centers(mob);
  const origin = all[0] ?? [0, 0, 0];
  return all.map((c) => c.map((value, i) => value - (origin[i] ?? 0)));
}

function expectVecClose(actual: number[], expected: number[], digits = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]!).toBeCloseTo(expected[i]!, digits);
  }
}

describe("Matrix (text unblocked)", () => {
  it("default element_to_mobject is MathTex and brackets are built", () => {
    const m = new Matrix([[1, 2], [3, 4]]);
    const entries = m.get_entries();
    expect(entries.submobjects.length).toBe(4);
    for (const entry of entries.submobjects) {
      expect(entry).toBeInstanceOf(MathTex);
      expect(entry.width).toBeGreaterThan(0);
      expect(entry.height).toBeGreaterThan(0);
    }
    expect(m.get_brackets()).toBeInstanceOf(VGroup);
    expect(m.get_brackets().submobjects.length).toBe(2);
    // Brackets are the same width and stretched on both sides.
    expect(m.get_brackets().height).toBeCloseTo(
      m.get_entries().height + 2 * MED_SMALL_BUFF,
      5,
    );
  });

  it("matches Python entry layout (stub elements, 2x3)", () => {
    // Python: Matrix([[7, 88, 999], [1.5, 2, 33]], element_to_mobject=stub)
    // with stub = Rectangle(width=0.3*len(str(item)), height=0.5).
    // Relative entry centers (after centering) were:
    //   [0,0], [1.15,0], [2.3,0], [-0.3,-0.8], [1.3,-0.8], [2.45,-0.8].
    const m = new Matrix(
      [
        [7, 88, 999],
        [1.5, 2, 33],
      ],
      {
        element_to_mobject: (item) =>
          new Rectangle({ width: 0.3 * String(item).length, height: 0.5 }),
      },
    );
    const rel = relativeCenters(m.get_entries());
    expect(rel.length).toBe(6);
    expectVecClose(rel[0]!, [0, 0, 0]);
    expectVecClose(rel[1]!, [1.15, 0, 0]);
    expectVecClose(rel[2]!, [2.3, 0, 0]);
    expectVecClose(rel[3]!, [-0.3, -0.8, 0]);
    expectVecClose(rel[4]!, [1.3, -0.8, 0]);
    expectVecClose(rel[5]!, [2.45, -0.8, 0]);
  });

  it("matches Python entry layout (stub elements, 1x3)", () => {
    const m = new Matrix([[7, 88, 999]], {
      element_to_mobject: (item) =>
        new Rectangle({ width: 0.3 * String(item).length, height: 0.5 }),
    });
    const rel = relativeCenters(m.get_entries());
    expectVecClose(rel[0]!, [0, 0, 0]);
    expectVecClose(rel[1]!, [1.15, 0, 0]);
    expectVecClose(rel[2]!, [2.3, 0, 0]);
  });

  it("honours h_buff / v_buff spacing", () => {
    const m = new Matrix([[0, 0], [0, 0]], {
      h_buff: 2.5,
      v_buff: 1.1,
      element_to_mobject: () => new Square({ side_length: 0.2 }),
    });
    const rel = relativeCenters(m.get_entries());
    expectVecClose(rel[1]!, [2.5, 0, 0]);
    expectVecClose(rel[2]!, [0, -1.1, 0]);
    expectVecClose(rel[3]!, [2.5, -1.1, 0]);
  });

  it("places brackets with bracket_h_buff gap", () => {
    const m = new Matrix([[1, 2], [3, 4]], { bracket_h_buff: 0.3 });
    const [lBracket, rBracket] = m.get_brackets().submobjects as VMobject[];
    // next_to(self, LEFT/RIGHT, buff) is applied while self is still just the
    // entries, so the gap is against the entries' bounding box.
    expect(lBracket!.get_right()[0]!).toBeCloseTo(
      m.get_entries().get_left()[0]! - 0.3,
      5,
    );
    expect(rBracket!.get_left()[0]!).toBeCloseTo(
      m.get_entries().get_right()[0]! + 0.3,
      5,
    );
    // The brackets are the outermost mobjects of the centered matrix.
    expect(m.get_left()[0]!).toBeCloseTo(lBracket!.get_left()[0]!, 5);
    expect(m.get_right()[0]!).toBeCloseTo(rBracket!.get_right()[0]!, 5);
  });

  it("stretch_brackets=false keeps the natural bracket height", () => {
    const stretched = new Matrix([[1], [2], [3], [4]], { bracket_v_buff: 0.25 });
    const plain = new Matrix([[1], [2], [3], [4]], {
      stretch_brackets: false,
      bracket_v_buff: 0.25,
    });
    expect(stretched.get_brackets().height).toBeCloseTo(
      stretched.get_entries().height + 0.5,
      5,
    );
    expect(plain.get_brackets().height).toBeLessThan(stretched.get_brackets().height);
  });

  it("IntegerMatrix uses Integer entries", () => {
    const m = new IntegerMatrix([[1, 2], [3, 4]]);
    for (const entry of m.get_entries().submobjects) {
      expect(entry).toBeInstanceOf(Integer);
    }
    expect(m.get_entries().submobjects.length).toBe(4);
    expect(m.get_brackets().submobjects.length).toBe(2);
  });

  it("DecimalMatrix uses DecimalNumber entries with config", () => {
    const m = new DecimalMatrix([[3.456, 2.122], [33.2244, 12.33]], {
      element_to_mobject_config: { num_decimal_places: 2 },
    });
    for (const entry of m.get_entries().submobjects) {
      expect(entry).toBeInstanceOf(DecimalNumber);
      expect((entry as DecimalNumber).num_decimal_places).toBe(2);
    }
  });

  it("get_columns / get_rows / set_column_colors / set_row_colors", () => {
    const m = new Matrix([[1, 2], [3, 4]]);
    const cols = m.get_columns();
    const rows = m.get_rows();
    expect(cols.submobjects.length).toBe(2);
    expect(rows.submobjects.length).toBe(2);
    expect((cols.submobjects[0] as VGroup).submobjects.length).toBe(2);
    expect((rows.submobjects[1] as VGroup).submobjects.length).toBe(2);
    expect(m.set_column_colors("RED", "GREEN")).toBe(m);
    expect(m.set_row_colors("BLUE", "YELLOW")).toBe(m);
  });

  it("add_background_rectangles_to_entries adds rectangles behind entries", () => {
    const m = new Matrix([[1, 2], [3, 4]], {
      add_background_rectangles_to_entries: true,
    });
    for (const entry of m.get_entries().submobjects as VMobject[]) {
      expect(entry.submobjects.length).toBeGreaterThan(0);
      expect(entry.submobjects[0]).toBeInstanceOf(BackgroundRectangle);
    }
  });

  it("matrix_to_tex_string / matrix_to_mobject", () => {
    expect(matrix_to_tex_string([[1, 2]])).toBe(
      "\\left[ \\begin{array}{cc}1 & 2\\end{array} \\right]",
    );
    const mob = matrix_to_mobject([[1, 2], [3, 4]]);
    expect(mob).toBeInstanceOf(MathTex);
    expect(mob.height).toBeGreaterThan(0);
  });

  it("get_det_text returns the expected structure and order", () => {
    const m = new Matrix([[2, 0], [-1, 1]]);
    const det = get_det_text(m);
    expect(det.submobjects.length).toBe(3);
    expect(det.submobjects[0]).toBeInstanceOf(Tex);
    expect((det.submobjects[0] as Tex).get_tex_string()).toContain("det");

    const det2 = get_det_text(m, 3);
    expect(det2.submobjects.length).toBe(5);
    expect(det2.submobjects[0]).toBeInstanceOf(Tex);
    expect(det2.submobjects[3]).toBeInstanceOf(MathTex);
    expect(det2.submobjects[4]).toBeInstanceOf(MathTex);
    // determinant result sits to the right of the equals sign.
    expect((det2.submobjects[4] as VMobject).get_center()[0]!).toBeGreaterThan(
      (det2.submobjects[3] as VMobject).get_center()[0]!,
    );

    const withBg = get_det_text(m, 3, true);
    expect((withBg.submobjects[0] as VMobject).submobjects.length).toBeGreaterThan(1);
  });
});

describe("Brace.get_text / get_tex (text unblocked)", () => {
  it("get_tex builds a MathTex at the tip", () => {
    const brace = new Brace(new Square());
    const tex = brace.get_tex("x");
    expect(tex).toBeInstanceOf(MathTex);
    // Tip is below the square by default, so the label sits below the center.
    expect(tex.get_center()[1]!).toBeLessThan(brace.get_tip()[1]! + 1e-9);
    expect(tex.width).toBeGreaterThan(0);
  });

  it("get_text builds a Tex at the tip", () => {
    const brace = new Brace(new Square());
    const text = brace.get_text("hello");
    expect(text).toBeInstanceOf(Tex);
    expect(text.get_tex_string()).toContain("hello");
  });

  it("passes kwargs through to put_at_tip", () => {
    const brace = new Brace(new Square());
    const tex = brace.get_tex("x", { buff: 1.0 });
    expect(tex).toBeInstanceOf(MathTex);
  });
});

describe("BraceLabel / BraceText (text unblocked)", () => {
  it("BraceLabel defaults to a MathTex label", () => {
    const bl = new BraceLabel(new Square(), "x");
    expect(bl.label).toBeInstanceOf(MathTex);
    expect(bl.submobjects.length).toBe(2);
  });

  it("BraceText defaults to a Text label", () => {
    const bt = new BraceText(new Square(), "hi");
    expect(bt.label).toBeInstanceOf(Text);
  });

  it("change_label rebuilds the label", () => {
    const bt = new BraceText(new Square(), "hi");
    bt.change_label("bye");
    expect(bt.label).toBeInstanceOf(Text);
    expect(bt.submobjects.length).toBe(2);
  });
});

describe("Label and string labels (text unblocked)", () => {
  it("Label wraps a Text instance like Python (0.5625 x 0.6063)", () => {
    const label = new Label(new Text("hi", { font: "sans" }));
    expect(label.submobjects.length).toBe(3);
    expect(label.submobjects[0]).toBeInstanceOf(BackgroundRectangle);
    expect(label.submobjects[1]).toBeInstanceOf(Text);
    expect(label.submobjects[2]).toBeInstanceOf(SurroundingRectangle);
    // Python: 0.5625 x 0.6063. The ported Text is within ~5% on width.
    expect(label.width).toBeGreaterThan(0.5625 * 0.95);
    expect(label.width).toBeLessThan(0.5625 * 1.05);
    expect(label.height).toBeCloseTo(0.6063, 3);
    expect((label.rendered_label as VMobject).height).toBeCloseTo(0.5063, 3);
  });

  it("Label renders a string with MathTex", () => {
    const label = new Label("x");
    expect(label.rendered_label).toBeInstanceOf(MathTex);
    expect(label.submobjects.length).toBe(3);
  });

  it("Label rejects unsupported label types", () => {
    expect(() => new Label(123 as unknown as string)).toThrow(TypeError);
  });

  it("LabeledLine puts a string label at label_position", () => {
    const line = new LabeledLine("0.5", 0.8, null, null, null, [-1, -1, 0], [1, 1, 0]);
    expect(line.label).toBeInstanceOf(Label);
    expect(line.label.rendered_label).toBeInstanceOf(MathTex);
    // start=(-1,-1,0), end=(1,1,0), label_position=0.8 -> (0.6, 0.6, 0).
    expect(line.label.get_center()[0]!).toBeCloseTo(0.6, 5);
    expect(line.label.get_center()[1]!).toBeCloseTo(0.6, 5);
  });

  it("LabeledArrow accepts a string label", () => {
    const arrow = new LabeledArrow("0.5", 0.5, null, null, null, [-1, -1, 0], [1, 1, 0]);
    expect(arrow.label).toBeInstanceOf(Label);
    expect(arrow.tip_length).toBeGreaterThan(0);
  });

  it("LabeledPolygram accepts a string label at its pole", () => {
    const ring = [
      [-1, -1, 0],
      [1, -1, 0],
      [1, 1, 0],
      [-1, 1, 0],
    ];
    const p = new LabeledPolygram(ring, { label: "P", precision: 0.01 });
    expect(p.label).toBeInstanceOf(Label);
    expect(p.label.rendered_label).toBeInstanceOf(MathTex);
    expect(p.radius).toBeGreaterThan(0);
    // Square ring pole is the center.
    expect(Math.abs(p.pole.toArray()[0] as number)).toBeLessThan(1e-6);
    expect(Math.abs(p.pole.toArray()[1] as number)).toBeLessThan(1e-6);
    expect(p.label.get_center()[0]!).toBeCloseTo(0, 5);
  });
});

describe("LabeledDot string labels (text unblocked)", () => {
  it("renders a string label with MathTex (black) and sizes the dot", () => {
    const dot = new LabeledDot("3");
    expect(dot.submobjects.length).toBe(1);
    const label = dot.submobjects[0] as VMobject;
    expect(label).toBeInstanceOf(MathTex);
    expectVecClose(label.get_center() as number[], dot.get_center() as number[]);
    // radius = buff + norm([w, h]) / 2
    const expectedRadius =
      SMALL_BUFF + Math.hypot(label.width, label.height) / 2;
    expect((dot as unknown as { radius: number }).radius).toBeCloseTo(expectedRadius, 5);
  });

  it("accepts a pre-rendered label", () => {
    const dot = new LabeledDot(new Text("ii", { font: "sans" }));
    expect(dot.submobjects[0]).toBeInstanceOf(Text);
  });
});

describe("module sanity", () => {
  it("Line still works (cycle-free import)", () => {
    const line = new Line([...LEFT], [...RIGHT]);
    expect(line.get_length()).toBeCloseTo(2, 5);
    expect(DOWN[1]).toBe(-1);
    expect(UP[1]).toBe(1);
  });
});
