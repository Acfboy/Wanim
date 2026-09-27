// Tests for the text/tex TODOs that were unblocked once the MathTex/Tex/Text
// pipeline landed. The expected values below were captured from the original
// Python manim (original manim, original manim) by monkeypatching the LaTeX classes
// with recording stubs (the original manim has no `latex` binary): label strings,
// counts, colors, font sizes and index/unit values are the real Python
// behaviour. See /tmp/opencode/todo/capture_truth.py.
import { describe, expect, it } from "vitest";
import { PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { BarChart, SampleSpace } from "../src/mobject/graphing/probability.js";
import { LogBase } from "../src/mobject/graphing/scale.js";
import { MathTex, Tex, Paragraph } from "../src/mobject/text/index.js";
import { Integer } from "../src/mobject/text/numbers.js";
import { Matrix } from "../src/mobject/matrix.js";
import {
  Table,
  MathTable,
  IntegerTable,
  DecimalTable,
} from "../src/mobject/table.js";
import { index_labels } from "../src/utils/debug.js";
import { Graph } from "../src/mobject/graph.js";
import { VDict, VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Vector } from "../src/mobject/geometry/line.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { config } from "../src/_config/utils.js";

function tex(mob: unknown): string {
  return (mob as { tex_string: string }).tex_string;
}

describe("PolarPlane.get_radian_label (python ground truth)", () => {
  it("formats compact fractions for PI radians", () => {
    const pp = new PolarPlane();
    expect(tex(pp.get_radian_label(0))).toBe("0");
    expect(tex(pp.get_radian_label(1 / 6))).toBe("\\tfrac{\\pi}{3}");
    expect(tex(pp.get_radian_label(0.25))).toBe("\\tfrac{\\pi}{2}");
    expect(tex(pp.get_radian_label(1 / 3))).toBe("\\tfrac{2\\pi}{3}");
    expect(tex(pp.get_radian_label(0.5))).toBe("\\pi");
    expect(tex(pp.get_radian_label(1))).toBe("2\\pi");
    expect(tex(pp.get_radian_label(1.5))).toBe("3\\pi");
  });

  it("formats non-compact fractions for PI radians", () => {
    const pp = new PolarPlane({ azimuth_compact_fraction: false });
    expect(tex(pp.get_radian_label(1 / 6))).toBe("\\tfrac{1}{3}\\pi");
    expect(tex(pp.get_radian_label(1 / 3))).toBe("\\tfrac{2}{3}\\pi");
    expect(tex(pp.get_radian_label(1.5))).toBe("3\\pi");
  });

  it("formats TAU radians", () => {
    const pp = new PolarPlane({ azimuth_units: "TAU radians" });
    expect(tex(pp.get_radian_label(0))).toBe("0");
    expect(tex(pp.get_radian_label(0.25))).toBe("\\tfrac{\\tau}{4}");
    expect(tex(pp.get_radian_label(0.5))).toBe("\\tfrac{\\tau}{2}");
    expect(tex(pp.get_radian_label(1))).toBe("\\tau");
  });
});

describe("PolarPlane.get_coordinate_labels (python ground truth)", () => {
  it("builds degrees / gradians / None azimuth labels verbatim", () => {
    const deg = new PolarPlane({ azimuth_units: "degrees" });
    const degLabels = deg.get_coordinate_labels([1, 2], [0.1, 0.2]);
    expect(degLabels.submobjects[1].submobjects.map(tex)).toEqual([
      "36^{\\circ}",
      "72^{\\circ}",
    ]);

    const grad = new PolarPlane({ azimuth_units: "gradians" });
    expect(grad.get_coordinate_labels([1], [0.1]).submobjects[1].submobjects.map(tex)).toEqual([
      "40^{g}",
    ]);

    const none = new PolarPlane({ azimuth_units: null });
    expect(none.get_coordinate_labels([1], [0.125, 0.25]).submobjects[1].submobjects.map(tex)).toEqual([
      "0.125",
      "0.25",
    ]);
  });

  it("defaults to 20 radian labels and returns the x-axis as the radius group", () => {
    const pp = new PolarPlane();
    const labels = pp.get_coordinate_labels();
    expect(labels.submobjects[1].submobjects.length).toBe(20);
    // Python: `r_mobs = self.get_x_axis().add_numbers(r_values)` returns the
    // axis itself (add_numbers returns self).
    expect(labels.submobjects[0]).toBe(pp.get_x_axis() as unknown as VMobject);
  });
});

describe("BarChart labels (python ground truth)", () => {
  it("_add_x_axis_labels builds one Tex per name below/above by sign", () => {
    const bc = new BarChart([1, -2, 3], { bar_names: ["a", "b", "c"] });
    const labels = bc.x_axis.labels;
    expect(labels).not.toBeNull();
    const subs = labels!.submobjects;
    expect(subs.map(tex)).toEqual(["\\text{a}", "\\text{b}", "\\text{c}"]);
    for (const label of subs) {
      expect((label as unknown as { font_size: number }).font_size).toBeCloseTo(24, 6);
    }
    // direction: UP (above) for negative bars, DOWN (below) for positive.
    const offsets = subs.map((label, i) => {
      const anchor = bc.x_axis.number_to_point(0.5 + i);
      return (label.get_center()[1] ?? 0) - (anchor[1] ?? 0);
    });
    expect(offsets[0]!).toBeLessThan(0);
    expect(offsets[1]!).toBeGreaterThan(0);
    expect(offsets[2]!).toBeLessThan(0);
  });

  it("get_bar_labels uses values, bar fill colors and sign direction", () => {
    const bc = new BarChart([1, -2, 3]);
    const labels = bc.get_bar_labels();
    expect(labels.submobjects.map(tex)).toEqual(["\\text{1}", "\\text{-2}", "\\text{3}"]);
    expect(
      labels.submobjects.map((m) => m.get_fill_color().toHex()),
    ).toEqual(["#003F5C", "#BC5090", "#FFA600"]);
    const offsets = labels.submobjects.map((label, i) => {
      const bar = bc.bars.submobjects[i]!;
      return (label.get_center()[1] ?? 0) - (bar.get_center()[1] ?? 0);
    });
    expect(offsets[0]!).toBeGreaterThan(0);
    expect(offsets[1]!).toBeLessThan(0);
    expect(offsets[2]!).toBeGreaterThan(0);
  });

  it("SampleSpace.add_title builds a Tex title above the square", () => {
    const ss = new SampleSpace();
    ss.add_title("My Space");
    expect(ss.title).toBeInstanceOf(Tex);
    expect(tex(ss.title)).toContain("My Space");
    expect((ss.title!.get_center()[1] ?? 0)).toBeGreaterThan((ss.get_center()[1] ?? 0));
  });
});

describe("Graph labels=True (python ground truth)", () => {
  it("builds one MathTex per vertex, colored with label_fill_color", () => {
    const g = new Graph([1, 2, 3], [[1, 2], [2, 3]], {
      labels: true,
      layout: new Map<number, number[]>([
        [1, [-2, 0, 0]],
        [2, [0, 0, 0]],
        [3, [2, 0, 0]],
      ]),
    });
    const labels = [...(g as unknown as { _labels: Map<number, MathTex> })._labels.values()];
    expect(labels.map(tex).sort()).toEqual(["1", "2", "3"]);
    for (const label of labels) {
      expect(label.get_color().toHex()).toBe("#000000");
    }
  });

  it("_create_vertex(label=True) builds a MathTex vertex label", () => {
    const g = new Graph([1], [], { layout: new Map([[1, [0, 0, 0]]]) });
    const created = g._create_vertex(9, null, true);
    const mob = created[3] as unknown as VMobject;
    expect(tex(mob.submobjects[0])).toBe("9");
  });
});

describe("index_labels (python ground truth)", () => {
  it("builds Integer labels with background stroke, sized and moved", () => {
    const mob = new Square();
    mob.add(new Square().shift([1, 0, 0]));
    const labels = index_labels(mob, 0.2, 7, "#00ff00");
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(2);
    expect(labels.submobjects.map((m) => (m as Integer).number)).toEqual([0, 1]);
    for (const label of labels.submobjects) {
      expect((label as VMobject).height).toBeCloseTo(0.2, 6);
      expect((label as VMobject).background_stroke_width).toBe(7);
      expect((label as VMobject).background_stroke_color?.toHex()).toBe("#00FF00");
    }
  });
});

describe("LogBase.get_custom_labels (python ground truth)", () => {
  it("wraps the formatted inverse exponent as an Integer unit", () => {
    const log = new LogBase(10, true);
    const labels = log.get_custom_labels([1, 10, 100], 0) as Integer[];
    expect(labels.map((m) => m.number)).toEqual([10, 10, 10]);
    expect(labels.map((m) => m.unit)).toEqual(["^{0}", "^{1}", "^{2}"]);

    const log2 = new LogBase(2, true);
    const labels2 = log2.get_custom_labels([1, 2, 4], 1) as Integer[];
    expect(labels2.map((m) => m.number)).toEqual([2, 2, 2]);
    expect(labels2.map((m) => m.unit)).toEqual(["^{0.0}", "^{1.0}", "^{2.0}"]);
  });
});

describe("VDict show_keys (python ground truth)", () => {
  it("adds a Tex key label left of each value", () => {
    const d = new VDict({ a: new Square(), b: new Square() }, true);
    expect(d.submobjects.map((m) => tex(m.submobjects[0])).sort()).toEqual([
      "\\text{a}",
      "\\text{b}",
    ]);
  });

  it("show_keys=false adds no labels", () => {
    const d = new VDict({ a: new Square() }, false);
    expect(d.submobjects[0]!.submobjects.length).toBe(0);
  });
});

describe("Vector.coordinate_label (python ground truth)", () => {
  it("builds a rounded column Matrix", () => {
    const v1 = new Vector([1, 2]);
    const l1 = v1.coordinate_label();
    expect(l1).toBeInstanceOf(Matrix);
    expect(l1.get_entries().submobjects.map(tex)).toEqual(["1", "2"]);

    const v2 = new Vector([-3, -2]);
    expect(v2.coordinate_label().get_entries().submobjects.map(tex)).toEqual(["-3", "-2"]);
  });

  it("honours integer_labels and n_dim", () => {
    const v = new Vector([-3, -2]);
    // n_dim=3 pads with the vector's z component (0).
    expect(v.coordinate_label(true, 3).get_entries().submobjects.map(tex)).toEqual([
      "-3",
      "-2",
      "0",
    ]);
    // non-integer leaves the float spelling (python keeps -3.0/-2.0).
    expect(v.coordinate_label(false).get_entries().submobjects.length).toBe(2);
  });
});

describe("VectorScene label helpers", () => {
  it("get_basis_vector_labels builds the i-hat/j-hat MathTex labels", () => {
    const vs = new VectorScene();
    const labels = vs.get_basis_vector_labels();
    expect(labels.submobjects.map(tex)).toEqual(["\\hat{\\imath}", "\\hat{\\jmath}"]);
  });

  it("label_vector builds a single MathTex label and returns it", () => {
    const vs = new VectorScene();
    const label = vs.label_vector(new Vector([1, 0, 0]), "x", false);
    expect(label).toBeInstanceOf(MathTex);
    expect(tex(label)).toBe("\\vec{\\textbf{x}}");
  });

  it("write_vector_coordinates writes and returns the Matrix label", () => {
    const vs = new VectorScene();
    const coords = vs.write_vector_coordinates(new Vector([2, -1]));
    expect(coords).toBeInstanceOf(Matrix);
    expect(coords.get_entries().submobjects.map(tex)).toEqual(["2", "-1"]);
  });

  it("vector_to_coords returns [matrix, x_line, y_line]", () => {
    const vs = new VectorScene();
    const result = vs.vector_to_coords(new Vector([-3, -2]), true, false);
    expect(result.length).toBe(3);
    expect(result[0]).toBeInstanceOf(Matrix);
    expect(result[0].get_entries().submobjects.map(tex)).toEqual(["-3", "-2"]);
  });
});

describe("Table defaults (python ground truth)", () => {
  it("Table defaults to Paragraph entries", () => {
    const t = new Table([
      ["a", "b"],
      [1, 2],
    ]);
    expect(t.elements.submobjects.length).toBe(4);
    for (const entry of t.elements.submobjects) expect(entry).toBeInstanceOf(Paragraph);
  });

  it("MathTable/IntegerTable/DecimalTable build MathTex/Integer/DecimalNumber", () => {
    expect(new MathTable([["x", "y"]]).elements.submobjects[0]).toBeInstanceOf(MathTex);
    expect(new IntegerTable([[3]]).elements.submobjects[0]).toBeInstanceOf(Integer);
    const decimal = new DecimalTable([[1.25]]);
    expect(decimal.elements.submobjects.length).toBe(1);
    expect((decimal.elements.submobjects[0] as unknown as { constructor: { name: string } }).constructor.name).toBe(
      "DecimalNumber",
    );
  });
});

describe("LinearTransformationScene.add_title", () => {
  it("builds a scaled Tex title for a string", () => {
    const scene = new LinearTransformationScene();
    scene.setup();
    const result = scene.add_title("Hello", 1.5, false);
    expect(result).toBe(scene);
    expect(scene.title).toBeInstanceOf(Tex);
    expect(tex(scene.title)).toContain("Hello");
  });
});

describe("genuine tex stubs report accurate messages", () => {
  it("config.tex_template explains TexTemplate is unavailable", () => {
    expect(() => config.tex_template).toThrow(/TexTemplate is not ported/);
  });
});
