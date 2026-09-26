import { describe, expect, it } from "vitest";
import {
  Text,
  MarkupText,
  Paragraph,
  MathTex,
  Tex,
  SingleStringMathTex,
  Title,
  BulletedList,
  register_font,
  remove_invisible_chars,
} from "../src/mobject/text/index.js";
import { DecimalNumber, Integer, Variable } from "../src/mobject/text/numbers.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { renderMarkupToSVGString } from "../src/mobject/text/pango/index.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

// Differential end-to-end coverage of the ported text stack. Python ground
// truths were produced with the manim 0.21 interpreter in
// `original manim` (see the task report). The bundled font is DejaVu
// Sans, which matches manim's Pango rendering when `font="sans"` is passed, so
// the size/layout assertions below compare against those Python values.

function totalPoints(mob: { get_all_points(): { shape: number[] } }): number {
  const shape = mob.get_all_points().shape;
  return shape[0] ?? 0;
}

function leaves(mob: VMobject): VMobject[] {
  const out: VMobject[] = [];
  const visit = (m: VMobject): void => {
    if (m.submobjects.length > 0) {
      for (const child of m.submobjects as VMobject[]) visit(child);
    } else if (totalPoints(m) > 0) {
      out.push(m);
    }
  };
  visit(mob);
  return out;
}

function bbox(mob: VMobject): { x0: number; x1: number; y0: number; y1: number } {
  const points = mob.points.toArray() as number[][];
  const xs = points.map((p) => p[0] as number);
  const ys = points.map((p) => p[1] as number);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

describe("Text e2e (vs Python manim, DejaVu Sans)", () => {
  // Python: Text("hello", font="sans"): width 1.5109, height 0.5156, 5 glyphs.
  it("Text('hello') size is within 5% of Python", () => {
    const t = new Text("hello", { font: "sans" });
    expect(t.submobjects.length).toBe(5);
    expect(t.chars.submobjects.length).toBe(5);
    expect(totalPoints(t)).toBeGreaterThan(100);
    expect(t.height).toBeGreaterThan(0.5156 * 0.95);
    expect(t.height).toBeLessThan(0.5156 * 1.05);
    expect(t.width).toBeGreaterThan(1.5109 * 0.95);
    expect(t.width).toBeLessThan(1.5109 * 1.05);
    // Centered by default.
    expect(Math.abs(t.get_center()[0] as number)).toBeLessThan(1e-6);
    expect(Math.abs(t.get_center()[1] as number)).toBeLessThan(1e-6);
    for (const glyph of t.submobjects as VMobject[]) {
      expect(totalPoints(glyph)).toBeGreaterThan(0);
      expect(glyph.get_fill_rgbas().shape[0]).toBeGreaterThan(0);
    }
  });

  // Python: Text("hi", font_size=48) height 0.5063; Text("hi", font_size=96)
  // height 1.0133 (font_size scales linearly).
  it("Text font_size scales the glyphs", () => {
    const a = new Text("hi", { font: "sans", font_size: 48 });
    const b = new Text("hi", { font: "sans", font_size: 96 });
    expect(a.height).toBeCloseTo(0.5063, 1);
    expect(b.height / a.height).toBeCloseTo(2, 1);
  });

  // Regression: "\n" must start a new line and never render a notdef glyph.
  // Python: Text("a\nb") -> 2 submobjects, width 0.3469, height 1.0328.
  it("Text splits on newlines instead of rendering a glyph", () => {
    const t = new Text("a\nb", { font: "sans" });
    expect(t.submobjects.length).toBe(2);
    const [first, second] = t.submobjects as VMobject[];
    expect(first.get_center()[1] as number).toBeGreaterThan(0);
    expect(second.get_center()[1] as number).toBeLessThan(0);
    expect(t.height).toBeCloseTo(1.0328, 1);
    expect(t.width).toBeLessThan(0.5);
    expect(t.text).toBe("ab");
  });

  it("three lines are evenly stacked", () => {
    const t = new Text("a\nb\nc", { font: "sans" });
    expect(t.submobjects.length).toBe(3);
    const ys = (t.submobjects as VMobject[]).map((m) => m.get_center()[1] as number);
    expect(ys[0]).toBeGreaterThan(ys[1]);
    expect(ys[1]).toBeGreaterThan(ys[2]);
    // Baseline spacing equals line_spacing / 4.8 * 0.05 => 13px*0.05 = 0.65;
    // the 'a' and 'c' glyph centers are exactly one line apart.
    expect((ys[0] as number) - (ys[2] as number)).toBeCloseTo(1.3, 2);
  });

  it("t2c colors the matching characters only", () => {
    const t = new Text("hello", { t2c: { ell: "red" } });
    const colors = (t.chars.submobjects as VMobject[]).map((c) => c.get_color().to_hex());
    expect(colors[0]).toBe("#FFFFFF");
    expect(colors[1]).toBe("#FC6255");
    expect(colors[2]).toBe("#FC6255");
    expect(colors[3]).toBe("#FC6255");
    expect(colors[4]).toBe("#FFFFFF");
  });

  it("multi-line t2c targets the right glyphs", () => {
    const t = new Text("ab\ncd", { t2c: { c: "red" } });
    const colors = (t.submobjects as VMobject[]).map((c) => c.get_color().to_hex());
    expect(colors).toEqual(["#FFFFFF", "#FFFFFF", "#FC6255", "#FFFFFF"]);
  });

  it("overlapping t2c styles raise ValueError like Python", () => {
    expect(() => new Text("abc", { t2c: { ab: "red", bc: "blue" } })).toThrow(/Ambiguous style/);
  });

  it("gradient and t2g assign per-character colors", () => {
    const g = new Text("hello", { gradient: ["red", "blue"] });
    const colors = (g.chars.submobjects as VMobject[]).map((c) => c.get_color().to_hex());
    expect(colors[0]).toBe("#FC6255");
    expect(colors[4]).toBe("#58C4DD");
    const t2g = new Text("ab", { t2g: { ab: ["red", "blue"] } });
    expect((t2g.chars.submobjects as VMobject[]).map((c) => c.get_color().to_hex())).toEqual([
      "#FC6255",
      "#58C4DD",
    ]);
  });

  it("t2f/t2s/t2w are accepted without changing outlines", () => {
    const plain = new Text("hello", { font: "sans" });
    const styled = new Text("hello", {
      font: "sans",
      t2f: { hello: "sans" },
      t2s: { hello: "ITALIC" },
      t2w: { hello: "BOLD" },
    });
    expect(styled.width).toBeCloseTo(plain.width, 5);
    expect(styled.height).toBeCloseTo(plain.height, 5);
  });

  it("height/width/font_size options behave", () => {
    expect(new Text("hello", { font: "sans", height: 1 }).height).toBeCloseTo(1, 6);
    expect(new Text("hello", { font: "sans", width: 2 }).width).toBeCloseTo(2, 6);
    const t = new Text("hello", { font: "sans" });
    t.font_size = 96;
    expect(t.font_size).toBeCloseTo(96, 3);
  });

  it("disable_ligatures keeps spaces as invisible chars", () => {
    const t = new Text("a b", { disable_ligatures: true });
    expect(t.submobjects.length).toBe(3);
    expect(t.chars.submobjects.length).toBe(3);
    // The middle submobject is a zero-radius Dot.
    expect(t.submobjects[1] instanceof Dot).toBe(true);
    expect(remove_invisible_chars(t).submobjects.length).toBe(2);
  });

  it("register_font is reported as unsupported", () => {
    expect(() => register_font("x.ttf")).toThrow(/not supported/);
  });
});

describe("MarkupText e2e", () => {
  it("renders <span foreground> colors", () => {
    const m = new MarkupText('<span foreground="red">ab</span>', { font: "sans" });
    expect(m.submobjects.length).toBe(2);
    expect(totalPoints(m)).toBeGreaterThan(50);
    for (const c of m.chars.submobjects as VMobject[]) {
      expect(c.get_color().to_hex()).toBe("#FF0000");
    }
  });

  it("<b>/<i> do not crash and keep glyphs (outlines unchanged)", () => {
    const bold = new MarkupText("<b>hi</b>", { font: "sans" });
    const italic = new MarkupText("<i>hi</i>", { font: "sans" });
    const plain = new MarkupText("hi", { font: "sans" });
    expect(totalPoints(bold)).toBeGreaterThan(0);
    expect(totalPoints(italic)).toBeGreaterThan(0);
    // Known deviation: bold/italic do not alter the DejaVu outlines.
    expect(bold.width).toBeCloseTo(plain.width, 5);
    expect(italic.width).toBeCloseTo(plain.width, 5);
  });

  it("supports <u> (decoration path) and newlines", () => {
    const m = new MarkupText("<u>ab</u>", { font: "sans" });
    // two glyphs + one underline decoration path
    expect(m.submobjects.length).toBe(3);
    const multi = new MarkupText("a\nb", { font: "sans" });
    expect(multi.submobjects.length).toBe(2);
    expect((multi.submobjects[0] as VMobject).get_center()[1] as number).toBeGreaterThan(
      (multi.submobjects[1] as VMobject).get_center()[1] as number,
    );
  });

  it("gradient tag assigns a per-character gradient", () => {
    const m = new MarkupText('<gradient from="red" to="blue">abc</gradient>', { font: "sans" });
    const colors = (m.chars.submobjects as VMobject[]).map((c) => c.get_color().to_hex());
    expect(colors).toEqual(["#FC6255", "#AA9399", "#58C4DD"]);
  });
});

describe("Paragraph e2e", () => {
  // Python: Paragraph("a", "b", font="sans") -> 2 lines, height 1.0328.
  it("stacks lines vertically and centers the block", () => {
    const p = new Paragraph("a", "b", { font: "sans" });
    expect(p.submobjects.length).toBe(2);
    expect(p.height).toBeCloseTo(1.0328, 1);
    expect(Math.abs(p.get_center()[0] as number)).toBeLessThan(1e-6);
    const ys = (p.submobjects as VMobject[]).map((l) => l.get_center()[1] as number);
    expect(ys[0]).toBeGreaterThan(ys[1]);
  });

  // Python: Paragraph("aa","b","ccc", alignment="center"): line centers x=0.
  it("supports center/left/right alignment", () => {
    const center = new Paragraph("aa", "b", "ccc", { font: "sans", alignment: "center" });
    for (const l of center.submobjects as VMobject[]) {
      expect(Math.abs(l.get_center()[0] as number)).toBeLessThan(1e-6);
    }
    const left = new Paragraph("aa", "b", "ccc", { font: "sans", alignment: "left" });
    const right = new Paragraph("aa", "b", "ccc", { font: "sans", alignment: "right" });
    const lc = (left.submobjects[0] as VMobject).get_center()[0] as number;
    const rc = (right.submobjects[0] as VMobject).get_center()[0] as number;
    expect(lc).toBeLessThan(0);
    expect(rc).toBeGreaterThan(0);
  });

  it("handles embedded newlines within an argument", () => {
    const p = new Paragraph("a\nb", "c", { font: "sans" });
    expect(p.submobjects.length).toBe(3);
    expect(totalPoints(p)).toBeGreaterThan(0);
  });
});

describe("MathTex / Tex / Title / BulletedList e2e", () => {
  it("MathTex('x^2') produces one part with superscript above the base", () => {
    const m = new MathTex("x^2");
    expect(m.submobjects.length).toBe(1);
    expect(totalPoints(m)).toBeGreaterThan(0);
    const ls = leaves(m);
    const base = ls[0] as VMobject;
    const sup = ls[1] as VMobject;
    expect((bbox(sup).y1 as number)).toBeGreaterThan(bbox(base).y1 as number);
  });

  it("MathTex('x_1') places the subscript below the base", () => {
    const m = new MathTex("x_1");
    const ls = leaves(m);
    expect(bbox(ls[1] as VMobject).y0 as number).toBeLessThan(bbox(ls[0] as VMobject).y0 as number);
  });

  it("MathTex('a','b') splits into two parts", () => {
    const m = new MathTex("a", "b");
    expect(m.submobjects.length).toBe(2);
    expect(m.get_part_by_tex("a")).not.toBeNull();
    expect(m.get_part_by_tex("b")).not.toBeNull();
  });

  // Regression: isolating substrings used to drop the isolated glyphs from the
  // part (the nested <g id=...> groups were not folded into their parent).
  it("substrings_to_isolate keeps every glyph", () => {
    const plain = new MathTex("a+b");
    const iso = new MathTex("a+b", { substrings_to_isolate: ["a", "b"] });
    expect(totalPoints(iso)).toBe(totalPoints(plain));
    expect(iso.get_part_by_tex("a")).not.toBeNull();
    expect(iso.get_part_by_tex("b")).not.toBeNull();
  });

  it("tex_to_color_map isolates and colors the matched substring", () => {
    const m = new MathTex("a+b", { tex_to_color_map: { a: "red" } });
    expect(totalPoints(m)).toBe(totalPoints(new MathTex("a+b")));
    expect((m.get_part_by_tex("a") as VMobject).get_color().to_hex()).toBe("#FC6255");
  });

  it("SingleStringMathTex (no cssId) still renders points", () => {
    const s = new SingleStringMathTex("x^2");
    expect(totalPoints(s)).toBeGreaterThan(0);
    expect(s.height).toBeGreaterThan(0.05);
  });

  it("common expressions render non-empty", () => {
    for (const expr of [
      "\\frac{1}{2}",
      "\\sum_{i=0}^n i",
      "\\int_0^1 x\\,dx",
      "e^{i\\pi}+1=0",
      "\\sqrt{2}",
      "\\left( \\frac{a}{b} \\right)",
    ]) {
      const m = new MathTex(expr);
      expect(totalPoints(m), expr).toBeGreaterThan(0);
      expect(m.height, expr).toBeGreaterThan(0);
    }
  });

  it("Tex('hello') converts text and renders", () => {
    const t = new Tex("hello");
    expect(totalPoints(t)).toBeGreaterThan(0);
    expect(t.height).toBeGreaterThan(0.05);
  });

  it("Title adds an underline spanning the frame width", () => {
    const title = new Title("Hi");
    expect(title.include_underline).toBe(true);
    expect(title.submobjects.length).toBe(2);
    expect(title.underline).not.toBeNull();
    expect((title.underline as VMobject).width).toBeCloseTo(14.2222 - 2, 3);
    const noUnderline = new Title("Hi", { include_underline: false });
    expect(noUnderline.underline).toBeNull();
    expect(noUnderline.submobjects.length).toBe(1);
  });

  it("BulletedList prepends a bullet to each item and stacks them", () => {
    const bl = new BulletedList("a", "b");
    expect(bl.submobjects.length).toBe(2);
    for (const part of bl.submobjects as VMobject[]) {
      expect(part.submobjects.length).toBeGreaterThan(0);
    }
    const first = bl.submobjects[0] as VMobject;
    const second = bl.submobjects[1] as VMobject;
    expect(first.get_center()[1] as number).toBeGreaterThan(second.get_center()[1] as number);
  });
});

describe("DecimalNumber / Integer / Variable e2e", () => {
  it("formats and lays out a decimal number", () => {
    expect(new DecimalNumber(3.14159).submobjects.length).toBe(4); // "3.14"
    expect(new DecimalNumber(1234.5).submobjects.length).toBe(8); // "1,234.50"
    expect(new DecimalNumber(1234.5, { group_with_commas: false }).submobjects.length).toBe(7);
    expect(new DecimalNumber(2.5, { num_decimal_places: 0 }).submobjects.length).toBe(1); // "2"
    expect(new DecimalNumber(2.5, { include_sign: true })._get_num_string(2.5)).toBe("+2.50");
    expect(new DecimalNumber(10, { unit: "\\text{m}" }).unit_sign).not.toBeNull();
  });

  it("renders digits through MathTex and scales with font_size", () => {
    const d = new DecimalNumber(0, { font_size: 96 });
    expect(totalPoints(d)).toBeGreaterThan(0);
    expect(d.height).toBeGreaterThan(0.2);
  });

  it("Integer rounds half to even then truncates", () => {
    expect(new Integer(2.5).get_value()).toBe(2);
    expect(new Integer(3.5).get_value()).toBe(4);
    expect(new Integer(-2.5).get_value()).toBe(-2);
  });

  it("Variable wires a label, equals sign and value", () => {
    const v = new Variable(0.5, "x");
    expect(v.submobjects.length).toBe(2);
    expect(v.label.submobjects.length).toBe(2); // "x" + "="
    expect(v.value.get_value()).toBe(0.5);
    v.tracker.set_value(3.25);
    v.value.update();
    expect(v.value.get_value()).toBe(3.25);
  });
});

describe("Pango renderer line layout", () => {
  it("places each newline-separated line at a distinct baseline", () => {
    const single = renderMarkupToSVGString("ab", { fontSize: 10 });
    const multi = renderMarkupToSVGString("a\nb", { fontSize: 10, lineSpacing: 13 });
    const ys = (svg: string): number[] =>
      [...svg.matchAll(/translate\([-\d.]+, ([-\d.]+)\)/g)].map((m) => Number(m[1]));
    const singleYs = new Set(ys(single));
    const multiYs = new Set(ys(multi));
    expect(singleYs.size).toBe(1);
    expect(multiYs.size).toBe(2);
    const [low, high] = [...multiYs].sort((a, b) => a - b) as [number, number];
    expect(high - low).toBeCloseTo(13, 5);
  });
});
