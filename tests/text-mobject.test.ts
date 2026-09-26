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
  remove_invisible_chars,
} from "../src/mobject/text/index.js";
import { renderMarkupToSVGString } from "../src/mobject/text/pango/index.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function totalPoints(mob: { get_all_points(): { shape: number[] } }): number {
  return mob.get_all_points().shape[0] ?? 0;
}

function bbox(mob: VMobject): { width: number; height: number; center: number[] } {
  return { width: mob.width, height: mob.height, center: mob.get_center() };
}

describe("PangoJS port (string path)", () => {
  it("renders markup to an SVG string with glyph paths and colors", () => {
    const svg = renderMarkupToSVGString('<span foreground="#ff0000">hi</span>', {
      fontSize: 10,
      color: "#ff0000",
    });
    expect(svg).toContain("<svg");
    expect(svg).toContain("<path");
    expect(svg).toContain("#ff0000");
    // two glyphs => at least two fill paths
    expect((svg.match(/fill="#ff0000"/g) ?? []).length).toBeGreaterThanOrEqual(2);
  });

  it("honours Pango <b>/<i>/<u> tags without crashing", () => {
    const svg = renderMarkupToSVGString("<b>bold</b> <i>it</i> <u>u</u>", { fontSize: 10 });
    expect(svg).toContain("<path");
    expect(svg).toContain('fill="none"'); // underline decoration
  });
});

// Python ground truth (original manim, manim 0.21.0, cairo):
//   Text("hello")           n_points 356, height 0.5156, width 1.5938, submobjects 5
//   MarkupText("<b>hi</b>") n_points 188, height 0.5062, width 0.7164
// The glyph outlines here come from opentype.js/DejaVuSans rather than
// manimpango/Pango, so coordinates differ; the tests assert structure and
// that sizes are in the same ballpark.
describe("Text / MarkupText / Paragraph", () => {
  it("Text('hello') has glyph submobjects, points and plausible size", () => {
    const text = new Text("hello");
    expect(text.submobjects.length).toBe(5);
    expect(text.chars.submobjects.length).toBe(5);
    expect(totalPoints(text)).toBeGreaterThan(100);
    const { width, height, center } = bbox(text);
    expect(height).toBeGreaterThan(0.2);
    expect(height).toBeLessThan(1.2);
    expect(width).toBeGreaterThan(0.8);
    expect(width).toBeLessThan(3.0);
    // Approximate ground-truth comparison against Python (manim 0.21.0, cairo):
    // height 0.5156 / width 1.5938. Font outlines differ, so allow 2x slack.
    expect(height).toBeGreaterThan(0.5156 * 0.5);
    expect(height).toBeLessThan(0.5156 * 2.0);
    expect(width).toBeGreaterThan(1.5938 * 0.5);
    expect(width).toBeLessThan(1.5938 * 2.0);
    expect(Math.abs(center[0] as number)).toBeLessThan(1e-6);
    expect(Math.abs(center[1] as number)).toBeLessThan(1e-6);
    for (const submob of text.submobjects as VMobject[]) {
      expect(submob.get_fill_rgbas().shape[0]).toBeGreaterThan(0);
    }
  });

  it("Text fill rgbas are opaque", () => {
    const text = new Text("hello");
    const first = text.submobjects[0] as VMobject;
    const rgba = (first.get_fill_rgbas().toArray() as number[][])[0] as number[];
    expect(rgba[3]).toBeCloseTo(1, 5);
    expect(first.get_fill_color().to_hex().toLowerCase()).toBe("#ffffff");
  });

  it("Text color argument is baked into the SVG", () => {
    const text = new Text("hi", { color: "#ff0000" });
    const first = text.submobjects[0] as VMobject;
    expect(first.get_fill_color().to_hex()).toBe("#FF0000");
  });

  it("MarkupText('<b>hi</b>') works and exposes chars", () => {
    const text = new MarkupText("<b>hi</b>");
    expect(text.chars.submobjects.length).toBe(2);
    expect(totalPoints(text)).toBeGreaterThan(50);
    const { width, height } = bbox(text);
    expect(height).toBeGreaterThan(0.2);
    expect(height).toBeLessThan(1.2);
    expect(width).toBeGreaterThan(0.3);
    expect(width).toBeLessThan(1.5);
  });

  it("Paragraph splits into lines of chars", () => {
    const paragraph = new Paragraph("ab", "cd");
    expect(paragraph.chars.submobjects.length).toBe(2);
    expect((paragraph.chars.submobjects[0] as VMobject).submobjects.length).toBe(2);
    expect((paragraph.chars.submobjects[1] as VMobject).submobjects.length).toBe(2);
  });

  it("remove_invisible_chars drops zero-radius dots", () => {
    const text = new Text("a b", { disable_ligatures: true });
    const cleaned = remove_invisible_chars(text);
    expect(cleaned.submobjects.length).toBe(2);
  });
});

// Python ground truth for MathTex/Tex could not be produced: the original manim
// has no LaTeX (`latex` binary missing), so `MathTex`/`Tex` raise
// FileNotFoundError there. MathJax substitutes for LaTeX in this port; the
// tests therefore assert structure/sanity only.
describe("SingleStringMathTex / MathTex / Tex", () => {
  it("SingleStringMathTex('x^2') produces points", () => {
    const tex = new SingleStringMathTex("x^2");
    expect(totalPoints(tex)).toBeGreaterThan(0);
    const { width, height } = bbox(tex);
    expect(height).toBeGreaterThan(0.05);
    expect(height).toBeLessThan(2.0);
    expect(width).toBeGreaterThan(0.05);
  });

  it("MathTex('x^2') breaks into one part and recolors", () => {
    const tex = new MathTex("x^2");
    expect(tex.submobjects.length).toBe(1);
    expect(tex.submobjects[0]).toBeInstanceOf(VMobject);
    expect(totalPoints(tex)).toBeGreaterThan(0);
    const firstPath = (tex.submobjects[0] as VMobject).submobjects[0] as VMobject;
    expect(firstPath.get_fill_color().to_hex()).toBe("#FFFFFF");
  });

  it("MathTex with multiple strings isolates parts and color maps", () => {
    const tex = new MathTex("a", "b", { tex_to_color_map: { a: "#ff0000" } });
    expect(tex.submobjects.length).toBe(2);
    const partA = tex.get_part_by_tex("a");
    expect(partA).not.toBeNull();
    expect((partA as VMobject).submobjects[0] as VMobject).toBeDefined();
  });

  it("Tex('hello') wraps plain text and renders", () => {
    const tex = new Tex("hello");
    expect(totalPoints(tex)).toBeGreaterThan(0);
    expect(tex.submobjects.length).toBe(1);
    const { height } = bbox(tex);
    expect(height).toBeGreaterThan(0.05);
    expect(height).toBeLessThan(2.0);
  });

  it("Title adds an underline and moves to the top edge", () => {
    const title = new Title("Hi");
    expect(totalPoints(title)).toBeGreaterThan(0);
    expect(title.underline).not.toBeNull();
    const { height } = bbox(title);
    expect(height).toBeGreaterThan(0);
    expect(title.get_center()[1]).toBeGreaterThan(3);
  });

  it("BulletedList adds a bullet per item", () => {
    const list = new BulletedList("one", "two");
    expect(list.submobjects.length).toBe(2);
    for (const part of list.submobjects as VMobject[]) {
      // first submobject of each part is the \cdot bullet
      expect(part.submobjects.length).toBeGreaterThanOrEqual(1);
    }
  });
});
