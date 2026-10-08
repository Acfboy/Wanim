// py2js auto-preload detection: `runPythonOnCanvas` must load MathJax for
// scenes that typeset math (over-inclusive by design) and stay silent for
// plain geometry, so the optional chunks are only fetched when needed.
import { describe, expect, it, vi } from "vitest";
import {
  detectPreloadNeeds,
  preloadForSource,
  resolvePreloadNeeds,
} from "../py2js/src/preload.js";

const PLAIN_SCENE = `from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        square = Square()
        square.rotate(PI / 4)
        self.play(Create(square))
        self.play(Transform(square, circle))
`;

function mocks() {
  return {
    mathjax: vi.fn(async () => undefined),
    fonts: vi.fn(async () => undefined),
  };
}

describe("detectPreloadNeeds", () => {
  it("recognises the common TeX-producing symbols", () => {
    const names = [
      "MathTex",
      "Tex",
      "SingleStringMathTex",
      "DecimalNumber",
      "Integer",
      "Matrix",
      "Table",
      "Brace",
      "BulletedList",
      "Title",
      "get_axis_labels",
    ];
    for (const name of names) {
      const source = `${PLAIN_SCENE}\n        ${name}(...)\n`;
      expect(detectPreloadNeeds(source).mathjax, name).toBe(true);
    }
  });

  it("recognises internal TeX triggers (include_numbers, Label)", () => {
    expect(detectPreloadNeeds("axes = Axes(x_axis_config={'include_numbers': True})").mathjax).toBe(true);
    expect(detectPreloadNeeds("label = Label('x')").mathjax).toBe(true);
    expect(detectPreloadNeeds("axes = Axes(x_length=4)").mathjax).toBe(false);
  });

  it("stays quiet for a plain geometry scene", () => {
    expect(detectPreloadNeeds(PLAIN_SCENE)).toEqual({ mathjax: false, fonts: false });
  });

  it("detects fonts for Pango text", () => {
    expect(detectPreloadNeeds("self.add(Text('hi'))").fonts).toBe(true);
    expect(detectPreloadNeeds("self.add(MarkupText('<b>hi</b>'))").fonts).toBe(true);
  });
});

describe("resolvePreloadNeeds", () => {
  it("auto follows detection, none disables, object is explicit", () => {
    expect(resolvePreloadNeeds(undefined, PLAIN_SCENE)).toEqual({ mathjax: false, fonts: false });
    expect(resolvePreloadNeeds("none", "MathTex('x')")).toEqual({ mathjax: false, fonts: false });
    expect(resolvePreloadNeeds({ mathjax: true }, PLAIN_SCENE)).toEqual({
      mathjax: true,
      fonts: false,
    });
  });
});

describe("preloadForSource", () => {
  it("triggers exactly the loaders a MathTex scene needs", async () => {
    const loaders = mocks();
    const needs = await preloadForSource("text = MathTex(r'x^2')", {}, loaders);
    expect(needs).toEqual({ mathjax: true, fonts: false });
    expect(loaders.mathjax).toHaveBeenCalledTimes(1);
    expect(loaders.fonts).not.toHaveBeenCalled();
  });

  it("does not trigger any loader for a plain scene", async () => {
    const loaders = mocks();
    await preloadForSource(PLAIN_SCENE, {}, loaders);
    expect(loaders.mathjax).not.toHaveBeenCalled();
    expect(loaders.fonts).not.toHaveBeenCalled();
  });

  it("loads fonts for Text", async () => {
    const loaders = mocks();
    await preloadForSource("self.add(Text('hi'))", {}, loaders);
    expect(loaders.fonts).toHaveBeenCalledTimes(1);
    expect(loaders.mathjax).not.toHaveBeenCalled();
  });

  it("honours preload: 'none'", async () => {
    const loaders = mocks();
    await preloadForSource("MathTex('x')", { preload: "none" }, loaders);
    expect(loaders.mathjax).not.toHaveBeenCalled();
  });

  it("auto fonts failures are swallowed, explicit ones propagate", async () => {
    const auto = mocks();
    auto.fonts.mockRejectedValueOnce(new Error("offline"));
    await expect(preloadForSource("Text('hi')", {}, auto)).resolves.toMatchObject({ fonts: true });

    const explicit = mocks();
    explicit.fonts.mockRejectedValueOnce(new Error("offline"));
    await expect(
      preloadForSource("Text('hi')", { preload: { fonts: true } }, explicit),
    ).rejects.toThrowError("offline");
  });

  it("propagates mathjax load failures", async () => {
    const loaders = mocks();
    loaders.mathjax.mockRejectedValueOnce(new Error("chunk 404"));
    await expect(preloadForSource("MathTex('x')", {}, loaders)).rejects.toThrowError("chunk 404");
  });
});
