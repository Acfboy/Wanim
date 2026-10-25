// Pure helpers behind `<wanim-scene>`.  The element itself is exercised in a
// real browser (demo/component.html under Playwright); here we only cover the
// parsing/validation helpers and the Node/SSR import safety.
import { describe, expect, it } from "vitest";
import {
  formatWanimError,
  normalizePythonSource,
  parsePreloadModules,
  qualityCanvasSize,
  registerWanimScene,
  resolveAutoplayAttribute,
  resolveLoopAttribute,
  WanimSceneElement,
} from "../src/element.js";

describe("resolveAutoplayAttribute", () => {
  it("defaults to true when the attribute is absent", () => {
    expect(resolveAutoplayAttribute(null)).toBe(true);
  });

  it("treats bare and unknown values as true", () => {
    expect(resolveAutoplayAttribute("")).toBe(true);
    expect(resolveAutoplayAttribute("autoplay")).toBe(true);
  });

  it('only the literal "false" opts out', () => {
    expect(resolveAutoplayAttribute("false")).toBe(false);
  });
});

describe("resolveLoopAttribute", () => {
  it("defaults to false and enables on presence", () => {
    expect(resolveLoopAttribute(null)).toBe(false);
    expect(resolveLoopAttribute("")).toBe(true);
    expect(resolveLoopAttribute("false")).toBe(false);
  });
});

describe("parsePreloadModules", () => {
  it("returns an empty list when unset", () => {
    expect(parsePreloadModules(null)).toEqual([]);
    expect(parsePreloadModules("   ")).toEqual([]);
  });

  it("splits on whitespace and de-duplicates", () => {
    expect(parsePreloadModules("mathjax fonts")).toEqual(["mathjax", "fonts"]);
    expect(parsePreloadModules("fonts  fonts")).toEqual(["fonts"]);
  });

  it("rejects unknown module names", () => {
    expect(() => parsePreloadModules("mathjax bogus")).toThrowError(/Unknown preload module/);
  });
});

describe("qualityCanvasSize", () => {
  it("maps the manim quality flags", () => {
    expect(qualityCanvasSize("-ql")).toEqual([854, 480]);
    expect(qualityCanvasSize("-qm")).toEqual([1280, 720]);
    expect(qualityCanvasSize("-qh")).toEqual([1920, 1080]);
    expect(qualityCanvasSize("-q4k")).toEqual([3840, 2160]);
  });

  it("falls back to -qm for unknown flags", () => {
    expect(qualityCanvasSize("-qx")).toEqual([1280, 720]);
  });
});

describe("formatWanimError", () => {
  it("uses the message, then the name, then String()", () => {
    expect(formatWanimError(new Error("boom"))).toBe("boom");
    expect(formatWanimError(new TypeError(""))).toBe("TypeError");
    expect(formatWanimError("plain")).toBe("plain");
  });
});

describe("normalizePythonSource", () => {
  it("drops the whitespace-only line before a closing script tag", () => {
    const source = "\nfrom manim import *\n\nclass Demo(Scene):\n    pass\n      ";
    expect(normalizePythonSource(source)).toBe("from manim import *\n\nclass Demo(Scene):\n    pass\n");
  });

  it("dedents an indented block like Python textwrap", () => {
    const source = "\n    from manim import *\n    class Demo(Scene):\n        def construct(self):\n            pass\n";
    expect(normalizePythonSource(source)).toBe(
      "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        pass\n",
    );
  });

  it("folds CRLF and leaves a normal file untouched", () => {
    expect(normalizePythonSource("a = 1\r\nb = 2\r\n")).toBe("a = 1\nb = 2\n");
    expect(normalizePythonSource("a = 1\nb = 2\n")).toBe("a = 1\nb = 2\n");
  });

  it("handles empty input", () => {
    expect(normalizePythonSource("\n   \n")).toBe("");
  });
});

describe("Node/SSR safety", () => {
  it("imports without a DOM and the class is constructible", () => {
    expect(typeof WanimSceneElement).toBe("function");
    expect(new WanimSceneElement()).toBeInstanceOf(WanimSceneElement);
  });

  it("registerWanimScene is a no-op without customElements", () => {
    expect(registerWanimScene()).toBe(false);
  });

  it("defaults the theme preference to auto", () => {
    expect(new WanimSceneElement().theme).toBe("auto");
  });
});
