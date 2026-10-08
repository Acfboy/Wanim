// Optional-module loader: lazy chunk semantics, idempotency and status API.
//
// `vi.resetModules()` gives each test a fresh copy of the module graph, which
// is what makes the "not loaded yet" assertions possible: the shared setup
// file already preloaded MathJax for the rest of the suite.
import { describe, expect, it, vi } from "vitest";

describe("mathjax loader", () => {
  it("throws a clear error before preload_mathjax", async () => {
    vi.resetModules();
    const mathjax = await import("../src/mobject/text/mathjax.js");
    expect(mathjax.is_mathjax_loaded()).toBe(false);
    expect(() => mathjax.texToSvg("x^2")).toThrowError(/preload_mathjax/);
    expect(() => mathjax.texToSvgCached("x^2")).toThrowError(/preload_mathjax/);
  });

  it("preload_mathjax is idempotent and makes texToSvg synchronous again", async () => {
    vi.resetModules();
    const mathjax = await import("../src/mobject/text/mathjax.js");
    const first = mathjax.preload_mathjax();
    const second = mathjax.preload_mathjax();
    expect(second).toBe(first);
    await Promise.all([first, second]);
    expect(mathjax.is_mathjax_loaded()).toBe(true);
    await expect(mathjax.preload_mathjax()).resolves.toBeUndefined();
    expect(mathjax.texToSvg("x^2")).toContain("<svg");
    // The cached variant reuses the sync path.
    expect(mathjax.texToSvgCached("x^2")).toBe(mathjax.texToSvgCached("x^2"));
  });
});

describe("preload_modules", () => {
  it("loads the requested modules together and reports status", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    expect(loader.modules_status()).toEqual({ mathjax: false, fonts: false });

    await loader.preload_modules({ mathjax: true, fonts: true });

    expect(loader.modules_status()).toEqual({ mathjax: true, fonts: true });
    expect(loader.is_module_loaded("mathjax")).toBe(true);
    expect(loader.is_module_loaded("fonts")).toBe(true);
  });

  it("does nothing for an empty plan", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    await loader.preload_modules({});
    expect(loader.modules_status()).toEqual({ mathjax: false, fonts: false });
  });

  it("rejects unknown module names", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    expect(() => loader.is_module_loaded("sprites" as never)).toThrowError(/unknown optional module/);
  });
});
