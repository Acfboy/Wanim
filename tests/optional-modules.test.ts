// Optional-submodule loader: lazy chunk semantics, name normalization,
// idempotency/concurrency and the status API.
//
// `vi.resetModules()` gives each test a fresh copy of the module graph, which
// is what makes the "not loaded yet" assertions possible: the shared setup
// file already preloaded MathJax for the rest of the suite.
import { describe, expect, it, vi } from "vitest";
// The default chain's Latin font, fetched by the loader on Node (fetch in
// `readBinaryFileAsync`); requires network access.
import { DEFAULT_FONT_URLS } from "../src/mobject/text/pango/font-loader.js";

const LATIN = DEFAULT_FONT_URLS[0] as string;

describe("preload_submodules", () => {
  it("is a no-op when names are omitted or empty", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    expect(loader.submodule_status()).toEqual({ fonts: false, mathjax: false });

    await expect(loader.preload_submodules()).resolves.toBeUndefined();
    await expect(loader.preload_submodules([])).resolves.toBeUndefined();

    expect(loader.submodule_status()).toEqual({ fonts: false, mathjax: false });
  });

  it("loads a single named submodule", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    await loader.preload_submodules("mathjax");
    expect(loader.submodule_status()).toEqual({ fonts: false, mathjax: true });
  });

  it("loads an array of names and de-duplicates it", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    await loader.preload_submodules(["mathjax", "mathjax"]);
    expect(loader.submodule_status()).toEqual({ fonts: false, mathjax: true });
  });

  it("shares one pending load between concurrent calls", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    const mathjax = await import("../src/mobject/text/mathjax.js");

    expect(mathjax.is_mathjax_loaded()).toBe(false);
    const first = loader.preload_submodules("mathjax");
    const second = loader.preload_submodules("mathjax");
    // The underlying loader caches its pending promise, so both calls wait on
    // the very same load.
    expect(mathjax.preload_mathjax()).toBe(mathjax.preload_mathjax());
    await Promise.all([first, second]);
    expect(loader.submodule_status().mathjax).toBe(true);
    // Repeated calls resolve immediately.
    await expect(loader.preload_submodules("mathjax")).resolves.toBeUndefined();
    expect(mathjax.is_mathjax_loaded()).toBe(true);
  });

  it("loads explicit font paths and reports the active chain in the status", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    const pango = await import("../src/mobject/text/pango/svg-renderer.js");
    const fontLoader = await import("../src/mobject/text/pango/font-loader.js");

    await loader.preload_submodules("fonts", { fonts: [LATIN] });
    expect(fontLoader.fontCache.has(LATIN)).toBe(true);
    // The one-off paths do not change the active chain used by the status.
    expect(loader.submodule_status().fonts).toBe(false);

    pango.configure_fonts([LATIN]);
    expect(loader.submodule_status().fonts).toBe(true);
    await expect(loader.preload_submodules("fonts")).resolves.toBeUndefined();
    pango.configure_fonts(null);
  });

  it("rejects unknown names with a guided ValueError", async () => {
    vi.resetModules();
    const loader = await import("../src/loader.js");
    const error = await loader.preload_submodules("sprites" as never).catch((thrown) => thrown);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("ValueError");
    expect(error.message).toMatch(/unknown submodule "sprites"/);
    expect(error.message).toMatch(/fonts/);
    expect(error.message).toMatch(/mathjax/);
    // Nothing was loaded before the name was rejected.
    expect(loader.submodule_status()).toEqual({ fonts: false, mathjax: false });
  });
});
