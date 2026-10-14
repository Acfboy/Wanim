import { afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  configure_fonts,
  fontCache,
  getDefaultFontPaths,
  getDefaultPangoRenderer,
  getTextPaths,
  loadFontSync,
  renderMarkupToSVGString,
  DEFAULT_FONT_URLS,
} from "../src/mobject/text/pango/index.js";
// `preload_fonts` / `preloadFonts` are internal since the unified loader API
// (`preload_submodules("fonts")`); tested here through their deep paths.
import { preloadFonts } from "../src/mobject/text/pango/font-loader.js";
import { preload_fonts } from "../src/mobject/text/pango/svg-renderer.js";
import { Text } from "../src/mobject/text/index.js";

// The default chain is the browser pair (Tinos for Latin, Noto Serif SC for
// Chinese) served from jsDelivr.  Node preloads URLs with `fetch` (see
// `readBinaryFileAsync`), after which the synchronous load path reads the
// warmed cache.  These tests therefore need network access.
const LATIN = DEFAULT_FONT_URLS[0] as string;
const CJK = DEFAULT_FONT_URLS[1] as string;

beforeAll(async () => {
  await preloadFonts([LATIN, CJK]);
});

describe("configure_fonts", () => {
  afterEach(() => configure_fonts(null));

  it("replaces the default chain and resets the shared renderer", () => {
    configure_fonts([LATIN, CJK]);
    expect(getDefaultFontPaths()).toEqual([LATIN, CJK]);

    const renderer = getDefaultPangoRenderer();
    expect(renderer.fonts).toHaveLength(2);

    const glyphs = getTextPaths(renderer.fonts, "A中", 0, 0, 48);
    expect(glyphs.map((glyph) => glyph.char)).toEqual(["A", "中"]);
    expect(glyphs.map((glyph) => glyph.fontIndex)).toEqual([0, 1]);
  });

  it("renders text through the configured chain", () => {
    configure_fonts([LATIN, CJK]);
    const text = new Text("矩阵 A");
    expect(text.submobjects.length).toBe(3);
    const svg = renderMarkupToSVGString("中文 A", { fontSize: 16 });
    expect(svg).toContain("<path");
    expect(svg).not.toContain("NaN");
  });

  it("configure_fonts(null) restores the environment default", () => {
    configure_fonts([LATIN]);
    configure_fonts(null);
    expect(getDefaultFontPaths()).toHaveLength(1);
    expect(getDefaultPangoRenderer().fonts).toHaveLength(1);
  });
});

describe("preloadFonts", () => {
  afterEach(() => {
    configure_fonts(null);
    fontCache.clear();
  });

  it("parses the chain and seeds the shared cache", async () => {
    fontCache.delete(LATIN);
    const fonts = await preloadFonts([LATIN]);
    expect(fonts).toHaveLength(1);
    expect(fontCache.get(LATIN)).toBe(fonts[0]);
    // A later synchronous load must hit the warmed cache, not the network.
    expect(loadFontSync(LATIN)).toBe(fonts[0]);
  });

  it("preload_fonts follows configure_fonts", async () => {
    configure_fonts([LATIN, CJK]);
    const fonts = await preload_fonts();
    expect(fonts).toHaveLength(2);
    expect(fonts.map((font) => font.unitsPerEm).every((units) => units > 0)).toBe(true);
  });

  it("rejects when nothing can be loaded in Node", async () => {
    await expect(preloadFonts([LATIN, "/nonexistent/font.ttf"])).rejects.toThrow();
  });
});
