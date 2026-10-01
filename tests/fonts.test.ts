import { afterEach, describe, expect, it } from "vitest";
import {
  configure_fonts,
  getDefaultFontPaths,
  getDefaultPangoRenderer,
  getTextPaths,
  renderMarkupToSVGString,
} from "../src/mobject/text/pango/index.js";
import { Text } from "../src/mobject/text/index.js";

const LATIN = new URL("../assets/fonts/TimesNewRoman.ttf", import.meta.url).pathname;
const CJK = new URL("../assets/fonts/FZShuSong-Z01S.ttf", import.meta.url).pathname;

// `tests/setup.ts` pins WANIM_FONT to the host's DejaVu Sans for the
// differential tests, so the "restored" chain below is that single font.
describe("configure_fonts", () => {
  afterEach(() => configure_fonts(null));

  it("replaces the default chain and resets the shared renderer", () => {
    configure_fonts([LATIN, CJK]);
    expect(getDefaultFontPaths()).toEqual([LATIN, CJK]);

    const renderer = getDefaultPangoRenderer();
    expect(renderer.fonts).toHaveLength(2);
    // Times' unitsPerEm is 2048, FZShuSong's is 256.
    expect(renderer.fonts[0]?.unitsPerEm).toBe(2048);
    expect(renderer.fonts[1]?.unitsPerEm).toBe(256);

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
    configure_fonts([LATIN, CJK]);
    configure_fonts(null);
    expect(getDefaultFontPaths()).toHaveLength(1);
    expect(getDefaultPangoRenderer().fonts).toHaveLength(1);
  });
});
