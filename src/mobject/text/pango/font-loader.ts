// Port of manim-wasm's PangoJS `font-loader.js` to Node/TypeScript.
//
// The reference implementation runs in the browser and loads fonts through
// opentype.js' URL loader. In Node we load the bundled DejaVu Sans font (or a
// caller-provided `.ttf`/`.otf`) from disk and parse it with
// `opentype.parse`. The glyph/run logic is otherwise unchanged.

import { existsSync, readFileSync } from "node:fs";
import * as path from "node:path";
import * as opentypeNs from "opentype.js";
import type { Font, Glyph, Path } from "opentype.js";
import type { StyledRun } from "./markup-parser.js";

// opentype.js ships both an ESM build (named exports, used by bundlers such as
// Vite/vitest) and a CJS/UMD build (single `module.exports` object, used by
// plain Node). Normalize both shapes here.
type OpentypeModule = typeof opentypeNs;
const opentype: OpentypeModule = ((opentypeNs as unknown as { default?: OpentypeModule }).default ??
  opentypeNs) as OpentypeModule;

const FONT_CACHE = new Map<string, Font>();

/** Bundled fallback font (DejaVu Sans), used when no font is specified. */
export const DEFAULT_FONT_PATH = new URL("./fonts/DejaVuSans.ttf", import.meta.url).pathname;

const SYSTEM_FONT_CANDIDATES = [
  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
];

/** Resolve the default font file, preferring the bundled asset. */
export function resolveDefaultFontPath(): string {
  if (existsSync(DEFAULT_FONT_PATH)) return DEFAULT_FONT_PATH;
  // When running from `dist/` the .ttf is not emitted by tsc; fall back to the
  // source tree (e.g. tests / dev servers run from the project root).
  const cwdCandidate = path.resolve(process.cwd(), "src", "mobject", "text", "pango", "fonts", "DejaVuSans.ttf");
  if (existsSync(cwdCandidate)) return cwdCandidate;
  for (const candidate of SYSTEM_FONT_CANDIDATES) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    `No font found. Expected the bundled font at ${DEFAULT_FONT_PATH}.`,
  );
}

/** Read a binary file into an ArrayBuffer using only the shimmed fs surface. */
function readBinaryFile(filePath: string): ArrayBuffer {
  // `readFileSync(path, "base64")` returns a base64 string; decoding through
  // `atob` avoids depending on the Node `Buffer` type in this port.
  const base64 = readFileSync(path.resolve(filePath), { encoding: "base64" });
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

/** Load and cache a font from a local file path. */
export function loadFontSync(fontPath: string = resolveDefaultFontPath()): Font {
  const cached = FONT_CACHE.get(fontPath);
  if (cached !== undefined) return cached;
  const parsed = parseFontFromBuffer(readBinaryFile(fontPath));
  FONT_CACHE.set(fontPath, parsed);
  return parsed;
}

/** Async variant of {@link loadFontSync} (mirrors PangoJS' `loadFont`). */
export async function loadFont(fontPath: string = resolveDefaultFontPath()): Promise<Font> {
  return loadFontSync(fontPath);
}

/** Parse a font from an ArrayBuffer (mirrors `opentype.parse`). */
export function parseFontFromBuffer(buffer: ArrayBuffer): Font {
  return opentype.parse(buffer);
}

/** Parse a font from a Uint8Array. */
export function loadFontFromBuffer(buffer: Uint8Array): Font {
  return parseFontFromBuffer(buffer.buffer as ArrayBuffer);
}

export interface FontMetrics {
  unitsPerEm: number;
  ascender: number;
  descender: number;
  height: number;
  underlinePosition: number;
  underlineThickness: number;
  strikethroughPosition: number;
  strikethroughThickness: number;
  capHeight: number;
  xHeight: number;
  lineGap: number;
}

/** Get normalized font metrics (mirrors PangoJS' `getFontMetrics`). */
export function getFontMetrics(font: Font): FontMetrics {
  const tables = font.tables as unknown as {
    hhea?: { lineGap?: number };
    os2?: {
      sTypoDescender?: number;
      usWinDescent?: number;
      sCapHeight?: number;
      sxHeight?: number;
    };
    post?: { underlinePosition?: number; underlineThickness?: number };
  };
  const unitsPerEm = font.unitsPerEm || 1000;
  const ascender = font.ascender || 0;
  const descender = font.descender || 0;
  const hhea = tables.hhea ?? {};
  const os2 = tables.os2 ?? {};
  const post = tables.post ?? {};

  const underlinePosition = os2.sTypoDescender ?? post.underlinePosition ?? -100;
  const underlineThickness = os2.usWinDescent ?? post.underlineThickness ?? 50;
  const strikethroughPosition = (os2.sCapHeight ?? ascender * 0.6) * 0.5;
  const strikethroughThickness = underlineThickness;

  return {
    unitsPerEm,
    ascender: ascender / unitsPerEm,
    descender: descender / unitsPerEm,
    height: (ascender - descender) / unitsPerEm,
    underlinePosition: underlinePosition / unitsPerEm,
    underlineThickness: underlineThickness / unitsPerEm,
    strikethroughPosition: strikethroughPosition / unitsPerEm,
    strikethroughThickness: strikethroughThickness / unitsPerEm,
    capHeight: (os2.sCapHeight ?? ascender * 0.7) / unitsPerEm,
    xHeight: (os2.sxHeight ?? ascender * 0.5) / unitsPerEm,
    lineGap: (hhea.lineGap ?? 0) / unitsPerEm,
  };
}

export interface GlyphPath {
  path: string;
  advanceWidth: number;
  leftSideBearing: number;
  glyph: Glyph;
}

/** Get the SVG path data for a single glyph (mirrors `getGlyphPath`). */
export function getGlyphPath(
  font: Font,
  glyphIndex: number,
  x: number,
  y: number,
  fontSize: number,
): GlyphPath | null {
  const glyph = font.glyphs.get(glyphIndex);
  if (!glyph) return null;
  const scale = fontSize / (font.unitsPerEm || 1000);
  const glyphPath: Path = glyph.getPath(x, y, fontSize);
  return {
    path: glyphPath.toPathData(2),
    advanceWidth: glyph.advanceWidth ? glyph.advanceWidth * scale : 0,
    leftSideBearing: glyph.leftSideBearing ? glyph.leftSideBearing * scale : 0,
    glyph,
  };
}

function findRunForCharIndex(runs: StyledRun[], charIndex: number): number {
  for (let i = 0; i < runs.length; i += 1) {
    const run = runs[i] as StyledRun;
    if (charIndex >= run.start && charIndex < run.end) return i;
  }
  return 0;
}

export interface PositionedGlyph {
  char: string;
  charIndex: number;
  glyphIndex: number;
  path: string;
  x: number;
  y: number;
  advanceWidth: number;
  width: number;
  leftSideBearing: number;
  runIndex: number;
}

/**
 * Lay out the glyphs of `text` at `(x, y)` and return each glyph's path data
 * (mirrors PangoJS' `getTextPaths`). Styled runs are used to attach a
 * `runIndex` to each glyph so that per-run styles (colour, baseline shift)
 * can be resolved by the renderer/decorator.
 */
export function getTextPaths(
  font: Font,
  text: string,
  x: number,
  y: number,
  fontSize: number,
  options: { runs?: StyledRun[] } = {},
): PositionedGlyph[] {
  const scale = fontSize / (font.unitsPerEm || 1000);
  const paths: PositionedGlyph[] = [];
  let currentX = x;

  let runs = options.runs;
  if (!runs || runs.length === 0) {
    runs = [{ text, start: 0, end: text.length, styles: {} }];
  }

  for (let charIndex = 0; charIndex < text.length; charIndex += 1) {
    const char = text[charIndex] as string;
    const runIndex = findRunForCharIndex(runs, charIndex);
    const glyphIndex = font.charToGlyphIndex(char);
    const glyph = font.glyphs.get(glyphIndex);

    if (glyph) {
      const glyphPath = glyph.getPath(currentX, y, fontSize);
      paths.push({
        char,
        charIndex,
        glyphIndex,
        path: glyphPath.toPathData(2),
        x: currentX,
        y,
        advanceWidth: (glyph.advanceWidth ?? 0) * scale,
        width: 0,
        leftSideBearing: glyph.leftSideBearing ? glyph.leftSideBearing * scale : 0,
        runIndex,
      });
      currentX += (glyph.advanceWidth ?? 0) * scale;
    } else {
      const notdefGlyph = font.glyphs.get(0);
      if (notdefGlyph) {
        const glyphPath = notdefGlyph.getPath(currentX, y, fontSize);
        paths.push({
          char,
          charIndex,
          glyphIndex: 0,
          path: glyphPath.toPathData(2),
          x: currentX,
          y,
          advanceWidth: (notdefGlyph.advanceWidth ?? 0) * scale,
          width: 0,
          leftSideBearing: notdefGlyph.leftSideBearing ? notdefGlyph.leftSideBearing * scale : 0,
          runIndex,
        });
        currentX += (notdefGlyph.advanceWidth ?? 0) * scale;
      }
    }
  }

  return paths;
}

/** Total advance width of `text` at `fontSize` (mirrors `getTextWidth`). */
export function getTextWidth(font: Font, text: string, fontSize: number): number {
  const scale = fontSize / (font.unitsPerEm || 1000);
  let width = 0;
  for (let i = 0; i < text.length; i += 1) {
    const glyphIndex = font.charToGlyphIndex(text[i] as string);
    const glyph = font.glyphs.get(glyphIndex);
    if (glyph) width += (glyph.advanceWidth ?? 0) * scale;
  }
  return width;
}

export const fontCache = FONT_CACHE;
