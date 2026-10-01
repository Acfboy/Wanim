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

/** True in a browser/worker (no synchronous filesystem). */
const IS_BROWSER = typeof window !== "undefined" || typeof (globalThis as { importScripts?: unknown }).importScripts === "function";

/**
 * Location of the default font.
 *
 * wanim does not bundle a font.  In Node the host's system font is used (see
 * {@link resolveDefaultFontPath}); in the browser this URL is fetched and the
 * demo server answers it with the host's default English font.
 */
export const DEFAULT_FONT_PATH = new URL("./fonts/DejaVuSans.ttf", import.meta.url).pathname;

/**
 * Font files shipped with wanim, in fallback order: the first font that has a
 * glyph for a character wins.  Times New Roman covers Latin/ASCII, FZShuSong
 * (方正书宋) covers CJK; together they mirror the usual Chinese textbook
 * convention of Times for Western text and a Song face for Chinese.
 */
export const DEFAULT_FONT_FILES = ["TimesNewRoman.ttf", "FZShuSong-Z01S.ttf"];

/** Legacy single-font name, kept as a last-resort browser fallback. */
export const LEGACY_FONT_FILE = "DejaVuSans.ttf";

const SYSTEM_FONT_CANDIDATES = [
  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
  "/System/Library/Fonts/Supplemental/Arial.ttf",
  "C:\\Windows\\Fonts\\arial.ttf",
];

/**
 * Resolve the default font file.
 *
 * The browser bundle is served next to a `fonts/` directory (provided by the
 * demo server, not the repository), so its URL-derived path is authoritative
 * there.  In Node we fall back to the source tree and then the system fonts.
 */
export function resolveDefaultFontPath(): string {
  if (IS_BROWSER) return DEFAULT_FONT_PATH;
  if (existsSync(DEFAULT_FONT_PATH)) return DEFAULT_FONT_PATH;
  // When running from `dist/` the font is not emitted by tsc; fall back to the
  // source tree (e.g. tests / dev servers run from the project root).
  const cwdCandidate = path.resolve(process.cwd(), "src", "mobject", "text", "pango", "fonts", "DejaVuSans.ttf");
  if (existsSync(cwdCandidate)) return cwdCandidate;
  for (const candidate of SYSTEM_FONT_CANDIDATES) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    `No font found. Install a system font or set one explicitly (looked for ${DEFAULT_FONT_PATH}).`,
  );
}

/**
 * Resolve the shipped default font chain.
 *
 * `WANIM_FONT` overrides the whole chain with a single file (used by the demo
 * scripts and the differential tests, which pin DejaVu Sans).  Otherwise the
 * browser fetches the shipped fonts next to the bundle while Node reads them
 * from `assets/fonts/`, falling back to the host's default English font when
 * that directory is absent (e.g. an installed package without assets).
 */
export function resolveDefaultFontPaths(): string[] {
  const envOverride = typeof process !== "undefined" ? process.env?.WANIM_FONT : undefined;
  if (envOverride) return [envOverride];
  if (IS_BROWSER) {
    return DEFAULT_FONT_FILES.map((file) => new URL(`./fonts/${file}`, import.meta.url).pathname);
  }
  const bundled = DEFAULT_FONT_FILES.map((file) => repoFontPath(file)).filter((file) => existsSync(file));
  if (bundled.length > 0) return bundled;
  return [resolveDefaultFontPath()];
}

/** Path of the legacy single default font (bundle dir in the browser). */
export function resolveLegacyFontPath(): string {
  if (IS_BROWSER) return new URL(`./fonts/${LEGACY_FONT_FILE}`, import.meta.url).pathname;
  return resolveDefaultFontPath();
}

/** Absolute path of a font file shipped in the package-level `assets/fonts/`. */
function repoFontPath(file: string): string {
  return new URL(`../../../../assets/fonts/${file}`, import.meta.url).pathname;
}

/**
 * Synchronously fetch `url` as raw bytes in the browser.  The font loader's
 * API is synchronous (`loadFontSync`) so a blocking XHR is the only option;
 * the font is served from the same origin as the bundle.
 */
function readBinaryFileBrowser(url: string): ArrayBuffer {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, false);
  xhr.overrideMimeType("text/plain; charset=x-user-defined");
  xhr.send();
  if (xhr.status !== 0 && (xhr.status < 200 || xhr.status >= 300)) {
    throw new Error(`Failed to load font ${url}: HTTP ${xhr.status}`);
  }
  const text = xhr.responseText;
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i += 1) bytes[i] = text.charCodeAt(i) & 0xff;
  return bytes.buffer;
}

/** Read a binary file into an ArrayBuffer (fs in Node, XHR in the browser). */
function readBinaryFile(filePath: string): ArrayBuffer {
  if (IS_BROWSER) return readBinaryFileBrowser(filePath);
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

/**
 * Load a fallback chain of fonts.
 *
 * In the browser a missing file (e.g. a deployment that predates the shipped
 * fonts) is skipped instead of aborting, and the legacy `DejaVuSans.ttf` is
 * tried when nothing else loaded; Node still throws on explicit bad paths.
 */
export function loadFontsFallbackSync(fontPaths: string[]): Font[] {
  const fonts: Font[] = [];
  for (const fontPath of fontPaths) {
    try {
      fonts.push(loadFontSync(fontPath));
    } catch (error) {
      if (!IS_BROWSER) throw error;
    }
  }
  if (fonts.length === 0) fonts.push(loadFontSync(resolveLegacyFontPath()));
  return fonts;
}

/** Load the shipped default chain (see {@link resolveDefaultFontPaths}). */
export function loadFontsSync(fontPaths: string[] = resolveDefaultFontPaths()): Font[] {
  return loadFontsFallbackSync(fontPaths);
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

/** Minimal shape of the opentype.js path commands we serialize. */
interface PathCommandLike {
  type: string;
  x?: number;
  y?: number;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
}

function formatCoordinate(value: number | undefined, factor: number): string {
  if (value === undefined || !Number.isFinite(value)) return "0";
  const rounded = Math.round(value * factor) / factor;
  return (Object.is(rounded, -0) ? 0 : rounded).toFixed(2);
}

/**
 * Serialize an opentype.js `Path` to SVG path data.
 *
 * opentype.js' own `Path.toPathData` rounds through string concatenation
 * (`Math.round(decimalPart + "e+" + places)`), which yields `NaN` when a
 * coordinate's fractional part serializes in exponent notation (e.g. `1e-7`).
 * Fine-scaled glyph outlines -- CJK glyphs at the default text size, in
 * particular -- hit this for some sizes, so wanim emits the path data itself.
 */
export function pathToSvgData(path: { commands: PathCommandLike[] }, decimalPlaces = 2): string {
  const factor = 10 ** decimalPlaces;
  const num = (value: number | undefined): string => formatCoordinate(value, factor);
  let d = "";
  for (const command of path.commands) {
    switch (command.type) {
      case "M":
        d += `M${num(command.x)} ${num(command.y)}`;
        break;
      case "L":
        d += `L${num(command.x)} ${num(command.y)}`;
        break;
      case "C":
        d += `C${num(command.x1)} ${num(command.y1)} ${num(command.x2)} ${num(command.y2)} ${num(command.x)} ${num(command.y)}`;
        break;
      case "Q":
        d += `Q${num(command.x1)} ${num(command.y1)} ${num(command.x)} ${num(command.y)}`;
        break;
      case "Z":
        d += "Z";
        break;
      default:
        break;
    }
  }
  return d;
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
    path: pathToSvgData(glyphPath),
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
  /** Index into the font chain that supplied this glyph. */
  fontIndex: number;
  path: string;
  x: number;
  y: number;
  advanceWidth: number;
  width: number;
  leftSideBearing: number;
  runIndex: number;
}

/**
 * Pick the first font in the chain that has a real glyph for `char`.
 * Falls back to the primary font's `.notdef` (index 0) when none has one.
 */
function selectFont(fonts: Font[], char: string): { font: Font; fontIndex: number; glyphIndex: number } {
  const first = fonts[0] as Font;
  let glyphIndex = first.charToGlyphIndex(char);
  if (glyphIndex !== 0) return { font: first, fontIndex: 0, glyphIndex };
  for (let index = 1; index < fonts.length; index += 1) {
    const font = fonts[index] as Font;
    const candidate = font.charToGlyphIndex(char);
    if (candidate !== 0) return { font, fontIndex: index, glyphIndex: candidate };
  }
  return { font: first, fontIndex: 0, glyphIndex: 0 };
}

/**
 * Lay out the glyphs of `text` at `(x, y)` and return each glyph's path data
 * (mirrors PangoJS' `getTextPaths`). Styled runs are used to attach a
 * `runIndex` to each glyph so that per-run styles (colour, baseline shift)
 * can be resolved by the renderer/decorator. When more than one font is given,
 * each character is drawn with the first font in the chain that has a glyph
 * for it (Latin from Times, CJK from FZShuSong, ...).
 */
export function getTextPaths(
  fonts: Font | Font[],
  text: string,
  x: number,
  y: number,
  fontSize: number,
  options: { runs?: StyledRun[]; lineSpacing?: number } = {},
): PositionedGlyph[] {
  const fontList = Array.isArray(fonts) ? fonts : [fonts];
  const lineSpacing = options.lineSpacing ?? 0;
  const paths: PositionedGlyph[] = [];
  let currentX = x;
  let lineIndex = 0;

  let runs = options.runs;
  if (!runs || runs.length === 0) {
    runs = [{ text, start: 0, end: text.length, styles: {} }];
  }

  for (let charIndex = 0; charIndex < text.length; charIndex += 1) {
    const char = text[charIndex] as string;
    if (char === "\n") {
      // Hard line break: reset the pen and advance one line (Pango treats
      // "\n" as a paragraph separator). No glyph is emitted for the break.
      currentX = x;
      lineIndex += 1;
      continue;
    }
    if (char === "\r") continue;
    const runIndex = findRunForCharIndex(runs, charIndex);
    const { font, fontIndex, glyphIndex } = selectFont(fontList, char);
    const scale = fontSize / (font.unitsPerEm || 1000);
    const glyph = font.glyphs.get(glyphIndex);
    const glyphY = y + lineIndex * lineSpacing;

    if (glyph) {
      const glyphPath = glyph.getPath(currentX, glyphY, fontSize);
      paths.push({
        char,
        charIndex,
        glyphIndex,
        fontIndex,
        path: pathToSvgData(glyphPath),
        x: currentX,
        y: glyphY,
        advanceWidth: (glyph.advanceWidth ?? 0) * scale,
        width: 0,
        leftSideBearing: glyph.leftSideBearing ? glyph.leftSideBearing * scale : 0,
        runIndex,
      });
      currentX += (glyph.advanceWidth ?? 0) * scale;
    } else {
      const notdefGlyph = font.glyphs.get(0);
      if (notdefGlyph) {
        const glyphPath = notdefGlyph.getPath(currentX, glyphY, fontSize);
        paths.push({
          char,
          charIndex,
          glyphIndex: 0,
          fontIndex,
          path: pathToSvgData(glyphPath),
          x: currentX,
          y: glyphY,
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
export function getTextWidth(fonts: Font | Font[], text: string, fontSize: number): number {
  const fontList = Array.isArray(fonts) ? fonts : [fonts];
  let width = 0;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i] as string;
    if (char === "\n" || char === "\r") continue;
    const { font, glyphIndex } = selectFont(fontList, char);
    const scale = fontSize / (font.unitsPerEm || 1000);
    const glyph = font.glyphs.get(glyphIndex);
    if (glyph) width += (glyph.advanceWidth ?? 0) * scale;
  }
  return width;
}

export const fontCache = FONT_CACHE;
