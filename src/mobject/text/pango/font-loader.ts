// Port of manim-wasm's PangoJS `font-loader.js` to Node/TypeScript.
//
// The reference implementation runs in the browser and loads fonts through
// opentype.js' URL loader. In Node we load the host's default font (or a
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
 * Thrown when the synchronous browser font path finds no parsed font.
 *
 * The browser has no non-blocking way to read a font synchronously, so
 * `loadFontSync` is cache-only there.  Call `preload_submodules("fonts")`
 * (or `preloadFonts`) first and let the async load finish; the synchronous
 * path then serves the warmed cache with no I/O.
 */
export class FontNotPreloadedError extends Error {
  readonly fontPath: string;

  constructor(fontPath: string, detail?: string) {
    super(
      `wanim: font ${fontPath} is not preloaded; ` +
        `await preload_submodules("fonts") (or preloadFonts) before building Text.` +
        (detail === undefined ? "" : ` (${detail})`),
    );
    this.name = "FontNotPreloadedError";
    this.fontPath = fontPath;
  }
}

/** True for http(s) URLs (fetched asynchronously on every platform). */
function isRemoteUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

/**
 * Location of the legacy single default font, relative to the bundle.
 *
 * wanim does not bundle a font.  In Node the host's system font is used (see
 * {@link resolveDefaultFontPath}); in the browser this path is only tried as a
 * last resort when nothing else loads (an embedder may serve a `fonts/`
 * directory next to the bundle).
 */
export const DEFAULT_FONT_PATH = new URL("./fonts/DejaVuSans.ttf", import.meta.url).pathname;

/**
 * Browser default chain, served from free public CDNs (jsDelivr) and pinned to
 * immutable upstream commits:
 *
 * - Tinos (Apache-2.0, 0.5 MB) is metric-compatible with Times New Roman.
 * - Noto Serif SC (OFL, Source Han Serif) from Fontsource: the
 *   `chinese-simplified` subset has ~8.1k glyphs (common simplified and
 *   traditional characters) in 2.3 MB. It is WOFF1, which opentype.js reads.
 *   The full Noto Serif SC OTF is 11.6 MB if more coverage is needed.
 *
 * Embedders can replace the chain with `configure_fonts([...])`; see
 * `getDefaultPangoRenderer` in ./svg-renderer.js.  In Node the same URLs can
 * be preloaded with `preloadFonts` (async `fetch`); the synchronous load path
 * then reads the warmed cache.
 */
export const DEFAULT_FONT_URLS = [
  "https://cdn.jsdelivr.net/gh/google/fonts@9710da1eacb3be272583c3224dcb70f9da6eadbb/ofl/tinos/Tinos-Regular.ttf",
  "https://cdn.jsdelivr.net/npm/@fontsource/noto-serif-sc@4.5.12/files/noto-serif-sc-chinese-simplified-400-normal.woff",
];

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
 * Resolve the default font chain.
 *
 * `WANIM_FONT` overrides the whole chain with a single file (used by the
 * differential tests, which pin DejaVu Sans).  `configure_fonts([...])` from
 * the package root is the embedder-facing equivalent and takes precedence over
 * everything here.  Otherwise the browser uses {@link DEFAULT_FONT_URLS} while
 * Node renders with the host's default English font
 * ({@link resolveDefaultFontPath}).
 */
export function resolveDefaultFontPaths(): string[] {
  const envOverride = typeof process !== "undefined" ? process.env?.WANIM_FONT : undefined;
  if (envOverride) return [envOverride];
  if (IS_BROWSER) return [...DEFAULT_FONT_URLS];
  return [resolveDefaultFontPath()];
}

/** Path of the legacy single default font (bundle dir in the browser). */
export function resolveLegacyFontPath(): string {
  if (IS_BROWSER) return new URL(`./fonts/${LEGACY_FONT_FILE}`, import.meta.url).pathname;
  return resolveDefaultFontPath();
}

/**
 * Read a binary file into an ArrayBuffer.
 *
 * Browser callers must go through the async cache: a synchronous network read
 * would block the page for the whole download.  Node still reads local files
 * synchronously (there is no UI thread to block) and rejects remote URLs.
 */
function readBinaryFile(filePath: string): ArrayBuffer {
  if (IS_BROWSER) {
    throw new FontNotPreloadedError(filePath, "the browser cache is cold");
  }
  if (isRemoteUrl(filePath)) {
    throw new FontNotPreloadedError(filePath, "remote fonts are only fetched asynchronously");
  }
  // `readFileSync(path, "base64")` returns a base64 string; decoding through
  // `atob` avoids depending on the Node `Buffer` type in this port.
  const base64 = readFileSync(path.resolve(filePath), { encoding: "base64" });
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

/**
 * Load and cache a font from a local file path.
 *
 * In the browser this is a cache-only read: preload the chain asynchronously
 * first.  A cold cache throws {@link FontNotPreloadedError} instead of
 * blocking on a synchronous XHR.  Node reads local files synchronously.
 */
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
 * In the browser a missing font is skipped when another font of the chain is
 * already cached; if none is cached this throws {@link FontNotPreloadedError}
 * so the host can await the async preload instead of silently blocking.  Node
 * still throws on explicit bad paths (and does not use the legacy fallback for
 * a non-empty chain).
 */
export function loadFontsFallbackSync(fontPaths: string[]): Font[] {
  const fonts: Font[] = [];
  const errors: unknown[] = [];
  for (const fontPath of fontPaths) {
    try {
      fonts.push(loadFontSync(fontPath));
    } catch (error) {
      if (!IS_BROWSER) throw error;
      errors.push(error);
    }
  }
  if (fonts.length === 0) {
    if (IS_BROWSER) {
      const first = errors.find((error) => error instanceof FontNotPreloadedError);
      if (first instanceof FontNotPreloadedError) throw first;
      throw new FontNotPreloadedError(
        fontPaths.join(", ") || "<default chain>",
        "no font in the chain is cached",
      );
    }
    try {
      fonts.push(loadFontSync(resolveLegacyFontPath()));
    } catch {
      throw new Error(
        `wanim: could not load any font from ${JSON.stringify(fontPaths)}; ` +
          "call configure_fonts([...]) with reachable TTF/OTF URLs.",
      );
    }
  }
  return fonts;
}

/** Load the shipped default chain (see {@link resolveDefaultFontPaths}). */
export function loadFontsSync(fontPaths: string[] = resolveDefaultFontPaths()): Font[] {
  return loadFontsFallbackSync(fontPaths);
}

/** Async binary read (browser `fetch`, Node `fs` read or `fetch` for URLs). */
async function readBinaryFileAsync(filePath: string): Promise<ArrayBuffer> {
  if (IS_BROWSER || isRemoteUrl(filePath)) {
    const response = await fetch(filePath);
    if (!response.ok) throw new Error(`Failed to load font ${filePath}: HTTP ${response.status}`);
    return response.arrayBuffer();
  }
  return readBinaryFile(filePath);
}

/**
 * Warm the in-memory font cache before the first text mobject is built.
 *
 * This is the async half of the font pipeline: download and parse the chain
 * without touching the synchronous renderer path.  Concurrent calls for the
 * same chain share one in-flight promise, so the startup warm-up and a scene's
 * preload cannot download or parse the same fonts twice.  A later `Text` then
 * finds the parsed fonts in {@link fontCache} and performs no I/O at all.
 *
 * Browser failures of individual fonts are skipped when at least one font
 * loads; a chain where nothing loads rejects so the caller can
 * `configure_fonts` a fallback chain.  Node rejects like `loadFontSync`.
 */
const PRELOAD_PROMISES = new Map<string, Promise<Font[]>>();

export function preloadFonts(fontPaths: string[] = resolveDefaultFontPaths()): Promise<Font[]> {
  const key = JSON.stringify(fontPaths);
  const pending = PRELOAD_PROMISES.get(key);
  if (pending !== undefined) return pending;
  // Keep only the in-flight promise: once it settles, a later call re-enters
  // the implementation, finds everything in FONT_CACHE and resolves without
  // parsing.  Dropping the settled entry also keeps failures (and explicit
  // cache resets in tests) from pinning a stale result.
  const promise = preloadFontsImpl(fontPaths).finally(() => {
    PRELOAD_PROMISES.delete(key);
  });
  PRELOAD_PROMISES.set(key, promise);
  return promise;
}

async function preloadFontsImpl(fontPaths: string[]): Promise<Font[]> {
  // Deduplicate before fetching: the default chain can repeat a fallback, and
  // concurrent duplicate downloads would parse the same font twice.
  const uniquePaths = [...new Set(fontPaths)];
  const results = await Promise.all(uniquePaths.map(async (fontPath) => {
    const cached = FONT_CACHE.get(fontPath);
    if (cached !== undefined) return { font: cached, error: null as unknown };
    try {
      const parsed = parseFontFromBuffer(await readBinaryFileAsync(fontPath));
      FONT_CACHE.set(fontPath, parsed);
      return { font: parsed, error: null as unknown };
    } catch (error) {
      return { font: null as Font | null, error };
    }
  }));
  const fonts: Font[] = [];
  const errors: unknown[] = [];
  for (const result of results) {
    if (result.font !== null) fonts.push(result.font);
    else {
      if (!IS_BROWSER) throw result.error;
      errors.push(result.error);
    }
  }
  if (fonts.length === 0) {
    throw new Error(`wanim: could not preload any font from ${JSON.stringify(fontPaths)}`, {
      cause: errors[0],
    });
  }
  return fonts;
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

/** Glyph outline `d` strings at the origin, keyed per font by
 * `glyphIndex|fontSize`.  The same character/size recurs across every Text
 * rebuilt during a drag (and across digit glyphs in DecimalNumber), so the
 * opentype `getPath` + coordinate formatting is done once. */
const GLYPH_PATH_DATA_CACHE = new WeakMap<Font, Map<string, string>>();
const GLYPH_PATH_DATA_LIMIT = 8192;

function glyphPathData(font: Font, glyphIndex: number, fontSize: number): string {
  let perFont = GLYPH_PATH_DATA_CACHE.get(font);
  if (perFont === undefined) {
    perFont = new Map<string, string>();
    GLYPH_PATH_DATA_CACHE.set(font, perFont);
  }
  const key = `${glyphIndex}|${fontSize}`;
  let data = perFont.get(key);
  if (data === undefined) {
    const glyph = font.glyphs.get(glyphIndex);
    data = glyph ? pathToSvgData(glyph.getPath(0, 0, fontSize)) : "";
    if (perFont.size >= GLYPH_PATH_DATA_LIMIT) perFont.clear();
    perFont.set(key, data);
  }
  return data;
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
 * for it (Latin from the primary font, CJK from a fallback, ...).
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
      // NOTE: the outline is emitted at the origin (and cached); the caller
      // places it with `x`/`y` (the SVG renderer emits the translate).
      paths.push({
        char,
        charIndex,
        glyphIndex,
        fontIndex,
        path: glyphPathData(font, glyphIndex, fontSize),
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
        paths.push({
          char,
          charIndex,
          glyphIndex: 0,
          fontIndex,
          path: glyphPathData(font, 0, fontSize),
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
