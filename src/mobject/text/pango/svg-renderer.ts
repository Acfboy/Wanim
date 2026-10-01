// Port of manim-wasm's PangoJS `svg-renderer.js` (string-producing path only).
//
// `renderToSVGString` mirrors the reference implementation. `renderToSVGElement`
// / `renderToContainer` (DOM-based) are intentionally not ported: the whole
// port runs in Node without a DOM. The SVG string produced here is consumed by
// `SVGMobject` (see ../../svg/svg_mobject.ts) which converts it to VMobject
// points.

import type { Font } from "opentype.js";
import { parseMarkup, buildStyledRuns, type StyledRun } from "./markup-parser.js";
import {
  getFontMetrics,
  getTextPaths,
  loadFontSync,
  loadFontsSync,
  preloadFonts,
  resolveDefaultFontPath,
  resolveDefaultFontPaths,
  parseFontFromBuffer,
  pathToSvgData,
  type FontMetrics,
} from "./font-loader.js";
import { generateDecorations, type Decoration } from "./decorator.js";

const DEFAULT_FONT_SIZE = 48;
const svgNS = "http://www.w3.org/2000/svg";

export interface PangoRendererOptions {
  fontSize?: number;
  fontUrl?: string;
  /** Path to a local `.ttf`/`.otf` file (Node replacement for `fontUrl`). */
  fontPath?: string;
  /** Fallback chain of `.ttf`/`.otf` files; takes precedence over `fontPath`. */
  fontPaths?: string[];
  color?: string;
  fontFamily?: string;
}

export interface RenderOptions {
  fontSize?: number;
  color?: string;
  /** Baseline-to-baseline distance (same user units as `fontSize`). */
  lineSpacing?: number;
}

export class PangoRenderer {
  fonts: Font[] = [];
  metrics: FontMetrics | null = null;
  fontSize: number;
  fontUrl: string;
  fontPath: string | null;
  fontPaths: string[] | null;
  defaultColor: string;
  defaultFontFamily: string;

  constructor(options: PangoRendererOptions = {}) {
    this.fontSize = options.fontSize ?? DEFAULT_FONT_SIZE;
    this.fontPath = options.fontPath ?? null;
    this.fontPaths = options.fontPaths ?? null;
    this.fontUrl = options.fontUrl ?? this.fontPath ?? "";
    this.defaultColor = options.color ?? "#000000";
    this.defaultFontFamily = options.fontFamily ?? "Times New Roman, FZShuSong, sans-serif";
  }

  /** Primary (first) font of the chain. */
  get font(): Font | null {
    return this.fonts[0] ?? null;
  }

  /** Load a single font from a local file path (Node) or the default. */
  async loadFont(fontPath: string = this.fontPath ?? resolveDefaultFontPath()): Promise<Font> {
    this.fonts = [loadFontSync(fontPath)];
    this.fontPath = fontPath;
    this.fontPaths = null;
    this.metrics = getFontMetrics(this.fonts[0] as Font);
    return this.fonts[0] as Font;
  }

  /** Synchronous single-font load. */
  loadFontSync(fontPath: string = this.fontPath ?? resolveDefaultFontPath()): Font {
    this.fonts = [loadFontSync(fontPath)];
    this.fontPath = fontPath;
    this.fontPaths = null;
    this.metrics = getFontMetrics(this.fonts[0] as Font);
    return this.fonts[0] as Font;
  }

  /** Synchronous fallback-chain load (defaults to the shipped fonts). */
  loadFontsSync(fontPaths: string[] = this.fontPaths ?? resolveDefaultFontPaths()): Font[] {
    this.fonts = loadFontsSync(fontPaths);
    this.fontPaths = fontPaths;
    this.metrics = getFontMetrics(this.fonts[0] as Font);
    return this.fonts;
  }

  /** Async fallback-chain load (defaults to the shipped fonts). */
  async loadFonts(fontPaths: string[] = this.fontPaths ?? resolveDefaultFontPaths()): Promise<Font[]> {
    return this.loadFontsSync(fontPaths);
  }

  /** Set the font from an already-parsed opentype Font. */
  setFont(font: Font): void {
    this.fonts = [font];
    this.fontPaths = null;
    this.metrics = getFontMetrics(font);
  }

  /** Parse a font from an ArrayBuffer and use it. */
  setFontFromBuffer(buffer: ArrayBuffer): void {
    this.setFont(parseFontFromBuffer(buffer));
  }

  /** Render Pango markup to an SVG string. */
  renderToSVGString(markup: string, options: RenderOptions = {}): string {
    const fontSize = options.fontSize ?? this.fontSize;
    const color = options.color ?? this.defaultColor;
    const lineSpacing = options.lineSpacing ?? 0;

    if (this.fonts.length === 0) {
      // Auto-load the default font chain so callers do not have to.
      this.loadFontsSync();
    }
    const fonts = this.fonts;
    const metrics = this.metrics ?? getFontMetrics(fonts[0] as Font);

    const { text, attributes } = parseMarkup(markup);
    const runs = buildStyledRuns(text, attributes);

    for (const run of runs) {
      const styles = run.styles as Record<string, unknown>;
      if (!styles.foreground) styles.foreground = color;
      if (!styles.font_family) styles.font_family = this.defaultFontFamily;
      if (!styles.font_size) styles.font_size = fontSize;
    }

    const glyphPositions = getTextPaths(fonts, text, 0, 0, fontSize, { runs, lineSpacing });

    const padding = fontSize * 0.2;

    let maxWidth = 0;
    for (const glyph of glyphPositions) {
      const glyphEnd = glyph.x + glyph.advanceWidth;
      if (glyphEnd > maxWidth) maxWidth = glyphEnd;
    }

    const baselineY = fontSize * 1.2;
    const textHeight = fontSize * 1.5;
    const lineCount = text.split("\n").length;

    const decorations = generateDecorations(glyphPositions, runs, metrics, fontSize, baselineY);

    const width = Math.max(maxWidth + padding * 2, 100);
    const height =
      textHeight + (lineCount - 1) * lineSpacing + padding * 2 + fontSize * 0.5;

    let svgContent = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="${svgNS}" viewBox="0 0 ${width.toFixed(2)} ${height.toFixed(2)}" width="${width.toFixed(2)}" height="${height.toFixed(2)}">
  <style>
    .text { font-family: sans-serif; }
  </style>
`;
    for (const dec of decorations) {
      svgContent += `  <path d="${dec.d}" stroke="${dec.stroke}" stroke-width="${dec.strokeWidth.toFixed(2)}" fill="none" transform="translate(${padding.toFixed(2)}, ${padding.toFixed(2)})" />\n`;
    }

    for (const glyph of glyphPositions) {
      const run = runs[glyph.runIndex] as StyledRun | undefined;
      const style = (run ? run.styles : {}) as Record<string, unknown>;

      const currentColor = (style.foreground as string) || color;
      const pathX = glyph.x + padding;
      let yOffset = baselineY + padding + glyph.y;
      if (style.baseline_shift) {
        yOffset += (style.baseline_shift as number) * fontSize;
      }

      let pathData = glyph.path;
      const glyphFont = fonts[glyph.fontIndex] ?? fonts[0];
      const glyphObj = glyphFont ? glyphFont.glyphs.get(glyph.glyphIndex) : undefined;
      if (glyphObj) pathData = pathToSvgData(glyphObj.getPath(0, 0, fontSize));

      svgContent += `  <path d="${pathData}" fill="${currentColor}" transform="translate(${pathX.toFixed(2)}, ${yOffset.toFixed(2)})" />\n`;
    }

    svgContent += "</svg>";
    return svgContent;
  }
}

let defaultRenderer: PangoRenderer | null = null;
let configuredFontPaths: string[] | null = null;

/**
 * Set the font chain used by every subsequently created `Text` / `MarkupText`
 * / `Paragraph` (and the number mobjects built from them).
 *
 * This is the embedder-facing configuration: call it once at startup, before
 * the first text mobject is created.  Paths are URLs in the browser (must be
 * CORS-enabled, e.g. jsDelivr) and file paths in Node; the first font that has
 * a glyph for a character wins, so pass e.g. `[latinFont, cjkFont]`.
 *
 * `configure_fonts(null)` restores the environment default
 * (`resolveDefaultFontPaths()`).  Already-created text mobjects keep the
 * outlines they were built with.
 */
export function configure_fonts(fontPaths: readonly string[] | null): void {
  configuredFontPaths = fontPaths !== null && fontPaths.length > 0 ? [...fontPaths] : null;
  // Drop the shared renderer so the next text mobject loads the new chain.
  defaultRenderer = null;
}

/** The active font chain: `configure_fonts` override or the environment default. */
export function getDefaultFontPaths(): string[] {
  return configuredFontPaths ?? resolveDefaultFontPaths();
}

/**
 * Download and parse the active font chain ahead of the first `Text`.
 *
 * Companion to {@link configure_fonts}: start it at app startup, after the
 * runtime libraries have loaded, and let the returned promise settle in the
 * background.  Paths are the same as the renderer will use, so a later text
 * mobject finds the parsed fonts in the loader's cache and does no I/O.  See
 * `preloadFonts` in font-loader.ts for the error semantics.
 */
export async function preload_fonts(fontPaths?: readonly string[]): Promise<Font[]> {
  return preloadFonts(fontPaths === undefined ? getDefaultFontPaths() : [...fontPaths]);
}

/** Shared renderer using the configured/default font chain. */
export function getDefaultPangoRenderer(): PangoRenderer {
  if (defaultRenderer === null) {
    defaultRenderer = new PangoRenderer();
    defaultRenderer.loadFontsSync(getDefaultFontPaths());
  }
  return defaultRenderer;
}

/** Convenience synchronous markup-to-SVG renderer (Node replacement for
 * `js.pangoRenderMarkupSync` in the reference implementation). */
export function renderMarkupToSVGString(markup: string, options: RenderOptions = {}): string {
  return getDefaultPangoRenderer().renderToSVGString(markup, options);
}

/** Async convenience wrapper (mirrors PangoJS' `renderMarkupToSVG`). */
export async function renderMarkupToSVG(markup: string, options: RenderOptions = {}): Promise<string> {
  const renderer = new PangoRenderer(options);
  await renderer.loadFonts();
  return renderer.renderToSVGString(markup, options);
}

export type { Decoration };
