// Port of manim-wasm's PangoJS `svg-renderer.js` (string-producing path only).
//
// `renderToSVGString` mirrors the reference implementation. `renderToSVGElement`
// / `renderToContainer` (DOM-based) are intentionally not ported: the whole
// port runs in Node without a DOM. The SVG string produced here is consumed by
// `SVGMobject` (see ../../svg/svg_mobject.ts) which converts it to VMobject
// points.

import type { Font } from "opentype.js";
import { parseMarkup, buildStyledRuns, type StyledRun } from "./markup-parser.js";
import { getFontMetrics, getTextPaths, loadFontSync, resolveDefaultFontPath, parseFontFromBuffer, type FontMetrics } from "./font-loader.js";
import { generateDecorations, type Decoration } from "./decorator.js";

const DEFAULT_FONT_SIZE = 48;
const svgNS = "http://www.w3.org/2000/svg";

export interface PangoRendererOptions {
  fontSize?: number;
  fontUrl?: string;
  /** Path to a local `.ttf`/`.otf` file (Node replacement for `fontUrl`). */
  fontPath?: string;
  color?: string;
  fontFamily?: string;
}

export interface RenderOptions {
  fontSize?: number;
  color?: string;
}

export class PangoRenderer {
  font: Font | null = null;
  metrics: FontMetrics | null = null;
  fontSize: number;
  fontUrl: string;
  fontPath: string | null;
  defaultColor: string;
  defaultFontFamily: string;

  constructor(options: PangoRendererOptions = {}) {
    this.fontSize = options.fontSize ?? DEFAULT_FONT_SIZE;
    this.fontPath = options.fontPath ?? null;
    this.fontUrl = options.fontUrl ?? this.fontPath ?? "";
    this.defaultColor = options.color ?? "#000000";
    this.defaultFontFamily = options.fontFamily ?? "DejaVu Sans, sans-serif";
  }

  /** Load the font from a local file path (Node) or the bundled default. */
  async loadFont(fontPath: string = this.fontPath ?? resolveDefaultFontPath()): Promise<Font> {
    this.font = loadFontSync(fontPath);
    this.fontPath = fontPath;
    this.metrics = getFontMetrics(this.font);
    return this.font;
  }

  /** Synchronous font load from a file path / the bundled default. */
  loadFontSync(fontPath: string = this.fontPath ?? resolveDefaultFontPath()): Font {
    this.font = loadFontSync(fontPath);
    this.fontPath = fontPath;
    this.metrics = getFontMetrics(this.font);
    return this.font;
  }

  /** Set the font from an already-parsed opentype Font. */
  setFont(font: Font): void {
    this.font = font;
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

    if (!this.font) {
      // Auto-load the bundled font so callers do not have to.
      this.loadFontSync();
    }
    const font = this.font as Font;
    const metrics = this.metrics ?? getFontMetrics(font);

    const { text, attributes } = parseMarkup(markup);
    const runs = buildStyledRuns(text, attributes);

    for (const run of runs) {
      const styles = run.styles as Record<string, unknown>;
      if (!styles.foreground) styles.foreground = color;
      if (!styles.font_family) styles.font_family = this.defaultFontFamily;
      if (!styles.font_size) styles.font_size = fontSize;
    }

    const glyphPositions = getTextPaths(font, text, 0, 0, fontSize, { runs });

    const padding = fontSize * 0.2;

    let maxWidth = 0;
    for (const glyph of glyphPositions) {
      const glyphEnd = glyph.x + glyph.advanceWidth;
      if (glyphEnd > maxWidth) maxWidth = glyphEnd;
    }

    const baselineY = fontSize * 1.2;
    const textHeight = fontSize * 1.5;

    const decorations = generateDecorations(glyphPositions, runs, metrics, fontSize, baselineY);

    const width = Math.max(maxWidth + padding * 2, 100);
    const height = textHeight + padding * 2 + fontSize * 0.5;

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
      let yOffset = baselineY + padding;
      if (style.baseline_shift) {
        yOffset += (style.baseline_shift as number) * fontSize;
      }

      let pathData = glyph.path;
      const glyphObj = font.glyphs.get(glyph.glyphIndex);
      if (glyphObj) pathData = glyphObj.getPath(0, 0, fontSize).toPathData(2);

      svgContent += `  <path d="${pathData}" fill="${currentColor}" transform="translate(${pathX.toFixed(2)}, ${yOffset.toFixed(2)})" />\n`;
    }

    svgContent += "</svg>";
    return svgContent;
  }
}

let defaultRenderer: PangoRenderer | null = null;

/** Shared renderer using the bundled DejaVu Sans font. */
export function getDefaultPangoRenderer(): PangoRenderer {
  if (defaultRenderer === null) {
    defaultRenderer = new PangoRenderer();
    defaultRenderer.loadFontSync(resolveDefaultFontPath());
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
  await renderer.loadFont();
  return renderer.renderToSVGString(markup, options);
}

export type { Decoration };
