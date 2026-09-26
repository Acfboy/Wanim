// Port of manim-wasm's PangoJS `decorator.js` (underline / strikethrough /
// overline path generation). The DOM-based `renderDecorationsToSVG` helper is
// omitted since the Node string path builds the SVG text directly.

import type { FontMetrics, PositionedGlyph } from "./font-loader.js";
import type { StyledRun } from "./markup-parser.js";

export interface Decoration {
  type: "path";
  d: string;
  stroke: string;
  strokeWidth: number;
  fill: string;
}

function generateWavyPath(x1: number, y: number, x2: number, amplitude = 3): string {
  const parts: string[] = [];
  parts.push(`M ${x1.toFixed(2)} ${y.toFixed(2)}`);

  const wavelength = 6;
  let x = x1;

  while (x < x2) {
    const nextX = Math.min(x + wavelength / 2, x2);
    const midX = (x + nextX) / 2;
    parts.push(`Q ${midX.toFixed(2)} ${(y - amplitude).toFixed(2)} ${nextX.toFixed(2)} ${y.toFixed(2)}`);
    x = nextX;
  }

  return parts.join(" ");
}

interface DecorationStyles {
  underlineStyle: string | null;
  underlineColor: string | null;
  strikethroughEnabled: boolean;
  strikethroughColor: string | null;
  overlineStyle: string | null;
  overlineColor: string | null;
}

function collectDecorationStyles(glyphPositions: PositionedGlyph[], runs: StyledRun[]): DecorationStyles {
  let underlineStyle: string | null = null;
  let underlineColor: string | null = null;
  let strikethroughEnabled = false;
  let strikethroughColor: string | null = null;
  let overlineStyle: string | null = null;
  let overlineColor: string | null = null;

  for (const glyph of glyphPositions) {
    const run = runs[glyph.runIndex];
    if (!run) continue;
    const style = run.styles as Record<string, unknown>;

    if (style.underline && style.underline !== "none") underlineStyle = style.underline as string;
    if (style.underline_color) underlineColor = style.underline_color as string;
    if (style.strikethrough === true) strikethroughEnabled = true;
    if (style.strikethrough_color) strikethroughColor = style.strikethrough_color as string;
    if (style.overline && style.overline !== "none") overlineStyle = style.overline as string;
    if (style.overline_color) overlineColor = style.overline_color as string;
  }

  return { underlineStyle, underlineColor, strikethroughEnabled, strikethroughColor, overlineStyle, overlineColor };
}

/** Generate decoration path data (mirrors PangoJS' `generateDecorations`). */
export function generateDecorations(
  glyphPositions: PositionedGlyph[],
  runs: StyledRun[],
  metrics: FontMetrics,
  fontSize: number,
  baselineY: number,
): Decoration[] {
  const decorations: Decoration[] = [];
  if (!glyphPositions || glyphPositions.length === 0 || !runs) return decorations;

  const firstGlyph = glyphPositions[0] as PositionedGlyph;
  const lastGlyph = glyphPositions[glyphPositions.length - 1] as PositionedGlyph;

  const startX = firstGlyph.x + (firstGlyph.leftSideBearing || 0);
  const endX = lastGlyph.x + lastGlyph.advanceWidth;

  const scale = fontSize;
  const underlineY = baselineY - metrics.underlinePosition * scale;
  const underlineThickness = Math.max(metrics.underlineThickness * scale, 1);
  const strikethroughY = baselineY - metrics.strikethroughPosition * scale;
  const strikethroughThickness = Math.max(metrics.strikethroughThickness * scale, 1);
  const overlineY = baselineY - metrics.ascender * scale;

  const {
    underlineStyle,
    underlineColor,
    strikethroughEnabled,
    strikethroughColor,
    overlineStyle,
    overlineColor,
  } = collectDecorationStyles(glyphPositions, runs);

  if (underlineStyle && underlineStyle !== "none") {
    const color = underlineColor || "#000000";
    if (underlineStyle === "single" || underlineStyle === "low") {
      decorations.push({
        type: "path",
        d: `M ${startX.toFixed(2)} ${underlineY.toFixed(2)} L ${endX.toFixed(2)} ${underlineY.toFixed(2)}`,
        stroke: color,
        strokeWidth: underlineThickness,
        fill: "none",
      });
    } else if (underlineStyle === "double") {
      const offset = underlineThickness * 2;
      decorations.push({
        type: "path",
        d: `M ${startX.toFixed(2)} ${underlineY.toFixed(2)} L ${endX.toFixed(2)} ${underlineY.toFixed(2)}`,
        stroke: color,
        strokeWidth: underlineThickness,
        fill: "none",
      });
      decorations.push({
        type: "path",
        d: `M ${startX.toFixed(2)} ${(underlineY + offset).toFixed(2)} L ${endX.toFixed(2)} ${(underlineY + offset).toFixed(2)}`,
        stroke: color,
        strokeWidth: underlineThickness,
        fill: "none",
      });
    } else if (underlineStyle === "error") {
      decorations.push({
        type: "path",
        d: generateWavyPath(startX, underlineY, endX, underlineThickness),
        stroke: color,
        strokeWidth: underlineThickness,
        fill: "none",
      });
    }
  }

  if (strikethroughEnabled) {
    decorations.push({
      type: "path",
      d: `M ${startX.toFixed(2)} ${strikethroughY.toFixed(2)} L ${endX.toFixed(2)} ${strikethroughY.toFixed(2)}`,
      stroke: strikethroughColor || "#000000",
      strokeWidth: strikethroughThickness,
      fill: "none",
    });
  }

  if (overlineStyle && overlineStyle !== "none") {
    decorations.push({
      type: "path",
      d: `M ${startX.toFixed(2)} ${overlineY.toFixed(2)} L ${endX.toFixed(2)} ${overlineY.toFixed(2)}`,
      stroke: overlineColor || "#000000",
      strokeWidth: underlineThickness,
      fill: "none",
    });
  }

  return decorations;
}
