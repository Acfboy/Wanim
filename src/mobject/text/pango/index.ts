export {
  parseMarkup,
  buildStyledRuns,
  UNDERLINE_MAP,
  WEIGHT_MAP,
  SIZE_MAP,
  type MarkupAttribute,
  type ParsedMarkup,
  type StyledRun,
} from "./markup-parser.js";
export {
  loadFont,
  loadFontSync,
  loadFontFromBuffer,
  parseFontFromBuffer,
  resolveDefaultFontPath,
  DEFAULT_FONT_PATH,
  getFontMetrics,
  getGlyphPath,
  getTextPaths,
  getTextWidth,
  fontCache,
  type FontMetrics,
  type GlyphPath,
  type PositionedGlyph,
} from "./font-loader.js";
export { generateDecorations, type Decoration } from "./decorator.js";
export {
  PangoRenderer,
  getDefaultPangoRenderer,
  renderMarkupToSVGString,
  renderMarkupToSVG,
  type PangoRendererOptions,
  type RenderOptions,
} from "./svg-renderer.js";
