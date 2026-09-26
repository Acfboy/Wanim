// Barrel for the ported manim.mobject.text / manim.mobject.tex modules.
//
// NOTE: DecimalNumber / Integer / Variable remain in ./numbers.ts and are
// exported separately from the package root.

export {
  Text,
  MarkupText,
  Paragraph,
  TextSetting,
  remove_invisible_chars,
  register_font,
  PANGO_WEIGHT_MAP,
  PANGO_SLANT_MAP,
  TEXT_MOB_SCALE_FACTOR,
  DEFAULT_LINE_SPACING_SCALE,
  TEXT2SVG_ADJUSTMENT_FACTOR,
  type TextOptions,
  type MarkupTextOptions,
  type ParagraphOptions,
} from "./text_mobject.js";

export {
  SingleStringMathTex,
  MathTex,
  MathTexPart,
  Tex,
  Title,
  BulletedList,
  MATHTEX_SUBSTRING,
  TEX_MOB_SCALE_FACTOR,
  type SingleStringMathTexOptions,
  type MathTexOptions,
  type TexOptions,
  type BulletedListOptions,
  type TitleOptions,
} from "./tex_mobject.js";

export { texToSvg, texToSvgCached, mathjaxSvgCache, type TexToSvgOptions } from "./mathjax.js";

export * from "./pango/index.js";
