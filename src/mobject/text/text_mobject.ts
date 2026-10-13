// Port of manim.mobject.text.text_mobject (Text, MarkupText, Paragraph,
// TextSetting). Pango rendering is provided by the TypeScript port of PangoJS
// under ./pango (opentype.js glyph outlines -> SVG string), and the SVG is
// converted to VMobject points by SVGMobject (svg_string support).

import { array, type NDArray } from "numpy-ts";
import { bump_mobject_geometry_epoch } from "../../utils/render_epoch.js";
import { DEFAULT_FONT_SIZE, NORMAL } from "../../constants.js";
import { ManimColor, WHITE, color_gradient, type ParsableManimColor } from "../../utils/color/core.js";

import { VMobject, VGroup, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { SVGMobject, close_path_contours, type SVGMobjectOptions } from "../svg/svg_mobject.js";
import { Dot } from "../geometry/arc.js";
import { pointRows } from "../../utils/arrays.js";
import { renderMarkupToSVGString } from "./pango/svg-renderer.js";

export const TEXT_MOB_SCALE_FACTOR = 0.05;
export const DEFAULT_LINE_SPACING_SCALE = 0.3;
export const TEXT2SVG_ADJUSTMENT_FACTOR = 4.8;
// Pango interprets the font size in points and converts to device units at the
// default 96 dpi resolution (96/72 = 4/3); manimpango's SVG therefore carries
// glyphs whose em is `font_size / TEXT2SVG_ADJUSTMENT_FACTOR * 4/3`. The
// PangoJS port works in em units directly, so scale the requested size by this
// factor to match real manim's glyph dimensions. Line spacing, in contrast, is
// used by Pango verbatim in device units (no point conversion).
export const PANGO_POINT_TO_PX_FACTOR = 96 / 72;

export const PANGO_WEIGHT_MAP: Record<string, string> = {
  THIN: "100",
  ULTRALIGHT: "200",
  LIGHT: "300",
  SEMILIGHT: "350",
  BOOK: "380",
  NORMAL: "normal",
  MEDIUM: "500",
  SEMIBOLD: "600",
  BOLD: "bold",
  ULTRABOLD: "800",
  HEAVY: "900",
  ULTRAHEAVY: "1000",
};

export const PANGO_SLANT_MAP: Record<string, string> = {
  NORMAL: "normal",
  ITALIC: "italic",
  OBLIQUE: "oblique",
};

/** Shared `_close_glyph_contours` implementation for Text / MarkupText: the
 * per-path closing lives in `close_path_contours` (svg_mobject), which the
 * `close_path_contours` option applies once per cached glyph outline. */
function closeGlyphContours(mob: VMobject): void {
  for (const each of mob) close_path_contours(each as VMobject);
}

function get_pango_attrs_markup(font: string | null, weight: string | null, slant: string | null): string[] {
  const attrs: string[] = [];
  if (font) attrs.push(`font_family="${font}"`);
  const weightStr = (weight ?? "").toUpperCase();
  if (weightStr && weightStr in PANGO_WEIGHT_MAP) attrs.push(`weight="${PANGO_WEIGHT_MAP[weightStr]}"`);
  const slantStr = (slant ?? "").toUpperCase();
  if (slantStr && slantStr in PANGO_SLANT_MAP) attrs.push(`style="${PANGO_SLANT_MAP[slantStr]}"`);
  return attrs;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

/** Local replacement for manimpango.TextSetting. */
export class TextSetting {
  start: number;
  end: number;
  font: string;
  slant: string;
  weight: string;
  color: string;
  line_num: number;

  constructor(
    start: number,
    end: number,
    options: { font?: string; slant?: string; weight?: string; color?: string; line_num?: number } = {},
  ) {
    this.start = start;
    this.end = end;
    this.font = options.font ?? "";
    this.slant = options.slant ?? "NORMAL";
    this.weight = options.weight ?? "NORMAL";
    this.color = options.color ?? "WHITE";
    this.line_num = options.line_num ?? -1;
  }

  clone(): TextSetting {
    return new TextSetting(this.start, this.end, {
      font: this.font,
      slant: this.slant,
      weight: this.weight,
      color: this.color,
      line_num: this.line_num,
    });
  }
}

/** Remove unwanted invisible characters (zero-radius Dots) from a mobject. */
export function remove_invisible_chars(mobject: VGroup | VMobject): VGroup {
  const mobject_without_dots = new VGroup();
  const first = (mobject as unknown as { __getitem__(i: number): VMobject }).__getitem__(0);
  if (first instanceof VGroup) {
    for (const submob of (mobject as VGroup).submobjects as VMobject[]) {
      mobject_without_dots.add(new VGroup(...(submob.submobjects.filter((k) => !(k instanceof Dot)) as VMobject[])));
    }
  } else {
    mobject_without_dots.add(...(mobject.submobjects as VMobject[]).filter((k) => !(k instanceof Dot)));
  }
  return mobject_without_dots;
}

export interface TextOptions extends SVGMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  color?: ParsableManimColor | null;
  font_size?: number;
  line_spacing?: number;
  font?: string;
  slant?: string;
  weight?: string;
  t2c?: Record<string, string>;
  t2f?: Record<string, string>;
  t2g?: Record<string, Iterable<ParsableManimColor>>;
  t2s?: Record<string, string>;
  t2w?: Record<string, string>;
  gradient?: Iterable<ParsableManimColor> | null;
  tab_width?: number;
  warn_missing_font?: boolean;
  disable_ligatures?: boolean;
}

export class Text extends SVGMobject {
  line_spacing: number;
  font: string;
  _font_size: number;
  slant: string;
  weight: string;
  gradient: Iterable<ParsableManimColor> | null;
  tab_width: number;
  t2c: Record<string, string>;
  t2f: Record<string, string>;
  t2g: Record<string, Iterable<ParsableManimColor>>;
  t2s: Record<string, string>;
  t2w: Record<string, string>;
  original_text: string;
  disable_ligatures: boolean;
  text: string;
  chars!: VGroup;
  initial_height = 0;

  static font_list(): string[] {
    // Font discovery is not available in the PangoJS port; return empty.
    return [];
  }

  constructor(text: string, options: TextOptions = {}) {
    const {
      fill_opacity = 1.0,
      stroke_width = 0,
      color = null,
      font_size = DEFAULT_FONT_SIZE,
      line_spacing = -1,
      font = "",
      slant = NORMAL,
      weight = NORMAL,
      t2c: t2cOption = null,
      t2f: t2fOption = null,
      t2g: t2gOption = null,
      t2s: t2sOption = null,
      t2w: t2wOption = null,
      gradient = null,
      tab_width = 4,
      warn_missing_font = true,
      disable_ligatures = false,
      height = null,
      width = null,
      should_center = true,
      use_svg_cache = false,
      ...kwargs
    } = options as TextOptions;

    super({
      ...(kwargs as SVGMobjectOptions),
      svg_string: null,
      close_path_contours: true,
      fill_opacity,
      stroke_width,
      height: height ?? null,
      width: width ?? null,
      should_center,
      use_svg_cache,
    });

    let resolvedFont = font;
    if (resolvedFont && warn_missing_font) {
      const fontsList = Text.font_list();
      if (resolvedFont.toLowerCase() === "sans-serif") resolvedFont = "sans";
      if (!fontsList.includes(resolvedFont)) {
        if (fontsList.includes(resolvedFont.charAt(0).toUpperCase() + resolvedFont.slice(1))) {
          resolvedFont = resolvedFont.charAt(0).toUpperCase() + resolvedFont.slice(1);
        } else if (fontsList.includes(resolvedFont.toLowerCase())) {
          resolvedFont = resolvedFont.toLowerCase();
        } else {
          console.warn(`Font ${resolvedFont} not in ${JSON.stringify(fontsList)}.`);
        }
      }
    }
    this.line_spacing = line_spacing;
    this.font = resolvedFont;
    this._font_size = Number(font_size);
    this.slant = slant;
    this.weight = weight;
    this.gradient = gradient;
    this.tab_width = tab_width;

    let t2c = t2cOption ?? {};
    let t2f = t2fOption ?? {};
    let t2g = t2gOption ?? {};
    let t2s = t2sOption ?? {};
    let t2w = t2wOption ?? {};
    t2c = (kwargs as Record<string, never>)["text2color"] ?? t2c;
    t2f = (kwargs as Record<string, never>)["text2font"] ?? t2f;
    t2g = (kwargs as Record<string, never>)["text2gradient"] ?? t2g;
    t2s = (kwargs as Record<string, never>)["text2slant"] ?? t2s;
    t2w = (kwargs as Record<string, never>)["text2weight"] ?? t2w;
    this.t2c = Object.fromEntries(Object.entries(t2c).map(([key, value]) => [key, new ManimColor(value).to_hex()]));
    this.t2f = t2f;
    this.t2g = t2g;
    this.t2s = t2s;
    this.t2w = t2w;

    this.original_text = text;
    this.disable_ligatures = disable_ligatures;
    let text_without_tabs = text;
    if (text.includes("\t")) text_without_tabs = text.replace(/\t/g, " ".repeat(this.tab_width));
    this.text = text_without_tabs;
    if (this.line_spacing === -1) {
      this.line_spacing = this._font_size + this._font_size * DEFAULT_LINE_SPACING_SCALE;
    } else {
      this.line_spacing = this._font_size + this._font_size * this.line_spacing;
    }

    const parsed_color: ManimColor = color ? new ManimColor(color) : new ManimColor(WHITE);
    const svg_string = this._text2svg(parsed_color.to_hex());

    // Re-run the SVG-string construction with the rendered markup.
    (this as unknown as { _svg_string: string | null })._svg_string = svg_string;
    this.generate_mobject();
    this.move_into_position();

    this.text = text;
    if (this.disable_ligatures) {
      this.submobjects = [...this._gen_chars().submobjects];
    bump_mobject_geometry_epoch();
    }
    this.chars = new VGroup(...(this.submobjects as VMobject[]));
    this.text = text_without_tabs.replace(/ /g, "").replace(/\n/g, "");

    // Glyph contours are already closed by `path_to_mobject` (and cached per
    // `d`); only fall back to closing here for unusual subclasses.
    if (!this._close_path_contours) this._close_glyph_contours();

    if (height === null && width === null) {
      this.scale(TEXT_MOB_SCALE_FACTOR);
    }
    this.initial_height = this.height;
  }

  _close_glyph_contours(): void {
    closeGlyphContours(this);
  }

  __repr__(): string {
    return `Text(${JSON.stringify(this.original_text)})`;
  }

  override toString(): string {
    return this.__repr__();
  }

  get font_size(): number {
    return (
      (this.height / this.initial_height / TEXT_MOB_SCALE_FACTOR) *
      2.4 *
      (this._font_size / DEFAULT_FONT_SIZE)
    );
  }

  set font_size(font_val: number) {
    if (font_val <= 0) {
      const error = new Error("font_size must be greater than 0.");
      error.name = "ValueError";
      throw error;
    }
    this.scale(font_val / this.font_size);
  }

  _gen_chars(): VGroup {
    const chars = new VGroup();
    let submobjects_char_index = 0;
    for (let char_index = 0; char_index < this.text.length; char_index += 1) {
      if (/\s/.test(this.text[char_index] as string)) {
        const space = new Dot([0, 0, 0], 0, 0, 0, WHITE, { stroke_opacity: 0 });
        if (char_index === 0) {
          space.move_to((this.submobjects[submobjects_char_index] as VMobject).get_center());
        } else {
          space.move_to((this.submobjects[submobjects_char_index - 1] as VMobject).get_center());
        }
        chars.add(space);
      } else {
        chars.add(this.submobjects[submobjects_char_index] as VMobject);
        submobjects_char_index += 1;
      }
    }
    return chars;
  }

  _find_indexes(word: string, text: string): Array<[number, number]> {
    const temp = word.match(/^\[([0-9-]*):([0-9-]*)\]$/);
    if (temp) {
      let start = temp[1] !== "" ? parseInt(temp[1] as string, 10) : 0;
      let end = temp[2] !== "" ? parseInt(temp[2] as string, 10) : text.length;
      start = start < 0 ? text.length + start : start;
      end = end < 0 ? text.length + end : end;
      return [[start, end]];
    }
    const indexes: Array<[number, number]> = [];
    let index = text.indexOf(word);
    while (index !== -1) {
      indexes.push([index, index + word.length]);
      index = text.indexOf(word, index + word.length);
    }
    return indexes;
  }

  _merge_settings(
    left_setting: TextSetting,
    right_setting: TextSetting,
    default_args: Record<string, unknown>,
  ): TextSetting {
    const contained = right_setting.end < left_setting.end;
    const new_setting = contained ? left_setting.clone() : right_setting.clone();
    new_setting.start = contained ? right_setting.end : left_setting.end;
    left_setting.end = right_setting.start;
    if (!contained) right_setting.end = new_setting.start;

    for (const arg of Object.keys(default_args)) {
      const left = (left_setting as unknown as Record<string, unknown>)[arg];
      const right = (right_setting as unknown as Record<string, unknown>)[arg];
      const defaultValue = default_args[arg];
      if (left !== defaultValue && right !== defaultValue) {
        const error = new Error(
          `Ambiguous style for text '${this.text.slice(right_setting.start, right_setting.end)}': '${arg}' cannot be both '${String(left)}' and '${String(right)}'.`,
        );
        error.name = "ValueError";
        throw error;
      }
      (right_setting as unknown as Record<string, unknown>)[arg] = left !== defaultValue ? left : right;
    }
    return new_setting;
  }

  _get_settings_from_t2xs(
    t2xs: Array<[Record<string, unknown>, string]>,
    default_args: Record<string, unknown>,
  ): TextSetting[] {
    const settings: TextSetting[] = [];
    const t2xwords = new Set<string>();
    for (const [t2x] of t2xs) for (const key of Object.keys(t2x)) t2xwords.add(key);
    for (const word of t2xwords) {
      const settingArgs: Record<string, string> = {};
      for (const [t2x, arg] of t2xs) {
        settingArgs[arg] = word in t2x ? String(t2x[word]) : String(default_args[arg]);
      }
      for (const [start, end] of this._find_indexes(word, this.text)) {
        settings.push(new TextSetting(start, end, settingArgs));
      }
    }
    return settings;
  }

  _get_settings_from_gradient(default_args: Record<string, unknown>): TextSetting[] {
    const settings: TextSetting[] = [];
    const args: Record<string, unknown> = { ...default_args };
    if (this.gradient) {
      const colors = color_gradient([...this.gradient], this.text.length);
      for (let i = 0; i < this.text.length; i += 1) {
        args["color"] = colors[i]?.to_hex();
        settings.push(new TextSetting(i, i + 1, args as { color: string }));
      }
    }
    for (const [word, gradient] of Object.entries(this.t2g)) {
      const colors = color_gradient([...gradient], word.length);
      for (const [start, end] of this._find_indexes(word, this.text)) {
        for (let i = start; i < end; i += 1) {
          args["color"] = colors[i - start]?.to_hex();
          settings.push(new TextSetting(i, i + 1, args as { color: string }));
        }
      }
    }
    return settings;
  }

  _text2settings(color: ParsableManimColor): TextSetting[] {
    const t2xs: Array<[Record<string, unknown>, string]> = [
      [this.t2f as Record<string, unknown>, "font"],
      [this.t2s as Record<string, unknown>, "slant"],
      [this.t2w as Record<string, unknown>, "weight"],
      [this.t2c as Record<string, unknown>, "color"],
    ];
    const default_args: Record<string, unknown> = {};
    for (const [, arg] of t2xs) {
      default_args[arg] = arg === "color" ? color : (this as unknown as Record<string, unknown>)[arg];
    }

    let settings = this._get_settings_from_t2xs(t2xs, default_args);
    settings.push(...this._get_settings_from_gradient(default_args));

    settings.sort((a, b) => a.start - b.start);
    for (let index = 0; index < settings.length; index += 1) {
      if (index + 1 === settings.length) break;
      const setting = settings[index] as TextSetting;
      const next_setting = settings[index + 1] as TextSetting;
      if (setting.end > next_setting.start) {
        const new_setting = this._merge_settings(setting, next_setting, default_args);
        let new_index = index + 1;
        while (new_index < settings.length && (settings[new_index] as TextSetting).start < new_setting.start) {
          new_index += 1;
        }
        settings.splice(new_index, 0, new_setting);
      }
    }

    const temp_settings = [...settings];
    let start = 0;
    for (const setting of settings) {
      if (setting.start !== start) {
        temp_settings.push(new TextSetting(start, setting.start, default_args as Record<string, string>));
      }
      start = setting.end;
    }
    if (start !== this.text.length) {
      temp_settings.push(new TextSetting(start, this.text.length, default_args as Record<string, string>));
    }
    settings = temp_settings.sort((a, b) => a.start - b.start);

    let line_num = 0;
    if (/\n/.test(this.text)) {
      for (const [for_start, for_end] of this._find_indexes("\n", this.text)) {
        for (const setting of settings) {
          if (setting.line_num === -1) setting.line_num = line_num;
          if (for_start < setting.end) {
            line_num += 1;
            const new_setting = setting.clone();
            setting.end = for_end;
            new_setting.start = for_end;
            new_setting.line_num = line_num;
            settings.push(new_setting);
            settings.sort((a, b) => a.start - b.start);
            break;
          }
        }
      }
    }
    for (const setting of settings) {
      if (setting.line_num === -1) setting.line_num = line_num;
    }
    return settings;
  }

  _text_to_pango_markup(color: string): string {
    const settings = this._text2settings(color);
    const result: string[] = [];
    for (const setting of settings) {
      const chunk = this.text.slice(setting.start, setting.end);
      if (chunk === "\n") {
        result.push("\n");
        continue;
      }
      const attrs: string[] = [];
      if (setting.color) attrs.push(`foreground="${setting.color}"`);
      attrs.push(...get_pango_attrs_markup(setting.font, setting.weight, setting.slant));
      result.push(`<span ${attrs.join(" ")}>${escapeHtml(chunk)}</span>`);
    }
    return result.join("");
  }

  _text2svg(color: ParsableManimColor): string {
    const markup = this._text_to_pango_markup(String(color));
    return renderMarkupToSVGString(markup, {
      fontSize: (this._font_size / TEXT2SVG_ADJUSTMENT_FACTOR) * PANGO_POINT_TO_PX_FACTOR,
      lineSpacing: this.line_spacing / TEXT2SVG_ADJUSTMENT_FACTOR,
      color: String(color),
    });
  }

  override init_colors(propagate_colors = true): this {
    super.init_colors(propagate_colors);
    return this;
  }
}

export interface MarkupTextOptions extends SVGMobjectOptions {
  fill_opacity?: number;
  stroke_width?: number;
  color?: ParsableManimColor | null;
  font_size?: number;
  line_spacing?: number;
  font?: string;
  slant?: string;
  weight?: string;
  justify?: boolean;
  gradient?: Iterable<ParsableManimColor> | null;
  tab_width?: number;
  warn_missing_font?: boolean;
  disable_ligatures?: boolean;
}

export class MarkupText extends SVGMobject {
  text: string;
  line_spacing: number;
  font: string;
  _font_size: number;
  slant: string;
  weight: string;
  gradient: Iterable<ParsableManimColor> | null;
  tab_width: number;
  justify: boolean;
  original_text: string;
  disable_ligatures: boolean;
  chars!: VGroup;
  initial_height = 0;

  static font_list(): string[] {
    return [];
  }

  constructor(text: string, options: MarkupTextOptions = {}) {
    const {
      fill_opacity = 1,
      stroke_width = 0,
      color = null,
      font_size = DEFAULT_FONT_SIZE,
      line_spacing = -1,
      font = "",
      slant = NORMAL,
      weight = NORMAL,
      justify = false,
      gradient = null,
      tab_width = 4,
      height = null,
      width = null,
      should_center = true,
      disable_ligatures = false,
      warn_missing_font = true,
      ...kwargs
    } = options as MarkupTextOptions;

    super({
      ...(kwargs as SVGMobjectOptions),
      svg_string: null,
      close_path_contours: true,
      fill_opacity,
      stroke_width,
      height: height ?? null,
      width: width ?? null,
      should_center,
    });

    this.text = text;
    this.line_spacing = line_spacing;
    let resolvedFont = font;
    if (resolvedFont && warn_missing_font) {
      const fontsList = Text.font_list();
      if (resolvedFont.toLowerCase() === "sans-serif") resolvedFont = "sans";
      if (!fontsList.includes(resolvedFont)) {
        if (fontsList.includes(resolvedFont.charAt(0).toUpperCase() + resolvedFont.slice(1))) {
          resolvedFont = resolvedFont.charAt(0).toUpperCase() + resolvedFont.slice(1);
        } else if (fontsList.includes(resolvedFont.toLowerCase())) {
          resolvedFont = resolvedFont.toLowerCase();
        } else {
          console.warn(`Font ${resolvedFont} not in ${JSON.stringify(fontsList)}.`);
        }
      }
    }
    this.font = resolvedFont;
    this._font_size = Number(font_size);
    this.slant = slant;
    this.weight = weight;
    this.gradient = gradient;
    this.tab_width = tab_width;
    this.justify = justify;

    this.original_text = text;
    this.disable_ligatures = disable_ligatures;
    let text_without_tabs = text;
    if (text.includes("\t")) text_without_tabs = text.replace(/\t/g, " ".repeat(this.tab_width));

    const colormap = this._extract_color_tags();
    if (colormap.length > 0) {
      console.warn('Using <color> tags in MarkupText is deprecated. Please use <span foreground="..."> instead.');
    }
    const gradientmap = this._extract_gradient_tags();

    if (this.line_spacing === -1) {
      this.line_spacing = this._font_size + this._font_size * DEFAULT_LINE_SPACING_SCALE;
    } else {
      this.line_spacing = this._font_size + this._font_size * this.line_spacing;
    }

    const parsed_color: ManimColor = color ? new ManimColor(color) : new ManimColor(WHITE);
    const svg_string = this._text2svg(parsed_color);

    (this as unknown as { _svg_string: string | null })._svg_string = svg_string;
    this.generate_mobject();
    this.move_into_position();

    this.chars = new VGroup(...(this.submobjects as VMobject[]));
    this.text = text_without_tabs.replace(/ /g, "").replace(/\n/g, "");

    if (!this._close_path_contours) this._close_glyph_contours();

    if (this.gradient) this.set_color_by_gradient(...this.gradient);
    for (const col of colormap) {
      const group = this.chars.__getitem__({
        start: col.start - col.start_offset,
        stop: col.end - col.start_offset - col.end_offset,
      }) as VMobject;
      group.set_color(this._parse_color(col.color));
    }
    for (const grad of gradientmap) {
      const group = this.chars.__getitem__({
        start: grad.start - grad.start_offset,
        stop: grad.end - grad.start_offset - grad.end_offset,
      }) as VMobject;
      group.set_color_by_gradient(this._parse_color(grad.from), this._parse_color(grad.to));
    }

    if (height === null && width === null) this.scale(TEXT_MOB_SCALE_FACTOR);
    this.initial_height = this.height;
  }

  _close_glyph_contours(): void {
    closeGlyphContours(this);
  }

  get font_size(): number {
    return (this.height / this.initial_height / TEXT_MOB_SCALE_FACTOR) * 2.4 * (this._font_size / DEFAULT_FONT_SIZE);
  }

  set font_size(font_val: number) {
    if (font_val <= 0) {
      const error = new Error("font_size must be greater than 0.");
      error.name = "ValueError";
      throw error;
    }
    this.scale(font_val / this.font_size);
  }

  _text2svg(color: ParsableManimColor | null): string {
    const manimColor = new ManimColor(color ?? undefined);
    const colorHex = manimColor.to_hex();
    let markup = `<span foreground="${colorHex}">${this.text}</span>`;
    const outerAttrs = get_pango_attrs_markup(this.font, this.weight, this.slant);
    if (outerAttrs.length > 0) markup = `<span ${outerAttrs.join(" ")}>${markup}</span>`;
    return renderMarkupToSVGString(markup, {
      fontSize: (this._font_size / TEXT2SVG_ADJUSTMENT_FACTOR) * PANGO_POINT_TO_PX_FACTOR,
      lineSpacing: this.line_spacing / TEXT2SVG_ADJUSTMENT_FACTOR,
      color: colorHex,
    });
  }

  _count_real_chars(s: string): number {
    let count = 0;
    let level = 0;
    const replaced = s.replace(/&[^;]+;/g, "x");
    for (const c of replaced) {
      if (c === "<") level += 1;
      if (c === ">" && level > 0) level -= 1;
      else if (c !== " " && c !== "\t" && level === 0) count += 1;
    }
    return count;
  }

  _extract_gradient_tags(): Array<{
    start: number;
    end: number;
    from: string;
    to: string;
    start_offset: number;
    end_offset: number;
  }> {
    const regex = /<gradient\s+from="([^"]+)"\s+to="([^"]+)"(\s+offset="([^"]+)")?>([\s\S]+?)<\/gradient>/g;
    const gradientmap: Array<{ start: number; end: number; from: string; to: string; start_offset: number; end_offset: number }> = [];
    let tag: RegExpExecArray | null;
    while ((tag = regex.exec(this.original_text)) !== null) {
      const start = this._count_real_chars(this.original_text.slice(0, tag.index));
      const end = start + this._count_real_chars(tag[5] as string);
      const offsets = tag[4] ? (tag[4] as string).split(",") : ["0"];
      const start_offset = offsets[0] ? parseInt(offsets[0] as string, 10) : 0;
      const end_offset = offsets.length === 2 && offsets[1] ? parseInt(offsets[1] as string, 10) : 0;
      gradientmap.push({ start, end, from: tag[1] as string, to: tag[2] as string, start_offset, end_offset });
    }
    this.text = this.text.replace(/<gradient[^>]+>([\s\S]+?)<\/gradient>/g, "$1");
    return gradientmap;
  }

  _parse_color(col: string): string {
    if (/^#[0-9a-f]{6}$/.test(col)) return col;
    return new ManimColor(col).to_hex();
  }

  _extract_color_tags(): Array<{
    start: number;
    end: number;
    color: string;
    start_offset: number;
    end_offset: number;
  }> {
    const regex = /<color\s+col="([^"]+)"(\s+offset="([^"]+)")?>([\s\S]+?)<\/color>/g;
    const colormap: Array<{ start: number; end: number; color: string; start_offset: number; end_offset: number }> = [];
    let tag: RegExpExecArray | null;
    while ((tag = regex.exec(this.original_text)) !== null) {
      const start = this._count_real_chars(this.original_text.slice(0, tag.index));
      const end = start + this._count_real_chars(tag[4] as string);
      const offsets = tag[3] ? (tag[3] as string).split(",") : ["0"];
      const start_offset = offsets[0] ? parseInt(offsets[0] as string, 10) : 0;
      const end_offset = offsets.length === 2 && offsets[1] ? parseInt(offsets[1] as string, 10) : 0;
      colormap.push({ start, end, color: tag[1] as string, start_offset, end_offset });
    }
    this.text = this.text.replace(/<color[^>]+>([\s\S]+?)<\/color>/g, "$1");
    return colormap;
  }

  __repr__(): string {
    return `MarkupText(${JSON.stringify(this.original_text)})`;
  }

  override toString(): string {
    return this.__repr__();
  }
}

export interface ParagraphOptions extends TextOptions {
  line_spacing?: number;
  alignment?: string | null;
}

export class Paragraph extends VGroup {
  line_spacing: number;
  alignment: string | null;
  consider_spaces_as_chars: boolean;
  lines_text!: Text;
  chars!: VGroup;
  lines!: [VMobject[], Array<string | null>];
  lines_initial_positions: NDArray[] = [];

  constructor(...args: Array<string | ParagraphOptions>) {
    // Last argument may be the options object (mirrors Python's **kwargs).
    let textParts: string[] = [];
    let options: ParagraphOptions = {};
    if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1] !== null) {
      options = args[args.length - 1] as ParagraphOptions;
      textParts = args.slice(0, -1) as string[];
    } else {
      textParts = args as string[];
    }
    const { line_spacing = -1, alignment = null, ...textOptions } = options;
    super();
    this.line_spacing = line_spacing;
    this.alignment = alignment;
    this.consider_spaces_as_chars = Boolean(textOptions.disable_ligatures ?? false);

    const lines_str = textParts.join("\n");
    this.lines_text = new Text(lines_str, { ...textOptions, line_spacing });
    const lines_str_list = lines_str.split("\n");
    this.chars = this._gen_chars(lines_str_list);

    this.lines = [[...this.chars] as VMobject[], new Array(this.chars.submobjects.length).fill(this.alignment)];
    this.lines_initial_positions = this.lines[0].map((line) => line.get_center());
    this.add(...this.lines[0]);
    this.move_to([0, 0, 0]);
    if (this.alignment) this._set_all_lines_alignments(this.alignment);
  }

  _gen_chars(lines_str_list: string[]): VGroup {
    let char_index_counter = 0;
    const chars = new VGroup();
    for (let line_no = 0; line_no < lines_str_list.length; line_no += 1) {
      const line_str = lines_str_list[line_no] as string;
      let char_count: number;
      if (this.consider_spaces_as_chars) {
        char_count = line_str.length;
      } else {
        char_count = 0;
        for (const char of line_str) if (!/\s/.test(char)) char_count += 1;
      }
      chars.add(new VGroup());
      const lineGroup = chars.__getitem__(line_no) as VGroup;
      const slice = this.lines_text.chars.__getitem__({
        start: char_index_counter,
        stop: char_index_counter + char_count,
      }) as VGroup;
      lineGroup.add(...(slice.submobjects as VMobject[]));
      char_index_counter += char_count;
      if (this.consider_spaces_as_chars) char_index_counter += 1;
    }
    return chars;
  }

  _set_all_lines_alignments(alignment: string): this {
    for (let line_no = 0; line_no < this.lines[0].length; line_no += 1) {
      this._change_alignment_for_a_line(alignment, line_no);
    }
    return this;
  }

  _set_line_alignment(alignment: string, line_no: number): this {
    this._change_alignment_for_a_line(alignment, line_no);
    return this;
  }

  _set_all_lines_to_initial_positions(): this {
    this.lines[1] = new Array(this.lines[0].length).fill(null);
    for (let line_no = 0; line_no < this.lines[0].length; line_no += 1) {
      const line = this.__getitem__(line_no) as VMobject;
      const initial = pointRows(this.lines_initial_positions[line_no] ?? [], this.dim)[0] ?? [];
      const center = pointRows(this.get_center(), this.dim)[0] ?? [0, 0, 0];
      line.move_to(center.map((value, index) => value + (initial[index] ?? 0)));
    }
    return this;
  }

  _set_line_to_initial_position(line_no: number): this {
    this.lines[1][line_no] = null;
    const line = this.__getitem__(line_no) as VMobject;
    const initial = pointRows(this.lines_initial_positions[line_no] ?? [], this.dim)[0] ?? [];
    const center = pointRows(this.get_center(), this.dim)[0] ?? [0, 0, 0];
    line.move_to(center.map((value, index) => value + (initial[index] ?? 0)));
    return this;
  }

  _change_alignment_for_a_line(alignment: string, line_no: number): void {
    this.lines[1][line_no] = alignment;
    const line = this.__getitem__(line_no) as VMobject;
    const center = line.get_center();
    if (alignment === "center") {
      line.move_to([this.get_center()[0] as number, center[1] as number, 0]);
    } else if (alignment === "right") {
      line.move_to([(this.get_right()[0] as number) - line.width / 2, center[1] as number, 0]);
    } else if (alignment === "left") {
      line.move_to([(this.get_left()[0] as number) + line.width / 2, center[1] as number, 0]);
    }
  }
}

/** Temporarily add a font file to the font search path (unsupported: PangoJS
 * uses a single bundled font, so this always raises). */
export function register_font(_fontFile: string): void {
  throw new Error("TODO: register_font is not supported by the PangoJS port (single bundled font).");
}
