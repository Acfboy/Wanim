// Port of manim.mobject.text.tex_mobject (SingleStringMathTex, MathTex, Tex,
// Title, BulletedList). LaTeX is compiled to SVG by MathJax (see
// ./mathjax.ts) instead of a local LaTeX installation, mirroring the
// manim-wasm reference implementation. Substring isolation uses MathJax's
// `\cssId` output and SVGMobject's `split_by_id` support.

import { BLACK, WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { config } from "../../_config/utils.js";
import {
  DEFAULT_FONT_SIZE,
  LEFT,
  MED_LARGE_BUFF,
  MED_SMALL_BUFF,
  RIGHT,
  SCALE_FACTOR_PER_FONT_POINT,
  SMALL_BUFF,
  UP,
  DOWN,
} from "../../constants.js";
import { VMobject, VGroup } from "../types/vectorized_mobject.js";
import { SVGMobject, type SVGMobjectOptions } from "../svg/svg_mobject.js";
import { Line } from "../geometry/line.js";
import { setMathTexLabelFactory } from "../geometry/arc.js";
import { setTexFactory } from "../types/vectorized_mobject.js";
import { texToSvg, texToSvgCached } from "./mathjax.js";

export const MATHTEX_SUBSTRING = "substring";
export const TEX_MOB_SCALE_FACTOR = 0.008;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export interface SingleStringMathTexOptions extends SVGMobjectOptions {
  stroke_width?: number;
  should_center?: boolean;
  height?: number | null;
  organize_left_to_right?: boolean;
  tex_environment?: string | null;
  tex_template?: unknown;
  font_size?: number;
  color?: ParsableManimColor | null;
}

export class SingleStringMathTex extends SVGMobject {
  _font_size: number;
  organize_left_to_right: boolean;
  tex_environment: string | null;
  tex_template: unknown;
  tex_string: string;
  initial_height = 0;

  constructor(texString: string, options: SingleStringMathTexOptions = {}) {
    const {
      stroke_width = 0,
      should_center = true,
      height = null,
      organize_left_to_right = false,
      tex_environment = "align*",
      tex_template = null,
      font_size = DEFAULT_FONT_SIZE,
      color = null,
      ...kwargs
    } = options;

    super({
      ...(kwargs as SVGMobjectOptions),
      svg_string: null,
      should_center,
      stroke_width,
      height: height ?? null,
      color: color ?? WHITE,
      path_string_config: {
        should_subdivide_sharp_curves: true,
        should_remove_null_curves: true,
      },
      split_by_id: true,
    });

    this._font_size = font_size;
    this.organize_left_to_right = organize_left_to_right;
    this.tex_environment = tex_environment;
    this.tex_template = tex_template ?? null;
    this.tex_string = texString;

    const svg_string = texToSvgCached(this._get_modified_expression(texString));
    (this as unknown as { _svg_string: string | null })._svg_string = svg_string;
    // Use the SVG-mobject cache (as Python's SVGMobject.__init__ does): the
    // first construction of a given expression parses the MathJax SVG, later
    // ones clone the cached geometry.  Calling `generate_mobject()` directly
    // re-parsed the SVG on every construction.
    this.init_svg_mobject(true);
    this.move_into_position();

    this.init_colors();
    // NOTE: the fixed `TEX_MOB_SCALE_FACTOR` rescale and the `font_size`
    // rescale are both uniform scalings about the centred origin, so in the
    // common case they compose into a single traversal. `initial_height` is
    // the height *before* the font-size part, which the `font_size` getter
    // divides by. The zero-height / non-positive font-size edge cases keep the
    // original two-step path (and its ValueError).
    if (height === null && this._font_size > 0 && this.height > 0) {
      const fontScale = this._font_size * SCALE_FACTOR_PER_FONT_POINT;
      this.scale(TEX_MOB_SCALE_FACTOR * fontScale);
      this.initial_height = this.height / fontScale;
    } else {
      this.scale(TEX_MOB_SCALE_FACTOR);
      this.initial_height = this.height;
      if (height === null) this.font_size = this._font_size;
    }
    if (this.organize_left_to_right) this._organize_submobjects_left_to_right();
  }

  __repr__(): string {
    return `${this.constructor.name}(${JSON.stringify(this.tex_string)})`;
  }

  override toString(): string {
    return this.__repr__();
  }

  get font_size(): number {
    return this.height / this.initial_height / SCALE_FACTOR_PER_FONT_POINT;
  }

  set font_size(fontVal: number) {
    if (fontVal <= 0) {
      const error = new Error("font_size must be greater than 0.");
      error.name = "ValueError";
      throw error;
    } else if (this.height > 0) {
      this.scale(fontVal / this.font_size);
    }
  }

  _get_modified_expression(texString: string): string {
    let result = texString;
    result = result.trim();
    result = this._modify_special_strings(result);
    return result;
  }

  _modify_special_strings(tex: string): string {
    tex = tex.trim();
    const shouldAddFiller = [
      tex === "\\over",
      tex === "\\overline",
      tex === "\\sqrt",
      tex === "\\sqrt{",
      tex.endsWith("_"),
      tex.endsWith("^"),
      tex.endsWith("dot"),
    ].some(Boolean);

    if (shouldAddFiller) tex += "{\\quad}";
    if (tex === "\\substack") tex = "\\quad";
    if (tex === "") tex = "\\quad";
    if (tex.startsWith("\\\\")) tex = tex.replace(/\\\\/g, "\\quad\\\\");

    const countLefts = tex.split("\\left").length - 1;
    const countRights = tex.split("\\right").length - 1;
    void countLefts;
    void countRights;
    const numLefts = tex
      .split("\\left")
      .slice(1)
      .filter((s) => s && "(){}[]|.\\".includes(s[0] as string)).length;
    const numRights = tex
      .split("\\right")
      .slice(1)
      .filter((s) => s && "(){}[]|.\\".includes(s[0] as string)).length;
    if (numLefts !== numRights) {
      tex = tex.replace(/\\left/g, "\\big");
      tex = tex.replace(/\\right/g, "\\big");
    }

    tex = this._remove_stray_braces(tex);

    for (const context of ["array"]) {
      const beginIn = tex.includes(`\\begin{${context}}`);
      const endIn = tex.includes(`\\end{${context}}`);
      if (beginIn !== endIn) tex = "";
    }
    return tex;
  }

  _remove_stray_braces(tex: string): string {
    let numLefts = (tex.match(/\{/g)?.length ?? 0) - (tex.match(/\\\{/g)?.length ?? 0) + (tex.match(/\\\\\{/g)?.length ?? 0);
    let numRights = (tex.match(/\}/g)?.length ?? 0) - (tex.match(/\\\}/g)?.length ?? 0) + (tex.match(/\\\\\}/g)?.length ?? 0);
    while (numRights > numLefts) {
      tex = "{" + tex;
      numLefts += 1;
    }
    while (numLefts > numRights) {
      tex = tex + "}";
      numRights += 1;
    }
    return tex;
  }

  _organize_submobjects_left_to_right(): this {
    this.sort((p) => p[0] ?? 0);
    return this;
  }

  get_tex_string(): string {
    return this.tex_string;
  }

  override init_colors(_propagate_colors = true): this {
    const recolor = (mob: VMobject): void => {
      if (mob.submobjects.length > 0) {
        for (const child of mob.submobjects) recolor(child as VMobject);
        return;
      }
      if (mob.get_color().to_hex() !== BLACK.to_hex()) return;
      mob.set_color(this.color);
      mob.init_colors();
    };
    for (const submobject of this.submobjects) recolor(submobject as VMobject);
    return this;
  }
}

export interface MathTexOptions extends SingleStringMathTexOptions {
  arg_separator?: string;
  substrings_to_isolate?: Iterable<string> | null;
  tex_to_color_map?: Record<string, ParsableManimColor> | null;
  tex_environment?: string | null;
}

export class MathTexPart extends VMobject {
  tex_string = "";

  __repr__(): string {
    return `${this.constructor.name}(${JSON.stringify(this.tex_string)})`;
  }

  override toString(): string {
    return this.__repr__();
  }
}

/** Construction cache for repeated `MathTex` expressions (live labels are
 * rebuilt every frame from an unchanging small string set).  Keyed by the
 * expression and simple scalar options; object options disable caching. */
const MATH_TEX_CACHE = new Map<string, MathTex>();
const MATH_TEX_CACHE_LIMIT = 256;

function mathTexCacheKey(texStrings: string[], options: MathTexOptions): string | null {
  const parts: unknown[] = ["MathTex"];
  const simpleKeys = [
    "arg_separator", "tex_environment", "font_size", "color", "stroke_width",
    "should_center", "height", "width", "organize_left_to_right",
    "disable_ligatures", "warn_missing_font",
  ] as const;
  for (const key of simpleKeys) {
    const value = (options as Record<string, unknown>)[key];
    if (value === undefined || value === null) continue;
    if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
      parts.push(key, value);
    } else {
      return null;
    }
  }
  const substrings = (options as Record<string, unknown>).substrings_to_isolate;
  if (substrings !== undefined && substrings !== null) return null;
  const colorMap = (options as Record<string, unknown>).tex_to_color_map;
  if (colorMap !== undefined && colorMap !== null) {
    if (typeof colorMap !== "object") return null;
    try {
      parts.push("tex_to_color_map", JSON.stringify(colorMap));
    } catch {
      return null;
    }
  }
  parts.push(...texStrings);
  return JSON.stringify(parts);
}

export class MathTex extends SingleStringMathTex {
  arg_separator!: string;
  substrings_to_isolate!: string[];
  tex_to_color_map!: Record<string, ParsableManimColor>;
  brace_notation_split_occurred = false;
  tex_strings!: string[];
  matched_strings_and_ids: Array<[string, string]> = [];

  constructor(...args: Array<string | MathTexOptions>) {
    let texStrings: string[] = [];
    let options: MathTexOptions = {};
    if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1] !== null) {
      options = args[args.length - 1] as MathTexOptions;
      texStrings = args.slice(0, -1) as string[];
    } else {
      texStrings = args as string[];
    }
    const cacheKey = (new.target as unknown) === MathTex ? mathTexCacheKey(texStrings, options) : null;
    if (cacheKey !== null) {
      const cached = MATH_TEX_CACHE.get(cacheKey);
      if (cached !== undefined) {
        return cached.copy() as MathTex;
      }
    }
    const {
      arg_separator = " ",
      substrings_to_isolate = null,
      tex_to_color_map = null,
      tex_environment = "align*",
      ...singleOptions
    } = options;

    const substrings = substrings_to_isolate === null ? [] : [...substrings_to_isolate];
    const colorMap: Record<string, ParsableManimColor> = tex_to_color_map ?? {};
    substrings.push(...Object.keys(colorMap));
    const braceSplit = { occurred: false };
    const prepared = MathTex._prepare_tex_strings(texStrings, braceSplit);
    let joined = MathTex._join_tex_strings_with_unique_delimiters(
      prepared,
      substrings,
      arg_separator,
    );

    // NOTE: substring isolation via `\cssId` is *not* transparent to the TeX
    // parser (unlike Python's dvisvgm `\special` raw groups). Isolating a
    // fragment such as `m_` (from `tex_to_color_map={"m_": BLUE}`) splits
    // `m_1` into `\cssId{...}{m_}1`, leaving a dangling `_` that makes
    // MathJax emit an `merror` ("Missing superscript or subscript argument")
    // for the *whole* expression. The SVG then carries no `uniqueNNN` groups,
    // `_break_up_by_substrings` falls back to a single root, and indexing
    // (e.g. `equation[1]`) throws `IndexError: range object index out of
    // range`. Preflight the isolated expression: if MathJax errors or any
    // outer id is missing, rebuild the geometry without substring isolation
    // so the per-argument submobjects survive; fragment colours fall back to
    // their containing parts (see `set_color_by_tex_to_color_map`).
    if (substrings.length > 0 && prepared.length > 0) {
      try {
        const probeSvg = texToSvgCached(joined.joined_string.trim());
        const hasMerror =
          probeSvg.includes("merror") ||
          probeSvg.includes("Missing superscript") ||
          probeSvg.includes("MissingSubSup");
        let hasAllOuterIds = true;
        for (let idx = 0; idx < prepared.length; idx += 1) {
          if (!probeSvg.includes(`id="unique${String(idx).padStart(3, "0")}"`)) {
            hasAllOuterIds = false;
            break;
          }
        }
        if (hasMerror || !hasAllOuterIds) {
          joined = MathTex._join_tex_strings_with_unique_delimiters(
            prepared,
            [],
            arg_separator,
          );
        }
      } catch {
        joined = MathTex._join_tex_strings_with_unique_delimiters(
          prepared,
          [],
          arg_separator,
        );
      }
    }

    super(joined.joined_string, {
      ...singleOptions,
      tex_environment,
      tex_template: (singleOptions as { tex_template?: unknown }).tex_template ?? null,
    });

    this.arg_separator = arg_separator;
    this.substrings_to_isolate = substrings;
    this.tex_to_color_map = colorMap;
    this.brace_notation_split_occurred = braceSplit.occurred;
    this.tex_strings = prepared;
    this.matched_strings_and_ids = joined.matches;
    this.tex_string = prepared.join(arg_separator);
    this._break_up_by_substrings();
    this.set_color_by_tex_to_color_map(this.tex_to_color_map);
    if (this.organize_left_to_right) this._organize_submobjects_left_to_right();
    if (cacheKey !== null && MATH_TEX_CACHE.size < MATH_TEX_CACHE_LIMIT) {
      MATH_TEX_CACHE.set(cacheKey, this.copy() as MathTex);
    }
  }

  static _prepare_tex_strings(texStrings: Iterable<string>, braceSplit: { occurred: boolean }): string[] {
    const validated = [...texStrings].map((value) => (typeof value === "string" ? value : String(value)));
    const validatedTwo: string[] = [];
    for (const texString of validated) {
      validatedTwo.push(...MathTex._split_double_braces(texString));
    }
    if (validatedTwo.length > validated.length) braceSplit.occurred = true;
    return validatedTwo.filter((value) => value.length > 0);
  }

  static _split_double_braces(texString: string): string[] {
    const segments: string[] = [];
    let current = "";
    let i = 0;
    let insideManim = false;
    let innerDepth = 0;

    while (i < texString.length) {
      if (texString[i] === "\\" && i + 1 < texString.length) {
        const nextCh = texString[i + 1] as string;
        if (nextCh === "\\" || nextCh === "{" || nextCh === "}") {
          current += texString.slice(i, i + 2);
          i += 2;
          continue;
        }
      }
      if (!insideManim) {
        if (texString.slice(i, i + 2) === "{{" && (i === 0 || /\s/.test(texString[i - 1] as string))) {
          segments.push(current);
          current = "";
          insideManim = true;
          innerDepth = 0;
          i += 2;
        } else {
          current += texString[i];
          i += 1;
        }
      } else {
        if (texString[i] === "{") {
          innerDepth += 1;
          current += texString[i];
          i += 1;
        } else if (texString[i] === "}" && innerDepth === 0 && texString.slice(i, i + 2) === "}}") {
          segments.push(current);
          current = "";
          insideManim = false;
          i += 2;
        } else if (texString[i] === "}") {
          innerDepth -= 1;
          current += texString[i];
          i += 1;
        } else {
          current += texString[i];
          i += 1;
        }
      }
    }
    segments.push(current);
    return segments;
  }

  static _join_tex_strings_with_unique_delimiters(
    texStrings: string[],
    substringsToIsolate: Iterable<string>,
    argSeparator: string,
  ): { joined_string: string; matches: Array<[string, string]> } {
    const matches: Array<[string, string]> = [];
    let joinedString = "";
    let ssIdx = 0;
    for (let idx = 0; idx < texStrings.length; idx += 1) {
      const texString = texStrings[idx] as string;
      const idStr = `unique${String(idx).padStart(3, "0")}`;
      matches.push([texString, idStr]);

      let unprocessedString = String(texString);
      let processedString = "";
      while (unprocessedString.length > 0) {
        const firstMatch = MathTex._locate_first_match(substringsToIsolate, unprocessedString);
        if (firstMatch) {
          const [processed, rest] = MathTex._handle_match(ssIdx, firstMatch, matches);
          processedString = processedString + processed;
          unprocessedString = rest;
          ssIdx += 1;
        } else {
          processedString = processedString + unprocessedString;
          unprocessedString = "";
        }
      }

      let stringPart = `\\cssId{${idStr}}{${processedString}}`;
      if (idx < texStrings.length - 1) stringPart += argSeparator;
      joinedString = joinedString + stringPart;
    }
    return { joined_string: joinedString, matches };
  }

  static _locate_first_match(
    substringsToIsolate: Iterable<string>,
    unprocessedString: string,
  ): RegExpExecArray | null {
    let firstMatchStart = unprocessedString.length;
    let firstMatchLength = 0;
    let firstMatch: RegExpExecArray | null = null;
    for (const substring of substringsToIsolate) {
      const regex = new RegExp(`^(.*?)(${escapeRegExp(substring)})(.*)`);
      const match = regex.exec(unprocessedString);
      if (match && (match[1] as string).length < firstMatchStart) {
        firstMatch = match;
        firstMatchStart = (match[1] as string).length;
        firstMatchLength = (match[2] as string).length;
      } else if (match && (match[1] as string).length === firstMatchStart) {
        if (firstMatchLength < (match[2] as string).length) {
          firstMatch = match;
          firstMatchStart = (match[1] as string).length;
          firstMatchLength = (match[2] as string).length;
        }
      }
    }
    return firstMatch;
  }

  static _handle_match(
    ssIdx: number,
    firstMatch: RegExpExecArray,
    matches: Array<[string, string]>,
  ): [string, string] {
    const preMatch = firstMatch[1] as string;
    const matchedString = firstMatch[2] as string;
    const postMatch = firstMatch[3] as string;
    const idStr = `unique${String(ssIdx).padStart(3, "0")}${MATHTEX_SUBSTRING}`;
    matches.push([matchedString, idStr]);
    const processedString = `${preMatch}\\cssId{${idStr}}{${matchedString}}`;
    return [processedString, postMatch];
  }

  get _substring_matches(): Array<[string, string]> {
    return this.matched_strings_and_ids.filter(([, id]) => id.endsWith(MATHTEX_SUBSTRING));
  }

  get _main_matches(): Array<[string, string]> {
    return this.matched_strings_and_ids.filter(([, id]) => !id.endsWith(MATHTEX_SUBSTRING));
  }

  _break_up_by_substrings(): this {
    const newSubmobjects: VMobject[] = [];
    try {
      for (const [texString, texStringId] of this._main_matches) {
        const mtp = new MathTexPart();
        mtp.tex_string = texString;
        const vgroup = this.id_to_vgroup_dict.get(texStringId);
        if (!vgroup) throw new Error("KeyError");
        mtp.add(...(vgroup.submobjects as VMobject[]));
        newSubmobjects.push(mtp);
      }
    } catch {
      console.error(
        `MathTex: Could not find SVG group for a tex part. Using fallback to root group.`,
      );
      const root = this.id_to_vgroup_dict.get("root");
      if (root) newSubmobjects.push(root);
    }
    this.submobjects = newSubmobjects;
    return this;
  }

  get_part_by_tex(tex: string, _kwargs: Record<string, unknown> = {}): VGroup | null {
    for (const [texStr, matchId] of this.matched_strings_and_ids) {
      if (texStr === tex) return this.id_to_vgroup_dict.get(matchId) ?? null;
    }
    return null;
  }

  set_color_by_tex(tex: string, color: ParsableManimColor, _kwargs: Record<string, unknown> = {}): this {
    for (const [texStr, matchId] of this.matched_strings_and_ids) {
      if (texStr === tex) this.id_to_vgroup_dict.get(matchId)?.set_color(color);
    }
    return this;
  }

  set_opacity_by_tex(
    tex: string,
    opacity = 0.5,
    remainingOpacity: number | null = null,
    _kwargs: Record<string, unknown> = {},
  ): this {
    if (remainingOpacity !== null) this.set_opacity(remainingOpacity);
    for (const [texStr, matchId] of this.matched_strings_and_ids) {
      if (texStr === tex) this.id_to_vgroup_dict.get(matchId)?.set_opacity(opacity);
    }
    return this;
  }

  set_color_by_tex_to_color_map(texsToColorMap: Record<string, ParsableManimColor>, _kwargs: Record<string, unknown> = {}): this {
    for (const [texs, color] of Object.entries(texsToColorMap)) {
      let colouredExact = false;
      for (const [matchTex, matchId] of this.matched_strings_and_ids) {
        if (matchTex === texs) {
          const vg = this.id_to_vgroup_dict.get(matchId);
          if (vg !== undefined) {
            vg.set_color(color);
            colouredExact = true;
          }
        }
      }
      // Fallback for fragment keys whose substring isolation was skipped to
      // keep MathJax valid (e.g. `m_` inside `\sqrt{m_1}`): colour the
      // containing per-argument parts instead of colouring nothing.
      if (!colouredExact) {
        for (const [matchTex, matchId] of this._main_matches) {
          if (matchTex.includes(texs)) this.id_to_vgroup_dict.get(matchId)?.set_color(color);
        }
      }
    }
    return this;
  }

  index_of_part(part: MathTex): number {
    const splitSelf = this.split();
    if (!splitSelf.includes(part as unknown as VMobject)) {
      const error = new Error("Trying to get index of part not in MathTex");
      error.name = "ValueError";
      throw error;
    }
    return splitSelf.indexOf(part as unknown as VMobject);
  }

  sort_alphabetically(): void {
    this.submobjects.sort((a, b) =>
      String((a as unknown as { get_tex_string(): string }).get_tex_string()).localeCompare(
        String((b as unknown as { get_tex_string(): string }).get_tex_string()),
      ),
    );
  }
}

export interface TexOptions extends MathTexOptions {
  arg_separator?: string;
  tex_environment?: string | null;
  /** manimlib-style alias of `tex_to_color_map`, used by 3b1b scenes. */
  t2c?: Record<string, ParsableManimColor> | null;
}

export class Tex extends MathTex {
  _original_tex_strings: string[];

  constructor(...args: Array<string | TexOptions>) {
    let texStrings: string[] = [];
    let options: TexOptions = {};
    if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1] !== null) {
      options = args[args.length - 1] as TexOptions;
      texStrings = args.slice(0, -1) as string[];
    } else {
      texStrings = args as string[];
    }
    const { arg_separator = "", tex_environment = "center", ...rest } = options;
    const restRecord = rest as Record<string, unknown>;
    // 3b1b scenes (manimlib-style toolchain) pass `t2c` to `Tex`; community
    // manim rejects it, but wanim maps it so the scenes render with colours.
    if (restRecord["t2c"] !== undefined && restRecord["t2c"] !== null && restRecord["tex_to_color_map"] === undefined) {
      restRecord["tex_to_color_map"] = restRecord["t2c"];
    }
    delete restRecord["t2c"];
    const converted = texStrings.map((ts) => Tex._recover_tex_string(ts));
    super(...converted, { ...rest, arg_separator, tex_environment } as MathTexOptions);
    this._original_tex_strings = texStrings;
  }

  override get_part_by_tex(tex: string, kwargs: Record<string, unknown> = {}): VGroup | null {
    return super.get_part_by_tex(Tex._convert_tex_to_mathtex(tex), kwargs)
      ?? super.get_part_by_tex(tex, kwargs);
  }

  override set_color_by_tex(tex: string, color: ParsableManimColor, kwargs: Record<string, unknown> = {}): this {
    super.set_color_by_tex(Tex._convert_tex_to_mathtex(tex), color, kwargs);
    // The string may have been kept raw by the MathJax fallback in the
    // constructor; colour that form too (idempotent when identical).
    if (Tex._convert_tex_to_mathtex(tex) !== tex) {
      super.set_color_by_tex(tex, color, kwargs);
    }
    return this;
  }

  override set_opacity_by_tex(
    tex: string,
    opacity = 0.5,
    remainingOpacity: number | null = null,
    kwargs: Record<string, unknown> = {},
  ): this {
    super.set_opacity_by_tex(Tex._convert_tex_to_mathtex(tex), opacity, remainingOpacity, kwargs);
    if (Tex._convert_tex_to_mathtex(tex) !== tex) {
      super.set_opacity_by_tex(tex, opacity, remainingOpacity, kwargs);
    }
    return this;
  }

  override set_color_by_tex_to_color_map(
    texsToColorMap: Record<string, ParsableManimColor>,
    kwargs: Record<string, unknown> = {},
  ): this {
    const converted: Record<string, ParsableManimColor> = {};
    for (const [texs, color] of Object.entries(texsToColorMap)) {
      // Converted form (when the constructor kept the `\text{}` wrapping)
      // plus the raw form (when it fell back to math mode below).
      converted[texs] = color;
      converted[Tex._convert_tex_to_mathtex(texs)] = color;
    }
    return super.set_color_by_tex_to_color_map(converted, kwargs);
  }

  /**
   * Convert a `Tex` (text-mode) string for the MathJax backend, falling back
   * to the raw string when the `\text{}` wrapping breaks math content.
   *
   * Real LaTeX recovers from text-mode `^`/`_` (nonstop mode inserts `$`
   * and still emits the glyphs/groups), but MathJax emits `<merror>` and
   * drops the `\cssId` groups. The missing ids make
   * `_break_up_by_substrings` fall back to the root `VGroup`, whose parts
   * then trip `TransformMatchingTex`'s `isinstance(MathTexPart)` assert
   * (e.g. 3b1b's `Tex(R"e^{a{t}}", t2c=...)`). If the wrapped form errors
   * while the raw form renders, keep the raw form.
   */
  static _recover_tex_string(tex: string): string {
    const converted = Tex._convert_tex_to_mathtex(tex);
    if (converted === tex) return converted;
    try {
      if (Tex._is_mathjax_error(converted) && !Tex._is_mathjax_error(tex)) return tex;
    } catch {
      // Probe failures are non-fatal; keep the converted form.
    }
    return converted;
  }

  static _is_mathjax_error(expression: string): boolean {
    // Probe the expression the way SingleStringMathTex will actually render
    // it (after _modify_special_strings): unbalanced \left/\right become
    // \big, stray braces are balanced, etc. Probing the raw string
    // misclassifies fragments like `e^{st} \left(` or `\right) = 0`, where
    // both the converted and raw forms error pre-modify but the raw form
    // renders fine post-modify (real LaTeX likewise recovers via nonstop
    // mode / the \big substitution).
    let probe: string;
    try {
      const tmp = Object.create(
        SingleStringMathTex.prototype,
      ) as SingleStringMathTex;
      probe = tmp._get_modified_expression(expression);
    } catch {
      probe = expression;
    }
    if (probe.trim() === "") probe = "\\quad";
    const svg = texToSvgCached(probe);
    return svg.includes("merror") || svg.includes("data-mjx-error");
  }

  static _convert_tex_to_mathtex(tex: string): string {
    const result: string[] = [];
    let i = 0;
    while (i < tex.length) {
      if (tex.slice(i, i + 2) === "$$") {
        const end = tex.indexOf("$$", i + 2);
        if (end !== -1) {
          result.push(tex.slice(i + 2, end));
          i = end + 2;
          continue;
        }
      }
      if (tex[i] === "$") {
        const end = tex.indexOf("$", i + 1);
        if (end !== -1) {
          result.push(tex.slice(i + 1, end));
          i = end + 1;
          continue;
        }
      }
      if (tex[i] === "\\") {
        let j = i + 1;
        while (j < tex.length && /[a-zA-Z]/.test(tex[j] as string)) j += 1;
        result.push(tex.slice(i, j));
        if (j < tex.length && tex[j] === "{") {
          let braceCount = 1;
          let k = j + 1;
          while (k < tex.length && braceCount > 0) {
            if (tex[k] === "{") braceCount += 1;
            else if (tex[k] === "}") braceCount -= 1;
            k += 1;
          }
          result.push(tex.slice(j, k));
          i = k;
        } else {
          i = j;
        }
        if (i < tex.length && tex[i] === " ") {
          result.push(" ");
          i += 1;
        }
        continue;
      }
      const textStart = i;
      while (i < tex.length && tex[i] !== "$" && tex[i] !== "\\") i += 1;
      if (i > textStart) result.push(`\\text{${tex.slice(textStart, i)}}`);
    }
    return result.join("");
  }
}

export interface BulletedListOptions extends TexOptions {
  buff?: number;
  dot_scale_factor?: number;
  tex_environment?: string | null;
}

export class BulletedList extends Tex {
  buff: number;
  dot_scale_factor: number;

  constructor(...args: Array<string | BulletedListOptions>) {
    let items: string[] = [];
    let options: BulletedListOptions = {};
    if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1] !== null) {
      options = args[args.length - 1] as BulletedListOptions;
      items = args.slice(0, -1) as string[];
    } else {
      items = args as string[];
    }
    const { buff = MED_LARGE_BUFF, dot_scale_factor = 2, tex_environment = null, ...rest } = options;
    const lineSeparatedItems = items.map((s) => s + "\\\\");
    super(...lineSeparatedItems, { ...rest, tex_environment } as TexOptions);
    this.buff = buff;
    this.dot_scale_factor = dot_scale_factor;
    for (const part of this.submobjects as VMobject[]) {
      const dot = new MathTex("\\cdot").scale(this.dot_scale_factor);
      const firstPart = part.__getitem__(0) as VMobject;
      dot.next_to(firstPart, LEFT, SMALL_BUFF);
      part.add_to_back(dot);
    }
    for (let index = 1; index < this.submobjects.length; index += 1) {
      (this.submobjects[index] as VMobject).next_to(this.submobjects[index - 1] as VMobject, DOWN, this.buff, LEFT);
    }
  }

  fade_all_but(indexOrString: number | string, opacity = 0.5): void {
    let part: VMobject | null = null;
    if (typeof indexOrString === "string") {
      part = this.get_part_by_tex(indexOrString) as VMobject | null;
      if (part === null) throw new Error(`Could not locate part by provided tex string '${indexOrString}'.`);
    } else if (typeof indexOrString === "number") {
      part = this.submobjects[indexOrString] as VMobject;
    } else {
      throw new TypeError(`Expected int or string, got ${String(indexOrString)}`);
    }
    for (const otherPart of this.submobjects as VMobject[]) {
      if (otherPart === part) otherPart.set_fill(undefined, 1);
      else otherPart.set_fill(undefined, opacity);
    }
  }
}

export interface TitleOptions extends TexOptions {
  include_underline?: boolean;
  match_underline_width_to_text?: boolean;
  underline_buff?: number;
}

export class Title extends Tex {
  include_underline: boolean;
  match_underline_width_to_text: boolean;
  underline_buff: number;
  underline: Line | null = null;

  constructor(...args: Array<string | TitleOptions>) {
    let textParts: string[] = [];
    let options: TitleOptions = {};
    if (args.length > 0 && typeof args[args.length - 1] === "object" && args[args.length - 1] !== null) {
      options = args[args.length - 1] as TitleOptions;
      textParts = args.slice(0, -1) as string[];
    } else {
      textParts = args as string[];
    }
    const {
      include_underline = true,
      match_underline_width_to_text = false,
      underline_buff = MED_SMALL_BUFF,
      ...rest
    } = options;
    super(...textParts, rest as TexOptions);
    this.include_underline = include_underline;
    this.match_underline_width_to_text = match_underline_width_to_text;
    this.underline_buff = underline_buff;
    this.to_edge(UP);
    if (this.include_underline) {
      const underlineWidth = Number(config.frame_width) - 2;
      const underline = new Line([...LEFT], [...RIGHT]);
      underline.next_to(this, DOWN, this.underline_buff);
      if (this.match_underline_width_to_text) underline.match_width(this);
      else underline.width = underlineWidth;
      this.add(underline);
      this.underline = underline;
    }
  }
}

// Registers the raw-string label renderer used by LabeledDot (see arc.ts).
// Kept as a registry rather than an import in arc.ts to avoid the
// arc.ts -> tex_mobject.ts -> line.ts -> arc.ts module cycle.
setMathTexLabelFactory((label: string, color) => new MathTex(label, { color }));

// Registers the key-label renderer used by VDict(show_keys=True); the
// registry lives in vectorized_mobject.ts so it need not import this module
// (which would create a cycle).
setTexFactory((text, color) => new Tex(text, color === undefined || color === null ? {} : { color }));
