// Port of manim.mobject.text.numbers (DecimalNumber, Integer, Variable).
//
// NOTE (text/tex): digit/unit glyphs are rendered with MathTex/
// SingleStringMathTex in Python. The TS port renders the same MathTex classes
// through MathJax (no local LaTeX install needed), so the default `mob_class`
// factory and the SingleStringMathTex spots for the ellipsis and the unit all
// work in Node. All pure-numeric behavior (format specs, num-string building,
// get/set/increment_value, the ValueTracker wiring of Variable) is ported
// faithfully; a custom `mob_class` factory can still be passed.
import type { NDArray } from "numpy-ts";
import {
  DEFAULT_FONT_SIZE,
  LEFT,
  RendererType,
  RIGHT,
} from "../../constants.js";
import { config } from "../../_config/utils.js";
import { type ComplexPoint } from "../../utils/arrays.js";
import { Mobject } from "../mobject.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { ValueTracker } from "../value_tracker.js";
import { MathTex, SingleStringMathTex } from "./tex_mobject.js";

export type DecimalNumberValue = number | ComplexPoint;

/** Factory building one digit/unit glyph mobject (stands in for the
 * MathTex/SingleStringMathTex classes, which need LaTeX). */
export type DecimalNumberMobFactory = (
  text: string,
  kwargs?: Record<string, unknown>,
) => VMobject;

/** Glyph factories backed by the ported MathTex/SingleStringMathTex
 * (LaTeX is rendered through MathJax, so no local LaTeX is needed). */
const mathTexFactory: DecimalNumberMobFactory = (text, kwargs = {}) =>
  new MathTex(text, kwargs as never);
const singleStringMathTexFactory: DecimalNumberMobFactory = (text, kwargs = {}) =>
  new SingleStringMathTex(text, kwargs as never);

export const string_to_mob_map = new Map<string, VMobject>();

// Glyph prototypes are cached per (factory, font size, string). The factory
// must be part of the key: distinct `mob_class` factories (e.g. a plain
// VMobject stub vs. MathTex) produce different glyph geometry, and sharing one
// cache entry across them yields wrong-sized characters. A WeakMap gives each
// factory a stable id without keeping it alive.
const _factory_ids = new WeakMap<object, number>();
let _next_factory_id = 1;

function factory_id(factory: DecimalNumberMobFactory): number {
  let id = _factory_ids.get(factory);
  if (id === undefined) {
    id = _next_factory_id;
    _next_factory_id += 1;
    _factory_ids.set(factory, id);
  }
  return id;
}

/** Geometry of a cached glyph prototype.
 *
 * The `minX/maxX/minY/maxY` fields are the extents of the *anchors* (the set
 * `get_critical_point`/`get_center` read), while the `fullMin.../fullMax...`
 * fields cover every stored point (the set `width`/`height` read). The two
 * differ because a cubic bezier's control handles can stick out past its
 * anchors. */
interface GlyphInfo {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  fullMinX: number;
  fullMaxX: number;
  fullMinY: number;
  fullMaxY: number;
}

const glyph_info_map = new WeakMap<VMobject, GlyphInfo>();

/** Measure a glyph prototype once; the hot path then works from these numbers
 * instead of re-scanning the (many) SVG points through the ndarray helpers. */
function measure_glyph(mob: VMobject): GlyphInfo {
  const left = mob.get_critical_point([-1, 0, 0]);
  const right = mob.get_critical_point([1, 0, 0]);
  const down = mob.get_critical_point([0, -1, 0]);
  const up = mob.get_critical_point([0, 1, 0]);
  let fullMinX = Infinity;
  let fullMaxX = -Infinity;
  let fullMinY = Infinity;
  let fullMaxY = -Infinity;
  for (const member of mob.family_members_with_points()) {
    const rows = (member.points.toArray() as unknown) as number[][];
    for (const point of rows) {
      const x = point[0] ?? 0;
      const y = point[1] ?? 0;
      if (x < fullMinX) fullMinX = x;
      if (x > fullMaxX) fullMaxX = x;
      if (y < fullMinY) fullMinY = y;
      if (y > fullMaxY) fullMaxY = y;
    }
  }
  if (!Number.isFinite(fullMinX)) {
    fullMinX = 0;
    fullMaxX = 0;
    fullMinY = 0;
    fullMaxY = 0;
  }
  return {
    minX: left[0] ?? 0,
    maxX: right[0] ?? 0,
    minY: down[1] ?? 0,
    maxY: up[1] ?? 0,
    fullMinX,
    fullMaxX,
    fullMinY,
    fullMaxY,
  };
}

/** Build (or fetch) a glyph prototype already sized to `font_size`.
 *
 * The key covers the factory, the font size and the string: the font size so
 * the prototype is scaled once and every caller gets a correctly-sized copy
 * with no `font_size =` rescale, and the factory so that different `mob_class`
 * renderers (a MathTex glyph vs. a plain VMobject stub) never share entries.
 * `NumberLine.get_number_mobject` builds a wrapper closure per call; that
 * closure is memoized on its underlying constructor (see `toMobFactory`) so
 * its factory id stays stable and the cache still applies. The cached
 * prototype is never mutated afterwards; callers copy it and apply their own
 * colour/position. */
function glyph_proto(
  stringValue: string,
  factory: DecimalNumberMobFactory,
  font_size: number,
  kwargs: Record<string, unknown>,
): VMobject {
  const key = `${factory_id(factory)}|${font_size}|${stringValue}`;
  let proto = string_to_mob_map.get(key);
  if (proto === undefined) {
    proto = factory(stringValue, kwargs);
    // NOTE: read via `in` first: the Mobject proxy throws AttributeError for
    // unknown properties, and plain VMobject factories (e.g. the test
    // dummyFactory) have no `font_size` at all.
    if ("font_size" in (proto as object)) {
      const current = (proto as unknown as { font_size: number }).font_size;
      if (current !== font_size) {
        (proto as unknown as { font_size: number }).font_size = font_size;
      }
    }
    string_to_mob_map.set(key, proto);
    glyph_info_map.set(proto, measure_glyph(proto));
  }
  return proto;
}

interface ContiguousPointStore {
  data?: { [index: number]: number; length: number };
  shape?: readonly number[];
  flags?: { C_CONTIGUOUS?: boolean };
}

/** Translate one mobject's points in place when the backing store is a plain
 * contiguous typed array, avoiding the `pointRows`/`pointArray` round trip
 * (and its ndarray allocations) that `shift()` performs. Falls back to
 * `shift()` for views / non-contiguous / exotic storages. */
function translate_points_in_place(mob: Mobject, delta: readonly number[]): void {
  const store = (mob as unknown as { points?: ContiguousPointStore }).points;
  const data = store?.data;
  const shape = store?.shape;
  if (
    data !== undefined &&
    data !== null &&
    shape !== undefined &&
    shape.length === 2 &&
    (store?.flags === undefined || store.flags.C_CONTIGUOUS !== false) &&
    data.length === (shape[0] ?? 0) * (shape[1] ?? 0)
  ) {
    const columns = shape[1] ?? mob.dim;
    for (let index = 0; index < data.length; index += 1) {
      data[index] += delta[index % columns] ?? 0;
    }
    return;
  }
  mob.shift(delta);
}

function translate_mob_in_place(mob: Mobject, delta: readonly number[]): void {
  for (const member of mob.family_members_with_points()) {
    translate_points_in_place(member, delta);
  }
}

interface NumberPart {
  mob: VMobject;
  info: GlyphInfo;
  tx: number;
  ty: number;
}

/** Round-half-to-even (banker's), mirroring np.round for scalars. */
function roundHalfEven(value: number): number {
  if (!Number.isFinite(value)) return value;
  const sign = Math.sign(value);
  const abs = Math.abs(value);
  const n = Math.floor(abs);
  const diff = abs - n;
  if (diff < 0.5) return sign * n;
  if (diff > 0.5) return sign * (n + 1);
  return sign * (n % 2 === 0 ? n : n + 1);
}

function roundToDecimals(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return roundHalfEven(value * factor) / factor;
}

/**
 * Exact decimal expansion of a finite double (via BigInt), split into the
 * integer part and the (possibly empty) fraction digits.
 *
 * Python's `"{:.Nf}".format(x)` rounds the *exact binary value* correctly
 * (ties to even); neither `toFixed` (round-half-up, so `0.125 -> 0.13`)
 * nor scaled float rounding (`99.999 * 100` loses precision, giving
 * `99.99`) reproduces that, hence the exact path.
 */
function exactDecimalParts(absValue: number): { intPart: string; fracDigits: string } {
  if (absValue === 0) return { intPart: "0", fracDigits: "" };
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setFloat64(0, absValue);
  const bits = new DataView(buffer).getBigUint64(0);
  const expBits = Number((bits >> 52n) & 0x7ffn);
  const fracBits = bits & ((1n << 52n) - 1n);
  let mantissa: bigint;
  let exp2: number;
  if (expBits === 0) {
    mantissa = fracBits;
    exp2 = -1074;
  } else {
    mantissa = (1n << 52n) | fracBits;
    exp2 = expBits - 1075;
  }
  if (exp2 >= 0) {
    return { intPart: (mantissa << BigInt(exp2)).toString(), fracDigits: "" };
  }
  const shift = BigInt(-exp2);
  const scaled = (mantissa * 5n ** shift).toString().padStart(-exp2 + 1, "0");
  const splitAt = scaled.length + exp2;
  return {
    intPart: scaled.slice(0, splitAt) === "" ? "0" : scaled.slice(0, splitAt),
    fracDigits: scaled.slice(splitAt),
  };
}

/** Round an exact `{intPart, fracDigits}` expansion to `decimals` fraction
 * digits, ties to even (mirrors Python float formatting). */
function roundExactDecimals(
  intPart: string,
  fracDigits: string,
  decimals: number,
): { intPart: string; frac: string } {
  const kept = (fracDigits + "0".repeat(decimals)).slice(0, decimals);
  const rest = fracDigits.slice(decimals);
  let increment = false;
  if (rest.length > 0) {
    const first = rest[0] ?? "0";
    if (first > "5") {
      increment = true;
    } else if (first === "5") {
      const tail = rest.slice(1);
      if (tail.split("").some((d) => d !== "0")) {
        increment = true;
      } else {
        // Exact tie: round to even on the last kept digit (or the
        // integer's last digit when no fraction digits are kept).
        const last = kept.length > 0 ? (kept[kept.length - 1] ?? "0") : (intPart[intPart.length - 1] ?? "0");
        increment = Number(last) % 2 === 1;
      }
    }
  }
  if (!increment) return { intPart, frac: kept };
  const combined = BigInt((intPart === "" ? "0" : intPart) + kept) + 1n;
  const text = combined.toString().padStart(intPart.length + decimals, "0");
  if (decimals === 0) return { intPart: text, frac: "" };
  return { intPart: text.slice(0, text.length - decimals), frac: text.slice(text.length - decimals) };
}

/** Apply a machine-generated `{field:sign,comma.decimalsf}` spec to one
 * real component (mirrors Python `spec.format(number)`). */
function formatWithSpec(spec: string, value: number): string {
  const match = /^\{([^:]*):([+-]?)(,?)\.(\d+)f\}$/.exec(spec);
  if (match === null) {
    throw new Error(`Unsupported format spec: ${spec}`);
  }
  const signFlag = match[2] ?? "";
  const commaFlag = match[3] ?? "";
  const decimals = Number(match[4] ?? "0");
  const negative = value < 0 || Object.is(value, -0);
  if (!Number.isFinite(value)) {
    // NOTE: mirrors Python, e.g. f"{float('inf'):.2f}" == "inf",
    // f"{float('inf'):+.2f}" == "+inf".
    const text = Number.isNaN(value) ? "nan" : "inf";
    return (negative ? "-" : signFlag === "+" ? "+" : "") + text;
  }
  const { intPart, fracDigits } = exactDecimalParts(Math.abs(value));
  const rounded = roundExactDecimals(intPart, fracDigits, decimals);
  let outInt = rounded.intPart;
  if (commaFlag === ",") {
    outInt = outInt.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  const out = decimals > 0 ? outInt + "." + rounded.frac : outInt;
  return (negative ? "-" : signFlag === "+" ? "+" : "") + out;
}

export interface DecimalNumberOptions extends VMobjectOptions {
  num_decimal_places?: number;
  mob_class?: DecimalNumberMobFactory;
  include_sign?: boolean;
  group_with_commas?: boolean;
  digit_buff_per_font_unit?: number;
  show_ellipsis?: boolean;
  unit?: string | null;
  unit_buff_per_font_unit?: number;
  include_background_rectangle?: boolean;
  edge_to_fix?: readonly number[] | NDArray;
  font_size?: number;
}

export class DecimalNumber extends VMobject {
  number: DecimalNumberValue;
  num_decimal_places: number;
  include_sign: boolean;
  mob_class: DecimalNumberMobFactory;
  group_with_commas: boolean;
  digit_buff_per_font_unit: number;
  show_ellipsis: boolean;
  unit: string | null;
  unit_buff_per_font_unit: number;
  include_background_rectangle: boolean;
  edge_to_fix: readonly number[] | NDArray;
  _font_size: number;
  initial_config: Record<string, unknown>;
  initial_height = 0;
  unit_sign: VMobject | null = null;
  /** Formatted string of the last build; lets `set_value` skip no-op updates
   * (e.g. a slider jitter that rounds to the same digits). */
  private _last_num_string: string | null = null;

  constructor(numberValue: DecimalNumberValue = 0, options: DecimalNumberOptions = {}) {
    const {
      num_decimal_places = 2,
      mob_class = mathTexFactory,
      include_sign = false,
      group_with_commas = true,
      digit_buff_per_font_unit = 0.001,
      show_ellipsis = false,
      unit = null,
      unit_buff_per_font_unit = 0,
      include_background_rectangle = false,
      edge_to_fix = [...LEFT],
      font_size = DEFAULT_FONT_SIZE,
      stroke_width = 0,
      fill_opacity = 1.0,
      ...kwargs
    } = options;
    super({ ...kwargs, fill_opacity, stroke_width });
    this.number = numberValue;
    this.num_decimal_places = num_decimal_places;
    this.include_sign = include_sign;
    this.mob_class = mob_class;
    this.group_with_commas = group_with_commas;
    this.digit_buff_per_font_unit = digit_buff_per_font_unit;
    this.show_ellipsis = show_ellipsis;
    this.unit = unit;
    this.unit_buff_per_font_unit = unit_buff_per_font_unit;
    this.include_background_rectangle = include_background_rectangle;
    this.edge_to_fix = edge_to_fix;
    this._font_size = font_size;
    this.fill_opacity = fill_opacity;

    this.initial_config = { ...kwargs };
    Object.assign(this.initial_config, {
      num_decimal_places,
      include_sign,
      group_with_commas,
      digit_buff_per_font_unit,
      show_ellipsis,
      unit,
      unit_buff_per_font_unit,
      include_background_rectangle,
      edge_to_fix,
      font_size,
      stroke_width,
      fill_opacity,
    });

    this._set_submobjects_from_number(numberValue);
    this.init_colors();
  }

  /** The font size of the tex mobject. */
  get font_size(): number {
    return (this.height / this.initial_height) * this._font_size;
  }

  set font_size(font_val: number) {
    if (font_val <= 0) {
      const error = new Error("font_size must be greater than 0.");
      error.name = "ValueError";
      throw error;
    } else if (this.height > 0) {
      // Sometimes manim generates a SingleStringMathTex mobject with
      // 0 height; it can't be scaled regardless, hence the guard.
      // Scale to a factor of the initial height so that setting
      // font_size does not depend on current size.
      this.scale(font_val / this.font_size);
    }
  }

  _set_submobjects_from_number(numberValue: DecimalNumberValue): void {
    this.number = numberValue;
    this.submobjects = [];

    const num_string = this._get_num_string(numberValue);
    this._last_num_string = num_string;
    const font_size = this._font_size;
    const buff = this.digit_buff_per_font_unit * font_size;

    // NOTE: the layout below reproduces the exact sequence of
    // next_to/center/move_to/align_to/shift calls from Python, but computes
    // the translations arithmetically from cached glyph extents and applies
    // them with a single in-place point translation per glyph. The original
    // per-step calls each re-scan every SVG point through the ndarray helpers
    // (get_critical_point -> get_points_defining_boundary), which dominated
    // DecimalNumber construction. Verified to match to ~1e-16.
    const parts: NumberPart[] = [];
    const addGlyph = (
      stringValue: string,
      factory: DecimalNumberMobFactory,
      kwargs: Record<string, unknown> = {},
    ): NumberPart => {
      const proto = glyph_proto(stringValue, factory, font_size, kwargs);
      const info = glyph_info_map.get(proto) as GlyphInfo;
      const mob = proto.copy() as VMobject;
      this.add(mob);
      const part: NumberPart = { mob, info, tx: 0, ty: 0 };
      parts.push(part);
      return part;
    };
    const bounds = (): { minX: number; maxX: number; minY: number; maxY: number } => {
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const part of parts) {
        minX = Math.min(minX, part.info.minX + part.tx);
        maxX = Math.max(maxX, part.info.maxX + part.tx);
        minY = Math.min(minY, part.info.minY + part.ty);
        maxY = Math.max(maxY, part.info.maxY + part.ty);
      }
      return { minX, maxX, minY, maxY };
    };

    for (const ch of num_string) addGlyph(ch, this.mob_class);

    // Add non-numerical bits
    if (this.show_ellipsis) {
      addGlyph("\\dots", singleStringMathTexFactory, { color: this.color });
    }

    // NOTE: mirrors `self.arrange(buff=..., aligned_edge=DOWN)`, i.e.
    // `next_to(prev, RIGHT, buff, DOWN)` per glyph.
    for (let index = 1; index < parts.length; index += 1) {
      const prev = parts[index - 1] as NumberPart;
      const cur = parts[index] as NumberPart;
      cur.tx += prev.info.maxX + prev.tx - (cur.info.minX + cur.tx) + buff;
      cur.ty += prev.info.minY + prev.ty - (cur.info.minY + cur.ty);
    }
    this._shift_parts(parts, bounds()); // this.center()

    if (this.unit !== null && this.unit !== undefined) {
      const groupBounds = bounds();
      const unitPart = addGlyph(this.unit, singleStringMathTexFactory);
      unitPart.tx =
        groupBounds.maxX -
        unitPart.info.minX +
        (this.unit_buff_per_font_unit + this.digit_buff_per_font_unit) * font_size;
      unitPart.ty = groupBounds.minY - unitPart.info.minY;
      this.unit_sign = unitPart.mob;
    }

    this._shift_parts(parts, bounds()); // this.move_to(ORIGIN)

    // Handle alignment of parts that should be aligned to the bottom
    for (let i = 0; i < num_string.length; i += 1) {
      const c = num_string[i] ?? "";
      if (c === "-" && num_string.length > i + 1) {
        const cur = parts[i] as NumberPart;
        const next = parts[i + 1] as NumberPart;
        // align_to(next, UP): match anchor tops.
        cur.ty += next.info.maxY + next.ty - (cur.info.maxY + cur.ty);
        // shift(DOWN * next.height / 2): height reads *all* points.
        cur.ty += -(next.info.fullMaxY - next.info.fullMinY) / 2;
      } else if (c === ",") {
        const cur = parts[i] as NumberPart;
        cur.ty += -(cur.info.fullMaxY - cur.info.fullMinY) / 2;
      }
    }
    if (this.unit && this.unit.startsWith("^")) {
      const unitPart = parts[parts.length - 1] as NumberPart;
      unitPart.ty += bounds().maxY - (unitPart.info.maxY + unitPart.ty);
    }

    for (const part of parts) {
      if (part.tx !== 0 || part.ty !== 0) {
        translate_mob_in_place(part.mob, [part.tx, part.ty, 0]);
      }
    }

    // Track the initial height to enable scaling via font_size. `height`
    // (length_over_dim) reads every point, not just the anchors.
    let fullMinY = Infinity;
    let fullMaxY = -Infinity;
    for (const part of parts) {
      fullMinY = Math.min(fullMinY, part.info.fullMinY + part.ty);
      fullMaxY = Math.max(fullMaxY, part.info.fullMaxY + part.ty);
    }
    this.initial_height = fullMaxY - fullMinY;

    if (this.include_background_rectangle) {
      this.add_background_rectangle();
    }
  }

  /** Apply the cumulative translation of a `center()`/`move_to()` step to
   * every part (mirrors `Mobject.shift` over the whole group). */
  private _shift_parts(
    parts: NumberPart[],
    target: { minX: number; maxX: number; minY: number; maxY: number },
  ): void {
    const dx = (target.minX + target.maxX) / 2;
    const dy = (target.minY + target.maxY) / 2;
    for (const part of parts) {
      part.tx -= dx;
      part.ty -= dy;
    }
  }

  _get_num_string(numberValue: DecimalNumberValue): string {
    let num_string: string;
    let roundedIsZero: boolean;
    if (typeof numberValue === "number") {
      num_string = formatWithSpec(this._get_formatter(), numberValue);
      roundedIsZero = roundToDecimals(numberValue, this.num_decimal_places) === 0;
    } else {
      num_string =
        formatWithSpec(
          this._get_formatter({ field_name: "0.real" }),
          numberValue.re,
        ) +
        formatWithSpec(
          this._get_formatter({ field_name: "0.imag", include_sign: true }),
          numberValue.im,
        ) +
        "i";
      roundedIsZero =
        roundToDecimals(numberValue.re, this.num_decimal_places) === 0 &&
        roundToDecimals(numberValue.im, this.num_decimal_places) === 0;
    }

    if (num_string.startsWith("-") && roundedIsZero) {
      num_string = (this.include_sign ? "+" : "") + num_string.slice(1);
    }
    return num_string;
  }

  _string_to_mob(
    stringValue: string,
    mob_class: DecimalNumberMobFactory | null = null,
    kwargs: Record<string, unknown> = {},
  ): VMobject {
    const factory = mob_class ?? this.mob_class;
    // The cached prototype is already sized to `_font_size`, so the copy is
    // ready to use without the previous (always-run) `font_size =` rescale.
    return glyph_proto(stringValue, factory, this._font_size, kwargs).copy() as VMobject;
  }

  _get_formatter(
    kwargs: {
      include_sign?: boolean;
      group_with_commas?: boolean;
      num_decimal_places?: number;
      field_name?: string;
    } = {},
  ): string {
    // Configuration is based first off instance attributes, but
    // overwritten by any keyword argument. Relevant keys:
    // - include_sign
    // - group_with_commas
    // - num_decimal_places
    // - field_name (e.g. 0 or 0.real)
    const include_sign = kwargs.include_sign ?? this.include_sign;
    const group_with_commas = kwargs.group_with_commas ?? this.group_with_commas;
    const num_decimal_places = kwargs.num_decimal_places ?? this.num_decimal_places;
    const field_name = kwargs.field_name ?? "";
    return `{${field_name}:${include_sign ? "+" : ""}${group_with_commas ? "," : ""}.${num_decimal_places}f}`;
  }

  _get_complex_formatter(): string {
    return (
      this._get_formatter({ field_name: "0.real" }) +
      this._get_formatter({ field_name: "0.imag", include_sign: true }) +
      "i"
    );
  }

  set_value(numberValue: DecimalNumberValue): this {
    // Creates a new number mob via `set_submobjects_from_number`, then
    // matches the properties (color, font_size, etc.) of the previous
    // mobject to the new one.

    // A value that formats to the same string (jitter within the displayed
    // precision) leaves the mobject identical; rebuilding it would only burn
    // glyph copies on every frame of a drag.
    const nextString = this._get_num_string(numberValue);
    if (nextString === this._last_num_string) {
      this.number = numberValue;
      return this;
    }

    // old_family needed with cairo
    const old_family = this.get_family();

    const old_font_size = this.font_size;
    const move_to_point = this.get_edge_center(this.edge_to_fix);
    const old_submobjects = [...this.submobjects];

    this._set_submobjects_from_number(numberValue);
    // The glyph prototypes are cached at `_font_size`, so an unscaled number
    // is already the right size and the round-trip through the `font_size`
    // setter (height scan + full point rescale) is a no-op.  Only rescale when
    // the number was actually scaled away from its base size.
    if (Math.abs(old_font_size - this._font_size) > 1e-9) {
      this.font_size = old_font_size;
    }
    this.move_to(move_to_point, this.edge_to_fix);
    // Style every new glyph from the previous generation (the last old glyph's
    // style covers any extra glyphs), which makes the trailing `init_colors`
    // propagation redundant.
    if (old_submobjects.length > 0) {
      const fallback = old_submobjects[old_submobjects.length - 1] as VMobject;
      for (let index = 0; index < this.submobjects.length; index += 1) {
        const source = (old_submobjects[index] ?? fallback) as VMobject;
        (this.submobjects[index] as VMobject).match_style(source);
      }
    } else {
      this.init_colors();
    }

    if (config.renderer === RendererType.CAIRO) {
      for (const mob of old_family) {
        // Dumb hack...due to how scene handles families of animated
        // mobjects; for compatibility with updaters to not leave the first
        // number in place while updating (not needed with opengl renderer).
        mob.points.fill(0);
      }
    }
    return this;
  }

  get_value(): DecimalNumberValue {
    return this.number;
  }

  increment_value(delta_t = 1): this {
    const value = this.get_value();
    if (typeof value === "number") {
      this.set_value(value + delta_t);
    } else {
      // NOTE: Python does `get_value() + delta_t`, which for a complex
      // value adds the float to the real part; mirrored here.
      this.set_value({ re: value.re + delta_t, im: value.im });
    }
    return this;
  }
}

export class Integer extends DecimalNumber {
  constructor(numberValue: DecimalNumberValue = 0, options: DecimalNumberOptions = {}) {
    super(numberValue, {
      ...options,
      num_decimal_places: options.num_decimal_places ?? 0,
    });
  }

  override get_value(): number {
    // NOTE: mirrors `int(np.round(super().get_value()))` (banker's
    // rounding via np.round, then truncation toward zero via int).
    const raw = super.get_value();
    const scalar = typeof raw === "number" ? raw : raw.re;
    return Math.trunc(roundToDecimals(scalar, 0));
  }
}

export type VariableLabel = string | VMobject;

export class Variable extends VMobject {
  label: VMobject;
  tracker: ValueTracker;
  value!: DecimalNumber | Integer;

  constructor(
    // NOTE: the Python parameter is named `var`; renamed because `var`
    // is a reserved word in TS/JS (positional calls are unaffected).
    variable: number,
    label: VariableLabel,
    var_type: typeof DecimalNumber | typeof Integer = DecimalNumber,
    num_decimal_places = 2,
    options: VMobjectOptions = {},
  ) {
    // NOTE: Python assigns label/tracker/value before
    // super().__init__(**kwargs); TS requires super() first (a language
    // constraint with no observable effect here). The label/equals steps
    // below still run before the tracker/value steps, as in Python.
    super(options);
    // NOTE: mirrors Python `self.label = MathTex(label) if isinstance(label,
    // str) else label`.
    if (typeof label === "string") {
      this.label = mathTexFactory(label);
    } else {
      this.label = label;
    }
    const equals = mathTexFactory("=");
    equals.next_to(this.label, RIGHT);
    this.label.add(equals);

    this.tracker = new ValueTracker(variable);

    if (var_type === DecimalNumber) {
      this.value = new DecimalNumber(this.tracker.get_value(), {
        num_decimal_places,
      });
    } else if (var_type === Integer) {
      this.value = new Integer(this.tracker.get_value());
    }

    this.value
      .add_updater((v: Mobject) => {
        (v as DecimalNumber).set_value(this.tracker.get_value());
      })
      .next_to(this.label, RIGHT);

    this.add(this.label, this.value);
  }
}
