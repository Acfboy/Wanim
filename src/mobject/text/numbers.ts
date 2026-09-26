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
  DOWN,
  LEFT,
  ORIGIN,
  RendererType,
  RIGHT,
  UP,
} from "../../constants.js";
import { config } from "../../_config/utils.js";
import { scalePoint, type ComplexPoint } from "../../utils/arrays.js";
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
    for (const ch of num_string) {
      this.add(this._string_to_mob(ch));
    }

    // Add non-numerical bits
    if (this.show_ellipsis) {
      this.add(
        this._string_to_mob("\\dots", singleStringMathTexFactory, {
          color: this.color,
        }),
      );
    }

    // NOTE: mirrors `self.arrange(buff=..., aligned_edge=DOWN)`; the TS
    // Mobject.arrange() drops kwargs, so the loop is inlined with the
    // aligned edge passed to next_to explicitly.
    const buff = this.digit_buff_per_font_unit * this._font_size;
    for (let index = 0; index < this.submobjects.length - 1; index += 1) {
      (this.submobjects[index + 1] as Mobject).next_to(
        this.submobjects[index] as Mobject,
        RIGHT,
        buff,
        DOWN,
      );
    }
    this.center();

    if (this.unit !== null && this.unit !== undefined) {
      this.unit_sign = this._string_to_mob(
        this.unit,
        singleStringMathTexFactory,
      );
      this.add(
        (this.unit_sign as VMobject).next_to(
          this,
          RIGHT,
          (this.unit_buff_per_font_unit + this.digit_buff_per_font_unit) *
            this._font_size,
          DOWN,
        ),
      );
    }

    this.move_to(ORIGIN);

    // Handle alignment of parts that should be aligned to the bottom
    for (let i = 0; i < num_string.length; i += 1) {
      const c = num_string[i] ?? "";
      if (c === "-" && num_string.length > i + 1) {
        (this.__getitem__(i) as Mobject).align_to(
          this.__getitem__(i + 1) as Mobject,
          UP,
        );
        (this.__getitem__(i) as Mobject).shift(
          scalePoint(DOWN, ((this.__getitem__(i + 1) as Mobject).height ?? 0) / 2, this.dim),
        );
      } else if (c === ",") {
        (this.__getitem__(i) as Mobject).shift(
          scalePoint(DOWN, ((this.__getitem__(i) as Mobject).height ?? 0) / 2, this.dim),
        );
      }
    }
    if (this.unit && this.unit.startsWith("^")) {
      (this.unit_sign as VMobject).align_to(this, UP);
    }

    // Track the initial height to enable scaling via font_size
    this.initial_height = this.height;

    if (this.include_background_rectangle) {
      this.add_background_rectangle();
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
    let cached = string_to_mob_map.get(stringValue);
    if (cached === undefined) {
      cached = factory(stringValue, kwargs);
      string_to_mob_map.set(stringValue, cached);
    }
    const mob = cached.copy() as VMobject;
    // NOTE: the tex classes carry a font_size property (used by
    // NumberLine.add_labels); plain VMobjects don't, so assign through a
    // structural type instead of the class interface.
    (mob as unknown as { font_size: number }).font_size = this._font_size;
    return mob;
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

    // old_family needed with cairo
    const old_family = this.get_family();

    const old_font_size = this.font_size;
    const move_to_point = this.get_edge_center(this.edge_to_fix);
    const old_submobjects = [...this.submobjects];

    this._set_submobjects_from_number(numberValue);
    this.font_size = old_font_size;
    this.move_to(move_to_point, this.edge_to_fix);
    const count = Math.min(this.submobjects.length, old_submobjects.length);
    for (let index = 0; index < count; index += 1) {
      (this.submobjects[index] as VMobject).match_style(
        old_submobjects[index] as VMobject,
      );
    }

    if (config.renderer === RendererType.CAIRO) {
      for (const mob of old_family) {
        // Dumb hack...due to how scene handles families of animated
        // mobjects; for compatibility with updaters to not leave the first
        // number in place while updating (not needed with opengl renderer).
        mob.points.fill(0);
      }
    }

    this.init_colors();
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
