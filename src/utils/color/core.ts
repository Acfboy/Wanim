import { array, NDArray } from "numpy-ts";
import { isNDArray, nestedRows, pointRows } from "../arrays.js";

export type ColorArray = NDArray<"float64">;
export type ParsableManimColor =
  | ManimColor
  | string
  | number
  | boolean
  | bigint
  | number[]
  | readonly number[]
  | NDArray
  | { get_hex: () => string }
  | { getHex: () => string };

const re_hex = /((?<=#)|(?<=0x))[A-F0-9]{3,8}/i;

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

function typeError(message: string): Error {
  return new TypeError(message);
}

const NAMED_COLORS: Record<string, [number, number, number]> = {
  BLACK: [0, 0, 0],
  WHITE: [1, 1, 1],
  BLUE: [0.34509803921568627, 0.7686274509803922, 0.8666666666666667],
  BLUE_A: [0.7803921568627451, 0.9137254901960784, 0.9450980392156862],
  BLUE_B: [0.611764705882353, 0.8627450980392157, 0.9215686274509803],
  BLUE_C: [0.34509803921568627, 0.7686274509803922, 0.8666666666666667],
  BLUE_D: [0.1607843137254902, 0.6705882352941176, 0.792156862745098],
  BLUE_E: [0.13725490196078433, 0.4196078431372549, 0.5568627450980392],
  DARKER_GRAY: [0.13333333333333333, 0.13333333333333333, 0.13333333333333333],
  DARKER_GREY: [0.13333333333333333, 0.13333333333333333, 0.13333333333333333],
  DARK_BLUE: [0.13725490196078433, 0.4196078431372549, 0.5568627450980392],
  DARK_BROWN: [0.5450980392156862, 0.27058823529411763, 0.07450980392156863],
  DARK_GRAY: [0.26666666666666666, 0.26666666666666666, 0.26666666666666666],
  DARK_GREY: [0.26666666666666666, 0.26666666666666666, 0.26666666666666666],
  GOLD: [0.9411764705882353, 0.6745098039215687, 0.37254901960784315],
  GOLD_A: [0.9686274509803922, 0.7803921568627451, 0.592156862745098],
  GOLD_B: [0.9764705882352941, 0.7176470588235294, 0.4588235294117647],
  GOLD_C: [0.9411764705882353, 0.6745098039215687, 0.37254901960784315],
  GOLD_D: [0.8823529411764706, 0.6313725490196078, 0.34509803921568627],
  GOLD_E: [0.7803921568627451, 0.5529411764705883, 0.27450980392156865],
  GRAY: [0.5333333333333333, 0.5333333333333333, 0.5333333333333333],
  GRAY_A: [0.8666666666666667, 0.8666666666666667, 0.8666666666666667],
  GRAY_B: [0.7333333333333333, 0.7333333333333333, 0.7333333333333333],
  GRAY_BROWN: [0.45098039215686275, 0.38823529411764707, 0.3411764705882353],
  GRAY_C: [0.5333333333333333, 0.5333333333333333, 0.5333333333333333],
  GRAY_D: [0.26666666666666666, 0.26666666666666666, 0.26666666666666666],
  GRAY_E: [0.13333333333333333, 0.13333333333333333, 0.13333333333333333],
  GREEN: [0.5137254901960784, 0.7568627450980392, 0.403921568627451],
  GREEN_A: [0.788235294117647, 0.8862745098039215, 0.6823529411764706],
  GREEN_B: [0.6509803921568628, 0.8117647058823529, 0.5490196078431373],
  GREEN_C: [0.5137254901960784, 0.7568627450980392, 0.403921568627451],
  GREEN_D: [0.4666666666666667, 0.6901960784313725, 0.36470588235294116],
  GREEN_E: [0.4117647058823529, 0.611764705882353, 0.3215686274509804],
  GREY: [0.5333333333333333, 0.5333333333333333, 0.5333333333333333],
  GREY_A: [0.8666666666666667, 0.8666666666666667, 0.8666666666666667],
  GREY_B: [0.7333333333333333, 0.7333333333333333, 0.7333333333333333],
  GREY_BROWN: [0.45098039215686275, 0.38823529411764707, 0.3411764705882353],
  GREY_C: [0.5333333333333333, 0.5333333333333333, 0.5333333333333333],
  GREY_D: [0.26666666666666666, 0.26666666666666666, 0.26666666666666666],
  GREY_E: [0.13333333333333333, 0.13333333333333333, 0.13333333333333333],
  LIGHTER_GRAY: [0.8666666666666667, 0.8666666666666667, 0.8666666666666667],
  LIGHTER_GREY: [0.8666666666666667, 0.8666666666666667, 0.8666666666666667],
  LIGHT_BROWN: [0.803921568627451, 0.5215686274509804, 0.24705882352941178],
  LIGHT_GRAY: [0.7333333333333333, 0.7333333333333333, 0.7333333333333333],
  LIGHT_GREY: [0.7333333333333333, 0.7333333333333333, 0.7333333333333333],
  LIGHT_PINK: [0.8627450980392157, 0.4588235294117647, 0.803921568627451],
  LOGO_BLACK: [0.20392156862745098, 0.20392156862745098, 0.20392156862745098],
  LOGO_BLUE: [0.3215686274509804, 0.34509803921568627, 0.5764705882352941],
  LOGO_GREEN: [0.5294117647058824, 0.7607843137254902, 0.6470588235294118],
  LOGO_RED: [0.8784313725490196, 0.47843137254901963, 0.37254901960784315],
  LOGO_WHITE: [0.9254901960784314, 0.9058823529411765, 0.8862745098039215],
  MAROON: [0.7725490196078432, 0.37254901960784315, 0.45098039215686275],
  MAROON_A: [0.9254901960784314, 0.6705882352941176, 0.7568627450980392],
  MAROON_B: [0.9254901960784314, 0.5725490196078431, 0.6705882352941176],
  MAROON_C: [0.7725490196078432, 0.37254901960784315, 0.45098039215686275],
  MAROON_D: [0.6352941176470588, 0.30196078431372547, 0.3803921568627451],
  MAROON_E: [0.5803921568627451, 0.25882352941176473, 0.30980392156862746],
  ORANGE: [1, 0.5254901960784314, 0.1843137254901961],
  PINK: [0.8196078431372549, 0.2784313725490196, 0.7411764705882353],
  PURE_BLUE: [0, 0, 1],
  PURE_CYAN: [0, 1, 1],
  PURE_GREEN: [0, 1, 0],
  PURE_MAGENTA: [1, 0, 1],
  PURE_RED: [1, 0, 0],
  PURE_YELLOW: [1, 1, 0],
  PURPLE: [0.6039215686274509, 0.4470588235294118, 0.6745098039215687],
  PURPLE_A: [0.792156862745098, 0.6392156862745098, 0.9098039215686274],
  PURPLE_B: [0.6941176470588235, 0.5372549019607843, 0.7764705882352941],
  PURPLE_C: [0.6039215686274509, 0.4470588235294118, 0.6745098039215687],
  PURPLE_D: [0.44313725490196076, 0.3333333333333333, 0.5098039215686274],
  PURPLE_E: [0.39215686274509803, 0.2549019607843137, 0.4470588235294118],
  RED: [0.9882352941176471, 0.3843137254901961, 0.3333333333333333],
  RED_A: [0.9686274509803922, 0.6313725490196078, 0.6392156862745098],
  RED_B: [1, 0.5019607843137255, 0.5019607843137255],
  RED_C: [0.9882352941176471, 0.3843137254901961, 0.3333333333333333],
  RED_D: [0.9019607843137255, 0.35294117647058826, 0.2980392156862745],
  RED_E: [0.8117647058823529, 0.3137254901960784, 0.26666666666666666],
  TEAL: [0.3607843137254902, 0.8156862745098039, 0.7019607843137254],
  TEAL_A: [0.6745098039215687, 0.9176470588235294, 0.8431372549019608],
  TEAL_B: [0.4627450980392157, 0.8666666666666667, 0.7529411764705882],
  TEAL_C: [0.3607843137254902, 0.8156862745098039, 0.7019607843137254],
  TEAL_D: [0.3333333333333333, 0.7568627450980392, 0.6549019607843137],
  TEAL_E: [0.28627450980392155, 0.6588235294117647, 0.5607843137254902],
  YELLOW: [0.9686274509803922, 0.8509803921568627, 0.43529411764705883],
  YELLOW_A: [1, 0.9450980392156862, 0.7137254901960784],
  YELLOW_B: [1, 0.9176470588235294, 0.5803921568627451],
  YELLOW_C: [0.9686274509803922, 0.8509803921568627, 0.43529411764705883],
  YELLOW_D: [0.9568627450980393, 0.8274509803921568, 0.27058823529411763],
  YELLOW_E: [0.9098039215686274, 0.7568627450980392, 0.10980392156862745],
};

function valuesOf(value: unknown): number[] {
  if (isNDArray(value)) {
    const rows = nestedRows(value);
    return Array.isArray(rows[0]) ? (rows[0] as unknown[]).map(Number) : [Number(rows[0])];
  }
  if (Array.isArray(value)) return value.map(Number);
  return [Number(value)];
}

export class ManimColor {
  protected internal: ColorArray;

  constructor(value: ParsableManimColor | null | undefined = null, alpha = 1) {
    if (value === null || value === undefined) {
      this.internal = array([0, 0, 0, alpha]) as ColorArray;
    } else if (value instanceof ManimColor) {
      // NOTE: Python shares the underlying array reference here
      // (self._internal_value = value._internal_value). We copy instead so
      // that later in-place alpha writes cannot alias another instance.
      this.internal = value.internal.copy() as ColorArray;
    } else if (typeof value === "number" || typeof value === "bigint" || typeof value === "boolean") {
      // NOTE: Python checks isinstance(value, int) which matches bool but
      // rejects float scalars (e.g. ManimColor(3.5) raises TypeError).
      // Mirror that: non-integer numbers are rejected instead of truncated.
      const numeric = Number(value);
      if (!Number.isInteger(numeric)) {
        throw typeError(
          "ManimColor only accepts int, str, list[int, int, int], " +
            "list[int, int, int, int], list[float, float, float], " +
            "list[float, float, float, float], not float",
        );
      }
      // NOTE: Python checks isinstance(value, int) which also matches bool.
      this.internal = ManimColor._internal_from_integer(numeric, alpha);
    } else if (typeof value === "string") {
      // Mirrors Python: first search for a hex payload after #/0x, else
      // fall back to named-color lookup.
      const found = value.match(re_hex);
      if (found !== null) {
        this.internal = ManimColor._internal_from_hex_string(found[0], alpha);
      } else {
        this.internal = ManimColor._internal_from_string(value, alpha);
      }
    } else if (Array.isArray(value) || isNDArray(value)) {
      const rawItems: unknown[] = isNDArray(value)
        ? [...((value as NDArray).toArray() as unknown[]).map((item) => item)]
        : [...(value as readonly unknown[])];
      const length = rawItems.length;
      if (length !== 3 && length !== 4) {
        throw valueError(
          `ManimColor only accepts lists/tuples/arrays of length 3 or 4, not ${length}`,
        );
      }
      // Mirrors Python `all(isinstance(x, float) for x in value)`: every
      // element must be a non-integer number to take the float path. Mixed
      // int/float input takes the int path (everything scaled by 1/255).
      // NOTE: a Python float like 1.0 arrives in JS as integer 1 and is
      // therefore treated as int; this boundary is inherently lossy in TS.
      // For NDArray input the dtype decides instead (numpy float scalars
      // subclass Python float, int scalars do not); numpy-ts infers float64
      // for plain array(...) calls, so only explicitly-typed int arrays take
      // the int path.
      const dtype = isNDArray(value) ? String((value as NDArray).dtype ?? "") : "";
      const allFloat = /int|uint|bool/i.test(dtype)
        ? false
        : /float|complex/i.test(dtype)
          ? true
          : rawItems.every(
              (item) => typeof item === "number" && !Number.isInteger(item),
            );
      if (allFloat) {
        if (length === 3) {
          this.internal = ManimColor._internal_from_rgb(rawItems as number[], alpha);
        } else {
          this.internal = ManimColor._internal_from_rgba(rawItems as number[]);
        }
      } else {
        if (length === 3) {
          this.internal = ManimColor._internal_from_int_rgb(rawItems as number[], alpha);
        } else {
          this.internal = ManimColor._internal_from_int_rgba(rawItems as number[]);
        }
      }
    } else if (
      typeof value === "object" &&
      value !== null &&
      ((typeof (value as { get_hex?: unknown }).get_hex === "function") ||
        (typeof (value as { getHex?: unknown }).getHex === "function"))
    ) {
      const holder = value as { get_hex?: () => string; getHex?: () => string };
      const hexSource = typeof holder.get_hex === "function" ? holder.get_hex() : (holder.getHex as () => string)();
      const found = hexSource.match(re_hex);
      if (found === null) throw valueError(`Failed to parse a color from ${String(value)}`);
      this.internal = ManimColor._internal_from_hex_string(found[0], alpha);
    } else {
      throw typeError(
        "ManimColor only accepts int, str, list[int, int, int], " +
          "list[int, int, int, int], list[float, float, float], " +
          `list[float, float, float, float], not ${(value as object)?.constructor?.name ?? typeof value}`,
      );
    }
  }

  static _construct_from_space(
    this: typeof ManimColor,
    space: number[] | readonly number[] | ColorArray,
  ): ManimColor {
    if (this === ManimColor) {
      // Fast plain-RGBA path: bypasses the int/float ambiguity of the
      // generic constructor (integral floats like 0.0/1.0 arrive in JS as
      // integers, so routing [0,1,1,1] through the constructor would take
      // the int-RGB path and scale by 1/255). Python has no such ambiguity
      // because numpy float scalars subclass Python float.
      const inst = new ManimColor(null);
      inst._internal_value = array([...valuesOf(space)].map(Number)) as ColorArray;
      return inst;
    }
    const ctor = this;
    return new ctor(space as unknown as ParsableManimColor);
  }

  static _from_internal(this: typeof ManimColor, value: ColorArray): ManimColor {
    if (this === ManimColor) {
      const inst = new ManimColor(null);
      inst._internal_value = (
        isNDArray(value) ? (value as ColorArray).copy() : array([...valuesOf(value)].map(Number))
      ) as ColorArray;
      return inst;
    }
    const ctor = this;
    return new ctor(value as unknown as ParsableManimColor);
  }

  static parse(
    this: typeof ManimColor,
    color: ParsableManimColor | readonly ParsableManimColor[] | null | undefined,
    alpha = 1,
  ): ManimColor | ManimColor[] {
    const ctor = this;
    const isSequence = Array.isArray(color);
    if (isSequence) {
      return (color as readonly ParsableManimColor[]).map((c) =>
        ctor._from_internal(new ManimColor(c, alpha)._internal_value),
      );
    }
    return ctor._from_internal(
      new ManimColor(color as ParsableManimColor | null | undefined, alpha)._internal_value,
    );
  }

  static from_rgb(
    this: typeof ManimColor,
    rgb: number[] | readonly number[] | ColorArray,
    alpha = 1,
  ): ManimColor {
    const ctor = this;
    return ctor._from_internal(new ManimColor(rgb as unknown as ParsableManimColor, alpha)._internal_value);
  }

  static from_rgba(
    this: typeof ManimColor,
    rgba: number[] | readonly number[] | ColorArray,
  ): ManimColor {
    const ctor = this;
    // NOTE: Python decides int-vs-float via `all(isinstance(x, float))`, so
    // [0.5, 0.25, 1.0, 0.5] takes the float path. In JS, integral floats
    // like 1.0 arrive as integers and the constructor would misroute such
    // input to the int (1/255) path. Since a 4-component rgba with every
    // value inside [0, 1] is overwhelmingly float data (int callers use the
    // 0-255 range with some component > 1), route that case directly through
    // the float path. Deviation: all-integer input within [0, 1] (e.g.
    // [0, 0, 0, 1]) is treated as float, whereas Python treats it as int.
    if (!isNDArray(rgba) && Array.isArray(rgba) && rgba.length === 4) {
      const items = (rgba as readonly unknown[]).map(Number);
      if (items.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) {
        return ctor._from_internal(ManimColor._internal_from_rgba(items));
      }
    }
    return new ctor(rgba as unknown as ParsableManimColor);
  }

  static from_hex(
    this: typeof ManimColor,
    hex_str: string,
    alpha = 1,
  ): ManimColor {
    const ctor = this;
    return ctor._from_internal(new ManimColor(hex_str, alpha)._internal_value);
  }

  static from_hsv(
    this: typeof ManimColor,
    hsv: number[] | readonly number[] | ColorArray,
    alpha = 1,
  ): ManimColor {
    const ctor = this;
    const items = valuesOf(hsv).map(Number);
    const rgb = hsvToRgb(items[0] ?? 0, items[1] ?? 0, items[2] ?? 0);
    // NOTE: colorsys returns Python floats, so the float path applies even
    // when the values happen to be integral (e.g. [1,0,0]); route directly
    // through _internal_from_rgb instead of the ambiguity-prone constructor.
    return ctor._from_internal(ManimColor._internal_from_rgb(rgb, alpha));
  }

  static from_hsl(
    this: typeof ManimColor,
    hsl: number[] | readonly number[] | ColorArray,
    alpha = 1,
  ): ManimColor {
    const ctor = this;
    const items = valuesOf(hsl).map(Number);
    const rgb = hlsToRgb(items[0] ?? 0, items[2] ?? 0, items[1] ?? 0);
    // NOTE: same computed-float reasoning as from_hsv above.
    return ctor._from_internal(ManimColor._internal_from_rgb(rgb, alpha));
  }

  static gradient(colors: ManimColor[], _length: number): ManimColor | ManimColor[] {
    void colors;
    const error = new Error("NotImplementedError");
    error.name = "NotImplementedError";
    throw error;
  }

  get internal_value(): ColorArray {
    return this.internal;
  }

  get _internal_value(): ColorArray {
    return this.internal;
  }

  set _internal_value(value: ColorArray) {
    if (!isNDArray(value)) {
      throw new TypeError("Value must be a NumPy array.");
    }
    if ((value as NDArray).shape[0] !== 4) {
      throw new TypeError("Array must have exactly 4 values.");
    }
    (this as unknown as { internal: ColorArray }).internal = value;
  }

  static _internal_from_integer(value: number, alpha: number): ColorArray {
    const normalized = Math.trunc(value);
    return array([
      ((normalized >>> 16) & 0xff) / 255,
      ((normalized >>> 8) & 0xff) / 255,
      ((normalized >>> 0) & 0xff) / 255,
      alpha,
    ]) as ColorArray;
  }

  static _internal_from_hex_string(hex_: string, alpha: number): ColorArray {
    let hex = hex_;
    if (hex.length === 3 || hex.length === 4) hex = [...hex].map((part) => part + part).join("");
    if (hex.length === 6) hex += "FF";
    else if (hex.length === 8) alpha = (Number.parseInt(hex, 16) & 0xff) / 255;
    else {
      const error = new Error(
        "Hex colors must be specified with either 0x or # as prefix and contain 6 or 8 hexadecimal numbers",
      );
      error.name = "ValueError";
      throw error;
    }
    const tmp = Number.parseInt(hex, 16);
    if (Number.isNaN(tmp)) {
      const error = new Error(
        "Hex colors must be specified with either 0x or # as prefix and contain 6 or 8 hexadecimal numbers",
      );
      error.name = "ValueError";
      throw error;
    }
    return array([
      ((tmp >>> 24) & 0xff) / 255,
      ((tmp >>> 16) & 0xff) / 255,
      ((tmp >>> 8) & 0xff) / 255,
      alpha,
    ]) as ColorArray;
  }

  static _internal_from_int_rgb(rgb: readonly number[] | NDArray, alpha = 1): ColorArray {
    const items = valuesOf(rgb).map(Number);
    const value = [0, 1, 2, 3].map((index) => (items[index] ?? 0) / 255);
    value[3] = alpha;
    return array(value) as ColorArray;
  }

  static _internal_from_rgb(rgb: readonly number[] | NDArray, alpha = 1): ColorArray {
    const items = valuesOf(rgb).map(Number);
    const value = [0, 1, 2, 3].map((index) => items[index] ?? 0);
    value[3] = alpha;
    return array(value) as ColorArray;
  }

  static _internal_from_int_rgba(rgba: readonly number[] | NDArray): ColorArray {
    return array(valuesOf(rgba).map((item) => Number(item) / 255)) as ColorArray;
  }

  static _internal_from_rgba(rgba: readonly number[] | NDArray): ColorArray {
    return array(valuesOf(rgba).map(Number)) as ColorArray;
  }

  static _internal_from_string(name: string, alpha: number): ColorArray {
    const named = NAMED_COLORS[name.toUpperCase()];
    if (named) return array([...named.slice(0, 3), alpha]) as ColorArray;
    const error = new Error(`Color ${name} not found`);
    error.name = "ValueError";
    throw error;
  }

  get internal_space(): ColorArray {
    return this.internal;
  }

  // NOTE: python exposes this as `_internal_space`; keep both spellings.
  get _internal_space(): ColorArray {
    return this.internal_space;
  }

  to_integer(): number {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    const raw = new Uint8Array([
      Math.trunc(values[0] * 255) & 0xff,
      Math.trunc(values[1] * 255) & 0xff,
      Math.trunc(values[2] * 255) & 0xff,
    ]);
    return raw[0] * 65536 + raw[1] * 256 + raw[2];
  }

  toInteger(): number {
    return this.to_integer();
  }

  to_rgb(): ColorArray {
    return array((pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1]).slice(0, 3)) as ColorArray;
  }

  toRgb(): ColorArray {
    return this.to_rgb();
  }

  to_int_rgb(): ColorArray {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    return array(values.slice(0, 3).map((item) => Math.trunc(item * 255))) as ColorArray;
  }

  toIntRgb(): ColorArray {
    return this.to_int_rgb();
  }

  to_rgba(): ColorArray {
    return this._internal_value.copy() as ColorArray;
  }

  toRgba(): ColorArray {
    return this.to_rgba();
  }

  to_int_rgba(): ColorArray {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    return array(values.map((item) => Math.trunc(item * 255))) as ColorArray;
  }

  toIntRgba(): ColorArray {
    return this.to_int_rgba();
  }

  to_rgba_with_alpha(alpha: number): ColorArray {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    return array([values[0], values[1], values[2], alpha]) as ColorArray;
  }

  toRgbaWithAlpha(alpha: number): ColorArray {
    return this.to_rgba_with_alpha(alpha);
  }

  to_int_rgba_with_alpha(alpha: number): ColorArray {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    const scaled = values.map((item) => item * 255);
    scaled[3] = alpha * 255;
    return array(scaled.map((item) => Math.trunc(item))) as ColorArray;
  }

  toIntRgbaWithAlpha(alpha: number): ColorArray {
    return this.to_int_rgba_with_alpha(alpha);
  }

  to_hex(withAlpha = false): string {
    const values = pointRows(this._internal_value, 4)[0] ?? [0, 0, 0, 1];
    let result = `#${[values[0], values[1], values[2]].map((item) => Math.trunc((item as number) * 255).toString(16).padStart(2, "0")).join("")}`;
    if (withAlpha) result += Math.trunc(((values[3] ?? 1) as number) * 255).toString(16).padStart(2, "0");
    return result.toUpperCase();
  }

  toHex(withAlpha = false): string {
    return this.to_hex(withAlpha);
  }

  to_hsv(): ColorArray {
    const rgb = pointRows(this.to_rgb(), 3)[0] ?? [0, 0, 0];
    return array(rgbToHsv(rgb[0] ?? 0, rgb[1] ?? 0, rgb[2] ?? 0)) as ColorArray;
  }

  toHsv(): ColorArray {
    return this.to_hsv();
  }

  to_hsl(): ColorArray {
    const rgb = pointRows(this.to_rgb(), 3)[0] ?? [0, 0, 0];
    const hls = rgbToHls(rgb[0] ?? 0, rgb[1] ?? 0, rgb[2] ?? 0);
    return array([hls[0], hls[2], hls[1]]) as ColorArray;
  }

  toHsl(): ColorArray {
    return this.to_hsl();
  }

  invert(withAlpha = false): ManimColor {
    const space = [...spaceOf(this)];
    if (withAlpha) {
      const result = space.map((item) => 1 - item);
      return constructFromSpace(this, result);
    }
    const alpha = space[3] ?? 1;
    const result = space.map((item) => 1 - item);
    result[result.length - 1] = alpha;
    return constructFromSpace(this, result);
  }

  interpolate(other: ManimColor, alpha: number): ManimColor {
    const left = spaceOf(this);
    const right = spaceOf(other);
    const result = left.map((item, index) => item * (1 - alpha) + (right[index] ?? 0) * alpha);
    return constructFromSpace(this, result);
  }

  darker(blend = 0.2): ManimColor {
    const alpha = (spaceOf(this)[3] ?? 1) as number;
    const black = (this.constructor as typeof ManimColor)._from_internal(BLACK._internal_value);
    return this.interpolate(black, blend).opacity(alpha);
  }

  lighter(blend = 0.2): ManimColor {
    const alpha = (spaceOf(this)[3] ?? 1) as number;
    const white = (this.constructor as typeof ManimColor)._from_internal(WHITE._internal_value);
    return this.interpolate(white, blend).opacity(alpha);
  }

  contrasting(threshold = 0.5, light: ManimColor | null = null, dark: ManimColor | null = null): ManimColor {
    const rgb = pointRows(this.to_rgb(), 3)[0] ?? [0, 0, 0];
    const [luminance] = rgbToYiq(rgb[0] ?? 0, rgb[1] ?? 0, rgb[2] ?? 0);
    const ctor = this.constructor as typeof ManimColor;
    if ((luminance ?? 0) < threshold) {
      if (light !== null && light !== undefined) return light;
      return ctor._from_internal(WHITE._internal_value);
    }
    if (dark !== null && dark !== undefined) return dark;
    return ctor._from_internal(BLACK._internal_value);
  }

  opacity(opacity: number): ManimColor {
    const tmp = [...spaceOf(this)];
    tmp[tmp.length - 1] = opacity;
    return constructFromSpace(this, tmp);
  }

  into<T extends ManimColor>(classType: { _from_internal: (value: ColorArray) => T }): T {
    return classType._from_internal(this._internal_value);
  }

  equals(other: ManimColor): boolean {
    try {
      return this.__eq__(other);
    } catch {
      return false;
    }
  }

  __repr__(): string {
    return `${(this.constructor as typeof ManimColor).name}('${this.to_hex()}')`;
  }

  __str__(): string {
    return `${this.to_hex()}`;
  }

  toString(): string {
    return this.__str__();
  }

  toRepr(): string {
    return this.__repr__();
  }

  __eq__(other: object): boolean {
    if (!(other instanceof ManimColor)) {
      throw typeError(
        `Cannot compare ${(this.constructor as typeof ManimColor).name} with ${(other as object)?.constructor?.name ?? typeof other}`,
      );
    }
    const left = pointRows(this._internal_value, 4)[0] ?? [];
    const right = pointRows((other as ManimColor)._internal_value, 4)[0] ?? [];
    return left.length === right.length && left.every((item, index) => Math.abs(item - (right[index] ?? 0)) <= 1e-8 + 1e-5 * Math.abs(right[index] ?? 0));
  }

  __add__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => item + other));
    return constructFromSpace(this, spaceOf(this).map((item, index) => item + (spaceOf(other)[index] ?? 0)));
  }

  __radd__(other: number | ManimColor): ManimColor {
    return this.__add__(other);
  }

  __sub__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => item - other));
    return constructFromSpace(this, spaceOf(this).map((item, index) => item - (spaceOf(other)[index] ?? 0)));
  }

  __rsub__(other: number | ManimColor): ManimColor {
    return this.__sub__(other);
  }

  __mul__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => item * other));
    return constructFromSpace(this, spaceOf(this).map((item, index) => item * (spaceOf(other)[index] ?? 0)));
  }

  __rmul__(other: number | ManimColor): ManimColor {
    return this.__mul__(other);
  }

  __truediv__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => item / other));
    return constructFromSpace(this, spaceOf(this).map((item, index) => item / (spaceOf(other)[index] ?? 0)));
  }

  __rtruediv__(other: number | ManimColor): ManimColor {
    return this.__truediv__(other);
  }

  __floordiv__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => Math.floor(item / other)));
    return constructFromSpace(this, spaceOf(this).map((item, index) => Math.floor(item / (spaceOf(other)[index] ?? 0))));
  }

  __rfloordiv__(other: number | ManimColor): ManimColor {
    return this.__floordiv__(other);
  }

  __mod__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => pythonMod(item, other)));
    return constructFromSpace(this, spaceOf(this).map((item, index) => pythonMod(item, spaceOf(other)[index] ?? 0)));
  }

  __rmod__(other: number | ManimColor): ManimColor {
    return this.__mod__(other);
  }

  __pow__(other: number | ManimColor): ManimColor {
    if (typeof other === "number") return constructFromSpace(this, spaceOf(this).map((item) => item ** other));
    return constructFromSpace(this, spaceOf(this).map((item, index) => item ** (spaceOf(other)[index] ?? 0)));
  }

  __rpow__(other: number | ManimColor): ManimColor {
    return this.__pow__(other);
  }

  __invert__(): ManimColor {
    return this.invert();
  }

  __int__(): number {
    return this.to_integer();
  }

  __getitem__(index: number): number {
    return spaceOf(this)[index] as number;
  }

  __and__(other: ManimColor): ManimColor {
    const result = ManimColor._internal_from_integer(this.to_integer() & other.to_integer(), 1);
    return constructFromSpace(this, [...pointRows(result, 4)[0] ?? [0, 0, 0, 1]]);
  }

  __or__(other: ManimColor): ManimColor {
    const result = ManimColor._internal_from_integer(this.to_integer() | other.to_integer(), 1);
    return constructFromSpace(this, [...pointRows(result, 4)[0] ?? [0, 0, 0, 1]]);
  }

  __xor__(other: ManimColor): ManimColor {
    const result = ManimColor._internal_from_integer(this.to_integer() ^ other.to_integer(), 1);
    return constructFromSpace(this, [...pointRows(result, 4)[0] ?? [0, 0, 0, 1]]);
  }

  __hash__(): number {
    const text = this.to_hex(true);
    let hash = 0;
    for (let index = 0; index < text.length; index += 1) {
      hash = (Math.imul(31, hash) + text.charCodeAt(index)) | 0;
    }
    return hash;
  }
}

function spaceOf(color: ManimColor): number[] {
  return [...(pointRows((color as unknown as { _internal_space: ColorArray })._internal_space, 4)[0] ?? [0, 0, 0, 1])];
}

function constructFromSpace(owner: ManimColor, space: number[]): ManimColor {
  const ctor = owner.constructor as typeof ManimColor;
  return ctor._construct_from_space(space as unknown as ColorArray);
}

function pythonMod(a: number, b: number): number {
  if (b === 0) return Number.NaN;
  return a - Math.floor(a / b) * b;
}

export function is_sequence(
  color: ParsableManimColor | readonly ParsableManimColor[] | null | undefined,
): boolean {
  return Array.isArray(color);
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  const v = maxc;
  if (minc === maxc) return [0.0, 0.0, v];
  const s = (maxc - minc) / maxc;
  const rc = (maxc - r) / (maxc - minc);
  const gc = (maxc - g) / (maxc - minc);
  const bc = (maxc - b) / (maxc - minc);
  let h: number;
  if (r === maxc) h = bc - gc;
  else if (g === maxc) h = 2.0 + rc - bc;
  else h = 4.0 + gc - rc;
  h = (((h / 6.0) % 1.0) + 1.0) % 1.0;
  return [h, s, v];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  if (s === 0.0) return [v, v, v];
  const i = Math.trunc(h * 6.0);
  const f = h * 6.0 - i;
  const p = v * (1.0 - s);
  const q = v * (1.0 - s * f);
  const t = v * (1.0 - s * (1.0 - f));
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function rgbToHls(r: number, g: number, b: number): [number, number, number] {
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  const l = (minc + maxc) / 2.0;
  if (minc === maxc) return [0.0, l, 0.0];
  let s: number;
  if (l <= 0.5) s = (maxc - minc) / (maxc + minc);
  else s = (maxc - minc) / (2.0 - maxc - minc);
  const rc = (maxc - r) / (maxc - minc);
  const gc = (maxc - g) / (maxc - minc);
  const bc = (maxc - b) / (maxc - minc);
  let h: number;
  if (r === maxc) h = bc - gc;
  else if (g === maxc) h = 2.0 + rc - bc;
  else h = 4.0 + gc - rc;
  h = (((h / 6.0) % 1.0) + 1.0) % 1.0;
  return [h, l, s];
}

function hlsToRgb(h: number, l: number, s: number): [number, number, number] {
  if (s === 0.0) return [l, l, l];
  let m2: number;
  if (l <= 0.5) m2 = l * (1.0 + s);
  else m2 = l + s - l * s;
  const m1 = 2.0 * l - m2;
  return [vFunc(m1, m2, h + 1.0 / 3.0), vFunc(m1, m2, h), vFunc(m1, m2, h - 1.0 / 3.0)];
}

function vFunc(m1: number, m2: number, h: number): number {
  const hue = ((h % 1.0) + 1.0) % 1.0;
  if (hue < 1.0 / 6.0) return m1 + (m2 - m1) * hue * 6.0;
  if (hue < 0.5) return m2;
  if (hue < 2.0 / 3.0) return m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0;
  return m1;
}

function rgbToYiq(r: number, g: number, b: number): [number, number, number] {
  const y = 0.3 * r + 0.59 * g + 0.11 * b;
  const i = 0.74 * (r - y) - 0.27 * (b - y);
  const q = 0.48 * (r - y) + 0.41 * (b - y);
  return [y, i, q];
}

export class HSV extends ManimColor {
  private hsvValues: ColorArray;
  private alphaValue: number;

  constructor(hsv: number[] | readonly number[] | ColorArray, alpha = 1) {
    super(null);
    const values = valuesOf(hsv);
    if (values.length === 3) {
      this.hsvValues = array([...values, alpha]) as ColorArray;
    } else if (values.length === 4) {
      this.hsvValues = array([...values]) as ColorArray;
    } else {
      const error = new Error("HSV Color must be an array of 3 values");
      error.name = "ValueError";
      throw error;
    }
    // NOTE: mirrors Python, where super().__init__(None) always leaves the
    // alpha at its default of 1.0: neither the 4th input element nor the
    // `alpha` parameter affects the converted RGBA alpha.
    this.alphaValue = 1;
    this.syncBaseInternal();
  }

  static _from_internal(value: ColorArray): HSV {
    const rows = pointRows(value, 4)[0] ?? [0, 0, 0, 1];
    const hsv = rgbToHsv(rows[0] ?? 0, rows[1] ?? 0, rows[2] ?? 0);
    return new HSV([...hsv, rows[3] ?? 1]);
  }

  private syncBaseInternal(): void {
    const rows = pointRows(this.hsvValues, 4)[0] ?? [0, 0, 0, 1];
    const rgb = hsvToRgb(rows[0] ?? 0, rows[1] ?? 0, rows[2] ?? 0);
    this.internal = array([rgb[0], rgb[1], rgb[2], this.alphaValue]) as ColorArray;
  }

  get hue(): number {
    return (pointRows(this.hsvValues, 4)[0] ?? [])[0] ?? 0;
  }

  set hue(hue: number) {
    const rows = pointRows(this.hsvValues, 4)[0] ?? [0, 0, 0, 1];
    rows[0] = hue;
    this.hsvValues = array(rows) as ColorArray;
    this.syncBaseInternal();
  }

  get saturation(): number {
    return (pointRows(this.hsvValues, 4)[0] ?? [])[1] ?? 0;
  }

  set saturation(saturation: number) {
    const rows = pointRows(this.hsvValues, 4)[0] ?? [0, 0, 0, 1];
    rows[1] = saturation;
    this.hsvValues = array(rows) as ColorArray;
    this.syncBaseInternal();
  }

  get value(): number {
    return (pointRows(this.hsvValues, 4)[0] ?? [])[2] ?? 0;
  }

  set value(value: number) {
    const rows = pointRows(this.hsvValues, 4)[0] ?? [0, 0, 0, 1];
    rows[2] = value;
    this.hsvValues = array(rows) as ColorArray;
    this.syncBaseInternal();
  }

  get h(): number {
    return this.hue;
  }

  set h(hue: number) {
    this.hue = hue;
  }

  get s(): number {
    return this.saturation;
  }

  set s(saturation: number) {
    this.saturation = saturation;
  }

  get v(): number {
    return this.value;
  }

  set v(value: number) {
    this.value = value;
  }

  override get internal_space(): ColorArray {
    return this.hsvValues;
  }

  override get _internal_space(): ColorArray {
    return this.internal_space;
  }

  override get _internal_value(): ColorArray {
    const rows = pointRows(this.hsvValues, 4)[0] ?? [0, 0, 0, 1];
    const rgb = hsvToRgb(rows[0] ?? 0, rows[1] ?? 0, rows[2] ?? 0);
    return array([rgb[0], rgb[1], rgb[2], this.alphaValue]) as ColorArray;
  }

  override set _internal_value(value: ColorArray) {
    if (!isNDArray(value)) {
      throw new TypeError("Value must be a NumPy array.");
    }
    if ((value as NDArray).shape[0] !== 4) {
      throw new TypeError("Array must have exactly 4 values.");
    }
    const rows = pointRows(value, 4)[0] ?? [0, 0, 0, 1];
    const hsv = rgbToHsv(rows[0] ?? 0, rows[1] ?? 0, rows[2] ?? 0);
    this.hsvValues = array(hsv) as ColorArray;
    this.alphaValue = rows[3] ?? 1;
    this.syncBaseInternal();
  }
}

const PALETTE_ORDER = [
  "WHITE", "GRAY_A", "GREY_A", "GRAY_B", "GREY_B", "GRAY_C", "GREY_C",
  "GRAY_D", "GREY_D", "GRAY_E", "GREY_E", "BLACK", "LIGHTER_GRAY",
  "LIGHTER_GREY", "LIGHT_GRAY", "LIGHT_GREY", "GRAY", "GREY", "DARK_GRAY",
  "DARK_GREY", "DARKER_GRAY", "DARKER_GREY", "PURE_RED", "PURE_GREEN",
  "PURE_BLUE", "PURE_CYAN", "PURE_MAGENTA", "PURE_YELLOW", "BLUE_A", "BLUE_B",
  "BLUE_C", "BLUE_D", "BLUE_E", "BLUE", "DARK_BLUE", "TEAL_A", "TEAL_B",
  "TEAL_C", "TEAL_D", "TEAL_E", "TEAL", "GREEN_A", "GREEN_B", "GREEN_C",
  "GREEN_D", "GREEN_E", "GREEN", "YELLOW_A", "YELLOW_B", "YELLOW_C",
  "YELLOW_D", "YELLOW_E", "YELLOW", "GOLD_A", "GOLD_B", "GOLD_C", "GOLD_D",
  "GOLD_E", "GOLD", "RED_A", "RED_B", "RED_C", "RED_D", "RED_E", "RED",
  "MAROON_A", "MAROON_B", "MAROON_C", "MAROON_D", "MAROON_E", "MAROON",
  "PURPLE_A", "PURPLE_B", "PURPLE_C", "PURPLE_D", "PURPLE_E", "PURPLE",
  "PINK", "LIGHT_PINK", "ORANGE", "LIGHT_BROWN", "DARK_BROWN", "GRAY_BROWN",
  "GREY_BROWN", "LOGO_WHITE", "LOGO_GREEN", "LOGO_BLUE", "LOGO_RED",
  "LOGO_BLACK",
];

class MersenneTwister {
  private static readonly N = 624;
  private static readonly M = 397;
  private static readonly MATRIX_A = 0x9908b0df;
  private static readonly UPPER_MASK = 0x80000000;
  private static readonly LOWER_MASK = 0x7fffffff;
  private readonly mt: number[] = new Array<number>(MersenneTwister.N).fill(0);
  private index = MersenneTwister.N;

  constructor(seed: number) {
    this.seedInt(seed);
  }

  private initGenrand(s: number): void {
    this.mt[0] = s >>> 0;
    for (let i = 1; i < MersenneTwister.N; i += 1) {
      const prev = this.mt[i - 1] ?? 0;
      this.mt[i] = (Math.imul(1812433253, prev ^ (prev >>> 30)) + i) >>> 0;
    }
    this.index = MersenneTwister.N;
  }

  private initByArray(key: number[]): void {
    this.initGenrand(19650218);
    const keyLength = key.length;
    let i = 1;
    let j = 0;
    let k = Math.max(MersenneTwister.N, keyLength);
    for (; k > 0; k -= 1) {
      const prev = this.mt[i - 1] ?? 0;
      this.mt[i] = (((this.mt[i] ?? 0) ^ Math.imul(prev ^ (prev >>> 30), 1664525)) + (keyLength > 0 ? (key[j] ?? 0) : 0) + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= MersenneTwister.N) {
        this.mt[0] = this.mt[MersenneTwister.N - 1] ?? 0;
        i = 1;
      }
      if (keyLength > 0 && j >= keyLength) j = 0;
    }
    for (let t = MersenneTwister.N - 1; t > 0; t -= 1) {
      const prev = this.mt[i - 1] ?? 0;
      this.mt[i] = (((this.mt[i] ?? 0) ^ Math.imul(prev ^ (prev >>> 30), 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= MersenneTwister.N) {
        this.mt[0] = this.mt[MersenneTwister.N - 1] ?? 0;
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  private seedInt(seed: number): void {
    // Mirrors CPython's Random.seed(int): little-endian 32-bit limbs of the
    // absolute value through init_by_array (seed 0 maps to key [0]).
    // NOTE: float/str seeds and >32-bit masking nuances of CPython are out of
    // scope; non-integer seeds are truncated.
    let v = BigInt(Number.isFinite(seed) ? Math.trunc(Math.abs(seed)) : 0);
    const limbs: number[] = [];
    while (v > 0n) {
      limbs.push(Number(v & 0xffffffffn));
      v >>= 32n;
    }
    this.initByArray(limbs.length > 0 ? limbs : [0]);
  }

  private int32(): number {
    if (this.index >= MersenneTwister.N) {
      for (let i = 0; i < MersenneTwister.N; i += 1) {
        const y = (((this.mt[i] ?? 0) & MersenneTwister.UPPER_MASK) | ((this.mt[(i + 1) % MersenneTwister.N] ?? 0) & MersenneTwister.LOWER_MASK)) >>> 0;
        this.mt[i] = ((this.mt[(i + MersenneTwister.M) % MersenneTwister.N] ?? 0) ^ (y >>> 1) ^ ((y & 1) !== 0 ? MersenneTwister.MATRIX_A : 0)) >>> 0;
      }
      this.index = 0;
    }
    let y = this.mt[this.index] ?? 0;
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  getrandbits(k: number): number {
    if (k <= 0) return 0;
    const words = Math.floor((k - 1) / 32) + 1;
    let r = 0;
    for (let i = 0; i < words; i += 1) r = r * 2 ** 32 + this.int32();
    return Math.floor(r / 2 ** (32 * words - k));
  }

  randbelow(n: number): number {
    const k = n.toString(2).length;
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }
}

export class RandomColorGenerator {
  private static singleton: RandomColorGenerator | null = null;
  private readonly twister: MersenneTwister | null;
  readonly colors: ManimColor[];

  constructor(seed: number | null = null, sample_colors: ManimColor[] | null = null) {
    this.twister = seed === null || seed === undefined ? null : new MersenneTwister(seed);
    this.colors = sample_colors ?? PALETTE_ORDER.map((name) => new ManimColor(name));
  }

  next(): ManimColor {
    const index = this.twister !== null
      ? this.twister.randbelow(this.colors.length)
      : Math.floor(Math.random() * this.colors.length);
    return (this.colors[index] ?? this.colors[0]) as ManimColor;
  }

  static _random_color(): ManimColor {
    if (RandomColorGenerator.singleton === null) {
      RandomColorGenerator.singleton = new RandomColorGenerator();
    }
    return RandomColorGenerator.singleton.next();
  }
}

export const BLACK = new ManimColor("BLACK");
export const WHITE = new ManimColor("WHITE");
export const RED = new ManimColor("RED");
export const GREEN = new ManimColor("GREEN");
export const BLUE = new ManimColor("BLUE");
export const YELLOW = new ManimColor("YELLOW");
export const ORANGE = new ManimColor("ORANGE");
export const PURPLE = new ManimColor("PURPLE");
export const GREY = new ManimColor("GREY");
export const GRAY = new ManimColor("GRAY");
export const BLUE_E = new ManimColor("BLUE_E");
export const BLUE_D = new ManimColor("BLUE_D");
export const BLUE_C = new ManimColor("BLUE_C");
export const GREEN_E = new ManimColor("GREEN_E");
export const MAROON_B = new ManimColor("MAROON_B");
export const PINK = new ManimColor("PINK");
export const DARK_GREY = new ManimColor("DARK_GREY");
export const DARK_GRAY = new ManimColor("DARK_GRAY");
export const LIGHT_GREY = new ManimColor("LIGHT_GREY");
export const LIGHT_GRAY = new ManimColor("LIGHT_GRAY");
export const YELLOW_C = new ManimColor("YELLOW_C");

export function interpolate_color(color1: ManimColor, color2: ManimColor, alpha: number): ManimColor {
  return color1.interpolate(color2, alpha);
}

export function color_to_rgb(color: ParsableManimColor): ColorArray {
  return new ManimColor(color).toRgb();
}

export function color_to_rgba(color: ParsableManimColor, alpha = 1): ColorArray {
  return new ManimColor(color).to_rgba_with_alpha(alpha);
}

export function color_to_int_rgb(color: ParsableManimColor): ColorArray {
  return new ManimColor(color).to_int_rgb();
}

export function rgb_to_color(rgb: number[] | readonly number[] | NDArray): ManimColor {
  // NOTE: mirrors Python `ManimColor.from_rgb(rgb)` exactly: the input is
  // passed through untouched so NDArray dtype decides the int-vs-float path
  // (float64 [1,0,0] is red, not near-black). Do NOT round-trip through
  // pointRows here -- that would stringify dtypes into plain JS integers
  // and misroute float arrays into the int (1/255) path.
  return ManimColor.from_rgb(rgb as number[] | readonly number[] | ColorArray);
}

export function rgba_to_color(rgba: number[] | readonly number[] | NDArray): ManimColor {
  return ManimColor.from_rgba(rgba as number[] | readonly number[] | ColorArray);
}

export function rgb_to_hex(rgb: number[] | readonly number[] | NDArray): string {
  // NOTE: mirrors Python `ManimColor.from_rgb(rgb).to_hex()`; see the
  // rgb_to_color note about preserving NDArray dtype.
  return ManimColor.from_rgb(rgb as number[] | readonly number[] | ColorArray).to_hex();
}

function uint8Trunc(value: number): number {
  // Mirrors numpy `.astype(int)` truncation toward zero used by
  // ManimColor.to_int_rgba_with_alpha: no modulo-256 wrapping.
  return Math.trunc(value);
}

export function color_to_int_rgba(color: ParsableManimColor, alpha = 1): number[] {
  const values = pointRows(new ManimColor(color).toRgba(), 4)[0] ?? [0, 0, 0, 1];
  return [values[0], values[1], values[2], alpha].map((item) => uint8Trunc(item * 255));
}

export function hex_to_rgb(hex_code: string): ColorArray {
  return new ManimColor(hex_code).to_rgb();
}

export function color_gradient(referenceColors: readonly ParsableManimColor[], lengthOfOutput: number): ManimColor[] {
  if (lengthOfOutput === 0) return [];
  const colors = referenceColors.map((color) => new ManimColor(color));
  if (colors.length === 0) throw new Error("Expected 1 or more reference colors. Got 0 colors.");
  if (colors.length === 1) return Array.from({ length: lengthOfOutput }, () => colors[0]);
  const rgbs = colors.map((color) => pointRows(color.toRgb(), 3)[0] ?? [0, 0, 0]);
  const result: ManimColor[] = [];
  for (let index = 0; index < lengthOfOutput; index += 1) {
    const alpha = lengthOfOutput === 1 ? 0 : index / (lengthOfOutput - 1);
    const scaled = alpha * (colors.length - 1);
    // Mirrors Python: floors[-1] = num_colors - 2 and alphas_mod1[-1] = 1, so
    // the final output is always the last reference color (this also governs
    // the lengthOfOutput === 1 case with more than 2 reference colors).
    const lower = index === lengthOfOutput - 1 ? colors.length - 2 : Math.min(colors.length - 2, Math.floor(scaled));
    const fraction = index === lengthOfOutput - 1 ? 1 : scaled - lower;
    result.push(new ManimColor(array(rgbs[lower].map((value, channel) => value * (1 - fraction) + rgbs[lower + 1][channel] * fraction)) as ColorArray));
  }
  return result;
}

export function invert_color(color: ManimColor): ManimColor {
  return color.invert();
}

export function average_color(...colors: ParsableManimColor[]): ManimColor {
  const rgbs = colors.map((color) => pointRows(color_to_rgb(color), 3)[0] ?? [0, 0, 0]);
  const mean_rgb = [0, 1, 2].map(
    (channel) => rgbs.reduce((sum, rgb) => sum + (rgb[channel] ?? 0), 0) / rgbs.length,
  );
  // NOTE: Python averages float64 numpy arrays, so from_rgb takes the float
  // path. A plain JS array would lose that dtype (integral floats like 0.0
  // arrive as integers and route to the int 1/255 path), so wrap explicitly.
  return rgb_to_color(array(mean_rgb));
}

export function random_color(): ManimColor {
  return RandomColorGenerator._random_color();
}

export function random_bright_color(): ManimColor {
  const curr_rgb = pointRows(color_to_rgb(random_color()), 3)[0] ?? [0, 0, 0];
  const new_rgb = curr_rgb.map((value) => 0.5 * (value + 1));
  // NOTE: see average_color: Python's `0.5 * (rgb + ones(3))` is a float64
  // numpy array, so wrap to keep from_rgb on the float path.
  return rgb_to_color(array(new_rgb));
}

export function get_shaded_rgb(
  rgb: number[] | readonly number[] | ColorArray,
  point: number[] | readonly number[] | ColorArray,
  unit_normal_vect: number[] | readonly number[] | ColorArray,
  light_source: number[] | readonly number[] | ColorArray,
): number[] {
  const rgbRows = pointRows(rgb, 3)[0] ?? [0, 0, 0];
  const pointRowsValue = pointRows(point, 3)[0] ?? [0, 0, 0];
  const normalRows = pointRows(unit_normal_vect, 3)[0] ?? [0, 0, 0];
  const lightRows = pointRows(light_source, 3)[0] ?? [0, 0, 0];
  const diff = lightRows.map((value, index) => value - (pointRowsValue[index] ?? 0));
  const norm = Math.hypot(...diff);
  const to_sun = norm > 0 ? diff.map((value) => value / norm) : diff.map(() => 0);
  const dot = normalRows.reduce((sum, value, index) => sum + value * (to_sun[index] ?? 0), 0);
  let light = 0.5 * dot ** 3;
  if (light < 0) light *= 0.5;
  return rgbRows.map((value) => (value ?? 0) + light);
}
