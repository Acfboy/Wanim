import { array, NDArray } from "numpy-ts";
import { isNDArray, nestedRows, pointRows } from "../arrays.js";

export type ColorArray = NDArray<"float64">;
export type ParsableManimColor = ManimColor | string | number | number[] | readonly number[] | NDArray;

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

function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

function parseHex(value: string, alpha: number): ColorArray {
  const match = value.match(/(?:#|0x)([0-9a-f]{3,8})$/i);
  if (!match) throw new Error(`Color ${value} not found`);
  let hex = match[1];
  if (hex.length === 3 || hex.length === 4) hex = [...hex].map((part) => part + part).join("");
  if (hex.length === 6) hex += "FF";
  if (hex.length !== 8) throw new Error("Hex colors must contain 6 or 8 hexadecimal numbers");
  const number = Number.parseInt(hex, 16);
  const values = [(number >>> 24) & 255, (number >>> 16) & 255, (number >>> 8) & 255, number & 255].map((part) => part / 255);
  values[3] = alpha;
  return array(values) as ColorArray;
}

function valuesOf(value: unknown): number[] {
  if (isNDArray(value)) {
    const rows = nestedRows(value);
    return Array.isArray(rows[0]) ? (rows[0] as unknown[]).map(Number) : [Number(rows[0])];
  }
  if (Array.isArray(value)) return value.map(Number);
  return [Number(value)];
}

export class ManimColor {
  private readonly internal: ColorArray;

  constructor(value: ParsableManimColor | null | undefined = null, alpha = 1) {
    if (value === null || value === undefined) {
      this.internal = array([0, 0, 0, alpha]) as ColorArray;
    } else if (value instanceof ManimColor) {
      this.internal = value.internal.copy() as ColorArray;
    } else if (typeof value === "number") {
      const normalized = Math.trunc(value);
      this.internal = array([
        ((normalized >> 16) & 255) / 255,
        ((normalized >> 8) & 255) / 255,
        (normalized & 255) / 255,
        alpha,
      ]) as ColorArray;
    } else if (typeof value === "string") {
      const named = NAMED_COLORS[value.toUpperCase()];
      if (named) this.internal = array([...named, alpha]) as ColorArray;
      else this.internal = parseHex(value, alpha);
    } else {
      const values = valuesOf(value);
      if (values.length !== 3 && values.length !== 4) {
        throw new Error(`ManimColor only accepts lists/tuples/arrays of length 3 or 4, not ${values.length}`);
      }
      const allFloat = isNDArray(value) || values.every((item) => typeof item === "number" && !Number.isInteger(item));
      const normalized = allFloat ? values : values.map((item) => item / 255);
      this.internal = array([
        normalized[0] ?? 0,
        normalized[1] ?? 0,
        normalized[2] ?? 0,
        values.length === 4 ? normalized[3] : alpha,
      ]) as ColorArray;
    }
  }

  static parse(value: ParsableManimColor | readonly ParsableManimColor[] | null | undefined, alpha = 1): ManimColor | ManimColor[] {
    if (Array.isArray(value) && value.length > 0 && (typeof value[0] === "string" || (value[0] as unknown) instanceof ManimColor || Array.isArray(value[0]) || isNDArray(value[0]))) {
      return (value as readonly ParsableManimColor[]).map((item) => new ManimColor(item, alpha));
    }
    return new ManimColor(value as ParsableManimColor | null | undefined, alpha);
  }

  static from_rgb(rgb: number[] | readonly number[], alpha = 1): ManimColor {
    return new ManimColor(array([...rgb]) as ColorArray, alpha);
  }

  static from_rgba(rgba: number[] | readonly number[]): ManimColor {
    return new ManimColor(array([...rgba]) as ColorArray);
  }

  static from_hex(value: string, alpha = 1): ManimColor {
    return new ManimColor(value, alpha);
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

  toRgb(): ColorArray {
    return array((pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1]).slice(0, 3)) as ColorArray;
  }

  toRgba(): ColorArray {
    return this.internal.copy() as ColorArray;
  }

  toRgbaWithAlpha(alpha: number): ColorArray {
    const values = pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1];
    return array([values[0], values[1], values[2], alpha]) as ColorArray;
  }

  toHex(withAlpha = false): string {
    const values = pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1];
    let result = `#${[values[0], values[1], values[2]].map((item) => Math.trunc(item * 255).toString(16).padStart(2, "0")).join("")}`;
    if (withAlpha) result += Math.trunc((values[3] ?? 1) * 255).toString(16).padStart(2, "0");
    return result.toUpperCase();
  }

  toInteger(): number {
    const values = pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1];
    return (Math.trunc(values[0] * 255) << 16) | (Math.trunc(values[1] * 255) << 8) | Math.trunc(values[2] * 255);
  }

  interpolate(other: ManimColor, alpha: number): ManimColor {
    const left = pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1];
    const right = pointRows(other.internal, 4)[0] ?? [0, 0, 0, 1];
    return new ManimColor(array(left.map((value, index) => value * (1 - alpha) + (right[index] ?? 0) * alpha)) as ColorArray);
  }

  opacity(opacity: number): ManimColor {
    return new ManimColor(this.toRgbaWithAlpha(opacity));
  }

  invert(withAlpha = false): ManimColor {
    const values = pointRows(this.internal, 4)[0] ?? [0, 0, 0, 1];
    const result = values.map((value, index) => (index === 3 && !withAlpha ? values[3] : 1 - value));
    return new ManimColor(array(result) as ColorArray);
  }

  equals(other: ManimColor): boolean {
    const left = pointRows(this.internal, 4)[0] ?? [];
    const right = pointRows(other.internal, 4)[0] ?? [];
    return left.length === right.length && left.every((value, index) => Math.abs(value - (right[index] ?? 0)) < 1e-8);
  }

  toString(): string {
    return this.toHex();
  }

  toRepr(): string {
    return `ManimColor('${this.toHex()}')`;
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
export const PINK = new ManimColor("PINK");
export const DARK_GREY = new ManimColor("DARK_GREY");
export const DARK_GRAY = new ManimColor("DARK_GRAY");
export const LIGHT_GREY = new ManimColor("LIGHT_GREY");
export const LIGHT_GRAY = new ManimColor("LIGHT_GRAY");

export function interpolate_color(color1: ManimColor, color2: ManimColor, alpha: number): ManimColor {
  return color1.interpolate(color2, alpha);
}

export function color_to_rgb(color: ParsableManimColor): ColorArray {
  return new ManimColor(color).toRgb();
}

export function rgb_to_color(rgb: number[] | readonly number[] | NDArray): ManimColor {
  return ManimColor.from_rgb([...pointRows(rgb, 3)[0]]);
}

function uint8(value: number): number {
  const converted = Math.trunc(value);
  const wrapped = converted % 256;
  return wrapped < 0 ? wrapped + 256 : wrapped;
}

export function color_to_int_rgba(color: ParsableManimColor, alpha: number): number[] {
  const values = pointRows(new ManimColor(color).toRgba(), 4)[0] ?? [0, 0, 0, 1];
  return [values[0], values[1], values[2], alpha].map((item) => uint8(item * 255));
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
    const lower = Math.min(colors.length - 2, Math.floor(scaled));
    const fraction = index === lengthOfOutput - 1 ? 1 : scaled - lower;
    result.push(new ManimColor(array(rgbs[lower].map((value, channel) => value * (1 - fraction) + rgbs[lower + 1][channel] * fraction)) as ColorArray));
  }
  return result;
}

export function invert_color(color: ManimColor): ManimColor {
  return color.invert();
}

export function with_alpha(color: ManimColor, alpha: number): ManimColor {
  return color.opacity(clamp(alpha));
}
