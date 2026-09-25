import { array, NDArray } from "numpy-ts";
import { isNDArray, nestedRows, pointRows } from "../arrays.js";

export type ColorArray = NDArray<"float64">;
export type ParsableManimColor = ManimColor | string | number | number[] | readonly number[] | NDArray;

const NAMED_COLORS: Record<string, [number, number, number]> = {
  BLACK: [0, 0, 0],
  WHITE: [1, 1, 1],
  RED: [0.9882352941176471, 0.3843137254901961, 0.3333333333333333],
  GREEN: [0.5137254901960784, 0.7568627450980392, 0.403921568627451],
  BLUE: [0.3450980392156863, 0.7686274509803922, 0.8666666666666667],
  YELLOW: [0.9686274509803922, 0.8509803921568627, 0.43529411764705883],
  ORANGE: [1, 0.6470588235294118, 0],
  PURPLE: [0.5019607843137255, 0, 0.5019607843137255],
  GREY: [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
  GRAY: [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
  BLUE_E: [0.13725490196078433, 0.4196078431372549, 0.5568627450980392],
  BLUE_D: [0.10196078431372549, 0.35294117647058826, 0.6431372549019608],
  BLUE_C: [0.09803921568627451, 0.37254901960784315, 0.803921568627451],
  TEAL: [0, 0.5019607843137255, 0.5019607843137255],
  PINK: [1, 0.6470588235294118, 0.807843137254902],
  MAROON: [0.5019607843137255, 0, 0],
  GOLD: [1, 0.8431372549019608, 0],
  LIGHT_GRAY: [0.7529411764705882, 0.7529411764705882, 0.7529411764705882],
  DARK_GRAY: [0.6627450980392157, 0.6627450980392157, 0.6627450980392157],
  DARK_GREY: [0.6627450980392157, 0.6627450980392157, 0.6627450980392157],
  LIGHT_GREY: [0.8274509803921568, 0.8274509803921568, 0.8274509803921568],
  BROWN: [0.6470588235294118, 0.16470588235294117, 0.16470588235294117],
  VIOLET: [0.9333333333333333, 0.5098039215686274, 0.9333333333333333],
  INDIGO: [0.29411764705882354, 0, 0.5098039215686274],
  CYAN: [0, 1, 1],
  MAGENTA: [1, 0, 1],
  BLACK_E: [0, 0, 0],
  WHITE_E: [1, 1, 1],
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
      throw new TypeError(`Expected numpy array, got ${typeof value}`);
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
    throw new Error(`Color ${name} not found`);
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
export const BROWN = new ManimColor("BROWN");

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
