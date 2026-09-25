import { array, copyto, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { ManimColor, color_to_int_rgba, type ParsableManimColor } from "../utils/color/core.js";
import { asArray, pointRows } from "../utils/arrays.js";
import { ORIGIN } from "../constants.js";

export interface CameraOptions {
  background_image?: string | null;
  frame_center?: readonly number[];
  image_mode?: string;
  n_channels?: number;
  pixel_array_dtype?: string;
  use_z_index?: boolean;
  background?: unknown;
  pixel_height?: number;
  pixel_width?: number;
  frame_height?: number;
  frame_width?: number;
  frame_rate?: number;
  background_color?: ParsableManimColor;
  background_opacity?: number;
  [key: string]: unknown;
}

export class Camera {
  background_image: string | null;
  frame_center: readonly number[];
  image_mode: string;
  n_channels: number;
  pixel_array_dtype: string;
  use_z_index: boolean;
  background!: NDArray;
  pixel_array!: NDArray;
  pixel_height: number;
  pixel_width: number;
  frame_height: number;
  frame_width: number;
  frame_rate: number;
  max_allowable_norm: number;
  protected _background_color: ManimColor;
  protected _background_opacity: number;

  constructor(options: CameraOptions = {}) {
    this.background_image = options.background_image ?? null;
    this.frame_center = options.frame_center ?? ORIGIN;
    this.image_mode = options.image_mode ?? "RGBA";
    this.n_channels = options.n_channels ?? 4;
    this.pixel_array_dtype = options.pixel_array_dtype ?? "uint8";
    this.use_z_index = options.use_z_index ?? true;
    this.pixel_height = options.pixel_height ?? config.pixel_height;
    this.pixel_width = options.pixel_width ?? config.pixel_width;
    this.frame_height = options.frame_height ?? config.frame_height;
    this.frame_width = options.frame_width ?? config.frame_width;
    this.frame_rate = options.frame_rate ?? config.frame_rate;
    this._background_color = new ManimColor(options.background_color ?? config.background_color);
    this._background_opacity = options.background_opacity ?? config.background_opacity;
    this.max_allowable_norm = this.frame_width;
    this.init_background();
    this.resize_frame_shape();
    this.reset();
  }

  get background_color(): ManimColor {
    return this._background_color;
  }

  set background_color(color: ParsableManimColor) {
    this._background_color = color instanceof ManimColor ? color : new ManimColor(color);
    this.init_background();
  }

  get background_opacity(): number {
    return this._background_opacity;
  }

  set background_opacity(alpha: number) {
    this._background_opacity = alpha;
    this.init_background();
  }

  init_background(): void {
    const rgba = color_to_int_rgba(this.background_color, this.background_opacity);
    const rows = Array.from({ length: this.pixel_height }, () =>
      Array.from({ length: this.pixel_width }, () => [...rgba]),
    );
    this.background = array(rows.length > 0 ? rows : []).reshape(this.pixel_height, this.pixel_width, this.n_channels).astype("uint8") as NDArray;
  }

  resize_frame_shape(fixedDimension = 0): void {
    const aspectRatio = this.pixel_width / this.pixel_height;
    if (fixedDimension === 0) this.frame_height = this.frame_width / aspectRatio;
    else this.frame_width = aspectRatio * this.frame_height;
  }

  reset(): this {
    this.pixel_array = this.background.copy();
    return this;
  }

  get rgb_max_val(): number {
    if (this.pixel_array_dtype.startsWith("uint")) {
      const bits = Number(this.pixel_array_dtype.slice(4)) || 8;
      return 2 ** bits - 1;
    }
    if (this.pixel_array_dtype.startsWith("int")) {
      const bits = Number(this.pixel_array_dtype.slice(3)) || 8;
      return 2 ** (bits - 1) - 1;
    }
    return 255;
  }

  convert_pixel_array(pixel_array: unknown): NDArray {
    const source = asArray(pixel_array).toArray() as unknown[][][];
    const converted = source.map((row) =>
      row.map((pixel) =>
        (Array.isArray(pixel) ? pixel : [pixel]).map((value) => Math.trunc(Number(value) * this.rgb_max_val)),
      ),
    );
    return (array(converted) as NDArray).astype(this.pixel_array_dtype as "uint8") as NDArray;
  }

  set_pixel_array(pixel_array: unknown, convert_from_floats = false): void {
    const converted = convert_from_floats ? this.convert_pixel_array(pixel_array) : asArray(pixel_array);
    if (
      this.pixel_array !== undefined &&
      this.pixel_array !== null &&
      shapesEqual((this.pixel_array as NDArray).shape, (converted as NDArray).shape)
    ) {
      copyto(this.pixel_array as never, converted as never);
    } else {
      this.pixel_array = (converted as NDArray).copy();
    }
  }

  on_screen_pixels(pixel_coords: NDArray | number[][]): boolean[] {
    const rows = pointRows(pixel_coords, 2);
    return rows.map((row) => {
      const x = row[0] ?? 0;
      const y = row[1] ?? 0;
      return x >= 0 && x < this.pixel_width && y >= 0 && y < this.pixel_height;
    });
  }

  adjusted_thickness(thickness: number): number {
    const big_sum: number = config.pixel_height + config.pixel_width;
    const this_sum: number = this.pixel_height + this.pixel_width;
    const factor = big_sum / this_sum;
    return 1 + (thickness - 1) * factor;
  }

  get_thickening_nudges(thickness: number): NDArray {
    const nudged = Math.trunc(thickness);
    const range: number[] = [];
    for (let index = Math.floor(-nudged / 2) + 1; index <= Math.floor(nudged / 2); index += 1) {
      range.push(index);
    }
    const pairs: number[][] = [];
    for (const first of range) for (const second of range) pairs.push([first, second]);
    return array(pairs.length > 0 ? pairs : []) as NDArray;
  }

  capture_mobject(_mobject: unknown): void {}

  get_image(pixelArray?: unknown): unknown { return pixelArray ?? this.pixel_array; }
}

function shapesEqual(first: readonly number[], second: readonly number[]): boolean {
  return first.length === second.length && first.every((value, index) => value === second[index]);
}
