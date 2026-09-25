import { array, type NDArray } from "numpy-ts";
import { isNDArray, numeric, pointRows } from "../../utils/arrays.js";
import { ManimColor, YELLOW_C, color_to_int_rgb, type ParsableManimColor } from "../../utils/color/core.js";
import { change_to_rgba_array, get_full_raster_image_path } from "../../utils/images.js";
import { Mobject, type MobjectOptions } from "../mobject.js";

export interface AbstractImageMobjectOptions extends MobjectOptions {
  pixel_array_dtype?: string;
  scale_to_resolution: number | false;
  resampling_algorithm?: number;
}

export const RESAMPLE_BICUBIC = 3;

export const RESAMPLING_ALGORITHMS: Record<string, number> = {
  nearest: 0,
  none: 0,
  bilinear: 2,
  linear: 2,
  bicubic: 3,
  cubic: 3,
};

export class AbstractImageMobject extends Mobject {
  pixel_array_dtype: string;
  scale_to_resolution: number | false;
  resampling_algorithm: number;

  constructor(options: AbstractImageMobjectOptions) {
    const {
      pixel_array_dtype = "uint8",
      scale_to_resolution,
      resampling_algorithm = RESAMPLE_BICUBIC,
      ...rest
    } = options;
    super(rest);
    this.pixel_array_dtype = pixel_array_dtype;
    this.scale_to_resolution = scale_to_resolution;
    this.resampling_algorithm = resampling_algorithm;
  }

  get_pixel_array(): NDArray {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  set_color(
    _color?: ParsableManimColor,
    _alpha?: number | null,
    _family?: boolean,
  ): this {
    const error = new Error();
    error.name = "NotImplementedError";
    throw error;
  }

  set_resampling_algorithm(resampling_algorithm: number): this {
    if (!Object.values(RESAMPLING_ALGORITHMS).includes(resampling_algorithm)) {
      const error = new Error(
        "resampling_algorithm has to be an int, one of the values defined in " +
          "RESAMPLING_ALGORITHMS or a Pillow resampling filter constant. " +
          "Available algorithms: 'bicubic' (or 'cubic'), 'nearest' (or 'none'), " +
          "'bilinear' (or 'linear').",
      );
      error.name = "ValueError";
      throw error;
    }
    this.resampling_algorithm = resampling_algorithm;
    return this;
  }
}

export interface ImageMobjectOptions extends AbstractImageMobjectOptions {
  pixel_array?: NDArray | number[][][];
  /** Positional-equivalent: image file path or raw pixel array. */
  filename_or_array?: string | NDArray | number[][] | number[][][];
  invert?: boolean;
  image_mode?: string;
  opacity?: number;
}

export const HIGH_QUALITY_PIXEL_HEIGHT = 1080;

function isImageMobjectOptions(value: unknown): value is ImageMobjectOptions {
  if (value === null || typeof value !== "object" || isNDArray(value) || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return (
    "pixel_array" in record ||
    "filename_or_array" in record ||
    "scale_to_resolution" in record ||
    "invert" in record ||
    "image_mode" in record ||
    "opacity" in record ||
    "pixel_array_dtype" in record
  );
}

export class ImageMobject extends AbstractImageMobject {
  pixel_array: NDArray;
  orig_alpha_pixel_array: number[][];
  stroke_opacity: number;
  fill_opacity = 1;
  invert_image: boolean;
  image_mode: string;
  path?: string;

  constructor(
    filename_or_array?: string | NDArray | number[][] | number[][][] | ImageMobjectOptions,
    scale_to_resolution: number | false = HIGH_QUALITY_PIXEL_HEIGHT,
    invert = false,
    image_mode = "RGBA",
    options: ImageMobjectOptions = {} as ImageMobjectOptions,
  ) {
    let source: string | NDArray | number[][] | number[][][] | undefined;
    let scale: number | false = scale_to_resolution;
    let invertFlag = invert;
    let mode = image_mode;
    let rest: Record<string, unknown> = { ...options };
    if (isImageMobjectOptions(filename_or_array)) {
      const opts = filename_or_array as ImageMobjectOptions;
      const { pixel_array, filename_or_array: fileOrArray, invert: inv, image_mode: im, opacity = 1, scale_to_resolution: s, ...remaining } = opts;
      source = (pixel_array ?? fileOrArray) as string | NDArray | number[][] | number[][][] | undefined;
      if (s !== undefined) scale = s;
      if (inv !== undefined) invertFlag = inv;
      if (im !== undefined) mode = im;
      rest = { ...remaining, opacity };
    } else {
      source = filename_or_array as string | NDArray | number[][] | number[][][] | undefined;
    }
    const opacity = Number((rest as { opacity?: number }).opacity ?? 1);
    delete (rest as { opacity?: number }).opacity;
    const pixel_array_dtype = ((rest as { pixel_array_dtype?: string }).pixel_array_dtype ?? "uint8") as string;
    super({ ...rest, scale_to_resolution: scale, pixel_array_dtype });
    this.fill_opacity = 1;
    this.stroke_opacity = 1;
    this.invert_image = invertFlag;
    this.image_mode = mode;
    if (typeof source === "string") {
      const path = get_full_raster_image_path(source);
      this.path = path;
      throw new Error(
        `TODO: ImageMobject file loading not ported yet (resolved path: ${path}); pass a pixel array instead`,
      );
    }
    const raw = (source === undefined ? array([]).reshape(0, 0, 4) : isNDArray(source) ? source : array(source)) as NDArray;
    this.pixel_array_dtype = pixel_array_dtype;
    let converted = change_to_rgba_array(raw, this.pixel_array_dtype);
    if (this.invert_image) {
      const max = 255;
      const voxels = (converted.toArray() as unknown[][][]).map((row) =>
        row.map((pixel) => (pixel as unknown[]).map(numeric)),
      );
      for (const row of voxels) {
        for (const pixel of row) {
          pixel[0] = max - (pixel[0] ?? 0);
          pixel[1] = max - (pixel[1] ?? 0);
          pixel[2] = max - (pixel[2] ?? 0);
        }
      }
      converted = array(voxels) as NDArray;
    }
    this.pixel_array = converted;
    this.orig_alpha_pixel_array = alphaChannel(this.pixel_array);
    if (opacity !== 1) this.set_opacity(opacity);
  }

  get_pixel_array(): NDArray {
    return this.pixel_array;
  }

  set_color(
    color: ParsableManimColor = YELLOW_C,
    alpha: number | null = null,
    family = true,
  ): this {
    const rgb = pointRows(color_to_int_rgb(color), 3)[0] ?? [0, 0, 0];
    const voxels = (this.pixel_array.toArray() as unknown[][][]).map((row) =>
      row.map((pixel) => [...(pixel as unknown[])].map(numeric)),
    );
    for (const row of voxels) {
      for (const pixel of row) {
        pixel[0] = rgb[0] ?? 0;
        pixel[1] = rgb[1] ?? 0;
        pixel[2] = rgb[2] ?? 0;
        if (alpha !== null && alpha !== undefined) pixel[3] = Math.trunc(255 * alpha);
      }
    }
    this.pixel_array = array(voxels) as NDArray;
    for (const submob of this.submobjects) {
      (submob as ImageMobject).set_color(color, alpha, family);
    }
    this.color = new ManimColor(color);
    return this;
  }

  set_opacity(alpha: number): this {
    const voxels = (this.pixel_array.toArray() as unknown[][][]).map((row) =>
      row.map((pixel) => [...(pixel as unknown[])].map(numeric)),
    );
    for (let y = 0; y < voxels.length; y += 1) {
      for (let x = 0; x < (voxels[y]?.length ?? 0); x += 1) {
        const pixel = voxels[y]?.[x];
        if (pixel !== undefined && pixel.length > 3) {
          pixel[3] = (this.orig_alpha_pixel_array[y]?.[x] ?? 0) * alpha;
        }
      }
    }
    this.pixel_array = (array(voxels) as NDArray).astype(this.pixel_array_dtype as "uint8") as NDArray;
    this.stroke_opacity = alpha;
    return this;
  }

  fade(darkness = 0.5, family = true): this {
    this.set_opacity(1 - darkness);
    super.fade(darkness, family);
    return this;
  }

  get_style(): Record<string, unknown> {
    return {
      fill_color: new ManimColor(this.color.toRgb()).toHex(),
      fill_opacity: this.fill_opacity,
    };
  }
}

function alphaChannel(pixel_array: NDArray): number[][] {
  const shape = (pixel_array as NDArray).shape;
  if (shape.length < 3 || (shape[2] ?? 0) < 4) return [];
  const voxels = (pixel_array as NDArray).toArray() as unknown[][][];
  return voxels.map((row) => row.map((pixel) => numeric((pixel as unknown[])[3])));
}
