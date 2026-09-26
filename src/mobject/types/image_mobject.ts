import { array, type NDArray } from "numpy-ts";
import { DOWN, LEFT, RIGHT, UP } from "../../constants.js";
import { config } from "../../_config/utils.js";
import { addPoints, isNDArray, numeric, pointRows } from "../../utils/arrays.js";
import { interpolate } from "../../utils/bezier.js";
import { ManimColor, WHITE, YELLOW_C, color_to_int_rgb, type ParsableManimColor } from "../../utils/color/core.js";
import { change_to_rgba_array, get_full_raster_image_path } from "../../utils/images.js";
import { SurroundingRectangle } from "../geometry/shape_matchers.js";
import type { MovingCamera } from "../../camera/moving_camera.js";
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

  set_resampling_algorithm(resampling_algorithm: number): this {    if (!Object.values(RESAMPLING_ALGORITHMS).includes(resampling_algorithm)) {
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

  reset_points(): this {
    // NOTE: the base Mobject constructor calls reset_points() before this
    // class assigns pixel_array (TS requires super() first, whereas Python
    // assigns it beforehand). Treat that early call as a plain point reset;
    // ImageMobject's constructor re-runs reset_points() once ready.
    if (!("pixel_array" in this) || (this as unknown as { pixel_array?: unknown }).pixel_array === undefined) {
      return super.reset_points();
    }
    this.points = array([
      addPoints([...UP], [...LEFT], this.dim),
      addPoints([...UP], [...RIGHT], this.dim),
      addPoints([...DOWN], [...LEFT], this.dim),
      addPoints([...DOWN], [...RIGHT], this.dim),
    ]) as NDArray;
    this.center();
    const shape = this.get_pixel_array().shape;
    const h = shape[0] ?? 0;
    const w = shape[1] ?? 0;
    let height: number;
    if (this.scale_to_resolution) {
      height = (h / this.scale_to_resolution) * Number(config.frame_height);
    } else {
      height = 3; // this is the case for ImageMobjectFromCamera
    }
    this.stretch_to_fit_height(height);
    this.stretch_to_fit_width(height * (w / h));
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
    // NOTE: the base Mobject constructor already called reset_points(),
    // but pixel_array was not assigned yet (see reset_points guard above).
    // Regenerate now that geometry inputs are available.
    this.reset_points();
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

  interpolate_color(mobject1: Mobject, mobject2: Mobject, alpha: number): this {
    if (!(mobject1 instanceof ImageMobject) || !(mobject2 instanceof ImageMobject)) {
      const error = new Error("Both mobjects must be ImageMobjects");
      error.name = "AssertionError";
      throw error;
    }
    if (mobject1.pixel_array.shape.join(",") !== mobject2.pixel_array.shape.join(",")) {
      const error = new Error(
        `Mobject pixel array shapes incompatible for interpolation.\nMobject 1 (${mobject1}) : ${mobject1.pixel_array.shape}\nMobject 2 (${mobject2}) : ${mobject2.pixel_array.shape}`,
      );
      error.name = "AssertionError";
      throw error;
    }
    this.fill_opacity = interpolate(mobject1.fill_opacity, mobject2.fill_opacity, alpha) as number;
    this.stroke_opacity = interpolate(mobject1.stroke_opacity, mobject2.stroke_opacity, alpha) as number;
    // NOTE: bezier.interpolate() flattens arrays to rows, so the elementwise
    // lerp over the (h, w, 4) pixel arrays is done on nested lists instead to
    // preserve the shape exactly (mirrors numpy broadcasting in Python).
    const first = mobject1.pixel_array.toArray() as unknown;
    const second = mobject2.pixel_array.toArray() as unknown;
    const blended = lerpNested(first, second, alpha);
    this.pixel_array = (array(blended as number[][][]) as NDArray).astype(
      this.pixel_array_dtype as "uint8",
    ) as NDArray;
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

function lerpNested(first: unknown, second: unknown, alpha: number): unknown {
  if (Array.isArray(first) && Array.isArray(second)) {
    return first.map((value, index) => lerpNested(value, second[index], alpha));
  }
  return (1 - alpha) * Number(first) + alpha * Number(second);
}

export interface ImageMobjectFromCameraOptions extends Omit<AbstractImageMobjectOptions, "scale_to_resolution"> {
  default_display_frame_config?: Record<string, unknown> | null;
  scale_to_resolution?: number | false;
}

export class ImageMobjectFromCamera extends AbstractImageMobject {
  camera: MovingCamera;
  default_display_frame_config: Record<string, unknown>;
  pixel_array: NDArray;
  display_frame?: SurroundingRectangle;

  constructor(
    camera: MovingCamera,
    default_display_frame_config: Record<string, unknown> | null = null,
    options: ImageMobjectFromCameraOptions = {},
  ) {
    const { default_display_frame_config: cfg, scale_to_resolution: _ignored, ...rest } = options;
    void _ignored;
    super({ ...rest, scale_to_resolution: false });
    this.camera = camera;
    this.default_display_frame_config = default_display_frame_config ??
      cfg ?? {
        stroke_width: 3,
        stroke_color: WHITE,
        buff: 0,
      };
    this.pixel_array = this.camera.pixel_array;
    // NOTE: the base Mobject constructor already called reset_points(), but
    // pixel_array was not assigned yet (see the guard in
    // AbstractImageMobject.reset_points). Regenerate now that the camera
    // pixel array is available, mirroring Python which assigns pixel_array
    // before super().__init__().
    this.reset_points();
  }

  override get_pixel_array(): NDArray {
    this.pixel_array = this.camera.pixel_array;
    return this.pixel_array;
  }

  add_display_frame(options: Record<string, unknown> = {}): this {
    const config = { ...this.default_display_frame_config, ...options };
    this.display_frame = new SurroundingRectangle(this, config);
    this.add(this.display_frame);
    return this;
  }

  override interpolate_color(mobject1: Mobject, mobject2: Mobject, alpha: number): this {
    if (!(mobject1 instanceof ImageMobjectFromCamera) || !(mobject2 instanceof ImageMobjectFromCamera)) {
      const error = new Error("Both mobjects must be ImageMobjectFromCamera");
      error.name = "AssertionError";
      throw error;
    }
    if (mobject1.pixel_array.shape.join(",") !== mobject2.pixel_array.shape.join(",")) {
      const error = new Error(
        `Mobject pixel array shapes incompatible for interpolation.\nMobject 1 (${mobject1}) : ${mobject1.pixel_array.shape}\nMobject 2 (${mobject2}) : ${mobject2.pixel_array.shape}`,
      );
      error.name = "AssertionError";
      throw error;
    }
    // NOTE: bezier.interpolate() flattens arrays to rows, so the elementwise
    // lerp over the (h, w, 4) pixel arrays is done on nested lists instead to
    // preserve the shape exactly (mirrors numpy broadcasting in Python).
    const first = mobject1.pixel_array.toArray() as unknown;
    const second = mobject2.pixel_array.toArray() as unknown;
    const blended = lerpNested(first, second, alpha);
    this.pixel_array = (array(blended as number[][][]) as NDArray).astype(
      this.pixel_array_dtype as "uint8",
    ) as NDArray;
    return this;
  }
}
