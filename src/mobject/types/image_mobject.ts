import { array, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "../../utils/arrays.js";
import { Mobject, type MobjectOptions } from "../mobject.js";

export interface AbstractImageMobjectOptions extends MobjectOptions {
  pixel_array_dtype?: string;
  scale_to_resolution: number | false;
  resampling_algorithm?: number;
}

export const RESAMPLE_BICUBIC = 3;

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
}

export interface ImageMobjectOptions extends AbstractImageMobjectOptions {
  pixel_array?: NDArray | number[][][];
  opacity?: number;
}

export const HIGH_QUALITY_PIXEL_HEIGHT = 1080;

export class ImageMobject extends AbstractImageMobject {
  pixel_array: NDArray;
  orig_alpha_pixel_array: number[][];
  stroke_opacity: number;

  constructor(options: ImageMobjectOptions) {
    const { pixel_array, opacity = 1, scale_to_resolution = HIGH_QUALITY_PIXEL_HEIGHT, ...rest } = options;
    super({ ...rest, scale_to_resolution });
    const source = pixel_array === undefined ? array([]).reshape(0, 0, 4) as NDArray : pixel_array;
    this.pixel_array = (isNDArray(source) ? source : array(source)) as NDArray;
    this.orig_alpha_pixel_array = alphaChannel(this.pixel_array);
    this.stroke_opacity = 1;
    if (opacity !== 1) this.set_opacity(opacity);
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
}

function alphaChannel(pixel_array: NDArray): number[][] {
  const shape = (pixel_array as NDArray).shape;
  if (shape.length < 3 || (shape[2] ?? 0) < 4) return [];
  const voxels = (pixel_array as NDArray).toArray() as unknown[][][];
  return voxels.map((row) => row.map((pixel) => numeric((pixel as unknown[])[3])));
}
