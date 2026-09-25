import { array, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "./arrays.js";
import { config } from "../_config/utils.js";
import { seek_full_path_from_defaults } from "./file_ops.js";

type PixelInput = NDArray | number[][] | number[][][] | number[];

export function change_to_rgba_array(image: PixelInput, dtype = "uint8"): NDArray {
  void dtype;
  let voxels: number[][][];
  if (isNDArray(image)) {
    const shape = (image as NDArray).shape;
    const raw = (image as NDArray).toArray() as unknown;
    if (shape.length <= 2) {
      const flat = (Array.isArray(raw) ? raw : [raw]) as unknown[];
      const flatNums = flat.flatMap((item) => Array.isArray(item) ? (item as unknown[]).map(numeric) : [numeric(item)]);
      const cols = shape.length === 2 ? shape[1] : flatNums.length;
      const grid: number[][] = [];
      for (let i = 0; i < flatNums.length; i += cols) grid.push(flatNums.slice(i, i + cols));
      voxels = grid.map((row) => row.map((v) => [v]));
    } else {
      voxels = (raw as unknown[][][]).map((row) =>
        row.map((pixel) => (Array.isArray(pixel) ? pixel : [pixel]).map(numeric)),
      );
    }
  } else if (Array.isArray(image) && image.length > 0 && Array.isArray((image as unknown[][])[0])) {
    const grid = image as unknown[][];
    if (Array.isArray(grid[0][0])) {
      voxels = (grid as unknown[][][]).map((row) =>
        row.map((pixel) => (Array.isArray(pixel) ? pixel : [pixel]).map(numeric)),
      );
    } else {
      voxels = (grid as unknown[][]).map((row) => row.map((v) => [numeric(v)]));
    }
  } else {
    const flat = (image as unknown[]).map(numeric);
    voxels = [flat.map((v) => [v])];
  }
  let channels = voxels[0]?.[0]?.length ?? 0;
  if (channels === 1) {
    voxels = voxels.map((row) => row.map((pixel) => [pixel[0] ?? 0, pixel[0] ?? 0, pixel[0] ?? 0]));
    channels = 3;
  }
  if (channels === 3) {
    voxels = voxels.map((row) => row.map((pixel) => [...pixel, 255]));
  }
  // NOTE: Python's change_to_rgba_array only uses `dtype` for the appended
  // alpha channel (`255 * np.ones(..., dtype=dtype)`) and returns the array
  // with its original dtype otherwise (float input stays float). So we must
  // NOT cast the whole array to `dtype` here: astype("uint8") would truncate
  // float pixels (e.g. 0.5 -> 0).
  return array(voxels) as NDArray;
}

export function get_full_raster_image_path(image_file_name: string): string {
  return seek_full_path_from_defaults(
    image_file_name,
    config.getDir("assets_dir"),
    [".jpg", ".jpeg", ".png", ".gif", ".ico"],
  );
}
