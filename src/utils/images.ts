import { array, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "./arrays.js";

type PixelInput = NDArray | number[][] | number[][][] | number[];

export function change_to_rgba_array(image: PixelInput, dtype = "uint8"): NDArray {
  let voxels: number[][][];
  if (isNDArray(image)) {
    const shape = (image as NDArray).shape;
    const raw = (image as NDArray).toArray() as unknown;
    if (shape.length <= 2) {
      const flat = (Array.isArray(raw) ? raw : [raw]) as unknown[];
      const flatNums = flat.map(numeric);
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
  return (array(voxels) as NDArray).astype(dtype as "uint8") as NDArray;
}
