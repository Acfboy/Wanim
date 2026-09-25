import { array, type NDArray } from "numpy-ts";
import { Camera, type CameraOptions } from "./camera.js";
import { isNDArray, numeric } from "../utils/arrays.js";

export interface ShiftedCamera {
  camera: Camera;
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
}

export type CameraWithStartPosition = [Camera, [number, number]];

function toVoxels(pixel_array: unknown): number[][][] {
  if (isNDArray(pixel_array)) {
    return (pixel_array as NDArray).toArray() as number[][][];
  }
  return (pixel_array as number[][][]).map((row) =>
    row.map((pixel) => (Array.isArray(pixel) ? pixel : [pixel]).map(numeric)),
  );
}

export class OldMultiCamera extends Camera {
  shifted_cameras: ShiftedCamera[];

  constructor(...args: unknown[]) {
    let options: CameraOptions = {};
    const pairs: CameraWithStartPosition[] = [];
    for (const arg of args) {
      if (Array.isArray(arg) && arg[0] instanceof Camera) {
        pairs.push(arg as CameraWithStartPosition);
      } else if (arg !== null && typeof arg === "object" && !Array.isArray(arg)) {
        options = { ...options, ...(arg as CameraOptions) };
      }
    }
    const shifted_cameras: ShiftedCamera[] = pairs.map(([camera, start]) => ({
      camera,
      start_x: start[1],
      start_y: start[0],
      end_x: start[1] + camera.pixel_width,
      end_y: start[0] + camera.pixel_height,
    }));
    super(options);
    this.shifted_cameras = shifted_cameras;
    this.init_background();
  }

  set_pixel_array(pixel_array: unknown, convert_from_floats = false): void {
    super.set_pixel_array(pixel_array, convert_from_floats);
    const source = toVoxels(pixel_array);
    for (const shifted_camera of this.shifted_cameras) {
      const sub = source
        .slice(shifted_camera.start_y, shifted_camera.end_y)
        .map((row) => row.slice(shifted_camera.start_x, shifted_camera.end_x));
      shifted_camera.camera.set_pixel_array(array(sub), convert_from_floats);
    }
  }

  init_background(): void {
    super.init_background();
    if (this.shifted_cameras === undefined) return;
    for (const shifted_camera of this.shifted_cameras) {
      shifted_camera.camera.init_background();
    }
  }
}
