import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import { Camera } from "../src/camera/camera.js";
import { OldMultiCamera } from "../src/camera/mapping_camera.js";

// Ground truths from original Python manim:
//   Camera(pixel_height=10, pixel_width=20).adjusted_thickness(4) == 301.0
//     (default config 1080x1920 -> big_sum 3000, this_sum 30, factor 100)
//   on_screen_pixels([[0,0],[19,9],[20,9],[-1,0],[5,10]]) (20x10 camera)
//     -> [True, True, False, False, False]
//   get_thickening_nudges(2) -> [[0,0],[0,1],[1,0],[1,1]]
//   get_thickening_nudges(1) -> [[0,0]]
//   get_thickening_nudges(3) -> 9 pairs from [-1,0,1]x[-1,0,1]
//   OldMultiCamera((c1,(0,0)), (c2,(0,2)), pixel_height=2, pixel_width=4):
//     set_pixel_array(arange(2*4*4).reshape(2,4,4)) copies through;
//     c1 gets rows 0:2, cols 0:2; c2 gets rows 0:2, cols 2:4.

function rowsOf(value: unknown): number[][][] {
  const arr = value as { toArray: () => unknown };
  return arr.toArray() as number[][][];
}

describe("Camera.adjusted_thickness", () => {
  it("scales stroke width by the global/default pixel sum ratio", () => {
    const camera = new Camera({ pixel_height: 10, pixel_width: 20 });
    expect(camera.adjusted_thickness(4)).toBe(301.0);
  });
  it("leaves thickness 1 unchanged", () => {
    const camera = new Camera({ pixel_height: 10, pixel_width: 20 });
    expect(camera.adjusted_thickness(1)).toBe(1);
  });
});

describe("Camera.on_screen_pixels", () => {
  it("masks out-of-screen pixel coordinates", () => {
    const camera = new Camera({ pixel_height: 10, pixel_width: 20 });
    const result = camera.on_screen_pixels(
      array([
        [0, 0],
        [19, 9],
        [20, 9],
        [-1, 0],
        [5, 10],
      ]),
    );
    expect(result).toEqual([true, true, false, false, false]);
  });
  it("accepts plain nested arrays", () => {
    const camera = new Camera({ pixel_height: 2, pixel_width: 2 });
    expect(
      camera.on_screen_pixels([
        [0, 0],
        [2, 2],
      ]),
    ).toEqual([true, false]);
  });
});

describe("Camera.get_thickening_nudges", () => {
  it("matches Python for thickness 2", () => {
    const camera = new Camera({ pixel_height: 4, pixel_width: 4 });
    expect(rowsOf(camera.get_thickening_nudges(2))).toEqual([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
  });
  it("matches Python for thickness 1", () => {
    const camera = new Camera({ pixel_height: 4, pixel_width: 4 });
    expect(rowsOf(camera.get_thickening_nudges(1))).toEqual([[0, 0]]);
  });
  it("matches Python for thickness 3", () => {
    const camera = new Camera({ pixel_height: 4, pixel_width: 4 });
    expect(rowsOf(camera.get_thickening_nudges(3))).toEqual([
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0, -1],
      [0, 0],
      [0, 1],
      [1, -1],
      [1, 0],
      [1, 1],
    ]);
  });
  it("truncates float thickness like int()", () => {
    const camera = new Camera({ pixel_height: 4, pixel_width: 4 });
    expect(rowsOf(camera.get_thickening_nudges(2.9))).toEqual(rowsOf(camera.get_thickening_nudges(2)));
  });
});

describe("OldMultiCamera.set_pixel_array", () => {
  it("copies to the main array and forwards slices to shifted cameras", () => {
    const c1 = new Camera({ pixel_height: 2, pixel_width: 2 });
    const c2 = new Camera({ pixel_height: 2, pixel_width: 2 });
    const mc = new OldMultiCamera([c1, [0, 0]], [c2, [0, 2]], { pixel_height: 2, pixel_width: 4 });
    const values: number[][][] = [];
    for (let y = 0; y < 2; y += 1) {
      const row: number[][] = [];
      for (let x = 0; x < 4; x += 1) {
        const base = (y * 4 + x) * 4;
        row.push([base, base + 1, base + 2, base + 3]);
      }
      values.push(row);
    }
    mc.set_pixel_array(array(values));
    expect(rowsOf(mc.pixel_array)).toEqual(values);
    expect(rowsOf(c1.pixel_array)).toEqual(values.map((row) => row.slice(0, 2)));
    expect(rowsOf(c2.pixel_array)).toEqual(values.map((row) => row.slice(2, 4)));
  });
});

describe("OldMultiCamera.init_background", () => {
  it("initialises main and shifted backgrounds with matching shapes", () => {
    const c1 = new Camera({ pixel_height: 2, pixel_width: 2 });
    const c2 = new Camera({ pixel_height: 2, pixel_width: 2 });
    const mc = new OldMultiCamera([c1, [0, 0]], [c2, [0, 2]], { pixel_height: 2, pixel_width: 4 });
    mc.init_background();
    expect([...mc.background.shape]).toEqual([2, 4, 4]);
    expect([...c1.background.shape]).toEqual([2, 2, 4]);
    expect([...c2.background.shape]).toEqual([2, 2, 4]);
    // Same fill colour everywhere: opaque black [0,0,0,255].
    expect(rowsOf(c1.background)).toEqual(rowsOf(new Camera({ pixel_height: 2, pixel_width: 2 }).background));
    expect(rowsOf(mc.background)[0]?.[0]).toEqual([0, 0, 0, 255]);
  });
});
