// Ground-truth tests for batch layers 14..14 (third file: point-cloud types,
// image mobjects, Camera.type_or_raise, graphing functions, Line buff,
// Elbow, Angle, _BooleanOps).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): none of these symbols touch text/tex.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../src/mobject/mobject.js";
import {
  Mobject1D,
  Mobject2D,
  PGroup,
  PMobject,
  PointCloudDot,
} from "../src/mobject/types/point_cloud_mobject.js";
import {
  ImageMobject,
  ImageMobjectFromCamera,
  HIGH_QUALITY_PIXEL_HEIGHT,
} from "../src/mobject/types/image_mobject.js";
import { Camera } from "../src/camera/camera.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import {
  ParametricFunction,
  ImplicitFunction,
  setIsolineComputer,
} from "../src/mobject/graphing/functions.js";
import { Line, Elbow, Angle } from "../src/mobject/geometry/line.js";
import { _BooleanOps } from "../src/mobject/geometry/boolean_ops.js";
import { Point } from "../src/mobject/types/point_cloud_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}
function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}
function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}
function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(actual[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

// ---------------------------------------------------------------------------
// Mobject1D
// Python:
//   m1 = Mobject1D() -> density 10, epsilon 0.1
//   m1.add_line([0,0,0],[1,0,0]) -> 10 pts, first [0,0,0], last [0.9,0,0]
//   Mobject1D(density=10).add_line(...) pts =
//     [[0,0,0],[0.1,0,0],[0.2,0,0],[0.30000000000000004,0,0],[0.4,0,0],
//      [0.5,0,0],[0.6000000000000001,0,0],[0.7000000000000001,0,0],[0.8,0,0],[0.9,0,0]]
//   zero-length line -> 1 pt [[1,2,3]]
//   add_line(color='#FF0000') -> rgbas[0] == [1,0,0,1]
//   default stroke_width 4
// ---------------------------------------------------------------------------
describe("Mobject1D", () => {
  it("defaults to density 10 / epsilon 0.1", () => {
    const m = new Mobject1D();
    expect(m.density).toBe(10);
    expectClose(m.epsilon, 0.1);
  });

  it("add_line samples arange(0,1,eps/len) like Python", () => {
    const m = new Mobject1D();
    m.add_line(array([0, 0, 0]), array([1, 0, 0]));
    expect(m.get_num_points()).toBe(10);
    expectRowsClose(rowsOf(m.points, 3).slice(0, 1), [[0, 0, 0]]);
    expectRowsClose(rowsOf(m.points, 3).slice(-1), [[0.9, 0, 0]]);
    expectRowsClose(rowsOf(m.points, 3), [
      [0, 0, 0],
      [0.1, 0, 0],
      [0.2, 0, 0],
      [0.30000000000000004, 0, 0],
      [0.4, 0, 0],
      [0.5, 0, 0],
      [0.6000000000000001, 0, 0],
      [0.7000000000000001, 0, 0],
      [0.8, 0, 0],
      [0.9, 0, 0],
    ]);
  });

  it("zero-length line yields a single point", () => {
    const m = new Mobject1D({ density: 10 });
    m.add_line(array([1, 2, 3]), array([1, 2, 3]));
    expect(m.get_num_points()).toBe(1);
    expectRowsClose(rowsOf(m.points, 3), [[1, 2, 3]]);
  });

  it("add_line honors an explicit color", () => {
    const m = new Mobject1D({ density: 10 });
    m.add_line(array([0, 0, 0]), array([1, 0, 0]), "#FF0000");
    expectRowsClose(rowsOf(m.rgbas, 4).slice(0, 1), [[1, 0, 0, 1]]);
    expect(m.get_stroke_width()).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Mobject2D
// Python: Mobject2D() -> density 25, epsilon 0.04; Mobject2D(density=7) ->
//   density 7, epsilon 0.14285714285714285
// ---------------------------------------------------------------------------
describe("Mobject2D", () => {
  it("defaults and custom density match Python", () => {
    const d = new Mobject2D();
    expect(d.density).toBe(25);
    expectClose(d.epsilon, 0.04);
    const c = new Mobject2D({ density: 7 });
    expect(c.density).toBe(7);
    expectClose(c.epsilon, 0.14285714285714285);
  });
});

// ---------------------------------------------------------------------------
// PGroup / PGroup.__init__
// Python:
//   PGroup(p1, p2) -> 2 submobjects, all PMobject instances
//   PGroup(p1, 'notamob') -> ValueError("All submobjects must be of type ...")
//   PGroup() -> 0 submobjects
//   fade_to('#FF0000', 0.5): submob rgbas[0] [1,1,0,1] -> [1,0.5,0,1];
//     group's own rgbas stays length 0
// ---------------------------------------------------------------------------
describe("PGroup", () => {
  it("collects PMobjects as submobjects", () => {
    const g = new PGroup(
      new PointCloudDot([4.5, 0, 0], 1, 2, 20, "#0000FF"),
      new PointCloudDot(),
    );
    expect(g.submobjects.length).toBe(2);
    expect(g.submobjects.every((x) => x instanceof PMobject)).toBe(true);
  });

  it("empty PGroup has no submobjects", () => {
    expect(new PGroup().submobjects.length).toBe(0);
  });

  it("rejects non-PMobject members with ValueError", () => {
    expect(() => new PGroup(new PointCloudDot(), "notamob" as unknown as PMobject)).toThrowError(
      /All submobjects must be of type PMobject/,
    );
    try {
      new PGroup(new PointCloudDot(), 42 as unknown as PMobject);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });

  it("fade_to blends submobjects but leaves own rgbas untouched", () => {
    const g = new PGroup(
      new PointCloudDot([0, 0, 0], 0.5, 2, 10),
      new PointCloudDot([1, 0, 0], 0.5, 2, 10),
    );
    const before = rowsOf((g.submobjects[0] as PMobject).rgbas, 4)[0]!;
    expectRowsClose([before], [[1, 1, 0, 1]]);
    g.fade_to("#FF0000", 0.5);
    const after = rowsOf((g.submobjects[0] as PMobject).rgbas, 4)[0]!;
    expectRowsClose([after], [[1, 0.5, 0, 1]]);
    expect(rowsOf(g.rgbas, 4).length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// ImageMobject
// Python (arr = [[[255,0,0,255],[0,255,0,255]],[[0,0,255,255],[255,255,0,128]]]):
//   shape (2,2,4) uint8; pix[0,0] [255,0,0,255]; points shape (4,3) with
//   first rows [-0.007407407407407408, 0.007407407407407408, 0] /
//             [ 0.007407407407407408, 0.007407407407407408, 0]
//   invert=True -> pix[0,0] [0,255,255,255]
//   set_color('#0000FF') -> pix[0,0] [0,0,255,255], pix[1,1] [0,0,255,128]
//   (alpha channel preserved per pixel)
//   set_opacity(0.5) -> pix[0,0] [255,0,0,127], pix[1,1] [255,255,0,64]
//   interpolate_color(a, b, 0.5) -> pix[0,0] [127,0,0,127],
//     pix[0,1] [127,255,127,255]; fill_opacity 1.0; style
//     {'fill_color': '#FFFFFF', 'fill_opacity': 1.0}
//   shape mismatch -> AssertionError
//   set_resampling_algorithm default 3; set 0 ok; set 99 -> ValueError
// ---------------------------------------------------------------------------
const TEST_PIXELS = [
  [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
  ],
  [
    [0, 0, 255, 255],
    [255, 255, 0, 128],
  ],
];

describe("ImageMobject", () => {
  it("builds from a pixel array with cairo-style corner points", () => {
    const im = new ImageMobject(array(TEST_PIXELS) as NDArray);
    expect(im.pixel_array.shape).toEqual([2, 2, 4]);
    expectRowsClose(
      [(im.pixel_array.toArray() as number[][][])[0]?.[0] ?? []],
      [[255, 0, 0, 255]],
    );
    expect(im.points.shape).toEqual([4, 3]);
    expectRowsClose(rowsOf(im.points, 3).slice(0, 2), [
      [-0.007407407407407408, 0.007407407407407408, 0],
      [0.007407407407407408, 0.007407407407407408, 0],
    ]);
  });

  it("invert flips RGB but keeps alpha", () => {
    const im = new ImageMobject(
      array(TEST_PIXELS) as NDArray,
      HIGH_QUALITY_PIXEL_HEIGHT,
      true,
    );
    expectRowsClose(
      [(im.pixel_array.toArray() as number[][][])[0]?.[0] ?? []],
      [[0, 255, 255, 255]],
    );
  });

  it("set_color rewrites RGB and preserves per-pixel alpha", () => {
    const im = new ImageMobject(array(TEST_PIXELS) as NDArray);
    im.set_color("#0000FF");
    const vox = im.pixel_array.toArray() as number[][][];
    expectRowsClose([vox[0]?.[0] ?? []], [[0, 0, 255, 255]]);
    expectRowsClose([vox[1]?.[1] ?? []], [[0, 0, 255, 128]]);
  });

  it("set_opacity scales the original alpha channel", () => {
    const im = new ImageMobject(array(TEST_PIXELS) as NDArray);
    im.set_opacity(0.5);
    const vox = im.pixel_array.toArray() as number[][][];
    expectRowsClose([vox[0]?.[0] ?? []], [[255, 0, 0, 127]]);
    expectRowsClose([vox[1]?.[1] ?? []], [[255, 255, 0, 64]]);
  });

  it("interpolate_color lerps pixel arrays elementwise", () => {
    const a = new ImageMobject(array(TEST_PIXELS) as NDArray);
    const b = new ImageMobject(
      array([
        [
          [0, 0, 0, 0],
          [255, 255, 255, 255],
        ],
        [
          [128, 128, 128, 255],
          [0, 0, 0, 255],
        ],
      ]) as NDArray,
    );
    const c = new ImageMobject(array(TEST_PIXELS) as NDArray);
    c.interpolate_color(a, b, 0.5);
    const vox = c.pixel_array.toArray() as number[][][];
    expectRowsClose([vox[0]?.[0] ?? []], [[127, 0, 0, 127]]);
    expectRowsClose([vox[0]?.[1] ?? []], [[127, 255, 127, 255]]);
    expectClose(c.fill_opacity, 1.0);
    expectClose(c.stroke_opacity, 1.0);
    expect(c.get_style()).toEqual({ fill_color: "#FFFFFF", fill_opacity: 1.0 });
  });

  it("interpolate_color rejects shape mismatches", () => {
    const a = new ImageMobject(array(TEST_PIXELS) as NDArray);
    const bad = new ImageMobject(array([[[0, 0, 0, 0]]]) as NDArray);
    // pad bad to a real 3x3x4 array of zeros
    const zeros3 = new ImageMobject(
      array(
        Array.from({ length: 3 }, () =>
          Array.from({ length: 3 }, () => [0, 0, 0, 0]),
        ),
      ) as NDArray,
    );
    expect(zeros3.pixel_array.shape).toEqual([3, 3, 4]);
    void bad;
    try {
      a.interpolate_color(a, zeros3, 0.5);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("AssertionError");
    }
  });

  it("set_resampling_algorithm validates its input", () => {
    const im = new ImageMobject(array(TEST_PIXELS) as NDArray);
    expect(im.resampling_algorithm).toBe(3);
    im.set_resampling_algorithm(0);
    expect(im.resampling_algorithm).toBe(0);
    try {
      im.set_resampling_algorithm(99);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });
});

describe("ImageMobjectFromCamera", () => {
  it("mirrors the camera pixel array and defaults", () => {
    const cam = new MovingCamera();
    expect(cam.pixel_array.shape).toEqual([1080, 1920, 4]);
    const fmc = new ImageMobjectFromCamera(cam);
    expect(fmc.pixel_array.shape).toEqual([1080, 1920, 4]);
    expect(fmc.scale_to_resolution).toBe(false);
    expect(fmc.default_display_frame_config["stroke_width"]).toBe(3);
    expect(fmc.default_display_frame_config["buff"]).toBe(0);
    expect(fmc.points.shape).toEqual([4, 3]);
    expect(fmc.get_pixel_array()).toBe(cam.pixel_array);
  });

  it("interpolate_color lerps camera frames elementwise", () => {
    const cam1 = new MovingCamera(null, { pixel_height: 2, pixel_width: 2 });
    cam1.pixel_array = array([
      [
        [10, 20, 30, 40],
        [50, 60, 70, 80],
      ],
      [
        [90, 100, 110, 120],
        [130, 140, 150, 160],
      ],
    ]) as NDArray;
    const cam2 = new MovingCamera(null, { pixel_height: 2, pixel_width: 2 });
    cam2.pixel_array = array([
      [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
      ],
      [
        [100, 100, 100, 100],
        [200, 200, 200, 200],
      ],
    ]) as NDArray;
    const f1 = new ImageMobjectFromCamera(cam1);
    const f2 = new ImageMobjectFromCamera(cam2);
    const f3 = new ImageMobjectFromCamera(cam1);
    f3.interpolate_color(f1, f2, 0.5);
    const vox = f3.pixel_array.toArray() as number[][][];
    expectRowsClose([vox[0]?.[0] ?? []], [[5, 10, 15, 20]]);
    expectRowsClose([vox[1]?.[1] ?? []], [[165, 170, 175, 180]]);
  });

  it("interpolate_color rejects non-camera mobjects", () => {
    const cam = new MovingCamera(null, { pixel_height: 2, pixel_width: 2 });
    const f = new ImageMobjectFromCamera(cam);
    const plain = new ImageMobject(array(TEST_PIXELS) as NDArray);
    try {
      f.interpolate_color(plain, plain, 0.5);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("AssertionError");
    }
  });
});

// ---------------------------------------------------------------------------
// Camera.type_or_raise
// Python (Camera(pixel_height=4, pixel_width=6)):
//   Square() -> VMobject; Point() -> PMobject;
//   ImageMobject(zeros) -> AbstractImageMobject; Mobject() -> Mobject;
//   type_or_raise(42) -> TypeError("Displaying an object of class ...")
// ---------------------------------------------------------------------------
describe("Camera.type_or_raise", () => {
  it("resolves renderable superclasses and raises TypeError otherwise", () => {
    const cam = new Camera({ pixel_height: 4, pixel_width: 6 });
    expect(cam.type_or_raise(new Square()).name).toBe("VMobject");
    expect(cam.type_or_raise(new Point()).name).toBe("PMobject");
    expect(
      cam.type_or_raise(new ImageMobject(array(TEST_PIXELS) as NDArray)).name,
    ).toBe("AbstractImageMobject");
    expect(cam.type_or_raise(new Mobject()).name).toBe("Mobject");
    expect(() => cam.type_or_raise(42 as unknown as Mobject)).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// ParametricFunction
// Python:
//   ParametricFunction(t -> [t, t^2, 0], t_range=[0,1,0.5], use_smoothing=False)
//     -> 8 pts, first rows [0,0,0],[1/6,1/12,0],[1/3,1/6,0],[0.5,0.25,0],[0.5,0.25,0]]
//     t_min/max/step 0/1/0.5; get_point_from_function(0.5) == [0.5,0.25,0]
//   defaults: t 0/1/0.01, dt 1e-8, discontinuities None, smoothing True,
//     vectorized False; default-range curve has 400 pts, first [0,0,0], last [1,0,0]
//   discontinuities=[0.5] variant -> 400 pts
// ---------------------------------------------------------------------------
describe("ParametricFunction", () => {
  it("samples t_range=[0,1,0.5] without smoothing like Python", () => {
    const pf = new ParametricFunction((t: number) => [t, t * t, 0], {
      t_range: [0, 1, 0.5],
      use_smoothing: false,
    });
    expect(pf.t_min).toBe(0);
    expect(pf.t_max).toBe(1);
    expectClose(pf.t_step, 0.5);
    expect(pf.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(pf.points, 3).slice(0, 5), [
      [0, 0, 0],
      [0.16666666666666666, 0.08333333333333333, 0],
      [0.3333333333333333, 0.16666666666666666, 0],
      [0.5, 0.25, 0],
      [0.5, 0.25, 0],
    ]);
    expectRowsClose([rowsOf(pf.get_point_from_function(0.5), 3)[0]!], [
      [0.5, 0.25, 0],
    ]);
  });

  it("exposes Python-compatible defaults", () => {
    const pf = new ParametricFunction((t: number) => [t, 0, 0], {
      use_smoothing: false,
    });
    expect([pf.t_min, pf.t_max, pf.t_step]).toEqual([0, 1, 0.01]);
    expectClose(pf.dt, 1e-8);
    expect(pf.discontinuities).toBeNull();
    expect(pf.use_vectorized).toBe(false);
    expect(pf.get_num_points()).toBe(400);
    expectRowsClose(rowsOf(pf.points, 3).slice(0, 1), [[0, 0, 0]]);
    expectRowsClose(rowsOf(pf.points, 3).slice(-1), [[1, 0, 0]]);
  });

  it("splits at discontinuities like Python (400 pts)", () => {
    const pf = new ParametricFunction((t: number) => [Math.cos(t), Math.sin(t), 0], {
      t_range: [0, 1],
      discontinuities: [0.5],
      use_smoothing: false,
    });
    expect(pf.get_num_points()).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// ImplicitFunction
// Python: ImplicitFunction(x^2+y^2-1, x_range=[-2,2], y_range=[-2,2])
//   -> x_range [-2,2], y_range [-2,2], min_depth 5, max_quads 1500, 2976 pts
//   defaults: x_range [-7.111111111111111, 7.111111111111111],
//             y_range [-4, 4] (from config frame_width/frame_height)
// TS has no `isosurfaces` port yet: without a registered computer it throws
// a TODO error; with a stub computer the pmin/pmax/depth plumbing is tested.
// ---------------------------------------------------------------------------
describe("ImplicitFunction", () => {
  it("stores ranges and marcher options like Python", () => {
    // Register a stub isoline computer so construction succeeds; it records
    // its inputs and returns no curves.
    let seen: { pmin: number[]; pmax: number[]; min_depth: number; max_quads: number } | null = null;
    setIsolineComputer((fn, pmin, pmax, min_depth, max_quads) => {
      seen = { pmin: [...pmin], pmax: [...pmax], min_depth, max_quads };
      void fn;
      return [];
    });
    try {
      const f = new ImplicitFunction((x: number, y: number) => x * x + y * y - 1, {
        x_range: [-2, 2],
        y_range: [-2, 2],
        use_smoothing: false,
      });
      expect(f.x_range).toEqual([-2, 2]);
      expect(f.y_range).toEqual([-2, 2]);
      expect(f.min_depth).toBe(5);
      expect(f.max_quads).toBe(1500);
      expect(seen).toEqual({ pmin: [-2, -2], pmax: [2, 2], min_depth: 5, max_quads: 1500 });
      expect(f.get_num_points()).toBe(0);
    } finally {
      setIsolineComputer(null);
    }
  });

  it("defaults ranges to the configured frame", () => {
    setIsolineComputer(() => []);
    try {
      const f = new ImplicitFunction((x: number, y: number) => x + y, {
        use_smoothing: false,
      });
      expectRowsClose([f.x_range], [[-7.111111111111111, 7.111111111111111]]);
      expect(f.y_range).toEqual([-4, 4]);
    } finally {
      setIsolineComputer(null);
    }
  });

  it("builds VMobject paths from stub isolines", () => {
    setIsolineComputer(() => [
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    try {
      const f = new ImplicitFunction((x: number, y: number) => x + y, {
        x_range: [-2, 2],
        y_range: [-2, 2],
        use_smoothing: false,
      });
      expect(f.get_num_points()).toBeGreaterThan(0);
      expectRowsClose(rowsOf(f.points, 3).slice(0, 1), [[1, 0, 0]]);
    } finally {
      setIsolineComputer(null);
    }
  });

  it("throws a TODO error when no isoline computer is registered", () => {
    // NOTE: the TS constructor stays lazy (empty mobject) while no computer
    // is registered and fails fast only once points are actually requested,
    // because the `isosurfaces` plot_isoline routine is not ported yet.
    setIsolineComputer(null);
    const f = new ImplicitFunction((x: number, y: number) => x * x + y * y - 1, {
      x_range: [-1, 1],
      y_range: [-1, 1],
    });
    expect(f.get_num_points()).toBe(0);
    expect(() => f.generate_points()).toThrowError(/isosurfaces|not ported/);
  });
});

// ---------------------------------------------------------------------------
// Line._account_for_buff
// Python:
//   Line([0,0,0],[4,0,0], buff=1.0) -> start [1,0,0], end [3,0,0], len 2.0
//   Line([0,0,0],[1,0,0], buff=1.0) unchanged (length < 2*buff)
//   _account_for_buff(0) unchanged
//   set_points_by_ends(..., buff=0.5) on [0,4] -> [0.5,0,0]..[3.5,0,0]
// ---------------------------------------------------------------------------
describe("Line._account_for_buff", () => {
  it("trims buff from both ends", () => {
    const l = new Line([0, 0, 0], [4, 0, 0], { buff: 1.0 });
    expectRowsClose([l.get_start()], [[1, 0, 0]]);
    expectRowsClose([l.get_end()], [[3, 0, 0]]);
    expectClose(l.get_length(), 2.0);
  });

  it("leaves short lines and zero buff alone", () => {
    const tiny = new Line([0, 0, 0], [1, 0, 0], { buff: 1.0 });
    expectRowsClose([tiny.get_start()], [[0, 0, 0]]);
    expectRowsClose([tiny.get_end()], [[1, 0, 0]]);
    const plain = new Line([0, 0, 0], [4, 0, 0]);
    plain._account_for_buff(0);
    expectRowsClose([plain.get_start()], [[0, 0, 0]]);
    expectRowsClose([plain.get_end()], [[4, 0, 0]]);
  });

  it("set_points_by_ends applies buff", () => {
    const l = new Line([0, 0, 0], [4, 0, 0]);
    l.set_points_by_ends(array([0, 0, 0]), array([4, 0, 0]), 0.5);
    expectRowsClose([l.get_start()], [[0.5, 0, 0]]);
    expectRowsClose([l.get_end()], [[3.5, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// Elbow
// Python:
//   Elbow() -> 8 pts:
//     [[0,0.2,0],[0.0666..,0.2,0],[0.1333..,0.2,0],[0.2,0.2,0],
//      [0.2,0.2,0],[0.2,0.1333..,0],[0.2,0.0666..,0],[0.2,0,0]]
//   Elbow(width=0.4, angle=pi/2) -> first [-0.4,~0,0], last [~0,0.4,0]
// ---------------------------------------------------------------------------
describe("Elbow", () => {
  it("builds the default L-shape like Python", () => {
    const e = new Elbow();
    expect(e.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(e.points, 3), [
      [0, 0.2, 0],
      [0.06666666666666667, 0.2, 0],
      [0.13333333333333333, 0.2, 0],
      [0.2, 0.2, 0],
      [0.2, 0.2, 0],
      [0.2, 0.13333333333333336, 0],
      [0.2, 0.06666666666666668, 0],
      [0.2, 0, 0],
    ]);
  });

  it("rescales and rotates with width/angle", () => {
    const e = new Elbow(0.4, Math.PI / 2);
    expect(e.get_num_points()).toBe(8);
    const rows = rowsOf(e.points, 3);
    expectClose(rows[0]?.[0] ?? NaN, -0.4);
    expectClose(rows[0]?.[1] ?? NaN, 0, 1e-9);
    expectClose(rows[rows.length - 1]?.[1] ?? NaN, 0.4);
  });
});

// ---------------------------------------------------------------------------
// Angle
// Python (a1 = Line([0,0,0],[1,0,0]), a2 = Line([0,0,0],[0,1,0])):
//   Angle(a1, a2).angle_value == pi/2, get_value(degrees=True) == 90.0, 32 pts
//   radius=0.5 + other_angle -> angle_value -3pi/2 (-4.71238898038469)
//   elbow=True -> 8 pts, first rows [0.4,0,0],[0.4,0.1333,0],[0.4,0.2666,0]
//   dot=True -> 1 submobject; get_lines() holds the 2 Lines
//   from_three_points([1,0,0],[0,0,0],[0,1,0]) -> pi/2
//   parallel lines -> angle_value 0.0, 0 pts (issue #1930, no raise)
// ---------------------------------------------------------------------------
describe("Angle", () => {
  it("measures a right angle like Python", () => {
    const a1 = new Line([0, 0, 0], [1, 0, 0]);
    const a2 = new Line([0, 0, 0], [0, 1, 0]);
    const ang = new Angle(a1, a2);
    expectClose(ang.angle_value, Math.PI / 2);
    expectClose(ang.get_value(true), 90.0);
    expectClose(ang.get_value(), Math.PI / 2);
    expect(ang.get_num_points()).toBe(32);
  });

  it("supports other_angle with explicit radius", () => {
    const ang = new Angle(new Line([0, 0, 0], [1, 0, 0]), new Line([0, 0, 0], [0, 1, 0]), {
      radius: 0.5,
      other_angle: true,
    });
    expectClose(ang.angle_value, -4.71238898038469);
  });

  it("builds the elbow variant as corners", () => {
    const ang = new Angle(new Line([0, 0, 0], [1, 0, 0]), new Line([0, 0, 0], [0, 1, 0]), {
      elbow: true,
    });
    expect(ang.get_num_points()).toBe(8);
    expectRowsClose(rowsOf(ang.points, 3).slice(0, 3), [
      [0.4, 0, 0],
      [0.4, 0.13333333333333333, 0],
      [0.4, 0.26666666666666666, 0],
    ]);
  });

  it("attaches a dot and remembers its lines", () => {
    const l1 = new Line([0, 0, 0], [1, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 1, 0]);
    const ang = new Angle(l1, l2, { dot: true });
    expect(ang.submobjects.length).toBe(1);
    const lines = ang.get_lines();
    expect(lines.submobjects.length).toBe(2);
    expect(lines.submobjects[0]).toBe(l1);
    expect(lines.submobjects[1]).toBe(l2);
  });

  it("from_three_points matches the two-line form", () => {
    const ang = Angle.from_three_points([1, 0, 0], [0, 0, 0], [0, 1, 0]);
    expectClose(ang.angle_value, Math.PI / 2);
  });

  it("collapses to an empty mobject for parallel lines", () => {
    const ang = new Angle(new Line([0, 0, 0], [1, 0, 0]), new Line([0, 1, 0], [1, 1, 0]));
    expect(ang.angle_value).toBe(0.0);
    expect(ang.get_num_points()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// _BooleanOps
// Python:
//   _convert_2d_to_3d_array([(1,2),(3,4)]) == [[1,2,0],[3,4,0]]
//   _convert_2d_to_3d_array([(1,2,3),(4,5,6)]) unchanged
//   z_dim=5 -> [[1,2,5]]; [] -> [] (empty array)
//   skia converters need `pathops` -> TS throws TODO
// ---------------------------------------------------------------------------
describe("_BooleanOps", () => {
  it("lifts 2D points to 3D and passes 3D through", () => {
    const b = new _BooleanOps();
    expectRowsClose(rowsOf(b._convert_2d_to_3d_array([[1, 2], [3, 4]]), 3), [
      [1, 2, 0],
      [3, 4, 0],
    ]);
    expectRowsClose(rowsOf(b._convert_2d_to_3d_array([[1, 2, 3], [4, 5, 6]]), 3), [
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expectRowsClose(rowsOf(b._convert_2d_to_3d_array([[1, 2]], 5), 3), [[1, 2, 5]]);
    expect(b._convert_2d_to_3d_array([]).shape[0]).toBe(0);
  });

  it("skia converters: to-path is ported, to-vmobject stays a TODO stub", () => {
    // NOTE (batch-14d ground truth): _convert_vmobject_to_skia_path is
    // implemented via the local SkiaPath shim and verified against Python
    // pathops output in tests/batch-targets-14d.test.ts (Square -> 6 cmds,
    // Circle -> 10 cmds; coords at 1e-6 due to Skia float32 rounding), so an
    // empty VMobject yields an empty path instead of throwing. Only the
    // reverse converter still needs pathops and stays a TODO stub.
    const b = new _BooleanOps();
    expect(b._convert_vmobject_to_skia_path(new VMobject()).commands.length).toBe(0);
    expect(() => b._convert_skia_path_to_vmobject({})).toThrowError(/pathops|not ported/);
  });
});

// ---------------------------------------------------------------------------
// Integration (batch symbols x earlier-ported code)
// ---------------------------------------------------------------------------
describe("batch-14 integration", () => {
  it("Angle built from buff-trimmed Lines still measures 90 degrees", () => {
    // Python: Line([0,0,0],[2,0,0],buff=0.5) -> [0.5,0,0]..[1.5,0,0];
    // an Angle over such trimmed lines keeps the same direction, hence pi/2.
    const l1 = new Line([0, 0, 0], [2, 0, 0], { buff: 0.5 });
    const l2 = new Line([0, 0, 0], [0, 2, 0], { buff: 0.5 });
    expectRowsClose([l1.get_start()], [[0.5, 0, 0]]);
    const ang = new Angle(l1, l2);
    expectClose(ang.get_value(true), 90.0);
  });

  it("PGroup of PointCloudDots arranges via the Mobject family machinery", () => {
    // Python: PGroup(p1,p2,p3).arrange_in_grid() keeps 3 PMobject submobjects.
    const g = new PGroup(new PointCloudDot(), new PointCloudDot(), new PointCloudDot());
    expect(g.submobjects.length).toBe(3);
    g.arrange_in_grid();
    expect(g.submobjects.length).toBe(3);
    expect(g.submobjects.every((x) => x instanceof PMobject)).toBe(true);
  });

  it("Camera.display routing groups a mixed scene by type_or_raise order", () => {
    // Python: it.groupby(mobjects, type_or_raise) batches consecutive same-type
    // mobjects; VMobject, PMobject, ImageMobject, Mobject each resolve distinctly.
    const cam = new Camera({ pixel_height: 4, pixel_width: 6 });
    const kinds = [
      cam.type_or_raise(new Square()).name,
      cam.type_or_raise(new Point()).name,
      cam.type_or_raise(new ImageMobject(array(TEST_PIXELS) as NDArray)).name,
      cam.type_or_raise(new Mobject()).name,
    ];
    expect(kinds).toEqual(["VMobject", "PMobject", "AbstractImageMobject", "Mobject"]);
  });

  it("ParametricFunction curve endpoints agree with Line endpoints", () => {
    // Python: ParametricFunction(t -> [t,0,0], t_range=[2,4,1], smoothing off)
    // spans [2,0,0]..[4,0,0], same as Line([2,0,0],[4,0,0]).
    const pf = new ParametricFunction((t: number) => [t, 0, 0], {
      t_range: [2, 4, 1],
      use_smoothing: false,
    });
    const rows = rowsOf(pf.points, 3);
    const line = new Line([2, 0, 0], [4, 0, 0]);
    expectRowsClose([rows[0]!], [[2, 0, 0]]);
    expectRowsClose([rows[rows.length - 1]!], [line.get_end()]);
  });
});
