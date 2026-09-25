import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import { LogBase } from "../src/mobject/graphing/scale.js";
import { Line3D } from "../src/mobject/three_d/three_dimensions.js";
import { rotation_matrix } from "../src/utils/space_ops.js";
import { adjacent_n_tuples, adjacent_pairs } from "../src/utils/iterables.js";
import { Camera } from "../src/camera/camera.js";
import { Mobject } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { RED } from "../src/utils/color/core.js";

function nums(value: unknown): number[] {
  if (Array.isArray(value)) {
    const flat = (x: unknown): number[] =>
      Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)];
    return flat(value);
  }
  const raw = (value as { toArray: () => unknown }).toArray();
  const flat = (x: unknown): number[] =>
    Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)];
  return flat(raw);
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = nums(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]!) * relTol);
    expect(Math.abs(values[i]! - expected[i]!)).toBeLessThanOrEqual(tol);
  }
}

function makeCam(): Camera {
  return new Camera({ pixel_height: 4, pixel_width: 6, frame_width: 12, frame_height: 8 });
}

function makeBox(): VMobject {
  const vm = new VMobject();
  vm.set_points_as_corners([
    [-1, -1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
  ]);
  return vm;
}

// ---------------------------------------------------------------------------
// manim.mobject.graphing.scale.LogBase (+ __init__)
// ---------------------------------------------------------------------------

describe("LogBase.__init__", () => {
  // Python: LogBase() -> base 10, custom_labels True.
  it("has the Python defaults", () => {
    const lb = new LogBase();
    expect(lb.base).toBe(10);
    expect(lb.custom_labels).toBe(true);
  });
  it("stores explicit arguments", () => {
    const lb = new LogBase(2, false);
    expect(lb.base).toBe(2);
    expect(lb.custom_labels).toBe(false);
  });
});

describe("LogBase.function", () => {
  // Python: LogBase(10).function(5) == 100000; LogBase(2).function(3) == 8;
  // function(0) == 1; function(-2) == 0.01.
  it("raises the base to scalar powers", () => {
    expect(new LogBase(10).function(5)).toBe(100000);
    expect(new LogBase(2).function(3)).toBe(8);
    expect(new LogBase(10).function(0)).toBe(1);
    expect(new LogBase(10).function(-2) as number).toBeCloseTo(0.01, 12);
  });
  it("broadcasts over arrays like base**value", () => {
    // Python: 10**np.array([1,2,3]) == [10,100,1000].
    expectClose(new LogBase(10).function(array([1, 2, 3])), [10, 100, 1000]);
    // Python: 10**np.array([[1,2],[3,4]]) == [[10,100],[1000,10000]].
    expectClose(
      new LogBase(10).function(array([[1, 2], [3, 4]])),
      [10, 100, 1000, 10000],
    );
  });
});

describe("LogBase.inverse_function (extended)", () => {
  // Python: LogBase(2).inverse_function(8) == 3.0.
  it("uses the configured base", () => {
    expect(new LogBase(2).inverse_function(8)).toBe(3.0);
  });
  it("inverts 2-D arrays elementwise", () => {
    // Python: LogBase(10).inverse_function([[1,10],[100,1000]]) == [[0,1],[2,3]].
    expectClose(
      new LogBase(10).inverse_function(array([[1, 10], [100, 1000]])),
      [0, 1, 2, 3],
    );
  });
  it("raises ValueError for negative scalars", () => {
    // Python raises ValueError("log(0) is undefined. ...") for -3 too.
    let thrown: unknown;
    try {
      new LogBase(10).inverse_function(-3);
    } catch (e) {
      thrown = e;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe(
      "log(0) is undefined. Make sure the value is in the domain of the function",
    );
  });
  it("returns -inf for a zero mixed with positives (no raise)", () => {
    // Python: LogBase(10).inverse_function(np.array([0,5])) ==
    //   [-inf, 0.69897...] with a RuntimeWarning, NOT a ValueError: the
    // condition `value.any() <= 0` is only true for all-zero arrays.
    const out = nums(new LogBase(10).inverse_function(array([0, 5])));
    expect(out[0]).toBe(-Infinity);
    expectClose([out[1]], [0.6989700043360189]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.get_start / get_end
// ---------------------------------------------------------------------------

describe("Line3D constructor + get_start/get_end", () => {
  // Python: Line3D() -> start [-1,0,0] (LEFT), end [1,0,0] (RIGHT),
  // thickness 0.02.
  it("defaults to LEFT/RIGHT with thickness 0.02", () => {
    const line = new Line3D();
    expectClose(line.get_start(), [-1, 0, 0]);
    expectClose(line.get_end(), [1, 0, 0]);
    expect(line.thickness).toBe(0.02);
  });
  it("stores explicit endpoints", () => {
    // Python: Line3D(start=[0,0,0], end=[2,2,2]) -> get_start [0,0,0],
    // get_end [2,2,2].
    const line = new Line3D([0, 0, 0], [2, 2, 2]);
    expectClose(line.get_start(), [0, 0, 0]);
    expectClose(line.get_end(), [2, 2, 2]);
  });
  it("tracks vect/length/direction like set_start_and_end_attrs", () => {
    const line = new Line3D([0, 0, 0], [2, 2, 2]);
    expectClose(line.vect, [2, 2, 2]);
    expectClose([line.length], [Math.sqrt(12)]);
    expectClose(line.direction, [2 / Math.sqrt(12), 2 / Math.sqrt(12), 2 / Math.sqrt(12)]);
  });
  it("keeps a custom thickness and the (2, resolution) default", () => {
    // Python: resolution=24 -> (2, 24).
    const line = new Line3D([0, 0, 0], [1, 0, 0], 0.05);
    expect(line.thickness).toBe(0.05);
    expect(line.resolution).toEqual([2, 24]);
  });
  it("accepts NDArray endpoints like np.array inputs", () => {
    // Python: Line3D(start=np.array([0,0,0]), end=np.array([2,2,2]))
    // (the docstring example passes np arrays).
    const line = new Line3D(array([0, 0, 0]), array([2, 2, 2]));
    expectClose(line.get_start(), [0, 0, 0]);
    expectClose(line.get_end(), [2, 2, 2]);
    expectClose(line.vect, [2, 2, 2]);
  });
  it("takes color in the Python 4th position without shifting resolution", () => {
    // Python signature: (start, end, thickness, color, resolution).
    const line = new Line3D([0, 0, 0], [1, 0, 0], 0.02, "#FF0000");
    expect(line.color).toBe("#FF0000");
    expect(line.resolution).toEqual([2, 24]);
    const line2 = new Line3D([0, 0, 0], [1, 0, 0], 0.02, null, 12);
    expect(line2.resolution).toEqual([2, 12]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.rotation_matrix
// ---------------------------------------------------------------------------

describe("rotation_matrix", () => {
  // Python (scipy Rotation.from_rotvec): rotation_matrix(pi/2, [0,0,1]).
  it("rotates 90 degrees about z", () => {
    expectClose(
      rotation_matrix(Math.PI / 2, [0, 0, 1]),
      [2.220446049250313e-16, -1, 0, 1, 2.220446049250313e-16, 0, 0, 0, 1],
    );
  });
  it("rotates 90 degrees about x", () => {
    // Python: [[1,0,0],[0,2.22e-16,-1],[0,1,2.22e-16]].
    expectClose(
      rotation_matrix(Math.PI / 2, [1, 0, 0]),
      [1, 0, 0, 0, 2.220446049250313e-16, -1, 0, 1, 2.220446049250313e-16],
    );
  });
  it("rotates 180 degrees about y", () => {
    // Python: [[-1,0,1.2246e-16],[0,1,0],[-1.2246e-16,0,-1]].
    expectClose(
      rotation_matrix(Math.PI, [0, 1, 0]),
      [-1, 0, 1.2246467991473532e-16, 0, 1, 0, -1.2246467991473532e-16, 0, -1],
    );
  });
  it("matches scipy for an arbitrary axis", () => {
    // Python: rotation_matrix(0.7, [1,2,3]) ==
    //   [[0.781639173907025, -0.4829292842142122, 0.3947397981737998],
    //    [0.5501172307043584, 0.8320301337746345, -0.07139249941787584],
    //    [-0.29395787843858057, 0.27295633888831433, 0.9160150668873173]].
    // NOTE: TS uses a hand-rolled Rodrigues formula while Python goes through
    // scipy's quaternion path, so last-ulp differences are possible; 1e-9
    // relative tolerance absorbs them.
    expectClose(
      rotation_matrix(0.7, [1, 2, 3]),
      [
        0.781639173907025, -0.4829292842142122, 0.3947397981737998,
        0.5501172307043584, 0.8320301337746345, -0.07139249941787584,
        -0.29395787843858057, 0.27295633888831433, 0.9160150668873173,
      ],
    );
  });
  it("returns identity for a zero axis (from_rotvec(0) == I)", () => {
    // Python: rotation_matrix(0.7, [0,0,0]) == eye(3).
    expectClose(rotation_matrix(0.7, [0, 0, 0]), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
  it("returns identity for a zero angle", () => {
    // Python: rotation_matrix(0.0, [1,2,3]) == eye(3).
    expectClose(rotation_matrix(0.0, [1, 2, 3]), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
  it("flips the sense for a negated axis", () => {
    // Python: rotation_matrix(pi/2, [0,0,-1]) ==
    //   [[2.22e-16,1,0],[-1,2.22e-16,0],[0,0,1]].
    expectClose(
      rotation_matrix(Math.PI / 2, [0, 0, -1]),
      [2.220446049250313e-16, 1, 0, -1, 2.220446049250313e-16, 0, 0, 0, 1],
    );
  });
  it("embeds in a 4x4 for homogeneous=True", () => {
    // Python: eye(4) with [:3,:3] = the 3x3 rotation.
    const m = rotation_matrix(Math.PI / 2, [0, 0, 1], true);
    expect(m.shape).toEqual([4, 4]);
    expectClose(m, [
      2.220446049250313e-16, -1, 0, 0,
      1, 2.220446049250313e-16, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.adjacent_pairs
// ---------------------------------------------------------------------------

describe("adjacent_pairs", () => {
  // Python: list(adjacent_pairs([1,2,3,4])) == [(1,2),(2,3),(3,4),(4,1)].
  // NOTE: Python returns a zip iterator of tuples; the TS port returns an
  // array of arrays (callers compare materialised lists).
  it("pairs cyclically", () => {
    expect(adjacent_pairs([1, 2, 3, 4])).toEqual([[1, 2], [2, 3], [3, 4], [4, 1]]);
  });
  it("wraps a single element onto itself", () => {
    // Python: list(adjacent_pairs([1])) == [(1,1)].
    expect(adjacent_pairs([1])).toEqual([[1, 1]]);
  });
  it("is empty for empty input", () => {
    expect(adjacent_pairs([])).toEqual([]);
  });
  it("works on strings and tuples", () => {
    // Python: list(adjacent_pairs('ab')) == [('a','b'),('b','a')].
    expect(adjacent_pairs("ab")).toEqual([["a", "b"], ["b", "a"]]);
    // Python: list(adjacent_pairs([(1,2),(3,4)])) ==
    //   [((1,2),(3,4)),((3,4),(1,2))].
    expect(adjacent_pairs([[1, 2], [3, 4]])).toEqual([[[1, 2], [3, 4]], [[3, 4], [1, 2]]]);
  });
  it("returns [] for non-positive n like bare zip()", () => {
    // Python: list(adjacent_n_tuples([1,2,3], -1)) == [].
    expect(adjacent_n_tuples([1, 2, 3], -1)).toEqual([]);
    expect(adjacent_n_tuples([1, 2, 3], 0)).toEqual([]);
  });
  it("raises TypeError for non-iterables", () => {
    // Python: adjacent_n_tuples(None, 2) / (123, 2) raise TypeError
    // ("'NoneType'/'int' object is not subscriptable").
    expect(() => adjacent_pairs(null as unknown as [])).toThrow(TypeError);
    expect(() => adjacent_pairs(123 as unknown as [])).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.camera.Camera
// ---------------------------------------------------------------------------

describe("Camera constructor", () => {
  it("derives frame_height from frame_width and aspect", () => {
    // Python: Camera(pixel_height=4, pixel_width=6, frame_width=12,
    //   frame_height=8) keeps 8.0/12 (12/(6/4) == 8).
    const cam = makeCam();
    expect(cam.frame_height).toBe(8.0);
    expect(cam.frame_width).toBe(12);
  });
  it("uses config defaults for the frame shape", () => {
    // Python defaults: frame_width 14.222..., frame_height 8.0
    // (14.222.../(1920/1080) == 8.0).
    const cam = new Camera({ pixel_height: 1080, pixel_width: 1920 });
    expectClose([cam.frame_width], [14.222222222222221]);
    expectClose([cam.frame_height], [8.0]);
  });
  it("builds a black opaque uint8 background", () => {
    // Python: background.shape (4,6,4), dtype uint8, background[0,0] [0,0,0,255].
    const cam = makeCam();
    expect(cam.background.shape).toEqual([4, 6, 4]);
    expect(nums(cam.background)[0]).toEqual(0);
    expect(nums(cam.background).slice(0, 4)).toEqual([0, 0, 0, 255]);
  });
  it("starts the pixel array as a copy of the background", () => {
    // Python: np.array_equal(cam.pixel_array, cam.background) after reset().
    const cam = makeCam();
    expect(nums(cam.pixel_array)).toEqual(nums(cam.background));
  });
  it("honours background color and opacity", () => {
    // Python: Camera(..., background_color='#FF0000',
    //   background_opacity=0.5).background[0,0] == [255,0,0,127].
    const cam = new Camera({
      pixel_height: 2,
      pixel_width: 2,
      frame_width: 8,
      frame_height: 8,
      background_color: "#FF0000",
      background_opacity: 0.5,
    });
    expect(nums(cam.background).slice(0, 4)).toEqual([255, 0, 0, 127]);
  });
  it("exposes rgb_max_val per dtype", () => {
    // Python: uint8 -> 255; uint16 -> 65535.
    expect(makeCam().rgb_max_val).toBe(255);
    expect(new Camera({ pixel_height: 2, pixel_width: 2, pixel_array_dtype: "uint16" }).rgb_max_val).toBe(65535);
  });
  it("sets max_allowable_norm from the configured frame width", () => {
    // Python: Camera(...).max_allowable_norm == config.frame_width (14.222...).
    expectClose([makeCam().max_allowable_norm], [14.222222222222221]);
  });
});

describe("Camera coordinate helpers", () => {
  it("get_coords_of_all_pixels spans the frame", () => {
    // Python: shape (4,6,2); [0,0] [-6,4]; [-1,-1] [4,-2].
    const cam = makeCam();
    const coords = cam.get_coords_of_all_pixels();
    expect(coords.shape).toEqual([4, 6, 2]);
    const rows = coords.toArray() as number[][][];
    expectClose(rows[0]![0]!, [-6, 4]);
    expectClose(rows[3]![5]!, [4, -2]);
  });
  it("points_to_subpixel_coords centres the frame", () => {
    // Python: [[0,0,0],[6,4,0],[-6,-4,0]] -> [[3,2],[6,0],[0,4]].
    const cam = makeCam();
    expectClose(
      cam.points_to_subpixel_coords(makeBox(), array([[0, 0, 0], [6, 4, 0], [-6, -4, 0]])),
      [3, 2, 6, 0, 0, 4],
    );
  });
  it("points_to_pixel_coords truncates", () => {
    // Python: same inputs -> [[3,2],[6,0],[0,4]]; [[1,1,0],[-1,-1,0]] -> [[3,1],[2,2]].
    const cam = makeCam();
    expect(nums(cam.points_to_pixel_coords(makeBox(), array([[1, 1, 0], [-1, -1, 0]])))).toEqual([3, 1, 2, 2]);
  });
  it("on_screen_pixels clips to the viewport", () => {
    // Python: [[0,0],[5,3],[6,0],[0,4],[-1,0]] -> [T,T,F,F,F].
    expect(makeCam().on_screen_pixels(array([[0, 0], [5, 3], [6, 0], [0, 4], [-1, 0]]))).toEqual([
      true, true, false, false, false,
    ]);
  });
  it("get_thickening_nudges matches arange meshgrids", () => {
    // Python: nudges(1) == [[0,0]]; nudges(2) == [[0,0],[0,1],[1,0],[1,1]].
    const cam = makeCam();
    expect(nums(cam.get_thickening_nudges(1))).toEqual([0, 0]);
    expect(nums(cam.get_thickening_nudges(2))).toEqual([0, 0, 0, 1, 1, 0, 1, 1]);
  });
  it("thickened_coordinates offsets per nudge", () => {
    // Python: thickened_coordinates([[1,1]], 2) ==
    //   [[1,1],[1,2],[2,1],[2,2]].
    expect(nums(makeCam().thickened_coordinates(array([[1, 1]]), 2))).toEqual([1, 1, 1, 2, 2, 1, 2, 2]);
  });
  it("transform_points_pre_display passes finite points through", () => {
    // Python: finite -> unchanged; inf -> [[0,0,0]].
    const cam = makeCam();
    expect(nums(cam.transform_points_pre_display(makeBox(), array([[1, 2, 3]])))).toEqual([1, 2, 3]);
    expect(nums(cam.transform_points_pre_display(makeBox(), array([[Infinity, 2, 3]])))).toEqual([0, 0, 0]);
  });
  it("adjust_out_of_range_points rescales violators", () => {
    // Python: [[100,0,0],[1,0,0]] -> [[14.222...,0,0],[1,0,0]].
    expectClose(
      makeCam().adjust_out_of_range_points(array([[100, 0, 0], [1, 0, 0]])),
      [14.222222222222221, 0, 0, 1, 0, 0],
    );
  });
  it("convert_pixel_array scales floats by rgb_max_val", () => {
    // Python: [[[0.5,0.25,1.0,0.0]]] -> [[[127,63,255,0]]].
    expect(nums(makeCam().convert_pixel_array(array([[[0.5, 0.25, 1.0, 0.0]]])))).toEqual([127, 63, 255, 0]);
  });
  it("adjusted_thickness scales against the config pixel sum", () => {
    // Python: Camera(4,6,...).adjusted_thickness(4) == 901.0
    // (1 + 3 * (1080+1920)/10).
    expect(makeCam().adjusted_thickness(4)).toBe(901.0);
  });
});

describe("Camera mobject plumbing", () => {
  it("type_or_raise classifies VMobject/PMobject/Mobject", () => {
    // Python: Square -> VMobject; PMobject -> PMobject.
    const cam = makeCam();
    expect(cam.type_or_raise(makeBox())).toBe(VMobject);
    expect(cam.type_or_raise(new PMobject())).toBe(PMobject);
    expect(cam.type_or_raise(new Mobject())).toBe(Mobject);
  });
  it("get_mobjects_to_display returns the family", () => {
    // Python: a bare Square displays as [Square], with or without submobjects.
    const cam = makeCam();
    expect(cam.get_mobjects_to_display([makeBox()])).toHaveLength(1);
    expect(cam.get_mobjects_to_display([makeBox()], false)).toHaveLength(1);
  });
  it("is_in_frame detects on/off-screen boxes", () => {
    // Python: Square(side 2) at origin True; shifted by 100 in x False.
    const cam = makeCam();
    expect(cam.is_in_frame(makeBox())).toBe(true);
    const far = makeBox().shift([100, 0, 0]);
    expect(cam.is_in_frame(far)).toBe(false);
  });
  it("reset_pixel_shape rebuilds the buffers", () => {
    // Python: reset_pixel_shape(2,3) -> pixel 2x3, array (2,3,4), frame 8.0/12.
    const cam = makeCam();
    cam.reset_pixel_shape(2, 3);
    expect([cam.pixel_height, cam.pixel_width]).toEqual([2, 3]);
    expect(cam.pixel_array.shape).toEqual([2, 3, 4]);
    expectClose([cam.frame_height, cam.frame_width], [8.0, 12]);
    expect(nums(cam.pixel_array)).toEqual(nums(cam.background));
  });
  it("display_point_cloud plots RED points", () => {
    // Python: PMobject with RED points at [0,0,0],[6,4,0] on a 4x6 camera:
    // pixel[2,3] == [252,98,85,255] (on-screen), pixel[0,5] stays [0,0,0,255].
    const cam = makeCam();
    const pm = new PMobject({ stroke_width: 1 });
    pm.set_points(array([[0, 0, 0], [6, 4, 0]]));
    const rgba = RED.to_rgba();
    pm.rgbas = array([[...rgba], [...rgba]]);
    cam.display_point_cloud(pm, pm.points, pm.rgbas, cam.adjusted_thickness(pm.stroke_width), cam.pixel_array);
    const rows = cam.pixel_array.toArray() as number[][][];
    expect([...rows[2]![3]!]).toEqual([252, 98, 85, 255]);
    expect([...rows[0]![5]!]).toEqual([0, 0, 0, 255]);
  });
});

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

describe("batch integration", () => {
  it("rotates Line3D endpoints with rotation_matrix", () => {
    // Python: R = rotation_matrix(0.7, [1,2,3]); R @ [1,0,0] ==
    //   [0.781639173907025, 0.5501172307043584, -0.29395787843858057].
    const R = nums(rotation_matrix(0.7, [1, 2, 3]));
    const at = (r: number, c: number): number => R[r * 3 + c]!;
    const rot = (p: readonly number[]): number[] => [
      at(0, 0) * p[0]! + at(0, 1) * p[1]! + at(0, 2) * p[2]!,
      at(1, 0) * p[0]! + at(1, 1) * p[1]! + at(1, 2) * p[2]!,
      at(2, 0) * p[0]! + at(2, 1) * p[1]! + at(2, 2) * p[2]!,
    ];
    const line = new Line3D([-1, 0, 0], [1, 0, 0]);
    const rotated = new Line3D(rot(nums(line.get_start())), rot(nums(line.get_end())));
    expectClose(rotated.get_end(), [0.781639173907025, 0.5501172307043584, -0.29395787843858057]);
    expectClose(nums(rotated.get_start()), nums(rotated.get_end()).map((v) => -v));
  });
  it("round-trips ticks through LogBase onto camera pixels", () => {
    // Ticks [1,10,100,1000] -> exponents [0,1,2,3] -> back to ticks;
    // the first tick point maps to pixel [3,1] on the 4x6 test camera
    // (same ground truth as points_to_pixel_coords([[1,1,0]])).
    const lb = new LogBase(10);
    const ticks = [1, 10, 100, 1000];
    const exponents = ticks.map((t) => lb.inverse_function(t) as number);
    // NOTE: Math.log(1000)/Math.log(10) is 2.9999999999999996 in V8
    // (Python prints 3.0 for the same ratio); compare with tolerance.
    expectClose(exponents, [0, 1, 2, 3]);
    const cam = makeCam();
    const px = nums(
      cam.points_to_pixel_coords(makeBox(), array([[exponents[1]!, exponents[1]!, 0]])),
    );
    expect(px).toEqual([3, 1]);
  });
  it("drives a mobject chain: family -> frame check -> pixel coords", () => {
    // Python: Square(side 2) family has 1 member, is in frame, and corner
    // [1,1,0] maps to pixel [3,1] on the 4x6 camera.
    const cam = makeCam();
    const box = makeBox();
    const fam = cam.get_mobjects_to_display([box]);
    expect(fam).toHaveLength(1);
    expect(cam.is_in_frame(fam[0]!)).toBe(true);
    // NOTE: point order inside VMobject.points is an interpolation detail
    // (anchors/handles), so locate the known corner instead of assuming
    // it is first.
    const rows = fam[0]!.points.toArray() as number[][];
    const corner = rows.find((r) => r[0] === 1 && r[1] === 1 && r[2] === 0);
    expect(corner).toBeDefined();
    expect(nums(cam.points_to_pixel_coords(fam[0]!, array([corner])))).toEqual([3, 1]);
  });
});
