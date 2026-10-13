import { describe, expect, it } from "vitest";
import { array, asarray } from "numpy-ts";
import { ManimColor, color_gradient, color_to_int_rgba } from "../src/utils/color/core.js";
import {
  Axes,
  CoordinateSystem,
  NumberPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

// Ground truths captured from the original Python manim
// (original manim):
//
// ManimColor.__init__:
//   ManimColor(None)                    -> [0, 0, 0, 1]
//   ManimColor(None, alpha=0.25)        -> [0, 0, 0, 0.25]
//   ManimColor(0xFF8000)                -> [1, 0.5019607843137255, 0, 1]
//   ManimColor(0xFF8000, alpha=0.7)     -> [1, 0.5019607843137255, 0, 0.7]
//   ManimColor(True)                    -> [0, 0, 0.00392156862745098, 1] (bool is int)
//   ManimColor(-1)                      -> [1, 1, 1, 1]
//   ManimColor("#FF8000")               -> [1, 0.5019607843137255, 0, 1]
//   ManimColor("0xFF8000")              -> same
//   ManimColor("#F80")                  -> [1, 0.5333333333333333, 0, 1]
//   ManimColor("#FF800080")             -> [1, 0.5019607843137255, 0, 0.5019607843137255]
//   ManimColor("#FF800080", alpha=0.9)  -> same (hex alpha wins)
//   ManimColor("#FF8000", alpha=0.9)    -> [1, 0.5019607843137255, 0, 0.9]
//   ManimColor("RED") / ("red")         -> [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 1]
//   ManimColor("RED", alpha=0.5)        -> [..., 0.5]
//   ManimColor("color #00ff00ish")      -> [0, 1, 0, 1] (hex search)
//   ManimColor(ManimColor("#FF0000"))   -> [1, 0, 0, 1] (alpha param ignored on copy)
//   ManimColor([255, 128, 0])           -> [1, 0.5019607843137255, 0, 1]
//   ManimColor([255, 128, 0], alpha=0.3)-> [..., 0.3]
//   ManimColor([0.1, 0.2, 0.3])         -> [0.1, 0.2, 0.3, 1]
//   ManimColor([0.1, 0.2, 0.3], alpha=0.3) -> [0.1, 0.2, 0.3, 0.3]
//   ManimColor([0.1, 0.2, 0.3, 0.4])    -> [0.1, 0.2, 0.3, 0.4]
//   ManimColor([255, 128, 0, 64])       -> [1, 0.5019607843137255, 0, 0.25098039215686274]
//   ManimColor([255, 0.5, 0])           -> [1, 0.00196078431372549, 0, 1] (mixed -> int path)
//   ManimColor((0.1, 0.2, 0.3)) tuple   -> float path
//   ManimColor(np.array([0.1,0.2,0.3])) -> float path
//   ManimColor(np.array([255,128,0]))   -> int path
//   ManimColor(np.array([255,128,0,64]) -> int rgba path
//   ManimColor(np.array([0.1,0.2,0.3,0.4])) -> float rgba path
//   ManimColor(Fake(get_hex="#00FF00")) -> [0, 1, 0, 1]
//   ManimColor([0.1, 0.2])              -> ValueError "... not 2"
//   ManimColor([1,2,3,4,5])             -> ValueError "... not 5"
//   ManimColor("notacolor")             -> ValueError "Color notacolor not found"
//   ManimColor(3.5)                     -> TypeError "... not <class 'float'>"
//   ManimColor(object())                -> TypeError "... not <class 'object'>"
//   ManimColor._construct_from_space(np.array([0.1,0.2,0.3,0.4])) -> same, type ManimColor
//   copy shares memory in Python (np.shares_memory True); TS port copies (documented deviation).
//
// CoordinateSystem.__rmatmul__ (Axes x_range=[-3,3,1], y_range=[-2,2,1], x_length=6, y_length=4):
//   ax.__matmul__([1,1,0])              -> [1.0000000000000002, 1.0, 0.0]
//   ax.__rmatmul__(c2p(1,1))            -> [1.0, 1.0000000000000004]
//   ax.__rmatmul__([0,0,0])             -> [0.0, 0.0]
//   ax.__rmatmul__((1.0,1.0,0.0))       -> [1.0, 1.0000000000000004]
//   roundtrip (0,0)->[0,0,0]->[0,0]; (1,1)->[1.0000...2,1,0]->[1,1.0000...4];
//     (-2,1.5)->[-1.9999999999999996,1.5,0]->[-1.9999999999999998,1.4999999999999996];
//     (2.5,-1)->[2.5,-1,0]->[2.5,-0.9999999999999992]
//   NumberPlane rmatmul(c2p(2,-1))      -> [2.0, -0.9999999999999996]
//   CoordinateSystem().__rmatmul__      -> NotImplementedError
//
// Integration:
//   VMobject().set_fill(0xFF8000, opacity=0.5): get_fill_color().to_hex() == "#FF8000",
//     get_fill_opacity() == 0.5.

function asNumbers(value: unknown): number[] {
  if (value !== null && typeof value === "object" && "toArray" in value) {
    const raw = (value as { toArray: () => unknown }).toArray();
    const flat = (x: unknown): number[] =>
      Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)];
    return flat(raw);
  }
  if (Array.isArray(value)) {
    const flat = (x: unknown): number[] =>
      Array.isArray(x) ? (x as unknown[]).flatMap(flat) : [Number(x)];
    return flat(value as unknown[]);
  }
  return [Number(value)];
}

function expectClose(actual: unknown, expected: readonly number[] | { toArray(): unknown }, relTol = 1e-9): void {
  const values = asNumbers(actual);
  const exp = Array.isArray(expected) ? expected : (expected as { toArray(): unknown }).toArray() as number[];
  expect(values).toHaveLength(exp.length);
  for (let i = 0; i < exp.length; i += 1) {
    const e = exp[i]!;
    const a = values[i]!;
    const tol = Math.max(1e-12, Math.abs(e) * relTol);
    expect(Math.abs(a - e)).toBeLessThanOrEqual(tol);
  }
}

function expectValueError(fn: () => void, message: string): void {
  let thrown: unknown;
  try {
    fn();
  } catch (e) {
    thrown = e;
  }
  expect(thrown).toBeInstanceOf(Error);
  expect((thrown as Error).name).toBe("ValueError");
  expect((thrown as Error).message).toBe(message);
}

function expectTypeError(fn: () => void, messagePart: string): void {
  let thrown: unknown;
  try {
    fn();
  } catch (e) {
    thrown = e;
  }
  expect(thrown).toBeInstanceOf(TypeError);
  expect((thrown as Error).message).toContain(messagePart);
}

describe("ManimColor.__init__ from None", () => {
  it("defaults to black opaque", () => {
    expectClose(new ManimColor(null)._internal_value, [0, 0, 0, 1]);
    expectClose(new ManimColor(undefined)._internal_value, [0, 0, 0, 1]);
  });
  it("honours an explicit alpha", () => {
    expectClose(new ManimColor(null, 0.25)._internal_value, [0, 0, 0, 0.25]);
  });
});

describe("ManimColor.__init__ from int", () => {
  it("splits 0xRRGGBB", () => {
    expectClose(new ManimColor(0xff8000)._internal_value, [1, 128 / 255, 0, 1]);
  });
  it("honours alpha", () => {
    expectClose(new ManimColor(0xff8000, 0.7)._internal_value, [1, 128 / 255, 0, 0.7]);
  });
  it("treats bool as int (True == 1)", () => {
    expectClose(new ManimColor(true)._internal_value, [0, 0, 1 / 255, 1]);
    expectClose(new ManimColor(false)._internal_value, [0, 0, 0, 1]);
  });
  it("wraps negative ints like Python infinite-precision shifts", () => {
    expectClose(new ManimColor(-1)._internal_value, [1, 1, 1, 1]);
  });
  it("rejects float scalars with TypeError like Python", () => {
    expectTypeError(() => new ManimColor(3.5), "ManimColor only accepts");
    expectTypeError(() => new ManimColor(Number.NaN), "ManimColor only accepts");
  });
});

describe("ManimColor.__init__ from str", () => {
  it("parses # and 0x hex", () => {
    expectClose(new ManimColor("#FF8000")._internal_value, [1, 128 / 255, 0, 1]);
    expectClose(new ManimColor("0xFF8000")._internal_value, [1, 128 / 255, 0, 1]);
  });
  it("expands short hex", () => {
    expectClose(new ManimColor("#F80")._internal_value, [1, 136 / 255, 0, 1]);
  });
  it("takes alpha from 8-digit hex, ignoring the alpha param", () => {
    expectClose(new ManimColor("#FF800080")._internal_value, [1, 128 / 255, 0, 128 / 255]);
    expectClose(new ManimColor("#FF800080", 0.9)._internal_value, [1, 128 / 255, 0, 128 / 255]);
  });
  it("uses the alpha param for 6-digit hex", () => {
    expectClose(new ManimColor("#FF8000", 0.9)._internal_value, [1, 128 / 255, 0, 0.9]);
  });
  it("finds hex embedded in a longer string", () => {
    expectClose(new ManimColor("color #00ff00ish")._internal_value, [0, 1, 0, 1]);
  });
  it("resolves named colors case-insensitively", () => {
    const expected = [0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 1];
    expectClose(new ManimColor("RED")._internal_value, expected);
    expectClose(new ManimColor("red")._internal_value, expected);
  });
  it("applies alpha to named colors", () => {
    expectClose(new ManimColor("RED", 0.5)._internal_value, [
      0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5,
    ]);
  });
  it("raises ValueError for unknown names", () => {
    expectValueError(() => new ManimColor("notacolor"), "Color notacolor not found");
  });
});

describe("ManimColor.__init__ from another ManimColor", () => {
  it("copies the value and ignores the alpha param", () => {
    const src = new ManimColor("#FF0000");
    expectClose(new ManimColor(src)._internal_value, [1, 0, 0, 1]);
    expectClose(new ManimColor(src, 0.3)._internal_value, [1, 0, 0, 1]);
  });
  it("does not alias the source array (documented deviation: Python shares the array)", () => {
    // NOTE: deviation — CPython does `self._internal_value = value._internal_value`
    // (shared reference, np.shares_memory True). The TS port copies so later
    // in-place alpha writes cannot alias another instance.
    const src = new ManimColor("#FF0000");
    const copy = new ManimColor(src);
    copy._internal_value = array([0, 1, 0, 1]) as never;
    expectClose(src._internal_value, [1, 0, 0, 1]);
  });
});

describe("ManimColor.__init__ from lists/tuples/arrays", () => {
  it("takes the int path for all-int length-3 input", () => {
    expectClose(new ManimColor([255, 128, 0])._internal_value, [1, 128 / 255, 0, 1]);
  });
  it("applies alpha on the int path", () => {
    expectClose(new ManimColor([255, 128, 0], 0.3)._internal_value, [1, 128 / 255, 0, 0.3]);
  });
  it("takes the float path for all-float input", () => {
    expectClose(new ManimColor([0.1, 0.2, 0.3])._internal_value, [0.1, 0.2, 0.3, 1]);
    expectClose(new ManimColor([0.1, 0.2, 0.3], 0.3)._internal_value, [0.1, 0.2, 0.3, 0.3]);
  });
  it("takes the float-rgba path for length-4 float input", () => {
    expectClose(new ManimColor([0.1, 0.2, 0.3, 0.4])._internal_value, [0.1, 0.2, 0.3, 0.4]);
  });
  it("takes the int-rgba path for length-4 int input", () => {
    expectClose(new ManimColor([255, 128, 0, 64])._internal_value, [
      1, 128 / 255, 0, 64 / 255,
    ]);
  });
  it("takes the int path for mixed int/float input", () => {
    expectClose(new ManimColor([255, 0.5, 0])._internal_value, [1, 0.5 / 255, 0, 1]);
  });
  it("accepts readonly tuples (float path)", () => {
    const t: readonly number[] = [0.1, 0.2, 0.3];
    expectClose(new ManimColor(t)._internal_value, [0.1, 0.2, 0.3, 1]);
  });
  it("uses dtype for NDArray: float arrays take the float path", () => {
    expectClose(new ManimColor(array([0.1, 0.2, 0.3]))._internal_value, [0.1, 0.2, 0.3, 1]);
    expectClose(new ManimColor(array([0.1, 0.2, 0.3, 0.4]))._internal_value, [
      0.1, 0.2, 0.3, 0.4,
    ]);
  });
  it("uses dtype for NDArray: int arrays take the int path", () => {
    expectClose(new ManimColor(asarray([255, 128, 0], "int32"))._internal_value, [
      1, 128 / 255, 0, 1,
    ]);
  });
  it("raises ValueError for bad lengths", () => {
    expectValueError(
      () => new ManimColor([0.1, 0.2]),
      "ManimColor only accepts lists/tuples/arrays of length 3 or 4, not 2",
    );
    expectValueError(
      () => new ManimColor([1, 2, 3, 4, 5]),
      "ManimColor only accepts lists/tuples/arrays of length 3 or 4, not 5",
    );
  });
});

describe("ManimColor.__init__ from get_hex objects and errors", () => {
  it("parses colors from get_hex() holders", () => {
    const holder = { get_hex: () => "#00FF00" };
    expectClose(new ManimColor(holder)._internal_value, [0, 1, 0, 1]);
  });
  it("raises TypeError for unsupported objects", () => {
    expectTypeError(() => new ManimColor({} as never), "ManimColor only accepts");
  });
});

describe("ManimColor internal plumbing", () => {
  it("exposes _internal_space identical to _internal_value on the base class", () => {
    const c = new ManimColor("RED");
    expectClose(c._internal_space, asNumbers(c._internal_value));
  });
  it("_construct_from_space rebuilds the same color", () => {
    const d = ManimColor._construct_from_space(array([0.1, 0.2, 0.3, 0.4]) as never);
    expect(d).toBeInstanceOf(ManimColor);
    expectClose(d._internal_value, [0.1, 0.2, 0.3, 0.4]);
  });
  it("_internal_value setter validates shape and type", () => {
    const c = new ManimColor("RED");
    let wrongLength: unknown;
    try {
      c._internal_value = array([1, 2, 3]) as never;
    } catch (e) {
      wrongLength = e;
    }
    expect((wrongLength as Error).message).toBe("Array must have exactly 4 values.");
    let notArray: unknown;
    try {
      c._internal_value = [1, 2, 3, 4] as never;
    } catch (e) {
      notArray = e;
    }
    expect((notArray as Error).message).toBe("Value must be a NumPy array.");
  });
});

describe("CoordinateSystem.__rmatmul__", () => {
  function makeAxes(): Axes {
    return new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
  }
  it("inverts coords_to_point at (1,1) (ground truth [1.0, 1.0000000000000004])", () => {
    const ax = makeAxes();
    const pt = ax.coords_to_point(1, 1);
    expectClose(ax.__rmatmul__(asNumbers(pt)), [1.0, 1.0000000000000004]);
  });
  it("maps the origin point back to (0,0)", () => {
    expectClose(makeAxes().__rmatmul__([0, 0, 0]), [0, 0]);
  });
  it("agrees with point_to_coords / p2c on the same input", () => {
    const ax = makeAxes();
    const pt = asNumbers(ax.coords_to_point(-2, 1.5));
    expectClose(ax.__rmatmul__(pt), asNumbers(ax.point_to_coords(pt)));
    expectClose(ax.__rmatmul__(pt), ax.p2c(pt));
  });
  it("accepts NDArray inputs", () => {
    const ax = makeAxes();
    const pt = ax.coords_to_point(1, 1);
    expectClose(ax.__rmatmul__(pt), [1.0, 1.0000000000000004]);
  });
  it("roundtrips several coordinates (Python ground truths)", () => {
    const ax = makeAxes();
    const cases: Array<{ coords: [number, number]; back: [number, number] }> = [
      { coords: [0, 0], back: [0, 0] },
      { coords: [1, 1], back: [1, 1.0000000000000004] },
      { coords: [-2, 1.5], back: [-1.9999999999999998, 1.4999999999999996] },
      { coords: [2.5, -1], back: [2.5, -0.9999999999999992] },
    ];
    for (const { coords, back } of cases) {
      const pt = asNumbers(ax.coords_to_point(coords[0], coords[1]));
      expectClose(ax.__rmatmul__(pt), back);
    }
  });
  it("works on NumberPlane too (ground truth [2.0, -0.9999999999999996])", () => {
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const pt = asNumbers(plane.coords_to_point(2, -1));
    expectClose(plane.__rmatmul__(pt), [2.0, -0.9999999999999996]);
  });
  it("raises NotImplementedError on the abstract base", () => {
    let thrown: unknown;
    try {
      new CoordinateSystem().__rmatmul__([0, 0, 0]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("NotImplementedError");
  });
});

describe("integration: batch symbols with already-ported code", () => {
  it("__matmul__ then __rmatmul__ roundtrips through a real Axes", () => {
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const pt = asNumbers(ax.__matmul__([1, 1, 0]));
    expectClose(pt, [1.0000000000000002, 1.0, 0.0]);
    expectClose(ax.__rmatmul__(pt), [1.0, 1.0000000000000004]);
  });
  it("__matmul__ accepts a Mobject via its center", () => {
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const mob = new VMobject();
    mob.set_points_as_corners([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    // Center is [1, 0, 0]; __matmul__ must treat it as coords (1, 0, 0).
    const viaMob = asNumbers(ax.__matmul__(mob));
    const direct = asNumbers(ax.coords_to_point(1, 0, 0));
    expectClose(viaMob, direct);
  });
  it("ManimColor from int drives VMobject fill (ground truth #FF8000 @ 0.5)", () => {
    const m = new VMobject();
    m.set_fill(new ManimColor(0xff8000), 0.5);
    expect(m.get_fill_color().to_hex()).toBe("#FF8000");
    expect(m.get_fill_opacity()).toBeCloseTo(0.5, 12);
  });
  it("ManimColor copy feeds config-style roundtrips without aliasing", () => {
    const base = new ManimColor("#00FF00");
    const copy = new ManimColor(base);
    expect(copy.to_hex()).toBe("#00FF00");
    const shifted = copy.opacity(0.25);
    expect(shifted.to_hex(true)).toBe("#00FF00" + "3F");
    // Original untouched (Python would share the array; port copies).
    expect(base.to_hex(true)).toBe("#00FF00FF");
  });
});

describe("review regressions: module-function edge fidelity", () => {
  it("color_gradient with length 1 returns the last reference color (Python ground truth ['#58C4DD'])", () => {
    // Python: floors[-1] = num_colors - 2, alphas_mod1[-1] = 1, so a
    // single output is always rgbs[-1], i.e. BLUE here.
    const out = color_gradient(["RED", "GREEN", "BLUE"], 1);
    expect(out).toHaveLength(1);
    expect(out[0]!.to_hex()).toBe("#58C4DD");
  });
  it("color_gradient 5-step matches Python ground truths", () => {
    const out = color_gradient(["RED", "GREEN", "BLUE"], 5);
    expect(out.map((c) => c.to_hex())).toEqual([
      "#FC6255",
      "#BF915E",
      "#83C167",
      "#6DC2A2",
      "#58C4DD",
    ]);
  });
  it("color_to_int_rgba truncates without mod-256 wrapping like to_int_rgba_with_alpha", () => {
    // Python ground truth: (RED + 2).to_int_rgba_with_alpha(0.5) == [762, 608, 595, 127].
    const big = new ManimColor("RED").__add__(2);
    expect(asNumbers(big.to_int_rgba_with_alpha(0.5))).toEqual([762, 608, 595, 127]);
    // In-range behavior unchanged (ground truth [252, 98, 85, 127]).
    expect(color_to_int_rgba("RED", 0.5)).toEqual([252, 98, 85, 127]);
  });
});
