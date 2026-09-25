import { array, type NDArray } from "numpy-ts";
import { describe, expect, it } from "vitest";
import { pointsFrom } from "../src/utils/arrays.js";
import {
  Group,
  Mobject,
  init_alignments,
  init_size,
  init_sizes,
  repeat_array,
} from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function asRows(values: number[][]): NDArray {
  return array(values) as NDArray;
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (Array.isArray(value)) return value.flatMap(flattenNumbers);
  if (value !== null && typeof value === "object" && "toArray" in value) {
    const toArray = (value as { toArray?: unknown }).toArray;
    if (typeof toArray === "function") return flattenNumbers(toArray.call(value));
  }
  throw new TypeError("Expected a numeric array-like value");
}

function expectNumbersClose(actual: unknown, expected: readonly number[], tolerance = 1e-12): void {
  const values = flattenNumbers(actual);
  expect(values).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(Number.isFinite(values[index])).toBe(true);
    expect(Math.abs(values[index] - expected[index])).toBeLessThanOrEqual(tolerance);
  }
}

function expectRowsClose(actual: NDArray, expected: readonly (readonly number[])[], tolerance = 1e-12): void {
  expect(Array.from(actual.shape)).toEqual([expected.length, expected[0]?.length ?? 0]);
  expectNumbersClose(actual, expected.flat(), tolerance);
}

function expectNamedError(action: () => unknown, name: string, message: string): void {
  let thrown: unknown;
  try {
    action();
  } catch (error) {
    thrown = error;
  }
  expect(thrown).toBeInstanceOf(Error);
  expect({
    name: (thrown as Error).name,
    message: (thrown as Error).message,
  }).toEqual({ name, message });
}

function interpolationPair(): [VMobject, VMobject, VMobject] {
  const first = new VMobject();
  const second = new VMobject();
  const result = new VMobject();
  first.fill_rgbas = asRows([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
  ]);
  second.fill_rgbas = asRows([
    [0.9, 0.8, 0.7, 0.6],
    [0.5, 0.4, 0.3, 0.2],
  ]);
  first.stroke_rgbas = asRows([[1, 0, 0, 0.25]]);
  second.stroke_rgbas = asRows([[0, 0.5, 1, 0.75]]);
  first.background_stroke_rgbas = asRows([[0, 0, 0, 0.1]]);
  second.background_stroke_rgbas = asRows([[0.2, 0.4, 0.6, 0.9]]);
  first.stroke_width = 2;
  second.stroke_width = 8;
  first.background_stroke_width = 1.5;
  second.background_stroke_width = 3.5;
  first.sheen_direction = [1, -2, 3];
  second.sheen_direction = [-4, 5, 6];
  first.sheen_factor = 0.125;
  second.sheen_factor = 0.875;
  return [first, second, result];
}

function proportionalVMobject(): VMobject {
  const mobject = new VMobject();
  mobject.set_points([
    [0, 0, 0],
    [1, 2, 0],
    [2, -1, 0],
    [3, 1, 0],
    [3, 1, 0],
    [4, 4, 0],
    [5, -2, 0],
    [6, 0, 0],
  ]);
  return mobject;
}

function styledVMobject(fillOpacity: number): VMobject {
  const mobject = new VMobject({
    fill_color: [0.2, 0.4, 0.6],
    stroke_color: [0.1, 0.3, 0.5],
  });
  mobject.fill_rgbas = asRows([[0.2, 0.4, 0.6, fillOpacity]]);
  mobject.stroke_rgbas = asRows([[0.1, 0.3, 0.5, 0.25]]);
  return mobject;
}

describe("Mobject._assert_valid_submobjects_internal", () => {
  it("returns the receiver for valid instances of the requested class", () => {
    const root = new Mobject({ name: "root" });
    const child = new Mobject({ name: "child" });
    expect(root._assert_valid_submobjects_internal([child], Mobject)).toBe(root);
    expect(root._assert_valid_submobjects_internal([], Mobject)).toBe(root);
  });

  it("reports Python's type error for a non-Mobject", () => {
    const root = new Mobject({ name: "root" });
    expect(() => root._assert_valid_submobjects_internal([3 as unknown as Mobject], Mobject)).toThrowError(
      "Only values of type Mobject can be added as submobjects of Mobject, but the value 3 (at index 0) is of type int.",
    );
  });

  it("rejects self using Python's value and index", () => {
    const root = new Mobject({ name: "root" });
    expectNamedError(
      () => root._assert_valid_submobjects_internal([root], Mobject),
      "ValueError",
      "Cannot add Mobject as a submobject of itself (at index 0).",
    );
  });

  it("enforces VMobject and preserves Python's Group suggestion", () => {
    const root = new Mobject({ name: "root" });
    const vector = new VMobject({ name: "vector" });
    expect(vector._assert_valid_submobjects_internal([new VMobject()], VMobject)).toBe(vector);
    expect(() => vector._assert_valid_submobjects_internal([root], VMobject)).toThrowError(
      "Only values of type VMobject can be added as submobjects of VMobject, but the value root (at index 0) is of type Mobject. You can try adding this value into a Group instead.",
    );
  });
});

describe("Mobject.interpolate_color", () => {
  it("keeps the base implementation abstract", () => {
    expectNamedError(
      () => new Mobject().interpolate_color(new Mobject(), new Mobject(), 0.5),
      "NotImplementedError",
      "Please override in a child class.",
    );
  });
});

describe("VMobject.interpolate_color", () => {
  it("matches Python for every assigned style attribute", () => {
    const [first, second, result] = interpolationPair();
    expect(result.interpolate_color(first, second, 0.25)).toBe(result);
    expectRowsClose(result.fill_rgbas, [
      [0.30000000000000004, 0.35000000000000003, 0.39999999999999997, 0.45000000000000007],
      [0.5, 0.5499999999999999, 0.5999999999999999, 0.6500000000000001],
    ]);
    expectRowsClose(result.stroke_rgbas, [[0.75, 0.125, 0.25, 0.375]]);
    expectRowsClose(result.background_stroke_rgbas, [[0.05, 0.1, 0.15, 0.30000000000000004]]);
    expectNumbersClose(result.stroke_width, [3.5]);
    expectNumbersClose(result.background_stroke_width, [2]);
    expectNumbersClose(result.sheen_direction, [-0.25, -0.25, 3.75]);
    expectNumbersClose(result.sheen_factor, [0.3125]);
  });

  it("copies alpha-one array attributes instead of aliasing Python's second object", () => {
    const [first, second, result] = interpolationPair();
    result.interpolate_color(first, second, 1);
    expect(result.fill_rgbas).not.toBe(second.fill_rgbas);
    expect(result.stroke_rgbas).not.toBe(second.stroke_rgbas);
    expect(result.background_stroke_rgbas).not.toBe(second.background_stroke_rgbas);
    expect(result.sheen_direction).not.toBe(second.sheen_direction);
    expectRowsClose(result.fill_rgbas, [
      [0.9, 0.8, 0.7, 0.6],
      [0.5, 0.4, 0.3, 0.2],
    ]);
    expectRowsClose(result.stroke_rgbas, [[0, 0.5, 1, 0.75]]);
    expectRowsClose(result.background_stroke_rgbas, [[0.2, 0.4, 0.6, 0.9]]);
    expectNumbersClose(result.stroke_width, [8]);
    expectNumbersClose(result.background_stroke_width, [3.5]);
    expectNumbersClose(result.sheen_direction, [-4, 5, 6]);
    expectNumbersClose(result.sheen_factor, [0.875]);
    result.fill_rgbas.set([0, 0], 0.123);
    result.sheen_direction[0] = -9;
    expectNumbersClose(second.fill_rgbas, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]);
    expectNumbersClose(second.sheen_direction, [-4, 5, 6]);
  });
});

describe("Mobject.has_points", () => {
  it("checks only its own points, matching Python", () => {
    const child = new Mobject();
    const group = new Group(child);
    expect(group.has_points()).toBe(false);
    child.set_points([[1, 2, 3]]);
    expect(child.has_points()).toBe(true);
    expect(group.has_points()).toBe(false);
    child.reset_points();
    expect(child.has_points()).toBe(false);
  });
});

describe("Mobject.point_from_proportion", () => {
  it("keeps the base implementation abstract", () => {
    expectNamedError(
      () => new Mobject().point_from_proportion(0.5),
      "NotImplementedError",
      "Please override in a child class.",
    );
  });

  it.each([
    [-0.1, "Alpha -0.1 not between 0 and 1."],
    [1.1, "Alpha 1.1 not between 0 and 1."],
  ])("rejects alpha %s with Python's ValueError", (alpha, message) => {
    expectNamedError(() => new VMobject().point_from_proportion(alpha), "ValueError", message);
  });

  it("reports Python's no-points ValueError", () => {
    expectNamedError(
      () => new VMobject().point_from_proportion(0.5),
      "ValueError",
      "Cannot call VMobject.point_from_proportion because VMobject has no points.",
    );
  });

  const cases: Array<[number, number[]]> = [
    [0, [0, 0, 0]],
    [0.2, [1.3687070271953794, 0.5648082612628875, 0]],
    [0.5, [3.329202845609931, 1.6849484172801439, 0]],
    [0.73, [4.557769536629362, 0.77346848069687, 0]],
    [1, [6, 0, 0]],
  ];

  it.each(cases)("matches Python's sampled arc length at alpha %s", (alpha, expected) => {
    expectNumbersClose(proportionalVMobject().point_from_proportion(alpha), expected);
  });
});

describe("Mobject.get_array_attrs", () => {
  it("returns only points for Mobject and VMobject", () => {
    expect(new Mobject().get_array_attrs()).toEqual(["points"]);
    expect(new VMobject().get_array_attrs()).toEqual(["points"]);
  });
});

describe("Mobject.get_updaters", () => {
  it("returns the live Python-ordered updater list", () => {
    const first = () => undefined;
    const second = () => undefined;
    const mobject = new Mobject();
    mobject.add_updater(first);
    mobject.add_updater(second, 0);
    expect(mobject.get_updaters()).toBe(mobject.updaters);
    expect(mobject.get_updaters()).toEqual([second, first]);
  });
});

describe("Mobject.repeat_array", () => {
  it("matches Python's axis-zero repetition", () => {
    const source = pointsFrom([
      [0.1, -0.2, 0.3],
      [1.25, 2.5, -3.75],
    ]);
    const repeated = repeat_array(source, 3);
    expect(repeated).not.toBe(source);
    expect(Array.from(repeated.shape)).toEqual([6, 3]);
    expectRowsClose(repeated, [
      [0.1, -0.2, 0.3],
      [1.25, 2.5, -3.75],
      [0.1, -0.2, 0.3],
      [1.25, 2.5, -3.75],
      [0.1, -0.2, 0.3],
      [1.25, 2.5, -3.75],
    ]);
    expectRowsClose(source, [
      [0.1, -0.2, 0.3],
      [1.25, 2.5, -3.75],
    ]);
  });

  it("preserves an empty point-array shape for positive counts", () => {
    const source = new Mobject().points;
    const repeated = repeat_array(source, 4);
    expect(Array.from(repeated.shape)).toEqual([0, 3]);
  });

  it("raises Python's reduce error for a zero count", () => {
    const source = pointsFrom([[1, 2, 3]]);
    expect(() => repeat_array(source, 0)).toThrowError(
      new TypeError("reduce() of empty iterable with no initial value"),
    );
  });
});

describe("Mobject.apply_complex_function", () => {
  it("matches Python's complex mapping and preserves z", () => {
    const mobject = new Mobject();
    mobject.set_points([
      [1, 2, 3],
      [-2, 0.5, 7],
    ]);
    const seen: number[][] = [];
    const result = mobject.apply_complex_function((value) => {
      seen.push([value.re, value.im]);
      const real = value.re * value.re - value.im * value.im + 1;
      const imaginary = 2 * value.re * value.im + 2;
      return { re: real, im: imaginary };
    });
    expect(result).toBe(mobject);
    expect(seen).toEqual([
      [1, 2],
      [-2, 0.5],
    ]);
    expectRowsClose(mobject.points, [
      [-2, 6, 3],
      [4.75, 0, 7],
    ]);
  });
});

describe("Mobject.copy", () => {
  it("deep-copies points, family, style, containers, and updater lists", () => {
    const child = new Mobject({ name: "child", color: [0.2, 0.4, 0.6] });
    child.set_points([
      [0, 0, 0],
      [1, 2, 3],
    ]);
    const nested = new Group(child);
    nested.set_points([
      [-1, -2, -3],
      [4, 5, 6],
    ]);
    const source = new Group(nested);
    source.set_color(child.get_color());
    source.set_points([
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const updater = () => undefined;
    source.add_updater(updater);
    Reflect.set(source, "metadata", { values: [1.25, 2.5] });

    const result = source.copy();
    const copiedNested = result.submobjects[0] as Group;
    const copiedChild = copiedNested.submobjects[0] as Mobject;
    const copiedMetadata = Reflect.get(result, "metadata") as { values: number[] };
    const sourceMetadata = Reflect.get(source, "metadata") as { values: number[] };

    expect(result).toBeInstanceOf(Group);
    expect(result).not.toBe(source);
    expect(result.original_id).not.toBe(source.original_id);
    expect(result.points).not.toBe(source.points);
    expect(copiedNested).not.toBe(nested);
    expect(copiedChild).not.toBe(child);
    expect(copiedChild.points).not.toBe(child.points);
    expect(result.color).not.toBe(source.color);
    expect(result.updaters).not.toBe(source.updaters);
    expect(result.updaters[0]).toBe(updater);
    expect(copiedMetadata).not.toBe(sourceMetadata);
    expect(copiedMetadata.values).not.toBe(sourceMetadata.values);
    expectNumbersClose(result.color.internal_value, [0.2, 0.4, 0.6, 1]);

    result.points.set([0, 0], 99);
    copiedChild.points.set([0, 1], -99);
    copiedMetadata.values[0] = -1.25;
    result.color.internal_value.set([0], 0.9);
    expectRowsClose(source.points, [
      [7, 8, 9],
      [10, 11, 12],
    ]);
    expectRowsClose(child.points, [
      [0, 0, 0],
      [1, 2, 3],
    ]);
    expectNumbersClose(sourceMetadata.values, [1.25, 2.5]);
    expectNumbersClose(source.color.internal_value, [0.2, 0.4, 0.6, 1]);
  });
});

describe("Mobject.arrange_in_grid helpers", () => {
  it("implements init_size precedence and iterable lengths", () => {
    expect(init_size(null, null, null)).toBeNull();
    expect(init_size(2, "ucl", [1, 2, 3, 4])).toBe(2);
    expect(init_size(null, "ucl", [1, 2, 3, 4])).toBe(3);
    expect(init_size(null, null, new Set([1, 2, 3]))).toBe(3);
    expect(init_size(null, null, (function* () {
      yield 1;
      yield 2;
    })())).toBe(2);
    expect(init_size(null, null, [])).toBe(0);
  });

  it("implements init_sizes fallback and explicit values", () => {
    expect(init_sizes(null, 3, [1.25, 2.5, 3.75], "row_heights")).toEqual([1.25, 2.5, 3.75]);
    expect(init_sizes([], 2, [4.5, 5.5], "col_widths")).toEqual([4.5, 5.5]);
    expect(init_sizes([1, null, 3], 3, [9, 8, 7], "row_heights")).toEqual([1, 8, 3]);
    expect(() => init_sizes([1], 2, [9, 8], "row_heights")).toThrowError(
      "row_heights has a mismatching size.",
    );
  });

  it("implements init_alignments fallback, explicit directions, and validation", () => {
    const vertical = { u: [0, 1, 0], c: [0, 0, 0], d: [0, -1, 0] };
    const horizontal = { l: [-1, 0, 0], c: [0, 0, 0], r: [1, 0, 0] };
    expect(
      init_alignments(null, 3, vertical, "row", [1, 0, 0], [0.25, -0.5, 0.75]),
    ).toEqual([
      [0.25, -0, 0],
      [0.25, -0, 0],
      [0.25, -0, 0],
    ]);
    expect(init_alignments("ucd", 3, vertical, "row", [1, 0, 0])).toEqual([
      [0, 1, 0],
      [0, 0, 0],
      [0, -1, 0],
    ]);
    expect(init_alignments("lcr", 3, horizontal, "col", [0, 1, 0])).toEqual([
      [-1, 0, 0],
      [0, 0, 0],
      [1, 0, 0],
    ]);
    expect(() => init_alignments("uu", 3, vertical, "row", [1, 0, 0])).toThrowError(
      "row_alignments has a mismatching size.",
    );
    expect(() => init_alignments("x", 1, horizontal, "col", [0, 1, 0])).toThrow();
  });
});

describe("Mobject.set_z_index", () => {
  it("recurses by default and supports non-integral values", () => {
    const grandchild = new Mobject();
    const child = new Group(grandchild);
    const root = new Group(child);
    expect(root.set_z_index(2.5)).toBe(root);
    expect([root.z_index, child.z_index, grandchild.z_index]).toEqual([2.5, 2.5, 2.5]);
    root.set_z_index(-1.25);
    expect([root.z_index, child.z_index, grandchild.z_index]).toEqual([-1.25, -1.25, -1.25]);
  });

  it("leaves descendants unchanged when family is false", () => {
    const child = new Mobject();
    const root = new Group(child);
    root.set_z_index(7.5, false);
    expect(root.z_index).toBe(7.5);
    expect(child.z_index).toBe(0);
  });
});

describe("VMobject.n_points_per_curve", () => {
  it("mirrors n_points_per_cubic_curve", () => {
    const standard = new VMobject();
    expect(standard.n_points_per_curve).toBe(4);
    const custom = new VMobject({ n_points_per_cubic_curve: 7 });
    expect(custom.n_points_per_cubic_curve).toBe(7);
    expect(custom.n_points_per_curve).toBe(7);
    custom.n_points_per_cubic_curve = 6;
    expect(custom.n_points_per_curve).toBe(6);
  });
});

describe("VMobject.has_new_path_started", () => {
  it("is true exactly when the point count is one modulo four", () => {
    for (let count = 0; count <= 10; count += 1) {
      const mobject = new VMobject();
      mobject.set_points(Array.from({ length: count }, (_, index) => [index, index * index, 0]));
      expect(mobject.has_new_path_started()).toBe(count % 4 === 1);
    }
  });
});

describe("VMobject.reverse_direction", () => {
  it("reverses all control points and returns self", () => {
    const source = [
      [0, 0, 0],
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ];
    const mobject = new VMobject();
    mobject.set_points(source);
    expect(mobject.reverse_direction()).toBe(mobject);
    expectRowsClose(mobject.points, [...source].reverse());
    mobject.reverse_direction();
    expectRowsClose(mobject.points, source);
  });
});

describe("Mobject.__getattr__", () => {
  it("creates Python-compatible dynamic getter and setter methods", () => {
    const mobject = new Mobject({ name: "dynamic" });
    Reflect.set(mobject, "foo", 0.375);
    const getter = mobject.__getattr__("get_foo") as () => unknown;
    const setter = mobject.__getattr__("set_bar") as (value: unknown) => Mobject;
    expect(getter()).toBe(0.375);
    expect(setter({ value: 7 })).toBe(mobject);
    expect(Reflect.get(mobject, "bar")).toEqual({ value: 7 });
  });

  it("supports the same compatibility layer through property access", () => {
    const mobject = new Mobject();
    const dynamicGetter = Reflect.get(mobject, "get_answer") as () => unknown;
    const dynamicSetter = Reflect.get(mobject, "set_answer") as (value: unknown) => Mobject;
    expect(dynamicSetter(42) === mobject).toBe(true);
    expect(dynamicGetter()).toBe(42);
  });

  it("raises Python's AttributeError for an unknown name", () => {
    expectNamedError(
      () => new Mobject().__getattr__("missing"),
      "AttributeError",
      "Mobject object has no attribute 'missing'",
    );
  });
});

describe("Mobject and VMobject get_color", () => {
  it("returns the exact Mobject color object", () => {
    const mobject = new Mobject({ color: [0.2, 0.4, 0.6] });
    expect(mobject.get_color()).toBe(mobject.color);
    expectNumbersClose(mobject.get_color().toRgba(), [0.2, 0.4, 0.6, 1]);
  });

  it.each([
    [0.75, [0.2, 0.4, 0.6, 1]],
    [0, [0.1, 0.3, 0.5, 1]],
  ])("uses Python's VMobject fill/stroke rule at fill opacity %s", (fillOpacity, expected) => {
    expectNumbersClose(styledVMobject(fillOpacity).get_color().toRgba(), expected);
  });
});

describe("additional Mobject edge cases", () => {
  it("accepts numeric returns from complex functions", () => {
    const mobject = new Mobject();
    mobject.set_points([[1, 2, 3], [-2, 0.5, 7]]);
    expect(mobject.apply_complex_function(() => 2)).toBe(mobject);
    expect(mobject.points.toArray()).toEqual([[2, 0, 3], [2, 0, 7]]);
  });

  it("rejects non-integral repeat counts", () => {
    const mobject = new Mobject();
    mobject.set_points([[1, 2, 3]]);
    expect(() => mobject.repeat(2.5)).toThrow(TypeError);
  });

  it("matches Python's repeat error for a fractional count", () => {
    // Ground truth: Python `[array] * 2.5` raises
    // "TypeError: can't multiply sequence by non-int of type 'float'".
    const mobject = new Mobject();
    mobject.set_points([[1, 2, 3]]);
    expect(() => mobject.repeat(2.5)).toThrowError(
      "can't multiply sequence by non-int of type 'float'",
    );
  });

  it("raises Python ValueError names for grid helper mismatches", () => {
    expectNamedError(
      () => init_sizes([1], 2, [9, 8], "row_heights"),
      "ValueError",
      "row_heights has a mismatching size.",
    );
    expectNamedError(
      () => init_alignments("uu", 3, { u: [0, 1, 0] }, "row", [1, 0, 0]),
      "ValueError",
      "row_alignments has a mismatching size.",
    );
    expectNamedError(
      () => init_alignments("x", 1, { c: [0, 0, 0] }, "col", [0, 1, 0]),
      "KeyError",
      "'x'",
    );
  });

  it("uses Python's all-zero fill rule for mixed VMobject opacities", () => {
    // Ground truth: Python `get_color` returns the stroke color only when
    // `np.all(fill_opacities == 0)`; a partially transparent fill still
    // yields the fill color.
    const mobject = new VMobject({
      fill_color: [0.2, 0.4, 0.6],
      stroke_color: [0.1, 0.3, 0.5],
    });
    mobject.fill_rgbas = asRows([
      [0.2, 0.4, 0.6, 0],
      [0.2, 0.4, 0.6, 0.5],
    ]);
    expectNumbersClose(mobject.get_color().toRgba(), [0.2, 0.4, 0.6, 1]);
    mobject.fill_rgbas = asRows([
      [0.2, 0.4, 0.6, 0],
      [0.2, 0.4, 0.6, 0],
    ]);
    expectNumbersClose(mobject.get_color().toRgba(), [0.1, 0.3, 0.5, 1]);
  });

  it("preserves a point-array width wider than three", () => {
    // Ground truth: Python Mobject has no set_points; points are assigned
    // directly (m.points = [[1,2,3,4]]) and repeat() preserves the width, so
    // the TS test mirrors that with a direct points assignment instead of the
    // dim-truncating set_points() helper.
    const mobject = new Mobject();
    mobject.points = array([[1, 2, 3, 4]]) as NDArray;
    mobject.repeat(2);
    expect(mobject.points.shape).toEqual([2, 4]);
    expect(mobject.points.toArray()).toEqual([[1, 2, 3, 4], [1, 2, 3, 4]]);
  });

  it("keeps a fresh VMobject's fill transparent and returns stroke color", () => {
    const mobject = new VMobject({ color: "#123456" });
    expectNumbersClose(mobject.get_color().toRgba(), [0.07058823529411765, 0.20392156862745098, 0.3372549019607843, 1]);
  });
});
