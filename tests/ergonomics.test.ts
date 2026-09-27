import { array, type NDArray } from "numpy-ts";
import { describe, expect, it } from "vitest";
import { RIGHT, UP } from "../src/constants.js";
import { YELLOW, BLUE, RED } from "../src/utils/color/core.js";
import { Mobject, hasNonFinite } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Square, Triangle } from "../src/mobject/geometry/polygram.js";
import { Axes, NumberPlane, ComplexPlane, ThreeDAxes, PolarPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine, UnitInterval } from "../src/mobject/graphing/number_line.js";
import { BarChart } from "../src/mobject/graphing/probability.js";
import { Graph } from "../src/mobject/graph.js";
import { VectorScene } from "../src/scene/vector_space_scene.js";
import { serialize_vmobject } from "../src/renderer/json_renderer.js";

/** Flatten a point buffer to plain rows. */
function rows(value: NDArray): number[][] {
  return (value.toArray() as unknown[]).map((row) =>
    Array.isArray(row) ? (row as number[]).map(Number) : [Number(row)],
  );
}

function expectPointsClose(a: NDArray, b: NDArray, tol = 1e-9): void {
  const ra = rows(a);
  const rb = rows(b);
  expect(ra.length).toBe(rb.length);
  for (let i = 0; i < ra.length; i += 1) {
    expect(ra[i].length).toBe(rb[i].length);
    for (let j = 0; j < ra[i].length; j += 1) {
      expect(Math.abs((ra[i][j] ?? 0) - (rb[i][j] ?? 0))).toBeLessThanOrEqual(tol);
    }
  }
}

describe("graphing options-bag (kwargs) normalization", () => {
  it("new Axes({x_range, y_range, x_length, y_length}) matches positional", () => {
    const opts = new Axes({ x_range: [-3, 3, 1], y_range: [-2, 2, 1], x_length: 6, y_length: 4 });
    const pos = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    expect(opts.x_range).toEqual(pos.x_range);
    expect(opts.y_range).toEqual(pos.y_range);
    expectPointsClose(opts.get_all_points(), pos.get_all_points());
  });

  it("new Axes(x_range, y_range) with a trailing bag matches positional", () => {
    const trailing = new Axes([-3, 3, 1], [-2, 2, 1], { x_length: 6, y_length: 4 });
    const pos = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectPointsClose(trailing.get_all_points(), pos.get_all_points());
  });

  it("new NumberPlane({x_range, y_range}) matches positional", () => {
    const opts = new NumberPlane({ x_range: [-3, 3, 1], y_range: [-2, 2, 1] });
    const pos = new NumberPlane([-3, 3, 1], [-2, 2, 1]);
    expectPointsClose(opts.get_all_points(), pos.get_all_points());
  });

  it("new NumberLine([-3, 3, 1], {length: 6}) matches the options bag form", () => {
    const positional = new NumberLine([-3, 3, 1], { length: 6 });
    const bag = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expect(Array.from(bag.x_range)).toEqual(Array.from(positional.x_range));
    expectPointsClose(positional.get_all_points(), bag.get_all_points());
  });

  it("new UnitInterval / ComplexPlane accept an options bag", () => {
    expect(new UnitInterval().x_range).toEqual([0, 1, 0.1]);
    const plane = new ComplexPlane({ x_range: [-2, 2, 1], y_range: [-2, 2, 1] });
    expect(plane.x_range).toEqual([-2, 2, 1]);
  });

  it("new BarChart({values, ...}) leading bag matches positional form", () => {
    const leading = new BarChart({ values: [1, 2, 3], y_range: [0, 3, 1] });
    const positional = new BarChart([1, 2, 3], { y_range: [0, 3, 1] });
    expect(leading.values).toEqual(positional.values);
    expectPointsClose(leading.get_all_points(), positional.get_all_points());
  });

  it("new Graph({vertices, edges}) leading bag matches positional form", () => {
    const leading = new Graph({ vertices: [1, 2, 3], edges: [[1, 2], [2, 3]] });
    const positional = new Graph([1, 2, 3], [[1, 2], [2, 3]]);
    expect([...leading.vertices.keys()]).toEqual([...positional.vertices.keys()]);
    expect(leading.edges.size).toBe(positional.edges.size);
  });

  it("axes.plot(fn, {color}) matches the positional form and stays finite", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const fn = (x: number): number => 0.5 * x;
    const bag = axes.plot(fn, { color: YELLOW });
    const positional = axes.plot(fn, null, false, null, 1, { color: YELLOW });
    expectPointsClose(bag.get_all_points(), positional.get_all_points());
    expect(hasNonFinite(bag.get_all_points())).toBe(false);
  });

  it("axes.plot(fn, range, {color}) stays finite (regression: silent NaN)", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const graph = axes.plot((x: number) => Math.sin(x), [-1, 1], { color: BLUE });
    expect(hasNonFinite(graph.get_all_points())).toBe(false);
  });

  it("get_area accepts a trailing options bag", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const graph = axes.plot((x: number) => x);
    expect(() => axes.get_area(graph, [-1, 1], { color: RED })).not.toThrow();
  });

  it("get_area bag form is identical to the positional form", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const graph = axes.plot((x: number) => 0.5 * x);
    const bag = axes.get_area(graph, [-1, 1], { color: RED });
    const positional = axes.get_area(graph, [-1, 1], RED, 0.3, null, {});
    expectPointsClose(bag.get_all_points(), positional.get_all_points());
  });

  it("get_riemann_rectangles accepts an options bag", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const graph = axes.plot((x: number) => 0.5 * x);
    const bag = axes.get_riemann_rectangles(graph, [-1, 1], { color: BLUE, stroke_width: 3 });
    const positional = axes.get_riemann_rectangles(graph, [-1, 1], 0.1, "left", 3, BLUE, 1, BLUE, true, null, false, 1.001);
    expect(bag.submobjects.length).toBe(positional.submobjects.length);
    expect(hasNonFinite(bag.get_all_points())).toBe(false);
  });

  it("get_vertical_lines_to_graph accepts an options bag", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const graph = axes.plot((x: number) => 0.5 * x);
    const bag = axes.get_vertical_lines_to_graph(graph, [-1, 1], 5, { color: YELLOW });
    const positional = axes.get_vertical_lines_to_graph(graph, [-1, 1], 5, { color: YELLOW });
    expectPointsClose(bag.get_all_points(), positional.get_all_points());
  });

  it("plot_line_graph accepts a trailing options bag", () => {
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const bag = axes.plot_line_graph([0, 1, 2], [0, 1, 4], { stroke_width: 5 });
    const positional = axes.plot_line_graph([0, 1, 2], [0, 1, 4], null, "PURE_YELLOW", true, 0.08, null, { stroke_width: 5 });
    expectPointsClose(bag.get_all_points(), positional.get_all_points());
  });

  it("new ThreeDAxes({...}) maps z_range and lengths by keyword", () => {
    const axes = new ThreeDAxes({
      x_range: [-2, 2, 1],
      y_range: [-2, 2, 1],
      z_range: [-1, 1, 1],
      num_axis_pieces: 5,
    });
    expect(axes.x_range).toEqual([-2, 2, 1]);
    expect(axes.z_range).toEqual([-1, 1, 1]);
    expect(axes.num_axis_pieces).toBe(5);
  });

  it("new PolarPlane({...}) maps its keywords", () => {
    const plane = new PolarPlane({ radius_max: 3, azimuth_units: "TAU radians", azimuth_step: 8 });
    expect(plane.azimuth_units).toBe("TAU radians");
    expect(plane.azimuth_step).toBe(8);
  });

  it("Mobject.arrange({direction, buff}) matches positional (regression: silent NaN)", () => {
    const build = (): VGroup =>
      new VGroup(new Square(), new Circle(), new Triangle());
    const bag = build();
    bag.arrange({ direction: RIGHT, buff: 0.5 });
    const positional = build();
    positional.arrange(RIGHT, 0.5);
    expect(hasNonFinite(bag.get_all_points())).toBe(false);
    expectPointsClose(bag.get_all_points(), positional.get_all_points());
  });

  it("Mobject.arrange accepts aligned_edge", () => {
    const group = new VGroup(new Square(), new Circle());
    expect(() => group.arrange({ direction: RIGHT, buff: 0.2, aligned_edge: UP })).not.toThrow();
    expect(hasNonFinite(group.get_all_points())).toBe(false);
  });

  it("VectorScene.add_vector(v, {color}) matches positional color", () => {
    const vsBag = new VectorScene();
    const bagArrow = vsBag.add_vector([2, 1, 0], { color: BLUE }, false);
    const vsPos = new VectorScene();
    const posArrow = vsPos.add_vector([2, 1, 0], BLUE, false);
    expectPointsClose(bagArrow.get_all_points(), posArrow.get_all_points());
    expect(bagArrow.get_color().to_hex()).toBe(posArrow.get_color().to_hex());
  });

  it("VectorScene.add_axes / add_plane accept a bag", () => {
    const vs = new VectorScene();
    const axes = vs.add_axes(false, { x_range: [-2, 2, 1] });
    expect(axes.x_range).toEqual([-2, 2, 1]);
    const plane = vs.add_plane({ x_range: [-2, 2, 1] });
    expect(plane.x_range).toEqual([-2, 2, 1]);
  });
});

describe("fail-fast on non-finite geometry", () => {
  it("Mobject.shift(3 * UP) throws the actionable error", () => {
    const circle = new Circle();
    // @ts-expect-error -- deliberately mirrors the documented JS footgun.
    expect(() => circle.shift(3 * UP)).toThrow(/non-finite point\/vector value/);
    expect(() => circle.shift(3 * UP)).toThrow(/JS does not broadcast/);
  });

  it("Mobject.scale / move_to reject non-finite inputs", () => {
    expect(() => new Circle().scale(Number.NaN)).toThrow(/non-finite/);
    expect(() => new Circle().scale(Number.POSITIVE_INFINITY)).toThrow(/non-finite/);
    expect(() => new Circle().move_to([Number.NaN, 0, 0])).toThrow(/non-finite/);
  });

  it("VMobject.set_points / set_anchors_and_handles reject non-finite inputs", () => {
    const vm = new VMobject();
    expect(() => vm.set_points([[Number.NaN, 0, 0]])).toThrow(/non-finite/);
    expect(() =>
      vm.set_anchors_and_handles(
        [[Number.NaN, 0, 0]],
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[1, 1, 0]],
      ),
    ).toThrow(/non-finite/);
  });

  it("the error message names the direct remediation", () => {
    try {
      new Circle().shift(3 * UP);
      throw new Error("expected shift to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain("scalePoint(UP, 3)");
    }
  });

  it("serialize_vmobject refuses to write poisoned points", () => {
    const vm = new Mobject() as unknown as VMobject;
    (vm as unknown as { points: NDArray }).points = array([[Number.NaN, 0, 0]]) as NDArray;
    expect(() => serialize_vmobject(vm)).toThrow(/non-finite points in Mobject/);
  });
});
