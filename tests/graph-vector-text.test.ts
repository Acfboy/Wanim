import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CoordinateSystem,
  Axes,
  GenericGraph,
  Graph,
  LayoutFunction,
  LinearTransformationScene,
  StreamLines,
  VectorField,
  VMobject,
  _tree_layout,
} from "../src/index.js";

function caught(callback: () => unknown): unknown {
  try {
    callback();
  } catch (error) {
    return error;
  }
  return undefined;
}

function makeStreamLines(): StreamLines {
  return new StreamLines(() => [0, 0, 0], {
    x_range: [0, 2, 0.5],
    y_range: [0, 2, 0.5],
    z_range: [0, 0, 0.5],
    virtual_time: 0.01,
    dt: 0.01,
    max_anchors_per_line: 2,
    padding: 0.2,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("LinearTransformationScene.get_transposed_matrix_transformation", () => {
  it("matches NumPy dot products for embedded 2D and full 3D matrices", () => {
    const scene = new LinearTransformationScene();
    const transform2D = scene.get_transposed_matrix_transformation([
      [1, 2],
      [3, 4],
    ]);
    const transform3D = scene.get_transposed_matrix_transformation([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    expect(transform2D([1, 2, 0]).toArray()).toEqual([7, 10, 0]);
    expect(transform3D([1, 2, 3]).toArray()).toEqual([30, 36, 42]);
  });

  it("raises ValueError for a matrix with bad dimensions", () => {
    const error = caught(() => new LinearTransformationScene().get_transposed_matrix_transformation([[1, 2, 3]]));

    expect(error).toMatchObject({ name: "ValueError", message: "Matrix has bad dimensions" });
  });
});

describe("graph layout symbols", () => {
  it("matches Python tree coordinates when slide shifts a subtree", () => {
    const graph = Graph._empty_networkx_graph();
    graph.addNodesFrom([0, 1, 2, 3, 4, 5]);
    graph.addEdgesFrom([
      [0, 3],
      [0, 2],
      [0, 1],
      [1, 4],
      [3, 5],
    ]);

    const positions = Object.fromEntries(
      [..._tree_layout(graph, 0, 1)].map(([vertex, point]) => [vertex, point.toArray()]),
    );

    expect(positions).toEqual({
      0: [0, 1, 0],
      1: [-1, 0, 0],
      2: [0, 0, 0],
      3: [1, 0, 0],
      4: [-1, -1, 0],
      5: [1, -1, 0],
    });
  });

  it("calls LayoutFunction for every vertex", () => {
    const graph = Graph._empty_networkx_graph();
    graph.addNodesFrom([2, 1]);
    const layout = new LayoutFunction().__call__(graph, 1);

    expect([...layout.keys()]).toEqual([2, 1]);
    expect([...layout.values()].map((point) => point.toArray())).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("keeps GenericGraph._empty_networkx_graph abstract", () => {
    const error = caught(() => GenericGraph._empty_networkx_graph());

    expect(error).toMatchObject({
      name: "NotImplementedError",
      message: "To be implemented in concrete subclasses",
    });
  });
});

describe("VectorField", () => {
  it("applies a custom color scheme across the configured gradient", () => {
    const field = new VectorField((point) => [point[0], 0, 0], {
      color_scheme: (vector) => vector[0],
      min_color_scheme_value: 0,
      max_color_scheme_value: 2,
      colors: ["#000000", "#FFFFFF"],
    });

    expect(field.color_scheme([3, 4, 0])).toBe(3);
    expect(field.pos_to_rgb([-1, 0, 0])).toEqual([0, 0, 0]);
    expect(field.pos_to_rgb([0.5, 0, 0])).toEqual([0.25, 0.25, 0.25]);
    expect(field.pos_to_rgb([3, 0, 0])).toEqual([1, 1, 1]);
  });

  it("matches the Python RK4 nudge result", () => {
    const field = new VectorField((point) => [point[0] ** 2, 0, 0], { color: "#FFFFFF" });
    const point = new VMobject();
    point.set_points([[1, 2, 3]]);

    expect(field.nudge(point, 0.1, 1, false)).toBe(field);
    expect(point.points.toArray()[0][0]).toBeCloseTo(1.1111104900521944, 12);
    expect(point.points.toArray()[0][1]).toBe(2);
    expect(point.points.toArray()[0][2]).toBe(3);
  });
});

describe("StreamLines", () => {
  it("matches Python outside_box boundaries", () => {
    const stream = makeStreamLines();

    expect(stream.outside_box([-0.2, 0, 0])).toBe(false);
    expect(stream.outside_box([-0.200001, 0, 0])).toBe(true);
    expect(stream.outside_box([2.2, 0, 0])).toBe(false);
    expect(stream.outside_box([2.200001, 0, 0])).toBe(true);
    expect(stream.outside_box([0, 2.2, 0])).toBe(false);
    expect(stream.outside_box([0, 2.200001, 0])).toBe(true);
    const zUpperBound = 0.5 + 0.2 - 0.5;
    expect(stream.outside_box([0, 0, zUpperBound])).toBe(false);
    expect(stream.outside_box([0, 0, zUpperBound + 0.000001])).toBe(true);
  });

  it("wraps time and interpolates lines in the start updater", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.25);
    const stream = makeStreamLines();

    stream.start_animation(false, 2, 0.3);

    const line = stream.stream_lines[0] as VMobject & {
      time: number;
      anim: { run_time: number; interpolate(alpha: number): void };
    };
    line.time = 0.005;
    const interpolate = vi.spyOn(line.anim, "interpolate");

    expect(stream.flow_animation).toBe(stream.updater);
    expect(stream.updaters).toContain(stream.updater);
    stream.updater?.(stream, 0.003);

    expect(line.time).toBeCloseTo(0.001, 12);
    expect(interpolate.mock.calls.some(([alpha]) => Math.abs(alpha - 0.1) < 1e-12)).toBe(true);
  });

  it("exposes the end-animation hide helper", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.25);
    const stream = makeStreamLines();
    stream.start_animation(false);
    stream.end_animation();
    const hideAndWait = stream.hide_and_wait;
    expect(hideAndWait).toBeTypeOf("function");

    stream.stroke_opacity = 0.4;
    hideAndWait?.(stream, 0.5);
    expect(stream.get_stroke_opacity()).toBe(0.4);
    hideAndWait?.(stream, 0);
    expect(stream.get_stroke_opacity()).toBe(0);
    hideAndWait?.(stream, 1);
    expect(stream.get_stroke_opacity()).toBe(1);
  });
});

describe("additional graph and coordinate cases", () => {
  it("lays out directed trees", () => {
    const graph = new (class extends Graph {})([0, 1, 2], [[0, 1], [1, 2]]);
    const positions = Object.fromEntries([..._tree_layout((graph as any)._graph, 0)].map(([key, value]) => [key, value.toArray()]));
    expect(positions).toEqual({ 0: [0, 2, 0], 1: [0, 0, 0], 2: [0, -2, 0] });
  });

  it("maps concrete coordinate vectors and rows", () => {
    // NOTE: Python/TS agree to ~4e-16 here (1.7142857142857153 vs ...157);
    // this is floating-point summation order, so compare with tolerance.
    const axes = new Axes();
    const close = (actual: unknown, expected: number[]) => {
      const values = (actual as { toArray: () => number[] }).toArray();
      expect(values).toHaveLength(expected.length);
      for (let i = 0; i < expected.length; i += 1) expect(values[i]).toBeCloseTo(expected[i], 12);
    };
    const closeRows = (actual: unknown, expected: number[][]) => {
      const rows = (actual as { toArray: () => number[][] }).toArray();
      expect(rows).toHaveLength(expected.length);
      for (let i = 0; i < expected.length; i += 1) {
        expect(rows[i]).toHaveLength(expected[i].length);
        for (let j = 0; j < expected[i].length; j += 1) expect(rows[i][j]).toBeCloseTo(expected[i][j], 12);
      }
    };
    close(axes.coords_to_point(2, 2), [1.7142857142857157, 1.5, 0]);
    close(axes.coords_to_point([1, 2, 0]), [0.8571428571428571, 1.5, 0]);
    closeRows(axes.coords_to_point([[0, 1], [1, 1], [1, 0]]), [
      [0, 0.75, 0],
      [0.8571428571428571, 0.75, 0],
      [0.8571428571428571, 0, 0],
    ]);
    closeRows(axes.coords_to_point([0, 1, 1], [1, 1, 0]), [
      [0, 0.8571428571428571, 0.8571428571428571],
      [0.75, 0.75, 0],
      [0, 0, 0],
    ]);
  });

  it("keeps the default vector-field palette aligned with Python", () => {
    // Ground truth from original manim:
    // VectorField(lambda p: [0,0,0]).pos_to_color(...) == '#236B8E' for any
    // input (norm is always 0 -> first gradient stop, BLUE_E). A varying
    // field reaches the last stop (#FC6255, RED) at norm == 2.
    const field = new VectorField(() => [0, 0, 0]);
    expect(field.pos_to_color([0, 0, 0]).toHex()).toBe("#236B8E");
    expect(field.pos_to_color([0, 2, 0]).toHex()).toBe("#236B8E");
    const varying = new VectorField((point) => [point[0], 0, 0]);
    expect(varying.pos_to_color([0, 0, 0]).toHex()).toBe("#236B8E");
    expect(varying.pos_to_color([2, 0, 0]).toHex()).toBe("#FC6255");
  });
});

// NOTE: manim.mobject.text / tex are OUT OF SCOPE for this port and have no tests.

describe("CoordinateSystem.coords_to_point", () => {
  it("remains abstract in the base coordinate system", () => {
    const system = new CoordinateSystem([0, 2], [-1, 1], 2, 1);
    const error = caught(() => system.coords_to_point(1, -1));

    expect(error).toMatchObject({ name: "NotImplementedError" });
  });
});

describe("VMobject.add_line_to", () => {
  it("adds Python-compatible cubic controls from the last point", () => {
    const line = new VMobject();
    line.set_points([[0, 0, 0]]);

    expect(line.add_line_to([3, 0, 0])).toBe(line);
    expect(line.points.toArray()).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
  });

  it("raises ValueError when no last point exists", () => {
    const error = caught(() => new VMobject().add_line_to([1, 2, 3]));

    expect(error).toMatchObject({ name: "ValueError" });
  });
});
