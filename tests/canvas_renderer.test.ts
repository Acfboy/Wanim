import { describe, expect, test } from "vitest";
import { array } from "numpy-ts";
import {
  backgroundCss,
  colorStyle,
  computeSubpathRanges,
  deserialize_vmobject,
  setCanvasPath,
  type MJSONMobject,
} from "../src/renderer/canvas.js";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Camera } from "../src/camera/camera.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

function cubic(a: number[], h1: number[], h2: number[], b: number[]): number[][] {
  return [a, h1, h2, b];
}

describe("computeSubpathRanges", () => {
  test("matches VMobject.get_subpath_split_indices_from_points", () => {
    const vmobject = new VMobject();
    const subpathA = [...cubic([0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]), ...cubic([3, 1, 0], [4, 1, 0], [5, 0, 0], [6, 0, 0])];
    const subpathB = cubic([10, 10, 0], [11, 10, 0], [12, 11, 0], [13, 11, 0]);
    const points = [...subpathA, ...subpathB];
    vmobject.set_points(array(points));
    const expected = (vmobject.get_subpath_split_indices_from_points(vmobject.points, 2).toArray() as number[][]);
    const actual = computeSubpathRanges(points, vmobject.n_points_per_cubic_curve, vmobject.tolerance_for_point_equality, 2);
    expect(actual).toEqual(expected);
    expect(actual).toEqual([[0, 8], [8, 12]]);
  });

  test("closed loop yields one range", () => {
    const points = [
      ...cubic([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]),
      ...cubic([0, 1, 0], [-1, 1, 0], [-1, 0, 0], [0, 0, 0]),
    ];
    expect(computeSubpathRanges(points, 4, 1e-6)).toEqual([[0, 8]]);
  });
});

describe("deserialize_vmobject", () => {
  test("restores geometry and style", () => {
    const record: MJSONMobject = {
      p: [...cubic([0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0])],
      f: [[0.5, 0.25, 0.1, 1]],
      s: [[1, 1, 1, 1]],
      bs: [[0, 0, 0, 0]],
      sw: 6,
      bsw: 3,
      jt: 1,
      cs: 2,
      z: 4,
      bi: null,
    };
    const vmobject = deserialize_vmobject(record);
    expect(vmobject.get_num_points()).toBe(4);
    expect(vmobject.stroke_width).toBe(6);
    expect(vmobject.background_stroke_width).toBe(3);
    expect(vmobject.joint_type).toBe("ROUND");
    expect(vmobject.cap_style).toBe("BUTT");
    expect(vmobject.z_index).toBe(4);
    expect((vmobject.fill_rgbas.toArray() as number[][])[0]).toEqual([0.5, 0.25, 0.1, 1]);
  });
});

interface Call {
  name: string;
  args: number[];
}

function mockContext(): { ctx: CanvasRenderingContext2D; calls: Call[]; state: Record<string, unknown> } {
  const calls: Call[] = [];
  const state: Record<string, unknown> = {};
  const record = (name: string) => (...args: unknown[]) => {
    calls.push({ name, args: args.map(Number) });
  };
  const ctx = {
    calls,
    state,
    setTransform: (...args: number[]) => {
      calls.push({ name: "setTransform", args });
      state.transform = args;
    },
    set fillStyle(value: string | CanvasGradient) { state.fillStyle = value; calls.push({ name: "fillStyle", args: [] }); },
    get fillStyle() { return state.fillStyle as string; },
    set strokeStyle(value: string | CanvasGradient) { state.strokeStyle = value; calls.push({ name: "strokeStyle", args: [] }); },
    get strokeStyle() { return state.strokeStyle as string; },
    set lineWidth(value: number) { state.lineWidth = value; },
    get lineWidth() { return state.lineWidth as number; },
    set lineJoin(value: string) { state.lineJoin = value; },
    get lineJoin() { return state.lineJoin as string; },
    set lineCap(value: string) { state.lineCap = value; },
    get lineCap() { return state.lineCap as string; },
    beginPath: record("beginPath"),
    moveTo: record("moveTo"),
    bezierCurveTo: record("bezierCurveTo"),
    closePath: record("closePath"),
    fill: record("fill"),
    stroke: record("stroke"),
    fillRect: record("fillRect"),
    createLinearGradient: () => ({ addColorStop: () => {} }),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls, state };
}

const pointRows = (value: unknown): number[][] => value as number[][];

describe("setCanvasPath", () => {
  test("emits one moveTo and bezierCurveTo per cubic in a subpath", () => {
    const { ctx, calls } = mockContext();
    const vmobject = new VMobject();
    vmobject.set_points(array([...cubic([0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]), ...cubic([3, 1, 0], [4, 1, 0], [5, 0, 0], [6, 0, 0])]));
    const camera = new Camera({ pixel_width: 100, pixel_height: 100, frame_width: 10, frame_height: 10 });
    setCanvasPath(ctx, camera, vmobject);
    expect(calls.filter((c) => c.name === "beginPath")).toHaveLength(1);
    expect(calls.filter((c) => c.name === "moveTo")).toHaveLength(1);
    expect(calls.filter((c) => c.name === "bezierCurveTo")).toHaveLength(2);
    expect(calls.filter((c) => c.name === "closePath")).toHaveLength(0);
  });

  test("closes a closed subpath", () => {
    const { ctx, calls } = mockContext();
    const square = new Square();
    const camera = new Camera({ pixel_width: 100, pixel_height: 100, frame_width: 10, frame_height: 10 });
    setCanvasPath(ctx, camera, square);
    expect(calls.filter((c) => c.name === "closePath").length).toBeGreaterThan(0);
    expect(calls.filter((c) => c.name === "bezierCurveTo").length).toBe(pointRows(square.points.toArray()).length / 4);
  });
});

describe("colorStyle", () => {
  test("uniform rgba collapses to a flat colour", () => {
    const { ctx } = mockContext();
    const vmobject = new Square();
    const camera = new Camera();
    const style = colorStyle(ctx, camera, [[1, 0.5, 0, 1], [1, 0.5, 0, 1]], vmobject);
    expect(style).toBe("rgba(255, 128, 0, 1)");
  });

  test("multi-colour produces a gradient", () => {
    const { ctx } = mockContext();
    const vmobject = new Square();
    const camera = new Camera();
    const style = colorStyle(ctx, camera, [[1, 0, 0, 1], [0, 0, 1, 1]], vmobject);
    expect(typeof style).toBe("object");
  });
});

describe("backgroundCss", () => {
  test("adds opacity", () => {
    expect(backgroundCss({ background_color: "#123456", background_opacity: 0.5 })).toBe("rgba(18, 52, 86, 0.5)");
  });
});

describe("CanvasSceneRenderer", () => {
  test("extract/draw timing is finite and frames counted", () => {
    const { ctx } = mockContext();
    const renderer = new CanvasSceneRenderer("", { ctx, drawFrames: true, recordFrames: false });
    // A 0x0 object list still exercises get_mobjects_to_display via update_frame.
    const scene = { mobjects: [] } as unknown as Parameters<CanvasSceneRenderer["update_frame"]>[0];
    renderer.update_frame(scene);
    renderer.add_frame(null, 3);
    expect(renderer.stats.frames).toBe(3);
    expect(Number.isFinite(renderer.stats.extractMs)).toBe(true);
    expect(Number.isFinite(renderer.stats.drawMs)).toBe(true);
  });
});
