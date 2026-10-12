import { describe, expect, test, vi } from "vitest";
import { array } from "numpy-ts";
import {
  applyMobject3DFlags,
  backgroundCss,
  clearCanvas,
  colorStyle,
  computeSubpathRanges,
  configureFrameCamera,
  deserialize_vmobject,
  drawFrame,
  drawVmobject,
  isThreeDFrame,
  pixelArrayToRgbaBytes,
  PreviewThreeDCamera,
  setCanvasPath,
  type MJSONFrame,
  type MJSONMobject,
} from "../src/renderer/canvas.js";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { serialize_frame } from "../src/renderer/json_renderer.js";
import { ImageMobject } from "../src/mobject/types/image_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Cube } from "../src/mobject/three_d/three_dimensions.js";
import { Camera } from "../src/camera/camera.js";
import { ThreeDCamera } from "../src/camera/three_d_camera.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { BLUE, DEGREES } from "../src/index.js";

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

function mockContext(): {
  ctx: CanvasRenderingContext2D;
  calls: Call[];
  state: Record<string, unknown>;
  fillStyles: unknown[];
  drawImageArgs: unknown[][];
} {
  const calls: Call[] = [];
  const state: Record<string, unknown> = {};
  const fillStyles: unknown[] = [];
  const drawImageArgs: unknown[][] = [];
  const record = (name: string) => (...args: unknown[]) => {
    calls.push({ name, args: args.map(Number) });
  };
  const ctx = {
    calls,
    state,
    fillStyles,
    drawImageArgs,
    setTransform: (...args: number[]) => {
      calls.push({ name: "setTransform", args });
      state.transform = args;
    },
    set fillStyle(value: string | CanvasGradient) { state.fillStyle = value; fillStyles.push(value); calls.push({ name: "fillStyle", args: [] }); },
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
    save: record("save"),
    restore: record("restore"),
    transform: record("transform"),
    drawImage: (...args: unknown[]) => {
      calls.push({ name: "drawImage", args: [] });
      drawImageArgs.push(args);
    },
    createLinearGradient: () => ({ addColorStop: () => {} }),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls, state, fillStyles, drawImageArgs };
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

describe("clearCanvas style cache", () => {
  test("re-writes the stroke after the context was reset between runs", () => {
    const { ctx, calls, state } = mockContext();
    const camera = new Camera({ pixel_width: 100, pixel_height: 100, frame_width: 10, frame_height: 10 });
    const square = new Square(2, { color: BLUE });
    clearCanvas(ctx, camera, "#000000");
    drawVmobject(ctx, camera, square);
    const afterFirst = calls.filter((call) => call.name === "strokeStyle").length;
    expect(afterFirst).toBeGreaterThan(0);
    // `canvas.width = ...` at the start of a rerun resets the real context to
    // its default styles; the cached stroke must not survive that.
    state.strokeStyle = "#000000";
    clearCanvas(ctx, camera, "#000000");
    drawVmobject(ctx, camera, square);
    expect(calls.filter((call) => call.name === "strokeStyle").length).toBeGreaterThan(afterFirst);
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

  test("fails loudly on PMobject instead of drawing a blank frame", () => {
    const { ctx } = mockContext();
    const renderer = new CanvasSceneRenderer("", { ctx, drawFrames: true, recordFrames: false });
    const pmobject = new PMobject();
    pmobject.add_points([[0, 0, 0]]);
    const scene = { mobjects: [pmobject] } as unknown as Parameters<CanvasSceneRenderer["update_frame"]>[0];
    let thrown: unknown = null;
    try {
      renderer.update_frame(scene);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("NotImplementedError");
    expect((thrown as Error).message).toMatch(/PMobject is not supported/);
  });

  test("draws an ImageMobject through a cached scratch canvas", () => {
    const putImageData = vi.fn();
    const scratchCanvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        createImageData: (width: number, height: number) => ({
          data: new Uint8ClampedArray(width * height * 4),
          width,
          height,
        }),
        putImageData,
      }),
    };
    vi.stubGlobal("document", { createElement: () => scratchCanvas });
    try {
      const { ctx, drawImageArgs } = mockContext();
      const renderer = new CanvasSceneRenderer("", { ctx, drawFrames: true, recordFrames: false });
      const image = new ImageMobject([[0, 1], [0.5, 1]]);
      const scene = { mobjects: [image] } as unknown as Parameters<CanvasSceneRenderer["update_frame"]>[0];
      renderer.update_frame(scene);
      expect(drawImageArgs).toHaveLength(1);
      expect(drawImageArgs[0]?.[0]).toBe(scratchCanvas);
      expect(drawImageArgs[0]?.slice(1)).toEqual([0, 0, 2, 2]);
      // Second frame reuses the cached scratch canvas (no re-upload).
      renderer.update_frame(scene);
      expect(drawImageArgs).toHaveLength(2);
      expect(putImageData).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("pixelArrayToRgbaBytes", () => {
  test("grayscale byte arrays replicate to RGB with opaque alpha", () => {
    const { data, width, height } = pixelArrayToRgbaBytes(array([[0, 128], [255, 64]]));
    expect([width, height]).toEqual([2, 2]);
    expect([...data]).toEqual([
      0, 0, 0, 255,
      128, 128, 128, 255,
      255, 255, 255, 255,
      64, 64, 64, 255,
    ]);
  });

  test("float arrays are scaled from 0..1", () => {
    const { data } = pixelArrayToRgbaBytes(array([[0, 1], [0.5, 1]]));
    expect(data[0]).toBe(0);
    expect(data[3]).toBe(255);
    expect(data[4]).toBe(255);
    expect(data[8]).toBe(128);
  });

  test("byte RGBA arrays pass through unchanged", () => {
    const rgba = array([[[10, 20, 30, 40], [50, 60, 70, 80]]]).astype("uint8");
    const { data } = pixelArrayToRgbaBytes(rgba);
    expect([...data]).toEqual([10, 20, 30, 40, 50, 60, 70, 80]);
  });
});

function threeDFrame(overrides: Partial<MJSONFrame["cam"]> = {}): MJSONFrame {
  return {
    cam: {
      fc: [0, 0, 0], fw: 14, fh: 8, "3d": true,
      fd: 20, phi: 65 * DEGREES, theta: -45 * DEGREES, gamma: 0, zoom: 1,
      ls: [-7, -9, 10], sh: 0.2, sa: true, ep: false,
      ...overrides,
    },
    m: [],
  };
}

describe("3D canvas rendering", () => {
  test("isThreeDFrame only matches 3d cameras", () => {
    expect(isThreeDFrame(threeDFrame())).toBe(true);
    expect(isThreeDFrame({ cam: { fc: [0, 0, 0], fw: 14, fh: 8 }, m: [] })).toBe(false);
  });

  test("configureFrameCamera restores the 3D camera state", () => {
    const cam = new PreviewThreeDCamera();
    configureFrameCamera(cam, threeDFrame());
    expect(cam.get_phi()).toBeCloseTo(65 * DEGREES, 12);
    expect(cam.get_theta()).toBeCloseTo(-45 * DEGREES, 12);
    expect(cam.get_focal_distance()).toBe(20);
    expect(cam.get_zoom()).toBe(1);
    expect(cam.should_apply_shading).toBe(true);
    expect(cam.exponential_projection).toBe(false);
    // Rotation matrix was regenerated from the restored angles.
    const expected = new ThreeDCamera();
    expected.set_phi(65 * DEGREES);
    expected.set_theta(-45 * DEGREES);
    expected.reset_rotation_matrix();
    expect(cam.get_rotation_matrix().toArray()).toEqual(expected.get_rotation_matrix().toArray());
  });

  test("applyMobject3DFlags restores s3d / fif / fo", () => {
    const cam = new PreviewThreeDCamera();
    const vmobject = new Square(1);
    applyMobject3DFlags(cam, vmobject, { p: [], sw: 0, bsw: 0, jt: 0, cs: 0, z: 0, s3d: true, fif: true, fo: [2, 0, 2] });
    expect((vmobject as { shade_in_3d?: boolean }).shade_in_3d).toBe(true);
    expect(cam.fixed_in_frame_mobjects.has(vmobject)).toBe(true);
    const func = cam.fixed_orientation_mobjects.get(vmobject);
    expect(func?.()).toEqual([2, 0, 2]);
  });

  test("drawFrame projects 3D points and shades shade_in_3d faces", () => {
    const cam = new PreviewThreeDCamera();
    const cube = new Cube(2);
    cube.set_fill(BLUE, 1);
    cam.set_phi(65 * DEGREES);
    cam.set_theta(-45 * DEGREES);
    const frame = serialize_frame(cam, [cube]) as unknown as MJSONFrame;
    // Cube faces are marked shade_in_3d in the serialized records.
    expect(frame.m.some((record) => record.s3d === true)).toBe(true);

    const { ctx, calls, fillStyles } = mockContext();
    drawFrame(ctx, cam, frame);
    // The projected x of a raw corner differs from its world coordinate, i.e. a
    // projection was actually applied before drawing.  (A Cube stores its
    // geometry in face submobjects, not on the top-level VGroup.)
    const face = cube.get_family().find(
      (member) => member instanceof VMobject && member.get_num_points() > 0,
    ) as VMobject;
    const rawX = (face.points.toArray() as number[][])[0]![0]!;
    const moveX = calls.find((call) => call.name === "moveTo")?.args[0];
    expect(moveX).not.toBe(rawX);
    // Shaded faces have two distinct rgba rows, so their fill is a gradient
    // (an object) rather than a flat rgba string.
    expect(fillStyles.some((style) => typeof style === "object")).toBe(true);
  });
});
