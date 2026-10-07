import { describe, expect, test } from "vitest";
import { create_canvas_backend, standard_backend } from "../src/renderer/backend.js";
import { Camera } from "../src/camera/camera.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { BLUE } from "../src/index.js";

interface Call {
  name: string;
  args: unknown[];
}

function mockContext(): {
  ctx: CanvasRenderingContext2D;
  calls: Call[];
  state: Record<string, unknown>;
} {
  const calls: Call[] = [];
  const state: Record<string, unknown> = {};
  const record = (name: string) => (...args: unknown[]) => {
    calls.push({ name, args });
  };
  const ctx = {
    save: record("save"),
    restore: record("restore"),
    setTransform: (...args: number[]) => {
      calls.push({ name: "setTransform", args });
      state.transform = args;
    },
    set fillStyle(value: string) { state.fillStyle = value; calls.push({ name: "fillStyle", args: [value] }); },
    get fillStyle() { return state.fillStyle as string; },
    set strokeStyle(value: string) { state.strokeStyle = value; calls.push({ name: "strokeStyle", args: [value] }); },
    get strokeStyle() { return state.strokeStyle as string; },
    set lineWidth(value: number) { state.lineWidth = value; calls.push({ name: "lineWidth", args: [value] }); },
    get lineWidth() { return state.lineWidth as number; },
    beginPath: record("beginPath"),
    moveTo: record("moveTo"),
    lineTo: record("lineTo"),
    bezierCurveTo: record("bezierCurveTo"),
    closePath: record("closePath"),
    fill: record("fill"),
    stroke: record("stroke"),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls, state };
}

describe("canvas backends", () => {
  test("create_canvas_backend returns the standard backend", () => {
    expect(create_canvas_backend()).toBe(standard_backend());
    expect(create_canvas_backend().name).toBe("standard");
  });

  test("the standard backend paints with the plain Canvas2D API", () => {
    const camera = new Camera({ pixel_width: 100, pixel_height: 100, frame_width: 10, frame_height: 10 });
    const backend = create_canvas_backend();
    const { ctx, calls } = mockContext();
    backend.draw(ctx, camera, new Square(2, { color: BLUE, fill_opacity: 1 }));
    // The square's beziers are replayed directly, without any jittered
    // re-tracing layer.
    expect(calls.some((call) => call.name === "bezierCurveTo")).toBe(true);
    expect(calls.some((call) => call.name === "fill")).toBe(true);
    backend.dispose();
  });
});
