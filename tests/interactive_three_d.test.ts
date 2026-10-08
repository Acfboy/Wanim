import { describe, expect, test } from "vitest";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Camera } from "../src/camera/camera.js";
import { ThreeDCamera } from "../src/camera/three_d_camera.js";
import { DEGREES } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Slider } from "../src/interactive/controls.js";
import type { InteractiveHost, PointerHandlers, TextEditOptions } from "../src/interactive/host.js";
import { default_camera_class } from "../src/renderer/scene_runner.js";
import { InteractiveScene } from "../src/scene/interactive_scene.js";
import {
  InteractiveThreeDScene,
  ORBIT_MAX_PHI,
  ORBIT_MAX_ZOOM,
  ORBIT_MIN_PHI,
  ORBIT_MIN_ZOOM,
} from "../src/scene/interactive_three_d_scene.js";
import { ThreeDScene } from "../src/scene/three_d_scene.js";

function mockContext(): CanvasRenderingContext2D {
  const noop = (): void => {};
  return {
    setTransform: noop,
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineJoin: "round",
    lineCap: "round",
    beginPath: noop,
    moveTo: noop,
    bezierCurveTo: noop,
    closePath: noop,
    fill: noop,
    stroke: noop,
    fillRect: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
  } as unknown as CanvasRenderingContext2D;
}

function makeRenderer3D(): CanvasSceneRenderer {
  return new CanvasSceneRenderer("", {
    ctx: mockContext(),
    camera_class: ThreeDCamera,
    drawFrames: true,
    recordFrames: false,
  });
}

function pointerEvent(clientX: number, clientY: number): PointerEvent {
  return { clientX, clientY } as unknown as PointerEvent;
}

class Fake3DHost implements InteractiveHost {
  readonly live = true;
  handlers: PointerHandlers | null = null;
  wheelHandler: ((event: WheelEvent) => void) | null = null;
  buttons = new Map<string, () => void>();
  buttonsEnabled = true;
  editor: TextEditOptions | null = null;
  nextPoint: number[] = [0, 0, 0];
  private resolvers: Array<() => void> = [];

  pointFromEvent(): number[] {
    return this.nextPoint;
  }

  onPointer(handlers: PointerHandlers): () => void {
    this.handlers = handlers;
    return () => {
      this.handlers = null;
    };
  }

  onWheel(handler: (event: WheelEvent) => void): () => void {
    this.wheelHandler = handler;
    return () => {
      this.wheelHandler = null;
    };
  }

  addButton(label: string, onClick: () => void, _icon?: unknown): () => void {
    this.buttons.set(label, onClick);
    return () => {};
  }

  setButtonsEnabled(enabled: boolean): void {
    this.buttonsEnabled = enabled;
  }

  onKey(_handlers: unknown): () => void {
    return () => {};
  }

  beginTextEdit(_mobject: unknown, _camera: unknown, options: TextEditOptions): void {
    this.editor = options;
  }

  endTextEdit(): void {
    this.editor = null;
  }

  nextFrame(): Promise<void> {
    return new Promise((resolve) => this.resolvers.push(resolve));
  }

  /** Release `count` pending frames and let the scene task advance. */
  async advance(count = 1): Promise<void> {
    for (let index = 0; index < count; index += 1) {
      const resolve = this.resolvers.shift();
      if (resolve !== undefined) resolve();
      for (let tick = 0; tick < 4; tick += 1) await Promise.resolve();
    }
  }

  click(label: string): void {
    this.buttons.get(label)?.();
  }

  press(point: number[], clientX = 0, clientY = 0): void {
    this.nextPoint = point;
    this.handlers?.down(pointerEvent(clientX, clientY));
  }

  drag(point: number[], clientX = 0, clientY = 0): void {
    this.nextPoint = point;
    this.handlers?.move(pointerEvent(clientX, clientY));
  }

  release(point: number[] = this.nextPoint): void {
    this.nextPoint = point;
    this.handlers?.up(pointerEvent(0, 0));
  }

  wheel(deltaY: number, deltaMode = 0): void {
    this.wheelHandler?.({ deltaY, deltaMode } as unknown as WheelEvent);
  }
}

/** Drive the default blueprint/construct pair to completion. */
async function finishStory(host: Fake3DHost, done: Promise<void> | void): Promise<void> {
  host.click("进入下一步");
  await host.advance(1);
  host.click("进入下一步");
  await host.advance(1);
  await done;
}

describe("InteractiveThreeDScene camera API", () => {
  test("defaults to a ThreeDCamera and borrows set_camera_orientation", () => {
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D() });
    expect(scene.camera).toBeInstanceOf(ThreeDCamera);
    scene.set_camera_orientation(60 * DEGREES, -30 * DEGREES, null, 1.5);
    const camera = scene.camera as ThreeDCamera;
    expect(camera.get_phi()).toBeCloseTo(60 * DEGREES, 12);
    expect(camera.get_theta()).toBeCloseTo(-30 * DEGREES, 12);
    expect(camera.get_zoom()).toBeCloseTo(1.5, 12);
    // The rotation matrix tracks the borrowed accessors (the renderer
    // refreshes it every frame via reset_rotation_matrix).
    camera.reset_rotation_matrix();
    expect(camera.get_rotation_rows()).toEqual(pointRows(camera.generate_rotation_matrix(), 3));
  });

  test("borrows set_to_default_angled_camera_orientation and fixed-in-frame helpers", () => {
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D() });
    scene.set_to_default_angled_camera_orientation();
    const camera = scene.camera as ThreeDCamera;
    expect(camera.get_phi()).toBeCloseTo(70 * DEGREES, 12);
    expect(camera.get_theta()).toBeCloseTo(-135 * DEGREES, 12);

    const label = new Circle(0.5);
    scene.add_fixed_in_frame_mobjects(label);
    expect(camera.fixed_in_frame_mobjects.has(label)).toBe(true);
    scene.remove_fixed_in_frame_mobjects(label);
    expect(camera.fixed_in_frame_mobjects.has(label)).toBe(false);
  });

  test("borrows begin/stop_ambient_camera_rotation into the interactive loop", async () => {
    class AmbientScene extends InteractiveThreeDScene {
      override construct(): void {
        this.begin_ambient_camera_rotation(0.5, "theta");
        this.stop_ambient_camera_rotation("theta");
        this.begin_ambient_camera_rotation(0.5, "theta");
      }
    }
    const host = new Fake3DHost();
    const scene = new AmbientScene({ renderer: makeRenderer3D(), host });
    scene.set_camera_orientation(70 * DEGREES, -45 * DEGREES);
    const camera = scene.camera as ThreeDCamera;
    const theta0 = camera.get_theta();

    const done = scene.render();
    await host.advance(1);
    host.click("进入下一步");
    await host.advance(3);
    expect(camera.get_theta()).not.toBeCloseTo(theta0, 6);
    host.click("进入下一步");
    await host.advance(1);
    await done;
  });

  test("default_camera_class picks ThreeDCamera for the interactive 3D scene", () => {
    class SubScene extends InteractiveThreeDScene {}
    class PlainInteractive extends InteractiveScene {}
    expect(default_camera_class(InteractiveThreeDScene)).toBe(ThreeDCamera);
    expect(default_camera_class(SubScene)).toBe(ThreeDCamera);
    expect(default_camera_class(ThreeDScene)).toBe(ThreeDCamera);
    expect(default_camera_class(PlainInteractive)).toBe(Camera);
  });
});

describe("InteractiveThreeDScene orbit", () => {
  test("background drag turns theta and tips phi with the set sensitivity", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    scene.orbit_sensitivity = 0.01;
    scene.set_camera_orientation(70 * DEGREES, -45 * DEGREES);
    const camera = scene.camera as ThreeDCamera;
    const theta0 = camera.get_theta();
    const phi0 = camera.get_phi();

    const done = scene.render();
    await host.advance(1);
    host.press([5, 5, 0], 100, 100);
    // Two incremental moves: +100 px horizontally, +60 px vertically.
    host.drag([5, 5, 0], 150, 130);
    host.drag([5, 5, 0], 200, 160);
    host.release();

    expect(camera.get_theta()).toBeCloseTo(theta0 - 100 * 0.01, 9);
    expect(camera.get_phi()).toBeCloseTo(phi0 - 60 * 0.01, 9);
    await finishStory(host, done);
  });

  test("phi is clamped strictly inside (0, PI)", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    scene.orbit_sensitivity = 0.1;
    const camera = scene.camera as ThreeDCamera;

    const done = scene.render();
    await host.advance(1);

    scene.set_camera_orientation(0.01, 0);
    host.press([5, 5, 0], 0, 0);
    host.drag([5, 5, 0], 0, 500);
    expect(camera.get_phi()).toBe(ORBIT_MIN_PHI);
    host.drag([5, 5, 0], 0, -5000);
    expect(camera.get_phi()).toBe(ORBIT_MAX_PHI);
    host.release();
    await finishStory(host, done);
  });

  test("disable_orbit_controls / orbit_controls clamp background gestures", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    scene.set_camera_orientation(70 * DEGREES, -45 * DEGREES);
    const camera = scene.camera as ThreeDCamera;
    const theta0 = camera.get_theta();

    const done = scene.render();
    await host.advance(1);
    host.press([5, 5, 0], 100, 100);
    host.drag([5, 5, 0], 300, 100);
    expect(camera.get_theta()).not.toBeCloseTo(theta0, 6);
    host.release();

    scene.disable_orbit_controls();
    const theta1 = camera.get_theta();
    host.press([5, 5, 0], 100, 100);
    host.drag([5, 5, 0], 400, 100);
    host.release();
    expect(camera.get_theta()).toBeCloseTo(theta1, 12);

    scene.enable_orbit_controls();
    expect(scene.orbit_controls).toBe(true);
    await finishStory(host, done);
  });

  test("a control under the pointer wins over the orbit", async () => {
    class SliderScene extends InteractiveThreeDScene {
      slider!: Slider;
      override blueprint(): void {
        this.slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [0, 10] });
        this.add(this.slider);
      }
    }
    const host = new Fake3DHost();
    const scene = new SliderScene({ renderer: makeRenderer3D(), host });
    scene.set_camera_orientation(70 * DEGREES, -45 * DEGREES);
    const camera = scene.camera as ThreeDCamera;
    const theta0 = camera.get_theta();

    const done = scene.render();
    await host.advance(1);
    host.press([0, 0, 0], 100, 100);
    host.drag([2, 0, 0], 500, 400);
    expect(scene.slider.get_value()).toBeCloseTo(10, 9);
    expect(camera.get_theta()).toBeCloseTo(theta0, 12);
    host.release();
    await finishStory(host, done);
  });

  test("a phase transition drops an in-flight orbit drag", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    scene.orbit_sensitivity = 0.01;
    scene.set_camera_orientation(70 * DEGREES, -45 * DEGREES);
    const camera = scene.camera as ThreeDCamera;

    const done = scene.render();
    await host.advance(1);
    host.press([5, 5, 0], 0, 0);
    host.drag([5, 5, 0], 100, 0);
    const theta1 = camera.get_theta();
    // Advance to the construct phase without releasing the pointer.
    host.click("进入下一步");
    await host.advance(1);
    host.drag([5, 5, 0], 400, 0);
    expect(camera.get_theta()).toBeCloseTo(theta1, 12);
    host.release();
    await finishStory(host, done);
  });
});

describe("InteractiveThreeDScene zoom", () => {
  test("wheel zoom is exponential in deltaY and respects the clamp", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    scene.zoom_sensitivity = 0.0015;
    const camera = scene.camera as ThreeDCamera;

    const done = scene.render();
    await host.advance(1);
    const zoom0 = camera.get_zoom();
    host.wheel(100);
    expect(camera.get_zoom()).toBeCloseTo(zoom0 * Math.exp(-100 * 0.0015), 9);

    // Long scrolls clamp instead of reaching zero / infinity.
    for (let index = 0; index < 50; index += 1) host.wheel(1000);
    expect(camera.get_zoom()).toBe(ORBIT_MIN_ZOOM);
    for (let index = 0; index < 50; index += 1) host.wheel(-1000);
    expect(camera.get_zoom()).toBe(ORBIT_MAX_ZOOM);
    await finishStory(host, done);
  });

  test("wheel deltaMode lines/pages are normalized to pixels", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    const camera = scene.camera as ThreeDCamera;
    const done = scene.render();
    await host.advance(1);

    const zoom0 = camera.get_zoom();
    host.wheel(1, 1);
    expect(camera.get_zoom()).toBeCloseTo(zoom0 * Math.exp(-16 * 0.0015), 9);
    const zoom1 = camera.get_zoom();
    host.wheel(1, 2);
    expect(camera.get_zoom()).toBeCloseTo(zoom1 * Math.exp(-100 * 0.0015), 9);
    await finishStory(host, done);
  });
});

describe("InteractiveThreeDScene headless", () => {
  test("runs one blueprint/construct pass and terminates", async () => {
    class Demo extends InteractiveThreeDScene {
      constructed = false;
      override blueprint(): void {
        const axes = new Circle(1);
        this.add(axes);
      }
      override construct(): void {
        this.constructed = true;
      }
    }
    const scene = new Demo({ renderer: makeRenderer3D() });
    const started = Date.now();
    await scene.render();
    expect(scene.constructed).toBe(true);
    expect(scene.mobjects.length).toBe(1);
    expect(Date.now() - started).toBeLessThan(5_000);
    // No host delivered any gesture: the camera keeps its configured angles.
    const camera = scene.camera as ThreeDCamera;
    expect(camera.get_theta()).toBeCloseTo(-90 * DEGREES, 12);
  });

  test("a non-ThreeDCamera subclass does not crash background gestures", async () => {
    const host = new Fake3DHost();
    const scene = new InteractiveThreeDScene({ renderer: makeRenderer3D(), host });
    // Force a plain 2D camera: orbit must be inert rather than throwing.
    (scene as unknown as { camera: Camera }).camera = new Camera();
    const done = scene.render();
    await host.advance(1);
    host.press([5, 5, 0], 0, 0);
    host.drag([5, 5, 0], 100, 100);
    host.wheel(100);
    host.release();
    await finishStory(host, done);
  });
});
