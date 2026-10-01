import { describe, expect, test } from "vitest";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Camera } from "../src/camera/camera.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { InputDecimalNumber, Slider, Switch } from "../src/interactive/controls.js";
import type { InteractiveHost, PointerHandlers, TextEditOptions } from "../src/interactive/host.js";
import { InteractiveScene } from "../src/scene/interactive_scene.js";
import { Scene } from "../src/scene/scene.js";
import { Wait } from "../src/animation/animation.js";
import {
  create_frame_scheduler,
  set_render_scheduler,
  type RenderScheduler,
} from "../src/utils/scheduler.js";

function mockContext(): CanvasRenderingContext2D {
  const noop = (): void => {};
  const ctx = {
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
  return ctx;
}

function makeRenderer(): CanvasSceneRenderer {
  return new CanvasSceneRenderer("", { ctx: mockContext(), drawFrames: true, recordFrames: false });
}

class FakeHost implements InteractiveHost {
  readonly live = true;
  handlers: PointerHandlers | null = null;
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

  addButton(label: string, onClick: () => void, _icon?: unknown): () => void {
    this.buttons.set(label, onClick);
    // Buttons stay resident; only the handler is detached.
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

  get pendingFrames(): number {
    return this.resolvers.length;
  }

  click(label: string): void {
    this.buttons.get(label)?.();
  }

  press(point: number[]): void {
    this.nextPoint = point;
    this.handlers?.down(new Event("pointerdown") as unknown as PointerEvent);
  }

  drag(point: number[]): void {
    this.nextPoint = point;
    this.handlers?.move(new Event("pointermove") as unknown as PointerEvent);
  }

  release(point: number[] = this.nextPoint): void {
    this.nextPoint = point;
    this.handlers?.up(new Event("pointerup") as unknown as PointerEvent);
  }
}

describe("Slider", () => {
  test("maps its value range onto the track", () => {
    const slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [1, 3], init_value: 1 });
    expect(slider.get_value()).toBe(1);
    slider.set_value(3);
    expect(slider.get_value()).toBe(3);
    expect(slider.knob.get_center()[0]).toBeCloseTo(2, 12);
    slider.set_value(2);
    expect(slider.knob.get_center()[0]).toBeCloseTo(0, 12);
  });

  test("clamps values and dragging picks the nearest track point", () => {
    const slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [0, 10], init_value: 0 });
    slider.set_value(100);
    expect(slider.get_value()).toBe(10);
    const context = { host: { live: false } as unknown as InteractiveHost, camera: new Camera() };
    slider.on_press([0, 1, 0], context);
    expect(slider.get_value()).toBeCloseTo(5, 12);
  });

  test("hit testing only fires near the track", () => {
    const slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [0, 1] });
    expect(slider.is_hit([0, 0.1, 0])).toBe(true);
    expect(slider.is_hit([0, 1, 0])).toBe(false);
    expect(slider.is_hit([5, 0, 0])).toBe(false);
  });

  test("opacity options, attach hides the knob and follows the value", () => {
    const slider = new Slider({
      start: [-2, 0, 0],
      end: [2, 0, 0],
      value_range: [0, 1],
      track_opacity: 0.2,
      knob_opacity: 0.8,
    });
    expect(slider.track.get_stroke_opacity()).toBeCloseTo(0.2, 6);
    expect(slider.knob.get_fill_opacity()).toBeCloseTo(0.8, 6);

    const marker = new Circle(0.1);
    marker.move_to(slider.knob.get_center());
    slider.attach(marker);
    expect(slider.knob.get_fill_opacity()).toBe(0);
    expect(slider.attached).toContain(marker);

    slider.set_value(1);
    expect(marker.get_center()[0]).toBeCloseTo(slider.knob.get_center()[0], 6);
    expect(marker.get_center()[0]).toBeCloseTo(2, 6);

    slider.unattach();
    expect(slider.attached.length).toBe(0);
    expect(slider.knob.get_fill_opacity()).toBeCloseTo(0.8, 6);
  });
});

describe("Switch", () => {
  test("toggles between true and false", () => {
    const on = new Switch(true);
    expect(on.get_value()).toBe(true);
    const context = { host: { live: false } as unknown as InteractiveHost, camera: new Camera() };
    on.on_press([0, 0, 0], context);
    expect(on.get_value()).toBe(false);
    on.on_press([0, 0, 0], context);
    expect(on.get_value()).toBe(true);
  });

  test("hit box covers the track", () => {
    const toggle = new Switch(false);
    toggle.move_to([1, 2, 0]);
    expect(toggle.is_hit([1, 2, 0])).toBe(true);
    expect(toggle.is_hit([5, 5, 0])).toBe(false);
  });
});

describe("InputDecimalNumber", () => {
  test("reflects text typed into the host editor", () => {
    const host = new FakeHost();
    const number = new InputDecimalNumber(0, { num_decimal_places: 1 });
    const context = { host, camera: new Camera() };
    number.on_press([0, 0, 0], context);
    expect(host.editor).not.toBeNull();
    host.editor?.onInput("2.5");
    expect(number.get_value()).toBeCloseTo(2.5, 12);
    host.editor?.onCommit("3");
    expect(number.get_value()).toBe(3);
    expect(number.editing).toBe(false);
  });

  test("hit testing covers the glyph bbox", () => {
    const number = new InputDecimalNumber(1, { num_decimal_places: 1 });
    expect(number.is_hit(number.get_center())).toBe(true);
    expect(number.is_hit([number.get_center()[0] + 100, 0, 0])).toBe(false);
  });
});

class DemoScene extends InteractiveScene {
  slider!: Slider;
  circle!: Circle;
  constructed = false;
  widths: number[] = [];

  blueprint(): void {
    this.slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [1, 3], init_value: 1 });
    this.circle = new Circle(1);
    this.circle.add_reaction((mobj) => {
      const width = this.slider.get_value();
      this.widths.push(width);
      (mobj as Circle).set_width(width);
    });
    this.add(this.slider, this.circle);
  }

  construct(): void {
    this.constructed = true;
  }
}

describe("InteractiveScene", () => {
  test("headless (node) runs one blueprint/construct pass and finishes", async () => {
    const renderer = makeRenderer();
    const scene = new DemoScene({ renderer });
    await scene.render();
    expect(scene.constructed).toBe(true);
    expect(scene.mobjects.length).toBe(2);
  });

  test("live interaction applies reactions only in the blueprint phase", async () => {
    const host = new FakeHost();
    const scene = new DemoScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    await host.advance(1);

    // Blueprint phase: reactions drive the circle.
    expect(scene.widths.length).toBeGreaterThan(0);
    host.press([2, 0, 0]);
    host.release();
    await host.advance(1);
    expect(scene.slider.get_value()).toBe(3);
    expect(scene.circle.width).toBeCloseTo(3, 6);

    // Advance to construct.
    host.click("进入下一步");
    await host.advance(1);
    expect(scene.constructed).toBe(true);

    // Reactions are now disabled: further interaction must not change the shape.
    const before = scene.widths.length;
    host.press([-2, 0, 0]);
    host.release();
    await host.advance(1);
    expect(scene.widths.length).toBe(before);

    // Finish the last phase: the run's controls are disabled (the DOM host
    // removes them when the run is torn down).
    host.click("进入下一步");
    await host.advance(1);
    await done;
    expect(host.buttonsEnabled).toBe(false);
  });

  test("restart rebuilds the whole story from the first blueprint", async () => {
    const host = new FakeHost();
    const scene = new DemoScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    await host.advance(1);
    const firstSlider = scene.slider;

    host.click("从头再来");
    await host.advance(2);
    expect(scene.slider).not.toBe(firstSlider);
    expect(scene.constructed).toBe(false);
    expect(scene.mobjects.length).toBe(2);

    host.click("进入下一步");
    await host.advance(1);
    host.click("进入下一步");
    await host.advance(1);
    await done;
  });

  test("updaters run during the construct phase", async () => {
    const host = new FakeHost();
    let updates = 0;
    class UpdaterScene extends InteractiveScene {
      override construct(): void {
        const dot = new Circle(0.2);
        dot.add_updater(() => {
          updates += 1;
        });
        this.add(dot);
      }
    }
    const scene = new UpdaterScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    await host.advance(1);
    updates = 0;
    host.click("进入下一步");
    await host.advance(1);
    expect(updates).toBeGreaterThan(0);
    host.click("进入下一步");
    await host.advance(1);
    await done;
  });

  test("a storyboard schedules multiple blueprint/construct pairs", async () => {
    const host = new FakeHost();
    const order: string[] = [];
    class StoryScene extends InteractiveScene {
      override blueprint(): void {
        order.push("blueprint0");
      }
      override construct(): void {
        order.push("construct0");
      }
      blueprint1(): void {
        order.push("blueprint1");
      }
      construct1(): void {
        order.push("construct1");
      }
      override setup(): void {
        this.set_storyboard([
          [this.blueprint, this.construct],
          [this.blueprint1, this.construct1],
        ]);
      }
    }
    const scene = new StoryScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    for (let step = 0; step < 4; step += 1) {
      await host.advance(1);
      host.click("进入下一步");
    }
    await host.advance(1);
    await done;
    expect(order).toEqual(["blueprint0", "construct0", "blueprint1", "construct1"]);
  });

  test("an aborted run stops even after a newer scheduler is installed", async () => {
    const host = new FakeHost();
    const scene = new DemoScene({ renderer: makeRenderer(), host });
    const schedulerA = create_frame_scheduler({ wait_for_frame: (cb) => setTimeout(cb, 0) });
    set_render_scheduler(schedulerA);
    try {
      const done = scene.render();
      await host.advance(1);
      // A newer run replaces the global scheduler before the old one notices
      // the abort. The old run must still stop: it belongs to scheduler A.
      const schedulerB = create_frame_scheduler({ wait_for_frame: (cb) => setTimeout(cb, 0) });
      set_render_scheduler(schedulerB);
      schedulerA.cancel();
      await host.advance(1);
      await done;
      expect(scene.run_aborted()).toBe(true);
    } finally {
      set_render_scheduler(null);
    }
  });

  test("play after an abort does not start a new animation", async () => {
    const renderer = makeRenderer();
    const scene = new Scene({ renderer });
    const scheduler = create_frame_scheduler({ wait_for_frame: (cb) => setTimeout(cb, 0) });
    set_render_scheduler(scheduler);
    try {
      const rendered = scene.render();
      if (rendered instanceof Promise) await rendered;
      scheduler.cancel();
      let plays = 0;
      const original = renderer.play.bind(renderer);
      renderer.play = ((...args: unknown[]) => {
        plays += 1;
        return original(...(args as never[]));
      }) as typeof renderer.play;
      const result = scene.play(new Wait({ run_time: 0.1 }));
      expect(plays).toBe(0);
      expect(result).toBe(scene);
    } finally {
      set_render_scheduler(null);
    }
  });

  test("an aborted renderer refuses to draw", () => {
    const controller = new AbortController();
    const renderer = new CanvasSceneRenderer("", {
      ctx: mockContext(),
      drawFrames: true,
      abortSignal: controller.signal,
    });
    const scene = new DemoScene({ renderer });
    controller.abort();
    renderer.update_frame(scene);
    expect(renderer.stats.drawMs).toBe(0);
    expect(renderer.stats.frames).toBe(0);
  });
});
