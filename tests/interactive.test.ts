import { describe, expect, test } from "vitest";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Camera } from "../src/camera/camera.js";
import { Arc, Circle } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { DecimalNumber } from "../src/mobject/text/numbers.js";
import { YELLOW } from "../src/utils/color/index.js";
import {
  InputDecimalMatrix,
  InputDecimalNumber,
  InputDecimalTable,
  InputInteger,
  PathSlider,
  Slider,
  Switch,
  isInteractiveControl,
} from "../src/interactive/controls.js";
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
    // A click (press without a drag) opens the editor on release.
    number.on_release();
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

  test("inline editor and caret track a forced display sign", () => {
    const host = new FakeHost();
    const number = new InputDecimalNumber(0, {
      num_decimal_places: 2,
      value_range: [-1, 1],
      include_sign: true,
    });
    number.on_press(number.get_center(), controlContext(host));
    number.on_release();
    // The editor text starts exactly like the rendered number, sign included.
    expect(host.editor?.initial).toBe("+0.00");

    const glyphs = number.submobjects.filter((member) => member !== number.highlight);
    // Caret at the end of "+0.00" sits after the last glyph...
    expect(number.interactive_caret_anchor(5)?.[0]).toBeCloseTo(number.get_right()[0], 9);
    // ...and at the start it sits before the displayed sign glyph.
    expect(number.interactive_caret_anchor(0)?.[0]).toBeCloseTo(glyphs[0]!.get_left()[0], 9);

    // Deleting the sign still anchors the text end at the number's right edge
    // (the display sign is skipped, not every position shifted by one glyph);
    // the start caret now sits after the display-only sign.
    host.editor?.onInput("0.00");
    expect(number.interactive_caret_anchor(4)?.[0]).toBeCloseTo(number.get_right()[0], 9);
    expect(number.interactive_caret_anchor(0)?.[0]).toBeCloseTo(glyphs[1]!.get_left()[0], 9);

    // A text that carries its own sign maps one-to-one onto the glyphs.
    host.editor?.onInput("-0.5");
    const signedGlyphs = number.submobjects.filter((member) => member !== number.highlight);
    expect(number.interactive_caret_anchor(4)?.[0]).toBeCloseTo(signedGlyphs[4]!.get_left()[0], 9);
    expect(number.interactive_caret_anchor(2)?.[0]).toBeCloseTo(signedGlyphs[2]!.get_left()[0], 9);
    host.editor?.onCommit("-0.5");
    expect(number.get_value()).toBeCloseTo(-0.5, 12);
  });
});

describe("InputDecimalNumber value_range, scrubbing and hover", () => {
  /** World-space vertical distance matching `pixels` on screen. */
  function worldPixels(camera: Camera, pixels: number): number {
    return pixels / (camera.pixel_height / camera.frame_height);
  }

  test("clamps the initial value, set_value and text edits into value_range", () => {
    const host = new FakeHost();
    const number = new InputDecimalNumber(5, { value_range: [0, 3] });
    expect(number.get_value()).toBe(3);

    number.set_value(-1);
    expect(number.get_value()).toBe(0);
    number.set_value(1.5);
    expect(number.get_value()).toBe(1.5);

    number.on_press(number.get_center(), controlContext(host));
    number.on_release();
    host.editor?.onInput("7");
    expect(number.get_value()).toBe(3);
    host.editor?.onInput("-2");
    expect(number.get_value()).toBe(0);
  });

  test("dragging vertically scrubs the value and clamps at both ends", () => {
    const host = new FakeHost();
    const camera = new Camera();
    const number = new InputDecimalNumber(0, { value_range: [0, 10], drag_pixels: 100, drag_span: 10 });
    const context = { host, camera };
    const cx = number.get_center()[0] ?? 0;
    const cy = number.get_center()[1] ?? 0;

    number.on_press([cx, cy, 0], context);
    // 20 px up = +2.
    number.on_drag([cx, cy + worldPixels(camera, 20), 0], context);
    expect(number.get_value()).toBeCloseTo(2, 6);
    number.on_drag([cx, cy + worldPixels(camera, 40), 0], context);
    expect(number.get_value()).toBeCloseTo(4, 6);

    // Dragging far past the top clamps to the maximum...
    number.on_drag([cx, cy + worldPixels(camera, 400), 0], context);
    expect(number.get_value()).toBe(10);
    // ...and reversing from the boundary responds immediately (the drag
    // accumulator must not keep "debt" from the clamped overshoot).
    number.on_drag([cx, cy + worldPixels(camera, 390), 0], context);
    expect(number.get_value()).toBeCloseTo(9, 6);

    number.on_release();
    // A drag is not a click: no editor may open.
    expect(host.editor).toBeNull();
  });

  test("a tiny jitter is still a click; a real drag is not", () => {
    const host = new FakeHost();
    const camera = new Camera();
    const context = { host, camera };

    const number = new InputDecimalNumber(1);
    const cx = number.get_center()[0] ?? 0;
    const cy = number.get_center()[1] ?? 0;
    number.on_press([cx, cy, 0], context);
    number.on_drag([cx, cy + worldPixels(camera, 2), 0], context);
    number.on_release();
    expect(host.editor).not.toBeNull();
    host.endTextEdit();

    const other = new InputDecimalNumber(1);
    const ox = other.get_center()[0] ?? 0;
    const oy = other.get_center()[1] ?? 0;
    other.on_press([ox, oy, 0], context);
    other.on_drag([ox, oy + worldPixels(camera, 20), 0], context);
    other.on_release();
    expect(host.editor).toBeNull();
  });

  test("prefers the host's screen scale over the camera's backing scale", () => {
    const host = new FakeHost();
    (host as unknown as { pixelsPerWorldUnit: () => [number, number] }).pixelsPerWorldUnit = () => [1000, 1000];
    const number = new InputDecimalNumber(0, { drag_pixels: 100, drag_span: 1 });
    const context = { host, camera: new Camera() };
    const cx = number.get_center()[0] ?? 0;
    const cy = number.get_center()[1] ?? 0;

    number.on_press([cx, cy, 0], context);
    // 0.1 world units at 1000 px/unit = 100 px -> the full drag_span.
    number.on_drag([cx, cy + 0.1, 0], context);
    expect(number.get_value()).toBeCloseTo(1, 6);
    number.on_release();
  });

  test("hover highlights the number in its own colour, behind the digits", () => {
    const number = new InputDecimalNumber(12, { color: YELLOW, highlight_opacity: 0.4 });
    const highlight = number.highlight;
    expect(highlight).not.toBeNull();
    expect(number.submobjects[0]).toBe(highlight);
    expect(number.get_family()).toContain(highlight);
    expect(highlight?.get_fill_opacity()).toBe(0);

    number.set_hovered(true);
    expect(number.hovered).toBe(true);
    expect(highlight?.get_fill_opacity()).toBeCloseTo(0.4, 6);
    expect(highlight?.get_fill_color().to_hex()).toBe(YELLOW.to_hex());

    number.set_hovered(false);
    expect(number.hovered).toBe(false);
    expect(highlight?.get_fill_opacity()).toBe(0);
  });

  test("the highlight matches the glyph bounding box (no layout impact)", () => {
    const number = new InputDecimalNumber(1234, { color: YELLOW, num_decimal_places: 0 });
    const plain = new DecimalNumber(1234, { num_decimal_places: 0, group_with_commas: false });
    expect(number.width).toBeCloseTo(plain.width, 9);
    expect(number.height).toBeCloseTo(plain.height, 9);

    const highlight = number.highlight!;
    expect(number.submobjects[0]).toBe(highlight);
    expect(highlight.get_left()[0]).toBeCloseTo(number.get_left()[0], 6);
    expect(highlight.get_right()[0]).toBeCloseTo(number.get_right()[0], 6);

    number.set_value(7);
    expect(number.submobjects[0]).toBe(highlight);
    expect(number.width).toBeCloseTo(highlight.width, 9);
  });

  test("InputInteger scrubs in whole steps", () => {
    const host = new FakeHost();
    const camera = new Camera();
    const number = new InputInteger(0, { drag_pixels: 100, drag_span: 10 });
    const context = { host, camera };
    const cx = number.get_center()[0] ?? 0;
    const cy = number.get_center()[1] ?? 0;

    number.on_press([cx, cy, 0], context);
    number.on_drag([cx, cy + worldPixels(camera, 4), 0], context);
    expect(number.get_value()).toBe(0);
    // Slow drags accumulate: the second 4 px step crosses one whole unit.
    number.on_drag([cx, cy + worldPixels(camera, 8), 0], context);
    expect(number.get_value()).toBe(1);
    number.on_drag([cx, cy + worldPixels(camera, 20), 0], context);
    expect(number.get_value()).toBe(2);
    number.on_release();
  });

  test("InteractiveScene highlights the hovered number and clears it off-target", async () => {
    class NumberScene extends InteractiveScene {
      number!: InputDecimalNumber;
      blueprint(): void {
        this.number = new InputDecimalNumber(3, { color: YELLOW });
        this.add(this.number);
      }
    }
    const host = new FakeHost();
    const scene = new NumberScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    await host.advance(1);

    host.drag(scene.number.get_center());
    expect(scene.number.hovered).toBe(true);
    expect(scene.number.highlight?.get_fill_opacity()).toBeGreaterThan(0);

    host.drag([5, 5, 0]);
    expect(scene.number.hovered).toBe(false);
    expect(scene.number.highlight?.get_fill_opacity()).toBe(0);

    host.click("进入下一步");
    await host.advance(1);
    host.click("进入下一步");
    await host.advance(1);
    await done;
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

function controlContext(host: FakeHost): { host: FakeHost; camera: Camera } {
  return { host, camera: new Camera() };
}

describe("PathSlider", () => {
  test("a straight path behaves like Slider (press/drag/set_value/attach)", () => {
    const path = new Line({ start: [-2, 0, 0], end: [2, 0, 0] });
    const slider = new PathSlider(path, { value_range: [0, 10] });
    expect(slider.get_value()).toBe(0);

    slider.set_value(7.5);
    expect(slider.get_value()).toBe(7.5);
    expect(slider.knob.get_center()[0]).toBeCloseTo(-2 + 0.75 * 4, 12);

    const context = controlContext(new FakeHost());
    slider.on_press([0, 1, 0], context);
    expect(slider.get_value()).toBeCloseTo(5, 12);
    // Incremental drag: the pointer motion is projected onto the tangent.
    slider.on_drag([0.4, 1, 0], context);
    expect(slider.get_value()).toBeCloseTo(6, 12);
    slider.on_release();

    const marker = new Circle(0.1);
    marker.move_to(slider.knob.get_center());
    slider.attach(marker);
    expect(slider.knob.get_fill_opacity()).toBe(0);
    expect(slider.attached).toContain(marker);
    slider.set_value(10);
    expect(marker.get_center()[0]).toBeCloseTo(2, 12);
    slider.unattach();
    expect(slider.knob.get_fill_opacity()).toBe(1);
  });

  test("open paths clamp at both ends", () => {
    const arc = new Arc({ radius: 1, start_angle: 0, angle: Math.PI / 2 });
    const slider = new PathSlider(arc, { value_range: [0, 10] });
    const context = controlContext(new FakeHost());

    slider.on_press([2, 0, 0], context);
    expect(slider.get_value()).toBeCloseTo(0, 9);
    slider.on_press([0, 2, 0], context);
    expect(slider.get_value()).toBeCloseTo(10, 9);
    slider.set_value(-5);
    expect(slider.get_value()).toBe(0);
    slider.set_value(15);
    expect(slider.get_value()).toBe(10);
  });

  test("closed paths map by proportion and wrap past the seam", () => {
    const circle = new Circle({ radius: 1 });
    const slider = new PathSlider(circle, { value_range: [0, 10], is_closed: true });
    expect(slider.is_closed).toBe(true);
    expect(slider.is_hit([1, 0.1, 0])).toBe(true);
    expect(slider.is_hit([0, 0, 0])).toBe(false);

    // 0.25 of the arc length is the top of the circle.
    slider.set_value(2.5);
    expect(slider.knob.get_center()[0]).toBeCloseTo(0, 6);
    expect(slider.knob.get_center()[1]).toBeCloseTo(1, 6);

    // Out-of-range values wrap modulo the range for a closed path.
    slider.set_value(12);
    expect(slider.get_value()).toBeCloseTo(2, 9);
    slider.set_value(-1);
    expect(slider.get_value()).toBeCloseTo(9, 9);

    // Dragging along the tangent from near the seam wraps to near the low end.
    const context = controlContext(new FakeHost());
    slider.set_value(9.5);
    const start = circle.point_from_proportion(0.95);
    const before = circle.point_from_proportion(0.949);
    const after = circle.point_from_proportion(0.951);
    const tangent: number[] = [
      (after[0] as number) - (before[0] as number),
      (after[1] as number) - (before[1] as number),
    ];
    const tangentLength = Math.hypot(tangent[0]!, tangent[1]!);
    tangent[0] = tangent[0]! / tangentLength;
    tangent[1] = tangent[1]! / tangentLength;
    const total = circle.get_arc_length();
    const pressPoint = [start[0] as number, start[1] as number, 0];
    const dragPoint = [
      pressPoint[0]! + tangent[0]! * 0.1 * total,
      pressPoint[1]! + tangent[1]! * 0.1 * total,
      0,
    ];
    slider.on_press(pressPoint, context);
    slider.on_drag(dragPoint, context);
    expect(slider.get_value()).toBeGreaterThan(0);
    expect(slider.get_value()).toBeLessThan(1);
    slider.on_release();
  });

  test("dragging perpendicular to the tangent leaves the value unchanged", () => {
    const circle = new Circle({ radius: 1 });
    const slider = new PathSlider(circle, { value_range: [0, 10], is_closed: true });
    const context = controlContext(new FakeHost());
    slider.set_value(2.5);
    const center = slider.knob.get_center();
    slider.on_press(center, context);
    slider.on_drag([center[0] ?? 0, (center[1] ?? 0) + 0.3, 0], context);
    expect(slider.get_value()).toBeCloseTo(2.5, 9);
  });

  test("supports a VGroup path by summing the family's arc lengths", () => {
    const path = new VGroup(
      new Line({ start: [-2, 0, 0], end: [0, 0, 0] }),
      new Line({ start: [0, 0, 0], end: [0, 2, 0] }),
    );
    const slider = new PathSlider(path, { value_range: [0, 2] });
    slider.set_value(1);
    expect(slider.knob.get_center()[0]).toBeCloseTo(0, 6);
    expect(slider.knob.get_center()[1]).toBeCloseTo(0, 6);
    slider.set_value(2);
    expect(slider.knob.get_center()[1]).toBeCloseTo(2, 6);
  });

  test("Slider is a PathSlider whose track is a Line", () => {
    const slider = new Slider({ start: [-2, 0, 0], end: [2, 0, 0] });
    expect(slider instanceof PathSlider).toBe(true);
    expect(slider.track).toBeInstanceOf(Line);
    expect(slider.start_point).toEqual([-2, 0, 0]);
    expect(slider.end_point).toEqual([2, 0, 0]);
  });
});

describe("InputInteger", () => {
  test("truncates like Integer and blocks the decimal point in the editor", () => {
    const host = new FakeHost();
    const number = new InputInteger(0);
    number.on_press([0, 0, 0], controlContext(host));
    number.on_release();
    expect(host.editor).not.toBeNull();
    expect(host.editor?.allow_decimal_point).toBe(false);

    host.editor?.onInput("7");
    expect(number.get_value()).toBe(7);
    // Even if the host text does contain a decimal point, Integer semantics
    // round then truncate.
    host.editor?.onInput("2.9");
    expect(number.get_value()).toBe(3);
    host.editor?.onCommit("-4.6");
    expect(number.get_value()).toBe(-5);
    expect(number.num_decimal_places).toBe(0);
  });
});

describe("InputDecimalNumber on_change", () => {
  test("fires after every successful edit (and not on unparsable text)", () => {
    const host = new FakeHost();
    const number = new InputDecimalNumber(0);
    let changes = 0;
    number.on_change = () => {
      changes += 1;
    };
    number.on_press([0, 0, 0], controlContext(host));
    number.on_release();
    host.editor?.onInput("2.5");
    expect(changes).toBe(1);
    host.editor?.onInput("not a number");
    expect(changes).toBe(1);
    host.editor?.onCommit("3");
    expect(changes).toBe(2);
  });
});

describe("InputDecimalMatrix", () => {
  test("every cell is interactive and editing relayouts without moving the center", () => {
    const host = new FakeHost();
    const matrix = new InputDecimalMatrix([
      [1, 2],
      [3, 4],
    ]);
    for (const row of matrix.mob_matrix) {
      for (const cell of row) {
        expect(isInteractiveControl(cell)).toBe(true);
      }
    }

    const centerBefore = matrix.get_center();
    const left = matrix.mob_matrix[0]![0]!;
    const right = matrix.mob_matrix[0]![1]!;
    const gapBefore = (right.get_center()[0] ?? 0) - (left.get_center()[0] ?? 0);

    left.on_press(left.get_center(), controlContext(host));
    left.on_release();
    expect(host.editor).not.toBeNull();
    host.editor?.onInput("123456");

    const gapAfter = (right.get_center()[0] ?? 0) - (left.get_center()[0] ?? 0);
    expect(gapAfter).toBeGreaterThan(gapBefore);
    expect(matrix.get_center()[0]).toBeCloseTo(centerBefore[0] ?? 0, 6);
    expect(matrix.get_center()[1]).toBeCloseTo(centerBefore[1] ?? 0, 6);

    // The wider column must not overlap its neighbour, and rows stay apart.
    for (let j = 0; j + 1 < matrix.mob_matrix[0]!.length; j += 1) {
      expect(matrix.mob_matrix[0]![j]!.get_right()[0] ?? 0).toBeLessThanOrEqual(
        (matrix.mob_matrix[0]![j + 1]!.get_left()[0] ?? 0) + 1e-9,
      );
    }
    for (let i = 0; i + 1 < matrix.mob_matrix.length; i += 1) {
      expect(matrix.mob_matrix[i]![0]!.get_bottom()[1] ?? 0).toBeGreaterThanOrEqual(
        (matrix.mob_matrix[i + 1]![0]!.get_top()[1] ?? 0) - 1e-9,
      );
    }
  });

  test("construction already uses the relayout layout (no jump on first edit)", () => {
    const matrix = new InputDecimalMatrix(
      [
        [1, 2],
        [3, 4],
      ],
      { element_to_mobject_config: { num_decimal_places: 1 } },
    );
    const width = matrix.width;
    const height = matrix.height;
    const centersBefore = matrix.mob_matrix.map((row) => row.map((cell) => [...cell.get_center()]));

    matrix.relayout();
    expect(matrix.width).toBeCloseTo(width, 9);
    expect(matrix.height).toBeCloseTo(height, 9);
    matrix.mob_matrix.forEach((row, i) => {
      row.forEach((cell, j) => {
        expect(cell.get_center()[0]).toBeCloseTo(centersBefore[i]![j]![0]!, 9);
        expect(cell.get_center()[1]).toBeCloseTo(centersBefore[i]![j]![1]!, 9);
      });
    });
  });

  test("InteractiveScene routes a pointer press to a matrix cell", async () => {
    class MatrixScene extends InteractiveScene {
      matrix!: InputDecimalMatrix;
      blueprint(): void {
        this.matrix = new InputDecimalMatrix([
          [1, 2],
          [3, 4],
        ]);
        this.add(this.matrix);
      }
    }
    const host = new FakeHost();
    const scene = new MatrixScene({ renderer: makeRenderer(), host });
    const done = scene.render();
    await host.advance(1);

    const cell = scene.matrix.mob_matrix[1]![0]!;
    host.press(cell.get_center());
    host.release(cell.get_center());
    await host.advance(1);
    expect(host.editor).not.toBeNull();
    host.editor?.onInput("42");
    expect(cell.get_value()).toBe(42);

    host.click("进入下一步");
    await host.advance(1);
    host.click("进入下一步");
    await host.advance(1);
    await done;
  });
});

describe("InputDecimalTable", () => {
  test("editing relayouts and rebuilds the lines around the new layout", () => {
    const host = new FakeHost();
    const table = new InputDecimalTable([
      [1, 2],
      [3, 4],
    ]);
    for (const row of table.mob_table) {
      for (const cell of row) {
        expect(isInteractiveControl(cell)).toBe(true);
      }
    }

    const centerBefore = table.get_center();
    const oldHorizontal = table.horizontal_lines;
    const oldVertical = table.vertical_lines;
    const cell = table.mob_table[0]![0]!;
    cell.on_press(cell.get_center(), controlContext(host));
    cell.on_release();
    host.editor?.onInput("12345");

    expect(table.get_center()[0]).toBeCloseTo(centerBefore[0] ?? 0, 6);
    expect(table.get_center()[1]).toBeCloseTo(centerBefore[1] ?? 0, 6);
    expect(table.horizontal_lines).not.toBe(oldHorizontal);
    expect(table.vertical_lines).not.toBe(oldVertical);

    // Two rows / two columns -> one inner line each, attached to the table.
    expect(table.horizontal_lines.submobjects.length).toBe(1);
    expect(table.vertical_lines.submobjects.length).toBe(1);
    const horizontal = table.horizontal_lines.__getitem__(0) as Line;
    const vertical = table.vertical_lines.__getitem__(0) as Line;
    expect(table.get_family()).toContain(horizontal);
    expect(table.get_family()).toContain(vertical);

    // The lines sit in the (new) gaps between rows/columns.
    const row0 = table.mob_table[0]![0]!;
    const row1 = table.mob_table[1]![0]!;
    const horizontalY = horizontal.get_center()[1] ?? 0;
    expect(horizontalY).toBeLessThan(row0.get_bottom()[1] ?? 0);
    expect(horizontalY).toBeGreaterThan(row1.get_top()[1] ?? 0);

    const col0 = table.mob_table[0]![0]!;
    const col1 = table.mob_table[0]![1]!;
    const verticalX = vertical.get_center()[0] ?? 0;
    expect(verticalX).toBeGreaterThan(col0.get_right()[0] ?? 0);
    expect(verticalX).toBeLessThan(col1.get_left()[0] ?? 0);
  });

  test("repeated relayouts replace the previous lines instead of stacking them", () => {
    const table = new InputDecimalTable(
      [
        [1, 2],
        [3, 4],
      ],
      { include_outer_lines: true },
    );
    const lineCount = (): number =>
      table.get_family().filter((member) => member instanceof Line).length;
    const initialLines = lineCount();
    const initialSubmobjects = table.submobjects.length;

    const cell = table.mob_table[0]![0]!;
    for (const value of [10, 20, 30]) {
      cell.set_value(value);
      (cell as unknown as { on_change?: () => void }).on_change?.();
      expect(lineCount()).toBe(initialLines);
      expect(table.submobjects.length).toBe(initialSubmobjects);
    }
  });
});
