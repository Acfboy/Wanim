import { describe, expect, test } from "vitest";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Slider } from "../src/interactive/controls.js";
import type { InteractiveHost, PointerHandlers, TextEditOptions } from "../src/interactive/host.js";
import { NullInteractiveHost } from "../src/interactive/host.js";
import {
  ScenePresentation,
  dark_from_hex,
  type PresentationState,
} from "../src/interactive/presentation.js";
import { InteractiveScene } from "../src/scene/interactive_scene.js";
import { Scene } from "../src/scene/scene.js";
import { apply_wanim_config, default_wanim_run_config } from "../src/interactive/config.js";
import type { SceneConstructor } from "../src/renderer/scene_runner.js";

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

function makeRenderer(): CanvasSceneRenderer {
  return new CanvasSceneRenderer("", { ctx: mockContext(), drawFrames: true, recordFrames: false });
}

class FakeHost implements InteractiveHost {
  readonly live = true;
  handlers: PointerHandlers | null = null;
  buttons = new Map<string, () => void>();
  buttonsEnabled = true;
  visible = new Map<string, boolean>();
  dark: boolean | null = null;
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
    return () => {
      this.buttons.delete(label);
    };
  }

  setButtonsEnabled(enabled: boolean): void {
    this.buttonsEnabled = enabled;
  }

  setButtonVisible(icon: string, visible: boolean): void {
    this.visible.set(icon, visible);
  }

  setControlTheme(dark: boolean): void {
    this.dark = dark;
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

  async advance(count = 1): Promise<void> {
    for (let index = 0; index < count; index += 1) {
      const resolve = this.resolvers.shift();
      if (resolve !== undefined) resolve();
      for (let tick = 0; tick < 6; tick += 1) await Promise.resolve();
    }
  }

  click(label: string): void {
    this.buttons.get(label)?.();
  }

  press(point: number[]): void {
    this.nextPoint = point;
    this.handlers?.down(new Event("pointerdown") as unknown as PointerEvent);
  }

  release(point: number[] = this.nextPoint): void {
    this.nextPoint = point;
    this.handlers?.up(new Event("pointerup") as unknown as PointerEvent);
  }
}

interface Factory {
  created: Array<InteractiveScene>;
  create: (sceneClass: SceneConstructor) => Scene;
}

function interactiveFactory(host: FakeHost): Factory {
  const created: Array<InteractiveScene> = [];
  const create = (sceneClass: SceneConstructor): Scene => {
    const scene = new sceneClass({ renderer: makeRenderer(), host });
    if (scene instanceof InteractiveScene) created.push(scene);
    return scene;
  };
  return { created, create };
}

class SliderScene extends InteractiveScene {
  slider!: Slider;
  constructs = 0;
  updates = 0;

  blueprint(): void {
    this.slider = new Slider({
      start: [-2, 0, 0],
      end: [2, 0, 0],
      value_range: [0, 10],
      init_value: 1,
    });
    const dot = new Circle(0.2);
    dot.add_reaction(() => {
      this.updates += 1;
    });
    this.add(this.slider, dot);
  }

  construct(): void {
    this.constructs += 1;
  }
}

class SecondScene extends SliderScene {}

describe("wanim_config", () => {
  test("defaults to slides_mode off and can be switched on", () => {
    const config = default_wanim_run_config();
    expect(config.slides_mode).toBe(false);
    apply_wanim_config(config, { slides_mode: true });
    expect(config.slides_mode).toBe(true);
    apply_wanim_config(config, {});
    expect(config.slides_mode).toBe(true);
  });

  test("rejects unknown keys and wrong types", () => {
    const config = default_wanim_run_config();
    expect(() => apply_wanim_config(config, { nope: 1 })).toThrowError(/unknown option 'nope'/);
    expect(() => apply_wanim_config(config, { slides_mode: "yes" })).toThrowError(/must be a bool/);
  });
});

describe("dark_from_hex", () => {
  test("light icons on dark backdrops, dark icons on light ones", () => {
    expect(dark_from_hex("#000000")).toBe(true);
    expect(dark_from_hex("#202428")).toBe(true);
    expect(dark_from_hex("#ffffff")).toBe(false);
    expect(dark_from_hex("#eeeeee")).toBe(false);
    // A translucent/unknown backdrop is treated as light (white page).
    expect(dark_from_hex("#000000", 0)).toBe(false);
  });
});

describe("ScenePresentation", () => {
  test("non-slides: only blueprints pause, constructs flow into the next scene", async () => {
    const host = new FakeHost();
    const factory = interactiveFactory(host);
    const controller = new ScenePresentation([SliderScene, SecondScene], {
      host,
      createScene: factory.create,
    });
    const run = controller.run();
    await host.advance(1);

    // Scene 1 blueprint: paused, next available, previous/replay hidden.
    expect(controller.state.sceneIndex).toBe(0);
    expect(controller.state.phaseKind).toBe("blueprint");
    expect(controller.state.canNext).toBe(true);
    expect(host.visible.get("next")).toBe(true);
    expect(host.visible.get("previous")).toBe(false);

    host.click("进入下一步");
    await host.advance(2);
    // Construct ran and flowed straight into scene 2's blueprint.
    expect(factory.created[0]!.constructs).toBe(1);
    expect(controller.state.sceneIndex).toBe(1);
    expect(controller.state.phaseKind).toBe("blueprint");

    host.click("进入下一步");
    await host.advance(2);
    expect(factory.created[1]!.constructs).toBe(1);
    await run;
    expect(controller.state.done).toBe(true);
  });

  test("slides: pause after every phase and remember blueprint inputs on previous", async () => {
    const host = new FakeHost();
    const factory = interactiveFactory(host);
    const controller = new ScenePresentation([SliderScene, SecondScene], {
      host,
      slidesMode: true,
      createScene: factory.create,
    });
    const run = controller.run();
    await host.advance(1);

    const first = factory.created[0]!;
    expect(controller.state.canPrevious).toBe(false);
    expect(controller.state.canReplay).toBe(true);
    // Reactions run while paused in the blueprint.
    expect(first.updates).toBeGreaterThan(0);

    // Drag the slider to its maximum.
    const sliderBefore = first.slider;
    host.press([2, 0, 0]);
    host.release();
    await host.advance(1);
    expect(first.slider.get_value()).toBe(10);

    // Next: the construct pauses in slides mode.
    host.click("进入下一步");
    await host.advance(2);
    expect(controller.state.phaseKind).toBe("construct");
    expect(first.constructs).toBe(1);
    expect(host.visible.get("previous")).toBe(true);

    // Previous: rebuild the blueprint with the remembered input.  Rewinding
    // within a scene reuses its instance (the state is rebuilt from scratch).
    host.click("上一页");
    await host.advance(2);
    expect(controller.state.phaseKind).toBe("blueprint");
    expect(first.slider).not.toBe(sliderBefore);
    expect(first.slider.get_value()).toBe(10);

    // Replay the blueprint: inputs are cleared back to the initial value.
    host.click("重播本段");
    await host.advance(2);
    expect(first.slider.get_value()).toBe(1);

    // Forward again: the construct replays (second run on this instance).
    host.click("进入下一步");
    await host.advance(2);
    expect(first.constructs).toBe(2);

    // Next from the last phase of scene 1 enters scene 2's blueprint.
    host.click("进入下一步");
    await host.advance(2);
    expect(controller.state.sceneIndex).toBe(1);
    expect(controller.state.phaseKind).toBe("blueprint");

    // Previous on the first page of scene 2 goes back to scene 1's last page.
    host.click("上一页");
    await host.advance(2);
    expect(controller.state.sceneIndex).toBe(0);
    expect(controller.state.phaseKind).toBe("construct");

    // Forward returns to scene 2's blueprint.
    host.click("进入下一步");
    await host.advance(2);
    expect(controller.state.sceneIndex).toBe(1);
    expect(controller.state.phaseKind).toBe("blueprint");

    controller.abort();
    await host.advance(1);
    await run;
  });

  test("restart rebuilds from the first page", async () => {
    const host = new FakeHost();
    const factory = interactiveFactory(host);
    const controller = new ScenePresentation([SliderScene], {
      host,
      slidesMode: true,
      createScene: factory.create,
    });
    const run = controller.run();
    await host.advance(1);
    host.click("进入下一步");
    await host.advance(2);
    expect(controller.state.phaseKind).toBe("construct");

    controller.restart();
    await host.advance(2);
    expect(controller.state.sceneIndex).toBe(0);
    expect(controller.state.phaseKind).toBe("blueprint");
    // Restart rebuilds the scene from scratch (a fresh instance).
    expect(factory.created.length).toBe(2);
    expect(factory.created[1]!.slider.get_value()).toBe(1);

    controller.abort();
    await host.advance(1);
    await run;
  });

  test("headless hosts play every phase without pausing", async () => {
    const factory = interactiveFactory(new FakeHost());
    const controller = new ScenePresentation([SliderScene, SecondScene], {
      host: new NullInteractiveHost(),
      slidesMode: true,
      createScene: factory.create,
    });
    await controller.run();
    expect(controller.state.done).toBe(true);
    expect(factory.created.map((scene) => scene.constructs)).toEqual([1, 1]);
  });

  test("a plain Scene contributes a single construct page", async () => {
    let constructs = 0;
    class Plain extends Scene {
      override construct(): void {
        constructs += 1;
      }
    }
    const host = new FakeHost();
    const controller = new ScenePresentation([Plain], { host, createScene: (cls) => new cls({ renderer: makeRenderer(), host }) });
    await controller.run();
    expect(constructs).toBe(1);
    expect(controller.state.done).toBe(true);
  });
});
