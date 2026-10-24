import { describe, expect, test } from "vitest";
import { CanvasSceneRenderer } from "../src/renderer/canvas_renderer.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Slider } from "../src/interactive/controls.js";
import type { InteractiveHost, PointerHandlers, TextEditOptions } from "../src/interactive/host.js";
import { InteractiveScene } from "../src/scene/interactive_scene.js";
import { TwoViewPresentation, type WanimMessage } from "../src/interactive/two_view.js";
import type { Scene } from "../src/scene/scene.js";
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
  doubleClickHandler: (() => void) | null = null;
  private resolvers: Array<() => void> = [];

  pointFromEvent(): number[] {
    return [0, 0, 0];
  }
  onPointer(handlers: PointerHandlers): () => void {
    this.handlers = handlers;
    return () => {
      this.handlers = null;
    };
  }
  onWheel(): () => void {
    return () => {};
  }
  onDoubleClick(handler: () => void): () => void {
    this.doubleClickHandler = handler;
    return () => {
      this.doubleClickHandler = null;
    };
  }
  addButton(_label: string, onClick: () => void, icon?: string): () => void {
    this.buttons.set(icon ?? _label, onClick);
    return () => {
      this.buttons.delete(icon ?? _label);
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
  beginTextEdit(_mobject: unknown, _camera: unknown, _options: TextEditOptions): void {}
  endTextEdit(): void {}
  nextFrame(): Promise<void> {
    return new Promise((resolve) => this.resolvers.push(resolve));
  }
  async advance(count = 1): Promise<void> {
    for (let index = 0; index < count; index += 1) {
      const resolve = this.resolvers.shift();
      if (resolve !== undefined) resolve();
      for (let tick = 0; tick < 8; tick += 1) await Promise.resolve();
    }
  }
  click(icon: string): void {
    this.buttons.get(icon)?.();
  }
  doubleClick(): void {
    this.doubleClickHandler?.();
  }
}

interface Factory {
  created: Array<InteractiveScene>;
  create: (sceneClass: SceneConstructor) => Scene;
}

function factory(host: FakeHost): Factory {
  const created: Array<InteractiveScene> = [];
  return {
    created,
    create: (sceneClass: SceneConstructor) => {
      const scene = new sceneClass({ renderer: makeRenderer(), host });
      if (scene instanceof InteractiveScene) created.push(scene);
      return scene;
    },
  };
}

/** A construct that stays in flight until the test releases it. */
function deferred(): { promise: Promise<void>; release: () => void } {
  let release: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

describe("TwoViewPresentation", () => {
  test("loops blueprint+construct and switching to interaction lets the reader drive", async () => {
    const host = new FakeHost();
    const controls: Slider[] = [];
    const gate = deferred();
    let constructs = 0;
    class Demo extends InteractiveScene {
      slider!: Slider;
      reports: WanimMessage[] = [];
      blueprint(): void {
        this.slider = new Slider({
          start: [-2, 0, 0],
          end: [2, 0, 0],
          value_range: [0, 10],
          init_value: 1,
        });
        controls.push(this.slider);
        this.add(this.slider, new Circle(0.2));
      }
      construct(): unknown {
        constructs += 1;
        return gate.promise;
      }
    }
    const created = factory(host);
    const messages: WanimMessage[] = [];
    const controller = new TwoViewPresentation([Demo], {
      host,
      mode: "two-view",
      instanceId: "demo-1",
      createScene: created.create,
      onMessage: (message) => messages.push(message),
    });
    const run = controller.run();
    await host.advance(2);

    // The animation state is looping: only replay + interact are visible.
    expect(controller.state.view).toBe("animation");
    expect(host.visible.get("replay")).toBe(true);
    expect(host.visible.get("interact")).toBe(true);
    expect(host.visible.get("next")).toBe(false);
    expect(constructs).toBe(1);
    expect(host.buttonsEnabled).toBe(true);

    // Interact while the construct is in flight: the player cancels it and
    // rebuilds the scene at the live blueprint.
    host.click("interact");
    await host.advance(1);
    gate.release();
    await host.advance(4);
    expect(controller.state.view).toBe("interaction");
    expect(controller.state.phaseKind).toBe("blueprint");
    expect(host.visible.get("next")).toBe(true);
    expect(host.visible.get("interact")).toBe(false);
    expect(host.visible.get("replay")).toBe(true);

    const active = created.created[created.created.length - 1]!;
    expect(active).toBeInstanceOf(Demo);
    expect(active.slider.get_value()).toBe(1);

    // Drag the slider to its maximum, then continue into the construct.
    host.handlers?.down(new Event("pointerdown") as unknown as PointerEvent);
    host.handlers?.up(new Event("pointerup") as unknown as PointerEvent);
    await host.advance(1);
    // Move the pointer to the slider end before pressing: the fake host maps
    // every event to [0,0,0], so drive the control directly instead.
    active.slider.set_value(7);
    host.click("next");
    await host.advance(3);
    expect(controller.state.phaseKind).toBe("construct");
    expect(host.visible.get("previous")).toBe(true);

    controller.abort();
    gate.release();
    await host.advance(4);
    await run;
    expect(messages).toEqual([]);
  });

  test("SEPARATE: animation plays construct only and interaction shows blueprint only", async () => {
    const host = new FakeHost();
    const events: string[] = [];
    class Separate extends InteractiveScene {
      SEPARATE = true;
      blueprint(): void {
        events.push("blueprint");
      }
      construct(): void {
        events.push("construct");
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([Separate], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(4);
    expect(events).toContain("construct");
    expect(events).not.toContain("blueprint");

    host.click("interact");
    await host.advance(3);
    expect(controller.state.view).toBe("interaction");
    expect(events.filter((event) => event === "blueprint").length).toBe(1);

    controller.abort();
    await host.advance(3);
    await run;
  });

  test("clear_stage hides the blueprint before the construct but keeps control values", async () => {
    const host = new FakeHost();
    let visibleMobjects = -1;
    let valueSeen = -1;
    class ClearScene extends InteractiveScene {
      slider!: Slider;
      blueprint(): void {
        this.slider = new Slider({
          start: [-2, 0, 0],
          end: [2, 0, 0],
          value_range: [0, 10],
          init_value: 3,
        });
        this.add(this.slider);
        this.clear_stage();
      }
      construct(): void {
        visibleMobjects = this.mobjects.length;
        valueSeen = this.slider.get_value();
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([ClearScene], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(4);
    expect(visibleMobjects).toBe(0);
    expect(valueSeen).toBe(3);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("clear_stage in interaction keeps the controls until the reader continues", async () => {
    const host = new FakeHost();
    let visibleBeforeContinue = -1;
    let visibleInConstruct = -1;
    class ClearScene extends InteractiveScene {
      slider!: Slider;
      blueprint(): void {
        this.slider = new Slider({
          start: [-2, 0, 0],
          end: [2, 0, 0],
          value_range: [0, 10],
          init_value: 3,
        });
        this.add(this.slider);
        this.clear_stage();
      }
      construct(): void {
        visibleInConstruct = this.mobjects.length;
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([ClearScene], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(2);
    host.click("interact");
    await host.advance(3);
    expect(controller.state.view).toBe("interaction");
    // The deferred clear has not run yet, so the slider is still on screen.
    const active = created.created[created.created.length - 1] as ClearScene;
    visibleBeforeContinue = active.mobjects.length;
    expect(visibleBeforeContinue).toBe(1);
    expect(active.stage_clear_pending).toBe(true);

    host.click("next");
    await host.advance(3);
    expect(visibleInConstruct).toBe(0);
    expect(active.slider.get_value()).toBe(3);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("uses dark control icons when the camera background is light", async () => {
    const host = new FakeHost();
    const renderer = makeRenderer();
    renderer.camera.background_color = "#ffffff";
    const created: InteractiveScene[] = [];
    const create = (sceneClass: SceneConstructor): Scene => {
      const scene = new sceneClass({ renderer, host });
      if (scene instanceof InteractiveScene) created.push(scene);
      return scene;
    };
    class WhiteScene extends InteractiveScene {
      construct(): void {}
    }
    const controller = new TwoViewPresentation([WhiteScene], {
      host,
      mode: "two-view",
      createScene: create,
    });
    const run = controller.run();
    await host.advance(3);
    expect(created.length).toBeGreaterThan(0);
    expect(host.dark).toBe(false);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("only construct: animation loops with replay but no interact", async () => {
    const host = new FakeHost();
    let constructs = 0;
    class AnimationOnly extends InteractiveScene {
      construct(): void {
        constructs += 1;
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([AnimationOnly], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(6);
    expect(constructs).toBeGreaterThan(1);
    expect(host.visible.get("replay")).toBe(true);
    expect(host.visible.get("interact")).toBe(false);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("only blueprint: starts in interaction with page navigation", async () => {
    const host = new FakeHost();
    const pages: number[] = [];
    class InteractionOnly extends InteractiveScene {
      setup(): void {
        this.set_storyboard([[this.first, null], [this.second, null]]);
      }
      first(): void {
        if (this.renderer?.skip_animations !== true) pages.push(1);
      }
      second(): void {
        if (this.renderer?.skip_animations !== true) pages.push(2);
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([InteractionOnly], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(4);
    expect(controller.state.view).toBe("interaction");
    expect(pages).toEqual([1]);
    expect(host.visible.get("next")).toBe(true);
    expect(host.visible.get("previous")).toBe(false);

    host.click("next");
    await host.advance(4);
    expect(pages).toEqual([1, 2]);
    expect(host.visible.get("next")).toBe(false);
    expect(host.visible.get("previous")).toBe(true);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("post_message carries the run id, scene and page without author ids", async () => {
    const host = new FakeHost();
    const messages: WanimMessage[] = [];
    class Exercise extends InteractiveScene {
      blueprint(): void {
        this.post_message("passed", true);
        this.post_message("score", 3);
        this.add(new Circle(0.2));
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([Exercise], {
      host,
      mode: "two-view",
      instanceId: "lesson-7",
      createScene: created.create,
      onMessage: (message) => messages.push(message),
    });
    const run = controller.run();
    await host.advance(4);
    expect(messages.length).toBeGreaterThanOrEqual(1);
    expect(messages[0]).toMatchObject({
      id: "lesson-7",
      key: "passed",
      value: true,
      scene: "Exercise",
      sceneIndex: 0,
      pageIndex: 0,
      phaseKind: "blueprint",
    });
    expect(messages.some((message) => message.key === "score" && message.value === 3)).toBe(true);

    controller.abort();
    await host.advance(2);
    await run;
  });

  test("theme is readable from setup", async () => {
    const host = new FakeHost();
    const themes: string[] = [];
    class Themed extends InteractiveScene {
      setup(): void {
        themes.push(this.theme);
      }
      construct(): void {}
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([Themed], {
      host,
      mode: "two-view",
      theme: "dark",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(3);
    expect(themes[0]).toBe("dark");
    controller.abort();
    await host.advance(2);
    await run;
  });

  test("double-click toggles from animation to interaction", async () => {
    const host = new FakeHost();
    const gate = deferred();
    class Demo extends InteractiveScene {
      blueprint(): void {
        this.add(new Circle(0.2));
      }
      construct(): unknown {
        return gate.promise;
      }
    }
    const created = factory(host);
    const controller = new TwoViewPresentation([Demo], {
      host,
      mode: "two-view",
      createScene: created.create,
    });
    const run = controller.run();
    await host.advance(2);
    expect(controller.state.view).toBe("animation");
    host.doubleClick();
    await host.advance(1);
    gate.release();
    await host.advance(4);
    expect(controller.state.view).toBe("interaction");
    controller.abort();
    await host.advance(2);
    await run;
  });
});
