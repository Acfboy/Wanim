import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../src/scene/scene.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Create, DrawBorderThenFill } from "../src/animation/creation.js";
import { Transform } from "../src/animation/transform.js";
import { RED } from "../src/utils/color/core.js";
import { apply_quality, run_scene } from "../src/renderer/scene_runner.js";

const dirs: string[] = [];

function render(sceneClass: new (...args: any[]) => Scene): Record<string, any> {
  apply_quality("-ql");
  const dir = mkdtempSync(join(tmpdir(), "wanim-rj-"));
  dirs.push(dir);
  const path = join(dir, "scene.mjson");
  run_scene(sceneClass, path);
  return JSON.parse(readFileSync(path, "utf-8"));
}

afterEach(() => {
  while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true });
});

describe("JSONCairoRenderer play-level options", () => {
  it("forwards run_time to compile_animation_data (30 frames at 15fps for run_time=2)", () => {
    class R extends Scene {
      construct(): void {
        this.play(new Create(new Square()), { run_time: 2 });
      }
    }
    const doc = render(R);
    // 2s * 15fps = 30 frames.
    expect(doc.meta.num_frames).toBe(30);
  });

  it("default run_time is 1s (15 frames)", () => {
    class R extends Scene {
      construct(): void {
        this.play(new Create(new Square()));
      }
    }
    expect(render(R).meta.num_frames).toBe(15);
  });
});

describe("rgba interpolation broadcasts a single row across points", () => {
  it("DrawBorderThenFill with a filled target has no NaN points or rgbas", () => {
    class D extends Scene {
      construct(): void {
        const s = new Square();
        s.set_fill(RED, 1);
        this.play(new DrawBorderThenFill(s));
      }
    }
    const doc = render(D);
    expect(doc.meta.num_frames).toBe(30);
    for (const frame of doc.frames) {
      for (const mob of frame.m) {
        for (const row of mob.p) for (const value of row) expect(Number.isNaN(value)).toBe(false);
        for (const key of ["f", "s", "bs"]) {
          if (mob[key] === null) continue;
          for (const row of mob[key]) for (const value of row) expect(Number.isNaN(value)).toBe(false);
        }
      }
    }
  });

  it("Transform between mobjects with different rgba row counts stays finite", () => {
    class T extends Scene {
      construct(): void {
        const square = new Square();
        square.set_fill(RED, 1);
        const circle = new Circle();
        circle.set_fill(RED, 1);
        this.play(new Transform(square, circle));
      }
    }
    const doc = render(T);
    for (const frame of doc.frames) {
      for (const mob of frame.m) {
        if (mob.f === null) continue;
        for (const row of mob.f) for (const value of row) expect(Number.isNaN(value)).toBe(false);
      }
    }
  });
});
