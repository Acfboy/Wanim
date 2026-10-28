import { describe, expect, it } from "vitest";
import { create_frame_scheduler } from "../src/utils/scheduler.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Mobject } from "../src/mobject/mobject.js";
import { MathTex } from "../src/mobject/text/tex_mobject.js";
import { clearCanvas, setCanvasTransform, PreviewCamera } from "../src/renderer/canvas.js";
import { mobject_geometry_epoch, mobject_structure_epoch } from "../src/utils/render_epoch.js";
import { AnimatedBoundary } from "../src/animation/changing.js";
import { RED } from "../src/utils/color/core.js";

describe("review fixes", () => {
  it("pacing tolerates a 60 Hz frame pair just under 1000/30 ms", async () => {
    let t = 0;
    const scheduler = create_frame_scheduler({
      fps: 30,
      now: () => t,
      wait_for_frame: (callback) => {
        setTimeout(() => {
          // rAF timestamps on a 60 Hz display are commonly 16.666666666666665,
          // whose pair is ~7e-15 ms below 1000/30.
          t += 16.666666666666665;
          callback(t);
        }, 0);
      },
    });
    await scheduler.next_frame();
    // Without the tolerance this skipped to the third tick (~50 ms).
    expect(t).toBeLessThan(40);
    expect(t).toBeCloseTo(33.33333333333333, 6);
  });

  it("VGroup.__setitem__ supports negative indices and rejects out-of-range keys", () => {
    const first = new Circle();
    const second = new Square();
    const replacement = new Circle();
    const group = new VGroup(first, second);
    group.__setitem__(-1, replacement);
    expect(group.submobjects[1]).toBe(replacement);
    expect(Object.keys(group)).not.toContain("-1");

    expect(() => group.__setitem__(99, new Circle())).toThrowError(/index out of range/);
  });

  it("uses __argnames__ rather than arity to decide whether an updater takes dt", () => {
    const mob = new Mobject();
    const withDt: Array<number | undefined> = [];
    const dtUpdater = (m: Mobject, dt: number) => {
      withDt.push(dt);
      void m;
    };
    (dtUpdater as unknown as { __argnames__: string[] }).__argnames__ = ["m", "dt"];
    mob.add_updater(dtUpdater);
    mob.update(0.25);
    expect(withDt).toEqual([0.25]);

    const noDt: Array<number | undefined> = [];
    const lambda = (m: Mobject, t: number) => {
      noDt.push(t);
      void m;
    };
    (lambda as unknown as { __argnames__: string[] }).__argnames__ = ["m", "t"];
    mob.clear_updaters();
    mob.add_updater(lambda);
    mob.update(0.5);
    // The second argument is not `dt`, so it must be called with one argument;
    // `t` stays undefined rather than silently becoming the elapsed time.
    expect(noDt).toEqual([undefined]);
  });

  it("MathTex cache keys include z_index", () => {
    const a = new MathTex("q");
    expect(a.z_index).toBe(0);
    const b = new MathTex("q", { z_index: 7 });
    expect(b.z_index).toBe(7);
  });
});

describe("HiDPI canvas", () => {
  it("scales the camera transform and background fill by pixelRatio", () => {
    const calls: unknown[][] = [];
    const ctx = {
      setTransform: (...args: unknown[]) => calls.push(["setTransform", ...args]),
      fillRect: (...args: unknown[]) => calls.push(["fillRect", ...args]),
      fillStyle: "",
    } as unknown as CanvasRenderingContext2D;
    const camera = new PreviewCamera({
      pixel_width: 100,
      pixel_height: 50,
      frame_width: 10,
      frame_height: 5,
    });
    setCanvasTransform(ctx, camera, 2);
    expect(calls[0]).toEqual(["setTransform", 20, 0, 0, -20, 100, 50]);

    calls.length = 0;
    clearCanvas(ctx, camera, "rgba(0,0,0,1)", 2);
    expect(calls[1]).toEqual(["fillRect", 0, 0, 200, 100]);
  });
});

describe("epoch bumps", () => {
  it("structural edits invalidate both geometry and structure caches", () => {
    const group = new VGroup(new Circle(), new Square());
    const geometry = mobject_geometry_epoch();
    const structure = mobject_structure_epoch();
    group.add(new Circle());
    expect(mobject_geometry_epoch()).toBeGreaterThan(geometry);
    expect(mobject_structure_epoch()).toBeGreaterThan(structure);
  });

  it("sort/invert and set_z_index invalidate their caches", () => {
    const group = new VGroup(new Square(), new Circle());
    const geometry = mobject_geometry_epoch();
    const structure = mobject_structure_epoch();
    group.sort((point) => point[0] ?? 0);
    expect(mobject_structure_epoch()).toBeGreaterThan(structure);
    expect(mobject_geometry_epoch()).toBeGreaterThan(geometry);

    const beforeZ = mobject_geometry_epoch();
    group.set_z_index(3);
    expect(mobject_geometry_epoch()).toBeGreaterThan(beforeZ);
  });
});

describe("AnimatedBoundary negative index", () => {
  it("wraps to the last color when index is 0", () => {
    const boundary = new AnimatedBoundary(new Square(), [RED], 3, 0.5, true);
    boundary.total_time = 2;
    boundary.update_boundary_copies(0);
    const fading = boundary.boundary_copies[1] as unknown as { get_stroke_color(): { to_hex(): string } };
    expect(fading.get_stroke_color().to_hex()).toBe(RED.to_hex());
  });
});
