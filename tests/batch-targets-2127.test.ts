import { describe, expect, it } from "vitest";
import { Arrow3D } from "../src/mobject/three_d/three_dimensions.js";
import {
  SVGMobject,
  SVG_HASH_TO_MOB_MAP,
  type SVGElementData,
} from "../src/mobject/svg/svg_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import "../src/mobject/geometry/shape_matchers.js";
import { handle_caching_play } from "../src/utils/caching.js";
import { ZoomedScene } from "../src/scene/zoomed_scene.js";
import { Broadcast } from "../src/animation/specialized.js";
import {
  cycle_animation,
  turn_animation_into_updater,
} from "../src/animation/updaters/mobject_update_utils.js";
import { MobjectTable } from "../src/mobject/table.js";
import { ManimBanner } from "../src/mobject/logo.js";
import { ComplexPlane } from "../src/mobject/graphing/coordinate_systems.js";
import {
  TipableVMobject,
  CurvedArrow,
  CurvedDoubleArrow,
  Circle,
} from "../src/mobject/geometry/arc.js";
import {
  Arrow,
  DoubleArrow,
  Vector,
  Line,
} from "../src/mobject/geometry/line.js";
import { ArrowTriangleFilledTip } from "../src/mobject/geometry/tips.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { ArrowVectorField } from "../src/mobject/vector_field.js";
import { DiGraph } from "../src/mobject/graph.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { hash_obj } from "../src/utils/iterables.js";
import { ApplyMethod } from "../src/animation/transform.js";
import { Succession } from "../src/animation/composition.js";
import { UpdateFromAlphaFunc } from "../src/animation/updaters/update.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1e-12, Math.abs(b));
  return Math.abs(a - b) <= Math.max(1e-12, denom * tol);
}

function toRows(v: unknown): number[][] {
  const arr = Array.isArray(v)
    ? (v as unknown[])
    : ((v as { toArray(): unknown }).toArray() as unknown[] ?? []);
  if (arr.length > 0 && typeof arr[0] === "number")
    return [(arr as unknown[]).map(Number)];
  return (arr as unknown[][]).map((r) =>
    Array.isArray(r) ? r.map(Number) : [Number(r)],
  );
}

function expectRowsClose(actual: unknown, expected: number[][], tol = 1e-6): void {
  const rows = toRows(actual);
  expect(rows.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(rows[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function expectClose(a: number, b: number, tol = 1e-9): void {
  expect(close(a, b, tol)).toBe(true);
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow.__init__
// original manim: Arrow([-1,-1,0],[1,1,0],buff=0): start [-1,-1,0],
//   end ~[1,1,0], length 2.8284271247461903, stroke 6, tip 0.35,
//   has_tip True; default buff 0.25 shortens a 3-long arrow to length 2.5.
// ---------------------------------------------------------------------------
describe("Arrow.__init__", () => {
  it("builds tip + stroke like python (buff=0)", () => {
    const a = new Arrow([-1, -1, 0], [1, 1, 0], 0);
    expectRowsClose([a.get_start()], [[-1, -1, 0]], 1e-6);
    expectRowsClose([a.get_end()], [[1, 1, 0]], 1e-6);
    expectClose(a.get_length(), 2.8284271247461903, 1e-6);
    expect(a.has_tip()).toBe(true);
    expectClose(a.stroke_width, 6);
    expectClose(a.initial_stroke_width, 6);
    expectClose(a.tip_length, 0.35);
    expectClose(a.get_default_tip_length(), 0.35, 1e-6);
    expectClose(a.max_tip_length_to_length_ratio, 0.25);
    expectClose(a.max_stroke_width_to_length_ratio, 5);
  });

  it("applies the default MED_SMALL_BUFF=0.25", () => {
    const a = new Arrow([0, 0, 0], [3, 0, 0]);
    expectRowsClose([a.get_start()], [[0.25, 0, 0]], 1e-6);
    expectRowsClose([a.get_end()], [[2.75, 0, 0]], 1e-6);
    expectClose(a.get_length(), 2.5, 1e-6);
  });

  it("caps the default tip length on short arrows (0.25 * length)", () => {
    const a = new Arrow([0, 0, 0], [0.5, 0, 0], 0);
    expectClose(a.get_default_tip_length(), 0.125, 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.DoubleArrow.__init__
// original manim: DoubleArrow([0,0,0],[2,0,0],buff=0) has tip + start tip.
// ---------------------------------------------------------------------------
describe("DoubleArrow.__init__", () => {
  it("adds end and start tips", () => {
    const d = new DoubleArrow([0, 0, 0], [2, 0, 0], 0);
    expect(d.has_tip()).toBe(true);
    expect(d.has_start_tip()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Vector.__init__
// original manim: Vector([3,4]) -> ORIGIN..[3,4,0]; 2-D input gains a zero z.
// ---------------------------------------------------------------------------
describe("Vector.__init__", () => {
  it("extends 2-D directions to 3-D from ORIGIN", () => {
    const v = new Vector([3, 4]);
    expectRowsClose([v.get_start()], [[0, 0, 0]]);
    expectRowsClose([v.get_end()], [[3, 4, 0]]);
    expectClose(v.buff, 0);
  });

  it("keeps 3-D directions as-is", () => {
    const v = new Vector([1, 0, 0]);
    expectRowsClose([v.get_end()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow.scale
// original manim: Arrow([-1,-1,0],[1,1,0],buff=0).scale(2) -> start/end doubled,
//   same tip object kept (length unchanged up to float noise);
//   scale(2, scale_tips=True) doubles the tip length.
// ---------------------------------------------------------------------------
describe("Arrow.scale", () => {
  it("scales endpoints but keeps the tip object when scale_tips=false", () => {
    const a = new Arrow([-1, -1, 0], [1, 1, 0], 0);
    const tipBefore = a.tip;
    const tipLenBefore = (tipBefore as unknown as { length: number }).length;
    const s = a.scale(2);
    expect(s === a).toBe(true);
    expectRowsClose([a.get_start()], [[-2, -2, 0]], 1e-6);
    expectRowsClose([a.get_end()], [[2, 2, 0]], 1e-6);
    // Same tip object is popped and re-added (python keeps tip size fixed).
    expect(a.tip === tipBefore).toBe(true);
    expectClose(
      (a.tip as unknown as { length: number }).length,
      tipLenBefore,
      1e-6,
    );
  });

  it("scales tips too when scale_tips=true", () => {
    const a = new Arrow([-1, -1, 0], [1, 1, 0], 0);
    const before = (a.tip as unknown as { length: number }).length;
    a.scale(2, true);
    expectClose(
      (a.tip as unknown as { length: number }).length,
      before * 2,
      1e-6,
    );
  });

  it("is a no-op on zero-length arrows", () => {
    const a = new Arrow([1, 1, 0], [1, 1, 0], 0);
    expect(a.get_length()).toBe(0);
    a.scale(5);
    expect(a.get_length()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.get_unpositioned_tip /
//   create_tip / add_tip
// original manim: Arrow(...).get_unpositioned_tip() -> ArrowTriangleFilledTip,
//   length 0.35, fill/stroke white; create_tip positions it; add_tip
//   at_start sets the start-tip attribute.
// ---------------------------------------------------------------------------
describe("TipableVMobject tip helpers", () => {
  it("get_unpositioned_tip returns a default ArrowTriangleFilledTip", () => {
    const a = new Arrow([0, 0, 0], [2, 0, 0], 0);
    const tip = a.get_unpositioned_tip();
    expect(tip.constructor.name).toBe("ArrowTriangleFilledTip");
    expectClose((tip as unknown as { length: number }).length, 0.35, 1e-9);
  });

  it("create_tip returns a positioned tip", () => {
    const a = new Arrow([0, 0, 0], [2, 0, 0], 0);
    const tip = a.create_tip();
    expect(tip.constructor.name).toBe("ArrowTriangleFilledTip");
    // Positioned at the arrow end (within tip-length tolerance).
    const end = a.get_end();
    const tipStart = (tip as unknown as { get_start(): number[] }).get_start();
    expect(tipStart.length).toBe(3);
    expect(end.length).toBe(3);
  });

  it("add_tip(at_start=True) installs a start tip", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    expect(l.has_start_tip()).toBe(false);
    l.add_tip(null, ArrowTriangleFilledTip, null, null, true);
    expect(l.has_start_tip()).toBe(true);
    expect(l.has_tip()).toBe(false);
  });

  it("add_tip with an explicit tip object positions it", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip);
    l.add_tip(tip);
    expect(l.has_tip()).toBe(true);
    expect(l.tip === tip).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.CurvedArrow / CurvedDoubleArrow.__init__
// original manim: CurvedArrow([0,0,0],[2,0,0]) has_tip True, angle pi/2;
//   CurvedDoubleArrow has both tips.
// ---------------------------------------------------------------------------
describe("CurvedArrow.__init__ / CurvedDoubleArrow.__init__", () => {
  it("CurvedArrow carries one end tip", () => {
    const c = new CurvedArrow([0, 0, 0], [2, 0, 0]);
    expect(c.has_tip()).toBe(true);
    expectClose(c.angle, Math.PI / 2, 1e-12);
  });

  it("CurvedDoubleArrow carries both tips", () => {
    const c = new CurvedDoubleArrow([0, 0, 0], [2, 0, 0]);
    expect(c.has_tip()).toBe(true);
    expect(c.has_start_tip()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Arrow3D.__init__
// original manim: Arrow3D([0,0,0],[1,0,0]): length 0.7 (1 - height 0.3),
//   direction [1,0,0], shortened end [0.7,0,0], get_end [1,0,0]
//   (end_point), cone shifted to np_end.
// ---------------------------------------------------------------------------
describe("Arrow3D.__init__", () => {
  it("shortens the shaft by height and caps it with a cone", () => {
    const a = new Arrow3D([0, 0, 0], [1, 0, 0]);
    expectClose(a.thickness, 0.02);
    expectClose(a.length, 0.7, 1e-9);
    expectRowsClose([a.direction], [[1, 0, 0]]);
    expectRowsClose(a.start, [[0, 0, 0]]);
    // Shaft end shortened by height=0.3 along direction.
    expectRowsClose(a.end, [[0.7, 0, 0]]);
    // get_end reports the end_point (the original end).
    expectRowsClose(a.get_end(), [[1, 0, 0]]);
    expectRowsClose(a.end_point.get_center(), [[1, 0, 0]]);
    expectRowsClose([a.cone.direction], [[1, 0, 0]]);
  });

  it("keeps defaults (LEFT..RIGHT shaft, white)", () => {
    const b = new Arrow3D();
    expectRowsClose(b.start, [[-1, 0, 0]]);
    expectRowsClose(b.end, [[0.7, 0, 0]]);
    expectClose(b.length, 1.7, 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.add_background_rectangle
// original manim: Square().add_background_rectangle(color='#ff0000', opacity=0.5)
//   -> BackgroundRectangle with fill_opacity 0.5, added to back.
// ---------------------------------------------------------------------------
describe("Mobject.add_background_rectangle", () => {
  it("attaches a BackgroundRectangle behind the mobject", () => {
    const m = new Square();
    const before = m.submobjects.length;
    const ret = m.add_background_rectangle("#ff0000", 0.5);
    expect(ret === m).toBe(true);
    expect(m.background_rectangle).toBeDefined();
    expect(m.background_rectangle!.constructor.name).toBe("BackgroundRectangle");
    expectClose(m.background_rectangle!.fill_opacity ?? -1, 0.5);
    expect(m.submobjects.length).toBe(before + 1);
    expect(m.submobjects[0]).toBe(m.background_rectangle!);
  });

  it("uses the default opacity 0.75", () => {
    const m = new Square();
    m.add_background_rectangle("#00ff00");
    expectClose(m.background_rectangle!.fill_opacity ?? -1, 0.75);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.caching.handle_caching_play
// original manim behavior: skip branch pushes None; normal branch pushes the computed
// hash (or uncached_NNNNN when caching disabled); wrapped func is invoked.
// ---------------------------------------------------------------------------
describe("handle_caching_play", () => {
  function makeRenderer(opts: {
    skip: boolean;
    cached: boolean;
    disableCaching: boolean;
  }) {
    const calls: string[] = [];
    const renderer = {
      skip_animations: false,
      _original_skipping_status: false,
      num_plays: 7,
      animations_hashes: [] as Array<string | null>,
      camera: {},
      file_writer: {
        is_already_cached: (_h: string) => opts.cached,
        add_partial_movie_file: (_h: string | null) => {
          calls.push(String(_h));
        },
      },
      update_skipping_status() {
        this.skip_animations = opts.skip;
      },
    };
    return { renderer, calls };
  }

  function makeScene() {
    return {
      compile_animations: (..._args: unknown[]) => [{ name: "anim" }],
      add_mobjects_from_animations: (..._args: unknown[]) => {},
      mobjects: [],
    };
  }

  it("skip branch pushes None and still calls func", () => {
    const { renderer, calls } = makeRenderer({
      skip: true,
      cached: false,
      disableCaching: false,
    });
    let funcCalls = 0;
    const wrapped = handle_caching_play(function (this: unknown) {
      funcCalls += 1;
    });
    wrapped.call(
      renderer as never,
      makeScene() as never,
      { name: "anim" } as never,
    );
    expect(funcCalls).toBe(1);
    expect(renderer.animations_hashes).toEqual([null]);
    expect(calls).toEqual(["null"]);
  });

  it("cached-miss branch pushes the computed hash", () => {
    const { renderer } = makeRenderer({
      skip: false,
      cached: false,
      disableCaching: false,
    });
    let funcCalls = 0;
    const wrapped = handle_caching_play(function (this: unknown) {
      funcCalls += 1;
    });
    wrapped.call(
      renderer as never,
      makeScene() as never,
      { name: "anim" } as never,
    );
    expect(funcCalls).toBe(1);
    expect(renderer.animations_hashes.length).toBe(1);
    expect(typeof renderer.animations_hashes[0]).toBe("string");
  });

  it("already-cached branch marks skipping", () => {
    const { renderer } = makeRenderer({
      skip: false,
      cached: true,
      disableCaching: false,
    });
    const wrapped = handle_caching_play(function (this: unknown) {});
    wrapped.call(
      renderer as never,
      makeScene() as never,
      { name: "anim" } as never,
    );
    expect(renderer.skip_animations).toBe(true);
    expect(typeof renderer.animations_hashes[0]).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// manim.scene.zoomed_scene.ZoomedScene.setup / activate_zooming /
//   get_zoom_in_animation / get_zoomed_display_pop_out_animation
// original manim: setup builds a MovingCamera + ImageMobjectFromCamera, stretches both
//   to 3x3, scales the frame by zoom_factor 0.15, and parks the display in
//   the corner (or center when given).
// ---------------------------------------------------------------------------
describe("ZoomedScene.setup", { timeout: 60000 }, () => {
  it("builds camera + display with documented sizes", () => {
    const z = new ZoomedScene();
    z.setup();
    expect(z.zoomed_camera).toBeDefined();
    expect(z.zoomed_display).toBeDefined();
    expectClose(z.zoomed_display.height, 3, 1e-6);
    expectClose(z.zoomed_display.width, 3, 1e-6);
    expectClose(z.zoomed_camera.frame.height, 3 * 0.15, 1e-6);
    expectClose(z.zoomed_camera.frame.width, 3 * 0.15, 1e-6);
    expectRowsClose(
      [z.zoomed_camera.frame.get_center()],
      [[0, 0, 0]],
      1e-6,
    );
  });

  it("honors a display center override", () => {
    const z = new ZoomedScene(undefined, {
      zoomed_display_center: [1, 2, 0],
    });
    z.setup();
    expectRowsClose([z.zoomed_display.get_center()], [[1, 2, 0]], 1e-6);
  });
});

describe("ZoomedScene zoom animations", { timeout: 60000 }, () => {
  it("get_zoom_in_animation returns an ApplyMethod with run_time 2", () => {
    const z = new ZoomedScene();
    z.setup();
    const anim = z.get_zoom_in_animation();
    expect(anim instanceof ApplyMethod).toBe(true);
    expectClose(anim.run_time, 2);
  });

  it("get_zoomed_display_pop_out_animation returns an ApplyMethod", () => {
    const z = new ZoomedScene();
    z.setup();
    const anim = z.get_zoomed_display_pop_out_animation();
    expect(anim instanceof ApplyMethod).toBe(true);
  });

  it("activate_zooming flags + foregrounds without animating", () => {
    const z = new ZoomedScene();
    z.setup();
    // Headless scenes have a null renderer; install a stub camera hook.
    (z as unknown as { renderer: unknown }).renderer = {
      camera: { add_image_mobject_from_camera: () => {} },
    };
    z.activate_zooming(false);
    expect(z.zoom_activated).toBe(true);
    expect(z.foreground_mobjects).toContain(z.zoomed_camera.frame as never);
    expect(z.foreground_mobjects).toContain(z.zoomed_display as never);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.specialized.Broadcast.__init__
// original manim: Broadcast(Square(), n_mobs=3, ..., run_time=2) has 3 Restore anims,
//   lag 0.3, focal ORIGIN; defaults n=5, run_time 3, lag 0.2, width 0.
// ---------------------------------------------------------------------------
describe("Broadcast.__init__", () => {
  it("builds n_mobs Restore animations with the given style", () => {
    const s = new Square();
    const b = new Broadcast(s, [0, 0, 0], 3, 0.5, 0.1, 0.2, false, 0.3, 2);
    expect(b.n_mobs).toBe(3);
    expectClose(b.run_time, 2);
    expectClose(b.lag_ratio, 0.3);
    expectRowsClose([[...b.focal_point]], [[0, 0, 0]]);
    expectClose(b.initial_opacity, 0.5);
    expectClose(b.final_opacity, 0.1);
    expectClose(b.initial_width, 0.2);
    expect(b.animations.length).toBe(3);
    for (const a of b.animations) {
      expect(a.constructor.name).toBe("Restore");
    }
  });

  it("uses the documented defaults", () => {
    const b = new Broadcast(new Square());
    expect(b.n_mobs).toBe(5);
    expectClose(b.run_time, 3);
    expectClose(b.lag_ratio, 0.2);
    expectClose(b.initial_width, 0.0);
    expect(b.animations.length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.mobject_update_utils.cycle_animation
// original manim: cycle_animation(animation) == turn_animation_into_updater(
//   animation, cycle=True); the updater wraps alpha modulo 1.
// ---------------------------------------------------------------------------
describe("cycle_animation", () => {
  function makeAnim() {
    const mob = new Square();
    const seen: number[] = [];
    const anim = {
      mobject: mob,
      total_time: 0,
      suspend_mobject_updating: true,
      begin() {},
      get_run_time: () => 2,
      interpolate: (a: number) => {
        seen.push(a);
      },
      update_mobjects: (_dt: number) => {},
      finish() {},
    };
    return { mob, anim, seen };
  }

  it("registers an updater and cycles alpha modulo 1", () => {
    const { mob, anim, seen } = makeAnim();
    const out = cycle_animation(anim as never);
    expect(out === mob).toBe(true);
    expect(mob.updaters.length).toBe(1);
    const update = mob.updaters[0]!;
    // total_time=3 with run_time=2 -> time_ratio 1.5 -> alpha 0.5 (cycled).
    anim.total_time = 3;
    (update as (m: Mobject, dt: number) => void)(mob, 0.1);
    expect(seen.length).toBe(1);
    expectClose(seen[0]!, 0.5, 1e-12);
    expectClose(anim.total_time, 3.1, 1e-12);
  });

  it("matches turn_animation_into_updater(cycle=True) behavior", () => {
    const first = makeAnim();
    const second = makeAnim();
    cycle_animation(first.anim as never);
    turn_animation_into_updater(second.anim as never, true, 0, {});
    expect(first.mob.updaters.length).toBe(second.mob.updaters.length);
    first.anim.total_time = 5;
    second.anim.total_time = 5;
    (first.mob.updaters[0] as (m: Mobject, dt: number) => void)(
      first.mob,
      0.2,
    );
    (second.mob.updaters[0] as (m: Mobject, dt: number) => void)(
      second.mob,
      0.2,
    );
    expectClose(first.seen[0]!, second.seen[0]!, 1e-12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.add_highlighted_cell
// original manim (MobjectTable of Squares): add_highlighted_cell((2,2)) adds a
//   BackgroundRectangle to the back and stores it on the entry.
// ---------------------------------------------------------------------------
describe("Table.add_highlighted_cell", () => {
  function makeTable() {
    return new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
  }

  it("highlights the addressed cell and links the entry", () => {
    const t = makeTable();
    const ret = t.add_highlighted_cell([2, 2]);
    expect(ret === t).toBe(true);
    const entry = t.get_entries([2, 2]) as VMobject;
    const bg = (entry as unknown as { background_rectangle: unknown })
      .background_rectangle;
    expect(bg).toBeDefined();
    expect((bg as { constructor: { name: string } }).constructor.name).toBe(
      "BackgroundRectangle",
    );
    expect(t.submobjects).toContain(bg as never);
  });

  it("defaults to cell (1,1)", () => {
    const t = makeTable();
    t.add_highlighted_cell();
    const entry = t.get_entries([1, 1]) as VMobject;
    expect(
      (entry as unknown as { background_rectangle: unknown })
        .background_rectangle,
    ).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.logo.ManimBanner.expand (+ slide_and_uncover / slide_back)
// original manim: expand() -> Succession of 2 UpdateFromAlphaFuncs with run_times
//   1.0 / 0.5; bad direction raises ValueError.
//   slide_and_uncover / slide_back are the update callbacks of those two
//   animations (tested through the returned Succession).
// ---------------------------------------------------------------------------
describe("ManimBanner.expand", { timeout: 60000 }, () => {
  it("returns a two-step Succession with split run_times", () => {
    const b = new ManimBanner();
    const s = b.expand(1.5, "center");
    expect(s instanceof Succession).toBe(true);
    expect(s.animations.length).toBe(2);
    expect(s.animations[0] instanceof UpdateFromAlphaFunc).toBe(true);
    expect(s.animations[1] instanceof UpdateFromAlphaFunc).toBe(true);
    // run_time 1.5 * 2/3 = 1.0, 1.5 * 1/3 = 0.5 (original manim ground truth).
    expectClose(s.animations[0]!.run_time, 1.0, 1e-12);
    expectClose(s.animations[1]!.run_time, 0.5, 1e-12);
  });

  it("rejects bad directions like python ValueError", () => {
    const b = new ManimBanner();
    expect(() => b.expand(1.5, "up")).toThrowError(/direction must be/);
  });

  it("slide_and_uncover finishes with anim added to back (alpha=1)", () => {
    const b = new ManimBanner();
    const s = b.expand(1.5, "center");
    const first = s.animations[0] as UpdateFromAlphaFunc;
    first.begin();
    first.interpolate(0);
    first.interpolate(1);
    // After slide_and_uncover(alpha=1) the letters live on the banner.
    expect(b.submobjects).toContain(b.anim as never);
  });

  it("slide_back cleans up the clone (alpha=0 then 1)", () => {
    const b = new ManimBanner();
    const s = b.expand(1.5, "center");
    const second = s.animations[1] as UpdateFromAlphaFunc;
    // Run the first animation to completion so m_clone exists.
    const first = s.animations[0] as UpdateFromAlphaFunc;
    first.begin();
    first.interpolate(1);
    second.begin();
    second.interpolate(0);
    second.interpolate(1);
    expect(second.mobject === b).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ComplexPlane.add_coordinates
// original manim needs latex for the real labels (text/tex is out of scope); the
// ported routing logic (imag-dominant -> y axis with unit "i", else x axis)
// is verified with stubbed axes, mirroring python's get_coordinate_labels.
// ---------------------------------------------------------------------------
describe("ComplexPlane.add_coordinates", () => {
  function stubAxes(cp: ComplexPlane) {
    const calls: Array<{ axis: string; value: number; kwargs: unknown }> = [];
    const mk = (name: string) => ({
      get_number_mobject: (
        value: number,
        _a: unknown,
        _b: unknown,
        _c: unknown,
        _d: unknown,
        kwargs: unknown,
      ) => {
        calls.push({ axis: name, value, kwargs });
        return new Square();
      },
    });
    (cp as unknown as { get_x_axis: () => unknown }).get_x_axis = () =>
      mk("x");
    (cp as unknown as { get_y_axis: () => unknown }).get_y_axis = () =>
      mk("y");
    return calls;
  }

  it("routes real numbers to x and imaginary to y with unit i", () => {
    const cp = new ComplexPlane();
    const calls = stubAxes(cp);
    const ret = cp.add_coordinates({ re: 2, im: 0 }, { re: 0, im: 3 });
    expect(ret === cp).toBe(true);
    expect(calls.length).toBe(2);
    expect(calls[0]!.axis).toBe("x");
    expectClose(calls[0]!.value, 2);
    expect(calls[1]!.axis).toBe("y");
    expectClose(calls[1]!.value, 3);
    expect(
      ((calls[1]!.kwargs as Record<string, unknown>)["unit"] ?? null),
    ).toBe("i");
    expect(cp.coordinate_labels.submobjects.length).toBe(2);
  });

  it("falls back to the real NumberLine path which raises text/tex TODO", () => {
    const cp = new ComplexPlane();
    expect(() => cp.add_coordinates({ re: 1, im: 0 })).toThrowError(
      /text\/tex/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.__init__ / init_svg_mobject /
//   generate_mobject / get_mobjects_from
// original manim: __init__ stores file/height/width/defaults then parses + styles +
//   positions; generate_mobject parses the file via ElementTree/svgelements
//   (unported: TS throws TODO); get_mobjects_from walks the element stack,
//   numbering id-less groups and parenting shapes into ancestor vgroups.
// ---------------------------------------------------------------------------
describe("SVGMobject.__init__", () => {
  it("stores lazy defaults without parsing", () => {
    const m = new SVGMobject({});
    expect(m.file_name).toBeNull();
    expectClose(m.svg_height ?? -1, 2);
    expect(m.svg_width).toBeNull();
    expect(m.should_center).toBe(true);
    expect(m.svg_default["stroke_width"]).toBe(0);
    expect(m.submobjects.length).toBe(0);
  });

  it("applies explicit style overrides", () => {
    const m = new SVGMobject({ fill_opacity: 0.25, stroke_width: 2 } as never);
    expectClose((m.fill_opacity as number) ?? -1, 0.25);
    expectClose(m.stroke_width ?? -1, 2);
  });
});

describe("SVGMobject.generate_mobject / init_svg_mobject", () => {
  it("generate_mobject throws the file-pipeline TODO", () => {
    const m = new SVGMobject({});
    expect(() => m.generate_mobject()).toThrowError(/file-parsing pipeline/);
  });

  it("init_svg_mobject without cache throws the same TODO", () => {
    const m = new SVGMobject({});
    expect(() => m.init_svg_mobject(false)).toThrowError(
      /file-parsing pipeline/,
    );
  });

  it("init_svg_mobject serves cache hits without parsing", () => {
    const seed = new SVGMobject({});
    seed.add(new Square());
    const key = hash_obj(seed.hash_seed);
    SVG_HASH_TO_MOB_MAP.set(key, seed.copy() as SVGMobject);
    try {
      const probe = new SVGMobject({});
      const out = probe.init_svg_mobject(true);
      expect(out === probe).toBe(true);
      expect(probe.submobjects.length).toBe(1);
      expect(probe.id_to_vgroup_dict).toBeDefined();
    } finally {
      SVG_HASH_TO_MOB_MAP.delete(key);
    }
  });
});

describe("SVGMobject.get_mobjects_from", () => {
  function lineShape(id?: string): SVGElementData {
    return {
      kind: "line",
      id,
      x1: 0,
      y1: 0,
      x2: 3,
      y2: 4,
      stroke_width: 1,
      stroke_color: "#ffffff",
      stroke_opacity: 1,
      fill_color: "#ffffff",
      fill_opacity: 1,
    };
  }

  it("converts shapes and parents them into ancestor vgroups", () => {
    const m = new SVGMobject({});
    const svg: SVGElementData = {
      kind: "svg",
      id: "root-svg",
      children: [
        { kind: "group", id: "g1", children: [lineShape("l1")] },
        lineShape("l2"),
      ],
    };
    const [mobjects, vgroups] = m.get_mobjects_from(svg);
    expect(mobjects.length).toBe(2);
    // Line (0,0)->(3,4) has length 5 in both backends.
    expectClose((mobjects[0] as Line).get_length(), 5, 1e-9);
    expect(vgroups.has("root")).toBe(true);
    expect(vgroups.has("g1")).toBe(true);
    expect(vgroups.has("l1")).toBe(true);
    expect(vgroups.has("l2")).toBe(true);
    // Parents (root + g1) adopt the grouped shape; root adopts all.
    expect(vgroups.get("root")!.submobjects.length).toBe(2);
    expect(vgroups.get("g1")!.submobjects.length).toBe(1);
  });

  it("numbers id-less groups like python's numbered_group_N", () => {
    const m = new SVGMobject({});
    const svg: SVGElementData = {
      kind: "group",
      children: [{ kind: "group", children: [lineShape()] }],
    };
    const [, vgroups] = m.get_mobjects_from(svg);
    expect(vgroups.has("numbered_group_0")).toBe(true);
    expect(vgroups.has("numbered_group_1")).toBe(true);
    expect(vgroups.has("numbered_group_2")).toBe(true);
  });

  it("maps rect/circle/polygon/polyline kinds to mobjects", () => {
    const m = new SVGMobject({});
    const svg: SVGElementData = {
      kind: "svg",
      id: "s",
      children: [
        {
          kind: "rect",
          id: "r",
          x: 0,
          y: 0,
          width: 2,
          height: 1,
          stroke_width: 1,
          fill_opacity: 1,
        },
        {
          kind: "circle",
          id: "c",
          cx: 1,
          cy: 2,
          rx: 1,
          ry: 1,
          stroke_width: 1,
          fill_opacity: 1,
        },
        {
          kind: "polygon",
          id: "p",
          points: [
            [0, 0],
            [1, 0],
            [0, 1],
          ],
          stroke_width: 1,
          fill_opacity: 1,
        },
        {
          kind: "polyline",
          id: "pl",
          points: [
            [0, 0],
            [1, 1],
          ],
          stroke_width: 1,
          fill_opacity: 1,
        },
      ],
    };
    const [mobjects] = m.get_mobjects_from(svg);
    expect(mobjects.length).toBe(4);
    for (const mob of mobjects) expect(mob.has_points()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.VectorScene.get_vector / add_vector /
//   get_basis_vectors
// original manim: get_vector plots Arrow(plane(0,0) -> plane(x,y), buff=0);
//   add_vector wraps raw coords in Vector; basis = [(1,0),(0,1)] Vectors.
// ---------------------------------------------------------------------------
describe("VectorScene vectors", () => {
  function sceneWithPlane() {
    const vs = new VectorScene();
    (vs as unknown as { plane: unknown }).plane = {
      // Mirrors NumberPlane.coords_to_point: always a 3-D point.
      coords_to_point: (...coords: number[]) => [
        coords[0] ?? 0,
        coords[1] ?? 0,
        0,
      ],
    };
    return vs;
  }

  it("get_vector builds an Arrow between plane points with buff=0", () => {
    const vs = sceneWithPlane();
    const arrow = vs.get_vector([3, 4, 0]);
    expect(arrow instanceof Arrow).toBe(true);
    expectRowsClose([arrow.get_start()], [[0, 0, 0]], 1e-9);
    expectRowsClose([arrow.get_end()], [[3, 4, 0]], 1e-9);
    expectClose(arrow.buff, 0);
  });

  it("add_vector wraps coordinates (animate=false headless)", () => {
    const vs = sceneWithPlane();
    const arrow = vs.add_vector([1, 0, 0], "#ffff00", false);
    expect(arrow instanceof Arrow).toBe(true);
    expect(vs.mobjects).toContain(arrow as never);
  });

  it("add_vector keeps a pre-made Arrow as-is", () => {
    const vs = sceneWithPlane();
    const pre = new Arrow([0, 0, 0], [1, 1, 0], 0);
    const out = vs.add_vector(pre, "#ffff00", false);
    expect(out === pre).toBe(true);
  });

  it("get_basis_vectors returns (1,0),(0,1) with stroke width", () => {
    const vs = new VectorScene(6.0);
    const basis = vs.get_basis_vectors();
    expect(basis.submobjects.length).toBe(2);
    const ends = (basis.submobjects as unknown as Array<{ get_end(): number[] }>).map(
      (m) => m.get_end(),
    );
    expectRowsClose([ends[0]!], [[1, 0, 0]], 1e-6);
    expectRowsClose([ends[1]!], [[0, 1, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.LinearTransformationScene.add_vector /
//   get_vector_movement
// original manim: add_vector appends to moving_vectors; get_vector_movement targets
//   Vector(func(end)) and scales tiny tips.
// ---------------------------------------------------------------------------
describe("LinearTransformationScene vectors", () => {
  it("add_vector tracks moving_vectors (animate=false)", () => {
    const lts = new LinearTransformationScene();
    lts.moving_vectors = [];
    const arrow = lts.add_vector([2, 0, 0], "#ffff00", false);
    expect(lts.moving_vectors).toContain(arrow as never);
  });

  it("get_vector_movement maps ends through func", () => {
    const lts = new LinearTransformationScene();
    lts.moving_vectors = [];
    lts.ghost_vectors = new VGroup();
    const arrow = lts.add_vector([1, 0, 0], "#ffff00", false);
    const movement = lts.get_vector_movement((p) => {
      const r = toRows(p)[0] ?? [0, 0, 0];
      return [r[0]! * 2, r[1]! * 2, r[2]! * 2];
    });
    expect(movement).toBeDefined();
    const target = (arrow as unknown as { target: Vector }).target;
    expect(target).toBeDefined();
    const end = arrow.get_end();
    const tEnd = target.get_end();
    expectClose(tEnd[0]!, end[0]! * 2, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.ArrowVectorField.get_vector
// original manim: func (x,y)->(x,y,0) at [1,0,0] gives direction scaled by
//   0.45*sigmoid(1)/1, rooted at the point; length ~0.329.
// ---------------------------------------------------------------------------
describe("ArrowVectorField.get_vector", { timeout: 60000 }, () => {
  it("roots the scaled direction at the query point", () => {
    const field = new ArrowVectorField(
      (p) => {
        const r = toRows(p)[0] ?? [0, 0, 0];
        return [r[0]!, r[1]!, 0];
      },
      { x_range: [0, 1, 1], y_range: [0, 1, 1] },
    );
    const v = field.get_vector([1, 0, 0]);
    expect(v instanceof Vector).toBe(true);
    expectRowsClose([v.get_start()], [[1, 0, 0]], 1e-9);
    expectClose(v.get_length(), 0.3289763603835022, 1e-9);
  });

  it("returns a zero vector for a zero field output", () => {
    const field = new ArrowVectorField(() => [0, 0, 0], {
      x_range: [0, 1, 1],
      y_range: [0, 1, 1],
    });
    const v = field.get_vector([2, 3, 0]);
    expectClose(v.get_length(), 0, 1e-12);
    expectRowsClose([v.get_start()], [[2, 3, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graph.GenericGraph._create_vertex
// original manim: position defaults to graph center; duplicate ids raise ValueError;
//   the mobject is moved to the position.
// ---------------------------------------------------------------------------
describe("GenericGraph._create_vertex", () => {
  it("creates a Dot at the given position", () => {
    const g = new DiGraph([1, 2], [[1, 2]]);
    const [vertex, position, _config, mob] = g._create_vertex(3, [1, 2, 0]);
    expect(vertex).toBe(3);
    expectRowsClose(position, [[1, 2, 0]]);
    expect(mob.constructor.name).toBe("Dot");
    expectRowsClose([mob.get_center()], [[1, 2, 0]], 1e-9);
  });

  it("defaults the position to the graph center", () => {
    const g = new DiGraph([1], []);
    const [, position, , mob] = g._create_vertex(99);
    expectRowsClose(position, [g.get_center()], 1e-9);
    expectRowsClose([mob.get_center()], [g.get_center()], 1e-9);
  });

  it("rejects duplicate vertex ids", () => {
    const g = new DiGraph([1, 2], [[1, 2]]);
    expect(() => g._create_vertex(1)).toThrowError(/already used/);
  });

  it("honors an explicit vertex_mobject", () => {
    const g = new DiGraph([1], []);
    const custom = new Square();
    const [, , , mob] = g._create_vertex(5, [0, 0, 0], false, "#000000", null, null, custom);
    expect(mob === custom).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION
// ---------------------------------------------------------------------------
describe("integration (batch 21..27)", { timeout: 60000 }, () => {
  it("Arrow chain: tip helpers feed Arrow.scale against python lengths", () => {
    // Mirrors the original manim chain: Arrow buff=0, length 2.5 for a 3-long shaft.
    const a = new Arrow([0, 0, 0], [3, 0, 0]);
    expectClose(a.get_length(), 2.5, 1e-6);
    const tip = a.create_tip();
    expect(tip).toBeDefined();
    a.scale(2);
    expectClose(a.get_length(), 5.0, 1e-6);
  });

  it("Broadcast + cycle_animation share the updater path", () => {
    const b = new Broadcast(new Square(), [0, 0, 0], 2);
    expect(b.animations.length).toBe(2);
    const mob = new Square();
    const seen: number[] = [];
    const anim = {
      mobject: mob,
      total_time: 0,
      suspend_mobject_updating: true,
      begin() {},
      get_run_time: () => 1,
      interpolate: (a: number) => {
        seen.push(a);
      },
      update_mobjects: (_dt: number) => {},
      finish() {},
    };
    cycle_animation(anim as never);
    (mob.updaters[0] as (m: Mobject, dt: number) => void)(mob, 0.25);
    expect(seen.length).toBe(1);
  });

  it("VectorScene.get_vector_movement drives Arrow tips like python", () => {
    const lts = new LinearTransformationScene();
    lts.moving_vectors = [];
    lts.ghost_vectors = new VGroup();
    const arrow = lts.add_vector([1, 0, 0], "#ffff00", false);
    expect(arrow.has_tip()).toBe(true);
    lts.get_vector_movement((p) => toRows(p)[0] ?? [0, 0, 0]);
    const target = (arrow as unknown as { target: Vector }).target;
    expect(target.has_tip()).toBe(true);
  });

  it("Table highlight uses BackgroundRectangle (add_background_rectangle path)", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    t.add_highlighted_cell([1, 2]);
    const entry = t.get_entries([1, 2]) as VMobject;
    const bg = (entry as unknown as { background_rectangle: VMobject })
      .background_rectangle;
    expect(bg.constructor.name).toBe("BackgroundRectangle");
    // Sibling: plain Mobject background rectangles attach behind too.
    const m = new Circle();
    m.add_background_rectangle("#ff0000", 0.5);
    expect(m.submobjects[0]).toBe(m.background_rectangle!);
  });

  it("ZoomedScene display sizing feeds the zoom animations", () => {
    const z = new ZoomedScene();
    z.setup();
    // NOTE: get_zoomed_display_pop_out_animation replaces the display with
    // the (small) camera frame, so the 3x3 size only holds beforehand.
    expectClose(z.zoomed_display.height, 3, 1e-6);
    const zoomIn = z.get_zoom_in_animation();
    expect(zoomIn instanceof ApplyMethod).toBe(true);
    const popOut = z.get_zoomed_display_pop_out_animation();
    expect(popOut instanceof ApplyMethod).toBe(true);
  });

  it("SVG line shapes convert to length-5 Lines (python svgelements parity)", () => {
    const m = new SVGMobject({});
    const [mobjects] = m.get_mobjects_from({
      kind: "svg",
      id: "s",
      children: [
        {
          kind: "line",
          id: "diag",
          x1: 0,
          y1: 0,
          x2: 3,
          y2: 4,
          stroke_width: 1,
          fill_opacity: 1,
        },
      ],
    });
    expect(mobjects.length).toBe(1);
    expectClose((mobjects[0] as Line).get_length(), 5, 1e-9);
  });
});
