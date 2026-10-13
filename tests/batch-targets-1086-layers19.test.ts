// Ground-truth tests for batch layers 19..19 (~1086 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): none of these symbols need the LaTeX pipeline.
//
// DEVIATIONS (documented inline):
//  (D1) TS Axes/NumberPlane/ThreeDAxes keep an EMPTY `axes` VGroup (no
//       NumberLine axes are constructed yet); Python builds real NumberLine
//       axes, so Python has background lines / unit sizes / axis ranges while
//       TS has empty groups here. get_x_axis/get_y_axis are exercised through
//       faithfully wired stand-in axes (same convention as
//       batch-targets-1021-layers19.test.ts).
//  (D2) TS BackgroundRectangle.set_style takes (fill_opacity, kwargs) while
//       Python takes (fill_opacity, **kwargs); observable behavior matches.
import { describe, expect, it, vi, afterEach } from "vitest";
import type { NDArray } from "numpy-ts";
import { Axes, NumberPlane, ThreeDAxes } from "../src/mobject/graphing/coordinate_systems.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Square, Triangle } from "../src/mobject/geometry/polygram.js";
import { Circle, Dot } from "../src/mobject/geometry/arc.js";
import { Line } from "../src/mobject/geometry/line.js";
import {
  SurroundingRectangle,
  BackgroundRectangle,
} from "../src/mobject/geometry/shape_matchers.js";
import {
  ArrowTriangleTip,
  ArrowTriangleFilledTip,
} from "../src/mobject/geometry/tips.js";
import { Scene } from "../src/scene/scene.js";
import { ThreeDScene } from "../src/scene/three_d_scene.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { Create } from "../src/animation/creation.js";
import { FadeOut } from "../src/animation/fading.js";
import { Transform } from "../src/animation/transform.js";
import { config } from "../src/_config/utils.js";
import { DOWN, PI } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

function expectRowsClose(actual: number[][] | { toArray(): unknown }, expected: number[][], tol = 1e-9): void {
  const expectedRows = (Array.isArray(expected)
    ? (expected as unknown[])
    : ((expected as { toArray(): unknown }).toArray() as unknown[])
  ).map((row) =>
    Array.isArray(row) ? row : ((row as { toArray(): unknown }).toArray() as number[]),
  ) as number[][];
  const actualRows = (Array.isArray(actual) ? actual : ((actual as { toArray(): unknown }).toArray() as unknown[])).map(
    (row) => (Array.isArray(row) ? row : (row as { toArray(): unknown }).toArray()),
  ) as number[][];
  expect(actualRows.length).toBe(expectedRows.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actualRows[i]?.length).toBe(expectedRows[i]?.length);
    for (let j = 0; j < (expectedRows[i]?.length ?? 0); j += 1) {
      expect(close(actualRows[i]?.[j] ?? NaN, expectedRows[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function plainRows(v: number[] | NDArray): number[][] {
  if (Array.isArray(v)) {
    return (v.length > 0 && Array.isArray(v[0]) ? v : [v]) as number[][];
  }
  return rowsOf(v as NDArray, 3);
}

function centerOf(m: Mobject): number[] {
  return (m as unknown as VMobject).get_center();
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.SurroundingRectangle.__init__
// Python (Square(side_length=2)):
//   SurroundingRectangle(sq): w 2.2, h 2.2, center [0,0,0], buff 0.1.
//   SurroundingRectangle(sq, buff=(0.5,1.0)): w 3.0, h 4.0.
//   Circle(radius=1).shift([2,0,0]): center [2,0,0], w/h 2.2.
//   Non-Mobject input raises TypeError.
// ---------------------------------------------------------------------------
describe("SurroundingRectangle.__init__", () => {
  it("pads with the default scalar buff", () => {
    const sr = new SurroundingRectangle(new Square(2));
    expectClose(sr.width, 2.2);
    expectClose(sr.height, 2.2);
    expectRowsClose([centerOf(sr)], [[0, 0, 0]]);
    expect(sr.buff).toBe(0.1);
  });

  it("pads axes independently with a tuple buff", () => {
    const sr = new SurroundingRectangle(new Square(2), { buff: [0.5, 1.0] });
    expectClose(sr.width, 3.0);
    expectClose(sr.height, 4.0);
    expect(sr.buff).toEqual([0.5, 1.0]);
  });

  it("moves to a shifted target", () => {
    const c = new Circle(1).shift([2, 0, 0]);
    const sr = new SurroundingRectangle(c as unknown as Mobject);
    expectRowsClose([centerOf(sr)], [[2, 0, 0]]);
    expectClose(sr.width, 2.2);
    expectClose(sr.height, 2.2);
  });

  it("rejects non-Mobject inputs with TypeError", () => {
    expect(() => new SurroundingRectangle("nope" as unknown as Mobject)).toThrowError(
      /Expected all inputs for parameter mobjects/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.BackgroundRectangle
// Python (Square(side_length=2)):
//   BackgroundRectangle(sq): fill 0.75, original 0.75, stroke_width 0,
//     w/h 2.0 (buff 0). color=WHITE fill 0.15 -> 0.15/0.15.
//   pointwise_become_partial(sq, 0, 0.5) -> fill 0.375.
//   set_style(0.3) -> fill 0.3, stroke_width 0.
// ---------------------------------------------------------------------------
describe("BackgroundRectangle", () => {
  it("uses scene-background defaults", () => {
    const bg = new BackgroundRectangle(new Square(2));
    expectClose(bg.fill_opacity, 0.75);
    expectClose(bg.original_fill_opacity, 0.75);
    expectClose(bg.stroke_width, 0);
    expectClose(bg.width, 2.0);
    expectClose(bg.height, 2.0);
  });

  it("honors explicit color/fill_opacity", () => {
    const bg = new BackgroundRectangle(new Square(2), { color: "WHITE", fill_opacity: 0.15 });
    expectClose(bg.fill_opacity, 0.15);
    expectClose(bg.original_fill_opacity, 0.15);
  });

  it("pointwise_become_partial scales the original fill opacity", () => {
    const sq = new Square(2);
    const bg = new BackgroundRectangle(sq as unknown as Mobject);
    expect(bg.pointwise_become_partial(sq as unknown as Mobject, 0, 0.5)).toBe(bg);
    expectClose(bg.fill_opacity, 0.375);
  });

  it("set_style only changes fill_opacity (D2)", () => {
    const bg = new BackgroundRectangle(new Square(2), { color: "WHITE", fill_opacity: 0.15 });
    bg.set_style(0.3);
    expectClose(bg.fill_opacity, 0.3);
    expectClose(bg.stroke_width, 0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add_mobjects_from_animations
// Python: FadeOut(sq) (non-introducer) adds sq (1 mobject, in family);
//   Create(sq) (introducer) is skipped (0 mobjects).
// ---------------------------------------------------------------------------
describe("Scene.add_mobjects_from_animations", () => {
  it("adds mobjects from non-introducer animations", () => {
    const s = new Scene();
    const sq = new Square();
    expect(new FadeOut(sq as unknown as Mobject).is_introducer()).toBe(false);
    s.add_mobjects_from_animations([new FadeOut(sq as unknown as Mobject)]);
    expect(s.mobjects.length).toBe(1);
    expect(s.get_mobject_family_members().includes(sq as unknown as Mobject)).toBe(true);
  });

  it("skips introducer animations", () => {
    const s = new Scene();
    const sq = new Square();
    expect(new Create(sq as unknown as VMobject).is_introducer()).toBe(true);
    s.add_mobjects_from_animations([new Create(sq as unknown as VMobject)]);
    expect(s.mobjects.length).toBe(0);
    expect(s.get_mobject_family_members().includes(sq as unknown as Mobject)).toBe(false);
  });

  it("does not duplicate mobjects already in the scene", () => {
    const s = new Scene();
    const sq = new Square();
    const other = new Square();
    s.add(sq as unknown as Mobject);
    const before = s.mobjects.length;
    s.add_mobjects_from_animations([new Transform(sq as unknown as Mobject, other as unknown as Mobject)]);
    expect(s.mobjects.length).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add_foreground_mobjects
// Python: add(m1); add_foreground_mobjects(m2) -> foreground [m2],
//   mobjects [m1, m2, m2] (add re-appends foreground members); returns self.
// ---------------------------------------------------------------------------
describe("Scene.add_foreground_mobjects", () => {
  it("tracks foreground members and returns self", () => {
    const s = new Scene();
    const m1 = new Square();
    const m2 = new Circle();
    s.add(m1 as unknown as Mobject);
    const ret = s.add_foreground_mobjects(m2 as unknown as Mobject);
    expect(ret).toBe(s);
    expect(s.foreground_mobjects.length).toBe(1);
    expect(s.foreground_mobjects[0]).toBe(m2 as unknown as Mobject);
    // Python: mobjects == [m1, m2, m2] (foreground re-appended by add).
    expect(s.mobjects.length).toBe(3);
    expect(s.mobjects[0]).toBe(m1 as unknown as Mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.become
// Python: Square(2).become(Circle(1)) returns self, 16 -> 32 points,
//   width stays 2.0, center origin. match_center keeps self center.
//   VGroup(Square,Circle).become(VGroup(Triangle,Dot)) keeps 2 submobjects.
// ---------------------------------------------------------------------------
describe("Mobject.become", () => {
  it("copies points and returns self", () => {
    const a = new Square(2);
    const b = new Circle(1);
    expect(a.get_num_points()).toBe(16);
    expect(b.get_num_points()).toBe(32);
    expect(a.become(b as unknown as Mobject)).toBe(a);
    expect(a.get_num_points()).toBe(32);
    expectClose(a.width, 2.0);
    expectRowsClose([centerOf(a)], [[0, 0, 0]]);
  });

  it("match_center keeps the caller's center", () => {
    // Python: Square().become(Circle().shift([3,0,0]), match_center=True)
    //   -> center stays [0,0,0].
    const a = new Square(2);
    const c = new Circle(1).shift([3, 0, 0]);
    a.become(c as unknown as Mobject, false, false, false, true);
    expectRowsClose([centerOf(a)], [[0, 0, 0]]);
  });

  it("works across matching groups", () => {
    const g1 = new VGroup(new Square(), new Circle());
    const g2 = new VGroup(new Triangle(), new Dot());
    g1.become(g2 as unknown as Mobject);
    expect(g1.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.begin_ambient_camera_rotation
// Python Cairo path: tracker gains 1 updater, tracker added to the scene;
//   updater increments by rate*dt. Bad `about` raises ValueError.
// ---------------------------------------------------------------------------
describe("ThreeDScene.begin_ambient_camera_rotation", () => {
  function trackerScene(): { scene: ThreeDScene; theta: ValueTracker; phi: ValueTracker; gamma: ValueTracker } {
    const scene = new ThreeDScene();
    const theta = new ValueTracker(0);
    const phi = new ValueTracker(0);
    const gamma = new ValueTracker(0);
    scene.camera = { theta_tracker: theta, phi_tracker: phi, gamma_tracker: gamma };
    return { scene, theta, phi, gamma };
  }

  it("adds an updater driving the theta tracker at the given rate", () => {
    const { scene, theta } = trackerScene();
    scene.begin_ambient_camera_rotation(0.02, "theta");
    expect(theta.updaters.length).toBe(1);
    expect(scene.mobjects.includes(theta as unknown as Mobject)).toBe(true);
    const update = theta.updaters[0] as (m: unknown, dt: number) => void;
    update(theta, 1.0);
    expectClose(theta.get_value(), 0.02);
    update(theta, 0.5);
    expectClose(theta.get_value(), 0.03);
  });

  it("targets the phi tracker when asked", () => {
    const { scene, theta, phi } = trackerScene();
    scene.begin_ambient_camera_rotation(0.5, "phi");
    expect(phi.updaters.length).toBe(1);
    expect(theta.updaters.length).toBe(0);
    (phi.updaters[0] as (m: unknown, dt: number) => void)(phi, 2.0);
    expectClose(phi.get_value(), 1.0);
  });

  it("rejects a bad `about` with ValueError", () => {
    const { scene } = trackerScene();
    expect(() => scene.begin_ambient_camera_rotation(0.02, "nonsense")).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.begin_3dillusion_camera_rotation
// Python (pure closure math, rate=1, theta origin 2.0, phi origin 1.0):
//   theta dt=0.5 -> 2.0+0.2*sin(0.5) = 2.0958851077208407, tracker 0.5;
//     second dt=0.5 -> 2.0+0.2*sin(1.0) = 2.1682941969615794.
//   phi dt=0.5 -> 1.0+0.1*cos(0.5)-0.1 = 0.9877582561890373;
//     second -> 1.0+0.1*cos(1.0)-0.1 = 0.9540302305868139.
// ---------------------------------------------------------------------------
describe("ThreeDScene.begin_3dillusion_camera_rotation", () => {
  function illusionScene(theta0: number, phi0: number): ThreeDScene {
    const scene = new ThreeDScene();
    const thetaTracker = new ValueTracker(theta0);
    const phiTracker = new ValueTracker(phi0);
    scene.renderer = { camera: { theta_tracker: thetaTracker, phi_tracker: phiTracker } };
    scene.begin_3dillusion_camera_rotation(1, phi0, theta0);
    return scene;
  }

  it("oscillates theta around its origin", () => {
    const scene = illusionScene(2.0, 1.0);
    const thetaTracker = scene.renderer.camera.theta_tracker as ValueTracker;
    expect(thetaTracker.updaters.length).toBe(1);
    expect(scene.mobjects.includes(thetaTracker as unknown as Mobject)).toBe(true);
    const update = thetaTracker.updaters[0] as (m: unknown, dt: number) => void;
    update(thetaTracker, 0.5);
    expectClose(thetaTracker.get_value(), 2.0958851077208407);
    update(thetaTracker, 0.5);
    expectClose(thetaTracker.get_value(), 2.1682941969615794);
  });

  it("oscillates phi around its origin", () => {
    const scene = illusionScene(2.0, 1.0);
    const phiTracker = scene.renderer.camera.phi_tracker as ValueTracker;
    expect(phiTracker.updaters.length).toBe(1);
    expect(scene.mobjects.includes(phiTracker as unknown as Mobject)).toBe(true);
    const update = phiTracker.updaters[0] as (m: unknown, dt: number) => void;
    update(phiTracker, 0.5);
    expectClose(phiTracker.get_value(), 0.9877582561890373);
    update(phiTracker, 0.5);
    expectClose(phiTracker.get_value(), 0.9540302305868139);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.add_fixed_orientation_mobjects /
//   .add_fixed_in_frame_mobjects
// Python Cairo paths: self.add(*mobjects) then delegate to the camera.
// ---------------------------------------------------------------------------
describe("ThreeDScene.add_fixed_*_mobjects", () => {
  it("add_fixed_orientation_mobjects adds and delegates to the camera", () => {
    const scene = new ThreeDScene();
    const mob = new Square();
    const camera = { add_fixed_orientation_mobjects: vi.fn() };
    scene.renderer = { camera };
    scene.add_fixed_orientation_mobjects(mob as unknown as Mobject);
    expect(scene.mobjects.includes(mob as unknown as Mobject)).toBe(true);
    expect(camera.add_fixed_orientation_mobjects).toHaveBeenCalledWith(
      mob,
      {},
    );
  });

  it("add_fixed_in_frame_mobjects adds and delegates to the camera", () => {
    const scene = new ThreeDScene();
    const mob = new Square();
    const camera = { add_fixed_in_frame_mobjects: vi.fn() };
    scene.camera = camera;
    scene.add_fixed_in_frame_mobjects(mob as unknown as Mobject);
    expect(scene.mobjects.includes(mob as unknown as Mobject)).toBe(true);
    expect(camera.add_fixed_in_frame_mobjects).toHaveBeenCalledWith(mob);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowTriangleFilledTip / ArrowTriangleTip.__init__
// Python: defaults fill 0, stroke 3, 0.35 x 0.35, 12 points;
//   tip_point [-0.425,~0,0], base [-0.075,~0,0], length 0.35, angle pi.
//   Filled: fill 1, stroke 0, same geometry.
//   Custom length=0.5/width=0.6 -> 0.5 x 0.6.
// ---------------------------------------------------------------------------
describe("ArrowTriangleTip.__init__ / ArrowTriangleFilledTip", () => {
  it("builds the default triangle tip", () => {
    const t = new ArrowTriangleTip();
    expectClose(t.width, 0.35);
    expectClose(t.height, 0.35);
    expect(t.get_num_points()).toBe(12);
    expectRowsClose(plainRows(t.tip_point), [[-0.42499999999999977, 0, 0]], 1e-9);
    expectRowsClose(plainRows(t.base), [[-0.07499999999999987, 0, 0]], 1e-9);
    expectClose(t.length, 0.35);
    // NOTE: atan2 branch: Python's (tip-base) y residual is +1e-16 (-> +pi)
    // while TS's is -7e-17 (-> -pi); both denote the same angle, so the
    // magnitude is compared (same convention as batch-targets-1115).
    expectClose(Math.abs(t.tip_angle), PI);
  });

  it("honors custom length/width", () => {
    const t = new ArrowTriangleTip(0, 3, 0.5, 0.6);
    expectClose(t.width, 0.5);
    expectClose(t.height, 0.6);
  });

  it("builds the filled variant with identical geometry", () => {
    const f = new ArrowTriangleFilledTip();
    expect(f).toBeInstanceOf(ArrowTriangleTip);
    expectClose(f.fill_opacity, 1);
    expectClose(f.stroke_width, 0);
    expectClose(f.width, 0.35);
    expectClose(f.height, 0.35);
    expect(f.get_num_points()).toBe(12);
    expectRowsClose(plainRows(f.tip_point), [[-0.42499999999999977, 0, 0]], 1e-9);
    expectClose(f.length, 0.35);
  });

  it("builds the filled variant at a custom size via options", () => {
    const f = new ArrowTriangleFilledTip(1, 0, { length: 0.5 });
    expectClose(f.width, 0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.get_tick
// Python (x_range [-3,3,1], length 6):
//   get_tick(1.0): start [1,-0.1,0], end [1,0.1,0], 4 points,
//     center [1,0,0] == n2p(1.0), length 0.2, stroke 2.0, white.
//   get_tick(0, 0.2): [0,-0.2,0] -> [0,0.2,0].
// ---------------------------------------------------------------------------
describe("NumberLine.get_tick", () => {
  it("builds the default tick at the number's point", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const tick = nl.get_tick(1.0);
    expect(tick).toBeInstanceOf(Line);
    expect(tick.get_num_points()).toBe(4);
    expectRowsClose([tick.get_start()], [[1, -0.1, 0]]);
    expectRowsClose([tick.get_end()], [[1, 0.1, 0]]);
    expectRowsClose([tick.get_center()], plainRows(nl.n2p(1.0)));
    expectClose(tick.get_length(), 0.2);
  });

  it("honors an explicit size", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const tick = nl.get_tick(0, 0.2);
    expectRowsClose([tick.get_start()], [[0, -0.2, 0]]);
    expectRowsClose([tick.get_end()], [[0, 0.2, 0]]);
    expectClose(tick.get_length(), 0.4);
  });

  it("matches the line style", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const tick = nl.get_tick(1.0);
    expectClose(tick.get_stroke_width(), 2.0);
    expect(tick.get_color().to_hex()).toBe(nl.get_color().to_hex());
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ThreeDAxes
// Python: x_range (-6,6,1), y_range (-5,5,1), z_range (-4,4,1);
//   x/y/z_length 10.5/10.5/6.5, dimension 3, light_source [-7,-9,10],
//   num_axis_pieces 20, z_normal DOWN.
//   c2p(0,0) [0,0,0]; c2p(6,0) [5.25,0,0]; c2p(-6,-5) [-5.25,-5.25,0].
// (D1): TS keeps `axes` empty, so axis objects/unit sizes are not available.
// ---------------------------------------------------------------------------
describe("ThreeDAxes", () => {
  it("stores Python's default ranges and lengths", () => {
    const td = new ThreeDAxes();
    expect([...td.x_range]).toEqual([-6, 6, 1]);
    expect([...td.y_range]).toEqual([-5, 5, 1]);
    expect([...td.z_range]).toEqual([-4, 4, 1]);
    expectClose(td.x_length ?? NaN, 10.5);
    expectClose(td.y_length ?? NaN, 10.5);
    expectClose(td.z_length ?? NaN, 6.5);
    expect(td.dimension).toBe(3);
    expect(td.light_source).toEqual([-7, -9, 10]);
    expect(td.num_axis_pieces).toBe(20);
    expect(td.z_normal).toEqual([...DOWN]);
  });

  it("maps coordinates with the linear length map", () => {
    const td = new ThreeDAxes();
    expectRowsClose(plainRows(td.coords_to_point(0, 0)), [[0, 0, 0]]);
    expectRowsClose(plainRows(td.coords_to_point(6, 0)), [[5.25, 0, 0]]);
    expectRowsClose(plainRows(td.coords_to_point(-6, -5)), [[-5.25, -5.25, 0]], 1e-9);
  });

  it("builds x/y/z NumberLine axes like python", () => {
    const td = new ThreeDAxes();
    expect(td.get_axes().submobjects.length).toBe(3);
    expect(td.get_z_axis()).toBeInstanceOf(NumberLine);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.bring_to_front
// Python: add(a, b); bring_to_front(a) -> [b, a], returns self.
//   With foreground: [Square, Triangle, Circle(fg)] (fg stays last).
// ---------------------------------------------------------------------------
describe("Scene.bring_to_front", () => {
  it("moves the mobject to the end and returns self", () => {
    const s = new Scene();
    const a = new Square();
    const b = new Circle();
    s.add(a as unknown as Mobject, b as unknown as Mobject);
    const ret = s.bring_to_front(a as unknown as Mobject);
    expect(ret).toBe(s);
    expect(s.mobjects[s.mobjects.length - 1]).toBe(a as unknown as Mobject);
    expect(s.mobjects[0]).toBe(b as unknown as Mobject);
  });

  it("keeps foreground members last like Python", () => {
    const s = new Scene();
    const m1 = new Square();
    const fg = new Circle();
    const m3 = new Triangle();
    s.add(m1 as unknown as Mobject);
    s.add_foreground_mobjects(fg as unknown as Mobject);
    s.bring_to_front(m3 as unknown as Mobject);
    expect(s.mobjects.map((m) => m.constructor.name)).toEqual([
      "Square",
      "Triangle",
      "Circle",
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.interactive_embed
// Python asserts OpenGL camera/renderer, then returns early unless valid;
//   TS mirrors the early-return and sets interactive_mode when valid.
// ---------------------------------------------------------------------------
describe("Scene.interactive_embed", () => {
  const keys = ["force_window", "write_to_movie", "format", "dry_run"] as const;
  const saved = new Map<string, unknown>();
  for (const k of keys) saved.set(k, config.get(k));

  afterEach(() => {
    for (const k of keys) config.set(k, saved.get(k));
  });

  it("stays out of interactive mode when invalid", () => {
    for (const k of keys) config.set(k, k === "format" ? "" : false);
    const s = new Scene();
    s.renderer = null;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      s.interactive_embed();
      expect(s.interactive_mode).toBe(false);
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("enters interactive mode when valid", () => {
    for (const k of keys) config.set(k, k === "format" ? "" : false);
    const s = new Scene();
    s.renderer = { window: {} };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      s.interactive_embed();
      expect(s.interactive_mode).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it("get_embedded_method queues a MethodWithArgs action", () => {
    for (const k of keys) config.set(k, k === "format" ? "" : false);
    const s = new Scene();
    const embedded_add = s.get_embedded_method("add");
    const mob = new Square();
    embedded_add(mob);
    expect(s.queue.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_x_axis /
//   .get_y_axis
// Python (Axes [-3,3,1]/[-2,2,1], 6x4): x NumberLine x_range [-3,3,1],
//   y NumberLine x_range [-2,2,1]; unit sizes 1.0/1.0.
// (D1): TS Axes carry no NumberLines yet; faithful NumberLine stand-ins are
// wired in, matching Python's axis geometry for these methods.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_x_axis / get_y_axis", () => {
  function wiredAxes(): Axes {
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const xAxis = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const yAxis = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    yAxis.rotate(Math.PI / 2);
    ax.axes = new VGroup(xAxis as unknown as VMobject, yAxis as unknown as VMobject);
    return ax;
  }

  it("returns the first/second axes", () => {
    const ax = wiredAxes();
    expect(ax.get_x_axis()).toBe(ax.axes.submobjects[0] as unknown as never);
    expect(ax.get_y_axis()).toBe(ax.axes.submobjects[1] as unknown as never);
  });

  it("exposes Python's axis ranges and unit sizes", () => {
    const ax = wiredAxes();
    expect([...(ax.get_x_axis() as unknown as NumberLine).x_range]).toEqual([-3, 3, 1]);
    expect([...(ax.get_y_axis() as unknown as NumberLine).x_range]).toEqual([-2, 2, 1]);
    expectClose(ax.get_x_unit_size(), 1.0);
    expectClose(ax.get_y_unit_size(), 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.get_unit_vector
// Python: [-3,3,1]/len 6 -> [1,0,0]; [0,4,1]/len 8 -> [2,0,0];
//   rotated pi/2 about zero -> [~0,1,0].
// ---------------------------------------------------------------------------
describe("NumberLine.get_unit_vector", () => {
  it("points along the line scaled by unit_size", () => {
    expectRowsClose(
      [new NumberLine({ x_range: [-3, 3, 1], length: 6 }).get_unit_vector()],
      [[1, 0, 0]],
    );
    expectRowsClose(
      [new NumberLine({ x_range: [0, 4, 1], length: 8 }).get_unit_vector()],
      [[2, 0, 0]],
    );
  });

  it("follows rotation", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    nl.rotate_about_zero(Math.PI / 2);
    expectRowsClose([nl.get_unit_vector()], [[2.220446049250313e-16, 1, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.NumberPlane
// Python: x_range (-7.111111111111111, 7.111111111111111, 1),
//   y_range (-4.0, 4.0, 1), x/y_length None;
//   c2p(1,1) [1,1,0]; p2c [1,1].
// (D1): TS builds no background lines yet (empty groups); Python has 22.
// ---------------------------------------------------------------------------
describe("NumberPlane", () => {
  it("stores Python's default ranges with null lengths", () => {
    const np = new NumberPlane();
    expectClose(np.x_range[0] ?? NaN, -7.111111111111111);
    expectClose(np.x_range[1] ?? NaN, 7.111111111111111);
    expect(np.x_range[2]).toBe(1);
    expect([...np.y_range]).toEqual([-4, 4, 1]);
    expect(np.x_length).toBeNull();
    expect(np.y_length).toBeNull();
  });

  it("maps coordinates about the range midpoint at unit size", () => {
    const np = new NumberPlane();
    expectRowsClose(plainRows(np.coords_to_point(1, 1)), [[1.0000000000000002, 1.0, 0.0]]);
    expect((np.point_to_coords([1, 1, 0]) as unknown as { toArray(): number[] }).toArray().map(Number)).toEqual([1, 1]);
  });

  it("builds the real background grid (python: 22 + 0)", () => {
    const np = new NumberPlane();
    expect(np.background_lines?.submobjects.length ?? 0).toBe(22);
    expect(np.faded_lines?.submobjects.length ?? 0).toBe(0);
  });

  it("maps linearly with explicit lengths", () => {
    // Python: NumberPlane([-3,3,1],[-2,2,1],6,4).coords_to_point(1,1)==[1,1,0].
    const np = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectRowsClose(plainRows(np.coords_to_point(1, 1)), [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols against earlier-ported machinery
// ---------------------------------------------------------------------------
describe("integration: layers-19 batch with earlier ports", () => {
  it("BackgroundRectangle wraps a become-morphed mobject", () => {
    // Python: Square(2).become(Circle(1)) keeps width 2.0; the background
    // rect around it is 2.0 x 2.0 with fill 0.75.
    const a = new Square(2);
    a.become(new Circle(1) as unknown as Mobject);
    const bg = new BackgroundRectangle(a as unknown as Mobject);
    expectClose(bg.width, 2.0);
    expectClose(bg.height, 2.0);
    expectRowsClose([centerOf(bg)], [[0, 0, 0]]);
  });

  it("SurroundingRectangle follows its target after become", () => {
    // Python: SurroundingRectangle(Square()) is 2.2; after the inner square
    // become()s a radius-1 circle (width 2.0) the rect is unchanged in size
    // class but re-targeting a fresh rect gives 2.2 again at origin.
    const sq = new Square();
    const first = new SurroundingRectangle(sq as unknown as Mobject, { buff: 0.1 });
    expectClose(first.width, 2.2);
    sq.become(new Circle(1) as unknown as Mobject);
    const second = new SurroundingRectangle(sq as unknown as Mobject, { buff: 0.1 });
    expectClose(second.width, 2.2);
    expectRowsClose([centerOf(second)], [[0, 0, 0]]);
  });

  it("Scene foreground + bring_to_front + animation intake compose", () => {
    // Python: add(a); add_foreground_mobjects(fg); bring_to_front(b) ->
    // [a, b, fg]; FadeOut(new) intake appends the new mobject.
    const s = new Scene();
    const a = new Square();
    const fg = new Circle();
    const b = new Triangle();
    s.add(a as unknown as Mobject);
    s.add_foreground_mobjects(fg as unknown as Mobject);
    s.bring_to_front(b as unknown as Mobject);
    expect(s.mobjects.map((m) => m.constructor.name)).toEqual([
      "Square",
      "Triangle",
      "Circle",
    ]);
    const fresh = new Square();
    s.add_mobjects_from_animations([new FadeOut(fresh as unknown as Mobject)]);
    expect(s.get_mobject_family_members().includes(fresh as unknown as Mobject)).toBe(true);
  });

  it("wired Axes agree with NumberLine ticks and unit vectors", () => {
    // Python: Axes([-3,3,1],[-2,2,1],6,4).c2p(1,0) == [1,0,0] ==
    // NumberLine([-3,3,1],6).n2p(1); tick center and unit vector match.
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    ax.axes = new VGroup(
      new NumberLine({ x_range: [-3, 3, 1], length: 6 }) as unknown as VMobject,
      new NumberLine({ x_range: [-2, 2, 1], length: 4 }) as unknown as VMobject,
    );
    const nl = ax.get_x_axis() as unknown as NumberLine;
    expectRowsClose(plainRows(ax.coords_to_point(1, 0)), plainRows(nl.n2p(1)));
    expectRowsClose([nl.get_tick(1.0).get_center()], [[1, 0, 0]]);
    expectRowsClose([nl.get_unit_vector()], [[1, 0, 0]]);
  });

  it("ThreeDScene ambient start/stop round-trips the tracker", () => {
    // Python Cairo: begin adds an updater + the tracker; stop clears both.
    const scene = new ThreeDScene();
    const theta = new ValueTracker(0);
    scene.camera = {
      theta_tracker: theta,
      phi_tracker: new ValueTracker(0),
      gamma_tracker: new ValueTracker(0),
    };
    scene.begin_ambient_camera_rotation(0.1, "theta");
    expect(theta.updaters.length).toBe(1);
    (theta.updaters[0] as (m: unknown, dt: number) => void)(theta, 1.0);
    expectClose(theta.get_value(), 0.1);
    scene.stop_ambient_camera_rotation("theta");
    expect(theta.updaters.length).toBe(0);
    expect(scene.mobjects.includes(theta as unknown as Mobject)).toBe(false);
  });

  it("NumberPlane midpoint mapping matches the Axes linear map", () => {
    // Python: NumberPlane([-3,3,1],[-2,2,1],6,4).c2p(1,1) == [1,1,0] ==
    // Axes([-3,3,1],[-2,2,1],6,4).c2p(1,1).
    const np = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectRowsClose(plainRows(np.coords_to_point(1, 1)), plainRows(ax.coords_to_point(1, 1)));
  });
});
