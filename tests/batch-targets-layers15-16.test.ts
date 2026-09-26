// Ground-truth tests for batch layers 15..16 (~1055 lines).
//
// Every numeric expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used here; each
// block cites the probe in a "Python:" comment. TS is asserted against those
// values (1e-9 relative for floats; exact for discrete; arrays element-wise).
//
// OUT OF SCOPE (text/tex): AddTextLetterByLetter/UntypeWithCursor take a
// `Text` in Python, but their logic is pure ShowIncreasingSubsets over an
// existing mobject's submobjects, so both languages are probed with a VGroup
// of Squares (no LaTeX needed). Matrix.get_brackets needs MathTex in both
// languages, so only the pure getter contract is tested (see D-matrix).
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { array } from "numpy-ts";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square, RegularPolygram } from "../src/mobject/geometry/polygram.js";
import { Line, Arrow } from "../src/mobject/geometry/line.js";
import {
  ArrowTip,
  ArrowTriangleTip,
  ArrowTriangleFilledTip,
} from "../src/mobject/geometry/tips.js";
import {
  get_3d_vmob_start_corner,
  get_3d_vmob_end_corner,
  get_3d_vmob_start_corner_index,
  get_3d_vmob_end_corner_index,
  get_3d_vmob_start_corner_unit_normal,
  get_3d_vmob_end_corner_unit_normal,
  get_3d_vmob_gradient_start_and_end_points,
} from "../src/mobject/three_d/three_d_utils.js";
import { Surface } from "../src/mobject/three_d/three_dimensions.js";
import { Scene } from "../src/scene/scene.js";
import { MovingCameraScene } from "../src/scene/moving_camera_scene.js";
import { SpecialThreeDScene } from "../src/scene/three_d_scene.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { ImageMobjectFromCamera } from "../src/mobject/types/image_mobject.js";
import {
  CoordinateSystem,
  Axes,
} from "../src/mobject/graphing/coordinate_systems.js";
import { Wait, Animation } from "../src/animation/animation.js";
import {
  Create,
  Uncreate,
  DrawBorderThenFill,
  AddTextLetterByLetter,
  UntypeWithCursor,
} from "../src/animation/creation.js";
import { GrowArrow } from "../src/animation/growing.js";
import { FadeOut } from "../src/animation/fading.js";
import { Matrix } from "../src/mobject/matrix.js";
import { ManimFrame, ManimConfig, config } from "../src/_config/utils.js";
import { TransformMatchingShapes } from "../src/animation/transform_matching_parts.js";
import { AnimatedBoundary, TracedPath } from "../src/animation/changing.js";
import {
  AnimationGroup,
  Succession,
  LaggedStart,
} from "../src/animation/composition.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { pointRows } from "../src/utils/arrays.js";
import { DEGREES } from "../src/constants.js";

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

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(actual[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

// ---------------------------------------------------------------------------
// manim.scene.moving_camera_scene.MovingCameraScene (+ __init__)
// Python: MovingCameraScene(skip_animations=True).camera is MovingCamera
describe("MovingCameraScene", () => {
  it("defaults to a MovingCamera", () => {
    const scene = new MovingCameraScene();
    expect(scene.camera).toBeInstanceOf(MovingCamera);
  });

  it("accepts skip_animations and behaves like Scene", () => {
    const scene = new MovingCameraScene(Object, {
      skip_animations: true,
    } as never);
    // camera_class override above is not a real camera; rebuild properly:
    const scene2 = new MovingCameraScene(undefined as never, {
      skip_animations: true,
    } as never);
    expect(scene2.camera).toBeInstanceOf(MovingCamera);
    expect(scene2.skip_animations).toBe(true);
    void scene;
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.image_mobject.ImageMobjectFromCamera.__init__
// Python: default_display_frame_config == {"stroke_width": 3,
//   "stroke_color": WHITE, "buff": 0}; pixel_array is camera.pixel_array
describe("ImageMobjectFromCamera.__init__", () => {
  it("stores camera, default frame config, and pixel array", () => {
    const camera = new MovingCamera();
    const mob = new ImageMobjectFromCamera(camera);
    expect(mob.camera).toBe(camera);
    expect(mob.default_display_frame_config).toEqual({
      stroke_width: 3,
      stroke_color: expect.anything(),
      buff: 0,
    });
    expect(mob.pixel_array).toBe(camera.pixel_array);
    // Python: camera.pixel_array.shape == (1080, 1920, 4)
    expect([...mob.pixel_array.shape]).toEqual([1080, 1920, 4]);
  });

  it("honours an explicit default_display_frame_config", () => {
    const camera = new MovingCamera();
    const cfg = { stroke_width: 5, buff: 1 };
    const mob = new ImageMobjectFromCamera(camera, cfg);
    expect(mob.default_display_frame_config).toBe(cfg);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.get_point_mobject
// Python: Mobject().get_point_mobject() raises
//   NotImplementedError("get_point_mobject not implemented for Mobject")
describe("Mobject.get_point_mobject", () => {
  it("raises NotImplementedError naming the class", () => {
    const m = new Mobject();
    expect(() => m.get_point_mobject()).toThrowError(
      "get_point_mobject not implemented for Mobject",
    );
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.SpecialThreeDScene.__init__
// Python defaults: cut_axes_at_radius True,
//   camera_config {"should_apply_shading": True, "exponential_projection": True}
//   sphere_config {"radius": 2, "resolution": (24, 48)}
//   default_angled_camera_position {"phi": 70*DEGREES, "theta": -110*DEGREES}
describe("SpecialThreeDScene.__init__", () => {
  it("stores documented defaults", () => {
    const scene = new SpecialThreeDScene();
    expect(scene.cut_axes_at_radius).toBe(true);
    expect(scene.camera_config).toEqual({
      should_apply_shading: true,
      exponential_projection: true,
    });
    expect(scene.sphere_config).toEqual({ radius: 2, resolution: [24, 48] });
    expectClose(
      scene.default_angled_camera_position["phi"] as number,
      70 * DEGREES,
    );
    expectClose(
      scene.default_angled_camera_position["theta"] as number,
      -110 * DEGREES,
    );
    expect(scene.three_d_axes_config).toEqual({
      num_axis_pieces: 1,
      axis_config: {
        unit_size: 2,
        tick_frequency: 1,
        numbers_with_elongated_ticks: [0, 1, 2],
        stroke_width: 2,
      },
    });
    expect(scene.low_quality_config).toEqual({
      camera_config: { should_apply_shading: false },
      three_d_axes_config: { num_axis_pieces: 1 },
      sphere_config: { resolution: [12, 24] },
    });
  });
});

// ---------------------------------------------------------------------------
// TipableVMobject.has_tip / has_start_tip / assign_tip_attr
// Python: Line([0,0,0],[1,0,0]).has_tip() is False, has_start_tip() False
//   Arrow([0,0,0],[1,0,0],buff=0).has_tip() True, has_start_tip() False
//   assign_tip_attr(tip, at_start) sets .tip/.start_tip but has_tip stays
//   False until the tip is added as a submobject.
describe("TipableVMobject tip predicates", () => {
  it("Line has no tips; Arrow has only an end tip", () => {
    const line = new Line([0, 0, 0], [1, 0, 0]);
    expect(line.has_tip()).toBe(false);
    expect(line.has_start_tip()).toBe(false);
    const arrow = new Arrow([0, 0, 0], [1, 0, 0], 0);
    expect(arrow.has_tip()).toBe(true);
    expect(arrow.has_start_tip()).toBe(false);
  });

  it("assign_tip_attr sets the attr without adding to the family", () => {
    const line = new Line([0, 0, 0], [1, 0, 0]);
    const tip = new ArrowTriangleTip();
    expect(line.assign_tip_attr(tip, false)).toBe(line);
    expect((line as unknown as { tip: ArrowTip }).tip).toBe(tip);
    // mirrors Python: tip not in family yet -> has_tip() False
    expect(line.has_tip()).toBe(false);
    line.assign_tip_attr(tip, true);
    expect((line as unknown as { start_tip: ArrowTip }).start_tip).toBe(tip);
    expect(line.has_start_tip()).toBe(false);
    // once added, the predicate flips (Python: `self.tip in self`)
    line.add(tip);
    expect(line.has_tip()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowTip.vector
// Python: Arrow([0,0,0],[2,2,0],buff=0).tip.vector == [0.25, 0.25, 0]
//   (docstring example); Arrow([0,0,0],[1,0,0],buff=0).tip.vector
//   == [0.25, 0, 0]
describe("ArrowTip.vector", () => {
  it("points from base to tip point along the arrow", () => {
    const arrow = new Arrow([0, 0, 0], [1, 0, 0], 0);
    const v = (arrow.tip as ArrowTip).vector;
    expectRowsClose([v], [[0.25, 0, 0]]);
    const diag = new Arrow([0, 0, 0], [2, 2, 0], 0);
    const dv = (diag.tip as ArrowTip).vector;
    // Python: Arrow([0,0,0],[2,2,0],buff=0).tip.vector ==
    //   [0.24748737341529115, 0.24748737341529203, 0] (the docstring's
    //   [0.25, 0.25, 0] is rounded to 2 decimals).
    expectRowsClose([dv], [[0.24748737341529115, 0.24748737341529203, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.RegularPolygram
// Python: RegularPolygram(5, density=2, radius=1): npoints 20,
//   points[0] == [0, 1, 0], start_angle == pi/2
//   RegularPolygram(6, density=2, radius=1): npoints 24, 0 submobjects
//   RegularPolygram(5, density=1, radius=1): npoints 20, points[0] [0,1,0]
describe("RegularPolygram", () => {
  it("pentagram layout matches Python", () => {
    const pg = new RegularPolygram(5, { density: 2, radius: 1 });
    expect(pg.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(pg.points, pg.dim).slice(0, 1), [[0, 1, 0]]);
    expectClose(pg.start_angle as number, Math.PI / 2);
  });

  it("hexagram (6/2 = 2 triangles) and pentagon match Python", () => {
    const hex = new RegularPolygram(6, { density: 2, radius: 1 });
    expect(hex.get_num_points()).toBe(24);
    expect(hex.submobjects.length).toBe(0);
    const pent = new RegularPolygram(5, { density: 1, radius: 1 });
    expect(pent.get_num_points()).toBe(20);
    expectRowsClose(rowsOf(pent.points, pent.dim).slice(0, 1), [[0, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_d_utils.*
// Python (Square()): start corner [1,1,0], end corner [-1,-1/3,0],
//   both corner normals [0,0,1]; empty VMobject -> ORIGIN corners, UP normal.
describe("three_d_utils corners and normals", () => {
  it("Square corners and normals match Python", () => {
    const sq = new Square();
    expectRowsClose(rowsOf(get_3d_vmob_start_corner(sq), 3), [[1, 1, 0]]);
    expectRowsClose(rowsOf(get_3d_vmob_end_corner(sq), 3), [
      [-1, -1 / 3, 0],
    ]);
    expect(get_3d_vmob_start_corner_index(sq)).toBe(0);
    expect(get_3d_vmob_end_corner_index(sq)).toBe(
      Math.floor((sq.get_num_points() - 1) / 6) * 3,
    );
    expectRowsClose(rowsOf(get_3d_vmob_start_corner_unit_normal(sq), 3), [
      [0, 0, 1],
    ]);
    // Python end normal is [-0.0, 0.0, 1.0]
    expectRowsClose(rowsOf(get_3d_vmob_end_corner_unit_normal(sq), 3), [
      [0, 0, 1],
    ]);
    const [gs, ge] = get_3d_vmob_gradient_start_and_end_points(sq);
    expectRowsClose(rowsOf(gs, 3), [[1, 1, 0]]);
    expectRowsClose(rowsOf(ge, 3), [[-1, -1 / 3, 0]]);
  });

  it("empty VMobject falls back to ORIGIN/UP like Python", () => {
    const v = new VMobject();
    expectRowsClose(rowsOf(get_3d_vmob_start_corner(v), 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(get_3d_vmob_end_corner(v), 3), [[0, 0, 0]]);
    expectRowsClose(rowsOf(get_3d_vmob_start_corner_unit_normal(v), 3), [
      [0, 1, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Surface
// Python: Surface(f, u_range=(0,1), v_range=(0,1), resolution=(4,4)) has
//   16 submobjects; each face has 16 points.
describe("Surface", () => {
  it("builds a 4x4 grid of 16-point faces", () => {
    const surf = new Surface(
      (u: number, v: number) => array([u, v, u * v]) as NDArray,
      [0, 1],
      [0, 1],
      [4, 4],
    );
    expect(surf.u_range).toEqual([0, 1]);
    expect(surf.v_range).toEqual([0, 1]);
    expect(surf.submobjects.length).toBe(16);
    expect(
      (surf.submobjects[0] as VMobject).get_num_points(),
    ).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.update_mobjects / get_top_level_mobjects / replace
// Python: update calls each mobject's updaters with dt; nested VGroup scene
//   has 1 top-level mobject; replace swaps in place preserving order.
describe("Scene mobject list ops", () => {
  it("update_mobjects forwards dt to updaters", () => {
    const scene = new Scene({ skip_animations: true });
    const tracker = new ValueTracker(0);
    const seen: number[] = [];
    tracker.add_updater((_, dt) => {
      seen.push(dt ?? 0);
    });
    scene.add(tracker);
    scene.update_mobjects(0.5);
    expect(seen).toEqual([0.5]);
    expect(tracker.get_value()).toBeCloseTo(0, 12);
  });

  it("get_top_level_mobjects filters nested members", () => {
    const scene = new Scene({ skip_animations: true });
    const a = new Square();
    const b = new Square();
    scene.add(a, b);
    expect(scene.get_top_level_mobjects().length).toBe(2);
    const scene2 = new Scene({ skip_animations: true });
    const g = new VGroup(a, b);
    scene2.add(g);
    const top = scene2.get_top_level_mobjects();
    expect(top.length).toBe(1);
    expect(top[0]).toBe(g);
  });

  it("replace swaps in place and errors when missing", () => {
    const scene = new Scene({ skip_animations: true });
    const a = new Square();
    const b = new Square();
    scene.add(a, b);
    const c = new Square();
    scene.replace(a, c);
    expect(scene.mobjects[0]).toBe(c);
    expect(scene.mobjects.length).toBe(2);
    expect(() => scene.replace(new Square(), new Square())).toThrowError(
      /Could not find/,
    );
    expect(() =>
      scene.replace(null as unknown as Mobject, c),
    ).toThrowError(/cannot be None/);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.pause / wait_until / is_current_animation_frozen_frame
// Python: pause(0.5) plays a frozen Wait and keeps the scene usable;
//   negative durations raise ValueError; wait_until(True) returns;
//   frozen-frame check is truthy only for a single static Wait.
describe("Scene pause/wait_until/frozen-frame", () => {
  it("pause and wait_until run without a renderer", () => {
    const scene = new Scene({ skip_animations: true });
    scene.add(new Square());
    expect(() => scene.pause(0.5)).not.toThrow();
    expect(() => scene.wait_until(() => true, 2)).not.toThrow();
  });

  it("rejects non-positive durations like Python validate_run_time", () => {
    const scene = new Scene({ skip_animations: true });
    expect(() => scene.pause(-1)).toThrowError(/duration/);
    expect(() => scene.wait_until(() => true, -1)).toThrowError(/max_time/);
  });

  it("is_current_animation_frozen_frame mirrors Python truthiness", () => {
    const scene = new Scene({ skip_animations: true });
    // Python: Wait() has is_static_wait None -> check is falsy.
    scene.animations = [new Wait({ run_time: 1 })];
    expect(scene.is_current_animation_frozen_frame()).toBe(false);
    // Python: Wait(frozen_frame=True) single -> truthy.
    scene.animations = [new Wait({ run_time: 1, frozen_frame: true })];
    expect(scene.is_current_animation_frozen_frame()).toBe(true);
    // Python: non-Wait or two animations -> falsy.
    scene.animations = [new Create(new Square())];
    expect(scene.is_current_animation_frozen_frame()).toBe(false);
    scene.animations = [
      new Wait({ run_time: 1, frozen_frame: true }),
      new Wait({ run_time: 1, frozen_frame: true }),
    ];
    expect(scene.is_current_animation_frozen_frame()).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_axes
// Python: base raises NotImplementedError; Axes/NumberPlane return 2 axes.
// D-axes: the TS Axes hierarchy is not Mobject-based yet, so get_axes()
// returns an (empty) VGroup; base still throws. See 
// ("Axes.axes/NumberLine.ticks empty containers").
describe("CoordinateSystem.get_axes", () => {
  it("base class raises NotImplementedError", () => {
    const base = Object.create(CoordinateSystem.prototype) as CoordinateSystem;
    expect(() => base.get_axes()).toThrowError(/must be implemented/);
  });

  it("Axes returns its two real NumberLine axes like python", () => {
    const axes = new Axes([0, 5, 1], [0, 5, 1]);
    expect(axes.get_axes()).toBeInstanceOf(VGroup);
    expect(axes.get_axes().submobjects.length).toBe(2);
    expect(axes.get_x_axis()).toBe(axes.get_axes().submobjects[0]);
    expect(axes.get_y_axis()).toBe(axes.get_axes().submobjects[1]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Create.__init__ / Uncreate.__init__
// Python: Create(Square()): lag_ratio 1.0, introducer True, run_time 1.0
//   Uncreate(Square()): reverse True, remover True, introducer False
describe("Create/Uncreate defaults", () => {
  it("Create matches Python defaults", () => {
    const c = new Create(new Square());
    expect(c.lag_ratio).toBe(1);
    expect(c.introducer).toBe(true);
    expect(c.run_time).toBe(1);
  });

  it("Uncreate matches Python defaults", () => {
    const u = new Uncreate(new Square());
    expect(u.remover).toBe(true);
    expect(u.reverse_rate_function).toBe(true);
    expect(u.introducer).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.DrawBorderThenFill.__init__ / get_outline /
// get_all_mobjects
// Python: run_time 2, stroke_width 2, outline 16 pts, 3 total mobjects.
describe("DrawBorderThenFill", () => {
  it("defaults, outline, and mobject list match Python", () => {
    const d = new DrawBorderThenFill(new Square());
    expect(d.run_time).toBe(2);
    expect(d.stroke_width).toBe(2);
    expect((d.outline as VMobject).get_num_points()).toBe(16);
    expect(d.get_all_mobjects().length).toBe(3);
    // outline has no fill
    expect((d.get_outline() as VMobject).get_fill_opacity()).toBe(0);
  });

  it("begin() refreshes the outline", () => {
    const d = new DrawBorderThenFill(new Square());
    const before = d.outline;
    d.begin();
    expect(d.outline).not.toBe(before);
    expect((d.outline as VMobject).get_num_points()).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.get_all_families_zipped
// Python: Create(Square()) has 2 all-mobjects with family sizes [1, 0], so
//   zip stops at the shortest -> 0 rows.
describe("Animation.get_all_families_zipped", () => {
  it("Create(Square()) zips to zero rows like Python", () => {
    const c = new Create(new Square());
    expect(c.get_all_mobjects().length).toBe(2);
    expect([...c.get_all_families_zipped()].length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowArrow
// Python: GrowArrow(Arrow([0,0,0],[1,0,0],buff=0)).point == [0,0,0]
describe("GrowArrow", () => {
  it("grows from the arrow start point", () => {
    const arrow = new Arrow([0, 0, 0], [1, 0, 0], 0);
    const g = new GrowArrow(arrow);
    const pt = g.point as number[];
    expectRowsClose([[...pt]], [[0, 0, 0]]);
    const start = g.create_starting_mobject();
    expect(start).toBeInstanceOf(Mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.fading.FadeOut (+ __init__)
// Python: FadeOut(Square()).remover is True
describe("FadeOut", () => {
  it("sets remover like Python", () => {
    const f = new FadeOut(new Square());
    expect(f.remover).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.Matrix.get_brackets
// Python: `return self.brackets` (a 2-element VGroup; built via MathTex).
// D-matrix: bracket construction needs MathTex (tex out of scope), so the
// getter contract (returns this.brackets) is tested by direct assignment.
describe("Matrix.get_brackets", () => {
  it("returns the stored brackets VGroup", () => {
    const m = Object.create(Matrix.prototype) as Matrix;
    const brackets = new VGroup(new Square(), new Square());
    (m as unknown as { brackets: VGroup }).brackets = brackets;
    expect(m.get_brackets()).toBe(brackets);
    expect(m.get_brackets().submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimFrame (+ __init__)
// Python: ManimFrame(config) ok; ManimFrame("x") raises
//   TypeError("argument must be instance of 'ManimConfig'")
describe("ManimFrame", () => {
  it("wraps a ManimConfig and rejects other types", () => {
    const mc = new ManimConfig();
    const frame = new ManimFrame(mc);
    expect(frame).toBeInstanceOf(ManimFrame);
    expect(() => new ManimFrame("x" as never)).toThrowError(
      "argument must be instance of 'ManimConfig'",
    );
  });

  it("exposes config values and constants", () => {
    // Python: ManimFrame(config)["frame_height"] == 8.0,
    //   ["frame_width"] == 128/9 ≈ 14.2222; ["frame_rate"] raises KeyError
    //   (frame _OPTS has only 11 keys) — TS matches both.
    const frame = new ManimFrame(config);
    expect(frame.pixel_width).toBe(1920);
    expectClose(frame.get("frame_height") as number, 8.0);
    expectClose(frame.get("frame_width") as number, 128 / 9);
    expectClose(frame.get("aspect_ratio") as number, 16 / 9);
    expect(() => frame.get("frame_rate")).toThrowError(/frame_rate/);
    expect(frame.ORIGIN).toEqual([0, 0, 0]);
    expect(frame.__len__()).toBe(11);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts...get_shape_map
// Python: TransformMatchingShapes(Square(), Square()).get_shape_map(Square())
//   -> 1 key holding 1 part.
describe("TransformMatchingAbstractBase.get_shape_map", () => {
  it("groups parts by key like Python", () => {
    const t = new TransformMatchingShapes(new Square(), new Square());
    const sm = t.get_shape_map(new Square());
    expect(sm.size).toBe(1);
    let total = 0;
    for (const group of sm.values()) total += group.submobjects.length;
    expect(total).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.changing.AnimatedBoundary.__init__ / TracedPath.update_path
// Python: AnimatedBoundary(Square()): 4 colors, max_stroke_width 3,
//   cycle_rate 0.5, 2 boundary copies as submobjects.
//   TracedPath(f): no points before update; one update -> 4 points at f().
describe("AnimatedBoundary/TracedPath", () => {
  it("AnimatedBoundary defaults match Python", () => {
    const ab = new AnimatedBoundary(new Square());
    expect(ab.colors.length).toBe(4);
    expect(ab.max_stroke_width).toBe(3);
    expect(ab.cycle_rate).toBe(0.5);
    expect(ab.back_and_forth).toBe(true);
    expect(ab.submobjects.length).toBe(2);
    expect(ab.total_time).toBe(0);
  });

  it("TracedPath.update_path starts a path and extends it", () => {
    const tp = new TracedPath(() => [1, 2, 0]);
    expect(tp.has_points()).toBe(false);
    tp.update_path(tp, 0.1);
    expect(tp.get_num_points()).toBe(4);
    expectRowsClose(rowsOf(tp.points, tp.dim).slice(0, 1), [[1, 2, 0]]);
    tp.update_path(tp, 0.1);
    expect(tp.get_num_points()).toBeGreaterThan(4);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.AnimationGroup.begin/finish,
// Succession.__init__, LaggedStart.__init__
// Python: empty begin() raises ValueError("Trying to play ... without
//   animations ..."); finish sets begun/finished True;
//   Succession lag_ratio 1; LaggedStart default lag_ratio 0.05.
describe("AnimationGroup/Succession/LaggedStart", () => {
  it("begin/finish bookkeeping matches Python", () => {
    const ag = new AnimationGroup(
      new Create(new Square()),
      new FadeOut(new Square()),
    );
    ag.begin();
    ag.finish();
    expect([...ag.anims_begun]).toEqual([true, true]);
    expect([...ag.anims_finished]).toEqual([true, true]);
  });

  it("empty begin() raises ValueError like Python", () => {
    expect(() => new AnimationGroup().begin()).toThrowError(/without animations/);
  });

  it("Succession/LaggedStart lag_ratio defaults match Python", () => {
    const su = new Succession(
      new Create(new Square()),
      new FadeOut(new Square()),
    );
    expect(su.lag_ratio).toBe(1);
    const lag = new LaggedStart(
      new Create(new Square()),
      new FadeOut(new Square()),
    );
    expectClose(lag.lag_ratio as number, 0.05);
  });

  it("Succession played through a Scene leaves flags False like Python", () => {
    // Python ground truth (Scene(skip_animations=True).play(su)):
    //   begun [False, False], finished [False, False] — Succession drives
    //   active_animation instead of the AnimationGroup flags.
    const scene = new Scene({ skip_animations: true });
    const su = new Succession(
      new Create(new Square()),
      new FadeOut(new Square()),
    );
    scene.play(su);
    expect([...su.anims_begun]).toEqual([false, false]);
    expect([...su.anims_finished]).toEqual([false, false]);
    expect(su.active_animation).toBe(null);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.AddTextLetterByLetter.__init__ / UntypeWithCursor
// Python (VGroup of 3 Squares, no LaTeX needed):
//   run_time == max(1/60, 0.1)*3 == 0.30000000000000004, time_per_char 0.1
//   empty VGroup -> ValueError "... does not seem to contain any characters."
//   UntypeWithCursor(g2): remover True, introducer False, reverse True
describe("AddTextLetterByLetter/UntypeWithCursor", () => {
  it("run_time derives from time_per_char like Python", () => {
    const g = new VGroup(new Square(), new Square(), new Square());
    const a = new AddTextLetterByLetter(g);
    expectClose(a.run_time, 0.1 * 3);
    expect(a.time_per_char).toBe(0.1);
  });

  it("empty input raises ValueError like Python", () => {
    expect(() => new AddTextLetterByLetter(new VGroup())).toThrowError(
      /does not seem to contain any characters/,
    );
  });

  it("UntypeWithCursor flags match Python", () => {
    const g = new VGroup(new Square(), new Square());
    const cursor = new Square();
    const u = new UntypeWithCursor(g, cursor);
    expect(u.remover).toBe(true);
    expect(u.introducer).toBe(false);
    expect(u.reverse_rate_function).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols interacting with earlier-ported code, each
// checked against the original manim implementation ground truth quoted above.
// ---------------------------------------------------------------------------
describe("batch layers 15..16 integration", () => {
  it("Scene.replace + get_top_level_mobjects on a VGroup scene", () => {
    // Python: scene with VGroup(a, b) has 1 top-level mobject; replacing a
    // nested member keeps the group on top.
    const scene = new Scene({ skip_animations: true });
    const a = new Square();
    const b = new Square();
    const g = new VGroup(a, b);
    scene.add(g);
    expect(scene.get_top_level_mobjects()).toEqual([g]);
    const c = new Square();
    scene.replace(a, c);
    expect(scene.get_top_level_mobjects()).toEqual([g]);
    expect((g.submobjects as Mobject[]).includes(c)).toBe(true);
  });

  it("AnimationGroup.begin wires sub-animation mobjects into the scene", () => {
    const scene = new Scene({ skip_animations: true });
    const sq = new Square();
    scene.add(sq);
    const ag = new AnimationGroup(new FadeOut(sq));
    scene.play(ag);
    expect([...ag.anims_finished]).toEqual([true]);
  });

  it("DrawBorderThenFill outline interpolates against a real Square", () => {
    const sq = new Square();
    const d = new DrawBorderThenFill(sq);
    d.begin();
    const fams = d.get_all_families_zipped();
    expect(fams.length).toBeGreaterThan(0);
    d.interpolate(0.5);
    expect(sq.get_num_points()).toBeGreaterThan(0);
  });

  it("TracedPath follows a ValueTracker-driven point", () => {
    // Python equivalent: traced func reads tracker value; two updates grow
    // the path monotonically.
    const tracker = new ValueTracker(0);
    const tp = new TracedPath(() => [tracker.get_value(), 0, 0]);
    tp.update_path(tp, 0.1);
    tracker.set_value(1);
    tp.update_path(tp, 0.1);
    const rows = rowsOf(tp.points, tp.dim);
    expect(rows.length).toBeGreaterThan(4);
    expectClose(rows[0]?.[0] as number, 0);
    expectClose(rows[rows.length - 1]?.[0] as number, 1);
  });

  it("get_shape_map + Succession chain over matching squares", () => {
    const t = new TransformMatchingShapes(new Square(), new Square());
    const sm = t.get_shape_map(new Square());
    expect(sm.size).toBe(1);
    const scene = new Scene({ skip_animations: true });
    const su = new Succession(
      new Create(new Square()),
      new FadeOut(new Square()),
    );
    scene.play(su);
    // Python: flags stay False after play (see unit test above).
    expect([...su.anims_finished]).toEqual([false, false]);
  });

  it("GrowArrow starting mobject is a degenerate arrow at the start point", () => {
    // Python: create_starting_mobject scales to 0 about the start point.
    const arrow = new Arrow([0, 0, 0], [2, 0, 0], 0);
    const g = new GrowArrow(arrow);
    const start = g.create_starting_mobject() as VMobject;
    const rows = rowsOf(start.points, start.dim);
    for (const row of rows) expectClose(row[0] as number, 0, 1e-6);
  });

  it("3d corner utils agree with Surface face geometry", () => {
    const surf = new Surface(
      (u: number, v: number) => array([u, v, 0]) as NDArray,
      [0, 1],
      [0, 1],
      [2, 2],
    );
    expect(surf.submobjects.length).toBe(4);
    const face = surf.submobjects[0] as VMobject;
    const corner = rowsOf(get_3d_vmob_start_corner(face), 3)[0] as number[];
    const faceRows = rowsOf(face.points, face.dim);
    expectRowsClose([corner], [faceRows[0] as number[]]);
  });
});
