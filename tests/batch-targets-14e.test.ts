// Ground-truth tests for batch layers 14..14 (~1004 lines).
//
// Every expectation below was captured from the original Python manim
// (original manim) before being asserted against the
// TS port. Float comparisons use tight tolerance (1e-9 where deterministic).
//
// NOTE (text/tex out of scope): SVGMobject.text_to_mobject only warns and
// returns null in both implementations; _ScaleBase/LogBase.get_custom_labels
// are stubs (Python's LogBase version needs a latex install).
import { describe, expect, it, vi } from "vitest";
import { NDArray, array } from "numpy-ts";

import { ManimConfig } from "../src/_config/utils.js";
import { RendererType, DEGREES } from "../src/constants.js";
import { VMobject, VGroup, VectorizedPoint } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { Point } from "../src/mobject/types/point_cloud_mobject.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Arc, Circle } from "../src/mobject/geometry/arc.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Arrow } from "../src/mobject/geometry/line.js";
import {
  ArrowTip,
  ArrowTriangleFilledTip,
} from "../src/mobject/geometry/tips.js";
import {
  get_3d_vmob_end_corner_index,
  get_3d_vmob_start_corner_index,
  get_3d_vmob_start_corner,
  get_3d_vmob_end_corner,
  get_3d_vmob_unit_normal,
} from "../src/mobject/three_d/three_d_utils.js";
import { ThreeDVMobject } from "../src/mobject/three_d/three_dimensions.js";
import { _ScaleBase, LogBase } from "../src/mobject/graphing/scale.js";
import { Scene } from "../src/scene/scene.js";
import { ThreeDScene, SpecialThreeDScene } from "../src/scene/three_d_scene.js";
import { VectorScene } from "../src/scene/vector_space_scene.js";
import { Camera } from "../src/camera/camera.js";
import { ApplyMethod } from "../src/animation/transform.js";
import { ShowPartial, Create } from "../src/animation/creation.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { Wait } from "../src/animation/animation.js";
import { pointRows } from "../src/utils/arrays.js";

function rows(value: NDArray, dim = 3): number[][] {
  return pointRows(value, dim);
}

function expectRowsClose(actual: number[][], expected: number[][], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      const a = actual[i]?.[j] ?? NaN;
      const e = expected[i]?.[j] ?? NaN;
      expect(Math.abs(a - e)).toBeLessThanOrEqual(tol + 1e-9 * Math.abs(e));
    }
  }
}

function bindMobjectMethod(m: Mobject, name: string) {
  const func = (m as unknown as Record<string, (...args: unknown[]) => unknown>)[name];
  if (typeof func !== "function") throw new Error(`no method ${name}`);
  return Object.assign(func.bind(m), { __self__: m, __func__: func });
}

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.set_points_as_corners
// Python: set_anchors_and_handles(interpolate(points[:-1], points[1:], t)
//   for t in _bezier_t_values); corners [[0,0,0],[1,0,0],[1,1,0]] give
//   8 points (2 curves), first row [0,0,0], rows 1..2 at thirds.
// ---------------------------------------------------------------------------
describe("VMobject.set_points_as_corners", () => {
  it("lays straight cubic segments between corners", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    // Python ground truth (np.round(...,6)):
    // [[0,0,0],[0.333333,0,0],[0.666667,0,0],[1,0,0],
    //  [1,0,0],[1,0.333333,0],[1,0.666667,0],[1,1,0]]
    expect(v.get_num_curves()).toBe(2);
    expectRowsClose(rows(v.points), [
      [0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0], [1, 0, 0],
      [1, 0, 0], [1, 1 / 3, 0], [1, 2 / 3, 0], [1, 1, 0],
    ], 1e-9);
  });

  it("raises ValueError on empty input like Python", () => {
    // Python `set_points_as_corners([])` raises
    // "ValueError: could not broadcast input array from shape (0,) into
    // shape (0,3)" (numpy broadcast failure).
    const v = new VMobject();
    expect(() => v.set_points_as_corners([])).toThrowError(
      /could not broadcast/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.insert_n_curves
// Python: 2 curves + insert_n_curves(1) -> 3 curves / 12 points:
// [[0,0,0],[0.166667,0,0],[0.333333,0,0],[0.5,0,0],[0.5,0,0],
//  [0.666667,0,0],[0.833333,0,0],[1,0,0],[1,0,0],[1,0.333333,0],
//  [1,0.666667,0],[1,1,0]]
// ---------------------------------------------------------------------------
describe("VMobject.insert_n_curves", () => {
  it("inserts one curve by splitting proportionally", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.insert_n_curves(1);
    expect(v.get_num_curves()).toBe(3);
    expect(v.get_num_points()).toBe(12);
    expectRowsClose(rows(v.points), [
      [0, 0, 0], [1 / 6, 0, 0], [2 / 6, 0, 0], [0.5, 0, 0],
      [0.5, 0, 0], [2 / 3, 0, 0], [5 / 6, 0, 0], [1, 0, 0],
      [1, 0, 0], [1, 1 / 3, 0], [1, 2 / 3, 0], [1, 1, 0],
    ], 1e-6);
  });

  it("preserves start/end anchors", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    v.insert_n_curves(2);
    expect(v.get_num_curves()).toBe(4);
    expectRowsClose([v.get_start()], [[0, 0, 0]]);
    expectRowsClose([v.get_end()], [[1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Arc.get_arc_center
// Python: Arc(radius=1, start_angle=0, angle=pi/2).get_arc_center() == [0,0,0]
// ---------------------------------------------------------------------------
describe("Arc.get_arc_center", () => {
  it("recovers the origin for a quarter arc", () => {
    const a = new Arc(1.0, 0, Math.PI / 2);
    expectRowsClose([a.get_arc_center()], [[0, 0, 0]], 1e-9);
  });

  it("returns the anchor itself when a1 == a2 (zero-radius case)", () => {
    const a = new Arc(1.0, 0, Math.PI / 2);
    const anchor = [3, 4, 0];
    a.set_anchors_and_handles([anchor], [[...anchor]], [[...anchor]], [anchor]);
    expectRowsClose([a.get_arc_center(false)], [anchor], 1e-12);
  });
});

// ---------------------------------------------------------------------------
// TipableVMobject.get_first_handle / get_last_handle
// Python (quarter arc): first [1, 0.065502, 0], last [0.065502, 1, 0]
// where 0.065502... == 4/3 * tan(pi/64).
// ---------------------------------------------------------------------------
describe("TipableVMobject.get_first_handle/get_last_handle", () => {
  it("returns the second and second-to-last points", () => {
    const a = new Arc(1.0, 0, Math.PI / 2);
    const t = (4 / 3) * Math.tan(Math.PI / 64);
    expectRowsClose([a.get_first_handle()], [[1, t, 0]], 1e-9);
    expectRowsClose([a.get_last_handle()], [[t, 1, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.handle_transform
// Python: mat=[[a,c],[b,d]], vec=[e,f,0]; apply_matrix then shift.
// translate(5,6) on corners [[0,0,0],[1,0,0]] -> center [5.5,6,0];
// scale(2,3) on corners [[1,1,0],[2,2,0]] -> center [3,4.5,0].
// ---------------------------------------------------------------------------
describe("SVGMobject.handle_transform", () => {
  it("applies a translation matrix", () => {
    const mob = new VMobject();
    mob.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    SVGMobject.handle_transform(mob, { a: 1, b: 0, c: 0, d: 1, e: 5, f: 6 });
    expectRowsClose([mob.get_center()], [[5.5, 6, 0]], 1e-9);
  });

  it("applies a scale matrix", () => {
    const mob = new VMobject();
    mob.set_points_as_corners([[1, 1, 0], [2, 2, 0]]);
    SVGMobject.handle_transform(mob, { a: 2, b: 0, c: 0, d: 3, e: 0, f: 0 });
    expectRowsClose([mob.get_start()], [[2, 3, 0]], 1e-9);
    expectRowsClose([mob.get_center()], [[3, 4.5, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.apply_style_to_mobject
// Python: mob.set_style(stroke_width, stroke.hexrgb, stroke.opacity,
//   fill.hexrgb, fill.opacity). (Python raises AttributeError when the shape
//   has no stroke/fill; the TS port treats null as "keep", asserted below.)
// ---------------------------------------------------------------------------
describe("SVGMobject.apply_style_to_mobject", () => {
  it("copies stroke/fill style onto the mobject", () => {
    const mob = new VMobject();
    mob.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    SVGMobject.apply_style_to_mobject(mob, {
      kind: "path",
      stroke_width: 7,
      stroke_color: "#ff0000",
      stroke_opacity: 0.5,
      fill_color: "#00ff00",
      fill_opacity: 0.25,
    });
    expect(mob.get_stroke_width()).toBe(7);
    expect(mob.get_stroke_opacity()).toBeCloseTo(0.5, 12);
    expect(mob.get_fill_opacity()).toBeCloseTo(0.25, 12);
    expect(mob.get_stroke_color()?.to_hex()).toBe("#FF0000");
    expect(mob.get_fill_color()?.to_hex()).toBe("#00FF00");
  });

  it("keeps existing style when the shape carries nulls", () => {
    const mob = new VMobject();
    mob.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const before = mob.get_stroke_width();
    SVGMobject.apply_style_to_mobject(mob, { kind: "path" });
    expect(mob.get_stroke_width()).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.text_to_mobject
// Python: warns "Unsupported element type" and returns None.
// ---------------------------------------------------------------------------
describe("SVGMobject.text_to_mobject", () => {
  it("warns and returns null", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      expect(SVGMobject.text_to_mobject({ kind: "text" })).toBeNull();
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.Point
// Python: Point([7,8,9]).points == [[7,8,9]]
// ---------------------------------------------------------------------------
describe("Point", () => {
  it("stores a single point at the location", () => {
    const p = new Point([7, 8, 9]);
    expect(p.get_num_points()).toBe(1);
    expectRowsClose(rows(p.points), [[7, 8, 9]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VectorizedPoint
// Python: points [[1,2,3]], get_location [1,2,3], width/height 0.01,
// set_location([4,5,6]) updates.
// ---------------------------------------------------------------------------
describe("VectorizedPoint", () => {
  it("tracks location with artificial width/height", () => {
    const vp = new VectorizedPoint([1, 2, 3]);
    expectRowsClose(rows(vp.points), [[1, 2, 3]]);
    expectRowsClose([[...(vp.get_location() as unknown as number[])]], [[1, 2, 3]]);
    expect(vp.width).toBe(0.01);
    expect(vp.height).toBe(0.01);
    vp.set_location([4, 5, 6]);
    expectRowsClose([[...(vp.get_location() as unknown as number[])]], [[4, 5, 6]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.get_pieces
// Python: Square().get_pieces(3) -> Group of 3 submobjects.
// ---------------------------------------------------------------------------
describe("Mobject.get_pieces", () => {
  it("splits into n contiguous pieces", () => {
    const sq = new Square();
    const pieces = sq.get_pieces(3);
    expect(pieces.submobjects.length).toBe(3);
    const [p0, p1, p2] = pieces.submobjects as VMobject[];
    // Contiguous: end of piece k == start of piece k+1; overall start/end
    // match the square (Python: start [1,1,0]).
    expectRowsClose([p0!.get_start()], [[1, 1, 0]], 1e-9);
    expectRowsClose([p0!.get_end()], [p1!.get_start()], 1e-9);
    expectRowsClose([p1!.get_end()], [p2!.get_start()], 1e-9);
    expectRowsClose([p2!.get_end()], [sq.get_end()], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_d_utils.*
// Python (Square, 16 points):
//   start index 0, end index 6, start corner [1,1,0],
//   end corner [-1,-0.3333,0], unit normal(0) [0,0,1];
// tiny 2-anchor VMobject -> UP [0,1,0].
// ---------------------------------------------------------------------------
describe("three_d_utils corner helpers", () => {
  it("computes start/end corner indices", () => {
    const sq = new Square();
    expect(sq.get_num_points()).toBe(16);
    expect(get_3d_vmob_start_corner_index(sq)).toBe(0);
    expect(get_3d_vmob_end_corner_index(sq)).toBe(6);
  });

  it("computes start/end corners", () => {
    const sq = new Square();
    expectRowsClose([[...(get_3d_vmob_start_corner(sq) as unknown as number[])]], [[1, 1, 0]], 1e-9);
    expectRowsClose([[...(get_3d_vmob_end_corner(sq) as unknown as number[])]], [[-1, -1 / 3, 0]], 1e-6);
  });

  it("computes unit normals, falling back to UP for <= 2 anchors", () => {
    const sq = new Square();
    expectRowsClose([[...(get_3d_vmob_unit_normal(sq, 0) as unknown as number[])]], [[0, 0, 1]], 1e-6);
    const tiny = new VMobject();
    tiny.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(tiny.get_anchors().length).toBeLessThanOrEqual(2);
    expectRowsClose([[...(get_3d_vmob_unit_normal(tiny, 0) as unknown as number[])]], [[0, 1, 0]], 1e-12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.ThreeDVMobject
// Python: __init__(shade_in_3d=True) -> VMobject subclass.
// ---------------------------------------------------------------------------
describe("ThreeDVMobject", () => {
  it("defaults shade_in_3d to true and accepts false", () => {
    const a = new ThreeDVMobject();
    expect(a).toBeInstanceOf(VMobject);
    expect(a.shade_in_3d).toBe(true);
    const b = new ThreeDVMobject(false);
    expect(b.shade_in_3d).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowTip / tip_point
// Python: ArrowTip() raises NotImplementedError; ArrowTriangleFilledTip has
// tip_point [-0.425, 0, 0], base [-0.075, 0, 0], length 0.35, angle pi;
// Arrow([0,0,0],[2,0,0],buff=0).tip.tip_point == [2,0,0].
// ---------------------------------------------------------------------------
describe("ArrowTip", () => {
  it("cannot be constructed directly", () => {
    expect(() => new ArrowTip({})).toThrowError(/implemented in inheriting/);
  });

  it("exposes tip_point/base/vector/angle/length on filled triangle tips", () => {
    const tip = new ArrowTriangleFilledTip();
    expectRowsClose([[...(tip.tip_point as unknown as number[])]], [[-0.425, 0, 0]], 1e-9);
    expectRowsClose([[...(tip.base as unknown as number[])]], [[-0.075, 0, 0]], 1e-9);
    expect(tip.length).toBeCloseTo(0.35, 12);
    // Python reports +pi here; the TS Triangle build leaves a ~1e-16
    // negative y in tip-base, so atan2 yields -pi. Same direction mod 2pi
    // (position_tip only uses it modulo a full turn), so assert mod 2pi.
    const TAU = 2 * Math.PI;
    expect(((tip.tip_angle % TAU) + TAU) % TAU).toBeCloseTo(Math.PI, 12);
  });

  it("lands the tip point at the arrow end", () => {
    const arr = new Arrow([0, 0, 0], [2, 0, 0], { buff: 0 });
    expectRowsClose([[...(arr.tip!.tip_point as unknown as number[])]], [[2, 0, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.scale._ScaleBase.get_custom_labels
// Python: _ScaleBase raises NotImplementedError; LogBase needs latex
// (FileNotFoundError in this env) and is a TODO stub in TS.
// ---------------------------------------------------------------------------
describe("_ScaleBase.get_custom_labels", () => {
  it("raises NotImplementedError on the base", () => {
    expect(() => new _ScaleBase().get_custom_labels([1, 2, 3])).toThrowError(
      expect.objectContaining({ name: "NotImplementedError" }),
    );
  });

  it("LogBase stub throws until text/tex lands", () => {
    // Deviation: Python builds MathTex labels (needs latex); the TS port
    // throws TODO since text/tex is out of scope for the entire port.
    expect(() => new LogBase().get_custom_labels([1, 2])).toThrowError(/not ported/);
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig.renderer
// Python: fresh config renderer None; 'opengl'/'OpenGL' -> OPENGL,
// 'cAirO' -> CAIRO; 42 -> ValueError "42 is not a valid RendererType".
// ---------------------------------------------------------------------------
describe("ManimConfig.renderer", () => {
  it("round-trips renderer names case-insensitively and rejects junk", () => {
    const cfg = new ManimConfig(false);
    cfg.renderer = "opengl";
    expect(cfg.renderer).toBe(RendererType.OPENGL);
    cfg.renderer = "OpenGL";
    expect(cfg.renderer).toBe(RendererType.OPENGL);
    cfg.renderer = "cAirO";
    expect(cfg.renderer).toBe(RendererType.CAIRO);
    expect(() => {
      cfg.renderer = 42 as unknown as string;
    }).toThrowError(/42 is not a valid RendererType/);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyMethod (__init__ + class)
// Python: stores method/method_args, inits Transform on method.__self__;
// create_target() copies __self__ and applies the method; invoking the
// method instead of passing it raises ValueError.
// ---------------------------------------------------------------------------
describe("ApplyMethod", () => {
  it("binds the transform to the method's mobject", () => {
    const sq = new Square();
    const bound = bindMobjectMethod(sq, "shift");
    const anim = new ApplyMethod(bound, [1, 0, 0]);
    expect(anim.mobject).toBe(sq);
    expect(anim.method_args).toEqual([[1, 0, 0]]);
  });

  it("create_target applies the method to a copy", () => {
    const sq = new Square();
    const anim = new ApplyMethod(bindMobjectMethod(sq, "shift"), [1, 0, 0]);
    const target = anim.create_target();
    expectRowsClose([target.get_center()], [[1, 0, 0]], 1e-9);
    expectRowsClose([sq.get_center()], [[0, 0, 0]], 1e-9);
  });

  it("rejects an already-invoked method", () => {
    const sq = new Square();
    expect(() => new ApplyMethod(sq.shift([1, 0, 0]) as never)).toThrowError(
      /accidentally invoked/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.ShowPartial (__init__) / Create
// Python: non-VMobject -> TypeError "... only works for VMobjects.";
// Create(mobject, lag_ratio=1, introducer=True); _get_bounds(alpha)=(0,alpha).
// ---------------------------------------------------------------------------
describe("ShowPartial/Create", () => {
  it("rejects non-VMobjects", () => {
    expect(() => new ShowPartial(new Mobject() as unknown as VMobject)).toThrowError(
      /only works for VMobjects/,
    );
  });

  it("base _get_bounds is abstract", () => {
    expect(() => new ShowPartial(new Square())._get_bounds(0.5)).toThrowError(
      expect.objectContaining({ name: "NotImplementedError" }),
    );
  });

  it("Create reveals [0, alpha] with lag_ratio 1", () => {
    const sq = new Square();
    const create = new Create(sq);
    expect(create._get_bounds(0.37)).toEqual([0, 0.37]);
    expect(create.lag_ratio).toBe(1);
    expect(create.introducer).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_restructured_mobject_list
// Python: VGroup(Square, Circle) + Dot minus Square -> [Circle, Dot].
// ---------------------------------------------------------------------------
describe("Scene.get_restructured_mobject_list", () => {
  it("descends into groups to excise removed submobjects", () => {
    const scene = new Scene();
    const sq = new Square();
    const ci = new Circle();
    const dot = new Circle();
    const grp = new VGroup(sq, ci);
    const out = scene.get_restructured_mobject_list([grp, dot], [sq]);
    expect(out.length).toBe(2);
    expect(out[0]).toBe(ci);
    expect(out[1]).toBe(dot);
  });

  it("keeps everything with an empty removal set", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Circle();
    expect(scene.get_restructured_mobject_list([a, b], []).length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.compile_animations (+ _collect_animation_mobjects
// via get_moving_mobjects, incl. AnimationGroup nesting)
// Python: Create(sq) with run_time=2 override; bound-method args rejected
// with "Use Mobject.animate"; junk raises "Unexpected argument".
// ---------------------------------------------------------------------------
describe("Scene.compile_animations", () => {
  it("prepares animations and applies play-kwarg overrides", () => {
    const scene = new Scene();
    const sq = new Square();
    const anims = scene.compile_animations(new Create(sq), { run_time: 2 });
    expect(anims.length).toBe(1);
    expect(anims[0]).toBeInstanceOf(Create);
    expect(anims[0]!.run_time).toBe(2);
  });

  it("rejects bound mobject methods and unexpected args", () => {
    const scene = new Scene();
    const sq = new Square();
    expect(() => scene.compile_animations(bindMobjectMethod(sq, "shift") as never)).toThrowError(
      /Mobject\.animate/,
    );
    expect(() => scene.compile_animations(42 as never)).toThrowError(/Unexpected argument/);
  });
});

describe("Scene.get_moving_mobjects", () => {
  it("collects animation mobjects including nested groups", () => {
    const scene = new Scene();
    const sq = new Square();
    const ci = new Circle();
    scene.add(sq, ci);
    const anim = new Create(sq);
    anim.begin();
    const moving = scene.get_moving_mobjects(anim);
    expect(moving).toContain(sq);
    const grouped = new AnimationGroup(new Create(sq));
    const moving2 = scene.get_moving_mobjects(grouped);
    expect(moving2).toContain(sq);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.__deepcopy__
// Python: fresh instance of same class; renderer/time_progression skipped;
// camera_class kept by reference; everything else deep-copied.
// ---------------------------------------------------------------------------
describe("Scene.__deepcopy__", () => {
  it("deep-copies mobjects while keeping camera_class by reference", () => {
    const scene = new Scene();
    const sq = new Square();
    scene.add(sq);
    const dup = scene.__deepcopy__();
    expect(dup).toBeInstanceOf(Scene);
    expect(dup).not.toBe(scene);
    expect(dup.camera_class).toBe(scene.camera_class);
    expect(dup.renderer).toBeUndefined();
    expect(dup.mobjects.length).toBe(1);
    expect(dup.mobjects[0]).not.toBe(sq);
    expect(dup.mobjects[0]).toBeInstanceOf(Square);
    dup.mobjects[0]!.shift([5, 0, 0]);
    expectRowsClose([sq.get_center()], [[0, 0, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene._get_animation_time_progression
// Python: single Wait + stop_condition -> "Waiting for <name>";
// single Wait -> "Waiting <num_plays>"; else "Animation <n>: <anim>[, etc.]".
// ---------------------------------------------------------------------------
describe("Scene._get_animation_time_progression", () => {
  function stubScene() {
    const scene = new Scene();
    scene.renderer = { num_plays: 7, skip_animations: true };
    return scene;
  }

  it("labels stop-condition waits", () => {
    const scene = stubScene();
    function myCond(): boolean {
      return true;
    }
    const tp = scene._get_animation_time_progression(
      [new Wait({ run_time: 1, stop_condition: myCond })],
      1,
    );
    expect(tp.description).toBe("Waiting for myCond");
  });

  it("labels plain waits with the play count", () => {
    const scene = stubScene();
    const tp = scene._get_animation_time_progression([new Wait({ run_time: 1 })], 1);
    expect(tp.description).toBe("Waiting 7");
  });

  it("labels animations with an etc. suffix for several", () => {
    const scene = stubScene();
    const sq = new Square();
    const one = scene._get_animation_time_progression([new Create(sq)], 1);
    expect(one.description.startsWith("Animation 7: ")).toBe(true);
    const two = scene._get_animation_time_progression([new Create(sq), new Create(sq)], 1);
    expect(two.description.endsWith(", etc.")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.wait
// Python: validates duration then plays a Wait; negative -> ValueError.
// ---------------------------------------------------------------------------
describe("Scene.wait", () => {
  it("plays a Wait and rejects non-positive durations", () => {
    const scene = new Scene();
    expect(() => scene.wait(0.5)).not.toThrow();
    expect(() => scene.wait(-1)).toThrowError(/duration/);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add_sound
// Python: no-op when renderer.skip_animations; else
// file_writer.add_sound(sound_file, time + time_offset, gain, **kwargs).
// ---------------------------------------------------------------------------
describe("Scene.add_sound", () => {
  it("skips when animations are skipped, else forwards to the file writer", () => {
    const skipped = new Scene({ skip_animations: true });
    skipped.renderer = { skip_animations: true, file_writer: { add_sound: vi.fn() } };
    skipped.add_sound("click.wav");
    expect(skipped.renderer.file_writer.add_sound).not.toHaveBeenCalled();

    const live = new Scene();
    const calls: unknown[][] = [];
    live.renderer = {
      skip_animations: false,
      time: 10,
      file_writer: { add_sound: (...args: unknown[]) => calls.push(args) },
    };
    live.add_sound("click.wav", 2, 0.5);
    expect(calls.length).toBe(1);
    expect(calls[0]![0]).toBe("click.wav");
    expect(calls[0]![1]).toBe(12);
    expect(calls[0]![2]).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene._configure_pygui / on_mouse_press
// Python needs DearPyGUI; TS warns. on_mouse_press fans out to callbacks.
// ---------------------------------------------------------------------------
describe("Scene gui stubs", () => {
  it("_configure_pygui warns instead of opening DearPyGUI", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      expect(() => new Scene()._configure_pygui()).not.toThrow();
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("on_mouse_press invokes every callback", () => {
    const scene = new Scene();
    const seen: string[] = [];
    scene.mouse_press_callbacks.push(() => seen.push("a"), () => seen.push("b"));
    scene.on_mouse_press([0, 0, 0], "LEFT", 0);
    expect(seen).toEqual(["a", "b"]);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.camera.Camera.get_mobjects_to_display
// Python: include_submobjects expands families (points only); excluded are
// removed; include_submobjects=False returns the list as-is.
// ---------------------------------------------------------------------------
describe("Camera.get_mobjects_to_display", () => {
  it("expands families and honors exclusions", () => {
    const cam = new Camera();
    const sq = new Square();
    const ci = new Circle();
    const shown = cam.get_mobjects_to_display([sq, ci]);
    expect(shown.length).toBe(2);
    expect(shown).toContain(sq);
    const minus = cam.get_mobjects_to_display([sq, ci], true, [sq]);
    expect(minus).not.toContain(sq);
    expect(minus).toContain(ci);
    const raw = cam.get_mobjects_to_display([sq, ci], false);
    expect(raw).toEqual([sq, ci]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.VectorScene.__init__
// Python: basis_vector_stroke_width=6.0 default.
// ---------------------------------------------------------------------------
describe("VectorScene.__init__", () => {
  it("stores the basis vector stroke width", () => {
    expect(new VectorScene().basis_vector_stroke_width).toBe(6.0);
    expect(new VectorScene(3.0).basis_vector_stroke_width).toBe(3.0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.__init__ +
// set_to_default_angled_camera_orientation
// Python defaults: phi 70deg, theta -135deg; overrides merge in.
// ---------------------------------------------------------------------------
describe("ThreeDScene", () => {
  it("records ambient rotation and default orientation kwargs", () => {
    const scene = new ThreeDScene();
    expect(scene.ambient_camera_rotation).toBeNull();
    expect(scene.default_angled_camera_orientation_kwargs.phi).toBeCloseTo(
      70 * DEGREES, 12,
    );
    expect(scene.default_angled_camera_orientation_kwargs.theta).toBeCloseTo(
      -135 * DEGREES, 12,
    );
  });

  it("forwards merged orientation kwargs to the camera", () => {
    const scene = new ThreeDScene();
    const seen: Record<string, number> = {};
    scene.renderer = {
      camera: {
        set_phi: (v: number) => { seen.phi = v; },
        set_theta: (v: number) => { seen.theta = v; },
        set_gamma: (v: number) => { seen.gamma = v; },
        set_zoom: (v: number) => { seen.zoom = v; },
        set_focal_distance: (v: number) => { seen.focal_distance = v; },
      },
    };
    scene.set_to_default_angled_camera_orientation();
    expect(seen.phi).toBeCloseTo(70 * DEGREES, 12);
    expect(seen.theta).toBeCloseTo(-135 * DEGREES, 12);
    scene.set_to_default_angled_camera_orientation({ phi: 1.0 });
    expect(seen.phi).toBe(1.0);
    expect(seen.theta).toBeCloseTo(-135 * DEGREES, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.SpecialThreeDScene.set_camera_to_default_position
// Python default position: phi 70deg, theta -110deg.
// ---------------------------------------------------------------------------
describe("SpecialThreeDScene", () => {
  it("sets the camera to the default angled position", () => {
    const scene = new SpecialThreeDScene();
    expect(scene.default_angled_camera_position.phi).toBeCloseTo(70 * DEGREES, 12);
    expect(scene.default_angled_camera_position.theta).toBeCloseTo(-110 * DEGREES, 12);
    const seen: Record<string, number> = {};
    scene.renderer = {
      camera: {
        set_phi: (v: number) => { seen.phi = v; },
        set_theta: (v: number) => { seen.theta = v; },
        set_gamma: (v: number) => { seen.gamma = v; },
        set_zoom: (v: number) => { seen.zoom = v; },
        set_focal_distance: (v: number) => { seen.focal_distance = v; },
      },
    };
    scene.set_camera_to_default_position();
    expect(seen.phi).toBeCloseTo(70 * DEGREES, 12);
    expect(seen.theta).toBeCloseTo(-110 * DEGREES, 12);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols combined with earlier-ported code.
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Create reveals half of a Square (12 points, start [1,1,0])", () => {
    // Python: sq.copy().pointwise_become_partial(sq, 0, 0.5) has 12 points,
    // start [1,1,0], end [-1,-1,0] — the Create(alpha=0.5) path.
    const sq = new Square();
    const create = new Create(sq);
    const [a, b] = create._get_bounds(0.5);
    const part = sq.copy() as VMobject;
    part.pointwise_become_partial(sq, a, b);
    expect(part.get_num_points()).toBe(12);
    expectRowsClose([part.get_start()], [[1, 1, 0]], 1e-9);
    expectRowsClose([part.get_end()], [[-1, -1, 0]], 1e-9);
  });

  it("ApplyMethod runs end-to-end through begin/interpolate", () => {
    const sq = new Square();
    const anim = new ApplyMethod(bindMobjectMethod(sq, "shift"), [1, 0, 0]);
    anim.begin();
    anim.interpolate(1.0);
    expectRowsClose([sq.get_center()], [[1, 0, 0]], 1e-9);
  });

  it("compile -> begin -> moving -> restructure round-trips in a Scene", () => {
    const scene = new Scene();
    const sq = new Square();
    const ci = new Circle();
    scene.add(sq, ci);
    const [anim] = scene.compile_animations(new Create(sq));
    anim!.begin();
    expect(scene.get_moving_mobjects(anim!).length).toBeGreaterThan(0);
    const rest = scene.get_restructured_mobject_list(scene.mobjects, [sq]);
    expect(rest).toEqual([ci]);
  });

  it("Arrow tip participates in VMobject family + 3d corner indexing", () => {
    const arr = new Arrow([0, 0, 0], [2, 0, 0], { buff: 0 });
    expect(arr.has_tip()).toBe(true);
    expectRowsClose([arr.get_first_handle()], rows(arr.points).slice(1, 2), 1e-12);
    expect(get_3d_vmob_start_corner_index(arr)).toBe(0);
    expect(get_3d_vmob_end_corner_index(arr)).toBe(
      Math.floor((arr.get_num_points() - 1) / 6) * 3,
    );
  });

  it("ThreeDScene orientation constants agree with config-driven geometry", () => {
    const scene = new ThreeDScene();
    const face = new ThreeDVMobject();
    face.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);
    expect(face.shade_in_3d).toBe(true);
    const [gradStart, gradEnd] = face.get_gradient_start_and_end_points();
    expect(gradStart.length).toBe(3);
    expect(gradEnd.length).toBe(3);
    expect(scene.default_angled_camera_orientation_kwargs.phi).toBeCloseTo(
      70 * DEGREES, 12,
    );
  });
});
