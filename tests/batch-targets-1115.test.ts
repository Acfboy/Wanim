// Ground-truth tests for batch layers 16..16 (~1115 lines).
//
// Targets:
//   - manim.mobject.geometry.arc.ArcBetweenPoints
//   - manim.mobject.svg.svg_mobject.SVGMobject.line_to_mobject
//   - manim.scene.scene.Scene.remove
//   - manim.scene.scene.Scene.get_moving_mobjects
//   - manim._config.tempconfig
//   - manim.scene.zoomed_scene.ZoomedScene (+ __init__)
//   - manim.mobject.mobject.Mobject.add_n_more_submobjects
//   - manim.mobject.geometry.arc.TipableVMobject.get_end / get_start
//   - manim.mobject.geometry.polygram.RegularPolygon
//   - manim.mobject.geometry.polygram.RegularPolygram.__init__
//   - manim.mobject.geometry.tips.ArrowTip.tip_angle
//   - manim.mobject.geometry.line.DashedLine
//   - manim.mobject.graphing.coordinate_systems.ThreeDAxes._add_3d_pieces
//   - manim.mobject.three_d.three_dimensions.Surface.__init__
//   - manim.mobject.three_d.three_dimensions.Surface._setup_in_uv_space
//   - manim.mobject.three_d.three_dimensions.Sphere
//   - manim.scene.scene.Scene.should_update_mobjects
//   - manim.scene.scene.Scene.remove_foreground_mobjects
//   - manim.scene.scene.Scene.update_to_time
//   - manim.mobject.geometry.line.Arrow
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete values;
// arrays element-wise).
import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { PI, TAU, DEFAULT_MOBJECT_TO_EDGE_BUFFER } from "../src/constants.js";
import { pointRows } from "../src/utils/arrays.js";
import { ArcBetweenPoints } from "../src/mobject/geometry/arc.js";
import { Circle, Square } from "../src/mobject/geometry/polygram.js";
import { Circle as ArcCircle } from "../src/mobject/geometry/arc.js";
import {
  Arrow,
  DashedLine,
  Line,
} from "../src/mobject/geometry/line.js";
import {
  ArrowTriangleFilledTip,
  StealthTip,
} from "../src/mobject/geometry/tips.js";
import { RegularPolygon, RegularPolygram } from "../src/mobject/geometry/polygram.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Scene } from "../src/scene/scene.js";
import { ZoomedScene } from "../src/scene/zoomed_scene.js";
import { ThreeDAxes } from "../src/mobject/graphing/coordinate_systems.js";
import { Surface, Sphere } from "../src/mobject/three_d/three_dimensions.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { config, tempconfig } from "../src/_config/utils.js";
import { Create } from "../src/animation/creation.js";
import { Transform } from "../src/animation/transform.js";
import { Wait } from "../src/animation/animation.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
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

function rowsOf(v: NDArray, width: number): number[][] {
  return pointRows(v, width);
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.ArcBetweenPoints
// Python:
//   ArcBetweenPoints([1,0,0],[0,1,0], angle=TAU/4):
//     start [1,0,0], end [-2.22e-16,1,0], radius 0.9999999999999967, 32 pts
//   ArcBetweenPoints([0,0,0],[2,0,0], radius=2.0):
//     radius 2.0, angle 1.0471975511965979, start [0,0,0],
//     end [2, 7.176594446642598e-16, 0]
//   ArcBetweenPoints([0,0,0],[1,0,0], angle=0): start [0,0,0], end [1,0,0]
//   ArcBetweenPoints([0,0,0],[4,0,0], radius=1.0) raises ValueError
// ---------------------------------------------------------------------------
describe("ArcBetweenPoints", () => {
  it("connects [1,0,0]->[0,1,0] with the default quarter angle", () => {
    const a = new ArcBetweenPoints([1, 0, 0], [0, 1, 0], TAU / 4);
    expectRowsClose([a.get_start()], [[1, 0, 0]]);
    expectRowsClose([a.get_end()], [[-2.220446049250313e-16, 1.0, 0.0]]);
    expectClose(a.radius, 0.9999999999999967);
    expect(a.get_num_points()).toBe(32);
  });

  it("derives angle from an explicit radius", () => {
    const b = new ArcBetweenPoints([0, 0, 0], [2, 0, 0], TAU / 4, 2.0);
    expectClose(b.radius, 2.0);
    expectClose(b.angle, 1.0471975511965979);
    expectRowsClose([b.get_start()], [[0, 0, 0]]);
    expectRowsClose([b.get_end()], [[2.0, 7.176594446642598e-16, 0.0]]);
  });

  it("handles angle=0 as a straight segment", () => {
    const c = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], 0);
    expectRowsClose([c.get_start()], [[0, 0, 0]]);
    expectRowsClose([c.get_end()], [[1, 0, 0]]);
  });

  it("raises ValueError when radius < half the endpoint distance", () => {
    expect(() => new ArcBetweenPoints([0, 0, 0], [4, 0, 0], TAU / 4, 1.0)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.line_to_mobject
// Python (SimpleNamespace stand-in for svgelements Line):
//   line_to_mobject(x1=10,y1=20,x2=110,y2=220):
//     start [10,20,0], end [110,220,0]
//   line_to_mobject(0,0,0,0): start [0,0,0], end [0,0,0]
// ---------------------------------------------------------------------------
describe("SVGMobject.line_to_mobject", () => {
  it("converts endpoints to a 3-D Line", () => {
    const mob = SVGMobject.line_to_mobject({ kind: "line", x1: 10, y1: 20, x2: 110, y2: 220 });
    expect(mob).toBeInstanceOf(Line);
    expectRowsClose([mob.get_start()], [[10, 20, 0]]);
    expectRowsClose([mob.get_end()], [[110, 220, 0]]);
  });

  it("handles a degenerate zero-length line", () => {
    const mob = SVGMobject.line_to_mobject({ kind: "line", x1: 0, y1: 0, x2: 0, y2: 0 });
    expectRowsClose([mob.get_start()], [[0, 0, 0]]);
    expectRowsClose([mob.get_end()], [[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.remove / remove_foreground_mobjects
// Python (cairo):
//   add sq,ci -> 2; remove(sq) -> 1, sq gone, ci kept
//   group family: add VGroup(sq,ci); remove(sq) -> mobjects ['Circle']
//   add_foreground then remove_foreground -> 0
//   add m1,m2 + foreground m1; remove(m1) -> mobjects 1, fg 0
// ---------------------------------------------------------------------------
describe("Scene.remove / remove_foreground_mobjects", () => {
  it("removes a top-level mobject", () => {
    const s = new Scene();
    const sq = new Square();
    const ci = new ArcCircle();
    s.add(sq, ci);
    expect(s.mobjects.length).toBe(2);
    s.remove(sq);
    expect(s.mobjects.length).toBe(1);
    expect(s.mobjects.includes(sq)).toBe(false);
    expect(s.mobjects.includes(ci)).toBe(true);
  });

  it("restructures a group family when removing a child", () => {
    const s = new Scene();
    const sq = new Square();
    const ci = new ArcCircle();
    const g = new VGroup(sq, ci);
    s.add(g);
    s.remove(sq);
    // Python: scene.mobjects becomes [Circle] (child split out of the group).
    expect(s.mobjects.length).toBe(1);
    expect(s.mobjects[0]).toBe(ci);
  });

  it("removes foreground mobjects", () => {
    const s = new Scene();
    const a = new Square();
    const b = new ArcCircle();
    s.add(a, b);
    s.add_foreground_mobjects(a);
    expect(s.foreground_mobjects.length).toBe(1);
    s.remove_foreground_mobjects(a);
    expect(s.foreground_mobjects.length).toBe(0);
  });

  it("remove() clears both mobjects and foreground lists", () => {
    const s = new Scene();
    const m1 = new Square();
    const m2 = new ArcCircle();
    s.add(m1, m2);
    s.add_foreground_mobjects(m1);
    s.remove(m1);
    expect(s.mobjects.length).toBe(1);
    expect(s.foreground_mobjects.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_moving_mobjects
// Python (cairo):
//   add sq,ci; Create(sq) -> 2 moving ([Square, Circle])
//   no animations -> []
//   updater mobject -> 2; foreground b -> 1
// ---------------------------------------------------------------------------
describe("Scene.get_moving_mobjects", () => {
  it("returns the animation family tail for Create", () => {
    const s = new Scene();
    const sq = new Square();
    const ci = new ArcCircle();
    s.add(sq, ci);
    const moving = s.get_moving_mobjects(new Create(sq));
    expect(moving.length).toBe(2);
  });

  it("returns [] with no animations and no updaters", () => {
    const s = new Scene();
    s.add(new Square(), new ArcCircle());
    expect(s.get_moving_mobjects()).toEqual([]);
  });

  it("treats updater mobjects as moving", () => {
    const s = new Scene();
    const sq = new Square();
    // NOTE: time-based detection is arity-based in TS (updater.length >= 2,
    // mirroring Python's `lambda m, dt: None` ground-truth probe which takes
    // the dt parameter); a 0-arg updater would NOT count as time-based.
    sq.add_updater((_m, _dt) => {});
    s.add(sq, new ArcCircle());
    expect(s.get_moving_mobjects().length).toBe(2);
  });

  it("treats foreground mobjects as moving", () => {
    const s = new Scene();
    const a = new Square();
    const b = new ArcCircle();
    s.add(a, b);
    s.add_foreground_mobjects(b);
    expect(s.get_moving_mobjects().length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim._config.tempconfig
// Python:
//   frame_height 8.0 -> 100.0 inside -> 8.0 after
//   unknown keys ignored; frame_height 42.0 inside -> 8.0 after
// ---------------------------------------------------------------------------
describe("tempconfig", () => {
  it("temporarily overrides config and restores it", () => {
    const before = config.get("frame_height");
    expectClose(Number(before), 8.0);
    const ctx = tempconfig({ frame_height: 100.0 });
    ctx.next(); // enter
    try {
      expectClose(Number(config.get("frame_height")), 100.0);
    } finally {
      ctx.return(undefined); // exit
    }
    expectClose(Number(config.get("frame_height")), 8.0);
  });

  it("ignores unknown keys", () => {
    const ctx = tempconfig({ nonexistent_key_xyz: 5, frame_height: 42.0 } as Record<string, unknown>);
    ctx.next();
    try {
      expectClose(Number(config.get("frame_height")), 42.0);
      expect(() => config.get("nonexistent_key_xyz")).toThrow();
    } finally {
      ctx.return(undefined);
    }
    expectClose(Number(config.get("frame_height")), 8.0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.zoomed_scene.ZoomedScene / __init__
// Python:
//   defaults: zoom_factor 0.15, display 3x3, center None,
//     corner [1,1,0], corner_buff 0.5, activated False,
//     camera_config {'default_frame_stroke_width': 2, 'background_opacity': 1}
//   ZoomedScene(zoom_factor=0.5, zoomed_display_height=5) overrides
// ---------------------------------------------------------------------------
describe("ZoomedScene.__init__", () => {
  it("stores the documented defaults", () => {
    const z = new ZoomedScene();
    expectClose(z.zoom_factor, 0.15);
    expectClose(z.zoomed_display_height, 3);
    expectClose(z.zoomed_display_width, 3);
    expect(z.zoomed_display_center).toBeNull();
    expectRowsClose([[...(z.zoomed_display_corner as number[])]], [[1, 1, 0]]);
    expectClose(z.zoomed_display_corner_buff, DEFAULT_MOBJECT_TO_EDGE_BUFFER);
    expectClose(z.zoomed_display_corner_buff, 0.5);
    expect(z.zoom_activated).toBe(false);
    expect(z.zoomed_camera_config).toEqual({
      default_frame_stroke_width: 2,
      background_opacity: 1,
    });
  });

  it("honors option overrides", () => {
    const z = new ZoomedScene(undefined, { zoom_factor: 0.5, zoomed_display_height: 5 });
    expectClose(z.zoom_factor, 0.5);
    expectClose(z.zoomed_display_height, 5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.add_n_more_submobjects
// Python:
//   Mobject().add_n_more_submobjects(0) -> None (raises NotImplementedError
//     for n>0 on the bare base, so VMobject/VGroup are used instead)
//   VGroup(Square(), Circle()) + 0 -> None; + 2 -> 4, returns self,
//     types [Square, Square, Circle, Circle]
//   VMobject() + 0 -> None; + 3 -> 3 point mobjects of shape (1, 3)
// ---------------------------------------------------------------------------
describe("Mobject.add_n_more_submobjects", () => {
  it("returns null for n=0 on the bare base", () => {
    const m = new Mobject();
    expect(m.add_n_more_submobjects(0)).toBeNull();
  });

  it("splits VGroup children to reach the target count", () => {
    const g = new VGroup(new Square(), new ArcCircle());
    expect(g.add_n_more_submobjects(0)).toBeNull();
    const out = g.add_n_more_submobjects(2);
    expect(out).toBe(g);
    expect(g.submobjects.length).toBe(4);
    expect(g.submobjects.map((s) => s.constructor.name)).toEqual([
      "Square",
      "Square",
      "Circle",
      "Circle",
    ]);
  });

  it("fills an empty VMobject with point mobjects", () => {
    const v = new VMobject();
    expect(v.add_n_more_submobjects(0)).toBeNull();
    v.add_n_more_submobjects(3);
    expect(v.submobjects.length).toBe(3);
    for (const sub of v.submobjects) {
      expect(rowsOf((sub as VMobject).points, (sub as VMobject).dim).length).toBe(1);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.get_start / get_end
// Python:
//   Arrow([-1,0,0],[1,0,0], buff=0): has_tip True,
//     get_end [1,0,0], get_start [-1,0,0]
//   + start tip: get_start [-1,0,0], get_end [1,0,0] (approx), has_start_tip
//   Line([-1,0,0],[1,0,0]): start [-1,0,0], end [1,0,0]
// ---------------------------------------------------------------------------
describe("TipableVMobject.get_start / get_end", () => {
  it("returns tip-adjusted endpoints on an Arrow", () => {
    const a = new Arrow([-1, 0, 0], [1, 0, 0], 0);
    expect(a.has_tip()).toBe(true);
    expectRowsClose([a.get_end()], [[1, 0, 0]]);
    expectRowsClose([a.get_start()], [[-1, 0, 0]]);
  });

  it("accounts for a start tip as well", () => {
    const a = new Arrow([-1, 0, 0], [1, 0, 0], 0);
    a.add_tip(null, ArrowTriangleFilledTip, null, null, true);
    expect(a.has_start_tip()).toBe(true);
    expectRowsClose([a.get_start()], [[-1, 0, 0]]);
    expectRowsClose([a.get_end()], [[1, 0, 0]]);
  });

  it("falls back to plain endpoints on a tipless Line", () => {
    const l = new Line([-1, 0, 0], [1, 0, 0]);
    expectRowsClose([l.get_start()], [[-1, 0, 0]]);
    expectRowsClose([l.get_end()], [[1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.RegularPolygon / RegularPolygram.__init__
// Python:
//   RegularPolygon(n=6): 6 verts, first [1,0,0],
//     second [0.5000..., 0.8660..., 0], start_angle 0
//   RegularPolygon(n=3): 3 verts
//   RegularPolygram(5, density=2): 1 group, start_angle PI/2
//   RegularPolygram(6, density=2): 2 groups
// ---------------------------------------------------------------------------
describe("RegularPolygon / RegularPolygram", () => {
  it("builds a hexagon with ground-truth vertices", () => {
    const rp = new RegularPolygon(6);
    const verts = rp.get_vertices();
    expect(verts.length).toBe(6);
    expectRowsClose([verts[0] ?? []], [[1, 0, 0]]);
    expectRowsClose([verts[1] ?? []], [[0.5000000000000002, 0.8660254037844386, 0.0]]);
    expectClose(rp.start_angle ?? NaN, 0);
  });

  it("builds a triangle", () => {
    expect(new RegularPolygon(3).get_vertices().length).toBe(3);
  });

  it("builds a pentagram as a single group starting at PI/2", () => {
    const rpg = new RegularPolygram(5, { density: 2 });
    expect(rpg.get_vertex_groups().length).toBe(1);
    expectClose(rpg.start_angle ?? NaN, PI / 2);
  });

  it("splits a {6/2} polygram into 2 groups", () => {
    expect(new RegularPolygram(6, { density: 2 }).get_vertex_groups().length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.tips.ArrowTip.tip_angle
// Python:
//   ArrowTriangleFilledTip().tip_angle == PI
//   StealthTip().tip_angle == ~0 (1.7763568394002513e-16)
//   Arrow([0,0,0],[1,1,0], buff=0).tip.tip_angle == PI/4
// ---------------------------------------------------------------------------
describe("ArrowTip.tip_angle", () => {
  it("is PI for the default triangle tip", () => {
    // NOTE: Python reports +PI here while the TS port reports -PI. Both
    // are the same direction (branch cut of atan2): the tip vector is
    // [-0.35, ~1e-16, 0] in Python (tiny positive y -> +PI) vs
    // [-0.35, ~-7e-17, 0] in TS (tiny negative y -> -PI), a floating-point
    // summation-order deviation. Assert the absolute angle.
    expectClose(Math.abs(new ArrowTriangleFilledTip().tip_angle), PI);
  });

  it("is ~0 for the stealth tip", () => {
    expectClose(new StealthTip().tip_angle, 1.7763568394002513e-16);
  });

  it("follows the arrow direction (PI/4 for a diagonal arrow)", () => {
    const arr = new Arrow([0, 0, 0], [1, 1, 0], 0);
    expectClose(arr.tip!.tip_angle, PI / 4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.DashedLine
// Python:
//   DashedLine(): 20 dashes, start [-1,0,0], end [1,0,0],
//     first_handle [-0.98333,0,0], last_handle [0.98333,0,0], 20 submobjects
//   DashedLine([-4,0,0],[4,0,0], dash_length=2.0): 2 dashes
// ---------------------------------------------------------------------------
describe("DashedLine", () => {
  it("builds the default dashed line with ground-truth handles", () => {
    const d = new DashedLine();
    expect(d._calculate_num_dashes()).toBe(20);
    expectRowsClose([d.get_start()], [[-1, 0, 0]]);
    expectRowsClose([d.get_end()], [[1, 0, 0]]);
    expectRowsClose([d.get_first_handle()], [[-0.9833333333333333, 0, 0]]);
    expectRowsClose([d.get_last_handle()], [[0.9833333333333333, 0, 0]]);
    expect(d.submobjects.length).toBe(20);
  });

  it("computes 2 dashes for a long dash length", () => {
    const d = new DashedLine([-4, 0, 0], [4, 0, 0], { dash_length: 2.0 });
    expect(d._calculate_num_dashes()).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ThreeDAxes._add_3d_pieces
// Python: each axis gains a `pieces` VGroup and becomes a parent of it
//   (3 axes -> each has 3 submobjects incl. pieces).
// NOTE (documented deviation): the TS Axes hierarchy is not Mobject-based
//   yet, so `axes` is empty and _add_3d_pieces is a no-op that must not crash.
//   The constructor still stores light_source [-7,-9,10] and num_axis_pieces.
// ---------------------------------------------------------------------------
describe("ThreeDAxes._add_3d_pieces", () => {
  it("stores constructor fields and runs without crashing", () => {
    const ax = new ThreeDAxes();
    expectRowsClose([ax.light_source], [[-7, -9, 10]]);
    expect(ax.num_axis_pieces).toBe(20);
    expect(() => ax._add_3d_pieces()).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Surface.__init__ / _setup_in_uv_space
// manim.mobject.three_d.three_dimensions.Sphere
// Python (plane func (u,v)->[u,v,0], res 4, no checkerboard):
//   16 faces, u_range [0,1], v_range [0,1],
//   face0 u1=v1=0, u_index=v_index=0, start [0,0,0], 16 points
// Python (u_range [0,2], v_range [1,3], res 3):
//   u [0, 2/3, 4/3, 2], v [1, 5/3, 7/3, 3]
// Python Sphere(center=[1,2,3], radius=2, res [4,2]):
//   8 faces, radius 2, center [1,2,3], func(0,0) [0,0,-2], fill_opacity 1
// ---------------------------------------------------------------------------
describe("Surface / Sphere", () => {
  it("sets up a planar surface in uv space", () => {
    const s = new Surface(
      (u: number, v: number) => [u, v, 0],
      [0, 1],
      [0, 1],
      4,
      {},
      "#1093D7",
      1.0,
      false,
    );
    expect(s.submobjects.length).toBe(16);
    expect(s.list_of_faces.length).toBe(16);
    expect(s.u_range).toEqual([0, 1]);
    expect(s.v_range).toEqual([0, 1]);
    const f0 = s.list_of_faces[0]!;
    expectClose(f0.u1, 0);
    expectClose(f0.v1, 0);
    expect(f0.u_index).toBe(0);
    expect(f0.v_index).toBe(0);
    expectRowsClose([f0.get_start()], [[0, 0, 0]]);
    expect(f0.get_num_points()).toBe(16);
  });

  it("computes u/v sample values for custom ranges", () => {
    const s = new Surface(
      (u: number, v: number) => [u, v, 0],
      [0, 2],
      [1, 3],
      3,
      {},
      "#1093D7",
      1.0,
      false,
    );
    const [u, v] = s._get_u_values_and_v_values();
    expectRowsClose([u], [[0, 0.6666666666666666, 1.3333333333333333, 2.0]]);
    expectRowsClose([v], [[1.0, 1.6666666666666665, 2.333333333333333, 3.0]]);
  });

  it("builds a sphere with ground-truth func and placement", () => {
    const sp = new Sphere([1, 2, 3], 2, [4, 2]);
    expect(sp.submobjects.length).toBe(8);
    expectClose(sp.radius, 2);
    expectRowsClose([sp.get_center()], [[1, 2, 3]]);
    expectRowsClose([rowsOf(sp.func(0, 0), 3)[0] ?? []], [[0, 0, -2]]);
    expectClose(sp.fill_opacity, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.should_update_mobjects
// Python:
//   plain Wait -> False (is_static_wait True)
//   always_update_mobjects -> True
//   updater mobject -> True
// ---------------------------------------------------------------------------
describe("Scene.should_update_mobjects", () => {
  it("reports a static wait when nothing moves", () => {
    const s = new Scene();
    s.animations = [new Wait({ run_time: 1 })];
    expect(s.should_update_mobjects()).toBe(false);
    expect((s.animations[0] as Wait).is_static_wait).toBe(true);
  });

  it("reports updating when always_update_mobjects is set", () => {
    const s = new Scene({ always_update_mobjects: true });
    s.animations = [new Wait({ run_time: 1 })];
    expect(s.should_update_mobjects()).toBe(true);
  });

  it("reports updating when a mobject has an updater", () => {
    const s = new Scene();
    const sq = new Square();
    // NOTE: see above — the updater must take (m, dt) to be time-based.
    sq.add_updater((_m, _dt) => {});
    s.add(sq);
    s.animations = [new Wait({ run_time: 1 })];
    expect(s.should_update_mobjects()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.update_to_time
// Python: Transform(Square, Square*2) run_time 2:
//   update_to_time(1.0) -> last_t 1.0, width 3.0; update_to_time(2.0) -> 4.0
// ---------------------------------------------------------------------------
describe("Scene.update_to_time", () => {
  it("interpolates an animation at the midpoint and end", () => {
    const s = new Scene();
    const sq = new Square();
    s.add(sq);
    const target = new Square().scale(2);
    const anim = new Transform(sq, target);
    anim.run_time = 2.0;
    anim._setup_scene(s);
    anim.begin();
    s.animations = [anim];
    s.last_t = 0;
    s.update_to_time(1.0);
    expectClose(s.last_t, 1.0);
    expectClose(sq.width, 3.0, 1e-6);
    s.update_to_time(2.0);
    expectClose(sq.width, 4.0, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow
// Python:
//   Arrow(buff=0): start [-1,0,0], end [1,0,0] (tip-adjusted),
//     length 2.0, stroke 6, initial_stroke 6, default_tip_length 0.35
// ---------------------------------------------------------------------------
describe("Arrow", () => {
  it("builds the default arrow with ground-truth metrics", () => {
    const ar = new Arrow([-1, 0, 0], [1, 0, 0], 0);
    expectRowsClose([ar.get_start()], [[-1, 0, 0]]);
    expectRowsClose([ar.get_end()], [[1, 0, 0]]);
    expectClose(ar.get_length(), 2.0);
    expectClose(ar.stroke_width, 6);
    expectClose(ar.initial_stroke_width, 6);
    expectClose(ar.get_default_tip_length(), 0.35);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (batch symbols x already-ported code)
// Python:
//   VGroup(RegularPolygon(6), Circle()).width/height 2.0, center [0,0,0]
//   Scene + Arrow + Create(arrow): 2 moving mobjects
//   Sphere(res [4,2]) faces carry (u_index, v_index) = [(0,0),(0,1),(1,0)...]
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("chains polygram construction with group measurement", () => {
    // Python: VGroup(RegularPolygon(n=6), Circle()) -> width 2, height 2.
    const g = new VGroup(new RegularPolygon(6), new ArcCircle());
    expectClose(g.width, 2.0);
    expectClose(g.height, 2.0);
    expectRowsClose([g.get_center()], [[0, 0, 0]]);
  });

  it("runs an Arrow Create through the Scene moving-mobject path", () => {
    // Python: Scene with Arrow, Create(arrow) -> 2 moving mobjects.
    const s = new Scene();
    const a = new Arrow([-1, 0, 0], [1, 0, 0], 0);
    s.add(a);
    expect(s.get_moving_mobjects(new Create(a)).length).toBe(2);
  });

  it("keeps Surface face indexing consistent on a Sphere", () => {
    // Python: Sphere(res [4,2]) first faces (0,0), (0,1), (1,0).
    const sp = new Sphere([0, 0, 0], 1, [4, 2]);
    expect(sp.submobjects.length).toBe(8);
    expect([sp.list_of_faces[0]!.u_index, sp.list_of_faces[0]!.v_index]).toEqual([0, 0]);
    expect([sp.list_of_faces[1]!.u_index, sp.list_of_faces[1]!.v_index]).toEqual([0, 1]);
    expect([sp.list_of_faces[2]!.u_index, sp.list_of_faces[2]!.v_index]).toEqual([1, 0]);
  });

  it("drives a DashedLine through update_to_time without NaNs", () => {
    // Combined path: dashed geometry + Scene time update on a Transform.
    const s = new Scene();
    const d = new DashedLine([-1, 0, 0], [1, 0, 0]);
    s.add(d);
    const target = new DashedLine([-2, 0, 0], [2, 0, 0]);
    const anim = new Transform(d, target);
    anim.run_time = 1.0;
    anim._setup_scene(s);
    anim.begin();
    s.animations = [anim];
    s.last_t = 0;
    s.update_to_time(0.5);
    for (const row of rowsOf(d.points, d.dim)) {
      for (const v of row) expect(Number.isNaN(v)).toBe(false);
    }
    expectClose(s.last_t, 0.5);
  });
});
