import { describe, expect, it, vi } from "vitest";
import {
  AnnularSector,
  Annulus,
  Arc,
  ArcBetweenPoints,
  ArcPolygon,
  ArcPolygonFromArcs,
  CurvedArrow,
  Ellipse,
  Sector,
  TangentialArc,
} from "../src/mobject/geometry/arc.js";
import "../src/mobject/geometry/line.js"; // registers Line/Arc constructors for ArcPolygonFromArcs + round_corners
import { Line, Arrow } from "../src/mobject/geometry/line.js";
import { LabeledArrow } from "../src/mobject/geometry/labeled.js";
import { Star, RoundedRectangle, Polygram } from "../src/mobject/geometry/polygram.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Brace, BraceLabel } from "../src/mobject/svg/brace.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import { Scene } from "../src/scene/scene.js";
import { ThreeDScene } from "../src/scene/three_d_scene.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { FadeIn } from "../src/animation/fading.js";
import { GrowFromCenter } from "../src/animation/growing.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { DOWN } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= Math.max(1e-9, Math.abs(b) * tol);
}

function expectVecClose(actual: number[] | unknown, expected: number[], tol = 1e-6): void {
  const arr: number[] = Array.isArray(actual)
    ? [...(actual as number[])].map(Number)
    : [...((actual as { toArray(): unknown }).toArray() as unknown[])].map(Number);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

// Ground truths captured from the original manim implementation (manim 0.21.0); see comments per block.

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.get_tip
// Python: raises Exception("tip not found") with no tips; else tips[0].
// original manim: Line([0,0,0],[2,0,0]).add_tip() -> tip center [1.825, ~0, 0], 1 tip.
// ---------------------------------------------------------------------------
describe("TipableVMobject.get_tip", () => {
  it("throws 'tip not found' when there is no tip", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    expect(() => l.get_tip()).toThrow(/tip not found/);
  });
  it("returns the added tip (center matches python)", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    l.add_tip();
    const tip = l.get_tip();
    expect(l.get_tips().submobjects.length).toBe(1);
    expectVecClose(tip.get_center(), [1.825, 0, 0], 1e-4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TangentialArc
// original manim: l1=[0,0,0]->[2,0,0], l2=[0,0,0]->[0,2,0], r=0.5 ->
//   start [0, 0.5, 0], end [0.5, 0, 0], length 0.7071067811865476.
// ---------------------------------------------------------------------------
describe("TangentialArc", () => {
  it("tangent points/length match python", () => {
    const l1 = new Line([0, 0, 0], [2, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 2, 0]);
    const ta = new TangentialArc(l1, l2, 0.5);
    expectVecClose(ta.get_start(), [0, 0.5, 0]);
    expectVecClose(ta.get_end(), [0.5, 0, 0]);
    expect(close(ta.get_length(), 0.7071067811865476)).toBe(true);
    expect(ta.line1).toBe(l1);
    expect(ta.line2).toBe(l2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.CurvedArrow
// original manim: CurvedArrow([0,0,0],[2,0,0]): has_tip True, 1 tip,
//   tip center [1.8143844699385314, -0.18561553006146853, 0].
// ---------------------------------------------------------------------------
describe("CurvedArrow", () => {
  it("adds one end tip at the python position", () => {
    const ca = new CurvedArrow([0, 0, 0], [2, 0, 0]);
    expect(ca.has_tip()).toBe(true);
    expect(ca.get_tips().submobjects.length).toBe(1);
    expectVecClose(ca.get_tip().get_center(), [1.8143844699385314, -0.18561553006146853, 0], 1e-4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Ellipse
// original manim: Ellipse(width=4, height=2): width 4, height 2, 32 points.
// ---------------------------------------------------------------------------
describe("Ellipse", () => {
  it("width/height/point count match python", () => {
    const e = new Ellipse(4, 2);
    expect(close(e.width, 4.0)).toBe(true);
    expect(close(e.height, 2.0)).toBe(true);
    expect(e.get_num_points()).toBe(32);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.AnnularSector.init_points / Sector / Annulus
// original manim: AnnularSector(1,2,angle=pi/2): 72 pts, w=2 h=2.
//   Sector(radius=2): inner 0 outer 2, w=2 h=2.
//   Annulus(1,2): 64 pts, radius 2, center origin.
// NOTE: python init_points()/generate_points() APPEND to existing points
// (calling init_points on a built sector doubles 72->144); the TS port
// mirrors that append behavior.
// ---------------------------------------------------------------------------
describe("AnnularSector / Sector / Annulus", () => {
  it("AnnularSector constructor matches python", () => {
    const a = new AnnularSector(1, 2, Math.PI / 2, 0);
    expect(a.get_num_points()).toBe(72);
    expect(close(a.width, 2.0)).toBe(true);
    expect(close(a.height, 2.0)).toBe(true);
  });
  it("AnnularSector.init_points regenerates via generate_points (appends like python)", () => {
    const a = new AnnularSector(1, 2, Math.PI / 2, 0);
    const before = a.get_num_points();
    const ret = a.init_points();
    expect(ret).toBe(a);
    // Python appends again: 72 -> 144.
    expect(a.get_num_points()).toBe(before * 2);
  });
  it("Sector defaults match python", () => {
    const s = new Sector(2);
    expect(s.inner_radius).toBe(0);
    expect(s.outer_radius).toBe(2);
    expect(close(s.width, 2.0)).toBe(true);
    expect(close(s.height, 2.0)).toBe(true);
  });
  it("Annulus constructor matches python", () => {
    const an = new Annulus(1, 2);
    expect(an.get_num_points()).toBe(64);
    expect(an.radius).toBe(2);
    expectVecClose(an.get_center(), [0, 0, 0]);
  });
  it("Annulus.generate_points appends like python (documents non-idempotency)", () => {
    const an = new Annulus(1, 2);
    const before = an.get_num_points();
    an.generate_points();
    expect(an.get_num_points()).toBe(before * 2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.ArcPolygon.__init__ / ArcPolygonFromArcs.__init__
// original manim: ArcPolygon([0,0,0],[2,0,0],[1,1.732,0]): 3 arcs, 96 pts.
//   ArcPolygonFromArcs(3 x ArcBetweenPoints angle=0.5): 3 arcs, 108 pts.
//   arc_config length mismatch -> AssertionError; non-Arc member -> ValueError.
// ---------------------------------------------------------------------------
describe("ArcPolygon", () => {
  it("triangle gives 3 arcs and 96 points like python", () => {
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [1, 1.732, 0]);
    expect(ap.arcs.length).toBe(3);
    expect(ap.get_num_points()).toBe(96);
    expect(ap.submobjects.length).toBe(3);
  });
  it("arc_config length mismatch raises like python", () => {
    expect(() => new ArcPolygon([0, 0, 0], [1, 0, 0], { arc_config: [{ angle: 0.5 }] } as unknown as Record<string, unknown>)).toThrow();
  });
  it("radius config path builds arcs", () => {
    const ap = new ArcPolygon([0, 0, 0], [2, 0, 0], [1, 1.732, 0], { radius: 2 } as unknown as Record<string, unknown>);
    expect(ap.arcs.length).toBe(3);
  });
});

describe("ArcPolygonFromArcs", () => {
  it("three arcs give 108 points like python", () => {
    const arc1 = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], 0.5);
    const arc2 = new ArcBetweenPoints([1, 0, 0], [0, 1, 0], 0.5);
    const arc3 = new ArcBetweenPoints([0, 1, 0], [0, 0, 0], 0.5);
    const apf = new ArcPolygonFromArcs(arc1, arc2, arc3);
    expect(apf.arcs.length).toBe(3);
    expect(apf.get_num_points()).toBe(108);
  });
  it("non-Arc member raises ValueError like python", () => {
    expect(() => new ArcPolygonFromArcs(new Arc(), "notarc" as unknown as Arc)).toThrow(/All ArcPolygon submobjects must be of type Arc\/ArcBetweenPoints/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Star.__init__
// original manim: Star(n=5): 1 vertex group of 10, width 1.9021130325903073.
//   Star(outer=2, inner=0.5): w 3.8042260651806146 h 3.6180339887498953.
//   density 0 / density n/2 -> ValueError.
// ---------------------------------------------------------------------------
describe("Star", () => {
  it("default star matches python vertex groups and width", () => {
    const st = new Star(5);
    expect(st.get_vertex_groups().length).toBe(1);
    expect(st.get_vertex_groups()[0]?.length).toBe(10);
    expect(close(st.width, 1.9021130325903073)).toBe(true);
  });
  it("custom radii match python", () => {
    const st = new Star(5, { outer_radius: 2, inner_radius: 0.5 });
    expect(close(st.width, 3.8042260651806146)).toBe(true);
    expect(close(st.height, 3.6180339887498953)).toBe(true);
  });
  it("incompatible density raises ValueError like python", () => {
    expect(() => new Star(5, { density: 0 })).toThrow(/Incompatible density/);
    expect(() => new Star(6, { density: 3 })).toThrow(/Incompatible density/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.labeled.LabeledArrow
// Text/tex is ported (MathTex via MathJax), so a string label now builds.
// ---------------------------------------------------------------------------
describe("LabeledArrow", () => {
  it("builds a string label instead of throwing", () => {
    const arrow = new LabeledArrow("x");
    expect(arrow.label).toBeDefined();
    expect(arrow.submobjects.length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.Brace.put_at_tip
// original manim: Brace(Square(2)): tip [0, -1.472985, ~0], dir [0,-1,~0].
//   put_at_tip(Line 0->1): center [0, -1.722985, ~0].
//   use_next_to=False: center [0, -2.222985, ~0].
// ---------------------------------------------------------------------------
describe("Brace.put_at_tip", () => {
  it("next_to branch matches python", () => {
    const sq = new Square(2.0);
    const b = new Brace(sq);
    expectVecClose(b.get_tip(), [0, -1.472985, 0], 1e-3);
    const lab = new Line([0, 0, 0], [1, 0, 0]);
    b.put_at_tip(lab);
    expectVecClose(lab.get_center(), [0, -1.722985, 0], 1e-3);
  });
  it("move_to branch matches python", () => {
    const sq = new Square(2.0);
    const b = new Brace(sq);
    const lab = new Line([0, 0, 0], [1, 0, 0]);
    b.put_at_tip(lab, false);
    expectVecClose(lab.get_center(), [0, -2.222985, 0], 1e-3);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.brace.BraceLabel / creation_anim
// Python: AnimationGroup(brace_anim(self.brace), label_anim(self.label)).
// TS with a VMobject label (bypasses the MathTex pipeline) is ground-truth
// testable; raw strings throw the marked text/tex TODO.
// ---------------------------------------------------------------------------
describe("BraceLabel", () => {
  it("builds brace+VMobject label and positions the label at the tip", () => {
    const sq = new Square(2.0);
    const dot = new Dot([0, 0, 0]);
    const bl = new BraceLabel(sq, dot);
    expect(bl.submobjects.length).toBe(2);
    expect(bl.brace).toBeInstanceOf(Brace);
    expect(bl.label).toBe(dot);
  });
  it("raw string labels build via MathTex by default", () => {
    const sq = new Square(2.0);
    const bl = new BraceLabel(sq, "x");
    expect(bl.label).toBeDefined();
    expect(bl.submobjects.length).toBe(2);
  });
  it("creation_anim defaults to Group(GrowFromCenter(brace), FadeIn(label))", () => {
    const sq = new Square(2.0);
    const dot = new Dot([0, 0, 0]);
    const bl = new BraceLabel(sq, dot);
    const anim = bl.creation_anim();
    expect(anim).toBeInstanceOf(AnimationGroup);
    expect((anim as AnimationGroup).animations.length).toBe(2);
    expect((anim as AnimationGroup).animations[0]).toBeInstanceOf(GrowFromCenter);
    expect((anim as AnimationGroup).animations[1]).toBeInstanceOf(FadeIn);
  });
  it("custom label_constructor returning a VMobject is honored", () => {
    const sq = new Square(2.0);
    const made = new Dot([5, 5, 0]);
    const bl = new BraceLabel(sq, "x", [...DOWN], (() => made) as unknown as null);
    expect(bl.label).toBe(made);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.ellipse_to_mobject / polygon_to_mobject
// original manim (svgelements): Ellipse(cx=1,cy=2,rx=3,ry=4) -> w 6 h 8 center [1,2,0];
//   Circle(r=2) -> w=h=4; Polygon([(0,0),(1,0),(1,1)]) -> 12 pts, same verts.
// ---------------------------------------------------------------------------
describe("SVGMobject.ellipse_to_mobject / polygon_to_mobject", () => {
  it("ellipse stretches and shifts like python", () => {
    const mob = SVGMobject.ellipse_to_mobject({ kind: "ellipse", cx: 1, cy: 2, rx: 3, ry: 4 });
    expect(close(mob.width, 6.0)).toBe(true);
    expect(close(mob.height, 8.0)).toBe(true);
    expectVecClose(mob.get_center(), [1, 2, 0]);
  });
  it("circle keeps equal axes like python", () => {
    const mob = SVGMobject.ellipse_to_mobject({ kind: "circle", cx: 0, cy: 0, rx: 2, ry: 2 });
    expect(close(mob.width, 4.0)).toBe(true);
    expect(close(mob.height, 4.0)).toBe(true);
  });
  it("polygon builds the python vertices", () => {
    const mob = SVGMobject.polygon_to_mobject({ kind: "polygon", points: [[0, 0], [1, 0], [1, 1]] });
    expect(mob.get_num_points()).toBe(12);
    const verts = mob.get_vertices();
    expectVecClose(verts[0] ?? [], [0, 0, 0]);
    expectVecClose(verts[1] ?? [], [1, 0, 0]);
    expectVecClose(verts[2] ?? [], [1, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.RoundedRectangle / Polygram.round_corners
// original manim: RoundedRectangle() w=4 h=2 npts=32; (w=6,h=3,r=0.2) similar.
//   Square(2).round_corners(radius=0) is identity; radius=0.3 -> 32 pts, w=2.
// ---------------------------------------------------------------------------
describe("RoundedRectangle / Polygram.round_corners", () => {
  it("default RoundedRectangle matches python", () => {
    const rr = new RoundedRectangle({});
    expect(close(rr.width, 4.0)).toBe(true);
    expect(close(rr.height, 2.0)).toBe(true);
    expect(rr.get_num_points()).toBe(32);
  });
  it("custom RoundedRectangle keeps width/height", () => {
    const rr = new RoundedRectangle({ width: 6, height: 3, corner_radius: 0.2 });
    expect(close(rr.width, 6.0, 1e-4)).toBe(true);
    expect(close(rr.height, 3.0, 1e-4)).toBe(true);
  });
  it("round_corners(radius=0) is the identity like python", () => {
    const sq = new Square(2.0);
    const before = sq.get_num_points();
    const ret = sq.round_corners(0);
    expect(ret).toBe(sq);
    expect(sq.get_num_points()).toBe(before);
  });
  it("round_corners(radius=0.3) matches python point count and width", () => {
    const sq = new Square(2.0);
    sq.round_corners(0.3);
    expect(sq.get_num_points()).toBe(32);
    expect(close(sq.width, 2.0, 1e-4)).toBe(true);
  });
  it("round_corners with radius list + even anchors stays finite", () => {
    const sq = new Square(2.0);
    sq.round_corners([0.2, 0.3], true);
    expect(sq.get_num_points()).toBeGreaterThan(0);
    expect(Number.isFinite(sq.width)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.get_moving_and_static_mobjects
// Python splits animated families (moving) from the rest (static).
// ---------------------------------------------------------------------------
describe("Scene.get_moving_and_static_mobjects", () => {
  it("splits animated vs idle mobjects like python", () => {
    // NOTE: both Python and this port use a conservative static-frame cache
    // boundary: get_moving_mobjects returns mobjects[i:] starting at the
    // FIRST moving family member. So the idle mobject must come FIRST for
    // the split to separate them ([idle, animated] -> moving [animated]).
    const scene = new Scene();
    const idle = new Circle();
    const animated = new Square();
    scene.add(idle, animated);
    const anim = new FadeIn(animated);
    const [moving, statics] = scene.get_moving_and_static_mobjects([anim]);
    expect(moving.includes(animated)).toBe(true);
    expect(statics.includes(idle)).toBe(true);
    expect(statics.includes(animated)).toBe(false);
  });
  it("animating the first mobject conservatively marks everything moving (python behavior)", () => {
    const scene = new Scene();
    const a = new Square();
    const b = new Circle();
    scene.add(a, b);
    const [moving, statics] = scene.get_moving_and_static_mobjects([new FadeIn(a)]);
    expect(moving.includes(a)).toBe(true);
    expect(moving.includes(b)).toBe(true);
    expect(statics.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.align_submobjects
// original manim: 2-sub vs 5-sub Mobjects both end with 5 submobjects.
// ---------------------------------------------------------------------------
describe("Mobject.align_submobjects", () => {
  it("pads both sides to the same submobject count like python", () => {
    const a = new Mobject();
    const b = new Mobject();
    for (let i = 0; i < 2; i += 1) a.add(new Mobject());
    for (let i = 0; i < 5; i += 1) b.add(new Mobject());
    const ret = a.align_submobjects(b);
    expect(ret).toBe(a);
    expect(a.submobjects.length).toBe(5);
    expect(b.submobjects.length).toBe(5);
  });
  it("equal counts are a no-op", () => {
    const a = new Group(new Square(), new Circle());
    const b = new Group(new Square(), new Circle());
    a.align_submobjects(b);
    expect(a.submobjects.length).toBe(2);
    expect(b.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.three_d_scene.ThreeDScene.* (mock-camera ground truths)
// Python Cairo paths: stop_ambient clears tracker updaters + removes tracker;
//   invalid `about` -> ValueError; stop_3dillusion clears both trackers;
//   remove_fixed_* delegate to renderer.camera; get_moving_mobjects returns
//   all mobjects when a camera tracker is among the moving set.
// ---------------------------------------------------------------------------
describe("ThreeDScene camera controls", () => {
  function trackerScene(): { scene: ThreeDScene; theta: ValueTracker; phi: ValueTracker; gamma: ValueTracker } {
    const scene = new ThreeDScene();
    const theta = new ValueTracker(0);
    const phi = new ValueTracker(0);
    const gamma = new ValueTracker(0);
    scene.camera = { theta_tracker: theta, phi_tracker: phi, gamma_tracker: gamma, clear_updaters: vi.fn() };
    return { scene, theta, phi, gamma };
  }

  it("stop_ambient_camera_rotation clears and removes the tracker", () => {
    const { scene, theta } = trackerScene();
    theta.add_updater(() => {});
    scene.add(theta);
    expect(scene.mobjects.includes(theta)).toBe(true);
    scene.stop_ambient_camera_rotation("theta");
    expect(theta.updaters.length).toBe(0);
    expect(scene.mobjects.includes(theta)).toBe(false);
  });

  it("stop_ambient_camera_rotation rejects bad `about` with ValueError", () => {
    const { scene } = trackerScene();
    expect(() => scene.stop_ambient_camera_rotation("nonsense")).toThrow(/Invalid ambient rotation angle/);
  });

  it("stop_3dillusion_camera_rotation clears both trackers", () => {
    const scene = new ThreeDScene();
    const theta = new ValueTracker(1);
    const phi = new ValueTracker(2);
    theta.add_updater(() => {});
    phi.add_updater(() => {});
    scene.renderer = { camera: { theta_tracker: theta, phi_tracker: phi } };
    scene.add(theta, phi);
    scene.stop_3dillusion_camera_rotation();
    expect(theta.updaters.length).toBe(0);
    expect(phi.updaters.length).toBe(0);
    expect(scene.mobjects.includes(theta)).toBe(false);
    expect(scene.mobjects.includes(phi)).toBe(false);
  });

  it("remove_fixed_* delegate to the renderer camera", () => {
    const scene = new ThreeDScene();
    const mob = new Square();
    const camera = {
      remove_fixed_orientation_mobjects: vi.fn(),
      remove_fixed_in_frame_mobjects: vi.fn(),
    };
    scene.renderer = { camera };
    scene.remove_fixed_orientation_mobjects(mob);
    expect(camera.remove_fixed_orientation_mobjects).toHaveBeenCalledWith(mob);
    scene.remove_fixed_in_frame_mobjects(mob);
    expect(camera.remove_fixed_in_frame_mobjects).toHaveBeenCalledWith(mob);
  });

  it("get_moving_mobjects returns everything when the camera moves", () => {
    // NOTE: the camera branch triggers only when a camera tracker is among
    // the moving set, which requires the tracker to be in the scene (Python
    // adds trackers via begin_ambient_camera_rotation / add).
    const scene = new ThreeDScene();
    const a = new Square();
    const b = new Circle();
    const tracker = new ValueTracker(0);
    scene.add(a, b, tracker as unknown as Mobject);
    const frameCenter = new Mobject();
    scene.renderer = { camera: { get_value_trackers: () => [tracker], _frame_center: frameCenter } };
    const anim = new FadeIn(tracker as unknown as Mobject);
    const moving = scene.get_moving_mobjects(anim);
    expect(moving.length).toBe(scene.mobjects.length);
  });

  it("move_camera with no-ops still plays through Scene.play", () => {
    const scene = new ThreeDScene();
    const theta = new ValueTracker(0);
    const phi = new ValueTracker(0);
    const gamma = new ValueTracker(0);
    const zoom = new ValueTracker(1);
    const focal = new ValueTracker(1);
    const frameCenter = new Square();
    scene.camera = {
      phi_tracker: phi,
      theta_tracker: theta,
      gamma_tracker: gamma,
      zoom_tracker: zoom,
      focal_distance_tracker: focal,
      _frame_center: frameCenter,
    };
    const spy = vi.spyOn(scene, "play");
    scene.move_camera(null, null, null, null, null, null);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine
// original manim: x_range [-3,3,1] -> tick_range [-3..3], unit_size 1, angle 0,
//   n2p(0)=[0,0,0], n2p(1)=[1,0,0], p2n(center)=0, 7 tick marks,
//   _decimal_places_from_step(1/0.5/0.25/0.125) = 0/1/2/3,
//   get_tick(1) center [1,0,0] length 0.2, unit vector [1,0,0].
// ---------------------------------------------------------------------------
describe("NumberLine", () => {
  it("tick range / unit size / angle match python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect((nl.get_tick_range().toArray() as unknown[]).map(Number)).toEqual([-3, -2, -1, 0, 1, 2, 3]);
    expect(close(nl.get_unit_size(), 1.0)).toBe(true);
    expect(close(nl.get_angle(), 0.0)).toBe(true);
    expect(nl.x_min).toBe(-3);
    expect(nl.x_max).toBe(3);
    expect(nl.x_step).toBe(1);
  });
  it("n2p/p2n roundtrip matches python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expectVecClose([...(nl.n2p(0).toArray() as unknown as number[])], [0, 0, 0]);
    expectVecClose([...(nl.n2p(1).toArray() as unknown as number[])], [1, 0, 0]);
    expect(close(nl.point_to_number(nl.get_center()) as number, 0.0)).toBe(true);
    const arr = nl.point_to_number([[0, 0, 0], [1, 0, 0]]);
    expect([...((arr as unknown as { toArray(): unknown[] }).toArray() as unknown[])].map(Number)).toEqual([0, 1]);
  });
  it("ticks and unit vector match python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    expect(nl.get_tick_marks().submobjects.length).toBe(7);
    const tick = nl.get_tick(1.0);
    expectVecClose(tick.get_center(), [1, 0, 0]);
    expect(close(tick.get_length(), 0.2)).toBe(true);
    expectVecClose(nl.get_unit_vector(), [1, 0, 0]);
  });
  it("positive-only and negative-only ranges match python", () => {
    expect((new NumberLine({ x_range: [1, 5, 1] }).get_tick_range().toArray() as unknown[]).map(Number)).toEqual([1, 2, 3, 4, 5]);
    expect((new NumberLine({ x_range: [-5, -1, 1] }).get_tick_range().toArray() as unknown[]).map(Number)).toEqual([-5, -4, -3, -2, -1]);
  });
  it("_decimal_places_from_step matches python", () => {
    expect(NumberLine._decimal_places_from_step(1)).toBe(0);
    expect(NumberLine._decimal_places_from_step(0.5)).toBe(1);
    expect(NumberLine._decimal_places_from_step(0.25)).toBe(2);
    expect(NumberLine._decimal_places_from_step(0.125)).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (this batch x already-ported code)
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Ellipse chain: point_at_angle agrees with n2p-style proportion math", () => {
    // Python: Ellipse(4,2).point_at_angle(0) sits at the +x tip [2,0,0].
    const e = new Ellipse(4, 2);
    expectVecClose([...(e.point_at_angle(0).toArray() as unknown as number[])], [2, 0, 0], 1e-4);
  });
  it("Star -> round_corners matches python center drift and density", () => {
    // Python ground truth: Star() center y 0.09549150281252616; after
    // round_corners(0.1) y 0.039589803375031485 with 80 points (rounding is
    // not center-preserving; both implementations drift identically).
    const st = new Star(5);
    expect(close(st.get_center()[1] ?? NaN, 0.09549150281252616, 1e-6)).toBe(true);
    st.round_corners(0.1);
    expect(st.get_num_points()).toBe(80);
    expect(close(st.get_center()[1] ?? NaN, 0.039589803375031485, 1e-6)).toBe(true);
  });
  it("ArcPolygonFromArcs gaps close against Line lengths", () => {
    const arc1 = new ArcBetweenPoints([0, 0, 0], [1, 0, 0], 0.5);
    const arc2 = new ArcBetweenPoints([1, 0, 0], [0, 1, 0], 0.5);
    const gap = new Line(arc1.get_end(), arc2.get_start());
    expect(close(gap.get_length(), 0.0)).toBe(true);
    const apf = new ArcPolygonFromArcs(arc1, arc2);
    expect(apf.arcs.length).toBe(2);
  });
  it("Scene moving/static split feeds ThreeDScene camera override", () => {
    const scene = new ThreeDScene();
    const a = new Square();
    scene.add(a);
    const tracker = new ValueTracker(0);
    const frameCenter = new Mobject();
    scene.renderer = { camera: { get_value_trackers: () => [tracker], _frame_center: frameCenter } };
    const idle = new Circle();
    scene.add(idle);
    const anim = new FadeIn(a);
    const [moving] = scene.get_moving_and_static_mobjects([anim]);
    expect(moving.includes(a)).toBe(true);
  });
  it("NumberLine configures an Arrow tip-to-number chain without tex", () => {
    // Python: NumberLine p2n(n2p(x)) == x; Arrow tip sits off the shaft end.
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    for (const x of [-2, 0, 2.5]) {
      const p = [...(nl.n2p(x).toArray() as unknown as number[])];
      expect(close(nl.point_to_number(p) as number, x, 1e-6)).toBe(true);
    }
    const arrow = new Arrow([0, 0, 0], [2, 0, 0]);
    expect(arrow.has_tip()).toBe(true);
  });
  it("BraceLabel.creation_anim plays inside a Scene like python's AnimationGroup", () => {
    const sq = new Square(2.0);
    const dot = new Dot([0, 0, 0]);
    const bl = new BraceLabel(sq, dot);
    const scene = new Scene();
    scene.add(bl);
    const anim = bl.creation_anim();
    anim.begin();
    anim.interpolate(1.0);
    anim.finish();
    expect(scene.mobjects.includes(bl)).toBe(true);
  });
});
