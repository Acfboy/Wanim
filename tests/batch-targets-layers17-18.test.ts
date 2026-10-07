// Ground-truth tests for layers 17..18 batch (~1011 lines).
//
// Python ground truths captured from the read-only reference
// (original manim 0.21.0) via the original manim implementation. Each block cites
// the exact Python output. Float comparisons use 1e-9 relative tolerance.
//
// OUT OF SCOPE (text/tex): no Text/Tex/MathTex objects are used anywhere
// here; Matrix tests use MobjectMatrix-style entries with an identity
// factory plus a brackets-patched subclass (LaTeX is unavailable in this
// environment, and TS Matrix._add_brackets throws TODO just like the stub).
// LabeledDot string labels (need MathTex) are asserted to throw TODO.
import { describe, expect, it } from "vitest";
import { Sphere } from "../src/mobject/three_d/three_dimensions.js";
import { Scene } from "../src/scene/scene.js";
import { Vector } from "../src/mobject/geometry/line.js";
import { Arrow, Line } from "../src/mobject/geometry/line.js";
import { GrowArrow } from "../src/animation/growing.js";
import { Matrix, MobjectMatrix, matrix_to_tex_string } from "../src/mobject/matrix.js";
import { MathTex } from "../src/mobject/text/index.js";
import { VectorScene } from "../src/scene/vector_space_scene.js";
import { Dot, LabeledDot } from "../src/mobject/geometry/arc.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { TransformMatchingShapes, TransformMatchingTex } from "../src/animation/transform_matching_parts.js";
import { Succession, AnimationGroup } from "../src/animation/composition.js";
import { Animation } from "../src/animation/animation.js";
import { TypeWithCursor, UntypeWithCursor } from "../src/animation/creation.js";
import { FocusOn, Flash, ShowPassingFlash } from "../src/animation/indication.js";
import { ChangingDecimal } from "../src/animation/numbers.js";
import { Transform, FadeTransform } from "../src/animation/transform.js";
import { FadeIn, FadeOut } from "../src/animation/fading.js";
import { ScreenRectangle, FullScreenRectangle } from "../src/mobject/frame.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { ArrowVectorField, VectorField } from "../src/mobject/vector_field.js";
import { Square, Rectangle } from "../src/mobject/geometry/polygram.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Group, Mobject } from "../src/mobject/mobject.js";
import { EndSceneEarlyException, RerunSceneException } from "../src/utils/exceptions.js";
import { ORIGIN, RIGHT } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1e-12, Math.abs(b));
  return Math.abs(a - b) <= Math.max(1e-12, denom * tol);
}

function expectPointClose(actual: unknown, expected: number[], tol = 1e-9): void {
  const arr = Array.isArray(actual)
    ? (actual as unknown[]).map(Number)
    : Array.from((actual as { toArray(): Iterable<unknown> }).toArray() as Iterable<unknown>).map(Number);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Sphere.__init__
// Python: Sphere(center=[1,2,3], radius=2.0, resolution=(6,3)):
//   radius 2.0, center [1.0, 2.0, 3.0] (up to fp noise), u_range (0, TAU),
//   v_range (0, PI). Default: radius 1, center ORIGIN.
//   func(0,0) == [0,0,-2.0] (radius * [cos0*sin0, sin0*sin0, -cos0]).
// ---------------------------------------------------------------------------
describe("Sphere.__init__", () => {
  it("stores radius and shifts to center", () => {
    const s = new Sphere([1, 2, 3], 2.0, [6, 3] as [number, number]);
    expect(s.radius).toBe(2.0);
    expectPointClose(s.get_center(), [1, 2, 3], 1e-6);
    expect(close(s.u_range[0] ?? NaN, 0)).toBe(true);
    expect(close(s.u_range[1] ?? NaN, Math.PI * 2)).toBe(true);
    expect(close(s.v_range[0] ?? NaN, 0)).toBe(true);
    expect(close(s.v_range[1] ?? NaN, Math.PI)).toBe(true);
  });

  it("parametric func matches Python radius*[cos u sin v, sin u sin v, -cos v]", () => {
    const s = new Sphere([0, 0, 0], 2.0, [6, 3] as [number, number]);
    // Python: func(0,0) == [0.0, 0.0, -2.0]
    expectPointClose(s.func(0, 0), [0, 0, -2.0]);
    // Python: func(pi, pi/2) == [-2, 0, 0]-ish
    expectPointClose(s.func(Math.PI, Math.PI / 2), [-2, 0, 0], 1e-6);
  });

  it("defaults to radius 1 at origin", () => {
    const s = new Sphere();
    expect(s.radius).toBe(1);
    expectPointClose(s.get_center(), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Vector
// Python: Vector([3,4]): start [0,0,0], end [3,4,0] (2-D extended with 0).
//   Vector([1,0,0], buff=0.1).buff == 0.1.
// ---------------------------------------------------------------------------
describe("Vector", () => {
  it("extends 2-D directions to 3-D from ORIGIN", () => {
    const v = new Vector([3, 4]);
    expectPointClose(v.get_start(), [0, 0, 0]);
    expectPointClose(v.get_end(), [3, 4, 0]);
  });

  it("stores buff", () => {
    // Python: Vector([1,0,0], buff=0.1): start [0.1,0,0], end [0.9,0,0]
    // (buff shortens both ends along the direction).
    const v = new Vector([1, 0, 0], 0.1);
    expect(v.buff).toBeCloseTo(0.1, 12);
    expectPointClose(v.get_start(), [0.1, 0, 0]);
    expectPointClose(v.get_end(), [0.9, 0, 0]);
  });

  it("defaults to RIGHT with buff 0", () => {
    const v = new Vector();
    expect(v.buff).toBe(0);
    expectPointClose(v.get_end(), [1, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.GrowArrow.__init__
// Python: Arrow([0,0,0],[2,0,0]): point == [0.25, 0, 0] (arrow start after
//   buff/shift: get_start, not ORIGIN), introducer True; starting mobject
//   centered at point with width 0.
// ---------------------------------------------------------------------------
describe("GrowArrow.__init__", () => {
  it("grows from the arrow start point", () => {
    const a = new Arrow([0, 0, 0], [2, 0, 0]);
    const g = new GrowArrow(a as unknown as Mobject & { get_start(): number[] });
    const pt = Array.isArray(g.point)
      ? [...(g.point as number[])]
      : [...(g.point as Mobject).get_center()];
    expectPointClose(pt, (a as Arrow).get_start(), 1e-6);
    expect(g.introducer).toBe(true);
  });

  it("starting mobject is scaled to zero about the start point", () => {
    const a = new Arrow([0, 0, 0], [2, 0, 0]);
    const g = new GrowArrow(a as unknown as Mobject & { get_start(): number[] });
    const start = g.create_starting_mobject();
    expectPointClose(start.get_center(), (a as Arrow).get_start(), 1e-6);
    expect(close(start.width, 0, 1e-6)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.matrix.Matrix
// Python (MobjectMatrix, brackets patched out — LaTeX missing here):
//   matrix_to_tex_string([[1,2],[3,4]]) ==
//     '\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]'
//   1-D input becomes a single column.
//   MobjectMatrix 2x2: v_buff 0.8, h_buff 1.3, entries 4, rows 2, cols 2.
//   DEVIATION: the port's grid is content-aware, so 2-wide squares get a pitch
//   of max(1.3, 2 + 0.25) = max(0.8, 2 + 0.25) = 2.25 instead of upstream's
//   fixed 1.3/0.8; centers below are the 2.25-pitch ones.
// ---------------------------------------------------------------------------
class NoBracketMatrix extends Matrix {
  override _add_brackets(): this {
    this.brackets = new VGroup();
    return this;
  }
}

describe("matrix_to_tex_string", () => {
  it("formats a 2x2 matrix", () => {
    expect(matrix_to_tex_string([[1, 2], [3, 4]])).toBe(
      "\\left[ \\begin{array}{cc}1 & 2 \\\\ 3 & 4\\end{array} \\right]",
    );
  });

  it("reshapes 1-D input to a single column", () => {
    // Python: matrix_to_tex_string([1,2,3]) ==
    //   '\\left[ \\begin{array}{c}1 \\\\ 2 \\\\ 3\\end{array} \\right]'
    expect(matrix_to_tex_string([1, 2, 3])).toBe(
      "\\left[ \\begin{array}{c}1 \\\\ 2 \\\\ 3\\end{array} \\right]",
    );
  });
});

describe("Matrix (Mobject entries, brackets patched)", () => {
  function make2x2(): NoBracketMatrix {
    const entry = (): VGroup => new VGroup(new Square());
    return new NoBracketMatrix(
      [[0, 1], [2, 3]],
      { element_to_mobject: (() => new Square()) as unknown as Matrix["element_to_mobject"] },
    );
  }

  it("organizes entries on the v_buff/h_buff grid", () => {
    const m = make2x2();
    expect(m.v_buff).toBe(0.8);
    expect(m.h_buff).toBe(1.3);
    expect(m.get_entries().submobjects.length).toBe(4);
    expect(m.get_rows().submobjects.length).toBe(2);
    expect(m.get_columns().submobjects.length).toBe(2);
    // DEVIATION: content-aware pitch (2.25) for the 2x2 Squares; the centered
    // centers are (-1.125, 1.125), (1.125, 1.125), (-1.125, -1.125),
    // (1.125, -1.125).
    expectPointClose(m.mob_matrix[0]![0]!.get_center(), [-1.125, 1.125, 0], 1e-6);
    expectPointClose(m.mob_matrix[0]![1]!.get_center(), [1.125, 1.125, 0], 1e-6);
    expectPointClose(m.mob_matrix[1]![0]!.get_center(), [-1.125, -1.125, 0], 1e-6);
    expectPointClose(m.mob_matrix[1]![1]!.get_center(), [1.125, -1.125, 0], 1e-6);
  });

  it("_add_brackets without patch builds MathTex brackets", () => {
    const m = new MobjectMatrix([[new Square()]], {
      element_to_mobject: ((x: unknown) => x) as unknown as Matrix["element_to_mobject"],
    });
    expect(m.get_brackets().submobjects.length).toBe(2);
    expect(m.get_entries().submobjects.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.vector_space_scene.VectorScene.position_x/y_coordinate
// Python: position_x_coordinate does next_to(x_line, -sign(vector[1])*UP)
//   + set_color(X_COLOR); position_y_coordinate does next_to(y_line,
//   sign(vector[0])*RIGHT) + set_color(Y_COLOR).
//   With Square() (side 2) and unit lines, vector [2,3,0]:
//   x center [0, -1.25, 0] color #83C167 (GREEN_C); y center [1.25, 0, 0].
// ---------------------------------------------------------------------------
describe("VectorScene.position_x/y_coordinate", () => {
  it("places the x label below the line for positive y", () => {
    const vs = new VectorScene();
    const xCoord = new Square();
    const xLine = new Line([-1, 0, 0], [1, 0, 0]);
    const out = vs.position_x_coordinate(xCoord, xLine, [2, 3, 0]);
    expect(out).toBe(xCoord);
    expectPointClose(xCoord.get_center(), [0, -1.25, 0], 1e-6);
    expect(xCoord.color.toHex().toUpperCase()).toBe("#83C167");
  });

  it("places the y label to the right of the line for positive x", () => {
    const vs = new VectorScene();
    const yCoord = new Square();
    const yLine = new Line([0, -1, 0], [0, 1, 0]);
    const out = vs.position_y_coordinate(yCoord, yLine, [2, 3, 0]);
    expect(out).toBe(yCoord);
    expectPointClose(yCoord.get_center(), [1.25, 0, 0], 1e-6);
  });

  it("flips sides for negative components", () => {
    const vs = new VectorScene();
    const xCoord = new Square();
    vs.position_x_coordinate(xCoord, new Line([-1, 0, 0], [1, 0, 0]), [2, -3, 0]);
    expectPointClose(xCoord.get_center(), [0, 1.25, 0], 1e-6);
    const yCoord = new Square();
    vs.position_y_coordinate(yCoord, new Line([0, -1, 0], [0, 1, 0]), [-2, 3, 0]);
    expectPointClose(yCoord.get_center(), [-1.25, 0, 0], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.Dot / LabeledDot
// Python: Dot(point=[1,2,0], radius=0.5): center [1,2,0], width 1.0,
//   fill_opacity 1.0, stroke_width 0.
//   LabeledDot(Square(side 1), radius=None): width ~1.6142 (buff + norm/2 * 2),
//   label recentered on the dot; explicit radius 0.7 -> width 1.4.
//   LabeledDot("x") needs MathTex -> TODO.
// ---------------------------------------------------------------------------
describe("Dot", () => {
  it("places a filled circle at the point", () => {
    const d = new Dot([1, 2, 0], 0.5);
    expectPointClose(d.get_center(), [1, 2, 0]);
    expect(close(d.width, 1.0)).toBe(true);
    expect(d.fill_opacity).toBe(1.0);
    expect(d.stroke_width).toBe(0);
  });
});

describe("LabeledDot", () => {
  it("derives radius from the label size when radius is null", () => {
    // Python: LabeledDot(Square(side 1)) width 1.614213562373095
    const label = new Square(1.0);
    const ld = new LabeledDot(label as unknown as Mobject, null);
    expect(close(ld.width, 1.614213562373095, 1e-6)).toBe(true);
    expect(ld.submobjects.length).toBe(1);
    expectPointClose(label.get_center(), ld.get_center(), 1e-6);
  });

  it("honors an explicit radius", () => {
    const label = new Square(1.0);
    const ld = new LabeledDot(label as unknown as Mobject, 0.7);
    expect(close(ld.width, 1.4)).toBe(true);
  });

  it("string labels build a MathTex label", () => {
    const ld = new LabeledDot("x");
    expect(ld.submobjects.length).toBe(1);
    expect(ld.submobjects[0]).toBeInstanceOf(MathTex);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts
// Python: TransformMatchingShapes(Circle(), Circle()): 3 animations
//   (Transform + FadeOut + FadeIn), to_add is the target.
//   TransformMatchingTex.get_mobject_key asserts MathTexPart (tex_string).
// ---------------------------------------------------------------------------
describe("TransformMatchingShapes.__init__", () => {
  it("builds Transform + FadeOut + FadeIn for disjoint shapes", () => {
    const c1 = new Circle();
    const c2 = new Circle().shift([1, 0, 0]);
    const tm = new TransformMatchingShapes(c1, c2);
    expect(tm.animations.length).toBe(3);
    expect(tm.animations[0]).toBeInstanceOf(Transform);
    expect(tm.to_add).toBe(c2 as unknown as Mobject);
    expect(tm.to_remove).toContain(c1 as unknown as Mobject);
  });
});

describe("TransformMatchingTex.__init__", () => {
  function texPart(s: string): Mobject {
    const g = new Group();
    (g as unknown as Record<string, unknown>)["tex_string"] = s;
    const sub = new Square();
    (sub as unknown as Record<string, unknown>)["tex_string"] = s;
    g.add(sub);
    return g;
  }

  it("matches parts by tex_string", () => {
    const a = texPart("a");
    const b = texPart("a");
    const tm = new TransformMatchingTex(a, b);
    expect(tm.to_add).toBe(b);
    expect(TransformMatchingTex.get_mobject_key(b.submobjects[0]!)).toBe("a");
  });

  it("rejects mobjects without tex_string", () => {
    expect(() => TransformMatchingTex.get_mobject_parts(new Square())).toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.animation.composition.Succession.begin / next_animation
// Python (succ.scene = None set manually; begin() without a scene attr
//   raises AttributeError — TS initializes scene to null instead):
//   timings [(0,1),(1,2)], begin selects animations[0], next advances,
//   interpolate(0.5) crosses into index 1, interpolate(1.0) exhausts.
// ---------------------------------------------------------------------------
describe("Succession.begin / next_animation", () => {
  function makeSuccession(): Succession {
    const s1 = new Square();
    const s2 = new Square();
    return new Succession(new FadeIn(s1), new FadeOut(s2));
  }

  it("begin activates the first animation", () => {
    const succ = makeSuccession();
    succ.begin();
    expect(succ.active_index).toBe(0);
    expect(succ.active_animation).toBe(succ.animations[0]);
  });

  it("next_animation finishes current and advances", () => {
    const succ = makeSuccession();
    succ.begin();
    succ.next_animation();
    expect(succ.active_index).toBe(1);
    expect(succ.active_animation).toBe(succ.animations[1]);
  });

  it("interpolate walks the schedule", () => {
    const succ = makeSuccession();
    succ.begin();
    succ.interpolate(0.25);
    expect(succ.active_index).toBe(0);
    succ.interpolate(0.5);
    expect(succ.active_index).toBe(1);
    succ.interpolate(1.0);
    expect(succ.active_animation).toBeNull();
  });

  it("empty succession begin raises ValueError", () => {
    const err = (() => {
      try {
        new Succession().begin();
      } catch (e) {
        return e as Error;
      }
      return null;
    })();
    expect(err?.name).toBe("ValueError");
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.interpolate
// Python: Animation(Square()).begin() snapshots starting_mobject;
//   interpolate(0.5) runs without error and leaves the mobject in place
//   (base interpolate_submobject is a no-op).
// ---------------------------------------------------------------------------
describe("Animation.interpolate", () => {
  it("zips families and keeps a no-op base submobject in place", () => {
    const m = new Square();
    const before = [...m.get_center()];
    const an = new Animation(m);
    an.begin();
    expect(an.starting_mobject).toBeDefined();
    an.interpolate(0.5);
    expectPointClose(m.get_center(), before);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.TypeWithCursor / UntypeWithCursor
// Python: TypeWithCursor defaults buff 0.1, keep_cursor_y True,
//   leave_cursor_on True, introducer True.
//   UntypeWithCursor: reverse_rate_function True, remover True,
//   introducer False. clean_up removes cursor only when leave_cursor_on
//   is False; update_submobject_list reveals prefix + parks cursor.
// ---------------------------------------------------------------------------
describe("TypeWithCursor", () => {
  function makePair(): { text: VGroup; cursor: Square } {
    const text = new VGroup(new Square(), new Square(), new Square());
    const cursor = new Square(0.2);
    return { text: text as unknown as VGroup, cursor };
  }

  it("carries the Python defaults", () => {
    const { text, cursor } = makePair();
    const tw = new TypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject);
    expect(tw.buff).toBeCloseTo(0.1, 12);
    expect(tw.keep_cursor_y).toBe(true);
    expect(tw.leave_cursor_on).toBe(true);
  });

  it("clean_up_from_scene removes the cursor only when not left on", () => {
    const scene = new Scene();
    const { text, cursor } = makePair();
    const tw = new TypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject, {
      leave_cursor_on: false,
    });
    scene.add(cursor as unknown as Mobject);
    tw.clean_up_from_scene(scene);
    expect(scene.mobjects).not.toContain(cursor as unknown as Mobject);
  });

  it("clean_up_from_scene keeps the cursor when left on", () => {
    const scene = new Scene();
    const { text, cursor } = makePair();
    const tw = new TypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject, {
      leave_cursor_on: true,
    });
    scene.add(cursor as unknown as Mobject);
    tw.clean_up_from_scene(scene);
    expect(scene.mobjects).toContain(cursor as unknown as Mobject);
  });

  it("update_submobject_list reveals the prefix and parks the cursor", () => {
    // Python ground truth (manim 0.21.0): VGroup(3 x Square()) + Square(0.2)
    // cursor, update_submobject_list(2) -> cursor center [1.2, 0, 0]
    // (sub right edge 1.0 + buff 0.1 + cursor half-width 0.1); index 0 parks
    // the cursor on the first submobject at [0, 0, 0].
    const { text, cursor } = makePair();
    const tw = new TypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject);
    tw.begin();
    tw.update_submobject_list(2);
    const subs = (tw as unknown as { all_submobs: Mobject[] }).all_submobs;
    expect(subs.length).toBe(3);
    expectPointClose((cursor as unknown as Mobject).get_center(), [1.2, 0, 0], 1e-6);
  });

  it("update_submobject_list parks the cursor on the first submobject at index 0", () => {
    // Python: update_submobject_list(0) -> cursor at [0, 0, 0]; index 1 ->
    // [1.2, 0, 0].
    const { text, cursor } = makePair();
    const tw = new TypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject);
    tw.begin();
    tw.update_submobject_list(0);
    expectPointClose((cursor as unknown as Mobject).get_center(), [0, 0, 0], 1e-6);
    tw.update_submobject_list(1);
    expectPointClose((cursor as unknown as Mobject).get_center(), [1.2, 0, 0], 1e-6);
  });
});

describe("UntypeWithCursor.__init__", () => {
  it("reverses with remover semantics", () => {
    const text = new VGroup(new Square(), new Square());
    const cursor = new Square(0.2);
    const ut = new UntypeWithCursor(text as unknown as Mobject, cursor as unknown as Mobject);
    expect(ut.reverse_rate_function).toBe(true);
    expect(ut.remover).toBe(true);
    expect(ut.introducer).toBe(false);
  });

  it("accepts Python's default cursor=None", () => {
    // Python: UntypeWithCursor(text, cursor=None) stores None.
    const text = new VGroup(new Square(), new Square());
    const ut = new UntypeWithCursor(text as unknown as Mobject);
    expect((ut as unknown as { cursor: unknown }).cursor).toBeNull();
    expect(ut.reverse_rate_function).toBe(true);
    expect(ut.remover).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.FocusOn / Flash / ShowPassingFlash
// Python: FocusOn([1,2,0]): focus_point [1,2,0], opacity 0.2, remover True,
//   run_time 2; create_target is a zero-width Dot.
//   Flash([0,0,0]): 12 lines, 12 ShowPassingFlash children.
//   Flash([1,0,0], num_lines=4): line 0 from [1.1,0,0] to [1.3,0,0].
//   ShowPassingFlash._get_bounds: 0 -> (0,0), 0.5 -> (0.4,0.6) (tw=0.2),
//   1 -> (1,1).
// ---------------------------------------------------------------------------
describe("FocusOn.__init__ / create_target", () => {
  it("stores focus parameters with remover semantics", () => {
    const f = new FocusOn([1, 2, 0]);
    expectPointClose(f.focus_point as number[], [1, 2, 0]);
    expect(f.opacity).toBeCloseTo(0.2, 12);
    expect(f.run_time).toBe(2);
    expect(f.remover).toBe(true);
  });

  it("create_target is a zero-width dot", () => {
    const f = new FocusOn([1, 2, 0]);
    const target = f.create_target();
    expect(close(target.width, 0)).toBe(true);
  });
});

describe("Flash.__init__", () => {
  it("builds one ShowPassingFlash per radial line", () => {
    const fl = new Flash([0, 0, 0]);
    expect(fl.num_lines).toBe(12);
    expect(fl.lines.submobjects.length).toBe(12);
    expect(fl.animations.length).toBe(12);
    expect(fl.animations[0]).toBeInstanceOf(ShowPassingFlash);
    expect(close(fl.line_length, 0.2)).toBe(true);
    expect(close(fl.flash_radius, 0.1)).toBe(true);
  });

  it("places radial lines around the point", () => {
    // Python: Flash([1,0,0], num_lines=4) line 0: [1.1,0,0] -> [1.3,0,0].
    const fl = new Flash([1, 0, 0], 0.2, 4, 0.1);
    const first = fl.lines.submobjects[0] as Line;
    expectPointClose(first.get_start(), [1.1, 0, 0], 1e-6);
    expectPointClose(first.get_end(), [1.3, 0, 0], 1e-6);
  });
});

describe("ShowPassingFlash.clean_up_from_scene", () => {
  it("restores the full curve after playing", () => {
    const line = new Line([0, 0, 0], [1, 0, 0]);
    const scene = new Scene();
    scene.add(line);
    const anim = new ShowPassingFlash(line as unknown as Parameters<typeof ShowPassingFlash>[0], 0.2);
    anim.begin();
    anim.interpolate(0.5);
    anim.clean_up_from_scene(scene);
    // Full-length curve restored: end-to-end distance back to 1.
    const start = (line as Line).get_start();
    const end = (line as Line).get_end();
    const dist = Math.hypot(end[0]! - start[0]!, end[1]! - start[1]!);
    expect(close(dist, 1, 1e-6)).toBe(true);
  });

  it("_get_bounds matches Python", () => {
    const anim = new ShowPassingFlash(
      new Line([0, 0, 0], [1, 0, 0]) as unknown as Parameters<typeof ShowPassingFlash>[0],
      0.2,
    );
    expect(anim._get_bounds(0)).toEqual([0, 0]);
    const [lo, hi] = anim._get_bounds(0.5);
    expect(close(lo, 0.4)).toBe(true);
    expect(close(hi, 0.6)).toBe(true);
    expect(anim._get_bounds(1)).toEqual([1, 1]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.numbers.ChangingDecimal.check_validity_of_input
// Python: TypeError("ChangingDecimal can only take in a DecimalNumber") for
//   non-decimals; valid decimals interpolate via number_update_func.
// ---------------------------------------------------------------------------
describe("ChangingDecimal.check_validity_of_input", () => {
  it("rejects mobjects without set_value", () => {
    // Python: ChangingDecimal(Square(), ...) raises
    // TypeError("ChangingDecimal can only take in a DecimalNumber").
    // (The structural set_value check runs before the Mobject typecheck.)
    expect(
      () => new ChangingDecimal(new Square() as unknown as Mobject, (a) => a),
    ).toThrow("ChangingDecimal can only take in a DecimalNumber");
  });

  it("check_validity_of_input method rejects too", () => {
    const fake = new Square() as unknown as Mobject & {
      set_value(v: number): void;
    };
    fake.set_value = (): void => {};
    const anim = new ChangingDecimal(fake as unknown as Mobject, (a) => a);
    expect(() => anim.check_validity_of_input(new Square() as unknown as Mobject)).toThrow(
      "ChangingDecimal can only take in a DecimalNumber",
    );
  });

  it("drives set_value through interpolate_mobject", () => {
    let current = 5;
    const fake = new Square() as unknown as Mobject & {
      number: number;
      set_value(v: number): void;
    };
    fake.set_value = (v: number): void => {
      current = v;
    };
    const anim = new ChangingDecimal(fake as unknown as Mobject, (a) => a * 10);
    anim.interpolate_mobject(0.5);
    expect(close(current, 5, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Transform.clean_up_from_scene /
//   FadeTransform.clean_up_from_scene
// Python Transform.clean_up: replaces mobject with target only when
//   replace_mobject_with_target_in_scene is set (default False).
// Python FadeTransform.clean_up: Animation.clean_up + remove group +
//   restore mobject[0] + add to_add_on_completion.
// ---------------------------------------------------------------------------
describe("Transform.clean_up_from_scene", () => {
  it("leaves the scene alone by default", () => {
    const s1 = new Square();
    const s2 = new Square().shift([2, 0, 0]);
    const scene = new Scene();
    scene.add(s1 as unknown as Mobject);
    const t = new Transform(s1 as unknown as Mobject, s2 as unknown as Mobject);
    t.begin();
    t.clean_up_from_scene(scene);
    expect(scene.mobjects).toContain(s1 as unknown as Mobject);
  });

  it("interpolates halfway between source and target", () => {
    const s1 = new Square();
    const s2 = new Square().shift([2, 0, 0]);
    const t = new Transform(s1 as unknown as Mobject, s2 as unknown as Mobject);
    t.begin();
    t.interpolate(0.5);
    expectPointClose((s1 as unknown as Mobject).get_center(), [1, 0, 0], 1e-6);
  });
});

describe("FadeTransform.clean_up_from_scene", () => {
  it("swaps the group for the completion target", () => {
    const s1 = new Square();
    const target = new Square().shift([1, 0, 0]);
    const scene = new Scene();
    const ft = new FadeTransform(s1 as unknown as Mobject, target as unknown as Mobject);
    scene.add(ft.mobject);
    ft.begin();
    ft.interpolate(1);
    ft.clean_up_from_scene(scene);
    expect(scene.mobjects).toContain(ft.to_add_on_completion);
    expect(scene.mobjects).not.toContain(ft.mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.render / remove_foreground_mobject / bring_to_back
// Python render(): setup + construct + tear_down + scene_finished, False
//   normally, True on RerunSceneException. remove_foreground_mobjects
//   restructures foreground_mobjects; bring_to_back prepends to mobjects.
// ---------------------------------------------------------------------------
describe("Scene.remove_foreground_mobject / bring_to_back", () => {
  it("removes a single foreground mobject", () => {
    const scene = new Scene();
    const a = new Mobject();
    const b = new Mobject();
    scene.add_foreground_mobjects(a, b);
    scene.remove_foreground_mobject(a);
    expect(scene.foreground_mobjects).toEqual([b]);
  });

  it("bring_to_back prepends in order", () => {
    // Python: self.remove(*mobjects); self.mobjects = list(mobjects) + ...
    const scene = new Scene();
    const a = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    scene.add(a, b, c);
    scene.bring_to_back(c);
    expect(scene.mobjects).toEqual([c, a, b]);
  });
});

describe("Scene.render", () => {
  function stubRenderer(): {
    num_plays: number;
    finished: boolean;
    renderer: Record<string, unknown>;
  } {
    const state = { num_plays: 0, finished: false, renderer: {} as Record<string, unknown> };
    state.renderer = {
      num_plays: 0,
      clear_screen(): void {},
      scene_finished(): void {
        state.finished = true;
      },
      file_writer: { abort_encode_jobs(): void {} },
    };
    return state;
  }

  it("runs construct and finishes, returning False", () => {
    const state = stubRenderer();
    let constructed = false;
    class DemoScene extends Scene {
      override construct(): void {
        constructed = true;
      }
    }
    const scene = new DemoScene();
    scene.renderer = state.renderer;
    expect(scene.render()).toBe(false);
    expect(constructed).toBe(true);
    expect(state.finished).toBe(true);
  });

  it("returns True on RerunSceneException after clearing", () => {
    const state = stubRenderer();
    let aborted = false;
    (state.renderer.file_writer as Record<string, () => void>).abort_encode_jobs = () => {
      aborted = true;
    };
    class RerunScene extends Scene {
      override construct(): void {
        throw new RerunSceneException("rerun");
      }
    }
    const scene = new RerunScene();
    scene.renderer = state.renderer;
    expect(scene.render()).toBe(true);
    expect(aborted).toBe(true);
  });

  it("survives EndSceneEarlyException and returns False", () => {
    const state = stubRenderer();
    class EarlyScene extends Scene {
      override construct(): void {
        throw new EndSceneEarlyException("early");
      }
    }
    const scene = new EarlyScene();
    scene.renderer = state.renderer;
    expect(scene.render()).toBe(false);
    expect(state.finished).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.frame.ScreenRectangle / FullScreenRectangle
// Python: ScreenRectangle(aspect_ratio=2.0, height=3): width 6, height 3.
//   FullScreenRectangle height == config.frame_height (8.0 default).
// ---------------------------------------------------------------------------
describe("ScreenRectangle.__init__ / FullScreenRectangle", () => {
  it("sizes from aspect ratio and height", () => {
    const sr = new ScreenRectangle(2.0, 3);
    expect(close(sr.width, 6.0)).toBe(true);
    expect(close(sr.height, 3.0)).toBe(true);
    expect(close(sr.aspect_ratio, 2.0)).toBe(true);
  });

  it("FullScreenRectangle spans the configured frame height", () => {
    const fsr = new FullScreenRectangle();
    // Python: FullScreenRectangle().height == 8.0 (default config).
    expect(close(fsr.height, 8.0)).toBe(true);
    expect(close(fsr.aspect_ratio, 16 / 9, 1e-6)).toBe(true);
  });

  it("FullScreenRectangle forwards aspect_ratio kwargs like Python", () => {
    // Python: FullScreenRectangle(aspect_ratio=2.0) keeps ratio 2.0 after the
    // frame-height rescale (uniform scale preserves the ratio).
    const fsr = new FullScreenRectangle({ aspect_ratio: 2.0 });
    expect(close(fsr.aspect_ratio, 2.0, 1e-6)).toBe(true);
    expect(close(fsr.height, 8.0)).toBe(true);
    expect(close(fsr.width, 16.0, 1e-6)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.camera.moving_camera.MovingCamera
// Python: get_mobjects_indicating_movement() == [frame]; _get_bounding_box
//   over two shifted Squares -> (-3, 4, 2, -2); skips the frame itself;
//   empty input raises.
// ---------------------------------------------------------------------------
describe("MovingCamera.get_mobjects_indicating_movement / _get_bounding_box", () => {
  it("reports the frame as the movement indicator", () => {
    const mc = new MovingCamera();
    expect(mc.get_mobjects_indicating_movement()).toEqual([mc.frame]);
  });

  it("computes the union bounding box", () => {
    // Python: bbox == (-3.0, 4.0, 2.0, -2.0) for Squares at [3,1,0], [-2,-1,0].
    const mc = new MovingCamera();
    const s1 = new Square().shift([3, 1, 0]);
    const s2 = new Square().shift([-2, -1, 0]);
    expect(mc._get_bounding_box([s1, s2] as Mobject[], false)).toEqual([-3, 4, 2, -2]);
  });

  it("skips the camera frame", () => {
    const mc = new MovingCamera();
    const s1 = new Square().shift([3, 1, 0]);
    // Python: bbox over [frame, s1] equals bbox over [s1] alone.
    expect(mc._get_bounding_box([mc.frame, s1] as Mobject[], false)).toEqual(
      mc._get_bounding_box([s1] as Mobject[], false),
    );
  });

  it("raises on empty input", () => {
    const mc = new MovingCamera();
    expect(() => mc._get_bounding_box([], false)).toThrow(/bounding box/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.VectorField.fit_to_coordinate_system
// Python: apply_function(lambda pos: coords_to_point(*pos)); returns self.
// ---------------------------------------------------------------------------
describe("VectorField.fit_to_coordinate_system", () => {
  it("returns self on a base field (Python: apply_function over no vectors)", () => {
    // Python ground truth: base VectorField holds no submobjects, so
    // fit_to_coordinate_system is a no-op returning self.
    const vf = new VectorField((p: number[]) => p);
    const fake = {
      coords_to_point: (...pos: number[]): number[] => pos.map((v) => v * 2),
    };
    const out = vf.fit_to_coordinate_system(
      fake as unknown as Parameters<VectorField["fit_to_coordinate_system"]>[0],
    );
    expect(out).toBe(vf);
  });

  it("spreads position components and moves arrows (ArrowVectorField)", () => {
    // Python ground truth (manim 0.21.0): ArrowVectorField(lambda p: [1,0,0],
    // x_range=[0,1], y_range=[0,1]) has 9 vectors; fitting through a doubling
    // map calls coords_to_point with 3 spread scalars per point and doubles
    // arrow centers, e.g. [0.16448818019175107, 0, 0] -> [0.32897636038350214,
    // 0, 0]; returns self.
    const calls: number[][] = [];
    const fake = {
      // NOTE: mirrors Python `coords_to_point(*pos)` — components arrive as
      // separate arguments, so the fake doubles each component.
      coords_to_point: (...pos: number[]): number[] => {
        calls.push([...pos]);
        return pos.map((v) => v * 2);
      },
    };
    const vf = new ArrowVectorField((p: number[]) => [1, 0, 0], {
      x_range: [0, 1],
      y_range: [0, 1],
    });
    expect(vf.submobjects.length).toBe(9);
    const before = vf.submobjects.map((v) => [...(v as Mobject).get_center()]);
    const out = vf.fit_to_coordinate_system(
      fake as unknown as Parameters<VectorField["fit_to_coordinate_system"]>[0],
    );
    expect(out).toBe(vf);
    // Positions were spread (3 numeric components per call, never a nested
    // array) — this distinguishes `coords_to_point(*pos)` from
    // `coords_to_point(pos)`.
    expect(calls.length).toBeGreaterThan(0);
    for (const args of calls) {
      expect(args.length).toBe(3);
      for (const v of args) {
        expect(typeof v).toBe("number");
      }
    }
    const after = vf.submobjects.map((v) => [...(v as Mobject).get_center()]);
    expectPointClose(after[0]!, [0.32897636038350214, 0, 0], 1e-6);
    for (let i = 0; i < before.length; i += 1) {
      expectPointClose(
        after[i]!,
        (before[i] as number[]).map((v) => v * 2),
        1e-6,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols interacting with earlier-ported code.
// ---------------------------------------------------------------------------
describe("integration (layers 17..18 with earlier batches)", () => {
  it("Vector direction feeds VectorScene y-coordinate placement", () => {
    // Python: Vector([3,4]) end [3,4,0]; position_y with vector [3,4,0]
    // puts the label at +x (sign(3) * RIGHT).
    const vs = new VectorScene();
    const v = new Vector([3, 4]);
    const label = new Square();
    vs.position_y_coordinate(
      label,
      new Line([0, -1, 0], [0, 1, 0]),
      [...v.get_end()],
    );
    expectPointClose(label.get_center(), [1.25, 0, 0], 1e-6);
  });

  it("Succession of FadeIn/FadeOut runs against Scene membership", () => {
    const scene = new Scene();
    const s1 = new Square();
    const s2 = new Square();
    const succ = new Succession(new FadeIn(s1 as unknown as Mobject), new FadeOut(s2 as unknown as Mobject));
    succ._setup_scene(scene);
    succ.begin();
    succ.interpolate(0.25);
    expect(succ.active_index).toBe(0);
    succ.interpolate(1.0);
    expect(succ.active_animation).toBeNull();
  });

  it("MobjectMatrix entries accept Scene bring_to_back ordering", () => {
    const m = new NoBracketMatrix(
      [[0, 1], [2, 3]],
      { element_to_mobject: (() => new Square()) as unknown as Matrix["element_to_mobject"] },
    );
    const scene = new Scene();
    const entries = m.get_entries().submobjects as Mobject[];
    scene.add(...entries);
    scene.bring_to_back(entries[3]!);
    expect(scene.mobjects[0]).toBe(entries[3]);
  });

  it("Flash lines survive a MovingCamera bounding-box query", () => {
    // Python ground truth: Flash([0,0,0], 0.2, 4, 0.1) lines span
    // [-0.3, 0.3] on both axes (flash_radius 0.1 + line_length 0.2).
    const mc = new MovingCamera();
    const fl = new Flash([0, 0, 0], 0.2, 4, 0.1);
    const box = mc._get_bounding_box(fl.lines.submobjects as Mobject[], false);
    expect(box.length).toBe(4);
    expect(close(box[0]!, -0.3, 1e-6)).toBe(true);
    expect(close(box[1]!, 0.3, 1e-6)).toBe(true);
    expect(close(box[2]!, 0.3, 1e-6)).toBe(true);
    expect(close(box[3]!, -0.3, 1e-6)).toBe(true);
  });

  it("TransformMatchingShapes + FadeTransform share the Group replace path", () => {
    const c1 = new Circle();
    const c2 = new Circle().shift([1, 0, 0]);
    const tm = new TransformMatchingShapes(c1 as unknown as Mobject, c2 as unknown as Mobject);
    expect(tm.animations[0]).toBeInstanceOf(Transform);
    const ft = new FadeTransform(c1 as unknown as Mobject, c2 as unknown as Mobject);
    expect(ft.stretch).toBe(true);
    expect(ft.dim_to_match).toBe(1);
  });
});
