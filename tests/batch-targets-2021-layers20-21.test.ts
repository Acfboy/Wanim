import { describe, expect, it } from "vitest";
import { config } from "../src/_config/utils.js";
import { Line, Angle, RightAngle } from "../src/mobject/geometry/line.js";
import {
  ArrowTriangleFilledTip,
  ArrowSquareFilledTip,
} from "../src/mobject/geometry/tips.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { handle_caching_play } from "../src/utils/caching.js";
import { ImageMobjectFromCamera } from "../src/mobject/types/image_mobject.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Group } from "../src/mobject/mobject.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { DecimalNumber } from "../src/mobject/text/numbers.js";
import { ImplicitFunction } from "../src/mobject/graphing/functions.js";
import { Scene } from "../src/scene/scene.js";
import {
  NumberPlane,
  Axes,
  ThreeDAxes,
  ComplexPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { TransformMatchingShapes } from "../src/animation/transform_matching_parts.js";
import { Write } from "../src/animation/creation.js";
import { Create } from "../src/animation/creation.js";
import {
  ApplyPointwiseFunctionToCenter,
  Restore,
  FadeTransformPieces,
} from "../src/animation/transform.js";
import { turn_animation_into_updater } from "../src/animation/updaters/mobject_update_utils.js";
import { MobjectTable } from "../src/mobject/table.js";
import { ManimBanner } from "../src/mobject/logo.js";
import { SampleSpace } from "../src/mobject/graphing/probability.js";
import { Line3D, Cube, Prism } from "../src/mobject/three_d/three_dimensions.js";
import { Wait } from "../src/animation/animation.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1e-12, Math.abs(b));
  return Math.abs(a - b) <= Math.max(1e-12, denom * tol);
}

function toRows(v: unknown): number[][] {
  const arr = Array.isArray(v)
    ? (v as unknown[])
    : (((v as { toArray(): unknown }).toArray() as unknown[]) ?? []);
  if (arr.length > 0 && typeof arr[0] === "number")
    return [(arr as unknown[]).map(Number)];
  return (arr as unknown[][]).map((r) =>
    Array.isArray(r)
      ? r.map(Number)
      : r !== null && typeof r === "object" && "toArray" in (r as object)
        ? ((r as { toArray(): unknown }).toArray() as unknown[]).map(Number)
        : [Number(r)],
  );
}

function expectRowsClose(actual: unknown, expected: number[][], tol = 1e-9): void {
  const expectedRows = (Array.isArray(expected)
    ? (expected as unknown[])
    : ((expected as { toArray(): unknown }).toArray() as unknown[])
  ).map((row) =>
    Array.isArray(row) ? row : ((row as { toArray(): unknown }).toArray() as number[]),
  ) as number[][];
  const rows = toRows(actual);
  expect(rows.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]?.length).toBe(expectedRows[i]?.length);
    for (let j = 0; j < (expectedRows[i]?.length ?? 0); j += 1) {
      expect(close(rows[i]?.[j] ?? NaN, expectedRows[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.set_path_arc
// original manim: Line([0,0,0],[1,0,0]).set_path_arc(0.5) -> path_arc 0.5,
//   start [0,0,0], end [1,-2.2e-17,0], 32 points.
// ---------------------------------------------------------------------------
describe("Line.set_path_arc", () => {
  it("stores the arc and rebuilds points like python", () => {
    const l = new Line([0, 0, 0], [1, 0, 0]);
    expect(l.path_arc).toBe(0);
    const ret = l.set_path_arc(0.5);
    expect(ret === l).toBe(true);
    expect(l.path_arc).toBe(0.5);
    expectRowsClose([l.get_start()], [[0, 0, 0]], 1e-9);
    expectRowsClose([l.get_end()], [[1, 0, 0]], 1e-6);
    expect(toRows(l.points).length).toBe(32);
  });
  it("zero arc is a straight segment like python", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    l.set_path_arc(0);
    expectRowsClose([l.get_end()], [[2, 0, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Angle.from_three_points
// original manim: from_three_points([1,0,0],[0,0,0],[0,1,0]).get_value(degrees=True)==90.
// ---------------------------------------------------------------------------
describe("Angle.from_three_points", () => {
  it("builds the ABC angle (90 deg) like python", () => {
    const a = Angle.from_three_points([1, 0, 0], [0, 0, 0], [0, 1, 0]);
    expect(close(a.get_value(true), 90.0, 1e-6)).toBe(true);
  });
  it("collinear opposite rays give 0 deg like python", () => {
    // original manim: the two lines are parallel -> no unique intersection (issue
    // #1930) -> Angle becomes empty with angle_value 0.0.
    const a = Angle.from_three_points([-1, 0, 0], [0, 0, 0], [1, 0, 0]);
    expect(close(a.get_value(true), 0.0, 1e-6)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.RightAngle.__init__
// original manim: RightAngle(l1, l2).elbow is True; length=0.7 stored as radius.
// ---------------------------------------------------------------------------
describe("RightAngle.__init__", () => {
  it("sets elbow like python", () => {
    const ra = new RightAngle(
      new Line([0, 0, 0], [1, 0, 0]),
      new Line([0, 0, 0], [0, 1, 0]),
    );
    expect(ra.elbow).toBe(true);
  });
  it("maps length to radius like python", () => {
    const ra = new RightAngle(
      new Line([0, 0, 0], [1, 0, 0]),
      new Line([0, 0, 0], [0, 1, 0]),
      0.7,
    );
    expect(close((ra as unknown as { radius: number }).radius ?? NaN, 0.7, 1e-12)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// tips: ArrowTriangleFilledTip / ArrowSquareFilledTip __init__
// original manim: both default to fill_opacity 1, stroke_width 0.
// ---------------------------------------------------------------------------
describe("ArrowTriangleFilledTip.__init__", () => {
  it("defaults to fill 1 / stroke 0 like python", () => {
    const t = new ArrowTriangleFilledTip();
    expect(t.fill_opacity).toBe(1);
    expect(t.stroke_width).toBe(0);
  });
});

describe("ArrowSquareFilledTip.__init__", () => {
  it("defaults to fill 1 / stroke 0 like python", () => {
    const t = new ArrowSquareFilledTip();
    expect(t.fill_opacity).toBe(1);
    expect(t.stroke_width).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.get_mob_from_shape_element
// original manim: rect/line/polygon dispatch to rect_to_mobject/line_to_mobject/
//   polygon_to_mobject; text warns and returns None; empty-point mobs -> None.
// ---------------------------------------------------------------------------
describe("SVGMobject.get_mob_from_shape_element", () => {
  it("dispatches rect to a Rectangle with points like python", () => {
    const svg = new SVGMobject();
    const mob = svg.get_mob_from_shape_element({
      kind: "rect",
      x: 0,
      y: 0,
      width: 10,
      height: 5,
      rx: 0,
      ry: 0,
    });
    expect(mob !== null).toBe(true);
    expect(mob!.has_points()).toBe(true);
    // original manim rect_to_mobject shifts center to (x+w/2, y+h/2) = [5, 2.5, 0]
    // (no y-flip at this stage: _convert_point_to_3d is the identity map).
    expectRowsClose([mob!.get_center()], [[5, 2.5, 0]], 1e-6);
  });
  it("dispatches line kind like python", () => {
    const svg = new SVGMobject();
    const mob = svg.get_mob_from_shape_element({
      kind: "line",
      x1: 0,
      y1: 0,
      x2: 3,
      y2: 4,
    });
    expect(mob !== null).toBe(true);
    expect(mob!.has_points()).toBe(true);
  });
  it("dispatches polygon kind like python", () => {
    const svg = new SVGMobject();
    const mob = svg.get_mob_from_shape_element({
      kind: "polygon",
      points: [
        [0, 0],
        [1, 0],
        [0, 1],
      ],
    });
    expect(mob !== null).toBe(true);
    expect(mob!.has_points()).toBe(true);
  });
  it("returns null for text/unsupported kinds like python", () => {
    const svg = new SVGMobject();
    expect(svg.get_mob_from_shape_element({ kind: "text" })).toBeNull();
    expect(svg.get_mob_from_shape_element({ kind: "bogus" })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.shape_matchers.BackgroundRectangle.__init__
// original manim: Square(2) -> fill_opacity 0.75, stroke_width 0, stroke_opacity 0,
//   original_fill_opacity 0.75.
// ---------------------------------------------------------------------------
describe("BackgroundRectangle.__init__", () => {
  it("stores python defaults like python", () => {
    const bg = new BackgroundRectangle(new Square(2));
    expect(close(bg.fill_opacity, 0.75, 1e-12)).toBe(true);
    expect(bg.stroke_width).toBe(0);
    expect(close(bg.stroke_opacity, 0, 1e-12)).toBe(true);
    expect(close(bg.original_fill_opacity, 0.75, 1e-12)).toBe(true);
  });
  it("honors explicit kwargs like python", () => {
    const bg = new BackgroundRectangle(new Square(2), { fill_opacity: 0.5 });
    expect(close(bg.fill_opacity, 0.5, 1e-12)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.caching.handle_caching_play.wrapper
// original manim semantics: skip path appends None hash; normal path appends the hash
//   and calls func. TS mirrors with console.debug/info instead of logging.
// ---------------------------------------------------------------------------
describe("handle_caching_play.wrapper", () => {
  it("skip path appends null and calls func like python", () => {
    let called = 0;
    const func = function (this: unknown) {
      called += 1;
    };
    const wrapped = handle_caching_play(func as never);
    const scene = new Scene();
    const renderer = {
      skip_animations: true,
      _original_skipping_status: true,
      num_plays: 0,
      animations_hashes: [] as Array<string | null>,
      camera: {},
      file_writer: {
        is_already_cached: () => false,
        add_partial_movie_file: () => {},
      },
      update_skipping_status() {},
    };
    wrapped.call(renderer as never, scene as never, new Create(new Square()) as never);
    expect(called).toBe(1);
    expect(renderer.animations_hashes).toEqual([null]);
  });
  it("normal path appends an uncached hash when caching disabled", () => {
    let called = 0;
    const func = function (this: unknown) {
      called += 1;
    };
    const wrapped = handle_caching_play(func as never);
    const scene = new Scene();
    const added: Array<string | null> = [];
    const renderer = {
      skip_animations: false,
      _original_skipping_status: false,
      num_plays: 7,
      animations_hashes: [] as Array<string | null>,
      camera: {},
      file_writer: {
        is_already_cached: () => false,
        add_partial_movie_file: (h: string | null) => {
          added.push(h);
        },
      },
      update_skipping_status() {},
    };
    // force the disable_caching branch deterministically
    const prev = config.disable_caching;
    config.disable_caching = true;
    try {
      wrapped.call(renderer as never, scene as never, new Create(new Square()) as never);
    } finally {
      config.disable_caching = prev;
    }
    expect(called).toBe(1);
    expect(renderer.animations_hashes).toEqual(["uncached_00007"]);
    expect(added).toEqual(["uncached_00007"]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.image_mobject.ImageMobjectFromCamera.add_display_frame
// original manim: builds SurroundingRectangle(self, **defaults), adds it, returns self.
// ---------------------------------------------------------------------------
describe("ImageMobjectFromCamera.add_display_frame", () => {
  it("attaches a display frame and returns self like python", () => {
    const cam = new MovingCamera(null, { pixel_height: 2, pixel_width: 2 });
    const f = new ImageMobjectFromCamera(cam);
    const ret = f.add_display_frame();
    expect(ret === f).toBe(true);
    expect(f.display_frame !== undefined).toBe(true);
    expect(f.submobjects.includes(f.display_frame!)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.restore
// original manim: Square save/shift/restore roundtrips center; bare restore raises.
// ---------------------------------------------------------------------------
describe("Mobject.restore", () => {
  it("restores saved state like python", () => {
    const m = new Square();
    m.save_state();
    m.shift([1, 2, 0]);
    expectRowsClose([m.get_center()], [[1, 2, 0]], 1e-9);
    const ret = m.restore();
    expect(ret === m).toBe(true);
    expectRowsClose([m.get_center()], [[0, 0, 0]], 1e-9);
  });
  it("throws without a saved state like python", () => {
    expect(() => new Square().restore()).toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.add_ticks
// original manim: x_range [-3,3,1] -> 7 ticks, tick[0] from [-3,-0.1,0] to [-3,0.1,0].
// ---------------------------------------------------------------------------
describe("NumberLine.add_ticks", () => {
  it("builds 7 ticks like python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    // constructor without numbers already calls add_ticks
    expect(nl.ticks.submobjects.length).toBe(7);
    const t0 = nl.ticks.submobjects[0] as Line;
    expectRowsClose([t0.get_start()], [[-3, -0.1, 0]], 1e-9);
    expectRowsClose([t0.get_end()], [[-3, 0.1, 0]], 1e-9);
  });
  it("returns self like python", () => {
    const nl = new NumberLine({ x_range: [0, 2, 1] });
    expect(nl.add_ticks() === nl).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.add_foreground_mobject
// original manim: adds to foreground_mobjects AND mobjects.
// ---------------------------------------------------------------------------
describe("Scene.add_foreground_mobject", () => {
  it("tracks foreground and scene mobjects like python", () => {
    const sc = new Scene();
    const s = new Square();
    const ret = sc.add_foreground_mobject(s);
    expect(ret === sc).toBe(true);
    expect(sc.foreground_mobjects.includes(s)).toBe(true);
    expect(sc.mobjects.includes(s)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.compile_animation_data
// original manim: Create -> returns self, duration 1.0; empty -> ValueError;
//   static Wait -> None + is_static_wait True.
// ---------------------------------------------------------------------------
describe("Scene.compile_animation_data", () => {
  it("compiles a Create like python", () => {
    const sc = new Scene();
    const r = sc.compile_animation_data(new Create(new Square()));
    expect(r === sc).toBe(true);
    expect(close(sc.duration, 1.0, 1e-9)).toBe(true);
    expect(sc.animations.length).toBe(1);
  });
  it("raises on empty like python", () => {
    expect(() => new Scene().compile_animation_data()).toThrow();
  });
  it("static Wait returns null like python", () => {
    const sc = new Scene();
    const w = new Wait({ run_time: 2 });
    const r = sc.compile_animation_data(w);
    expect(r).toBeNull();
    expect(w.is_static_wait).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.NumberPlane._get_lines_parallel_to_axis
// original manim: NumberPlane([-3,3,1],[-2,2,1],6,4) axes, freq 1 ratio 4 ->
//   (len(lines1), len(lines2)) == (2, 13); ratio 0 -> (3, 0).
// NOTE: the TS Axes hierarchy is not Mobject-based yet (get_axes() is an
//   empty VGroup until the NumberLine integration lands), so the axes are
//   passed explicitly as NumberLines with the same ranges/lengths Python
//   builds internally. The count logic only reads x_range/x_min plus
//   get_start/get_end/get_unit_vector, all of which NumberLine provides.
// ---------------------------------------------------------------------------
describe("NumberPlane._get_lines_parallel_to_axis", () => {
  function makePlaneAxes(): [NumberLine, NumberLine] {
    const xa = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const ya = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    return [xa, ya];
  }
  it("splits faded/non-faded counts like python", () => {
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const [xa, ya] = makePlaneAxes();
    const [l1, l2] = plane._get_lines_parallel_to_axis(xa, ya, 1.0, 4);
    expect(l1.submobjects.length).toBe(2);
    expect(l2.submobjects.length).toBe(13);
  });
  it("ratio 0 disables faded lines like python", () => {
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const [xa, ya] = makePlaneAxes();
    const [l1, l2] = plane._get_lines_parallel_to_axis(xa, ya, 1.0, 0);
    expect(l1.submobjects.length).toBe(3);
    expect(l2.submobjects.length).toBe(0);
  });
  it("shifts each line along the perpendicular unit vector like python", () => {
    // original manim: every line in lines1/lines2 is parallel to the x axis, i.e. its
    // direction has ~zero y/z extent.
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const [xa, ya] = makePlaneAxes();
    const [l1, l2] = plane._get_lines_parallel_to_axis(xa, ya, 1.0, 4);
    for (const ln of [...l1.submobjects, ...l2.submobjects]) {
      const v = (ln as Line).get_vector();
      expect(close(v[1] ?? NaN, 0, 1e-9)).toBe(true);
      expect(close(v[2] ?? NaN, 0, 1e-9)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// LinearTransformationScene add_*_mobject
// original manim (after setup()): each appends to its list and adds to the scene.
// ---------------------------------------------------------------------------
describe("LinearTransformationScene.add_*_mobject", () => {
  it("tracks transformable/background/foreground lists like python", () => {
    // NOTE: setup() is run with show_basis_vectors=false: the basis-vector
    // path needs Vector (a later batch), while add_*_mobject (this batch's
    // targets) only need the three tracking lists + scene add.
    const sc = new LinearTransformationScene(true, true, null, null, false, false);
    sc.setup();
    const a = new Square();
    const b = new Square();
    const c = new Square();
    sc.add_transformable_mobject(a);
    expect(sc.transformable_mobjects.includes(a)).toBe(true);
    expect(sc.mobjects.includes(a)).toBe(true);
    sc.add_background_mobject(b);
    expect(sc.background_mobjects.includes(b)).toBe(true);
    expect(sc.mobjects.includes(b)).toBe(true);
    sc.add_foreground_mobject(c);
    expect(
      (sc.foreground_mobjects as unknown[]).includes(c as unknown),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform_matching_parts.TransformMatchingShapes.get_mobject_key
// original manim: Square keys equal, Circle differs; returns an int.
// ---------------------------------------------------------------------------
describe("TransformMatchingShapes.get_mobject_key", () => {
  it("is stable per shape and differs across shapes like python", () => {
    const k1 = TransformMatchingShapes.get_mobject_key(new Square());
    const k2 = TransformMatchingShapes.get_mobject_key(new Square());
    const k3 = TransformMatchingShapes.get_mobject_key(new Circle());
    expect(k1).toBe(k2);
    expect(k1).not.toBe(k3);
    expect(Number.isInteger(k1)).toBe(true);
  });
  it("restores the mobject (no visible side effect) like python", () => {
    const s = new Square();
    const before = toRows(s.points);
    TransformMatchingShapes.get_mobject_key(s);
    expect(toRows(s.points)).toEqual(before);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.creation.Write.begin
// original manim: begin() works; reverse flips submobjects twice (begin+finish).
// ---------------------------------------------------------------------------
describe("Write.begin", () => {
  it("begins without error like python", () => {
    const sq = new Square();
    const w = new Write(sq);
    w.begin();
    expect(w.mobject === sq).toBe(true);
  });
  it("reverse inverts on begin and restores on finish like python", () => {
    const sq = new Square();
    const before = toRows(sq.points);
    const w = new Write(sq, { reverse: true });
    w.begin();
    w.finish();
    expect(toRows(sq.points)).toEqual(before);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyPointwiseFunctionToCenter.begin
// NOTE: upstream Python __init__ drops the mobject arg
//   (super().__init__(mobject.move_to) -> TypeError), so construction raises
//   in the original manim. The TS port threads a dummy pointwise function and rewires
//   the method facade to move_to; begin() then sets method_args to
//   [function(center)] exactly as Python's begin() intends.
// ---------------------------------------------------------------------------
describe("ApplyPointwiseFunctionToCenter.begin", () => {
  it("sets method_args to function(center) like python's begin()", () => {
    const sq = new Square();
    const a = new ApplyPointwiseFunctionToCenter(
      (p: number[]) => [p[0]! + 1, p[1]!, p[2]!],
      sq,
    );
    a.begin();
    expectRowsClose([a.method_args[0] as number[]], [[1, 0, 0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.Restore.__init__
// original manim: Restore(m) binds mobject.restore; begin/interpolate restores state.
// ---------------------------------------------------------------------------
describe("Restore.__init__", () => {
  it("binds the mobject restore method like python", () => {
    const sq = new Square();
    sq.save_state();
    sq.shift([1, 0, 0]);
    const r = new Restore(sq);
    expect(
      (r.method as unknown as { __self__: unknown }).__self__ === sq,
    ).toBe(true);
    r.begin();
    r.interpolate(1);
    expectRowsClose([sq.get_center()], [[0, 0, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.FadeTransformPieces.begin
// original manim: mobject has 2 submobjects after begin (aligned pair).
// ---------------------------------------------------------------------------
describe("FadeTransformPieces.begin", () => {
  it("aligns the submobject pair like python", () => {
    const ft = new FadeTransformPieces(new Circle(), new Square());
    ft.begin();
    expect((ft.mobject as Group).submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.mobject_update_utils.turn_animation_into_updater
// original manim: adds one updater to the mobject, returns it, total_time 0,
//   suspend_mobject_updating False.
// ---------------------------------------------------------------------------
describe("turn_animation_into_updater", () => {
  it("attaches an updater and returns the mobject like python", () => {
    const sq = new Square();
    const anim = new Create(sq);
    const mob = turn_animation_into_updater(anim);
    expect(mob === sq).toBe(true);
    expect(sq.updaters.length).toBe(1);
    // original manim total_time is -delay == -0.0 for delay=0 (-0.0 == 0 in Python;
    // Object.is distinguishes them, so compare with tolerance).
    expect(close(anim.total_time, 0, 1e-12)).toBe(true);
    expect(anim.suspend_mobject_updating).toBe(false);
  });
  it("honors delay like python", () => {
    const sq = new Square();
    const anim = new Create(sq);
    turn_animation_into_updater(anim, false, 0.5);
    expect(anim.total_time).toBe(-0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.get_highlighted_cell
// original manim: returns BackgroundRectangle with fill 0.75.
// ---------------------------------------------------------------------------
describe("Table.get_highlighted_cell", () => {
  it("returns a BackgroundRectangle like python", () => {
    // NOTE: Table([["a",...]]) string entries need Paragraph (text/tex, out
    // of scope); MobjectTable with the identity factory exercises the same
    // get_highlighted_cell -> BackgroundRectangle(cell) path. original manim ground
    // truth on Table([["a","b"],["c","d"]]).get_highlighted_cell((2,2)):
    // BackgroundRectangle with fill_opacity 0.75, stroke_width 0.
    const tb = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: (m: unknown) => m as never },
    );
    const cell = tb.get_highlighted_cell([2, 2]);
    expect(cell instanceof BackgroundRectangle).toBe(true);
    expect(close(cell.fill_opacity, 0.75, 1e-12)).toBe(true);
    expect(cell.stroke_width).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.logo.ManimBanner.expand.shift (inner closure)
// original manim: expand() validates direction (ValueError otherwise) and returns a
//   Succession; the shift closure restores shapes then slides per direction.
// ---------------------------------------------------------------------------
describe("ManimBanner.expand", () => {
  it("rejects bad directions like python", () => {
    expect(() => new ManimBanner().expand(1.5, "up")).toThrow();
  });
  it("returns a 2-step Succession and saves shapes state like python", () => {
    const b = new ManimBanner();
    const s = b.expand();
    expect(s.animations.length).toBe(2);
    expect(b.shapes.saved_state !== null).toBe(true);
  });
  it("center shift closure slides shapes by half the eased offset like python", () => {
    // Drives the inner shift() closure through the Succession's first
    // UpdateFromAlphaFunc (slide_and_uncover): with direction="center" the
    // shapes move by half the eased slide vector. original manim ground truth
    // (m_shape_offset 6.25 + overshoot 0.8 = 7.05, ease_in_out_cubic):
    //   alpha=0.25 -> dx 0.2203125; alpha=0.5 -> dx 1.7625.
    // shift() restores shapes to the expand()-time saved state on every
    // call, so both deltas are absolute from that saved center.
    const b = new ManimBanner();
    const s = b.expand(1.5, "center");
    const c0 = [...b.shapes.get_center()];
    const sub = s.animations[0]!;
    sub.begin();
    sub.interpolate(0.25);
    expect(close(b.shapes.get_center()[0]! - c0[0]!, 0.2203125, 1e-9)).toBe(true);
    expect(close(b.shapes.get_center()[1]! - c0[1]!, 0, 1e-9)).toBe(true);
    sub.interpolate(0.5);
    expect(close(b.shapes.get_center()[0]! - c0[0]!, 1.7625, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.probability.SampleSpace divide/getitem
// original manim: divide_horizontally([0.3,0.7]) -> 2 parts added as submobject;
//   divide_vertically([0.5]) -> 2 parts; [0] delegates to horizontal_parts[0].
// ---------------------------------------------------------------------------
describe("SampleSpace.divide_horizontally", () => {
  it("splits into 2 horizontal parts like python", () => {
    const s = new SampleSpace();
    const ret = s.divide_horizontally([0.3, 0.7]);
    expect(ret === s).toBe(true);
    expect(s.horizontal_parts!.submobjects.length).toBe(2);
    expect(s.submobjects.includes(s.horizontal_parts!)).toBe(true);
  });
});

describe("SampleSpace.divide_vertically", () => {
  it("splits into 2 vertical parts like python", () => {
    const s = new SampleSpace();
    const ret = s.divide_vertically([0.5]);
    expect(ret === s).toBe(true);
    // complete_p_list appends the 0.5 remainder -> 2 parts
    expect(s.vertical_parts!.submobjects.length).toBe(2);
    expect(s.submobjects.includes(s.vertical_parts!)).toBe(true);
  });
});

describe("SampleSpace.__getitem__", () => {
  it("delegates to horizontal_parts like python", () => {
    const s = new SampleSpace();
    s.divide_horizontally([0.3, 0.7]);
    expect(s.__getitem__(0) === s.horizontal_parts!.__getitem__(0)).toBe(true);
  });
  it("delegates to vertical_parts when no horizontal split like python", () => {
    const s = new SampleSpace();
    s.divide_vertically([0.5]);
    expect(s.__getitem__(1) === s.vertical_parts!.__getitem__(1)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_lines_to_point
// original manim: returns a VGroup of [horizontal, vertical] lines.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_lines_to_point", () => {
  it("returns a VGroup of horizontal+vertical lines like python", () => {
    const lines = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4).get_lines_to_point([1, 1, 0]);
    expect(lines.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_implicit_curve
// original manim: circle x^2+y^2-1 -> ImplicitFunction with points.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_implicit_curve", () => {
  it("returns an ImplicitFunction (reads the built axes' scaling)", () => {
    const graph = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4).plot_implicit_curve(
      (x, y) => x * x + y * y - 1,
    );
    expect(graph).toBeInstanceOf(ImplicitFunction);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.plot_surface
// original manim (cairo): Surface over (u, v) -> c2p(u, v, f(u, v)) with u_range,
//   v_range stored and sampled faces built.
// NOTE (was DEVIATION D-surface-z, now fixed): ThreeDAxes.coords_to_point
//   maps all three coordinates, so plot_surface corners carry the faithful
//   z = axisPosition(f(u,v)) exactly like the original manim (see rows below).
// ---------------------------------------------------------------------------
describe("CoordinateSystem.plot_surface", () => {
  it("stores ranges and places x/y through c2p like python", () => {
    const ax = new ThreeDAxes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const surf = ax.plot_surface((u, v) => u + v, [0, 1], [0, 1], null, 2, {
      resolution: 4,
    });
    expect(surf.u_range).toEqual([0, 1]);
    expect(surf.v_range).toEqual([0, 1]);
    // original manim (resolution=4): 16 submob faces. TS Surface keeps geometry in
    // the face submobjects (the group itself holds no points).
    expect(surf.submobjects.length).toBe(16);
    expect(surf.list_of_faces.length).toBe(16);
    // x/y of each face corner match c2p(u, v, f) x/y (original manim: c2p is the
    // linear map; e.g. c2p(0,0,0) == origin [0,0,0] on this symmetric grid).
    const origin = toRows(ax.c2p(0, 0, 0))[0]!;
    expectRowsClose([origin], [[0, 0, 0]], 1e-9);
    const face0 = surf.list_of_faces[0]!;
    const corners = toRows(face0.points);
    expect(corners.length).toBeGreaterThan(0);
    const c2p00 = toRows(ax.c2p(face0.u1, face0.v1, 0))[0]!;
    expect(close(corners[0]?.[0] ?? NaN, c2p00[0] ?? NaN, 1e-9)).toBe(true);
    expect(close(corners[0]?.[1] ?? NaN, c2p00[1] ?? NaN, 1e-9)).toBe(true);
    // original manim (resolution 4, f(u,v)=u+v) face0 corners, full rows:
    // [[0,0,0],[1/12,0,~0.067708],[2/12,0,~0.135417],[3/12,0,0.203125]].
    // Each corner (x, 0, z) satisfies [x, 0, z] == c2p(x, 0, x) exactly.
    expect(corners.length).toBeGreaterThanOrEqual(4);
    for (let i = 0; i < 4; i += 1) {
      const corner = corners[i] ?? [NaN, NaN, NaN];
      const expected = toRows(ax.c2p(corner[0] ?? 0, 0, corner[0] ?? 0))[0]!;
      // NOTE: 1e-6, not 1e-9: Surface samples u/v on a float grid, so the
      // corner x already carries ~6e-11 sampling noise that the z-mapping
      // (slope 0.8125) propagates; the original manim print itself only gives 6dp.
      expect(close(corner[0] ?? NaN, expected[0] ?? NaN, 1e-6)).toBe(true);
      expect(close(corner[1] ?? NaN, expected[1] ?? NaN, 1e-6)).toBe(true);
      expect(close(corner[2] ?? NaN, expected[2] ?? NaN, 1e-6)).toBe(true);
    }
    // Spot-check against the original manim print: p1 z ~= 0.067708 (absolute tol:
    // the print itself only carries 6dp, so relative-1e-6 is meaningless).
    expect(Math.abs((corners[1]?.[2] ?? NaN) - 0.067708) <= 1e-6).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_vertical_lines_to_graph
// original manim: x_range [0,2] num_lines 5 -> 5 lines.
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_vertical_lines_to_graph", () => {
  it("returns num_lines vertical lines like python", () => {
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const g = ax.plot((x) => x * x);
    expect(g.has_points()).toBe(true);
    const lines = ax.get_vertical_lines_to_graph(g, [0, 2], 5);
    expect(lines.submobjects.length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem.get_T_label
// original manim (label=None, no tex): VGroup of [triangle, v_line].
// ---------------------------------------------------------------------------
describe("CoordinateSystem.get_T_label", () => {
  it("returns [triangle, vertical line] like python", () => {
    const ax = new Axes([-1, 10, 1], [-1, 10, 1], 9, 6);
    const g = ax.plot((x) => x);
    const group = ax.get_T_label(4, g, null);
    expect(group.submobjects.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Axis/coordinate labels now render through MathTex / DecimalNumber (MathJax).
// ---------------------------------------------------------------------------
describe("axis/coordinate labels (MathTex-rendered)", () => {
  it("Axes.get_axis_labels builds two positioned labels", () => {
    const labels = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4).get_axis_labels("x", "y");
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(2);
    for (const label of labels.submobjects) {
      expect(label).toBeInstanceOf(VMobject);
    }
  });
  it("ThreeDAxes.get_axis_labels builds three positioned labels", () => {
    const labels = new ThreeDAxes([-3, 3, 1], [-2, 2, 1], 6, 4).get_axis_labels(
      "x",
      "y",
      "z",
    );
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(3);
  });
  it("ComplexPlane.get_coordinate_labels builds DecimalNumber labels", () => {
    const labels = new ComplexPlane().get_coordinate_labels(1, 2);
    expect(labels).toBeInstanceOf(VGroup);
    expect(labels.submobjects.length).toBe(2);
    expect((labels.submobjects[0] as unknown as DecimalNumber).get_value()).toBe(1);
    expect((labels.submobjects[1] as unknown as DecimalNumber).get_value()).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.parallel_to
// original manim: direction matches (dot +/-1), default length 5.
// ---------------------------------------------------------------------------
describe("Line3D.parallel_to", () => {
  it("returns a parallel line of length 5 like python", () => {
    const l1 = new Line3D([2, 0, 0], [0, 1, 1]);
    const par = Line3D.parallel_to(l1);
    const dot =
      (par.vect[0]! * l1.vect[0]! + par.vect[1]! * l1.vect[1]! + par.vect[2]! * l1.vect[2]!) /
      (Math.hypot(...par.vect) * Math.hypot(...l1.vect));
    expect(close(Math.abs(dot), 1, 1e-9)).toBe(true);
    expect(close(Math.hypot(...par.vect), 5, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.perpendicular_to
// original manim: perpendicular (dot 0); collinear point raises ValueError.
// ---------------------------------------------------------------------------
describe("Line3D.perpendicular_to", () => {
  it("returns a perpendicular line like python", () => {
    const l1 = new Line3D([2, 0, 0], [0, 1, 1]);
    const perp = Line3D.perpendicular_to(l1);
    const dot =
      perp.vect[0]! * l1.vect[0]! + perp.vect[1]! * l1.vect[1]! + perp.vect[2]! * l1.vect[2]!;
    const denom = Math.hypot(...perp.vect) * Math.hypot(...l1.vect);
    expect(close(dot / denom, 0, 1e-9)).toBe(true);
  });
  it("raises when no perpendicular exists like python", () => {
    try {
      Line3D.perpendicular_to(new Line3D([0, 0, 0], [1, 0, 0]), [0.5, 0, 0]);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (this batch x already-ported code)
// ---------------------------------------------------------------------------
describe("integration (batch layers20-21)", () => {
  it("set_path_arc + restore roundtrip matches python", () => {
    // original manim: save_state, set_path_arc(0.5), restore() -> path_arc stays 0.5
    // (become() does not copy it back) while the points revert to the saved
    // straight segment ending at [1,0,0].
    const l = new Line([0, 0, 0], [1, 0, 0]);
    l.save_state();
    l.set_path_arc(0.5);
    expect(l.path_arc).toBe(0.5);
    l.restore();
    expect(l.path_arc).toBe(0.5);
    expectRowsClose([l.get_end()], [[1, 0, 0]], 1e-9);
  });
  it("RightAngle + from_three_points share the Angle pipeline like python", () => {
    // original manim: both produce elbow/measured angles off the same Line pair.
    const l1 = new Line([0, 0, 0], [1, 0, 0]);
    const l2 = new Line([0, 0, 0], [0, 1, 0]);
    const ra = new RightAngle(l1, l2);
    const a = Angle.from_three_points([1, 0, 0], [0, 0, 0], [0, 1, 0]);
    expect(ra.elbow).toBe(true);
    expect(close(a.get_value(true), 90, 1e-6)).toBe(true);
  });
  it("Table highlight + BackgroundRectangle defaults agree like python", () => {
    const tb = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: (m: unknown) => m as never },
    );
    const hl = tb.get_highlighted_cell([1, 1]);
    const bg = new BackgroundRectangle(new Square());
    expect(close(hl.fill_opacity, bg.fill_opacity, 1e-12)).toBe(true);
    expect(hl.stroke_width).toBe(bg.stroke_width);
  });
  it("SampleSpace parts are VMobjects supporting get_center like python", () => {
    const s = new SampleSpace();
    s.divide_horizontally([0.25, 0.75]);
    const p0 = s.__getitem__(0);
    expectRowsClose([(p0 as VGroup).get_center()], [(s.horizontal_parts!.submobjects[0] as VGroup).get_center()] as number[][], 1e-9);
  });
  it("Write + turn_animation_into_updater compose on one mobject like python", () => {
    const sq = new Square();
    const w = new Write(sq);
    w.begin();
    turn_animation_into_updater(new Create(sq));
    expect(sq.updaters.length).toBe(1);
  });
  it("Axes plot + NumberPlane grid + SVG rect chain like python", () => {
    // original manim: ax.plot(x) has 240 points; NumberPlane._get_lines_parallel_to_axis
    // on explicit NumberLines gives (2, 13); rect_to_mobject centers match.
    const ax = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    const g = ax.plot((x) => x);
    expect(g.has_points()).toBe(true);
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    const xa = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    const ya = new NumberLine({ x_range: [-2, 2, 1], length: 4 });
    const [l1, l2] = plane._get_lines_parallel_to_axis(xa, ya, 1.0, 4);
    expect(l1.submobjects.length + l2.submobjects.length).toBe(15);
    const svg = new SVGMobject();
    const mob = svg.get_mob_from_shape_element({
      kind: "rect",
      x: 0,
      y: 0,
      width: 2,
      height: 2,
      rx: 0,
      ry: 0,
    });
    expect(mob!.has_points()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ComplexPlane +
// ComplexPlane._get_default_coordinate_values
// original manim (x_range=[-3,3,1], y_range=[-2,2,1]):
//   [-3,-2,-1,1,2,3,-2j,-1j,1j,2j]; n2p(1+2j)=[1.0,2.0,0.0]; p2n back=(1+2j).
// ---------------------------------------------------------------------------
describe("ComplexPlane", () => {
  it("_get_default_coordinate_values matches python", () => {
    const cp = new ComplexPlane([-3, 3, 1], [-2, 2, 1]);
    const vals = cp._get_default_coordinate_values();
    expect(vals.length).toBe(10);
    const reals = (vals.slice(0, 6) as number[]).map(Number);
    expect(reals).toEqual([-3, -2, -1, 1, 2, 3]);
    const imags = vals.slice(6) as Array<{ re: number; im: number }>;
    expect(imags.map((c) => c.re)).toEqual([0, 0, 0, 0]);
    expect(imags.map((c) => c.im)).toEqual([-2, -1, 1, 2]);
  });
  it("n2p/p2n roundtrip matches python", () => {
    const cp = new ComplexPlane([-3, 3, 1], [-2, 2, 1]);
    const p = cp.n2p({ re: 1, im: 2 });
    expectRowsClose(p, [[1, 2, 0]], 1e-9);
    const back = cp.p2n([1, 2, 0]);
    expect(close(back.re, 1, 1e-9)).toBe(true);
    expect(close(back.im, 2, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.rotate_about_zero
// original manim (x_range=[0,4,1]): start [-2,0,0], end [2,0,0], n2p(0)=[-2,0,0];
// rotate_about_zero(pi/2) -> start [-2,0,0], end [-2,4,0].
// ---------------------------------------------------------------------------
describe("NumberLine.rotate_about_zero", () => {
  it("rotates about n2p(0) like python", () => {
    const nl = new NumberLine({ x_range: [0, 4, 1] });
    expectRowsClose([nl.get_start()], [[-2, 0, 0]], 1e-9);
    expectRowsClose([nl.get_end()], [[2, 0, 0]], 1e-9);
    const ret = nl.rotate_about_zero(Math.PI / 2);
    expect(ret === nl).toBe(true);
    expectRowsClose([nl.get_start()], [[-2, 0, 0]], 1e-9);
    expectRowsClose([nl.get_end()], [[-2, 4, 0]], 1e-9);
  });
  it("zero angle is the identity like python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1] });
    const before = [nl.get_start(), nl.get_end()];
    nl.rotate_about_zero(0);
    expectRowsClose([nl.get_start()], [before[0] as number[]], 1e-12);
    expectRowsClose([nl.get_end()], [before[1] as number[]], 1e-12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Cube.init_points
// original manim: 6 faces, each with points; cube w/h/d = 2.0; init_points returns self.
// ---------------------------------------------------------------------------
describe("Cube.init_points", () => {
  it("builds six faces like python", () => {
    const c = new Cube(2);
    expect(c.submobjects.length).toBe(6);
    for (const face of c.submobjects) {
      expect((face as VGroup).has_points()).toBe(true);
    }
    expect(close(c.width, 2, 1e-9)).toBe(true);
    expect(close(c.height, 2, 1e-9)).toBe(true);
    expect(close(c.depth, 2, 1e-9)).toBe(true);
  });
  it("init_points regenerates and returns self like python", () => {
    // original manim: generate_points() appends (no clearing), so an explicit second
    // init_points() doubles the faces to 12 in BOTH implementations.
    const c = new Cube(2);
    const ret = c.init_points();
    expect(ret === c).toBe(true);
    expect(c.submobjects.length).toBe(12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Prism.generate_points
// original manim (dimensions=[3,2,1]): width 3.0, height 2.0, depth 1.0, 6 faces.
// ---------------------------------------------------------------------------
describe("Prism.generate_points", () => {
  it("rescales to the dimensions like python", () => {
    const p = new Prism([3, 2, 1]);
    expect(p.submobjects.length).toBe(6);
    expect(close(p.width, 3, 1e-9)).toBe(true);
    expect(close(p.height, 2, 1e-9)).toBe(true);
    expect(close(p.depth, 1, 1e-9)).toBe(true);
  });
  it("generate_points returns self like python", () => {
    const p = new Prism([3, 2, 1]);
    expect(p.generate_points() === p).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.__init__
// original manim: thickness 0.02, resolution int 24 -> (2, 24), start LEFT, end RIGHT.
// ---------------------------------------------------------------------------
describe("Line3D.__init__", () => {
  it("stores defaults like python", () => {
    const l = new Line3D([-1, 0, 0], [1, 0, 0]);
    expect(l.thickness).toBe(0.02);
    expect(l.resolution).toEqual([2, 24]);
    expectRowsClose(l.start, [[-1, 0, 0]], 1e-12);
    expectRowsClose(l.end, [[1, 0, 0]], 1e-12);
    expectRowsClose(l.vect, [[2, 0, 0]], 1e-12);
    expect(close(l.length, 2, 1e-12)).toBe(true);
    expectRowsClose(l.direction, [[1, 0, 0]], 1e-12);
  });
  it("honors explicit thickness/color/resolution like python", () => {
    const l = new Line3D([0, 0, 0], [0, 1, 0], 0.05, "RED", [4, 8]);
    expect(l.thickness).toBe(0.05);
    expect(l.resolution).toEqual([4, 8]);
    expect(close(l.length, 1, 1e-12)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ThreeDAxes.get_y/z_axis_label
// original manim (Square(0.5) labels, ranges [-3,3,1]/[-2,2,1]/[-2,2,1]):
//   y-label center [0.525, 3.5, 0], z-label center [0.35, 0, 3.25],
//   and the label object is returned as-is (VMobject passthrough).
// ---------------------------------------------------------------------------
describe("ThreeDAxes.get_y_axis_label/get_z_axis_label", () => {
  it("places the y-axis label like python (center [0.525, 3.5, 0])", () => {
    const ax = new ThreeDAxes([-3, 3, 1], [-2, 2, 1]);
    const center = ax.get_y_axis_label(new Square(0.5))!.get_center();
    expect(close(center[0] ?? NaN, 0.525, 1e-9)).toBe(true);
    expect(close(center[1] ?? NaN, 3.5, 1e-9)).toBe(true);
    expect(close(center[2] ?? NaN, 0, 1e-9)).toBe(true);
  });
  it("places the z-axis label like python (center [0.35, 0, 3.25])", () => {
    const ax = new ThreeDAxes([-3, 3, 1], [-2, 2, 1]);
    const center = ax.get_z_axis_label(new Square(0.5))!.get_center();
    expect(close(center[0] ?? NaN, 0.35, 1e-9)).toBe(true);
    expect(close(center[1] ?? NaN, 0, 1e-9)).toBe(true);
    expect(close(center[2] ?? NaN, 3.25, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (new symbols x already-ported code)
// ---------------------------------------------------------------------------
describe("integration (batch layers20-21b)", () => {
  it("ComplexPlane default values feed coordinate counts like python", () => {
    const cp = new ComplexPlane([-3, 3, 1], [-2, 2, 1]);
    const vals = cp._get_default_coordinate_values();
    const tickTotal =
      (cp.get_x_axis().get_tick_range().toArray() as unknown[]).length +
      (cp.get_y_axis().get_tick_range().toArray() as unknown[]).length;
    // Both axes already exclude the origin tick, so the y!=0 filter drops
    // nothing: vals = x_ticks + y_ticks (original manim default plane: 14 + 8 = 22).
    expect(vals.length).toBe(tickTotal);
  });
  it("Cube restore roundtrip after shift matches python Mobject.restore", () => {
    const c = new Cube(2);
    c.save_state();
    const center0 = [...c.get_center()];
    c.shift([1, 2, 3]);
    c.restore();
    expectRowsClose([c.get_center()], [center0], 1e-9);
  });
  it("Prism + Line3D agree on extents like python", () => {
    // original manim: Prism([3,2,1]) spans x in [-1.5,1.5]; Line3D tracks its vector.
    const p = new Prism([3, 2, 1]);
    const l = new Line3D([-1.5, 0, 0], [1.5, 0, 0]);
    expect(close(p.width, l.length, 1e-9)).toBe(true);
  });
  it("rotate_about_zero then add_ticks agree like python", () => {
    // original manim: NumberLine([-2,2,1]) has 5 ticks; rotation preserves the count.
    const nl = new NumberLine({ x_range: [-2, 2, 1] });
    nl.rotate_about_zero(Math.PI / 4);
    nl.add_ticks();
    expect(nl.ticks.submobjects.length).toBe(5);
  });
});
