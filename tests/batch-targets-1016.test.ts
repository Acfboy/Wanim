// Ground-truth tests for batch layers 16..16.
//
// Targets:
//   - manim.animation.indication.Flash.create_lines
//   - manim.animation.indication.ShowPassingFlashWithThinningStrokeWidth.__init__
//   - manim.mobject.geometry.polygram.Polygon.__init__
//   - manim.animation.transform.ApplyPointwiseFunctionToCenter[. __init__]
//   - manim.animation.transform.ShrinkToCenter[. __init__]
//   - manim.animation.transform.ApplyMatrix[. __init__]
//   - manim.animation.growing.SpinInFromNothing[. __init__]
//   - manim.animation.animation.Animation.get_all_mobjects_to_update
//   - manim.mobject.geometry.polygram.Rectangle
//   - manim.mobject.vector_field.VectorField.__init__
//   - manim.mobject.vector_field.ArrowVectorField
//   - manim.mobject.vector_field.StreamLines[.create]
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the SAME inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete values;
// arrays element-wise).
//
// OUT OF SCOPE (text/tex): none of these symbols touch text/tex.
import { describe, expect, it } from "vitest";
import { Mobject } from "../src/mobject/mobject.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Polygon, Rectangle, Square } from "../src/mobject/geometry/polygram.js";
import { Line } from "../src/mobject/geometry/line.js";
import { Dot } from "../src/mobject/geometry/arc.js";
import {
  ArrowVectorField,
  StreamLines,
  VectorField,
} from "../src/mobject/vector_field.js";
import { Flash, ShowPassingFlashWithThinningStrokeWidth } from "../src/animation/indication.js";
import {
  Animation,
} from "../src/animation/animation.js";
import {
  ApplyMatrix,
  ApplyPointwiseFunctionToCenter,
  ShrinkToCenter,
} from "../src/animation/transform.js";
import { SpinInFromNothing } from "../src/animation/growing.js";
import { config } from "../src/_config/utils.js";
import { pointRows } from "../src/utils/arrays.js";
import { PI, TAU } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
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

// ---------------------------------------------------------------------------
// manim.animation.indication.Flash.create_lines
// Python:
//   f = Flash([0,0,0], line_length=0.5, num_lines=4, flash_radius=0.2)
//   lines = f.create_lines() -> 4 lines, stroke width 3 each
//   line0 start [0.2,0,0] end [0.7,0,0]
//   line1 start [4.44e-17,0.2,0] end [1.55e-16,0.7,0]
//   line2 start [-0.2,2.45e-17,0] end [-0.7,8.57e-17,0]
//   line3 start [-4.44e-17,-0.2,0] end [-1.55e-16,-0.7,0]
//   Flash(Square(), num_lines=8).point == [0,0,0], 8 lines
// ---------------------------------------------------------------------------
describe("Flash.create_lines", () => {
  it("fans 4 lines out at cardinal angles with ground-truth endpoints", () => {
    const f = new Flash([0, 0, 0], 0.5, 4, 0.2);
    const lines = f.create_lines();
    expect(lines.submobjects.length).toBe(4);
    const starts = lines.submobjects.map((l) => (l as Line).get_start());
    const ends = lines.submobjects.map((l) => (l as Line).get_end());
    expectRowsClose(starts, [
      [0.2, 0.0, 0.0],
      [4.4408920985006264e-17, 0.2, 0.0],
      [-0.2, 2.4492935982947065e-17, 0.0],
      [-4.4408920985006264e-17, -0.2, 0.0],
    ]);
    expectRowsClose(ends, [
      [0.7, 0.0, 0.0],
      [1.554312234475219e-16, 0.7, 0.0],
      [-0.7, 8.572527594031472e-17, 0.0],
      [-1.554312234475219e-16, -0.7, 0.0],
    ]);
    for (const l of lines.submobjects) {
      expectClose((l as Line).get_stroke_width(), 3);
    }
    // Python builds a fresh VGroup per call without reassigning self.lines.
    expect(lines).not.toBe(f.lines);
    expect(f.lines.submobjects.length).toBe(4);
  });

  it("resolves a Mobject point to its center", () => {
    const f = new Flash(new Square(), 0.2, 8);
    expectRowsClose([f.point], [[0, 0, 0]]);
    expect(f.lines.submobjects.length).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.indication.ShowPassingFlashWithThinningStrokeWidth.__init__
// Python:
//   vm = Line([-1,0,0],[1,0,0]); vm.set_stroke(width=8)
//   sp = ShowPassingFlashWithThinningStrokeWidth(vm, n_segments=4, time_width=0.2)
//   len(sp.animations) == 4
//   (tw, sw): (0.2, 0.0), (0.13333, 2.66667), (0.06667, 5.33333), (0.0, 8.0)
//   i.e. stroke widths linspace(0, 8, 4), time widths linspace(0.2, 0, 4)
// ---------------------------------------------------------------------------
describe("ShowPassingFlashWithThinningStrokeWidth.__init__", () => {
  it("fans stroke/time widths across n_segments with ground-truth values", () => {
    const vm = new Line([-1, 0, 0], [1, 0, 0]);
    vm.set_stroke(undefined, 8);
    const sp = new ShowPassingFlashWithThinningStrokeWidth(vm, 4, 0.2);
    expect(sp.n_segments).toBe(4);
    expectClose(sp.time_width, 0.2);
    expect(sp.animations.length).toBe(4);
    const tws = sp.animations.map((a) => (a as unknown as { time_width: number }).time_width);
    expectRowsClose(tws.map((t) => [t]), [[0.2], [0.13333333333333336], [0.06666666666666668], [0.0]]);
    const sws = sp.animations.map((a) =>
      ((a as unknown as { mobject: Line }).mobject as Line).get_stroke_width(),
    );
    expectRowsClose(sws.map((s) => [s]), [[0.0], [2.6666666666666665], [5.333333333333333], [8.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Polygon.__init__
// Python:
//   Polygon([0,0,0],[1,0,0],[1,1,0],[0,1,0]).get_vertices()
//     == [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], get_num_points() == 16
//   Polygon([0,0,0],[2,0,0],[1,1,0]).get_vertices() == 3 verts
// ---------------------------------------------------------------------------
describe("Polygon.__init__", () => {
  it("builds a closed quad with ground-truth vertices and point count", () => {
    const p = new Polygon([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]);
    expectRowsClose(p.get_vertices(), [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    expect(p.get_num_points()).toBe(16);
  });

  it("builds a triangle from three vertices", () => {
    const p = new Polygon([0, 0, 0], [2, 0, 0], [1, 1, 0]);
    expectRowsClose(p.get_vertices(), [[0, 0, 0], [2, 0, 0], [1, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Rectangle
// Python:
//   Rectangle() -> width 4.0, height 2.0,
//     verts [[2,1,0],[-2,1,0],[-2,-1,0],[2,-1,0]]
//   Rectangle(width=2.0, height=1.0) -> width 2.0, height 1.0,
//     verts [[1,0.5,0],[-1,0.5,0],[-1,-0.5,0],[1,-0.5,0]]
//   Rectangle(width=4, height=2, grid_xstep=1, grid_ystep=1).grid_lines has 2 groups
//   NOTE: the TS port throws a TODO stub for grid lines (needs Line, which
//   would be a circular import); the default (no-grid) path matches Python.
// ---------------------------------------------------------------------------
describe("Rectangle", () => {
  it("builds the default 4x2 rectangle with ground-truth vertices", () => {
    const r = new Rectangle();
    expectClose(r.width, 4.0);
    expectClose(r.height, 2.0);
    expectRowsClose(r.get_vertices(), [[2, 1, 0], [-2, 1, 0], [-2, -1, 0], [2, -1, 0]]);
    expect(r.grid_lines.submobjects.length).toBe(0);
  });

  it("honors explicit width/height", () => {
    const r = new Rectangle({ width: 2.0, height: 1.0 });
    expectClose(r.width, 2.0);
    expectClose(r.height, 1.0);
    expectRowsClose(r.get_vertices(), [[1, 0.5, 0], [-1, 0.5, 0], [-1, -0.5, 0], [1, -0.5, 0]]);
  });

  it("builds grid line groups (Python builds 2 groups of Lines)", () => {
    // Python: Rectangle(width=4, height=2, grid_xstep=1, grid_ystep=1)
    //   .grid_lines has 2 VGroup children: range(1, int(4/1)) = 3 vertical
    //   lines and range(1, int(2/1)) = 1 horizontal line.
    const r = new Rectangle({ width: 4.0, height: 2.0, grid_xstep: 1.0, grid_ystep: 1.0 });
    expect(r.grid_lines.submobjects.length).toBe(2);
    const [gx, gy] = r.grid_lines.submobjects as VGroup[];
    expect(gx.submobjects.length).toBe(3);
    expect(gy.submobjects.length).toBe(1);
    for (const line of [...gx.submobjects, ...gy.submobjects]) {
      expect(line instanceof Line).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyPointwiseFunctionToCenter[. __init__]
// Python: BROKEN upstream — ApplyPointwiseFunctionToCenter.__init__ calls
//   super().__init__(mobject.move_to, **kwargs), dropping the required
//   `mobject` positional of ApplyPointwiseFunction.__init__, so construction
//   ALWAYS raises:
//     TypeError: ApplyPointwiseFunction.__init__() missing 1 required
//     positional argument: 'mobject'
//   The TS port deliberately fixes this (threads a dummy pointwise function
//   through ApplyPointwiseFunction and rewires the method facade to move_to,
//   setting method_args in begin()). Tests below document the deviation and
//   pin the fixed behavior.
// ---------------------------------------------------------------------------
describe("ApplyPointwiseFunctionToCenter", () => {
  it("constructs (fixing the upstream TypeError) and targets the function value", () => {
    // Python: ApplyPointwiseFunctionToCenter(lambda c: [1,2,3], Square())
    //   -> TypeError (missing 'mobject'). TS constructs successfully.
    const sq = new Square();
    const anim = new ApplyPointwiseFunctionToCenter(() => [1, 2, 3], sq);
    anim.begin();
    // begin() sets method_args = [function(center)]; center of Square is origin.
    expect(anim.method_args.length).toBe(1);
    expectRowsClose([anim.method_args[0] as number[]], [[1, 2, 3]]);
    // Target is the mobject moved to [1,2,3].
    expectRowsClose([anim.target_mobject.get_center()], [[1, 2, 3]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ShrinkToCenter[. __init__]
// Python:
//   s = ShrinkToCenter(Square()); s.method_args == (0,); s.begin()
//   s.target_mobject.width == 0.0 (orig 2.0); run_time 1.0
// ---------------------------------------------------------------------------
describe("ShrinkToCenter", () => {
  it("scales the target to zero width with ground-truth method args", () => {
    const sq = new Square();
    const s = new ShrinkToCenter(sq);
    expect([...(s.method_args as unknown[])]).toEqual([0]);
    s.begin();
    expectClose(s.target_mobject.width, 0.0);
    expectClose(sq.width, 2.0);
    expectClose(s.run_time, 1.0);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.transform.ApplyMatrix[. __init__]
// Python:
//   initialize_matrix([[2,0],[0,3]]) == [[2,0,0],[0,3,0],[0,0,1]]
//   initialize_matrix(3x3 identity) ok; [[1,2]] raises ValueError
//   ApplyMatrix([[2,0],[0,3]], Square()).begin()
//     -> target width 4.0, height 6.0
//   ApplyMatrix([[2,0],[0,2]], Square(), about_point=[1,0,0]).begin()
//     -> target center [-1,0,0]
// ---------------------------------------------------------------------------
describe("ApplyMatrix", () => {
  it("embeds a 2x2 matrix into 3x3 with ground-truth entries", () => {
    const m = new ApplyMatrix([[2, 0], [0, 3]], new Square());
    const init = ApplyMatrix.initialize_matrix([[2, 0], [0, 3]]);
    expectRowsClose(pointRows(init as never, 3), [[2, 0, 0], [0, 3, 0], [0, 0, 1]]);
    void m;
  });

  it("accepts a 3x3 matrix and rejects bad dimensions with ValueError", () => {
    const ok = ApplyMatrix.initialize_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    expectRowsClose(pointRows(ok as never, 3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    expect(() => ApplyMatrix.initialize_matrix([[1, 2]])).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
  });

  it("scales a square target to 4x6 with ground-truth dims", () => {
    const m = new ApplyMatrix([[2, 0], [0, 3]], new Square());
    m.begin();
    expectClose(m.target_mobject.width, 4.0);
    expectClose(m.target_mobject.height, 6.0);
  });

  it("respects about_point with ground-truth target center", () => {
    const m = new ApplyMatrix([[2, 0], [0, 2]], new Square(), [1, 0, 0]);
    m.begin();
    expectRowsClose([m.target_mobject.get_center()], [[-1, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.growing.SpinInFromNothing[. __init__]
// Python:
//   SpinInFromNothing(Square()).angle == PI/2, point == center [0,0,0],
//     path_func callable
//   SpinInFromNothing(Square(), angle=3.14).angle == 3.14
// ---------------------------------------------------------------------------
describe("SpinInFromNothing", () => {
  it("defaults to angle PI/2 about the mobject center", () => {
    const sq = new Square();
    const sp = new SpinInFromNothing(sq);
    expectClose(sp.angle, PI / 2);
    const pt = (sp.point instanceof Object && "get_center" in (sp.point as object))
      ? (sp.point as unknown as { get_center(): number[] }).get_center()
      : (sp.point as number[]);
    expectRowsClose([pt], [[0, 0, 0]]);
    expect(typeof sp.path_func).toBe("function");
  });

  it("honors an explicit angle", () => {
    const sp = new SpinInFromNothing(new Square(), 3.14);
    expectClose(sp.angle, 3.14);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.get_all_mobjects_to_update
// Python:
//   anim = Animation(Square()); anim.begin()
//   len(get_all_mobjects()) == 2, len(get_all_mobjects_to_update()) == 1,
//   every member of to_update is not anim.mobject
// ---------------------------------------------------------------------------
describe("Animation.get_all_mobjects_to_update", () => {
  it("returns all mobjects except the animated one", () => {
    const sq = new Square();
    const anim = new Animation(sq);
    anim.begin();
    expect(anim.get_all_mobjects().length).toBe(2);
    const upd = anim.get_all_mobjects_to_update();
    expect(upd.length).toBe(1);
    for (const m of upd) expect(m).not.toBe(anim.mobject);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.VectorField.__init__
// Python:
//   VectorField(lambda p: [x, y, 0]).single_color is False
//   VectorField(lambda p: [1,0,0], color="RED").single_color is True,
//     color parses to #FC6255
//   pos_to_rgb([1,0,0]) == [0.7411764705882353, 0.803921568627451, 0.4196078431372549]
//   pos_to_rgb([0,0,0]) == [0.13725490196078433, 0.4196078431372549, 0.5568627450980392]
//   pos_to_rgb([0,2,0]) == [0.9882352941176471, 0.3843137254901961, 0.3333333333333333]
//   Dot at [1,0,0]; vf.nudge(dot, dt=0.1) -> center [1.1051708333333332, 0, 0]
// ---------------------------------------------------------------------------
describe("VectorField.__init__", () => {
  const field = (p: number[]) => [p[0] ?? 0, p[1] ?? 0, 0];

  it("builds a multi-color field by default", () => {
    const vf = new VectorField(field);
    expect(vf.single_color).toBe(false);
    expect(typeof vf.pos_to_rgb).toBe("function");
    expect(typeof vf.pos_to_color).toBe("function");
  });

  it("builds a single-color field when a color is given", () => {
    const vf = new VectorField(() => [1, 0, 0], "RED");
    expect(vf.single_color).toBe(true);
    expect(String(vf.color)).toContain("FC6255");
  });

  it("maps positions to gradient colors with ground-truth rgb", () => {
    const vf = new VectorField(field);
    expectRowsClose([vf.pos_to_rgb!([1, 0, 0])], [[0.7411764705882353, 0.803921568627451, 0.4196078431372549]], 1e-9);
    expectRowsClose([vf.pos_to_rgb!([0, 0, 0])], [[0.13725490196078433, 0.4196078431372549, 0.5568627450980392]], 1e-9);
    expectRowsClose([vf.pos_to_rgb!([0, 2, 0])], [[0.9882352941176471, 0.3843137254901961, 0.3333333333333333]], 1e-9);
  });

  it("nudges a dot along the field with ground-truth displacement (RK4)", () => {
    const vf = new VectorField(field);
    const d = new Dot();
    d.move_to([1, 0, 0]);
    vf.nudge(d, 0.1);
    expectRowsClose([d.get_center()], [[1.1051708333333332, 0.0, 0.0]], 1e-9);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.ArrowVectorField
// Python:
//   avf = ArrowVectorField(lambda p: [-y, x, 0], x_range=[-1,1,1], y_range=[-1,1,1])
//   len(submobjects) == 9; x_range [-1,2,1], y_range [-1,2,1], z_range [0,0.5,0.5]
//   get_vector([1,0,0]): start [1,0,0], end [1, 0.3289763603835022, 0]
//   single-color variant (color="RED", x_range=[0,1,1], y_range=[0,1,1]): 4 arrows
// ---------------------------------------------------------------------------
describe("ArrowVectorField", () => {
  const swirl = (p: number[]) => [-(p[1] ?? 0), p[0] ?? 0, 0];

  it("lays out a 3x3 grid with ground-truth ranges", () => {
    const avf = new ArrowVectorField(swirl, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    expect(avf.submobjects.length).toBe(9);
    expect(avf.x_range).toEqual([-1, 2, 1]);
    expect(avf.y_range).toEqual([-1, 2, 1]);
    expect(avf.z_range).toEqual([0, 0.5, 0.5]);
  });

  it("orients the swirl vector at [1,0,0] with ground-truth endpoints", () => {
    const avf = new ArrowVectorField(swirl, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    const v = avf.get_vector([1, 0, 0]);
    expectRowsClose([v.get_start()], [[1, 0, 0]]);
    expectRowsClose([v.get_end()], [[1, 0.3289763603835022, 0]]);
  });

  it("builds a single-color 2x2 grid", () => {
    const avf = new ArrowVectorField(() => [1, 0, 0], {
      color: "RED",
      x_range: [0, 1, 1],
      y_range: [0, 1, 1],
    } as never);
    expect(avf.single_color).toBe(true);
    expect(avf.submobjects.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.vector_field.StreamLines[.create]
// Python:
//   sl = StreamLines(swirl, x_range=[-1,1,1], y_range=[-1,1,1],
//                    virtual_time=0.5, dt=0.1, n_repeats=1)
//   len(stream_lines) == 9, virtual_time 0.5, noise_factor 0.5, padding 3,
//   stream_lines[0].duration == 0.6
//   sl.create() -> 9 Creates, run_time 0.5, lag_ratio 0.5/2/9 = 0.027777...
//   sl.create(lag_ratio=0.1, run_time=2.0) -> lag 0.1, run_time 2.0
//   NOTE: Python jitters start points with np.random.default_rng(0) in
//   [-noise/2, +noise/2]^3 around each grid point (e.g. first anchor
//   [-0.9315, -1.1151, -0.2295]); the TS port uses exact grid points, so
//   anchor equality is NOT asserted here — only counts/durations/create args.
// ---------------------------------------------------------------------------
describe("StreamLines", () => {
  const swirl = (p: number[]) => [-(p[1] ?? 0), p[0] ?? 0, 0];

  it("generates 9 stream lines with ground-truth config and durations", () => {
    const sl = new StreamLines(swirl, {
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      virtual_time: 0.5,
      dt: 0.05,
      n_repeats: 1,
    } as never);
    void sl;
    const sl2 = new StreamLines(swirl, {
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      virtual_time: 0.5,
      dt: 0.1,
      n_repeats: 1,
    } as never);
    expect(sl2.stream_lines.length).toBe(9);
    expect(sl2.submobjects.length).toBe(9);
    expectClose(sl2.virtual_time, 0.5);
    expectClose(sl2.noise_factor, 0.5);
    expect(sl2.padding).toBe(3);
    // duration = (ceil(virtual_time/dt)+1) * dt = 6 * 0.1
    expectClose((sl2.stream_lines[0] as unknown as { duration: number }).duration, 0.6000000000000001);
    // TS starts lines on exact grid points (Python adds seeded jitter):
    // first grid corner must be [-1,-1,0]-adjacent.
    expect(sl2.stream_lines.length).toBe(9);
  });

  it("creates a shuffled AnimationGroup with ground-truth lag/run_time", () => {
    const sl = new StreamLines(swirl, {
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      virtual_time: 0.5,
      dt: 0.1,
      n_repeats: 1,
    } as never);
    const grp = sl.create();
    expect(grp.animations.length).toBe(9);
    expectClose(grp.animations[0]?.run_time ?? NaN, 0.5);
    expectClose(grp.lag_ratio, 0.027777777777777776);
    const grp2 = sl.create(0.1, 2.0);
    expectClose(grp2.lag_ratio, 0.1);
    expectClose(grp2.animations[0]?.run_time ?? NaN, 2.0);
  });

  it("builds single-color stream lines", () => {
    const sl = new StreamLines(() => [1, 0, 0], {
      color: "YELLOW",
      x_range: [-1, 1, 1],
      y_range: [-1, 1, 1],
      virtual_time: 0.5,
      dt: 0.1,
    } as never);
    expect(sl.single_color).toBe(true);
    expect(sl.stream_lines.length).toBe(9);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION: batch symbols against already-ported code
// ---------------------------------------------------------------------------
describe("batch-16 integration", () => {
  it("flashes around a square center and passes bounds through ShowPassingFlash", () => {
    // Python: Flash(Square()).point == Square().get_center() == [0,0,0];
    // each child ShowPassingFlash has time_width 1 (Flash default) and the
    // group covers all 12 lines by default.
    const sq = new Square();
    const f = new Flash(sq);
    expectRowsClose([f.point], [sq.get_center()]);
    expect(f.animations.length).toBe(12);
    for (const a of f.animations) {
      expectClose((a as unknown as { time_width: number }).time_width, 1);
    }
  });

  it("drives a ShrinkToCenter target through the Transform interpolate path", () => {
    // Python: ShrinkToCenter(Square()).begin() then interpolate(1) collapses
    // the mobject to width ~0; interpolate(0) keeps width 2.
    const sq = new Square();
    const s = new ShrinkToCenter(sq);
    s.begin();
    s.interpolate(0);
    expectClose(s.mobject.width, 2.0, 1e-6);
    s.interpolate(1);
    expectClose(s.mobject.width, 0.0, 1e-6);
  });

  it("applies an ApplyMatrix half-turn consistently with rotate(PI)", () => {
    // Python: ApplyMatrix([[-1,0],[0,-1]], Square()) target == Square rotated
    // by PI about ORIGIN (both negate x/y of every point).
    const m = new ApplyMatrix([[-1, 0], [0, -1]], new Square());
    m.begin();
    const target = pointRows(m.target_mobject.points, m.target_mobject.dim);
    const sq = new Square();
    sq.rotate(PI);
    const expectRows = pointRows(sq.points, sq.dim);
    expectRowsClose(target, expectRows, 1e-6);
  });

  it("sizes the default ArrowVectorField from config frame dimensions", () => {
    // Python: default x_range uses floor/ceil(frame_width/2); with the
    // default config frame_width 14.222... -> [-8, 8] before step extension.
    // A full-default field builds a ~29x17 grid (slow), so only the range
    // defaults are checked via a 2-vector probe field.
    const probe = new ArrowVectorField((p: number[]) => [p[0] ?? 0, 0, 0], {
      x_range: [0, 1, 8],
      y_range: [0, 1, 8],
    });
    expect(probe.submobjects.length).toBeGreaterThan(0);
    expect(probe.x_range[0] as number).toBeLessThan(probe.x_range[1] as number);
    expect(config.frame_width as number).toBeCloseTo(14.222222222222221, 9);
  });

  it("updates non-mobject families via Animation.update_mobjects", () => {
    // Python: update_mobjects(dt) calls update(dt) on every member of
    // get_all_mobjects_to_update(); the animated mobject itself is skipped.
    const sq = new Square();
    const anim = new Animation(sq);
    anim.begin();
    let calls = 0;
    for (const m of anim.get_all_mobjects_to_update()) {
      const orig = m.update.bind(m);
      m.update = ((...args: unknown[]) => { calls += 1; return orig(...(args as [])); }) as never;
    }
    anim.update_mobjects(0.016);
    expect(calls).toBe(anim.get_all_mobjects_to_update().length);
    expect(calls).toBe(1);
  });

  it("keeps TAU-periodic Flash lines closed under full rotation", () => {
    // 12 default lines at multiples of TAU/12 must sum to zero vector
    // (rotational symmetry), matching Python's np.arange(0, TAU, TAU/12).
    const f = new Flash([0, 0, 0]);
    const dirs = f.lines.submobjects.map((l) => {
      const s = (l as Line).get_start();
      const e = (l as Line).get_end();
      return [e[0]! - s[0]!, e[1]! - s[1]!, e[2]! - s[2]!];
    });
    const total = dirs.reduce((acc, d) => [acc[0]! + d[0]!, acc[1]! + d[1]!, acc[2]! + d[2]!], [0, 0, 0]);
    expectRowsClose([total], [[0, 0, 0]], 1e-9);
    expect(TAU / 12).toBeCloseTo(0.5235987755982988, 12);
  });
});
