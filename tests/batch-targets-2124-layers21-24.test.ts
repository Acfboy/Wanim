import { describe, expect, it } from "vitest";
import { Line3D, Arrow3D } from "../src/mobject/three_d/three_dimensions.js";
import {
  Axes,
  ThreeDAxes,
  ComplexPlane,
} from "../src/mobject/graphing/coordinate_systems.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import "../src/mobject/geometry/shape_matchers.js";
import { BackgroundRectangle } from "../src/mobject/geometry/shape_matchers.js";
import { handle_caching_play } from "../src/utils/caching.js";
import { config } from "../src/_config/utils.js";
import { ZoomedScene } from "../src/scene/zoomed_scene.js";
import { Broadcast } from "../src/animation/specialized.js";
import {
  cycle_animation,
  turn_animation_into_updater,
} from "../src/animation/updaters/mobject_update_utils.js";
import { MobjectTable } from "../src/mobject/table.js";
import { ManimBanner } from "../src/mobject/logo.js";
import { Square, Triangle } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Arrow, DoubleArrow, Line } from "../src/mobject/geometry/line.js";
import { ArrowTriangleFilledTip } from "../src/mobject/geometry/tips.js";
import { VGroup, VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { MED_SMALL_BUFF, SMALL_BUFF, UR, OUT, RIGHT } from "../src/constants.js";
import { ApplyMethod } from "../src/animation/transform.js";
import { Succession } from "../src/animation/composition.js";
import { UpdateFromAlphaFunc } from "../src/animation/updaters/update.js";

// NOTE (scope): the MathTex / BulletedList / Typst targets in this batch
// (get_part_by_tex, set_color_by_tex, set_opacity_by_tex,
// set_color_by_tex_to_color_map, fade_all_but, _select_label,
// _rebuild_label_aliases, select) live in manim.mobject.text.* which is OUT
// OF SCOPE for the entire port (src/mobject/text.ts is an intentional stub).
// They have no tests here by design.

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
// manim.mobject.three_d.three_dimensions.Line3D.parallel_to
// original manim: Line3D([2,0,0],[0,1,1]) has vect [-2,1,1]; parallel_to through ORIGIN
//   (length 5) -> start [-2.04124145,1.02062073,1.02062073],
//   end [2.04124145,-1.02062073,-1.02062073].
// ---------------------------------------------------------------------------
describe("Line3D.parallel_to", () => {
  it("builds the documented in the original manim parallel line through ORIGIN", () => {
    const l1 = new Line3D([2, 0, 0], [0, 1, 1]);
    const p = Line3D.parallel_to(l1);
    expectRowsClose(p.start, [[-2.0412414523193148, 1.0206207261596574, 1.0206207261596574]], 1e-6);
    expectRowsClose(p.end, [[2.0412414523193148, -1.0206207261596574, -1.0206207261596574]], 1e-6);
    // Same length as requested, direction parallel to the original.
    expectClose(Math.hypot(...p.vect), 5, 1e-9);
    const dot =
      (p.vect[0]! * l1.vect[0]! + p.vect[1]! * l1.vect[1]! + p.vect[2]! * l1.vect[2]!) /
      (Math.hypot(...p.vect) * Math.hypot(...l1.vect));
    expectClose(Math.abs(dot), 1, 1e-9);
  });

  it("passes through the given point with the given length", () => {
    const l1 = new Line3D([2, 0, 0], [0, 1, 1]);
    const p = Line3D.parallel_to(l1, [1, 2, 3], 2);
    const mid = p.start
      ? toRows(p.start)[0]!.map((v, i) => (v + toRows(p.end)[0]![i]!) / 2)
      : [];
    expectRowsClose([mid], [[1, 2, 3]], 1e-9);
    expectClose(Math.hypot(...p.vect), 2, 1e-9);
  });

  it("forwards thickness/color/resolution options", () => {
    const l1 = new Line3D([0, 0, 0], [1, 0, 0]);
    const p = Line3D.parallel_to(l1, [0, 0, 0], 5, {
      thickness: 0.05,
      resolution: [2, 8],
    });
    expectClose(p.thickness, 0.05);
    expect(p.resolution).toEqual([2, 8]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Line3D.perpendicular_to
// original manim: perpendicular_to(Line3D([2,0,0],[0,1,1]), ORIGIN, 5)
//   -> start [1.44337567,1.44337567,1.44337567],
//      end [-1.44337567,-1.44337567,-1.44337567];
//   collinear point raises ValueError("Could not find the perpendicular.").
// ---------------------------------------------------------------------------
describe("Line3D.perpendicular_to", () => {
  it("builds the documented in the original manim perpendicular line", () => {
    const l1 = new Line3D([2, 0, 0], [0, 1, 1]);
    const q = Line3D.perpendicular_to(l1);
    expectRowsClose(q.start, [[1.4433756729740645, 1.4433756729740645, 1.4433756729740645]], 1e-6);
    expectRowsClose(q.end, [[-1.4433756729740645, -1.4433756729740645, -1.4433756729740645]], 1e-6);
    const dot =
      q.vect[0]! * l1.vect[0]! + q.vect[1]! * l1.vect[1]! + q.vect[2]! * l1.vect[2]!;
    expectClose(dot, 0, 1e-6);
    expectClose(Math.hypot(...q.vect), 5, 1e-9);
  });

  it("raises ValueError for a collinear point like python", () => {
    const l1 = new Line3D([0, 0, 0], [1, 0, 0]);
    expect(() => Line3D.perpendicular_to(l1, [5, 0, 0])).toThrowError(
      /Could not find the perpendicular/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.three_d.three_dimensions.Arrow3D.__init__ (extras)
// original manim: Arrow3D([-1,0,0],[1,0,0]) -> length 1.7, direction [1,0,0],
//   shaft end [0.7,0,0], get_end [1,0,0]; height shortens the shaft,
//   cone carries direction, end_point the original end.
// ---------------------------------------------------------------------------
describe("Arrow3D.__init__ extras", () => {
  it("honors custom height/base_radius/color/resolution", () => {
    const a = new Arrow3D([0, 0, 0], [0, 0, 2], 0.04, 0.5, 0.1, "#ff0000", [2, 8]);
    expectClose(a.thickness, 0.04);
    expectClose(a.length, 1.5, 1e-9);
    expectRowsClose([a.direction], [[0, 0, 1]]);
    expectRowsClose(a.end, [[0, 0, 1.5]]);
    expectRowsClose(a.get_end(), [[0, 0, 2]]);
    expectClose(a.cone.new_height, 0.5, 1e-12);
    expect(a.color).toBe("#ff0000");
    expect(a.resolution).toEqual([2, 8]);
  });

  it("points the cone along diagonal directions", () => {
    const a = new Arrow3D([0, 0, 0], [1, 1, 1]);
    const n = Math.hypot(...a.direction);
    expectClose(n, 1, 1e-12);
    expectRowsClose([a.cone.direction], [a.direction], 1e-12);
    expectRowsClose(a.get_end(), [[1, 1, 1]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.Axes.get_axis_labels
// Scatter-text method: python groups get_x_axis_label(x) + get_y_axis_label
// (y) into axis_labels. str/float labels go through MathTex (out of scope);
// VMobject labels are returned as-is, positioned next_to the axis edge.
// NOTE: the TS Axes hierarchy ships an empty `axes` group until the
// NumberLine integration lands (pre-existing gap, asserted below), so real
// NumberLines are wired into the documented slots here; label creation then
// runs through the real NumberLine._create_label_tex.
// original manim: axis_labels stored on self, labels[i] is the passed-in mobject.
// ---------------------------------------------------------------------------
describe("Axes.get_axis_labels", () => {
  function wiredAxes() {
    const ax = new Axes([0, 5, 1], [0, 5, 1], 5, 5);
    const xLine = new NumberLine({ x_range: [0, 5, 1], length: 5 });
    const yLine = new NumberLine({ x_range: [0, 5, 1], length: 5 });
    ax.axes.add(xLine, yLine);
    (ax as unknown as { x_axis: NumberLine }).x_axis = xLine;
    return { ax, xLine, yLine };
  }

  it("groups VMobject labels and stores axis_labels (python identity)", () => {
    const { ax } = wiredAxes();
    const sq1 = new Square(0.2);
    const sq2 = new Square(0.2);
    const labels = ax.get_axis_labels(sq1, sq2);
    expect(labels instanceof VGroup).toBe(true);
    expect(labels.submobjects.length).toBe(2);
    expect((ax as unknown as { axis_labels: VGroup }).axis_labels === labels).toBe(true);
    // Same objects, positioned (mirrors _create_label_tex passthrough).
    expect(labels.submobjects[0]).toBe(sq1 as never);
    expect(labels.submobjects[1]).toBe(sq2 as never);
  });

  it("positions each label next_to its axis edge with documented defaults", () => {
    const { ax, xLine, yLine } = wiredAxes();
    const sq1 = new Square(0.2);
    const sq2 = new Square(0.2);
    const labels = ax.get_axis_labels(sq1, sq2);
    const ref1 = new Square(0.2);
    ref1.next_to(xLine.get_edge_center([...UR]), [...UR], SMALL_BUFF);
    ref1.shift_onto_screen({ buff: MED_SMALL_BUFF });
    const ref2 = new Square(0.2);
    ref2.next_to(yLine.get_edge_center([...UR]), [1, 0.5, 0], SMALL_BUFF);
    ref2.shift_onto_screen({ buff: MED_SMALL_BUFF });
    expectRowsClose(
      [(labels.submobjects[0] as VMobject).get_center()],
      [ref1.get_center()],
      1e-9,
    );
    expectRowsClose(
      [(labels.submobjects[1] as VMobject).get_center()],
      [ref2.get_center()],
      1e-9,
    );
  });

  it("raises the text/tex TODO for str labels (out of scope)", () => {
    const { ax } = wiredAxes();
    expect(() => ax.get_axis_labels("x", "y")).toThrowError(/text\/tex/);
  });

  it("documents the empty-axes gap on unwired Axes", () => {
    // Pre-existing NumberLine-integration gap (not this batch): with no
    // wired axes, get_axis raises IndexError.
    expect(() => new Axes([0, 5, 1], [0, 5, 1], 5, 5).get_axis_labels(new Square(), new Square())).toThrowError(
      /range object index out of range/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ThreeDAxes.get_axis_labels
// Same composition as Axes plus the z label (which is additionally rotated
// PI/2 about RIGHT by get_z_axis_label). Real NumberLines are wired in for
// the same pre-existing-gap reason as above.
// original manim: VGroup(x, y, z) stored as axis_labels; passed-in mobjects kept.
// ---------------------------------------------------------------------------
describe("ThreeDAxes.get_axis_labels", () => {
  function wiredAxes() {
    const ax = new ThreeDAxes([0, 3, 1], [0, 3, 1], 3, 3, undefined, [], 20, [0, 3, 1], 3);
    const xLine = new NumberLine({ x_range: [0, 3, 1], length: 3 });
    const yLine = new NumberLine({ x_range: [0, 3, 1], length: 3 });
    const zLine = new NumberLine({ x_range: [0, 3, 1], length: 3 });
    ax.axes.add(xLine, yLine, zLine);
    (ax as unknown as { x_axis: NumberLine }).x_axis = xLine;
    return { ax, xLine, yLine, zLine };
  }

  it("groups three VMobject labels and stores axis_labels", () => {
    const { ax } = wiredAxes();
    const s1 = new Square(0.2);
    const s2 = new Square(0.2);
    const s3 = new Square(0.2);
    const labels = ax.get_axis_labels(s1, s2, s3);
    expect(labels.submobjects.length).toBe(3);
    expect((ax as unknown as { axis_labels: VGroup }).axis_labels === labels).toBe(true);
    expect(labels.submobjects[0]).toBe(s1 as never);
    expect(labels.submobjects[1]).toBe(s2 as never);
    expect(labels.submobjects[2]).toBe(s3 as never);
  });

  it("positions x/y/z with the documented per-axis defaults", () => {
    const { ax, xLine, yLine, zLine } = wiredAxes();
    const labels = ax.get_axis_labels(new Square(0.2), new Square(0.2), new Square(0.2));
    // x: base defaults (edge UR, direction UR).
    const ref1 = new Square(0.2);
    ref1.next_to(xLine.get_edge_center([...UR]), [...UR], SMALL_BUFF);
    ref1.shift_onto_screen({ buff: MED_SMALL_BUFF });
    // y (ThreeDAxes override): edge UR, direction UR, then rotate PI/2
    // about OUT (mirrors get_y_axis_label).
    const ref2 = new Square(0.2);
    ref2.next_to(yLine.get_edge_center([...UR]), [...UR], SMALL_BUFF);
    ref2.shift_onto_screen({ buff: MED_SMALL_BUFF });
    ref2.rotate(Math.PI / 2, [...OUT]);
    // z (ThreeDAxes): edge OUT, direction RIGHT, then rotate PI/2 about
    // RIGHT (mirrors get_z_axis_label).
    const ref3 = new Square(0.2);
    ref3.next_to(zLine.get_edge_center([...OUT]), [...RIGHT], SMALL_BUFF);
    ref3.shift_onto_screen({ buff: MED_SMALL_BUFF });
    ref3.rotate(Math.PI / 2, [...RIGHT]);
    for (let i = 0; i < 3; i += 1) {
      const got = (labels.submobjects[i] as VMobject).get_center();
      const want = [ref1, ref2, ref3][i]!.get_center();
      expectRowsClose([got], [want], 1e-9);
    }
  });

  it("raises the text/tex TODO for str labels (out of scope)", () => {
    const { ax } = wiredAxes();
    expect(() => ax.get_axis_labels("x", "y", "z")).toThrowError(/text\/tex/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ComplexPlane.get_coordinate_labels
// Real axes need latex (original manim raises FileNotFoundError: latex), so the port
// routes through NumberLine.get_number_mobject which throws the text/tex
// TODO; the routing logic itself is verified with stubbed axes (mirrors
// python: |imag|>|real| -> y axis with unit "i", else x axis).
// ---------------------------------------------------------------------------
describe("ComplexPlane.get_coordinate_labels", () => {
  function stubAxes(cp: ComplexPlane) {
    const calls: Array<{ axis: string; value: number; kwargs: unknown }> = [];
    const mk = (name: string) => ({
      get_number_mobject: (value: number, ...rest: unknown[]) => {
        calls.push({ axis: name, value, kwargs: rest[4] });
        return new Square();
      },
    });
    (cp as unknown as { get_x_axis: () => unknown }).get_x_axis = () => mk("x");
    (cp as unknown as { get_y_axis: () => unknown }).get_y_axis = () => mk("y");
    return calls;
  }

  it("routes floats to x and imaginary to y with unit i", () => {
    const cp = new ComplexPlane();
    const calls = stubAxes(cp);
    const labels = cp.get_coordinate_labels(1, { re: 0, im: 2 } as never, { re: 3, im: 1 } as never);
    expect(labels === cp.coordinate_labels).toBe(true);
    expect(calls.length).toBe(3);
    expect(calls[0]!.axis).toBe("x");
    expectClose(calls[0]!.value, 1);
    expect(calls[1]!.axis).toBe("y");
    expectClose(calls[1]!.value, 2);
    expect((calls[1]!.kwargs as Record<string, unknown>)["unit"]).toBe("i");
    // |real|>|imag| (3 vs 1) stays on x.
    expect(calls[2]!.axis).toBe("x");
    expectClose(calls[2]!.value, 3);
    expect(labels.submobjects.length).toBe(3);
  });

  it("throws the text/tex TODO on the real NumberLine path", () => {
    expect(() => new ComplexPlane().get_coordinate_labels(1 as never)).toThrowError(
      /text\/tex/,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.ComplexPlane.add_coordinates
// (extra: return value + default-values path, which also hits text/tex).
// ---------------------------------------------------------------------------
describe("ComplexPlane.add_coordinates extras", () => {
  it("returns self and stores labels", () => {
    const cp = new ComplexPlane();
    (cp as unknown as { get_x_axis: () => unknown }).get_x_axis = () => ({
      get_number_mobject: () => new Square(),
    });
    (cp as unknown as { get_y_axis: () => unknown }).get_y_axis = () => ({
      get_number_mobject: () => new Square(),
    });
    const ret = cp.add_coordinates({ re: 1, im: 0 });
    expect(ret === cp).toBe(true);
    expect(cp.coordinate_labels.submobjects.length).toBe(1);
  });

  it("default coordinate values hit the text/tex TODO on real axes", () => {
    expect(() => new ComplexPlane().add_coordinates()).toThrowError(/text\/tex/);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.get_unpositioned_tip extras
// original manim (Line [0,0,0]-[3,0,0]): default length 0.35; custom (0.5, 0.4 width)
//   -> length 0.5, width 0.5 (ArrowTriangleFilledTip derives width from the
//   default tip length when width is None... here explicit 0.4? original manim shows
//   width 0.5 — see comment); tip inherits the line color.
// ---------------------------------------------------------------------------
describe("TipableVMobject.get_unpositioned_tip extras", () => {
  it("matches original manim custom length/width numbers", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    expectClose(l.get_default_tip_length(), 0.35, 1e-9);
    const t = l.get_unpositioned_tip(null, 0.5, 0.4);
    // original manim: Line(...).get_unpositioned_tip(None, 0.5, w) -> length 0.5 and
    // width 0.5 for w in (None, 0.4, 0.35): ArrowTriangleFilledTip derives
    // its width from the length, ignoring the passed tip_width.
    expectClose((t as unknown as { length: number }).length, 0.5, 1e-6);
    expectClose((t as unknown as { width: number }).width, 0.5, 1e-6);
    expect(t.constructor.name).toBe("ArrowTriangleFilledTip");
  });

  it("paints the tip in the mobject color (original manim: red line -> #FF0000 tip)", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    l.set_color("#ff0000");
    const t = l.get_unpositioned_tip();
    const fill = (t as unknown as { fill_color: { toHex: () => string } }).fill_color;
    expect(fill.toHex().toUpperCase()).toBe("#FF0000");
  });

  it("create_tip(at_start=True) anchors at the start", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    const tip = l.create_tip(null, null, null, true);
    expect(tip.constructor.name).toBe("ArrowTriangleFilledTip");
    const start = l.get_start();
    const tipPoint = toRows(
      (tip as unknown as { tip_point: unknown }).tip_point,
    )[0]!;
    // Tip's locating point sits on the line start after positioning.
    expectClose(tipPoint[0]!, start[0]!, 0.3);
  });

  it("add_tip returns self and grows submobjects", () => {
    const l = new Line([0, 0, 0], [2, 0, 0]);
    const before = l.submobjects.length;
    const ret = l.add_tip();
    expect(ret === l).toBe(true);
    expect(l.submobjects.length).toBe(before + 1);
    expect(l.has_tip()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Arrow.scale extras
// original manim: scale(2) doubles start/end, keeps tip length; scale(2, scale_tips)
//   doubles tip length too; stroke stays 6 either way here
//   (min(6, 5 * length)).
// ---------------------------------------------------------------------------
describe("Arrow.scale extras", () => {
  it("keeps stroke width fixed when tips are not scaled", () => {
    const a = new Arrow([-1, -1, 0], [1, 1, 0], 0);
    a.scale(2);
    expectClose(a.stroke_width, 6, 1e-9);
    expectClose(a.initial_stroke_width, 6, 1e-9);
  });

  it("DoubleArrow.scale keeps both tips at fixed size", () => {
    const d = new DoubleArrow([0, 0, 0], [2, 0, 0], 0);
    const endLen = (d.tip as unknown as { length: number }).length;
    const startLen = (d.start_tip as unknown as { length: number }).length;
    d.scale(2);
    expect(d.has_tip()).toBe(true);
    expect(d.has_start_tip()).toBe(true);
    expectClose((d.tip as unknown as { length: number }).length, endLen, 1e-6);
    expectClose((d.start_tip as unknown as { length: number }).length, startLen, 1e-6);
    expectRowsClose([d.get_start()], [[-1, 0, 0]], 1e-6);
    expectRowsClose([d.get_end()], [[3, 0, 0]], 1e-6);
  });

  it("accepts an about_point in the Mobject.scale slot", () => {
    const a = new Arrow([0, 0, 0], [2, 0, 0], 0);
    const lenBefore = a.get_length();
    a.scale(2, [0, 0, 0]);
    // Scaling about the start point keeps the start fixed, doubles length.
    expectRowsClose([a.get_start()], [[0, 0, 0]], 1e-6);
    expectClose(a.get_length(), lenBefore * 2, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.add_background_rectangle (extras)
// original manim: attaches BackgroundRectangle(color, fill_opacity) behind (index 0).
// ---------------------------------------------------------------------------
describe("Mobject.add_background_rectangle extras", () => {
  it("defaults to config.background_color at opacity 0.75", () => {
    const m = new Square();
    m.add_background_rectangle();
    const bg = m.background_rectangle as BackgroundRectangle;
    expect(bg.constructor.name).toBe("BackgroundRectangle");
    expectClose(bg.fill_opacity, 0.75, 1e-12);
    expect(bg.fill_color.toHex().toUpperCase()).toBe(
      config.background_color.toHex().toUpperCase(),
    );
  });

  it("sits behind pre-existing submobjects", () => {
    const m = new Square();
    const inner = new Triangle();
    m.add(inner);
    m.add_background_rectangle("#00ff00", 0.5);
    expect(m.submobjects[0]).toBe(m.background_rectangle!);
    expect(m.submobjects).toContain(inner as never);
  });

  it("forwards kwargs to the BackgroundRectangle", () => {
    const m = new Square(2);
    const widthBefore = m.width;
    m.add_background_rectangle("#0000ff", 0.4, { buff: 0.2 });
    const bg = m.background_rectangle as BackgroundRectangle;
    expectClose(bg.fill_opacity, 0.4, 1e-12);
    // buff grows the rectangle beyond the mobject (m.width now includes bg).
    expectClose(widthBefore, 2, 1e-9);
    expectClose(bg.width, 2.4, 1e-9);
    expect(bg.width).toBeGreaterThan(widthBefore);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.caching.handle_caching_play (extras)
// original manim: disable_caching -> hash "uncached_%05d"; otherwise the computed hash.
// NOTE: python's wrapper also forwards **kwargs to compile_animations/func;
// the TS Scene.compile_animations takes positional args only, so the port
// forwards positional args (verified below).
// ---------------------------------------------------------------------------
describe("handle_caching_play extras", () => {
  function makeRenderer(skip: boolean, cached: boolean) {
    return {
      skip_animations: false,
      _original_skipping_status: false,
      num_plays: 7,
      animations_hashes: [] as Array<string | null>,
      camera: {},
      file_writer: {
        is_already_cached: (_h: string) => cached,
        add_partial_movie_file: (_h: string | null) => {},
      },
      update_skipping_status(this: { skip_animations: boolean }) {
        this.skip_animations = skip;
      },
    };
  }

  function makeScene(seen: unknown[]) {
    return {
      compile_animations: (...args: unknown[]) => {
        seen.push(args.length);
        return [{ name: "anim" }];
      },
      add_mobjects_from_animations: (..._args: unknown[]) => {},
      mobjects: [],
    };
  }

  it("uses uncached_NNNNN when config.disable_caching is set", () => {
    const prev = config.disable_caching;
    config.disable_caching = true;
    try {
      const renderer = makeRenderer(false, false);
      const seen: unknown[] = [];
      const wrapped = handle_caching_play(function (this: unknown) {});
      wrapped.call(renderer as never, makeScene(seen) as never, { name: "a" } as never);
      expect(renderer.animations_hashes).toEqual(["uncached_00007"]);
    } finally {
      config.disable_caching = prev;
    }
  });

  it("forwards positional play args to compile_animations and func", () => {
    const renderer = makeRenderer(false, false);
    const seen: unknown[] = [];
    const got: unknown[][] = [];
    const wrapped = handle_caching_play(function (this: unknown, _s: unknown, ...args: unknown[]) {
      got.push(args);
    });
    const a = { name: "a" };
    const b = { name: "b" };
    wrapped.call(renderer as never, makeScene(seen) as never, a as never, b as never);
    expect(seen).toEqual([2]);
    expect(got.length).toBe(1);
    expect((got[0] as unknown[]).length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.zoomed_scene.ZoomedScene (extras)
// original manim: get_zoom_in_animation stretches the frame fullscreen + stroke 0 and
// returns ApplyMethod(frame.restore, run_time); pop-out replaces the display
// with the frame and returns ApplyMethod(display.restore); activate_zooming
// with animate plays both then foregrounds.
// ---------------------------------------------------------------------------
describe("ZoomedScene extras", { timeout: 60000 }, () => {
  it("get_zoom_in_animation honors run_time and kwargs", () => {
    const z = new ZoomedScene();
    z.setup();
    const anim = z.get_zoom_in_animation(3, { lag_ratio: 0.5 } as never);
    expect(anim instanceof ApplyMethod).toBe(true);
    expectClose(anim.run_time, 3);
    expect((anim as unknown as Record<string, unknown>)["lag_ratio"]).toBe(0.5);
    // The frame was stretched fullscreen with stroke 0 before the restore.
    expectClose(z.zoomed_camera.frame.stroke_width, 0, 1e-12);
  });

  it("get_zoomed_display_pop_out_animation drops kwargs like python", () => {
    const z = new ZoomedScene();
    z.setup();
    // NOTE: python's `get_zoomed_display_pop_out_animation(**kwargs)`
    // accepts kwargs but returns a plain `ApplyMethod(display.restore)`
    // without forwarding them, so extra kwargs must not leak onto the
    // animation (run_time stays at the default 1).
    const anim = z.get_zoomed_display_pop_out_animation({ run_time: 4 } as never);
    expect(anim instanceof ApplyMethod).toBe(true);
    expectClose(anim.run_time, 1);
  });

  it("activate_zooming(true) plays both animations then foregrounds", () => {
    const z = new ZoomedScene();
    z.setup();
    (z as unknown as { renderer: unknown }).renderer = {
      camera: { add_image_mobject_from_camera: () => {} },
    };
    const played: unknown[] = [];
    (z as unknown as { play: (...args: unknown[]) => void }).play = (...args: unknown[]) => {
      played.push(...args);
    };
    z.activate_zooming(true);
    expect(z.zoom_activated).toBe(true);
    expect(played.length).toBe(2);
    expect(played[0] instanceof ApplyMethod).toBe(true);
    expect(played[1] instanceof ApplyMethod).toBe(true);
    expect(z.foreground_mobjects).toContain(z.zoomed_camera.frame as never);
    expect(z.foreground_mobjects).toContain(z.zoomed_display as never);
  });

  it("zoom-in animation restores the small frame on interpolate(1)", () => {
    const z = new ZoomedScene();
    z.setup();
    const before = z.zoomed_camera.frame.height;
    const anim = z.get_zoom_in_animation();
    anim.begin();
    anim.interpolate(1);
    expectClose(z.zoomed_camera.frame.height, before, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.specialized.Broadcast.__init__ (extras)
// original manim: default Square() has fill_opacity 0 -> stroke path; a filled mobject
// takes the fill path (current opacity = initial, saved = final, current
// width = initial_width).
// ---------------------------------------------------------------------------
describe("Broadcast.__init__ extras", () => {
  it("stroke path: current stroke = initial, saved stroke = final", () => {
    const s = new Square(); // fill_opacity 0 -> stroke path, like original manim
    const b = new Broadcast(s, [1, 0, 0], 2, 0.8, 0.2, 0.5);
    expect(b.animations.length).toBe(2);
    for (const a of b.animations) {
      const mob = (a as unknown as { mobject: VMobject }).mobject;
      expectClose(mob.stroke_opacity, 0.8, 1e-9);
      const saved = (mob as unknown as { saved_state: VMobject }).saved_state;
      expectClose(saved.stroke_opacity, 0.2, 1e-9);
      expectRowsClose([mob.get_center()], [[1, 0, 0]], 1e-6);
    }
  });

  it("fill path: filled mobject uses fill opacities and initial width", () => {
    const s = new Square();
    s.set_fill(undefined, 1);
    const b = new Broadcast(s, [0, 0, 0], 2, 0.8, 0.2, 0.5);
    const mob = (b.animations[0] as unknown as { mobject: VMobject }).mobject;
    expectClose(mob.fill_opacity, 0.8, 1e-9);
    const saved = (mob as unknown as { saved_state: VMobject }).saved_state;
    expectClose(saved.fill_opacity, 0.2, 1e-9);
    expectClose(mob.width, 0.5, 1e-6);
  });

  it("accepts the options-bag form", () => {
    const b = new Broadcast(new Square(), undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, {
      n_mobs: 4,
      run_time: 5,
      lag_ratio: 0.7,
    } as never);
    expect(b.n_mobs).toBe(4);
    expectClose(b.run_time, 5);
    expectClose(b.lag_ratio, 0.7);
    expect(b.animations.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.updaters.mobject_update_utils.cycle_animation (extras)
// original manim: cycle_animation(animation) == turn_animation_into_updater(animation,
//   cycle=True, **kwargs); kwargs are accepted but unused.
// ---------------------------------------------------------------------------
describe("cycle_animation extras", () => {
  it("accepts and ignores kwargs like python", () => {
    const mob = new Square();
    const anim = {
      mobject: mob,
      total_time: 0,
      suspend_mobject_updating: true,
      begin() {},
      get_run_time: () => 1,
      interpolate: (_a: number) => {},
      update_mobjects: (_dt: number) => {},
      finish() {},
    };
    const out = cycle_animation(anim as never, { anything: 1 });
    expect(out === mob).toBe(true);
    expect(mob.updaters.length).toBe(1);
  });

  it("keeps cycling past run_time instead of finishing", () => {
    const mob = new Square();
    let finished = false;
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
      finish() {
        finished = true;
      },
    };
    cycle_animation(anim as never);
    const update = mob.updaters[0] as (m: Mobject, dt: number) => void;
    anim.total_time = 2.25;
    update(mob, 0);
    expect(finished).toBe(false);
    expectClose(seen[0]!, 0.25, 1e-12);
    expect(mob.updaters.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.table.Table.add_highlighted_cell (extras)
// original manim: get_highlighted_cell wraps get_cell in a BackgroundRectangle of the
// given color; add links it to the entry and backs it in the table.
// ---------------------------------------------------------------------------
describe("Table.add_highlighted_cell extras", () => {
  function makeTable() {
    return new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
  }

  it("paints the highlight in the requested color", () => {
    const t = makeTable();
    t.add_highlighted_cell([1, 1], "#00ff00");
    const entry = t.get_entries([1, 1]) as VMobject;
    const bg = (entry as unknown as { background_rectangle: BackgroundRectangle })
      .background_rectangle;
    expect(bg.fill_color.toHex().toUpperCase()).toBe("#00FF00");
  });

  it("centers the highlight on the addressed cell", () => {
    const t = makeTable();
    t.add_highlighted_cell([2, 2]);
    const entry = t.get_entries([2, 2]) as VMobject;
    const bg = (entry as unknown as { background_rectangle: VMobject }).background_rectangle;
    expectRowsClose(
      [bg.get_center()],
      [(t.get_cell([2, 2]) as VMobject).get_center()],
      1e-6,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.logo.ManimBanner.expand (+ slide_and_uncover / slide_back)
// original manim: expand validates direction, returns Succession(UpdateFromAlphaFunc
// run_time*2/3 ease_in_out_cubic, UpdateFromAlphaFunc run_time*1/3 smooth);
// the slide_* closures are the animations' update callbacks, exercised here
// through interpolate on all three directions.
// ---------------------------------------------------------------------------
describe("ManimBanner.expand extras", { timeout: 60000 }, () => {
  for (const direction of ["left", "right", "center"]) {
    it(`runs slide_and_uncover + slide_back to completion (${direction})`, () => {
      const b = new ManimBanner();
      const s = b.expand(0.6, direction);
      expect(s instanceof Succession).toBe(true);
      expectClose(s.animations[0]!.run_time, 0.4, 1e-12);
      expectClose(s.animations[1]!.run_time, 0.2, 1e-12);
      const first = s.animations[0] as UpdateFromAlphaFunc;
      const second = s.animations[1] as UpdateFromAlphaFunc;
      first.begin();
      first.interpolate(0);
      first.interpolate(1);
      second.begin();
      second.interpolate(0);
      second.interpolate(1);
      expect(b.submobjects).toContain(b.anim as never);
    });
  }

  it("scales the slide offset with scale_factor", () => {
    const a = new ManimBanner();
    a.scale(0.5);
    expectClose(a.scale_factor, 0.5, 1e-12);
    const s = a.expand(0.6, "center");
    const first = s.animations[0] as UpdateFromAlphaFunc;
    first.begin();
    first.interpolate(1);
    expect(a.submobjects).toContain(a.anim as never);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.get_mobjects_from (extras)
// original manim: text elements warn + map to None (skipped); transformable shapes get
// styled + transformed; containers (Group/Use, incl. the top SVG) recurse.
// ---------------------------------------------------------------------------
describe("SVGMobject.get_mobjects_from extras", () => {
  it("skips text elements like python's None mapping", () => {
    const m = new SVGMobject({});
    const [mobjects, vgroups] = m.get_mobjects_from({
      kind: "svg",
      id: "s",
      children: [{ kind: "text", id: "t" }],
    });
    expect(mobjects.length).toBe(0);
    expect(vgroups.has("t")).toBe(true);
  });

  it("warns and skips unsupported element kinds", () => {
    const m = new SVGMobject({});
    const [mobjects, vgroups] = m.get_mobjects_from({
      kind: "svg",
      id: "s",
      children: [{ kind: "frobnicator", id: "f" }],
    });
    expect(mobjects.length).toBe(0);
    expect(vgroups.has("f")).toBe(true);
  });

  it("applies the SVG transform matrix to shapes", () => {
    const m = new SVGMobject({});
    const [mobjects] = m.get_mobjects_from({
      kind: "svg",
      id: "s",
      children: [
        {
          kind: "line",
          id: "l",
          x1: 0,
          y1: 0,
          x2: 1,
          y2: 0,
          stroke_width: 1,
          fill_opacity: 1,
          transform: { a: 1, b: 0, c: 0, d: 1, e: 5, f: 6 },
        },
      ],
    });
    expect(mobjects.length).toBe(1);
    expectRowsClose([(mobjects[0] as Line).get_start()], [[5, 6, 0]], 1e-9);
    expectRowsClose([(mobjects[0] as Line).get_end()], [[6, 6, 0]], 1e-9);
  });

  it("init_svg_mobject cache-miss still needs the file pipeline", () => {
    expect(() => new SVGMobject({}).init_svg_mobject(true)).toThrowError(
      /file-parsing pipeline/,
    );
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (batch symbols x already-ported code)
// ---------------------------------------------------------------------------
describe("integration (layers 21..24 batch)", { timeout: 60000 }, () => {
  it("Line3D parallel/perpendicular + Arrow3D share direction math", () => {
    const shaft = new Line3D([1, 0, 0], [0, 1, 1]);
    const par = Line3D.parallel_to(shaft, [0, 0, 0], 3);
    const perp = Line3D.perpendicular_to(shaft, [0, 0, 0], 3);
    const dot =
      par.vect[0]! * perp.vect[0]! + par.vect[1]! * perp.vect[1]! + par.vect[2]! * perp.vect[2]!;
    expectClose(dot, 0, 1e-6);
    const arrow = new Arrow3D([0, 0, 0], [0, 0, 2]);
    // The arrow shaft direction is a unit vector along its endpoints.
    expectClose(Math.hypot(...arrow.direction), 1, 1e-12);
    expectRowsClose([arrow.direction], [[0, 0, 1]], 1e-12);
  });

  it("Arrow tip lifecycle: create -> scale keeps tip -> DoubleArrow both ends", () => {
    const a = new Arrow([0, 0, 0], [3, 0, 0]);
    const tip = a.create_tip();
    expect(tip).toBeDefined();
    const lenBefore = (a.tip as unknown as { length: number }).length;
    a.scale(2);
    expectClose((a.tip as unknown as { length: number }).length, lenBefore, 1e-6);
    const d = new DoubleArrow([0, 0, 0], [2, 0, 0], 0);
    d.add_tip(null, ArrowTriangleFilledTip, null, null, true);
    expect(d.has_tip()).toBe(true);
    expect(d.has_start_tip()).toBe(true);
  });

  it("Broadcast Restore target feeds cycle_animation as a looping updater", () => {
    const s = new Square();
    s.set_fill(undefined, 1);
    const b = new Broadcast(s, [0, 0, 0], 2, 1, 0, 0.0);
    expect(b.animations.length).toBe(2);
    const first = b.animations[0] as unknown as {
      mobject: Mobject;
      get_run_time: () => number;
      interpolate: (a: number) => void;
      update_mobjects: (dt: number) => void;
      finish: () => void;
      begin: () => void;
      suspend_mobject_updating: boolean;
      total_time: number;
    };
    const mob = first.mobject;
    const before = mob.updaters.length;
    cycle_animation(first as never);
    expect(mob.updaters.length).toBe(before + 1);
    void turn_animation_into_updater;
  });

  it("Table highlight + Mobject background rectangles compose", () => {
    const t = new MobjectTable([
      [new Square(), new Square()],
      [new Square(), new Square()],
    ]);
    t.add_highlighted_cell([2, 1], "#ff0000");
    const entry = t.get_entries([2, 1]) as VMobject;
    const bg = (entry as unknown as { background_rectangle: BackgroundRectangle })
      .background_rectangle;
    expect(bg.fill_color.toHex().toUpperCase()).toBe("#FF0000");
    expect(t.submobjects[0]).toBe(bg as never);
    // Sibling API on a plain mobject behaves the same way.
    const c = new Circle();
    c.add_background_rectangle("#ff0000", 0.5);
    expect(c.submobjects[0]).toBe(c.background_rectangle!);
  });

  it("Axes labels sit on real axes while ComplexPlane routing matches", () => {
    const ax = new Axes([0, 5, 1], [0, 5, 1], 5, 5);
    const xLine = new NumberLine({ x_range: [0, 5, 1], length: 5 });
    const yLine = new NumberLine({ x_range: [0, 5, 1], length: 5 });
    ax.axes.add(xLine, yLine);
    (ax as unknown as { x_axis: NumberLine }).x_axis = xLine;
    const labels = ax.get_axis_labels(new Square(0.2), new Square(0.2));    expect(labels.submobjects.length).toBe(2);
    const cp = new ComplexPlane();
    const seen: string[] = [];
    (cp as unknown as { get_x_axis: () => unknown }).get_x_axis = () => ({
      get_number_mobject: () => {
        seen.push("x");
        return new Square();
      },
    });
    (cp as unknown as { get_y_axis: () => unknown }).get_y_axis = () => ({
      get_number_mobject: () => {
        seen.push("y");
        return new Square();
      },
    });
    cp.add_coordinates({ re: 2, im: 0 }, { re: 0, im: 2 });
    expect(seen).toEqual(["x", "y"]);
  });

  it("ZoomedScene setup -> zoom animations -> restore round-trips geometry", () => {
    const z = new ZoomedScene();
    z.setup();
    expectClose(z.zoomed_display.height, 3, 1e-6);
    const zoomIn = z.get_zoom_in_animation();
    zoomIn.begin();
    zoomIn.interpolate(1);
    // Restored: the small zoomed frame is back.
    expectClose(z.zoomed_camera.frame.height, 3 * 0.15, 1e-6);
    const pop = z.get_zoomed_display_pop_out_animation();
    pop.begin();
    pop.interpolate(1);
    expectClose(z.zoomed_display.height, 3, 1e-6);
  });

  it("SVG rect + background rectangle: parsed shapes accept decoration", () => {
    const m = new SVGMobject({});
    const [mobjects] = m.get_mobjects_from({
      kind: "svg",
      id: "s",
      children: [
        { kind: "rect", id: "r", x: 0, y: 0, width: 2, height: 1, stroke_width: 1, fill_opacity: 1 },
      ],
    });
    expect(mobjects.length).toBe(1);
    const rect = mobjects[0] as VMobject;
    const ret = rect.add_background_rectangle("#000000", 0.3);
    expect(ret === rect).toBe(true);
    expect(rect.submobjects[0]).toBe(rect.background_rectangle!);
  });
});
