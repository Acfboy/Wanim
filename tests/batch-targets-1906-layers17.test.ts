import { describe, expect, it } from "vitest";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import { Triangle, RegularPolygon, RegularPolygram } from "../src/mobject/geometry/polygram.js";
import { Line, Arrow } from "../src/mobject/geometry/line.js";
import { Arc } from "../src/mobject/geometry/arc.js";
import { ArrowTriangleFilledTip, ArrowTriangleTip } from "../src/mobject/geometry/tips.js";
import { CoordinateSystem, Axes, NumberPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { TAU } from "../src/constants.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1e-12, Math.abs(b));
  return Math.abs(a - b) <= Math.max(1e-12, denom * tol);
}

function toNums(v: unknown): number[] {
  if (Array.isArray(v)) {
    // may be nested rows
    const flat: number[] = [];
    const walk = (x: unknown): void => {
      if (Array.isArray(x)) { for (const e of x) walk(e); return; }
      flat.push(Number(x));
    };
    // if top-level is list of points, keep as flat; tests slice as needed
    walk(v);
    return flat;
  }
  const arr = (v as { toArray(): unknown }).toArray() as unknown;
  const flat: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) { for (const e of x) walk(e); return; }
    flat.push(Number(x));
  };
  walk(arr);
  return flat;
}

function toRows(v: unknown): number[][] {
  const arr = Array.isArray(v)
    ? (v as unknown[])
    : (((v as { toArray(): unknown }).toArray() as unknown[]) ?? []);
  // scalar point -> single row
  if (arr.length > 0 && typeof arr[0] === "number") return [(arr as unknown[]).map(Number)];
  return (arr as unknown[][]).map((r) => (Array.isArray(r) ? r.map(Number) : [Number(r)]));
}

function expectRowsClose(actual: unknown, expected: number[][], tol = 1e-9): void {
  const rows = toRows(actual);
  expect(rows.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]?.length).toBe(expected[i]?.length);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(rows[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

// Ground truths captured from the original manim implementation (manim 0.21.0); see comments per block.

// ---------------------------------------------------------------------------
// manim.mobject.graphing.number_line.NumberLine.number_to_point
// original manim: NumberLine(x_range=[-3,3,1], length=6): start [-3,0,0], end [3,0,0];
//   n2p(0)=[0,0,0], n2p(1)=[1,0,0], n2p(-3)=[-3,0,0], n2p(3)=[3,0,0],
//   n2p([1,2,3])=[[1,0,0],[2,0,0],[3,0,0]], n2p(-1.5)=[-1.5,0,0].
//   NumberLine(x_range=[0,4,0.5], length=4): n2p(0)=[-2,0,0], n2p(4)=[2,0,0],
//   n2p(2)=[0,0,0]. Default NumberLine(): x_range [-7,7,1], n2p(0)=[0,0,0].
// ---------------------------------------------------------------------------
describe("NumberLine.number_to_point", () => {
  it("maps scalars on [-3,3] length=6 like python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expectRowsClose(nl.number_to_point(0), [[0, 0, 0]]);
    expectRowsClose(nl.number_to_point(1), [[1, 0, 0]]);
    expectRowsClose(nl.number_to_point(-3), [[-3, 0, 0]]);
    expectRowsClose(nl.number_to_point(3), [[3, 0, 0]]);
    expectRowsClose(nl.number_to_point(-1.5), [[-1.5, 0, 0]]);
  });
  it("maps a list of values row-wise like python", () => {
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    expectRowsClose(nl.number_to_point([1, 2, 3]), [[1, 0, 0], [2, 0, 0], [3, 0, 0]]);
  });
  it("respects a shifted range (n2p(2) is the center)", () => {
    const nl = new NumberLine({ x_range: [0, 4, 0.5], length: 4 });
    expectRowsClose(nl.number_to_point(0), [[-2, 0, 0]]);
    expectRowsClose(nl.number_to_point(4), [[2, 0, 0]]);
    expectRowsClose(nl.number_to_point(2), [[0, 0, 0]]);
    // n2p alias (@ operator path) agrees
    expectRowsClose(nl.n2p(2), [[0, 0, 0]]);
  });
  it("default line maps 0 to origin", () => {
    const nl = new NumberLine();
    expect(nl.x_range).toEqual([-7, 7, 1]);
    expectRowsClose(nl.number_to_point(0), [[0, 0, 0]]);
  });
  it("follows rotation (vertical line maps onto y axis)", () => {
    // original manim: rotate(90deg) -> start ~[0,-3,0], end ~[0,3,0];
    //   n2p(1)=[~0,1,0], n2p(-3)=[~0,-3,0].
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    nl.rotate(Math.PI / 2);
    expectRowsClose(nl.number_to_point(1), [[0, 1, 0]], 1e-6);
    expectRowsClose(nl.number_to_point(-3), [[0, -3, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.Triangle
// original manim: Triangle().get_vertices() =
//   [[~0,1,0],[-0.8660254,-0.5,0],[0.8660254,-0.5,0]]; equals RegularPolygon(3).
// ---------------------------------------------------------------------------
describe("Triangle", () => {
  it("has the python equilateral vertices", () => {
    const t = new Triangle();
    expectRowsClose(t.get_vertices(), [
      [0, 1, 0],
      [-0.8660254037844386, -0.5, 0],
      [0.8660254037844386, -0.5, 0],
    ], 1e-6);
  });
  it("equals RegularPolygon(n=3) vertex-for-vertex like python", () => {
    const t = new Triangle();
    const r3 = new RegularPolygon(3);
    expectRowsClose(t.get_vertices(), r3.get_vertices() as number[][], 1e-9);
  });
  it("closes the loop (3 anchors, all on the unit circle)", () => {
    const t = new Triangle();
    const verts = t.get_vertices() as number[][];
    expect(verts.length).toBe(3);
    for (const v of verts) {
      expect(close(Math.hypot(v[0] ?? 0, v[1] ?? 0), 1.0, 1e-6)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.polygram.RegularPolygon.__init__
// original manim: RegularPolygon() -> 6 verts starting at [1,0,0] (start_angle 0);
//   n=5 starts at [0,1,0] (TAU/4); n=4 radius=2 -> verts on circle r=2;
//   RegularPolygon(n=5, radius=2, start_angle=0.5)[0]=[1.7551651237807453,
//   0.958851077208406, 0]. start_angle attr: 0 for n=6, TAU/4 for n=5.
// ---------------------------------------------------------------------------
describe("RegularPolygon.__init__", () => {
  it("defaults to a hexagon starting at [1,0,0] like python", () => {
    const r = new RegularPolygon();
    const verts = r.get_vertices() as number[][];
    expect(verts.length).toBe(6);
    expectRowsClose([verts[0] ?? []], [[1, 0, 0]], 1e-9);
    expect(r.start_angle).toBe(0);
    expectRowsClose(verts, [
      [1, 0, 0],
      [0.5, 0.8660254037844386, 0],
      [-0.5, 0.8660254037844386, 0],
      [-1, 0, 0],
      [-0.5, -0.8660254037844386, 0],
      [0.5, -0.8660254037844386, 0],
    ], 1e-6);
  });
  it("odd n defaults to start_angle TAU/4 like python", () => {
    const r = new RegularPolygon(5);
    expect(close(r.start_angle ?? NaN, TAU / 4, 1e-12)).toBe(true);
    const verts = r.get_vertices() as number[][];
    expectRowsClose([verts[0] ?? []], [[0, 1, 0]], 1e-6);
  });
  it("honors radius and explicit start_angle like python", () => {
    const r = new RegularPolygon(5, { radius: 2, start_angle: 0.5 });
    expect(close(r.start_angle ?? NaN, 0.5, 1e-12)).toBe(true);
    const verts = r.get_vertices() as number[][];
    expectRowsClose([verts[0] ?? []], [[1.7551651237807453, 0.958851077208406, 0]], 1e-9);
    for (const v of verts) {
      expect(close(Math.hypot(v[0] ?? 0, v[1] ?? 0), 2.0, 1e-6)).toBe(true);
    }
  });
  it("square with radius 2 sits on the r=2 circle like python", () => {
    const r = new RegularPolygon(4, { radius: 2 });
    expectRowsClose(r.get_vertices(), [
      [2, 0, 0],
      [0, 2, 0],
      [-2, 0, 0],
      [0, -2, 0],
    ], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.position_tip
// original manim (Line [0,0,0]->[3,0,0]): unpositioned tip_point
//   [-0.425,~0,0], base [-0.075,~0,0]; after position_tip tip_point
//   [3,0,0], base [2.65,~0,0], center [2.825,~0,0],
//   _init_positioning_axis [~0,1,0]. Diagonal [0,0,0]->[2,2,0]:
//   tip_point [2,2,0], base [1.7525126265847089,1.752512626584708,0].
//   at_start: tip_point [0,0,0], base [0.35,~0,0].
//   vertical [0,0,0]->[0,3,0]: tip_point [0,3,0], base [~0,2.65,0].
// ---------------------------------------------------------------------------
describe("TipableVMobject.position_tip", () => {
  it("places the tip point on the line end (x-axis) like python", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    const ret = l.position_tip(tip, false);
    // NOTE: identity compared as a boolean: pretty-printing a Mobject proxy
    // on failure OOMs the vitest worker (see tip-base test below).
    expect(ret === tip).toBe(true);
    expectRowsClose(tip.tip_point, [[3, 0, 0]], 1e-6);
    expectRowsClose(tip.base, [[2.65, 0, 0]], 1e-6);
    expectRowsClose([tip.get_center()], [[2.825, 0, 0]], 1e-6);
  });
  it("records the perpendicular positioning axis like python", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip, false);
    const axis = (l as unknown as { _init_positioning_axis: number[] })._init_positioning_axis;
    expectRowsClose([axis], [[0, 1, 0]], 1e-6);
  });
  it("is idempotent for a second tip (same anchor) like python", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    const t1 = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(t1, false);
    const t2 = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(t2, false);
    expectRowsClose(t1.tip_point, toRows(t2.tip_point), 1e-9);
  });
  it("handles a diagonal line like python", () => {
    const l = new Line([0, 0, 0], [2, 2, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip);
    expectRowsClose(tip.tip_point, [[2, 2, 0]], 1e-6);
    expectRowsClose(tip.base, [[1.7525126265847089, 1.752512626584708, 0]], 1e-6);
  });
  it("handles at_start like python", () => {
    const l = new Line([0, 0, 0], [3, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip, true);
    expectRowsClose(tip.tip_point, [[0, 0, 0]], 1e-6);
    expectRowsClose(tip.base, [[0.35, 0, 0]], 1e-6);
  });
  it("handles a vertical line like python", () => {
    const l = new Line([0, 0, 0], [0, 3, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip);
    expectRowsClose(tip.tip_point, [[0, 3, 0]], 1e-6);
    expectRowsClose(tip.base, [[0, 2.65, 0]], 1e-6);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.arc.TipableVMobject.reset_endpoints_based_on_tip
// original manim: Line [0,0,0]->[4,0,0] + positioned tip -> reset gives end
//   [3.65,~0,0] == tip.base, start unchanged; at_start reset gives start
//   [0.35,~0,0] == tip.base; zero-length line returns self unchanged.
// ---------------------------------------------------------------------------
describe("TipableVMobject.reset_endpoints_based_on_tip", () => {
  it("shortens the end to the tip base like python", () => {
    const l = new Line([0, 0, 0], [4, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip, false);
    const ret = l.reset_endpoints_based_on_tip(tip, false);
    // NOTE: boolean identity (see position_tip note: formatting a Mobject
    // proxy on failure OOMs the worker).
    expect(ret === l).toBe(true);
    expectRowsClose([l.get_start()], [[0, 0, 0]], 1e-9);
    expectRowsClose([l.get_end()], [[3.65, 0, 0]], 1e-6);
    expectRowsClose(tip.base, [l.get_end()] as unknown as number[][], 1e-9);
    expect(close(l.get_length(), 3.65, 1e-6)).toBe(true);
  });
  it("moves the start to the tip base when at_start like python", () => {
    const l = new Line([0, 0, 0], [4, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip, true);
    l.reset_endpoints_based_on_tip(tip, true);
    expectRowsClose([l.get_start()], [[0.35, 0, 0]], 1e-6);
    expectRowsClose([l.get_end()], [[4, 0, 0]], 1e-9);
  });
  it("is a no-op returning self for a zero-length line like python", () => {
    const l = new Line([0, 0, 0], [0, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    const before = toNums(l.points);
    const ret = l.reset_endpoints_based_on_tip(tip, false);
    expect(ret === l).toBe(true);
    expect(toNums(l.points)).toEqual(before);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.geometry.line.Line.get_vector
// original manim: Line([-1,0,0],[2,0,0])->[3,0,0]; [0,0,0]->[0,3,0] -> [0,3,0];
//   [1,1,1]->[4,5,6] -> [3,4,5]; degenerate -> [0,0,0].
// ---------------------------------------------------------------------------
describe("Line.get_vector", () => {
  it("returns end - start like python", () => {
    expectRowsClose([new Line([-1, 0, 0], [2, 0, 0]).get_vector()], [[3, 0, 0]]);
    expectRowsClose([new Line([0, 0, 0], [0, 3, 0]).get_vector()], [[0, 3, 0]]);
    expectRowsClose([new Line([1, 1, 1], [4, 5, 6]).get_vector()], [[3, 4, 5]]);
  });
  it("is zero for a degenerate line like python", () => {
    expectRowsClose([new Line([0, 0, 0], [0, 0, 0]).get_vector()], [[0, 0, 0]]);
  });
  it("matches get_end/get_start difference after transforms", () => {
    const l = new Line([1, 2, 3], [4, 6, 8]);
    l.shift([1, 1, 1]);
    const v = l.get_vector();
    const diff = (l.get_end() as number[]).map((x, i) => x - (l.get_start() as number[])[i]!);
    expectRowsClose([v], [diff]);
    expectRowsClose([v], [[3, 4, 5]]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.graphing.coordinate_systems.CoordinateSystem
// original manim: CoordinateSystem() -> dimension 2, x_range [-7,7,1], y_range [-4,4,1],
//   x/y_length None, num_sampled_graph_points_per_tick 10;
//   coords_to_point/get_axes/get_origin/c2p/polar_to_point/pr2pt/get_axis/
//   get_x_axis/get_x_unit_size/get_axis_labels raise NotImplementedError;
//   point_to_coords/p2c/point_to_polar/pt2pr raise (NotImplementedError via
//   subclass stub / missing arg); explicit ranges append the default step.
//   Axes._origin_shift([1,5,1])=1, ([-5,-1,1])=-1, ([-3,3,1])=0.
// ---------------------------------------------------------------------------
describe("CoordinateSystem", () => {
  it("stores defaults like python", () => {
    const cs = new CoordinateSystem();
    expect(cs.dimension).toBe(2);
    expect(cs.x_range).toEqual([-7, 7, 1]);
    expect(cs.y_range).toEqual([-4, 4, 1]);
    expect(cs.x_length).toBeNull();
    expect(cs.y_length).toBeNull();
    expect(cs.num_sampled_graph_points_per_tick).toBe(10);
  });
  it("expands 2-element ranges with the default step like python", () => {
    const cs = new CoordinateSystem([0, 5], [-2, 2, 0.5], 10, 4, 3);
    expect(cs.x_range).toEqual([0, 5, 1]);
    expect(cs.y_range).toEqual([-2, 2, 0.5]);
    expect(cs.x_length).toBe(10);
    expect(cs.y_length).toBe(4);
    expect(cs.dimension).toBe(3);
  });
  it("abstract methods raise NotImplementedError like python", () => {
    // NOTE: both Python and the TS port raise NotImplementedError with an
    // empty message, so assert on the error `name` (vitest regex matches the
    // message, which is "" on both sides).
    const names = (fn: () => void): string => {
      try { fn(); } catch (e) { return (e as Error).name; }
      return "<no throw>";
    };
    const cs = new CoordinateSystem();
    for (const fn of [
      () => cs.coords_to_point(1, 2),
      () => cs.c2p(1, 2),
      () => cs.get_axes(),
      () => cs.get_axis(0),
      () => cs.get_origin(),
      () => cs.get_x_axis(),
      () => cs.get_y_axis(),
      () => cs.get_x_unit_size(),
      () => cs.get_y_unit_size(),
      () => cs.get_axis_labels(),
      () => cs.point_to_coords([1, 2, 0]),
      () => cs.p2c([1, 2, 0]),
      () => cs.polar_to_point(1, 0.5),
      () => cs.pr2pt(1, 0.5),
    ]) {
      expect(names(fn)).toBe("NotImplementedError");
    }
  });
  it("point_to_polar/pr2pt delegate (throw via point_to_coords) like python", () => {
    const cs = new CoordinateSystem();
    expect(() => cs.point_to_polar([1, 0, 0])).toThrow();
    expect(() => cs.pt2pr([1, 0, 0])).toThrow();
  });
  it("Axes._origin_shift matches python", () => {
    expect(Axes._origin_shift([1, 5, 1])).toBe(1);
    expect(Axes._origin_shift([-5, -1, 1])).toBe(-1);
    expect(Axes._origin_shift([-3, 3, 1])).toBe(0);
  });
  it("c2p/p2c aliases delegate to coords_to_point/point_to_coords", () => {
    // original manim NumberPlane(x_range=[-3,3,1], y_range=[-2,2,1], x_length=6,
    // y_length=4): coords_to_point(1,1)=[1,1,0], point_to_coords([1,1,0])=[1,1];
    // the c2p/p2c aliases return the same values.
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectRowsClose(plane.c2p(1, 1), [[1, 1, 0]], 1e-9);
    expectRowsClose(plane.coords_to_point(1, 1), [[1, 1, 0]], 1e-9);
    const viaAlias = plane.p2c([1, 1, 0]);
    const viaDirect = plane.point_to_coords([1, 1, 0]);
    expect(viaAlias).toEqual(viaDirect);
    expect(viaAlias.length).toBe(2);
    expect(close(viaAlias[0] ?? NaN, 1, 1e-9)).toBe(true);
    expect(close(viaAlias[1] ?? NaN, 1, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// INTEGRATION (this batch x already-ported code)
// ---------------------------------------------------------------------------
describe("integration (batch layers17)", () => {
  it("NumberLine n2p/p2n roundtrip + get_vector chain matches python", () => {
    // original manim: NumberLine([-3,3,1], length=6): p2n(n2p(x)) == x for x in
    // [-2, 0, 2.5]; Line vector from n2p(0)->n2p(2) is [2,0,0].
    const nl = new NumberLine({ x_range: [-3, 3, 1], length: 6 });
    for (const x of [-2, 0, 2.5]) {
      const p = toRows(nl.n2p(x))[0] ?? [0, 0, 0];
      expect(close(nl.point_to_number(p) as number, x, 1e-6)).toBe(true);
    }
    const a = toRows(nl.n2p(0))[0] ?? [0, 0, 0];
    const b = toRows(nl.n2p(2))[0] ?? [0, 0, 0];
    expectRowsClose([new Line(a, b).get_vector()], [[2, 0, 0]], 1e-6);
  });
  it("Triangle vertices feed Line.get_vector edge lengths (equilateral)", () => {
    // original manim: all three Triangle edges have length sqrt(3) ~= 1.7320508075688772.
    const t = new Triangle();
    const verts = t.get_vertices() as number[][];
    const lens = [0, 1, 2].map(
      (i) => new Line(verts[i] ?? [], verts[(i + 1) % 3] ?? []).get_vector(),
    );
    for (const v of lens) {
      expect(close(Math.hypot(...(v as [number, number, number])), Math.sqrt(3), 1e-6)).toBe(true);
    }
  });
  it("Arrow tip pipeline (position_tip + reset) matches python end shortening", () => {
    // original manim: Arrow([0,0,0],[2,0,0]) vector is [1.5,0,0] (buff+tip shorten it);
    // a manual Line+tip reset ends at [3.65,~0,0] for a length-4 line.
    const arrow = new Arrow([0, 0, 0], [2, 0, 0]);
    expect(arrow.has_tip()).toBe(true);
    expectRowsClose([arrow.get_vector()], [[1.5, 0, 0]], 1e-6);
    const l = new Line([0, 0, 0], [4, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    l.position_tip(tip, false);
    l.reset_endpoints_based_on_tip(tip, false);
    expect(close(l.get_length(), 3.65, 1e-6)).toBe(true);
  });
  it("RegularPolygon + Arc tip share the Triangle base (tips build on Triangle)", () => {
    // The default ArrowTriangleFilledTip is Triangle-based; its unpositioned
    // tip_point matches the fresh-tip ground truth [-0.425,~0,0], base
    // [-0.075,~0,0] (original manim). NOTE: `instanceof` is evaluated as an expression
    // (boolean) rather than via toBeInstanceOf: vitest's error formatter OOMs
    // when pretty-printing a whole Mobject proxy on failure.
    const l = new Line([0, 0, 0], [1, 0, 0]);
    const tip = l.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    expect(tip instanceof ArrowTriangleTip).toBe(true);
    expectRowsClose(tip.tip_point, [[-0.425, 0, 0]], 1e-6);
    expectRowsClose(tip.base, [[-0.075, 0, 0]], 1e-6);
  });
  it("Axes concrete mapping inverts via NumberPlane (origin + unit sizes)", () => {
    // original manim Axes(x_range=[-3,3,1], y_range=[-2,2,1], x_length=6, y_length=4):
    // get_origin()=[0,0,0], coords_to_point(1,1)=[1,1,0] (1e-9),
    // x/y unit sizes 1. The TS NumberPlane implements the same linear map.
    const plane = new NumberPlane([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectRowsClose(plane.coords_to_point(0, 0), [[0, 0, 0]], 1e-9);
    expectRowsClose(plane.coords_to_point(1, 1), [[1, 1, 0]], 1e-6);
    expect(plane.point_to_coords([0, 0, 0])).toEqual([0, 0]);
    // RegularPolygon density chain still consistent alongside graphing code
    const rp = new RegularPolygram(7, { density: 3 });
    expect(rp.get_vertex_groups().length).toBe(1);
  });
  it("Arc + position_tip composes (arc end anchor gets the tip)", () => {
    // original manim: Arc(radius=1, angle=pi/2) end [~0,1,0]; positioning a tip there
    // puts tip_point exactly on the arc end.
    const arc = new Arc(1, 0, Math.PI / 2);
    const tip = arc.get_unpositioned_tip(ArrowTriangleFilledTip, null, null);
    arc.position_tip(tip, false);
    expectRowsClose(tip.tip_point, [arc.get_end()] as unknown as number[][], 1e-9);
  });
});
