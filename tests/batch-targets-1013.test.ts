// Ground-truth tests for batch 1013 (layers 3..3).
//
// Expected values were captured from the original Python manim
// (original manim 0.21.0); each describe block cites
// the probe used. Float comparisons use tight tolerance (1e-9 abs).
import { describe, expect, it, vi, afterEach } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { readFileSync as nodeReadFileSync } from "node:fs";

// Mocked stdin for prompt_user_for_choice (which does readFileSync(0)).
// Non-stdin reads fall through to the real implementation. The mock lives
// at file scope because ESM namespace objects are frozen (vi.spyOn fails).
let mockStdinValue = "*";
vi.mock("node:fs", async (importOriginal) => {
  const mod = await importOriginal<typeof import("node:fs")>();
  const real = mod.readFileSync as typeof nodeReadFileSync;
  return {
    ...mod,
    readFileSync: ((...args: unknown[]) => {
      if (args[0] === 0) return mockStdinValue;
      return (real as (...a: unknown[]) => unknown)(...args);
    }) as typeof nodeReadFileSync,
  };
});
import * as rate from "../src/utils/rate_functions.js";
import { compass_directions } from "../src/utils/space_ops.js";
import {
  path_along_arc,
  path_along_circles,
  spiral_path,
} from "../src/utils/paths.js";
import { _Memoizer } from "../src/utils/hash.js";
import { Facet, QuickHull } from "../src/utils/qhull.js";
import { Group, Mobject } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Point } from "../src/mobject/types/point_cloud_mobject.js";
import {
  get_scene_classes_from_module,
  is_child_scene,
  parse_scene_choice,
  prompt_user_for_choice,
  type SceneClass,
  type SceneModule,
} from "../src/utils/module_ops.js";
import { Scene } from "../src/scene/scene.js";
import { ManimConfig } from "../src/_config/utils.js";
import {
  HSV,
  ManimColor,
  rgba_to_color,
} from "../src/utils/color/core.js";

const TOL = 1e-9;

function close(a: number, b: number, tol = TOL): boolean {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
}

function expectNum(actual: number, expected: number): void {
  expect(close(Number(actual), expected)).toBe(true);
}

function expectArrClose(actual: NDArray | number[] | number[][], expected: number[] | number[][], tol = TOL): void {
  const raw = Array.isArray(actual) ? actual : ((actual as NDArray).toArray() as unknown);
  const flat = (Array.isArray(raw) ? raw : [raw]).flat(Infinity as number) as unknown[];
  const expFlat = (expected as unknown[]).flat(Infinity as number) as unknown[];
  expect(flat.length).toBe(expFlat.length);
  for (let i = 0; i < expFlat.length; i += 1) {
    const a = Number(flat[i]);
    const e = Number(expFlat[i]);
    expect(close(a, e, tol)).toBe(true);
  }
}

function rowsOf(value: NDArray): number[][] {
  return (value.toArray() as number[][]).map((row) => row.map(Number));
}

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.*
// Ground truth probe:
//   the original manim implementation -c "from manim.utils import rate_functions as rf;
//   ts=[0.0,0.25,0.5,0.75,1.0,-0.5,1.5]; {n: [float(getattr(rf,n)(t)) for t in ts]}"
// ---------------------------------------------------------------------------
const RATE_TS = [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 1.5];
const RATE_GT: Record<string, number[]> = {
  linear: [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0],
  smooth: [0.0, 0.07010371654510815, 0.5, 0.929896283454892, 1.0, 0.0, 1.0],
  smoothstep: [0.0, 0.15625, 0.5, 0.84375, 1.0, 0.0, 1.0],
  smootherstep: [0.0, 0.103515625, 0.5, 0.896484375, 1.0, 0.0, 1.0],
  smoothererstep: [0.0, 0.070556640625, 0.5, 0.929443359375, 1.0, 0.0, 1.0],
  slow_into: [0.0, 0.6614378277661477, 0.8660254037844386, 0.9682458365518543, 1.0, 0.0, 1.0],
  running_start: [0.0, -0.1766357421875, 0.0703125, 0.7481689453125, 1.0, 0.0, 1.0],
  lingering: [0.0, 0.3125, 0.625, 0.9375, 1.0, 0.0, 1.0],
  exponential_decay: [0.0, 0.9179150013761012, 0.9932620530009145, 0.9994469156298522, 0.9999546000702375, 0.0, 1.0],
  ease_in_sine: [0.0, 0.07612046748871326, 0.2928932188134524, 0.6173165676349102, 0.9999999999999999, 0.0, 1.0],
  ease_out_sine: [0.0, 0.3826834323650898, 0.7071067811865475, 0.9238795325112867, 1.0, 0.0, 1.0],
  ease_in_out_sine: [-0.0, 0.1464466094067262, 0.49999999999999994, 0.8535533905932737, 1.0, 0.0, 1.0],
  ease_in_quad: [0.0, 0.0625, 0.25, 0.5625, 1.0, 0.0, 1.0],
  ease_out_quad: [0.0, 0.4375, 0.75, 0.9375, 1.0, 0.0, 1.0],
  ease_in_out_quad: [0.0, 0.125, 0.5, 0.875, 1.0, 0.0, 1.0],
  ease_in_cubic: [0.0, 0.015625, 0.125, 0.421875, 1.0, 0.0, 1.0],
  ease_out_cubic: [0.0, 0.578125, 0.875, 0.984375, 1.0, 0.0, 1.0],
  ease_in_out_cubic: [0.0, 0.0625, 0.5, 0.9375, 1.0, 0.0, 1.0],
  ease_in_quart: [0.0, 0.00390625, 0.0625, 0.31640625, 1.0, 0.0, 1.0],
  ease_out_quart: [0.0, 0.68359375, 0.9375, 0.99609375, 1.0, 0.0, 1.0],
  ease_in_out_quart: [0.0, 0.03125, 0.5, 0.96875, 1.0, 0.0, 1.0],
  ease_in_quint: [0.0, 0.0009765625, 0.03125, 0.2373046875, 1.0, 0.0, 1.0],
  ease_out_quint: [0.0, 0.7626953125, 0.96875, 0.9990234375, 1.0, 0.0, 1.0],
  ease_in_out_quint: [0.0, 0.015625, 0.5, 0.984375, 1.0, 0.0, 1.0],
  ease_in_expo: [0.0, 0.005524271728019903, 0.03125, 0.1767766952966369, 1.0, 0.0, 1.0],
  ease_out_expo: [0.0, 0.8232233047033631, 0.96875, 0.99447572827198, 1.0, 0.0, 1.0],
  ease_in_out_expo: [0.0, 0.015625, 0.5, 0.984375, 1.0, 0.0, 1.0],
  ease_in_circ: [0.0, 0.031754163448145745, 0.1339745962155614, 0.3385621722338523, 1.0, 0.0, 1.0],
  ease_out_circ: [0.0, 0.6614378277661477, 0.8660254037844386, 0.9682458365518543, 1.0, 0.0, 1.0],
  ease_in_out_circ: [0.0, 0.0669872981077807, 0.5, 0.9330127018922193, 1.0, 0.0, 1.0],
  ease_in_back: [0.0, -0.06413656250000001, -0.08769750000000004, 0.18259031249999969, 0.9999999999999998, 0.0, 1.0],
  ease_out_back: [2.220446049250313e-16, 0.8174096875000002, 1.0876975, 1.0641365625, 1.0, 0.0, 1.0],
  ease_in_out_back: [-0.0, -0.09968184375, 0.5, 1.09968184375, 1.0, 0.0, 1.0],
  ease_in_elastic: [0.0, -0.005524271728019903, -0.015625000000000045, 0.08838834764831832, 1.0, 0.0, 1.0],
  ease_out_elastic: [0.0, 0.9116116523516816, 1.015625, 1.00552427172802, 1.0, 0.0, 1.0],
  ease_in_out_elastic: [0.0, 0.011969444423734044, 0.5, 0.988030555576266, 1.0, 0.0, 1.0],
  ease_out_bounce: [0.0, 0.47265625, 0.765625, 0.97265625, 1.0, 0.0, 1.0],
};

describe("rate_functions ground truth", () => {
  for (const [name, expected] of Object.entries(RATE_GT)) {
    it(`${name} matches python at 7 sample points`, () => {
      const fn = (rate as unknown as Record<string, (t: number) => number>)[name];
      expect(typeof fn).toBe("function");
      RATE_TS.forEach((t, i) => {
        expectNum(fn(t), expected[i] ?? 0);
      });
    });
  }

  it("smooth honors a custom inflection (matches python smooth(t, 4.0))", () => {
    // Probe: [float(rf.smooth(t, 4.0)) for t in (0.25, 0.5, 0.75)]
    // -> [0.19661193324148185, 0.5, 0.8033880667585181].
    const expected = [0.19661193324148185, 0.5, 0.8033880667585181];
    [0.25, 0.5, 0.75].forEach((t, i) => {
      expectNum((rate.smooth as unknown as (t: number, k: number) => number)(t, 4.0), expected[i] ?? 0);
    });
  });

  it("running_start honors pull_factor=0.0 (matches python)", () => {
    // Probe: float(rf.running_start(0.5, 0.0)) == 0.34375
    expectNum(
      (rate.running_start as unknown as (t: number, p: number) => number)(0.5, 0.0),
      0.34375,
    );
  });

  it("exponential_decay honors half_life=0.5 (matches python)", () => {
    // Probe: float(rf.exponential_decay(0.5, 0.5)) == 0.6321205588285577
    expectNum(
      (rate.exponential_decay as unknown as (t: number, h: number) => number)(0.5, 0.5),
      0.6321205588285577,
    );
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.compass_directions
// Probe: compass_directions().tolist(), compass_directions(8).tolist(),
//        compass_directions(1).tolist()
// ---------------------------------------------------------------------------
describe("compass_directions", () => {
  it("n=4 gives RIGHT/UP/LEFT/DOWN", () => {
    expectArrClose(compass_directions(), [
      [1.0, 0.0, 0.0],
      [2.220446049250313e-16, 1.0, 0.0],
      [-1.0, 1.2246467991473532e-16, 0.0],
      [-2.220446049250313e-16, -1.0, 0.0],
    ]);
  });

  it("n=8 gives octagon directions", () => {
    expectArrClose(compass_directions(8), [
      [1.0, 0.0, 0.0],
      [0.7071067811865475, 0.7071067811865476, 0.0],
      [2.220446049250313e-16, 1.0, 0.0],
      [-0.7071067811865475, 0.7071067811865477, 0.0],
      [-1.0, 1.2246467991473532e-16, 0.0],
      [-0.7071067811865475, -0.7071067811865475, 0.0],
      [-2.220446049250313e-16, -1.0, 0.0],
      [0.7071067811865474, -0.7071067811865478, 0.0],
    ]);
  });

  it("n=1 gives just the start vector", () => {
    expectArrClose(compass_directions(1), [[1.0, 0.0, 0.0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.paths.*
// Probe: path_along_arc(pi/2), path_along_arc(0.001) [straight fallback],
//        spiral_path(pi), spiral_path(0.001), path_along_circles(pi/2, origin)
//        applied to start=[[0,0,0],[1,0,0]], end=[[1,1,0],[2,1,0]] at a=0/0.5/1
// ---------------------------------------------------------------------------
describe("path functions", () => {
  const START = array([[0, 0, 0], [1, 0, 0]]) as NDArray;
  const END = array([[1, 1, 0], [2, 1, 0]]) as NDArray;

  it("path_along_arc(pi/2)", () => {
    const path = path_along_arc(Math.PI / 2);
    expect(rowsOf(path(START, END, 0))).toEqual([[0, 0, 0], [1, 0, 0]]);
    expectArrClose(path(START, END, 0.5), [
      [0.7071067811865476, 0.29289321881345265, 0.0],
      [1.7071067811865475, 0.29289321881345265, 0.0],
    ]);
    expectArrClose(path(START, END, 1.0), [
      [0.9999999999999999, 0.9999999999999999, 0.0],
      [2.0, 0.9999999999999999, 0.0],
    ]);
  });

  it("path_along_arc tiny angle falls back to straight interpolate", () => {
    const path = path_along_arc(0.001);
    expectArrClose(path(START, END, 0.5), [[0.5, 0.5, 0], [1.5, 0.5, 0]]);
    expectArrClose(path(START, END, 1.0), [[1, 1, 0], [2, 1, 0]]);
  });

  it("spiral_path(pi)", () => {
    const path = spiral_path(Math.PI);
    expectArrClose(path(START, END, 0.0), [[0, 0, 0], [1, 0, 0]]);
    expectArrClose(path(START, END, 0.5), [
      [0.5000000000000001, -0.4999999999999999, 0.0],
      [1.5, -0.4999999999999999, 0.0],
    ]);
    expectArrClose(path(START, END, 1.0), [[1, 1, 0], [2, 1, 0]]);
  });

  it("spiral_path tiny angle falls back to straight interpolate", () => {
    const path = spiral_path(0.001);
    expectArrClose(path(START, END, 0.5), [[0.5, 0.5, 0], [1.5, 0.5, 0]]);
  });

  it("path_along_circles(pi/2, origin)", () => {
    const path = path_along_circles(Math.PI / 2, array([0, 0, 0]) as NDArray);
    expectArrClose(path(START, END, 0.0), [[0, 0, 0], [1, 0, 0]]);
    expectArrClose(path(START, END, 0.5), [
      [0.7071067811865475, 1.895269253967044e-16, 0.0],
      [1.4142135623730951, 3.0054922785922005e-16, 0.0],
    ]);
    expectArrClose(path(START, END, 1.0), [[1, 1, 0], [2, 1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing._Memoizer.{check_already_processed_decorator.layer,
//   check_already_processed, mark_as_processed}
// Probe: tuple dedup (equal tuples -> AP), list identity (equal lists distinct),
//        primitives passthrough, decorator func/method, mark_as_processed.
// ---------------------------------------------------------------------------
describe("_Memoizer", () => {
  it("check_already_processed dedups the same object, passes primitives through", () => {
    const m = new _Memoizer();
    const obj = { a: 1 };
    expect(m.check_already_processed(obj)).toBe(obj);
    expect(m.check_already_processed(obj)).toBe(_Memoizer.ALREADY_PROCESSED_PLACEHOLDER);
    expect(m.check_already_processed(5)).toBe(5);
    expect(m.check_already_processed("x")).toBe("x");
  });

  it("layer decorator (function case) memoizes repeated calls", () => {
    const m = new _Memoizer();
    const layer = m.check_already_processed_decorator();
    const seen: unknown[] = [];
    const f = layer((obj: unknown) => {
      seen.push(obj);
      return `processed:${JSON.stringify(obj)}`;
    });
    const first = f({ k: [4, 5] });
    expect(typeof first).toBe("string");
    // NOTE: deviation from Python (documented in src/utils/hash.ts): Python
    // memoizes equal *tuples* by value, so f((4,5)) twice yields "AP" the
    // second time. JS cannot distinguish tuples from lists, so the TS port
    // keys every non-primitive by identity: distinct-but-equal objects are
    // processed twice (safe direction — missed dedups only bloat output).
    const a = { v: 1 };
    expect(f(a)).toContain("processed");
    expect(f({ v: 1 })).toContain("processed");
    expect(seen.length).toBe(3);
    expect(f(a)).toBe(_Memoizer.ALREADY_PROCESSED_PLACEHOLDER);
  });

  it("layer decorator (method case, is_method=True) memoizes on second arg", () => {
    const m = new _Memoizer();
    const layer = m.check_already_processed_decorator(true);
    const meth = layer((_self: unknown, obj: unknown) => ["meth", obj]);
    const self = {};
    const arg = { n: 9 };
    expect(meth(self, arg)).toEqual(["meth", arg]);
    expect(meth(self, arg)).toBe(_Memoizer.ALREADY_PROCESSED_PLACEHOLDER);
  });

  it("mark_as_processed causes the next check to return AP", () => {
    const m = new _Memoizer();
    const obj = { t: [7, 8] };
    m.mark_as_processed(obj);
    expect(m.check_already_processed(obj)).toBe(_Memoizer.ALREADY_PROCESSED_PLACEHOLDER);
  });

  it("distinct equal-content objects are NOT collapsed (matches python list behavior)", () => {
    // Probe: l1=[1,2]; l2=[1,2] -> check(l1) is l1, check(l2) is l2
    // (lists are unhashable in Python -> id branch, same as TS identity).
    const m = new _Memoizer();
    const l1 = [1, 2];
    const l2 = [1, 2];
    expect(m.check_already_processed(l1)).toBe(l1);
    expect(m.check_already_processed(l2)).toBe(l2);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.QuickHull.{classify, initialize, compute_horizon}
// ---------------------------------------------------------------------------
describe("QuickHull", () => {
  it("classify records outside set + eye and shrinks unclaimed", () => {
    // Probe: unclaimed=[[0,0],[1,0],[0.5,2]], internal=[0.5,-1],
    // facet=[[0,0],[1,0]] (normal [0,1]) ->
    // outside=[[0.5,2]], eye=[0.5,2], unclaimed=[[0,0],[1,0]].
    const qh = new QuickHull();
    qh.unclaimed = [[0, 0], [1, 0], [0.5, 2.0]];
    qh.internal = [0.5, -1];
    const facet = new Facet([[0, 0], [1, 0]], [0.5, -1]);
    expect(facet.normal.map(Number)).toHaveLength(2);
    expectNum(facet.normal[0] ?? 0, 0.0);
    expectNum(facet.normal[1] ?? 0, 1.0);
    qh.classify(facet);
    const entry = qh.outside.get(facet.key());
    expect(entry).toBeDefined();
    expect(entry?.outside).toEqual([[0.5, 2.0]]);
    expect(entry?.eye).toEqual([0.5, 2.0]);
    expect(qh.unclaimed).toEqual([[0, 0], [1, 0]]);
  });

  it("classify with no point outside records empty outside and null eye", () => {
    // Probe: same facet but internal=[0.5,0.5] (normal [0,-1]) ->
    // outside=[], eye=None, unclaimed unchanged.
    const qh = new QuickHull();
    qh.unclaimed = [[0, 0], [1, 0], [0.5, 2.0]];
    qh.internal = [0.5, 0.5];
    const facet = new Facet([[0, 0], [1, 0]], [0.5, 0.5]);
    qh.classify(facet);
    const entry = qh.outside.get(facet.key());
    expect(entry?.outside).toEqual([]);
    expect(entry?.eye).toBeNull();
    expect(qh.unclaimed).toEqual([[0, 0], [1, 0], [0.5, 2.0]]);
  });

  it("classify on empty unclaimed records (None, None)", () => {
    const qh = new QuickHull();
    qh.unclaimed = [];
    const facet = new Facet([[0, 0], [1, 0]], [0.5, 0.5]);
    qh.classify(facet);
    const entry = qh.outside.get(facet.key());
    expect(entry?.outside).toBeNull();
    expect(entry?.eye).toBeNull();
  });

  it("compute_horizon on a triangle finds 2 visible facets + 2 boundary edges", () => {
    // Probe: triangle facets from internal=[0.5,0.25], eye=[0.5,2.0],
    // start at facet [[1,0],[0.5,1]] -> 2 horizon facets, 2 boundary,
    // centers sorted == [[0.25,0.5],[0.75,0.5]].
    const qh = new QuickHull();
    qh.internal = [0.5, 0.25];
    const fA = new Facet([[0, 0], [1, 0]], [0.5, 0.25]);
    const fB = new Facet([[1, 0], [0.5, 1]], [0.5, 0.25]);
    const fC = new Facet([[0.5, 1], [0, 0]], [0.5, 0.25]);
    qh.facets = [fA, fB, fC];
    const addNeighbor = (qh as unknown as { addNeighbor: (sf: unknown, f: Facet) => void }).addNeighbor.bind(qh);
    for (const f of [fA, fB, fC]) for (const sf of f.subfacets) addNeighbor(sf, f);
    const horizon = qh.compute_horizon([0.5, 2.0], fB);
    expect(horizon.facets.length).toBe(2);
    expect(horizon.boundary.length).toBe(2);
    const centers = horizon.facets.map((f) => f.center.map(Number)).sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0));
    expectArrClose(array(centers) as NDArray, [[0.25, 0.5], [0.75, 0.5]]);
  });

  it("initialize builds dim+1 facets and classifies every point", () => {
    // initialize() samples a random simplex, so only structural invariants
    // are asserted (both ports use an unseeded RNG here).
    const pts = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [2, 0.5]];
    const qh = new QuickHull();
    qh.initialize(pts);
    expect(qh.facets.length).toBe(3);
    expect(qh.internal).toHaveLength(2);
    expect(qh.outside.size).toBe(3);
    const remaining = (qh.unclaimed ?? []).length;
    let accounted = remaining;
    for (const entry of qh.outside.values()) accounted += (entry.outside ?? []).length;
    expect(accounted).toBe(pts.length);
  });

  it("build rejects degenerate input like python", () => {
    const qh = new QuickHull();
    expect(() => qh.build([[0]])).toThrow();
    expect(() => qh.build([[0], [1]])).toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.{Group.__init__, Mobject.add_to_back,
//   Mobject.length_over_dim}
// Probes: Group(Mobject(), Mobject()) has 2 submobjects; add_to_back puts the
// new mobject first; re-adding moves to back; length_over_dim on
// points=[[0,0,0],[2,3,1]] -> (2,3,1); empty -> 0.
// ---------------------------------------------------------------------------
describe("Mobject / Group", () => {
  it("Group.__init__ adds the given mobjects", () => {
    const g = new Group(new Mobject(), new Mobject());
    expect(g.submobjects.length).toBe(2);
    const empty = new Group();
    expect(empty.submobjects.length).toBe(0);
  });

  it("add_to_back prepends new mobjects", () => {
    const parent = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    parent.add(b, c);
    const d = new Mobject();
    parent.add_to_back(d);
    expect(parent.submobjects.length).toBe(3);
    expect(parent.submobjects[0]).toBe(d);
  });

  it("add_to_back moves an existing submobject to the back", () => {
    const parent = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    parent.add(b, c);
    parent.add_to_back(b);
    expect(parent.submobjects[0]).toBe(b);
    expect(parent.submobjects.length).toBe(2);
  });

  it("length_over_dim measures axis extents", () => {
    const m = new Mobject();
    m.points = array([[0, 0, 0], [2, 3, 1]]) as NDArray;
    expectNum(m.length_over_dim(0), 2.0);
    expectNum(m.length_over_dim(1), 3.0);
    expectNum(m.length_over_dim(2), 1.0);
  });

  it("length_over_dim of an empty mobject is 0", () => {
    expectNum(new Mobject().length_over_dim(0), 0);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.{set_sheen_direction,
//   insert_n_curves_to_point_list, set_fill, set_stroke}
// Probes: sheen_direction == UP after set; insert into an 8-point (2-curve)
// path with n=1 -> shape (12,3), first rows [[0,0,0],[0.5,0,0]]; single point
// with n=2 -> 8 repeats; set_fill('#FF0000',0.5) -> fill_rgbas [1,0,0,0.5],
// fill_opacity 0.5; set_stroke('#00FF00',width=5,opacity=0.7).
// ---------------------------------------------------------------------------
describe("VMobject style setters", () => {
  it("set_sheen_direction stores a copy of the direction", () => {
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.set_sheen_direction([0, 1, 0]);
    expect(v.sheen_direction).toEqual([0, 1, 0]);
    expect(v.get_sheen_direction()).toEqual([0, 1, 0]);
  });

  it("set_sheen_direction with family=false leaves children alone", () => {
    const parent = new VMobject();
    parent.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    const child = new VMobject();
    child.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    parent.add(child);
    const before = [...child.sheen_direction];
    parent.set_sheen_direction([1, 0, 0], false);
    expect(parent.sheen_direction).toEqual([1, 0, 0]);
    expect(child.sheen_direction).toEqual(before);
    parent.set_sheen_direction([0, 0, 1], true);
    expect(child.sheen_direction).toEqual([0, 0, 1]);
  });

  it("insert_n_curves_to_point_list adds one curve to a 2-curve path", () => {
    const v = new VMobject();
    const pts = array([
      [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
      [3, 1, 0], [3, 2, 0], [3, 3, 0], [4, 3, 0],
    ]) as NDArray;
    const out = v.insert_n_curves_to_point_list(1, pts);
    expect(out.shape).toEqual([12, 3]);
    const rows = rowsOf(out);
    expectArrClose(array([rows[0]]) as NDArray, [[0, 0, 0]]);
    expectArrClose(array([rows[1]]) as NDArray, [[0.5, 0, 0]]);
  });

  it("insert_n_curves_to_point_list repeats a single point nppcc*n times", () => {
    const v = new VMobject();
    const out = v.insert_n_curves_to_point_list(2, array([[1, 2, 3]]) as NDArray);
    expect(out.shape).toEqual([8, 3]);
    for (const row of rowsOf(out)) expect(row).toEqual([1, 2, 3]);
  });

  it("set_fill sets color + opacity", () => {
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.set_fill("#FF0000", 0.5);
    expect(rowsOf(v.fill_rgbas)[0]).toEqual([1, 0, 0, 0.5]);
    expectNum(v.fill_opacity, 0.5);
  });

  it("set_stroke sets color + width + opacity", () => {
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.set_stroke("#00FF00", 5, 0.7);
    expect(rowsOf(v.stroke_rgbas)[0]).toEqual([0, 1, 0, 0.7]);
    expectNum(v.stroke_width, 5);
    expectNum(v.stroke_opacity, 0.7);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.point_cloud_mobject.Point.__init__
// Probe: Point([1,2,3]).location=[1,2,3], points=[[1,2,3]];
//        Point().location=ORIGIN, points=[[0,0,0]].
// ---------------------------------------------------------------------------
describe("Point.__init__", () => {
  it("stores a custom location and generates one point", () => {
    const p = new Point([1, 2, 3]);
    expect([...(p.location as number[])]).toEqual([1, 2, 3]);
    expect(rowsOf(p.points)).toEqual([[1, 2, 3]]);
  });

  it("defaults to ORIGIN", () => {
    const p = new Point();
    expect(rowsOf(p.points)).toEqual([[0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.module_ops.{get_scene_classes_from_module.is_child_scene,
//   prompt_user_for_choice}
// Probe: module with A(Scene, __module__=mymod), B(Scene, othermod),
// NotScene, Scene -> only [A].
// prompt_user_for_choice is interactive in Python (console.input); the TS
// port reads fd 0, so its pure parsing core (parse_scene_choice, shared by
// the port) is tested directly plus prompt_user_for_choice with mocked stdin.
// ---------------------------------------------------------------------------
describe("module_ops", () => {
  function makeModule(): { module: SceneModule; A: SceneClass; B: SceneClass } {
    class A extends Scene {
      override construct(): void { /* noop */ }
    }
    class B extends Scene {
      override construct(): void { /* noop */ }
    }
    Object.defineProperty(A, "name", { value: "A" });
    Object.defineProperty(B, "name", { value: "B" });
    (A as unknown as Record<string, unknown>).__module__ = "mymod";
    (B as unknown as Record<string, unknown>).__module__ = "othermod";
    const module: SceneModule = { __name__: "mymod", __file__: "mymod.py", A, B, Scene };
    return { module, A: A as unknown as SceneClass, B: B as unknown as SceneClass };
  }

  it("is_child_scene accepts only Scene subclasses owned by the module", () => {
    const { module, A, B } = makeModule();
    expect(is_child_scene(A, module)).toBe(true);
    expect(is_child_scene(B, module)).toBe(false);
    expect(is_child_scene(Scene, module)).toBe(false);
    expect(is_child_scene(class {}, module)).toBe(false);
    expect(is_child_scene(42, module)).toBe(false);
  });

  it("get_scene_classes_from_module returns only owned scenes", () => {
    const { module, A } = makeModule();
    const found = get_scene_classes_from_module(module);
    expect(found.length).toBe(1);
    expect(found[0]).toBe(A);
  });

  it("parse_scene_choice handles '*', lists, and bad input", () => {
    const { module } = makeModule();
    const classes = get_scene_classes_from_module(module);
    expect(parse_scene_choice(classes, "*")).toEqual(classes);
    expect(parse_scene_choice(classes, "1")).toEqual(classes);
    expect(() => parse_scene_choice(classes, "2")).toThrow();
    expect(() => parse_scene_choice(classes, "abc")).toThrow();
  });

  it("prompt_user_for_choice selects via mocked stdin", () => {
    // The port reads fd 0 via node:fs readFileSync, mocked at module level
    // (see vi.mock at the top of this file) to avoid blocking on stdin.
    mockStdinValue = "*";
    const { module } = makeModule();
    const classes = get_scene_classes_from_module(module);
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    try {
      expect(prompt_user_for_choice(classes)).toEqual(classes);
    } finally {
      log.mockRestore();
    }
  });

  it("prompt_user_for_choice parses a numeric list via mocked stdin", () => {
    mockStdinValue = "1";
    const { module } = makeModule();
    const classes = get_scene_classes_from_module(module);
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    try {
      expect(prompt_user_for_choice(classes)).toEqual(classes);
    } finally {
      log.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig.background_opacity
// Probe: fresh ManimConfig().background_opacity is None; setting 0.5 works;
// setting 2 / -0.1 raises ValueError("background_opacity must be ...").
// NOTE: the TS port defaults the global config value to 1 and does not flip
// the movie extension to .mov on transparent backgrounds (Python does via
// resolve_movie_file_extension); only get/set validation is ported.
// ---------------------------------------------------------------------------
describe("ManimConfig.background_opacity", () => {
  it("round-trips a valid value", () => {
    const cfg = new ManimConfig();
    cfg.background_opacity = 0.5;
    expectNum(cfg.background_opacity, 0.5);
    expectNum(cfg.__getitem__("background_opacity"), 0.5);
  });

  it("rejects values outside [0, 1] with ValueError", () => {
    const cfg = new ManimConfig();
    expect(() => {
      cfg.background_opacity = 2;
    }).toThrowError(/background_opacity must be/);
    expect(() => {
      cfg.background_opacity = -0.1;
    }).toThrowError(/background_opacity must be/);
    try {
      cfg.background_opacity = 2;
      expect.unreachable();
    } catch (error) {
      expect((error as Error).name).toBe("ValueError");
    }
  });

  it("accepts the 0/1 boundaries", () => {
    const cfg = new ManimConfig();
    cfg.background_opacity = 0;
    expectNum(cfg.background_opacity, 0);
    cfg.background_opacity = 1;
    expectNum(cfg.background_opacity, 1);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.*
// Ground-truth probes (python -c, 0.21.0):
//   invert/interpolate/contrasting/opacity/from_*/parse/arithmetic/bitwise,
//   HSV._from_internal(np.array([1,0,0,1])).to_hex() == '#FF0000',
//   rgba_to_color(np.array([1.0,0,0,0.5])) internal == [1,0,0,0.5],
//   rgba_to_color([255,0,0,128]) internal == [1,0,0,128/255].
// NOTE: ManimColor.from_rgb([1,0,0]) (ints) takes the int 0-255 path, so
// [1,0,0] is near-black; float input [1.0,...] is indistinguishable from int
// in JS and the port routes it the same way. Tests use 0-255 ints for
// from_rgb and true floats for from_hsv/from_hsl.
// NOTE 2: Python routes a *mixed* int/float list like [1.0,0,0,0.5] through
// the int (1/255) path (all(isinstance(x, float)) is False), yielding
// [0.0039,0,0,0.00196]. The TS from_rgba heuristic routes any 4-component
// array with all values in [0,1] through the float path instead (JS cannot
// distinguish 1.0 from 1), so that input is NOT asserted here; the float
// behavior is covered via a float64 NDArray, which both ports agree on.
// ---------------------------------------------------------------------------
function internalOf(color: ManimColor): number[] {
  return [...((color._internal_value.toArray() as unknown[]).map(Number))];
}

function expectInternal(color: ManimColor, expected: number[]): void {
  const got = internalOf(color);
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const g = got[i] ?? 0;
    const e = expected[i] ?? 0;
    if (Number.isNaN(e)) {
      expect(Number.isNaN(g)).toBe(true);
    } else {
      expect(close(g, e)).toBe(true);
    }
  }
}

describe("ManimColor", () => {
  it("invert without/with alpha", () => {
    expectInternal(new ManimColor("#112233").invert(), [0.9333333333333333, 0.8666666666666667, 0.8, 1.0]);
    expectInternal(new ManimColor("#112233").invert(true), [0.9333333333333333, 0.8666666666666667, 0.8, 0.0]);
  });

  it("interpolate blends toward the other color", () => {
    expectInternal(new ManimColor("#FF0000").interpolate(new ManimColor("#0000FF"), 0.25), [0.75, 0.0, 0.25, 1.0]);
    expect(new ManimColor("#FF0000").interpolate(new ManimColor("#0000FF"), 0.5).to_hex()).toBe("#7F007F");
  });

  it("contrasting returns white on dark, black on light", () => {
    expect(new ManimColor("#000000").contrasting().to_hex()).toBe("#FFFFFF");
    expect(new ManimColor("#FFFFFF").contrasting().to_hex()).toBe("#000000");
  });

  it("opacity replaces the alpha channel", () => {
    expectInternal(new ManimColor("#FF0000").opacity(0.3), [1.0, 0.0, 0.0, 0.3]);
  });

  it("from_rgb / from_hex / from_hsv / from_hsl", () => {
    expectInternal(ManimColor.from_rgb([255, 0, 0]), [1.0, 0.0, 0.0, 1.0]);
    expectInternal(ManimColor.from_hex("#112233"), [0.06666666666666667, 0.13333333333333333, 0.2, 1.0]);
    expectInternal(ManimColor.from_hsv([0.0, 1.0, 1.0]), [1.0, 0.0, 0.0, 1.0]);
    expectInternal(ManimColor.from_hsl([0.0, 1.0, 0.5]), [1.0, 0.0, 0.0, 1.0]);
  });

  it("parse handles a single color and a list", () => {
    expectInternal(ManimColor.parse("#FF0000") as ManimColor, [1.0, 0.0, 0.0, 1.0]);
    const parsed = ManimColor.parse(["#FF0000", "#00FF00"]) as ManimColor[];
    expect(parsed.map((c) => c.to_hex())).toEqual(["#FF0000", "#00FF00"]);
  });

  it("arithmetic with another color", () => {
    const r = new ManimColor("#FF0000");
    const b = new ManimColor("#0000FF");
    expectInternal(r.__add__(b), [1.0, 0.0, 1.0, 2.0]);
    expectInternal(r.__sub__(b), [1.0, 0.0, -1.0, 0.0]);
    expectInternal(r.__mul__(b), [0.0, 0.0, 0.0, 1.0]);
    expectInternal(r.__truediv__(b), [Infinity, NaN, 0.0, 1.0]);
    expectInternal(r.__floordiv__(b), [Infinity, NaN, 0.0, 1.0]);
    expectInternal(r.__mod__(b), [NaN, NaN, 0.0, 0.0]);
    expectInternal(r.__pow__(b), [1.0, 1.0, 0.0, 1.0]);
  });

  it("arithmetic with a scalar", () => {
    const r = new ManimColor("#FF0000");
    expectInternal(r.__add__(2), [3.0, 2.0, 2.0, 3.0]);
    expectInternal(r.__sub__(2), [-1.0, -2.0, -2.0, -1.0]);
    expectInternal(r.__mul__(2), [2.0, 0.0, 0.0, 2.0]);
    expectInternal(r.__truediv__(2), [0.5, 0.0, 0.0, 0.5]);
    expectInternal(r.__floordiv__(2), [0.0, 0.0, 0.0, 0.0]);
    expectInternal(r.__mod__(2), [1.0, 0.0, 0.0, 1.0]);
    expectInternal(r.__pow__(2), [1.0, 0.0, 0.0, 1.0]);
  });

  it("bitwise ops go through the integer representation", () => {
    const r = new ManimColor("#FF0000");
    const b = new ManimColor("#0000FF");
    expectInternal(r.__and__(b), [0.0, 0.0, 0.0, 1.0]);
    expectInternal(r.__or__(b), [1.0, 0.0, 1.0, 1.0]);
    expectInternal(r.__xor__(b), [1.0, 0.0, 1.0, 1.0]);
  });

  it("HSV._from_internal converts RGBA to HSV red", () => {
    const hsv = HSV._from_internal(array([1, 0, 0, 1]) as NDArray);
    expect(hsv.to_hex()).toBe("#FF0000");
  });

  it("rgba_to_color builds from float RGBA data (matches python)", () => {
    // Probe: rgba_to_color(np.array([1.0,0,0,0.5]))._internal_value
    // == [1, 0, 0, 0.5]; rgba_to_color([255,0,0,128]) ==
    // [1, 0, 0, 0.5019607843137255].
    expectInternal(rgba_to_color(array([1.0, 0.0, 0.0, 0.5]) as NDArray), [1.0, 0.0, 0.0, 0.5]);
    expectInternal(rgba_to_color([255, 0, 0, 128]), [1.0, 0.0, 0.0, 128 / 255]);
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols interacting with previously ported code.
// ---------------------------------------------------------------------------
describe("batch-1013 integration", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rate function output drives a spiral path alpha (matches python combo)", () => {
    // Probe: spiral_path(pi)(s, e, smooth(0.25)) with
    // s=[[0,0,0]], e=[[1,0,0]] and smooth(0.25)=0.07010371654510815.
    const alpha = rate.smooth(0.25);
    const path = spiral_path(Math.PI);
    const got = rowsOf(
      path(array([[0, 0, 0]]) as NDArray, array([[1, 0, 0]]) as NDArray, alpha),
    );
    expect(got.length).toBe(1);
    // Python re-evaluation of the same combo:
    // spiral_path(pi)([[0,0,0]], [[1,0,0]], smooth(0.25)).tolist()
    // == [[-0.06841040553554682, -0.015314943287485403, 0.0]].
    expectNum(got[0]?.[0] ?? 0, -0.06841040553554682, 1e-9);
    expectNum(got[0]?.[1] ?? 0, -0.015314943287485403, 1e-9);
  });

  it("VMobject chain: set_points + set_fill + set_sheen_direction + insert_n_curves", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 1, 0]]);
    v.set_fill("#00FF00", 0.8);
    v.set_sheen_direction([0, 1, 0]);
    expect(v.get_sheen_direction()).toEqual([0, 1, 0]);
    expectNum(v.fill_opacity, 0.8);
    const grown = v.insert_n_curves_to_point_list(1, v.points);
    expect(grown.shape[0]).toBe(v.points.shape[0] + 4);
  });

  it("Group of VMobjects: length_over_dim spans the family", () => {
    // Probe (python): Group(Square-ish corners at x=0..1, x=2..3)
    // length_over_dim(0) == 3.
    const a = new VMobject();
    a.set_points(array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) as NDArray);
    const b = new VMobject();
    b.set_points(array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]]) as NDArray);
    const g = new Group(a, b);
    expectNum(g.length_over_dim(0), 3.0);
    expectNum(g.length_over_dim(1), 1.0);
  });

  it("ManimColor.interpolate midpoint fills a VMobject (python combo)", () => {
    // Probe: ManimColor('#FF0000').interpolate(ManimColor('#0000FF'), 0.5)
    // .to_hex() == '#7F007F'; a VMobject filled from that hex keeps it.
    const mid = new ManimColor("#FF0000").interpolate(new ManimColor("#0000FF"), 0.5);
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.set_fill(mid.to_hex(), 1.0);
    expect(rowsOf(v.fill_rgbas)[0]?.slice(0, 3)).toEqual([127 / 255, 0, 127 / 255]);
  });

  it("compass direction can serve as a path axis (matches python combo)", () => {
    // Probe: path_along_arc(pi/2, LEFT)([[1,0,0]], [[0,1,0]], 1.0).tolist()
    // == [[1.0000000000000002, 0.9999999999999998, 0.0]] — a flipped axis
    // mirrors the arc, so alpha=1 does NOT land on the endpoint.
    const dirs = rowsOf(compass_directions(4));
    // NOTE: python prints 2.22e-16 here while the port yields 6.12e-17
    // (cos(pi/2)); both are float noise around 0, so compare loosely.
    expectArrClose(array([dirs[1]]) as NDArray, [[0, 1.0, 0.0]]);
    const path = path_along_arc(Math.PI / 2, dirs[2] ?? [0, 0, 1]);
    const out = rowsOf(
      path(array([[1, 0, 0]]) as NDArray, array([[0, 1, 0]]) as NDArray, 1.0),
    );
    expectNum(out[0]?.[0] ?? 0, 1.0000000000000002);
    expectNum(out[0]?.[1] ?? 0, 0.9999999999999998);
  });
});
