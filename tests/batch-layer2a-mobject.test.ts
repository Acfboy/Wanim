// Ground-truth tests for layer-2 batch, part A: Mobject core methods,
// VMobject geometry queries, ApplyWave homotopy, value trackers,
// vector fields, coordinate systems.
//
// Every expected value below was captured from the original Python manim
// (original manim) — see comments citing the exact probe.
// Float comparisons use 1e-9 absolute tolerance (values are O(1..10)).
import { describe, expect, it, afterEach } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Group, Mobject, setBackgroundRectangleFactory, setDefaultCameraFactory } from "../src/mobject/mobject.js";
import {
  VMobject,
  VGroup,
} from "../src/mobject/types/vectorized_mobject.js";
import { VMobjectFromSVGPath } from "../src/mobject/svg/svg_mobject.js";
import { ApplyWave } from "../src/animation/indication.js";
import { ComplexValueTracker, ValueTracker } from "../src/mobject/value_tracker.js";
import { VectorField } from "../src/mobject/vector_field.js";
import { Axes } from "../src/mobject/graphing/coordinate_systems.js";

function rowsOf(value: NDArray): number[][] {
  return (value.toArray() as number[][]).map((row) => row.map(Number));
}

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

function expectVecClose(actual: ArrayLike<number> | NDArray, expected: readonly number[], tol = 1e-9): void {
  const values = Array.isArray(actual)
    ? [...actual].map(Number)
    : (actual as NDArray).toArray() as number[];
  const flat = (values as unknown[]).flat(Infinity as number) as number[];
  expect(flat).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(Number(flat[i]), expected[i] ?? 0, tol)).toBe(true);
  }
}

function expectRowsClose(actual: NDArray, expected: readonly (readonly number[])[], tol = 1e-9): void {
  const rows = rowsOf(actual);
  expect(rows).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]).toHaveLength(expected[i]?.length ?? 0);
    for (let j = 0; j < (expected[i]?.length ?? 0); j += 1) {
      expect(close(rows[i]?.[j] ?? NaN, expected[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

/** A square-like VMobject (manim has Square; wanim geometry does not yet). */
function square(size = 2): VMobject {
  const v = new VMobject();
  const h = size / 2;
  v.set_points_as_corners([
    [-h, -h, 0],
    [h, -h, 0],
    [h, h, 0],
    [-h, h, 0],
    [-h, -h, 0],
  ]);
  return v;
}

afterEach(() => {
  setBackgroundRectangleFactory(null);
  Mobject.set_default();
});

// ---------------------------------------------------------------------------
// Mobject.add_updater / remove_updater / clear_updaters / suspend_updating
// ---------------------------------------------------------------------------
describe("Mobject.add_updater", () => {
  it("appends by default and inserts at index", () => {
    const m = new Mobject();
    const f = () => {};
    const g = () => {};
    const h = () => {};
    expect(m.add_updater(f)).toBe(m);
    m.add_updater(g);
    m.add_updater(h, 0);
    expect(m.updaters).toEqual([h, f, g]);
  });

  it("call_updater invokes 1-arg updater immediately (dt unused)", () => {
    // Python: add_updater(lambda x: calls.append(1), call_updater=True)
    const m = new Mobject();
    const calls: number[] = [];
    m.add_updater(() => {
      calls.push(1);
    }, null, true);
    expect(calls).toEqual([1]);
  });

  it("call_updater invokes dt-updater immediately with dt=0", () => {
    // Python ground truth: calls == [0] right after add_updater(..., call_updater=True)
    const m = new Mobject();
    const calls: number[] = [];
    m.add_updater((_mob, dt) => {
      calls.push(dt);
    }, null, true);
    expect(calls).toEqual([0]);
    // NOTE: Python detects dt-updaters via inspect.signature ("dt" in params);
    // the TS port uses function arity (length >= 2) instead, since JS has no
    // reliable parameter-name introspection. Standard 1-arg / 2-arg updaters
    // behave identically under both rules.
  });

  it("update(dt) propagates recursively; suspend blocks; resume re-fires with dt=0", () => {
    // Python ground truth sequence for [plain, dt] updaters:
    //   after add(call_updater on dt one): [0]
    //   after update(0.5): [0, 1, 0.5]
    //   after suspend + update(1.0): unchanged
    //   after resume_updating(): [0, 1, 0.5, 1, 0]
    //   after clear_updaters + update: unchanged
    const m = new Mobject();
    const calls: number[] = [];
    m.add_updater(() => {
      calls.push(1);
    });
    m.add_updater((_mob, dt: number) => {
      calls.push(dt);
    }, null, true);
    expect(calls).toEqual([0]);
    m.update(0.5);
    expect(calls).toEqual([0, 1, 0.5]);
    m.suspend_updating();
    expect(m.updating_suspended).toBe(true);
    m.update(1.0);
    expect(calls).toEqual([0, 1, 0.5]);
    m.resume_updating();
    expect(m.updating_suspended).toBe(false);
    expect(calls).toEqual([0, 1, 0.5, 1, 0]);
    m.clear_updaters();
    m.update(1.0);
    expect(calls).toEqual([0, 1, 0.5, 1, 0]);
  });

  it("update recurses into submobjects", () => {
    const parent = new Group(new Mobject(), new Mobject());
    const seen: string[] = [];
    parent.add_updater(() => {
      seen.push("parent");
    });
    (parent.submobjects[0] as Mobject).add_updater(() => {
      seen.push("child");
    });
    parent.update(0.25);
    expect(seen).toEqual(["parent", "child"]);
  });
});

describe("Mobject.remove_updater", () => {
  it("removes every instance of a duplicated updater", () => {
    // Python: add twice, remove once -> len(updaters) == 0 (while-loop removal)
    const m = new Mobject();
    const f = () => {};
    m.add_updater(f);
    m.add_updater(f);
    expect(m.remove_updater(f)).toBe(m);
    expect(m.updaters).toHaveLength(0);
  });

  it("clear_updaters recurses by default", () => {
    const child = new Mobject();
    child.add_updater(() => {});
    const parent = new Group(child);
    parent.add_updater(() => {});
    parent.clear_updaters();
    expect(parent.updaters).toHaveLength(0);
    expect(child.updaters).toHaveLength(0);
  });

  it("clear_updaters(recursive=false) keeps child updaters", () => {
    const child = new Mobject();
    child.add_updater(() => {});
    const parent = new Group(child);
    parent.clear_updaters(false);
    expect(child.updaters).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Mobject.__iter__ / __len__ / split / nonempty_submobjects / shuffle / invert
// ---------------------------------------------------------------------------
describe("Mobject iteration protocol", () => {
  it("__iter__ yields nothing for an empty mobject", () => {
    // Python: list(Mobject()) == []
    expect([...new Mobject()]).toEqual([]);
  });

  it("__iter__ yields self when it has points, then submobjects", () => {
    // Python: list(Square()) == [Square] (16 points)
    const s = square();
    const items = [...s];
    expect(items).toHaveLength(1);
    expect(items[0]).toBe(s);
    const g = new Group(new Mobject(), new Mobject());
    expect([...g]).toEqual(g.submobjects);
  });

  it("__len__ counts self-with-points plus submobjects", () => {
    // Python: len(Mobject()) == 0, len(Square()) == 1
    expect(new Mobject().__len__()).toBe(0);
    expect(square().__len__()).toBe(1);
    expect(new Group(new Mobject(), new Mobject()).__len__()).toBe(2);
  });

  it("__contains__ mirrors Python `in` via __iter__ (identity)", () => {
    // Python: manim's Mobject defines no __contains__, so `x in g` scans
    // __iter__ (self-with-points first, then submobjects).  Notably
    // rapydscript's `ρσ_in` probes `typeof arr.__contains__`, which raised
    // AttributeError through the Mobject proxy before this method existed
    // (observed as `m not in numbers` on a VGroup).
    const a = square();
    const b = square();
    const g = new VGroup(a, b);
    expect(g.__contains__(a)).toBe(true);
    expect(g.__contains__(b)).toBe(true);
    expect(g.__contains__(square())).toBe(false);
    expect(new VGroup().__contains__(a)).toBe(false);
    // A mobject with points contains itself via __iter__ order.
    const s = square();
    expect(s.__contains__(s)).toBe(true);
    expect(s.__contains__(a)).toBe(false);
    const h = new Group(new Mobject(), new Mobject());
    for (const member of h.submobjects) expect(h.__contains__(member)).toBe(true);
    expect(h.__contains__(new Mobject())).toBe(false);
    // Complements split like the determinant-puzzle scene does:
    // every part is either in `numbers` or in `non_numbers`, never both.
    const parts = [square(), square(), square(), square()];
    const det = new VGroup(...parts);
    const numbers = new VGroup(parts[1]!, parts[3]!);
    const nonNumbers = new VGroup(...parts.filter((m) => !numbers.__contains__(m)));
    expect(nonNumbers.submobjects).toEqual([parts[0], parts[2]]);
    for (const part of parts) {
      expect(det.__contains__(part)).toBe(true);
      expect(numbers.__contains__(part) !== nonNumbers.__contains__(part)).toBe(true);
    }
  });

  it("split returns [self] plus submobjects when there are points", () => {
    // Python: VGroup(sq, sq).split() == [sq, sq] (VGroup itself has no points)
    const a = square();
    const b = square();
    const g = new VGroup(a, b);
    expect(g.split()).toEqual([a, b]);
    const s = square();
    expect(s.split()).toEqual([s]);
  });

  it("nonempty_submobjects filters pointless, childless submobjects", () => {
    // Python: VGroup(Square(), VMobject()).nonempty_submobjects() has length 1
    const g = new VGroup(square(), new VMobject());
    expect(g.nonempty_submobjects()).toHaveLength(1);
    const full = new VGroup(square(), square());
    expect(full.nonempty_submobjects()).toHaveLength(2);
  });

  it("shuffle preserves membership and invert reverses order", () => {
    const members = [new Mobject(), new Mobject(), new Mobject(), new Mobject()];
    const g = new Group(...members);
    g.shuffle();
    // NOTE: compare by identity — Mobject proxies stringify to their shared
    // name, so Array.sort() cannot order them, and each carries a random
    // original_id that defeats deep equality.
    expect(g.submobjects).toHaveLength(members.length);
    for (const member of members) expect(g.submobjects).toContain(member);
    const h = new Group(...members);
    h.invert();
    expect(h.submobjects).toEqual([...members].reverse());
  });

  it("invert(recursive=true) inverts nested families", () => {
    const inner = new Group(new Mobject(), new Mobject(), new Mobject());
    const before = [...inner.submobjects];
    const outer = new Group(inner, new Mobject());
    outer.invert(true);
    expect(inner.submobjects).toEqual([...before].reverse());
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_family / insert / set_default / get_image
// ---------------------------------------------------------------------------
describe("Mobject.get_family", () => {
  it("lists self plus descendants depth-first without duplicates", () => {
    // Python: Group(VGroup(s, r), m, v).get_family() ==
    //   [Group, VGroup(Square, Rectangle), Square, Rectangle, Mobject, VMobject]
    const s = square();
    const r = square();
    const vg = new VGroup(s, r);
    const m = new Mobject();
    const v = new VMobject();
    const gr = new Group(vg, m, v);
    expect(gr.get_family()).toEqual([gr, vg, s, r, m, v]);
  });
});

describe("Mobject.insert", () => {
  it("inserts at the given index", () => {
    // Python: add(a, b); insert(1, c) -> order [a, c, b]
    const a = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    const g = new Group();
    g.add(a, b);
    expect(g.insert(1, c)).toBe(g);
    expect(g.submobjects).toEqual([a, c, b]);
  });
});

describe("Mobject.set_default", () => {
  it("overrides constructor defaults until reset", () => {
    // Python: Square.set_default(color=GREEN, fill_opacity=0.25) -> s.color #83C167;
    // Square.set_default() restores (#FFFFFF, 0.0).
    Mobject.set_default({ color: "GREEN" });
    const green = new Mobject().color.to_hex();
    expect(green.toUpperCase()).toBe("#83C167");
    Mobject.set_default();
    expect(new Mobject().color.to_hex().toUpperCase()).toBe("#FFFFFF");
  });
});

describe("Mobject.get_image", () => {
  it("throws a clearly-marked TODO without a raster camera backend", () => {
    // Python raises NameError at runtime (Camera is not imported in
    // mobject.py); the TS port raises a TODO error instead — cairo rendering
    // is a later roadmap step.
    expect(() => square().get_image()).toThrow(/TODO.*Camera/i);
  });

  it("delegates to an injected camera factory when present", () => {
    const seen: Mobject[] = [];
    setDefaultCameraFactory(() => ({
      capture_mobject: (m: Mobject) => {
        seen.push(m);
      },
      get_image: () => "fake-image",
    }));
    const s = square();
    expect(s.get_image()).toBe("fake-image");
    expect(seen).toEqual([s]);
    setDefaultCameraFactory(null);
  });
});

// ---------------------------------------------------------------------------
// Mobject.set_color / fade / apply_function_to_submobject_positions /
// add_background_rectangle_to_submobjects / interpolate
// ---------------------------------------------------------------------------
describe("Mobject.set_color", () => {
  it("sets color on self and recurses into the family", () => {
    // Python: Square().set_color(RED) -> color #FC6255, fill_opacity stays 0.0
    const s = square();
    s.set_color("RED");
    expect(s.color.to_hex().toUpperCase()).toBe("#FC6255");
    expect(s.get_fill_opacity()).toBe(0);
    const inner = square();
    const g = new Group(inner);
    g.set_color("BLUE");
    expect(inner.color.to_hex().toUpperCase()).toBe("#58C4DD");
  });
});

describe("Mobject.fade", () => {
  it("base Mobject.fade only recurses (no self opacity change)", () => {
    // Python Mobject.fade: `if family: for submob: submob.fade(...)` then return.
    const g = new Group(new Mobject());
    expect(g.fade(0.3)).toBe(g);
  });

  it("VMobject.fade scales fill and stroke opacities", () => {
    // Python: Square fill 0.0 / stroke 1.0 -> fade(0.5) -> (0.0, 0.5)
    const s = square();
    expect(s.get_fill_opacity()).toBe(0);
    expect(s.get_stroke_opacity()).toBe(1);
    s.fade(0.5);
    expect(close(s.get_fill_opacity(), 0)).toBe(true);
    expect(close(s.get_stroke_opacity(), 0.5)).toBe(true);
  });

  it("VMobject.fade recurses into submobjects", () => {
    const a = square();
    const g = new VGroup(a);
    g.fade(0.5);
    expect(close(a.get_stroke_opacity(), 0.5)).toBe(true);
  });
});

describe("VMobject.set_opacity", () => {
  it("sets fill, stroke, and background opacities (Python VMobject parity)", () => {
    // Python: Square() fill 0.0 / stroke 1.0 / background 1.0 ->
    // set_opacity(0.25) -> (0.25, 0.25, 0.25); fill rgbas[0] == [1,1,1,0.25].
    const s = square();
    expect(s.set_opacity(0.25)).toBe(s);
    expect(close(s.get_fill_opacity(), 0.25)).toBe(true);
    expect(close(s.get_stroke_opacity(), 0.25)).toBe(true);
    expect(close(s.get_stroke_opacity(true), 0.25)).toBe(true);
    const fillRows = rowsOf(s.get_fill_rgbas());
    expectVecClose(fillRows[0] ?? [], [1, 1, 1, 0.25]);
  });

  it("recurses into the family when family=true", () => {
    // Python: VGroup(g, k).set_opacity(0.5) -> g.fill 0.5, k.background 0.5.
    const g = square();
    const k = square();
    const vg = new VGroup(g, k);
    vg.set_opacity(0.5);
    expect(close(g.get_fill_opacity(), 0.5)).toBe(true);
    expect(close(k.get_stroke_opacity(true), 0.5)).toBe(true);
  });
});

describe("Mobject.apply_function_to_submobject_positions", () => {
  it("moves each submobject to func(center)", () => {
    // Python: Group(Dot(), Dot()) shifted by [1,0,0] -> centers [[1,0,0],[1,0,0]]
    const a = square();
    const b = square();
    const g = new Group(a, b);
    g.apply_function_to_submobject_positions((point) => [point[0] + 1, point[1], point[2]]);
    for (const child of [a, b]) {
      expectVecClose(child.get_center(), [1, 0, 0]);
    }
  });
});

describe("Mobject.add_background_rectangle_to_submobjects", () => {
  it("is a no-op for a mobject without submobjects", () => {
    // Python: Square().add_background_rectangle_to_submobjects() adds nothing.
    const s = square();
    expect(s.add_background_rectangle_to_submobjects()).toBe(s);
    expect(s.submobjects).toHaveLength(0);
  });

  it("delegates each submobject to the registered background factory", () => {
    // Python passes **kwargs straight through to add_background_rectangle;
    // the TS port splits {color, opacity, ...rest} and forwards positionally.
    const received: Array<{ color: unknown; opacity: unknown; rest: unknown }> = [];
    setBackgroundRectangleFactory((mob, color, opacity, kwargs) => {
      received.push({ color, opacity, rest: kwargs });
      return new VMobject();
    });
    const g = new Group(square(), square());
    g.add_background_rectangle_to_submobjects({ color: "RED", opacity: 0.5, extra: 1 });
    expect(received).toHaveLength(2);
    expect(received[0]?.opacity).toBe(0.5);
    expect(received[0]?.rest).toEqual({ extra: 1 });
  });
});

describe("Mobject.interpolate", () => {
  it("base Mobject.interpolate raises via interpolate_color", () => {
    // Python: Mobject().interpolate(m1, m2, 0.25) raises NotImplementedError
    // ("Please override in a child class.") from interpolate_color.
    const m1 = new Mobject();
    m1.set_points([[0, 0, 0]]);
    const m2 = new Mobject();
    m2.set_points([[2, 4, 6]]);
    expect(() => new Mobject().interpolate(m1, m2, 0.25)).toThrowError(/override in a child class/);
  });

  it("VMobject.interpolate lerps points and colors", () => {
    // Python: Square().interpolate(sq, blue-shifted-sq, 0.25) ->
    // center [0.25, 0.25, 0], fill #D5F0F6 (white@0 op lerp blue@.25).
    const first = square(2);
    const second = square(2);
    second.set_color("BLUE");
    second.shift([1, 1, 0]);
    const result = square(2);
    result.interpolate(first, second, 0.25);
    const c1 = first.get_center();
    const c2 = second.get_center();
    const expected = c1.map((v, i) => v * 0.75 + (c2[i] ?? 0) * 0.25);
    expectVecClose(result.get_center(), expected);
    expect(result.get_fill_color().to_hex().toUpperCase()).toBe("#D5F0F6");
  });
});

// ---------------------------------------------------------------------------
// VMobject.__init__
// ---------------------------------------------------------------------------
describe("VMobject.__init__", () => {
  it("matches Python defaults", () => {
    // Captured from manim 0.21.0 VMobject():
    // fill_op 0, stroke_op 1, stroke_w 4, bg BLACK/1.0/0, sheen 0/UL,
    // close_new False, pre 0.01, smooth False, bgimg None, shade False,
    // tol 1e-6, nppc 4, cap/joint AUTO.
    const v = new VMobject();
    expect(v.n_points_per_cubic_curve).toBe(4);
    expect(v.fill_opacity).toBe(0);
    expect(v.stroke_opacity).toBe(1);
    expect(v.stroke_width).toBe(4);
    expect(v.background_stroke_color?.to_hex().toUpperCase()).toBe("#000000");
    expect(v.background_stroke_opacity).toBe(1);
    expect(v.background_stroke_width).toBe(0);
    expect(v.sheen_factor).toBe(0);
    expect(v.sheen_direction).toEqual([-1, 1, 0]);
    expect(v.close_new_points).toBe(false);
    expect(v.pre_function_handle_to_anchor_scale_factor).toBe(0.01);
    expect(v.make_smooth_after_applying_functions).toBe(false);
    expect(v.background_image).toBe(null);
    expect(v.shade_in_3d).toBe(false);
    expect(v.tolerance_for_point_equality).toBe(1e-6);
    expect(String(v.cap_style)).toContain("AUTO");
    expect(String(v.joint_type)).toContain("AUTO");
  });

  it("color kwarg drives fill and stroke; fill_color overrides fill only", () => {
    // Python: VMobject(color=RED) -> fill+stroke #FC6255;
    // VMobject(fill_color=BLUE) -> fill #58C4DD, stroke stays #FFFFFF.
    const both = new VMobject({ color: "RED" });
    expect(both.get_fill_color().to_hex().toUpperCase()).toBe("#FC6255");
    expect(both.get_stroke_color().to_hex().toUpperCase()).toBe("#FC6255");
    const fillOnly = new VMobject({ fill_color: "BLUE" });
    expect(fillOnly.get_fill_color().to_hex().toUpperCase()).toBe("#58C4DD");
    expect(fillOnly.get_stroke_color().to_hex().toUpperCase()).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// VMobject.get_subpaths / get_cubic_bezier_tuples / get_subcurve / get_direction
// ---------------------------------------------------------------------------
describe("VMobject.get_subpaths", () => {
  it("returns one subpath per continuous run of curves", () => {
    // Python: two disjoint 4-point runs -> 2 subpaths of shape (4, 3).
    const v = new VMobject();
    v.set_points([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
      [10, 10, 0],
      [11, 10, 0],
      [12, 10, 0],
      [13, 10, 0],
    ]);
    const subs = v.get_subpaths();
    expect(subs).toHaveLength(2);
    expect(Array.from(subs[0]?.shape ?? [])).toEqual([4, 3]);
    expect(Array.from(subs[1]?.shape ?? [])).toEqual([4, 3]);
  });

  it("keeps a single continuous path whole", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
    ]);
    expect(v.get_subpaths()).toHaveLength(1);
  });
});

describe("VMobject.get_cubic_bezier_tuples", () => {
  it("groups points into 4-point tuples", () => {
    // Python: corners (0,0,0)->(4,0,0) ->
    // [[[0,0,0],[1.333,0,0],[2.667,0,0],[4,0,0]]]
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
    ]);
    const tuples = v.get_cubic_bezier_tuples();
    expect(tuples).toHaveLength(1);
    expectRowsClose(array(tuples[0] ?? []) as NDArray, [
      [0, 0, 0],
      [1.3333333333333333, 0, 0],
      [2.6666666666666665, 0, 0],
      [4, 0, 0],
    ]);
  });
});

describe("VMobject.get_subcurve", () => {
  it("extracts a partial single-curve segment", () => {
    // Python: subcurve(0.25, 0.75) of the (0,0,0)->(4,0,0) line ->
    // [[1,0,0],[1.667,0,0],[2.333,0,0],[3,0,0]]
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
    ]);
    const sub = v.get_subcurve(0.25, 0.75);
    expectRowsClose(sub.points, [
      [1, 0, 0],
      [1.6666666666666665, 0, 0],
      [2.333333333333333, 0, 0],
      [3, 0, 0],
    ]);
  });

  it("full range reproduces the original points", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
    ]);
    expectRowsClose(v.get_subcurve(0, 1).points, rowsOf(v.points));
  });

  it("half of a 4-curve closed loop yields 3 curves starting at the origin", () => {
    // Python: closed 4-corner loop, subcurve(0, 0.5) -> 12 points, first [0,0,0].
    const c = new VMobject();
    c.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
      [4, 4, 0],
      [0, 4, 0],
      [0, 0, 0],
    ]);
    expect(c.is_closed()).toBe(true);
    const half = c.get_subcurve(0, 0.5);
    expect(half.get_num_points()).toBe(12);
    expectVecClose(rowsOf(half.points)[0] ?? [], [0, 0, 0]);
  });

  it("wraps around a closed loop when a > b", () => {
    // Python: subcurve(0.75, 0.25) -> 12 points, first [0,4,0], last [4,0,0].
    const c = new VMobject();
    c.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
      [4, 4, 0],
      [0, 4, 0],
      [0, 0, 0],
    ]);
    const wrapped = c.get_subcurve(0.75, 0.25);
    expect(wrapped.get_num_points()).toBe(12);
    const pts = rowsOf(wrapped.points);
    expectVecClose(pts[0] ?? [], [0, 4, 0]);
    expectVecClose(pts[pts.length - 1] ?? [], [4, 0, 0]);
  });
});

describe("VMobject.get_direction", () => {
  it("reports CCW for counter-clockwise and CW for clockwise loops", () => {
    // Python: Circle().get_direction() == 'CCW'; Square() == 'CCW'.
    const ccw = new VMobject();
    ccw.set_points_as_corners([
      [0, 0, 0],
      [4, 0, 0],
      [4, 4, 0],
      [0, 4, 0],
      [0, 0, 0],
    ]);
    expect(ccw.get_direction()).toBe("CCW");
    const cw = new VMobject();
    cw.set_points_as_corners([
      [0, 0, 0],
      [0, 4, 0],
      [4, 4, 0],
      [4, 0, 0],
      [0, 0, 0],
    ]);
    expect(cw.get_direction()).toBe("CW");
  });
});

// ---------------------------------------------------------------------------
// ApplyWave.__init__.homotopy
// ---------------------------------------------------------------------------
describe("ApplyWave homotopy", () => {
  function waveMobject(): VMobject {
    // Same extent as manim Line([-2,0,0],[2,0,0]): x in [-2, 2].
    const v = new VMobject();
    v.set_points_as_corners([
      [-2, 0, 0],
      [2, 0, 0],
    ]);
    return v;
  }

  it("matches Python homotopy values (defaults: UP, amplitude 0.2, tw 1)", () => {
    // Python: homotopy(0,0,0,0)   -> [0,0,0]
    //         homotopy(0,0,0,0.5) -> [0,0.2,0]
    //         homotopy(1,0.5,0,0.5) -> [1,0.6,0]
    //         homotopy(-2,1,0,1)  -> [-2,1,0]
    //         homotopy(2,1,0,0.25)-> [2,1,0]
    const anim = new ApplyWave(waveMobject());
    expectVecClose(anim.homotopy(0, 0, 0, 0), [0, 0, 0]);
    expectVecClose(anim.homotopy(0, 0, 0, 0.5), [0, 0.2, 0]);
    expectVecClose(anim.homotopy(1, 0.5, 0, 0.5), [1, 0.6, 0]);
    expectVecClose(anim.homotopy(-2, 1, 0, 1), [-2, 1, 0]);
    expectVecClose(anim.homotopy(2, 1, 0, 0.25), [2, 1, 0]);
  });

  it("matches Python for ripples=2 and custom direction/amplitude/width", () => {
    // Python: ripples=2: homotopy(0,0,0,0.5)->[0,0,0], (1,0,0,0.5)->[1,0.2,0]
    // time_width=0.5, amplitude=0.4, direction=[0,0,1]:
    //   (0,0,0,0.5)->[0,0,0.4], (0,0,0,0.75)->[0,0,0]
    const rippled = new ApplyWave(waveMobject(), undefined, 0.2, undefined, 1, 2);
    expectVecClose(rippled.homotopy(0, 0, 0, 0.5), [0, 0, 0]);
    expectVecClose(rippled.homotopy(1, 0, 0, 0.5), [1, 0.2, 0]);
    const custom = new ApplyWave(waveMobject(), [0, 0, 1], 0.4, undefined, 0.5);
    expectVecClose(custom.homotopy(0, 0, 0, 0.5), [0, 0, 0.4]);
    expectVecClose(custom.homotopy(0, 0, 0, 0.75), [0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// ComplexValueTracker.get_value
// ---------------------------------------------------------------------------
describe("ComplexValueTracker.get_value", () => {
  it("returns the complex value stored in points", () => {
    // Python: ComplexValueTracker(-2+1j).get_value() == (-2+1j);
    // set_value(3+2j) -> points [[3, 2, 0]].
    const tracker = new ComplexValueTracker({ re: -2, im: 1 });
    expect(tracker.get_value()).toEqual({ re: -2, im: 1 });
    tracker.set_value({ re: 3, im: 2 });
    expect(tracker.get_value()).toEqual({ re: 3, im: 2 });
    expectRowsClose(tracker.points, [[3, 2, 0]]);
  });

  it("plain ValueTracker still returns a number", () => {
    const tracker = new ValueTracker(2.5);
    expect(tracker.get_value()).toBe(2.5);
  });
});

// ---------------------------------------------------------------------------
// VectorField.get_colored_background_image
// ---------------------------------------------------------------------------
describe("VectorField.get_colored_background_image", () => {
  it("returns a uint8 (ph, pw, 3) array honoring the color gradient", () => {
    // Python (default config 1920x1080, sampling_rate=240 -> (4, 8, 3)):
    //   constant [0.5,0,0] -> [106,171,112] everywhere
    //   constant [1.5,0,0] -> [248,187,104]
    //   constant [0,0,0]   -> [35,107,142]
    const vf = new VectorField(() => [0.5, 0, 0]);
    const image = vf.get_colored_background_image(240);
    expect(Array.from(image.shape)).toEqual([4, 8, 3]);
    const pixels = image.toArray() as number[][][];
    for (const row of pixels) {
      for (const pixel of row) expect(pixel.map(Math.trunc)).toEqual([106, 171, 112]);
    }
    const hot = new VectorField(() => [1.5, 0, 0]);
    expect((hot.get_colored_background_image(240).toArray() as number[][][])[0]?.[0]?.map(Math.trunc)).toEqual([248, 187, 104]);
    const cold = new VectorField(() => [0, 0, 0]);
    expect((cold.get_colored_background_image(240).toArray() as number[][][])[0]?.[0]?.map(Math.trunc)).toEqual([35, 107, 142]);
  });

  it("raises ValueError for single-color fields", () => {
    // Python: "There is no point in generating an image if the vector field
    // uses a single color."
    const vf = new VectorField(() => [1, 0, 0], "#FF0000");
    let thrown: unknown;
    try {
      vf.get_colored_background_image();
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toContain("single color");
  });
});

// ---------------------------------------------------------------------------
// CoordinateSystem.pr2pt / pt2pr
// ---------------------------------------------------------------------------
describe("CoordinateSystem.pr2pt / pt2pr", () => {
  it("converts polar to point and back", () => {
    // Python Axes(x_range=[-3,3,1], y_range=[-2,2,1], x_length=6, y_length=4):
    //   pr2pt(1, 0) == [1, 0, 0]; pt2pr(c2p(1, 0)) == [1, ~4.4e-16].
    const axes = new Axes([-3, 3, 1], [-2, 2, 1], 6, 4);
    expectVecClose(axes.pr2pt(1.0, 0.0), [1, 0, 0]);
    // NOTE: coords_to_point with scalar-per-axis input returns a flat 1-D point.
    const flat = (axes.coords_to_point(1, 0).toArray() as number[]).map(Number);
    const back = axes.pt2pr(flat);
    expect(close(back[0] ?? NaN, 1)).toBe(true);
    expect(close(back[1] ?? NaN, 0)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// VMobjectFromSVGPath.handle_commands.add_quad (via QuadraticBezier segments)
// ---------------------------------------------------------------------------
describe("VMobjectFromSVGPath quadratic handling", () => {
  it("converts quadratics with (start+2cp)/3, (2cp+end)/3 (nppc=4)", () => {
    // Python add_quad: add_cubic(start, (start+cp+cp)/3, (cp+cp+end)/3, end).
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "QuadraticBezier", control: [1, 1], end: [2, 0] },
    ]);
    expectRowsClose(v.points, [[0, 0, 0], [2 / 3, 2 / 3, 0], [4 / 3, 2 / 3, 0], [2, 0, 0]]);
  });
});
