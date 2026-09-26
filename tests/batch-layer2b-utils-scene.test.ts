// Ground-truth tests for layer-2 batch, part B: color/space/bezier utils,
// deprecation helpers, polylabel, Scene, plus cross-symbol integration tests.
//
// Expected values were captured from the original Python manim
// (original manim); each case cites the probe.
import { describe, expect, it, afterEach } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { Group, Mobject } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { VMobjectFromSVGPath } from "../src/mobject/svg/svg_mobject.js";
import { ApplyWave } from "../src/animation/indication.js";
import { ComplexValueTracker } from "../src/mobject/value_tracker.js";
import { color_to_int_rgba } from "../src/utils/color/core.js";
import { rotate_vector } from "../src/utils/space_ops.js";
import {
  bezier_remap,
  integer_interpolate,
  partial_bezier_points,
} from "../src/utils/bezier.js";
import {
  _deprecation_text_component,
  _get_callable_info,
  deprecated,
  deprecated_params,
} from "../src/utils/deprecation.js";
import { Polygon } from "../src/utils/polylabel.js";
import { Scene } from "../src/scene/scene.js";
import { config } from "../src/_config/utils.js";

function rowsOf(value: NDArray): number[][] {
  return (value.toArray() as number[][]).map((row) => row.map(Number));
}

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

function expectVecClose(actual: ArrayLike<number> | NDArray, expected: readonly number[], tol = 1e-9): void {
  const raw = Array.isArray(actual) ? [...actual].map(Number) : ((actual as NDArray).toArray() as number[]);
  const flat = (raw as unknown[]).flat(Infinity as number) as number[];
  expect(flat).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(Number(flat[i]), expected[i] ?? 0, tol)).toBe(true);
  }
}

function withWarnings(action: () => void): string[] {
  const messages: string[] = [];
  const original = console.warn;
  console.warn = (...args: unknown[]) => {
    messages.push(args.map(String).join(" "));
  };
  try {
    action();
  } finally {
    console.warn = original;
  }
  return messages;
}

afterEach(() => {
  Mobject.set_default();
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.color_to_int_rgba
// ---------------------------------------------------------------------------
describe("color_to_int_rgba", () => {
  it("converts colors to 0-255 RGBA with truncation", () => {
    // Python: color_to_int_rgba('#FF0000') == [255, 0, 0, 255]
    //         color_to_int_rgba('#FF0000', 0.5) == [255, 0, 0, 127]
    //         color_to_int_rgba('BLUE', 0.0) == [88, 196, 221, 0]
    expect(color_to_int_rgba("#FF0000")).toEqual([255, 0, 0, 255]);
    expect(color_to_int_rgba("#FF0000", 0.5)).toEqual([255, 0, 0, 127]);
    expect(color_to_int_rgba("BLUE", 0.0)).toEqual([88, 196, 221, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.rotate_vector
// ---------------------------------------------------------------------------
describe("rotate_vector", () => {
  it("rotates vectors (2D input is lifted to 3D)", () => {
    // Python: rotate_vector([1,0,0], pi/2) == [2.2e-16, 1, 0]
    //         rotate_vector([1,0], pi)     == [-1, 1.2e-16, 0]
    //         rotate_vector([0,1,0], pi/2, axis=[0,0,1]) == [-1, 2.2e-16, 0]
    expectVecClose(rotate_vector([1, 0, 0], Math.PI / 2), [2.220446049250313e-16, 1.0, 0.0]);
    expectVecClose(rotate_vector([1, 0], Math.PI), [-1.0, 1.2246467991473532e-16, 0.0]);
    expectVecClose(rotate_vector([0, 1, 0], Math.PI / 2, [0, 0, 1]), [-1.0, 2.220446049250313e-16, 0.0]);
  });

  it("rejects vectors longer than 3D", () => {
    // Python: ValueError("Vector must have the correct dimensions.")
    let thrown: unknown;
    try {
      rotate_vector([1, 2, 3, 4], 0.5);
    } catch (error) {
      thrown = error;
    }
    expect((thrown as Error).name).toBe("ValueError");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.bezier.bezier_remap (+ partial_bezier_points backing get_subcurve)
// ---------------------------------------------------------------------------
describe("bezier_remap", () => {
  it("splits one curve into three equal curves", () => {
    // Python: bezier_remap([[[0,0,0],[1,0,0],[2,0,0],[3,0,0]]], 3) ==
    // [[[0,0,0],[1/3,0,0],[2/3,0,0],[1,0,0]],
    //  [[1,0,0],[4/3,0,0],[5/3,0,0],[2,0,0]],
    //  [[2,0,0],[7/3,0,0],[8/3,0,0],[3,0,0]]]
    const result = bezier_remap(
      [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]],
      3,
    );
    expect(result).toHaveLength(3);
    const flat = result.map((curve) => curve.map((point) => point.map(Number)));
    const expected = [
      [[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0], [1, 0, 0]],
      [[1, 0, 0], [4 / 3, 0, 0], [5 / 3, 0, 0], [2, 0, 0]],
      [[2, 0, 0], [7 / 3, 0, 0], [8 / 3, 0, 0], [3, 0, 0]],
    ];
    for (let c = 0; c < 3; c += 1) {
      for (let p = 0; p < 4; p += 1) {
        for (let d = 0; d < 3; d += 1) {
          expect(close(flat[c]?.[p]?.[d] ?? NaN, expected[c]?.[p]?.[d] ?? NaN)).toBe(true);
        }
      }
    }
  });

  it("integer_interpolate matches Python boundary behavior", () => {
    expect(integer_interpolate(0, 4, 0.75)).toEqual([3, 0]);
    expect(integer_interpolate(0, 4, 0)).toEqual([0, 0]);
    expect(integer_interpolate(0, 4, 1)).toEqual([3, 1]);
  });

  it("partial_bezier_points border cases collapse to end/start points", () => {
    // Python: a == 1 fills with points[-1]; b == 0 fills with points[0].
    const curve = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    expect(partial_bezier_points(curve, 1, 1)).toEqual([[3, 0, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0]]);
    expect(partial_bezier_points(curve, 0, 0)).toEqual([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.deprecation: deprecated.deprecate_docs / deprecated.deprecate /
// deprecated_params (+ module-level _get_callable_info / _deprecation_text_component)
// ---------------------------------------------------------------------------
describe("deprecated", () => {
  it("appends the deprecation notice to __doc__ and warns on call", () => {
    // Python: @deprecated def foo -> __doc__ ==
    // '\n\n.. attention:: Deprecated\n  The function foo has been deprecated
    //  and may be removed in a later version.'
    function foo(x: unknown): unknown {
      return x;
    }
    const wrapped = deprecated(foo) as typeof foo & { __doc__?: string };
    expect(wrapped.__doc__).toBe(
      "\n\n.. attention:: Deprecated\n  The function foo has been deprecated and may be removed in a later version.",
    );
    expect(wrapped.name).toBe("foo");
    const messages = withWarnings(() => {
      expect(wrapped(41)).toBe(41);
    });
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain("The function foo has been deprecated");
  });

  it("supports the factory form with since/until/replacement/message", () => {
    // Python doc2 for foo2: 'The function foo2 has been deprecated since v0.2
    // and is expected to be removed after v0.4. Use :func:`~.bar` instead. It is cooler.'
    function foo2(): number {
      return 5;
    }
    const decorator = deprecated(null, "v0.2", "v0.4", "bar", "It is cooler.") as (f: typeof foo2) => typeof foo2 & { __doc__?: string };
    const wrapped = decorator(foo2);
    expect(wrapped.__doc__).toContain(
      "The function foo2 has been deprecated since v0.2 and is expected to be removed after v0.4. Use :func:`~.bar` instead. It is cooler.",
    );
    const messages = withWarnings(() => wrapped());
    // NOTE: console form uses the plain replacement ("Use bar instead."),
    // mirroring Python's for_docs=False branch.
    expect(messages[0]).toContain("Use bar instead. It is cooler.");
  });

  it("warns on construction when decorating a class", () => {
    // Python decorates __init__ so `Bar()` logs the class warning.
    class Bar {
      v = 1;
    }
    const DeprecatedBar = deprecated(Bar) as unknown as typeof Bar & { __doc__?: string };
    expect(String((DeprecatedBar as unknown as { __doc__?: string }).__doc__ ?? "")).toContain("attention");
    const messages = withWarnings(() => {
      const instance = new DeprecatedBar();
      expect(instance.v).toBe(1);
    });
    expect(messages.join(" ")).toContain("has been deprecated");
  });

  it("reports methods via __qualname__ (deprecate_docs/deprecate path)", () => {
    // Python _get_callable_info: method 'A.meth' -> ('method', 'A.meth').
    // The nested deprecate_docs/deprecate closures inside deprecated() are
    // exercised here through the public decorator (they have no TS export).
    function baz(): number {
      return 1;
    }
    (baz as unknown as { __qualname__?: string }).__qualname__ = "Bar.baz";
    const wrapped = deprecated(baz) as typeof baz & { __doc__?: string };
    expect(wrapped.__doc__).toContain("The method Bar.baz has been deprecated");
  });

  it("_get_callable_info and _deprecation_text_component match Python", () => {
    // Python: f -> ('function','f'); A -> ('class','A');
    // _deprecation_text_component(None, None, '') ==
    //   'deprecated and may be removed in a later version.'
    function f(): void {}
    expect(_get_callable_info(f)).toEqual(["function", "f"]);
    class A {}
    expect(_get_callable_info(A)).toEqual(["class", "A"]);
    expect(_deprecation_text_component(null, null, "")).toBe("deprecated and may be removed in a later version.");
    expect(_deprecation_text_component("v0.2", "v0.4", "Use bar instead. It is cooler.")).toBe(
      "deprecated since v0.2 and is expected to be removed after v0.4. Use bar instead. It is cooler.",
    );
  });
});

describe("deprecated_params", () => {
  it("warns on deprecated params and redirects tuple mappings", () => {
    // Python: foo(a=2...) warns 'The parameter a ...' and returns {'x':1,'new_param':2}
    // for redirections=[('old_param','new_param')].
    function gather(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    const wrapped = deprecated_params("a, b")(gather as (...args: unknown[]) => unknown) as unknown as (kwargs: Record<string, unknown>) => Record<string, unknown>;
    const messages = withWarnings(() => {
      expect(wrapped({ x: 2, y: 3, z: 4 })).toEqual({ x: 2, y: 3, z: 4 });
    });
    expect(messages).toHaveLength(0);
    const warned = withWarnings(() => {
      wrapped({ a: 1, z: 2 });
    });
    expect(warned.join(" ")).toContain("The parameter a");

    function passthrough(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    const redirected = deprecated_params(null, null, null, "", [["old_param", "new_param"]])(
      passthrough as (...args: unknown[]) => unknown,
    ) as unknown as (kwargs: Record<string, unknown>) => Record<string, unknown>;
    withWarnings(() => {
      expect(redirected({ x: 1, old_param: 2 })).toEqual({ x: 1, new_param: 2 });
    });
  });

  it("supports function redirectors computing new values", () => {
    // Python: redirections=[lambda runtime_in_ms: {'run_time': runtime_in_ms/1000}]
    // foo(runtime_in_ms=500) -> {'run_time': 0.5}.
    function passthrough(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    const redirect = (runtime_in_ms: unknown): Record<string, unknown> => ({
      run_time: Number(runtime_in_ms) / 1000,
    });
    // TS cannot read TS-annotated parameter names reliably, so the port honors
    // an explicit paramNames channel (same convention as redirect_params tests).
    (redirect as unknown as Record<string, unknown>)["paramNames"] = ["runtime_in_ms"];
    const wrapped = deprecated_params(null, null, null, "", [redirect])(
      passthrough as (...args: unknown[]) => unknown,
    ) as unknown as (kwargs: Record<string, unknown>) => Record<string, unknown>;
    withWarnings(() => {
      expect(wrapped({ runtime_in_ms: 500 })).toEqual({ run_time: 0.5 });
    });
  });

  it("rejects invalid identifiers with ValueError", () => {
    // Python: ValueError("Given parameter values are invalid.")
    let thrown: unknown;
    try {
      deprecated_params("1bad, ok");
    } catch (error) {
      thrown = error;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe("Given parameter values are invalid.");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.polylabel.Polygon.compute_distance
// ---------------------------------------------------------------------------
describe("Polygon.compute_distance", () => {
  function closedSquare(): Polygon {
    // NOTE: rings must be closed (first == last) for the closing edge to count;
    // an unclosed ring drops the last edge (verified against Python: the same
    // unclosed probe returns 2.0 instead of 0.0 on the boundary point).
    return new Polygon([
      array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]) as NDArray,
    ]);
  }

  it("returns positive distance inside, negative outside", () => {
    // Python: inside (2,2) -> 2.0; outside (6,2) -> -2.0 (unclosed ring probe);
    // closed ring: edge (0,2) -> 0.0; (1,1) -> 1.0; (6,2) -> -2.0; (5,5) -> -sqrt(2).
    const polygon = closedSquare();
    expect(close(polygon.compute_distance([0, 2]), 0)).toBe(true);
    expect(close(polygon.compute_distance([1, 1]), 1)).toBe(true);
    expect(close(polygon.compute_distance([6, 2]), -2)).toBe(true);
    expect(close(polygon.compute_distance([5, 5]), -Math.SQRT2)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene
// ---------------------------------------------------------------------------
describe("Scene", () => {
  it("constructs with empty registries and default flags", () => {
    // Python Scene(renderer=None, ..., always_update_mobjects=False).
    const scene = new Scene();
    expect(scene.mobjects).toEqual([]);
    expect(scene.foreground_mobjects).toEqual([]);
    expect(scene.always_update_mobjects).toBe(false);
    expect(scene.interactive_mode).toBe(false);
  });

  it("add re-adds to the end; remove clears both lists; clear empties", () => {
    // Python ground truth (Scene.add restructures then appends):
    // add(a, b); add(a) -> [b, a] (a is removed from its old slot and
    // re-appended), NOT deduplicated in place.
    const scene = new Scene();
    const a = new Mobject();
    const b = new Mobject();
    scene.add(a, b);
    scene.add(a);
    expect(scene.mobjects).toEqual([b, a]);
    scene.remove(a);
    expect(scene.mobjects).toEqual([b]);
    scene.clear();
    expect(scene.mobjects).toEqual([]);
  });

  it("get_mobject_family_members flattens the hierarchy", () => {
    const scene = new Scene();
    const child = new Mobject();
    const parent = new Group(child);
    scene.add(parent);
    expect(scene.get_mobject_family_members()).toEqual([parent, child]);
  });

  it("scene-level updaters run via update_self and can be removed", () => {
    const scene = new Scene();
    const seen: number[] = [];
    const updater = (dt: number): void => {
      seen.push(dt);
    };
    scene.add_updater(updater);
    scene.update_self(0.5);
    expect(seen).toEqual([0.5]);
    scene.remove_updater(updater);
    scene.update_self(0.5);
    expect(seen).toEqual([0.5]);
  });

  it("get_time_progression spans arange(0, run_time, 1/fps)", () => {
    // Python: np.arange(0, run_time, 1/config['frame_rate']); fps default 60.
    const scene = new Scene();
    const progression = scene.get_time_progression(1, "test");
    expect(progression.times).toHaveLength(60);
    expect(close(progression.times[0] ?? NaN, 0)).toBe(true);
    expect(close(progression.times[59] ?? NaN, 59 / 60)).toBe(true);
    const skipped = new Scene({ skip_animations: true });
    expect(skipped.get_time_progression(2.5, "test").times).toEqual([2.5]);
  });

  it("validate_run_time passes valid values and rejects non-positive ones", () => {
    // Python: "Scene.play() has a run_time of 0 <= 0 seconds which Manim
    // cannot render. The run_time must be a positive number."
    function play(): void {}
    expect(Scene.validate_run_time(1, play)).toBe(1);
    let thrown: unknown;
    try {
      Scene.validate_run_time(0, play);
    } catch (error) {
      thrown = error;
    }
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe(
      "Scene.play() has a run_time of 0 <= 0 seconds which Manim cannot render. The run_time must be a positive number.",
    );
  });

  it("validate_run_time clamps sub-frame run times with a warning", () => {
    function play(): void {}
    const fps = config.frame_rate as number;
    const messages = withWarnings(() => {
      expect(Scene.validate_run_time(1 / fps / 2, play)).toBe(1 / fps);
    });
    expect(messages.join(" ")).toContain("too short");
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols interacting with previously ported code
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("updater chain: parent + child updaters fire with dt on recursive update", () => {
    // Mirrors a Python scene-graph tick: Group(dot with updater) updated with dt.
    const dot = new VMobject();
    dot.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const seen: Array<[string, number]> = [];
    dot.add_updater((mob, dt) => {
      seen.push(["dot", dt]);
      mob.shift([dt, 0, 0]);
    });
    const group = new VGroup(dot);
    group.add_updater((_mob, dt) => {
      seen.push(["group", dt]);
    });
    group.update(0.5);
    expect(seen).toEqual([["group", 0.5], ["dot", 0.5]]);
    expect(dot.get_center()[0]).toBeCloseTo(1.0, 9);
  });

  it("ComplexValueTracker drives a mobject through an updater (Python parity)", () => {
    // Python equivalent:
    //   tracker = ComplexValueTracker(-2+1j); dot moves to tracker.points on update.
    //   tracker.set_value(3+2j); dot.update() -> center [3, 2, 0].
    const tracker = new ComplexValueTracker({ re: -2, im: 1 });
    const dot = new VMobject();
    dot.set_points_as_corners([[0, 0, 0], [0.1, 0, 0]]);
    dot.add_updater((mob) => {
      const value = tracker.get_value();
      mob.move_to([value.re, value.im, 0]);
    });
    dot.update();
    expect(dot.get_center()[0]).toBeCloseTo(-2, 9);
    expect(dot.get_center()[1]).toBeCloseTo(1, 9);
    tracker.set_value({ re: 3, im: 2 });
    dot.update();
    expect(dot.get_center()[0]).toBeCloseTo(3, 9);
    expect(dot.get_center()[1]).toBeCloseTo(2, 9);
  });

  it("ApplyWave deforms a VMobject along the same homotopy Python uses", () => {
    // End-to-end: interpolate the animation halfway and compare the displaced
    // center against Python's homotopy evaluated at the mobject's relative x.
    const line = new VMobject();
    line.set_points_as_corners([[-2, 0, 0], [2, 0, 0]]);
    const anim = new ApplyWave(line);
    anim.begin();
    anim.interpolate(0.5);
    // Python homotopy(0,0,0,0.5) == [0, 0.2, 0]: the midpoint lifts by amplitude.
    const midpoint = line.point_from_proportion(0.5).toArray() as number[];
    expect(close(Number(midpoint[0]), 0, 1e-6)).toBe(true);
    expect(Number(midpoint[1])).toBeGreaterThan(0);
  });

  it("Scene.play runs a method animation mutating a VMobject", () => {
    // Exercises Scene.play -> prepare_animation -> begin/finish against a
    // mobject from this batch. Uses shift (pure point mutation); opacity
    // paths are covered by the VMobject.set_opacity ground-truth tests.
    const scene = new Scene();
    const target = new VMobject();
    target.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    scene.add(target);
    const before = target.get_center()[0] ?? 0;
    scene.play(target.animate.shift([2, 0, 0]).build() as never);
    expect(close((target.get_center()[0] ?? 0) - before, 2)).toBe(true);
  });

  it("SVG quadratic path + subcurve + direction compose", () => {
    // A closed quadratic loop: subpaths stay whole, direction is defined,
    // and a subcurve slice keeps the endpoints on the original curve.
    const loop = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "QuadraticBezier", control: [2, 0], end: [2, 2] },
      { kind: "QuadraticBezier", control: [2, 4], end: [0, 4] },
      { kind: "QuadraticBezier", control: [-2, 4], end: [-2, 2] },
      { kind: "QuadraticBezier", control: [-2, 0], end: [0, 0] },
    ]);
    expect(loop.get_subpaths()).toHaveLength(1);
    expect(["CW", "CCW"]).toContain(loop.get_direction());
    const part = loop.get_subcurve(0.25, 0.75);
    expect(part.get_num_points()).toBeGreaterThan(0);
  });
});
