import { describe, expect, it } from "vitest";
import { createRuntimeCore } from "../py2js/src/runtime_core.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Circle } from "../src/mobject/geometry/arc.js";

/**
 * Regression tests for the deliberate removal of the per-Mobject Proxy.
 *
 * Internal TS code now reads fields directly; Python-facing reads go through
 * the bridge's `getDot`, which supplies the dynamic get_/set_ helpers,
 * AttributeError on genuine misses, and the array-like compatibility used by
 * splats.  Protocol probes must remain undefined.
 */
const core = createRuntimeCore({
  wanim: { Mobject },
  np: {},
  NDArray: class FakeNDArray {},
});

describe("Mobject without a Proxy", () => {
  it("does not intercept arbitrary JS property reads", () => {
    const mob = new Mobject();
    expect((mob as unknown as Record<string, unknown>)["missing"]).toBeUndefined();
    expect((mob as unknown as Record<string, unknown>)["get_answer"]).toBeUndefined();
  });

  it("still exposes the real toJSON summary", () => {
    const parsed = JSON.parse(JSON.stringify(new Circle())) as Record<string, unknown>;
    expect(parsed["type"]).toBe("Circle");
    expect(typeof parsed["num_points"]).toBe("number");
  });

  it("recreates dynamic get_/set_/AttributeError semantics at the Python boundary", () => {
    const mob = new Mobject();
    (mob as unknown as Record<string, unknown>)["answer"] = 42;
    const getter = core.getDot(mob, "get_answer") as () => unknown;
    expect(getter()).toBe(42);

    const setter = core.getDot(mob, "set_answer") as (value: unknown) => Mobject;
    expect(setter(7)).toBe(mob);
    expect((mob as unknown as Record<string, unknown>)["answer"]).toBe(7);

    expect(() => core.getDot(mob, "definitely_missing")).toThrow(/no attribute/);
  });

  it("keeps protocol probes and array-like access safe", () => {
    const mob = new Mobject();
    expect(core.getDot(mob, "then")).toBeUndefined();
    expect(core.getDot(mob, "__argnames__")).toBeUndefined();

    const circle = new Circle();
    expect(core.getDot(circle, "length")).toBe(1);
    expect(core.getDot(circle, "0")).toBe(circle);
  });
});
