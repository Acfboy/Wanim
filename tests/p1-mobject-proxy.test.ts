// P1-G: Mobject proxy robustness.
//
// The proxy used to throw AttributeError for ANY unknown property, which made
// pretty-format / JSON.stringify / introspection enumerate unboundedly and
// OOM. Genuine misses must still raise AttributeError, but protocol probes
// (then/toJSON/Symbol.toPrimitive/constructor/Symbol.toStringTag/inspect
// symbols) resolve safely.
import { describe, expect, it } from "vitest";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";

describe("Mobject proxy robustness", () => {
  it("JSON.stringify is safe and returns a small serializable summary", () => {
    const square = new Square();
    const json = JSON.stringify(square);
    expect(typeof json).toBe("string");
    const parsed = JSON.parse(json) as Record<string, unknown>;
    expect(parsed["type"]).toBe("Square");
    expect(typeof parsed["num_points"]).toBe("number");
    expect(typeof parsed["submobjects"]).toBe("number");
  });

  it("exposes a toJSON summary function", () => {
    const vm = new VMobject();
    const summary = (vm as unknown as { toJSON: () => Record<string, unknown> }).toJSON();
    expect(summary["type"]).toBe("VMobject");
    expect(summary["num_points"]).toBe(0);
  });

  it("is not a thenable and has no Symbol.toPrimitive", () => {
    const vm = new VMobject();
    expect((vm as unknown as { then?: unknown }).then).toBeUndefined();
    expect((vm as unknown as Record<symbol, unknown>)[Symbol.toPrimitive]).toBeUndefined();
    expect((vm as unknown as Record<symbol, unknown>)[Symbol.toStringTag]).toBeUndefined();
  });

  it("still exposes the real constructor and raises AttributeError on genuine misses", () => {
    const vm = new VMobject();
    expect(vm.constructor.name).toBe("VMobject");
    expect(() => (vm as unknown as Record<string, unknown>)["definitely_not_a_real_attribute"]).toThrow(/AttributeError|no attribute/);
  });

  it("a failing expect(mob).toBe(...) does not OOM", () => {
    const vm = new VMobject();
    let caught: unknown;
    try {
      expect(vm).toBe("not-a-mobject");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
  });
});
