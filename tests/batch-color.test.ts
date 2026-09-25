import { describe, expect, it } from "vitest";
import { array } from "numpy-ts";
import { ManimColor, invert_color } from "../src/utils/color/core.js";

// Ground truths captured from the original Python manim
// (original manim):
//   ManimColor._internal_from_rgb([0.1,0.2,0.3])              -> [0.1, 0.2, 0.3, 1.0]
//   ..._from_rgb([0.1,0.2,0.3], alpha=0.5)                    -> [0.1, 0.2, 0.3, 0.5]
//   ..._from_rgba([0.1,0.2,0.3,0.4])                          -> [0.1, 0.2, 0.3, 0.4]
//   ..._from_int_rgb([255,128,0])                             -> [1.0, 0.50196..., 0.0, 1.0]
//   ..._from_int_rgb([255,128,0], alpha=0.25)                 -> [1.0, 0.50196..., 0.0, 0.25]
//   ..._from_hex_string("FF8000", 1.0)                        -> [1.0, 0.50196..., 0.0, 1.0]
//   ..._from_hex_string("FF800080", 1.0)                      -> [1.0, 0.50196..., 0.0, 0.50196...]
//   ..._from_hex_string("F80", 1.0)                           -> [1.0, 0.5333..., 0.0, 1.0]
//   ..._from_string("red", 0.5)                               -> [0.98823..., 0.38431..., 0.33333..., 0.5]
//   ..._from_integer(0xFF8000, 0.7)                           -> [1.0, 0.50196..., 0.0, 0.7]
//   ..._from_int_rgba([255,128,0,64])                         -> [1.0, 0.50196..., 0.0, 0.25098...]
//   ManimColor("#FF8000")._internal_value                     -> [1.0, 0.50196..., 0.0, 1.0]
//   invert_color(ManimColor("#FF0000"))._internal_value        -> [0.0, 1.0, 1.0, 1.0]
//   _internal_from_rgb([0.1,0.2,0.3,0.9])                     -> [0.1, 0.2, 0.3, 1.0] (alpha slot overwritten)
//   _internal_from_rgb([0.1,0.2])                             -> [0.1, 0.2, 0.0, 1.0] (resize pads 0)
//   _internal_from_int_rgb([255,128,0,64])                    -> [1.0, 0.50196..., 0.0, 1.0] (alpha overwritten)
//   _internal_from_hex_string("F808", 1.0)                    -> [1.0, 0.5333..., 0.0, 0.5333...]
//   _internal_from_integer(0x000000, 1.0)                     -> [0.0, 0.0, 0.0, 1.0]

function asNumbers(value: unknown): number[] {
  if (value !== null && typeof value === "object" && "toArray" in value) {
    const raw = (value as { toArray: () => unknown }).toArray();
    return (raw as unknown[]).map(Number);
  }
  return (value as unknown[]).map(Number);
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = asNumbers(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const e = expected[i];
    const a = values[i];
    const tol = Math.max(1e-12, Math.abs(e) * relTol);
    expect(Math.abs(a - e)).toBeLessThanOrEqual(tol);
  }
}

describe("ManimColor._internal_from_rgb", () => {
  it("converts float rgb with default alpha", () => {
    expectClose(ManimColor._internal_from_rgb([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3, 1.0]);
  });
  it("honours an explicit alpha", () => {
    expectClose(ManimColor._internal_from_rgb([0.1, 0.2, 0.3], 0.5), [0.1, 0.2, 0.3, 0.5]);
  });
  it("overwrites a 4th component like np.resize + value[3] = alpha", () => {
    expectClose(ManimColor._internal_from_rgb([0.1, 0.2, 0.3, 0.9]), [0.1, 0.2, 0.3, 1.0]);
  });
  it("pads short inputs with 0 like np.resize", () => {
    expectClose(ManimColor._internal_from_rgb([0.1, 0.2]), [0.1, 0.2, 0.0, 1.0]);
  });
  it("accepts an NDArray input", () => {
    expectClose(ManimColor._internal_from_rgb(array([0.1, 0.2, 0.3])), [0.1, 0.2, 0.3, 1.0]);
  });
});

describe("ManimColor._internal_from_rgba", () => {
  it("passes float rgba through", () => {
    expectClose(ManimColor._internal_from_rgba([0.1, 0.2, 0.3, 0.4]), [0.1, 0.2, 0.3, 0.4]);
  });
});

describe("ManimColor._internal_from_int_rgb", () => {
  it("scales int rgb by 1/255 with default alpha", () => {
    expectClose(ManimColor._internal_from_int_rgb([255, 128, 0]), [1.0, 128 / 255, 0.0, 1.0]);
  });
  it("honours an explicit alpha", () => {
    expectClose(ManimColor._internal_from_int_rgb([255, 128, 0], 0.25), [1.0, 128 / 255, 0.0, 0.25]);
  });
  it("overwrites a 4th component with alpha", () => {
    expectClose(ManimColor._internal_from_int_rgb([255, 128, 0, 64]), [1.0, 128 / 255, 0.0, 1.0]);
  });
});

describe("ManimColor._internal_from_hex_string", () => {
  it("parses 6-digit hex", () => {
    expectClose(ManimColor._internal_from_hex_string("FF8000", 1.0), [1.0, 128 / 255, 0.0, 1.0]);
  });
  it("takes alpha from the 4th byte of 8-digit hex", () => {
    expectClose(ManimColor._internal_from_hex_string("FF800080", 1.0), [1.0, 128 / 255, 0.0, 128 / 255]);
  });
  it("expands 3-digit hex", () => {
    expectClose(ManimColor._internal_from_hex_string("F80", 1.0), [1.0, 136 / 255, 0.0, 1.0]);
  });
  it("expands 4-digit hex and takes alpha from it", () => {
    expectClose(ManimColor._internal_from_hex_string("F808", 1.0), [1.0, 136 / 255, 0.0, 136 / 255]);
  });
  it("raises ValueError for bad lengths", () => {
    let thrown: unknown;
    try {
      ManimColor._internal_from_hex_string("ZZ", 1.0);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe(
      "Hex colors must be specified with either 0x or # as prefix and contain 6 or 8 hexadecimal numbers",
    );
  });
});

describe("ManimColor._internal_from_string", () => {
  it("resolves names case-insensitively and applies alpha", () => {
    expectClose(ManimColor._internal_from_string("red", 0.5), [
      0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5,
    ]);
  });
  it("raises ValueError for unknown names", () => {
    let thrown: unknown;
    try {
      ManimColor._internal_from_string("notacolor", 1.0);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("ValueError");
    expect((thrown as Error).message).toBe("Color notacolor not found");
  });
});

describe("ManimColor._internal_from_integer", () => {
  it("splits 0xRRGGBB with alpha", () => {
    expectClose(ManimColor._internal_from_integer(0xff8000, 0.7), [1.0, 128 / 255, 0.0, 0.7]);
  });
  it("handles zero", () => {
    expectClose(ManimColor._internal_from_integer(0x000000, 1.0), [0.0, 0.0, 0.0, 1.0]);
  });
});

describe("ManimColor._internal_from_int_rgba", () => {
  it("scales int rgba by 1/255", () => {
    expectClose(ManimColor._internal_from_int_rgba([255, 128, 0, 64]), [1.0, 128 / 255, 0.0, 64 / 255]);
  });
});

describe("ManimColor._internal_value", () => {
  it("returns the [r,g,b,a] float array", () => {
    const c = new ManimColor("#FF8000");
    expectClose(c._internal_value, [1.0, 128 / 255, 0.0, 1.0]);
    expectClose(c.internal_value, [1.0, 128 / 255, 0.0, 1.0]);
  });
});

describe("invert_color", () => {
  it("linearly inverts rgb and keeps alpha by default", () => {
    expectClose(invert_color(new ManimColor("#FF0000"))._internal_value, [0.0, 1.0, 1.0, 1.0]);
  });
  it("inverts alpha too when asked", () => {
    const out = new ManimColor("#FF0000").invert(true);
    expectClose(out._internal_value, [0.0, 1.0, 1.0, 0.0]);
  });
});
