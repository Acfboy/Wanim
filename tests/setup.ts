// Differential text tests compare wanim's layout against Python manim rendered
// on this host with DejaVu Sans (`Text(..., font="sans")`).  Pin the host's
// default font through the same override the demo scripts use, so the
// environment's font chain never changes those reference numbers.
import { expect } from "vitest";
import { resolveDefaultFontPath } from "../src/mobject/text/pango/font-loader.js";
import { preload_mathjax } from "../src/mobject/text/mathjax.js";

// Public wanim APIs that mirror manim return numpy-ts `NDArray`s (the Python
// object is a `np.ndarray`).  Compare them to plain arrays by value so the
// existing value-level assertions keep expressing the same ground truth.
function isNDArrayLike(value: unknown): boolean {
  return (
    value !== null &&
    typeof value === "object" &&
    "shape" in (value as object) &&
    typeof (value as { toArray?: unknown }).toArray === "function"
  );
}

function ndValuesEqual(a: unknown, b: unknown): boolean {
  if (isNDArrayLike(a)) a = (a as { toArray(): unknown }).toArray();
  if (isNDArrayLike(b)) b = (b as { toArray(): unknown }).toArray();
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      if (!ndValuesEqual(a[i], b[i])) return false;
    }
    return true;
  }
  return Object.is(a, b);
}

expect.addEqualityTesters([
  (a: unknown, b: unknown) => {
    if (!isNDArrayLike(a) && !isNDArrayLike(b)) return undefined;
    return ndValuesEqual(a, b);
  },
]);

process.env.WANIM_FONT ??= resolveDefaultFontPath();

// MathJax is an on-demand chunk in the browser build; in Node the suite builds
// MathTex/DecimalNumber/... synchronously, so load it once before any test
// runs.  (Browser pages get the same behavior through py2js' auto-preload.)
await preload_mathjax();
