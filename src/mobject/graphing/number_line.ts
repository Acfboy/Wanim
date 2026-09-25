import { VGroup } from "../types/vectorized_mobject.js";

// NOTE: mirrors CPython repr/str for floats: scientific notation when the
// exponent is < -4 or >= 16 (e.g. str(1e-5) == '1e-05', which has no dot).
// Integer-valued JS numbers map to the Python int case (str(1) == '1');
// Python float inputs like 2.0 (str '2.0' -> 1 decimal) are indistinguishable
// from ints in JS and keep the int behaviour.
function pythonFloatStr(step: number): string {
  if (Number.isInteger(step)) return String(step);
  const magnitude = Math.abs(step);
  if (magnitude < 1e-4 || magnitude >= 1e16) {
    const [mantissa, exponent = "+0"] = step.toExponential().split("e");
    const sign = exponent.startsWith("-") ? "-" : "+";
    const digits = exponent.replace(/^[+-]/, "").padStart(2, "0");
    return `${mantissa}e${sign}${digits}`;
  }
  return String(step);
}
// NOTE: minimal port. NumberLine's label/text rendering (Tex, DecimalNumber)
// is out of scope for the interpreter target; only the pure-computation
// pieces translated so far are included here.
export class NumberLine extends VGroup {
  static _decimal_places_from_step(step: number): number {
    const step_str = pythonFloatStr(step);
    if (!step_str.includes(".")) return 0;
    return (step_str.split(".").pop() ?? "").length;
  }
}
