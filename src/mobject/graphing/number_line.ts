import { VGroup } from "../types/vectorized_mobject.js";

// NOTE: minimal port. NumberLine's label/text rendering (Tex, DecimalNumber)
// is out of scope for the interpreter target; only the pure-computation
// pieces translated so far are included here.
export class NumberLine extends VGroup {
  static _decimal_places_from_step(step: number): number {
    const step_str = String(step);
    if (!step_str.includes(".")) return 0;
    return (step_str.split(".").pop() ?? "").length;
  }
}
