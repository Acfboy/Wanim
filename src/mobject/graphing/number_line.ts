import { type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import { DEFAULT_ARROW_TIP_LENGTH } from "../../constants.js";
import { VGroup } from "../types/vectorized_mobject.js";
import { LinearBase, _ScaleBase } from "./scale.js";

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

function arange(start: number, stop: number, step: number): number[] {
  // NOTE: mirrors np.arange (half-open interval, values computed as
  // start + i * step rather than accumulated).
  if (step === 0) throw new Error("step cannot be zero");
  const result: number[] = [];
  if (step > 0) {
    for (let i = 0; start + i * step < stop; i += 1) result.push(start + i * step);
  } else {
    for (let i = 0; start + i * step > stop; i += 1) result.push(start + i * step);
  }
  return result;
}

// NOTE: minimal port. NumberLine's label/text rendering (Tex, DecimalNumber)
// is out of scope for the interpreter target; only the pure-computation
// pieces translated so far are included here.
export class NumberLine extends VGroup {
  x_range: [number, number, number];
  x_min: number;
  x_max: number;
  x_step: number;
  length: number | null = null;
  unit_size = 1;
  include_ticks = true;
  tick_size = 0.1;
  numbers_with_elongated_ticks: number[] = [];
  longer_tick_multiple = 2;
  exclude_origin_tick = false;
  rotation = 0;
  include_tip = false;
  tip_width = DEFAULT_ARROW_TIP_LENGTH;
  tip_height = DEFAULT_ARROW_TIP_LENGTH;
  font_size = 36;
  include_numbers = false;
  scaling: _ScaleBase = new LinearBase();

  constructor(options: {
    x_range?: readonly [number, number] | readonly [number, number, number] | null;
    length?: number | null;
    unit_size?: number;
    include_ticks?: boolean;
    tick_size?: number;
    numbers_with_elongated_ticks?: Iterable<number> | null;
    longer_tick_multiple?: number;
    exclude_origin_tick?: boolean;
    rotation?: number;
    include_tip?: boolean;
    tip_width?: number;
    tip_height?: number;
    include_numbers?: boolean;
    font_size?: number;
    scaling?: _ScaleBase;
  } = {}) {
    super();
    const frame_x_radius = Number(config.get("frame_x_radius") ?? 7.111111111111111);
    let x_range: [number, number, number];
    if (options.x_range === null || options.x_range === undefined) {
      x_range = [Math.round(-frame_x_radius), Math.round(frame_x_radius), 1];
    } else if (options.x_range.length === 2) {
      x_range = [options.x_range[0] ?? 0, options.x_range[1] ?? 0, 1];
    } else {
      x_range = [options.x_range[0] ?? 0, options.x_range[1] ?? 0, options.x_range[2] ?? 1];
    }
    this.scaling = options.scaling ?? new LinearBase();
    // NOTE: Python stores x_range raw and x_min/x_max/x_step scaled:
    // self.x_min, self.x_max, self.x_step = scaling.function(self.x_range).
    this.x_range = x_range;
    const scaled = this.scaling.function([...x_range]) as NDArray;
    const scaledRows = (scaled.toArray() as unknown[]).map(Number);
    this.x_min = scaledRows[0] ?? x_range[0];
    this.x_max = scaledRows[1] ?? x_range[1];
    this.x_step = scaledRows[2] ?? x_range[2];
    if (options.length !== undefined) this.length = options.length;
    if (options.unit_size !== undefined) this.unit_size = options.unit_size;
    if (options.include_ticks !== undefined) this.include_ticks = options.include_ticks;
    if (options.tick_size !== undefined) this.tick_size = options.tick_size;
    if (options.numbers_with_elongated_ticks !== undefined && options.numbers_with_elongated_ticks !== null) {
      this.numbers_with_elongated_ticks = [...options.numbers_with_elongated_ticks];
    }
    if (options.longer_tick_multiple !== undefined) this.longer_tick_multiple = options.longer_tick_multiple;
    if (options.exclude_origin_tick !== undefined) this.exclude_origin_tick = options.exclude_origin_tick;
    if (options.rotation !== undefined) this.rotation = options.rotation;
    if (options.include_tip !== undefined) this.include_tip = options.include_tip;
    if (options.tip_width !== undefined) this.tip_width = options.tip_width;
    if (options.tip_height !== undefined) this.tip_height = options.tip_height;
    if (options.include_numbers !== undefined) this.include_numbers = options.include_numbers;
    if (options.font_size !== undefined) this.font_size = options.font_size;
  }

  static _decimal_places_from_step(step: number): number {
    const step_str = pythonFloatStr(step);
    if (!step_str.includes(".")) return 0;
    return (step_str.split(".").pop() ?? "").length;
  }

  get_tick_range(): NDArray {
    const [x_min, x_max, x_step] = this.x_range;
    let max = x_max ?? 0;
    if (!this.include_tip) max += 1e-6;

    // NOTE: Python chains the comparisons: x_min < x_max < 0 means both
    // x_min < x_max and x_max < 0 (and symmetrically for the second term).
    let tick_range: number[];
    if (((x_min ?? 0) < (x_max ?? 0) && (x_max ?? 0) < 0) || ((x_max ?? 0) > (x_min ?? 0) && (x_min ?? 0) > 0)) {
      tick_range = arange(x_min ?? 0, max, x_step ?? 1);
    } else {
      let start_point = 0;
      if (this.exclude_origin_tick) start_point += x_step ?? 0;
      const x_min_segment = arange(start_point, Math.abs(x_min ?? 0) + 1e-6, x_step ?? 1).map((value) => -value);
      const x_max_segment = arange(start_point, max, x_step ?? 1);
      // NOTE: np.unique returns sorted unique values.
      tick_range = [...new Set([...x_min_segment, ...x_max_segment])].sort((a, b) => a - b);
    }
    return this.scaling.function(tick_range) as NDArray;
  }
}
