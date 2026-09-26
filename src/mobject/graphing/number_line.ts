import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import { DEFAULT_ARROW_TIP_LENGTH, DOWN, LEFT, MED_SMALL_BUFF, OUT, UP } from "../../constants.js";
import { isNDArray, norm, pointRows, scalePoint, subtractPoints } from "../../utils/arrays.js";
import { interpolate } from "../../utils/bezier.js";
import { merge_dicts_recursively } from "../../utils/config_ops.js";
import { angle_of_vector, normalize } from "../../utils/space_ops.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { Mobject } from "../mobject.js";
import { Line } from "../geometry/line.js";
import type { ArrowTip } from "../geometry/tips.js";
import { DecimalNumber, type DecimalNumberMobFactory } from "../text/numbers.js";
import { MathTex } from "../text/tex_mobject.js";
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

// NOTE: NumberLine's label/text rendering (DecimalNumber/MathTex) is wired
// through the ported MathJax-backed text pipeline; the pure-computation
// pieces (scaling, ticks, n2p/p2n) are included here as well.

/** Normalize a `label_constructor` (a class like MathTex, or an already-built
 * factory) into DecimalNumber's `mob_class` factory. */
function toMobFactory(ctor: unknown): DecimalNumberMobFactory | undefined {
  if (typeof ctor !== "function") return undefined;
  const fn = ctor as (...args: unknown[]) => VMobject;
  return (text: string, kwargs: Record<string, unknown> = {}) => {
    try {
      return new (fn as unknown as new (t: string, k?: Record<string, unknown>) => VMobject)(
        text,
        kwargs,
      );
    } catch {
      return fn(text, kwargs);
    }
  };
}
function isSinglePoint(point: unknown): boolean {
  if (isNDArray(point)) {
    return (point.shape ?? []).length <= 1;
  }
  if (Array.isArray(point)) {
    return point.length === 0 || typeof point[0] === "number";
  }
  return true;
}

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
  // NOTE: mirrors Python NumberLine.__init__ (tip_shape defaults to None for
  // the default arrow tip). The ArrowTip pipeline lives on TipableVMobject;
  // this VGroup-based port stores the option opaquely (see include_tip).
  tip_shape: (new (...args: never[]) => ArrowTip) | null = null;
  font_size = 36;
  include_numbers = false;
  scaling: _ScaleBase = new LinearBase();
  // NOTE: mirrors Python NumberLine.__init__ (label_direction defaults to
  // DOWN, label_constructor to MathTex). MathTex renders through MathJax; the
  // constructor is stored opaquely and consulted when label mobjects are
  // constructed.
  label_direction: readonly number[] | NDArray = [...DOWN];
  label_constructor: unknown = MathTex;
  line_to_number_buff = MED_SMALL_BUFF;
  decimal_number_config: Record<string, unknown> = {};
  numbers_to_exclude: number[] = [];
  numbers_to_include: number[] | null = null;
  // NOTE: populated by add_numbers()/add_labels() (mirrors Python, where the
  // attributes only exist after those calls).
  numbers: VGroup | null = null;
  labels: VGroup | null = null;
  // NOTE: populated by add_ticks() (a later batch); the field exists here so
  // get_tick_marks() has its faithful target to return.
  ticks: VGroup = new VGroup();

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
    stroke_width?: number;
    include_tip?: boolean;
    tip_width?: number;
    tip_height?: number;
    tip_shape?: (new (...args: never[]) => ArrowTip) | null;
    include_numbers?: boolean;
    font_size?: number;
    scaling?: _ScaleBase;
    label_direction?: readonly number[] | NDArray;
    label_constructor?: unknown;
    line_to_number_buff?: number;
    decimal_number_config?: Record<string, unknown> | null;
    numbers_to_exclude?: Iterable<number> | null;
    numbers_to_include?: Iterable<number> | null;
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
    const stroke_width = options.stroke_width ?? 2.0;
    if (options.include_tip !== undefined) this.include_tip = options.include_tip;
    if (options.tip_width !== undefined) this.tip_width = options.tip_width;
    if (options.tip_height !== undefined) this.tip_height = options.tip_height;
    if (options.tip_shape !== undefined) this.tip_shape = options.tip_shape;
    if (options.include_numbers !== undefined) this.include_numbers = options.include_numbers;
    if (options.font_size !== undefined) this.font_size = options.font_size;
    if (options.label_direction !== undefined) this.label_direction = options.label_direction;
    if (options.label_constructor !== undefined) this.label_constructor = options.label_constructor;
    if (options.line_to_number_buff !== undefined) this.line_to_number_buff = options.line_to_number_buff;
    if (options.decimal_number_config !== undefined && options.decimal_number_config !== null) {
      this.decimal_number_config = { ...options.decimal_number_config };
    } else {
      // NOTE: mirrors Python `decimal_number_config = {"num_decimal_places":
      // self._decimal_places_from_step(x_range[2])}`.
      this.decimal_number_config = {
        num_decimal_places: NumberLine._decimal_places_from_step(x_range[2] ?? 1),
      };
    }
    if (options.numbers_to_exclude !== undefined && options.numbers_to_exclude !== null) {
      this.numbers_to_exclude = [...options.numbers_to_exclude];
    }
    if (options.numbers_to_include !== undefined && options.numbers_to_include !== null) {
      this.numbers_to_include = [...options.numbers_to_include];
    }
    // Baseline line geometry (mirrors Python NumberLine.__init__: a Line from
    // x_range[0]*RIGHT to x_range[1]*RIGHT, then set_length(length) (which
    // also refreshes unit_size) or scale(unit_size), then center()).
    // Ticks/tips/numbers need text rendering and stay out of scope.
    this.set_points_as_corners([
      [x_range[0], 0, 0],
      [x_range[1], 0, 0],
    ]);
    // NOTE: Python tests `if self.length:` (falsy for None *and* 0).
    if (this.length) {
      const [lineStart, lineEnd] = this.get_start_and_end();
      const baseLength = norm(subtractPoints(lineEnd, lineStart, this.dim));
      this.scale(this.length / baseLength);
      this.unit_size = this.get_unit_size();
    } else {
      this.scale(this.unit_size);
    }
    // NOTE: Python passes stroke_width into Line.__init__; this VGroup-based
    // port applies it to the baseline stroke instead (same observable style,
    // which get_tick().match_style inherits for the ticks below).
    this.set_stroke(undefined, stroke_width, undefined, false);
    this.center();
    if (this.include_tip) {
      // NOTE: Python calls self.add_tip(tip_shape=..., tip_length=tip_height,
      // tip_width=tip_width) (NumberLine is a TipableVMobject). This port
      // extends VGroup, so the tip is built through an equivalent Line (which
      // is a TipableVMobject) and the resulting tip geometry is attached to
      // self at the same end. The baseline endpoints are left as-is, matching
      // the observable geometry (get_start/get_end/unit_size are unchanged).
      const tipLine = new Line(this.get_start(), this.get_end());
      tipLine.add_tip(null, this.tip_shape as never, this.tip_height, this.tip_width);
      const tip = tipLine.tip;
      if (tip !== undefined) {
        this.add(tip as unknown as VMobject);
        (this as unknown as { tip: ArrowTip }).tip = tip;
      }
    }
    if (this.include_ticks) {
      this.add_ticks();
    }
    if (this.rotation) this.rotate(this.rotation);
    if (this.include_numbers || this.numbers_to_include !== null) {
      if (this.scaling.custom_labels) {
        const tick_range = (this.get_tick_range().toArray() as unknown[]).map(Number);
        const num_decimal_places = Number(
          (this.decimal_number_config as Record<string, unknown>)["num_decimal_places"] ?? 0,
        );
        const custom_labels = this.scaling.get_custom_labels(tick_range, num_decimal_places);
        const entries: Record<string, string | number | VMobject> = {};
        tick_range.forEach((value, index) => {
          entries[String(value)] = custom_labels[index] as string | number | VMobject;
        });
        this.add_labels(entries);
      } else {
        this.add_numbers(
          this.numbers_to_include ?? null,
          this.numbers_to_exclude,
          this.font_size,
          undefined,
          {},
        );
      }
    }
  }

  static _decimal_places_from_step(step: number): number {
    const step_str = pythonFloatStr(step);
    if (!step_str.includes(".")) return 0;
    return (step_str.split(".").pop() ?? "").length;
  }

  rotate_about_zero(
    angle: number,
    axis: readonly number[] | NDArray = [...OUT],
    about_edge: readonly number[] | NDArray | null = null,
  ): this {
    return this.rotate_about_number(0, angle, axis, about_edge);
  }

  rotate_about_number(
    number: number,
    angle: number,
    axis: readonly number[] | NDArray = [...OUT],
    about_edge: readonly number[] | NDArray | null = null,
  ): this {
    return this.rotate(angle, axis, this.n2p(number), about_edge ?? undefined);
  }

  add_ticks(): this {
    const ticks = new VGroup();
    const elongated_tick_size = this.tick_size * this.longer_tick_multiple;
    const elongated_tick_offsets = this.numbers_with_elongated_ticks.map(
      (value) => value - this.x_min,
    );
    for (const x of (this.get_tick_range().toArray() as unknown[]).map(Number)) {
      let size = this.tick_size;
      // NOTE: mirrors np.any(np.isclose(x - self.x_min,
      // elongated_tick_offsets)) with numpy's default tolerances.
      const shifted = x - this.x_min;
      const matches = elongated_tick_offsets.some((offset) =>
        Math.abs(shifted - offset) <= 1e-08 + 1e-05 * Math.abs(offset),
      );
      if (matches) {
        size = elongated_tick_size;
      }
      ticks.add(this.get_tick(x, size));
    }
    this.add(ticks);
    this.ticks = ticks;
    return this;
  }

  get_tick(x: number, size: number | null = null): Line {
    const resolved = size ?? this.tick_size;
    const result = new Line(
      scalePoint(DOWN, resolved, this.dim),
      scalePoint(UP, resolved, this.dim),
    );
    result.rotate(this.get_angle());
    result.move_to(this.number_to_point(x));
    result.match_style(this);
    return result;
  }

  get_angle(): number {
    const [start, end] = this.get_start_and_end();
    return angle_of_vector(subtractPoints(end, start, this.dim)) as number;
  }

  get_unit_vector(): number[] {
    const [start, end] = this.get_start_and_end();
    return normalize(subtractPoints(end, start, this.dim)).map(
      (value) => value * this.unit_size,
    );
  }

  get_projection(point: readonly number[] | NDArray): number[] {
    // NOTE: mirrors Line.get_projection (Python NumberLine is a Line, so it
    // inherits this); this port extends VGroup, so the same projection onto
    // the baseline is computed here.
    const [start, end] = this.get_start_and_end();
    const unit_vect = normalize(subtractPoints(end, start, this.dim));
    const pointRowsValue = pointRows(point, this.dim)[0] ?? [0, 0, 0];
    const dot = subtractPoints(pointRowsValue, start, this.dim).reduce(
      (sum, value, index) => sum + value * (unit_vect[index] ?? 0),
      0,
    );
    return start.map((value, index) => value + dot * (unit_vect[index] ?? 0));
  }

  get_tick_marks(): VGroup {
    return this.ticks;
  }

  get_tick_range(): NDArray {    const [x_min, x_max, x_step] = this.x_range;
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

  point_to_number(point: number[] | number[][] | NDArray): number | NDArray {
    const [start, end] = this.get_start_and_end();
    const span = subtractPoints(end, start, this.dim);
    const unit_vect = normalize(span);
    const denom = span.reduce((sum, value, index) => sum + value * (unit_vect[index] ?? 0), 0);
    const rows = pointRows(point, this.dim);
    const proportions = rows.map((row) => {
      const diff = subtractPoints(row, start, this.dim);
      return diff.reduce((sum, value, index) => sum + value * (unit_vect[index] ?? 0), 0) / denom;
    });
    if (isSinglePoint(point)) {
      return interpolate(this.x_min, this.x_max, proportions[0] ?? 0) as number;
    }
    return interpolate(this.x_min, this.x_max, array(proportions) as NDArray) as NDArray;
  }

  pointToNumber(point: number[] | number[][] | NDArray): number | NDArray {
    return this.point_to_number(point);
  }

  p2n(point: number[] | number[][] | NDArray): number | NDArray {
    return this.point_to_number(point);
  }

  number_to_point(number: number | number[] | NDArray): NDArray {
    // NOTE: mirrors Python `number = np.asarray(number);
    // scalar = number.ndim == 0` (a 0-d array counts as scalar) followed by
    // `alphas = float(alphas) if scalar else np.vstack(alphas)`.
    const scalar =
      typeof number === "number" ||
      (isNDArray(number) && (number.shape ?? []).length === 0);
    const scaled = this.scaling.inverse_function(number as number);
    const x_min = this.x_range[0] ?? 0;
    const span = (this.x_range[1] ?? 0) - (this.x_range[0] ?? 0);
    const [start, end] = this.get_start_and_end();
    if (scalar) {
      const value =
        typeof scaled === "number"
          ? scaled
          : Number(pointRows(scaled, 1)[0]?.[0] ?? 0);
      const result = interpolate(start, end, (value - x_min) / span);
      return (isNDArray(result) ? result : array(result as unknown as number[])) as NDArray;
    }
    // NOTE: one alpha per input value (np.vstack semantics); flatten any
    // nesting element-wise rather than reading a single point row.
    const flat: number[] = [];
    const collect = (value: unknown): void => {
      if (Array.isArray(value)) {
        for (const item of value) collect(item);
      } else {
        flat.push(Number(value));
      }
    };
    collect(isNDArray(scaled) ? scaled.toArray() : scaled);
    const alphas = array(flat.map((value) => (value - x_min) / span)) as NDArray;
    const result = interpolate(start, end, alphas);
    return (isNDArray(result) ? result : array(result as unknown as number[])) as NDArray;
  }

  n2p(number: number | number[] | NDArray): NDArray {
    return this.number_to_point(number);
  }

  __matmul__(other: number | number[] | NDArray): NDArray {
    return this.n2p(other);
  }

  __rmatmul__(other: number[] | NDArray | Mobject): number | NDArray {
    let target: unknown = other;
    if (target instanceof Mobject) target = (target as Mobject).get_center();
    return this.p2n(target as number[] | NDArray);
  }

  get_unit_size(): number {
    // NOTE: Python calls self.get_length() (Line.get_length); this port's
    // NumberLine extends VGroup (not Line), so the length is computed
    // inline from the endpoints instead.
    const [start, end] = this.get_start_and_end();
    const length = norm(subtractPoints(end, start, this.dim));
    const val: number = length / ((this.x_range[1] ?? 0) - (this.x_range[0] ?? 0));
    return val;
  }

  get_number_mobject(
    x: number,
    direction: readonly number[] | NDArray | null = null,
    buff: number | null = null,
    font_size: number | null = null,
    label_constructor: unknown = null,
    number_config: Record<string, unknown> = {},
  ): VMobject {
    // Faithful port of Python NumberLine.get_number_mobject: builds a
    // positioned DecimalNumber whose digits are rendered with
    // `label_constructor` (which defaults to the line's label_constructor,
    // i.e. MathTex in Python; DecimalNumber's own default MathTex factory is
    // used when none is given).
    const merged = merge_dicts_recursively(this.decimal_number_config, number_config);
    const resolvedDirection = direction ?? this.label_direction;
    const resolvedBuff = buff ?? this.line_to_number_buff;
    const resolvedFontSize = font_size ?? this.font_size;
    const resolvedConstructor = label_constructor ?? this.label_constructor;

    const options: Record<string, unknown> = {
      font_size: resolvedFontSize,
      ...merged,
    };
    const factory = toMobFactory(resolvedConstructor);
    if (factory !== undefined) options.mob_class = factory;

    const numMob = new DecimalNumber(x, options) as unknown as VMobject;
    numMob.next_to(this.number_to_point(x), resolvedDirection, resolvedBuff);
    const dx = (resolvedDirection as readonly number[])[0] ?? 0;
    if (x < 0 && dx === 0) {
      numMob.shift(scalePoint(LEFT, (numMob.__getitem__(0) as Mobject).width / 2, this.dim));
    }
    return numMob;
  }

  get_number_mobjects(...numbers: number[]): VGroup {
    let resolved: number[] = numbers;
    if (resolved.length === 0) {
      // NOTE: Python calls self.default_numbers_to_display(), which is not
      // defined anywhere in manim 0.21.0 (only referenced) — calling it
      // raises AttributeError. Mirrored here.
      const fallback = (this as unknown as { default_numbers_to_display?: unknown }).default_numbers_to_display;
      if (typeof fallback !== "function") {
        const error = new Error("'NumberLine' object has no attribute 'default_numbers_to_display'");
        error.name = "AttributeError";
        throw error;
      }
      resolved = (fallback as () => number[]).call(this);
    }
    return new VGroup(...resolved.map((number) => this.get_number_mobject(number)));
  }

  get_labels(): VGroup {
    return this.get_number_mobjects();
  }

  add_numbers(
    x_values: Iterable<number> | null = null,
    excluding: Iterable<number> | null = null,
    font_size: number | null = null,
    label_constructor: unknown = null,
    number_config: Record<string, unknown> = {},
  ): this {
    let values: number[];
    if (x_values === null || x_values === undefined) {
      values = (this.get_tick_range().toArray() as unknown[]).map(Number);
    } else if (isNDArray(x_values)) {
      values = ((x_values as NDArray).toArray() as unknown[]).map(Number);
    } else {
      values = [...x_values];
    }
    const resolvedExcluding = excluding ?? this.numbers_to_exclude;
    const resolvedFontSize = font_size ?? this.font_size;
    const resolvedConstructor = label_constructor ?? this.label_constructor;
    const numbers = new VGroup();
    for (const x of values) {
      if ([...resolvedExcluding].includes(x)) continue;
      numbers.add(
        this.get_number_mobject(
          x,
          undefined,
          undefined,
          resolvedFontSize,
          resolvedConstructor,
          number_config,
        ),
      );
    }
    this.add(numbers);
    this.numbers = numbers;
    return this;
  }

  add_labels(
    dict_values: Map<number, string | number | VMobject> | Record<string, string | number | VMobject>,
    direction: readonly number[] | NDArray | null = null,
    buff: number | null = null,
    font_size: number | null = null,
    label_constructor: unknown = null,
  ): this {
    const resolvedDirection = direction ?? this.label_direction;
    const resolvedBuff = buff ?? this.line_to_number_buff;
    const resolvedFontSize = font_size ?? this.font_size;
    const resolvedConstructor = label_constructor ?? this.label_constructor;
    const labels = new VGroup();
    const entries: Array<[number, string | number | VMobject]> = dict_values instanceof Map
      ? [...dict_values.entries()]
      : Object.entries(dict_values).map(([key, value]) => [Number(key), value]);
    for (const [x, label] of entries) {
      const mob = this._create_label_tex(label, resolvedConstructor);
      if ("font_size" in (mob as unknown as object)) {
        ((mob as unknown as Record<string, unknown>)["font_size"] as number) = resolvedFontSize;
      } else {
        const error = new Error(`${mob} is not compatible with add_labels.`);
        error.name = "AttributeError";
        throw error;
      }
      mob.next_to(this.number_to_point(x), resolvedDirection, resolvedBuff);
      labels.add(mob);
    }
    this.labels = labels;
    this.add(labels);
    return this;
  }

  _create_label_tex(
    label_tex: string | number | VMobject,
    label_constructor: unknown = null,
    kwargs: Record<string, unknown> = {},
  ): VMobject {
    // Mirrors Python NumberLine._create_label_tex: VMobject inputs are
    // returned as-is; everything else is stringified and passed to
    // label_constructor (MathTex by default, rendered through MathJax).
    if (label_tex instanceof VMobject) return label_tex;
    const resolvedConstructor = label_constructor ?? this.label_constructor ?? MathTex;
    const factory = toMobFactory(resolvedConstructor);
    if (factory === undefined) {
      throw new Error(
        `NumberLine._create_label_tex: label_constructor ${String(resolvedConstructor)} is not callable`,
      );
    }
    return factory(String(label_tex), kwargs);
  }
}

export class UnitInterval extends NumberLine {
  constructor(
    unit_size = 10,
    numbers_with_elongated_ticks: Iterable<number> | null = null,
    decimal_number_config: Record<string, unknown> | null = null,
    options: ConstructorParameters<typeof NumberLine>[0] = {},
  ) {
    const resolved_ticks = numbers_with_elongated_ticks ?? [0, 1];
    const resolved_decimal = decimal_number_config ?? { num_decimal_places: 1 };
    super({
      ...options,
      x_range: [0, 1, 0.1],
      unit_size,
      numbers_with_elongated_ticks: resolved_ticks,
      decimal_number_config: resolved_decimal,
    });
  }
}
