import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import { DEFAULT_ARROW_TIP_LENGTH, DOWN, LEFT, MED_SMALL_BUFF, OUT } from "../../constants.js";
import { isNDArray, norm, numeric, pointRows, scalePoint, subtractPoints } from "../../utils/arrays.js";
import { interpolate } from "../../utils/bezier.js";
import { merge_dicts_recursively } from "../../utils/config_ops.js";
import { WHITE, type ParsableManimColor } from "../../utils/color/core.js";
import { angle_of_vector, normalize } from "../../utils/space_ops.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { Mobject, isPlainOptionsObject } from "../mobject.js";
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
 * factory) into DecimalNumber's `mob_class` factory.
 *
 * The wrapper is memoized on the underlying constructor: DecimalNumber keys
 * its glyph cache on the factory's identity, so handing out a fresh closure
 * per call would miss the cache and re-render every digit. */
const _mob_factory_cache = new WeakMap<object, DecimalNumberMobFactory>();

function toMobFactory(ctor: unknown): DecimalNumberMobFactory | undefined {
  if (typeof ctor !== "function") return undefined;
  const fn = ctor as (...args: unknown[]) => VMobject;
  const cached = _mob_factory_cache.get(fn);
  if (cached !== undefined) return cached;
  const factory: DecimalNumberMobFactory = (text, kwargs = {}) => {
    try {
      return new (fn as unknown as new (t: string, k?: Record<string, unknown>) => VMobject)(
        text,
        kwargs,
      );
    } catch {
      return fn(text, kwargs);
    }
  };
  _mob_factory_cache.set(fn, factory);
  return factory;
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

/** Options bag for NumberLine (mirrors the Python keyword parameters). */
export interface NumberLineOptions {
  x_range?: readonly [number, number] | readonly [number, number, number] | null;
  length?: number | null;
  unit_size?: number;
  include_ticks?: boolean;
  tick_size?: number;
  numbers_with_elongated_ticks?: Iterable<number> | null;
  longer_tick_multiple?: number;
  exclude_origin_tick?: boolean;
  rotation?: number;
  // NOTE: NumberLine extends Line in Python, so it forwards these VMobject
  // style kwargs through Line.__init__. The TS port extends VGroup, so the
  // values are read back and applied to the baseline here (ticks inherit
  // them via match_style).
  stroke_width?: number;
  color?: ParsableManimColor;
  stroke_color?: ParsableManimColor;
  fill_color?: ParsableManimColor;
  stroke_opacity?: number;
  fill_opacity?: number;
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
  // NOTE: populated by the include_tip branch (mirrors Python's `self.tip`
  // / `self.start_tip` attributes installed by TipableVMobject.add_tip).
  tip?: ArrowTip;
  start_tip?: ArrowTip;
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

  constructor(
    optionsOrRange: NumberLineOptions | readonly number[] | null = {},
    ...extra: unknown[]
  ) {
    super();
    // Accept a trailing plain options bag as well as a Python-style positional
    // x_range/length:
    //   new NumberLine({ x_range: [-3, 3, 1] })
    //   new NumberLine([-3, 3, 1], { length: 6 })
    const options: NumberLineOptions = {};
    const positionals: unknown[] = [];
    const consume = (value: unknown): void => {
      if (value === null || value === undefined) return;
      if (isPlainOptionsObject(value)) {
        Object.assign(options, value);
        return;
      }
      positionals.push(value);
    };
    consume(optionsOrRange);
    for (const item of extra) consume(item);
    if (positionals.length > 0) options.x_range = positionals[0] as NumberLineOptions["x_range"];
    if (positionals.length > 1) options.length = positionals[1] as number | null;
    if (positionals.length > 2) options.unit_size = positionals[2] as number;
    if (positionals.length > 3) options.include_ticks = positionals[3] as boolean;
    if (positionals.length > 4) options.tick_size = positionals[4] as number;
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
    // NOTE: Python passes the style kwargs (color, stroke_color, fill_color,
    // stroke_opacity, fill_opacity, stroke_width) into Line.__init__, which
    // seeds them through VMobject.__init__. This VGroup-based port reproduces
    // that seeding on the baseline (same fill/stroke rgba layout: `color` sets
    // both rgb channels while the fill keeps alpha 0). get_tick() calls
    // match_style(this) below, so the ticks inherit the same style.
    const fillColor = (options.fill_color ?? options.color ?? WHITE) as ParsableManimColor;
    const strokeColor = (options.stroke_color ?? options.color ?? WHITE) as ParsableManimColor;
    const fillOpacity = options.fill_opacity ?? 0;
    const strokeOpacity = options.stroke_opacity ?? 1;
    this.set_fill(fillColor, fillOpacity, false);
    this.set_stroke(strokeColor, stroke_width, strokeOpacity, false);
    this.center();
    if (this.include_tip) {
      // NOTE: Python calls self.add_tip(tip_shape=..., tip_length=tip_height,
      // tip_width=tip_width) (NumberLine is a TipableVMobject). This port
      // extends VGroup, so the tip is built through an equivalent Line (which
      // is a TipableVMobject) and the resulting tip geometry is attached to
      // self at the same end. `reset_endpoints_based_on_tip` shortens the
      // baseline to the tip base (before the tip is added, as in Python); the
      // logical endpoints stay at the original x_range ends via the
      // get_start/get_end overrides below (mirroring TipableVMobject).
      // NOTE: Python's TipableVMobject.get_unpositioned_tip styles the tip
      // with `self.get_color()` (both fill and stroke), so the helper Line is
      // seeded with the axis color before add_tip.
      const tipLine = new Line(this.get_start(), this.get_end(), { color: this.get_color() });
      tipLine.add_tip(null, this.tip_shape as never, this.tip_height, this.tip_width);
      const tip = tipLine.tip;
      if (tip !== undefined) {
        this.put_start_and_end_on(this.get_start(), tip.base);
        // Python: self.tip.set_stroke(self.stroke_color, self.stroke_width)
        tip.set_stroke(this.get_stroke_color(), this.get_stroke_width());
        this.add(tip as unknown as VMobject);
        this.tip = tip;
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

  // NOTE: mirrors TipableVMobject.has_tip/has_start_tip (the tip is a
  // submobject) and the get_start/get_end overrides, which keep the logical
  // endpoints at the tip apex while the baseline points stop at the tip base.
  has_tip(): boolean {
    return this.tip !== undefined && this.submobjects.includes(this.tip as unknown as VMobject);
  }

  has_start_tip(): boolean {
    return this.start_tip !== undefined && this.submobjects.includes(this.start_tip as unknown as VMobject);
  }

  override get_start(): number[] {
    if (this.has_start_tip()) {
      return (this.start_tip as ArrowTip).get_start();
    }
    return super.get_start();
  }

  override get_end(): number[] {
    if (this.has_tip()) {
      const tip = this.tip as ArrowTip;
      return tip.get_start();
    }
    return super.get_end();
  }

  // Cached flat endpoints of the baseline: `[sx, sy, sz, ex, ey, ez]`.
  // Every point mutation replaces `this.points` with a fresh NDArray, so the
  // cache is invalidated by comparing the reference (plus the tip identities,
  // since get_start/get_end may read the tip apex). This turns the repeated
  // `pointAt` + `throw_error_if_no_points` work in the graphing hot paths into
  // a single computation per transform. A single cache object keeps the
  // per-call Proxy trap count down (Mobject instances are Proxies).
  private _endpointsCache: {
    points: unknown;
    pointsVersion: number;
    tip: unknown;
    startTip: unknown;
    flat: number[];
  } | null = null;

  private _ensure_endpoints_flat(): number[] {
    const cache = this._endpointsCache;
    const points = this.points;
    const pointsVersion = this.points_version;
    const tip = this.tip;
    const startTip = this.start_tip;
    if (
      cache !== null &&
      cache.points === points &&
      cache.pointsVersion === pointsVersion &&
      cache.tip === tip &&
      cache.startTip === startTip
    ) {
      return cache.flat;
    }
    const start = this.get_start();
    const end = this.get_end();
    const dim = Math.max(start.length, end.length, this.dim);
    const flat = new Array<number>(dim * 2);
    for (let i = 0; i < dim; i += 1) {
      flat[i] = start[i] ?? 0;
      flat[dim + i] = end[i] ?? 0;
    }
    this._endpointsCache = { points, pointsVersion, tip, startTip, flat };
    return flat;
  }

  override get_start_and_end(): [number[], number[]] {
    const flat = this._ensure_endpoints_flat();
    const dim = flat.length / 2;
    return [flat.slice(0, dim), flat.slice(dim)];
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
    // Precompute the per-line constants once for the whole batch: the flat
    // tick values, the (unchanged) baseline angle, and the fully resolved
    // style that every tick inherits via `match_style`. `get_tick` rebuilds
    // these for standalone calls, but `add_ticks` must not pay for them 15+
    // times.
    const tickValues = (this.get_tick_range().toArray() as unknown[]).map(Number);
    const angle = this._baseline_angle();
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const style = this._tick_style_snapshot();
    for (const x of tickValues) {
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
      ticks.add(this._make_tick(x, size, cos, sin, style));
    }
    this.add(ticks);
    this.ticks = ticks;
    return this;
  }

  /** Baseline angle computed straight from the cached flat endpoints (avoids
   * the two `slice` allocations in `get_start_and_end` + `subtractPoints`
   * round-trip). Numerically identical to {@link get_angle}. */
  private _baseline_angle(): number {
    const flat = this._ensure_endpoints_flat();
    const dim = flat.length / 2;
    return Math.atan2(
      (flat[dim + 1] ?? 0) - (flat[1] ?? 0),
      (flat[dim] ?? 0) - (flat[0] ?? 0),
    );
  }

  /** Live style handles shared by every tick produced in one batch. The
   * rgba arrays are treated as immutable/copy-on-write throughout the
   * codebase (every `update_rgbas_array`/`apply_over_attr_arrays` assignment
   * replaces the array instead of mutating it in place), so handing the same
   * reference to all ticks is value-identical to `match_style` and avoids the
   * expensive `get_style` + `set_style` round-trip (which dominated
   * `add_ticks`). */
  private _tick_style_snapshot(): {
    fill_rgbas: NDArray;
    stroke_rgbas: NDArray;
    background_stroke_rgbas: NDArray;
    fill_opacity: number;
    stroke_opacity: number;
    stroke_width: number;
    background_stroke_color: VMobject["background_stroke_color"];
    background_stroke_opacity: number;
    background_stroke_width: number;
    sheen_factor: number;
    sheen_direction: number[];
    background_image: string | null;
  } {
    return {
      fill_rgbas: this.fill_rgbas,
      stroke_rgbas: this.stroke_rgbas,
      background_stroke_rgbas: this.background_stroke_rgbas,
      fill_opacity: this.fill_opacity,
      stroke_opacity: this.stroke_opacity,
      stroke_width: this.stroke_width,
      background_stroke_color: this.background_stroke_color,
      background_stroke_opacity: this.background_stroke_opacity,
      background_stroke_width: this.background_stroke_width,
      sheen_factor: this.sheen_factor,
      sheen_direction: this.sheen_direction,
      background_image: this.background_image,
    };
  }

  /** Apply a {@link _tick_style_snapshot} directly to a tick. Equivalent to
   * `tick.set_style({ ...this.get_style(), family: false })` (what
   * `match_style` does) without re-parsing colors or rebuilding the arrays. */
  private _apply_tick_style(
    tick: Line,
    style: ReturnType<NumberLine["_tick_style_snapshot"]>,
  ): void {
    tick.fill_rgbas = style.fill_rgbas;
    tick.stroke_rgbas = style.stroke_rgbas;
    tick.background_stroke_rgbas = style.background_stroke_rgbas;
    tick.fill_opacity = style.fill_opacity;
    tick.stroke_opacity = style.stroke_opacity;
    tick.stroke_width = style.stroke_width;
    tick.background_stroke_color = style.background_stroke_color;
    tick.background_stroke_opacity = style.background_stroke_opacity;
    tick.background_stroke_width = style.background_stroke_width;
    tick.sheen_factor = style.sheen_factor;
    tick.sheen_direction = [...style.sheen_direction];
    tick.background_image = style.background_image;
  }

  /** Build a tick `Line` directly at its final position, applying the
   * precomputed style. Mathematically (and, for the default axes, bit-for-bit)
   * equal to the Python `Line(DOWN*size, UP*size).rotate(angle).move_to(p)`
   * followed by `match_style(self)`, without constructing the line at the
   * origin, rotating it, translating it and re-deriving the whole style. */
  private _make_tick(
    x: number,
    size: number,
    cos: number,
    sin: number,
    style: ReturnType<NumberLine["_tick_style_snapshot"]>,
  ): Line {
    const center = this.number_to_point_flat(x);
    const cx = center[0] ?? 0;
    const cy = center[1] ?? 0;
    const cz = center[2] ?? 0;
    const result = new Line(
      [cx + sin * size, cy - cos * size, cz],
      [cx - sin * size, cy + cos * size, cz],
    );
    this._apply_tick_style(result, style);
    return result;
  }

  get_tick(x: number, size: number | null = null): Line {
    const resolved = size ?? this.tick_size;
    const angle = this._baseline_angle();
    return this._make_tick(
      x,
      resolved,
      Math.cos(angle),
      Math.sin(angle),
      this._tick_style_snapshot(),
    );
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

  /** Internal flat-array fast path for a single scalar `number_to_point`.
   * Returns a plain point row (`[x, y, z]`) so hot callers (Axes.coords_to_point,
   * tick/label placement) avoid allocating an NDArray for every point. The
   * public `number_to_point` still wraps this in an NDArray at the boundary. */
  number_to_point_flat(number: number): number[] {
    const scaled = this.scaling.inverse_function(number);
    const value = typeof scaled === "number" ? scaled : numeric(pointRows(scaled, 1)[0]?.[0] ?? 0);
    const xMin = this.x_range[0] ?? 0;
    const span = (this.x_range[1] ?? 0) - (this.x_range[0] ?? 0);
    const alpha = (value - xMin) / span;
    const flat = this._ensure_endpoints_flat();
    const dim = flat.length / 2;
    const result = new Array<number>(dim);
    for (let i = 0; i < dim; i += 1) {
      result[i] = (1 - alpha) * (flat[i] ?? 0) + alpha * (flat[dim + i] ?? 0);
    }
    return result;
  }

  /** Shared scalar interpolation of the baseline endpoints (mirrors
   * `interpolate(start, end, alpha)` for the point-row case, without the
   * numpy-ts round-trip). */
  private _interpolate_points_flat(alpha: number): number[] {
    const flat = this._ensure_endpoints_flat();
    const dim = flat.length / 2;
    const result = new Array<number>(dim);
    for (let i = 0; i < dim; i += 1) {
      result[i] = (1 - alpha) * (flat[i] ?? 0) + alpha * (flat[dim + i] ?? 0);
    }
    return result;
  }

  /** Batch variant of {@link number_to_point_flat}: applies the same mapping
   * to many values while reading the endpoint/scaling constants only once.
   * Used by Axes.plot's vectorized sampling path. */
  number_to_point_flat_many(values: readonly number[]): number[][] {
    const flat = this._ensure_endpoints_flat();
    const dim = flat.length / 2;
    const xMin = this.x_range[0] ?? 0;
    const span = (this.x_range[1] ?? 0) - (this.x_range[0] ?? 0);
    const scaling = this.scaling;
    const out: number[][] = new Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
      const scaled = scaling.inverse_function(values[i] ?? 0);
      const value = typeof scaled === "number" ? scaled : numeric(pointRows(scaled, 1)[0]?.[0] ?? 0);
      const alpha = (value - xMin) / span;
      const row = new Array<number>(dim);
      for (let d = 0; d < dim; d += 1) {
        row[d] = (1 - alpha) * (flat[d] ?? 0) + alpha * (flat[dim + d] ?? 0);
      }
      out[i] = row;
    }
    return out;
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
    if (scalar) {
      const value =
        typeof scaled === "number"
          ? scaled
          : numeric(pointRows(scaled, 1)[0]?.[0] ?? 0);
      return array(this._interpolate_points_flat((value - x_min) / span)) as NDArray;
    }
    const [start, end] = this.get_start_and_end();
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
    x_values: Iterable<number> | number | null = null,
    excluding: Iterable<number> | number | null = null,
    font_size: number | null = null,
    label_constructor: unknown = null,
    number_config: Record<string, unknown> = {},
  ): this {
    // NOTE: 2019-era manim used the variadic form `add_numbers(*values)`
    // (e.g. `add_numbers(*np.arange(0, 1.2, 0.2))`, `add_numbers(*range(1, 12))`),
    // which py2js compiles to `add_numbers.apply(obj, values)`. Modern manim
    // takes a single iterable instead. There is deliberately no `...extra`
    // rest parameter: the py2js kwargs bridge maps unknown keywords (e.g.
    // `buff`, `num_decimal_places`) onto the trailing `number_config`
    // options bag, but only when the last named parameter is an options bag
    // and no rest parameter is present. A rest parameter diverts those
    // keywords into a trailing plain object that lands in the wrong
    // positional slot (previously `label_constructor`, or even `excluding`,
    // whose spread then threw `TypeError: resolvedExcluding is not
    // iterable`). Legacy variadic positionals are collected from `arguments`
    // below instead.
    const extra: unknown[] = Array.prototype.slice.call(arguments, 5) as unknown[];
    const isVariadic =
      typeof x_values === "number" ||
      typeof x_values === "bigint" ||
      extra.length > 0;
    // A plain options/kwargs bag is never a valid value list or exclusion
    // list: the cross-language bridge can hand one through any positional
    // slot when parameter names are unavailable for keyword mapping. Such
    // bags are folded back into the number config instead of crashing.
    let absorbedConfig: Record<string, unknown> = {};
    const absorbOptionsBag = (value: unknown): boolean => {
      if (value === null || value === undefined) return false;
      if (typeof value !== "object") return false;
      if (Array.isArray(value) || isNDArray(value)) return false;
      if (typeof (value as Iterable<number>)[Symbol.iterator] === "function") return false;
      absorbedConfig = { ...(value as Record<string, unknown>), ...absorbedConfig };
      return true;
    };
    const isRecord = (value: unknown): value is Record<string, unknown> =>
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      !isNDArray(value) &&
      typeof (value as Iterable<unknown>)[Symbol.iterator] !== "function";
    const toNumberList = (value: unknown): number[] => {
      if (typeof value === "number" || typeof value === "bigint") return [Number(value)];
      if (isNDArray(value)) return ((value as NDArray).toArray() as unknown[]).map(Number);
      if (Array.isArray(value)) return (value as unknown[]).map(Number);
      return [...(value as Iterable<number>)].map(Number);
    };
    let values: number[];
    if (isVariadic) {
      const raw: unknown[] = [x_values, excluding, font_size, label_constructor, number_config, ...extra];
      // A trailing kwargs bag (null-prototype object tagged by the py2js
      // bridge) is not a value. The bridge normally maps kwargs onto named
      // slots already, so just drop such bags here.
      const withoutKwargs = raw.filter((item) => {
        if (item === null || item === undefined) return false;
        if (typeof item === "object" && !Array.isArray(item) && !isNDArray(item)) {
          const proto = Object.getPrototypeOf(item);
          if (proto === null) return false;
          // Plain option bags (e.g. `{font_size: 16}` arriving as trailing
          // object) are not numeric values.
          if (proto === Object.prototype) {
            const keys = Object.keys(item as Record<string, unknown>);
            if (keys.length > 0 && keys.every((k) => typeof k === "string")) {
              const hasNonNumeric = keys.some((k) =>
                ["excluding", "font_size", "label_constructor", "number_config", "fontSize"].includes(k),
              );
              if (hasNonNumeric) return false;
            }
          }
        }
        return true;
      });
      values = [];
      for (const item of withoutKwargs) {
        if (typeof item === "number" || typeof item === "bigint") {
          values.push(Number(item));
        } else if (isNDArray(item)) {
          values.push(...((item as NDArray).toArray() as unknown[]).map(Number));
        } else if (Array.isArray(item)) {
          values.push(...(item as unknown[]).map(Number));
        } else if (
          item !== null &&
          typeof item === "object" &&
          typeof (item as Iterable<number>)[Symbol.iterator] === "function"
        ) {
          values.push(...[...(item as Iterable<unknown>)].map(Number));
        }
      }
    } else if (x_values === null || x_values === undefined) {
      values = (this.get_tick_range().toArray() as unknown[]).map(Number);
    } else if (absorbOptionsBag(x_values)) {
      // NOTE: `x_values` itself arrived as an options bag (bridge artifact);
      // fall back to the default tick range.
      values = (this.get_tick_range().toArray() as unknown[]).map(Number);
    } else {
      values = toNumberList(x_values);
    }
    // NOTE: `excluding` may similarly arrive as an options bag (e.g.
    // `{font_size: 16, buff: 0.25}` when keyword mapping is unavailable);
    // recover it into the number config so the exclusion spread below cannot
    // throw `TypeError: resolvedExcluding is not iterable`. In variadic mode
    // the option slots hold numeric values instead, so defaults apply there
    // (mirroring the old behavior).
    let cleanExcluding: unknown = excluding;
    if (!isVariadic && absorbOptionsBag(cleanExcluding)) cleanExcluding = null;
    const baseExcluding = isVariadic
      ? (this.numbers_to_exclude ?? [])
      : ((cleanExcluding as Iterable<number> | number | null) ?? this.numbers_to_exclude ?? []);
    const excludingList = toNumberList(baseExcluding);
    const mergedConfig: Record<string, unknown> = isVariadic
      ? {}
      : {
          ...absorbedConfig,
          ...(isRecord(number_config) ? number_config : {}),
        };
    // NOTE: mirrors Python, where `add_numbers(..., **kwargs)` forwards the
    // leftover keywords to `get_number_mobject` (so `buff=`/`direction=`
    // bind there while the rest reaches `DecimalNumber`). Explicit slots win
    // over bag entries.
    const bagValue = (key: string): unknown => mergedConfig[key];
    const resolvedFontSize = isVariadic
      ? this.font_size
      : (font_size ?? (bagValue("font_size") as number | null) ?? (bagValue("fontSize") as number | null) ?? this.font_size);
    const resolvedConstructor = isVariadic
      ? this.label_constructor
      : (label_constructor ?? bagValue("label_constructor") ?? this.label_constructor);
    const resolvedDirection = isVariadic
      ? undefined
      : ((bagValue("direction") as readonly number[] | NDArray | null | undefined) ?? undefined);
    const resolvedBuff = isVariadic ? undefined : ((bagValue("buff") as number | null | undefined) ?? undefined);
    for (const key of ["font_size", "fontSize", "label_constructor", "direction", "buff"]) delete mergedConfig[key];
    const numbers = new VGroup();
    for (const x of values) {
      if (excludingList.includes(x)) continue;
      numbers.add(
        this.get_number_mobject(
          x,
          resolvedDirection,
          resolvedBuff,
          resolvedFontSize,
          resolvedConstructor,
          mergedConfig,
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
    options: NumberLineOptions | readonly number[] | null = {},
  ) {
    const resolved_ticks = numbers_with_elongated_ticks ?? [0, 1];
    const resolved_decimal = decimal_number_config ?? { num_decimal_places: 1 };
    const base_options: NumberLineOptions = options !== null && !Array.isArray(options)
      ? { ...(options as NumberLineOptions) }
      : {};
    if (Array.isArray(options)) {
      base_options.x_range = options as unknown as readonly [number, number, number];
    }
    super({
      ...base_options,
      x_range: [0, 1, 0.1],
      unit_size,
      numbers_with_elongated_ticks: resolved_ticks,
      decimal_number_config: resolved_decimal,
    });
  }
}
