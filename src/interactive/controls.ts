/**
 * Interactive controls: `PathSlider` (and its `Slider` line subclass),
 * `Switch`, `InputDecimalNumber` (plus `InputInteger`) and the editable
 * `InputDecimalMatrix` / `InputDecimalTable` containers.
 *
 * Each control is a normal `VMobject` (so it can be added to a scene, animated
 * and moved like any other mobject) that additionally implements the
 * {@link InteractiveControl} protocol used by `InteractiveScene`:
 *
 *  - `is_hit(point)` does hit testing in world coordinates;
 *  - `on_press` / `on_drag` / `on_release` mutate the control's value;
 *  - `get_value()` reads it back, so other mobjects can react to it in a
 *    blueprint `add_reaction` callback.
 *
 * The controls deliberately keep their interaction geometry (a slider's track,
 * a switch's bounding box) as world-space data captured at construction time:
 * they are meant to be `self.add(...)`-ed directly, not nested in groups that
 * move afterwards.
 */
import type { ParsableManimColor } from "../utils/color/index.js";
import { BLACK, BLUE_E, GREY_B, GREY_C, WHITE, YELLOW, interpolate_color, ManimColor } from "../utils/color/index.js";
import { LEFT, RIGHT } from "../constants.js";
import { Mobject } from "../mobject/mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import { Dot } from "../mobject/geometry/arc.js";
import { Line } from "../mobject/geometry/line.js";
import { isPlainOptionsBag, Rectangle, RoundedRectangle } from "../mobject/geometry/polygram.js";
import { Text } from "../mobject/text/text_mobject.js";
import { DecimalMatrix, type MatrixOptions } from "../mobject/matrix.js";
import { DecimalTable, type TableData, type TableOptions } from "../mobject/table.js";
import {
  DecimalNumber,
  roundToDecimals,
  type DecimalNumberOptions,
  type DecimalNumberValue,
} from "../mobject/text/numbers.js";
import type { InteractiveHost } from "./host.js";

/** Everything a control needs from the host to service an interaction. */
export interface InteractiveContext {
  host: InteractiveHost;
  camera: unknown;
}

/**
 * Symbol marker identifying interactive controls. A symbol (rather than a
 * string property) is used because wanim Mobjects are Proxies that raise an
 * `AttributeError` for unknown *string* properties; unknown symbols safely read
 * as `undefined`.
 */
export const INTERACTIVE_CONTROL = Symbol("wanim.interactive_control");

/** Protocol implemented by every interactive control. */
export interface InteractiveControl {
  get_value(): unknown;
  is_hit(point: readonly number[]): boolean;
  on_press(point: readonly number[], context: InteractiveContext): void;
  on_drag(point: readonly number[], context: InteractiveContext): void;
  on_release(): void;
  /**
   * Optional hover notification, delivered while the pointer moves over the
   * control and no drag is in progress. Controls that draw a hover highlight
   * implement it; the scene ignores controls that do not.
   */
  set_hovered?(hovered: boolean): void;
}

/** Duck-typed check that `mobject` is an interactive control. */
export function isInteractiveControl(value: unknown): value is Mobject & InteractiveControl {
  return Boolean(value) && (value as { [INTERACTIVE_CONTROL]?: unknown })[INTERACTIVE_CONTROL] === true;
}

function point2d(point: readonly number[]): [number, number] {
  return [point[0] ?? 0, point[1] ?? 0];
}

/** Read the first three components of a point-like value (array or NDArray). */
function point3d(point: unknown): [number, number, number] {
  const row = point as readonly number[];
  return [row[0] ?? 0, row[1] ?? 0, row[2] ?? 0];
}

/** Distance from `p` to the segment `a`-`b`, in the xy plane. */
function distanceToSegment(p: readonly number[], a: readonly number[], b: readonly number[]): number {
  const px = p[0] ?? 0;
  const py = p[1] ?? 0;
  const ax = a[0] ?? 0;
  const ay = a[1] ?? 0;
  const bx = b[0] ?? 0;
  const by = b[1] ?? 0;
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** Clamp `value` to `[0, 1]`. */
function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** Python-style `value % 1`, i.e. always in `[0, 1)`. */
function wrap01(value: number): number {
  const wrapped = value - Math.floor(value);
  return wrapped === 1 ? 0 : wrapped;
}

/** Python-style non-negative modulo. */
function mod(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

/**
 * Screen pixels per world unit, per axis, read off a camera-like object.
 * Falls back to a 100 px/unit guess when the camera does not expose its
 * pixel/frame geometry (custom or headless cameras).
 */
function pixelScale(camera: unknown): [number, number] {
  const cam = camera as {
    pixel_width?: number;
    pixel_height?: number;
    frame_width?: number;
    frame_height?: number;
  } | null;
  const pixelWidth = cam?.pixel_width;
  const pixelHeight = cam?.pixel_height;
  const frameWidth = cam?.frame_width;
  const frameHeight = cam?.frame_height;
  const scaleX =
    typeof pixelWidth === "number" && typeof frameWidth === "number" && pixelWidth > 0 && frameWidth > 0
      ? pixelWidth / frameWidth
      : 0;
  const scaleY =
    typeof pixelHeight === "number" && typeof frameHeight === "number" && pixelHeight > 0 && frameHeight > 0
      ? pixelHeight / frameHeight
      : 0;
  return [scaleX > 0 ? scaleX : 100, scaleY > 0 ? scaleY : 100];
}

export interface PathSliderOptions {
  value_range?: readonly number[];
  init_value?: number;
  color?: ParsableManimColor;
  stroke_width?: number;
  /** Opacity of the track (0 = invisible). Defaults to 1. */
  track_opacity?: number;
  /** Opacity of the knob (0 = invisible). Defaults to 1. */
  knob_opacity?: number;
  /** Knob radius (world units). */
  knob_radius?: number;
  /** Extra pick radius around the track (world units). */
  hit_radius?: number;
  /** Treat the path as a loop, wrapping the value across the seam. */
  is_closed?: boolean;
}

/**
 * A draggable slider whose track is an arbitrary VMobject path.
 *
 * The value is mapped onto the path by *arc-length proportion*: the low end of
 * `value_range` sits at the start of the path and the high end at its end.
 * With `is_closed` the path is treated as a loop, so dragging past the seam
 * wraps the value back around. Dragging is incremental and projected onto the
 * local tangent, so the knob follows the pointer only along the path.
 *
 * `attach(mobject)` parents a mobject to the knob so it follows the value;
 * while anything is attached the knob itself is hidden, so the attached mobject
 * is what the user sees (use `unattach` to restore the knob).
 */
export class PathSlider extends VGroup implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  path: VMobject;
  value_range: [number, number];
  accent_color: ParsableManimColor;
  stroke_width: number;
  track_opacity: number;
  knob_opacity: number;
  knob_radius: number;
  hit_radius: number;
  // NOTE: shadows the inherited `VMobject.is_closed()` method with the
  // interactive flag. PathSlider's own algorithms use the flag; the slider
  // itself is a group whose points are never passed to the VMobject
  // closedness check.
  // @ts-expect-error - intentional boolean shadow of VMobject.is_closed()
  is_closed: boolean;
  track: VMobject;
  knob: Dot;
  /** Mobjects currently parented to the knob via {@link attach}. */
  attached: Mobject[] = [];
  private _value: number;
  /** Arc-length proportion along the path; the source of truth for the knob. */
  private _t = 0;
  private _lastDragPoint: number[] | null = null;

  constructor(path: VMobject, options: PathSliderOptions = {}) {
    super();
    const {
      value_range = [0, 1],
      init_value,
      color = YELLOW,
      stroke_width = 6,
      track_opacity = 1,
      knob_opacity = 1,
      knob_radius = 0.12,
      hit_radius = 0.3,
      is_closed = false,
    } = options;
    const low = value_range[0] ?? 0;
    const high = value_range[1] ?? 1;
    this.path = path;
    this.track = path;
    this.value_range = [low, high];
    this.accent_color = color;
    this.stroke_width = stroke_width;
    this.track_opacity = track_opacity;
    this.knob_opacity = knob_opacity;
    this.knob_radius = knob_radius;
    this.hit_radius = hit_radius;
    this.is_closed = is_closed;
    this._value = low;
    path.set_stroke(color, stroke_width, track_opacity);
    this.knob = new Dot({
      point: [0, 0, 0],
      radius: knob_radius,
      color,
      fill_opacity: knob_opacity,
      stroke_width: 0,
    });
    this.add(path, this.knob);
    this.set_value(init_value ?? low);
  }

  get_value(): number {
    return this._value;
  }

  set_value(value: number): this {
    const [low, high] = this.value_range;
    const span = high - low;
    if (span === 0) {
      this._value = low;
      this._t = 0;
      this._move_knob();
      return this;
    }
    let resolved = value;
    if (this.is_closed && (value < low || value > high)) {
      // Closed path: out-of-range values wrap around the loop.
      resolved = low + mod(value - low, span);
    } else {
      resolved = Math.max(low, Math.min(high, value));
    }
    this._value = resolved;
    this._t = (resolved - low) / span;
    this._move_knob();
    return this;
  }

  /**
   * Parent `mobject` to the knob so it moves with the slider value. The knob
   * becomes transparent so only the attached mobject is visible.
   */
  attach(mobject: Mobject): this {
    if (this.attached.includes(mobject)) return this;
    this.attached.push(mobject);
    this.knob.add(mobject);
    this._update_knob_visibility();
    return this;
  }

  /**
   * Detach one mobject (or, with no argument, every attached mobject). When
   * nothing remains attached the knob's configured opacity is restored.
   */
  unattach(mobject?: Mobject): this {
    if (mobject === undefined) {
      for (const attached of [...this.attached]) this.knob.remove(attached);
      this.attached = [];
    } else {
      if (!this.attached.includes(mobject)) return this;
      this.knob.remove(mobject);
      this.attached = this.attached.filter((candidate) => candidate !== mobject);
    }
    this._update_knob_visibility();
    return this;
  }

  is_hit(point: readonly number[]): boolean {
    const samples = this._sample_points();
    for (let index = 0; index + 1 < samples.length; index += 1) {
      if (distanceToSegment(point, samples[index]!, samples[index + 1]!) <= this.hit_radius) {
        return true;
      }
    }
    if (this.is_closed && samples.length > 1) {
      const last = samples[samples.length - 1]!;
      const first = samples[0]!;
      if (distanceToSegment(point, last, first) <= this.hit_radius) return true;
    }
    const knobCenter = this.knob.get_center();
    const knobRadius = Math.max(this.hit_radius, this.knob_radius * 2);
    return (
      Math.hypot((point[0] ?? 0) - (knobCenter[0] ?? 0), (point[1] ?? 0) - (knobCenter[1] ?? 0)) <=
      knobRadius
    );
  }

  on_press(point: readonly number[], _context: InteractiveContext): void {
    const [low, high] = this.value_range;
    const t = this._proportion_from_point(point);
    this.set_value(low + t * (high - low));
    this._lastDragPoint = [point[0] ?? 0, point[1] ?? 0, point[2] ?? 0];
  }

  on_drag(point: readonly number[], context: InteractiveContext): void {
    const last = this._lastDragPoint;
    if (last === null) {
      this.on_press(point, context);
      return;
    }
    const dx = (point[0] ?? 0) - (last[0] ?? 0);
    const dy = (point[1] ?? 0) - (last[1] ?? 0);
    const total = this._total_length();
    if (total > 0 && (dx !== 0 || dy !== 0)) {
      const [tangentX, tangentY] = this._tangent_at(this._t);
      this._set_proportion(this._t + (dx * tangentX + dy * tangentY) / total);
    }
    this._lastDragPoint = [point[0] ?? 0, point[1] ?? 0, point[2] ?? 0];
  }

  on_release(): void {
    this._lastDragPoint = null;
  }

  private _update_knob_visibility(): void {
    const opacity = this.attached.length > 0 ? 0 : this.knob_opacity;
    // `family = false`: only the knob's own points change, never the attached
    // mobjects' styles.
    this.knob.set_opacity(opacity, false);
  }

  /** Move the knob to the current proportion (without changing the value). */
  private _move_knob(): void {
    this.knob.move_to(this._point_at(this._t));
  }

  /** Set value/proportion directly from an arc-length proportion. */
  private _set_proportion(t: number): void {
    const bounded = this.is_closed ? wrap01(t) : clamp01(t);
    const [low, high] = this.value_range;
    this._t = bounded;
    this._value = low + bounded * (high - low);
    this._move_knob();
  }

  /**
   * Pointful leaves of the path, in family order. `VGroup.get_arc_length()`
   * returns 0 for a group (no curves of its own), so proportions must be
   * computed from each family member's own arc length.
   */
  private _parts(): Array<{ vm: VMobject; length: number }> {
    const parts: Array<{ vm: VMobject; length: number }> = [];
    for (const mob of this.path.get_family()) {
      const vm = mob as VMobject;
      const length = vm.get_arc_length();
      if (length > 0) parts.push({ vm, length });
    }
    return parts;
  }

  private _total_length(): number {
    let total = 0;
    for (const part of this._parts()) total += part.length;
    return total;
  }

  /** Point on the path at arc-length proportion `t` (in `[0, 1]`). */
  private _point_at(t: number): number[] {
    const parts = this._parts();
    if (parts.length === 0) {
      return point3d(this.path.get_center());
    }
    if (!this.is_closed && t >= 1) {
      return point3d(parts[parts.length - 1]!.vm.point_from_proportion(1));
    }
    const bounded = this.is_closed ? wrap01(t) : clamp01(t);
    let total = 0;
    for (const part of parts) total += part.length;
    const target = bounded * total;
    let current = 0;
    for (const part of parts) {
      if (current + part.length >= target) {
        const residue = clamp01((target - current) / part.length);
        return point3d(part.vm.point_from_proportion(residue));
      }
      current += part.length;
    }
    return point3d(parts[parts.length - 1]!.vm.point_from_proportion(1));
  }

  /** Number of equal-proportion samples used by hit testing / pressing. */
  private _sample_count(): number {
    let curves = 0;
    for (const mob of this.path.get_family()) curves += (mob as VMobject).get_num_curves();
    return Math.max(96, 16 * curves);
  }

  private _sample_points(): number[][] {
    const count = this._sample_count();
    const points: number[][] = [];
    for (let index = 0; index <= count; index += 1) {
      points.push(this._point_at(index / count));
    }
    return points;
  }

  /**
   * Nearest path proportion to `point`: a coarse sample scan followed by a
   * ternary-search refinement between the two neighbouring samples.
   */
  private _proportion_from_point(point: readonly number[]): number {
    const count = this._sample_count();
    const px = point[0] ?? 0;
    const py = point[1] ?? 0;
    let bestT = 0;
    let bestDistance = Infinity;
    for (let index = 0; index <= count; index += 1) {
      const t = index / count;
      const candidate = this._point_at(t);
      const dx = px - (candidate[0] ?? 0);
      const dy = py - (candidate[1] ?? 0);
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestT = t;
      }
    }
    const step = 1 / count;
    let low = bestT - step;
    let high = bestT + step;
    if (!this.is_closed) {
      low = Math.max(0, low);
      high = Math.min(1, high);
    }
    const bracketWidth = high - low;
    // Ternary search on the (non-squared) distance.
    for (let iteration = 0; iteration < 80; iteration += 1) {
      const m1 = low + (high - low) / 3;
      const m2 = high - (high - low) / 3;
      if (this._distance_at(px, py, m1) <= this._distance_at(px, py, m2)) high = m2;
      else low = m1;
    }
    // Function-value comparisons stall at ~1e-8 around a smooth minimum, so
    // finish with a few chord projections: for a straight path the first one
    // is the analytic projection and reaches machine precision.
    let t = (low + high) / 2;
    let width = bracketWidth;
    for (let iteration = 0; iteration < 4; iteration += 1) {
      t = this._project_on_chord(px, py, t - width, t + width);
      width *= 0.25;
    }
    return t;
  }

  /**
   * Proportion whose path point is closest to `(px, py)` when projected onto
   * the straight chord between `lo` and `hi` (which may wrap for closed paths).
   */
  private _project_on_chord(px: number, py: number, lo: number, hi: number): number {
    let boundedLo = lo;
    let boundedHi = hi;
    if (!this.is_closed) {
      boundedLo = Math.max(0, Math.min(1, boundedLo));
      boundedHi = Math.max(0, Math.min(1, boundedHi));
    }
    const p0 = this._point_at(boundedLo);
    const p1 = this._point_at(boundedHi);
    const vx = (p1[0] ?? 0) - (p0[0] ?? 0);
    const vy = (p1[1] ?? 0) - (p0[1] ?? 0);
    const lengthSq = vx * vx + vy * vy;
    if (lengthSq < 1e-24) return (lo + hi) / 2;
    const u = clamp01(
      ((px - (p0[0] ?? 0)) * vx + (py - (p0[1] ?? 0)) * vy) / lengthSq,
    );
    return boundedLo + u * (boundedHi - boundedLo);
  }

  private _distance_at(px: number, py: number, t: number): number {
    const bounded = this.is_closed ? wrap01(t) : clamp01(t);
    const candidate = this._point_at(bounded);
    return Math.hypot(px - (candidate[0] ?? 0), py - (candidate[1] ?? 0));
  }

  /** Unit tangent (direction of increasing proportion) at `t`. */
  private _tangent_at(t: number): [number, number] {
    const epsilon = 1e-3;
    let t0: number;
    let t1: number;
    if (this.is_closed) {
      t0 = wrap01(t - epsilon);
      t1 = wrap01(t + epsilon);
    } else {
      t0 = Math.max(0, t - epsilon);
      t1 = Math.min(1, t + epsilon);
    }
    const p0 = this._point_at(t0);
    const p1 = this._point_at(t1);
    const dx = (p1[0] ?? 0) - (p0[0] ?? 0);
    const dy = (p1[1] ?? 0) - (p0[1] ?? 0);
    const length = Math.hypot(dx, dy);
    if (length < 1e-12) return [0, 0];
    return [dx / length, dy / length];
  }
}

export interface SliderOptions extends PathSliderOptions {
  start?: readonly number[];
  end?: readonly number[];
}

/**
 * A draggable slider drawn as a line with a dot knob (a {@link PathSlider}
 * whose path is the straight `start`-`end` segment).
 *
 * `value_range` is inclusive: dragging to `start` yields the first element, to
 * `end` the last. `init_value` (defaulting to the first element) sets the knob
 * position at construction time.
 */
export class Slider extends PathSlider {
  declare track: Line;
  start_point: number[];
  end_point: number[];

  constructor(options: SliderOptions = {}) {
    const {
      start = [-1, 0, 0],
      end = [1, 0, 0],
      color = YELLOW,
      stroke_width = 6,
      ...rest
    } = options;
    const startPoint = [start[0] ?? 0, start[1] ?? 0, start[2] ?? 0];
    const endPoint = [end[0] ?? 0, end[1] ?? 0, end[2] ?? 0];
    const line = new Line({
      start: startPoint,
      end: endPoint,
      color,
      stroke_width,
    });
    super(line, { color, stroke_width, ...rest });
    this.start_point = startPoint;
    this.end_point = endPoint;
    this.track = line;
  }
}

export interface SwitchOptions {
  /** Initial boolean value. */
  value?: boolean;
  /** Width of the track (world units). */
  width?: number;
  /** Height of the track (world units). */
  height?: number;
  /** Accent colour used when the switch is on. */
  color?: ParsableManimColor;
}

/**
 * A blue/white toggle switch. Clicking flips `get_value()` between `true` and
 * `false`.
 */
export class Switch extends VGroup implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  track: Rectangle;
  knob: Rectangle;
  track_width: number;
  track_height: number;
  accent_color: ParsableManimColor;
  private _value: boolean;

  constructor(valueOrOptions: boolean | SwitchOptions = false, options: SwitchOptions = {}) {
    super();
    let bag: SwitchOptions;
    if (typeof valueOrOptions === "boolean") {
      bag = { ...options, value: valueOrOptions };
    } else {
      bag = { ...valueOrOptions, ...options };
    }
    const { value = false, width = 0.66, height = 0.33, color = BLUE_E } = bag;
    this.track_width = width;
    this.track_height = height;
    this.accent_color = color;
    this._value = value;

    // Off: white track with a grey knob; on: blue track with a white knob. The
    // knob covers exactly one half (with a little padding).
    const pad = height * 0.14;
    this.track = new Rectangle({
      width,
      height,
      stroke_width: 0,
    });
    this.track.set_fill(WHITE, 1);
    this.knob = new Rectangle({
      width: width / 2 - 2 * pad,
      height: height - 2 * pad,
      color: WHITE,
      stroke_width: 0,
    });
    this.knob.set_fill(WHITE, 1);
    this.knob.move_to(this.track.get_center());
    this.add(this.track, this.knob);
    this._apply_state();
  }

  get_value(): boolean {
    return this._value;
  }

  set_value(value: boolean): this {
    this._value = Boolean(value);
    this._apply_state();
    return this;
  }

  toggle(): this {
    return this.set_value(!this._value);
  }

  is_hit(point: readonly number[]): boolean {
    const left = this.track.get_left();
    const right = this.track.get_right();
    const bottom = this.track.get_bottom();
    const top = this.track.get_top();
    const pad = this.track_height * 0.25;
    const [x, y] = point2d(point);
    return (
      x >= (left[0] ?? 0) - pad &&
      x <= (right[0] ?? 0) + pad &&
      y >= (bottom[1] ?? 0) - pad &&
      y <= (top[1] ?? 0) + pad
    );
  }

  on_press(_point: readonly number[], _context: InteractiveContext): void {
    this.toggle();
  }

  on_drag(_point: readonly number[], _context: InteractiveContext): void {}

  on_release(): void {}

  private _apply_state(): void {
    // Off: white track, grey knob (left). On: blue track, white knob (right).
    const on = this._value;
    this.track.set_fill(on ? this.accent_color : GREY_C, 1);
    this.knob.set_fill(WHITE, 1);
    const center = this.track.get_center();
    const quarter = this.track_width / 4;
    const x = (center[0] ?? 0) + (on ? quarter : -quarter);
    this.knob.move_to([x, center[1] ?? 0, center[2] ?? 0]);
  }
}

export interface ChipOptions {
  /** Initial selected state. Defaults to `false`. */
  value?: boolean;
  /** Accent colour: the label colour and, when selected, the border/fill tint. */
  color?: ParsableManimColor;
  /** Chip height in world units. Defaults to `0.52`. */
  height?: number;
  /** Label font size. Defaults to `24`. */
  font_size?: number;
  /** Horizontal padding around the label. Defaults to `0.22`. */
  padding?: number;
  /** Corner radius. Defaults to `height * 0.22`. */
  corner_radius?: number;
  /** Fill colour of an unselected chip. Defaults to a dark grey. */
  fill_color?: ParsableManimColor;
  /** Border colour of an unselected chip. Defaults to a mid grey. */
  stroke_color?: ParsableManimColor;
  /** Stroke width. Defaults to `1.5`. */
  stroke_width?: number;
}

/**
 * A tag-like toggle chip: a rounded box with a centred label and a trailing
 * `×` (shown in every state, like the original explorable-video tag rows).
 * Clicking anywhere on the chip flips its boolean value; selected chips get an
 * accent-tinted fill, border and label.
 */
export class Chip extends VGroup implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  box: RoundedRectangle;
  label: Text;
  cross: Text;
  accent_color: ParsableManimColor;
  unselected_fill: ParsableManimColor;
  unselected_stroke: ParsableManimColor;
  stroke_width: number;
  private _value: boolean;

  constructor(
    label: string | (ChipOptions & { label?: string }) = "",
    options: ChipOptions = {},
  ) {
    super();
    const bag: ChipOptions & { label?: string } = isPlainOptionsBag(label)
      ? { ...(label as ChipOptions & { label?: string }), ...options }
      : { ...options };
    const positionalLabel = isPlainOptionsBag(label) ? undefined : (label as string);
    const {
      value = false,
      color = WHITE,
      height = 0.52,
      font_size = 24,
      padding = 0.22,
      corner_radius = height * 0.22,
      fill_color = "#202020",
      stroke_color = GREY_C,
      stroke_width = 1.5,
    } = bag;
    const text = bag.label ?? positionalLabel ?? "";
    this._value = value;
    this.accent_color = color;
    this.unselected_fill = fill_color;
    this.unselected_stroke = stroke_color;
    this.stroke_width = stroke_width;

    this.label = new Text(text, { font_size, color });
    this.cross = new Text("×", { font_size: font_size * 0.9, color: GREY_B });
    const gap = padding * 0.5;
    const contentWidth = this.label.get_width() + gap + this.cross.get_width();
    this.box = new RoundedRectangle({
      width: contentWidth + 2 * padding,
      height,
      corner_radius,
      stroke_width,
    });
    this.box.set_fill(fill_color, 1);
    this.box.set_stroke(stroke_color, stroke_width);
    this.label.move_to(this.box.get_center());
    this.label.shift([-(this.cross.get_width() + gap) / 2, 0, 0]);
    this.cross.next_to(this.label, RIGHT, gap);
    this.add(this.box, this.label, this.cross);
    this._apply_state();
  }

  get_value(): boolean {
    return this._value;
  }

  set_value(value: boolean): this {
    this._value = Boolean(value);
    this._apply_state();
    return this;
  }

  toggle(): this {
    return this.set_value(!this._value);
  }

  is_hit(point: readonly number[]): boolean {
    const left = this.box.get_left();
    const right = this.box.get_right();
    const bottom = this.box.get_bottom();
    const top = this.box.get_top();
    const pad = this.box.height * 0.15;
    const [x, y] = point2d(point);
    return (
      x >= (left[0] ?? 0) - pad &&
      x <= (right[0] ?? 0) + pad &&
      y >= (bottom[1] ?? 0) - pad &&
      y <= (top[1] ?? 0) + pad
    );
  }

  on_press(_point: readonly number[], _context: InteractiveContext): void {
    this.toggle();
  }

  on_drag(_point: readonly number[], _context: InteractiveContext): void {}

  on_release(): void {}

  private _apply_state(): void {
    const on = this._value;
    const accent = new ManimColor(this.accent_color);
    this.box.set_fill(on ? interpolate_color(new ManimColor(BLACK), accent, 0.18) : this.unselected_fill, 1);
    this.box.set_stroke(on ? this.accent_color : this.unselected_stroke, this.stroke_width);
    this.label.set_color(on ? this.accent_color : GREY_B);
    this.cross.set_color(on ? this.accent_color : this.unselected_stroke);
  }
}

export interface InputDecimalNumberOptions extends DecimalNumberOptions {
  /**
   * Inclusive `[low, high]` bound for the value. Text edits and vertical drags
   * are clamped into it (overflow -> high, underflow -> low); `null` (the
   * default) leaves the number unbounded.
   */
  value_range?: readonly number[] | null;
  /**
   * Vertical screen distance (in pixels) that sweeps `drag_span` while
   * scrubbing. Defaults to `150`.
   */
  drag_pixels?: number;
  /**
   * Value change produced by a `drag_pixels`-long drag. Defaults to the
   * `value_range` span, or ten display steps (`10 * 10^-num_decimal_places`)
   * when the number is unbounded.
   */
  drag_span?: number;
  /** Fill opacity of the hover highlight layer. Defaults to `0.25`. */
  highlight_opacity?: number;
}

/** A vertical drag shorter than this (in screen pixels) is still a click. */
const SCRUB_START_PIXELS = 4;

/**
 * A `DecimalNumber` whose value can be edited by clicking it (the host opens an
 * inline text editor anchored over the number) or scrubbed by pressing it and
 * dragging up/down. While the pointer hovers the number a translucent
 * rectangle in the number's own colour is drawn behind the digits.
 *
 * With `value_range` the value is clamped to `[low, high]` both on text edits
 * and on drags, so `get_value()` always stays inside the range.
 */
export class InputDecimalNumber extends DecimalNumber implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  editing = false;
  /**
   * Called after a successful edit updates the value (every keystroke while
   * editing, every scrub step while dragging, and again on commit). `null`
   * means "no change handler".
   */
  on_change: (() => void) | null = null;
  /** Inclusive `[low, high]` bound, or `null` when the number is unbounded. */
  value_range: [number, number] | null;
  drag_pixels: number;
  drag_span: number | null;
  highlight_opacity: number;
  /** Whether the pointer currently hovers the number. */
  hovered = false;
  /** Highlight rectangle drawn behind the glyphs (`null` before construction). */
  highlight: Rectangle | null = null;
  private _press_context: InteractiveContext | null = null;
  private _press_point: number[] | null = null;
  private _last_drag_point: number[] | null = null;
  private _scrubbing = false;
  /**
   * Fractional drag accumulator: slow drags accumulate until they cross a
   * display step instead of rounding away on every pointer event.
   */
  private _scrub_value = 0;
  /**
   * Current inline-editor text while editing (`null` outside a session). The
   * rendered number always shows the forced sign (`include_sign`), so the
   * caret mapping must know whether the text carries its own sign.
   */
  private _edit_text: string | null = null;

  constructor(numberValue: DecimalNumberValue = 0, options: InputDecimalNumberOptions = {}) {
    const {
      value_range = null,
      drag_pixels = 150,
      drag_span = null,
      highlight_opacity = 0.25,
      ...decimalOptions
    } = options;
    // Grouping separators (10,000) make the caret/glyph mapping confusing for a
    // live-edited number, so the interactive input defaults to no grouping.
    super(clampToValueRange(numberValue, value_range), { group_with_commas: false, ...decimalOptions });
    const low = value_range?.[0] ?? 0;
    const high = value_range?.[1] ?? 1;
    this.value_range = value_range === null || value_range === undefined ? null : [low, high];
    this.drag_pixels = drag_pixels > 0 ? drag_pixels : 150;
    this.drag_span = drag_span;
    this.highlight_opacity = highlight_opacity;
    this._sync_highlight();
  }

  override get_value(): DecimalNumberValue {
    return super.get_value();
  }

  /**
   * Set the value, clamping it into `value_range` first (no-op for complex
   * values, which are not scrubbed).
   */
  override set_value(value: DecimalNumberValue): this {
    // The highlight must be absent while DecimalNumber rebuilds/repositions
    // its glyphs: as the first "old" submobject it would donate its own
    // translucent style to the first glyph, and it would skew the bounding box
    // `move_to` reads while re-laying out.
    const hadHighlight = this.highlight !== null;
    if (hadHighlight) this._detach_highlight();
    super.set_value(this._clamp_value(value));
    if (this.highlight !== null) this._sync_highlight();
    return this;
  }

  is_hit(point: readonly number[]): boolean {
    const left = this.get_left();
    const right = this.get_right();
    const bottom = this.get_bottom();
    const top = this.get_top();
    const pad = Math.max(this.height, 0.2) * 0.4;
    const [x, y] = point2d(point);
    return (
      x >= (left[0] ?? 0) - pad &&
      x <= (right[0] ?? 0) + pad &&
      y >= (bottom[1] ?? 0) - pad &&
      y <= (top[1] ?? 0) + pad
    );
  }

  on_press(point: readonly number[], context: InteractiveContext): void {
    // The gesture is decided on release: a click opens the editor, a vertical
    // drag scrubs the value.
    this._press_context = context;
    this._press_point = [...point3d(point)];
    this._last_drag_point = [...point3d(point)];
    this._scrubbing = false;
    this._scrub_value = this._scalar_value();
  }

  on_drag(point: readonly number[], context: InteractiveContext): void {
    const last = this._last_drag_point;
    if (last === null) {
      this.on_press(point, context);
      return;
    }
    // Prefer the host's screen scale (a CSS-scaled canvas would otherwise make
    // a "150 px" drag depend on the backing-store size).
    const [pixelsX, pixelsY] =
      context.host.pixelsPerWorldUnit?.(context.camera) ?? pixelScale(context.camera);
    const press = this._press_point ?? last;
    const dx = (point[0] ?? 0) - (press[0] ?? 0);
    const dy = (point[1] ?? 0) - (press[1] ?? 0);
    if (!this._scrubbing) {
      if (Math.hypot(dx * pixelsX, dy * pixelsY) < SCRUB_START_PIXELS) {
        this._last_drag_point = [...point3d(point)];
        return;
      }
      this._scrubbing = true;
    }
    const step = ((point[1] ?? 0) - (last[1] ?? 0)) * pixelsY * (this._drag_span_value() / this.drag_pixels);
    this._scrub_value += step;
    const quantized = this._quantize_value(this._scrub_value);
    const clamped = this._clamp_scalar(quantized);
    // Hitting a bound drops the accumulated overshoot, so reversing the drag
    // from the boundary responds immediately. In-range integer quantization
    // keeps the fraction (0.8 stays 0.8 until the next step crosses 1).
    if (clamped !== quantized) this._scrub_value = clamped;
    this.set_value(clamped);
    this.on_change?.();
    this._last_drag_point = [...point3d(point)];
  }

  on_release(): void {
    const context = this._press_context;
    const wasScrubbing = this._scrubbing;
    this._press_context = null;
    this._press_point = null;
    this._last_drag_point = null;
    this._scrubbing = false;
    if (!wasScrubbing && context !== null) this._begin_edit(context);
  }

  /** Update the hover highlight (called by the scene on pointer motion). */
  set_hovered(hovered: boolean): void {
    const next = Boolean(hovered);
    if (this.hovered === next) return;
    this.hovered = next;
    this._apply_highlight_style();
  }

  /** Open the host's inline editor for this number. */
  protected _begin_edit(context: InteractiveContext): void {
    this.editing = true;
    const initial = this._format_value();
    this._edit_text = initial;
    context.host.beginTextEdit(this, context.camera, {
      initial,
      allow_decimal_point: this._allow_decimal_point(),
      onInput: (text) => this._apply_text(text),
      onCommit: (text) => {
        this._apply_text(text);
        this.editing = false;
        this._edit_text = null;
      },
      onCancel: () => {
        this.editing = false;
        this._edit_text = null;
      },
    });
  }

  /** Whether the inline editor accepts a decimal point (default: yes). */
  protected _allow_decimal_point(): boolean {
    return true;
  }

  protected _apply_text(text: string): void {
    this._edit_text = text;
    const value = Number.parseFloat(text);
    if (Number.isFinite(value)) {
      this.set_value(value);
      this.on_change?.();
    }
  }

  protected _format_value(): string {
    const value = this.get_value();
    // The editor starts from exactly what is rendered (same sign, same
    // rounding), so the initial caret maps one-to-one onto the glyphs.
    // Grouping separators are dropped: the editor only accepts a numeric
    // literal (`1,000` would parse as `1`).
    if (typeof value !== "number") return value.re.toFixed(this.num_decimal_places);
    return this._get_num_string(value).replace(/,/g, "");
  }

  /**
   * World-space anchor for the character insertion point at `index` (0 = before
   * the first glyph, `glyphs` = after the last). Used by the browser host to
   * draw the editing marker under the digit being edited. Returns `null` when
   * the number has no glyphs yet.
   */
  interactive_caret_anchor(index: number): number[] | null {
    // `_set_submobjects_from_number` adds one submobject per displayed
    // character (in order); unit/background submobjects, if any, come after,
    // and the hover highlight sits before the glyphs.
    const display = this._get_num_string(this.number);
    const glyphs = this.submobjects.filter((member) => member !== this.highlight).slice(0, display.length);
    if (glyphs.length === 0) return null;
    // The displayed number may carry a sign the editor text does not (e.g.
    // `include_sign` renders `+0.00` while the text is `0.00`), or vice versa.
    // Shift the index past a display-only sign so the caret stays on the digit
    // the user is editing instead of one glyph to its left.
    const text = this._edit_text ?? this._format_value();
    const displaySign = display.startsWith("+") || display.startsWith("-") ? 1 : 0;
    const textSign = text.startsWith("+") || text.startsWith("-") ? 1 : 0;
    const glyphIndex = index + displaySign - textSign;
    const bottom = this.get_bottom();
    const y = bottom[1] ?? 0;
    if (glyphIndex >= glyphs.length) {
      const right = glyphs[glyphs.length - 1]!.get_right();
      return [right[0] ?? 0, y, 0];
    }
    const left = glyphs[Math.max(0, glyphIndex)]!.get_left();
    return [left[0] ?? 0, y, 0];
  }

  /** The value as a real number (complex values scrub their real part). */
  private _scalar_value(): number {
    const value = this.get_value();
    return typeof value === "number" ? value : value.re;
  }

  /** Clamp a real value into `value_range` (complex values pass through). */
  private _clamp_value(value: DecimalNumberValue): DecimalNumberValue {
    return typeof value === "number" ? this._clamp_scalar(value) : value;
  }

  /** Clamp a scalar into `value_range` (identity when unbounded). */
  private _clamp_scalar(value: number): number {
    const range = this.value_range;
    if (range === null) return value;
    return Math.max(range[0], Math.min(range[1], value));
  }

  /** Value change produced by `drag_pixels` of vertical drag. */
  private _drag_span_value(): number {
    if (this.drag_span !== null) return this.drag_span;
    if (this.value_range !== null) return this.value_range[1] - this.value_range[0];
    return 10 / 10 ** this.num_decimal_places;
  }

  /** Snap one scrub step (identity for decimals; integers round). */
  protected _quantize_value(value: number): number {
    return value;
  }

  /** Remove the highlight from the submobject list (before a glyph rebuild). */
  private _detach_highlight(): void {
    const highlight = this.highlight;
    if (highlight === null) return;
    const index = this.submobjects.indexOf(highlight);
    if (index >= 0) this.submobjects.splice(index, 1);
  }

  /** Re-fit the highlight to the glyphs and put it behind them. */
  private _sync_highlight(): void {
    if (this.submobjects.length === 0) return;
    let highlight = this.highlight;
    if (highlight === null) {
      highlight = new Rectangle({ width: 1, height: 1, stroke_width: 0 });
      this.highlight = highlight;
    }
    const left = this.get_left()[0] ?? 0;
    const right = this.get_right()[0] ?? 0;
    const bottom = this.get_bottom()[1] ?? 0;
    const top = this.get_top()[1] ?? 0;
    const center = this.get_center();
    highlight.stretch_to_fit_width(Math.max(right - left, 1e-6));
    highlight.stretch_to_fit_height(Math.max(top - bottom, 1e-6));
    highlight.move_to(center);
    this.submobjects.unshift(highlight);
    this._apply_highlight_style();
  }

  private _apply_highlight_style(): void {
    const highlight = this.highlight;
    if (highlight === null) return;
    highlight.set_fill(this.get_color() ?? WHITE, this.hovered ? this.highlight_opacity : 0);
  }
}

/** Clamp a (possibly complex) initial value into `value_range`. */
function clampToValueRange(
  value: DecimalNumberValue,
  range: readonly number[] | null | undefined,
): DecimalNumberValue {
  if (range === null || range === undefined || typeof value !== "number") return value;
  const low = range[0] ?? 0;
  const high = range[1] ?? 1;
  return Math.max(low, Math.min(high, value));
}

/**
 * An `Integer` whose value can be edited by clicking it or scrubbed by
 * dragging. Text edits are rounded/truncated like {@link Integer}; the inline
 * editor does not accept a decimal point, and a scrub drag snaps to whole
 * integers.
 */
export class InputInteger extends InputDecimalNumber {
  constructor(numberValue: DecimalNumberValue = 0, options: InputDecimalNumberOptions = {}) {
    super(numberValue, {
      ...options,
      num_decimal_places: options.num_decimal_places ?? 0,
    });
  }

  override get_value(): number {
    // NOTE: mirrors `Integer.get_value`: `int(np.round(value))`.
    const raw = super.get_value();
    const scalar = typeof raw === "number" ? raw : raw.re;
    return Math.trunc(roundToDecimals(scalar, 0));
  }

  protected override _allow_decimal_point(): boolean {
    return false;
  }

  protected override _quantize_value(value: number): number {
    return Math.trunc(roundToDecimals(value, 0));
  }
}

/** Build an `InputDecimalNumber` for a matrix/table cell. */
function inputDecimalNumberElement(
  item: unknown,
  config: Record<string, unknown> = {},
): VMobject {
  return new InputDecimalNumber(item as DecimalNumberValue, config as DecimalNumberOptions);
}

/** Register `relayout` as the change handler of every interactive cell. */
function attachRelayout(cells: VMobject[][], relayout: () => void): void {
  for (const row of cells) {
    for (const cell of row) {
      if (isInteractiveControl(cell)) {
        (cell as unknown as { on_change: (() => void) | null }).on_change = relayout;
      }
    }
  }
}

export interface InputDecimalMatrixOptions extends MatrixOptions {}

/**
 * A {@link DecimalMatrix} whose entries are editable {@link InputDecimalNumber}
 * cells. Editing a cell re-runs {@link InputDecimalMatrix.relayout}, so the
 * grid and brackets follow the new cell sizes.
 *
 * The grid layout is the content-aware one implemented by
 * {@link DecimalMatrix._organize_mob_matrix}: pitches are `h_buff`/`v_buff`
 * while the entries fit, and grow to `entry + entry_gap` when they do not.
 * Construction therefore already matches an equivalent {@link DecimalMatrix},
 * and edits never overlap neighbouring cells.
 */
export class InputDecimalMatrix extends DecimalMatrix {
  constructor(matrix: unknown[][], options: InputDecimalMatrixOptions = {}) {
    const {
      element_to_mobject = inputDecimalNumberElement,
      element_to_mobject_config = { num_decimal_places: 1 },
      ...rest
    } = options;
    super(matrix, { element_to_mobject, element_to_mobject_config, ...rest });
    attachRelayout(this.mob_matrix, () => this.relayout());
  }

  /**
   * Re-place the cells on the shared content-aware grid (the same algorithm
   * used at construction), re-fit the brackets, and shift everything back to
   * its previous center.
   */
  relayout(): this {
    const savedCenter = this.get_center();
    this._organize_mob_matrix(this.mob_matrix);
    if (this.stretch_brackets) {
      this.brackets.stretch_to_fit_height(this.elements.height + 2 * this.bracket_v_buff);
    }
    const brackets = this.brackets.submobjects as VMobject[];
    const leftBracket = brackets[0];
    const rightBracket = brackets[1];
    if (leftBracket !== undefined) {
      leftBracket.next_to(this.elements, [...LEFT], this.bracket_h_buff);
    }
    if (rightBracket !== undefined) {
      rightBracket.next_to(this.elements, [...RIGHT], this.bracket_h_buff);
    }
    const newCenter = this.get_center();
    this.shift([
      (savedCenter[0] ?? 0) - (newCenter[0] ?? 0),
      (savedCenter[1] ?? 0) - (newCenter[1] ?? 0),
      (savedCenter[2] ?? 0) - (newCenter[2] ?? 0),
    ]);
    return this;
  }
}

export interface InputDecimalTableOptions extends TableOptions {}

/**
 * A {@link DecimalTable} whose entries are editable {@link InputDecimalNumber}
 * cells. Editing a cell re-runs {@link InputDecimalTable.relayout}, which
 * re-arranges the grid and rebuilds the table lines around the new layout.
 */
export class InputDecimalTable extends DecimalTable {
  constructor(table: TableData | TableOptions = [], options: InputDecimalTableOptions = {}) {
    const {
      element_to_mobject = inputDecimalNumberElement,
      element_to_mobject_config = { num_decimal_places: 1 },
      ...rest
    } = options;
    super(table, { element_to_mobject, element_to_mobject_config, ...rest });
    attachRelayout(this.mob_table, () => this.relayout());
  }

  /**
   * Re-arrange the cells (same cells, moved in place) and rebuild the
   * horizontal/vertical lines, keeping the table centered where it was.
   */
  relayout(): this {
    const savedCenter = this.get_center();
    this._organize_mob_table(this.mob_table);
    // `Table._add_*_lines` adds each `Line` directly to the table (the
    // `horizontal_lines`/`vertical_lines` VGroups only wrap them), so removing
    // the groups themselves is a no-op and every edit would stack another set
    // of lines (and widen the table's bbox, making each new horizontal line
    // longer).  Remove the wrapped lines instead.
    this.remove(...this.horizontal_lines.submobjects, ...this.vertical_lines.submobjects);
    this._add_horizontal_lines();
    this._add_vertical_lines();
    const newCenter = this.get_center();
    this.shift([
      (savedCenter[0] ?? 0) - (newCenter[0] ?? 0),
      (savedCenter[1] ?? 0) - (newCenter[1] ?? 0),
      (savedCenter[2] ?? 0) - (newCenter[2] ?? 0),
    ]);
    return this;
  }
}

// Explicit Python parameter names for the py2js kwargs bridge. py2js maps
// Python keyword arguments onto positional slots via `Function.toString`
// parameter names, but esbuild renames colliding locals when bundling, so the
// names are pinned here (see `arrange_in_grid` for the same pattern).
(PathSlider as unknown as { __argnames__?: string[] }).__argnames__ = ["path", "options"];
// `Slider` takes a single options bag (its track is derived from `start`/`end`),
// so it must pin its *own* names: a class static `__argnames__` is inherited by
// subclasses, and without this override the bridge would read PathSlider's
// ["path", "options"], map every Slider kwarg into the (nonexistent) options
// slot of that shape and construct `new Slider(undefined, extras)`, dropping
// `value_range`/`init_value`/... entirely.
(Slider as unknown as { __argnames__?: string[] }).__argnames__ = ["options"];
(Chip as unknown as { __argnames__?: string[] }).__argnames__ = ["label", "options"];
(InputDecimalNumber as unknown as { __argnames__?: string[] }).__argnames__ = ["numberValue", "options"];
(InputInteger as unknown as { __argnames__?: string[] }).__argnames__ = ["numberValue", "options"];
(InputDecimalMatrix as unknown as { __argnames__?: string[] }).__argnames__ = ["matrix", "options"];
(InputDecimalTable as unknown as { __argnames__?: string[] }).__argnames__ = ["table", "options"];
