/**
 * Interactive controls: `PathSlider` (and its `Slider` line subclass),
 * `DraggablePoint`, `Switch`, `Chip`, `InputDecimalNumber` (plus
 * `InputInteger`) and the editable `InputDecimalMatrix` / `InputDecimalTable`
 * containers.
 *
 * Each control is a normal `VMobject` (so it can be added to a scene, animated
 * and moved like any other mobject) that additionally implements the
 * {@link InteractiveControl} protocol used by `InteractiveScene`:
 *
 *  - `is_hit(point)` does hit testing in world coordinates;
 *  - `on_press` / `on_drag` / `on_release` mutate the control's value;
 *  - `get_value()` reads it back, so other mobjects can react to it in a
 *    blueprint `add_reaction` callback;
 *  - `on_input(listener)` observes user-originated value changes (the
 *    programmatic `set_value` path stays silent; see {@link InputListener}).
 *
 * The controls deliberately keep their interaction geometry (a slider's track,
 * a switch's bounding box) as world-space data captured at construction time:
 * they are meant to be `self.add(...)`-ed directly, not nested in groups that
 * move afterwards.
 */
import { array, type NDArray } from "numpy-ts";
import type { ParsableManimColor } from "../utils/color/index.js";
import { BLACK, BLUE_E, GREY_B, GREY_C, WHITE, YELLOW, interpolate_color, ManimColor } from "../utils/color/index.js";
import { DEFAULT_DOT_RADIUS, LEFT, RIGHT } from "../constants.js";
import { Mobject } from "../mobject/mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import { Dot, type DotOptions } from "../mobject/geometry/arc.js";
import { Line } from "../mobject/geometry/line.js";
import { isPlainOptionsBag, Rectangle, RoundedRectangle } from "../mobject/geometry/polygram.js";
import { pointRows, type ComplexPoint } from "../utils/arrays.js";
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
   * implement it; the scene ignores controls that do not. The world-space
   * pointer position is passed along when available, so a control can restrict
   * the highlight to a sub-part (e.g. a slider's knob).
   */
  set_hovered?(hovered: boolean, point?: readonly number[]): void;
  /**
   * Optional: observe user-originated value changes (see {@link InputListener}).
   * `InteractiveScene` subscribes to every control it finds so that a
   * blueprint's reactions are re-applied when the reader changes something —
   * including edits typed into an inline editor, which never reach the scene's
   * pointer handlers.
   */
  on_input?(listener: InputListener): unknown;
  /**
   * Optional: fire the `on_input` listeners for a value written by the player.
   * Called once per control when the player restores remembered values
   * (previous/replay rewinds, including pages it fast-forwards through), so
   * courseware that mirrors control values into its own state can rebuild that
   * state from the restored controls.  It fires even when the value is
   * unchanged, unlike a user gesture; `set_value` itself stays silent.
   */
  notify_input?(previous: unknown, next: unknown): void;
}

/** Duck-typed check that `mobject` is an interactive control. */
export function isInteractiveControl(value: unknown): value is Mobject & InteractiveControl {
  return Boolean(value) && (value as { [INTERACTIVE_CONTROL]?: unknown })[INTERACTIVE_CONTROL] === true;
}

/**
 * Callback registered through `on_input`.
 *
 * Receives the control itself, the value before the change and the value after
 * it.  User gestures fire it; `set_value` (or `toggle`) called programmatically
 * stays silent, so writing control values back from a blueprint reaction never
 * echoes into the listener.  The one programmatic exception is the player
 * restoring remembered values on "previous"/replay: every restored control
 * fires its listeners once (through {@link InteractiveControl.notify_input},
 * even for a value equal to the current one), so state derived from controls
 * is rebuilt after a rewind.  A listener should update shared state only;
 * writing to other controls belongs in a scene reaction (which then runs after
 * all input listeners for the frame).
 *
 * NOTE (sync timing): listeners run synchronously inside the input path
 * (`on_press` / `on_drag` / the inline editor's input callback), i.e. outside
 * the scene's frame tick.  Browser hosts dispatch pointer and editor events
 * between animation frames, so a listener always completes before the next
 * frame's reactions run; there is deliberately no frame-batched dispatch.
 * Should a future host invoke control handlers from inside a reaction, the
 * listener would run mid-tick instead of before it.  Restored-value
 * notifications run inside the player's rewind, after every restored control
 * has its new value and before that page's reaction pass.
 */
export type InputListener = (control: Mobject, previous: unknown, next: unknown) => void;

/** Whether a user gesture produced a real value change. */
function inputValueChanged(previous: unknown, next: unknown): boolean {
  if (previous === next) return false;
  if (
    typeof previous === "object" &&
    previous !== null &&
    typeof next === "object" &&
    next !== null
  ) {
    const before = previous as Partial<ComplexPoint>;
    const after = next as Partial<ComplexPoint>;
    if (typeof before.re === "number" && typeof after.re === "number") {
      return before.re !== after.re || (before.im ?? 0) !== (after.im ?? 0);
    }
  }
  return true;
}

/** Fire `listeners` for a user-originated value change (no-op when unchanged). */
function emitInput(
  control: Mobject,
  listeners: readonly InputListener[],
  previous: unknown,
  next: unknown,
): void {
  if (listeners.length === 0 || !inputValueChanged(previous, next)) return;
  notifyInput(control, listeners, previous, next);
}

/**
 * Fire `listeners` unconditionally, for a value the player restored.  Unlike
 * {@link emitInput} this ignores "unchanged": every restored control is
 * announced once so state mirrored from controls is rebuilt.
 */
function notifyInput(
  control: Mobject,
  listeners: readonly InputListener[],
  previous: unknown,
  next: unknown,
): void {
  for (const listener of listeners) listener(control, previous, next);
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
  /**
   * Scale factor applied to the knob while the pointer hovers it or the slider
   * is being dragged. Ignored while a mobject is attached (the knob is hidden
   * then). Defaults to 1.25.
   */
  knob_active_scale?: number;
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
  knob_active_scale: number;
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
  /** Listeners registered through {@link PathSlider.on_input}. */
  input_listeners: InputListener[] = [];
  /** Whether the pointer currently hovers the knob itself (not the track). */
  hovered = false;
  private _value: number;
  /** Arc-length proportion along the path; the source of truth for the knob. */
  private _t = 0;
  private _lastDragPoint: number[] | null = null;
  /** Whether a press on this slider is in progress. */
  private _grabbing = false;
  /** Scale factor currently applied to the knob (1 when at rest). */
  private _knob_scale = 1;

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
      knob_active_scale = 1.25,
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
    this.knob_active_scale = knob_active_scale > 0 ? knob_active_scale : 1;
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
   * Register a listener for value changes: user gestures, plus the player
   * restoring remembered values on a rewind. See {@link InputListener}.
   */
  on_input(listener: InputListener): this {
    this.input_listeners.push(listener);
    return this;
  }

  /** Fire the listeners for a restored value (see {@link InteractiveControl}). */
  notify_input(previous: unknown, next: unknown): void {
    notifyInput(this, this.input_listeners, previous, next);
  }

  /**
   * Parent `mobject` to the knob so it moves with the slider value. The knob
   * becomes transparent so only the attached mobject is visible.
   */
  attach(mobject: Mobject): this {
    if (this.attached.includes(mobject)) return this;
    // Scale the knob back to its resting size before adopting the child:
    // `scale` walks the whole family, so a still-grown knob would resize the
    // attached mobject too.
    this._set_knob_scale(1);
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
    this._apply_knob_scale();
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
    return this._hit_knob(point);
  }

  on_press(point: readonly number[], _context: InteractiveContext): void {
    this._grabbing = true;
    this._apply_knob_scale();
    const [low, high] = this.value_range;
    const previous = this._value;
    const t = this._proportion_from_point(point);
    this.set_value(low + t * (high - low));
    emitInput(this, this.input_listeners, previous, this._value);
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
      const previous = this._value;
      const [tangentX, tangentY] = this._tangent_at(this._t);
      this._set_proportion(this._t + (dx * tangentX + dy * tangentY) / total);
      emitInput(this, this.input_listeners, previous, this._value);
    }
    this._lastDragPoint = [point[0] ?? 0, point[1] ?? 0, point[2] ?? 0];
  }

  on_release(): void {
    this._lastDragPoint = null;
    this._grabbing = false;
    this._apply_knob_scale();
  }

  /**
   * Update the knob's hover state (called by the scene on pointer motion).
   * The knob only grows while the pointer is within its pick radius; hovering
   * elsewhere on the track leaves it at its resting size.
   */
  set_hovered(hovered: boolean, point?: readonly number[]): void {
    const next = Boolean(hovered) && point !== undefined && this._hit_knob(point);
    if (this.hovered === next) return;
    this.hovered = next;
    this._apply_knob_scale();
  }

  /** Whether `point` falls within the knob's pick radius. */
  private _hit_knob(point: readonly number[]): boolean {
    const center = pointRows(this.knob.get_center(), 3)[0] ?? [0, 0, 0];
    const radius = Math.max(this.hit_radius, this.knob_radius * 2);
    return Math.hypot((point[0] ?? 0) - (center[0] ?? 0), (point[1] ?? 0) - (center[1] ?? 0)) <= radius;
  }

  /**
   * Grow the knob while the pointer hovers it or the slider is being dragged.
   * No-op while a mobject is attached, so the attached mobject (the visible
   * knob then) is never resized.
   */
  private _apply_knob_scale(): void {
    const active = this.attached.length === 0 && (this._grabbing || this.hovered);
    this._set_knob_scale(active ? this.knob_active_scale : 1);
  }

  /** Scale the knob (about its own centre) to `target` times its base size. */
  private _set_knob_scale(target: number): void {
    if (target === this._knob_scale) return;
    this.knob.scale(target / this._knob_scale, this.knob.get_center());
    this._knob_scale = target;
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

export interface DraggablePointOptions extends DotOptions {
  /**
   * Pick radius around the dot's centre, in world units. Defaults to
   * `max(0.3, 2 * radius)` so small dots remain easy to grab.
   */
  hit_radius?: number;
  /**
   * Scale factor applied to the dot while the pointer hovers it or it is being
   * dragged. Defaults to 1.25. Ignored while a mobject is attached (the dot is
   * then the anchor of the attached mobject and must not resize it).
   */
  active_scale?: number;
  /**
   * Optional grid snap: a number (both axes) or `[x_step, y_step]`. The
   * dragged position is rounded to the nearest multiple of the step(s).
   * `null` (the default) keeps the point continuous.
   */
  snap?: number | readonly number[] | null;
}

/**
 * A free point on the plane that the reader can drag anywhere.
 *
 * It is an ordinary {@link Dot} (same construction as `Dot(point, **kwargs)`)
 * that also implements the {@link InteractiveControl} protocol:
 *
 * ```python
 * p = DraggablePoint([1, 2, 0], color=YELLOW)
 * self.add(p)
 * readout.add_reaction(lambda m: m.set_value(p.get_center()[0]))
 * ```
 *
 * `get_center()` is the dot's current position (an ordinary manim call, just
 * like on any other mobject, and exact even when a label is attached);
 * `set_value(...)` / `move_to(...)` move it from code without firing
 * `on_input`. `attach(mobject)` makes a mobject follow the point, mirroring
 * `Slider.attach`.
 *
 * Unlike a slider the point has no track, so hit testing is a circle around
 * its current centre: put the control directly on the stage instead of inside
 * a group that later moves or scales.
 */
export class DraggablePoint extends Dot implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  /** Pick radius around the dot's centre (world units). */
  hit_radius: number;
  /** Hover/drag growth factor (`1` disables it). */
  active_scale: number;
  /** Optional grid snap (see {@link DraggablePointOptions.snap}). */
  snap: number | [number, number] | null;
  /** Mobjects currently attached with {@link attach}. */
  attached: Mobject[] = [];
  /** Listeners registered through {@link DraggablePoint.on_input}. */
  input_listeners: InputListener[] = [];
  /** Whether the pointer currently hovers the dot. */
  hovered = false;
  private _base_radius: number;
  /** Exact position of the dot's centre (kept in sync with `shift`). */
  private _point!: [number, number, number];
  private _grabbing = false;
  /** Pointer-to-centre offset captured on press, so a drag never jumps. */
  private _grab_offset: [number, number] = [0, 0];
  /** Scale factor currently applied to the dot (1 when at rest). */
  private _visual_scale = 1;

  constructor(
    point: readonly number[] | NDArray = [0, 0, 0],
    options: DraggablePointOptions = {},
  ) {
    const {
      hit_radius = -1,
      active_scale = 1.25,
      snap = null,
      radius = DEFAULT_DOT_RADIUS,
      ...dotOptions
    } = options;
    const dotRadius =
      typeof radius === "number" && Number.isFinite(radius) && radius > 0
        ? radius
        : DEFAULT_DOT_RADIUS;
    super({ point, radius: dotRadius, ...dotOptions });
    const initial = pointRows(point, 3)[0] ?? [0, 0, 0];
    this._point = [initial[0] ?? 0, initial[1] ?? 0, initial[2] ?? 0];
    this._base_radius = dotRadius;
    this.hit_radius = hit_radius > 0 ? hit_radius : Math.max(0.3, 2 * dotRadius);
    this.active_scale = active_scale > 0 ? active_scale : 1;
    this.snap = normalizeSnap(snap);
  }

  /**
   * The dot's current position. Returns exactly the dragged point even when
   * mobjects are attached (their offsets never shift the reported centre).
   */
  override get_center(): NDArray {
    // `super(...)` generates the circle's points before `_point` exists.
    if (!("_point" in (this as object))) return super.get_center();
    return array([...this._point]) as NDArray;
  }

  /** The control protocol value: the same position as {@link get_center}. */
  get_value(): NDArray {
    return this.get_center();
  }

  /**
   * Move the point from code (silent: does not fire `on_input`). `move_to` and
   * `.shift(...)` also keep the reported centre in sync.
   */
  set_value(point: readonly number[] | NDArray): this {
    const row = pointRows(point, 3)[0] ?? [0, 0, 0];
    const [x, y] = this._snap_xy(row[0] ?? 0, row[1] ?? 0);
    this._move_center_to([x, y, row[2] ?? 0]);
    return this;
  }

  /**
   * Register a listener for user-originated position changes (dragging), plus
   * the player restoring remembered values on a rewind. See
   * {@link InputListener}.
   */
  on_input(listener: InputListener): this {
    this.input_listeners.push(listener);
    return this;
  }

  /** Fire the listeners for a restored value (see {@link InteractiveControl}). */
  notify_input(previous: unknown, next: unknown): void {
    notifyInput(this, this.input_listeners, previous, next);
  }

  /**
   * Parent `mobject` to the point so it follows every drag. Attach labels after
   * positioning them; the point's centre stays the anchor.
   */
  attach(mobject: Mobject): this {
    if (this.attached.includes(mobject)) return this;
    // Reset any hover growth first: `scale` walks the family, so a grown dot
    // would resize the newly attached mobject too.
    this._set_visual_scale(1);
    this.attached.push(mobject);
    this.add(mobject);
    return this;
  }

  /** Detach one mobject, or every attached mobject when called with no argument. */
  unattach(mobject?: Mobject): this {
    if (mobject === undefined) {
      for (const attached of [...this.attached]) this.remove(attached);
      this.attached = [];
    } else {
      if (!this.attached.includes(mobject)) return this;
      this.remove(mobject);
      this.attached = this.attached.filter((candidate) => candidate !== mobject);
    }
    this._apply_hover_scale();
    return this;
  }

  is_hit(point: readonly number[]): boolean {
    const radius = Math.max(this.hit_radius, 2 * this._base_radius * this._visual_scale);
    return Math.hypot((point[0] ?? 0) - this._point[0], (point[1] ?? 0) - this._point[1]) <= radius;
  }

  on_press(point: readonly number[], _context: InteractiveContext): void {
    this._grabbing = true;
    this._apply_hover_scale();
    this._grab_offset = [this._point[0] - (point[0] ?? 0), this._point[1] - (point[1] ?? 0)];
  }

  on_drag(point: readonly number[], context: InteractiveContext): void {
    if (!this._grabbing) {
      // A drag without a preceding press (hosts always send one, but custom
      // hosts may not): fall back to a normal press so the offset is captured.
      this.on_press(point, context);
      return;
    }
    const previous: [number, number, number] = [...this._point];
    const [x, y] = this._snap_xy(
      (point[0] ?? 0) + this._grab_offset[0],
      (point[1] ?? 0) + this._grab_offset[1],
    );
    this._move_center_to([x, y, this._point[2]]);
    const next = this._point;
    if (previous[0] !== next[0] || previous[1] !== next[1]) {
      notifyInput(this, this.input_listeners, array(previous) as NDArray, array([...next]) as NDArray);
    }
  }

  /**
   * Keep the tracked position in sync with any external positioning
   * (`move_to`, `to_corner`, `.shift(...)`). `shift` is the primitive behind
   * all of them, so overriding it here is enough for ordinary scene code.
   */
  override shift(...vectors: Array<readonly number[] | NDArray>): this {
    // `super(...)` builds the dot's points, and the Arc generator shifts them
    // before this class's field initialisers have run. Mobject proxies throw
    // on unknown *reads*, so probe with `in` instead of comparing to undefined.
    if (!("_point" in (this as object))) return super.shift(...vectors);
    let dx = 0;
    let dy = 0;
    let dz = 0;
    for (const value of vectors) {
      const row = pointRows(value, 3)[0] ?? [0, 0, 0];
      dx += row[0] ?? 0;
      dy += row[1] ?? 0;
      dz += row[2] ?? 0;
    }
    const result = super.shift(...vectors);
    if (dx !== 0 || dy !== 0 || dz !== 0) {
      this._point = [this._point[0] + dx, this._point[1] + dy, this._point[2] + dz];
    }
    return result;
  }

  on_release(): void {
    this._grabbing = false;
    this._apply_hover_scale();
  }

  /** Update the hover highlight (called by the scene on pointer motion). */
  set_hovered(hovered: boolean): void {
    const next = Boolean(hovered);
    if (this.hovered === next) return;
    this.hovered = next;
    this._apply_hover_scale();
  }

  /** Shift the whole family so the dot's own centre lands on `target`. */
  private _move_center_to(target: readonly number[]): void {
    this.shift([
      (target[0] ?? 0) - this._point[0],
      (target[1] ?? 0) - this._point[1],
      (target[2] ?? 0) - this._point[2],
    ]);
  }

  private _snap_xy(x: number, y: number): [number, number] {
    const snap = this.snap;
    if (snap === null) return [x, y];
    if (Array.isArray(snap)) {
      const [stepX, stepY] = snap as [number, number];
      return [stepX > 0 ? Math.round(x / stepX) * stepX : x, stepY > 0 ? Math.round(y / stepY) * stepY : y];
    }
    const step = snap as number;
    return step > 0 ? [Math.round(x / step) * step, Math.round(y / step) * step] : [x, y];
  }

  private _apply_hover_scale(): void {
    const active = this.attached.length === 0 && (this._grabbing || this.hovered);
    this._set_visual_scale(active ? this.active_scale : 1);
  }

  private _set_visual_scale(target: number): void {
    if (target === this._visual_scale) return;
    this.scale(target / this._visual_scale, [...this._point]);
    this._visual_scale = target;
  }
}

/** Normalize a snap option (`null`, scalar or `[x, y]`) to its stored form. */
function normalizeSnap(value: number | readonly number[] | null): number | [number, number] | null {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value)) {
    const stepX = Number((value as readonly number[])[0] ?? 0);
    const stepY = Number((value as readonly number[])[1] ?? stepX);
    if (stepX <= 0 && stepY <= 0) return null;
    return [stepX > 0 ? stepX : 0, stepY > 0 ? stepY : 0];
  }
  const step = Number(value);
  return step > 0 ? step : null;
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
  /** Listeners registered through {@link Switch.on_input}. */
  input_listeners: InputListener[] = [];
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

  /**
   * Register a listener for value changes: user gestures, plus the player
   * restoring remembered values on a rewind. See {@link InputListener}.
   */
  on_input(listener: InputListener): this {
    this.input_listeners.push(listener);
    return this;
  }

  /** Fire the listeners for a restored value (see {@link InteractiveControl}). */
  notify_input(previous: unknown, next: unknown): void {
    notifyInput(this, this.input_listeners, previous, next);
  }

  is_hit(point: readonly number[]): boolean {
    const left = pointRows(this.track.get_left(), 3)[0] ?? [0, 0, 0];
    const right = pointRows(this.track.get_right(), 3)[0] ?? [0, 0, 0];
    const bottom = pointRows(this.track.get_bottom(), 3)[0] ?? [0, 0, 0];
    const top = pointRows(this.track.get_top(), 3)[0] ?? [0, 0, 0];
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
    const previous = this._value;
    this.toggle();
    emitInput(this, this.input_listeners, previous, this._value);
  }

  on_drag(_point: readonly number[], _context: InteractiveContext): void {}

  on_release(): void {}

  private _apply_state(): void {
    // Off: white track, grey knob (left). On: blue track, white knob (right).
    const on = this._value;
    this.track.set_fill(on ? this.accent_color : GREY_C, 1);
    this.knob.set_fill(WHITE, 1);
    const center = pointRows(this.track.get_center(), 3)[0] ?? [0, 0, 0];
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
  /** Listeners registered through {@link Chip.on_input}. */
  input_listeners: InputListener[] = [];
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

  /**
   * Register a listener for value changes: user gestures, plus the player
   * restoring remembered values on a rewind. See {@link InputListener}.
   */
  on_input(listener: InputListener): this {
    this.input_listeners.push(listener);
    return this;
  }

  /** Fire the listeners for a restored value (see {@link InteractiveControl}). */
  notify_input(previous: unknown, next: unknown): void {
    notifyInput(this, this.input_listeners, previous, next);
  }

  is_hit(point: readonly number[]): boolean {
    const left = pointRows(this.box.get_left(), 3)[0] ?? [0, 0, 0];
    const right = pointRows(this.box.get_right(), 3)[0] ?? [0, 0, 0];
    const bottom = pointRows(this.box.get_bottom(), 3)[0] ?? [0, 0, 0];
    const top = pointRows(this.box.get_top(), 3)[0] ?? [0, 0, 0];
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
    const previous = this._value;
    this.toggle();
    emitInput(this, this.input_listeners, previous, this._value);
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
  /** Listeners registered through {@link InputDecimalNumber.on_input}. */
  input_listeners: InputListener[] = [];
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
   * Register a listener for value changes: every keystroke while editing, every
   * scrub step while dragging, the commit, and the player restoring remembered
   * values on a rewind. See {@link InputListener} for the callback contract.
   */
  on_input(listener: InputListener): this {
    this.input_listeners.push(listener);
    return this;
  }

  /** Fire the listeners for a restored value (see {@link InteractiveControl}). */
  notify_input(previous: unknown, next: unknown): void {
    notifyInput(this, this.input_listeners, previous, next);
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
    const left = pointRows(this.get_left(), 3)[0] ?? [0, 0, 0];
    const right = pointRows(this.get_right(), 3)[0] ?? [0, 0, 0];
    const bottom = pointRows(this.get_bottom(), 3)[0] ?? [0, 0, 0];
    const top = pointRows(this.get_top(), 3)[0] ?? [0, 0, 0];
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
    const previous = this.get_value();
    this.set_value(clamped);
    emitInput(this, this.input_listeners, previous, this.get_value());
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
      const previous = this.get_value();
      this.set_value(value);
      emitInput(this, this.input_listeners, previous, this.get_value());
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
    const bottom = pointRows(this.get_bottom(), 3)[0] ?? [0, 0, 0];
    const y = bottom[1] ?? 0;
    if (glyphIndex >= glyphs.length) {
      const right = pointRows(glyphs[glyphs.length - 1]!.get_right(), 3)[0] ?? [0, 0, 0];
      return [right[0] ?? 0, y, 0];
    }
    const left = pointRows(glyphs[Math.max(0, glyphIndex)]!.get_left(), 3)[0] ?? [0, 0, 0];
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
    const left = pointRows(this.get_left(), 3)[0]?.[0] ?? 0;
    const right = pointRows(this.get_right(), 3)[0]?.[0] ?? 0;
    const bottom = pointRows(this.get_bottom(), 3)[0]?.[1] ?? 0;
    const top = pointRows(this.get_top(), 3)[0]?.[1] ?? 0;
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

/** Register `relayout` as an input listener of every interactive cell. */
function attachRelayout(cells: VMobject[][], relayout: () => void): void {
  for (const row of cells) {
    for (const cell of row) {
      if (!isInteractiveControl(cell)) continue;
      const interactive = cell as unknown as { on_input?: (listener: InputListener) => unknown };
      interactive.on_input?.(relayout);
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
    const newCenter = pointRows(this.get_center(), this.dim)[0] ?? [0, 0, 0];
    const savedCenterRow = pointRows(savedCenter, this.dim)[0] ?? [0, 0, 0];
    this.shift([
      (savedCenterRow[0] ?? 0) - (newCenter[0] ?? 0),
      (savedCenterRow[1] ?? 0) - (newCenter[1] ?? 0),
      (savedCenterRow[2] ?? 0) - (newCenter[2] ?? 0),
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
    const newCenter = pointRows(this.get_center(), this.dim)[0] ?? [0, 0, 0];
    const savedCenterRow = pointRows(savedCenter, this.dim)[0] ?? [0, 0, 0];
    this.shift([
      (savedCenterRow[0] ?? 0) - (newCenter[0] ?? 0),
      (savedCenterRow[1] ?? 0) - (newCenter[1] ?? 0),
      (savedCenterRow[2] ?? 0) - (newCenter[2] ?? 0),
    ]);
    return this;
  }
}

// Explicit Python parameter names for the py2js kwargs bridge. py2js maps
// Python keyword arguments onto positional slots via `Function.toString`
// parameter names, but esbuild renames colliding locals when bundling, so the
// names are pinned here (see `arrange_in_grid` for the same pattern).
(DraggablePoint as unknown as { __argnames__?: string[] }).__argnames__ = ["point", "options"];
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
