/**
 * Interactive controls: `Slider`, `Switch` and `InputDecimalNumber`.
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
import { BLUE_E, GREY_C, WHITE, YELLOW } from "../utils/color/index.js";
import { Mobject } from "../mobject/mobject.js";
import { VGroup } from "../mobject/types/vectorized_mobject.js";
import { Dot } from "../mobject/geometry/arc.js";
import { Line } from "../mobject/geometry/line.js";
import { Rectangle } from "../mobject/geometry/polygram.js";
import { DecimalNumber, type DecimalNumberOptions, type DecimalNumberValue } from "../mobject/text/numbers.js";
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
}

/** Duck-typed check that `mobject` is an interactive control. */
export function isInteractiveControl(value: unknown): value is Mobject & InteractiveControl {
  return Boolean(value) && (value as { [INTERACTIVE_CONTROL]?: unknown })[INTERACTIVE_CONTROL] === true;
}

function point2d(point: readonly number[]): [number, number] {
  return [point[0] ?? 0, point[1] ?? 0];
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

export interface SliderOptions {
  start?: readonly number[];
  end?: readonly number[];
  value_range?: readonly number[];
  init_value?: number;
  color?: ParsableManimColor;
  stroke_width?: number;
  /** Opacity of the track line (0 = invisible). Defaults to 1. */
  track_opacity?: number;
  /** Opacity of the knob (0 = invisible). Defaults to 1. */
  knob_opacity?: number;
  /** Knob radius (world units). */
  knob_radius?: number;
  /** Extra pick radius around the track (world units). */
  hit_radius?: number;
}

/**
 * A draggable slider drawn as a line with a dot knob.
 *
 * `value_range` is inclusive: dragging to `start` yields the first element, to
 * `end` the last. `init_value` (defaulting to the first element) sets the knob
 * position at construction time.
 *
 * `attach(mobject)` parents a mobject to the knob so it follows the value;
 * while anything is attached the knob itself is hidden, so the attached mobject
 * is what the user sees (use `unattach` to restore the knob).
 */
export class Slider extends VGroup implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  start_point: number[];
  end_point: number[];
  value_range: [number, number];
  accent_color: ParsableManimColor;
  stroke_width: number;
  track_opacity: number;
  knob_opacity: number;
  knob_radius: number;
  hit_radius: number;
  track: Line;
  knob: Dot;
  /** Mobjects currently parented to the knob via {@link attach}. */
  attached: Mobject[] = [];
  private _value: number;

  constructor(options: SliderOptions = {}) {
    super();
    const {
      start = [-1, 0, 0],
      end = [1, 0, 0],
      value_range = [0, 1],
      init_value,
      color = YELLOW,
      stroke_width = 6,
      track_opacity = 1,
      knob_opacity = 1,
      knob_radius = 0.12,
      hit_radius = 0.3,
    } = options;
    this.start_point = [start[0] ?? 0, start[1] ?? 0, start[2] ?? 0];
    this.end_point = [end[0] ?? 0, end[1] ?? 0, end[2] ?? 0];
    const low = value_range[0] ?? 0;
    const high = value_range[1] ?? 1;
    this.value_range = [low, high];
    this.accent_color = color;
    this.stroke_width = stroke_width;
    this.track_opacity = track_opacity;
    this.knob_opacity = knob_opacity;
    this.knob_radius = knob_radius;
    this.hit_radius = hit_radius;
    this._value = init_value ?? low;

    this.track = new Line({
      start: this.start_point,
      end: this.end_point,
      color,
      stroke_width,
    });
    this.track.set_stroke(color, stroke_width, track_opacity);
    this.knob = new Dot({
      point: this._pointForValue(this._value),
      radius: knob_radius,
      color,
      fill_opacity: knob_opacity,
      stroke_width: 0,
    });
    this.add(this.track, this.knob);
  }

  get_value(): number {
    return this._value;
  }

  set_value(value: number): this {
    const [low, high] = this.value_range;
    const clamped = Math.max(low, Math.min(high, value));
    this._value = clamped;
    const [x, y, z] = this._pointForValue(clamped);
    this.knob.move_to([x, y, z]);
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

  private _update_knob_visibility(): void {
    const opacity = this.attached.length > 0 ? 0 : this.knob_opacity;
    // `family = false`: only the knob's own points change, never the attached
    // mobjects' styles.
    this.knob.set_opacity(opacity, false);
  }

  is_hit(point: readonly number[]): boolean {
    return distanceToSegment(point, this.start_point, this.end_point) <= this.hit_radius;
  }

  on_press(point: readonly number[], _context: InteractiveContext): void {
    this._setFromPoint(point);
  }

  on_drag(point: readonly number[], _context: InteractiveContext): void {
    this._setFromPoint(point);
  }

  on_release(): void {}

  private _setFromPoint(point: readonly number[]): void {
    const [px, py] = point2d(point);
    const [ax, ay] = point2d(this.start_point);
    const [bx, by] = point2d(this.end_point);
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSq = dx * dx + dy * dy;
    let t = lengthSq === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
    const [low, high] = this.value_range;
    this.set_value(low + t * (high - low));
  }

  private _fraction(value: number): number {
    const [low, high] = this.value_range;
    if (high === low) return 0;
    return Math.max(0, Math.min(1, (value - low) / (high - low)));
  }

  private _pointForValue(value: number): [number, number, number] {
    const t = this._fraction(value);
    return [
      (this.start_point[0] ?? 0) + t * ((this.end_point[0] ?? 0) - (this.start_point[0] ?? 0)),
      (this.start_point[1] ?? 0) + t * ((this.end_point[1] ?? 0) - (this.start_point[1] ?? 0)),
      (this.start_point[2] ?? 0) + t * ((this.end_point[2] ?? 0) - (this.start_point[2] ?? 0)),
    ];
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

/**
 * A `DecimalNumber` whose value can be edited by clicking it (the host opens an
 * inline text editor anchored over the number).
 */
export class InputDecimalNumber extends DecimalNumber implements InteractiveControl {
  readonly [INTERACTIVE_CONTROL] = true;
  editing = false;

  constructor(numberValue: DecimalNumberValue = 0, options: DecimalNumberOptions = {}) {
    // Grouping separators (10,000) make the caret/glyph mapping confusing for a
    // live-edited number, so the interactive input defaults to no grouping.
    super(numberValue, { group_with_commas: false, ...options });
  }

  override get_value(): DecimalNumberValue {
    return super.get_value();
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

  on_press(_point: readonly number[], context: InteractiveContext): void {
    this.editing = true;
    context.host.beginTextEdit(this, context.camera, {
      initial: this._format_value(),
      onInput: (text) => this._apply_text(text),
      onCommit: (text) => {
        this._apply_text(text);
        this.editing = false;
      },
      onCancel: () => {
        this.editing = false;
      },
    });
  }

  on_drag(_point: readonly number[], _context: InteractiveContext): void {}

  on_release(): void {}

  private _apply_text(text: string): void {
    const value = Number.parseFloat(text);
    if (Number.isFinite(value)) this.set_value(value);
  }

  private _format_value(): string {
    const value = this.get_value();
    const scalar = typeof value === "number" ? value : value.re;
    return scalar.toFixed(this.num_decimal_places);
  }

  /**
   * World-space anchor for the character insertion point at `index` (0 = before
   * the first glyph, `glyphs` = after the last). Used by the browser host to
   * draw the editing marker under the digit being edited. Returns `null` when
   * the number has no glyphs yet.
   */
  interactive_caret_anchor(index: number): number[] | null {
    // `_set_submobjects_from_number` adds one submobject per displayed
    // character (in order); unit/background submobjects, if any, come after.
    const display = this._get_num_string(this.number);
    const glyphs = this.submobjects.slice(0, display.length);
    if (glyphs.length === 0) return null;
    const bottom = this.get_bottom();
    const y = bottom[1] ?? 0;
    if (index >= glyphs.length) {
      const right = glyphs[glyphs.length - 1]!.get_right();
      return [right[0] ?? 0, y, 0];
    }
    const left = glyphs[Math.max(0, index)]!.get_left();
    return [left[0] ?? 0, y, 0];
  }
}
