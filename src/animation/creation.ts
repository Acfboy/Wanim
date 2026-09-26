import { Animation, classDefaultsOf, type AnimationOptions } from "./animation.js";
import { Succession } from "./composition.js";
import { Mobject, Group } from "../mobject/mobject.js";
import { VMobject, VGroup } from "../mobject/types/vectorized_mobject.js";
import type { ParsableManimColor } from "../utils/color/core.js";
import { config } from "../_config/utils.js";
import { RIGHT, TAU } from "../constants.js";
import { integer_interpolate } from "../utils/bezier.js";
import { double_smooth, linear } from "../utils/rate_functions.js";
import { pointRows, addPoints, scalePoint, subtractPoints } from "../utils/arrays.js";

export interface ShowPartialOptions extends AnimationOptions {
  mobject?: VMobject;
}

export class ShowPartial extends Animation {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    // NOTE: mirrors Python `getattr(mobject, "pointwise_become_partial",
    // None)`: a missing attribute yields None (then TypeError). Mobjects in
    // this port are Proxies that raise AttributeError on unknown property
    // access, so the lookup is guarded the same way getattr's default is.
    let pointwise: unknown;
    try {
      pointwise = (mobject as unknown as { pointwise_become_partial?: unknown })?.pointwise_become_partial;
    } catch {
      pointwise = undefined;
    }
    if (typeof pointwise !== "function") {
      const constructorName = new.target.name;
      throw new TypeError(`${constructorName} only works for VMobjects.`);
    }
    super(mobject, options);
  }

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): void {
    const mobject = submobject as VMobject;
    const [a, b] = this._get_bounds(alpha);
    mobject.pointwise_become_partial(startingSubmobject as VMobject, a, b);
  }

  _get_bounds(_alpha: number): [number, number] {
    const error = new Error("Please use Create or ShowPassingFlash");
    error.name = "NotImplementedError";
    throw error;
  }
}

export class Create extends ShowPartial {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    super(mobject, { lag_ratio: 1, introducer: true, ...options });
  }

  _get_bounds(alpha: number): [number, number] {
    return [0, alpha];
  }
}

export class Uncreate extends Create {
  constructor(mobject: VMobject, options: ShowPartialOptions = {}) {
    super(mobject, { reverse_rate_function: true, introducer: false, remover: true, ...options });
  }
}

export class ShowIncreasingSubsets extends Animation {
  all_submobs: Mobject[];
  int_func: (value: number) => number;

  constructor(group: Mobject, options: AnimationOptions & { int_func?: (value: number) => number } = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const { int_func, suspend_mobject_updating, reverse_rate_function, ...rest } = options;
    const allSubmobs = [...group.submobjects];
    const resolvedIntFunc = (int_func as ((value: number) => number) | undefined) ?? (clsDefaults.int_func as ((value: number) => number) | undefined) ?? Math.floor;
    for (const mobject of allSubmobs) mobject.set_opacity(0);
    super(group, {
      ...rest,
      suspend_mobject_updating: (suspend_mobject_updating as boolean | undefined) ?? (clsDefaults.suspend_mobject_updating as boolean | undefined) ?? false,
      reverse_rate_function: (reverse_rate_function as boolean | undefined) ?? (clsDefaults.reverse_rate_function as boolean | undefined) ?? false,
    });
    this.all_submobs = allSubmobs;
    this.int_func = resolvedIntFunc;
  }

  interpolate_mobject(alpha: number): void {
    const nSubmobs = this.all_submobs.length;
    const value = this.reverse_rate_function ? 1 - this.rate_func(alpha) : this.rate_func(alpha);
    const index = Math.trunc(Number(this.int_func(value * nSubmobs)));
    this.update_submobject_list(index);
  }

  update_submobject_list(index: number): void {
    if (!Number.isInteger(index)) {
      throw new TypeError("slice indices must be integers or None or have an __index__ method");
    }
    for (const mobject of this.all_submobs.slice(0, index)) mobject.set_opacity(1);
    for (const mobject of this.all_submobs.slice(index)) mobject.set_opacity(0);
  }
}

export class ShowSubmobjectsOneByOne extends ShowIncreasingSubsets {
  constructor(group: Iterable<Mobject>, options: AnimationOptions & { int_func?: (value: number) => number } = {}) {
    const { int_func = Math.ceil, ...rest } = options;
    const newGroup = new Group(...group);
    super(newGroup, { int_func, ...rest });
  }

  update_submobject_list(index: number): void {
    const current = this.all_submobs.slice(0, index);
    for (const mobject of current.slice(0, -1)) mobject.set_opacity(0);
    if (current.length > 0) current[current.length - 1].set_opacity(1);
  }
}

export interface AddTextLetterByLetterOptions extends AnimationOptions {
  suspend_mobject_updating?: boolean;
  int_func?: (value: number) => number;
  rate_func?: AnimationOptions["rate_func"];
  time_per_char?: number;
  run_time?: number;
  reverse_rate_function?: boolean;
  introducer?: boolean;
}

// NOTE: text/tex rendering is out of scope for this port (see system rules),
// but AddTextLetterByLetter itself is pure ShowIncreasingSubsets logic over
// an existing mobject's submobjects, so it is ported faithfully with the
// `Text` parameter typed as Mobject.
export class AddTextLetterByLetter extends ShowIncreasingSubsets {
  time_per_char: number;

  constructor(text: Mobject, options: AddTextLetterByLetterOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      suspend_mobject_updating = (clsDefaults.suspend_mobject_updating as boolean | undefined) ?? false,
      int_func = (clsDefaults.int_func as ((value: number) => number) | undefined) ?? Math.ceil,
      rate_func = (clsDefaults.rate_func as AnimationOptions["rate_func"] | undefined) ?? linear,
      time_per_char = (clsDefaults.time_per_char as number | undefined) ?? 0.1,
      run_time = (clsDefaults.run_time as number | undefined) ?? null,
      reverse_rate_function = (clsDefaults.reverse_rate_function as boolean | undefined) ?? false,
      introducer = (clsDefaults.introducer as boolean | undefined) ?? true,
      ...rest
    } = options;
    if (!text.family_members_with_points() || text.family_members_with_points().length === 0) {
      const error = new Error(`The text mobject ${text.toString()} does not seem to contain any characters.`);
      error.name = "ValueError";
      throw error;
    }
    let resolvedRunTime = run_time;
    if (resolvedRunTime === null || resolvedRunTime === undefined) {
      resolvedRunTime = Math.max(1 / config.frame_rate, time_per_char) * text.submobjects.length;
    }
    super(text, {
      ...rest,
      suspend_mobject_updating,
      int_func,
      rate_func,
      run_time: resolvedRunTime,
      reverse_rate_function,
      introducer,
    });
    this.time_per_char = time_per_char;
  }
}

export interface DrawBorderThenFillOptions extends AnimationOptions {
  stroke_width?: number;
  stroke_color?: ParsableManimColor | null;
}

export class DrawBorderThenFill extends Animation {
  stroke_width: number;
  stroke_color: ParsableManimColor | null;
  outline!: Mobject;

  constructor(vmobject: VMobject, options: DrawBorderThenFillOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      run_time = (clsDefaults.run_time as number | undefined) ?? 2,
      rate_func = (clsDefaults.rate_func as AnimationOptions["rate_func"]) ?? double_smooth,
      stroke_width = (clsDefaults.stroke_width as number | undefined) ?? 2,
      stroke_color = (clsDefaults.stroke_color as ParsableManimColor | null | undefined) ?? null,
      introducer = (clsDefaults.introducer as boolean | undefined) ?? true,
      ...rest
    } = options;
    DrawBorderThenFill.typecheckInput(vmobject);
    super(vmobject, { ...rest, run_time, rate_func, introducer });
    this.stroke_width = stroke_width;
    this.stroke_color = stroke_color;
    this.outline = this.get_outline();
  }

  _typecheck_input(vmobject: VMobject): void {
    // NOTE: mirrors Python `self.__class__.__name__` so subclasses (Write,
    // Unwrite, ...) report their own name in the TypeError.
    if (!(vmobject instanceof VMobject)) {
      throw new TypeError(
        `${(this.constructor as unknown as { name: string }).name} only works for vectorized Mobjects`,
      );
    }
  }

  private static typecheckInput(vmobject: VMobject): void {
    if (!(vmobject instanceof VMobject)) {
      throw new TypeError(`DrawBorderThenFill only works for vectorized Mobjects`);
    }
  }

  begin(): void {
    this.outline = this.get_outline();
    super.begin();
  }

  get_outline(): Mobject {
    const outline = this.mobject.copy();
    outline.set_fill(undefined, 0);
    for (const sm of outline.family_members_with_points()) {
      (sm as VMobject).set_stroke(this.get_stroke_color(sm as VMobject), this.stroke_width);
    }
    return outline;
  }

  get_stroke_color(vmobject: VMobject): ParsableManimColor {
    if (this.stroke_color) return this.stroke_color;
    else if (vmobject.get_stroke_width() > 0) return vmobject.get_stroke_color() as unknown as ParsableManimColor;
    return vmobject.get_color() as unknown as ParsableManimColor;
  }

  override get_all_mobjects(): Mobject[] {
    return [...super.get_all_mobjects(), this.outline];
  }

  override get_all_families_zipped(): Mobject[][] {
    const first = this.mobject.family_members_with_points();
    const second = this.starting_mobject.family_members_with_points();
    const third = (this.outline ?? this.mobject).family_members_with_points();
    return first.map((mob, index) => [mob, second[index] ?? mob, third[index] ?? mob]);
  }

  override interpolate_mobject(alpha: number): void {
    const families = this.get_all_families_zipped();
    for (let index = 0; index < families.length; index += 1) {
      const mobs = families[index] ?? [];
      this.interpolate_submobject(
        mobs[0] as Mobject,
        mobs[1] as Mobject,
        mobs[2] as Mobject,
        this.get_sub_alpha(alpha, index, families.length),
      );
    }
  }

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, outlineOrAlpha: Mobject | number, alpha?: number): void {
    // NOTE: Python's signature is
    // `(submobject, starting_submobject, outline, alpha)` (see the `Fixme`
    // there: it does not match the parent class); the extra `outline` part
    // is threaded through `get_all_mobjects`/`get_all_families_zipped`.
    // The union form also accepts the base-class 3-argument call shape.
    const outline = typeof outlineOrAlpha === "number" ? (this.outline ?? startingSubmobject) : outlineOrAlpha;
    const resolvedAlpha = typeof outlineOrAlpha === "number" ? outlineOrAlpha : (alpha as number);
    const [index, subalpha] = integer_interpolate(0, 2, resolvedAlpha);
    if (index === 0) {
      (submobject as VMobject).pointwise_become_partial(outline as VMobject, 0, subalpha);
      submobject.match_style(outline);
    } else {
      submobject.interpolate(outline, startingSubmobject, subalpha);
    }
  }
}

export interface WriteOptions extends Omit<DrawBorderThenFillOptions, "run_time" | "lag_ratio"> {
  rate_func?: AnimationOptions["rate_func"];
  reverse?: boolean;
  run_time?: number | null;
  lag_ratio?: number | null;
  remover?: boolean;
  introducer?: boolean;
}

// NOTE: module-local so Unwrite.__init__ can use it before super().__init__
// (TS cannot touch `this` before `super()`, unlike Python's
// `self._set_default_config_from_length(...)` call).
function setDefaultConfigFromLength(
  vmobject: VMobject,
  runTime: number | null,
  lagRatio: number | null,
): [number, number] {
  const length = vmobject.family_members_with_points().length;
  let resolvedRunTime = runTime;
  if (resolvedRunTime === null || resolvedRunTime === undefined) {
    resolvedRunTime = length < 15 ? 1 : 2;
  }
  let resolvedLagRatio = lagRatio;
  if (resolvedLagRatio === null || resolvedLagRatio === undefined) {
    resolvedLagRatio = Math.min(4.0 / Math.max(1.0, length), 0.2);
  }
  return [resolvedRunTime, resolvedLagRatio];
}

// NOTE: Write is documented for Text/VMobjects, but its logic (inherited
// DrawBorderThenFill machinery plus run_time/lag_ratio defaults from the
// submobject count) is mobject-agnostic, so the parameter is typed as
// Mobject/VMobject here. Text rendering itself remains out of scope.
export class Write extends DrawBorderThenFill {
  reverse: boolean;

  constructor(vmobject: VMobject, options: WriteOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      rate_func = (clsDefaults.rate_func as AnimationOptions["rate_func"] | undefined) ?? linear,
      reverse = (clsDefaults.reverse as boolean | undefined) ?? false,
      run_time: runTimeOption = (clsDefaults.run_time as number | null | undefined) ?? null,
      lag_ratio: lagRatioOption = (clsDefaults.lag_ratio as number | null | undefined) ?? null,
      ...rest
    } = options;
    const [run_time, lag_ratio] = setDefaultConfigFromLength(
      vmobject,
      (runTimeOption as number | null | undefined) ?? null,
      (lagRatioOption as number | null | undefined) ?? null,
    );
    if (!("remover" in rest)) (rest as Record<string, unknown>)["remover"] = reverse;
    super(vmobject, {
      ...rest,
      rate_func,
      run_time,
      lag_ratio,
      introducer: (rest as AnimationOptions).introducer ?? !(reverse as boolean),
    });
    this.reverse = reverse as boolean;
  }

  _set_default_config_from_length(
    vmobject: VMobject,
    runTime: number | null,
    lagRatio: number | null,
  ): [number, number] {
    return setDefaultConfigFromLength(vmobject, runTime, lagRatio);
  }

  reverse_submobjects(): void {
    this.mobject.invert(true);
  }

  override begin(): void {
    if (this.reverse) this.reverse_submobjects();
    super.begin();
  }

  override finish(): void {
    super.finish();
    if (this.reverse) this.reverse_submobjects();
  }
}

export interface UnwriteOptions extends WriteOptions {}

export class Unwrite extends Write {
  // NOTE (deviation): Python's Unwrite.__init__ captures `rate_func` as a
  // named parameter but never forwards it to Write (a custom rate_func is
  // silently dropped and linear is always used). The TS port forwards it,
  // which agrees on the default and honors an explicit value instead of
  // dropping it.
  constructor(vmobject: VMobject, options: UnwriteOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      rate_func = (clsDefaults.rate_func as AnimationOptions["rate_func"] | undefined) ?? linear,
      reverse = (clsDefaults.reverse as boolean | undefined) ?? true,
      run_time: runTimeOption = (clsDefaults.run_time as number | null | undefined) ?? null,
      lag_ratio: lagRatioOption = (clsDefaults.lag_ratio as number | null | undefined) ?? null,
      ...rest
    } = options;
    const [run_time, lag_ratio] = setDefaultConfigFromLength(
      vmobject,
      (runTimeOption as number | null | undefined) ?? null,
      (lagRatioOption as number | null | undefined) ?? null,
    );
    super(vmobject, {
      ...rest,
      rate_func,
      run_time,
      lag_ratio,
      reverse_rate_function: true,
      reverse,
    });
  }
}

export class RemoveTextLetterByLetter extends AddTextLetterByLetter {
  constructor(text: Mobject, options: AddTextLetterByLetterOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    super(text, {
      reverse_rate_function: (clsDefaults.reverse_rate_function as boolean | undefined) ?? true,
      introducer: (clsDefaults.introducer as boolean | undefined) ?? false,
      remover: (clsDefaults.remover as boolean | undefined) ?? true,
      ...options,
    });
  }
}

// TODO, this is broken... (mirrors the Python comment above AddTextWordByWord)
export class AddTextWordByWord extends Succession {
  time_per_char: number;

  constructor(
    textMobject: Mobject,
    options: AnimationOptions & { time_per_char?: number; run_time?: number | null } = {},
  ) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      // NOTE: mirrors Python, where the `run_time` parameter is accepted
      // but never forwarded (the sub-animations carry their own run_times).
      run_time: _ignoredRunTime = (clsDefaults.run_time as number | null | undefined) ?? null,
      time_per_char = (clsDefaults.time_per_char as number | undefined) ?? 0.06,
      ...rest
    } = options;
    void _ignoredRunTime;
    const tpc = time_per_char;
    const anims: Animation[] = [];
    for (const word of textMobject) {
      const wordLength = (word as Mobject).__len__();
      anims.push(
        new ShowIncreasingSubsets(word as Mobject, { run_time: tpc * wordLength }),
        new Animation(word as Mobject, { run_time: 0.005 * wordLength ** 1.5 }),
      );
    }
    super(...anims, rest);
    this.time_per_char = tpc;
  }
}

export interface TypeWithCursorOptions extends AddTextLetterByLetterOptions {
  buff?: number;
  keep_cursor_y?: boolean;
  leave_cursor_on?: boolean;
}

export class TypeWithCursor extends AddTextLetterByLetter {
  cursor: Mobject;
  buff: number;
  keep_cursor_y: boolean;
  leave_cursor_on: boolean;
  y_cursor!: number;

  constructor(text: Mobject, cursor: Mobject, options: TypeWithCursorOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      buff = (clsDefaults.buff as number | undefined) ?? 0.1,
      keep_cursor_y = (clsDefaults.keep_cursor_y as boolean | undefined) ?? true,
      leave_cursor_on = (clsDefaults.leave_cursor_on as boolean | undefined) ?? true,
      time_per_char = (clsDefaults.time_per_char as number | undefined) ?? 0.1,
      reverse_rate_function = (clsDefaults.reverse_rate_function as boolean | undefined) ?? false,
      introducer = (clsDefaults.introducer as boolean | undefined) ?? true,
      ...rest
    } = options;
    super(text, {
      ...rest,
      time_per_char,
      reverse_rate_function,
      introducer,
    });
    this.cursor = cursor;
    this.buff = buff;
    this.keep_cursor_y = keep_cursor_y;
    this.leave_cursor_on = leave_cursor_on;
  }

  override begin(): void {
    this.y_cursor = this.cursor.get_y();
    (this.cursor as unknown as Record<string, unknown>)["initial_position"] = this.mobject.get_center();
    if (this.keep_cursor_y) this.cursor.set_y(this.y_cursor);

    this.cursor.set_opacity(0);
    this.mobject.add(this.cursor);
    super.begin();
  }

  override finish(): void {
    if (this.leave_cursor_on) {
      this.cursor.set_opacity(1);
    } else {
      this.cursor.set_opacity(0);
      this.mobject.remove(this.cursor);
    }
    super.finish();
  }

  override clean_up_from_scene(scene: { remove: (...mobjects: Mobject[]) => void }): void {
    if (!this.leave_cursor_on) scene.remove(this.cursor);
    super.clean_up_from_scene(scene);
  }

  override update_submobject_list(index: number): void {
    for (const mobject of this.all_submobs.slice(0, index)) mobject.set_opacity(1);
    for (const mobject of this.all_submobs.slice(index)) mobject.set_opacity(0);

    const initialPosition = (this.cursor as unknown as Record<string, number[]>)["initial_position"] as number[];
    if (index !== 0) {
      this.cursor.next_to(this.all_submobs[index - 1], RIGHT, this.buff).set_y(initialPosition[1]);
    } else {
      this.cursor.move_to(this.all_submobs[0]).set_y(initialPosition[1]);
    }

    if (this.keep_cursor_y) this.cursor.set_y(this.y_cursor);
    this.cursor.set_opacity(1);
  }
}

export interface UntypeWithCursorOptions extends TypeWithCursorOptions {}

export class UntypeWithCursor extends TypeWithCursor {
  constructor(text: Mobject, cursor: Mobject, options: UntypeWithCursorOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    super(text, cursor, {
      time_per_char: (clsDefaults.time_per_char as number | undefined) ?? 0.1,
      reverse_rate_function: (clsDefaults.reverse_rate_function as boolean | undefined) ?? true,
      introducer: (clsDefaults.introducer as boolean | undefined) ?? false,
      remover: (clsDefaults.remover as boolean | undefined) ?? true,
      ...options,
    });
  }
}

export interface SpiralInOptions extends AnimationOptions {
  scale_factor?: number;
  fade_in_fraction?: number;
}

interface SpiralShapeState {
  final_position: number[];
  initial_position: number[];
}

export class SpiralIn extends Animation {
  shapes: Mobject;
  scale_factor: number;
  shape_center: number[];
  fade_in_fraction: number;

  constructor(shapes: Mobject, options: SpiralInOptions = {}) {
    const clsDefaults = classDefaultsOf(new.target);
    const {
      scale_factor = (clsDefaults.scale_factor as number | undefined) ?? 8,
      fade_in_fraction = (clsDefaults.fade_in_fraction as number | undefined) ?? 0.3,
      ...rest
    } = options;
    const shapesCopy = shapes.copy();
    const shapeCenter = [...shapes.get_center()];
    for (const shape of shapes) {
      const state = shape as unknown as SpiralShapeState;
      const finalPosition = [...shape.get_center()];
      state.final_position = finalPosition;
      state.initial_position = addPoints(
        finalPosition,
        scalePoint(subtractPoints(finalPosition, shapeCenter, shape.dim), scale_factor, shape.dim),
        shape.dim,
      );
      shape.move_to(state.initial_position);
      shape.save_state();
    }
    super(shapes, { ...rest, introducer: true });
    this.shapes = shapesCopy;
    this.scale_factor = scale_factor;
    this.shape_center = shapeCenter;
    this.fade_in_fraction = fade_in_fraction;
  }

  override interpolate_mobject(alpha: number): void {
    const resolvedAlpha = this.rate_func(alpha);
    const originals = [...this.shapes];
    const currents = [...this.mobject];
    if (originals.length !== currents.length) {
      const error = new Error("zip() argument 2 is shorter than argument 1 (strict=True)");
      error.name = "ValueError";
      throw error;
    }
    for (let index = 0; index < originals.length; index += 1) {
      const originalShape = originals[index] as VMobject;
      const shape = currents[index] as Mobject & Partial<SpiralShapeState>;
      shape.restore?.();
      const fillOpacity = originalShape.get_fill_opacity();
      const strokeOpacity = originalShape.get_stroke_opacity();
      const newFillOpacity = Math.min(fillOpacity, (resolvedAlpha * fillOpacity) / this.fade_in_fraction);
      const newStrokeOpacity = Math.min(strokeOpacity, (resolvedAlpha * strokeOpacity) / this.fade_in_fraction);
      if (shape.final_position !== undefined && shape.initial_position !== undefined) {
        shape.shift(scalePoint(subtractPoints(shape.final_position, shape.initial_position, shape.dim), resolvedAlpha, shape.dim));
      }
      shape.rotate(TAU * resolvedAlpha, undefined, this.shape_center);
      shape.rotate(-TAU * resolvedAlpha, undefined, shape.get_center_of_mass());
      (shape as unknown as VMobject).set_fill(undefined, newFillOpacity);
      (shape as unknown as VMobject).set_stroke(undefined, undefined, newStrokeOpacity);
    }
  }
}
