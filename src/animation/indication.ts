import { Animation, classDefaultsOf, smooth, type AnimationOptions, type RateFunction } from "./animation.js";
import { AnimationGroup, Succession, type AnimationGroupOptions } from "./composition.js";
import { Create, ShowPartial, Uncreate, type ShowPartialOptions } from "./creation.js";
import { FadeIn, FadeOut } from "./fading.js";
import { Homotopy } from "./movement.js";
import { Transform, type TransformOptions } from "./transform.js";
import { UpdateFromFunc } from "./updaters/update.js";
import { Mobject } from "../mobject/mobject.js";
import { VMobject, VGroup } from "../mobject/types/vectorized_mobject.js";
import { Circle, Dot } from "../mobject/geometry/arc.js";
import { Line } from "../mobject/geometry/line.js";
import { Rectangle } from "../mobject/geometry/polygram.js";
import { SurroundingRectangle } from "../mobject/geometry/shape_matchers.js";
import { config } from "../_config/utils.js";
import { interpolate, inverse_interpolate } from "../utils/bezier.js";
import { cloneNumericArray, linspace } from "../utils/arrays.js";
import { there_and_back, wiggle } from "../utils/rate_functions.js";
import { normalize } from "../utils/space_ops.js";
import { GREY, ManimColor, type ParsableManimColor } from "../utils/color/core.js";
import { DEFAULT_STROKE_WIDTH, ORIGIN, RIGHT, SMALL_BUFF, TAU, UP } from "../constants.js";

export interface FocusOnOptions extends TransformOptions {
  opacity?: number;
  color?: ParsableManimColor;
  run_time?: number;
}

export class FocusOn extends Transform {
  focus_point: number[] | Mobject;
  color: ParsableManimColor;
  opacity: number;

  constructor(
    focus_point: number[] | Mobject,
    opacity = 0.2,
    color: ParsableManimColor = GREY,
    run_time = 2,
    options: FocusOnOptions = {},
  ) {
    const { opacity: _o, color: _c, run_time: _r, ...rest } = options;
    void _o; void _c; void _r;
    const starting_dot = new Dot(
      [...ORIGIN],
      Number(config.frame_x_radius) + Number(config.frame_y_radius),
      0,
      0,
      color,
    );
    super(starting_dot, { ...rest, run_time, remover: true });
    this.focus_point = focus_point;
    this.color = color;
    this.opacity = opacity;
  }

  override create_target(): Dot {
    const little_dot = new Dot([...ORIGIN], 0, 0, 1, this.color);
    little_dot.set_fill(this.color, this.opacity);
    const focusPoint = this.focus_point;
    little_dot.add_updater((d: Mobject) => d.move_to(focusPoint as Mobject));
    return little_dot;
  }
}

export interface IndicateOptions extends TransformOptions {
  scale_factor?: number;
  color?: ParsableManimColor;
  rate_func?: RateFunction;
}

export class Indicate extends Transform {
  color: ParsableManimColor;
  scale_factor: number;

  constructor(
    mobject: Mobject,
    scale_factor = 1.2,
    color: ParsableManimColor = new ManimColor("PURE_YELLOW"),
    rate_func: RateFunction = there_and_back,
    options: IndicateOptions = {},
  ) {
    const { scale_factor: _s, color: _c, rate_func: _r, ...rest } = options;
    void _s; void _c; void _r;
    super(mobject, { ...rest, rate_func });
    this.color = color;
    this.scale_factor = scale_factor;
  }

  override create_target(): Mobject {
    const target = this.mobject.copy();
    target.scale(this.scale_factor);
    target.set_color(this.color);
    return target;
  }
}

export interface FlashOptions extends AnimationGroupOptions {
  line_length?: number;
  num_lines?: number;
  flash_radius?: number;
  line_stroke_width?: number;
  color?: ParsableManimColor;
  time_width?: number;
  run_time?: number;
}

export class Flash extends AnimationGroup {
  point: number[];
  color: ParsableManimColor;
  line_length: number;
  num_lines: number;
  flash_radius: number;
  line_stroke_width: number;
  time_width: number;
  lines: VGroup;
  animation_config: Record<string, unknown>;

  constructor(
    point: number[] | Mobject,
    line_length = 0.2,
    num_lines = 12,
    flash_radius = 0.1,
    line_stroke_width = 3,
    color: ParsableManimColor = new ManimColor("PURE_YELLOW"),
    time_width = 1,
    run_time = 1.0,
    options: FlashOptions = {},
  ) {
    const {
      line_length: _ll, num_lines: _nl, flash_radius: _fr,
      line_stroke_width: _lw, color: _c, time_width: _tw, run_time: _rt,
      ...rest
    } = options;
    void _ll; void _nl; void _fr; void _lw; void _c; void _tw; void _rt;
    const resolvedPoint = point instanceof Mobject ? [...point.get_center()] : [...point];
    // NOTE: Python assigns these pre-super().__init__ attributes (point,
    // color, line_length, ..., run_time) before delegating; TS requires
    // super() first, so locals carry the values until assignment below.
    // Python's pre-super `self.run_time` is overwritten by
    // init_run_time(None) -> max_end_time, which the base constructor
    // already computes, so it is not re-applied here.
    const lines = Flash.createLines(resolvedPoint, line_length, num_lines, flash_radius, line_stroke_width, color);
    const animations = Flash.createLineAnims(lines, time_width, run_time, rest);
    super(...animations, { ...rest, group: lines });
    this.point = resolvedPoint;
    this.color = color;
    this.line_length = line_length;
    this.num_lines = num_lines;
    this.flash_radius = flash_radius;
    this.line_stroke_width = line_stroke_width;
    this.time_width = time_width;
    this.animation_config = { ...rest };
    this.lines = lines;
  }

  private static createLines(
    point: number[],
    line_length: number,
    num_lines: number,
    flash_radius: number,
    line_stroke_width: number,
    color: ParsableManimColor,
  ): VGroup {
    const lines = new VGroup();
    const step = TAU / num_lines;
    for (let index = 0; index < num_lines; index += 1) {
      const angle = index * step;
      const end = [point[0] + line_length * RIGHT[0], point[1] + line_length * RIGHT[1], point[2] + line_length * RIGHT[2]];
      const line = new Line(point, end);
      line.shift([flash_radius * RIGHT[0], flash_radius * RIGHT[1], flash_radius * RIGHT[2]]);
      line.rotate(angle, undefined, point);
      lines.add(line);
    }
    lines.set_color(color);
    lines.set_stroke(undefined, line_stroke_width);
    return lines;
  }

  private static createLineAnims(
    lines: VGroup,
    time_width: number,
    run_time: number,
    animation_config: Record<string, unknown>,
  ): ShowPassingFlash[] {
    return [...lines.submobjects].map(
      (line) => new ShowPassingFlash(line as VMobject, time_width, { ...(animation_config as ShowPartialOptions), run_time }),
    );
  }

  create_lines(): VGroup {
    // NOTE: mirrors Python, which builds and returns a fresh VGroup without
    // reassigning `self.lines` (only `__init__` assigns it).
    return Flash.createLines(this.point, this.line_length, this.num_lines, this.flash_radius, this.line_stroke_width, this.color);
  }

  create_line_anims(): ShowPassingFlash[] {
    return Flash.createLineAnims(this.lines, this.time_width, this.run_time, this.animation_config);
  }
}

export class ShowPassingFlash extends ShowPartial {
  time_width: number;

  constructor(mobject: VMobject, time_width = 0.1, options: ShowPartialOptions = {}) {
    super(mobject, { remover: true, introducer: true, ...options });
    this.time_width = time_width;
  }

  _get_bounds(alpha: number): [number, number] {
    const tw = this.time_width;
    let upper = interpolate(0, 1 + tw, alpha) as number;
    let lower = upper - tw;
    upper = Math.min(upper, 1);
    lower = Math.max(lower, 0);
    return [lower, upper];
  }

  clean_up_from_scene(scene: { remove: (mobject: Mobject) => void }): void {
    super.clean_up_from_scene(scene);
    for (const [submob, start] of this.get_all_families_zipped()) {
      (submob as VMobject).pointwise_become_partial(start as VMobject, 0, 1);
    }
  }
}

export class ShowPassingFlashWithThinningStrokeWidth extends AnimationGroup {
  n_segments: number;
  time_width: number;

  constructor(
    vmobject: VMobject,
    n_segments = 10,
    time_width = 0.1,
    remover = true,
    options: AnimationGroupOptions & { time_width?: number } = {},
  ) {
    const { time_width: _tw, ...rest } = options;
    void _tw;
    // NOTE: Python assigns n_segments/time_width/remover pre-super(); the
    // base constructor overwrites remover via its defaults (it is not
    // forwarded), so only n_segments/time_width survive — mirrored here by
    // assigning after super().
    // Mirrors `kwargs.pop("time_width", self.time_width)`.
    const max_time_width = (options.time_width as number | undefined) ?? time_width;
    const max_stroke_width = vmobject.get_stroke_width();
    const stroke_widths = linspace(0, max_stroke_width, n_segments);
    const time_widths = linspace(max_time_width, 0, n_segments);
    const animations = stroke_widths.map(
      (stroke_width, index) =>
        new ShowPassingFlash(
          (vmobject.copy() as VMobject).set_stroke(undefined, stroke_width),
          time_widths[index] ?? 0,
          { ...(rest as ShowPartialOptions) },
        ),
    );
    super(...animations);
    this.n_segments = n_segments;
    this.time_width = time_width;
    void remover;
  }
}

export interface ApplyWaveOptions extends AnimationOptions {
  direction?: readonly number[];
  amplitude?: number;
  wave_func?: RateFunction;
  time_width?: number;
  ripples?: number;
}

export class ApplyWave extends Homotopy {
  constructor(
    mobject: Mobject,
    direction: readonly number[] = UP,
    amplitude = 0.2,
    wave_func: RateFunction = smooth,
    time_width = 1,
    ripples = 1,
    run_time = 2,
    options: AnimationOptions = {},
  ) {
    const x_min = mobject.get_left()[0] ?? 0;
    const x_max = mobject.get_right()[0] ?? 0;
    const vect = normalize([...direction]).map((value) => value * amplitude);

    const wave = (t: number): number => {
      // Mirrored looks better in the way the wave is used.
      t = 1 - t;

      // Clamp input
      if (t >= 1 || t <= 0) return 0;

      const phases = ripples * 2;
      let phase = Math.trunc(t * phases);
      if (phase === 0) {
        // First rising ripple
        return wave_func(t * phases);
      } else if (phase === phases - 1) {
        // last ripple. Rising or falling depending on the number of ripples
        // The (ripples % 2)-term is used to make this distinction.
        t -= phase / phases; // Time relative to the phase
        return (1 - wave_func(t * phases)) * (2 * (ripples % 2) - 1);
      } else {
        // Longer phases:
        phase = Math.trunc((phase - 1) / 2);
        t -= (2 * phase + 1) / phases;

        // Similar to last ripple:
        return (1 - 2 * wave_func(t * ripples)) * (1 - 2 * (phase % 2));
      }
    };

    const homotopy = (x: number, y: number, z: number, t: number): [number, number, number] => {
      const upper = interpolate(0, 1 + time_width, t) as number;
      const lower = upper - time_width;
      const relative_x = inverse_interpolate(x_min, x_max, x) as number;
      const wave_phase = inverse_interpolate(lower, upper, relative_x) as number;
      const nudge = wave(wave_phase);
      return [x + nudge * (vect[0] ?? 0), y + nudge * (vect[1] ?? 0), z + nudge * (vect[2] ?? 0)];
    };
    super(homotopy, mobject, { run_time, ...options });
  }
}

export interface ApplyWaveKwargs extends ApplyWaveOptions {
  run_time?: number;
}

export function createApplyWave(
  mobject: Mobject,
  options: ApplyWaveKwargs = {},
): ApplyWave {
  const { direction = UP, amplitude = 0.2, wave_func = smooth, time_width = 1, ripples = 1, run_time = 2, ...rest } = options;
  return new ApplyWave(mobject, direction, amplitude, wave_func, time_width, ripples, run_time, rest);
}

export interface WiggleOptions extends AnimationOptions {
  scale_value?: number;
  rotation_angle?: number;
  n_wiggles?: number;
  scale_about_point?: readonly number[] | null;
  rotate_about_point?: readonly number[] | null;
}

export class Wiggle extends Animation {
  scale_value: number;
  rotation_angle: number;
  n_wiggles: number;
  scale_about_point: number[] | null;
  rotate_about_point: number[] | null;

  constructor(mobject: Mobject, options: WiggleOptions | number = {}) {
    // Python `Wiggle(mobject, scale_value=1.1, ...)` allows `scale_value` to be
    // passed positionally; accept a numeric second argument for parity.
    if (typeof options === "number") options = { scale_value: options };
    const clsDefaults = classDefaultsOf(new.target);
    const {
      scale_value = (clsDefaults.scale_value as number | undefined) ?? 1.1,
      rotation_angle = (clsDefaults.rotation_angle as number | undefined) ?? 0.01 * TAU,
      n_wiggles = (clsDefaults.n_wiggles as number | undefined) ?? 6,
      scale_about_point = (clsDefaults.scale_about_point as readonly number[] | null | undefined) ?? null,
      rotate_about_point = (clsDefaults.rotate_about_point as readonly number[] | null | undefined) ?? null,
      run_time = (clsDefaults.run_time as number | undefined) ?? 2,
      ...rest
    } = options;
    super(mobject, { ...rest, run_time });
    this.scale_value = scale_value;
    this.rotation_angle = rotation_angle;
    this.n_wiggles = n_wiggles;
    this.scale_about_point = scale_about_point === null || scale_about_point === undefined ? null : [...scale_about_point];
    this.rotate_about_point = rotate_about_point === null || rotate_about_point === undefined ? null : [...rotate_about_point];
  }

  get_scale_about_point(): number[] {
    if (this.scale_about_point === null) return this.mobject.get_center();
    return this.scale_about_point;
  }

  get_rotate_about_point(): number[] {
    if (this.rotate_about_point === null) return this.mobject.get_center();
    return this.rotate_about_point;
  }

  override interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): void {
    // Mirrors Python `submobject.points[:, :] = starting_submobject.points`
    // (value copy into distinct storage, never a reference alias).
    submobject.points = cloneNumericArray(startingSubmobject.points);
    submobject.scale(
      interpolate(1, this.scale_value, there_and_back(alpha)) as number,
      this.get_scale_about_point(),
    );
    submobject.rotate(
      (wiggle as (t: number, wiggles?: number) => number)(alpha, this.n_wiggles) * this.rotation_angle,
      undefined,
      this.get_rotate_about_point(),
    );
  }
}

export interface CircumscribeOptions extends AnimationGroupOptions {
  shape?: typeof Rectangle | typeof Circle;
  fade_in?: boolean;
  fade_out?: boolean;
  time_width?: number;
  buff?: number;
  color?: ParsableManimColor;
  run_time?: number;
  stroke_width?: number;
}

export class Circumscribe extends Succession {
  constructor(
    mobject: Mobject,
    shape: typeof Rectangle | typeof Circle = Rectangle,
    fade_in = false,
    fade_out = false,
    time_width = 0.3,
    buff = SMALL_BUFF,
    color: ParsableManimColor = new ManimColor("PURE_YELLOW"),
    run_time = 1,
    stroke_width = DEFAULT_STROKE_WIDTH,
    options: CircumscribeOptions = {},
  ) {
    const {
      shape: shapeOption,
      fade_in: fadeInOption,
      fade_out: fadeOutOption,
      time_width: timeWidthOption,
      buff: buffOption,
      color: colorOption,
      run_time: runTimeOption,
      stroke_width: strokeWidthOption,
      ...rest
    } = options;
    const resolvedShape = shapeOption ?? shape;
    const resolvedFadeIn = fadeInOption ?? fade_in;
    const resolvedFadeOut = fadeOutOption ?? fade_out;
    const resolvedTimeWidth = timeWidthOption ?? time_width;
    const resolvedBuff = buffOption ?? buff;
    const resolvedColor = colorOption ?? color;
    const resolvedRunTime = runTimeOption ?? run_time;
    const resolvedStrokeWidth = strokeWidthOption ?? stroke_width;
    let frame: SurroundingRectangle | Circle;
    if (resolvedShape === Rectangle) {
      frame = new SurroundingRectangle(mobject, {
        color: resolvedColor,
        buff: resolvedBuff,
        stroke_width: resolvedStrokeWidth,
      });
    } else if (resolvedShape === Circle) {
      const circle = new Circle(null, {
        color: resolvedColor,
        stroke_width: resolvedStrokeWidth,
      });
      circle.surround(mobject, 0, false, 1);
      const radius = circle.width / 2;
      circle.scale((radius + resolvedBuff) / radius);
      frame = circle;
    } else {
      const error = new Error("shape should be either Rectangle or Circle.");
      error.name = "ValueError";
      throw error;
    }

    if (resolvedFadeIn && resolvedFadeOut) {
      super(
        new FadeIn(frame, { run_time: resolvedRunTime / 2 }),
        new FadeOut(frame, { run_time: resolvedRunTime / 2 }),
        rest,
      );
    } else if (resolvedFadeIn) {
      frame.reverse_direction();
      super(
        new FadeIn(frame, { run_time: resolvedRunTime / 2 }),
        new Uncreate(frame as VMobject, { run_time: resolvedRunTime / 2 }),
        rest,
      );
    } else if (resolvedFadeOut) {
      super(
        new Create(frame as VMobject, { run_time: resolvedRunTime / 2 }),
        new FadeOut(frame, { run_time: resolvedRunTime / 2 }),
        rest,
      );
    } else {
      super(
        new ShowPassingFlash(frame as VMobject, resolvedTimeWidth, { run_time: resolvedRunTime }),
        rest,
      );
    }
  }
}

export interface BlinkOptions extends AnimationGroupOptions {
  time_on?: number;
  time_off?: number;
  blinks?: number;
  hide_at_end?: boolean;
}

export class Blink extends Succession {
  constructor(
    mobject: Mobject,
    time_on = 0.5,
    time_off = 0.5,
    blinks = 1,
    hide_at_end = false,
    options: BlinkOptions = {},
  ) {
    const {
      time_on: timeOnOption,
      time_off: timeOffOption,
      blinks: blinksOption,
      hide_at_end: hideAtEndOption,
      ...rest
    } = options;
    const resolvedTimeOn = timeOnOption ?? time_on;
    const resolvedTimeOff = timeOffOption ?? time_off;
    const resolvedBlinks = blinksOption ?? blinks;
    const resolvedHideAtEnd = hideAtEndOption ?? hide_at_end;
    const onAnim = (): UpdateFromFunc =>
      new UpdateFromFunc(mobject, (mob: Mobject) => { mob.set_opacity(1.0); }, { run_time: resolvedTimeOn });
    const offAnim = (): UpdateFromFunc =>
      new UpdateFromFunc(mobject, (mob: Mobject) => { mob.set_opacity(0.0); }, { run_time: resolvedTimeOff });
    // NOTE: mirrors Python `[on, off] * blinks` (list repetition): each blink
    // gets fresh animation instances sharing the same mobject.
    const animations: UpdateFromFunc[] = [];
    for (let index = 0; index < resolvedBlinks; index += 1) {
      animations.push(onAnim(), offAnim());
    }
    if (!resolvedHideAtEnd) {
      animations.push(onAnim());
    }
    super(...animations, rest);
  }
}
