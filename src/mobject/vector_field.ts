import { array, NDArray } from "numpy-ts";
import { VGroup, VMobject, VectorizedPoint } from "./types/vectorized_mobject.js";
import { Mobject } from "./mobject.js";
import { CoordinateSystem } from "./graphing/coordinate_systems.js";
import { Animation, type AnimationOptions, linear } from "../animation/animation.js";
import { AnimationGroup, Succession } from "../animation/composition.js";
import { UpdateFromAlphaFunc } from "../animation/updaters/update.js";
import { Create, ShowPartial } from "../animation/creation.js";
import { BLUE_E, GREEN, ManimColor, RED, YELLOW, color_gradient, color_to_rgb, interpolate_color, rgb_to_color, type ParsableManimColor } from "../utils/color/core.js";
import { inverse_interpolate } from "../utils/bezier.js";
import { addPoints, arange, linspace, norm, pointRows, pointsFrom, scalePoint, subtractPoints } from "../utils/arrays.js";
import { clip as _clip } from "../utils/simple_functions.js";
import { ease_out_sine } from "../utils/rate_functions.js";
import { config } from "../_config/utils.js";
import { OUT, RIGHT, UP } from "../constants.js";

export type VectorFunction = (point: number[] | NDArray) => number[] | NDArray;
export type ColorScheme = (vector: number[] | NDArray) => number;

function vectorOf(value: number[] | NDArray): number[] {
  return pointRows(value, 3)[0] ?? [0, 0, 0];
}

function interpolateRgb(first: number[], second: number[], alpha: number): number[] {
  return first.map((value, index) => value * (1 - alpha) + (second[index] ?? 0) * alpha);
}

export const DEFAULT_SCALAR_FIELD_COLORS: ParsableManimColor[] = [BLUE_E, GREEN, YELLOW, RED];

export interface VectorFieldOptions {
  color?: ParsableManimColor | null;
  color_scheme?: ColorScheme | null;
  min_color_scheme_value?: number;
  max_color_scheme_value?: number;
  colors?: ParsableManimColor[];
  [key: string]: unknown;
}

export class VectorField extends VGroup {
  func: VectorFunction;
  single_color: boolean;
  color_scheme: ColorScheme;
  min_color_scheme_value: number;
  max_color_scheme_value: number;
  colors: ParsableManimColor[];
  rgbs: number[][];
  // NOTE: mirrors Python, where pos_to_rgb/pos_to_color only exist on
  // multi-color fields (single-color fields have neither attribute).
  // `declare` (no runtime field) so `"pos_to_rgb" in field` is false for
  // single-color fields, exactly like Python's missing attribute.
  declare pos_to_rgb?: (position: number[] | NDArray) => number[];
  declare pos_to_color?: (position: number[] | NDArray) => ManimColor;
  submob_movement_updater: MobjectUpdater | null = null;

  constructor(func: VectorFunction, color: ParsableManimColor | VectorFieldOptions | null = {}, color_scheme?: ColorScheme | null, min_color_scheme_value = 0, max_color_scheme_value = 2, colors: ParsableManimColor[] = DEFAULT_SCALAR_FIELD_COLORS) {
    super();
    const looksLikeOptions = color !== null && typeof color === "object" && !(color instanceof ManimColor) && !(color instanceof NDArray) && !Array.isArray(color);
    const options: VectorFieldOptions = looksLikeOptions ? color as VectorFieldOptions : { color: color as ParsableManimColor | null, color_scheme, min_color_scheme_value, max_color_scheme_value, colors };
    this.func = func;
    this.min_color_scheme_value = options.min_color_scheme_value ?? 0;
    this.max_color_scheme_value = options.max_color_scheme_value ?? 2;
    this.colors = options.colors ?? DEFAULT_SCALAR_FIELD_COLORS;
    const default_color_scheme = function color_scheme(vec: number[] | NDArray): number { return norm(vectorOf(vec)); };
    if (options.color === undefined || options.color === null) {
      this.single_color = false;
      this.color_scheme = options.color_scheme ?? default_color_scheme;
      this.rgbs = this.colors.map((color) => pointRows(new ManimColor(color).toRgb(), 3)[0] ?? [0, 0, 0]);
      const pos_to_rgb = (position: number[] | NDArray): number[] => {
        const value = this.color_scheme(vectorOf(this.func(position)));
        const clipped = _clip(value, this.min_color_scheme_value, this.max_color_scheme_value);
        let alpha = (clipped - this.min_color_scheme_value) / (this.max_color_scheme_value - this.min_color_scheme_value);
        alpha *= this.rgbs.length - 1;
        const first = this.rgbs[Math.trunc(alpha)] ?? this.rgbs[0];
        const second = this.rgbs[Math.min(Math.trunc(alpha + 1), this.rgbs.length - 1)] ?? first;
        alpha %= 1;
        return interpolateRgb(first, second, alpha);
      };
      this.pos_to_rgb = pos_to_rgb;
      this.pos_to_color = (position) => rgb_to_color(pos_to_rgb(position));
    } else {
      this.single_color = true;
      this.color_scheme = options.color_scheme ?? default_color_scheme;
      this.color = new ManimColor(options.color);
      this.rgbs = [];
    }
  }

  fit_to_coordinate_system(coordinateSystem: CoordinateSystem): this {
    this.apply_function((position) => vectorOf(coordinateSystem.coords_to_point(position)));
    return this;
  }

  runge_kutta(p: number[], step_size: number): number[] {
    const k1 = vectorOf(this.func(p));
    const k2 = vectorOf(this.func(addPoints(p, scalePoint(k1, step_size * 0.5, 3), 3)));
    const k3 = vectorOf(this.func(addPoints(p, scalePoint(k2, step_size * 0.5, 3), 3)));
    const k4 = vectorOf(this.func(addPoints(p, scalePoint(k3, step_size, 3), 3)));
    return scalePoint(addPoints(addPoints(k1, scalePoint(k2, 2, 3), 3), addPoints(scalePoint(k3, 2, 3), k4, 3), 3), step_size / 6, 3);
  }

  nudge(mob: Mobject, dt = 1, substeps = 1, pointwise = false): this {
    const runge_kutta = (p: number[], step_size: number): number[] => this.runge_kutta(p, step_size);
    const step_size = dt / substeps;
    for (let index = 0; index < substeps; index += 1) {
      if (pointwise) mob.apply_function((point) => addPoints(point, runge_kutta(point, step_size), 3));
      else mob.shift(runge_kutta(mob.get_center(), step_size));
    }
    return this;
  }

  nudge_submobjects(dt = 1, substeps = 1, pointwise = false): this {
    for (const mobject of this.submobjects) this.nudge(mobject, dt, substeps, pointwise);
    return this;
  }

  get_nudge_updater(speed = 1, pointwise = false): (mob: Mobject, dt: number) => VectorField {
    return (mob: Mobject, dt: number) => this.nudge(mob, dt * speed, 1, pointwise);
  }

  getNudgeUpdater(speed = 1, pointwise = false): (mob: Mobject, dt: number) => VectorField {
    return this.get_nudge_updater(speed, pointwise);
  }

  start_submobject_movement(speed = 1, pointwise = false): this {
    this.stop_submobject_movement();
    this.submob_movement_updater = (mob: Mobject, dt: number): void => {
      (mob as VectorField).nudge_submobjects(dt * speed, 1, pointwise);
    };
    this.add_updater(this.submob_movement_updater);
    return this;
  }

  stop_submobject_movement(): this {
    if (this.submob_movement_updater !== null) {
      this.remove_updater(this.submob_movement_updater);
    }
    this.submob_movement_updater = null;
    return this;
  }

  get_colored_background_image(sampling_rate = 5): NDArray {
    if (this.single_color) {
      const error = new Error(
        "There is no point in generating an image if the vector field uses a single color.",
      );
      error.name = "ValueError";
      throw error;
    }
    const pos_to_rgb = this.pos_to_rgb;
    if (!pos_to_rgb) {
      const error = new Error(
        "There is no point in generating an image if the vector field uses a single color.",
      );
      error.name = "ValueError";
      throw error;
    }
    const pixel_height = config.pixel_height as number;
    const pixel_width = config.pixel_width as number;
    const frame_width = config.frame_width as number;
    const frame_height = config.frame_height as number;
    const ph = Math.trunc(pixel_height / sampling_rate);
    const pw = Math.trunc(pixel_width / sampling_rate);
    const x_array = linspace(-frame_width / 2, frame_width / 2, pw);
    const y_array = linspace(frame_height / 2, -frame_height / 2, ph);
    const rows: number[][][] = [];
    for (let row = 0; row < ph; row += 1) {
      const line: number[][] = [];
      for (let column = 0; column < pw; column += 1) {
        const rgb = pos_to_rgb([x_array[column] ?? 0, y_array[row] ?? 0, 0]);
        line.push(rgb.map((channel) => Math.max(0, Math.min(255, Math.trunc(channel * 255)))));
      }
      rows.push(line);
    }
    // NOTE: Python returns a PIL Image; the TypeScript runtime has no image
    // object, so the uint8 (ph, pw, 3) array backing that image is returned.
    return array(rows.length > 0 ? rows : [[]]).astype("uint8") as NDArray;
  }

  getColoredBackgroundImage(sampling_rate = 5): NDArray {
    return this.get_colored_background_image(sampling_rate);
  }

  get_vectorized_rgba_gradient_function(
    start: number,
    end: number,
    colors: ParsableManimColor[],
  ): (values: number[] | NDArray, opacity?: number) => NDArray {
    const rgbs: number[][] = colors.map((color) => pointRows(color_to_rgb(color), 3)[0] ?? [0, 0, 0]);

    const func = (values: number[] | NDArray, opacity = 1.0): NDArray => {
      // NOTE: mirrors Python `np.array(values)` flattening: a 1-D array of
      // N values yields N gradient rows. pointRows(..., 1) would treat a
      // flat 1-D NDArray as a single row, so flatten NDArrays explicitly.
      let flat: number[];
      if (Array.isArray(values)) {
        flat = (values as number[]).map(Number);
      } else {
        const raw = (values as NDArray).toArray() as unknown;
        flat = [];
        const walk = (node: unknown): void => {
          if (Array.isArray(node)) {
            for (const item of node) walk(item);
          } else {
            flat.push(Number(node));
          }
        };
        walk(raw);
      }
      const new_rgbas: number[][] = flat.map((value) => {
        const alpha = Math.min(1, Math.max(0, inverse_interpolate(start, end, value) as number));
        const scaled = alpha * (rgbs.length - 1);
        const index = Math.trunc(scaled);
        const nextIndex = Math.min(index + 1, rgbs.length - 1);
        const inter = scaled % 1;
        const first = rgbs[index] ?? [0, 0, 0];
        const second = rgbs[nextIndex] ?? first;
        const rgb = first.map((component, dim) => (1 - inter) * component + inter * (second[dim] ?? 0));
        return [...rgb, opacity];
      });
      return array(new_rgbas.length > 0 ? new_rgbas : [[]]) as NDArray;
    };
    return func;
  }
}

export interface StreamLinesOptions extends VectorFieldOptions {
  x_range?: number[] | null;
  y_range?: number[] | null;
  z_range?: number[] | null;
  three_dimensions?: boolean;
  noise_factor?: number | null;
  n_repeats?: number;
  dt?: number;
  virtual_time?: number;
  max_anchors_per_line?: number;
  padding?: number;
  stroke_width?: number;
  opacity?: number;
}

export class StreamLines extends VectorField {
  x_range: number[];
  y_range: number[];
  z_range: number[];
  ranges: number[][];
  noise_factor: number;
  n_repeats: number;
  virtual_time: number;
  max_anchors_per_line: number;
  padding: number;
  stroke_width: number;
  opacity: number;
  stream_lines: VMobject[];
  flow_animation: ((mob: StreamLines, dt: number) => void) | null = null;
  flow_speed = 1;
  time_width = 0.3;
  outside_box: (point: number[] | NDArray) => boolean;
  updater: ((mob: StreamLines, dt: number) => void) | null = null;
  hide_and_wait: ((mob: Mobject, alpha: number) => void) | null = null;

  constructor(func: VectorFunction, options: StreamLinesOptions = {}) {
    super(func, options);
    this.x_range = options.x_range && options.x_range.length > 0 ? [...options.x_range] : [Math.floor(-config.frame_width / 2), Math.ceil(config.frame_width / 2)];
    this.y_range = options.y_range && options.y_range.length > 0 ? [...options.y_range] : [Math.floor(-config.frame_height / 2), Math.ceil(config.frame_height / 2)];
    this.ranges = [this.x_range, this.y_range];
    if (options.three_dimensions || (options.z_range && options.z_range.length > 0)) {
      this.z_range = options.z_range && options.z_range.length > 0 ? [...options.z_range] : [...this.y_range];
      this.ranges.push(this.z_range);
    } else {
      this.z_range = [0, 0];
      this.ranges.push(this.z_range);
    }
    for (const range of this.ranges) {
      if (range.length === 2) range.push(0.5);
      range[1] += range[2];
    }
    this.x_range = this.ranges[0];
    this.y_range = this.ranges[1];
    this.z_range = this.ranges[2];
    this.noise_factor = options.noise_factor ?? this.y_range[2] / 2;
    this.n_repeats = options.n_repeats ?? 1;
    this.virtual_time = options.virtual_time ?? 3;
    this.max_anchors_per_line = options.max_anchors_per_line ?? 100;
    this.padding = options.padding ?? 3;
    this.stroke_width = options.stroke_width ?? 1;
    this.opacity = options.opacity ?? 1;
    this.stream_lines = [];
    const stream = this;
    function outside_box(p: number[] | NDArray): boolean {
      const point = vectorOf(p);
      return point[0] < stream.x_range[0] - stream.padding || point[0] > stream.x_range[1] + stream.padding - stream.x_range[2] || point[1] < stream.y_range[0] - stream.padding || point[1] > stream.y_range[1] + stream.padding - stream.y_range[2] || point[2] < stream.z_range[0] - stream.padding || point[2] > stream.z_range[1] + stream.padding - stream.z_range[2];
    }
    this.outside_box = outside_box;
    this.generate_stream_lines(options.dt ?? 0.05);
  }

  private generate_stream_lines(dt: number): void {
    const maxSteps = Math.ceil(this.virtual_time / dt) + 1;
    const startPoints: number[][] = [];
    for (let repeat = 0; repeat < this.n_repeats; repeat += 1) {
      for (const x of arange(this.x_range[0], this.x_range[1], this.x_range[2])) {
        for (const y of arange(this.y_range[0], this.y_range[1], this.y_range[2])) {
          for (const z of arange(this.z_range[0], this.z_range[1], this.z_range[2])) startPoints.push([x, y, z]);
        }
      }
    }
    for (const start of startPoints) {
      const points: number[][] = [start];
      for (let step = 0; step < maxSteps; step += 1) {
        const last = points[points.length - 1];
        const next = addPoints(last, scalePoint(vectorOf(this.func(last)), dt, 3), 3);
        if (this.outside_box(next)) break;
        points.push(next);
      }
      const line = new VMobject({ stroke_width: this.stroke_width });
      if (this.single_color) line.set_stroke(this.color, this.stroke_width, this.opacity);
      else line.set_stroke(undefined, this.stroke_width, this.opacity);
      const step = Math.max(1, Math.floor(points.length / this.max_anchors_per_line));
      line.set_points_smoothly(points.filter((_, index) => index % step === 0));
      (line as VMobject & { duration: number }).duration = maxSteps * dt;
      this.stream_lines.push(line);
      this.add(line);
    }
  }

  start_animation(warm_up = true, flow_speed = 1, time_width = 0.3, line_animation_class: any = null, options: AnimationOptions = {}): this {
    for (const line of this.stream_lines) {
      const duration = (line as VMobject & { duration?: number }).duration ?? this.virtual_time;
      const AnimationClass = line_animation_class ?? Animation;
      const animation = new AnimationClass(line, { run_time: duration / flow_speed, rate_func: options.rate_func ?? linear, ...options });
      (animation as Animation & { time_width?: number }).time_width = time_width;
      animation.begin();
      (line as VMobject & { time?: number; anim?: Animation }).anim = animation;
      (line as VMobject & { time?: number }).time = Math.random() * this.virtual_time;
      if (warm_up) (line as VMobject & { time?: number }).time = -(line as VMobject & { time?: number }).time!;
    }
    const stream = this;
    function updater(mob: StreamLines, dt: number): void {
      for (const line of mob.stream_lines) {
        const state = line as VMobject & { time: number; anim: Animation };
        state.time += dt * flow_speed;
        if (state.time >= stream.virtual_time) state.time -= stream.virtual_time;
        state.anim.interpolate(Math.min(Math.max(state.time / state.anim.run_time, 0), 1));
      }
    }
    this.updater = updater;
    this.add_updater(this.updater as (mob: Mobject, dt: number) => void);
    this.flow_animation = this.updater;
    this.flow_speed = flow_speed;
    this.time_width = time_width;
    return this;
  }

  end_animation(): AnimationGroup {
    if (this.flow_animation === null) {
      const error = new Error("You have to start the animation before fading it out.");
      error.name = "ValueError";
      throw error;
    }
    const stream = this;
    const dt = 1 / (config.frame_rate as number);

    function hide_and_wait(mob: Mobject, alpha: number): void {
      if (alpha === 0) mob.set_stroke(undefined, undefined, 0);
      else if (alpha === 1) mob.set_stroke(undefined, undefined, 1);
    }

    function finish_updater_cycle(line: VMobject, alpha: number): void {
      const state = line as VMobject & { time: number; anim: Animation };
      state.time += dt * stream.flow_speed;
      state.anim.interpolate(Math.min(state.time / state.anim.run_time, 1));
      if (alpha === 1) {
        stream.remove(state.anim.mobject);
        state.anim.finish();
      }
    }

    this.hide_and_wait = hide_and_wait;
    const max_run_time = this.virtual_time / this.flow_speed;
    const creation_rate_func = ease_out_sine;
    const creation_staring_speed = (creation_rate_func(0.001) as number) * 1000;
    const creation_run_time =
      (max_run_time / (1 + this.time_width)) * creation_staring_speed;
    // creation_run_time is calculated so that the creation animation starts at the same speed
    // as the regular line flash animation but eases out.

    const animations: Animation[] = [];
    if (this.flow_animation) this.remove_updater(this.flow_animation as (mob: Mobject, dt: number) => void);
    this.flow_animation = null;

    for (const line of this.stream_lines) {
      const state = line as VMobject & { time: number; anim: Animation };
      const create = new Create(line, { run_time: creation_run_time, rate_func: creation_rate_func });
      if (state.time <= 0) {
        animations.push(
          new Succession(
            new UpdateFromAlphaFunc(line, hide_and_wait as (mobject: Mobject, ...args: unknown[]) => void, { run_time: -state.time / this.flow_speed }),
            create,
          ),
        );
        this.remove(state.anim.mobject);
        state.anim.finish();
      } else {
        const remaining_time = max_run_time - state.time / this.flow_speed;
        animations.push(
          new Succession(
            new UpdateFromAlphaFunc(line, finish_updater_cycle as (mobject: Mobject, ...args: unknown[]) => void, { run_time: remaining_time }),
            create,
          ),
        );
      }
    }
    return new AnimationGroup(...animations);
  }
}

type MobjectUpdater = (mob: Mobject, dt: number) => void;
