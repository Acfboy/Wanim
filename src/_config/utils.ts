import { accessSync, constants as fsConstants } from "node:fs";
import { QUALITIES, DEFAULT_QUALITY, RendererType, ORIGIN, UP, DOWN, RIGHT, LEFT, IN, OUT, X_AXIS, Y_AXIS, Z_AXIS, UL, UR, DL, DR } from "../constants.js";
import { ManimColor, BLACK, type ParsableManimColor } from "../utils/color/core.js";

export const MANIM_CONFIG_OPTIONS = [
  "assets_dir", "background_color", "background_opacity", "custom_folders", "disable_caching", "disable_caching_warning", "dry_run", "encoder_queue_size", "enable_wireframe", "ffmpeg_loglevel", "format", "flush_cache", "frame_height", "frame_rate", "frame_width", "frame_x_radius", "frame_y_radius", "from_animation_number", "images_dir", "input_file", "media_embed", "media_width", "log_dir", "log_to_file", "max_files_cached", "max_inflight_encoders", "media_dir", "movie_file_extension", "notify_outdated_version", "output_file", "partial_movie_dir", "pixel_height", "pixel_width", "plugins", "preview", "progress_bar", "quality", "save_as_gif", "save_sections", "save_last_frame", "save_pngs", "scene_names", "seed", "show_in_file_browser", "tex_dir", "tex_template", "tex_template_file", "text_dir", "upto_animation_number", "renderer", "enable_gui", "gui_location", "use_projection_fill_shaders", "use_projection_stroke_shaders", "verbosity", "video_dir", "sections_dir", "fullscreen", "window_position", "window_size", "window_monitor", "write_all", "write_to_movie", "zero_pad", "force_window", "no_latex_cleanup", "preview_command",
] as const;

export type ManimConfigKey = (typeof MANIM_CONFIG_OPTIONS)[number] | string;

const defaults: Record<string, unknown> = {
  assets_dir: "./",
  background_color: BLACK,
  background_opacity: 1,
  custom_folders: false,
  disable_caching: false,
  disable_caching_warning: false,
  dry_run: false,
  encoder_queue_size: 8,
  enable_wireframe: false,
  ffmpeg_loglevel: "error",
  format: "mp4",
  flush_cache: false,
  frame_height: 8,
  frame_rate: 60,
  frame_width: 14.222222222222221,
  frame_x_radius: 7.111111111111111,
  frame_y_radius: 4,
  from_animation_number: 0,
  images_dir: "./media/images",
  input_file: "",
  media_embed: false,
  media_width: 0,
  log_dir: "./media/logs",
  log_to_file: false,
  max_files_cached: 100,
  max_inflight_encoders: 1,
  media_dir: "./media",
  movie_file_extension: ".mp4",
  notify_outdated_version: true,
  output_file: "",
  partial_movie_dir: "./media/videos",
  pixel_height: 1080,
  pixel_width: 1920,
  plugins: [],
  preview: false,
  progress_bar: "display",
  quality: DEFAULT_QUALITY,
  save_as_gif: false,
  save_sections: false,
  save_last_frame: false,
  save_pngs: false,
  scene_names: [],
  seed: null,
  show_in_file_browser: false,
  tex_dir: "./media/Tex",
  tex_template: "",
  tex_template_file: "",
  text_dir: "./media/texts",
  upto_animation_number: -1,
  renderer: RendererType.CAIRO,
  enable_gui: false,
  gui_location: "0,0",
  use_projection_fill_shaders: false,
  use_projection_stroke_shaders: false,
  verbosity: "INFO",
  video_dir: "./media/videos",
  sections_dir: "./media/sections",
  fullscreen: false,
  window_position: "UR",
  window_size: "default",
  window_monitor: 0,
  write_all: false,
  write_to_movie: true,
  zero_pad: 4,
  force_window: false,
  no_latex_cleanup: false,
  preview_command: "",
};

function pythonListRepr(values: unknown[]): string {
  const render = (value: unknown): string => {
    if (typeof value === "string") return `'${value}'`;
    if (value === null || value === undefined) return "None";
    if (typeof value === "boolean") return value ? "True" : "False";
    return String(value);
  };
  return `[${values.map(render).join(", ")}]`;
}

function cloneValue(value: unknown): unknown {
  if (value instanceof ManimColor) return new ManimColor(value);
  if (Array.isArray(value)) return value.map(cloneValue);
  if (value && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) result[key] = cloneValue(item);
    return result;
  }
  return value;
}

export class ManimConfig {
  private readonly values: Record<string, unknown>;

  constructor(initializeDefaults = false) {
    this.values = {};
    for (const key of MANIM_CONFIG_OPTIONS) this.values[key] = initializeDefaults ? cloneValue(defaults[key]) : null;
  }

  __getitem__(key: string): any {
    if (typeof key !== "string") {
      const error = new TypeError(`attribute name must be string, not '${typeof key}'`);
      throw error;
    }
    if (Object.prototype.hasOwnProperty.call(this.values, key)) return this.values[key];
    if (key === "frame_size") return [this.pixel_width, this.pixel_height];
    if (key === "aspect_ratio") return this.pixel_height === 0 ? 0 : this.pixel_width / this.pixel_height;
    if (key === "top") return this.frame_y_radius;
    if (key === "bottom") return -this.frame_y_radius;
    if (key === "left_side") return -this.frame_x_radius;
    if (key === "right_side") return this.frame_x_radius;
    if (key === "frame_center") return [0, 0, 0];
    const error = new Error(`'ManimConfig' object has no attribute '${key}'`);
    error.name = "AttributeError";
    throw error;
  }

  __setitem__(key: string, value: unknown): void {
    if (!Object.prototype.hasOwnProperty.call(this.values, key)) {
      const error = new Error(`'ManimConfig' object has no attribute '${key}'`);
      error.name = "AttributeError";
      throw error;
    }
    if (key === "frame_height") { this.frame_height = Number(value); return; }
    if (key === "frame_width") { this.frame_width = Number(value); return; }
    if (key === "frame_x_radius") { this.frame_x_radius = Number(value); return; }
    if (key === "frame_y_radius") { this.frame_y_radius = Number(value); return; }
    if (key === "background_color") { this.background_color = value as ParsableManimColor; return; }
    if (key === "background_opacity") { this.background_opacity = Number(value); return; }
    if (key === "quality") { this.quality = value as string; return; }
    if (key === "renderer") { this.renderer = value as RendererType; return; }
    this.values[key] = value;
  }

  get(key: string): any {
    return this.__getitem__(key);
  }

  set(key: string, value: unknown): void {
    this.__setitem__(key, value);
  }

  get background_color(): ManimColor {
    return this.__getitem__("background_color");
  }

  set background_color(value: ParsableManimColor) {
    this.values.background_color = new ManimColor(value);
  }

  get background_opacity(): number {
    return this.__getitem__("background_opacity");
  }

  set background_opacity(value: number) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      const error = new Error("background_opacity must be 0 <= background_opacity <= 1");
      error.name = "ValueError";
      throw error;
    }
    this.values.background_opacity = value;
  }

  get frame_height(): number {
    return this.__getitem__("frame_height");
  }

  set frame_height(value: number) {
    this.values.frame_height = value;
    this.values.frame_y_radius = value / 2;
  }

  get frame_width(): number {
    return this.__getitem__("frame_width");
  }

  set frame_width(value: number) {
    this.values.frame_width = value;
    this.values.frame_x_radius = value / 2;
  }

  get frame_x_radius(): number {
    return this.__getitem__("frame_x_radius");
  }

  set frame_x_radius(value: number) {
    this.values.frame_x_radius = value;
    this.values.frame_width = value * 2;
  }

  get frame_y_radius(): number {
    return this.__getitem__("frame_y_radius");
  }

  set frame_y_radius(value: number) {
    this.values.frame_y_radius = value;
    this.values.frame_height = value * 2;
  }

  get renderer(): RendererType {
    return this.__getitem__("renderer");
  }

  set renderer(value: RendererType | string) {
    this.values.renderer = value === "cairo" || value === "opengl" ? value as RendererType : RendererType.CAIRO;
  }

  get quality(): string | null {
    const keys = ["pixel_width", "pixel_height", "frame_rate"];
    const current: Record<string, unknown> = {};
    for (const key of keys) current[key] = this.values[key];
    for (const qual of Object.keys(QUALITIES)) {
      const expected = QUALITIES[qual]!;
      if (keys.every((key) => current[key] === (expected as unknown as Record<string, unknown>)[key])) {
        return qual;
      }
    }
    return null;
  }

  set quality(value: string | null) {
    if (value === null || value === undefined) {
      return;
    }
    if (!(value in QUALITIES)) {
      const error = new Error(`quality must be one of ${JSON.stringify(Object.keys(QUALITIES))}`);
      error.name = "KeyError";
      throw error;
    }
    const q = QUALITIES[value]!;
    this.frame_size = [q.pixel_width, q.pixel_height];
    this.frame_rate = q.frame_rate;
  }

  _set_from_list(key: string, val: unknown, values: unknown[]): void {
    if ((values as unknown[]).includes(val)) {
      this.values[key] = val;
    } else {
      // Python interpolates the list with str(values), i.e. repr() of each
      // element: "['mp4', 'gif']". Mirror that exactly.
      const error = new Error(`attempted to set ${key} to ${String(val)}; must be in ${pythonListRepr(values)}`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_from_enum(key: string, enum_value: unknown, enum_class: Record<string, unknown> & ((value: unknown) => unknown)): void {
    try {
      this.values[key] = (enum_class as (value: unknown) => unknown)(enum_value);
    } catch {
      const error = new Error(`${String(enum_value)} is not a valid ${(enum_class as { name?: string }).name ?? "enum"}`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_boolean(key: string, val: unknown): void {
    // Python uses `if val in [True, False]` (== comparison), so 1/0 and 1.0/0.0
    // are accepted as booleans too. Mirror that with loose matching.
    if (val === true || val === false || val === 1 || val === 0) {
      this.values[key] = val;
    } else {
      const error = new Error(`${key} must be boolean`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_tuple(key: string, val: unknown): void {
    if (Array.isArray(val)) {
      this.values[key] = val;
    } else {
      const error = new Error(`${key} must be tuple`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_str(key: string, val: unknown): void {
    if (typeof val === "string") {
      this.values[key] = val;
    } else if (!val) {
      this.values[key] = "";
    } else {
      const error = new Error(`${key} must be str or falsy value`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_between(key: string, val: number, lo: number, hi: number): void {
    if (lo <= val && val <= hi) {
      this.values[key] = val;
    } else {
      const error = new Error(`${key} must be ${String(lo)} <= ${key} <= ${String(hi)}`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_int_between(key: string, val: number, lo: number, hi: number): void {
    if (lo <= val && val <= hi) {
      this.values[key] = val;
    } else {
      const error = new Error(`${key} must be an integer such that ${String(lo)} <= ${key} <= ${String(hi)}`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_pos_number(key: string, val: number, allow_inf: boolean): void {
    if (Number.isInteger(val) && val > -1) {
      this.values[key] = val;
    } else if (allow_inf && (val === -1 || val === Infinity)) {
      this.values[key] = Infinity;
    } else {
      const error = new Error(`${key} must be a non-negative integer (use -1 for infinity)`);
      error.name = "ValueError";
      throw error;
    }
  }

  _set_dir(key: string, val: string | { toString(): string }): void {
    if (typeof val === "string") {
      this.values[key] = val;
    } else {
      this.values[key] = String(val);
    }
  }

  get dry_run(): boolean {
    return this.__getitem__("dry_run");
  }

  set dry_run(val: boolean) {
    this.values["dry_run"] = val;
    if (val) {
      this.write_to_movie = false;
      this.values["write_all"] = false;
      this.values["save_last_frame"] = false;
      this.values["format"] = null;
    }
  }

  get frame_size(): [number, number] {
    return [this.__getitem__("pixel_width"), this.__getitem__("pixel_height")];
  }

  set frame_size(value: readonly [number, number] | [number, number]) {
    this.values["pixel_width"] = value[0];
    this.values["pixel_height"] = value[1];
  }

  get encoder_queue_size(): number {
    return this.__getitem__("encoder_queue_size");
  }

  set encoder_queue_size(value: number) {
    if (Number.isInteger(value) && value >= 1) {
      this.values["encoder_queue_size"] = value;
    } else {
      const error = new Error("encoder_queue_size must be a positive integer");
      error.name = "ValueError";
      throw error;
    }
  }

  get max_inflight_encoders(): number {
    return this.__getitem__("max_inflight_encoders");
  }

  set max_inflight_encoders(value: number) {
    if (Number.isInteger(value) && value >= 1) {
      this.values["max_inflight_encoders"] = value;
    } else {
      const error = new Error("max_inflight_encoders must be a positive integer");
      error.name = "ValueError";
      throw error;
    }
  }

  get from_animation_number(): number {
    return this.__getitem__("from_animation_number");
  }

  set from_animation_number(value: number) {
    this.values["from_animation_number"] = value;
  }

  get scene_names(): string[] {
    return this.__getitem__("scene_names");
  }

  set scene_names(value: string[]) {
    this.values["scene_names"] = value;
  }

  get plugins(): string[] {
    return this.__getitem__("plugins");
  }

  set plugins(value: string[]) {
    this.values["plugins"] = value;
  }

  get window_size(): string | number[] {
    return this.__getitem__("window_size");
  }

  set window_size(value: string | number[]) {
    this.values["window_size"] = value;
  }

  get window_position(): string {
    return this.__getitem__("window_position");
  }

  set window_position(value: string) {
    this.values["window_position"] = value;
  }

  get tex_template_file(): string {
    return this.__getitem__("tex_template_file");
  }

  set tex_template_file(val: string) {
    if (val) {
      try {
        accessSync(val, fsConstants.R_OK);
        this.values["tex_template_file"] = val;
      } catch {
        console.warn(`Custom TeX template ${val} not found or not readable.`);
      }
    } else {
      this.values["tex_template_file"] = val;
    }
  }

  get pixel_height(): number { return this.__getitem__("pixel_height"); }
  set pixel_height(value: number) { this.__setitem__("pixel_height", value); }
  get pixel_width(): number { return this.__getitem__("pixel_width"); }
  set pixel_width(value: number) { this.__setitem__("pixel_width", value); }
  get frame_rate(): number { return this.__getitem__("frame_rate"); }
  set frame_rate(value: number) { this.__setitem__("frame_rate", value); }
  get seed(): unknown { return this.__getitem__("seed"); }
  set seed(value: unknown) { this.__setitem__("seed", value); }
  get verbosity(): string { return this.__getitem__("verbosity"); }
  set verbosity(value: string) { this.__setitem__("verbosity", value); }
  get write_to_movie(): boolean { return this.__getitem__("write_to_movie"); }
  set write_to_movie(value: boolean) { this.__setitem__("write_to_movie", value); }
  get preview(): boolean { return this.__getitem__("preview"); }
  set preview(value: boolean) { this.__setitem__("preview", value); }

  getDir(key: string): string {
    const value = this.__getitem__(key);
    return typeof value === "string" ? value : String(value ?? "");
  }

  copy(): ManimConfig {
    const result = new ManimConfig(false);
    for (const key of Object.keys(this.values)) result.values[key] = cloneValue(this.values[key]);
    return result;
  }
}

export const config = new ManimConfig(true);

export function config_file_paths(): string[] {
  const runtime = globalThis as typeof globalThis & { process?: { platform?: string; env?: Record<string, string | undefined> } };
  const platform = runtime.process?.platform ?? "linux";
  const home = runtime.process?.env?.HOME ?? runtime.process?.env?.USERPROFILE ?? "";
  const libraryWide = decodeURIComponent(new URL("../../src/_config/default.cfg", import.meta.url).pathname);
  const userWide = platform.startsWith("win32")
    ? `${home}/AppData/Roaming/Manim/manim.cfg`
    : `${home}/.config/manim/manim.cfg`;
  return [libraryWide, userWide, "manim.cfg"];
}

export function _determine_quality(qual: string | null | undefined): string | null {
  for (const [quality, values] of Object.entries(QUALITIES)) {
    if (values.flag !== null && values.flag === qual) return quality;
  }
  return qual ?? null;
}

export class ManimFrame {
  private readonly configObject: ManimConfig;
  constructor(configObject: ManimConfig = config) {
    if (!(configObject instanceof ManimConfig)) throw new TypeError("argument must be instance of 'ManimConfig'");
    this.configObject = configObject;
  }

  __getitem__(key: string): any {
    if (key === "top") return 1;
    if (key === "bottom") return -1;
    if (key === "left_side") return -1;
    if (key === "right_side") return 1;
    if (key === "frame_x_radius" || key === "frame_y_radius" || key === "frame_width" || key === "frame_height") return this.configObject[key];
    const constants: Record<string, unknown> = { UP, DOWN, RIGHT, LEFT, IN, OUT, ORIGIN, X_AXIS, Y_AXIS, Z_AXIS, UL, UR, DL, DR };
    if (Object.prototype.hasOwnProperty.call(constants, key)) return constants[key];
    throw new Error(String(key));
  }
}
