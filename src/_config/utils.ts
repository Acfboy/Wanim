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

  get quality(): string {
    return this.__getitem__("quality");
  }

  set quality(value: string) {
    this.__setitem__("quality", _determine_quality(value));
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
