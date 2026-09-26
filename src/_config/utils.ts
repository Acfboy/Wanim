import { accessSync, constants as fsConstants, existsSync, readFileSync, statSync } from "node:fs";
import { QUALITIES, DEFAULT_QUALITY, RendererType, ORIGIN, UP, DOWN, RIGHT, LEFT, IN, OUT, X_AXIS, Y_AXIS, Z_AXIS, UL, UR, DL, DR } from "../constants.js";
import { ManimColor, BLACK, type ParsableManimColor } from "../utils/color/core.js";

export const MANIM_CONFIG_OPTIONS = [
  "assets_dir", "background_color", "background_opacity", "custom_folders", "disable_caching", "disable_caching_warning", "dry_run", "encoder_queue_size", "enable_wireframe", "ffmpeg_loglevel", "format", "flush_cache", "frame_height", "frame_rate", "frame_width", "frame_x_radius", "frame_y_radius", "from_animation_number", "images_dir", "input_file", "media_embed", "media_width", "log_dir", "log_to_file", "max_files_cached", "max_inflight_encoders", "media_dir", "movie_file_extension", "notify_outdated_version", "output_file", "partial_movie_dir", "pixel_height", "pixel_width", "plugins",   "preview", "progress_bar", "quality", "save_as_gif", "save_sections", "save_last_frame", "save_pngs", "scene_names", "seed", "show_in_file_browser",    "tex_dir", "tex_template", "tex_template_file", "text_dir", "upto_animation_number", "renderer", "enable_gui", "gui_location", "use_projection_fill_shaders", "use_projection_stroke_shaders", "verbosity", "video_dir", "sections_dir", "fullscreen", "window_position", "window_size", "window_monitor", "write_all", "write_to_movie", "zero_pad", "force_window", "no_latex_cleanup", "preview_command",
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

function keyError(message: string): Error {
  const error = new Error(message);
  error.name = "KeyError";
  return error;
}

function purePosixNormalize(path: string): string {
  // NOTE: mirrors pathlib.PurePosixPath: single dots and empty segments
  // (from "./" prefixes or doubled slashes) are dropped, ".." is kept
  // literally, and a leading "/" marks an absolute path.
  if (path === "") return path;
  const absolute = path.startsWith("/");
  const kept: string[] = [];
  for (const segment of path.split("/")) {
    if (segment === "" || segment === ".") continue;
    kept.push(segment);
  }
  const joined = kept.join("/");
  if (absolute) return `/${joined}`;
  return joined === "" ? "." : joined;
}

function formatG(value: number): string {  // NOTE: mirrors Python f"{value:g}": general format with 6 significant
  // digits, trailing zeros stripped, fixed notation for small exponents.
  if (!Number.isFinite(value)) {
    if (Number.isNaN(value)) return "nan";
    return value > 0 ? "inf" : "-inf";
  }
  if (value === 0) return "0";
  let text = value.toPrecision(6);
  if (text.includes("e") || text.includes("E")) {
    text = text.replace(/(\.\d*?)0+e/, "$1e").replace(/\.e/, "e");
  } else if (text.includes(".")) {
    text = text.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  }
  return text;
}

export interface ManimConfigCliArgs {
  file?: string | { suffix?: string; toString(): string } | null;
  config_file?: string | null;
  scene_names?: string[] | null;
  output_file?: string;
  from_animation_number?: number[] | null;
  quality?: string | null;
  resolution?: number[] | null;
  frame_rate?: number | null;
  custom_folders?: boolean;
  media_dir?: string | null;
  tex_template?: string | null;
  write_to_movie?: boolean | null;
  gui_location?: unknown;
  [key: string]: unknown;
}

export class ManimConfig {
  private readonly values: Record<string, unknown>;
  private parserData: ConfigParserData | null = null;

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
    if (key === "top") return this.top;
    if (key === "bottom") return this.bottom;
    if (key === "left_side") return this.left_side;
    if (key === "right_side") return this.right_side;
    if (key === "frame_center") return [0, 0, 0];
    if (key === "transparent") return this.transparent;
    const error = new Error(`'ManimConfig' object has no attribute '${key}'`);
    error.name = "AttributeError";
    throw error;
  }

  __setitem__(key: string, value: unknown): void {
    // NOTE: `transparent` is a derived property (see its accessor), not a
    // stored option, so it is handled before the stored-key guard — mirroring
    // Python, where __setitem__ dispatches through the property setter.
    if (key === "transparent") { this.transparent = value as boolean; return; }
    if (!Object.prototype.hasOwnProperty.call(this.values, key)) {
      const error = new Error(`'ManimConfig' object has no attribute '${key}'`);
      error.name = "AttributeError";
      throw error;
    }
    if (key === "frame_height") { this.frame_height = Number(value); return; }
    if (key === "frame_width") { this.frame_width = Number(value); return; }
    if (key === "frame_x_radius") { this.frame_x_radius = Number(value); return; }
    if (key === "frame_y_radius") { this.frame_y_radius = Number(value); return; }
    if (key === "frame_size") { this.frame_size = value as [number, number]; return; }
    if (key === "background_color") { this.background_color = value as ParsableManimColor; return; }
    if (key === "background_opacity") { this.background_opacity = Number(value); return; }
    if (key === "quality") { this.quality = value as string; return; }
    if (key === "renderer") { this.renderer = value as RendererType; return; }
    if (key === "pixel_height") { this.pixel_height = value as number; return; }
    if (key === "pixel_width") { this.pixel_width = value as number; return; }
    if (key === "seed") { this.seed = value as number | null; return; }
    if (key === "verbosity") { this.verbosity = value as string; return; }
    if (key === "write_to_movie") { this.write_to_movie = value as boolean; return; }
    if (key === "preview") { this.preview = value as boolean; return; }
    if (key === "show_in_file_browser") { this.show_in_file_browser = value as boolean; return; }
    if (key === "progress_bar") { this.progress_bar = value as string; return; }
    if (key === "log_to_file") { this.log_to_file = value as boolean; return; }
    if (key === "notify_outdated_version") { this.notify_outdated_version = value as boolean; return; }
    if (key === "save_last_frame") { this.save_last_frame = value as boolean; return; }
    if (key === "write_all") { this.write_all = value as boolean; return; }
    if (key === "save_pngs") { this.save_pngs = value as boolean; return; }
    if (key === "save_as_gif") { this.save_as_gif = value as boolean; return; }
    if (key === "save_sections") { this.save_sections = value as boolean; return; }
    if (key === "enable_wireframe") { this.enable_wireframe = value as boolean; return; }
    if (key === "force_window") { this.force_window = value as boolean; return; }
    if (key === "no_latex_cleanup") { this.no_latex_cleanup = value as boolean; return; }
    if (key === "preview_command") { this.preview_command = value as string; return; }
    if (key === "upto_animation_number") { this.upto_animation_number = value as number; return; }
    if (key === "max_files_cached") { this.max_files_cached = value as number; return; }
    if (key === "window_monitor") { this.window_monitor = value as number; return; }
    if (key === "flush_cache") { this.flush_cache = value as boolean; return; }
    if (key === "disable_caching") { this.disable_caching = value as boolean; return; }
    if (key === "disable_caching_warning") { this.disable_caching_warning = value as boolean; return; }
    if (key === "enable_gui") { this.enable_gui = value as boolean; return; }
    if (key === "gui_location") { this.gui_location = value; return; }
    if (key === "fullscreen") { this.fullscreen = value as boolean; return; }
    if (key === "use_projection_fill_shaders") { this.use_projection_fill_shaders = value as boolean; return; }
    if (key === "use_projection_stroke_shaders") { this.use_projection_stroke_shaders = value as boolean; return; }
    if (key === "zero_pad") { this.zero_pad = value as number; return; }
    if (key === "assets_dir") { this.assets_dir = value as string; return; }
    if (key === "log_dir") { this.log_dir = value as string; return; }
    if (key === "sections_dir") { this.sections_dir = value as string; return; }
    if (key === "text_dir") { this.text_dir = value as string; return; }
    if (key === "tex_dir") { this.tex_dir = value as string; return; }
    if (key === "custom_folders") { this.custom_folders = value as string; return; }
    if (key === "input_file") { this.input_file = value as string; return; }
    if (key === "output_file") { this.output_file = value as string; return; }
    this.values[key] = value;
  }

  get(key: string): any {
    return this.__getitem__(key);
  }

  set(key: string, value: unknown): void {
    this.__setitem__(key, value);
  }

  update(obj: ManimConfig | Record<string, unknown>): void {
    if (obj instanceof ManimConfig) {
      const renderer = obj.renderer;
      Object.assign(this.values, obj.values);
      if (renderer !== null && renderer !== undefined) {
        // Apply the renderer through its setter so that the
        // ConvertToOpenGL class bases stay in sync with the config value.
        this.renderer = renderer;
      }
      // NOTE: the stored value is read directly because the tex_template
      // property getter is a text/tex stub (out of scope for the entire port).
      if (obj.values["tex_template"]) {
        this.tex_template = obj.values["tex_template"];
      }
    } else if (obj !== null && typeof obj === "object" && !Array.isArray(obj)) {
      // First update the underlying store, then other properties
      for (const [key, value] of Object.entries(obj)) {
        if (Object.prototype.hasOwnProperty.call(this.values, key)) {
          this.set(key, value);
        }
      }
      for (const [key, value] of Object.entries(obj)) {
        if (!Object.prototype.hasOwnProperty.call(this.values, key)) {
          this.set(key, value);
        }
      }
    }
  }

  // don't allow to delete anything
  __delitem__(key: string): never {
    void key;
    const error = new Error(
      "'ManimConfig' object does not support item deletion",
    );
    error.name = "AttributeError";
    throw error;
  }

  __delattr__(key: string): never {
    void key;
    const error = new Error(
      "'ManimConfig' object does not support item deletion",
    );
    error.name = "AttributeError";
    throw error;
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

  get top(): number[] {
    return [0, this.frame_y_radius, 0];
  }

  get bottom(): number[] {
    return [0, -this.frame_y_radius, 0];
  }

  get left_side(): number[] {
    return [-this.frame_x_radius, 0, 0];
  }

  get right_side(): number[] {
    return [this.frame_x_radius, 0, 0];
  }

  __contains__(key: unknown): boolean {
    try {
      if (typeof key !== "string") {
        const error = new Error("assert isinstance(key, str)");
        error.name = "AssertionError";
        throw error;
      }
      this.__getitem__(key);
      return true;
    } catch (error) {
      if ((error as Error).name === "AttributeError") return false;
      throw error;
    }
  }

  *[Symbol.iterator](): IterableIterator<string> {
    yield* Object.keys(this.values);
  }

  __len__(): number {
    return Object.keys(this.values).length;
  }

  __repr__(): string {
    let rep = "";
    for (const key of Object.keys(this.values).sort()) {
      rep += `${key}: ${String(this.values[key])}, `;
    }
    return rep;
  }

  toString(): string {
    return this.__repr__();
  }

  __copy__(): ManimConfig {
    return this.copy();
  }

  __deepcopy__(_memo?: Record<string, unknown>): ManimConfig {
    return this.copy();
  }

  get renderer(): RendererType {
    return this.__getitem__("renderer");
  }

  set renderer(value: RendererType | string) {
    // NOTE: mirrors Python `RendererType(value)` (lookup by value, raising
    // `ValueError: <value> is not a valid RendererType`): strings match
    // case-insensitively, enum members pass through, anything else throws.
    // Python additionally swaps Mobject/VMobject bases via the
    // ConvertToOpenGL metaclass here; neither the metaclass nor the OpenGL
    // classes are ported yet, so there are no bases to swap.
    const normalized = typeof value === "string" ? value.toLowerCase() : value;
    if (!Object.values(RendererType).includes(normalized as RendererType)) {
      const error = new Error(`${String(value)} is not a valid RendererType`);
      error.name = "ValueError";
      throw error;
    }
    this.values.renderer = normalized as RendererType;
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
      // Mirror Python _set_dir: Path objects become strings, everything else
      // is stored as-is (e.g. custom_folders may hold a boolean).
      this.values[key] = typeof val === "boolean" ? val : String(val);
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

  get tex_template(): unknown {
    throw new Error("TODO: text/tex not ported yet");
  }

  set tex_template(_val: unknown) {
    throw new Error("TODO: text/tex not ported yet");
  }

  digest_file(filename: string): this {
    let isFile = false;
    try {
      isFile = existsSync(filename) && statSync(filename).isFile();
    } catch {
      isFile = false;
    }
    if (!isFile) {
      const error = new Error(
        `[Errno 2] Error: --config_file could not find a valid config file: '${filename}'`,
      );
      (error as unknown as Record<string, unknown>)["errno"] = 2;
      error.name = "FileNotFoundError";
      throw error;
    }
    return this.digest_parser(make_config_parser(filename));
  }

  get pixel_height(): number { return this.__getitem__("pixel_height"); }
  set pixel_height(value: number) { this._set_pos_number("pixel_height", value, false); }
  get pixel_width(): number { return this.__getitem__("pixel_width"); }
  set pixel_width(value: number) { this._set_pos_number("pixel_width", value, false); }
  get frame_rate(): number { return this.__getitem__("frame_rate"); }
  set frame_rate(value: number) { this.__setitem__("frame_rate", value); }
  get seed(): unknown { return this.__getitem__("seed"); }
  set seed(value: number | null) {
    if (value === null || value === undefined) {
      return;
    }
    this._set_pos_number("seed", value, false);
  }
  get verbosity(): string { return this.__getitem__("verbosity"); }
  set verbosity(value: string) {
    this._set_from_list(
      "verbosity",
      value,
      ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    );
    // NOTE: Python also calls logger.setLevel(val) here. The TS port has no
    // shared global logger instance yet, so only the stored value is updated.
  }
  get write_to_movie(): boolean { return this.__getitem__("write_to_movie"); }
  set write_to_movie(value: boolean) { this._set_boolean("write_to_movie", value); }
  get preview(): boolean { return this.__getitem__("preview") || this.__getitem__("enable_gui"); }
  set preview(value: boolean) { this._set_boolean("preview", value); }
  get show_in_file_browser(): boolean { return this.__getitem__("show_in_file_browser"); }
  set show_in_file_browser(value: boolean) { this._set_boolean("show_in_file_browser", value); }
  get progress_bar(): string { return this.__getitem__("progress_bar"); }
  set progress_bar(value: string) { this._set_from_list("progress_bar", value, ["none", "display", "leave"]); }
  get log_to_file(): boolean { return this.__getitem__("log_to_file"); }
  set log_to_file(value: boolean) { this._set_boolean("log_to_file", value); }
  get notify_outdated_version(): boolean { return this.__getitem__("notify_outdated_version"); }
  set notify_outdated_version(value: boolean) { this._set_boolean("notify_outdated_version", value); }
  get save_last_frame(): boolean { return this.__getitem__("save_last_frame"); }
  set save_last_frame(value: boolean) { this._set_boolean("save_last_frame", value); }
  get write_all(): boolean { return this.__getitem__("write_all"); }
  set write_all(value: boolean) { this._set_boolean("write_all", value); }
  get save_pngs(): boolean { return this.__getitem__("save_pngs"); }
  set save_pngs(value: boolean) { this._set_boolean("save_pngs", value); }
  get save_as_gif(): boolean { return this.__getitem__("save_as_gif"); }
  set save_as_gif(value: boolean) { this._set_boolean("save_as_gif", value); }
  get save_sections(): boolean { return this.__getitem__("save_sections"); }
  set save_sections(value: boolean) { this._set_boolean("save_sections", value); }
  // NOTE: `transparent` is a derived property in the Python original
  // (background_opacity < 1.0), not a stored option: it is readable via
  // __getitem__/__contains__ but excluded from iteration and __len__.
  // The movie-extension resolution is a rendering concern and is not ported.
  get transparent(): boolean {
    const opacity: unknown = this.values["background_opacity"];
    if (typeof opacity !== "number") {
      const error = new Error("assert isinstance(background_opacity, float)");
      error.name = "AssertionError";
      throw error;
    }
    return opacity < 1;
  }
  set transparent(value: boolean) {
    this.background_opacity = value ? 0 : 1;
  }
  get enable_wireframe(): boolean { return this.__getitem__("enable_wireframe"); }
  set enable_wireframe(value: boolean) { this._set_boolean("enable_wireframe", value); }
  get force_window(): boolean { return this.__getitem__("force_window"); }
  set force_window(value: boolean) { this._set_boolean("force_window", value); }
  get no_latex_cleanup(): boolean { return this.__getitem__("no_latex_cleanup"); }
  set no_latex_cleanup(value: boolean) { this._set_boolean("no_latex_cleanup", value); }
  get preview_command(): string { return this.__getitem__("preview_command"); }
  set preview_command(value: string) { this._set_str("preview_command", value); }
  get upto_animation_number(): number { return this.__getitem__("upto_animation_number"); }
  set upto_animation_number(value: number) { this._set_pos_number("upto_animation_number", value, true); }
  get max_files_cached(): number { return this.__getitem__("max_files_cached"); }
  set max_files_cached(value: number) { this._set_pos_number("max_files_cached", value, true); }
  get window_monitor(): number { return this.__getitem__("window_monitor"); }
  set window_monitor(value: number) { this._set_pos_number("window_monitor", value, true); }
  get flush_cache(): boolean { return this.__getitem__("flush_cache"); }
  set flush_cache(value: boolean) { this._set_boolean("flush_cache", value); }
  get disable_caching(): boolean { return this.__getitem__("disable_caching"); }
  set disable_caching(value: boolean) { this._set_boolean("disable_caching", value); }
  get disable_caching_warning(): boolean { return this.__getitem__("disable_caching_warning"); }
  set disable_caching_warning(value: boolean) { this._set_boolean("disable_caching_warning", value); }
  get enable_gui(): boolean { return this.__getitem__("enable_gui"); }
  set enable_gui(value: boolean) { this._set_boolean("enable_gui", value); }
  get gui_location(): unknown { return this.__getitem__("gui_location"); }
  set gui_location(value: unknown) { this._set_tuple("gui_location", value); }
  get fullscreen(): boolean { return this.__getitem__("fullscreen"); }
  set fullscreen(value: boolean) { this._set_boolean("fullscreen", value); }
  get use_projection_fill_shaders(): boolean { return this.__getitem__("use_projection_fill_shaders"); }
  set use_projection_fill_shaders(value: boolean) { this._set_boolean("use_projection_fill_shaders", value); }
  get use_projection_stroke_shaders(): boolean { return this.__getitem__("use_projection_stroke_shaders"); }
  set use_projection_stroke_shaders(value: boolean) { this._set_boolean("use_projection_stroke_shaders", value); }
  get zero_pad(): number { return this.__getitem__("zero_pad"); }
  set zero_pad(value: number) { this._set_int_between("zero_pad", value, 0, 9); }
  get assets_dir(): string { return this.__getitem__("assets_dir"); }
  set assets_dir(value: string) { this._set_dir("assets_dir", value); }
  get log_dir(): string { return this.__getitem__("log_dir"); }
  set log_dir(value: string) { this._set_dir("log_dir", value); }
  get sections_dir(): string { return this.__getitem__("sections_dir"); }
  set sections_dir(value: string) { this._set_dir("sections_dir", value); }
  get text_dir(): string { return this.__getitem__("text_dir"); }
  set text_dir(value: string) { this._set_dir("text_dir", value); }
  get tex_dir(): string { return this.__getitem__("tex_dir"); }
  set tex_dir(value: string) { this._set_dir("tex_dir", value); }
  get custom_folders(): unknown { return this.__getitem__("custom_folders"); }
  set custom_folders(value: string) { this._set_dir("custom_folders", value); }
  get input_file(): string { return this.__getitem__("input_file"); }
  set input_file(value: string) { this._set_dir("input_file", value); }
  get output_file(): string { return this.__getitem__("output_file"); }
  set output_file(value: string) { this._set_dir("output_file", value); }

  getDir(key: string): string {
    const value = this.__getitem__(key);
    return typeof value === "string" ? value : String(value ?? "");
  }

  copy(): ManimConfig {
    const result = new ManimConfig(false);
    for (const key of Object.keys(this.values)) result.values[key] = cloneValue(this.values[key]);
    return result;
  }

  digest_parser(parser: ConfigParserData): this {
    this.parserData = parser;

    const sectionOf = (name: string): Map<string, string> => {
      const section = parser.get(name);
      if (!section) {
        // NOTE: mirrors configparser, where parser[name] raises KeyError
        // for a missing section.
        throw keyError(name);
      }
      return section;
    };
    const getOrFallback = (section: Map<string, string>, key: string, fallback: string): string => {
      const value = section.get(key);
      return value === undefined ? fallback : value;
    };
    const getRequired = (sectionName: string, key: string): string => {
      const section = sectionOf(sectionName);
      const value = section.get(key);
      if (value === undefined) {
        const error = new Error(`No option '${key}' in section: '${sectionName}'`);
        error.name = "NoOptionError";
        throw error;
      }
      return value;
    };
    const getBoolean = (sectionName: string, key: string, fallback = false): boolean => {
      const section = sectionOf(sectionName);
      const raw = section.get(key);
      if (raw === undefined) return fallback;
      const normalized = raw.trim().toLowerCase();
      if (["1", "yes", "true", "on"].includes(normalized)) return true;
      if (["0", "no", "false", "off"].includes(normalized)) return false;
      const error = new Error(`Not a boolean: ${raw}`);
      error.name = "ValueError";
      throw error;
    };
    const getInt = (sectionName: string, key: string): number | null => {
      const section = sectionOf(sectionName);
      const raw = section.get(key);
      if (raw === undefined) {
        // NOTE: mirrors SectionProxy.getint, whose default fallback is None:
        // a missing option yields None (assigned as-is below; e.g. seed=None
        // is ignored by the seed setter), it does NOT raise.
        return null;
      }
      if (!/^[+-]?\d+$/.test(raw.trim())) {
        const error = new Error(`invalid literal for int() with base 10: '${raw}'`);
        error.name = "ValueError";
        throw error;
      }
      return Number.parseInt(raw.trim(), 10);
    };
    const getFloat = (sectionName: string, key: string, fallback?: number): number | null => {
      const section = sectionOf(sectionName);
      const raw = section.get(key);
      if (raw === undefined) {
        // NOTE: same SectionProxy fallback=None reasoning as getInt above.
        if (fallback !== undefined) return fallback;
        return null;
      }
      const value = Number(raw);
      if (Number.isNaN(value)) {
        const error = new Error(`could not convert string to float: '${raw}'`);
        error.name = "ValueError";
        throw error;
      }
      return value;
    };

    // boolean keys
    for (const key of [
      "notify_outdated_version",
      "write_to_movie",
      "save_last_frame",
      "write_all",
      "save_pngs",
      "save_as_gif",
      "save_sections",
      "preview",
      "show_in_file_browser",
      "log_to_file",
      "disable_caching",
      "disable_caching_warning",
      "flush_cache",
      "custom_folders",
      "enable_gui",
      "fullscreen",
      "use_projection_fill_shaders",
      "use_projection_stroke_shaders",
      "enable_wireframe",
      "force_window",
      "no_latex_cleanup",
      "dry_run",
    ] as const) {
      const value = getBoolean("CLI", key, false);
      if (key === "custom_folders") {
        // NOTE: Python stores the raw boolean via _set_dir; the TS
        // custom_folders setter only accepts strings, so store directly.
        (this as unknown as { _set_dir(key: string, val: unknown): void })._set_dir(key, value);
      } else {
        (this as unknown as Record<string, unknown>)[key] = value;
      }
    }

    // int keys
    for (const key of [
      "from_animation_number",
      "upto_animation_number",
      "max_files_cached",
      "max_inflight_encoders",
      "encoder_queue_size",
      // the next two must be set BEFORE digesting frame_width and frame_height
      "pixel_height",
      "pixel_width",
      "seed",
      "window_monitor",
      "zero_pad",
    ] as const) {
      (this as unknown as Record<string, unknown>)[key] = getInt("CLI", key);
    }

    // str keys
    for (const key of [
      "assets_dir",
      "verbosity",
      "media_dir",
      "log_dir",
      "video_dir",
      "sections_dir",
      "images_dir",
      "text_dir",
      "tex_dir",
      "partial_movie_dir",
      "input_file",
      "output_file",
      "movie_file_extension",
      "background_color",
      "renderer",
      "window_position",
      "preview_command",
    ] as const) {
      const value = getOrFallback(sectionOf("CLI"), key, "");
      if (
        key === "media_dir" || key === "video_dir" || key === "images_dir" ||
        key === "partial_movie_dir" || key === "movie_file_extension"
      ) {
        // NOTE: these have no dedicated TS accessors; store generically.
        this.set(key, value);
      } else {
        (this as unknown as Record<string, unknown>)[key] = value;
      }
    }

    // float keys
    for (const key of ["background_opacity", "frame_rate"] as const) {
      (this as unknown as Record<string, unknown>)[key] = getFloat("CLI", key);
    }

    // tuple keys
    const gui_location = getRequired("CLI", "gui_location").split(/[;,\-]/).map((part) => Number.parseInt(part.trim(), 10));
    this.gui_location = gui_location;

    const window_size = getRequired("CLI", "window_size");
    if (window_size !== "default") {
      this.window_size = window_size.split(/[;,\-]/).map((part) => Number.parseInt(part.trim(), 10));
    } else {
      this.window_size = window_size;
    }

    // plugins
    const plugins = getOrFallback(sectionOf("CLI"), "plugins", "");
    this.plugins = plugins === "" ? [] : plugins.split(",");

    // the next two must be set AFTER digesting pixel_width and pixel_height
    this.set("frame_height", getFloat("CLI", "frame_height", 8.0));
    const frame_width_raw = sectionOf("CLI").get("frame_width");
    if (frame_width_raw === undefined || frame_width_raw === null || String(frame_width_raw).trim() === "" || String(frame_width_raw).trim().toLowerCase() === "none") {
      this.set("frame_width", (this.get("frame_height") as number) * (this.get("aspect_ratio") as number));
    } else {
      this.set("frame_width", Number(frame_width_raw));
    }

    // other logic
    const tex_template_file = sectionOf("CLI").get("tex_template_file");
    if (tex_template_file) this.tex_template_file = tex_template_file;

    const progress_bar = sectionOf("CLI").get("progress_bar");
    if (progress_bar) this.progress_bar = progress_bar;

    const ffmpeg_loglevel = sectionOf("ffmpeg").get("loglevel");
    if (ffmpeg_loglevel) {
      (this as unknown as { values: Record<string, unknown> }).values["ffmpeg_loglevel"] = ffmpeg_loglevel;
    }

    let media_embed: boolean | null;
    const raw_embed = sectionOf("jupyter").get("media_embed");
    if (raw_embed === undefined) {
      // NOTE: mirrors configparser.getboolean with no fallback, which raises
      // NoOptionError for a missing option (only ValueError is caught below
      // in Python, so this propagates there too).
      const error = new Error(`No option 'media_embed' in section: 'jupyter'`);
      error.name = "NoOptionError";
      throw error;
    }
    const normalized = raw_embed.trim().toLowerCase();
    if (["1", "yes", "true", "on"].includes(normalized)) media_embed = true;
    else if (["0", "no", "false", "off"].includes(normalized)) media_embed = false;
    else media_embed = null;
    this.set("media_embed", media_embed);

    const media_width = sectionOf("jupyter").get("media_width");
    if (media_width) this.set("media_width", media_width);

    const quality = getOrFallback(sectionOf("CLI"), "quality", "");
    if (quality) this.quality = _determine_quality(quality);

    return this;
  }

  digest_args(args: ManimConfigCliArgs): this {
    // NOTE: mirrors ManimConfig.digest_args (manim/_config/utils.py). The
    // `args` object stands in for argparse.Namespace (as returned by
    // main_utils.parse_args()); unknown keys ride along via the index
    // signature so the hasattr loops below behave like Python's.
    const file_suffix = typeof args.file === "string"
      ? (args.file.endsWith(".cfg") ? ".cfg" : "")
      : (args.file?.suffix ?? "");
    // if the input file is a config file, parse it properly
    if (file_suffix === ".cfg") args.config_file = String(args.file);

    // if args.file is `-`, the animation code has to be taken from STDIN, so
    // the input file path shouldn't be absolute, since that file won't be read.
    if (args.file !== undefined && args.file !== null && String(args.file) === "-") {
      this.input_file = String(args.file);
    }

    // if a config file has been passed, digest it first so that other CLI
    // flags supersede it
    if (args.config_file) this.digest_file(args.config_file);

    // read input_file from the args if it wasn't set by the config file
    if (!this.input_file) {
      const file_string = String(args.file);
      this.input_file = file_string.startsWith("/")
        ? file_string
        : `${process.cwd()}/${file_string}`;
    }

    this.scene_names = (args.scene_names ?? []) as string[];
    this.output_file = String(args.output_file ?? "");

    for (const key of [
      "notify_outdated_version",
      "preview",
      "show_in_file_browser",
      "write_to_movie",
      "save_last_frame",
      "save_pngs",
      "save_as_gif",
      "save_sections",
      "write_all",
      "disable_caching",
      "format",
      "flush_cache",
      "progress_bar",
      "transparent",
      "scene_names",
      "verbosity",
      "renderer",
      "background_color",
      "enable_gui",
      "fullscreen",
      "use_projection_fill_shaders",
      "use_projection_stroke_shaders",
      "zero_pad",
      "enable_wireframe",
      "force_window",
      "dry_run",
      "no_latex_cleanup",
      "preview_command",
      "seed",
      "max_inflight_encoders",
      "encoder_queue_size",
    ] as const) {
      if (Object.prototype.hasOwnProperty.call(args, key)) {
        const attr = args[key];
        // if attr is None, then no argument was passed and we should
        // not change the current config
        if (attr !== undefined && attr !== null) this.set(key, attr);
      }
    }

    for (const key of [
      "media_dir", // always set this one first
      "log_dir",
      "log_to_file", // always set this one last
    ] as const) {
      if (Object.prototype.hasOwnProperty.call(args, key)) {
        const attr = args[key];
        // if attr is None, then no argument was passed and we should
        // not change the current config
        if (attr !== undefined && attr !== null) this.set(key, attr);
      }
    }

    if (this.get("save_last_frame")) this.set("write_to_movie", false);

    // Handle the -n flag.
    const nflag = args.from_animation_number;
    if (nflag !== undefined && nflag !== null) {
      this.from_animation_number = nflag[0] ?? 0;
      if (nflag.length < 2 || nflag[1] === undefined) {
        console.info(
          `No end scene number specified in -n option. Rendering from ${nflag[0]} onwards...`,
        );
      } else {
        this.upto_animation_number = nflag[1] as number;
      }
    }

    // Handle the quality flags
    this.quality = _determine_quality((args.quality ?? null) as string | null);

    // Handle the -r flag.
    const rflag = args.resolution;
    if (rflag !== undefined && rflag !== null) {
      this.pixel_width = Number.parseInt(String(rflag[0]), 10);
      this.pixel_height = Number.parseInt(String(rflag[1]), 10);
    }

    const fps = args.frame_rate;
    if (fps !== undefined && fps !== null) this.frame_rate = Number(fps);

    // Handle --custom_folders
    if (args.custom_folders) {
      for (const opt of [
        "media_dir",
        "video_dir",
        "sections_dir",
        "images_dir",
        "text_dir",
        "tex_dir",
        "log_dir",
        "partial_movie_dir",
      ] as const) {
        this.set(opt, this.parserData?.get("custom_folders")?.get(opt) ?? null);
      }
      // --media_dir overrides the default.cfg file
      if (Object.prototype.hasOwnProperty.call(args, "media_dir") && args.media_dir) {
        this.set("media_dir", args.media_dir);
      }
    }

    // Handle --tex_template
    if (args.tex_template) {
      throw new Error("TODO: text/tex not ported yet (ManimConfig.digest_args --tex_template needs TexTemplate.from_file)");
    }

    if (this.renderer === RendererType.OPENGL && args.write_to_movie === null) {
      // --write_to_movie was not passed on the command line, so don't generate video.
      this.set("write_to_movie", false);
    }

    // Handle --gui_location flag.
    if (args.gui_location !== undefined && args.gui_location !== null) {
      this.gui_location = args.gui_location;
    }

    return this;
  }

  get_dir(key: string, kwargs: Record<string, unknown> = {}): string | null {
    const dirs = [
      "assets_dir",
      "media_dir",
      "video_dir",
      "sections_dir",
      "images_dir",
      "text_dir",
      "tex_dir",
      "log_dir",
      "input_file",
      "output_file",
      "partial_movie_dir",
    ];
    if (!dirs.includes(key)) {
      throw keyError(
        "must pass one of "
        + "{media,video,images,text,tex,log}_dir "
        + "or {input,output}_file",
      );
    }

    const remaining = dirs.filter((candidate) => candidate !== key);

    const all_args: Record<string, unknown> = {};
    for (const candidate of remaining) all_args[candidate] = this.get(candidate);
    for (const [name, value] of Object.entries(kwargs)) all_args[name] = value;
    all_args["quality"] = `${this.get("pixel_height")}p${formatG(this.get("frame_rate") as number)}`;

    const raw = this.get(key);
    if (typeof raw !== "string") {
      const error = new Error(`assert isinstance(path, str)`);
      error.name = "AssertionError";
      throw error;
    }
    let path = raw;
    const template = raw;
    while (path.includes("{")) {
      try {
        path = path.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_match, name: string) => {
          if (!Object.prototype.hasOwnProperty.call(all_args, name) || all_args[name] === undefined) {
            const missing = new Error(name);
            missing.name = "KeyError";
            throw missing;
          }
          return String(all_args[name]);
        });
      } catch (error) {
        if ((error as Error).name === "KeyError") {
          throw keyError(
            `${key} ${template} requires the following keyword arguments: ${(error as Error).message}`,
          );
        }
        throw error;
      }
      // NOTE: if no placeholder matched (e.g. a literal "{{"), Python would
      // loop forever; break defensively instead.
      if (path === raw) break;
    }
    if (!path) return null;
    // NOTE: Python returns a pathlib.Path, which normalizes away "./" and
    // duplicate slashes (Path("./media/Tex").as_posix() == "media/Tex").
    return purePosixNormalize(path);
  }
}

export const config = new ManimConfig(true);

/**
 * Temporarily modifies the global ``config`` object using a generator-based
 * context manager (mirroring Python's ``@contextmanager`` ``tempconfig``).
 *
 * Inside the ``with``-equivalent span, the modified config is used; after
 * the generator completes, the config is restored to its original state.
 *
 * Keys not present in the config are ignored, mirroring Python's
 * ``{k: v for k, v in temp.items() if k in original}`` filter.
 *
 * Usage:
 *
 *     const ctx = tempconfig({ frame_height: 100.0 });
 *     ctx.next(); // enter: config is updated
 *     try {
 *       // ... use the modified config ...
 *     } finally {
 *       ctx.return(undefined); // exit: config is restored
 *     }
 */
export function* tempconfig(
  temp: ManimConfig | Record<string, unknown>,
): Generator<void, void, void> {
  const original = config.copy();

  const raw: Record<string, unknown> = {};
  if (temp instanceof ManimConfig) {
    // NOTE: mirrors ``temp.items()`` (ManimConfig.__iter__ yields stored keys).
    for (const key of temp) raw[key] = temp.get(key);
  } else {
    Object.assign(raw, temp);
  }

  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(raw)) {
    // NOTE: mirrors ``k in original`` (ManimConfig.__contains__, which
    // probes __getitem__ and swallows AttributeError): unknown keys are
    // dropped instead of raising.
    try {
      config.get(key);
    } catch {
      continue;
    }
    filtered[key] = value;
  }

  // NOTE: mirrors Python's comment — update() mutates the shared object in
  // place (assignment would only rebind the local). Do NOT replace this
  // with an assignment.
  config.update(filtered);
  try {
    yield;
  } finally {
    config.update(original); // update, not assignment!
  }
}

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

export type ConfigParserData = Map<string, Map<string, string>>;

function parse_ini_text(text: string): ConfigParserData {
  const data: ConfigParserData = new Map();
  let section: Map<string, string> | null = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#") || line.startsWith(";")) continue;
    const sectionMatch = line.match(/^\[(.+)\]$/);
    if (sectionMatch) {
      const name = (sectionMatch[1] ?? "").trim();
      section = new Map();
      data.set(name, section);
      continue;
    }
    const separator = line.indexOf("=");
    if (separator === -1 || section === null) continue;
    const key = line.slice(0, separator).trim();
    let value = line.slice(separator + 1).trim();
    const commentHash = value.search(/(^|\s)[#;]/);
    if (commentHash !== -1) value = value.slice(0, commentHash).trim();
    section.set(key, value);
  }
  return data;
}

function merge_parser_data(target: ConfigParserData, source: ConfigParserData): void {
  for (const [section, values] of source) {
    let existing = target.get(section);
    if (!existing) {
      existing = new Map();
      target.set(section, existing);
    }
    for (const [key, value] of values) existing.set(key, value);
  }
}

export function make_config_parser(custom_file: string | null = null): ConfigParserData {
  const [libraryWide, userWide, folderWide] = config_file_paths();
  let libraryText: string;
  try {
    libraryText = readFileSync(libraryWide, { encoding: "utf-8" });
  } catch {
    throw new Error(`Could not read required config file: ${libraryWide}`);
  }
  const parser: ConfigParserData = parse_ini_text(libraryText);
  const otherFiles = [userWide, custom_file ?? folderWide];
  for (const path of otherFiles) {
    if (!existsSync(path)) continue;
    try {
      merge_parser_data(parser, parse_ini_text(readFileSync(path, { encoding: "utf-8" })));
    } catch {
      continue;
    }
  }
  return parser;
}

export function _determine_quality(qual: string | null | undefined): string | null {
  for (const [quality, values] of Object.entries(QUALITIES)) {
    if (values.flag !== null && values.flag === qual) return quality;
  }
  return qual ?? null;
}

const MANIM_FRAME_OPTS = [
  "pixel_width",
  "pixel_height",
  "aspect_ratio",
  "frame_height",
  "frame_width",
  "frame_y_radius",
  "frame_x_radius",
  "top",
  "bottom",
  "left_side",
  "right_side",
] as const;

const MANIM_FRAME_CONSTANTS: Record<string, readonly number[]> = {
  UP, DOWN, RIGHT, LEFT, IN, OUT, ORIGIN, X_AXIS, Y_AXIS, Z_AXIS, UL, UR, DL, DR,
};

export class ManimFrame {
  static readonly _OPTS: ReadonlySet<string> = new Set<string>(MANIM_FRAME_OPTS);
  static readonly _CONSTANTS: Record<string, readonly number[]> = MANIM_FRAME_CONSTANTS;

  private readonly configObject: ManimConfig;

  constructor(configObject: ManimConfig = config) {
    if (!(configObject instanceof ManimConfig)) throw new TypeError("argument must be instance of 'ManimConfig'");
    // NOTE: mirrors Python's `self.__dict__["_c"] = c` (bypassing the
    // immutable __setattr__); the instance is frozen below so later
    // attribute assignment raises, matching Python.
    this.configObject = configObject;
    Object.freeze(this);
  }

  __getitem__(key: string): any {
    if ((ManimFrame._OPTS as ReadonlySet<string>).has(key)) return this.configObject.get(key);
    if (Object.prototype.hasOwnProperty.call(MANIM_FRAME_CONSTANTS, key)) return MANIM_FRAME_CONSTANTS[key];
    const error = new Error(`'${key}'`);
    error.name = "KeyError";
    throw error;
  }

  get(key: string): any {
    return this.__getitem__(key);
  }

  *[Symbol.iterator](): IterableIterator<string> {
    yield* MANIM_FRAME_OPTS;
    yield* Object.keys(MANIM_FRAME_CONSTANTS);
  }

  __len__(): number {
    return MANIM_FRAME_OPTS.length;
  }

  get __length__(): number {
    return this.__len__();
  }

  __setitem__(key: unknown, _value: unknown): never {
    void key;
    throw new TypeError("'ManimFrame' object does not support item assignment");
  }

  __delitem__(key: unknown): never {
    void key;
    throw new TypeError("'ManimFrame' object does not support item deletion");
  }

  // NOTE: mirrors Python's `setattr(ManimFrame, opt, property(...))` loop:
  // every _OPTS/_CONSTANTS key is readable as an attribute.
  get pixel_width(): unknown { return this.__getitem__("pixel_width"); }
  get pixel_height(): unknown { return this.__getitem__("pixel_height"); }
  get aspect_ratio(): unknown { return this.__getitem__("aspect_ratio"); }
  get frame_height(): unknown { return this.__getitem__("frame_height"); }
  get frame_width(): unknown { return this.__getitem__("frame_width"); }
  get frame_y_radius(): unknown { return this.__getitem__("frame_y_radius"); }
  get frame_x_radius(): unknown { return this.__getitem__("frame_x_radius"); }
  get top(): unknown { return this.__getitem__("top"); }
  get bottom(): unknown { return this.__getitem__("bottom"); }
  get left_side(): unknown { return this.__getitem__("left_side"); }
  get right_side(): unknown { return this.__getitem__("right_side"); }
  get UP(): unknown { return this.__getitem__("UP"); }
  get DOWN(): unknown { return this.__getitem__("DOWN"); }
  get RIGHT(): unknown { return this.__getitem__("RIGHT"); }
  get LEFT(): unknown { return this.__getitem__("LEFT"); }
  get IN(): unknown { return this.__getitem__("IN"); }
  get OUT(): unknown { return this.__getitem__("OUT"); }
  get ORIGIN(): unknown { return this.__getitem__("ORIGIN"); }
  get X_AXIS(): unknown { return this.__getitem__("X_AXIS"); }
  get Y_AXIS(): unknown { return this.__getitem__("Y_AXIS"); }
  get Z_AXIS(): unknown { return this.__getitem__("Z_AXIS"); }
  get UL(): unknown { return this.__getitem__("UL"); }
  get UR(): unknown { return this.__getitem__("UR"); }
  get DL(): unknown { return this.__getitem__("DL"); }
  get DR(): unknown { return this.__getitem__("DR"); }
}
