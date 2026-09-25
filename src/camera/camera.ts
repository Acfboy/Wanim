import { array, copyto, zeros, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { ManimColor, color_to_int_rgba, type ParsableManimColor } from "../utils/color/core.js";
import { asArray, pointRows } from "../utils/arrays.js";
import { ORIGIN } from "../constants.js";
import { Mobject, setDefaultCameraFactory } from "../mobject/mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { PMobject } from "../mobject/types/point_cloud_mobject.js";
import { AbstractImageMobject } from "../mobject/types/image_mobject.js";
import { list_difference_update } from "../utils/iterables.js";
import { extract_mobject_family_members } from "../utils/family.js";
import { get_full_raster_image_path } from "../utils/images.js";

const logger = {
  info(..._args: unknown[]): void {},
};

export const LINE_JOIN_MAP: Record<string, string | null> = {
  AUTO: null,
  ROUND: "ROUND",
  BEVEL: "BEVEL",
  MITER: "MITER",
};

export const CAP_STYLE_MAP: Record<string, string | null> = {
  AUTO: null,
  ROUND: "ROUND",
  BUTT: "BUTT",
  SQUARE: "SQUARE",
};

export interface CameraOptions {
  background_image?: string | null;
  frame_center?: readonly number[];
  image_mode?: string;
  n_channels?: number;
  pixel_array_dtype?: string;
  cairo_line_width_multiple?: number;
  use_z_index?: boolean;
  background?: unknown;
  pixel_height?: number;
  pixel_width?: number;
  frame_height?: number;
  frame_width?: number;
  frame_rate?: number;
  background_color?: ParsableManimColor;
  background_opacity?: number;
  [key: string]: unknown;
}

export interface CaptureOptions {
  include_submobjects?: boolean;
  excluded_mobjects?: Iterable<Mobject> | null;
  [key: string]: unknown;
}

type DisplayHandler = (batch: Mobject[], pixel_array: NDArray) => void;

function cairoNotPorted(): Error {
  return new Error("TODO: cairo rendering not ported yet");
}

function pilNotPorted(): Error {
  return new Error("TODO: PIL image handling not ported yet");
}

export class Camera {
  background_image: string | null;
  image_mode: string;
  n_channels: number;
  pixel_array_dtype: string;
  cairo_line_width_multiple: number;
  use_z_index: boolean;
  background!: NDArray;
  pixel_array!: NDArray;
  pixel_height: number;
  pixel_width: number;
  frame_rate: number;
  max_allowable_norm: number;
  canvas: unknown = null;
  pixel_array_to_cairo_context: Map<object, unknown> = new Map();
  background_colored_vmobject_displayer: BackgroundColoredVMobjectDisplayer | null = null;
  display_funcs: Array<[Function, DisplayHandler]> = [];
  protected _background_color: ManimColor;
  protected _background_opacity: number;
  // NOTE: frame geometry is exposed via accessors (not plain fields) so
  // that MovingCamera can override them with frame-backed properties,
  // mirroring the Python @property definitions.
  private _frame_center!: readonly number[];
  private _frame_height = 0;
  private _frame_width = 0;

  constructor(options: CameraOptions = {}) {
    this.background_image = options.background_image ?? null;
    this.frame_center = options.frame_center ?? ORIGIN;
    this.image_mode = options.image_mode ?? "RGBA";
    this.n_channels = options.n_channels ?? 4;
    this.pixel_array_dtype = options.pixel_array_dtype ?? "uint8";
    this.cairo_line_width_multiple = options.cairo_line_width_multiple ?? 0.01;
    this.use_z_index = options.use_z_index ?? true;
    if (options.background !== undefined) {
      this.background = asArray(options.background);
    }
    this.pixel_height = options.pixel_height ?? config.pixel_height;
    this.pixel_width = options.pixel_width ?? config.pixel_width;
    this.frame_height = options.frame_height ?? config.frame_height;
    this.frame_width = options.frame_width ?? config.frame_width;
    this.frame_rate = options.frame_rate ?? config.frame_rate;
    this._background_color = new ManimColor(options.background_color ?? config.background_color);
    this._background_opacity = options.background_opacity ?? config.background_opacity;
    this.max_allowable_norm = config.frame_width;
    this.init_background();
    this.resize_frame_shape();
    this.reset();
  }

  __deepcopy__(_memo?: unknown): Camera {
    this.canvas = null;
    return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
  }

  get frame_center(): readonly number[] {
    return this._frame_center;
  }

  set frame_center(value: readonly number[]) {
    this._frame_center = value;
  }

  get frame_height(): number {
    return this._frame_height;
  }

  set frame_height(value: number) {
    this._frame_height = value;
  }

  get frame_width(): number {
    return this._frame_width;
  }

  set frame_width(value: number) {
    this._frame_width = value;
  }

  get background_color(): ManimColor {
    return this._background_color;
  }

  set background_color(color: ParsableManimColor) {
    this._background_color = color instanceof ManimColor ? color : new ManimColor(color);
    this.init_background();
  }

  get background_opacity(): number {
    return this._background_opacity;
  }

  set background_opacity(alpha: number) {
    this._background_opacity = alpha;
    this.init_background();
  }

  type_or_raise(mobject: Mobject): Function {
    this.display_funcs = [
      [VMobject, (batch: Mobject[], pa: NDArray) => this.display_multiple_vectorized_mobjects(batch as VMobject[], pa)],
      [PMobject, (batch: Mobject[], pa: NDArray) => this.display_multiple_point_cloud_mobjects(batch as PMobject[], pa)],
      [AbstractImageMobject, (batch: Mobject[], pa: NDArray) => this.display_multiple_image_mobjects(batch as AbstractImageMobject[], pa)],
      [Mobject, (_batch: Mobject[], _pa: NDArray) => {}],
    ];
    for (const [type, _handler] of this.display_funcs) {
      if (mobject instanceof (type as unknown as new (...args: never[]) => Mobject)) {
        return type;
      }
    }
    throw new TypeError(`Displaying an object of class ${Mobject.name} is not supported`);
  }

  reset_pixel_shape(new_height: number, new_width: number): void {
    this.pixel_width = new_width;
    this.pixel_height = new_height;
    this.init_background();
    this.resize_frame_shape();
    this.reset();
  }

  resize_frame_shape(fixedDimension = 0): void {
    const aspectRatio = this.pixel_width / this.pixel_height;
    if (fixedDimension === 0) this.frame_height = this.frame_width / aspectRatio;
    else this.frame_width = aspectRatio * this.frame_height;
  }

  init_background(): void {
    const height = this.pixel_height;
    const width = this.pixel_width;
    if (this.background_image !== null) {
      get_full_raster_image_path(this.background_image);
      throw pilNotPorted();
    }
    const rgba = color_to_int_rgba(this.background_color, this.background_opacity);
    const rows = Array.from({ length: height }, () =>
      Array.from({ length: width }, () => [...rgba]),
    );
    this.background = array(rows.length > 0 ? rows : []).reshape(height, width, this.n_channels).astype("uint8") as NDArray;
  }

  get_image(pixel_array?: unknown): unknown {
    if (pixel_array === undefined || pixel_array === null) {
      pixel_array = this.pixel_array;
    }
    return pixel_array;
  }

  convert_pixel_array(pixel_array: unknown): NDArray {
    const source = asArray(pixel_array).toArray() as unknown[][][];
    const converted = source.map((row) =>
      row.map((pixel) =>
        (Array.isArray(pixel) ? pixel : [pixel]).map((value) => Math.trunc(Number(value) * this.rgb_max_val)),
      ),
    );
    return (array(converted) as NDArray).astype(this.pixel_array_dtype as "uint8") as NDArray;
  }

  set_pixel_array(pixel_array: unknown, convert_from_floats = false): void {
    const converted = convert_from_floats ? this.convert_pixel_array(pixel_array) : asArray(pixel_array);
    if (
      this.pixel_array !== undefined &&
      this.pixel_array !== null &&
      shapesEqual((this.pixel_array as NDArray).shape, (converted as NDArray).shape)
    ) {
      copyto(this.pixel_array as never, converted as never);
    } else {
      this.pixel_array = (converted as NDArray).copy();
    }
  }

  set_background(pixel_array: unknown, convert_from_floats = false): void {
    this.background = (
      convert_from_floats ? this.convert_pixel_array(pixel_array) : array(asArray(pixel_array).toArray() as unknown[])
    ) as NDArray;
  }

  make_background_from_func(coords_to_colors_func: (coords: number[]) => number[]): NDArray {
    logger.info("Starting set_background");
    const coords = this.get_coords_of_all_pixels().toArray() as number[][][];
    const colored = coords.map((row) => row.map((pixel) => coords_to_colors_func(pixel)));
    logger.info("Ending set_background");
    return this.convert_pixel_array(colored);
  }

  set_background_from_func(coords_to_colors_func: (coords: number[]) => number[]): void {
    this.set_background(this.make_background_from_func(coords_to_colors_func));
  }

  reset(): this {
    this.set_pixel_array(this.background);
    return this;
  }

  set_frame_to_background(background: unknown): void {
    this.set_pixel_array(background);
  }

  get_mobjects_to_display(
    mobjects: Iterable<Mobject>,
    include_submobjects = true,
    excluded_mobjects?: Iterable<Mobject> | null,
  ): Mobject[] {
    let displayed: Mobject[];
    if (include_submobjects) {
      displayed = extract_mobject_family_members(Array.from(mobjects), this.use_z_index, true);
      if (excluded_mobjects) {
        const all_excluded = extract_mobject_family_members(Array.from(excluded_mobjects), this.use_z_index);
        displayed = list_difference_update(displayed, all_excluded);
      }
    } else {
      displayed = Array.from(mobjects);
    }
    return displayed;
  }

  is_in_frame(mobject: Mobject): boolean {
    const fc = pointRows(this.frame_center, 3)[0] ?? [0, 0, 0];
    const fh = this.frame_height;
    const fw = this.frame_width;
    return !(
      (mobject.get_right()[0] ?? 0) < (fc[0] ?? 0) - fw / 2 ||
      (mobject.get_bottom()[1] ?? 0) > (fc[1] ?? 0) + fh / 2 ||
      (mobject.get_left()[0] ?? 0) > (fc[0] ?? 0) + fw / 2 ||
      (mobject.get_top()[1] ?? 0) < (fc[1] ?? 0) - fh / 2
    );
  }

  capture_mobject(mobject: Mobject, options: CaptureOptions = {}): void {
    return this.capture_mobjects([mobject], options);
  }

  capture_mobjects(mobjects: Iterable<Mobject>, options: CaptureOptions = {}): void {
    const displayed = this.get_mobjects_to_display(
      mobjects,
      options.include_submobjects ?? true,
      options.excluded_mobjects ?? null,
    );
    const typed = displayed.map((mobject) => ({ mobject, type: this.type_or_raise(mobject) }));
    let index = 0;
    while (index < typed.length) {
      const groupType = typed[index]?.type;
      let end = index + 1;
      while (end < typed.length && typed[end]?.type === groupType) end += 1;
      const batch = typed.slice(index, end).map((entry) => entry.mobject);
      const handler = this.display_funcs.find(([type]) => type === groupType)?.[1];
      if (handler !== undefined) handler(batch, this.pixel_array);
      index = end;
    }
  }

  get_cached_cairo_context(pixel_array: NDArray): unknown {
    return this.pixel_array_to_cairo_context.get(pixel_array) ?? null;
  }

  cache_cairo_context(pixel_array: NDArray, ctx: unknown): void {
    this.pixel_array_to_cairo_context.set(pixel_array, ctx);
  }

  get_cairo_context(_pixel_array: NDArray): unknown {
    throw cairoNotPorted();
  }

  display_multiple_vectorized_mobjects(_vmobjects: VMobject[], _pixel_array: NDArray): void {
    throw cairoNotPorted();
  }

  display_multiple_non_background_colored_vmobjects(_vmobjects: Iterable<VMobject>, _pixel_array: NDArray): void {
    throw cairoNotPorted();
  }

  display_vectorized(_vmobject: VMobject, _ctx: unknown): this {
    throw cairoNotPorted();
  }

  set_cairo_context_path(_ctx: unknown, _vmobject: VMobject): this {
    throw cairoNotPorted();
  }

  set_cairo_context_color(_ctx: unknown, _rgbas: unknown, _vmobject: VMobject): this {
    throw cairoNotPorted();
  }

  apply_fill(_ctx: unknown, _vmobject: VMobject): this {
    throw cairoNotPorted();
  }

  apply_stroke(_ctx: unknown, _vmobject: VMobject, _background = false): this {
    throw cairoNotPorted();
  }

  get_stroke_rgbas(vmobject: VMobject, background = false): NDArray {
    return vmobject.get_stroke_rgbas(background);
  }

  get_fill_rgbas(vmobject: VMobject): NDArray {
    return vmobject.get_fill_rgbas();
  }

  get_background_colored_vmobject_displayer(): BackgroundColoredVMobjectDisplayer {
    if (this.background_colored_vmobject_displayer === null) {
      this.background_colored_vmobject_displayer = new BackgroundColoredVMobjectDisplayer(this);
    }
    return this.background_colored_vmobject_displayer;
  }

  display_multiple_background_colored_vmobjects(_cvmobjects: Iterable<VMobject>, _pixel_array: NDArray): this {
    throw cairoNotPorted();
  }

  display_multiple_point_cloud_mobjects(pmobjects: Iterable<PMobject>, pixel_array: NDArray): void {
    for (const pmobject of pmobjects) {
      this.display_point_cloud(
        pmobject,
        pmobject.points,
        pmobject.rgbas,
        this.adjusted_thickness(pmobject.stroke_width),
        pixel_array,
      );
    }
  }

  display_point_cloud(
    _pmobject: PMobject,
    points: NDArray,
    rgbas: NDArray,
    thickness: number,
    pixel_array: NDArray,
  ): void {
    const pointRowsList = pointRows(points, 3);
    if (pointRowsList.length === 0) return;
    let pixel_coords = this.points_to_pixel_coords(_pmobject, points);
    pixel_coords = this.thickened_coordinates(pixel_coords, thickness);
    const rgba_len = (pixel_array as NDArray).shape[2] ?? 4;
    const colorRows = pointRows(rgbas, rgba_len).map((row) =>
      row.map((value) => Math.trunc(Number(value) * this.rgb_max_val)),
    );
    const coordRows = pointRows(pixel_coords, 2);
    const target_len = coordRows.length;
    const factor = colorRows.length === 0 ? 0 : Math.floor(target_len / colorRows.length);
    let tiled: number[][] = [];
    for (let repeat = 0; repeat < factor; repeat += 1) tiled.push(...colorRows.map((row) => [...row]));
    tiled = tiled.slice(0, target_len);
    while (tiled.length < target_len) tiled.push([...(colorRows[0] ?? new Array(rgba_len).fill(0))]);
    const on_screen = this.on_screen_pixels(pixel_coords);
    const kept_coords = coordRows.filter((_, index) => on_screen[index] ?? false);
    const kept_rgbas = tiled.filter((_, index) => on_screen[index] ?? false);
    const ph = this.pixel_height;
    const pw = this.pixel_width;
    for (let index = 0; index < kept_coords.length; index += 1) {
      const flat = (kept_coords[index]?.[1] ?? 0) * pw + (kept_coords[index]?.[0] ?? 0);
      const row = Math.floor(flat / pw);
      const col = flat - row * pw;
      const color = kept_rgbas[index] ?? [];
      for (let channel = 0; channel < rgba_len; channel += 1) {
        (pixel_array as NDArray).set([row, col, channel], color[channel] ?? 0);
      }
    }
  }

  display_multiple_image_mobjects(_image_mobjects: Iterable<AbstractImageMobject>, _pixel_array: NDArray): void {
    throw pilNotPorted();
  }

  display_image_mobject(_image_mobject: AbstractImageMobject, _pixel_array: NDArray): void {
    throw pilNotPorted();
  }

  overlay_rgba_array(_pixel_array: NDArray, _new_array: NDArray): void {
    throw pilNotPorted();
  }

  overlay_PIL_image(_pixel_array: NDArray, _image: unknown): void {
    throw pilNotPorted();
  }

  adjust_out_of_range_points(points: NDArray): NDArray {
    const raw = asArray(points).toArray() as unknown;
    const rows: number[][] = (
      Array.isArray(raw) && raw.length > 0 && Array.isArray(raw[0])
        ? (raw as unknown[][])
        : [raw as unknown[]]
    ).map((row) => (Array.isArray(row) ? row : [row]).map(Number));
    const has_violator = rows.flat().some((value) => value > this.max_allowable_norm);
    if (!has_violator) return points;
    const adjusted = rows.map((row) => {
      const norm = Math.hypot(...row);
      if (norm > this.max_allowable_norm) {
        return row.map((value) => (this.max_allowable_norm * value) / norm);
      }
      return row;
    });
    return array(adjusted) as NDArray;
  }

  transform_points_pre_display(_mobject: Mobject, points: NDArray): NDArray {
    const rows = pointRows(points, 3);
    if (!rows.flat().every((value) => Number.isFinite(Number(value)))) {
      return array([[0, 0, 0]]) as NDArray;
    }
    return points;
  }

  points_to_subpixel_coords(_mobject: Mobject, points: NDArray): NDArray {
    const transformed = pointRows(this.transform_points_pre_display(_mobject, points), 3);
    const fc = pointRows(this.frame_center, 3)[0] ?? [0, 0, 0];
    const width_mult = this.pixel_width / this.frame_width;
    const width_add = this.pixel_width / 2;
    let height_mult = this.pixel_height / this.frame_height;
    const height_add = this.pixel_height / 2;
    height_mult *= -1;
    const result = transformed.map((point) => [
      ((point[0] ?? 0) - (fc[0] ?? 0)) * width_mult + width_add,
      ((point[1] ?? 0) - (fc[1] ?? 0)) * height_mult + height_add,
    ]);
    return array(result.length > 0 ? result : []) as NDArray;
  }

  points_to_pixel_coords(mobject: Mobject, points: NDArray): NDArray {
    const subpixel = pointRows(this.points_to_subpixel_coords(mobject, points), 2);
    return array(
      subpixel.length > 0 ? subpixel.map((row) => row.map((value) => Math.trunc(value))) : [],
    ) as NDArray;
  }

  on_screen_pixels(pixel_coords: NDArray | number[][]): boolean[] {
    const rows = pointRows(pixel_coords, 2);
    return rows.map((row) => {
      const x = row[0] ?? 0;
      const y = row[1] ?? 0;
      return x >= 0 && x < this.pixel_width && y >= 0 && y < this.pixel_height;
    });
  }

  adjusted_thickness(thickness: number): number {
    const big_sum: number = config.pixel_height + config.pixel_width;
    const this_sum: number = this.pixel_height + this.pixel_width;
    const factor = big_sum / this_sum;
    return 1 + (thickness - 1) * factor;
  }

  get_thickening_nudges(thickness: number): NDArray {
    const nudged = Math.trunc(thickness);
    const range: number[] = [];
    for (let index = Math.floor(-nudged / 2) + 1; index <= Math.floor(nudged / 2); index += 1) {
      range.push(index);
    }
    const pairs: number[][] = [];
    for (const first of range) for (const second of range) pairs.push([first, second]);
    return array(pairs.length > 0 ? pairs : []) as NDArray;
  }

  thickened_coordinates(pixel_coords: NDArray | number[][], thickness: number): NDArray {
    const nudges = pointRows(this.get_thickening_nudges(thickness), 2);
    const coords = pointRows(pixel_coords, 2);
    const thickened: number[][] = [];
    for (const nudge of nudges) {
      for (const coord of coords) {
        thickened.push([(coord[0] ?? 0) + (nudge[0] ?? 0), (coord[1] ?? 0) + (nudge[1] ?? 0)]);
      }
    }
    return array(thickened.length > 0 ? thickened : []) as NDArray;
  }

  get_coords_of_all_pixels(): NDArray {
    const full_space_dims = [this.frame_width, this.frame_height];
    const full_pixel_dims = [this.pixel_width, this.pixel_height];
    const centered: number[][][] = [];
    for (let row = 0; row < this.pixel_height; row += 1) {
      const line: number[][] = [];
      for (let col = 0; col < this.pixel_width; col += 1) {
        const uncentered_x = ((col * (full_space_dims[0] ?? 0)) / (full_pixel_dims[0] ?? 1));
        const uncentered_y = ((row * (full_space_dims[1] ?? 0)) / (full_pixel_dims[1] ?? 1));
        line.push([
          uncentered_x - (full_space_dims[0] ?? 0) / 2,
          -1 * (uncentered_y - (full_space_dims[1] ?? 0) / 2),
        ]);
      }
      centered.push(line);
    }
    return array(centered.length > 0 ? centered : []) as NDArray;
  }

  get rgb_max_val(): number {
    if (this.pixel_array_dtype.startsWith("uint")) {
      const bits = Number(this.pixel_array_dtype.slice(4)) || 8;
      return 2 ** bits - 1;
    }
    if (this.pixel_array_dtype.startsWith("int")) {
      const bits = Number(this.pixel_array_dtype.slice(3)) || 8;
      return 2 ** (bits - 1) - 1;
    }
    return 255;
  }
}

export class BackgroundColoredVMobjectDisplayer {
  camera: Camera;
  file_name_to_pixel_array_map: Record<string, NDArray> = {};
  pixel_array: NDArray;

  constructor(camera: Camera) {
    this.camera = camera;
    this.pixel_array = (camera.pixel_array as NDArray).copy();
    this.reset_pixel_array();
  }

  reset_pixel_array(): void {
    this.pixel_array = zeros([...(this.pixel_array as NDArray).shape]) as NDArray;
  }

  resize_background_array(
    _background_array: NDArray,
    _new_width: number,
    _new_height: number,
    _mode = "RGBA",
  ): NDArray {
    throw pilNotPorted();
  }

  resize_background_array_to_match(background_array: NDArray, pixel_array: NDArray): NDArray {
    void pixel_array;
    return this.resize_background_array(
      background_array,
      (pixel_array as NDArray).shape[1] ?? 0,
      (pixel_array as NDArray).shape[0] ?? 0,
      (pixel_array as NDArray).shape[2] === 4 ? "RGBA" : "RGB",
    );
  }

  get_background_array(image: string | NDArray | null): NDArray {
    const image_key = String(image);
    if (Object.prototype.hasOwnProperty.call(this.file_name_to_pixel_array_map, image_key)) {
      return this.file_name_to_pixel_array_map[image_key] as NDArray;
    }
    if (typeof image === "string") {
      get_full_raster_image_path(image);
      throw pilNotPorted();
    }
    let back_array = asArray(image);
    const pixel_array = this.pixel_array;
    if (!shapesEqual(back_array.shape, pixel_array.shape)) {
      back_array = this.resize_background_array_to_match(back_array, pixel_array);
    }
    this.file_name_to_pixel_array_map[image_key] = back_array;
    return back_array;
  }

  display(...cvmobjects: VMobject[]): NDArray | null {
    const groups = group_consecutive(cvmobjects, (candidate) => candidate.get_background_image());
    let curr_array: NDArray | null = null;
    for (const [image, batch] of groups) {
      const background_array = this.get_background_array(image);
      const pixel_array = this.pixel_array;
      this.camera.display_multiple_non_background_colored_vmobjects(batch, pixel_array);
      const new_array = scaled_product(background_array, pixel_array);
      curr_array = elementwise_maximum(curr_array, new_array);
      this.reset_pixel_array();
    }
    return curr_array;
  }
}

function scaled_product(background: NDArray, pixels: NDArray): NDArray {
  const backgroundRows = background.toArray() as number[][][];
  const pixelRows = pixels.toArray() as number[][][];
  const result = backgroundRows.map((row, rowIndex) =>
    row.map((pixel, colIndex) =>
      pixel.map((value, channel) =>
        (Number(value) * Number(pixelRows[rowIndex]?.[colIndex]?.[channel] ?? 0)) / 255,
      ),
    ),
  );
  return array(result) as NDArray;
}

function elementwise_maximum(first: NDArray | null, second: NDArray): NDArray {
  if (first === null) return (second as NDArray).copy();
  const firstRows = (first as NDArray).toArray() as number[][][];
  const secondRows = (second as NDArray).toArray() as number[][][];
  const result = secondRows.map((row, rowIndex) =>
    row.map((pixel, colIndex) =>
      pixel.map((value, channel) =>
        Math.max(Number(value), Number(firstRows[rowIndex]?.[colIndex]?.[channel] ?? 0)),
      ),
    ),
  );
  return array(result) as NDArray;
}

function group_consecutive<T, K>(items: T[], key: (item: T) => K): Array<[K, T[]]> {
  const groups: Array<[K, T[]]> = [];
  for (const item of items) {
    const itemKey = key(item);
    const last = groups[groups.length - 1];
    if (last !== undefined && last[0] === itemKey) last[1].push(item);
    else groups.push([itemKey, [item]]);
  }
  return groups;
}

function shapesEqual(first: readonly number[], second: readonly number[]): boolean {
  return first.length === second.length && first.every((value, index) => value === second[index]);
}

// Lets Mobject.get_image() with no explicit camera mirror Python's
// `Camera().capture_mobject(self)` once this module is loaded. The import
// direction stays camera -> mobject, so no module cycle is introduced.
setDefaultCameraFactory(() => new Camera());
