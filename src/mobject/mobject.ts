import { array, concatenate, type NDArray } from "numpy-ts";
import { MethodWithArgs } from "../data_structures.js";
import { DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, MED_SMALL_BUFF, ORIGIN, RIGHT, UP } from "../constants.js";
import { ManimColor, WHITE, interpolate_color, type ParsableManimColor } from "../utils/color/core.js";
import { MultiAnimationOverrideException } from "../utils/exceptions.js";
import { remove_list_redundancies } from "../utils/iterables.js";
import { straight_path, type PathFuncType } from "../utils/paths.js";
import { addPoints, isNDArray, nestedRows, norm, numeric, pointAt, pointFrom, pointRows, pointsFrom, pythonTypeName, scalePoint, subtractPoints, toPythonRepr, toPythonString } from "../utils/arrays.js";
import { Animation } from "../animation/animation.js";
import { config } from "../_config/utils.js";

export type MobjectUpdater = ((mobject: Mobject) => void) | ((mobject: Mobject, dt: number) => void);
export type MobjectArray = Mobject[];

export interface MobjectOptions {
  color?: ParsableManimColor | null;
  name?: string;
  dim?: number;
  target?: Mobject | null;
  z_index?: number;
  [key: string]: unknown;
}

export type FunctionLike = (...args: any[]) => any;
const overrideFunctions = new WeakMap<object, FunctionLike>();

export function get_animation_override(method: FunctionLike): FunctionLike | undefined {
  return overrideFunctions.get(method) ?? (method as FunctionLike & { _override_animate?: FunctionLike })._override_animate;
}

function cloneValue(value: unknown, seen = new Map<object, unknown>()): unknown {
  if (value === null || value === undefined || typeof value !== "object") return value;
  if (isNDArray(value)) return value.copy();
  if (value instanceof ManimColor) return new ManimColor(value);
  if (Array.isArray(value)) return value.map((item) => cloneValue(item, seen));
  if (value instanceof Set) return new Set([...value].map((item) => cloneValue(item, seen)));
  if (value instanceof Map) return new Map([...value].map(([key, item]) => [key, cloneValue(item, seen)]));
  if (value instanceof Mobject) return value.copy_with_seen(seen);
  const object = value as Record<string, unknown>;
  const result: Record<string, unknown> = {};
  seen.set(value, result);
  for (const [key, item] of Object.entries(object)) result[key] = cloneValue(item, seen);
  return result;
}

function vector(value: unknown, dim = 3): number[] {
  return pointFrom(value, dim).toArray() as number[];
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return value !== null && value !== undefined && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function";
}

function pointArray(rows: number[][]): NDArray {
  return array(rows) as NDArray;
}

export function repeat_array_impl(value: NDArray, count: number, dim = 3): NDArray {
  if (!Number.isInteger(count)) throw new TypeError("can't multiply sequence by non-int of type 'float'");
  if (count <= 0) throw new TypeError("reduce() of empty iterable with no initial value");
  const rows = nestedRows(value).map((row) => Array.isArray(row) ? row.map((item) => numeric(item)) : [numeric(row)]);
  const width = rows[0]?.length ?? dim;
  const result: number[][] = [];
  for (let index = 0; index < count; index += 1) result.push(...rows.map((row) => [...row, ...Array.from({ length: Math.max(0, width - row.length) }, () => 0)]));
  return array(result.length > 0 ? result : []).reshape(result.length, width) as NDArray;
}

export const repeat_array = repeat_array_impl;

export function init_size(num: number | null | undefined, alignments: string | null | undefined, sizes: Iterable<unknown> | null | undefined): number | null {
  if (num !== null && num !== undefined) return num;
  if (alignments !== null && alignments !== undefined) return alignments.length;
  if (sizes !== null && sizes !== undefined) return [...sizes].length;
  return null;
}

export function init_sizes(sizes: readonly (number | null | undefined)[] | null | undefined, num: number, measures: readonly number[], name: string): number[] {
  const actual = sizes === null || sizes === undefined || sizes.length === 0 ? Array.from({ length: num }, () => null) : sizes;
  if (actual.length !== num) {
    const error = new Error(`${name} has a mismatching size.`);
    error.name = "ValueError";
    throw error;
  }
  return actual.map((size, index) => size === null || size === undefined ? measures[index] : size);
}

export function init_alignments(alignments: string | null | undefined, num: number, char_to_direction: Record<string, readonly number[]>, name: string, dir_: readonly number[] | NDArray, cell_alignment: readonly number[] | NDArray = [0, 0, 0]): number[][] {
  const direction = pointFrom(dir_, 3).toArray() as number[];
  const fallback = pointFrom(cell_alignment, 3).toArray() as number[];
  if (alignments === null || alignments === undefined) return Array.from({ length: num }, () => fallback.map((value, index) => value * (direction[index] ?? 0)));
  if (alignments.length !== num) {
    const error = new Error(`${name}_alignments has a mismatching size.`);
    error.name = "ValueError";
    throw error;
  }
    return [...alignments].map((character) => {
    const direction = char_to_direction[character];
    if (!direction) {
      const error = new Error(`'${character}'`);
      error.name = "KeyError";
      throw error;
    }
    return [...direction];
  });
}

function createMobjectProxy(target: Mobject): Mobject {
  return new Proxy(target, {
    get: (object, property, receiver) => {
      if (property in object || typeof property === "symbol") return Reflect.get(object, property, receiver);
      if (typeof property === "string" && property.startsWith("get_")) {
        return () => {
          const name = property.slice(4);
          if (!(name in object)) {
            const error = new Error(`${object.constructor.name} object has no attribute '${name}'`);
            error.name = "AttributeError";
            throw error;
          }
          return (object as unknown as Record<string, unknown>)[name];
        };
      }
      if (typeof property === "string" && property.startsWith("set_")) {
        return (value: unknown) => {
          (object as unknown as Record<string, unknown>)[property.slice(4)] = value;
          return receiver;
        };
      }
      if (property === "size" || property === "nodeType" || property === "tagName" || (typeof property === "string" && property.startsWith("@@"))) return undefined;
      const error = new Error(`${object.constructor.name} object has no attribute '${String(property)}'`);
      error.name = "AttributeError";
      throw error;
    },
    set: (object, property, value, receiver) => Reflect.set(object, property, value, receiver),
  });
}

export type BackgroundRectangleFactory = (
  mobject: Mobject,
  color: ParsableManimColor | null | undefined,
  fill_opacity: number,
  kwargs: Record<string, unknown>,
) => Mobject;

let backgroundRectangleFactory: BackgroundRectangleFactory | null = null;

export function setBackgroundRectangleFactory(factory: BackgroundRectangleFactory | null): void {
  backgroundRectangleFactory = factory;
}

export interface ImageCapableCamera {
  capture_mobject(mobject: Mobject): void;
  get_image(): unknown;
}

let defaultCameraFactory: (() => ImageCapableCamera) | null = null;

export function setDefaultCameraFactory(factory: (() => ImageCapableCamera) | null): void {
  defaultCameraFactory = factory;
}

export class Mobject {
  name: string;
  dim: number;
  target: Mobject | null;
  z_index: number;
  point_hash: string | null = null;
  submobjects: Mobject[] = [];
  updaters: MobjectUpdater[] = [];
  updating_suspended = false;
  color: ManimColor;
  points: NDArray;
  saved_state: Mobject | null = null;
  original_id: string;
  background_rectangle: Mobject | null = null;
  readonly animation_overrides: Map<unknown, FunctionLike> = new Map();

  constructor(options: MobjectOptions = {}) {
    const normalized = typeof options === "string" || Array.isArray(options) ? { color: options as ParsableManimColor } : options;
    const ctor = this.constructor as unknown as { _wanimDefaultKwargs?: Record<string, unknown> };
    const defaults = ctor._wanimDefaultKwargs ?? {};
    const merged: Record<string, unknown> = { ...defaults, ...normalized };
    this.name = (merged.name as string | undefined) ?? this.constructor.name;
    this.dim = (merged.dim as number | undefined) ?? 3;
    this.target = (merged.target as Mobject | null | undefined) ?? null;
    this.z_index = (merged.z_index as number | undefined) ?? 0;
    this.color = new ManimColor((merged.color as ParsableManimColor | undefined) ?? WHITE);
    this.points = array([]) as NDArray;
    this.original_id = String(Math.random());
    for (const [key, value] of Object.entries(merged)) {
      if (!(key in this)) (this as Record<string, unknown>)[key] = value;
    }
    this.reset_points();
    this.generate_points();
    Mobject.prototype.init_colors.call(this);
    return createMobjectProxy(this);
  }

  static set_default(kwargs: Record<string, unknown> = {}): void {
    const ctor = this as unknown as { _wanimDefaultKwargs?: Record<string, unknown> };
    if (Object.keys(kwargs).length > 0) {
      ctor._wanimDefaultKwargs = { ...(ctor._wanimDefaultKwargs ?? {}), ...kwargs };
    } else {
      delete ctor._wanimDefaultKwargs;
    }
  }

  static setDefault(kwargs: Record<string, unknown> = {}): void {
    (this as unknown as typeof Mobject).set_default(kwargs);
  }

  get animate(): _AnimationBuilder {
    return createAnimationBuilder(this);
  }

  get always(): _UpdaterBuilder {
    return createUpdaterBuilder(this);
  }

  get_array_attrs(): string[] {
    return ["points"];
  }

  getArrayAttrs(): string[] {
    return this.get_array_attrs();
  }

  get_updaters(): MobjectUpdater[] {
    return this.updaters;
  }

  getUpdaters(): MobjectUpdater[] {
    return this.get_updaters();
  }

  get_time_based_updaters(): MobjectUpdater[] {
    return this.updaters.filter((updater) => updater.length >= 2);
  }

  getTimeBasedUpdaters(): MobjectUpdater[] {
    return this.get_time_based_updaters();
  }

  has_time_based_updater(): boolean {
    return this.updaters.some((updater) => updater.length >= 2);
  }

  hasTimeBasedUpdater(): boolean {
    return this.has_time_based_updater();
  }

  get_family_updaters(): MobjectUpdater[] {
    return this.get_family().flatMap((mobject) => mobject.get_updaters());
  }

  getFamilyUpdaters(): MobjectUpdater[] {
    return this.get_family_updaters();
  }

  get_color(): ManimColor {
    return this.color;
  }

  getColor(): ManimColor {
    return this.get_color();
  }

  has_points(): boolean {
    return this.get_num_points() > 0;
  }

  has_no_points(): boolean {
    return !this.has_points();
  }

  get_num_points(): number {
    return Array.isArray(this.points) ? this.points.length : this.points.shape[0];
  }

  getNumPoints(): number {
    return this.get_num_points();
  }

  reset_points(): this {
    this.points = array([]).reshape(0, this.dim) as NDArray;
    return this;
  }

  generate_points(): this {
    return this;
  }

  init_colors(): this {
    return this;
  }

  _assert_valid_submobjects(submobjects: Iterable<Mobject>): this {
    return this._assert_valid_submobjects_internal(submobjects, Mobject);
  }

  _assert_valid_submobjects_internal(submobjects: Iterable<Mobject>, mob_class: Function): this {
    let index = 0;
    for (const submob of submobjects) {
      if (!(submob instanceof (mob_class as new (...args: any[]) => Mobject))) {
        let errorMessage = `Only values of type ${mob_class.name} can be added as submobjects of ${this.constructor.name}, but the value ${toPythonString(submob)} (at index ${index}) is of type ${pythonTypeName(submob)}.`;
        if ((submob as unknown) instanceof Mobject) errorMessage += " You can try adding this value into a Group instead.";
        throw new TypeError(errorMessage);
      }
      if (submob === this) {
        const error = new Error(`Cannot add ${this.constructor.name} as a submobject of itself (at index ${index}).`);
        error.name = "ValueError";
        throw error;
      }
      index += 1;
    }
    return this;
  }

  add(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    const values: Mobject[] = [];
    for (const item of mobjects) {
      if (item instanceof Mobject) values.push(item);
      else if (isIterable(item)) values.push(...item as Iterable<Mobject>);
      else values.push(item as Mobject);
    }
    this._assert_valid_submobjects(values);
    const unique = remove_list_redundancies(values);
    if (values.length !== unique.length) {
      console.warn(
        "Attempted adding some Mobject as a child more than once, " +
          "this is not possible. Repetitions are ignored.",
      );
    }
    this.submobjects = this.submobjects.filter((mobject) => !unique.includes(mobject));
    this.submobjects.push(...unique);
    return this;
  }

  insert(index: number, mobject: Mobject): this {
    this._assert_valid_submobjects([mobject]);
    this.submobjects.splice(index, 0, mobject);
    return this;
  }

  add_to_back(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    const values: Mobject[] = [];
    for (const item of mobjects) {
      if (item instanceof Mobject) values.push(item);
      else if (isIterable(item)) values.push(...item as Iterable<Mobject>);
      else values.push(item as Mobject);
    }
    this._assert_valid_submobjects(values);
    this.remove(...values);
    const unique = remove_list_redundancies(values);
    this.submobjects = [...unique, ...this.submobjects];
    return this;
  }

  addToBack(...mobjects: (Mobject | Iterable<Mobject>)[]): this {
    return this.add_to_back(...mobjects);
  }

  remove(...mobjects: Mobject[]): this {
    for (const mobject of mobjects) {
      const index = this.submobjects.indexOf(mobject);
      if (index >= 0) this.submobjects.splice(index, 1);
    }
    return this;
  }

  get_family(_recurse = true): Mobject[] {
    const subFamilies = this.submobjects.map((child) => child.get_family());
    const all: Mobject[] = [this];
    for (const family of subFamilies) all.push(...family);
    return remove_list_redundancies(all);
  }

  getFamily(_recurse = true): Mobject[] {
    return this.get_family(_recurse);
  }

  family_members_with_points(): Mobject[] {
    return this.get_family().filter((mobject) => mobject.get_num_points() > 0);
  }

  familyMembersWithPoints(): Mobject[] {
    return this.family_members_with_points();
  }

  get_group_class(): typeof Group {
    return Group;
  }

  getGroupClass(): typeof Group {
    return this.get_group_class();
  }

  static get_mobject_type_class(): typeof Mobject {
    return Mobject;
  }

  static getMobjectTypeClass(): typeof Mobject {
    return Mobject.get_mobject_type_class();
  }

  copy(): this {
    return this.copy_with_seen(new Map<object, unknown>());
  }

  copy_with_seen(seen: Map<object, unknown>): this {
    if (seen.has(this)) return seen.get(this) as this;
    const raw = Object.create(Object.getPrototypeOf(this)) as this;
    const result = createMobjectProxy(raw);
    seen.set(this, result);
    for (const [key, value] of Object.entries(this)) {
      if (key === "target" && value instanceof Mobject) (raw as unknown as Record<string, unknown>).target = value.copy_with_seen(seen);
      else (raw as unknown as Record<string, unknown>)[key] = cloneValue(value, seen);
    }
    raw.original_id = String(Math.random());
    return result as this;
  }

  __deepcopy__(clone_from_id: Map<object, unknown> = new Map()): this {
    if (clone_from_id.has(this)) return clone_from_id.get(this) as this;
    const raw = Object.create(Object.getPrototypeOf(this)) as this;
    clone_from_id.set(this, raw);
    for (const [key, value] of Object.entries(this)) {
      (raw as unknown as Record<string, unknown>)[key] = cloneValue(value, clone_from_id);
    }
    raw.original_id = String(Math.random());
    return createMobjectProxy(raw) as this;
  }

  generate_target(useDeepcopy = false): Mobject | null {
    this.target = null;
    this.target = this.copy();
    return this.target;
  }

  get_image(camera?: ImageCapableCamera | null): unknown {
    const cam = camera ?? (defaultCameraFactory !== null ? defaultCameraFactory() : null);
    if (cam === null || cam === undefined) {
      throw new Error("TODO: cairo rendering not ported yet (Mobject.get_image needs a Camera with a raster backend)");
    }
    cam.capture_mobject(this);
    return cam.get_image();
  }

  getImage(camera?: ImageCapableCamera | null): unknown {
    return this.get_image(camera);
  }

  show(camera?: ImageCapableCamera | null): void {
    (this.get_image(camera) as { show(): void }).show();
  }

  save_image(name: string | null = null): void {
    const image = this.get_image() as { save(path: string): void };
    const videoDir = config.get_dir("video_dir") ?? ".";
    const fileName = `${name ?? String(this.name)}.png`;
    image.save(`${videoDir}/${fileName}`);
  }

  saveImage(name: string | null = null): void {
    this.save_image(name);
  }

  get_center(): number[] {
    return this.get_critical_point(ORIGIN);
  }

  getCenter(): number[] {
    return this.get_center();
  }

  get_critical_point(direction: readonly number[] | NDArray = ORIGIN): number[] {
    const points = this.family_members_with_points().flatMap((mobject) => pointRows(mobject.points, this.dim));
    const directionValues = pointFrom(direction, this.dim).toArray() as number[];
    const result = Array.from({ length: this.dim }, () => 0);
    if (points.length === 0) return result;
    for (let dim = 0; dim < this.dim; dim += 1) {
      const key = Number(directionValues[dim] ?? 0);
      const values = points.map((point) => point[dim] ?? 0);
      const minimum = Math.min(...values);
      const maximum = Math.max(...values);
      result[dim] = key < 0 ? minimum : key > 0 ? maximum : (minimum + maximum) / 2;
    }
    return result;
  }

  getCriticalPoint(direction: readonly number[] = ORIGIN): number[] {
    return this.get_critical_point(direction);
  }

  get_left(): number[] { return this.get_critical_point([-1, 0, 0]); }
  get_right(): number[] { return this.get_critical_point([1, 0, 0]); }
  get_top(): number[] { return this.get_critical_point([0, 1, 0]); }
  get_bottom(): number[] { return this.get_critical_point([0, -1, 0]); }
  get_center_point(): number[] { return this.get_center(); }

  get_coord(dim: number, direction: readonly number[] | NDArray = ORIGIN): number {
    return this.get_critical_point(direction)[dim] ?? 0;
  }

  get_x(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(0, direction); }
  get_y(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(1, direction); }
  get_z(direction: readonly number[] | NDArray = ORIGIN): number { return this.get_coord(2, direction); }

  get_extremum_along_dim(dim: number, key: number): number {
    const points = this.family_members_with_points().flatMap((mobject) => pointRows(mobject.points, this.dim));
    if (points.length === 0) return 0;
    const values = points.map((point) => point[dim] ?? 0);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    return key < 0 ? minimum : key > 0 ? maximum : (minimum + maximum) / 2;
  }

  get width(): number { return this.length_over_dim(0); }
  set width(value: number) { this.scale_to_fit_width(value); }
  get height(): number { return this.length_over_dim(1); }
  set height(value: number) { this.scale_to_fit_height(value); }
  get depth(): number { return this.length_over_dim(2); }
  set depth(value: number) { this.scale_to_fit_depth(value); }

  length_over_dim(dim: number): number {
    const points = this.family_members_with_points().flatMap((mobject) => pointRows(mobject.points, this.dim));
    if (points.length === 0) return 0;
    return Math.max(...points.map((point) => point[dim] ?? 0)) - Math.min(...points.map((point) => point[dim] ?? 0));
  }

  scale_to_fit_width(value: number): this { return this.rescale_to_fit(value, 0); }
  scale_to_fit_height(value: number): this { return this.rescale_to_fit(value, 1); }
  scale_to_fit_depth(value: number): this { return this.rescale_to_fit(value, 2); }

  rescale_to_fit(value: number, dim: number): this {
    const current = this.length_over_dim(dim);
    return current === 0 ? this : this.scale(value / current);
  }

  scale(scalar: number, aboutPoint?: readonly number[]): this {
    const center = aboutPoint ? vector(aboutPoint, this.dim) : this.get_center();
    for (const mobject of this.family_members_with_points()) {
      const rows = pointRows(mobject.points, this.dim).map((point) => addPoints(scalePoint(subtractPoints(point, center, this.dim), scalar, this.dim), center, this.dim));
      mobject.points = pointArray(rows);
    }
    return this;
  }

  shift(vectorValue: readonly number[]): this {
    const shift = Array.from({ length: this.dim }, (_, index) => Number(vectorValue[index] ?? 0));
    for (const mobject of this.family_members_with_points()) {
      mobject.points = pointArray(pointRows(mobject.points, this.dim).map((point) => addPoints(point, shift, this.dim)));
    }
    return this;
  }

  move_to(pointOrMobject: Mobject | readonly number[] | NDArray, alignedEdge: readonly number[] = ORIGIN): this {
    const target = pointOrMobject instanceof Mobject ? pointOrMobject.get_critical_point(alignedEdge) : vector(pointOrMobject, this.dim);
    return this.shift(subtractPoints(target, this.get_critical_point(alignedEdge), this.dim));
  }

  center(): this { return this.shift(scalePoint(this.get_center(), -1)); }
  move_to_origin(): this { return this.move_to(ORIGIN); }
  set_x(value: number): this { return this.set_coord(value, 0); }
  set_y(value: number): this { return this.set_coord(value, 1); }
  set_z(value: number): this { return this.set_coord(value, 2); }

  set_coord(value: number, dim: number, direction: readonly number[] = ORIGIN): this {
    const point = this.get_critical_point(direction);
    const delta = value - (point[dim] ?? 0);
    const shift = Array.from({ length: this.dim }, (_, index) => index === dim ? delta : 0);
    return this.shift(shift);
  }

  rotate(angle: number, axis: readonly number[] = [0, 0, 1], aboutPoint?: readonly number[]): this {
    const length = norm(axis);
    if (length === 0) return this;
    const unit = scalePoint(axis, 1 / length, 3);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dot = unit[0] * unit[0] + unit[1] * unit[1] + unit[2] * unit[2];
    const center = aboutPoint ? vector(aboutPoint, this.dim) : this.get_center();
    for (const mobject of this.family_members_with_points()) {
      const rotated = pointRows(mobject.points, this.dim).map((point) => {
        const x = point[0] - center[0];
        const y = point[1] - center[1];
        const z = (point[2] ?? 0) - (center[2] ?? 0);
        const crossX = unit[1] * z - unit[2] * y;
        const crossY = unit[2] * x - unit[0] * z;
        const crossZ = unit[0] * y - unit[1] * x;
        const factor = (1 - cos) / dot;
        return [
          center[0] + x * cos + crossX * sin + unit[0] * factor * (unit[0] * x + unit[1] * y + unit[2] * z),
          center[1] + y * cos + crossY * sin + unit[1] * factor * (unit[0] * x + unit[1] * y + unit[2] * z),
          center[2] + z * cos + crossZ * sin + unit[2] * factor * (unit[0] * x + unit[1] * y + unit[2] * z),
        ].slice(0, this.dim);
      });
      mobject.points = pointArray(rotated);
    }
    return this;
  }

  stretch(factor: number, dim: number, aboutPoint?: readonly number[], aboutEdge?: readonly number[]): this {
    const func = (points: NDArray): NDArray => {
      const rows = pointRows(points, this.dim).map((point) => {
        const scaled = [...point];
        scaled[dim] = (scaled[dim] ?? 0) * factor;
        return scaled;
      });
      return pointArray(rows);
    };
    this.apply_points_function_about_point(func, aboutPoint, aboutEdge);
    return this;
  }

  apply_function(func: (point: number[]) => number[] | NDArray, aboutPoint?: readonly number[], aboutEdge?: readonly number[]): this {
    let about_point: readonly number[] | undefined = aboutPoint;
    const about_edge: readonly number[] | undefined = aboutEdge;
    if (about_point === undefined && about_edge === undefined) {
      about_point = ORIGIN;
    }
    const multi_mapping_function = (points: NDArray): NDArray => {
      return pointArray(
        pointRows(points, this.dim).map((point) => vector(func(point), this.dim)),
      );
    };
    this.apply_points_function_about_point(
      multi_mapping_function,
      about_point,
      about_edge,
    );
    return this;
  }

  apply_points_function_about_point(func: (points: NDArray) => NDArray, aboutPoint?: readonly number[], aboutEdge?: readonly number[]): this {
    const center = aboutPoint ? vector(aboutPoint, this.dim) : this.get_critical_point(aboutEdge ?? ORIGIN);
    for (const mobject of this.family_members_with_points()) {
      const centered = pointRows(mobject.points, this.dim).map((point) => subtractPoints(point, center, this.dim));
      const result = pointRows(func(pointArray(centered)), this.dim);
      mobject.points = pointArray(result.map((point) => addPoints(point, center, this.dim)));
    }
    return this;
  }

  apply_function_to_position(func: (point: number[]) => number[] | NDArray): this {
    this.move_to(func(this.get_center()));
    return this;
  }

  applyFunctionToPosition(func: (point: number[]) => number[] | NDArray): this {
    return this.apply_function_to_position(func);
  }

  apply_function_to_submobject_positions(func: (point: number[]) => number[] | NDArray): this {
    for (const submob of this.submobjects) submob.apply_function_to_position(func);
    return this;
  }

  applyFunctionToSubmobjectPositions(func: (point: number[]) => number[] | NDArray): this {
    return this.apply_function_to_submobject_positions(func);
  }

  apply_over_attr_arrays(func: (value: NDArray) => NDArray): this {    for (const attr of this.get_array_attrs()) {
      const current = (this as unknown as Record<string, NDArray>)[attr];
      if (current !== undefined) (this as unknown as Record<string, NDArray>)[attr] = func(current);
    }
    return this;
  }

  apply_complex_function(function_: (value: { re: number; im: number }) => ({ re: number; im: number } | { real: number; imag: number } | number | NDArray), about_point?: readonly number[], about_edge?: readonly number[]): this {
    const R3_func = (point: number[]): NDArray => {
      const [x, y, z = 0] = point;
      const xyComplex = function_({ re: x, im: y });
      if (isNDArray(xyComplex)) return xyComplex;
      if (typeof xyComplex === "number") return array([xyComplex, 0, z]) as NDArray;
      const real = "re" in xyComplex ? xyComplex.re : xyComplex.real;
      const imaginary = "re" in xyComplex ? xyComplex.im : xyComplex.imag;
      return array([real, imaginary, z]) as NDArray;
    };
    return this.apply_function(R3_func, about_point, about_edge);
  }

  repeat(count: number): this {
    const repeat_array = (value: NDArray): NDArray => repeat_array_impl(value, count, this.dim);
    for (const mobject of this.family_members_with_points()) mobject.apply_over_attr_arrays(repeat_array);
    return this;
  }

  repeat_submobject(submob: Mobject): Mobject {
    return submob.copy();
  }

  reverse_points(): this {
    for (const mobject of this.family_members_with_points()) mobject.points = pointArray(pointRows(mobject.points, this.dim).reverse());
    return this;
  }

  copy_array_from(value: unknown): this {
    this.points = pointsFrom(value, this.dim);
    return this;
  }

  set_points(points: unknown): this { return this.copy_array_from(points); }

  get_start(): number[] { this.throw_error_if_no_points("get_start"); return pointAt(this.points, 0, this.dim); }
  get_end(): number[] { this.throw_error_if_no_points("get_end"); return pointAt(this.points, pointRows(this.points, this.dim).length - 1, this.dim); }
  get_midpoint(): NDArray { return this.point_from_proportion(0.5); }

  point_from_proportion(alpha: number): NDArray {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  proportion_from_point(point: readonly number[]): number {
    throw new Error("Please override in a child class.");
  }

  interpolate_color(mobject1: Mobject, mobject2: Mobject, alpha: number): this {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  save_state(): this {
    if ("saved_state" in this && this.saved_state !== undefined) {
      // Prevent exponential growth of data.
      this.saved_state = null;
    }
    this.saved_state = this.copy();
    return this;
  }

  restore(): this {
    if (this.saved_state === null) throw new Error("Trying to restore without having saved");
    this.become(this.saved_state);
    return this;
  }

  set_color(color: ParsableManimColor = "PURE_YELLOW", alpha: unknown = null, family = true): this {
    void alpha;
    this.color = new ManimColor(color);
    if (family) for (const submob of this.submobjects) submob.set_color(color, alpha, family);
    return this;
  }

  setColor(color: ParsableManimColor = "PURE_YELLOW", alpha: unknown = null, family = true): this { return this.set_color(color, alpha, family); }

  set_opacity(opacity: number, family = true): this {
    this.color = this.color.opacity(opacity);
    if (family) for (const submob of this.submobjects) submob.set_opacity(opacity, family);
    return this;
  }

  setOpacity(opacity: number, family = true): this { return this.set_opacity(opacity, family); }

  set_fill(color?: ParsableManimColor, opacity?: number, family = true): this {
    if (color !== undefined) this.set_color(color, null, family);
    if (opacity !== undefined) this.set_opacity(opacity, family);
    return this;
  }

  set_stroke(color?: ParsableManimColor, width?: number, opacity?: number, family = true): this {
    if (color !== undefined) this.set_color(color, null, family);
    if (opacity !== undefined) this.set_opacity(opacity, family);
    return this;
  }

  match_style(other: Mobject, family = true): this {
    this.set_color(other.get_color(), null, family);
    return this;
  }

  match_color(other: Mobject): this { return this.set_color(other.get_color()); }

  fade(darkness = 0.5, family = true): this {
    if (family) for (const submob of this.submobjects) submob.fade(darkness, family);
    return this;
  }

  to_original_color(): this {
    this.set_color(this.color);
    return this;
  }

  toOriginalColor(): this {
    return this.to_original_color();
  }

  fade_to(color: ParsableManimColor, alpha: number, family = true): this {
    if (this.get_num_points() > 0) this.set_color(interpolate_color(this.get_color(), new ManimColor(color), alpha), null, false);
    if (family) for (const submob of this.submobjects) submob.fade_to(color, alpha);
    return this;
  }

  add_background_rectangle(
    color?: ParsableManimColor | null,
    opacity = 0.75,
    kwargs: Record<string, unknown> = {},
  ): this {
    if (backgroundRectangleFactory === null) {
      throw new Error("TODO: add_background_rectangle needs BackgroundRectangle (manim.mobject.geometry.shape_matchers), not ported yet");
    }
    this.background_rectangle = backgroundRectangleFactory(this, color ?? null, opacity, kwargs);
    this.add_to_back(this.background_rectangle);
    return this;
  }

  addBackgroundRectangle(
    color?: ParsableManimColor | null,
    opacity = 0.75,
    kwargs: Record<string, unknown> = {},
  ): this {
    return this.add_background_rectangle(color, opacity, kwargs);
  }

  add_background_rectangle_to_submobjects(kwargs: Record<string, unknown> = {}): this {
    const { color = null, opacity = 0.75, ...rest } = kwargs as {
      color?: ParsableManimColor | null;
      opacity?: number;
      [key: string]: unknown;
    };
    for (const submobject of this.submobjects) submobject.add_background_rectangle(color, opacity, rest);
    return this;
  }

  addBackgroundRectangleToSubmobjects(kwargs: Record<string, unknown> = {}): this {
    return this.add_background_rectangle_to_submobjects(kwargs);
  }

  add_updater(update_function: MobjectUpdater, index: number | null = null, call_updater = false): this {
    if (index === null || index === undefined) this.updaters.push(update_function);
    else this.updaters.splice(index, 0, update_function);
    if (call_updater) {
      if (update_function.length >= 2) (update_function as (mobject: Mobject, dt: number) => void)(this, 0);
      else (update_function as (mobject: Mobject) => void)(this);
    }
    return this;
  }

  addUpdater(update_function: MobjectUpdater, index: number | null = null, call_updater = false): this {
    return this.add_updater(update_function, index, call_updater);
  }

  remove_updater(updateFunction: MobjectUpdater): this {
    this.updaters = this.updaters.filter((updater) => updater !== updateFunction);
    return this;
  }

  update(dt = 0, recursive = true): this {
    if (this.updating_suspended) return this;
    for (const updater of [...this.updaters]) {
      if (updater.length >= 2) (updater as (mobject: Mobject, dt: number) => void)(this, dt);
      else (updater as (mobject: Mobject) => void)(this);
    }
    if (recursive) for (const submob of this.submobjects) submob.update(dt, recursive);
    return this;
  }

  clear_updaters(recursive = true): this {
    this.updaters = [];
    if (recursive) for (const submob of this.submobjects) submob.clear_updaters();
    return this;
  }

  clearUpdaters(recursive = true): this {
    return this.clear_updaters(recursive);
  }

  match_updaters(other: Mobject): this {
    this.clear_updaters();
    for (const updater of other.get_updaters()) this.add_updater(updater);
    return this;
  }

  matchUpdaters(other: Mobject): this {
    return this.match_updaters(other);
  }

  suspend_updating(recursive = true): this {
    this.updating_suspended = true;
    if (recursive) for (const submob of this.submobjects) submob.suspend_updating(recursive);
    return this;
  }

  resume_updating(recursive = true): this {
    this.updating_suspended = false;
    if (recursive) for (const submob of this.submobjects) submob.resume_updating(recursive);
    this.update(0, recursive);
    return this;
  }
  suspendUpdating(recursive = true): this { return this.suspend_updating(recursive); }
  resumeUpdating(recursive = true): this { return this.resume_updating(recursive); }

  get_array_attrs_with_values(): Record<string, NDArray> {
    const result: Record<string, NDArray> = {};
    for (const attr of this.get_array_attrs()) result[attr] = (this as unknown as Record<string, NDArray>)[attr];
    return result;
  }

  set(options: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(options)) (this as unknown as Record<string, unknown>)[key] = value;
    return this;
  }

  __getattr__(attr: string): FunctionLike {
    if (attr.startsWith("get_")) {
      const to_get = attr.slice(4);
      return () => {
        if (!(to_get in this)) {
          const error = new Error(`${this.constructor.name} object has no attribute '${to_get}'`);
          error.name = "AttributeError";
          throw error;
        }
        return (this as unknown as Record<string, unknown>)[to_get];
      };
    }
    if (attr.startsWith("set_")) {
      const to_set = attr.slice(4);
      return (value: unknown) => {
        (this as unknown as Record<string, unknown>)[to_set] = value;
        return this;
      };
    }
    const error = new Error(`${this.constructor.name} object has no attribute '${attr}'`);
    error.name = "AttributeError";
    throw error;
  }

  __getattr_getter(): FunctionLike {
    return this.__getattr__("get_foo");
  }

  __getattr_setter(): FunctionLike {
    return this.__getattr__("set_foo");
  }

  arrange(direction: readonly number[] = RIGHT, buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, center = true): this {
    for (let index = 0; index < this.submobjects.length - 1; index += 1) this.submobjects[index + 1].next_to(this.submobjects[index], direction, buff);
    if (center) this.center();
    return this;
  }

  next_to(mobject: Mobject, direction: readonly number[] = RIGHT, buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER): this {
    const directionVector = vector(direction, this.dim);
    const target = addPoints(mobject.get_critical_point(directionVector), scalePoint(directionVector, buff + this.length_over_dim(0) / 2, this.dim));
    return this.move_to(target);
  }

  arrange_in_grid(
    rows: number | null = null,
    cols: number | null = null,
    buff: number | [number, number] = MED_SMALL_BUFF,
    cell_alignment: readonly number[] = ORIGIN,
    row_alignments: string | null = null,
    col_alignments: string | null = null,
    row_heights: Iterable<number | null | undefined> | null = null,
    col_widths: Iterable<number | null | undefined> | null = null,
    flow_order = "rd",
  ): this {
    const mobs = [...this.submobjects];
    const startPos = this.get_center();
    const row_heights_list = row_heights === null ? null : [...row_heights];
    const col_widths_list = col_widths === null ? null : [...col_widths];
    let columnCount = init_size(cols, col_alignments, col_widths_list);
    let rowCount = init_size(rows, row_alignments, row_heights_list);
    if (rowCount === null && columnCount === null) columnCount = Math.ceil(Math.sqrt(mobs.length));
    if (rowCount === null) rowCount = Math.ceil(mobs.length / (columnCount as number));
    if (columnCount === null) columnCount = Math.ceil(mobs.length / rowCount);
    if (rowCount * columnCount < mobs.length) {
      const error = new Error("Too few rows and columns to fit all submobjetcs.");
      error.name = "ValueError";
      throw error;
    }
    const buffX = Array.isArray(buff) ? buff[0] : buff;
    const buffY = Array.isArray(buff) ? buff[1] : buff;
    const rowDirections = init_alignments(row_alignments, rowCount, { u: [0, 1, 0], c: [0, 0, 0], d: [0, -1, 0] }, "row", [1, 0, 0], vector(cell_alignment, 3));
    const colDirections = init_alignments(col_alignments, columnCount, { l: [-1, 0, 0], c: [0, 0, 0], r: [1, 0, 0] }, "col", [0, 1, 0], vector(cell_alignment, 3));
    const mappers: Record<string, (r: number, c: number) => number> = {
      dr: (r, c) => (rowCount - r - 1) + c * rowCount,
      dl: (r, c) => (rowCount - r - 1) + (columnCount - c - 1) * rowCount,
      ur: (r, c) => r + c * rowCount,
      ul: (r, c) => r + (columnCount - c - 1) * rowCount,
      rd: (r, c) => (rowCount - r - 1) * columnCount + c,
      ld: (r, c) => (rowCount - r - 1) * columnCount + (columnCount - c - 1),
      ru: (r, c) => r * columnCount + c,
      lu: (r, c) => r * columnCount + (columnCount - c - 1),
    };
    const mapper = mappers[flow_order];
    if (!mapper) {
      const error = new Error('flow_order must be one of the following values: "dr", "rd", "ld" "dl", "ru", "ur", "lu", "ul".');
      error.name = "ValueError";
      throw error;
    }
    rowDirections.reverse();
    const rowHeightsList = row_heights_list ? [...row_heights_list].reverse() : [];
    const colWidthsList = col_widths_list ? [...col_widths_list] : [];
    const placeholder = new Mobject();
    while (mobs.length < rowCount * columnCount) mobs.push(placeholder);
    const grid = Array.from({ length: rowCount }, (_, r) => Array.from({ length: columnCount }, (_, c) => mobs[mapper(r, c)]));
    const measuredHeights = grid.map((row) => Math.max(...row.map((mob) => mob.height)));
    const measuredWidths = Array.from({ length: columnCount }, (_, c) => Math.max(...grid.map((row) => row[c].width)));
    const heights = init_sizes(rowHeightsList, rowCount, measuredHeights, "row_heights");
    const widths = init_sizes(colWidthsList, columnCount, measuredWidths, "col_widths");
    let y = 0;
    for (let r = 0; r < rowCount; r += 1) {
      let x = 0;
      for (let c = 0; c < columnCount; c += 1) {
        const mob = grid[r][c];
        if (mob !== placeholder) {
          const alignment = addPoints(rowDirections[r], colDirections[c], 3);
          const start = [x, y, 0];
          const end = [x + widths[c], y + heights[r], 0];
          const cell = new Mobject();
          cell.points = pointsFrom([start, end], 3);
          mob.move_to(cell, alignment);
        }
        x += widths[c] + buffX;
      }
      y += heights[r] + buffY;
    }
    return this.move_to(startPos);
  }

  set_z_index(z_index_value: number, family = true): this {
    if (family) for (const submob of this.submobjects) submob.set_z_index(z_index_value, family);
    this.z_index = z_index_value;
    return this;
  }

  setZIndex(value: number, family = true): this { return this.set_z_index(value, family); }

  get_z_index_reference_point(): number[] { return this.get_center(); }

  throw_error_if_no_points(methodName = "method"): void {
    if (!this.has_no_points()) return;
    const pointful = this.family_members_with_points();
    let message = `Cannot call ${this.constructor.name}.${methodName} because ${this.toString()} has no points.`;
    if (pointful.length > 0) message += ` Its family contains ${pointful.length} mobject${pointful.length === 1 ? "" : "s"} with points.`;
    const error = new Error(message);
    error.name = "ValueError";
    throw error;
  }

  become(other: Mobject): this {
    this.points = pointsFrom(other.points, this.dim);
    this.set_color(other.get_color(), null, false);
    this.submobjects = other.submobjects.map((child) => child.copy());
    return this;
  }

  replace(other: Mobject, dim_to_match = 0, stretch = false): this {
    if (stretch) {
      this.scale_to_fit_width(other.width);
      this.scale_to_fit_height(other.height);
    } else {
      const currentLength = this.length_over_dim(dim_to_match);
      const targetLength = other.length_over_dim(dim_to_match);
      if (currentLength !== 0) this.scale(targetLength / currentLength);
    }
    return this.shift([other.get_center()[0] - this.get_center()[0], other.get_center()[1] - this.get_center()[1], other.get_center()[2] - this.get_center()[2]]);
  }

  match_points(other: Mobject): this {
    for (let index = 0; index < this.get_family().length; index += 1) {
      const source = other.get_family()[index];
      if (source) this.get_family()[index].points = pointsFrom(source.points, this.dim);
    }
    return this;
  }

  align_data(other: Mobject, skipPointAlignment = false): this {
    this.null_point_align(other);
    this.align_submobjects(other);
    if (!skipPointAlignment) this.align_points(other);
    for (let index = 0; index < this.submobjects.length; index += 1) this.submobjects[index].align_data(other.submobjects[index]);
    return this;
  }

  null_point_align(other: Mobject): this {
    if (this.has_no_points() && other.has_points()) other.push_self_into_submobjects();
    if (other.has_no_points() && this.has_points()) this.push_self_into_submobjects();
    return this;
  }

  align_submobjects(other: Mobject): this {
    while (this.submobjects.length < other.submobjects.length) this.submobjects.push(new Mobject());
    while (other.submobjects.length < this.submobjects.length) other.submobjects.push(new Mobject());
    return this;
  }

  align_points(other: Mobject): this {
    const count1 = this.get_num_points();
    const count2 = other.get_num_points();
    if (count1 < count2) this.align_points_with_larger(other);
    else if (count2 < count1) other.align_points_with_larger(this);
    return this;
  }

  align_points_with_larger(_larger_mobject: Mobject): this {
    const error = new Error("Please override in a child class.");
    error.name = "NotImplementedError";
    throw error;
  }

  push_self_into_submobjects(): this {
    const copy = this.copy();
    copy.submobjects = [];
    this.reset_points();
    this.add(copy);
    return this;
  }

  *__iter__(): IterableIterator<Mobject> {
    if (this.has_points()) yield this;
    yield* this.submobjects;
  }

  *[Symbol.iterator](): IterableIterator<Mobject> {
    yield* this.__iter__();
  }

  __len__(): number { return this.submobjects.length + (this.has_points() ? 1 : 0); }

  split(): Mobject[] {
    const result: Mobject[] = this.get_num_points() > 0 ? [this] : [];
    return [...result, ...this.submobjects];
  }

  nonempty_submobjects(): Mobject[] {
    return this.submobjects.filter(
      (submob) => submob.submobjects.length !== 0 || submob.get_num_points() !== 0,
    );
  }

  reduce_across_dimension(
    reduce_func: (values: Iterable<number>) => number,
    dim: number,
  ): number | null {
    if (!(dim >= 0 && dim <= 2)) {
      throw new Error(`AssertionError: dim ${dim} is not in [0, 2]`);
    }
    if (this.submobjects.length === 0 && this.get_num_points() === 0) {
      // If we have no points and no submobjects, return None.
      return null;
    }

    // If we do not have points (but do have submobjects) use only the
    // points from those.
    let rv: number | null;
    if (this.get_num_points() === 0) {
      rv = null;
    } else {
      // Otherwise, be sure to include our own points.
      rv = reduce_func(
        pointRows(this.points, this.dim).map((point) => point[dim] ?? 0),
      );
    }
    // Recursively ask submobjects (if any) for the biggest/smallest
    // dimension they have and compare it to the return value.
    for (const mobj of this.submobjects) {
      const value = mobj.reduce_across_dimension(reduce_func, dim);
      if (rv === null) {
        rv = value;
      } else if (value !== null) {
        rv = reduce_func([value, rv]);
      }
    }
    return rv;
  }

  get_merged_array(array_attr: string): NDArray {
    let result = (this as unknown as Record<string, NDArray>)[array_attr];
    for (const submob of this.submobjects) {
      result = concatenate([result, submob.get_merged_array(array_attr)], 0) as NDArray;
    }
    return result;
  }

  get_all_points(): NDArray {
    return this.get_merged_array("points");
  }

  nonemptySubmobjects(): Mobject[] {
    return this.nonempty_submobjects();
  }

  shuffle(recursive = false): this {
    if (recursive) for (const submob of this.submobjects) submob.shuffle(true);
    // Fisher-Yates, mirroring random.shuffle.
    for (let index = this.submobjects.length - 1; index > 0; index -= 1) {
      const other = Math.floor(Math.random() * (index + 1));
      const temp = this.submobjects[index] as Mobject;
      this.submobjects[index] = this.submobjects[other] as Mobject;
      this.submobjects[other] = temp;
    }
    return this;
  }

  shuffle_submobjects(...args: unknown[]): this {
    const [recursive = false] = args as [boolean?];
    return this.shuffle(recursive);
  }

  shuffleSubmobjects(...args: unknown[]): this {
    return this.shuffle_submobjects(...args);
  }

  invert(recursive = false): this {
    if (recursive) for (const submob of this.submobjects) submob.invert(true);
    this.submobjects.reverse();
    return this;
  }

  interpolate(mobject1: Mobject, mobject2: Mobject, alpha: number, path_func: PathFuncType | null = null): this {
    const func = (path_func ?? straight_path()) as PathFuncType;
    this.points = func(mobject1.points, mobject2.points, alpha);
    this.interpolate_color(mobject1, mobject2, alpha);
    return this;
  }

  __getitem__(index: number | { start: number; end: number }): Mobject | Mobject[] {
    if (typeof index === "number") {
      const value = this.submobjects[index];
      if (!value) throw new Error("Mobject index out of range");
      return value;
    }
    return this.submobjects.slice(index.start, index.end);
  }

  toString(): string { return this.name; }
  valueOf(): string { return this.toString(); }
}

export class Group extends Mobject {
  constructor(...mobjects: (Mobject | Iterable<Mobject>)[]) {
    super();
    this.add(...mobjects);
  }
}

export class _MethodAnimation extends Animation {
  methods: MethodWithArgs[];
  target_mobject: Mobject;

  constructor(mobject: Mobject, methods: MethodWithArgs[]) {
    super(mobject);
    this.methods = methods;
    this.target_mobject = mobject.target ?? mobject.copy();
  }

  finish(): void {
    for (const item of this.methods) item.method.apply(this.mobject, Object.keys(item.kwargs).length > 0 ? [...item.args, item.kwargs] : item.args);
    super.finish();
  }
}

export class _AnimationBuilder {
  mobject: Mobject;
  overridden_animation: any = null;
  is_chaining = false;
  methods: MethodWithArgs[] = [];
  cannot_pass_args = false;
  anim_args: Record<string, unknown> = {};

  constructor(mobject: Mobject) {
    this.mobject = mobject;
    this.mobject.generate_target();
    let callableProxy: _AnimationBuilder;
    const builder = function (...args: any[]): _AnimationBuilder {
      builder.call((args[0] as Record<string, unknown> | undefined) ?? {});
      return callableProxy;
    } as unknown as _AnimationBuilder;
    Object.setPrototypeOf(builder, _AnimationBuilder.prototype);
    Object.assign(builder, this);
    callableProxy = new Proxy(builder, {
      get: (target, property, receiver) => {
        if (property in target || typeof property === "symbol") return Reflect.get(target, property, receiver);
        return (...args: any[]) => target.update_target(String(property), args, {});
      },
    });
    (builder as unknown as { _callable_proxy: _AnimationBuilder })._callable_proxy = callableProxy;
    return callableProxy;
  }

  __getattr__(method_name: string): (...args: any[]) => this {
    return (...args: any[]) => this.update_target(method_name, args, {});
  }

  call(args: Record<string, unknown> = {}): this {
    if (this.cannot_pass_args) {
      const error = new Error("Animation arguments must be passed before accessing methods and can only be passed once");
      error.name = "ValueError";
      throw error;
    }
    this.anim_args = args;
    this.cannot_pass_args = true;
    return this;
  }

  update_target(method_name: string, method_args: any[] = [], method_kwargs: Record<string, any> = {}): this {
    const target = this.mobject.target ?? this.mobject;
    const method = (target as unknown as Record<string, unknown>)[method_name];
    if (typeof method !== "function") throw new Error(`${this.mobject.constructor.name} has no method '${method_name}'`);
    const override = get_animation_override(method as FunctionLike);
    if ((this.is_chaining && override) || this.overridden_animation) {
      const error = new Error("Method chaining is currently not supported for overridden animations");
      error.name = "NotImplementedError";
      throw error;
    }
    if (override) this.overridden_animation = override(this.mobject, ...method_args, { anim_args: this.anim_args, ...method_kwargs });
    else {
      this.methods.push(new MethodWithArgs(method as FunctionLike, method_args, method_kwargs));
      (method as FunctionLike).apply(target, Object.keys(method_kwargs).length > 0 ? [...method_args, method_kwargs] : method_args);
    }
    this.is_chaining = true;
    this.cannot_pass_args = true;
    return ((this as unknown as { _callable_proxy?: _AnimationBuilder })._callable_proxy ?? this) as this;
  }

  build(): any {
    const animation = this.overridden_animation ?? new _MethodAnimation(this.mobject, this.methods);
    for (const [key, value] of Object.entries(this.anim_args)) (animation as Record<string, unknown>)[key] = value;
    return animation;
  }
}

export function createAnimationBuilder(mobject: Mobject): _AnimationBuilder {
  return new _AnimationBuilder(mobject);
}

export class _UpdaterBuilder {
  _mobject: Mobject;

  constructor(mobject: Mobject) {
    this._mobject = mobject;
    const proxy = new Proxy(this, {
      get: (target, property, receiver) => {
        if (property in target || typeof property === "symbol") return Reflect.get(target, property, receiver);
        return (...args: any[]) => {
          target.add_updater(String(property), ...args);
          return (target as unknown as { _proxy?: _UpdaterBuilder })._proxy ?? target;
        };
      },
    });
    (this as unknown as { _proxy: _UpdaterBuilder })._proxy = proxy;
    return proxy;
  }

  __getattr__(name: string): (...args: any[]) => this {
    return (...args: any[]) => this.add_updater(name, ...args);
  }

  add_updater(method_name: string, ...method_args: any[]): this {
    const possible_kwargs = method_args.length > 0 && method_args[method_args.length - 1] !== null && typeof method_args[method_args.length - 1] === "object" && !Array.isArray(method_args[method_args.length - 1]) ? method_args[method_args.length - 1] as Record<string, unknown> : null;
    const positional_args = possible_kwargs === null ? method_args : method_args.slice(0, -1);
    this._mobject.add_updater((mob: Mobject) => {
      const method = (mob as unknown as Record<string, unknown>)[method_name];
      if (typeof method === "function") (method as FunctionLike).apply(mob, possible_kwargs === null ? positional_args : [...positional_args, possible_kwargs]);
    }, undefined, true);
    return ((this as unknown as { _proxy?: _UpdaterBuilder })._proxy ?? this) as this;
  }
}

export function createUpdaterBuilder(mobject: Mobject): _UpdaterBuilder {
  return new _UpdaterBuilder(mobject);
}

export function override_animate(method: FunctionLike): (animationMethod: FunctionLike) => FunctionLike {
  function decorator(animationMethod: FunctionLike): FunctionLike {
    overrideFunctions.set(method, animationMethod);
    (method as FunctionLike & { _override_animate?: FunctionLike })._override_animate = animationMethod;
    return animationMethod;
  }
  return decorator;
}
