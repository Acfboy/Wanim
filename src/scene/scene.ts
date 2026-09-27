import { platform } from "node:os";
import { array, matmul, transpose, type NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { DEFAULT_WAIT_TIME, OUT, RendererType, SHIFT_VALUE } from "../constants.js";
import { MethodWithArgs } from "../data_structures.js";
import { Mobject, _AnimationBuilder } from "../mobject/mobject.js";
import { pointRows } from "../utils/arrays.js";
import { extract_mobject_family_members } from "../utils/family.js";
import { list_difference_update, list_update } from "../utils/iterables.js";
import { flatten_iterable_parameters } from "../utils/parameter_parsing.js";
import { cross, normalize, rotation_matrix } from "../utils/space_ops.js";
import { Animation, Wait, prepare_animation } from "../animation/animation.js";
import { AnimationGroup } from "../animation/composition.js";
import { EndSceneEarlyException, RerunSceneException } from "../utils/exceptions.js";
import { is_gif_format, open_file } from "../utils/file_ops.js";
import {
  get_render_scheduler,
  is_async_rendering,
  is_thenable,
  type Awaitable,
  type RenderScheduler,
} from "../utils/scheduler.js";
import { DefaultSectionType } from "./section.js";

/**
 * Mirrors `manim.utils.file_ops.open_media_file`: opens the files produced
 * by `file_writer` (last-frame image and/or movie) in a viewer or file
 * browser, driven by `config.preview` / `config.show_in_file_browser`.
 */
function open_media_file(file_writer: any): void {
  const file_paths: string[] = [];
  if (config.save_last_frame) {
    file_paths.push(file_writer.image_file_path);
  }
  if (config.get("write_to_movie") && !is_gif_format()) {
    file_paths.push(file_writer.movie_file_path);
  }
  if (config.get("write_to_movie") && is_gif_format()) {
    file_paths.push(file_writer.gif_file_path);
  }
  for (const file_path of file_paths) {
    if (config.show_in_file_browser) {
      open_file(file_path, true);
    }
    if (config.preview) {
      open_file(file_path, false);
      // NOTE: mirrors Python `logger.info(...)`; the TS port has no shared
      // global logger, so the message goes to the console (same convention
      // as the `console.warn` calls elsewhere in this file).
      console.info(`Previewed File at: '${file_path}'`);
    }
  }
}

export type SceneInteractAction = SceneInteractRerun | SceneInteractContinue | MethodWithArgs | string;

/** Mirrors manim.utils.opengl.translation_matrix (4x4 homogeneous). */
function translation_matrix_4x4(x = 0, y = 0, z = 0): NDArray {
  return array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1],
  ]) as NDArray;
}

/** Mirrors manim.utils.opengl.z_rotation_matrix (4x4 homogeneous). */
function z_rotation_matrix_4x4(z = 0): NDArray {
  const c = Math.cos(z);
  const s = Math.sin(z);
  return array([
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]) as NDArray;
}

function vec3(value: number[] | readonly number[] | NDArray): number[] {
  return [...(pointRows(value, 3)[0] ?? [0, 0, 0])];
}

function mat44(value: NDArray): number[][] {
  const rows = pointRows(value, 4);
  while (rows.length < 4) rows.push([0, 0, 0, 0]);
  return rows.slice(0, 4);
}

function mat44Column(matrix: NDArray, column: number, length = 3): number[] {
  return mat44(matrix).slice(0, length).map((row) => row[column] ?? 0);
}

export class SceneInteractContinue {
  sender: string;

  constructor(sender: string) {
    this.sender = sender;
  }
}

export class SceneInteractRerun {
  sender: string;
  kwargs: Record<string, unknown>;

  constructor(sender: string, kwargs: Record<string, unknown> = {}) {
    this.sender = sender;
    this.kwargs = kwargs;
  }
}

export class RerunSceneHandler {
  queue: Array<SceneInteractAction> | { put: (action: SceneInteractAction) => void };

  constructor(
    queue: Array<SceneInteractAction> | { put: (action: SceneInteractAction) => void },
  ) {
    this.queue = queue;
  }

  on_modified(_event: unknown): void {
    const action = new SceneInteractRerun("file");
    if (Array.isArray(this.queue)) this.queue.push(action);
    else this.queue.put(action);
  }
}

export function load_module_into_namespace(
  module: Record<string, unknown>,
  namespace: Record<string, unknown>,
): void {
  for (const name in module) namespace[name] = (module as Record<string, unknown>)[name];
}

function formatG(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Number.isInteger(value) && Math.abs(value) < 1e21) return String(value);
  let text = value.toPrecision(6);
  if (text.includes("e") || text.includes("E")) {
    text = text.replace(/(\.\d*?)0+e/, "$1e").replace(/\.e/, "e");
  } else if (text.includes(".")) {
    text = text.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  }
  return text;
}

export class TimeProgression implements Iterable<number> {
  times: number[];
  description: string;
  total: number | null;
  leave: boolean;
  ascii: boolean | null;
  disabled: boolean;

  constructor(
    times: number[],
    description: string,
    total: number | null = null,
    leave = true,
    ascii: boolean | null = null,
    disabled = false,
  ) {
    this.times = times;
    this.description = description;
    this.total = total;
    this.leave = leave;
    this.ascii = ascii;
    this.disabled = disabled;
  }

  [Symbol.iterator](): IterableIterator<number> {
    return this.times[Symbol.iterator]();
  }

  /** Mirrors the tqdm progress-bar `close()` (a no-op here; kept for API parity). */
  close(): void {}
}

/**
 * Minimal mirror of `srt.Subtitle` (index/content/start/end), with the
 * `datetime.timedelta` bounds stored as seconds since the scene start.
 * The `srt` package has no TS equivalent, so this local stand-in keeps
 * `add_subcaption` behavior identical for the fields manim uses.
 */
export class Subtitle {
  index: number;
  content: string;
  start: number;
  end: number;

  constructor(index = 0, content = "", start = 0, end = 0) {
    this.index = index;
    this.content = content;
    this.start = start;
    this.end = end;
  }
}

export interface SceneConstructorOptions {
  renderer?: any;
  camera_class?: new (...args: any[]) => any;
  always_update_mobjects?: boolean;
  random_seed?: number | null;
  skip_animations?: boolean;
}

function isSceneOptionsBag(value: unknown): value is SceneConstructorOptions {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return false;
  return ["renderer", "camera_class", "always_update_mobjects", "random_seed", "skip_animations"].some((key) => key in (value as Record<string, unknown>));
}

export interface ScenePlayOptions {
  subcaption?: string | null;
  subcaption_duration?: number | null;
  subcaption_offset?: number;
  [key: string]: unknown;
}

function isScenePlayOptions(value: unknown): value is ScenePlayOptions {
  // Any trailing plain object is the play-options bag (Animation and Mobject
  // instances carry class prototypes and never match).
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Local stand-in for `copy.deepcopy(value, memo)` used by
 * `Scene.__deepcopy__` (the TS runtime has no `copy` module). Mobjects go
 * through their memo-aware `copy_with_seen`; arrays, plain objects, Maps
 * and Sets recurse; class instances exposing `copy()` (Animation, NDArray,
 * ...) use it; anything else (functions, class references) is atomic.
 */
function sceneDeepClone(value: unknown, memo: Map<object, unknown>): unknown {
  if (value === null || value === undefined || typeof value !== "object") {
    return value;
  }
  const known = memo.get(value as object);
  if (known !== undefined) return known;
  if (value instanceof Mobject) return value.copy_with_seen(memo);
  if (Array.isArray(value)) {
    const result: unknown[] = [];
    memo.set(value, result);
    for (const item of value) result.push(sceneDeepClone(item, memo));
    return result;
  }
  if (value instanceof Map) {
    const result = new Map<unknown, unknown>();
    memo.set(value, result);
    for (const [key, item] of value) result.set(key, sceneDeepClone(item, memo));
    return result;
  }
  if (value instanceof Set) {
    const result = new Set<unknown>();
    memo.set(value, result);
    for (const item of value) result.add(sceneDeepClone(item, memo));
    return result;
  }
  const proto = Object.getPrototypeOf(value);
  if (proto === Object.prototype || proto === null) {
    const result: Record<string, unknown> = {};
    memo.set(value, result);
    for (const [key, item] of Object.entries(value)) {
      result[key] = sceneDeepClone(item, memo);
    }
    return result;
  }
  const copyable = value as { copy?: () => unknown };
  if (typeof copyable.copy === "function") {
    const result = copyable.copy();
    memo.set(value, result);
    return result;
  }
  return value;
}

export class Scene {
  mobjects: Mobject[] = [];
  foreground_mobjects: Mobject[] = [];
  meshes: Mobject[] = [];
  animations: Animation[] = [];
  stop_condition: (() => boolean) | null = null;
  moving_mobjects: Mobject[] = [];
  static_mobjects: Mobject[] = [];
  time_progression: TimeProgression | null = null;
  duration = 0;
  last_t = 0;
  camera: any = null;
  renderer: any = null;
  camera_class: new (...args: any[]) => any;
  key_to_function_map: Map<string, () => void> = new Map();
  mouse_press_callbacks: Array<() => void> = [];
  interactive_mode = false;
  quit_interaction = false;
  camera_target: number[] = [0, 0, 0];
  // NOTE: Python uses OpenGLPoint (an OpenGLMobject) here; the OpenGL
  // renderer is not ported yet, so plain Mobjects track the positions.
  mouse_point: Mobject = new Mobject();
  mouse_drag_point: Mobject = new Mobject();
  always_update_mobjects: boolean;
  random_seed: number | null;
  skip_animations: boolean;
  skip_animation_preview = false;
  queue: Array<SceneInteractAction> = [];
  widgets: Array<Record<string, unknown>> = [];
  dearpygui_imported = false;
  subcaptions: Subtitle[] = [];

  constructor(
    renderer?: SceneConstructorOptions | any,
    camera_class?: (new (...args: any[]) => any) | SceneConstructorOptions | null,
    always_update_mobjects: boolean | SceneConstructorOptions = false,
    random_seed?: number | null,
    skip_animations = false,
  ) {
    // NOTE: Python's signature is
    // `(renderer, camera_class, always_update_mobjects, random_seed,
    // skip_animations)`. The TS port additionally accepts a single trailing
    // plain-object options bag (used by tests and TS callers), e.g.
    // `new Scene({ skip_animations: true })`.
    let resolvedRenderer: any = renderer ?? null;
    let resolvedCameraClass: (new (...args: any[]) => any) | null | undefined = camera_class as new (...args: any[]) => any;
    let resolvedAlwaysUpdate = always_update_mobjects as boolean;
    let resolvedSeed = random_seed;
    let resolvedSkip = skip_animations;
    const trailing = [renderer, camera_class, always_update_mobjects].find((value) =>
      isSceneOptionsBag(value),
    ) as SceneConstructorOptions | undefined;
    if (trailing !== undefined) {
      if (trailing.renderer !== undefined) resolvedRenderer = trailing.renderer;
      if (trailing.camera_class !== undefined) {
        resolvedCameraClass = trailing.camera_class as new (...args: any[]) => any;
      }
      if (trailing.always_update_mobjects !== undefined) {
        resolvedAlwaysUpdate = trailing.always_update_mobjects;
      }
      if (trailing.random_seed !== undefined) resolvedSeed = trailing.random_seed;
      if (trailing.skip_animations !== undefined) resolvedSkip = trailing.skip_animations;
    }
    this.camera_class = (resolvedCameraClass ?? class {}) as new (...args: any[]) => any;
    this.always_update_mobjects = resolvedAlwaysUpdate ?? false;
    // NOTE: non-numeric values (e.g. an options bag forwarded by a subclass
    // super() call) fall back to the configured seed, mirroring Python's
    // `random_seed if random_seed is not None else config.seed`.
    this.random_seed = typeof resolvedSeed === "number"
      ? resolvedSeed
      : ((config.seed as number | null) ?? null);
    this.skip_animations = resolvedSkip ?? false;
    this.renderer = resolvedRenderer;
    this.camera = this.camera_class === null ? null : new this.camera_class();
    // Adopt the renderer's camera when present so the scene and the renderer
    // (e.g. the serializer's JSONCairoRenderer) share one camera instance.
    if (this.renderer?.camera !== undefined && this.renderer?.camera !== null) {
      this.camera = this.renderer.camera;
    }
    this.renderer?.init_scene?.(this);
    // NOTE: Python seeds the `random` and `numpy.random` singletons here.
    // The TS runtime has no seedable global RNG, so only the field above is
    // stored (see ).
  }

  /** The time since the start of the scene. */
  get time(): number {
    return (this.renderer?.time as number | undefined) ?? 0;
  }

  /**
   * Deep-copy support mirroring Python's `Scene.__deepcopy__`: every field
   * is cloned into a fresh instance of the same class except `renderer`
   * and `time_progression` (skipped), while `camera_class` is kept by
   * reference (Python's `copy.deepcopy` of a class is the identity).
   */
  __deepcopy__(clone_from_id: Map<object, unknown> = new Map()): this {
    const raw = Object.create(Object.getPrototypeOf(this)) as this;
    clone_from_id.set(this, raw);
    for (const [key, value] of Object.entries(this)) {
      if (key === "renderer" || key === "time_progression") continue;
      if (key === "camera_class") {
        (raw as unknown as Record<string, unknown>)[key] = value;
        continue;
      }
      (raw as unknown as Record<string, unknown>)[key] = sceneDeepClone(
        value,
        clone_from_id,
      );
    }
    return raw;
  }

  add(...mobjects: Mobject[]): this {
    // NOTE: mirrors Python Scene.add. The OpenGL branch splits Object3D
    // meshes from mobjects; the OpenGL renderer (and Object3D) is not
    // ported, so every argument is treated as a mobject there (same
    // limitation as remove() below documents).
    if (config.renderer === RendererType.OPENGL) {
      const new_mobjects = [...mobjects];
      this.remove(...new_mobjects);
      this.mobjects.push(...new_mobjects);
      return this;
    }
    const new_and_foreground_mobjects: Mobject[] = [
      ...mobjects,
      ...this.foreground_mobjects,
    ];
    this.restructure_mobjects(new_and_foreground_mobjects);
    this.mobjects.push(...new_and_foreground_mobjects);
    if (this.moving_mobjects.length > 0) {
      this.restructure_mobjects(
        new_and_foreground_mobjects,
        "moving_mobjects",
      );
      this.moving_mobjects.push(...new_and_foreground_mobjects);
    }
    return this;
  }

  add_mobjects_from_animations(animations: Animation[]): void {
    let curr_mobjects = this.get_mobject_family_members();
    for (const animation of animations) {
      if (animation.is_introducer()) continue;
      // Anything animated that's not already in the
      // scene gets added to the scene
      const mob = animation.mobject;
      if (mob !== null && mob !== undefined && !curr_mobjects.includes(mob)) {
        this.add(mob);
        curr_mobjects = [...curr_mobjects, ...mob.get_family()];
      }
    }
  }

  remove(...mobjects: Mobject[]): this {
    if (config.renderer === RendererType.OPENGL) {
      const mobjects_to_remove: Mobject[] = [];
      const meshes_to_remove = new Set<Mobject>();
      // NOTE: Python splits Object3D meshes from Mobjects here; the OpenGL
      // renderer (and Object3D) is not ported yet, so every argument is
      // treated as a mesh for the meshes list, mirroring the previous
      // behavior of this port.
      for (const mobject_or_mesh of mobjects) {
        mobjects_to_remove.push(mobject_or_mesh);
        meshes_to_remove.add(mobject_or_mesh);
      }
      this.mobjects = this.get_restructured_mobject_list(this.mobjects, mobjects_to_remove);

      function lambda_function(mesh: Mobject): boolean {
        return !meshes_to_remove.has(mesh);
      }
      this.meshes = this.meshes.filter(lambda_function);
      return this;
    } else {
      for (const list_name of ["mobjects", "foreground_mobjects"] as const) {
        this.restructure_mobjects(mobjects, list_name, false);
      }
      return this;
    }
  }

  clear(): this {
    this.mobjects = [];
    this.foreground_mobjects = [];
    return this;
  }

  // TODO, remove this, and calls to this
  add_foreground_mobjects(...mobjects: Mobject[]): this {
    this.foreground_mobjects = list_update(this.foreground_mobjects, mobjects);
    this.add(...mobjects);
    return this;
  }

  add_foreground_mobject(mobject: Mobject): this {
    return this.add_foreground_mobjects(mobject);
  }

  remove_foreground_mobjects(...to_remove: Mobject[]): this {
    this.restructure_mobjects(to_remove, "foreground_mobjects");
    return this;
  }

  remove_foreground_mobject(mobject: Mobject): this {
    return this.remove_foreground_mobjects(mobject);
  }

  bring_to_front(...mobjects: Mobject[]): this {
    this.add(...mobjects);
    return this;
  }

  bring_to_back(...mobjects: Mobject[]): this {
    this.remove(...mobjects);
    this.mobjects = [...mobjects, ...this.mobjects];
    return this;
  }

  replace(old_mobject: Mobject, new_mobject: Mobject): void {
    if (old_mobject === null || old_mobject === undefined || new_mobject === null || new_mobject === undefined) {
      const error = new Error("Specified mobjects cannot be None");
      error.name = "ValueError";
      throw error;
    }

    function replace_in_list(mobj_list: Mobject[], old_m: Mobject, new_m: Mobject): boolean {
      // Avoid duplicate references to the same object in self.mobjects
      if (mobj_list.includes(new_m)) {
        if (old_m === new_m) {
          // In this case, one could say that the old Mobject was already found.
          // No replacement is needed, since old_m is new_m, so no action is required.
          // This might be unexpected, so raise a warning.
          console.warn(
            `Attempted to replace ${(old_m as Mobject).constructor.name} ` +
              "with itself in Scene.mobjects.",
          );
          return true;
        }
        mobj_list.splice(mobj_list.indexOf(new_m), 1);
      }

      // We use breadth-first search because some Mobjects get very deep and
      // we expect top-level elements to be the most common targets for replace.
      for (let i = 0; i < mobj_list.length; i += 1) {
        // Is this the old mobject?
        if (mobj_list[i] === old_m) {
          // If so, write the new object to the same spot and stop looking.
          mobj_list[i] = new_m;
          return true;
        }
      }
      // Now check all the children of all these mobs.
      for (const mob of mobj_list) {
        if (replace_in_list(mob.submobjects, old_m, new_m)) {
          // If we found it in a submobject, stop looking.
          return true;
        }
      }
      // If we did not find the mobject in the mobject list or any submobjects,
      // (or the list was empty), indicate we did not make the replacement.
      return false;
    }

    // Make use of short-circuiting conditionals to check mobjects and then
    // foreground_mobjects
    const replaced = replace_in_list(this.mobjects, old_mobject, new_mobject) ||
      replace_in_list(this.foreground_mobjects, old_mobject, new_mobject);

    if (!replaced) {
      const error = new Error(`Could not find ${old_mobject} in scene`);
      error.name = "ValueError";
      throw error;
    }
  }

  next_section(
    name = "unnamed",
    section_type: string = DefaultSectionType.NORMAL,
    skip_animations = false,
  ): void {
    this.renderer.file_writer.next_section(name, section_type, skip_animations);
  }

  should_update_mobjects(): boolean {
    const wait_animation = this.animations[0];
    if (!(wait_animation instanceof Wait)) {
      const error = new Error("assert isinstance(wait_animation, Wait)");
      error.name = "AssertionError";
      throw error;
    }
    if (wait_animation.is_static_wait === null) {
      // NOTE: Python `or`-chains the raw values; the TS port coerces to
      // boolean explicitly (a bare `this.updaters` array would always be
      // truthy in JS, so its length is tested instead, matching Python's
      // empty-list-is-falsy semantics).
      const should_update = Boolean(
        this.always_update_mobjects ||
          this.updaters.length > 0 ||
          wait_animation.stop_condition !== null ||
          this.get_mobject_family_members().some((mob) => mob.has_time_based_updater()),
      );
      wait_animation.is_static_wait = !should_update;
    }
    return !wait_animation.is_static_wait;
  }

  get_top_level_mobjects(): Mobject[] {
    // Return only those which are not in the family
    // of another mobject from the scene
    const families = this.mobjects.map((m) => m.get_family());

    function is_top_level(mobject: Mobject): boolean {
      const num_families = families.reduce(
        (count, family) => count + (family.includes(mobject) ? 1 : 0),
        0,
      );
      return num_families === 1;
    }

    return this.mobjects.filter(is_top_level);
  }

  get_mobject_family_members(): Mobject[] {
    if (config.renderer === RendererType.OPENGL) {
      return this.mobjects.flatMap((mobject) => mobject.get_family());
    }
    // NOTE: the Python original asserts the Cairo renderer here; any
    // non-OpenGL renderer takes this path. With no renderer attached yet
    // (plain `new Scene()`), there is no camera, so z-index sorting is off.
    return extract_mobject_family_members(
      this.mobjects,
      this.renderer?.camera?.use_z_index ?? false,
    );
  }

  get_moving_mobjects(...animations: Animation[]): Mobject[] {
    // NOTE (dependency pull-in): ported to support
    // ThreeDScene.get_moving_mobjects, which calls super().
    function collect_animation_mobjects(nested: Iterable<Animation>): Mobject[] {
      const result: Mobject[] = [];
      for (const anim of nested) {
        // Keep composition wrappers as conservative boundaries for the
        // static frame cache, in addition to their nested animations.
        result.push(...anim.mobject.get_family());
        if (anim instanceof AnimationGroup) {
          result.push(...collect_animation_mobjects(anim.animations));
        }
      }
      return result;
    }
    const animation_mobjects = new Set(collect_animation_mobjects(animations));
    const mobjects = this.get_mobject_family_members();
    const consider_moving = (mob: Mobject): boolean => {
      return (
        animation_mobjects.has(mob) ||
        mob.updaters.length > 0 ||
        this.foreground_mobjects.includes(mob) ||
        mob.submobjects.some((sub) => consider_moving(sub))
      );
    };
    const index = mobjects.findIndex((mob) => consider_moving(mob));
    return index === -1 ? [] : mobjects.slice(index);
  }

  get_moving_and_static_mobjects(
    animations: Iterable<Animation>,
  ): [Mobject[], Mobject[]] {
    const anims = [...animations];
    const all_mobjects = list_update(this.mobjects, this.foreground_mobjects);
    const all_mobject_families = extract_mobject_family_members(
      all_mobjects,
      this.renderer?.camera?.use_z_index ?? false,
      true,
    );
    const moving_mobjects = this.get_moving_mobjects(...anims);
    const all_moving_mobject_families = extract_mobject_family_members(
      moving_mobjects,
      this.renderer?.camera?.use_z_index ?? false,
    );
    const static_mobjects = list_difference_update(
      all_mobject_families,
      all_moving_mobject_families,
    );
    return [all_moving_mobject_families, static_mobjects];
  }

  restructure_mobjects(
    to_remove: Iterable<Mobject>,
    mobject_list_name = "mobjects",
    extract_families = true,
  ): this {
    // NOTE: with no renderer attached yet (plain `new Scene()`), there is
    // no camera, so z-index sorting is off.
    let removed: Iterable<Mobject> = to_remove;
    if (extract_families) {
      removed = extract_mobject_family_members(
        to_remove,
        this.renderer?.camera?.use_z_index ?? false,
      );
    }
    const list = (this as unknown as Record<string, Mobject[]>)[mobject_list_name] ?? [];
    (this as unknown as Record<string, Mobject[]>)[mobject_list_name] =
      this.get_restructured_mobject_list(list, removed);
    return this;
  }

  get_restructured_mobject_list(
    mobjects: Iterable<Mobject>,
    to_remove: Iterable<Mobject>,
  ): Mobject[] {
    const new_mobjects: Mobject[] = [];

    function add_safe_mobjects_from_list(
      list_to_examine: Iterable<Mobject>,
      set_to_remove: Set<Mobject>,
    ): void {
      for (const mob of list_to_examine) {
        if (set_to_remove.has(mob)) continue;
        // NOTE: non-Mobject items (TS CoordinateSystem hierarchy, e.g. via
        // VectorScene.add_plane) have no get_family; keep them as-is.
        const family: Set<Mobject> =
          typeof (mob as Mobject)?.get_family === "function"
            ? new Set((mob as Mobject).get_family())
            : new Set([mob]);
        const intersect = [...set_to_remove].filter((m) => family.has(m));
        if (intersect.length > 0) {
          add_safe_mobjects_from_list(mob.submobjects, new Set(intersect));
        } else {
          new_mobjects.push(mob);
        }
      }
    }

    add_safe_mobjects_from_list(mobjects, new Set(to_remove));
    return new_mobjects;
  }

  get_run_time(animations: Animation[]): number {
    const run_time = Math.max(...animations.map((animation) => animation.run_time));
    return Scene.validate_run_time(
      run_time,
      this.play as (...args: unknown[]) => unknown,
      "total run_time",
    );
  }

  compile_animations(
    ...args: Array<Animation | Mobject | _AnimationBuilder | ScenePlayOptions>
  ): Animation[] {
    // NOTE: Python's signature is `compile_animations(self, *args, **kwargs)`
    // where kwargs are play-level animation overrides. The TS port reads a
    // trailing plain object as those overrides (Animation, Mobject and
    // _AnimationBuilder instances carry class prototypes and never match),
    // the same convention as `play` below.
    let kwargs: Record<string, unknown> = {};
    let animArgs = args;
    const last = args[args.length - 1];
    if (
      last !== undefined &&
      isScenePlayOptions(last) &&
      !(last instanceof Animation) &&
      !(last instanceof Mobject) &&
      !(last instanceof _AnimationBuilder)
    ) {
      kwargs = last as Record<string, unknown>;
      animArgs = args.slice(0, -1);
    }
    const animations: Animation[] = [];
    // Allow passing a generator to self.play instead of comma separated arguments.
    const arg_anims = flatten_iterable_parameters<
      Animation | Mobject | _AnimationBuilder
    >(animArgs as Array<Animation | Mobject | _AnimationBuilder>);
    for (const arg of arg_anims) {
      // NOTE: mirrors the `except TypeError: if inspect.ismethod(arg)` branch:
      // a bound mobject method (carrying `__self__`, i.e. an already-bound
      // `mobject.method` reference rather than an invocation result) is
      // rejected up front with the "use Mobject.animate" error, since the TS
      // `prepare_animation` would otherwise invoke the function.
      // NOTE: a `_AnimationBuilder` is a callable Proxy whose unknown property
      // reads raise AttributeError (Python's `inspect.ismethod` simply returns
      // False for it), so the `__self__` probe must be guarded — reading it
      // unguarded would reject every `.animate` call.
      let boundSelf: unknown;
      try {
        boundSelf = (arg as unknown as { __self__?: unknown }).__self__;
      } catch {
        boundSelf = undefined;
      }
      if (typeof arg === "function" && boundSelf instanceof Mobject) {
        throw new TypeError(
          "Passing Mobject methods to Scene.play is no longer supported. Use Mobject.animate instead.",
        );
      }
      try {
        animations.push(
          prepare_animation(
            arg as Animation | _AnimationBuilder | Mobject | (() => Animation),
          ),
        );
      } catch (error) {
        if (error instanceof TypeError) {
          throw new TypeError(
            `Unexpected argument ${String(arg)} passed to Scene.play().`,
          );
        }
        throw error;
      }
    }
    for (const animation of animations) {
      for (const [key, value] of Object.entries(kwargs)) {
        (animation as unknown as Record<string, unknown>)[key] = value;
      }
    }
    return animations;
  }

  _get_animation_time_progression(
    animations: Animation[],
    duration: number,
  ): TimeProgression {
    let time_progression: TimeProgression;
    if (animations.length === 1 && animations[0] instanceof Wait) {
      const stop_condition = (animations[0] as Wait).stop_condition;
      if (stop_condition !== null && stop_condition !== undefined) {
        time_progression = this.get_time_progression(
          duration,
          `Waiting for ${stop_condition.name}`,
          -1,
          true,
        );
      } else {
        time_progression = this.get_time_progression(
          duration,
          `Waiting ${this.renderer?.num_plays ?? 0}`,
        );
      }
    } else {
      time_progression = this.get_time_progression(
        duration,
        `Animation ${this.renderer?.num_plays ?? 0}: ${String(animations[0])}${animations.length > 1 ? ", etc." : ""}`,
      );
    }
    return time_progression;
  }

  play(...args: Array<Animation | Mobject | ScenePlayOptions>): this | Promise<this> {
    // NOTE: Python's signature is `play(*args, subcaption, subcaption_duration,
    // subcaption_offset, **kwargs)` with kwargs forwarded to the renderer.
    // The TS port reads a trailing plain object as those play options
    // (Animation/Mobject instances never match); remaining kwargs are
    // accepted and ignored since the cairo renderer is not ported yet.
    let options: ScenePlayOptions = {};
    let playArgs = args;
    const last = args[args.length - 1];
    if (isScenePlayOptions(last)) {
      options = last;
      playArgs = args.slice(0, -1);
    }
    const {
      subcaption = null,
      subcaption_duration = null,
      subcaption_offset = 0,
      ..._rendererKwargs
    } = options;
    void _rendererKwargs;
    // If we are in interactive embedded mode, make sure this is running on
    // the main thread (required for OpenGL). The TS runtime is single
    // threaded, so only the interactive-mode + OpenGL part is mirrored.
    if (this.interactive_mode && (config.renderer as unknown) === RendererType.OPENGL) {
      this.queue.push(
        new SceneInteractRerun("play", {
          subcaption,
          subcaption_duration,
          subcaption_offset,
          ..._rendererKwargs,
        }),
      );
      return this;
    }
    const start_time = this.time;
    let play_result: Awaitable<unknown> = undefined;
    if (typeof this.renderer?.play === "function") {
      // Mirror Python `self.renderer.play(self, *args, **kwargs)`: the
      // subcaption keys are consumed above; any remaining renderer kwargs ride
      // as a trailing plain-object argument (TS has no `**kwargs` channel).
      const rendererArgs: unknown[] = Object.keys(_rendererKwargs).length > 0
        ? [...playArgs, _rendererKwargs]
        : [...playArgs];
      play_result = this.renderer.play(this, ...rendererArgs);
    } else {
      // NOTE: stand-in for `self.renderer.play(self, *args, **kwargs)` — with
      // no cairo renderer attached (plain `new Scene()` or a test stub that
      // only exposes `render`), animations run inline here.
      const prepared = flatten_iterable_parameters<Animation | Mobject>(
        playArgs as Array<Animation | Mobject | Iterable<Animation | Mobject>>,
      ).map((animation) => prepare_animation(animation));
      this.animations.push(...prepared);
      for (const animation of prepared) {
        animation._setup_scene(this);
        animation.begin();
        animation.finish();
        animation.clean_up_from_scene(this);
      }
    }
    // The trailing bookkeeping (subcaptions) runs once the animation finishes.
    // In real-time mode `play_internal` returns a promise and the caller
    // (`await self.play(...)` from rapydscript, or the browser runner) awaits
    // it; otherwise the whole body already ran synchronously.
    const finish = (): this => {
      const run_time = this.time - start_time;
      if (subcaption) {
        const duration = subcaption_duration ?? run_time;
        // The start of the subcaption needs to be offset by the
        // run_time of the animation because it is added after
        // the animation has already been played (and Scene.time
        // has already been updated).
        this.add_subcaption(subcaption, duration, -run_time + (subcaption_offset ?? 0));
      }
      return this;
    };
    if (is_thenable(play_result)) {
      return Promise.resolve(play_result).then(finish);
    }
    return finish();
  }

  compile_animation_data(
    ...args: Array<Animation | Mobject | _AnimationBuilder | ScenePlayOptions>
  ): this | null {
    // NOTE TODO : returns statement of this method are wrong. It should return nothing, as it makes a little sense to get any information from this method.
    // The return are kept to keep webgl renderer from breaking.
    if (args.length === 0) {
      const error = new Error("Called Scene.play with no animations");
      error.name = "ValueError";
      throw error;
    }

    this.animations = this.compile_animations(...args);
    this.add_mobjects_from_animations(this.animations);

    this.last_t = 0;
    this.stop_condition = null;
    this.moving_mobjects = [];
    this.static_mobjects = [];

    this.duration = this.get_run_time(this.animations);
    if (this.animations.length === 1 && this.animations[0] instanceof Wait) {
      if (this.should_update_mobjects()) {
        this.update_mobjects(0); // Any problems with this?
        this.stop_condition = (this.animations[0] as Wait).stop_condition;
      } else {
        // Static image logic when the wait is static is done by the renderer, not here.
        (this.animations[0] as Wait).is_static_wait = true;
        return null;
      }
    }

    return this;
  }

  begin_animations(): void {
    for (const animation of this.animations) {
      animation._setup_scene(this);
      animation.begin();
    }

    if (config.renderer === RendererType.CAIRO) {
      // Paint all non-moving objects onto the screen, so they don't
      // have to be rendered every frame
      [this.moving_mobjects, this.static_mobjects] =
        this.get_moving_and_static_mobjects(this.animations);
    }
  }

  play_internal(skip_rendering = false): void | Promise<void> {
    if (is_async_rendering()) {
      return this.play_internal_async(skip_rendering, get_render_scheduler()!);
    }
    this.duration = this.get_run_time(this.animations);
    this.time_progression = this._get_animation_time_progression(
      this.animations,
      this.duration,
    );
    for (const t of this.time_progression) {
      this.update_to_time(t);
      if (!skip_rendering && !this.skip_animation_preview) {
        // NOTE: with no renderer attached yet (plain `new Scene()`), there
        // is nothing to render into, so the call is skipped.
        this.renderer?.render(this, t, this.moving_mobjects);
      }
      if (this.stop_condition !== null && this.stop_condition()) {
        this.time_progression.close();
        break;
      }
    }

    this.finish_play_internal();
  }

  /**
   * Real-time variant of {@link play_internal}: identical work, but control is
   * handed back to the event loop after every drawn frame and only resumed
   * once the scheduler says the next logical frame is due.  The synchronous
   * path above is left byte-for-byte intact so Node/test runs are unaffected.
   */
  private async play_internal_async(
    skip_rendering: boolean,
    scheduler: RenderScheduler,
  ): Promise<void> {
    this.duration = this.get_run_time(this.animations);
    this.time_progression = this._get_animation_time_progression(
      this.animations,
      this.duration,
    );
    try {
      for (const t of this.time_progression) {
        this.update_to_time(t);
        if (!skip_rendering && !this.skip_animation_preview) {
          this.renderer?.render(this, t, this.moving_mobjects);
        }
        if (this.stop_condition !== null && this.stop_condition()) {
          this.time_progression.close();
          break;
        }
        if (scheduler.cancelled) break;
        await scheduler.next_frame();
        if (scheduler.cancelled) break;
      }
    } finally {
      this.finish_play_internal();
    }
  }

  private finish_play_internal(): void {
    for (const animation of this.animations) {
      animation.finish();
      animation.clean_up_from_scene(this);
    }
    if (!this.renderer?.skip_animations) {
      this.update_mobjects(0);
    }
    // TODO: The OpenGLRenderer does not have the property static.image.
    if (this.renderer) {
      this.renderer.static_image = null;
    }
    // Closing the progress bar at the end of the play.
    this.time_progression?.close();
  }

  interact(
    shell: {
      pt_app: { app: { exit: (error?: unknown) => void } } | null;
    },
    keyboard_thread: { join: () => void },
  ): void {
    // NOTE: Python asserts isinstance(self.renderer, OpenGLRenderer) and
    // watches the input file with watchdog's Observer to queue reruns; the
    // OpenGL renderer and watchdog have no TS equivalent, so the file
    // observer is skipped and interaction runs purely off this.queue (the
    // action handling below otherwise mirrors Python verbatim).
    const eofError = (): Error => {
      const error = new Error("EOF");
      error.name = "EOFError";
      return error;
    };

    this.quit_interaction = false;
    let keyboard_thread_needs_join = shell?.pt_app !== null && shell?.pt_app !== undefined;
    if (this.queue.length !== 0) {
      const error = new Error("assert self.queue.qsize() == 0");
      error.name = "AssertionError";
      throw error;
    }

    let last_time = Date.now() / 1000;
    while (!(
      (
        this.renderer?.window !== null &&
        this.renderer?.window !== undefined &&
        this.renderer.window.is_closing
      ) ||
      this.quit_interaction
    )) {
      if (this.queue.length > 0) {
        const action = this.queue.shift() as SceneInteractAction;
        if (action instanceof SceneInteractRerun) {
          // Intentionally skip calling join() on the file thread to save time.
          if (action.sender !== "keyboard") {
            if (shell?.pt_app) {
              shell.pt_app.app.exit(eofError());
            }
            throw new RerunSceneException();
          }
          keyboard_thread.join();

          if ("from_animation_number" in action.kwargs) {
            (config as unknown as Record<string, unknown>)["from_animation_number"] =
              action.kwargs["from_animation_number"];
          }
          // # TODO: This option only makes sense if interactive_embed() is run at the
          // # end of a scene by default.
          // if "upto_animation_number" in action.kwargs:
          //     config["upto_animation_number"] = action.kwargs[
          //         "upto_animation_number"
          //     ]

          keyboard_thread.join();
          throw new RerunSceneException();
        } else if (action instanceof SceneInteractContinue) {
          // Intentionally skip calling join() on the file thread to save time.
          if (action.sender !== "keyboard" && shell?.pt_app) {
            shell.pt_app.app.exit(eofError());
          }
          keyboard_thread.join();
          // Remove exit_keyboard from the queue if necessary.
          while (this.queue.length > 0) {
            this.queue.shift();
          }
          keyboard_thread_needs_join = false;
          break;
        } else {
          // NOTE: Python calls action.method(*action.args, **action.kwargs);
          // the TS MethodWithArgs carries kwargs but its method channel is
          // positional-only, so kwargs ride no separate channel here.
          const methodAction = action as MethodWithArgs;
          methodAction.method(...methodAction.args);
        }
      } else {
        if (this.renderer) {
          this.renderer.animation_start_time = 0;
        }
        const now = Date.now() / 1000;
        const dt = now - last_time;
        last_time = now;
        // NOTE: skipped when no renderer is attached (see play_internal).
        this.renderer?.render(this, dt, this.moving_mobjects);
        this.update_mobjects(dt);
        this.update_meshes(dt);
        this.update_self(dt);
      }
    }

    // Join the keyboard thread if necessary.
    if (shell !== null && shell !== undefined && keyboard_thread_needs_join) {
      shell.pt_app?.app.exit(eofError());
      keyboard_thread.join();
      // Remove exit_keyboard from the queue if necessary.
      while (this.queue.length > 0) {
        this.queue.shift();
      }
    }

    // NOTE: watchdog observer stop/join skipped (see above).

    if (this.dearpygui_imported && config.get("enable_gui")) {
      // NOTE: DearPyGUI has no TypeScript equivalent (same limitation as
      // _configure_pygui); the stop_dearpygui() call is skipped.
    }

    if (this.renderer?.window !== null && this.renderer?.window !== undefined && this.renderer.window.is_closing) {
      this.renderer.window.destroy();
    }
  }

  embed(): void {
    // NOTE: Python asserts isinstance(self.renderer, OpenGLRenderer); the
    // OpenGL renderer is not ported yet, so renderer access below is guarded.
    if (!config.preview) {
      console.warn("Called embed() while no preview window is available.");
      return;
    }
    if (config.write_to_movie) {
      console.warn("embed() is skipped while writing to a file.");
      return;
    }

    if (this.renderer) {
      this.renderer.animation_start_time = 0;
      this.renderer.render(this, -1, this.moving_mobjects);
    }

    // NOTE: the IPython InteractiveShellEmbed session (post_run_cell render
    // hook, caller-locals namespace, and the play/wait/add/remove/interact
    // shortcuts) has no TS equivalent and is skipped; see
    // .
    console.warn("embed requires an IPython terminal session, which is not available in the TypeScript runtime.");

    // End scene when exiting an embed.
    throw new Error("Exiting scene.");
  }

  add_subcaption(content: string, duration = 1, offset = 0): void {
    const store = this.renderer?.file_writer?.subcaptions as Subtitle[] | undefined;
    const start = this.time + offset;
    const end = start + duration;
    // NOTE: with no renderer/file_writer (plain `new Scene()`), entries land
    // in the local `subcaptions` list instead of raising.
    const target = store ?? this.subcaptions;
    target.push(new Subtitle(target.length, content, start, end));
  }

  add_sound(
    sound_file: string,
    time_offset = 0,
    gain: number | null = null,
    kwargs: Record<string, unknown> = {},
  ): void {
    if (this.renderer?.skip_animations) return;
    const time = this.time + time_offset;
    // NOTE: the file-writer/renderer backend is not ported yet, so this
    // forwards to the renderer's writer the same way Python does
    // (`add_sound(sound_file, time, gain, **kwargs)`); the kwargs bag rides
    // as a trailing object since TS has no **kwargs channel.
    this.renderer.file_writer.add_sound(sound_file, time, gain, kwargs);
  }

  wait(
    duration: number = DEFAULT_WAIT_TIME,
    stop_condition: (() => boolean) | null = null,
    frozen_frame: boolean | null = null,
  ): this | Promise<this> {
    duration = Scene.validate_run_time(duration, this.wait as (...args: unknown[]) => unknown, "duration");
    return this.play(
      new Wait({ run_time: duration, stop_condition, frozen_frame }),
    );
  }

  update_mobjects(dt: number): void {
    for (const mobj of this.mobjects) mobj.update(dt);
  }

  update_meshes(dt: number): void {
    for (const obj of this.meshes) {
      for (const mesh of obj.get_family()) mesh.update(dt);
    }
  }

  update_to_time(t: number): void {
    const dt = t - this.last_t;
    this.last_t = t;
    for (const animation of this.animations) {
      animation.update_mobjects(dt);
      const alpha = t / animation.run_time;
      animation.interpolate(alpha);
    }
    this.update_mobjects(dt);
    this.update_meshes(dt);
    this.update_self(dt);
  }

  pause(duration: number = DEFAULT_WAIT_TIME): this | Promise<this> {
    duration = Scene.validate_run_time(duration, this.pause as (...args: unknown[]) => unknown, "duration");
    return this.wait(duration, null, true);
  }

  wait_until(stop_condition: () => boolean, max_time = 60): this | Promise<this> {
    max_time = Scene.validate_run_time(max_time, this.wait_until as (...args: unknown[]) => unknown, "max_time");
    return this.wait(max_time, stop_condition);
  }

  is_current_animation_frozen_frame(): boolean {
    if (this.animations === null || this.animations === undefined) {
      throw new Error("assert self.animations is not None");
    }
    const first = this.animations[0];
    if (first === undefined) {
      const error = new RangeError("list index out of range");
      error.name = "IndexError";
      throw error;
    }
    return (
      first instanceof Wait &&
      this.animations.length === 1 &&
      (first as Wait).is_static_wait === true
    );
  }

  updaters: Array<(dt: number) => void> = [];

  render(preview = false): boolean | Promise<boolean> {
    this.setup();
    let construct_result: Awaitable<unknown>;
    try {
      construct_result = this.construct();
    } catch (error) {
      return this.handle_render_error(error, preview);
    }
    // In real-time mode `construct` is async (rapydscript emits
    // `await self.play(...)`), so the render only finishes once it settles.
    if (is_thenable(construct_result)) {
      return Promise.resolve(construct_result).then(
        () => this.finalize_render(preview),
        (error: unknown) => this.handle_render_error(error, preview),
      );
    }
    return this.finalize_render(preview);
  }

  private handle_render_error(error: unknown, preview: boolean): boolean {
    if (error instanceof EndSceneEarlyException) {
      // pass
    } else if (error instanceof RerunSceneException) {
      this.remove(...this.mobjects);
      // TODO: The CairoRenderer does not have the method clear_screen()
      this.renderer.clear_screen();
      this.renderer.num_plays = 0;
      // The rerun replaces the file writer; tear down its encode jobs so
      // no worker is still writing a partial file the new writer may
      // reuse. Encoder failures propagate: a rerun must not silently
      // continue past corrupt output.
      this.renderer.file_writer.abort_encode_jobs({
        reraise_encoder_failures: true,
      });
      return true;
    } else {
      // A mid-play exception leaves an unsealed encode job whose
      // non-daemon worker would hang the process at exit.
      this.renderer.file_writer.abort_encode_jobs();
      throw error;
    }
    return this.finalize_render(preview);
  }

  private finalize_render(preview: boolean): boolean {
    this.tear_down();
    // We have to reset these settings in case of multiple renders.
    this.renderer.scene_finished(this);

    // Show info only if animations are rendered or to get image
    if (
      this.renderer.num_plays ||
      config.get("format") === "png" ||
      config.save_last_frame
    ) {
      // NOTE: mirrors Python `logger.info(...)`; see open_media_file above.
      console.info(
        `Rendered ${String(this)}\nPlayed ${this.renderer.num_plays} animations`,
      );
    }

    // If preview open up the render after rendering.
    if (preview) {
      config.preview = true;
    }

    if (config.preview || config.show_in_file_browser) {
      open_media_file(this.renderer.file_writer);
    }

    return false;
  }

  setup(): void {}

  tear_down(): void {}

  construct(): void {}

  toString(): string {
    return this.constructor.name;
  }

  update_self(dt: number): void {
    for (const func of this.updaters) func(dt);
  }

  add_updater(func: (dt: number) => void): void {
    this.updaters.push(func);
  }

  remove_updater(func: (dt: number) => void): void {
    this.updaters = this.updaters.filter((candidate) => candidate !== func);
  }

  get_time_progression(
    run_time: number,
    description: string,
    n_iterations: number | null = null,
    override_skip_animations = false,
  ): TimeProgression {
    let times: number[];
    if ((this.renderer?.skip_animations ?? false) && !override_skip_animations) {
      times = [run_time];
    } else {
      const step = 1 / (config.frame_rate as number);
      times = [];
      // np.arange(0, run_time, step): values are i * step, not accumulated.
      for (let i = 0; i * step < run_time; i += 1) times.push(i * step);
    }
    return new TimeProgression(
      times,
      description,
      n_iterations,
      (config.get("progress_bar") as string) === "leave",
      platform() === "win32" ? true : null,
      (config.get("progress_bar") as string) === "none",
    );
  }

  static validate_run_time(
    run_time: number,
    method: (...args: unknown[]) => unknown,
    parameter_name = "run_time",
  ): number {
    const method_name = `${this.name}.${method.name}()`;
    if (run_time <= 0) {
      const error = new Error(
        `${method_name} has a ${parameter_name} of ${formatG(run_time)} <= 0 seconds which Manim cannot render. ` +
          `The ${parameter_name} must be a positive number.`,
      );
      error.name = "ValueError";
      throw error;
    }
    const fps = config.frame_rate as number;
    const seconds_per_frame = 1 / fps;
    if (run_time < seconds_per_frame) {
      console.warn(
        `The original ${parameter_name} of ${method_name}, ${formatG(run_time)} seconds, is too short for the current frame ` +
          `rate of ${formatG(fps)} FPS. Rendering with the shortest possible ${parameter_name} of ${formatG(seconds_per_frame)} seconds instead.`,
      );
      run_time = seconds_per_frame;
    }
    return run_time;
  }

  on_mouse_motion(point: number[] | readonly number[] | NDArray, d_point: number[] | readonly number[] | NDArray): void {
    const target = vec3(point);
    const delta = vec3(d_point);
    this.mouse_point.move_to(target);
    const pressedKeys = this.renderer?.pressed_keys as Set<number> | number[] | undefined;
    const hasShift = pressedKeys instanceof Set
      ? pressedKeys.has(SHIFT_VALUE)
      : Array.isArray(pressedKeys) && pressedKeys.includes(SHIFT_VALUE);
    if (hasShift) {
      const shift = [-delta[0], -delta[1], -delta[2]];
      shift[0] = (shift[0] ?? 0) * (this.camera.get_width() / 2);
      shift[1] = (shift[1] ?? 0) * (this.camera.get_height() / 2);
      const transform = this.camera.inverse_rotation_matrix as NDArray;
      const rotated = matmul(transpose(transform), array(shift)) as NDArray;
      this.camera.shift(vec3(rotated));
    }
  }

  on_mouse_scroll(point: number[] | readonly number[] | NDArray, offset: number[] | readonly number[] | NDArray): void {
    // NOTE: Python asserts isinstance(self.camera, OpenGLCamera); the
    // OpenGL renderer/camera is not ported yet and `camera` is `any`, so
    // the check is omitted and behavior binds to the real camera once it
    // lands.
    if (!config.use_projection_stroke_shaders) {
      const off = vec3(offset);
      const factor = 1 + Math.atan(-2.1 * (off[1] ?? 0));
      this.camera.scale(factor, this.camera_target);
    }
    this.mouse_scroll_orbit_controls(point, offset);
  }

  on_mouse_drag(
    point: number[] | readonly number[] | NDArray,
    d_point: number[] | readonly number[] | NDArray,
    buttons: number,
    modifiers: number,
  ): void {
    // NOTE: same OpenGLCamera note as on_mouse_scroll.
    const target = vec3(point);
    const delta = vec3(d_point);
    this.mouse_drag_point.move_to(target);
    if (buttons === 1) {
      this.camera.increment_theta(-(delta[0] ?? 0));
      this.camera.increment_phi(delta[1] ?? 0);
    } else if (buttons === 4) {
      const camera_x_axis = mat44Column(this.camera.model_matrix as NDArray, 0);
      const horizontal_shift_vector = camera_x_axis.map((value) => -(delta[0] ?? 0) * value);
      const outCross = cross([...OUT], camera_x_axis);
      const vertical_shift_vector = outCross.map((value) => -(delta[1] ?? 0) * value);
      const total_shift_vector = horizontal_shift_vector.map(
        (value, index) => value + (vertical_shift_vector[index] ?? 0),
      );
      this.camera.shift(total_shift_vector.map((value) => 1.1 * value));
    }
    this.mouse_drag_orbit_controls(point, d_point, buttons, modifiers);
  }

  on_key_press(symbol: number, modifiers: number): void {
    void modifiers;
    let char: string;
    try {
      char = String.fromCodePoint(symbol);
    } catch {
      console.warn("The value of the pressed key is too large.");
      return;
    }
    if (char === "r") {
      this.camera.to_default_state();
      this.camera_target = [0, 0, 0];
    } else if (char === "q") {
      this.quit_interaction = true;
    } else if (this.key_to_function_map.has(char)) {
      this.key_to_function_map.get(char)?.();
    }
  }

  on_key_release(symbol: number, modifiers: number): void {
    void symbol;
    void modifiers;
  }

  mouse_scroll_orbit_controls(point: number[] | readonly number[] | NDArray, offset: number[] | readonly number[] | NDArray): void {
    void point;
    const off = vec3(offset);
    const target = vec3(this.camera_target);
    const position = vec3(this.camera.get_position());
    const sign = Math.sign(off[1] ?? 0);
    const shiftVector = target.map((value, index) => 0.01 * (value - (position[index] ?? 0)) * sign);
    this.camera.model_matrix = matmul(
      translation_matrix_4x4(shiftVector[0] ?? 0, shiftVector[1] ?? 0, shiftVector[2] ?? 0),
      this.camera.model_matrix as NDArray,
    );
  }

  mouse_drag_orbit_controls(
    point: number[] | readonly number[] | NDArray,
    d_point: number[] | readonly number[] | NDArray,
    buttons: number,
    modifiers: number,
  ): void {
    void point;
    void modifiers;
    const delta = vec3(d_point);
    // Left click drag.
    if (buttons === 1) {
      // Translate to target the origin and rotate around the z axis.
      const negatedTarget = vec3(this.camera_target).map((value) => -value);
      this.camera.model_matrix = matmul(
        z_rotation_matrix_4x4(-(delta[0] ?? 0)),
        matmul(
          translation_matrix_4x4(negatedTarget[0] ?? 0, negatedTarget[1] ?? 0, negatedTarget[2] ?? 0),
          this.camera.model_matrix as NDArray,
        ) as NDArray,
      );
      // Rotation off of the z axis.
      const camera_position = vec3(this.camera.get_position());
      const camera_y_axis = mat44Column(this.camera.model_matrix as NDArray, 1);
      const axis_of_rotation = normalize(cross(camera_y_axis, camera_position));
      let rotation = rotation_matrix(delta[1] ?? 0, axis_of_rotation, true);
      const maximum_polar_angle = this.camera.maximum_polar_angle as number;
      const minimum_polar_angle = this.camera.minimum_polar_angle as number;
      const potential_camera_model_matrix = matmul(rotation, this.camera.model_matrix as NDArray) as NDArray;
      const potential_camera_location = mat44Column(potential_camera_model_matrix, 3);
      const potential_camera_y_axis = mat44Column(potential_camera_model_matrix, 1);
      const axisSign = (potential_camera_y_axis[2] ?? 0) !== 0
        ? Math.sign(potential_camera_y_axis[2] ?? 0)
        : 1;
      const potential_polar_angle = axisSign * Math.acos(
        (potential_camera_location[2] ?? 0) / Math.hypot(...potential_camera_location),
      );
      if (minimum_polar_angle <= potential_polar_angle && potential_polar_angle <= maximum_polar_angle) {
        this.camera.model_matrix = potential_camera_model_matrix;
      } else {
        const currentSign = (camera_y_axis[2] ?? 0) !== 0 ? Math.sign(camera_y_axis[2] ?? 0) : 1;
        const current_polar_angle = currentSign * Math.acos(
          (camera_position[2] ?? 0) / Math.hypot(...camera_position),
        );
        let polar_angle_delta: number;
        if (potential_polar_angle > maximum_polar_angle) {
          polar_angle_delta = maximum_polar_angle - current_polar_angle;
        } else {
          polar_angle_delta = minimum_polar_angle - current_polar_angle;
        }
        rotation = rotation_matrix(polar_angle_delta, axis_of_rotation, true);
        this.camera.model_matrix = matmul(rotation, this.camera.model_matrix as NDArray);
      }
      // Translate to target the original target.
      const target = vec3(this.camera_target);
      this.camera.model_matrix = matmul(
        translation_matrix_4x4(target[0] ?? 0, target[1] ?? 0, target[2] ?? 0),
        this.camera.model_matrix as NDArray,
      );
    } else if (buttons === 4) {
      // Right click drag.
      const camera_x_axis = mat44Column(this.camera.model_matrix as NDArray, 0);
      const horizontal_shift_vector = camera_x_axis.map((value) => -(delta[0] ?? 0) * value);
      const outCross = cross([...OUT], camera_x_axis);
      const vertical_shift_vector = outCross.map((value) => -(delta[1] ?? 0) * value);
      const total_shift_vector = horizontal_shift_vector.map(
        (value, index) => value + (vertical_shift_vector[index] ?? 0),
      );
      this.camera.model_matrix = matmul(
        translation_matrix_4x4(total_shift_vector[0] ?? 0, total_shift_vector[1] ?? 0, total_shift_vector[2] ?? 0),
        this.camera.model_matrix as NDArray,
      );
      this.camera_target = vec3(this.camera_target).map(
        (value, index) => value + (total_shift_vector[index] ?? 0),
      );
    }
  }

  check_interactive_embed_is_valid(): boolean {
    if (config.get("force_window")) return true;
    if (this.skip_animation_preview) {
      console.warn("Disabling interactive embed as 'skip_animation_preview' is enabled");
      return false;
    } else if (config.get("write_to_movie")) {
      console.warn("Disabling interactive embed as 'write_to_movie' is enabled");
      return false;
    } else if (config.get("format")) {
      console.warn(
        "Disabling interactive embed as '--format' is set as " + config.get("format"),
      );
      return false;
    } else if (!this.renderer?.window) {
      console.warn("Disabling interactive embed as no window was created");
      return false;
    } else if (config.get("dry_run")) {
      console.warn("Disabling interactive embed as dry_run is enabled");
      return false;
    }
    return true;
  }

  /**
   * Mirrors the nested `get_embedded_method(method_name)` factory inside
   * `interactive_embed`: returns a closure pushing
   * `MethodWithArgs(method, args, kwargs)` onto this scene's queue.
   * Extracted as a method so the closure body is directly testable
   * (it would otherwise be hidden inside `interactive_embed`, which
   * requires an IPython/OpenGL session).
   */
  get_embedded_method(methodName: string): (...args: unknown[]) => void {
    const method = (this as unknown as Record<string, (...args: unknown[]) => void>)[methodName];
    const embedded_method = (...args: unknown[]): void => {
      // NOTE: mirrors Python's MethodWithArgs(method, args, kwargs); the
      // TS interactive shell has no **kwargs channel, so kwargs is {}.
      this.queue.push(new MethodWithArgs(method, [...args], {}));
    };
    return embedded_method;
  }

  /**
   * Mirrors the nested `embedded_rerun(*args, **kwargs)` inside
   * `interactive_embed.ipython`: queues `SceneInteractRerun("keyboard")`
   * and exits the embedded shell.
   */
  create_embedded_rerun(shell: { exiter: () => void }): (...args: unknown[]) => void {
    const embedded_rerun = (..._args: unknown[]): void => {
      this.queue.push(new SceneInteractRerun("keyboard"));
      shell.exiter();
    };
    return embedded_rerun;
  }

  /**
   * Mirrors the nested `rerun_callback(sender, data)` inside
   * `_configure_pygui`: queues `SceneInteractRerun("gui")`.
   */
  create_rerun_callback(): (sender: unknown, data: unknown) => void {
    const rerun_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractRerun("gui"));
    };
    return rerun_callback;
  }

  /**
   * Mirrors the nested `continue_callback(sender, data)` inside
   * `_configure_pygui`: queues `SceneInteractContinue("gui")`.
   */
  create_continue_callback(): (sender: unknown, data: unknown) => void {
    const continue_callback = (_sender: unknown, _data: unknown): void => {
      this.queue.push(new SceneInteractContinue("gui"));
    };
    return continue_callback;
  }

  /**
   * Mirrors the nested `scene_selection_callback(sender, data)` inside
   * `_configure_pygui`: sets `config.scene_names` to the selected scene
   * and queues `SceneInteractRerun("gui")`. Python reads the selection
   * via `dpg.get_value(sender)`; the TS port accepts either a plain
   * string or a `{ get_value() }` handle since DearPyGUI is unavailable.
   */
  create_scene_selection_callback(): (sender: { get_value?: () => string } | string, data: unknown) => void {
    const scene_selection_callback = (sender: { get_value?: () => string } | string, _data: unknown): void => {
      const selected = typeof sender === "string"
        ? sender
        : sender.get_value?.() ?? "";
      config.scene_names = [selected];
      this.queue.push(new SceneInteractRerun("gui"));
    };
    return scene_selection_callback;
  }

  interactive_embed(): void {
    if (!this.check_interactive_embed_is_valid()) return;
    this.interactive_mode = true;

    const ipython = (
      shell: { exiter: () => void; call: (options: { local_ns: Record<string, unknown> }) => void },
      namespace: Record<string, unknown>,
    ): void => {
      const load_module_into_namespace = (
        module: Record<string, unknown>,
        target: Record<string, unknown>,
      ): void => {
        for (const name in module) target[name] = module[name];
      };

      // Mirrors `import manim` / `import manim.opengl` namespace injection.
      // There is no IPython kernel in the TypeScript runtime, so the
      // modules resolve to empty namespaces here.
      load_module_into_namespace({}, namespace);
      load_module_into_namespace({}, namespace);

      const embedded_rerun = this.create_embedded_rerun(shell);
      namespace["rerun"] = embedded_rerun;

      shell.call({ local_ns: namespace });
      this.queue.push(new SceneInteractContinue("keyboard"));
    };
    void ipython;

    const get_embedded_method = (methodName: string): ((...args: unknown[]) => void) => {
      return this.get_embedded_method(methodName);
    };
    void get_embedded_method;
    console.warn("interactive_embed requires an IPython terminal session, which is not available in the TypeScript runtime.");
  }

  _configure_pygui(update = true): void {
    // NOTE: DearPyGUI has no TypeScript equivalent; only the queue/config
    // behavior of the GUI callbacks is ported. The window construction itself
    // is out of scope for the interpreter target.
    const rerun_callback = this.create_rerun_callback();

    const continue_callback = this.create_continue_callback();

    const scene_selection_callback = this.create_scene_selection_callback();
    void rerun_callback;
    void continue_callback;
    void scene_selection_callback;
    void update;
    console.warn("_configure_pygui requires DearPyGUI, which is not available in the TypeScript runtime.");
  }

  on_mouse_press(
    _point: number[] | readonly number[] | NDArray,
    _button: string,
    _modifiers: number,
  ): void {
    for (const func of this.mouse_press_callbacks) func();
  }
}
