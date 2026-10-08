import { array, type NDArray } from "numpy-ts";
import { Animation, classDefaultsOf, type AnimationOptions, type RateFunction, linear, smooth } from "./animation.js";
import { Mobject, Group, __registerBuilderAnimationClass } from "../mobject/mobject.js";
import { VMobject, VGroup } from "../mobject/types/vectorized_mobject.js";
import { MethodWithArgs } from "../data_structures.js";
import { identityMatrix, matrixRows, pointFrom, pointRows, pointsFrom } from "../utils/arrays.js";
import { path_along_arc, path_along_circles, type PathFuncType } from "../utils/paths.js";
import { squish_rate_func } from "../utils/rate_functions.js";
import { DEGREES, DEFAULT_POINTWISE_FUNCTION_RUN_TIME, ORIGIN, OUT } from "../constants.js";
import { mobject_structure_epoch } from "../utils/render_epoch.js";
import type { ParsableManimColor } from "../utils/color/core.js";

export interface TransformOptions extends AnimationOptions {
  path_func?: PathFuncType | null;
  path_arc?: number;
  path_arc_axis?: number[] | readonly number[] | NDArray;
  path_arc_centers?: unknown;
  replace_mobject_with_target_in_scene?: boolean;
}

export class Transform extends Animation {
  target_mobject: Mobject;
  target_copy!: Mobject;
  path_arc_axis!: number[] | readonly number[] | NDArray;
  path_arc_centers!: unknown;
  private _path_arc!: number;
  private _path_func!: PathFuncType;
  replace_mobject_with_target_in_scene: boolean;
  /** Cached zipped families (rebuilt in `begin`); see get_all_families_zipped. */
  private _families_zipped: Mobject[][] | null = null;
  /** Structure epoch the cached zip was built from; see get_all_families_zipped. */
  private _families_zipped_epoch = -1;

  constructor(mobject: Mobject, targetMobject?: Mobject | TransformOptions, options: TransformOptions = {}) {
    const actualOptions = targetMobject instanceof Mobject || targetMobject === undefined ? options : (targetMobject as TransformOptions);
    const actualTarget = targetMobject instanceof Mobject ? targetMobject : mobject.copy();
    super(mobject, actualOptions);
    this.path_arc_axis = (actualOptions.path_arc_axis as number[] | readonly number[] | NDArray) ?? [...OUT];
    this.path_arc_centers = actualOptions.path_arc_centers ?? null;
    this.replace_mobject_with_target_in_scene = actualOptions.replace_mobject_with_target_in_scene ?? false;
    // NOTE: mirrors Python __init__ order: axis/centers first, then the
    // path_arc setter (which builds _path_func from axis), then path_func.
    this.path_arc = actualOptions.path_arc ?? 0;
    const pathFunc = actualOptions.path_func ?? null;
    if (pathFunc !== null && pathFunc !== undefined) {
      this.path_func = pathFunc;
    } else if (this.path_arc_centers !== null && this.path_arc_centers !== undefined) {
      this.path_func = path_along_circles(
        this._path_arc,
        this.path_arc_centers as never,
        this.path_arc_axis as never,
      );
    }
    this.target_mobject = actualTarget;
  }

  get path_arc(): number {
    return this._path_arc;
  }

  set path_arc(pathArc: number) {
    this._path_arc = pathArc;
    this._path_func = path_along_arc(pathArc, this.path_arc_axis as never);
  }

  get path_func(): PathFuncType {
    return this._path_func;
  }

  set path_func(pathFunc: PathFuncType | null | undefined) {
    if (pathFunc !== null && pathFunc !== undefined) {
      this._path_func = pathFunc;
    }
  }

  get pathArc(): number {
    return this.path_arc;
  }

  set pathArc(value: number) {
    this.path_arc = value;
  }

  get pathFunc(): PathFuncType {
    return this.path_func;
  }

  set pathFunc(value: PathFuncType | null | undefined) {
    this.path_func = value;
  }

  create_target(): Mobject {
    // Has no meaningful effect here, but may be useful
    // in subclasses
    return this.target_mobject;
  }

  override begin(): void {
    // Use a copy of target_mobject for the align_data
    // call so that the actual target_mobject stays
    // preserved.
    this.target_mobject = this.create_target();
    this.target_copy = this.target_mobject.copy();
    // Note, this potentially changes the structure
    // of both mobject and target_mobject
    this.mobject.align_data(this.target_copy);
    this._families_zipped = null;
    super.begin();
  }

  override interpolate_mobject(alpha: number): void {
    const families = this.get_all_families_zipped();
    // Fast path: with zero lag every face shares one sub-alpha
    // (`get_sub_alpha` is index-independent when `lag_ratio === 0` and has no
    // overrides in the codebase), so one `rate_func` evaluation replaces one
    // per face per frame.
    if (this.lag_ratio === 0 && families.length > 0) {
      const subAlpha = this.get_sub_alpha(alpha, 0, families.length);
      for (let index = 0; index < families.length; index += 1) {
        const mobs = families[index] ?? [];
        this.interpolate_submobject(
          mobs[0] as Mobject,
          mobs[1] as Mobject,
          mobs[2] as Mobject,
          subAlpha,
        );
      }
      return;
    }
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

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, targetCopyOrAlpha: Mobject | number, alpha?: number): void {
    // NOTE: Python's signature is
    // `(submobject, starting_submobject, target_copy, alpha)`; the extra
    // `target_copy` is threaded through `get_all_families_zipped`. The
    // union form also accepts the base-class 3-argument call shape, in
    // which case the aligned copy is used as the target.
    if (typeof targetCopyOrAlpha === "number") {
      submobject.interpolate(startingSubmobject, this.target_copy ?? this.target_mobject, targetCopyOrAlpha, this.path_func);
    } else {
      submobject.interpolate(startingSubmobject, targetCopyOrAlpha, alpha as number, this.path_func);
    }
  }

  override get_all_mobjects(): Mobject[] {
    const mobs = [this.mobject, this.starting_mobject, this.target_mobject];
    if (this.target_copy !== undefined && this.target_copy !== null) mobs.push(this.target_copy);
    return mobs;
  }

  override get_all_families_zipped(): Mobject[][] {
    // NOTE: mirrors `zip(*(mob.family_members_with_points() for mob in mobs))`
    // stopping at the SHORTEST family (like animation.ts), rather than
    // padding shorter families with the current mobject.
    // The family structure is stable across frames for most animations, so
    // the zip is computed once per `begin` instead of once per frame (which
    // walked every face of every family three times per frame).  It is NOT
    // stable in general: an updater on `starting_mobject`/`target_copy` may
    // rebuild a subtree (e.g. `DecimalNumber.set_value` recreating its glyphs),
    // and a cached zip would keep interpolating against detached leaves whose
    // points may have been blanked.  Validate against the global structure
    // epoch so the cache survives point-only frames but is dropped on any
    // child-list replacement.
    const structureEpoch = mobject_structure_epoch();
    if (this._families_zipped !== null && this._families_zipped_epoch === structureEpoch) {
      return this._families_zipped;
    }
    const mobs = [
      this.mobject,
      this.starting_mobject,
      this.target_copy ?? this.target_mobject,
    ];
    const families = mobs.map((mob) => (mob as Mobject).family_members_with_points());
    if (families.length === 0) return [];
    const length = Math.min(...families.map((family) => family.length));
    this._families_zipped = Array.from({ length }, (_, index) => families.map((family) => family[index] as Mobject));
    this._families_zipped_epoch = structureEpoch;
    return this._families_zipped;
  }

  clean_up_from_scene(scene: any): void {
    super.clean_up_from_scene(scene);
    if ((this as unknown as { replace_mobject_with_target_in_scene?: boolean }).replace_mobject_with_target_in_scene) scene?.replace(this.mobject, this.target_mobject);
  }
}

export type MobjectBoundMethod = ((...args: any[]) => unknown) & {
  __self__?: unknown;
  __func__?: (...args: any[]) => unknown;
};

/**
 * Builds a bound-method facade for `mobject[name]`, carrying `__self__` /
 * `__func__` the way CPython bound methods do, so it passes
 * `ApplyMethod.check_validity_of_input` and works with `create_target`.
 * (Same pattern as `ApplyPointwiseFunction` in this file.)
 */
function boundMethodOf(mobject: Mobject, name: string): MobjectBoundMethod {
  const unbound = (mobject as unknown as Record<string, unknown>)[name];
  if (typeof unbound !== "function") {
    throw new Error(`${mobject.constructor.name} has no method '${name}'`);
  }
  const bound = function (...args: unknown[]): unknown {
    return (unbound as (...callArgs: unknown[]) => unknown).apply(mobject, args);
  } as MobjectBoundMethod;
  bound.__self__ = mobject;
  bound.__func__ = unbound as (...args: never[]) => unknown;
  return bound;
}

export class ApplyMethod extends Transform {
  method: MobjectBoundMethod;
  method_args: unknown[];

  constructor(method: MobjectBoundMethod, ...methodArgs: unknown[]) {
    // NOTE: Python calls `self.check_validity_of_input(method)` before
    // `super().__init__`; TS cannot touch `this` before `super()`, so the
    // same check runs as a static here and as an instance method below
    // (same pattern as MoveToTarget in this file).
    // NOTE: Python's `**kwargs` (forwarded to Transform) have no positional
    // TS spelling; every rest arg is a method arg, and a trailing plain
    // object stays a method-kwarg dict per `create_target` below.
    ApplyMethod.validateInput(method);
    super((method as unknown as { __self__: Mobject }).__self__);
    this.method = method;
    this.method_args = methodArgs;
  }

  check_validity_of_input(method: MobjectBoundMethod): void {
    ApplyMethod.validateInput(method);
  }

  checkValidityOfInput(method: MobjectBoundMethod): void {
    this.check_validity_of_input(method);
  }

  private static validateInput(method: MobjectBoundMethod): void {
    // Mirrors `inspect.ismethod(method)`: a bare function (or anything
    // without a bound `__self__`, i.e. an already-invoked result) means the
    // caller passed `mobject.method(...)` instead of `mobject.method`.
    // TS callers attach `__self__`/`__func__` the way CPython bound methods
    // carry them. NOTE: Mobjects are Proxies that raise AttributeError on
    // unknown property reads, so reading `__self__` off an already-invoked
    // result (a Mobject) throws here; like Python (where such a value fails
    // `inspect.ismethod`), that still means ValueError, not AttributeError.
    if (typeof method !== "function") {
      const error = new Error(
        "Whoops, looks like you accidentally invoked the method you want to animate",
      );
      error.name = "ValueError";
      throw error;
    }
    let self: unknown;
    try {
      self = (method as unknown as { __self__?: unknown }).__self__;
    } catch {
      self = undefined;
    }
    if (self === undefined || self === null) {
      const error = new Error(
        "Whoops, looks like you accidentally invoked the method you want to animate",
      );
      error.name = "ValueError";
      throw error;
    }
    if (!(self instanceof Mobject)) {
      const error = new Error(
        `assert isinstance(method.__self__, (Mobject, OpenGLMobject)): got ${self === null ? "null" : typeof self}`,
      );
      error.name = "AssertionError";
      throw error;
    }
  }

  override create_target(): Mobject {
    const method = this.method;
    // Make sure it's a list so that args.pop() works
    const args = [...this.method_args];
    let method_kwargs: Record<string, unknown> = {};
    if (args.length > 0) {
      const last = args[args.length - 1];
      if (
        last !== null &&
        typeof last === "object" &&
        !(last instanceof Mobject) &&
        !Array.isArray(last)
      ) {
        method_kwargs = args.pop() as Record<string, unknown>;
      }
    }
    const target = ((method as unknown as { __self__: Mobject }).__self__).copy();
    const func = (method as unknown as { __func__: (...args: any[]) => unknown }).__func__;
    // Mirrors `method.__func__(target, *args, **method_kwargs)`; TS has no
    // **kwargs channel, so a non-empty kwarg dict rides as the last
    // positional (same convention as _MethodAnimation.finish in this repo).
    if (Object.keys(method_kwargs).length > 0) func.apply(target, [...args, method_kwargs]);
    else func.apply(target, args);
    return target;
  }
}

export class ApplyFunction extends Transform {
  function: (mobject: Mobject) => Mobject;

  constructor(
    function_: (mobject: Mobject) => Mobject,
    mobject: Mobject,
    options: TransformOptions = {},
  ) {
    super(mobject, options);
    this.function = function_;
  }

  override create_target(): Mobject {
    const target = this.function(this.mobject.copy());
    if (!(target instanceof Mobject)) {
      throw new TypeError(
        "Functions passed to ApplyFunction must return object of type Mobject",
      );
    }
    return target;
  }
}

/**
 * Mirrors the `Animation.__init__` option handling for subclasses that must
 * call `super(...)` without options (TS cannot touch `this` before `super()`,
 * while Python forwards `**kwargs` through the super call). The result is
 * identical: per-class defaults underneath, explicit options on top, unknown
 * keys copied as attributes.
 */
function forwardAnimationOptions(anim: Animation, cls: unknown, options: AnimationOptions): void {
  const clsDefaults = classDefaultsOf(cls);
  const merged = { ...clsDefaults, ...options };
  if (merged.run_time !== undefined) anim.run_time = merged.run_time as number;
  if (merged.rate_func !== undefined) anim.rate_func = merged.rate_func as RateFunction;
  if (merged.reverse_rate_function !== undefined) anim.reverse_rate_function = merged.reverse_rate_function as boolean;
  if (merged.name !== undefined) anim.name = merged.name as string;
  if (merged.remover !== undefined) anim.remover = merged.remover as boolean;
  if (merged.introducer !== undefined) anim.introducer = merged.introducer as boolean;
  if (merged.suspend_mobject_updating !== undefined) anim.suspend_mobject_updating = merged.suspend_mobject_updating as boolean;
  if (merged.lag_ratio !== undefined) anim.lag_ratio = merged.lag_ratio as number;
  if (merged._on_finish !== undefined) anim.on_finish = merged._on_finish as (scene: any) => void;
  for (const [key, value] of Object.entries(merged)) {
    if (!["mobject", "run_time", "lag_ratio", "rate_func", "reverse_rate_function", "name", "remover", "introducer", "suspend_mobject_updating", "_on_finish", "use_override"].includes(key)) {
      (anim as unknown as Record<string, unknown>)[key] = value;
    }
  }
}

export type PointwiseFunction = (point: number[]) => number[] | unknown;

export class ApplyPointwiseFunction extends ApplyMethod {
  constructor(
    function_: PointwiseFunction,
    mobject: Mobject,
    options: AnimationOptions = {},
  ) {
    // NOTE: Python passes the bound method `mobject.apply_function`
    // (carrying `__self__`/`__func__`). TS prototype methods carry neither,
    // so an equivalent bound-method facade is built here (same validation
    // and `create_target` behavior downstream in ApplyMethod).
    const unbound = (mobject as unknown as { apply_function: (...args: never[]) => unknown }).apply_function;
    const bound = function (...args: unknown[]): unknown {
      return (unbound as (...callArgs: unknown[]) => unknown).apply(mobject, args);
    } as MobjectBoundMethod;
    bound.__self__ = mobject;
    bound.__func__ = unbound as (...args: never[]) => unknown;
    super(bound, function_);
    const cls = new.target ?? ApplyPointwiseFunction;
    const clsDefaults = classDefaultsOf(cls);
    forwardAnimationOptions(this, cls, {
      run_time: (clsDefaults.run_time as number | undefined) ?? DEFAULT_POINTWISE_FUNCTION_RUN_TIME,
      ...options,
    });
  }
}

export class ApplyPointwiseFunctionToCenter extends ApplyPointwiseFunction {
  function: (center: number[]) => number[] | unknown;

  constructor(
    function_: (center: number[]) => number[] | unknown,
    mobject: Mobject,
    options: AnimationOptions = {},
  ) {
    // NOTE: mirrors Python `self.function = function;
    // super().__init__(mobject.move_to, **kwargs)`. Python's super call passes
    // only the bound `move_to` method (dropping `function`, which becomes a
    // method_arg in begin()); the TS ApplyPointwiseFunction constructor needs
    // an explicit (function, mobject) pair, so a dummy pointwise function is
    // threaded through and the method facade is then rewired to `move_to`
    // (same bound-method pattern as ApplyPointwiseFunction above).
    super((point: number[]) => point, mobject, options);
    this.function = function_;
    const unbound = (mobject as unknown as { move_to: (...args: never[]) => unknown }).move_to;
    const bound = function (...args: unknown[]): unknown {
      return (unbound as (...callArgs: unknown[]) => unknown).apply(mobject, args);
    } as MobjectBoundMethod;
    bound.__self__ = mobject;
    bound.__func__ = unbound as (...args: never[]) => unknown;
    this.method = bound;
    this.method_args = [];
  }

  override begin(): void {
    this.method_args = [(this.function as (center: number[]) => unknown)(this.mobject.get_center())];
    super.begin();
  }
}
export class ApplyMatrix extends ApplyPointwiseFunction {
  constructor(matrix: unknown, mobject: Mobject, about_point: readonly number[] = ORIGIN, options: AnimationOptions = {}) {
    const initialized = ApplyMatrix.initialize_matrix(matrix);
    const origin = pointRows(pointFrom(about_point, 3), 3)[0] ?? [0, 0, 0];
    const rows = matrixRows(initialized);
    // NOTE: mirrors Python `func(p) = np.dot(p - about_point, matrix.T) +
    // about_point` (row-vector convention).
    const func = (point: number[]): number[] => {
      const centered = point.map((value, index) => value - (origin[index] ?? 0));
      const result = rows.map((row) =>
        row.reduce((sum, value, column) => sum + value * (centered[column] ?? 0), 0),
      );
      return result.map((value, index) => value + (origin[index] ?? 0));
    };
    super(func, mobject, options);
  }

  static initialize_matrix(matrix: unknown): NDArray {
    const values = matrixRows(matrix);
    if (values.length === 2 && values.every((row) => row.length === 2)) {
      const result = identityMatrix(3);
      result[0][0] = values[0][0];
      result[0][1] = values[0][1];
      result[1][0] = values[1][0];
      result[1][1] = values[1][1];
      return array(result) as NDArray;
    }
    if (values.length !== 3 || values.some((row) => row.length !== 3)) {
      const error = new Error("Matrix has bad dimensions");
      error.name = "ValueError";
      throw error;
    }
    return array(values) as NDArray;
  }

  initialize_matrix(matrix: unknown): NDArray {
    return ApplyMatrix.initialize_matrix(matrix);
  }
}

export class FadeToColor extends ApplyMethod {
  constructor(mobject: Mobject, color: ParsableManimColor, options: AnimationOptions = {}) {
    // NOTE: mirrors Python `super().__init__(mobject.set_color, color,
    // **kwargs)`; the bound-method facade carries __self__/__func__ like a
    // CPython bound method, and animation kwargs ride via
    // forwardAnimationOptions (same pattern as ApplyPointwiseFunction).
    super(boundMethodOf(mobject, "set_color"), color);
    forwardAnimationOptions(this, new.target ?? FadeToColor, options);
  }
}

export class ScaleInPlace extends ApplyMethod {
  constructor(mobject: Mobject, scale_factor: number, options: AnimationOptions = {}) {
    // NOTE: mirrors Python `super().__init__(mobject.scale, scale_factor,
    // **kwargs)`; see FadeToColor for the facade/options pattern.
    super(boundMethodOf(mobject, "scale"), scale_factor);
    forwardAnimationOptions(this, new.target ?? ScaleInPlace, options);
  }
}

export class ShrinkToCenter extends ScaleInPlace {
  constructor(mobject: Mobject, options: AnimationOptions = {}) {
    super(mobject, 0, options);
  }
}

export class Restore extends ApplyMethod {
  constructor(mobject: Mobject, options: AnimationOptions = {}) {
    // NOTE: mirrors Python `super().__init__(mobject.restore, **kwargs)`;
    // see FadeToColor for the facade/options pattern.
    super(boundMethodOf(mobject, "restore"));
    forwardAnimationOptions(this, new.target ?? Restore, options);
  }
}

export class FadeTransform extends Transform {
  to_add_on_completion: Mobject;
  stretch: boolean;
  dim_to_match: number;
  ending_mobject: Mobject;

  constructor(
    mobject: Mobject,
    targetMobject: Mobject,
    stretchOrOptions: boolean | (AnimationOptions & { stretch?: boolean; dim_to_match?: number }) = true,
    dimToMatchOrUndefined: number = 1,
    options: AnimationOptions & { stretch?: boolean; dim_to_match?: number } = {},
  ) {
    let stretch = true;
    let dimToMatch = 1;
    let rest: AnimationOptions = {};
    if (typeof stretchOrOptions === "object" && stretchOrOptions !== null) {
      const { stretch: s = true, dim_to_match: d = 1, ...r } = stretchOrOptions as AnimationOptions & { stretch?: boolean; dim_to_match?: number };
      stretch = s;
      dimToMatch = d;
      rest = r;
    } else {
      stretch = stretchOrOptions as boolean;
      dimToMatch = dimToMatchOrUndefined;
      const { stretch: _s, dim_to_match: _d, ...r } = options;
      void _s; void _d;
      rest = r;
    }
    mobject.save_state?.();
    super(new Group(mobject, targetMobject.copy()), rest);
    this.to_add_on_completion = targetMobject;
    this.stretch = stretch;
    this.dim_to_match = dimToMatch;
    this.ending_mobject = targetMobject;
  }

  begin(): void {
    this.ending_mobject = this.mobject.copy();
    Animation.prototype.begin.call(this);
    const start = this.starting_mobject as Group;
    const end = this.ending_mobject as Group;
    this.ghost_to(start.submobjects[1], start.submobjects[0]);
    this.ghost_to(end.submobjects[0], end.submobjects[1]);
  }

  ghost_to(source: Mobject, target: Mobject): void {
    if (target.get_num_points() > 0 || target.submobjects.length > 0) source.replace(target, this.dim_to_match, this.stretch);
    source.set_opacity(0);
  }

  get_all_mobjects(): Mobject[] {
    return [this.mobject, this.starting_mobject, this.ending_mobject];
  }

  override get_all_families_zipped(): Mobject[][] {
    // NOTE: Python explicitly calls Animation.get_all_families_zipped(self)
    // here (2-family zip), skipping Transform's 3-family override.
    return Animation.prototype.get_all_families_zipped.call(this);
  }

  clean_up_from_scene(scene: any): void {
    Animation.prototype.clean_up_from_scene.call(this, scene);
    scene?.remove(this.mobject);
    this.mobject.submobjects[0]?.restore?.();
    scene?.add(this.to_add_on_completion);
  }
}

export type ComplexFunctionResult =
  | { re: number; im: number }
  | { real: number; imag: number }
  | number;

export type ComplexFunction = (value: { re: number; im: number }) => ComplexFunctionResult;

export class ApplyComplexFunction extends ApplyMethod {
  function: ComplexFunction;

  constructor(function_: ComplexFunction, mobject: Mobject, options: AnimationOptions = {}) {
    // NOTE: mirrors Python, which passes the bound method
    // `mobject.apply_complex_function` (carrying `__self__`/`__func__`) to
    // `ApplyMethod.__init__`; the facade here is equivalent (same pattern
    // as ApplyPointwiseFunction in this file).
    super(boundMethodOf(mobject, "apply_complex_function"), function_);
    this.function = function_;
    forwardAnimationOptions(this, new.target ?? ApplyComplexFunction, options);
  }

  _init_path_func(): void {
    const func1 = this.function({ re: 1, im: 0 });
    let re: number;
    let im: number;
    if (typeof func1 === "number") {
      re = func1;
      im = 0;
    } else if ("re" in func1) {
      re = func1.re;
      im = func1.im;
    } else {
      re = func1.real;
      im = func1.imag;
    }
    // NOTE: mirrors Python `self.path_arc = np.log(func1).imag`, i.e. the
    // argument (angle) of the complex value func(1).
    this.path_arc = Math.atan2(im, re);
    // NOTE: Python then calls `super()._init_path_func()`, but no ancestor
    // (Transform/Animation) defines `_init_path_func` — that call would raise
    // AttributeError in Python, and nothing in manim ever calls
    // `_init_path_func`, so the port stops after the path_arc assignment.
  }
}

export class FadeTransformPieces extends FadeTransform {
  begin(): void {
    (this.mobject as Group).submobjects[0]?.align_submobjects((this.mobject as Group).submobjects[1]);
    super.begin();
  }

  ghost_to(source: Mobject, target: Mobject): void {
    const sourceFamily = source.get_family();
    const targetFamily = target.get_family();
    if (sourceFamily.length !== targetFamily.length) {
      const error = new Error(`zip() argument 2 is shorter than argument 1 (strict=True)`);
      error.name = "ValueError";
      throw error;
    }
    for (let index = 0; index < sourceFamily.length; index += 1) {
      super.ghost_to(sourceFamily[index], targetFamily[index]);
    }
  }
}

export class MoveToTarget extends Transform {
  constructor(mobject: Mobject, options: TransformOptions = {}) {
    // NOTE: Python calls `self.check_validity_of_input(mobject)` before
    // `super().__init__`; TS cannot touch `this` before `super()`, so the
    // same check runs as a static here and as an instance method below.
    MoveToTarget.validateInput(mobject);
    const target = ((mobject as unknown as { target?: Mobject | null }).target ?? new Mobject());
    super(mobject, target, options);
  }

  check_validity_of_input(mobject: Mobject): void {
    MoveToTarget.validateInput(mobject);
  }

  checkValidityOfInput(mobject: Mobject): void {
    this.check_validity_of_input(mobject);
  }

  private static validateInput(mobject: Mobject): void {
    if (!("target" in (mobject as unknown as object))) {
      const error = new Error("MoveToTarget called on mobjectwithout attribute 'target'");
      error.name = "ValueError";
      throw error;
    }
  }
}

export class _MethodAnimation extends MoveToTarget {
  methods: MethodWithArgs[];

  constructor(mobject: Mobject, methods: MethodWithArgs[]) {
    super(mobject);
    this.methods = methods;
  }

  override finish(): void {
    for (const item of this.methods) {
      const bound = item.method as unknown as MobjectBoundMethod;
      // NOTE: mirrors Python `item.method.__func__(self.mobject,
      // *item.args, **item.kwargs)`; TS has no **kwargs channel, so a
      // non-empty kwarg dict rides as the last positional (same convention
      // as ApplyMethod.create_target in this file). The `?? item.method`
      // fallback covers MethodWithArgs built from plain functions (e.g. the
      // `.animate` builder), which carry no `__func__`.
      const func = bound.__func__ ?? (item.method as (...args: any[]) => unknown);
      if (Object.keys(item.kwargs ?? {}).length > 0) func.apply(this.mobject, [...item.args, item.kwargs]);
      else func.apply(this.mobject, item.args);
    }
    super.finish();
  }
}

export class ReplacementTransform extends Transform {
  constructor(mobject: Mobject, targetMobject: Mobject, options: TransformOptions = {}) {
    super(mobject, targetMobject, { replace_mobject_with_target_in_scene: true, ...options });
  }
}

export class TransformFromCopy extends Transform {
  constructor(mobject: Mobject, targetMobject: Mobject, options: TransformOptions = {}) {
    super(targetMobject, mobject, options);
  }

  override interpolate(alpha: number): void {
    super.interpolate(1 - alpha);
  }
}

export class ClockwiseTransform extends Transform {
  constructor(
    mobject: Mobject,
    targetMobject: Mobject,
    path_arc = -Math.PI,
    options: TransformOptions = {},
  ) {
    super(mobject, targetMobject, { path_arc, ...options });
  }
}

export class CounterclockwiseTransform extends Transform {
  constructor(
    mobject: Mobject,
    targetMobject: Mobject,
    path_arc = Math.PI,
    options: TransformOptions = {},
  ) {
    super(mobject, targetMobject, { path_arc, ...options });
  }
}

export interface CyclicReplaceOptions extends TransformOptions {
  path_arc?: number;
}

export class CyclicReplace extends Transform {
  group: Group | VGroup;

  constructor(...args: Array<Mobject | Group | VGroup | TransformOptions | number>) {
    let options: CyclicReplaceOptions = {};
    const last = args[args.length - 1];
    if (
      last !== undefined &&
      !(last instanceof Mobject) &&
      typeof last === "object" &&
      !Array.isArray(last)
    ) {
      options = last as CyclicReplaceOptions;
      args = args.slice(0, -1);
    }
    const { path_arc: pathArc = 90 * DEGREES, ...rest } = options;
    let group: Group | VGroup;
    if (args.length === 1 && (args[0] instanceof Group || args[0] instanceof VGroup)) {
      group = args[0] as Group | VGroup;
    } else {
      group = new Group(...(args as Mobject[]));
    }
    super(group, { path_arc: pathArc, ...rest });
    this.group = group;
  }

  override create_target(): Group | VGroup {
    const target = this.group.copy() as unknown as Group | VGroup;
    const subs = (target as unknown as Group).submobjects;
    const groupSubs = (this.group as unknown as Group).submobjects;
    const cycled = [subs[subs.length - 1] as Mobject, ...(subs.slice(0, -1) as Mobject[])];
    if (cycled.length !== groupSubs.length) {
      const error = new Error(`zip() argument 2 is shorter than argument 1 (strict=True)`);
      error.name = "ValueError";
      throw error;
    }
    for (let index = 0; index < cycled.length; index += 1) {
      (cycled[index] as Mobject).move_to(groupSubs[index] as Mobject);
    }
    return target;
  }
}

export class Swap extends CyclicReplace {}

export class TransformAnimations extends Transform {
  start_anim: Animation;
  end_anim: Animation;

  constructor(
    startAnim: Animation,
    endAnim: Animation,
    rate_func: RateFunction = squish_rate_func(smooth),
    options: TransformOptions = {},
  ) {
    let resolvedRateFunc = rate_func;
    let rest: TransformOptions = { ...(options as TransformOptions) };
    if (
      rate_func !== null &&
      typeof rate_func === "object" &&
      typeof (rate_func as unknown as { call?: unknown }).call !== "function"
    ) {
      rest = { ...(rate_func as unknown as TransformOptions), ...options };
      resolvedRateFunc = (rest.rate_func as RateFunction | undefined) ?? squish_rate_func(smooth);
    }
    let runTime: number;
    if ("run_time" in rest && rest.run_time !== undefined) {
      runTime = rest.run_time as number;
      const { run_time: _rt, ...r } = rest;
      void _rt;
      rest = r;
    } else {
      runTime = Math.max(startAnim.run_time, endAnim.run_time);
    }
    for (const anim of [startAnim, endAnim]) {
      anim.set_run_time(runTime);
    }
    if (
      startAnim.starting_mobject !== undefined &&
      startAnim.starting_mobject !== null &&
      endAnim.starting_mobject !== undefined &&
      endAnim.starting_mobject !== null &&
      startAnim.starting_mobject.get_num_points() !==
        endAnim.starting_mobject.get_num_points()
    ) {
      startAnim.starting_mobject.align_data(endAnim.starting_mobject);
      for (const anim of [startAnim, endAnim]) {
        if (
          anim instanceof Transform &&
          anim.starting_mobject !== undefined &&
          anim.starting_mobject !== null
        ) {
          anim.starting_mobject.align_data((anim as Transform).target_mobject);
        }
      }
    }
    super(startAnim.mobject, endAnim.mobject, { rate_func: resolvedRateFunc, ...rest });
    this.start_anim = startAnim;
    this.end_anim = endAnim;
    // NOTE: mirrors Python exactly: `run_time` was popped out of kwargs (or
    // never present), so `super().__init__` resets `self.run_time` to the
    // default (1.0). The pre-super max computation only sizes the sub-anims.
    // Do NOT re-assign `this.run_time = runTime` here (verified: Python gives
    // run_time 1.0 even for max(1, 2) == 2 or an explicit run_time=5).
    // Rewire starting and ending mobjects
    startAnim.mobject = this.starting_mobject;
    endAnim.mobject = this.target_mobject;
  }

  override interpolate(alpha: number): void {
    this.start_anim.interpolate(alpha);
    this.end_anim.interpolate(alpha);
    super.interpolate(alpha);
  }
}

// NOTE: late binding for `_AnimationBuilder.build()` (see its NOTE in
// mobject/mobject.js): Python resolves manim.animation.transform's
// _MethodAnimation with a function-local import inside build(); the static TS
// edge in the other direction is unresolvable (this module must load
// types/vectorized_mobject.js, whose `VMobject extends Mobject` needs
// Mobject's body already evaluated), so the faithful class registers itself
// here instead. No new module-graph edge is added by this call.
__registerBuilderAnimationClass(_MethodAnimation);
