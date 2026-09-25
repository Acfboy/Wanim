import { array, type NDArray } from "numpy-ts";
import { Animation, type AnimationOptions, linear } from "./animation.js";
import { Mobject, Group } from "../mobject/mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { identityMatrix, matrixRows, pointFrom, pointRows, pointsFrom } from "../utils/arrays.js";
import { path_along_arc, path_along_circles, type PathFuncType } from "../utils/paths.js";
import { ORIGIN, OUT } from "../constants.js";

export interface TransformOptions extends AnimationOptions {
  path_func?: PathFuncType | null;
  path_arc?: number;
  path_arc_axis?: number[] | readonly number[] | NDArray;
  path_arc_centers?: unknown;
  replace_mobject_with_target_in_scene?: boolean;
}

export class Transform extends Animation {
  target_mobject: Mobject;
  path_arc_axis!: number[] | readonly number[] | NDArray;
  path_arc_centers!: unknown;
  private _path_arc!: number;
  private _path_func!: PathFuncType;
  replace_mobject_with_target_in_scene: boolean;

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
    return this.mobject.copy();
  }

  interpolate_submobject(submobject: Mobject, startingSubmobject: Mobject, alpha: number): void {
    // NOTE: mirrors Python `submobject.interpolate(starting, target_copy,
    // alpha, self.path_func)`; the TS Animation base tracks no target_copy,
    // so the (aligned) target_mobject points are used directly.
    submobject.interpolate(startingSubmobject, this.target_mobject, alpha, this.path_func);
  }

  get_all_mobjects(): Mobject[] {
    return [this.mobject, this.starting_mobject, this.target_mobject];
  }

  clean_up_from_scene(scene: any): void {
    super.clean_up_from_scene(scene);
    if ((this as unknown as { replace_mobject_with_target_in_scene?: boolean }).replace_mobject_with_target_in_scene) scene?.replace(this.mobject, this.target_mobject);
  }
}

export class ApplyPointwiseFunction extends Transform {
  function: (mobject: Mobject) => Mobject;

  constructor(function_: (mobject: Mobject) => Mobject, mobject: Mobject, options: AnimationOptions = {}) {
    super(mobject, function_(mobject.copy()), options);
    this.function = function_;
  }

  create_target(): Mobject {
    const target = this.function(this.mobject.copy());
    if (!(target instanceof Mobject)) throw new TypeError("Functions passed to ApplyPointwiseFunction must return object of type Mobject");
    return target;
  }
}

export class ApplyMatrix extends ApplyPointwiseFunction {
  constructor(matrix: unknown, mobject: Mobject, about_point: readonly number[] = ORIGIN, options: AnimationOptions = {}) {
    const initialized = ApplyMatrix.initialize_matrix(matrix);
    const origin = pointRows(pointFrom(about_point, 3), 3)[0] ?? [0, 0, 0];
    super((pointMobject) => {
      pointMobject.apply_function((point) => {
        const centered = point.map((value, index) => value - (origin[index] ?? 0));
        const m = matrixRows(initialized);
        const result = m[0].map((_, row) => m[row].reduce((sum, value, column) => sum + value * centered[column], 0));
        return result.map((value, index) => value + (origin[index] ?? 0));
      });
      return pointMobject;
    }, mobject, options);
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

export class FadeTransform extends Transform {
  to_add_on_completion: Mobject;
  stretch: boolean;
  dim_to_match: number;
  ending_mobject: Mobject;

  constructor(mobject: Mobject, targetMobject: Mobject, options: AnimationOptions & { stretch?: boolean; dim_to_match?: number } = {}) {
    const stretch = options.stretch ?? true;
    const dimToMatch = options.dim_to_match ?? 1;
    mobject.save_state?.();
    super(new Group(mobject, targetMobject.copy()), options);
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

  clean_up_from_scene(scene: any): void {
    Animation.prototype.clean_up_from_scene.call(this, scene);
    scene?.remove(this.mobject);
    this.mobject.submobjects[0]?.restore?.();
    scene?.add(this.to_add_on_completion);
  }
}

export class ApplyComplexFunction extends Transform {
  constructor(function_: (value: { re: number; im: number }) => { re: number; im: number }, mobject: Mobject, options: AnimationOptions = {}) {
    const target = mobject.copy();
    target.apply_complex_function(function_);
    super(mobject, target, options);
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
