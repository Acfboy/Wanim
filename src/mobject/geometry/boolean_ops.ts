import { array, type NDArray } from "numpy-ts";
import { config } from "../../_config/utils.js";
import { RendererType } from "../../constants.js";
import { isNDArray, numeric, pointRows } from "../../utils/arrays.js";
import { VMobject } from "../types/vectorized_mobject.js";

/** Verbs emitted by {@link SkiaPath}, mirroring `pathops.PathVerb`. */
export enum PathVerb {
  MOVE = "move",
  LINE = "line",
  QUAD = "quad",
  CUBIC = "cubic",
  CLOSE = "close",
}

export interface SkiaPathCommand {
  verb: PathVerb;
  /** Control/end points as [x, y] pairs (empty for CLOSE). */
  points: number[][];
}

/**
 * Minimal local stand-in for `pathops.Path` (Skia).
 *
 * The Python implementation uses the `pathops` (Skia) library for boolean
 * path operations; there is no equivalent npm dependency available here (and
 * third-party packages other than numpy-ts are out of scope), so this shim
 * records the same move/quad/cubic/close command sequence the original
 * builds. It additionally implements the `[verb, points]` iteration protocol
 * used by `_convert_skia_path_to_vmobject`, and `getPen()` returning itself
 * (mirroring `SkiaPath.getPen()` used by the boolean combinators).
 */
export class SkiaPath {
  commands: SkiaPathCommand[] = [];

  moveTo(x: number, y: number): this {
    this.commands.push({ verb: PathVerb.MOVE, points: [[x, y]] });
    return this;
  }

  lineTo(x: number, y: number): this {
    this.commands.push({ verb: PathVerb.LINE, points: [[x, y]] });
    return this;
  }

  quadTo(cpx: number, cpy: number, x: number, y: number): this {
    this.commands.push({ verb: PathVerb.QUAD, points: [[cpx, cpy], [x, y]] });
    return this;
  }

  cubicTo(
    cp1x: number, cp1y: number,
    cp2x: number, cp2y: number,
    x: number, y: number,
  ): this {
    this.commands.push({
      verb: PathVerb.CUBIC,
      points: [[cp1x, cp1y], [cp2x, cp2y], [x, y]],
    });
    return this;
  }

  close(): this {
    this.commands.push({ verb: PathVerb.CLOSE, points: [] });
    return this;
  }

  getPen(): this {
    return this;
  }

  *[Symbol.iterator](): Generator<[PathVerb, number[][]], void, void> {
    for (const command of this.commands) {
      yield [command.verb, command.points];
    }
  }
}

export class _BooleanOps extends VMobject {
  _convert_2d_to_3d_array(points: Iterable<number[] | readonly number[] | NDArray>, z_dim = 0): NDArray {
    const list_of_points: number[][] = [...points].map((point) =>
      isNDArray(point)
        ? ((point as NDArray).toArray() as unknown[]).map(numeric)
        : [...(point as readonly number[])].map(numeric),
    );
    for (let index = 0; index < list_of_points.length; index += 1) {
      const point = list_of_points[index] as number[];
      if (point.length === 2) list_of_points[index] = [...point, z_dim];
    }
    return array(list_of_points) as NDArray;
  }

  _convert_vmobject_to_skia_path(vmobject: VMobject): SkiaPath {
    const path = new SkiaPath();

    // Mirrors `np.all(np.isfinite(vmobject.points))`.
    const flat = (vmobject.points.toArray() as unknown[]).flat(Infinity) as unknown[];
    const all_finite = flat.every((value) => Number.isFinite(Number(value)));
    let points: NDArray;
    if (all_finite) {
      points = vmobject.points;
    } else {
      points = array([[0, 0, 0]]) as NDArray; // point invalid?
    }

    if (pointRows(points, vmobject.dim).length === 0) {
      // what? No points so return empty path
      return path;
    }

    // In OpenGL it's quadratic bezier curves while on Cairo it's cubic...
    if (config.renderer === RendererType.OPENGL) {
      const subpaths = vmobject.get_subpaths_from_points(points);
      for (const subpath of subpaths) {
        // NOTE: Python calls `get_bezier_tuples_from_points` (quadratic
        // triples, OpenGL renderer only). wanim's VMobject stores cubic
        // tuples; chunk by n_points_per_curve so quadratic-mode mobjects
        // (nppc == 3) yield exact quads, mirroring `for _p0, p1, p2 in quads`.
        const sub_rows = pointRows(subpath, vmobject.dim);
        const nppc = vmobject.n_points_per_curve;
        const start = sub_rows[0] ?? [0, 0, 0];
        path.moveTo(start[0] ?? 0, start[1] ?? 0);
        for (let index = 0; index + nppc <= sub_rows.length; index += nppc) {
          const tuple = sub_rows.slice(index, index + nppc);
          const p1 = tuple[1] ?? tuple[0] ?? [0, 0, 0];
          const p2 = tuple[nppc - 1] ?? tuple[0] ?? [0, 0, 0];
          path.quadTo(p1[0] ?? 0, p1[1] ?? 0, p2[0] ?? 0, p2[1] ?? 0);
        }
        if (vmobject.consider_points_equals(sub_rows[0], sub_rows[sub_rows.length - 1])) {
          path.close();
        }
      }
    } else if (config.renderer === RendererType.CAIRO) {
      const subpaths = vmobject.gen_subpaths_from_points_2d(points);
      for (const subpath of subpaths) {
        const quads = vmobject.gen_cubic_bezier_tuples_from_points(subpath);
        const sub_rows = pointRows(subpath, vmobject.dim);
        const start = sub_rows[0] ?? [0, 0, 0];
        path.moveTo(start[0] ?? 0, start[1] ?? 0);
        for (const quad of quads) {
          const p1 = quad[1] ?? [0, 0, 0];
          const p2 = quad[2] ?? [0, 0, 0];
          const p3 = quad[3] ?? [0, 0, 0];
          path.cubicTo(
            p1[0] ?? 0, p1[1] ?? 0,
            p2[0] ?? 0, p2[1] ?? 0,
            p3[0] ?? 0, p3[1] ?? 0,
          );
        }
        if (
          vmobject.consider_points_equals_2d(
            sub_rows[0] ?? [0, 0, 0],
            sub_rows[sub_rows.length - 1] ?? [0, 0, 0],
          )
        ) {
          path.close();
        }
      }
    }

    return path;
  }

  _convert_skia_path_to_vmobject(path: SkiaPath): VMobject {
    const vmobject = this as unknown as VMobject;
    let current_path_start: number[] = [0, 0, 0];
    for (const [path_verb, points] of path) {
      if (path_verb === PathVerb.MOVE) {
        const parts = pointRows(this._convert_2d_to_3d_array(points), this.dim);
        for (const part of parts) {
          current_path_start = [...part];
          vmobject.start_new_path(part);
        }
      } else if (path_verb === PathVerb.CUBIC) {
        const [n1 = [0, 0, 0], n2 = [0, 0, 0], n3 = [0, 0, 0]] = pointRows(
          this._convert_2d_to_3d_array(points),
          this.dim,
        );
        vmobject.add_cubic_bezier_curve_to(n1, n2, n3);
      } else if (path_verb === PathVerb.LINE) {
        const parts = pointRows(this._convert_2d_to_3d_array(points), this.dim);
        vmobject.add_line_to(parts[0] ?? [0, 0, 0]);
      } else if (path_verb === PathVerb.CLOSE) {
        vmobject.add_line_to(current_path_start);
      } else if (path_verb === PathVerb.QUAD) {
        const [n1 = [0, 0, 0], n2 = [0, 0, 0]] = pointRows(
          this._convert_2d_to_3d_array(points),
          this.dim,
        );
        vmobject.add_quadratic_bezier_curve_to(n1, n2);
      } else {
        throw new Error(`Unsupported: ${path_verb}`);
      }
    }
    return vmobject;
  }
}

/**
 * Local stand-ins for the `pathops` boolean combinators (union, difference,
 * intersection, xor) used by the classes below.
 *
 * The Python implementation delegates to the Skia `pathops` library, which
 * has no equivalent among the allowed dependencies (only numpy-ts may be
 * used), so these throw until a local boolean-op routine lands. They exist
 * as named functions so the call sites below stay verbatim mirrors of the
 * originals.
 */
function pathops_union(paths: SkiaPath[], outpen: { getPen(): SkiaPath }): void {
  void paths;
  void outpen;
  throw new Error("TODO: boolean ops need the `pathops` (skia) library, not ported yet");
}

function pathops_difference(paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath }): void {
  void paths;
  void clip;
  void outpen;
  throw new Error("TODO: boolean ops need the `pathops` (skia) library, not ported yet");
}

function pathops_intersection(paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath }): void {
  void paths;
  void clip;
  void outpen;
  throw new Error("TODO: boolean ops need the `pathops` (skia) library, not ported yet");
}

function pathops_xor(paths: SkiaPath[], clip: SkiaPath[], outpen: { getPen(): SkiaPath }): void {
  void paths;
  void clip;
  void outpen;
  throw new Error("TODO: boolean ops need the `pathops` (skia) library, not ported yet");
}

export class Union extends _BooleanOps {
  constructor(...args: unknown[]) {
    let vmobjects: VMobject[];
    let options: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as Record<string, unknown>;
      vmobjects = args.slice(0, -1) as VMobject[];
    } else {
      vmobjects = args as VMobject[];
    }
    if (vmobjects.length < 2) {
      const error = new Error("At least 2 mobjects needed for Union.");
      error.name = "ValueError";
      throw error;
    }
    super(options);
    const paths = vmobjects.map((vmobject) => this._convert_vmobject_to_skia_path(vmobject));
    const outpen = new SkiaPath();
    pathops_union(paths, outpen.getPen());
    this._convert_skia_path_to_vmobject(outpen);
  }
}

export class Difference extends _BooleanOps {
  constructor(subject: VMobject, clip: VMobject, options: Record<string, unknown> = {}) {
    super(options);
    const outpen = new SkiaPath();
    pathops_difference(
      [this._convert_vmobject_to_skia_path(subject)],
      [this._convert_vmobject_to_skia_path(clip)],
      outpen.getPen(),
    );
    this._convert_skia_path_to_vmobject(outpen);
  }
}

export class Intersection extends _BooleanOps {
  constructor(...args: unknown[]) {
    let vmobjects: VMobject[];
    let options: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (
      args.length > 0 &&
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last) &&
      !(last instanceof VMobject)
    ) {
      options = last as Record<string, unknown>;
      vmobjects = args.slice(0, -1) as VMobject[];
    } else {
      vmobjects = args as VMobject[];
    }
    if (vmobjects.length < 2) {
      const error = new Error("At least 2 mobjects needed for Intersection.");
      error.name = "ValueError";
      throw error;
    }
    super(options);
    const outpen = new SkiaPath();
    pathops_intersection(
      [this._convert_vmobject_to_skia_path(vmobjects[0] as VMobject)],
      [this._convert_vmobject_to_skia_path(vmobjects[1] as VMobject)],
      outpen.getPen(),
    );
    let current = outpen;
    for (let index = 2; index < vmobjects.length; index += 1) {
      const next = new SkiaPath();
      pathops_intersection(
        [current],
        [this._convert_vmobject_to_skia_path(vmobjects[index] as VMobject)],
        next.getPen(),
      );
      current = next;
    }
    this._convert_skia_path_to_vmobject(current);
  }
}

export class Exclusion extends _BooleanOps {
  constructor(subject: VMobject, clip: VMobject, options: Record<string, unknown> = {}) {
    super(options);
    const outpen = new SkiaPath();
    pathops_xor(
      [this._convert_vmobject_to_skia_path(subject)],
      [this._convert_vmobject_to_skia_path(clip)],
      outpen.getPen(),
    );
    this._convert_skia_path_to_vmobject(outpen);
  }
}
