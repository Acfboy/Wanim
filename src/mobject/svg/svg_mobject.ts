import { array, type NDArray } from "numpy-ts";
import { RendererType } from "../../constants.js";
import { get_quadratic_approximation_of_cubic } from "../../utils/bezier.js";
import { hash_obj } from "../../utils/iterables.js";
import { get_full_vector_image_path } from "../../utils/images.js";
import { pointRows } from "../../utils/arrays.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";

export const SVG_HASH_TO_MOB_MAP: Map<number, SVGMobject> = new Map();

export function _convert_point_to_3d(x: number, y: number): NDArray {
  return array([x, y, 0.0]) as NDArray;
}

export interface SVGMobjectOptions extends VMobjectOptions {
  file_name?: string | null;
  should_center?: boolean;
  height?: number | null;
  width?: number | null;
  svg_default?: Record<string, unknown> | null;
  path_string_config?: Record<string, unknown> | null;
  use_svg_cache?: boolean;
}

export interface SVGXmlElement {
  tag: string;
  attrib: Record<string, string>;
  children: SVGXmlElement[];
}

export interface SVGXmlTree {
  getroot(): SVGXmlElement;
}

export function createSVGXmlElement(
  tag: string,
  attrib: Record<string, string> = {},
): SVGXmlElement {
  return { tag, attrib: { ...attrib }, children: [] };
}

export function subSVGXmlElement(
  parent: SVGXmlElement,
  tag: string,
  attrib: Record<string, string> = {},
): SVGXmlElement {
  const child = createSVGXmlElement(tag, attrib);
  parent.children.push(child);
  return child;
}

export class SVGMobject extends VMobject {
  file_name: string | null;
  should_center: boolean;
  svg_height: number | null;
  svg_width: number | null;
  svg_default: Record<string, unknown>;
  path_string_config: Record<string, unknown>;
  id_to_vgroup_dict: Map<string, VMobject> = new Map();

  constructor(options: SVGMobjectOptions = {}) {
    super({
      ...options,
      color: null,
      stroke_color: null as unknown as VMobjectOptions["stroke_color"],
      fill_color: null as unknown as VMobjectOptions["fill_color"],
    });
    this.file_name = options.file_name ?? null;
    this.should_center = options.should_center ?? true;
    this.svg_height = options.height ?? 2;
    this.svg_width = options.width ?? null;
    this.svg_default =
      options.svg_default ??
      ({
        color: null,
        opacity: null,
        fill_color: null,
        fill_opacity: null,
        stroke_width: 0,
        stroke_color: null,
        stroke_opacity: null,
      } as Record<string, unknown>);
    this.path_string_config = options.path_string_config ?? {};
    void options.use_svg_cache;
  }

  get hash_seed(): [string, Record<string, unknown>, Record<string, unknown>, string | null, RendererType] {
    return [
      this.constructor.name,
      this.svg_default,
      this.path_string_config,
      this.file_name,
      get_config_renderer(),
    ];
  }

  generate_config_style_dict(): Record<string, string> {
    const keys_converting_dict: Record<string, string[]> = {
      fill: ["color", "fill_color"],
      "fill-opacity": ["opacity", "fill_opacity"],
      stroke: ["color", "stroke_color"],
      "stroke-opacity": ["opacity", "stroke_opacity"],
      "stroke-width": ["stroke_width"],
    };
    const svg_default_dict = this.svg_default;
    const result: Record<string, string> = {};
    for (const [svg_key, style_keys] of Object.entries(keys_converting_dict)) {
      for (const style_key of style_keys) {
        if (svg_default_dict[style_key] === null || svg_default_dict[style_key] === undefined) continue;
        result[svg_key] = String(svg_default_dict[style_key]);
      }
    }
    return result;
  }

  init_svg_mobject(_use_svg_cache: boolean): this {
    throw new Error("TODO: text/tex not ported yet");
  }

  get_file_path(): string {
    if (this.file_name === null || this.file_name === undefined) {
      const error = new Error("Must specify file for SVGMobject");
      error.name = "ValueError";
      throw error;
    }
    return get_full_vector_image_path(this.file_name);
  }

  modify_xml_tree(element_tree: SVGXmlTree): SVGXmlTree {
    const config_style_dict = this.generate_config_style_dict();
    const style_keys = new Set([
      "fill",
      "fill-opacity",
      "stroke",
      "stroke-opacity",
      "stroke-width",
      "style",
    ]);
    const root = element_tree.getroot();
    const root_style_dict: Record<string, string> = {};
    for (const [key, value] of Object.entries(root.attrib)) {
      if (style_keys.has(key)) root_style_dict[key] = value;
    }

    const new_root = createSVGXmlElement("svg", {});
    const config_style_node = subSVGXmlElement(new_root, "g", config_style_dict);
    const root_style_node = subSVGXmlElement(
      config_style_node,
      "g",
      root_style_dict,
    );
    root_style_node.children.push(...root.children);
    return { getroot: () => new_root };
  }

  generate_mobject(): this {
    throw new Error("TODO: text/tex not ported yet");
  }

  move_into_position(): this {
    if (this.should_center) {
      this.center();
    }
    if (this.svg_height !== null && this.svg_height !== undefined) {
      this.set({ height: this.svg_height });
    }
    if (this.svg_width !== null && this.svg_width !== undefined) {
      this.set({ width: this.svg_width });
    }
    return this;
  }
}

function get_config_renderer(): RendererType {
  const override = (globalThis as Record<string, unknown>)["__wanim_renderer__"];
  if (override === RendererType.OPENGL || override === "opengl") return RendererType.OPENGL;
  return RendererType.CAIRO;
}

export function get_svg_hash_seed(mob: SVGMobject): unknown[] {
  return [...mob.hash_seed];
}

export function svg_hash_value(mob: SVGMobject): number {
  return hash_obj(mob.hash_seed);
}

export type SVGSegmentKind = "Move" | "Line" | "QuadraticBezier" | "CubicBezier" | "Close";

export interface SVGPathSegment {
  kind: SVGSegmentKind;
  end?: [number, number];
  control?: [number, number];
  control1?: [number, number];
  control2?: [number, number];
}

export interface VMobjectFromSVGPathOptions extends VMobjectOptions {
  long_lines?: boolean;
  should_subdivide_sharp_curves?: boolean;
  should_remove_null_curves?: boolean;
}

export class VMobjectFromSVGPath extends VMobject {
  path_obj: SVGPathSegment[];
  long_lines: boolean;
  should_subdivide_sharp_curves: boolean;
  should_remove_null_curves: boolean;

  constructor(path_obj: SVGPathSegment[], options: VMobjectFromSVGPathOptions = {}) {
    super(options);
    this.path_obj = path_obj;
    this.long_lines = options.long_lines ?? false;
    this.should_subdivide_sharp_curves = options.should_subdivide_sharp_curves ?? false;
    this.should_remove_null_curves = options.should_remove_null_curves ?? false;
    // The base Mobject constructor already called generate_points(), but at
    // that time path_obj was not yet assigned (TS requires super() first,
    // whereas Python sets path_obj before super().__init__()). Regenerate
    // now that the path is available so points are populated at construction.
    this.generate_points();
  }

  generate_points(): this {
    this.handle_commands();
    if (get_config_renderer() === RendererType.OPENGL) {
      // NOTE: subdivide_sharp_curves / get_points_without_null_curves live
      // in another translation batch; only the Cairo path is ported here.
      if (this.should_subdivide_sharp_curves) {
        throw new Error("TODO: VMobjectFromSVGPath subdivide_sharp_curves not ported yet");
      }
      if (this.should_remove_null_curves) {
        throw new Error("TODO: VMobjectFromSVGPath get_points_without_null_curves not ported yet");
      }
    }
    return this;
  }

  init_points(): this {
    this.generate_points();
    return this;
  }

  handle_commands(): this {
    // Guard: the base Mobject constructor invokes generate_points() (which
    // delegates here) before this class assigns path_obj. Treat that early
    // call as an empty path; the constructor regenerates afterwards.
    const segments = this.path_obj ?? [];
    const all_points: number[][] = [];
    let last_move: number[] | null = null;
    let curve_start: number[] | null = null;
    let last_true_move: number[] | null = null;

    function move_pen(pt: number[], true_move = false): void {
      last_move = pt;
      if (curve_start === null) curve_start = last_move;
      if (true_move) last_true_move = last_move;
    }

    const rowsOf = (value: NDArray): number[][] => pointRows(value, this.dim);
    const rowOf = (value: NDArray): number[] => rowsOf(value)[0] ?? [0, 0, 0];

    if (this.n_points_per_curve === 4) {
      const add_cubic = (start: number[], cp1: number[], cp2: number[], end: number[]): void => {
        if (all_points.length % 4 !== 0) throw new Error(`AssertionError: ${all_points.length}`);
        all_points.push(start, cp1, cp2, end);
        move_pen(end);
      };

      const add_quad = (start: number[], cp: number[], end: number[]): void => {
        add_cubic(
          start,
          start.map((v, i) => (v + (cp[i] ?? 0) + (cp[i] ?? 0)) / 3),
          end.map((v, i) => ((cp[i] ?? 0) + (cp[i] ?? 0) + v) / 3),
          end,
        );
        move_pen(end);
      };

      const add_line = (start: number[], end: number[]): void => {
        add_cubic(
          start,
          start.map((v, i) => (v + v + (end[i] ?? 0)) / 3),
          start.map((v, i) => (v + (end[i] ?? 0) + (end[i] ?? 0)) / 3),
          end,
        );
        move_pen(end);
      };

      for (const segment of segments) {
        if (segment.kind === "Move") {
          move_pen(rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))), true);
        } else if (segment.kind === "Line") {
          add_line(last_move ?? [0, 0, 0], rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))));
        } else if (segment.kind === "QuadraticBezier") {
          add_quad(
            last_move ?? [0, 0, 0],
            rowOf(_convert_point_to_3d(...(segment.control ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))),
          );
        } else if (segment.kind === "CubicBezier") {
          add_cubic(
            last_move ?? [0, 0, 0],
            rowOf(_convert_point_to_3d(...(segment.control1 ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.control2 ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))),
          );
        } else if (segment.kind === "Close") {
          const diff = Math.hypot(
            ...((last_move ?? [0, 0, 0]).map((v, i) => v - ((last_true_move ?? [0, 0, 0])[i] ?? 0)) ?? [0]),
          );
          if (Math.abs(diff) > 0.0001) add_line(last_move ?? [0, 0, 0], last_true_move ?? [0, 0, 0]);
          curve_start = null;
        } else {
          throw new Error(`Not implemented: ${segment.kind}`);
        }
      }
    } else {
      const add_cubic = (start: number[], cp1: number[], cp2: number[], end: number[]): void => {
        if (all_points.length % 3 !== 0) throw new Error(`AssertionError: ${all_points.length}`);
        const two_quads = pointRows(
          get_quadratic_approximation_of_cubic(
            array(start) as NDArray,
            array(cp1) as NDArray,
            array(cp2) as NDArray,
            array(end) as NDArray,
          ),
          this.dim,
        );
        all_points.push(...two_quads.slice(0, 3));
        all_points.push(...two_quads.slice(3));
        move_pen(end);
      };

      const add_quad = (start: number[], cp: number[], end: number[]): void => {
        if (all_points.length % 3 !== 0) throw new Error(`AssertionError: ${all_points.length}`);
        all_points.push(start, cp, end);
        move_pen(end);
      };

      const add_line = (start: number[], end: number[]): void => {
        let current = start;
        if (this.long_lines) {
          const midpoint = current.map((v, i) => (v + (end[i] ?? 0)) / 2);
          add_quad(current, current.map((v, i) => (v + (midpoint[i] ?? 0)) / 2), midpoint);
          current = midpoint;
        }
        add_quad(
          current,
          current.map((v, i) => (v + (end[i] ?? 0)) / 2),
          end,
        );
        move_pen(end);
      };

      for (const segment of segments) {
        if (segment.kind === "Move") {
          move_pen(rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))), true);
        } else if (segment.kind === "Line") {
          add_line(last_move ?? [0, 0, 0], rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))));
        } else if (segment.kind === "QuadraticBezier") {
          add_quad(
            last_move ?? [0, 0, 0],
            rowOf(_convert_point_to_3d(...(segment.control ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))),
          );
        } else if (segment.kind === "CubicBezier") {
          add_cubic(
            last_move ?? [0, 0, 0],
            rowOf(_convert_point_to_3d(...(segment.control1 ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.control2 ?? [0, 0]))),
            rowOf(_convert_point_to_3d(...(segment.end ?? [0, 0]))),
          );
        } else if (segment.kind === "Close") {
          const diff = Math.hypot(
            ...((last_move ?? [0, 0, 0]).map((v, i) => v - ((last_true_move ?? [0, 0, 0])[i] ?? 0)) ?? [0]),
          );
          if (Math.abs(diff) > 0.0001) add_line(last_move ?? [0, 0, 0], last_true_move ?? [0, 0, 0]);
          curve_start = null;
        } else {
          throw new Error(`Not implemented: ${segment.kind}`);
        }
      }
    }

    this.points = array(all_points.length > 0 ? all_points : []).reshape(
      all_points.length,
      this.dim,
    ) as NDArray;
    return this;
  }
}
