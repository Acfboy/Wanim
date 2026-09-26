import { array, type NDArray } from "numpy-ts";
import { RendererType } from "../../constants.js";
import { get_quadratic_approximation_of_cubic } from "../../utils/bezier.js";
import { hash_obj } from "../../utils/iterables.js";
import { get_full_vector_image_path } from "../../utils/images.js";
import { pointRows } from "../../utils/arrays.js";
import type { ParsableManimColor } from "../../utils/color/core.js";
import { VMobject, VGroup, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Circle } from "../geometry/arc.js";
import { Line } from "../geometry/line.js";
import { Polygon, Rectangle, RoundedRectangle } from "../geometry/polygram.js";

export const SVG_HASH_TO_MOB_MAP: Map<number, SVGMobject> = new Map();

export function _convert_point_to_3d(x: number, y: number): NDArray {
  return array([x, y, 0.0]) as NDArray;
}

export interface SVGMobjectOptions extends VMobjectOptions {
  file_name?: string | null;
  should_center?: boolean;
  height?: number | null;
  width?: number | null;
  /** Combined fill+stroke opacity override (mirrors the Python parameter). */
  opacity?: number | null;
  svg_default?: Record<string, unknown> | null;
  path_string_config?: Record<string, unknown> | null;
  use_svg_cache?: boolean;
}

/**
 * Local structural stand-in for an `svgelements` element (the third-party
 * SVG parsing library used by the Python implementation, which is not
 * ported). The `kind` mirrors the svgelements class (`SVG`/`Group`/`Use`
 * containers and `Path`/`SimpleLine`/`Rect`/`Circle`/`Ellipse`/`Polygon`/
 * `Polyline`/`Text` shapes); `id` mirrors `element.values["id"]`; the
 * remaining fields mirror the shape attributes and style/transform info
 * read by the conversion methods below.
 */
export interface SVGElementData {
  kind: string;
  id?: unknown;
  children?: SVGElementData[];
  /** Path element: pre-parsed segments (arcs already approximated). */
  segments?: SVGPathSegment[];
  /** Line element endpoints. */
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  /** Rect element geometry. */
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  rx?: number;
  ry?: number;
  /** Circle/Ellipse element center and radii. */
  cx?: number;
  cy?: number;
  /** Polygon/Polyline vertices as [x, y] pairs. */
  points?: Array<[number, number] | readonly [number, number]>;
  /** Style info (mirrors shape.stroke_width / stroke.hexrgb /
   * stroke.opacity / fill.hexrgb / fill.opacity). */
  stroke_width?: number | null;
  stroke_color?: ParsableManimColor | null;
  stroke_opacity?: number | null;
  fill_color?: ParsableManimColor | null;
  fill_opacity?: number | null;
  /** Transform matrix (mirrors Transformable) and its `.apply` flag. */
  transform?: SVGMatrixData | null;
  apply_transform?: boolean;
}

/** Mirrors an `svgelements` transformation matrix (a/b/c/d/e/f). */
export interface SVGMatrixData {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
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
  id_to_vgroup_dict: Map<string, VGroup> = new Map();

  constructor(options: SVGMobjectOptions = {}) {
    // NOTE: the svg-specific style keys accept null ("keep file value",
    // mirroring Python's None); strip them before delegating since
    // VMobjectOptions types them as non-nullable.
    const {
      fill_color: opt_fill_color,
      fill_opacity: opt_fill_opacity,
      stroke_color: opt_stroke_color,
      stroke_opacity: opt_stroke_opacity,
      stroke_width: opt_stroke_width,
      ...rest_options
    } = options as SVGMobjectOptions & {
      fill_color?: ParsableManimColor | null;
      fill_opacity?: number | null;
      stroke_color?: ParsableManimColor | null;
      stroke_opacity?: number | null;
      stroke_width?: number | null;
    };
    super({
      ...rest_options,
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
    // NOTE: the Python __init__ parses the SVG file here
    // (init_svg_mobject), then applies set_style with the explicit style
    // arguments and moves into position. File parsing needs the ElementTree
    // + svgelements pipeline (unported), so construction stays lazy; only
    // the explicit non-null style overrides are applied, mirroring the
    // Python set_style call where None means "keep what the file set".
    const style_override: {
      fill_color?: ParsableManimColor | null;
      fill_opacity?: number | null;
      stroke_color?: ParsableManimColor | null;
      stroke_opacity?: number | null;
      stroke_width?: number | null;
    } = {};
    if (options.fill_color !== undefined && options.fill_color !== null) {
      style_override.fill_color = options.fill_color;
    }
    if (opt_fill_opacity !== undefined && opt_fill_opacity !== null) {
      style_override.fill_opacity = opt_fill_opacity;
    }
    if (options.stroke_color !== undefined && options.stroke_color !== null) {
      style_override.stroke_color = options.stroke_color;
    }
    if (opt_stroke_opacity !== undefined && opt_stroke_opacity !== null) {
      style_override.stroke_opacity = opt_stroke_opacity;
    }
    if (opt_stroke_width !== undefined && opt_stroke_width !== null) {
      style_override.stroke_width = opt_stroke_width;
    }
    if (Object.keys(style_override).length > 0) {
      this.set_style(style_override);
    }
    // NOTE: Python also calls move_into_position() here; with no parsed
    // points yet that is a no-op in practice, so it is left to the caller
    // (mirroring how generate_mobject callers behave once parsing lands).
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

  get_mobjects_from(
    svg: SVGElementData,
  ): [VMobject[], Map<string, VGroup>] {
    const result: VMobject[] = [];
    const stack: Array<[SVGElementData, number]> = [[svg, 1]];
    let group_id_number = 0;
    let vgroup_stack: string[] = ["root"];
    const vgroup_names: string[] = ["root"];
    const vgroups = new Map<string, VGroup>([["root", new VGroup()]]);
    while (stack.length > 0) {
      const popped = stack.pop() as [SVGElementData, number];
      const element = popped[0];
      const depth = popped[1];
      // Reduce stack heights
      vgroup_stack = vgroup_stack.slice(0, depth);
      let group_name: string;
      if (element.id !== undefined && element.id !== null) {
        // Mirrors `str(element.values["id"])` (raises KeyError when absent).
        group_name = String(element.id);
      } else {
        group_name = `numbered_group_${group_id_number}`;
        group_id_number += 1;
      }
      const vg = new VGroup();
      vgroup_names.push(group_name);
      vgroup_stack.push(group_name);
      vgroups.set(group_name, vg);
      void vgroup_names;

      if (element.kind === "group" || element.kind === "use" || element.kind === "svg") {
        // Mirrors `element[::-1]`: push reversed so pops come out in order.
        const children = element.children ?? [];
        for (let index = children.length - 1; index >= 0; index -= 1) {
          stack.push([children[index] as SVGElementData, depth + 1]);
        }
      }
      // Add element to the parent vgroup
      try {
        if (
          element.kind === "path" ||
          element.kind === "line" ||
          element.kind === "rect" ||
          element.kind === "circle" ||
          element.kind === "ellipse" ||
          element.kind === "polygon" ||
          element.kind === "polyline" ||
          element.kind === "text"
        ) {
          const mob = this.get_mob_from_shape_element(element);
          if (mob !== null) {
            result.push(mob);
            for (const parent_name of vgroup_stack.slice(0, -1)) {
              vgroups.get(parent_name)?.add(mob);
            }
          }
        }
      } catch (e) {
        console.error(`Exception occurred in 'get_mobjects_from'. Details: ${e}`);
      }
    }

    return [result, vgroups];
  }

  get_mob_from_shape_element(shape: SVGElementData): VMobject | null {
    let mob: VMobject | null;
    if (shape.kind === "path") {
      mob = this.path_to_mobject(shape);
    } else if (shape.kind === "line") {
      mob = SVGMobject.line_to_mobject(shape);
    } else if (shape.kind === "rect") {
      mob = SVGMobject.rect_to_mobject(shape);
    } else if (shape.kind === "circle" || shape.kind === "ellipse") {
      mob = SVGMobject.ellipse_to_mobject(shape);
    } else if (shape.kind === "polygon") {
      mob = SVGMobject.polygon_to_mobject(shape);
    } else if (shape.kind === "polyline") {
      mob = this.polyline_to_mobject(shape);
    } else if (shape.kind === "text") {
      mob = SVGMobject.text_to_mobject(shape);
    } else {
      console.warn(`Unsupported element type: ${shape.kind}`);
      mob = null;
    }
    if (mob === null || !mob.has_points()) {
      return null;
    }
    SVGMobject.apply_style_to_mobject(mob, shape);
    // NOTE: every svgelements shape is Transformable; `.apply` defaults to
    // true, so a present transform is applied unless explicitly disabled.
    if (
      shape.transform !== undefined &&
      shape.transform !== null &&
      (shape.apply_transform ?? true)
    ) {
      SVGMobject.handle_transform(mob, shape.transform);
    }
    return mob;
  }

  static handle_transform(mob: VMobject, matrix: SVGMatrixData): VMobject {
    const mat = [
      [matrix.a, matrix.c],
      [matrix.b, matrix.d],
    ];
    const vec = [matrix.e, matrix.f, 0.0];
    mob.apply_matrix(mat);
    mob.shift(vec);
    return mob;
  }

  static apply_style_to_mobject(mob: VMobject, shape: SVGElementData): VMobject {
    mob.set_style({
      stroke_width: shape.stroke_width ?? null,
      stroke_color: shape.stroke_color ?? null,
      stroke_opacity: shape.stroke_opacity ?? null,
      fill_color: shape.fill_color ?? null,
      fill_opacity: shape.fill_opacity ?? null,
    });
    return mob;
  }

  path_to_mobject(path: SVGElementData): VMobjectFromSVGPath {
    return new VMobjectFromSVGPath(
      path.segments ?? [],
      { ...(this.path_string_config as VMobjectFromSVGPathOptions) },
    );
  }

  static line_to_mobject(line: SVGElementData): Line {
    return new Line(
      _convert_point_to_3d(line.x1 ?? 0, line.y1 ?? 0),
      _convert_point_to_3d(line.x2 ?? 0, line.y2 ?? 0),
    );
  }

  static rect_to_mobject(rect: SVGElementData): Rectangle {
    const width = rect.width ?? 0;
    const height = rect.height ?? 0;
    const rx = rect.rx ?? 0;
    const ry = rect.ry ?? 0;
    let mob: Rectangle;
    if (rx === 0 || ry === 0) {
      mob = new Rectangle({ width, height });
    } else {
      // NOTE: RoundedRectangle with a nonzero corner radius needs
      // Polygram.round_corners (ArcBetweenPoints/Line), which throws its own
      // TODO in this port; the construction below mirrors the original.
      const rounded = new RoundedRectangle({
        width,
        height: (height * rx) / ry,
        corner_radius: rx,
      });
      rounded.stretch_to_fit_height(height);
      mob = rounded;
    }
    mob.shift(
      _convert_point_to_3d((rect.x ?? 0) + width / 2, (rect.y ?? 0) + height / 2),
    );
    return mob;
  }

  static ellipse_to_mobject(ellipse: SVGElementData): Circle {
    const mob = new Circle(ellipse.rx ?? 0);
    if (ellipse.rx !== ellipse.ry) {
      mob.stretch_to_fit_height(2 * (ellipse.ry ?? 0));
    }
    mob.shift(_convert_point_to_3d(ellipse.cx ?? 0, ellipse.cy ?? 0));
    return mob;
  }

  static polygon_to_mobject(polygon: SVGElementData): Polygon {
    // NOTE: passed as plain rows (Python passes a list of 1D arrays; the
    // local pointRows helper does not unpack arrays of NDArray, so the
    // conversion happens here).
    const points: number[][] = (polygon.points ?? []).map(
      (point) => pointRows(_convert_point_to_3d(point[0] ?? 0, point[1] ?? 0), 3)[0] ?? [0, 0, 0],
    );
    return new Polygon(...points);
  }

  polyline_to_mobject(polyline: SVGElementData): VMobject {
    // NOTE: see polygon_to_mobject on the plain-rows conversion.
    const points: number[][] = (polyline.points ?? []).map(
      (point) => pointRows(_convert_point_to_3d(point[0] ?? 0, point[1] ?? 0), this.dim)[0] ?? [0, 0, 0],
    );
    const vmobject_class = (
      this.constructor as unknown as typeof VMobject
    ).get_mobject_type_class();
    return new vmobject_class().set_points_as_corners(points);
  }

  static text_to_mobject(_text: SVGElementData): VMobject | null {
    // NOTE: faithful to the original, which only warns and returns None
    // (text rendering is out of scope for the entire port).
    console.warn("Unsupported element type: text (SVG text is not ported yet)");
    return null;
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
