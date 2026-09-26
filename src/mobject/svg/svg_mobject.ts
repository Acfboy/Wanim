import { array, type NDArray } from "numpy-ts";
import { RIGHT, RendererType } from "../../constants.js";
import { get_quadratic_approximation_of_cubic } from "../../utils/bezier.js";
import { hash_obj } from "../../utils/iterables.js";
import { get_full_vector_image_path } from "../../utils/images.js";
import { pointRows } from "../../utils/arrays.js";
import type { ParsableManimColor } from "../../utils/color/core.js";
import { ManimColor } from "../../utils/color/core.js";
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
  /** SVG content as a string (bypasses file I/O). */
  svg_string?: string | null;
  /** Split the parsed SVG by `<g id="...">` groups (used by MathTex). */
  split_by_id?: boolean;
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
  _svg_string: string | null;
  _split_by_id: boolean;
  _id_to_vgroup: Map<string, VGroup> = new Map();
  _all_vgroups: VGroup[] = [];
  _vgroup_stack: VGroup[] = [];
  _root_vgroup: VGroup = new VGroup();

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
    this.svg_height = options.height === undefined ? 2 : options.height;
    this.svg_width = options.width === undefined ? null : options.width;
    this._svg_string = options.svg_string ?? null;
    this._split_by_id = options.split_by_id ?? false;
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
    const use_svg_cache = options.use_svg_cache ?? false;
    // NOTE: mirrors Python `SVGMobject.__init__`: parse the SVG (from a string
    // in this port; file parsing is still unported), then apply `set_style`
    // with the explicit style arguments, then `move_into_position`.
    if (this._svg_string !== null) {
      this.init_svg_mobject(use_svg_cache);
    }
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
    // Keep the requested color for `init_colors` (MathTex recolor) while
    // leaving per-shape colors from the SVG intact.
    if (options.color !== undefined && options.color !== null) {
      this.color = new ManimColor(options.color);
    }
    if (this._svg_string !== null) {
      this.move_into_position();
    }
  }

  get hash_seed(): [string, Record<string, unknown>, Record<string, unknown>, string | null, RendererType, string | null] {
    return [
      this.constructor.name,
      this.svg_default,
      this.path_string_config,
      this.file_name,
      get_config_renderer(),
      this._svg_string,
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

  init_svg_mobject(use_svg_cache: boolean): this {
    // NOTE: mirrors Python `SVGMobject.init_svg_mobject`: on a cache hit
    // the cached mobject is copied, its submobjects are added here, and its
    // id->vgroup dict is adopted. On a miss `generate_mobject` parses the
    // SVG file (which needs the ElementTree + svgelements file pipeline,
    // not ported yet) and the result is cached.
    let hash_val: number | null = null;
    if (use_svg_cache) {
      hash_val = hash_obj(this.hash_seed);
      const cached = SVG_HASH_TO_MOB_MAP.get(hash_val);
      if (cached !== undefined) {
        const mob = cached.copy();
        this.add(...mob.submobjects);
        this.id_to_vgroup_dict = mob.id_to_vgroup_dict;
        return this;
      }
    }

    this.generate_mobject();
    if (use_svg_cache && hash_val !== null) {
      SVG_HASH_TO_MOB_MAP.set(hash_val, this.copy());
    }
    return this;
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
    // NOTE: mirrors Python `SVGMobject.generate_mobject`. In this port the SVG
    // is supplied as a string (used by Text / MarkupText / MathTex); parsing
    // an SVG *file* still needs the ElementTree + svgelements pipeline and is
    // not ported, so that branch keeps throwing.
    if (this._svg_string !== null) {
      this._generate_from_svg_string(this._svg_string);
      return this;
    }
    throw new Error("TODO: SVGMobject.generate_mobject needs the SVG file-parsing pipeline (ElementTree + svgelements), not ported yet");
  }

  /** Parse an SVG string into submobjects (Node equivalent of svgelements). */
  _generate_from_svg_string(svgString: string): this {
    const root = parse_svg_xml(svgString);
    this._id_to_vgroup = new Map();
    this._all_vgroups = [];
    this._vgroup_stack = [];
    this._root_vgroup = new VGroup();
    const allMobjects: VMobject[] = [];
    this._walk_svg_node(root, {}, IDENTITY_AFFINE, allMobjects, 0);

    // Python always keeps a "root" vgroup holding every leaf (used as a
    // fallback by MathTex._break_up_by_substrings).
    this._id_to_vgroup.set("root", this._root_vgroup);
    this.id_to_vgroup_dict = this._id_to_vgroup;

    // `split_by_id` prefers the id'd top-level vgroups (MathTex); when the SVG
    // carries no such ids (e.g. SingleStringMathTex) fall back to the flat leaf
    // list, mirroring Python's `self.add(*mobjects)`.
    if (this._split_by_id && this._all_vgroups.length > 0) {
      this.add(...this._all_vgroups);
    } else {
      this.add(...allMobjects);
    }

    // SVG coordinates are y-down; manim works in y-up.
    this.flip(RIGHT);
    return this;
  }

  _walk_svg_node(
    node: SvgXmlNode,
    inheritedAttrs: Record<string, string>,
    parentTransform: AffineMatrix,
    allMobjects: VMobject[],
    idDepth: number,
  ): void {
    const tag = node.tag;
    if (tag === "#document") {
      for (const child of node.children) {
        this._walk_svg_node(child, inheritedAttrs, parentTransform, allMobjects, idDepth);
      }
      return;
    }
    if (SVG_SKIP_TAGS.has(tag)) return;

    const attrs = effective_svg_attrs(node, inheritedAttrs);
    const localTransform = parse_svg_transform(attrs["transform"]);
    const totalTransform = multiply_affine(parentTransform, localTransform);
    const id = attrs["id"];

    if (tag === "g" || tag === "svg") {
      if (tag === "g" && id !== undefined && id !== "") {
        const vg = new VGroup();
        this._vgroup_stack.push(vg);
        for (const child of node.children) {
          this._walk_svg_node(child, attrs, totalTransform, allMobjects, idDepth + 1);
        }
        this._vgroup_stack.pop();
        this._id_to_vgroup.set(id, vg);
        if (idDepth === 0 && !id.endsWith("substring")) this._all_vgroups.push(vg);
      } else {
        for (const child of node.children) {
          this._walk_svg_node(child, attrs, totalTransform, allMobjects, idDepth);
        }
      }
      return;
    }

    const mob = this._mob_from_svg_node(tag, attrs, totalTransform);
    if (mob !== null) {
      allMobjects.push(mob);
      // NOTE: mirrors Python `SVGMobject.get_mobjects_from`: every leaf is added
      // to *all* enclosing id'd vgroups (not just the innermost one), plus the
      // synthetic "root" group. This is what makes a MathTex id like
      // `unique000` contain the glyphs of nested `unique000substring` groups as
      // well.
      this._root_vgroup.add(mob);
      for (const vg of this._vgroup_stack) vg.add(mob);
    }
  }

  _mob_from_svg_node(tag: string, attrs: Record<string, string>, transform: AffineMatrix): VMobject | null {
    const shape: SVGElementData = {
      kind: tag,
      id: attrs["id"],
      stroke_width: parse_svg_float(attrs["stroke-width"], 0),
      stroke_color: svg_color_to_manim(attrs["stroke"]),
      stroke_opacity: parse_svg_float(attrs["stroke-opacity"], 1),
      fill_color: svg_color_to_manim(attrs["fill"]),
      fill_opacity: attrs["fill"] === undefined || svg_color_to_manim(attrs["fill"]) === null
        ? 0
        : parse_svg_float(attrs["fill-opacity"], 1),
      transform,
      apply_transform: true,
    };
    if (tag === "path") {
      const d = attrs["d"] ?? "";
      shape.segments = parse_svg_path_data(d);
      if (shape.segments.length === 0) return null;
    } else if (tag === "line") {
      shape.x1 = parse_svg_float(attrs["x1"], 0);
      shape.y1 = parse_svg_float(attrs["y1"], 0);
      shape.x2 = parse_svg_float(attrs["x2"], 0);
      shape.y2 = parse_svg_float(attrs["y2"], 0);
    } else if (tag === "rect") {
      shape.x = parse_svg_float(attrs["x"], 0);
      shape.y = parse_svg_float(attrs["y"], 0);
      shape.width = parse_svg_float(attrs["width"], 0);
      shape.height = parse_svg_float(attrs["height"], 0);
      shape.rx = parse_svg_float(attrs["rx"], 0);
      shape.ry = parse_svg_float(attrs["ry"], 0);
    } else if (tag === "circle") {
      shape.cx = parse_svg_float(attrs["cx"], 0);
      shape.cy = parse_svg_float(attrs["cy"], 0);
      shape.rx = parse_svg_float(attrs["r"], 0);
      shape.ry = shape.rx;
    } else if (tag === "ellipse") {
      shape.cx = parse_svg_float(attrs["cx"], 0);
      shape.cy = parse_svg_float(attrs["cy"], 0);
      shape.rx = parse_svg_float(attrs["rx"], 0);
      shape.ry = parse_svg_float(attrs["ry"], 0);
    } else if (tag === "polygon" || tag === "polyline") {
      shape.points = parse_svg_points(attrs["points"]);
    } else {
      return null;
    }
    return this.get_mob_from_shape_element(shape);
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

/**
 * Parses SVG path data (the ``d`` attribute content, or equivalently the
 * string form accepted by ``svgelements.Path`` in the Python implementation)
 * into pre-parsed segments.
 *
 * Supports ``M/m L/l H/h V/v C/c S/s Q/q T/t Z/z``. Arc commands (``A/a``)
 * are not supported — the Python implementation converts arcs to quads via
 * ``svgelements`` (``approximate_arcs_with_quads``), which is not ported.
 */
export function parse_svg_path_data(d: string): SVGPathSegment[] {
  const tokens = d.match(/[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) ?? [];
  const isCommand = (token: string): boolean => /^[AaCcHhLlMmQqSsTtVvZz]$/.test(token);
  const segments: SVGPathSegment[] = [];
  let command = "";
  let relative = false;
  let current: [number, number] = [0, 0];
  let subpathStart: [number, number] = [0, 0];
  let prevCubicControl2: [number, number] | null = null;
  let prevQuadControl: [number, number] | null = null;
  let index = 0;

  const readNumber = (): number => {
    const token = tokens[index];
    if (token === undefined || isCommand(token)) {
      throw new Error(`Invalid SVG path data: expected a number but found '${token ?? "end of path"}'`);
    }
    index += 1;
    return parseFloat(token);
  };
  const reflect = (point: [number, number]): [number, number] => [
    2 * current[0] - point[0],
    2 * current[1] - point[1],
  ];

  // Parses one parameter group for the active command; returns whether the
  // command accepts further implicit groups (i.e. more numbers followed).
  const parseGroup = (): void => {
    switch (command) {
      case "M": {
        const x = readNumber();
        const y = readNumber();
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "Move", end: point });
        current = point;
        subpathStart = point;
        prevCubicControl2 = null;
        prevQuadControl = null;
        // NOTE: per the SVG spec, subsequent implicit coordinate pairs after
        // a moveto are treated as lineto commands.
        command = "L";
        break;
      }
      case "L": {
        const x = readNumber();
        const y = readNumber();
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "Line", end: point });
        current = point;
        prevCubicControl2 = null;
        prevQuadControl = null;
        break;
      }
      case "H": {
        const x = readNumber();
        const point: [number, number] = relative ? [current[0] + x, current[1]] : [x, current[1]];
        segments.push({ kind: "Line", end: point });
        current = point;
        prevCubicControl2 = null;
        prevQuadControl = null;
        break;
      }
      case "V": {
        const y = readNumber();
        const point: [number, number] = relative ? [current[0], current[1] + y] : [current[0], y];
        segments.push({ kind: "Line", end: point });
        current = point;
        prevCubicControl2 = null;
        prevQuadControl = null;
        break;
      }
      case "C": {
        const x1 = readNumber();
        const y1 = readNumber();
        const x2 = readNumber();
        const y2 = readNumber();
        const x = readNumber();
        const y = readNumber();
        const control1: [number, number] = relative
          ? [current[0] + x1, current[1] + y1]
          : [x1, y1];
        const control2: [number, number] = relative
          ? [current[0] + x2, current[1] + y2]
          : [x2, y2];
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "CubicBezier", control1, control2, end: point });
        current = point;
        prevCubicControl2 = control2;
        prevQuadControl = null;
        break;
      }
      case "S": {
        const x2 = readNumber();
        const y2 = readNumber();
        const x = readNumber();
        const y = readNumber();
        const control1 = prevCubicControl2 === null ? [...current] as [number, number] : reflect(prevCubicControl2);
        const control2: [number, number] = relative
          ? [current[0] + x2, current[1] + y2]
          : [x2, y2];
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "CubicBezier", control1, control2, end: point });
        current = point;
        prevCubicControl2 = control2;
        prevQuadControl = null;
        break;
      }
      case "Q": {
        const x1 = readNumber();
        const y1 = readNumber();
        const x = readNumber();
        const y = readNumber();
        const control: [number, number] = relative
          ? [current[0] + x1, current[1] + y1]
          : [x1, y1];
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "QuadraticBezier", control, end: point });
        current = point;
        prevCubicControl2 = null;
        prevQuadControl = control;
        break;
      }
      case "T": {
        const x = readNumber();
        const y = readNumber();
        const control = prevQuadControl === null ? [...current] as [number, number] : reflect(prevQuadControl);
        const point: [number, number] = relative
          ? [current[0] + x, current[1] + y]
          : [x, y];
        segments.push({ kind: "QuadraticBezier", control, end: point });
        current = point;
        prevCubicControl2 = null;
        prevQuadControl = control;
        break;
      }
      default:
        throw new Error(`Invalid SVG path data: unexpected command '${command}'`);
    }
  };

  while (index < tokens.length) {
    const token = tokens[index] as string;
    if (isCommand(token)) {
      index += 1;
      const upper = token.toUpperCase();
      relative = token !== upper;
      if (upper === "Z") {
        segments.push({ kind: "Close" });
        current = subpathStart;
        prevCubicControl2 = null;
        prevQuadControl = null;
        command = "";
        continue;
      }
      if (upper === "A") {
        throw new Error("TODO: SVG arc commands (A/a) are not ported yet");
      }
      command = upper;
      continue;
    }
    if (command === "") {
      throw new Error(`Invalid SVG path data: expected a command but found '${token}'`);
    }
    parseGroup();
  }
  return segments;
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

  constructor(path_obj: SVGPathSegment[] | string, options: VMobjectFromSVGPathOptions = {}) {
    super(options);
    this.path_obj = typeof path_obj === "string" ? parse_svg_path_data(path_obj) : path_obj;
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

// ---------------------------------------------------------------------------
// Minimal SVG-string parsing (Node has no DOMParser). The SVG strings consumed
// here are generated by this port (PangoJS markup renderer, MathJax), so a
// small non-validating parser is sufficient.
// ---------------------------------------------------------------------------

/** A parsed XML element (attribute values are entity-decoded). */
export interface SvgXmlNode {
  tag: string;
  attrib: Record<string, string>;
  children: SvgXmlNode[];
}

const SVG_SKIP_TAGS = new Set(["style", "defs", "title", "desc", "metadata"]);

/** 2D affine transform in SVG's a/b/c/d/e/f representation. */
export interface AffineMatrix {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

export const IDENTITY_AFFINE: AffineMatrix = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

function decode_svg_entities(value: string): string {
  return value.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (match, body: string) => {
    if (body.startsWith("#x") || body.startsWith("#X")) {
      const code = parseInt(body.slice(2), 16);
      return Number.isNaN(code) ? match : String.fromCodePoint(code);
    }
    if (body.startsWith("#")) {
      const code = parseInt(body.slice(1), 10);
      return Number.isNaN(code) ? match : String.fromCodePoint(code);
    }
    const named: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
    return named[body] ?? match;
  });
}

function parse_svg_attributes(attributeString: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  const attrRegex = /([^\s=/]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let match: RegExpExecArray | null;
  while ((match = attrRegex.exec(attributeString)) !== null) {
    const name = (match[1] as string).toLowerCase();
    attrs[name] = decode_svg_entities(match[2] ?? match[3] ?? "");
  }
  return attrs;
}

function strip_svg_namespace(tag: string): string {
  const colon = tag.indexOf(":");
  return (colon === -1 ? tag : tag.slice(colon + 1)).toLowerCase();
}

/** Parse an SVG/XML string into a lightweight element tree. */
export function parse_svg_xml(xml: string): SvgXmlNode {
  const root: SvgXmlNode = { tag: "#document", attrib: {}, children: [] };
  const stack: SvgXmlNode[] = [root];
  let i = 0;
  const n = xml.length;

  while (i < n) {
    const lt = xml.indexOf("<", i);
    if (lt === -1) break;
    if (xml.startsWith("<!--", lt)) {
      const end = xml.indexOf("-->", lt);
      i = end === -1 ? n : end + 3;
      continue;
    }
    if (xml.startsWith("<![CDATA[", lt)) {
      const end = xml.indexOf("]]>", lt);
      i = end === -1 ? n : end + 3;
      continue;
    }
    if (xml.startsWith("<?", lt) || xml.startsWith("<!", lt)) {
      const end = xml.indexOf(">", lt);
      i = end === -1 ? n : end + 1;
      continue;
    }

    let j = lt + 1;
    let quote = "";
    while (j < n) {
      const ch = xml[j] as string;
      if (quote) {
        if (ch === quote) quote = "";
      } else if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === ">") {
        break;
      }
      j += 1;
    }
    let inner = xml.slice(lt + 1, j);
    i = j + 1;

    if (inner.startsWith("/")) {
      const name = strip_svg_namespace(inner.slice(1).trim());
      for (let k = stack.length - 1; k > 0; k -= 1) {
        if ((stack[k] as SvgXmlNode).tag === name) {
          stack.length = k;
          break;
        }
      }
      continue;
    }

    const selfClosing = inner.endsWith("/");
    if (selfClosing) inner = inner.slice(0, -1);
    const spaceIndex = inner.search(/\s/);
    const rawTag = (spaceIndex === -1 ? inner : inner.slice(0, spaceIndex)).trim();
    if (rawTag === "") continue;
    const attributeString = spaceIndex === -1 ? "" : inner.slice(spaceIndex + 1);
    const node: SvgXmlNode = {
      tag: strip_svg_namespace(rawTag),
      attrib: parse_svg_attributes(attributeString),
      children: [],
    };
    (stack[stack.length - 1] as SvgXmlNode).children.push(node);
    if (!selfClosing) stack.push(node);
  }

  return root;
}

function effective_svg_attrs(node: SvgXmlNode, inherited: Record<string, string>): Record<string, string> {
  const attrs: Record<string, string> = {};
  for (const [key, value] of Object.entries(inherited)) {
    if (key !== "transform" && key !== "id") attrs[key] = value;
  }
  for (const [key, value] of Object.entries(node.attrib)) attrs[key] = value;
  const style = node.attrib["style"];
  if (style) {
    for (const part of style.split(";")) {
      const colon = part.indexOf(":");
      if (colon === -1) continue;
      const key = part.slice(0, colon).trim().toLowerCase();
      const value = part.slice(colon + 1).trim();
      if (key !== "") attrs[key] = value;
    }
  }
  return attrs;
}

export function multiply_affine(first: AffineMatrix, second: AffineMatrix): AffineMatrix {
  return {
    a: first.a * second.a + first.c * second.b,
    b: first.b * second.a + first.d * second.b,
    c: first.a * second.c + first.c * second.d,
    d: first.b * second.c + first.d * second.d,
    e: first.a * second.e + first.c * second.f + first.e,
    f: first.b * second.e + first.d * second.f + first.f,
  };
}

/** Parse an SVG `transform` attribute into a single affine matrix. */
export function parse_svg_transform(transform: string | undefined): AffineMatrix {
  let result: AffineMatrix = { ...IDENTITY_AFFINE };
  if (!transform) return result;

  const regex = /([a-zA-Z]+)\s*\(([^)]*)\)/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(transform)) !== null) {
    const name = (match[1] as string).toLowerCase();
    const args = (match[2] as string)
      .split(/[\s,]+/)
      .map((value) => parseFloat(value))
      .filter((value) => !Number.isNaN(value));
    let local: AffineMatrix = { ...IDENTITY_AFFINE };
    if (name === "translate") {
      local = { ...IDENTITY_AFFINE, e: args[0] ?? 0, f: args[1] ?? 0 };
    } else if (name === "scale") {
      const sx = args[0] ?? 1;
      const sy = args[1] ?? sx;
      local = { a: sx, b: 0, c: 0, d: sy, e: 0, f: 0 };
    } else if (name === "matrix") {
      local = {
        a: args[0] ?? 1,
        b: args[1] ?? 0,
        c: args[2] ?? 0,
        d: args[3] ?? 1,
        e: args[4] ?? 0,
        f: args[5] ?? 0,
      };
    } else if (name === "rotate") {
      const angle = ((args[0] ?? 0) * Math.PI) / 180;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      local = { a: cos, b: sin, c: -sin, d: cos, e: 0, f: 0 };
      if (args.length >= 3) {
        const cx = args[1] ?? 0;
        const cy = args[2] ?? 0;
        result = multiply_affine(result, { ...IDENTITY_AFFINE, e: cx, f: cy });
        result = multiply_affine(result, local);
        result = multiply_affine(result, { ...IDENTITY_AFFINE, e: -cx, f: -cy });
        continue;
      }
    }
    result = multiply_affine(result, local);
  }
  return result;
}

function parse_svg_float(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = parseFloat(value);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function svg_color_to_manim(value: string | undefined): ParsableManimColor | null {
  if (value === undefined) return null;
  const trimmed = value.trim().toLowerCase();
  if (trimmed === "" || trimmed === "none") return null;
  // MathJax emits `currentColor`; map it to black so `SingleStringMathTex`'s
  // init_colors can recolor it (matching real manim's black LaTeX output).
  if (trimmed === "currentcolor") return "#000000";
  return value.trim();
}

function parse_svg_points(value: string | undefined): Array<[number, number]> {
  if (!value) return [];
  const numbers = value
    .split(/[\s,]+/)
    .map((entry) => parseFloat(entry))
    .filter((entry) => !Number.isNaN(entry));
  const points: Array<[number, number]> = [];
  for (let index = 0; index + 1 < numbers.length; index += 2) {
    points.push([numbers[index] as number, numbers[index + 1] as number]);
  }
  return points;
}
