import { VGroup, type VMobjectOptions } from "./types/vectorized_mobject.js";

export interface TableOptions extends VMobjectOptions {
  v_buff?: number;
  h_buff?: number;
}

/** Minimal port of manim.mobject.table.Table.
 *
 * NOTE: the full Table.__init__ (building Tex/Paragraph entries, lines and
 * background rectangles) needs the text/LaTeX pipeline, which is out of
 * scope for this port (see ). This scaffold keeps the
 * layout-critical fields (h_buff/v_buff) and the scale() override so that
 * Table-adjacent code typechecks and scale behavior stays faithful.
 */
export class Table extends VGroup {
  v_buff: number;
  h_buff: number;

  constructor(options: TableOptions = {}) {
    super();
    const { v_buff = 0.8, h_buff = 1.3, ...rest } = options;
    void rest;
    this.v_buff = v_buff;
    this.h_buff = h_buff;
  }

  override scale(
    scale_factor: number,
    // NOTE: same positional convention as VMobject.scale (see note there):
    // second slot is either a scale_stroke flag or an about_point.
    scale_stroke_or_about_point: boolean | readonly number[] | import("numpy-ts").NDArray | null = false,
    about_point?: readonly number[] | import("numpy-ts").NDArray | null,
    about_edge?: readonly number[] | import("numpy-ts").NDArray | null,
  ): this {
    // h_buff and v_buff must be adjusted so that Table.get_cell
    // can construct an accurate polygon for a cell.
    this.h_buff *= scale_factor;
    this.v_buff *= scale_factor;
    let scale_stroke = false;
    let resolved_about_point = about_point;
    if (typeof scale_stroke_or_about_point === "boolean") {
      scale_stroke = scale_stroke_or_about_point;
    } else if (scale_stroke_or_about_point !== undefined && scale_stroke_or_about_point !== null) {
      resolved_about_point = scale_stroke_or_about_point;
    }
    super.scale(scale_factor, scale_stroke, resolved_about_point ?? undefined, about_edge ?? undefined);
    return this;
  }
}
