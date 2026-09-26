import { VGroup, VMobject, type VMobjectOptions } from "./types/vectorized_mobject.js";
import { get_vectorized_mobject_class } from "./utils.js";

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
  row_labels: VMobject[] | null = null;
  col_labels: VMobject[] | null = null;
  top_left_entry: VMobject | null = null;
  row_dim = 0;
  col_dim = 0;
  mob_table: VMobject[][] = [];
  elements!: VGroup;
  elements_without_labels!: VGroup;
  horizontal_lines!: VGroup;
  vertical_lines!: VGroup;
  line_config: Record<string, unknown> = {};

  constructor(options: TableOptions = {}) {
    super();
    const { v_buff = 0.8, h_buff = 1.3, ...rest } = options;
    void rest;
    this.v_buff = v_buff;
    this.h_buff = h_buff;
  }

  _add_labels(mob_table: VMobject[][]): VMobject[][] {
    if (this.row_labels !== null && this.row_labels !== undefined) {
      for (let k = 0; k < this.row_labels.length; k += 1) {
        mob_table[k] = [this.row_labels[k] as VMobject, ...(mob_table[k] ?? [])];
      }
    }
    if (this.col_labels !== null && this.col_labels !== undefined) {
      if (this.row_labels !== null && this.row_labels !== undefined) {
        if (this.top_left_entry !== null && this.top_left_entry !== undefined) {
          const col_labels = [this.top_left_entry, ...this.col_labels];
          mob_table.unshift(col_labels);
        } else {
          // Placeholder to use arrange_in_grid if top_left_entry is not set.
          const dummy_mobject = new (get_vectorized_mobject_class())();
          const col_labels = [dummy_mobject as VMobject, ...this.col_labels];
          mob_table.unshift(col_labels);
        }
      } else {
        mob_table.unshift(this.col_labels);
      }
    }
    return mob_table;
  }

  get_horizontal_lines(): VGroup {
    return this.horizontal_lines;
  }

  get_vertical_lines(): VGroup {
    return this.vertical_lines;
  }

  get_entries_without_labels(pos: readonly number[] | null = null): VMobject | VGroup {
    if (pos !== null && pos !== undefined) {
      const index = this.col_dim * ((pos[0] ?? 0) - 1) + ((pos[1] ?? 0) - 1);
      return this.elements_without_labels.__getitem__(index) as VMobject;
    }
    return this.elements_without_labels;
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
