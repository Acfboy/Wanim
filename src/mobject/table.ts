// Port of manim.mobject.table.Table (plus MathTable/MobjectTable/
// IntegerTable/DecimalTable).
//
// NOTE (text/tex out of scope): the default `element_to_mobject`
// (Paragraph) and the MathTable/IntegerTable/DecimalTable defaults
// (MathTex/Integer/DecimalNumber) need the LaTeX/text pipeline, so they
// throw `TODO: text/tex not ported yet` when actually called to build an
// entry (same convention as Matrix in ./matrix.ts). Every code path that
// only needs already-constructed mobjects (MobjectTable, explicit
// element_to_mobject factories, labels passed in by the caller, grid
// layout, lines, colors, cells, create()) is ported faithfully.
// NOTE: import order matters at runtime: ./types/vectorized_mobject.js
// (hence ../mobject/mobject.js) must evaluate before the animation modules
// below, because mobject.js and animation.js import each other and only the
// mobject-first direction avoids a TDZ error (pre-existing cycle).
import { VGroup, VMobject } from "./types/vectorized_mobject.js";
import { ORIGIN } from "../constants.js";
import type { Animation, AnimationClass } from "../animation/animation.js";
import { AnimationGroup } from "../animation/composition.js";
import { Create, Write } from "../animation/creation.js";
import { FadeIn } from "../animation/fading.js";
import { BLACK, ManimColor, type ParsableManimColor } from "../utils/color/core.js";
import { Line, type LineOptions } from "./geometry/line.js";
import { Polygon } from "./geometry/polygram.js";
import { BackgroundRectangle } from "./geometry/shape_matchers.js";
import { get_vectorized_mobject_class } from "./utils.js";

export type TableData = Iterable<Iterable<unknown>>;

export type TableElementFactory = (
  item: unknown,
  config?: Record<string, unknown>,
) => VMobject;

function texNotPorted(name: string): TableElementFactory {
  return () => {
    throw new Error(`TODO: text/tex not ported yet (${name})`);
  };
}

export interface TableOptions {
  row_labels?: Iterable<VMobject> | null;
  col_labels?: Iterable<VMobject> | null;
  top_left_entry?: VMobject | null;
  v_buff?: number;
  h_buff?: number;
  include_outer_lines?: boolean;
  include_inner_lines?: boolean;
  add_background_rectangles_to_entries?: boolean;
  entries_background_color?: ParsableManimColor;
  include_background_rectangle?: boolean;
  background_rectangle_color?: ParsableManimColor;
  element_to_mobject?: TableElementFactory;
  element_to_mobject_config?: Record<string, unknown>;
  arrange_in_grid_config?: Record<string, unknown>;
  line_config?: Record<string, unknown>;
  [key: string]: unknown;
}

function isTableOptionsBag(value: TableData | TableOptions): value is TableOptions {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== "function"
  );
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

export class Table extends VGroup {
  row_labels: VMobject[] | null = null;
  col_labels: VMobject[] | null = null;
  top_left_entry: VMobject | null = null;
  row_dim = 0;
  col_dim = 0;
  v_buff: number;
  h_buff: number;
  include_outer_lines = false;
  include_inner_lines = true;
  add_background_rectangles_to_entries = false;
  entries_background_color: ManimColor;
  include_background_rectangle = false;
  background_rectangle_color: ManimColor;
  element_to_mobject: TableElementFactory;
  element_to_mobject_config: Record<string, unknown> = {};
  arrange_in_grid_config: Record<string, unknown> = {};
  line_config: Record<string, unknown> = {};
  mob_table: VMobject[][] = [];
  elements!: VGroup;
  elements_without_labels!: VGroup;
  horizontal_lines!: VGroup;
  vertical_lines!: VGroup;

  constructor(tableOrOptions: TableData | TableOptions = [], options: TableOptions = {}) {
    super();
    // NOTE: mirrors Python `Table(table, row_labels=None, ..., **kwargs)`
    // (matrix-style `(data, options)` call convention, same as Matrix).
    // Back-compat: `new Table()` / `new Table({h_buff, v_buff})` (the
    // pre-existing scaffold signature) keep working — a trailing plain
    // object is treated as the options bag and an absent/empty table
    // builds an empty scaffold instead of raising (Python raises
    // IndexError on `Table([])`).
    let table: TableData;
    let opts: TableOptions;
    if (isTableOptionsBag(tableOrOptions)) {
      table = [];
      opts = { ...tableOrOptions, ...options };
    } else {
      table = tableOrOptions;
      opts = options;
    }
    const {
      row_labels = null,
      col_labels = null,
      top_left_entry = null,
      v_buff = 0.8,
      h_buff = 1.3,
      include_outer_lines = false,
      include_inner_lines = true,
      add_background_rectangles_to_entries = false,
      entries_background_color = BLACK,
      include_background_rectangle = false,
      background_rectangle_color = BLACK,
      element_to_mobject = texNotPorted("Table element_to_mobject defaults to Paragraph"),
      element_to_mobject_config = {},
      arrange_in_grid_config = {},
      line_config = {},
      ..._kwargs
    } = opts;
    void _kwargs;
    const rowLabelData = row_labels === null || row_labels === undefined ? [] : [...row_labels];
    const colLabelData = col_labels === null || col_labels === undefined ? [] : [...col_labels];
    this.row_labels = rowLabelData.length > 0 ? rowLabelData : null;
    this.col_labels = colLabelData.length > 0 ? colLabelData : null;
    this.top_left_entry = top_left_entry ?? null;
    this.v_buff = v_buff;
    this.h_buff = h_buff;
    this.include_outer_lines = include_outer_lines;
    this.include_inner_lines = include_inner_lines;
    this.add_background_rectangles_to_entries = add_background_rectangles_to_entries;
    this.entries_background_color = new ManimColor(entries_background_color);
    this.include_background_rectangle = include_background_rectangle;
    this.background_rectangle_color = new ManimColor(background_rectangle_color);
    this.element_to_mobject = element_to_mobject;
    this.element_to_mobject_config = element_to_mobject_config;
    this.arrange_in_grid_config = arrange_in_grid_config;
    this.line_config = line_config;

    const tableData: unknown[][] = [...table].map((row) => [...row]);
    if (tableData.length === 0 || (tableData[0] ?? []).length === 0) {
      // Empty scaffold (see constructor NOTE above).
      this.row_dim = 0;
      this.col_dim = 0;
      this.elements_without_labels = new VGroup();
      this.elements = new VGroup();
      this.mob_table = [];
      this.add(this.elements);
      this.center();
      this.horizontal_lines = new VGroup();
      this.vertical_lines = new VGroup();
      return;
    }
    this.row_dim = tableData.length;
    this.col_dim = (tableData[0] as unknown[]).length;

    for (const row of tableData) {
      if (row.length !== this.col_dim) {
        throw valueError("Not all rows in table have the same length.");
      }
    }

    let mobTable: VMobject[][] = this._table_to_mob_table(tableData);
    this.elements_without_labels = new VGroup(...mobTable.flat());
    mobTable = this._add_labels(mobTable);
    this._organize_mob_table(mobTable);
    this.elements = new VGroup(...mobTable.flat());

    if ((this.elements.__getitem__(0) as VMobject).get_all_points().size === 0) {
      this.elements.remove(this.elements.__getitem__(0) as VMobject);
    }

    this.add(this.elements);
    this.center();
    this.mob_table = mobTable;
    this._add_horizontal_lines();
    this._add_vertical_lines();
    if (this.add_background_rectangles_to_entries) {
      this.add_background_to_entries(this.entries_background_color);
    }
    if (this.include_background_rectangle) {
      this._attach_background_rectangle(this, this.background_rectangle_color);
    }
  }

  _table_to_mob_table(table: unknown[][]): VMobject[][] {
    return table.map((row) =>
      row.map((item) => this.element_to_mobject(item, this.element_to_mobject_config)),
    );
  }

  _organize_mob_table(table: VMobject[][]): VGroup {
    const helpTable = new VGroup();
    for (let i = 0; i < table.length; i += 1) {
      const row = table[i] as VMobject[];
      for (let j = 0; j < row.length; j += 1) {
        helpTable.add((table[i] as VMobject[])[j] as VMobject);
      }
    }
    // Mirrors `help_table.arrange_in_grid(rows=..., cols=...,
    // buff=(h_buff, v_buff), **arrange_in_grid_config)`.
    const cfg = (this.arrange_in_grid_config ?? {}) as {
      rows?: number;
      cols?: number;
      buff?: number | [number, number];
      cell_alignment?: readonly number[];
      row_alignments?: string | null;
      col_alignments?: string | null;
      row_heights?: Array<number | null>;
      col_widths?: Array<number | null>;
      flow_order?: string;
    };
    helpTable.arrange_in_grid(
      cfg.rows ?? table.length,
      cfg.cols ?? ((table[0] as VMobject[] | undefined)?.length ?? 0),
      cfg.buff ?? [this.h_buff, this.v_buff],
      cfg.cell_alignment ?? ORIGIN,
      cfg.row_alignments ?? null,
      cfg.col_alignments ?? null,
      cfg.row_heights ?? null,
      cfg.col_widths ?? null,
      cfg.flow_order ?? "rd",
    );
    return helpTable;
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

  _add_horizontal_lines(): this {
    const anchorLeft = (this.get_left()[0] ?? 0) - 0.5 * this.h_buff;
    const anchorRight = (this.get_right()[0] ?? 0) + 0.5 * this.h_buff;
    const lineGroup = new VGroup();
    const lineOptions = () => ({ ...(this.line_config as Record<string, unknown>) }) as LineOptions;
    if (this.include_outer_lines) {
      let anchor = ((this.get_rows().__getitem__(0) as VGroup).get_top()[1] ?? 0) + 0.5 * this.v_buff;
      let line = new Line([anchorLeft, anchor, 0], [anchorRight, anchor, 0], lineOptions());
      lineGroup.add(line);
      this.add(line);
      anchor = ((this.get_rows().__getitem__(-1) as VGroup).get_bottom()[1] ?? 0) - 0.5 * this.v_buff;
      line = new Line([anchorLeft, anchor, 0], [anchorRight, anchor, 0], lineOptions());
      lineGroup.add(line);
      this.add(line);
    }
    if (this.include_inner_lines) {
      const rows = this.get_rows();
      for (let k = 0; k < this.mob_table.length - 1; k += 1) {
        const anchor = ((rows.__getitem__(k + 1) as VGroup).get_top()[1] ?? 0) + 0.5 * (
          ((rows.__getitem__(k) as VGroup).get_bottom()[1] ?? 0) -
          ((rows.__getitem__(k + 1) as VGroup).get_top()[1] ?? 0)
        );
        const line = new Line([anchorLeft, anchor, 0], [anchorRight, anchor, 0], lineOptions());
        lineGroup.add(line);
        this.add(line);
      }
    }
    this.horizontal_lines = lineGroup;
    return this;
  }

  _add_vertical_lines(): this {
    const rows = this.get_rows();
    const anchorTop = (rows.get_top()[1] ?? 0) + 0.5 * this.v_buff;
    const anchorBottom = (rows.get_bottom()[1] ?? 0) - 0.5 * this.v_buff;
    const lineGroup = new VGroup();
    const lineOptions = () => ({ ...(this.line_config as Record<string, unknown>) }) as LineOptions;
    if (this.include_outer_lines) {
      const columns = this.get_columns();
      let anchor = ((columns.__getitem__(0) as VGroup).get_left()[0] ?? 0) - 0.5 * this.h_buff;
      let line = new Line([anchor, anchorTop, 0], [anchor, anchorBottom, 0], lineOptions());
      lineGroup.add(line);
      this.add(line);
      anchor = ((columns.__getitem__(-1) as VGroup).get_right()[0] ?? 0) + 0.5 * this.h_buff;
      line = new Line([anchor, anchorTop, 0], [anchor, anchorBottom, 0], lineOptions());
      lineGroup.add(line);
      this.add(line);
    }
    if (this.include_inner_lines) {
      const columns = this.get_columns();
      for (let k = 0; k < ((this.mob_table[0] as VMobject[] | undefined)?.length ?? 0) - 1; k += 1) {
        const anchor = ((columns.__getitem__(k + 1) as VGroup).get_left()[0] ?? 0) + 0.5 * (
          ((columns.__getitem__(k) as VGroup).get_right()[0] ?? 0) -
          ((columns.__getitem__(k + 1) as VGroup).get_left()[0] ?? 0)
        );
        const line = new Line([anchor, anchorBottom, 0], [anchor, anchorTop, 0], lineOptions());
        lineGroup.add(line);
        this.add(line);
      }
    }
    this.vertical_lines = lineGroup;
    return this;
  }

  get_horizontal_lines(): VGroup {
    return this.horizontal_lines;
  }

  get_vertical_lines(): VGroup {
    return this.vertical_lines;
  }

  get_columns(): VGroup {
    const nCols = (this.mob_table[0] as VMobject[] | undefined)?.length ?? 0;
    return new VGroup(
      ...Array.from({ length: nCols }, (_, i) =>
        new VGroup(...this.mob_table.map((row) => (row[i] as VMobject))),
      ),
    );
  }

  get_rows(): VGroup {
    return new VGroup(...this.mob_table.map((row) => new VGroup(...row)));
  }

  set_column_colors(...colors: Array<ParsableManimColor | ParsableManimColor[]>): this {
    const columns = this.get_columns();
    const count = Math.min(colors.length, columns.submobjects.length);
    for (let index = 0; index < count; index += 1) {
      (columns.submobjects[index] as VMobject).set_color(colors[index] as ParsableManimColor);
    }
    return this;
  }

  set_row_colors(...colors: Array<ParsableManimColor | ParsableManimColor[]>): this {
    const rows = this.get_rows();
    const count = Math.min(colors.length, rows.submobjects.length);
    for (let index = 0; index < count; index += 1) {
      (rows.submobjects[index] as VMobject).set_color(colors[index] as ParsableManimColor);
    }
    return this;
  }

  get_entries(pos: readonly number[] | null = null): VMobject | VGroup {
    if (pos !== null && pos !== undefined) {
      const nCols = (this.mob_table[0] as VMobject[] | undefined)?.length ?? 0;
      let index: number;
      if (
        this.row_labels !== null &&
        this.col_labels !== null &&
        this.top_left_entry === null
      ) {
        index = nCols * ((pos[0] ?? 0) - 1) + ((pos[1] ?? 0) - 1) - 1;
      } else {
        index = nCols * ((pos[0] ?? 0) - 1) + ((pos[1] ?? 0) - 1);
      }
      return this.elements.__getitem__(index) as VMobject;
    }
    return this.elements;
  }

  get_entries_without_labels(pos: readonly number[] | null = null): VMobject | VGroup {
    if (pos !== null && pos !== undefined) {
      const index = this.col_dim * ((pos[0] ?? 0) - 1) + ((pos[1] ?? 0) - 1);
      return this.elements_without_labels.__getitem__(index) as VMobject;
    }
    return this.elements_without_labels;
  }

  get_row_labels(): VGroup {
    if (this.row_labels !== null && this.row_labels !== undefined) {
      return new VGroup(...this.row_labels);
    }
    return new VGroup();
  }

  get_col_labels(): VGroup {
    if (this.col_labels !== null && this.col_labels !== undefined) {
      return new VGroup(...this.col_labels);
    }
    return new VGroup();
  }

  get_labels(): VGroup {
    const labelGroup = new VGroup();
    if (this.top_left_entry !== null && this.top_left_entry !== undefined) {
      labelGroup.add(this.top_left_entry);
    }
    for (const label of [this.col_labels, this.row_labels]) {
      if (label !== null && label !== undefined) {
        labelGroup.add(...label);
      }
    }
    return labelGroup;
  }

  /** Direct BackgroundRectangle construction (mirrors what
   *  ``VMobject.add_background_rectangle`` does in Python). This avoids the
   *  global factory hook, which stays unwired so the "TODO: ... not ported
   *  yet" behavior asserted elsewhere is preserved. */
  private _attach_background_rectangle(target: VMobject, color: ParsableManimColor): BackgroundRectangle {
    const rect = new BackgroundRectangle(target, { color: new ManimColor(color) });
    target.background_rectangle = rect;
    target.add_to_back(rect);
    return rect;
  }

  add_background_to_entries(color: ParsableManimColor = BLACK): this {
    for (const mob of this.elements.submobjects) {
      this._attach_background_rectangle(mob as VMobject, color);
    }
    return this;
  }

  get_cell(pos: readonly number[] = [1, 1], kwargs: Record<string, unknown> = {}): Polygon {
    const row = this.get_rows().__getitem__((pos[0] ?? 1) - 1) as VGroup;
    const col = this.get_columns().__getitem__((pos[1] ?? 1) - 1) as VGroup;
    const edgeUL = [
      (col.get_left()[0] ?? 0) - this.h_buff / 2,
      (row.get_top()[1] ?? 0) + this.v_buff / 2,
      0,
    ];
    const edgeUR = [
      (col.get_right()[0] ?? 0) + this.h_buff / 2,
      (row.get_top()[1] ?? 0) + this.v_buff / 2,
      0,
    ];
    const edgeDL = [
      (col.get_left()[0] ?? 0) - this.h_buff / 2,
      (row.get_bottom()[1] ?? 0) - this.v_buff / 2,
      0,
    ];
    const edgeDR = [
      (col.get_right()[0] ?? 0) + this.h_buff / 2,
      (row.get_bottom()[1] ?? 0) - this.v_buff / 2,
      0,
    ];
    return new Polygon(edgeUL, edgeUR, edgeDR, edgeDL, kwargs);
  }

  get_highlighted_cell(
    pos: readonly number[] = [1, 1],
    color: ParsableManimColor = "PURE_YELLOW",
    kwargs: Record<string, unknown> = {},
  ): BackgroundRectangle {
    const cell = this.get_cell(pos);
    return new BackgroundRectangle(cell, { color: new ManimColor(color), ...kwargs });
  }

  add_highlighted_cell(
    pos: readonly number[] = [1, 1],
    color: ParsableManimColor = "PURE_YELLOW",
    kwargs: Record<string, unknown> = {},
  ): this {
    const bgCell = this.get_highlighted_cell(pos, color, kwargs);
    this.add_to_back(bgCell);
    const entry = this.get_entries(pos) as VMobject;
    entry.background_rectangle = bgCell;
    return this;
  }

  create(
    lag_ratio = 1,
    line_animation: AnimationClass = Create,
    label_animation: AnimationClass = Write,
    element_animation: AnimationClass = Create,
    entry_animation: AnimationClass = FadeIn,
    kwargs: Record<string, unknown> = {},
  ): AnimationGroup {
    const animations: Animation[] = [
      new line_animation(new VGroup(this.vertical_lines, this.horizontal_lines), kwargs),
      new element_animation(this.elements_without_labels.set_z_index(2), kwargs),
    ];

    if (this.get_labels().submobjects.length > 0) {
      animations.push(new label_animation(this.get_labels(), kwargs));
    }

    if ((this.get_entries() as VGroup).submobjects.length > 0) {
      for (const entry of this.elements_without_labels.submobjects) {
        const background = (entry as VMobject).background_rectangle;
        // Mirrors `try: ... entry.background_rectangle except AttributeError:
        // continue` — entries without a background rectangle are skipped.
        if (background !== null && background !== undefined) {
          animations.push(new entry_animation(background, kwargs));
        }
      }
    }

    return new AnimationGroup(...animations, { lag_ratio });
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

export interface MathTableOptions extends TableOptions {}

export class MathTable extends Table {
  constructor(table: TableData | TableOptions = [], options: MathTableOptions = {}) {
    const {
      element_to_mobject = texNotPorted("MathTable element_to_mobject defaults to MathTex"),
      ...rest
    } = options;
    super(table, { element_to_mobject, ...rest });
  }
}

export interface MobjectTableOptions extends TableOptions {}

export class MobjectTable extends Table {
  constructor(table: TableData | TableOptions = [], options: MobjectTableOptions = {}) {
    const {
      element_to_mobject = ((m: unknown) => m as VMobject) as TableElementFactory,
      ...rest
    } = options;
    super(table, { element_to_mobject, ...rest });
  }
}

export interface IntegerTableOptions extends TableOptions {}

export class IntegerTable extends Table {
  constructor(table: TableData | TableOptions = [], options: IntegerTableOptions = {}) {
    const {
      element_to_mobject = texNotPorted("IntegerTable element_to_mobject defaults to Integer"),
      ...rest
    } = options;
    super(table, { element_to_mobject, ...rest });
  }
}

export interface DecimalTableOptions extends TableOptions {}

export class DecimalTable extends Table {
  constructor(table: TableData | TableOptions = [], options: DecimalTableOptions = {}) {
    const {
      element_to_mobject = texNotPorted("DecimalTable element_to_mobject defaults to DecimalNumber"),
      element_to_mobject_config = { num_decimal_places: 1 },
      ...rest
    } = options;
    super(table, { element_to_mobject, element_to_mobject_config, ...rest });
  }
}
