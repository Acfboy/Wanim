import { type NDArray } from "numpy-ts";
import { Mobject, type MobjectSlice } from "../mobject.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { Rectangle, type RectangleOptions } from "../geometry/polygram.js";
import { Brace, type BraceOptions } from "../svg/brace.js";
import { Axes } from "./coordinate_systems.js";
import { config } from "../../_config/utils.js";
import { DOWN, LEFT, MED_SMALL_BUFF, RIGHT, SMALL_BUFF, UP } from "../../constants.js";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { tuplify } from "../../utils/iterables.js";
import { BLUE_E, DARK_GREY, GREEN_E, LIGHT_GREY, MAROON_B, YELLOW, color_gradient, type ParsableManimColor } from "../../utils/color/core.js";

export const EPSILON = 0.0001;

export interface SampleSpaceOptions extends RectangleOptions {
  default_label_scale_val?: number;
}

/** A subdivision group produced by divide_horizontally/divide_vertically,
 *  carrying the braces/labels built for it (mirrors Python setting
 *  `parts.braces`, `parts.labels` and `parts.label_kwargs`). */
export type SampleSpaceDivision = VGroup & {
  braces?: VGroup;
  labels?: VGroup;
  label_kwargs?: {
    labels: VGroup;
    direction: readonly number[] | NDArray;
    buff: number;
  };
};

export class SampleSpace extends Rectangle {
  default_label_scale_val: number;
  label: string | null = null;
  title?: VMobject;
  horizontal_parts?: SampleSpaceDivision;
  vertical_parts?: SampleSpaceDivision;

  constructor(options: SampleSpaceOptions = {}) {
    const {
      height = 3,
      width = 3,
      fill_color = DARK_GREY,
      fill_opacity = 1,
      stroke_width = 0.5,
      stroke_color = LIGHT_GREY,
      default_label_scale_val = 1,
      ...rest
    } = options;
    super({ height, width, fill_color, fill_opacity, stroke_width, stroke_color, ...rest });
    this.default_label_scale_val = default_label_scale_val;
  }

  add_title(_title = "Sample space", _buff: number = MED_SMALL_BUFF): this {
    // NOTE: Python builds a `Tex` title here; text/tex is out of scope for
    // the entire port.
    throw new Error("TODO: text/tex not ported yet (SampleSpace.add_title needs Tex)");
  }

  add_label(label: string): this {
    this.label = label;
    return this;
  }

  complete_p_list(p_list: number | Iterable<number> | NDArray): number[] {
    const tuplified = isNDArray(p_list)
      ? pointRows(p_list, 1).map((row) => row[0] ?? 0)
      : (tuplify(p_list as Iterable<number>) as number[]);
    const new_p_list = [...tuplified];
    const remainder = 1.0 - new_p_list.reduce((sum, value) => sum + value, 0);
    if (Math.abs(remainder) > EPSILON) new_p_list.push(remainder);
    return new_p_list;
  }

  get_division_along_dimension(
    p_list: number | Iterable<number> | NDArray,
    dim: number,
    colors: Iterable<ParsableManimColor>,
    vect: readonly number[] | NDArray,
  ): VGroup {
    const p_list_complete = this.complete_p_list(p_list);
    const colors_in_gradient = color_gradient([...colors], p_list_complete.length);

    const vectRow = pointRows(vect, 3)[0] ?? [0, 0, 0];
    const negVect = vectRow.map((value) => -value);
    let last_point = this.get_edge_center(negVect);
    const parts = new VGroup();
    for (let index = 0; index < p_list_complete.length; index += 1) {
      const factor = p_list_complete[index] ?? 0;
      const color = colors_in_gradient[index] as ParsableManimColor;
      const part = new SampleSpace();
      part.set_fill(color, 1);
      part.replace(this, 0, true);
      part.stretch(factor, dim);
      part.move_to(last_point, negVect);
      last_point = part.get_edge_center(vectRow);
      parts.add(part);
    }
    return parts;
  }

  get_horizontal_division(
    p_list: number | Iterable<number> | NDArray,
    colors: Iterable<ParsableManimColor> = [GREEN_E, BLUE_E],
    vect: readonly number[] | NDArray = [...DOWN],
  ): VGroup {
    return this.get_division_along_dimension(p_list, 1, colors, vect);
  }

  get_vertical_division(
    p_list: number | Iterable<number> | NDArray,
    colors: Iterable<ParsableManimColor> = [MAROON_B, YELLOW],
    vect: readonly number[] | NDArray = [...RIGHT],
  ): VGroup {
    return this.get_division_along_dimension(p_list, 0, colors, vect);
  }

  divide_horizontally(
    p_list: number | Iterable<number> | NDArray,
    colors: Iterable<ParsableManimColor> = [GREEN_E, BLUE_E],
    vect: readonly number[] | NDArray = [...DOWN],
  ): this {
    this.horizontal_parts = this.get_horizontal_division(p_list, colors, vect) as SampleSpaceDivision;
    this.add(this.horizontal_parts);
    return this;
  }

  divide_vertically(
    p_list: number | Iterable<number> | NDArray,
    colors: Iterable<ParsableManimColor> = [MAROON_B, YELLOW],
    vect: readonly number[] | NDArray = [...RIGHT],
  ): this {
    this.vertical_parts = this.get_vertical_division(p_list, colors, vect) as SampleSpaceDivision;
    this.add(this.vertical_parts);
    return this;
  }

  get_subdivision_braces_and_labels(
    parts: VGroup,
    labels: Array<string | VMobject>,
    direction: readonly number[] | NDArray,
    buff = SMALL_BUFF,
    min_num_quads = 1,
  ): VGroup {
    const label_mobs = new VGroup();
    const braces = new VGroup();
    const count = Math.min(labels.length, parts.submobjects.length);
    for (let index = 0; index < count; index += 1) {
      const label = labels[index] as string | VMobject;
      const part = parts.submobjects[index] as VMobject;
      // NOTE: min_num_quads is accepted for signature compatibility and
      // forwarded; the TS Brace port does not implement it yet.
      const brace = new Brace(part, { direction, buff, min_num_quads } as BraceOptions);
      let label_mob: VMobject;
      if (label instanceof VMobject) {
        label_mob = label;
      } else {
        // NOTE: Python builds a `MathTex` label here; text/tex is out of
        // scope for the entire port.
        throw new Error("TODO: text/tex not ported yet (SampleSpace labels need MathTex)");
      }
      label_mob.next_to(brace, direction, buff);
      braces.add(brace);
      label_mobs.add(label_mob);
    }
    const division = parts as SampleSpaceDivision;
    division.braces = braces;
    division.labels = label_mobs;
    division.label_kwargs = {
      labels: label_mobs.copy() as VGroup,
      direction,
      buff,
    };
    return new VGroup(division.braces, division.labels);
  }

  get_side_braces_and_labels(
    labels: Array<string | VMobject>,
    direction: readonly number[] | NDArray = [...LEFT],
    buff = SMALL_BUFF,
    min_num_quads = 1,
  ): VGroup {
    if (this.horizontal_parts === undefined || this.horizontal_parts === null) {
      throw new Error("AssertionError: SampleSpace has no horizontal_parts (call divide_horizontally first)");
    }
    const parts: VGroup = this.horizontal_parts;
    return this.get_subdivision_braces_and_labels(parts, labels, direction, buff, min_num_quads);
  }

  get_top_braces_and_labels(
    labels: Array<string | VMobject>,
    buff = SMALL_BUFF,
    min_num_quads = 1,
  ): VGroup {
    if (this.vertical_parts === undefined || this.vertical_parts === null) {
      throw new Error("AssertionError: SampleSpace has no vertical_parts (call divide_vertically first)");
    }
    const parts: VGroup = this.vertical_parts;
    return this.get_subdivision_braces_and_labels(parts, labels, [...UP], buff, min_num_quads);
  }

  get_bottom_braces_and_labels(
    labels: Array<string | VMobject>,
    direction: readonly number[] | NDArray = [...DOWN],
    buff = SMALL_BUFF,
    min_num_quads = 1,
  ): VGroup {
    if (this.vertical_parts === undefined || this.vertical_parts === null) {
      throw new Error("AssertionError: SampleSpace has no vertical_parts (call divide_vertically first)");
    }
    const parts: VGroup = this.vertical_parts;
    return this.get_subdivision_braces_and_labels(parts, labels, direction, buff, min_num_quads);
  }

  add_braces_and_labels(): this {
    // NOTE: `in` mirrors Python's hasattr here: plain attribute reads go
    // through the Mobject proxy, which raises AttributeError (not undefined)
    // for attributes that were never assigned (e.g. `braces` on a division
    // that has none yet).
    for (const attr of ["horizontal_parts", "vertical_parts"] as const) {
      if (!(attr in this)) continue;
      const parts = this[attr] as SampleSpaceDivision | undefined | null;
      if (parts === undefined || parts === null) continue;
      for (const subattr of ["braces", "labels"] as const) {
        if (!(subattr in (parts as object))) continue;
        const sub = parts[subattr] as VGroup | undefined | null;
        if (sub !== undefined && sub !== null) this.add(sub);
      }
    }
    return this;
  }

  addBracesAndLabels(): this {
    return this.add_braces_and_labels();
  }

  override __getitem__(value: number | MobjectSlice): Mobject {
    if (this.horizontal_parts !== undefined && this.horizontal_parts !== null) {
      return this.horizontal_parts.__getitem__(value) as Mobject;
    } else if (this.vertical_parts !== undefined && this.vertical_parts !== null) {
      return this.vertical_parts.__getitem__(value) as Mobject;
    }
    return super.__getitem__(value) as Mobject;
  }
}

export interface BarChartOptions {
  bar_names?: string[] | null;
  y_range?: number[] | null;
  x_length?: number | null;
  y_length?: number | null;
  bar_colors?: ParsableManimColor[];
  bar_width?: number;
  bar_fill_opacity?: number;
  bar_stroke_width?: number;
  x_axis_config?: Record<string, unknown> | null;
  y_axis_config?: Record<string, unknown> | null;
  axis_config?: Record<string, unknown> | null;
  tips?: boolean;
}

// NOTE: the bar geometry (_create_bar/_add_bars/change_bar_values) is ported
// in full. The x-axis/bar labels and y-axis numbers need Tex/DecimalNumber
// (text/tex, out of scope for the entire port), so _add_x_axis_labels and
// get_bar_labels throw TODO errors and the y_axis.add_numbers() tail of
// __init__ binds once the NumberLine integration lands; the bars themselves
// cannot join the axes group yet because the TS Axes hierarchy is not
// Mobject-based (no add/add_to_back); see .
export class BarChart extends Axes {
  values: number[];
  bar_names: string[] | null;
  bar_colors: ParsableManimColor[];
  bar_width: number;
  bar_fill_opacity: number;
  bar_stroke_width: number;
  bars: VGroup = new VGroup();
  x_labels: VGroup | null = null;
  bar_labels: VGroup | null = null;

  constructor(values: number[] = [], options: BarChartOptions = {}) {
    const {
      bar_names = null,
      y_range = null,
      x_length = null,
      y_length = null,
      bar_colors = [
        "#003f5c",
        "#58508d",
        "#bc5090",
        "#ff6361",
        "#ffa600",
      ],
      bar_width = 0.6,
      bar_fill_opacity = 0.7,
      bar_stroke_width = 3,
      x_axis_config: passed_x_axis_config = null,
      y_axis_config: passed_y_axis_config = null,
      axis_config: passed_axis_config = null,
      tips = false,
    } = options;

    const resolved_y_length = y_length ?? config.frame_height - 4;
    const x_range = [0, values.length, 1];
    let resolved_y_range: number[];
    if (y_range === null) {
      const lo = values.length > 0 ? Math.min(0, Math.min(...values)) : 0;
      const hi = values.length > 0 ? Math.max(0, Math.max(...values)) : 0;
      const peak = values.length > 0 ? Math.max(...values) : 0;
      resolved_y_range = [lo, hi, Math.round((peak / resolved_y_length) * 100) / 100];
    } else if (y_range.length === 2) {
      const peak = values.length > 0 ? Math.max(...values) : 0;
      resolved_y_range = [...y_range, Math.round((peak / resolved_y_length) * 100) / 100];
    } else {
      resolved_y_range = [...y_range];
    }
    const resolved_x_length = x_length ?? Math.min(values.length, config.frame_width - 2);

    // NOTE: Python defaults x_axis_config to {"font_size": 24,
    // "label_constructor": Tex}; Tex is out of scope, so the constructor is
    // kept as an opaque placeholder until text lands.
    const x_axis_config: Record<string, unknown> = { font_size: 24, label_constructor: "Tex" };
    Axes._update_default_configs([x_axis_config], [passed_x_axis_config]);

    super(
      x_range,
      resolved_y_range,
      resolved_x_length,
      resolved_y_length,
      passed_axis_config,
      x_axis_config,
      passed_y_axis_config,
      tips,
    );

    this.values = [...values];
    this.bar_names = bar_names === null ? null : [...bar_names];
    this.bar_colors = [...bar_colors];
    this.bar_width = bar_width;
    this.bar_fill_opacity = bar_fill_opacity;
    this.bar_stroke_width = bar_stroke_width;

    this._add_bars();

    if (this.bar_names !== null && this.bar_names !== undefined) {
      this._add_x_axis_labels();
    }

    // NOTE: NumberLine.add_numbers builds DecimalNumber (text) labels and the
    // TS Axes hierarchy is not Mobject-based yet, so this binds once the
    // NumberLine integration lands.
    (this as unknown as { y_axis?: { add_numbers?: () => void } }).y_axis?.add_numbers?.();
  }

  _update_colors(): void {
    this.bars.set_color_by_gradient(...this.bar_colors);
  }

  _create_bar(bar_number: number, value: number): Rectangle {
    // bar measurements relative to the axis

    // distance from between the y-axis and the top of the bar
    const top_point = pointRows(this.c2p(0, value), 3)[0] ?? [0, 0, 0];
    const base_point = pointRows(this.c2p(0, 0), 3)[0] ?? [0, 0, 0];
    const bar_h = Math.abs((top_point[1] ?? 0) - (base_point[1] ?? 0));
    // width of the bar
    const width_point = pointRows(this.c2p(this.bar_width, 0), 3)[0] ?? [0, 0, 0];
    const bar_w = (width_point[0] ?? 0) - (base_point[0] ?? 0);

    const bar = new Rectangle({
      height: bar_h,
      width: bar_w,
      stroke_width: this.bar_stroke_width,
      fill_opacity: this.bar_fill_opacity,
    });

    const pos = value >= 0 ? [...UP] : [...DOWN];
    const anchor = pointRows(this.c2p(bar_number + 0.5, 0), 3)[0] ?? [0, 0, 0];
    bar.next_to(anchor, pos, 0);
    return bar;
  }

  _add_bars(): void {
    for (let i = 0; i < this.values.length; i += 1) {
      const tmp_bar = this._create_bar(i, this.values[i] ?? 0);
      this.bars.add(tmp_bar);
    }

    this._update_colors();
    // NOTE: Python calls self.add_to_back(self.bars); the Axes hierarchy is
    // Mobject-based (VGroup), so the bars are added behind the axes.
    this.add_to_back(this.bars);
  }

  _add_x_axis_labels(): void {
    // NOTE: labels are built with self.x_axis.label_constructor (Tex) and
    // positioned via NumberLine helpers; text/tex is out of scope.
    throw new Error("TODO: text/tex not ported yet (BarChart._add_x_axis_labels needs Tex labels)");
  }

  get_bar_labels(
    color: ParsableManimColor | null = null,
    font_size = 24,
    buff: number = MED_SMALL_BUFF,
    label_constructor: unknown = "Tex",
  ): VGroup {
    void color;
    void font_size;
    void buff;
    void label_constructor;
    // NOTE: labels are MathTex/Tex mobjects (text/tex, out of scope).
    throw new Error("TODO: text/tex not ported yet (BarChart.get_bar_labels needs Tex labels)");
  }

  change_bar_values(values: Iterable<number>, update_colors = true): this {
    const values_list = [...values];
    const bars = this.bars.submobjects as Rectangle[];
    const count = Math.min(bars.length, values_list.length);
    for (let i = 0; i < count; i += 1) {
      const bar = bars[i] as Rectangle;
      const value = values_list[i] ?? 0;
      const chart_val = this.values[i] ?? 0;

      let bar_lim: number[];
      let aligned_edge: readonly number[];
      if (chart_val > 0) {
        bar_lim = bar.get_bottom();
        aligned_edge = [...DOWN];
      } else {
        bar_lim = bar.get_top();
        aligned_edge = [...UP];
      }

      // check if the bar has height
      if (chart_val !== 0) {
        const quotient = value / chart_val;
        if (quotient < 0) {
          aligned_edge = chart_val > 0 ? [...UP] : [...DOWN];

          // if the bar is already positive, then we now want to move it
          // so that it is negative. So, we move the top edge of the bar
          // to the location of the previous bottom

          // if already negative, then we move the bottom edge of the bar
          // to the location of the previous top
        }
        bar.stretch_to_fit_height(Math.abs(quotient) * bar.height);
      } else {
        // create a new bar since the current one has a height of zero (doesn't exist)
        const temp_bar = this._create_bar(i, value);
        this.bars.remove(bar);
        this.bars.insert(i, temp_bar);
      }

      bar.move_to(bar_lim, aligned_edge);
    }

    if (update_colors) {
      this._update_colors();
    }

    this.values.splice(0, values_list.length, ...values_list);
    return this;
  }
}
