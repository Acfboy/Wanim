import { type NDArray } from "numpy-ts";
import { VGroup, VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { Axes } from "./coordinate_systems.js";
import { config } from "../../_config/utils.js";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { tuplify } from "../../utils/iterables.js";
import { DARK_GREY, LIGHT_GREY, type ParsableManimColor } from "../../utils/color/core.js";

export const EPSILON = 0.0001;

export interface SampleSpaceOptions extends VMobjectOptions {
  height?: number;
  width?: number;
  fill_color?: ParsableManimColor;
  fill_opacity?: number;
  stroke_width?: number;
  stroke_color?: ParsableManimColor;
  default_label_scale_val?: number;
}

// NOTE: Python's SampleSpace extends Rectangle (geometry/polygram). The
// Rectangle port is out of scope for this batch, so SampleSpace currently
// extends VMobject directly with matching width/height constructor params.
export class SampleSpace extends VMobject {
  default_label_scale_val: number;
  label: string | null = null;
  horizontal_parts?: VMobject & { braces?: VMobject; labels?: VMobject };
  vertical_parts?: VMobject & { braces?: VMobject; labels?: VMobject };

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
    super({ fill_color, fill_opacity, stroke_width, stroke_color, ...rest });
    void height;
    void width;
    this.default_label_scale_val = default_label_scale_val;
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

  add_braces_and_labels(): this {
    for (const attr of ["horizontal_parts", "vertical_parts"] as const) {
      const parts = this[attr] as (VMobject & { braces?: VMobject; labels?: VMobject }) | undefined;
      if (parts === undefined || parts === null) continue;
      for (const subattr of ["braces", "labels"] as const) {
        const sub = (parts as unknown as Record<string, unknown>)[subattr];
        if (sub !== undefined && sub !== null) this.add(sub as VMobject);
      }
    }
    return this;
  }

  addBracesAndLabels(): this {
    return this.add_braces_and_labels();
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
}

// NOTE: only the text-independent core of BarChart is ported here (range
// computation plus _update_colors). Building the bars/labels/axes numbers
// needs Rectangle (ported) but also Tex/DecimalNumber labels, which are out
// of scope (text/tex not ported yet); see .
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

    super(x_range, resolved_y_range, resolved_x_length, resolved_y_length);

    this.values = [...values];
    this.bar_names = bar_names === null ? null : [...bar_names];
    this.bar_colors = [...bar_colors];
    this.bar_width = bar_width;
    this.bar_fill_opacity = bar_fill_opacity;
    this.bar_stroke_width = bar_stroke_width;
  }

  _update_colors(): void {
    this.bars.set_color_by_gradient(...this.bar_colors);
  }
}
