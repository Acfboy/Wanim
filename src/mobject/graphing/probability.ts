import { type NDArray } from "numpy-ts";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
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
