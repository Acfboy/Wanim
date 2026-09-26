import type { NDArray } from "numpy-ts";
import {
  DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
  DOWN,
  LEFT,
  ORIGIN,
  OUT,
  RIGHT,
  UP,
} from "../../constants.js";
import { BLACK, type ParsableManimColor } from "../../utils/color/core.js";
import { addPoints, pointAt, pointRows } from "../../utils/arrays.js";
import { Mobject } from "../mobject.js";
import {
  VMobjectFromSVGPath,
  type VMobjectFromSVGPathOptions,
} from "./svg_mobject.js";

export interface BraceOptions extends VMobjectFromSVGPathOptions {
  direction?: readonly number[] | NDArray;
  buff?: number;
  sharpness?: number;
  stroke_width?: number;
  fill_opacity?: number;
  background_stroke_width?: number;
  background_stroke_color?: ParsableManimColor | null;
}

export class Brace extends VMobjectFromSVGPath {
  buff: number;
  _tip_point_index = 0;

  constructor(mobject: Mobject, options: BraceOptions = {}) {
    const {
      direction = [...DOWN],
      buff = 0.2,
      sharpness = 2,
      stroke_width = 0,
      fill_opacity = 1.0,
      background_stroke_width = 0,
      background_stroke_color = BLACK,
      ...rest
    } = options;
    // NOTE: Python assigns self.buff before super().__init__(); TS requires
    // super() first, so it is assigned on the field declaration above and
    // refreshed here before the geometry below needs it.
    const path_string_template =
      "m0.01216 0c-0.01152 0-0.01216 6.103e-4 -0.01216 0.01311v0.007762c0.06776 " +
      "0.122 0.1799 0.1455 0.2307 0.1455h{0}c0.03046 3.899e-4 0.07964 0.00449 " +
      "0.1246 0.02636 0.0537 0.02695 0.07418 0.05816 0.08648 0.07769 0.001562 " +
      "0.002538 0.004539 0.002563 0.01098 0.002563 0.006444-2e-8 0.009421-2.47e-" +
      "5 0.01098-0.002563 0.0123-0.01953 0.03278-0.05074 0.08648-0.07769 0.04491" +
      "-0.02187 0.09409-0.02597 0.1246-0.02636h{0}c0.05077 0 0.1629-0.02346 " +
      "0.2307-0.1455v-0.007762c-1.78e-6 -0.0125-6.365e-4 -0.01311-0.01216-0.0131" +
      "1-0.006444-3.919e-8 -0.009348 2.448e-5 -0.01091 0.002563-0.0123 0.01953-" +
      "0.03278 0.05074-0.08648 0.07769-0.04491 0.02187-0.09416 0.02597-0.1246 " +
      "0.02636h{1}c-0.04786 0-0.1502 0.02094-0.2185 0.1256-0.06833-0.1046-0.1706" +
      "-0.1256-0.2185-0.1256h{1}c-0.03046-3.899e-4 -0.07972-0.004491-0.1246-0.02" +
      "636-0.0537-0.02695-0.07418-0.05816-0.08648-0.07769-0.001562-0.002538-" +
      "0.004467-0.002563-0.01091-0.002563z";
    const default_min_width = 0.90552;

    const directionRow = pointRows(direction, 3)[0] ?? [0, -1, 0];
    // NOTE: mirrors `angle = -np.arctan2(*direction[:2]) + np.pi`. numpy's
    // arctan2(x1, x2) takes the x-coordinate FIRST here (the unpacked slice
    // is [x, y]), so the faithful JS form is atan2(x, y) — not atan2(y, x).
    const angle = -Math.atan2(directionRow[0] ?? 0, directionRow[1] ?? 0) + Math.PI;
    mobject.rotate(-angle, [...OUT], [...ORIGIN]);
    const left = mobject.get_corner(addPoints(DOWN, LEFT, 3));
    const right = mobject.get_corner(addPoints(DOWN, RIGHT, 3));
    const target_width = (right[0] ?? 0) - (left[0] ?? 0);
    const linear_section_length = Math.max(
      0,
      (target_width * sharpness - default_min_width) / 2,
    );

    const path_string = path_string_template
      .split("{0}")
      .join(String(linear_section_length))
      .split("{1}")
      .join(String(-linear_section_length));

    super(path_string, {
      stroke_width,
      fill_opacity,
      background_stroke_width,
      background_stroke_color,
      ...rest,
    });
    this.buff = buff;
    this.flip([...RIGHT]);
    const bottom = this.get_bottom();
    const rows = pointRows(this.points, this.dim);
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i] ?? [];
      const distance = Math.hypot(
        ...row.map((value, dim) => value - (bottom[dim] ?? 0)),
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    this._tip_point_index = bestIndex;
    this.stretch_to_fit_width(target_width);
    const upLeft = this.get_corner(addPoints(UP, LEFT, 3));
    this.shift([
      (left[0] ?? 0) - (upLeft[0] ?? 0) + this.buff * (DOWN[0] as number),
      (left[1] ?? 0) - (upLeft[1] ?? 0) + this.buff * (DOWN[1] as number),
      (left[2] ?? 0) - (upLeft[2] ?? 0) + this.buff * (DOWN[2] as number),
    ]);

    for (const mob of [mobject, this]) {
      mob.rotate(angle, [...OUT], [...ORIGIN]);
    }
  }

  put_at_tip(
    mob: Mobject,
    use_next_to = true,
    kwargs: { buff?: number; [key: string]: unknown } = {},
  ): this {
    if (use_next_to) {
      const direction = (this.get_direction() as number[]).map((value: number) => Math.round(value));
      if (kwargs.buff !== undefined) {
        mob.next_to(this.get_tip(), direction, kwargs.buff as number);
      } else {
        mob.next_to(this.get_tip(), direction);
      }
    } else {
      mob.move_to(this.get_tip());
      const buff = (kwargs.buff as number | undefined) ?? DEFAULT_MOBJECT_TO_MOBJECT_BUFFER;
      const shift_distance = mob.width / 2.0 + buff;
      const direction = this.get_direction() as number[];
      mob.shift(direction.map((value: number) => value * shift_distance));
    }
    return this;
  }

  get_text(..._text: string[]): never {
    throw new Error("TODO: text/tex not ported yet (Brace.get_text needs Tex)");
  }

  get_tex(..._tex: string[]): never {
    throw new Error("TODO: text/tex not ported yet (Brace.get_tex needs MathTex)");
  }

  get_tip(): number[] {
    return pointAt(this.points, this._tip_point_index, this.dim);
  }

  get_direction(): any {
    // NOTE: Python's Brace shadows VMobject.get_direction (which returns the
    // winding direction "CW"/"CCW") with an incompatible vector return type;
    // the `any` return type carries that shadowing into TS.
    const tip = this.get_tip();
    const center = this.get_center();
    const vect = tip.map((value, dim) => value - (center[dim] ?? 0));
    const norm = Math.hypot(...vect);
    return vect.map((value) => value / norm);
  }
}
