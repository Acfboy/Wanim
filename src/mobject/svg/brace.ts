import type { NDArray } from "numpy-ts";
import {
  DEFAULT_FONT_SIZE,
  DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
  DOWN,
  LEFT,
  ORIGIN,
  OUT,
  RIGHT,
  UP,
} from "../../constants.js";
import { BLACK, type ParsableManimColor } from "../../utils/color/core.js";
import { addPoints, pointAt, pointRows, subtractPoints } from "../../utils/arrays.js";
// NOTE (import order matters at runtime): mobject-land modules must evaluate
// before the animation modules below, because mobject.js and animation.js
// import each other and only the mobject-first direction avoids a TDZ error
// (same pre-existing cycle as documented in ../table.ts).
import { Group, Mobject } from "../mobject.js";
import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import type { Animation } from "../../animation/animation.js";
import { AnimationGroup } from "../../animation/composition.js";
import { FadeIn } from "../../animation/fading.js";
import { GrowFromCenter } from "../../animation/growing.js";
import { Arc } from "../geometry/arc.js";
import { Line } from "../geometry/line.js";
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

export class BraceBetweenPoints extends Brace {
  constructor(
    point_1: readonly number[] | NDArray,
    point_2: readonly number[] | NDArray,
    direction: readonly number[] | NDArray = [...ORIGIN],
    options: BraceOptions = {},
  ) {
    const { direction: directionOption, ...rest } = options;
    let resolved_direction = [...(pointRows(directionOption ?? direction, 3)[0] ?? [...ORIGIN])];
    // NOTE: mirrors `if all(direction == ORIGIN)`: every component equals.
    if (resolved_direction.every((value, index) => value === (ORIGIN[index] ?? 0))) {
      const p1 = pointRows(point_1, 3)[0] ?? [0, 0, 0];
      const p2 = pointRows(point_2, 3)[0] ?? [0, 0, 0];
      const line_vector = subtractPoints(p2, p1, 3);
      resolved_direction = [line_vector[1] ?? 0, -(line_vector[0] ?? 0), 0];
    }
    super(new Line(point_1, point_2), { direction: resolved_direction, ...rest });
  }
}

export interface ArcBraceOptions extends BraceOptions {}

export class ArcBrace extends Brace {
  constructor(
    arc: Arc | null = null,
    direction: readonly number[] | NDArray = [...RIGHT],
    options: ArcBraceOptions = {},
  ) {
    const { direction: directionOption, ...rest } = options;
    const resolved_direction = directionOption ?? direction;
    const resolved_arc = arc ?? new Arc(1, -1, 2);
    const arc_end_angle = resolved_arc.start_angle + resolved_arc.angle;
    const line = new Line(
      [0, resolved_arc.start_angle, 0],
      [0, arc_end_angle, 0],
    );
    const scale_shift = (RIGHT as readonly number[]).map(
      (value) => value * Math.log(resolved_arc.radius),
    );
    if (resolved_arc.radius >= 1) {
      line.scale(resolved_arc.radius, [...ORIGIN]);
      super(line, { direction: resolved_direction, ...rest });
      this.scale(1 / resolved_arc.radius, [...ORIGIN]);
    } else {
      super(line, { direction: resolved_direction, ...rest });
    }
    if (resolved_arc.radius >= 0.3) {
      this.shift(scale_shift);
    } else {
      this.shift((RIGHT as readonly number[]).map((value) => value * Math.log(0.3)));
    }
    // NOTE: mirrors `self.apply_complex_function(np.exp)`: map z = x + iy
    // to exp(z) = e^x (cos y + i sin y), keeping z.
    this.apply_complex_function(({ re, im }: { re: number; im: number }) => {
      const magnitude = Math.exp(re);
      return { re: magnitude * Math.cos(im), im: magnitude * Math.sin(im) };
    });
    this.shift(resolved_arc.get_arc_center());
  }
}

export interface BraceLabelOptions extends VMobjectOptions {
  brace_direction?: readonly number[] | NDArray;
  label_constructor?: unknown;
  font_size?: number;
  buff?: number;
  brace_config?: BraceOptions | null;
}

function isBraceLabelKwargs(value: unknown): value is Record<string, unknown> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    !(value instanceof VMobject) &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}

export class BraceLabel extends VMobject {
  label_constructor: unknown;
  brace_direction: readonly number[] | NDArray;
  brace!: Brace;
  label!: VMobject;

  constructor(
    obj: Mobject,
    text: string | string[] | VMobject,
    brace_direction: readonly number[] | NDArray = [...DOWN],
    label_constructor: unknown = null,
    font_size = DEFAULT_FONT_SIZE,
    buff = 0.2,
    brace_config: BraceOptions | null = null,
    options: BraceLabelOptions = {},
  ) {
    // NOTE: Python's label_constructor defaults to MathTex, which needs the
    // text/LaTeX pipeline that is out of scope for this port; it is stored
    // opaquely (null until set) and only invoked when label mobjects must be
    // constructed (throwing a marked TODO for raw strings without one).
    const {
      brace_direction: directionOption,
      label_constructor: constructorOption,
      font_size: fontSizeOption,
      buff: buffOption,
      brace_config: braceConfigOption,
      ...rest
    } = options;
    super(rest);
    this.label_constructor = constructorOption ?? label_constructor;
    this.brace_direction = directionOption ?? brace_direction;
    const resolved_buff = buffOption ?? buff;
    const resolved_config = braceConfigOption ?? brace_config ?? {};
    const resolved_font_size = fontSizeOption ?? font_size;

    this.brace = new Brace(obj, {
      direction: [...pointRows(this.brace_direction, this.dim)[0] ?? [...DOWN]],
      buff: resolved_buff,
      ...resolved_config,
    });

    this.label = BraceLabel._build_label(
      text,
      this.label_constructor,
      resolved_font_size,
      {},
    );

    this.brace.put_at_tip(this.label);
    this.add(this.brace, this.label);
  }

  static _build_label(
    text: unknown,
    label_constructor: unknown,
    font_size: number | null,
    kwargs: Record<string, unknown>,
  ): VMobject {
    // NOTE: mirrors Python `BraceLabel.__init__`: a tuple/list of strings is
    // unpacked into the label constructor, otherwise the single text is
    // passed (plus font_size). VMobject inputs are returned as-is (mirroring
    // NumberLine._create_label_tex leniency); anything else needs the text
    // pipeline that is out of scope for this port.
    if (text instanceof VMobject) return text;
    if (typeof label_constructor === "function") {
      const ctor = label_constructor as (...args: unknown[]) => unknown;
      const args = Array.isArray(text) ? [...text] : [String(text)];
      const built = ctor(
        ...args,
        { ...(font_size !== null ? { font_size } : {}), ...kwargs },
      );
      if (built instanceof VMobject) return built;
      throw new Error(
        "TODO: text/tex not ported yet (BraceLabel label_constructor did not return a VMobject)",
      );
    }
    throw new Error(
      "TODO: text/tex not ported yet (BraceLabel label construction needs MathTex/Text)",
    );
  }

  creation_anim(
    // NOTE: `any[]` rest params keep arbitrary Animation subclasses
    // (FadeIn, GrowFromCenter, ...) assignable under strictFunctionTypes.
    label_anim: new (mobject: Mobject, ...args: any[]) => Animation = FadeIn,
    brace_anim: new (mobject: Mobject, ...args: any[]) => Animation = GrowFromCenter,
  ): AnimationGroup {
    return new AnimationGroup(
      new brace_anim(this.brace),
      new label_anim(this.label),
    );
  }

  shift_brace(obj: Mobject | Mobject[], kwargs: Record<string, unknown> = {}): this {
    let target: Mobject;
    if (Array.isArray(obj)) {
      target = new Group(...obj);
    } else {
      target = obj;
    }
    this.brace = new Brace(target, {
      direction: [...pointRows(this.brace_direction, this.dim)[0] ?? [...DOWN]],
      ...(kwargs as BraceOptions),
    });
    this.brace.put_at_tip(this.label);
    return this;
  }

  change_label(...args: unknown[]): this {
    let text: unknown[] = args;
    let kwargs: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (args.length > 0 && isBraceLabelKwargs(last)) {
      kwargs = last as Record<string, unknown>;
      text = args.slice(0, -1);
    }
    this.remove(this.label);
    this.label = BraceLabel._build_label(
      text.length === 1 ? text[0] : text,
      this.label_constructor,
      null,
      kwargs,
    );
    this.brace.put_at_tip(this.label);
    this.add(this.label);
    return this;
  }

  change_brace_label(obj: Mobject | Mobject[], ...args: unknown[]): this {
    let text: unknown[] = args;
    let kwargs: Record<string, unknown> = {};
    const last = args[args.length - 1];
    if (args.length > 0 && isBraceLabelKwargs(last)) {
      kwargs = last as Record<string, unknown>;
      text = args.slice(0, -1);
    }
    this.shift_brace(obj);
    this.change_label(...text, ...(Object.keys(kwargs).length > 0 ? [kwargs] : []));
    return this;
  }
}
