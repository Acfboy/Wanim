import { config } from "../../_config/utils.js";
import { SMALL_BUFF } from "../../constants.js";
import { BLACK, type ParsableManimColor } from "../../utils/color/core.js";
import { Group, Mobject } from "../mobject.js";
import { singleOpacity, type VMobject } from "../types/vectorized_mobject.js";
import { RoundedRectangle, type RoundedRectangleOptions } from "./polygram.js";

type VMobjectStyleOptions = Parameters<VMobject["set_style"]>[0];

export interface SurroundingRectangleOptions extends RoundedRectangleOptions {
  buff?: number | [number, number];
}

function isOptionsObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (value instanceof Mobject) return false;
  return true;
}

export class SurroundingRectangle extends RoundedRectangle {
  buff: number | [number, number];

  constructor(...args: unknown[]) {
    let mobjects: Mobject[] = [];
    let options: SurroundingRectangleOptions = {};
    if (args.length > 0 && isOptionsObject(args[args.length - 1])) {
      options = args[args.length - 1] as SurroundingRectangleOptions;
      mobjects = args.slice(0, -1) as Mobject[];
    } else {
      mobjects = args as Mobject[];
    }
    const { color = "PURE_YELLOW", buff = SMALL_BUFF, corner_radius = 0.0, ...rest } = options;
    for (const mob of mobjects) {
      if (!(mob instanceof Mobject)) {
        throw new TypeError("Expected all inputs for parameter mobjects to be a Mobjects");
      }
    }
    let buff_x: number;
    let buff_y: number;
    if (Array.isArray(buff)) {
      buff_x = buff[0] ?? 0;
      buff_y = buff[1] ?? 0;
    } else {
      buff_x = buff_y = buff;
    }
    const group = new Group(...mobjects);
    super({
      color,
      width: group.width + 2 * buff_x,
      height: group.height + 2 * buff_y,
      corner_radius,
      ...rest,
    });
    this.buff = buff;
    this.move_to(group);
  }
}

export interface BackgroundRectangleOptions extends SurroundingRectangleOptions {
  stroke_width?: number;
  stroke_opacity?: number;
  fill_opacity?: number;
}

export class BackgroundRectangle extends SurroundingRectangle {
  original_fill_opacity: number;

  constructor(...args: unknown[]) {
    let mobjects: Mobject[] = [];
    let options: BackgroundRectangleOptions = {};
    if (args.length > 0 && isOptionsObject(args[args.length - 1])) {
      options = args[args.length - 1] as BackgroundRectangleOptions;
      mobjects = args.slice(0, -1) as Mobject[];
    } else {
      mobjects = args as Mobject[];
    }
    const {
      color = null,
      stroke_width = 0,
      stroke_opacity = 0,
      fill_opacity = 0.75,
      buff = 0,
      ...rest
    } = options;
    super(...mobjects, {
      color: (color ?? config.background_color) as ParsableManimColor,
      stroke_width,
      stroke_opacity,
      fill_opacity,
      buff,
      ...rest,
    });
    this.original_fill_opacity = this.fill_opacity;
  }

  pointwise_become_partial(mobject: Mobject, a: unknown, b: number): this {
    void mobject;
    void a;
    this.set_fill(undefined, b * this.original_fill_opacity);
    return this;
  }

  override set_style(
    fill_opacity: number | VMobjectStyleOptions = 0.75,
    kwargs: Record<string, unknown> = {},
  ): this {
    // Unchangeable style, except for fill_opacity.
    // All other style arguments are ignored.
    const normalized: VMobjectStyleOptions = typeof fill_opacity === "number"
      ? { fill_opacity }
      : (fill_opacity ?? {});
    const { fill_opacity: opacity = 0.75, ...rest } = normalized;
    const extra = { ...rest, ...kwargs };
    super.set_style({
      stroke_color: BLACK,
      stroke_width: 0,
      fill_color: BLACK,
      fill_opacity: singleOpacity(opacity) ?? 0.75,
    });
    if (Object.keys(extra).length > 0) {
      console.info(`Argument ${JSON.stringify(extra)} is ignored in BackgroundRectangle.set_style.`);
    }
    return this;
  }
}
