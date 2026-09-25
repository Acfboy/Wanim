import { array, type NDArray } from "numpy-ts";
import { DEFAULT_STROKE_WIDTH } from "../../constants.js";
import { Mobject, type MobjectOptions } from "../mobject.js";

export interface PMobjectOptions extends MobjectOptions {
  stroke_width?: number;
}

export class PMobject extends Mobject {
  stroke_width: number;
  rgbas: NDArray;

  constructor(options: PMobjectOptions = {}) {
    const { stroke_width = DEFAULT_STROKE_WIDTH, ...rest } = options;
    super(rest);
    this.stroke_width = stroke_width;
    this.rgbas = array([]).reshape(0, 4) as NDArray;
    this.reset_points();
  }

  reset_points(): this {
    this.rgbas = array([]).reshape(0, 4) as NDArray;
    this.points = array([]).reshape(0, 3) as NDArray;
    return this;
  }
}
