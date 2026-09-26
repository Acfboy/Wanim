import { config } from "../_config/utils.js";
import { Rectangle, type RectangleOptions } from "./geometry/polygram.js";

export interface ScreenRectangleOptions extends RectangleOptions {
  aspect_ratio?: number;
  height?: number;
}

export class ScreenRectangle extends Rectangle {
  constructor(aspect_ratio = 16.0 / 9.0, height = 4, options: ScreenRectangleOptions = {}) {
    const {
      aspect_ratio: aspectRatioOption,
      height: heightOption,
      ...rest
    } = options;
    const resolved_ratio = aspectRatioOption ?? aspect_ratio;
    const resolved_height = heightOption ?? height;
    super({ width: resolved_ratio * resolved_height, height: resolved_height, ...rest });
  }

  get aspect_ratio(): number {
    return this.width / this.height;
  }

  set aspect_ratio(value: number) {
    this.stretch_to_fit_width(value * this.height);
  }
}

export interface FullScreenRectangleOptions extends RectangleOptions {
  aspect_ratio?: number;
}

export class FullScreenRectangle extends ScreenRectangle {
  constructor(options: FullScreenRectangleOptions = {}) {
    // NOTE: mirrors Python `super().__init__(**kwargs); self.height =
    // config["frame_height"]` — kwargs (e.g. aspect_ratio) are forwarded to
    // ScreenRectangle, then height is scaled to the configured frame height
    // (uniform scale, so the aspect ratio is preserved).
    const { aspect_ratio = 16.0 / 9.0, height: _ignoredHeight = 4, ...rest } = options;
    void _ignoredHeight;
    super(aspect_ratio, 4, rest);
    this.height = Number(config.frame_height);
  }
}
