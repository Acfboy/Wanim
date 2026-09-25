import type { NDArray } from "numpy-ts";
import { config } from "../_config/utils.js";
import { ORIGIN } from "../constants.js";
import { ManimColor, WHITE } from "../utils/color/core.js";
import { Camera, type CameraOptions, type CaptureOptions } from "./camera.js";
import { ScreenRectangle } from "../mobject/frame.js";
import { Mobject } from "../mobject/mobject.js";

export interface MovingCameraOptions extends CameraOptions {
  fixed_dimension?: number;
  default_frame_stroke_color?: ManimColor;
  default_frame_stroke_width?: number;
}

export class MovingCamera extends Camera {
  frame: Mobject;
  fixed_dimension: number;
  default_frame_stroke_color: ManimColor;
  default_frame_stroke_width: number;
  // NOTE: written by the frame_width/frame_height setters while super()
  // runs (before the frame exists), so they must NOT be class fields:
  // useDefineForClassFields would re-define (wipe) them after super().
  private declare _pending_frame_width: number | undefined;
  private declare _pending_frame_height: number | undefined;
  private declare _pending_frame_center: readonly number[] | undefined;

  constructor(
    frame: Mobject | null = null,
    options: MovingCameraOptions = {},
  ) {
    const {
      fixed_dimension = 0,
      default_frame_stroke_color = WHITE,
      default_frame_stroke_width = 0,
      ...rest
    } = options;
    // NOTE: Python assigns self.frame BEFORE super().__init__() because the
    // base constructor writes self.frame_width/self.frame_height, which on
    // this subclass are frame-backed properties. TS requires super() first,
    // so during super() the frame is not yet present: the accessors below
    // stash those writes in _pending_* fields and replay them onto the
    // frame afterwards (same final state as Python).
    super(rest);
    this.fixed_dimension = fixed_dimension;
    this.default_frame_stroke_color = default_frame_stroke_color instanceof ManimColor
      ? default_frame_stroke_color
      : new ManimColor(default_frame_stroke_color);
    this.default_frame_stroke_width = default_frame_stroke_width;
    let resolved = frame;
    if (resolved === null || resolved === undefined) {
      resolved = new ScreenRectangle(16.0 / 9.0, Number(config.frame_height));
      (resolved as ScreenRectangle).set_stroke(
        this.default_frame_stroke_color,
        this.default_frame_stroke_width,
      );
    }
    this.frame = resolved;
    if (this._pending_frame_center !== undefined) {
      this.frame.move_to(this._pending_frame_center);
      this._pending_frame_center = undefined;
    }
    if (this._pending_frame_height !== undefined) {
      this.frame_height = this._pending_frame_height;
      this._pending_frame_height = undefined;
    }
    if (this._pending_frame_width !== undefined) {
      this.frame_width = this._pending_frame_width;
      this._pending_frame_width = undefined;
    }
    this.resize_frame_shape(this.fixed_dimension);
  }

  // TODO, make these work for a rotated frame
  override get frame_height(): number {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      return this._pending_frame_height ?? Number(config.frame_height);
    }
    return frame.height;
  }

  override set frame_height(value: number) {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      this._pending_frame_height = value;
      return;
    }
    frame.stretch_to_fit_height(value);
  }

  override get frame_width(): number {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      return this._pending_frame_width ?? Number(config.frame_width);
    }
    return frame.width;
  }

  override set frame_width(value: number) {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      this._pending_frame_width = value;
      return;
    }
    frame.stretch_to_fit_width(value);
  }

  override get frame_center(): readonly number[] {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      return this._pending_frame_center ?? [...ORIGIN];
    }
    return frame.get_center();
  }

  override set frame_center(frame_center: readonly number[] | Mobject) {
    const frame = (this as { frame?: Mobject }).frame;
    if (frame === undefined) {
      if (!(frame_center instanceof Mobject)) {
        this._pending_frame_center = [...frame_center];
      }
      return;
    }
    frame.move_to(frame_center as Mobject);
  }

  override capture_mobjects(mobjects: Iterable<Mobject>, options: CaptureOptions = {}): void {
    // self.reset_frame_center()
    // self.realign_frame_shape()
    super.capture_mobjects(mobjects, options);
  }

  override get_cached_cairo_context(_pixel_array: NDArray): null {
    // Since the frame can be moving around, the cairo context used for
    // updating should be regenerated at each frame. So no caching.
    return null;
  }

  override cache_cairo_context(_pixel_array: NDArray, _ctx: unknown): void {
    // Since the frame can be moving around, the cairo context used for
    // updating should be regenerated at each frame. So no caching.
  }

  get_mobjects_indicating_movement(): Mobject[] {
    return [this.frame];
  }
}
