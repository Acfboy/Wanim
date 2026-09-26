import type { NDArray } from "numpy-ts";
import { ApplyMethod, type MobjectBoundMethod } from "../animation/transform.js";
import { MovingCamera, type MovingCameraOptions } from "../camera/moving_camera.js";
import { MultiCamera } from "../camera/multi_camera.js";
import {
  DEFAULT_MOBJECT_TO_EDGE_BUFFER,
  ORIGIN,
  RIGHT,
  UP,
} from "../constants.js";
import type { Mobject } from "../mobject/mobject.js";
import { ImageMobjectFromCamera, type ImageMobjectFromCameraOptions } from "../mobject/types/image_mobject.js";
import { MovingCameraScene, type MovingCameraSceneOptions } from "./moving_camera_scene.js";

/**
 * Builds a bound-method facade for `mobject[name]`, carrying `__self__` /
 * `__func__` the way CPython bound methods do, so it passes
 * `ApplyMethod.check_validity_of_input` (mirrors the private `boundMethodOf`
 * helper in animation/transform.ts, which is not exported).
 */
function boundMethodOf(mobject: Mobject, name: string): MobjectBoundMethod {
  const unbound = (mobject as unknown as Record<string, unknown>)[name];
  if (typeof unbound !== "function") {
    throw new Error(`${mobject.constructor.name} has no method '${name}'`);
  }
  const bound = function (...args: unknown[]): unknown {
    return (unbound as (...callArgs: unknown[]) => unknown).apply(mobject, args);
  } as MobjectBoundMethod;
  bound.__self__ = mobject;
  bound.__func__ = unbound as (...args: never[]) => unknown;
  return bound;
}

export interface ZoomedSceneOptions extends MovingCameraSceneOptions {
  camera_class?: new (...args: any[]) => any;
  zoomed_display_height?: number;
  zoomed_display_width?: number;
  zoomed_display_center?: readonly number[] | NDArray | null;
  zoomed_display_corner?: readonly number[] | NDArray;
  zoomed_display_corner_buff?: number;
  zoomed_camera_config?: Record<string, unknown>;
  zoomed_camera_image_mobject_config?: Record<string, unknown>;
  zoomed_camera_frame_starting_position?: readonly number[] | NDArray;
  zoom_factor?: number;
  image_frame_stroke_width?: number;
  zoom_activated?: boolean;
}

const DEFAULT_ZOOMED_CAMERA_CONFIG: Record<string, unknown> = {
  default_frame_stroke_width: 2,
  background_opacity: 1,
};

export class ZoomedScene extends MovingCameraScene {
  zoomed_display_height: number;
  zoomed_display_width: number;
  zoomed_display_center: readonly number[] | NDArray | null;
  zoomed_display_corner: readonly number[] | NDArray;
  zoomed_display_corner_buff: number;
  zoomed_camera_config: Record<string, unknown>;
  zoomed_camera_image_mobject_config: Record<string, unknown>;
  zoomed_camera_frame_starting_position: readonly number[] | NDArray;
  zoom_factor: number;
  image_frame_stroke_width: number;
  zoom_activated: boolean;
  zoomed_camera!: MovingCamera;
  zoomed_display!: ImageMobjectFromCamera;

  constructor(
    camera_class: new (...args: any[]) => any = MultiCamera as unknown as new (...args: any[]) => any,
    options: ZoomedSceneOptions = {},
  ) {
    const {
      camera_class: cameraClassOption,
      zoomed_display_height = 3,
      zoomed_display_width = 3,
      zoomed_display_center = null,
      zoomed_display_corner = [
        (UP[0] ?? 0) + (RIGHT[0] ?? 0),
        (UP[1] ?? 0) + (RIGHT[1] ?? 0),
        (UP[2] ?? 0) + (RIGHT[2] ?? 0),
      ],
      zoomed_display_corner_buff = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
      zoomed_camera_config = { ...DEFAULT_ZOOMED_CAMERA_CONFIG },
      zoomed_camera_image_mobject_config = {},
      zoomed_camera_frame_starting_position = [...ORIGIN],
      zoom_factor = 0.15,
      image_frame_stroke_width = 3,
      zoom_activated = false,
      ...rest
    } = options;
    // NOTE: Python assigns the fields above before super().__init__(); TS
    // requires super() first. The base Scene constructor does not read any
    // of these fields, so assigning afterwards is behaviorally identical.
    super((cameraClassOption as new (...args: any[]) => any) ?? camera_class, rest);
    this.zoomed_display_height = zoomed_display_height;
    this.zoomed_display_width = zoomed_display_width;
    this.zoomed_display_center = zoomed_display_center;
    this.zoomed_display_corner = zoomed_display_corner;
    this.zoomed_display_corner_buff = zoomed_display_corner_buff;
    this.zoomed_camera_config = zoomed_camera_config;
    this.zoomed_camera_image_mobject_config = zoomed_camera_image_mobject_config;
    this.zoomed_camera_frame_starting_position = zoomed_camera_frame_starting_position;
    this.zoom_factor = zoom_factor;
    this.image_frame_stroke_width = image_frame_stroke_width;
    this.zoom_activated = zoom_activated;
  }

  override setup(): void {
    super.setup();
    // Initialize camera and display
    const zoomed_camera = new MovingCamera(
      null,
      { ...(this.zoomed_camera_config as MovingCameraOptions) },
    );
    const zoomed_display = new ImageMobjectFromCamera(
      zoomed_camera,
      null,
      { ...(this.zoomed_camera_image_mobject_config as ImageMobjectFromCameraOptions) },
    );
    zoomed_display.add_display_frame();
    for (const mob of [zoomed_camera.frame, zoomed_display]) {
      mob.stretch_to_fit_height(this.zoomed_display_height);
      mob.stretch_to_fit_width(this.zoomed_display_width);
    }
    zoomed_camera.frame.scale(this.zoom_factor);

    // Position camera and display
    zoomed_camera.frame.move_to(this.zoomed_camera_frame_starting_position);
    if (this.zoomed_display_center !== null && this.zoomed_display_center !== undefined) {
      zoomed_display.move_to(this.zoomed_display_center);
    } else {
      zoomed_display.to_corner(
        this.zoomed_display_corner,
        this.zoomed_display_corner_buff,
      );
    }

    this.zoomed_camera = zoomed_camera;
    this.zoomed_display = zoomed_display;
  }

  activate_zooming(animate = false): void {
    this.zoom_activated = true;
    // NOTE: Python carries `# type: ignore[union-attr]` here because the
    // camera is only a MultiCamera when constructed with the default
    // camera_class; the TS renderer/camera fields are `any`, so no
    // ignore-comment is needed.
    this.renderer.camera.add_image_mobject_from_camera(this.zoomed_display);
    if (animate) {
      this.play(this.get_zoom_in_animation());
      this.play(this.get_zoomed_display_pop_out_animation());
    }
    this.add_foreground_mobjects(
      this.zoomed_camera.frame,
      this.zoomed_display,
    );
  }

  get_zoom_in_animation(
    run_time = 2,
    kwargs: Record<string, unknown> = {},
  ): ApplyMethod {
    const frame = this.zoomed_camera.frame;
    // NOTE: Python branches on isinstance(self.camera, OpenGLCamera) for
    // frame_shape; the OpenGL renderer/camera is not ported yet, so the
    // Cairo path (frame_width/frame_height) always applies.
    const full_frame_height: number = this.camera.frame_height;
    const full_frame_width: number = this.camera.frame_width;
    frame.save_state();
    frame.stretch_to_fit_width(full_frame_width);
    frame.stretch_to_fit_height(full_frame_height);
    frame.center();
    frame.set_stroke(undefined, 0);
    // NOTE: Python forwards run_time/**kwargs to the Transform constructor;
    // the TS ApplyMethod constructor has no animation-options channel, so
    // they are assigned onto the instance afterwards (explicit kwargs win,
    // mirroring Python's `run_time=run_time, **kwargs`).
    const animation = new ApplyMethod(boundMethodOf(frame, "restore"));
    animation.run_time = run_time;
    for (const [key, value] of Object.entries(kwargs)) {
      (animation as unknown as Record<string, unknown>)[key] = value;
    }
    return animation;
  }

  get_zoomed_display_pop_out_animation(
    kwargs: Record<string, unknown> = {},
  ): ApplyMethod {
    const display = this.zoomed_display;
    display.save_state();
    display.replace(this.zoomed_camera.frame, 0, true);
    const animation = new ApplyMethod(boundMethodOf(display, "restore"));
    for (const [key, value] of Object.entries(kwargs)) {
      (animation as unknown as Record<string, unknown>)[key] = value;
    }
    return animation;
  }

  get_zoom_factor(): number {
    const zoom_factor: number = this.zoomed_camera.frame.height / this.zoomed_display.height;
    return zoom_factor;
  }
}
