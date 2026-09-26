import type { Mobject } from "../mobject/mobject.js";
import type { ImageMobjectFromCamera } from "../mobject/types/image_mobject.js";
import { list_difference_update } from "../utils/iterables.js";
import { type CaptureOptions } from "./camera.js";
import { MovingCamera, type MovingCameraOptions } from "./moving_camera.js";

export interface MultiCameraOptions extends MovingCameraOptions {
  image_mobjects_from_cameras?: Iterable<ImageMobjectFromCamera> | null;
  allow_cameras_to_capture_their_own_display?: boolean;
}

export class MultiCamera extends MovingCamera {
  image_mobjects_from_cameras: ImageMobjectFromCamera[];
  allow_cameras_to_capture_their_own_display: boolean;

  constructor(
    image_mobjects_from_cameras: Iterable<ImageMobjectFromCamera> | null = null,
    allow_cameras_to_capture_their_own_display = false,
    options: MultiCameraOptions = {},
  ) {
    const {
      image_mobjects_from_cameras: imfcOption,
      allow_cameras_to_capture_their_own_display: allowOption,
      ...rest
    } = options;
    // NOTE: Python assigns the fields below before super().__init__(); TS
    // requires super() first. The base Camera constructor calls reset(),
    // which is overridden here and reads image_mobjects_from_cameras, so the
    // field is pre-initialized via `declare`-free assignment order below and
    // reset() guards the construction-time call (see below).
    super(null, rest);
    this.image_mobjects_from_cameras = [];
    const resolved = imfcOption ?? image_mobjects_from_cameras;
    if (resolved !== null && resolved !== undefined) {
      for (const imfc of resolved) this.add_image_mobject_from_camera(imfc);
    }
    this.allow_cameras_to_capture_their_own_display =
      allowOption ?? allow_cameras_to_capture_their_own_display;
  }

  add_image_mobject_from_camera(
    image_mobject_from_camera: ImageMobjectFromCamera,
  ): void {
    // A silly method to have right now, but maybe there are things
    // we want to guarantee about any imfc's added later.
    const imfc = image_mobject_from_camera;
    if (!(imfc.camera instanceof MovingCamera)) {
      const error = new Error("assert isinstance(imfc.camera, MovingCamera)");
      error.name = "AssertionError";
      throw error;
    }
    this.image_mobjects_from_cameras.push(imfc);
  }

  update_sub_cameras(): void {
    for (const imfc of this.image_mobjects_from_cameras) {
      const [pixel_height = 0, pixel_width = 0] = this.pixel_array.shape.slice(0, 2);
      // NOTE: Python int() truncates toward zero; pixel counts are
      // non-negative here, matching Math.trunc.
      imfc.camera.reset_pixel_shape(
        Math.trunc((pixel_height * imfc.height) / this.frame_height),
        Math.trunc((pixel_width * imfc.width) / this.frame_width),
      );
    }
  }

  override reset(): this {
    // NOTE: the base Camera constructor calls reset() before this class
    // assigns image_mobjects_from_cameras (TS requires super() first,
    // whereas Python assigns before super().__init__()); guard the
    // construction-time call.
    for (const imfc of this.image_mobjects_from_cameras ?? []) {
      imfc.camera.reset();
    }
    super.reset();
    return this;
  }

  override capture_mobjects(
    mobjects: Iterable<Mobject>,
    options: CaptureOptions = {},
  ): void {
    this.update_sub_cameras();
    for (const imfc of this.image_mobjects_from_cameras) {
      let to_add = [...mobjects];
      if (!this.allow_cameras_to_capture_their_own_display) {
        to_add = list_difference_update(to_add, imfc.get_family());
      }
      imfc.camera.capture_mobjects(to_add, options);
    }
    super.capture_mobjects(mobjects, options);
  }

  override get_mobjects_indicating_movement(): Mobject[] {
    return [
      this.frame,
      ...this.image_mobjects_from_cameras.map((imfc) => imfc.camera.frame),
    ];
  }
}
