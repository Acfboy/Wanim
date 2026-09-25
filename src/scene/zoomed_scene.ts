import { MovingCamera } from "../camera/moving_camera.js";
import type { Mobject } from "../mobject/mobject.js";
import { Scene } from "./scene.js";

// NOTE: minimal port. Full `ZoomedScene` (MovingCameraScene base, MultiCamera,
// ImageMobjectFromCamera display, activate_zooming and zoom animations) is left
// for a later batch; only `get_zoom_factor` is ported here.
export class ZoomedScene extends Scene {
  declare zoomed_camera: MovingCamera;
  declare zoomed_display: Mobject;

  get_zoom_factor(): number {
    const zoom_factor: number = this.zoomed_camera.frame.height / this.zoomed_display.height;
    return zoom_factor;
  }
}
