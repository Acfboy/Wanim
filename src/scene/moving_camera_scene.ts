import type { Animation } from "../animation/animation.js";
import { MovingCamera } from "../camera/moving_camera.js";
import type { Mobject } from "../mobject/mobject.js";
import { Scene } from "./scene.js";
import { extract_mobject_family_members } from "../utils/family.js";
import { list_update } from "../utils/iterables.js";

export interface MovingCameraSceneOptions {
  renderer?: unknown;
  camera_class?: new (...args: any[]) => any;
  always_update_mobjects?: boolean;
  random_seed?: number | null;
  skip_animations?: boolean;
  [key: string]: unknown;
}

export class MovingCameraScene extends Scene {
  constructor(
    camera_class: new (...args: any[]) => any = MovingCamera as unknown as new (...args: any[]) => any,
    options: MovingCameraSceneOptions = {},
  ) {
    super(
      (options.renderer as null) ?? null,
      (options.camera_class as new (...args: any[]) => any) ?? camera_class,
      options.always_update_mobjects ?? false,
      (options.random_seed as number | null | undefined) ?? undefined,
      (options.skip_animations as boolean | undefined) ?? false,
    );
  }

  override get_moving_mobjects(...animations: Animation[]): Mobject[] {
    const moving_mobjects = super.get_moving_mobjects(...animations);
    const all_moving_mobjects = extract_mobject_family_members(moving_mobjects);
    const movement_indicators = (this.renderer.camera as MovingCamera).get_mobjects_indicating_movement();
    for (const movement_indicator of movement_indicators) {
      if (all_moving_mobjects.includes(movement_indicator)) {
        // When one of these is moving, the camera should
        // consider all mobjects to be moving
        return list_update(this.mobjects, moving_mobjects);
      }
    }
    return moving_mobjects;
  }
}
