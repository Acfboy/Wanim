import { config } from "../_config/utils.js";
import { DEGREES, RendererType } from "../constants.js";
import type { Animation } from "../animation/animation.js";
import { Transform } from "../animation/transform.js";
import { Camera } from "../camera/camera.js";
import { Mobject } from "../mobject/mobject.js";
import { ValueTracker } from "../mobject/value_tracker.js";
import { Scene } from "./scene.js";

export interface ThreeDSceneOptions {
  renderer?: unknown;
  camera_class?: new (...args: any[]) => any;
  always_update_mobjects?: boolean;
  [key: string]: unknown;
}

export class ThreeDScene extends Scene {
  ambient_camera_rotation: unknown;
  default_angled_camera_orientation_kwargs: Record<string, number>;

  constructor(
    camera_class: new (...args: any[]) => any = Camera as unknown as new (...args: any[]) => any,
    ambient_camera_rotation: unknown = null,
    default_angled_camera_orientation_kwargs: Record<string, number> | null = null,
    options: ThreeDSceneOptions = {},
  ) {
    // NOTE: Python defaults camera_class to ThreeDCamera, which has not
    // been ported yet (later batch); the Cairo Camera is used instead.
    // All camera interaction goes through `any`-typed handles, so behavior
    // binds to the real camera once it lands.
    super(
      (options.renderer as null) ?? null,
      options.camera_class ?? camera_class,
      options.always_update_mobjects ?? false,
    );
    this.ambient_camera_rotation = ambient_camera_rotation;
    if (default_angled_camera_orientation_kwargs === null) {
      default_angled_camera_orientation_kwargs = {
        phi: 70 * DEGREES,
        theta: -135 * DEGREES,
      };
    }
    this.default_angled_camera_orientation_kwargs = default_angled_camera_orientation_kwargs;
  }

  set_camera_orientation(
    phi: number | null = null,
    theta: number | null = null,
    gamma: number | null = null,
    zoom: number | null = null,
    focal_distance: number | null = null,
    frame_center: Mobject | number[] | readonly number[] | null = null,
    kwargs: Record<string, unknown> = {},
  ): void {
    void kwargs;
    if (phi !== null && phi !== undefined) {
      this.renderer.camera.set_phi(phi);
    }
    if (theta !== null && theta !== undefined) {
      this.renderer.camera.set_theta(theta);
    }
    if (focal_distance !== null && focal_distance !== undefined) {
      this.renderer.camera.set_focal_distance(focal_distance);
    }
    if (gamma !== null && gamma !== undefined) {
      this.renderer.camera.set_gamma(gamma);
    }
    if (zoom !== null && zoom !== undefined) {
      this.renderer.camera.set_zoom(zoom);
    }
    if (frame_center !== null && frame_center !== undefined) {
      this.renderer.camera._frame_center.move_to(frame_center);
    }
  }

  begin_ambient_camera_rotation(rate = 0.02, about = "theta"): void {
    // TODO, use a ValueTracker for rate, so that it
    // can begin and end smoothly
    const aboutKey: string = about.toLowerCase();
    try {
      if (config.renderer === RendererType.CAIRO) {
        const trackers: Record<string, ValueTracker> = {
          theta: this.camera.theta_tracker,
          phi: this.camera.phi_tracker,
          gamma: this.camera.gamma_tracker,
        };
        const tracker = trackers[aboutKey] as ValueTracker;
        tracker.add_updater((_m, dt) => {
          tracker.increment_value(rate * (dt ?? 0));
        });
        this.add(tracker);
      } else if (config.renderer === RendererType.OPENGL) {
        const cam = this.camera;
        const methods: Record<string, (value: number) => void> = {
          theta: (value: number) => cam.increment_theta(value),
          phi: (value: number) => cam.increment_phi(value),
          gamma: (value: number) => cam.increment_gamma(value),
        };
        cam.add_updater((_m: unknown, dt: number) => {
          (methods[aboutKey] as (value: number) => void)(rate * dt);
        });
        this.add(this.camera);
      }
    } catch (error) {
      const wrapper = new Error("Invalid ambient rotation angle.");
      wrapper.name = "ValueError";
      (wrapper as unknown as { cause: unknown }).cause = error;
      throw wrapper;
    }
  }

  stop_ambient_camera_rotation(about = "theta"): void {
    const aboutKey: string = about.toLowerCase();
    try {
      if (config.renderer === RendererType.CAIRO) {
        const trackers: Record<string, ValueTracker> = {
          theta: this.camera.theta_tracker,
          phi: this.camera.phi_tracker,
          gamma: this.camera.gamma_tracker,
        };
        const tracker = trackers[aboutKey] as ValueTracker;
        tracker.clear_updaters();
        this.remove(tracker);
      } else if (config.renderer === RendererType.OPENGL) {
        this.camera.clear_updaters();
      }
    } catch (error) {
      const wrapper = new Error("Invalid ambient rotation angle.");
      wrapper.name = "ValueError";
      (wrapper as unknown as { cause: unknown }).cause = error;
      throw wrapper;
    }
  }

  begin_3dillusion_camera_rotation(
    rate = 1,
    origin_phi: number | null = null,
    origin_theta: number | null = null,
  ): void {
    const thetaOrigin: number = origin_theta ?? this.renderer.camera.theta_tracker.get_value();
    const phiOrigin: number = origin_phi ?? this.renderer.camera.phi_tracker.get_value();

    const val_tracker_theta = new ValueTracker(0);
    const update_theta = (m: unknown, dt: number): void => {
      val_tracker_theta.increment_value(dt * rate);
      const val_for_left_right = 0.2 * Math.sin(val_tracker_theta.get_value());
      (m as ValueTracker).set_value(thetaOrigin + val_for_left_right);
    };
    this.renderer.camera.theta_tracker.add_updater(update_theta);
    this.add(this.renderer.camera.theta_tracker);

    const val_tracker_phi = new ValueTracker(0);
    const update_phi = (m: unknown, dt: number): void => {
      val_tracker_phi.increment_value(dt * rate);
      const val_for_up_down = 0.1 * Math.cos(val_tracker_phi.get_value()) - 0.1;
      (m as ValueTracker).set_value(phiOrigin + val_for_up_down);
    };
    this.renderer.camera.phi_tracker.add_updater(update_phi);
    this.add(this.renderer.camera.phi_tracker);
  }

  stop_3dillusion_camera_rotation(): void {
    this.renderer.camera.theta_tracker.clear_updaters();
    this.remove(this.renderer.camera.theta_tracker);
    this.renderer.camera.phi_tracker.clear_updaters();
    this.remove(this.renderer.camera.phi_tracker);
  }

  move_camera(
    phi: number | null = null,
    theta: number | null = null,
    gamma: number | null = null,
    zoom: number | null = null,
    focal_distance: number | null = null,
    frame_center: Mobject | number[] | readonly number[] | null = null,
    added_anims: Animation[] = [],
    kwargs: Record<string, unknown> = {},
  ): void {
    void kwargs;
    const anims: Animation[] = [];
    if (config.renderer === RendererType.CAIRO) {
      const value_tracker_pairs: Array<[number | null, ValueTracker]> = [
        [phi, this.camera.phi_tracker],
        [theta, this.camera.theta_tracker],
        [focal_distance, this.camera.focal_distance_tracker],
        [gamma, this.camera.gamma_tracker],
        [zoom, this.camera.zoom_tracker],
      ];
      for (const [value, tracker] of value_tracker_pairs) {
        if (value !== null && value !== undefined) {
          anims.push((tracker.animate as unknown as { set_value: (value: number) => Animation }).set_value(value));
        }
      }
      if (frame_center !== null && frame_center !== undefined) {
        anims.push(
          (this.camera._frame_center.animate as unknown as { move_to: (target: unknown) => Animation }).move_to(frame_center),
        );
      }
    } else if (config.renderer === RendererType.OPENGL) {
      const cam = this.camera;
      const cam2 = cam.copy();
      const methods: Record<string, (value: unknown) => void> = {
        theta: (value: unknown) => cam2.set_theta(value),
        phi: (value: unknown) => cam2.set_phi(value),
        gamma: (value: unknown) => cam2.set_gamma(value),
        zoom: (value: unknown) => cam2.scale(value),
        frame_center: (value: unknown) => cam2.move_to(value),
      };
      let frameCenterValue: unknown = frame_center;
      if (frameCenterValue !== null && frameCenterValue !== undefined) {
        const candidate = frameCenterValue as { get_center?: () => unknown };
        if (typeof candidate.get_center === "function") {
          frameCenterValue = candidate.get_center();
        }
        frameCenterValue = [...(frameCenterValue as Iterable<number>)].map(Number);
      }
      let zoom_value: number | null = null;
      if (zoom !== null && zoom !== undefined) {
        zoom_value = config.frame_height / ((zoom as number) * cam.height);
      }
      const targets: Array<[unknown, string]> = [
        [theta, "theta"],
        [phi, "phi"],
        [gamma, "gamma"],
        [zoom_value, "zoom"],
        [frameCenterValue, "frame_center"],
      ];
      for (const [value, method] of targets) {
        if (value !== null && value !== undefined) {
          (methods[method] as (value: unknown) => void)(value);
        }
      }
      if (focal_distance !== null && focal_distance !== undefined) {
        console.warn("focal distance of OpenGLCamera can not be adjusted.");
      }
      anims.push(new Transform(cam as Mobject, cam2 as Mobject));
    }
    this.play(...anims, ...added_anims);
    // These lines are added to improve performance. If manim thinks that frame_center is moving,
    // it is required to redraw every object. These lines remove frame_center from the Scene once
    // its animation is done, ensuring that manim does not think that it is moving. Since the
    // frame_center is never actually drawn, this shouldn't break anything.
    if (frame_center !== null && frame_center !== undefined && config.renderer === RendererType.CAIRO) {
      this.remove(this.camera._frame_center);
    }
  }

  override get_moving_mobjects(...animations: Animation[]): Mobject[] {
    const moving_mobjects = super.get_moving_mobjects(...animations);
    const camera_mobjects: unknown[] = [
      ...this.renderer.camera.get_value_trackers(),
      this.renderer.camera._frame_center,
    ];
    if (camera_mobjects.some((cm) => moving_mobjects.includes(cm as Mobject))) {
      return this.mobjects;
    }
    return moving_mobjects;
  }

  add_fixed_orientation_mobjects(...args: Array<Mobject | Record<string, unknown>>): void {
    // NOTE: Python takes (**kwargs) alongside *mobjects; the TS port reads
    // an optional trailing plain-object argument as those kwargs.
    let kwargs: Record<string, unknown> = {};
    let mobjects = args as Mobject[];
    const last = args[args.length - 1];
    if (last !== undefined && !(last instanceof Mobject) && typeof last === "object" && !Array.isArray(last)) {
      kwargs = last as Record<string, unknown>;
      mobjects = args.slice(0, -1) as Mobject[];
    }
    if (config.renderer === RendererType.CAIRO) {
      this.add(...mobjects);
      this.renderer.camera.add_fixed_orientation_mobjects(...mobjects, kwargs);
    } else if (config.renderer === RendererType.OPENGL) {
      for (const mob of mobjects) {
        (mob as unknown as { fix_orientation: () => void }).fix_orientation();
        this.add(mob);
      }
    }
  }

  add_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    if (config.renderer === RendererType.CAIRO) {
      this.add(...mobjects);
      this.camera.add_fixed_in_frame_mobjects(...mobjects);
    } else if (config.renderer === RendererType.OPENGL) {
      for (const mob of mobjects) {
        (mob as unknown as { fix_in_frame: () => void }).fix_in_frame();
        this.add(mob);
      }
    }
  }

  remove_fixed_orientation_mobjects(...mobjects: Mobject[]): void {
    if (config.renderer === RendererType.CAIRO) {
      this.renderer.camera.remove_fixed_orientation_mobjects(...mobjects);
    } else if (config.renderer === RendererType.OPENGL) {
      for (const mob of mobjects) {
        (mob as unknown as { unfix_orientation: () => void }).unfix_orientation();
        this.remove(mob);
      }
    }
  }

  remove_fixed_in_frame_mobjects(...mobjects: Mobject[]): void {
    if (config.renderer === RendererType.CAIRO) {
      this.renderer.camera.remove_fixed_in_frame_mobjects(...mobjects);
    } else if (config.renderer === RendererType.OPENGL) {
      for (const mob of mobjects) {
        (mob as unknown as { unfix_from_frame: () => void }).unfix_from_frame();
        this.remove(mob);
      }
    }
  }

  set_to_default_angled_camera_orientation(kwargs: Record<string, number> = {}): void {
    const merged: Record<string, number> = {
      ...this.default_angled_camera_orientation_kwargs,
      ...kwargs,
    };
    this.set_camera_orientation(
      merged.phi ?? null,
      merged.theta ?? null,
      merged.gamma ?? null,
      merged.zoom ?? null,
      merged.focal_distance ?? null,
      null,
      merged as Record<string, unknown>,
    );
  }
}
