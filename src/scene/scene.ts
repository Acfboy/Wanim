import { Mobject } from "../mobject/mobject.js";
import { Animation, prepare_animation } from "../animation/animation.js";

export class Scene {
  mobjects: Mobject[] = [];
  foreground_mobjects: Mobject[] = [];
  animations: Animation[] = [];
  camera: any = null;
  renderer: any = null;
  key_to_function_map: Map<string, () => void> = new Map();
  interactive_mode = false;
  always_update_mobjects: boolean;

  constructor(renderer: any = null, camera_class: new (...args: any[]) => any = class {}, always_update_mobjects = false) {
    this.renderer = renderer;
    this.camera = new camera_class();
    this.always_update_mobjects = always_update_mobjects;
  }

  add(...mobjects: Mobject[]): this {
    for (const mobject of mobjects) if (!this.mobjects.includes(mobject)) this.mobjects.push(mobject);
    return this;
  }

  remove(...mobjects: Mobject[]): this {
    for (const mobject of mobjects) {
      const index = this.mobjects.indexOf(mobject);
      if (index >= 0) this.mobjects.splice(index, 1);
    }
    return this;
  }

  get_mobject_family_members(): Mobject[] {
    return this.mobjects.flatMap((mobject) => mobject.get_family());
  }

  play(...animations: (Animation | Mobject)[]): this {
    const prepared = animations.map((animation) => prepare_animation(animation));
    this.animations.push(...prepared);
    for (const animation of prepared) {
      animation._setup_scene(this);
      animation.begin();
      animation.finish();
      animation.clean_up_from_scene(this);
    }
    return this;
  }

  wait(_duration = 1): this { return this; }

  on_key_press(symbol: number, _modifiers: number): void {
    const char = String.fromCodePoint(symbol);
    if (char === "q") this.interactive_mode = false;
    else this.key_to_function_map.get(char)?.();
  }

  on_key_release(symbol: number, modifiers: number): void {
    void symbol;
    void modifiers;
  }
}
