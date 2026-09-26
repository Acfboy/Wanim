import { config } from "../_config/utils.js";
import type { Animation } from "../animation/animation.js";
import type { Mobject } from "../mobject/mobject.js";
import type { Scene } from "../scene/scene.js";
import { get_hash_from_play_call } from "./hash.js";

/** Minimal surface of the OpenGL renderer used by `handle_caching_play`. */
export interface CachingRenderer {
  skip_animations: boolean;
  _original_skipping_status: boolean;
  num_plays: number;
  animations_hashes: Array<string | null>;
  camera: unknown;
  file_writer: {
    is_already_cached(hash: string): boolean;
    add_partial_movie_file(hash: string | null): void;
  };
  update_skipping_status(): void;
}

export type PlayLikeFunc = (
  this: CachingRenderer,
  scene: Scene,
  ...args: unknown[]
) => void;

/**
 * Decorator that returns a wrapped version of func that will compute
 * the hash of the play invocation.
 *
 * The returned function will act according to the computed hash: either skip
 * the animation because it's already cached, or let the invoked function
 * play normally.
 */
export function handle_caching_play(func: PlayLikeFunc): PlayLikeFunc {
  // NOTE : This is only kept for OpenGL renderer.
  // The play logic of the cairo renderer has been refactored and does not need this function anymore.
  // When OpenGL renderer will have a proper testing system,
  // the play logic of the latter has to be refactored in the same way the cairo renderer has been, and thus this
  // method has to be deleted.
  function wrapper(this: CachingRenderer, scene: Scene, ...args: unknown[]): void {
    this.skip_animations = this._original_skipping_status;
    this.update_skipping_status();
    const animations = scene.compile_animations(
      ...(args as Array<Animation | Mobject>),
    );
    scene.add_mobjects_from_animations(animations);
    if (this.skip_animations) {
      console.debug(`Skipping animation ${this.num_plays}`);
      func.call(this, scene, ...args);
      // If the animation is skipped, we mark its hash as None.
      // When sceneFileWriter will start combining partial movie files, it won't take into account None hashes.
      this.animations_hashes.push(null);
      this.file_writer.add_partial_movie_file(null);
      return;
    }
    let hash_play: string;
    if (!config.disable_caching) {
      const mobjects_on_scene = scene.mobjects;
      hash_play = get_hash_from_play_call(
        scene,
        this.camera,
        animations,
        mobjects_on_scene,
      );
      if (this.file_writer.is_already_cached(hash_play)) {
        console.info(
          `Animation ${this.num_plays} : Using cached data (hash : ${hash_play})`,
        );
        this.skip_animations = true;
      }
    } else {
      hash_play = `uncached_${String(this.num_plays).padStart(5, "0")}`;
    }
    this.animations_hashes.push(hash_play);
    this.file_writer.add_partial_movie_file(hash_play);
    console.debug(
      `List of the first few animation hashes of the scene: ${String(this.animations_hashes.slice(0, 5))}`,
    );
    func.call(this, scene, ...args);
  }

  return wrapper;
}
