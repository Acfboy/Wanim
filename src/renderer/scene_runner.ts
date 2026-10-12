/**
 * Scene execution harness mirroring `the original manim interpreter/__main__.py`.
 *
 * `run_scene` creates a `JSONCairoRenderer` for a Scene class, runs the scene
 * with the standard manim machinery and writes the serialized `.mjson`
 * document.  `run_scenes_from_module` discovers Scene subclasses exported by
 * a module and writes one `.mjson` per scene.
 *
 * Node.js rendering is meant for testing/benchmarking the library and for
 * producing reference `.mjson` documents; wanim's product runs scenes live in
 * the browser (`browser.ts`), so the Node-only helpers keep their own defaults
 * and are not part of the user-facing API.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { config } from "../_config/utils.js";
import { Camera } from "../camera/camera.js";
import { MovingCamera } from "../camera/moving_camera.js";
import { ThreeDCamera } from "../camera/three_d_camera.js";
import { Scene } from "../scene/scene.js";
import { MovingCameraScene } from "../scene/moving_camera_scene.js";
import { SpecialThreeDScene, ThreeDScene } from "../scene/three_d_scene.js";
import { InteractiveThreeDScene } from "../scene/interactive_three_d_scene.js";
import { get_scene_classes_from_module, type SceneModule } from "../utils/module_ops.js";
import { is_thenable } from "../utils/scheduler.js";
import { JSONCairoRenderer } from "./json_renderer.js";

/** (label, pixel_width, pixel_height, frame_rate) — mirrors __main__.py. */
export const QUALITY_FLAGS: Record<string, [string, number, number, number]> = {
  "-ql": ["low", 854, 480, 15],
  "-qm": ["medium", 1280, 720, 30],
  "-qh": ["high", 1920, 1080, 60],
  "-q4k": ["4k", 3840, 2160, 60],
};

export type SceneConstructor = new (...args: any[]) => Scene;

/** Mirrors `apply_quality`: sets config resolution / frame rate. */
export function apply_quality(flag = "-qm"): string {
  const entry = QUALITY_FLAGS[flag];
  if (entry === undefined) {
    const error = new Error(`unknown quality flag: ${flag}`);
    error.name = "ValueError";
    throw error;
  }
  const [label, pixelWidth, pixelHeight, frameRate] = entry;
  config.pixel_width = pixelWidth;
  config.pixel_height = pixelHeight;
  config.frame_rate = frameRate;
  return label;
}

function isSubclassOf(ctor: unknown, base: unknown): boolean {
  let proto: unknown = (ctor as { prototype?: unknown }).prototype;
  const seen = new Set<unknown>();
  while (proto !== null && proto !== undefined && typeof proto === "object") {
    if (seen.has(proto)) return false;
    seen.add(proto);
    if ((proto as { constructor?: unknown }).constructor === base) return true;
    proto = Object.getPrototypeOf(proto);
  }
  return false;
}

/** The camera class the scene would use by default (mirrors __main__.py). */
export function default_camera_class(sceneClass: SceneConstructor): new (...args: any[]) => Camera {
  // `InteractiveThreeDScene` composes `InteractiveScene` (for the interactive
  // loop) with ThreeDScene's camera methods, so it is not a `ThreeDScene`
  // subclass and needs its own arm here.
  if (
    isSubclassOf(sceneClass, ThreeDScene) ||
    isSubclassOf(sceneClass, InteractiveThreeDScene)
  ) {
    return ThreeDCamera as unknown as new (...args: any[]) => Camera;
  }
  if (isSubclassOf(sceneClass, MovingCameraScene)) {
    return MovingCamera as unknown as new (...args: any[]) => Camera;
  }
  return Camera;
}

export function construct_scene(sceneClass: SceneConstructor, renderer: JSONCairoRenderer): Scene {
  // NOTE: Python's `scene_class(renderer=renderer)` forwards the keyword
  // through `**kwargs` down the MRO.  TS constructors do not share that
  // channel, so the renderer is routed into the options bag at the position
  // each scene hierarchy expects.
  if (isSubclassOf(sceneClass, SpecialThreeDScene)) {
    return new sceneClass(undefined, undefined, undefined, undefined, undefined, undefined, { renderer });
  }
  if (isSubclassOf(sceneClass, ThreeDScene)) {
    return new sceneClass(undefined, undefined, undefined, { renderer });
  }
  if (isSubclassOf(sceneClass, MovingCameraScene)) {
    return new sceneClass(undefined, { renderer });
  }
  return new sceneClass({ renderer });
}

/**
 * Run one Scene class and serialize it to `outputPath`. Returns the path.
 *
 * Hand-written JS scenes run synchronously (and return the path directly);
 * rapydscript-translated scenes have an `async construct` (the compiler inserts
 * `await self.play(...)`), so rendering completes asynchronously and a promise
 * for the path is returned instead.
 */
export function run_scene(sceneClass: SceneConstructor, outputPath: string): string | Promise<string> {
  const renderer = new JSONCairoRenderer(outputPath, {
    camera_class: default_camera_class(sceneClass),
  });
  const scene = construct_scene(sceneClass, renderer);
  const rendered = scene.render();
  if (is_thenable(rendered)) {
    return Promise.resolve(rendered).then(() => renderer.write_output());
  }
  return renderer.write_output();
}

export interface RunScenesOptions {
  outputDir?: string;
  sceneNames?: string[];
  quality?: string;
}

/**
 * Discover Scene subclasses exported by `module` and write one `.mjson` per
 * scene named `<SceneName>.mjson`.
 *
 * Test/benchmark harness: the low default keeps Node runs fast; the browser
 * entries default to `-qm`.
 */
export function run_scenes_from_module(
  module: Record<string, unknown>,
  options: RunScenesOptions = {},
): string[] | Promise<string[]> {
  apply_quality(options.quality ?? "-ql");
  const sceneClasses = get_scene_classes_from_module(module as unknown as SceneModule) as unknown as SceneConstructor[];
  const selected = options.sceneNames && options.sceneNames.length > 0
    ? sceneClasses.filter((sceneClass) => options.sceneNames!.includes(sceneClass.name))
    : sceneClasses;
  const outputDir = options.outputDir ?? join("media", "interpreted");
  mkdirSync(outputDir, { recursive: true });
  const results = selected.map((sceneClass) => run_scene(sceneClass, join(outputDir, `${sceneClass.name}.mjson`)));
  if (results.some(is_thenable)) {
    return Promise.all(results.map((result) => Promise.resolve(result)));
  }
  return results as string[];
}
