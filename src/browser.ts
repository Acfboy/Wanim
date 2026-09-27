/**
 * Browser-facing helpers for the Canvas2D backend.
 *
 * These wrap the Scene execution and serialized-document playback in a small
 * API that the demo page (and any embedder) can call.  The functions are
 * DOM-aware but otherwise dependency-free; the whole module is bundled for the
 * browser by `scripts/build-browser.mjs`.
 */
import { configureWasm as numpyConfigureWasm, wasmFreeBytes } from "numpy-ts";
import { Camera } from "./camera/camera.js";
import type { MJSONDocument, MJSONFrame, MJSONMeta } from "./renderer/canvas.js";
import { backgroundCss } from "./renderer/canvas.js";
import {
  CanvasSceneRenderer,
  MJSONCanvasPlayer,
  type CanvasRenderStats,
  type MJSONPlayerHandle,
  type MJSONPlayerOptions,
} from "./renderer/canvas_renderer.js";
import {
  apply_quality,
  construct_scene,
  default_camera_class,
  type SceneConstructor,
} from "./renderer/scene_runner.js";

/**
 * Resize numpy-ts' fixed WebAssembly heap.  Must be called before the first
 * array operation; if the heap is already initialised numpy-ts throws, which
 * is surfaced to the caller.  A larger heap avoids the silent "results are now
 * JS-backed" cliff that slows mobject interpolation dramatically.
 */
export function configure_numpy_wasm(options: { maxMemory?: number; scratchSize?: number }): void {
  numpyConfigureWasm(options);
}

/** Free bytes left in the numpy-ts WebAssembly heap (diagnostics). */
export function numpy_wasm_free_bytes(): number {
  return wasmFreeBytes();
}

export interface BrowserRenderOptions {
  /** Quality flag (`-ql`/`-qm`/`-qh`/`-q4k`); defaults to `-ql`. */
  quality?: string;
  /** Draw every frame to the canvas during the run (default true). */
  drawFrames?: boolean;
  /** Serialize every frame into `frames` for later playback (default false). */
  recordFrames?: boolean;
  onProgress?: (info: { frames: number; plays: number; elapsedMs: number }) => void;
}

export interface BrowserRenderResult {
  stats: CanvasRenderStats;
  /** Recorded frame groups, present when `recordFrames` was set. */
  frames: MJSONFrame[];
  meta: MJSONMeta;
}

function make_meta(camera: Camera, numFrames: number, numPlays: number): MJSONMeta {
  return {
    width: camera.pixel_width,
    height: camera.pixel_height,
    frame_rate: camera.frame_rate,
    background_color: camera.background_color.to_hex(),
    background_opacity: camera.background_opacity,
    num_frames: numFrames,
    num_plays: numPlays,
  };
}

/**
 * Run one Scene class against a canvas.
 *
 * - Benchmark / limit-FPS: `drawFrames: true, recordFrames: false`.
 * - Preview: `recordFrames: true, drawFrames: false`, then play `frames`.
 */
export function render_scene_to_canvas(
  sceneClass: SceneConstructor,
  canvas: HTMLCanvasElement,
  options: BrowserRenderOptions = {},
): BrowserRenderResult {
  const cameraClass = default_camera_class(sceneClass);
  apply_quality(options.quality ?? "-ql");
  const camera = new cameraClass();
  canvas.width = camera.pixel_width;
  canvas.height = camera.pixel_height;
  const ctx = canvas.getContext("2d");
  if (ctx === null) throw new Error("2d canvas context unavailable");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = backgroundCss({
    background_color: camera.background_color.to_hex(),
    background_opacity: camera.background_opacity,
  });
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const renderer = new CanvasSceneRenderer("", {
    camera_class: cameraClass,
    ctx,
    drawFrames: options.drawFrames ?? true,
    recordFrames: options.recordFrames ?? false,
    onFrame: options.onProgress,
  });
  const scene = construct_scene(sceneClass, renderer as never);
  scene.render();
  return {
    stats: renderer.stats,
    frames: renderer.frames as unknown as MJSONFrame[],
    meta: make_meta(renderer.camera, renderer.stats.frames, renderer.num_plays),
  };
}

/** Convenience: a throughput benchmark (draw every frame, never serialize). */
export function benchmark_scene(
  sceneClass: SceneConstructor,
  canvas: HTMLCanvasElement,
  options: Omit<BrowserRenderOptions, "drawFrames" | "recordFrames"> = {},
): BrowserRenderResult {
  return render_scene_to_canvas(sceneClass, canvas, {
    ...options,
    drawFrames: true,
    recordFrames: false,
  });
}

/** Play a recorded frame set (as produced by `render_scene_to_canvas`). */
export function play_frames(
  canvas: HTMLCanvasElement,
  meta: MJSONMeta,
  frames: MJSONFrame[],
  options: MJSONPlayerOptions = {},
): MJSONPlayerHandle {
  const player = new MJSONCanvasPlayer(canvas, { meta, frames });
  return player.play(options);
}

/** Parse an `.mjson` string. */
export function parse_mjson(text: string): MJSONDocument {
  return JSON.parse(text) as MJSONDocument;
}

/** Parse and play an `.mjson` document on a canvas. */
export function play_mjson(
  canvas: HTMLCanvasElement,
  text: string,
  options: MJSONPlayerOptions = {},
): MJSONPlayerHandle {
  return new MJSONCanvasPlayer(canvas, parse_mjson(text)).play(options);
}
