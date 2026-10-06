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
import { InteractiveScene } from "./scene/interactive_scene.js";
import { DomInteractiveHost } from "./interactive/dom_host.js";
import {
  create_frame_scheduler,
  get_render_scheduler,
  is_thenable,
  set_render_scheduler,
  type RenderScheduler,
} from "./utils/scheduler.js";

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
  /**
   * Render in real time: `Scene.play` yields to the event loop after every
   * drawn frame and only resumes once the target frame interval has elapsed.
   * Defaults to true — this is what makes the in-browser preview live rather
   * than "compute everything, then play".
   */
  realtime?: boolean;
  /** Target logical frame rate in realtime mode (default 30). */
  fps?: number;
  /** Use the WebGL2 backend for 3D geometry when available (default true). */
  useWebGL?: boolean;
  /** Abort a realtime run; the scene stops at the next frame boundary. */
  signal?: AbortSignal;
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
 * The default (`realtime: true`) is the live preview: frames are drawn as they
 * are produced and the scene task yields between frames, so the page stays
 * responsive.  Set `realtime: false` to run the whole scene synchronously
 * (used for benchmarking/recording).
 *
 * - Benchmark / limit-FPS: `drawFrames: true, recordFrames: false, realtime: false`.
 * - Recorded playback: `recordFrames: true, drawFrames: false, realtime: false`,
 *   then play `frames`.
 */
export async function render_scene_to_canvas(
  sceneClass: SceneConstructor,
  canvas: HTMLCanvasElement,
  options: BrowserRenderOptions = {},
): Promise<BrowserRenderResult> {
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
    abortSignal: options.signal,
    onFrame: options.onProgress,
    useWebGL: options.useWebGL,
  });
  const scene = construct_scene(sceneClass, renderer as never);
  // Interactive scenes need a DOM host to receive pointer events and to place
  // the restart/next buttons.  Without a DOM (Node) the scene falls back to its
  // headless host and runs a single pass.
  let interactiveHost: DomInteractiveHost | null = null;
  if (scene instanceof InteractiveScene && typeof document !== "undefined") {
    interactiveHost = new DomInteractiveHost(canvas);
    scene.interactive_host = interactiveHost;
  }

  const realtime = options.realtime ?? true;
  const previousScheduler = get_render_scheduler();
  const scheduler: RenderScheduler | null = realtime
    ? create_frame_scheduler({ fps: options.fps ?? 30 })
    : null;
  if (scheduler !== null) set_render_scheduler(scheduler);
  const onAbort = (): void => scheduler?.cancel();
  options.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const rendered = scene.render();
    if (is_thenable(rendered)) await rendered;
  } finally {
    options.signal?.removeEventListener("abort", onAbort);
    // Only restore if a newer run has not already installed its own scheduler:
    // an aborted run can finish after the next run has started.
    if (scheduler !== null && get_render_scheduler() === scheduler) {
      set_render_scheduler(previousScheduler);
    }
    interactiveHost?.dispose();
    renderer.dispose();
  }
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
): Promise<BrowserRenderResult> {
  return render_scene_to_canvas(sceneClass, canvas, {
    ...options,
    drawFrames: true,
    recordFrames: false,
    realtime: options.realtime ?? false,
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

// Real-time scheduling primitives, re-exported so embedders (and the demo
// page) can drive or cancel a live render without importing internals.
export {
  create_frame_scheduler,
  get_render_scheduler,
  is_async_rendering,
  set_render_scheduler,
} from "./utils/scheduler.js";
export type { FrameSchedulerOptions, RenderScheduler } from "./utils/scheduler.js";
