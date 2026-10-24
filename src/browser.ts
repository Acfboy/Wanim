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
import type { Scene } from "./scene/scene.js";
import { InteractiveScene } from "./scene/interactive_scene.js";
import { DomInteractiveHost } from "./interactive/dom_host.js";
import { NullInteractiveHost, type InteractiveHost } from "./interactive/host.js";
import {
  ScenePresentation,
  type PresentationState,
} from "./interactive/presentation.js";
import {
  TwoViewPresentation,
  type TwoViewState,
  type WanimMessage,
} from "./interactive/two_view.js";
import {
  normalize_presentation_mode,
  normalize_presentation_theme,
  type PresentationMode,
  type PresentationTheme,
} from "./interactive/config.js";
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
  /** Quality flag (`-ql`/`-qm`/`-qh`/`-q4k`); defaults to `-qm`. */
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
  /**
   * Background colour override as a CSS hex string (e.g. `"#ffffff"`).  When
   * omitted, the scene/camera background is kept.
   */
  background?: string;
  /** Light/dark hint readable from Python as `self.theme` in `setup()`. */
  theme?: PresentationTheme;
  /** Abort a realtime run; the scene stops at the next frame boundary. */
  signal?: AbortSignal;
  onProgress?: (info: { frames: number; plays: number; elapsedMs: number }) => void;
  /**
   * Called with the constructed `Scene` instance right before it starts
   * rendering.  Embedders use this to drive live scenes (e.g. an
   * `InteractiveScene`'s `next_step()` / `restart()`) without importing wanim
   * internals.  The instance is only valid for the duration of the run.
   */
  onSceneReady?: (scene: unknown) => void;
}

export interface BrowserRenderResult {
  stats: CanvasRenderStats;
  /** Recorded frame groups, present when `recordFrames` was set. */
  frames: MJSONFrame[];
  meta: MJSONMeta;
}

/** Monotonic id used to tell several embeds on one page apart. */
let nextPresentationId = 1;

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
  apply_quality(options.quality ?? "-qm");
  const camera = new cameraClass();
  if (options.background !== undefined) {
    camera.background_color = options.background;
  }
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
  if (options.background !== undefined) {
    // The scene adopts the renderer's camera, which the renderer constructs
    // itself; apply the backdrop there too.
    renderer.camera.background_color = options.background;
  }
  const scene = construct_scene(sceneClass, renderer as never);
  const theme = normalize_presentation_theme(options.theme);
  if (theme !== undefined) scene.theme = theme;
  options.onSceneReady?.(scene);
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
  // Attach explicitly as well: a concurrently started run replaces the global
  // scheduler, and this scene must keep its own pacing/cancellation.
  scene.set_run_scheduler(scheduler);
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

export interface BrowserPresentationOptions extends BrowserRenderOptions {
  /**
   * How the presentation is paced.  `normal` (default) pauses only on
   * blueprint pages; `slides` pauses after every phase; `two-view` loops the
   * animation and lets the reader double-click into a live blueprint;
   * `animation` loops the animation only (used for GIF export).
   */
  mode?: PresentationMode | string;
  /** Stable id attached to every `post_message` from this run. */
  instanceId?: string;
  /** Called each time a scene becomes active (fresh or rebuilt). */
  onSceneActivated?: (scene: Scene, index: number, sceneClass: SceneConstructor) => void;
  /** Called once with the navigation controller driving the presentation. */
  onController?: (controller: PresentationController) => void;
  /** Called whenever the presentation position/state changes. */
  onStateChange?: (state: PresentationState | TwoViewState) => void;
  /** Forwarded `Scene.post_message(key, value)` calls. */
  onMessage?: (message: WanimMessage) => void;
  /** Called after one complete animation loop (two-view/animation modes). */
  onLoop?: () => void;
}

/** The navigation surface shared by both players. */
export type PresentationController = ScenePresentation | TwoViewPresentation;

export interface BrowserPresentationResult extends BrowserRenderResult {
  /** Navigation handle: `next()`, `previous()`, `replay()`, `restart()`. */
  controller: PresentationController;
}

/**
 * Play several Scene classes in order on one canvas as a single presentation.
 *
 * This is the multi-scene companion of {@link render_scene_to_canvas}:
 *
 *  - scenes are constructed lazily when they become active and rebuilt when a
 *    navigation re-enters them;
 *  - each presentation owns its scheduler, so several presentations on one
 *    page run and abort independently;
 *  - a DOM host draws the bottom-right navigation icons; the JS handle
 *    (`onController` / the returned `controller`) exposes the same actions.
 */
export async function render_presentation_to_canvas(
  sceneClasses: SceneConstructor[],
  canvas: HTMLCanvasElement,
  options: BrowserPresentationOptions = {},
): Promise<BrowserPresentationResult> {
  if (sceneClasses.length === 0) {
    throw new Error("render_presentation_to_canvas requires at least one Scene class");
  }
  apply_quality(options.quality ?? "-qm");
  const realtime = options.realtime ?? true;
  const scheduler: RenderScheduler | null = realtime
    ? create_frame_scheduler({ fps: options.fps ?? 30 })
    : null;
  const domHost = typeof document !== "undefined" ? new DomInteractiveHost(canvas) : null;
  const host: InteractiveHost = domHost ?? new NullInteractiveHost();
  let controller: PresentationController | null = null;
  const onAbort = (): void => {
    scheduler?.cancel();
    controller?.abort();
  };
  options.signal?.addEventListener("abort", onAbort, { once: true });

  const stats: CanvasRenderStats = {
    frames: 0,
    recordCount: 0,
    extractMs: 0,
    serializeMs: 0,
    drawMs: 0,
    playMs: 0,
    totalMs: 0,
  };
  const frames: MJSONFrame[] = [];
  const harvested = new WeakSet<Scene>();
  let lastCamera: Camera | null = null;
  let numPlays = 0;

  const harvest = (scene: Scene): void => {
    if (harvested.has(scene)) return;
    harvested.add(scene);
    const renderer = scene.renderer as CanvasSceneRenderer | null | undefined;
    if (renderer === null || renderer === undefined) return;
    numPlays += renderer.num_plays;
    frames.push(...(renderer.frames as unknown as MJSONFrame[]));
    for (const key of Object.keys(stats) as Array<keyof CanvasRenderStats>) {
      stats[key] += renderer.stats[key];
    }
    renderer.dispose();
  };

  const mode: PresentationMode = normalize_presentation_mode(options.mode) ?? "normal";
  const theme = normalize_presentation_theme(options.theme);
  const instanceId = options.instanceId ?? `wanim-${nextPresentationId++}`;
  const twoView = mode === "two-view" || mode === "animation";
  // Every scene gets a message sink so `post_message` works in the classic
  // players too; the two-view player installs a richer one itself.
  let activeSceneIndex = 0;
  const forwardMessage = (scene: Scene, key: unknown, value: unknown): void => {
    options.onMessage?.({
      id: instanceId,
      key,
      value,
      scene: scene.constructor.name,
      sceneIndex: activeSceneIndex,
      pageIndex: 0,
      phaseKind: "construct",
    });
  };

  const createScene = (sceneClass: SceneConstructor): Scene => {
    const cameraClass = default_camera_class(sceneClass);
    const camera = new cameraClass();
    if (options.background !== undefined) camera.background_color = options.background;
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
    if (options.background !== undefined) {
      renderer.camera.background_color = options.background;
    }
    const scene = construct_scene(sceneClass, renderer as never);
    if (theme !== undefined) scene.theme = theme;
    // Attach the scheduler explicitly instead of installing a global one: a
    // second presentation on the same page must not steal this run's pacing.
    scene.set_run_scheduler(scheduler);
    if (scene instanceof InteractiveScene) scene.interactive_host = host;
    if (!twoView) scene.message_sink = (key, value) => forwardMessage(scene, key, value);
    lastCamera = renderer.camera;
    options.onSceneReady?.(scene);
    return scene;
  };

  const presentationController: PresentationController = twoView
    ? new TwoViewPresentation(sceneClasses, {
        host,
        mode: mode as "two-view" | "animation",
        theme,
        instanceId,
        createScene,
        disposeScene: (scene) => harvest(scene),
        onSceneActivated: (scene, index, sceneClass) => {
          activeSceneIndex = index;
          options.onSceneActivated?.(scene, index, sceneClass);
        },
        onStateChange: options.onStateChange,
        onMessage: options.onMessage,
        onLoop: options.onLoop,
      })
    : new ScenePresentation(sceneClasses, {
        host,
        slidesMode: mode === "slides",
        createScene,
        disposeScene: (scene) => harvest(scene),
        onSceneActivated: (scene, index, sceneClass) => {
          activeSceneIndex = index;
          options.onSceneActivated?.(scene, index, sceneClass);
        },
        onStateChange: options.onStateChange,
      });
  controller = presentationController;
  options.onController?.(presentationController);
  try {
    await presentationController.run();
  } finally {
    options.signal?.removeEventListener("abort", onAbort);
    domHost?.dispose();
  }
  return {
    stats,
    frames,
    meta: make_meta(lastCamera ?? new Camera(), stats.frames, numPlays),
    controller: presentationController,
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
  const player = new MJSONCanvasPlayer(canvas, { meta, frames }, {
    backend: options.backend,
  });
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
  const player = new MJSONCanvasPlayer(canvas, parse_mjson(text), {
    backend: options.backend,
  });
  return player.play(options);
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
