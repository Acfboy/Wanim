/**
 * Browser entry point for the `py2js` bridge.
 *
 * The bridge combines the compiler (`compiler.ts` → `py2js.browser.js`) and the
 * runtime (`runtime.ts` → `py-runtime.browser.js`) and adds the DOM pieces that
 * need both:
 *
 *   - {@link runPythonOnCanvas} — take a `<canvas>` element plus Python source
 *     and run the scene against wanim's real-time Canvas2D backend.
 *   - {@link benchInteractiveScene} — benchmark an `InteractiveScene` blueprint
 *     frame by frame.
 *
 * Both `compilePython` (re-exported from the compiler bundle) and the whole
 * runtime surface are re-exported here, so importing the bridge is enough for
 * embedders that only deal with this entry point.
 *
 * This module is bundled by `scripts/build-py2js.mjs` into
 * `dist/bridge.browser.js`, with the sibling bundles, wanim and `numpy-ts` left
 * external so the page's import map (or the relative files next to this one)
 * resolves the very same module instances the canvas backend uses.
 */
import * as wanim from "../../dist/index.js";
import * as np from "numpy-ts";
import { compilePython } from "./compiler.js";
import {
  createRuntimeCore,
  installPythonShims,
  type PythonRuntime,
} from "./runtime.js";
import {
  preloadForSource,
  type RunPythonPreload,
} from "./preload.js";

export * from "./compiler.js";
export * from "./runtime.js";

type Any = any;

const core = createRuntimeCore({
  wanim: wanim as unknown as Record<string, Any>,
  np: np as unknown as Record<string, Any>,
  NDArray: (np as Any).NDArray,
});

export const buildRuntime = core.buildRuntime;
export { wanim };

// --- execution sandbox -----------------------------------------------------
//
// The translated JS is `eval`-ed in a hidden iframe's global scope rather than
// the page's, so the user's `var`/`class` declarations stay isolated and can be
// thrown away between runs.

interface Sandbox {
  frame: HTMLIFrameElement;
  win: Any;
  injected: Record<string, Any>;
}

function createSandbox(runtime: PythonRuntime): Sandbox {
  const frame = document.createElement("iframe");
  frame.style.display = "none";
  document.body.appendChild(frame);
  const win = frame.contentWindow as Any;
  for (const [key, value] of Object.entries(runtime.globals)) {
    try {
      win[key] = value;
    } catch {
      /* read-only globals */
    }
  }
  win.console = console;
  win.globalThis = win;
  win.__name__ = "__main__";
  installPythonShims({
    Object: win.Object,
    Array: win.Array,
    String: win.String,
    Number: win.Number,
    Boolean: win.Boolean,
    BigInt: win.BigInt,
    get ρσ_str() {
      return win.ρσ_str;
    },
  });
  return { frame, win, injected: runtime.globals };
}

function isSceneSubclass(value: unknown): boolean {
  if (typeof value !== "function") return false;
  const proto = (value as { prototype?: unknown }).prototype;
  if (proto === undefined || proto === null || typeof proto !== "object") return false;
  if (proto === (wanim.Scene as unknown as { prototype: object }).prototype) return false;
  return proto instanceof wanim.Scene;
}

function discoverScenes(win: Any, injected: Record<string, Any>): Record<string, new (...a: Any[]) => Any> {
  const found: Record<string, new (...a: Any[]) => Any> = {};
  for (const key of Object.keys(win)) {
    if (key.startsWith("ρσ")) continue;
    const value = win[key];
    if (injected[key] === value) continue;
    if (isSceneSubclass(value)) found[key] = value as new (...a: Any[]) => Any;
  }
  return found;
}

/**
 * Declaration order of `class X(...)` statements in the Python source.  The
 * compiled module also creates window properties in order, but scanning the
 * source guarantees the play order is exactly what the user wrote (helper
 * classes are not `Scene` subclasses and are skipped later).
 */
function sceneDeclarationOrder(source: string): string[] {
  const names: string[] = [];
  const pattern = /^[ \t]*class[ \t]+([A-Za-z_]\w*)[ \t]*(?:\(|:)/gm;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(source)) !== null) names.push(match[1]!);
  return names;
}

/**
 * Scene subclasses in source order.  Only `Scene` subclasses take part in the
 * presentation (helper classes the user defines are ignored); a class that
 * declaration scanning somehow missed keeps its discovery order at the end.
 */
function orderedScenes(
  scenes: Record<string, new (...a: Any[]) => Any>,
  source: string,
): Array<[string, new (...a: Any[]) => Any]> {
  const ordered: Array<[string, new (...a: Any[]) => Any]> = [];
  const seen = new Set<string>();
  for (const name of sceneDeclarationOrder(source)) {
    const sceneClass = scenes[name];
    if (sceneClass === undefined || seen.has(name)) continue;
    seen.add(name);
    ordered.push([name, sceneClass]);
  }
  for (const name of Object.keys(scenes)) {
    if (seen.has(name)) continue;
    ordered.push([name, scenes[name]!]);
  }
  return ordered;
}

/**
 * Build the per-run `wanim_config` global.  Each `runPythonOnCanvas` call gets
 * its own config object, so two presentations on one page never share settings.
 *
 * No `__argnames__`: with one, rapydscript's `ρσ_interpolate_kwargs` would drop
 * the keyword bag (`slides_mode=True`) because no keyword names the `options`
 * slot.  Without it the bag itself arrives as the single argument, and
 * `apply_wanim_config` reads its entries.
 */
function makeWanimConfig(runConfig: Any): Any {
  return (options: Any): Any => (wanim as Any).apply_wanim_config(runConfig, options);
}

/**
 * Await the optional wanim modules (MathJax / fonts) the scene is likely to
 * need, before any scene code runs.  See ./preload.ts for detection.
 */
async function applyPreload(
  source: string,
  options: { preload?: RunPythonPreload },
): Promise<void> {
  await preloadForSource(source, options, {
    mathjax: () => (wanim as Any).preload_submodules("mathjax"),
    fonts: () => (wanim as Any).preload_submodules("fonts"),
  });
}

export interface RunPythonOptions {
  /** manim quality flag controlling resolution / frame rate (default `-qm`). */
  quality?: string;
  /** Render live, yielding to the browser between frames (default `true`). */
  realtime?: boolean;
  /** Target logical frame rate for live playback (default 30). */
  fps?: number;
  drawFrames?: boolean;
  recordFrames?: boolean;
  /** Background colour override (CSS hex); keeps the scene background when omitted. */
  background?: string;
  /**
   * Optional scene name: run only that scene.  By default every `Scene`
   * subclass found plays, in source declaration order (each is a segment of
   * the presentation).
   */
  sceneName?: string;
  /** Called with each scene's name as it becomes active. */
  onScene?: (name: string) => void;
  /**
   * Called with the constructed `Scene` instance right before it starts
   * rendering.  Useful for driving a live `InteractiveScene` from an embedder
   * (`scene.next_step()`, `scene.restart()`); the instance is only valid for
   * the duration of the run.
   */
  onSceneReady?: (scene: unknown) => void;
  /**
   * Called once with the navigation controller (`next()`, `previous()`,
   * `replay()`, `restart()`); the same handle is on the result.  Slides mode
   * comes from `wanim_config({"slides_mode": True})` in the Python source.
   */
  onController?: (controller: unknown) => void;
  /**
   * Optional-module preload plan.  `"auto"` (default) detects MathJax and font
   * needs from the Python source; `"none"` disables preloading; an object
   * picks the modules explicitly.
   */
  preload?: RunPythonPreload;
  signal?: AbortSignal;
}

export interface RunPythonResult {
  /** Name of the first scene played (backward-compatible field). */
  sceneName: string;
  /** Names of every scene in the presentation, in play order. */
  sceneNames: string[];
  stats: Any;
  result: Any;
  /** Navigation handle for the presentation (see {@link RunPythonOptions.onController}). */
  controller: Any;
}

/**
 * Compile `source` and play the `Scene` subclasses onto `canvas`, in source
 * order.  A `wanim_config({...})` call in the source configures the run
 * (currently `slides_mode`).
 *
 * The scene runs in real time: wanim yields to the browser between frames, so
 * a `self.wait()` is an actual pause and the canvas updates live.
 */
export async function runPythonOnCanvas(
  source: string,
  canvas: HTMLCanvasElement,
  options: RunPythonOptions = {},
): Promise<RunPythonResult> {
  const { js, numpyNames } = await compilePython(source);
  await applyPreload(source, options);
  const runtime = buildRuntime();
  for (const name of numpyNames) {
    if (name === "*") Object.assign(runtime.globals, runtime.np);
    else if (name in runtime.np) runtime.globals[name] = runtime.np[name];
  }
  // Per-run presentation configuration, set from Python through
  // `wanim_config({...})`; injected before the sandbox copies the globals.
  const runConfig = (wanim as Any).default_wanim_run_config();
  runtime.globals["wanim_config"] = makeWanimConfig(runConfig);
  const sandbox = createSandbox(runtime);
  try {
    sandbox.win.eval(js);
    // The compiled module defines its own baselib `ρσ_splice` (plain-array
    // `Array.prototype.splice`), which throws on NDArray slice-stores once
    // the getter-only `length` is installed (error 325: `row[:n] = -np.inf`).
    // Overwrite it with the numpy-aware runtime version (plain arrays keep
    // baselib behavior inside `pythonSplice`).
    try {
      const splice = (runtime.globals as Any)?.["ρσ_splice"];
      if (typeof splice === "function") sandbox.win["ρσ_splice"] = splice;
    } catch {
      /* keep the baselib version: single-dim NDArray stores keep failing loudly */
    }
    const scenes = discoverScenes(sandbox.win, sandbox.injected);
    let selected = orderedScenes(scenes, source);
    if (selected.length === 0) throw new Error("No Scene subclass found in the code.");
    if (options.sceneName !== undefined && options.sceneName !== "") {
      const wanted = selected.filter(([name]) => name === options.sceneName);
      if (wanted.length === 0) {
        throw new Error(
          `Scene '${options.sceneName}' not found. Available: ${selected.map(([name]) => name).join(", ")}`,
        );
      }
      selected = wanted;
    }
    const sceneClasses = selected.map(([, sceneClass]) => sceneClass);
    const sceneNames = selected.map(([name]) => name);
    const result = await (wanim as Any).render_presentation_to_canvas(sceneClasses, canvas, {
      quality: options.quality ?? "-qm",
      realtime: options.realtime ?? true,
      fps: options.fps ?? 30,
      drawFrames: options.drawFrames ?? true,
      recordFrames: options.recordFrames ?? false,
      background: options.background,
      signal: options.signal,
      slidesMode: Boolean(runConfig.slides_mode),
      onSceneReady: options.onSceneReady,
      onSceneActivated: (_scene: Any, _index: number, sceneClass: Any) => {
        options.onScene?.(sceneClass.name);
      },
      onController: options.onController,
    });
    return {
      sceneName: sceneNames[0]!,
      sceneNames,
      stats: (result as Any).stats,
      result,
      controller: (result as Any).controller,
    };
  } finally {
    sandbox.frame.remove();
  }
}

// --- interactive performance benchmark -------------------------------------

export interface InteractiveBenchOptions {
  /** manim quality flag controlling resolution / frame rate (default `-qm`). */
  quality?: string;
  /** Measured frames (default 120). */
  frames?: number;
  /** Unmeasured warm-up frames (default 5). */
  warmup?: number;
  /** Optional scene name; defaults to the first InteractiveScene found. */
  sceneName?: string;
  /** Optional-module preload plan (same semantics as `RunPythonOptions.preload`). */
  preload?: RunPythonPreload;
  /**
   * Called immediately before each measured frame, with the scene instance and
   * the frame index.  Scenes expose a `bench_step(index, total)` method to
   * simulate a control change; the harness forwards to it.
   */
  step?: (scene: Any, index: number, total: number) => void;
}

export interface InteractiveBenchResult {
  sceneName: string;
  /** Milliseconds spent in `interactive_frame()` for each measured frame. */
  frameTimes: number[];
  /** Renderer counters accumulated during the measured frames. */
  stats: { frames: number; extractMs: number; drawMs: number };
}

/**
 * Benchmark the per-frame work of an `InteractiveScene` blueprint.
 *
 * The scene is built once (`enter_blueprint`), then each frame calls
 * `interactive_frame()` — exactly the reactions + draw a live host performs
 * while a control is dragged.  No `requestAnimationFrame` pacing is involved:
 * the returned times are pure per-frame CPU cost.
 */
export async function benchInteractiveScene(
  source: string,
  canvas: HTMLCanvasElement,
  options: InteractiveBenchOptions = {},
): Promise<InteractiveBenchResult> {
  const { js, numpyNames } = await compilePython(source);
  await applyPreload(source, options);
  const runtime = buildRuntime();
  for (const name of numpyNames) {
    if (name === "*") Object.assign(runtime.globals, runtime.np);
    else if (name in runtime.np) runtime.globals[name] = runtime.np[name];
  }
  const sandbox = createSandbox(runtime);
  try {
    sandbox.win.eval(js);
    // Same `ρσ_splice` overwrite as in `runPythonOnCanvas` (error 325).
    try {
      const splice = (runtime.globals as Any)?.["ρσ_splice"];
      if (typeof splice === "function") sandbox.win["ρσ_splice"] = splice;
    } catch {
      /* keep the baselib version */
    }
    const scenes = discoverScenes(sandbox.win, sandbox.injected);
    const candidates = Object.entries(scenes).filter(([, ctor]) =>
      isInteractiveSceneClass(ctor),
    );
    if (candidates.length === 0) {
      throw new Error("No InteractiveScene subclass found in the code.");
    }
    const sceneName = options.sceneName ?? candidates[0]![0];
    const entry = candidates.find(([name]) => name === sceneName);
    if (entry === undefined) {
      throw new Error(
        `InteractiveScene '${sceneName}' not found. Available: ${candidates.map(([name]) => name).join(", ")}`,
      );
    }
    const sceneClass = entry[1] as new (...args: Any[]) => Any;

    (wanim as Any).apply_quality(options.quality ?? "-qm");
    const cameraClass = (wanim as Any).default_camera_class(sceneClass);
    const camera = new cameraClass();
    canvas.width = camera.pixel_width;
    canvas.height = camera.pixel_height;
    const ctx = canvas.getContext("2d");
    if (ctx === null) throw new Error("2d canvas context unavailable");
    const renderer = new (wanim as Any).CanvasSceneRenderer("", {
      camera_class: cameraClass,
      ctx,
      drawFrames: true,
      recordFrames: false,
    });
    const scene = (wanim as Any).construct_scene(sceneClass, renderer);
    scene.enter_blueprint();

    const warmup = options.warmup ?? 5;
    for (let index = 0; index < warmup; index += 1) scene.interactive_frame();

    const startFrames = renderer.stats.frames;
    const startExtract = renderer.stats.extractMs;
    const startDraw = renderer.stats.drawMs;
    const total = options.frames ?? 120;
    const frameTimes: number[] = [];
    for (let index = 0; index < total; index += 1) {
      options.step?.(scene, index, total);
      const startedAt = performance.now();
      scene.interactive_frame();
      frameTimes.push(performance.now() - startedAt);
    }
    return {
      sceneName,
      frameTimes,
      stats: {
        frames: renderer.stats.frames - startFrames,
        extractMs: renderer.stats.extractMs - startExtract,
        drawMs: renderer.stats.drawMs - startDraw,
      },
    };
  } finally {
    sandbox.frame.remove();
  }
}

function isInteractiveSceneClass(value: unknown): boolean {
  if (typeof value !== "function") return false;
  const proto = (value as { prototype?: unknown }).prototype;
  if (proto === undefined || proto === null || typeof proto !== "object") return false;
  return proto instanceof (wanim as Any).InteractiveScene;
}
