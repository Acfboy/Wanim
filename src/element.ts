/**
 * `<wanim-scene>` — a zero-config custom element that runs manim Python in any
 * page.
 *
 * Two ways to use it:
 *
 * ```html
 * <!-- 1. Inline: just load this module and put Python in a child script. -->
 * <script type="module" src="/wanim/browser/wanim-element.browser.js"></script>
 * <wanim-scene quality="-qm" render-style="standard" autoplay>
 *   <script type="text/x-python">
 * from manim import *
 * class Demo(Scene):
 *     def construct(self):
 *         self.play(Create(Circle()))
 *   </script>
 * </wanim-scene>
 *
 * <!-- 2. Scripted: assign `.code` and await `.run()`. -->
 * <wanim-scene id="s" quality="-qm" autoplay="false"></wanim-scene>
 * <script type="module">
 *   const el = document.getElementById("s");
 *   el.addEventListener("wanim-done", (e) => console.log(e.detail.stats));
 *   el.code = "from manim import *\n...";
 *   await el.run();
 * </script>
 * ```
 *
 * The element is lazy: importing this module only registers the element.  The
 * Python bridge (`bridge.browser.js`), the rapydscript compiler and any optional
 * module (MathJax / fonts) are fetched on first use from the same
 * directory as this bundle, so the whole `browser/` directory is deployable
 * as-is.
 *
 * The module is also importable under Node/SSR: the element is only defined
 * when `customElements` exists, and everything except DOM APIs is a pure
 * function.
 *
 * Exposed API:
 *
 * | Attribute          | Property            | Notes                                      |
 * | ------------------ | ------------------- | ------------------------------------------ |
 * | `code`             | `.code`             | Python source (also a child script or src) |
 * | `quality`          | `.quality`          | manim quality flag, default `-qm`          |
 * | `background`       | —                   | CSS colour override                        |
 * | `fps`              | —                   | realtime frame rate, default 30            |
 * | `scene`            | `.sceneName`        | scene class name                           |
 * | `src`              | —                   | fetch Python from a URL                    |
 * | `autoplay`         | —                   | default true; `autoplay="false"` disables  |
 * | `loop`             | —                   | re-run when a run finishes                 |
 * | `preload-modules`  | —                   | space list: mathjax fonts                  |
 * | `compiler-src`     | —                   | rapydscript URL override                   |
 *
 * Methods: `run()` (async, resolves with the render stats), `stop()`,
 * `next()` / `previous()` / `replay()` (drive the live presentation),
 * `restart()` (re-runs).  Events (bubbling, composed): `wanim-ready`,
 * `wanim-run`, `wanim-scene` (detail.name), `wanim-done` (detail.stats),
 * `wanim-error` (detail.error).
 */

/** Render statistics returned by `run()` (a subset of wanim's stats). */
export interface WanimSceneStats {
  frames: number;
  totalMs: number;
  extractMs: number;
  drawMs: number;
  [key: string]: unknown;
}

/** Optional modules that can be prefetched through `preload-modules`. */
export type WanimPreloadModule = "mathjax" | "fonts";

/** Minimal shape of the lazily imported `bridge.browser.js` module. */
interface Py2jsModule {
  runPythonOnCanvas(
    source: string,
    canvas: HTMLCanvasElement,
    options: Record<string, unknown>,
  ): Promise<{ sceneName: string; stats: WanimSceneStats; result: unknown }>;
  wanim?: {
    preload_submodules?: (
      names?: WanimPreloadModule | readonly WanimPreloadModule[],
      options?: { fonts?: readonly string[] },
    ) => Promise<unknown>;
  };
}

/** Minimal shape of the presentation controller as seen from the element. */
interface PresentationControllerLike {
  next?: () => void;
  previous?: () => void;
  replay?: () => void;
  restart?: () => void;
  abort?: () => void;
}

const TAG_NAME = "wanim-scene";
const DEFAULT_QUALITY = "-qm";
const DEFAULT_FPS = 30;

/** Canvas backing-store sizes for the quality flags (16:9). */
const QUALITY_SIZES: Record<string, [number, number]> = {
  "-ql": [854, 480],
  "-qm": [1280, 720],
  "-qh": [1920, 1080],
  "-q4k": [3840, 2160],
};

/**
 * `autoplay` is a boolean attribute whose *default* is true: absent means auto
 * play, `autoplay="false"` opts out, any other value means true.
 */
export function resolveAutoplayAttribute(value: string | null): boolean {
  if (value === null) return true;
  return value !== "false";
}

/** `loop` follows regular boolean-attribute semantics. */
export function resolveLoopAttribute(value: string | null): boolean {
  if (value === null) return false;
  return value !== "false";
}

/** Parse and validate a `preload-modules` attribute value. */
export function parsePreloadModules(value: string | null | undefined): WanimPreloadModule[] {
  if (value === null || value === undefined) return [];
  const modules: WanimPreloadModule[] = [];
  for (const token of value.trim().split(/\s+/)) {
    if (token === "") continue;
    if (token !== "mathjax" && token !== "fonts") {
      throw new Error(
        `Unknown preload module "${token}"; expected a space-separated list of: mathjax fonts`,
      );
    }
    if (!modules.includes(token)) modules.push(token);
  }
  return modules;
}

/** Canvas backing-store size for a quality flag (defaults to `-qm`). */
export function qualityCanvasSize(quality: string): [number, number] {
  return QUALITY_SIZES[quality] ?? QUALITY_SIZES[DEFAULT_QUALITY]!;
}

/**
 * Normalize Python source coming from HTML (`<script type="text/x-python">`
 * keeps the surrounding markup indentation and any whitespace-only line before
 * the closing tag) or from a JS string: CRLF is folded to LF, blank edge lines
 * are dropped and the common leading indentation is removed.  A faithfully
 * indented `.py` file is unchanged by this.
 */
export function normalizePythonSource(source: string): string {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  let start = 0;
  while (start < lines.length && lines[start]!.trim() === "") start += 1;
  let end = lines.length;
  while (end > start && lines[end - 1]!.trim() === "") end -= 1;
  const body = lines.slice(start, end);
  let indent: string | null = null;
  for (const line of body) {
    if (line.trim() === "") continue;
    const prefix = /^[ \t]*/.exec(line)![0];
    if (indent === null) {
      indent = prefix;
    } else {
      let length = 0;
      while (length < indent.length && length < prefix.length && indent[length] === prefix[length]) {
        length += 1;
      }
      indent = indent.slice(0, length);
    }
  }
  const dedented =
    indent === null || indent === ""
      ? body
      : body.map((line) => (line.startsWith(indent!) ? line.slice(indent.length) : line));
  return dedented.length > 0 ? `${dedented.join("\n")}\n` : "";
}

/** Human-readable message for an error thrown by a run. */
export function formatWanimError(error: unknown): string {
  if (error instanceof Error) return error.message !== "" ? error.message : error.name;
  return String(error);
}

/** Resolve a file next to this bundle (the element and its siblings). */
function siblingUrl(name: string): string {
  return new URL(name, import.meta.url).href;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

let py2jsPromise: Promise<Py2jsModule> | null = null;

/** Lazy-import the Python bridge shipped next to this bundle (cached). */
function loadPy2js(): Promise<Py2jsModule> {
  if (py2jsPromise === null) {
    const url = siblingUrl("bridge.browser.js");
    py2jsPromise = import(url).then(
      (module) => module as Py2jsModule,
      (error: unknown) => {
        // Let a later run retry the download after a transient failure.
        py2jsPromise = null;
        throw error;
      },
    );
  }
  return py2jsPromise;
}

const compilerLoads = new Map<string, Promise<void>>();

/** Inject the rapydscript compiler `<script>` once; shared by all elements. */
function ensureRapydscript(url: string): Promise<void> {
  if ((globalThis as { RapydScript?: unknown }).RapydScript !== undefined) {
    return Promise.resolve();
  }
  const existing = compilerLoads.get(url);
  if (existing !== undefined) return existing;
  const pending = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => {
      compilerLoads.delete(url);
      reject(new Error(`Failed to load the rapydscript compiler from ${url}`));
    };
    (document.head ?? document.documentElement).appendChild(script);
  });
  compilerLoads.set(url, pending);
  return pending;
}

// In Node/SSR there is no HTMLElement; extending a stub keeps the module
// importable (the class is never upgraded there).
const HTMLElementBase: typeof HTMLElement =
  typeof HTMLElement === "undefined"
    ? (class {} as unknown as typeof HTMLElement)
    : HTMLElement;

const SHADOW_CSS = `
:host { display: block; }
.stage { position: relative; }
canvas {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  background: #000;
  touch-action: none;
}
.status {
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #5c6b7a;
  padding: 4px 2px 0;
  white-space: pre-wrap;
}
.status:empty { display: none; }
.status.error { color: #c62828; }
`;

/** The custom element examined and exercised by the demo page. */
export class WanimSceneElement extends HTMLElementBase {
  static get observedAttributes(): string[] {
    return [
      "code",
      "quality",
      "background",
      "fps",
      "scene",
      "autoplay",
      "loop",
      "preload-modules",
      "src",
      "compiler-src",
    ];
  }

  private _code: string | null = null;
  private _quality: string | null = null;
  private _background: string | null = null;
  private _fps: number | null = null;
  private _sceneName: string | null = null;
  private _autoplay = true;
  private _loop = false;

  private shadow: ShadowRoot | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private statusEl: HTMLElement | null = null;
  private resizeObserver: ResizeObserver | null = null;

  private controller: AbortController | null = null;
  private runPromise: Promise<WanimSceneStats> | null = null;
  private generation = 0;
  private liveController: PresentationControllerLike | null = null;
  private loopTimer: ReturnType<typeof setTimeout> | null = null;
  private autoplayScheduled = false;
  private autoplayStarted = false;

  // --- properties ----------------------------------------------------------

  /** Python source; falls back to a child script or the `src` attribute. */
  get code(): string {
    return this._code ?? this.inlineCode() ?? "";
  }

  set code(value: string) {
    this._code = value;
    this.setAttribute("code", value);
  }

  /** manim quality flag controlling resolution / frame rate. */
  get quality(): string {
    return this._quality ?? DEFAULT_QUALITY;
  }

  set quality(value: string) {
    this._quality = value;
    this.setAttribute("quality", value);
  }

  /** Scene class name to run; defaults to the first scene found. */
  get sceneName(): string {
    return this._sceneName ?? "";
  }

  set sceneName(value: string) {
    this._sceneName = value;
    this.setAttribute("scene", value);
  }

  /** Whether a run is currently in flight (read-only). */
  get running(): boolean {
    return this.controller !== null;
  }

  // --- lifecycle -----------------------------------------------------------

  connectedCallback(): void {
    this.syncFromAttributes();
    this.ensureShadow();
    this.applyCanvasQualitySize();
    this.observeResize();
    this.dispatch("wanim-ready", {});
    if (this.preloadModules().length > 0) {
      void this.loadRequestedModules().catch((error) => {
        this.reportError(error);
      });
    }
    this.scheduleAutoplay();
  }

  disconnectedCallback(): void {
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.clearLoopTimer();
    this.stop();
  }

  attributeChangedCallback(name: string, _oldValue: string | null, value: string | null): void {
    switch (name) {
      case "code":
        this._code = value;
        break;
      case "quality":
        this._quality = value;
        this.applyCanvasQualitySize();
        break;
      case "background":
        this._background = value;
        break;
      case "fps": {
        const parsed = value === null || value === "" ? Number.NaN : Number(value);
        this._fps = Number.isFinite(parsed) ? parsed : null;
        break;
      }
      case "scene":
        this._sceneName = value;
        break;
      case "autoplay":
        this._autoplay = resolveAutoplayAttribute(value);
        break;
      case "loop":
        this._loop = resolveLoopAttribute(value);
        break;
      default:
        break;
    }
  }

  // --- methods -------------------------------------------------------------

  /**
   * Run the scene.  Resolves with the render stats once the scene finishes;
   * rejects on compile/render errors (also reported through `wanim-error`).
   * If a run is already in flight it is stopped first, then restarted.
   */
  async run(): Promise<WanimSceneStats> {
    await this.stopCurrent();
    return this.startRun();
  }

  /** Abort the current run (if any) without waiting for it to settle. */
  stop(): void {
    this.clearLoopTimer();
    const controller = this.controller;
    if (controller === null) return;
    this.controller = null;
    this.runPromise = null;
    this.liveController = null;
    this.generation += 1;
    controller.abort();
    this.setStatus("");
  }

  /**
   * Ask the live presentation to advance to the next page.  Returns `false`
   * when no presentation is running.
   */
  next(): boolean {
    const controller = this.liveController;
    if (controller === null || typeof controller.next !== "function") return false;
    controller.next();
    return true;
  }

  /** Ask the live presentation to go back one page (slides mode). */
  previous(): boolean {
    const controller = this.liveController;
    if (controller === null || typeof controller.previous !== "function") return false;
    controller.previous();
    return true;
  }

  /** Ask the live presentation to replay the current page (slides mode). */
  replay(): boolean {
    const controller = this.liveController;
    if (controller === null || typeof controller.replay !== "function") return false;
    controller.replay();
    return true;
  }

  /** The live presentation controller, if a run is active. */
  get presentation(): PresentationControllerLike | null {
    return this.liveController;
  }

  /** Re-run the scene from the beginning (same as `run()`). */
  restart(): Promise<WanimSceneStats> {
    return this.run();
  }

  // --- internals -----------------------------------------------------------

  private dispatch(name: string, detail: Record<string, unknown>): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  private reportError(error: unknown): void {
    const message = formatWanimError(error);
    this.setStatus(`Error: ${message}`, true);
    console.error(`[${TAG_NAME}] ${message}`, error);
    this.dispatch("wanim-error", { error });
  }

  private syncFromAttributes(): void {
    this._code = this.getAttribute("code");
    this._quality = this.getAttribute("quality");
    this._background = this.getAttribute("background");
    this._sceneName = this.getAttribute("scene");
    this._autoplay = resolveAutoplayAttribute(this.getAttribute("autoplay"));
    this._loop = resolveLoopAttribute(this.getAttribute("loop"));
    const fps = this.getAttribute("fps");
    const parsed = fps === null || fps === "" ? Number.NaN : Number(fps);
    this._fps = Number.isFinite(parsed) ? parsed : null;
  }

  private ensureShadow(): void {
    if (this.canvasEl !== null) return;
    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = SHADOW_CSS;
    const stage = document.createElement("div");
    stage.className = "stage";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("part", "canvas");
    stage.appendChild(canvas);
    const status = document.createElement("div");
    status.className = "status";
    status.setAttribute("part", "status");
    status.setAttribute("role", "status");
    root.append(style, stage, status);
    this.shadow = root;
    this.canvasEl = canvas;
    this.statusEl = status;
  }

  private observeResize(): void {
    if (this.resizeObserver !== null) return;
    if (typeof ResizeObserver === "undefined") {
      this.fitCanvas();
      return;
    }
    this.resizeObserver = new ResizeObserver(() => this.fitCanvas());
    this.resizeObserver.observe(this);
    this.fitCanvas();
  }

  /**
   * Size the canvas' *displayed* box to the host width while the intrinsic
   * (backing-store) size keeps the quality resolution.  The aspect ratio
   * follows the backing store, so the element never letterboxes itself.
   * Called on every host resize (and once after the backing store changes).
   */
  private fitCanvas(): void {
    const canvas = this.canvasEl;
    if (canvas === null) return;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    const [fallbackWidth, fallbackHeight] = qualityCanvasSize(this.quality);
    canvas.style.aspectRatio = `${canvas.width || fallbackWidth} / ${canvas.height || fallbackHeight}`;
  }

  private applyCanvasQualitySize(): void {
    const canvas = this.canvasEl;
    if (canvas === null || this.controller !== null) return;
    const [width, height] = qualityCanvasSize(this.quality);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvas.style.aspectRatio = `${width} / ${height}`;
  }

  private setStatus(text: string, isError = false): void {
    const status = this.statusEl;
    if (status === null) return;
    status.textContent = text;
    status.classList.toggle("error", isError);
  }

  private inlineCode(): string | null {
    const script = this.querySelector('script[type="text/x-python"]');
    const text = script?.textContent;
    return text !== null && text !== undefined && text.trim() !== "" ? text : null;
  }

  private hasSource(): boolean {
    if (this._code !== null && this._code.trim() !== "") return true;
    if (this.inlineCode() !== null) return true;
    const src = this.getAttribute("src");
    return src !== null && src.trim() !== "";
  }

  private async resolveSource(signal: AbortSignal): Promise<string> {
    if (this._code !== null && this._code.trim() !== "") {
      return normalizePythonSource(this._code);
    }
    const inline = this.inlineCode();
    if (inline !== null) return normalizePythonSource(inline);
    const src = this.getAttribute("src");
    if (src !== null && src.trim() !== "") {
      const url = new URL(src, document.baseURI).href;
      const response = await fetch(url, { signal });
      if (!response.ok) {
        throw new Error(`Failed to fetch scene source from ${url}: HTTP ${response.status}`);
      }
      return normalizePythonSource(await response.text());
    }
    throw new Error(
      'no Python source: set `.code`, add a <script type="text/x-python"> child, or a `src` attribute',
    );
  }

  private compilerUrl(): string {
    const override = this.getAttribute("compiler-src");
    if (override !== null && override.trim() !== "") {
      return new URL(override, document.baseURI).href;
    }
    return siblingUrl("rapydscript.js");
  }

  private preloadModules(): WanimPreloadModule[] {
    return parsePreloadModules(this.getAttribute("preload-modules"));
  }

  private async loadRequestedModules(py2js?: Py2jsModule): Promise<void> {
    const modules = this.preloadModules();
    if (modules.length === 0) return;
    const api = (py2js ?? (await loadPy2js())).wanim;
    if (typeof api?.preload_submodules !== "function") {
      throw new Error("preload_submodules is not available in this wanim build");
    }
    await api.preload_submodules(modules);
  }

  private scheduleAutoplay(): void {
    if (this.autoplayScheduled || this.autoplayStarted || !this._autoplay) return;
    this.autoplayScheduled = true;
    const start = (): void => {
      this.autoplayScheduled = false;
      if (this.autoplayStarted || !this.isConnected || !this._autoplay) return;
      // No source yet means the page will assign `.code` and call `run()`.
      if (!this.hasSource()) return;
      this.autoplayStarted = true;
      void this.run().catch(() => {
        /* already reported through wanim-error */
      });
    };
    if (typeof document !== "undefined" && document.readyState === "loading") {
      // The element can be upgraded before its child script is parsed (module
      // scripts run after parsing, but a define() before the markup does not).
      document.addEventListener("DOMContentLoaded", start, { once: true });
    } else {
      queueMicrotask(start);
    }
  }

  private async stopCurrent(): Promise<void> {
    const controller = this.controller;
    if (controller === null) return;
    const pending = this.runPromise;
    this.stop();
    if (pending !== null) {
      try {
        await pending;
      } catch {
        /* an aborted run may reject; the new run supersedes it */
      }
    }
  }

  private async startRun(): Promise<WanimSceneStats> {
    const generation = ++this.generation;
    const controller = new AbortController();
    this.controller = controller;
    this.liveController = null;
    this.clearLoopTimer();
    this.setStatus("Loading…");
    this.dispatch("wanim-run", {});
    const promise = this.execute(generation, controller);
    this.runPromise = promise;
    try {
      return await promise;
    } finally {
      if (this.generation === generation) {
        this.controller = null;
        this.runPromise = null;
      }
    }
  }

  private async execute(
    generation: number,
    controller: AbortController,
  ): Promise<WanimSceneStats> {
    try {
      const source = await this.resolveSource(controller.signal);
      const py2js = await loadPy2js();
      this.setStatus("Compiling…");
      await this.loadRequestedModules(py2js);
      await ensureRapydscript(this.compilerUrl());
      if (generation !== this.generation) return this.emptyStats();
      const canvas = this.canvasEl;
      if (canvas === null) throw new Error("element is not connected");
      const result = await py2js.runPythonOnCanvas(source, canvas, {
        quality: this.quality,
        realtime: true,
        fps: this._fps ?? DEFAULT_FPS,
        drawFrames: true,
        recordFrames: false,
        background: this._background ?? undefined,
        sceneName: this._sceneName ?? undefined,
        signal: controller.signal,
        onScene: (name: string) => {
          if (generation !== this.generation) return;
          this.setStatus(`Running ${name}…`);
          this.dispatch("wanim-scene", { name });
        },
        onController: (controller: unknown) => {
          if (generation === this.generation) {
            this.liveController = controller as PresentationControllerLike;
          }
        },
      });
      if (generation !== this.generation) return this.emptyStats();
      this.liveController = null;
      this.setStatus(`Done · ${result.stats.frames} frames`);
      this.dispatch("wanim-done", { stats: result.stats, sceneName: result.sceneName });
      if (this._loop) this.scheduleLoop();
      return result.stats;
    } catch (error) {
      if (generation === this.generation && !controller.signal.aborted && !isAbortError(error)) {
        this.reportError(error);
      }
      throw error;
    }
  }

  private emptyStats(): WanimSceneStats {
    return { frames: 0, totalMs: 0, extractMs: 0, drawMs: 0 };
  }

  private scheduleLoop(): void {
    this.clearLoopTimer();
    this.loopTimer = setTimeout(() => {
      this.loopTimer = null;
      if (!this._loop || !this.isConnected || this.controller !== null) return;
      void this.run().catch(() => {
        /* reported through wanim-error */
      });
    }, 250);
    // Do not keep a Node process alive for a page-only loop.
    const timer = this.loopTimer as unknown as { unref?: () => void };
    timer.unref?.();
  }

  private clearLoopTimer(): void {
    if (this.loopTimer !== null) {
      clearTimeout(this.loopTimer);
      this.loopTimer = null;
    }
  }
}

/** Register the element (no-op under Node/SSR).  Returns whether it is defined. */
export function registerWanimScene(tagName = TAG_NAME): boolean {
  if (typeof customElements === "undefined") return false;
  if (customElements.get(tagName) !== undefined) return true;
  customElements.define(tagName, WanimSceneElement);
  return true;
}

registerWanimScene();

export default WanimSceneElement;
