/**
 * Scene execution with a live Canvas2D backend, plus a serialized-document
 * player.  This is the browser analogue of `JSONCairoRenderer`: instead of
 * rasterizing into an RGBA pixel array (Python) or serializing only (Node), it
 * can draw the current display list straight onto a `<canvas>`.
 *
 * Two modes:
 *  - `recordFrames`: capture the serialized display list of every frame so it
 *    can be played back smoothly after the (synchronous) scene run.
 *  - `drawFrames`: draw every frame during the run, which is what a
 *    throughput/limit-FPS benchmark needs.  Serialization is skipped.
 */
import type { Mobject } from "../mobject/mobject.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import type { Scene } from "../scene/scene.js";
import {
  clearCanvas,
  drawFrame,
  drawVmobject,
  isThreeDFrame,
  PreviewCamera,
  PreviewThreeDCamera,
  setCanvasTransform,
  backgroundCss,
  type CanvasCamera,
  type MJSONDocument,
  type MJSONFrame,
} from "./canvas.js";
import {
  JSONCairoRenderer,
  serialize_frame,
  type JSONCairoRendererOptions,
} from "./json_renderer.js";
import { is_thenable } from "../utils/scheduler.js";

export interface CanvasSceneRendererOptions extends JSONCairoRendererOptions {
  ctx: CanvasRenderingContext2D;
  /** Draw each frame during the run (default true). */
  drawFrames?: boolean;
  /** Serialize every frame into `frames` for later playback (default false). */
  recordFrames?: boolean;
  /** Called after each recorded/rendered frame, for progress UI. */
  onFrame?: (info: CanvasRenderProgress) => void;
}

export interface CanvasRenderProgress {
  frames: number;
  plays: number;
  elapsedMs: number;
}

export interface CanvasRenderStats {
  /** Total frames emitted (sum of `repeat`). */
  frames: number;
  /** Distinct frame records produced. */
  recordCount: number;
  extractMs: number;
  serializeMs: number;
  drawMs: number;
  /** Time spent inside `Scene.play` (animation interpolation + draw). */
  playMs: number;
  totalMs: number;
}

interface MutableCamera extends CanvasCamera {
  frame_center: readonly number[];
  frame_width: number;
  frame_height: number;
  background_color: { to_hex(): string };
  background_opacity: number;
  get_mobjects_to_display(mobjects: Iterable<Mobject>, includeSubmobjects?: boolean): Mobject[];
  reset_rotation_matrix?(): void;
}

/**
 * Runs a Scene like `JSONCairoRenderer`, but the display list can be drawn to a
 * Canvas2D context and/or recorded as serialized frames.
 */
export class CanvasSceneRenderer extends JSONCairoRenderer {
  readonly ctx: CanvasRenderingContext2D;
  drawFrames: boolean;
  recordFrames: boolean;
  readonly stats: CanvasRenderStats = {
    frames: 0,
    recordCount: 0,
    extractMs: 0,
    serializeMs: 0,
    drawMs: 0,
    playMs: 0,
    totalMs: 0,
  };
  onFrame: ((info: CanvasRenderProgress) => void) | null;
  private _pendingRecord: MJSONFrame | null = null;
  private _startedAt = 0;
  private _started = false;
  private _lastProgress = 0;

  constructor(outputPath: string, options: CanvasSceneRendererOptions) {
    super(outputPath, options);
    this.ctx = options.ctx;
    this.drawFrames = options.drawFrames ?? true;
    this.recordFrames = options.recordFrames ?? false;
    this.onFrame = options.onFrame ?? null;
    // Count scene construction (Tex/MathJax, mobject setup) as part of the
    // run; the renderer is created immediately before the Scene.
    this._startedAt = performance.now();
    this._started = true;
  }

  override play(scene: Scene, ...args: Array<unknown>): void | Promise<void> {
    if (!this._started) {
      this._started = true;
      this._startedAt = performance.now();
    }
    const startedAt = performance.now();
    const result = super.play(scene, ...args as never[]);
    const finish = (): void => {
      this.stats.playMs += performance.now() - startedAt;
      this._emitProgress(true);
    };
    if (is_thenable(result)) {
      return Promise.resolve(result).then(finish);
    }
    finish();
  }

  override update_frame(
    scene: Scene,
    mobjects: Iterable<Mobject> | null = null,
    includeSubmobjects = true,
    ignoreSkipping = true,
    ...kwargs: unknown[]
  ): void {
    if (this.skip_animations && !ignoreSkipping) return;
    const camera = this.camera as unknown as MutableCamera;
    const t0 = performance.now();
    // A ThreeDCamera's z-sorting and projection depend on the rotation matrix,
    // which `capture_mobjects` normally regenerates.  The Canvas backend draws
    // from the display list directly, so regenerate it here.
    if (typeof camera.reset_rotation_matrix === "function") {
      camera.reset_rotation_matrix();
    }
    const displayed = camera.get_mobjects_to_display(scene.mobjects, true) as Mobject[];
    const t1 = performance.now();
    this.stats.extractMs += t1 - t0;

    if (this.drawFrames) {
      const background = backgroundCss({
        background_color: camera.background_color.to_hex(),
        background_opacity: camera.background_opacity,
      });
      clearCanvas(this.ctx, camera, background);
      setCanvasTransform(this.ctx, camera);
      for (const mobject of displayed) {
        if (mobject instanceof VMobject) drawVmobject(this.ctx, camera, mobject);
      }
      this.stats.drawMs += performance.now() - t1;
    }

    if (this.recordFrames) {
      const t2 = performance.now();
      const record = serialize_frame(this.camera, scene.mobjects, displayed) as unknown as MJSONFrame;
      this.stats.serializeMs += performance.now() - t2;
      this._pendingRecord = record;
    }
  }

  /** Count repeats without the (expensive) structural dedup of the JSON renderer. */
  override add_frame(_frame: unknown, numFrames = 1): void {
    const dt = 1 / this.camera.frame_rate;
    if (this.skip_animations) return;
    this.time += numFrames * dt;
    this.stats.frames += numFrames;
    if (this.recordFrames && this._pendingRecord !== null) {
      this._pendingRecord.repeat = numFrames;
      (this.frames as unknown as MJSONFrame[]).push(this._pendingRecord);
      this.stats.recordCount += 1;
      this._pendingRecord = null;
    }
    this._emitProgress(false);
  }

  override scene_finished(scene: Scene): void {
    // A scene that only `add`s mobjects (or adds none) never enters the frame
    // loop, so nothing would be drawn and the preview would stay blank.  Draw
    // the final display list once so a static scene shows its result.
    if (this.num_plays === 0 && this.stats.frames === 0) {
      this.update_frame(scene);
      this.add_frame(null);
    }
    this.stats.totalMs = performance.now() - this._startedAt;
    this._emitProgress(true);
  }

  override clear_screen(): void {
    super.clear_screen();
    this._pendingRecord = null;
    this.stats.frames = 0;
    this.stats.recordCount = 0;
    this.stats.extractMs = 0;
    this.stats.serializeMs = 0;
    this.stats.drawMs = 0;
    this.stats.playMs = 0;
    this._startedAt = performance.now();
    this._started = true;
  }

  private _emitProgress(force: boolean): void {
    if (this.onFrame === null) return;
    const now = performance.now();
    if (!force && now - this._lastProgress < 50) return;
    this._lastProgress = now;
    this.onFrame({
      frames: this.stats.frames,
      plays: this.num_plays,
      elapsedMs: now - this._startedAt,
    });
  }
}

export interface MJSONPlayerOptions {
  fps?: number;
  loop?: boolean;
  /** Called with the 0-based index of every displayed frame. */
  onFrame?: (frameIndex: number) => void;
  /** Called when a non-looping playback reaches the end. */
  onEnd?: () => void;
}

/** Playback handle returned by {@link MJSONCanvasPlayer.play}. */
export class MJSONPlayerHandle {
  stopped = false;
  private _raf = 0;

  constructor(private readonly cancel: () => void) {}

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.cancel();
  }

  /** @internal */
  _setRaf(id: number): void {
    this._raf = id;
  }

  /** @internal */
  _getRaf(): number {
    return this._raf;
  }
}

/**
 * Plays a serialized mjson document on a canvas in (near) real time.  The
 * document's `repeat` counts are expanded so playback matches the original
 * scene timing at `meta.frame_rate`.
 */
export class MJSONCanvasPlayer {
  readonly document: MJSONDocument;
  readonly is3D: boolean;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly camera: PreviewCamera | PreviewThreeDCamera;
  private readonly background: string;
  private readonly flatFrames: MJSONFrame[] = [];

  constructor(canvas: HTMLCanvasElement, document: MJSONDocument) {
    this.canvas = canvas;
    this.document = document;
    const meta = document.meta;
    canvas.width = meta.width;
    canvas.height = meta.height;
    const ctx = canvas.getContext("2d");
    if (ctx === null) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.background = backgroundCss(meta);
    const cameraOptions = {
      pixel_width: meta.width,
      pixel_height: meta.height,
      frame_rate: meta.frame_rate,
      background_color: meta.background_color,
      background_opacity: meta.background_opacity,
    };
    this.is3D = document.frames.some(isThreeDFrame);
    this.camera = this.is3D
      ? new PreviewThreeDCamera(cameraOptions)
      : new PreviewCamera(cameraOptions);
    for (const frame of document.frames) {
      const repeat = Math.max(1, Math.trunc(frame.repeat ?? 1));
      for (let index = 0; index < repeat; index += 1) this.flatFrames.push(frame);
    }
  }

  get totalFrames(): number {
    return this.flatFrames.length;
  }

  /** Draw a single frame (by index into the flattened timeline). */
  renderFrame(index: number): void {
    const frame = this.flatFrames[Math.max(0, Math.min(index, this.flatFrames.length - 1))];
    if (frame === undefined) return;
    drawFrame(this.ctx, this.camera, frame, this.background);
  }

  play(options: MJSONPlayerOptions = {}): MJSONPlayerHandle {
    const fps = options.fps ?? this.document.meta.frame_rate;
    const loop = options.loop ?? true;
    const startedAt = performance.now();
    let handle: MJSONPlayerHandle;
    const tick = (): void => {
      const elapsed = (performance.now() - startedAt) / 1000;
      let index = Math.floor(elapsed * fps);
      if (this.flatFrames.length === 0) {
        handle.stop();
        options.onEnd?.();
        return;
      }
      if (index >= this.flatFrames.length) {
        if (!loop) {
          this.renderFrame(this.flatFrames.length - 1);
          handle.stop();
          options.onEnd?.();
          return;
        }
        index %= this.flatFrames.length;
      }
      this.renderFrame(index);
      options.onFrame?.(index);
      handle._setRaf(requestAnimationFrame(tick));
    };
    handle = new MJSONPlayerHandle(() => cancelAnimationFrame(handle._getRaf()));
    handle._setRaf(requestAnimationFrame(tick));
    return handle;
  }
}
