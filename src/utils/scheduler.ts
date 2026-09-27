/**
 * Cooperative frame scheduling for real-time rendering.
 *
 * The manim execution model is synchronous: `Scene.play` runs a `for` loop over
 * the animation's time progression and returns only once every frame has been
 * produced.  On the main thread that freezes the page for the whole animation.
 *
 * This module provides the small amount of machinery needed to hand control
 * back to the browser between frames without rewriting the synchronous parts:
 *
 *  - {@link create_frame_scheduler} builds a scheduler whose `next_frame()`
 *    resolves on `requestAnimationFrame`, but only once a target frame interval
 *    (default 30 fps) has elapsed, so logical frames are decoupled from the
 *    display refresh rate.
 *  - {@link set_render_scheduler} installs it globally; the scene machinery
 *    checks {@link is_async_rendering} and, when a scheduler is active, runs the
 *    frame loop as an `async` task that `await`s `next_frame()` after every
 *    drawn frame.  When no scheduler is installed (Node, tests) the loop stays
 *    fully synchronous, so existing callers and the generated `.mjson` are
 *    byte-for-byte unchanged.
 */

/** Anything that can be awaited, used to accept both `this` and `Promise<this>`. */
export type Awaitable<T> = T | Promise<T>;

/** Runtime shape check for promise-like values (duck typed, cross-realm safe). */
export function is_thenable(value: unknown): value is PromiseLike<unknown> {
  return (
    (typeof value === "object" && value !== null) ||
    typeof value === "function"
  ) && typeof (value as { then?: unknown }).then === "function";
}

export interface RenderScheduler {
  /** Always `true`; synchronous/"no-op" schedulers are represented by `null`. */
  readonly async: boolean;
  /** Resolve when it is time to compute the next logical frame. */
  next_frame(): Promise<void>;
  /** Request that the current run stop at the next frame boundary. */
  cancel(): void;
  /** Whether {@link cancel} has been called. */
  readonly cancelled: boolean;
}

export interface FrameSchedulerOptions {
  /** Target logical frame rate; the yield waits at least `1000/fps` ms. */
  fps?: number;
  /**
   * Wait primitive; defaults to `requestAnimationFrame` when available, else a
   * ~16 ms `setTimeout`.  Exposed for tests and non-DOM hosts.
   */
  wait_for_frame?: (callback: (time: number) => void) => void;
}

function now(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

function default_wait_for_frame(callback: (time: number) => void): void {
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(callback);
  } else {
    setTimeout(() => callback(now()), 16);
  }
}

class FrameScheduler implements RenderScheduler {
  readonly async = true;
  cancelled = false;
  private readonly interval: number;
  private readonly wait_for_frame: (callback: (time: number) => void) => void;
  private last_frame_at = now();

  constructor(options: FrameSchedulerOptions = {}) {
    const fps = options.fps ?? 30;
    this.interval = fps > 0 ? 1000 / fps : 0;
    this.wait_for_frame = options.wait_for_frame ?? default_wait_for_frame;
  }

  async next_frame(): Promise<void> {
    if (this.cancelled) return;
    // Wait for the display to present, then skip any extra rAF ticks until the
    // logical frame interval has elapsed.
    for (;;) {
      await new Promise<void>((resolve) => this.wait_for_frame(() => resolve()));
      if (this.cancelled) return;
      const at = now();
      if (at - this.last_frame_at >= this.interval) {
        this.last_frame_at = at;
        return;
      }
    }
  }

  cancel(): void {
    this.cancelled = true;
  }
}

/** Build a real-time frame scheduler. */
export function create_frame_scheduler(options: FrameSchedulerOptions = {}): RenderScheduler {
  return new FrameScheduler(options);
}

let active_scheduler: RenderScheduler | null = null;

/** Install (or clear) the scheduler consulted by {@link is_async_rendering}. */
export function set_render_scheduler(scheduler: RenderScheduler | null): void {
  active_scheduler = scheduler;
}

/** The currently installed scheduler, if any. */
export function get_render_scheduler(): RenderScheduler | null {
  return active_scheduler;
}

/** True when scene playback should yield to the event loop between frames. */
export function is_async_rendering(): boolean {
  return active_scheduler !== null && active_scheduler.async;
}
