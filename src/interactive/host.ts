/**
 * Host abstraction for interactive scenes.
 *
 * `InteractiveScene` owns the interaction *logic* (hit testing, control state,
 * the frame loop) but must stay renderer/DOM agnostic so it can be driven under
 * Node (tests, serialization) as well as from a real `<canvas>` in the browser.
 * A host supplies the small amount of environment-specific behaviour:
 *
 *  - mapping pointer events to world-space points (browser: canvas + camera);
 *  - creating the overlay buttons below the canvas;
 *  - opening an inline text editor for `InputDecimalNumber`;
 *  - pacing the interaction loop (`requestAnimationFrame` in the browser).
 *
 * {@link NullInteractiveHost} is the headless implementation: it never yields,
 * so a scene run with it performs a single blueprint/construct pass and then
 * finishes (the behaviour Node's `run_scene` relies on).
 */
import type { Mobject } from "../mobject/mobject.js";

/**
 * Callbacks an {@link InteractiveHost} invokes for pointer activity. The raw
 * event is handed back so the scene can convert it to world coordinates via
 * {@link InteractiveHost.pointFromEvent} with its own camera.
 */
export interface PointerHandlers {
  down: (event: PointerEvent) => void;
  move: (event: PointerEvent) => void;
  up: (event: PointerEvent) => void;
  /**
   * Optional: the pointer left the interactive surface (e.g. the canvas).
   * Scenes use it to drop hover highlights. Hosts without a pointer-leave
   * concept may omit it.
   */
  leave?: (event: PointerEvent) => void;
}

/** Options for an inline text edit session opened by the host. */
export interface TextEditOptions {
  /** Initial text shown in the editor. */
  initial: string;
  /**
   * Whether the numeric editor accepts a decimal point. Defaults to `true`;
   * integer inputs pass `false` so `.` is stripped/blocked.
   */
  allow_decimal_point?: boolean;
  /** Called on every keystroke with the current editor text. */
  onInput: (text: string) => void;
  /** Called when the editor is committed (Enter/blur). */
  onCommit: (text: string) => void;
  /** Called when the editor is dismissed without committing (Escape). */
  onCancel?: () => void;
}

/** Callbacks for the scene-level keyboard shortcuts. */
export interface KeyHandlers {
  /** Advance to the next story phase. */
  next: () => void;
  /** Restart the whole story. */
  restart: () => void;
}

/** Icon names the host knows how to draw (fallback is the text label). */
export type InteractiveButtonIcon = "restart" | "next";

/** Environment services an {@link InteractiveScene} needs. */
export interface InteractiveHost {
  /** `true` when the host can service continuous interaction (a browser). */
  readonly live: boolean;
  /** Convert a pointer event to a world-space point using `camera`. */
  pointFromEvent(event: PointerEvent, camera: unknown): number[];
  /**
   * Screen (client) pixels per world unit for pointer motion, used by controls
   * that translate drag distances into values. Optional: when omitted, the
   * controls fall back to the camera's backing-store pixels, which is exact
   * only when the canvas is displayed at its natural size. A DOM host whose
   * canvas is CSS-scaled should implement this so a 100-px pointer drag always
   * measures 100 screen pixels.
   */
  pixelsPerWorldUnit?(camera: unknown): [number, number];
  /** Register pointer callbacks; returns a disposer. */
  onPointer(handlers: PointerHandlers): () => void;
  /**
   * Register a wheel/scroll callback; returns a disposer.  Optional so
   * existing hosts stay valid: scenes that care about zoom only wire it when
   * the host implements it (a headless host has no wheel to deliver).
   */
  onWheel?(handler: (event: WheelEvent) => void): () => void;
  /** Add an overlay button; returns a disposer that detaches its click handler. */
  addButton(label: string, onClick: () => void, icon?: InteractiveButtonIcon): () => void;
  /**
   * Enable/disable the overlay buttons. Buttons are greyed out while an action
   * is unavailable (e.g. while an animation plays); they are removed entirely
   * when the run is torn down (see `dispose` on the DOM host).
   */
  setButtonsEnabled(enabled: boolean): void;
  /** Register keyboard shortcuts (space = next, r = restart). */
  onKey(handlers: KeyHandlers): () => void;
  /** Open an inline editor anchored to `mobject`. */
  beginTextEdit(mobject: Mobject, camera: unknown, options: TextEditOptions): void;
  /** Close any open inline editor. */
  endTextEdit(): void;
  /** Resolve when the next interaction frame may run. */
  nextFrame(): Promise<void>;
}

/** Headless host: no DOM, no pacing; drives a single pass. */
export class NullInteractiveHost implements InteractiveHost {
  readonly live = false;
  pointFromEvent(): number[] {
    return [0, 0, 0];
  }
  onPointer(_handlers: PointerHandlers): () => void {
    return () => {};
  }
  onWheel(_handler: (event: WheelEvent) => void): () => void {
    return () => {};
  }
  addButton(_label: string, _onClick: () => void, _icon?: InteractiveButtonIcon): () => void {
    return () => {};
  }
  setButtonsEnabled(_enabled: boolean): void {}
  onKey(_handlers: KeyHandlers): () => void {
    return () => {};
  }
  beginTextEdit(_mobject: Mobject, _camera: unknown, _options: TextEditOptions): void {}
  endTextEdit(): void {}
  nextFrame(): Promise<void> {
    return Promise.resolve();
  }
}
