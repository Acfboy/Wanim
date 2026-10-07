/**
 * Pluggable Canvas2D paint backends.
 *
 * The serialized display list is backend-agnostic: `CanvasSceneRenderer` and
 * the `.mjson` player hand every `VMobject` to a {@link CanvasBackend}, which
 * owns *how* that shape is painted.  The default backend replays the beziers
 * with the plain Canvas2D API (see `canvas.ts`).
 *
 * A backend is stateless with respect to the scene: the camera world->device
 * transform is already installed on the context when `draw` is called, so a
 * backend may either use it directly or reset it and work in device space.
 */
import type { VMobject } from "../mobject/types/vectorized_mobject.js";
import { drawVmobject, type CanvasCamera } from "./canvas.js";

/**
 * A per-canvas VMobject painter.  Implementations must be safe to reuse across
 * frames and scenes; any per-mobject memoisation lives inside the backend.
 */
export interface CanvasBackend {
  /** Stable identifier for the backend. */
  readonly name: string;
  /**
   * Paint one VMobject.  The camera world->device transform is already set on
   * `ctx`; `seed` is an optional stable key (the display-list index) used when
   * the mobject identity cannot be relied on, as in the serialized player.
   */
  draw(ctx: CanvasRenderingContext2D, camera: CanvasCamera, vmobject: VMobject, seed?: number): void;
  /** Release any cached state. */
  dispose(): void;
}

const STANDARD_BACKEND: CanvasBackend = {
  name: "standard",
  draw: (ctx, camera, vmobject) => {
    drawVmobject(ctx, camera, vmobject);
  },
  dispose: () => {},
};

/** The plain Canvas2D backend (no per-frame caching). */
export function standard_backend(): CanvasBackend {
  return STANDARD_BACKEND;
}

/** Create the default plain Canvas2D backend. */
export function create_canvas_backend(): CanvasBackend {
  return STANDARD_BACKEND;
}
