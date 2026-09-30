/**
 * Browser implementation of {@link InteractiveHost}.
 *
 * Wires an `InteractiveScene` to a real `<canvas>`:
 *  - pointer events on the canvas are converted to world-space points using the
 *    scene camera's frame geometry;
 *  - the two control buttons ("restart" / "next") are appended below the canvas;
 *  - `InputDecimalNumber` editing opens a small `<textarea>` positioned over the
 *    number.
 *
 * This module only touches `document` inside its methods, so importing it in a
 * non-DOM environment is safe; the class is simply never instantiated there.
 */
import type { Mobject } from "../mobject/mobject.js";
import type {
  InteractiveHost,
  PointerHandlers,
  TextEditOptions,
} from "./host.js";

interface CameraLike {
  pixel_width?: number;
  pixel_height?: number;
  frame_width?: number;
  frame_height?: number;
  frame_center?: readonly number[];
  points_to_subpixel_coords?: (mobject: Mobject, points: unknown) => unknown;
}

function pointRows(value: unknown): number[][] {
  return value as number[][];
}

export class DomInteractiveHost implements InteractiveHost {
  readonly live = true;
  private readonly canvas: HTMLCanvasElement;
  private readonly controls: HTMLElement;
  private readonly disposers: Array<() => void> = [];
  private editor: HTMLTextAreaElement | null = null;
  private closing = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.controls = document.createElement("div");
    this.controls.className = "wanim-interactive-controls";
    this.controls.style.display = "flex";
    this.controls.style.gap = "0.5rem";
    this.controls.style.marginTop = "0.5rem";
    this.controls.style.justifyContent = "center";
    // Place the bar below the canvas *wrapper* when there is one, so a flex
    // layout that centres the canvas still shows the buttons underneath rather
    // than beside it.
    const anchor = canvas.parentElement ?? canvas;
    const anchorParent = anchor.parentElement;
    if (anchorParent !== null) anchorParent.insertBefore(this.controls, anchor.nextSibling);
    else if (canvas.parentElement !== null) canvas.parentElement.appendChild(this.controls);
    else document.body.appendChild(this.controls);
  }

  /** Remove the control bar (the host is per render, so this is a teardown). */
  dispose(): void {
    for (const dispose of this.disposers.splice(0)) {
      try {
        dispose();
      } catch {
        /* ignore */
      }
    }
    this.endTextEdit();
    this.controls.remove();
  }

  pointFromEvent(event: PointerEvent, camera: unknown): number[] {
    const rect = this.canvas.getBoundingClientRect();
    const width = this.canvas.width || rect.width || 1;
    const height = this.canvas.height || rect.height || 1;
    const scaleX = rect.width > 0 ? width / rect.width : 1;
    const scaleY = rect.height > 0 ? height / rect.height : 1;
    const pixelX = (event.clientX - rect.left) * scaleX;
    const pixelY = (event.clientY - rect.top) * scaleY;
    return this.pixelToWorld(pixelX, pixelY, camera as CameraLike);
  }

  onPointer(handlers: PointerHandlers): () => void {
    const down = (event: PointerEvent): void => {
      // Prevent the default focus change: opening the number editor focuses a
      // textarea, and the browser's implicit focus handling on pointerdown
      // would otherwise blur it immediately and close the editor.
      event.preventDefault();
      handlers.down(event);
    };
    const move = (event: PointerEvent): void => {
      handlers.move(event);
    };
    const up = (event: PointerEvent): void => {
      handlers.up(event);
    };
    this.canvas.addEventListener("pointerdown", down);
    // Capture window-level move/up so a drag survives leaving the canvas.
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    const dispose = (): void => {
      this.canvas.removeEventListener("pointerdown", down);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    this.disposers.push(dispose);
    return dispose;
  }

  addButton(label: string, onClick: () => void): () => void {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = "wanim-interactive-button";
    button.style.padding = "0.35rem 0.9rem";
    button.style.borderRadius = "0.4rem";
    button.style.border = "1px solid #3b6ea5";
    button.style.background = "#1f6feb";
    button.style.color = "#fff";
    button.style.cursor = "pointer";
    button.addEventListener("click", onClick);
    this.controls.appendChild(button);
    const dispose = (): void => {
      button.removeEventListener("click", onClick);
      button.remove();
    };
    this.disposers.push(dispose);
    return dispose;
  }

  beginTextEdit(mobject: Mobject, camera: unknown, options: TextEditOptions): void {
    this.endTextEdit();
    const textarea = document.createElement("textarea");
    textarea.value = options.initial;
    textarea.rows = 1;
    textarea.style.position = "fixed";
    textarea.style.zIndex = "1000";
    textarea.style.fontSize = "13px";
    textarea.style.padding = "2px 4px";
    textarea.style.textAlign = "center";
    textarea.style.border = "1px solid #1f6feb";
    textarea.style.borderRadius = "3px";
    textarea.style.background = "#111";
    textarea.style.color = "#fff";
    textarea.style.resize = "none";
    textarea.style.overflow = "hidden";
    const rect = this.mobjectClientRect(mobject, camera as CameraLike);
    textarea.style.left = `${rect.left}px`;
    textarea.style.top = `${rect.top}px`;
    textarea.style.width = `${Math.max(48, rect.width + 24)}px`;
    textarea.style.height = `${Math.max(24, rect.height + 10)}px`;
    document.body.appendChild(textarea);
    this.editor = textarea;
    this.closing = false;

    textarea.addEventListener("input", () => options.onInput(textarea.value));
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        options.onCommit(textarea.value);
        this.endTextEdit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        options.onCancel?.();
        this.endTextEdit();
      }
    });
    textarea.addEventListener("blur", () => {
      if (this.closing) return;
      options.onCommit(textarea.value);
      this.endTextEdit();
    });

    textarea.focus();
    textarea.select();
  }

  endTextEdit(): void {
    const editor = this.editor;
    if (editor === null) return;
    this.closing = true;
    this.editor = null;
    editor.remove();
  }

  nextFrame(): Promise<void> {
    return new Promise((resolve) => {
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
      else setTimeout(resolve, 16);
    });
  }

  private pixelToWorld(pixelX: number, pixelY: number, camera: CameraLike): number[] {
    const pixelWidth = camera.pixel_width ?? this.canvas.width ?? 1;
    const pixelHeight = camera.pixel_height ?? this.canvas.height ?? 1;
    const frameWidth = camera.frame_width ?? 14;
    const frameHeight = camera.frame_height ?? 8;
    const center = camera.frame_center ?? [0, 0, 0];
    const widthMult = pixelWidth / frameWidth;
    const heightMult = -pixelHeight / frameHeight;
    const worldX = (pixelX - pixelWidth / 2) / widthMult + (center[0] ?? 0);
    const worldY = (pixelY - pixelHeight / 2) / heightMult + (center[1] ?? 0);
    return [worldX, worldY, 0];
  }

  private mobjectClientRect(mobject: Mobject, camera: CameraLike): { left: number; top: number; width: number; height: number } {
    const canvasRect = this.canvas.getBoundingClientRect();
    if (typeof camera.points_to_subpixel_coords !== "function") {
      return { left: canvasRect.left + canvasRect.width / 2, top: canvasRect.top + canvasRect.height / 2, width: 0, height: 0 };
    }
    const left = mobject.get_left();
    const right = mobject.get_right();
    const top = mobject.get_top();
    const bottom = mobject.get_bottom();
    let corners: number[][] = [];
    try {
      corners = pointRows(
        camera.points_to_subpixel_coords(mobject, [
          [left[0] ?? 0, top[1] ?? 0, 0],
          [right[0] ?? 0, bottom[1] ?? 0, 0],
        ]),
      );
    } catch {
      corners = [];
    }
    if (corners.length < 2) {
      return { left: canvasRect.left + canvasRect.width / 2, top: canvasRect.top + canvasRect.height / 2, width: 0, height: 0 };
    }
    const scaleX = this.canvas.width > 0 ? canvasRect.width / this.canvas.width : 1;
    const scaleY = this.canvas.height > 0 ? canvasRect.height / this.canvas.height : 1;
    const x1 = (corners[0]?.[0] ?? 0) * scaleX + canvasRect.left;
    const y1 = (corners[0]?.[1] ?? 0) * scaleY + canvasRect.top;
    const x2 = (corners[1]?.[0] ?? 0) * scaleX + canvasRect.left;
    const y2 = (corners[1]?.[1] ?? 0) * scaleY + canvasRect.top;
    return {
      left: Math.min(x1, x2),
      top: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
    };
  }
}
