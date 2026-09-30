/**
 * Browser implementation of {@link InteractiveHost}.
 *
 * Wires an `InteractiveScene` to a real `<canvas>`:
 *  - pointer events on the canvas are converted to world-space points using the
 *    scene camera's frame geometry;
 *  - the two control buttons ("restart" / "next") live in a bar below the
 *    canvas. The bar is reused across runs so the buttons never disappear; the
 *    buttons are greyed out (disabled) while an action is unavailable;
 *  - `InputDecimalNumber` editing opens a transparent `<textarea>` laid exactly
 *    over the rendered number, so it looks like the number itself is being
 *    edited: the caret blinks in place and the rendered digits update live.
 *
 * This module only touches `document` inside its methods, so importing it in a
 * non-DOM environment is safe; the class is simply never instantiated there.
 */
import type { Mobject } from "../mobject/mobject.js";
import type {
  InteractiveButtonIcon,
  InteractiveHost,
  KeyHandlers,
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

/**
 * Whether a key press should be swallowed by the numeric inline editor:
 * single printable characters other than a digit, `-` or `.` (letters, ...).
 * Navigation/editing keys and modifier shortcuts are left alone.
 */
function isBlockedNumberKey(event: KeyboardEvent): boolean {
  if (event.ctrlKey || event.metaKey || event.altKey) return false;
  if (event.key.length !== 1) return false;
  return !/[0-9.\-]/.test(event.key);
}

/** One control bar per canvas, so re-runs reuse (never duplicate) the buttons. */
const CONTROLS_BARS = new WeakMap<HTMLCanvasElement, HTMLElement>();

const ENABLED_STYLE = {
  background: "#0d3b66",
  color: "#dbe9f7",
  border: "1px solid #0d3b66",
  cursor: "pointer",
  opacity: "1",
};

const DISABLED_STYLE = {
  background: "#5c6b7a",
  color: "#aeb9c4",
  border: "1px solid #5c6b7a",
  cursor: "not-allowed",
  opacity: "1",
};

/** Small inline SVG icons (24x24 viewBox) for the control buttons. */
const BUTTON_ICONS: Record<InteractiveButtonIcon, string> = {
  restart: "M17.65 6.35A7.95 7.95 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z",
  next: "M8 5v14l11-7z",
};

function applyIcon(button: HTMLButtonElement, icon: InteractiveButtonIcon): void {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "14");
  svg.setAttribute("height", "14");
  svg.style.display = "block";
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", BUTTON_ICONS[icon]);
  path.setAttribute("fill", "currentColor");
  svg.appendChild(path);
  button.appendChild(svg);
}

function ensureStyles(): void {
  if (document.getElementById("wanim-interactive-styles") !== null) return;
  const style = document.createElement("style");
  style.id = "wanim-interactive-styles";
  style.textContent = `
.wanim-interactive-button {
  padding: 6px;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0;
  font: inherit;
  font-weight: 600;
}
.wanim-inline-editor {
  border: 0;
  outline: 0;
  box-shadow: none;
  overflow: hidden;
  white-space: pre;
}
.wanim-inline-editor::selection { background: transparent; }
.wanim-inline-marker {
  pointer-events: none;
  z-index: 1001;
  color: #ffffff;
  font: 700 15px ui-monospace, monospace;
  line-height: 1;
  transform: translateX(-50%);
}
`;
  document.head.appendChild(style);
}

export class DomInteractiveHost implements InteractiveHost {
  readonly live = true;
  private readonly canvas: HTMLCanvasElement;
  private readonly controls: HTMLElement;
  private readonly buttonEls: HTMLButtonElement[] = [];
  private readonly disposers: Array<() => void> = [];
  private editor: HTMLTextAreaElement | null = null;
  private marker: HTMLDivElement | null = null;
  private caretRaf = 0;
  private editorMobject: Mobject | null = null;
  private editorCamera: CameraLike | null = null;
  private editorOptions: TextEditOptions | null = null;
  private closing = false;
  private buttonsEnabled = true;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    ensureStyles();
    const existing = CONTROLS_BARS.get(canvas);
    if (existing !== undefined && existing.isConnected) {
      // Reuse the resident bar; drop the previous run's buttons.
      this.controls = existing;
      this.controls.replaceChildren();
    } else {
      this.controls = document.createElement("div");
      this.controls.className = "wanim-interactive-controls";
      this.controls.style.display = "flex";
      this.controls.style.gap = "6px";
      this.controls.style.zIndex = "10";
      // Overlay the bottom-left corner of the canvas.
      const wrapper = canvas.parentElement;
      if (wrapper !== null) {
        if (getComputedStyle(wrapper).position === "static") wrapper.style.position = "relative";
        this.controls.style.position = "absolute";
        this.controls.style.left = "8px";
        this.controls.style.bottom = "8px";
        wrapper.appendChild(this.controls);
      } else {
        this.controls.style.position = "fixed";
        this.controls.style.left = "8px";
        this.controls.style.bottom = "8px";
        document.body.appendChild(this.controls);
      }
      CONTROLS_BARS.set(canvas, this.controls);
    }
  }

  /**
   * Detach this run's listeners and grey out its buttons, but leave the button
   * bar resident under the canvas (a later run reuses it).
   */
  dispose(): void {
    for (const dispose of this.disposers.splice(0)) {
      try {
        dispose();
      } catch {
        /* ignore */
      }
    }
    this.endTextEdit();
    this.buttonsEnabled = false;
    this.applyButtonStyle();
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

  addButton(label: string, onClick: () => void, icon?: InteractiveButtonIcon): () => void {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "wanim-interactive-button";
    button.title = label;
    button.setAttribute("aria-label", label);
    if (icon !== undefined) applyIcon(button, icon);
    else button.textContent = label;
    button.addEventListener("click", onClick);
    this.controls.appendChild(button);
    this.buttonEls.push(button);
    this.applyButtonStyle();
    // Keep the button resident; only its click handler is detached.
    const dispose = (): void => {
      button.removeEventListener("click", onClick);
    };
    this.disposers.push(dispose);
    return dispose;
  }

  setButtonsEnabled(enabled: boolean): void {
    this.buttonsEnabled = enabled;
    this.applyButtonStyle();
  }

  onKey(handlers: KeyHandlers): () => void {
    const listener = (event: KeyboardEvent): void => {
      if (this.editor !== null) return;
      if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        handlers.next();
      } else if (event.key === "r" || event.key === "R") {
        handlers.restart();
      }
    };
    window.addEventListener("keydown", listener);
    const dispose = (): void => {
      window.removeEventListener("keydown", listener);
    };
    this.disposers.push(dispose);
    return dispose;
  }

  beginTextEdit(mobject: Mobject, camera: unknown, options: TextEditOptions): void {
    this.endTextEdit();
    this.editorMobject = mobject;
    this.editorCamera = camera as CameraLike;
    this.editorOptions = options;
    this.closing = false;

    // A borderless, fully transparent editor laid over the rendered number:
    // the canvas digits show through and update live. The visible caret is our
    // own element (the native one cannot track the rendered glyphs).
    const textarea = document.createElement("textarea");
    textarea.className = "wanim-inline-editor";
    textarea.value = options.initial;
    textarea.rows = 1;
    textarea.style.position = "fixed";
    textarea.style.zIndex = "1000";
    textarea.style.margin = "0";
    textarea.style.padding = "0";
    textarea.style.background = "transparent";
    textarea.style.color = "transparent";
    textarea.style.caretColor = "transparent";
    textarea.style.textAlign = "left";
    textarea.style.resize = "none";
    document.body.appendChild(textarea);

    const marker = document.createElement("div");
    marker.className = "wanim-inline-marker";
    marker.textContent = "^";
    marker.style.position = "fixed";
    document.body.appendChild(marker);

    this.editor = textarea;
    this.marker = marker;
    this.positionEditor();

    textarea.addEventListener("input", () => {
      // Keep the editor numeric-only (letters/non-numeric pastes are dropped),
      // with at most one leading minus sign.
      const raw = textarea.value.replace(/[^0-9.\-]/g, "");
      const negative = raw.includes("-");
      let cleaned = raw.replace(/-/g, "");
      if (negative) cleaned = "-" + cleaned;
      if (cleaned !== textarea.value) {
        textarea.value = cleaned;
        textarea.setSelectionRange(cleaned.length, cleaned.length);
      }
      options.onInput(textarea.value);
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        this.commitEditor();
      } else if (event.key === "Escape") {
        event.preventDefault();
        options.onCancel?.();
        this.endTextEdit();
      } else if (isBlockedNumberKey(event)) {
        // Single printable non-digit keys are ignored so the editor text never
        // drifts away from the rendered number.
        event.preventDefault();
      }
    });
    textarea.addEventListener("blur", () => {
      if (this.closing) return;
      this.commitEditor();
    });
    // Capture phase: fires before the canvas handler so clicking anywhere else
    // commits the edit. It is added *after* the opening pointerdown, so it does
    // not close the editor on the very click that opened it.
    document.addEventListener("pointerdown", this.onDocumentPointerDown, true);

    textarea.focus();
    // Put the insertion point at the end (not a full selection): the `^` marker
    // tracks `selectionStart`, so an end caret reads most naturally.
    const end = textarea.value.length;
    textarea.setSelectionRange(end, end);
    this.caretRaf = requestAnimationFrame(this.caretLoop);
  }

  endTextEdit(): void {
    if (this.caretRaf !== 0) {
      cancelAnimationFrame(this.caretRaf);
      this.caretRaf = 0;
    }
    document.removeEventListener("pointerdown", this.onDocumentPointerDown, true);
    this.editorOptions = null;
    this.editorMobject = null;
    this.editorCamera = null;
    const editor = this.editor;
    if (editor !== null) {
      this.closing = true;
      this.editor = null;
      editor.remove();
    }
    if (this.marker !== null) {
      this.marker.remove();
      this.marker = null;
    }
  }

  /** Commit the open editor (if any) to its owner and close it. */
  private commitEditor(): void {
    const editor = this.editor;
    const options = this.editorOptions;
    if (editor !== null && options !== null) options.onCommit(editor.value);
    this.endTextEdit();
  }

  private readonly onDocumentPointerDown = (event: PointerEvent): void => {
    if (this.editor === null) return;
    if (event.target === this.editor) return;
    this.commitEditor();
  };

  private readonly caretLoop = (): void => {
    if (this.editor === null) return;
    this.positionEditor();
    this.caretRaf = requestAnimationFrame(this.caretLoop);
  };

  /**
   * Keep the transparent editor glued to the rendered number and place the `^`
   * marker under the digit at the current insertion point. Runs every frame
   * while editing so the marker follows both the number's changing width and
   * the text cursor.
   */
  private positionEditor(): void {
    const editor = this.editor;
    const marker = this.marker;
    if (editor === null || marker === null || this.editorMobject === null || this.editorCamera === null) return;
    const rect = this.mobjectClientRect(this.editorMobject, this.editorCamera);
    const height = Math.max(16, rect.height);
    editor.style.left = `${rect.left}px`;
    editor.style.top = `${rect.top}px`;
    editor.style.width = `${Math.max(24, rect.width)}px`;
    editor.style.height = `${height}px`;
    editor.style.font = `${Math.max(12, height * 0.85)}px ui-monospace, monospace`;
    editor.style.lineHeight = `${height}px`;

    const index = editor.selectionStart ?? editor.value.length;
    const anchor = this.inlineAnchor(this.editorMobject, index);
    if (anchor !== null) {
      const client = this.worldToClient(anchor, this.editorMobject, this.editorCamera);
      marker.style.left = `${client.x}px`;
      marker.style.top = `${client.y + 2}px`;
    } else {
      marker.style.left = `${rect.left + rect.width + 2}px`;
      marker.style.top = `${rect.top + rect.height + 2}px`;
    }
  }

  /** Ask the control for its character insertion anchor (world coordinates). */
  private inlineAnchor(mobject: Mobject, index: number): number[] | null {
    const provider = mobject as { interactive_caret_anchor?: (index: number) => number[] | null };
    if (typeof provider.interactive_caret_anchor !== "function") return null;
    try {
      return provider.interactive_caret_anchor(index);
    } catch {
      return null;
    }
  }

  /** Convert a single world point to client (viewport) coordinates. */
  private worldToClient(point: number[], mobject: Mobject, camera: CameraLike): { x: number; y: number } {
    const canvasRect = this.canvas.getBoundingClientRect();
    if (typeof camera.points_to_subpixel_coords !== "function") {
      return { x: canvasRect.left + canvasRect.width / 2, y: canvasRect.top + canvasRect.height / 2 };
    }
    let corners: number[][] = [];
    try {
      corners = pointRows(camera.points_to_subpixel_coords(mobject, [point]));
    } catch {
      corners = [];
    }
    const row = corners[0];
    if (row === undefined) {
      return { x: canvasRect.left + canvasRect.width / 2, y: canvasRect.top + canvasRect.height / 2 };
    }
    const scaleX = this.canvas.width > 0 ? canvasRect.width / this.canvas.width : 1;
    const scaleY = this.canvas.height > 0 ? canvasRect.height / this.canvas.height : 1;
    return {
      x: (row[0] ?? 0) * scaleX + canvasRect.left,
      y: (row[1] ?? 0) * scaleY + canvasRect.top,
    };
  }

  nextFrame(): Promise<void> {
    return new Promise((resolve) => {
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
      else setTimeout(resolve, 16);
    });
  }

  private applyButtonStyle(): void {
    const style = this.buttonsEnabled ? ENABLED_STYLE : DISABLED_STYLE;
    for (const button of this.buttonEls) {
      if (!button.isConnected) continue;
      button.disabled = !this.buttonsEnabled;
      button.style.background = style.background;
      button.style.color = style.color;
      button.style.border = style.border;
      button.style.cursor = style.cursor;
      button.style.opacity = style.opacity;
    }
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
