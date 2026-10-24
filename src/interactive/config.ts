/**
 * Host-facing presentation options.
 *
 * Pacing is owned by the embedder: the JavaScript API's `mode` option and the
 * `<wanim-scene mode="...">` attribute decide how a scene is played.  Python
 * code no longer configures it (the old `wanim_config` helper was removed in
 * 0.1.2).
 */

/** How a presentation is paced. */
export type PresentationMode = "normal" | "slides" | "two-view" | "animation";

/** Colour theme supplied by the embedder. */
export type PresentationTheme = "light" | "dark";

/**
 * Normalize a mode coming from a JS option or an HTML attribute.  Both
 * `two-view` and `two_view` are accepted; an unknown/omitted value returns
 * `undefined` so the caller can use `"normal"`.
 */
export function normalize_presentation_mode(value: unknown): PresentationMode | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.trim().toLowerCase().replace(/_/g, "-");
  if (text === "normal" || text === "default" || text === "interactive") return "normal";
  if (text === "slides" || text === "slide" || text === "slideshow") return "slides";
  if (text === "two-view" || text === "twoview" || text === "two") return "two-view";
  if (text === "animation" || text === "loop" || text === "animated") return "animation";
  return undefined;
}

/** Normalize a light/dark theme; unknown values return `undefined`. */
export function normalize_presentation_theme(value: unknown): PresentationTheme | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.trim().toLowerCase();
  if (text === "light" || text === "dark") return text;
  return undefined;
}
