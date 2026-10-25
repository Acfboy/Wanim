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
 * Theme preference accepted from JS options and HTML attributes.  Concrete
 * values are passed through; `"auto"` asks wanim to read the viewer's
 * `prefers-color-scheme` at run time.
 */
export type PresentationThemePreference = PresentationTheme | "auto";

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

/**
 * Normalize a theme preference coming from a JS option or an HTML attribute.
 * `"light"`/`"dark"` are passed through, `"auto"` (and `"system"`) ask wanim
 * to read the viewer's colour scheme at run time; an unknown/omitted value
 * returns `undefined` so the caller can fall back to `"auto"`.
 */
export function normalize_presentation_theme(
  value: unknown,
): PresentationThemePreference | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.trim().toLowerCase();
  if (text === "light" || text === "dark" || text === "auto") return text;
  if (text === "system") return "auto";
  return undefined;
}

/**
 * Resolve a theme preference to the concrete value a scene sees as
 * `self.theme`.  `"auto"` (and an omitted preference) follow `prefersDark`,
 * which callers read from `matchMedia("(prefers-color-scheme: dark)")` where
 * a DOM exists; headless runs pass `false` and stay on `"light"`.
 */
export function resolve_presentation_theme(
  preference: PresentationThemePreference | undefined,
  prefersDark: boolean,
): PresentationTheme {
  if (preference === "light") return "light";
  if (preference === "dark") return "dark";
  return prefersDark ? "dark" : "light";
}

/** Read the viewer's colour scheme; `false` where there is no DOM to ask. */
export function prefers_dark_colour_scheme(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}
