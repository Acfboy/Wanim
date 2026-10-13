/**
 * Per-run presentation configuration, set from user code through
 * `wanim_config({...})`.
 *
 * The Python bridge injects `wanim_config` as a global (see
 * `py2js/src/browser.ts`); the options object only accepts the keys that are
 * meaningful for a *run* (as opposed to the manim-wide `config`).  Unknown
 * keys raise a `ValueError`-named error so typos surface immediately instead
 * of being silently ignored:
 *
 * ```python
 * wanim_config({"slides_mode": True})
 * ```
 *
 * | Key           | Type | Default | Meaning                                              |
 * | ------------- | ---- | ------- | ---------------------------------------------------- |
 * | `slides_mode` | bool | `False` | Pause after every blueprint *and* construct phase.   |
 *
 * Configuration is per run: each `runPythonOnCanvas` call builds a fresh
 * {@link WanimRunConfig}, so two presentations on one page never share it.
 */

/** Options a `wanim_config({...})` call may set. */
export interface WanimRunConfig {
  /**
   * Presentation mode.  When `false` (default) only blueprint phases pause for
   * user input and everything else plays continuously; when `true` every
   * blueprint/construct phase pauses until the user advances, and the
   * previous-page / replay-page navigation becomes available.
   */
  slides_mode: boolean;
}

/** Keys accepted by {@link apply_wanim_config} (for error messages/tests). */
export const WANIM_RUN_CONFIG_KEYS = ["slides_mode"] as const;

/** The default per-run configuration. */
export function default_wanim_run_config(): WanimRunConfig {
  return { slides_mode: false };
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

/**
 * Best-effort unwrap of the value rapydscript hands to a Python call with
 * keyword arguments: a plain dict (`wanim_config({"slides_mode": True})`)
 * arrives as a plain object, while `wanim_config(slides_mode=True)` may arrive
 * as rapydscript's kwargs bag (`Object.create(null)` with a marker symbol) or
 * as a single `{slides_mode: ...}` bag.  All three expose their entries as
 * enumerable own string keys.
 */
function optionEntries(options: unknown): Array<[string, unknown]> {
  if (options === null || options === undefined) return [];
  if (typeof options !== "object") {
    throw valueError(`wanim_config() expects a dict of options, got ${typeof options}`);
  }
  return Object.entries(options as Record<string, unknown>);
}

/**
 * Apply one `wanim_config({...})` call, merging it into `config`.  Throws a
 * `ValueError` for unknown keys or wrongly typed values.  Returns `config`.
 */
export function apply_wanim_config(config: WanimRunConfig, options: unknown): WanimRunConfig {
  for (const [key, value] of optionEntries(options)) {
    switch (key) {
      case "slides_mode":
        if (typeof value !== "boolean") {
          throw valueError(`wanim_config: slides_mode must be a bool, got ${typeof value}`);
        }
        config.slides_mode = value;
        break;
      default:
        throw valueError(
          `wanim_config: unknown option '${key}' (allowed: ${WANIM_RUN_CONFIG_KEYS.join(", ")})`,
        );
    }
  }
  return config;
}
