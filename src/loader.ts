/**
 * Optional-dependency loader for the browser build.
 *
 * wanim's heavy third-party dependencies (MathJax for TeX typesetting) are not
 * part of the main bundle; they are emitted as separate chunks and pulled in on
 * demand.  This module is the public, embedder-facing entry point for that:
 *
 *   - {@link preload_submodules} loads one or several named submodules; each
 *     loader caches its pending promise, so repeated and concurrent calls are
 *     idempotent.
 *   - {@link submodule_status} reports which submodules are already in memory.
 *
 * Nothing here downloads anything until a preload function is called, so a
 * page that never uses TeX or web fonts never pays for those chunks.  Font
 * *configuration* is a separate concern: see
 * `configure_fonts` in `mobject/text/pango/svg-renderer.ts`.
 */
import { fontCache, getDefaultFontPaths } from "./mobject/text/pango/index.js";
import { preload_fonts } from "./mobject/text/pango/svg-renderer.js";
import { is_mathjax_loaded, preload_mathjax } from "./mobject/text/mathjax.js";

/** Names accepted by {@link preload_submodules} / {@link submodule_status}. */
export type SubmoduleName = "fonts" | "mathjax";

export interface PreloadSubmodulesOptions {
  /**
   * Font paths for a `"fonts"` load.  When omitted, the active chain is used:
   * the `configure_fonts` override or the environment default (the free CDN
   * pair in the browser).
   */
  fonts?: readonly string[];
}

const SUBMODULE_NAMES: readonly SubmoduleName[] = ["fonts", "mathjax"];

/** Python-style ValueError (a plain `Error` whose name reads "ValueError"). */
function value_error(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

/** Flatten, validate and de-duplicate the requested names (order preserved). */
function normalize_names(names?: SubmoduleName | readonly SubmoduleName[]): SubmoduleName[] {
  if (names === undefined) return [];
  const requested = typeof names === "string" ? [names] : names;
  const unique: SubmoduleName[] = [];
  for (const name of requested) {
    if (!SUBMODULE_NAMES.includes(name)) {
      throw value_error(
        `wanim: unknown submodule ${JSON.stringify(name)}; expected one of ` +
          `${SUBMODULE_NAMES.map((name) => `"${name}"`).join(", ")} ` +
          `(a single name or an array of names)`,
      );
    }
    if (!unique.includes(name)) unique.push(name);
  }
  return unique;
}

/** True when every font of the active chain is parsed and cached. */
function is_fonts_loaded(): boolean {
  const paths = getDefaultFontPaths();
  return paths.length > 0 && paths.every((path) => fontCache.has(path));
}

/**
 * Load the named optional submodules.
 *
 * `names` may be a single name or an array; duplicates load once.  Omitting
 * `names` (or passing an empty array) is a no-op.  Each submodule's loader
 * caches its pending promise, so repeated and concurrent calls share a single
 * load.  A `"fonts"` load uses `options.fonts` when given and the active chain
 * (`configure_fonts` or the default CDN fonts) otherwise.  Unknown names
 * reject with a `ValueError` listing the accepted submodules.
 */
export async function preload_submodules(
  names?: SubmoduleName | readonly SubmoduleName[],
  options: PreloadSubmodulesOptions = {},
): Promise<void> {
  const requested = normalize_names(names);
  await Promise.all(
    requested.map((name) => {
      switch (name) {
        case "mathjax":
          return preload_mathjax();
        case "fonts":
          return preload_fonts(options.fonts);
      }
    }),
  );
}

/** Snapshot of the optional-submodule load state. */
export function submodule_status(): { fonts: boolean; mathjax: boolean } {
  return {
    fonts: is_fonts_loaded(),
    mathjax: is_mathjax_loaded(),
  };
}
