/**
 * Optional-dependency loader for the browser build.
 *
 * wanim's heavy third-party dependencies (MathJax for TeX typesetting) are not
 * part of the main bundle; they are emitted as separate chunks and pulled in on
 * demand.  This module is the public, embedder-facing entry point for that:
 *
 *   - {@link preload_mathjax} loads MathJax and caches the pending promise
 *     (idempotent, safe to call concurrently).
 *   - {@link preload_modules} combines it with the existing font preloader in a
 *     single call.
 *   - {@link modules_status} / {@link is_module_loaded} let embedders and tests
 *     check what is already in memory.
 *
 * Nothing here downloads anything until a preload function is called, so a
 * page that never uses TeX never pays for those chunks.
 */
import {
  fontCache,
  getDefaultFontPaths,
  preload_fonts,
} from "./mobject/text/pango/index.js";
import { is_mathjax_loaded, preload_mathjax } from "./mobject/text/mathjax.js";

/** Names accepted by {@link is_module_loaded} / {@link modules_status}. */
export type OptionalModuleName = "mathjax" | "fonts";

export interface ModulesStatus {
  /** MathJax (TeX -> SVG) runtime loaded. */
  mathjax: boolean;
  /** Every font of the active chain is parsed and cached (see `preload_fonts`). */
  fonts: boolean;
}

export interface PreloadModulesOptions {
  /** Load MathJax (for `Tex`/`MathTex`/`DecimalNumber`/`Matrix`/...). */
  mathjax?: boolean;
  /**
   * Load fonts: `true` uses the active chain (`configure_fonts` or the default
   * CDN fonts), an array overrides it for this call.
   */
  fonts?: boolean | readonly string[];
}

/** True when the named optional module has finished loading. */
export function is_module_loaded(name: OptionalModuleName): boolean {
  switch (name) {
    case "mathjax":
      return is_mathjax_loaded();
    case "fonts": {
      const paths = getDefaultFontPaths();
      return paths.length > 0 && paths.every((path) => fontCache.has(path));
    }
    default: {
      const unknown: never = name;
      throw new Error(`wanim: unknown optional module ${String(unknown)}`);
    }
  }
}

/** Snapshot of the optional-module load state (fonts, MathJax). */
export function modules_status(): ModulesStatus {
  return {
    mathjax: is_module_loaded("mathjax"),
    fonts: is_module_loaded("fonts"),
  };
}

/**
 * Load the requested optional modules in parallel.
 *
 * `preload_modules({ mathjax: true, fonts: true })` is the one-call equivalent
 * of the individual preload functions plus `preload_fonts()`.  Unset/false
 * flags are no-ops.
 */
export async function preload_modules(options: PreloadModulesOptions = {}): Promise<void> {
  const tasks: Promise<unknown>[] = [];
  if (options.mathjax) tasks.push(preload_mathjax());
  if (options.fonts !== undefined && options.fonts !== false) {
    tasks.push(preload_fonts(typeof options.fonts === "boolean" ? undefined : options.fonts));
  }
  await Promise.all(tasks);
}

export { preload_fonts, preload_mathjax };
