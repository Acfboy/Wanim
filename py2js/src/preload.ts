/**
 * Preload planning for the py2js browser bridge.
 *
 * wanim's heavy optional dependencies (MathJax, fonts) are loaded on demand.
 * `runPythonOnCanvas` inspects the user's Python source to decide what a scene
 * will need before it runs, so the first `MathTex(...)` never hits an unloaded
 * module.  Detection is intentionally over-inclusive: downloading MathJax for
 * a scene that ends up not using it is harmless, while missing it fails at
 * construction time.
 *
 * Kept dependency-free so it can be unit-tested without a DOM.
 */

/** `preload` option accepted by `runPythonOnCanvas`. */
export type RunPythonPreload =
  | "auto"
  | "none"
  | { mathjax?: boolean; fonts?: boolean };

export interface PreloadNeeds {
  mathjax: boolean;
  fonts: boolean;
}

/**
 * Python identifiers that route through MathJax, directly or through an
 * internal `MathTex` (`get_axis_labels`, `include_numbers`, ...).  Deliberately
 * broad: see the module comment.
 */
const MATHJAX_PATTERN =
  /\b(?:MathTex|SingleStringMathTex|Tex|BulletedList|Title|DecimalNumber|Integer|Variable|Matrix|IntegerMatrix|Table|IntegerTable|MathTable|Brace|BraceBetweenPoints|BraceLabel|BraceText|Label|LabeledLine|LabeledArrow|LabeledPolygram|get_axis_labels|get_coordinate_labels|get_graph_label|include_numbers|numbers_to_include|InputDecimalNumber)\b/;

/** Python identifiers that build Pango/opentype text (not TeX). */
const FONTS_PATTERN = /\b(?:Text|MarkupText|Paragraph|BraceText|register_font)\b/;

/** Detect which optional modules `source` is likely to need. */
export function detectPreloadNeeds(source: string): PreloadNeeds {
  return {
    mathjax: MATHJAX_PATTERN.test(source),
    fonts: FONTS_PATTERN.test(source),
  };
}

/**
 * Resolve the user-facing `preload` option into concrete needs.
 *
 * `"auto"` (the default) runs {@link detectPreloadNeeds}; `"none"` disables
 * preloading entirely (useful when the page manages modules itself); an object
 * enables exactly the listed modules.
 */
export function resolvePreloadNeeds(
  preload: RunPythonPreload | undefined,
  source: string,
): PreloadNeeds {
  if (preload === "none") return { mathjax: false, fonts: false };
  if (preload === undefined || preload === "auto") {
    return detectPreloadNeeds(source);
  }
  return {
    mathjax: preload.mathjax ?? false,
    fonts: preload.fonts ?? false,
  };
}

/** The optional-module loaders, injectable for tests/embedders. */
export interface PreloadFunctions {
  mathjax: () => Promise<unknown>;
  fonts: () => Promise<unknown>;
}

/**
 * Await exactly the loaders the resolved plan asks for.
 *
 * Auto-detected font loads are best-effort (wanim's synchronous fallback still
 * loads fonts on first use); everything the caller listed explicitly — and all
 * MathJax loads, including auto-detected ones — propagates failures.
 * Returns the resolved needs, mostly for tests and diagnostics.
 */
export async function preloadForSource(
  source: string,
  options: { preload?: RunPythonPreload },
  loaders: PreloadFunctions,
): Promise<PreloadNeeds> {
  const spec = options.preload;
  if (spec === "none") return { mathjax: false, fonts: false };
  const auto = spec === undefined || spec === "auto";
  const needs = resolvePreloadNeeds(spec, source);
  const tasks: Promise<unknown>[] = [];
  if (needs.mathjax) tasks.push(loaders.mathjax());
  if (needs.fonts) {
    const fonts = loaders.fonts();
    tasks.push(auto ? fonts.catch(() => undefined) : fonts);
  }
  await Promise.all(tasks);
  return needs;
}
