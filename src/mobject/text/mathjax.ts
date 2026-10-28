// LaTeX -> SVG conversion via MathJax, replacing the browser-only
// `js.MathJaxTex2Svg` bridge used by the manim-wasm reference implementation.
//
// MathJax runs fully in Node (no DOM): the `liteAdaptor` provides a minimal
// document implementation and `mathjax.document` performs the TeX -> SVG
// conversion. The produced SVG uses `currentColor` for strokes/fills, matching
// MathJax's browser output; callers that want a concrete colour should remap
// that (see SVGMobject / SingleStringMathTex).
//
// `mathjax-full` is a large optional dependency: it lives in
// `./mathjax_runtime.js`, which is loaded on demand through
// {@link preload_mathjax}.  The public `texToSvg*` helpers stay synchronous;
// before the runtime is loaded they throw a clear error instead of crashing
// with a missing-module import.
import type { MathJaxRuntime } from "./mathjax_runtime.js";

export interface TexToSvgOptions {
  display?: boolean;
  /**
   * LaTeX environment to wrap the expression in (`align*`, `gather*`, ...).
   * manim's `tex_environment` is a document-level LaTeX environment; MathJax
   * needs an inner math environment instead, and `center`/`flushleft` are not
   * math environments at all.  See {@link normalizeTexEnvironment}.
   */
  environment?: string | null;
}

let runtime: MathJaxRuntime | null = null;
let pending: Promise<void> | null = null;

/** True once {@link preload_mathjax} has finished. */
export function is_mathjax_loaded(): boolean {
  return runtime !== null;
}

/**
 * Load MathJax (the TeX input and SVG output pipelines) into memory.
 *
 * The returned promise is cached: concurrent and repeated calls share one
 * pending load and resolve immediately once MathJax is ready.  Browser users
 * may call this during app startup to pay the download/parse cost up front;
 * `runPythonOnCanvas` calls it automatically for scenes that typeset math.
 */
export function preload_mathjax(): Promise<void> {
  if (pending === null) {
    pending = import("./mathjax_runtime.js")
      .then((mod) => {
        runtime = mod.create_mathjax_runtime();
      })
      .catch((error: unknown) => {
        // Allow a later retry after a transient chunk/network failure.
        pending = null;
        throw error;
      });
  }
  return pending;
}

function getMathJax(): MathJaxRuntime {
  if (runtime === null) {
    throw new Error(
      "wanim: MathJax is not loaded, so TeX mobjects cannot be built yet. " +
        "Call `await preload_submodules(\"mathjax\")` before constructing " +
        "Tex/MathTex/DecimalNumber/Matrix/... . " +
        "`runPythonOnCanvas` does this automatically for scenes whose source uses TeX.",
    );
  }
  return runtime;
}

/** Extract the `<svg>...</svg>` fragment from MathJax's container output. */
function extractSvg(html: string): string {
  const start = html.indexOf("<svg");
  const end = html.lastIndexOf("</svg>");
  if (start === -1 || end === -1) return html;
  return html.slice(start, end + "</svg>".length);
}

/**
 * Map manim's document-level `tex_environment` to a MathJax math environment.
 *
 * manim wraps the whole `TeX` document in the environment (default `align*` for
 * `MathTex`, `center` for `Tex`).  MathJax's `convert()` expects a math
 * expression, so document-level environments cannot be passed through
 * verbatim: `\begin{center}` is a "Misplaced" error, while `align*`/`gather*`
 * work but are better expressed as the inner `aligned`/`gathered` variants.
 * Returns `null` when no wrapper is needed.
 */
export function normalizeTexEnvironment(
  environment: string | null | undefined,
  expression: string,
): string | null {
  if (environment === null || environment === undefined) return null;
  const name = environment.trim();
  if (name === "") return null;
  // An expression that already contains its own environment must not be
  // double-wrapped (users sometimes pass a complete `\begin{...}` block).
  if (/^\s*\\begin\{/.test(expression)) return null;
  // `\cssId{...}{\begin{aligned}...\end{aligned}}` is the form MathTex uses
  // for a single aligned part: the environment is already inside the wrapper.
  if (/^\s*\\cssId\{[^{}]*\}\{\\begin\{/.test(expression)) return null;
  const documentOnly = new Set([
    "center",
    "flushleft",
    "flushright",
    "document",
    "equation",
    "equation*",
  ]);
  if (documentOnly.has(name)) return null;
  if (name === "align" || name === "align*") return "aligned";
  if (name === "gather" || name === "gather*") return "gathered";
  if (name === "multline" || name === "multline*") return "gathered";
  return name;
}

/** Convert a TeX expression to an SVG string using MathJax's TeX -> SVG pipeline. */
export function texToSvg(expression: string, options: TexToSvgOptions = {}): string {
  const { doc, adaptor } = getMathJax();
  const environment = normalizeTexEnvironment(options.environment, expression);
  const source = environment === null
    ? expression
    : `\\begin{${environment}}${expression}\\end{${environment}}`;
  const node = doc.convert(source, { display: options.display ?? false });
  const html = adaptor.outerHTML(node as never);
  return extractSvg(html);
}

const SVG_CACHE = new Map<string, string>();
/** `always_redraw(lambda: MathTex(f"{t}"))` otherwise grows this without bound. */
const SVG_CACHE_LIMIT = 256;

/** Cached variant of {@link texToSvg}. */
export function texToSvgCached(expression: string, options: TexToSvgOptions = {}): string {
  const key = `${options.display ? "D" : "I"}:${options.environment ?? ""}:${expression}`;
  const cached = SVG_CACHE.get(key);
  if (cached !== undefined) return cached;
  const svg = texToSvg(expression, options);
  SVG_CACHE.set(key, svg);
  if (SVG_CACHE.size > SVG_CACHE_LIMIT) {
    const oldest = SVG_CACHE.keys().next().value;
    if (oldest !== undefined) SVG_CACHE.delete(oldest);
  }
  return svg;
}

export { SVG_CACHE as mathjaxSvgCache };
