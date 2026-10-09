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

/** Convert a TeX expression to an SVG string using MathJax's TeX -> SVG pipeline. */
export function texToSvg(expression: string, options: TexToSvgOptions = {}): string {
  const { doc, adaptor } = getMathJax();
  const node = doc.convert(expression, { display: options.display ?? false });
  const html = adaptor.outerHTML(node as never);
  return extractSvg(html);
}

const SVG_CACHE = new Map<string, string>();

/** Cached variant of {@link texToSvg}. */
export function texToSvgCached(expression: string, options: TexToSvgOptions = {}): string {
  const key = `${options.display ? "D" : "I"}:${expression}`;
  const cached = SVG_CACHE.get(key);
  if (cached !== undefined) return cached;
  const svg = texToSvg(expression, options);
  SVG_CACHE.set(key, svg);
  return svg;
}

export { SVG_CACHE as mathjaxSvgCache };
