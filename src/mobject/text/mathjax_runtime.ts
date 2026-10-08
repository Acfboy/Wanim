// Heavy MathJax runtime, loaded on demand by `./mathjax.js`.
//
// This module is the *only* place that statically imports `mathjax-full`, so
// the browser bundler emits it as a separate chunk; the main bundle keeps the
// synchronous `texToSvg` facade and only pulls this file in through
// `preload_mathjax()` / py2js' automatic preload.
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

export interface MathJaxDocument {
  convert(tex: string, options?: { display?: boolean }): unknown;
}

export interface MathJaxRuntime {
  doc: MathJaxDocument;
  adaptor: ReturnType<typeof liteAdaptor>;
}

/** Build the TeX -> SVG document exactly once per preload. */
export function create_mathjax_runtime(): MathJaxRuntime {
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({ packages: AllPackages });
  const svg = new SVG({ fontCache: "none" });
  const doc = mathjax.document("", { InputJax: tex, OutputJax: svg }) as unknown as MathJaxDocument;
  return { doc, adaptor };
}
