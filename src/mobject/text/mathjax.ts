// LaTeX -> SVG conversion via MathJax, replacing the browser-only
// `js.MathJaxTex2Svg` bridge used by the manim-wasm reference implementation.
//
// MathJax runs fully in Node (no DOM): the `liteAdaptor` provides a minimal
// document implementation and `mathjax.document` performs the TeX -> SVG
// conversion. The produced SVG uses `currentColor` for strokes/fills, matching
// MathJax's browser output; callers that want a concrete colour should remap
// that (see SVGMobject / SingleStringMathTex).

import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

export interface TexToSvgOptions {
  display?: boolean;
}

type MathJaxDocument = { convert(tex: string, options?: { display?: boolean }): unknown };

let cachedDocument: MathJaxDocument | null = null;
let cachedAdaptor: ReturnType<typeof liteAdaptor> | null = null;

function getMathJax(): { doc: MathJaxDocument; adaptor: ReturnType<typeof liteAdaptor> } {
  if (cachedDocument === null || cachedAdaptor === null) {
    const adaptor = liteAdaptor();
    RegisterHTMLHandler(adaptor);
    const tex = new TeX({ packages: AllPackages });
    const svg = new SVG({ fontCache: "none" });
    cachedDocument = mathjax.document("", { InputJax: tex, OutputJax: svg }) as unknown as MathJaxDocument;
    cachedAdaptor = adaptor;
  }
  return { doc: cachedDocument, adaptor: cachedAdaptor };
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
