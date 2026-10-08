// Differential text tests compare wanim's layout against Python manim rendered
// on this host with DejaVu Sans (`Text(..., font="sans")`).  Pin the legacy
// single font through the same override the demo scripts use, so the shipped
// Times New Roman + FZShuSong chain does not change those reference numbers.
import { resolveDefaultFontPath } from "../src/mobject/text/pango/font-loader.js";
import { preload_mathjax } from "../src/mobject/text/mathjax.js";

process.env.WANIM_FONT ??= resolveDefaultFontPath();

// MathJax is an on-demand chunk in the browser build; in Node the suite builds
// MathTex/DecimalNumber/... synchronously, so load it once before any test
// runs.  (Browser pages get the same behavior through py2js' auto-preload.)
await preload_mathjax();
