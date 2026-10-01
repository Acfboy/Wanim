// Differential text tests compare wanim's layout against Python manim rendered
// on this host with DejaVu Sans (`Text(..., font="sans")`).  Pin the legacy
// single font through the same override the demo scripts use, so the shipped
// Times New Roman + FZShuSong chain does not change those reference numbers.
import { resolveDefaultFontPath } from "../src/mobject/text/pango/font-loader.js";

process.env.WANIM_FONT ??= resolveDefaultFontPath();
