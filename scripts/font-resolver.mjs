// Locate a default English sans-serif font on the host machine.
//
// wanim's `Text`/`MarkupText` need a `.ttf`/`.otf` to build glyph outlines, but
// the font is deliberately *not* stored in the repository.  In the browser the
// demo server serves whatever this resolves at `<bundle dir>/fonts/*.ttf`.
import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";

const CANDIDATES = [
  process.env.WANIM_FONT,
  // Linux (DejaVu is also what manim/Pango picks by default)
  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
  "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
  "/usr/share/fonts/TTF/DejaVuSans.ttf",
  // macOS
  "/System/Library/Fonts/Supplemental/Arial.ttf",
  "/Library/Fonts/Arial.ttf",
  // Windows
  "C:\\Windows\\Fonts\\arial.ttf",
  "C:\\Windows\\Fonts\\segoeui.ttf",
];

/** Absolute path to a usable default font, or `null` if none was found. */
export function resolveDefaultFontFile() {
  for (const candidate of CANDIDATES) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  // Fall back to fontconfig, which is present on most Linux desktops.
  try {
    const out = execFileSync("fc-match", ["-f", "%{file}", "sans-serif"], {
      encoding: "utf-8",
    }).trim();
    if (out && existsSync(out) && !out.endsWith(".ttc")) return out;
  } catch {
    /* fc-match unavailable */
  }
  return null;
}
