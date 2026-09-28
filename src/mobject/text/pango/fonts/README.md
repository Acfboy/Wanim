# Default font

wanim does **not** ship a font. `Text`/`MarkupText` need a `.ttf`/`.otf` to
build glyph outlines, so wanim resolves the host machine's default English
sans-serif font instead:

- **Node** — `resolveDefaultFontPath()` in `../font-loader.ts` probes common
  system paths (DejaVu Sans, Liberation Sans, Noto Sans, Arial).
- **Browser** — the bundle fetches `./fonts/DejaVuSans.ttf` next to itself; the
  local demo server (`scripts/demo-server.mjs`) answers that request with the
  host's default English font.

To use a specific font, pass it explicitly (e.g. `Text("hi", font="...")`) or,
for the demo server, set the `WANIM_FONT` environment variable to a font file.
