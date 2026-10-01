# Fonts

The shipped default fonts live in [`assets/fonts`](../../../../../assets/fonts),
in fallback order:

1. `TimesNewRoman.ttf` — Latin/ASCII (and the primary font for metrics).
2. `FZShuSong-Z01S.ttf` — 方正书宋, CJK.

`getTextPaths` walks this chain per character and draws each glyph with the
first font that has one, so mixed text such as `矩阵 A` uses Times for `A` and
FZShuSong for the Chinese characters.

Resolution (`resolveDefaultFontPaths()` in [`../font-loader.ts`](../font-loader.ts)):

- **Node** — reads `assets/fonts/`; if the directory is missing (e.g. an
  installed package without assets) it falls back to the host's default
  English font (DejaVu Sans, Liberation Sans, Noto Sans, Arial).
- **Browser** — fetches `./fonts/<file>` next to the bundle; `build:browser`
  copies the files there. A missing file is skipped, and `DejaVuSans.ttf`
  (this directory's legacy name) is the last-resort fallback for deployments
  that predate the shipped fonts.
- `WANIM_FONT=/path/to/font.ttf` overrides the whole chain with a single file
  (the differential tests pin DejaVu Sans this way).

`font`, `weight`, `slant` and `<span font_family=...>` are still parsed but do
not select a different face: the chain is fixed at load time.
