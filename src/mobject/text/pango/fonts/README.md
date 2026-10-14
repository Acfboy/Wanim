# Fonts

The default font chain has one font per script; the first font with a glyph
for a character wins.

## Browser

Free CDN fonts, pinned to immutable upstream commits (`DEFAULT_FONT_URLS` in
[`../font-loader.ts`](../font-loader.ts)):

1. [Tinos](https://cdn.jsdelivr.net/gh/google/fonts@9710da1eacb3be272583c3224dcb70f9da6eadbb/ofl/tinos/Tinos-Regular.ttf)
   (Apache-2.0, 0.5 MB) — metric-compatible with Times New Roman, for Latin/ASCII.
2. [Noto Serif SC, `chinese-simplified`
   subset](https://cdn.jsdelivr.net/npm/@fontsource/noto-serif-sc@4.5.12/files/noto-serif-sc-chinese-simplified-400-normal.woff)
   (OFL, Source Han Serif, 2.3 MB WOFF1) — Song/Ming face; ~8.1k glyphs,
   covering common simplified and traditional characters. The full
   [Noto Serif SC OTF](https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@f8d157532fbfaeda587e826d4cd5b21a49186f7c/Serif/SubsetOTF/SC/NotoSerifSC-Regular.otf)
   (11.6 MB) has broader coverage (CJK Extension B, more symbols) if needed.

Embedders override the chain at startup, before creating any text mobject:

```ts
import { configure_fonts } from "wanim";

configure_fonts([
  "https://cdn.example.com/Times.ttf", // Latin
  "https://cdn.example.com/Song.otf",  // CJK
]);
```

URLs must be CORS-enabled (jsDelivr sends `Access-Control-Allow-Origin: *`) and
TTF/OTF (opentype.js does not read WOFF2). `configure_fonts(null)` restores the
built-in chain.

## Node

`resolveDefaultFontPaths()` renders with the host's default English font
(DejaVu Sans, Liberation Sans, Noto Sans, Arial; see `SYSTEM_FONT_CANDIDATES`
in [`../font-loader.ts`](../font-loader.ts)). `WANIM_FONT=/path/to/font.ttf`
overrides the whole chain with a single file (the differential tests pin DejaVu
Sans this way). The browser chain works in Node too: preload the URLs once
(e.g. `preload_submodules("fonts", { fonts: DEFAULT_FONT_URLS })`) and the
synchronous render path then reads the cached fonts.

## Other

`font`, `weight`, `slant` and `<span font_family=...>` are parsed but do not
select a different face: the chain is fixed at load time.
