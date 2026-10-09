# Wanim

Wanim is a **manim-compatible animation library**. Its animation library
(`src/`) is a TypeScript port of [manim](https://www.manim.community/) that has
been modified and extended for the web, with a Canvas2D backend that renders
**live in the browser**.

Thanks to [rapydscript-ng](https://github.com/kovidgoyal/rapydscript-ng), wanim can run **almost unmodified manim
Python**: you drop a normal manim scene onto a `<canvas>` and it plays in real
time.

## Run manim Python on a canvas

```js
import { runPythonOnCanvas } from "py2js";

await runPythonOnCanvas(pythonSource, document.querySelector("canvas"), {
  quality: "-qm",   // 1280x720 @ 30fps
  realtime: true,
});
```

Given this (unmodified) scene:

```python
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        square = Square().rotate(PI / 4)
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
```

`runPythonOnCanvas(...)` compiles it with rapydscript and renders it live — no
pre-rendering to video, the canvas updates while the scene runs.

## `<wanim-scene>` web component

```html
<script type="module" src="./dist/wanim-element.browser.js"></script>

<wanim-scene quality="-qm" autoplay>
  <script type="text/x-python">
from manim import *
class Demo(Scene):
    def construct(self):
        self.play(Create(Circle()))
  </script>
</wanim-scene>
```

Importing the bundle only registers the custom element; `py2js.browser.js`,
`rapydscript.js` and optional modules download from the same `dist/` directory
on first run.  The element supports `code`, `quality`, `render-style`,
`background`, `fps`, `scene`, `autoplay`, `loop`, `src`, `preload-modules` and
`compiler-src`; `run()`/`stop()`/`next()`/`restart()` methods and
`wanim-ready`/`wanim-run`/`wanim-scene`/`wanim-done`/`wanim-error` events.
See `demo/component.html` for live examples and `packaging/wanim/README.md`
for the full attribute table.

## Demo site

```bash
pnpm install
pnpm demo            # builds dist/ if needed, then serves http://localhost:5174/demo/
```

`pnpm demo` opens a small editor: pick an example (or paste your own manim
scene), press **Run**, and watch it play on the canvas.

- `demo/` holds only the page (`index.html`, `demo.js`, `style.css`); the
  browser bundles live in `dist/` (gitignored) and are produced by
  `pnpm build:demo` (`build:browser`, `build:py2js`, `build:rapydscript`).
  `pnpm demo` runs that build automatically when the bundles are missing.
- `demo/embed.html` is the minimal iframe player behind the editor's
  **嵌入 / Embed** button; it reads the compressed code from the URL fragment
  and speaks the postMessage protocol described below.
- `build:rapydscript` clones rapydscript-ng at a pinned commit and applies
  `py2js/patches/rapydscript-ng.patch` — wanim does not vendor it.
- Fonts are **not** stored in the repository. Node uses the host's system font;
  the demo server answers `fonts/*.ttf` with the host's default English font
  (override with `WANIM_FONT=/path/to/font.ttf`).

## Embedding with an iframe (`#c=` fragment)

The editor's **嵌入 / Embed** dialog (and the helpers exported by the library)
pack the Python source into a minimal player page URL, so a blog can paste a
single `<iframe>` tag.  The code travels in the URL **fragment**, which is
never sent to the server:

```js
import { build_embed_hash, build_embed_iframe, decode_embed_code } from "wanim";

const hash = await build_embed_hash(pythonSource, {
  quality: "-qm",       // -ql | -qm | -qh | -q4k
});
// "#c=z.<base64url(deflate-raw)>[&q=-qm]"
const iframe = build_embed_iframe(`https://host.example/embed${hash}`, {
  width: 960,
  height: 540,
});
```

Tokens are `z.<base64url>` (deflate-raw through `CompressionStream`) with a
`u.<base64url>` stored fallback for environments without it; UTF-8 (including
Chinese and emoji) round-trips.  `parse_embed_hash` / `decode_embed_hash` read
a fragment back, and malformed tokens/flags throw a readable error.

The player page speaks a small postMessage protocol and only accepts messages
whose `event.source` is `window.parent`:

| Direction | Payload |
| --- | --- |
| embed → parent | `{ source: "wanim-embed", v: 1, type: "ready" \| "scene" \| "done" \| "error", sceneName?, stats?, error? }` |
| parent → embed | `{ type: "wanim:run" \| "wanim:stop" \| "wanim:set-code", code?, quality? }` |

`wanim:set-code` updates the code (and optional flags) and re-runs only when
the player is already running; follow it with `wanim:run` otherwise.

## Optional modules (browser)

The browser bundle keeps MathJax out of the main file: it is
emitted as content-hashed chunks under `dist/chunks/` and fetched only when
needed, so pages that never typeset TeX never
download them. Serve `dist/` **including `chunks/`** — the bundle imports them
by relative URL.

```js
import { preload_submodules, submodule_status } from "wanim";

// one name, several names, or a de-duplicated array (all is idempotent):
await preload_submodules("mathjax");
await preload_submodules(["mathjax", "fonts"]);
// `fonts` defaults to the active chain; pass paths for a one-off chain:
await preload_submodules("fonts", { fonts: ["/fonts/MyFont.ttf"] });

submodule_status(); // { fonts, mathjax }
```

`runPythonOnCanvas` inspects the Python source and preloads automatically
(`preload: "none"` disables it; an object picks modules explicitly). The TeX
helpers stay synchronous: calling `texToSvg`/building a `MathTex` before the
MathJax submodule is loaded throws with instructions instead of failing
obscurely, and in Node call `await preload_submodules("mathjax")` once at
startup.

## Library (Node)

```bash
pnpm build           # tsc -> dist/
pnpm test            # vitest
pnpm typecheck
pnpm test:py2js      # py2js source preprocessor tests
```

## Packaging (npm)

The publishable packages are assembled from `packaging/` (metadata: package.json
/ README / LICENSE) plus the build artifacts into the gitignored `packages-out/`:

```bash
pnpm build:packages   # rebuild dist/ and assemble wanim-core, wanim-py, wanim
pnpm pack:packages    # assemble from the current dist/ and run npm pack
```

`wanim-core` ships the tsc `dist/` (Node ESM + d.ts) and the browser bundles
under `browser/`; `wanim-py` ships the `--bare`-external py2js bundle (import
map resolves `wanim`/`numpy-ts`) and the Node toolbox; the combined `wanim`
package puts every relative-import bundle in one `browser/` directory for
zero-config use.

## Layout

- `src/` — the manim-compatible library (mobjects, animations, cameras,
  renderers, `.mjson` serialization).
- `src/renderer/canvas*.ts` — the Canvas2D browser backend
  (`render_scene_to_canvas`).
- `py2js/` — Python → JS bridge: preprocessing, runtime shims and the
  rapydscript patch.
- `demo/` — the web editor page (bundles are built into `dist/`).
