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
- `build:rapydscript` clones rapydscript-ng at a pinned commit and applies
  `py2js/patches/rapydscript-ng.patch` — wanim does not vendor it.
- Fonts are **not** stored in the repository. Node uses the host's system font;
  the demo server answers `fonts/*.ttf` with the host's default English font
  (override with `WANIM_FONT=/path/to/font.ttf`).

## Library (Node)

```bash
pnpm build           # tsc -> dist/
pnpm test            # vitest
pnpm typecheck
pnpm test:py2js      # py2js source preprocessor tests
```

## Layout

- `src/` — the manim-compatible library (mobjects, animations, cameras,
  renderers, `.mjson` serialization).
- `src/renderer/canvas*.ts` — the Canvas2D browser backend
  (`render_scene_to_canvas`).
- `py2js/` — Python → JS bridge: preprocessing, runtime shims and the
  rapydscript patch.
- `demo/` — the web editor page (bundles are built into `dist/`).
