# wanim

**Manim Python in the browser, in one package.** `wanim` combines the three
pieces of the Wanim stack so a `<canvas>` runs (almost unmodified) manim scenes
with zero bundling and zero configuration:

- [`wanim-core`](https://www.npmjs.com/package/wanim-core) — the manim-compatible
  animation library with a real-time Canvas2D backend;
- [`wanim-py`](https://www.npmjs.com/package/wanim-py) — the patched rapydscript
  Python-to-JS bridge;
- prebuilt browser bundles of both, in a single `browser/` directory whose files
  import each other by relative URL.

## Zero-config browser embed

Serve the `browser/` directory (all files, including `chunks/`) and add:

```html
<script src="./browser/rapydscript.js"></script>
<script type="importmap">
{
  "imports": {
    "py2js": "./browser/py2js.browser.js",
    "wanim": "./browser/wanim.browser.js",
    "numpy-ts": "./browser/numpy-ts.browser.js"
  }
}
</script>

<canvas id="stage" width="1280" height="720"></canvas>
<script type="module">
import { runPythonOnCanvas } from "py2js";

// A plain shapes scene...
await runPythonOnCanvas(
  `from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle().set_fill(PINK, opacity=0.5)
        square = Square().rotate(PI / 4)
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
`,
  document.getElementById("stage"),
  { quality: "-qm" },
);

// ...and TeX works too (MathJax is fetched on demand).
await runPythonOnCanvas(
  `from manim import *

class Formula(Scene):
    def construct(self):
        self.play(Write(MathTex(r"\\int_0^1 x^2\\,dx = \\tfrac{1}{3}")))
`,
  document.getElementById("stage"),
  { quality: "-qm" },
);
</script>
```

Because the bundle pair is addressed relatively, the import map's `wanim` /
`numpy-ts` entries matter only if your own code imports those names; `py2js`
alone is enough for `runPythonOnCanvas`.

### Optional modules

`MathTex`/`Tex`, the hand-drawn style and text fonts are on-demand:

```js
import { preload_submodules, submodule_status } from "wanim";

await preload_submodules("mathjax");                    // before first TeX use
await preload_submodules(["mathjax", "fonts"]);         // several at once
submodule_status();                                     // { fonts, mathjax }
```

The main bundle stays ~1.4 MiB (~0.4 MiB gzip); MathJax (~2 MiB)
is fetched only when needed. Keep `browser/chunks/` on the server.

## `<wanim-scene>` web component

For the smallest possible embed, load the element bundle from the same
`browser/` directory and drop a `<wanim-scene>` into the page — no import map
or page script required:

```html
<script type="module" src="/wanim/browser/wanim-element.browser.js"></script>

<wanim-scene quality="-qm" render-style="standard" autoplay>
  <script type="text/x-python">
from manim import *
class Demo(Scene):
    def construct(self):
        self.play(Create(Circle()))
  </script>
</wanim-scene>
```

Importing the bundle only registers the element (and is a no-op under
Node/SSR). `py2js.browser.js`, `rapydscript.js` and any optional module are
fetched from the element bundle's own directory on first run.

For scripted use, assign `.code` and `await .run()`; all events bubble and
cross shadow boundaries:

```html
<wanim-scene id="s" quality="-qm" autoplay="false"></wanim-scene>
<script type="module">
  const el = document.getElementById("s");
  el.addEventListener("wanim-done", (e) => console.log(e.detail.stats));
  el.addEventListener("wanim-error", (e) => console.error(e.detail.error));
  el.code = "from manim import *\n...";
  await el.run();
</script>
```

| Attribute | Property | Meaning |
| --- | --- | --- |
| `code` | `.code` | Python source (or a `<script type="text/x-python">` child / `src`) |
| `quality` | `.quality` | manim quality flag, default `-qm` |
| `background` | — | CSS colour override |
| `fps` | — | realtime frame rate, default 30 |
| `scene` | `.sceneName` | scene class name to run |
| `src` | — | URL to fetch the Python source from |
| `autoplay` | — | default true; `autoplay="false"` disables |
| `loop` | — | re-run when a run finishes |
| `preload-modules` | — | space-separated `mathjax fonts` to prefetch |
| `compiler-src` | — | rapydscript URL override |

Methods: `run()` (async, resolves with the render stats), `stop()`, `next()`
(advance a live `InteractiveScene`; returns `false` otherwise), `restart()`.
Events: `wanim-ready`, `wanim-run`, `wanim-scene` (`detail.name`),
`wanim-done` (`detail.stats`), `wanim-error` (`detail.error`).

The element renders a `<canvas part="canvas">` plus a status/error line
(`part="status"`) inside its shadow root; see `demo/component.html` in the
repository for a full page.

## Node

```js
import * as wanim from "wanim";      // wanim-core + wanim-py Node API
import { MathTex } from "wanim";

await wanim.preload_submodules("mathjax");
console.log(new MathTex("a^2 + b^2 = c^2").width > 0);
```

`wanim` re-exports `wanim-core` and the Node-safe half of `wanim-py`. The
browser-only functions (`runPythonOnCanvas`, …) live in `browser/` and are not
part of the Node entry point. Use the individual packages for a smaller
dependency graph.

## Files

```
browser/
  wanim.browser.js         wanim-core bundle
  numpy-ts.browser.js      numpy-ts (MIT), shared WASM-backed array runtime
  py2js.browser.js         Python bridge (relative imports into this directory)
  rapydscript.js           patched rapydscript-ng compiler (BSD-2-Clause)
  wanim-element.browser.js <wanim-scene> custom element (lazy bridge loading)
  chunks/                  MathJax (Apache-2.0) on demand
dist/                    Node ESM entry (re-exports wanim-core + wanim-py)
```

## Caveats

- Do not mix copies of the bundles: `py2js.browser.js` checks
  `instanceof wanim.Scene`, so the whole `browser/` directory must be served
  together.
- The Python bridge supports `manim`, `numpy`, `random`, `math`, `itertools`;
  any other Python import raises a friendly error.
- Text fonts default to free CDN fonts (Tinos + Noto Serif SC); call
  `configure_fonts([...])` to self-host.
- See the sub-package READMEs for the full API and Node details.

## License

MIT. Bundled upstream code keeps its own license (numpy-ts MIT, MathJax
Apache-2.0, opentype.js MIT, rapydscript-ng BSD-2-Clause).
