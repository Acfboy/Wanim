# wanimjs

**Manim Python in the browser, in one package.** `wanimjs` ships the
whole Wanim stack with subpath exports:

| Subpath | What it is |
| --- | --- |
| `wanimjs` | the manim-compatible **core** library (Node ESM + types); also exported as `wanimjs/core` |
| `wanimjs/py2js` | the Python→JS **compiler** (Node-safe: preprocessor + error formatter) |
| `wanimjs/py-runtime` | the Python **runtime** (Node-safe: rapydscript shims, `math`/`random`/`itertools` facades, preload planner) |
| `wanimjs/bridge` | **compiler + runtime + DOM glue** (`runPythonOnCanvas`, `benchInteractiveScene`) — browser-only |
| `wanimjs/browser` | prebuilt browser bundle of the core library (also `./browser/core`) |
| `wanimjs/browser/py2js` | prebuilt browser bundle of the compiler |
| `wanimjs/browser/py-runtime` | prebuilt browser bundle of the runtime |
| `wanimjs/browser/bridge` | prebuilt browser bundle of the bridge (loads its siblings relatively) |
| `wanimjs/browser/rapydscript` | the patched rapydscript-ng compiler (classic `<script>`, sets `globalThis.RapydScript`) |
| `wanimjs/browser/element` | the `<wanim-scene>` custom element (lazy-loads the bridge) |

The bundles are already built; consumers never need esbuild/tsc to run a
scene.  **`numpy-ts` is not bundled** — it is a regular `dependency` of this
package and must be provided by your import map or bundler (see below).
MathJax ships as an on-demand chunk, so TeX works in plain import-map pages
without a bundler.

## Zero-config browser embed

Serve the `browser/` directory (all files, including `chunks/`) and provide
`numpy-ts` yourself — copy `numpy-ts.browser.js` from
`node_modules/numpy-ts/dist/` next to the other files, or map it to a CDN:

```html
<script src="./browser/rapydscript.js"></script>
<script type="importmap">
{
  "imports": {
    "wanim": "./browser/wanim.browser.js",
    "py2js": "./browser/bridge.browser.js",
    "numpy-ts": "./browser/numpy-ts.browser.js"
  }
}
</script>

<canvas id="stage" width="1280" height="720"></canvas>
<script type="module">
import { runPythonOnCanvas } from "py2js";

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
</script>
```

The bridge resolves `./py2js.browser.js`, `./py-runtime.browser.js` and
`./wanim.browser.js` *relatively*, so only the import map entries your own
code uses are required.  Keep the whole `browser/` directory together: the
bridge checks `instanceof wanim.Scene`.

### Providing numpy-ts

- **numpy-ts** is needed for every run (the library's array type), and it
  already ships a browser ESM (`numpy-ts/dist/numpy-ts.browser.js`): copy it
  next to the other files and add the import-map entry above.  Load it before
  wanim's first array operation; `configureWasm({ maxMemory })` is available if
  you need a bigger WASM heap.  Keep a single shared copy — it owns the WASM
  heap.

**MathJax** needs nothing from you: the bundle always carries it as an
on-demand chunk (`browser/chunks/mathjax_runtime-*.js`, fetched only when a
scene actually typesets math), so import-map pages, CDNs and bundlers all work
without an extra dependency to wire up.  Pages that only use shapes or plain
text never download it.

### Optional modules

`MathTex`/`Tex`, the hand-drawn style and text fonts are on-demand:

```js
import { preload_submodules, submodule_status } from "wanim";

await preload_submodules("mathjax");                    // before first TeX use
await preload_submodules(["mathjax", "fonts"]);         // several at once
submodule_status();                                     // { fonts, mathjax }
```

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
Node/SSR). `bridge.browser.js`, `rapydscript.js` and any optional module are
fetched from the element bundle's own directory on first run (the element
still needs `numpy-ts` from the import map, like the bridge).

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

## Node and bundlers

```js
import * as wanim from "wanimjs";        // library
import { MathTex } from "wanimjs";

await wanim.preload_submodules("mathjax");
console.log(new MathTex("a^2 + b^2 = c^2").width > 0);

// Node-safe halves of the Python pipeline (tooling, tests):
import { preprocessImports, formatCompileError } from "wanimjs/py2js";
import { createRuntimeCore, buildPythonMath } from "wanimjs/py-runtime";
```

The browser-only bridge (`runPythonOnCanvas`, `benchInteractiveScene`) lives
under `wanimjs/bridge`, whose `browser` condition resolves to the
prebuilt `browser/bridge.browser.js`; a bundler should keep `numpy-ts` a real
dependency so it resolves normally.  TypeScript declarations for every
subpath are included (`dist/*.d.ts`).

## Files

```
browser/
  wanim.browser.js         library bundle (numpy-ts external)
  py2js.browser.js         Python→JS compiler (rapydscript global)
  py-runtime.browser.js    Python runtime shims (no external imports)
  bridge.browser.js        compiler + runtime + DOM glue (numpy-ts external)
  rapydscript.js           patched rapydscript-ng compiler (BSD-2-Clause)
  wanim-element.browser.js <wanim-scene> custom element
  chunks/                  on-demand MathJax (lazy, loaded on first TeX use)
dist/                      Node ESM entries + TypeScript declarations
src/_config/default.cfg    runtime config read by the Node build
src/templates/             scene templates read by the Node build
```

## Caveats

- Do not mix copies of the bundles: the bridge checks `instanceof
  wanim.Scene`, so the whole `browser/` directory must be served together.
- `numpy-ts` must be a single shared module instance (it owns the WASM heap);
  keep one import-map entry / one bundled copy.
- The Python bridge supports `manim`, `numpy`, `random`, `math`, `itertools`;
  any other Python import raises a friendly error.
- Text fonts default to free CDN fonts (Tinos + Noto Serif SC); call
  `configure_fonts([...])` to self-host.

## License

MIT. Upstream dependencies keep their own licenses (numpy-ts MIT, MathJax
Apache-2.0, opentype.js MIT, rapydscript-ng BSD-2-Clause).
