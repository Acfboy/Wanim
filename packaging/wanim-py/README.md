# wanim-py

Run **(almost unmodified) manim Python in the browser**. `wanim-py` is the
Python-to-JavaScript bridge for [wanim-core](https://www.npmjs.com/package/wanim-core):

```
manim Python --rapydscript--> JavaScript --wanim-core--> live <canvas>
```

It ships the browser bridge (`compilePython`, `runPythonOnCanvas`), the patched
[rapydscript-ng](https://github.com/kovidgoyal/rapydscript-ng) compiler, and a
DOM-free Node entry point with the preprocessor / runtime toolbox.

## Browser quick start

Copy or serve the `browser/` directory of this package, but resolve `wanim` and
`numpy-ts` against the page's wanim runtime through an import map. All three
entries must resolve to the *same* module instances wanim itself uses:

```html
<!-- 1. the patched compiler defines window.RapydScript (classic script) -->
<script src="./browser/rapydscript.js"></script>

<!-- 2. map the bare specifiers used by the bridge -->
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
  { quality: "-qm" }, // 1280x720 @ 30fps, rendered live
);
</script>
```

Or use both wanim packages with a bundler: `npm install wanim-py wanim`
(`wanim` is an optional peer providing the runtime). Bundlers that follow the
`browser` export condition resolve `import { runPythonOnCanvas } from
"wanim-py/browser"` and `"wanim"` without an import map. If you only installed
`wanim-core`, alias `wanim` to `wanim-core/browser` instead.

`runPythonOnCanvas` compiles the source, preloads the optional modules it
detects (MathJax / fonts — see `preload` in the options;
`preload: "none"` disables it) and renders the first `Scene` subclass in real
time. The translated code runs in a hidden iframe so repeated runs do not leak
globals.

### API (browser)

| Export | Purpose |
| --- | --- |
| `compilePython(source)` | preprocess + rapydscript → `{ js, numpyNames }` |
| `runPythonOnCanvas(source, canvas, options?)` | compile and render live |
| `benchInteractiveScene(source, canvas, options?)` | per-frame benchmark for interactive scenes |
| `buildRuntime()`, `installPythonShims()` | runtime shims (also useful for custom hosts) |

Requires `globalThis.RapydScript` (load `browser/rapydscript.js` first); a clear
error is thrown otherwise.

## Node

The package root is the DOM-free entry point. It exposes the Python source
preprocessor, the compile-error formatter, the optional-module preload planner
and the runtime shim builders — useful for tooling and tests. It has **no
runtime dependencies**:

```js
import { preprocessImports, detectPreloadNeeds, createRuntimeCore } from "wanim-py";

const { code, numpyNames, unsupported } = preprocessImports(pythonSource);
```

The browser-only functions (`compilePython`, `runPythonOnCanvas`, …) are **not**
exported from the Node entry; import them from `wanim-py/browser` in a browser
context (or use the prebuilt `browser/py2js.browser.js`).

## How it works

The Python is compiled by a patched rapydscript compiler that (a) routes
arithmetic through `ρσ_op_*` helpers so Python operator overloading survives,
and (b) rewrites `self.play(...)` / `self.wait(...)` into `await self...(...)`
so real-time rendering is transparent. `runtime_core.js` wraps wanim's ES
classes so rapydscript can subclass them, and implements NumPy (via `numpy-ts`),
`random`, `math` and `itertools` facades.

## Bundled third-party code

| File | Upstream | License |
| --- | --- | --- |
| `browser/rapydscript.js` | [rapydscript-ng](https://github.com/kovidgoyal/rapydscript-ng) @ `818f187` + the wanim patch | BSD-2-Clause |

## Caveats

- `wanim` / `numpy-ts` must resolve to the same instances the canvas backend
  uses: the runner discovers scenes with `prototype instanceof wanim.Scene`.
  Do not load two copies of the bundles.
- The Python runtime is deliberately small: `manim`, `numpy`, `random`, `math`
  and `itertools` imports are supported; any other import raises a friendly
  error (the rest of the Python standard library is not available).
- The compiler is a ~1.4 MB script; it is loaded lazily by the demo, but this
  package does not do that for you.

## License

MIT. Bundled upstream code keeps its own license (see above).
