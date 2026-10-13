# wanim

**manim, live in the browser.** wanim is a manim-compatible animation engine
for the web. It takes almost unmodified [manim](https://www.manim.community/)
Python, compiles it to JavaScript inside the page, and plays it on a
`<canvas>` in real time: no server, no LaTeX installation, no pre-rendered
video.

```python
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle().set_fill(PINK, opacity=0.5)
        square = Square().rotate(PI / 4)
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
```

That scene runs as written in a page, and it can also be extended with
interactive controls (sliders, switches, editable numbers and matrices) that
drive the picture while the reader plays with them.

## Features

- **Stock manim API.** `Scene`, mobjects, `self.play`, `.animate`,
  `ValueTracker`, `Transform`, `MathTex`, `ThreeDScene`, and the rest of the
  common surface keep their Python names and keyword arguments.
- **Real-time playback.** Scenes run frame by frame on a Canvas2D backend and
  yield to the browser between frames, so `self.wait()` is an actual pause.
- **Interactive scenes.** `InteractiveScene` separates a live *blueprint*
  (controls and reactions that run every frame) from a *construct* phase (an
  ordinary animation script) that the reader advances to. Interactive 3D
  scenes support orbit and zoom.
- **Math and text without LaTeX.** `MathTex` and `Tex` are typeset with
  MathJax, which ships as an on-demand chunk. Text uses open-licensed fonts.
- **Embeddable.** A `<wanim-scene>` custom element for plain HTML, a
  code-in-URL iframe for blogs and CMSs, and a JavaScript API for custom
  players and editors.
- **Modular packaging.** One npm package with subpath exports: the library,
  the Python compiler, the runtime and the browser bridge can be imported
  separately.

## Installation

```bash
npm install wanimjs
# or
pnpm add wanimjs
```

Requirements: a browser with `CompressionStream` support for the embed helpers
(current Chrome, Edge, Firefox and Safari), and Node.js 18 or newer when you
use the package from Node.

The package has one runtime dependency that you may need to wire up yourself
in a browser without a bundler: `numpy-ts`, the array library that wanim's
`numpy` shim is built on. Bundlers resolve it automatically. See
[Browser setup](#browser-setup) for the import-map variant.

## Quick start

### In a bundled app (Vite, webpack, esbuild, Rollup)

```js
import { runPythonOnCanvas } from "wanimjs/bridge";

const source = `
from manim import *

class Hello(Scene):
    def construct(self):
        self.play(Write(Text("Hello, wanim!")))
        self.wait(0.5)
`;

await runPythonOnCanvas(source, document.querySelector("canvas"), {
  quality: "-qm",   // 1280 x 720 at 30 fps
  realtime: true,
});
```

### With the `<wanim-scene>` element

The element needs no page script. Serve the package's `browser/` directory as
static files (see [Browser setup](#browser-setup)), map `numpy-ts` in an import
map, and put Python inside the tag:

```html
<script type="importmap">
{
  "imports": {
    "numpy-ts": "/vendor/numpy-ts.browser.js"
  }
}
</script>
<script type="module" src="/vendor/wanimjs/browser/wanim-element.browser.js"></script>

<wanim-scene quality="-qm">
  <script type="text/x-python">
from manim import *

class Demo(Scene):
    def construct(self):
        self.play(Create(Circle()))
  </script>
</wanim-scene>
```

### From Node

The Node entry points cover the library itself and the parts of the Python
pipeline that do not need a DOM:

```js
import { Circle, Square, PI } from "wanimjs";
import { preprocessImports } from "wanimjs/py2js";
```

Rendering to a canvas requires a browser. `wanimjs/bridge` is browser-only.

## Package entry points

| Import | What you get | Environment |
| --- | --- | --- |
| `wanimjs` (also `wanimjs/core`) | The manim-compatible library: mobjects, animations, scenes, cameras, `build_embed_hash` and friends | Node, bundlers |
| `wanimjs/py2js` | Python → JavaScript compiler helpers (preprocessing, error formatting) | Node-safe |
| `wanimjs/py-runtime` | Python runtime pieces: rapydscript shims, `math` / `random` / `itertools` facades, preload planner | Node-safe |
| `wanimjs/bridge` | `runPythonOnCanvas` and `benchInteractiveScene`: compiler, runtime and DOM glue in one | Browser |
| `wanimjs/browser` (also `wanimjs/browser/core`) | Prebuilt browser bundle of the library | Browser |
| `wanimjs/browser/py2js` | Prebuilt browser bundle of the compiler | Browser |
| `wanimjs/browser/py-runtime` | Prebuilt browser bundle of the runtime | Browser |
| `wanimjs/browser/bridge` | Prebuilt browser bundle of the bridge | Browser |
| `wanimjs/browser/rapydscript` | The patched rapydscript-ng compiler; a classic script that sets `globalThis.RapydScript` | Browser |
| `wanimjs/browser/element` | The `<wanim-scene>` custom element | Browser |

Bundlers pick the browser build automatically through the `browser` export
condition. TypeScript declarations are included for every entry point.

## Browser setup

wanim's browser bundles reference each other with relative URLs, so the
`browser/` directory of the package must be served as one unit, including its
`chunks/` subdirectory.

**Without a bundler**, copy `browser/` and `numpy-ts.browser.js` (from
`node_modules/numpy-ts/dist/`) next to each other, then map the bare
specifiers with an import map:

```html
<script src="./browser/rapydscript.js"></script>
<script type="importmap">
{
  "imports": {
    "wanim": "./browser/wanim.browser.js",
    "numpy-ts": "./numpy-ts.browser.js",
    "wanimjs/bridge": "./browser/bridge.browser.js"
  }
}
</script>

<canvas id="stage" width="1280" height="720"></canvas>
<script type="module">
  import { runPythonOnCanvas } from "wanimjs/bridge";

  await runPythonOnCanvas(
    `from manim import *
class Hello(Scene):
    def construct(self):
        self.play(Create(Circle()))
`,
    document.getElementById("stage"),
    { quality: "-qm" },
  );
</script>
```

`rapydscript.js` must be a classic script loaded before the first run. It
defines the compiler global that the bridge uses.

**Keep one copy of `numpy-ts`.** It owns the WebAssembly heap, so the page must
never load two instances. Use one import-map entry or one bundled copy.

To give numpy-ts a larger heap, call `configureWasm({ maxMemory })` from
`numpy-ts` before any array operation.

### Fonts

Text is rendered with open-licensed fonts (Tinos for Latin, Noto Serif SC for
Chinese). Without configuration, wanim loads them from a public CDN. To avoid
depending on a third-party origin, self-host the font files and register them:

```js
import { configure_fonts } from "wanimjs";

configure_fonts(["/fonts/Tinos-Regular.ttf", "/fonts/NotoSerifSC-Regular.woff"]);
```

Call this before the first scene runs.

### Optional modules

MathJax and the font chain are loaded on demand. `runPythonOnCanvas` detects
which ones a scene needs and preloads them. You can also control this yourself:

```js
import { preload_submodules, submodule_status } from "wanimjs";

await preload_submodules("mathjax");
await preload_submodules(["mathjax", "fonts"]);
submodule_status(); // { fonts: ..., mathjax: ... }
```

Pass `preload: "none"` to `runPythonOnCanvas` to skip automatic preloading, or an
object such as `{ mathjax: true, fonts: false }` to choose modules explicitly.
Pages that never typeset math never download MathJax.

## Writing scenes

wanim runs ordinary manim scenes. A file may contain several `Scene`
subclasses. They play in source order as one presentation, and helper classes
that are not scenes are ignored.

### Supported Python imports

- `from manim import *` (and `import manim`)
- `numpy` (mapped to numpy-ts), `random`, `math`, `itertools`

Any other import raises a readable error when the scene is compiled. The
rest of the Python standard library is not available in the browser.

### Interactive scenes

`InteractiveScene` splits a scene into two methods:

- `blueprint()` builds the picture and wires up controls. It stays live:
  every `add_reaction` callback runs on every frame, so the picture follows
  the reader's input.
- `construct()` runs when the reader presses **Next**. It is an ordinary
  animation script, and `add_updater` works here as usual.

```python
from manim import *

class Interactive(InteractiveScene):
    def blueprint(self):
        axes = Axes(x_range=[-1, 8, 1], y_range=[-2, 2, 1], x_length=8, y_length=4)
        circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.6)

        slider = Slider(
            start=axes.c2p(1, -1.6),
            end=axes.c2p(7, -1.6),
            value_range=(0.4, 2.5),
            init_value=1.0,
            color=YELLOW,
        )
        circle.add_reaction(lambda m: m.set_width(slider.get_value()))

        switch = Switch(True).to_corner(UR, buff=0.8)
        circle.add_reaction(lambda m: m.set_color(GREEN if switch.get_value() else RED))

        self.add(axes, circle, slider, switch)

    def construct(self):
        hint = Text("Drag the slider, flip the switch", font_size=24)
        self.play(FadeIn(hint.to_edge(DOWN)), run_time=0.8)
        self.wait(0.5)
```

Available controls:

| Control | Interaction | `get_value()` returns |
| --- | --- | --- |
| `Slider(start, end, value_range, init_value)` | drag along a line segment | `float` |
| `PathSlider(path, value_range, is_closed)` | drag along any curve or closed loop | `float` |
| `Switch(value)` | click to toggle | `bool` |
| `InputDecimalNumber(value, num_decimal_places)` | click to type; drag up or down to scrub | `float` |
| `InputInteger(value)` | click to type; drag up or down to scrub | `int` |
| `InputDecimalMatrix([[...], ...])` | edit any cell; the layout adjusts | cell `get_value()` |
| `InputDecimalTable([[...], ...])` | edit any cell; grid lines rebuild | cell `get_value()` |

Controls are ordinary mobjects. You can position them, animate them, and
attach other mobjects to a slider's knob with `attach()`.

Scene-level reactions work too: `self.add_reaction(fn)` runs once per frame
with the scene as its argument, which is useful for refreshing a group of
objects from several controls.

Keep reactions cheap, because they run every frame. For numeric readouts,
prefer `DecimalNumber.set_value(...)` over rebuilding a `Text` in each frame.

### Multi-step stories

`set_storyboard([...])` (alias `set_script`) takes a list of
`[blueprint, construct]` pairs. The reader advances through them one at a time:

```python
class Story(InteractiveScene):
    def setup(self):
        self.set_storyboard([
            [self.pick_value, self.show_value],
            [self.pick_size, self.show_size],
        ])
```

A `None` in either slot skips that half. A single `blueprint()` and
`construct()` pair needs no storyboard.

### Slides mode

By default, a scene pauses only at the start of its blueprint phase, and each
`construct` plays through to the next scene. For lecture-style material, turn
on slides mode at the top of the file:

```python
wanim_config({"slides_mode": True})
```

In slides mode every blueprint and construct phase ends on a pause. The
navigation controls go back a page, replay the current segment, or go forward.
When the reader returns to a blueprint, the control values they set are
remembered.

### 3D

`ThreeDScene` and `InteractiveThreeDScene` work as in manim. In an idle
blueprint, the reader can drag to orbit the camera and scroll to zoom.
Labels and controls that must stay upright on screen belong in
`add_fixed_in_frame_mobjects`.

## Navigation and keyboard

Scenes that pause show three buttons in the bottom-right corner of the
canvas: previous page, replay this segment, and next page.

| Key | Action |
| --- | --- |
| `Space` or `→` | next page |
| `←` | previous page |
| `↑` | replay this segment |

Shortcuts are ignored while the reader is typing into a number input. With
several scenes on one page, shortcuts act on the canvas that was clicked last.

Programmatic navigation is available through the controller that
`runPythonOnCanvas` returns, or through the `onController` callback:

```js
const { controller } = await runPythonOnCanvas(source, canvas, {
  onController: (c) => { /* keep a reference */ },
});

controller.next();
controller.previous();
controller.replay();
controller.restart();
```

## Embedding

### `<wanim-scene>`

The element supports these attributes:

| Attribute | Property | Meaning |
| --- | --- | --- |
| `code` | `.code` | Python source. Also read from a child `<script type="text/x-python">` or from `src` |
| `src` | — | URL of a `.py` file to fetch |
| `quality` | `.quality` | `-ql`, `-qm` (default), `-qh` or `-q4k` |
| `fps` | — | target frame rate, default `30` |
| `scene` | `.sceneName` | which scene class to run, when the file defines several |
| `background` | — | CSS colour that overrides the scene background |
| `autoplay` | — | on by default; `autoplay="false"` disables it |
| `loop` | — | run again when a run finishes |
| `preload-modules` | — | space-separated list: `mathjax fonts` |
| `compiler-src` | — | URL of a different `rapydscript.js` |

Methods: `run()` (resolves with render statistics), `stop()`, `next()`,
`previous()`, `replay()`, `restart()`.

Events bubble and cross shadow boundaries: `wanim-ready`, `wanim-run`,
`wanim-scene` (`detail.name`), `wanim-done` (`detail.stats`) and `wanim-error`
(`detail.error`).

Style the element's parts from outside:

```css
wanim-scene::part(canvas) { border-radius: 8px; }
wanim-scene::part(status) { font-size: 12px; }
```

### Code in the URL fragment (iframe)

For blogs and CMSs that accept only an `<iframe>`, encode the Python into the
URL fragment of a player page:

```js
import { build_embed_hash, build_embed_iframe } from "wanimjs";

const hash = await build_embed_hash(pythonSource, { quality: "-qm" });
const html = build_embed_iframe(`https://your-host.example/embed.html${hash}`, {
  width: 960,
  height: 540,
  title: "Square to circle",
});
```

The fragment is compressed with `deflate-raw` and base64url-encoded. It is never
sent to a server, so the code stays out of request logs. The generated iframe
is responsive and sandboxed to `allow-scripts allow-same-origin`.

The player page speaks a small postMessage protocol. It accepts messages only
from `window.parent`:

| Direction | Message |
| --- | --- |
| player → host | `{ source: "wanim-embed", v: 1, type: "ready" \| "scene" \| "done" \| "error", ... }` |
| host → player | `{ type: "wanim:run" }` |
| host → player | `{ type: "wanim:stop" }` |
| host → player | `{ type: "wanim:set-code", code, quality? }` |

`wanim:set-code` replaces the code and re-runs only if the player is already
running, so send `wanim:run` afterwards when you want an immediate run.

The repository includes `demo/embed.html`, a reference implementation of this
player. You can copy it as the starting point for your own host page.

### Security

Everything in this section compiles and runs Python in the visitor's browser.
Only embed code you trust, and treat `src` URLs and `build_embed_hash` input
the same way you would treat any script.

## Compatibility notes

- The API follows manim Community Edition (the reference version is v0.21).
  Where a feature is missing or behaves differently, the test suite and the
  differences list are the source of truth. If something in your scene runs
  under manim but not under wanim, please open an issue with the scene.
- Output is a live `<canvas>`. wanim does not produce video files.
- Math is typeset with MathJax, not LaTeX. Most common `MathTex` input works;
  unusual LaTeX packages do not.
- Text uses the configured font chain. Glyph coverage depends on the fonts
  you provide.

## Development

Prerequisites: Node.js 20 or newer and [pnpm](https://pnpm.io/).

```bash
git clone https://github.com/Acfboy/Wanim.git
cd Wanim
pnpm install
pnpm build:demo    # type-check, bundle the browser runtime, compile rapydscript, build the element
pnpm demo          # editor at http://localhost:5174/demo/
```

`build:demo` clones rapydscript-ng at a pinned commit and applies
`py2js/patches/rapydscript-ng.patch`, so the first build needs network access.
The checkout is cached in `.rapydscript-ng/`.

| Command | Purpose |
| --- | --- |
| `pnpm build` | compile the TypeScript library to `dist/` |
| `pnpm build:browser` | bundle the browser runtime |
| `pnpm build:demo` | build everything the demo and the browser bundles need |
| `pnpm build:packages` | assemble the publishable package in `packages-out/` |
| `pnpm pack:packages` | assemble from the current `dist/` and run `npm pack` |
| `pnpm test` | library test suite (Vitest) |
| `pnpm test:py2js` | preprocessor, compiler-error and runtime tests |
| `pnpm typecheck` | TypeScript check without emitting |

Ports: the demo server defaults to `5174`; pass another port as the first
argument to `node scripts/demo-server.mjs <port>`.

### Layout

```
src/                  manim-compatible library in TypeScript
  mobject/            mobjects, text, geometry, graphing, 3D
  animation/          animations and transforms
  scene/              Scene, InteractiveScene, ThreeDScene and variants
  interactive/        controls, presentation controller, DOM host
  renderer/           Canvas2D backend (render_scene_to_canvas)
  element.ts          the <wanim-scene> custom element
  embed.ts            code-in-fragment encoding and the iframe helper
py2js/                Python → JavaScript bridge
  src/                preprocessing, compiler, runtime shims, bridge
  patches/            rapydscript-ng patch
demo/                 editor page and iframe player
scripts/              bundling, packaging and demo server
packaging/wanim/      package.json, README and LICENSE for the npm package
tests/                Vitest suites for the library
```

## Contributing

Bug reports and minimal failing scenes are the most useful contribution right
now. Open an issue with the Python source, the expected output and what wanim
does instead. For code changes, run `pnpm typecheck`, `pnpm test` and
`pnpm test:py2js` before opening a pull request.

## License

MIT. Third-party dependencies keep their own licences: numpy-ts (MIT),
MathJax (Apache-2.0), opentype.js (MIT), and rapydscript-ng (BSD-2-Clause) for
the patched compiler.
