# wanim-core

The core library of [Wanim](https://github.com/Acfboy/Wanim): a
**manim-compatible animation library** written in TypeScript, with a real-time
**Canvas2D backend** that renders live in the browser.

`wanim-core` is the animation engine only. To run *unmodified manim Python* on
a `<canvas>` (which needs a Python-to-JS compiler), use
[`wanim-py`](https://www.npmjs.com/package/wanim-py) or the combined
[`wanim`](https://www.npmjs.com/package/wanim) package.

## Install

```bash
npm install wanim-core
```

## Browser

Serve the whole `browser/` directory from the package (keep `chunks/` next to
`wanim.browser.js`) and point an import map at the two bundles:

```html
<script type="importmap">
{
  "imports": {
    "wanim-core": "./browser/wanim.browser.js",
    "numpy-ts": "./browser/numpy-ts.browser.js"
  }
}
</script>
<script type="module">
import { Scene, Circle, Create, FadeOut, render_scene_to_canvas } from "wanim-core";

class Demo extends Scene {
  construct() {
    const circle = new Circle().set_fill("#fc6255", 0.5);
    this.play(new Create(circle));
    this.play(new FadeOut(circle));
  }
}

await render_scene_to_canvas(Demo, document.querySelector("canvas"), { quality: "-qm" });
</script>
```

The bundles import each other by relative URL, so a direct
`import ... from "./browser/wanim.browser.js"` also works without an import map
(that is how the combined `wanim` package is wired).

### Optional modules

MathJax is **not** part of `wanim.browser.js`; it is emitted
as content-hashed chunks in `browser/chunks/` and fetched on demand. Serve the
chunks directory, then preload explicitly or let the API lazy-load:

| API | Purpose |
| --- | --- |
| `await preload_submodules("mathjax")` | TeX support (`Tex`, `MathTex`, `DecimalNumber`, …) |
| `await preload_submodules("fonts", { fonts?: string[] })` | parse and cache the active (or given) font chain |
| `await preload_submodules(["mathjax", "fonts"])` | load several submodules in parallel |
| `submodule_status()` | `{ fonts, mathjax }` introspection |

All preloads are idempotent and share one pending load per submodule; unknown
names throw a `ValueError`. The TeX helpers (`texToSvg`, building a `MathTex`)
stay synchronous and throw with instructions if MathJax has not been loaded; in
Node, call `await preload_submodules("mathjax")` once at startup before
building TeX mobjects.

## Node

```js
import { MathTex, preload_submodules } from "wanim-core";

await preload_submodules("mathjax");

const formula = new MathTex("E = mc^2");
console.log(formula.width > 0); // true
```

`wanim-core` in Node builds and serializes mobjects/scenes (the `.mjson`
format). Live rasterization requires a `<canvas>`, so rendering itself is
browser-only. `dist/` is plain ES2022 ESM with bundled `.d.ts` declarations.

## Fonts

The package does not redistribute the proprietary fonts used by the Wanim
development tree (Times New Roman / FZShuSong). In Node the text pipeline falls
back to the host's system font; for reproducible typography install a font and
point wanim at it:

```js
import { configure_fonts } from "wanim-core";
configure_fonts(["/path/to/your-font.ttf"]);
// or set WANIM_FONT=/path/to/your-font.ttf before importing wanim-core
```

In the browser the default chain is fetched from free public CDNs (Tinos from
the Google Fonts repo, Noto Serif SC from Fontsource); pass your own URLs to
`configure_fonts([...])` to self-host.

## Bundled third-party code

The shipped `browser/` files contain third-party code under their own licenses:

| File | Upstream | License |
| --- | --- | --- |
| `browser/numpy-ts.browser.js` | [numpy-ts](https://github.com/dupontcyborg/numpy-ts) | MIT |
| `browser/chunks/mathjax_runtime-*.js` | [MathJax](https://www.mathjax.org/) | Apache-2.0 |
| `browser/wanim.browser.js` (bundled dependency) | [opentype.js](https://opentype.js.org/) | MIT |

`mathjax-full`, `numpy-ts` and `opentype.js` are regular npm
dependencies for the Node entry point.

## Caveats

- Serve `browser/chunks/` together with the bundle; chunk names are
  content-hashed, so update both when upgrading.
- The bundled fonts are fetched from a CDN by default; offline / intranet
  deployments should self-host and call `configure_fonts()`.
- `dist/` reads `src/_config/default.cfg` and `src/templates/` at runtime for
  the manim config and template APIs; both are included in the package.

## License

MIT.
