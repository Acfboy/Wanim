# py2js

Run (almost unmodified) Python manim scenes in the browser.

```
manim Python  --rapydscript-->  JavaScript  --wanim-->  live <canvas>
```

`py2js` is a thin bridge between the patched
[rapydscript-ng](https://github.com/kovidgoyal/rapydscript-ng) compiler and
wanim's Canvas2D renderer. It is split into three browser entries plus two
Node-safe entries:

- `src/compiler.ts` — translate Python to JS (`compilePython`); depends on
  nothing but the global `RapydScript` compiler;
- `src/runtime.ts` — the rapydscript shims and the Python `math` / `random` /
  `itertools` facades (`createRuntimeCore`, `installPythonShims`);
- `src/browser.ts` — the bridge that ties them together and adds the DOM
  sandbox (`runPythonOnCanvas`, `benchInteractiveScene`):

```ts
// 1. Translate manim Python to JavaScript.
const { js } = await compilePython(pythonSource);

// 2. Compile and run a scene directly on a <canvas>.
await runPythonOnCanvas(pythonSource, canvasElement, { quality: "-qm" });
```

## Why a patch?

rapydscript normally translates `self.play(...)` into a plain method call. wanim
renders in real time and yields to the browser between frames, so the scene
methods are `async`; the patch in `patches/rapydscript-ng.patch` (built
from the pinned upstream commit by `../scripts/build-rapydscript-browser.mjs`):

- routes arithmetic through `ρσ_op_*` helpers, so Python operator overloading
  (`UP + LEFT`, matrix/scalar math, numpy arithmetic) survives; and
- rewrites `self.play(...)`/`self.wait(...)` into `await self...(...)` and marks
  the enclosing method `async`, keeping the Python source synchronous-looking.

`src/runtime_core.ts` supplies the matching runtime: it wraps wanim's ES
classes so rapydscript can instantiate them, implements the `ρσ_op_*` helpers,
and builds small numpy, `random` and `math` facades (`src/python_random.ts`, a
CPython-compatible MT19937 port, and `src/python_math.ts`, a CPython-compatible
`math` port). `src/preprocess.ts` strips the
`manim`/`numpy`/`random`/`math` imports (their exports are injected as globals) and turns any
other import into a friendly error, because the rest of the Python standard library is not
available.

## Tests

```bash
pnpm test:py2js
```

## Packaging

`../scripts/build-py2js.mjs` bundles the three browser entries into
`dist/py2js.browser.js` (compiler), `dist/py-runtime.browser.js` (runtime) and
`dist/bridge.browser.js` (bridge). The compiler and runtime bundles have no
external imports; the bridge keeps `wanim` and `numpy-ts` external, referenced
by *relative* URLs in the dev layout. With `--bare` a package variant of the
bridge is emitted to `dist-pkg/bridge.browser.js`, keeping `numpy-ts` a bare
specifier (the sibling bundles stay relative; the package ships them side by
side in `browser/`). The DOM-free toolbox is exported from
`src/compiler_node.ts` (preprocessing, error formatting) and
`src/runtime_node.ts` (preload planning, runtime shims, Python facades) — the
Node entries of the `@wanimjs/wanim/py2js` and `@wanimjs/wanim/py-runtime`
subpaths.
