# py2js

Run (almost unmodified) Python manim scenes in the browser.

```
manim Python  --rapydscript-->  JavaScript  --wanim-->  live <canvas>
```

`py2js` is a thin bridge between the patched
[rapydscript-ng](https://github.com/kovidgoyal/rapydscript-ng) compiler and
wanim's Canvas2D renderer. It exposes two functions (see
`src/browser.ts`):

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
and builds small numpy and `random` facades (`src/python_random.ts`, a
CPython-compatible MT19937 port). `src/preprocess.ts` strips the
`manim`/`numpy`/`random` imports (their exports are injected as globals) and turns any
other import into a friendly error, because the rest of the Python standard library is not
available.

## Tests

```bash
pnpm test:py2js
```
