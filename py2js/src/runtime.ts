/**
 * Runtime browser entry for the `py2js` bridge.
 *
 * This is the environment the compiled Python JS executes against: the
 * rapydscript shims (`runtime_core.ts`), the CPython-compatible `math` /
 * `random` / `itertools` facades, the `from <module> import *` name tables and
 * the optional-module (MathJax / fonts) preload planner.  It has no wanim or
 * numpy-ts imports of its own: `createRuntimeCore()` takes the wanim and
 * numpy-ts namespaces as arguments, and the bridge (`browser.ts`) injects the
 * very module instances the canvas backend uses.
 *
 * This module is bundled by `scripts/build-py2js.mjs` into
 * `dist/py-runtime.browser.js`.
 */
export * from "./runtime_core.js";
export * from "./python_math.js";
export * from "./python_random.js";
export * from "./python_itertools.js";
export * from "./preload.js";
