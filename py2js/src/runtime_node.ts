/**
 * Node entry point for the runtime half of `py2js`
 * (`@wanimjs/wanim/py-runtime`).
 *
 * Exports the DOM-free runtime toolbox: the rapydscript shim builders
 * (`createRuntimeCore`, `installPythonShims`), the CPython-compatible `math` /
 * `random` / `itertools` facades, the `from <module> import *` name tables and
 * the optional-module preload planner.  They have no runtime dependencies at
 * all, so they can be unit-tested and reused by Node tooling.
 *
 * The browser bridge (`browser.ts` → `@wanimjs/wanim/bridge`) is browser-only:
 * it needs a DOM (`document`, `<canvas>`) and the global `RapydScript`
 * compiler.
 */
export * from "./runtime_core.js";
export * from "./python_math.js";
export * from "./python_random.js";
export * from "./python_itertools.js";
export * from "./preload.js";
