/**
 * Node entry point for the `wanim-py` package.
 *
 * Only the DOM-free half of the bridge is exported here: the Python source
 * preprocessor, the compile-error formatter, the optional-module preload
 * planner and the runtime shim builders.  They are useful in Node for tooling
 * (linting / transforming manim Python, unit-testing the runtime) and have no
 * runtime dependencies at all.
 *
 * The browser entry point (`wanim-py/browser`, bundled from `browser.ts`) is
 * browser-only: it needs a DOM (`document`, `<canvas>`, `XMLHttpRequest`) and
 * the global `RapydScript` compiler, and it is loaded through an import map
 * rather than by a bundler.  See the package README.
 */
export * from "./preprocess.js";
export * from "./compile_error.js";
export * from "./preload.js";
export * from "./python_math.js";
export * from "./python_random.js";
export * from "./python_itertools.js";
export * from "./runtime_core.js";
