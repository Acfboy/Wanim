/**
 * Node entry point for the compiler half of `py2js`
 * (`@wanimjs/wanim/py2js`).
 *
 * Exports the DOM-free Python source preprocessor (native-import stripping)
 * and the compile-error formatter: useful in Node for tooling (linting /
 * transforming manim Python, unit-testing the import handling).  The actual
 * translation needs the patched rapydscript compiler, which is browser-only
 * (`RapydScript` global); see `compiler.ts`.
 */
export * from "./preprocess.js";
export * from "./compile_error.js";
