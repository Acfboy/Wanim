/**
 * Compiler browser entry for the `py2js` bridge.
 *
 * {@link compilePython} translates (almost unmodified) manim Python into
 * JavaScript using the patched rapydscript compiler.  It is deliberately
 * dependency-free: no wanim, no numpy-ts, no runtime shims.  The companion
 * `runtime.ts` supplies the runtime the compiled JS executes against, and
 * `browser.ts` (the bridge) ties the two together.
 *
 * This module is bundled by `scripts/build-py2js.mjs` into
 * `dist/py2js.browser.js`.
 */
import { preprocessImports, unsupportedImportError } from "./preprocess.js";
import { formatCompileError } from "./compile_error.js";

export * from "./preprocess.js";
export * from "./compile_error.js";

type Any = any;

/** The subset of the rapydscript browser global this module relies on. */
interface RapydScriptCompiler {
  compile(source: string, filename?: string, options?: Record<string, unknown>): string | Promise<string>;
}

function getRapydScript(): RapydScriptCompiler {
  const rs = (globalThis as Any).RapydScript;
  if (rs === undefined || typeof rs.compile !== "function") {
    throw new Error(
      "rapydscript compiler is not loaded. Include the patched compiler " +
        "(built by `pnpm build:rapydscript`) before running Python.",
    );
  }
  return rs as RapydScriptCompiler;
}

export interface CompiledScene {
  js: string;
  numpyNames: string[];
}

/**
 * Translate manim Python into JavaScript.
 *
 * The `manim`/`numpy`/`random`/`math`/`itertools` imports are stripped (their
 * exports are injected as globals by the runtime's `buildRuntime`) and any
 * other import raises a friendly error, because the rest of the Python
 * standard library is not available in the runtime.
 *
 * Everything else is handled by the patched compiler itself with
 * `python_extensions: true` (see `py2js/patches/rapydscript-ng.patch`):
 *
 * - syntax: tuple trailing commas, starred list/tuple displays, multi-`for`
 *   comprehensions, multi-line generator expressions, `@`/`@=` matrix
 *   multiplication, complex literals (`1.5j`), dotted/tuple `except`
 *   handlers, numpy multi-dim subscripts (`a[:, 1]`, `a[i, j] = v`);
 * - Python semantics the baselib lacks, lowered to `__wan_*` runtime calls
 *   (provided by `runtime_core.ts`): `d.get(...)` -> `__wan_dict_get`,
 *   `abs`/`sum`/`reversed` -> `__wan_abs`/`__wan_sum`/`__wan_reversed`,
 *   `a @ b` -> `__wan_matmul`, `1j` -> `__wan_complex`,
 *   `[*a]` -> `__wan_spread`, multi-`for` comprehensions ->
 *   `__wan_flatten1`, multi-dim subscripts ->
 *   `__wan_getitem`/`__wan_setitem`/`__wan_slice`;
 * - Python truthiness in boolean contexts with `python_truthiness: true`:
 *   `not`/`and`/`or`, `if`/`while` conditions, conditional expressions,
 *   comprehension `if` clauses and `assert` are lowered through `ρσ_bool`
 *   (baselib), so an empty list/dict/set/string is falsy as in CPython
 *   (`if not crits:` must not take JavaScript's `![] === false` branch);
 * - identifiers that collide with JavaScript reserved words
 *   (`new`, `default`, `let`, ...) are renamed to `<name>_modified`.
 */
export async function compilePython(source: string): Promise<CompiledScene> {
  const pre = preprocessImports(source);
  const importError = unsupportedImportError(pre);
  if (importError !== null) throw importError;
  try {
    const js = await getRapydScript().compile(pre.code, "scene.py", {
      operator_overloading: true,
      await_scene_methods: true,
      python_extensions: true,
      python_truthiness: true,
      bare: true,
      beautify: true,
    });
    return { js, numpyNames: pre.numpyNames };
  } catch (error) {
    // Syntax errors carry a line/column; render the source line and a caret.
    throw formatCompileError(error, pre.code);
  }
}
