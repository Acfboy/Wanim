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
import {
  desugarAbsBuiltin,
  desugarComplexLiterals,
  desugarDictGet,
  desugarDottedExcept,
  desugarMatmul,
  desugarMultiForComprehensions,
  desugarNumpySubscripts,
  desugarReversedBuiltin,
  desugarStarredDisplays,
  desugarStarredGenerators,
  desugarSumBuiltin,
  desugarTupleTrailingCommas,
  preprocessImports,
  renameJsReservedWords,
  unsupportedImportError,
} from "./preprocess.js";
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
 */
export async function compilePython(source: string): Promise<CompiledScene> {
  // rapydscript-ng only parses a single `for` clause per comprehension, a
  // single index/slice per `[...]`, `@` as a decorator, no `j`-suffixed
  // complex literals, a bare name per `except` handler, no trailing comma
  // inside `(...)` tuples, no `*` spread inside list/tuple displays, no newline
  // between a generator expression's element and its `for`, and its
  // `reversed()` only accepts plain arrays/strings, baselib `abs` is
  // `Math.abs` (NaN on complex), baselib `sum` accumulates with plain JS
  // `+` (which stringifies NDArray terms), and plain-object dict literals
  // have no `.get`, so all twelve are
  // desugared before compilation:
  // multi-`for` comprehensions into nested single-`for` ones (plus
  // `__wan_flatten1` calls), dotted/tuple `except` handlers
  // (`except np.linalg.LinAlgError:`) into hoisted `__wan_exc_N` aliases,
  // numpy multi-dim subscripts (`a[:, 1]`) into
  // `__wan_getitem`/`__wan_slice` calls, matrix multiplication (`a @ b`)
  // into `__wan_matmul` calls, complex literals (`0j`) into
  // `__wan_complex(0, imag)` calls, tuple trailing commas (`(a, b,)`,
  // `(x,)`) into `(a, b)`/`[x]`, starred list/tuple displays (`[*a, b]`,
  // `(*a, b)`) into `[].concat(__wan_spread(a), [b])` calls, starred
  // generator unpackings (`f(*(x for x in xs))`) into unpackings of the
  // equivalent list comprehension (`f(*([x for x in xs]))`), bare
  // `reversed(x)` calls into `__wan_reversed(x)` calls, bare `abs(x)`
  // calls into `__wan_abs(x)` calls, bare `sum(...)` calls into
  // `__wan_sum(...)` calls, and `d.get(k[, dflt])` calls into
  // `__wan_dict_get(d, k[, dflt])` calls.  The tuple step runs
  // first so later passes never see the trailing comma; the starred-display
  // step runs right after so later passes only see plain displays; the
  // complex step runs before the matmul step so it never sees the `j`
  // suffix; the matmul step runs before the subscript step so `@` operands
  // containing `[...]` (e.g. `A[:, 0] @ b`) still reach the subscript step
  // intact; the starred-generator step runs before the multi-`for` step so a
  // multi-`for` generator reaches it as a list comprehension.
  const pre = preprocessImports(
    desugarDottedExcept(
      desugarDictGet(
        desugarMultiForComprehensions(
          desugarStarredGenerators(
            desugarNumpySubscripts(
              desugarReversedBuiltin(
                desugarSumBuiltin(
                  desugarAbsBuiltin(
                    desugarMatmul(
                      desugarComplexLiterals(
                        desugarStarredDisplays(
                          desugarTupleTrailingCommas(renameJsReservedWords(source)),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    ),
  );
  const importError = unsupportedImportError(pre);
  if (importError !== null) throw importError;
  try {
    const js = await getRapydScript().compile(pre.code, "scene.py", {
      operator_overloading: true,
      await_scene_methods: true,
      bare: true,
      beautify: true,
    });
    return { js, numpyNames: pre.numpyNames };
  } catch (error) {
    // Syntax errors carry a line/column; render the source line and a caret.
    throw formatCompileError(error, pre.code);
  }
}
