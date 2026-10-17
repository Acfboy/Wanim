/**
 * Source pre-processor for the "native" modules `manim`, `numpy`, `random`,
 * `math` and `itertools`.
 *
 * rapydscript resolves imports to `.pyj` files, and it does not support star
 * imports at all.  Instead of shipping a huge generated stub for the wanim API
 * we simply drop these imports and expose every wanim export (plus a numpy
 * facade, a Python-compatible `random` module (see `python_random.ts`), a
 * Python-compatible `math` module (see `python_math.ts`), and a
 * Python-compatible `itertools` module (see `python_itertools.ts`)) as
 * globals of the execution context, so the user's Python source is
 * used *unmodified*.  Any symbol imported explicitly from numpy is recorded so
 * the runner can add the matching global alias; names imported from `random`,
 * `math` or `itertools` are rebound from the respective global
 * (`seed = random.seed`, `sin = math.sin`, `combinations =
 * itertools.combinations`) because the modules' functions are
 * detachment-safe closures.
 *
 * Every OTHER import (`import os`, `from collections import ...`, ...) cannot
 * work in the browser/JS runtime, because neither the rest of the Python
 * standard library nor third-party packages exist there.  Instead of passing
 * those lines through to rapydscript (which fails with a cryptic error) we
 * collect them in `unsupported` so the caller can show a friendly explanation.
 *
 * Detection is a deliberately simple, line-oriented heuristic: a statement is
 * only recognised when a line (ignoring leading whitespace) starts with
 * `import` or `from`.  This means it does NOT match imports hidden inside
 * strings or after a `;`, and it DOES falsely flag a line inside a
 * triple-quoted string that literally begins with `import`/`from`.  Those
 * limitations are acceptable here because the shipped input subset never
 * relies on them.
 *
 * Everything else manim Python needs is handled by the patched rapydscript
 * compiler itself (see `py2js/patches/rapydscript-ng.patch` and the
 * `python_extensions` parse option): tuple trailing commas, starred
 * list/tuple displays, multi-`for` comprehensions, multi-line generator
 * expressions, numpy multi-dim subscripts and slice stores, `@`/`@=`
 * matrices, complex literals, dotted/tuple `except` handlers, `.get` calls
 * and reserved-word identifiers.  This module deliberately contains no
 * syntax rewriting any more: source-level heuristics could not see inside
 * f-strings or nested brackets and are replaced by parser-level lowering.
 */

import { ITERTOOLS_PUBLIC, MATH_PUBLIC, RANDOM_PUBLIC } from "./public_names.js";

export interface UnsupportedImport {
  module: string;
  line: number;
  text: string;
}

export interface PreprocessResult {
  code: string;
  numpyNames: string[];
  stripped: string[];
  unsupported: UnsupportedImport[];
}


const MANIM_FROM = /^\s*from\s+manim(\.[\w.]+)?\s+import\s+(.+?)\s*(?:#.*)?$/;
const MANIM_IMPORT = /^\s*import\s+manim(\.[\w.]+)?(\s+as\s+\w+)?\s*(?:#.*)?$/;
const NUMPY_FROM = /^\s*from\s+numpy(\.[\w.]+)?\s+import\s+(.+?)\s*(?:#.*)?$/;
const NUMPY_IMPORT = /^\s*import\s+numpy(\.[\w.]+)?(\s+as\s+\w+)?\s*(?:#.*)?$/;
// `random` is provided as a runtime global (see `python_random.ts`).
const RANDOM_FROM = /^\s*from\s+random\s+import\s+(.+?)\s*(?:#.*)?$/;
const RANDOM_IMPORT = /^\s*import\s+random(\s+as\s+(\w+))?\s*(?:#.*)?$/;
// `math` is provided as a runtime global (see `python_math.ts`).
const MATH_FROM = /^\s*from\s+math\s+import\s+(.+?)\s*(?:#.*)?$/;
const MATH_IMPORT = /^\s*import\s+math(\s+as\s+(\w+))?\s*(?:#.*)?$/;
// `itertools` is provided as a runtime global (see `python_itertools.ts`).
const ITERTOOLS_FROM = /^\s*from\s+itertools\s+import\s+(.+?)\s*(?:#.*)?$/;
const ITERTOOLS_IMPORT = /^\s*import\s+itertools(\s+as\s+(\w+))?\s*(?:#.*)?$/;

// Any `from X import (` statement (no matter the module) may span several
// lines; join them so both the module name and the line number survive.
const PAREN_FROM_IMPORT = /^\s*from\s+[.\w]+\s+import\s*\(/;
const FROM_IMPORT = /^\s*from\s+([.\w]+)\s+import\s+(.+?)\s*(?:#.*)?$/;
const IMPORT = /^\s*import\s+(.+?)\s*(?:#.*)?$/;
// One comma-separated item of an `import a, b as c` statement.
const IMPORT_MODULE = /^([.\w]+)(?:\s+as\s+[\w.]+)?$/;

function parseImportedNames(spec: string): string[] {
  return spec
    .replace(/[()]/g, " ")
    .split(",")
    .map((part) => part.trim().split(/\s+as\s+/)[0].trim())
    .filter((name) => name.length > 0);
}

function parseImportModules(spec: string): string[] {
  return spec
    .split(",")
    .map((part) => part.trim().match(IMPORT_MODULE))
    .filter((match): match is RegExpMatchArray => match !== null)
    .map((match) => match[1]);
}

/** Split a `from X import a, b as c` spec into `[original, alias]` pairs. */
function parseFromNames(spec: string): [string, string][] {
  return spec
    .replace(/[()]/g, " ")
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => {
      const alias = part.split(/\s+as\s+/);
      const name = alias[0]!.trim();
      return [name, (alias.length > 1 ? alias[1]! : name).trim()] as [string, string];
    });
}

function normalize(line: string): string {
  return line.replace(/\s+/g, " ").trim();
}

export function preprocessImports(source: string): PreprocessResult {
  const lines = source.split("\n");
  const out: string[] = [];
  const numpyNames: string[] = [];
  const stripped: string[] = [];
  const unsupported: UnsupportedImport[] = [];
  // Internal alias the emitted `random` bindings read the module through.
  // `from random import random` rebinds the *user-visible* name `random` to
  // the function (exactly like CPython), so emitted code must not read the
  // module through that shadowable name; it reads `__wan_random` (suffixed
  // when user code already uses it) which is bound once, before any
  // shadowing can happen.
  let randomAlias = "__wan_random";
  while (source.includes(randomAlias)) randomAlias += "_";
  let randomAliasEmitted = false;
  // Internal alias the emitted `math` bindings read the module through.
  // `from math import pi` (or any other name) rebinds a user-visible name, and
  // a later `import math as m` must still see the module, so emitted code
  // reads `__wan_math` (suffixed when user code already uses it) which is
  // bound once, before any shadowing can happen.  Same pattern as `random`
  // above; `math` functions are detachment-safe closures too.
  let mathAlias = "__wan_math";
  while (source.includes(mathAlias)) mathAlias += "_";
  let mathAliasEmitted = false;
  // Internal alias the emitted `itertools` bindings read the module through.
  // `from itertools import chain` rebinds a user-visible name, and a later
  // `import itertools as it` must still see the module, so emitted code
  // reads `__wan_itertools` (suffixed when user code already uses it) which is
  // bound once, before any shadowing can happen.  Same pattern as `random`
  // above; `itertools` functions are detachment-safe closures too.
  let itertoolsAlias = "__wan_itertools";
  while (source.includes(itertoolsAlias)) itertoolsAlias += "_";
  let itertoolsAliasEmitted = false;

  for (let i = 0; i < lines.length; i += 1) {
    let line = lines[i];
    const startLine = i + 1;

    // Join a parenthesised import spread over several lines (any module).
    if (PAREN_FROM_IMPORT.test(line)) {
      let joined = line;
      while (!/\)/.test(joined) && i + 1 < lines.length) {
        i += 1;
        joined += "\n" + lines[i];
      }
      line = joined.replace(/\n/g, " ");
    }

    // Emitted replacement lines must keep the original statement's indentation:
    // an `import` inside a function/method body (e.g. `import math` in
    // `construct`) would otherwise dedent to column 0 and rapydscript fails
    // with "Inconsistent indentation".
    const indent = line.match(/^[ \t]*/)?.[0] ?? "";

    const numpyFrom = line.match(NUMPY_FROM);
    if (numpyFrom) {
      const spec = numpyFrom[2].trim();
      if (spec === "*") numpyNames.push("*");
      else numpyNames.push(...parseImportedNames(spec));
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      continue;
    }
    if (NUMPY_IMPORT.test(line)) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      continue;
    }
    if (MANIM_FROM.test(line) || MANIM_IMPORT.test(line)) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      continue;
    }

    // `random` is provided as a runtime global (see `python_random.ts`):
    // `import random` needs no rewrite, `import random as rnd` binds the
    // alias, and `from random import ...` rebinds each name from the module
    // (whose functions are detachment-safe closures, so `seed = random.seed`
    // keeps working as a plain call).  All emitted bindings read the module
    // through the internal `randomAlias`, never the shadowable `random`.
    const randomImport = line.match(RANDOM_IMPORT);
    if (randomImport) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!randomAliasEmitted) {
        out.push(`${indent}${randomAlias} = random`);
        randomAliasEmitted = true;
      }
      const alias = randomImport[2];
      if (alias !== undefined && alias !== "random") out.push(`${indent}${alias} = ${randomAlias}`);
      continue;
    }
    const randomFrom = line.match(RANDOM_FROM);
    if (randomFrom) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!randomAliasEmitted) {
        out.push(`${indent}${randomAlias} = random`);
        randomAliasEmitted = true;
      }
      for (const [name, alias] of parseFromNames(randomFrom[1])) {
        if (name === "*") {
          for (const pub of RANDOM_PUBLIC) out.push(`${indent}${pub} = ${randomAlias}.${pub}`);
        } else {
          out.push(`${indent}${alias} = ${randomAlias}.${name}`);
        }
      }
      continue;
    }

    // `math` is provided as a runtime global (see `python_math.ts`):
    // `import math` needs no rewrite, `import math as m` binds the
    // alias, and `from math import ...` rebinds each name from the module.
    // All emitted bindings read the module through the internal `mathAlias`,
    // never the shadowable `math`.
    const mathImport = line.match(MATH_IMPORT);
    if (mathImport) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!mathAliasEmitted) {
        out.push(`${indent}${mathAlias} = math`);
        mathAliasEmitted = true;
      }
      const alias = mathImport[2];
      if (alias !== undefined && alias !== "math") out.push(`${indent}${alias} = ${mathAlias}`);
      continue;
    }
    const mathFrom = line.match(MATH_FROM);
    if (mathFrom) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!mathAliasEmitted) {
        out.push(`${indent}${mathAlias} = math`);
        mathAliasEmitted = true;
      }
      for (const [name, alias] of parseFromNames(mathFrom[1])) {
        if (name === "*") {
          for (const pub of MATH_PUBLIC) out.push(`${indent}${pub} = ${mathAlias}.${pub}`);
        } else {
          out.push(`${indent}${alias} = ${mathAlias}.${name}`);
        }
      }
      continue;
    }

    // `itertools` is provided as a runtime global (see `python_itertools.ts`):
    // `import itertools` needs no rewrite, `import itertools as it` binds the
    // alias, and `from itertools import ...` rebinds each name from the module.
    // All emitted bindings read the module through the internal
    // `itertoolsAlias`, never the shadowable `itertools`.
    const itertoolsImport = line.match(ITERTOOLS_IMPORT);
    if (itertoolsImport) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!itertoolsAliasEmitted) {
        out.push(`${indent}${itertoolsAlias} = itertools`);
        itertoolsAliasEmitted = true;
      }
      const alias = itertoolsImport[2];
      if (alias !== undefined && alias !== "itertools") out.push(`${indent}${alias} = ${itertoolsAlias}`);
      continue;
    }
    const itertoolsFrom = line.match(ITERTOOLS_FROM);
    if (itertoolsFrom) {
      stripped.push(line.trim());
      out.push(`${indent}# [wanim] stripped: ${line.trim()}`);
      if (!itertoolsAliasEmitted) {
        out.push(`${indent}${itertoolsAlias} = itertools`);
        itertoolsAliasEmitted = true;
      }
      for (const [name, alias] of parseFromNames(itertoolsFrom[1])) {
        if (name === "*") {
          for (const pub of ITERTOOLS_PUBLIC) out.push(`${indent}${pub} = ${itertoolsAlias}.${pub}`);
        } else {
          out.push(`${indent}${alias} = ${itertoolsAlias}.${name}`);
        }
      }
      continue;
    }

    const fromMatch = line.match(FROM_IMPORT);
    if (fromMatch) {
      unsupported.push({ module: fromMatch[1], line: startLine, text: normalize(line) });
      out.push(line);
      continue;
    }
    const importMatch = line.match(IMPORT);
    if (importMatch) {
      const text = normalize(line);
      for (const module of parseImportModules(importMatch[1])) {
        unsupported.push({ module, line: startLine, text });
      }
      out.push(line);
      continue;
    }

    out.push(line);
  }

  return { code: out.join("\n"), numpyNames, stripped, unsupported };
}

/** Human-readable explanation for a single unsupported import. */
export function formatUnsupportedImportError(item: UnsupportedImport): string {
  return (
    `wanim: cannot use Python module '${item.module}' (line ${item.line}: ${item.text}).\n` +
    "Only the manim API, numpy, and the Python 'random', 'math' and 'itertools' modules are available in the browser/JS " +
    "runtime; the rest of the Python standard library and third-party packages cannot be imported.\n" +
    "Use 'from manim import *' and 'import numpy as np' instead (both are provided as globals; " +
    "'import random', 'import math' and 'import itertools' are supported too).\n" +
    "Math functions such as sin/cos/exp are available as numpy functions, e.g. " +
    "'from numpy import sin, cos, exp' or 'np.sin(...)' (plain JS Math is not exposed)."
  );
}

/** An `Error` describing every unsupported import, or `null` if there is none. */
export function unsupportedImportError(result: PreprocessResult): Error | null {
  if (result.unsupported.length === 0) return null;
  return new Error(result.unsupported.map(formatUnsupportedImportError).join("\n\n"));
}
