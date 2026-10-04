/**
 * Very small source pre-processor for the "native" modules `manim` and
 * `numpy`.
 *
 * rapydscript resolves imports to `.pyj` files, and it does not support star
 * imports at all.  Instead of shipping a huge generated stub for the wanim API
 * we simply drop these imports and expose every wanim export (and a numpy
 * facade) as globals of the execution context, so the user's Python source is
 * used *unmodified*.  Any symbol imported explicitly from numpy is recorded so
 * the runner can add the matching global alias.
 *
 * Every OTHER import (`import os`, `from math import sin`, ...) cannot work in
 * the browser/JS runtime, because neither the Python standard library nor
 * third-party packages exist there.  Instead of passing those lines through to
 * rapydscript (which fails with a cryptic error) we collect them in
 * `unsupported` so the caller can show a friendly explanation.
 *
 * Detection is a deliberately simple, line-oriented heuristic: a statement is
 * only recognised when a line (ignoring leading whitespace) starts with
 * `import` or `from`.  This means it does NOT match imports hidden inside
 * strings or after a `;`, and it DOES falsely flag a line inside a
 * triple-quoted string that literally begins with `import`/`from`.  Those
 * limitations are acceptable here because the shipped input subset never
 * relies on them.
 */

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

// JavaScript reserves a number of words that are perfectly good Python
// identifiers (`new`, `this`, `default`, `function`, `let`, ...).  rapydscript
// passes them straight through, so `new = Text(...)` either fails with a
// confusing parse error (operator-words such as `new`) or with "X is a reserved
// word".  We rewrite such identifiers to `<name>_modified` *before* compilation,
// so the translated JS stays valid.  Python keywords (`class`, `for`, `in`,
// `and`, ...) are left alone -- they are not valid variable names in Python
// anyway -- and `super` is kept because wanim's rapydscript patch gives it
// special meaning (`super().__init__(...)`).
const JS_RESERVED_RENAMES: Record<string, string> = (() => {
  const words = [
    "arguments", "case", "catch", "const", "debugger", "default", "delete",
    "do", "enum", "eval", "export", "extends", "false", "function",
    "implements", "instanceof", "interface", "let", "new", "null", "package",
    "private", "protected", "public", "static", "switch", "this", "throw",
    "true", "typeof", "var", "void",
  ];
  const map: Record<string, string> = {};
  for (const word of words) map[word] = `${word}_modified`;
  return map;
})();

const IDENT_START = /[A-Za-z_]/;
const IDENT_PART = /[A-Za-z0-9_]/;

/** Consume a Python string literal starting at `start` (which is a quote). */
function readString(source: string, start: number): number {
  const quote = source[start];
  const triple = source.startsWith(quote.repeat(3), start);
  const delimiter = triple ? quote.repeat(3) : quote;
  let i = start + delimiter.length;
  while (i < source.length) {
    if (source[i] === "\\") {
      i += 2;
      continue;
    }
    if (source.startsWith(delimiter, i)) return i + delimiter.length;
    // Unterminated single-quoted strings end at the newline; triple-quoted
    // strings may span lines.
    if (!triple && (source[i] === "\n" || source[i] === "\r")) return i;
    i += 1;
  }
  return i;
}

/**
 * Rename identifiers that collide with JavaScript reserved words.
 *
 * The scan skips comments and string literals and never touches an attribute
 * name (`obj.new` stays `obj.new`).  This is a deliberately simple, source-level
 * heuristic: it does not look inside f-string `{...}` interpolations, and it
 * cannot tell a keyword argument name (`f(new=1)`) from a variable, so both are
 * renamed together (which stays consistent as long as the callee is compiled
 * from the same source).
 */
export function renameJsReservedWords(source: string): string {
  let out = "";
  let i = 0;
  let previous: string | null = null; // last non-space character emitted
  while (i < source.length) {
    const ch = source[i]!;
    if (ch === "#") {
      const end = source.indexOf("\n", i);
      const stop = end === -1 ? source.length : end;
      out += source.slice(i, stop);
      i = stop;
      continue;
    }
    if (ch === '"' || ch === "'") {
      const stop = readString(source, i);
      out += source.slice(i, stop);
      i = stop;
      previous = '"';
      continue;
    }
    if (IDENT_START.test(ch)) {
      let j = i + 1;
      while (j < source.length && IDENT_PART.test(source[j]!)) j += 1;
      const word = source.slice(i, j);
      const renamed: string =
        previous !== "." && Object.prototype.hasOwnProperty.call(JS_RESERVED_RENAMES, word)
          ? JS_RESERVED_RENAMES[word]!
          : word;
      out += renamed;
      previous = renamed[renamed.length - 1]!;
      i = j;
      continue;
    }
    out += ch;
    if (!/\s/.test(ch)) previous = ch;
    i += 1;
  }
  return out;
}

const MANIM_FROM = /^\s*from\s+manim(\.[\w.]+)?\s+import\s+(.+?)\s*(?:#.*)?$/;
const MANIM_IMPORT = /^\s*import\s+manim(\.[\w.]+)?(\s+as\s+\w+)?\s*(?:#.*)?$/;
const NUMPY_FROM = /^\s*from\s+numpy(\.[\w.]+)?\s+import\s+(.+?)\s*(?:#.*)?$/;
const NUMPY_IMPORT = /^\s*import\s+numpy(\.[\w.]+)?(\s+as\s+\w+)?\s*(?:#.*)?$/;

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

function normalize(line: string): string {
  return line.replace(/\s+/g, " ").trim();
}

export function preprocessImports(source: string): PreprocessResult {
  const lines = source.split("\n");
  const out: string[] = [];
  const numpyNames: string[] = [];
  const stripped: string[] = [];
  const unsupported: UnsupportedImport[] = [];

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

    const numpyFrom = line.match(NUMPY_FROM);
    if (numpyFrom) {
      const spec = numpyFrom[2].trim();
      if (spec === "*") numpyNames.push("*");
      else numpyNames.push(...parseImportedNames(spec));
      stripped.push(line.trim());
      out.push(`# [wanim] stripped: ${line.trim()}`);
      continue;
    }
    if (NUMPY_IMPORT.test(line)) {
      stripped.push(line.trim());
      out.push(`# [wanim] stripped: ${line.trim()}`);
      continue;
    }
    if (MANIM_FROM.test(line) || MANIM_IMPORT.test(line)) {
      stripped.push(line.trim());
      out.push(`# [wanim] stripped: ${line.trim()}`);
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
    "Only the manim API and numpy are available in the browser/JS runtime; the Python " +
    "standard library and third-party packages cannot be imported.\n" +
    "Use 'from manim import *' and 'import numpy as np' instead (both are provided as globals).\n" +
    "Math functions such as sin/cos/exp are available as numpy functions, e.g. " +
    "'from numpy import sin, cos, exp' or 'np.sin(...)' (plain JS Math is not exposed)."
  );
}

/** An `Error` describing every unsupported import, or `null` if there is none. */
export function unsupportedImportError(result: PreprocessResult): Error | null {
  if (result.unsupported.length === 0) return null;
  return new Error(result.unsupported.map(formatUnsupportedImportError).join("\n\n"));
}

// --- multi-`for` comprehension desugar --------------------------------------
//
// rapydscript-ng's `read_comprehension` only accepts a single
// `for ... in ... [if ...]` clause, so valid Python such as
// `[f(i, j) for j in ys for i in xs]` fails with
// "Unexpected token keyword «for», expected keyword «if»".
// Python (and therefore manim) accepts any number of `for`/`if` clauses.
//
// We rewrite multi-`for` comprehensions into nested single-`for`
// comprehensions plus a one-level flatten helper (`__wan_flatten1`, provided
// by the runtime in `runtime_core.ts`), preserving clause order and therefore
// iteration order:
//
//   `[E for a in A for b in B if C]`
//   -> `__wan_flatten1([[E for b in B if (C)] for a in A])`
//
// Set/dict/generator forms are reduced to the list form first and then
// wrapped in a single-`for` comprehension of their own kind:
//
//   `{E for a in A for b in B}`      -> `{__wan_y for __wan_y in __wan_flatten1(...)}`
//   `(E for a in A for b in B)`      -> `(__wan_g for __wan_g in __wan_flatten1(...))`
//   `{K: V for a in A for b in B}`   -> `{__wan_kv[0]: __wan_kv[1] for __wan_kv in __wan_flatten1(...)}`
//
// (where `...` is the nested single-`for` list comprehension; for dicts its
// element is the pair `[K, V]`.)
//
// The scan skips string literals and `#` comments and only splits on `for`/
// `in`/`if`/`else` keywords at bracket depth zero with ASCII word boundaries,
// so keywords inside nested brackets, strings, or identifiers (e.g. `format`)
// are never treated as clause boundaries. Anything that does not parse cleanly
// (unbalanced brackets, `else` in clause position, `async for`, tuple display
// in a `(...)` head, ...) is left untouched so the compiler still reports it.
//
// Known limitations (documented in ): a bare lambda with
// a conditional body as an iteration target (e.g. `for a in lambda: 1 if c
// else 2`) would confuse the clause splitter; generator comprehensions lose
// laziness (the flattened list is built eagerly); and `:=` (walrus) targets
// inside a multi-`for` comprehension bind one scope further out than CPython.

/** Global runtime helper (see `runtime_core.ts`) used by the desugared code. */
export const WANIM_FLATTEN_HELPER = "__wan_flatten1";

/** Global runtime helpers for numpy multi-dim subscripts (see below). */
export const WANIM_GETITEM_HELPER = "__wan_getitem";
export const WANIM_SLICE_HELPER = "__wan_slice";

interface DesugarClause {
  target: string;
  iter: string;
  conds: string[];
}

const WORD_CHAR = /[\p{L}\p{N}_]/u;

function isWordCharAt(text: string, index: number): boolean {
  if (index < 0 || index >= text.length) return false;
  return WORD_CHAR.test(text[index]!);
}

/** Match an ASCII keyword at `pos` with word boundaries on both sides. */
function matchKeyword(
  source: string,
  pos: number,
  keyword: string,
): boolean {
  if (!source.startsWith(keyword, pos)) return false;
  if (isWordCharAt(source, pos - 1)) return false;
  if (isWordCharAt(source, pos + keyword.length)) return false;
  // `obj.for` / `obj.if` cannot be Python keywords (attribute access takes
  // names, not keywords); never treat them as clause boundaries.
  if (source[pos - 1] === ".") return false;
  return true;
}

const CLAUSE_KEYWORDS = ["for", "in", "if", "else"] as const;

interface KeywordOccurrence {
  keyword: "for" | "in" | "if" | "else";
  pos: number;
}

const BRACKET_PAIRS: Record<string, string> = {
  "(": ")",
  "[": "]",
  "{": "}",
};

/** Find the mate of the bracket at `openPos`, skipping strings/comments. */
function findMatchingBracket(source: string, openPos: number): number {
  const closer = BRACKET_PAIRS[source[openPos]!];
  if (closer === undefined) return -1;
  let depth = 0;
  let i = openPos;
  while (i < source.length) {
    const ch = source[i]!;
    if (ch === "#") {
      const end = source.indexOf("\n", i);
      i = end === -1 ? source.length : end;
      continue;
    }
    if (ch === '"' || ch === "'") {
      i = readString(source, i);
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") depth += 1;
    else if (ch === ")" || ch === "]" || ch === "}") {
      depth -= 1;
      if (depth === 0) return source[i] === closer ? i : -1;
    }
    i += 1;
  }
  return -1;
}

/** Collect depth-zero `for`/`in`/`if`/`else` keywords in `[begin, end). */
function collectClauseKeywords(
  source: string,
  begin: number,
  end: number,
): KeywordOccurrence[] {
  const found: KeywordOccurrence[] = [];
  let depth = 0;
  let i = begin;
  while (i < end) {
    const ch = source[i]!;
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      const stop = nl === -1 || nl >= end ? end : nl;
      i = stop;
      continue;
    }
    if (ch === '"' || ch === "'") {
      i = Math.min(readString(source, i), end);
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") {
      depth += 1;
      i += 1;
      continue;
    }
    if (ch === ")" || ch === "]" || ch === "}") {
      depth -= 1;
      i += 1;
      continue;
    }
    if (depth === 0) {
      for (const keyword of CLAUSE_KEYWORDS) {
        if (matchKeyword(source, i, keyword)) {
          found.push({ keyword, pos: i });
          break;
        }
      }
    }
    i += 1;
  }
  return found;
}

/** True when `text` contains `ch` outside nested brackets/strings/comments. */
function hasDepthZeroChar(text: string, ch: string): boolean {
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    const c = text[i]!;
    if (c === "#") {
      const nl = text.indexOf("\n", i);
      i = nl === -1 ? text.length : nl;
      continue;
    }
    if (c === '"' || c === "'") {
      i = readString(text, i);
      continue;
    }
    if (c === "(" || c === "[" || c === "{") depth += 1;
    else if (c === ")" || c === "]" || c === "}") depth -= 1;
    else if (depth === 0 && c === ch) return true;
    i += 1;
  }
  return false;
}

/** Position of the first depth-zero `ch` in `text`, or -1. */
function firstDepthZeroChar(text: string, ch: string): number {
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    const c = text[i]!;
    if (c === "#") {
      const nl = text.indexOf("\n", i);
      i = nl === -1 ? text.length : nl;
      continue;
    }
    if (c === '"' || c === "'") {
      i = readString(text, i);
      continue;
    }
    if (c === "(" || c === "[" || c === "{") depth += 1;
    else if (c === ")" || c === "]" || c === "}") depth -= 1;
    else if (depth === 0 && c === ch) return i;
    i += 1;
  }
  return -1;
}

function pickFreshName(base: string, source: string): string {
  let name = base;
  while (source.includes(name)) name += "_";
  return name;
}

interface ComprehensionParts {
  kind: "[" | "(" | "{";
  element: string;
  dictKey: string | null;
  dictValue: string | null;
  clauses: DesugarClause[];
}

/** Parse `[begin, end)` as a multi-`for` comprehension head; null if not one. */
function parseMultiForComprehension(
  source: string,
  open: string,
  begin: number,
  end: number,
): ComprehensionParts | null {
  const occurrences = collectClauseKeywords(source, begin, end);
  // The element expression comes first and may itself contain depth-zero
  // `if`/`in`/`else` (e.g. a ternary `a if c else b`); the clause chain starts
  // at the first depth-zero `for` (a `for` inside the element would need
  // brackets, which raise the depth).
  const firstFor = occurrences.findIndex((occ) => occ.keyword === "for");
  if (firstFor === -1) return null;
  const head = source.slice(begin, occurrences[firstFor]!.pos);
  let element = "";
  let dictKey: string | null = null;
  let dictValue: string | null = null;
  if (open === "{") {
    if (hasDepthZeroChar(head, ",")) return null;
    const colon = firstDepthZeroChar(head, ":");
    if (colon !== -1) {
      dictKey = head.slice(0, colon).trim();
      dictValue = head.slice(colon + 1).trim();
      if (dictKey === "" || dictValue === "") return null;
    } else {
      element = head.trim();
      if (element === "") return null;
    }
  } else {
    if (open === "(") {
      // A `(...)` head with a top-level comma/colon is a tuple display, call
      // argument list, slice, or annotation -- never a generator's element.
      if (hasDepthZeroChar(head, ",") || hasDepthZeroChar(head, ":")) return null;
      if (/^\s*(yield|\*)/.test(head)) return null;
    }
    element = head.trim();
    if (element === "") return null;
  }

  const clauses: DesugarClause[] = [];
  let index = firstFor;
  let clauseStart = begin;
  while (index < occurrences.length) {
    const forOcc = occurrences[index]!;
    if (forOcc.keyword !== "for") return null;
    // `async for` belongs to asyncio; the emitter below only writes plain
    // `for`, so leave such comprehensions to the compiler.
    if (/\basync\s*$/.test(source.slice(clauseStart, forOcc.pos))) return null;
    const inOffset = occurrences
      .slice(index + 1)
      .findIndex((occ) => occ.keyword === "in");
    if (inOffset === -1) return null;
    const inOcc = occurrences[index + 1 + inOffset]!;
    const target = source.slice(forOcc.pos + 3, inOcc.pos).trim();
    if (target === "" || /\basync\b/.test(target)) return null;
    // The first non-`in` keyword after the separator ends the iterable; any
    // further `in` (e.g. a malformed `for a in x in y`) stays inside verbatim.
    let afterIn = index + 1 + inOffset + 1;
    while (
      afterIn < occurrences.length &&
      occurrences[afterIn]!.keyword === "in"
    ) {
      afterIn += 1;
    }
    const iterEnd = afterIn < occurrences.length ? occurrences[afterIn]!.pos : end;
    const iter = source.slice(inOcc.pos + 2, iterEnd).trim();
    if (iter === "") return null;
    const conds: string[] = [];
    let cursor = afterIn;
    while (
      cursor < occurrences.length &&
      occurrences[cursor]!.keyword === "if"
    ) {
      const ifOcc = occurrences[cursor]!;
      const next = cursor + 1;
      const condEnd = next < occurrences.length ? occurrences[next]!.pos : end;
      const cond = source.slice(ifOcc.pos + 2, condEnd).trim();
      if (cond === "") return null;
      conds.push(cond);
      cursor = next;
    }
    if (cursor < occurrences.length) {
      // Only `for` (next clause) may follow; `else` (a ternary) or a second
      // `in` means this is not a plain clause chain.
      if (occurrences[cursor]!.keyword !== "for") return null;
    }
    clauses.push({ target, iter, conds });
    clauseStart = iterEnd;
    index = cursor;
  }
  if (clauses.length < 2) return null;
  return {
    kind: open as "[" | "(" | "{",
    element,
    dictKey,
    dictValue,
    clauses,
  };
}

function condSuffix(conds: string[]): string {
  if (conds.length === 0) return "";
  return ` if (${conds.map((cond) => `(${cond})`).join(" and ")})`;
}

/** Rewrite the first multi-`for` comprehension in `source`, or null. */
function rewriteFirstMultiFor(source: string): string | null {
  let i = 0;
  while (i < source.length) {
    const ch = source[i]!;
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      i = nl === -1 ? source.length : nl + 1;
      continue;
    }
    if (ch === '"' || ch === "'") {
      i = readString(source, i);
      continue;
    }
    if (ch !== "(" && ch !== "[" && ch !== "{") {
      i += 1;
      continue;
    }
    const close = findMatchingBracket(source, i);
    if (close === -1) {
      i += 1;
      continue;
    }
    const parts = parseMultiForComprehension(source, ch, i + 1, close);
    if (parts === null) {
      i += 1;
      continue;
    }
    const helper = WANIM_FLATTEN_HELPER;
    // Innermost clause first, then wrap outwards; clause order (and hence
    // iteration order) is preserved.
    const last = parts.clauses[parts.clauses.length - 1]!;
    let nested: string;
    if (parts.dictKey !== null) {
      nested = `[[${parts.dictKey}, ${parts.dictValue}] for ${last.target} in ${last.iter}${condSuffix(last.conds)}]`;
    } else {
      nested = `[${parts.element} for ${last.target} in ${last.iter}${condSuffix(last.conds)}]`;
    }
    for (let k = parts.clauses.length - 2; k >= 0; k -= 1) {
      const clause = parts.clauses[k]!;
      nested = `${helper}([${nested} for ${clause.target} in ${clause.iter}${condSuffix(clause.conds)}])`;
    }
    let replacement: string;
    if (parts.kind === "[") {
      replacement = nested;
    } else if (parts.dictKey !== null) {
      const pair = pickFreshName("__wan_kv", source);
      replacement = `{${pair}[0]: ${pair}[1] for ${pair} in ${nested}}`;
    } else if (parts.kind === "{") {
      const item = pickFreshName("__wan_y", source);
      replacement = `{${item} for ${item} in ${nested}}`;
    } else {
      const item = pickFreshName("__wan_g", source);
      replacement = `(${item} for ${item} in ${nested})`;
    }
    return source.slice(0, i) + replacement + source.slice(close + 1);
  }
  return null;
}

/**
 * Rewrite every multi-`for` comprehension into nested single-`for`
 * comprehensions (see above). Single-`for` comprehensions pass through
 * untouched. Runs to a fixpoint so a multi-`for` comprehension nested inside
 * another one is rewritten as well; each rewrite strictly reduces the number
 * of multi-`for` comprehensions, so this terminates.
 */
export function desugarMultiForComprehensions(source: string): string {
  // The helper name is reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses it, leave the source alone rather than silently
  // calling the wrong function.
  if (source.includes(WANIM_FLATTEN_HELPER)) return source;
  let out = source;
  for (let round = 0; round < 100; round += 1) {
    const rewritten = rewriteFirstMultiFor(out);
    if (rewritten === null) return out;
    out = rewritten;
  }
  return out;
}

// --- numpy multi-dim subscript desugar -------------------------------------
//
// rapydscript-ng parses only a single index or a single `start:stop` slice
// inside `[...]`, so valid numpy code such as `pts[:, 1]` or `a[1:2, 0]`
// fails with "Unexpected token: punc «,»".  We rewrite multi-dim subscripts
// (a top-level comma inside `[...]`) into runtime helper calls:
//
//   `pts[:, 1]`    -> `__wan_getitem(pts, __wan_slice(None, None), 1)`
//   `a[1:2, 0]`    -> `__wan_getitem(a, __wan_slice(1, 2), 0)`
//   `a[::-1, ...]` -> `__wan_getitem(a, __wan_slice(None, None, -1), "...")`
//   `a[0][:, 1]`   -> `__wan_getitem(a[0], __wan_slice(None, None), 1)`
//
// (helpers provided by the runtime in `runtime_core.ts`; `...` is Python's
// `Ellipsis`.)  Single subscripts (`a[0]`, `a[1:2]`, `a[-1]`) pass through
// untouched so rapydscript's own (well-tested) slice path keeps handling
// them.  List displays (`[1, 2]`), calls and dicts are never rewritten: a `[`
// only counts as a subscript when it directly follows an object expression
// (identifier, `)`/`]` group, or attribute chain) and not a Python keyword
// (`in`, `return`, ...), so `for i in [1, 2]` is left alone.
//
// The object expression is found by scanning backwards from the `[`; anything
// that does not parse cleanly (unbalanced brackets, keywords, string-literal
// objects, ...) is left untouched so the compiler still reports it.
//
// Known limitations: a bare `lambda x: x` as a dimension is kept verbatim
// (its `:` would otherwise look like a slice); `await` is the only keyword
// allowed at the head of an object chain (`await foo(x)[a, b]`); comments
// containing brackets/quotes inside a subscripted call's arguments may defeat
// the backwards scan (the subscript is then left as-is).

const PYTHON_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await",
  "break", "class", "continue", "def", "del", "elif", "else", "except",
  "finally", "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
  "try", "while", "with", "yield",
]);

const IDENT_CHAR = /[\p{L}\p{N}_]/u;

function isIdentCharAt(text: string, index: number): boolean {
  if (index < 0 || index >= text.length) return false;
  return IDENT_CHAR.test(text[index]!);
}

function isCloser(ch: string): boolean {
  return ch === ")" || ch === "]" || ch === "}";
}

function isHorizontalSpace(ch: string): boolean {
  return ch === " " || ch === "\t" || ch === "\r" || ch === "\f" || ch === "\v";
}

/** True when the quote at `pos` is escaped by an odd run of backslashes. */
function isEscaped(text: string, pos: number): boolean {
  let backslashes = 0;
  let i = pos - 1;
  while (i >= 0 && text[i] === "\\") {
    backslashes += 1;
    i -= 1;
  }
  return backslashes % 2 === 1;
}

/**
 * Given the index of a closing quote, return the index just before the
 * matching opening quote (or -1 when it cannot be found).  Handles escapes,
 * triple-quoted strings and (for single-quoted strings) `\`-continuations.
 */
function skipStringBackward(text: string, endPos: number): number {
  const quote = text[endPos]!;
  const triple =
    endPos >= 2 && text[endPos - 1] === quote && text[endPos - 2] === quote;
  if (triple) {
    let p = endPos - 5;
    while (p >= 0) {
      if (
        text[p] === quote &&
        text[p + 1] === quote &&
        text[p + 2] === quote &&
        !isEscaped(text, p)
      ) {
        return p - 1;
      }
      p -= 1;
    }
    return -1;
  }
  let j = endPos - 1;
  while (j >= 0) {
    const ch = text[j]!;
    if (ch === "\n") {
      // A `\` at end of line continues a single-quoted string.
      if (j > 0 && text[j - 1] === "\\" && !isEscaped(text, j - 1)) {
        j -= 2;
        continue;
      }
      return -1;
    }
    if (ch === quote && !isEscaped(text, j)) return j - 1;
    j -= 1;
  }
  return -1;
}

/**
 * Given the index of a `)`/`]`/`}`, return the index of its mate while
 * skipping string literals (or -1).  Bracket kinds are not distinguished,
 * mirroring the laxity of the forward matcher above.
 */
function findOpenerBackward(text: string, closePos: number): number {
  let depth = 0;
  let j = closePos;
  while (j >= 0) {
    const ch = text[j]!;
    if (ch === '"' || ch === "'") {
      j = skipStringBackward(text, j);
      continue;
    }
    if (ch === ")" || ch === "]" || ch === "}") depth += 1;
    else if (ch === "(" || ch === "[" || ch === "{") {
      depth -= 1;
      if (depth === 0) return j;
    }
    j -= 1;
  }
  return -1;
}

/**
 * Split `text` on every depth-zero `sep` (nesting and strings respected).
 * Newlines are kept verbatim (valid inside `[...]` by implicit continuation).
 */
function splitDepthZero(text: string, sep: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  let i = 0;
  while (i < text.length) {
    const ch = text[i]!;
    if (ch === "#") {
      const nl = text.indexOf("\n", i);
      i = nl === -1 ? text.length : nl;
      continue;
    }
    if (ch === '"' || ch === "'") {
      i = readString(text, i);
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") depth += 1;
    else if (ch === ")" || ch === "]" || ch === "}") depth -= 1;
    else if (depth === 0 && ch === sep) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
    i += 1;
  }
  parts.push(text.slice(start));
  return parts;
}

/**
 * Find the start of the object expression subscripted by a `[` whose matching
 * source position's already-emitted output is `out` (i.e. the object is the
 * tail of `out`).  Returns the start index, or null when the `[` is not a
 * subscript (list display, keyword-led like `in [1, 2]`, ...) or the object
 * cannot be determined (in which case the caller leaves the code untouched).
 */
function subscriptObjStart(out: string): number | null {
  let i = out.length - 1;
  while (i >= 0 && isHorizontalSpace(out[i]!)) i -= 1;
  if (i < 0 || out[i] === "\n") return null;
  let start: number | null = null;
  for (;;) {
    if (i < 0) return start;
    const ch = out[i]!;
    if (isCloser(ch)) {
      const opener = findOpenerBackward(out, i);
      if (opener === -1) return null;
      start = opener;
      i = opener - 1;
      while (i >= 0 && isHorizontalSpace(out[i]!)) i -= 1;
      if (i < 0 || out[i] === "\n") return start;
      if (isIdentCharAt(out, i) || out[i] === "." || isCloser(out[i]!)) continue;
      return start;
    }
    if (IDENT_CHAR.test(ch)) {
      let j = i;
      while (j >= 0 && IDENT_CHAR.test(out[j]!)) j -= 1;
      const word = out.slice(j + 1, i + 1);
      if (PYTHON_KEYWORDS.has(word)) {
        // `await foo(x)[a, b]` parses as `await (foo(x)[a, b])`, so the
        // object starts after the keyword; every other keyword means this
        // `[` is not a subscript (`in [1, 2]`, `return [..]`, ...).
        if (start !== null && word === "await") {
          let k = i + 1;
          while (k < out.length && isHorizontalSpace(out[k]!)) k += 1;
          return k;
        }
        return null;
      }
      start = j + 1;
      i = j;
      while (i >= 0 && isHorizontalSpace(out[i]!)) i -= 1;
      if (i < 0 || out[i] === "\n") return start;
      if (out[i] === ".") {
        i -= 1;
        while (i >= 0 && isHorizontalSpace(out[i]!)) i -= 1;
        if (i < 0 || out[i] === "\n") return null;
        continue;
      }
      // Two bare words in a row (`foo bar[x]`) or a group directly glued
      // after a word (`(x)foo[y]`) cannot be an object chain.
      if (isIdentCharAt(out, i) || isCloser(out[i]!)) return null;
      return start;
    }
    return start;
  }
}

/** Rewrite one comma-separated dimension; null when it is not supported. */
function rewriteSubscriptDim(dim: string): string | null {
  const trimmed = dim.trim();
  if (trimmed === "") return null;
  if (trimmed === "..." || trimmed === "Ellipsis") return '"..."';
  // A bare lambda's `:` is not a slice (`a[lambda x: x, 0]`).
  if (/^lambda(?![\p{L}\p{N}_])/u.test(trimmed)) {
    return desugarNumpySubscripts(trimmed);
  }
  const slices = splitDepthZero(trimmed, ":");
  if (slices.length === 1) return desugarNumpySubscripts(trimmed);
  if (slices.length > 3) return null;
  const bounds = slices.map((part) =>
    part.trim() === "" ? "None" : desugarNumpySubscripts(part.trim()),
  );
  return `${WANIM_SLICE_HELPER}(${bounds.join(", ")})`;
}

/**
 * Rewrite numpy multi-dim subscripts (`a[:, 1]`, `a[1:2, 0]`) into
 * `__wan_getitem`/`__wan_slice` runtime calls (see above) so rapydscript-ng
 * -- which only parses a single index/slice per `[...]` -- can compile them.
 * Anything that does not parse cleanly passes through untouched.
 */
export function desugarNumpySubscripts(source: string): string {
  // The helper names are reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses them, leave the source alone rather than silently
  // calling the wrong function.
  if (
    source.includes(WANIM_GETITEM_HELPER) ||
    source.includes(WANIM_SLICE_HELPER)
  ) {
    return source;
  }
  let out = "";
  let i = 0;
  while (i < source.length) {
    const ch = source[i]!;
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      if (nl === -1) {
        out += source.slice(i);
        break;
      }
      out += source.slice(i, nl);
      i = nl;
      continue;
    }
    if (ch === '"' || ch === "'") {
      const stop = readString(source, i);
      out += source.slice(i, stop);
      i = stop;
      continue;
    }
    if (ch !== "[") {
      out += ch;
      i += 1;
      continue;
    }
    const close = findMatchingBracket(source, i);
    if (close === -1) {
      out += ch;
      i += 1;
      continue;
    }
    const dims = splitDepthZero(source.slice(i + 1, close), ",");
    if (dims.length < 2) {
      out += ch;
      i += 1;
      continue;
    }
    const rewritten: string[] = [];
    let ok = true;
    for (const dim of dims) {
      const part = rewriteSubscriptDim(dim);
      if (part === null) {
        ok = false;
        break;
      }
      rewritten.push(part);
    }
    if (!ok) {
      out += ch;
      i += 1;
      continue;
    }
    const objStart = subscriptObjStart(out);
    if (objStart === null) {
      out += ch;
      i += 1;
      continue;
    }
    const obj = out.slice(objStart);
    out =
      out.slice(0, objStart) +
      `${WANIM_GETITEM_HELPER}(${obj}, ${rewritten.join(", ")})`;
    i = close + 1;
  }
  return out;
}
