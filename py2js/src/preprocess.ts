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
export const WANIM_SETITEM_HELPER = "__wan_setitem";
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

/** Assignment operators handled for multi-dim subscript targets. Longest first. */
const SUBSCRIPT_ASSIGN_OPS = [
  "**=",
  "//=",
  ">>=",
  "<<=",
  "+=",
  "-=",
  "*=",
  "/=",
  "%=",
  "&=",
  "|=",
  "^=",
  "@=",
  "=",
];

/**
 * Look for an assignment operator right after a `]` at `close`.
 * Only horizontal gaps and `\`-continuations may sit between `]` and `=`;
 * a newline, `;`, comment or comparison (`==`) means this is not a store.
 */
function parseAssignOpAfter(
  source: string,
  close: number,
): { op: string; opEnd: number } | null {
  let j = close + 1;
  while (j < source.length) {
    const ch = source[j]!;
    if (ch === " " || ch === "\t" || ch === "\f" || ch === "\v") {
      j += 1;
      continue;
    }
    if (
      ch === "\\" &&
      (source[j + 1] === "\n" || source[j + 1] === "\r")
    ) {
      if (isEscaped(source, j)) return null;
      j += source[j + 1] === "\r" && source[j + 2] === "\n" ? 3 : 2;
      continue;
    }
    break;
  }
  if (j >= source.length) return null;
  const rest = source[j];
  if (rest === "\n" || rest === "\r" || rest === ";" || rest === "#") {
    return null;
  }
  for (const op of SUBSCRIPT_ASSIGN_OPS) {
    if (!source.startsWith(op, j)) continue;
    if (op === "=" && source[j + 1] === "=") return null;
    if (op !== "=" && source[j + op.length] === "=") return null;
    return { op, opEnd: j + op.length };
  }
  return null;
}

/**
 * Find the end of an assignment value starting at `start` (first char after
 * the operator): the first depth-zero `;`/newline/`#`, or end of source.
 * Brackets, strings and `\`-continuations are respected.
 */
function findAssignValueEnd(source: string, start: number): number {
  let depth = 0;
  let i = start;
  while (i < source.length) {
    const ch = source[i]!;
    if (ch === '"' || ch === "'") {
      i = readString(source, i);
      continue;
    }
    if (ch === "\\" && (source[i + 1] === "\n" || source[i + 1] === "\r")) {
      if (!isEscaped(source, i)) {
        i += source[i + 1] === "\r" && source[i + 2] === "\n" ? 3 : 2;
        continue;
      }
    }
    if (ch === "#") {
      if (depth === 0) return i;
      const nl = source.indexOf("\n", i);
      i = nl === -1 ? source.length : nl;
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") depth += 1;
    else if (ch === ")" || ch === "]" || ch === "}") {
      if (depth === 0) return i;
      depth -= 1;
    } else if (depth === 0 && (ch === ";" || ch === "\n" || ch === "\r")) {
      return i;
    }
    i += 1;
  }
  return i;
}

/** Skip gaps (spaces, newlines, continuations, comments) before a value. */
function skipToAssignValue(source: string, pos: number): number {
  let i = pos;
  while (i < source.length) {
    const ch = source[i]!;
    if (
      ch === " " ||
      ch === "\t" ||
      ch === "\f" ||
      ch === "\v" ||
      ch === "\n" ||
      ch === "\r"
    ) {
      i += 1;
      continue;
    }
    if (ch === "\\" && (source[i + 1] === "\n" || source[i + 1] === "\r")) {
      if (isEscaped(source, i)) return i;
      i += source[i + 1] === "\r" && source[i + 2] === "\n" ? 3 : 2;
      continue;
    }
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      i = nl === -1 ? source.length : nl + 1;
      continue;
    }
    if (ch === '"' || ch === "'") return i;
    return i;
  }
  return i;
}

/**
 * Rewrite numpy multi-dim subscripts (`a[:, 1]`, `a[1:2, 0]`) into
 * `__wan_getitem`/`__wan_slice` runtime calls (see above) so rapydscript-ng
 * -- which only parses a single index/slice per `[...]` -- can compile them.
 * Multi-dim subscript *stores* (`a[:, 1] = v`, `a[:, 1] += v`) become
 * `__wan_setitem` calls instead: a bare `__wan_getitem(...) = ...` is not
 * valid JavaScript (`Invalid left-hand side in assignment`), so the store
 * must be detected here. Anything that does not parse cleanly passes through
 * untouched.
 */
export function desugarNumpySubscripts(source: string): string {
  // The helper names are reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses them, leave the source alone rather than silently
  // calling the wrong function.
  if (
    source.includes(WANIM_GETITEM_HELPER) ||
    source.includes(WANIM_SETITEM_HELPER) ||
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
    const assign = parseAssignOpAfter(source, close);
    if (assign !== null) {
      const valueStart = skipToAssignValue(source, assign.opEnd);
      const valueEnd = findAssignValueEnd(source, valueStart);
      const rawValue = source.slice(valueStart, valueEnd).trim();
      if (rawValue === "") {
        out += ch;
        i += 1;
        continue;
      }
      const obj = out.slice(objStart).trim();
      if (obj === "") {
        out += ch;
        i += 1;
        continue;
      }
      const dimsJoin = rewritten.join(", ");
      const value = desugarNumpySubscripts(rawValue);
      const prefix = out.slice(0, objStart);
      if (assign.op === "=") {
        out = `${prefix}${WANIM_SETITEM_HELPER}(${obj}, ${dimsJoin}, (${value}))`;
      } else {
        const binOp = assign.op.slice(0, -1);
        const current = `${WANIM_GETITEM_HELPER}(${obj}, ${dimsJoin})`;
        const rhs =
          binOp === "@"
            ? `${WANIM_MATMUL_HELPER}(${current}, (${value}))`
            : `(${current} ${binOp} (${value}))`;
        out = `${prefix}${WANIM_SETITEM_HELPER}(${obj}, ${dimsJoin}, ${rhs})`;
      }
      i = valueEnd;
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

// --- matmul `@` desugar ------------------------------------------------------
//
// rapydscript-ng tokenises `@` but only accepts it as a decorator (`@name` at
// the start of a logical line); a binary `a @ b` (PEP 465 matrix
// multiplication, ubiquitous in numpy/manim code such as `rot @ center`)
// fails with "Unexpected token: operator «@»".  We rewrite binary `@` (and
// the augmented `a @= b`) into runtime helper calls before compilation:
//
//   `a @ b`   -> `__wan_matmul(a, b)`
//   `a @= b`  -> `a = __wan_matmul(a, b)`
//
// (helper provided by the runtime in `runtime_core.ts`; chains associate to
// the left, so `a @ b @ c` becomes `__wan_matmul(__wan_matmul(a, b), c)`.)
//
// Only `@` with the precedence of Python's multiplicative operators is
// rewritten: the left operand absorbs same-precedence `*`/`/`/`//`/`%` and
// higher-precedence `**` chains (`a * b @ c` -> `__wan_matmul(a * b, c)`)
// while the right operand absorbs `**`/unary/postfix only
// (`a @ b * c` -> `__wan_matmul(a, b) * c`, `a @ -b` keeps the sign).
// Anything that does not parse cleanly (decorators, strings, comments,
// `@@`, f-string interpolations, `#`-commented gaps, ...) is left untouched
// so the compiler still reports it instead of being silently miscompiled.
//
// Known limitations (documented in ): like the other
// desugars this is a source-level heuristic, not a full parser — an `@`
// inside an f-string `{...}` interpolation is not rewritten, and an `@=`
// target with side effects (`a[i()] @= b`) is evaluated twice.

/** Global runtime helper (see `runtime_core.ts`) used by the desugared code. */
export const WANIM_MATMUL_HELPER = "__wan_matmul";

const MATMUL_NUMBER_RE =
  /^(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9_]+)?[jJ]?|\.[0-9][0-9_]*(?:[eE][+-]?[0-9_]+)?[jJ]?)/;

/** Skip gaps after an operand going forward; null on a `#` comment. */
function matmulGapForward(
  text: string,
  i: number,
  allowNewline: boolean,
): number | null {
  while (i < text.length) {
    const ch = text[i]!;
    if (isHorizontalSpace(ch)) {
      i += 1;
      continue;
    }
    if (ch === "\\" && (text[i + 1] === "\n" || text[i + 1] === "\r")) {
      if (isEscaped(text, i)) return null;
      i += text[i + 1] === "\r" && text[i + 2] === "\n" ? 3 : 2;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      if (!allowNewline) return i;
      i += 1;
      continue;
    }
    if (ch === "#") return null;
    return i;
  }
  return i;
}

/** Skip gaps before an operand going backward; null on a `#` comment. */
function matmulGapBackward(
  text: string,
  i: number,
  allowNewline: boolean,
): number | null {
  while (i >= 0) {
    const ch = text[i]!;
    if (isHorizontalSpace(ch)) {
      i -= 1;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      if (!allowNewline) return i;
      let back = i - 1;
      if (ch === "\n" && text[back] === "\r") back -= 1;
      // A `\` immediately before the newline is an explicit continuation.
      if (back >= 0 && text[back] === "\\" && !isEscaped(text, back)) {
        i = back - 1;
        continue;
      }
      // Implicit continuation only happens inside brackets: the line we are
      // leaving must not contain a `#` comment, whose text would otherwise
      // be mistaken for code when scanning backwards.
      const prevNl = Math.max(
        text.lastIndexOf("\n", back),
        text.lastIndexOf("\r", back),
      );
      if (findMatmulCommentHash(text, prevNl + 1, back + 1) !== -1) {
        return null;
      }
      i = back;
      continue;
    }
    return i;
  }
  return i;
}

/** Index of a `#` starting a comment in `[from, to)`, or -1. */
function findMatmulCommentHash(
  text: string,
  from: number,
  to: number,
): number {
  let j = Math.max(0, from);
  const stop = Math.min(text.length, to);
  while (j < stop) {
    const ch = text[j]!;
    if (ch === '"' || ch === "'") {
      j = Math.min(readString(text, j), stop);
      continue;
    }
    if (ch === "#") return j;
    if (ch === "\n" || ch === "\r") return -1;
    j += 1;
  }
  return -1;
}

/** True when a `+`/`-`/`~` at `k` is unary (not a binary operator). */
function matmulIsUnary(text: string, k: number): boolean {
  let j = k - 1;
  while (j >= 0 && isHorizontalSpace(text[j]!)) j -= 1;
  if (j < 0) return true;
  const ch = text[j]!;
  if (ch === "\n" || ch === "\r") return true;
  // After a value (or a closing bracket/quote/dot) the sign is binary.
  if (/[A-Za-z0-9_)\]}"'`.]/.test(ch)) return false;
  return true;
}

/**
 * Extend an operand start leftwards over attribute/call chains:
 * `a.b`, `f(x)`, `a[i]`, `f(x)(y)`.  Returns the new start, or null when a
 * keyword is hit (e.g. `return [x]` stops at the keyword).
 */
function matmulChainBackward(
  text: string,
  i: number,
  start: number,
  allowNewline: boolean,
): number | null {
  for (;;) {
    const j = matmulGapBackward(text, i, allowNewline);
    if (j === null) return null;
    if (j < 0) return start;
    const ch = text[j]!;
    if (ch === ")" || ch === "]" || ch === "}") {
      const opener = findOpenerBackward(text, j);
      if (opener === -1) return null;
      start = opener;
      i = opener - 1;
      continue;
    }
    if (ch === ".") {
      const k = matmulGapBackward(text, j - 1, allowNewline);
      if (k === null || k < 0 || !IDENT_PART.test(text[k]!)) return null;
      let s = k;
      while (s >= 0 && IDENT_PART.test(text[s]!)) s -= 1;
      s += 1;
      if (PYTHON_KEYWORDS.has(text.slice(s, k + 1))) return null;
      start = s;
      i = s - 1;
      continue;
    }
    if (IDENT_START.test(ch)) {
      let s = j;
      while (s >= 0 && IDENT_PART.test(text[s]!)) s -= 1;
      s += 1;
      // A keyword directly ahead cannot extend the operand (`for x in y`,
      // `return [x]`); anything else is a callee/object (`f` in `f(x)`).
      if (PYTHON_KEYWORDS.has(text.slice(s, j + 1))) return start;
      start = s;
      i = s - 1;
      continue;
    }
    return start;
  }
}

/**
 * Parse one operand unit backwards: `[start, end)` (end exclusive), or null.
 * `i` is the inclusive index where the unit ends.
 */
function matmulParseUnitBackward(
  text: string,
  i: number,
  allowNewline: boolean,
): [number, number] | null {
  const end = matmulGapBackward(text, i, allowNewline);
  if (end === null || end < 0) return null;
  const ch = text[end]!;
  let start: number;
  if (ch === ")" || ch === "]" || ch === "}") {
    const opener = findOpenerBackward(text, end);
    if (opener === -1) return null;
    const chained = matmulChainBackward(text, opener - 1, opener, allowNewline);
    if (chained === null) return null;
    start = chained;
  } else if (IDENT_START.test(ch)) {
    let s = end;
    while (s >= 0 && IDENT_PART.test(text[s]!)) s -= 1;
    s += 1;
    const word = text.slice(s, end + 1);
    if (
      PYTHON_KEYWORDS.has(word) &&
      word !== "True" &&
      word !== "False" &&
      word !== "None"
    ) {
      return null;
    }
    const chained = matmulChainBackward(text, s - 1, s, allowNewline);
    if (chained === null) return null;
    start = chained;
  } else if (ch === '"' || ch === "'") {
    const before = skipStringBackward(text, end);
    if (before === -1) return null;
    start = before + 1;
    if (start - 1 >= 0 && /[A-Za-z_]/.test(text[start - 1]!)) {
      let s = start - 1;
      while (s >= 0 && /[A-Za-z0-9_]/.test(text[s]!)) s -= 1;
      if (/^[rRbBuUfF]+$/.test(text.slice(s + 1, start))) start = s + 1;
    }
  } else if (/[0-9]/.test(ch) || ch === ".") {
    if (
      ch === "." &&
      text[end - 1] === "." &&
      text[end - 2] === "."
    ) {
      // `...` (Ellipsis literal).
      start = end - 2;
    } else if (ch === "." && !/[0-9]/.test(text[end - 1] ?? "")) {
      // A lone trailing dot (`a.`); invalid Python, leave it to the compiler.
      return null;
    } else {
      let s = end;
      while (s >= 0 && /[0-9a-zA-Z_.]/.test(text[s]!)) s -= 1;
      // Exponent sign in `1e-5`/`2E+3` stops the run above; rejoin it.
      if (
        (text[s] === "-" || text[s] === "+") &&
        (text[s + 1] === "e" || text[s + 1] === "E") &&
        s - 1 >= 0 &&
        /[0-9.]/.test(text[s - 1]!)
      ) {
        s -= 1;
        while (s >= 0 && /[0-9a-zA-Z_.]/.test(text[s]!)) s -= 1;
      }
      s += 1;
      const candidate = text.slice(s, end + 1);
      const match = MATMUL_NUMBER_RE.exec(candidate);
      if (match === null || match[0].length !== candidate.length) return null;
      start = s;
    }
  } else {
    return null;
  }
  // Leading unary signs (`-a @ b`, `a * -b @ c`).
  for (;;) {
    const k = matmulGapBackward(text, start - 1, allowNewline);
    if (k === null) return null;
    if (
      k >= 0 &&
      (text[k] === "+" || text[k] === "-" || text[k] === "~") &&
      matmulIsUnary(text, k)
    ) {
      start = k;
      continue;
    }
    break;
  }
  return [start, end + 1];
}

/**
 * Parse one right-hand operand forward: exclusive end index, or null.  The
 * operand absorbs postfix trailers (`.attr`, calls, subscripts) and `**`
 * powers, but stops before same-precedence `*`/`/`/`//`/`%` and everything
 * of lower precedence.
 */
function matmulParseRight(
  text: string,
  pos: number,
  allowNewline: boolean,
): number | null {
  let i = matmulGapForward(text, pos, allowNewline);
  if (i === null) return null;
  while (text[i] === "+" || text[i] === "-" || text[i] === "~") {
    i = matmulGapForward(text, i + 1, allowNewline);
    if (i === null) return null;
  }
  if (i >= text.length) return null;
  const ch = text[i]!;
  if (ch === '"' || ch === "'") {
    i = readString(text, i);
  } else if (ch === "(" || ch === "[" || ch === "{") {
    const close = findMatchingBracket(text, i);
    if (close === -1) return null;
    i = close + 1;
  } else if (text.startsWith("...", i)) {
    i += 3;
  } else if (/[A-Za-z_]/.test(ch)) {
    let j = i + 1;
    while (j < text.length && /[A-Za-z0-9_]/.test(text[j]!)) j += 1;
    const word = text.slice(i, j);
    if (
      PYTHON_KEYWORDS.has(word) &&
      word !== "True" &&
      word !== "False" &&
      word !== "None"
    ) {
      return null;
    }
    i = j;
  } else if (/[0-9]/.test(ch) || (ch === "." && /[0-9]/.test(text[i + 1] ?? ""))) {
    const match = MATMUL_NUMBER_RE.exec(text.slice(i));
    if (match === null) return null;
    i += match[0].length;
  } else {
    return null;
  }
  for (;;) {
    const h = matmulGapForward(text, i, allowNewline);
    if (h === null) return null;
    const t = text[h] ?? "";
    if (t === ".") {
      let k = matmulGapForward(text, h + 1, allowNewline);
      if (k === null || k >= text.length || !/[A-Za-z_]/.test(text[k]!)) {
        return null;
      }
      k += 1;
      while (k < text.length && /[A-Za-z0-9_]/.test(text[k]!)) k += 1;
      i = k;
      continue;
    }
    if (t === "(" || t === "[") {
      const close = findMatchingBracket(text, h);
      if (close === -1) return null;
      i = close + 1;
      continue;
    }
    break;
  }
  // `**` binds tighter than `@`: `a @ b ** c` is `a @ (b ** c)`.
  const h = matmulGapForward(text, i, allowNewline);
  if (
    h !== null &&
    text.startsWith("**", h) &&
    text[h + 2] !== "*"
  ) {
    const expEnd = matmulParseRight(
      text,
      matmulGapForward(text, h + 2, allowNewline) ?? text.length,
      allowNewline,
    );
    if (expEnd === null) return null;
    i = expEnd;
  }
  return i;
}

/** True when the `@` at `p` starts a decorator (line-leading, depth zero). */
function matmulIsDecorator(text: string, p: number): boolean {
  let k = p - 1;
  while (k >= 0 && isHorizontalSpace(text[k]!)) k -= 1;
  if (k < 0) return true;
  const ch = text[k]!;
  if (ch !== "\n" && ch !== "\r") return false;
  // A `\` at the end of the previous line continues it: `x = a \<newline> @ b`
  // is a binary operator, not a decorator.
  let m = k - 1;
  while (m >= 0 && (text[m] === "\n" || text[m] === "\r")) m -= 1;
  while (m >= 0 && isHorizontalSpace(text[m]!)) m -= 1;
  if (m >= 0 && text[m] === "\\" && !isEscaped(text, m)) return false;
  return true;
}

/** Rewrite binary `@` (skipping `@=` for the later augmented pass). */
function rewriteMatmulBinary(source: string): string {
  let out = "";
  let i = 0;
  let depth = 0;
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
    if (ch === "(" || ch === "[" || ch === "{") {
      depth += 1;
      out += ch;
      i += 1;
      continue;
    }
    if (ch === ")" || ch === "]" || ch === "}") {
      depth = Math.max(0, depth - 1);
      out += ch;
      i += 1;
      continue;
    }
    if (ch !== "@") {
      out += ch;
      i += 1;
      continue;
    }
    const next = source[i + 1] ?? "";
    if (next === "@" || next === "=") {
      // `@@` is invalid Python; `@=` is handled by the augmented pass.
      out += ch;
      i += 1;
      continue;
    }
    if (depth === 0 && matmulIsDecorator(source, i)) {
      out += ch;
      i += 1;
      continue;
    }
    const allowNewline = depth > 0;
    const left = matmulLeftStart(out, allowNewline);
    const rightEnd = matmulParseRight(source, i + 1, allowNewline);
    if (left === null || rightEnd === null || rightEnd <= i + 1) {
      out += ch;
      i += 1;
      continue;
    }
    const [start, end] = left;
    const right = source.slice(i + 1, rightEnd).trim();
    if (right === "") {
      out += ch;
      i += 1;
      continue;
    }
    out =
      out.slice(0, start) +
      `${WANIM_MATMUL_HELPER}(${out.slice(start, end).trim()}, ${right})`;
    i = rightEnd;
  }
  return out;
}

/**
 * Find the left operand of a binary `@`: `[start, end)` into `out` (the
 * already-emitted prefix), absorbing same-precedence `*`/`/`/`//`/`%` and
 * higher-precedence `**` chains to its left.  Returns null when no clean
 * operand boundary exists.
 */
function matmulLeftStart(
  out: string,
  allowNewline: boolean,
): [number, number] | null {
  if (out.length === 0) return null;
  const unit = matmulParseUnitBackward(out, out.length - 1, allowNewline);
  if (unit === null) return null;
  let [start, end] = unit;
  for (;;) {
    const j = matmulGapBackward(out, start - 1, allowNewline);
    if (j === null) return null;
    let opStart = -1;
    if (j >= 1 && out[j] === "*" && out[j - 1] === "*") opStart = j - 1;
    else if (j >= 1 && out[j] === "/" && out[j - 1] === "/") opStart = j - 1;
    else if (
      j >= 0 &&
      (out[j] === "*" || out[j] === "/" || out[j] === "%")
    ) {
      opStart = j;
    } else {
      break;
    }
    const prev = matmulParseUnitBackward(out, opStart - 1, allowNewline);
    if (prev === null) break;
    start = prev[0];
  }
  return [start, end];
}

/** Rewrite augmented `target @= value` into `target = __wan_matmul(...)`. */
function rewriteMatmulAugmented(source: string): string {
  let out = "";
  let i = 0;
  let depth = 0;
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
    if (ch === "(" || ch === "[" || ch === "{") {
      depth += 1;
      out += ch;
      i += 1;
      continue;
    }
    if (ch === ")" || ch === "]" || ch === "}") {
      depth = Math.max(0, depth - 1);
      out += ch;
      i += 1;
      continue;
    }
    if (
      ch !== "@" ||
      source[i + 1] !== "=" ||
      source[i + 2] === "="
    ) {
      out += ch;
      i += 1;
      continue;
    }
    if (depth === 0 && matmulIsDecorator(source, i)) {
      out += ch;
      i += 1;
      continue;
    }
    const allowNewline = depth > 0;
    // The target is a single unit (a Name, attribute or subscript); operator
    // chains cannot be assignment targets.
    const target = matmulParseUnitBackward(out, out.length - 1, allowNewline);
    const rightEnd = matmulParseRight(source, i + 2, allowNewline);
    if (
      target === null ||
      rightEnd === null ||
      rightEnd <= i + 2 ||
      out.length === 0
    ) {
      out += ch;
      i += 1;
      continue;
    }
    const [start, end] = target;
    const targetText = out.slice(start, end).trim();
    const right = source.slice(i + 2, rightEnd).trim();
    if (targetText === "" || right === "") {
      out += ch;
      i += 1;
      continue;
    }
    out =
      out.slice(0, start) +
      `${targetText} = ${WANIM_MATMUL_HELPER}(${targetText}, ${right})`;
    i = rightEnd;
  }
  return out;
}

/**
 * Rewrite PEP 465 matrix multiplication (`a @ b`, `a @= b`) into
 * `__wan_matmul(...)` runtime calls (see above) so rapydscript-ng -- which
 * only accepts `@` as a decorator -- can compile them.  Binary operators are
 * rewritten before augmented ones so `x @= a @ b` keeps the right-hand side
 * together (`x = __wan_matmul(x, __wan_matmul(a, b))`).  Anything that does
 * not parse cleanly passes through untouched.
 */
export function desugarMatmul(source: string): string {
  // The helper name is reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses it, leave the source alone rather than silently
  // calling the wrong function.
  if (source.includes(WANIM_MATMUL_HELPER)) return source;
  return rewriteMatmulAugmented(rewriteMatmulBinary(source));
}

// --- complex literal desugar -------------------------------------------------
//
// rapydscript-ng's tokenizer has no complex-number support: a decimal literal
// with the `j`/`J` suffix (`0j`, `1.5j`, `1e-3j`, `.5j`, `1_000j`) fails with
// "Invalid syntax: 0j".  We rewrite every such literal into a runtime factory
// call before compilation:
//
//   `0j`    -> `__wan_complex(0, 0)`
//   `3j`    -> `__wan_complex(0, 3)`
//   `1e-3j` -> `__wan_complex(0, 0.001)`
//
// (helper provided by the runtime in `runtime_core.ts`; the imaginary part is
// normalized through `Number(...)` so Python underscore separators and
// exponent forms become plain JS numeric literals.)
//
// Only decimal float/int forms take the `j` suffix in CPython: hexadecimal,
// octal and binary literals (`0x1j`, `0o17j`, `0b1j`) are syntax errors there,
// so they are deliberately not matched (the compiler still reports them).  A
// digit run preceded by an identifier character, `_` or `.` is part of a name
// or an attribute access (`abc0j`, `x.1`) and is likewise left untouched; the
// scan skips string literals and `#` comments, so `"0j"` and `# 0j` pass
// through verbatim.  An uppercase `J` is accepted like CPython.
//
// The string form (`complex("1+2j")`) needs no rewrite and is handled by the
// runtime's `complex` builtin directly.
//
// Known limitations: like the other desugars this is a source-level heuristic
// -- a `j`-suffixed literal inside an f-string `{...}` interpolation is
// rewritten (harmless: it still denotes the same complex value), and code that
// already mentions `__wan_complex` is left entirely alone (reserved name, see
// above).

/** Global runtime helper (see `runtime_core.ts`) used by the desugared code. */
export const WANIM_COMPLEX_HELPER = "__wan_complex";

/** Global runtime helper (see `runtime_core.ts`) used by the desugared code. */
export const WANIM_REVERSED_HELPER = "__wan_reversed";

const COMPLEX_LITERAL_RE =
  /^(?:[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9_]+)?|\.[0-9][0-9_]*(?:[eE][+-]?[0-9_]+)?)[jJ]/;

const COMPLEX_PREV_CHAR = /[A-Za-z0-9_.]/;

/**
 * Rewrite Python complex literals (`0j`, `1.5j`, ...) into
 * `__wan_complex(0, imag)` runtime calls (see above) so rapydscript-ng --
 * which rejects the `j` suffix -- can compile them.  Anything that does not
 * parse cleanly passes through untouched.
 */
export function desugarComplexLiterals(source: string): string {
  // The helper name is reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses it, leave the source alone rather than silently
  // calling the wrong function.
  if (source.includes(WANIM_COMPLEX_HELPER)) return source;
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
    const next = source[i + 1] ?? "";
    const isDigit = ch >= "0" && ch <= "9";
    const isDotStart =
      ch === "." && next >= "0" && next <= "9";
    if (!isDigit && !isDotStart) {
      out += ch;
      i += 1;
      continue;
    }
    // A digit glued to a name/number/attribute (`abc0j`, `1.5` revisited,
    // `x.1`) is not a literal start.
    const prev = out.length === 0 ? "" : out[out.length - 1]!;
    if (COMPLEX_PREV_CHAR.test(prev)) {
      out += ch;
      i += 1;
      continue;
    }
    const match = COMPLEX_LITERAL_RE.exec(source.slice(i));
    if (match === null) {
      out += ch;
      i += 1;
      continue;
    }
    const literal = match[0];
    const imagValue = Number(literal.slice(0, -1).replace(/_/g, ""));
    if (Number.isNaN(imagValue)) {
      out += ch;
      i += 1;
      continue;
    }
    out += `${WANIM_COMPLEX_HELPER}(0, ${String(imagValue)})`;
    i += literal.length;
  }
  return out;
}

// --- `reversed()` builtin desugar ------------------------------------------------
//
// rapydscript-ng's `ρσ_reversed` only accepts plain JS arrays and strings
// ("reversed() can only be called on arrays or strings"), while CPython's
// `reversed()` works on any object with `__reversed__` or the sequence
// protocol (`__len__` + `__getitem__`) -- notably wanim `Mobject`/`VGroup`
// (e.g. `zip(names, reversed(parts))`), which are iterable class instances.
// We rewrite bare `reversed(x)` calls into `__wan_reversed(x)` runtime calls
// (helper provided by the runtime in `runtime_core.ts`, mirroring CPython
// semantics and returning a plain array that the rest of the pipeline --
// `zip`, `for`, `list`, ... -- already handles).
//
// The scan skips string literals and `#` comments and never touches attribute
// access (`obj.reversed(...)` stays as-is). When user code defines its own
// `reversed` (`def`/`class`/assignment at line start) the source is left
// untouched so the local binding keeps working.
//
// Known limitations (documented in ): shadowing
// `reversed` via `lambda`/`global`/`nonlocal`/import alias, or calling it
// through an alias (`r = reversed; r(x)`), is not detected.

const REVERSED_NAME = "reversed";

/** True when user code binds its own `reversed` at statement level. */
function shadowsReversedBuiltin(source: string): boolean {
  return /(^|\n)[ \t]*(def|class)[ \t]+reversed(?![\p{L}\p{N}_])/u.test(source) ||
    /(^|\n)[ \t]*reversed[ \t]*=(?![=>])/m.test(source);
}

/**
 * Rewrite bare `reversed(...)` calls into `__wan_reversed(...)` (see above).
 * Anything that does not parse cleanly passes through untouched.
 */
export function desugarReversedBuiltin(source: string): string {
  // The helper name is reserved for the runtime (see `runtime_core.ts`); if
  // user code already uses it, leave the source alone rather than silently
  // calling the wrong function.
  if (source.includes(WANIM_REVERSED_HELPER)) return source;
  if (!source.includes(REVERSED_NAME)) return source;
  if (shadowsReversedBuiltin(source)) return source;
  let out = "";
  let i = 0;
  // Last non-whitespace character emitted (a `.` means attribute access).
  let previous = "";
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
      previous = '"';
      continue;
    }
    if (IDENT_START.test(ch)) {
      let j = i + 1;
      while (j < source.length && IDENT_PART.test(source[j]!)) j += 1;
      const word = source.slice(i, j);
      if (word === REVERSED_NAME && previous !== ".") {
        let k = j;
        while (k < source.length && /\s/.test(source[k]!)) k += 1;
        if (source[k] === "(") {
          out += WANIM_REVERSED_HELPER;
          previous = WANIM_REVERSED_HELPER[WANIM_REVERSED_HELPER.length - 1]!;
          i = j;
          continue;
        }
      }
      out += word;
      previous = word[word.length - 1]!;
      i = j;
      continue;
    }
    out += ch;
    if (!/\s/.test(ch)) previous = ch;
    i += 1;
  }
  return out;
}
