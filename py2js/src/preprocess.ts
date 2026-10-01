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
