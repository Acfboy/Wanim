/**
 * Turn a rapydscript compiler error into a readable message.
 *
 * rapydscript raises `SyntaxError`-like objects carrying `message`, `line`
 * (1-based) and `col` (0-based) rather than native errors, and their bare
 * message does not say *where* the mistake is.  This renders the offending
 * source line with a caret under the reported column so low-level mistakes
 * such as an unclosed parenthesis are immediately obvious.
 */
export interface CompilerErrorLike {
  message?: unknown;
  line?: unknown;
  col?: unknown;
  filename?: unknown;
}

const DEFAULT_FILENAME = "scene.py";

export function formatCompileError(error: unknown, source: string): Error {
  if (error === null || typeof error !== "object") {
    return error instanceof Error ? error : new Error(String(error));
  }
  const e = error as CompilerErrorLike;
  const message = typeof e.message === "string" ? e.message : String(error);
  const line = typeof e.line === "number" ? e.line : null;
  if (line === null || line < 1) {
    return error instanceof Error ? error : new Error(message);
  }

  const lines = source.split(/\r\n?|\n/);
  const text = lines[line - 1] ?? "";
  const col = typeof e.col === "number" ? Math.max(0, e.col) : 0;
  // Keep the caret on the line even if the column points past its end (EOF).
  const caret = " ".repeat(Math.min(col, text.length)) + "^";
  const filename =
    typeof e.filename === "string" && e.filename.length > 0 ? e.filename : DEFAULT_FILENAME;

  return new Error(`${message}\n  ${filename}:${line}:${col + 1}\n\n${text}\n${caret}`);
}
