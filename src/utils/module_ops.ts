import { existsSync } from "node:fs";
import { extname, resolve, sep } from "node:path";

export interface SceneModule {
  __name__: string;
  __file__: string;
  [key: string]: unknown;
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  throw error;
}

function fileNotFoundError(message: string): Error {
  const error = new Error(message);
  error.name = "FileNotFoundError";
  throw error;
}

/** Load a scene module from a file path (or "-" for stdin).
 *
 * Faithful in its validation behavior (suffix / existence checks); dynamic
 * Python execution has no TS equivalent, so a successfully located file
 * yields a lightweight module record instead of an executed namespace.
 */
export function get_module(file_name: string): SceneModule {
  if (file_name === "-") {
    throw new Error("TODO: get_module('-') requires stdin code execution; not ported yet");
  }
  const resolved = resolve(file_name);
  if (!existsSync(resolved)) {
    throw fileNotFoundError(`${file_name} not found`);
  }
  if (extname(resolved) !== ".py") {
    throw valueError(`${file_name} is not a valid Manim python script.`);
  }
  const module_name = resolved
    .replace(/\.py$/, "")
    .split(sep)
    .filter((part: string) => part.length > 0)
    .join(".");
  return { __name__: module_name, __file__: resolved };
}
