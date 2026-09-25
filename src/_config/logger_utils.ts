export interface SectionProxyLike extends Iterable<[string, string] | string> {
  get(key: string): string;
  [key: string]: string | ((...args: any[]) => any) | Iterable<[string, string]>;
}

export interface RichTheme {
  [key: string]: string | number | boolean | null;
}

function entriesOf(parser: SectionProxyLike): [string, string][] {
  if (typeof (parser as unknown as Map<string, string>).entries === "function" && parser instanceof Map) return [...(parser as unknown as Map<string, string>).entries()];
  const result: [string, string][] = [];
  if (typeof (parser as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function") {
    for (const item of parser) {
      if (Array.isArray(item) && item.length >= 2) result.push([String(item[0]), String(item[1])]);
      else if (typeof item === "string") result.push([item, valueOf(parser, item)]);
    }
  }
  if (result.length === 0) {
    for (const [key, value] of Object.entries(parser as Record<string, unknown>)) {
      if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") result.push([key, String(value)]);
    }
  }
  return result;
}

function valueOf(parser: SectionProxyLike, key: string): string {
  const direct = (parser as unknown as Record<string, unknown>)[key];
  if (typeof direct === "string" || typeof direct === "number" || typeof direct === "boolean") return String(direct);
  const value = parser.get(key);
  return String(value);
}

const namedStyles = new Set([
  "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", "gray",
  "bright_black", "bright_red", "bright_green", "bright_yellow", "bright_blue", "bright_magenta", "bright_cyan", "bright_white",
  "bold", "dim", "italic", "underline", "blink", "blink2", "reverse", "strike", "conceal", "frame", "encircle", "overline", "none",
]);

function validStyle(value: string): boolean {
  const source = value.trim().toLowerCase();
  if (source === "") return true;
  if (/^(?:#[0-9a-f]{3,8}|0x[0-9a-f]{3,8})$/.test(source)) return true;
  const functional = source.match(/^(rgb|rgba|hsl|hsla)\((.*)\)$/);
  if (functional) {
    const parts = functional[2].split(",").map((part) => part.trim());
    if (!parts.every((part) => /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?%?$/i.test(part))) return false;
    return functional[1] === "rgb" || functional[1] === "hsl" ? parts.length === 3 : parts.length === 4;
  }
  return source.split(/\s+/).every((part) => namedStyles.has(part));
}

export interface LogRecord {
  levelname: string;
  module: string;
  message: string;
  args?: unknown;
  [key: string]: unknown;
}

export class JSONFormatter {
  format(record: LogRecord): string {
    return JSON.stringify({
      levelname: record.levelname,
      module: record.module,
      message: getMaskedMessage(record),
    });
  }
}

function getMaskedMessage(record: LogRecord): string {
  // Mirrors logging.Formatter.format(record) after masking record.args with
  // "<>" (mapping values key-by-key, tuples positionally). The input record
  // is never mutated (Python deep-copies it first).
  const args = record.args;
  if (args === null || args === undefined) return record.message;
  if (typeof args === "object" && !Array.isArray(args)) {
    return record.message.replace(
      /%\(([^)]+)\)[sdri]/g,
      (_match, _key) => "<>",
    );
  }
  const values = (Array.isArray(args) ? args : [args]) as unknown[];
  const masked = new Array(values.length).fill("<>");
  let index = 0;
  void masked;
  return record.message.replace(/%[sdri]/g, () => {
    index += 1;
    return "<>";
  });
}

export interface ManimLogger {
  debug: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  warning: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
  level: string;
}

export interface ManimConsole {
  print: (...args: unknown[]) => void;
}

import { appendFileSync, writeFileSync } from "node:fs";

// NOTE: mirrors logging.FileHandler(log_path, mode="w") + JSONFormatter
// attached to the shared "manim" logger. The TS port has no shared logger
// registry (make_logger returns standalone loggers), so file targets are
// kept in a module-level registry: set_file_logger truncates/creates the
// file, and loggers created by make_logger append one JSON line per call.
const fileLoggerPaths: string[] = [];

export function get_file_logger_paths(): string[] {
  return [...fileLoggerPaths];
}

function appendToFileLoggers(levelname: string, args: unknown[]): void {
  if (fileLoggerPaths.length === 0) return;
  const message = args.map((arg) => {
    if (typeof arg === "string") return arg;
    try {
      return JSON.stringify(arg) ?? String(arg);
    } catch {
      return String(arg);
    }
  }).join(" ");
  const line = JSON.stringify({ levelname, module: "manim", message }) + "\n";
  for (const path of fileLoggerPaths) {
    try {
      appendFileSync(path, line, { encoding: "utf-8" });
    } catch {
      // NOTE: logging must never crash the program; ignore file errors.
    }
  }
}

export function set_file_logger(
  scene_name: string,
  module_name: string,
  log_dir: string | { toString(): string },
): void {
  // NOTE: the log file name is <module>_<scene>.log (see the Python
  // docstring); the file is opened with mode="w", i.e. truncated.
  const log_file_name = `${String(module_name)}_${String(scene_name)}.log`;
  const dir = String(log_dir);
  const log_file_path = dir.endsWith("/") || dir.endsWith("\\")
    ? `${dir}${log_file_name}`
    : `${dir}/${log_file_name}`;
  writeFileSync(log_file_path, "", { encoding: "utf-8" });
  if (!fileLoggerPaths.includes(log_file_path)) fileLoggerPaths.push(log_file_path);
}

export function make_logger(
  parser: SectionProxyLike,
  verbosity: string,
): [ManimLogger, ManimConsole, ManimConsole] {
  const theme = parse_theme(parser);
  void theme;
  const levelOrder = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
  const active = verbosity.toUpperCase();
  const activeIndex = levelOrder.indexOf(active);

  const makeConsole = (stderr: boolean): ManimConsole => ({
    print: (...args: unknown[]): void => {
      if (stderr) globalThis.console.error(...args);
      else globalThis.console.log(...args);
    },
  });
  const stdoutConsole = makeConsole(false);
  const errorConsole = makeConsole(true);

  const enabled = (level: string): boolean => {
    const index = levelOrder.indexOf(level);
    return activeIndex === -1 || index >= activeIndex;
  };
  const logger: ManimLogger = {
    level: active,
    debug: (...args: unknown[]): void => { if (enabled("DEBUG")) { stdoutConsole.print(...args); appendToFileLoggers("DEBUG", args); } },
    info: (...args: unknown[]): void => { if (enabled("INFO")) { stdoutConsole.print(...args); appendToFileLoggers("INFO", args); } },
    warning: (...args: unknown[]): void => { if (enabled("WARNING")) { errorConsole.print(...args); appendToFileLoggers("WARNING", args); } },
    warn: (...args: unknown[]): void => { if (enabled("WARNING")) { errorConsole.print(...args); appendToFileLoggers("WARNING", args); } },
    error: (...args: unknown[]): void => { if (enabled("ERROR")) { errorConsole.print(...args); appendToFileLoggers("ERROR", args); } },
  };
  return [logger, stdoutConsole, errorConsole];
}

export function parse_theme(parser: SectionProxyLike): RichTheme | null {
  const theme: RichTheme = {};
  for (const [key, value] of entriesOf(parser)) theme[key.replaceAll("_", ".")] = value;
  const width = theme["log.width"];
  const height = theme["log.height"];
  if (width === undefined) {
    const error = new Error("'log.width'");
    error.name = "KeyError";
    throw error;
  }
  if (height === undefined) {
    const error = new Error("'log.height'");
    error.name = "KeyError";
    throw error;
  }
  const parseDimension = (value: string | number | boolean | null): number | null => {
    if (String(value) === "-1") return null;
    const text = String(value);
    if (!/^[+-]?\d+$/.test(text)) {
      const error = new Error(`invalid literal for int() with base 10: '${text}'`);
      error.name = "ValueError";
      throw error;
    }
    return Number(text);
  };
  const parsedWidth = parseDimension(width);
  const parsedHeight = parseDimension(height);
  theme["log.width"] = parsedWidth;
  theme["log.height"] = parsedHeight;
  theme["log.timestamps"] = false;
  const custom: RichTheme = {};
  for (const [key, value] of Object.entries(theme)) {
    if (key === "log.width" || key === "log.height" || key === "log.timestamps") continue;
    if (typeof value === "string" && !validStyle(value)) return null;
    custom[key] = value;
  }
  return custom;
}
