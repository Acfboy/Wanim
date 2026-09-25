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
