// Minimal browser POSIX `node:path` shim (enough for the config/font helpers
// that survive tree-shaking into the bundle).
export const sep = "/";
export const delimiter = ":";

export function join(...parts) {
  return parts.filter((part) => part && part.length > 0).join("/").replace(/\/+/g, "/");
}

export function resolve(...parts) {
  let resolved = "";
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    const part = parts[index];
    if (!part) continue;
    resolved = `${part}/${resolved}`;
    if (part.startsWith("/")) break;
  }
  return normalize(resolved);
}

export function normalize(value) {
  const absolute = value.startsWith("/");
  const trailing = value.endsWith("/");
  const segments = [];
  for (const segment of value.split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") segments.pop();
    else segments.push(segment);
  }
  return `${absolute ? "/" : ""}${segments.join("/")}${trailing && segments.length > 0 ? "/" : ""}`;
}

export function dirname(value) {
  const index = value.lastIndexOf("/");
  if (index < 0) return ".";
  if (index === 0) return "/";
  return value.slice(0, index);
}

export function basename(value, ext = "") {
  const base = value.slice(value.lastIndexOf("/") + 1);
  return ext && base.endsWith(ext) ? base.slice(0, -ext.length) : base;
}

export function extname(value) {
  const base = basename(value);
  const index = base.lastIndexOf(".");
  return index <= 0 ? "" : base.slice(index);
}

export function parse(value) {
  const dir = dirname(value);
  const ext = extname(value);
  const base = basename(value);
  return { root: value.startsWith("/") ? "/" : "", dir, base, ext, name: base.slice(0, base.length - ext.length) };
}

export default { sep, delimiter, join, resolve, normalize, dirname, basename, extname, parse };
