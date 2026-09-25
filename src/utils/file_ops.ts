import { existsSync } from "node:fs";
import * as path from "node:path";

const templateNames = ["Axes", "Default", "MovingCamera"];

function templateUrl(): string {
  return decodeURIComponent(new URL("../../src/templates", import.meta.url).pathname).replace(/\/$/, "");
}

export function get_template_names(): string[] {
  return [...templateNames];
}

export function get_template_path(): string {
  return templateUrl();
}

export function seek_full_path_from_defaults(
  file_name: string,
  default_dir: string,
  extensions: string[],
): string {
  const possible_paths = [expanduser(file_name)];
  for (const extension of ["", ...extensions]) {
    possible_paths.push(path.join(default_dir, `${file_name}${extension}`));
  }
  for (const candidate of possible_paths) {
    if (existsSync(candidate)) return candidate;
  }
  const error = new Error(
    `From: ${process.cwd()}, could not find ${file_name} at either of these locations: [${possible_paths.map((p) => `'${p}'`).join(", ")}]`,
  );
  error.name = "OSError";
  throw error;
}

function expanduser(file_name: string): string {
  if (file_name === "~") return process.env["HOME"] ?? file_name;
  if (file_name.startsWith("~/")) return (process.env["HOME"] ?? "~") + file_name.slice(1);
  return file_name;
}
