import { existsSync, mkdirSync, readFileSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, extname, join, parse, resolve } from "node:path";
import { platform } from "node:os";
import { config } from "../_config/utils.js";

export const MANIM_VERSION = "0.21.0";

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
    possible_paths.push(join(default_dir, `${file_name}${extension}`));
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

export function is_mp4_format(): boolean {
  return config.get("format") === "mp4";
}

export function is_gif_format(): boolean {
  return config.get("format") === "gif";
}

export function is_webm_format(): boolean {
  return config.get("format") === "webm";
}

export function is_mov_format(): boolean {
  return config.get("format") === "mov";
}

export function is_png_format(): boolean {
  return config.get("format") === "png";
}

export function add_version_before_extension(file_name: string): string {
  const parsed = parse(file_name);
  return join(parsed.dir, `${parsed.name}_ManimCE_v${MANIM_VERSION}${parsed.ext}`);
}

export function open_file(file_path: string, in_browser = false): void {
  const current_os = platform();
  let target = file_path;
  if (in_browser) {
    target = dirname(file_path);
  }
  const preview_command = config.get("preview_command") as string;
  if (preview_command) {
    spawnSync(preview_command, [target], { stdio: "ignore", shell: true });
    return;
  }
  if (current_os === "win32") {
    spawnSync("cmd", ["/c", "start", "", target], { stdio: "ignore", shell: false });
    return;
  }
  if (current_os === "linux") {
    spawnSync("xdg-open", [target], { stdio: "ignore" });
    return;
  }
  if (current_os === "cygwin" || current_os.startsWith("CYGWIN")) {
    spawnSync("cygstart", [target], { stdio: "ignore" });
    return;
  }
  if (current_os === "darwin") {
    if (in_browser) {
      spawnSync("open", ["-R", file_path], { stdio: "ignore" });
    } else {
      spawnSync("open", [target], { stdio: "ignore" });
    }
    return;
  }
  const error = new Error("Unable to identify your operating system...");
  error.name = "OSError";
  throw error;
}

export function add_import_statement(file: string): void {
  const import_line = "from manim import *";
  const content = readFileSync(file, { encoding: "utf-8" });
  writeFileSync(file, `${import_line}\n${content}`, { encoding: "utf-8" });
}

export function guarantee_existence(path: string): string {
  mkdirSync(path, { recursive: true });
  const resolved = resolve(path);
  if (!existsSync(resolved)) {
    const error = new Error(`No such file or directory: '${resolved}'`);
    error.name = "OSError";
    throw error;
  }
  return resolved;
}

export function add_extension_if_not_present(file_name: string, extension: string): string {
  if (extname(file_name) !== extension) return file_name + extension;
  return file_name;
}

export function modify_atime(file_path: string): void {
  const mtime = statSync(file_path).mtime;
  utimesSync(file_path, new Date(), mtime);
}
