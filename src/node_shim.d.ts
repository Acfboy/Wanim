declare module "node:fs" {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string, options: { encoding: string }): string;
  export function writeFileSync(path: string, data: string, options?: { encoding: string }): void;
  export function appendFileSync(path: string, data: string, options?: { encoding: string }): void;
  export function readdirSync(path: string): string[];
  export function unlinkSync(path: string): void;
  export function accessSync(path: string, mode?: number): void;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export interface Stats {
    mtime: Date;
    mtimeMs: number;
    atimeMs: number;
    isDirectory(): boolean;
    isFile(): boolean;
  }
  export function statSync(path: string): Stats;
  export function utimesSync(path: string, atime: Date, mtime: Date): void;
  export const constants: { R_OK: number; W_OK: number; X_OK: number; F_OK: number };
}

declare module "node:child_process" {
  export function spawnSync(
    command: string,
    args?: readonly string[],
    options?: unknown,
  ): unknown;
}

declare module "node:path" {
  export interface ParsedPath {
    dir: string;
    name: string;
    ext: string;
  }
  export function join(...parts: string[]): string;
  export function parse(path: string): ParsedPath;
  export function extname(path: string): string;
  export function dirname(path: string): string;
  export function resolve(...parts: string[]): string;
  export const sep: string;
}

declare module "node:os" {
  export function platform(): string;
}

declare const process: {
  cwd(): string;
  env: Record<string, string | undefined>;
};
