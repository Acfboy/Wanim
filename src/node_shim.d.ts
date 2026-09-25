declare module "node:fs" {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string, options: { encoding: string }): string;
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
}

declare const process: {
  cwd(): string;
  env: Record<string, string | undefined>;
};
