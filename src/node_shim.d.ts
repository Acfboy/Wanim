declare module "node:fs" {
  export function existsSync(path: string): boolean;
}

declare module "node:path" {
  export function join(...parts: string[]): string;
}

declare const process: {
  cwd(): string;
  env: Record<string, string | undefined>;
};
