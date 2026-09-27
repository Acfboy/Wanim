// Browser build shim for `node:fs/promises` (used only by numpy-ts' guarded
// Node file-IO path, which never runs in the browser).
function unavailable(name) {
  return async () => {
    throw new Error(`wanim browser build: fs/promises.${name}() is not available.`);
  };
}

export const readFile = unavailable("readFile");
export const writeFile = unavailable("writeFile");
export const readdir = unavailable("readdir");
export const mkdir = unavailable("mkdir");
export const stat = unavailable("stat");
export const unlink = unavailable("unlink");

export default { readFile, writeFile, readdir, mkdir, stat, unlink };
