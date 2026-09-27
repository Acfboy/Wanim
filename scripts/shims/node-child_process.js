// Browser build shim for `node:child_process`.  Needing it in the browser
// means a TeX/typst/ffmpeg code path was reached, which cannot work there.
function unavailable(name) {
  return () => {
    throw new Error(`wanim browser build: child_process.${name}() is not available.`);
  };
}

export const spawnSync = unavailable("spawnSync");
export const spawn = unavailable("spawn");
export const execSync = unavailable("execSync");
export const exec = unavailable("exec");

export default { spawnSync, spawn, execSync, exec };
