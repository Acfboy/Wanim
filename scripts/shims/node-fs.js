// Browser build shim for `node:fs`.
//
// wanim's Node entry points read config files, TeX caches and fonts from disk.
// None of that exists in the browser; the Canvas2D demo does not call these
// paths (MathTex uses MathJax, not a local LaTeX install).  Anything that does
// reach here fails loudly instead of silently misbehaving.
function unavailable(name) {
  return (...args) => {
    throw new Error(
      `wanim browser build: fs.${name}() is not available (browser has no filesystem).`,
    );
  };
}

export const existsSync = () => false;
export const readFileSync = unavailable("readFileSync");
export const writeFileSync = unavailable("writeFileSync");
export const appendFileSync = unavailable("appendFileSync");
export const mkdirSync = unavailable("mkdirSync");
export const statSync = unavailable("statSync");
export const accessSync = unavailable("accessSync");
export const utimesSync = unavailable("utimesSync");
export const unlinkSync = unavailable("unlinkSync");
export const readdirSync = () => [];
export const constants = { R_OK: 4, W_OK: 2, F_OK: 0 };

export default {
  existsSync,
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  statSync,
  accessSync,
  utimesSync,
  unlinkSync,
  readdirSync,
  constants,
};
