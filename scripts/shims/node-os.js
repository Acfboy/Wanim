// Browser build shim for `node:os`.
export const platform = () => "browser";
export const release = () => "browser";
export const cpus = () => [];
export const totalmem = () => 0;
export const freemem = () => 0;
export const EOL = "\n";

export default { platform, release, cpus, totalmem, freemem, EOL };
