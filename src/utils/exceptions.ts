export class EndSceneEarlyException extends Error {
  constructor(message = "") {
    super(message);
    this.name = "EndSceneEarlyException";
  }
}

export class RerunSceneException extends Error {
  constructor(message = "") {
    super(message);
    this.name = "RerunSceneException";
  }
}

export class MultiAnimationOverrideException extends Error {
  args: unknown[];

  constructor(...args: unknown[]) {
    const message = args.length === 0 ? "" : args.length === 1 ? String(args[0]) : `(${args.map((value) => typeof value === "string" ? `'${value}'` : String(value)).join(", ")})`;
    super(message);
    this.name = "MultiAnimationOverrideException";
    this.args = args;
  }
}
