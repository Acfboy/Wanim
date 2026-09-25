export interface MethodWithArgsLike {
  method: (...args: any[]) => any;
  args: any[] | Iterable<any>;
  kwargs: Record<string, any>;
}

export class MethodWithArgs {
  method: (...args: any[]) => any;
  args: any[];
  kwargs: Record<string, any>;

  constructor(method: (...args: any[]) => any, args?: any[] | Iterable<any>, kwargs?: Record<string, any>) {
    if (arguments.length < 3) throw new TypeError("MethodWithArgs.__init__() missing required arguments");
    this.method = method;
    this.args = args === undefined ? [] : Array.isArray(args) ? args : Array.from(args);
    this.kwargs = kwargs as Record<string, any>;
  }
}
