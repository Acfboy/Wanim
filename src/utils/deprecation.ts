export type AnyCallable = (...args: unknown[]) => unknown;

export interface CallableWithMeta extends AnyCallable {
  __qualname__?: string;
  __class__?: { name?: string };
  prototype?: unknown;
}

function callableTypeName(callable_: AnyCallable): string {
  const proto = (callable_ as CallableWithMeta).prototype;
  if (typeof callable_ === "function" && proto !== undefined && (callable_ as Function).prototype !== undefined) {
    try {
      const source = Function.prototype.toString.call(callable_);
      if (source.startsWith("class ")) return "class";
    } catch {
      /* fall through */
    }
  }
  return "function";
}

export function _get_callable_info(callable_: AnyCallable): [string, string] {
  let what = callableTypeName(callable_);
  const name =
    (callable_ as CallableWithMeta).__qualname__ ?? callable_.name ?? "";
  if (what === "function" && name.includes(".")) what = "method";
  else if (what !== "function") what = "class";
  return [what, name];
}

export function _deprecation_text_component(
  since: string | null = null,
  until: string | null = null,
  message: string | null = null,
): string {
  const sincePart = since ? `since ${since} ` : "";
  const untilPart = until
    ? `is expected to be removed after ${until}`
    : "may be removed in a later version";
  const msg = message ? ` ${message}` : "";
  return `deprecated ${sincePart}and ${untilPart}.${msg}`;
}

export interface DeprecatedOptions {
  since?: string | null;
  until?: string | null;
  replacement?: string | null;
  message?: string | null;
}

export function deprecated(
  func?: AnyCallable | null,
  since: string | null = null,
  until: string | null = null,
  replacement: string | null = null,
  message: string | null = "",
): AnyCallable | ((func: AnyCallable) => AnyCallable) {
  // If used as factory:
  if (func === null || func === undefined) {
    return (inner: AnyCallable) =>
      deprecated(inner, since, until, replacement, message) as AnyCallable;
  }

  const [what, name] = _get_callable_info(func);

  function warning_msg(for_docs = false): string {
    let msg = message;
    if (replacement !== null && replacement !== undefined) {
      let repl = replacement;
      if (for_docs) {
        const mapper: Record<string, string> = {
          class: "class",
          method: "meth",
          function: "func",
        };
        repl = `:${mapper[what]}:\`~.${replacement}\``;
      }
      msg = `Use ${repl} instead.${msg ? ` ${msg}` : ""}`;
    }
    const deprecatedText = _deprecation_text_component(since, until, msg);
    return `The ${what} ${name} has been ${deprecatedText}`;
  }

  function deprecate_docs(target: AnyCallable): void {
    const warning = warning_msg(true);
    const docString =
      (target as AnyCallable & { __doc__?: string }).__doc__ ?? "";
    (target as AnyCallable & { __doc__?: string }).__doc__ =
      `${docString}\n\n.. attention:: Deprecated\n  ${warning}`;
  }

  function deprecate(
    this: unknown,
    ...args: unknown[]
  ): unknown {
    console.warn(warning_msg());
    return (func as (...a: unknown[]) => unknown).apply(this, args);
  }

  if (callableTypeName(func) !== "function") {
    deprecate_docs(func);
    const originalInit = (
      func as unknown as { prototype: { constructor: AnyCallable } }
    ).prototype?.constructor;
    void originalInit;
    const Original = func as unknown as new (...args: unknown[]) => object;
    class DeprecatedClass extends Original {
      constructor(...args: unknown[]) {
        console.warn(warning_msg());
        super(...args);
      }
    }
    Object.defineProperty(DeprecatedClass, "name", {
      value: func.name,
      configurable: true,
    });
    (DeprecatedClass as unknown as CallableWithMeta).__qualname__ = (
      func as CallableWithMeta
    ).__qualname__ ?? func.name;
    deprecate_docs(DeprecatedClass as unknown as AnyCallable);
    return DeprecatedClass as unknown as AnyCallable;
  }

  Object.defineProperty(deprecate, "name", {
    value: func.name,
    configurable: true,
  });
  (deprecate as CallableWithMeta).__qualname__ =
    (func as CallableWithMeta).__qualname__ ?? func.name;
  deprecate_docs(deprecate as AnyCallable);
  return deprecate as AnyCallable;
}

export type Redirector =
  | [string, string]
  | ((...args: unknown[]) => Record<string, unknown>);

export interface RedirectorFn extends AnyCallable {
  paramNames?: string[];
}

function redirectorParamNames(redirector: Redirector): string[] {
  if (Array.isArray(redirector)) return [];
  const fn = redirector as RedirectorFn;
  if (fn.paramNames) return fn.paramNames;
  const source = Function.prototype.toString.call(redirector);
  const match =
    source.match(/^[^(]*\(([^)]*)\)/) ??
    source.match(/^\(?([^)=]*?)\)?\s*=>/);
  const raw = (match?.[1] ?? "").trim();
  if (!raw) return [];
  return raw
    .split(",")
    .map((part) =>
      part
        .trim()
        .replace(/=.*$/, "")
        .replace(/^\.{3}/, "")
        .trim(),
    )
    .filter((part) => part.length > 0);
}

export function redirect_params(
  kwargs: Record<string, unknown>,
  used: string[],
  redirections: Redirector[],
): void {
  for (const redirector of redirections) {
    if (Array.isArray(redirector)) {
      const [old_param, new_param] = redirector;
      if (used.includes(old_param)) {
        kwargs[new_param] = kwargs[old_param];
        delete kwargs[old_param];
      }
    } else {
      const redirector_params = redirectorParamNames(redirector);
      const redirector_args: Record<string, unknown> = {};
      for (const redirector_param of redirector_params) {
        if (used.includes(redirector_param)) {
          redirector_args[redirector_param] = kwargs[redirector_param];
          delete kwargs[redirector_param];
        }
      }
      if (Object.keys(redirector_args).length > 0) {
        const values = redirector_params.map((name) =>
          name in redirector_args ? redirector_args[name] : undefined,
        );
        Object.assign(
          kwargs,
          (redirector as (...args: unknown[]) => Record<string, unknown>)(
            ...values,
          ),
        );
      }
    }
  }
}

export interface DeprecatedParamsOptions {
  since?: string | null;
  until?: string | null;
  message?: string;
  redirections?: Redirector[];
}

function warning_msg(
  func: AnyCallable,
  used: string[],
  since: string | null,
  until: string | null,
  message: string,
): string {
  const [what, name] = _get_callable_info(func);
  const plural = used.length > 1;
  const parameter_s = plural ? "s" : "";
  const used_ =
    plural
      ? `${used.slice(0, -1).join(", ")} and ${used[used.length - 1]}`
      : used[0];
  const has_have_been = plural ? "have been" : "has been";
  const deprecated = _deprecation_text_component(since, until, message);
  return `The parameter${parameter_s} ${used_} of ${what} ${name} ${has_have_been} ${deprecated}`;
}

export function deprecated_params(
  params: string | Iterable<string> | null = null,
  since: string | null = null,
  until: string | null = null,
  message = "",
  redirections: Redirector[] | null = null,
): (func: AnyCallable) => AnyCallable {
  if (typeof params === "function") {
    throw new Error(
      "deprecate_parameters requires arguments to be specified.",
    );
  }

  let paramList: string[] =
    params === null
      ? []
      : typeof params === "string"
        ? params.split(/[,\s]+/)
        : [...params];

  const redirects: Redirector[] = redirections === null ? [] : [...redirections];
  for (const redirector of redirects) {
    if (Array.isArray(redirector)) {
      paramList.push(redirector[0]);
    } else {
      paramList.push(...redirectorParamNames(redirector));
    }
  }
  paramList = [...new Set(paramList)];

  const identifier = /^[^\d\W]\w*$/u;
  if (!paramList.every((param) => identifier.test(param))) {
    throw new Error("Given parameter values are invalid.");
  }

  function redirect_params_closure(
    kwargs: Record<string, unknown>,
    used: string[],
  ): void {
    redirect_params(kwargs, used, redirects);
  }

  return function deprecate_params_decorator(func: AnyCallable): AnyCallable {
    function deprecate_params(
      this: unknown,
      ...args: unknown[]
    ): unknown {
      const kwargs =
        args.length > 0 &&
        args[args.length - 1] !== null &&
        typeof args[args.length - 1] === "object" &&
        !Array.isArray(args[args.length - 1])
          ? (args[args.length - 1] as Record<string, unknown>)
          : null;
      const used =
        kwargs === null
          ? []
          : paramList.filter((param) => param in kwargs);
      if (used.length > 0 && kwargs !== null) {
        console.warn(warning_msg(func, used, since, until, message));
        redirect_params_closure(kwargs, used);
      }
      return (func as (...a: unknown[]) => unknown).apply(this, args);
    }
    Object.defineProperty(deprecate_params, "name", {
      value: func.name,
      configurable: true,
    });
    (deprecate_params as CallableWithMeta).__qualname__ =
      (func as CallableWithMeta).__qualname__ ?? func.name;
    return deprecate_params;
  };
}
