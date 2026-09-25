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
