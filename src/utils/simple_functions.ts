export function clip<T>(a: T, min_a: T, max_a: T): T {
  const kind = (value: unknown): string => value === null || value === undefined ? "none" : typeof value;
  const valueKind = kind(a);
  if (valueKind !== kind(min_a) || valueKind !== kind(max_a) || (valueKind !== "number" && valueKind !== "string" && valueKind !== "boolean")) {
    throw new TypeError(`'<' not supported between instances of '${valueKind === "number" ? "float" : valueKind === "string" ? "str" : valueKind === "boolean" ? "bool" : "NoneType"}' and '${kind(min_a) === "number" ? "int" : kind(min_a) === "string" ? "str" : kind(min_a) === "boolean" ? "bool" : "NoneType"}'`);
  }
  const comparable = a as any;
  const minimum = min_a as any;
  const maximum = max_a as any;
  if (comparable < minimum) return min_a;
  if (comparable > maximum) return max_a;
  return a;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
