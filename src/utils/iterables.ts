function sameValue(first: unknown, second: unknown): boolean {
  if (typeof first === "number" && typeof second === "number") return !Number.isNaN(first) && !Number.isNaN(second) && first === second;
  if (typeof first === "boolean" && typeof second === "number") return Number(first) === second;
  if (typeof first === "number" && typeof second === "boolean") return first === Number(second);
  return Object.is(first, second) || first === second;
}

function ensureHashable(value: unknown): void {
  if (Array.isArray(value) || value instanceof Map || value instanceof Set || (value !== null && typeof value === "object" && (value as { constructor?: { name?: string } }).constructor?.name === "Object")) {
    throw new TypeError(`unhashable type: '${Array.isArray(value) ? "list" : value instanceof Map ? "dict" : value instanceof Set ? "set" : "dict"}'`);
  }
}

export function remove_list_redundancies<T>(lst: readonly T[] | string | Iterable<T>): T[] {
  if (lst instanceof Set) throw new TypeError("'set' object is not reversible");
  const values = typeof lst === "string" ? [...lst] as T[] : Array.isArray(lst) ? [...lst] : Array.from(lst);
  const reversedResult: T[] = [];
  const used: T[] = [];
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    ensureHashable(value);
    if (!used.some((item) => sameValue(item, value))) {
      reversedResult.push(value);
      used.push(value);
    }
  }
  reversedResult.reverse();
  return reversedResult;
}
