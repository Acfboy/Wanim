function isIterableValue(value: unknown): value is Iterable<unknown> {
  return (
    value !== null &&
    value !== undefined &&
    typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function"
  );
}

export function flatten_iterable_parameters<T>(args: Iterable<T | Iterable<T>>): T[] {
  const flattened_parameters: T[] = [];
  for (const arg of args) {
    if (isIterableValue(arg)) {
      for (const item of arg as Iterable<T>) {
        flattened_parameters.push(item);
      }
    } else {
      flattened_parameters.push(arg as T);
    }
  }
  return flattened_parameters;
}
