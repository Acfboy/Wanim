export interface HasData {
  data: Record<string, unknown>;
}

export interface HasUniforms {
  uniforms: Record<string, unknown>;
}

export interface _HasData {
  data: Record<string, unknown>;
}

export interface _HasUniforms {
  uniforms: Record<string, number | readonly number[]>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

export function merge_dicts_recursively(...dicts: Array<Record<string, unknown>>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const dict of dicts) {
    for (const [key, value] of Object.entries(dict)) {
      if (key in result && isPlainObject(result[key]) && isPlainObject(value)) {
        result[key] = merge_dicts_recursively(
          result[key] as Record<string, unknown>,
          value as Record<string, unknown>,
        );
      } else {
        result[key] = value;
      }
    }
  }
  return result;
}

export function update_dict_recursively(current_dict: Record<string, unknown>, ...others: Array<Record<string, unknown>>): void {
  const updated = merge_dicts_recursively(current_dict, ...others);
  for (const [key, value] of Object.entries(updated)) {
    current_dict[key] = value;
  }
}

export class _Data<T = unknown> {
  name = "";

  __set_name__(_obj: unknown, name: string): void {
    this.name = name;
  }

  __get__(obj: HasData): T {
    return obj.data[this.name] as T;
  }

  __set__(obj: HasData, value: T): void {
    obj.data[this.name] = value;
  }
}

export class _Uniforms<T = unknown> {
  name = "";

  __set_name__(_obj: unknown, name: string): void {
    this.name = name;
  }

  __get__(obj: HasUniforms): T {
    return obj.uniforms[this.name] as T;
  }

  __set__(obj: HasUniforms, num: T): void {
    obj.uniforms[this.name] = num;
  }
}

export class DictAsObject {
  [key: string]: unknown;

  constructor(dictin: Record<string, unknown>) {
    Object.assign(this, dictin);
  }
}
