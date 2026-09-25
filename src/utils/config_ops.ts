export interface HasUniforms {
  uniforms: Record<string, unknown>;
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
