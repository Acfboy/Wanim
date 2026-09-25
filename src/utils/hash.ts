const constants = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

function rightRotate(value: number, amount: number): number {
  return (value >>> amount) | (value << (32 - amount));
}

export function sha256Hex(value: string): string {
  const bytes = new TextEncoder().encode(value);
  const bitLength = bytes.length * 8;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const data = new Uint8Array(paddedLength);
  data.set(bytes);
  data[bytes.length] = 0x80;
  const view = new DataView(data.buffer);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000), false);
  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;
  const words = new Uint32Array(64);
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(offset + index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const s0 = rightRotate(words[index - 15], 7) ^ rightRotate(words[index - 15], 18) ^ (words[index - 15] >>> 3);
      const s1 = rightRotate(words[index - 2], 17) ^ rightRotate(words[index - 2], 19) ^ (words[index - 2] >>> 10);
      words[index] = (words[index - 16] + s0 + words[index - 7] + s1) >>> 0;
    }
    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;
    for (let index = 0; index < 64; index += 1) {
      const s1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + constants[index] + words[index]) >>> 0;
      const s0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }
  return [h0, h1, h2, h3, h4, h5, h6, h7].map((value) => value.toString(16).padStart(8, "0")).join("");
}

export const KEYS_TO_FILTER_OUT = new Set([
  "original_id",
  "background",
  "pixel_array",
  "pixel_array_to_cairo_context",
]);

function pythonRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "string") return `'${value}'`;
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number" || typeof value === "bigint") return String(value);
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

export function _canonical_field_title(title: unknown): unknown {
  if (title === null || title === undefined) {
    return null;
  }
  // Map JS runtime types onto Python type names: Python reports
  // builtins.int for ints, builtins.float for floats, builtins.str,
  // builtins.bool. JS typeof/ctor names (Number/String/Boolean) differ,
  // so translate explicitly to match `type(x).__module__/__qualname__`.
  let qualName: string;
  switch (typeof title) {
    case "number":
      qualName = Number.isInteger(title) ? "int" : "float";
      break;
    case "string":
      qualName = "str";
      break;
    case "boolean":
      qualName = "bool";
      break;
    case "bigint":
      qualName = "int";
      break;
    default: {
      const titleType = (title as object)?.constructor;
      qualName = titleType?.name ?? typeof title;
      break;
    }
  }
  const moduleName = "builtins";
  return [`${moduleName}.${qualName}`, pythonRepr(title)];
}

function stringHash32(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(31, hash) + value.charCodeAt(index);
    hash |= 0;
  }
  return hash;
}

let nextObjectId = 1;
const objectIds = new WeakMap<object, number>();

function identityOf(value: object): number {
  let id = objectIds.get(value);
  if (id === undefined) {
    id = nextObjectId;
    nextObjectId += 1;
    objectIds.set(value, id);
  }
  return id;
}

function hashOf(value: unknown): number {
  if (value === null || value === undefined) return stringHash32(String(value));
  switch (typeof value) {
    case "number":
      return stringHash32(`float:${String(value)}`);
    case "bigint":
      return stringHash32(`int:${String(value)}`);
    case "boolean":
      return value ? 1 : 0;
    case "string":
      return stringHash32(value);
    case "function":
      return stringHash32(Function.prototype.toString.call(value));
    default:
      break;
  }
  if (typeof value === "object") {
    try {
      return stringHash32(JSON.stringify(value) ?? String(identityOf(value as object)));
    } catch {
      return identityOf(value as object);
    }
  }
  return stringHash32(String(value));
}

export class _Memoizer {
  static ALREADY_PROCESSED_PLACEHOLDER = "AP";
  static THRESHOLD_WARNING = 170_000;

  private _already_processed = new Set<string>();
  private _keep_alive = new Map<number, unknown>();

  constructor() {
    this._already_processed = new Set<string>();
    this._keep_alive = new Map<number, unknown>();
  }

  check_already_processed_decorator(is_method = false): (func: (obj: unknown) => unknown) => (obj: unknown) => unknown {
    const self = this;
    return function layer(func: (obj: unknown) => unknown): (obj: unknown) => unknown {
      if (is_method) {
        return function (obj: unknown): unknown {
          return self._handle_already_processed(obj, (value: unknown) =>
            (func as unknown as (selfArg: unknown, objArg: unknown) => unknown)(self, value),
          );
        };
      }
      return function (obj: unknown): unknown {
        return self._handle_already_processed(obj, func);
      };
    };
  }

  check_already_processed(obj: unknown): unknown {
    return this._handle_already_processed(obj, (value: unknown) => value);
  }

  mark_as_processed(obj: unknown): void {
    this._handle_already_processed(obj, (value: unknown) => value);
    this._return(obj, (value: unknown) => identityOf(value as object), (value: unknown) => value, false);
  }

  _handle_already_processed(obj: unknown, default_function: (value: unknown) => unknown): unknown {
    if (
      (typeof obj === "number" || typeof obj === "string" || typeof obj === "boolean") &&
      obj !== (_Memoizer.ALREADY_PROCESSED_PLACEHOLDER as unknown)
    ) {
      return obj;
    }
    try {
      return this._return(obj, hashOf, default_function);
    } catch {
      // Fall back to identity-based membership when hashing fails.
    }
    return this._return(obj, (value: unknown) => identityOf(value as object), default_function);
  }

  _return(
    obj: unknown,
    obj_to_membership_sign: (value: unknown) => number,
    default_func: (value: unknown) => unknown,
    memoizing = true,
  ): unknown {
    const sign_kind = obj_to_membership_sign === hashOf ? "h" : "i";
    const signature = obj_to_membership_sign(obj);
    const membership = `${sign_kind}:${String(signature)}`;
    if (this._already_processed.has(membership)) {
      return _Memoizer.ALREADY_PROCESSED_PLACEHOLDER;
    }
    if (memoizing) {
      this._already_processed.add(membership);
      if (obj !== null && (typeof obj === "object" || typeof obj === "function")) {
        this._keep_alive.set(identityOf(obj as object), obj);
      }
    }
    return default_func(obj);
  }
}

export class _CustomEncoder {
  _memoizer: _Memoizer;
  include_pixel_array: boolean;

  constructor(memoizer: _Memoizer, include_pixel_array = false) {
    this._memoizer = memoizer;
    this.include_pixel_array = include_pixel_array;
  }

  default(obj: unknown): unknown {
    if (typeof obj === "function") {
      return this._cleaned_iterable({
        code: Function.prototype.toString.call(obj),
        nonlocals: {},
      } as Record<string, unknown>);
    }
    if (obj !== null && typeof obj === "object" && "shape" in obj && "toArray" in obj) {
      const ndarray = obj as { toArray: () => unknown };
      const raw = ndarray.toArray();
      const size = Array.isArray(raw) ? raw.length : 1;
      if (size > 1000) {
        return `TRUNCATED ARRAY: ${pythonRepr(raw)}`;
      }
      return pythonRepr(raw);
    }
    if (obj !== null && (typeof obj === "object" || typeof obj === "function")) {
      const record = obj as Record<string, unknown>;
      const cleaned: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(record)) {
        if (KEYS_TO_FILTER_OUT.has(key) && !(this.include_pixel_array && key === "pixel_array")) {
          continue;
        }
        cleaned[key] = value;
      }
      return this._cleaned_iterable(cleaned);
    }
    if (typeof obj === "number" && Number.isInteger(obj) && obj >= 0 && obj <= 255) {
      return Math.trunc(obj);
    }
    return String((obj as object)?.constructor?.name ?? typeof obj);
  }

  _cleaned_iterable(iterable: unknown[] | Record<string, unknown>): unknown[] | Record<string, unknown> {
    const cleanList = (lst: unknown[]): unknown[] => {
      const processed: unknown[] = new Array(lst.length);
      for (let index = 0; index < lst.length; index += 1) {
        let el = this._memoizer.check_already_processed(lst[index]);
        if (Array.isArray(el)) {
          el = cleanList(el);
        } else if (el !== null && typeof el === "object" && !Array.isArray(el)) {
          el = cleanDict(el as Record<string, unknown>);
        }
        processed[index] = el;
      }
      return processed;
    };
    const cleanDict = (dct: Record<string, unknown>): Record<string, unknown> => {
      const processed: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(dct)) {
        const next = this._memoizer.check_already_processed(value);
        if (KEYS_TO_FILTER_OUT.has(key) && !(this.include_pixel_array && key === "pixel_array")) {
          continue;
        }
        const nextKey: string = key;
        if (typeof next === "object" && next !== null && !Array.isArray(next)) {
          processed[nextKey] = cleanDict(next as Record<string, unknown>);
        } else if (Array.isArray(next)) {
          processed[nextKey] = cleanList(next);
        } else {
          processed[nextKey] = next;
        }
      }
      return processed;
    };
    if (Array.isArray(iterable)) {
      return cleanList(iterable);
    }
    if (iterable !== null && typeof iterable === "object") {
      return cleanDict(iterable as Record<string, unknown>);
    }
    throw new TypeError("'iterable' is neither an iterable nor a dictionary.");
  }

  encode(obj: unknown): string {
    this._memoizer.mark_as_processed(obj);
    if (Array.isArray(obj) || (obj !== null && typeof obj === "object" && obj.constructor === Object)) {
      return JSON.stringify(this._cleaned_iterable(obj as Record<string, unknown>), (_key, value: unknown) => {
        if (typeof value === "bigint") return Number(value);
        if (typeof value === "function") return this.default(value);
        if (value !== null && typeof value === "object" && !(value instanceof Array) && (value as object).constructor !== Object) {
          return this.default(value);
        }
        return value;
      });
    }
    return JSON.stringify(obj, (_key, value: unknown) => {
      if (typeof value === "function") return this.default(value);
      return value;
    });
  }
}

export interface DtypeDescriptor {
  str?: string;
  names?: readonly string[] | null;
  fields?: Record<string, readonly unknown[]>;
  subdtype?: readonly [unknown, readonly number[]] | null;
  itemsize?: number;
  alignment?: number;
  isalignedstruct?: boolean;
  newbyteorder?: (order: string) => DtypeDescriptor;
}

export function _canonical_dtype_descriptor(dtype: string | DtypeDescriptor): unknown {
  if (typeof dtype === "string") {
    return dtype;
  }
  const canonical: DtypeDescriptor =
    typeof dtype.newbyteorder === "function" ? dtype.newbyteorder("<") : dtype;
  if (canonical.names !== null && canonical.names !== undefined) {
    return [
      "struct",
      (canonical.names ?? []).map((name) => {
        const field = canonical.fields?.[name] ?? [];
        return [
          name,
          _canonical_dtype_descriptor(field[0] as string | DtypeDescriptor),
          field[1],
          field.length > 2 ? _canonical_field_title(field[2]) : null,
        ];
      }),
      canonical.itemsize,
      canonical.alignment,
      canonical.isalignedstruct,
    ];
  }
  if (canonical.subdtype) {
    const [base, shape] = canonical.subdtype;
    return ["subarray", _canonical_dtype_descriptor(base as string | DtypeDescriptor), [...shape]];
  }
  return canonical.str;
}

export function _get_json(
  obj: unknown,
  memoizer: _Memoizer,
  include_pixel_array = false,
): string {
  return new _CustomEncoder(memoizer, include_pixel_array).encode(obj);
}

export function get_json(obj: unknown, include_pixel_array = false): string {
  return _get_json(obj, new _Memoizer(), include_pixel_array);
}
