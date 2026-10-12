/**
 * Python `random` module for the wanim browser/JS runtime.
 *
 * Manim scenes (including original 3b1b sources) routinely use the Python
 * standard-library `random` module (`import random`, `random.seed(...)`,
 * `random.random()`, ...), which cannot exist in the browser as-is.  This
 * module ports CPython's `random` (backed by MT19937) so that
 * `preprocess.ts` can accept `import random` / `from random import ...`
 * and compiled scenes observe the same deterministic sequences as CPython.
 *
 * Fidelity notes (verified against CPython 3.12):
 *   - The MT19937 core (`init_genrand` / `init_by_array` / tempering /
 *     53-bit `random()`) is the verbatim algorithm from
 *     `Modules/_randommodule.c`, so seeded sequences match CPython exactly.
 *   - Integer seeds use `abs(n)` split into little-endian 32-bit words;
 *     every other seed type goes through `hash()` cast to unsigned 64-bit
 *     (exactly like `random_seed()` in `_randommodule.c`).
 *   - `str`/`bytes` seeds with `version=2` (the default) hash with SHA-512
 *     (`int.from_bytes(data + sha512(data))`); a small pure-BigInt SHA-512
 *     is vendored below so this matches CPython without async WebCrypto.
 *   - All real-valued / sequence / integer distributions are mechanical
 *     ports of `Lib/random.py` and therefore consume the core generator in
 *     the same order as CPython.
 *
 * Known deviations from CPython (all inherent to the JS runtime):
 *   - `getrandbits(k)` for `k > 53` returns a JS number, so bits past the
 *     53-bit mantissa lose precision (rapydscript represents every int as a
 *     double; big ints are unrepresentable anyway).
 *   - `seed(float('nan'))` is non-deterministic in CPython itself (the hash
 *     of NaN varies per run), so this port seeds deterministically with 0.
 *   - `seed(None)` draws from `crypto.getRandomValues` (or time/`Math.random`
 *     as a fallback), matching CPython's "seed from OS entropy" behaviour
 *     without promising identical output.
 */

type Any = any;

// ---------------------------------------------------------------------------
// SHA-512 (pure BigInt; only used to seed `str`/`bytes` like CPython does).
// ---------------------------------------------------------------------------

const SHA512_MASK = (1n << 64n) - 1n;

const SHA512_K = [
  0x428a2f98d728ae22n, 0x7137449123ef65cdn, 0xb5c0fbcfec4d3b2fn, 0xe9b5dba58189dbbcn,
  0x3956c25bf348b538n, 0x59f111f1b605d019n, 0x923f82a4af194f9bn, 0xab1c5ed5da6d8118n,
  0xd807aa98a3030242n, 0x12835b0145706fben, 0x243185be4ee4b28cn, 0x550c7dc3d5ffb4e2n,
  0x72be5d74f27b896fn, 0x80deb1fe3b1696b1n, 0x9bdc06a725c71235n, 0xc19bf174cf692694n,
  0xe49b69c19ef14ad2n, 0xefbe4786384f25e3n, 0x0fc19dc68b8cd5b5n, 0x240ca1cc77ac9c65n,
  0x2de92c6f592b0275n, 0x4a7484aa6ea6e483n, 0x5cb0a9dcbd41fbd4n, 0x76f988da831153b5n,
  0x983e5152ee66dfabn, 0xa831c66d2db43210n, 0xb00327c898fb213fn, 0xbf597fc7beef0ee4n,
  0xc6e00bf33da88fc2n, 0xd5a79147930aa725n, 0x06ca6351e003826fn, 0x142929670a0e6e70n,
  0x27b70a8546d22ffcn, 0x2e1b21385c26c926n, 0x4d2c6dfc5ac42aedn, 0x53380d139d95b3dfn,
  0x650a73548baf63den, 0x766a0abb3c77b2a8n, 0x81c2c92e47edaee6n, 0x92722c851482353bn,
  0xa2bfe8a14cf10364n, 0xa81a664bbc423001n, 0xc24b8b70d0f89791n, 0xc76c51a30654be30n,
  0xd192e819d6ef5218n, 0xd69906245565a910n, 0xf40e35855771202an, 0x106aa07032bbd1b8n,
  0x19a4c116b8d2d0c8n, 0x1e376c085141ab53n, 0x2748774cdf8eeb99n, 0x34b0bcb5e19b48a8n,
  0x391c0cb3c5c95a63n, 0x4ed8aa4ae3418acbn, 0x5b9cca4f7763e373n, 0x682e6ff3d6b2b8a3n,
  0x748f82ee5defb2fcn, 0x78a5636f43172f60n, 0x84c87814a1f0ab72n, 0x8cc702081a6439ecn,
  0x90befffa23631e28n, 0xa4506cebde82bde9n, 0xbef9a3f7b2c67915n, 0xc67178f2e372532bn,
  0xca273eceea26619cn, 0xd186b8c721c0c207n, 0xeada7dd6cde0eb1en, 0xf57d4f7fee6ed178n,
  0x06f067aa72176fban, 0x0a637dc5a2c898a6n, 0x113f9804bef90daen, 0x1b710b35131c471bn,
  0x28db77f523047d84n, 0x32caab7b40c72493n, 0x3c9ebe0a15c9bebcn, 0x431d67c49c100d4cn,
  0x4cc5d4becb3e42b6n, 0x597f299cfc657e2an, 0x5fcb6fab3ad6faecn, 0x6c44198c4a475817n,
];

function sha512Rotr(x: bigint, n: bigint): bigint {
  return ((x >> n) | (x << (64n - n))) & SHA512_MASK;
}

/** SHA-512 digest of `data` (exactly the FIPS 180-4 padding/rounds). */
function sha512(data: Uint8Array): Uint8Array {
  const bitLen = BigInt(data.length) * 8n;
  // 0x80 byte, zero pad, then the 128-bit big-endian length; total a
  // multiple of 128 bytes. The high 64 length bits are always zero here
  // (inputs are far shorter than 2**64 bits).
  const withOne = data.length + 1;
  const zeros = (112 - (withOne % 128) + 128) % 128;
  const total = withOne + zeros + 16;
  const msg = new Uint8Array(total);
  msg.set(data);
  msg[data.length] = 0x80;
  const view = new DataView(msg.buffer, msg.byteOffset, msg.byteLength);
  view.setBigUint64(total - 8, bitLen, false);

  let h0 = 0x6a09e667f3bcc908n;
  let h1 = 0xbb67ae8584caa73bn;
  let h2 = 0x3c6ef372fe94f82bn;
  let h3 = 0xa54ff53a5f1d36f1n;
  let h4 = 0x510e527fade682d1n;
  let h5 = 0x9b05688c2b3e6c1fn;
  let h6 = 0x1f83d9abfb41bd6bn;
  let h7 = 0x5be0cd19137e2179n;

  const w = new Array<bigint>(80);
  for (let off = 0; off < total; off += 128) {
    for (let i = 0; i < 16; i += 1) w[i] = view.getBigUint64(off + i * 8, false);
    for (let i = 16; i < 80; i += 1) {
      const a = w[i - 2]!;
      const b = w[i - 15]!;
      const s1 = sha512Rotr(a, 19n) ^ sha512Rotr(a, 61n) ^ (a >> 6n);
      const s0 = sha512Rotr(b, 1n) ^ sha512Rotr(b, 8n) ^ (b >> 7n);
      w[i] = (s1 + w[i - 7]! + s0 + w[i - 16]!) & SHA512_MASK;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    for (let i = 0; i < 80; i += 1) {
      const s1 = sha512Rotr(e, 14n) ^ sha512Rotr(e, 18n) ^ sha512Rotr(e, 41n);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + s1 + ch + SHA512_K[i]! + w[i]!) & SHA512_MASK;
      const s0 = sha512Rotr(a, 28n) ^ sha512Rotr(a, 34n) ^ sha512Rotr(a, 39n);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + maj) & SHA512_MASK;
      h = g; g = f; f = e; e = (d + t1) & SHA512_MASK;
      d = c; c = b; b = a; a = (t1 + t2) & SHA512_MASK;
    }
    h0 = (h0 + a) & SHA512_MASK;
    h1 = (h1 + b) & SHA512_MASK;
    h2 = (h2 + c) & SHA512_MASK;
    h3 = (h3 + d) & SHA512_MASK;
    h4 = (h4 + e) & SHA512_MASK;
    h5 = (h5 + f) & SHA512_MASK;
    h6 = (h6 + g) & SHA512_MASK;
    h7 = (h7 + h) & SHA512_MASK;
  }

  const out = new Uint8Array(64);
  const outView = new DataView(out.buffer);
  const hs = [h0, h1, h2, h3, h4, h5, h6, h7];
  for (let i = 0; i < 8; i += 1) outView.setBigUint64(i * 8, hs[i]!, false);
  return out;
}

// ---------------------------------------------------------------------------
// Exact integer / float hashing (CPython compatible).
// ---------------------------------------------------------------------------

/** `P = 2**61 - 1`, the prime modulus of CPython's numeric hashes. */
const HASH_MOD = (1n << 61n) - 1n;

/** Positive modulo (BigInt `%` keeps the dividend's sign). */
function modPos(a: bigint, m: bigint): bigint {
  const r = a % m;
  return r < 0n ? r + m : r;
}

function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  let b = modPos(base, mod);
  let e = exp;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % mod;
    b = (b * b) % mod;
    e >>= 1n;
  }
  return result;
}

/**
 * Hash of an integer value (CPython `long_hash` for values that fit the
 * fast paths, extended to arbitrary BigInts via the same 30-bit-digit
 * rotation so huge seeds hash identically).
 */
function hashBigInt(n: bigint): bigint {
  if (n === 0n) return 0n;
  const negative = n < 0n;
  let mag = negative ? -n : n;
  // Fast paths mirroring `long_hash`'s switch on digit count.
  if (mag < 1n << 30n) {
    const v = Number(mag);
    return negative ? (v === 1 ? -2n : BigInt(-v)) : BigInt(v);
  }
  const digits: number[] = [];
  while (mag > 0n) {
    digits.push(Number(mag & 0x3fffffffn));
    mag >>= 30n;
  }
  let x = 0n;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    x = (((x << 30n) & HASH_MOD) | (x >> 31n)) + BigInt(digits[i]!);
    if (x >= HASH_MOD) x -= HASH_MOD;
  }
  let signed = negative ? -x : x;
  if (signed === -1n) signed = -2n;
  return signed;
}

/**
 * Hash of a float (CPython `_Py_HashDouble`): exact rational hash of the
 * binary value, sign-symmetric, with `-1` mapped to `-2`.  NaN hashes
 * non-deterministically in CPython itself, so this port returns 0.
 */
function hashFloat(x: number): bigint {
  if (Number.isNaN(x)) return 0n;
  if (!Number.isFinite(x)) return x > 0 ? 314159n : -314159n;
  if (Number.isInteger(x) && Math.abs(x) < 2 ** 63) {
    const n = BigInt(x);
    return n === -1n ? -2n : n;
  }
  // Exact decomposition: x = mantissa * 2**exp2 (mantissa odd or zero).
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, x, false);
  const bytes = new Uint8Array(buf);
  let bits = 0n;
  for (const byte of bytes) bits = (bits << 8n) | BigInt(byte);
  const sign = bits >> 63n === 1n;
  const expBits = Number((bits >> 52n) & 0x7ffn);
  let mantissa: bigint;
  let exp2: number;
  if (expBits === 0) {
    mantissa = bits & ((1n << 52n) - 1n);
    exp2 = -1074;
  } else {
    mantissa = ((1n << 52n) | (bits & ((1n << 52n) - 1n)));
    exp2 = expBits - 1075;
  }
  if (mantissa === 0n) return 0n;
  let num = mantissa;
  let den = 1n;
  if (exp2 >= 0) {
    num = mantissa << BigInt(exp2);
  } else {
    den = 1n << BigInt(-exp2);
    while ((num & 1n) === 0n && den > 1n) {
      num >>= 1n;
      den >>= 1n;
    }
  }
  // h = num * den**-1 (mod P) on the magnitude, then re-apply the sign.
  const hPos = (modPos(num, HASH_MOD) * modPow(den, HASH_MOD - 2n, HASH_MOD)) % HASH_MOD;
  let h = sign ? -hPos : hPos;
  if (h === -1n) h = -2n;
  return h;
}

// ---------------------------------------------------------------------------
// Small Python-semantics helpers.
// ---------------------------------------------------------------------------

/** Coerce a rapydscript int-like (`operator.index`) to a JS number. */
function asIndex(value: Any, name: string): number {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") {
    if (!Number.isInteger(value)) {
      throw new TypeError(`'float' object cannot be interpreted as an integer (${name})`);
    }
    return value;
  }
  if (typeof value === "bigint") return Number(value);
  if (value !== null && value !== undefined && typeof value.__index__ === "function") {
    return asIndex(value.__index__(), name);
  }
  throw new TypeError(`'${typeof value}' object cannot be interpreted as an integer (${name})`);
}

/**
 * Detect a trailing rapydscript kwargs bag (mirrors `detectKwargs` in
 * `runtime_core.ts`, which cannot be imported here without a module cycle).
 */
function takeKwargsBag(args: Any[]): Record<string, Any> | null {
  if (args.length === 0) return null;
  const last = args[args.length - 1];
  if (last === null || typeof last !== "object") return null;
  if (Object.getPrototypeOf(last) !== null) return null;
  let tagged = false;
  for (const symbol of Object.getOwnPropertySymbols(last)) {
    if (String(symbol).includes("kwargs-object")) {
      tagged = true;
      break;
    }
  }
  if (!tagged) {
    for (const key of Object.keys(last)) {
      if (key.startsWith("kwargs-object-Symbol-")) {
        tagged = true;
        break;
      }
    }
  }
  if (!tagged) return null;
  const plain: Record<string, Any> = {};
  for (const key of Object.keys(last)) plain[key] = last[key];
  return plain;
}

/** Map a kwargs bag onto positional slots (unknown keys raise TypeError). */
function applyKwargs(fn: string, names: string[], positional: Any[], kwargs: Record<string, Any>): Any[] {
  const out = positional.slice();
  for (const [key, value] of Object.entries(kwargs)) {
    if (key.startsWith("kwargs-object")) continue;
    const index = names.indexOf(key);
    if (index < 0) {
      throw new TypeError(`${fn}() got an unexpected keyword argument '${key}'`);
    }
    out[index] = value;
  }
  return out;
}

/** Split `(…args, maybeKwargsBag)` into positionals plus mapped names. */
function splitCall(fn: string, names: string[], args: Any[]): Any[] {
  const bag = takeKwargsBag(args);
  const positional = bag !== null ? args.slice(0, -1) : args.slice();
  return bag !== null ? applyKwargs(fn, names, positional, bag) : positional;
}

function required(value: Any, fn: string, name: string): Any {
  if (value === undefined) {
    throw new TypeError(`${fn}() missing required argument '${name}'`);
  }
  return value;
}

/** Right-bisect (`bisect.bisect`), ported for `choices`/`sample(counts=…`). */
function bisectRight(a: readonly Any[], x: number, lo = 0, hi?: number): number {
  let high = hi === undefined ? a.length : hi;
  let low = lo;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (x < (a[mid] as number)) high = mid;
    else low = mid + 1;
  }
  return low;
}

/** Lanczos `lgamma` (only needed by `binomialvariate`'s BTRS setup). */
function logGamma(x: number): number {
  // Reflection for the left half-plane.
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const z = x - 1;
  let a = c[0]!;
  for (let i = 1; i < 9; i += 1) a += c[i]! / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

// ---------------------------------------------------------------------------
// `random.Random` — MT19937 core plus every CPython distribution method.
// ---------------------------------------------------------------------------

const MT_N = 624;
const MT_M = 397;
const MT_MATRIX_A = 0x9908b0df;
const MT_UPPER_MASK = 0x80000000;
const MT_LOWER_MASK = 0x7fffffff;

const NV_MAGICCONST = 4 * Math.exp(-0.5) / Math.sqrt(2);
const LOG4 = Math.log(4);
const SG_MAGICCONST = 1 + Math.log(4.5);
const TWOPI = Math.PI * 2;

export class PythonRandom {
  private mt = new Uint32Array(MT_N);
  private index = MT_N + 1;
  private gaussNext: number | null = null;

  constructor(seed?: Any) {
    this.seed(seed);
  }

  // -- core seeding (mirrors `random_seed` in `_randommodule.c`) ----------

  private initGenrand(s: number): void {
    this.mt[0] = s >>> 0;
    for (let i = 1; i < MT_N; i += 1) {
      const prev = this.mt[i - 1]!;
      this.mt[i] = (Math.imul(1812433253, prev ^ (prev >>> 30)) + i) >>> 0;
    }
    this.index = MT_N;
  }

  private initByArray(key: readonly number[]): void {
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(MT_N, key.length);
    for (; k; k -= 1) {
      const prev = this.mt[i]!;
      const mixed = this.mt[i - 1]! ^ (this.mt[i - 1]! >>> 30);
      this.mt[i] = (prev ^ Math.imul(mixed, 1664525)) >>> 0;
      this.mt[i] = (this.mt[i]! + (key[j]! >>> 0) + (j >>> 0)) >>> 0;
      i += 1;
      j += 1;
      if (i >= MT_N) {
        this.mt[0] = this.mt[MT_N - 1]!;
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k2 = MT_N - 1; k2; k2 -= 1) {
      const mixed = this.mt[i - 1]! ^ (this.mt[i - 1]! >>> 30);
      this.mt[i] = (this.mt[i]! ^ Math.imul(mixed, 1566083941)) >>> 0;
      this.mt[i] = (this.mt[i]! - i) >>> 0;
      i += 1;
      if (i >= MT_N) {
        this.mt[0] = this.mt[MT_N - 1]!;
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  /** Split a non-negative BigInt into little-endian 32-bit words. */
  private static wordsOf(n: bigint): number[] {
    if (n === 0n) return [0];
    const out: number[] = [];
    let rest = n;
    while (rest > 0n) {
      out.push(Number(rest & 0xffffffffn));
      rest >>= 32n;
    }
    return out;
  }

  private seedNone(): void {
    // CPython seeds from OS entropy, falling back to time+pid.  Neither is
    // observable here; `crypto.getRandomValues` (else time+Math.random)
    // provides the same "unpredictable" behaviour.
    const key: number[] = [];
    try {
      const cryptoObj = (globalThis as Any).crypto;
      if (cryptoObj !== undefined && typeof cryptoObj.getRandomValues === "function") {
        const bytes = cryptoObj.getRandomValues(new Uint8Array(MT_N * 4));
        for (let i = 0; i < MT_N; i += 1) {
          key.push(
            ((bytes[i * 4]! | (bytes[i * 4 + 1]! << 8) |
              (bytes[i * 4 + 2]! << 16) | (bytes[i * 4 + 3]! << 24)) >>> 0),
          );
        }
      }
    } catch {
      /* fall through to the time-based fallback */
    }
    if (key.length === 0) {
      let t = BigInt(Date.now()) * 1000000n + BigInt(Math.floor(Math.random() * 1000000));
      if (t === 0n) t = 0x9e3779b97f4a7c15n;
      for (let i = 0; i < MT_N; i += 1) {
        t = (t * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
        key.push(Number((t >> 11n) & 0xffffffffn));
      }
    }
    this.initByArray(key);
    this.gaussNext = null;
  }

  private seedInt(n: bigint): void {
    this.initByArray(PythonRandom.wordsOf(n < 0n ? -n : n));
    this.gaussNext = null;
  }

  private seedHash(h: bigint): void {
    // `(size_t)hash`: wrap the signed hash into unsigned 64-bit, then split
    // into little-endian words (empty means zero).
    const unsigned = ((h % (1n << 64n)) + (1n << 64n)) % (1n << 64n);
    this.initByArray(PythonRandom.wordsOf(unsigned));
    this.gaussNext = null;
  }

  /**
   * `seed(a=None, version=2)`.  Accepts a trailing rapydscript kwargs bag.
   * `None`/missing seeds from entropy; ints (and bools) seed directly;
   * non-integral floats seed from their exact CPython hash; `str`/`bytes`
   * follow the `version` dispatch of `Random.seed`.
   */
  seed(...args: Any[]): void {
    const [a, version] = splitCall("seed", ["a", "version"], args);
    const v = version === undefined ? 2 : version;
    if (a === undefined || a === null) {
      this.seedNone();
      return;
    }
    if (typeof a === "boolean") {
      this.seedInt(a ? 1n : 0n);
      return;
    }
    if (typeof a === "number") {
      // JS numbers cannot tell an int literal (`seed(7)`) from a float one
      // (`seed(7.0)`), and real scenes only ever write ints.  Integer-valued
      // numbers therefore take the int path (`abs`, matching CPython's int
      // seeds exactly, including negatives and huge values); genuinely
      // fractional values take the `hash()` path.  The two agree for every
      // non-negative integral value; they only differ for negative or huge
      // integral *float* literals (`seed(-7.0)`), which never occur in
      // practice.
      if (Number.isInteger(a)) {
        this.seedInt(BigInt(a));
      } else {
        this.seedHash(hashFloat(a));
      }
      return;
    }
    if (typeof a === "bigint") {
      this.seedInt(a);
      return;
    }
    if (typeof a === "string") {
      this.seedStr(new TextEncoder().encode(a), v);
      return;
    }
    if (a instanceof Uint8Array) {
      this.seedStr(a, v);
      return;
    }
    if (Array.isArray(a)) {
      this.seedStr(Uint8Array.from(a.map((elt) => Number(elt) & 0xff)), v);
      return;
    }
    throw new TypeError(
      "The only supported seed types are: None,\nint, float, str, bytes, and bytearray.",
    );
  }

  private seedStr(data: Uint8Array, version: Any): void {
    if (version === 1) {
      // Legacy (pre-2.3) string seeding, kept for reproducibility.
      let x = data.length > 0 ? BigInt(data[0]!) << 7n : 0n;
      for (const c of data) x = ((1000003n * x) ^ BigInt(c)) & 0xffffffffffffffffn;
      x ^= BigInt(data.length);
      if (x === BigInt(-1)) x = BigInt(-2);
      this.seedInt(x);
      return;
    }
    if (version === 2) {
      const digest = sha512(data);
      const combined = new Uint8Array(data.length + digest.length);
      combined.set(data);
      combined.set(digest, data.length);
      let n = 0n;
      for (const byte of combined) n = (n << 8n) | BigInt(byte);
      this.seedInt(n);
      return;
    }
    // Any other `version` seeds the raw bytes as an int (CPython ignores
    // `version` for non-str/bytes and for unknown versions here).
    let n = 0n;
    for (const byte of data) n = (n << 8n) | BigInt(byte);
    this.seedInt(n);
  }

  // -- core generation -----------------------------------------------------

  private genrandUint32(): number {
    if (this.index >= MT_N) {
      for (let kk = 0; kk < MT_N - MT_M; kk += 1) {
        const y = (this.mt[kk]! & MT_UPPER_MASK) | (this.mt[kk + 1]! & MT_LOWER_MASK);
        this.mt[kk] = this.mt[kk + MT_M]! ^ (y >>> 1) ^ (y & 1 ? MT_MATRIX_A : 0);
      }
      for (let kk = MT_N - MT_M; kk < MT_N - 1; kk += 1) {
        const y = (this.mt[kk]! & MT_UPPER_MASK) | (this.mt[kk + 1]! & MT_LOWER_MASK);
        this.mt[kk] = this.mt[kk + (MT_M - MT_N)]! ^ (y >>> 1) ^ (y & 1 ? MT_MATRIX_A : 0);
      }
      const y = (this.mt[MT_N - 1]! & MT_UPPER_MASK) | (this.mt[0]! & MT_LOWER_MASK);
      this.mt[MT_N - 1] = this.mt[MT_M - 1]! ^ (y >>> 1) ^ (y & 1 ? MT_MATRIX_A : 0);
      this.index = 0;
    }
    let y = this.mt[this.index]!;
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `random()`: next float in `[0, 1)` with 53-bit resolution. */
  random(): number {
    const a = this.genrandUint32() >>> 5;
    const b = this.genrandUint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** `getrandbits(k)`: int with `k` random bits (exact up to 53 bits). */
  getrandbits(...args: Any[]): number {
    const [kRaw] = splitCall("getrandbits", ["k"], args);
    const k = asIndex(required(kRaw, "getrandbits", "k"), "k");
    if (k < 0) throw new Error("number of bits must be non-negative");
    if (k === 0) return 0;
    if (k <= 32) return this.genrandUint32() >>> (32 - k);
    const words = Math.floor((k - 1) / 32) + 1;
    let result = 0n;
    let remaining = k;
    for (let i = 0; i < words; i += 1) {
      let r = this.genrandUint32();
      if (remaining < 32) r >>>= 32 - remaining;
      result |= BigInt(r >>> 0) << BigInt(32 * i);
      remaining -= 32;
    }
    // Rapydscript ints are doubles; exact only within 53 bits (see header).
    return Number(result);
  }

  /** `getstate()`: restorable state `[version, [...624 words, index], gauss]`. */
  getstate(): Any[] {
    return [3, [...this.mt, this.index], this.gaussNext];
  }

  /** `setstate(state)`: restore `getstate()` output (v2 or v3). */
  setstate(...args: Any[]): void {
    const [state] = splitCall("setstate", ["state"], args);
    if (!Array.isArray(state)) throw new TypeError("state vector must be a tuple");
    const version = state[0];
    let internal: Any;
    if (version === 3) {
      internal = state[1];
      this.gaussNext = (state[2] as number | null) ?? null;
    } else if (version === 2) {
      internal = state[1];
      this.gaussNext = (state[2] as number | null) ?? null;
    } else {
      throw new Error(`state with version ${String(version)} passed to Random.setstate() of version 3`);
    }
    if (!Array.isArray(internal) || internal.length !== MT_N + 1) {
      throw new Error("state vector is the wrong size");
    }
    for (let i = 0; i < MT_N; i += 1) {
      const elt = Number(internal[i]);
      if (!Number.isFinite(elt)) throw new TypeError("state vector must be a tuple");
      // v2 states stored signed ints; convert to unsigned 32-bit.
      this.mt[i] = version === 2
        ? (Math.trunc(elt) % 4294967296 + 4294967296) % 4294967296
        : elt >>> 0;
    }
    const index = Math.trunc(Number(internal[MT_N]));
    if (index < 0 || index > MT_N) throw new Error("invalid state");
    this.index = index;
  }

  // -- integer helpers -----------------------------------------------------

  private randBelow(n: number): number {
    if (!Number.isInteger(n) || n <= 0) throw new Error("empty range in randbelow()");
    const k = n.toString(2).length;
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** `randbytes(n)`: `n` random bytes. */
  randbytes(...args: Any[]): Uint8Array {
    const [nRaw] = splitCall("randbytes", ["n"], args);
    const n = asIndex(required(nRaw, "randbytes", "n"), "n");
    if (n < 0) throw new Error("negative byte count");
    const bits = this.getrandbits(n * 8);
    let big = BigInt(Math.trunc(bits));
    // `getrandbits` is exact only to 53 bits; generate long requests in
    // 32-bit chunks so every byte is uniformly random.
    const out = new Uint8Array(n);
    if (n * 8 > 53) {
      for (let i = 0; i < n; i += 1) out[i] = this.genrandUint32() & 0xff;
      return out;
    }
    for (let i = 0; i < n; i += 1) {
      out[i] = Number(big & 0xffn);
      big >>= 8n;
    }
    return out;
  }

  /** `randrange(start, stop=None, step=1)`. */
  randrange(...args: Any[]): number {
    const [startRaw, stopRaw, stepRaw] = splitCall("randrange", ["start", "stop", "step"], args);
    const start = asIndex(required(startRaw, "randrange", "start"), "start");
    if (stopRaw === undefined || stopRaw === null) {
      if (stepRaw !== undefined && stepRaw !== null && asIndex(stepRaw, "step") !== 1) {
        throw new TypeError("Missing a non-None stop argument");
      }
      if (start > 0) return this.randBelow(start);
      throw new Error("empty range for randrange()");
    }
    const stop = asIndex(stopRaw, "stop");
    const width = stop - start;
    const step = stepRaw === undefined || stepRaw === null ? 1 : asIndex(stepRaw, "step");
    if (step === 1) {
      if (width > 0) return start + this.randBelow(width);
      throw new Error(`empty range in randrange(${String(start)}, ${String(stop)})`);
    }
    let n: number;
    if (step > 0) n = Math.floor((width + step - 1) / step);
    else if (step < 0) n = Math.floor((width + step + 1) / step);
    else throw new Error("zero step for randrange()");
    if (n <= 0) throw new Error(`empty range in randrange(${String(start)}, ${String(stop)}, ${String(step)})`);
    return start + step * this.randBelow(n);
  }

  /** `randint(a, b)`: random int in `[a, b]`, end points included. */
  randint(...args: Any[]): number {
    const [aRaw, bRaw] = splitCall("randint", ["a", "b"], args);
    const a = asIndex(required(aRaw, "randint", "a"), "a");
    const b = asIndex(required(bRaw, "randint", "b"), "b");
    return this.randrange(a, b + 1);
  }

  // -- sequence methods (operate on plain JS arrays / strings) -------------

  private static seqLen(seq: Any): number {
    if (seq === null || seq === undefined || typeof seq.length !== "number") {
      throw new TypeError("Population must be a sequence.  For dicts or sets, use sorted(d).");
    }
    return seq.length;
  }

  /** `choice(seq)`: random element of a non-empty sequence. */
  choice(...args: Any[]): Any {
    const [seq] = splitCall("choice", ["seq"], args);
    const n = PythonRandom.seqLen(required(seq, "choice", "seq"));
    if (n === 0) throw new Error("Cannot choose from an empty sequence");
    return seq[this.randBelow(n)];
  }

  /** `shuffle(x)`: shuffle a list in place. */
  shuffle(...args: Any[]): void {
    const [x] = splitCall("shuffle", ["x"], args);
    const seq = required(x, "shuffle", "x");
    const n = PythonRandom.seqLen(seq);
    for (let i = n - 1; i >= 1; i -= 1) {
      const j = this.randBelow(i + 1);
      const tmp = seq[i];
      seq[i] = seq[j];
      seq[j] = tmp;
    }
  }

  /** `sample(population, k, counts=None)`: `k` unique random elements. */
  sample(...args: Any[]): Any[] {
    const [population, kRaw, counts] = splitCall("sample", ["population", "k", "counts"], args);
    const pop = required(population, "sample", "population");
    const k = asIndex(required(kRaw, "sample", "k"), "k");
    const n = PythonRandom.seqLen(pop);
    if (counts !== undefined && counts !== null) {
      const cum: number[] = [];
      let total = 0;
      for (let i = 0; i < n; i += 1) {
        const c = asIndex(counts[i], "counts");
        total += c;
        cum.push(total);
      }
      if (cum.length !== n) throw new Error("The number of counts does not match the population");
      if (total <= 0) throw new Error("Total of counts must be greater than zero");
      const selections = this.sampleIndices(total, k);
      return selections.map((s) => pop[bisectRight(cum, s)]);
    }
    if (!(k >= 0 && k <= n)) throw new Error("Sample larger than population or is negative");
    const result = new Array<Any>(k);
    let setsize = 21;
    if (k > 5) setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    if (n <= setsize) {
      const pool = Array.from({ length: n }, (_, i) => pop[i]);
      for (let i = 0; i < k; i += 1) {
        const j = this.randBelow(n - i);
        result[i] = pool[j];
        pool[j] = pool[n - i - 1];
      }
    } else {
      const selected = new Set<number>();
      for (let i = 0; i < k; i += 1) {
        let j = this.randBelow(n);
        while (selected.has(j)) j = this.randBelow(n);
        selected.add(j);
        result[i] = pop[j];
      }
    }
    return result;
  }

  private sampleIndices(total: number, k: number): number[] {
    if (!(k >= 0 && k <= total)) throw new Error("Sample larger than population or is negative");
    const result = new Array<number>(k);
    let setsize = 21;
    if (k > 5) setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    if (total <= setsize) {
      const pool = Array.from({ length: total }, (_, i) => i);
      for (let i = 0; i < k; i += 1) {
        const j = this.randBelow(total - i);
        result[i] = pool[j]!;
        pool[j] = pool[total - i - 1]!;
      }
    } else {
      const selected = new Set<number>();
      for (let i = 0; i < k; i += 1) {
        let j = this.randBelow(total);
        while (selected.has(j)) j = this.randBelow(total);
        selected.add(j);
        result[i] = j;
      }
    }
    return result;
  }

  /** `choices(population, weights=None, cum_weights=None, k=1)` (with replacement). */
  choices(...args: Any[]): Any[] {
    const [population, weights, cumWeights, kRaw] = splitCall(
      "choices", ["population", "weights", "cum_weights", "k"], args,
    );
    const pop = required(population, "choices", "population");
    const n = PythonRandom.seqLen(pop);
    const k = kRaw === undefined || kRaw === null ? 1 : asIndex(kRaw, "k");
    let cum = cumWeights;
    if (cum === undefined || cum === null) {
      if (weights === undefined || weights === null) {
        const out = new Array<Any>(k);
        const nf = n + 0.0;
        for (let i = 0; i < k; i += 1) out[i] = pop[Math.floor(this.random() * nf)];
        return out;
      }
      if (typeof weights === "number") {
        throw new TypeError(`The number of choices must be a keyword argument: k=${String(weights)}`);
      }
      cum = [];
      let total = 0;
      for (let i = 0; i < weights.length; i += 1) {
        total += Number(weights[i]);
        (cum as number[]).push(total);
      }
    } else if (weights !== undefined && weights !== null) {
      throw new TypeError("Cannot specify both weights and cumulative weights");
    }
    if (cum.length !== n) throw new Error("The number of weights does not match the population");
    const total = Number(cum[cum.length - 1]) + 0.0;
    if (!(total > 0)) throw new Error("Total of weights must be greater than zero");
    if (!Number.isFinite(total)) throw new Error("Total of weights must be finite");
    const out = new Array<Any>(k);
    const hi = n - 1;
    for (let i = 0; i < k; i += 1) out[i] = pop[bisectRight(cum, this.random() * total, 0, hi)];
    return out;
  }

  // -- real-valued distributions -------------------------------------------

  /** `uniform(a, b)`: random float in `[a, b)` (modulo rounding). */
  uniform(...args: Any[]): number {
    const [aRaw, bRaw] = splitCall("uniform", ["a", "b"], args);
    const a = Number(required(aRaw, "uniform", "a"));
    const b = Number(required(bRaw, "uniform", "b"));
    return a + (b - a) * this.random();
  }

  /** `triangular(low=0.0, high=1.0, mode=None)`. */
  triangular(...args: Any[]): number {
    const [lowRaw, highRaw, modeRaw] = splitCall("triangular", ["low", "high", "mode"], args);
    let low = lowRaw === undefined || lowRaw === null ? 0 : Number(lowRaw);
    let high = highRaw === undefined || highRaw === null ? 1 : Number(highRaw);
    const u = this.random();
    let c: number;
    if (modeRaw === undefined || modeRaw === null) {
      c = 0.5;
    } else if (high === low) {
      return low;
    } else {
      c = (Number(modeRaw) - low) / (high - low);
    }
    if (u > c) {
      const u2 = 1 - u;
      const c2 = 1 - c;
      const tmp = low;
      low = high;
      high = tmp;
      return low + (high - low) * Math.sqrt(u2 * c2);
    }
    return low + (high - low) * Math.sqrt(u * c);
  }

  /** `normalvariate(mu=0.0, sigma=1.0)` (Kinderman–Monahan ratio method). */
  normalvariate(...args: Any[]): number {
    const [muRaw, sigmaRaw] = splitCall("normalvariate", ["mu", "sigma"], args);
    const mu = muRaw === undefined || muRaw === null ? 0 : Number(muRaw);
    const sigma = sigmaRaw === undefined || sigmaRaw === null ? 1 : Number(sigmaRaw);
    for (;;) {
      const u1 = this.random();
      const u2 = 1 - this.random();
      const z = NV_MAGICCONST * (u1 - 0.5) / u2;
      const zz = (z * z) / 4;
      if (zz <= -Math.log(u2)) return mu + z * sigma;
    }
  }

  /** `gauss(mu=0.0, sigma=1.0)` (Box–Muller, with the cached second deviate). */
  gauss(...args: Any[]): number {
    const [muRaw, sigmaRaw] = splitCall("gauss", ["mu", "sigma"], args);
    const mu = muRaw === undefined || muRaw === null ? 0 : Number(muRaw);
    const sigma = sigmaRaw === undefined || sigmaRaw === null ? 1 : Number(sigmaRaw);
    let z = this.gaussNext;
    this.gaussNext = null;
    if (z === null) {
      const x2pi = this.random() * TWOPI;
      const g2rad = Math.sqrt(-2 * Math.log(1 - this.random()));
      z = Math.cos(x2pi) * g2rad;
      this.gaussNext = Math.sin(x2pi) * g2rad;
    }
    return mu + z * sigma;
  }

  /** `lognormvariate(mu, sigma)`. */
  lognormvariate(...args: Any[]): number {
    const [muRaw, sigmaRaw] = splitCall("lognormvariate", ["mu", "sigma"], args);
    return Math.exp(this.normalvariate(
      required(muRaw, "lognormvariate", "mu"),
      required(sigmaRaw, "lognormvariate", "sigma"),
    ));
  }

  /** `expovariate(lambd=1.0)`. */
  expovariate(...args: Any[]): number {
    const [lambdRaw] = splitCall("expovariate", ["lambd"], args);
    const lambd = lambdRaw === undefined || lambdRaw === null ? 1 : Number(lambdRaw);
    return -Math.log(1 - this.random()) / lambd;
  }

  /** `vonmisesvariate(mu, kappa)` (Fisher's circular-data algorithm). */
  vonmisesvariate(...args: Any[]): number {
    const [muRaw, kappaRaw] = splitCall("vonmisesvariate", ["mu", "kappa"], args);
    const mu = Number(required(muRaw, "vonmisesvariate", "mu"));
    const kappa = Number(required(kappaRaw, "vonmisesvariate", "kappa"));
    if (kappa <= 1e-6) return TWOPI * this.random();
    const s = 0.5 / kappa;
    const r = s + Math.sqrt(1 + s * s);
    let z: number;
    for (;;) {
      const u1 = this.random();
      z = Math.cos(Math.PI * u1);
      const d = z / (r + z);
      const u2 = this.random();
      if (u2 < 1 - d * d || u2 <= (1 - d) * Math.exp(d)) break;
    }
    const q = 1 / r;
    const f = (q + z) / (1 + q * z);
    const u3 = this.random();
    const theta = u3 > 0.5 ? (mu + Math.acos(f)) % TWOPI : (mu - Math.acos(f)) % TWOPI;
    return theta;
  }

  /** `gammavariate(alpha, beta)` (Cheng for α>1, GS for α<1). */
  gammavariate(...args: Any[]): number {
    const [alphaRaw, betaRaw] = splitCall("gammavariate", ["alpha", "beta"], args);
    const alpha = Number(required(alphaRaw, "gammavariate", "alpha"));
    const beta = Number(required(betaRaw, "gammavariate", "beta"));
    if (!(alpha > 0) || !(beta > 0)) {
      throw new Error("gammavariate: alpha and beta must be > 0.0");
    }
    if (alpha > 1) {
      const ainv = Math.sqrt(2 * alpha - 1);
      const bbb = alpha - LOG4;
      const ccc = alpha + ainv;
      for (;;) {
        const u1 = this.random();
        if (!(1e-7 < u1 && u1 < 0.9999999)) continue;
        const u2 = 1 - this.random();
        const v = Math.log(u1 / (1 - u1)) / ainv;
        const x = alpha * Math.exp(v);
        const z = u1 * u1 * u2;
        const r = bbb + ccc * v - x;
        if (r + SG_MAGICCONST - 4.5 * z >= 0 || r >= Math.log(z)) return x * beta;
      }
    } else if (alpha === 1) {
      return -Math.log(1 - this.random()) * beta;
    }
    // 0 < alpha < 1: ALGORITHM GS of Kennedy & Gentle.
    for (;;) {
      const u = this.random();
      const b = (Math.E + alpha) / Math.E;
      const p = b * u;
      let x: number;
      if (p <= 1) x = p ** (1 / alpha);
      else x = -Math.log((b - p) / alpha);
      const u1 = this.random();
      if (p > 1) {
        if (u1 <= x ** (alpha - 1)) return x * beta;
      } else if (u1 <= Math.exp(-x)) {
        return x * beta;
      }
    }
  }

  /** `betavariate(alpha, beta)` (via two gamma deviates). */
  betavariate(...args: Any[]): number {
    const [alphaRaw, betaRaw] = splitCall("betavariate", ["alpha", "beta"], args);
    const alpha = required(alphaRaw, "betavariate", "alpha");
    const beta = required(betaRaw, "betavariate", "beta");
    const y = this.gammavariate(alpha, 1);
    if (y) return y / (y + (this.gammavariate(beta, 1) as number));
    return 0;
  }

  /** `paretovariate(alpha)`. */
  paretovariate(...args: Any[]): number {
    const [alphaRaw] = splitCall("paretovariate", ["alpha"], args);
    const alpha = Number(required(alphaRaw, "paretovariate", "alpha"));
    return (1 - this.random()) ** (-1 / alpha);
  }

  /** `weibullvariate(alpha, beta)`. */
  weibullvariate(...args: Any[]): number {
    const [alphaRaw, betaRaw] = splitCall("weibullvariate", ["alpha", "beta"], args);
    const alpha = Number(required(alphaRaw, "weibullvariate", "alpha"));
    const beta = Number(required(betaRaw, "weibullvariate", "beta"));
    return alpha * (-Math.log(1 - this.random())) ** (1 / beta);
  }

  /** `binomialvariate(n=1, p=0.5)` (Devroye BG + Hörmann BTRS). */
  binomialvariate(...args: Any[]): number {
    const [nRaw, pRaw] = splitCall("binomialvariate", ["n", "p"], args);
    const n = nRaw === undefined || nRaw === null ? 1 : asIndex(nRaw, "n");
    const p = pRaw === undefined || pRaw === null ? 0.5 : Number(pRaw);
    if (n < 0) throw new Error("n must be non-negative");
    if (p <= 0 || p >= 1) {
      if (p === 0) return 0;
      if (p === 1) return n;
      throw new Error("p must be in the range 0.0 <= p <= 1.0");
    }
    if (n === 1) return this.random() < p ? 1 : 0;
    if (p > 0.5) return n - (this.binomialvariate(n, 1 - p) as number);
    if (n * p < 10) {
      let x = 0;
      let y = 0;
      const c = Math.log2(1 - p);
      if (!c) return x;
      for (;;) {
        y += Math.floor(Math.log2(this.random()) / c) + 1;
        if (y > n) return x;
        x += 1;
      }
    }
    const spq = Math.sqrt(n * p * (1 - p));
    const b = 1.15 + 2.53 * spq;
    const a = -0.0873 + 0.0248 * b + 0.01 * p;
    const c = n * p + 0.5;
    const vr = 0.92 - 4.2 / b;
    let alpha = 0;
    let lpq = 0;
    let m = 0;
    let h = 0;
    let setupComplete = false;
    for (;;) {
      const u = this.random() - 0.5;
      const us = 0.5 - Math.abs(u);
      const k = Math.floor(((2 * a) / us + b) * u + c);
      if (k < 0 || k > n) continue;
      const v = this.random();
      if (us >= 0.07 && v <= vr) return k;
      if (!setupComplete) {
        alpha = (2.83 + 5.1 / b) * spq;
        lpq = Math.log(p / (1 - p));
        m = Math.floor((n + 1) * p);
        h = logGamma(m + 1) + logGamma(n - m + 1);
        setupComplete = true;
      }
      const vv = (v * alpha) / (a / (us * us) + b);
      if (Math.log(vv) <= h - logGamma(k + 1) - logGamma(n - k + 1) + (k - m) * lpq) {
        return k;
      }
    }
  }
}

// `__argnames__` lets rapydscript map keyword arguments onto positional
// slots for instance-method calls too (each method also accepts a trailing
// kwargs bag directly, so both call shapes work).
(PythonRandom.prototype as Any).gammavariate.__argnames__ = ["alpha", "beta"];

/**
 * `random.SystemRandom` — OS-entropy generator.  `seed`/`getstate`/
 * `setstate` raise like CPython's `_notimplemented`.
 */
export class SystemRandom extends PythonRandom {
  constructor(seed?: Any) {
    super(undefined);
    if (seed !== undefined && seed !== null) this.seed(seed);
  }

  override seed(..._args: Any[]): void {
    throw new Error("SystemRandom seed() not implemented");
  }

  override getstate(): Any[] {
    throw new Error("SystemRandom getstate() not implemented");
  }

  override setstate(..._args: Any[]): void {
    throw new Error("SystemRandom setstate() not implemented");
  }

  override random(): number {
    try {
      const cryptoObj = (globalThis as Any).crypto;
      if (cryptoObj !== undefined && typeof cryptoObj.getRandomValues === "function") {
        const bytes = cryptoObj.getRandomValues(new Uint8Array(7));
        let n = 0n;
        for (const byte of bytes) n = (n << 8n) | BigInt(byte);
        return Number(n >> 3n) / 9007199254740992;
      }
    } catch {
      /* fall through */
    }
    return Math.random();
  }

  override getrandbits(...args: Any[]): number {
    const [kRaw] = splitCall("getrandbits", ["k"], args);
    const k = asIndex(required(kRaw, "getrandbits", "k"), "k");
    if (k < 0) throw new Error("number of bits must be non-negative");
    if (k === 0) return 0;
    try {
      const cryptoObj = (globalThis as Any).crypto;
      if (cryptoObj !== undefined && typeof cryptoObj.getRandomValues === "function") {
        const numBytes = Math.floor((k + 7) / 8);
        const bytes = cryptoObj.getRandomValues(new Uint8Array(numBytes));
        let n = 0n;
        for (const byte of bytes) n = (n << 8n) | BigInt(byte);
        n >>= BigInt(numBytes * 8 - k);
        return Number(n);
      }
    } catch {
      /* fall through */
    }
    return super.getrandbits(k);
  }

  override randbytes(...args: Any[]): Uint8Array {
    const [nRaw] = splitCall("randbytes", ["n"], args);
    const n = asIndex(required(nRaw, "randbytes", "n"), "n");
    try {
      const cryptoObj = (globalThis as Any).crypto;
      if (cryptoObj !== undefined && typeof cryptoObj.getRandomValues === "function") {
        return cryptoObj.getRandomValues(new Uint8Array(n));
      }
    } catch {
      /* fall through */
    }
    return super.randbytes(n);
  }
}

// ---------------------------------------------------------------------------
// Module singleton (`random.seed` etc. are bound closures, so
// `from random import seed` keeps working after detachment).
// ---------------------------------------------------------------------------

export interface PythonRandomModule extends Record<string, Any> {
  Random: Any;
  SystemRandom: Any;
}

function annotate(fn: Any, names: string[]): Any {
  fn.__argnames__ = [...names];
  return fn;
}

/** Build the `random` module object exposed as a runtime global. */
export function buildPythonRandom(): PythonRandomModule {
  const inst = new PythonRandom();
  // Class constructors must work as plain calls (rapydscript cannot `new`
  // an ES6 class); each factory keeps `instanceof` working via the prototype.
  const factoryFor = (Ctor: Any, names: string[]): Any => {
    const factory = function (...args: Any[]): PythonRandom {
      const bag = takeKwargsBag(args);
      const ctorArgs = bag !== null ? args.slice(0, -1) : args;
      const positional = bag !== null ? applyKwargs(Ctor.name, names, ctorArgs, bag) : ctorArgs;
      return new Ctor(...positional);
    } as Any;
    factory.prototype = Ctor.prototype;
    factory.__argnames__ = [...names];
    return factory;
  };
  const module: PythonRandomModule = {
    seed: annotate((...args: Any[]) => (inst.seed as Any)(...args), ["a", "version"]),
    random: annotate(() => inst.random(), []),
    getrandbits: annotate((...args: Any[]) => (inst.getrandbits as Any)(...args), ["k"]),
    randbytes: annotate((...args: Any[]) => (inst.randbytes as Any)(...args), ["n"]),
    getstate: annotate(() => inst.getstate(), []),
    setstate: annotate((...args: Any[]) => (inst.setstate as Any)(...args), ["state"]),
    randrange: annotate((...args: Any[]) => (inst.randrange as Any)(...args), ["start", "stop", "step"]),
    randint: annotate((...args: Any[]) => (inst.randint as Any)(...args), ["a", "b"]),
    choice: annotate((...args: Any[]) => (inst.choice as Any)(...args), ["seq"]),
    choices: annotate((...args: Any[]) => (inst.choices as Any)(...args),
      ["population", "weights", "cum_weights", "k"]),
    shuffle: annotate((...args: Any[]) => (inst.shuffle as Any)(...args), ["x"]),
    sample: annotate((...args: Any[]) => (inst.sample as Any)(...args), ["population", "k", "counts"]),
    uniform: annotate((...args: Any[]) => (inst.uniform as Any)(...args), ["a", "b"]),
    triangular: annotate((...args: Any[]) => (inst.triangular as Any)(...args), ["low", "high", "mode"]),
    normalvariate: annotate((...args: Any[]) => (inst.normalvariate as Any)(...args), ["mu", "sigma"]),
    gauss: annotate((...args: Any[]) => (inst.gauss as Any)(...args), ["mu", "sigma"]),
    lognormvariate: annotate((...args: Any[]) => (inst.lognormvariate as Any)(...args), ["mu", "sigma"]),
    expovariate: annotate((...args: Any[]) => (inst.expovariate as Any)(...args), ["lambd"]),
    vonmisesvariate: annotate((...args: Any[]) => (inst.vonmisesvariate as Any)(...args), ["mu", "kappa"]),
    gammavariate: annotate((...args: Any[]) => (inst.gammavariate as Any)(...args), ["alpha", "beta"]),
    betavariate: annotate((...args: Any[]) => (inst.betavariate as Any)(...args), ["alpha", "beta"]),
    paretovariate: annotate((...args: Any[]) => (inst.paretovariate as Any)(...args), ["alpha"]),
    weibullvariate: annotate((...args: Any[]) => (inst.weibullvariate as Any)(...args), ["alpha", "beta"]),
    binomialvariate: annotate((...args: Any[]) => (inst.binomialvariate as Any)(...args), ["n", "p"]),
    Random: factoryFor(PythonRandom, ["x"]),
    SystemRandom: factoryFor(SystemRandom, ["x"]),
  };
  return module;
}
