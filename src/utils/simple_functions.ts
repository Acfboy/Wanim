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

const chooseCache = new Map<string, number>();

export function choose(n: number, k: number): number {
  const key = `${n},${k}`;
  const cached = chooseCache.get(key);
  if (cached !== undefined) return cached;
  const value = comb(n, k);
  if (chooseCache.size >= 10) {
    const oldest = chooseCache.keys().next();
    if (!oldest.done) chooseCache.delete(oldest.value);
  }
  chooseCache.set(key, value);
  return value;
}

function comb(n: number, k: number): number {
  if (!Number.isInteger(n) || !Number.isInteger(k)) {
    // NOTE: mirrors math.comb, which raises TypeError for non-integers
    // ("'float' object cannot be interpreted as an integer").
    const error = new Error("n and k must be integers");
    error.name = "TypeError";
    throw error;
  }
  if (k < 0 || n < 0) {
    const error = new Error(`n must be a non-negative integer and k must satisfy 0 <= k <= n, got n=${n}, k=${k}`);
    error.name = "ValueError";
    throw error;
  }
  // NOTE: mirrors math.comb, which returns 0 (rather than raising) when k > n.
  if (k > n) return 0;
  // NOTE: math.comb is exact (bigint); use BigInt so large coefficients
  // round-trip to the same double python reports, e.g. choose(100, 50).
  const small = Math.min(k, n - k);
  let result = 1n;
  const bigN = BigInt(n);
  for (let i = 1; i <= small; i += 1) {
    result = (result * (bigN - BigInt(small) + BigInt(i))) / BigInt(i);
  }
  return Number(result);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function binary_search(
  function_: (h: number) => number,
  target: number,
  lower_bound: number,
  upper_bound: number,
  tolerance = 1e-4,
): number | null {
  let lh = lower_bound;
  let rh = upper_bound;
  let mh: number = (lh + rh) / 2;
  while (Math.abs(rh - lh) > tolerance) {
    mh = (lh + rh) / 2;
    const lx = function_(lh);
    const mx = function_(mh);
    const rx = function_(rh);
    if (lx === target) return lh;
    if (rx === target) return rh;
    if (lx <= target && target <= rx) {
      if (mx > target) rh = mh;
      else lh = mh;
    } else if (lx > target && target > rx) {
      const tmp = lh;
      lh = rh;
      rh = tmp;
    } else {
      return null;
    }
  }
  return mh;
}
