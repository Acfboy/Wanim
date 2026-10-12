/**
 * Public name tables for the Python standard-library modules the runtime
 * provides (`math`, `random`, `itertools`).
 *
 * `preprocess.ts` needs these lists to rewrite `from math import *` (and the
 * `random` / `itertools` variants) into explicit rebinds, but it must not
 * depend on the runtime modules themselves: the compiler bundle is meant to
 * stay free of the runtime code (and of the wanim/numpy-ts imports only the
 * runtime needs).  Keeping the tables in a dependency-free module lets both
 * sides share the single source of truth.
 */

/** Names exported by `from math import *` (functions + constants). */
export const MATH_PUBLIC = [
  "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt",
  "ceil", "comb", "copysign", "cos", "cosh", "degrees", "dist", "erf",
  "erfc", "exp", "exp2", "expm1", "fabs", "factorial", "floor", "fmod",
  "frexp", "fsum", "gamma", "gcd", "hypot", "isclose", "isfinite",
  "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma", "log", "log10",
  "log1p", "log2", "modf", "nextafter", "perm", "pow", "prod", "radians",
  "remainder", "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "ulp",
  "pi", "e", "tau", "inf", "nan",
];

/** Names exported by `from random import *` (mirrors `random.__all__`). */
export const RANDOM_PUBLIC = [
  "Random", "SystemRandom",
  "betavariate", "binomialvariate", "choice", "choices", "expovariate",
  "gammavariate", "gauss", "getrandbits", "getstate", "lognormvariate",
  "normalvariate", "paretovariate", "randbytes", "randint", "random",
  "randrange", "sample", "seed", "setstate", "shuffle", "triangular",
  "uniform", "vonmisesvariate", "weibullvariate",
];

/** Names exported by `from itertools import *`. */
export const ITERTOOLS_PUBLIC = [
  "accumulate", "batched", "chain", "combinations",
  "combinations_with_replacement", "compress", "count", "cycle",
  "dropwhile", "filterfalse", "groupby", "islice", "pairwise",
  "permutations", "product", "repeat", "starmap", "takewhile",
  "tee", "zip_longest",
];
