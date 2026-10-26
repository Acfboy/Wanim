// Regression test: tuple values inside Python dict literals must compile to
// arrays, not comma expressions.
//
// `{"value_range": (0, 15)}` used to compile the value to `0, 15` (a JS comma
// expression), so the dict ended up holding only the last element (`15`) and
// any consumer reading `value_range[0]` saw `undefined`.  The fix wraps an
// AST_Seq dict value in an AST_Array in rapydscript's `object_()` parser (see
// py2js/patches/rapydscript-ng.patch).
//
// Run with:  pnpm test:py2js   (needs `pnpm build:rapydscript` first).
// Skips with a notice when the browser compiler bundle is not built, so the
// rest of the node test suite stays runnable without a rapydscript checkout.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");
const rapydscriptPath = resolve(root, "dist", "rapydscript.js");

if (!existsSync(rapydscriptPath)) {
  console.log("skip - dist/rapydscript.js not built (run pnpm build:rapydscript)");
  process.exit(0);
}

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

// Load the patched browser compiler into a plain sandbox; `bare: false` output
// carries rapydscript's baselib, so it runs standalone.
const sandbox = {
  console: { log() {}, warn() {}, error() {} },
  self: null,
  window: null,
  navigator: { userAgent: "node" },
  setTimeout,
  clearTimeout,
};
sandbox.self = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync(rapydscriptPath, "utf8"), sandbox);

// Minimal stand-ins for the wanim runtime helpers the parser lowers
// Python-only syntax to (the real ones live in py2js/src/runtime_core.ts).
sandbox.__wan_spread = (x) => Array.from(x);
sandbox.__wan_flatten1 = (x) => [].concat(...x);
sandbox.__wan_abs = (x) => Math.abs(x);
sandbox.__wan_sum = (xs) => [...xs].reduce((a, b) => a + b, 0);
sandbox.__wan_reversed = (x) => [...x].reverse();
sandbox.__wan_dict_get = (obj, key, ...rest) => {
  if (Object.prototype.hasOwnProperty.call(obj, key)) return obj[key];
  return rest.length > 0 ? rest[0] : null;
};
sandbox.__wan_matmul = (a, b) => a * b;
sandbox.__wan_complex = (re, im) => ({ re, im });
sandbox.__wan_slice = (start, stop, step) =>
  `${start === null ? "" : start}:${stop === null ? "" : stop}` +
  (step === undefined ? "" : `:${step === null ? "" : step}`);
sandbox.__wan_getitem = (obj, ...specs) => {
  let value = obj;
  for (const spec of specs) {
    if (typeof spec === "number") value = value[spec < 0 ? value.length + spec : spec];
    else if (spec === "...") continue;
    else {
      const [start, stop] = spec.split(":");
      value = value.slice(start === "" ? 0 : Number(start), stop === "" ? undefined : Number(stop));
    }
  }
  return value;
};
sandbox.__wan_setitem = (obj, ...args) => {
  const value = args[args.length - 1];
  const specs = args.slice(0, -1);
  let target = obj;
  for (let i = 0; i < specs.length - 1; i += 1) target = target[specs[i]];
  target[specs[specs.length - 1]] = value;
  return value;
};

async function compilePython(source, extraOptions = {}) {
  return sandbox.RapydScript.compile(source, "test.py", {
    bare: false,
    beautify: true,
    python_extensions: true,
    ...extraOptions,
  });
}

async function runPython(source, extraOptions) {
  const js = await compilePython(source, extraOptions);
  const lines = [];
  const previous = sandbox.console;
  sandbox.console = { log: (...args) => lines.push(args.join(" ")), warn() {}, error() {} };
  try {
    vm.runInContext(js, sandbox);
  } finally {
    sandbox.console = previous;
  }
  return lines;
}

await check("tuple values in dict literals compile to arrays", async () => {
  const lines = await runPython(
    [
      'd = {"v": (-15, 15)}',
      'e = {"v": (0, 15), "w": (1, 2)}',
      'print(len(d["v"]))',
      'print(d["v"][0])',
      'print(d["v"][1])',
      'print(len(e["v"]))',
      'print(e["v"][0])',
      'print(e["w"][1])',
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["2", "-15", "15", "2", "0", "2"]);
});

await check("nested tuple in a dict value stays a single element", async () => {
  const lines = await runPython(
    [
      'n = {"k": ((1, 2), 3)}',
      'print(len(n["k"]))',
      'print(n["k"][0][1])',
      'print(n["k"][1])',
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["2", "2", "3"]);
});

await check("keyword-argument tuples still compile to arrays", async () => {
  const lines = await runPython(
    [
      "def f(value_range=(0, 1)):",
      "    return value_range[1]",
      "print(f(value_range=(-4, 9)))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["9"]);
});

// --- parser-level Python features (python_extensions) -----------------------

await check("tuple trailing commas and one-element tuples", async () => {
  const lines = await runPython(
    [
      "a = (1, 2,)",
      "b = (3,)",
      "c = ((1, 2), 3,)",
      "print(len(a), len(b), b[0], len(c), c[0][1])",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["2 1 3 2 2"]);
});

await check("starred list/tuple displays use __wan_spread", async () => {
  const lines = await runPython(
    [
      "a = [1, 2]",
      "b = [*a, 3]",
      "c = (*a, 4)",
      "d = [0, *a, *a]",
      "print(b, c, d)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["[1, 2, 3] [1, 2, 4] [0, 1, 2, 1, 2]"]);
});

await check("multi-for comprehensions lower through __wan_flatten1", async () => {
  const lines = await runPython(
    [
      "pairs = [i * 10 + j for i in [1, 2] for j in [3, 4]]",
      "flat = [i for i in [1, 2] for j in [3, 4] if j == 4]",
      "three = [a + b + c for a in [1] for b in [2] for c in [3]]",
      "print(pairs)",
      "print(flat)",
      "print(three)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["[13, 14, 23, 24]", "[1, 2]", "[6]"]);
});

await check("multi-line generator expressions parse", async () => {
  const lines = await runPython(
    [
      "total = sum((",
      "    x",
      "    for x in [1, 2, 3]",
      "))",
      "print(total)",
      "print(list(",
      "    y * 2",
      "    for y in [1, 2]",
      "))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["6", "[2, 4]"]);
});

await check("complex literals become __wan_complex objects", async () => {
  const lines = await runPython(
    [
      "z = 1.5j",
      "print(z.re, z.im)",
      "v = 1e-3j",
      "print(v.im)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["0 1.5", "0.001"]);
});

await check("matrix multiplication lowers to __wan_matmul", async () => {
  const lines = await runPython(
    [
      "x = 6 @ 7",
      "x @= 2",
      "print(x)",
      "print(1 + 2 @ 3)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["84", "7"]);
});

await check("dotted and tuple except handlers", async () => {
  const lines = await runPython(
    [
      "class Errors:",
      "    Value = ValueError",
      "try:",
      "    raise ValueError('boom')",
      "except Errors.Value as e:",
      "    print('caught', str(e))",
      "try:",
      "    raise TypeError('bad')",
      "except (ValueError, TypeError) as e:",
      "    print('caught2', str(e))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["caught ValueError: boom", "caught2 TypeError: bad"]);
});

await check("dict.get, abs, sum and reversed use the __wan_* helpers", async () => {
  const lines = await runPython(
    [
      'd = {"a": 1}',
      'print(d.get("a"))',
      'print(d.get("b", 5))',
      "print(abs(-3))",
      "print(sum([1, 2, 3]))",
      "print(list(reversed([1, 2, 3])))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["1", "5", "3", "6", "[3, 2, 1]"]);
});

await check("builtin rewrites respect module-level shadowing", async () => {
  const js = await compilePython("def sum(xs):\n    return 42\nprint(sum([1]))\n");
  assert.ok(!js.includes("__wan_sum"));
  const js2 = await compilePython("values = [1]\nprint(sum(values))\n");
  assert.ok(js2.includes("__wan_sum"));
});

await check("multi-dim subscripts lower to __wan_getitem/__wan_setitem", async () => {
  const js = await compilePython(
    [
      "x = a[:, 1]",
      "y = a[1:2, 0]",
      "z = a[::-1, ...]",
      "w = a[0][:, 1]",
      "a[:, 1] = v",
      "a[:, 1] += v",
      "a[1:2, 0] *= w",
      "",
    ].join("\n"),
  );
  assert.ok(js.includes("__wan_getitem(a, __wan_slice(null, null), 1)"));
  assert.ok(js.includes("__wan_getitem(a, __wan_slice(1, 2), 0)"));
  assert.ok(js.includes('__wan_getitem(a, __wan_slice(null, null, -1), "...")'));
  assert.ok(js.includes("__wan_getitem(a[0], __wan_slice(null, null), 1)"));
  assert.ok(js.includes("__wan_setitem(a, __wan_slice(null, null), 1, v)"));
  assert.ok(
    js.includes(
      "__wan_setitem(a, __wan_slice(null, null), 1, __wan_getitem(a, __wan_slice(null, null), 1) + v)",
    ),
  );
  assert.ok(js.includes("__wan_setitem(a, __wan_slice(1, 2), 0, __wan_getitem(a, __wan_slice(1, 2), 0) * w)"));
});

await check("stepped slice stores go through __wan_setitem", async () => {
  const js = await compilePython("a[::2] = q\na[::2] *= 3\n");
  assert.ok(js.includes("__wan_setitem(a, __wan_slice(null, null, 2), q)"));
  assert.ok(
    js.includes(
      "__wan_setitem(a, __wan_slice(null, null, 2), __wan_getitem(a, __wan_slice(null, null, 2)) * 3)",
    ),
  );
});

await check("reserved-word identifiers are renamed", async () => {
  const lines = await runPython(
    [
      "new = 3",
      "default = 1",
      "default += 1",
      "let = new + default",
      "print(let)",
      "obj = {}",
      "obj.new = 9",
      "print(obj.new)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["5", "9"]);
});

await check("reserved words are accepted as parameter names", async () => {
  const lines = await runPython(
    [
      "def f(new=1, do=2):",
      "    return new + do",
      "print(f(), f(new=3, do=4))",
      "g = lambda m, old, new: old + new",
      "print(g(1, 2, 3))",
      "def h(*new):",
      "    return len(new)",
      "print(h(1, 2, 3))",
      "class new:",
      "    pass",
      "d = new()",
      "print(isinstance(d, new))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["3 7", "5", "3", "True"]);
});

await check("tuple branches of a conditional expression become arrays", async () => {
  const lines = await runPython(
    [
      "c = 1",
      "a, b = (-1, 0) if c < 0 else (0, 1)",
      "print(a, b)",
      "c = -1",
      "x, y = (-1, 0) if c < 0 else (0, 1)",
      "print(x, y)",
      "d = {'k': (1, 2) if c < 0 else (3, 4)}",
      "print(d['k'][0], d['k'][1])",
      "print(((5, 6) if c < 0 else (7, 8))[1])",
      "e = [v for v in ((1, 2) if c < 0 else (3, 4))]",
      "print(e)",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["0 1", "-1 0", "1 2", "6", "[1, 2]"]);
});

await check("f-strings still work", async () => {
  const lines = await runPython(
    ["a = 3", 'b = f"This is {a} and {a * 2}"', "print(b)", ""].join("\n"),
  );
  assert.deepEqual(lines, ["This is 3 and 6"]);
});

// --- python_truthiness ------------------------------------------------------
//
// Python containers are falsy when empty; JavaScript's `[]`/`{}` are truthy.
// The compiler lowers every boolean context through the baselib `ρσ_bool`, so
// `if not crits:` takes the "empty" branch (the bug this option was added
// for), `[] or x` evaluates to `x`, and `[] and x` evaluates to `[]`.

const TRUTHY = { python_truthiness: true };

await check("python_truthiness: bool()/if follow Python container truthiness", async () => {
  const lines = await runPython(
    [
      "def f(x):",
      "    return 1 if x else 0",
      "print(f([]), f({}), f(''), f(0), f(0.0), f(None), f(()))",
      "print(f([0]), f({'a': 1}), f(' '), f(1), f([[]]))",
      "print(bool([]), bool([0]), bool({}), bool({'a': 1}))",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, [
    "0 0 0 0 0 0 0",
    "1 1 1 1 1",
    "False True False True",
  ]);
});

await check("python_truthiness: `not` on containers", async () => {
  const lines = await runPython(
    [
      "print(not [], not [1], not {}, not {'a': 1}, not '', not 'x')",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["True False True False True False"]);
});

await check("python_truthiness: `and`/`or` return operands and short-circuit", async () => {
  const lines = await runPython(
    [
      "def boom():",
      "    print('BOOM')",
      "    return 'b'",
      "print([] and 'x')",
      "print([] or 'x')",
      "print([0] and 'x')",
      "print([0] or 'x')",
      "print(0 or '' or 'y')",
      "print('a' and '' and 'c')",
      "print(0 and boom())",
      "print([1] or boom())",
      "",
    ].join("\n"),
    TRUTHY,
  );
  // `boom()` must never run (both operands short-circuit).
  assert.deepEqual(lines, ["[]", "x", "x", "[0]", "y", "", "0", "[1]"]);
});

await check("python_truthiness: empty-list branch selection (the crits bug)", async () => {
  const lines = await runPython(
    [
      "crits = []",
      "if not crits:",
      "    print('single')",
      "else:",
      "    print('split', crits[0])",
      "crits.append(2)",
      "if not crits:",
      "    print('single')",
      "else:",
      "    print('split', crits[0])",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["single", "split 2"]);
});

await check("python_truthiness: while/comprehension/assert use Python truthiness", async () => {
  const lines = await runPython(
    [
      "items = [1, 2]",
      "times = 0",
      "while items:",
      "    items.pop()",
      "    times += 1",
      "print(times)",
      "flags = [[], [1], []]",
      "print([i for i in range(3) if flags[i]])",
      "assert [1]",
      "assert {'a': 1}",
      "print('asserts-pass')",
      "caught = False",
      "try:",
      "    assert []",
      "except AssertionError:",
      "    caught = True",
      "print(caught)",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["2", "[1]", "asserts-pass", "True"]);
});

await check("python_truthiness: numpy-style arrays", async () => {
  // Stub the numpy-ts NDArray surface (shape/size/dtype/tolist/item) the
  // baselib detects; the real arrays behave the same way.
  sandbox.fake_nd_empty = { shape: [0], size: 0, dtype: "float64", tolist: () => [] };
  sandbox.fake_nd_zero = {
    shape: [1],
    size: 1,
    dtype: "float64",
    item: () => 0,
    tolist: () => [0],
  };
  sandbox.fake_nd_two = {
    shape: [2],
    size: 2,
    dtype: "float64",
    item: () => 1,
    tolist: () => [1, 1],
  };
  const lines = await runPython(
    [
      "print(bool(fake_nd_empty), bool(fake_nd_zero))",
      "ambiguous = False",
      "try:",
      "    print(bool(fake_nd_two))",
      "except ValueError:",
      "    ambiguous = True",
      "print(ambiguous)",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["False False", "True"]);
});

await check("python_truthiness: __bool__ is honoured", async () => {
  sandbox.fake_tracker = { __bool__: () => false };
  const lines = await runPython(
    ["print(bool(fake_tracker), not fake_tracker)", ""].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["False True"]);
});

await check("python_truthiness: off by default keeps upstream JS truthiness", async () => {
  const lines = await runPython(
    ["print(1 if [] else 0)", "print([] or 'x', [] and 'x')", ""].join("\n"),
  );
  assert.deepEqual(lines, ["1", "[] x"]);
});

await check("python_truthiness: tricky operand shapes", async () => {
  const lines = await runPython(
    [
      "print((1, 2) and 3)",
      "print(() or 'empty')",
      "print(not (1 < 2))",
      "print('a' if (1, 2) else 'b')",
      "print(f\"{[] or 'f'}\")",
      "f = lambda: [] or 'lam'",
      "print(f())",
      "d = {'k': [] or 'v'}",
      "print(d['k'])",
      "g = [[1], []]",
      "print([(x or 'none') for x in g])",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ["3", "empty", "False", "a", "f", "lam", "v", '[[1], "none"]']);
});

await check("python_truthiness: and/or inside generators", async () => {
  // NOTE: a bare `yield` *nested inside* an expression is broken upstream
  // (`[] || yield "first";`), so only generator bodies are covered here.
  const lines = await runPython(
    [
      "def gen(flag):",
      "    yield [1] and flag",
      "    yield [] or 'y'",
      "    yield flag and []",
      "print(list(gen('x')))",
      "",
    ].join("\n"),
    TRUTHY,
  );
  assert.deepEqual(lines, ['["x", "y", []]']);
});

await check("python_truthiness: auto-await inside and/or stays valid", async () => {
  // `await` must stay in the same expression: wrapping the right operand in a
  // function would produce `() => await ...` (a syntax error) and break the
  // real-time scene renderer.
  const js = await compilePython(
    [
      "class Scene:",
      "    def play(self, *args, **kwargs):",
      "        return 'played'",
      "    def run(self):",
      "        first = self.play(1) or 'fallback'",
      "        second = [1] and self.play(2)",
      "        return first, second",
      "Scene().run()",
      "",
    ].join("\n"),
    { ...TRUTHY, await_scene_methods: true },
  );
  assert.match(js, /await self\.play\(/);
  assert.doesNotMatch(js, /=>\s*await\s/);
  vm.runInContext(js, sandbox);
});

await check("python truthiness: all()/any() (baselib, option-independent)", async () => {
  const lines = await runPython(
    [
      "print(all([[1], [2]]), all([[1], []]), all([]))",
      "print(any([[], [1]]), any([[], []]), any([]))",
      "print(all([{'a': 1}]), any([{}]), all([{}, 1]))",
      "",
    ].join("\n"),
  );
  assert.deepEqual(lines, ["True False True", "True False False", "True False False"]);
});

await check("await_scene_methods: self.move_camera is auto-awaited", async () => {
  // `move_camera` blocks on an internal play (ThreeDScene), so the
  // real-time renderer must await it like `play`/`wait`; otherwise the
  // camera animation runs detached and interactive 3D scenes never move.
  const js = await compilePython(
    [
      "class Demo:",
      "    def construct(self):",
      "        self.move_camera(phi=75, theta=30)",
      "        self.wait()",
      "",
    ].join("\n"),
    { ...TRUTHY, await_scene_methods: true },
  );
  assert.match(js, /await.*move_camera/);
});

console.log(`\n${passed} passed`);
