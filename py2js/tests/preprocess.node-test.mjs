// Plain-node test for py2js/src/preprocess.ts.
//
// Run with:  pnpm test:py2js
// (builds py2js/dist/preprocess.js with tsc, then runs this file)
// or directly after a build:  node py2js/tests/preprocess.node-test.mjs
import assert from "node:assert/strict";

import {
  desugarDictGet,
  desugarDottedExcept,
  desugarMatmul,
  desugarMultiForComprehensions,
  desugarNumpySubscripts,
  desugarReversedBuiltin,
  desugarStarredDisplays,
  desugarStarredGenerators,
  desugarSumBuiltin,
  desugarTupleTrailingCommas,
  formatUnsupportedImportError,
  preprocessImports,
  renameJsReservedWords,
  unsupportedImportError,
} from "../dist/preprocess.js";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

check("from manim import * is stripped and not flagged", () => {
  const r = preprocessImports("from manim import *\nCircle()\n");
  assert.equal(r.unsupported.length, 0);
  assert.deepEqual(r.stripped, ["from manim import *"]);
  assert.deepEqual(r.numpyNames, []);
  assert.ok(r.code.includes("# [wanim] stripped: from manim import *"));
  assert.ok(r.code.includes("Circle()"));
});

check("import numpy as np is stripped and not flagged", () => {
  const r = preprocessImports("import numpy as np\nx = np.array([1])\n");
  assert.equal(r.unsupported.length, 0);
  assert.deepEqual(r.stripped, ["import numpy as np"]);
  assert.ok(r.code.includes("# [wanim] stripped: import numpy as np"));
  assert.ok(r.code.includes("x = np.array([1])"));
});

check("from numpy import array, sin records names and is not flagged", () => {
  const r = preprocessImports("from numpy import array, sin\n");
  assert.equal(r.unsupported.length, 0);
  assert.deepEqual(r.numpyNames, ["array", "sin"]);
  assert.deepEqual(r.stripped, ["from numpy import array, sin"]);
});

check("import os is reported with module and line number", () => {
  const src = "from manim import *\n\nimport os\n";
  const r = preprocessImports(src);
  assert.equal(r.unsupported.length, 1);
  assert.deepEqual(r.unsupported[0], { module: "os", line: 3, text: "import os" });
});

check("import matplotlib.pyplot as plt reports dotted module", () => {
  const r = preprocessImports("import matplotlib.pyplot as plt\n");
  assert.equal(r.unsupported.length, 1);
  assert.equal(r.unsupported[0].module, "matplotlib.pyplot");
  assert.equal(r.unsupported[0].line, 1);
});

check("from math import sin, cos is stripped and rebound", () => {
  const r = preprocessImports("from math import sin, cos\n");
  assert.equal(r.unsupported.length, 0);
  assert.ok(r.code.includes("__wan_math = math"));
  assert.ok(r.code.includes("sin = __wan_math.sin"));
  assert.ok(r.code.includes("cos = __wan_math.cos"));
});

check("import math variants bind the module alias", () => {
  const plain = preprocessImports("import math\nx = math.sqrt(4)\n");
  assert.equal(plain.unsupported.length, 0);
  assert.ok(plain.code.includes("__wan_math = math"));
  assert.ok(plain.code.includes("x = math.sqrt(4)"));
  const aliased = preprocessImports("import math as m\nx = m.pi\n");
  assert.equal(aliased.unsupported.length, 0);
  assert.ok(aliased.code.includes("m = __wan_math"));
  const star = preprocessImports("from math import *\n");
  assert.equal(star.unsupported.length, 0);
  assert.ok(star.code.includes("comb = __wan_math.comb"));
  assert.ok(star.code.includes("pi = __wan_math.pi"));
  const names = preprocessImports("from math import comb as C\nx = C(3, 1)\n");
  assert.equal(names.unsupported.length, 0);
  assert.ok(names.code.includes("C = __wan_math.comb"));
});

check("multi-line from-import is reported on the `from` line", () => {
  const src = "from manim import *\nfrom collections import (\n    OrderedDict,\n)\n";
  const r = preprocessImports(src);
  assert.equal(r.unsupported.length, 1);
  assert.equal(r.unsupported[0].module, "collections");
  assert.equal(r.unsupported[0].line, 2);
});

check("import a, b reports every module", () => {
  const r = preprocessImports("import a, b as bee\n");
  assert.deepEqual(
    r.unsupported.map((u) => u.module),
    ["a", "b"],
  );
  assert.equal(r.unsupported[0].line, 1);
  assert.equal(r.unsupported[0].text, "import a, b as bee");
});

check("import random is stripped and not flagged", () => {
  const r = preprocessImports("from manim import *\nimport random\nrandom.seed(7)\n");
  assert.equal(r.unsupported.length, 0);
  assert.deepEqual(r.stripped, ["from manim import *", "import random"]);
  assert.ok(r.code.includes("# [wanim] stripped: import random"));
  assert.ok(r.code.includes("random.seed(7)"));
});

check("import random as rnd binds the alias", () => {
  const r = preprocessImports("import random as rnd\nrnd.seed(7)\n");
  assert.equal(r.unsupported.length, 0);
  assert.ok(r.code.includes("__wan_random = random"));
  assert.ok(r.code.includes("rnd = __wan_random"));
  assert.ok(r.code.includes("rnd.seed(7)"));
});

check("from random import names rebind from the module global", () => {
  const r = preprocessImports("from random import seed, random\nseed(7)\n");
  assert.equal(r.unsupported.length, 0);
  assert.ok(r.code.includes("seed = __wan_random.seed"));
  assert.ok(r.code.includes("random = __wan_random.random"));
});

check("mixed random imports stay bound to the module, not the shadow", () => {
  // `from random import random` rebinds `random` to the function (as in
  // CPython); the alias import must still see the module.
  const r = preprocessImports("from random import random\nimport random as rnd\nrnd.seed(7)\n");
  assert.equal(r.unsupported.length, 0);
  const aliasDef = r.code.indexOf("__wan_random = random");
  const shadow = r.code.indexOf("random = __wan_random.random");
  const use = r.code.indexOf("rnd = __wan_random");
  assert.ok(aliasDef >= 0 && shadow > aliasDef && use > aliasDef);
});

check("from random import with alias and multiline parens", () => {
  const r = preprocessImports("from random import randint as ri\nx = ri(1, 5)\n");
  assert.equal(r.unsupported.length, 0);
  assert.ok(r.code.includes("ri = __wan_random.randint"));
  const multi = preprocessImports("from random import (\n    seed,\n    shuffle,\n)\n");
  assert.equal(multi.unsupported.length, 0);
  assert.ok(multi.code.includes("seed = __wan_random.seed"));
  assert.ok(multi.code.includes("shuffle = __wan_random.shuffle"));
});

check("from random import * binds every public name", () => {
  const r = preprocessImports("from random import *\nseed(7)\n");
  assert.equal(r.unsupported.length, 0);
  assert.ok(r.code.includes("seed = __wan_random.seed"));
  assert.ok(r.code.includes("Random = __wan_random.Random"));
});

check("other stdlib imports are still rejected", () => {
  const r = preprocessImports("import random\nimport os\n");
  assert.equal(r.unsupported.length, 1);
  assert.equal(r.unsupported[0].module, "os");
});

check("comments and strings that look like imports are not flagged", () => {
  const src = '# import os\nx = "import os"\ny = frommath\n';
  const r = preprocessImports(src);
  assert.equal(r.unsupported.length, 0);
});

check("unsupportedImportError is null for supported code", () => {
  const r = preprocessImports("from manim import *\nimport numpy as np\n");
  assert.equal(unsupportedImportError(r), null);
});

check("unsupportedImportError contains the module name", () => {
  const r = preprocessImports("import os\n");
  const err = unsupportedImportError(r);
  assert.ok(err instanceof Error);
  assert.ok(err.message.includes("os"));
});

check("formatUnsupportedImportError mentions module, line and text", () => {
  const msg = formatUnsupportedImportError({ module: "os", line: 3, text: "import os" });
  assert.ok(msg.includes("'os'"));
  assert.ok(msg.includes("line 3"));
  assert.ok(msg.includes("import os"));
  assert.ok(msg.includes("from manim import *"));
  assert.ok(msg.includes("import numpy as np"));
});

check("renameJsReservedWords renames `new` consistently", () => {
  const out = renameJsReservedWords("new = Text('x')\nnew.scale(2)\n");
  assert.equal(out, "new_modified = Text('x')\nnew_modified.scale(2)\n");
});

check("renameJsReservedWords renames other JS keywords", () => {
  const out = renameJsReservedWords("default = 1\nlet = this\n");
  assert.equal(out, "default_modified = 1\nlet_modified = this_modified\n");
});

check("renameJsReservedWords leaves attribute names alone", () => {
  const out = renameJsReservedWords("obj.new\nself.default\nfoo.new()\n");
  assert.equal(out, "obj.new\nself.default\nfoo.new()\n");
});

check("renameJsReservedWords leaves strings and comments alone", () => {
  const out = renameJsReservedWords('# new\nx = "new"\ny = \'default\'\n');
  assert.equal(out, '# new\nx = "new"\ny = \'default\'\n');
});

check("renameJsReservedWords skips triple-quoted strings", () => {
  const out = renameJsReservedWords('s = """new\ndefault"""\nnew = 1\n');
  assert.equal(out, 's = """new\ndefault"""\nnew_modified = 1\n');
});

check("renameJsReservedWords leaves Python keywords and super alone", () => {
  const out = renameJsReservedWords("for i in xs:\n    super().__init__()\n");
  assert.equal(out, "for i in xs:\n    super().__init__()\n");
});

check("desugar leaves single-for comprehensions alone", () => {
  const src = "x = [i + 1 for i in xs]\ny = [i for i in xs if i]\n";
  assert.equal(desugarMultiForComprehensions(src), src);
});

check("desugar rewrites double-for list comprehension, order preserved", () => {
  const out = desugarMultiForComprehensions(
    "x = [f(i, j) for j in [1, 0, -1] for i in [-1, 0, 1]]\n",
  );
  assert.equal(
    out,
    "x = __wan_flatten1([[f(i, j) for i in [-1, 0, 1]] for j in [1, 0, -1]])\n",
  );
});

check("desugar handles the la1_7 grid case", () => {
  const src =
    'grid = VGroup(*[\n    MathTex("(%d,%d)" % (i, j), font_size=26)\n    for j in [1, 0, -1] for i in [-1, 0, 1]\n])\n';
  const out = desugarMultiForComprehensions(src);
  assert.ok(out.includes("__wan_flatten1([[MathTex("));
  assert.ok(out.includes("for i in [-1, 0, 1]] for j in [1, 0, -1]])"));
  assert.ok(out.startsWith("grid = VGroup(*__wan_flatten1("));
});

check("desugar nests triple-for outside-in", () => {
  const out = desugarMultiForComprehensions("x = [e for a in A for b in B for c in C]\n");
  assert.equal(
    out,
    "x = __wan_flatten1([__wan_flatten1([[e for c in C] for b in B]) for a in A])\n",
  );
});

check("desugar combines multiple ifs with and", () => {
  const out = desugarMultiForComprehensions(
    "x = [e for a in A if p if q for b in B if r]\n",
  );
  assert.equal(
    out,
    "x = __wan_flatten1([[e for b in B if ((r))] for a in A if ((p) and (q))])\n",
  );
});

check("desugar handles set/dict/generator forms", () => {
  assert.equal(
    desugarMultiForComprehensions("x = {e for a in A for b in B}\n"),
    "x = {__wan_y for __wan_y in __wan_flatten1([[e for b in B] for a in A])}\n",
  );
  assert.equal(
    desugarMultiForComprehensions("x = (e for a in A for b in B)\n"),
    "x = (__wan_g for __wan_g in __wan_flatten1([[e for b in B] for a in A]))\n",
  );
  assert.equal(
    desugarMultiForComprehensions("x = {k: v for a in A for b in B}\n"),
    "x = {__wan_kv[0]: __wan_kv[1] for __wan_kv in __wan_flatten1([[[k, v] for b in B] for a in A])}\n",
  );
});

check("desugar ignores for/if inside strings, comments and identifiers", () => {
  const src =
    '# for i in xs\nx = ["for", "if"]\ny = [format(m) for m in ms]\n';
  assert.equal(desugarMultiForComprehensions(src), src);
});

check("desugar leaves indexing, calls and tuples alone", () => {
  const src = "a[0]\na[1:2]\nf(x, y)\n(x, y)\n(x for x in xs)\n";
  assert.equal(desugarMultiForComprehensions(src), src);
});

check("desugar keeps ternary elements and nested multi-for", () => {
  assert.equal(
    desugarMultiForComprehensions("x = [a if c else b for a in A for b in B]\n"),
    "x = __wan_flatten1([[a if c else b for b in B] for a in A])\n",
  );
  assert.equal(
    desugarMultiForComprehensions("x = [[e for a in A for b in B] for c in C]\n"),
    "x = [__wan_flatten1([[e for b in B] for a in A]) for c in C]\n",
  );
});

check("desugar leaves async-for alone", () => {
  const src = "x = [e async for a in A for b in B]\n";
  assert.equal(desugarMultiForComprehensions(src), src);
});

check("starred generators become starred list comprehensions", () => {
  assert.equal(
    desugarStarredGenerators("x = VGroup(*(Circle(c) for c in cols))\n"),
    "x = VGroup(*[Circle(c) for c in cols])\n",
  );
  assert.equal(
    desugarStarredGenerators(
      "students = VGroup(*(\n    Circle(radius=0.3, color=c, fill_opacity=1)\n    for c in [RED, GREEN, YELLOW]\n))\n",
    ),
    "students = VGroup(*[\n    Circle(radius=0.3, color=c, fill_opacity=1)\n    for c in [RED, GREEN, YELLOW]\n])\n",
  );
  assert.equal(
    desugarStarredGenerators("x = f(a, *(e for a in A for b in B))\n"),
    "x = f(a, *[e for a in A for b in B])\n",
  );
});

check("starred generators leave non-generators alone", () => {
  const src =
    "f(*args)\nf(a * (b))\nf(a, *b)\nf(*(a, b))\nx = (e for e in E)\nx = a * (e for e in E)\n";
  assert.equal(desugarStarredGenerators(src), src);
});

check("starred generators ignore stars in strings and comments", () => {
  const src = 'x = "*(e for e in E)"\n# f(*(e for e in E))\ny = f(*args)\n';
  assert.equal(desugarStarredGenerators(src), src);
});

check("subscript desugar rewrites pts[:, 1]", () => {
  assert.equal(
    desugarNumpySubscripts("idx = int(np.argmin(np.abs(pts[:, 1] - b)))\n"),
    "idx = int(np.argmin(np.abs(__wan_getitem(pts, __wan_slice(None, None), 1) - b)))\n",
  );
});

check("subscript desugar handles ranges, steps and ellipsis", () => {
  assert.equal(
    desugarNumpySubscripts("x = a[1:2, 0]\n"),
    "x = __wan_getitem(a, __wan_slice(1, 2), 0)\n",
  );
  assert.equal(
    desugarNumpySubscripts("x = a[::-1, ...]\n"),
    'x = __wan_getitem(a, __wan_slice(None, None, -1), "...")\n',
  );
  assert.equal(
    desugarNumpySubscripts("x = a[0, 1]\n"),
    "x = __wan_getitem(a, 0, 1)\n",
  );
});

check("subscript desugar chains through calls and prior subscripts", () => {
  assert.equal(
    desugarNumpySubscripts("x = a[0][:, 1]\n"),
    "x = __wan_getitem(a[0], __wan_slice(None, None), 1)\n",
  );
  assert.equal(
    desugarNumpySubscripts("x = foo(bar)[1, 2]\n"),
    "x = __wan_getitem(foo(bar), 1, 2)\n",
  );
  assert.equal(
    desugarNumpySubscripts("x = self.grid.values[:, 0]\n"),
    "x = __wan_getitem(self.grid.values, __wan_slice(None, None), 0)\n",
  );
});

check("subscript desugar leaves single subscripts and displays alone", () => {
  const src =
    "a[0]\na[1:2]\na[-1]\n" +
    "x = [1, 2]\nfor i in [1, 2]:\n    pass\n" +
    "return [a, b]\nd = {k: v}\ns = a['k']\n";
  assert.equal(desugarNumpySubscripts(src), src);
});

check("subscript desugar handles nested multi-dim inside dims", () => {
  assert.equal(
    desugarNumpySubscripts("x = a[b[:, 1], 0]\n"),
    "x = __wan_getitem(a, __wan_getitem(b, __wan_slice(None, None), 1), 0)\n",
  );
});

check("subscript desugar keeps keyword-led objects (error 242)", () => {
  assert.equal(
    desugarNumpySubscripts("col = YELLOW if bounded[j, i] else BLUE_D\n"),
    "col = YELLOW if __wan_getitem(bounded, j, i) else BLUE_D\n",
  );
  assert.equal(
    desugarNumpySubscripts("x = a[i] if a[j, k] else b\n"),
    "x = a[i] if __wan_getitem(a, j, k) else b\n",
  );
  assert.equal(
    desugarNumpySubscripts("return a[x, y]\n"),
    "return __wan_getitem(a, x, y)\n",
  );
  // Two bare identifiers in a row are still not a subscript.
  assert.equal(desugarNumpySubscripts("x = foo bar[0, 1]\n"), "x = foo bar[0, 1]\n");
  // Bare keyword-led displays are still not subscripts.
  assert.equal(desugarNumpySubscripts("for i in [1, 2]:\n    pass\n"), "for i in [1, 2]:\n    pass\n");
});

check("subscript desugar leaves strings and comments alone", () => {
  const src = '# a[:, 1]\nx = "a[:, 1]"\n';
  assert.equal(desugarNumpySubscripts(src), src);
});

check("subscript desugar rewrites single-dim slice stores (error 339)", () => {
  // `intensity[::17] *= 0.3` compiled to `ρσ_eslice(intensity, 17) *= 0.3`
  // (Invalid left-hand side in assignment); stepped and augmented
  // single-dim slice stores now go through `__wan_setitem`.
  assert.equal(
    desugarNumpySubscripts("intensity[::17] *= 0.3\n"),
    "__wan_setitem(intensity, __wan_slice(None, None, 17), (__wan_getitem(intensity, __wan_slice(None, None, 17)) * (0.3)))\n",
  );
  assert.equal(
    desugarNumpySubscripts("a[::-1] = v\n"),
    "__wan_setitem(a, __wan_slice(None, None, -1), (v))\n",
  );
  assert.equal(
    desugarNumpySubscripts("a[1:2] += v\n"),
    "__wan_setitem(a, __wan_slice(1, 2), (__wan_getitem(a, __wan_slice(1, 2)) + (v)))\n",
  );
  // Paths rapydscript-ng already handles stay untouched: reads, plain
  // step-less slice stores (via `ρσ_splice`) and plain index stores (via
  // `AST_ItemAccess`).
  assert.equal(desugarNumpySubscripts("x = a[::2]\n"), "x = a[::2]\n");
  assert.equal(desugarNumpySubscripts("a[1:2] = v\n"), "a[1:2] = v\n");
  assert.equal(desugarNumpySubscripts("a[0] *= 2\n"), "a[0] *= 2\n");
});

check("matmul desugar rewrites rot @ center", () => {
  assert.equal(
    desugarMatmul("new_center = rot @ old_center\n"),
    "new_center = __wan_matmul(rot, old_center)\n",
  );
});

check("matmul desugar chains left-associatively", () => {
  assert.equal(
    desugarMatmul("X = Q @ np.diag(u) @ Q.T\n"),
    "X = __wan_matmul(__wan_matmul(Q, np.diag(u)), Q.T)\n",
  );
});

check("matmul desugar respects multiplicative precedence", () => {
  assert.equal(
    desugarMatmul("x = a + b @ c\n"),
    "x = a + __wan_matmul(b, c)\n",
  );
  assert.equal(
    desugarMatmul("x = a * b @ c\n"),
    "x = __wan_matmul(a * b, c)\n",
  );
  assert.equal(
    desugarMatmul("x = a @ b * c\n"),
    "x = __wan_matmul(a, b) * c\n",
  );
  assert.equal(
    desugarMatmul("x = a @ b ** 2\n"),
    "x = __wan_matmul(a, b ** 2)\n",
  );
  assert.equal(
    desugarMatmul("x = a @ -b\n"),
    "x = __wan_matmul(a, -b)\n",
  );
  assert.equal(
    desugarMatmul("x = f(a @ b, c)\n"),
    "x = f(__wan_matmul(a, b), c)\n",
  );
});

check("matmul desugar handles attributes, subscripts and calls", () => {
  assert.equal(
    desugarMatmul("x = self.basis @ self.model[word]\n"),
    "x = __wan_matmul(self.basis, self.model[word])\n",
  );
  assert.equal(
    desugarMatmul("x = A[:, 0] @ b\n"),
    "x = __wan_matmul(A[:, 0], b)\n",
  );
  assert.equal(
    desugarMatmul("x = 2 * np.array([self.basis @ self.model[w] for w in ws])\n"),
    "x = 2 * np.array([__wan_matmul(self.basis, self.model[w]) for w in ws])\n",
  );
});

check("matmul desugar rewrites @=", () => {
  assert.equal(
    desugarMatmul("x @= M\n"),
    "x = __wan_matmul(x, M)\n",
  );
  assert.equal(
    desugarMatmul("self.x @= M @ v\n"),
    "self.x = __wan_matmul(self.x, __wan_matmul(M, v))\n",
  );
});

check("matmul desugar leaves decorators, strings and comments alone", () => {
  const src =
    "@property\ndef f(self):\n    pass\n" +
    "@dec(arg)\ndef g(self):\n    pass\n" +
    'x = "a @ b"\n# y = a @ b\n';
  assert.equal(desugarMatmul(src), src);
});

check("matmul desugar leaves unparseable @ for the compiler", () => {
  assert.equal(desugarMatmul("x = a @ @ b\n"), "x = a @ @ b\n");
  assert.equal(
    desugarMatmul("x = a @\n"),
    "x = a @\n",
  );
});

check("reversed desugar rewrites bare reversed() calls", () => {
  assert.equal(
    desugarReversedBuiltin("for name, part in zip(names, reversed(parts)):\n"),
    "for name, part in zip(names, __wan_reversed(parts)):\n",
  );
  assert.equal(
    desugarReversedBuiltin("x = list(reversed([1, 2, 3]))\n"),
    "x = list(__wan_reversed([1, 2, 3]))\n",
  );
  assert.equal(
    desugarReversedBuiltin("x = reversed (seq)\n"),
    "x = __wan_reversed (seq)\n",
  );
});

check("reversed desugar leaves attributes, strings, comments and shadowing alone", () => {
  assert.equal(
    desugarReversedBuiltin("x = obj.reversed(seq)\n"),
    "x = obj.reversed(seq)\n",
  );
  assert.equal(
    desugarReversedBuiltin('x = "reversed(seq)"\n# y = reversed(seq)\n'),
    'x = "reversed(seq)"\n# y = reversed(seq)\n',
  );
  assert.equal(
    desugarReversedBuiltin("myreversed(x)\n"),
    "myreversed(x)\n",
  );
  assert.equal(
    desugarReversedBuiltin("def reversed(x):\n    return x\n"),
    "def reversed(x):\n    return x\n",
  );
  assert.equal(
    desugarReversedBuiltin("reversed = list\nx = reversed(y)\n"),
    "reversed = list\nx = reversed(y)\n",
  );
  assert.equal(
    desugarReversedBuiltin("x = __wan_reversed(y)\n"),
    "x = __wan_reversed(y)\n",
  );
});

check("sum desugar rewrites bare sum() calls", () => {
  assert.equal(
    desugarSumBuiltin("wave_fn = lambda t: sum(a * np.sin(f * t) for a, f in zip(amps, freqs))\n"),
    "wave_fn = lambda t: __wan_sum(a * np.sin(f * t) for a, f in zip(amps, freqs))\n",
  );
  assert.equal(
    desugarSumBuiltin("x = sum ([1, 2, 3])\n"),
    "x = __wan_sum ([1, 2, 3])\n",
  );
  assert.equal(
    desugarSumBuiltin("x = sum(vals, start)\n"),
    "x = __wan_sum(vals, start)\n",
  );
});

check("sum desugar leaves attributes, numpy imports, strings, comments and shadowing alone", () => {
  assert.equal(
    desugarSumBuiltin("x = np.sum(a, axis=0)\n"),
    "x = np.sum(a, axis=0)\n",
  );
  assert.equal(
    desugarSumBuiltin("x = obj.sum(seq)\n"),
    "x = obj.sum(seq)\n",
  );
  assert.equal(
    desugarSumBuiltin('x = "sum(seq)"\n# y = sum(seq)\n'),
    'x = "sum(seq)"\n# y = sum(seq)\n',
  );
  assert.equal(
    desugarSumBuiltin("mysum(x)\n"),
    "mysum(x)\n",
  );
  assert.equal(
    desugarSumBuiltin("def sum(x):\n    return x\n"),
    "def sum(x):\n    return x\n",
  );
  assert.equal(
    desugarSumBuiltin("sum = list\nx = sum(y)\n"),
    "sum = list\nx = sum(y)\n",
  );
  assert.equal(
    desugarSumBuiltin("x = __wan_sum(y)\n"),
    "x = __wan_sum(y)\n",
  );
  assert.equal(
    desugarSumBuiltin("from numpy import sum\nx = sum(a, axis=0)\n"),
    "from numpy import sum\nx = sum(a, axis=0)\n",
  );
  assert.equal(
    desugarSumBuiltin("from numpy import *\nx = sum(a)\n"),
    "from numpy import *\nx = sum(a)\n",
  );
  // `sum` aliased away from numpy keeps the builtin meaning.
  assert.equal(
    desugarSumBuiltin("from numpy import sum as s\nx = sum(a)\n"),
    "from numpy import sum as s\nx = __wan_sum(a)\n",
  );
});

check("dict.get desugar rewrites .get() calls (error 275)", () => {
  assert.equal(
    desugarDictGet("counts[ch] = counts.get(ch, 0) + 1\n"),
    "counts[ch] = __wan_dict_get(counts, ch, 0) + 1\n",
  );
  assert.equal(
    desugarDictGet("x = counts.get(ch)\n"),
    "x = __wan_dict_get(counts, ch)\n",
  );
  assert.equal(
    desugarDictGet("x = self.d.get(k, 1)\n"),
    "x = __wan_dict_get(self.d, k, 1)\n",
  );
  assert.equal(
    desugarDictGet("x = d[k].get(k2, 0)\n"),
    "x = __wan_dict_get(d[k], k2, 0)\n",
  );
  // Longer attribute names, strings and comments pass through.
  assert.equal(
    desugarDictGet("x = a.get_center()\n"),
    "x = a.get_center()\n",
  );
  assert.equal(
    desugarDictGet('x = "s.get(x)"\n# y = d.get(x)\n'),
    'x = "s.get(x)"\n# y = d.get(x)\n',
  );
  assert.equal(
    desugarDictGet("x = __wan_dict_get(d, k)\n"),
    "x = __wan_dict_get(d, k)\n",
  );
});

check("except desugar hoists dotted handlers before try", () => {
  assert.equal(
    desugarDottedExcept(
      "        try:\n            x = 1\n        except np.linalg.LinAlgError:\n            return False\n",
    ),
    "        __wan_exc_0 = np.linalg.LinAlgError\n" +
      "        try:\n            x = 1\n        except __wan_exc_0:\n            return False\n",
  );
});

check("except desugar flattens tuples and keeps as-names", () => {
  assert.equal(
    desugarDottedExcept(
      "try:\n    pass\nexcept (ValueError, np.linalg.LinAlgError) as e:\n    print(e)\n",
    ),
    "__wan_exc_0 = np.linalg.LinAlgError\n" +
      "try:\n    pass\nexcept ValueError, __wan_exc_0 as e:\n    print(e)\n",
  );
  assert.equal(
    desugarDottedExcept("try:\n    pass\nexcept (ValueError):\n    pass\n"),
    "try:\n    pass\nexcept ValueError:\n    pass\n",
  );
});

check("except desugar binds each handler to the nearest same-indent try", () => {
  assert.equal(
    desugarDottedExcept(
      "try:\n    try:\n        pass\n    except a.B:\n        pass\nexcept C:\n    pass\n",
    ),
    "try:\n    __wan_exc_0 = a.B\n    try:\n        pass\n    except __wan_exc_0:\n        pass\nexcept C:\n    pass\n",
  );
});

check("except desugar leaves supported handlers, strings and comments alone", () => {
  const src =
    "try:\n    pass\nexcept Exception:\n    pass\nexcept ValueError, TypeError as e:\n    pass\nexcept:\n    pass\n";
  assert.equal(desugarDottedExcept(src), src);
  assert.equal(
    desugarDottedExcept('x = "except np.E:"\n# except np.E:\ntry:\n    pass\nexcept ValueError:\n    pass\n'),
    'x = "except np.E:"\n# except np.E:\ntry:\n    pass\nexcept ValueError:\n    pass\n',
  );
});

check("except desugar skips except-star, multiline headers and missing try", () => {
  assert.equal(
    desugarDottedExcept("try:\n    pass\nexcept* ValueError:\n    pass\n"),
    "try:\n    pass\nexcept* ValueError:\n    pass\n",
  );
  assert.equal(
    desugarDottedExcept("try:\n    pass\nexcept (ValueError,\n        TypeError):\n    pass\n"),
    "try:\n    pass\nexcept (ValueError,\n        TypeError):\n    pass\n",
  );
  assert.equal(
    desugarDottedExcept("def f():\n    except np.E:\n        pass\n"),
    "def f():\n    except np.E:\n        pass\n",
  );
});

check("tuple desugar strips multi-element trailing commas", () => {
  assert.equal(desugarTupleTrailingCommas("x = (1, 2,)\n"), "x = (1, 2 )\n");
  assert.equal(
    desugarTupleTrailingCommas("x = (\n    1,\n    2,\n)\n"),
    "x = (\n    1,\n    2 \n)\n",
  );
  assert.equal(
    desugarTupleTrailingCommas("pairs = [\n    (a, b),\n    (c, d,),\n]\n"),
    "pairs = [\n    (a, b),\n    (c, d ),\n]\n",
  );
});

check("tuple desugar rewrites single-element tuples to lists", () => {
  assert.equal(desugarTupleTrailingCommas("x = (1,)\n"), "x = [1 ]\n");
  assert.equal(desugarTupleTrailingCommas("return (x,)\n"), "return [x ]\n");
  assert.equal(
    desugarTupleTrailingCommas("x = ((a, b,),)\n"),
    "x = [(a, b ) ]\n",
  );
});

check("tuple desugar leaves calls, imports and plain tuples alone", () => {
  // Multi-element call trailing commas are dropped (valid either way);
  // single-argument calls already compile, so they pass through.
  assert.equal(desugarTupleTrailingCommas("f(1, 2,)\n"), "f(1, 2 )\n");
  assert.equal(desugarTupleTrailingCommas("f(x,)\n"), "f(x,)\n");
  assert.equal(desugarTupleTrailingCommas("x = (1, 2)\n"), "x = (1, 2)\n");
  assert.equal(desugarTupleTrailingCommas("x = ()\n"), "x = ()\n");
  assert.equal(
    desugarTupleTrailingCommas("from x import (a,)\n"),
    "from x import (a,)\n",
  );
  assert.equal(
    desugarTupleTrailingCommas("x = [1, 2,]\nd = {'a': 1,}\n"),
    "x = [1, 2,]\nd = {'a': 1,}\n",
  );
});

check("tuple desugar keeps subscripts exact and skips strings/comments", () => {
  assert.equal(desugarTupleTrailingCommas("x = a[(1,)]\n"), "x = a[(1 )]\n");
  assert.equal(
    desugarTupleTrailingCommas('x = "(1, 2,)"\n# y = (3, 4,)\n'),
    'x = "(1, 2,)"\n# y = (3, 4,)\n',
  );
  assert.equal(
    desugarTupleTrailingCommas("x = (\n    1,\n    2,  # done\n)\n"),
    "x = (\n    1,\n    2   # done\n)\n",
  );
});

check("except desugar output is stable", () => {
  const src = "try:\n    pass\nexcept np.E:\n    pass\n";
  const once = desugarDottedExcept(src);
  assert.equal(desugarDottedExcept(once), once);
});

check("starred desugar rewrites [*seam, (i, j)]", () => {
  assert.equal(
    desugarStarredDisplays("new_seams.append([*seam, (i + 1, j_prime)])\n"),
    "new_seams.append([].concat(__wan_spread(seam), [(i + 1, j_prime)]))\n",
  );
});

check("starred desugar groups plains and handles tuples", () => {
  assert.equal(
    desugarStarredDisplays("x = [a, *b, c, *d]\n"),
    "x = [].concat([a], __wan_spread(b), [c], __wan_spread(d))\n",
  );
  assert.equal(
    desugarStarredDisplays("x = (*a, b)\n"),
    "x = [].concat(__wan_spread(a), [b])\n",
  );
  assert.equal(
    desugarStarredDisplays("x = [*a]\n"),
    "x = [].concat(__wan_spread(a))\n",
  );
  assert.equal(
    desugarStarredDisplays("x = [*a,]\n"),
    "x = [].concat(__wan_spread(a))\n",
  );
});

check("starred desugar rewrites nested displays to a fixpoint", () => {
  assert.equal(
    desugarStarredDisplays("x = [*ys, [*a, b]]\n"),
    "x = [].concat(__wan_spread(ys), [[].concat(__wan_spread(a), [b])])\n",
  );
});

check("starred desugar leaves calls, subscripts and plain displays alone", () => {
  assert.equal(
    desugarStarredDisplays("g = VGroup(*items)\n"),
    "g = VGroup(*items)\n",
  );
  assert.equal(
    desugarStarredDisplays("x = [a * b, c]\n"),
    "x = [a * b, c]\n",
  );
  assert.equal(
    desugarStarredDisplays("x = (a, b)\n"),
    "x = (a, b)\n",
  );
  assert.equal(
    desugarStarredDisplays("return [*a]\n"),
    "return [].concat(__wan_spread(a))\n",
  );
  assert.equal(
    desugarStarredDisplays('x = "[*a]"\n# y = [*a]\n'),
    'x = "[*a]"\n# y = [*a]\n',
  );
});

check("starred desugar skips comprehensions but rewrites displays inside them", () => {
  assert.equal(
    desugarStarredDisplays("x = [e for e in es]\n"),
    "x = [e for e in es]\n",
  );
  assert.equal(
    desugarStarredDisplays("x = [[*s, 1] for s in ss]\n"),
    "x = [[].concat(__wan_spread(s), [1]) for s in ss]\n",
  );
});

check("starred desugar output is stable", () => {
  const src = "x = [*a, b]\n";
  const once = desugarStarredDisplays(src);
  assert.equal(desugarStarredDisplays(once), once);
});

console.log(`\n${passed} tests passed`);
