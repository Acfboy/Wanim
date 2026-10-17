// Plain-node test for py2js/src/preprocess.ts.
//
// Run with:  pnpm test:py2js
// (builds py2js/dist/preprocess.js with tsc, then runs this file)
// or directly after a build:  node py2js/tests/preprocess.node-test.mjs
import assert from "node:assert/strict";

import {
  formatUnsupportedImportError,
  preprocessImports,
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

console.log(`\n${passed} tests passed`);
