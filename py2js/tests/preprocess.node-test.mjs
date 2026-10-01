// Plain-node test for py2js/src/preprocess.ts.
//
// Run with:  pnpm test:py2js
// (builds py2js/dist/preprocess.js with tsc, then runs this file)
// or directly after a build:  node py2js/tests/preprocess.node-test.mjs
import assert from "node:assert/strict";

import {
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

check("from math import sin, cos reports module math", () => {
  const r = preprocessImports("from math import sin, cos\n");
  assert.equal(r.unsupported.length, 1);
  assert.equal(r.unsupported[0].module, "math");
  assert.equal(r.unsupported[0].line, 1);
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

console.log(`\n${passed} tests passed`);
