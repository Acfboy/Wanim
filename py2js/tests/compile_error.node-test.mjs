// Plain-node test for py2js/src/compile_error.ts.
//
// Run with:  pnpm test:py2js
// (builds py2js/dist/{preprocess,compile_error}.js with tsc, then runs the
// node tests)
import assert from "node:assert/strict";

import { formatCompileError } from "../dist/compile_error.js";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

check("renders the offending line with a caret under the reported column", () => {
  const source = "mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nself.add(mat)\n";
  const error = formatCompileError(
    {
      message: "Unexpected end of file: missing closing ')' for this call",
      line: 1,
      col: 11,
    },
    source,
  );
  assert.equal(
    error.message,
    "Unexpected end of file: missing closing ')' for this call\n" +
      "  scene.py:1:12\n\n" +
      "mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n" +
      "           ^",
  );
});

check("keeps a caret on the line when the column points past the end", () => {
  const error = formatCompileError({ message: "boom", line: 2, col: 40 }, "x = 1\ny = 2\n");
  assert.equal(error.message, "boom\n  scene.py:2:41\n\ny = 2\n     ^");
});

check("uses the compiler filename when present", () => {
  const error = formatCompileError(
    { message: "boom", line: 1, col: 0, filename: "my_scene.py" },
    "x = 1\n",
  );
  assert.ok(error.message.includes("my_scene.py:1:1"));
});

check("passes native errors through unchanged", () => {
  const original = new Error("plain failure");
  assert.equal(formatCompileError(original, "x = 1\n"), original);
});

check("formats non-object values", () => {
  assert.equal(formatCompileError("oops", "").message, "oops");
});

console.log(`\ncompile_error: ${passed} passed`);
