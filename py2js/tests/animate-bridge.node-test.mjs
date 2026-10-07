// Regression test: kwargs bags through the `.animate` builder must keep
// their tag all the way to the target method.
//
// `circles.animate.set_fill(opacity=0.25)` compiles to
// `update_target({opacity: 0.25(tagged)})`, which re-applies the bag
// positionally so the wrapped-method kwargs bridge maps `opacity` onto the
// `set_fill` slot. If an intermediate builder method is itself wrapped, the
// bridge mistakes the bag for kwargs-to-map, detags it via an extras merge,
// and the plain object lands in `color` -> `ManimColor ... not Object`.
//
// Run with:  pnpm test:py2js   (needs `pnpm build` first for dist/index.js)
import assert from "node:assert/strict";

import * as npReal from "numpy-ts";

import * as wanim from "../../dist/index.js";
import { createRuntimeCore } from "../dist/runtime_core.js";

const core = createRuntimeCore({
  wanim,
  np: npReal,
  NDArray: npReal.NDArray,
});
const ns = core.wrapWanimNamespace(wanim);
const { Circle, VGroup } = ns;

// Exact rapydscript runtime semantics (baselib-internal.pyj).
const kwargsSymbol = Symbol("kwargs-object");
function desugar(entries) {
  const ans = Object.create(null);
  ans[kwargsSymbol] = true;
  Object.assign(ans, entries);
  return ans;
}
function hasTag(bag) {
  if (Object.getPrototypeOf(bag) !== null) return false;
  return Object.getOwnPropertySymbols(bag).some((s) => String(s).includes("kwargs-object"));
}

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

check("animate.set_fill(opacity=...) updates target without throwing", () => {
  const c1 = new Circle({ radius: 2 });
  const c2 = new Circle({ radius: 2 });
  const circles = new VGroup(c1, c2);
  const setFill = circles.animate.set_fill;
  setFill(desugar({ opacity: 0.25 }));
  // `.animate` records onto the target copy (Python semantics); the live
  // mobject only changes when the animation plays.
  const target = circles.target;
  assert.equal(target.submobjects[0].fill_opacity, 0.25);
  assert.equal(target.submobjects[1].fill_opacity, 0.25);
});

check("animate.set_fill(color=..., opacity=...) maps both slots", () => {
  const c1 = new Circle({ radius: 2 });
  const circles = new VGroup(c1);
  circles.animate.set_fill(desugar({ color: "#ff0000", opacity: 0.5 }));
  const t = circles.target.submobjects[0];
  assert.equal(t.fill_opacity, 0.5);
  const rows = t.fill_rgbas.toArray();
  assert.ok(Math.abs(rows[0][0] - 1) < 1e-9);
  assert.ok(Math.abs(rows[0][3] - 0.5) < 1e-9);
});

check("MethodWithArgs record keeps the kwargs tag for replay", () => {
  const c1 = new Circle({ radius: 2 });
  const circles = new VGroup(c1);
  const builder = circles.animate;
  builder.set_fill(desugar({ opacity: 0.25 }));
  const recorded = builder.methods[builder.methods.length - 1];
  assert.ok(hasTag(recorded.kwargs));
  assert.equal(recorded.kwargs.opacity, 0.25);
});

check("Slider receives its keyword options through the bridge", () => {
  // `Slider` extends `PathSlider`; it used to read PathSlider's inherited
  // __argnames__ and build with no options (default range [0, 1], value 0).
  const slider = ns.Slider(
    desugar({ start: [-2, 0, 0], end: [2, 0, 0], value_range: [0.5, 3], init_value: 1 }),
  );
  assert.deepEqual(slider.value_range, [0.5, 3]);
  assert.equal(slider.get_value(), 1);
  assert.deepEqual(slider.start_point, [-2, 0, 0]);
});

console.log(`\n${passed} animate-bridge checks passed`);
