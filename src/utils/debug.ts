import type { Mobject } from "../mobject/mobject.js";
import { BLACK, type ParsableManimColor } from "./color/core.js";
import type { VGroup } from "../mobject/types/vectorized_mobject.js";

// NOTE: Python prints id(mobject) (the CPython memory address). JS has no
// equivalent, so a stable per-object numeric id is assigned instead.
const _ids = new WeakMap<object, number>();
let _next_id = 1;

function _idOf(mobject: Mobject): number {
  const existing = _ids.get(mobject);
  if (existing !== undefined) return existing;
  const assigned = _next_id;
  _next_id += 1;
  _ids.set(mobject, assigned);
  return assigned;
}

export function print_family(mobject: Mobject, n_tabs = 0): void {
  console.log("\t".repeat(n_tabs), mobject.toString(), _idOf(mobject));
  for (const submob of mobject.submobjects) print_family(submob, n_tabs + 1);
}

export function index_labels(
  mobject: Mobject,
  label_height = 0.15,
  background_stroke_width = 5,
  background_stroke_color: ParsableManimColor = BLACK,
  kwargs: Record<string, unknown> = {},
): VGroup {
  void mobject;
  void label_height;
  void background_stroke_width;
  void background_stroke_color;
  void kwargs;
  // NOTE: mirrors Python `index_labels`, which builds one Integer label per
  // submobject (styled, resized to label_height, moved onto the submobject).
  // Integer labels need the text pipeline that is out of scope for this port.
  throw new Error("TODO: text/tex not ported yet (index_labels needs Integer)");
}
