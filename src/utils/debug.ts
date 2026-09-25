import type { Mobject } from "../mobject/mobject.js";

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
