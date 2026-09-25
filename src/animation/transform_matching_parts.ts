import { AnimationGroup } from "./composition.js";
import type { Mobject } from "../mobject/mobject.js";
import { pointRows } from "../utils/arrays.js";

/** Deterministic 32-bit FNV-1a hash (Python `hash()` is salted per process, so exact values never match; only match/no-match semantics are preserved). */
function fnv1a32(data: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < data.length; index += 1) {
    hash ^= data.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash | 0;
}

export class TransformMatchingAbstractBase extends AnimationGroup {
  // NOTE: full construction (`__init__`/`get_shape_map`/`clean_up_from_scene`)
  // depends on FadeIn/FadeOut (manim.animation.fading, not ported yet) and is
  // left for a later batch. The key-mapping statics below are standalone.
  static get_mobject_parts(_mobject: Mobject): Mobject[] {
    const error = new Error("To be implemented in subclass.");
    error.name = "NotImplementedError";
    throw error;
  }

  static get_mobject_key(_mobject: Mobject): number | string {
    const error = new Error("To be implemented in subclass.");
    error.name = "NotImplementedError";
    throw error;
  }
}

export class TransformMatchingShapes extends TransformMatchingAbstractBase {
  static override get_mobject_parts(mobject: Mobject): Mobject[] {
    return mobject.family_members_with_points();
  }

  static override get_mobject_key(mobject: Mobject): number {
    mobject.save_state();
    mobject.center();
    mobject.set({ height: 1 });
    // NOTE: Python uses `np.round(points, 3)` (banker's rounding) hashed via
    // `hash(bytes)` (salted per process); here half-up rounding plus a
    // deterministic FNV-1a hash preserves match/no-match semantics.
    const rounded = pointRows(mobject.points, mobject.dim).map((row) =>
      row.map((value) => Math.round(value * 1000) / 1000 + 0.0),
    );
    const result = fnv1a32(JSON.stringify(rounded));
    mobject.restore();
    return result;
  }
}
