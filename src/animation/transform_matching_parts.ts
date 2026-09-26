import { AnimationGroup, type AnimationGroupOptions } from "./composition.js";
import { Group, type Mobject } from "../mobject/mobject.js";
import { VMobject, VGroup } from "../mobject/types/vectorized_mobject.js";
import { Transform, FadeTransformPieces } from "./transform.js";
import { FadeIn, FadeOut } from "./fading.js";
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

export type MobjectKey = number | string;
export type KeyMapInput = Map<MobjectKey, MobjectKey> | Record<string | number, MobjectKey> | null | undefined;

export interface TransformMatchingOptions extends AnimationGroupOptions {
  transform_mismatches?: boolean;
  fade_transform_mismatches?: boolean;
  key_map?: KeyMapInput;
}

function keyMapEntries(key_map: KeyMapInput): Array<[MobjectKey, MobjectKey]> {
  if (key_map === null || key_map === undefined) return [];
  if (key_map instanceof Map) return [...key_map.entries()];
  return Object.entries(key_map).map(([key, value]) => [key, value] as [MobjectKey, MobjectKey]);
}

/** Tolerant map lookup: plain-object key maps stringify numeric keys, so fall back across number/string spellings. */
function findShapeKey(
  shape_map: Map<MobjectKey, VGroup | Group>,
  key: MobjectKey,
): MobjectKey | undefined {
  if (shape_map.has(key)) return key;
  if (typeof key === "string" && key !== "" && !Number.isNaN(Number(key)) && shape_map.has(Number(key))) {
    return Number(key);
  }
  if (typeof key === "number" && shape_map.has(String(key))) {
    return String(key);
  }
  return undefined;
}

function lookupShape(
  shape_map: Map<MobjectKey, VGroup | Group>,
  key: MobjectKey,
): (VGroup | Group) | undefined {
  const found = findShapeKey(shape_map, key);
  return found === undefined ? undefined : shape_map.get(found);
}

function hasShape(shape_map: Map<MobjectKey, VGroup | Group>, key: MobjectKey): boolean {
  return findShapeKey(shape_map, key) !== undefined;
}

export class TransformMatchingAbstractBase extends AnimationGroup {
  to_remove: Mobject[];
  to_add: Mobject;

  constructor(
    mobject: Mobject,
    target_mobject: Mobject,
    options: TransformMatchingOptions = {},
  ) {
    const {
      transform_mismatches = false,
      fade_transform_mismatches = false,
      key_map = null,
      ...kwargs
    } = options;
    // NOTE: Python selects among OpenGLVGroup/OpenGLGroup/VGroup/Group by
    // renderer and mobject type. The OpenGL renderer is not ported, so the
    // Cairo branch applies: VGroup for VMobjects, Group otherwise.
    const group_type = (mobject instanceof VMobject ? VGroup : Group) as unknown as new () => VGroup & Group;

    // NOTE: Python builds the animation list before calling
    // `super().__init__(*anims)`; TS requires super() first, so the parts
    // below run as statics before super() and only field assignment happens
    // afterwards (same final state).
    const ctor = new.target as unknown as typeof TransformMatchingAbstractBase;
    const source_map = TransformMatchingAbstractBase.buildShapeMap(ctor, mobject);
    const target_map = TransformMatchingAbstractBase.buildShapeMap(ctor, target_mobject);

    const sourceKeys = new Set<MobjectKey>(source_map.keys());
    const targetKeys = new Set<MobjectKey>(target_map.keys());

    // Create two mobjects whose submobjects all match each other
    // according to whatever keys are used for source_map and
    // target_map
    const transform_source = new group_type();
    const transform_target = new group_type();
    for (const key of sourceKeys) {
      if (targetKeys.has(key)) {
        (transform_source as Group).add(source_map.get(key) as Mobject);
        (transform_target as Group).add(target_map.get(key) as Mobject);
      }
    }
    const anims: AnimationGroup["animations"] = [
      new Transform(transform_source as unknown as Mobject, transform_target as unknown as Mobject, { ...kwargs }),
    ];
    // User can manually specify when one part should transform
    // into another despite not matching by using key_map
    const key_mapped_source = new group_type();
    const key_mapped_target = new group_type();
    for (const [key1, key2] of keyMapEntries(key_map)) {
      const source_part = lookupShape(source_map, key1);
      const target_part = lookupShape(target_map, key2);
      if (source_part !== undefined && target_part !== undefined) {
        (key_mapped_source as Group).add(source_part as unknown as Mobject);
        (key_mapped_target as Group).add(target_part as unknown as Mobject);
        source_map.delete(findShapeKey(source_map, key1) as MobjectKey);
        target_map.delete(findShapeKey(target_map, key2) as MobjectKey);
      }
    }
    if ((key_mapped_source as unknown as Group).submobjects.length > 0) {
      anims.push(
        new FadeTransformPieces(
          key_mapped_source as unknown as Mobject,
          key_mapped_target as unknown as Mobject,
          { ...kwargs },
        ),
      );
    }

    const fade_source = new group_type();
    const fade_target = new group_type();
    for (const key of [...source_map.keys()]) {
      if (!hasShape(target_map, key)) {
        (fade_source as Group).add(source_map.get(key) as unknown as Mobject);
      }
    }
    for (const key of [...target_map.keys()]) {
      if (!hasShape(source_map, key)) {
        (fade_target as Group).add(target_map.get(key) as unknown as Mobject);
      }
    }
    const fade_target_copy = (fade_target as unknown as Mobject).copy();

    if (transform_mismatches) {
      const mismatchKwargs = { ...kwargs };
      if (!("replace_mobject_with_target_in_scene" in mismatchKwargs)) {
        (mismatchKwargs as Record<string, unknown>)["replace_mobject_with_target_in_scene"] = true;
      }
      anims.push(
        new Transform(
          fade_source as unknown as Mobject,
          fade_target as unknown as Mobject,
          mismatchKwargs,
        ),
      );
    } else if (fade_transform_mismatches) {
      anims.push(
        new FadeTransformPieces(
          fade_source as unknown as Mobject,
          fade_target as unknown as Mobject,
          { ...kwargs },
        ),
      );
    } else {
      anims.push(
        new FadeOut(fade_source as unknown as Mobject, {
          ...(kwargs as Record<string, unknown>),
          target_position: fade_target as unknown as Mobject,
        }),
        new FadeIn(fade_target_copy, {
          ...(kwargs as Record<string, unknown>),
          target_position: fade_target as unknown as Mobject,
        }),
      );
    }

    super(...anims);
    this.to_remove = [mobject, fade_target_copy];
    this.to_add = target_mobject;
  }

  private static buildShapeMap(
    ctor: typeof TransformMatchingAbstractBase,
    mobject: Mobject,
  ): Map<MobjectKey, VGroup | Group> {
    const shape_map = new Map<MobjectKey, VGroup | Group>();
    for (const sm of ctor.get_mobject_parts(mobject)) {
      const key = ctor.get_mobject_key(sm);
      if (!shape_map.has(key)) {
        // NOTE: Python picks OpenGLVGroup under the OpenGL renderer; only
        // the Cairo VGroup branch applies here.
        shape_map.set(key, new VGroup());
      }
      (shape_map.get(key) as VGroup).add(sm as unknown as VMobject);
    }
    return shape_map;
  }

  get_shape_map(mobject: Mobject): Map<MobjectKey, VGroup | Group> {
    const ctor = this.constructor as unknown as typeof TransformMatchingAbstractBase;
    return TransformMatchingAbstractBase.buildShapeMap(ctor, mobject);
  }

  override clean_up_from_scene(scene: any): void {
    // Interpolate all animations back to 0 to ensure source mobjects remain unchanged.
    for (const anim of this.animations) {
      anim.interpolate(0);
    }
    scene?.remove(this.mobject);
    scene?.remove(...this.to_remove);
    scene?.add(this.to_add);
  }

  static get_mobject_parts(_mobject: Mobject): Mobject[] {
    const error = new Error("To be implemented in subclass.");
    error.name = "NotImplementedError";
    throw error;
  }

  static get_mobject_key(_mobject: Mobject): MobjectKey {
    const error = new Error("To be implemented in subclass.");
    error.name = "NotImplementedError";
    throw error;
  }
}

export class TransformMatchingShapes extends TransformMatchingAbstractBase {
  constructor(
    mobject: Mobject,
    target_mobject: Mobject,
    options: TransformMatchingOptions = {},
  ) {
    const {
      transform_mismatches = false,
      fade_transform_mismatches = false,
      key_map = null,
      ...kwargs
    } = options;
    super(mobject, target_mobject, {
      transform_mismatches,
      fade_transform_mismatches,
      key_map,
      ...kwargs,
    });
  }

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

// NOTE: text/tex rendering is out of scope for this port, but
// TransformMatchingTex itself is pure key-matching logic over the
// `tex_string` attribute (no LaTeX binary needed), so it is ported
// faithfully. `MathTexPart` (manim.mobject.text.tex_mobject) is not ported;
// the `isinstance(mobject, MathTexPart)` check is expressed structurally
// (a string `tex_string` attribute), and `Tex`/`MathTex` construction
// elsewhere remains a TODO stub.
export interface TexStringCarrier extends Mobject {
  tex_string: string;
}

function readTexString(mobject: Mobject): unknown {
  try {
    return (mobject as unknown as Record<string, unknown>)["tex_string"];
  } catch {
    // NOTE: Mobjects in this port raise on unknown property reads, which
    // is the equivalent of Python's missing attribute (AssertionError below).
    return undefined;
  }
}

export class TransformMatchingTex extends TransformMatchingAbstractBase {
  constructor(
    mobject: Mobject,
    target_mobject: Mobject,
    options: TransformMatchingOptions = {},
  ) {
    const {
      transform_mismatches = false,
      fade_transform_mismatches = false,
      key_map = null,
      ...kwargs
    } = options;
    super(mobject, target_mobject, {
      transform_mismatches,
      fade_transform_mismatches,
      key_map,
      ...kwargs,
    });
  }

  static override get_mobject_parts(mobject: Mobject): Mobject[] {
    // NOTE: Python also accepts the OpenGL Group/VGroup variants; only the
    // Cairo Group/VGroup branch applies here (OpenGL is not ported).
    if (mobject instanceof Group || mobject instanceof VGroup) {
      return mobject.submobjects.flatMap((submobject) =>
        TransformMatchingTex.get_mobject_parts(submobject),
      );
    }
    if (readTexString(mobject) === undefined) {
      const error = new Error(
        `assert hasattr(mobject, "tex_string"): got ${mobject.constructor?.name ?? typeof mobject}`,
      );
      error.name = "AssertionError";
      throw error;
    }
    return [...mobject.submobjects];
  }

  static override get_mobject_key(mobject: Mobject): string {
    const texString = readTexString(mobject);
    if (typeof texString !== "string") {
      const error = new Error(
        `assert isinstance(mobject, MathTexPart): got ${mobject.constructor?.name ?? typeof mobject}`,
      );
      error.name = "AssertionError";
      throw error;
    }
    return texString;
  }
}
