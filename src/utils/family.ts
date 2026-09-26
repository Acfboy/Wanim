import type { Mobject } from "../mobject/mobject.js";
import { remove_list_redundancies } from "./iterables.js";

export function extract_mobject_family_members(
  mobjects: Iterable<Mobject>,
  use_z_index = false,
  only_those_with_points = false,
): Mobject[] {
  const listed = Array.from(mobjects);
  const extracted = remove_list_redundancies(
    listed.flatMap((mobject) => {
      // NOTE: the TS CoordinateSystem hierarchy (NumberPlane/Axes) is not
      // Mobject-based yet, but Scene.add(VectorScene.add_plane/add_axes)
      // passes those objects through restructure_mobjects. Such items have
      // no get_family; their "family" is just themselves (matches the
      // pre-restructure pass-through behavior for that path).
      if (typeof (mobject as Mobject)?.get_family !== "function") return [mobject];
      return only_those_with_points
        ? (mobject as Mobject).family_members_with_points()
        : (mobject as Mobject).get_family();
    }),
  );
  if (use_z_index) {
    return [...extracted].sort((a, b) => a.z_index - b.z_index);
  }
  return extracted;
}

export function restructure_list_to_exclude_certain_family_members(
  mobject_list: Mobject[],
  to_remove: Mobject[],
): Mobject[] {
  const new_list: Mobject[] = [];
  const removed = extract_mobject_family_members(to_remove);

  function add_safe_mobjects_from_list(
    list_to_examine: Mobject[],
    set_to_remove: Set<Mobject>,
  ): void {
    for (const mob of list_to_examine) {
      if (set_to_remove.has(mob)) continue;
      // NOTE: see extract_mobject_family_members -- non-Mobject items
      // (TS CoordinateSystem hierarchy) have no get_family; keep them as-is.
      const family: Mobject[] =
        typeof mob?.get_family === "function" ? mob.get_family() : [mob];
      const intersect = new Set(family.filter((member) => set_to_remove.has(member)));
      if (intersect.size > 0) {
        add_safe_mobjects_from_list(mob.submobjects, intersect);
      } else {
        new_list.push(mob);
      }
    }
  }

  add_safe_mobjects_from_list(mobject_list, new Set(removed));
  return new_list;
}
