import type { Mobject } from "../mobject/mobject.js";
import { remove_list_redundancies } from "./iterables.js";

export function extract_mobject_family_members(
  mobjects: Iterable<Mobject>,
  use_z_index = false,
  only_those_with_points = false,
): Mobject[] {
  const listed = Array.from(mobjects);
  const extracted = remove_list_redundancies(
    listed.flatMap((mobject) =>
      only_those_with_points
        ? mobject.family_members_with_points()
        : mobject.get_family(),
    ),
  );
  if (use_z_index) {
    return [...extracted].sort((a, b) => a.z_index - b.z_index);
  }
  return extracted;
}
