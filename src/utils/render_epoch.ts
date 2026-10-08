/**
 * Global revision counter for renderable mobject geometry.
 *
 * The WebGL 3D backend caches uploaded world-space geometry per VMobject and
 * validates the cache against the camera's colour inputs.  Revalidating every
 * face on every frame costs microseconds each; with thousands of faces that
 * dominates the frame even when nothing changed.  Bumping this counter from
 * the low-level point/rgba setters lets the backend skip per-face validation
 * entirely on frames where no geometry or colour changed (the common case:
 * camera movement during interaction).
 */
let GEOMETRY_EPOCH = 0;

export function mobject_geometry_epoch(): number {
  return GEOMETRY_EPOCH;
}

export function bump_mobject_geometry_epoch(): void {
  GEOMETRY_EPOCH += 1;
}

/**
 * Global revision counter for mobject family structure (the `submobjects`
 * tree).
 *
 * Unlike {@link mobject_geometry_epoch}, this is NOT bumped by point writes:
 * it changes only when children are added, removed or replaced.  Consumers
 * that cache per-family traversals (e.g. `Transform`'s zipped families) use it
 * to stay valid across frames while still noticing structural rebuilds, such
 * as a `DecimalNumber` updater recreating its glyphs mid-animation.
 */
let STRUCTURE_EPOCH = 0;

export function mobject_structure_epoch(): number {
  return STRUCTURE_EPOCH;
}

export function bump_mobject_structure_epoch(): void {
  STRUCTURE_EPOCH += 1;
}
