// ESM resolve hook that maps the bare specifier "wanim" (and "wanim/<sub>")
// to the built package, so a user's scene file can simply do
//   import { Scene, Circle, Create } from "wanim";
// regardless of where that file lives on disk.
//
// WANIM_DIST must point at the built `dist/` directory (set by profile_scene.mjs).
import { pathToFileURL } from "node:url";
import path from "node:path";

const dist = process.env.WANIM_DIST;

export async function resolve(specifier, context, nextResolve) {
  if (dist && (specifier === "wanim" || specifier.startsWith("wanim/"))) {
    const sub = specifier === "wanim" ? "index.js" : specifier.slice("wanim/".length);
    const target = path.join(dist, sub);
    return { url: pathToFileURL(target).href, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
