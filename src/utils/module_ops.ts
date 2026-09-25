import { existsSync, readFileSync } from "node:fs";
import { extname, resolve, sep } from "node:path";
import { config } from "../_config/utils.js";
import {
  CHOOSE_NUMBER_MESSAGE,
  INVALID_NUMBER_MESSAGE,
  NO_SCENE_MESSAGE,
  SCENE_NOT_FOUND_MESSAGE,
} from "../constants.js";
import type { Scene as SceneType } from "../scene/scene.js";

export interface SceneModule {
  __name__: string;
  __file__: string;
  [key: string]: unknown;
}

/** A Scene subclass (constructor). Mirrors Python's `type[Scene]`. */
export type SceneClass = (new (...args: never[]) => SceneType) & {
  __name__?: string;
  __module__?: string;
};

function sceneDisplayName(scene_class: SceneClass): string {
  return scene_class.__name__ ?? scene_class.name;
}

function exitError(message: string): Error {
  // NOTE: Python calls sys.exit() here; the TS port throws instead since
  // there is no SystemExit equivalent in the web-runnable runtime.
  return new Error(message);
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  throw error;
}

function fileNotFoundError(message: string): Error {
  const error = new Error(message);
  error.name = "FileNotFoundError";
  throw error;
}

/** Load a scene module from a file path (or "-" for stdin).
 *
 * Faithful in its validation behavior (suffix / existence checks); dynamic
 * Python execution has no TS equivalent, so a successfully located file
 * yields a lightweight module record instead of an executed namespace.
 */
export function get_module(file_name: string): SceneModule {
  if (file_name === "-") {
    throw new Error("TODO: get_module('-') requires stdin code execution; not ported yet");
  }
  const resolved = resolve(file_name);
  if (!existsSync(resolved)) {
    throw fileNotFoundError(`${file_name} not found`);
  }
  if (extname(resolved) !== ".py") {
    throw valueError(`${file_name} is not a valid Manim python script.`);
  }
  const module_name = resolved
    .replace(/\.py$/, "")
    .split(sep)
    .filter((part: string) => part.length > 0)
    .join(".");
  return { __name__: module_name, __file__: resolved };
}

export function is_child_scene(obj: unknown, module: SceneModule): boolean {
  if (typeof obj !== "function") return false;
  const ctor = obj as { prototype?: unknown; __module__?: unknown };
  // NOTE: the `obj != Scene` exclusion falls out of isSceneSubclass: for the
  // Scene base class itself every prototype in its chain is either
  // self-constructed or not named "Scene", so the walk returns false.
  if (!isSceneSubclass(obj)) return false;
  if (typeof ctor.__module__ === "string") return ctor.__module__.startsWith(module.__name__);
  // NOTE: TS classes carry no __module__ attribute; membership in the
  // module record (established by the getmembers-style scan in
  // get_scene_classes_from_module) implies module ownership, so accept.
  return true;
}

/**
 * Structural stand-in for `issubclass(obj, Scene)`. Walks the prototype
 * chain for a constructor named "Scene" instead of importing the Scene
 * class, which would introduce a module cycle (module_ops -> scene ->
 * animation -> mobject). Class names are part of wanim's API fidelity
 * guarantees, so the name check is stable within this codebase.
 */
function isSceneSubclass(obj: object): boolean {
  let proto: unknown = (obj as { prototype?: unknown }).prototype;
  const seen = new Set<unknown>();
  while (proto !== null && proto !== undefined && typeof proto === "object") {
    if (seen.has(proto)) return false;
    seen.add(proto);
    const protoCtor = (proto as { constructor?: unknown }).constructor;
    if (protoCtor !== obj && (protoCtor as { name?: string } | undefined)?.name === "Scene") {
      return true;
    }
    proto = Object.getPrototypeOf(proto);
  }
  return false;
}

export function get_scene_classes_from_module(module: SceneModule): SceneClass[] {
  function is_child_scene(obj: unknown, mod: SceneModule): boolean {
    void mod;
    return is_child_scene_outer(obj, module);
  }

  return Object.values(module).filter((member) => is_child_scene(member, module)) as SceneClass[];
}

const is_child_scene_outer = is_child_scene;

export function get_scenes_to_render(scene_classes: SceneClass[]): SceneClass[] {
  if (scene_classes.length === 0) {
    console.error(NO_SCENE_MESSAGE);
    return [];
  }
  if (config.write_all) {
    return scene_classes;
  }
  const result: SceneClass[] = [];
  for (const scene_name of config.scene_names as string[]) {
    let found = false;
    for (const scene_class of scene_classes) {
      if (sceneDisplayName(scene_class) === scene_name) {
        result.push(scene_class);
        found = true;
        break;
      }
    }
    if (!found && scene_name !== "") {
      console.error(SCENE_NOT_FOUND_MESSAGE.replace("{}", scene_name));
    }
  }
  if (result.length > 0) {
    return result;
  }
  if (scene_classes.length === 1) {
    config.scene_names = [sceneDisplayName(scene_classes[0] as SceneClass)];
    return [scene_classes[0] as SceneClass];
  }
  return prompt_user_for_choice(scene_classes);
}

/** Pure choice-string parsing shared by prompt_user_for_choice (testable without stdin). */
export function parse_scene_choice(scene_classes: SceneClass[], user_input: string): SceneClass[] {
  const num_to_class = new Map<number, SceneClass>();
  scene_classes.forEach((scene_class, index) => {
    num_to_class.set(index + 1, scene_class);
  });
  if (user_input === "*") {
    return [...scene_classes];
  }
  return user_input
    .trim()
    .split(/\s*,\s*/)
    .map((num_str) => {
      if (!/^[+-]?\d+$/.test(num_str.trim())) {
        // Mirrors Python's ValueError branch (int() parse failure).
        throw exitError("No scenes were selected. Exiting.");
      }
      const selected = num_to_class.get(Number.parseInt(num_str.trim(), 10));
      if (selected === undefined) {
        // Mirrors Python's KeyError branch.
        throw exitError(INVALID_NUMBER_MESSAGE);
      }
      return selected;
    });
}

export function prompt_user_for_choice(scene_classes: SceneClass[]): SceneClass[] {
  // NOTE: Python sets SceneFileWriter.force_output_as_scene_name = True here;
  // there is no SceneFileWriter port yet (rendering is out of scope), so only
  // the selection behavior is ported.
  const num_to_class = new Map<number, SceneClass>();
  scene_classes.forEach((scene_class, index) => {
    const name = sceneDisplayName(scene_class);
    console.log(`${index + 1}: ${name}`);
    num_to_class.set(index + 1, scene_class);
  });
  let user_input: string;
  try {
    process.stdout.write(`[log.message] ${CHOOSE_NUMBER_MESSAGE} [/log.message]`);
    user_input = readFileSync(0, "utf-8").split("\n")[0] ?? "";
  } catch {
    // Mirrors Python's EOFError branch (sys.exit(1)).
    throw exitError("EOFError");
  }
  const selected = parse_scene_choice(scene_classes, user_input);
  config.scene_names = selected.map(sceneDisplayName);
  return selected;
}
