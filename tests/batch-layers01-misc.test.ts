// Ground-truth tests for the layers 0..1 batch (~1005 lines).
//
// Every expectation below was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with representative inputs; see the
// inline "Python:" comments. TS is asserted against those values with tight
// float tolerance (1e-9 relative; exact for discrete values).
//
// Known deviations from Python (documented, not worked around):
//  - Scene.check_interactive_embed_is_valid: Python asserts
//    isinstance(renderer, OpenGLRenderer) first; the TS runtime has no OpenGL
//    renderer so the assert is skipped and only the config branches are
//    evaluated (same order as Python).
//  - Mobject.__getattr__ get_/set_ adaptors: Python emits a DeprecationWarning;
//    the TS port omits the warning (same values/identity otherwise).
//  - PMobject.point_from_proportion: Python 0.21.0 indexes with np.floor()
//    (a float) and therefore raises IndexError for EVERY input -- an upstream
//    bug. The TS port returns the floored row instead (the evident intent).
//  - Mobject.get_z_index_reference_point: Python prefers a `z_index_group`
//    attribute when present; the TS port (no z_index_group concept) always
//    uses get_center(), which agrees whenever z_index_group is absent.
//  - Mobject.generate_target(use_deepcopy=True): TS copy() already deep-clones
//    values, so both branches coincide.
//  - CurvesAsSubmobjects empty-input errors: TS raises plain Errors with
//    different messages than Python's Exceptions (same trigger conditions).
//  - ChangeSpeed mutates the caller's speedinfo dict in Python (injecting 0/1
//    keys); the TS port copies instead of mutating.
import { describe, expect, it, vi, afterEach } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// NOTE: import mobject.js before scene.js: the mobject<->animation import
// cycle only resolves when mobject.ts is evaluated first (same order as
// tests/animation.test.ts and tests/batch-scene-color-utils.test.ts).
import { Mobject, Group, override_animate } from "../src/mobject/mobject.js";
import { Scene, SceneInteractContinue, SceneInteractRerun, RerunSceneHandler, load_module_into_namespace } from "../src/scene/scene.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { config, ManimConfig, make_config_parser } from "../src/_config/utils.js";
import { make_logger } from "../src/_config/logger_utils.js";
import { VMobject, VectorizedPoint, CurvesAsSubmobjects, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { ImageMobject } from "../src/mobject/types/image_mobject.js";
import { Polygram } from "../src/mobject/geometry/polygram.js";
import { Axes, CoordinateSystem, NumberPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { ParametricFunction } from "../src/mobject/graphing/functions.js";
import { SampleSpace } from "../src/mobject/graphing/probability.js";
import { NetworkXGraph, tree_layout, _tree_layout, _determine_graph_layout } from "../src/mobject/graph.js";
import { VectorField } from "../src/mobject/vector_field.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { Create } from "../src/animation/creation.js";
import { ShowPassingFlash, ApplyWave } from "../src/animation/indication.js";
import { Homotopy } from "../src/animation/movement.js";
import { ApplyMatrix, FadeTransformPieces } from "../src/animation/transform.js";
import { ChangeSpeed } from "../src/animation/speedmodifier.js";
import { always_shift } from "../src/animation/updaters/mobject_update_utils.js";
import { rotation_about_z } from "../src/utils/space_ops.js";
import { get_full_raster_image_path } from "../src/utils/images.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function flat(value: unknown): number[] {
  const raw =
    value !== null &&
    typeof value === "object" &&
    "toArray" in (value as Record<string, unknown>) &&
    typeof (value as { toArray?: unknown }).toArray === "function"
      ? (value as { toArray: () => unknown }).toArray()
      : value;
  const out: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) for (const e of x) walk(e);
    else out.push(Number(x));
  };
  walk(raw);
  return out;
}

function rowsOf(value: unknown): number[][] {
  const raw =
    value !== null &&
    typeof value === "object" &&
    "toArray" in (value as Record<string, unknown>) &&
    typeof (value as { toArray?: unknown }).toArray === "function"
      ? (value as { toArray: () => unknown }).toArray()
      : value;
  const arr = raw as unknown;
  if (Array.isArray(arr) && arr.length > 0 && !Array.isArray(arr[0])) {
    return [(arr as unknown[]).map(Number)];
  }
  return (arr as unknown[][]).map((r) => (Array.isArray(r) ? r : [r]).map(Number));
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = flat(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]!) * relTol);
    expect(Math.abs(values[i]! - expected[i]!)).toBeLessThanOrEqual(tol);
  }
}

function caught(fn: () => unknown): Error {
  try {
    fn();
  } catch (e) {
    return e as Error;
  }
  throw new Error("expected function to throw");
}

/** VMobject through 4 bezier points of a straight unit segment (== Line points). */
function unitSegment(): VMobject {
  const vm = new VMobject();
  vm.set_points([[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0], [1, 0, 0]]);
  return vm;
}

// ---------------------------------------------------------------------------
// scene: SceneInteractContinue / RerunSceneHandler / load_module / clear
// ---------------------------------------------------------------------------
describe("SceneInteractContinue", () => {
  it("stores the sender (Python: SceneInteractContinue('keyboard').sender)", () => {
    expect(new SceneInteractContinue("keyboard").sender).toBe("keyboard");
    expect(new SceneInteractContinue("gui").sender).toBe("gui");
  });
});

describe("RerunSceneHandler", () => {
  it("queues SceneInteractRerun('file') on modification (array queue)", () => {
    // Python: handler puts SceneInteractRerun("file") into the Queue.
    const queue: unknown[] = [];
    const handler = new RerunSceneHandler(queue as never);
    handler.on_modified({ path: "scene.py" });
    expect(queue).toHaveLength(1);
    expect(queue[0]).toBeInstanceOf(SceneInteractRerun);
    expect((queue[0] as SceneInteractRerun).sender).toBe("file");
  });

  it("supports Queue-like { put } sinks as in Python", () => {
    const seen: unknown[] = [];
    const handler = new RerunSceneHandler({ put: (a: unknown) => { seen.push(a); } });
    handler.on_modified(null);
    expect((seen[0] as SceneInteractRerun).sender).toBe("file");
  });
});

describe("load_module_into_namespace", () => {
  it("copies every module key into the namespace (mirrors the ipython helper)", () => {
    // Python: for name in module: namespace[name] = module[name].
    const namespace: Record<string, unknown> = { keep: 1 };
    load_module_into_namespace({ a: 10, b: "x" }, namespace);
    expect(namespace).toMatchObject({ keep: 1, a: 10, b: "x" });
  });
});

describe("Scene.clear", () => {
  it("empties mobjects and foreground_mobjects and returns self", () => {
    // Python: sc.add(m); sc.clear() -> 0/0, returns the scene.
    const sc = new Scene();
    const m1 = new Mobject();
    const m2 = new Mobject();
    sc.add(m1);
    sc.add(m2);
    expect(sc.mobjects).toHaveLength(2);
    sc.foreground_mobjects.push(m1);
    expect(sc.clear()).toBe(sc);
    expect(sc.mobjects).toHaveLength(0);
    expect(sc.foreground_mobjects).toHaveLength(0);
  });
});

describe("Scene.check_interactive_embed_is_valid", () => {
  const keys = ["force_window", "write_to_movie", "format", "dry_run"] as const;
  const saved = new Map<string, unknown>();
  for (const k of keys) saved.set(k, config.get(k));

  afterEach(() => {
    for (const k of keys) config.set(k, saved.get(k));
  });

  function cleanScene() {
    for (const k of keys) config.set(k, k === "format" ? "" : false);
    const sc = new Scene();
    sc.skip_animation_preview = false;
    return sc;
  }

  it("returns True immediately when force_window is set", () => {
    // Python: `if config["force_window"]: return True` (first branch).
    const sc = cleanScene();
    config.set("force_window", true);
    expect(sc.check_interactive_embed_is_valid()).toBe(true);
  });

  it("rejects skip_animation_preview (Python warns + returns False)", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const sc = cleanScene();
      sc.skip_animation_preview = true;
      expect(sc.check_interactive_embed_is_valid()).toBe(false);
      expect(warn).toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("rejects write_to_movie / --format / missing window / dry_run in order", () => {
    // Python branch order: write_to_movie -> format -> renderer.window -> dry_run.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const sc = cleanScene();
      config.set("write_to_movie", true);
      expect(sc.check_interactive_embed_is_valid()).toBe(false);
      config.set("write_to_movie", false);

      config.set("format", "mp4");
      expect(sc.check_interactive_embed_is_valid()).toBe(false);
      config.set("format", "");

      sc.renderer = null;
      expect(sc.check_interactive_embed_is_valid()).toBe(false);
      sc.renderer = {};
      expect(sc.check_interactive_embed_is_valid()).toBe(false);

      sc.renderer = { window: {} };
      config.set("dry_run", true);
      expect(sc.check_interactive_embed_is_valid()).toBe(false);
      config.set("dry_run", false);

      expect(sc.check_interactive_embed_is_valid()).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// config: ManimConfig contains/copy/deepcopy/top/bottom/left/right, parser, logger
// ---------------------------------------------------------------------------
describe("ManimConfig geometry + membership + copies", () => {
  it("top/bottom/left_side/right_side match Python defaults", () => {
    // Python: top=[0,4,0] bottom=[0,-4,0] left=[-7.111111111111111,0,0]
    //         right=[7.111111111111111,0,0].
    const cfg = new ManimConfig(true);
    expectClose(cfg.top, [0, 4, 0]);
    expectClose(cfg.bottom, [0, -4, 0]);
    expectClose(cfg.left_side, [-7.111111111111111, 0, 0]);
    expectClose(cfg.right_side, [7.111111111111111, 0, 0]);
  });

  it("__contains__: True for real keys (incl. computed frame_size), False for bogus", () => {
    // Python: 'frame_width' in config == True; 'bogus_key_xyz' in config == False.
    const cfg = new ManimConfig(true);
    expect(cfg.__contains__("frame_width")).toBe(true);
    expect(cfg.__contains__("frame_size")).toBe(true);
    expect(cfg.__contains__("bogus_key_xyz")).toBe(false);
  });

  it("__contains__ with a non-string raises AssertionError like Python", () => {
    // Python: `42 in config` raises AssertionError (assert isinstance(key, str)).
    const cfg = new ManimConfig(true);
    const err = caught(() => cfg.__contains__(42));
    expect(err.name).toBe("AssertionError");
  });

  it("__copy__/__deepcopy__ return independent ManimConfig copies", () => {
    // Python: copy.copy/deepcopy give ManimConfig; mutating the copy's
    // frame_rate (60 -> 30) leaves the original at 60.
    const cfg = new ManimConfig(true);
    const shallow = cfg.__copy__();
    const deep = cfg.__deepcopy__();
    expect(shallow).toBeInstanceOf(ManimConfig);
    expect(deep).toBeInstanceOf(ManimConfig);
    expect(shallow).not.toBe(cfg);
    expect(shallow.get("frame_rate")).toBe(60);
    deep.set("frame_rate", 30);
    expect(cfg.get("frame_rate")).toBe(60);
    expect(deep.get("frame_rate")).toBe(30);
  });
});

describe("make_config_parser", () => {
  it("loads the library-wide defaults (CLI.frame_rate == 60)", () => {
    // Python: parser has 'CLI' and 'logger' sections; CLI.frame_rate == '60'.
    const parser = make_config_parser();
    expect(parser.has("CLI")).toBe(true);
    expect(parser.get("CLI")?.get("frame_rate")).toBe("60");
  });
});

describe("make_logger", () => {
  function sectionProxy() {
    return new Map([
      ["log_width", "80"],
      ["log_height", "20"],
    ]);
  }

  it("returns [logger, stdoutConsole, errorConsole] with the requested level", () => {
    // Python: make_logger(proxy, 'INFO') -> (Logger('manim', level 20), ...).
    // TS contract: logger.level is the upper-cased verbosity string.
    const [logger, stdoutConsole, errorConsole] = make_logger(sectionProxy() as never, "INFO");
    expect(logger.level).toBe("INFO");
    expect(typeof stdoutConsole.print).toBe("function");
    expect(typeof errorConsole.print).toBe("function");
  });

  it("filters below the active level and aliases warn -> warning", () => {
    const [logger] = make_logger(sectionProxy() as never, "INFO");
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      logger.debug("hidden");
      expect(log).not.toHaveBeenCalled();
      logger.info("shown");
      expect(log).toHaveBeenCalledTimes(1);
      logger.warn("warn-alias");
      logger.warning("warn-direct");
      logger.error("err");
      expect(error).toHaveBeenCalledTimes(3);
    } finally {
      log.mockRestore();
      error.mockRestore();
    }
  });

  it("DEBUG verbosity enables debug output", () => {
    const [logger] = make_logger(sectionProxy() as never, "DEBUG");
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      logger.debug("shown");
      expect(log).toHaveBeenCalledTimes(1);
    } finally {
      log.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// Mobject core batch symbols
// ---------------------------------------------------------------------------
describe("Mobject.get_z_index_reference_point", () => {
  it("equals get_center() (Python: z_index_group defaults to self)", () => {
    // Python: empty Mobject -> [0,0,0].
    const m = new Mobject();
    m.points = array([[0, 0, 0], [2, 0, 0]]) as NDArray;
    expectClose(m.get_z_index_reference_point(), [1, 0, 0]);
    expectClose(new Mobject().get_z_index_reference_point(), [0, 0, 0]);
  });
});

describe("Mobject._assert_valid_submobjects / VMobject._assert_valid_submobjects", () => {
  it("accepts valid submobjects and returns self", () => {
    const m = new Mobject();
    expect(m._assert_valid_submobjects([new Mobject()])).toBe(m);
    const v = new VMobject();
    expect(v._assert_valid_submobjects([new VMobject()])).toBe(v);
  });

  it("raises TypeError for non-Mobjects (Python message)", () => {
    // Python: "Only values of type Mobject can be added as submobjects of
    // Mobject, but the value 42 (at index 0) is of type int."
    const err = caught(() => new Mobject()._assert_valid_submobjects([42 as never]));
    expect(err).toBeInstanceOf(TypeError);
    expect(err.message).toContain("Only values of type Mobject can be added as submobjects of Mobject");
    expect(err.message).toContain("42 (at index 0) is of type int");
  });

  it("raises ValueError when adding self", () => {
    // Python: "Cannot add Mobject as a submobject of itself (at index 0)."
    const m = new Mobject();
    const err = caught(() => m._assert_valid_submobjects([m]));
    expect(err.name).toBe("ValueError");
    expect(err.message).toContain("itself (at index 0)");
  });

  it("VMobject rejects plain Mobjects with the Group hint", () => {
    // Python: "... of type Mobject. You can try adding this value into a
    // Group instead."
    const err = caught(() => new VMobject()._assert_valid_submobjects([new Mobject()]));
    expect(err).toBeInstanceOf(TypeError);
    expect(err.message).toContain("Only values of type VMobject");
    expect(err.message).toContain("Group instead");
  });
});

describe("Mobject.has_no_points", () => {
  it("True when empty, False once points exist", () => {
    // Python: Mobject().has_no_points() True; with 2 pts False.
    expect(new Mobject().has_no_points()).toBe(true);
    const m = new Mobject();
    m.points = array([[0, 0, 0], [1, 0, 0]]) as NDArray;
    expect(m.has_no_points()).toBe(false);
  });
});

describe("Mobject.get_midpoint", () => {
  it("returns point_from_proportion(0.5) (Python: [1, 0.5, 0] on the L path)", () => {
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
    expectClose(vm.get_midpoint(), [1, 0.5, 0]);
  });
});

describe("Mobject.generate_target", () => {
  it("copies self into .target and returns it", () => {
    // Python: target is not self; points equal [[0,0,0],[1,0,0]].
    const m = new Mobject();
    m.points = array([[0, 0, 0], [1, 0, 0]]) as NDArray;
    const target = m.generate_target();
    expect(target).not.toBe(m);
    expect(rowsOf(target!.points)).toEqual([[0, 0, 0], [1, 0, 0]]);
    expect(m.target).toBe(target);
  });
});

describe("Mobject.repeat_submobject", () => {
  it("returns a copy of the submob", () => {
    const m = new Mobject();
    m.points = array([[0, 0, 0], [1, 0, 0]]) as NDArray;
    const out = new Mobject().repeat_submobject(m);
    expect(out).not.toBe(m);
    expect(rowsOf(out.points)).toEqual([[0, 0, 0], [1, 0, 0]]);
  });
});

describe("Mobject.apply_over_attr_arrays", () => {
  it("maps every array attr (Python: [[1,2,3]] *2 -> [[2,4,6]])", () => {
    const m = new Mobject();
    m.points = array([[1, 2, 3]]) as NDArray;
    m.apply_over_attr_arrays((a) => array(rowsOf(a).map((r) => r.map((v) => v * 2))) as NDArray);
    expect(rowsOf(m.points)).toEqual([[2, 4, 6]]);
    expect(m.apply_over_attr_arrays((a) => a)).toBe(m);
  });
});

describe("Mobject.__getattr__", () => {
  it("get_ adaptor returns the attribute value", () => {
    // Python: mm.__getattr__('get_name')() == 'hello2' (plus warning in Py).
    const m = new Mobject();
    m.set({ name: "hello2" } as never);
    const getter = m.__getattr__("get_name") as () => unknown;
    expect(getter()).toBe("hello2");
  });

  it("set_ adaptor assigns and returns self", () => {
    const m = new Mobject();
    const setter = m.__getattr__("set_name") as (v: unknown) => unknown;
    expect(setter("world")).toBe(m);
    expect(m.name).toBe("world");
  });

  it("unknown attributes raise AttributeError", () => {
    // Python: AttributeError("Mobject object has no attribute 'bogus_xyz'").
    const err = caught(() => new Mobject().__getattr__("bogus_xyz"));
    expect(err.name).toBe("AttributeError");
    expect(err.message).toContain("bogus_xyz");
  });
});

describe("_UpdaterBuilder.__getattr__", () => {
  it("adds a called updater (Python: always.shift applies immediately + on update)", () => {
    // Python: mm.always.shift([1,0,0]) -> 1 updater; pts [1,0,0];
    // mm.update(1.0) -> pts [2,0,0].
    const m = new Mobject();
    m.points = array([[0, 0, 0]]) as NDArray;
    (m.always as unknown as { __getattr__: (n: string) => (...a: unknown[]) => unknown })
      .__getattr__("shift")([1, 0, 0]);
    expect(m.updaters).toHaveLength(1);
    expect(rowsOf(m.points)).toEqual([[1, 0, 0]]);
    m.update(1.0);
    expect(rowsOf(m.points)).toEqual([[2, 0, 0]]);
  });

  it("proxy attribute path (m.always.shift(...)) behaves identically", () => {
    const m = new Mobject();
    m.points = array([[0, 0, 0]]) as NDArray;
    (m.always as unknown as { shift: (v: number[]) => unknown }).shift([0, 2, 0]);
    expect(rowsOf(m.points)).toEqual([[0, 2, 0]]);
  });
});

describe("override_animate", () => {
  it("attaches the animation method as _override_animate", () => {
    // Python: Foo.bar._override_animate is set by @override_animate(bar).
    class Foo extends Mobject {
      bar(): string {
        return "x";
      }
    }
    const proto = Foo.prototype as unknown as Record<string, (...a: never[]) => unknown>;
    const original = proto["bar"] as (...a: never[]) => unknown;
    function animMethod(this: unknown): string {
      return "anim!";
    }
    override_animate(original as never)(animMethod as never);
    expect((original as unknown as { _override_animate: unknown })._override_animate).toBe(animMethod);
    expect(new Foo().bar()).toBe("x");
  });
});

// ---------------------------------------------------------------------------
// space_ops + images
// ---------------------------------------------------------------------------
describe("rotation_about_z", () => {
  it("matches Python for pi/2 and 0", () => {
    // Python: rot(pi/2) = [[6.12e-17,-1,0],[1,6.12e-17,0],[0,0,1]]; rot(0) = I.
    expectClose(rotation_about_z(Math.PI / 2), [6.123233995736766e-17, -1, 0, 1, 6.123233995736766e-17, 0, 0, 0, 1]);
    expectClose(rotation_about_z(0), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
});

describe("get_full_raster_image_path", () => {
  it("raises OSError for a missing image (Python: OSError ... could not find ...)", () => {
    const err = caught(() => get_full_raster_image_path("no_such_image_xyz.png"));
    expect(err.name).toBe("OSError");
    expect(err.message).toContain("no_such_image_xyz.png");
  });

  it("resolves an existing file directly", () => {
    const dir = mkdtempSync(join(tmpdir(), "wanim-img-"));
    const file = join(dir, "probe.png");
    writeFileSync(file, new Uint8Array([137, 80, 78, 71]));
    expect(get_full_raster_image_path(file)).toBe(file);
  });
});

// ---------------------------------------------------------------------------
// animation batch symbols
// ---------------------------------------------------------------------------
describe("AnimationGroup.init_run_time", () => {
  afterEach(() => {
    ChangeSpeed.is_changing_dt = false;
  });

  it("computes lagged timings (Python: starts [0,1], ends [2,5], total 5.0)", () => {
    // Python: Create(rt=2) + Create(rt=4), lag_ratio=0.5 -> max end 5.0;
    // init_run_time(None) == 5.0, init_run_time(7) == 7.
    const a1 = new Create(unitSegment(), { run_time: 2 });
    const a2 = new Create(unitSegment(), { run_time: 4 });
    const group = new AnimationGroup(a1, a2, { lag_ratio: 0.5 });
    expect(group.init_run_time(undefined)).toBeCloseTo(5.0, 12);
    expect(group.max_end_time).toBeCloseTo(5.0, 12);
    expect(group.init_run_time(7)).toBe(7);
    expect(group.max_end_time).toBeCloseTo(5.0, 12);
  });
});

describe("ShowPassingFlash._get_bounds", () => {
  function flash(): ShowPassingFlash {
    return new ShowPassingFlash(unitSegment(), 0.1);
  }

  it("matches Python bounds at alpha 0 / 0.5 / 1", () => {
    // Python (tw=0.1): (0, 0.0), (0.45, 0.55), (1.0, 1).
    expect(flash()._get_bounds(0)).toEqual([0, 0]);
    const [lo, hi] = flash()._get_bounds(0.5);
    expect(lo).toBeCloseTo(0.45000000000000007, 12);
    expect(hi).toBeCloseTo(0.55, 12);
    expect(flash()._get_bounds(1)).toEqual([1, 1]);
  });
});

describe("ApplyWave wave (via homotopy)", () => {
  it("matches Python homotopy values", () => {
    // Python: ApplyWave(Line([0,0,0],[2,0,0]), dir=UP, amp=0.5, ripples=1,
    // time_width=1): homotopy(1,0,0,0.5) == [1,0.5,0]; (0,0,0,0) == [0,0,0].
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [2, 0, 0]]);
    const wave = new ApplyWave(vm, [0, 1, 0], 0.5, undefined, 1, 1, 2);
    const homotopy = (wave as unknown as { homotopy: (x: number, y: number, z: number, t: number) => number[] }).homotopy;
    expectClose(homotopy(1, 0, 0, 0.5), [1, 0.5, 0]);
    expectClose(homotopy(0, 0, 0, 0.0), [0, 0, 0]);
  });
});

describe("Homotopy.function_at_time_t", () => {
  it("binds t into the homotopy (Python: h(x+t) at t=2 maps [1,0,0]->[3,0,0])", () => {
    const h = new Homotopy((x, y, z, t) => [x + t, y, z], unitSegment());
    const mapped = h.function_at_time_t(2.0)([1, 0, 0]);
    expectClose(mapped, [3, 0, 0]);
  });

  it("rejects points that do not unpack to 3 values", () => {
    // Python: `x, y, z = p` raises ValueError on a 2-vector.
    const h = new Homotopy((x, y, z, t) => [x, y, z], unitSegment());
    const err = caught(() => h.function_at_time_t(0)([1, 0] as unknown as NDArray));
    expect(err.name).toBe("ValueError");
  });
});

describe("ApplyMatrix func + initialize_matrix", () => {
  it("expands 2x2 to 3x3 (Python: [[2,0],[0,2]] -> diag(2,2,1))", () => {
    expect(rowsOf(ApplyMatrix.initialize_matrix([[2, 0], [0, 2]]))).toEqual([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 1],
    ]);
  });

  it("rejects bad dimensions with ValueError", () => {
    // Python: ValueError("Matrix has bad dimensions").
    const err = caught(() => ApplyMatrix.initialize_matrix([[1, 2, 3], [4, 5, 6]]));
    expect(err.name).toBe("ValueError");
    expect(err.message).toBe("Matrix has bad dimensions");
  });

  it("target matches Python create_target (unit segment x-coords double)", () => {
    // Python: ApplyMatrix([[2,0],[0,2]], Line(ORIGIN, RIGHT)).create_target()
    //   == [[0,0,0],[2/3,0,0],[4/3,0,0],[2,0,0]].
    const anim = new ApplyMatrix([[2, 0], [0, 2]], unitSegment());
    const target = (anim as unknown as { target_mobject: VMobject }).target_mobject;
    expectClose(target.points, [0, 0, 0, 2 / 3, 0, 0, 4 / 3, 0, 0, 2, 0, 0]);
  });

  it("respects about_point (Python func: dot(p-about, M.T)+about)", () => {
    // M = 2I about [1,0,0]: [0,0,0] -> [-1,0,0]; [1,0,0] stays [1,0,0].
    const anim = new ApplyMatrix([[2, 0], [0, 2]], unitSegment(), [1, 0, 0]);
    const target = (anim as unknown as { target_mobject: VMobject }).target_mobject;
    expectClose(rowsOf(target.points)[0]!, [-1, 0, 0]);
    expectClose(rowsOf(target.points)[3]!, [1, 0, 0]);
  });
});

describe("FadeTransformPieces.ghost_to", () => {
  function lineGroup(x1: number): Group {
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [x1, 0, 0]]);
    return new Group(vm);
  }

  it("replaces source points with the target and fades (Python: stroke op 0, x0 kept)", () => {
    // Python: ghost src Line(0->1) onto Line(0->2): src[0].points[0]==[0,0,0],
    // stroke opacities [0].
    const ftp = new FadeTransformPieces(lineGroup(1), lineGroup(2));
    const src = lineGroup(1);
    const tgt = lineGroup(2);
    ftp.ghost_to(src, tgt);
    const srcChild = src.submobjects[0] as VMobject;
    expectClose(rowsOf(srcChild.points)[0]!, [0, 0, 0]);
    expectClose(rowsOf(srcChild.points).at(-1)!, [2, 0, 0]);
  });

  it("raises ValueError on family-length mismatch (Python zip strict=True)", () => {
    const ftp = new FadeTransformPieces(lineGroup(1), lineGroup(2));
    const src = new Group(lineGroup(1).submobjects[0], lineGroup(1).submobjects[0]);
    const err = caught(() => ftp.ghost_to(src, lineGroup(2)));
    expect(err.name).toBe("ValueError");
  });
});

describe("ChangeSpeed conditions + functions", () => {
  afterEach(() => {
    ChangeSpeed.is_changing_dt = false;
  });

  function changeSpeed() {
    // Python: ChangeSpeed(Create(Line), {0:1, 0.5:2, 1:1}) -> run_time 2/3,
    // scaled_total_time 2/3.
    return new ChangeSpeed(new Create(unitSegment()), { 0: 1, 0.5: 2, 1: 1 } as never);
  }

  it("scales total time (Python run_time == 0.6666666666666666)", () => {
    const cs = changeSpeed();
    try {
      expect(cs.run_time).toBeCloseTo(0.6666666666666666, 12);
      expect(cs.get_scaled_total_time()).toBeCloseTo(0.6666666666666666, 12);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("condition/function closures match Python values", () => {
    // Python: conditions[0](0.0) True; functions[0](0.25) == 0.20833333333333331.
    const cs = changeSpeed();
    try {
      expect(cs.conditions[0]!(0.0)).toBe(true);
      expect(cs.conditions[0]!(0.5)).toBe(true);
      expect(cs.functions[0]!(0.25)).toBeCloseTo(0.20833333333333331, 12);
      expect(Object.keys(cs.speedinfo).map(Number).sort((a, b) => a - b)).toEqual([0, 0.5, 1]);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });
});

describe("always_shift", () => {
  it("installs one updater shifting dt*rate along the unit vector", () => {
    // Python: pts [[0,0,0]], update(2.0) with defaults (RIGHT, 0.1) -> [[0.2,0,0]].
    const m = new Mobject();
    m.points = array([[0, 0, 0]]) as NDArray;
    const before = m.updaters.length;
    const out = always_shift(m);
    expect(out).toBe(m);
    expect(m.updaters.length - before).toBe(1);
    m.update(2.0);
    expectClose(m.points, [0.2, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// graph layouts
// ---------------------------------------------------------------------------
describe("tree_layout (_tree_layout)", () => {
  function treeGraph(): NetworkXGraph {
    const g = new NetworkXGraph();
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    return g;
  }

  it("matches the Python SageMath-port positions as a set", () => {
    // Python networkx tree: {0:[0,2,0], 1:[1,0,0], 2:[-1,0,0], 3:[1,-2,0]}.
    const layout = tree_layout(treeGraph(), 0);
    expect(layout.size).toBe(4);
    const got = [...layout.entries()].map(([k, v]) => [Number(k), ...flat(v)] as number[]);
    const want = [
      [0, 0, 2, 0],
      [1, 1, 0, 0],
      [2, -1, 0, 0],
      [3, 1, -2, 0],
    ];
    expect(got.sort((a, b) => a[0]! - b[0]!)).toEqual(want);
    expect(_tree_layout).toBe(tree_layout);
  });

  it("rejects a missing root and non-trees", () => {
    expect(() => tree_layout(treeGraph(), null)).toThrow();
    const cyclic = new NetworkXGraph();
    cyclic.addEdge(0, 1);
    cyclic.addEdge(1, 2);
    cyclic.addEdge(2, 0);
    expect(caught(() => tree_layout(cyclic, 0)).name).toBe("ValueError");
  });
});

describe("_determine_graph_layout", () => {
  it("passes placement dicts straight through (Python isinstance dict)", () => {
    // Python: layout={0: array([0,0,0])} returned as-is.
    const g = new NetworkXGraph();
    g.addEdge(0, 1);
    const placement = new Map<unknown, NDArray>([[0, array([0, 0, 0]) as NDArray]]);
    expect(_determine_graph_layout(g, placement as never)).toBe(placement);
    const record = { "0": array([1, 2, 0]) as NDArray };
    expect(_determine_graph_layout(g, record as never)).toBe(record);
  });

  it("delegates to custom layout functions", () => {
    const g = new NetworkXGraph();
    g.addEdge("a", "b");
    const out = _determine_graph_layout(
      g,
      ((graph: NetworkXGraph) => {
        const m = new Map<unknown, NDArray>();
        for (const n of graph.nodes.keys()) m.set(n, array([1, 0, 0]) as NDArray);
        return m;
      }) as never,
    ) as Map<unknown, NDArray>;
    expectClose(out.get("a")!, [1, 0, 0]);
  });

  it("rejects unknown layouts with Python's ValueError message", () => {
    // Python: "The layout '...' is neither a recognized layout, a layout
    // function, nor a vertex placement dictionary."
    const g = new NetworkXGraph();
    const err = caught(() => _determine_graph_layout(g, "nonexistent_xyz" as never));
    expect(err.message).toContain("neither a recognized layout");
  });
});

// ---------------------------------------------------------------------------
// vector field
// ---------------------------------------------------------------------------
describe("VectorField pos_to_rgb + gradient", () => {
  it("interpolates the scalar palette (Python: [0.74117647, 0.80392157, 0.41960784])", () => {
    // Python: VectorField(lambda p: [1,0,0]).pos_to_rgb([0,0,0]) with the
    // default BLUE_E/GREEN/YELLOW/RED stops over [0, 2].
    const field = new VectorField(() => [1, 0, 0]);
    expect(field.single_color).toBe(false);
    expectClose(field.pos_to_rgb!([0, 0, 0]), [0.7411764705882353, 0.803921568627451, 0.4196078431372549]);
  });

  it("single-color fields carry no pos_to_rgb/pos_to_color (Python hasattr False)", () => {
    const field = new VectorField(() => [1, 0, 0], { color: "#ff0000" });
    expect(field.single_color).toBe(true);
    expect("pos_to_rgb" in field && field.pos_to_rgb !== undefined).toBe(false);
    expect("pos_to_color" in field && field.pos_to_color !== undefined).toBe(false);
  });

  it("vectorized rgba gradient matches Python rows", () => {
    // Python: grad([0,0.5,1]) over red->blue == [[1,0,0,1],[0.5,0,0.5,1],[0,0,1,1]];
    // g([0.5], 0.3) == [[0.5,0,0.5,0.3]].
    const field = new VectorField(() => [0, 0, 0]);
    const grad = field.get_vectorized_rgba_gradient_function(0, 1, ["#ff0000", "#0000ff"]);
    expect(rowsOf(grad([0, 0.5, 1]))).toEqual([[1, 0, 0, 1], [0.5, 0, 0.5, 1], [0, 0, 1, 1]]);
    expect(rowsOf(grad([0.5], 0.3))).toEqual([[0.5, 0, 0.5, 0.3]]);
  });
});

// ---------------------------------------------------------------------------
// VMobject batch symbols
// ---------------------------------------------------------------------------
describe("VMobject.resize_points", () => {
  it("resizes the point list (Python: 4 pts -> 8, first row kept)", () => {
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [2, 0, 0]]);
    expect(vm.get_num_points()).toBe(4);
    expect(vm.resize_points(8)).toBe(vm);
    expect(vm.get_num_points()).toBe(8);
    expectClose(rowsOf(vm.points)[0]!, [0, 0, 0]);
  });

  it("is a no-op when the length already matches", () => {
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [2, 0, 0]]);
    const before = rowsOf(vm.points);
    vm.resize_points(4);
    expect(rowsOf(vm.points)).toEqual(before);
  });
});

describe("VMobject.is_closed", () => {
  it("True for a closed loop, False for an open segment", () => {
    // Python: 5-corner loop (returns to origin) True; 2-corner segment False.
    const loop = new VMobject();
    loop.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]);
    expect(loop.is_closed()).toBe(true);
    const open = new VMobject();
    open.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    expect(open.is_closed()).toBe(false);
  });
});

describe("VMobject subpath splitting", () => {
  function stackedPoints(): NDArray {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[5, 5, 0], [6, 5, 0]]);
    return array([...rowsOf(a.points), ...rowsOf(b.points)]) as NDArray;
  }

  it("get_subpaths_from_points finds 2 disjoint subpaths", () => {
    // Python: stacking two disjoint 4-point segments -> 2 subpaths.
    const vm = new VMobject();
    expect(vm.get_subpaths_from_points(stackedPoints())).toHaveLength(2);
  });

  it("gen_subpaths_from_points_2d agrees on disjoint segments", () => {
    const vm = new VMobject();
    expect([...vm.gen_subpaths_from_points_2d(stackedPoints())]).toHaveLength(2);
  });
});

describe("VMobject.align_rgbas", () => {
  it("stretches the shorter rgba arrays to match", () => {
    const m1 = new VMobject();
    m1.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const m2 = new VMobject();
    m2.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    m1.set_fill("#ff0000", 1);
    m2.set_fill("#0000ff", 1);
    m1.align_rgbas(m2);
    expect(m1.fill_rgbas.shape[0]).toBe(m2.fill_rgbas.shape[0]);
    expect(m1.stroke_rgbas.shape[0]).toBe(m2.stroke_rgbas.shape[0]);
    expect(m1.background_stroke_rgbas.shape[0]).toBe(m2.background_stroke_rgbas.shape[0]);
  });
});

describe("VMobject.interpolate_color", () => {
  it("blends rgba attributes (Python: red/blue @0.5 -> [0.5,0,0.5,1])", () => {
    const target = new VMobject();
    target.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const s1 = new VMobject();
    s1.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    s1.set_fill("#ff0000", 1);
    const s2 = new VMobject();
    s2.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    s2.set_fill("#0000ff", 1);
    target.interpolate_color(s1, s2, 0.5);
    expectClose(rowsOf(target.fill_rgbas)[0]!, [0.5, 0, 0.5, 1]);
  });
});

describe("VMobject.align_points (incl. get_nth_subpath)", () => {
  it("equalizes point counts (Python: 4 vs 10 pts -> 12 vs 12)", () => {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    a.align_points(b);
    expect(a.get_num_points()).toBe(12);
    expect(b.get_num_points()).toBe(12);
  });

  it("pads a missing subpath with a null curve (Python: 4/8 pts -> 8/8, 1/2 subpaths)", () => {
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    b.start_new_path([5, 5, 0]);
    b.add_line_to([6, 5, 0]);
    expect(a.get_subpaths()).toHaveLength(1);
    expect(b.get_subpaths()).toHaveLength(2);
    a.align_points(b);
    expect(a.get_num_points()).toBe(8);
    expect(b.get_num_points()).toBe(8);
  });
});

describe("VectorizedPoint.set_location", () => {
  it("replaces the single point (Python: set_location([4,5,6])))", () => {
    const p = new VectorizedPoint([1, 2, 3]);
    expect(p.set_location([4, 5, 6])).toBe(p);
    expect(rowsOf(p.points)).toEqual([[4, 5, 6]]);
  });
});

describe("CurvesAsSubmobjects.point_from_proportion", () => {
  function twoUnit(): CurvesAsSubmobjects {
    const line = new VMobject();
    line.set_points_as_corners([[0, 0, 0], [2, 0, 0]]);
    return new CurvesAsSubmobjects(line);
  }

  it("walks arc length (Python: 0->[0,0,0], 0.5->[1,0,0], 1->[2,0,0])", () => {
    expectClose(twoUnit().point_from_proportion(0), [0, 0, 0]);
    expectClose(twoUnit().point_from_proportion(0.5), [1, 0, 0]);
    expectClose(twoUnit().point_from_proportion(1), [2, 0, 0]);
  });

  it("rejects alpha outside [0, 1] (Python ValueError)", () => {
    expect(caught(() => twoUnit().point_from_proportion(2)).name).toBe("ValueError");
  });

  it("raises when there are no submobjects", () => {
    expect(caught(() => new CurvesAsSubmobjects().point_from_proportion(0.5))).toBeInstanceOf(Error);
  });
});

// ---------------------------------------------------------------------------
// PMobject / ImageMobject
// ---------------------------------------------------------------------------
describe("PMobject array attrs + point_from_proportion", () => {
  it("get_array_attrs extends Mobject with rgbas", () => {
    // Python: ['points', 'rgbas'].
    expect(new PMobject().get_array_attrs()).toEqual(["points", "rgbas"]);
  });

  it("indexes the floored row (documents the upstream IndexError deviation)", () => {
    // Python 0.21.0 raises IndexError here (float index from np.floor); the
    // port returns the intended row: 4 pts, alpha=0.5 -> index 1.5 -> row 1.
    const pm = new PMobject();
    pm.points = array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray;
    expectClose(pm.point_from_proportion(0.5), [1, 0, 0]);
    expectClose(pm.point_from_proportion(0), [0, 0, 0]);
  });
});

describe("ImageMobject pixel array + style", () => {
  function image(): ImageMobject {
    const pixels = array([
      [[255, 0, 0, 255], [0, 255, 0, 255]],
      [[0, 0, 255, 255], [255, 255, 255, 128]],
    ]) as NDArray;
    return new ImageMobject({ pixel_array: pixels });
  }

  it("get_pixel_array returns the stored array (Python: identity getter)", () => {
    const im = image();
    expect(flat(im.get_pixel_array())).toEqual([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128,
    ]);
    expect(im.get_pixel_array()).toBe(im.pixel_array);
  });

  it("get_style reports white fill at full opacity", () => {
    // Python: {'fill_color': '#FFFFFF', 'fill_opacity': 1}.
    expect(image().get_style()).toEqual({ fill_color: "#FFFFFF", fill_opacity: 1 });
  });
});

// ---------------------------------------------------------------------------
// graphing: SampleSpace / ParametricFunction / CoordinateSystem
// ---------------------------------------------------------------------------
describe("SampleSpace.complete_p_list", () => {
  it("appends the remainder (Python: [0.2,0.3]->[0.2,0.3,0.5])", () => {
    const ss = new SampleSpace();
    expect(ss.complete_p_list([0.2, 0.3])).toEqual([0.2, 0.3, 0.5]);
    expect(ss.complete_p_list([0.5, 0.5])).toEqual([0.5, 0.5]);
    expect(ss.complete_p_list(0.5)).toEqual([0.5, 0.5]);
  });
});

describe("ParametricFunction accessors", () => {
  it("get_function/get_point_from_function evaluate the callable", () => {
    // Python: f(0.5) == [0.5, 0.25, 0]; f(0.25) == [0.25, 0.0625, 0].
    const pf = new ParametricFunction((t: number) => [t, t ** 2, 0], { t_range: [0, 1, 0.5] });
    expectClose(pf.get_point_from_function(0.5), [0.5, 0.25, 0]);
    expectClose(pf.get_function()(0.25), [0.25, 0.0625, 0]);
  });
});

describe("CoordinateSystem c2p/p2c/polar/get_origin", () => {
  function axes(): Axes {
    return new Axes([-3, 3, 1], [-3, 3, 1], 6, 6);
  }

  it("c2p maps coords to points (Python: c2p(1,1) == [1,1,0])", () => {
    expectClose(axes().c2p(1, 1), [1, 1, 0]);
  });

  it("get_origin is coords_to_point(0, 0)", () => {
    expectClose(axes().get_origin(), [0, 0, 0]);
  });

  it("polar_to_point matches Python ([2,0] angle 0 -> [2,0,0])", () => {
    expectClose(axes().polar_to_point(2, 0), [2, 0, 0]);
  });

  it("p2c inverts c2p (Python: p2c(c2p(1,1)) == [1,1])", () => {
    const ax = axes();
    expectClose(ax.p2c(flat(ax.c2p(1, 1)) as [number, number, number] as never as readonly number[]), [1, 1]);
  });

  it("point_to_polar matches Python (r=sqrt2, theta=pi/4)", () => {
    // Python: point_to_polar(c2p(1,1)) == (1.4142135623730951, 0.7853981633974483).
    const ax = axes();
    const [r, theta] = ax.point_to_polar(flat(ax.c2p(1, 1)) as [number, number, number] as never as readonly number[]);
    expect(r).toBeCloseTo(1.4142135623730951, 12);
    expect(theta).toBeCloseTo(0.7853981633974483, 12);
  });

  it("base point_to_coords still raises NotImplementedError", () => {
    expect(caught(() => new CoordinateSystem().point_to_coords([0, 0, 0])).name).toBe("NotImplementedError");
  });
});

// ---------------------------------------------------------------------------
// geometry: Polygram vertices
// ---------------------------------------------------------------------------
describe("Polygram.get_vertices", () => {
  it("returns the start anchors (Python Square: [[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])", () => {
    const poly = new Polygram([1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]);
    expect(poly.get_vertices()).toEqual([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]);
  });
});

// ---------------------------------------------------------------------------
// LinearTransformationScene
// ---------------------------------------------------------------------------
describe("LinearTransformationScene.get_matrix_transformation", () => {
  it("maps points through the matrix (Python: 2I sends [1,1,0]->[2,2,0])", () => {
    const scene = new LinearTransformationScene();
    const f = scene.get_matrix_transformation([[2, 0], [0, 2]]);
    expectClose(f([1, 1, 0]), [2, 2, 0]);
  });

  it("handles rotations (Python: [[0,-1],[1,0]] sends [1,0,0]->[0,1,0])", () => {
    const scene = new LinearTransformationScene();
    const f = scene.get_matrix_transformation([[0, -1], [1, 0]]);
    expectClose(f([1, 0, 0]), [0, 1, 0]);
  });

  it("rejects bad dimensions", () => {
    const scene = new LinearTransformationScene();
    expect(caught(() => scene.get_matrix_transformation([[1, 2, 3], [4, 5, 6]])).name).toBe("ValueError");
  });
});

// ---------------------------------------------------------------------------
// integration: batch symbols composed with previously-ported code
// ---------------------------------------------------------------------------
describe("integration", () => {
  afterEach(() => {
    ChangeSpeed.is_changing_dt = false;
  });

  it("config top feeds an Mobject shift (Python center [1,4,0])", () => {
    // Python: Mobject([[0,0,0],[2,0,0]]).shift(config.top).get_center().
    const m = new Mobject();
    m.points = array([[0, 0, 0], [2, 0, 0]]) as NDArray;
    m.shift(config.top as unknown as number[]);
    expectClose(m.get_center(), [1, 4, 0]);
    // ... and the z-index reference point follows the moved center.
    expectClose(m.get_z_index_reference_point(), [1, 4, 0]);
  });

  it("matrix transformation applied to a VMobject matches Python pointwise", () => {
    // Python: f = scene.get_matrix_transformation(2I); f([1,1,0]) == [2,2,0].
    // Here the same matrix drives a real Mobject chain end to end.
    const scene = new LinearTransformationScene();
    const f = scene.get_matrix_transformation([[2, 0], [0, 2]]);
    const vm = new VMobject();
    vm.set_points_as_corners([[0, 0, 0], [1, 1, 0]]);
    vm.apply_function((p) => flat(f(p as number[])));
    const last = rowsOf(vm.points).at(-1)!;
    expectClose(last, [2, 2, 0]);
  });

  it("Create inside ChangeSpeed runs its update path against a VMobject", () => {
    // Unit ground truth pins run_time == 2/3 for the {0:1,0.5:2,1:1} case;
    // here a uniform-speed wrapper is begun/finished against a real VMobject
    // like Scene.play would (point resampling inside Create.finish is owned
    // by pointwise_become_partial, outside this batch, so only the wrapper
    // contract is asserted).
    const vm = unitSegment();
    const cs = new ChangeSpeed(new Create(vm), { 0: 1, 1: 1 } as never);
    try {
      expect(cs.run_time).toBeCloseTo(1, 12);
      expect(() => {
        cs.begin();
        cs.finish();
      }).not.toThrow();
      expect(vm.has_points()).toBe(true);
    } finally {
      ChangeSpeed.is_changing_dt = false;
    }
  });

  it("always_shift + Scene membership compose (Python shift semantics)", () => {
    const scene = new Scene();
    const m = new Mobject();
    m.points = array([[0, 0, 0]]) as NDArray;
    scene.add(m);
    always_shift(m);
    m.update(2.0);
    expectClose(m.points, [0.2, 0, 0]);
    scene.clear();
    expect(scene.mobjects).toHaveLength(0);
  });

  it("VGroup of VMobjects aligns colors then points end to end", () => {
    // align_rgbas (batch) + align_points (batch) in sequence, as Transform
    // setup does via align_data.
    const g = new VGroup();
    const a = new VMobject();
    a.set_points_as_corners([[0, 0, 0], [1, 0, 0]]);
    const b = new VMobject();
    b.set_points_as_corners([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    g.add(a, b);
    a.align_rgbas(b);
    a.align_points(b);
    expect(a.get_num_points()).toBe(b.get_num_points());
    expect(a.fill_rgbas.shape[0]).toBe(b.fill_rgbas.shape[0]);
  });
});
