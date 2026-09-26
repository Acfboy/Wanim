import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { array } from "numpy-ts";
// NOTE: import mobject.js before scene.js: the mobject<->animation import
// cycle only resolves when mobject.ts is evaluated first (same order as
// tests/animation.test.ts).
import { Mobject } from "../src/mobject/mobject.js";
import { get_smooth_open_cubic_bezier_handle_points, is_closed } from "../src/utils/bezier.js";
import { choose } from "../src/utils/simple_functions.js";
import {
  HSV,
  ManimColor,
  RandomColorGenerator,
  is_sequence,
} from "../src/utils/color/core.js";
import { ManimConfig } from "../src/_config/utils.js";
import { JSONFormatter } from "../src/_config/logger_utils.js";
import { Section } from "../src/scene/section.js";
import {
  add_extension_if_not_present,
  guarantee_existence,
  modify_atime,
} from "../src/utils/file_ops.js";
import {
  RerunSceneHandler,
  Scene,
  SceneInteractRerun,
} from "../src/scene/scene.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import {
  AbstractImageMobject,
  RESAMPLING_ALGORITHMS,
} from "../src/mobject/types/image_mobject.js";
import { resize_array } from "../src/utils/iterables.js";
import {
  cartesian_to_spherical,
  cross,
  norm_squared,
  normalize,
} from "../src/utils/space_ops.js";
import { NumberLine } from "../src/mobject/graphing/number_line.js";
import {
  Axes,
  CoordinateSystem,
  ThreeDAxes,
  type AxisShadingSubmobject,
  type ShadedAxis,
} from "../src/mobject/graphing/coordinate_systems.js";
import { LinearBase, _ScaleBase } from "../src/mobject/graphing/scale.js";
import { ArrowTip } from "../src/mobject/geometry/tips.js";
import { TipableVMobject } from "../src/mobject/geometry/arc.js";
import { Sphere, Surface } from "../src/mobject/three_d/three_dimensions.js";
import { EndSceneEarlyException, RerunSceneException } from "../src/utils/exceptions.js";

// ---------------------------------------------------------------------------
// helpers (ground truths captured from ./original manim via the original manim implementation)
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
  const raw = (value as { toArray: () => unknown }).toArray() as unknown;
  if (!Array.isArray(raw)) return [[Number(raw)]];
  return (raw as unknown[]).map((r) => (Array.isArray(r) ? r : [r]).map(Number));
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = flat(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]!) * relTol);
    expect(Math.abs(values[i]! - expected[i]!)).toBeLessThanOrEqual(tol);
  }
}

function expectErrorName(fn: () => unknown, name: string): void {
  try {
    fn();
  } catch (e) {
    expect((e as Error).name).toBe(name);
    return;
  }
  throw new Error(`expected ${name} to be thrown`);
}

// ---------------------------------------------------------------------------
// manim.utils.bezier.get_smooth_open_cubic_bezier_handle_points
// ---------------------------------------------------------------------------
describe("get_smooth_open_cubic_bezier_handle_points", () => {
  it("matches python handles for 4 anchors", () => {
    const [h1, h2] = get_smooth_open_cubic_bezier_handle_points([
      [0, 0, 0],
      [1, 2, 0],
      [3, 1, 0],
      [4, 4, 0],
    ]);
    expectClose(h1, [
      0.22222222222222232, 1.0222222222222221, 0.0,
      1.5555555555555554, 1.9555555555555555, 0.0,
      3.555555555555556, 1.1555555555555557, 0.0,
    ]);
    expectClose(h2, [
      0.44444444444444464, 2.0444444444444443, 0.0,
      2.444444444444444, 0.8444444444444443, 0.0,
      3.7777777777777777, 2.577777777777778, 0.0,
    ]);
  });

  it("matches python handles for 3 anchors", () => {
    const [h1, h2] = get_smooth_open_cubic_bezier_handle_points([
      [0, 0, 0],
      [1, 1, 1],
      [2, 0, 0],
    ]);
    expectClose(h1, [0.33333333333333337, 0.5, 0.5, 1.3333333333333333, 1.0, 1.0]);
    expectClose(h2, [0.6666666666666667, 1.0, 1.0, 1.6666666666666665, 0.5, 0.5]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.bezier.is_closed
// ---------------------------------------------------------------------------
describe("is_closed", () => {
  it("matches python open/closed judgements", () => {
    expect(is_closed([[0, 0, 0], [1, 2, 3], [3, 2, 1], [0, 0, 0]])).toBe(true);
    expect(is_closed([[0, 0, 0], [1, 2, 3], [3, 2, 1], [1e-10, 1e-10, 1e-10]])).toBe(true);
    expect(is_closed([[0, 0, 0], [1, 2, 3], [3, 2, 1], [1e-2, 1e-2, 1e-2]])).toBe(false);
    expect(is_closed([[0, 0, 0], [1, 0, 0], [2, 0, 0]])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.simple_functions.choose
// ---------------------------------------------------------------------------
describe("choose", () => {
  it("matches math.comb values", () => {
    expect(choose(5, 2)).toBe(10);
    expect(choose(0, 0)).toBe(1);
    expect(choose(10, 0)).toBe(1);
    expect(choose(10, 10)).toBe(1);
    expect(choose(52, 5)).toBe(2598960);
    expect(choose(100, 50)).toBe(100891344545564193334812497256);
  });

  it("returns 0 for k > n like math.comb (python ground truth: no error)", () => {
    expect(choose(5, 6)).toBe(0);
    expect(choose(0, 5)).toBe(0);
  });

  it("raises ValueError for negative inputs, TypeError for non-integers", () => {
    expectErrorName(() => choose(-1, 2), "ValueError");
    expectErrorName(() => choose(5, -1), "ValueError");
    // python ground truth: math.comb raises TypeError for floats
    expectErrorName(() => choose(5.5, 2), "TypeError");
    expectErrorName(() => choose(5, 2.5), "TypeError");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.ManimColor.parse.is_sequence (+ parse)
// ---------------------------------------------------------------------------
describe("is_sequence / ManimColor.parse", () => {
  it("mirrors isinstance(color, (list, tuple))", () => {
    expect(is_sequence(["RED", "BLUE"])).toBe(true);
    expect(is_sequence([])).toBe(true);
    expect(is_sequence("RED")).toBe(false);
    expect(is_sequence(null)).toBe(false);
    expect(is_sequence(undefined)).toBe(false);
  });

  it("parses single vs list like python", () => {
    expect(ManimColor.parse("RED")).toBeInstanceOf(ManimColor);
    const parsed = ManimColor.parse(["RED", "BLUE"]);
    expect(Array.isArray(parsed)).toBe(true);
    expect((parsed as ManimColor[])).toHaveLength(2);
    // Python: parse([]) -> [] (empty list is still a sequence)
    expect(ManimColor.parse([])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// HSV hue / saturation / value / _internal_value
// ---------------------------------------------------------------------------
describe("HSV", () => {
  it("exposes hue/saturation/value (python: 0.5/0.5/0.8)", () => {
    const h = new HSV([0.5, 0.5, 0.8]);
    expect(h.hue).toBeCloseTo(0.5, 12);
    expect(h.saturation).toBeCloseTo(0.5, 12);
    expect(h.value).toBeCloseTo(0.8, 12);
    expect(h.h).toBeCloseTo(0.5, 12);
    expect(h.s).toBeCloseTo(0.5, 12);
    expect(h.v).toBeCloseTo(0.8, 12);
  });

  it("converts to RGBA like colorsys (python: [0.4, 0.8, 0.8, 1.0])", () => {
    const h = new HSV([0.5, 0.5, 0.8]);
    expectClose(h._internal_value, [0.4, 0.8, 0.8, 1.0]);
  });

  it("keeps the 4th input element in _internal_space but alpha stays 1.0 (python quirk)", () => {
    const h = new HSV([0.1, 0.2, 0.3, 0.9]);
    expectClose(h.internal_space, [0.1, 0.2, 0.3, 0.9]);
    expectClose(h._internal_value, [0.3, 0.276, 0.24, 1.0]);
  });

  it("setters update the converted RGBA value", () => {
    const h = new HSV([0.5, 0.5, 0.8]);
    h.hue = 0;
    expect(h.hue).toBeCloseTo(0, 12);
    // colorsys.hsv_to_rgb(0, 0.5, 0.8) == (0.8, 0.4, 0.4)
    expectClose(h._internal_value, [0.8, 0.4, 0.4, 1.0]);
    h.saturation = 0;
    expectClose(h._internal_value, [0.8, 0.8, 0.8, 1.0]);
    h.value = 0.5;
    expectClose(h._internal_value, [0.5, 0.5, 0.5, 1.0]);
  });

  it("_from_internal inverts RGB via rgb_to_hsv (python: red -> [0,1,1] + alpha)", () => {
    const r = HSV._from_internal(array([1.0, 0.0, 0.0, 0.5]));
    expectClose(r.internal_space, [0.0, 1.0, 1.0, 0.5]);
    expectClose(r._internal_value, [1.0, 0.0, 0.0, 1.0]);
  });

  it("_internal_value setter validates input like python", () => {
    const h = new HSV([0.5, 0.5, 0.8]);
    expectErrorName(() => {
      h._internal_value = [1, 0, 0, 1] as unknown as typeof h._internal_value;
    }, "TypeError");
    expectErrorName(() => {
      h._internal_value = array([1, 0, 0]) as unknown as typeof h._internal_value;
    }, "TypeError");
    h._internal_value = array([0.0, 1.0, 0.0, 1.0]);
    // colorsys.rgb_to_hsv(0, 1, 0) == (1/3, 1, 1)
    expect(h.hue).toBeCloseTo(1 / 3, 9);
    expect(h.saturation).toBeCloseTo(1, 12);
    expect(h.value).toBeCloseTo(1, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_fill_rgbas / get_stroke_rgbas
// ---------------------------------------------------------------------------
describe("VMobject rgbas getters", () => {
  it("fresh VMobject matches python 1-row defaults", () => {
    const vm = new VMobject();
    // Python: fill [[1,1,1,0]] (WHITE, opacity 0), stroke [[1,1,1,1]]
    expect(rowsOf(vm.get_fill_rgbas())).toEqual([[1, 1, 1, 0]]);
    expect(rowsOf(vm.get_stroke_rgbas())).toEqual([[1, 1, 1, 1]]);
  });

  it("returns the stored arrays after styling", () => {
    const vm = new VMobject();
    vm.set_points([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    vm.set_fill("RED", 0.5);
    const fill = rowsOf(vm.get_fill_rgbas());
    expect(fill).toHaveLength(4);
    for (const row of fill) expect(row[3]).toBeCloseTo(0.5, 12);
    const stroke = rowsOf(vm.get_stroke_rgbas());
    expect(stroke).toHaveLength(4);
  });

  it("falls back to zeros((1,4)) when the attribute is missing (python AttributeError path)", () => {
    const vm = new VMobject();
    (vm as unknown as Record<string, unknown>).fill_rgbas = undefined;
    (vm as unknown as Record<string, unknown>).stroke_rgbas = undefined;
    (vm as unknown as Record<string, unknown>).background_stroke_rgbas = undefined;
    expect(rowsOf(vm.get_fill_rgbas())).toEqual([[0, 0, 0, 0]]);
    expect(rowsOf(vm.get_stroke_rgbas())).toEqual([[0, 0, 0, 0]]);
    expect(rowsOf(vm.get_stroke_rgbas(true))).toEqual([[0, 0, 0, 0]]);
  });

  it("background flag selects background_stroke_rgbas (python: [[0,0,0,1]])", () => {
    const vm = new VMobject();
    vm.set_points([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    // NOTE: python only resizes rgbas arrays on set_fill/set_stroke calls, so
    // without styling the background array keeps its 1 init row; the TS port
    // matches that shape. background_stroke_opacity defaults to 1.0 in both.
    expect(rowsOf(vm.get_stroke_rgbas(true))).toEqual([[0, 0, 0, 1]]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.color.core.RandomColorGenerator
// ---------------------------------------------------------------------------
describe("RandomColorGenerator", () => {
  it("reproduces python's seeded sequence (seed 42)", () => {
    const rnd = new RandomColorGenerator(42);
    expect(rnd.next().toHex()).toBe("#8B4513");
    expect(rnd.next().toHex()).toBe("#BBBBBB");
    expect(rnd.next().toHex()).toBe("#BBBBBB");
  });

  it("is reproducible across instances with the same seed", () => {
    const a = new RandomColorGenerator(42);
    const b = new RandomColorGenerator(42);
    expect(a.next().toHex()).toBe(b.next().toHex());
    expect(a.next().toHex()).toBe(b.next().toHex());
  });

  it("samples from the full 89-color palette by default", () => {
    expect(new RandomColorGenerator(1).colors).toHaveLength(89);
  });

  it("stays within a custom palette", () => {
    const palette = [new ManimColor("RED"), new ManimColor("GREEN"), new ManimColor("BLUE")];
    const hexes = new Set(palette.map((c) => c.toHex()));
    const rnd = new RandomColorGenerator(1, palette);
    for (let i = 0; i < 10; i += 1) {
      expect(hexes.has(rnd.next().toHex())).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig.__init__
// ---------------------------------------------------------------------------
describe("ManimConfig.__init__", () => {
  it("initializes every option to null (python: dict.fromkeys(_OPTS) -> None)", () => {
    const cfg = new ManimConfig();
    expect(cfg.get("frame_rate")).toBeNull();
    expect(cfg.get("pixel_width")).toBeNull();
    expect(cfg.get("background_opacity")).toBeNull();
    expect(cfg.get("format")).toBeNull();
  });

  it("raises AttributeError for unknown keys like python", () => {
    const cfg = new ManimConfig();
    expectErrorName(() => cfg.get("nope"), "AttributeError");
  });
});

// ---------------------------------------------------------------------------
// manim.scene.section.Section (+ __init__)
// ---------------------------------------------------------------------------
describe("Section", () => {
  it("stores constructor fields and starts with no partial files", () => {
    const sec = new Section("default.normal", "vid.mp4", "mysec", false);
    expect(sec.type_).toBe("default.normal");
    expect(sec.video).toBe("vid.mp4");
    expect(sec.name).toBe("mysec");
    expect(sec.skip_animations).toBe(false);
    expect(sec.partial_movie_files).toEqual([]);
    expect(sec.is_empty()).toBe(true);
    expect(sec.toRepr()).toBe("<Section 'mysec' stored in 'vid.mp4'>");
  });

  it("filters None entries from partial movie files (python ground truth)", () => {
    const sec = new Section("default.normal", "vid.mp4", "mysec", false);
    sec.partial_movie_files.push(null);
    sec.partial_movie_files.push("a.mp4");
    expect(sec.get_clean_partial_movie_files()).toEqual(["a.mp4"]);
    expect(sec.is_empty()).toBe(false);
  });

  it("get_dict raises ValueError without a video path", () => {
    expectErrorName(() => new Section("t", null, "n", false).get_dict("/tmp"), "ValueError");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.file_ops
// ---------------------------------------------------------------------------
describe("file_ops", () => {
  it("guarantee_existence creates nested dirs and returns an existing path", () => {
    const base = mkdtempSync(join(tmpdir(), "wanim-"));
    const target = join(base, "a", "b");
    const resolved = guarantee_existence(target);
    expect(statSync(resolved).isDirectory()).toBe(true);
    // idempotent on existing dirs
    expect(guarantee_existence(target)).toBe(resolved);
  });

  it("add_extension_if_not_present mirrors python with_suffix behaviour", () => {
    expect(add_extension_if_not_present("file.mp4", ".mp4")).toBe("file.mp4");
    expect(add_extension_if_not_present("file", ".mp4")).toBe("file.mp4");
    expect(add_extension_if_not_present("file.txt", ".mp4")).toBe("file.txt.mp4");
  });

  it("modify_atime refreshes atime while keeping mtime", () => {
    const base = mkdtempSync(join(tmpdir(), "wanim-"));
    const file = join(base, "f.txt");
    writeFileSync(file, "x");
    const old = new Date("2001-01-01T00:00:00Z");
    utimesSync(file, old, old);
    modify_atime(file);
    const stat = statSync(file);
    // python: os.utime(path, (time.time(), mtime)) -> mtime preserved
    expect(stat.mtimeMs).toBe(old.getTime());
    expect(stat.atimeMs).toBeGreaterThan(old.getTime());
    expect(readFileSync(file, "utf-8")).toBe("x");
  });
});

// ---------------------------------------------------------------------------
// manim._config.logger_utils.JSONFormatter
// ---------------------------------------------------------------------------
describe("JSONFormatter", () => {
  it("formats records as {levelname, module, message} JSON (python ground truth)", () => {
    const fmt = new JSONFormatter();
    const parsed = JSON.parse(
      fmt.format({ levelname: "INFO", module: "mod", message: "hello %s", args: ["world"] }),
    );
    expect(parsed).toEqual({ levelname: "INFO", module: "mod", message: "hello <>" });
  });

  it("masks mapping args positionally like python's deepcopy+mask", () => {
    const fmt = new JSONFormatter();
    const parsed = JSON.parse(
      fmt.format({ levelname: "INFO", module: "m", message: "hi %(name)s", args: { name: "x" } }),
    );
    expect(parsed.message).toBe("hi <>");
  });

  it("leaves messages without args untouched and does not mutate the record", () => {
    const fmt = new JSONFormatter();
    const record = { levelname: "INFO", module: "m", message: "plain", args: ["a", "b"] as unknown[] };
    const parsed = JSON.parse(fmt.format(record));
    expect(parsed.message).toBe("plain");
    expect(record.args).toEqual(["a", "b"]);
  });
});

// ---------------------------------------------------------------------------
// scene: SceneInteractRerun(+__init__), RerunSceneHandler(+__init__), Scene
// ---------------------------------------------------------------------------
describe("SceneInteractRerun", () => {
  it("stores sender and kwargs (python: **kwargs)", () => {
    // NOTE: the TS port takes kwargs as a single object instead of **kwargs.
    const rr = new SceneInteractRerun("file", { from_animation_number: 3 });
    expect(rr.sender).toBe("file");
    expect(rr.kwargs).toEqual({ from_animation_number: 3 });
    expect(new SceneInteractRerun("gui").kwargs).toEqual({});
  });
});

describe("RerunSceneHandler", () => {
  it("queues a file rerun action on modification", () => {
    const queue: unknown[] = [];
    const handler = new RerunSceneHandler(queue as never);
    expect(handler.queue).toBe(queue);
    handler.on_modified({});
    expect(queue).toHaveLength(1);
    const action = queue[0] as SceneInteractRerun;
    expect(action).toBeInstanceOf(SceneInteractRerun);
    expect(action.sender).toBe("file");
  });
});

describe("Scene basics", () => {
  it("__str__ returns the class name", () => {
    expect(new Scene().toString()).toBe("Scene");
    class MyScene extends Scene {}
    expect(new MyScene().toString()).toBe("MyScene");
  });

  it("setup/construct/tear_down are no-ops returning undefined (python: None)", () => {
    const s = new Scene();
    expect(s.setup()).toBeUndefined();
    expect(s.construct()).toBeUndefined();
    expect(s.tear_down()).toBeUndefined();
  });

  it("update_self runs every scene updater with dt", () => {
    const s = new Scene();
    const calls: number[] = [];
    s.updaters.push((dt: number) => calls.push(dt));
    s.updaters.push((dt: number) => calls.push(dt * 2));
    s.update_self(0.5);
    expect(calls).toEqual([0.5, 1.0]);
  });

  it("validate_run_time passes valid run times through", () => {
    function construct(): void {}
    expect(Scene.validate_run_time(1.0, construct)).toBe(1.0);
  });

  it("validate_run_time raises ValueError for non-positive run times", () => {
    function construct(): void {}
    expectErrorName(() => Scene.validate_run_time(0, construct), "ValueError");
    expectErrorName(() => Scene.validate_run_time(-1.0, construct), "ValueError");
    try {
      Scene.validate_run_time(0, construct);
      throw new Error("unreachable");
    } catch (e) {
      expect((e as Error).message).toContain("Scene.construct()");
      expect((e as Error).message).toContain("<= 0");
    }
  });

  it("validate_run_time clamps sub-frame run times with a warning", () => {
    function construct(): void {}
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      // default frame_rate 60 -> shortest possible run time is 1/60
      expect(Scene.validate_run_time(1 / 120, construct)).toBeCloseTo(1 / 60, 15);
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });

  it("get_time_progression yields arange(0, run_time, 1/fps) (python: 60 steps for 1s@60fps)", () => {
    const s = new Scene();
    const prog = s.get_time_progression(1.0, "test");
    expect(prog.times).toHaveLength(60);
    expect(prog.times[0]).toBe(0);
    expect(prog.times[59]).toBeCloseTo(59 / 60, 12);
    expect(prog.description).toBe("test");
  });

  it("get_time_progression collapses to [run_time] when skipping animations", () => {
    const s = new Scene();
    s.renderer = { skip_animations: true };
    expect([...s.get_time_progression(2.5, "x")]).toEqual([2.5]);
  });
});

// ---------------------------------------------------------------------------
// Mobject: stretch.func, apply_function.multi_mapping_function,
// align_points_with_larger, null_point_align, get_num_points
// ---------------------------------------------------------------------------
describe("Mobject point ops", () => {
  it("get_num_points counts rows (python: len(self.points))", () => {
    const m = new Mobject();
    expect(m.get_num_points()).toBe(0);
    m.set_points([[0, 0, 0], [1, 0, 0]]);
    expect(m.get_num_points()).toBe(2);
  });

  it("stretch.func scales one dim about the center (python ground truth)", () => {
    const m = new Mobject();
    m.set_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    m.stretch(2.0, 0);
    expect(rowsOf(m.points)).toEqual([[-0.5, 2.0, 3.0], [5.5, 5.0, 6.0]]);
  });

  it("apply_function maps row-wise like np.apply_along_axis (python ground truth)", () => {
    const m = new Mobject();
    m.set_points([[1.0, 1.0, 0.0]]);
    m.apply_function((p) => [p[0]! * 2, p[1]! * 2, p[2]! * 2]);
    expect(rowsOf(m.points)).toEqual([[2.0, 2.0, 0.0]]);
  });

  it("align_points_with_larger raises NotImplementedError on the base class", () => {
    expectErrorName(() => new Mobject().align_points_with_larger(new Mobject()), "NotImplementedError");
  });

  it("null_point_align pushes the pointed mobject into submobjects (python: [0, 1])", () => {
    const a = new Mobject();
    const b = new Mobject();
    b.set_points([[0, 0, 0]]);
    a.null_point_align(b);
    expect(a.submobjects).toHaveLength(0);
    expect(b.submobjects).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// AbstractImageMobject.set_resampling_algorithm
// ---------------------------------------------------------------------------
describe("set_resampling_algorithm", () => {
  it("accepts the ported RESAMPLING_ALGORITHMS values (python ints 0/2/3)", () => {
    expect(RESAMPLING_ALGORITHMS).toMatchObject({
      nearest: 0, none: 0, bilinear: 2, linear: 2, bicubic: 3, cubic: 3,
    });
    const m = new AbstractImageMobject({ scale_to_resolution: 1080 });
    expect(m.set_resampling_algorithm(2)).toBe(m);
    expect(m.resampling_algorithm).toBe(2);
    m.set_resampling_algorithm(0);
    expect(m.resampling_algorithm).toBe(0);
  });

  it("raises ValueError for unknown algorithms", () => {
    const m = new AbstractImageMobject({ scale_to_resolution: 1080 });
    expectErrorName(() => m.set_resampling_algorithm(1), "ValueError");
    expectErrorName(() => m.set_resampling_algorithm(99), "ValueError");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.resize_array
// ---------------------------------------------------------------------------
describe("resize_array", () => {
  it("cycles rows to extend and truncates to shrink (python ground truth)", () => {
    expect(rowsOf(resize_array(array([[1, 2], [3, 4]]), 3))).toEqual([[1, 2], [3, 4], [1, 2]]);
    expect(rowsOf(resize_array(array([[1, 2], [3, 4]]), 1))).toEqual([[1, 2]]);
    expect(rowsOf(resize_array(array([[1, 2], [3, 4]]), 2))).toEqual([[1, 2], [3, 4]]);
    expect(flat(resize_array(array([1, 2, 3]), 2))).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// VMobject anchors / curves / background image
// ---------------------------------------------------------------------------
describe("VMobject curve queries", () => {
  function eightPointVm(): VMobject {
    const vm = new VMobject();
    vm.set_points([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0]]);
    return vm;
  }

  it("get_num_curves divides point count by nppcc (python ground truth: 2 and 0)", () => {
    expect(eightPointVm().get_num_curves()).toBe(2);
    expect(new VMobject().get_num_curves()).toBe(0);
  });

  it("get_start_anchors / get_end_anchors slice every 4th point", () => {
    expect(eightPointVm().get_start_anchors()).toEqual([[0, 0, 0], [4, 0, 0]]);
    expect(eightPointVm().get_end_anchors()).toEqual([[3, 0, 0], [7, 0, 0]]);
  });

  it("get_background_image defaults to null (python: None)", () => {
    expect(new VMobject().get_background_image()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// ThreeDAxes._set_axis_shading.make_func
// NOTE: Python's ThreeDAxes.__init__ ends with
//   `if config.renderer == RendererType.CAIRO:
//        self._add_3d_pieces(); self._set_axis_shading()`
// so under the default cairo renderer the constructor already binds the
// gradient/normal/sheen once; an explicit _set_axis_shading() call below
// re-binds (idempotent), hence set_sheen is seen twice.
// ---------------------------------------------------------------------------
describe("ThreeDAxes._set_axis_shading", () => {
  it("binds gradient endpoints to +/- light_source edge centers", () => {
    // Python default light_source is 9*DOWN + 7*LEFT + 10*OUT == [-7, -9, 10]
    const axes = new ThreeDAxes();
    const xAxis = axes.get_x_axis() as unknown as VMobject;
    const fam = xAxis.family_members_with_points()[0]!;
    // The constructor already applied shading once (see note above): the
    // original manim gradient for the default light_source is
    // [[5.25, 0.175, 0], [-5.25, -0.175, 0]].
    expect(fam.get_sheen_factor()).toBe(0.2);
    expect(
      (fam.get_gradient_start_and_end_points() as number[][]).map((p) => p.map((v) => Math.round(v * 1e6) / 1e6)),
    ).toEqual([[5.25, 0.175, 0], [-5.25, -0.175, 0]]);
    // Explicit re-bind is idempotent.
    axes._set_axis_shading();
    expect(fam.get_sheen_factor()).toBe(0.2);
    expect(fam.get_unit_normal(null)).toEqual([1, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// graphing scales: _ScaleBase(+__init__/function/inverse) and LinearBase
// ---------------------------------------------------------------------------
describe("scales", () => {
  it("_ScaleBase stores custom_labels and raises NotImplementedError", () => {
    expect(new _ScaleBase().custom_labels).toBe(false);
    expect(new _ScaleBase(true).custom_labels).toBe(true);
    const base = new _ScaleBase();
    expectErrorName(() => base.function(1.0), "NotImplementedError");
    expectErrorName(() => base.inverse_function(1.0), "NotImplementedError");
  });

  it("LinearBase scales by scale_factor (python: 2.0 * 3.0 == 6.0, 6.0 / 2.0 == 3.0)", () => {
    const lb = new LinearBase(2.0);
    expect(lb.function(3.0)).toBe(6.0);
    expect(lb.inverse_function(6.0)).toBe(3.0);
    expect(new LinearBase().function(5)).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// ArrowTip.__init__ / TipableVMobject.get_default_tip_length
// ---------------------------------------------------------------------------
describe("tips", () => {
  it("ArrowTip.__init__ raises NotImplementedError", () => {
    expectErrorName(() => new ArrowTip(), "NotImplementedError");
  });

  it("get_default_tip_length returns tip_length (python default 0.35)", () => {
    expect(new TipableVMobject({ tip_length: 0.35 }).get_default_tip_length()).toBe(0.35);
    expect(new TipableVMobject().get_default_tip_length()).toBe(0.35);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops: cross, cartesian_to_spherical, norm_squared, normalize
// ---------------------------------------------------------------------------
describe("space_ops", () => {
  it("cross matches numpy (python: [0,0,1] and [-3,6,-3])", () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(cross([1, 2, 3], [4, 5, 6])).toEqual([-3, 6, -3]);
  });

  it("cartesian_to_spherical matches python (r, theta, phi)", () => {
    expectClose(cartesian_to_spherical([1, 0, 0]), [1.0, 0.0, 1.5707963267948966]);
    expectClose(cartesian_to_spherical([1, 1, 1]), [1.7320508075688772, 0.7853981633974483, 0.9553166181245092]);
    expect(cartesian_to_spherical([0, 0, 0])).toEqual([0, 0, 0]);
    expectClose(cartesian_to_spherical([0, 0, 2]), [2.0, 0.0, 0.0]);
  });

  it("norm_squared dots a vector with itself", () => {
    expect(norm_squared([3, 4])).toBe(25);
    expect(norm_squared([0, 0, 0])).toBe(0);
  });

  it("normalize divides by the norm, zeros stay zeros", () => {
    expectClose(normalize([3, 4, 0]), [0.6, 0.8, 0.0]);
    expect(normalize([0, 0, 0])).toEqual([0, 0, 0]);
    expect(normalize([0, 0], [1, 0])).toEqual([1, 0]);
    // NOTE: python `fall_back or np.zeros(...)` raises ValueError for an
    // ndarray fallback (ambiguous truth value); the TS port sensibly returns
    // a copy of the fallback instead.
    expect(normalize([0, 0, 0], null)).toEqual([0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// NumberLine._decimal_places_from_step / Axes._origin_shift
// ---------------------------------------------------------------------------
describe("number line / axes statics", () => {
  it("_decimal_places_from_step counts str() decimals (python: [0,1,2,1,1,2])", () => {
    expect(NumberLine._decimal_places_from_step(1)).toBe(0);
    expect(NumberLine._decimal_places_from_step(0.5)).toBe(1);
    expect(NumberLine._decimal_places_from_step(0.25)).toBe(2);
    expect(NumberLine._decimal_places_from_step(0.1)).toBe(1);
    expect(NumberLine._decimal_places_from_step(2.5)).toBe(1);
    expect(NumberLine._decimal_places_from_step(0.01)).toBe(2);
    expect(NumberLine._decimal_places_from_step(0.30000000000000004)).toBe(17);
    expect(NumberLine._decimal_places_from_step(1e-7)).toBe(0);
    // python str() uses scientific notation below 1e-4: '1e-05' has no dot
    expect(NumberLine._decimal_places_from_step(1e-5)).toBe(0);
    // ...but a mantissa dot still counts: str(1.5e-5) == '1.5e-05' -> '5e-05'
    expect(NumberLine._decimal_places_from_step(1.5e-5)).toBe(5);
    // NOTE: JS cannot distinguish int 2 from float 2.0 (python str(2.0) ==
    // '2.0' -> 1); integer-valued inputs keep the int behaviour (0).
  });

  it("Axes._origin_shift compensates for axes missing zero (python: [1,-1,0,0])", () => {
    expect(Axes._origin_shift([1, 5, 1])).toBe(1);
    expect(Axes._origin_shift([-5, -1, 1])).toBe(-1);
    expect(Axes._origin_shift([-5, 5, 1])).toBe(0);
    expect(Axes._origin_shift([0, 5, 1])).toBe(0);
  });

  it("CoordinateSystem._origin_shift raises NotImplementedError", () => {
    expectErrorName(() => CoordinateSystem._origin_shift([0, 1, 1]), "NotImplementedError");
  });
});

// ---------------------------------------------------------------------------
// three_d: Sphere.func / Surface._get_u_values_and_v_values
// ---------------------------------------------------------------------------
describe("three_d surfaces", () => {
  it("Sphere.func matches radius-scaled spherical coords (python ground truth)", () => {
    const s = new Sphere([0, 0, 0], 2.0, [4, 3]);
    expectClose(s.func(0.5, 1.0), [1.4769205252082576, 0.8068453602226698, -1.0806046117362795]);
    expectClose(s.func(0, 0), [0.0, 0.0, -2.0]);
  });

  it("Surface._get_u_values_and_v_values linspaces the ranges (python ground truth)", () => {
    const s = new Sphere([0, 0, 0], 2.0, [4, 3]);
    const [u, v] = s._get_u_values_and_v_values();
    expect(u).toHaveLength(5);
    expect(v).toHaveLength(4);
    for (let i = 0; i < u.length; i += 1) {
      expect(u[i]).toBeCloseTo([0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586][i]!, 12);
    }
    for (let i = 0; i < v.length; i += 1) {
      expect(v[i]).toBeCloseTo([0.0, 1.0471975511965976, 2.0943951023931953, 3.141592653589793][i]!, 12);
    }
    const plane = new Surface((uu, vv) => array([uu, vv, 0]), [0, 2], [-1, 1], [2, 4]);
    const [su, sv] = plane._get_u_values_and_v_values();
    expect(su).toEqual([0.0, 1.0, 2.0]);
    expect(sv).toEqual([-1.0, -0.5, 0.0, 0.5, 1.0]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.exceptions
// ---------------------------------------------------------------------------
describe("exceptions", () => {
  it("both scene-control exceptions are Errors", () => {
    expect(new EndSceneEarlyException("x")).toBeInstanceOf(Error);
    expect(new EndSceneEarlyException("x").name).toBe("EndSceneEarlyException");
    expect(new RerunSceneException("y")).toBeInstanceOf(Error);
    expect(new RerunSceneException("y").name).toBe("RerunSceneException");
  });
});
