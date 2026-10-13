import { describe, expect, it, vi, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { array } from "numpy-ts";
import {
  quaternion_conjugate,
  quaternion_mult,
  R3_to_complex,
  complex_to_R3,
  center_of_mass,
  angle_of_vector,
  shoelace,
} from "../src/utils/space_ops.js";
import { lingering } from "../src/utils/rate_functions.js";
import type { RateFunction } from "../src/utils/rate_functions.js";
import {
  _canonical_field_title,
  _Memoizer,
  _CustomEncoder,
} from "../src/utils/hash.js";
import {
  merge_dicts_recursively,
  _Data,
  _Uniforms,
} from "../src/utils/config_ops.js";
import type { _HasData, _HasUniforms } from "../src/utils/config_ops.js";
import {
  QuickHullPoint,
  SubFacet,
  Facet,
  Horizon,
} from "../src/utils/qhull.js";
import {
  get_video_metadata,
} from "../src/utils/commands.js";
import type { VideoMetadata } from "../src/utils/commands.js";
import { Mobject } from "../src/mobject/mobject.js";
import { Scene } from "../src/scene/scene.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Animation } from "../src/animation/animation.js";
import { flatten_iterable_parameters } from "../src/utils/parameter_parsing.js";
import {
  is_mp4_format,
  is_gif_format,
  is_mov_format,
  is_png_format,
  is_webm_format,
  add_version_before_extension,
  add_import_statement,
  open_file,
} from "../src/utils/file_ops.js";
import { Camera } from "../src/camera/camera.js";
import { get_module } from "../src/utils/module_ops.js";
import { _typst_hash } from "../src/utils/typst_file_writing.js";
import { make_even, tuplify } from "../src/utils/iterables.js";
import {
  tex_hash,
  delete_nonsvg_files,
} from "../src/utils/tex_file_writing.js";
import { ManimConfig, config } from "../src/_config/utils.js";
import { RendererType } from "../src/constants.js";

// NOTE: TexTemplate.from_file is covered by tests/batch-layer0-tex.test.ts.
// tex_hash / delete_nonsvg_files are pure file utilities (no LaTeX needed)
// and are covered below; Tex/Text mobjects remain out of scope (see
// src/mobject/text.ts).

function nums(value: unknown): number[] {
  const raw = (value as { toArray: () => unknown }).toArray();
  const out: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) {
      for (const e of x) walk(e);
    } else if (
      x !== null &&
      typeof x === "object" &&
      "toArray" in (x as object) &&
      typeof (x as { toArray?: unknown }).toArray === "function"
    ) {
      walk((x as { toArray(): unknown }).toArray());
    } else {
      out.push(Number(x));
    }
  };
  walk(raw);
  return out;
}

function expectCloseNums(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = nums(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]!) * relTol);
    expect(Math.abs(values[i]! - expected[i]!)).toBeLessThanOrEqual(tol);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.RateFunction (Protocol / type alias)
// Python: class RateFunction(Protocol) with __call__(t, *args, **kwargs)->float.
// The TS port is `type RateFunction = (t, ...args)=>number`. Test conformance
// plus lingering ground truths (lingering wraps squish(identity,0,0.8)).
// Python: [lingering(t) for t in [0,0.2,0.4,0.8,0.9,1.0,-1,2]]
//   == [0.0,0.25,0.5,1.0,1.0,1.0,0,1]
// ---------------------------------------------------------------------------
describe("RateFunction", () => {
  it("accepts a conforming function", () => {
    const f: RateFunction = (t: number) => t;
    expect(f(0.5)).toBe(0.5);
  });
  it("lingering matches python ground truths", () => {
    expect(lingering(0)).toBe(0);
    expect(lingering(0.2)).toBeCloseTo(0.25, 12);
    expect(lingering(0.4)).toBeCloseTo(0.5, 12);
    expect(lingering(0.8)).toBe(1);
    expect(lingering(0.9)).toBe(1);
    expect(lingering(1.0)).toBe(1);
    expect(lingering(-1)).toBe(0);
    expect(lingering(2)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.lingering.identity (nested closure)
// Python identity(t)=t; lingering == squish(identity,0,0.8) under unit_interval,
// i.e. lingering(t) == min(max(t/0.8,0),1) clamped. Covered above; here assert
// the 0.8*knee explicitly (0.8 -> 1.0, just past -> 1).
// ---------------------------------------------------------------------------
describe("lingering.identity", () => {
  it("behaves as t/0.8 clamped to [0,1]", () => {
    expect(lingering(0.79)).toBeCloseTo(0.79 / 0.8, 12);
    expect(lingering(0.8)).toBe(1);
    expect(lingering(0.8001)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.quaternion_conjugate
// Python: [1,-2,-3,-4] for [1,2,3,4] (copies input, negates [1:]).
// ---------------------------------------------------------------------------
describe("quaternion_conjugate", () => {
  it("negates the vector part", () => {
    expect(quaternion_conjugate([1, 2, 3, 4])).toEqual([1, -2, -3, -4]);
  });
  it("leaves the scalar part for a 1-element input", () => {
    expect(quaternion_conjugate([5])).toEqual([5]);
  });
  it("does not mutate the input", () => {
    const q = [1, 2, 3, 4];
    quaternion_conjugate(q);
    expect(q).toEqual([1, 2, 3, 4]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.quaternion_mult
// Python: () -> [1,0,0,0]; ([1,0,0,0],[0,1,0,0]) -> [0,1,0,0];
//         ([1,2,3,4],[5,6,7,8]) -> [-60,12,30,24].
// ---------------------------------------------------------------------------
describe("quaternion_mult", () => {
  it("returns the identity for no inputs", () => {
    expect(quaternion_mult()).toEqual([1, 0, 0, 0]);
  });
  it("multiplies identity by i", () => {
    expect(quaternion_mult([1, 0, 0, 0], [0, 1, 0, 0])).toEqual([0, 1, 0, 0]);
  });
  it("matches the Hamilton product ground truth", () => {
    expect(quaternion_mult([1, 2, 3, 4], [5, 6, 7, 8])).toEqual([-60, 12, 30, 24]);
  });
  it("returns a single quaternion unchanged", () => {
    expect(quaternion_mult([1, 2, 3, 4])).toEqual([1, 2, 3, 4]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.R3_to_complex / complex_to_R3
// Python: R3_to_complex([3,4,5]) == (3+4j); complex_to_R3(3+4j) == [3,4,0].
// TS represents complex as {re,im}; complex_to_R3 also accepts [re,im]/number.
// ---------------------------------------------------------------------------
describe("R3_to_complex", () => {
  it("takes the first two components", () => {
    expect(R3_to_complex([3, 4, 5])).toEqual({ re: 3, im: 4 });
  });
  it("defaults missing components to 0", () => {
    expect(R3_to_complex([7])).toEqual({ re: 7, im: 0 });
  });
});

describe("complex_to_R3", () => {
  it("converts {re,im} to [re,im,0]", () => {
    expect(complex_to_R3({ re: 3, im: 4 })).toEqual([3, 4, 0]);
  });
  it("converts pairs and scalars", () => {
    expect(complex_to_R3([3, 4])).toEqual([3, 4, 0]);
    expect(complex_to_R3(5)).toEqual([5, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.center_of_mass
// Python: mean of [[0,0,0],[2,2,2]] == [1,1,1]; single [[1,2,3]] -> [1,2,3].
// ---------------------------------------------------------------------------
describe("center_of_mass", () => {
  it("averages rows (NDArray input)", () => {
    const out = center_of_mass(array([[0, 0, 0], [2, 2, 2]]));
    expect(out.shape[0]).toBe(3);
    for (const v of out.toArray() as number[]) expect(v).toBeCloseTo(1, 12);
  });
  it("returns the single point for one row", () => {
    const out = center_of_mass(array([[1, 2, 3]]));
    const values = out.toArray() as number[];
    expect(values[0]).toBeCloseTo(1, 12);
    expect(values[1]).toBeCloseTo(2, 12);
    expect(values[2]).toBeCloseTo(3, 12);
  });
  it("supports plain row lists", () => {
    expect(center_of_mass([[0, 0, 0], [2, 2, 2]])).toEqual([1, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.angle_of_vector
// Python: [1,0,0]->0; [0,1,0]->pi/2; [-1,0,0]->pi; [1,1,0]->pi/4;
//         [[1,0],[0,1]] (shape (2,2)) -> [0, pi/2].
// ---------------------------------------------------------------------------
describe("angle_of_vector", () => {
  it("matches cardinal angles", () => {
    expect(angle_of_vector([1, 0, 0])).toBeCloseTo(0, 12);
    expect(angle_of_vector([0, 1, 0])).toBeCloseTo(Math.PI / 2, 12);
    expect(angle_of_vector([-1, 0, 0])).toBeCloseTo(Math.PI, 12);
    expect(angle_of_vector([1, 1, 0])).toBeCloseTo(Math.PI / 4, 12);
  });
  it("handles (2,n) arrays column-wise", () => {
    const out = angle_of_vector(array([[1, 0], [0, 1]]));
    expect((out as { shape: number[] }).shape[0]).toBe(2);
    const values = (out as { toArray(): number[] }).toArray();
    expect(values[0]).toBeCloseTo(0, 12);
    expect(values[1]).toBeCloseTo(Math.PI / 2, 12);
  });
  it("throws for degenerate (1,n) input", () => {
    expect(() => angle_of_vector(array([[1, 2, 3]]))).toThrow(
      "Vector must have the correct dimensions. (2, n)",
    );
  });
  it("throws TypeError for nested plain lists (Python complex() fails)", () => {
    expect(() => angle_of_vector([[1, 0], [0, 1]])).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.shoelace
// Python: square [[0,0],[4,0],[4,4],[0,4]] -> -16.0;
//         triangle [[0,0],[1,0],[0,1]] -> -0.5 (trapezoid sign convention).
// ---------------------------------------------------------------------------
describe("shoelace", () => {
  it("matches the signed square area", () => {
    expect(shoelace(array([[0, 0], [4, 0], [4, 4], [0, 4]]))).toBeCloseTo(-16, 12);
  });
  it("matches the signed triangle area", () => {
    expect(shoelace(array([[0, 0], [1, 0], [0, 1]]))).toBeCloseTo(-0.5, 12);
  });
  it("supports plain lists", () => {
    expect(shoelace([[0, 0], [4, 0], [4, 4], [0, 4]])).toBeCloseTo(-16, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.animation.animation.Animation.run_time (property)
// Python: default 1.0; settable; negative raises ValueError.
// ---------------------------------------------------------------------------
describe("Animation.run_time", () => {
  it("defaults to 1.0", () => {
    expect(new Animation(new Mobject()).run_time).toBe(1);
  });
  it("stores positive values", () => {
    const a = new Animation(new Mobject());
    a.run_time = 2.5;
    expect(a.run_time).toBe(2.5);
  });
  it("rejects negative values", () => {
    const a = new Animation(new Mobject());
    expect(() => {
      a.run_time = -1;
    }).toThrow("cannot be negative");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing._canonical_field_title
// Python: None -> None; 5 -> ['builtins.int','5']; 'hi' -> ['builtins.str',"'hi'"].
// NOTE: for class instances Python reports the real module (__main__.Foo);
// the TS port always reports builtins.<CtorName> (documented deviation).
// ---------------------------------------------------------------------------
describe("_canonical_field_title", () => {
  it("maps None to None", () => {
    expect(_canonical_field_title(null)).toBeNull();
    expect(_canonical_field_title(undefined)).toBeNull();
  });
  it("maps ints and strings", () => {
    expect(_canonical_field_title(5)).toEqual(["builtins.int", "5"]);
    expect(_canonical_field_title(5.5)).toEqual(["builtins.float", "5.5"]);
    expect(_canonical_field_title("hi")).toEqual(["builtins.str", "'hi'"]);
    expect(_canonical_field_title(true)).toEqual(["builtins.bool", "True"]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing._Memoizer / _Memoizer.__init__
// Python: fresh _already_processed set + _keep_alive dict, both empty;
// check_already_processed returns obj first time, 'AP' second time for objects;
// primitives pass through untouched.
// ---------------------------------------------------------------------------
describe("_Memoizer.__init__", () => {
  it("starts with empty processed/keep-alive stores", () => {
    const m = new _Memoizer();
    expect((m as unknown as { _already_processed: Set<string> })._already_processed.size).toBe(0);
    expect((m as unknown as { _keep_alive: Map<number, unknown> })._keep_alive.size).toBe(0);
  });
});

describe("_Memoizer", () => {
  it("returns objects once, then the placeholder", () => {
    const m = new _Memoizer();
    const o = [1, 2];
    expect(m.check_already_processed(o)).toBe(o);
    expect(m.check_already_processed(o)).toBe("AP");
  });
  it("passes primitives through", () => {
    const m = new _Memoizer();
    expect(m.check_already_processed(5)).toBe(5);
    expect(m.check_already_processed("x")).toBe("x");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing._CustomEncoder
// Python is a json.JSONEncoder subclass with memoizer + include_pixel_array;
// TS is a standalone helper with default()/_cleaned_iterable()/encode().
// Exact ndarray payloads differ (Python emits NDARRAY:<hash>, TS emits the
// repr for small arrays) so only structural paths are asserted here.
// ---------------------------------------------------------------------------
describe("_CustomEncoder", () => {
  it("stores the memoizer and flag", () => {
    const m = new _Memoizer();
    const enc = new _CustomEncoder(m, true);
    expect(enc._memoizer).toBe(m);
    expect(enc.include_pixel_array).toBe(true);
    expect(new _CustomEncoder(m).include_pixel_array).toBe(false);
  });
  it("encodes uint8-range ints as ints", () => {
    const enc = new _CustomEncoder(new _Memoizer());
    expect(enc.default(7)).toBe(7);
  });
  it("cleans plain objects via __dict__-like entries", () => {
    const enc = new _CustomEncoder(new _Memoizer());
    expect(enc.default({ x: 1 })).toEqual({ x: 1 });
  });
  it("filters KEYS_TO_FILTER_OUT unless pixel arrays are included", () => {
    const enc = new _CustomEncoder(new _Memoizer());
    expect(enc.default({ original_id: "a", x: 1 })).toEqual({ x: 1 });
    const withPixels = new _CustomEncoder(new _Memoizer(), true);
    expect(withPixels.default({ pixel_array: [1], x: 1 })).toEqual({ pixel_array: [1], x: 1 });
  });
});

// ---------------------------------------------------------------------------
// manim.utils.config_ops.merge_dicts_recursively
// Python: {'a':1,'n':{'x':1,'y':2}} + {'n':{'y':3,'z':4},'b':2}
//   -> {'a':1,'n':{'x':1,'y':3,'z':4},'b':2} (later dicts win, recurse).
// ---------------------------------------------------------------------------
describe("merge_dicts_recursively", () => {
  it("merges nested dicts with later-wins priority", () => {
    expect(
      merge_dicts_recursively({ a: 1, n: { x: 1, y: 2 } }, { n: { y: 3, z: 4 }, b: 2 }),
    ).toEqual({ a: 1, n: { x: 1, y: 3, z: 4 }, b: 2 });
  });
  it("returns {} for no inputs and overwrites scalars", () => {
    expect(merge_dicts_recursively()).toEqual({});
    expect(merge_dicts_recursively({ a: 1 }, { a: 2 })).toEqual({ a: 2 });
  });
});

// ---------------------------------------------------------------------------
// manim.utils.config_ops._HasData / _HasUniforms (Protocols)
// Python: _HasData{data: dict}, _HasUniforms{uniforms: dict}; exercised via
// the _Data/_Uniforms descriptors reading obj.data[name]/obj.uniforms[name].
// ---------------------------------------------------------------------------
describe("_HasData", () => {
  it("works with the _Data descriptor", () => {
    const d = new _Data<number>();
    d.__set_name__({}, "v");
    const obj: _HasData = { data: { v: 3 } };
    expect(d.__get__(obj)).toBe(3);
    d.__set__(obj, 4);
    expect(obj.data["v"]).toBe(4);
  });
});

describe("_HasUniforms", () => {
  it("works with the _Uniforms descriptor", () => {
    const u = new _Uniforms<number>();
    u.__set_name__({}, "w");
    const obj: _HasUniforms = { uniforms: { w: 1 } };
    expect(u.__get__(obj)).toBe(1);
    u.__set__(obj, 2);
    expect(obj.uniforms["w"]).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.QuickHullPoint / __init__
// Python stores the ndarray coordinates as-is; TS stores a numeric copy.
// ---------------------------------------------------------------------------
describe("QuickHullPoint.__init__", () => {
  it("copies coordinates", () => {
    const src = [1, 2, 3];
    const p = new QuickHullPoint(src);
    expect(p.coordinates).toEqual([1, 2, 3]);
    src[0] = 9;
    expect(p.coordinates[0]).toBe(1);
  });
});

describe("QuickHullPoint", () => {
  it("round-trips NDArray coordinates", () => {
    expect(new QuickHullPoint(array([1, 2, 3])).coordinates).toEqual([1, 2, 3]);
  });
  it("equals compares by value, rejects other types", () => {
    expect(new QuickHullPoint([1, 2]).equals(new QuickHullPoint([1, 2]))).toBe(true);
    expect(new QuickHullPoint([1, 2]).equals(new QuickHullPoint([1, 3]))).toBe(false);
    expect(() => new QuickHullPoint([1]).equals("x")).toThrow();
  });
  it("has a deterministic key", () => {
    expect(new QuickHullPoint([1, 2]).key()).toBe(new QuickHullPoint([1, 2]).key());
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.Facet.compute_normal
// Python triangle [[0,0,0],[1,0,0],[0,1,0]]: center [1/3,1/3,0];
// internal [0.1,0.1,1] (above) -> normal [0,0,-1];
// internal [0.1,0.1,-1] (below) -> [0,0,1]; unit length.
// ---------------------------------------------------------------------------
describe("Facet.compute_normal", () => {
  it("points away from an internal point above the facet", () => {
    const f = new Facet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.1, 0.1, 1.0]);
    expect(f.center[0]).toBeCloseTo(1 / 3, 12);
    expect(f.center[1]).toBeCloseTo(1 / 3, 12);
    expect(f.normal[2]).toBeCloseTo(-1, 9);
    expect(Math.hypot(...f.normal)).toBeCloseTo(1, 12);
  });
  it("flips when the internal point is below", () => {
    const f = new Facet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.1, 0.1, -1.0]);
    expect(f.normal[2]).toBeCloseTo(1, 9);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.SubFacet / Facet / Horizon
// Python: SubFacet wraps coordinates + frozenset of points, eq by point set;
// Facet has center/normal/subfacets (frozenset), eq by subfacets;
// Horizon() starts with empty facets set + empty boundary list.
// NOTE: TS adds addFacet/hasFacet helpers + key()/equals(); Python uses
// __hash__/__eq__ directly. Structural behaviour is asserted here.
// ---------------------------------------------------------------------------
describe("SubFacet", () => {
  it("compares by point set independent of order", () => {
    expect(new SubFacet([[0, 0], [1, 1]]).equals(new SubFacet([[1, 1], [0, 0]]))).toBe(true);
    expect(new SubFacet([[0, 0]]).equals(new SubFacet([[0, 1]]))).toBe(false);
    expect(() => new SubFacet([[0, 0]]).equals("x")).toThrow();
  });
});

describe("Facet", () => {
  it("builds one subfacet per vertex and compares by key", () => {
    const a = new Facet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.1, 0.1, 1.0]);
    expect(a.subfacets).toHaveLength(3);
    const b = new Facet([[0, 1, 0], [0, 0, 0], [1, 0, 0]], [0.1, 0.1, 1.0]);
    expect(a.equals(b)).toBe(true);
    expect(() => a.equals("x")).toThrow();
  });
});

describe("Horizon", () => {
  it("starts empty", () => {
    const h = new Horizon();
    expect(h.facets).toHaveLength(0);
    expect(h.boundary).toHaveLength(0);
  });
  it("tracks added facets", () => {
    const h = new Horizon();
    const f = new Facet([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.1, 0.1, 1.0]);
    h.addFacet(f);
    expect(h.hasFacet(f)).toBe(true);
    expect(h.facets).toHaveLength(1);
    h.addFacet(f);
    expect(h.facets).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.commands.VideoMetadata (TypedDict)
// Python fields: width,height,nb_frames,duration,avg_frame_rate,codec_name,
// pix_fmt. get_video_metadata needs libav and is a TODO stub in TS.
// ---------------------------------------------------------------------------
describe("VideoMetadata", () => {
  it("has the documented shape", () => {
    const meta: VideoMetadata = {
      width: 1920,
      height: 1080,
      nb_frames: "100",
      duration: "1.666667",
      avg_frame_rate: "60/1",
      codec_name: "h264",
      pix_fmt: "yuv420p",
    };
    expect(meta.width).toBe(1920);
    expect(meta.nb_frames).toBe("100");
  });
  it("get_video_metadata is an unported stub", () => {
    expect(() => get_video_metadata("x.mp4")).toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.set
// Python: mob.set(foo=1) sets attr and returns self; set(a=1,b=2) sets both.
// ---------------------------------------------------------------------------
describe("Mobject.set", () => {
  it("sets attributes and returns self", () => {
    const m = new Mobject();
    const out = m.set({ foo: 1 } as Record<string, unknown>);
    expect(out).toBe(m);
    expect((m as unknown as Record<string, unknown>)["foo"]).toBe(1);
  });
  it("sets several attributes at once", () => {
    const m = new Mobject();
    m.set({ a: 1, b: 2 } as Record<string, unknown>);
    expect((m as unknown as Record<string, unknown>)["a"]).toBe(1);
    expect((m as unknown as Record<string, unknown>)["b"]).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// manim.scene.scene.Scene.remove.lambda_function (inner filter for meshes)
// Python cairo path only restructures mobjects/foreground and leaves
// `meshes` untouched; only the opengl path filters meshes with
// `lambda_function(mesh) = mesh not in removed` (verified against the
// original: cairo Scene().remove(a) with meshes [a, b] keeps both).
// ---------------------------------------------------------------------------
describe("Scene.remove.lambda_function", () => {
  it("removes mobjects from the scene", () => {
    const s = new Scene();
    const m1 = new Mobject();
    const m2 = new Mobject();
    s.add(m1, m2);
    s.remove(m1);
    expect(s.mobjects.includes(m1)).toBe(false);
    expect(s.mobjects.includes(m2)).toBe(true);
  });
  it("leaves meshes untouched on the cairo path", () => {
    const s = new Scene();
    const a = new Mobject();
    const b = new Mobject();
    s.meshes = [a, b];
    s.remove(a);
    expect(s.meshes).toEqual([a, b]);
  });
  it("filters meshes not in the removal set on the opengl path", () => {
    const prev = config.renderer;
    config.renderer = RendererType.OPENGL;
    try {
      const s = new Scene();
      const a = new Mobject();
      const b = new Mobject();
      s.meshes = [a, b];
      s.remove(a);
      expect(s.meshes).toEqual([b]);
    } finally {
      config.renderer = prev;
    }
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.value_tracker.ValueTracker.set_value
// Python: self.points[0,0] = value; returns self.
// ---------------------------------------------------------------------------
describe("ValueTracker.set_value", () => {
  it("stores the value and returns self", () => {
    const vt = new ValueTracker(3.0);
    expect(vt.get_value()).toBeCloseTo(3, 12);
    expect(vt.set_value(5.0)).toBe(vt);
    expect(vt.get_value()).toBeCloseTo(5, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.parameter_parsing.flatten_iterable_parameters
// Python: [[1,2],3,(4,5)] -> [1,2,3,4,5]; [] -> []; ['ab',[1]] -> ['a','b',1]
// (strings are Iterable, so they are extended char-wise).
// ---------------------------------------------------------------------------
describe("flatten_iterable_parameters", () => {
  it("flattens one level", () => {
    expect(flatten_iterable_parameters([[1, 2], 3, [4, 5]])).toEqual([1, 2, 3, 4, 5]);
  });
  it("handles empty input", () => {
    expect(flatten_iterable_parameters([])).toEqual([]);
  });
  it("extends strings char-wise like Python", () => {
    expect(flatten_iterable_parameters(["ab", [1]] as unknown as Iterable<number | Iterable<number>>)).toEqual([
      "a",
      "b",
      1,
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.file_ops format predicates
// Python reads global config['format']; TS reads the shared config singleton.
// ---------------------------------------------------------------------------
describe("file_ops format predicates", () => {
  const prev = config.get("format");
  afterEach(() => {
    config.set("format", prev);
  });
  it("detects mp4/gif/mov/png/webm", () => {
    config.set("format", "mp4");
    expect(is_mp4_format()).toBe(true);
    expect(is_gif_format()).toBe(false);
    config.set("format", "gif");
    expect(is_gif_format()).toBe(true);
    expect(is_mp4_format()).toBe(false);
    config.set("format", "mov");
    expect(is_mov_format()).toBe(true);
    config.set("format", "png");
    expect(is_png_format()).toBe(true);
    config.set("format", "webm");
    expect(is_webm_format()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.file_ops.add_version_before_extension
// Python: Path('movie.mp4') -> 'movie_ManimCE_v0.21.0.mp4'.
// ---------------------------------------------------------------------------
describe("add_version_before_extension", () => {
  it("inserts the version before the suffix", () => {
    expect(add_version_before_extension("movie.mp4")).toContain("movie_ManimCE_v");
    expect(add_version_before_extension("movie.mp4").endsWith(".mp4")).toBe(true);
    expect(add_version_before_extension("/a/b/clip.avi").endsWith(".avi")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.file_ops.open_file / add_import_statement
// Python open_file dispatches to xdg-open/cygstart/open/startfile or
// config.preview_command; add_import_statement prepends 'from manim import *'.
// open_file is smoke-tested with preview_command='true' (harmless no-op).
// ---------------------------------------------------------------------------
describe("open_file", () => {
  it("uses the preview command when set", () => {
    const prev = config.get("preview_command");
    config.set("preview_command", "true");
    try {
      expect(() => open_file("/tmp/wanim-open-test.mp4")).not.toThrow();
    } finally {
      config.set("preview_command", prev);
    }
  });
});

describe("add_import_statement", () => {
  it("prepends the manim import line", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-import-"));
    try {
      const file = path.join(dir, "scene.py");
      writeFileSync(file, "x = 1\n", { encoding: "utf-8" });
      add_import_statement(file);
      expect(readFileSync(file, { encoding: "utf-8" }).startsWith("from manim import *\n")).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// manim.camera.camera.Camera.reset / resize_frame_shape
// Python reset: set_pixel_array(background) (copy); resize with fixed_dimension
// 0 keeps width (height = width/aspect), 1 keeps height (width = aspect*height).
// ---------------------------------------------------------------------------
describe("Camera.reset", () => {
  it("restores the pixel array from the background copy", () => {
    const cam = new Camera({ pixel_height: 2, pixel_width: 2 });
    const before = nums(cam.pixel_array);
    const hacked = [...before];
    hacked[0] = (hacked[0] ?? 0) + 7;
    (cam.pixel_array as unknown as { set: (idx: number[], v: number) => void }).set?.([0, 0, 0], hacked[0]!);
    cam.reset();
    expectCloseNums(cam.pixel_array, nums(cam.background));
  });
  it("returns self", () => {
    const cam = new Camera({ pixel_height: 2, pixel_width: 2 });
    expect(cam.reset()).toBe(cam);
  });
});

describe("Camera.resize_frame_shape", () => {
  it("fixes width by default (height = width/aspect)", () => {
    const cam = new Camera({ pixel_height: 2, pixel_width: 4, frame_width: 8 });
    cam.resize_frame_shape(0);
    expect(cam.frame_width).toBeCloseTo(8, 12);
    expect(cam.frame_height).toBeCloseTo(4, 12);
  });
  it("fixes height when fixed_dimension=1", () => {
    // NOTE: the constructor already calls resize_frame_shape(0), so set the
    // height explicitly after construction before resizing with fixed height.
    // Python: width = aspect * height = 2 * 8 = 16.
    const cam = new Camera({ pixel_height: 2, pixel_width: 4, frame_height: 8 });
    cam.frame_height = 8;
    cam.resize_frame_shape(1);
    expect(cam.frame_height).toBeCloseTo(8, 12);
    expect(cam.frame_width).toBeCloseTo(16, 12);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.module_ops.get_module
// Python validates existence + .py suffix then execs the module; TS has no
// Python execution, so a located file yields a lightweight {__name__,__file__}
// record. Validation behaviour (FileNotFound/ValueError, '-' TODO) is tested.
// ---------------------------------------------------------------------------
describe("get_module", () => {
  it("returns a module record for an existing .py file", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-mod-"));
    try {
      const file = path.join(dir, "my_scene.py");
      writeFileSync(file, "x = 1\n", { encoding: "utf-8" });
      const mod = get_module(file);
      expect(mod.__file__).toBe(file);
      expect(mod.__name__).toContain("my_scene");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("throws FileNotFoundError for missing files", () => {
    expect(() => get_module("/tmp/wanim-definitely-missing-xyz.py")).toThrow();
    try {
      get_module("/tmp/wanim-definitely-missing-xyz.py");
    } catch (e) {
      expect((e as Error).name).toBe("FileNotFoundError");
    }
  });
  it("throws ValueError for non-python files", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-mod-"));
    try {
      const file = path.join(dir, "note.txt");
      writeFileSync(file, "x", { encoding: "utf-8" });
      expect(() => get_module(file)).toThrow();
      try {
        get_module(file);
      } catch (e) {
        expect((e as Error).name).toBe("ValueError");
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// manim.utils.typst_file_writing._typst_hash
// Python: sha256(content)[:16]; 'hello' -> 2cf24dba5fb0a30e, '' -> e3b0c44...
// (typst helpers are not tex*; this pure hash is in scope.)
// ---------------------------------------------------------------------------
describe("_typst_hash", () => {
  it("matches sha256 truncation", () => {
    expect(_typst_hash("hello")).toBe("2cf24dba5fb0a30e");
    expect(_typst_hash("")).toBe("e3b0c44298fc1c14");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex_file_writing.tex_hash
// Python: sha256(str(expression).encode()).hexdigest()[:16];
// tex_hash('hello') == '2cf24dba5fb0a30e' (verified against manim 0.21.0).
// ---------------------------------------------------------------------------
describe("tex_hash", () => {
  it("matches sha256 truncation of str(expression)", () => {
    expect(tex_hash("hello")).toBe("2cf24dba5fb0a30e");
    expect(tex_hash("")).toBe("e3b0c44298fc1c14");
  });
  it("stringifies non-string input like Python str()", () => {
    expect(tex_hash(5)).toBe(tex_hash("5"));
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex_file_writing.delete_nonsvg_files
// Python: deletes every entry in config tex_dir whose suffix is not in
// {".svg", ".tex", *additional_endings}.
// ---------------------------------------------------------------------------
describe("delete_nonsvg_files", () => {
  it("keeps .svg/.tex and deletes everything else", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-texdir-"));
    const prev = config.get("tex_dir");
    config.set("tex_dir", dir);
    try {
      for (const name of ["keep.svg", "keep.tex", "drop.pdf", "drop.log", "drop.dvi"]) {
        writeFileSync(path.join(dir, name), "x", { encoding: "utf-8" });
      }
      delete_nonsvg_files();
      expect(readdirSync(dir).sort()).toEqual(["keep.svg", "keep.tex"]);
    } finally {
      config.set("tex_dir", prev);
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("honours additional_endings", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-texdir-"));
    const prev = config.get("tex_dir");
    config.set("tex_dir", dir);
    try {
      for (const name of ["a.pdf", "b.log"]) {
        writeFileSync(path.join(dir, name), "x", { encoding: "utf-8" });
      }
      delete_nonsvg_files([".pdf"]);
      expect(readdirSync(dir)).toEqual(["a.pdf"]);
    } finally {
      config.set("tex_dir", prev);
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.make_even / tuplify
// Python: make_even([1,2],[3,4,5,6]) == ([1,1,2,2],[3,4,5,6]);
//         make_even([1,2],[3,4,5,6,7]) == ([1,1,1,2,2],[3,4,5,6,7]);
//         tuplify('str') == ('str',), tuplify([1,2]) == (1,2).
// ---------------------------------------------------------------------------
describe("make_even", () => {
  it("extends the shorter list favouring earlier elements", () => {
    expect(make_even([1, 2], [3, 4, 5, 6])).toEqual([[1, 1, 2, 2], [3, 4, 5, 6]]);
  });
  it("handles odd extension", () => {
    expect(make_even([1, 2], [3, 4, 5, 6, 7])).toEqual([[1, 1, 1, 2, 2], [3, 4, 5, 6, 7]]);
  });
  it("handles empty inputs", () => {
    expect(make_even([], [])).toEqual([[], []]);
  });
});

describe("tuplify", () => {
  it("wraps strings and scalars, passes iterables through", () => {
    expect(tuplify("str")).toEqual(["str"]);
    expect(tuplify([1, 2])).toEqual([1, 2]);
    expect(tuplify(5)).toEqual([5]);
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.types.vectorized_mobject.VMobject.get_sheen_factor
// Python: returns self.sheen_factor (default 0.0).
// ---------------------------------------------------------------------------
describe("VMobject.get_sheen_factor", () => {
  it("returns the sheen factor", () => {
    const v = new VMobject();
    expect(v.get_sheen_factor()).toBe(0);
    v.sheen_factor = 0.7;
    expect(v.get_sheen_factor()).toBeCloseTo(0.7, 12);
  });
});

// ---------------------------------------------------------------------------
// manim._config.utils.ManimConfig setters / properties
// Ground truths captured from the installed manim 0.21.0 (global config +
// copies). The TS port stores values in `values`; Python derives radii and
// quality from width/height/pixels, which is asserted structurally here.
// ---------------------------------------------------------------------------
describe("ManimConfig._set_boolean", () => {
  it("accepts booleans", () => {
    const c = new ManimConfig(true);
    c._set_boolean("dry_run", true);
    expect(c.get("dry_run")).toBe(true);
  });
  it("accepts 1/0 like Python (`in [True, False]` uses ==)", () => {
    const c = new ManimConfig(true);
    c._set_boolean("dry_run", 1);
    expect(c.get("dry_run")).toBe(1);
  });
  it("rejects non-boolean strings", () => {
    expect(() => new ManimConfig(true)._set_boolean("dry_run", "yes")).toThrow(
      "dry_run must be boolean",
    );
  });
});

describe("ManimConfig._set_from_list", () => {
  it("accepts listed values", () => {
    const c = new ManimConfig(true);
    c._set_from_list("format", "mp4", ["mp4", "gif"]);
    expect(c.get("format")).toBe("mp4");
  });
  it("rejects unlisted values", () => {
    expect(() => new ManimConfig(true)._set_from_list("format", "avi", ["mp4", "gif"])).toThrow(
      "attempted to set format to avi; must be in ['mp4', 'gif']",
    );
  });
});

describe("ManimConfig._set_from_enum", () => {
  const Fruit = ((v: unknown): unknown => {
    if (v === 1 || v === "APPLE") return "APPLE";
    if (v === 2 || v === "BANANA") return "BANANA";
    throw new Error("bad");
  }) as unknown as Record<string, unknown> & ((value: unknown) => unknown);
  Object.defineProperty(Fruit, "name", { value: "Fruit" });
  it("converts values via the enum", () => {
    const c = new ManimConfig(true);
    c._set_from_enum("fruit", 1, Fruit);
    expect(c.get("fruit")).toBe("APPLE");
  });
  it("rejects invalid values", () => {
    expect(() => new ManimConfig(true)._set_from_enum("fruit", 42, Fruit)).toThrow(
      "42 is not a valid Fruit",
    );
  });
});

describe("ManimConfig frame geometry", () => {
  it("frame_width/height store directly; radii are half", () => {
    const c = new ManimConfig(true);
    c.frame_width = 10;
    expect(c.frame_width).toBe(10);
    expect(c.frame_x_radius).toBeCloseTo(5, 12);
    c.frame_height = 6;
    expect(c.frame_height).toBe(6);
    expect(c.frame_y_radius).toBeCloseTo(3, 12);
  });
  it("frame_x_radius/frame_y_radius scale width/height", () => {
    const c = new ManimConfig(true);
    c.frame_x_radius = 5;
    expect(c.frame_width).toBeCloseTo(10, 12);
    c.frame_y_radius = 3;
    expect(c.frame_height).toBeCloseTo(6, 12);
  });
  it("frame_size maps to pixel_width/pixel_height", () => {
    const c = new ManimConfig(true);
    c.frame_size = [800, 600];
    expect(c.frame_size).toEqual([800, 600]);
    expect(c.get("pixel_width")).toBe(800);
  });
});

describe("ManimConfig.quality", () => {
  it("derives the default quality from pixels + frame rate", () => {
    expect(new ManimConfig(true).quality).toBe("high_quality");
  });
  it("applies known qualities to pixels + frame rate", () => {
    const c = new ManimConfig(true);
    c.quality = "low_quality";
    expect(c.quality).toBe("low_quality");
    expect(c.frame_size).toEqual([854, 480]);
    expect(c.frame_rate).toBe(15);
  });
  it("returns null when pixels match nothing", () => {
    const c = new ManimConfig(true);
    c.frame_size = [100, 100];
    expect(c.quality).toBeNull();
  });
  it("rejects unknown qualities", () => {
    expect(() => {
      new ManimConfig(true).quality = "nope";
    }).toThrow();
  });
});

describe("ManimConfig._set_pos_number", () => {
  it("accepts non-negative ints", () => {
    const c = new ManimConfig(true);
    c._set_pos_number("y", 5, false);
    expect(c.get("y")).toBe(5);
  });
  it("maps -1 to infinity when allowed", () => {
    const c = new ManimConfig(true);
    c._set_pos_number("y", -1, true);
    expect(c.get("y")).toBe(Infinity);
  });
  it("rejects negatives", () => {
    expect(() => new ManimConfig(true)._set_pos_number("y", -2, false)).toThrow(
      "must be a non-negative integer",
    );
  });
});

describe("ManimConfig.dry_run", () => {
  it("disables movie output when enabled", () => {
    const c = new ManimConfig(true);
    c.dry_run = true;
    expect(c.dry_run).toBe(true);
    expect(c.write_to_movie).toBe(false);
    expect(c.get("format")).toBeNull();
  });
});

describe("ManimConfig._set_between", () => {
  it("accepts in-range values", () => {
    const c = new ManimConfig(true);
    c._set_between("x", 5, 0, 10);
    expect(c.get("x")).toBe(5);
  });
  it("rejects out-of-range values", () => {
    expect(() => new ManimConfig(true)._set_between("x", 11, 0, 10)).toThrow(
      "x must be 0 <= x <= 10",
    );
  });
});

describe("ManimConfig._set_str", () => {
  it("accepts strings and maps falsy to ''", () => {
    const c = new ManimConfig(true);
    c._set_str("s", "hi");
    expect(c.get("s")).toBe("hi");
    c._set_str("s", 0 as unknown as string);
    expect(c.get("s")).toBe("");
  });
  it("rejects truthy non-strings", () => {
    expect(() => new ManimConfig(true)._set_str("s", 5 as unknown as string)).toThrow(
      "s must be str or falsy value",
    );
  });
});

describe("ManimConfig.encoder_queue_size", () => {
  it("accepts positive ints", () => {
    const c = new ManimConfig(true);
    c.encoder_queue_size = 4;
    expect(c.encoder_queue_size).toBe(4);
  });
  it("rejects non-positive values", () => {
    expect(() => {
      new ManimConfig(true).encoder_queue_size = 0;
    }).toThrow("encoder_queue_size must be a positive integer");
  });
});

describe("ManimConfig.max_inflight_encoders", () => {
  it("accepts positive ints", () => {
    const c = new ManimConfig(true);
    c.max_inflight_encoders = 2;
    expect(c.max_inflight_encoders).toBe(2);
  });
  it("rejects non-positive values", () => {
    expect(() => {
      new ManimConfig(true).max_inflight_encoders = 0;
    }).toThrow("max_inflight_encoders must be a positive integer");
  });
});

describe("ManimConfig.background_color", () => {
  it("wraps values as ManimColor", () => {
    const c = new ManimConfig(true);
    c.background_color = "#ff0000";
    expect(String(c.background_color).toUpperCase()).toContain("FF0000");
  });
});

describe("ManimConfig.from_animation_number", () => {
  it("stores the value", () => {
    const c = new ManimConfig(true);
    c.from_animation_number = 3;
    expect(c.from_animation_number).toBe(3);
  });
});

describe("ManimConfig.frame_rate", () => {
  it("stores the value", () => {
    const c = new ManimConfig(true);
    c.frame_rate = 30;
    expect(c.frame_rate).toBe(30);
  });
});

describe("ManimConfig.scene_names", () => {
  it("stores the value", () => {
    const c = new ManimConfig(true);
    c.scene_names = ["A"];
    expect(c.scene_names).toEqual(["A"]);
  });
});

describe("ManimConfig.plugins", () => {
  it("defaults to [] and stores values", () => {
    expect(new ManimConfig(true).plugins).toEqual([]);
    const c = new ManimConfig(true);
    c.plugins = ["p"];
    expect(c.plugins).toEqual(["p"]);
  });
});

describe("ManimConfig.window_size", () => {
  it("defaults to 'default' and stores values", () => {
    expect(new ManimConfig(true).window_size).toBe("default");
    const c = new ManimConfig(true);
    c.window_size = "100,100";
    expect(c.window_size).toBe("100,100");
  });
});

describe("ManimConfig.window_position", () => {
  it("defaults to 'UR' and stores values", () => {
    expect(new ManimConfig(true).window_position).toBe("UR");
    const c = new ManimConfig(true);
    c.window_position = "UL";
    expect(c.window_position).toBe("UL");
  });
});

describe("ManimConfig.tex_template_file", () => {
  it("stores readable files and clears on empty", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-texcfg-"));
    try {
      const file = path.join(dir, "t.tex");
      writeFileSync(file, "x", { encoding: "utf-8" });
      const c = new ManimConfig(true);
      c.tex_template_file = file;
      expect(String(c.tex_template_file)).toContain("t.tex");
      c.tex_template_file = "";
      expect(c.tex_template_file).toBe("");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("warns and keeps the old value for unreadable files", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const c = new ManimConfig(true);
    c.tex_template_file = "/tmp/wanim-definitely-missing-tex.tex";
    expect(warn).toHaveBeenCalled();
  });
});

describe("ManimConfig._set_dir", () => {
  it("stores strings as-is", () => {
    const c = new ManimConfig(true);
    c._set_dir("assets_dir", "/tmp2");
    expect(c.get("assets_dir")).toBe("/tmp2");
  });
});

describe("ManimConfig._set_tuple", () => {
  it("accepts arrays as tuples", () => {
    const c = new ManimConfig(true);
    c._set_tuple("t", [1, 2]);
    expect(c.get("t")).toEqual([1, 2]);
  });
  it("rejects non-tuples", () => {
    expect(() => new ManimConfig(true)._set_tuple("t", "x")).toThrow("t must be tuple");
  });
});

describe("ManimConfig._set_int_between", () => {
  it("accepts in-range values (floats allowed, like Python)", () => {
    const c = new ManimConfig(true);
    c._set_int_between("x", 5.5, 0, 10);
    expect(c.get("x")).toBe(5.5);
  });
  it("rejects out-of-range values", () => {
    expect(() => new ManimConfig(true)._set_int_between("x", 11, 0, 10)).toThrow(
      "x must be an integer such that 0 <= x <= 10",
    );
  });
});
