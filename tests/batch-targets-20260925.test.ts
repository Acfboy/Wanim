// Ground-truth tests for the layers 2..3 batch (~1000 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim) with the inputs used below; each case cites the
// probe in a "Python:" comment. TS is asserted against those values with
// tight float tolerance (absolute 1e-9 unless noted; exact for discrete).
//
// Known deviations from Python (documented, not worked around):
//  - hashing JSON separators: Python json.dumps uses (', ', ': ') while the
//    TS port uses JSON.stringify (compact). _hash_ndarray is unaffected
//    (Python uses compact separators there too) and is asserted exactly;
//    get_json/_key_to_hash are compared semantically (parsed JSON / crc32 of
//    the same string), see the hashing describe block.
//  - ValueTracker.__floordiv__/__mod__/__truediv__ by zero: Python operates
//    on np.float64 scalars, so `ValueTracker(1) // 0` yields nan/inf with a
//    RuntimeWarning instead of raising. The TS port deliberately throws
//    ZeroDivisionError; tests assert the TS behavior and note the difference.
//  - ManimColor.__hash__: Python returns hash(hex-with-alpha) (SipHash,
//    differs per process); the TS port uses a deterministic Java-style
//    string hash. Tests assert determinism + consistency, not exact values.
//  - earclip_triangulation: Python delegates the final clip to mapbox_earcut
//    after merging hole bridges itself; the TS port has a local ear-clip.
//    Both triangulations are valid but triangle order may differ, so tests
//    assert validity (count, index range, area preservation) plus exact
//    equality where the port currently agrees.
//  - ManimConfig.tex_template: text/tex is OUT OF SCOPE for the entire port;
//    the TS getter/setter are TODO stubs that throw. Tests assert the stub.
//  - Section.get_dict success path needs real video probing (out of scope);
//    only the ValueError path (video=None) is asserted exactly.
//  - Scene.interactive_embed.ipython is a nested closure in Python requiring
//    an IPython/OpenGL session; the TS port exposes the extracted pieces
//    get_embedded_method()/create_embedded_rerun(), which are tested.
//  - AbstractImageMobject cannot be instantiated directly in Python
//    (super().__init__ -> reset_points -> get_pixel_array raises
//    NotImplementedError); the TS constructor only stores attributes, so it
//    *can* be constructed. Tests assert stored attributes + validation.
//  - RandomColorGenerator: Python keeps `choice` bound method + full color
//    list; TS keeps `twister` + `colors` and exposes next(). Seeded sequences
//    match exactly (verified seed=42 below).
//  - qhull containers: Python Horizon.facets is a set of Facet objects and
//    QuickHull.neighbors maps SubFacet->set[Facet]; TS uses arrays/Maps keyed
//    by key() strings with the same dedup semantics.
import { describe, expect, it } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Mobject, Group } from "../src/mobject/mobject.js";
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import {
  AbstractImageMobject,
  RESAMPLE_BICUBIC,
} from "../src/mobject/types/image_mobject.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { TipableVMobject } from "../src/mobject/geometry/arc.js";
import { unit_interval, zero, squish_rate_func } from "../src/utils/rate_functions.js";
import {
  rotation_matrix_from_quaternion,
  rotation_matrix_transpose,
  get_winding_number,
  earclip_triangulation,
  perpendicular_bisector,
} from "../src/utils/space_ops.js";
import {
  _hash_ndarray,
  _Memoizer,
  _CustomEncoder,
  _get_json,
  get_json,
  crc32,
} from "../src/utils/hash.js";
import { Facet, QuickHull, Horizon } from "../src/utils/qhull.js";
import {
  path_along_circles,
  path_along_arc,
  spiral_path,
} from "../src/utils/paths.js";
import {
  ManimColor,
  HSV,
  color_to_rgb,
  color_to_rgba,
  color_to_int_rgb,
  hex_to_rgb,
  RandomColorGenerator,
} from "../src/utils/color/core.js";
import { point_lies_on_bezier } from "../src/utils/bezier.js";
import { Section } from "../src/scene/section.js";
import {
  Scene,
  SceneInteractRerun,
  SceneInteractContinue,
} from "../src/scene/scene.js";
import { MethodWithArgs } from "../src/data_structures.js";
import { LinearTransformationScene } from "../src/scene/vector_space_scene.js";
import { Axes } from "../src/mobject/graphing/coordinate_systems.js";
import { Animation, Wait, Add } from "../src/animation/animation.js";
import { AnimationGroup, Succession } from "../src/animation/composition.js";
import { ManimConfig, make_config_parser } from "../src/_config/utils.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function flat(value: NDArray): number[] {
  return ((value as NDArray).toArray() as unknown[]).flat(Infinity as number[]).map(Number);
}

function mat(value: NDArray): number[][] {
  return (value.toArray() as unknown[][]).map((row) =>
    (Array.isArray(row) ? row : [row]).map(Number),
  );
}

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

function expectMatClose(actual: NDArray, expected: readonly (readonly number[])[], tol = 1e-9): void {
  const rows = mat(actual);
  expect(rows).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]).toHaveLength(expected[i]!.length);
    for (let j = 0; j < expected[i]!.length; j += 1) {
      expect(close(rows[i]![j]!, expected[i]![j]!, tol)).toBe(true);
    }
  }
}

function expectVecClose(actual: readonly number[] | NDArray, expected: readonly number[], tol = 1e-9): void {
  const raw = Array.isArray(actual) ? [...actual].map(Number) : flat(actual);
  expect(raw).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(raw[i]!, expected[i]!, tol)).toBe(true);
  }
}

function polygonArea2d(verts: readonly (readonly number[])[], tris: readonly number[]): number {
  let area = 0;
  for (let k = 0; k < tris.length; k += 3) {
    const a = verts[tris[k]!]!;
    const b = verts[tris[k + 1]!]!;
    const c = verts[tris[k + 2]!]!;
    area += Math.abs((b[0]! - a[0]!) * (c[1]! - a[1]!) - (c[0]! - a[0]!) * (b[1]! - a[1]!)) / 2;
  }
  return area;
}

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject.SVGMobject.get_file_path
// ---------------------------------------------------------------------------
describe("SVGMobject.get_file_path", () => {
  it("raises ValueError when file_name is None", () => {
    // Python: SVGMobject with file_name=None -> ValueError("Must specify file for SVGMobject")
    const mob = new SVGMobject();
    expect(() => mob.get_file_path()).toThrowError(/Must specify file for SVGMobject/);
    try {
      mob.get_file_path();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });

  it("resolves an existing svg file path containing the file name", () => {
    // Python: get_file_path returns a Path to the found file (OSError when missing).
    const dir = mkdtempSync(join(tmpdir(), "wanim-svg-"));
    writeFileSync(join(dir, "probe.svg"), "<svg></svg>");
    const mob = new SVGMobject({ file_name: join(dir, "probe.svg") });
    const found = mob.get_file_path();
    expect(found).toContain("probe.svg");
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.mobject.Mobject.remove / reduce_across_dimension / get_merged_array
// ---------------------------------------------------------------------------
describe("Mobject.remove", () => {
  it("removes submobjects by identity and returns self", () => {
    // Python: p.add(a,b,c); p.remove(b) -> [a, c]; remove(c) twice -> [a]
    const a = new Mobject();
    const b = new Mobject();
    const c = new Mobject();
    const p = new Mobject();
    p.add(a, b, c);
    expect(p.remove(b)).toBe(p);
    expect(p.submobjects).toEqual([a, c]);
    p.remove(c);
    p.remove(c);
    expect(p.submobjects).toEqual([a]);
  });

  it("ignores mobjects that are not submobjects", () => {
    const p = new Mobject();
    const stray = new Mobject();
    expect(() => p.remove(stray)).not.toThrow();
    expect(p.submobjects).toHaveLength(0);
  });
});

describe("Mobject.reduce_across_dimension", () => {
  it("computes max/min across own points plus submobjects", () => {
    // Python: points [[1,2,3],[4,5,6]] + sub [[0,0,0],[10,10,10]]:
    //   max dim0 -> 10.0, min dim1 -> 0.0
    const m = new Mobject();
    m.set({ points: array([[1, 2, 3], [4, 5, 6]]) as NDArray });
    const sub = new Mobject();
    sub.set({ points: array([[0, 0, 0], [10, 10, 10]]) as NDArray });
    m.add(sub);
    expect(m.reduce_across_dimension((vals) => Math.max(...vals), 0)).toBeCloseTo(10, 9);
    expect(m.reduce_across_dimension((vals) => Math.min(...vals), 1)).toBeCloseTo(0, 9);
  });

  it("returns null for an empty mobject", () => {
    // Python: Mobject().reduce_across_dimension(max, 0) is None
    expect(new Mobject().reduce_across_dimension((vals) => Math.max(...vals), 0)).toBeNull();
  });

  it("uses only submobject points when self has none", () => {
    const m = new Mobject();
    const sub = new Mobject();
    sub.set({ points: array([[3, 1, 2]]) as NDArray });
    m.add(sub);
    expect(m.reduce_across_dimension((vals) => Math.min(...vals), 0)).toBeCloseTo(3, 9);
  });
});

describe("Mobject.get_merged_array", () => {
  it("concatenates depth-first like Python", () => {
    // Python: merged points == [[1,2,3],[4,5,6],[0,0,0],[10,10,10]]
    const m = new Mobject();
    m.set({ points: array([[1, 2, 3], [4, 5, 6]]) as NDArray });
    const sub = new Mobject();
    sub.set({ points: array([[0, 0, 0], [10, 10, 10]]) as NDArray });
    m.add(sub);
    expectMatClose(m.get_merged_array("points"), [
      [1, 2, 3],
      [4, 5, 6],
      [0, 0, 0],
      [10, 10, 10],
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.rate_functions.unit_interval / zero / squish_rate_func
// ---------------------------------------------------------------------------
describe("rate_functions decorators", () => {
  it("unit_interval clamps outside [0,1]", () => {
    // Python: unit_interval(t*t) at [-0.5,0,0.25,1,1.5] -> [0,0,0.0625,1,1]
    const f = unit_interval(((t: number) => t * t) as (t: number) => number);
    expect([f(-0.5), f(0), f(0.25), f(1), f(1.5)]).toEqual([0, 0, 0.0625, 1, 1]);
  });

  it("zero returns 0 outside [0,1]", () => {
    // Python: zero(t+1) at [-0.5,0,0.5,1,1.5] -> [0,1,1.5,2,0]
    const f = zero(((t: number) => t + 1) as (t: number) => number);
    expect([f(-0.5), f(0), f(0.5), f(1), f(1.5)]).toEqual([0, 1, 1.5, 2, 0]);
  });

  it("squish_rate_func remaps [a,b] onto [0,1]", () => {
    // Python: squish(identity, .4, .6) at [0,.3,.4,.5,.6,.7,1] -> [0,0,0,.5,1,1,1]
    const f = squish_rate_func(((t: number) => t) as (t: number) => number, 0.4, 0.6);
    expect([f(0), f(0.3), f(0.4), f(0.5), f(0.6), f(0.7), f(1)]).toEqual([0, 0, 0, 0.5, 1, 1, 1]);
  });

  it("squish_rate_func with a == b returns a", () => {
    // Python: squish(identity, .5, .5)(t) -> 0.5 for any t
    const f = squish_rate_func(((t: number) => t) as (t: number) => number, 0.5, 0.5);
    expect(f(0)).toBe(0.5);
    expect(f(0.7)).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops: rotation fns / winding / earclip / bisector
// ---------------------------------------------------------------------------
describe("space_ops rotation matrices", () => {
  // Python quat for angle=1 about z: [0.8775825618903728, 0, 0, 0.479425538604203]
  const quat = [0.8775825618903728, 0, 0, 0.479425538604203];

  it("rotation_matrix_from_quaternion matches Python", () => {
    expectMatClose(rotation_matrix_from_quaternion(quat), [
      [0.5403023058681398, -0.8414709848078965, 0],
      [0.8414709848078965, 0.5403023058681398, 0],
      [0, 0, 1],
    ]);
  });

  it("rotation_matrix_transpose on z-axis matches Python", () => {
    expectMatClose(rotation_matrix_transpose(1.0, [0, 0, 1]), [
      [0.5403023058681398, 0.8414709848078965, 0],
      [-0.8414709848078965, 0.5403023058681398, 0],
      [0, 0, 1],
    ]);
  });

  it("rotation_matrix_transpose off-axis matches Python", () => {
    expectMatClose(rotation_matrix_transpose(0.5, [1, 0, 0]), [
      [0.9999999999999999, 0, 0],
      [0, 0.8775825618903726, 0.479425538604203],
      [0, -0.479425538604203, 0.8775825618903726],
    ]);
  });

  it("rotation_matrix_transpose with negative z uses sign flip", () => {
    // Python: rotation_about_z(0.5 * sign(-1)).T
    expectMatClose(rotation_matrix_transpose(0.5, [0, 0, -1]), [
      [0.8775825618903728, -0.479425538604203, 0],
      [0.479425538604203, 0.8775825618903728, 0],
      [0, 0, 1],
    ]);
  });
});

describe("get_winding_number", () => {
  it("is 1 for a square around the origin", () => {
    // Python: 1.0
    const pts = [
      [1, 0, 0],
      [0, 1, 0],
      [-1, 0, 0],
      [0, -1, 0],
    ];
    expect(close(get_winding_number(pts), 1.0)).toBe(true);
  });

  it("is 0 for a polygon not containing the origin", () => {
    // Python: 0.0
    const pts = [
      [3, 0, 0],
      [2, 1, 0],
      [2, -1, 0],
    ];
    expect(close(get_winding_number(pts), 0.0)).toBe(true);
  });
});

describe("earclip_triangulation", () => {
  it("triangulates a quad into a valid covering", () => {
    // Python (mapbox_earcut) gives [2, 3, 0, 0, 1, 2]; the TS port has a
    // local ear-clip giving [3, 0, 1, 1, 2, 3]. Both are valid triangulations
    // of the unit square (same 2-triangle area cover), so assert validity.
    const verts = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ];
    const tris = earclip_triangulation(verts, [4]);
    expect(tris).toHaveLength(6);
    for (const idx of tris) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(4);
    }
    expect(close(polygonArea2d(verts, tris), 1.0, 1e-9)).toBe(true);
  });

  it("triangulates a polygon with a hole into a valid covering", () => {
    // NOTE: the hole ring must wind opposite to the outer ring (CW here);
    // a same-orientation hole is invalid input and both implementations
    // double-cover it. Python (mapbox_earcut) gives
    // [3,0,4,7,4,0,3,4,5,7,0,1,2,3,5,6,7,1,2,5,6,6,1,2]; the TS local
    // ear-clip gives a different but equally valid 8-triangle cover
    // (verified: same strict-interior grid coverage as Python, area 8.0).
    const verts = [
      [0, 0, 0],
      [3, 0, 0],
      [3, 3, 0],
      [0, 3, 0],
      [1, 1, 0],
      [1, 2, 0],
      [2, 2, 0],
      [2, 1, 0],
    ];
    const tris = earclip_triangulation(verts, [4, 8]);
    expect(tris.length % 3).toBe(0);
    for (const idx of tris) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(8);
    }
    expect(close(polygonArea2d(verts, tris), 8.0, 1e-9)).toBe(true);
  });
});

describe("perpendicular_bisector", () => {
  it("bisects [(0,0,0),(2,0,0)] like Python", () => {
    // Python: [[1,2,0],[1,-2,0]]
    const [p, q] = perpendicular_bisector([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    expectVecClose(p!, [1, 2, 0]);
    expectVecClose(q!, [1, -2, 0]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.hashing: _hash_ndarray / _handle_already_processed /
//   _cleaned_iterable._key_to_hash / get_json
// ---------------------------------------------------------------------------
describe("hashing", () => {
  it("_hash_ndarray matches Python exactly", () => {
    // Python: "NDARRAY:\"<f8\":(2, 2):6bab56d2...a714b07d"
    expect(_hash_ndarray(array([[1, 2], [3, 4]]))).toBe(
      "NDARRAY:\"<f8\":(2, 2):6bab56d2f81d4b5a2dbf102bf6a6ff7d5211a475fc5f97813f977e8ba714b07d",
    );
  });

  it("_hash_ndarray matches Python for int/1-d/empty arrays", () => {
    // Python: int64 [1,2,3] -> 'NDARRAY:"<i8":(3,):e2e2033a...59c54ef'
    expect(_hash_ndarray(array([1, 2, 3], "int64"))).toBe(
      "NDARRAY:\"<i8\":(3,):e2e2033ae7e19d680599d4eb0a1359a2b48ec5baac75066c317fbf85159c54ef",
    );
    // Python: [0.5] -> 'NDARRAY:"<f8":(1,):4cfa5b42...7ccce15ed'
    expect(_hash_ndarray(array([0.5]))).toBe(
      "NDARRAY:\"<f8\":(1,):4cfa5b42ca669328764e67cd9a34bb8f90b16ed7ca8d85e8443783d7ccce15ed",
    );
    // Python: [] -> 'NDARRAY:"<f8":(0,):e3b0c442...991b7852b855' (sha256 of empty)
    expect(_hash_ndarray(array([]))).toBe(
      "NDARRAY:\"<f8\":(0,):e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("_Memoizer._handle_already_processed passes primitives through", () => {
    // Python: str/int/float/complex returned as-is without memoizing
    const m = new _Memoizer();
    let called = 0;
    expect(m._handle_already_processed("hello", () => { called += 1; return "X"; })).toBe("hello");
    expect(m._handle_already_processed(5, () => { called += 1; return "X"; })).toBe(5);
    expect(m._handle_already_processed(10n, () => { called += 1; return "X"; })).toBe(10n);
    expect(called).toBe(0);
  });

  it("repeated nulls collapse to AP like Python", () => {
    // Python: get_json({'a': None, 'b': None}) == '{"a": null, "b": "AP"}'
    expect(get_json({ a: null, b: null })).toBe('{"a":null,"b":"AP"}');
  });

  it("_Memoizer._handle_already_processed flags repeats with AP", () => {
    // Python: same list object twice -> second yields ALREADY_PROCESSED_PLACEHOLDER;
    // two *distinct* equal lists are both processed (id-based fallback).
    // DEVIATION (documented in hash.ts): Python dedups repeated equal
    // *tuples* by value; JS cannot tell tuples from lists, so TS uses
    // identity for all arrays (avoids corrupting list data with false APs).
    const m = new _Memoizer();
    const obj = [1, 2];
    expect(m._handle_already_processed(obj, (v) => v)).toBe(obj);
    expect(m._handle_already_processed(obj, (v) => v)).toBe(_Memoizer.ALREADY_PROCESSED_PLACEHOLDER);
    const m2 = new _Memoizer();
    expect(m2._handle_already_processed([1, 2], (v) => v)).toEqual([1, 2]);
    expect(m2._handle_already_processed([1, 2], (v) => v)).toEqual([1, 2]);
    // Repeated equal tuples stay expanded in TS (Python would emit "AP"
    // for the second one).
    expect(JSON.parse(get_json({ x: [1, 2], y: [1, 2] }))).toEqual({ x: [1, 2], y: [1, 2] });
  });

  it("_CustomEncoder._cleaned_iterable hashes non-JSON keys like Python", () => {
    // Python: {(1,2): 'x', 'a': 1} -> {1640622079: 'x', 'a': 1}, where
    // 1640622079 == zlib.crc32(b'[1, 2]').
    // NOTE: TS JSON.stringify emits "[1,2]" (no space) while Python emits
    // "[1, 2]", so the TS-computed key hash differs in value; assert the
    // mechanism instead: non-string keys become crc32(_get_json(key)).
    const enc = new _CustomEncoder(new _Memoizer());
    const cleaned = enc._cleaned_iterable({ a: 1 }) as Record<string, unknown>;
    expect(cleaned).toEqual({ a: 1 });
    const keyHash = enc._key_to_hash([1, 2]);
    expect(keyHash).toBe(crc32(_get_json([1, 2], new _Memoizer())));
    // And the crc32 implementation itself matches zlib on identical input:
    // Python zlib.crc32(b'[1, 2]') == 1640622079
    expect(crc32("[1, 2]")).toBe(1640622079);
  });

  it("_CustomEncoder._cleaned_iterable replaces circular refs with AP", () => {
    // Python: d['self'] = d -> {'self': {'self': 'AP'}}
    const enc = new _CustomEncoder(new _Memoizer());
    const d: Record<string, unknown> = {};
    d["self"] = d;
    expect(enc._cleaned_iterable(d)).toEqual({
      self: { self: _Memoizer.ALREADY_PROCESSED_PLACEHOLDER },
    });
  });

  it("get_json matches Python semantically (modulo separators)", () => {
    // Python: '{"b": 2, "a": [1, 2, {"c": null}]}'
    expect(JSON.parse(get_json({ b: 2, a: [1, 2, { c: null }] }))).toEqual({
      b: 2,
      a: [1, 2, { c: null }],
    });
  });
});

// ---------------------------------------------------------------------------
// manim.utils.qhull.Facet.__init__ / QuickHull._recursive_horizon
// ---------------------------------------------------------------------------
describe("qhull Facet.__init__", () => {
  it("computes center/normal/subfacets like Python", () => {
    // Python: center [1/3,1/3,0], normal [0,0,-1] (internal [0.1,0.1,0.1]
    // sits above the xy-plane facet, so the normal points away/down),
    // 3 subfacets, coordinates preserved.
    const f = new Facet(
      [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
      [0.1, 0.1, 0.1],
    );
    expectVecClose(f.center, [1 / 3, 1 / 3, 0]);
    expectVecClose(f.normal, [-0, -0, -1]);
    expect(f.subfacets).toHaveLength(3);
    expect(f.coordinates).toEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
  });
});

describe("QuickHull._recursive_horizon", () => {
  function tetraHull(): { qh: QuickHull; facets: Facet[]; internal: number[] } {
    const verts = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const qh = new QuickHull();
    const internal = [0.25, 0.25, 0.25];
    qh.internal = [...internal];
    for (let c = 0; c < 4; c += 1) {
      const facet = new Facet(verts.filter((_, i) => i !== c), internal);
      qh.facets.push(facet);
    }
    for (const facet of qh.facets) {
      for (const sub of facet.subfacets) {
        const key = sub.key();
        let bucket = qh.neighbors.get(key);
        if (bucket === undefined) {
          bucket = new Map();
          qh.neighbors.set(key, bucket);
        }
        bucket.set(facet.key(), facet);
      }
    }
    return { qh, facets: qh.facets, internal };
  }

  it("returns true and collects the visible cap for an outside eye", () => {
    // Python: eye = center + 5*normal -> True, 1 facet, 3 boundary edges
    const { qh, facets } = tetraHull();
    const eye = facets[0]!.center.map((v, i) => v + 5 * facets[0]!.normal[i]!);
    const horizon = new Horizon();
    expect(qh._recursive_horizon(eye, facets[0]!, horizon)).toBe(true);
    expect(horizon.facets).toHaveLength(1);
    expect(horizon.boundary).toHaveLength(3);
  });

  it("returns false for an interior eye", () => {
    // Python: _recursive_horizon(internal, facet, h) -> False
    const { qh, facets, internal } = tetraHull();
    expect(qh._recursive_horizon(internal, facets[0]!, new Horizon())).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// paths: path_along_circles.path / path_along_arc.path / spiral_path.path
// ---------------------------------------------------------------------------
describe("path functions", () => {
  const s0 = array([[0, 0, 0], [1, 0, 0]]) as NDArray;
  const s1 = array([[1, 1, 0], [2, 1, 0]]) as NDArray;

  it("path_along_circles.path matches Python", () => {
    // Python (arc=1, center=origin, alpha=.5):
    // [[0.6785040502472879,0.1990785116430849,0],[1.5560866121376606,0.19907851164308502,0]]
    const p = path_along_circles(1.0, array([0, 0, 0]) as NDArray);
    expectMatClose(p(s0, s1, 0.5), [
      [0.6785040502472879, 0.1990785116430849, 0],
      [1.5560866121376606, 0.19907851164308502, 0],
    ]);
  });

  it("path_along_arc.path matches Python", () => {
    // Python (arc=1, alpha=.5):
    // [[0.6276709606105183,0.37232903938948203,0],[1.6276709606105184,0.37232903938948203,0]]
    const p = path_along_arc(1.0);
    expectMatClose(p(s0, s1, 0.5), [
      [0.6276709606105183, 0.37232903938948203, 0],
      [1.6276709606105184, 0.37232903938948203, 0],
    ]);
  });

  it("path_along_arc.path with arc=pi matches Python", () => {
    // Python: ~[[1,-1.1e-16,0],[2,-1.1e-16,0]]
    const p = path_along_arc(Math.PI);
    expectMatClose(p(s0, s1, 0.5), [
      [0.9999999999999999, -1.1102230246251565e-16, 0],
      [2.0, -1.1102230246251565e-16, 0],
    ]);
  });

  it("path_along_arc.path with tiny angle is straight interpolation", () => {
    // Python (arc=0.001 < threshold): [[0.5,0.5,0],[1.5,0.5,0]]
    const p = path_along_arc(0.001);
    expectMatClose(p(s0, s1, 0.5), [
      [0.5, 0.5, 0],
      [1.5, 0.5, 0],
    ]);
  });

  it("spiral_path.path matches Python", () => {
    // Python (angle=2, alpha=.5):
    // [[0.6908866453380181,-0.15058433946987837,0],[1.690886645338018,-0.15058433946987837,0]]
    const p = spiral_path(2.0);
    expectMatClose(p(s0, s1, 0.5), [
      [0.6908866453380181, -0.15058433946987837, 0],
      [1.690886645338018, -0.15058433946987837, 0],
    ]);
  });

  it("spiral_path.path with tiny angle is straight interpolation", () => {
    // Python: [[0.5,0.5,0],[1.5,0.5,0]]
    const p = spiral_path(0.001);
    expectMatClose(p(s0, s1, 0.5), [
      [0.5, 0.5, 0],
      [1.5, 0.5, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// PMobject.__init__ / AbstractImageMobject.__init__
// ---------------------------------------------------------------------------
describe("PMobject.__init__", () => {
  it("stores stroke_width and starts with empty points", () => {
    // Python: PMobject().stroke_width == 4 (DEFAULT_STROKE_WIDTH), points (0,3)
    const p = new PMobject();
    expect(p.stroke_width).toBe(4);
    expect(p.points.shape).toEqual([0, 3]);
    const custom = new PMobject({ stroke_width: 7 });
    expect(custom.stroke_width).toBe(7);
  });
});

describe("AbstractImageMobject.__init__", () => {
  it("stores scale/dtype/resampling attributes", () => {
    // Python signature defaults: pixel_array_dtype='uint8',
    // resampling_algorithm=Resampling.BICUBIC (== 3).
    // NOTE: Python cannot instantiate the abstract class directly
    // (reset_points -> get_pixel_array raises NotImplementedError); the TS
    // port only stores attributes, so construction succeeds.
    const m = new AbstractImageMobject({ scale_to_resolution: 720 });
    expect(m.scale_to_resolution).toBe(720);
    expect(m.pixel_array_dtype).toBe("uint8");
    expect(m.resampling_algorithm).toBe(RESAMPLE_BICUBIC);
    expect(m.resampling_algorithm).toBe(3);
  });

  it("rejects unknown resampling algorithms like Python", () => {
    const m = new AbstractImageMobject({ scale_to_resolution: 720 });
    expect(() => m.set_resampling_algorithm(999)).toThrowError(/resampling_algorithm/);
  });
});

// ---------------------------------------------------------------------------
// ValueTracker.__init__ / get_value / dunders
// ---------------------------------------------------------------------------
describe("ValueTracker", () => {
  it("__init__ defaults to 0 and stores the value", () => {
    // Python: ValueTracker().get_value() == 0.0, ValueTracker(3.5) -> 3.5
    expect(new ValueTracker().get_value()).toBe(0);
    expect(new ValueTracker(3.5).get_value()).toBe(3.5);
  });

  it("get_value reads points[0,0]", () => {
    const v = new ValueTracker(2);
    v.set_value(-1.25);
    expect(v.get_value()).toBe(-1.25);
  });

  it("__bool__ follows the value", () => {
    // Python: bool(ValueTracker(0)) False, bool(ValueTracker(2.5)) True
    expect(new ValueTracker(0).__bool__()).toBe(false);
    expect(new ValueTracker(2.5).__bool__()).toBe(true);
  });

  it("__floordiv__/__ifloordiv__ floor like Python", () => {
    // Python: 7.5//2 -> 3.0, -7.5//2 -> -4.0
    expect(new ValueTracker(7.5).__floordiv__(2).get_value()).toBe(3);
    expect(new ValueTracker(-7.5).__floordiv__(2).get_value()).toBe(-4);
    const v = new ValueTracker(7.5);
    const self = v.__ifloordiv__(2);
    expect(self).toBe(v);
    expect(v.get_value()).toBe(3);
  });

  it("__mod__/__imod__ follow floored Python semantics", () => {
    // Python: 7.5%2 -> 1.5, -7.5%2 -> 0.5
    expect(new ValueTracker(7.5).__mod__(2).get_value()).toBeCloseTo(1.5, 12);
    expect(new ValueTracker(-7.5).__mod__(2).get_value()).toBeCloseTo(0.5, 12);
    const v = new ValueTracker(7.5);
    expect(v.__imod__(2)).toBe(v);
    expect(v.get_value()).toBeCloseTo(1.5, 12);
  });

  it("__mul__/__imul__", () => {
    // Python: ValueTracker(3)*2.5 -> 7.5
    expect(new ValueTracker(3).__mul__(2.5).get_value()).toBe(7.5);
    const v = new ValueTracker(3);
    expect(v.__imul__(2.5)).toBe(v);
    expect(v.get_value()).toBe(7.5);
  });

  it("__pow__/__ipow__", () => {
    // Python: ValueTracker(2)**10 -> 1024.0
    expect(new ValueTracker(2).__pow__(10).get_value()).toBe(1024);
    const v = new ValueTracker(2);
    expect(v.__ipow__(10)).toBe(v);
    expect(v.get_value()).toBe(1024);
  });

  it("__truediv__/__itruediv__", () => {
    // Python: 7.5/2 -> 3.75
    expect(new ValueTracker(7.5).__truediv__(2).get_value()).toBe(3.75);
    const v = new ValueTracker(7.5);
    expect(v.__itruediv__(2)).toBe(v);
    expect(v.get_value()).toBe(3.75);
  });

  it("division by zero throws ZeroDivisionError in TS (deviation documented)", () => {
    // DEVIATION: Python works on np.float64 scalars, so e.g.
    // ValueTracker(1).__truediv__(0) -> inf with RuntimeWarning (no raise);
    // the TS port deliberately raises ZeroDivisionError instead.
    for (const op of ["__floordiv__", "__mod__", "__truediv__"] as const) {
      expect(() => (new ValueTracker(1) as unknown as Record<string, (x: number) => unknown>)[op]!(0)).toThrowError(
        expect.objectContaining({ name: "ZeroDivisionError" }),
      );
    }
    for (const op of ["__ifloordiv__", "__imod__", "__itruediv__"] as const) {
      expect(() => (new ValueTracker(1) as unknown as Record<string, (x: number) => unknown>)[op]!(0)).toThrowError(
        expect.objectContaining({ name: "ZeroDivisionError" }),
      );
    }
  });
});

// ---------------------------------------------------------------------------
// ManimConfig.tex_template (OUT OF SCOPE stub) / digest_file
// ---------------------------------------------------------------------------
describe("ManimConfig.tex_template", () => {
  it("getter and setter are clearly-marked TODO stubs (text/tex out of scope)", () => {
    // Python returns a TexTemplate; the port skips the whole text/tex stack.
    const cfg = new ManimConfig();
    expect(() => cfg.tex_template).toThrowError(/TODO/);
    expect(() => {
      cfg.tex_template = {};
    }).toThrowError(/TODO/);
  });
});

describe("ManimConfig.digest_file", () => {
  it("digests a cfg file like Python", () => {
    // Python: digest_file with [CLI] frame_width=10 -> frame_width 10.0
    const dir = mkdtempSync(join(tmpdir(), "wanim-cfg-"));
    const path = join(dir, "test.cfg");
    writeFileSync(path, "[CLI]\nframe_width = 10\npixel_width = 640\n");
    const cfg = new ManimConfig();
    expect(cfg.digest_file(path)).toBe(cfg);
    expect(Number(cfg.frame_width)).toBeCloseTo(10, 9);
    expect(Number(cfg.pixel_width)).toBe(640);
  });

  it("raises FileNotFoundError for a missing file like Python", () => {
    // Python: FileNotFoundError(ENOENT, '--config_file ...')
    const cfg = new ManimConfig();
    try {
      cfg.digest_file("/nonexistent-xyz.cfg");
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("FileNotFoundError");
      expect((e as Record<string, unknown>)["errno"]).toBe(2);
    }
  });
});

// ---------------------------------------------------------------------------
// VMobject.update_rgbas_array / get_stroke_color / get_fill_color /
// get_fill_opacity / get_stroke_opacity / get_nth_curve_function
// ---------------------------------------------------------------------------
describe("VMobject color/opacity accessors", () => {
  function colored(): VMobject {
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) as NDArray);
    v.set_fill("#FF0000", 0.5);
    v.set_stroke("#00FF00", 4, 0.25);
    return v;
  }

  it("get_fill_color/get_stroke_color match Python", () => {
    // Python: '#FF0000' / '#00FF00'
    const v = colored();
    expect(v.get_fill_color().to_hex()).toBe("#FF0000");
    expect(v.get_stroke_color()!.to_hex()).toBe("#00FF00");
  });

  it("get_fill_opacity/get_stroke_opacity match Python", () => {
    // Python: 0.5 / 0.25
    const v = colored();
    expect(v.get_fill_opacity()).toBeCloseTo(0.5, 9);
    expect(v.get_stroke_opacity()).toBeCloseTo(0.25, 9);
  });

  it("update_rgbas_array with color+opacity matches Python", () => {
    // Python: set 4 points, update fill_rgbas blue/0.75 -> [[0,0,1,0.75]]
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.update_rgbas_array("fill_rgbas", "#0000FF", 0.75);
    expectMatClose(v.fill_rgbas as NDArray, [[0, 0, 1, 0.75]]);
  });

  it("update_rgbas_array with opacity only keeps rgb", () => {
    // Python: default white fill, opacity 0.3 -> [[1,1,1,0.3]]
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]) as NDArray);
    v.update_rgbas_array("fill_rgbas", null, 0.3);
    expectMatClose(v.fill_rgbas as NDArray, [[1, 1, 1, 0.3]]);
  });

  it("get_nth_curve_function matches Python bezier evaluation", () => {
    // Python: VMobject through [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
    // curve 0 at t=0.5 -> [0.75, 0.5, 0]
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) as NDArray);
    expectVecClose(v.get_nth_curve_function(0)(0.5), [0.75, 0.5, 0]);
  });
});

// ---------------------------------------------------------------------------
// bezier.point_lies_on_bezier
// ---------------------------------------------------------------------------
describe("point_lies_on_bezier", () => {
  const cp = array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) as NDArray;
  it("finds endpoints and midpoints on the curve", () => {
    // Python: origin True, [0.75,0.5,0] (t=0.5) True
    expect(point_lies_on_bezier(array([0, 0, 0]) as NDArray, cp)).toBe(true);
    expect(point_lies_on_bezier(array([0.75, 0.5, 0]) as NDArray, cp)).toBe(true);
  });

  it("rejects far-off points", () => {
    // Python: [5,5,0] False
    expect(point_lies_on_bezier(array([5, 5, 0]) as NDArray, cp)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// ManimColor core: _construct_from_space / to_hsv / to_hsl / _from_internal /
// from_rgba / __repr__ / __str__ / __int__ / __getitem__ / __hash__
// ---------------------------------------------------------------------------
describe("ManimColor core dunders and converters", () => {
  it("to_hsv matches Python colorsys", () => {
    // Python: red -> [0,1,1]; GREEN -> [0.28148...,0.46632...,0.75686...]
    expectVecClose(flat(new ManimColor("#FF0000").to_hsv()), [0, 1, 1]);
    expectVecClose(flat(new ManimColor("GREEN").to_hsv()), [
      0.2814814814814815, 0.46632124352331605, 0.7568627450980392,
    ]);
  });

  it("to_hsl matches Python colorsys", () => {
    // Python: red -> [0,1,0.5]; GREEN -> [0.28148...,0.42056...,0.58039...]
    expectVecClose(flat(new ManimColor("#FF0000").to_hsl()), [0, 1, 0.5]);
    expectVecClose(flat(new ManimColor("GREEN").to_hsl()), [
      0.2814814814814815, 0.42056074766355134, 0.580392156862745,
    ]);
  });

  it("_construct_from_space and _from_internal preserve rgba", () => {
    // Python: both round-trip [0.1,0.2,0.3,0.4]
    expectVecClose(
      flat(ManimColor._construct_from_space([0.1, 0.2, 0.3, 0.4]).to_rgba()),
      [0.1, 0.2, 0.3, 0.4],
    );
    expectVecClose(
      flat(ManimColor._from_internal(array([0.1, 0.2, 0.3, 0.4]) as NDArray).to_rgba()),
      [0.1, 0.2, 0.3, 0.4],
    );
  });

  it("from_rgba distinguishes float and int input like Python", () => {
    // Python: [0.5,0.25,1.0,0.5] stays float; [255,0,0,255] scales to [1,0,0,1]
    expectVecClose(flat(ManimColor.from_rgba([0.5, 0.25, 1.0, 0.5]).to_rgba()), [0.5, 0.25, 1, 0.5]);
    expectVecClose(flat(ManimColor.from_rgba([255, 0, 0, 255]).to_rgba()), [1, 0, 0, 1]);
  });

  it("__repr__/__str__ match Python", () => {
    // Python: repr "ManimColor('#123456')", str '#123456'
    expect(new ManimColor("#123456").__repr__()).toBe("ManimColor('#123456')");
    expect(new ManimColor("#123456").__str__()).toBe("#123456");
    expect(String(new ManimColor("#123456"))).toBe("#123456");
  });

  it("__int__ matches Python to_integer", () => {
    // Python: int(#FF0000) == 16711680
    expect(new ManimColor("#FF0000").__int__()).toBe(16711680);
  });

  it("__getitem__ indexes the internal [r,g,b,a]", () => {
    // Python: #123456 -> [0.07058...,0.20392...,0.33725...,1.0]
    const c = new ManimColor("#123456");
    expectVecClose([c.__getitem__(0), c.__getitem__(1), c.__getitem__(2), c.__getitem__(3)], [
      0.07058823529411765, 0.20392156862745098, 0.33725490196078434, 1.0,
    ]);
  });

  it("__hash__ is deterministic and consistent with equality", () => {
    // DEVIATION: Python hash() is SipHash (per-process); TS uses a stable
    // Java-style string hash of the hex+alpha. Assert the contract instead.
    const a = new ManimColor("#FF0000");
    const b = new ManimColor("#FF0000");
    expect(a.__hash__()).toBe(b.__hash__());
    expect(Number.isInteger(a.__hash__())).toBe(true);
    expect(a.__hash__()).toBe(a.__hash__());
  });
});

describe("ManimColor helpers and subclasses", () => {
  it("HSV stores hue and converts to red", () => {
    // Python: HSV([0,1,1]).to_rgb() == [1,0,0]
    const h = new HSV([0, 1, 1]);
    expectVecClose(flat(h.to_rgb()), [1, 0, 0]);
    expect(h.hue ?? (h as unknown as { h: number }).h).toBeCloseTo(0, 9);
  });

  it("HSV rejects bad lengths like Python", () => {
    // Python: ValueError("HSV Color must be an array of 3 values")
    expect(() => new HSV([0.5, 0.5] as unknown as number[])).toThrowError(/array of 3 values/);
  });

  it("color_to_rgb / color_to_rgba / color_to_int_rgb / hex_to_rgb", () => {
    // Python: RED -> [0.98823...,0.38431...,0.33333...];
    // rgba alpha 0.5 appended; int rgb [252,98,85]; #00FF00 -> [0,1,0]
    expectVecClose(flat(color_to_rgb("RED")), [0.9882352941176471, 0.3843137254901961, 0.3333333333333333]);
    expectVecClose(flat(color_to_rgba("RED", 0.5)), [
      0.9882352941176471, 0.3843137254901961, 0.3333333333333333, 0.5,
    ]);
    expect([...flat(color_to_int_rgb("RED"))]).toEqual([252, 98, 85]);
    expectVecClose(flat(hex_to_rgb("#00FF00")), [0, 1, 0]);
  });

  it("RandomColorGenerator(seed) reproduces Python's sequence", () => {
    // Python random.Random(42).choice over _all_manim_colors (89 colors):
    // ['#8B4513','#BBBBBB','#BBBBBB','#ACEAD7','#29ABCA', ...]
    const gen = new RandomColorGenerator(42);
    expect(gen.colors).toHaveLength(89);
    expect(gen.colors[0]!.to_hex()).toBe("#FFFFFF");
    const seq = [gen.next().to_hex(), gen.next().to_hex(), gen.next().to_hex(), gen.next().to_hex(), gen.next().to_hex()];
    expect(seq).toEqual(["#8B4513", "#BBBBBB", "#BBBBBB", "#ACEAD7", "#29ABCA"]);
  });

  it("RandomColorGenerator honors custom sample_colors", () => {
    const only = [new ManimColor("#123456")];
    const gen = new RandomColorGenerator(null, only);
    expect(gen.next().to_hex()).toBe("#123456");
  });
});

// ---------------------------------------------------------------------------
// Section.get_dict
// ---------------------------------------------------------------------------
describe("Section.get_dict", () => {
  it("raises ValueError when no video is assigned, like Python", () => {
    // Python: "Section 'mysec' cannot be exported as dict, it does not have
    // a video path assigned to it"
    const s = new Section("default.normal", null, "mysec", false);
    expect(() => s.get_dict("/tmp")).toThrowError(
      /Section 'mysec' cannot be exported as dict, it does not have a video path assigned to it/,
    );
    try {
      s.get_dict("/tmp");
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });
});

// ---------------------------------------------------------------------------
// Scene.camera / interactive_embed.get_embedded_method / .ipython pieces
// ---------------------------------------------------------------------------
describe("Scene.camera", () => {
  it("instantiates the given camera class", () => {
    // Python: Scene(camera_class=Camera).camera is a Camera instance
    class FakeCamera {
      marker = "cam";
    }
    const scene = new Scene(null, FakeCamera as unknown as new () => unknown);
    expect(scene.camera).toBeInstanceOf(FakeCamera);
    const def = new Scene();
    expect(def.camera).toBeDefined();
  });
});

describe("Scene interactive_embed pieces", () => {
  it("get_embedded_method queues MethodWithArgs with method+args", () => {
    // Python: get_embedded_method(name) pushes MethodWithArgs(method, args, kwargs)
    const scene = new Scene();
    const fn = scene.get_embedded_method("play");
    fn(1, 2);
    expect(scene.queue).toHaveLength(1);
    const item = scene.queue[0]! as unknown as MethodWithArgs;
    expect(item).toBeInstanceOf(MethodWithArgs);
    expect((item as unknown as { method: unknown }).method).toBe(
      (scene as unknown as Record<string, unknown>)["play"],
    );
    expect((item as unknown as { args: unknown }).args).toEqual([1, 2]);
  });

  it("create_embedded_rerun (interactive_embed.ipython's embedded_rerun) queues rerun and exits", () => {
    // Python embedded_rerun(*a, **k): queue SceneInteractRerun("keyboard") + shell exit
    const scene = new Scene();
    let exited = false;
    const rerun = scene.create_embedded_rerun({ exiter: () => { exited = true; } });
    rerun();
    expect(exited).toBe(true);
    expect(scene.queue).toHaveLength(1);
    expect(scene.queue[0]).toBeInstanceOf(SceneInteractRerun);
  });
});

// ---------------------------------------------------------------------------
// Axes._update_default_configs / LinearTransformationScene.update_default_configs
// ---------------------------------------------------------------------------
describe("default-config merging", () => {
  it("Axes._update_default_configs deep-merges like Python", () => {
    // Python: ({a:1,nested:{x:1,y:2}},) + ({nested:{y:99},b:2},) ->
    // {'a':1,'nested':{'x':1,'y':99},'b':2}
    const d: Array<Record<string, unknown>> = [{ a: 1, nested: { x: 1, y: 2 } }];
    Axes._update_default_configs(d, [{ nested: { y: 99 }, b: 2 }]);
    expect(d[0]).toEqual({ a: 1, nested: { x: 1, y: 99 }, b: 2 });
  });

  it("Axes._update_default_configs skips null passed configs", () => {
    const d: Array<Record<string, unknown>> = [{ a: 1 }];
    Axes._update_default_configs(d, [null as unknown as Record<string, unknown>]);
    expect(d[0]).toEqual({ a: 1 });
  });

  it("LinearTransformationScene.update_default_configs matches Python", () => {
    // Python: None passed config leaves defaults untouched
    const d: Array<Record<string, unknown>> = [{ a: 1 }];
    LinearTransformationScene.update_default_configs(d, [null]);
    expect(d[0]).toEqual({ a: 1 });
    const d2: Array<Record<string, unknown>> = [{ a: 1, n: { x: 1 } }];
    LinearTransformationScene.update_default_configs(d2, [{ n: { x: 2 } }]);
    expect(d2[0]).toEqual({ a: 1, n: { x: 2 } });
  });
});

// ---------------------------------------------------------------------------
// AnimationGroup._setup_scene / Succession._setup_scene /
// Wait.clean_up_from_scene / Add.clean_up_from_scene
// ---------------------------------------------------------------------------
describe("composition _setup_scene", () => {
  it("AnimationGroup._setup_scene forwards to every child", () => {
    // Python: for anim in self.animations: anim._setup_scene(scene)
    const w1 = new Wait();
    const w2 = new Wait();
    const seen: unknown[] = [];
    w1._setup_scene = (s: unknown) => { seen.push(["w1", s]); };
    w2._setup_scene = (s: unknown) => { seen.push(["w2", s]); };
    const group = new AnimationGroup(w1, w2);
    const fake = { tag: "scene" };
    group._setup_scene(fake);
    expect(seen).toEqual([["w1", fake], ["w2", fake]]);
  });

  it("Succession._setup_scene with None is a no-op (scene stays None)", () => {
    // Python: if scene is None: return
    const s = new Succession(new Wait());
    s._setup_scene(null);
    expect(s.scene).toBeNull();
  });

  it("Succession._setup_scene records scene and adds non-introducer mobjects when introducer", () => {
    // Python: introducer succession adds anim.mobject of non-introducer anims
    const m = new Mobject();
    const inner = new Animation(m);
    const s = new Succession(inner, { introducer: true } as unknown as Record<string, unknown>);
    const scene = new Scene();
    s._setup_scene(scene as unknown as Parameters<Succession["_setup_scene"]>[0]);
    expect(s.scene).toBe(scene);
    expect(scene.mobjects).toContain(m);
  });
});

describe("Wait/Add clean_up_from_scene", () => {
  it("Wait.clean_up_from_scene is a no-op", () => {
    // Python: `pass` — scene untouched
    const scene = new Scene();
    const m = new Mobject();
    scene.add(m);
    new Wait(m).clean_up_from_scene(scene as never);
    expect(scene.mobjects).toContain(m);
  });

  it("Add.clean_up_from_scene is a no-op", () => {
    // Python: `pass`
    const scene = new Scene();
    new Add(new Mobject()).clean_up_from_scene(scene as never);
    expect(scene.mobjects).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// TipableVMobject.__init__
// ---------------------------------------------------------------------------
describe("TipableVMobject.__init__", () => {
  it("stores tip defaults like Python", () => {
    // Python: tip_length 0.35 (DEFAULT_ARROW_TIP_LENGTH), normal OUT, tip_style {}
    const t = new TipableVMobject();
    expect(t.tip_length).toBeCloseTo(0.35, 9);
    expectVecClose([...t.normal_vector] as number[], [0, 0, 1]);
    expect(t.tip_style).toEqual({});
  });

  it("honors overrides", () => {
    const t = new TipableVMobject({ tip_length: 0.5, tip_style: { a: 1 } });
    expect(t.tip_length).toBe(0.5);
    expect(t.tip_style).toEqual({ a: 1 });
  });
});

// ---------------------------------------------------------------------------
// integration: batch symbols interacting with previously ported code
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Mobject family + reduce_across_dimension + VGroup chain vs Python", () => {
    // Python: VGroup of two Squares: reduce max dim0 == 1.0-ish envelope.
    // Here: two VMobject squares offset by 2 on x; merged array has 8+ rows.
    const g = new VGroup();
    for (const dx of [0, 2]) {
      const sq = new VMobject();
      sq.set_points(
        array([[0 + dx, 0, 0], [1 + dx, 0, 0], [1 + dx, 1, 0], [0 + dx, 1, 0]]) as NDArray,
      );
      g.add(sq);
    }
    expect(g.reduce_across_dimension((vals) => Math.max(...vals), 0)).toBeCloseTo(3, 9);
    expect(g.reduce_across_dimension((vals) => Math.min(...vals), 0)).toBeCloseTo(0, 9);
    expect(g.get_merged_array("points").shape[0]).toBe(8);
    // remove path inside a group
    const first = g.submobjects[0]!;
    g.remove(first);
    expect(g.submobjects).toHaveLength(1);
    expect(g.reduce_across_dimension((vals) => Math.max(...vals), 0)).toBeCloseTo(3, 9);
  });

  it("AnimationGroup._setup_scene + play() runs Wait/Add against a Scene", () => {
    // Exercises composition + scene + animation cleanup together.
    const scene = new Scene();
    const m = new Mobject();
    const group = new AnimationGroup(new Add(m), new Wait(m));
    const fakeScene = scene as unknown as Parameters<AnimationGroup["_setup_scene"]>[0];
    group._setup_scene(fakeScene);
    // Add is an introducer; Succession-style add not automatic here, but
    // scene.add works and cleanup leaves the mobject (Add.cleanup is pass).
    scene.add(m);
    group.clean_up_from_scene(fakeScene);
    expect(scene.mobjects).toContain(m);
    // remover propagation: base Animation.clean_up removes when remover
    const m2 = new Mobject();
    scene.add(m2);
    const remover = new Animation(m2, { remover: true });
    remover.clean_up_from_scene(scene as never);
    expect(scene.mobjects).not.toContain(m2);
  });

  it("config digest_file feeds frame geometry used with rate/paths code", () => {
    // digest a cfg, then use a squished rate value as a path alpha and
    // compare against the same Python computation.
    const dir = mkdtempSync(join(tmpdir(), "wanim-cfgint-"));
    const path = join(dir, "c.cfg");
    writeFileSync(path, "[CLI]\nframe_width = 10\n");
    const cfg = new ManimConfig();
    cfg.digest_file(path);
    expect(Number(cfg.frame_width)).toBeCloseTo(10, 9);
    const rate = squish_rate_func(((t: number) => t) as (t: number) => number, 0.4, 0.6);
    const alpha = rate(0.5); // 0.5
    const p = path_along_arc(Math.PI);
    const out = p(
      array([[0, 0, 0]]) as NDArray,
      array([[Number(cfg.frame_width), 0, 0]]) as NDArray,
      alpha,
    );
    // Python: path_along_arc(pi)([[0,0,0]], [[10,0,0]], 0.5) == [[5, -5, 0]]
    expectMatClose(out, [[5, -5, 0]]);
  });

  it("ManimColor.from_rgba -> VMobject.set_fill -> get_fill_color round-trips", () => {
    // Python: VMobject().set_fill(from_rgba([255,0,0,255])) -> '#FF0000'
    const v = new VMobject();
    v.set_points(array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) as NDArray);
    v.set_fill(ManimColor.from_rgba([255, 0, 0, 255]), 1);
    expect(v.get_fill_color().to_hex()).toBe("#FF0000");
    expect(v.get_fill_opacity()).toBeCloseTo(1, 9);
    // nth-curve + point-on-bezier agree on the same VMobject data
    const fn = v.get_nth_curve_function(0);
    expectVecClose(fn(0.5), [0.75, 0.5, 0]);
    expect(point_lies_on_bezier(array(fn(0.5)) as NDArray, v.points as NDArray)).toBe(true);
  });

  it("ValueTracker drives get_value into scene Embed queue types", () => {
    // ValueTracker arithmetic chain, then scene queue carries the value.
    const v = new ValueTracker(7.5);
    const w = v.__mod__(2).__mul__(2); // 1.5 * 2 = 3.0
    expect(w.get_value()).toBeCloseTo(3.0, 12);
    const scene = new Scene();
    const fn = scene.get_embedded_method("wait");
    fn(w.get_value());
    const item = scene.queue[0]! as unknown as MethodWithArgs;
    expect((item as unknown as { args: unknown }).args).toEqual([3]);
  });

  it("make_config_parser + digest_parser default frame size matches Python", () => {
    // Python default frame_width 14.222... (from default.cfg)
    const cfg = new ManimConfig();
    cfg.digest_parser(make_config_parser());
    expect(Number(cfg.frame_width)).toBeCloseTo(14.222222222222221, 9);
  });
});
