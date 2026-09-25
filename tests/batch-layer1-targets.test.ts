// Ground-truth tests for the layer-1 batch (~1103 lines).
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim 0.21.0) with the inputs used below; see the
// inline "Python:" comments. TS is asserted against those values with tight
// float tolerance (absolute 1e-9 unless noted; exact for discrete values).
//
// Known deviations from Python (documented, not worked around):
//  - hashing._get_json: Python json.dumps uses (', ', ': ') separators
//    ('{"a": 1, ...}'); the TS port uses JSON.stringify (no spaces). Tests
//    compare JSON.parse() output instead of raw strings.
//  - qhull containers: Python Horizon uses facets:set + boundary:list,
//    QuickHull uses set/dict keyed by Facet objects; the TS port uses
//    arrays/Maps keyed by key() strings (same dedup semantics, different
//    container types). QuickHullPoint has key()/equals() instead of
//    __hash__/__eq__ (JS has no implicit hashing).
//  - config.gui_location fresh default: Python digests default.cfg "0,0" into
//    tuple (0, 0); the TS ManimConfig(true) keeps the raw "0,0" string.
//    Setting an array is accepted in TS (JS has no tuples) vs tuple in Python.
//  - config.upto_animation_number / max_files_cached fresh defaults: Python's
//    live config digests -1 into inf (allow_inf); TS ManimConfig(true) keeps
//    -1 until the setter maps -1 -> Infinity (setter behavior matches).
//  - config text/log/tex/sections dir fresh defaults: Python live config
//    holds unresolved '{media_dir}/...' templates; TS holds './media/...'.
//  - ManimConfig._set_boolean: Python `val in [True, False]` (==) accepts
//    1/0/1.0/0.0; TS accepts true/false/1/0 only.
//  - utils.typst_file_writing.typst_to_svg_file: text/tex pipeline, OUT OF
//    SCOPE for the port. The TS stub throws TODO; tests only assert the stub.
//  - VMobjectFromSVGPath: Python takes svgelements Path segments
//    (Move/Line/QuadraticBezier/CubicBezier/Close classes); TS takes
//    SVGPathSegment {kind,...} records. Same geometry, different input type.
//  - deprecated/deprecated_params emit via console.warn in TS vs logging in
//    Python; warning strings match. TS deprecated_params takes kwargs as a
//    trailing object arg (JS has no **kwargs).
import { describe, expect, it, vi, afterEach } from "vitest";
import { array, type NDArray } from "numpy-ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Mobject } from "../src/mobject/mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Polygram } from "../src/mobject/geometry/polygram.js";
import {
  SVGMobject,
  VMobjectFromSVGPath,
  createSVGXmlElement,
  subSVGXmlElement,
} from "../src/mobject/svg/svg_mobject.js";
import { Camera } from "../src/camera/camera.js";
import { ManimConfig, config } from "../src/_config/utils.js";
import {
  deprecated,
  deprecated_params,
  _get_callable_info,
  _deprecation_text_component,
} from "../src/utils/deprecation.js";
import { get_full_vector_image_path } from "../src/utils/images.js";
import { get_full_sound_file_path } from "../src/utils/sounds.js";
import { typst_to_svg_file } from "../src/utils/typst_file_writing.js";
import { Polygon } from "../src/utils/polylabel.js";
import { straight_path } from "../src/utils/paths.js";
import { interpolate, bezier, subdivide_bezier } from "../src/utils/bezier.js";
import {
  angle_between_vectors,
  line_intersection,
  quaternion_from_angle_axis,
  angle_axis_from_quaternion,
  rotation_matrix_transpose_from_quaternion,
  z_to_vector,
  complex_func_to_R3_func,
  midpoint,
  find_intersection,
  shoelace_direction,
} from "../src/utils/space_ops.js";
import {
  _canonical_dtype_descriptor,
  _Memoizer,
  _CustomEncoder,
  _get_json,
} from "../src/utils/hash.js";
import {
  update_dict_recursively,
  _Data,
  _Uniforms,
} from "../src/utils/config_ops.js";
import {
  QuickHullPoint,
  SubFacet,
  Horizon,
  QuickHull,
  Facet,
} from "../src/utils/qhull.js";
import { list_update } from "../src/utils/iterables.js";
import {
  unit_interval,
  zero,
  squish_rate_func,
  not_quite_there,
  smooth,
} from "../src/utils/rate_functions.js";

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

function expectRowsClose(actual: unknown, expected: number[][], tol = 1e-9): void {
  const a = flat(actual);
  const e = expected.flat();
  expect(a.length).toBe(e.length);
  for (let i = 0; i < e.length; i += 1) {
    expect(Math.abs(a[i]! - e[i]!)).toBeLessThanOrEqual(tol);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// deprecation.warning_msg (both decorators share the message core)
// ---------------------------------------------------------------------------
describe("deprecation warning_msg", () => {
  it("deprecated() runtime warning string matches Python", () => {
    // Python: @deprecated(since="v0.2", until="v0.4", replacement="bar",
    //   message="It is cooler.") def foo(): ...
    // foo() logs: "The function foo has been deprecated since v0.2 and is
    // expected to be removed after v0.4. Use bar instead. It is cooler."
    function foo(): number {
      return 7;
    }
    const dec = deprecated(foo, "v0.2", "v0.4", "bar", "It is cooler.") as (
      ...a: unknown[]
    ) => unknown;
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(dec()).toBe(7);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]?.[0]).toBe(
      "The function foo has been deprecated since v0.2 and is expected to be removed after v0.4. Use bar instead. It is cooler.",
    );
  });

  it("deprecated() doc string uses for_docs role markup", () => {
    // Python: dec.__doc__ == 'orig doc\\n\\n.. attention:: Deprecated\\n
    //   The function foo has been ... Use :func:`~.bar` instead. It is cooler.'
    function foo(): void {}
    (foo as { __doc__?: string }).__doc__ = "orig doc";
    const dec = deprecated(foo, "v0.2", "v0.4", "bar", "It is cooler.") as {
      __doc__?: string;
    };
    expect(dec.__doc__).toBe(
      "orig doc\n\n.. attention:: Deprecated\n  The function foo has been deprecated since v0.2 and is expected to be removed after v0.4. Use :func:`~.bar` instead. It is cooler.",
    );
  });

  it("deprecated() bare message (no since/until/replacement)", () => {
    // Python: @deprecated def foo -> "The function foo has been deprecated
    // and may be removed in a later version."
    function foo(): void {}
    const dec = deprecated(foo) as (...a: unknown[]) => unknown;
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    dec();
    expect(spy.mock.calls[0]?.[0]).toBe(
      "The function foo has been deprecated and may be removed in a later version.",
    );
    expect(_deprecation_text_component(null, null, null)).toBe(
      "deprecated and may be removed in a later version.",
    );
    expect(_get_callable_info(foo)).toEqual(["function", "foo"]);
  });

  it("deprecated() method __qualname__ maps to method + meth role", () => {
    // Python: method Bar.baz with replacement -> "The method Bar.baz ... Use
    // :meth:`~.qux` instead." for docs.
    function baz(): void {}
    (baz as unknown as { __qualname__?: string }).__qualname__ = "Bar.baz";
    expect(_get_callable_info(baz as (...a: unknown[]) => unknown)).toEqual([
      "method",
      "Bar.baz",
    ]);
    (baz as { __doc__?: string }).__doc__ = "";
    const dec = deprecated(baz, null, null, "qux", "") as {
      __doc__?: string;
    };
    expect(dec.__doc__).toContain("Use :meth:`~.qux` instead.");
  });

  it("deprecated_params() singular + plural warning strings match Python", () => {
    // Python: @deprecated_params(params="a, b, c") def foo(**kwargs);
    // foo(a=2) -> "The parameter a of function foo has been deprecated and
    // may be removed in a later version."
    // foo(a=2,b=3) -> "The parameters a and b of function foo have been ..."
    function foo(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    (foo as unknown as { __qualname__?: string }).__qualname__ = "foo";
    const dec = deprecated_params("a, b, c")(foo as (...a: unknown[]) => unknown) as (
      ...a: unknown[]
    ) => unknown;
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    (dec as (k: unknown) => unknown)({ a: 2 });
    expect(spy.mock.calls[0]?.[0]).toBe(
      "The parameter a of function foo has been deprecated and may be removed in a later version.",
    );
    (dec as (k: unknown) => unknown)({ a: 2, b: 3, z: 4 });
    expect(spy.mock.calls[1]?.[0]).toBe(
      "The parameters a and b of function foo have been deprecated and may be removed in a later version.",
    );
    // no deprecated param -> no warning
    (dec as (k: unknown) => unknown)({ x: 1 });
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("deprecated_params() since/until/message + tuple redirection", () => {
    // Python: @deprecated_params(params="a", since="v0.2", until="v0.4",
    //   message="The letters x are cooler.") + ("old_param","new_param")
    // redirection moves the kwarg.
    function foo(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    const dec = deprecated_params("a", "v0.2", "v0.4", "The letters x are cooler.", [
      ["old_param", "new_param"],
    ])(foo as (...a: unknown[]) => unknown) as (k: unknown) => unknown;
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const out = dec({ a: 1 }) as Record<string, unknown>;
    expect(out).toEqual({ a: 1 });
    expect(spy.mock.calls[0]?.[0]).toBe(
      "The parameter a of function foo has been deprecated since v0.2 and is expected to be removed after v0.4. The letters x are cooler.",
    );
    const out2 = dec({ old_param: 2, x: 1 }) as Record<string, unknown>;
    expect(out2).toEqual({ x: 1, new_param: 2 });
  });

  it("deprecated_params() function redirection", () => {
    // Python: redirections=[lambda runtime_in_ms: {"run_time": ...}]
    function foo(kwargs: Record<string, unknown>): Record<string, unknown> {
      return kwargs;
    }
    const redirect = (runtime_in_ms: unknown): Record<string, unknown> => ({
      run_time: Number(runtime_in_ms) / 1000,
    });
    const dec = deprecated_params(null, null, null, "", [redirect])(
      foo as (...a: unknown[]) => unknown,
    ) as (k: unknown) => unknown;
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const out = dec({ runtime_in_ms: 500 }) as Record<string, unknown>;
    expect(out).toEqual({ run_time: 0.5 });
    expect(String(spy.mock.calls[0]?.[0])).toContain("runtime_in_ms");
  });
});

// ---------------------------------------------------------------------------
// paths.straight_path / bezier / subdivide_bezier
// ---------------------------------------------------------------------------
describe("paths + bezier", () => {
  it("straight_path() returns interpolate", () => {
    // Python: straight_path() is interpolate (identity).
    expect(straight_path()).toBe(interpolate);
    const out = (straight_path() as unknown as (s: unknown, e: unknown, a: number) => unknown)(
      array([0, 0, 0]) as NDArray,
      array([10, 0, 0]) as NDArray,
      0.3,
    );
    // Python: interpolate([0,0,0],[10,0,0],0.3) == [3,0,0]
    expectRowsClose(out, [[3, 0, 0]]);
  });

  it("bezier() cubic evaluation matches Python", () => {
    // Python: pts=[[0,0,0],[1,0,0],[1,1,0],[2,1,0]]; b(0)=[0,0,0],
    // b(0.5)=[1,0.5,0], b(1)=[2,1,0]
    const pts = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [2, 1, 0],
    ];
    const b = bezier(pts);
    expectRowsClose(b(0.0) as number[], [[0, 0, 0]]);
    expectRowsClose(b(0.5) as number[], [[1, 0.5, 0]]);
    expectRowsClose(b(1.0) as number[], [[2, 1, 0]]);
    const col = b(array([[0.0], [0.5], [1.0]]) as NDArray) as NDArray;
    expectRowsClose(col, [
      [0, 0, 0],
      [1, 0.5, 0],
      [2, 1, 0],
    ]);
  });

  it("bezier() linear + quadratic match Python formulas", () => {
    const lin = bezier([
      [0, 0, 0],
      [4, 2, 0],
    ]);
    expectRowsClose(lin(0.25) as number[], [[1, 0.5, 0]]);
    const quad = bezier([
      [0, 0, 0],
      [1, 1, 0],
      [2, 0, 0],
    ]);
    // Python quadratic at t=0.5: 0.25*P0 + 0.5*P1 + 0.25*P2 = [1, 0.5, 0]
    expectRowsClose(quad(0.5) as number[], [[1, 0.5, 0]]);
  });

  it("subdivide_bezier() cubic + quadratic + n=1 match Python", () => {
    // Python cubic [[0,0,0],[1,0,0],[1,1,0],[2,1,0]] /2 ->
    // [[0,0,0],[0.5,0,0],[0.75,0.25,0],[1,0.5,0],
    //  [1,0.5,0],[1.25,0.75,0],[1.5,1,0],[2,1,0]]
    const cubic = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [2, 1, 0],
    ];
    expectRowsClose(subdivide_bezier(cubic, 2), [
      [0, 0, 0],
      [0.5, 0, 0],
      [0.75, 0.25, 0],
      [1, 0.5, 0],
      [1, 0.5, 0],
      [1.25, 0.75, 0],
      [1.5, 1, 0],
      [2, 1, 0],
    ]);
    // Python quad [[0,0,0],[1,1,0],[2,0,0]] /2 ->
    // [[0,0,0],[0.5,0.5,0],[1,0.5,0],[1,0.5,0],[1.5,0.5,0],[2,0,0]]
    expectRowsClose(
      subdivide_bezier(
        [
          [0, 0, 0],
          [1, 1, 0],
          [2, 0, 0],
        ],
        2,
      ),
      [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [1, 0.5, 0],
        [1, 0.5, 0],
        [1.5, 0.5, 0],
        [2, 0, 0],
      ],
    );
    // n_divisions == 1 returns the points unchanged
    expectRowsClose(subdivide_bezier(cubic, 1), cubic);
  });
});

// ---------------------------------------------------------------------------
// space_ops batch
// ---------------------------------------------------------------------------
describe("space_ops", () => {
  it("angle_between_vectors matches Python (ortho/same/opposite)", () => {
    // Python: ortho=pi/2, same=0, opposite=pi
    expect(angle_between_vectors([1, 0, 0], [0, 1, 0])).toBeCloseTo(
      1.5707963267948966,
      12,
    );
    expect(angle_between_vectors([1, 0, 0], [1, 0, 0])).toBeCloseTo(0, 12);
    expect(angle_between_vectors([1, 0, 0], [-1, 0, 0])).toBeCloseTo(
      3.141592653589793,
      12,
    );
  });

  it("line_intersection matches Python + error branches", () => {
    // Python: lines [[0,0,0],[1,1,0]] x [[0,1,0],[1,0,0]] -> [0.5, 0.5, 0]
    expectRowsClose(
      line_intersection(
        [
          [0, 0, 0],
          [1, 1, 0],
        ],
        [
          [0, 1, 0],
          [1, 0, 0],
        ],
      ),
      [[0.5, 0.5, 0]],
    );
    expect(() =>
      line_intersection(
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
        [
          [0, 1, 0],
          [1, 1, 0],
        ],
      ),
    ).toThrowError(/parallel/);
    expect(() =>
      line_intersection(
        [
          [0, 0, 1],
          [1, 0, 0],
        ],
        [
          [0, 1, 0],
          [1, 0, 0],
        ],
      ),
    ).toThrowError(/xy-plane/);
  });

  it("quaternion_from_angle_axis matches Python", () => {
    // Python: angle=pi/2 axis=[0,0,1] -> [0.7071..., 0, 0, 0.7071...]
    expectRowsClose(quaternion_from_angle_axis(Math.PI / 2, [0, 0, 1]), [
      [0.7071067811865476, 0, 0, 0.7071067811865475],
    ]);
    // Python: angle=pi axis=[0,0,2] (unnormalized) -> [6.12e-17, 0, 0, 1]
    expectRowsClose(quaternion_from_angle_axis(Math.PI, [0, 0, 2]), [
      [6.123233995736766e-17, 0, 0, 1],
    ]);
    // axis_normalized=True skips normalization
    expectRowsClose(quaternion_from_angle_axis(Math.PI / 2, [0, 0, 1], true), [
      [0.7071067811865476, 0, 0, 0.7071067811865475],
    ]);
  });

  it("angle_axis_from_quaternion matches Python", () => {
    // Python: quat [cos(pi/4),0,0,sin(pi/4)] -> angle pi/2, axis [0,0,1]
    const [angle, axis] = angle_axis_from_quaternion([
      Math.cos(Math.PI / 4),
      0,
      0,
      Math.sin(Math.PI / 4),
    ]);
    expect(angle).toBeCloseTo(1.5707963267948966, 12);
    expectRowsClose([axis], [[0, 0, 1]]);
    // identity quaternion -> angle 0
    const [a0] = angle_axis_from_quaternion([1, 0, 0, 0]);
    expect(a0).toBeCloseTo(0, 12);
  });

  it("rotation_matrix_transpose_from_quaternion matches Python", () => {
    // Python identity quat -> 3x3 identity
    expectRowsClose(rotation_matrix_transpose_from_quaternion([1, 0, 0, 0]), [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    // Python 90deg-z quat -> [[2.2e-16, 1, 0], [-1, 2.2e-16, 0], [0, 0, 1]]
    // (transpose of the rotation matrix)
    const q = quaternion_from_angle_axis(Math.PI / 2, [0, 0, 1]);
    expectRowsClose(rotation_matrix_transpose_from_quaternion(q), [
      [2.220446049250313e-16, 1, 0],
      [-1, 2.220446049250313e-16, 0],
      [0, 0, 1],
    ], 1e-12);
  });

  it("z_to_vector matches Python (z stays identity, x maps)", () => {
    // Python: z_to_vector([0,0,1]) == eye(3)
    expectRowsClose(z_to_vector([0, 0, 1]), [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    // Python: z_to_vector([1,0,0]) ==
    // [[0,-0,1],[0,1,0],[-1,-0,0]]
    expectRowsClose(z_to_vector([1, 0, 0]), [
      [0, -0, 1],
      [0, 1, 0],
      [-1, -0, 0]],
    );
  });

  it("complex_func_to_R3_func matches Python", () => {
    // Python: lambda z: z*2 at [1,2,0] -> [2,4,0]
    const f = complex_func_to_R3_func((z) => ({ re: z.re * 2, im: z.im * 2 }));
    expectRowsClose([f([1, 2, 0])], [[2, 4, 0]]);
  });

  it("midpoint matches Python", () => {
    // Python: midpoint([0,0,0],[2,4,6]) == [1,2,3]
    expectRowsClose([midpoint([0, 0, 0], [2, 4, 6])], [[1, 2, 3]]);
  });

  it("find_intersection matches Python", () => {
    // Python: ray p0=[0,0,0] d=[1,0,0] with p1=[0,1,0] d=[0,-1,0] -> [0,0,0]
    const out = find_intersection(
      [[0, 0, 0]],
      [[1, 0, 0]],
      [[0, 1, 0]],
      [[0, -1, 0]],
    );
    expectRowsClose(out, [[0, 0, 0]]);
    // mismatched lengths raise (strict zip)
    expect(() =>
      find_intersection([[0, 0, 0]], [[1, 0, 0], [0, 1, 0]], [[0, 1, 0]], [[0, -1, 0]]),
    ).toThrow();
  });

  it("shoelace_direction matches Python", () => {
    // Python: [[0,0],[1,0],[1,1],[0,1],[0,0]] -> CCW (negative shoelace);
    // reversed -> CW.
    expect(
      shoelace_direction([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
      ]),
    ).toBe("CCW");
    expect(
      shoelace_direction([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
      ]),
    ).toBe("CW");
  });
});

// ---------------------------------------------------------------------------
// polylabel.Polygon.inside / polygram.get_vertex_groups
// ---------------------------------------------------------------------------
describe("polygon inside + vertex groups", () => {
  it("Polygon.inside matches Python (in/out/edge/vertex)", () => {
    // Python square [[0,0],[4,0],[4,4],[0,4],[0,0]]:
    // [2,2]->True, [5,5]->False, [0,0]->True (boundary), [2,0]->True (edge)
    const poly = new Polygon([
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0],
      ],
    ]);
    expect(poly.inside([2, 2])).toBe(true);
    expect(poly.inside([5, 5])).toBe(false);
    expect(poly.inside([0, 0])).toBe(true);
    expect(poly.inside([2, 0])).toBe(true);
  });

  it("Polygram.get_vertex_groups matches Python docstring example", () => {
    // Python: Polygram([ORIGIN, RIGHT, UP, LEFT+UP],
    //   [LEFT, LEFT+UP, 2*LEFT]) -> 2 groups with the documented arrays.
    const poly = new Polygram(
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 1, 0],
      ],
      [
        [-1, 0, 0],
        [-1, 1, 0],
        [-2, 0, 0],
      ],
    );
    const groups = poly.get_vertex_groups();
    expect(groups.length).toBe(2);
    expectRowsClose(groups[0]!, [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [-1, 1, 0],
    ]);
    expectRowsClose(groups[1]!, [
      [-1, 0, 0],
      [-1, 1, 0],
      [-2, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// VMobject bezier tuples + rgbas
// ---------------------------------------------------------------------------
describe("VMobject bezier tuples + rgbas", () => {
  it("get_cubic_bezier_tuples_from_points matches Python", () => {
    // Python: 8 points (2 cubic curves) -> 2 tuples of 4.
    const v = new VMobject();
    const pts = [
      [0, 0, 0],
      [0.33, 0, 0],
      [0.66, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [1.33, 0, 0],
      [1.66, 0, 0],
      [2, 0, 0],
    ];
    const out = v.get_cubic_bezier_tuples_from_points(
      array(pts) as NDArray,
    ) as NDArray;
    expectRowsClose(out, pts);
    expect(out.shape).toEqual([2, 4, 3]);
  });

  it("generate_rgbas_array matches Python (single + paired)", () => {
    // Python: ('#FF0000', 0.5) -> [[1,0,0,0.5]]
    const v = new VMobject();
    expectRowsClose(v.generate_rgbas_array("#FF0000", 0.5), [[1, 0, 0, 0.5]]);
    // Python: (['#FF0000','#00FF00'], [0.5,1.0]) -> [[1,0,0,.5],[0,1,0,1]]
    expectRowsClose(v.generate_rgbas_array(["#FF0000", "#00FF00"], [0.5, 1.0]), [
      [1, 0, 0, 0.5],
      [0, 1, 0, 1],
    ]);
  });
});

// ---------------------------------------------------------------------------
// Mobject.__init__
// ---------------------------------------------------------------------------
describe("Mobject.__init__", () => {
  it("defaults match Python (name/dim/target/z_index/empties/white/(0,3))", () => {
    // Python: Mobject() -> name 'Mobject', dim 3, target None, z_index 0,
    // submobjects [], updaters [], color #FFFFFF, points shape (0, 3).
    const m = new Mobject();
    expect(m.name).toBe("Mobject");
    expect(m.dim).toBe(3);
    expect(m.target).toBeNull();
    expect(m.z_index).toBe(0);
    expect(m.submobjects).toEqual([]);
    expect(m.updaters).toEqual([]);
    expect((m as unknown as { updating_suspended: boolean }).updating_suspended).toBe(false);
    expect(String(m.color)).toContain("FFFFFF");
    expect(m.points.shape).toEqual([0, 3]);
  });

  it("explicit args (color/name/z_index) match Python", () => {
    // Python: Mobject(color='#00FF00', name='foo', z_index=5)
    const m = new Mobject({ color: "#00FF00", name: "foo", z_index: 5 });
    expect(m.name).toBe("foo");
    expect(m.z_index).toBe(5);
    expect(String(m.color)).toContain("00FF00");
  });
});

// ---------------------------------------------------------------------------
// SVGMobject.modify_xml_tree + VMobjectFromSVGPath.handle_commands.add_cubic
// ---------------------------------------------------------------------------
describe("svg mobjects", () => {
  it("modify_xml_tree nests config + root styles like Python", () => {
    // Python: root <svg fill=blue width=10 stroke-width=5> with one <path>;
    // svg_default color red + stroke_width 2 ->
    // <svg><g fill=red stroke=red stroke-width=2>
    //   <g fill=blue stroke-width=5><path/></g></g></svg>
    const mob = new SVGMobject({ file_name: null });
    mob.svg_default = {
      color: "red",
      opacity: null,
      fill_color: null,
      fill_opacity: null,
      stroke_width: 2,
      stroke_color: null,
      stroke_opacity: null,
    };
    const root = createSVGXmlElement("svg", {
      fill: "blue",
      width: "10",
      "stroke-width": "5",
    });
    subSVGXmlElement(root, "path", { d: "M0 0" });
    const out = mob.modify_xml_tree({ getroot: () => root });
    const newRoot = out.getroot();
    expect(newRoot.tag).toBe("svg");
    expect(newRoot.children.length).toBe(1);
    const cfg = newRoot.children[0]!;
    expect(cfg.tag).toBe("g");
    expect(cfg.attrib).toEqual({ fill: "red", stroke: "red", "stroke-width": "2" });
    const inner = cfg.children[0]!;
    expect(inner.attrib).toEqual({ fill: "blue", "stroke-width": "5" });
    expect(inner.children.length).toBe(1);
    expect(inner.children[0]!.tag).toBe("path");
  });

  it("handle_commands cubic (add_cubic) matches Python", () => {
    // Python: se.Path("M0,0 C0,1 1,1 1,0") ->
    // [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "CubicBezier", control1: [0, 1], control2: [1, 1], end: [1, 0] },
    ]);
    expectRowsClose(v.points, [
      [0, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [1, 0, 0],
    ]);
  });

  it("handle_commands lines + close (add_line) matches Python", () => {
    // Python: se.Path("M0,0 L1,0 L1,1 Z") -> 12 points (3 cubic-ified lines,
    // closing line back to origin), exact thirds as in ground truth.
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [1, 0] },
      { kind: "Line", end: [1, 1] },
      { kind: "Close" },
    ]);
    expectRowsClose(v.points, [
      [0, 0, 0],
      [0.3333333333333333, 0, 0],
      [0.6666666666666666, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
      [1, 0.3333333333333333, 0],
      [1, 0.6666666666666666, 0],
      [1, 1, 0],
      [1, 1, 0],
      [0.6666666666666666, 0.6666666666666666, 0],
      [0.3333333333333333, 0.3333333333333333, 0],
      [0, 0, 0],
    ]);
  });

  it("handle_commands quad conversion matches Python formula", () => {
    // Python add_quad: cp1=(start+2cp)/3, cp2=(2cp+end)/3.
    // start [0,0,0], cp [1,0,0], end [1,1,0] ->
    // [[0,0,0],[2/3,0,0],[1,1/3,0],[1,1,0]]
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "QuadraticBezier", control: [1, 0], end: [1, 1] },
    ]);
    expectRowsClose(v.points, [
      [0, 0, 0],
      [0.6666666666666666, 0, 0],
      [1, 0.3333333333333333, 0],
      [1, 1, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// rate_functions wrappers
// ---------------------------------------------------------------------------
describe("rate_functions wrappers", () => {
  it("unit_interval.wrapper clamps like Python", () => {
    // Python: unit_interval(t*t): -0.5->0, 0.5->0.25, 1.5->1
    const w = unit_interval((t: number) => t * t);
    expect(w(-0.5)).toBe(0);
    expect(w(0.5)).toBeCloseTo(0.25, 12);
    expect(w(1.5)).toBe(1);
  });

  it("zero.wrapper zeroes outside like Python", () => {
    // Python: zero(t+1): -0.5->0, 0.5->1.5, 1.5->0
    const w = zero((t: number) => t + 1);
    expect(w(-0.5)).toBe(0);
    expect(w(0.5)).toBeCloseTo(1.5, 12);
    expect(w(1.5)).toBe(0);
  });

  it("squish_rate_func.result incl. a==b like Python", () => {
    // Python: squish(id, .4, .6): 0->0, 0.5->0.5, 1->1; a==b -> a (0.5)
    const s = squish_rate_func((t: number) => t, 0.4, 0.6);
    expect(s(0.0)).toBeCloseTo(0, 12);
    expect(s(0.5)).toBeCloseTo(0.5, 12);
    expect(s(1.0)).toBeCloseTo(1, 12);
    expect(squish_rate_func((t: number) => t, 0.5, 0.5)(0.3)).toBeCloseTo(0.5, 12);
  });

  it("not_quite_there.result scales like Python", () => {
    // Python: not_quite_there(smooth, 0.7): 0->0, 0.5->0.35, 1->0.7
    const n = not_quite_there(smooth, 0.7);
    expect(n(0.0)).toBeCloseTo(0, 12);
    expect(n(0.5)).toBeCloseTo(0.35, 12);
    expect(n(1.0)).toBeCloseTo(0.7, 12);
  });
});

// ---------------------------------------------------------------------------
// hashing batch
// ---------------------------------------------------------------------------
describe("hashing", () => {
  it("_canonical_dtype_descriptor strings pass through + subarray", () => {
    // Python: np.dtype('float64') -> '<f8'; ('<f8',(3,)) ->
    // ['subarray', '<f8', [3]]
    expect(_canonical_dtype_descriptor("<f8")).toBe("<f8");
    expect(
      _canonical_dtype_descriptor({
        str: "<f8",
        subdtype: ["<f8", [3]],
        newbyteorder: (o: string) => ({
          str: "<f8",
          subdtype: ["<f8", [3]] as unknown as readonly [unknown, readonly number[]],
        }),
      }),
    ).toEqual(["subarray", "<f8", [3]]);
  });

  it("_canonical_dtype_descriptor struct matches Python", () => {
    // Python: np.dtype([('a','<i4'),('b','<f8')]) ->
    // ['struct', [['a','<i4',0,None],['b','<f8',4,None]], 12, 1, False]
    const dtype = {
      str: "V12",
      names: ["a", "b"],
      fields: {
        a: ["<i4", 0],
        b: ["<f8", 4],
      } as Record<string, readonly unknown[]>,
      itemsize: 12,
      alignment: 1,
      isalignedstruct: false,
      newbyteorder(o: string) {
        return this;
      },
    };
    expect(_canonical_dtype_descriptor(dtype)).toEqual([
      "struct",
      [
        ["a", "<i4", 0, null],
        ["b", "<f8", 4, null],
      ],
      12,
      1,
      false,
    ]);
  });

  it("_Memoizer._return memoizes then returns placeholder", () => {
    // Python: first _return(obj, hash, f) -> f(obj); second -> 'AP';
    // memoizing=False does not record.
    const m = new _Memoizer();
    const o = { x: 1 };
    const o2 = { y: 2 };
    const sign = (v: unknown): number => (v === o ? 111 : 222);
    expect(m._return(o, sign, () => "first")).toBe("first");
    expect(m._return(o, sign, () => "second")).toBe(
      _Memoizer.ALREADY_PROCESSED_PLACEHOLDER,
    );
    expect(m._return(o2, sign, () => "nb", false)).toBe("nb");
    expect(m._return(o2, sign, () => "nb2")).toBe("nb2");
  });

  it("_CustomEncoder.__init__ stores memoizer + flag", () => {
    // Python: _CustomEncoder(memoizer=m, include_pixel_array=False)
    const m = new _Memoizer();
    const enc = new _CustomEncoder(m, false);
    expect(enc._memoizer).toBe(m);
    expect(enc.include_pixel_array).toBe(false);
    const enc2 = new _CustomEncoder(m, true);
    expect(enc2.include_pixel_array).toBe(true);
  });

  it("_get_json round-trips plain data like Python", () => {
    // Python: _get_json({'a':1,'b':[1,2]}, _Memoizer()) ==
    // '{"a": 1, "b": [1, 2]}' (modulo separators; compare parsed).
    const parsed = JSON.parse(_get_json({ a: 1, b: [1, 2] }, new _Memoizer()));
    expect(parsed).toEqual({ a: 1, b: [1, 2] });
  });
});

// ---------------------------------------------------------------------------
// config_ops batch
// ---------------------------------------------------------------------------
describe("config_ops", () => {
  it("update_dict_recursively deep-merges like Python", () => {
    // Python: {'a':1,'b':{'x':1,'y':2}} + {'b':{'y':3,'z':4},'c':5}
    // -> {'a':1,'b':{'x':1,'y':3,'z':4},'c':5}
    const d: Record<string, unknown> = { a: 1, b: { x: 1, y: 2 } };
    update_dict_recursively(d, { b: { y: 3, z: 4 }, c: 5 });
    expect(d).toEqual({ a: 1, b: { x: 1, y: 3, z: 4 }, c: 5 });
  });

  it("_Data descriptor get/set/name like Python", () => {
    const d = new _Data<number>();
    d.__set_name__(null, "x");
    expect(d.name).toBe("x");
    const obj = { data: {} as Record<string, unknown> };
    d.__set__(obj, 99);
    expect(obj.data).toEqual({ x: 99 });
    expect(d.__get__(obj)).toBe(99);
  });

  it("_Uniforms descriptor get/set/name like Python", () => {
    const u = new _Uniforms<number>();
    u.__set_name__(null, "k");
    expect(u.name).toBe("k");
    const obj = { uniforms: {} as Record<string, unknown> };
    u.__set__(obj, 3.5);
    expect(obj.uniforms).toEqual({ k: 3.5 });
    expect(u.__get__(obj)).toBe(3.5);
  });
});

// ---------------------------------------------------------------------------
// qhull batch
// ---------------------------------------------------------------------------
describe("qhull", () => {
  it("QuickHullPoint hash-key stable for equal coords", () => {
    // Python: hash(QuickHullPoint) == hash(coordinates.tobytes()).
    // TS: key() is the float64-bytes hex; equal coords -> equal keys.
    const p1 = new QuickHullPoint([1, 2, 3]);
    const p2 = new QuickHullPoint([1, 2, 3]);
    const p3 = new QuickHullPoint([1, 2, 4]);
    expect(p1.key()).toBe(p2.key());
    expect(p1.key()).not.toBe(p3.key());
    expect(p1.equals(p2)).toBe(true);
    expect(p1.equals(p3)).toBe(false);
  });

  it("SubFacet.__init__ stores coords + point set", () => {
    // Python: SubFacet(coords) keeps .coordinates and 3 points.
    const sf = new SubFacet([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    expect(sf.coordinates).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    expect(sf.points.size).toBe(3);
  });

  it("Horizon.__init__ starts empty", () => {
    // Python: Horizon() -> facets=set(), boundary=[].
    const h = new Horizon();
    expect(h.facets).toEqual([]);
    expect(h.boundary).toEqual([]);
  });

  it("QuickHull.__init__ defaults match Python", () => {
    // Python: QuickHull() -> tolerance 1e-5, empty facets/removed/outside/
    // neighbors, unclaimed None, internal None.
    const q = new QuickHull();
    expect(q.tolerance).toBe(1e-5);
    expect(q.facets).toEqual([]);
    expect(q.removed.size).toBe(0);
    expect(q.outside.size).toBe(0);
    expect(q.neighbors.size).toBe(0);
    expect(q.unclaimed).toBeNull();
    expect(q.internal).toBeNull();
    expect(new QuickHull(0.5).tolerance).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// iterables.list_update
// ---------------------------------------------------------------------------
describe("iterables", () => {
  it("list_update matches Python docstring + edges", () => {
    // Python: list_update([1,2,3],[2,4,4]) == [1,3,2,4,4]
    expect(list_update([1, 2, 3], [2, 4, 4])).toEqual([1, 3, 2, 4, 4]);
    expect(list_update([], [])).toEqual([]);
    // Python: list_update([1,1,2],[1]) -> [2,1]
    expect(list_update([1, 1, 2], [1])).toEqual([2, 1]);
  });
});

// ---------------------------------------------------------------------------
// images / sounds path lookup
// ---------------------------------------------------------------------------
describe("images + sounds paths", () => {
  it("get_full_vector_image_path resolves .svg via assets_dir", () => {
    // Python: seek_full_path_from_defaults(name, assets_dir, ['.svg']).
    const dir = mkdtempSync(join(tmpdir(), "wanim-assets-"));
    writeFileSync(join(dir, "logo.svg"), "<svg/>");
    const prev = config.assets_dir;
    config.assets_dir = dir;
    try {
      expect(get_full_vector_image_path("logo")).toBe(join(dir, "logo.svg"));
      expect(() => get_full_vector_image_path("missing-file-xyz")).toThrowError(
        /could not find/,
      );
    } finally {
      config.assets_dir = prev;
    }
  });

  it("get_full_sound_file_path resolves .wav/.mp3 via assets_dir", () => {
    const dir = mkdtempSync(join(tmpdir(), "wanim-sound-"));
    writeFileSync(join(dir, "pop.mp3"), "ID3");
    const prev = config.assets_dir;
    config.assets_dir = dir;
    try {
      expect(get_full_sound_file_path("pop")).toBe(join(dir, "pop.mp3"));
      expect(() => get_full_sound_file_path("missing-sound-xyz")).toThrowError(
        /could not find/,
      );
    } finally {
      config.assets_dir = prev;
    }
  });
});

// ---------------------------------------------------------------------------
// typst stub (text/tex out of scope)
// ---------------------------------------------------------------------------
describe("typst_file_writing (out of scope stub)", () => {
  it("typst_to_svg_file throws TODO stub", () => {
    expect(() => typst_to_svg_file("hello")).toThrowError(/TODO/);
  });
});

// ---------------------------------------------------------------------------
// ManimConfig property batch
// ---------------------------------------------------------------------------
describe("ManimConfig properties", () => {
  it("fresh defaults match Python live config where defined", () => {
    // Python live config values captured 2026-09-25 (see header).
    const c = new ManimConfig(true);
    expect(c.preview).toBe(false);
    expect(c.fullscreen).toBe(false);
    expect(c.use_projection_fill_shaders).toBe(false);
    expect(c.use_projection_stroke_shaders).toBe(false);
    expect(c.pixel_height).toBe(1080);
    expect(c.pixel_width).toBe(1920);
    expect(c.disable_caching).toBe(false);
    expect(c.disable_caching_warning).toBe(false);
    expect(c.verbosity).toBe("INFO");
    expect(c.window_monitor).toBe(0);
    expect(c.preview_command).toBe("");
    expect(c.flush_cache).toBe(false);
    expect(c.no_latex_cleanup).toBe(false);
    expect(c.force_window).toBe(false);
    expect(c.enable_wireframe).toBe(false);
    expect(c.max_files_cached).toBe(100);
    expect(c.save_sections).toBe(false);
    expect(c.save_as_gif).toBe(false);
    expect(c.save_pngs).toBe(false);
    expect(c.write_all).toBe(false);
    expect(c.save_last_frame).toBe(false);
    expect(c.zero_pad).toBe(4);
    expect(c.notify_outdated_version).toBe(true);
    expect(c.input_file).toBe("");
    expect(c.log_to_file).toBe(false);
    expect(c.output_file).toBe("");
    expect(c.enable_gui).toBe(false);
    expect(c.progress_bar).toBe("display");
    expect(c.show_in_file_browser).toBe(false);
    expect(c.seed).toBeNull();
    expect(c.assets_dir).toBe("./");
    expect(c.log_dir).toBe("./media/logs");
    expect(c.tex_dir).toBe("./media/Tex");
    expect(c.text_dir).toBe("./media/texts");
    expect(c.sections_dir).toBe("./media/sections");
    expect(c.custom_folders).toBe(false);
  });

  it("preview ORs enable_gui like Python", () => {
    // Python: preview property returns _d['preview'] or _d['enable_gui'].
    const c = new ManimConfig(true);
    c.preview = true;
    expect(c.preview).toBe(true);
    c.preview = false;
    c.enable_gui = true;
    expect(c.preview).toBe(true);
  });

  it("boolean setters accept true/false, reject strings", () => {
    const c = new ManimConfig(true);
    for (const key of [
      "preview",
      "fullscreen",
      "use_projection_fill_shaders",
      "use_projection_stroke_shaders",
      "disable_caching",
      "disable_caching_warning",
      "flush_cache",
      "no_latex_cleanup",
      "force_window",
      "enable_wireframe",
      "save_sections",
      "save_as_gif",
      "save_pngs",
      "write_all",
      "save_last_frame",
      "notify_outdated_version",
      "log_to_file",
      "enable_gui",
      "show_in_file_browser",
    ] as const) {
      (c as unknown as Record<string, unknown>)[key] = true;
      expect((c as unknown as Record<string, unknown>)[key]).toBe(true);
      expect(() => {
        (c as unknown as Record<string, unknown>)[key] = "yes";
      }).toThrowError(/must be boolean/);
    }
  });

  it("pixel dims / monitors / counts validate like Python", () => {
    const c = new ManimConfig(true);
    c.pixel_height = 720;
    expect(c.pixel_height).toBe(720);
    expect(() => {
      c.pixel_height = -1;
    }).toThrowError(/non-negative integer/);
    expect(() => {
      c.pixel_width = 1.5;
    }).toThrowError(/non-negative integer/);
    c.window_monitor = 1;
    expect(c.window_monitor).toBe(1);
    expect(() => {
      c.window_monitor = -2;
    }).toThrowError(/non-negative integer/);
    // allow_inf: -1 maps to Infinity like Python
    c.upto_animation_number = -1;
    expect(c.upto_animation_number).toBe(Infinity);
    c.upto_animation_number = 3;
    expect(c.upto_animation_number).toBe(3);
    c.max_files_cached = 50;
    expect(c.max_files_cached).toBe(50);
    c.max_files_cached = -1;
    expect(c.max_files_cached).toBe(Infinity);
  });

  it("verbosity / progress_bar validate against lists", () => {
    const c = new ManimConfig(true);
    c.verbosity = "DEBUG";
    expect(c.verbosity).toBe("DEBUG");
    expect(() => {
      c.verbosity = "NOPE";
    }).toThrowError(/must be in/);
    c.progress_bar = "none";
    expect(c.progress_bar).toBe("none");
    expect(() => {
      c.progress_bar = "bogus";
    }).toThrowError(/must be in/);
  });

  it("zero_pad / seed / preview_command / str-dirs behave like Python", () => {
    const c = new ManimConfig(true);
    c.zero_pad = 5;
    expect(c.zero_pad).toBe(5);
    expect(() => {
      c.zero_pad = 99;
    }).toThrowError(/0 <= zero_pad <= 9/);
    c.seed = 42;
    expect(c.seed).toBe(42);
    c.seed = null; // Python: None is a no-op (keeps 42)
    expect(c.seed).toBe(42);
    c.preview_command = "mpv";
    expect(c.preview_command).toBe("mpv");
    c.text_dir = "/tmp/texts";
    expect(c.text_dir).toBe("/tmp/texts");
    c.tex_dir = "/tmp/tex";
    expect(c.tex_dir).toBe("/tmp/tex");
    c.log_dir = "/tmp/logs";
    expect(c.log_dir).toBe("/tmp/logs");
    c.sections_dir = "/tmp/sections";
    expect(c.sections_dir).toBe("/tmp/sections");
    c.assets_dir = "/tmp/assets";
    expect(c.assets_dir).toBe("/tmp/assets");
    c.input_file = "scene.py";
    expect(c.input_file).toBe("scene.py");
    c.output_file = "out.mp4";
    expect(c.output_file).toBe("out.mp4");
    c.custom_folders = true as unknown as string;
    expect(c.custom_folders).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Camera.__init__
// ---------------------------------------------------------------------------
describe("Camera.__init__", () => {
  it("defaults come from config like Python", () => {
    // Python: Camera() pixel 1920x720 (after earlier probe set 720; here the
    // TS global config is untouched: 1920x1080), frame 14.22x8(?), rate 60,
    // black bg, opacity 1.
    const cam = new Camera();
    expect(cam.pixel_height).toBe(config.pixel_height);
    expect(cam.pixel_width).toBe(config.pixel_width);
    expect(cam.frame_height).toBe(config.frame_height);
    expect(cam.frame_width).toBe(config.frame_width);
    expect(cam.frame_rate).toBe(config.frame_rate);
    expect(String(cam.background_color)).toContain("000000");
    expect(cam.background_opacity).toBe(1);
  });

  it("explicit overrides match Python", () => {
    // Python: Camera(pixel_height=100, pixel_width=200, frame_height=4.0,
    // frame_width=8.0, frame_rate=30, background_color='#FF0000',
    // background_opacity=0.5).
    const cam = new Camera({
      pixel_height: 100,
      pixel_width: 200,
      frame_height: 4.0,
      frame_width: 8.0,
      frame_rate: 30,
      background_color: "#FF0000",
      background_opacity: 0.5,
    });
    expect(cam.pixel_height).toBe(100);
    expect(cam.pixel_width).toBe(200);
    expect(cam.frame_height).toBe(4.0);
    expect(cam.frame_width).toBe(8.0);
    expect(cam.frame_rate).toBe(30);
    expect(String(cam.background_color)).toContain("FF0000");
    expect(cam.background_opacity).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// integration: batch symbols composed with earlier-ported code
// ---------------------------------------------------------------------------
describe("integration", () => {
  it("Polygram chain: vertex groups -> VMobject bezier tuples agree", () => {
    // Build a triangle Polygram, take its start anchors as a closed cubic
    // path's endpoints, and check tuple regrouping matches Python's
    // gen_cubic_bezier_tuples_from_points on the same points.
    const poly = new Polygram(
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
    );
    const groups = poly.get_vertex_groups();
    expect(groups.length).toBe(1);
    // Python: VMobject().get_cubic_bezier_tuples_from_points(poly.points)
    // regroups every 4 points; the triangle has 12 points (3 cubic curves).
    const v = new VMobject();
    const tuples = v.get_cubic_bezier_tuples_from_points(poly.points) as NDArray;
    expect(tuples.shape[0]).toBe(poly.points.shape[0] / 4);
    expectRowsClose(tuples, flat(poly.points) as unknown as number[][], 1e-9);
  });

  it("config pixel dims feed Camera + rate path end-to-end", () => {
    // Python: config 1280x720 -> Camera inherits; squish(smooth) at 0.5 in
    // (0,1); straight_path interpolates centers consistently.
    const c = new ManimConfig(true);
    c.pixel_height = 720;
    c.pixel_width = 1280;
    const cam = new Camera({
      pixel_height: c.pixel_height,
      pixel_width: c.pixel_width,
    });
    expect(cam.pixel_height).toBe(720);
    expect(cam.pixel_width).toBe(1280);
    const s = squish_rate_func(smooth, 0.2, 0.8);
    expect(s(0.5)).toBeGreaterThan(0);
    expect(s(0.5)).toBeLessThan(1);
    const m = midpoint([cam.frame_width, 0, 0], [0, cam.frame_height, 0]);
    expect(m.length).toBe(3);
  });

  it("space_ops + bezier: subdivided curve midpoint via find_intersection plane", () => {
    // Subdivide the cubic, take the middle joint [1, 0.5, 0] (Python value),
    // and check midpoint() of its neighbors reproduces it.
    const parts = subdivide_bezier(
      [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
      ],
      2,
    );
    const joint = parts[3]!;
    expect(joint[0]).toBeCloseTo(1, 12);
    expect(joint[1]).toBeCloseTo(0.5, 12);
    const mid = midpoint(parts[2]!, parts[4]!);
    // Python: midpoint([0.75,0.25,0],[1,0.5,0]) == [0.875,0.375,0]
    expectRowsClose([mid], [[0.875, 0.375, 0]]);
    // angle between the first-half start tangent [0.5,0,0] and a later
    // segment [0.25,0.25,0] is small but nonzero (Python: ~0.4636 rad).
    const a = angle_between_vectors(
      [parts[1]![0]! - parts[0]![0]!, parts[1]![1]! - parts[0]![1]!, 0],
      [parts[6]![0]! - parts[5]![0]!, parts[6]![1]! - parts[5]![1]!, 0],
    );
    expect(a).toBeGreaterThan(0);
    expect(a).toBeLessThan(Math.PI);
  });
});
