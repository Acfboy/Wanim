import { describe, expect, it, vi, afterEach } from "vitest";
import { array } from "numpy-ts";
import { _Data } from "../src/utils/config_ops.js";
import {
  uniq_chain,
  listify,
  resize_preserving_order,
  batch_by_property,
  list_difference_update,
  hash_obj,
} from "../src/utils/iterables.js";
import {
  _get_callable_info,
  _deprecation_text_component,
  redirect_params,
  deprecated_params,
} from "../src/utils/deprecation.js";
import {
  interpolate,
  _get_subdivision_matrix,
  get_quadratic_approximation_of_cubic,
} from "../src/utils/bezier.js";
import { normalize_along_axis } from "../src/utils/space_ops.js";
import { Polygon, Cell } from "../src/utils/polylabel.js";
import { RendererType } from "../src/constants.js";
import {
  _convert_point_to_3d,
  SVGMobject,
  VMobjectFromSVGPath,
} from "../src/mobject/svg/svg_mobject.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { Mobject } from "../src/mobject/mobject.js";
import { DefaultSectionType } from "../src/scene/section.js";
import { sigmoid } from "../src/utils/simple_functions.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function flat(value: unknown): number[] {
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

function rowsOf(value: unknown): number[][] {
  return ((value as { toArray: () => unknown }).toArray() as unknown[][]).map(
    (r) => (Array.isArray(r) ? r : [r]).map(Number),
  );
}

function expectClose(actual: unknown, expected: readonly number[], relTol = 1e-9): void {
  const values = flat(actual);
  expect(values).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const tol = Math.max(1e-12, Math.abs(expected[i]!) * relTol);
    expect(Math.abs(values[i]! - expected[i]!)).toBeLessThanOrEqual(tol);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
  delete (globalThis as Record<string, unknown>)["__wanim_renderer__"];
});

// ---------------------------------------------------------------------------
// manim.utils.config_ops._Data
// Python: descriptor reading/writing obj.data[name] after __set_name__.
// ---------------------------------------------------------------------------
describe("_Data", () => {
  it("reads and writes obj.data[name]", () => {
    const d = new _Data<number>();
    d.__set_name__({}, "myattr");
    const obj = { data: { myattr: 5 } };
    expect(d.__get__(obj)).toBe(5);
    d.__set__(obj, 9);
    expect(obj.data["myattr"]).toBe(9);
  });
  it("starts with an empty name", () => {
    expect(new _Data().name).toBe("");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.uniq_chain
// Python: tuple(uniq_chain([1,2],[2,3],[1,4,4])) == (1,2,3,4)
// ---------------------------------------------------------------------------
describe("uniq_chain", () => {
  it("yields unique elements in order", () => {
    expect([...uniq_chain([1, 2], [2, 3], [1, 4, 4])]).toEqual([1, 2, 3, 4]);
  });
  it("handles empty inputs", () => {
    expect([...uniq_chain([], [])]).toEqual([]);
    expect([...uniq_chain([1, 1, 1])]).toEqual([1]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.listify
// Python: listify("str")==['str']; listify((1,2))==[1,2]; listify(5)==[5]
// ---------------------------------------------------------------------------
describe("listify", () => {
  it("wraps strings instead of splitting them", () => {
    expect(listify("str")).toEqual(["str"]);
  });
  it("converts iterables to arrays", () => {
    expect(listify([1, 2])).toEqual([1, 2]);
    expect(listify(new Set([1, 2]))).toEqual([1, 2]);
  });
  it("wraps non-iterables", () => {
    expect(listify(5)).toEqual([5]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.resize_preserving_order
// Python: resize([[1,2],[3,4]],3)==[[1,2],[1,2],[3,4]];
//         resize([],5)==[0,0,0,0,0]; resize([[1,2],[3,4]],1)==[[1,2]]
// ---------------------------------------------------------------------------
describe("resize_preserving_order", () => {
  it("extends favouring earlier elements", () => {
    expectClose(
      resize_preserving_order(array([[1, 2], [3, 4]]), 3),
      [1, 2, 1, 2, 3, 4],
    );
  });
  it("returns zeros for empty input", () => {
    expectClose(resize_preserving_order(array([]), 5), [0, 0, 0, 0, 0]);
  });
  it("truncates to one row", () => {
    expectClose(resize_preserving_order(array([[1, 2], [3, 4]]), 1), [1, 2]);
  });
  it("returns the same array when lengths match", () => {
    const a = array([[1, 2], [3, 4]]);
    expect(resize_preserving_order(a, 2)).toBe(a);
  });
  it("handles 1-D input", () => {
    // Python: resize_preserving_order(np.array([1,2]),4) == [1,1,2,2]
    expectClose(resize_preserving_order(array([1, 2]), 4), [1, 1, 2, 2]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.batch_by_property
// Python: batch_by_property([(1,2),(3,4),(5,6,7),(8,9)],len)
//   == [([(1,2),(3,4)],2), ([(5,6,7)],3), ([(8,9)],2)]
// ---------------------------------------------------------------------------
describe("batch_by_property", () => {
  it("groups consecutive items with equal property", () => {
    const out = batch_by_property(
      [[1, 2], [3, 4], [5, 6, 7], [8, 9]],
      (t: number[]) => t.length,
    );
    expect(out).toEqual([
      [[[1, 2], [3, 4]], 2],
      [[[5, 6, 7]], 3],
      [[[8, 9]], 2],
    ]);
  });
  it("returns [] for empty input", () => {
    expect(batch_by_property([], (x: number) => x)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.list_difference_update
// Python: list_difference_update([1,2,3,4],[2,4]) == [1,3]
// ---------------------------------------------------------------------------
describe("list_difference_update", () => {
  it("drops elements present in l2", () => {
    expect(list_difference_update([1, 2, 3, 4], [2, 4])).toEqual([1, 3]);
  });
  it("keeps everything when l2 is empty", () => {
    expect(list_difference_update([1, 2], [])).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.iterables.hash_obj
// NOTE: Python uses salted builtin hash(); the TS port uses a deterministic
// Java-String/FNV-style hash, so exact values intentionally differ. Test
// structural properties instead.
// ---------------------------------------------------------------------------
describe("hash_obj", () => {
  it("is deterministic for equal values", () => {
    expect(hash_obj([1, 2])).toBe(hash_obj([1, 2]));
    expect(hash_obj("abc")).toBe(hash_obj("abc"));
  });
  it("is order-independent for dicts and sets (like Python's sorted-tuple hash)", () => {
    expect(hash_obj({ a: 1, b: 2 })).toBe(hash_obj({ b: 2, a: 1 }));
    expect(hash_obj(new Set([1, 2]))).toBe(hash_obj(new Set([2, 1])));
  });
  it("distinguishes different values", () => {
    expect(hash_obj([1, 2])).not.toBe(hash_obj([1, 3]));
    expect(hash_obj("a")).not.toBe(hash_obj("b"));
  });
});

// ---------------------------------------------------------------------------
// manim.utils.deprecation._get_callable_info
// Python: foo->('function','foo'); A.meth->('method','A.meth'); A->('class','A')
// ---------------------------------------------------------------------------
describe("_get_callable_info", () => {
  it("classifies plain functions", () => {
    function foo(): void {}
    expect(_get_callable_info(foo)).toEqual(["function", "foo"]);
  });
  it("classifies dotted qualnames as methods", () => {
    function meth(): void {}
    (meth as unknown as Record<string, unknown>)["__qualname__"] = "A.meth";
    expect(_get_callable_info(meth)).toEqual(["method", "A.meth"]);
  });
  it("classifies classes", () => {
    class A {}
    expect(_get_callable_info(A as unknown as (...a: unknown[]) => unknown)).toEqual([
      "class",
      "A",
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.deprecation._deprecation_text_component
// Python: () -> 'deprecated and may be removed in a later version.'
//         ('v1','v2','msg') -> 'deprecated since v1 and is expected to be
//         removed after v2. msg'
// ---------------------------------------------------------------------------
describe("_deprecation_text_component", () => {
  it("renders the default component", () => {
    expect(_deprecation_text_component()).toBe(
      "deprecated and may be removed in a later version.",
    );
  });
  it("renders since/until/message", () => {
    expect(_deprecation_text_component("v1", "v2", "msg")).toBe(
      "deprecated since v1 and is expected to be removed after v2. msg",
    );
  });
});

// ---------------------------------------------------------------------------
// manim.utils.deprecation.deprecated_params.redirect_params (nested in Python)
// Python examples: ('old_param'->'new_param'); lambda runtime_in_ms;
//                  lambda buff -> buff_x/buff_y.
// ---------------------------------------------------------------------------
describe("redirect_params", () => {
  it("redirects tuple redirections", () => {
    const kwargs: Record<string, unknown> = { x: 1, old_param: 2 };
    redirect_params(kwargs, ["old_param"], [["old_param", "new_param"]]);
    expect(kwargs).toEqual({ x: 1, new_param: 2 });
  });
  it("redirects via a mapping function", () => {
    const fn = (runtime_in_ms: unknown): Record<string, unknown> => ({
      run_time: (runtime_in_ms as number) / 1000,
    });
    (fn as unknown as Record<string, unknown>)["paramNames"] = ["runtime_in_ms"];
    const kwargs: Record<string, unknown> = { runtime_in_ms: 2000 };
    redirect_params(kwargs, ["runtime_in_ms"], [fn]);
    expect(kwargs).toEqual({ run_time: 2.0 });
  });
  it("leaves kwargs untouched when nothing is used", () => {
    const kwargs: Record<string, unknown> = { x: 1 };
    redirect_params(kwargs, [], [["old_param", "new_param"]]);
    expect(kwargs).toEqual({ x: 1 });
  });
});

describe("deprecated_params", () => {
  it("warns and redirects deprecated kwargs", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const decorated = deprecated_params(null, null, null, "", [
      ["old_param", "new_param"],
    ])((kwargs: unknown) => kwargs);
    const out = decorated({ x: 1, old_param: 2 }) as Record<string, unknown>;
    expect(out).toEqual({ x: 1, new_param: 2 });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]![0])).toContain("old_param");
  });
  it("does not warn when no deprecated param is used", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const decorated = deprecated_params("a, b")((kwargs: unknown) => kwargs);
    decorated({ x: 1 });
    expect(warn).not.toHaveBeenCalled();
  });
  it("throws when used bare without arguments", () => {
    expect(() =>
      (deprecated_params as unknown as (f: unknown) => unknown)(() => {}),
    ).toThrow();
  });
});

// ---------------------------------------------------------------------------
// manim.utils.bezier.interpolate
// Python: interpolate(1.0,3.0,0.25)==1.5;
//         interpolate([0,0,0],[2,2,2],0.5)==[1,1,1]
// ---------------------------------------------------------------------------
describe("interpolate", () => {
  it("interpolates scalars", () => {
    // Python ground truth: 1.5
    expect(interpolate(1.0, 3.0, 0.25)).toBe(1.5);
  });
  it("interpolates point arrays with scalar alpha", () => {
    // Python ground truth: [1,1,1]
    expectClose(
      interpolate(array([0, 0, 0]), array([2, 2, 2]), 0.5),
      [1, 1, 1],
    );
  });
  it("clamps at the endpoints", () => {
    expect(interpolate(2.0, 5.0, 0)).toBe(2.0);
    expect(interpolate(2.0, 5.0, 1)).toBe(5.0);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.bezier._get_subdivision_matrix
// Python: (4,2) 8x4 matrix row0 [1,0,0,0], row1 [.5,.5,0,0]...;
//         (2,2) == [[1,0],[.5,.5],[.5,.5],[0,1]]; (5,2) raises.
// ---------------------------------------------------------------------------
describe("_get_subdivision_matrix", () => {
  it("builds the cubic subdivision matrix", () => {
    const m = _get_subdivision_matrix(4, 2);
    expect([...m.shape]).toEqual([8, 4]);
    const rows = rowsOf(m);
    expect(rows[0]).toEqual([1, 0, 0, 0]);
    expectClose(m, [1, 0, 0, 0, 0.5, 0.5, 0, 0, 0.25, 0.5, 0.25, 0, 0.125, 0.375, 0.375, 0.125, 0.125, 0.375, 0.375, 0.125, 0, 0.25, 0.5, 0.25, 0, 0, 0.5, 0.5, 0, 0, 0, 1]);
  });
  it("builds the linear subdivision matrix", () => {
    expectClose(_get_subdivision_matrix(2, 2), [1, 0, 0.5, 0.5, 0.5, 0.5, 0, 1]);
  });
  it("builds the single-point matrix of ones", () => {
    // Python: _get_subdivision_matrix(1,3) == [[1],[1],[1]]
    const m = _get_subdivision_matrix(1, 3);
    expect([...m.shape]).toEqual([3, 1]);
    expectClose(m, [1, 1, 1]);
  });
  it("rejects unsupported point counts", () => {
    expect(() => _get_subdivision_matrix(5, 2)).toThrow(
      "This function does not support subdividing Bézier curves with 0 or more than 4 control points.",
    );
  });
});

// ---------------------------------------------------------------------------
// manim.utils.bezier.get_quadratic_approximation_of_cubic
// Python: cubic (0,0,0),(1,0,0),(1,1,0),(0,1,0) ->
//  [[0,0,0],[.75,0,0],[.75,.5,0],[.75,.5,0],[.75,1,0],[0,1,0]]
// ---------------------------------------------------------------------------
describe("get_quadratic_approximation_of_cubic", () => {
  it("approximates a single cubic with two quadratics", () => {
    const out = get_quadratic_approximation_of_cubic(
      array([0, 0, 0]),
      array([1, 0, 0]),
      array([1, 1, 0]),
      array([0, 1, 0]),
    );
    expect([...out.shape]).toEqual([6, 3]);
    expectClose(out, [0, 0, 0, 0.75, 0, 0, 0.75, 0.5, 0, 0.75, 0.5, 0, 0.75, 1, 0, 0, 1, 0]);
  });
  it("handles batched inputs (2 curves -> 12 rows)", () => {
    const out = get_quadratic_approximation_of_cubic(
      array([[0, 0, 0], [1, 1, 1]]),
      array([[1, 0, 0], [2, 1, 1]]),
      array([[1, 1, 0], [2, 2, 1]]),
      array([[0, 1, 0], [1, 2, 1]]),
    );
    expect([...out.shape]).toEqual([12, 3]);
    const rows = rowsOf(out);
    expect(rows[0]).toEqual([0, 0, 0]);
    expect(rows[6]).toEqual([1, 1, 1]);
  });
  it("rejects mismatched batch sizes", () => {
    expect(() =>
      get_quadratic_approximation_of_cubic(
        array([0, 0, 0]),
        array([[1, 0, 0], [2, 0, 0]]),
        array([1, 1, 0]),
        array([0, 1, 0]),
      ),
    ).toThrow("All arguments must be Point3D or Point3D_Array.");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.space_ops.normalize_along_axis
// Python broadcasts via repeat+reshape, which misaligns norms for non-last
// axes (ground truths below are verbatim Python outputs):
//   normalize([[3,4],[0,0]], axis=0) == [[1,1.333..],[0,0]]
//   normalize([[1,2],[3,4]], axis=1) == [[.447,.894],[.6,.8]]
// ---------------------------------------------------------------------------
describe("normalize_along_axis", () => {
  it("normalizes columns (axis=0, repeat-broadcast semantics)", () => {
    const a = array([[3.0, 4.0], [0.0, 0.0]]);
    expect(normalize_along_axis(a, 0)).toBe(a); // in place, like `array /=`
    expectClose(a, [1, 1.3333333333333333, 0, 0]);
  });
  it("normalizes rows (axis=1)", () => {
    const a = array([[1.0, 2.0], [3.0, 4.0]]);
    normalize_along_axis(a, 1);
    expectClose(a, [0.4472135954999579, 0.8944271909999159, 0.6, 0.8]);
  });
  it("leaves zero slices as zero (norm 0 -> 1)", () => {
    const a = array([[0.0, 0.0], [0.0, 0.0]]);
    normalize_along_axis(a, 1);
    expectClose(a, [0, 0, 0, 0]);
  });
  it("supports negative axes", () => {
    const a = array([[1.0, 2.0], [3.0, 4.0]]);
    normalize_along_axis(a, -1);
    expectClose(a, [0.4472135954999579, 0.8944271909999159, 0.6, 0.8]);
  });
  it("matches Python on 3-D input with a non-last axis", () => {
    // Python: normalize([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]], axis=0)
    //   == [[[0.14142,0.28284],[0.36380,0.48507],[0.52705,0.63246]],
    //       [[0.64993,0.74278],[0.74485,0.82761],[0.81989,0.89443]]]
    const a = array([
      [[1, 2], [3, 4], [5, 6]],
      [[7, 8], [9, 10], [11, 12]],
    ]);
    normalize_along_axis(a, 0);
    expectClose(a, [
      0.1414213562373095, 0.282842712474619, 0.36380343755449945,
      0.48507125007266594, 0.5270462766947299, 0.6324555320336759,
      0.6499336836196815, 0.7427813527082074, 0.7448452997421311,
      0.827605888602368, 0.8198915917499229, 0.8944271909999159,
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.polylabel.Polygon (+ __init__, _ray_crosses_segment,
// _is_point_on_segment) and Cell
// Python square [[0,0],[4,0],[4,4],[0,4]]: area 16, centroid [2,2];
//   inside([2,2]) True, inside([5,5]) False; dist([2,2])==2,
//   dist([5,5])==-sqrt(2); Cell([2,2],1): d==2, p==2+sqrt(2).
// ---------------------------------------------------------------------------
describe("Polygon", () => {
  const square: number[][] = [[0, 0], [4, 0], [4, 4], [0, 4]];
  it("computes area and centroid", () => {
    const p = new Polygon([square]);
    expect(p.area).toBe(16);
    expectClose(p.centroid, [2, 2]);
  });
  it("builds start/stop/diff/norm boundary arrays", () => {
    // Python: start [[0,0],[1,0],[1,1]], stop [[1,0],[1,1],[0,1]],
    // diff [[1,0],[0,1],[-1,0]], norm [[1,0],[0,1],[-1,0]] for unit square.
    const p = new Polygon([[[0, 0], [1, 0], [1, 1], [0, 1]]]);
    expect(rowsOf(p.start)).toEqual([[0, 0], [1, 0], [1, 1]]);
    expect(rowsOf(p.stop)).toEqual([[1, 0], [1, 1], [0, 1]]);
    expect(rowsOf(p.diff)).toEqual([[1, 0], [0, 1], [-1, 0]]);
    expectClose(p.norm, [1, 0, 0, 1, -1, 0]);
  });
  it("detects inside/outside points", () => {
    const p = new Polygon([square]);
    expect(p.inside([2, 2])).toBe(true);
    expect(p.inside([5, 5])).toBe(false);
    expect(p.inside([0, 0])).toBe(true); // on boundary counts as inside
  });
  it("detects points on segments", () => {
    const p = new Polygon([square]);
    expect(p._is_point_on_segment(2, 0, 0, 0, 4, 0)).toBe(true);
    expect(p._is_point_on_segment(2, 1, 0, 0, 4, 0)).toBe(false);
  });
  it("detects ray crossings", () => {
    const p = new Polygon([square]);
    expect(p._ray_crosses_segment(2, 2, 0, 0, 0, 4)).toBe(false);
    expect(p._ray_crosses_segment(5, 2, 0, 0, 0, 4)).toBe(false);
    expect(p._ray_crosses_segment(-1, 2, 0, 0, 0, 4)).toBe(true);
  });
  it("computes signed distances", () => {
    const p = new Polygon([square]);
    expect(p.compute_distance([2, 2])).toBeCloseTo(2, 12);
    expect(p.compute_distance([5, 5])).toBeCloseTo(-Math.SQRT2, 12);
  });
});

describe("Cell", () => {
  it("stores center/half-size, signed distance and priority", () => {
    const p = new Polygon([[[0, 0], [4, 0], [4, 4], [0, 4]]]);
    const c = new Cell([2, 2], 1.0, p);
    expect(c.h).toBe(1.0);
    expect(c.d).toBeCloseTo(2, 12);
    expect(c.p).toBeCloseTo(2 + Math.SQRT2, 12);
    expectClose(c.c, [2, 2]);
  });
  it("compares by distance", () => {
    const p = new Polygon([[[0, 0], [4, 0], [4, 4], [0, 4]]]);
    const inside = new Cell([2, 2], 1, p);
    const outside = new Cell([6, 6], 1, p);
    expect(inside.greaterThan(outside)).toBe(true);
    expect(outside.lessThan(inside)).toBe(true);
    expect(inside.greaterThanOrEqual(inside)).toBe(true);
    expect(inside.lessThanOrEqual(inside)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// manim.constants.RendererType + manim.scene.section.DefaultSectionType
// Python: RendererType.CAIRO=='cairo', OPENGL=='opengl';
//         DefaultSectionType.NORMAL == 'default.normal'.
// ---------------------------------------------------------------------------
describe("RendererType", () => {
  it("has cairo and opengl values", () => {
    expect(RendererType.CAIRO).toBe("cairo");
    expect(RendererType.OPENGL).toBe("opengl");
  });
});

describe("DefaultSectionType", () => {
  it("maps NORMAL to 'default.normal'", () => {
    expect(DefaultSectionType.NORMAL).toBe("default.normal");
  });
});

// ---------------------------------------------------------------------------
// manim.mobject.svg.svg_mobject._convert_point_to_3d
// Python: _convert_point_to_3d(1,2) == [1,2,0]
// ---------------------------------------------------------------------------
describe("_convert_point_to_3d", () => {
  it("appends z=0", () => {
    expectClose(_convert_point_to_3d(1, 2), [1, 2, 0]);
  });
});

// ---------------------------------------------------------------------------
// SVGMobject.generate_config_style_dict + hash_seed
// Python defaults -> {'stroke-width': '0'}; full style maps fill/fill-opacity/
// stroke/stroke-opacity/stroke-width; hash_seed ==
// (cls, svg_default, path_string_config, file_name, config.renderer).
// ---------------------------------------------------------------------------
describe("SVGMobject.generate_config_style_dict", () => {
  it("maps the default style (only stroke-width survives)", () => {
    const s = new SVGMobject();
    expect(s.generate_config_style_dict()).toEqual({ "stroke-width": "0" });
  });
  it("maps a fully-specified style", () => {
    const s = new SVGMobject({
      svg_default: {
        color: "red",
        opacity: 0.5,
        fill_color: "blue",
        fill_opacity: 0.7,
        stroke_width: 3,
        stroke_color: "green",
        stroke_opacity: 0.9,
      },
    });
    expect(s.generate_config_style_dict()).toEqual({
      fill: "blue",
      "fill-opacity": "0.7",
      stroke: "green",
      "stroke-opacity": "0.9",
      "stroke-width": "3",
    });
  });
});

describe("SVGMobject.hash_seed", () => {
  it("returns the 5-tuple with the cairo renderer by default", () => {
    const s = new SVGMobject();
    const seed = s.hash_seed;
    expect(seed[0]).toBe("SVGMobject");
    expect(seed[3]).toBeNull();
    expect(seed[4]).toBe(RendererType.CAIRO);
  });
  it("honours the __wanim_renderer__ override", () => {
    (globalThis as Record<string, unknown>)["__wanim_renderer__"] = "opengl";
    const s = new SVGMobject();
    expect(s.hash_seed[4]).toBe(RendererType.OPENGL);
  });
});

// ---------------------------------------------------------------------------
// VMobjectFromSVGPath.handle_commands.move_pen (nested closure in Python).
// Tested through handle_commands with the TS segment representation.
// Python: Path('M 0 0 L 10 0 Z') -> 8 cubic points closing the loop;
//         Path('M 1 2') (move only) -> shape (0,3);
//         Path('M 0 0 L 3 4') -> [[0,0,0],[1,1.333],[2,2.667],[3,4,0]].
// ---------------------------------------------------------------------------
describe("VMobjectFromSVGPath.handle_commands", () => {
  it("converts a move+line+close path into closed cubic points", () => {
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [10, 0] },
      { kind: "Close" },
    ]);
    v.handle_commands();
    expect([...v.points.shape]).toEqual([8, 3]);
    expectClose(v.points, [
      0, 0, 0, 3.3333333333333335, 0, 0, 6.666666666666667, 0, 0, 10, 0, 0,
      10, 0, 0, 6.666666666666667, 0, 0, 3.3333333333333335, 0, 0, 0, 0, 0,
    ]);
  });
  it("produces an empty (0,3) array for a move-only path", () => {
    const v = new VMobjectFromSVGPath([{ kind: "Move", end: [1, 2] }]);
    v.handle_commands();
    expect([...v.points.shape]).toEqual([0, 3]);
  });
  it("converts a single line to one cubic segment", () => {
    const v = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [3, 4] },
    ]);
    v.handle_commands();
    expectClose(v.points, [0, 0, 0, 1, 1.3333333333333333, 0, 2, 2.6666666666666665, 0, 3, 4, 0]);
  });
  it("Close with coincident endpoints adds no extra segment", () => {
    // A trailing Close back at the start is equivalent to an explicit
    // return line: both yield the same 8 cubic points.
    const viaClose = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [10, 0] },
      { kind: "Line", end: [0, 0] },
      { kind: "Close" },
    ]);
    const explicit = new VMobjectFromSVGPath([
      { kind: "Move", end: [0, 0] },
      { kind: "Line", end: [10, 0] },
      { kind: "Line", end: [0, 0] },
    ]);
    expect([...viaClose.points.shape]).toEqual([8, 3]);
    expectClose(viaClose.points, flat(explicit.points));
  });
});

// ---------------------------------------------------------------------------
// VMobject.get_mobject_type_class (static)
// Python: VMobject.get_mobject_type_class() is VMobject.
// ---------------------------------------------------------------------------
describe("VMobject.get_mobject_type_class", () => {
  it("returns VMobject", () => {
    expect(VMobject.get_mobject_type_class()).toBe(VMobject);
  });
});

// ---------------------------------------------------------------------------
// Mobject.reset_points / generate_points / init_colors
// Python: fresh Mobject has points shape (0,3); init_colors/generate_points
// return self and are no-ops on the base class.
// ---------------------------------------------------------------------------
describe("Mobject base-class point/color init", () => {
  it("starts with empty (0,3) points", () => {
    const m = new Mobject();
    expect([...m.points.shape]).toEqual([0, 3]);
  });
  it("reset_points empties points and returns self", () => {
    const m = new Mobject();
    m.points = array([[1, 2, 3]]) as never;
    expect(m.reset_points()).toBe(m);
    expect([...m.points.shape]).toEqual([0, 3]);
  });
  it("generate_points and init_colors return self", () => {
    const m = new Mobject();
    expect(m.generate_points()).toBe(m);
    expect(m.init_colors()).toBe(m);
  });
});

// ---------------------------------------------------------------------------
// VMobject.consider_points_equals / set_points / get_last_point /
// gen_cubic_bezier_tuples_from_points
// Python: allclose([0,0,0],[1e-7,0,0]) True, vs [1,0,0] False;
// set_points stores; get_last_point returns points[-1];
// 8 points -> 2 tuples of 4.
// ---------------------------------------------------------------------------
describe("VMobject point helpers", () => {
  it("consider_points_equals uses allclose with the point tolerance", () => {
    const v = new VMobject();
    expect(v.consider_points_equals([0, 0, 0], [1e-7, 0, 0])).toBe(true);
    expect(v.consider_points_equals([0, 0, 0], [1, 0, 0])).toBe(false);
  });
  it("set_points stores points and returns self", () => {
    const v = new VMobject();
    expect(v.set_points([[0, 0, 0], [1, 1, 1]])).toBe(v);
    expect(rowsOf(v.points)).toEqual([[0, 0, 0], [1, 1, 1]]);
  });
  it("get_last_point returns the final point", () => {
    const v = new VMobject();
    v.set_points([[0, 0, 0], [1, 1, 1]]);
    expectClose(v.get_last_point(), [1, 1, 1]);
  });
  it("gen_cubic_bezier_tuples_from_points splits every 4 points", () => {
    const v = new VMobject();
    const tuples = v.gen_cubic_bezier_tuples_from_points(
      array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0]]),
    );
    expect(tuples).toHaveLength(2);
    expect(tuples[0]![0]).toEqual([0, 0, 0]);
    expect(tuples[1]![0]).toEqual([4, 0, 0]);
  });
  it("gen_cubic_bezier_tuples_from_points drops the remainder", () => {
    const v = new VMobject();
    const tuples = v.gen_cubic_bezier_tuples_from_points(
      array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [9, 9, 9]]),
    );
    expect(tuples).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.simple_functions.sigmoid
// Python: sigmoid(0)==0.5; sigmoid(1)==0.7310585786300049
// ---------------------------------------------------------------------------
describe("sigmoid", () => {
  it("matches the logistic function", () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(1)).toBeCloseTo(0.7310585786300049, 15);
    expect(sigmoid(-100)).toBeCloseTo(0, 12);
  });
});
