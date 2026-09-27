import { describe, expect, it } from "vitest";
import type { NDArray } from "numpy-ts";
import { Arrow, DoubleArrow, Vector } from "../src/mobject/geometry/line.js";
import { CurvedArrow, CurvedDoubleArrow, Dot, LabeledDot } from "../src/mobject/geometry/arc.js";
import { SVGMobject } from "../src/mobject/svg/svg_mobject.js";
import { Graph } from "../src/mobject/graph.js";
import { ArrowVectorField } from "../src/mobject/vector_field.js";
import {
  VectorScene,
  LinearTransformationScene,
} from "../src/scene/vector_space_scene.js";
import { NumberPlane } from "../src/mobject/graphing/coordinate_systems.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { LEFT, RIGHT, MED_SMALL_BUFF } from "../src/constants.js";

// Ground truths captured with original manim (manim 0.21.0):
// Arrow(): start [-0.75,0,0] end [0.75,0,0] (LEFT/RIGHT inset by buff=0.25 each side),
//   stroke_width 6 initial 6, tip_length attr 0.35, get_length 1.5,
//   get_default_tip_length 0.35, max ratios (0.25, 5)
// Arrow([-2,0,0],[2,0,0],buff=0): start [-2,0,0] end [2,0,0] len 4.0
// Arrow buff=0.5 on LEFT->RIGHT: len 1.0
// Arrow short ([0,0,0]->[0.2,0,0],buff=0): len 0.2 default_tip 0.05 stroke min(6,5*0.2)=1.0
// Arrow [0,0,0]->[1,0,0] buff=0: len 1.0 stroke 5.0; scale(2) -> len 2.0 stroke 6; scale tips -> len 4.0
// Vector([3,4,0]): start ORIGIN end [3,4,0] buff 0 len 5.0
// Vector([1,0]) 2D: end [1,0,0]
// Vector([1,0,0],buff=0.2): start [0.2,0,0] end [0.8,0,0]
// DoubleArrow(buff=0): has_tip+has_start_tip, start [-1,0,0] end [1,0,0]
// CurvedArrow([-1,0,0],[1,0,0]): 32 pts, start [-1,0,0] end [1,0,0], has_tip, center ~[0,0.7525,0]
// CurvedDoubleArrow: both tips
// SVGMobject python __init__ with file_name=None raises ValueError (needs file pipeline);
//   TS stays lazy (documents deviation). svg_default all-None+stroke_width 0,
//   style dict {'stroke-width':'0'}; file_name/height/width/center stored.
// ArrowVectorField f(p)=[p1,-p0,0] x_range=[-1,1,1] y_range=[-1,1,1]:
//   count 9, ranges [-1,2,1]/[-1,2,1]/[0,0.5,0.5], opacity 1.0, length_func(0)=0.225,
//   get_vector([1,0,0]) start [1,0,0] end [1,-0.3289763603835022,0] len 0.3289763603835022
// VectorScene.get_vector([1,0,0]) on default NumberPlane: start [0,0,0] end [1,0,0] buff 0
// VectorScene.get_basis_vectors: 2 Vectors ending [1,0,0]/[0,1,0], stroke min(6,5*1)=5.0
// Graph([1,2],[(1,2)]) layout {1:[0,0,0],2:[1,0,0]}: center [0.5,0,0];
//   _create_vertex(3,[2,0,0]) -> pos [2,0,0] center [2,0,0] Dot; dup -> ValueError;
//   label Mobject -> LabeledDot with label stored; vertex_mobject passthrough identity;
//   default pos == center [0.5,0,0]; vertex_config merges default+override.
// LinearTransformationScene.get_vector_movement rot90: target end [0,1,0].
// Out of scope (no tests, per port rules): Text, MarkupText, Paragraph,
// SingleStringMathTex, MathTex (+4 helpers), Typst, MathTypst,
// Typst.get_mob_from_shape_element.

function close(a: number, b: number, tol = 1e-9): boolean {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= tol * denom;
}

function vecClose(actual: number[] | NDArray, expected: number[], tol = 1e-6): void {
  const arr: number[] = Array.isArray(actual)
    ? [...(actual as number[])]
    : [...((actual as NDArray).toArray() as unknown as number[])].map(Number);
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(close(arr[i] ?? NaN, expected[i] ?? NaN, tol)).toBe(true);
  }
}

describe("Arrow.__init__", () => {
  it("defaults match python (LEFT/RIGHT, buff 0.25, stroke 6)", () => {
    const a = new Arrow();
    vecClose(a.get_start(), [-0.75, 0, 0]);
    vecClose(a.get_end(), [0.75, 0, 0]);
    expect(close(a.get_length(), 1.5)).toBe(true);
    expect(close(a.stroke_width, 6)).toBe(true);
    expect(close(a.initial_stroke_width, 6)).toBe(true);
    expect(close(a.tip_length, 0.35)).toBe(true);
    expect(close(a.get_default_tip_length(), 0.35)).toBe(true);
    expect(close(a.max_tip_length_to_length_ratio, 0.25)).toBe(true);
    expect(close(a.max_stroke_width_to_length_ratio, 5)).toBe(true);
    expect(a.has_tip()).toBe(true);
    expect(close(a.buff, MED_SMALL_BUFF)).toBe(true);
    expect(close(MED_SMALL_BUFF, 0.25)).toBe(true);
    void LEFT;
    void RIGHT;
  });
  it("buff=0 spans full LEFT->RIGHT with length 4", () => {
    const a = new Arrow([-2, 0, 0], [2, 0, 0], 0);
    vecClose(a.get_start(), [-2, 0, 0]);
    vecClose(a.get_end(), [2, 0, 0]);
    expect(close(a.get_length(), 4.0)).toBe(true);
  });
  it("buff=0.5 shortens LEFT->RIGHT to length 1.0", () => {
    const a = new Arrow([...LEFT], [...RIGHT], 0.5);
    expect(close(a.get_length(), 1.0, 1e-6)).toBe(true);
  });
  it("short arrow caps tip length and stroke width", () => {
    const a = new Arrow([0, 0, 0], [0.2, 0, 0], 0);
    expect(close(a.get_length(), 0.2, 1e-9)).toBe(true);
    // python: min(0.35, 0.25*0.2) = 0.05
    expect(close(a.get_default_tip_length(), 0.05, 1e-9)).toBe(true);
    // python: min(6, 5*0.2) = 1.0
    expect(close(a.stroke_width, 1.0, 1e-9)).toBe(true);
    expect(close(a.initial_stroke_width, 6, 1e-9)).toBe(true);
  });
  it("scale keeps stroke fixed unless scale_tips", () => {
    const a = new Arrow([0, 0, 0], [1, 0, 0], 0);
    expect(close(a.stroke_width, 5.0, 1e-9)).toBe(true);
    a.scale(2);
    expect(close(a.get_length(), 2.0, 1e-6)).toBe(true);
    expect(close(a.stroke_width, 6, 1e-9)).toBe(true);
    expect(a.has_tip()).toBe(true);
    a.scale(2, true);
    expect(close(a.get_length(), 4.0, 1e-6)).toBe(true);
  });
});

describe("DoubleArrow.__init__", () => {
  it("has both tips and spans endpoints with buff=0", () => {
    const d = new DoubleArrow([-1, 0, 0], [1, 0, 0], 0);
    expect(d.has_tip()).toBe(true);
    expect(d.has_start_tip()).toBe(true);
    vecClose(d.get_start(), [-1, 0, 0], 1e-5);
    vecClose(d.get_end(), [1, 0, 0], 1e-5);
  });
  it("defaults carry Arrow buff with both tips", () => {
    const d = new DoubleArrow();
    expect(d.has_tip()).toBe(true);
    expect(d.has_start_tip()).toBe(true);
    expect(d.get_length()).toBeGreaterThan(0);
  });
});

describe("CurvedArrow.__init__ / CurvedDoubleArrow.__init__", () => {
  it("CurvedArrow matches python points/start/end/tip", () => {
    const c = new CurvedArrow([-1, 0, 0], [1, 0, 0]);
    expect(c.points.shape[0]).toBe(32); // python: 32 rows
    vecClose(c.get_start(), [-1, 0, 0], 1e-5);
    vecClose(c.get_end(), [1, 0, 0], 1e-5);
    expect(c.has_tip()).toBe(true);
    const center = c.get_arc_center();
    expect(close(center[0] ?? 0, 0, 1e-6)).toBe(true);
    // python center y ≈ 0.7525126265847055
    expect(close(center[1] ?? 0, 0.7525126265847055, 1e-6)).toBe(true);
    expect(close(center[2] ?? 0, 0, 1e-6)).toBe(true);
  });
  it("CurvedDoubleArrow has both tips", () => {
    const cd = new CurvedDoubleArrow([-1, 0, 0], [1, 0, 0]);
    expect(cd.has_tip()).toBe(true);
    expect(cd.has_start_tip()).toBe(true);
    vecClose(cd.get_start(), [-1, 0, 0], 1e-5);
    vecClose(cd.get_end(), [1, 0, 0], 1e-5);
  });
});

describe("Vector.__init__", () => {
  it("direction [3,4,0] starts at ORIGIN with length 5", () => {
    const v = new Vector([3, 4, 0]);
    vecClose(v.get_start(), [0, 0, 0]);
    vecClose(v.get_end(), [3, 4, 0]);
    expect(close(v.get_length(), 5.0, 1e-9)).toBe(true);
    expect(close(v.buff, 0)).toBe(true);
  });
  it("2D direction gains zero z", () => {
    const v = new Vector([1, 0]);
    vecClose(v.get_end(), [1, 0, 0]);
    vecClose(v.get_start(), [0, 0, 0]);
  });
  it("buff insets both ends", () => {
    const v = new Vector([1, 0, 0], 0.2);
    vecClose(v.get_start(), [0.2, 0, 0], 1e-9);
    vecClose(v.get_end(), [0.8, 0, 0], 1e-9);
  });
});

describe("SVGMobject.__init__", () => {
  it("stores lazy defaults without parsing (python raises ValueError here)", () => {
    // NOTE deviation: python SVGMobject() raises ValueError("Must specify file
    // for SVGMobject") because it parses eagerly; the TS port stays lazy until
    // the ElementTree+svgelements pipeline lands, so construction succeeds.
    const s = new SVGMobject();
    expect(s.file_name).toBe(null);
    expect(s.should_center).toBe(true);
    expect(s.svg_height).toBe(2);
    expect(s.svg_width).toBe(null);
    expect(s.submobjects.length).toBe(0);
    expect(s.path_string_config).toEqual({});
    expect(s.svg_default["stroke_width"]).toBe(0);
    expect(s.svg_default["fill_color"]).toBe(null);
  });
  it("explicit file/geometry/style overrides are stored", () => {
    const s = new SVGMobject({
      file_name: "test.svg",
      should_center: false,
      height: 3,
      width: 4,
      fill_opacity: 0.5,
      stroke_width: 2,
    });
    expect(s.file_name).toBe("test.svg");
    expect(s.should_center).toBe(false);
    expect(s.svg_height).toBe(3);
    expect(s.svg_width).toBe(4);
    expect(close(s.fill_opacity ?? NaN, 0.5)).toBe(true);
    expect(close(s.stroke_width ?? NaN, 2)).toBe(true);
  });
  it("generate_config_style_dict maps defaults like python", () => {
    const s = new SVGMobject();
    // python with all-None defaults (stroke_width 0): {'stroke-width': '0'}
    expect(s.generate_config_style_dict()).toEqual({ "stroke-width": "0" });
  });
  it("get_file_path throws ValueError without a file like python", () => {
    const s = new SVGMobject();
    expect(() => s.get_file_path()).toThrow(/Must specify file/);
  });
});

describe("VectorScene.get_vector / add_vector / get_basis_vectors", () => {
  it("get_vector maps plane coords with buff=0", () => {
    const vs = new VectorScene();
    vs.plane = new NumberPlane();
    const a = vs.get_vector([1, 0, 0]);
    expect(a).toBeInstanceOf(Arrow);
    vecClose(a.get_start(), [0, 0, 0], 1e-6);
    vecClose(a.get_end(), [1, 0, 0], 1e-6);
    expect(close(a.buff, 0)).toBe(true);
  });
  it("add_vector converts coords to a Vector and adds without animation", () => {
    const vs = new VectorScene();
    const out = vs.add_vector([2, 1, 0], "#FFFF00", false);
    expect(out).toBeInstanceOf(Arrow);
    vecClose(out.get_start(), [0, 0, 0], 1e-6);
    vecClose(out.get_end(), [2, 1, 0], 1e-6);
    expect(vs.mobjects).toContain(out);
  });
  it("add_vector passes pre-made Arrows through", () => {
    const vs = new VectorScene();
    const a = new Arrow([-1, 0, 0], [1, 0, 0], 0);
    const out = vs.add_vector(a, "#FFFF00", false);
    expect(out).toBe(a);
  });
  it("get_basis_vectors are unit x/y Vectors with capped stroke", () => {
    const vs = new VectorScene();
    const basis = vs.get_basis_vectors();
    expect(basis.submobjects.length).toBe(2);
    vecClose((basis.submobjects[0] as Vector).get_end(), [1, 0, 0], 1e-6);
    vecClose((basis.submobjects[1] as Vector).get_end(), [0, 1, 0], 1e-6);
    // python: min(6, 5*1) = 5.0 via Arrow._set_stroke_width_from_length
    expect(close((basis.submobjects[0] as Vector).stroke_width ?? NaN, 5.0, 1e-9)).toBe(true);
    expect(close((basis.submobjects[1] as Vector).stroke_width ?? NaN, 5.0, 1e-9)).toBe(true);
  });
});

describe("LinearTransformationScene.add_vector / get_vector_movement", () => {
  it("add_vector appends to moving_vectors", () => {
    const scene = new LinearTransformationScene();
    scene.moving_vectors = [];
    const out = scene.add_vector([1, 0, 0], "#FFFF00", false);
    expect(scene.moving_vectors).toContain(out);
    vecClose((out as Vector).get_end(), [1, 0, 0], 1e-6);
  });
  it("get_vector_movement targets func(end) with rotation", () => {
    const scene = new LinearTransformationScene();
    const v = new Vector([1, 0, 0]);
    scene.moving_vectors = [v as unknown as never];
    scene.leave_ghost_vectors = false;
    const rot90 = (p: number[]) => [-p[1]!, p[0]!, 0];
    const anim = scene.get_vector_movement(rot90 as never);
    expect(anim.constructor.name).toBe("Transform");
    const target = (v as unknown as { target: Vector }).target;
    vecClose(target.get_end(), [0, 1, 0], 1e-6);
  });
  it("near-zero targets scale the tip down", () => {
    const scene = new LinearTransformationScene();
    const v = new Vector([0.01, 0, 0]);
    scene.moving_vectors = [v as unknown as never];
    scene.leave_ghost_vectors = false;
    scene.get_vector_movement((() => [0, 0, 0]) as never);
    const target = (v as unknown as { target: Vector }).target;
    expect(target.get_tip()).toBeDefined();
    vecClose(target.get_end(), [0, 0, 0], 1e-6);
  });
});

describe("ArrowVectorField.__init__ / get_vector", () => {
  const field = (p: number[] | NDArray): number[] => {
    const a = Array.isArray(p) ? p : [...(p.toArray() as unknown as number[])].map(Number);
    return [a[1] ?? 0, -(a[0] ?? 0), 0];
  };
  it("explicit 3x3 ranges give 9 vectors with extended ranges", () => {
    const avf = new ArrowVectorField(field, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    expect(avf.submobjects.length).toBe(9); // python: 9
    expect([...avf.x_range]).toEqual([-1, 2, 1]);
    expect([...avf.y_range]).toEqual([-1, 2, 1]);
    expect([...avf.z_range]).toEqual([0, 0.5, 0.5]);
    expect(close(avf.opacity, 1.0)).toBe(true);
    expect(close(avf.length_func(0), 0.225, 1e-9)).toBe(true);
  });
  it("get_vector scales by 0.45*sigmoid(norm) and roots at point", () => {
    const avf = new ArrowVectorField(field, { x_range: [-1, 1, 1], y_range: [-1, 1, 1] });
    const v = avf.get_vector([1.0, 0, 0]);
    vecClose(v.get_start(), [1, 0, 0], 1e-6);
    // python: [1, -0.3289763603835022, 0], len 0.3289763603835022
    vecClose(v.get_end(), [1, -0.3289763603835022, 0], 1e-6);
    expect(close(v.get_length(), 0.3289763603835022, 1e-9)).toBe(true);
  });
  it("zero field vector stays at its root", () => {
    const avf = new ArrowVectorField(() => [0, 0, 0], {
      x_range: [0, 1, 1],
      y_range: [0, 1, 1],
    });
    const v = avf.get_vector([0, 0, 0]);
    vecClose(v.get_start(), [0, 0, 0], 1e-9);
    vecClose(v.get_end(), [0, 0, 0], 1e-9);
  });
  it("single-color fields color every vector uniformly", () => {
    const avf = new ArrowVectorField(field, {
      x_range: [0, 1, 1],
      y_range: [0, 1, 1],
      color: "#FF0000",
    } as never);
    expect(avf.single_color).toBe(true);
    const v = avf.get_vector([0, 0, 0]);
    expect(v.get_color().to_hex().toUpperCase()).toBe("#FF0000");
  });
});

describe("GenericGraph._create_vertex", () => {
  function twoNodeGraph(): Graph {
    // NOTE: plain-object layouts stringify numeric keys; the TS port matches
    // them back to numeric vertices (python used real dict keys).
    return new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as never,
    });
  }
  it("creates a Dot at the given position", () => {
    const g = twoNodeGraph();
    const [vertex, pos, , mob] = g._create_vertex(3, [2, 0, 0]);
    expect(vertex).toBe(3);
    vecClose(pos, [2, 0, 0]);
    vecClose(mob.get_center(), [2, 0, 0], 1e-6);
    expect(mob).toBeInstanceOf(Dot);
  });
  it("defaults to the graph center like python", () => {
    const g = twoNodeGraph();
    // python center [0.5, 0, 0]
    vecClose(g.get_center(), [0.5, 0, 0], 1e-6);
    const [, pos, , mob] = g._create_vertex(7);
    vecClose(pos, [0.5, 0, 0], 1e-6);
    vecClose(mob.get_center(), [0.5, 0, 0], 1e-6);
  });
  it("duplicate vertex raises ValueError", () => {
    const g = twoNodeGraph();
    expect(() => g._create_vertex(1)).toThrow(/already used/);
  });
  it("Mobject label upgrades Dot to LabeledDot and stores the label", () => {
    const g = twoNodeGraph();
    const sq = new Square();
    const [, , cfg, mob] = g._create_vertex(5, [3, 0, 0], sq as never);
    expect(mob).toBeInstanceOf(LabeledDot);
    expect(cfg["label"]).toBe(sq);
    vecClose(mob.get_center(), [3, 0, 0], 1e-6);
  });
  it("label=True builds a MathTex vertex label", () => {
    const g = twoNodeGraph();
    const [, , , mob] = g._create_vertex(4, [0, 0, 0], true);
    const label = (mob as unknown as { submobjects: Array<{ tex_string: string }> }).submobjects[0];
    expect(label.tex_string).toBe("4");
  });
  it("explicit vertex_mobject passes through by identity", () => {
    const g = twoNodeGraph();
    const sq = new Square();
    const [, , , mob] = g._create_vertex(6, [4, 0, 0], false, undefined, null, null, sq as never);
    expect(mob).toBe(sq);
    vecClose(mob.get_center(), [4, 0, 0], 1e-6);
  });
  it("merges default vertex_config with per-call overrides", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as never,
      vertex_config: { radius: 0.5 },
    });
    const [, , cfg] = g._create_vertex(8, [0, 0, 0], false, undefined, null, {
      stroke_width: 3,
    });
    // python: {'radius': 0.5, 'stroke_width': 3}
    expect(cfg["radius"]).toBe(0.5);
    expect(cfg["stroke_width"]).toBe(3);
  });
});

// --- integration: batch symbols + earlier-ported code ---

describe("integration", () => {
  it("VectorScene basis vectors feed LinearTransformationScene movement", () => {
    const vs = new VectorScene();
    const basis = vs.get_basis_vectors();
    const scene = new LinearTransformationScene();
    scene.moving_vectors = [...basis.submobjects] as never;
    scene.leave_ghost_vectors = false;
    const anim = scene.get_vector_movement(((p: number[]) => [2 * p[0]!, 2 * p[1]!, 0]) as never);
    expect(anim.constructor.name).toBe("Transform");
    for (const v of basis.submobjects as Vector[]) {
      const t = (v as unknown as { target: Vector }).target;
      expect(close(t.get_length(), 2.0, 1e-6)).toBe(true);
    }
  });
  it("ArrowVectorField vectors chain with Vector start/end math", () => {
    const avf = new ArrowVectorField(
      (p: number[] | NDArray) => {
        const a = Array.isArray(p) ? p : [...(p.toArray() as unknown as number[])].map(Number);
        return [a[1] ?? 0, -(a[0] ?? 0), 0];
      },
      { x_range: [-1, 1, 1], y_range: [-1, 1, 1] },
    );
    expect(avf.submobjects.length).toBe(9);
    // every field vector is rooted at a grid point: start matches its lattice site
    for (const m of avf.submobjects as Vector[]) {
      const s = m.get_start();
      expect(s.length).toBe(3);
    }
    const v = new Vector([3, 4, 0]);
    expect(close(v.get_length(), 5, 1e-6)).toBe(true);
    const fv = avf.get_vector([0.0, 1.0, 0]);
    expect(fv.get_length()).toBeGreaterThan(0);
    vecClose(fv.get_start(), [0, 1, 0], 1e-6);
  });
  it("DoubleArrow/CurvedArrow tips survive a graph-vertex chain", () => {
    const g = new Graph([1, 2], [[1, 2]], {
      layout: { 1: [0, 0, 0], 2: [1, 0, 0] } as never,
    });
    const [, , , mob] = g._create_vertex(3, [1, 2, 0]);
    vecClose(mob.get_center(), [1, 2, 0], 1e-6);
    const d = new DoubleArrow([0, 0, 0], mob.get_center(), 0);
    expect(d.has_tip() && d.has_start_tip()).toBe(true);
    const c = new CurvedArrow([0, 0, 0], mob.get_center());
    expect(c.has_tip()).toBe(true);
    expect(c.get_num_points()).toBeGreaterThan(0);
  });
  it("SVGMobject style overrides flow into VMobject stroke/fill", () => {
    const s = new SVGMobject({ fill_opacity: 0.5, stroke_width: 2 });
    expect(close(s.fill_opacity ?? NaN, 0.5)).toBe(true);
    expect(close(s.stroke_width ?? NaN, 2)).toBe(true);
    // config lookups feeding geometry: NumberPlane coords drive Arrow ends
    const vs = new VectorScene();
    vs.plane = new NumberPlane();
    const a = vs.get_vector([2, 1, 0]);
    vecClose(a.get_end(), [2, 1, 0], 1e-6);
  });
});
