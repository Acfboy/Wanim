import { describe, expect, it } from "vitest";
import { array, eye, type NDArray } from "numpy-ts";
import { RandomColorGenerator, ManimColor } from "../src/utils/color/core.js";
import {
  VMobject,
  VGroup,
  VectorizedPoint,
} from "../src/mobject/types/vectorized_mobject.js";
import { ThreeDVMobject } from "../src/mobject/three_d/three_dimensions.js";
import { Mobject } from "../src/mobject/mobject.js";
import { ValueTracker } from "../src/mobject/value_tracker.js";
import { Scene } from "../src/scene/scene.js";
import { ThreeDScene } from "../src/scene/three_d_scene.js";
import {
  VectorScene,
  X_COLOR,
  Y_COLOR,
} from "../src/scene/vector_space_scene.js";
import { RED, BLUE, GREEN, WHITE } from "../src/utils/color/core.js";
import {
  DEGREES,
  OUT,
  RIGHT,
  SHIFT_VALUE,
  UP,
} from "../src/constants.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= Math.max(1e-12, Math.abs(b) * tol);
}

function expectRowsClose(actual: NDArray, expected: number[][], tol = 1e-9): void {
  const expectedRows = (Array.isArray(expected)
    ? (expected as unknown[])
    : ((expected as { toArray(): unknown }).toArray() as unknown[])
  ).map((row) =>
    Array.isArray(row) ? row : ((row as { toArray(): unknown }).toArray() as number[]),
  ) as number[][];
  const rows = (actual.toArray() as unknown[][]).map((r) =>
    (r as unknown[]).map(Number),
  );
  expect(rows.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(rows[i]?.length).toBe(expectedRows[i]?.length);
    for (let j = 0; j < (expectedRows[i]?.length ?? 0); j += 1) {
      expect(close(rows[i]?.[j] ?? NaN, expectedRows[i]?.[j] ?? NaN, tol)).toBe(true);
    }
  }
}

function toList(value: NDArray): number[] {
  const raw = value.toArray() as unknown;
  const out: number[] = [];
  const walk = (item: unknown): void => {
    if (Array.isArray(item)) {
      for (const sub of item) walk(sub);
      return;
    }
    out.push(Number(item));
  };
  if (Array.isArray(raw)) for (const item of raw) walk(item);
  else walk(raw);
  return out;
}

// ---------------------------------------------------------------------------
// RandomColorGenerator.next
// Ground truth (original manim): seed=42 over [RED,BLUE,GREEN] ->
//   ['#83C167','#FC6255','#FC6255','#83C167','#58C4DD']
// seed=0 default palette first three -> ['#F7D96F','#F7C797','#888888'], 89 colors.
// ---------------------------------------------------------------------------
describe("RandomColorGenerator.next", () => {
  it("reproduces the seeded choice sequence (seed=42, [RED,BLUE,GREEN])", () => {
    const rnd = new RandomColorGenerator(42, [RED, BLUE, GREEN]);
    const hexes = [rnd.next(), rnd.next(), rnd.next(), rnd.next(), rnd.next()].map(
      (c) => c.to_hex(),
    );
    expect(hexes).toEqual(["#83C167", "#FC6255", "#FC6255", "#83C167", "#58C4DD"]);
  });

  it("reproduces the default-palette seeded sequence (seed=0)", () => {
    const rnd = new RandomColorGenerator(0);
    expect(rnd.colors.length).toBe(89);
    expect([rnd.next(), rnd.next(), rnd.next()].map((c) => c.to_hex())).toEqual([
      "#F7D96F",
      "#F7C797",
      "#888888",
    ]);
  });

  it("returns ManimColor instances from the configured list", () => {
    const rnd = new RandomColorGenerator(7, [RED]);
    for (let i = 0; i < 3; i += 1) {
      const c = rnd.next();
      expect(c).toBeInstanceOf(ManimColor);
      expect(c.to_hex()).toBe("#FC6255");
    }
  });
});

// ---------------------------------------------------------------------------
// VMobject.get_color
// Ground truth (original manim): default VMobject -> stroke (#FFFFFF); color=RED ->
// #FC6255; fill .5 -> fill color; fill 0 + stroke BLUE -> stroke color.
// ---------------------------------------------------------------------------
describe("VMobject.get_color", () => {
  it("returns the stroke color when every fill opacity is zero (default)", () => {
    const v = new VMobject();
    expect(v.get_color().to_hex()).toBe("#FFFFFF");
    expect(v.get_color().to_hex()).toBe(v.get_stroke_color()?.to_hex());
  });

  it("returns the fill color once fill opacity is nonzero", () => {
    const v = new VMobject({ color: RED });
    expect(v.get_color().to_hex()).toBe("#FC6255");
    const v2 = new VMobject({ color: BLUE });
    v2.set_fill(BLUE, 0.5);
    expect(v2.get_color().to_hex()).toBe(v2.get_fill_color().to_hex());
  });

  it("falls back to stroke color when fill was made fully transparent", () => {
    const v = new VMobject({ color: RED });
    v.set_fill(undefined, 0);
    v.set_stroke(BLUE, undefined, 1);
    expect(v.get_color().to_hex()).toBe("#58C4DD");
  });
});

// ---------------------------------------------------------------------------
// ValueTracker.increment_value
// Ground truth (original manim): 1.5 +2.25 -> 3.75; -5 -> -1.25; returns self.
// ---------------------------------------------------------------------------
describe("ValueTracker.increment_value", () => {
  it("adds the delta to the stored value", () => {
    const vt = new ValueTracker(1.5);
    vt.increment_value(2.25);
    expect(close(vt.get_value(), 3.75)).toBe(true);
    vt.increment_value(-5);
    expect(close(vt.get_value(), -1.25)).toBe(true);
  });

  it("returns self and handles a zero delta", () => {
    const vt = new ValueTracker(2);
    expect(vt.increment_value(0)).toBe(vt);
    expect(close(vt.get_value(), 2)).toBe(true);
  });

  it("accumulates fractional deltas", () => {
    const vt = new ValueTracker(0);
    vt.increment_value(0.1);
    vt.increment_value(0.2);
    expect(close(vt.get_value(), 0.3)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// VMobject.rotate_sheen_direction
// Ground truth (original manim): UP rotated by PI about OUT -> [~0,-1,0];
// family=True rotates every family member; family=False only self.
// ---------------------------------------------------------------------------
describe("VMobject.rotate_sheen_direction", () => {
  it("rotates UP by PI about OUT to -UP (family=false)", () => {
    const v = new VMobject();
    v.set_sheen_direction(UP);
    const ret = v.rotate_sheen_direction(Math.PI, [...OUT], false);
    expect(ret).toBe(v);
    const dir = v.get_sheen_direction();
    expect(close(dir[0] ?? NaN, 0)).toBe(true);
    expect(close(dir[1] ?? NaN, -1)).toBe(true);
    expect(close(dir[2] ?? NaN, 0)).toBe(true);
  });

  it("rotates the whole family by default (RIGHT by PI/2 -> UP)", () => {
    const parent = new VMobject();
    const child = new VMobject();
    child.set_sheen_direction(RIGHT, false);
    parent.add(child);
    parent.rotate_sheen_direction(Math.PI / 2);
    const cdir = (child as VMobject).sheen_direction;
    expect(close(cdir[0] ?? NaN, 0, 1e-9)).toBe(true);
    expect(close(cdir[1] ?? NaN, 1, 1e-9)).toBe(true);
    // Parent default UL=[-1,1,0] rotated +90 deg about z -> [-1,-1,0].
    const pdir = (parent as VMobject).sheen_direction;
    expect(close(pdir[0] ?? NaN, -1)).toBe(true);
    expect(close(pdir[1] ?? NaN, -1)).toBe(true);
  });

  it("leaves submobjects untouched with family=false", () => {
    const parent = new VMobject();
    const child = new VMobject();
    child.set_sheen_direction(RIGHT, false);
    parent.add(child);
    parent.set_sheen_direction(UP, false);
    parent.rotate_sheen_direction(Math.PI / 2, [...OUT], false);
    const cdir = (child as VMobject).sheen_direction;
    expect(close(cdir[0] ?? NaN, 1)).toBe(true);
    expect(close(cdir[1] ?? NaN, 0)).toBe(true);
    const pdir = (parent as VMobject).sheen_direction;
    expect(close(pdir[0] ?? NaN, -1)).toBe(true);
    expect(close(pdir[1] ?? NaN, 0, 1e-9)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// VMobject.get_nth_curve_length_pieces
// Ground truth (original manim): unit-line curve, default sampling -> nine ~1/9
// pieces; sample_points=5 -> four 0.25 pieces; out-of-range -> AssertionError.
// ---------------------------------------------------------------------------
describe("VMobject.get_nth_curve_length_pieces", () => {
  it("splits a unit line into nine ~1/9 pieces by default", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const pieces = v.get_nth_curve_length_pieces(0);
    expect(pieces.shape[0]).toBe(9);
    const values = toList(pieces);
    for (const value of values) expect(close(value, 1 / 9)).toBe(true);
    expect(close(values.reduce((a, b) => a + b, 0), 1)).toBe(true);
  });

  it("honors an explicit sample_points count", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const pieces = v.get_nth_curve_length_pieces(0, 5);
    expect(pieces.shape[0]).toBe(4);
    for (const value of toList(pieces)) expect(close(value, 0.25)).toBe(true);
  });

  it("measures each curve of a two-curve path independently", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
    ]);
    expect(v.get_num_curves()).toBe(2);
    for (const n of [0, 1]) {
      const values = toList(v.get_nth_curve_length_pieces(n));
      expect(values.length).toBe(9);
      expect(close(values.reduce((a, b) => a + b, 0), 1)).toBe(true);
    }
  });

  it("throws AssertionError for an out-of-range curve index", () => {
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    let thrown: unknown;
    try {
      v.get_nth_curve_length_pieces(5);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe("AssertionError");
  });
});

// ---------------------------------------------------------------------------
// VMobject.set_shade_in_3d
// Ground truth (original manim): propagates to family; z_index_as_group links child.
// ---------------------------------------------------------------------------
describe("VMobject.set_shade_in_3d", () => {
  it("propagates the flag through the family and returns self", () => {
    const parent = new VMobject();
    const child = new VMobject();
    parent.add(child);
    expect(parent.set_shade_in_3d(true)).toBe(parent);
    expect(parent.shade_in_3d).toBe(true);
    expect((child as VMobject).shade_in_3d).toBe(true);
    parent.set_shade_in_3d(false);
    expect(parent.shade_in_3d).toBe(false);
    expect((child as VMobject).shade_in_3d).toBe(false);
  });

  it("optionally groups z_index under the caller", () => {
    const parent = new VMobject();
    const child = new VMobject();
    parent.add(child);
    parent.set_shade_in_3d(false, true);
    expect(child.z_index_group).toBe(parent);
    parent.set_shade_in_3d(true, false);
    // Without the flag the previous group link is left untouched.
    expect(child.z_index_group).toBe(parent);
  });
});

// ---------------------------------------------------------------------------
// VectorizedPoint.__init__
// Ground truth (original manim): defaults location=ORIGIN, width/height=0.01,
// 1 point, stroke_width=0, fill_opacity=0; custom kwargs honored.
// ---------------------------------------------------------------------------
describe("VectorizedPoint.__init__", () => {
  it("builds the default point at the origin", () => {
    const vp = new VectorizedPoint();
    expectRowsClose(vp.get_location().reshape(1, 3) as NDArray, [[0, 0, 0]]);
    expect(vp.artificial_width).toBe(0.01);
    expect(vp.artificial_height).toBe(0.01);
    expect(vp.width).toBe(0.01);
    expect(vp.height).toBe(0.01);
    expect(vp.get_num_points()).toBe(1);
    expect(vp.stroke_width).toBe(0);
    expect(close(vp.get_fill_opacity(), 0)).toBe(true);
  });

  it("honors a custom location, color, opacities, and artificial size", () => {
    const vp = new VectorizedPoint([1, 2, 3], RED, 0.5, 2, 0.03, 0.04);
    expectRowsClose(vp.get_location().reshape(1, 3) as NDArray, [[1, 2, 3]]);
    expect(vp.artificial_width).toBe(0.03);
    expect(vp.artificial_height).toBe(0.04);
    expect(close(vp.get_fill_opacity(), 0.5)).toBe(true);
    expect(vp.stroke_width).toBe(2);
    expect(vp.get_color().to_hex()).toBe("#FC6255");
  });
});

// ---------------------------------------------------------------------------
// ThreeDVMobject.__init__
// Ground truth (original manim): shade_in_3d=True by default; False when passed.
// ---------------------------------------------------------------------------
describe("ThreeDVMobject.__init__", () => {
  it("defaults shade_in_3d to true", () => {
    expect(new ThreeDVMobject().shade_in_3d).toBe(true);
  });

  it("accepts an explicit shade_in_3d=False", () => {
    expect(new ThreeDVMobject({ shade_in_3d: false }).shade_in_3d).toBe(false);
  });

  it("is a VMobject with working colors", () => {
    const t = new ThreeDVMobject();
    expect(t).toBeInstanceOf(VMobject);
    expect(t.get_color().to_hex()).toBe("#FFFFFF");
  });
});

// ---------------------------------------------------------------------------
// Mobject.get_all_points
// Ground truth (original manim): empty -> shape (0,3); merged parent+child in
// depth-first pre-order.
// ---------------------------------------------------------------------------
describe("Mobject.get_all_points", () => {
  it("returns an empty (0,3) array for a fresh mobject", () => {
    const m = new Mobject();
    const all = m.get_all_points();
    expect(all.shape[0]).toBe(0);
    expect(toList(all).length).toBe(0);
  });

  it("merges parent and child points in pre-order", () => {
    const parent = new Mobject();
    parent.set_points([
      [0, 0, 0],
      [1, 1, 1],
    ]);
    const child = new Mobject();
    child.set_points([[2, 2, 2]]);
    parent.add(child);
    expectRowsClose(parent.get_all_points(), [
      [0, 0, 0],
      [1, 1, 1],
      [2, 2, 2],
    ]);
  });

  it("visits nested submobjects depth-first", () => {
    const root = new Mobject();
    root.set_points([[0, 0, 0]]);
    const a = new Mobject();
    a.set_points([[1, 0, 0]]);
    const b = new Mobject();
    b.set_points([[2, 0, 0]]);
    const bChild = new Mobject();
    bChild.set_points([[3, 0, 0]]);
    b.add(bChild);
    root.add(a, b);
    expectRowsClose(root.get_all_points(), [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// Scene mouse/keyboard handlers.
// Python originals assert OpenGLCamera/OpenGLRenderer; the TS port is
// renderer-agnostic, so tests use a fake camera+renderer pair implementing
// the same surface (width/height, model_matrix, get_position, shift,
// to_default_state, pressed_keys) and compare against numpy ground truth
// computed in the original manim (see comments for the derivations).
// ---------------------------------------------------------------------------
class FakeCamera {
  model_matrix: NDArray = array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 11],
    [0, 0, 0, 1],
  ]) as NDArray;
  inverse_rotation_matrix: NDArray = eye(3) as NDArray;
  maximum_polar_angle = Math.PI / 2;
  minimum_polar_angle = -Math.PI / 2;
  width = 10;
  height = 6;
  shifted: number[][] = [];
  defaulted = 0;

  get_width(): number {
    return this.width;
  }

  get_height(): number {
    return this.height;
  }

  get_position(): number[] {
    const m = this.model_matrix.toArray() as unknown[][];
    return [Number(m[0]?.[3]), Number(m[1]?.[3]), Number(m[2]?.[3])];
  }

  shift(vector: number[] | NDArray): this {
    const v = Array.isArray(vector)
      ? [...vector]
      : toList(vector as NDArray);
    this.shifted.push(v);
    return this;
  }

  to_default_state(): void {
    this.defaulted += 1;
  }
}

function makeScene(opts: { shift?: boolean } = {}): { scene: Scene; camera: FakeCamera } {
  const camera = new FakeCamera();
  const scene = new Scene(null, FakeCamera as unknown as new (...args: never[]) => FakeCamera);
  // Scene constructor builds its own camera; swap in the instrumented one.
  scene.camera = camera as unknown as FakeCamera;
  scene.renderer = { pressed_keys: opts.shift ? new Set([SHIFT_VALUE]) : new Set<number>() };
  scene.mouse_point = new Mobject();
  scene.mouse_point.set_points([[9, 9, 9]]);
  return { scene, camera };
}

describe("Scene.on_mouse_motion", () => {
  it("moves mouse_point to the cursor", () => {
    const { scene } = makeScene();
    scene.on_mouse_motion([1, 2, 3], [0.1, 0.2, 0]);
    expectRowsClose(scene.mouse_point.points, [[1, 2, 3]]);
  });

  it("pans the camera while SHIFT is held (ground truth: [-0.5,-0.6,0])", () => {
    // Python: shift=-d_point; shift[0]*=width/2=5; shift[1]*=height/2=3;
    // rotated by identity inverse rotation -> camera.shift([-0.5,-0.6,0]).
    const { scene, camera } = makeScene({ shift: true });
    scene.on_mouse_motion([1, 2, 3], [0.1, 0.2, 0]);
    expect(camera.shifted.length).toBe(1);
    const s = camera.shifted[0] ?? [];
    expect(close(s[0] ?? NaN, -0.5)).toBe(true);
    expect(close(s[1] ?? NaN, -0.6)).toBe(true);
    expect(close(s[2] ?? NaN, 0)).toBe(true);
  });

  it("does not touch the camera without SHIFT", () => {
    const { scene, camera } = makeScene();
    scene.on_mouse_motion([1, 2, 3], [0.1, 0.2, 0]);
    expect(camera.shifted.length).toBe(0);
  });
});

describe("Scene.mouse_scroll_orbit_controls", () => {
  it("dollies toward the target on positive scroll (shift=[0,0,-0.11])", () => {
    // Ground truth (original manim, numpy): target=[0,0,0], pos=[0,0,11],
    // camera_to_target=[0,0,-11], *sign(+1), shift=0.01* -> [0,0,-0.11];
    // model = T(shift) @ model, so translation z becomes 10.89.
    const { scene, camera } = makeScene();
    scene.camera_target = [0, 0, 0];
    scene.mouse_scroll_orbit_controls([0, 0, 0], [0, 1, 0]);
    const m = camera.model_matrix.toArray() as unknown[][];
    expect(close(Number(m[2]?.[3]), 11 - 0.11)).toBe(true);
    expect(close(Number(m[0]?.[3]), 0)).toBe(true);
  });

  it("reverses direction on negative scroll", () => {
    const { scene, camera } = makeScene();
    scene.camera_target = [0, 0, 0];
    scene.mouse_scroll_orbit_controls([0, 0, 0], [0, -1, 0]);
    const m = camera.model_matrix.toArray() as unknown[][];
    expect(close(Number(m[2]?.[3]), 11 + 0.11)).toBe(true);
  });
});

describe("Scene.on_key_press", () => {
  it("resets the camera and target on 'r'", () => {
    const { scene, camera } = makeScene();
    scene.camera_target = [5, 5, 5];
    scene.on_key_press(114, 0); // ord('r')
    expect(camera.defaulted).toBe(1);
    expect(scene.camera_target).toEqual([0, 0, 0]);
  });

  it("sets quit_interaction on 'q'", () => {
    const { scene } = makeScene();
    expect(scene.quit_interaction).toBe(false);
    scene.on_key_press(113, 0); // ord('q')
    expect(scene.quit_interaction).toBe(true);
  });

  it("dispatches mapped keys and ignores unmapped ones", () => {
    const { scene } = makeScene();
    let calls = 0;
    scene.key_to_function_map.set("x", () => {
      calls += 1;
    });
    scene.on_key_press(120, 0); // ord('x')
    expect(calls).toBe(1);
    scene.on_key_press(121, 0); // ord('y') unmapped: no-op, no throw
    expect(calls).toBe(1);
    expect(scene.quit_interaction).toBe(false);
  });

  it("warns and returns for an out-of-range code point", () => {
    const { scene } = makeScene();
    // Python chr() raises for huge values (caught as warning path); TS
    // String.fromCodePoint throws RangeError, also mapped to warn+return.
    expect(() => scene.on_key_press(0x110000, 0)).not.toThrow();
    expect(scene.quit_interaction).toBe(false);
  });
});

describe("Scene.mouse_drag_orbit_controls", () => {
  it("right-drag (buttons=4) pans model matrix and target together", () => {
    // Ground truth (original manim, numpy): identity rotation, x_axis=[1,0,0],
    // d=[0.5,0.25,0] -> total=[-0.5,-0.25,0].
    const { scene, camera } = makeScene();
    camera.model_matrix = array([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 11],
      [0, 0, 0, 1],
    ]) as NDArray;
    scene.camera_target = [0, 0, 0];
    scene.mouse_drag_orbit_controls([0, 0, 0], [0.5, 0.25, 0], 4, 0);
    const m = camera.model_matrix.toArray() as unknown[][];
    expect(close(Number(m[0]?.[3]), -0.5)).toBe(true);
    expect(close(Number(m[1]?.[3]), -0.25)).toBe(true);
    expect(close(Number(m[2]?.[3]), 11)).toBe(true);
    expect(close(scene.camera_target[0] ?? NaN, -0.5)).toBe(true);
    expect(close(scene.camera_target[1] ?? NaN, -0.25)).toBe(true);
  });

  it("left-drag with pure z rotation matches the numpy derivation", () => {
    // Ground truth (original manim): M0=T(0,0,11), d=[0.1,0,0], target=0 ->
    // final = Rz(-0.1) @ M0 (polar check passes at 0 rad), i.e.
    // [[c,s,0,0],[-s,c,0,0],[0,0,1,11],...] with c=cos(.1), s=sin(.1).
    const { scene, camera } = makeScene();
    camera.model_matrix = array([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 11],
      [0, 0, 0, 1],
    ]) as NDArray;
    scene.camera_target = [0, 0, 0];
    scene.mouse_drag_orbit_controls([0, 0, 0], [0.1, 0, 0], 1, 0);
    const c = Math.cos(0.1);
    const s = Math.sin(0.1);
    expectRowsClose(camera.model_matrix, [
      [c, s, 0, 0],
      [-s, c, 0, 0],
      [0, 0, 1, 11],
      [0, 0, 0, 1],
    ]);
  });

  it("ignores other mouse buttons", () => {
    const { scene, camera } = makeScene();
    const before = (camera.model_matrix.toArray() as unknown[][]).map((r) =>
      [...(r as unknown[])].map(Number),
    );
    scene.mouse_drag_orbit_controls([0, 0, 0], [0.5, 0.5, 0], 2, 0);
    expectRowsClose(camera.model_matrix, before);
  });
});

// ---------------------------------------------------------------------------
// ThreeDScene
// Ground truth (original manim): ambient_camera_rotation=None,
// default kwargs {phi: 70*DEGREES, theta: -135*DEGREES}.
// ---------------------------------------------------------------------------
describe("ThreeDScene", () => {
  it("stores the default angled-camera orientation kwargs", () => {
    const scene = new ThreeDScene();
    expect(scene.ambient_camera_rotation).toBeNull();
    expect(close(scene.default_angled_camera_orientation_kwargs.phi ?? NaN, 70 * DEGREES)).toBe(true);
    expect(close(scene.default_angled_camera_orientation_kwargs.theta ?? NaN, -135 * DEGREES)).toBe(true);
  });

  it("accepts custom ambient rotation and orientation kwargs", () => {
    const scene = new ThreeDScene(
      undefined,
      0.5,
      { phi: 1.0, theta: 2.0 },
    );
    expect(scene.ambient_camera_rotation).toBe(0.5);
    expect(scene.default_angled_camera_orientation_kwargs).toEqual({ phi: 1.0, theta: 2.0 });
  });

  it("set_to_default_angled_camera_orientation forwards merged kwargs", () => {
    const scene = new ThreeDScene();
    const calls: Array<Record<string, unknown>> = [];
    scene.renderer = {
      camera: {
        set_phi: (v: number) => calls.push({ method: "set_phi", v }),
        set_theta: (v: number) => calls.push({ method: "set_theta", v }),
        set_gamma: (v: number) => calls.push({ method: "set_gamma", v }),
        set_zoom: (v: number) => calls.push({ method: "set_zoom", v }),
        set_focal_distance: (v: number) => calls.push({ method: "set_focal_distance", v }),
        _frame_center: { move_to: () => undefined },
      },
    };
    scene.set_to_default_angled_camera_orientation();
    const byMethod = Object.fromEntries(calls.map((c) => [c.method, c.v]));
    expect(close(byMethod["set_phi"] as number, 70 * DEGREES)).toBe(true);
    expect(close(byMethod["set_theta"] as number, -135 * DEGREES)).toBe(true);
    // Overrides win over the defaults.
    calls.length = 0;
    scene.set_to_default_angled_camera_orientation({ phi: 0.5 });
    const byMethod2 = Object.fromEntries(calls.map((c) => [c.method, c.v]));
    expect(byMethod2["set_phi"]).toBe(0.5);
    expect(close(byMethod2["set_theta"] as number, -135 * DEGREES)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// VectorScene
// Ground truth (original manim): basis_vector_stroke_width defaults to 6.0.
// position_*_coordinate tints the label (X/Y color) and returns it.
// ---------------------------------------------------------------------------
describe("VectorScene", () => {
  it("defaults basis_vector_stroke_width to 6.0", () => {
    expect(new VectorScene().basis_vector_stroke_width).toBe(6.0);
    expect(new VectorScene(2.5).basis_vector_stroke_width).toBe(2.5);
  });

  it("add_plane/add_axes register the objects on the scene", () => {
    const scene = new VectorScene();
    const plane = scene.add_plane();
    expect(scene.mobjects.includes(plane as unknown as Mobject)).toBe(true);
    const axes = scene.add_axes();
    expect(scene.mobjects.includes(axes as unknown as Mobject)).toBe(true);
  });

  it("position_x_coordinate tints with X_COLOR and returns the label", () => {
    const scene = new VectorScene();
    const label = new VMobject();
    const line = new VMobject();
    line.set_points_as_corners([
      [0, 0, 0],
      [1, 1, 0],
    ]);
    const ret = scene.position_x_coordinate(label, line, [1, 1, 0]);
    expect(ret).toBe(label);
    expect(label.get_color().to_hex()).toBe(X_COLOR.to_hex());
  });

  it("position_y_coordinate tints with Y_COLOR and returns the label", () => {
    const scene = new VectorScene();
    const label = new VMobject();
    const line = new VMobject();
    line.set_points_as_corners([
      [0, 0, 0],
      [1, 1, 0],
    ]);
    const ret = scene.position_y_coordinate(label, line, [1, 1, 0]);
    expect(ret).toBe(label);
    expect(label.get_color().to_hex()).toBe(Y_COLOR.to_hex());
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols working with previously-ported code.
// ---------------------------------------------------------------------------
describe("batch integration", () => {
  it("curve length pieces sum to the arc length (VMobject chain)", () => {
    // Python: get_arc_length() of the unit line is 1.0.
    const v = new VMobject();
    v.set_points_as_corners([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const piecesSum = toList(v.get_nth_curve_length_pieces(0)).reduce((a, b) => a + b, 0);
    expect(close(piecesSum, v.get_arc_length())).toBe(true);
    expect(close(v.get_arc_length(), 1)).toBe(true);
  });

  it("ValueTracker can drive a VMobject fill opacity end to end", () => {
    const tracker = new ValueTracker(0.2);
    tracker.increment_value(0.3); // -> 0.5, mirrors python 0.2+0.3
    const v = new VMobject({ color: RED });
    v.set_fill(RED, tracker.get_value());
    expect(close(v.get_fill_opacity(), 0.5)).toBe(true);
    expect(v.get_color().to_hex()).toBe("#FC6255");
  });

  it("VGroup of VectorizedPoints reports merged points via get_all_points", () => {
    const group = new VGroup(
      new VectorizedPoint([1, 0, 0]),
      new VectorizedPoint([0, 2, 0]),
    );
    expectRowsClose(group.get_all_points(), [
      [1, 0, 0],
      [0, 2, 0],
    ]);
  });

  it("ThreeDVMobject shade flag coexists with sheen rotation", () => {
    const t = new ThreeDVMobject();
    t.set_sheen_direction(UP);
    t.set_shade_in_3d(true);
    t.rotate_sheen_direction(Math.PI, [...OUT], false);
    expect(t.shade_in_3d).toBe(true);
    const dir = t.get_sheen_direction();
    expect(close(dir[1] ?? NaN, -1)).toBe(true);
  });

  it("Scene family lookup sees added mobjects and key dispatch works", () => {
    // NOTE: VectorScene.add_plane() registers a NumberPlane, but the TS
    // NumberPlane does not extend Mobject yet (Python: VGroup subclass),
    // so get_mobject_family_members() cannot traverse it. Use a plain
    // VMobject for the family-membership half of this test.
    const scene = new VectorScene();
    const mob = new VMobject();
    scene.add(mob);
    const members = scene.get_mobject_family_members();
    expect(members.includes(mob)).toBe(true);
    let calls = 0;
    scene.key_to_function_map.set("a", () => {
      calls += 1;
    });
    scene.on_key_press(97, 0);
    expect(calls).toBe(1);
  });

  it("RandomColorGenerator colors feed VMobject.set_fill round-trip", () => {
    const rnd = new RandomColorGenerator(42, [GREEN]);
    const picked = rnd.next();
    const v = new VMobject();
    v.set_fill(picked, 1);
    expect(v.get_color().to_hex()).toBe("#83C167");
    void WHITE;
  });
});
