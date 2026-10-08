import { describe, expect, it } from "vitest";
import { ThreeDCamera, _cornerNormalsFlat } from "../src/camera/three_d_camera.js";
import { get_shaded_rgb } from "../src/utils/color/core.js";
import { pointRows } from "../src/utils/arrays.js";
import { ThreeDAxes } from "../src/mobject/graphing/coordinate_systems.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import {
  get_3d_vmob_unit_normal,
  has_unit_normal_override,
} from "../src/mobject/three_d/three_d_utils.js";

const ROOT_THIRD = 1 / Math.sqrt(3);

function lightPoint(camera: ThreeDCamera): number[] {
  return pointRows(camera.light_source.points, 3)[0] ?? [0, 0, 0];
}

/** Colour `get_shaded_rgb` produces for a white base at `corner` with the
 *  `[1, 1, 1]` override normal (normalized). */
function expectedOverrideColor(camera: ThreeDCamera, corner: number[]): number[] {
  return [
    ...get_shaded_rgb([1, 1, 1], corner, [ROOT_THIRD, ROOT_THIRD, ROOT_THIRD], lightPoint(camera)),
    1,
  ];
}

function axisPieces(axes: ThreeDAxes, index: 0 | 1 | 2): VMobject[] {
  const axis = [axes.get_x_axis(), axes.get_y_axis(), axes.get_z_axis()][index] as unknown as {
    pieces: { submobjects: VMobject[] };
  };
  return axis.pieces.submobjects;
}

describe("ThreeDAxes axis shading (per-mobject get_unit_normal override)", () => {
  it("applies ThreeDAxes._set_axis_shading: own get_unit_normal is installed", () => {
    const axes = new ThreeDAxes();
    const pieces = axisPieces(axes, 0);
    expect(pieces).toHaveLength(20);
    for (const piece of pieces) {
      expect(has_unit_normal_override(piece)).toBe(true);
      expect(piece.get_unit_normal([0, 0, 0])).toEqual([1, 1, 1]);
    }
  });

  it("shades each x-axis piece with the normalized [1,1,1] override", () => {
    const axes = new ThreeDAxes();
    const camera = new ThreeDCamera();
    for (const piece of axisPieces(axes, 0)) {
      const corner = pointRows(piece.points, 3)[0]!;
      const rows = camera.get_stroke_rgbas(piece).toArray() as number[][];
      const expected = expectedOverrideColor(camera, corner);
      for (let channel = 0; channel < 4; channel += 1) {
        expect(rows[0]![channel]).toBeCloseTo(expected[channel]!, 10);
      }
      // The override is uniform across the piece; both stops match.
      expect(rows[1]).toEqual(rows[0]);
    }
  });

  it("reports the overridden gradient endpoints, not the geometric corners", () => {
    const axes = new ThreeDAxes();
    const camera = new ThreeDCamera();
    const xAxis = axes.get_x_axis() as unknown as VMobject;
    const piece = axisPieces(axes, 0)[0]!;
    const expected = (piece.get_gradient_start_and_end_points() as number[][]).map((p) =>
      p.map((value) => Math.round(value * 1e6) / 1e6),
    );
    const info = camera.get_draw_color_info(piece, piece.get_stroke_rgbas(false));
    expect(info).not.toBeNull();
    expect(info!.start.map((value) => Math.round(value * 1e6) / 1e6)).toEqual(expected[0]);
    expect(info!.end.map((value) => Math.round(value * 1e6) / 1e6)).toEqual(expected[1]);
    // The default light source gradient spans the x-axis edges.
    expect(expected).toEqual([[5.25, 0.175, 0], [-5.25, -0.175, 0]]);
    expect(xAxis).toBeInstanceOf(VMobject);
  });

  it("adjacent pieces form a smooth monotonic gradient (no alternating dark/light)", () => {
    const axes = new ThreeDAxes();
    const camera = new ThreeDCamera();
    for (const index of [0, 1, 2] as const) {
      const reds = axisPieces(axes, index).map((piece) => {
        const rows = camera.get_stroke_rgbas(piece).toArray() as number[][];
        return rows[0]![0]!;
      });
      const deltas = reds.slice(1).map((value, i) => value - reds[i]!);
      const direction = Math.sign(deltas[0] ?? 0);
      expect(direction).not.toBe(0);
      for (const delta of deltas) {
        expect(Math.abs(delta)).toBeLessThan(0.02);
        expect(Math.sign(delta)).toBe(direction);
      }
      const spread = Math.max(...reds) - Math.min(...reds);
      expect(spread).toBeLessThan(0.25);
      expect(spread).toBeGreaterThan(0.005);
    }
  });

  it("get_3d_vmob_unit_normal and _cornerNormalsFlat honour the override", () => {
    const axes = new ThreeDAxes();
    const piece = axisPieces(axes, 1)[0]!;
    const normal = get_3d_vmob_unit_normal(piece, 0).toArray() as number[];
    expect(normal[0]).toBeCloseTo(ROOT_THIRD, 12);
    expect(normal[1]).toBeCloseTo(ROOT_THIRD, 12);
    expect(normal[2]).toBeCloseTo(ROOT_THIRD, 12);
    const flat = _cornerNormalsFlat(piece.points, piece)!;
    expect(flat.normal0[0]).toBeCloseTo(ROOT_THIRD, 12);
    expect(flat.normal1[1]).toBeCloseTo(ROOT_THIRD, 12);
  });

  it("plain shade_in_3d objects keep the geometric normal path", () => {
    const axes = new ThreeDAxes();
    const piece = axisPieces(axes, 0)[0]!;
    // Fresh copy without the installed override: same geometry, no override.
    const plain = new VMobject();
    plain.set_points(piece.points.copy() as never);
    plain.set_shade_in_3d(true);
    expect(has_unit_normal_override(plain)).toBe(false);
    expect(get_3d_vmob_unit_normal(plain, 0).toArray()).toEqual([0, 1, 0]);
    const flat = _cornerNormalsFlat(plain.points, plain)!;
    expect(flat.normal0).toEqual([0, 1, 0]);
    expect(flat.normal1).toEqual([0, 1, 0]);
  });

  it("a zero-vector override falls back to the geometric normal", () => {
    const axes = new ThreeDAxes();
    const piece = axisPieces(axes, 0)[0]!;
    (piece as unknown as { get_unit_normal: (p: unknown) => number[] }).get_unit_normal = () => [0, 0, 0];
    expect(get_3d_vmob_unit_normal(piece, 0).toArray()).toEqual([0, 1, 0]);
    const camera = new ThreeDCamera();
    const info = camera.get_draw_color_info(piece, piece.get_stroke_rgbas(false))!;
    const corner = pointRows(piece.points, 3)[0]!;
    const expected = get_shaded_rgb([1, 1, 1], corner, [0, 1, 0], lightPoint(camera));
    for (let channel = 0; channel < 3; channel += 1) {
      expect(info.colors[0]![channel]).toBeCloseTo(expected[channel]!, 10);
    }
  });

  it("a custom override drives get_mobject_base_colors normals", () => {
    const axes = new ThreeDAxes();
    const piece = axisPieces(axes, 0)[0] as unknown as VMobject;
    (piece as unknown as { get_unit_normal: (p: unknown) => number[] }).get_unit_normal = () => [0, 1, 1];
    const camera = new ThreeDCamera();
    const base = camera.get_mobject_base_colors(piece)!;
    const expected = 1 / Math.sqrt(2);
    expect(base.stroke!.normal0[1]).toBeCloseTo(expected, 12);
    expect(base.stroke!.normal0[2]).toBeCloseTo(expected, 12);
    expect(base.fill!.normal0).toEqual(base.stroke!.normal0);
  });
});
