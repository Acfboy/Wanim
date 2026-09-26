import { array, type NDArray } from "numpy-ts";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { polylabel } from "../../utils/polylabel.js";
import { VGroup } from "../types/vectorized_mobject.js";
import { Polygram, type PolygramOptions } from "./polygram.js";

/**
 * A label consisting of text surrounded by a frame.
 *
 * NOTE: text/tex rendering (MathTex, Text, Typst) is out of scope for the
 * entire port, and this class exists to display a text label — so the
 * constructor is a clearly-marked stub until text support lands. The
 * signature mirrors the Python original.
 */
export class Label extends VGroup {
  rendered_label!: unknown;
  background_rect!: unknown;
  frame!: unknown;

  constructor(
    label: string | unknown,
    label_config: Record<string, unknown> | null = null,
    box_config: Record<string, unknown> | null = null,
    frame_config: Record<string, unknown> | null = null,
    options: Record<string, unknown> = {},
  ) {
    super();
    void label;
    void label_config;
    void box_config;
    void frame_config;
    void options;
    throw new Error("TODO: text/tex not ported yet");
  }
}

/** Exact element-wise comparison of two points, mirroring np.array_equal. */
function pointsEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

export interface LabeledPolygramOptions extends PolygramOptions {
  label?: unknown;
  precision?: number;
  label_config?: Record<string, unknown> | null;
  box_config?: Record<string, unknown> | null;
  frame_config?: Record<string, unknown> | null;
}

/**
 * Constructs a polygram containing a label box at its pole of inaccessibility.
 *
 * NOTE: the label itself is rendered text (MathTex/Text/Typst in the Python
 * original), so constructing this class currently throws via the Label stub
 * (text/tex is out of scope for the entire port). The pole/radius computation
 * below mirrors the original verbatim.
 */
export class LabeledPolygram extends Polygram {
  label!: Label;
  pole!: NDArray;
  radius!: number;

  constructor(...args: unknown[]) {
    let options: LabeledPolygramOptions = {};
    let vertexGroups: unknown[] = args;
    const last = args[args.length - 1];
    if (
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last)
    ) {
      options = last as LabeledPolygramOptions;
      vertexGroups = args.slice(0, -1);
    }
    const {
      label,
      precision = 0.01,
      label_config = null,
      box_config = null,
      frame_config = null,
      ...rest
    } = options;
    if (label === undefined) {
      throw new TypeError(
        "LabeledPolygram.__init__() missing 1 required keyword-only argument: 'label'",
      );
    }
    super(...vertexGroups, rest);

    // Create Label
    this.label = new Label(label as string, label_config, box_config, frame_config);

    // Close Vertex Groups
    const rings: number[][][] = vertexGroups.map((group) => {
      const rows = pointRows(group, this.dim);
      const first = rows[0];
      const lastRow = rows[rows.length - 1];
      if (
        first !== undefined &&
        lastRow !== undefined &&
        pointsEqual(first, lastRow)
      ) {
        return rows;
      }
      return first === undefined ? rows : [...rows, [...first]];
    });

    // Compute the Pole of Inaccessibility
    const cell = polylabel(rings, precision);
    // NOTE: mirrors np.pad(cell.c, (0, 1), "constant") on the 2D pole.
    const center = (cell.c.toArray() as number[]).map(Number);
    this.pole = array([center[0] ?? 0, center[1] ?? 0, 0]) as NDArray;
    this.radius = cell.d;

    // Position the label at the pole
    this.label.move_to(this.pole);
    this.add(this.label);
  }
}
