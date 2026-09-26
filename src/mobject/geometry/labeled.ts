import { array, type NDArray } from "numpy-ts";
import { LEFT, MED_SMALL_BUFF, RIGHT } from "../../constants.js";
import { config } from "../../_config/utils.js";
import { RendererType } from "../../constants.js";
import { isNDArray, pointRows, subtractPoints } from "../../utils/arrays.js";
import { polylabel } from "../../utils/polylabel.js";
import type { Mobject } from "../mobject.js";
import { VGroup } from "../types/vectorized_mobject.js";
import { Arrow, Line, type LineEndpoint, type LineOptions } from "./line.js";
import { ArrowTriangleFilledTip } from "./tips.js";
import type { TipConstructor } from "./arc.js";
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

function isLineOptionsObject(value: unknown): value is LineOptions {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (value instanceof VGroup) return false;
  if (isNDArray(value)) return false;
  return true;
}

export interface LabeledLineOptions extends LineOptions {
  label_position?: number;
  label_config?: Record<string, unknown> | null;
  box_config?: Record<string, unknown> | null;
  frame_config?: Record<string, unknown> | null;
}

export class LabeledLine extends Line {
  label!: Label;

  constructor(
    label: string | unknown,
    label_position = 0.5,
    label_config: Record<string, unknown> | null = null,
    box_config: Record<string, unknown> | null = null,
    frame_config: Record<string, unknown> | null = null,
    ...lineArgs: unknown[]
  ) {
    // NOTE: mirrors Python `LabeledLine(label, label_position=0.5,
    // label_config=None, box_config=None, frame_config=None, *args,
    // **kwargs)`: trailing line args are (start, end, buff/path_arc...,
    // options). An options bag may also carry the label_* keys, which win
    // over the positionals (same convention as Line/Arrow).
    let start: LineEndpoint = [...LEFT];
    let end: LineEndpoint = [...RIGHT];
    let positionalBuff: number | undefined;
    let positionalPathArc: number | undefined;
    let lineOptions: LineOptions = {};
    let resolved_position = label_position;
    let resolved_label_config = label_config;
    let resolved_box_config = box_config;
    let resolved_frame_config = frame_config;
    const positional = [...lineArgs];
    const last = positional[positional.length - 1];
    if (positional.length > 0 && isLineOptionsObject(last)) {
      const {
        label_position: optPosition,
        label_config: optLabelConfig,
        box_config: optBoxConfig,
        frame_config: optFrameConfig,
        ...rest
      } = last as LabeledLineOptions;
      lineOptions = rest as LineOptions;
      if (optPosition !== undefined) resolved_position = optPosition;
      if (optLabelConfig !== undefined) resolved_label_config = optLabelConfig as Record<string, unknown> | null;
      if (optBoxConfig !== undefined) resolved_box_config = optBoxConfig as Record<string, unknown> | null;
      if (optFrameConfig !== undefined) resolved_frame_config = optFrameConfig as Record<string, unknown> | null;
      positional.pop();
    }
    if (positional.length > 0) start = positional[0] as LineEndpoint;
    if (positional.length > 1) end = positional[1] as LineEndpoint;
    if (positional.length > 2) {
      if (typeof positional[2] === "number") positionalBuff = positional[2] as number;
      else Object.assign(lineOptions, positional[2] as LineOptions);
    }
    if (positional.length > 3) {
      if (typeof positional[3] === "number") positionalPathArc = positional[3] as number;
      else Object.assign(lineOptions, positional[3] as LineOptions);
    }
    super(start, end, positionalBuff ?? lineOptions.buff ?? 0, positionalPathArc ?? lineOptions.path_arc ?? 0, lineOptions);

    // Create Label (text/tex rendering is out of scope; the Label stub
    // throws here exactly where Python would build MathTex/Text).
    this.label = new Label(
      label as string,
      resolved_label_config,
      resolved_box_config,
      resolved_frame_config,
    );

    // Compute Label Position
    const [line_start, line_end] = this.get_start_and_end();
    const new_vec = subtractPoints(line_end, line_start, this.dim).map(
      (value) => value * (resolved_position ?? 0.5),
    );
    const label_coords = line_start.map((value, index) => value + (new_vec[index] ?? 0));
    this.label.move_to(label_coords);
    this.add(this.label);
  }
}

export interface LabeledArrowOptions extends LabeledLineOptions {
  stroke_width?: number;
  buff?: number;
  tip_shape?: TipConstructor | null;
  max_tip_length_to_length_ratio?: number;
  max_stroke_width_to_length_ratio?: number;
}

function isPlainOptionsObject(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (value instanceof VGroup) return false;
  if (isNDArray(value)) return false;
  // NOTE: stricter than isLineOptionsObject: class instances (Mobject
  // endpoints, ArrowTip, ...) are not plain option bags even though they
  // are non-array objects.
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Constructs an arrow containing a label box somewhere along its length.
 *
 * NOTE: mirrors Python `class LabeledArrow(LabeledLine, Arrow)` via the MRO
 * LabeledArrow -> LabeledLine -> Arrow -> Line: label arguments are consumed
 * exactly like LabeledLine, while stroke/tip defaults follow Arrow
 * (stroke_width=6, buff=MED_SMALL_BUFF, ArrowTriangleFilledTip end tip with
 * length-proportional stroke narrowing).
 */
export class LabeledArrow extends LabeledLine {
  max_tip_length_to_length_ratio: number;
  max_stroke_width_to_length_ratio: number;
  initial_stroke_width: number;

  constructor(
    label: string | unknown,
    label_position = 0.5,
    label_config: Record<string, unknown> | null = null,
    box_config: Record<string, unknown> | null = null,
    frame_config: Record<string, unknown> | null = null,
    ...lineArgs: unknown[]
  ) {
    // Inject Arrow.__init__ defaults (stroke_width=6, buff=MED_SMALL_BUFF)
    // wherever the caller left them unspecified, then let LabeledLine parse
    // the line arguments as usual. An explicit option key wins, mirroring
    // Python's **kwargs flow into Arrow.__init__.
    const forwarded = [...lineArgs];
    const last = forwarded[forwarded.length - 1];
    let tip_shape: TipConstructor | null | undefined;
    if (forwarded.length > 0 && isPlainOptionsObject(last)) {
      const bag = { ...(last as LabeledArrowOptions) };
      if (bag.stroke_width === undefined) bag.stroke_width = 6;
      if (bag.buff === undefined) bag.buff = MED_SMALL_BUFF;
      tip_shape = bag.tip_shape ?? undefined;
      delete bag.tip_shape;
      delete bag.max_tip_length_to_length_ratio;
      delete bag.max_stroke_width_to_length_ratio;
      forwarded[forwarded.length - 1] = bag;
    } else {
      forwarded.push({ stroke_width: 6, buff: MED_SMALL_BUFF } as LabeledArrowOptions);
    }
    super(label, label_position, label_config, box_config, frame_config, ...forwarded);
    const bagAfter = forwarded[forwarded.length - 1] as LabeledArrowOptions;
    this.max_tip_length_to_length_ratio =
      (typeof bagAfter === "object" && bagAfter !== null
        ? (bagAfter as LabeledArrowOptions).max_tip_length_to_length_ratio
        : undefined) ?? 0.25;
    this.max_stroke_width_to_length_ratio =
      (typeof bagAfter === "object" && bagAfter !== null
        ? (bagAfter as LabeledArrowOptions).max_stroke_width_to_length_ratio
        : undefined) ?? 5;
    // TODO, should this be affected when Arrow.set_stroke is called?
    this.initial_stroke_width = this.stroke_width;
    this.add_tip(null, tip_shape ?? ArrowTriangleFilledTip);
    this._set_stroke_width_from_length();
  }

  override get_default_tip_length(): number {
    const max_ratio = this.max_tip_length_to_length_ratio;
    return Math.min(this.tip_length, max_ratio * this.get_length());
  }

  _set_stroke_width_from_length(): this {
    const max_ratio = this.max_stroke_width_to_length_ratio;
    const width = Math.min(this.initial_stroke_width, max_ratio * this.get_length());
    if (config.renderer === RendererType.OPENGL) {
      this.set_stroke(undefined, width, undefined, false);
    } else {
      this.set_stroke(undefined, width, undefined, false);
    }
    return this;
  }
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
