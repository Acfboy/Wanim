// Port of manim.mobject.matrix (Matrix and helpers).
//
// Element mobjects default to MathTex, and the brackets/determinant helpers
// use MathTex/Tex. Text rendering is provided by the ported MathJax/MathTex
// pipeline, so no LaTeX install is required.
import { DOWN, DR, LEFT, MED_SMALL_BUFF, RIGHT } from "../constants.js";
import type { ParsableManimColor } from "../utils/color/core.js";
import { BackgroundRectangle } from "./geometry/shape_matchers.js";
import { setMatrixFactory } from "./geometry/line.js";
import { VGroup, VMobject } from "./types/vectorized_mobject.js";
import { MathTex, Tex, type MathTexOptions } from "./text/tex_mobject.js";
import {
  DecimalNumber,
  Integer,
  type DecimalNumberOptions,
  type DecimalNumberValue,
} from "./text/numbers.js";

export type MatrixElementFactory = (
  item: unknown,
  config?: Record<string, unknown>,
) => VMobject;

function mathTexElement(item: unknown, config: Record<string, unknown> = {}): VMobject {
  // NOTE: mirrors Python `element_to_mobject(item, **config)` with the
  // default `element_to_mobject=MathTex`.
  return new MathTex(String(item), config as MathTexOptions);
}

function decimalNumberElement(item: unknown, config: Record<string, unknown> = {}): VMobject {
  return new DecimalNumber(item as DecimalNumberValue, config as DecimalNumberOptions);
}

function integerElement(item: unknown, config: Record<string, unknown> = {}): VMobject {
  return new Integer(item as DecimalNumberValue, config as DecimalNumberOptions);
}

export function matrix_to_tex_string(matrix: unknown): string {
  // NOTE: mirrors Python `np.array(matrix).astype("str")` plus the
  // `ndim == 1` reshape to a single column.
  let rows: unknown[][];
  if (Array.isArray(matrix) && matrix.length > 0 && !Array.isArray(matrix[0])) {
    rows = (matrix as unknown[]).map((item) => [item]);
  } else {
    rows = (matrix ?? []) as unknown[][];
  }
  const stringRows = rows.map((row) => (row as unknown[]).map((item) => String(item)));
  const nCols = stringRows.length > 0 ? stringRows[0]!.length : 0;
  const prefix = `\\left[ \\begin{array}{${"c".repeat(nCols)}}`;
  const suffix = "\\end{array} \\right]";
  return prefix + stringRows.map((row) => row.join(" & ")).join(" \\\\ ") + suffix;
}

export function matrix_to_mobject(matrix: unknown): MathTex {
  return new MathTex(matrix_to_tex_string(matrix));
}

export interface MatrixOptions {
  v_buff?: number;
  h_buff?: number;
  bracket_h_buff?: number;
  bracket_v_buff?: number;
  add_background_rectangles_to_entries?: boolean;
  include_background_rectangle?: boolean;
  element_to_mobject?: MatrixElementFactory;
  element_to_mobject_config?: Record<string, unknown>;
  element_alignment_corner?: readonly number[];
  left_bracket?: string;
  right_bracket?: string;
  stretch_brackets?: boolean;
  bracket_config?: Record<string, unknown>;
  [key: string]: unknown;
}

export class Matrix extends VMobject {
  v_buff: number;
  h_buff: number;
  bracket_h_buff: number;
  bracket_v_buff: number;
  add_background_rectangles_to_entries: boolean;
  include_background_rectangle: boolean;
  element_to_mobject: MatrixElementFactory;
  element_to_mobject_config: Record<string, unknown>;
  element_alignment_corner: readonly number[];
  left_bracket: string;
  right_bracket: string;
  stretch_brackets: boolean;
  elements!: VGroup;
  mob_matrix!: VMobject[][];
  brackets!: VGroup;

  constructor(matrix: unknown[][], options: MatrixOptions = {}) {
    const {
      v_buff = 0.8,
      h_buff = 1.3,
      bracket_h_buff = MED_SMALL_BUFF,
      bracket_v_buff = MED_SMALL_BUFF,
      add_background_rectangles_to_entries = false,
      include_background_rectangle = false,
      element_to_mobject = mathTexElement,
      element_to_mobject_config = {},
      element_alignment_corner = [...DR],
      left_bracket = "[",
      right_bracket = "]",
      stretch_brackets = true,
      bracket_config = {},
      ...kwargs
    } = options;
    super(kwargs);
    this.v_buff = v_buff;
    this.h_buff = h_buff;
    this.bracket_h_buff = bracket_h_buff;
    this.bracket_v_buff = bracket_v_buff;
    this.add_background_rectangles_to_entries = add_background_rectangles_to_entries;
    this.include_background_rectangle = include_background_rectangle;
    this.element_to_mobject = element_to_mobject;
    this.element_to_mobject_config = element_to_mobject_config;
    this.element_alignment_corner = element_alignment_corner;
    this.left_bracket = left_bracket;
    this.right_bracket = right_bracket;
    this.stretch_brackets = stretch_brackets;
    const mobMatrix = this._matrix_to_mob_matrix(matrix);
    this._organize_mob_matrix(mobMatrix);
    this.elements = new VGroup(...mobMatrix.flat());
    this.add(this.elements);
    this._add_brackets(this.left_bracket, this.right_bracket, bracket_config);
    this.center();
    this.mob_matrix = mobMatrix;
    if (this.add_background_rectangles_to_entries) {
      this.add_background_to_entries();
    }
    if (this.include_background_rectangle) {
      this.add_background_rectangle();
    }
  }

  _matrix_to_mob_matrix(matrix: unknown[][]): VMobject[][] {
    return (matrix as unknown[][]).map((row) =>
      (row as unknown[]).map((item) =>
        this.element_to_mobject(item, this.element_to_mobject_config),
      ),
    );
  }

  _organize_mob_matrix(matrix: VMobject[][]): this {
    for (let i = 0; i < matrix.length; i += 1) {
      const row = matrix[i]!;
      for (let j = 0; j < row.length; j += 1) {
        const mob = row[j]!;
        // NOTE: mirrors Python `i * self.v_buff * DOWN + j * self.h_buff *
        // RIGHT` with DOWN = [0, -1, 0], RIGHT = [1, 0, 0].
        mob.move_to(
          [
            j * this.h_buff * (RIGHT[0] as number),
            i * this.v_buff * (DOWN[1] as number),
            0,
          ],
          this.element_alignment_corner,
        );
      }
    }
    return this;
  }

  _add_brackets(left = "[", right = "]", kwargs: Record<string, unknown> = {}): this {
    // Height per row of LaTeX array with default settings.
    const BRACKET_HEIGHT = 0.5977;

    const n = Math.floor(this.height / BRACKET_HEIGHT) + 1;
    const emptyTexArray = [
      "\\begin{array}{c}",
      ...Array.from({ length: n }, () => "\\quad \\\\"),
      "\\end{array}",
    ].join("");
    const texLeft = ["\\left" + left, emptyTexArray, "\\right."].join("");
    const texRight = ["\\left.", emptyTexArray, "\\right" + right].join("");
    const lBracket = new MathTex(texLeft, kwargs as MathTexOptions);
    const rBracket = new MathTex(texRight, kwargs as MathTexOptions);

    const bracketPair = new VGroup(lBracket, rBracket);
    if (this.stretch_brackets) {
      bracketPair.stretch_to_fit_height(this.height + 2 * this.bracket_v_buff);
    }
    lBracket.next_to(this, [...LEFT], this.bracket_h_buff);
    rBracket.next_to(this, [...RIGHT], this.bracket_h_buff);
    this.brackets = bracketPair;
    this.add(lBracket, rBracket);
    return this;
  }

  get_columns(): VGroup {
    return new VGroup(
      ...Array.from({ length: this.mob_matrix[0]!.length }, (_, i) =>
        new VGroup(...this.mob_matrix.map((row) => row[i]!)),
      ),
    );
  }

  set_column_colors(...colors: ParsableManimColor[]): this {
    const columns = this.get_columns();
    const count = Math.min(colors.length, columns.submobjects.length);
    for (let index = 0; index < count; index += 1) {
      (columns.submobjects[index] as VMobject).set_color(colors[index]!);
    }
    return this;
  }

  get_rows(): VGroup {
    return new VGroup(...this.mob_matrix.map((row) => new VGroup(...row)));
  }

  set_row_colors(...colors: ParsableManimColor[]): this {
    const rows = this.get_rows();
    const count = Math.min(colors.length, rows.submobjects.length);
    for (let index = 0; index < count; index += 1) {
      (rows.submobjects[index] as VMobject).set_color(colors[index]!);
    }
    return this;
  }

  add_background_to_entries(): this {
    // NOTE: mirrors Python `mob.add_background_rectangle()` (defaults:
    // color=None -> background color, opacity 0.75). Constructed directly
    // instead of going through the global factory hook, which stays unwired
    // (same convention as Table.add_background_to_entries).
    for (const mob of this.get_entries()) {
      const target = mob as VMobject;
      const rect = new BackgroundRectangle(target, {});
      target.background_rectangle = rect;
      target.add_to_back(rect);
    }
    return this;
  }

  get_mob_matrix(): VMobject[][] {
    return this.mob_matrix;
  }

  get_entries(): VGroup {
    return this.elements;
  }

  get_brackets(): VGroup {
    return this.brackets;
  }
}

export interface DecimalMatrixOptions extends MatrixOptions {}

export class DecimalMatrix extends Matrix {
  constructor(matrix: unknown[][], options: DecimalMatrixOptions = {}) {
    const {
      element_to_mobject = decimalNumberElement,
      element_to_mobject_config = { num_decimal_places: 1 },
      ...rest
    } = options;
    super(matrix, {
      element_to_mobject,
      element_to_mobject_config,
      ...rest,
    });
  }
}

export interface IntegerMatrixOptions extends MatrixOptions {}

export class IntegerMatrix extends Matrix {
  constructor(matrix: unknown[][], options: IntegerMatrixOptions = {}) {
    const { element_to_mobject = integerElement, ...rest } = options;
    super(matrix, { element_to_mobject, ...rest });
  }
}

export interface MobjectMatrixOptions extends MatrixOptions {}

export class MobjectMatrix extends Matrix {
  constructor(matrix: unknown[][], options: MobjectMatrixOptions = {}) {
    const {
      element_to_mobject = ((m: unknown) => m) as MatrixElementFactory,
      ...rest
    } = options;
    super(matrix, { element_to_mobject, ...rest });
  }
}

export function get_det_text(
  matrix: Matrix,
  determinant: number | string | null = null,
  backgroundRect = false,
  initialScaleFactor = 2,
): VGroup {
  const parens = new MathTex("(", ")");
  parens.scale(initialScaleFactor);
  parens.stretch_to_fit_height(matrix.height);
  // NOTE: mirrors Python `l_paren, r_paren = parens` (unpacking the two
  // MathTex parts).
  const lParen = parens.submobjects[0] as VMobject;
  const rParen = parens.submobjects[1] as VMobject;
  lParen.next_to(matrix, [...LEFT], 0.1);
  rParen.next_to(matrix, [...RIGHT], 0.1);
  const det = new Tex("det");
  det.scale(initialScaleFactor);
  det.next_to(lParen, [...LEFT], 0.1);
  if (backgroundRect) {
    det.add_background_rectangle();
  }
  const detText = new VGroup(det, lParen, rParen);
  if (determinant !== null && determinant !== undefined) {
    const eq = new MathTex("=");
    eq.next_to(rParen, [...RIGHT], 0.1);
    const result = new MathTex(String(determinant));
    result.next_to(eq, [...RIGHT], 0.2);
    detText.add(eq, result);
  }
  return detText;
}

// Registers the Matrix builder used by Vector.coordinate_label (see line.ts).
setMatrixFactory((matrix, options = {}) => new Matrix(matrix, options));
