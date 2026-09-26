// Port of manim.mobject.matrix (Matrix and helpers).
//
// NOTE (text/tex out of scope): element mobjects default to MathTex and the
// brackets/determinant helpers need MathTex/Tex (a LaTeX install). The
// layout logic itself (_matrix_to_mob_matrix, _organize_mob_matrix,
// get_columns/get_rows, row/column colors, entries) is pure mobject code and
// is ported faithfully; every branch that would construct rendered text
// throws `TODO: text/tex not ported yet` instead.
import { DOWN, DR, MED_SMALL_BUFF, RIGHT } from "../constants.js";
import type { ParsableManimColor } from "../utils/color/core.js";
import { Mobject } from "./mobject.js";
import { VGroup, VMobject } from "./types/vectorized_mobject.js";

export type MatrixElementFactory = (
  item: unknown,
  config?: Record<string, unknown>,
) => VMobject;

function texNotPorted(name: string): () => never {
  return () => {
    throw new Error(`TODO: text/tex not ported yet (${name})`);
  };
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

export function matrix_to_mobject(_matrix: unknown): VMobject {
  throw new Error("TODO: text/tex not ported yet (matrix_to_mobject needs MathTex)");
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
      element_to_mobject = texNotPorted("Matrix element_to_mobject defaults to MathTex"),
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
      for (const mob of this.elements) {
        (mob as VMobject).add_background_rectangle();
      }
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

  _add_brackets(_left = "[", _right = "]", _kwargs: Record<string, unknown> = {}): this {
    throw new Error("TODO: text/tex not ported yet (Matrix._add_brackets needs MathTex)");
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
    for (const mob of this.get_entries()) {
      (mob as VMobject).add_background_rectangle();
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
      element_to_mobject = texNotPorted("DecimalMatrix element_to_mobject defaults to DecimalNumber"),
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
    const {
      element_to_mobject = texNotPorted("IntegerMatrix element_to_mobject defaults to Integer"),
      ...rest
    } = options;
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
  _matrix: Matrix,
  _determinant?: number | string | null,
  _backgroundRect = false,
  _initialScaleFactor = 2,
): VGroup {
  throw new Error("TODO: text/tex not ported yet (get_det_text needs MathTex/Tex)");
}
