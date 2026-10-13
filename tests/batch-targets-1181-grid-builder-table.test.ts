// Ground-truth tests for batch layers 16..16 (~1181 lines):
//   manim.mobject.mobject.Mobject.arrange_in_grid
//   manim.mobject.mobject._AnimationBuilder.build
//   manim.mobject.table.Table (+ MobjectTable subclass paths)
//
// Every numeric expectation was captured from the ORIGINAL Python manim
// (original manim) with the inputs used here;
// each block cites the probe in a "Python:" comment. TS is asserted against
// those values (1e-9 relative for floats; exact for discrete; arrays
// element-wise).
//
// OUT OF SCOPE (text/tex): Table/MathTable/IntegerTable/DecimalTable default
// element_to_mobject factories (Paragraph/MathTex/Integer/DecimalNumber) need
// the LaTeX pipeline and throw `TODO: text/tex not ported yet` in TS. Those
// defaults are only asserted to throw; all layout tests use MobjectTable with
// an identity factory over Squares (no tex involved).
import { describe, expect, it } from "vitest";
import { pointRows } from "../src/utils/arrays.js";
// NOTE (import order matters): the first VALUE import must reach
// mobject.js before the animation modules (mobject <-> animation form a
// pre-existing cycle that only resolves mobject-first; esbuild drops the
// type-only mobject.js import below, so the value-used VMobject/Square
// imports lead). See the same note in src/mobject/table.ts.
import { VMobject, VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Mobject, Group, _AnimationBuilder } from "../src/mobject/mobject.js";
import { Animation } from "../src/animation/animation.js";
import { _MethodAnimation as TransformMethodAnimation } from "../src/animation/transform.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { Table, MobjectTable, MathTable, IntegerTable, DecimalTable } from "../src/mobject/table.js";

function close(a: number, b: number, tol = 1e-9): boolean {
  if (Object.is(a, b)) return true;
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom <= tol;
}

function expectClose(actual: number, expected: number, tol = 1e-9): void {
  expect(close(actual, expected, tol)).toBe(true);
}

function expectCenter2(actual: number[], expected: number[]): void {
  expectClose(actual[0] ?? NaN, expected[0] ?? NaN);
  expectClose(actual[1] ?? NaN, expected[1] ?? NaN);
  expectClose(actual[2] ?? NaN, expected[2] ?? 0);
}

function centers2(group: VGroup | Group): number[][] {
  return (group.submobjects as Mobject[]).map((m) => m.get_center());
}

const idFactory = (m: unknown) => m as VMobject;

function squares(n: number): Square[] {
  return Array.from({ length: n }, () => new Square());
}

// ---------------------------------------------------------------------------
// Mobject.arrange_in_grid
// Python:
//   VGroup(*[Square() for _ in range(6)]).arrange_in_grid(rows=2, cols=3, buff=0.1)
//   -> centers (-2.1,1.05),(0,1.05),(2.1,1.05),(-2.1,-1.05),(0,-1.05),(2.1,-1.05)
//   (middle x prints as -4.44e-16, i.e. 0 within tolerance)
// ---------------------------------------------------------------------------
describe("Mobject.arrange_in_grid basic layout", () => {
  it("lays 6 squares out in a 2x3 grid with buff=0.1", () => {
    const boxes = new VGroup(...squares(6));
    const ret = boxes.arrange_in_grid(2, 3, 0.1);
    expect(ret).toBe(boxes);
    const got = centers2(boxes);
    const want = [
      [-2.1, 1.05, 0],
      [0, 1.05, 0],
      [2.1, 1.05, 0],
      [-2.1, -1.05, 0],
      [0, -1.05, 0],
      [2.1, -1.05, 0],
    ];
    expect(got.length).toBe(6);
    for (let i = 0; i < 6; i += 1) expectCenter2(got[i]!, want[i]!);
  });

  it("defaults to an auto (cols-first) grid identical to rows=2,cols=3 for 6 mobs", () => {
    // Python: .arrange_in_grid(buff=0.1) on 6 squares -> same centers as above.
    const boxes = new VGroup(...squares(6));
    boxes.arrange_in_grid(null, null, 0.1);
    const got = centers2(boxes);
    const want = [
      [-2.1, 1.05, 0],
      [0, 1.05, 0],
      [2.1, 1.05, 0],
      [-2.1, -1.05, 0],
      [0, -1.05, 0],
      [2.1, -1.05, 0],
    ];
    for (let i = 0; i < 6; i += 1) expectCenter2(got[i]!, want[i]!);
  });

  it("honors tuple buff=(0.25, 0.5) on a 2x2 grid", () => {
    // Python: buff=(0.25,0.5) -> x=1.125, y=1.25 corners.
    const boxes = new VGroup(...squares(4));
    boxes.arrange_in_grid(2, 2, [0.25, 0.5]);
    const got = centers2(boxes);
    const want = [
      [-1.125, 1.25, 0],
      [1.125, 1.25, 0],
      [-1.125, -1.25, 0],
      [1.125, -1.25, 0],
    ];
    for (let i = 0; i < 4; i += 1) expectCenter2(got[i]!, want[i]!);
  });
});

// ---------------------------------------------------------------------------
// arrange_in_grid flow_order
// Python (2x2, buff=0.5, Square side 2 -> half-gap layout at +/-1.25):
//   rd: [(-1.25,1.25),(1.25,1.25),(-1.25,-1.25),(1.25,-1.25)]
//   dr: [(-1.25,1.25),(-1.25,-1.25),(1.25,1.25),(1.25,-1.25)]
//   lu: [(1.25,-1.25),(-1.25,-1.25),(1.25,1.25),(-1.25,1.25)]
//   ur: [(-1.25,-1.25),(-1.25,1.25),(1.25,-1.25),(1.25,1.25)]
// ---------------------------------------------------------------------------
describe("Mobject.arrange_in_grid flow_order", () => {
  const cases: Array<[string, number[][]]> = [
    ["rd", [[-1.25, 1.25], [1.25, 1.25], [-1.25, -1.25], [1.25, -1.25]]],
    ["dr", [[-1.25, 1.25], [-1.25, -1.25], [1.25, 1.25], [1.25, -1.25]]],
    ["lu", [[1.25, -1.25], [-1.25, -1.25], [1.25, 1.25], [-1.25, 1.25]]],
    ["ur", [[-1.25, -1.25], [-1.25, 1.25], [1.25, -1.25], [1.25, 1.25]]],
  ];
  for (const [flow, want] of cases) {
    it(`fills in ${flow} order`, () => {
      const boxes = new VGroup(...squares(4));
      boxes.arrange_in_grid(2, 2, 0.5, [0, 0, 0], null, null, null, null, flow);
      const got = centers2(boxes);
      for (let i = 0; i < 4; i += 1) expectCenter2(got[i]!, [...want[i]!, 0]);
    });
  }

  it("raises ValueError for an unknown flow_order", () => {
    // Python: ValueError('flow_order must be one of the following values: ...').
    const boxes = new VGroup(...squares(4));
    try {
      boxes.arrange_in_grid(2, 2, 0.5, [0, 0, 0], null, null, null, null, "xx");
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
      expect((e as Error).message).toContain("flow_order must be one of");
    }
  });
});

// ---------------------------------------------------------------------------
// arrange_in_grid errors + explicit sizes + alignments
// Python:
//   rows=1,cols=2 over 5 -> ValueError('Too few rows and columns to fit all submobjetcs.')
//   row_alignments='u' over 2 rows -> ValueError('row_alignments has a mismatching size.')
//   row_heights=[1.0] over 2 rows -> ValueError('row_heights has a mismatching size.')
//   row_heights=[3.0,None], col_widths=[None,4.0], buff=0.1 ->
//     centers (-1.55,1.3),(1.55,1.3),(-1.55,-1.3),(1.55,-1.3)
//   5 mobs in 2x3 buff=0.1 -> first five of the 2x3 layout (placeholder fills)
//   shifted group keeps its center after arrange
// ---------------------------------------------------------------------------
describe("Mobject.arrange_in_grid sizes, alignments and errors", () => {
  it("raises ValueError when rows*cols is too small", () => {
    const boxes = new VGroup(...squares(5));
    expect(() => boxes.arrange_in_grid(1, 2)).toThrowError(
      expect.objectContaining({ name: "ValueError" }),
    );
    try {
      boxes.arrange_in_grid(1, 2);
      expect.unreachable();
    } catch (e) {
      expect((e as Error).message).toBe("Too few rows and columns to fit all submobjetcs.");
    }
  });

  it("raises ValueError on mismatched row_alignments / row_heights", () => {
    expect(() =>
      new VGroup(...squares(4)).arrange_in_grid(2, 2, 0.5, [0, 0, 0], "u"),
    ).toThrowError(expect.objectContaining({ name: "ValueError" }));
    expect(() =>
      new VGroup(...squares(4)).arrange_in_grid(2, 2, 0.5, [0, 0, 0], "u"),
    ).toThrowError("row_alignments has a mismatching size.");
    expect(() =>
      new VGroup(...squares(4)).arrange_in_grid(2, 2, 0.5, [0, 0, 0], null, null, [1.0]),
    ).toThrowError("row_heights has a mismatching size.");
  });

  it("honors explicit row_heights and col_widths", () => {
    const boxes = new VGroup(...squares(4));
    boxes.arrange_in_grid(2, 2, 0.1, [0, 0, 0], null, null, [3.0, null], [null, 4.0], "rd");
    const got = centers2(boxes);
    const want = [
      [-1.55, 1.3, 0],
      [1.55, 1.3, 0],
      [-1.55, -1.3, 0],
      [1.55, -1.3, 0],
    ];
    for (let i = 0; i < 4; i += 1) expectCenter2(got[i]!, want[i]!);
  });

  it("infers grid size from alignments and resolves labeled alignments", () => {
    // Python: row_alignments='ud', col_alignments='lr', buff=0.1 over 4 squares
    // -> (+/-1.05, +/-1.05) corners in rd order.
    const boxes = new VGroup(...squares(4));
    boxes.arrange_in_grid(null, null, 0.1, [0, 0, 0], "ud", "lr", null, null, "rd");
    const got = centers2(boxes);
    const want = [
      [-1.05, 1.05, 0],
      [1.05, 1.05, 0],
      [-1.05, -1.05, 0],
      [1.05, -1.05, 0],
    ];
    for (let i = 0; i < 4; i += 1) expectCenter2(got[i]!, want[i]!);
  });

  it("fills a ragged 5-in-2x3 grid leaving the last cell empty", () => {
    // Python: 5 squares in 2x3 -> first five 2x3 positions.
    const boxes = new VGroup(...squares(5));
    boxes.arrange_in_grid(2, 3, 0.1);
    expect(boxes.submobjects.length).toBe(5);
    const got = centers2(boxes);
    const want = [
      [-2.1, 1.05, 0],
      [0, 1.05, 0],
      [2.1, 1.05, 0],
      [-2.1, -1.05, 0],
      [0, -1.05, 0],
    ];
    for (let i = 0; i < 5; i += 1) expectCenter2(got[i]!, want[i]!);
  });

  it("moves the arranged grid back to the pre-arrange center", () => {
    // Python: shift([5,5,0]) then arrange keeps center [5,5,0].
    const boxes = new VGroup(...squares(4));
    boxes.shift([5, 5, 0]);
    boxes.arrange_in_grid(2, 2, 0.5);
    expectCenter2(boxes.get_center(), [5, 5, 0]);
  });
});

// ---------------------------------------------------------------------------
// _AnimationBuilder.build
// Python:
//   type(Square().animate).__name__ == '_AnimationBuilder'
//   b.shift([1,0,0]) is b  (chaining returns the same builder)
//   build() -> _MethodAnimation, 1 method, target center [1,0,0], .mobject is sq
//   animate(run_time=2.5).shift([2,0,0]).build().run_time == 2.5, target [2,0,0]
//   builder(run_time=3) after a method -> ValueError(...before accessing methods...)
//   .no_such_method_xyz -> AttributeError
//   begin(); finish() moves the mobject onto the target ([1,2,0])
//   .shift([1,0,0]).scale(2) -> 2 methods, target width 4.0 (Square width 2)
//   empty .build() -> _MethodAnimation with 0 methods
// ---------------------------------------------------------------------------
describe("_AnimationBuilder.build", () => {
  // Faithful to Python: build() constructs manim.animation.transform's
  // _MethodAnimation (a MoveToTarget that blends mobject -> target
  // mid-interpolation). Assertions compare small scalars (never whole
  // animations) because serializing a Mobject graph on failure exhausts the
  // vitest worker heap.
  it("builds a _MethodAnimation bound to the mobject with recorded methods", () => {
    const sq = new Square();
    const b = sq.animate;
    expect((b as unknown as { constructor: { name: string } }).constructor.name).toBe(
      "_AnimationBuilder",
    );
    const chained = (b as unknown as Record<string, (v: number[]) => unknown>)["shift"]!([1, 0, 0]);
    expect(chained === (b as unknown)).toBe(true);
    const anim = (b as _AnimationBuilder).build() as TransformMethodAnimation;
    expect(anim instanceof Animation).toBe(true);
    expect(anim instanceof TransformMethodAnimation).toBe(true);
    expect(anim.constructor.name).toBe("_MethodAnimation");
    expect(anim.mobject === (sq as unknown)).toBe(true);
    expect(anim.target_mobject === (sq.target as unknown)).toBe(true);
    expect(anim.methods.length).toBe(1);
    expectCenter2((sq.target as Mobject).get_center(), [1, 0, 0]);
    // Original mobject untouched until begin/finish.
    expectCenter2(sq.get_center(), [0, 0, 0]);
  });

  it("copies anim_args (e.g. run_time) onto the built animation", () => {
    // Python: sq.animate(run_time=2.5).shift([2,0,0]).build().
    // TS spells __call__ as .call({...}).
    const sq = new Square();
    const b = sq.animate;
    b.call({ run_time: 2.5 });
    (b as unknown as Record<string, (v: number[]) => unknown>)["shift"]!([2, 0, 0]);
    const anim = b.build() as TransformMethodAnimation & { run_time?: number };
    expect(anim.run_time).toBe(2.5);
    expectCenter2((sq.target as Mobject).get_center(), [2, 0, 0]);
  });

  it("rejects passing animation args after accessing methods", () => {
    // Python: sq.animate.shift([1,0,0]) then builder(run_time=3) -> ValueError.
    const sq = new Square();
    const b = sq.animate;
    (b as unknown as Record<string, (v: number[]) => unknown>)["shift"]!([1, 0, 0]);
    expect(() => b.call({ run_time: 3 })).toThrowError(
      "Animation arguments must be passed before accessing methods and can only be passed once",
    );
    try {
      b.call({ run_time: 3 });
      expect.unreachable();
    } catch (e) {
      expect((e as Error).name).toBe("ValueError");
    }
  });

  it("raises AttributeError for unknown methods at access time", () => {
    const sq = new Square();
    expect(() => sq.animate.__getattr__("no_such_method_xyz")).toThrowError(
      expect.objectContaining({ name: "AttributeError" }),
    );
  });

  it("begin/interpolate/finish blends the mobject onto the recorded target", () => {
    // Python: animate.shift([1,2,0]); begin(); interpolate(0.5) -> center
    // [0.5,1.0,0] (smooth rate is 0.5 at alpha 0.5); finish() -> [1,2,0].
    // NOTE: a single builder (one `.animate` access) — each access regenerates
    // the target, mirroring Python's per-access _AnimationBuilder.
    const sq = new Square();
    const b = sq.animate as unknown as _AnimationBuilder &
      Record<string, (v: number[]) => unknown>;
    b["shift"]!([1, 2, 0]);
    const built = (b as _AnimationBuilder).build() as TransformMethodAnimation;
    built.begin();
    built.interpolate(0.5);
    expectCenter2(sq.get_center(), [0.5, 1.0, 0]);
    built.finish();
    expectCenter2(sq.get_center(), [1, 2, 0]);
    expectCenter2((sq.target as Mobject).get_center(), [1, 2, 0]);
  });

  it("chains multiple methods and builds an empty animation", () => {
    // Python: .shift([1,0,0]).scale(2) -> 2 methods, target width 4.0.
    const sq = new Square();
    const b = sq.animate as unknown as _AnimationBuilder &
      Record<string, (...a: unknown[]) => unknown>;
    b["shift"]!([1, 0, 0]);
    b["scale"]!(2);
    const anim = (b as _AnimationBuilder).build() as TransformMethodAnimation;
    expect(anim.methods.length).toBe(2);
    expectClose((sq.target as Mobject).width, 4.0);

    const sq2 = new Square();
    const empty = (sq2.animate as _AnimationBuilder).build() as TransformMethodAnimation;
    expect(empty.constructor.name).toBe("_MethodAnimation");
    expect(empty instanceof TransformMethodAnimation).toBe(true);
    expect(empty.methods.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Table (via MobjectTable + identity factory; no tex)
// Python (2x2 squares, h_buff=1.3, v_buff=0.8):
//   row_dim 2 col_dim 2, center [0,0,0], 2 rows / 2 cols
//   elements 4, without-labels 4
//   entry centers (-1.65,1.4),(1.65,1.4),(-1.65,-1.4),(1.65,-1.4)
//   get_entries((1,1)) is elements[0]; get_entries((2,2)) is elements[3]
//   1 horizontal + 1 vertical inner line
//   get_cell((1,1)) verts [[-3.3,2.8,0],[~0,2.8,0],[~0,~0,0],[-3.3,~0,0]]
// ---------------------------------------------------------------------------
describe("Table basic layout (MobjectTable)", () => {
  function plainTable(): MobjectTable {
    return new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
  }

  it("builds a centered 2x2 grid with the right entries and lines", () => {
    const t = plainTable();
    expect(t.row_dim).toBe(2);
    expect(t.col_dim).toBe(2);
    expectCenter2(t.get_center(), [0, 0, 0]);
    expect(t.get_rows().submobjects.length).toBe(2);
    expect(t.get_columns().submobjects.length).toBe(2);
    const entries = t.get_entries() as VGroup;
    expect(entries.submobjects.length).toBe(4);
    const bare = t.get_entries_without_labels() as VGroup;
    expect(bare.submobjects.length).toBe(4);
    const want = [
      [-1.65, 1.4, 0],
      [1.65, 1.4, 0],
      [-1.65, -1.4, 0],
      [1.65, -1.4, 0],
    ];
    for (let i = 0; i < 4; i += 1) {
      expectCenter2((entries.submobjects[i] as Mobject).get_center(), want[i]!);
    }
    expect(t.get_entries([1, 1])).toBe(entries.submobjects[0]);
    expect(t.get_entries([2, 2])).toBe(entries.submobjects[3]);
    expect((t.get_entries_without_labels([2, 2]) as Mobject)).toBe(bare.submobjects[3]);
    expect(t.get_horizontal_lines().submobjects.length).toBe(1);
    expect(t.get_vertical_lines().submobjects.length).toBe(1);
  });

  it("computes cell polygons around entries", () => {
    const t = plainTable();
    const verts = pointRows(t.get_cell([1, 1]).get_vertices(), 3);
    const want = [
      [-3.3, 2.8, 0],
      [0, 2.8, 0],
      [0, 0, 0],
      [-3.3, 0, 0],
    ];
    expect(verts.length).toBe(4);
    for (let i = 0; i < 4; i += 1) {
      expectClose(verts[i]![0] ?? NaN, want[i]![0] ?? NaN);
      expectClose(verts[i]![1] ?? NaN, want[i]![1] ?? NaN);
      expectClose(verts[i]![2] ?? NaN, want[i]![2] ?? NaN);
    }
  });

  it("raises ValueError for ragged rows", () => {
    // Python: ValueError('Not all rows in table have the same length.').
    expect(
      () =>
        new MobjectTable([[new Square()], [new Square(), new Square()]], {
          element_to_mobject: idFactory,
        }),
    ).toThrowError("Not all rows in table have the same length.");
  });

  it("scales h_buff/v_buff alongside the geometry", () => {
    // Python: buffs 1.3/0.8 -> scale(2) -> 2.6/1.6.
    const t = plainTable();
    expect(t.h_buff).toBe(1.3);
    expect(t.v_buff).toBe(0.8);
    t.scale(2);
    expectClose(t.h_buff, 2.6);
    expectClose(t.v_buff, 1.6);
  });
});

// ---------------------------------------------------------------------------
// Table labels / lines / colors / highlight / create
// Python:
//   row(2)+col(2) labels, no top-left -> mob_table 3x3, elements 8,
//     row_labels 2, col_labels 2, labels 4
//   entry (2,2) center [0,0]; without-labels (1,1) center [0,0]
//   hlines 2, vlines 2; hline starts x=-4.95, y in {1.4,-1.4}
//   get_cell((2,2)) verts [[-1.65,1.4],[1.65,1.4],[1.65,-1.4],[-1.65,-1.4]]
//   top_left_entry given -> labels 5, elements 9, entries((1,1)) is tle
//   include_outer_lines=True -> h 3 v 3; include_inner_lines=False -> 0/0;
//     outer-only -> 2/2
//   add_background_rectangles_to_entries=True -> every bare entry has one;
//     create() -> 6 anims; with labels (no bg) -> 3 anims; plain -> 2 anims
//   set_column_colors(RED, BLUE) -> fill #FC6255 / #58C4DD
//   empty table -> no row/col labels, no labels
// ---------------------------------------------------------------------------
describe("Table labels, lines, colors and create", () => {
  function labeledTable(): MobjectTable {
    return new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      {
        element_to_mobject: idFactory,
        row_labels: [new Square(), new Square()],
        col_labels: [new Square(), new Square()],
      },
    );
  }

  it("adds row+column labels with a dummy placeholder corner", () => {
    const t = labeledTable();
    expect(t.mob_table.length).toBe(3);
    expect(t.mob_table[0]?.length).toBe(3);
    expect((t.get_entries() as VGroup).submobjects.length).toBe(8);
    expect(t.get_row_labels().submobjects.length).toBe(2);
    expect(t.get_col_labels().submobjects.length).toBe(2);
    expect(t.get_labels().submobjects.length).toBe(4);
    expectCenter2((t.get_entries([2, 2]) as Mobject).get_center(), [0, 0, 0]);
    expectCenter2((t.get_entries_without_labels([1, 1]) as Mobject).get_center(), [0, 0, 0]);
    expect(t.get_horizontal_lines().submobjects.length).toBe(2);
    expect(t.get_vertical_lines().submobjects.length).toBe(2);
    const starts = (t.get_horizontal_lines().submobjects as Mobject[]).map((l) =>
      pointRows((l as unknown as { get_start(): unknown }).get_start(), 3)[0] ?? [0, 0, 0],
    );
    expectClose(starts[0]![0] ?? NaN, -4.95);
    expectClose(starts[0]![1] ?? NaN, 1.4);
    expectClose(starts[1]![1] ?? NaN, -1.4);
  });

  it("resolves labeled cell polygons", () => {
    const t = labeledTable();
    const verts = pointRows(t.get_cell([2, 2]).get_vertices(), 3);
    const want = [
      [-1.65, 1.4, 0],
      [1.65, 1.4, 0],
      [1.65, -1.4, 0],
      [-1.65, -1.4, 0],
    ];
    for (let i = 0; i < 4; i += 1) {
      expectClose(verts[i]![0] ?? NaN, want[i]![0] ?? NaN);
      expectClose(verts[i]![1] ?? NaN, want[i]![1] ?? NaN);
      expectClose(verts[i]![2] ?? NaN, want[i]![2] ?? NaN);
    }
  });

  it("keeps an explicit top-left entry as entries((1,1))", () => {
    const tle = new Square().scale(0.5);
    const t = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      {
        element_to_mobject: idFactory,
        row_labels: [new Square(), new Square()],
        col_labels: [new Square(), new Square()],
        top_left_entry: tle,
      },
    );
    expect(t.get_labels().submobjects.length).toBe(5);
    expect((t.get_entries() as VGroup).submobjects.length).toBe(9);
    expect(t.get_entries([1, 1])).toBe(tle);
  });

  it("counts outer/inner lines like Python", () => {
    const mk = (opts: Record<string, unknown>) =>
      new MobjectTable(
        [
          [new Square(), new Square()],
          [new Square(), new Square()],
        ],
        { element_to_mobject: idFactory, ...opts },
      );
    const outer = mk({ include_outer_lines: true });
    expect(outer.get_horizontal_lines().submobjects.length).toBe(3);
    expect(outer.get_vertical_lines().submobjects.length).toBe(3);
    const none = mk({ include_inner_lines: false });
    expect(none.get_horizontal_lines().submobjects.length).toBe(0);
    expect(none.get_vertical_lines().submobjects.length).toBe(0);
    const outerOnly = mk({ include_inner_lines: false, include_outer_lines: true });
    expect(outerOnly.get_horizontal_lines().submobjects.length).toBe(2);
    expect(outerOnly.get_vertical_lines().submobjects.length).toBe(2);
  });

  it("paints per-column and per-row colors", () => {
    // Python fills: RED -> #FC6255, BLUE -> #58C4DD.
    const t = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
    t.set_column_colors("RED", "BLUE");
    const cols = t.get_columns();
    const fill0 = ((cols.submobjects[0] as VGroup).submobjects[0] as VMobject)
      .get_fill_color()
      .to_hex();
    const fill1 = ((cols.submobjects[1] as VGroup).submobjects[0] as VMobject)
      .get_fill_color()
      .to_hex();
    expect(fill0.toUpperCase()).toBe("#FC6255");
    expect(fill1.toUpperCase()).toBe("#58C4DD");
    t.set_row_colors("GREEN", "YELLOW");
    const rows = t.get_rows();
    expect(rows.submobjects.length).toBe(2);
  });

  it("highlights a cell with a background rectangle", () => {
    const t = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
    t.add_highlighted_cell([2, 2], "GREEN");
    expect((t.get_entries([2, 2]) as VMobject).background_rectangle).not.toBeNull();
  });

  it("builds create() groups with lines + elements (+ labels + entry bgs)", () => {
    // Python: plain -> 2 anims; labeled -> 3; with entry bgs -> 6.
    const plain = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
    const agPlain = plain.create();
    // NOTE: compared by constructor name (not toBeInstanceOf printing the
    // whole table graph on failure — that exhausts the worker heap).
    expect(agPlain instanceof AnimationGroup).toBe(true);
    expect(agPlain.animations.length).toBe(2);

    const agLabeled = labeledTable().create();
    expect(agLabeled.animations.length).toBe(3);
    expect(agLabeled.lag_ratio).toBe(1);

    const withBg = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory, add_background_rectangles_to_entries: true },
    );
    for (const e of (withBg.get_entries_without_labels() as VGroup).submobjects) {
      expect((e as VMobject).background_rectangle).not.toBeNull();
    }
    expect(withBg.create().animations.length).toBe(6);
  });

  it("returns empty label groups when no labels are present", () => {
    const t = new MobjectTable([[new Square()]], { element_to_mobject: idFactory });
    expect(t.get_row_labels().submobjects.length).toBe(0);
    expect(t.get_col_labels().submobjects.length).toBe(0);
    expect(t.get_labels().submobjects.length).toBe(0);
  });

  it("default factories build Paragraph/MathTex/Integer/DecimalNumber entries", () => {
    // Text/tex rendering now works through the MathJax-backed pipeline, so
    // the defaults build real entries (one per data cell).
    expect(new Table([[new Square()]]).elements.submobjects.length).toBe(1);
    expect(new MathTable([["a"]]).elements.submobjects.length).toBe(1);
    expect(new IntegerTable([["1"]]).elements.submobjects.length).toBe(1);
    expect(new DecimalTable([["1.5"]]).elements.submobjects.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Integration: batch symbols against earlier-ported machinery
// ---------------------------------------------------------------------------
describe("integration: grid + builder + table", () => {
  it("animates a table via the _AnimationBuilder update path", () => {
    // Python: t.animate.shift([1,0,0]); begin(); finish() moves center to [1,0,0].
    const t = new MobjectTable(
      [
        [new Square(), new Square()],
        [new Square(), new Square()],
      ],
      { element_to_mobject: idFactory },
    );
    const proxy = t.animate as unknown as _AnimationBuilder &
      Record<string, (...a: unknown[]) => unknown>;
    proxy["shift"]!([1, 0, 0]);
    const anim = (proxy as _AnimationBuilder).build() as TransformMethodAnimation;
    expect(anim.constructor.name).toBe("_MethodAnimation");
    anim.begin();
    anim.finish();
    expectCenter2(t.get_center(), [1, 0, 0]);
  });

  it("arranges tables in a grid like Group(Table, Table)", () => {
    // Python: 2 fresh 2x2 MobjectTables in VGroup.arrange_in_grid(1, 2, 1):
    // each table is 6.6 wide (entries span 5.3 + inner lines overhang
    // h_buff/2 on each side) x 5.6 high; grid centers land at x=-3.8/+3.8.
    const mk = () =>
      new MobjectTable(
        [
          [new Square(), new Square()],
          [new Square(), new Square()],
        ],
        { element_to_mobject: idFactory },
      );
    const g = new VGroup(mk(), mk());
    g.arrange_in_grid(1, 2, 1);
    const got = centers2(g);
    expectClose(got[0]![0] ?? NaN, -3.8);
    expectClose(got[1]![0] ?? NaN, 3.8);
    expectClose(got[0]![1] ?? NaN, 0);
    expectClose(got[1]![1] ?? NaN, 0);
  });
});
