import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject/mobject.js";
import { VGroup, VMobject } from "../mobject/types/vectorized_mobject.js";
import type { ParsableManimColor } from "../utils/color/core.js";
import { BLUE_D, BLUE_E, ManimColor, WHITE } from "../utils/color/core.js";
import { pointRows } from "../utils/arrays.js";
import { smooth, type RateFunction } from "../utils/rate_functions.js";

export interface AnimatedBoundaryOptions {
  colors?: ParsableManimColor[];
  max_stroke_width?: number;
  cycle_rate?: number;
  back_and_forth?: boolean;
  draw_rate_func?: RateFunction;
  fade_rate_func?: RateFunction;
  [key: string]: unknown;
}

export class AnimatedBoundary extends VGroup {
  colors: ParsableManimColor[];
  max_stroke_width: number;
  cycle_rate: number;
  back_and_forth: boolean;
  draw_rate_func: RateFunction;
  fade_rate_func: RateFunction;
  vmobject: VMobject;
  boundary_copies: VMobject[];
  total_time = 0.0;

  constructor(
    vmobject: VMobject,
    colors: ParsableManimColor[] = [BLUE_D, new ManimColor("BLUE_B"), BLUE_E, new ManimColor("GREY_BROWN")],
    max_stroke_width = 3,
    cycle_rate = 0.5,
    back_and_forth = true,
    draw_rate_func: RateFunction = smooth,
    fade_rate_func: RateFunction = smooth,
    options: AnimatedBoundaryOptions = {},
  ) {
    // NOTE: Python passes **kwargs to VGroup.__init__; the TS VGroup takes
    // only submobjects, so the options bag is accepted for signature
    // compatibility but has no Mobject-level kwargs to forward to.
    void options;
    super();
    this.colors = colors;
    this.max_stroke_width = max_stroke_width;
    this.cycle_rate = cycle_rate;
    this.back_and_forth = back_and_forth;
    this.draw_rate_func = draw_rate_func;
    this.fade_rate_func = fade_rate_func;
    this.vmobject = vmobject;
    this.boundary_copies = [0, 1].map(() =>
      (vmobject.copy() as VMobject).set_style({ stroke_width: 0, fill_opacity: 0 }),
    );
    this.add(...this.boundary_copies);
    this.total_time = 0.0;
    this.add_updater((m, dt) => (m as AnimatedBoundary).update_boundary_copies(dt));
  }

  update_boundary_copies(dt: number): void {
    // Not actual time, but something which passes at
    // an altered rate to make the implementation below
    // cleaner
    const time = this.total_time * this.cycle_rate;
    const [growing, fading] = this.boundary_copies as [VMobject, VMobject];
    const colors = this.colors;
    const msw = this.max_stroke_width;
    const vmobject = this.vmobject;

    const index = Math.trunc(time % colors.length);
    const alpha = time % 1;
    const draw_alpha = this.draw_rate_func(alpha);
    const fade_alpha = this.fade_rate_func(alpha);

    let bounds: [number, number];
    if (this.back_and_forth && Math.trunc(time) % 2 === 1) {
      bounds = [1.0 - draw_alpha, 1.0];
    } else {
      bounds = [0.0, draw_alpha];
    }
    this.full_family_become_partial(growing, vmobject, ...bounds);
    growing.set_stroke(colors[index], msw);

    if (time >= 1) {
      this.full_family_become_partial(fading, vmobject, 0, 1);
      fading.set_stroke(colors[index - 1], (1 - fade_alpha) * msw);
    }

    this.total_time += dt;
  }

  full_family_become_partial(mob1: VMobject, mob2: VMobject, a: number, b: number): this {
    const family1 = mob1.family_members_with_points();
    const family2 = mob2.family_members_with_points();
    const length = Math.min(family1.length, family2.length);
    for (let index = 0; index < length; index += 1) {
      (family1[index] as VMobject).pointwise_become_partial(family2[index] as VMobject, a, b);
    }
    return this;
  }
}

export interface TracedPathOptions {
  stroke_width?: number;
  stroke_color?: ParsableManimColor | null;
  dissipating_time?: number | null;
  [key: string]: unknown;
}

export class TracedPath extends VMobject {
  traced_point_func: () => number[] | NDArray;
  dissipating_time: number | null;
  time: number | null;

  constructor(
    traced_point_func: () => number[] | NDArray,
    stroke_width = 2,
    stroke_color: ParsableManimColor | null = WHITE,
    dissipating_time: number | null = null,
    options: TracedPathOptions = {},
  ) {
    super({ stroke_color: stroke_color ?? undefined, stroke_width, ...(options as Record<string, unknown>) });
    this.traced_point_func = traced_point_func;
    this.dissipating_time = dissipating_time;
    this.time = this.dissipating_time ? 1.0 : null;
    this.add_updater((mob, dt) => (this as TracedPath).update_path(mob as Mobject, dt));
  }

  update_path(_mob: Mobject, dt: number): void {
    const new_point = this.traced_point_func();
    if (!this.has_points()) {
      this.start_new_path(new_point);
    }
    this.add_line_to(new_point);
    if (this.dissipating_time) {
      if (this.time === null) {
        throw new Error("TracedPath time is None while dissipating_time is set");
      }
      this.time += dt;
      if (this.time - 1 > this.dissipating_time) {
        const nppcc = this.n_points_per_curve;
        this.set_points(array(pointRows(this.points, this.dim).slice(nppcc)) as NDArray);
      }
    }
  }
}
