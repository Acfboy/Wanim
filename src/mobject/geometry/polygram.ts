import { VMobject, type VMobjectOptions } from "../types/vectorized_mobject.js";
import { isNDArray, pointRows } from "../../utils/arrays.js";
import { BLUE, type ParsableManimColor } from "../../utils/color/core.js";

export interface PolygramOptions extends VMobjectOptions {
  color?: ParsableManimColor;
}

export class Polygram extends VMobject {
  constructor(...args: unknown[]) {
    let vertex_groups: unknown[] = args;
    let options: PolygramOptions = {};
    const last = args[args.length - 1];
    if (last !== undefined && typeof last === "object" && !Array.isArray(last) && !isNDArray(last)) {
      options = last as PolygramOptions;
      vertex_groups = args.slice(0, -1);
    }
    super({ color: BLUE, ...options });
    for (const vertices of vertex_groups) {
      const rows = pointRows(vertices, this.dim);
      if (rows.length === 0) continue;
      const [first_vertex, ...rest] = rows;
      this.start_new_path(first_vertex);
      this.add_points_as_corners([...rest, first_vertex ?? [0, 0, 0]]);
    }
  }

  get_vertices(): number[][] {
    return this.get_start_anchors();
  }
}
