import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { pointRows, isNDArray } from "../../utils/arrays.js";
import { Polygon, type PolygramOptions } from "../geometry/polygram.js";
import { Graph } from "../graph.js";
import { QuickHull, QuickHullPoint, type HullPoints } from "../../utils/qhull.js";
import { Dot3D } from "./three_dimensions.js";

export class Polyhedron extends VGroup {
  faces_config: PolygramOptions;
  graph_config: Record<string, unknown>;
  vertex_coords: number[][];
  vertex_indices: number[];
  layout: Map<number, number[]>;
  faces_list: number[][];
  face_coords: number[][][];
  edges: [number, number][];
  faces: VGroup;
  graph: Graph;

  constructor(
    vertex_coords: Array<readonly number[] | NDArray> = [],
    faces_list: number[][] = [],
    faces_config: PolygramOptions = {},
    graph_config: Record<string, unknown> = {},
  ) {
    super();
    this.faces_config = { fill_opacity: 0.5, shade_in_3d: true, ...faces_config };
    this.graph_config = {
      vertex_type: Dot3D,
      edge_config: { stroke_opacity: 0 },
      ...graph_config,
    };
    this.vertex_coords = vertex_coords.map(
      (vertex) => [...(pointRows(vertex, 3)[0] ?? [0, 0, 0])],
    );
    this.vertex_indices = this.vertex_coords.map((_, index) => index);
    this.layout = new Map(this.vertex_coords.map((vertex, index) => [index, vertex] as [number, number[]]));
    this.faces_list = faces_list.map((face) => [...face]);
    this.face_coords = faces_list.map((face) =>
      face.map((vertex_index) => [...(this.layout.get(vertex_index) ?? [0, 0, 0])]),
    );
    this.edges = this.get_edges(this.faces_list);
    this.faces = this.create_faces(this.face_coords);
    this.graph = new Graph(this.vertex_indices, this.edges, {});
    // NOTE: the TS Graph does not support layout=/vertex_type yet, so its
    // vertices are empty placeholder VMobjects sitting at the origin; seed
    // one point per vertex so the layout move below lands on the intended
    // centers (mirroring what Python's Graph(layout=...) with Dot3D
    // vertices does). Without points move_to() would silently no-op and
    // extract_face_coords()/update_faces() would read all-zero centers.
    for (const [index, position] of this.layout) {
      const vertex = this.graph.vertices.get(index);
      if (vertex instanceof VMobject && vertex.get_num_points() === 0) {
        vertex.set_points(array([[...position]]) as NDArray);
      }
      vertex?.move_to(position);
    }
    this.add(this.faces, this.graph);
    this.add_updater((mob: Mobject) => {
      this.update_faces(mob);
    });
  }

  get_edges(faces_list: number[][]): [number, number][] {
    const edges: [number, number][] = [];
    for (const face of faces_list) {
      const rotated = [...face.slice(1), ...(face.length > 0 ? [face[0] as number] : [])];
      for (let index = 0; index < face.length; index += 1) {
        edges.push([face[index] as number, rotated[index] as number]);
      }
    }
    return edges;
  }

  create_faces(face_coords: number[][][]): VGroup {
    const face_group = new VGroup();
    for (const face of face_coords) {
      face_group.add(new Polygon(face, this.faces_config));
    }
    return face_group;
  }

  update_faces(m: Mobject): this {
    void m;
    const face_coords = this.extract_face_coords();
    const new_faces = this.create_faces(face_coords);
    this.faces.match_points(new_faces);
    return this;
  }

  extract_face_coords(): number[][][] {
    const new_vertex_coords = this.vertex_indices.map((vertex) => [
      ...(this.graph.getItem(vertex) as Mobject).get_center(),
    ]);
    const layout = new Map(new_vertex_coords.map((coords, index) => [index, coords] as [number, number[]]));
    return this.faces_list.map((face) =>
      face.map((vertex_index) => [...(layout.get(vertex_index) ?? [0, 0, 0])]),
    );
  }
}

export interface PlatonicSolidOptions {
  faces_config?: PolygramOptions;
  graph_config?: Record<string, unknown>;
}

export class Tetrahedron extends Polyhedron {
  constructor(
    edge_length = 1,
    faces_config: PolygramOptions = {},
    graph_config: Record<string, unknown> = {},
  ) {
    const unit = (edge_length * Math.sqrt(2)) / 4;
    super(
      [
        [unit, unit, unit],
        [unit, -unit, -unit],
        [-unit, unit, -unit],
        [-unit, -unit, unit],
      ],
      [
        [0, 1, 2],
        [3, 0, 2],
        [0, 1, 3],
        [3, 1, 2],
      ],
      faces_config,
      graph_config,
    );
  }
}

export class Octahedron extends Polyhedron {
  constructor(
    edge_length = 1,
    faces_config: PolygramOptions = {},
    graph_config: Record<string, unknown> = {},
  ) {
    const unit = (edge_length * Math.sqrt(2)) / 2;
    super(
      [
        [unit, 0, 0],
        [-unit, 0, 0],
        [0, unit, 0],
        [0, -unit, 0],
        [0, 0, unit],
        [0, 0, -unit],
      ],
      [
        [2, 4, 1],
        [0, 4, 2],
        [4, 3, 0],
        [1, 3, 4],
        [3, 5, 0],
        [1, 5, 3],
        [2, 5, 1],
        [0, 5, 2],
      ],
      faces_config,
      graph_config,
    );
  }
}

export class Icosahedron extends Polyhedron {
  constructor(
    edge_length = 1,
    faces_config: PolygramOptions = {},
    graph_config: Record<string, unknown> = {},
  ) {
    const unit_a = (edge_length * (1 + Math.sqrt(5))) / 4;
    const unit_b = edge_length * (1 / 2);
    super(
      [
        [0, unit_b, unit_a],
        [0, -unit_b, unit_a],
        [0, unit_b, -unit_a],
        [0, -unit_b, -unit_a],
        [unit_b, unit_a, 0],
        [unit_b, -unit_a, 0],
        [-unit_b, unit_a, 0],
        [-unit_b, -unit_a, 0],
        [unit_a, 0, unit_b],
        [unit_a, 0, -unit_b],
        [-unit_a, 0, unit_b],
        [-unit_a, 0, -unit_b],
      ],
      [
        [1, 8, 0],
        [1, 5, 7],
        [8, 5, 1],
        [7, 3, 5],
        [5, 9, 3],
        [8, 9, 5],
        [3, 2, 9],
        [9, 4, 2],
        [8, 4, 9],
        [0, 4, 8],
        [6, 4, 0],
        [6, 2, 4],
        [11, 2, 6],
        [3, 11, 2],
        [0, 6, 10],
        [10, 1, 0],
        [10, 7, 1],
        [11, 7, 3],
        [10, 11, 7],
        [10, 11, 6],
      ],
      faces_config,
      graph_config,
    );
  }
}

export class Dodecahedron extends Polyhedron {
  constructor(
    edge_length = 1,
    faces_config: PolygramOptions = {},
    graph_config: Record<string, unknown> = {},
  ) {
    const unit_a = (edge_length * (1 + Math.sqrt(5))) / 4;
    const unit_b = (edge_length * (3 + Math.sqrt(5))) / 4;
    const unit_c = edge_length * (1 / 2);
    super(
      [
        [unit_a, unit_a, unit_a],
        [unit_a, unit_a, -unit_a],
        [unit_a, -unit_a, unit_a],
        [unit_a, -unit_a, -unit_a],
        [-unit_a, unit_a, unit_a],
        [-unit_a, unit_a, -unit_a],
        [-unit_a, -unit_a, unit_a],
        [-unit_a, -unit_a, -unit_a],
        [0, unit_c, unit_b],
        [0, unit_c, -unit_b],
        [0, -unit_c, -unit_b],
        [0, -unit_c, unit_b],
        [unit_c, unit_b, 0],
        [-unit_c, unit_b, 0],
        [unit_c, -unit_b, 0],
        [-unit_c, -unit_b, 0],
        [unit_b, 0, unit_c],
        [-unit_b, 0, unit_c],
        [unit_b, 0, -unit_c],
        [-unit_b, 0, -unit_c],
      ],
      [
        [18, 16, 0, 12, 1],
        [3, 18, 16, 2, 14],
        [3, 10, 9, 1, 18],
        [1, 9, 5, 13, 12],
        [0, 8, 4, 13, 12],
        [2, 16, 0, 8, 11],
        [4, 17, 6, 11, 8],
        [17, 19, 5, 13, 4],
        [19, 7, 15, 6, 17],
        [6, 15, 14, 2, 11],
        [19, 5, 9, 10, 7],
        [7, 10, 3, 14, 15],
      ],
      faces_config,
      graph_config,
    );
  }
}

export interface ConvexHull3DOptions {
  tolerance?: number;
  faces_config?: PolygramOptions;
  graph_config?: Record<string, unknown>;
}

export class ConvexHull3D extends Polyhedron {
  constructor(...args: unknown[]) {
    // Mirrors Python `ConvexHull3D(*points, tolerance=1e-5, **kwargs)`: a
    // trailing plain-object argument is the options bag (tolerance /
    // faces_config / graph_config); everything before it is a point.
    let options: ConvexHull3DOptions = {};
    let points: unknown[] = args;
    const last = args[args.length - 1];
    if (
      last !== null &&
      typeof last === "object" &&
      !Array.isArray(last) &&
      !isNDArray(last)
    ) {
      options = last as ConvexHull3DOptions;
      points = args.slice(0, -1);
    }
    const { tolerance = 1e-5, faces_config = {}, graph_config = {} } = options;
    // Build Convex Hull (mirrors Python: QuickHull(tolerance).build(array)).
    const hull = new QuickHull(tolerance);
    hull.build(points as unknown as HullPoints);

    // Extract faces: for each surviving facet, collect the de-duplicated
    // vertex indices of its subfacets' points.
    // NOTE: Python keys vertices by QuickHullPoint identity; the TS qhull
    // port stores only coordinate rows on SubFacet, so identity is the
    // coordinates' byte key (same equality semantics).
    const vertices: number[][] = [];
    const faces: number[][] = [];
    let c = 0;
    const d = new Map<string, number>();
    for (const facet of hull.facets) {
      if (hull.removed.has(facet.key())) continue;
      const tmp = new Set<string>();
      for (const subfacet of facet.subfacets) {
        for (const coordinates of subfacet.coordinates) {
          const key = new QuickHullPoint(coordinates).key();
          if (!d.has(key)) {
            vertices.push([...coordinates]);
            d.set(key, c);
            c += 1;
          }
          tmp.add(key);
        }
      }
      faces.push([...tmp].map((key) => d.get(key) as number));
    }
    super(vertices, faces, faces_config, graph_config);
  }
}
