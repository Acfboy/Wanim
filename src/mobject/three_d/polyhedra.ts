import { array, type NDArray } from "numpy-ts";
import { Mobject } from "../mobject.js";
import { VGroup, VMobject } from "../types/vectorized_mobject.js";
import { pointRows } from "../../utils/arrays.js";
import { Polygon, type PolygramOptions } from "../geometry/polygram.js";
import { Graph } from "../graph.js";
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
