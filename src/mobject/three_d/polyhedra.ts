export class Polyhedron {
  faces_list: number[][];

  constructor(faces_list: number[][] = []) {
    this.faces_list = faces_list;
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
}
