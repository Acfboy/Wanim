import { Mobject, Group } from "./mobject.js";
import { VMobject, VGroup } from "./types/vectorized_mobject.js";
import { array } from "numpy-ts";
import type { NDArray } from "numpy-ts";
import { addPoints, norm, pointRows, pointsFrom } from "../utils/arrays.js";

export type GraphVertex = string | number | boolean | object;
export type GraphEdge = readonly [GraphVertex, GraphVertex];

export interface NetworkXGraphLike {
  nodes: Map<GraphVertex, Record<string, unknown>>;
  edges: Set<string>;
  addNode(node: GraphVertex): void;
  addNodesFrom(nodes: Iterable<GraphVertex>): void;
  addEdge(u: GraphVertex, v: GraphVertex): void;
  addEdgesFrom(edges: Iterable<GraphEdge>): void;
  neighbors(node: GraphVertex): GraphVertex[];
  hasNode(node: GraphVertex): boolean;
}

function edgeKey(u: GraphVertex, v: GraphVertex): string {
  return `${String(u)}\u0000${String(v)}`;
}

export class NetworkXGraph implements NetworkXGraphLike {
  nodes = new Map<GraphVertex, Record<string, unknown>>();
  edges = new Set<string>();
  edge_pairs: [GraphVertex, GraphVertex][] = [];
  directed = false;

  addNode(node: GraphVertex): void { if (!this.nodes.has(node)) this.nodes.set(node, {}); }
  addNodesFrom(nodes: Iterable<GraphVertex>): void { for (const node of nodes) this.addNode(node); }
  addEdge(u: GraphVertex, v: GraphVertex): void {
    this.addNode(u);
    this.addNode(v);
    const key = edgeKey(u, v);
    if (!this.edges.has(key)) this.edge_pairs.push([u, v]);
    this.edges.add(key);
    if (this.directed) this.edges.add(edgeKey(v, u));
  }
  addEdgesFrom(edges: Iterable<GraphEdge>): void { for (const [u, v] of edges) this.addEdge(u, v); }
  neighbors(node: GraphVertex): GraphVertex[] {
    const result: GraphVertex[] = [];
    for (const [u, v] of this.edge_pairs) {
      if (u === node) result.push(v);
      if (!this.directed && v === node) result.push(u);
    }
    return [...new Set(result)];
  }
  hasNode(node: GraphVertex): boolean { return this.nodes.has(node); }
}

export interface LayoutFunction {
  (graph: NetworkXGraphLike, scale?: number | number[], ...args: any[]): Map<GraphVertex, NDArray> | Record<string, NDArray>;
}

export const LayoutFunction = class LayoutFunction {
  __call__(graph: NetworkXGraphLike, _scale: number | number[] = 2, ..._args: any[]): Map<GraphVertex, NDArray> {
    const result = new Map<GraphVertex, NDArray>();
    for (const node of graph.nodes.keys()) result.set(node, array([0, 0, 0]) as NDArray);
    return result;
  }
};

export class GenericGraph extends VMobject {
  vertices: Map<GraphVertex, Mobject>;
  edges: Map<string, Mobject>;
  _graph: NetworkXGraphLike;
  _layout: Map<GraphVertex, NDArray>;

  constructor(vertices: GraphVertex[] = [], edges: GraphEdge[] = [], options: Record<string, unknown> = {}) {
    super(options);
    this.vertices = new Map();
    this.edges = new Map();
    this._layout = new Map();
    this._graph = (this.constructor as typeof GenericGraph)._empty_networkx_graph();
    this._graph.addNodesFrom(vertices);
    this._graph.addEdgesFrom(edges);
    for (const vertex of vertices) {
      const mobject = new VMobject({ name: String(vertex) });
      this.vertices.set(vertex, mobject);
      this.add(mobject);
    }
  }

  static _empty_networkx_graph(): NetworkXGraphLike {
    const error = new Error("To be implemented in concrete subclasses");
    error.name = "NotImplementedError";
    throw error;
  }

  static empty_networkx_graph(): NetworkXGraphLike {
    return GenericGraph._empty_networkx_graph();
  }

  get_empty_networkx_graph(): NetworkXGraphLike {
    return (this.constructor as typeof GenericGraph)._empty_networkx_graph();
  }

  getItem(key: GraphVertex | GraphEdge): Mobject {
    if (this.vertices.has(key as GraphVertex)) return this.vertices.get(key as GraphVertex) as Mobject;
    const edge = this.edges.get(edgeKey(...(key as GraphEdge)));
    if (edge) return edge;
    throw new Error(`Could not find ${String(key)} in vertices or edges`);
  }
}

export class Graph extends GenericGraph {
  static _empty_networkx_graph(): NetworkXGraphLike {
    return new NetworkXGraph();
  }
}

export class DiGraph extends GenericGraph {
  static _empty_networkx_graph(): NetworkXGraphLike {
    const graph = new NetworkXGraph();
    graph.directed = true;
    return graph;
  }
}

export function tree_layout(
  graph: NetworkXGraphLike,
  rootVertex: GraphVertex | null = null,
  scale: number | number[] | null = 2,
  vertexSpacing: number[] | null = null,
  orientation = "down",
): Map<GraphVertex, NDArray> {
  if (rootVertex === null || !graph.hasNode(rootVertex)) throw new Error("The tree layout requires the root_vertex parameter");
  const allNodes = [...graph.nodes.keys()];
  const reached = new Set<GraphVertex>([rootVertex]);
  const pending: GraphVertex[] = [rootVertex];
  while (pending.length > 0) {
    const node = pending.pop() as GraphVertex;
    for (const neighbor of graph.neighbors(node)) if (!reached.has(neighbor)) { reached.add(neighbor); pending.push(neighbor); }
  }
  const degreeSum: number = [...reached].reduce<number>((count, node) => count + Number(graph.neighbors(node).length), 0);
  const edgeCount = (graph as NetworkXGraph).directed ? (graph as NetworkXGraph).edge_pairs.length : degreeSum / 2;
  if (reached.size !== allNodes.length || allNodes.length !== edgeCount + 1) {
    const error = new Error("The tree layout must be used with trees");
    error.name = "ValueError";
    throw error;
  }
  const children = new Map<GraphVertex, GraphVertex[]>();
  children.set(rootVertex, graph.neighbors(rootVertex));
  for (const node of graph.nodes.keys()) if (!children.has(node)) children.set(node, []);
  const pos = new Map<GraphVertex, [number, number]>();
  const obstruction: number[] = Array.from({ length: graph.nodes.size }, () => 0);
  const obstructionAt = (y: number): number => obstruction[y < 0 ? obstruction.length + y : y] ?? 0;
  const setObstruction = (y: number, value: number): void => { obstruction[y < 0 ? obstruction.length + y : y] = value; };
  const stack: GraphVertex[][] = [[...children.get(rootVertex) as GraphVertex[]]];
  const stick: GraphVertex[] = [rootVertex];
  const parent = new Map<GraphVertex, GraphVertex>();
  for (const child of children.get(rootVertex) as GraphVertex[]) parent.set(child, rootVertex);
  const slide = (v: GraphVertex, dx: number): void => {
    let level: GraphVertex[] = [v];
    while (level.length > 0) {
      const nextLevel: GraphVertex[] = [];
      for (const u of level) {
        const current = pos.get(u) ?? [0, 0];
        const x = current[0] + dx;
        const y = current[1];
        setObstruction(y, Math.max(x + 1, obstructionAt(y)));
        pos.set(u, [x, y]);
        nextLevel.push(...(children.get(u) ?? []));
      }
      level = nextLevel;
    }
  };
  const orientationSign = orientation === "down" ? -1 : 1;
  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    if (current.length === 0) {
      const p = stick.pop() as GraphVertex;
      stack.pop();
      const cp = children.get(p) ?? [];
      const y = orientationSign * stack.length;
      if (cp.length === 0) pos.set(p, [obstructionAt(y), y]);
      else {
        const x: number = cp.reduce<number>((sum, child) => sum + Number(pos.get(child)?.[0] ?? 0), 0) / cp.length;
        pos.set(p, [x, y]);
        const ox = obstructionAt(y);
        if (x < ox) {
          slide(p, ox - x);
          pos.set(p, [ox, y]);
        }
      }
      setObstruction(y, (pos.get(p)?.[0] ?? 0) + 1);
      continue;
    }
    const t = current.pop() as GraphVertex;
    const pt = parent.get(t) ?? rootVertex;
    const ct = graph.neighbors(t).filter((u) => u !== pt);
    for (const child of ct) parent.set(child, t);
    children.set(t, [...ct]);
    stack.push(ct);
    stick.push(t);
  }
  const values = [...pos.values()];
  if (values.length === 0) return new Map();
  const xMin = Math.min(...values.map((item) => item[0]));
  const xMax = Math.max(...values.map((item) => item[0]));
  const yMin = Math.min(...values.map((item) => item[1]));
  const yMax = Math.max(...values.map((item) => item[1]));
  const center = [(xMin + xMax) / 2, (yMin + yMax) / 2, 0];
  const width = xMax - xMin;
  const height = yMax - yMin;
  let factor: number | number[];
  if (vertexSpacing) factor = [vertexSpacing[0], vertexSpacing[1], 0];
  else if (typeof scale === "number" && (width > 0 || height > 0)) factor = 2 * scale / Math.max(width, height);
  else if (Array.isArray(scale)) factor = [scale[0] !== undefined && width > 0 ? 2 * scale[0] / width : 1, scale[1] !== undefined && height > 0 ? 2 * scale[1] / height : 1, 0];
  else factor = 1;
  const result = new Map<GraphVertex, NDArray>();
  for (const [vertex, [x, y]] of pos) {
    const delta = [x - center[0], y - center[1], 0];
    const scaled = typeof factor === "number" ? delta.map((value) => value * factor) : delta.map((value, index) => value * (factor[index] ?? 1));
    result.set(vertex, array(scaled) as NDArray);
  }
  return result;
}

export const _tree_layout = tree_layout;

export type LayoutName = "circular" | "kamada_kawai" | "multipartite" | "planar" | "random" | "shell" | "spectral" | "spiral" | "spring" | "tree";

function notPortedLayout(name: string): LayoutFunction {
  return (_graph: NetworkXGraphLike, _scale?: number | number[]) => {
    throw new Error(`TODO: the "${name}" graph layout is not ported yet`);
  };
}

export const _layouts: Record<LayoutName, LayoutFunction> = {
  circular: notPortedLayout("circular"),
  kamada_kawai: notPortedLayout("kamada_kawai"),
  multipartite: notPortedLayout("multipartite"),
  planar: notPortedLayout("planar"),
  random: notPortedLayout("random"),
  shell: notPortedLayout("shell"),
  spectral: notPortedLayout("spectral"),
  spiral: notPortedLayout("spiral"),
  spring: notPortedLayout("spring"),
  tree: tree_layout as unknown as LayoutFunction,
};

export function _determine_graph_layout(
  nx_graph: NetworkXGraphLike,
  layout: LayoutName | Map<GraphVertex, NDArray> | Record<string, NDArray> | LayoutFunction = "spring",
  layout_scale: number | number[] = 2,
  layout_config: Record<string, unknown> | null = null,
): Map<GraphVertex, NDArray> | Record<string, NDArray> {
  const config = layout_config ?? {};
  if (layout instanceof Map || (typeof layout === "object" && layout !== null && !((layout as unknown) instanceof Function) && !(typeof layout === "string"))) {
    return layout as Map<GraphVertex, NDArray> | Record<string, NDArray>;
  } else if (typeof layout === "string" && layout in _layouts) {
    const auto_layout = (_layouts[layout as LayoutName] as unknown as (
      graph: NetworkXGraphLike,
      scale?: number | number[],
      config?: Record<string, unknown>,
    ) => Map<GraphVertex, NDArray>)(nx_graph, layout_scale, config);
    if (auto_layout.size === 0) return new Map();
    const first = auto_layout.values().next().value as NDArray | undefined;
    if ((config as Record<string, unknown>)["dim"] === 3 || (first?.shape?.[0] ?? 0) === 3) {
      return auto_layout;
    } else {
      const result = new Map<GraphVertex, NDArray>();
      for (const [key, value] of auto_layout) {
        const rows = value.toArray() as unknown;
        const flat = (Array.isArray(rows) ? rows : [rows]).map(Number);
        result.set(key, array([...flat, 0]) as NDArray);
      }
      return result;
    }
  } else {
    try {
      return (layout as LayoutFunction)(nx_graph, layout_scale);
    } catch {
      throw new Error(
        `The layout '${String(layout)}' is neither a recognized layout, a layout function, nor a vertex placement dictionary.`,
      );
    }
  }
}
