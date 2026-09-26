import { Mobject, Group } from "./mobject.js";
import { VMobject, VGroup } from "./types/vectorized_mobject.js";
import { array } from "numpy-ts";
import type { NDArray } from "numpy-ts";
import { addPoints, isNDArray, norm, pointRows, pointsFrom } from "../utils/arrays.js";
import { BLACK, type ParsableManimColor } from "../utils/color/core.js";
import { Dot, LabeledDot } from "./geometry/arc.js";

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

export type EdgeConstructor = new (...args: any[]) => Mobject;

export class GenericGraph extends VMobject {
  vertices: Map<GraphVertex, Mobject>;
  edges: Map<string, Mobject>;
  _graph: NetworkXGraphLike;
  _layout: Map<GraphVertex, NDArray>;
  _edge_config: Map<string, Record<string, unknown>> = new Map();
  _tip_config: Map<string, Record<string, unknown>> = new Map();
  _labels: Map<GraphVertex, unknown> = new Map();
  _vertex_config: Map<GraphVertex, Record<string, unknown>> = new Map();
  default_vertex_config: Record<string, unknown> = {};
  default_edge_config: Record<string, unknown> = {};

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

  _populate_edge_dict(_edges: GraphEdge[], _edge_type: EdgeConstructor): void {
    const error = new Error("To be implemented in concrete subclasses");
    error.name = "NotImplementedError";
    throw error;
  }

  _create_vertex(
    vertex: GraphVertex,
    position: number[] | readonly number[] | NDArray | null = null,
    label: boolean | Mobject | unknown = false,
    label_fill_color: ParsableManimColor = BLACK,
    vertex_type: typeof Dot | typeof LabeledDot | (new (config?: Record<string, unknown>) => Mobject) | null = null,
    vertex_config: Record<string, unknown> | null = null,
    vertex_mobject: Mobject | null = null,
  ): [GraphVertex, NDArray, Record<string, unknown>, Mobject] {
    // NOTE: mirrors Python `GenericGraph._create_vertex`: the position
    // defaults to the graph's center, the config starts from a copy of
    // default_vertex_config, and a non-null label upgrades Dot to
    // LabeledDot. `label=True` renders MathTex in Python (text/tex, out of
    // scope); a custom vertex_type is constructed with the config bag.
    const np_position: NDArray = position === null || position === undefined
      ? array([...this.get_center()]) as NDArray
      : isNDArray(position)
        ? position
        : array([...(position as readonly number[])].map(Number)) as NDArray;

    let resolved_config: Record<string, unknown> = vertex_config ?? {};
    void label_fill_color;

    if (this.vertices.has(vertex)) {
      const error = new Error(
        `Vertex identifier '${String(vertex)}' is already used for a vertex in this graph.`,
      );
      error.name = "ValueError";
      throw error;
    }

    let resolvedLabel: unknown;
    if (label === true) {
      // NOTE: Python renders `MathTex(vertex, color=label_fill_color)`;
      // text/tex rendering is out of scope for the entire port.
      throw new Error("TODO: text/tex not ported yet (Graph vertex label=True needs MathTex)");
    } else if (this._labels.has(vertex)) {
      resolvedLabel = this._labels.get(vertex);
    } else if (!(label instanceof Mobject)) {
      resolvedLabel = null;
    } else {
      resolvedLabel = label;
    }

    const base_vertex_config: Record<string, unknown> = { ...this.default_vertex_config };
    Object.assign(base_vertex_config, resolved_config);
    resolved_config = base_vertex_config;

    let ctor = vertex_type ?? Dot;
    if (resolvedLabel !== null && resolvedLabel !== undefined) {
      resolved_config["label"] = resolvedLabel;
      if (ctor === Dot) {
        ctor = LabeledDot;
      }
    }

    let mobject: Mobject;
    if (vertex_mobject !== null && vertex_mobject !== undefined) {
      mobject = vertex_mobject;
    } else if (ctor === LabeledDot) {
      const { label: cfgLabel, radius: cfgRadius, buff: cfgBuff, ...rest } = resolved_config as {
        label?: string | Mobject;
        radius?: number | null;
        buff?: number;
        [key: string]: unknown;
      };
      mobject = new LabeledDot(
        cfgLabel as string | Mobject,
        (cfgRadius as number | null) ?? null,
        (cfgBuff as number | undefined) ?? undefined,
        rest,
      );
    } else if (ctor === Dot) {
      // NOTE: Dot reads every constructor parameter from the trailing
      // options bag as well, so the whole config can go there (mirrors
      // Python `Dot(**vertex_config)`).
      mobject = new Dot(undefined, undefined, undefined, undefined, undefined, resolved_config);
    } else {
      const custom = ctor as new (config?: Record<string, unknown>) => Mobject;
      mobject = new custom({ ...resolved_config });
    }

    mobject.move_to(np_position);

    return [vertex, np_position, resolved_config, mobject];
  }

  _add_created_vertex(
    vertex: GraphVertex,
    position: number[] | NDArray,
    vertex_config: Record<string, unknown>,
    vertex_mobject: Mobject,
  ): Mobject {
    if (this.vertices.has(vertex)) {
      const error = new Error(
        `Vertex identifier '${String(vertex)}' is already used for a vertex in this graph.`,
      );
      error.name = "ValueError";
      throw error;
    }
    this._graph.addNode(vertex);
    this._layout.set(vertex, isNDArray(position) ? position : array(position) as NDArray);
    if ("label" in vertex_config) this._labels.set(vertex, vertex_config["label"]);
    this._vertex_config.set(vertex, vertex_config);
    this.vertices.set(vertex, vertex_mobject);
    (this.vertices.get(vertex) as Mobject).move_to(position);
    this.add(this.vertices.get(vertex) as Mobject);
    return this.vertices.get(vertex) as Mobject;
  }

  _add_vertex(
    vertex: GraphVertex,
    position: number[] | readonly number[] | NDArray | null = null,
    label: boolean | Mobject | unknown = false,
    label_fill_color: ParsableManimColor = BLACK,
    vertex_type: typeof Dot | typeof LabeledDot | (new (config?: Record<string, unknown>) => Mobject) | null = null,
    vertex_config: Record<string, unknown> | null = null,
    vertex_mobject: Mobject | null = null,
  ): Mobject {
    // NOTE: mirrors Python `GenericGraph._add_vertex`, which unpacks
    // `self._create_vertex(...)` positionally into `self._add_created_vertex`.
    // (Python defaults vertex_type to Dot and annotates vertex_mobject as
    // dict|None; the TS port follows the sibling _create_vertex instead —
    // null resolves to Dot there, which is behaviorally identical — and
    // types vertex_mobject as the Mobject it actually is.)
    const [, createdPosition, createdConfig, createdMobject] = this._create_vertex(
      vertex,
      position,
      label,
      label_fill_color,
      vertex_type,
      vertex_config,
      vertex_mobject,
    );
    return this._add_created_vertex(vertex, createdPosition, createdConfig, createdMobject);
  }
}

export class Graph extends GenericGraph {
  static _empty_networkx_graph(): NetworkXGraphLike {
    return new NetworkXGraph();
  }

  override _populate_edge_dict(edges: GraphEdge[], edge_type: EdgeConstructor): void {
    for (const [u, v] of edges) {
      const cfg = this._edge_config.get(edgeKey(u, v)) ?? {};
      this.edges.set(
        edgeKey(u, v),
        new edge_type(
          (this.getItem(u) as Mobject).get_center(),
          (this.getItem(v) as Mobject).get_center(),
          { z_index: -1, ...cfg },
        ),
      );
    }
  }
}

export class DiGraph extends GenericGraph {
  static _empty_networkx_graph(): NetworkXGraphLike {
    const graph = new NetworkXGraph();
    graph.directed = true;
    return graph;
  }

  override _populate_edge_dict(edges: GraphEdge[], edge_type: EdgeConstructor): void {
    for (const [u, v] of edges) {
      const cfg = this._edge_config.get(edgeKey(u, v)) ?? {};
      const edge = new edge_type(this.getItem(u), this.getItem(v), { z_index: -1, ...cfg });
      this.edges.set(edgeKey(u, v), edge);
    }
    for (const [u, v] of edges) {
      const edge = this.edges.get(edgeKey(u, v)) as Mobject;
      const tipCfg = this._tip_config.get(edgeKey(u, v)) ?? {};
      // NOTE: Python calls edge.add_tip(**tip_config); TipableVMobject.add_tip
      // takes positional arguments, so the known keys are mapped explicitly.
      (edge as unknown as {
        add_tip(tip?: unknown, tip_shape?: unknown, tip_length?: unknown, tip_width?: unknown, at_start?: unknown): void;
      }).add_tip(
        (tipCfg["tip"] as unknown) ?? null,
        (tipCfg["tip_shape"] as unknown) ?? null,
        (tipCfg["tip_length"] as unknown) ?? null,
        (tipCfg["tip_width"] as unknown) ?? null,
        (tipCfg["at_start"] as unknown) ?? false,
      );
    }
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
