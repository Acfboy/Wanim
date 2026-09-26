import { Mobject, Group, override_animate } from "./mobject.js";
import { VMobject, VGroup } from "./types/vectorized_mobject.js";
import { array } from "numpy-ts";
import type { NDArray } from "numpy-ts";
import { isNDArray, pointRows } from "../utils/arrays.js";
import { BLACK, type ParsableManimColor } from "../utils/color/core.js";
import { Dot, LabeledDot } from "./geometry/arc.js";
import { Line } from "./geometry/line.js";
import type { Animation } from "../animation/animation.js";
import { AnimationGroup } from "../animation/composition.js";
import { Create, Uncreate } from "../animation/creation.js";

export type GraphVertex = string | number | boolean | object;
export type GraphEdge = readonly [GraphVertex, GraphVertex];

export interface NetworkXGraphLike {
  nodes: Map<GraphVertex, Record<string, unknown>>;
  edges: Set<string>;
  addNode(node: GraphVertex): void;
  addNodesFrom(nodes: Iterable<GraphVertex>): void;
  addEdge(u: GraphVertex, v: GraphVertex): void;
  addEdgesFrom(edges: Iterable<GraphEdge>): void;
  removeNode(node: GraphVertex): void;
  removeEdge(u: GraphVertex, v: GraphVertex): void;
  getEdgePairs(): GraphEdge[];
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
  removeNode(node: GraphVertex): void {
    this.nodes.delete(node);
    this.edge_pairs = this.edge_pairs.filter(([u, v]) => u !== node && v !== node);
    this._rebuildEdgeSet();
  }
  removeEdge(u: GraphVertex, v: GraphVertex): void {
    // NOTE: networkx remove_edge raises NetworkXError for absent edges; the
    // TS graph layer checks membership first (raising ValueError, like the
    // Python dict lookup), so this stays lenient by design.
    let index = this.edge_pairs.findIndex(([a, b]) => a === u && b === v);
    if (index === -1 && !this.directed) {
      index = this.edge_pairs.findIndex(([a, b]) => a === v && b === u);
    }
    if (index !== -1) this.edge_pairs.splice(index, 1);
    this._rebuildEdgeSet();
  }
  getEdgePairs(): GraphEdge[] {
    return this.edge_pairs.map(([u, v]) => [u, v] as GraphEdge);
  }
  private _rebuildEdgeSet(): void {
    this.edges = new Set<string>();
    for (const [u, v] of this.edge_pairs) {
      this.edges.add(edgeKey(u, v));
      if (this.directed) this.edges.add(edgeKey(v, u));
    }
  }
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

export type GraphVertexType =
  | typeof Dot
  | typeof LabeledDot
  | (new (config?: Record<string, unknown>) => Mobject)
  | null;

export interface GenericGraphOptions {
  labels?: boolean | Map<GraphVertex, unknown> | Record<string, unknown>;
  label_fill_color?: ParsableManimColor;
  layout?: LayoutName | Map<GraphVertex, NDArray> | Record<string, unknown> | LayoutFunction;
  layout_scale?: number | number[];
  layout_config?: Record<string, unknown> | null;
  vertex_type?: GraphVertexType;
  vertex_config?: Record<string, unknown> | null;
  vertex_mobjects?: Map<GraphVertex, Mobject> | Record<string, Mobject> | null;
  edge_type?: EdgeConstructor;
  partitions?: GraphVertex[][] | null;
  root_vertex?: GraphVertex | null;
  edge_config?: Record<string, unknown> | Map<unknown, Record<string, unknown>> | null;
  [key: string]: unknown;
}

/** Plain-object check for trailing options bags (mirrors the convention used
 * by Line/Arrow/LabeledLine: class instances, arrays, Maps and NDArrays are
 * never option bags). */
function isGraphOptionsBag(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (value instanceof Mobject) return false;
  if (value instanceof Map) return false;
  if (isNDArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function normalizeVertexMobjects(
  vertex_mobjects: Map<GraphVertex, Mobject> | Record<string, Mobject> | null | undefined,
): Array<[GraphVertex, Mobject]> {
  if (vertex_mobjects === null || vertex_mobjects === undefined) return [];
  if (vertex_mobjects instanceof Map) return [...vertex_mobjects.entries()];
  return Object.entries(vertex_mobjects) as [GraphVertex, Mobject][];
}

/** Split an edge_config into the global tip defaults, the global edge
 * defaults and the per-edge overrides. Mirrors Python `not isinstance(k,
 * tuple)`: array pairs (and edgeKey strings matching a graph edge) are
 * per-edge entries, everything else is a global option. */
function splitEdgeConfig(
  edge_config: Record<string, unknown> | Map<unknown, Record<string, unknown>> | null | undefined,
  edges: GraphEdge[],
  pop_tip_config = true,
): {
  default_tip_config: Record<string, unknown>;
  default_edge_config: Record<string, unknown>;
  per_edge_config: Array<{ pair: [GraphVertex, GraphVertex]; config: Record<string, unknown> }>;
} {
  let default_tip_config: Record<string, unknown> = {};
  let default_edge_config: Record<string, unknown> = {};
  const per_edge_config: Array<{ pair: [GraphVertex, GraphVertex]; config: Record<string, unknown> }> = [];
  if (edge_config === null || edge_config === undefined) {
    return { default_tip_config, default_edge_config, per_edge_config };
  }
  const entries: Array<[unknown, unknown]> = edge_config instanceof Map
    ? [...edge_config.entries()]
    : Object.entries(edge_config);
  for (const [key, value] of entries) {
    // NOTE: only __init__ pops the global "tip_config" (mirroring Python's
    // `edge_config.pop("tip_config", {})` there); add_edges treats it as an
    // ordinary option, like Python.
    if (pop_tip_config && typeof key === "string" && key === "tip_config") {
      default_tip_config = { ...((value as Record<string, unknown>) ?? {}) };
      continue;
    }
    if (Array.isArray(key) && key.length === 2) {
      per_edge_config.push({
        pair: [key[0] as GraphVertex, key[1] as GraphVertex],
        config: { ...((value as Record<string, unknown>) ?? {}) },
      });
      continue;
    }
    if (typeof key === "string" && key.includes("\u0000")) {
      const match = edges.find((e) => edgeKey(e[0], e[1]) === key);
      if (match !== undefined) {
        per_edge_config.push({
          pair: [match[0], match[1]],
          config: { ...((value as Record<string, unknown>) ?? {}) },
        });
        continue;
      }
    }
    default_edge_config[key as string] = value;
  }
  return { default_tip_config, default_edge_config, per_edge_config };
}

export interface ParsedVertexArgs {
  positions?: Map<GraphVertex, number[] | readonly number[] | NDArray> | Record<string, number[] | readonly number[] | NDArray> | null;
  labels?: boolean | Map<GraphVertex, boolean | Mobject | unknown> | Record<string, boolean | Mobject | unknown> | null;
  label_fill_color?: ParsableManimColor;
  vertex_type?: GraphVertexType;
  vertex_config?: Record<string, unknown> | null;
  vertex_mobjects?: Map<GraphVertex, Mobject> | Record<string, Mobject> | null;
  [key: string]: unknown;
}

const VERTEX_ARG_KEYS = new Set([
  "positions",
  "labels",
  "label_fill_color",
  "vertex_type",
  "vertex_config",
  "vertex_mobjects",
]);

/** Split `(*vertices, **kwargs)`-style args into leading vertices and the
 * trailing options bag (recognized by its known keys, like
 * CoordinateSystem.add_coordinates). */
function parseVertexArgs(args: unknown[]): [GraphVertex[], ParsedVertexArgs] {
  let vertexArgs: unknown[] = args;
  let opts: ParsedVertexArgs = {};
  const last = args[args.length - 1];
  if (
    args.length > 0 &&
    isGraphOptionsBag(last) &&
    Object.keys(last).some((key) => VERTEX_ARG_KEYS.has(key))
  ) {
    opts = last as ParsedVertexArgs;
    vertexArgs = args.slice(0, -1);
  }
  return [vertexArgs as GraphVertex[], opts];
}

/** Variant of parseVertexArgs that additionally pulls an `anim_args` bag
 * (used by the @override_animate animation overrides). */
function parseVertexArgsWithAnim(
  args: unknown[],
): [GraphVertex[], ParsedVertexArgs, Record<string, unknown>] {
  const [vertices, opts] = parseVertexArgs(args);
  const anim_args = { ...((opts["anim_args"] as Record<string, unknown> | undefined) ?? {}) };
  return [vertices, opts, anim_args];
}

/** Split bare `(*items)`-style args (remove_vertices/remove_edges animation
 * overrides) into items plus an optional trailing `{ anim_args }` bag. */
function parseBareArgsWithAnim(
  args: unknown[],
): { vertices: unknown[]; anim_args: Record<string, unknown> } {
  let items: unknown[] = args;
  let anim_args: Record<string, unknown> = {};
  const last = args[args.length - 1];
  if (
    args.length > 0 &&
    isGraphOptionsBag(last) &&
    Object.keys(last).length === 1 &&
    "anim_args" in last
  ) {
    anim_args = { ...((last["anim_args"] as Record<string, unknown>) ?? {}) };
    items = args.slice(0, -1);
  }
  return { vertices: items, anim_args };
}

function normalizePositions(
  positions: ParsedVertexArgs["positions"],
): Array<[GraphVertex, number[] | readonly number[] | NDArray]> {
  if (positions === null || positions === undefined) return [];
  if (positions instanceof Map) return [...positions.entries()];
  return Object.entries(positions) as [GraphVertex, number[] | readonly number[] | NDArray][];
}

function normalizeLabels(
  labels: ParsedVertexArgs["labels"],
): Array<[GraphVertex, boolean | Mobject | unknown]> {
  if (labels === null || labels === undefined || typeof labels === "boolean") return [];
  if (labels instanceof Map) return [...labels.entries()];
  return Object.entries(labels) as [GraphVertex, boolean | Mobject | unknown][];
}

const EDGE_ARG_KEYS = new Set([
  ...VERTEX_ARG_KEYS,
  "edge_type",
  "edge_config",
]);

/** Split `(*edges, edge_type=..., edge_config=..., **kwargs)`-style args:
 * leading array/string items are edges, an optional trailing options bag
 * carries edge_type/edge_config plus vertex kwargs forwarded to
 * add_vertices for newly created vertices. */
function parseEdgeArgs(
  args: unknown[],
): [GraphEdge[], Record<string, unknown>] {
  let edgeArgs: unknown[] = args;
  let opts: Record<string, unknown> = {};
  const last = args[args.length - 1];
  if (
    args.length > 0 &&
    isGraphOptionsBag(last) &&
    Object.keys(last).some((key) => EDGE_ARG_KEYS.has(key))
  ) {
    opts = last as Record<string, unknown>;
    edgeArgs = args.slice(0, -1);
  }
  return [
    edgeArgs.map((e) =>
      typeof e === "string" && e.includes("\u0000")
        ? (e.split("\u0000") as unknown as GraphEdge)
        : (e as GraphEdge),
    ),
    opts,
  ];
}

/** Variant of parseEdgeArgs that additionally pulls an `anim_args` bag. */
function parseEdgeArgsWithAnim(
  args: unknown[],
): [GraphEdge[], Record<string, unknown>, Record<string, unknown>] {
  const [edges, opts] = parseEdgeArgs(args);
  const anim_args = { ...((opts["anim_args"] as Record<string, unknown> | undefined) ?? {}) };
  return [edges, opts, anim_args];
}

export interface ChangeLayoutOptions {
  layout?: LayoutName | Map<GraphVertex, NDArray> | Record<string, unknown> | LayoutFunction;
  layout_scale?: number | number[];
  layout_config?: Record<string, unknown> | null;
  partitions?: GraphVertex[][] | null;
  root_vertex?: GraphVertex | null;
  [key: string]: unknown;
}

function hasLayoutKeys(value: Record<string, unknown>): boolean {
  return ["layout", "layout_scale", "layout_config", "partitions", "root_vertex"].some(
    (key) => key in value,
  );
}

/** Look a vertex up in a layout map, tolerating stringified numeric keys
 * (JS plain-object placement dicts stringify their keys, unlike Python). */
function lookupLayout(
  layout: Map<GraphVertex, NDArray>,
  vertex: GraphVertex,
): NDArray | undefined {
  if (layout.has(vertex)) return layout.get(vertex);
  for (const [key, value] of layout) {
    if (String(key) === String(vertex)) return value;
  }
  return undefined;
}

export class GenericGraph extends VMobject {
  vertices: Map<GraphVertex, Mobject>;
  edges: Map<string, Mobject>;
  _graph: NetworkXGraphLike;
  _layout: Map<GraphVertex, NDArray>;
  _edge_config: Map<string, Record<string, unknown>> = new Map();
  _tip_config: Map<string, Record<string, unknown>> = new Map();
  _edge_pairs: Map<string, [GraphVertex, GraphVertex]> = new Map();
  _labels: Map<GraphVertex, unknown> = new Map();
  _vertex_config: Map<GraphVertex, Record<string, unknown>> = new Map();
  default_vertex_config: Record<string, unknown> = {};
  default_edge_config: Record<string, unknown> = {};
  // NOTE: mirrors Python, where only the concrete subclasses (Graph/DiGraph)
  // define update_edges while GenericGraph.__init__ already registers
  // `self.update_edges` as an updater. The base throws (like
  // _populate_edge_dict); the subclass override binds on concrete graphs.
  // (Python raises AttributeError when reading it on the bare base instead;
  // the base is documented abstract and never instantiated directly.)
  update_edges(_graph: GenericGraph): unknown {
    const error = new Error("To be implemented in concrete subclasses");
    error.name = "NotImplementedError";
    throw error;
  }

  constructor(vertices: GraphVertex[] = [], edges: GraphEdge[] = [], options: GenericGraphOptions = {}) {
    const {
      labels = false,
      label_fill_color = BLACK,
      layout = "spring",
      layout_scale = 2,
      layout_config = null,
      vertex_type = Dot,
      vertex_config = null,
      vertex_mobjects = null,
      edge_type = Line,
      partitions = null,
      root_vertex = null,
      edge_config = null,
      ...rest
    } = options;
    super(rest);
    this.vertices = new Map();
    this.edges = new Map();
    this._layout = new Map();
    this._graph = (this.constructor as typeof GenericGraph)._empty_networkx_graph();
    this._graph.addNodesFrom(vertices);
    this._graph.addEdgesFrom(edges);

    if (labels instanceof Map) {
      this._labels = new Map(labels);
    } else if (typeof labels === "object" && labels !== null) {
      this._labels = new Map(
        Object.entries(labels as Record<string, unknown>) as [GraphVertex, unknown][],
      );
    } else if (labels === true) {
      // NOTE: Python builds `{v: MathTex(v, color=label_fill_color)}`; the
      // text/LaTeX pipeline is out of scope for the entire port.
      void label_fill_color;
      throw new Error("TODO: text/tex not ported yet (Graph labels=True needs MathTex)");
    } else {
      this._labels = new Map();
    }

    let resolved_vertex_type = vertex_type ?? Dot;
    if (this._labels.size > 0 && resolved_vertex_type === Dot) {
      resolved_vertex_type = LabeledDot;
    }

    // NOTE: mirrors Python `vertex_config.get(v, copy(default))`: JS object
    // keys are strings, so numeric vertices match their stringified key
    // (Python `1 in {1: ...}`); reference-equality is kept for parity
    // otherwise. Per-vertex entries alias the caller's dict, like Python.
    const raw_vertex_config: Record<string, unknown> = { ...(vertex_config ?? {}) };
    const default_vertex_config: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(raw_vertex_config)) {
      if (!vertices.some((v) => String(v) === key)) {
        default_vertex_config[key] = value;
      }
    }
    this.default_vertex_config = default_vertex_config;
    for (const v of vertices) {
      const own = Object.prototype.hasOwnProperty.call(raw_vertex_config, v as unknown as string)
        ? (raw_vertex_config[v as unknown as string] as Record<string, unknown>)
        : { ...default_vertex_config };
      this._vertex_config.set(v, own);
    }
    for (const [v, label] of this._labels) {
      const cfg = this._vertex_config.get(v);
      if (cfg !== undefined) cfg["label"] = label;
    }

    for (const v of vertices) {
      this.vertices.set(
        v,
        this._build_vertex_mobject(resolved_vertex_type, this._vertex_config.get(v) ?? {}),
      );
    }
    for (const [v, mobj] of normalizeVertexMobjects(vertex_mobjects)) {
      this.vertices.set(this._match_vertex_key(v), mobj);
    }

    try {
      this.change_layout(layout, layout_scale, layout_config, partitions, root_vertex);
    } catch (error) {
      // NOTE: networkx automatic layouts (spring/circular/...) need networkx
      // and are unavailable in TS (see the _layouts stubs); a faithful
      // change_layout therefore throws `TODO: ... not ported yet`. The
      // constructor degrades gracefully to the constructed (center)
      // positions so `new Graph(...)` stays usable (mirroring how the
      // previous placeholder constructor behaved); calling change_layout
      // directly still raises. Manual (dict) and tree layouts take the
      // faithful path above.
      if (error instanceof Error && /not ported yet/.test(error.message)) {
        for (const v of vertices) {
          const center = this.vertices.get(v)?.get_center() ?? [0, 0, 0];
          this._layout.set(v, array([...center]) as NDArray);
        }
      } else {
        throw error;
      }
    }

    // NOTE: mirrors Python, which pops "tip_config" off the passed dict
    // (mutating the caller's object); the TS port copies first instead.
    const { default_tip_config, default_edge_config, per_edge_config } =
      splitEdgeConfig(edge_config, edges);
    this.default_edge_config = default_edge_config;
    for (const e of edges) {
      const key = edgeKey(e[0], e[1]);
      const override = per_edge_config.find(
        (entry) => edgeKey(entry.pair[0], entry.pair[1]) === key,
      );
      if (override !== undefined) {
        const { tip_config, ...rest_config } = override.config;
        this._tip_config.set(
          key,
          { ...(tip_config as Record<string, unknown> | undefined ?? default_tip_config) },
        );
        this._edge_config.set(key, { ...rest_config });
      } else {
        this._tip_config.set(key, { ...default_tip_config });
        this._edge_config.set(key, { ...default_edge_config });
      }
    }
    this._populate_edge_dict(edges, edge_type);

    this.add(...this.vertices.values(), ...this.edges.values());

    // NOTE: mirrors Python `self.add_updater(self.update_edges)` (bound
    // subclass override); the eager read raises AttributeError on the
    // abstract base through the Mobject proxy, like Python.
    const boundUpdate = this.update_edges;
    this.add_updater(() => {
      (boundUpdate as (graph: GenericGraph) => void).call(this, this);
    });
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
    // NOTE: mirrors Python `elif k in self.edges` (False for anything that
    // is not an edge tuple) instead of spreading arbitrary keys into edgeKey.
    if (Array.isArray(key) && key.length === 2) {
      const edge = this.edges.get(edgeKey(key[0] as GraphVertex, key[1] as GraphVertex));
      if (edge) return edge;
    }
    const error = new Error(`Could not find ${String(key)} in vertices or edges`);
    error.name = "ValueError";
    throw error;
  }

  __getitem__(key: GraphVertex | GraphEdge): Mobject {
    return this.getItem(key);
  }

  /** Construct a vertex mobject from a type plus its config bag, mirroring
   * Python `vertex_type(**vertex_config)` (with Dot/LabeledDot handled
   * explicitly, like _create_vertex). */
  _build_vertex_mobject(
    vertex_type: GraphVertexType,
    vertex_config: Record<string, unknown>,
  ): Mobject {
    const ctor = vertex_type ?? Dot;
    if (ctor === LabeledDot) {
      const { label, radius, buff, ...rest } = vertex_config as {
        label?: string | Mobject;
        radius?: number | null;
        buff?: number;
        [key: string]: unknown;
      };
      return new LabeledDot(
        label as string | Mobject,
        (radius as number | null) ?? null,
        (buff as number | undefined) ?? undefined,
        rest,
      );
    }
    if (ctor === Dot) {
      // NOTE: Dot reads every constructor parameter from the trailing
      // options bag as well, so the whole config goes there (mirrors
      // Python `Dot(**vertex_config)`).
      return new Dot(undefined, undefined, undefined, undefined, undefined, { ...vertex_config });
    }
    const custom = ctor as new (config?: Record<string, unknown>) => Mobject;
    return new custom({ ...vertex_config });
  }

  /** Resolve a vertex key against the vertices map. Exact matches win;
   * numeric vertices also match their stringified form (JS plain-object
   * configs can only carry string keys, unlike Python dicts). */
  _match_vertex_key(key: GraphVertex): GraphVertex {
    if (this.vertices.has(key)) return key;
    if (typeof key === "string" && key.trim() !== "" && !Number.isNaN(Number(key))) {
      const numeric = Number(key);
      for (const v of this.vertices.keys()) {
        if (v === numeric) return v;
      }
    }
    return key;
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

  _create_vertices(
    ...args: unknown[]
  ): Array<[GraphVertex, NDArray, Record<string, unknown>, Mobject]> {
    return this._create_vertices_from(...parseVertexArgs(args));
  }

  /** Shared implementation behind _create_vertices/add_vertices (and the
   * _add_vertices_animation override): mirrors Python `_create_vertices`
   * verbatim, including the default-merging of positions/labels/configs. */
  _create_vertices_from(
    vertices: GraphVertex[],
    opts: ParsedVertexArgs,
  ): Array<[GraphVertex, NDArray, Record<string, unknown>, Mobject]> {
    const graphCenter = this.get_center();
    // NOTE: mirrors Python dict lookups (`positions[v]`, `labels[v]`,
    // `vertex_config[v]`): JS plain-object configs stringify numeric keys,
    // so "3" must match vertex 3 (unlike the _match_vertex_key fallback
    // against existing vertices, these keys are resolved against the
    // vertices being created).
    const matchNew = (key: GraphVertex): GraphVertex => {
      if (vertices.some((v) => v === key)) return key;
      if (typeof key === "string" && key.trim() !== "" && !Number.isNaN(Number(key))) {
        const numeric = Number(key);
        const found = vertices.find((v) => v === numeric);
        if (found !== undefined) return found;
      }
      return key;
    };
    const basePositions = new Map<GraphVertex, number[] | readonly number[] | NDArray>(
      vertices.map((v) => [v, graphCenter]),
    );
    for (const [v, position] of normalizePositions(opts.positions)) {
      basePositions.set(matchNew(v), position);
    }

    let resolvedLabels: Map<GraphVertex, boolean | Mobject | unknown>;
    if (typeof opts.labels === "boolean") {
      resolvedLabels = new Map(vertices.map((v) => [v, opts.labels as boolean]));
    } else {
      resolvedLabels = new Map(vertices.map((v) => [v, false as unknown]));
      for (const [v, label] of normalizeLabels(opts.labels)) {
        resolvedLabels.set(matchNew(v), label);
      }
    }

    const raw = { ...(opts.vertex_config ?? {}) } as Record<string, unknown>;
    let base_vertex_config: Record<string, unknown>;
    if (opts.vertex_config === null || opts.vertex_config === undefined) {
      base_vertex_config = { ...this.default_vertex_config };
    } else {
      base_vertex_config = { ...this.default_vertex_config };
      for (const [key, value] of Object.entries(raw)) {
        if (!vertices.some((v) => String(v) === key)) {
          base_vertex_config[key] = value;
        }
      }
    }
    const perVertexConfig = new Map<GraphVertex, Record<string, unknown>>();
    for (const v of vertices) {
      if (
        opts.vertex_config !== null &&
        opts.vertex_config !== undefined &&
        Object.prototype.hasOwnProperty.call(raw, v as unknown as string)
      ) {
        perVertexConfig.set(v, raw[v as unknown as string] as Record<string, unknown>);
      } else {
        perVertexConfig.set(v, { ...base_vertex_config });
      }
    }

    const mobjects = normalizeVertexMobjects(opts.vertex_mobjects);
    const mobjectMap = new Map<GraphVertex, Mobject>();
    for (const [v, mobj] of mobjects) {
      mobjectMap.set(matchNew(v), mobj);
    }

    return vertices.map((v) => {
      const [createdVertex, createdPosition, createdConfig, createdMobject] =
        this._create_vertex(
          v,
          basePositions.get(v) ?? graphCenter,
          resolvedLabels.get(v) ?? false,
          opts.label_fill_color,
          opts.vertex_type,
          perVertexConfig.get(v) ?? {},
          mobjectMap.has(v) ? (mobjectMap.get(v) as Mobject) : null,
        );
      return [createdVertex, createdPosition, createdConfig, createdMobject] as [
        GraphVertex, NDArray, Record<string, unknown>, Mobject,
      ];
    });
  }

  add_vertices(...args: unknown[]): Mobject[] {
    const [vertices, opts] = parseVertexArgs(args);
    return this._create_vertices_from(vertices, opts).map(
      ([vertex, position, vertex_config, vertex_mobject]) =>
        this._add_created_vertex(vertex, position, vertex_config, vertex_mobject),
    );
  }

  _add_vertices_animation(...args: unknown[]): AnimationGroup {
    const [vertices, opts, anim_args] = parseVertexArgsWithAnim(args);
    const animation = (anim_args["animation"] as new (
      mobject: VMobject,
      options?: Record<string, unknown>,
    ) => Animation) ?? Create;
    const rest_anim_args = { ...anim_args };
    delete rest_anim_args["animation"];
    const vertex_mobjects = this._create_vertices_from(vertices, opts);
    const on_finish = (scene: { remove(mobject: Mobject): void }): void => {
      for (const v of vertex_mobjects) {
        scene.remove(v[3]);
        this._add_created_vertex(v[0], v[1], v[2], v[3]);
      }
    };
    return new AnimationGroup(
      ...vertex_mobjects.map((v) => new animation(v[3] as VMobject, rest_anim_args)),
      { group: this as unknown as VGroup, _on_finish: on_finish },
    );
  }

  _remove_vertex(vertex: GraphVertex): VGroup {
    if (!this.vertices.has(vertex)) {
      const error = new Error(
        `The graph does not contain a vertex with identifier '${String(vertex)}'`,
      );
      error.name = "ValueError";
      throw error;
    }

    this._graph.removeNode(vertex);
    this._layout.delete(vertex);
    if (this._labels.has(vertex)) {
      this._labels.delete(vertex);
    }
    this._vertex_config.delete(vertex);

    const edge_tuples = [...this._edge_pairs.entries()]
      .filter(([, pair]) => pair[0] === vertex || pair[1] === vertex)
      .map(([key]) => key);
    for (const key of edge_tuples) {
      // NOTE: mirrors Python, which pops only _edge_config here (a stale
      // _tip_config entry is left behind upstream too).
      this._edge_config.delete(key);
    }
    const to_remove: Mobject[] = edge_tuples.map((key) => {
      const pair = this._edge_pairs.get(key) as [GraphVertex, GraphVertex];
      this._edge_pairs.delete(key);
      return this.edges.get(key) as Mobject;
    });
    for (const key of edge_tuples) {
      this.edges.delete(key);
    }
    to_remove.push(this.vertices.get(vertex) as Mobject);
    this.vertices.delete(vertex);

    this.remove(...to_remove);
    return new (this.get_group_class())(...(to_remove as VMobject[]));
  }

  remove_vertices(...vertices: GraphVertex[]): VGroup {
    const mobjects: Mobject[] = [];
    for (const v of vertices) {
      mobjects.push(...this._remove_vertex(v).submobjects);
    }
    return new (this.get_group_class())(...(mobjects as VMobject[]));
  }

  _remove_vertices_animation(...args: unknown[]): AnimationGroup {
    const { vertices, anim_args } = parseBareArgsWithAnim(args);
    const animation = (anim_args["animation"] as new (
      mobject: VMobject,
      options?: Record<string, unknown>,
    ) => Animation) ?? Uncreate;
    const rest_anim_args = { ...anim_args };
    delete rest_anim_args["animation"];
    const mobjects = this.remove_vertices(...(vertices as GraphVertex[]));
    return new AnimationGroup(
      ...mobjects.submobjects.map((mobj) => new animation(mobj as VMobject, rest_anim_args)),
      { group: this as unknown as VGroup },
    );
  }

  _add_edge(
    edge: GraphEdge,
    edge_type: EdgeConstructor = Line,
    edge_config: Record<string, unknown> | null = null,
  ): VGroup {
    // NOTE: mirrors Python `edge_config if given else
    // default_edge_config.copy()`: a passed dict is used as-is (aliased),
    // otherwise a copy of the defaults.
    let resolved_config: Record<string, unknown> = edge_config ?? {
      ...this.default_edge_config,
    };
    const added_mobjects: Mobject[] = [
      ...(edge.filter((v) => !this.vertices.has(v)) as GraphVertex[]).map((v) =>
        this._add_vertex(v),
      ),
    ];
    const [u, v] = edge as [GraphVertex, GraphVertex];

    this._graph.addEdge(u, v);

    const base_edge_config = { ...this.default_edge_config };
    Object.assign(base_edge_config, resolved_config);
    resolved_config = base_edge_config;
    const key = edgeKey(u, v);
    this._edge_config.set(key, resolved_config);

    const edge_mobject = new edge_type(
      this.getItem(u).get_center(),
      this.getItem(v).get_center(),
      { z_index: -1, ...resolved_config },
    );
    this.edges.set(key, edge_mobject);
    this._edge_pairs.set(key, [u, v]);

    this.add(edge_mobject);
    added_mobjects.push(edge_mobject);
    return new (this.get_group_class())(...(added_mobjects as VMobject[]));
  }

  add_edges(...args: unknown[]): VGroup {
    const [edges, opts] = parseEdgeArgs(args);
    const edge_type = (opts.edge_type as EdgeConstructor | undefined) ?? Line;
    const raw_edge_config = (opts.edge_config as
      | Record<string, unknown>
      | Map<unknown, Record<string, unknown>>
      | undefined) ?? {};
    const { default_edge_config: non_edge_defaults, per_edge_config } = splitEdgeConfig(
      raw_edge_config,
      edges,
      // NOTE: mirrors Python add_edges, which has no "tip_config" special
      // case (unlike __init__): every non-edge key is a plain edge option.
      false,
    );
    const base_edge_config = { ...this.default_edge_config, ...non_edge_defaults };
    const resolved_edge_config = new Map<string, Record<string, unknown>>();
    for (const e of edges) {
      resolved_edge_config.set(edgeKey(e[0], e[1]), { ...base_edge_config });
    }
    for (const entry of per_edge_config) {
      const key = edgeKey(entry.pair[0], entry.pair[1]);
      Object.assign(resolved_edge_config.get(key) ?? {}, entry.config);
      if (!resolved_edge_config.has(key)) {
        resolved_edge_config.set(key, { ...entry.config });
      }
    }

    const vertex_kwarg_keys = [
      "positions",
      "labels",
      "label_fill_color",
      "vertex_type",
      "vertex_config",
      "vertex_mobjects",
    ];
    const has_vertex_kwargs = vertex_kwarg_keys.some((k) => k in opts);
    const seen = new Set<GraphVertex>();
    const new_vertices: GraphVertex[] = [];
    for (const e of edges) {
      for (const v of e) {
        if (!this.vertices.has(v) && !seen.has(v)) {
          seen.add(v);
          new_vertices.push(v);
        }
      }
    }
    const added_vertices = has_vertex_kwargs
      ? this.add_vertices(
          ...new_vertices,
          Object.fromEntries(vertex_kwarg_keys.filter((k) => k in opts).map((k) => [k, opts[k]])),
        )
      : this.add_vertices(...new_vertices);

    const added_mobjects: Mobject[] = [...added_vertices];
    for (const e of edges) {
      added_mobjects.push(
        ...this._add_edge(
          e,
          edge_type,
          resolved_edge_config.get(edgeKey(e[0], e[1])) ?? {},
        ).submobjects,
      );
    }
    return new (this.get_group_class())(...(added_mobjects as VMobject[]));
  }

  _add_edges_animation(...args: unknown[]): AnimationGroup {
    const [edges, opts, anim_args] = parseEdgeArgsWithAnim(args);
    const animation = (anim_args["animation"] as new (
      mobject: VMobject,
      options?: Record<string, unknown>,
    ) => Animation) ?? Create;
    const rest_anim_args = { ...anim_args };
    delete rest_anim_args["animation"];
    const { anim_args: _dropped, ...forward_opts } = opts;
    void _dropped;
    const mobjects = this.add_edges(...edges, ...(Object.keys(forward_opts).length > 0 ? [forward_opts] : []));
    return new AnimationGroup(
      ...mobjects.submobjects.map((mobj) => new animation(mobj as VMobject, rest_anim_args)),
      { group: this as unknown as VGroup },
    );
  }

  _remove_edge(edge: GraphEdge | string): Mobject {
    const key = typeof edge === "string" ? edge : edgeKey(edge[0], edge[1]);
    if (!this.edges.has(key)) {
      const error = new Error(`The graph does not contain a edge '${String(edge)}'`);
      error.name = "ValueError";
      throw error;
    }

    const edge_mobject = this.edges.get(key) as Mobject;
    this.edges.delete(key);

    const pair = this._edge_pairs.get(key);
    this._edge_pairs.delete(key);
    if (pair !== undefined) {
      this._graph.removeEdge(pair[0], pair[1]);
    }
    // NOTE: mirrors Python `self._edge_config.pop(edge, None)` (lenient).
    this._edge_config.delete(key);

    this.remove(edge_mobject);
    return edge_mobject;
  }

  remove_edges(...edges: Array<GraphEdge | string>): VGroup {
    const edge_mobjects = edges.map((edge) => this._remove_edge(edge));
    return new (this.get_group_class())(...(edge_mobjects as VMobject[]));
  }

  _remove_edges_animation(...args: unknown[]): AnimationGroup {
    const { vertices: edges, anim_args } = parseBareArgsWithAnim(args);
    const animation = (anim_args["animation"] as new (
      mobject: VMobject,
      options?: Record<string, unknown>,
    ) => Animation) ?? Uncreate;
    const rest_anim_args = { ...anim_args };
    delete rest_anim_args["animation"];
    const mobjects = this.remove_edges(...(edges as Array<GraphEdge | string>));
    return new AnimationGroup(
      ...mobjects.submobjects.map((mobj) => new animation(mobj as VMobject, rest_anim_args)),
    );
  }

  static from_networkx(
    nxgraph: NetworkXGraphLike,
    options: GenericGraphOptions = {},
  ): GenericGraph {
    const ctor = this as unknown as new (
      vertices: GraphVertex[],
      edges: GraphEdge[],
      options?: GenericGraphOptions,
    ) => GenericGraph;
    return new ctor([...nxgraph.nodes.keys()], nxgraph.getEdgePairs(), options);
  }

  change_layout(
    layout:
      | LayoutName
      | Map<GraphVertex, NDArray>
      | Record<string, unknown>
      | LayoutFunction
      | ChangeLayoutOptions = "spring",
    layout_scale: number | number[] = 2,
    layout_config: Record<string, unknown> | null = null,
    partitions: GraphVertex[][] | null = null,
    root_vertex: GraphVertex | null = null,
    options: ChangeLayoutOptions = {},
  ): this {
    // NOTE: mirrors Python `change_layout(layout="spring", layout_scale=2,
    // layout_config=None, partitions=None, root_vertex=None)`: a trailing
    // (or leading) options bag carrying layout keys wins over positionals,
    // following the Line/Arrow convention of this port.
    let resolved_layout = layout;
    let resolved_scale = layout_scale;
    let resolved_config = layout_config;
    let resolved_partitions = partitions;
    let resolved_root = root_vertex;
    const bags: Record<string, unknown>[] = [];
    if (isGraphOptionsBag(layout) && hasLayoutKeys(layout)) bags.push(layout);
    if (isGraphOptionsBag(options) && Object.keys(options).length > 0) bags.push(options);
    for (const bag of bags) {
      if ("layout" in bag) resolved_layout = bag["layout"] as typeof resolved_layout;
      if ("layout_scale" in bag) resolved_scale = bag["layout_scale"] as typeof resolved_scale;
      if ("layout_config" in bag) {
        resolved_config = bag["layout_config"] as Record<string, unknown> | null;
      }
      if ("partitions" in bag) resolved_partitions = bag["partitions"] as GraphVertex[][] | null;
      if ("root_vertex" in bag) resolved_root = bag["root_vertex"] as GraphVertex | null;
    }
    const final_config: Record<string, unknown> = { ...(resolved_config ?? {}) };
    if (resolved_partitions !== null && !("partitions" in final_config)) {
      final_config["partitions"] = resolved_partitions;
    }
    if (resolved_root !== null && !("root_vertex" in final_config)) {
      final_config["root_vertex"] = resolved_root;
    }

    const raw = _determine_graph_layout(
      this._graph,
      resolved_layout as LayoutName | Map<GraphVertex, NDArray> | Record<string, NDArray> | LayoutFunction,
      resolved_scale,
      final_config,
    );
    this._layout = raw instanceof Map
      ? (raw as Map<GraphVertex, NDArray>)
      : new Map<GraphVertex, NDArray>(
          Object.entries(raw as Record<string, NDArray>) as [GraphVertex, NDArray][],
        );

    for (const v of this.vertices.keys()) {
      const position = lookupLayout(this._layout, v);
      if (position === undefined) {
        const error = new Error(String(v));
        error.name = "KeyError";
        throw error;
      }
      this.getItem(v).move_to(position);
    }
    return this;
  }
}

export class Graph extends GenericGraph {
  static _empty_networkx_graph(): NetworkXGraphLike {
    return new NetworkXGraph();
  }

  override _populate_edge_dict(edges: GraphEdge[], edge_type: EdgeConstructor): void {
    for (const [u, v] of edges) {
      const key = edgeKey(u, v);
      const cfg = this._edge_config.get(key) ?? {};
      this.edges.set(
        key,
        new edge_type(
          (this.getItem(u) as Mobject).get_center(),
          (this.getItem(v) as Mobject).get_center(),
          { z_index: -1, ...cfg },
        ),
      );
      this._edge_pairs.set(key, [u, v]);
    }
  }

  override update_edges(graph: GenericGraph): this {
    for (const [key, edge] of graph.edges) {
      const pair = (graph as Graph)._edge_pairs.get(key);
      if (pair === undefined) continue;
      // NOTE: mirrors Python `buff=self._edge_config.get("buff", 0)` /
      // `path_arc=self._edge_config.get("path_arc", 0)` verbatim: _edge_config
      // is keyed by edge tuples, so "buff"/"path_arc" never hit and the
      // updater always uses 0 (an upstream quirk, kept faithfully).
      (edge as Line).set_points_by_ends(
        (graph as Graph).getItem(pair[0]).get_center(),
        (graph as Graph).getItem(pair[1]).get_center(),
        0,
        0,
      );
    }
    return this;
  }

  toString(): string {
    return this.__repr__();
  }

  __repr__(): string {
    return `Undirected graph on ${this.vertices.size} vertices and ${this.edges.size} edges`;
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
      const key = edgeKey(u, v);
      const cfg = this._edge_config.get(key) ?? {};
      const edge = new edge_type(this.getItem(u), this.getItem(v), { z_index: -1, ...cfg });
      this.edges.set(key, edge);
      this._edge_pairs.set(key, [u, v]);
    }
    for (const [u, v] of edges) {
      const key = edgeKey(u, v);
      const edge = this.edges.get(key) as Mobject;
      const tipCfg = this._tip_config.get(key) ?? {};
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

  override update_edges(graph: GenericGraph): this {
    for (const [key, edge] of graph.edges) {
      const pair = (graph as DiGraph)._edge_pairs.get(key);
      if (pair === undefined) continue;
      const tipable = edge as unknown as {
        pop_tips(): VGroup;
        add_tip(tip?: unknown): void;
        set_points_by_ends(start: unknown, end: unknown, buff?: number, path_arc?: number): void;
      };
      const tip = tipable.pop_tips().submobjects[0];
      // NOTE: passing the vertex mobjects (not their centers) makes the tip
      // stop at the vertex boundary; buff/path_arc mirror the Graph quirk
      // above (always 0, like Python `self._edge_config.get(..., 0)`).
      tipable.set_points_by_ends(
        (graph as DiGraph).getItem(pair[0]),
        (graph as DiGraph).getItem(pair[1]),
        0,
        0,
      );
      tipable.add_tip(tip);
    }
    return this;
  }

  toString(): string {
    return this.__repr__();
  }

  __repr__(): string {
    return `Directed graph on ${this.vertices.size} vertices and ${this.edges.size} edges`;
  }
}

// NOTE: mirrors Python's @override_animate(...) decorators on the four
// mutating methods (animation methods are looked up by the animation system;
// see Mobject.override_animate).
override_animate(GenericGraph.prototype.add_vertices)(
  GenericGraph.prototype._add_vertices_animation,
);
override_animate(GenericGraph.prototype.remove_vertices)(
  GenericGraph.prototype._remove_vertices_animation,
);
override_animate(GenericGraph.prototype.add_edges)(
  GenericGraph.prototype._add_edges_animation,
);
override_animate(GenericGraph.prototype.remove_edges)(
  GenericGraph.prototype._remove_edges_animation,
);

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

export type LayoutName = "circular" | "kamada_kawai" | "multipartite" | "partite" | "planar" | "random" | "shell" | "spectral" | "spiral" | "spring" | "tree";

function notPortedLayout(name: string): LayoutFunction {
  return (_graph: NetworkXGraphLike, _scale?: number | number[]) => {
    throw new Error(`TODO: the "${name}" graph layout is not ported yet`);
  };
}

export const _layouts: Record<LayoutName, LayoutFunction> = {
  circular: notPortedLayout("circular"),
  kamada_kawai: notPortedLayout("kamada_kawai"),
  multipartite: notPortedLayout("multipartite"),
  // NOTE: Python names this layout "partite"; the "multipartite" key above is
  // kept for backwards compatibility with the earlier TS port.
  partite: notPortedLayout("partite"),
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
