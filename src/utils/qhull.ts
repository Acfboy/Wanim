import { array, linalg, random, type NDArray } from "numpy-ts";
import { isNDArray, numeric } from "./arrays.js";

export type HullPoints = NDArray | number[][] | readonly (readonly number[])[];

function toRowMatrix(points: HullPoints): number[][] {
  if (isNDArray(points)) {
    const raw = (points as NDArray).toArray() as unknown;
    if (Array.isArray(raw) && raw.length > 0 && Array.isArray(raw[0])) {
      return (raw as unknown[][]).map((row) => row.map(numeric));
    }
    return [];
  }
  return (points as number[][]).map((row) => [...row].map(numeric));
}

function pointKey(coordinates: readonly number[]): string {
  const buffer = new ArrayBuffer(coordinates.length * 8);
  const view = new DataView(buffer);
  coordinates.forEach((value, index) => view.setFloat64(index * 8, value));
  return Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function dot(a: readonly number[], b: readonly number[]): number {
  return a.reduce((sum, value, index) => sum + value * (b[index] ?? 0), 0);
}

function valueError(message: string): Error {
  const error = new Error(message);
  error.name = "ValueError";
  return error;
}

export class QuickHullPoint {
  coordinates: number[];

  constructor(coordinates: number[] | readonly number[] | NDArray) {
    this.coordinates = isNDArray(coordinates)
      ? ((coordinates as NDArray).toArray() as unknown[]).map(numeric)
      : [...(coordinates as readonly number[])].map(numeric);
  }

  key(): string {
    return pointKey(this.coordinates);
  }

  equals(other: unknown): boolean {
    if (!(other instanceof QuickHullPoint)) throw valueError(String(other));
    if (this.coordinates.length !== other.coordinates.length) return false;
    return this.coordinates.every((value, index) => value === other.coordinates[index]);
  }
}

export class SubFacet {
  coordinates: number[][];
  points: Set<string>;

  constructor(coordinates: number[][]) {
    this.coordinates = coordinates.map((row) => [...row]);
    this.points = new Set(this.coordinates.map((row) => new QuickHullPoint(row).key()));
  }

  key(): string {
    return [...this.points].sort().join("|");
  }

  equals(other: unknown): boolean {
    if (!(other instanceof SubFacet)) throw valueError(String(other));
    if (this.points.size !== other.points.size) return false;
    for (const key of this.points) if (!other.points.has(key)) return false;
    return true;
  }
}

export class Facet {
  coordinates: number[][];
  center: number[];
  normal: number[];
  subfacets: SubFacet[];

  constructor(coordinates: number[][], internal: number[] | readonly number[]) {
    this.coordinates = coordinates.map((row) => [...row]);
    const dim = this.coordinates[0]?.length ?? 0;
    this.center = this.coordinates[0]?.map((_, index) => this.coordinates.reduce((sum, row) => sum + (row[index] ?? 0), 0) / this.coordinates.length) ?? [];
    void dim;
    this.normal = this.compute_normal([...internal]);
    this.subfacets = this.coordinates.map((_, index) =>
      new SubFacet(this.coordinates.filter((__, rowIndex) => rowIndex !== index)),
    );
  }

  compute_normal(internal: number[]): number[] {
    const centered = this.coordinates.map((row) => row.map((value, index) => value - (this.center[index] ?? 0)));
    const result = linalg.svd(array(centered) as never) as unknown as { vt: NDArray };
    const rows = ((result.vt as NDArray).toArray() as unknown[][]).map((row) => row.map(numeric));
    let normal = [...(rows[rows.length - 1] ?? this.center.map(() => 0))];
    const length = Math.hypot(...normal);
    normal = normal.map((value) => value / length);
    if (dot(normal, this.center.map((value, index) => value - (internal[index] ?? 0))) < 0) {
      normal = normal.map((value) => value * -1);
    }
    return normal;
  }

  key(): string {
    return this.subfacets.map((subfacet) => subfacet.key()).sort().join("||");
  }

  equals(other: unknown): boolean {
    if (!(other instanceof Facet)) throw valueError(String(other));
    return this.key() === other.key();
  }
}

export class Horizon {
  facets: Facet[] = [];
  facetKeys = new Set<string>();
  boundary: SubFacet[] = [];

  addFacet(facet: Facet): void {
    const key = facet.key();
    if (!this.facetKeys.has(key)) {
      this.facetKeys.add(key);
      this.facets.push(facet);
    }
  }

  hasFacet(facet: Facet): boolean {
    return this.facetKeys.has(facet.key());
  }
}

export class QuickHull {
  facets: Facet[] = [];
  removed = new Set<string>();
  outside = new Map<string, { outside: number[][] | null; eye: number[] | null }>();
  neighbors = new Map<string, Map<string, Facet>>();
  unclaimed: number[][] | null = null;
  internal: number[] | null = null;
  tolerance: number;

  constructor(tolerance = 1e-5) {
    this.tolerance = tolerance;
  }

  initialize(points: HullPoints): void {
    const rows = toRowMatrix(points);
    const count = rows.length;
    const dim = rows[0]?.length ?? 0;
    const rng = random.default_rng();
    const picked = rng.choice(count, dim + 1, false) as unknown;
    const indices: number[] = Array.isArray(picked)
      ? (picked as unknown[]).map((item) => Math.trunc(Number(item)))
      : isNDArray(picked)
        ? ((picked as NDArray).toArray() as unknown[]).map((item) => Math.trunc(Number(item)))
        : [Math.trunc(Number(picked))];
    const simplex = indices.map((index) => [...(rows[index] ?? [])]);
    this.unclaimed = rows.map((row) => [...row]);
    const internal: number[] = simplex[0]?.map((_, index) => simplex.reduce((sum, row) => sum + (row[index] ?? 0), 0) / simplex.length) ?? [];
    this.internal = internal;
    for (let c = 0; c < simplex.length; c += 1) {
      const facet = new Facet(simplex.filter((__, rowIndex) => rowIndex !== c), internal);
      this.classify(facet);
      this.facets.push(facet);
    }
    for (const facet of this.facets) {
      for (const subfacet of facet.subfacets) {
        this.addNeighbor(subfacet, facet);
      }
    }
  }

  classify(facet: Facet): void {
    if (this.unclaimed === null) {
      throw new Error("Call .initialize() before using .classify().");
    }
    if (this.unclaimed.length === 0) {
      this.outside.set(facet.key(), { outside: null, eye: null });
      return;
    }
    const projections = this.unclaimed.map((point) =>
      dot(point.map((value, index) => value - (facet.center[index] ?? 0)), facet.normal),
    );
    let arg = 0;
    for (let index = 1; index < projections.length; index += 1) {
      if ((projections[index] ?? -Infinity) > (projections[arg] ?? -Infinity)) arg = index;
    }
    const mask = projections.map((value) => (value ?? -Infinity) > this.tolerance);
    const eye = (projections[arg] ?? -Infinity) > this.tolerance ? [...(this.unclaimed[arg] ?? [])] : null;
    const outside = this.unclaimed.filter((__, index) => mask[index]);
    this.outside.set(facet.key(), { outside, eye });
    this.unclaimed = this.unclaimed.filter((__, index) => !mask[index]);
  }

  compute_horizon(eye: number[], start_facet: Facet): Horizon {
    const horizon = new Horizon();
    this._recursive_horizon(eye, start_facet, horizon);
    return horizon;
  }

  _recursive_horizon(eye: number[], facet: Facet, horizon: Horizon): boolean {
    const visible = dot(facet.normal, eye.map((value, index) => value - (facet.center[index] ?? 0))) > 0;
    if (!visible) return false;
    horizon.addFacet(facet);
    for (const subfacet of facet.subfacets) {
      const candidates = this.neighbors.get(subfacet.key());
      const neighbor = candidates === undefined
        ? undefined
        : [...candidates.values()].find((candidate) => candidate.key() !== facet.key());
      if (neighbor !== undefined && !horizon.hasFacet(neighbor) && !this._recursive_horizon(eye, neighbor, horizon)) {
        horizon.boundary.push(subfacet);
      }
    }
    return true;
  }

  build(points: HullPoints): void {
    const rows = toRowMatrix(points);
    const num = rows.length;
    const dim = rows[0]?.length ?? 0;
    if (dim === 0 || num < dim + 1) {
      throw valueError("Not enough points supplied to build Convex Hull!");
    }
    if (dim === 1) {
      throw valueError("The Convex Hull of 1D data is its min-max!");
    }
    this.initialize(points);
    if (this.unclaimed === null || this.internal === null) {
      throw new Error("QuickHull failed to initialize.");
    }
    while (true) {
      let updated = false;
      for (const facet of this.facets) {
        if (this.removed.has(facet.key())) continue;
        const entry = this.outside.get(facet.key());
        const eye = entry?.eye ?? null;
        if (eye !== null) {
          updated = true;
          const horizon = this.compute_horizon(eye, facet);
          for (const visible of horizon.facets) {
            const points_to_append = this.outside.get(visible.key())?.outside;
            if (points_to_append === null || points_to_append === undefined) {
              throw new Error("AssertionError: expected outside points for visible facet.");
            }
            this.unclaimed = [...(this.unclaimed ?? []), ...points_to_append.map((row) => [...row])];
            this.removed.add(visible.key());
            for (const subfacet of visible.subfacets) {
              const bucket = this.neighbors.get(subfacet.key());
              if (bucket !== undefined) {
                bucket.delete(visible.key());
                if (bucket.size === 0) this.neighbors.delete(subfacet.key());
              }
            }
          }
          for (const subfacet of horizon.boundary) {
            const fresh = new Facet([...subfacet.coordinates.map((row) => [...row]), [...eye]], this.internal as number[]);
            this.classify(fresh);
            this.facets.push(fresh);
            for (const nsf of fresh.subfacets) this.addNeighbor(nsf, fresh);
          }
        }
      }
      if (!updated) break;
    }
  }

  private addNeighbor(subfacet: SubFacet, facet: Facet): void {
    const key = subfacet.key();
    let bucket = this.neighbors.get(key);
    if (bucket === undefined) {
      bucket = new Map();
      this.neighbors.set(key, bucket);
    }
    bucket.set(facet.key(), facet);
  }
}
