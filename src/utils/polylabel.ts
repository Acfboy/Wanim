import { array, type NDArray } from "numpy-ts";

type Point2D = [number, number];

export class Polygon {
  array: NDArray;
  start: NDArray;
  stop: NDArray;
  diff: NDArray;
  norm: NDArray;
  area = 0;
  centroid: NDArray = array([0, 0]) as NDArray;

  constructor(rings: Array<number[][] | NDArray>) {
    const np_rings: number[][][] = rings.map((ring) =>
      Array.isArray(ring)
        ? (ring as number[][]).map((point) => [Number(point[0]), Number(point[1])])
        : ((ring as NDArray).toArray() as number[][]).map((point) => [
            Number(point[0]),
            Number(point[1]),
          ]),
    );
    const counts = np_rings.map((ring) => ring.length);
    const csum: number[] = [];
    let total = 0;
    for (const count of counts) {
      total += count;
      csum.push(total);
    }
    const flat: Point2D[] = np_rings.flat() as Point2D[];
    this.array = array(flat.length > 0 ? flat : []) as NDArray;

    const last = csum[csum.length - 1] ?? 0;
    const excludeStart = new Set(csum.map((value) => value - 1));
    const excludeStop = new Set(csum.map((value) => value % last));
    const startRows = flat.filter((_, i) => !excludeStart.has(i));
    const stopRows = flat.filter((_, i) => !excludeStop.has(i));
    this.start = array(startRows.length > 0 ? startRows : []) as NDArray;
    this.stop = array(stopRows.length > 0 ? stopRows : []) as NDArray;

    const consecutive: Point2D[] = [];
    for (let i = 0; i < flat.length - 1; i += 1) {
      const a = flat[i] ?? [0, 0];
      const b = flat[i + 1] ?? [0, 0];
      consecutive.push([b[0] - a[0], b[1] - a[1]]);
    }
    const excludeDiff = new Set(csum.slice(0, -1).map((value) => value - 1));
    const diffs = consecutive.filter((_, i) => !excludeDiff.has(i));
    this.diff = array(diffs.length > 0 ? diffs : []) as NDArray;
    const diffRows = diffs;
    const normRows = diffRows.map(([dx, dy]) => {
      const squared = dx * dx + dy * dy;
      return [dx / squared, dy / squared];
    });
    this.norm = array(normRows.length > 0 ? normRows : []) as NDArray;

    const x = startRows.map((row) => row[0] ?? 0);
    const y = startRows.map((row) => row[1] ?? 0);
    const xr = stopRows.map((row) => row[0] ?? 0);
    const yr = stopRows.map((row) => row[1] ?? 0);
    this.area =
      0.5 *
      (x.reduce((acc, value, i) => acc + value * (yr[i] ?? 0), 0) -
        xr.reduce((acc, value, i) => acc + value * (y[i] ?? 0), 0));
    if (this.area) {
      let cxSum = 0;
      let cySum = 0;
      for (let i = 0; i < x.length; i += 1) {
        const factor =
          (x[i] ?? 0) * (yr[i] ?? 0) - (xr[i] ?? 0) * (y[i] ?? 0);
        cxSum += ((x[i] ?? 0) + (xr[i] ?? 0)) * factor;
        cySum += ((y[i] ?? 0) + (yr[i] ?? 0)) * factor;
      }
      this.centroid = array([
        cxSum / (6.0 * this.area),
        cySum / (6.0 * this.area),
      ]) as NDArray;
    }
  }

  compute_distance(point: number[] | NDArray): number {
    const p = Array.isArray(point)
      ? [Number(point[0]), Number(point[1])]
      : ((point as NDArray).toArray() as number[]).map(Number).slice(0, 2);
    const startRows = this.start.toArray() as number[][];
    const diffRows = this.diff.toArray() as number[][];
    const normRows = this.norm.toArray() as number[][];
    let minDist = Infinity;
    for (let i = 0; i < startRows.length; i += 1) {
      const s = startRows[i] ?? [0, 0];
      const d = diffRows[i] ?? [0, 0];
      const n = normRows[i] ?? [0, 0];
      const rel = [p[0] - (s[0] ?? 0), p[1] - (s[1] ?? 0)];
      const scalar = (n[0] ?? 0) * rel[0] + (n[1] ?? 0) * rel[1];
      const clip = Math.min(1, Math.max(0, scalar));
      const qx = (s[0] ?? 0) + (d[0] ?? 0) * clip - p[0];
      const qy = (s[1] ?? 0) + (d[1] ?? 0) * clip - p[1];
      const dist = Math.hypot(qx, qy);
      if (dist < minDist) minDist = dist;
    }
    const d = minDist === Infinity ? 0 : minDist;
    return this.inside(p) ? d : -d;
  }

  _is_point_on_segment(
    x_point: number,
    y_point: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ): boolean {
    if (
      Math.min(x0, x1) <= x_point &&
      x_point <= Math.max(x0, x1) &&
      Math.min(y0, y1) <= y_point &&
      y_point <= Math.max(y0, y1)
    ) {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const cross = dx * (y_point - y0) - dy * (x_point - x0);
      return Boolean(isClose(cross, 0.0));
    }
    return false;
  }

  _ray_crosses_segment(
    x_point: number,
    y_point: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ): boolean {
    if ((y0 > y_point) !== (y1 > y_point)) {
      const slope = (x1 - x0) / (y1 - y0);
      const x_intersect = slope * (y_point - y0) + x0;
      return Boolean(x_point < x_intersect);
    }
    return false;
  }

  inside(point: number[] | NDArray): boolean {
    const p = Array.isArray(point)
      ? [Number(point[0]), Number(point[1])]
      : ((point as NDArray).toArray() as number[]).map(Number).slice(0, 2);
    const [point_x, point_y] = p;
    const startRows = this.start.toArray() as number[][];
    const stopRows = this.stop.toArray() as number[][];
    const segment_count = startRows.length;

    for (let i = 0; i < segment_count; i += 1) {
      if (
        this._is_point_on_segment(
          point_x,
          point_y,
          startRows[i]?.[0] ?? 0,
          startRows[i]?.[1] ?? 0,
          stopRows[i]?.[0] ?? 0,
          stopRows[i]?.[1] ?? 0,
        )
      ) {
        return true;
      }
    }

    let crossings = 0;
    for (let i = 0; i < segment_count; i += 1) {
      if (
        this._ray_crosses_segment(
          point_x,
          point_y,
          startRows[i]?.[0] ?? 0,
          startRows[i]?.[1] ?? 0,
          stopRows[i]?.[0] ?? 0,
          stopRows[i]?.[1] ?? 0,
        )
      ) {
        crossings += 1;
      }
    }

    return crossings % 2 === 1;
  }
}

function isClose(a: number, b: number, rtol = 1e-05, atol = 1e-08): boolean {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

export class Cell {
  c: NDArray;
  h: number;
  d: number;
  p: number;

  constructor(c: number[] | NDArray, h: number, polygon: Polygon) {
    this.c = Array.isArray(c) ? (array([...c.map(Number)]) as NDArray) : c;
    this.h = h;
    this.d = polygon.compute_distance(this.c);
    this.p = this.d + this.h * Math.sqrt(2);
  }

  lessThan(other: Cell): boolean {
    return this.d < other.d;
  }

  greaterThan(other: Cell): boolean {
    return this.d > other.d;
  }

  lessThanOrEqual(other: Cell): boolean {
    return this.d <= other.d;
  }

  greaterThanOrEqual(other: Cell): boolean {
    return this.d >= other.d;
  }

  valueOf(): number {
    return this.d;
  }
}
