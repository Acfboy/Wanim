/**
 * WebGL2 backend for `ThreeDCamera` frames.
 *
 * The Canvas2D backend projects every point, resolves every face's shading and
 * builds one path per face on the CPU, then asks the rasterizer to fill and
 * stroke each face separately — that caps a 24x48 sphere at a few FPS.  This
 * backend keeps the cairo semantics but moves the work to the GPU:
 *
 *  - Geometry is uploaded in world space once per shape change.  The vertex
 *    shader applies exactly `ThreeDCamera.project_points` (rotation matrix,
 *    linear/exponential perspective, zoom) and then the same world->device
 *    transform as `setCanvasTransform`.
 *  - Fills are triangulated on the CPU (ear clipping) and stroked with
 *    screen-space miter joins computed in the vertex shader, so stroke width
 *    matches `cairo_line_width_multiple` behaviour at any depth.
 *  - Colours come from `ThreeDCamera.get_draw_color_info`, which reproduces
 *    `modified_rgbas` + `set_cairo_context_color`: a flat colour for uniform
 *    rows, otherwise a two-stop linear gradient between the VMobject's gradient
 *    endpoints, evaluated per fragment exactly like `cairo.LinearGradient`.
 *  - Painter's order matches cairo: the sorted display list is replayed in
 *    order (per mobject: background stroke, fill, foreground stroke), with the
 *    triangles of every mobject appended to one element buffer and drawn in a
 *    single `drawElements` call (WebGL preserves primitive order).
 *
 * Mobjects this backend cannot express (multiple fill subpaths, >2 colour
 * stops, fixed-in-frame/fixed-orientation members, background images) return
 * `null` from {@link WebGLThreeDRenderer.ensure_record} and are drawn by the
 * Canvas2D fallback instead.
 */
import type { NDArray } from "numpy-ts";
import { _cContiguousRows } from "../utils/arrays.js";
import { VMobject } from "../mobject/types/vectorized_mobject.js";
import { ThreeDCamera, type DrawColorInfo, type MobjectColorInfos } from "../camera/three_d_camera.js";
import { computeSubpathRangesFlat } from "./canvas.js";
import { mobject_geometry_epoch } from "../utils/render_epoch.js";

/** Interleaved vertex layout (floats): position(3) prev(3) next(3)
 *  grad0(3) grad1(3) side(1) halfWidth(1) color0(4) color1(4) */
const FLOATS_PER_VERTEX = 25;
const POSITION_OFFSET = 0;
const PREV_OFFSET = 3;
const NEXT_OFFSET = 6;
const GRAD0_OFFSET = 9;
const GRAD1_OFFSET = 12;
const SIDE_OFFSET = 15;
const HALF_WIDTH_OFFSET = 16;
const COLOR0_OFFSET = 17;
const COLOR1_OFFSET = 21;

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_prev;
layout(location = 2) in vec3 a_next;
layout(location = 3) in vec3 a_grad0;
layout(location = 4) in vec3 a_grad1;
layout(location = 5) in float a_side;
layout(location = 8) in float a_half_width;
layout(location = 6) in vec4 a_color0;
layout(location = 7) in vec4 a_color1;

uniform mat3 u_rot;          // rotation matrix rows as columns
uniform vec3 u_center;
uniform float u_focal;
uniform float u_zoom;
uniform bool u_exponential;
uniform vec4 u_screen;       // widthMult, heightMult, addX, addY
uniform vec2 u_viewport;     // pixel width / height

out float v_t;
out vec4 v_color0;
out vec4 v_color1;

vec2 project(vec3 world) {
  vec3 p = world - u_center;
  vec3 r = vec3(dot(u_rot[0], p), dot(u_rot[1], p), dot(u_rot[2], p));
  float factor;
  if (u_exponential) {
    factor = r.z < 0.0 ? u_focal / (u_focal - r.z) : exp(r.z / u_focal);
  } else {
    factor = u_focal / (u_focal - r.z);
    if (u_focal - r.z < 0.0) factor = 1e6;
  }
  vec2 proj = r.xy * factor * u_zoom;
  return vec2(proj.x * u_screen.x + u_screen.z, proj.y * (-u_screen.y) + u_screen.w);
}

void main() {
  vec2 cur = project(a_position);
  vec2 pos = cur;
  if (a_side != 0.0) {
    vec2 prev = project(a_prev);
    vec2 next = project(a_next);
    vec2 d1 = cur - prev;
    vec2 d2 = next - cur;
    float l1 = length(d1);
    float l2 = length(d2);
    vec2 n1 = l1 > 0.0 ? vec2(-d1.y, d1.x) / l1 : vec2(0.0);
    vec2 n2 = l2 > 0.0 ? vec2(-d2.y, d2.x) / l2 : vec2(0.0);
    vec2 m = n1 + n2;
    float lm = length(m);
    vec2 dir;
    if (lm < 1e-6) {
      dir = l1 > 0.0 ? n1 : n2;
    } else {
      m /= lm;
      float denom = dot(m, n1);
      // Cairo's default miter limit is 10; clamp there instead of beveling.
      dir = denom > 1e-4 ? m * min(1.0 / denom, 10.0) : n1;
    }
    pos = cur + dir * (a_half_width * a_side);
  }
  vec2 g0 = project(a_grad0);
  vec2 g1 = project(a_grad1);
  vec2 seg = g1 - g0;
  float len2 = dot(seg, seg);
  v_t = len2 > 0.0 ? dot(pos - g0, seg) / len2 : 0.0;
  v_color0 = a_color0;
  v_color1 = a_color1;
  gl_Position = vec4(
    pos.x / u_viewport.x * 2.0 - 1.0,
    1.0 - pos.y / u_viewport.y * 2.0,
    0.0,
    1.0
  );
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float v_t;
in vec4 v_color0;
in vec4 v_color1;
out vec4 outColor;
void main() {
  outColor = mix(v_color0, v_color1, clamp(v_t, 0.0, 1.0));
}
`;

interface GeometryRecord {
  points: unknown;
  /** The camera's combined fill/stroke/bg-stroke colour infos. */
  colorInfo: MobjectColorInfos;
  strokeWidthRaw: unknown;
  bgStrokeWidthRaw: unknown;
  jointType: string;
  capStyle: string;
  lineMultiple: number;
  zoomBucket: number;
  vertexData: Float32Array;
  /** Triangle indices, already offset by `vertexOffset`. */
  indexData: Uint32Array;
  vertexCount: number;
  /** Start of this record's vertices inside the shared vertex heap (floats). */
  vertexOffset: number;
}

export interface WebGLThreeDRendererOptions {
  antialias?: boolean;
}

function isColorTransparent(info: DrawColorInfo): boolean {
  if (info.gradient && info.colors.length >= 2) {
    return (info.colors[0]?.[3] ?? 0) === 0 && (info.colors[1]?.[3] ?? 0) === 0;
  }
  return (info.colors[0]?.[3] ?? 0) === 0;
}

function flattenCubic(
  out: number[][],
  p0: number[],
  p1: number[],
  p2: number[],
  p3: number[],
  tolerance: number,
): void {
  // Straight segments (the overwhelmingly common case for surfaces) collapse
  // to their endpoints; `set_points_as_corners` produces exactly collinear
  // handles.
  const dx = (p3[0] ?? 0) - (p0[0] ?? 0);
  const dy = (p3[1] ?? 0) - (p0[1] ?? 0);
  const dz = (p3[2] ?? 0) - (p0[2] ?? 0);
  const chord2 = dx * dx + dy * dy + dz * dz;
  const cross = (p: number[]): number => {
    const ax = (p[0] ?? 0) - (p0[0] ?? 0);
    const ay = (p[1] ?? 0) - (p0[1] ?? 0);
    const az = (p[2] ?? 0) - (p0[2] ?? 0);
    const cx = dy * az - dz * ay;
    const cy = dz * ax - dx * az;
    const cz = dx * ay - dy * ax;
    return Math.sqrt(cx * cx + cy * cy + cz * cz);
  };
  const d1 = cross(p1);
  const d2 = cross(p2);
  if (d1 * d1 <= tolerance * tolerance * Math.max(chord2, 1e-30) && d2 * d2 <= tolerance * tolerance * Math.max(chord2, 1e-30)) {
    out.push(p3);
    return;
  }
  // de Casteljau split at 0.5.
  const m01 = mid(p0, p1);
  const m12 = mid(p1, p2);
  const m23 = mid(p2, p3);
  const m012 = mid(m01, m12);
  const m123 = mid(m12, m23);
  const m = mid(m012, m123);
  flattenCubic(out, p0, m01, m012, m, tolerance);
  flattenCubic(out, m, m123, m23, p3, tolerance);
}

function mid(a: number[], b: number[]): number[] {
  return [
    ((a[0] ?? 0) + (b[0] ?? 0)) / 2,
    ((a[1] ?? 0) + (b[1] ?? 0)) / 2,
    ((a[2] ?? 0) + (b[2] ?? 0)) / 2,
  ];
}

function pointsEqual(a: number[], b: number[], atol: number): boolean {
  const rtol = 1e-5;
  for (let dim = 0; dim < 3; dim += 1) {
    const e = a[dim] ?? NaN;
    const s = b[dim] ?? NaN;
    const diff = Math.abs(e - s);
    if (!((diff <= atol + rtol * Math.abs(s) && Number.isFinite(diff)) || e === s)) return false;
  }
  return true;
}

/**
 * Ear-clipping triangulation of a simple (possibly non-planar-ish) 3D polygon.
 *
 * The polygon is projected onto its dominant plane (chosen from the Newell
 * normal) before clipping: a cube face parallel to the world XY plane has zero
 * area there, so the previous world-XY projection produced no triangles and
 * dropped the face entirely.
 */
function triangulatePolygon(poly: number[][]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  let nx = 0;
  let ny = 0;
  let nz = 0;
  for (let i = 0; i < n; i += 1) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    nx += ((a[1] ?? 0) - (b[1] ?? 0)) * ((a[2] ?? 0) + (b[2] ?? 0));
    ny += ((a[2] ?? 0) - (b[2] ?? 0)) * ((a[0] ?? 0) + (b[0] ?? 0));
    nz += ((a[0] ?? 0) - (b[0] ?? 0)) * ((a[1] ?? 0) + (b[1] ?? 0));
  }
  const ax = Math.abs(nx);
  const ay = Math.abs(ny);
  const az = Math.abs(nz);
  // Which 3D axes map to the 2D (u, v) plane: drop the dominant normal axis.
  let uAxis: number;
  let vAxis: number;
  if (ax >= ay && ax >= az) { uAxis = 1; vAxis = 2; }
  else if (ay >= az) { uAxis = 2; vAxis = 0; }
  else { uAxis = 0; vAxis = 1; }
  const flat: number[][] = poly.map((p) => [p[uAxis] ?? 0, p[vAxis] ?? 0]);
  return triangulatePolygon2D(flat);
}

/** Ear clipping in 2D (any orientation). */
function triangulatePolygon2D(poly: number[][]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  const indices: number[] = [];
  let area = 0;
  for (let i = 0; i < n; i += 1) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    area += (a[0] ?? 0) * (b[1] ?? 0) - (b[0] ?? 0) * (a[1] ?? 0);
  }
  if (area === 0) return [];
  const ccw = area > 0;
  const remaining: number[] = [];
  for (let i = 0; i < n; i += 1) remaining.push(ccw ? i : n - 1 - i);
  let guard = n * n + 16;
  while (remaining.length > 3 && guard > 0) {
    guard -= 1;
    let clipped = false;
    for (let k = 0; k < remaining.length; k += 1) {
      const i0 = remaining[(k + remaining.length - 1) % remaining.length]!;
      const i1 = remaining[k]!;
      const i2 = remaining[(k + 1) % remaining.length]!;
      const a = poly[i0]!;
      const b = poly[i1]!;
      const c = poly[i2]!;
      const cross =
        ((b[0] ?? 0) - (a[0] ?? 0)) * ((c[1] ?? 0) - (a[1] ?? 0)) -
        ((b[1] ?? 0) - (a[1] ?? 0)) * ((c[0] ?? 0) - (a[0] ?? 0));
      if (cross <= 0) continue;
      let contains = false;
      for (const other of remaining) {
        if (other === i0 || other === i1 || other === i2) continue;
        const p = poly[other]!;
        const c1 =
          ((b[0] ?? 0) - (a[0] ?? 0)) * ((p[1] ?? 0) - (a[1] ?? 0)) -
          ((b[1] ?? 0) - (a[1] ?? 0)) * ((p[0] ?? 0) - (a[0] ?? 0));
        const c2 =
          ((c[0] ?? 0) - (b[0] ?? 0)) * ((p[1] ?? 0) - (b[1] ?? 0)) -
          ((c[1] ?? 0) - (b[1] ?? 0)) * ((p[0] ?? 0) - (b[0] ?? 0));
        const c3 =
          ((a[0] ?? 0) - (c[0] ?? 0)) * ((p[1] ?? 0) - (c[1] ?? 0)) -
          ((a[1] ?? 0) - (c[1] ?? 0)) * ((p[0] ?? 0) - (c[0] ?? 0));
        if (c1 >= 0 && c2 >= 0 && c3 >= 0) {
          contains = true;
          break;
        }
      }
      if (contains) continue;
      indices.push(i0, i1, i2);
      remaining.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  if (remaining.length === 3) indices.push(remaining[0]!, remaining[1]!, remaining[2]!);
  return indices;
}

export class WebGLThreeDRenderer {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;
  private ebo: WebGLBuffer | null = null;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private records = new Map<VMobject, GeometryRecord>();
  /** CPU staging copy of the shared vertex heap (offsets survive growth). */
  private vertexHeap = new Float32Array(4096 * FLOATS_PER_VERTEX);
  private vertexHeapUsed = 0;
  /** GPU vertex buffer capacity, in floats. */
  private vertexCapacity = 0;
  /** Freed ranges inside the heap, `{ offset, size }` in floats. */
  private freeRanges: Array<{ offset: number; size: number }> = [];
  private indexScratch = new Uint32Array(8192);
  private indexCount = 0;
  /** GPU element buffer capacity, in indices. */
  private indexCapacity = 0;
  private width = 0;
  private height = 0;
  /**
   * When the global geometry epoch and the camera's record-relevant state are
   * unchanged from the previous frame, every cached record is still valid and
   * `prepare` can skip per-face validation entirely (camera-only movement).
   */
  private recordsValid = false;
  private frameState = {
    epoch: -1,
    lineMultiple: NaN,
    zoomBucket: NaN,
    shading: false,
    fif: -1,
    fio: -1,
    frameHeight: NaN,
    pixelHeight: NaN,
  };

  constructor(options: WebGLThreeDRendererOptions = {}) {
    try {
      const canvas = typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(1, 1)
        : typeof document !== "undefined"
          ? document.createElement("canvas")
          : null;
      if (canvas === null) return;
      const gl = (canvas as HTMLCanvasElement).getContext("webgl2", {
        alpha: true,
        antialias: options.antialias ?? true,
        // Keep the default (premultiplied) framebuffer: blending with
        // SRC_ALPHA leaves premultiplied values in the buffer, which is what
        // `drawImage` expects when compositing the offscreen canvas.
        premultipliedAlpha: true,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: false,
      }) as WebGL2RenderingContext | null;
      if (gl === null) return;
      this.canvas = canvas;
      this.gl = gl;
      this.init(gl);
    } catch {
      this.gl = null;
    }
  }

  available(): boolean {
    return this.gl !== null && this.program !== null && this.vbo !== null && this.ebo !== null && this.vao !== null;
  }

  private init(gl: WebGL2RenderingContext): void {
    const compile = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (shader === null) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("wanim webgl shader", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };
    const vs = compile(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    if (vs === null || fs === null) return;
    const program = gl.createProgram();
    if (program === null) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("wanim webgl link", gl.getProgramInfoLog(program));
      return;
    }
    this.program = program;
    gl.useProgram(program);
    for (const name of ["u_rot", "u_center", "u_focal", "u_zoom", "u_exponential", "u_screen", "u_viewport"]) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    const ebo = gl.createBuffer();
    if (vao === null || vbo === null || ebo === null) return;
    this.vao = vao;
    this.vbo = vbo;
    this.ebo = ebo;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const stride = FLOATS_PER_VERTEX * 4;
    const attribs: Array<[number, number, number, number]> = [
      // location, size, offsetFloats, divisor(0)
      [0, 3, POSITION_OFFSET, 0],
      [1, 3, PREV_OFFSET, 0],
      [2, 3, NEXT_OFFSET, 0],
      [3, 3, GRAD0_OFFSET, 0],
      [4, 3, GRAD1_OFFSET, 0],
      [5, 1, SIDE_OFFSET, 0],
      [8, 1, HALF_WIDTH_OFFSET, 0],
      [6, 4, COLOR0_OFFSET, 0],
      [7, 4, COLOR1_OFFSET, 0],
    ];
    for (const [location, size, offsetFloats] of attribs) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offsetFloats * 4);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  /** Resize/clear the offscreen surface and reset the per-frame batches. */
  begin(camera: ThreeDCamera): void {
    const gl = this.gl;
    if (gl === null || this.canvas === null) return;
    const width = camera.pixel_width;
    const height = camera.pixel_height;
    if (this.width !== width || this.height !== height) {
      this.width = width;
      this.height = height;
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.indexCount = 0;
    const state = this.camera_state(camera);
    this.recordsValid =
      state.epoch === this.frameState.epoch &&
      state.lineMultiple === this.frameState.lineMultiple &&
      state.zoomBucket === this.frameState.zoomBucket &&
      state.shading === this.frameState.shading &&
      state.fif === this.frameState.fif &&
      state.fio === this.frameState.fio &&
      state.frameHeight === this.frameState.frameHeight &&
      state.pixelHeight === this.frameState.pixelHeight;
    this.frameState = state;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private camera_state(camera: ThreeDCamera): WebGLThreeDRenderer["frameState"] {
    const zoom = camera.get_zoom();
    return {
      epoch: mobject_geometry_epoch(),
      lineMultiple: camera.cairo_line_width_multiple,
      zoomBucket: Math.round(Math.log2(Math.max(1, zoom)) * 2),
      shading: camera.should_apply_shading,
      fif: camera.fixed_in_frame_mobjects.size,
      fio: camera.fixed_orientation_mobjects.size,
      frameHeight: camera.frame_height,
      pixelHeight: camera.pixel_height,
    };
  }

  /**
   * Build (or validate from cache) the geometry record of a mobject.  Returns
   * `null` when this backend cannot draw it; the caller then falls back to
   * Canvas2D for the frame.
   */
  prepare(mobject: VMobject, camera: ThreeDCamera): GeometryRecord | null {
    if (this.recordsValid) {
      const existing = this.records.get(mobject);
      if (existing !== undefined) return existing;
    }
    return this.ensure_record(mobject, camera);
  }

  /** Append a prepared record to the current frame batch. */
  push(record: GeometryRecord): void {
    if (record.indexData.length === 0) return;
    const needed = this.indexCount + record.indexData.length;
    this.ensure_index_capacity(needed);
    this.indexScratch.set(record.indexData, this.indexCount);
    this.indexCount = needed;
  }

  /** Draw the accumulated batch (one `drawElements`, painter order preserved). */
  flush(camera: ThreeDCamera): void {
    const gl = this.gl;
    if (gl === null || this.program === null || this.vao === null || this.vbo === null || this.ebo === null) return;
    if (this.indexCount === 0) return;
    const pixelWidth = camera.pixel_width;
    const pixelHeight = camera.pixel_height;
    const frameWidth = camera.frame_width;
    const frameHeight = camera.frame_height;
    const widthMult = pixelWidth / frameWidth;
    const heightMult = pixelHeight / frameHeight;
    const frameCenter = camera.frame_center;
    const [r0, r1, r2] = camera.get_rotation_rows();
    gl.useProgram(this.program);
    gl.uniformMatrix3fv(this.uniforms["u_rot"] ?? null, false, new Float32Array([
      r0[0] ?? 0, r0[1] ?? 0, r0[2] ?? 0,
      r1[0] ?? 0, r1[1] ?? 0, r1[2] ?? 0,
      r2[0] ?? 0, r2[1] ?? 0, r2[2] ?? 0,
    ]));
    gl.uniform3f(
      this.uniforms["u_center"] ?? null,
      frameCenter[0] ?? 0,
      frameCenter[1] ?? 0,
      frameCenter[2] ?? 0,
    );
    gl.uniform1f(this.uniforms["u_focal"] ?? null, camera.get_focal_distance());
    gl.uniform1f(this.uniforms["u_zoom"] ?? null, camera.get_zoom());
    gl.uniform1i(this.uniforms["u_exponential"] ?? null, camera.exponential_projection ? 1 : 0);
    gl.uniform4f(
      this.uniforms["u_screen"] ?? null,
      widthMult,
      heightMult,
      pixelWidth / 2 - (frameCenter[0] ?? 0) * widthMult,
      pixelHeight / 2 + (frameCenter[1] ?? 0) * heightMult,
    );
    gl.uniform2f(this.uniforms["u_viewport"] ?? null, pixelWidth, pixelHeight);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, this.indexScratch.subarray(0, this.indexCount));
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  /** Upload a record's vertices into the shared heap (once per shape change). */
  private alloc_vertices(record: GeometryRecord): void {
    const gl = this.gl;
    const count = record.vertexData.length;
    if (count === 0) {
      record.vertexOffset = 0;
      return;
    }
    let offset = -1;
    for (let index = 0; index < this.freeRanges.length; index += 1) {
      const range = this.freeRanges[index]!;
      if (range.size >= count) {
        offset = range.offset;
        if (range.size === count) this.freeRanges.splice(index, 1);
        else {
          range.offset += count;
          range.size -= count;
        }
        break;
      }
    }
    if (offset < 0) {
      if (this.freeRanges.length > 0 && this.vertexHeapUsed + count > this.vertexHeap.length) {
        this.compact_vertices();
      }
      offset = this.vertexHeapUsed;
      this.vertexHeapUsed += count;
      this.ensure_vertex_capacity(this.vertexHeapUsed);
    }
    if (offset + count > this.vertexHeap.length) {
      console.error("alloc OOB", { offset, count, heap: this.vertexHeap.length, used: this.vertexHeapUsed, free: this.freeRanges.slice(0, 5), vc: record.vertexCount });
    }
    this.vertexHeap.set(record.vertexData, offset);
    record.vertexOffset = offset;
    // Global indices: `indexData` is local to the record right after build.
    // Indices address vertices, so the float offset must be divided by the
    // vertex stride.
    const vertexBase = offset / FLOATS_PER_VERTEX;
    const indexData = record.indexData;
    if (indexData.length > 0) {
      for (let index = 0; index < indexData.length; index += 1) indexData[index] = indexData[index]! + vertexBase;
    }
    if (gl !== null && this.vbo !== null) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, offset * 4, record.vertexData);
    }
  }

  private free_vertices(record: GeometryRecord): void {
    const count = record.vertexCount * FLOATS_PER_VERTEX;
    if (count > 0) this.freeRanges.push({ offset: record.vertexOffset, size: count });
  }

  /** Repack all live records, removing fragmentation, and re-upload. */
  private compact_vertices(): void {
    const gl = this.gl;
    const live = [...this.records.values()]
      .filter((record) => record.vertexCount > 0)
      .sort((a, b) => a.vertexOffset - b.vertexOffset);
    const heap = new Float32Array(Math.max(this.vertexHeap.length, 4096 * FLOATS_PER_VERTEX));
    let offset = 0;
    for (const record of live) {
      const count = record.vertexCount * FLOATS_PER_VERTEX;
      const deltaVertices = (offset - record.vertexOffset) / FLOATS_PER_VERTEX;
      heap.set(this.vertexHeap.subarray(record.vertexOffset, record.vertexOffset + count), offset);
      record.vertexOffset = offset;
      if (deltaVertices !== 0) {
        for (let index = 0; index < record.indexData.length; index += 1) {
          record.indexData[index] = record.indexData[index]! + deltaVertices;
        }
      }
      offset += count;
    }
    this.vertexHeap = heap;
    this.vertexHeapUsed = offset;
    this.freeRanges = [];
    if (gl !== null && this.vbo !== null) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, heap, 0, offset);
    }
  }

  private ensure_vertex_capacity(floats: number): void {
    const gl = this.gl;
    if (floats > this.vertexHeap.length) {
      let size = this.vertexHeap.length;
      while (size < floats) size *= 2;
      const heap = new Float32Array(size);
      heap.set(this.vertexHeap);
      this.vertexHeap = heap;
    }
    if (gl === null || this.vbo === null) return;
    if (floats <= this.vertexCapacity) return;
    let capacity = Math.max(this.vertexCapacity * 2, floats, 4096 * FLOATS_PER_VERTEX);
    if (capacity < this.vertexHeap.length) capacity = this.vertexHeap.length;
    this.vertexCapacity = capacity;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertexHeap, 0, this.vertexHeapUsed);
  }

  private ensure_index_capacity(indices: number): void {
    const gl = this.gl;
    if (indices > this.indexScratch.length) {
      let size = this.indexScratch.length;
      while (size < indices) size *= 2;
      const scratch = new Uint32Array(size);
      scratch.set(this.indexScratch);
      this.indexScratch = scratch;
    }
    if (gl === null || this.ebo === null) return;
    if (indices <= this.indexCapacity) return;
    this.indexCapacity = Math.max(this.indexCapacity * 2, indices, 8192);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indexCapacity * 4, gl.DYNAMIC_DRAW);
  }

  private ensure_record(vmobject: VMobject, camera: ThreeDCamera): GeometryRecord | null {
    const vm = vmobject;
    if (vm.shade_in_3d !== true) return null;
    if (camera.fixed_in_frame_mobjects.has(vm) || camera.fixed_orientation_mobjects.has(vm)) return null;
    if (vm.background_image !== null && vm.background_image !== undefined) return null;
    const points = vm.points;
    if (points === undefined || points === null || (points.shape?.[0] ?? 0) === 0) return null;
    // One combined lookup validates points/rgba sources/light for all three
    // colour slots; the remaining fields are read directly.
    const colorInfo = camera.get_mobject_color_infos(vm);
    const zoom = camera.get_zoom();
    const zoomBucket = Math.round(Math.log2(Math.max(1, zoom)) * 2);
    const lineMultiple = camera.cairo_line_width_multiple;
    const existing = this.records.get(vm);
    if (
      existing !== undefined &&
      existing.points === points &&
      existing.colorInfo === colorInfo &&
      existing.strokeWidthRaw === vm.stroke_width &&
      existing.bgStrokeWidthRaw === vm.background_stroke_width &&
      existing.jointType === vm.joint_type &&
      existing.capStyle === vm.cap_style &&
      existing.lineMultiple === lineMultiple &&
      existing.zoomBucket === zoomBucket
    ) {
      return existing;
    }
    const record = this.build_record(vm, camera, colorInfo, zoom, zoomBucket, lineMultiple);
    if (record === null) return null;
    if (existing !== undefined) this.free_vertices(existing);
    this.alloc_vertices(record);
    this.records.set(vm, record);
    return record;
  }

  private build_record(
    vm: VMobject,
    camera: ThreeDCamera,
    colorInfo: GeometryRecord["colorInfo"],
    zoom: number,
    zoomBucket: number,
    lineMultiple: number,
  ): GeometryRecord | null {
    const pts = _cContiguousRows(vm.points);
    if (pts === null || pts.rowDim < 3 || pts.rowCount < 4) return null;
    const fillInfo = colorInfo.fill;
    const strokeInfo = colorInfo.stroke;
    const bgStrokeInfo = colorInfo.bgStroke;
    const strokeWidth = vm.get_stroke_width(false);
    const bgStrokeWidth = vm.get_stroke_width(true);
    const drawFill = fillInfo !== null && !isColorTransparent(fillInfo);
    const drawStroke = strokeWidth > 0 && strokeInfo !== null && !isColorTransparent(strokeInfo);
    const drawBgStroke = bgStrokeWidth > 0 && bgStrokeInfo !== null && !isColorTransparent(bgStrokeInfo);
    const base = {
      points: vm.points,
      colorInfo,
      strokeWidthRaw: vm.stroke_width,
      bgStrokeWidthRaw: vm.background_stroke_width,
      jointType: vm.joint_type,
      capStyle: vm.cap_style,
      lineMultiple,
      zoomBucket,
    };
    if (!drawFill && !drawStroke && !drawBgStroke) {
      // Nothing visible: an empty record still lets the mobject be "handled".
      return {
        ...base,
        vertexData: new Float32Array(0),
        indexData: new Uint32Array(0),
        vertexCount: 0,
        vertexOffset: 0,
      };
    }
    // Flattening tolerance: a quarter device pixel at the current zoom.
    const tolerance = Math.max(camera.frame_height / camera.pixel_height, 1e-9) * 0.25 / Math.max(1, zoom);
    const ranges = computeSubpathRangesFlat(
      pts.data,
      pts.rowDim,
      pts.rowCount,
      vm.n_points_per_cubic_curve,
      vm.tolerance_for_point_equality,
      2,
    );
    if (ranges.length === 0) return null;
    const subpaths: number[][][] = [];
    for (const [start, end] of ranges) {
      const polyline = this.flatten_subpath(pts.data, pts.rowDim, start!, end!, tolerance);
      if (polyline.length >= 2) subpaths.push(polyline);
    }
    if (subpaths.length === 0) return null;
    if (drawFill && subpaths.length > 1) return null;

    const fillIndices: number[] = [];
    let fillPolygon: number[][] | null = null;
    if (drawFill) {
      const poly = subpaths[0]!;
      const closed = pointsEqual(poly[0]!, poly[poly.length - 1]!, vm.tolerance_for_point_equality);
      const body = closed ? poly.slice(0, -1) : poly;
      if (body.length >= 3) {
        const tris = triangulatePolygon(body);
        if (tris.length > 0) {
          fillPolygon = body;
          fillIndices.push(...tris);
        }
      }
    }

    // Device-space stroke half-widths (see `apply_stroke`: width *
    // cairo_line_width_multiple, then the world->device x scale).
    const deviceScale = camera.pixel_width / camera.frame_width;
    const halfWidth = (strokeWidth * lineMultiple * deviceScale) / 2;
    const bgHalfWidth = (bgStrokeWidth * lineMultiple * deviceScale) / 2;

    const vertices: number[] = [];
    const indices: number[] = [];
    const addVertex = (
      p: number[],
      prev: number[],
      next: number[],
      side: number,
      vertexHalfWidth: number,
      info: DrawColorInfo | null,
    ): number => {
      const index = vertices.length / FLOATS_PER_VERTEX;
      const c0 = info?.colors[0] ?? [0, 0, 0, 0];
      const c1 = info?.colors[1] ?? c0;
      vertices.push(
        p[0] ?? 0, p[1] ?? 0, p[2] ?? 0,
        prev[0] ?? 0, prev[1] ?? 0, prev[2] ?? 0,
        next[0] ?? 0, next[1] ?? 0, next[2] ?? 0,
        info?.start[0] ?? 0, info?.start[1] ?? 0, info?.start[2] ?? 0,
        info?.end[0] ?? 0, info?.end[1] ?? 0, info?.end[2] ?? 0,
        side,
        vertexHalfWidth,
        c0[0] ?? 0, c0[1] ?? 0, c0[2] ?? 0, c0[3] ?? 0,
        c1[0] ?? 0, c1[1] ?? 0, c1[2] ?? 0, c1[3] ?? 0,
      );
      return index;
    };

    // Cairo order per mobject: background stroke, fill, foreground stroke.
    if (drawBgStroke) {
      this.append_stroke_geometry(subpaths, addVertex, indices, bgStrokeInfo, bgHalfWidth);
    }
    if (fillPolygon !== null && fillIndices.length > 0) {
      const vertexBase = vertices.length / FLOATS_PER_VERTEX;
      for (const p of fillPolygon) addVertex(p, p, p, 0, 0, fillInfo);
      for (const tri of fillIndices) indices.push(vertexBase + tri);
    }
    if (drawStroke) {
      this.append_stroke_geometry(subpaths, addVertex, indices, strokeInfo, halfWidth);
    }

    return {
      ...base,
      vertexData: new Float32Array(vertices),
      indexData: new Uint32Array(indices),
      vertexCount: vertices.length / FLOATS_PER_VERTEX,
      vertexOffset: 0,
    };
  }

  private append_stroke_geometry(
    subpaths: number[][][],
    addVertex: (
      p: number[],
      prev: number[],
      next: number[],
      side: number,
      halfWidth: number,
      info: DrawColorInfo | null,
    ) => number,
    indices: number[],
    info: DrawColorInfo | null,
    halfWidth: number,
  ): void {
    for (const poly of subpaths) {
      const n = poly.length;
      if (n < 2) continue;
      const closed = pointsEqual(poly[0]!, poly[n - 1]!, 1e-6);
      const count = closed ? n - 1 : n;
      const pointAt = (index: number): number[] => {
        if (index < 0) return closed ? poly[count - 1]! : poly[0]!;
        if (index >= count) return closed ? poly[index - count]! : poly[count - 1]!;
        return poly[index]!;
      };
      // A closed subpath has `count` segments (the last one wraps back to the
      // first point); an open one has `count - 1`.
      const segments = closed ? count : count - 1;
      for (let i = 0; i < segments; i += 1) {
        const p0 = pointAt(i);
        const p1 = pointAt(i + 1);
        const prev0 = pointAt(i - 1);
        const next0 = pointAt(i + 1);
        const prev1 = pointAt(i);
        const next1 = pointAt(i + 2);
        const a = addVertex(p0, prev0, next0, 1, halfWidth, info);
        const b = addVertex(p0, prev0, next0, -1, halfWidth, info);
        const c = addVertex(p1, prev1, next1, 1, halfWidth, info);
        const d = addVertex(p1, prev1, next1, -1, halfWidth, info);
        indices.push(a, b, c, b, d, c);
      }
    }
  }

  private flatten_subpath(
    data: ArrayLike<number>,
    rowDim: number,
    start: number,
    end: number,
    tolerance: number,
  ): number[][] {
    const pointAt = (index: number): number[] => {
      const base = index * rowDim;
      return [Number(data[base] ?? 0), Number(data[base + 1] ?? 0), Number(data[base + 2] ?? 0)];
    };
    const polyline: number[][] = [];
    if (end - start < 4) {
      for (let i = start; i < end; i += 1) polyline.push(pointAt(i));
      return polyline;
    }
    polyline.push(pointAt(start));
    for (let i = start; i + 3 < end; i += 4) {
      flattenCubic(
        polyline,
        pointAt(i),
        pointAt(i + 1),
        pointAt(i + 2),
        pointAt(i + 3),
        tolerance,
      );
    }
    return polyline;
  }
}
