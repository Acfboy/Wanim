/**
 * Code-in-fragment embedding helpers.
 *
 * The playgrounds ship an "embed / publish to blog" feature that packs the
 * Python source into the URL fragment of a minimal player page:
 *
 *   `#c=<token>[&q=<quality>]`
 *
 * Fragments are never sent to the server, so the code stays out of access logs
 * and server-side storage.  Two token encodings are defined:
 *
 *   `z.<base64url(deflate-raw(utf8(code)))>`  compressed (preferred)
 *   `u.<base64url(utf8(code))>`               stored verbatim (fallback)
 *
 * `deflate-raw` is used through the platform `CompressionStream` (browsers and
 * Node 18+).  When it is unavailable or throws, {@link encode_embed_code}
 * silently falls back to the `u.` form; decoding a `z.` token requires
 * `DecompressionStream` and fails with a clear error otherwise.
 */

/** Prefix of a deflate-raw-compressed token. */
const COMPRESSED_PREFIX = "z.";
/** Prefix of a stored (uncompressed) token. */
const STORED_PREFIX = "u.";

/** manim quality flags accepted in an embed hash. */
const QUALITY_FLAGS = new Set(["-ql", "-qm", "-qh", "-q4k"]);

/** `source` field stamped on every embed postMessage payload. */
export const EMBED_MESSAGE_SOURCE = "wanim-embed";

export interface EmbedEncodeOptions {
  /**
   * `"auto"` (default) compresses with deflate-raw when the platform supports
   * it; `"store"` always emits the uncompressed `u.` form (tests/debugging).
   */
  compression?: "auto" | "store";
}

export interface EmbedHashOptions {
  /** manim quality flag (`-ql` / `-qm` / `-qh` / `-q4k`). */
  quality?: string;
}

export interface EmbedIframeOptions {
  /** Intrinsic width in CSS pixels (default 960). */
  width?: number;
  /** Intrinsic height in CSS pixels (default 540). */
  height?: number;
  /** iframe `title` attribute (default `"wanim scene"`). */
  title?: string;
}

/** The pieces carried by an embed hash, before the code is decoded. */
export interface ParsedEmbedHash {
  token: string;
  quality?: string;
}

/** An embed hash with the code already decoded. */
export interface DecodedEmbedHash {
  code: string;
  quality?: string;
}

const encoder = new TextEncoder();

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// --- base64url --------------------------------------------------------------

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  // Chunked to keep String.fromCharCode's argument count bounded.
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  const base64 = btoa(binary);
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(payload: string): Uint8Array {
  if (!/^[A-Za-z0-9_-]*$/.test(payload)) {
    throw new Error("Invalid embed token: the payload is not valid base64url.");
  }
  const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
  const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
  let binary: string;
  try {
    binary = atob(padded);
  } catch {
    throw new Error("Invalid embed token: the base64url payload could not be decoded.");
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

// --- deflate-raw streams ----------------------------------------------------

interface ByteTransform {
  readable: ReadableStream<Uint8Array>;
  writable: WritableStream<Uint8Array>;
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value !== undefined) {
      chunks.push(value);
      total += value.byteLength;
    }
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function transformBytes(bytes: Uint8Array, make: () => ByteTransform): Promise<Uint8Array> {
  const stream = make();
  const writing = (async () => {
    const writer = stream.writable.getWriter();
    await writer.write(bytes);
    await writer.close();
  })();
  const reading = collect(stream.readable);
  // Promise.all keeps both branches observed even when one rejects.
  const [, result] = await Promise.all([writing, reading]);
  return result;
}

function makeDeflateStream(): ByteTransform {
  return new CompressionStream("deflate-raw") as unknown as ByteTransform;
}

function makeInflateStream(): ByteTransform {
  return new DecompressionStream("deflate-raw") as unknown as ByteTransform;
}

// --- public API -------------------------------------------------------------

/**
 * Encode Python source as an embed token (`z.<base64url>` when the platform can
 * deflate it, otherwise `u.<base64url>`).  Never throws for ordinary input: a
 * failing/absent `CompressionStream` falls back to the stored form.
 */
export async function encode_embed_code(
  code: string,
  options: EmbedEncodeOptions = {},
): Promise<string> {
  const bytes = encoder.encode(code);
  // Compressing nothing would only grow the token; keep the canonical stored
  // form for the empty string.
  if (bytes.length === 0) return STORED_PREFIX;
  if (options.compression !== "store" && typeof CompressionStream === "function") {
    try {
      const compressed = await transformBytes(bytes, makeDeflateStream);
      return COMPRESSED_PREFIX + bytesToBase64Url(compressed);
    } catch {
      // Fall through to the stored form.
    }
  }
  return STORED_PREFIX + bytesToBase64Url(bytes);
}

/**
 * Decode an embed token produced by {@link encode_embed_code}.  Malformed or
 * unsupported tokens throw an `Error` with a human-readable reason.
 */
export async function decode_embed_code(token: string): Promise<string> {
  if (typeof token !== "string" || token.length === 0) {
    throw new Error("Invalid embed token: expected a non-empty string.");
  }
  if (token.startsWith(STORED_PREFIX)) {
    const bytes = base64UrlToBytes(token.slice(STORED_PREFIX.length));
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new Error("Invalid embed token: the stored payload is not valid UTF-8.");
    }
  }
  if (token.startsWith(COMPRESSED_PREFIX)) {
    if (typeof DecompressionStream !== "function") {
      throw new Error(
        "Cannot decode a compressed embed token: DecompressionStream is unavailable in this environment.",
      );
    }
    const bytes = base64UrlToBytes(token.slice(COMPRESSED_PREFIX.length));
    try {
      const raw = await transformBytes(bytes, makeInflateStream);
      return new TextDecoder("utf-8", { fatal: true }).decode(raw);
    } catch (error) {
      throw new Error(
        `Invalid embed token: the deflate payload could not be decoded (${errorMessage(error)}).`,
      );
    }
  }
  throw new Error('Invalid embed token: expected a "z." (deflate-raw) or "u." (stored) prefix.');
}

/**
 * Build the hash fragment for an embed URL: `#c=<token>[&q=...]`.  `quality`
 * is validated here so a broken flag never reaches a player page.
 */
export async function build_embed_hash(
  code: string,
  options: EmbedHashOptions = {},
): Promise<string> {
  const token = await encode_embed_code(code);
  const params = new URLSearchParams();
  params.set("c", token);
  if (options.quality !== undefined) {
    if (!QUALITY_FLAGS.has(options.quality)) {
      throw new Error(
        `Unknown quality flag "${options.quality}"; expected one of -ql, -qm, -qh, -q4k.`,
      );
    }
    params.set("q", options.quality);
  }
  return `#${params.toString()}`;
}

/**
 * Parse an embed hash (with or without the leading `#`).  Throws when the `c`
 * token is missing or a flag is unknown.
 */
export function parse_embed_hash(hash: string): ParsedEmbedHash {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const params = new URLSearchParams(raw);
  const token = params.get("c");
  if (token === null || token === "") {
    throw new Error('Invalid embed hash: missing the "c" parameter.');
  }
  const parsed: ParsedEmbedHash = { token };
  const quality = params.get("q");
  if (quality !== null && quality !== "") {
    if (!QUALITY_FLAGS.has(quality)) {
      throw new Error(
        `Unknown quality flag "${quality}" in the embed hash; expected one of -ql, -qm, -qh, -q4k.`,
      );
    }
    parsed.quality = quality;
  }
  return parsed;
}

/** Parse and decode an embed hash in one step. */
export async function decode_embed_hash(hash: string): Promise<DecodedEmbedHash> {
  const parsed = parse_embed_hash(hash);
  return {
    code: await decode_embed_code(parsed.token),
    quality: parsed.quality,
  };
}

function escapeAttribute(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Build the ready-to-paste `<iframe>` snippet for an embed URL.
 *
 * The inline style keeps the frame responsive (full width up to `width`,
 * height derived from the aspect ratio) and `sandbox` is set to
 * `allow-scripts allow-same-origin`: scripts are required to run the player,
 * and same-origin access is required for the player's own static assets
 * (fonts, wanim chunks) and for the postMessage protocol.  See the deployment
 * notes in the repo reports for the trade-offs.
 */
export function build_embed_iframe(url: string, options: EmbedIframeOptions = {}): string {
  const width = options.width ?? 960;
  const height = options.height ?? 540;
  const title = options.title ?? "wanim scene";
  return (
    `<iframe src="${escapeAttribute(url)}" title="${escapeAttribute(title)}" ` +
    `width="${width}" height="${height}" loading="lazy" allowfullscreen ` +
    `sandbox="allow-scripts allow-same-origin" ` +
    `style="border:0;display:block;width:100%;max-width:${width}px;` +
    `aspect-ratio:${width}/${height};height:auto"></iframe>`
  );
}
