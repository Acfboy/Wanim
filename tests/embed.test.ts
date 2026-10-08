// Codec behind the iframe embedding flow: `z.` (deflate-raw) and `u.`
// (stored) tokens, hash building/parsing and the iframe snippet.  The
// compression fallback is exercised by forcing `compression: "store"` and by
// stubbing the platform streams away.
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  build_embed_hash,
  build_embed_iframe,
  decode_embed_code,
  decode_embed_hash,
  EMBED_MESSAGE_SOURCE,
  encode_embed_code,
  parse_embed_hash,
} from "../src/embed.js";

const SAMPLE = `from manim import *

class Demo(Scene):
    def construct(self):
        self.play(Create(Circle()))
        self.wait()
`;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("encode_embed_code / decode_embed_code", () => {
  it("round-trips through the compressed form when CompressionStream exists", async () => {
    const token = await encode_embed_code(SAMPLE);
    expect(token.startsWith("z.")).toBe(true);
    expect(await decode_embed_code(token)).toBe(SAMPLE);
  });

  it("round-trips Chinese source", async () => {
    const code = `from manim import *\n\nclass 中文场景(Scene):\n    def construct(self):\n        text = Text("你好，世界！")\n        self.play(Write(text))\n`;
    const token = await encode_embed_code(code);
    expect(await decode_embed_code(token)).toBe(code);
  });

  it("round-trips emoji and astral characters", async () => {
    const code = `Text("动画 🎬 → 🚀 — 𝕏 ∑")`;
    const token = await encode_embed_code(code);
    expect(await decode_embed_code(token)).toBe(code);
  });

  it("round-trips long repetitive code", async () => {
    const code = "self.wait(1)\n".repeat(20000);
    const token = await encode_embed_code(code);
    expect(token.startsWith("z.")).toBe(true);
    expect(await decode_embed_code(token)).toBe(code);
    // Deflate must actually shrink a highly repetitive payload.
    expect(token.length).toBeLessThan(code.length);
  });

  it("round-trips the empty string through the stored form", async () => {
    const token = await encode_embed_code("");
    expect(token).toBe("u.");
    expect(await decode_embed_code(token)).toBe("");
  });

  it('forces the stored form with compression: "store"', async () => {
    const token = await encode_embed_code(SAMPLE, { compression: "store" });
    expect(token.startsWith("u.")).toBe(true);
    expect(await decode_embed_code(token)).toBe(SAMPLE);
  });

  it("falls back to the stored form when CompressionStream is unavailable", async () => {
    vi.stubGlobal("CompressionStream", undefined);
    const token = await encode_embed_code(SAMPLE);
    expect(token.startsWith("u.")).toBe(true);
    expect(await decode_embed_code(token)).toBe(SAMPLE);
  });

  it("falls back to the stored form when CompressionStream construction throws", async () => {
    vi.stubGlobal(
      "CompressionStream",
      class {
        constructor() {
          throw new TypeError('Unsupported compression format: "deflate-raw"');
        }
      },
    );
    const token = await encode_embed_code(SAMPLE);
    expect(token.startsWith("u.")).toBe(true);
    expect(await decode_embed_code(token)).toBe(SAMPLE);
  });

  it("rejects a compressed token when DecompressionStream is unavailable", async () => {
    const token = await encode_embed_code(SAMPLE);
    expect(token.startsWith("z.")).toBe(true);
    vi.stubGlobal("DecompressionStream", undefined);
    await expect(decode_embed_code(token)).rejects.toThrow(/DecompressionStream is unavailable/);
  });

  it("rejects empty and non-string tokens", async () => {
    await expect(decode_embed_code("")).rejects.toThrow(/non-empty string/);
    await expect(decode_embed_code(undefined as unknown as string)).rejects.toThrow(
      /non-empty string/,
    );
  });

  it("rejects unknown prefixes", async () => {
    await expect(decode_embed_code("x.abc")).rejects.toThrow(/expected a "z\." \(deflate-raw\) or "u\."/);
    await expect(decode_embed_code("abc")).rejects.toThrow(/prefix/);
  });

  it("rejects malformed base64url payloads", async () => {
    await expect(decode_embed_code("u.!!!!")).rejects.toThrow(/not valid base64url/);
    await expect(decode_embed_code("z.!!!!")).rejects.toThrow(/not valid base64url/);
  });

  it("rejects a truncated deflate payload with a clear error", async () => {
    const token = await encode_embed_code(SAMPLE);
    const truncated = `${token.slice(0, token.length - 12)}`;
    await expect(decode_embed_code(truncated)).rejects.toThrow(/deflate payload could not be decoded/);
  });

  it("rejects stored bytes that are not valid UTF-8", async () => {
    // 0xFF is never a valid UTF-8 byte.
    await expect(decode_embed_code("u._w")).rejects.toThrow(/not valid UTF-8/);
  });

  it("exports the postMessage source constant", () => {
    expect(EMBED_MESSAGE_SOURCE).toBe("wanim-embed");
  });
});

describe("build_embed_hash / parse_embed_hash / decode_embed_hash", () => {
  it("builds a hash with c/q and parses it back", async () => {
    const hash = await build_embed_hash(SAMPLE, { quality: "-qh" });
    expect(hash.startsWith("#c=")).toBe(true);
    expect(hash).toContain("&q=-qh");
    const parsed = parse_embed_hash(hash);
    expect(parsed.quality).toBe("-qh");
    expect(await decode_embed_code(parsed.token)).toBe(SAMPLE);
  });

  it("omits optional flags", async () => {
    const hash = await build_embed_hash(SAMPLE);
    expect(hash).not.toContain("&q=");
    expect(parse_embed_hash(hash).quality).toBeUndefined();
  });

  it("ignores legacy style flags from old links", () => {
    const parsed = parse_embed_hash("c=u.aGk&q=-qm&style=legacy");
    expect(parsed.token).toBe("u.aGk");
    expect(parsed.quality).toBe("-qm");
  });

  it("accepts a hash without the leading # and rejects a missing token", () => {
    const parsed = parse_embed_hash("c=u.aGk&q=-qm");
    expect(parsed.token).toBe("u.aGk");
    expect(parsed.quality).toBe("-qm");
    expect(() => parse_embed_hash("#q=-qm")).toThrow(/missing the "c"/);
    expect(() => parse_embed_hash("")).toThrow(/missing the "c"/);
  });

  it("rejects unknown quality flags in build and parse", async () => {
    await expect(build_embed_hash(SAMPLE, { quality: "-qz" })).rejects.toThrow(/Unknown quality/);
    expect(() => parse_embed_hash("c=u.aGk&q=-qz")).toThrow(/Unknown quality/);
  });

  it("decode_embed_hash returns code and flags together", async () => {
    const hash = await build_embed_hash(SAMPLE, { quality: "-ql" });
    await expect(decode_embed_hash(hash)).resolves.toEqual({
      code: SAMPLE,
      quality: "-ql",
    });
  });
});

describe("build_embed_iframe", () => {
  it("produces a lazy, sandboxed, responsive iframe snippet", () => {
    const snippet = build_embed_iframe("https://example.com/embed#c=u.aGk", {
      width: 800,
      height: 450,
      title: "demo scene",
    });
    expect(snippet).toContain('src="https://example.com/embed#c=u.aGk"');
    expect(snippet).toContain('title="demo scene"');
    expect(snippet).toContain('loading="lazy"');
    expect(snippet).toContain('sandbox="allow-scripts allow-same-origin"');
    expect(snippet).toContain("allowfullscreen");
    expect(snippet).toContain("aspect-ratio:800/450");
    expect(snippet).toContain("max-width:800px");
  });

  it("defaults to 960x540 and escapes attribute values", () => {
    const snippet = build_embed_iframe('https://example.com/embed?a="b"&c=<x>');
    expect(snippet).toContain("aspect-ratio:960/540");
    expect(snippet).toContain("&quot;b&quot;");
    expect(snippet).toContain("&amp;c=&lt;x&gt;");
  });
});
