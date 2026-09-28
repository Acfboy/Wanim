#!/usr/bin/env node
// Local server for the demo page (`demo/`).
//
//   pnpm demo
//
// The page loads its browser bundles from `../dist/` (`dist/` is gitignored),
// so the bundles are built first if they are missing.  The repository root is
// served so the page can reach both `demo/` and `dist/`; `fonts/*.ttf` requests
// are answered with the host's default English font, so no font is stored in
// the repository.  Refresh the bundles with `pnpm build:demo`.
//
// Usage: node scripts/demo-server.mjs [port]
import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveDefaultFontFile } from "./font-resolver.mjs";

const here = resolve(fileURLToPath(import.meta.url), "..");
const root = resolve(here, "..");
const port = Number(process.argv[2] ?? 5174);
const fontFile = resolveDefaultFontFile();

const REQUIRED = ["wanim.browser.js", "py2js.browser.js", "rapydscript.js"].map((name) =>
  join(root, "dist", name),
);
if (REQUIRED.some((file) => !existsSync(file))) {
  console.log("wanim: demo bundles are missing, running `pnpm build:demo`…");
  const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
  const result = spawnSync(pnpm, ["build:demo"], { cwd: root, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjson": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".ttf": "font/ttf",
  ".otf": "font/otf",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${port}`);
    let path = decodeURIComponent(url.pathname);

    // Fonts are served from the host, never from the repository.
    if (/(^|\/)fonts\/[^/]+\.(ttf|otf)$/.test(path)) {
      if (fontFile === null) {
        res.writeHead(404).end("No system font found; set WANIM_FONT");
        return;
      }
      const body = await readFile(fontFile);
      res.writeHead(200, { "Content-Type": "font/ttf", "Cache-Control": "no-store" });
      res.end(body);
      return;
    }

    if (path === "/") {
      res.writeHead(302, { Location: "/demo/" }).end();
      return;
    }
    if (path.endsWith("/")) path += "index.html";
    const target = normalize(join(root, path));
    if (!target.startsWith(root)) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    const info = await stat(target);
    if (info.isDirectory()) {
      res.writeHead(302, { Location: path + "/" }).end();
      return;
    }
    const body = await readFile(target);
    res.writeHead(200, {
      "Content-Type": MIME[extname(target)] ?? "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch {
    res.writeHead(404).end("Not found");
  }
});

server.listen(port, () => {
  console.log(`wanim demo: http://localhost:${port}/demo/`);
  if (fontFile === null) {
    console.warn("wanim: no default English font found; Text/MarkupText scenes will fail. Set WANIM_FONT.");
  }
});
