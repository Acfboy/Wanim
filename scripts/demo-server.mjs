#!/usr/bin/env node
// Local server for the demo page (`demo/`).
//
//   pnpm demo
//
// The page loads its browser bundles from `../dist/` (`dist/` is gitignored),
// so the bundles are built first if they are missing.  The repository root is
// served so the page can reach both `demo/` and `dist/` — including the
// on-demand `dist/chunks/` (MathJax) that the wanim bundle imports
// lazily; the shipped fonts are copied to `dist/fonts/` by `build:browser` and
// served as static files.  Refresh the bundles with `pnpm build:demo`.
//
// Usage: node scripts/demo-server.mjs [port]
import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = resolve(fileURLToPath(import.meta.url), "..");
const root = resolve(here, "..");
const port = Number(process.argv[2] ?? 5174);

const REQUIRED = ["wanim.browser.js", "bridge.browser.js", "py2js.browser.js", "py-runtime.browser.js", "rapydscript.js", "wanim-element.browser.js"].map(
  (name) => join(root, "dist", name),
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
});
