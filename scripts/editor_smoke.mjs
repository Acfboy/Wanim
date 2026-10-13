#!/usr/bin/env node
// Headless smoke test for the live Python editor demo: compile a preset with
// the browser rapydscript compiler, run it through wanim in real time and
// verify the canvas got drawn (and, optionally, save a screenshot).
//
// Usage: node scripts/editor_smoke.mjs [preset] [--out screenshot.png] [--code file.py]
import { createServer } from "node:http";
import { readFile, stat, writeFile } from "node:fs/promises";
import { extname, dirname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".mjson": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8", ".ttf": "font/ttf", ".svg": "image/svg+xml",
  ".png": "image/png", ".py": "text/x-python; charset=utf-8",
};

function startServer() {
  return new Promise((resolveServer) => {
    const server = createServer(async (req, res) => {
      try {
        const url = new URL(req.url, "http://localhost");
        let path = decodeURIComponent(url.pathname);
        if (path.endsWith("/")) path += "index.html";
        const target = normalize(join(wanimRoot, path));
        if (!target.startsWith(wanimRoot)) { res.writeHead(403).end("Forbidden"); return; }
        const info = await stat(target);
        if (info.isDirectory()) { res.writeHead(302, { Location: path + "/" }).end(); return; }
        const body = await readFile(target);
        res.writeHead(200, { "Content-Type": MIME[extname(target)] ?? "application/octet-stream", "Cache-Control": "no-store" });
        res.end(body);
      } catch {
        res.writeHead(404).end("Not found");
      }
    });
    server.listen(0, "127.0.0.1", () => resolveServer(server));
  });
}

async function main() {
  const argv = process.argv.slice(2);
  // Default to a preset whose final frame still has content on the canvas
  // (SquareToCircle ends with FadeOut, i.e. a black canvas).
  const preset = argv[0] && !argv[0].startsWith("--") ? argv[0] : "pointcloud";
  const outIndex = argv.indexOf("--out");
  const out = outIndex >= 0 ? resolve(argv[outIndex + 1]) : null;
  const codeIndex = argv.indexOf("--code");
  const codePath = codeIndex >= 0 ? resolve(argv[codeIndex + 1]) : null;
  const profileIndex = argv.indexOf("--profile");
  const profilePath = profileIndex >= 0 ? resolve(argv[profileIndex + 1]) : null;
  const timeoutMs = Number(process.env.EDITOR_SMOKE_TIMEOUT_MS ?? 240000);
  const pythonCode = codePath !== null ? await readFile(codePath, "utf-8") : null;

  const server = await startServer();
  const port = server.address().port;
  const { chromium } = await import("playwright");
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });
    page.on("pageerror", (error) => errors.push(String(error)));
    await page.goto(`http://127.0.0.1:${port}/demo/index.html`, { waitUntil: "load" });
    // demo.js loads its ESM imports asynchronously; wait until the default
    // preset has been inserted into the CodeMirror editor.  The page exposes
    // `window.__wanimDemo` for automation.
    await page.waitForFunction("window.__wanimDemo?.editor.getValue().length > 0", null, { timeout: 30000 });
    if (pythonCode !== null) {
      await page.evaluate((code) => {
        window.__wanimDemo.editor.setValue(code);
      }, pythonCode);
      console.log(`code: ${pythonCode.length} chars from ${codePath}`);
    } else {
      await page.selectOption("#preset", preset);
      const codeLen = await page.evaluate(() => window.__wanimDemo.editor.getValue().length);
      console.log(`preset ${preset}: ${codeLen} chars`);
    }
    let cdp = null;
    if (profilePath !== null) {
      cdp = await page.context().newCDPSession(page);
      await cdp.send("Profiler.enable");
      await cdp.send("Profiler.start");
    }
    await page.click("#run");
    // The playground-style output pane marks the run's terminal line with
    // `.output-done` (`●` prefix) / `.output-error` (`✗` prefix).
    await page.waitForFunction(
      "document.querySelector('#output .output-done, #output .output-error') !== null",
      null,
      { timeout: timeoutMs },
    );
    if (cdp !== null) {
      const { profile } = await cdp.send("Profiler.stop");
      await writeFile(profilePath, JSON.stringify(profile));
      console.log(`wrote profile -> ${profilePath}`);
    }
    const failed = await page.evaluate(
      () => document.querySelector("#output .output-error") !== null,
    );
    const output = await page.textContent("#output");
    const nonBlank = await page.evaluate(() => {
      const canvas = document.getElementById("stage");
      const ctx = canvas.getContext("2d");
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let lit = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] > 20 || data[i + 1] > 20 || data[i + 2] > 20) lit += 1;
      }
      return lit;
    });
    if (out) await page.screenshot({ path: out });
    const fpsMatch = /Done: (\d+) frames in ([\d.]+) s/.exec(output);
    if (fpsMatch) {
      const frames = Number(fpsMatch[1]);
      const seconds = Number(fpsMatch[2]);
      console.log(`live: ${frames} frames in ${seconds.toFixed(2)} s (${(frames / seconds).toFixed(1)} fps)`);
    }
    console.log(`status: ${failed ? "Error." : "Done."}`);
    console.log(output.trim());
    console.log(`lit pixels: ${nonBlank}`);
    if (errors.length) console.error("browser errors:\n" + errors.join("\n"));
    const ok = !failed && nonBlank > 100;
    process.exit(ok ? 0 : 1);
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
