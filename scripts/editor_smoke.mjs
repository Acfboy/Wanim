#!/usr/bin/env node
// Headless smoke test for the Python editor demo: compile a preset with the
// browser rapydscript compiler, run it through wanim and verify the canvas got
// drawn (and, optionally, save a screenshot).
//
// Usage: node scripts/editor_smoke.mjs [preset] [--out screenshot.png]
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
  const preset = argv[0] && !argv[0].startsWith("--") ? argv[0] : "square";
  const outIndex = argv.indexOf("--out");
  const out = outIndex >= 0 ? resolve(argv[outIndex + 1]) : null;
  const codeIndex = argv.indexOf("--code");
  const codePath = codeIndex >= 0 ? resolve(argv[codeIndex + 1]) : null;
  const profileIndex = argv.indexOf("--profile");
  const profilePath = profileIndex >= 0 ? resolve(argv[profileIndex + 1]) : null;
  const repeatIndex = argv.indexOf("--repeat");
  const repeat = repeatIndex >= 0 ? Number(argv[repeatIndex + 1]) : 1;
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
    await page.goto(`http://127.0.0.1:${port}/demo/editor.html`, { waitUntil: "load" });
    await page.waitForFunction("typeof window.RapydScript !== 'undefined'", null, { timeout: 30000 });
    // editor.js finishes loading its ESM imports asynchronously; wait until it
    // has installed the listeners and loaded the default preset.
    await page.waitForFunction("document.getElementById('code').value.length > 0", null, { timeout: 30000 });
    if (pythonCode !== null) {
      await page.evaluate((code) => { document.getElementById("code").value = code; }, pythonCode);
      console.log(`code: ${pythonCode.length} chars from ${codePath}`);
    } else {
      await page.click(`[data-preset="${preset}"]`);
      const codeLen = await page.evaluate(() => document.getElementById("code").value.length);
      console.log(`preset ${preset}: ${codeLen} chars`);
    }
    const genFps = [];
    let cdp = null;
    if (profilePath !== null) {
      cdp = await page.context().newCDPSession(page);
      await cdp.send("Profiler.enable");
    }
    for (let run = 0; run < repeat; run += 1) {
      if (cdp !== null && run === repeat - 1) await cdp.send("Profiler.start");
      await page.evaluate(() => { document.getElementById("status").textContent = "__reset__"; });
      await page.click("#run");
      await page.waitForFunction(
        "document.getElementById('status').textContent.startsWith('Playing back') || document.getElementById('status').textContent.startsWith('Error')",
        null,
        { timeout: timeoutMs },
      );
      const text = await page.textContent("#stats");
      const match = /generation: .*\(([\d.]+) gen fps\)/.exec(text);
      if (match) genFps.push(Number(match[1]));
      if (cdp !== null && run === repeat - 1) {
        const { profile } = await cdp.send("Profiler.stop");
        await writeFile(profilePath, JSON.stringify(profile));
        console.log(`wrote profile -> ${profilePath}`);
      }
    }
    if (genFps.length > 0) console.log(`gen fps (best): ${Math.max(...genFps).toFixed(1)}  runs: ${genFps.join(", ")}`);
    // Let a few frames play so the canvas is non-trivial.
    await page.waitForTimeout(1200);
    const status = await page.textContent("#status");
    const stats = await page.textContent("#stats");
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
    console.log(`status: ${status.trim()}`);
    console.log(stats.trim());
    console.log(`lit pixels: ${nonBlank}`);
    if (errors.length) console.error("browser errors:\n" + errors.join("\n"));
    const ok = status.startsWith("Playing back") && nonBlank > 100;
    process.exit(ok ? 0 : 1);
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
