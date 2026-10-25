#!/usr/bin/env node
// Headless-browser smoke test for the interactive-plane features:
//
//   1. DraggablePoint: the reader drags the dot, `get_center()` follows, an
//      attached label follows too, and a reaction reads the new position.
//   2. InteractiveLinearTransformationScene: the scene installs no controls of
//      its own; the page wires a DraggablePoint to `set_matrix`, and dragging
//      it updates the transformed plane and tracked vectors.
//
// Playwright drives real pointer events on a canvas; the scene reads its
// state back through `window.__scene`.  Both scenes are compiled from Python
// by the same py2js bridge the playground uses, so this also covers the
// keyword/constructor translation path.
//
// Usage: node scripts/interactive_smoke.mjs [--headed] [--keep]
// Prerequisite: `pnpm build:browser && pnpm build:py2js` (or build:demo).

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wanimRoot = resolve(here, "..");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjson": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".ttf": "font/ttf",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".py": "text/x-python; charset=utf-8",
};

const DRAG_SCENE = `from manim import *

class DragDemo(InteractiveScene):
    def blueprint(self):
        self.point = DraggablePoint([1, 1, 0], color=YELLOW, radius=0.12)
        self.label = Dot([1, 1.5, 0], radius=0.08, color=BLUE)
        self.point.attach(self.label)

        # A reaction that reads get_center() and moves a plain dot: no TeX/font
        # load is needed, so the smoke test stays offline-deterministic.
        self.marker = Dot([0, 0, 0], radius=0.1, color=TEAL)
        self.marker.add_reaction(
            lambda m: m.move_to([
                self.point.get_center()[0] * 0.5,
                self.point.get_center()[1] * 0.5,
                0,
            ])
        )
        self.add(self.point, self.marker)
`;

const LINEAR_SCENE = `from manim import *

class LinearDemo(InteractiveLinearTransformationScene):
    def blueprint(self):
        self.add_unit_square(animate=False)
        self.vector = self.add_vector([2, 1, 0], color=BLUE, animate=False)

        # The scene has no built-in dragging: the author owns the interaction.
        self.handle = DraggablePoint([1, 0, 0], color=GREEN)
        def on_i_handle(control, previous, next_value):
            point = control.get_center()
            matrix = self.get_matrix()
            matrix[0][0] = point[0]
            matrix[1][0] = point[1]
            self.set_matrix(matrix)
        self.handle.on_input(on_i_handle)
        self.add(self.handle)

    def construct(self):
        # In construct the apply_* family animates like the base scene.
        self.apply_matrix([[1, 1], [0, 1]], run_time=2)
`;

const SCENES = { drag: DRAG_SCENE, linear: LINEAR_SCENE };

function pageHtml(source) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>wanim interactive smoke</title>
<style>
  html, body { margin: 0; background: #111; }
  #stage { display: block; width: 960px; height: 540px; touch-action: none; }
</style>
<script type="importmap">
{
  "imports": {
    "wanim": "/dist/wanim.browser.js",
    "numpy-ts": "/dist/numpy-ts.browser.js",
    "py2js": "/dist/bridge.browser.js"
  }
}
</script>
</head>
<body>
<canvas id="stage" width="1280" height="720"></canvas>
<script src="/dist/rapydscript.js"></script>
<script type="module">
  import { runPythonOnCanvas } from "py2js";

  window.__errors = [];
  window.__scene = null;
  window.addEventListener("error", (event) => window.__errors.push(String(event.error ?? event.message)));
  window.addEventListener("unhandledrejection", (event) => window.__errors.push(String(event.reason)));

  const canvas = document.getElementById("stage");
  window.__run = runPythonOnCanvas(${JSON.stringify(source)}, canvas, {
    quality: "-qm",
    realtime: true,
    fps: 30,
    onSceneReady: (scene) => { window.__scene = scene; },
  }).catch((error) => {
    window.__errors.push(String(error?.stack ?? error));
  });
</script>
</body>
</html>`;
}

function startServer() {
  return new Promise((resolveServer) => {
    const server = createServer(async (req, res) => {
      try {
        const url = new URL(req.url, "http://localhost");
        if (url.pathname === "/smoke") {
          const scene = SCENES[url.searchParams.get("scene")] ?? DRAG_SCENE;
          res.writeHead(200, { "Content-Type": MIME[".html"], "Cache-Control": "no-store" });
          res.end(pageHtml(scene));
          return;
        }
        let path = decodeURIComponent(url.pathname);
        if (path.endsWith("/")) path += "index.html";
        const target = normalize(join(wanimRoot, path));
        if (!target.startsWith(wanimRoot)) {
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
    server.listen(0, "127.0.0.1", () => resolveServer(server));
  });
}

async function ensureBuilt() {
  for (const file of ["dist/wanim.browser.js", "dist/bridge.browser.js", "dist/numpy-ts.browser.js"]) {
    try {
      await stat(join(wanimRoot, file));
    } catch {
      throw new Error(`missing ${file}; run \`pnpm build:browser && pnpm build:py2js\` first`);
    }
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function near(actual, expected, tolerance = 0.03) {
  return Math.abs(actual - expected) <= tolerance;
}

async function main() {
  const argv = process.argv.slice(2);
  const headed = argv.includes("--headed");
  await ensureBuilt();

  const server = await startServer();
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;
  const { chromium } = await import("playwright");
  const browser = await chromium.launch({ headless: !headed, args: ["--no-sandbox"] });
  let failures = 0;

  const runCase = async (name, sceneName, body) => {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(String(error)));
    try {
      await page.goto(`${base}/smoke?scene=${sceneName}`, { waitUntil: "load" });
      await page.waitForFunction(
        `window.__scene !== null && window.__scene.interactive_host?.live === true`,
        null,
        { timeout: 60000 },
      );
      await page.waitForFunction(
        sceneName === "linear"
          ? "window.__scene.square !== undefined && window.__scene.handle !== undefined"
          : "window.__scene.point !== undefined",
        null,
        { timeout: 60000 },
      );
      // One presentation frame so setup/reactions have drawn once.
      await page.evaluate(
        () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))),
      );

      const box = await page.locator("#stage").boundingBox();
      assert(box !== null, "canvas has no bounding box");
      const frame = await page.evaluate(() => {
        const camera = window.__scene.camera;
        return { width: camera.frame_width, height: camera.frame_height };
      });
      const toClient = (world) => ({
        x: box.x + ((world[0] + frame.width / 2) / frame.width) * box.width,
        y: box.y + ((frame.height / 2 - world[1]) / frame.height) * box.height,
      });
      const drag = async (from, to) => {
        const start = toClient(from);
        const end = toClient(to);
        await page.mouse.move(start.x, start.y);
        await page.mouse.down();
        await page.mouse.move(end.x, end.y, { steps: 12 });
        await page.mouse.up();
        await page.evaluate(
          () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))),
        );
      };
      const state = () =>
        page.evaluate(() => {
          const scene = window.__scene;
          const center = (mob) => {
            const value = mob.get_center();
            return [value[0], value[1]];
          };
          const vector = scene.vector ? scene.vector.get_vector() : null;
          return {
            point: scene.point ? center(scene.point) : null,
            marker: scene.marker ? center(scene.marker) : null,
            label: scene.label ? center(scene.label) : null,
            handle: scene.handle ? center(scene.handle) : null,
            matrix: scene.get_matrix?.() ?? null,
            vector: vector ? [vector[0], vector[1]] : null,
            square: scene.square ? center(scene.square) : null,
            errors: window.__errors,
          };
        });

      await body({ page, drag, state, errors, toClient });
      const sceneErrors = await page.evaluate(() => window.__errors);
      assert(sceneErrors.length === 0, `scene errors: ${sceneErrors.join("; ")}`);
      assert(errors.length === 0, `console errors: ${errors.join("; ")}`);
      console.log(`ok - ${name}`);
    } catch (error) {
      failures += 1;
      let diagnostics = "";
      try {
        diagnostics = JSON.stringify(
          await page.evaluate(() => ({
            errors: window.__errors ?? null,
            hasScene: window.__scene !== null,
            sceneKeys: window.__scene
              ? Object.keys(window.__scene).filter((key) => key.startsWith("interactive") || key.startsWith("basis") || key === "point")
              : null,
            host: window.__scene?.interactive_host?.live ?? null,
          })),
          null,
          2,
        );
      } catch {
        /* page may already be gone */
      }
      console.error(`FAIL - ${name}\n${error?.stack ?? error}\n${diagnostics}`);
    } finally {
      await page.close();
    }
  };

  await runCase(
    "DraggablePoint drags, attaches a label and reports get_center",
    "drag",
    async ({ drag, state }) => {
      await drag([1, 1], [-2, 0.5]);
      const after = await state();
      assert(near(after.point[0], -2) && near(after.point[1], 0.5), `point at ${after.point}`);
      assert(
        near(after.label[0], -2) && near(after.label[1], 1),
        `attached label at ${after.label}`,
      );
      assert(
        near(after.marker[0], -1) && near(after.marker[1], 0.25),
        `reaction marker at ${after.marker}`,
      );
    },
  );

  await runCase(
    "InteractiveLinearTransformationScene follows a user-wired handle",
    "linear",
    async ({ page, drag, state }) => {
      await drag([1, 0], [2, 1]);
      const after = await state();
      const matrix = after.matrix;
      assert(matrix !== null, "no matrix");
      assert(
        near(matrix[0][0], 2) && near(matrix[1][0], 1) && near(matrix[1][1], 1),
        `matrix ${JSON.stringify(matrix)}`,
      );
      assert(near(after.handle[0], 2) && near(after.handle[1], 1), `handle ${after.handle}`);
      // Unit square centre (0.5, 0.5) -> M @ (0.5, 0.5) = (1, 1).
      assert(near(after.square[0], 1) && near(after.square[1], 1), `square ${after.square}`);
      // The tracked vector shows M @ [2, 1] = [4, 3].
      assert(after.vector[0] > 3.4 && after.vector[1] > 2.4, `vector ${after.vector}`);

      // Advance to construct: apply_matrix must animate (not jump) and then
      // compose S @ M with S = [[1, 1], [0, 1]] and M from the blueprint drag.
      const M = after.matrix;
      const expected = [
        [M[0][0] + M[1][0], M[0][1] + M[1][1]],
        [M[1][0], M[1][1]],
      ];
      const expectedSquare = [
        (expected[0][0] + expected[0][1]) * 0.5,
        (expected[1][0] + expected[1][1]) * 0.5,
      ];
      const before = await page.evaluate(() => {
        const c = window.__scene.square.get_center();
        return [c[0], c[1]];
      });
      await page.evaluate(() => window.__scene.next_step());
      // Sample mid-animation (2 s animation, sampled at ~0.7 s): the square
      // must be off both endpoints.
      await page.waitForTimeout(700);
      const mid = await page.evaluate(() => {
        const c = window.__scene.square.get_center();
        return [c[0], c[1]];
      });
      // Let the 2 s animation finish (generous wall-clock budget).
      await page.waitForFunction(
        (exp) => {
          const m = window.__scene.get_matrix();
          return (
            Math.abs(m[0][0] - exp[0][0]) < 0.05 &&
            Math.abs(m[0][1] - exp[0][1]) < 0.05 &&
            Math.abs(m[1][0] - exp[1][0]) < 0.05 &&
            Math.abs(m[1][1] - exp[1][1]) < 0.05
          );
        },
        expected,
        { timeout: 15000 },
      );
      const done = await state();
      assert(
        near(done.matrix[0][0], expected[0][0]) && near(done.matrix[1][1], expected[1][1]),
        `composed matrix ${JSON.stringify(done.matrix)} vs ${JSON.stringify(expected)}`,
      );
      assert(
        near(done.square[0], expectedSquare[0]) && near(done.square[1], expectedSquare[1]),
        `square ${done.square} vs ${expectedSquare}`,
      );
      const moved =
        Math.hypot(mid[0] - before[0], mid[1] - before[1]) > 0.02 &&
        Math.hypot(mid[0] - done.square[0], mid[1] - done.square[1]) > 0.02;
      assert(moved, `animation jumped: before=${before} mid=${mid} end=${done.square}`);
    },
  );

  await browser.close();
  server.close();
  if (failures > 0) {
    console.error(`${failures} interactive smoke case(s) failed`);
    process.exit(1);
  }
  console.log("interactive smoke: all cases passed");
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});
