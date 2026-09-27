// Browser manim editor: Python source -> rapydscript -> wanim -> Canvas2D.
//
// Pipeline (mirrors py2js's Node CLI):
//   1. `preprocessImports` strips `from manim import *` / `import numpy as np`
//      (py2js exposes both as globals instead).
//   2. The browser rapydscript compiler (`demo/rapydscript.js`, built from the
//      py2js-patched third-party checkout) translates the Python to JS with
//      `operator_overloading`.
//   3. The compiled JS runs in a hidden iframe whose globals are the py2js
//      runtime (wrapped wanim classes, numpy facade, `ρσ_op_*` helpers).
//   4. Scene subclasses are discovered and rendered with
//      `wanim.render_scene_to_canvas`, then played back on the canvas.
import { configureWasm } from "numpy-ts";

try {
  configureWasm({ maxMemory: 1024 * 1024 * 1024 });
} catch (error) {
  console.warn("numpy-ts configureWasm failed:", error.message);
}

const wanim = await import("wanim");
const py2js = await import("py2js");

const codeEl = document.getElementById("code");
const qualityEl = document.getElementById("quality");
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");
const showJsBtn = document.getElementById("show-js");
const jsOutputEl = document.getElementById("js-output");

let player = null;
let sandbox = null;

const PRESETS = {
  square: `from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        square = Square()
        square.rotate(PI / 4)
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
`,
  opening: `from manim import *

class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \\LaTeX")
        basel = MathTex(r"\\sum_{n=1}^\\infty \\frac{1}{n^2} = \\frac{\\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(Write(title), FadeIn(basel, shift=DOWN))
        self.wait()
        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(Transform(title, transform_title))
        self.wait()
`,
  threeD: `from manim import *

class RotatingSphere(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(65 * DEGREES, -45 * DEGREES)
        self.play(Create(axes))
        self.begin_ambient_camera_rotation(0.3)
        sphere = Sphere()
        sphere.set_fill(GRAY, 1)
        self.add(sphere)
        self.play(Rotate(sphere, 360 * DEGREES, OUT), run_time=2)
        self.wait(1)
`,
};

function setStatus(text) { statusEl.textContent = text; }
function setBusy(busy) {
  runBtn.disabled = busy;
  stopBtn.disabled = !busy;
}

function stopPlayback() {
  if (player) { player.stop(); player = null; }
  stopBtn.disabled = true;
}

function destroySandbox() {
  if (sandbox && sandbox.frame) sandbox.frame.remove();
  sandbox = null;
}

/**
 * Create the execution context: a hidden same-origin iframe whose global scope
 * holds the py2js runtime.  A fresh iframe per run gives a clean global scope
 * (rapydscript's `private_scope: false` output declares top-level `var`s there).
 */
function createSandbox(runtime) {
  const frame = document.createElement("iframe");
  frame.style.display = "none";
  document.body.appendChild(frame);
  const win = frame.contentWindow;
  for (const [key, value] of Object.entries(runtime.globals)) {
    try { win[key] = value; } catch { /* read-only globals */ }
  }
  win.console = console;
  win.globalThis = win;
  win.__name__ = "__main__";
  py2js.installPythonShims({
    String: win.String,
    get ρσ_str() { return win.ρσ_str; },
  });
  return { frame, win, injected: runtime.globals };
}

function isSceneSubclass(value) {
  if (typeof value !== "function") return false;
  const proto = value.prototype;
  if (proto === undefined || proto === null || typeof proto !== "object") return false;
  if (proto === wanim.Scene.prototype) return false;
  return proto instanceof wanim.Scene;
}

function discoverScenes(win, injected) {
  const found = {};
  for (const key of Object.keys(win)) {
    if (key.startsWith("ρσ")) continue;
    const value = win[key];
    if (injected[key] === value) continue;
    if (isSceneSubclass(value)) found[key] = value;
  }
  return found;
}

async function compilePython(source) {
  const pre = py2js.preprocessImports(source);
  if (typeof window.RapydScript === "undefined") {
    throw new Error("rapydscript compiler not loaded (demo/rapydscript.js)");
  }
  const js = await window.RapydScript.compile(pre.code, "scene.py", {
    operator_overloading: true,
    bare: true,
    beautify: true,
  });
  return { js, numpyNames: pre.numpyNames };
}

function applyNumpyNames(runtime, numpyNames) {
  const extra = {};
  for (const name of numpyNames) {
    if (name === "*") Object.assign(extra, runtime.np);
    else if (name in runtime.np) extra[name] = runtime.np[name];
  }
  return extra;
}

async function run() {
  stopPlayback();
  destroySandbox();
  setBusy(true);
  statsEl.textContent = "";
  setStatus("Compiling Python with rapydscript…");
  await new Promise((resolve) => setTimeout(resolve, 20));
  try {
    const { js, numpyNames } = await compilePython(codeEl.value);
    jsOutputEl.textContent = js;

    setStatus("Executing translated JS…");
    await new Promise((resolve) => setTimeout(resolve, 20));
    const runtime = py2js.buildRuntime();
    Object.assign(runtime.globals, applyNumpyNames(runtime, numpyNames));
    sandbox = createSandbox(runtime);
    try {
      sandbox.win.eval(js);
    } catch (error) {
      throw new Error(`translated JS threw: ${error.message}`);
    }

    const scenes = discoverScenes(sandbox.win, sandbox.injected);
    const names = Object.keys(scenes);
    if (names.length === 0) throw new Error("no Scene subclass found in the code");
    const sceneClass = scenes[names[0]];

    setStatus(`Running ${names[0]} (recording frames)…`);
    await new Promise((resolve) => setTimeout(resolve, 20));
    const result = wanim.render_scene_to_canvas(sceneClass, stage, {
      quality: qualityEl.value,
      drawFrames: false,
      recordFrames: true,
    });
    const { stats, frames, meta } = result;
    const setupMs = Math.max(0, stats.totalMs - stats.playMs);
    statsEl.textContent =
      `scenes: ${names.join(", ")}\n` +
      `frames: ${stats.frames} (${stats.recordCount} records), plays: ${meta.num_plays}\n` +
      `generation: ${stats.totalMs.toFixed(0)} ms ` +
      `(${(stats.frames / (stats.totalMs / 1000)).toFixed(1)} gen fps)\n` +
      `setup (construct/Tex): ${setupMs.toFixed(0)} ms · play loop: ${stats.playMs.toFixed(0)} ms\n` +
      `extract: ${stats.extractMs.toFixed(0)} ms · serialize: ${stats.serializeMs.toFixed(0)} ms`;
    setStatus("Playing back recorded frames.");
    player = wanim.play_frames(stage, meta, frames, {
      loop: true,
      fps: meta.frame_rate,
    });
    stopBtn.disabled = false;
  } catch (error) {
    setStatus(`Error: ${error.message}`);
    console.error(error);
  } finally {
    setBusy(false);
  }
}

async function loadPreset(name) {
  const source = PRESETS[name];
  if (source === undefined) return;
  codeEl.value = source;
  setStatus(`Loaded preset: ${name}`);
}

runBtn.addEventListener("click", run);
stopBtn.addEventListener("click", () => { stopPlayback(); setStatus("Stopped."); });
showJsBtn.addEventListener("click", () => { jsOutputEl.hidden = !jsOutputEl.hidden; });
document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => loadPreset(button.dataset.preset));
});

loadPreset("square");
