// wanim web editor: (almost unmodified) manim Python in the browser.
//
//   Python --(rapydscript)--> JS --(wanim, live Canvas2D)--> the preview
//
// The scene is *not* pre-rendered frame by frame and then played back: wanim
// runs the translated scene in real time, yielding to the browser between
// frames (see `render_scene_to_canvas({ realtime: true })`).  rapydscript turns
// `self.play(...)` into `await self.play(...)` and marks the enclosing method
// async, so the Python still reads synchronously.
import { configureWasm } from "numpy-ts";

try {
  configureWasm({ maxMemory: 1024 * 1024 * 1024 });
} catch (error) {
  console.warn("numpy-ts configureWasm failed:", error.message);
}

const wanim = await import("wanim");
const py2js = await import("py2js");

// Target logical frame rate for live playback.  Yields to requestAnimationFrame
// until at least 1000/TARGET_FPS ms have passed before computing the next frame.
const TARGET_FPS = 30;
// Render resolution / animation frame rate (manim quality flag).  Kept fixed so
// the UI stays simple; `-qm` is 1280x720 @ 30fps and matches TARGET_FPS.
const QUALITY = "-qm";

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
  static: `from manim import *

class StaticOnly(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        square = Square()
        square.rotate(PI / 4)
        self.add(circle, square)
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
  sin: `from manim import *
import numpy as np

class AxesPlotSin(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, TAU, PI / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=7,
            y_length=3,
        )
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=GREEN)
        self.play(Create(axes), Create(sin_graph), Create(cos_graph))
        self.wait(0.2)
`,
  unitcircle: `from manim import *

class SineCurveUnitCircle(Scene):
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()

    def show_axis(self):
        x_axis = Line(np.array([-6, 0, 0]), np.array([6, 0, 0]))
        y_axis = Line(np.array([-4, -2, 0]), np.array([-4, 2, 0]))
        self.add(x_axis, y_axis)
        self.origin_point = np.array([-4, 0, 0])
        self.curve_start = np.array([-3, 0, 0])

    def show_circle(self):
        circle = Circle(radius=1)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point
        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += dt * rate
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x, y, 0]), color=YELLOW_A, stroke_width=2)

        self.curve = VGroup()
        self.curve.add(Line(self.curve_start, self.curve_start))

        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(), np.array([x, y, 0]), color=YELLOW_D)
            self.curve.add(new_line)
            return self.curve

        dot.add_updater(go_around_circle)
        self.add(dot)
        self.add(orbit, always_redraw(get_line_to_circle),
                 always_redraw(get_line_to_curve), always_redraw(get_curve))
        self.wait(8.5)
        dot.remove_updater(go_around_circle)
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
  threeDCamera: `from manim import *

class ThreeDCameraRotation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(circle, axes)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait()
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.wait()
`,
};

const codeEl = document.getElementById("code");
const highlightEl = document.getElementById("highlight");
const highlightCodeEl = document.getElementById("highlight-code");
const presetEl = document.getElementById("preset");
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const runBtn = document.getElementById("run");

let sandbox = null;
let currentRender = null;
let runGeneration = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function setOutput(text, isError) {
  outputEl.textContent = text;
  outputEl.classList.toggle("error", Boolean(isError));
}

function setBusy(busy) {
  runBtn.disabled = busy;
}

// --- syntax highlighting ---------------------------------------------------
// Prism is loaded from a CDN; if that fails the textarea still works.  The
// highlighted <pre> sits behind a transparent textarea and scrolls in sync.
function highlight() {
  const code = codeEl.value;
  const prism = window.Prism;
  if (prism?.languages?.python) {
    highlightCodeEl.innerHTML = prism.highlight(code, prism.languages.python, "python") + "\n";
  } else {
    highlightCodeEl.textContent = code + "\n";
  }
  syncScroll();
}

function syncScroll() {
  highlightEl.scrollTop = codeEl.scrollTop;
  highlightEl.scrollLeft = codeEl.scrollLeft;
}

codeEl.addEventListener("input", highlight);
codeEl.addEventListener("scroll", syncScroll);
// Keep the highlighted layer in sync once Prism finishes loading.
window.addEventListener("load", highlight);
window.addEventListener("DOMContentLoaded", highlight);

// Tab inserts spaces instead of moving focus (Python indentation).
codeEl.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  event.preventDefault();
  const start = codeEl.selectionStart;
  const end = codeEl.selectionEnd;
  codeEl.setRangeText("    ", start, end, "end");
  highlight();
});

// --- execution sandbox -----------------------------------------------------
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

function destroySandbox() {
  if (sandbox?.frame) sandbox.frame.remove();
  sandbox = null;
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

function applyNumpyNames(runtime, numpyNames) {
  const extra = {};
  for (const name of numpyNames) {
    if (name === "*") Object.assign(extra, runtime.np);
    else if (name in runtime.np) extra[name] = runtime.np[name];
  }
  return extra;
}

// --- compiler --------------------------------------------------------------
let rapydscriptPromise = null;

function loadRapydscript() {
  if (window.RapydScript) return Promise.resolve();
  if (rapydscriptPromise === null) {
    // Lazy-load the ~1.4 MB compiler only when the user actually runs code.
    rapydscriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "./rapydscript.js";
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load rapydscript.js"));
      document.head.appendChild(script);
    });
  }
  return rapydscriptPromise;
}

async function compilePython(source) {
  await loadRapydscript();
  const pre = py2js.preprocessImports(source);
  const js = await window.RapydScript.compile(pre.code, "scene.py", {
    operator_overloading: true,
    await_scene_methods: true,
    bare: true,
    beautify: true,
  });
  return { js, numpyNames: pre.numpyNames };
}

// --- run -------------------------------------------------------------------
function stopCurrent() {
  if (currentRender !== null) {
    currentRender.abort();
    currentRender = null;
  }
  destroySandbox();
}

async function run() {
  stopCurrent();
  const generation = ++runGeneration;
  setBusy(true);
  setOutput("Compiling…", false);
  setStatus("Compiling…");
  try {
    const { js, numpyNames } = await compilePython(codeEl.value);
    if (generation !== runGeneration) return;

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
    if (names.length === 0) throw new Error("No Scene subclass found in the code.");
    const sceneClass = scenes[names[0]];

    const controller = new AbortController();
    currentRender = controller;
    setStatus(`Running ${names[0]}…`);
    setOutput(`Running ${names[0]}…`, false);
    await new Promise((resolve) => setTimeout(resolve, 20));

    const result = await wanim.render_scene_to_canvas(sceneClass, stage, {
      quality: QUALITY,
      realtime: true,
      fps: TARGET_FPS,
      drawFrames: true,
      recordFrames: false,
      signal: controller.signal,
    });
    if (generation !== runGeneration) return;
    const { stats } = result;
    setStatus("Done.");
    setOutput(
      `Done: ${stats.frames} frames in ${(stats.totalMs / 1000).toFixed(2)} s.\n` +
        `extract ${stats.extractMs.toFixed(0)} ms · draw ${stats.drawMs.toFixed(0)} ms`,
      false,
    );
  } catch (error) {
    if (generation !== runGeneration) return;
    setStatus("Error.");
    setOutput(error?.stack ?? String(error), true);
    console.error(error);
  } finally {
    if (generation === runGeneration) {
      currentRender = null;
      setBusy(false);
    }
  }
}

function loadPreset(name) {
  const source = PRESETS[name];
  if (source === undefined) return;
  codeEl.value = source;
  highlight();
  setStatus(`Loaded preset: ${name}`);
}

runBtn.addEventListener("click", () => { void run(); });
presetEl.addEventListener("change", () => loadPreset(presetEl.value));

loadPreset(presetEl.value);
setOutput("Press Run to execute the Python scene.");
