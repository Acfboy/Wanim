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
  opening: `from manim import *

class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \\LaTeX")
        basel = MathTex(r"\\sum_{n=1}^\\infty \\frac{1}{n^2} = \\frac{\\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in basel]),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\\\ applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ]
                )
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()
`,
  formula: `from manim import *

class MovingFrameBox(Scene):
    def construct(self):
        text = MathTex(
            "\\\\frac{d}{dx}f(x)g(x)=", "f(x)\\\\frac{d}{dx}g(x)", "+",
            "g(x)\\\\frac{d}{dx}f(x)"
        )
        self.play(Write(text))
        framebox1 = SurroundingRectangle(text[1], buff=.1)
        framebox2 = SurroundingRectangle(text[3], buff=.1)
        self.play(
            Create(framebox1),
        )
        self.wait()
        self.play(
            ReplacementTransform(framebox1, framebox2),
        )
        self.wait()
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

class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 24
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )

        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1, stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()
        self.add(axes, gauss_plane)
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

// --- compiler --------------------------------------------------------------
let rapydscriptPromise = null;

function loadRapydscript() {
  if (window.RapydScript) return Promise.resolve();
  if (rapydscriptPromise === null) {
    // Lazy-load the ~1.4 MB compiler only when the user actually runs code.
    rapydscriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "../dist/rapydscript.js";
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load rapydscript.js"));
      document.head.appendChild(script);
    });
  }
  return rapydscriptPromise;
}

// --- run -------------------------------------------------------------------
function stopCurrent() {
  if (currentRender !== null) {
    currentRender.abort();
    currentRender = null;
  }
}

async function run() {
  stopCurrent();
  const generation = ++runGeneration;
  setBusy(true);
  setOutput("Compiling…", false);
  setStatus("Compiling…");
  try {
    // The patched rapydscript compiler is loaded lazily on first run.
    await loadRapydscript();
    if (generation !== runGeneration) return;

    const controller = new AbortController();
    currentRender = controller;
    const { stats } = await py2js.runPythonOnCanvas(codeEl.value, stage, {
      quality: QUALITY,
      realtime: true,
      fps: TARGET_FPS,
      drawFrames: true,
      recordFrames: false,
      signal: controller.signal,
      onScene: (name) => {
        setStatus(`Running ${name}…`);
        setOutput(`Running ${name}…`, false);
      },
    });
    if (generation !== runGeneration) return;
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
