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

// Configure the text font chain before any scene is built. `DEFAULT_FONT_URLS`
// is the free CDN pair (Tinos for Latin, Noto Serif SC for Chinese); embedders
// pass their own URLs/paths here instead, e.g. configure_fonts(["/fonts/a.ttf"]).
const wanim = await import("wanim");
wanim.configure_fonts(wanim.DEFAULT_FONT_URLS);

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
  pointcloud: `from manim import *

class PointCloudDemo(Scene):
    def construct(self):
        gradient = PointCloudDot(LEFT * 3.5, radius=1.6, stroke_width=2)
        gradient.set_color_by_gradient(BLUE, GREEN)
        thick = PointCloudDot(RIGHT * 3.5, radius=1.6, stroke_width=4, color=RED)
        wave = PMobject(stroke_width=2)
        points = [
            [-6 + i * 0.05, 1.2 * np.sin(-6 + i * 0.05), 0]
            for i in range(241)
        ]
        wave.add_points(points, color=YELLOW)
        wave.shift(DOWN * 2.2)

        self.add(gradient, thick, wave)
        self.wait(0.5)
        self.play(wave.animate.shift(UP * 3.2), run_time=2)
        self.wait()
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
  interactive: `from manim import *

class InteractiveDemo(InteractiveScene):
    def blueprint(self):
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2, 1],
            x_length=6,
            y_length=3,
            axis_config={"include_numbers": True},
        )
        axes.to_edge(LEFT)
        slider = Slider(
            start=axes.c2p(0, 0),
            end=axes.c2p(4, 0),
            value_range=(0.5, 3),
            init_value=1,
            color=YELLOW,
            stroke_width=6,
        )
        circle = Circle(radius=1, color=BLUE)
        circle.add_reaction(lambda m: m.set_width(slider.get_value()))
        self.slider = slider
        self.circle = circle
        self.add(axes, slider, circle)

        num = InputDecimalNumber(2, num_decimal_places=1, color=YELLOW)
        num.to_corner(UL)
        square = Square(side_length=2, color=YELLOW)
        square.add_reaction(lambda m: m.set_width(num.get_value()))
        toggler = Switch(True)
        toggler.to_corner(UR).shift(DOWN * 0.5)

        def tint(m):
            m.set_color(GREEN if toggler.get_value() else RED)

        square.add_reaction(tint)
        self.add(num, toggler, square)

    def construct(self):
        self.play(FadeOut(self.circle))
`,
  interactive3d: `from manim import *

class InteractiveThreeDDemo(InteractiveThreeDScene):
    def blueprint(self):
        # 空白处按住拖动 = 旋转相机，滚轮 / 触控板 = 缩放
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        axes = ThreeDAxes()
        sphere = Sphere(radius=1.4, color=BLUE, fill_opacity=0.85)
        sphere.add_reaction(lambda m: m.move_to([0, 0, height_slider.get_value()]))

        # 控件用 add_fixed_in_frame_mobjects 固定在屏幕上，
        # 这样指针命中坐标不受相机旋转影响，drag 球体时不会误触 orbit
        height_slider = Slider(
            start=[-5.0, -3.0, 0],
            end=[-1.5, -3.0, 0],
            value_range=(-1.5, 1.5),
            init_value=0,
            color=YELLOW,
        )
        height_label = Text("sphere z", font_size=28).next_to(height_slider, UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(height_slider, height_label)

        hint = Text("拖动旋转 / 滚轮缩放", font_size=30)
        hint.to_corner(UL)
        self.add_fixed_in_frame_mobjects(hint)

        self.hint = hint
        self.sphere = sphere
        self.add(axes, sphere)

    def construct(self):
        # 动画期间 orbit 暂停（与交互控件同一套 phase 规则）
        self.play(FadeOut(self.hint))
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

let currentRun = null;
let runGeneration = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function setOutput(text, isError) {
  outputEl.textContent = text;
  outputEl.classList.toggle("error", Boolean(isError));
}

function setBusy(busy) {
  // Interactive scenes never resolve on their own (they run until the user
  // finishes the story or starts another run), so the Run button stays
  // clickable: pressing it again aborts the current run and starts over.
  runBtn.textContent = busy ? "运行中… / Running" : "运行 / Run";
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
async function stopCurrent() {
  const run = currentRun;
  if (run === null) return;
  currentRun = null;
  run.controller.abort();
  // Wait for the previous run (and its interactive scene loop) to fully stop
  // before the next one starts, so their schedulers/hosts don't overlap.
  try {
    await run.promise;
  } catch {
    /* the aborted run may reject; ignore */
  }
}

async function run() {
  await stopCurrent();
  const generation = ++runGeneration;
  setBusy(true);
  setOutput("Compiling…", false);
  setStatus("Compiling…");
  try {
    // The patched rapydscript compiler is loaded lazily on first run.
    await loadRapydscript();
    if (generation !== runGeneration) return;

    const controller = new AbortController();
    const promise = py2js.runPythonOnCanvas(codeEl.value, stage, {
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
    currentRun = { controller, promise };
    const { stats } = await promise;
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
      currentRun = null;
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

// --- embed ------------------------------------------------------------------
// Builds a paste-ready <iframe> pointing at ./embed.html with the current code
// compressed into the URL fragment (`#c=<token>`).  The fragment never reaches
// the server, so the code stays client-side.
const embedBtn = document.getElementById("embed");
const embedModal = document.getElementById("embed-modal");
const embedWidthEl = document.getElementById("embed-width");
const embedHeightEl = document.getElementById("embed-height");
const embedQualityEl = document.getElementById("embed-quality");
const embedSnippetEl = document.getElementById("embed-snippet");
const embedWarningEl = document.getElementById("embed-warning");
const embedStatusEl = document.getElementById("embed-status");
const embedCopyEl = document.getElementById("embed-copy");
const embedCloseEl = document.getElementById("embed-close");

// Browsers and blogging platforms vary; past this length a fragment URL is
// likely to be truncated somewhere in the pipeline.
const MAX_EMBED_URL = 30000;
let embedGeneration = 0;

async function generateEmbed() {
  const generation = ++embedGeneration;
  embedStatusEl.textContent = "生成中… / Generating…";
  embedSnippetEl.value = "";
  embedWarningEl.hidden = true;
  const width = Math.max(1, Math.round(Number(embedWidthEl.value) || 960));
  const height = Math.max(1, Math.round(Number(embedHeightEl.value) || 540));
  try {
    const hash = await wanim.build_embed_hash(codeEl.value, {
      quality: embedQualityEl.value,
    });
    const url = new URL("embed.html", location.href).href + hash;
    const snippet = wanim.build_embed_iframe(url, { width, height, title: "wanim scene" });
    if (generation !== embedGeneration) return;
    embedSnippetEl.value = snippet;
    if (url.length > MAX_EMBED_URL) {
      embedWarningEl.textContent =
        `URL 约 ${url.length} 字符，超过 ${MAX_EMBED_URL}；部分平台可能截断。建议缩短代码或降低画质。`;
      embedWarningEl.hidden = false;
    }
    embedStatusEl.textContent = `URL ${url.length} 字符`;
  } catch (error) {
    if (generation !== embedGeneration) return;
    embedStatusEl.textContent = "生成失败 / Failed";
    setOutput(error?.stack ?? String(error), true);
    console.error(error);
  }
}

function openEmbed() {
  embedModal.hidden = false;
  void generateEmbed();
}

function closeEmbed() {
  embedModal.hidden = true;
}

embedBtn.addEventListener("click", openEmbed);
embedCloseEl.addEventListener("click", closeEmbed);
embedModal.addEventListener("click", (event) => {
  if (event.target === embedModal) closeEmbed();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !embedModal.hidden) closeEmbed();
});
for (const el of [embedWidthEl, embedHeightEl, embedQualityEl]) {
  el.addEventListener("change", () => { void generateEmbed(); });
}
embedCopyEl.addEventListener("click", async () => {
  const text = embedSnippetEl.value;
  if (text === "") return;
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    /* clipboard API unavailable (http / permissions): select the text instead */
  }
  if (!copied) {
    embedSnippetEl.focus();
    embedSnippetEl.select();
    try {
      copied = document.execCommand("copy");
    } catch {
      copied = false;
    }
  }
  embedStatusEl.textContent = copied ? "已复制 / Copied" : "复制失败，请手动选择文本";
});
