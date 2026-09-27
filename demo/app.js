import { configureWasm } from "numpy-ts";

// The numpy-ts WebAssembly heap is fixed at first use and silently falls back
// to slow JS-backed arrays when it fills.  wanim's mobject interpolation churns
// through a lot of temporary arrays, so give it a generous heap up front.
// (Must run before wanim is imported: wanim's module evaluation performs array
// operations.)
try {
  configureWasm({ maxMemory: 1024 * 1024 * 1024 });
} catch (error) {
  console.warn("numpy-ts configureWasm failed (heap already initialised):", error.message);
}

const wanim = await import("wanim");

const codeEl = document.getElementById("code");
const qualityEl = document.getElementById("quality");
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const previewBtn = document.getElementById("preview");
const benchmarkBtn = document.getElementById("benchmark");
const stopBtn = document.getElementById("stop");
const loopEl = document.getElementById("loop");
const mjsonFile = document.getElementById("mjson-file");

let player = null;

const PRESETS = {
  basic: "../examples/basic_scene.js",
  opening: "../examples/opening_manim.js",
  sine: "../examples/sine_curve_unit_circle.js",
};

function setStatus(text) {
  statusEl.textContent = text;
}

function setBusy(busy) {
  previewBtn.disabled = busy;
  benchmarkBtn.disabled = busy;
  stopBtn.disabled = !busy;
}

async function loadPreset(name) {
  const url = PRESETS[name];
  if (!url) return;
  const response = await fetch(url);
  codeEl.value = await response.text();
  setStatus(`Loaded preset: ${name}`);
}

async function loadSceneClasses(code) {
  const blob = new Blob([code], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  try {
    const module = await import(/* @vite-ignore */ url);
    const classes = wanim.get_scene_classes_from_module(module);
    if (classes.length === 0) {
      throw new Error("No Scene subclasses found in the code.");
    }
    return classes;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function stopPlayback() {
  if (player) {
    player.stop();
    player = null;
  }
  stopBtn.disabled = true;
}

async function preview() {
  stopPlayback();
  setBusy(true);
  setStatus("Running scene (recording frames)…");
  statsEl.textContent = "";
  await new Promise((resolve) => setTimeout(resolve, 30));
  try {
    const classes = await loadSceneClasses(codeEl.value);
    const sceneClass = classes[0];
    const result = wanim.render_scene_to_canvas(sceneClass, stage, {
      quality: qualityEl.value,
      drawFrames: false,
      recordFrames: true,
    });
    const { stats, frames, meta } = result;
    statsEl.textContent =
      `scene: ${sceneClass.name}\n` +
      `frames: ${stats.frames} (${stats.recordCount} records), plays: ${meta.num_plays}\n` +
      `generation: ${stats.totalMs.toFixed(0)} ms ` +
      `(${(stats.frames / (stats.totalMs / 1000)).toFixed(1)} gen fps)\n` +
      `extract: ${stats.extractMs.toFixed(0)} ms · serialize: ${stats.serializeMs.toFixed(0)} ms\n` +
      `playing back at ${meta.frame_rate} fps…`;
    setStatus("Playing back recorded frames.");
    player = wanim.play_frames(stage, meta, frames, {
      loop: loopEl.checked,
      fps: meta.frame_rate,
      onEnd: () => { setStatus("Playback finished."); stopPlayback(); },
    });
    stopBtn.disabled = false;
  } catch (error) {
    setStatus(`Error: ${error.message}`);
    console.error(error);
  } finally {
    setBusy(false);
  }
}

async function benchmark() {
  stopPlayback();
  setBusy(true);
  setStatus("Benchmarking: drawing every frame directly (no serialization)…");
  statsEl.textContent = "";
  await new Promise((resolve) => setTimeout(resolve, 30));
  try {
    const classes = await loadSceneClasses(codeEl.value);
    const sceneClass = classes[0];
    const result = wanim.render_scene_to_canvas(sceneClass, stage, {
      quality: qualityEl.value,
      drawFrames: true,
      recordFrames: false,
      onProgress: (info) => {
        statsEl.textContent = `rendering… ${info.frames} frames, ${(info.elapsedMs / 1000).toFixed(1)}s`;
      },
    });
    const { stats, meta } = result;
    const playMs = stats.playMs || (stats.totalMs - 0);
    const playFps = playMs > 0 ? stats.frames / (playMs / 1000) : Infinity;
    const setupMs = Math.max(0, stats.totalMs - playMs);
    const totalFps = stats.frames / (stats.totalMs / 1000);
    const drawFps = stats.drawMs > 0 ? stats.frames / (stats.drawMs / 1000) : Infinity;
    statsEl.textContent =
      `scene: ${sceneClass.name}  quality: ${qualityEl.value}\n` +
      `frames: ${stats.frames}, plays: ${meta.num_plays}\n` +
      `render loop (play): ${stats.playMs.toFixed(0)} ms  →  ${playFps.toFixed(1)} fps\n` +
      `  canvas draw inside loop: ${stats.drawMs.toFixed(0)} ms  →  ${drawFps.toFixed(1)} fps\n` +
      `one-time setup (construct/Tex): ${setupMs.toFixed(0)} ms\n` +
      `total wall: ${stats.totalMs.toFixed(0)} ms  →  ${totalFps.toFixed(1)} fps`;
    setStatus("Benchmark complete.");
    console.log("[wanim benchmark]", stats);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
    console.error(error);
  } finally {
    setBusy(false);
  }
}

function playMjsonText(text, name) {
  stopPlayback();
  try {
    const document = wanim.parse_mjson(text);
    setStatus(`Loaded ${name}: ${document.frames.length} records, ${document.meta.num_frames} frames @ ${document.meta.frame_rate} fps.`);
    statsEl.textContent = "";
    player = new wanim.MJSONCanvasPlayer(stage, document).play({
      loop: loopEl.checked,
      fps: document.meta.frame_rate,
    });
    stopBtn.disabled = false;
  } catch (error) {
    setStatus(`Error parsing ${name}: ${error.message}`);
  }
}

previewBtn.addEventListener("click", preview);
benchmarkBtn.addEventListener("click", benchmark);
stopBtn.addEventListener("click", () => { stopPlayback(); setStatus("Stopped."); });

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => loadPreset(button.dataset.preset));
});

mjsonFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  setStatus(`Reading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`);
  const text = await file.text();
  playMjsonText(text, file.name);
});

// Default preset.
loadPreset("basic").catch((error) => setStatus(`Failed to load preset: ${error.message}`));