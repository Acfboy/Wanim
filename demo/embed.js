// Minimal embed player for the wanim demo.
//
// The Python source travels in the URL fragment (`#c=<token>[&q=...]`),
// which is never sent to the server.  The token is produced by the editor's
// "Embed" dialog (`wanim.build_embed_hash`) and decoded here.
//
// postMessage protocol (see also `demo/index.html` and the dynmath player):
//
//   embed -> parent  { source: "wanim-embed", v: 1, type: "ready" | "scene" | "done" | "error",
//                      sceneName?, stats?, error? }
//   parent -> embed  { type: "wanim:run" | "wanim:stop" | "wanim:set-code",
//                      code?, quality? }
//
// Only messages whose `event.source` is `window.parent` are accepted.
import { configureWasm } from "numpy-ts";

try {
  configureWasm({ maxMemory: 1024 * 1024 * 1024 });
} catch (error) {
  console.warn("numpy-ts configureWasm failed:", error.message);
}

const py2js = await import("py2js");
const wanim = await import("wanim");
wanim.configure_fonts(wanim.DEFAULT_FONT_URLS);

const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const SOURCE = wanim.EMBED_MESSAGE_SOURCE;

const params = new URLSearchParams(location.hash.replace(/^#/, ""));
let quality = params.get("q") || "-qm";
let code = null;
let running = false;
let currentRun = null;
let generation = 0;

function post(type, payload = {}) {
  if (window.parent === window) return;
  window.parent.postMessage({ source: SOURCE, v: 1, type, ...payload }, "*");
}

function showOverlay(text, isError = false) {
  overlay.textContent = text;
  overlay.classList.toggle("error", isError);
  overlay.hidden = false;
}

function hideOverlay() {
  overlay.hidden = true;
}

function messageOf(error) {
  return error instanceof Error ? error.message : String(error);
}

// --- rapydscript compiler ---------------------------------------------------
let rapydscriptPromise = null;

function loadRapydscript() {
  if (window.RapydScript) return Promise.resolve();
  if (rapydscriptPromise === null) {
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

async function stopCurrent() {
  const run = currentRun;
  if (run === null) return;
  currentRun = null;
  running = false;
  generation += 1;
  run.controller.abort();
  try {
    await run.promise;
  } catch {
    /* the aborted run rejects; the caller supersedes it */
  }
}

async function run() {
  if (code === null) {
    const message = "Missing #c=<code> in the URL.";
    showOverlay(message, true);
    post("error", { error: message });
    return;
  }
  await stopCurrent();
  const generationAtStart = ++generation;
  const controller = new AbortController();
  running = true;
  showOverlay("Compiling…");
  try {
    await loadRapydscript();
    if (generationAtStart !== generation) return;
    const promise = py2js.runPythonOnCanvas(code, canvas, {
      quality,
      realtime: true,
      fps: 30,
      drawFrames: true,
      recordFrames: false,
      signal: controller.signal,
      onScene: (name) => {
        if (generationAtStart !== generation) return;
        // Compilation finished and the scene is about to render: drop the
        // status text so interactive scenes are not covered by a perpetual
        // "Running …" overlay. Errors still show through the catch below.
        hideOverlay();
        post("scene", { sceneName: name });
      },
    });
    currentRun = { controller, promise };
    const result = await promise;
    if (generationAtStart !== generation) return;
    currentRun = null;
    running = false;
    hideOverlay();
    post("done", { sceneName: result.sceneName, stats: result.stats });
  } catch (error) {
    if (generationAtStart !== generation) return;
    currentRun = null;
    running = false;
    if (controller.signal.aborted) return;
    const message = messageOf(error);
    showOverlay(message, true);
    post("error", { error: message });
  }
}

window.addEventListener("message", (event) => {
  // Only the hosting page may drive the player.
  if (event.source !== window.parent) return;
  const data = event.data;
  if (data === null || typeof data !== "object") return;
  switch (data.type) {
    case "wanim:run":
      void run();
      break;
    case "wanim:stop":
      void stopCurrent();
      break;
    case "wanim:set-code":
      if (typeof data.code !== "string") return;
      code = data.code;
      if (typeof data.quality === "string" && data.quality !== "") quality = data.quality;
      // Re-run only when the player is already running; otherwise the host
      // follows up with `wanim:run`.
      if (running) void run();
      break;
    default:
      break;
  }
});

post("ready", {});

const token = params.get("c");
if (token === null || token === "") {
  const message = "Missing #c=<code> in the URL.";
  showOverlay(message, true);
  post("error", { error: message });
} else {
  try {
    code = await wanim.decode_embed_code(token);
  } catch (error) {
    const message = `Could not decode the embedded code: ${messageOf(error)}`;
    showOverlay(message, true);
    post("error", { error: message });
  }
  if (code !== null) void run();
}
