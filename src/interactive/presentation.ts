/**
 * `ScenePresentation`: play several `Scene` classes in order as one
 * presentation, with slide-style navigation.
 *
 * The task description it implements:
 *
 *  - scenes are played in declaration order, and a storyboard scene
 *    (`InteractiveScene`) contributes its `(blueprint, construct)` phases;
 *  - without `slides_mode` only `blueprint` phases pause for the user, every
 *    other phase plays and continues on its own;
 *  - with `slides_mode` every phase pauses, and the previous-page /
 *    replay-page controls become available;
 *  - "previous" rewinds to the *previous segment* (the page before the current
 *    one, possibly in the previous scene), restoring the interactive controls'
 *    values so a blueprint remembers the user's input;
 *  - "replay" re-runs the current segment: a blueprint is rebuilt (clearing
 *    the user's input) and a construct replays its animation;
 *  - "restart" (driven from JS only) rebuilds the whole presentation.
 *
 * The player is renderer/DOM agnostic: it drives a caller-provided
 * {@link InteractiveHost} and creates scenes through a caller-provided factory
 * (so it can be unit-tested headless and can also drive one pre-built scene).
 * A presentation owns its pacing: scenes get their scheduler attached
 * explicitly, so two presentations on one page never interfere.
 *
 * Pages are rebuilt from the start of their scene whenever the player moves
 * backwards, because a phase function is the only source of truth for the
 * scene state.  The rebuild is fast-forwarded (animations computed without
 * drawing or waiting); only the page being entered normally is drawn.
 */
import type { Scene } from "../scene/scene.js";
import type { SceneConstructor } from "../renderer/scene_runner.js";
import type { InteractiveHost } from "./host.js";
import { is_thenable } from "../utils/scheduler.js";

/** The kind of half a story step is made of. */
export type PresentationPhaseKind = "blueprint" | "construct";

/** One replayable page: a named phase function and its kind. */
export interface PresentationPhase {
  kind: PresentationPhaseKind;
  run: () => unknown;
}

/** What a paused page can ask the presentation to do next. */
export type PresentationAction = "next" | "previous" | "replay" | "restart" | "abort";

/** Observable presentation position/state, reported through `onStateChange`. */
export interface PresentationState {
  /** Index of the active scene (0-based). */
  sceneIndex: number;
  /** Number of scenes in the presentation. */
  sceneCount: number;
  /** Class name of the active scene. */
  sceneName: string;
  /** Index of the paused phase (0-based). */
  phaseIndex: number;
  /** Number of phases of the active scene. */
  phaseCount: number;
  /** Kind of the paused phase. */
  phaseKind: PresentationPhaseKind;
  /** Whether the run is in slides mode. */
  slidesMode: boolean;
  /** Whether the presentation finished (non-slides mode). */
  done: boolean;
  /** Whether the next-page control is available. */
  canNext: boolean;
  /** Whether the previous-page control is available. */
  canPrevious: boolean;
  /** Whether the replay-page control is available. */
  canReplay: boolean;
}

/**
 * Structural view of the stepwise methods an `InteractiveScene` exposes for
 * the player.  Duck typing (the `is_interactive_scene` marker) keeps this
 * module free of a runtime import cycle with `interactive_scene.ts`.
 */
interface InteractiveSceneLike {
  is_interactive_scene: true;
  presentation_phases(): PresentationPhase[];
  presentation_run_phase(phase: PresentationPhase): Promise<void>;
  presentation_frame(): void;
  presentation_reset(): void;
  presentation_control_values(): unknown[];
  presentation_set_control_values(values: unknown[]): void;
  attach_host(host: InteractiveHost): void;
  detach_host(): void;
  requested_action(): PresentationAction | null;
}

function as_interactive(scene: Scene): InteractiveSceneLike | null {
  if ((scene as { is_interactive_scene?: boolean }).is_interactive_scene !== true) return null;
  return scene as unknown as InteractiveSceneLike;
}

export interface ScenePresentationOptions {
  /** Environment services (pointer events, buttons, frame pacing). */
  host: InteractiveHost;
  /** Create (once per activation) the scene instance for a scene class. */
  createScene: (sceneClass: SceneConstructor) => Scene;
  /**
   * Run in slides mode: pause after every phase, expose previous/replay.
   * Defaults to `false` (only blueprints pause).
   */
  slidesMode?: boolean;
  /**
   * Dispose hook called when the player drops a scene (leaving it, restarting
   * or finishing).  The player itself only detaches the host.
   */
  disposeScene?: (scene: Scene) => void;
  /** Called right after a scene instance is prepared and ready to run. */
  onSceneActivated?: (scene: Scene, sceneIndex: number, sceneClass: SceneConstructor) => void;
  /** Called whenever the observable state changes. */
  onStateChange?: (state: PresentationState) => void;
  /** Called when the presentation finishes (non-slides) or is aborted. */
  onFinished?: (state: PresentationState) => void;
}

/** Play a list of scene classes as one navigable presentation. */
export class ScenePresentation {
  readonly sceneClasses: SceneConstructor[];
  readonly host: InteractiveHost;
  readonly slidesMode: boolean;

  private readonly createScene: (sceneClass: SceneConstructor) => Scene;
  private readonly disposeScene: ((scene: Scene) => void) | null;
  private readonly onSceneActivated: ScenePresentationOptions["onSceneActivated"];
  private readonly onStateChange: ScenePresentationOptions["onStateChange"];
  private readonly onFinished: ScenePresentationOptions["onFinished"];

  private scene: Scene | null = null;
  private sceneIndex = -1;
  private phases: PresentationPhase[] = [];
  private phaseIndex = 0;
  private pending: PresentationAction | null = null;
  private abortRequested = false;
  private done = false;
  private running = false;
  private readonly disposers: Array<() => void> = [];
  /**
   * Remembered interactive-control values per blueprint page, keyed
   * `sceneIndex:phaseIndex`.  This is what makes "previous" return to a
   * blueprint with the user's inputs intact.
   */
  private readonly controlValues = new Map<string, unknown[]>();

  constructor(sceneClasses: SceneConstructor[], options: ScenePresentationOptions) {
    this.sceneClasses = [...sceneClasses];
    this.host = options.host;
    this.slidesMode = options.slidesMode ?? false;
    this.createScene = options.createScene;
    this.disposeScene = options.disposeScene ?? null;
    this.onSceneActivated = options.onSceneActivated;
    this.onStateChange = options.onStateChange;
    this.onFinished = options.onFinished;
  }

  /** The current observable state (meaningful while paused / after running). */
  get state(): PresentationState {
    const sceneClass = this.sceneClasses[this.sceneIndex];
    const phase = this.phases[this.phaseIndex];
    const done = this.done;
    return {
      sceneIndex: Math.max(0, this.sceneIndex),
      sceneCount: this.sceneClasses.length,
      sceneName: sceneClass?.name ?? "",
      phaseIndex: this.phaseIndex,
      phaseCount: this.phases.length,
      phaseKind: phase?.kind ?? "construct",
      slidesMode: this.slidesMode,
      done,
      canNext: !done && this.has_current() && this.can_advance(),
      canPrevious: !done && this.slidesMode && this.can_step_back(),
      canReplay: !done && this.slidesMode && this.has_current(),
    };
  }

  // --- external controls -----------------------------------------------------

  /** Advance to the next page (finishes at the last page in slides mode). */
  next(): void {
    this.request("next");
  }

  /** Rewind to the previous page (possibly the previous scene's last page). */
  previous(): void {
    if (!this.slidesMode) return;
    this.request("previous");
  }

  /** Re-run the current page (blueprint: clear inputs, construct: replay). */
  replay(): void {
    if (!this.slidesMode) return;
    this.request("replay");
  }

  /** Rebuild the whole presentation from its first page. */
  restart(): void {
    if (!this.running || this.done) {
      void this.run();
      return;
    }
    this.request("restart");
  }

  /** Abort the run at the next frame boundary. */
  abort(): void {
    this.abortRequested = true;
    this.pending = null;
  }

  /** Whether the presentation is currently running. */
  get active(): boolean {
    return this.running && !this.done;
  }

  private request(action: PresentationAction): void {
    if (!this.running || this.done) return;
    this.pending = action;
  }

  // --- run loop ---------------------------------------------------------------

  /**
   * Run the presentation until it finishes (non-slides) or is aborted.  A
   * slides-mode run parks on its last page until `restart()`/`abort()` is
   * called from JS.
   */
  async run(): Promise<void> {
    if (this.running) return;
    if (this.sceneClasses.length === 0) return;
    this.running = true;
    this.done = false;
    this.abortRequested = false;
    this.pending = null;
    this.controlValues.clear();
    this.install_controls();
    try {
      await this.activate_scene(0);
      await this.enter_phase(0, false);
      while (!this.abortRequested && !this.done) {
        const phase = this.phases[this.phaseIndex]!;
        if (!this.should_pause(phase.kind)) {
          if (!(await this.advance_phase())) {
            this.done = true;
            break;
          }
          continue;
        }
        this.show_controls(phase.kind);
        this.emit_state();
        const action = await this.wait_for_action();
        this.hide_controls();
        if (this.abortRequested) break;
        switch (action) {
          case "abort":
            this.abortRequested = true;
            break;
          case "next":
            if (phase.kind === "blueprint") this.record_control_values();
            if (!(await this.advance_phase())) {
              this.done = true;
            }
            break;
          case "previous":
            if (phase.kind === "blueprint") this.record_control_values();
            await this.step_back();
            break;
          case "replay":
            await this.replay_phase();
            break;
          case "restart":
            this.controlValues.clear();
            await this.activate_scene(0);
            await this.enter_phase(0, false);
            break;
        }
      }
    } finally {
      this.hide_controls();
      const finishedState = this.state;
      this.dispose_current_scene();
      this.uninstall_controls();
      this.running = false;
      this.emit_state();
      this.onFinished?.(finishedState);
    }
  }

  /**
   * Register the overlay buttons and keyboard shortcuts.  The labels double as
   * the buttons' accessible names; "进入下一步" is also what external tooling
   * (the slides harness) looks for to drive a story.
   */
  private install_controls(): void {
    this.hide_controls();
    this.disposers.push(this.host.addButton("上一页", () => this.previous(), "previous"));
    this.disposers.push(this.host.addButton("重播本段", () => this.replay(), "replay"));
    this.disposers.push(this.host.addButton("进入下一步", () => this.next(), "next"));
    this.disposers.push(
      this.host.onKey({
        next: () => this.next(),
        previous: () => this.previous(),
        replay: () => this.replay(),
      }),
    );
  }

  private uninstall_controls(): void {
    for (const dispose of this.disposers.splice(0)) {
      try {
        dispose();
      } catch {
        /* ignore */
      }
    }
  }

  // --- scene lifecycle --------------------------------------------------------

  private async activate_scene(index: number): Promise<void> {
    this.dispose_current_scene();
    const sceneClass = this.sceneClasses[index]!;
    const scene = this.createScene(sceneClass);
    this.scene = scene;
    this.sceneIndex = index;
    this.reset_scene(scene);
    scene.setup();
    this.phases = this.phases_of(scene);
    this.phaseIndex = 0;
    const interactive = as_interactive(scene);
    if (interactive !== null) interactive.attach_host(this.host);
    this.apply_control_theme(scene);
    this.onSceneActivated?.(scene, index, sceneClass);
    this.emit_state();
  }

  private dispose_current_scene(): void {
    const scene = this.scene;
    if (scene === null) return;
    this.scene = null;
    as_interactive(scene)?.detach_host();
    this.disposeScene?.(scene);
  }

  private phases_of(scene: Scene): PresentationPhase[] {
    const interactive = as_interactive(scene);
    if (interactive !== null) return interactive.presentation_phases();
    // A plain `Scene` (or `ThreeDScene`, ...) is a single construct page.
    return [{ kind: "construct", run: () => scene.construct() }];
  }

  /** Drop all scene state so the phase functions can rebuild from scratch. */
  private reset_scene(scene: Scene): void {
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.presentation_reset();
      return;
    }
    scene.mobjects = [];
    scene.foreground_mobjects = [];
    scene.moving_mobjects = [];
    scene.static_mobjects = [];
    scene.animations = [];
    scene.updaters = [];
    scene.last_t = 0;
    const renderer = scene.renderer as {
      clear_screen?: () => void;
      num_plays?: number;
      time?: number;
    } | null;
    if (renderer !== null && renderer !== undefined) {
      renderer.clear_screen?.();
      renderer.num_plays = 0;
      renderer.time = 0;
    }
  }

  // --- page execution ----------------------------------------------------------

  /**
   * Run the phase at `index` from the current (live) state and settle on it:
   * blueprints restore remembered control values, one frame is drawn.
   */
  private async enter_phase(index: number, replay: boolean): Promise<void> {
    const phase = this.phases[index]!;
    await this.run_phase(phase);
    if (phase.kind === "blueprint" && !replay) this.restore_control_values(index);
    this.settle();
  }

  /**
   * Rebuild the current scene from its start and fast-forward to `target`:
   * earlier phases run without drawing and without waiting, the target
   * `construct` also completes instantly unless `replay` is set (which lets
   * the animation play), and target blueprints restore their values.
   */
  private async rewind_to(target: number, replay: boolean): Promise<void> {
    const scene = this.scene;
    if (scene === null) return;
    this.reset_scene(scene);
    scene.setup();
    this.phases = this.phases_of(scene);
    for (let index = 0; index <= target; index += 1) {
      const phase = this.phases[index]!;
      const isTarget = index === target;
      const skip = !isTarget || (phase.kind === "construct" && !replay);
      if (skip) await this.run_phase_skipping(phase);
      else await this.run_phase(phase);
      if (phase.kind === "blueprint" && (!isTarget || !replay)) {
        this.restore_control_values(index);
      }
    }
    this.phaseIndex = target;
    this.settle();
  }

  private async replay_phase(): Promise<void> {
    const phase = this.phases[this.phaseIndex]!;
    if (phase.kind === "blueprint") {
      // Replaying a blueprint starts from the page's default state: drop the
      // remembered values so the rebuilt controls show their initial values.
      this.controlValues.delete(this.value_key(this.phaseIndex));
    }
    await this.rewind_to(this.phaseIndex, true);
  }

  private async advance_phase(): Promise<boolean> {
    if (this.phaseIndex + 1 < this.phases.length) {
      this.phaseIndex += 1;
      await this.enter_phase(this.phaseIndex, false);
      return true;
    }
    if (this.sceneIndex + 1 < this.sceneClasses.length) {
      await this.activate_scene(this.sceneIndex + 1);
      await this.enter_phase(0, false);
      return true;
    }
    return false;
  }

  private async step_back(): Promise<boolean> {
    if (this.phaseIndex > 0) {
      await this.rewind_to(this.phaseIndex - 1, false);
      return true;
    }
    if (this.sceneIndex > 0) {
      await this.activate_scene(this.sceneIndex - 1);
      this.phaseIndex = this.phases.length - 1;
      await this.rewind_to(this.phaseIndex, false);
      return true;
    }
    return false;
  }

  private async run_phase(phase: PresentationPhase): Promise<void> {
    const scene = this.scene;
    if (scene === null || this.abortRequested) return;
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      await interactive.presentation_run_phase(phase);
      return;
    }
    const result = phase.run.call(scene);
    if (is_thenable(result)) await result;
  }

  /**
   * Run a phase instantly: the renderer neither draws nor waits, so the phase
   * only advances the state (used while rebuilding earlier pages).
   */
  private async run_phase_skipping(phase: PresentationPhase): Promise<void> {
    const renderer = this.scene?.renderer as
      | { drawFrames?: boolean; set_skip_animations?: (skip: boolean) => void }
      | null
      | undefined;
    const savedDraw = renderer?.drawFrames;
    const setSkip = renderer?.set_skip_animations;
    setSkip?.call(renderer, true);
    if (renderer !== null && renderer !== undefined) renderer.drawFrames = false;
    try {
      await this.run_phase(phase);
    } finally {
      setSkip?.call(renderer, false);
      if (renderer !== null && renderer !== undefined && savedDraw !== undefined) {
        renderer.drawFrames = savedDraw;
      }
    }
  }

  /** Draw one frame in the settled phase and notify state listeners. */
  private settle(): void {
    if (this.abortRequested) return;
    this.tick();
    this.emit_state();
  }

  private tick_plain_scene(scene: Scene): void {
    const frameRate = (scene.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    scene.update_mobjects(dt);
    scene.update_self(dt);
    if (typeof scene.renderer?.render === "function") {
      scene.renderer.render(scene, scene.time, scene.moving_mobjects);
    }
  }

  /** Tick the active page; called by the wait loop. */
  private tick(): void {
    const scene = this.scene;
    if (scene === null) return;
    const interactive = as_interactive(scene);
    if (interactive !== null) interactive.presentation_frame();
    else this.tick_plain_scene(scene);
  }

  // --- waiting / actions -------------------------------------------------------

  private should_pause(kind: PresentationPhaseKind): boolean {
    if (!this.host.live) return false;
    if (kind === "blueprint") return true;
    return this.slidesMode;
  }

  private async wait_for_action(): Promise<PresentationAction> {
    for (;;) {
      const action = this.take_action();
      if (action !== null) return action;
      if (this.abortRequested) return "abort";
      if (this.scene?.run_aborted()) return "abort";
      this.tick();
      await this.host.nextFrame();
    }
  }

  private take_action(): PresentationAction | null {
    if (this.pending !== null) {
      const action = this.pending;
      this.pending = null;
      return action;
    }
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    const action = interactive?.requested_action() ?? null;
    if (action === null) return null;
    // Scene-level JS calls (`scene.previous_step()`) honour the same slides
    // mode gate as the player's own methods.
    if (!this.slidesMode && (action === "previous" || action === "replay")) return null;
    return action;
  }

  // --- control values ----------------------------------------------------------

  private value_key(phaseIndex: number): string {
    return `${this.sceneIndex}:${phaseIndex}`;
  }

  private record_control_values(): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    if (interactive === null) return;
    this.controlValues.set(this.value_key(this.phaseIndex), interactive.presentation_control_values());
  }

  private restore_control_values(phaseIndex: number): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    if (interactive === null) return;
    const values = this.controlValues.get(this.value_key(phaseIndex));
    if (values !== undefined) interactive.presentation_set_control_values(values);
  }

  // --- controls (buttons / state) ----------------------------------------------

  private has_current(): boolean {
    return this.scene !== null && this.sceneIndex >= 0;
  }

  private can_advance(): boolean {
    if (this.phaseIndex + 1 < this.phases.length) return true;
    if (this.sceneIndex + 1 < this.sceneClasses.length) return true;
    // A trailing blueprint page still needs an affordance to finish the story.
    return this.phases[this.phaseIndex]?.kind === "blueprint";
  }

  private can_step_back(): boolean {
    return this.phaseIndex > 0 || this.sceneIndex > 0;
  }

  private show_controls(_kind: PresentationPhaseKind): void {
    // Outside slides mode the only control is "next" (and only blueprint
    // phases ever pause): no previous/replay affordance appears.
    const slides = this.slidesMode;
    this.host.setButtonsEnabled(true);
    this.host.setButtonVisible?.("previous", slides && this.can_step_back());
    this.host.setButtonVisible?.("replay", slides);
    this.host.setButtonVisible?.("next", this.can_advance());
  }

  private hide_controls(): void {
    this.host.setButtonsEnabled(false);
    this.host.setButtonVisible?.("previous", false);
    this.host.setButtonVisible?.("replay", false);
    this.host.setButtonVisible?.("next", false);
  }

  private emit_state(): void {
    this.onStateChange?.(this.state);
  }

  /**
   * Tell the host whether the current backdrop is dark, so an overlaid control
   * bar can pick a readable icon colour on light and dark backgrounds alike.
   */
  private apply_control_theme(scene: Scene): void {
    if (typeof this.host.setControlTheme !== "function") return;
    const camera = scene.camera as {
      background_color?: { to_hex?: () => string };
      background_opacity?: number;
    } | null;
    if (camera === null || camera === undefined) return;
    const hex = camera.background_color?.to_hex?.();
    if (typeof hex !== "string") return;
    const opacity = camera.background_opacity ?? 1;
    this.host.setControlTheme(dark_from_hex(hex, opacity));
  }
}

/**
 * Relative luminance test for a `#rrggbb` colour: returns whether a control
 * overlay drawn over it should use light icons.  A translucent (or unknown)
 * backdrop is treated as light, matching the common white page.
 */
export function dark_from_hex(hex: string, opacity = 1): boolean {
  if (opacity < 0.5) return false;
  const match = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
  if (match === null) return true;
  const value = Number.parseInt(match[1]!, 16);
  const r = (value >> 16) & 0xff;
  const g = (value >> 8) & 0xff;
  const b = value & 0xff;
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance < 0.5;
}
