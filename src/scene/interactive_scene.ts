/**
 * `InteractiveScene`: a Scene variant whose mobjects can react to on-canvas
 * controls, and whose story is split into a `blueprint` (interactive) phase and
 * a `construct` (animated) phase.
 *
 * The design goal is a *minimal* extension of the
 * manim API:
 *
 *  - {@link blueprint} is to reactions what `construct` is to updaters: while a
 *    scene is in its blueprint phase, `Mobject.add_reaction(...)` callbacks are
 *    re-applied every frame, so a slider's value can drive a shape.
 *  - After the user presses "next", `construct` runs with `add_updater`
 *    callbacks active and reactions disabled.
 *  - The scene never stops on its own: it keeps redrawing (a "preprint",
 *    i.e. an endless `play`) until the user presses "next" on the final phase,
 *    the render is aborted, or the scene is nested in a headless run.
 *
 * Multiple blueprint/construct pairs can be scheduled with
 * {@link set_storyboard}; each pair is played in order, with the "next" button
 * advancing from a blueprint to its construct and then to the next pair.
 */
import { Mobject } from "../mobject/mobject.js";
import { Scene, type SceneConstructorOptions } from "./scene.js";
import { isInteractiveControl, type InteractiveControl } from "../interactive/controls.js";
import { NullInteractiveHost, type InteractiveHost } from "../interactive/host.js";
import { is_thenable } from "../utils/scheduler.js";

/** A callable story step; `null` means "skip this half". */
export type InteractiveStepFn = (() => unknown) | unknown;
export type InteractiveStep = { blueprint: InteractiveStepFn; construct: InteractiveStepFn };

type PhaseKind = "blueprint" | "construct";
interface Phase {
  kind: PhaseKind;
  run: () => unknown;
}

export interface InteractiveSceneOptions extends SceneConstructorOptions {
  /** Environment services; defaults to a headless {@link NullInteractiveHost}. */
  host?: InteractiveHost;
}

export class InteractiveScene extends Scene {
  /** Marker used to detect interactive scenes without an instanceof check. */
  readonly is_interactive_scene = true;
  /** Host services (pointer mapping, buttons, inline editing). */
  interactive_host: InteractiveHost = new NullInteractiveHost();

  private _storyboard: InteractiveStep[] | null = null;
  private _phaseKind: PhaseKind = "blueprint";
  private _activeControl: InteractiveControl | null = null;
  private _disposers: Array<() => void> = [];
  private _installed = false;
  private _requestNext = false;
  private _requestRestart = false;

  constructor(options: InteractiveSceneOptions = {}) {
    super(options);
    const host = (options as InteractiveSceneOptions).host;
    if (host !== undefined && host !== null) {
      this.interactive_host = host;
    }
  }

  /** Overridden by user scenes; runs while interactions are live. */
  blueprint(): void {}

  /**
   * Schedule a sequence of `(blueprint, construct)` pairs. Each entry may be a
   * two-element array or an object `{ blueprint, construct }`; pass `None`/`null`
   * for a half that should be skipped. Once called, `blueprint`/`construct` are
   * no longer invoked implicitly, only the scheduled steps are.
   */
  set_storyboard(steps: Array<InteractiveStepFn[] | InteractiveStep> = []): this {
    this._storyboard = steps.map((step) => {
      if (Array.isArray(step)) {
        return { blueprint: step[0] ?? null, construct: step[1] ?? null };
      }
      return { blueprint: step.blueprint ?? null, construct: step.construct ?? null };
    });
    return this;
  }

  /** Alias of {@link set_storyboard}. */
  set_script(steps: Array<InteractiveStepFn[] | InteractiveStep> = []): this {
    return this.set_storyboard(steps);
  }

  /** Request a full restart at the next frame boundary. */
  next_step(): void {
    this._requestNext = true;
  }

  nextStep(): void {
    this.next_step();
  }

  /** Request that the whole story restart from the first blueprint. */
  restart(): void {
    this._requestRestart = true;
  }

  override render(preview = false): boolean | Promise<boolean> {
    this.capture_run_scheduler();
    this.setup();
    const host = this.interactive_host ?? new NullInteractiveHost();
    const run = this.run_interactive(host);
    return run.then(
      () => this.finalize_render(preview),
      (error: unknown) => this.handle_render_error(error, preview),
    );
  }

  /** The interactive driver: run every phase until done/restart/abort. */
  async run_interactive(host: InteractiveHost = this.interactive_host): Promise<void> {
    this.install_host(host);
    try {
      for (;;) {
        const phases = this.build_phases();
        let restart = false;
        let stop = false;
        for (const phase of phases) {
          const action = await this.run_phase(phase, host);
          if (action === "restart") {
            restart = true;
            break;
          }
          if (action === "finish" || action === "abort") {
            stop = true;
            break;
          }
        }
        if (stop) break;
        if (restart) {
          this.reset_interaction_state();
          continue;
        }
        break;
      }
    } finally {
      host.setButtonsEnabled(false);
      this.uninstall_host();
    }
  }

  private build_phases(): Phase[] {
    const steps: InteractiveStep[] =
      this._storyboard ??
      [{ blueprint: () => this.blueprint(), construct: () => this.construct() }];
    const phases: Phase[] = [];
    for (const step of steps) {
      const blueprint = this.to_thunk(step.blueprint);
      if (blueprint !== null) phases.push({ kind: "blueprint", run: blueprint });
      const construct = this.to_thunk(step.construct);
      if (construct !== null) phases.push({ kind: "construct", run: construct });
    }
    // A storyboard with no blueprint half (e.g. `[(None, construct0), ...]`)
    // still expects the shared `blueprint()` method to have built `self.*`
    // state first. Without this, `construct` phases see `self.cells` etc. as
    // `undefined` (`undefined is not iterable`). Prepend the default blueprint
    // once when the subclass overrides it; a base no-op blueprint adds no
    // value and would only cost an extra "next" click.
    if (
      this._storyboard !== null &&
      phases.length > 0 &&
      phases.every((phase) => phase.kind !== "blueprint") &&
      (this.blueprint as unknown) !== (InteractiveScene.prototype.blueprint as unknown)
    ) {
      phases.unshift({ kind: "blueprint", run: () => this.blueprint() });
    }
    return phases;
  }

  private to_thunk(fn: InteractiveStepFn): (() => unknown) | null {
    if (typeof fn !== "function") return null;
    const target = fn as ((...args: unknown[]) => unknown) & { __self__?: unknown };
    if (target.__self__ === this) return () => target();
    // rapydscript methods are unbound prototype functions using `this` as self.
    return () => target.call(this);
  }

  private async run_phase(phase: Phase, host: InteractiveHost): Promise<"next" | "restart" | "abort" | "finish"> {
    if (this.is_aborted()) return "abort";
    this._phaseKind = phase.kind;
    // Buttons are unavailable while a phase function (typically a `play`
    // animation) runs; they light up again once the scene is idle and the user
    // can interact / advance.
    host.setButtonsEnabled(false);
    const result = phase.run.call(this);
    if (is_thenable(result)) await result;
    // The run may have been aborted while the phase ran (e.g. the user pressed
    // the run button again); do not enter the interaction loop of a dead run.
    if (this.is_aborted()) return "abort";
    if (!host.live) {
      // Headless: draw one frame for the phase and move on, so Node runs and
      // tests terminate naturally.
      this.tick();
      return "next";
    }
    host.setButtonsEnabled(true);
    const action = await this.interactive_loop(host);
    host.setButtonsEnabled(false);
    return action;
  }

  private async interactive_loop(host: InteractiveHost): Promise<"next" | "restart" | "abort"> {
    for (;;) {
      if (this._requestRestart) {
        this._requestRestart = false;
        return "restart";
      }
      if (this._requestNext) {
        this._requestNext = false;
        return "next";
      }
      if (this.is_aborted()) return "abort";
      this.tick();
      await host.nextFrame();
    }
  }

  private tick(): void {
    const frameRate = (this.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    // Blueprint phases drive mobjects through reactions; construct phases use
    // the ordinary updater machinery (matching `add_reaction` /
    // `add_updater` semantics).
    if (this._phaseKind === "blueprint") this.apply_reactions();
    else this.update_mobjects(dt);
    this.update_self(dt);
    if (typeof this.renderer?.render === "function") {
      this.renderer.render(this, this.time, this.moving_mobjects);
    }
  }

  /**
   * Build the blueprint without entering the host-driven render loop.
   *
   * For embedders that own their event loop (performance benchmarks, custom
   * hosts): call `enter_blueprint()`, then `interactive_frame()` once per
   * frame.  A normal `render()` must not follow (it would run `setup()` and
   * the blueprint again).
   */
  enter_blueprint(): void {
    this.setup();
    this._phaseKind = "blueprint";
    this.blueprint();
  }

  /**
   * Run exactly one blueprint frame: re-apply every reaction, run self
   * updaters and draw.  This is the per-frame work a live host performs while
   * the user drags a control, exposed for benchmarks and custom hosts.
   */
  interactive_frame(): void {
    this.tick();
  }

  private apply_reactions(): void {
    for (const mobject of this.interactive_family()) {
      for (const reaction of mobject.reactions) reaction(mobject);
    }
  }

  private interactive_family(): Mobject[] {
    const seen = new Set<Mobject>();
    const out: Mobject[] = [];
    for (const root of this.mobjects) {
      for (const member of root.get_family()) {
        if (seen.has(member)) continue;
        seen.add(member);
        out.push(member);
      }
    }
    return out;
  }

  private interactive_controls(): Array<Mobject & InteractiveControl> {
    return this.interactive_family().filter(isInteractiveControl);
  }

  private find_control_at(point: readonly number[]): (Mobject & InteractiveControl) | null {
    const controls = this.interactive_controls();
    for (let index = controls.length - 1; index >= 0; index -= 1) {
      const control = controls[index]!;
      if (control.is_hit(point)) return control;
    }
    return null;
  }

  private install_host(host: InteractiveHost): void {
    if (this._installed) return;
    this._installed = true;
    this._disposers.push(
      host.onPointer({
        down: (event) => this.handle_pointer_down(this.point_from_event(host, event)),
        move: (event) => this.handle_pointer_move(this.point_from_event(host, event)),
        up: () => this.handle_pointer_up(),
      }),
    );
    this._disposers.push(host.addButton("从头再来", () => this.restart(), "restart"));
    this._disposers.push(host.addButton("进入下一步", () => this.next_step(), "next"));
    this._disposers.push(
      host.onKey({
        next: () => this.next_step(),
        restart: () => this.restart(),
      }),
    );
  }

  private uninstall_host(): void {
    for (const dispose of this._disposers) {
      try {
        dispose();
      } catch {
        /* ignore */
      }
    }
    this._disposers = [];
    this._installed = false;
    this.interactive_host.endTextEdit();
    this._activeControl = null;
  }

  private handle_pointer_down(point: number[]): void {
    if (this._phaseKind !== "blueprint") return;
    const control = this.find_control_at(point);
    if (control === null) return;
    this._activeControl = control;
    control.on_press(point, { host: this.interactive_host, camera: this.camera });
  }

  private handle_pointer_move(point: number[]): void {
    if (this._activeControl === null) return;
    this._activeControl.on_drag(point, { host: this.interactive_host, camera: this.camera });
  }

  private handle_pointer_up(): void {
    if (this._activeControl === null) return;
    this._activeControl.on_release();
    this._activeControl = null;
  }

  private is_aborted(): boolean {
    // Judge against this run's own scheduler: a newer run installs a different
    // global scheduler, and consulting that would keep a dead run alive.
    return this.run_aborted();
  }

  private point_from_event(host: InteractiveHost, event: PointerEvent): number[] {
    return host.pointFromEvent(event, this.camera);
  }

  private reset_interaction_state(): void {
    this.mobjects = [];
    this.foreground_mobjects = [];
    this.moving_mobjects = [];
    this.static_mobjects = [];
    this.animations = [];
    this.updaters = [];
    this.last_t = 0;
    this._activeControl = null;
    this.interactive_host.endTextEdit();
    const renderer = this.renderer as
      | { clear_screen?: () => void; num_plays?: number; time?: number }
      | null
      | undefined;
    if (renderer !== null && renderer !== undefined) {
      renderer.clear_screen?.();
      renderer.num_plays = 0;
      renderer.time = 0;
    }
  }
}
