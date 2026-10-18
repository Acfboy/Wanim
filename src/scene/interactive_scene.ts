/**
 * `InteractiveScene`: a Scene variant whose mobjects can react to on-canvas
 * controls, and whose story is split into a `blueprint` (interactive) phase and
 * a `construct` (animated) phase.
 *
 * The design goal is a *minimal* extension of the
 * manim API:
 *
 *  - {@link blueprint} is to reactions what `construct` is to updaters: a
 *    `Mobject.add_reaction(...)` callback is applied once when the blueprint
 *    page is entered and again whenever the reader interacts (a control value
 *    changes, a background gesture runs), so a slider's value drives a shape
 *    without recomputing it on idle frames.  The scene itself can also carry
 *    reactions through {@link add_reaction}, for global refresh callbacks that
 *    do not belong to any single mobject.
 *  - After the user presses "next", `construct` runs with `add_updater`
 *    callbacks active and reactions disabled.
 *  - A scene run on its own is one single-scene presentation: the
 *    {@link ScenePresentation} player drives its phases, so pausing behaviour,
 *    buttons, keyboard shortcuts and navigation match multi-scene runs.
 *
 * Multiple blueprint/construct pairs can be scheduled with
 * {@link set_storyboard}; each pair is played in order.
 *
 * The stepwise surface consumed by {@link ScenePresentation} (phase list, one
 * phase execution, one interaction frame, control-value snapshots) is public
 * but internal: user scenes implement `blueprint`/`construct`, not these.
 */
import { Mobject } from "../mobject/mobject.js";
import { Scene, type SceneConstructorOptions } from "./scene.js";
import { isInteractiveControl, type InteractiveControl } from "../interactive/controls.js";
import { NullInteractiveHost, type InteractiveHost } from "../interactive/host.js";
import {
  ScenePresentation,
  type PresentationAction,
  type PresentationPhase,
} from "../interactive/presentation.js";
import type { SceneConstructor } from "../renderer/scene_runner.js";
import { is_thenable } from "../utils/scheduler.js";

/** A callable story step; `null` means "skip this half". */
export type InteractiveStepFn = (() => unknown) | unknown;
export type InteractiveStep = { blueprint: InteractiveStepFn; construct: InteractiveStepFn };

type PhaseKind = "blueprint" | "construct";
type Phase = PresentationPhase;

export interface InteractiveSceneOptions extends SceneConstructorOptions {
  /** Environment services; defaults to a headless {@link NullInteractiveHost}. */
  host?: InteractiveHost;
}

export class InteractiveScene extends Scene {
  /** Marker used to detect interactive scenes without an instanceof check. */
  readonly is_interactive_scene = true;
  /** Host services (pointer mapping, buttons, inline editing). */
  interactive_host: InteractiveHost = new NullInteractiveHost();
  /**
   * Default for the player driving this scene directly (`render()`): pause
   * after every phase.  A multi-scene presentation passes its own value.
   */
  slides_mode = false;
  /**
   * Scene-level reactions: each runs once when the blueprint page is entered
   * and again after any user input, after every mobject's own reactions, and
   * receives the scene.  This is the `Scene.add_reaction` counterpart of
   * `Mobject.add_reaction` for callbacks that are not attached to a shape (it
   * replaces the empty-`VGroup` holder pattern).  Cleared together with the
   * mobjects on restart/replay.
   */
  reactions: Array<(scene: InteractiveScene) => void> = [];

  /** The presentation currently driving this scene, when there is one. */
  private _presentation: ScenePresentation | null = null;
  private _storyboard: InteractiveStep[] | null = null;
  private _phaseKind: PhaseKind = "blueprint";
  private _activeControl: InteractiveControl | null = null;
  private _hoveredControl: (Mobject & InteractiveControl) | null = null;
  private _controlsCache: Array<Mobject & InteractiveControl> | null = null;
  /**
   * Set while the blueprint still needs a reaction pass: on entry it is set by
   * {@link presentation_run_phase}, and every user input sets it again through
   * {@link mark_input}.  Cleared by {@link tick} once the reactions have run,
   * so an idle page renders without recomputing the model.
   */
  private _reactionsPending = false;
  /** Controls whose `on_input` is already wired into {@link mark_input}. */
  private readonly _observedControls = new WeakSet<Mobject>();
  private _disposers: Array<() => void> = [];
  private _installed = false;
  private _requestNext = false;
  private _requestRestart = false;
  private _requestPrevious = false;
  private _requestReplay = false;

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
   * Register a scene-level interaction reaction (same call shape as
   * {@link Mobject.add_reaction}, but the callback receives this scene).  In
   * the blueprint phase it is applied on entry and after user input; it is
   * inert during `construct`, like every reaction.
   */
  add_reaction(reaction: (scene: InteractiveScene) => void): this {
    this.reactions.push(reaction);
    return this;
  }

  addReaction(reaction: (scene: InteractiveScene) => void): this {
    return this.add_reaction(reaction);
  }

  /** Remove a reaction registered through {@link add_reaction}. */
  remove_reaction(reaction: (scene: InteractiveScene) => void): this {
    this.reactions = this.reactions.filter((candidate) => candidate !== reaction);
    return this;
  }

  removeReaction(reaction: (scene: InteractiveScene) => void): this {
    return this.remove_reaction(reaction);
  }

  /**
   * Adding/removing mobjects changes which interactive controls exist (slides
   * that swap whole views in and out), so drop the control cache. Without this
   * a removed slider would keep receiving hit tests from the stale cache.
   */
  override add(...mobjects: Array<Mobject | Iterable<Mobject>>): this {
    this._controlsCache = null;
    return super.add(...mobjects);
  }

  override remove(...mobjects: Array<Mobject | Iterable<Mobject>>): this {
    this._controlsCache = null;
    return super.remove(...mobjects);
  }

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

  /** Request advancement to the next page at the next frame boundary. */
  next_step(): void {
    if (this._presentation !== null) {
      this._presentation.next();
      return;
    }
    this._requestNext = true;
  }

  nextStep(): void {
    this.next_step();
  }

  /** Request a jump to the previous page (slides mode). */
  previous_step(): void {
    if (this._presentation !== null) {
      this._presentation.previous();
      return;
    }
    this._requestPrevious = true;
  }

  previousStep(): void {
    this.previous_step();
  }

  /** Request a re-run of the current page (slides mode). */
  replay_step(): void {
    if (this._presentation !== null) {
      this._presentation.replay();
      return;
    }
    this._requestReplay = true;
  }

  replayStep(): void {
    this.replay_step();
  }

  /** Request that the whole presentation restart from the first page. */
  restart(): void {
    if (this._presentation !== null) {
      this._presentation.restart();
      return;
    }
    this._requestRestart = true;
  }

  // --- presentation surface (consumed by ScenePresentation) -------------------

  /** The scene's phases, in story order. */
  presentation_phases(): Phase[] {
    return this.build_phases();
  }

  /**
   * Run one phase function to completion without pausing.  The player decides
   * whether the phase pauses afterwards.
   */
  async presentation_run_phase(phase: Phase): Promise<void> {
    if (this.is_aborted()) return;
    this._phaseKind = phase.kind;
    // Entering a blueprint always earns one reaction pass, even without input,
    // so the page's initial state reflects the controls' initial values.
    if (phase.kind === "blueprint") this._reactionsPending = true;
    // A gesture must not survive a phase boundary: `move`/`up` events of an
    // in-flight drag are ignored outside the blueprint phase, so clearing the
    // state here is what stops a stale drag from jumping on the next frame.
    this.on_interaction_reset();
    // The phase (re)builds the scene's mobjects, so the cached control list and
    // any hover highlight from the previous phase are stale.
    this.clear_hover();
    this._controlsCache = null;
    const result = phase.run.call(this);
    if (is_thenable(result)) await result;
    // The phase has built its controls: wire their `on_input` now so an edit
    // typed into an inline editor (which never reaches the pointer handlers)
    // still schedules a reaction pass.
    if (phase.kind === "blueprint") this.interactive_controls();
  }

  /** One interaction frame: reactions/updaters + self updaters + draw. */
  presentation_frame(): void {
    this.tick();
  }

  /**
   * Settle a fast-forwarded page without drawing.  The player calls this after
   * running a page with animations skipped, so the state a later page builds on
   * is complete: a blueprint applies its reactions once (with the remembered
   * control values), a construct runs its updaters once, exactly like the
   * `settle()` frame a live page gets.
   */
  presentation_settle_phase(): void {
    this.tick(false);
  }

  /** Drop all scene state so phases can rebuild from scratch. */
  presentation_reset(): void {
    this.reset_interaction_state();
  }

  /** Snapshot the current value of every interactive control, in tree order. */
  presentation_control_values(): unknown[] {
    const values: unknown[] = [];
    for (const control of this.interactive_controls()) {
      try {
        values.push(control.get_value());
      } catch {
        values.push(undefined);
      }
    }
    return values;
  }

  /**
   * Restore a snapshot produced by {@link presentation_control_values}.  The
   * rebuilt controls are matched positionally (the same code recreates them in
   * the same order); a control without `set_value` or a shorter snapshot is
   * skipped.
   *
   * Every restored control then fires its `on_input` listeners once through
   * `notify_input`, even when the value equals the current one, so courseware
   * that mirrors control values into its own state rebuilds that state from the
   * restored controls (a plain `set_value` stays silent, which is what keeps a
   * blueprint reaction's write-back from echoing).  Values are all written
   * before any listener runs, so a listener reading sibling controls sees the
   * complete restored snapshot.
   */
  presentation_set_control_values(values: unknown[]): void {
    const controls = this.interactive_controls();
    const count = Math.min(controls.length, values.length);
    const restored: Array<{ control: Mobject & InteractiveControl; previous: unknown; next: unknown }> = [];
    for (let index = 0; index < count; index += 1) {
      const control = controls[index] as Mobject &
        InteractiveControl & { set_value?: (value: unknown) => unknown };
      const value = values[index];
      if (value === undefined || typeof control.set_value !== "function") continue;
      const previous = this.read_control_value(control);
      try {
        control.set_value(value);
      } catch {
        /* keep going: a stale value must not break the rebuild */
      }
      restored.push({ control, previous, next: this.read_control_value(control) });
    }
    for (const { control, previous, next } of restored) {
      try {
        control.notify_input?.(previous, next);
      } catch {
        /* a listener must not break the remainder of the restore */
      }
    }
  }

  private read_control_value(control: Mobject & InteractiveControl): unknown {
    try {
      return control.get_value();
    } catch {
      return undefined;
    }
  }

  /**
   * Take the pending navigation request set through the `*_step` methods, if
   * any.  The player polls this so external JS can keep calling the
   * scene-level methods.
   */
  requested_action(): PresentationAction | null {
    if (this._requestRestart) {
      this._requestRestart = false;
      return "restart";
    }
    if (this._requestNext) {
      this._requestNext = false;
      return "next";
    }
    if (this._requestPrevious) {
      this._requestPrevious = false;
      return "previous";
    }
    if (this._requestReplay) {
      this._requestReplay = false;
      return "replay";
    }
    return null;
  }

  /** The kind of phase currently being displayed. */
  get phase_kind(): PhaseKind {
    return this._phaseKind;
  }

  /**
   * Register the pointer/wheel handlers on `host`.  Buttons and keyboard
   * shortcuts are owned by the presentation player (or by the embedder);
   * attaching is idempotent so a rebuild of a scene can re-attach safely.
   */
  attach_host(host: InteractiveHost): void {
    this.interactive_host = host;
    if (this._installed) return;
    this._installed = true;
    this._disposers.push(
      host.onPointer({
        down: (event) => this.handle_pointer_down(this.point_from_event(host, event), event),
        move: (event) => this.handle_pointer_move(this.point_from_event(host, event), event),
        up: (event) => this.handle_pointer_up(event),
        leave: () => this.clear_hover(),
      }),
    );
    if (typeof host.onWheel === "function") {
      this._disposers.push(host.onWheel((event) => this.handle_pointer_wheel(event)));
    }
  }

  /** Detach the pointer/wheel handlers and any open inline editor. */
  detach_host(): void {
    for (const dispose of this._disposers.splice(0)) {
      try {
        dispose();
      } catch {
        /* ignore */
      }
    }
    this._installed = false;
    this.interactive_host.endTextEdit();
    this._activeControl = null;
    this.clear_hover();
    this._controlsCache = null;
    this.on_interaction_reset();
  }

  override render(preview = false): boolean | Promise<boolean> {
    this.capture_run_scheduler();
    const host = this.interactive_host ?? new NullInteractiveHost();
    const run = this.run_interactive(host);
    return run.then(
      () => this.finalize_render(preview),
      (error: unknown) => this.handle_render_error(error, preview),
    );
  }

  /**
   * Drive this scene through the presentation player.  Used by {@link render}
   * and available to embedders that own the event loop.
   */
  async run_interactive(host: InteractiveHost = this.interactive_host): Promise<void> {
    const presentation = new ScenePresentation(
      [this.constructor as unknown as SceneConstructor],
      {
        host,
        slidesMode: this.slides_mode,
        createScene: () => this,
      },
    );
    this._presentation = presentation;
    try {
      await presentation.run();
    } finally {
      this._presentation = null;
    }
  }

  private build_phases(): Phase[] {
    const steps: InteractiveStep[] = this._storyboard ?? [this.default_step()];
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

  /**
   * The implicit `(blueprint, construct)` pair of a scene without a
   * storyboard.  The construct half is only added when the subclass overrides
   * `construct`: a blueprint-only scene is a live interactive model, and an
   * implicit no-op construct would only add a fake page whose "next" ends the
   * run.  The blueprint half is always kept (even as the base no-op) so a bare
   * interactive scene, e.g. an orbit-only 3D model, still has a live page.
   */
  private default_step(): InteractiveStep {
    const overridden = (this.construct as unknown) !== (Scene.prototype.construct as unknown);
    return {
      blueprint: () => this.blueprint(),
      construct: overridden ? () => this.construct() : null,
    };
  }

  private to_thunk(fn: InteractiveStepFn): (() => unknown) | null {
    if (typeof fn !== "function") return null;
    const target = fn as ((...args: unknown[]) => unknown) & { __self__?: unknown };
    if (target.__self__ === this) return () => target();
    // rapydscript methods are unbound prototype functions using `this` as self.
    return () => target.call(this);
  }

  private tick(render = true): void {
    const frameRate = (this.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    // Blueprint phases drive mobjects through reactions; construct phases use
    // the ordinary updater machinery (matching `add_reaction` /
    // `add_updater` semantics).  Reactions only run when the page was just
    // entered or the reader interacted: an idle blueprint re-renders (hover,
    // camera gestures) without recomputing the model.
    if (this._phaseKind === "blueprint") {
      if (this._reactionsPending) {
        this._reactionsPending = false;
        this.apply_reactions();
      }
    } else {
      this.update_mobjects(dt);
    }
    this.update_self(dt);
    if (render && typeof this.renderer?.render === "function") {
      this.renderer.render(this, this.time, this.moving_mobjects);
    }
  }

  /**
   * Note that the reader just did something in the blueprint phase, so the
   * next frame re-applies the reactions.  Inputs outside the blueprint phase
   * are inert, like every reaction.
   */
  private mark_input(): void {
    if (this._phaseKind === "blueprint") this._reactionsPending = true;
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
    this._reactionsPending = true;
    this.blueprint();
    this.interactive_controls();
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
    for (const reaction of this.reactions) reaction(this);
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
    if (this._controlsCache === null) {
      this._controlsCache = this.interactive_family().filter(isInteractiveControl);
      this.observe_control_input(this._controlsCache);
    }
    return this._controlsCache;
  }

  /**
   * Wire every interactive control's `on_input` into {@link mark_input}.  This
   * covers gestures that bypass the scene's pointer handlers (typing into an
   * inline editor) and immediate-mode controls; `set_value` stays silent, so a
   * reaction writing values back never echoes.  Registration is once per
   * control: the cache is rebuilt whenever mobjects are added or removed.
   */
  private observe_control_input(controls: Array<Mobject & InteractiveControl>): void {
    for (const control of controls) {
      if (this._observedControls.has(control)) continue;
      this._observedControls.add(control);
      control.on_input?.(() => this.mark_input());
    }
  }

  private find_control_at(point: readonly number[]): (Mobject & InteractiveControl) | null {
    const controls = this.interactive_controls();
    for (let index = controls.length - 1; index >= 0; index -= 1) {
      const control = controls[index]!;
      if (control.is_hit(point)) return control;
    }
    return null;
  }

  /**
   * Move the hover highlight to `control` (or clear it with `null`). Only
   * controls implementing the optional `set_hovered` are notified. `point` is
   * forwarded on every call (not only on a control change) so a control can
   * track which of its sub-parts the pointer is over (e.g. a slider knob).
   */
  private set_hovered_control(control: (Mobject & InteractiveControl) | null, point?: readonly number[]): void {
    if (control !== this._hoveredControl) {
      this._hoveredControl?.set_hovered?.(false);
      this._hoveredControl = control;
    }
    control?.set_hovered?.(true, point);
  }

  /** Re-evaluate which control the pointer is over. */
  private update_hover(point: readonly number[]): void {
    this.set_hovered_control(this.find_control_at(point), point);
  }

  /** Drop any active hover highlight (phase change, pointer leave, teardown). */
  private clear_hover(): void {
    this.set_hovered_control(null);
  }

  /**
   * Pointer/wheel routing shared with subclasses.
   *
   * A press first hit-tests the interactive controls; only when it lands on
   * the background is {@link on_background_pointer_down} invoked, so controls
   * always win over background gestures (e.g. a 3D camera orbit).  The same
   * applies to motion while a control drag is active.  Background gestures,
   * like control interaction, only run while the blueprint phase is idle —
   * animations never compete with user input.
   */
  protected handle_pointer_down(point: number[], event?: PointerEvent): void {
    if (this._phaseKind !== "blueprint") return;
    this.mark_input();
    const control = this.find_control_at(point);
    if (control === null) {
      this.clear_hover();
      this.on_background_pointer_down(point, event);
      return;
    }
    this.set_hovered_control(control, point);
    this._activeControl = control;
    control.on_press(point, { host: this.interactive_host, camera: this.camera });
  }

  protected handle_pointer_move(point: number[], event?: PointerEvent): void {
    if (this._activeControl !== null) {
      this.mark_input();
      this._activeControl.on_drag(point, { host: this.interactive_host, camera: this.camera });
      return;
    }
    if (this._phaseKind !== "blueprint") return;
    this.mark_input();
    this.update_hover(point);
    this.on_background_pointer_move(point, event);
  }

  protected handle_pointer_up(event?: PointerEvent): void {
    if (this._activeControl !== null) {
      this.mark_input();
      this._activeControl.on_release();
      this._activeControl = null;
      // Hover updates are suspended while a drag is active, so re-evaluate the
      // hover at the release point instead of leaving pre-drag state behind
      // (e.g. a knob grown from hovering it before the drag started).
      if (event !== undefined) this.update_hover(this.point_from_event(this.interactive_host, event));
      return;
    }
    if (this._phaseKind !== "blueprint") return;
    this.mark_input();
    this.on_background_pointer_up(event);
  }

  protected handle_pointer_wheel(event: WheelEvent): void {
    if (this._phaseKind !== "blueprint") return;
    this.mark_input();
    this.on_background_wheel(event);
  }

  /**
   * Called when a pointer press misses every interactive control.  Subclasses
   * override this to implement background gestures (e.g. camera orbit in
   * `InteractiveThreeDScene`); the default is a no-op.
   */
  protected on_background_pointer_down(_point: number[], _event?: PointerEvent): void {}

  /** Called for pointer motion that is not dragging an interactive control. */
  protected on_background_pointer_move(_point: number[], _event?: PointerEvent): void {}

  /** Called when a background press ends (a control press ends via its release). */
  protected on_background_pointer_up(_event?: PointerEvent): void {}

  /** Called for wheel events while the blueprint phase is idle. */
  protected on_background_wheel(_event: WheelEvent): void {}

  /**
   * Called whenever interaction state is torn down: a story restart, a new
   * render, or the end of a phase.  Subclasses use it to drop in-flight
   * gesture state (e.g. an orbit drag).
   */
  protected on_interaction_reset(): void {}

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
    this.reactions = [];
    this.last_t = 0;
    this._reactionsPending = false;
    this._activeControl = null;
    this.clear_hover();
    this._controlsCache = null;
    this.interactive_host.endTextEdit();
    this.on_interaction_reset();
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
