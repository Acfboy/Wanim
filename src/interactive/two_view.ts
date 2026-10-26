/**
 * `TwoViewPresentation`: the "watch, then take over" player.
 *
 * Two-view scenes are primarily an animation: the player loops the authored
 * story (`blueprint` silently with the controls' initial values, then
 * `construct`) until the reader double-clicks the canvas or presses the
 * bottom-right interact icon.  The interaction state then shows the live
 * blueprint, lets the reader set values, and plays the construct on demand;
 * double-clicking again returns to the looping animation.
 *
 * The scene decides which of the four shapes it wants:
 *
 *  - `blueprint` + `construct` (the default): animation loops
 *    blueprint→construct; interaction pauses on the blueprint.
 *  - `SEPARATE = True` on the class: the animation loops `construct` only and
 *    the interaction state shows `blueprint` only; the two halves are
 *    independent.
 *  - `self.clear_stage()` at the end of a blueprint: the animation still runs
 *    blueprint→construct, but the construct starts on an empty canvas while
 *    still reading `self.slider.get_value()`.
 *  - only `construct` (animation only) or only `blueprint` (interaction only):
 *    the other half simply does not exist.
 *
 * The player owns its scene lifecycle: navigating rebuilds the scene from its
 * start and fast-forwards earlier pages (without drawing) so a page always
 * starts from the same state as if the story had been played into it.  It also
 * remembers control values per page, like `ScenePresentation` does for slides.
 */
import type { Scene } from "../scene/scene.js";
import type { SceneConstructor } from "../renderer/scene_runner.js";
import {
  InteractiveScene,
  type InteractivePage,
  type TwoViewPlan,
} from "../scene/interactive_scene.js";
import type { InteractiveHost } from "./host.js";
import { is_thenable, type RenderScheduler } from "../utils/scheduler.js";
import {
  PRESENTATION_CONTROL_LABELS,
  dark_from_hex,
  type PresentationPhase,
  type PresentationState,
} from "./presentation.js";
import type { PresentationTheme } from "./config.js";

/** A `post_message(key, value)` forwarded to the embedding page. */
export interface WanimMessage {
  /** Per-run id supplied by the host (unique across a page's embeds). */
  id: string;
  /** Author-chosen message name, e.g. `"passed"`. */
  key: unknown;
  /** Optional payload. */
  value: unknown;
  /** Active scene class. */
  scene: string;
  sceneIndex: number;
  /** Zero-based page (storyboard step) the message came from. */
  pageIndex: number;
  phaseKind: "blueprint" | "construct";
}

/** Observable state of a two-view run. */
export interface TwoViewState extends PresentationState {
  mode: "two-view" | "animation";
  /** `animation` while the demo loops, `interaction` while the reader drives. */
  view: "animation" | "interaction";
  pageIndex: number;
  pageCount: number;
  /** Whether the interact icon / double-click can switch to interaction. */
  canInteract: boolean;
  /** Whether the animation half exists (double-click can leave interaction). */
  canAnimate: boolean;
}

type Action = "next" | "previous" | "replay" | "interact" | "animate" | "restart" | "abort";
type PassResult = Action | "cycle" | "no-animation";

/** Per-phase scheduler wrapper (same role as the one in `presentation.ts`). */
class PhaseScheduler implements RenderScheduler {
  readonly async = true;
  private phaseCancelled = false;

  constructor(private readonly base: RenderScheduler) {}

  get cancelled(): boolean {
    return this.phaseCancelled || this.base.cancelled;
  }

  next_frame(): Promise<void> {
    return this.base.next_frame();
  }

  cancel(): void {
    this.phaseCancelled = true;
  }

  resume(): void {
    this.phaseCancelled = false;
  }
}

interface InteractiveSceneLike {
  is_interactive_scene: true;
  two_view_plan(): TwoViewPlan;
  presentation_run_phase(phase: PresentationPhase): Promise<void>;
  presentation_frame(): void;
  presentation_settle_phase(): void;
  presentation_reset(): void;
  presentation_apply_stage_clear(): void;
  presentation_control_values(): unknown[];
  presentation_set_control_values(values: unknown[]): void;
  attach_host(host: InteractiveHost): void;
  detach_host(): void;
  requested_action(): "next" | "previous" | "replay" | "restart" | "abort" | null;
  interaction_enabled: boolean;
}

function as_interactive(scene: Scene): InteractiveSceneLike | null {
  if ((scene as { is_interactive_scene?: boolean }).is_interactive_scene !== true) return null;
  return scene as unknown as InteractiveSceneLike;
}

function plain_plan(scene: Scene): TwoViewPlan {
  const page: InteractivePage = { blueprint: null, construct: () => scene.construct() };
  return {
    separate: false,
    has_blueprint: false,
    has_construct: true,
    pages: [page],
    animation_pages: [page],
  };
}

export interface TwoViewPresentationOptions {
  host: InteractiveHost;
  createScene: (sceneClass: SceneConstructor) => Scene;
  mode: "two-view" | "animation";
  theme?: PresentationTheme;
  instanceId?: string;
  disposeScene?: (scene: Scene) => void;
  onSceneActivated?: (scene: Scene, sceneIndex: number, sceneClass: SceneConstructor) => void;
  onStateChange?: (state: TwoViewState) => void;
  onFinished?: (state: TwoViewState) => void;
  /** Forwarded `post_message` calls. */
  onMessage?: (message: WanimMessage) => void;
  /** Called after the animation loop finished one full pass (GIF export). */
  onLoop?: () => void;
}

/** Play a two-view (or animation-only) presentation on a host. */
export class TwoViewPresentation {
  readonly sceneClasses: SceneConstructor[];
  readonly host: InteractiveHost;
  readonly mode: "two-view" | "animation";
  readonly instanceId: string;

  private readonly createScene: (sceneClass: SceneConstructor) => Scene;
  private readonly theme: PresentationTheme | undefined;
  private readonly disposeScene: ((scene: Scene) => void) | null;
  private readonly onSceneActivated: TwoViewPresentationOptions["onSceneActivated"];
  private readonly onStateChange: TwoViewPresentationOptions["onStateChange"];
  private readonly onFinished: TwoViewPresentationOptions["onFinished"];
  private readonly onMessage: TwoViewPresentationOptions["onMessage"];
  private readonly onLoop: TwoViewPresentationOptions["onLoop"];

  private scene: Scene | null = null;
  private sceneIndex = -1;
  private sceneClass: SceneConstructor | null = null;
  private plan: TwoViewPlan | null = null;
  private pageIndex = 0;
  private subphase: "blueprint" | "construct" = "blueprint";
  private view: "animation" | "interaction" = "animation";
  private running = false;
  private done = false;
  private abortRequested = false;
  private pending: Action | null = null;
  private phaseScheduler: PhaseScheduler | null = null;
  private readonly disposers: Array<() => void> = [];
  private readonly controlValues = new Map<string, unknown[]>();

  constructor(sceneClasses: SceneConstructor[], options: TwoViewPresentationOptions) {
    this.sceneClasses = [...sceneClasses];
    this.host = options.host;
    this.mode = options.mode;
    this.instanceId = options.instanceId ?? "wanim-scene";
    this.createScene = options.createScene;
    this.theme = options.theme;
    this.disposeScene = options.disposeScene ?? null;
    this.onSceneActivated = options.onSceneActivated;
    this.onStateChange = options.onStateChange;
    this.onFinished = options.onFinished;
    this.onMessage = options.onMessage;
    this.onLoop = options.onLoop;
  }

  // --- public state ----------------------------------------------------------

  get state(): TwoViewState {
    const page = this.current_page();
    const base: PresentationState = {
      sceneIndex: Math.max(0, this.sceneIndex),
      sceneCount: this.sceneClasses.length,
      sceneName: this.sceneClass?.name ?? "",
      phaseIndex: this.pageIndex,
      phaseCount: this.plan?.pages.length ?? 0,
      phaseKind: this.subphase,
      slidesMode: false,
      done: this.done,
      canNext: this.can_next(),
      canPrevious: this.can_previous(),
      canReplay: this.running && this.current_page() !== null,
    };
    return {
      ...base,
      mode: this.mode,
      view: this.view,
      pageIndex: this.pageIndex,
      pageCount: this.plan?.pages.length ?? 0,
      canInteract: this.mode === "two-view" && this.running && this.has_interaction_here(),
      canAnimate: this.has_animation_here() || this.plan?.has_construct === true,
    };
  }

  get active(): boolean {
    return this.running && !this.done;
  }

  // --- external controls -----------------------------------------------------

  next(): void {
    this.request("next");
  }

  previous(): void {
    this.request("previous");
  }

  replay(): void {
    this.request("replay");
  }

  restart(): void {
    this.request("restart");
  }

  /** Switch from the looping animation to the live interaction page. */
  enterInteraction(): void {
    this.request("interact");
  }

  /** Switch back from interaction to the looping animation. */
  enterAnimation(): void {
    this.request("animate");
  }

  abort(): void {
    this.abortRequested = true;
    this.pending = null;
  }

  private request(action: Action): void {
    if (!this.running || this.done) return;
    if (this.view === "animation") {
      if (action === "interact" && !this.has_interaction_here()) return;
      if (action === "next" || action === "previous" || action === "animate") return;
      if (action === "replay" && this.mode === "animation" && this.host.live && false) return;
      this.pending = action;
      return;
    }
    if (action === "interact") return;
    if (action === "next" && !this.can_next()) return;
    if (action === "previous" && !this.can_previous()) return;
    if (action === "animate" && !this.has_animation_here()) return;
    this.pending = action;
  }

  // --- run loop --------------------------------------------------------------

  async run(): Promise<void> {
    if (this.running) return;
    if (this.sceneClasses.length === 0) return;
    this.running = true;
    this.done = false;
    this.abortRequested = false;
    this.pending = null;
    this.controlValues.clear();
    this.view = "animation";
    this.install_controls();
    this.install_double_click();
    try {
      // Headless hosts (Node tests/serialization) perform one complete pass and
      // then finish instead of looping forever.
      if (!this.host.live) {
        await this.run_headless();
        return;
      }
      await this.run_live();
    } finally {
      this.hide_controls();
      const finished = this.state;
      this.dispose_current_scene();
      this.uninstall_controls();
      this.running = false;
      this.emit_state();
      this.onFinished?.(finished);
    }
  }

  private async run_live(): Promise<void> {
    for (;;) {
      if (this.abortRequested) return;
      if (this.mode === "animation") {
        const result = await this.run_animation_pass();
        if (result === "abort") return;
        if (result === "no-animation") {
          this.done = true;
          return;
        }
        // `cycle`, `replay`, and `restart` all simply loop from the first page.
        continue;
      }
      if (this.view === "animation") {
        const result = await this.run_animation_pass();
        if (result === "abort") return;
        if (result === "interact") {
          if (!(await this.start_interaction_from_current())) {
            // Nothing to interact with: keep looping.
            continue;
          }
          const end = await this.run_interaction_session();
          if (end === "abort") return;
          if (end === "animate" || end === "restart") {
            this.view = "animation";
            continue;
          }
          return;
        }
        if (result === "no-animation") {
          if (!(await this.start_first_interaction())) return;
          const end = await this.run_interaction_session();
          if (end === "abort") return;
          if (end === "animate" || end === "restart") {
            this.view = "animation";
            continue;
          }
          return;
        }
        // cycle / replay / restart: loop the animation from the start.
        continue;
      }
      // Interaction is entered only from the branches above; if a session
      // somehow lands here, rebuild it.
      this.view = "animation";
    }
  }

  private async run_headless(): Promise<void> {
    await this.prepare_scene(0);
    const plan = this.plan!;
    if (plan.has_construct) {
      for (const page of plan.animation_pages) {
        if (page.construct === null) continue;
        if (page.blueprint !== null && !plan.separate) {
          this.set_interaction_enabled(false);
          await this.run_phase({ kind: "blueprint", run: page.blueprint });
          this.settle_skipped_phase();
          this.apply_stage_clear();
          this.set_interaction_enabled(true);
        }
        await this.run_phase({ kind: "construct", run: page.construct });
      }
    } else if (plan.has_blueprint) {
      const page = plan.pages.find((candidate) => candidate.blueprint !== null);
      if (page?.blueprint !== undefined && page.blueprint !== null) {
        await this.jump_to_page(this.sceneIndex, plan.pages.indexOf(page), "blueprint");
      }
    }
    this.done = true;
  }

  // --- animation state -------------------------------------------------------

  /**
   * Run one complete animation pass over every scene, returning early when the
   * reader asks for a view change/replay.  `cycle` means the pass finished.
   */
  private async run_animation_pass(): Promise<PassResult> {
    let played = false;
    this.view = "animation";
    for (let sceneIndex = 0; sceneIndex < this.sceneClasses.length; sceneIndex += 1) {
      if (this.abortRequested) return "abort";
      await this.prepare_scene(sceneIndex);
      const plan = this.plan!;
      if (!plan.has_construct) {
        continue;
      }
      this.show_animation_controls(plan);
      const pages = plan.separate
        ? plan.animation_pages
        : plan.pages.filter((page) => page.construct !== null);
      for (const page of pages) {
        if (this.abortRequested) return "abort";
        this.pageIndex = plan.pages.indexOf(page);
        this.subphase = "construct";
        this.set_interaction_enabled(false);
        this.show_animation_controls(plan);
        const action = await this.run_watching(async () => {
          if (page.blueprint !== null && !plan.separate) {
            this.subphase = "blueprint";
            await this.run_phase({ kind: "blueprint", run: page.blueprint });
            this.settle();
            this.apply_control_theme();
            this.apply_stage_clear();
          }
          this.subphase = "construct";
          await this.run_phase({ kind: "construct", run: page.construct as () => unknown });
        });
        played = true;
        if (action !== null) {
          this.set_interaction_enabled(true);
          return action;
        }
        await this.host.nextFrame();
      }
    }
    this.set_interaction_enabled(true);
    if (!played) return "no-animation";
    this.onLoop?.();
    return "cycle";
  }

  // --- interaction state -----------------------------------------------------

  /** Find a blueprint page to start from, using the current animation page. */
  private async start_interaction_from_current(): Promise<boolean> {
    if (this.plan === null) return false;
    const plan = this.plan;
    // Prefer the current page or the next blueprint page of this scene.
    for (let index = this.pageIndex; index < plan.pages.length; index += 1) {
      if (plan.pages[index]?.blueprint !== null && plan.pages[index]?.blueprint !== undefined) {
        this.view = "interaction";
        await this.jump_to_page(this.sceneIndex, index, "blueprint");
        return true;
      }
    }
    // Then the previous blueprint pages.
    for (let index = this.pageIndex - 1; index >= 0; index -= 1) {
      if (plan.pages[index]?.blueprint !== null && plan.pages[index]?.blueprint !== undefined) {
        this.view = "interaction";
        await this.jump_to_page(this.sceneIndex, index, "blueprint");
        return true;
      }
    }
    // Finally any other scene with an interaction page.
    return this.start_first_interaction();
  }

  private async start_first_interaction(): Promise<boolean> {
    const startScene = Math.max(0, this.sceneIndex);
    const order = [
      ...Array.from({ length: this.sceneClasses.length }, (_, i) => (startScene + i) % this.sceneClasses.length),
    ];
    for (const sceneIndex of order) {
      await this.prepare_scene(sceneIndex);
      const plan = this.plan!;
      if (!plan.has_blueprint) continue;
      const index = plan.pages.findIndex((page) => page.blueprint !== null);
      if (index < 0) continue;
      this.view = "interaction";
      await this.jump_to_page(sceneIndex, index, "blueprint");
      return true;
    }
    return false;
  }

  /** Wait for the reader while a page is live.  Returns the ending action. */
  private async run_interaction_session(): Promise<Action> {
    this.view = "interaction";
    this.set_interaction_enabled(true);
    this.show_interaction_controls();
    for (;;) {
      if (this.abortRequested) return "abort";
      const action = await this.wait_for_action();
      if (action === "abort") return "abort";
      if (action === "restart") {
        if (!(await this.start_first_interaction())) return "abort";
        continue;
      }
      if (action === "animate") return "animate";
      if (action === "next") {
        if (this.subphase === "blueprint" && this.plan?.separate !== true) {
          const page = this.current_page();
          if (page !== null && page.construct !== null) {
            this.record_control_values();
            const next = await this.run_construct_live(page);
            if (next !== null) {
              if (next === "abort" || next === "animate") return next;
              this.pending = next;
            }
            continue;
          }
        }
        await this.handle_interaction_action("next");
        continue;
      }
      if (action === "previous" || action === "replay") {
        await this.handle_interaction_action(action);
        continue;
      }
    }
  }

  private async handle_interaction_action(action: Action): Promise<void> {
    if (action === "abort") return;
    if (action === "next") {
      if (this.subphase === "construct") {
        await this.goto_next_page();
        return;
      }
      // A blueprint without a construct continues to the next page (the
      // interaction-only shape), otherwise `>` plays the construct.
      await this.goto_next_page();
      return;
    }
    if (action === "previous") {
      if (this.subphase === "construct") {
        await this.jump_to_page(this.sceneIndex, this.pageIndex, "blueprint");
        return;
      }
      await this.goto_previous_page();
      return;
    }
    if (action === "replay") {
      if (this.subphase === "blueprint" || this.plan?.separate === true) {
        this.controlValues.delete(this.value_key(this.pageIndex));
        await this.jump_to_page(this.sceneIndex, this.pageIndex, "blueprint");
      } else {
        await this.jump_to_page(this.sceneIndex, this.pageIndex, "construct");
      }
      return;
    }
    if (action === "restart") {
      await this.start_first_interaction();
    }
  }

  private async run_construct_live(page: InteractivePage): Promise<Action | null> {
    if (page.construct === null) return null;
    this.subphase = "construct";
    this.set_interaction_enabled(false);
    this.apply_stage_clear();
    this.show_interaction_controls();
    const action = await this.run_watching(async () => {
      await this.run_phase({ kind: "construct", run: page.construct as () => unknown });
    });
    this.set_interaction_enabled(true);
    this.settle();
    this.show_interaction_controls();
    return action;
  }

  private async goto_next_page(): Promise<void> {
    const plan = this.plan;
    if (plan === null) return;
    if (plan.separate) {
      // SEPARATE: the interaction state shows blueprints only; the paired
      // construct must never run here. Skip construct-only pages.
      for (let index = this.pageIndex + 1; index < plan.pages.length; index += 1) {
        if (plan.pages[index]!.blueprint !== null) {
          await this.jump_to_page(this.sceneIndex, index, "blueprint");
          return;
        }
      }
      const originScene = this.sceneIndex;
      const originPage = this.pageIndex;
      for (let sceneIndex = this.sceneIndex + 1; sceneIndex < this.sceneClasses.length; sceneIndex += 1) {
        await this.prepare_scene(sceneIndex);
        const nextPlan = this.plan!;
        const index = nextPlan.pages.findIndex((page) => page.blueprint !== null);
        if (index >= 0) {
          await this.jump_to_page(sceneIndex, index, "blueprint");
          return;
        }
      }
      // No later blueprint: stay on the current page (no next page).
      if (this.sceneIndex !== originScene) {
        await this.jump_to_page(originScene, originPage, "blueprint");
      }
      return;
    }
    if (this.pageIndex + 1 < plan.pages.length) {
      const next = plan.pages[this.pageIndex + 1]!;
      const subphase = next.blueprint !== null ? "blueprint" : "construct";
      await this.jump_to_page(this.sceneIndex, this.pageIndex + 1, subphase);
      return;
    }
    if (this.sceneIndex + 1 < this.sceneClasses.length) {
      await this.prepare_scene(this.sceneIndex + 1);
      const nextPlan = this.plan!;
      if (nextPlan.pages.length === 0) {
        // Skip an empty scene and continue searching.
        if (this.sceneIndex + 1 < this.sceneClasses.length) await this.goto_next_page();
        return;
      }
      const subphase = nextPlan.pages[0]!.blueprint !== null ? "blueprint" : "construct";
      await this.jump_to_page(this.sceneIndex, 0, subphase);
    }
  }

  private async goto_previous_page(): Promise<void> {
    const plan = this.plan;
    if (plan === null) return;
    if (plan.separate) {
      // SEPARATE: stay within blueprints only; never land on a construct.
      for (let index = this.pageIndex - 1; index >= 0; index -= 1) {
        if (plan.pages[index]!.blueprint !== null) {
          await this.jump_to_page(this.sceneIndex, index, "blueprint");
          return;
        }
      }
      const originScene = this.sceneIndex;
      const originPage = this.pageIndex;
      for (let sceneIndex = this.sceneIndex - 1; sceneIndex >= 0; sceneIndex -= 1) {
        await this.prepare_scene(sceneIndex);
        const previousPlan = this.plan!;
        for (let index = previousPlan.pages.length - 1; index >= 0; index -= 1) {
          if (previousPlan.pages[index]!.blueprint !== null) {
            await this.jump_to_page(sceneIndex, index, "blueprint");
            return;
          }
        }
      }
      if (this.sceneIndex !== originScene) {
        await this.jump_to_page(originScene, originPage, "blueprint");
      }
      return;
    }
    if (this.pageIndex > 0) {
      const previous = plan.pages[this.pageIndex - 1]!;
      const subphase = previous.construct !== null ? "construct" : "blueprint";
      await this.jump_to_page(this.sceneIndex, this.pageIndex - 1, subphase);
      return;
    }
    if (this.sceneIndex > 0) {
      await this.prepare_scene(this.sceneIndex - 1);
      const previousPlan = this.plan!;
      if (previousPlan.pages.length === 0) return;
      const index = previousPlan.pages.length - 1;
      const previous = previousPlan.pages[index]!;
      const subphase = previous.construct !== null ? "construct" : "blueprint";
      await this.jump_to_page(this.sceneIndex, index, subphase);
    }
  }

  // --- scene lifecycle -------------------------------------------------------

  private async prepare_scene(index: number): Promise<void> {
    if (!this.host.live) {
      // Keep a single instance for headless runs; rebuilding would duplicate
      // work for no observable benefit.
      if (this.scene === null || this.sceneIndex !== index) {
        this.activate_scene(index);
      }
    } else if (this.scene === null || this.sceneIndex !== index) {
      this.activate_scene(index);
    }
    const scene = this.scene;
    if (scene === null) return;
    this.reset_scene(scene);
    scene.setup();
    this.plan = as_interactive(scene)?.two_view_plan() ?? plain_plan(scene);
    this.pageIndex = 0;
    this.subphase = this.plan.has_blueprint ? "blueprint" : "construct";
    as_interactive(scene)?.attach_host(this.host);
    this.apply_theme(scene);
    this.onSceneActivated?.(scene, this.sceneIndex, this.sceneClass as SceneConstructor);
  }

  private activate_scene(index: number): void {
    this.dispose_current_scene();
    const sceneClass = this.sceneClasses[index]!;
    const scene = this.createScene(sceneClass);
    this.scene = scene;
    this.sceneIndex = index;
    this.sceneClass = sceneClass;
    this.apply_theme(scene);
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.attach_host(this.host);
      scene.message_sink = (key, value) => this.forward_message(key, value);
    }
    this.phaseScheduler = null;
    const base = scene.run_scheduler;
    if (base !== null) {
      this.phaseScheduler = new PhaseScheduler(base);
      scene.set_run_scheduler(this.phaseScheduler);
    }
  }

  private apply_theme(scene: Scene): void {
    if (this.theme !== undefined) scene.theme = this.theme;
  }

  private dispose_current_scene(): void {
    const scene = this.scene;
    if (scene === null) return;
    this.scene = null;
    as_interactive(scene)?.detach_host();
    this.disposeScene?.(scene);
    this.phaseScheduler = null;
    this.sceneClass = null;
  }

  private reset_scene(scene: Scene): void {
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.presentation_reset();
      interactive.interaction_enabled = true;
      return;
    }
    scene.mobjects = [];
    scene.foreground_mobjects = [];
    scene.moving_mobjects = [];
    scene.static_mobjects = [];
    scene.animations = [];
    scene.updaters = [];
    scene.last_t = 0;
    const renderer = scene.renderer as
      | { clear_screen?: () => void; num_plays?: number; time?: number; drawFrames?: boolean }
      | null
      | undefined;
    if (renderer !== null && renderer !== undefined) {
      renderer.clear_screen?.();
      renderer.num_plays = 0;
      renderer.time = 0;
    }
  }

  /** Rebuild the scene and fast-forward to `pageIndex`/`subphase`. */
  private async jump_to_page(
    sceneIndex: number,
    pageIndex: number,
    subphase: "blueprint" | "construct",
  ): Promise<void> {
    await this.prepare_scene(sceneIndex);
    const plan = this.plan!;
    const clamped = Math.max(0, Math.min(pageIndex, Math.max(0, plan.pages.length - 1)));
    // SEPARATE: interaction shows blueprints only; never enter a construct
    // subphase when the page has a blueprint to show.
    if (plan.separate && plan.pages[clamped]?.blueprint !== null) {
      subphase = "blueprint";
    }
    this.pageIndex = clamped;
    this.subphase = subphase;
    for (let index = 0; index < clamped; index += 1) {
      await this.run_skipped_page(plan.pages[index]!, index);
    }
    const page = plan.pages[clamped];
    if (page === undefined) return;
    if (subphase === "construct") {
      if (page.blueprint !== null) {
        await this.run_skipped_phase({ kind: "blueprint", run: page.blueprint });
        this.restore_control_values(clamped);
        this.settle_skipped_phase();
      }
      if (page.construct !== null) {
        const action = await this.run_construct_live(page);
        if (action !== null) this.pending = action;
      }
      return;
    }
    // Show the blueprint live (or, for a construct-only page, its construct).
    if (page.blueprint !== null) {
      this.set_interaction_enabled(true);
      await this.run_phase({ kind: "blueprint", run: page.blueprint });
      this.restore_control_values(clamped);
      this.settle();
    } else if (page.construct !== null) {
      const action = await this.run_construct_live(page);
      if (action !== null) this.pending = action;
    }
    this.show_interaction_controls();
    this.emit_state();
  }

  private async run_skipped_page(page: InteractivePage, index: number): Promise<void> {
    if (page.blueprint !== null) {
      await this.run_skipped_phase({ kind: "blueprint", run: page.blueprint });
      this.restore_control_values(index);
      this.settle_skipped_phase();
      this.apply_stage_clear();
    }
    // SEPARATE: the two halves are independent, so fast-forwarding to a
    // blueprint must not execute earlier constructs.
    if (this.plan?.separate === true) return;
    if (page.construct !== null) {
      await this.run_skipped_phase({ kind: "construct", run: page.construct });
      this.settle_skipped_phase();
    }
  }

  // --- phase execution -------------------------------------------------------

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

  private async run_skipped_phase(phase: PresentationPhase): Promise<void> {
    const renderer = this.scene?.renderer as
      | { drawFrames?: boolean; set_skip_animations?: (skip: boolean) => void }
      | null
      | undefined;
    const savedDraw = renderer?.drawFrames;
    renderer?.set_skip_animations?.(true);
    if (renderer !== null && renderer !== undefined) renderer.drawFrames = false;
    try {
      await this.run_phase(phase);
    } finally {
      renderer?.set_skip_animations?.(false);
      if (renderer !== null && renderer !== undefined && savedDraw !== undefined) {
        renderer.drawFrames = savedDraw;
      }
    }
  }

  private settle(): void {
    const scene = this.scene;
    if (scene === null) return;
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.presentation_frame();
      return;
    }
    const frameRate = (scene.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    scene.update_mobjects(dt);
    scene.update_self(dt);
    scene.renderer?.render?.(scene, scene.time, scene.moving_mobjects);
  }

  private settle_skipped_phase(): void {
    const scene = this.scene;
    if (scene === null) return;
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.presentation_settle_phase();
      return;
    }
    const frameRate = (scene.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    scene.update_mobjects(dt);
    scene.update_self(dt);
  }

  private tick(): void {
    const scene = this.scene;
    if (scene === null) return;
    const interactive = as_interactive(scene);
    if (interactive !== null) {
      interactive.presentation_frame();
      return;
    }
    const frameRate = (scene.camera?.frame_rate as number | undefined) ?? 30;
    const dt = frameRate > 0 ? 1 / frameRate : 1 / 30;
    scene.update_mobjects(dt);
    scene.update_self(dt);
    scene.renderer?.render?.(scene, scene.time, scene.moving_mobjects);
  }

  private apply_stage_clear(): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    interactive?.presentation_apply_stage_clear();
  }

  private set_interaction_enabled(enabled: boolean): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    if (interactive !== null) interactive.interaction_enabled = enabled;
  }

  // --- waiting / actions -----------------------------------------------------

  private async wait_for_action(): Promise<Action> {
    for (;;) {
      const action = this.take_action();
      if (action !== null) return action;
      if (this.abortRequested) return "abort";
      if (this.scene?.run_aborted()) return "abort";
      this.tick();
      await this.host.nextFrame();
    }
  }

  private async run_watching(run: () => Promise<void>): Promise<Action | null> {
    let settled = false;
    let failure: unknown = null;
    const promise = run().then(
      () => {
        settled = true;
      },
      (error: unknown) => {
        settled = true;
        failure = error;
      },
    );
    try {
      for (;;) {
        if (settled) {
          if (failure !== null) throw failure;
          return null;
        }
        const action = this.take_action();
        if (action !== null) {
          this.phaseScheduler?.cancel();
          await promise;
          if (failure !== null) throw failure;
          return action;
        }
        if (this.abortRequested || this.scene?.run_aborted()) {
          this.phaseScheduler?.cancel();
          await promise;
          return "abort";
        }
        await this.host.nextFrame();
      }
    } finally {
      this.phaseScheduler?.resume();
    }
  }

  private take_action(): Action | null {
    let action = this.pending;
    this.pending = null;
    if (action === null) {
      const interactive = this.scene === null ? null : as_interactive(this.scene);
      const requested = interactive?.requested_action() ?? null;
      if (requested === "abort") action = "abort";
      else if (requested !== null) action = requested;
    }
    if (action === null) return null;
    if (action === "next" && this.view === "animation") return null;
    if (action === "previous" && this.view === "animation") return null;
    if (action === "interact" && !this.has_interaction_here()) return null;
    if (action === "animate" && !this.has_animation_here()) return null;
    return action;
  }

  // --- control values --------------------------------------------------------

  private value_key(pageIndex: number): string {
    return `${this.sceneIndex}:${pageIndex}`;
  }

  private record_control_values(): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    if (interactive === null) return;
    this.controlValues.set(this.value_key(this.pageIndex), interactive.presentation_control_values());
  }

  private restore_control_values(pageIndex: number): void {
    const interactive = this.scene === null ? null : as_interactive(this.scene);
    if (interactive === null) return;
    const values = this.controlValues.get(this.value_key(pageIndex));
    if (values !== undefined) interactive.presentation_set_control_values(values);
  }

  // --- feature predicates ----------------------------------------------------

  private current_page(): InteractivePage | null {
    return this.plan?.pages[this.pageIndex] ?? null;
  }

  private has_interaction_here(): boolean {
    return this.plan?.has_blueprint === true;
  }

  private has_animation_here(): boolean {
    return this.plan?.has_construct === true;
  }

  private can_next(): boolean {
    if (this.view === "animation") return false;
    const page = this.current_page();
    if (page === null) return false;
    if (this.plan?.separate === true) {
      // SEPARATE: the paired construct is not a next step; only a later
      // blueprint page counts.
      return this.has_next_interaction_page();
    }
    if (this.subphase === "blueprint" && page.construct !== null) return true;
    return this.has_next_page();
  }

  private can_previous(): boolean {
    if (this.view === "animation") return false;
    if (this.plan?.separate === true) {
      return this.has_previous_interaction_page();
    }
    if (this.subphase === "construct") return true;
    return this.pageIndex > 0 || this.sceneIndex > 0;
  }

  private has_next_page(): boolean {
    if (this.plan !== null && this.pageIndex + 1 < this.plan.pages.length) return true;
    return this.sceneIndex + 1 < this.sceneClasses.length;
  }

  private has_next_interaction_page(): boolean {
    if (this.plan !== null) {
      for (let index = this.pageIndex + 1; index < this.plan.pages.length; index += 1) {
        if (this.plan.pages[index]!.blueprint !== null) return true;
      }
    }
    // Later scenes are not loaded yet; optimistically report a next page so
    // cross-scene blueprints stay reachable. `goto_next_page` stays put when
    // none exists.
    return this.sceneIndex + 1 < this.sceneClasses.length;
  }

  private has_previous_interaction_page(): boolean {
    if (this.plan !== null) {
      for (let index = this.pageIndex - 1; index >= 0; index -= 1) {
        if (this.plan.pages[index]!.blueprint !== null) return true;
      }
    }
    return this.sceneIndex > 0;
  }

  // --- messages --------------------------------------------------------------

  private forward_message(key: unknown, value: unknown): void {
    if (this.onMessage === undefined) return;
    this.onMessage({
      id: this.instanceId,
      key,
      value,
      scene: this.sceneClass?.name ?? "",
      sceneIndex: this.sceneIndex,
      pageIndex: this.pageIndex,
      phaseKind: this.subphase,
    });
  }

  // --- controls --------------------------------------------------------------

  private install_controls(): void {
    this.hide_controls();
    this.disposers.push(
      this.host.addButton(PRESENTATION_CONTROL_LABELS.previous, () => this.previous(), "previous"),
    );
    this.disposers.push(
      this.host.addButton(PRESENTATION_CONTROL_LABELS.replay, () => this.replay(), "replay"),
    );
    this.disposers.push(this.host.addButton(PRESENTATION_CONTROL_LABELS.next, () => this.next(), "next"));
    this.disposers.push(
      this.host.addButton(
        PRESENTATION_CONTROL_LABELS.interact,
        () => this.enterInteraction(),
        "interact",
      ),
    );
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

  private install_double_click(): void {
    if (this.mode !== "two-view" || typeof this.host.onDoubleClick !== "function") return;
    this.disposers.push(
      this.host.onDoubleClick(() => {
        if (this.view === "animation") {
          if (this.has_interaction_here()) this.pending = "interact";
        } else if (this.has_animation_here()) {
          this.pending = "animate";
        }
      }),
    );
  }

  private show_animation_controls(plan: TwoViewPlan | null): void {
    this.apply_control_theme();
    this.host.setButtonsEnabled(true);
    this.host.setButtonVisible?.("previous", false);
    this.host.setButtonVisible?.(
      "replay",
      this.mode !== "animation",
    );
    this.host.setButtonVisible?.("next", false);
    this.host.setButtonVisible?.(
      "interact",
      this.mode === "two-view" && plan?.has_blueprint === true,
    );
    this.emit_state();
  }

  private show_interaction_controls(): void {
    this.apply_control_theme();
    const canPrevious = this.can_previous();
    const canNext = this.can_next();
    this.host.setButtonsEnabled(true);
    this.host.setButtonVisible?.("previous", canPrevious);
    this.host.setButtonVisible?.("replay", true);
    this.host.setButtonVisible?.("next", canNext);
    this.host.setButtonVisible?.("interact", false);
    this.emit_state();
  }

  /**
   * Tell the host whether the current canvas backdrop is light, so the icon
   * bar can switch to dark-grey glyphs.  Re-evaluated whenever the controls
   * become visible: a scene may change `camera.background_color` while it
   * builds its blueprint.
   */
  private apply_control_theme(): void {
    if (typeof this.host.setControlTheme !== "function") return;
    const camera = this.scene?.camera as
      | { background_color?: { to_hex?: () => string }; background_opacity?: number }
      | null
      | undefined;
    const hex = camera?.background_color?.to_hex?.();
    if (typeof hex !== "string") return;
    this.host.setControlTheme(dark_from_hex(hex, camera?.background_opacity ?? 1));
  }

  private hide_controls(): void {
    this.host.setButtonsEnabled(false);
    this.host.setButtonVisible?.("previous", false);
    this.host.setButtonVisible?.("replay", false);
    this.host.setButtonVisible?.("next", false);
    this.host.setButtonVisible?.("interact", false);
  }

  private emit_state(): void {
    this.onStateChange?.(this.state);
  }
}
