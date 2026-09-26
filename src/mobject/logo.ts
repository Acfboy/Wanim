import type { NDArray } from "numpy-ts";
import {
  DEFAULT_FONT_SIZE,
  DOWN,
  LEFT,
  ORIGIN,
  RIGHT,
  SCALE_FACTOR_PER_FONT_POINT,
  UP,
} from "../constants.js";
import type { ParsableManimColor } from "../utils/color/core.js";
import { ease_in_out_cubic, smooth } from "../utils/rate_functions.js";
import { override_animation } from "../animation/animation.js";
import { AnimationGroup, Succession } from "../animation/composition.js";
import { Create, SpiralIn } from "../animation/creation.js";
import { FadeIn } from "../animation/fading.js";
import { UpdateFromAlphaFunc } from "../animation/updaters/update.js";
import { Mobject } from "./mobject.js";
import { VGroup, VMobject } from "./types/vectorized_mobject.js";
import { VMobjectFromSVGPath } from "./svg/svg_mobject.js";
import { Circle } from "./geometry/arc.js";
import { Square, Triangle } from "./geometry/polygram.js";

/** SVG path data for Manim's banner letters (M, a, n, i, m).
 * Mirrors Python MANIM_SVG_PATHS (svgelements.Path strings). */
export const MANIM_SVG_PATHS: string[] = [
  "M4.64259-2.092154L2.739726-6.625156C2.660025-6.824408 2.650062-6.824408 2.381071-6.824408H.52802C.348692-6.824408 .199253-6.824408 .199253-6.645081C.199253-6.475716 .37858-6.475716 .428394-6.475716C.547945-6.475716 .816936-6.455791 1.036115-6.37609V-1.05604C1.036115-.846824 1.036115-.408468 .358655-.348692C.169365-.328767 .169365-.18929 .169365-.179328C.169365 0 .328767 0 .508095 0H2.052304C2.231631 0 2.381071 0 2.381071-.179328C2.381071-.268991 2.30137-.33873 2.221669-.348692C1.454545-.408468 1.454545-.826899 1.454545-1.05604V-6.017435L1.464508-6.027397L3.895392-.209215C3.975093-.029888 4.044832 0 4.104608 0C4.224159 0 4.254047-.079701 4.303861-.199253L6.744707-6.027397L6.75467-6.017435V-1.05604C6.75467-.846824 6.75467-.408468 6.07721-.348692C5.88792-.328767 5.88792-.18929 5.88792-.179328C5.88792 0 6.047323 0 6.22665 0H8.886675C9.066002 0 9.215442 0 9.215442-.179328C9.215442-.268991 9.135741-.33873 9.05604-.348692C8.288917-.408468 8.288917-.826899 8.288917-1.05604V-5.768369C8.288917-5.977584 8.288917-6.41594 8.966376-6.475716C9.066002-6.485679 9.155666-6.535492 9.155666-6.645081C9.155666-6.824408 9.006227-6.824408 8.826899-6.824408H6.90411C6.645081-6.824408 6.625156-6.824408 6.535492-6.615193L4.64259-2.092154ZM4.343711-1.912827C4.423412-1.743462 4.433375-1.733499 4.552927-1.693649L4.11457-.637609H4.094645L1.823163-6.057285C1.77335-6.1868 1.693649-6.356164 1.554172-6.475716H2.420922L4.343711-1.912827ZM1.334994-.348692H1.165629C1.185554-.37858 1.205479-.408468 1.225405-.428394C1.235367-.438356 1.235367-.448319 1.24533-.458281L1.334994-.348692ZM7.103362-6.475716H8.159402C7.940224-6.22665 7.940224-5.967621 7.940224-5.788294V-1.036115C7.940224-.856787 7.940224-.597758 8.169365-.348692H6.884184C7.103362-.597758 7.103362-.856787 7.103362-1.036115V-6.475716Z",
  "M1.464508-4.024907C1.464508-4.234122 1.743462-4.393524 2.092154-4.393524C2.669988-4.393524 2.929016-4.124533 2.929016-3.516812V-2.789539C1.77335-2.440847 .249066-2.042341 .249066-.916563C.249066-.308842 .71731 .139477 1.354919 .139477C1.92279 .139477 2.381071-.059776 2.929016-.557908C3.038605-.049813 3.257783 .139477 3.745953 .139477C4.174346 .139477 4.483188-.019925 4.861768-.428394L4.712329-.637609L4.612702-.537983C4.582814-.508095 4.552927-.498132 4.503113-.498132C4.363636-.498132 4.293898-.587796 4.293898-.747198V-3.347447C4.293898-4.184309 3.536737-4.712329 2.321295-4.712329C1.195517-4.712329 .438356-4.204234 .438356-3.457036C.438356-3.048568 .67746-2.799502 1.085928-2.799502C1.484433-2.799502 1.763387-3.038605 1.763387-3.377335C1.763387-3.676214 1.464508-3.88543 1.464508-4.024907ZM2.919054-.996264C2.650062-.687422 2.450809-.56787 2.211706-.56787C1.912827-.56787 1.703611-.836862 1.703611-1.235367C1.703611-1.8132 2.122042-2.231631 2.919054-2.440847V-.996264Z",
  "M2.948941-4.044832C3.297634-4.044832 3.466999-3.775841 3.466999-3.217933V-.806974C3.466999-.438356 3.337484-.278954 2.998755-.239103V0H5.339975V-.239103C4.951432-.268991 4.851806-.388543 4.851806-.806974V-3.307597C4.851806-4.164384 4.323786-4.712329 3.506849-4.712329C2.909091-4.712329 2.450809-4.433375 2.082192-3.845579V-4.592777H.179328V-4.353674C.617684-4.283935 .707347-4.184309 .707347-3.765878V-.836862C.707347-.418431 .627646-.328767 .179328-.239103V0H2.580324V-.239103C2.211706-.288917 2.092154-.438356 2.092154-.806974V-3.466999C2.092154-3.576588 2.530511-4.044832 2.948941-4.044832Z",
  "M2.15193-4.592777H.239103V-4.353674C.67746-4.26401 .767123-4.174346 .767123-3.765878V-.836862C.767123-.428394 .697385-.348692 .239103-.239103V0H2.6401V-.239103C2.291407-.288917 2.15193-.428394 2.15193-.806974V-4.592777ZM1.454545-6.884184C1.026152-6.884184 .67746-6.535492 .67746-6.117061C.67746-5.668742 1.006227-5.339975 1.444583-5.339975S2.221669-5.668742 2.221669-6.107098C2.221669-6.535492 1.882939-6.884184 1.454545-6.884184Z",
  "M2.929016-4.044832C3.317559-4.044832 3.466999-3.815691 3.466999-3.217933V-.806974C3.466999-.398506 3.35741-.268991 2.988792-.239103V0H5.32005V-.239103C4.971357-.278954 4.851806-.428394 4.851806-.806974V-3.466999C4.851806-3.576588 5.310087-4.044832 5.69863-4.044832C6.07721-4.044832 6.22665-3.805729 6.22665-3.217933V-.806974C6.22665-.388543 6.117061-.268991 5.738481-.239103V0H8.109589V-.239103C7.721046-.259029 7.611457-.37858 7.611457-.806974V-3.307597C7.611457-4.164384 7.083437-4.712329 6.266501-4.712329C5.69863-4.712329 5.32005-4.483188 4.801993-3.845579C4.503113-4.473225 4.154421-4.712329 3.526775-4.712329S2.440847-4.443337 2.062267-3.845579V-4.592777H.179328V-4.353674C.617684-4.293898 .707347-4.174346 .707347-3.765878V-.836862C.707347-.428394 .617684-.318804 .179328-.239103V0H2.550436V-.239103C2.201743-.288917 2.092154-.428394 2.092154-.806974V-3.466999C2.092154-3.58655 2.530511-4.044832 2.929016-4.044832Z",
];

export class ManimBanner extends VGroup {
  font_color: ParsableManimColor;
  scale_factor: number;
  M: VMobjectFromSVGPath;
  circle: Circle;
  square: Square;
  triangle: Triangle;
  shapes: VGroup;
  anim: VGroup;

  constructor(dark_theme = true) {
    super();

    const logo_green = "#81b29a";
    const logo_blue = "#454866";
    const logo_red = "#e07a5f";
    const m_height_over_anim_height = 0.75748;

    this.font_color = dark_theme ? "#ece6e2" : "#343434";
    this.scale_factor = 1.0;

    this.M = new VMobjectFromSVGPath(MANIM_SVG_PATHS[0] as string);
    this.M.flip([...RIGHT]).center();
    this.M.set({ stroke_width: 0 }).scale(
      7 * DEFAULT_FONT_SIZE * SCALE_FACTOR_PER_FONT_POINT,
    );
    this.M.set_fill(this.font_color, 1).shift([
      2.25 * (LEFT[0] as number),
      1.5 * (UP[1] as number),
      0,
    ]);

    this.circle = new Circle(null, { color: logo_green, fill_opacity: 1 }).shift([...LEFT]);
    this.square = new Square(2.0, { color: logo_blue, fill_opacity: 1 }).shift([...UP]);
    this.triangle = new Triangle({ color: logo_red, fill_opacity: 1 }).shift([...RIGHT]);
    this.shapes = new VGroup(this.triangle, this.square, this.circle);
    this.add(this.shapes, this.M);
    this.move_to([...ORIGIN]);

    const anim = new VGroup();
    const letterPaths = MANIM_SVG_PATHS.slice(1);
    for (let ind = 0; ind < letterPaths.length; ind += 1) {
      const tex = new VMobjectFromSVGPath(letterPaths[ind] as string);
      tex.flip([...RIGHT]).center();
      tex.set({ stroke_width: 0 }).scale(
        DEFAULT_FONT_SIZE * SCALE_FACTOR_PER_FONT_POINT,
      );
      if (ind > 0) {
        tex.next_to(anim, [...RIGHT], 0.01);
      }
      tex.align_to(this.M, [...DOWN]);
      anim.add(tex);
    }
    anim.set_fill(this.font_color, 1);
    anim.height = m_height_over_anim_height * this.M.height;

    // NOTE: "anim" is only shown in the expanded state
    // and thus not yet added to the submobjects of self.
    this.anim = anim;
  }

  override scale(
    scale_factor: number,
    // NOTE: same positional convention as VMobject.scale: the second slot
    // is either a scale_stroke flag or an about_point.
    scale_stroke_or_about_point: boolean | readonly number[] | NDArray | null = false,
    about_point?: readonly number[] | NDArray | null,
    about_edge?: readonly number[] | NDArray | null,
  ): this {
    this.scale_factor *= scale_factor;
    let scale_stroke = false;
    let resolved_about_point = about_point;
    let resolved_about_edge = about_edge;
    if (typeof scale_stroke_or_about_point === "boolean") {
      scale_stroke = scale_stroke_or_about_point;
    } else if (scale_stroke_or_about_point !== undefined && scale_stroke_or_about_point !== null) {
      resolved_about_point = scale_stroke_or_about_point;
    }
    // NOTE: self.anim is only added to self after expand()
    if (!this.submobjects.includes(this.anim)) {
      this.anim.scale(
        scale_factor,
        scale_stroke,
        resolved_about_point ?? undefined,
        resolved_about_edge ?? undefined,
      );
    }
    super.scale(
      scale_factor,
      scale_stroke,
      resolved_about_point ?? undefined,
      resolved_about_edge ?? undefined,
    );
    return this;
  }

  create(run_time = 2): AnimationGroup {
    return new AnimationGroup(
      new SpiralIn(this.shapes, { run_time }),
      new FadeIn(this.M, { run_time: run_time / 2 }),
      { lag_ratio: 0.1 },
    );
  }

  expand(run_time = 1.5, direction = "center"): Succession {
    if (direction !== "left" && direction !== "right" && direction !== "center") {
      const error = new Error("direction must be 'left', 'right' or 'center'.");
      error.name = "ValueError";
      throw error;
    }

    const m_shape_offset = 6.25 * this.scale_factor;
    const shape_sliding_overshoot = this.scale_factor * 0.8;
    const m_anim_buff = 0.06;
    this.anim.next_to(this.M, [...RIGHT], m_anim_buff).align_to(this.M, [...DOWN]);
    this.anim.set_opacity(0);
    this.shapes.save_state();
    const m_clone = (this.anim.__getitem__(-1) as Mobject).copy();
    this.add(m_clone);
    m_clone.move_to(this.shapes);

    this.M.save_state();
    const left_group = new VGroup(this.M, this.anim, m_clone as VMobject);

    const shift = (vector: number[]): void => {
      this.shapes.restore();
      left_group.align_to(this.M.saved_state as Mobject, [...LEFT]);
      if (direction === "right") {
        this.shapes.shift(vector);
      } else if (direction === "center") {
        this.shapes.shift(vector.map((value) => value / 2));
        left_group.shift(vector.map((value) => -value / 2));
      } else if (direction === "left") {
        left_group.shift(vector.map((value) => -value));
      }
    };

    const self = this;
    const slide_and_uncover = (mob: Mobject, alpha: unknown): void => {
      const banner = mob as ManimBanner;
      const a = alpha as number;
      shift([
        a * (m_shape_offset + shape_sliding_overshoot) * (RIGHT[0] as number),
        a * (m_shape_offset + shape_sliding_overshoot) * (RIGHT[1] as number),
        a * (m_shape_offset + shape_sliding_overshoot) * (RIGHT[2] as number),
      ]);

      // Add letters when they are covered
      for (const letter of banner.anim) {
        if (banner.square.get_center()[0]! > (letter as Mobject).get_center()[0]!) {
          (letter as Mobject).set_opacity(1);
          self.add_to_back(letter as Mobject);
        }
      }

      // Finish animation
      if (a === 1) {
        self.remove(self.anim);
        self.add_to_back(self.anim);
        banner.shapes.set_z_index(0);
        banner.shapes.save_state();
        banner.M.save_state();
      }
    };

    const slide_back = (mob: Mobject, alpha: unknown): void => {
      const banner = mob as ManimBanner;
      const a = alpha as number;
      if (a === 0) {
        m_clone.set_opacity(1);
        m_clone.move_to(banner.anim.__getitem__(-1) as Mobject);
        banner.anim.set_opacity(1);
      }

      shift([
        a * shape_sliding_overshoot * (LEFT[0] as number),
        a * shape_sliding_overshoot * (LEFT[1] as number),
        a * shape_sliding_overshoot * (LEFT[2] as number),
      ]);

      if (a === 1) {
        banner.remove(m_clone);
        banner.add_to_back(banner.shapes);
      }
    };

    return new Succession(
      new UpdateFromAlphaFunc(
        this,
        slide_and_uncover,
        { run_time: (run_time * 2) / 3, rate_func: ease_in_out_cubic },
      ),
      new UpdateFromAlphaFunc(
        this,
        slide_back,
        { run_time: (run_time * 1) / 3, rate_func: smooth },
      ),
    );
  }
}

override_animation(Create)(ManimBanner.prototype.create);
