import { describe, expect, it } from "vitest";
import { DecimalNumber } from "../src/mobject/text/numbers.js";
import { Text } from "../src/mobject/text/text_mobject.js";
import { MathTex } from "../src/mobject/text/tex_mobject.js";
import { VGroup } from "../src/mobject/types/vectorized_mobject.js";
import { FadeIn } from "../src/animation/fading.js";
import { BLUE, GREY_A, RED } from "../src/utils/color/core.js";

/**
 * Regression: `FadeIn` on a group holding a `DecimalNumber` with an updater
 * used to leave the number collapsed.  The updater runs on the animation's
 * `starting_mobject`/`target_copy` while `Transform` reuses a zip of the
 * families computed once in `begin`; rebuilding the digits mid-animation
 * replaced the glyph children, so the cached zip interpolated against
 * detached (blanked) leaves and zeroed the original's points.  The next
 * `set_value` then read `font_size == 0` and threw
 * "font_size must be greater than 0." (slides calc1_limits/calc5).
 */
describe("DecimalNumber updater inside a FadeIn group", () => {
  it("keeps its height and set_value does not throw after the animation", () => {
    const dnX = new DecimalNumber(3.0, { num_decimal_places: 2, font_size: 30, color: RED });
    const dnY = new DecimalNumber(1.0, { num_decimal_places: 4, font_size: 30, color: BLUE });
    dnX.add_updater((m) => (m as DecimalNumber).set_value(2.95));
    dnY.add_updater((m) => (m as DecimalNumber).set_value(0.04704));
    const readout = new VGroup(
      new Text("x =", { font_size: 26, color: GREY_A }),
      dnX,
      new MathTex("\\frac{\\sin x}{x}=", { font_size: 34, color: BLUE }),
      dnY,
    );
    readout.arrange({ direction: [1, 0, 0], buff: 0.15 });
    readout.move_to([0, 2.2, 0]);

    const animation = new FadeIn(readout);
    animation.begin();
    for (let frame = 0; frame <= 5; frame += 1) {
      animation.update_mobjects(1 / 15);
      animation.interpolate(frame / 5);
    }
    expect(() => animation.finish()).not.toThrow();
    expect(dnX.get_height()).toBeGreaterThan(0);
    expect(dnY.get_height()).toBeGreaterThan(0);
  });

  it("reuses the zipped families until the structure changes", () => {
    const dn = new DecimalNumber(1.5);
    const animation = new FadeIn(dn);
    animation.begin();
    const first = animation.get_all_families_zipped();
    expect(first.length).toBeGreaterThan(0);
    // Point-only frames keep the cache.
    expect(animation.get_all_families_zipped() === first).toBe(true);
    // Replacing a child list invalidates it.
    dn.submobjects = [...dn.submobjects];
    expect(animation.get_all_families_zipped() === first).toBe(false);
  });
});
