import { describe, expect, it, vi } from "vitest";
import { array } from "numpy-ts";
import {
  Group,
  Mobject,
  _AnimationBuilder,
  _UpdaterBuilder,
  override_animate,
} from "../src/mobject/mobject.js";
import {
  Animation,
  override_animation,
  getAnimationOverride,
  linear,
  type AnimationClass,
} from "../src/animation/animation.js";
import { AnimationGroup } from "../src/animation/composition.js";
import { Create, ShowPartial, ShowIncreasingSubsets } from "../src/animation/creation.js";
import { ChangeSpeed } from "../src/animation/speedmodifier.js";
import { ComplexHomotopy, Homotopy } from "../src/animation/movement.js";
import { ApplyMatrix, FadeTransform } from "../src/animation/transform.js";
import { bind_mobject_method, f_always } from "../src/animation/updaters/mobject_update_utils.js";
import { pointRows } from "../src/utils/arrays.js";
import { VMobject, VGroup, VectorizedPoint } from "../src/mobject/types/vectorized_mobject.js";
import { Scene } from "../src/scene/scene.js";

function point(value: unknown): number[] {
  return pointRows(value, 3)[0] ?? [0, 0, 0];
}

function matrix(value: unknown): number[][] {
  return pointRows(value, 3);
}

class TrackedPoint extends VectorizedPoint {
  opacityValue = 1;

  set_opacity(opacity: number, _family = true): this {
    this.opacityValue = opacity;
    return this;
  }
}

function thrown(action: () => unknown): Error {
  try {
    action();
  } catch (error) {
    return error as Error;
  }
  throw new Error("Expected an exception");
}

describe("assigned animation symbols", () => {
  it("leaves Scene key state unchanged on key release", () => {
    const scene = new Scene();
    const callback = vi.fn();
    scene.interactive_mode = true;
    scene.key_to_function_map.set("x", callback);

    expect(scene.on_key_release("x".charCodeAt(0), 0)).toBeUndefined();
    expect(scene.interactive_mode).toBe(true);
    expect(callback).not.toHaveBeenCalled();
  });

  it("builds AnimationGroup timings with cumulative lag", () => {
    const first = new Animation(new Mobject(), { run_time: 2 });
    const second = new Animation(new Mobject(), { run_time: 1 });
    const third = new Animation(new Mobject(), { run_time: 3 });
    const group = new AnimationGroup(first, second, third, { lag_ratio: 0.5 });

    group.build_animations_with_timings();

    expect(group.anims_with_timings.start).toEqual([0, 1, 1.5]);
    expect(group.anims_with_timings.end).toEqual([2, 2, 4.5]);
    expect(group.anims_with_timings[0]?.anim).toBe(first);
    expect(group.anims_with_timings[1]?.anim).toBe(second);
    expect(group.anims_with_timings[2]?.anim).toBe(third);
    expect(group.max_end_time).toBe(4.5);
  });

  it("reports VGroup add type errors like Python", () => {
    const group = new VGroup();
    const numberError = thrown(() => group.add(7));
    const mobjectError = thrown(() => group.add(new Mobject()));
    const nestedError = thrown(() => group.add([new VMobject(), 7]));

    expect(numberError).toBeInstanceOf(TypeError);
    expect(numberError.message).toBe(
      "Only values of type VMobject can be added as submobjects of VGroup, but the value 7 (at index 0 of parameter 0) is of type int.",
    );
    expect(mobjectError).toBeInstanceOf(TypeError);
    expect(mobjectError.message).toBe(
      "Only values of type VMobject can be added as submobjects of VGroup, but the value Mobject (at index 0 of parameter 0) is of type Mobject. You can try adding this value into a Group instead.",
    );
    expect(nestedError).toBeInstanceOf(TypeError);
    expect(nestedError.message).toBe(
      "Only values of type VMobject can be added as submobjects of VGroup, but the value 7 (at index 1 of parameter 0) is of type int.",
    );
  });

  it("exposes Animation flags and run-time accessors", () => {
    const plain = new Animation(new Mobject());
    const flagged = new Animation(new Mobject(), { introducer: true, remover: true });

    expect(plain.is_introducer()).toBe(false);
    expect(plain.is_remover()).toBe(false);
    expect(flagged.is_introducer()).toBe(true);
    expect(flagged.is_remover()).toBe(true);
    expect(flagged.get_run_time()).toBe(1);
    expect(flagged.set_run_time(2.5)).toBe(flagged);
    expect(flagged.get_run_time()).toBe(2.5);
    expect(thrown(() => flagged.set_run_time(-1)).message).toBe(
      "The run_time of Animation cannot be negative. The given value was -1.",
    );
  });

  it("raises the Python NotImplementedError from ShowPartial bounds", () => {
    const partial = new ShowPartial(new VMobject());
    const error = thrown(() => partial._get_bounds(0.4));

    expect(error.name).toBe("NotImplementedError");
    expect(error.message).toBe("Please use Create or ShowPassingFlash");
  });

  it("uses the Python Create bounds implementation", () => {
    const create = new Create(new VMobject());

    expect(create._get_bounds(0)).toEqual([0, 0]);
    expect(create._get_bounds(0.25)).toEqual([0, 0.25]);
    expect(create._get_bounds(1)).toEqual([0, 1]);
  });

  it("updates increasing subset opacities at the requested boundary", () => {
    const children = [new TrackedPoint(), new TrackedPoint(), new TrackedPoint()];
    const animation = new ShowIncreasingSubsets(new Group(...children));

    expect(children.map((child) => child.opacityValue)).toEqual([0, 0, 0]);
    animation.update_submobject_list(2);
    expect(children.map((child) => child.opacityValue)).toEqual([1, 1, 0]);
    animation.update_submobject_list(0);
    expect(children.map((child) => child.opacityValue)).toEqual([0, 0, 0]);
    animation.update_submobject_list(3);
    expect(children.map((child) => child.opacityValue)).toEqual([1, 1, 1]);
  });

  it("evaluates a Homotopy mapping function at the requested time", () => {
    const animation = new Homotopy(
      (x, y, z, t) => [x + 2 * t, y + 3 * t, z + 4 * t],
      new Mobject(),
    );
    const result = animation.function_at_time_t(0.25)(array([1, 2, 3]));

    expect(point(result)).toEqual([1.5, 2.75, 4]);
  });

  it("wraps a complex homotopy while preserving the input height", () => {
    const animation = new ComplexHomotopy(
      ({ re, im }, t) => ({ re: re + 2 * t, im: im - 4 * t }),
      new Mobject(),
    );

    expect(point(animation.homotopy(3, 4, 5, 0.5))).toEqual([4, 2, 5]);
  });

  it("expands 2 by 2 matrices and validates dimensions", () => {
    expect(matrix(ApplyMatrix.initialize_matrix([[1, 2], [3, 4]]))).toEqual([
      [1, 2, 0],
      [3, 4, 0],
      [0, 0, 1],
    ]);
    expect(
      matrix(
        ApplyMatrix.initialize_matrix([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]),
      ),
    ).toEqual([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    const error = thrown(() => ApplyMatrix.initialize_matrix([[1, 2, 3], [4, 5]]));
    expect(error.name).toBe("ValueError");
    expect(error.message).toBe("Matrix has bad dimensions");
  });

  it("ghosts a source to a pointful target and hides it", () => {
    const source = new TrackedPoint([1, 2, 0]);
    const target = new VectorizedPoint([3, 4, 0]);
    const animation = new FadeTransform(source, target, { stretch: false, dim_to_match: 0 });

    animation.ghost_to(source, target);

    expect(point(source.points)).toEqual([3, 4, 0]);
    expect(source.opacityValue).toBe(0);
  });

  it("hides a source without replacing it when the target is empty", () => {
    const source = new TrackedPoint([1, 2, 3]);
    const target = new Mobject();
    const animation = new FadeTransform(source, target);

    animation.ghost_to(source, target);

    expect(point(source.points)).toEqual([1, 2, 3]);
    expect(source.opacityValue).toBe(0);
  });

  it("registers an animation override and returns the decorated function", () => {
    class CustomAnimation extends Animation {}
    const method = (): undefined => undefined;
    const decorated = override_animation(CustomAnimation)(method);

    expect(decorated).toBe(method);
    expect((method as typeof method & { _override_animation: AnimationClass })._override_animation).toBe(CustomAnimation);
    expect(getAnimationOverride(method)).toBe(CustomAnimation);
  });

  it("computes ChangeSpeed's scaled total time from adjacent speed nodes", () => {
    const animation = new ChangeSpeed(
      new Animation(new Mobject(), { run_time: 2 }),
      { 0: 1, 0.5: 2, 1: 1 },
      linear,
      false,
    );

    expect(animation.get_scaled_total_time()).toBeCloseTo(2 / 3, 12);
  });

  it("evaluates f_always generators on every updater call", () => {
    const mobject = new VectorizedPoint([0, 0, 0]);
    let calls = 0;
    let value = 3;
    const result = f_always(bind_mobject_method(mobject, "set_x"), () => {
      calls += 1;
      return value;
    });

    expect(result).toBe(mobject);
    mobject.update(0);
    expect(mobject.get_x()).toBe(3);
    value = 5;
    mobject.update(0);
    expect(mobject.get_x()).toBe(5);
    expect(calls).toBe(2);
  });

  it("builds and applies a normal AnimationBuilder method", () => {
    const mobject = new VectorizedPoint([1, 0, 0]);
    const builder = mobject.animate as unknown as _AnimationBuilder & Record<string, (...args: any[]) => any>;
    builder.call({ run_time: 2.5 });
    const animation = builder.shift([0, 2, 0]).build();

    expect(animation).toBeInstanceOf(Animation);
    expect(animation.get_run_time()).toBe(2.5);
    animation.finish();
    expect(point(mobject.points)).toEqual([1, 2, 0]);
  });

  it("rejects repeated animation arguments with Python's ValueError", () => {
    const mobject = new VectorizedPoint([1, 0, 0]);
    const builder = mobject.animate as unknown as _AnimationBuilder;
    builder.call({ run_time: 1 });
    const error = thrown(() => builder.call({ run_time: 2 }));
    expect(error.name).toBe("ValueError");
    expect(error.message).toBe(
      "Animation arguments must be passed before accessing methods and can only be passed once",
    );
  });

  it("uses override_animate when an AnimationBuilder method is marked", () => {
    class MarkedMobject extends Mobject {
      operation(): this {
        return this;
      }
    }

    const method = MarkedMobject.prototype.operation;
    const factory = (mobject: Mobject, ...args: any[]): Animation =>
      new Animation(mobject, args[args.length - 1] as Record<string, unknown>);
    override_animate(method)(factory);
    const mobject = new MarkedMobject();
    const builder = mobject.animate as unknown as Record<string, (...args: any[]) => any>;
    const animation = builder.operation().build();

    expect(animation).toBeInstanceOf(Animation);
    expect(animation.mobject).toBe(mobject);
  });

  it("adds an updater immediately through the _UpdaterBuilder proxy", () => {
    const mobject = new VectorizedPoint([0, 0, 0]);
    const builder = mobject.always as unknown as _UpdaterBuilder & Record<string, (...args: any[]) => any>;

    builder.shift([2, 0, 0]);

    expect(mobject.updaters).toHaveLength(1);
    expect(point(mobject.points)).toEqual([2, 0, 0]);
    mobject.shift([-1, 0, 0]);
    mobject.update(0);
    expect(point(mobject.points)).toEqual([3, 0, 0]);
  });

  it("rejects invalid nested VGroup values with Python type names", () => {
    const group = new VGroup();
    expect(() => group.add(null as unknown as VMobject)).toThrowError(
      "the value None (at index 0 of parameter 0) is of type NoneType.",
    );
    expect(() => group.add(1.5 as unknown as VMobject)).toThrowError(
      "the value 1.5 (at index 0 of parameter 0) is of type float.",
    );
  });

  it("rejects fractional subset indices", () => {
    const children = [new TrackedPoint(), new TrackedPoint()];
    const animation = new ShowIncreasingSubsets(new Group(...children));
    expect(() => animation.update_submobject_list(1.5)).toThrowError(TypeError);
  });

  it("copies homotopy arrays and rejects malformed arity", () => {
    const animation = new Homotopy((x, y, z) => [x + 1, y + 2, z + 3], new Mobject());
    const source = array([1, 2, 3]);
    const result = animation.function_at_time_t(0.5)(source);
    expect(result).not.toBe(source);
    expect(point(result)).toEqual([2, 4, 6]);
    expect(() => animation.function_at_time_t(0.5)([1, 2])).toThrowError(/unpack/);
  });

  it("accepts scalar complex homotopy results", () => {
    const animation = new ComplexHomotopy(() => 1.5, new Mobject());
    expect(point(animation.homotopy(4, 5, 6, 0.25))).toEqual([1.5, 0, 6]);
  });

  it("applies non-stretching ghost replacement along the selected dimension", () => {
    const source = new Mobject();
    source.set_points([[0, 0, 0], [2, 0, 0]]);
    const target = new Mobject();
    target.set_points([[0, 0, 0], [4, 0, 0]]);
    const animation = new FadeTransform(source, target, { stretch: false, dim_to_match: 0 });
    animation.ghost_to(source, target);
    expect(source.points.toArray()).toEqual([[0, 0, 0], [4, 0, 0]]);
    expect(source.get_color().toRgba()[3]).toBe(0);
  });
});
