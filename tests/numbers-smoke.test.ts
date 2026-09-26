import { describe, expect, it } from "vitest";
import { DecimalNumber, Integer, Variable, string_to_mob_map } from "../src/mobject/text/numbers.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";

const dummyFactory = (text: string) => {
  void text;
  return new VMobject();
};

describe("numbers smoke", () => {
  it("formats like Python", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(0, { mob_class: dummyFactory });
    expect(d._get_num_string(0)).toBe("0.00");
    expect(d._get_num_string(2.5)).toBe("2.50");
    expect(d._get_num_string(2.675)).toBe("2.67");
    expect(d._get_num_string(1.005)).toBe("1.00");
    expect(d._get_num_string(0.125)).toBe("0.12");
    expect(d._get_num_string(99.999)).toBe("100.00");
    expect(d._get_num_string(1234567.891)).toBe("1,234,567.89");
    expect(d._get_num_string(-1234567.891)).toBe("-1,234,567.89");
    expect(d._get_num_string(-0.001)).toBe("0.00");
    expect(d._get_formatter()).toBe("{:,.2f}");
    expect(d._get_complex_formatter()).toBe("{0.real:,.2f}{0.imag:+, .2f}i".replace(" ", ""));
  });

  it("include_sign and zero normalization", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(0, { mob_class: dummyFactory, include_sign: true });
    expect(d._get_num_string(-0.001)).toBe("+0.00");
    expect(d._get_num_string(2.5)).toBe("+2.50");
  });

  it("set/get/increment_value", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(5, { mob_class: dummyFactory });
    expect(d.get_value()).toBe(5);
    d.increment_value(2);
    expect(d.get_value()).toBe(7);
    d.increment_value();
    expect(d.get_value()).toBe(8);
    d.set_value(-0.001);
    // NOTE: mirrors Python: rounded(-0.001, 2) == 0, so the minus is stripped.
    expect(d._get_num_string(d.get_value() as number)).toBe("0.00");
  });

  it("Integer rounds half-even then truncates", () => {
    string_to_mob_map.clear();
    expect(new Integer(2.5, { mob_class: dummyFactory }).get_value()).toBe(2);
    expect(new Integer(3.5, { mob_class: dummyFactory }).get_value()).toBe(4);
    expect(new Integer(-2.5, { mob_class: dummyFactory }).get_value()).toBe(-2);
    expect(new Integer(2.1, { mob_class: dummyFactory }).get_value()).toBe(2);
  });

  it("default mob_class renders digits/glyphs via MathTex", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(0);
    expect(d.submobjects.length).toBeGreaterThan(0);
    expect(d._get_num_string(0)).toBe("0.00");
    // python Variable: label = MathTex("x") with MathTex("=") appended;
    // value = DecimalNumber(0.5, num_decimal_places=2).
    const v = new Variable(0.5, "x");
    expect(v.submobjects.length).toBe(2);
    expect(v.label.submobjects.length).toBe(2);
    expect(v.value.get_value()).toBe(0.5);
    const vm = new VMobject();
    const v2 = new Variable(0.5, vm);
    expect(v2.label).toBe(vm);
    expect(vm.submobjects.length).toBe(1);
    expect(v2.value.get_value()).toBe(0.5);
  });

  it("complex num strings", () => {
    string_to_mob_map.clear();
    const d = new DecimalNumber(0, { mob_class: dummyFactory });
    expect(d._get_num_string({ re: 1.5, im: -2.5 })).toBe("1.50-2.50i");
  });
});
