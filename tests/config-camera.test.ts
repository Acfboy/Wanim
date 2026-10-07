import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { parse_theme, type SectionProxyLike } from "../src/_config/logger_utils.js";
import { ManimConfig, _determine_quality, config_file_paths } from "../src/_config/utils.js";
import { Camera } from "../src/camera/camera.js";
import { MethodWithArgs } from "../src/data_structures.js";
import { MultiAnimationOverrideException } from "../src/utils/exceptions.js";
import { get_template_names, get_template_path } from "../src/utils/file_ops.js";
import { remove_list_redundancies } from "../src/utils/iterables.js";
import { clip } from "../src/utils/simple_functions.js";
import { ManimColor, RED } from "../src/utils/color/core.js";

function section(values: Record<string, string>): SectionProxyLike {
  return {
    ...values,
    get(key: string): string {
      if (!Object.prototype.hasOwnProperty.call(values, key)) throw new Error(`missing ${key}`);
      return values[key];
    },
    *[Symbol.iterator](): Iterator<[string, string]> {
      yield* Object.entries(values);
    },
  };
}

describe("Camera background properties", () => {
  it("returns the constructor color and opacity without changing the color alpha", () => {
    const camera = new Camera({
      pixel_width: 2,
      pixel_height: 1,
      frame_width: 4,
      frame_height: 2,
      background_color: new ManimColor("#123456", 0.2),
      background_opacity: 0.35,
    });

    expect(camera.background_color.toRgba().tolist()).toEqual([
      0.07058823529411765,
      0.20392156862745098,
      0.33725490196078434,
      0.2,
    ]);
    expect(camera.background_opacity).toBe(0.35);
  });

  it("updates the rendered background when either property is set", () => {
    const camera = new Camera({
      pixel_width: 2,
      pixel_height: 1,
      frame_width: 4,
      frame_height: 2,
      background_color: new ManimColor("#123456", 0.2),
      background_opacity: 0.35,
    });

    camera.background_color = RED;
    camera.background_opacity = 0.75;

    expect(camera.background_color).toBe(RED);
    expect(camera.background_opacity).toBe(0.75);
    expect(camera.background.shape).toEqual([1, 2, 4]);
    expect(camera.background.tolist()).toEqual([
      [
        [252, 98, 85, 191],
        [252, 98, 85, 191],
      ],
    ]);
  });
});

describe("parse_theme", () => {
  it("normalizes option names and returns only custom styles", () => {
    const theme = parse_theme(section({
      logging_keyword: "bold yellow",
      log_width: "100",
      log_height: "40",
      log_timestamps: "True",
      repr_number: "green",
    }));

    expect(theme).toEqual({
      "logging.keyword": "bold yellow",
      "repr.number": "green",
    });
  });

  it("supports automatic dimensions and empty style values", () => {
    const theme = parse_theme(section({
      log_level: "",
      log_message: "",
      log_width: "-1",
      log_height: "-1",
      log_timestamps: "True",
    }));

    expect(theme).toEqual({
      "log.level": "",
      "log.message": "",
    });
  });

  it("returns null for an invalid Rich style", () => {
    expect(parse_theme(section({
      logging_keyword: "not a style",
      log_width: "-1",
      log_height: "-1",
      log_timestamps: "False",
    }))).toBeNull();
  });

  it("rejects missing or non-integer dimensions", () => {
    expect(() => parse_theme(section({ log_width: "40" }))).toThrow();
    expect(() => parse_theme(section({ log_width: "wide", log_height: "40" }))).toThrow();
    expect(() => parse_theme(section({ log_width: "80", log_height: "tall" }))).toThrow();
  });
});

describe("config_file_paths", () => {
  it("returns library, user, and folder paths in precedence order", () => {
    const paths = config_file_paths();
    const libraryCfg = fileURLToPath(new URL("../src/_config/default.cfg", import.meta.url));

    expect(paths).toHaveLength(3);
    expect(paths[0]).toBe(libraryCfg);
    expect(paths[1]).toBe(`${process.env.HOME}/.config/manim/manim.cfg`);
    expect(paths[2]).toBe("manim.cfg");
  });
});

describe("ManimConfig.__getitem__", () => {
  it("delegates known keys to their property values", () => {
    const manimConfig = new ManimConfig();
    manimConfig.background_opacity = 0.35;
    manimConfig.background_color = "#123456";

    expect(manimConfig.__getitem__("background_opacity")).toBe(0.35);
    expect(manimConfig.__getitem__("background_color").toHex()).toBe("#123456");
  });

  it("starts a fresh Python ManimConfig with null option values", () => {
    const manimConfig = new ManimConfig();

    expect(manimConfig.__getitem__("background_opacity")).toBeNull();
    expect(manimConfig.__getitem__("background_color")).toBeNull();
    expect(manimConfig.__getitem__("frame_height")).toBeNull();
    expect(manimConfig.__getitem__("quality")).toBeNull();
  });

  it("raises the Python AttributeError for an unknown key", () => {
    expect(() => new ManimConfig().__getitem__("not_a_key")).toThrow(
      "'ManimConfig' object has no attribute 'not_a_key'",
    );

    try {
      new ManimConfig().__getitem__("not_a_key");
    } catch (error) {
      expect(error).toMatchObject({ name: "AttributeError" });
    }
  });
});

describe("_determine_quality", () => {
  it.each([
    ["h", "high_quality"],
    ["m", "medium_quality"],
    ["l", "low_quality"],
    ["k", "fourk_quality"],
    ["p", "production_quality"],
  ])("expands the %s flag", (flag, expected) => {
    expect(_determine_quality(flag)).toBe(expected);
  });

  it.each(["example_quality", "custom", ""])("passes %j through unchanged", (quality) => {
    expect(_determine_quality(quality)).toBe(quality);
  });

  it("passes null through unchanged", () => {
    expect(_determine_quality(null)).toBeNull();
  });
});

describe("template discovery", () => {
  it("returns every template stem", () => {
    expect(get_template_names()).toEqual(expect.arrayContaining(["Axes", "Default", "MovingCamera"]));
    expect(get_template_names()).toHaveLength(3);
  });

  it("returns the absolute templates directory", () => {
    const path = get_template_path();
    const expected = fileURLToPath(new URL("../src/templates", import.meta.url));

    expect(path).toBe(expected);
    expect(path.startsWith("/")).toBe(true);
  });
});

describe("clip", () => {
  it("clips numbers to the inclusive interval", () => {
    expect(clip(10, 11, 20)).toBe(11);
    expect(clip(15, 11, 20)).toBe(15);
    expect(clip(21, 11, 20)).toBe(20);
    expect(clip(11, 11, 20)).toBe(11);
    expect(clip(20, 11, 20)).toBe(20);
  });

  it("supports other Python-comparable values", () => {
    expect(clip("a", "h", "k")).toBe("h");
  });
});

describe("remove_list_redundancies", () => {
  it("keeps the last occurrence of each hashable value", () => {
    expect(remove_list_redundancies([1, 2, 1, 3, 2])).toEqual([1, 3, 2]);
    expect(remove_list_redundancies(["b", "a", "b"])).toEqual(["a", "b"]);
  });

  it("handles empty and already-deduplicated lists", () => {
    expect(remove_list_redundancies([])).toEqual([]);
    expect(remove_list_redundancies([3, 1, 2])).toEqual([3, 1, 2]);
  });
});

describe("MethodWithArgs", () => {
  it("stores the callable, positional arguments, and keyword arguments", () => {
    const method = (value: string): string => value;
    const args = [1, "two"];
    const kwargs = { enabled: true, label: "x" };
    const wrapped = new MethodWithArgs(method, args, kwargs);

    expect(wrapped.method).toBe(method);
    expect(wrapped.args).toBe(args);
    expect(wrapped.kwargs).toBe(kwargs);
    expect(wrapped.method(...wrapped.args)).toBe(1);
  });

  it("requires all three Python dataclass fields", () => {
    expect(() => new MethodWithArgs((value: string) => value)).toThrow(TypeError);
  });
});

describe("MultiAnimationOverrideException", () => {
  it("is an Exception-compatible error with the Python default message", () => {
    const error = new MultiAnimationOverrideException();

    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("MultiAnimationOverrideException");
    expect(error.message).toBe("");
  });

  it("preserves an explicit message", () => {
    expect(new MultiAnimationOverrideException("boom").message).toBe("boom");
  });
});

describe("additional Python edge cases", () => {
  it("wraps camera alpha overflow and resets the pixel buffer", () => {
    const camera = new Camera({ pixel_width: 1, pixel_height: 1, frame_width: 1, frame_height: 1, background_color: "#436f80", background_opacity: 1.5 });
    expect(camera.background.tolist()).toEqual([[[67, 111, 128, 126]]]);
    expect(camera.pixel_array.tolist()).toEqual([[[67, 111, 128, 126]]]);
  });

  it("matches Python hash-equality deduplication for booleans and NaN", () => {
    expect(remove_list_redundancies([1, true, false])).toEqual([true, false]);
    const first = Number.NaN;
    const second = Number.NaN;
    const result = remove_list_redundancies([first, second]);
    expect(result).toHaveLength(2);
    expect(Number.isNaN(result[0])).toBe(true);
    expect(Number.isNaN(result[1])).toBe(true);
  });

  it("rejects mixed comparable types like Python", () => {
    expect(() => clip(null as unknown as number, 0, 1)).toThrow(TypeError);
    expect(() => clip("10" as unknown as number, 0, 20)).toThrow(TypeError);
  });

  it("exposes computed config keys", () => {
    const manimConfig = new ManimConfig();
    manimConfig.pixel_width = 1280;
    manimConfig.pixel_height = 720;
    expect(manimConfig.__getitem__("frame_size")).toEqual([1280, 720]);
    expect(manimConfig.__getitem__("aspect_ratio")).toBe(1280 / 720);
  });
});
