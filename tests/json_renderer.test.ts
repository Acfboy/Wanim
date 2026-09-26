import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Camera } from "../src/camera/camera.js";
import { MovingCamera } from "../src/camera/moving_camera.js";
import { config } from "../src/_config/utils.js";
import { CapStyleType, LineJointType } from "../src/constants.js";
import { VMobject } from "../src/mobject/types/vectorized_mobject.js";
import { PMobject } from "../src/mobject/types/point_cloud_mobject.js";
import { Square } from "../src/mobject/geometry/polygram.js";
import { Circle } from "../src/mobject/geometry/arc.js";
import { Scene } from "../src/scene/scene.js";
import { MovingCameraScene } from "../src/scene/moving_camera_scene.js";
import { Create } from "../src/animation/creation.js";
import { Transform } from "../src/animation/transform.js";
import {
  _DummyFileWriter,
  JSONCairoRenderer,
  PyInt,
  jsonDumpsPython,
  serialize_frame,
  serialize_vmobject,
} from "../src/renderer/json_renderer.js";
import { apply_quality, default_camera_class, run_scene } from "../src/renderer/scene_runner.js";

function newTmpFile(name: string): string {
  const dir = mkdtempSync(join(tmpdir(), "wanim-json-"));
  return join(dir, name);
}

describe("jsonDumpsPython / PyInt", () => {
  it("writes ints without decimals and floats with a trailing .0", () => {
    expect(jsonDumpsPython(new PyInt(854))).toBe("854");
    expect(jsonDumpsPython(1)).toBe("1.0");
    expect(jsonDumpsPython(1.5)).toBe("1.5");
    expect(jsonDumpsPython(null)).toBe("null");
    expect(jsonDumpsPython(true)).toBe("true");
    expect(jsonDumpsPython("x")).toBe('"x"');
    expect(jsonDumpsPython({ a: new PyInt(2), b: [1, null] })).toBe('{"a":2,"b":[1.0,null]}');
  });
});

describe("serialize_vmobject", () => {
  it("serializes rounded points and the full style surface", () => {
    const v = new VMobject();
    v.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1.123456789, 1, 0]]);
    const rec = serialize_vmobject(v, null);

    const points = rec.p as number[][];
    expect(Array.isArray(points)).toBe(true);
    expect(points.length % 4).toBe(0);
    for (const row of points) expect(row.length).toBe(3);
    const flat = points.flat();
    expect(flat).not.toContain(1.123456789);
    expect(flat).toContain(1.123457);

    expect(rec.sw).toBe(4);
    expect(rec.bsw).toBe(0);
    expect((rec.jt as PyInt).value).toBe(LineJointType.AUTO);
    expect((rec.cs as PyInt).value).toBe(CapStyleType.AUTO);
    expect((rec.z as PyInt).value).toBe(0);
    expect(rec.bi).toBe(null);
    expect(rec.f).not.toBeNull();
    expect(rec.s).not.toBeNull();
    expect(rec.bs).not.toBeNull();
  });

  it("maps joint/cap style names and numeric values to integers", () => {
    const v = new VMobject({ joint_type: "ROUND", cap_style: "BUTT" });
    const rec = serialize_vmobject(v, null);
    expect((rec.jt as PyInt).value).toBe(LineJointType.ROUND);
    expect((rec.cs as PyInt).value).toBe(CapStyleType.BUTT);

    v.set_cap_style(CapStyleType.SQUARE);
    expect((serialize_vmobject(v).cs as PyInt).value).toBe(CapStyleType.SQUARE);
  });

  it("rejects raster background images", () => {
    const v = new VMobject();
    (v as unknown as { background_image: unknown }).background_image = 123;
    expect(() => serialize_vmobject(v, null)).toThrow(/background image/);
  });
});

describe("serialize_frame", () => {
  it("serializes camera geometry and the display list", () => {
    const camera = new Camera();
    const frame = serialize_frame(camera, [new Square()]);
    const cam = frame.cam as Record<string, unknown>;
    expect(cam.fc).toEqual([0, 0, 0]);
    expect(cam.fw).toBe(Number(camera.frame_width));
    expect(cam.fh).toBe(Number(camera.frame_height));

    const records = frame.m as Array<Record<string, unknown>>;
    expect(records.length).toBeGreaterThanOrEqual(1);
    expect(Array.isArray(records[0]!.p)).toBe(true);
    expect((records[0]!.p as number[][]).length).toBeGreaterThan(0);
  });

  it("rejects PMobjects", () => {
    const camera = new Camera();
    const pm = new PMobject();
    pm.add_points([[0, 0, 0]]);
    expect(() => serialize_frame(camera, [pm])).toThrow(/PMobjects/);
  });
});

describe("_DummyFileWriter", () => {
  it("exposes the SceneFileWriter surface Scene reaches for", () => {
    const fw = new _DummyFileWriter();
    expect(fw.sections.length).toBe(1);
    expect(fw.sections[0]!.skip_animations).toBe(false);
    fw.next_section();
    expect(fw.sections.length).toBe(2);
    expect(fw.is_already_cached("x")).toBe(false);
    expect(Array.isArray(fw.subcaptions)).toBe(true);
    fw.add_partial_movie_file("h");
    fw.begin_animation(true);
    fw.end_animation(true);
    fw.write_frame(null);
    fw.save_image(null);
    fw.abort_encode_jobs({ reraise_encoder_failures: true });
    fw.add_sound("a");
    fw.finish();
    expect(typeof fw.movie_file_path).toBe("string");
    expect(typeof fw.image_file_path).toBe("string");
    expect(typeof fw.gif_file_path).toBe("string");
  });
});

describe("JSONCairoRenderer", () => {
  it("dedups identical consecutive frames and writes the document key order", () => {
    const path = newTmpFile("dedup.mjson");
    const renderer = new JSONCairoRenderer(path);
    const scene = new Scene();
    scene.mobjects = [new Square()];

    renderer.update_frame(scene);
    renderer.add_frame(null);
    renderer.num_plays += 1;
    renderer.update_frame(scene);
    renderer.add_frame(null);

    expect(renderer.frames.length).toBe(1);
    expect((renderer.frames[0]!.repeat as PyInt).value).toBe(2);

    expect(renderer.write_output()).toBe(path);
    const raw = readFileSync(path, { encoding: "utf-8" });
    const doc = JSON.parse(raw);
    expect(Object.keys(doc)).toEqual(["meta", "frames"]);
    expect(Object.keys(doc.meta)).toEqual([
      "width",
      "height",
      "frame_rate",
      "background_color",
      "background_opacity",
      "num_frames",
      "num_plays",
    ]);
    expect(Number.isInteger(doc.meta.width)).toBe(true);
    expect(Number.isInteger(doc.meta.num_frames)).toBe(true);
    expect(doc.meta.num_frames).toBe(2);
    expect(doc.meta.num_plays).toBe(1);
    expect(raw).toContain(`"width":${config.pixel_width}`);
    expect(raw).toContain('"repeat":2');
  });

  it("clear_screen resets frames and pending state", () => {
    const renderer = new JSONCairoRenderer(newTmpFile("clear.mjson"));
    const scene = new Scene();
    scene.mobjects = [new Square()];
    renderer.update_frame(scene);
    renderer.add_frame(null);
    expect(renderer.frames.length).toBe(1);
    renderer.clear_screen();
    expect(renderer.frames.length).toBe(0);
    expect(renderer._pending_frame).toBe(null);
  });
});

describe("scene_runner", () => {
  it("selects MovingCamera for MovingCameraScene subclasses, else Camera", () => {
    class MCS extends MovingCameraScene {}
    class Plain extends Scene {}
    expect(default_camera_class(MCS)).toBe(MovingCamera);
    expect(default_camera_class(Plain)).toBe(Camera);
  });

  it("apply_quality sets the low-quality resolution", () => {
    const saved = { w: config.pixel_width, h: config.pixel_height, fr: config.frame_rate };
    try {
      expect(apply_quality("-ql")).toBe("low");
      expect(config.pixel_width).toBe(854);
      expect(config.pixel_height).toBe(480);
      expect(config.frame_rate).toBe(15);
    } finally {
      config.pixel_width = saved.w;
      config.pixel_height = saved.h;
      config.frame_rate = saved.fr;
    }
  });

  it("runs a full scene through the JSON renderer end-to-end", () => {
    const saved = { w: config.pixel_width, h: config.pixel_height, fr: config.frame_rate };
    const path = newTmpFile("SquareToCircle.mjson");
    class SquareToCircle extends Scene {
      override construct(): void {
        const square = new Square();
        const circle = new Circle();
        this.add(square);
        this.play(new Create(square));
        this.play(new Transform(square, circle));
        this.wait(1);
      }
    }
    try {
      apply_quality("-ql");
      expect(run_scene(SquareToCircle, path)).toBe(path);
      const doc = JSON.parse(readFileSync(path, { encoding: "utf-8" }));
      expect(doc.meta.width).toBe(854);
      expect(doc.meta.height).toBe(480);
      expect(doc.meta.frame_rate).toBe(15);
      expect(doc.meta.num_plays).toBe(3);
      expect(doc.frames.length).toBeGreaterThanOrEqual(1);
      expect(doc.meta.num_frames).toBe(doc.frames.reduce((sum: number, f: { repeat: number }) => sum + f.repeat, 0));
      expect(doc.frames.length).toBeLessThan(doc.meta.num_frames);

      const first = doc.frames[0];
      expect(Array.isArray(first.m)).toBe(true);
      expect(first.m.length).toBeGreaterThanOrEqual(1);
      const record = first.m[0];
      for (const key of ["p", "f", "s", "bs", "sw", "bsw", "jt", "cs", "z", "bi"]) {
        expect(key in record).toBe(true);
      }
      expect(Array.isArray(record.p)).toBe(true);
      expect(record.p.length % 4).toBe(0);
    } finally {
      config.pixel_width = saved.w;
      config.pixel_height = saved.h;
      config.frame_rate = saved.fr;
    }
  });
});
