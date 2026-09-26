import { config } from "../_config/utils.js";
import { RendererType } from "../constants.js";
import { Mobject } from "./mobject.js";
import { VMobject } from "./types/vectorized_mobject.js";
import { PMobject } from "./types/point_cloud_mobject.js";

export function get_mobject_class(): typeof Mobject {
  if (config.renderer === RendererType.CAIRO) return Mobject;
  // NOTE: OpenGLMobject is not ported (only the cairo renderer is
  // supported), so any non-cairo renderer raises here (mirrors Python's
  // `raise NotImplementedError(...)`).
  const error = new Error("Base mobjects are not implemented for the active renderer.");
  error.name = "NotImplementedError";
  throw error;
}

export function get_vectorized_mobject_class(): typeof VMobject {
  if (config.renderer === RendererType.CAIRO) return VMobject;
  // NOTE: OpenGLVMobject is not ported (only the cairo renderer is
  // supported), so any non-cairo renderer raises here (mirrors Python's
  // `raise NotImplementedError(...)`).
  const error = new Error("Vectorized mobjects are not implemented for the active renderer.");
  error.name = "NotImplementedError";
  throw error;
}

export function get_point_mobject_class(): typeof PMobject {
  if (config.renderer === RendererType.CAIRO) return PMobject;
  // NOTE: OpenGLPMobject is not ported (only the cairo renderer is
  // supported), so any non-cairo renderer raises here (mirrors Python's
  // `raise NotImplementedError(...)`).
  const error = new Error("Point cloud mobjects are not implemented for the active renderer.");
  error.name = "NotImplementedError";
  throw error;
}
