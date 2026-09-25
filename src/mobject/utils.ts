import { config } from "../_config/utils.js";
import { RendererType } from "../constants.js";
import { Mobject } from "./mobject.js";

export function get_mobject_class(): typeof Mobject {
  if (config.renderer === RendererType.CAIRO) return Mobject;
  // NOTE: OpenGLMobject is not ported (only the cairo renderer is
  // supported), so any non-cairo renderer raises here.
  throw new Error("Base mobjects are not implemented for the active renderer.");
}
