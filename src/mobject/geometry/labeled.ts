import { VGroup } from "../types/vectorized_mobject.js";

/**
 * A label consisting of text surrounded by a frame.
 *
 * NOTE: text/tex rendering (MathTex, Text, Typst) is out of scope for the
 * entire port, and this class exists to display a text label — so the
 * constructor is a clearly-marked stub until text support lands. The
 * signature mirrors the Python original.
 */
export class Label extends VGroup {
  rendered_label!: unknown;
  background_rect!: unknown;
  frame!: unknown;

  constructor(
    label: string | unknown,
    label_config: Record<string, unknown> | null = null,
    box_config: Record<string, unknown> | null = null,
    frame_config: Record<string, unknown> | null = null,
    options: Record<string, unknown> = {},
  ) {
    super();
    void label;
    void label_config;
    void box_config;
    void frame_config;
    void options;
    throw new Error("TODO: text/tex not ported yet");
  }
}
