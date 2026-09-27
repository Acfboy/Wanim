import { sha256Hex } from "./hash.js";

export const TYPST_COMPILATION_FONT_SIZE = 10;

export const TYPST_TEMPLATE = `#set page(width: auto, height: auto, margin: 0pt, fill: none)
#set text(size: {text_size}pt)
{preamble}
{body}
`;

export function _typst_hash(content: string): string {
  return sha256Hex(content).slice(0, 16);
}

export function typst_to_svg_file(
  _typst_code: string,
  _preamble = "",
  _text_size: number = TYPST_COMPILATION_FONT_SIZE,
  _font_paths?: Array<string> | null,
): string {
  throw new Error(
    "TODO: Typst rendering is not ported yet (requires the external `typst` compiler binary)",
  );
}
