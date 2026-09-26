// Port of manim-wasm's PangoJS `markup-parser.js` (parse Pango markup into
// plain text plus a list of styled attribute ranges). Kept as close to the
// reference implementation as possible; the only deliberate changes are the
// TypeScript typing and the removal of dead self-closing-tag branches.

const CSS_COLOR_NAMES: Record<string, string> = {
  aliceblue: "#f0f8ff", antiquewhite: "#faebd7", aqua: "#00ffff", aquamarine: "#7fffd4",
  azure: "#f0ffff", beige: "#f5f5dc", bisque: "#ffe4c4", black: "#000000",
  blanchedalmond: "#ffebcd", blue: "#0000ff", blueviolet: "#8a2be2", brown: "#a52a2a",
  burlywood: "#deb887", cadetblue: "#5f9ea0", chartreuse: "#7fff00", chocolate: "#d2691e",
  coral: "#ff7f50", cornflowerblue: "#6495ed", cornsilk: "#fff8dc", crimson: "#dc143c",
  cyan: "#00ffff", darkblue: "#00008b", darkcyan: "#008b8b", darkgoldenrod: "#b8860b",
  darkgray: "#a9a9a9", darkgreen: "#006400", darkkhaki: "#bdb76b", darkmagenta: "#8b008b",
  darkolivegreen: "#556b2f", darkorange: "#ff8c00", darkorchid: "#9932cc", darkred: "#8b0000",
  darksalmon: "#e9967a", darkseagreen: "#8fbc8f", darkslateblue: "#483d8b", darkslategray: "#2f4f4f",
  darkturquoise: "#00ced1", darkviolet: "#9400d3", deeppink: "#ff1493", deepskyblue: "#00bfff",
  dimgray: "#696969", dodgerblue: "#1e90ff", firebrick: "#b22222", floralwhite: "#fffaf0",
  forestgreen: "#228b22", fuchsia: "#ff00ff", gainsboro: "#dcdcdc", ghostwhite: "#f8f8ff",
  gold: "#ffd700", goldenrod: "#daa520", gray: "#808080", green: "#008000",
  greenyellow: "#adff2f", honeydew: "#f0fff0", hotpink: "#ff69b4", indianred: "#cd5c5c",
  indigo: "#4b0082", ivory: "#fffff0", khaki: "#f0e68c", lavender: "#e6e6fa",
  lavenderblush: "#fff0f5", lawngreen: "#7cfc00", lemonchiffon: "#fffacd", lightblue: "#add8e6",
  lightcoral: "#f08080", lightcyan: "#e0ffff", lightgoldenrodyellow: "#fafad2", lightgray: "#d3d3d3",
  lightgreen: "#90ee90", lightpink: "#ffb6c1", lightsalmon: "#ffa07a", lightseagreen: "#20b2aa",
  lightskyblue: "#87cefa", lightslategray: "#778899", lightsteelblue: "#b0c4de", lightyellow: "#ffffe0",
  lime: "#00ff00", limegreen: "#32cd32", linen: "#faf0e6", magenta: "#ff00ff",
  maroon: "#800000", mediumaquamarine: "#66cdaa", mediumblue: "#0000cd", mediumorchid: "#ba55d3",
  mediumpurple: "#9370db", mediumseagreen: "#3cb371", mediumslateblue: "#7b68ee", mediumspringgreen: "#00fa9a",
  mediumturquoise: "#48d1cc", mediumvioletred: "#c71585", midnightblue: "#191970", mintcream: "#f5fffa",
  mistyrose: "#ffe4e1", moccasin: "#ffe4b5", navajowhite: "#ffdead", navy: "#000080",
  oldlace: "#fdf5e6", olive: "#808000", olivedrab: "#6b8e23", orange: "#ffa500", orangered: "#ff4500",
  orchid: "#da70d6", palegoldenrod: "#eee8aa", palegreen: "#98fb98", paleturquoise: "#afeeee",
  palevioletred: "#db7093", papayawhip: "#ffefd5", peachpuff: "#ffdab9", peru: "#cd853f",
  pink: "#ffc0cb", plum: "#dda0dd", powderblue: "#b0e0e6", purple: "#800080", rebeccapurple: "#663399",
  red: "#ff0000", rosybrown: "#bc8f8f", royalblue: "#4169e1", saddlebrown: "#8b4513", salmon: "#fa8072",
  sandybrown: "#f4a460", seagreen: "#2e8b57", seashell: "#fff5ee", sienna: "#a0522d", silver: "#c0c0c0",
  skyblue: "#87ceeb", slateblue: "#6a5acd", slategray: "#708090", snow: "#fffafa", springgreen: "#00ff7f",
  steelblue: "#4682b4", tan: "#d2b48c", teal: "#008080", thistle: "#d8bfd8", tomato: "#ff6347",
  turquoise: "#40e0d0", violet: "#ee82ee", wheat: "#f5deb3", white: "#ffffff", whitesmoke: "#f5f5f5",
  yellow: "#ffff00", yellowgreen: "#9acd32",
};

// Supported tags and their default attributes.
const TAG_DEFAULTS: Record<string, Record<string, unknown>> = {
  b: { weight: "bold" },
  i: { style: "italic" },
  u: { underline: "single" },
  s: { strikethrough: "true" },
  tt: { font_family: "monospace" },
  big: { scale: 1.2 },
  small: { scale: 0.83 },
  sub: { baseline_shift: -0.5, scale: 0.7 },
  sup: { baseline_shift: 0.5, scale: 0.7 },
};

export const UNDERLINE_MAP: Record<string, string> = {
  none: "none",
  single: "underline",
  double: "underline double",
  low: "underline",
  error: "underline wavy",
};

const STYLE_MAP: Record<string, string> = {
  normal: "normal",
  italic: "italic",
  oblique: "oblique",
};

export const WEIGHT_MAP: Record<string, number> = {
  ultralight: 100,
  light: 300,
  normal: 400,
  bold: 700,
  ultrabold: 800,
  heavy: 900,
};

export const SIZE_MAP: Record<string, number> = {
  "xx-small": 0.64,
  "x-small": 0.8,
  small: 0.9,
  medium: 1.0,
  large: 1.2,
  "x-large": 1.44,
  "xx-large": 1.728,
  smaller: 0.83,
  larger: 1.2,
};

export interface MarkupAttribute {
  start?: number;
  end?: number;
  foreground?: string;
  background?: string;
  font_family?: string;
  font_size?: number;
  font_style?: string;
  font_weight?: number;
  underline?: string;
  underline_color?: string;
  overline?: string;
  overline_color?: string;
  strikethrough?: boolean;
  strikethrough_color?: string;
  rise?: number;
  baseline_shift?: number;
  letter_spacing?: number;
  lang?: string;
  scale?: number;
  _depth?: number;
}

export interface ParsedMarkup {
  text: string;
  attributes: MarkupAttribute[];
}

interface OpenTag {
  name: string;
  attrs: MarkupAttribute;
  startIndex: number;
  depth: number;
}

/** Parse Pango markup text into plain text plus the attribute ranges. */
export function parseMarkup(markup: string): ParsedMarkup {
  const attributes: MarkupAttribute[] = [];
  const tagStack: OpenTag[] = [];
  let plainText = "";
  let i = 0;
  let currentIndex = 0;
  let skipDepth = 0;

  const entityMap: Record<string, string> = {
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
    "&quot;": '"',
    "&apos;": "'",
  };

  function processEntity(str: string): string {
    const known = entityMap[str];
    if (known) return known;
    const numMatch = str.match(/^&#(\d+);$/);
    if (numMatch) return String.fromCharCode(parseInt(numMatch[1] as string, 10));
    const hexMatch = str.match(/^&#x([0-9a-fA-F]+);$/);
    if (hexMatch) return String.fromCharCode(parseInt(hexMatch[1] as string, 16));
    return str;
  }

  while (i < markup.length) {
    if (markup[i] === "<") {
      const tagEnd = markup.indexOf(">", i);
      if (tagEnd === -1) break;

      const tagContent = markup.slice(i + 1, tagEnd);

      if (tagContent.startsWith("!--") || tagContent.startsWith("![CDATA[")) {
        i = tagEnd + 1;
        continue;
      }

      const isClosingTag = tagContent.startsWith("/");
      const tagName = isClosingTag
        ? (tagContent.slice(1).trim().toLowerCase().split(/\s/)[0] ?? "")
        : (tagContent.trim().toLowerCase().split(/\s/)[0] ?? "");

      if (isClosingTag) {
        const openTag = tagStack.pop();
        if (openTag && openTag.name === tagName) {
          if (skipDepth > 0) {
            skipDepth -= 1;
          } else {
            const attr: MarkupAttribute = { ...openTag.attrs };
            const start = Math.max(0, openTag.startIndex);
            const end = Math.min(plainText.length, currentIndex);
            if (end > start) {
              attr.start = start;
              attr.end = end;
              attr._depth = openTag.depth;
              attributes.push(attr);
            }
          }
        }
      } else if (tagContent.endsWith("/")) {
        // Self-closing tag: applies to nothing, ignored.
      } else {
        let tagAttrs: MarkupAttribute = {};
        if (tagName === "span") {
          tagAttrs = parseSpanAttributes(tagContent);
        } else if (TAG_DEFAULTS[tagName]) {
          tagAttrs = { ...TAG_DEFAULTS[tagName] } as MarkupAttribute;
          Object.assign(tagAttrs, parseSpanAttributes(tagContent));
        }
        tagStack.push({
          name: tagName,
          attrs: tagAttrs,
          startIndex: currentIndex,
          depth: tagStack.length,
        });
      }

      i = tagEnd + 1;
    } else if (markup[i] === "&") {
      const entityEnd = markup.indexOf(";", i);
      if (entityEnd !== -1) {
        const entity = markup.slice(i, entityEnd + 1);
        plainText += processEntity(entity);
        currentIndex++;
        i = entityEnd + 1;
      } else {
        plainText += markup[i];
        currentIndex++;
        i++;
      }
    } else {
      plainText += markup[i];
      currentIndex++;
      i++;
    }
  }

  while (tagStack.length > 0) {
    const openTag = tagStack.pop() as OpenTag;
    const attr: MarkupAttribute = { ...openTag.attrs };
    const start = Math.max(0, openTag.startIndex);
    const end = plainText.length;
    if (end > start) {
      attr.start = start;
      attr.end = end;
      attr._depth = openTag.depth;
      attributes.push(attr);
    }
  }

  return { text: plainText, attributes };
}

function parseSpanAttributes(content: string): MarkupAttribute {
  const attrs: MarkupAttribute = {};
  if (content === "span") return attrs;

  const attrRegex = /(\w+(?:_[\w]+)?)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/g;
  let match: RegExpExecArray | null;
  while ((match = attrRegex.exec(content)) !== null) {
    const attrName = match[1] as string;
    const value = (match[2] ?? match[3] ?? match[4] ?? "") as string;

    switch (attrName) {
      case "foreground":
      case "fgcolor":
      case "color":
        attrs.foreground = parseColor(value);
        break;
      case "background":
      case "bgcolor":
        attrs.background = parseColor(value);
        break;
      case "font_family":
      case "face":
        attrs.font_family = value.replace(/^["']|["']$/g, "");
        break;
      case "font_size":
      case "size":
        attrs.font_size = parseSize(value);
        break;
      case "font_style":
      case "style":
        attrs.font_style = STYLE_MAP[value] ?? value;
        break;
      case "font_weight":
      case "weight":
        attrs.font_weight = WEIGHT_MAP[value] ?? (parseInt(value, 10) || 400);
        break;
      case "underline":
        attrs.underline = value;
        break;
      case "underline_color":
        attrs.underline_color = parseColor(value);
        break;
      case "overline":
        attrs.overline = value;
        break;
      case "overline_color":
        attrs.overline_color = parseColor(value);
        break;
      case "strikethrough":
        attrs.strikethrough = value === "true";
        break;
      case "strikethrough_color":
        attrs.strikethrough_color = parseColor(value);
        break;
      case "rise":
        attrs.rise = parseSize(value);
        break;
      case "baseline_shift":
        attrs.baseline_shift = parseBaselineShift(value);
        break;
      case "letter_spacing":
        attrs.letter_spacing = parseSize(value);
        break;
      case "lang":
      case "language":
        attrs.lang = value;
        break;
      case "scale":
        attrs.scale = parseFloat(value);
        break;
      default:
        break;
    }
  }

  return attrs;
}

/** Parse a color value (hex, rgb, or named color). */
function parseColor(value: string): string {
  if (!value) return "";
  const v = value.trim();

  if (v.startsWith("#")) {
    const hex = v.slice(1);
    if (hex.length === 3) {
      return `#${hex[0]}${hex[0]}${hex[1]}${hex[1]}${hex[2]}${hex[2]}`;
    }
    if (hex.length === 6) return `#${hex.toLowerCase()}`;
    if (hex.length === 8) return `#${hex.slice(0, 6)}`;
  }

  const rgbMatch = v.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (rgbMatch) {
    const r = parseInt(rgbMatch[1] as string, 10);
    const g = parseInt(rgbMatch[2] as string, 10);
    const b = parseInt(rgbMatch[3] as string, 10);
    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }

  const lower = v.toLowerCase();
  return CSS_COLOR_NAMES[lower] ?? v;
}

function parseSize(value: string): number {
  if (!value) return 0;
  const v = value.trim();

  if (SIZE_MAP[v] !== undefined) return SIZE_MAP[v] as number;

  const emMatch = v.match(/^([\d.]+)em$/);
  if (emMatch) return parseFloat(emMatch[1] as string);

  const pctMatch = v.match(/^(\d+(?:\.\d+)?)%$/);
  if (pctMatch) return parseFloat(pctMatch[1] as string) / 100;

  const ptMatch = v.match(/^(\d+(?:\.\d+)?)pt$/);
  if (ptMatch) return parseFloat(ptMatch[1] as string);

  const pxMatch = v.match(/^(\d+(?:\.\d+)?)px$/);
  if (pxMatch) return parseFloat(pxMatch[1] as string);

  const num = parseFloat(v);
  if (!Number.isNaN(num)) {
    if (num > 100) return num / 1024;
    return num;
  }
  return 0;
}

function parseBaselineShift(value: string): number {
  const v = value.trim();
  if (v === "superscript") return 0.5;
  if (v === "subscript") return -0.5;
  if (v === "baseline") return 0;
  return parseSize(v) || 0;
}

export interface StyledRun {
  text: string;
  start: number;
  end: number;
  styles: Record<string, unknown>;
}

/** Build styled runs from text and attributes (merges overlapping ranges). */
export function buildStyledRuns(text: string, attributes: MarkupAttribute[]): StyledRun[] {
  if (!attributes || attributes.length === 0) {
    return [{ text, start: 0, end: text.length, styles: {} }];
  }

  const sortedAttrs = [...attributes].sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  const changes: Array<{ index: number; key: string; value: unknown; type: "start" | "end"; depth: number }> = [];
  for (const attr of sortedAttrs) {
    const start = attr.start ?? 0;
    const end = attr.end ?? text.length;
    const depth = attr._depth ?? 0;
    for (const [key, value] of Object.entries(attr)) {
      if (key === "start" || key === "end" || key === "_depth") continue;
      changes.push({ index: start, key, value, type: "start", depth });
      changes.push({ index: end, key, value, type: "end", depth });
    }
  }

  changes.sort((a, b) => {
    if (a.index !== b.index) return a.index - b.index;
    if (a.type !== b.type) return a.type === "end" ? -1 : 1;
    if (a.type === "start") return (a.depth || 0) - (b.depth || 0);
    return 0;
  });

  const runs: StyledRun[] = [];
  let currentStyles: Record<string, unknown> = {};
  let lastIndex = 0;

  for (const change of changes) {
    if (change.index > lastIndex) {
      runs.push({ text: text.slice(lastIndex, change.index), start: lastIndex, end: change.index, styles: { ...currentStyles } });
      lastIndex = change.index;
    }
    if (change.type === "start") {
      currentStyles[change.key] = change.value;
    } else {
      delete currentStyles[change.key];
    }
  }

  if (lastIndex < text.length) {
    runs.push({ text: text.slice(lastIndex), start: lastIndex, end: text.length, styles: { ...currentStyles } });
  }

  return runs;
}
