export const ORIGIN = [0, 0, 0] as const;
export const UP = [0, 1, 0] as const;
export const DOWN = [0, -1, 0] as const;
export const RIGHT = [1, 0, 0] as const;
export const LEFT = [-1, 0, 0] as const;
export const IN = [0, 0, -1] as const;
export const OUT = [0, 0, 1] as const;
export const X_AXIS = [1, 0, 0] as const;
export const Y_AXIS = [0, 1, 0] as const;
export const Z_AXIS = [0, 0, 1] as const;
export const UL = [-1, 1, 0] as const;
export const UR = [1, 1, 0] as const;
export const DL = [-1, -1, 0] as const;
export const DR = [1, -1, 0] as const;
export const START_X = 30;
export const START_Y = 20;
export const DEFAULT_DOT_RADIUS = 0.08;
export const DEFAULT_SMALL_DOT_RADIUS = 0.04;
export const DEFAULT_DASH_LENGTH = 0.05;
export const DEFAULT_ARROW_TIP_LENGTH = 0.35;
export const SMALL_BUFF = 0.1;
export const MED_SMALL_BUFF = 0.25;
export const MED_LARGE_BUFF = 0.5;
export const LARGE_BUFF = 1;
export const DEFAULT_MOBJECT_TO_EDGE_BUFFER = MED_LARGE_BUFF;
export const DEFAULT_MOBJECT_TO_MOBJECT_BUFFER = MED_SMALL_BUFF;
export const DEFAULT_POINTWISE_FUNCTION_RUN_TIME = 3;
export const DEFAULT_WAIT_TIME = 1;
export const DEFAULT_POINT_DENSITY_2D = 25;
export const DEFAULT_POINT_DENSITY_1D = 10;
export const DEFAULT_STROKE_WIDTH = 4;
export const DEFAULT_ANIMATION_RUN_TIME = 1;
export const DEFAULT_ANIMATION_LAG_RATIO = 0;
export const DEFAULT_FONT_SIZE = 48;
export const SCALE_FACTOR_PER_FONT_POINT = 1 / 960;
export const PI = Math.PI;
export const TAU = Math.PI * 2;
export const DEGREES = TAU / 360;
export const NORMAL = "NORMAL";
export const ITALIC = "ITALIC";
export const OBLIQUE = "OBLIQUE";
export const BOLD = "BOLD";
export const SHIFT_VALUE = 65505;
export const CTRL_VALUE = 65507;

// Messages (mirrors manim.constants message strings verbatim).
export const SCENE_NOT_FOUND_MESSAGE = `
   {} is not in the script
`;
export const CHOOSE_NUMBER_MESSAGE = `
Choose number corresponding to desired scene/arguments.
(Use comma separated list for multiple entries or use "*" to select all scenes.)
Choice(s): `;
export const INVALID_NUMBER_MESSAGE = "Invalid scene numbers have been specified. Aborting.";
export const NO_SCENE_MESSAGE = `
   There are no scenes inside that module
`;

export const DEFAULT_QUALITY = "high_quality";

export interface QualityDict {
  flag: string | null;
  pixel_height: number;
  pixel_width: number;
  frame_rate: number;
}

export const QUALITIES: Record<string, QualityDict> = {
  fourk_quality: { flag: "k", pixel_height: 2160, pixel_width: 3840, frame_rate: 60 },
  production_quality: { flag: "p", pixel_height: 1440, pixel_width: 2560, frame_rate: 60 },
  high_quality: { flag: "h", pixel_height: 1080, pixel_width: 1920, frame_rate: 60 },
  medium_quality: { flag: "m", pixel_height: 720, pixel_width: 1280, frame_rate: 30 },
  low_quality: { flag: "l", pixel_height: 480, pixel_width: 854, frame_rate: 15 },
  example_quality: { flag: null, pixel_height: 480, pixel_width: 854, frame_rate: 30 },
};

export enum RendererType {
  CAIRO = "cairo",
  OPENGL = "opengl",
}

export enum LineJointType {
  AUTO = 0,
  ROUND = 1,
  BEVEL = 2,
  MITER = 3,
}

export enum CapStyleType {
  AUTO = 0,
  ROUND = 1,
  BUTT = 2,
  SQUARE = 3,
}
