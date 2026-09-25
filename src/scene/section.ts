export const DefaultSectionType = {
  NORMAL: "default.normal",
} as const;

export type DefaultSectionType =
  (typeof DefaultSectionType)[keyof typeof DefaultSectionType];
