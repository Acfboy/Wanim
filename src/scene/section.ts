import { get_video_metadata } from "../utils/commands.js";

export const DefaultSectionType = {
  NORMAL: "default.normal",
} as const;

export type DefaultSectionType =
  (typeof DefaultSectionType)[keyof typeof DefaultSectionType];

export class Section {
  type_: string;
  video: string | null;
  name: string;
  skip_animations: boolean;
  partial_movie_files: Array<string | null>;

  constructor(
    type_: string,
    video: string | null,
    name: string,
    skip_animations: boolean,
  ) {
    this.type_ = type_;
    this.video = video;
    this.name = name;
    this.skip_animations = skip_animations;
    this.partial_movie_files = [];
  }

  is_empty(): boolean {
    return this.partial_movie_files.length === 0;
  }

  get_clean_partial_movie_files(): string[] {
    return this.partial_movie_files.filter(
      (el): el is string => el !== null,
    );
  }

  get_dict(sections_dir: string): Record<string, unknown> {
    if (this.video === null) {
      throw new Error(
        `Section '${this.name}' cannot be exported as dict, it does not have a video path assigned to it`,
      );
    }
    // NOTE (interpreter scope): video probing / media export is out of scope;
    // get_video_metadata is a TODO stub and throws until it is ported.
    const video_metadata = get_video_metadata(`${sections_dir}/${this.video}`);
    return {
      name: this.name,
      type: this.type_,
      video: this.video,
      ...video_metadata,
    };
  }

  toRepr(): string {
    return `<Section '${this.name}' stored in '${this.video}'>`;
  }
}
