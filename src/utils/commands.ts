export interface VideoMetadata {
  width: number;
  height: number;
  nb_frames: string;
  duration: string;
  avg_frame_rate: string;
  codec_name: string;
  pix_fmt: string;
}

export function get_video_metadata(_path_to_video: string): VideoMetadata {
  throw new Error("TODO: get_video_metadata requires native video probing (av); not ported yet");
}

export function capture(_command: string | string[], _cwd?: string | null, _command_input?: string | null): [string, string, number] {
  throw new Error("TODO: capture requires subprocess execution; not ported yet");
}
