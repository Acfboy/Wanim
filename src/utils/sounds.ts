import { config } from "../_config/utils.js";
import { seek_full_path_from_defaults } from "./file_ops.js";

// Still in use by add_sound() function in scene_file_writer.py
export function get_full_sound_file_path(sound_file_name: string): string {
  return seek_full_path_from_defaults(
    sound_file_name,
    config.getDir("assets_dir"),
    [".wav", ".mp3"],
  );
}
