export * from "./constants.js";
export * from "./data_structures.js";
export * from "./utils/arrays.js";
export * from "./utils/color/index.js";
export * from "./utils/exceptions.js";
export * from "./utils/file_ops.js";
export * from "./utils/iterables.js";
export * from "./utils/simple_functions.js";
export {
  R3_to_complex,
  angle_of_vector,
  cartesian_to_spherical,
  center_of_mass,
  complex_to_R3,
  cross,
  cross2d,
  earclip_triangulation,
  get_winding_number,
  norm_squared,
  normalize_along_axis,
  perpendicular_bisector,
  quaternion_conjugate,
  quaternion_mult,
  rotation_matrix_from_quaternion,
  rotation_matrix_transpose,
  shoelace,
} from "./utils/space_ops.js";
// NOTE: ./utils/arrays.js also exports `normalize` (element-wise, no
// fall_back); the barrel keeps that one, while space_ops.normalize (with
// fall_back, mirroring manim.utils.space_ops.normalize) stays importable
// from ./utils/space_ops.js directly.
export type { Quaternion } from "./utils/space_ops.js";
export * from "./utils/images.js";
export * from "./utils/config_ops.js";
export * from "./utils/commands.js";
export * from "./utils/parameter_parsing.js";
export * from "./utils/module_ops.js";
export * from "./utils/typst_file_writing.js";
export * from "./utils/qhull.js";
export * from "./_config/logger_utils.js";
export * from "./_config/utils.js";
export * from "./camera/camera.js";
export * from "./camera/mapping_camera.js";
export * from "./camera/moving_camera.js";
export * from "./camera/multi_camera.js";
export * from "./mobject/mobject.js";
export * from "./mobject/frame.js";
export * from "./mobject/table.js";
export * from "./mobject/value_tracker.js";
export * from "./mobject/types/vectorized_mobject.js";
export * from "./mobject/types/point_cloud_mobject.js";
export * from "./mobject/types/image_mobject.js";
export * from "./mobject/graph.js";
export * from "./mobject/vector_field.js";
export * from "./mobject/text/index.js";
export * from "./mobject/text/numbers.js";
export * from "./mobject/logo.js";
export * from "./mobject/svg/svg_mobject.js";
export * from "./mobject/svg/brace.js";
export * from "./mobject/graphing/coordinate_systems.js";
export * from "./mobject/graphing/scale.js";
export * from "./mobject/graphing/functions.js";
export * from "./mobject/geometry/arc.js";
export * from "./mobject/geometry/boolean_ops.js";
export * from "./mobject/geometry/labeled.js";
export * from "./mobject/geometry/line.js";
export * from "./mobject/geometry/polygram.js";
export * from "./mobject/geometry/shape_matchers.js";
export * from "./mobject/geometry/tips.js";
export * from "./mobject/three_d/three_dimensions.js";
export * from "./mobject/three_d/three_d_utils.js";
export * from "./mobject/three_d/polyhedra.js";
export * from "./scene/scene.js";
export * from "./scene/moving_camera_scene.js";
export * from "./scene/three_d_scene.js";
export * from "./scene/vector_space_scene.js";
export * from "./scene/zoomed_scene.js";
export * from "./animation/animation.js";
export * from "./animation/changing.js";
export * from "./animation/composition.js";
export * from "./animation/creation.js";
export * from "./animation/fading.js";
export * from "./animation/movement.js";
export * from "./animation/numbers.js";
export * from "./animation/rotation.js";
export * from "./animation/transform.js";
// NOTE: both ./mobject/mobject.js (animate-builder helper) and
// ./animation/transform.js (faithful port of manim.animation.transform)
// export `_MethodAnimation`; the explicit re-export below resolves the
// star-export ambiguity in favor of the transform port (manim defines exactly
// one `_MethodAnimation`, in animation.transform). The mobject helper stays
// importable via its deep path.
export { _MethodAnimation } from "./animation/transform.js";
export * from "./animation/transform_matching_parts.js";
export * from "./animation/speedmodifier.js";
export * from "./animation/specialized.js";
export * from "./animation/updaters/mobject_update_utils.js";
export * from "./animation/updaters/update.js";
export * from "./renderer/index.js";
