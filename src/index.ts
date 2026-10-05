// NOTE: side-effect import must stay first: it sizes the numpy-ts WASM heap
// before any wanim module performs an array operation (numpy-ts only accepts
// configureWasm() before the heap is initialised).
import "./utils/numpy_config.js";
export { configure_numpy_wasm_heap } from "./utils/numpy_config.js";
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
  angle_axis_from_quaternion,
  angle_between_vectors,
  angle_of_vector,
  cartesian_to_spherical,
  center_of_mass,
  compass_directions,
  complex_func_to_R3_func,
  complex_to_R3,
  cross,
  cross2d,
  earclip_triangulation,
  find_intersection,
  get_unit_normal,
  get_winding_number,
  line_intersection,
  midpoint,
  norm_squared,
  normalize_along_axis,
  perpendicular_bisector,
  quaternion_conjugate,
  quaternion_from_angle_axis,
  quaternion_mult,
  regular_vertices,
  rotate_vector,
  rotation_about_z,
  rotation_matrix,
  rotation_matrix_from_quaternion,
  rotation_matrix_transpose,
  shoelace,
  shoelace_direction,
  spherical_to_cartesian,
  thick_diagonal,
  z_to_vector,
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
export * from "./utils/bezier.js";
export * from "./utils/paths.js";
export * from "./utils/rate_functions.js";
// NOTE: both ./animation/animation.js and ./utils/rate_functions.js export a
// type named `RateFunction` (with different signatures); the explicit type
// re-export below resolves the star-export ambiguity in favor of the
// animation signature (manim.animation.animation.RateFunction).
export type { RateFunction } from "./animation/animation.js";
export * from "./utils/caching.js";
export * from "./utils/family.js";
export * from "./utils/deprecation.js";
export * from "./utils/hash.js";
export * from "./utils/debug.js";
export * from "./utils/sounds.js";
export * from "./utils/tex.js";
export * from "./utils/tex_file_writing.js";
// NOTE: ./utils/polylabel.js also exports an internal `Polygon` class that
// would collide with ./mobject/geometry/polygram.js `Polygon`; only the public
// helpers are re-exported here.
export { Cell, polylabel } from "./utils/polylabel.js";
export * from "./_config/logger_utils.js";
export * from "./_config/utils.js";
export * from "./camera/camera.js";
export * from "./camera/three_d_camera.js";
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
export * from "./mobject/matrix.js";
export * from "./mobject/vector_field.js";
export * from "./mobject/utils.js";
export * from "./mobject/text/index.js";
export * from "./mobject/text/numbers.js";
export * from "./mobject/logo.js";
export * from "./mobject/svg/svg_mobject.js";
export * from "./mobject/svg/brace.js";
export * from "./mobject/graphing/coordinate_systems.js";
export * from "./mobject/graphing/number_line.js";
export * from "./mobject/graphing/probability.js";
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
export * from "./scene/section.js";
export * from "./scene/interactive_scene.js";
export * from "./interactive/controls.js";
export * from "./interactive/host.js";
export * from "./scene/moving_camera_scene.js";
export * from "./scene/three_d_scene.js";
export * from "./scene/vector_space_scene.js";
export * from "./scene/zoomed_scene.js";
export * from "./animation/animation.js";
export * from "./animation/changing.js";
export * from "./animation/composition.js";
export * from "./animation/creation.js";
export * from "./animation/fading.js";
export * from "./animation/growing.js";
export * from "./animation/indication.js";
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
export * from "./renderer/scene_runner.js";
export * from "./renderer/index.js";
export * from "./browser.js";
