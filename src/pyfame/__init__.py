__version__ = "0.8.0"
__author__ = "Gavin Bosman"

from pyfame.analysis import get_facial_color_means, get_optical_flow
from pyfame.io import get_directory_walk, create_output_directory, map_directory_structure, get_video_capture, get_video_writer
from pyfame.io.conversion import apply_conversion_image_to_video, apply_conversion_video_to_mp4
from pyfame.logging import setup_logging
from pyfame.manipulation.color import layer_color, layer_color_brightness, layer_color_saturation
from pyfame.manipulation.mask import layer_mask
from pyfame.manipulation.occlusion import layer_occlusion_path, layer_occlusion_bar, layer_occlusion_blur, layer_occlusion_noise
from pyfame.manipulation.spatial import layer_spatial_grid_shuffle, layer_spatial_landmark_shuffle
from pyfame.manipulation.stylize import layer_stylize_point_light, apply_image_overlay
from pyfame.manipulation.temporal import generate_shuffled_block_array, apply_frame_shuffle
from pyfame.mesh import get_mask_from_path, get_mesh, get_mesh_coordinates, get_mesh_coordinates_from_path
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.timing import timing_sigmoid, timing_constant, timing_gaussian, timing_linear
from pyfame.layer import Layer, LayerPipeline, apply_layers
from pyfame.util import get_variable_name, compute_rot_angle, compute_line_intersection, FileReadError, FileWriteError, FaceNotFoundError, ImageShapeError, UnrecognizedExtensionError
from pyfame.util.util_constants import *
from pyfame.util.util_display_parameter_options import *

__all__ = [
    "get_facial_color_means", "get_optical_flow",
    "get_directory_walk", "create_output_directory", "map_directory_structure", "get_video_capture", "get_video_writer",
    "apply_conversion_image_to_video", "apply_conversion_video_to_mp4",
    "setup_logging",
    "layer_color", "layer_color_brightness", "layer_color_saturation",
    "layer_mask",
    "layer_occlusion_path", "layer_occlusion_bar", "layer_occlusion_blur", "layer_occlusion_noise",
    "layer_spatial_grid_shuffle", "layer_spatial_landmark_shuffle", 
    "layer_stylize_point_light", "apply_image_overlay",
    "generate_shuffled_block_array", "apply_frame_shuffle",
    "get_mask_from_path", "get_mesh", "get_mesh_coordinates", "get_mesh_coordinates_from_path",
    "timing_constant", "timing_linear", "timing_sigmoid", "timing_gaussian",
    "Layer", "LayerPipeline", "apply_layers",
    "get_variable_name", "compute_rot_angle", "compute_line_intersection", 

    "FACE_OVAL_IDX", "FACE_OVAL_TIGHT_IDX", "LEFT_EYE_IDX", "LEFT_IRIS_IDX", "RIGHT_EYE_IDX", "RIGHT_IRIS_IDX", 
    "NOSE_IDX", "MOUTH_IDX", "LIPS_IDX", "LEFT_CHEEK_IDX", "RIGHT_CHEEK_IDX", "CHIN_IDX", "HEMI_FACE_TOP_IDX",
    "HEMI_FACE_BOTTOM_IDX", "HEMI_FACE_LEFT_IDX", "HEMI_FACE_RIGHT_IDX", "LEFT_EYE_PATH", "LEFT_IRIS_PATH", 
    "RIGHT_EYE_PATH", "RIGHT_IRIS_PATH", "NOSE_PATH", "MOUTH_PATH", "LIPS_PATH", "FACE_OVAL_PATH", "FACE_OVAL_TIGHT_PATH",
    "HEMI_FACE_TOP_PATH", "HEMI_FACE_BOTTOM_PATH", "HEMI_FACE_LEFT_PATH", "HEMI_FACE_RIGHT_PATH", "CHEEKS_PATH", 
    "LEFT_CHEEK_PATH", "RIGHT_CHEEK_PATH", "CHEEKS_NOSE_PATH", "BOTH_EYES_PATH", "FACE_SKIN_PATH", "CHIN_PATH",

    "FileReadError", "FileWriteError", "UnrecognizedExtensionError", "FaceNotFoundError", "ImageShapeError", 

    "FACE_OVAL_MASK", "FACE_SKIN_MASK", "EYES_MASK", "IRISES_MASK", "LIPS_MASK", "HEMI_FACE_LEFT_MASK", "HEMI_FACE_RIGHT_MASK",
    "HEMI_FACE_BOTTOM_MASK", "HEMI_FACE_TOP_MASK", "EYES_NOSE_MOUTH_MASK", "MASK_OPTIONS", "COLOR_SPACE_BGR", "COLOR_SPACE_HSV",
    "COLOR_SPACE_GRAYSCALE", "COLOR_SPACE_OPTIONS", "COLOR_RED", "COLOR_BLUE", "COLOR_GREEN", "COLOR_YELLOW", "OCCLUSION_FILL_BLACK",
    "OCCLUSION_FILL_MEAN", "BLUR_METHOD_AVERAGE", "BLUR_METHOD_GAUSSIAN", "BLUR_METHOD_MEDIAN",
    "NOISE_METHOD_PIXELATE", "NOISE_METHOD_SALT_AND_PEPPER", "NOISE_METHOD_GAUSSIAN", "LOW_LEVEL_GRID_SHUFFLE", 
    "HIGH_LEVEL_GRID_SHUFFLE", "SPARSE_OPTICAL_FLOW", "DENSE_OPTICAL_FLOW", "SHOW_HISTORY_ORIGIN", 
    "SHOW_HISTORY_RELATIVE", "FRAME_SHUFFLE_RANDOM", "FRAME_SHUFFLE_RANDOM_W_REPLACEMENT", "FRAME_SHUFFLE_REVERSE", 
    "FRAME_SHUFFLE_RIGHT_CYCLIC_SHIFT", "FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT", "FRAME_SHUFFLE_PALINDROME", "FRAME_SHUFFLE_INTERLEAVE",
    "FRAME_SHUFFLE_METHODS", "EQUATE_IMAGES_CROP", "EQUATE_IMAGES_PAD",

    "display_convex_landmark_paths", "display_concave_landmark_paths", "display_all_landmark_paths",
    "display_face_mask_options", "display_color_space_options", "display_shift_color_options",
    "display_occlusion_fill_options", "display_blur_method_options", "display_noise_method_options",
    "display_scramble_method_options", "display_optical_flow_options", "display_history_mode_options",
    "display_shuffle_method_options", "display_timing_function_options",
]