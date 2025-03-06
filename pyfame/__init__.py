__version__ = "0.7.1"
__author__ = "Gavin Bosman"

from .pyfame import mask_face_region, occlude_face_region, blur_face_region, apply_noise, facial_scramble, point_light_display, get_optical_flow, extract_face_color_means, generate_shuffled_block_array, shuffle_frame_order, face_color_shift, face_saturation_shift, face_brightness_shift
from .setup_logging import setup_logging
from .utils import *
from .exceptions import *

setup_logging()

__all__ = [
    "mask_face_region", "occlude_face_region", "blur_face_region", "apply_noise", 
    "facial_scramble", "point_light_display", "get_optical_flow", "extract_face_color_means", 
    "generate_shuffled_block_array", "shuffle_frame_order", "face_color_shift", "face_saturation_shift", 
    "face_brightness_shift", "create_path", "calculate_rot_angle", "compute_line_intersection", 
    "get_min_max_bgr", "transcode_video_to_mp4", "constant", "sigmoid", "linear", "gaussian", 
    "FaceNotFoundError", "UnrecognizedExtensionError", "FileReadError", "FileWriteError"
]