import pytest
from pyfame.utils.exceptions import *
from pyfame.core.analysis import extract_face_color_means

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input and output directories
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        extract_face_color_means(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing color_space parameter
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, color_space=2.5)
    with pytest.raises(ValueError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, color_space=100)
    with pytest.raises(ValueError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, color_space="test")
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs="test")
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence="test")
    with pytest.raises(ValueError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        extract_face_color_means(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        extract_face_color_means(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)
    with pytest.raises(FaceNotFoundError):
        extract_face_color_means(input_dir="tests\\data\\no_face.png", output_dir=out_dir_valid)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        extract_face_color_means(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)