import pytest
from pyfame.util.util_exceptions import *
from pyfame.manipulation.occlusion.layer_occlusion import blur_face_region

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input_dir parameter
    with pytest.raises(TypeError):
        blur_face_region(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        blur_face_region(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    # exception testing output_dir parameter
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing blur_method parameter
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, blur_method=2.5)
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid, blur_method="null")
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid, blur_method=100)
    
    # exception testing k_size parameter
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, k_size="test")
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, k_size=0)
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs=1)
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2)
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2)
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        blur_face_region(input_dir="tests\\data\\no_face.png", output_dir=out_dir_valid)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        blur_face_region(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)