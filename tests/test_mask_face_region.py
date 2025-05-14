import pytest
from pyfame.utils.exceptions import *
from pyfame.core.occlusion import mask_face_region

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input_dir parameter
    with pytest.raises(TypeError):
        mask_face_region(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        mask_face_region(input_dir=in_dir_invalid, output_dir=out_dir_valid)

    # exception testing output_dir parameter
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing mask_type parameter
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, mask_type="test")
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, mask_type=99)
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs="test")
    
    # exception testing background_color parameter
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, background_color=[125,125,125])
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, background_color=(125,155))
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, background_color=(125,155,12.0))

    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2)
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2)
    with pytest.raises(ValueError):
        mask_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)

    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        mask_face_region(input_dir="tests\\data\\no_face.png", output_dir=out_dir_valid)
    with pytest.raises(FaceNotFoundError):
        mask_face_region(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        mask_face_region(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)