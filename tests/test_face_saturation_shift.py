import pytest
from pyfame.util.util_exceptions import *
from pyfame.manipulation.color.apply_color_brightness_shift import face_saturation_shift

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input_dir and output_dir
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        face_saturation_shift(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing onset_t parameter
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, onset_t="test")
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, onset_t=-2.0)
    
    # exception testing offset_t parameter
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, offset_t="test")
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, offset_t=-2.0)
    
    # exception testing shift_magnitude parameter
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, shift_magnitude="test")
    
    # exception testing landmark_regions parameter
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions={"test":2})
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[])
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[1,2,3])
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[[1,2,3]])
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs=1)
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2)
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2)
    with pytest.raises(ValueError):
        face_saturation_shift(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        face_saturation_shift(input_dir="tests\\data\\no_face.png", output_dir=out_dir_valid)
    with pytest.raises(FaceNotFoundError):
        face_saturation_shift(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        face_saturation_shift(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)