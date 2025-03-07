import pytest
from pyfame.core.exceptions import *
from pyfame.core.occlusion import blur_face_region

def test_exception_handling():
    in_dir_valid = "tests\\data\\01-02-01-01-01-01-01.mp4"
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
    
    # exception testing blur_method
    with pytest.raises(TypeError):
        blur_face_region(input_dir=in_dir_valid, output_dir=out_dir_valid, blur_method=2.5)
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid, blur_method="null")
    with pytest.raises(ValueError):
        blur_face_region(input_dir=in_dir_valid, output_dir=in_dir_valid, blur_method=100)
    