import pytest
from pyfame.utils.exceptions import *
from pyfame.core.point_light_display import generate_point_light_display

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input and output directories
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        generate_point_light_display(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing landmark_regions parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions={"test":2})
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[])
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[1,2,3])
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, landmark_regions=[[1,2,3]])
    
    # exception testing point_density parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, point_density="test")
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, point_density=1.5)
    
    # exception testing show_history parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, show_history="test")
    
    # exception testing history_mode parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_mode="test")
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_mode=100)
    
    # exception testing history_window_msec parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_window_msec="test")
    
    # exception testing point_color parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, point_color=[1,2,3])
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, point_color=(1,2))
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, point_color=('a','b','c'))

    # exception testing history_color parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_color=[1,2,3])
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_color=(1,2))
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, history_color=('a','b','c'))
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs="test")
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence="test")
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        generate_point_light_display(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        generate_point_light_display(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)

    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        generate_point_light_display(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)
