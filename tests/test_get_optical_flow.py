import pytest
from pyfame.core.exceptions import *
from pyfame.core.analysis import get_optical_flow

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input and output directories
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        get_optical_flow(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing optical_flow_type parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, optical_flow_type=2.5)
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, optical_flow_type="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, optical_flow_type=100)
    
    # exception testing landmarks_to_track parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, landmarks_to_track={"a":2})
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, landmarks_to_track=[2.5,2,3])
    
    # exception testing max_corners parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, max_corners="test")
    
    # exception testing corner_quality_lvl parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, corner_quality_lvl="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, corner_quality_lvl=2.5)
    
    # exception testing min_corner_distance parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, min_corner_distance="test")
    
    # exception testing block_size parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, block_size="test")
    
    # exception testing win_size parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, win_size=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, win_size=(10,11.2))
    
    # exception testing max_pyr_lvl parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, max_pyr_lvl="test")
    
    # exception testing pyr_scale paramerer
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, pyr_scale="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, pyr_scale=2.5)
    
    # exception testing max_lk_iter parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, max_iter="test")
    
    # exception testing lk_accuracy_thresh parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, lk_accuracy_thresh="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, lk_accuracy_thresh=2.5)
    
    # exception testing poly_sigma parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, poly_sigma="test")
    
    # exeption testing point_color parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, point_color=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, point_color=(10,11.2))

    # exception testing point_radius parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, point_radius="test")
    
    # exception testing vector_color parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, vector_color=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, vector_color=(10,11.2))
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs="test")
    
    # exception testing mediapipe config parameters
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.5)
    
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.5)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        get_optical_flow(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)

    # testing exceptions for unrecognized file extensions
    with pytest.raises(UnrecognizedExtensionError):
        get_optical_flow(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)