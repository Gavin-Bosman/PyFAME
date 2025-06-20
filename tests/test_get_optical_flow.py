import pytest
from pyfame.util.util_exceptions import *
from pyfame.analysis import get_optical_flow

def test_exception_handling(valid_input_dir, valid_output_dir, invalid_input_dir, invalid_output_dir):

    # exception testing input and output directories
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=1, output_dir=valid_output_dir)
    with pytest.raises(OSError):
        get_optical_flow(input_dir=invalid_input_dir, output_dir=valid_output_dir)
    
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=1)
    with pytest.raises(OSError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=invalid_output_dir)
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_input_dir)
    
    # exception testing optical_flow_type parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, optical_flow_type=2.5)
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, optical_flow_type="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, optical_flow_type=100)
    
    # exception testing landmarks_to_track parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, landmarks_to_track={"a":2})
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, landmarks_to_track=[2.5,2,3])
    
    # exception testing max_corners parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, max_corners="test")
    
    # exception testing corner_quality_lvl parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, corner_quality_lvl="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, corner_quality_lvl=2.5)
    
    # exception testing min_corner_distance parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, min_corner_distance="test")
    
    # exception testing block_size parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, block_size="test")
    
    # exception testing win_size parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, win_size=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, win_size=(10,11.2))
    
    # exception testing max_pyr_lvl parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, max_pyr_lvl="test")
    
    # exception testing pyr_scale paramerer
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, pyr_scale="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, pyr_scale=2.5)
    
    # exception testing max_lk_iter parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, max_iter="test")
    
    # exception testing lk_accuracy_thresh parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, lk_accuracy_thresh="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, lk_accuracy_thresh=2.5)
    
    # exception testing poly_sigma parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, poly_sigma="test")
    
    # exeption testing point_color parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, point_color=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, point_color=(10,11.2))

    # exception testing point_radius parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, point_radius="test")
    
    # exception testing vector_color parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, vector_color=[1,2])
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, vector_color=(10,11.2))
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, with_sub_dirs="test")
    
    # exception testing mediapipe config parameters
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, min_detection_confidence="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, min_detection_confidence=2.5)
    
    with pytest.raises(TypeError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        get_optical_flow(input_dir=valid_input_dir, output_dir=valid_output_dir, min_tracking_confidence=2.5)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        get_optical_flow(input_dir="tests\\data\\no_face.mp4", output_dir=valid_output_dir)

    # testing exceptions for unrecognized file extensions
    with pytest.raises(UnrecognizedExtensionError):
        get_optical_flow(input_dir="tests\\data\\invalid_ext.webp", output_dir=valid_output_dir)