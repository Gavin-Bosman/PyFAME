import pytest
from pyfame.util.util_exceptions import *
from pyfame.analysis import get_facial_color_means

def test_exception_handling(valid_input_dir, valid_output_dir, invalid_input_dir, invalid_output_dir):

    # exception testing input and output directories
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=1, output_dir=valid_output_dir)
    with pytest.raises(OSError):
        get_facial_color_means(input_dir=invalid_input_dir, output_dir=valid_output_dir)
    
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=1)
    with pytest.raises(OSError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=invalid_output_dir)
    with pytest.raises(ValueError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_input_dir)
    
    # exception testing color_space parameter
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, color_space=2.5)
    with pytest.raises(ValueError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, color_space=100)
    with pytest.raises(ValueError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, color_space="test")
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, with_sub_dirs="test")
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, min_detection_confidence="test")
    with pytest.raises(ValueError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        get_facial_color_means(input_dir=valid_input_dir, output_dir=valid_output_dir, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        get_facial_color_means(input_dir="tests\\data\\no_face.mp4", output_dir=valid_output_dir)
    with pytest.raises(FaceNotFoundError):
        get_facial_color_means(input_dir="tests\\data\\no_face.png", output_dir=valid_output_dir)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        get_facial_color_means(input_dir="tests\\data\\invalid_ext.webp", output_dir=valid_output_dir)