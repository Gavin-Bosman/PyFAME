import pytest
from pyfame.utilities.exceptions import *
from pyfame.analyse import analyse_facial_colour_means

def test_exception_handling(valid_input_dir, valid_output_dir, invalid_input_dir, invalid_output_dir):

    # exception testing input and output directories
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=1, output_directory=valid_output_dir)
    with pytest.raises(OSError):
        analyse_facial_colour_means(input_directory=invalid_input_dir, output_directory=valid_output_dir)
    
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=1)
    with pytest.raises(OSError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=invalid_output_dir)
    with pytest.raises(ValueError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_input_dir)
    
    # exception testing color_space parameter
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, colour_space=2.5)
    with pytest.raises(ValueError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, colour_space=100)
    with pytest.raises(ValueError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, colour_space="test")
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, with_sub_dirs="test")
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, min_detection_confidence="test")
    with pytest.raises(ValueError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, min_tracking_confidence="test")
    with pytest.raises(ValueError):
        analyse_facial_colour_means(input_directory=valid_input_dir, output_directory=valid_output_dir, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        analyse_facial_colour_means(input_directory="tests\\data\\no_face.mp4", output_directory=valid_output_dir)
    with pytest.raises(FaceNotFoundError):
        analyse_facial_colour_means(input_directory="tests\\data\\no_face.png", output_directory=valid_output_dir)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        analyse_facial_colour_means(input_directory="tests\\data\\invalid_ext.webp", output_directory=valid_output_dir)