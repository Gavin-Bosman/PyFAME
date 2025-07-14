import pytest
import os
from pyfame.file_access import create_output_directory

def test_create_output_directory(valid_input_dir, invalid_input_dir, sample_video_path):

    # Test the return value
    dir_path = create_output_directory(valid_input_dir, "temp")
    assert type(dir_path) == str
    assert os.path.exists(dir_path) == True

    # Test error handling for input params
    with pytest.raises(TypeError):
        create_output_directory(1, "temp")
    with pytest.raises(OSError):
        create_output_directory(invalid_input_dir, "temp")
    with pytest.raises(ValueError):
        create_output_directory(sample_video_path, "temp")

    with pytest.raises(TypeError):
        create_output_directory(valid_input_dir, 1)
    with pytest.raises(ValueError):
        create_output_directory(valid_input_dir, "")