import pytest
from pyfame.file_access import map_directory_structure

def test_map_directory_structure(valid_input_dir, invalid_input_dir, valid_output_dir, invalid_output_dir, sample_video_path):

    with pytest.raises(TypeError):
        map_directory_structure(1, valid_output_dir, True)
    with pytest.raises(OSError):
        map_directory_structure(invalid_input_dir, valid_output_dir, True)
    with pytest.raises(ValueError):
        map_directory_structure(sample_video_path, valid_output_dir, True)
    
    with pytest.raises(TypeError):
        map_directory_structure(valid_input_dir, 1, True)
    with pytest.raises(OSError):
        map_directory_structure(valid_input_dir, invalid_output_dir, True)
    with pytest.raises(ValueError):
        map_directory_structure(valid_input_dir, sample_video_path, True)
    
    with pytest.raises(TypeError):
        map_directory_structure(valid_input_dir, valid_output_dir, 1)