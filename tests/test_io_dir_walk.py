import pytest
from pyfame.io import get_directory_walk

def test_get_directory_walk(valid_input_dir, invalid_input_dir, sample_video_path):

    sub_dirs, file_paths = get_directory_walk(valid_input_dir, True)
    assert isinstance(file_paths, list) == True
    assert isinstance(file_paths[0], str) == True
    assert isinstance(sub_dirs, list) == True
    assert isinstance(sub_dirs[0], str) == True

    with pytest.raises(TypeError):
        get_directory_walk(1, False)
    with pytest.raises(OSError):
        get_directory_walk(invalid_input_dir, False)
    with pytest.raises(ValueError):
        get_directory_walk(sample_video_path, False)
    
    with pytest.raises(TypeError):
        get_directory_walk(valid_input_dir, 1)